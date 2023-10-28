from __future__ import annotations

from typing import Optional
from yacs.config import CfgNode
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from deeponto.onto import Taxonomy, TaxonomyNegativeSampler
from deeponto.align.mapping import ReferenceMapping
import logging

logger = logging.getLogger(__name__)


class TaxonomyTrainingDataset(torch.utils.data.Dataset):
    """The training dataset enables dynamic negative sampling."""

    def __init__(
        self,
        taxonomy: Taxonomy,
        training_subsumptions: list,  # list of subsumption pairs for training
        n_negative_samples: int = 10,
        node_attribute: Optional[str] = None,
    ):
        self.taxonomy = taxonomy
        self.subsumptions = training_subsumptions
        self.negative_sampler = TaxonomyNegativeSampler(taxonomy)
        self.n_negative_samples = n_negative_samples
        self.node_attribute = node_attribute
        self.get_name = lambda x: self.taxonomy.get_node_attributes(x)[self.node_attribute]

    def __len__(self):
        return len(self.subsumptions)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if not isinstance(idx, slice):
            subj, obj = self.subsumptions[idx]
            # NOTE: Renew negative samples for every `__getitem__` call.
            negative_objs = self.negative_sampler.sample(subj, self.n_negative_samples)
            if self.node_attribute:
                subj = self.get_name(subj)
                obj = self.get_name(obj)
                negative_objs = [self.get_name(n) for n in negative_objs]
            return CfgNode({"subject": subj, "object": obj, "negative_objects": negative_objs})
        else:
            samples = []
            for subj, obj in self.subsumptions[idx]:
                negative_objs = self.negative_sampler.sample(subj, self.n_negative_samples)
                if self.node_attribute:
                    subj = self.get_name(subj)
                    obj = self.get_name(obj)
                    negative_objs = [self.get_name(n) for n in negative_objs]
                sample = CfgNode({"subject": subj, "object": obj, "negative_objects": negative_objs})
                samples.append(sample)
            return samples


def read_taxonomy_data(data_file: str):
    logger.info(f"Loading taxonomy data from: {data_file}")
    return [x.to_tuple() for x in ReferenceMapping.read_table_mappings(data_file)]


def transitivity_data_splitting(taxonomy: Taxonomy):
    # base edges must be included in the training data for the transitivity setting
    base_edges = taxonomy.edges
    all_edges = []
    for node in taxonomy.nodes:
        all_edges += list(map(lambda p: (node, p), taxonomy.get_parents(node, apply_transitivity=True)))
    transitive_edges = list(set(all_edges) - set(base_edges))
    train_edges, eval_edges = train_test_split(transitive_edges, test_size=0.2)
    val_edges, test_edges = train_test_split(eval_edges, test_size=0.5)
    return CfgNode({"base": base_edges, "train": train_edges, "val": val_edges, "test": test_edges})


def subsumption_data_splitting(taxonomy: Taxonomy):
    # the subsumption setting sets a portion of edges as "missing"
    neg_sampler = TaxonomyNegativeSampler(taxonomy)
    base_edges = taxonomy.edges
    train_edges, eval_edges = train_test_split(base_edges, test_size=0.2)
    val_edges, test_edges = train_test_split(eval_edges, test_size=0.5)

    train_df = pd.DataFrame(train_edges, columns=["SrcEntity", "TgtEntity"])

    val_df = []
    for s, t in tqdm(val_edges, desc="Validation Set Negative Sampling"):
        t_cands = []
        while len(t_cands) < 100:
            t_cands += neg_sampler.sample(s, 200)
            t_cands = list(set(t_cands))
        t_cands = [t] + t_cands[:99]  # 1 correct candidate and 99 negative candidates
        np.random.shuffle(t_cands)
        val_df.append((s, t, t_cands))
    val_df = pd.DataFrame(val_df, columns=["SrcEntity", "TgtEntity", "TgtCandidates"])

    test_df = []
    for s, t in tqdm(test_edges, desc="Test Set Negative Sampling"):
        t_cands = []
        while len(t_cands) < 100:
            t_cands += neg_sampler.sample(s, 200)
            t_cands = list(set(t_cands))
        t_cands = [t] + t_cands[:99]  # 1 correct candidate and 99 negative candidates
        np.random.shuffle(t_cands)
        test_df.append((s, t, t_cands))
    test_df = pd.DataFrame(test_df, columns=["SrcEntity", "TgtEntity", "TgtCandidates"])

    return train_df, val_df, test_df
