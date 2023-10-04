from __future__ import annotations

from yacs.config import CfgNode
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from deeponto.onto import Taxonomy, TaxonomyNegativeSampler


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
