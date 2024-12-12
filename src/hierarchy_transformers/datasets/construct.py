# Copyright 2023 Yuan He

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import json
from pathlib import Path

from deeponto.onto import Taxonomy, TaxonomyNegativeSampler
from deeponto.utils import save_file
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


class HierarchyDatasetConstructor:
    """
    Class for contructing hierarchy understanding datasets.
    """
    def __init__(self, hierarchy: Taxonomy):
        self.hierarchy = hierarchy
        self.neg_sampler = TaxonomyNegativeSampler(self.hierarchy)

    def get_hard_negative(self, entity_id: str):
        """
        Get a hard negative subsumer (sibling) for the input entity.
        """
        parents = self.hierarchy.get_parents(entity_id)
        ancestors = self.hierarchy.get_parents(entity_id, True)
        siblings = []
        for parent in parents:
            siblings += self.hierarchy.get_children(parent)
        hard_negatives = set(siblings) - set([entity_id]) - set(ancestors)
        return list(hard_negatives)

    def get_transitive_edges(self, edges: list):
        """
        Get all indirect subsumptions by transitive reasoning.
        """
        trans_edges = []
        for child, _ in edges:
            trans_edges += [(child, parent) for parent in self.hierarchy.get_parents(child, True)]
        return list(set(trans_edges) - set(edges))

    def save_entity_lexicon(self, output_dir: str):
        """
        Save the entity lexicon.
        """
        entity_lexicon = dict()
        for n in self.hierarchy.nodes:
            entity_lexicon[n] = self.hierarchy.get_node_attributes(n)
        save_file(entity_lexicon, f"{output_dir}/entity_lexicon.json")

    def save_dataset(self, dataset: list, output_file: str):
        """
        Save the constructed dataset.
        """
        with open(f"{output_file}", "w+") as f:
            f.writelines("\n".join([json.dumps(sample) for sample in dataset]))

    def construct_example(self, child: str, parent: str, num_negative: int = 10):
        """
        Construct negative examples given a positive `(child, parent)` pair.
        """
        example = {"child": child, "parent": parent}
        example["random_negatives"] = self.neg_sampler.sample(child, num_negative)
        example["hard_negatives"] = (self.get_hard_negative(child) + example["random_negatives"])[:num_negative]
        return example

    def construct(self, output_dir: str, num_negative: int = 10, eval_size=0.1):
        """
        Construct the multi-hop and mixed-hop datasets.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        base_edges = [(child, parent) for parent, child in self.hierarchy.edges]
        trans_edges = self.get_transitive_edges(base_edges)
        assert not set(trans_edges).intersection(set(base_edges))

        base_examples = []
        for child, parent in tqdm(base_edges, desc="base"):
            base_examples.append(self.construct_example(child, parent, num_negative))

        trans_examples = []
        for child, parent in tqdm(trans_edges, desc="trans"):
            trans_examples.append(self.construct_example(child, parent, num_negative))

        _, trans_eval_examples = train_test_split(trans_examples, test_size=eval_size)
        trans_val_examples, trans_test_examples = train_test_split(trans_eval_examples, test_size=0.5)

        trans_task_name = "multi"
        Path(f"{output_dir}/{trans_task_name}").mkdir(parents=True, exist_ok=True)
        self.save_dataset(base_examples, f"{output_dir}/{trans_task_name}/train.jsonl")
        # self.save_dataset(trans_train_examples, f"{output_dir}/transitivity/trans_train.jsonl")
        self.save_dataset(trans_val_examples, f"{output_dir}/{trans_task_name}/val.jsonl")
        self.save_dataset(trans_test_examples, f"{output_dir}/{trans_task_name}/test.jsonl")
        self.save_entity_lexicon(f"{output_dir}/{trans_task_name}")

        base_train_examples, base_eval_examples = train_test_split(base_examples, test_size=eval_size)
        base_val_examples, base_test_examples = train_test_split(base_eval_examples, test_size=0.5)
        # base_train_edges = [(x["child"], x["parent"]) for x in base_train_examples]
        # trans_base_train_edges = self.get_transitive_edges(base_train_edges)
        # trans_base_train_examples = []
        # for child, parent in tqdm(trans_base_train_edges, desc="trans on base_train"):
        #     trans_base_train_examples.append(self.construct_example(child, parent, num_negative))
        pred_task_name = "mixed"
        Path(f"{output_dir}/{pred_task_name}").mkdir(parents=True, exist_ok=True)
        self.save_dataset(base_train_examples, f"{output_dir}/{pred_task_name}/train.jsonl")
        # self.save_dataset(trans_base_train_examples, f"{output_dir}/induc/trans_base_train.jsonl")
        self.save_dataset(base_val_examples + trans_val_examples, f"{output_dir}/{pred_task_name}/val.jsonl")
        self.save_dataset(base_test_examples + trans_test_examples, f"{output_dir}/{pred_task_name}/test.jsonl")
        self.save_entity_lexicon(f"{output_dir}/{pred_task_name}")
