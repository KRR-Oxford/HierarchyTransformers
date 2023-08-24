# Copyright 2023 Yuan He. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import networkx as nx
import random


class HypernymGraph:
    r"""Class for building a graph with directed edges representing hyponym-hypernym relationships.
    
    The input data file should be a `.tsv` file containing the `SubEntity` and `SuperEntity` columns.
    """
    
    def __init__(self, data_file: str):
        if not data_file.endswith('.tsv'):
            raise ValueError("Input data file should be a .tsv file.")
        self.edges = pd.read_csv(data_file, delimiter="\t")
        self.edges = [(child, parent) for child, parent in self.edges.values.tolist()]  # transform to tuples
        self.graph = nx.DiGraph(self.edges)  # directed graph
        self.entities = list(self.graph.nodes)
        self.ent2idx = {self.entities[i]: i for i in range(len(self.entities))}
        self.idx2ent = {v: k for k, v in self.ent2idx.items()}
        self.idx_graph = nx.Graph([(self.ent2idx[c], self.ent2idx[p]) for c, p in self.edges])
        print(self)
        
    def __str__(self):
        return f"Subsumption Graph containing {len(self.entities)} nodes and {len(self.edges)} edges."

    def get_hypernyms(self, entity_name: str):
        """Get a set of super-entities (hypernyms) for a given entity.
        """
        return set(self.graph.successors(entity_name))
    
    def sample_negative_hypernyms(self, entity_name: str, n_samples: int, weighted: bool = True):
        """Sample negative hypernyms (i.e., not linked by a directed edge) with replacement for a given entity.
        """
        negative_hypernym_pool = list(set(self.entities) - self.get_hypernyms(entity_name))
        if weighted:
            weights = [self.graph.degree(n) for n in negative_hypernym_pool]
            return random.choices(negative_hypernym_pool, weights=weights, k=n_samples)
        else:
            return random.choices(negative_hypernym_pool, k=n_samples)
        
    