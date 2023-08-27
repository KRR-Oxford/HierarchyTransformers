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

from __future__ import annotations

from typing import Optional
import pandas as pd
import networkx as nx
import numpy as np
import torch


class HypernymGraph:
    r"""Class for building a graph with directed edges representing hyponym-hypernym relationships.

    The input data file should be a `.tsv` file containing the `SubEntity` and `SuperEntity` columns.
    """

    def __init__(self, data_file: str):
        if not data_file.endswith(".tsv"):
            raise ValueError("Input data file should be a .tsv file.")
        self.edges = pd.read_csv(data_file, delimiter="\t")
        self.edges = [(child, parent) for child, parent in self.edges.values.tolist()]  # transform to tuples
        self.graph = nx.DiGraph(self.edges)  # directed graph
        self.entities = list(self.graph.nodes)
        self.ent2idx = {self.entities[i]: i for i in range(len(self.entities))}
        self.idx2ent = {v: k for k, v in self.ent2idx.items()}
        self.idx_graph = nx.Graph([(self.ent2idx[c], self.ent2idx[p]) for c, p in self.edges])

        self._entity_buffer_weighted = False
        self._entity_buffer = []  # buffer for hypernym sampling
        self._default_buffer_size = 2000

        print(self)

    def __str__(self):
        return f"Hypernym Graph containing {len(self.entities)} nodes and {len(self.edges)} edges."

    def __len__(self):
        return len(self.entities)

    def get_hypernyms(self, entity_name: str):
        """Get a set of super-entities (hypernyms) for a given entity."""
        return set(self.graph.successors(entity_name))

    def _fill_entity_buffer(self, weighted: bool, buffer_size: Optional[int] = None):
        """Buffer a large collection of entities sampled with replacement for faster negative sampling."""
        buffer_size = buffer_size if buffer_size else self._default_buffer_size
        self._entity_buffer_weighted = weighted
        if weighted:
            weights = np.array([self.graph.degree(ent) for ent in self.entities])
            probs = weights / weights.sum()
            self._entity_buffer = np.random.choice(self.entities, size=buffer_size, p=probs)
        else:
            self._entity_buffer = np.random.choice(self.entities, size=buffer_size)
        # print(f"Recharging entity buffer: num={len(self.entity_buffer)}, weighted={weighted}")

    def sample_negative_hypernyms(
        self, entity_name: str, n_samples: int, weighted: bool = False, buffer_size: Optional[int] = None
    ):
        """Sample negative hypernyms (i.e., not linked by a directed edge) with replacement for a given entity
        from buffered entities.
        """
        negative_hypernyms = []
        hypernyms = self.get_hypernyms(entity_name)
        # refill the buffer if not enough or changing type (weighted or not)
        while len(negative_hypernyms) < n_samples:
            if len(self._entity_buffer) < n_samples or self._entity_buffer_weighted != weighted:
                self._fill_entity_buffer(weighted=weighted, buffer_size=buffer_size)
            negative_hypernyms += list(filter(lambda x: x not in hypernyms, self._entity_buffer[:n_samples]))
            self._entity_buffer = self._entity_buffer[n_samples:]  # remove the samples from the buffer
        # print(len(negative_hypernyms))
        # print(self.ent2idx[entity_name])
        return negative_hypernyms[:n_samples]


class HypernymDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        hypernym_graph: HypernymGraph,
        n_negative_samples: int = 10,
        weighted_negative_sampling: bool = False,
        negative_buffer_size: int = 1000000,
    ):
        self.graph = hypernym_graph
        self.n_negative_samples = n_negative_samples
        self.weighted_negative_sampling = weighted_negative_sampling
        self.negative_buffer_size = negative_buffer_size

    def __len__(self):
        return len(self.graph.edges)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        edge = self.graph.edges[idx]
        # NOTE: Renew negative samples for every `__getitem__` call.
        negatives = self.graph.sample_negative_hypernyms(
            entity_name=edge[0],
            n_samples=self.n_negative_samples,
            weighted=self.weighted_negative_sampling,
            buffer_size=self.negative_buffer_size,
        )
        sample = [self.graph.ent2idx[ent] for ent in [edge[0], edge[1]] + negatives]
        return torch.tensor(sample)
