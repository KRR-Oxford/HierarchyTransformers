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

from typing import Union
from gensim.models.poincare import PoincareKeyedVectors
import numpy as np
import torch
from ..graph import HypernymGraph


class ReconstructionEvaluator:
    def __init__(self, graph: HypernymGraph, embeddings: Union[torch.Tensor, PoincareKeyedVectors]):
        self.graph = graph

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
            self.embedding_dict = PoincareKeyedVectors(
                vector_size=embeddings.shape[1], vector_count=embeddings.shape[0]
            )
            # assuming the embeddings are stored in the same order as the graph entities
            for ent in self.graph.entities:
                self.embedding_dict[ent] = embeddings[self.graph.ent2idx[ent]]
        elif isinstance(embeddings, PoincareKeyedVectors):
            self.embedding_dict = embeddings
        else:
            raise ValueError(f"Unknown input embeddings type: {type(embeddings)}.")

    def evaluate_hypernym_mean_rank_and_AP(self):
        """Evaluate the hypernym Mean Rank and Mean Average Precision scores for all entities."""
        all_ranks = []
        all_aps = []
        for ent in self.graph.entities:
            results = self.get_hypernym_average_precision(ent, return_ranks=True)
            if results:
                ap, ranks = results
                all_ranks += ranks
                all_aps.append(ap)
        return {"mean_rank": np.mean(all_ranks), "MAP": np.mean(all_aps)}

    def get_hypernym_ranks(self, entity_name: str):
        """Get the rank(s) of the correct hypernyms for the given entity."""
        hypernyms = self.graph.get_hypernyms(entity_name)
        rank_dict = dict()

        if not hypernyms:
            return rank_dict
        distances_to_hypernyms = self.embedding_dict.distances(entity_name, other_nodes=hypernyms)
        rest = set(self.graph.entities) - hypernyms
        distances_to_rest = self.embedding_dict.distances(entity_name, other_nodes=rest)

        # compute how many negative hypernyms with distances less than each positive one
        for h, dh in zip(hypernyms, distances_to_hypernyms):
            rank_dict[h] = (np.array(distances_to_rest) < dh).sum() + 1
        return rank_dict

    def get_hypernym_average_precision(self, entity_name: str, return_ranks: bool = True):
        """Get the average precision (AP) of the correct hypernyms for the given entity.

        Optional to return the individual hypernym ranks before calculating AP.
        """
        rank_dict = self.get_hypernym_ranks(entity_name)
        if not rank_dict:
            return None
        ranks = np.array(list(rank_dict.values()))
        map_ranks = np.sort(ranks) + np.arange(
            len(ranks)
        )  # consider all hypernyms as wanted documents from a sequence of queries
        avg_precision = (np.arange(1, len(map_ranks) + 1) / np.sort(map_ranks)).mean()
        if return_ranks:
            return avg_precision, list(ranks)
        else:
            return avg_precision
