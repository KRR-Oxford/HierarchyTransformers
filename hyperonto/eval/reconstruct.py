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

from typing import Union, Optional
from gensim.models.poincare import PoincareKeyedVectors
import numpy as np
from tqdm import tqdm
from ..poincare import PoincareBallModel
from ..graph import HypernymGraph


class ReconstructionEvaluator:
    r"""Class for evaluating the quality of Poincare embeddings through several intrinsic reconstruction settings.

    Reconstruction settings proposed by [Nickel et al. 2017](https://arxiv.org/abs/1705.08039)
        1. Hypernym Query (Mean Ranks, mAP of the correct hypernyms); 
        NOTE: these scores do not really reflect the quality because child nodes can be closer than the parent nodes

    Reconstruction settings proposed by us:
        1. Centripetal Path: the hyerpernym path is towards the center of the Poincare ball (indicated by monotonically decreasing norms)

    """

    def __init__(self, graph: HypernymGraph, embeddings: Union[PoincareBallModel, PoincareKeyedVectors]):
        self.graph = graph

        if isinstance(embeddings, PoincareBallModel):
            torch_model = embeddings
            embeddings = torch_model.state_dict()["embed.weight"].detach().cpu().numpy()
            self.embedding_dict = PoincareKeyedVectors(vector_size=embeddings.shape[1], vector_count=0)
            keys = []
            for i in range(len(torch_model.idx2ent)):
                keys.append(torch_model.idx2ent[i])
            self.embedding_dict.add_vectors(keys, embeddings)
            # for ent in tqdm(self.graph.entities, desc="Transform torch embeddings into dict", unit="entity"):
            #     self.embedding_dict[ent] = embeddings[self.graph.ent2idx[ent]]
        elif isinstance(embeddings, PoincareKeyedVectors):
            self.embedding_dict = embeddings
        else:
            raise ValueError(f"Unknown input embeddings type: {type(embeddings)}.")

    def evaluate_hypernym_mean_rank_and_AP(self, max_eval_nums: Optional[int] = None):
        """Evaluate the hypernym Mean Rank and Mean Average Precision scores for all entities."""
        all_ranks = []
        all_aps = []
        # set smaller evaluation values for debugging very large graphs
        eval_entities = (
            self.graph.entities
            if not max_eval_nums
            else list(np.random.choice(self.graph.entities, replace=False, size=max_eval_nums))
        )
        for ent in tqdm(eval_entities, desc="Reconstruction evaluation (retrieval-based)", unit="entity"):
            results = self.get_hypernym_average_precision(ent, return_ranks=True)
            if results:
                ap, ranks = results
                all_ranks += ranks
                all_aps.append(ap)
        return {"mean_rank": np.mean(all_ranks), "MAP": np.mean(all_aps)}

    def get_hypernym_ranks(self, entity_name: str):
        """Get the rank(s) of the correct hypernyms for the given entity 
        based on Poincare distances (shorter distance => higher rank). 
        """
        hypernyms = self.graph.get_hypernyms(entity_name)
        hypernym_idxs = [self.embedding_dict.get_index(h) for h in hypernyms]
        rank_dict = dict()

        if not hypernyms:
            return rank_dict

        # NOTE: compute distances only once saving more than half of the time
        all_distances = self.embedding_dict.distances(entity_name)
        positive_relation_distances = all_distances[hypernym_idxs]
        negative_relation_distances = np.ma.array(all_distances, mask=False)
        negative_relation_distances.mask[hypernym_idxs] = True
        ranks = (negative_relation_distances[None, :] < positive_relation_distances[:, None]).sum(axis=1) + 1
        for h, r in zip(hypernyms, ranks):
            rank_dict[h] = r
        return rank_dict

    def get_hypernym_average_precision(self, entity_name: str, return_ranks: bool = True):
        """Get the average precision (AP) of the correct hypernyms for the given entity 
        based on Poincare distances (shorter distance => higher rank).

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
