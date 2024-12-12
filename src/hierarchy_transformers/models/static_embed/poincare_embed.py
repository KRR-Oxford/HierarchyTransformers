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

import logging

import torch
from geoopt import ManifoldParameter
from geoopt.manifolds import PoincareBall

logger = logging.getLogger(__name__)


class PoincareStaticEmbedding(torch.nn.Module):
    r"""Class for the static hyperbolic embedding models:

        - [1] Poincaré Embedding by [Nickel et al., NeurIPS 2017](https://arxiv.org/abs/1705.08039).
        - [2] Hyperbolic Entailment Cone by [Ganea et al., ICML 2018](https://arxiv.org/abs/1804.01882).

    both of which lie in a unit Poincaré ball. According to [2], it is better to apply the entailment cone loss in the post-training phase of a Poincaré embedding model in [1].

    Attributes:
        entities (list): The list of input entity IDs (fixed).
        idx2ent (dict): A dictionary that stores the `(index, entity_id)` pairs.
        ent2idx (dict): A dictionary that stores the `(entity_id, index)` pairs.
        embed_dim (int): The embedding dimension of this model.
        manifold (geoopt.manifolds.PoincareBall): The hyperbolic manifold (Poincaré Ball) of this model.
        embed (torch.nn.Embedding): The static hyperbolic embeddings for entities.
    """

    def __init__(self, entity_ids: list, embed_dim: int, init_weights: float = 1e-3):
        super().__init__()

        self.entities = entity_ids
        self.idx2ent = {idx: ent for idx, ent in enumerate(self.entities)}
        self.ent2idx = {v: k for k, v in self.idx2ent.items()}
        self.embed_dim = embed_dim
        self.manifold = PoincareBall()
        self.dist = self.manifold.dist

        # initialise static embedding
        self.embed = torch.nn.Embedding(
            num_embeddings=len(self.idx2ent),  # fixed num embeddings,
            embedding_dim=self.embed_dim,
            sparse=False,
            max_norm=1.0,  # unit poincare ball projection
        )
        self.embed.weight.data.uniform_(-init_weights, init_weights)
        self.embed.weight = ManifoldParameter(self.embed.weight, manifold=self.manifold)
        logger.info(f"Init static hyperbolic embedding for {len(self.idx2ent)} entities.")

    def forward(self, inputs: torch.Tensor):
        """Forward propagation.

        The inputs are organised as `(batch_size, num_entities, embed_dim)` where `dim=` includes `(child, parent, negative_parents*)`.
        """

        input_embeds = self.embed(
            inputs
        )  # (batch_size, num_entities, hidden_dim), dim 1 includes (child, parent, negative_parents*)
        objects = input_embeds.narrow(dim=1, start=1, length=input_embeds.size(1) - 1)  # use .narrow to keep dim
        subject = input_embeds.narrow(dim=1, start=0, length=1).expand_as(objects)
        return subject, objects
