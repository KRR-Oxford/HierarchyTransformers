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

import torch

torch.set_default_dtype(torch.float64)
from geoopt.manifolds import PoincareBall
from geoopt.tensor import ManifoldParameter
from .graph import SubsumptionGraph


class PoincareOntologyEmbedding(torch.nn.Module):
    """Class for the Poincare embedding model using hyperbolic distances as loss heuristics."""

    def __init__(
        self,
        graph: SubsumptionGraph,
        embed_dim: int,  # Poincare ball dimension
        static_embed: bool,  # initialise static embedding weights or not
    ):
        super().__init__()

        # do not save the graph directly as pickling is expensive
        self.idx2ent = graph.idx2ent
        self.ent2idx = graph.ent2idx
        self.embed_dim = embed_dim
        self.static_embed = static_embed

        self.manifold = PoincareBall()
        # d(u, v) = arcosh(1 + 2 \frac{\|u - v \|^2}{(1 - \| u \|^2)(1 - \| v \|^2)}) or the one defined with mobius addition
        self.dist = self.manifold.dist

        if self.static_embed:
            self.embed = self.init_static_graph_embedding(len(graph), self.embed_dim, 1e-3)

    def init_static_graph_embedding(self, static_entity_size: int, embed_dim: int, init_weights: float):
        # init embedding weights to somewhere near the origin
        static_embedding = torch.nn.Embedding(static_entity_size, embed_dim, sparse=False, max_norm=1.0)
        static_embedding.weight.data.uniform_(-init_weights, init_weights)
        static_embedding.weight = ManifoldParameter(static_embedding.weight, manifold=self.manifold)
        return static_embedding

    def unpack_embeddings(self, inputs: torch.Tensor):
        """Split input tensor into subject and objects

        NOTE: the first object is the related one and the rest are negative samples.
        """
        input_embeds = self.embed(
            inputs
        )  # (batch_size, num_entities, hidden_dim), dim 1 includes (child, parent, negative_parents*)
        objects = input_embeds.narrow(dim=1, start=1, length=input_embeds.size(1) - 1)  # use .narrow to keep dim
        subject = input_embeds.narrow(dim=1, start=0, length=1).expand_as(objects)
        return subject, objects

    def forward(self, inputs: torch.Tensor):
        subject, objects = self.unpack_embeddings(inputs)
        return self.dist(subject, objects)
