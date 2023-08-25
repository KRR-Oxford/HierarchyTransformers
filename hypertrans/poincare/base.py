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
from geoopt.manifolds import PoincareBall
from geoopt.tensor import ManifoldParameter


class PoincareBase(torch.nn.Module):
    """Class for the Poincare embedding model using hyperbolic distances as loss heuristics.
    """
    def __init__(self, vocab_size: int, embed_dim: int, init_weights: float = 1e-3):
        super().__init__()

        self.manifold = PoincareBall()
        # init embedding weights to somewhere near the origin
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, sparse=False, max_norm=1.0)
        self.embedding.weight.data.uniform_(-init_weights, init_weights)
        self.embedding.weight = ManifoldParameter(self.embedding.weight, manifold=self.manifold)
        self.dist = self.manifold.dist  # d(u, v) = arcosh(1 + 2 \frac{\|u - v \|^2}{(1 - \| u \|^2)(1 - \| v \|^2)}) or the one defined with mobius addition

    def unpack_embeddings(self, inputs: torch.Tensor):
        """Split input tensor into subject and objects

        NOTE: the first object is the related one and the rest are negative samples.
        """
        input_embeds = self.embedding(inputs)  # (batch_size, num_entities, hidden_dim), dim 1 includes (child, parent, negative_parents*)
        objects = input_embeds.narrow(dim=1, start=1, length=input_embeds.size(1) - 1)  # use .narrow to keep dim
        subject = input_embeds.narrow(dim=1, start=0, length=1).expand_as(objects)
        return subject, objects

    def forward(self, inputs: torch.Tensor):
        subject, objects = self.unpack_embeddings(inputs)
        return self.dist(subject, objects)

    def dist_loss(self, pred_dists: torch.Tensor):
        """Computing log-softmax loss over poincare distances between the subject entity and the object entities.
        """
        # pred_dists has shape (batch_size, num_distances);
        # the first one is always the distance with the related entity
        # sum(logsoftmax(pred_dists)) / batch_size
        # NOTE: this is equivalent to:
        # loss_func = torch.nn.CrossEntropyLoss()
        # loss_func(pred.neg(), torch.zeros(batch_size, dtype=torch.long).to(0))
        return (
            -torch.sum(-pred_dists[:, 0] - torch.log(torch.sum(torch.exp(-pred_dists), dim=1))) / pred_dists.shape[0]
        )
