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

import torch
import torch.nn.functional as F
from geoopt.manifolds import PoincareBall


class ClusteringTripletLoss(torch.nn.Module):
    """Hyperbolic loss that clusters entities in subsumptions.

    Essentially, this loss is expected to achieve:
    $$
        d(child, parent) < d(child, non-parent)
    $$

    Inputs are presented in `(rep_anchor, rep_positive, rep_negative)`.
    """

    def __init__(self, manifold: PoincareBall, margin: float):
        super().__init__()
        self.manifold = manifold
        self.margin = margin

    def get_config_dict(self):
        config = {
            "distance_metric": f"PoincareBall(c={self.manifold.c}).dist",
            "margin": self.margin,
        }
        return config

    def forward(self, rep_anchor: torch.Tensor, rep_positive: torch.Tensor, rep_negative: torch.Tensor):
        distances_positive = self.manifold.dist(rep_anchor, rep_positive)
        distances_negative = self.manifold.dist(rep_anchor, rep_negative)
        cluster_triplet_loss = F.relu(distances_positive - distances_negative + self.margin)
        return cluster_triplet_loss.mean()


class ClusteringConstrastiveLoss(torch.nn.Module):
    """Hyperbolic loss that clusters entities in subsumptions.

    Essentially, this loss is expected to achieve:
    $$
        d(child, parent) < d(child, non-parent)
    $$

    Inputs are presented in `(rep_anchor, rep_other, label)`.
    """

    def __init__(self, manifold: PoincareBall, positive_margin: float, negative_margin: float):
        super().__init__()
        self.manifold = manifold
        self.positive_margin = positive_margin
        self.negative_margin = negative_margin

    def get_config_dict(self):
        config = {
            "distance_metric": f"PoincareBall(c={self.manifold.c}).dist",
            "positive_margin": self.positive_margin,
            "negative_margin": self.negative_margin,
        }
        return config

    def forward(self, rep_anchor: torch.Tensor, rep_other: torch.Tensor, labels: torch.Tensor):
        # self.dist = lambda u, v: torch.acosh(1 + 2 * (u - v).norm(dim=-1).pow(2) / (1 - u.norm(dim=-1).pow(2)) * (1 - v.norm(dim=-1).pow(2)))
        distances = self.manifold.dist(rep_anchor, rep_other)
        positive_loss = labels.float() * F.relu(distances - self.positive_margin)
        negative_loss = (1 - labels).float() * F.relu(self.negative_margin - distances)
        cluster_loss = positive_loss + negative_loss
        return cluster_loss.mean()
