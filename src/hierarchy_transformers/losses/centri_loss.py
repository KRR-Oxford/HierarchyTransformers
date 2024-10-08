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


class CentripetalTripletLoss(torch.nn.Module):
    r"""Hyperbolic loss that regulates the norms of child and parent entities.

    Essentially, this loss is expected to achieve:
    
    $$d(child, origin) > d(parent, origin)$$

    Inputs are presented in `(rep_anchor, rep_positive, rep_negative)` but only `(rep_anchor, rep_positive)` pairs are involved in this loss.
    """

    def __init__(self, manifold: PoincareBall, embed_dim: int, margin: float):
        super().__init__()
        self.manifold = manifold
        self.margin = margin
        # self.manifold_origin = self.manifold.origin(embed_dim)

    def get_config_dict(self):
        config = {
            "distance_metric": f"PoincareBall(c={self.manifold.c}).dist(_, origin)",
            "margin": self.margin,
        }
        return config

    def forward(self, rep_anchor: torch.Tensor, rep_positive: torch.Tensor, rep_negative: torch.Tensor):
        """Forward propagation.

        Args:
            rep_anchor (torch.Tensor): The input tensor for child entities.
            rep_positive (torch.Tensor): The input tensor for parent entities.
            rep_negative (torch.Tensor): The input tensor for negative parent entities (actually not required in this loss).
        """
        rep_anchor_hyper_norms = self.manifold.dist0(rep_anchor)
        rep_positive_hyper_norms = self.manifold.dist0(rep_positive)
        # child further than parent w.r.t. origin
        centri_triplet_loss = F.relu(self.margin + rep_positive_hyper_norms - rep_anchor_hyper_norms)
        return centri_triplet_loss.mean()


class CentripetalContrastiveLoss(torch.nn.Module):
    r"""Hyperbolic loss that regulates the norms of child and parent entities.

    Essentially, this loss is expected to achieve:
    
    $$d(child, origin) > d(parent, origin)$$

    Inputs are presented in `(rep_anchor, rep_other, label)` but only `label==1` ones are involved in this loss.
    """

    def __init__(self, manifold: PoincareBall, embed_dim: int, margin: float):
        super().__init__()
        self.manifold = manifold
        self.margin = margin
        # self.manifold_origin = self.manifold.origin(embed_dim)

    def get_config_dict(self):
        config = {
            "distance_metric": f"PoincareBall(c={self.manifold.c}).dist(_, origin)",
            "margin": self.margin,
        }
        return config

    def forward(self, rep_anchor: torch.Tensor, rep_other: torch.Tensor, labels: torch.Tensor):
        """Forward propagation.

        Args:
            rep_anchor (torch.Tensor): The input tensor for child entities.
            rep_other (torch.Tensor): The input tensor for parent and negative parent entities.
            labels (torch.Tensor): Labels indicating whether each `(anchor, other)` pair is a positive subsumption or a negative one (only `label=1` pairs will take an effect in this loss).
        """

        rep_anchor_hyper_norms = self.manifold.dist0(rep_anchor)
        rep_other_hyper_norms = self.manifold.dist0(rep_other)
        # child further than parent w.r.t. origin
        centri_loss = labels.float() * F.relu(self.margin + rep_other_hyper_norms - rep_anchor_hyper_norms)
        return centri_loss.sum() / labels.float().sum()
