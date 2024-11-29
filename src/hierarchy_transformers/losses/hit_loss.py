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

from typing import Iterable, Dict
import logging
import torch
import torch.nn.functional as F
from geoopt.manifolds import PoincareBall
from ..models import HierarchyTransformer
from ..utils import format_citation

logger = logging.getLogger(__name__)


class HierarchyTransformerLoss(torch.nn.Module):
    """
    Hyperbolic loss that linearly combines hperbolic clustering loss and hyperbolic Centripetal loss and applies weights for joint optimisation.
    """

    def __init__(
        self,
        model: HierarchyTransformer,
        clustering_loss_weight: float = 1.0,
        clustering_loss_margin: float = 5.0,
        centripetal_loss_weight: float = 1.0,
        centripetal_loss_margin: float = 0.5,
    ):
        super().__init__()

        self.model = model
        self.cluster_loss = HyperbolicClusteringLoss(self.model.manifold, clustering_loss_margin)
        self.centri_loss = HyperbolicCentripetalLoss(self.model.manifold, centripetal_loss_margin)
        self.cluster_weight = clustering_loss_weight
        self.centri_weight = centripetal_loss_weight

    def get_config_dict(self):
        # distance_metric_name = self.distance_metric.__name__
        config = {"distance_metric": f"PoincareBall(c={self.manifold.c}).dist and dist0"}
        config[HyperbolicClusteringLoss.__name__] = {
            "weight": self.cluster_weight,
            **self.cluster_loss.get_config_dict(),
        }
        config[HyperbolicCentripetalLoss.__name__] = {
            "weight": self.centri_weight,
            **self.centri_loss.get_config_dict(),
        }
        return config

    def forward(self, sentence_features: Iterable[Dict[str, torch.Tensor]], labels: torch.Tensor):
        """
        Forward propagation that extends from [`sentence_transformers.losses`](https://github.com/UKPLab/sentence-transformers/tree/master/sentence_transformers/losses).
        """
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        assert len(reps) == 3
        rep_anchor, rep_positive, rep_negative = reps

        # compute and combine hyperbolic clustering and centripetal losses
        cluster_loss = self.cluster_loss(rep_anchor, rep_positive, rep_negative)
        centri_loss = self.centri_loss(rep_anchor, rep_positive, rep_negative)
        combined_loss = self.cluster_weight * cluster_loss + self.centri_weight * centri_loss

        # batch reporting
        report = {
            "cluster_loss": round(cluster_loss.item(), 6),
            "centri_loss": round(centri_loss.item(), 6),
            "combined_loss": round(combined_loss.item(), 6),
        }
        logger.info(report)

        return combined_loss

    @property
    def citation(self) -> str:
        return format_citation(
            """ 
            @article{he2024language,
              title={Language models as hierarchy encoders},
              author={He, Yuan and Yuan, Zhangdie and Chen, Jiaoyan and Horrocks, Ian},
              journal={arXiv preprint arXiv:2401.11374},
              year={2024}
            }
            """
        )


class HyperbolicClusteringLoss(torch.nn.Module):
    """Hyperbolic loss that clusters entities in subsumptions.

    Essentially, this loss is expected to achieve:

    $$d(child, parent) < d(child, negative)$$

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
        """Forward propagation.

        Args:
            rep_anchor (torch.Tensor): The input tensor for child entities.
            rep_positive (torch.Tensor): The input tensor for parent entities.
            rep_negative (torch.Tensor): The input tensor for negative parent entities.
        """
        distances_positive = self.manifold.dist(rep_anchor, rep_positive)
        distances_negative = self.manifold.dist(rep_anchor, rep_negative)
        cluster_triplet_loss = F.relu(distances_positive - distances_negative + self.margin)
        return cluster_triplet_loss.mean()

    @property
    def citation(self) -> str:
        return format_citation(
            """ 
            @article{he2024language,
              title={Language models as hierarchy encoders},
              author={He, Yuan and Yuan, Zhangdie and Chen, Jiaoyan and Horrocks, Ian},
              journal={arXiv preprint arXiv:2401.11374},
              year={2024}
            }
            """
        )


class HyperbolicCentripetalLoss(torch.nn.Module):
    r"""Hyperbolic loss that regulates the norms of child and parent entities.

    Essentially, this loss is expected to achieve:

    $$d(child, origin) > d(parent, origin)$$

    Inputs are presented in `(rep_anchor, rep_positive, rep_negative)` but only `(rep_anchor, rep_positive)` pairs are involved in this loss.
    """

    def __init__(self, manifold: PoincareBall, margin: float):
        super().__init__()
        self.manifold = manifold
        self.margin = margin

    def get_config_dict(self):
        config = {
            "distance_metric": f"PoincareBall(c={self.manifold.c}).dist0",
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

    @property
    def citation(self) -> str:
        return format_citation(
            """ 
            @article{he2024language,
              title={Language models as hierarchy encoders},
              author={He, Yuan and Yuan, Zhangdie and Chen, Jiaoyan and Horrocks, Ian},
              journal={arXiv preprint arXiv:2401.11374},
              year={2024}
            }
            """
        )
