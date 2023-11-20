from typing import Iterable, Dict
import torch
import torch.nn.functional as F
from sentence_transformers.SentenceTransformer import SentenceTransformer
import logging
from geoopt.manifolds import PoincareBall
import numpy as np

logger = logging.getLogger(__name__)


class HyperbolicLoss(torch.nn.Module):
    """Combined loss of three hyperbolic loss functions:
    1. Clustering loss: related entities are closer to each other;
    2. Centripetal loss: parents closer to the origin than children;
    3. Cone loss: angle at child w.r.t. the parent smaller than the half aperture at parent.
    """

    def __init__(
        self,
        model: SentenceTransformer,
        training_mode: str = "bigger_ball",  # or "smaller_cube"
        loss_weights: dict = {"cluster": 1.0, "centri": 1.0, "cone": 1.0},
        min_distance: float = 0.1,
        cluster_loss_margin: float = 1.0,
        centri_loss_margin: float = 0.1,
        min_euclidean_norm: float = 0.1,
        cone_loss_margin: float = 0.1,
    ):
        super(HyperbolicLoss, self).__init__()
        self.model = model
        self.training_mode = training_mode
        self.embed_dim = self.model._first_module().get_word_embedding_dimension()
        if self.training_mode == "bigger_ball":
            self.curvature = 1 / self.embed_dim
        elif self.training_mode == "smaller_cube":
            self.curvature = 1.0
        self.manifold = PoincareBall(c=self.curvature)
        self.dist = self.manifold.dist
        # self.dist = lambda u, v: torch.acosh(1 + 2 * (u - v).norm(dim=-1).pow(2) / (1 - u.norm(dim=-1).pow(2)) * (1 - v.norm(dim=-1).pow(2)))
        self.manifold_origin = self.manifold.origin(self.embed_dim)
        self.loss_weights = loss_weights
        self.min_distance = min_distance
        self.cluster_loss_margin = cluster_loss_margin
        self.centri_loss_margin = centri_loss_margin
        self.min_euclidean_norm = min_euclidean_norm
        self.cone_loss_margin = cone_loss_margin
        
        # self._running_loss = 0.
        # self._num_batches = 0

    def half_cone_aperture(self, cone_tip: torch.Tensor):
        """Angle between the axis [0, x] (line through 0 and x) and the boundary of the cone at x,
        where x is the cone tip.
        """
        # cone tip means the point x is the tip of the hyperbolic cone
        norm_tip = cone_tip.norm(dim=-1).clamp(min=self.min_euclidean_norm)  # to prevent undefined aperture
        return torch.arcsin(self.min_euclidean_norm * (1 - (norm_tip**2)) / norm_tip)

    def cone_angle_at_u(self, cone_tip: torch.Tensor, u: torch.Tensor):
        """Angle between the axis [0, x] and the line [x, u]. This angle should be smaller than the
        half cone aperture at x for real children.
        """
        # parent point is treated as the cone tip
        norm_tip = cone_tip.norm(dim=-1)
        norm_child = u.norm(dim=-1)
        dot_prod = (cone_tip * u).sum(dim=-1)
        edist = (cone_tip - u).norm(dim=-1)  # euclidean distance
        numerator = dot_prod * (1 + norm_tip**2) - norm_tip**2 * (1 + norm_child**2)
        denomenator = norm_tip * edist * torch.sqrt(1 + (norm_child**2) * (norm_tip**2) - 2 * dot_prod)
        return torch.arccos(numerator / denomenator)

    def energy(self, cone_tip: torch.Tensor, u: torch.Tensor):
        """Enery function defined as: max(0, cone_angle(u) - half_cone_aperture) given a cone tip."""
        return F.relu(self.cone_angle_at_u(cone_tip, u) - self.half_cone_aperture(cone_tip))

    def get_config_dict(self):
        # distance_metric_name = self.distance_metric.__name__
        config = dict()
        config["cluster"] = {
            "weight": self.loss_weights["cluster"],
            "min_distance": self.min_distance,
            "margin": self.cluster_loss_margin,
        }
        config["centri"] = {"weight": self.loss_weights["centri"], "margin": self.centri_loss_margin}
        config["cone"] = {"weight": self.loss_weights["cone"], "margin": self.cone_loss_margin}
        return config

    def forward(self, sentence_features: Iterable[Dict[str, torch.Tensor]], labels: torch.Tensor):
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        assert len(reps) == 2
        rep_anchor, rep_other = reps

        if self.training_mode == "smaller_cube":
            # shrink the cube into the unit ball
            rep_anchor = rep_anchor / np.sqrt(self.embed_dim)
            rep_other = rep_other / np.sqrt(self.embed_dim)

        # CLUSTERING LOSS
        distances = self.dist(rep_anchor, rep_other)
        # cluster_losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2))
        cluster_loss = (
            labels.float() * F.relu(distances - self.min_distance).pow(2) + 
            (1 - labels).float() * F.relu(self.cluster_loss_margin - distances).pow(2)
        )

        cluster_loss = cluster_loss.mean()

        # CENTRIPETAL LOSS
        rep_anchor_hyper_norms = self.dist(rep_anchor, self.manifold_origin.to(rep_anchor.device))
        rep_other_hyper_norms = self.dist(rep_other, self.manifold_origin.to(rep_other.device))
        # child further than parent w.r.t. origin
        centri_loss = labels.float() * F.relu(self.centri_loss_margin + rep_other_hyper_norms - rep_anchor_hyper_norms)
        centri_loss = centri_loss.sum() / labels.float().sum()

        # ENTAILMENT CONE LOSS (only make sense for unit Poincare Ball)
        energies = self.energy(cone_tip=rep_other, u=rep_anchor)
        cone_loss = (
            labels.float() * energies.pow(2) + 
            (1 - labels).float() * F.relu(self.cone_loss_margin - energies).pow(2)
        )
        cone_loss = cone_loss.mean()
        cone_loss = 0.0 if torch.isnan(cone_loss) else cone_loss

        loss = (
            self.loss_weights["cluster"] * cluster_loss
            + self.loss_weights["centri"] * centri_loss
            + self.loss_weights["cone"] * cone_loss
        )
        logger.info(
            f"weighted={loss:.6f};" + 
            f"cluster={cluster_loss:.6f};" +
            f"centri={centri_loss:.6f};" + 
            f"cone={cone_loss:.6f}"
        )
        return loss
