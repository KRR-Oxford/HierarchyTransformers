from typing import Iterable, Dict
import torch
import torch.nn.functional as F
from sentence_transformers.SentenceTransformer import SentenceTransformer
import logging
from geoopt.manifolds import PoincareBall

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
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
        manifold: PoincareBall,
        loss_weights: dict = {"cluster": 1.0, "centri": 1.0, "cone": 1.0},
    ):
        super(HyperbolicLoss, self).__init__()
        self.manifold = manifold
        self.distance_metric = manifold.dist
        self.model = model
        self.loss_weights = loss_weights
        self.min_norm = 0.1

    def half_cone_aperture(self, cone_tip: torch.Tensor):
        """Angle between the axis [0, x] (line through 0 and x) and the boundary of the cone at x,
        where x is the cone tip.
        """
        # cone tip means the point x is the tip of the hyperbolic cone
        norm_tip = cone_tip.norm(dim=-1).clamp(min=self.min_norm)  # to prevent undefined aperture
        return torch.arcsin(self.min_norm * (1 - (norm_tip**2)) / norm_tip)

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
        return {"distance_metric": self.distance_metric.__name__, **self.loss_weights}

    def forward(self, sentence_features: Iterable[Dict[str, torch.Tensor]], labels: torch.Tensor):
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        assert len(reps) == 2
        rep_anchor, rep_other = reps

        # CLUSTERING LOSS
        distances = self.distance_metric(rep_anchor, rep_other)
        # cluster_losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2))
        cluster_loss = 0.5 * (
            labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(1.0 - distances).pow(2)
        )
        cluster_loss = cluster_loss.mean()

        # CENTRIPETAL LOSS
        rep_anchor_hyper_norms = self.distance_metric(
            rep_anchor, self.manifold.origin(rep_anchor.shape).to(rep_anchor.device)
        )
        rep_other_hyper_norms = self.distance_metric(
            rep_other, self.manifold.origin(rep_other.shape).to(rep_other.device)
        )
        # child further than parent w.r.t. origin
        centri_loss = labels.float() * F.relu(0.1 + rep_other_hyper_norms - rep_anchor_hyper_norms)
        centri_loss = centri_loss.sum() / labels.float().sum()

        # ENTAILMENT CONE LOSS
        energies = self.energy(cone_tip=rep_other, u=rep_anchor)
        cone_loss = labels.float() * energies.pow(2) + (1 - labels).float() * F.relu(0.1 - energies).pow(2)
        cone_loss = cone_loss.mean()

        # logger.info(labels)
        # logger.info(f"{distances}")
        # logger.info(f"{rep_other_hyper_norms - rep_anchor_hyper_norms}")
        loss = (
            self.loss_weights["cluster"] * cluster_loss
            + self.loss_weights["centri"] * centri_loss
            + self.loss_weights["cone"] * cone_loss
        )
        logger.info(f"weighted_loss={loss}; cluster={cluster_loss}; centri={centri_loss}; cone={cone_loss}.")
        # loss = cluster_loss + centri_loss + cone_loss
        return loss
