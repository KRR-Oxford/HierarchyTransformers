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
        logging.info(f"Poincare ball curvature: {self.manifold.c}")

        self.distance_metric = self.manifold.dist
        # self.dist = lambda u, v: torch.acosh(1 + 2 * (u - v).norm(dim=-1).pow(2) / (1 - u.norm(dim=-1).pow(2)) * (1 - v.norm(dim=-1).pow(2)))

        # loss combination
        self.loss_weights = loss_weights

        # clustering loss (distance loss)
        self.min_distance = min_distance
        self.cluster_loss_margin = cluster_loss_margin

        self.manifold_origin = self.manifold.origin(self.embed_dim)
        self.centri_loss_margin = centri_loss_margin

        self.min_euclidean_norm = min_euclidean_norm
        self.cone_loss_margin = cone_loss_margin

    def get_config_dict(self):
        # distance_metric_name = self.distance_metric.__name__
        config = {"distance_metric": f"PoincareBall(c={self.curvature})"}
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

        # if self.training_mode == "smaller_cube":
        #     # shrink the cube into the unit ball
        #     rep_anchor = rep_anchor / np.sqrt(self.embed_dim)
        #     rep_other = rep_other / np.sqrt(self.embed_dim)

        # CLUSTERING LOSS
        distances = self.distance_metric(rep_anchor, rep_other)
        cluster_loss = labels.float() * F.relu(distances - self.min_distance) + (1 - labels).float() * F.relu(
            self.cluster_loss_margin - distances
        )
        cluster_loss = cluster_loss.mean()

        # CENTRIPETAL LOSS
        rep_anchor_hyper_norms = self.distance_metric(rep_anchor, self.manifold_origin.to(rep_anchor.device))
        rep_other_hyper_norms = self.distance_metric(rep_other, self.manifold_origin.to(rep_other.device))
        # child further than parent w.r.t. origin
        centri_loss = labels.float() * F.relu(self.centri_loss_margin + rep_other_hyper_norms - rep_anchor_hyper_norms)
        centri_loss = centri_loss.sum() / labels.float().sum()

        # ENTAILMENT CONE LOSS (only make sense for unit Poincare Ball)
        # energies = self.energy(cone_tip=rep_other, u=rep_anchor)
        # cone_loss = labels.float() * energies.pow(2) + (1 - labels).float() * F.relu(self.cone_loss_margin - energies).pow(2)
        # cone_loss = cone_loss.mean()

        loss = (
            self.loss_weights["cluster"] * cluster_loss
            + self.loss_weights["centri"] * centri_loss
            # + self.loss_weights["cone"] * cone_loss
        )
        logger.info(
            f"weighted={loss:.6f};"
            + f"cluster={cluster_loss:.6f};"
            + f"centri={centri_loss:.6f};"
            # + f"cone={cone_loss:.6f}"
        )
        # logger.info(f"weighted={loss}; cluster={cluster_loss}; centri={centri_loss}; cone={cone_loss}")
        return loss
