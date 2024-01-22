import torch
import torch.nn.functional as F
from geoopt.manifolds import PoincareBall


class CentripetalTripletLoss(torch.nn.Module):
    r"""Hyperbolic loss that regulates the norms of child and parent entities.

    Essentially, this loss is expected to achieve:
    $$
        d(child, origin) > d(parent, origin)
    $$

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
        rep_anchor_hyper_norms = self.manifold.dist0(rep_anchor)
        rep_positive_hyper_norms = self.manifold.dist0(rep_positive)
        # child further than parent w.r.t. origin
        centri_triplet_loss = F.relu(self.margin + rep_positive_hyper_norms - rep_anchor_hyper_norms)
        return centri_triplet_loss.mean()


class CentripetalContrastiveLoss(torch.nn.Module):
    r"""Hyperbolic loss that regulates the norms of child and parent entities.

    Essentially, this loss is expected to achieve:
    $$
        d(child, origin) > d(parent, origin)
    $$

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
        rep_anchor_hyper_norms = self.manifold.dist0(rep_anchor)
        rep_other_hyper_norms = self.manifold.dist0(rep_other)
        # child further than parent w.r.t. origin
        centri_loss = labels.float() * F.relu(self.margin + rep_other_hyper_norms - rep_anchor_hyper_norms)
        return centri_loss.sum() / labels.float().sum()
