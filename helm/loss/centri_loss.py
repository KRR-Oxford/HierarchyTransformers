import torch
import torch.nn.functional as F
from geoopt.manifolds import PoincareBall


class CentripetalLoss(torch.nn.Module):
    """Loss for regulating the distances of child and parent entities to the manifold's origin, i.e.,
        
        dist(child, origin) > dist(parent, origin).
    """
    def __init__(self, manifold: PoincareBall, embed_dim: int, margin: float):
        super(CentripetalLoss, self).__init__()
        self.manifold = manifold
        self.margin = margin
        self.manifold_origin = self.manifold.origin(embed_dim)

    def get_config_dict(self):
        config = {"distance_metric": f"PoincareBall(c={self.manifold.c}).dist(_, origin)", "margin": self.margin}
        return config

    def forward(self, rep_anchor: torch.Tensor, rep_other: torch.Tensor, labels: torch.Tensor):
        rep_anchor_hyper_norms = self.manifold.dist(rep_anchor, self.manifold_origin.to(rep_anchor.device))
        rep_other_hyper_norms = self.manifold.dist(rep_other, self.manifold_origin.to(rep_other.device))
        # child further than parent w.r.t. origin
        centri_loss = labels.float() * F.relu(self.margin + rep_other_hyper_norms - rep_anchor_hyper_norms)
        centri_loss = centri_loss.sum() / labels.float().sum()


class CentripetalTripletLoss(torch.nn.Module):
    """A variant of the `CentripetalLoss` when inputs are triplets.
    """
    def __init__(self, manifold: PoincareBall, embed_dim: int, margin: float):
        super(CentripetalTripletLoss, self).__init__()
        self.manifold = manifold
        self.margin = margin
        self.manifold_origin = self.manifold.origin(embed_dim)

    def get_config_dict(self):
        config = {"distance_metric": f"PoincareBall(c={self.manifold.c}).dist(_, origin)", "margin": self.margin}
        return config

    def forward(self, rep_anchor: torch.Tensor, rep_positive: torch.Tensor, rep_negative: torch.Tensor):
        rep_anchor_hyper_norms = self.manifold.dist(rep_anchor, self.manifold_origin.to(rep_anchor.device))
        rep_positive_hyper_norms = self.manifold.dist(rep_positive, self.manifold_origin.to(rep_positive.device))
        # child further than parent w.r.t. origin
        centri_loss = F.relu(self.margin + rep_positive_hyper_norms - rep_anchor_hyper_norms)
        centri_loss = centri_loss.mean()
