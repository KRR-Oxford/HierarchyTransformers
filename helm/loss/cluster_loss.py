import torch
import torch.nn.functional as F
from geoopt.manifolds import PoincareBall


class ClusteringLoss(torch.nn.Module):
    def __init__(self, manifold: PoincareBall, positive_margin: float, negative_margin: float):
        super(ClusteringLoss, self).__init__()
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
        distances = self.manifold.dist(rep_anchor, rep_other)
        positive_loss = labels.float() * F.relu(distances - self.positive_margin)
        negative_loss = (1 - labels).float() * F.relu(self.negative_margin - distances)
        cluster_loss = positive_loss + negative_loss
        return cluster_loss.mean()


class ClusteringTripletLoss(torch.nn.Module):
    def __init__(self, manifold: PoincareBall, margin: float):
        super(ClusteringTripletLoss, self).__init__()
        self.manifold = manifold
        self.margin = margin

    def get_config_dict(self):
        config = {
            "distance_metric": f"PoincareBall(c={self.manifold.c}).dist",
            "margin": self.margin,
        }
        return config

    def forward(self, rep_anchor: torch.Tensor, rep_positive: torch.Tensor, rep_negative: torch.Tensor):
        distance_positive = self.manifold.dist(rep_anchor, rep_positive)
        distance_negative = self.manifold.dist(rep_anchor, rep_negative)
        cluster_triplet_loss = F.relu(distance_positive - distance_negative + self.margin)
        return cluster_triplet_loss.mean()
