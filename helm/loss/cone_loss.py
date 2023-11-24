import torch
import torch.nn.functional as F
from geoopt.manifolds import PoincareBall


class EntailmentConeLoss(torch.nn.Module):
    """Entailment cone loss defined in unit Poincare ball."""

    def __init__(self, manifold: PoincareBall, min_euclidean_norm: float, margin: float):
        super(EntailmentConeLoss, self).__init__()
        self.manifold = manifold
        assert self.manifold.c == 1.0, f"Entailment cone loss is not defined for curvature: {manifold.c}."
        self.min_euclidean_norm = min_euclidean_norm
        self.margin = margin

    def get_config_dict(self):
        config = {"distance_metric": "PoincareBall(c=1.0).cone_angle", "margin": self.margin}
        return config

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

    def forward(self, rep_anchor: torch.Tensor, rep_other: torch.Tensor, labels: torch.Tensor):
        # anchors are children
        energies = self.energy(cone_tip=rep_other, u=rep_anchor)
        cone_loss = labels.float() * energies + (1 - labels).float() * F.relu(self.margin - energies)
        return cone_loss.mean()


class EntailmentConeTripletLoss(torch.nn.Module):
    """A variant of the `EntailmentConeLoss` when inputs are triplets."""

    def __init__(self, manifold: PoincareBall, min_euclidean_norm: float, margin: float):
        super(EntailmentConeTripletLoss, self).__init__()
        self.manifold = manifold
        assert self.manifold.c == 1.0, f"Entailment cone loss is not defined for curvature: {manifold.c}."
        self.min_euclidean_norm = min_euclidean_norm
        self.margin = margin

    def get_config_dict(self):
        config = {"distance_metric": "PoincareBall(c=1.0).cone_angle", "margin": self.margin}
        return config

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

    def forward(self, rep_anchor: torch.Tensor, rep_positive: torch.Tensor, rep_negative: torch.Tensor):
        # anchors are children
        energies_positive = self.energy(cone_tip=rep_positive, u=rep_anchor)
        energies_negative = self.energy(cone_tip=rep_negative, u=rep_anchor)
        cone_triplet_loss = F.relu(energies_positive - energies_negative + self.margin)
        return cone_triplet_loss.mean()
