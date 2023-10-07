from __future__ import annotations

import torch

torch.set_default_dtype(torch.float64)
from sentence_transformers import SentenceTransformer
from geoopt.manifolds import PoincareBall
from geoopt.tensor import ManifoldParameter
from deeponto.onto import Taxonomy
from typing import Optional


class HyperOntoEmbedfromLM(torch.nn.Module):
    """Hyperbolic Ontology Embedding from Language Models."""

    def __init__(
        self,
        taxonomy: Taxonomy,
        embed_dim: int,  # Poincare ball dimension
        sbert_model: Optional[str] = "all-MiniLM-L6-v2",
        freeze_sbert: bool = True,
    ):
        super().__init__()

        self.taxonomy = taxonomy
        # self.idx2ent = {idx: ent for idx, ent in enumerate(self.taxonomy.nodes)}
        # self.ent2idx = {v: k for k, v in self.idx2ent.items()}
        self.embed_dim = embed_dim
        self.sbert = SentenceTransformer(sbert_model)
        if freeze_sbert:
            for param in self.sbert.parameters():
                param.requires_grad = False
        self.sbert_dim = self.sbert.get_sentence_embedding_dimension()
        self.manifold = PoincareBall()
        # d(u, v) = arcosh(1 + 2 \frac{\|u - v \|^2}{(1 - \| u \|^2)(1 - \| v \|^2)}) or the one defined with mobius addition
        self.dist = self.manifold.dist

        self.euclidean_to_hyperbolic = torch.nn.Sequential(
            torch.nn.Linear(
                self.sbert_dim, self.embed_dim
            ),  # linear mapping from sbert embeddings to the tangent plane of hyperbolic space
            ExpMap0(self.manifold),
            MobiusLinear(self.manifold, self.embed_dim, self.embed_dim),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, subject, *objects):
        subject = torch.tensor(self.sbert.encode(subject)).to(self.device)  # (batch_size, sbert_dim)
        subject = self.euclidean_to_hyperbolic(subject)
        objects = torch.stack([torch.tensor(self.sbert.encode(obj)) for obj in objects]).to(
            self.device
        )  # (batch_size, n_object, sbert_dim)
        objects = self.euclidean_to_hyperbolic(objects)
        subject = subject.unsqueeze(1).expand_as(objects)  # unsqueeze at dim=1, i.e., n_object
        return subject, objects


class ExpMap0(torch.nn.Module):
    def __init__(self, manifold: PoincareBall):
        super().__init__()
        self.manifold = manifold

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.manifold.expmap0(input)


class MobiusLinear(torch.nn.Module):
    def __init__(self, manifold: PoincareBall, in_dim: int, out_dim: int, init_weights: float = 10e-2):
        super().__init__()
        self.manifold = manifold
        self.weights = torch.nn.Parameter(torch.empty((in_dim, out_dim)))
        torch.nn.init.uniform_(self.weights, -init_weights, init_weights)
        self.weights = ManifoldParameter(self.weights, self.manifold)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.manifold.mobius_matvec(self.weights, input)


class ClusteringLoss(torch.nn.Module):
    """Clustering loss to make entities that form a subsumption relationship 
    to be closer to each other, and adding penalty to too close entities.
    """
    def __init__(self, manifold: PoincareBall, min_dist: float = 10e-2):
        super().__init__()
        self.manifold = manifold
        self.min_dist = min_dist
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, subject, objects):
        """The first object is the correct object and the rest are the negative objects."""
        pred_dists = self.manifold.dist(subject, objects)
        correct_object_indices = torch.tensor([0] * len(pred_dists)).to(pred_dists.device)
        # penalising the distance if too small
        penalty = torch.clamp(self.min_dist - pred_dists[:, 0], min=0.0) ** 2
        # - torch.mean(- (pred_dists[:, 0]) - torch.log(torch.sum(torch.exp(-pred_dists), dim=1)))
        return self.cross_entropy(-pred_dists, correct_object_indices) + penalty.mean()


class CentripetalLoss(torch.nn.Module):
    """Centripetal loss to enforce the subsumee (child entity) staying closer to the boundary than
    the subsumer (parent entity). This is achieved by making the Euclidean norm of the subsumer smaller than
    the Euclidean norm of the subsumee.
    """
    def forward(self, subject, objects):
        subject_norm = torch.norm(subject[:, 0, :], dim=-1)  # just one copy of the subject is needed
        object_norm = torch.norm(objects[:, 0, :], dim=-1)  # just the correct object is needed
        # we would like to push away the subject (child) and pull in the object (parent) towards the center of origin
        return torch.clamp(object_norm - subject_norm, min=0.0) ** 2


class HyperOntoEmbedStatic(torch.nn.Module):
    """Hyperbolic Ontology Embedding Static Version (Fixed Embedding Size)."""

    def __init__(
        self,
        taxonomy: Taxonomy,
        embed_dim: int,  # Poincare ball dimension
    ):
        super().__init__()

        self.taxonomy = taxonomy
        self.idx2ent = {idx: ent for idx, ent in enumerate(self.taxonomy.nodes)}
        self.ent2idx = {v: k for k, v in self.idx2ent.items()}
        self.embed_dim = embed_dim
        self.manifold = PoincareBall()
        # d(u, v) = arcosh(1 + 2 \frac{\|u - v \|^2}{(1 - \| u \|^2)(1 - \| v \|^2)}) or the one defined with mobius addition
        self.dist = self.manifold.dist
        self.embed = self.init_static_graph_embedding(len(self.idx2ent), self.embed_dim, 1e-3)

    def init_static_graph_embedding(self, static_entity_size: int, embed_dim: int, init_weights: float):
        # init embedding weights to somewhere near the origin
        static_embedding = torch.nn.Embedding(static_entity_size, embed_dim, sparse=False, max_norm=1.0)
        static_embedding.weight.data.uniform_(-init_weights, init_weights)
        static_embedding.weight = ManifoldParameter(static_embedding.weight, manifold=self.manifold)
        return static_embedding

    def forward(self, inputs: torch.Tensor):
        """Split input tensor into subject and objects

        NOTE: the first object is the related one and the rest are negative samples.
        """
        input_embeds = self.embed(
            inputs
        )  # (batch_size, num_entities, hidden_dim), dim 1 includes (child, parent, negative_parents*)
        objects = input_embeds.narrow(dim=1, start=1, length=input_embeds.size(1) - 1)  # use .narrow to keep dim
        subject = input_embeds.narrow(dim=1, start=0, length=1).expand_as(objects)
        return subject, objects
        # return self.dist(subject, objects)
