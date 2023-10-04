import torch

torch.set_default_dtype(torch.float64)
from sentence_transformers import SentenceTransformer
from geoopt.manifolds import PoincareBall
from geoopt.tensor import ManifoldParameter
from deeponto.onto import Taxonomy
from typing import Optional


class HOLM(torch.nn.Module):
    """Hyperbolic Ontology Embedding from Language Models."""

    def __init__(
        self,
        taxonomy: Taxonomy,
        embed_dim: int,  # Poincare ball dimension
        init_from_sbert: Optional[str] = "all-MiniLM-L6-v2",
    ):
        super().__init__()

        # do not save the graph directly as pickling is expensive
        self.taxonomy = taxonomy
        self.idx2ent = {idx: ent for idx, ent in enumerate(self.taxonomy.nodes)}
        self.ent2idx = {v: k for k, v in self.idx2ent.items()}
        self.embed_dim = embed_dim
        self.static_embed = not init_from_sbert
        self.sbert = SentenceTransformer(init_from_sbert) if init_from_sbert else None

        self.manifold = PoincareBall()
        # d(u, v) = arcosh(1 + 2 \frac{\|u - v \|^2}{(1 - \| u \|^2)(1 - \| v \|^2)}) or the one defined with mobius addition
        self.dist = self.manifold.dist

        if self.static_embed:
            self.embed = self.init_static_graph_embedding(len(self.idx2ent), self.embed_dim, 1e-3)

    def init_static_graph_embedding(self, static_entity_size: int, embed_dim: int, init_weights: float):
        # init embedding weights to somewhere near the origin
        static_embedding = torch.nn.Embedding(static_entity_size, embed_dim, sparse=False, max_norm=1.0)
        static_embedding.weight.data.uniform_(-init_weights, init_weights)
        static_embedding.weight = ManifoldParameter(static_embedding.weight, manifold=self.manifold)
        return static_embedding

    def forward(self, subject, *objects):
        pass

    # def unpack_embeddings(self, inputs: torch.Tensor):
    #     """Split input tensor into subject and objects

    #     NOTE: the first object is the related one and the rest are negative samples.
    #     """
    #     input_embeds = self.embed(
    #         inputs
    #     )  # (batch_size, num_entities, hidden_dim), dim 1 includes (child, parent, negative_parents*)
    #     objects = input_embeds.narrow(dim=1, start=1, length=input_embeds.size(1) - 1)  # use .narrow to keep dim
    #     subject = input_embeds.narrow(dim=1, start=0, length=1).expand_as(objects)
    #     return subject, objects

    # def forward(self, inputs: torch.Tensor):
    #     subject, objects = self.unpack_embeddings(inputs)
    #     return self.dist(subject, objects)
