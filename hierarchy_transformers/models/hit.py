import torch
from geoopt.manifolds import PoincareBall
from sentence_transformers import SentenceTransformer, models
import logging

logger = logging.getLogger(__name__)


def load_pretrained(pretrained: str, device: torch.device) -> SentenceTransformer:
    """Load a sentence transformer from either the `sentence_transformers` library
    or `transformers` library.
    """
    try:
        # Load from sentence_transformers library
        pretrained_model = SentenceTransformer(pretrained, device=device)
        transformer = pretrained_model._modules["0"]
        pooling = pretrained_model._modules["1"]
        assert isinstance(pooling, models.Pooling)
        logger.info(f"Load `{pretrained}` from `sentence-transformers` with existing pooling.")
    except:
        # Load from huggingface transformers library
        transformer = models.Transformer(pretrained, max_seq_length=256)
        pooling = models.Pooling(transformer.get_word_embedding_dimension())
        logger.info(f"Load `{pretrained}` from `huggingface-transformers` with new pooling.")

    return SentenceTransformer(modules=[transformer, pooling], device=device)


def get_circum_poincareball(embed_dim: int) -> PoincareBall:
    curvature = 1 / embed_dim
    manifold = PoincareBall(c=curvature)
    logging.info(f"Poincare ball curvature: {manifold.c}")
    return manifold
