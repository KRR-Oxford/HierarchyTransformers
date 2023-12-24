import torch
from geoopt.manifolds import PoincareBall
from sentence_transformers import SentenceTransformer, models
import logging
from typing import Union, Optional, Iterable

logger = logging.getLogger(__name__)


class HierarchyTransformer(SentenceTransformer):
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        modules: Optional[Iterable[torch.nn.Module]] = None,
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
        use_auth_token: Union[bool, str, None] = None,
    ):
        super().__init__(model_name_or_path, modules, device, cache_folder, use_auth_token)
        self.embed_dim = self._first_module().get_word_embedding_dimension()
        self.manifold = self.get_circum_poincareball(self.embed_dim)

    @classmethod
    def load_pretrained(cls, pretrained: str, device: torch.device):
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

        return cls(modules=[transformer, pooling], device=device)

    @staticmethod
    def get_circum_poincareball(embed_dim: int) -> PoincareBall:
        curvature = 1 / embed_dim
        manifold = PoincareBall(c=curvature)
        logging.info(f"Poincare ball curvature: {manifold.c}")
        return manifold
