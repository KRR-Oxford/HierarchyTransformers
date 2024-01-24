# Copyright 2023 Yuan He

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from geoopt.manifolds import PoincareBall
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Transformer
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
        # PoincareBall in geoopt will be wrongly classified as a sub-module
        # so we use a dictionary to store the manifold
        self._register_buffer = {"manifold": self.get_circum_poincareball(self.embed_dim)}

    @property
    def embed_dim(self):
        return self._first_module().get_word_embedding_dimension()

    @property
    def manifold(self):
        return self._register_buffer["manifold"]

    @classmethod
    def load_pretrained(cls, pretrained: str, device: Optional[torch.device] = None):
        """Load a sentence transformer from either the `sentence_transformers` library
        or `transformers` library.
        """
        try:
            # Load from sentence_transformers library
            pretrained_model = SentenceTransformer(pretrained, device=device)
            transformer = pretrained_model._modules["0"]
            pooling = pretrained_model._modules["1"]
            assert isinstance(pooling, Pooling)
            logger.info(f"Load `{pretrained}` from `sentence-transformers` with existing pooling.")
        except:
            # Load from huggingface transformers library
            transformer = Transformer(pretrained, max_seq_length=256)
            pooling = Pooling(transformer.get_word_embedding_dimension())
            logger.info(f"Load `{pretrained}` from `huggingface-transformers` with new pooling.")

        return cls(modules=[transformer, pooling], device=device)

    @staticmethod
    def get_circum_poincareball(embed_dim: int) -> PoincareBall:
        curvature = 1 / embed_dim
        manifold = PoincareBall(c=curvature)
        logging.info(f"Poincare ball curvature: {manifold.c}")
        return manifold
