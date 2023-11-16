from __future__ import annotations

import torch
# torch.set_default_dtype(torch.float64)
from transformers import AutoTokenizer, AutoModel
from geoopt.manifolds import PoincareBall
from geoopt.tensor import ManifoldParameter
from deeponto.onto import Taxonomy
from typing import Optional, Callable
import numpy as np

from .pooling import *

class SentenceBERT(torch.nn.Module):
    
    def __init__(
        self, 
        pretrained_model: str = "all-MiniLM-L6-v2", 
        pooling_func: Callable = mean_pooling,
        max_length: int = 128
    ):
        super().__init__()
        self.pretrained = AutoModel.from_pretrained(pretrained_model)
        self.tokenizer = AutoModel.from_tokenizer(pretrained_model)
        self.pooling = pooling_func
        self.max_length = max_length
        
    def forward(self, texts):
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
        model_output = self.pretrained(**encoded_input)
        return model_output
