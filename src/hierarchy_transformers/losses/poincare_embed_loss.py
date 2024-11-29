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
from ..utils import format_citation

class PoincareEmbeddingStaticLoss(torch.nn.Module):
    """Poincare embedding loss.
    
    Essentially, this loss is expected to achieve:

    $$d(child, parent) < d(child, negative)$$

    Inputs are presented in `(subject, *objects)` where the first `object` is positive and the rest are negative.
    
    This is designed for the static embedding implementation.
    """
    
    def __init__(self, manifold: PoincareBall):
        super().__init__()
        self.manifold = manifold
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        
    def forward(self, subject: torch.Tensor, objects: torch.Tensor):
        # first object is always the correct one
        pred_dists = self.manifold.dist(subject, objects)
        correct_object_indices = torch.tensor([0] * len(pred_dists)).to(pred_dists.device)
        return self.cross_entropy(-pred_dists, correct_object_indices)

    @property
    def citation(self) -> str:
        return format_citation(
            """ 
            @article{nickel2017poincare,
              title={Poincar{\'e} embeddings for learning hierarchical representations},
              author={Nickel, Maximillian and Kiela, Douwe},
              journal={Advances in neural information processing systems},
              volume={30},
              year={2017}
            }
            """
        )