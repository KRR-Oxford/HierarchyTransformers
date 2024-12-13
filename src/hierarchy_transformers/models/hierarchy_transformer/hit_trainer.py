# Copyright 2024 Yuan He

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from typing import Any

import torch
from sentence_transformers.trainer import SentenceTransformerTrainer

from .hit import HierarchyTransformer


class HierarchyTransformerTrainer(SentenceTransformerTrainer):
    r"""A simple extension of `SentenceTransformerTrainer` to monitor and log batch losses of `HierarchyTransformer`."""

    def compute_loss(
        self,
        model: HierarchyTransformer,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch=None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        loss_dict = super().compute_loss(
            model=model, inputs=inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
        )
        outputs = None
        if return_outputs:
            loss_dict, outputs = loss_dict
        self.log(
            {
                "cluster_loss": round(loss_dict["cluster_loss"].item(), 4),
                "centri_loss": round(loss_dict["centri_loss"].item(), 4),
                "combined_loss": round(loss_dict["loss"].item(), 4),
            }
        )

        return (loss_dict["loss"], outputs) if return_outputs else loss_dict["loss"]
