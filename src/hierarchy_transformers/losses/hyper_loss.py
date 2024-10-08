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

from typing import Iterable, Dict, Union, Tuple
import torch
import logging

from .cluster_loss import *
from .centri_loss import *
from .cone_loss import *
from ..models import HierarchyTransformer

logger = logging.getLogger(__name__)


class HyperbolicLoss(torch.nn.Module):
    """
    Hyperbolic loss that combines defined individual losses and applies weights."""

    def __init__(
        self,
        model: HierarchyTransformer,
        apply_triplet_loss: bool = False,
        *weight_and_loss: Tuple[
            float, Union[ClusteringConstrastiveLoss, CentripetalContrastiveLoss, EntailmentConeConstrastiveLoss]
        ],
    ):
        super().__init__()

        self.model = model
        self.apply_triplet_loss = apply_triplet_loss
        self.weight_and_loss = weight_and_loss

    def get_config_dict(self):
        # distance_metric_name = self.distance_metric.__name__
        config = {"distance_metric": f"combined"}
        for weight, loss_func in self.weight_and_loss:
            config[type(loss_func).__name__] = {"weight": weight, **loss_func.get_config_dict()}
        return config

    def forward(self, sentence_features: Iterable[Dict[str, torch.Tensor]], labels: torch.Tensor):
        """
        Forward propagation that extends from [`sentence_transformers.losses`](https://github.com/UKPLab/sentence-transformers/tree/master/sentence_transformers/losses).
        """
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]

        if not self.apply_triplet_loss:
            assert len(reps) == 2
            rep_anchor, rep_other = reps
        else:
            assert len(reps) == 3
            rep_anchor, rep_positive, rep_negative = reps

        weighted_loss = 0.0
        report = {"weighted": None}
        for weight, loss_func in self.weight_and_loss:
            if not self.apply_triplet_loss:
                cur_loss = loss_func(rep_anchor, rep_other, labels)
            else:
                cur_loss = loss_func(rep_anchor, rep_positive, rep_negative)
            report[type(loss_func).__name__] = round(cur_loss.item(), 6)
            weighted_loss += weight * cur_loss
        report["weighted"] = round(weighted_loss.item(), 6)
        logging.info(report)

        return weighted_loss
