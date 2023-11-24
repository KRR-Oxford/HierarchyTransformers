from typing import Iterable, Dict, Union, Tuple
import torch
import torch.nn.functional as F
from sentence_transformers.SentenceTransformer import SentenceTransformer
import logging

from .cluster_loss import *
from .centri_loss import *
from .cone_loss import *

logger = logging.getLogger(__name__)


class HyperbolicLoss(torch.nn.Module):
    """Combined loss of multiple hyperbolic loss functions."""

    def __init__(
        self,
        model: SentenceTransformer,
        *weight_and_loss: Tuple[float, Union[ClusteringLoss, CentripetalLoss, EntailmentConeLoss]],
    ):
        super(HyperbolicLoss, self).__init__()

        self.model = model
        self.weight_and_loss = weight_and_loss

    def get_config_dict(self):
        # distance_metric_name = self.distance_metric.__name__
        config = {"distance_metric": f"combined"}
        for weight, loss_func in self.loss_dict.items():
            config[type(loss_func).__name__] = {"weight": weight, **loss_func.get_config_dict()}
        return config

    def forward(self, sentence_features: Iterable[Dict[str, torch.Tensor]], labels: torch.Tensor):
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        assert len(reps) == 2
        rep_anchor, rep_other = reps

        weighted_loss = 0.0
        report = dict()
        for weight, loss_func in self.weight_and_loss:
            cur_loss = loss_func(rep_anchor, rep_other, labels)
            report[type(loss_func).__name__] = cur_loss.item()
            weighted_loss += weight * cur_loss
        report["weighted"] = round(weighted_loss.item(), 6)
        logging.info(report)

        return weighted_loss


class HyperbolicTripletLoss(torch.nn.Module):
    """Combined loss of multiple hyperbolic triplet loss functions."""

    def __init__(
        self,
        model: SentenceTransformer,
        *weight_and_loss: Tuple[float, Union[ClusteringTripletLoss, CentripetalTripletLoss, EntailmentConeTripletLoss]],
    ):
        super(HyperbolicLoss, self).__init__()

        self.model = model
        self.weight_and_loss = weight_and_loss

    def get_config_dict(self):
        # distance_metric_name = self.distance_metric.__name__
        config = {"distance_metric": f"combined"}
        for weight, loss_func in self.loss_dict.items():
            config[type(loss_func).__name__] = {"weight": weight, **loss_func.get_config_dict()}
        return config

    def forward(self, sentence_features: Iterable[Dict[str, torch.Tensor]], labels: torch.Tensor):
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        assert len(reps) == 3
        rep_anchor, rep_positive, rep_other = reps

        weighted_loss = 0.0
        report = dict()
        for weight, loss_func in self.weight_and_loss:
            cur_loss = loss_func(rep_anchor, rep_positive, rep_other)
            report[type(loss_func).__name__] = cur_loss.item()
            weighted_loss += weight * cur_loss
        report["weighted"] = round(weighted_loss.item(), 6)
        logging.info(report)

        return weighted_loss
