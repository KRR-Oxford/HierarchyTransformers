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
from __future__ import annotations

import math

import torch
from tqdm import tqdm


def f1_score(predictions: torch.Tensor, labels: torch.Tensor, truth_label: int = 1):
    """Pytorch tensor implementation of `f1_score` computation.

    Args:
        predictions (torch.Tensor): Predictions.
        labels (torch.Tensor): Reference labels.
        truth_label (int, optional): Specify which label represents the truth. Defaults to `1`.

    Returns:
        results (dict): result dictionary containing `Precision`, `Recall`, and `F1`.
    """
    tp = torch.sum((labels == truth_label) & (predictions == truth_label))  # correct and positive
    fp = torch.sum((labels != truth_label) & (predictions == truth_label))  # incorrect but positive
    fn = torch.sum((labels == truth_label) & (predictions != truth_label))  # correct but negative
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return {"precision": precision.item(), "recall": recall.item(), "f1": f1.item()}  # recall is ACC+


def accurarcy(predictions: torch.Tensor, labels: torch.Tensor):
    """Pytorch tensor implementation of `accuracy` computation."""
    acc = torch.sum(labels == predictions) / len(labels)
    return {"accuracy": acc.item()}


def accurarcy_on_negatives(predictions: torch.Tensor, labels: torch.Tensor, truth_label: int = 1):
    """Pytorch tensor implementation of `accuracy_on_negatives` computation.

    That is, it computes accuracy only w.r.t. negative samples (with `label != truth_label`).
    """
    neg_acc = torch.sum((labels == predictions) & (labels != truth_label)) / torch.sum(labels != truth_label)
    return {"accuracy_on_negatives": neg_acc.item()}


def evaluate_by_threshold(
    scores: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
    truth_label: int = 1,
    smaller_scores_better: bool = False,
):
    r"""Compute evaluation metrics (`Precision`, `Recall`, `F1`, `Accurarcy`, `Accurarcy-`) based on the threshold.

    Args:
        scores (torch.Tensor): Prediction scores.
        labels (torch.Tensor): Reference labels.
        threshold (float): Threshold that splits the positive and negative predictions.
        truth_label (int): Specify which label represents the truth. Defaults to `1`.
        smaller_scores_better (bool): Specify if smaller than threshold indicates positive or not. Defaults to `False`.
    """

    # thresholding
    if smaller_scores_better:
        predictions = scores <= threshold
    else:
        predictions = scores > threshold
    # compute results
    results = {
        "threshold": threshold,
        **f1_score(predictions=predictions, labels=labels, truth_label=truth_label),
        **accurarcy(predictions=predictions, labels=labels),
        **accurarcy_on_negatives(predictions=predictions, labels=labels, truth_label=truth_label),
    }
    return results


def grid_search(
    scores: torch.Tensor,
    labels: torch.Tensor,
    threshold_granularity: int = 100,
    truth_label: int = 1,
    smaller_scores_better: bool = False,
    primary_metric: str = "f1",
    best_primary_metric_value: float = -math.inf,
    preformatted_best_results: dict = {},
):
    """Grid search the best scoring threshold.

    Args:
        scores (torch.Tensor): Prediction scores.
        labels (torch.Tensor): Reference labels.
        threshold_granularity (int, optional): A score scaling factor to determine the granularity of grid search. Defaults to `100`.
        truth_label (int): Specify which label represents the truth. Defaults to `1`.
        smaller_scores_better (bool): Specify if smaller than threshold indicates positive or not. Defaults to `False`.
        primary_metric (str, optional): The primary evaluation metric to determine the grid search result. Defaults to `"F1"`.
        best_primary_metric_value (Optional[float], optional): Best previous primary metric value. Defaults to `-math.inf`.
        preformatted_best_results (dict, optional): Preformatted best results dictionary. Defaults to `{}`.
    """

    best_results = None

    # grid search start and end are confined by the prediction scores
    start = int(scores.min() * threshold_granularity)
    end = int(scores.max() * threshold_granularity)

    # grid search to update the best results
    for threshold in tqdm(range(start, end), desc="Thresholding"):
        threshold = threshold / threshold_granularity
        results = evaluate_by_threshold(
            scores=scores,
            labels=labels,
            threshold=threshold,
            truth_label=truth_label,
            smaller_scores_better=smaller_scores_better,
        )
        if results[primary_metric] >= best_primary_metric_value:
            best_results = preformatted_best_results
            best_results.update(results)
            best_primary_metric_value = results[primary_metric]

    return best_results
