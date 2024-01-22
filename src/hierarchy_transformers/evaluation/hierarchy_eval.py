from __future__ import annotations
import torch
from typing import Optional, Tuple
from tqdm.auto import tqdm
import math


class HierarchyEvaluator:
    """Base evaluator for evaluating hierarchy encoding in models."""

    def __init__(self):
        pass

    def inference(self):
        raise NotImplementedError

    @staticmethod
    def evaluate_by_threshold(
        scores: torch.Tensor,
        labels: torch.Tensor,
        threshold: float,
        smaller_scores_better: bool = True,
        truth_label: int = 1,
    ):
        r"""Evaluate Precision, Recall, F1, and Accurarcy based on the threshold.

        Args:
            scores (torch.Tensor): resulting scores.
            labels (torch.Tensor): positive: `labels==1`; negative: `labels==0`.
            threshold (float): threshold that splits the positive and negative predictions.
            smaller_scores_better (bool): smaller than threshold indicates positive or not.
            truth_label (int): label that indicates a ground truth.
        """
        if smaller_scores_better:
            predictions = scores <= threshold
        else:
            predictions = scores > threshold

        tp = torch.sum((labels == truth_label) & (predictions == truth_label))  # correct and positive
        fp = torch.sum((labels != truth_label) & (predictions == truth_label))  # incorrect but positive
        fn = torch.sum((labels == truth_label) & (predictions != truth_label))  # correct but negative
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        acc = torch.sum(labels == predictions) / len(labels)
        neg_acc = torch.sum((labels == predictions) & (labels != truth_label)) / torch.sum(labels != truth_label)
        return {
            "threshold": threshold,
            "P": precision.item(),
            "R": recall.item(),  # recall is ACC+
            "F1": f1.item(),
            "ACC": acc.item(),
            "ACC-": neg_acc.item(),
        }

    @staticmethod
    def search_best_threshold(
        scores: torch.Tensor,
        labels: torch.Tensor,
        threshold_granularity: int = 100,
        smaller_scores_better: bool = True,
        truth_label: int = 1,
        determined_metric: str = "F1",
        best_determined_metric_value: Optional[float] = None,
        preformatted_best_results: dict = {},
    ):
        if not best_determined_metric_value:
            best_determined_metric_value = -math.inf

        best_results = None
        start = int(scores.min() * threshold_granularity)
        end = int(scores.max() * threshold_granularity)
        for threshold in tqdm(range(start, end), desc=f"Thresholding"):
            threshold = threshold / threshold_granularity
            results = HierarchyEvaluator.evaluate_by_threshold(
                scores, labels, threshold, smaller_scores_better, truth_label
            )
            if results[determined_metric] >= best_determined_metric_value:
                best_results = preformatted_best_results
                best_results.update(results)
                best_determined_metric_value = results[determined_metric]

        return best_results


# make this evaluation method stand-alone
evaluate_by_threshold = HierarchyEvaluator.evaluate_by_threshold
search_best_threshold = HierarchyEvaluator.search_best_threshold
