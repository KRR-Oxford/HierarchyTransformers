import torch


def threshold_evaluate(
    scores: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
    smaller_the_better: bool = True,
    truth_label: int = 1,
):
    r"""Evaluate Precision, Recall, F1, and Accurarcy based on the threshold.

    Args:
        scores (torch.Tensor): resulting scores.
        labels (torch.Tensor): positive: `labels==1`; negative: `labels==0`.
        threshold (float): threshold that splits the positive and negative predictions.
        smaller_the_better (bool): smaller than threshold indicates positive or not.
        truth_label (int): label that indicates a ground truth.
    """
    if smaller_the_better:
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
