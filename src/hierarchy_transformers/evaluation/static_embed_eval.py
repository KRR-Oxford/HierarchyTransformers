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

import logging
import os.path
import warnings

import pandas as pd
import torch
from torch.utils.data import DataLoader

from hierarchy_transformers.losses import HyperbolicEntailmentConeStaticLoss, PoincareEmbeddingStaticLoss
from hierarchy_transformers.models import PoincareStaticEmbedding

from .metrics import evaluate_by_threshold, grid_search

logger = logging.getLogger(__name__)


class PoincareStaticEmbeddingEvaluator:
    """Evaluating hyperbolic static embedding models ([1] and [2]) for predicting entity hierarchical relationships.

        - [1] Poincaré Embedding by [Nickel et al., NeurIPS 2017](https://arxiv.org/abs/1705.08039).
        - [2] Hyperbolic Entailment Cone by [Ganea et al., ICML 2018](https://arxiv.org/abs/1804.01882).

    both of which lie in a unit Poincaré ball. According to [2], it is better to apply the entailment cone loss in the post-training phase of a Poincaré embedding model in [1].

    The main evaluation metrics are `Precision`, `Recall`, and `F-score`, with overall accuracy (`ACC`) and accuracy on negatives (`ACC-`) additionally reported. The results are written in a `.csv`. If a result file already exists, then values are appended.

    The labels need to be `0` for unrelated pairs and `1` for related pairs.

    Args:
        examples (list[int]): List of input examples containing entity IDs. Each example is formatted as `[child_id, parent_id, *negative_ids]`.
        batch_size (int): Evaluation batch size.
        truth_label (int, optional): Specify which label represents the truth. Defaults to `1`.
    """

    def __init__(self, eval_examples: list, batch_size: int, truth_label: int = 1):
        self.examples = eval_examples
        self.batch_size = batch_size
        self.truth_label = truth_label
        # result file
        self.results = pd.DataFrame(
            columns=["threshold", "precision", "recall", "f1", "accuracy", "accuracy_on_negatives"]
        )

    def inference(
        self,
        model: PoincareStaticEmbedding,
        loss: PoincareEmbeddingStaticLoss | HyperbolicEntailmentConeStaticLoss,
        device: torch.device,
    ):
        """The probing method of the pre-trained hyperbolic static embedding models. It output scores that indicate hierarchical relationships between entities."""

        # set up scoring function according to input loss
        # NOTE: both scores are smaller the better
        if isinstance(loss, PoincareEmbeddingStaticLoss):
            # distance scoring from [Nickel et al., NeurIPS 2017]
            def score_func(subject: torch.Tensor, objects: torch.Tensor, norm_score_weight: float = 1000.0):
                dists = loss.manifold.dist(subject, objects)
                subject_norms = subject.norm(dim=-1)
                objects_norms = objects.norm(dim=-1)
                return (1 + norm_score_weight * (objects_norms - subject_norms)) * dists

        elif isinstance(loss, HyperbolicEntailmentConeStaticLoss):
            # hyperbolic entailment cone scoring from [Ganea et al., ICML 2018]
            score_func = lambda subject, objects: loss.energy(objects, subject)
        else:
            raise ValueError(f"Unknown loss function type: {type(loss)}.")

        # set model to eval mode
        model.eval()

        # make predictions
        dataloader = DataLoader(torch.tensor(self.examples).to(device), shuffle=False, batch_size=self.batch_size)
        num_negatives = len(self.examples[0]) - 2  # each example is formatted as [child, parent, *negatives]
        scores = []
        labels = []
        with torch.no_grad():
            for batch in dataloader:
                subject, objects = model(batch)
                cur_scores = score_func(subject, objects)
                scores.append(cur_scores.reshape((-1,)))
        scores = torch.concat(scores, dim=0)
        labels = torch.tensor(
            ([self.truth_label] + [1 - self.truth_label] * num_negatives) * (int(len(scores) / (1 + num_negatives)))
        ).to(scores.device)
        assert len(labels) == len(scores)

        return scores, labels

    def __call__(
        self,
        model: PoincareStaticEmbedding,
        loss: PoincareEmbeddingStaticLoss | HyperbolicEntailmentConeStaticLoss,
        device: torch.device,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
        best_threshold: float | None = None,
    ):
        """Compute the evaluation metrics for the given model.

        Args:
            model (HierarchyTransformer): The model to evaluate.
            loss (Union[PoincareEmbeddingStaticLoss, HyperbolicEntailmentConeStaticLoss]): The training loss function decides which scoring function to be used.
            device (torch.device): The torch device used for evaluation.
            output_path (str, optional): Path to save the evaluation results `.csv` file. Defaults to `None`.
            epoch (int, optional): The epoch number. Defaults to `-1`.
            steps (int, optional): The number of steps. Defaults to `-1`.
            best_threshold (float, optional): The best overall threshold searched on a validation set (used for testing). Defaults to `None`.

        Returns:
            Dict[str, float]: A dictionary containing the evaluation metrics.
        """

        if best_threshold:
            # Testing with pre-defined hyperparameters
            logger.info(f"Evaluate on given hyperparemeters `best_threshold={best_threshold}`.")

            # Compute the scores
            scores, labels = self.inference(model=model, loss=loss, device=device)

            # Compute the evaluation metrics
            best_results = evaluate_by_threshold(
                scores=scores,
                labels=labels,
                threshold=best_threshold,
                truth_label=self.truth_label,
                smaller_scores_better=True,
            )

            # log the results
            if os.path.exists(os.path.join(output_path, "results.tsv")):
                self.results = pd.read_csv(os.path.join(output_path, "results.tsv"), sep="\t", index_col=0)
            else:
                warnings.warn("No previous `results.tsv` detected.")
            self.results.loc["testing"] = best_results
        else:
            # Validation with no pre-defined hyerparameters
            logger.info("Evaluate with grid search on hyperparameters `best_threshold` (overall threshold)")
            best_f1 = -1.0
            best_results = None

            # Compute the scores
            scores, labels = self.inference(model=model, loss=loss, device=device)

            # Perform grid search on hyperparameters
            cur_best_results = grid_search(
                scores=scores,
                labels=labels,
                threshold_granularity=1 if isinstance(loss, PoincareEmbeddingStaticLoss) else 100,
                truth_label=self.truth_label,
                smaller_scores_better=True,
                primary_metric="f1",
                best_primary_metric_value=best_f1,
                preformatted_best_results={},
            )
            if cur_best_results:
                best_results = cur_best_results
                best_f1 = best_results["f1"]

            idx = f"epoch={epoch}" if epoch != "validation" else epoch
            self.results.loc[idx] = best_results

        self.results.to_csv(os.path.join(output_path, "results.tsv"), sep="\t")

        logger.info(f"Eval results: {best_results}")

        return best_results
