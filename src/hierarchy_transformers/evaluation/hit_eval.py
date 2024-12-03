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

from typing import Optional
import os.path, warnings, logging
import pandas as pd
import torch

from sentence_transformers.evaluation import SentenceEvaluator
from hierarchy_transformers import HierarchyTransformer
from .metrics import evaluate_by_threshold, grid_search

logger = logging.getLogger(__name__)


class HierarchyTransformerEvaluator(SentenceEvaluator):
    """
    Evaluating HiT models for predicting entity hierarchical relationships.

    The main evaluation metrics are `Precision`, `Recall`, and `F-score`, with overall accuracy (`ACC`) and accuracy on negatives (`ACC-`) additionally reported. The results are written in a `.csv`. If a result file already exists, then values are appended.

    The labels need to be `0` for unrelated pairs and `1` for related pairs.

    Args:
        child_entities (list[str]): List of child entity names.
        parent_entities (list[str]): List of parent entity names.
        labels (list[int]): List of reference labels.
        batch_size (int): Evaluation batch size.
        truth_label (int, optional): Specify which label represents the truth. Defaults to `1`.
    """

    def __init__(
        self,
        child_entities: list[str],
        parent_entities: list[str],
        labels: list[int],
        batch_size: int,
        truth_label: int = 1,
    ):
        super().__init__()
        # set primary metric for model selection
        self.primary_metric = "f1"
        # input evaluation examples
        self.child_entities = child_entities
        self.parent_entities = parent_entities
        self.labels = labels
        # eval batch size
        self.batch_size = batch_size
        # truth reference label
        self.truth_label = truth_label
        # result file
        self.results = pd.DataFrame(
            columns=["centri_weight", "threshold", "precision", "recall", "f1", "accuracy", "accuracy_on_negatives"]
        )
        # NOTE: static transformation staticmethod to do

    def inference(
        self,
        model: HierarchyTransformer,
        centri_weight: float,
        child_embeds: Optional[torch.Tensor] = None,
        parent_embeds: Optional[torch.Tensor] = None,
    ):
        """
        The default probing method of the HiT model. It output scores that indicate hierarchical relationships between entities.

        Optional `child_embeds` and `parent_embeds` are used to save time from repetitive encoding.
        """
        if child_embeds is None:
            logger.info("Encode child entities.")
            child_embeds = model.encode(
                sentences=self.child_entities, batch_size=self.batch_size, convert_to_tensor=True
            )
        if parent_embeds is None:
            logger.info("Encode parent entities.")
            parent_embeds = model.encode(
                sentences=self.parent_entities, batch_size=self.batch_size, convert_to_tensor=True
            )
        dists = model.manifold.dist(child_embeds, parent_embeds)
        child_norms = model.manifold.dist0(child_embeds)
        parent_norms = model.manifold.dist0(parent_embeds)
        return -(dists + centri_weight * (parent_norms - child_norms))

    def __call__(
        self,
        model: HierarchyTransformer,
        output_path: Optional[str] = None,
        epoch: int = -1,
        steps: int = -1,
        best_centri_weight: Optional[float] = None,
        best_threshold: Optional[float] = None,
    ) -> dict[str, float]:
        """
        Compute the evaluation metrics for the given model.

        Args:
            model (HierarchyTransformer): The model to evaluate.
            output_path (str, optional): Path to save the evaluation results `.csv` file. Defaults to `None`.
            epoch (int, optional): The epoch number. Defaults to `-1`.
            steps (int, optional): The number of steps. Defaults to `-1`.
            best_centri_weight (float, optional): The best centripetal score weight searched on a validation set (used for testing). Defaults to `None`.
            best_threshold (float, optional): The best overall threshold searched on a validation set (used for testing). Defaults to `None`.
            
        Returns:
            Dict[str, float]: A dictionary containing the evaluation metrics.
        """

        # best thresholds and metric searched on validation sets
        assert type(best_centri_weight) == type(
            best_threshold
        ), "Inconsistent types of hyperparameters 'best_centri_weight' (centripetal score weight) and 'best_threshold' (overall threshold)"

        logger.info("Encode child entities.")
        child_embeds = model.encode(sentences=self.child_entities, batch_size=self.batch_size, convert_to_tensor=True)
        logger.info("Encode parent entities.")
        parent_embeds = model.encode(sentences=self.parent_entities, batch_size=self.batch_size, convert_to_tensor=True)

        if best_centri_weight and best_threshold:

            # Testing with pre-defined hyperparameters
            logger.info(
                f"Evaluate on given hyperparemeters `best_centri_weight={best_centri_weight}` (centripetal score weight) and `best_threshold={best_threshold}` (overall threshold)."
            )

            # Compute the scores
            scores = self.inference(
                model=model,
                centri_weight=best_centri_weight,
                child_embeds=child_embeds,
                parent_embeds=parent_embeds,
            )

            # Compute the evaluation metrics
            best_results = evaluate_by_threshold(
                scores=scores,
                labels=torch.tensor(self.labels).to(scores.device),
                threshold=best_threshold,
                truth_label=self.truth_label,
                smaller_scores_better=False,
            )
            best_results["centri_weight"] = best_centri_weight
            try:
                self.results = pd.read_csv(os.path.join(output_path, "results.tsv"), sep="\t")
            except:
                warnings.warn("No previous `results.tsv` detected.")
            self.results.loc["testing"] = best_results
        else:
            # Validation with no pre-defined hyerparameters
            logger.info(
                f"Evaluate with grid search on hyperparameters `best_centri_weight` (centripetal score weight) and `best_threshold` (overall threshold)."
            )
            best_f1 = -1.0
            best_results = None
            is_updated = True

            for centri_weight in range(50):
                # early stop if increasing the centri score weight does not help
                if not is_updated:
                    break
                is_updated = False

                centri_weight /= 10

                # Compute the scores
                scores = self.inference(
                    model=model, centri_weight=centri_weight, child_embeds=child_embeds, parent_embeds=parent_embeds
                )

                # Perform grid search on hyperparameters
                cur_best_results = grid_search(
                    scores=scores,
                    labels=torch.tensor(self.labels).to(scores.device),
                    threshold_granularity=100,
                    truth_label=self.truth_label,
                    smaller_scores_better=False,
                    primary_metric="f1",
                    best_primary_metric_value=best_f1,
                    preformatted_best_results={"centri_weight": centri_weight},
                )
                if cur_best_results:
                    best_results = cur_best_results
                    best_f1 = best_results["f1"]
                    is_updated = True

            idx = f"epoch={epoch}"
            self.results.loc[idx] = best_results

        self.results.to_csv(os.path.join(output_path, "results.tsv"), sep="\t")

        logger.info(f"Eval results: {best_results}")

        return best_results
