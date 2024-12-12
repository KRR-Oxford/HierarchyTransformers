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
from string import Template

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator

from .metrics import evaluate_by_threshold, grid_search

logger = logging.getLogger(__name__)


class SentenceTransformerEvaluator(SentenceEvaluator):
    """Evaluating sBERT models for predicting entity hierarchical relationships.

    The main evaluation metrics are `Precision`, `Recall`, and `F-score`, with overall accuracy (`ACC`) and accuracy on negatives (`ACC-`) additionally reported. The results are written in a `.csv`. If a result file already exists, then values are appended.

    The labels need to be `0` for unrelated pairs and `1` for related pairs.

    Args:
        child_entities (list[str]): List of child entity names.
        parent_entities (list[str]): List of parent entity names.
        labels (list[int]): List of reference labels.
        batch_size (int): Evaluation batch size.
        truth_label (int, optional): Specify which label represents the truth. Defaults to `1`.
        template (str, optional): The probing template.
    """

    def __init__(
        self,
        child_entities: list[str],
        parent_entities: list[str],
        labels: list[int],
        batch_size: int,
        truth_label: int = 1,
        template: str = "${child} is a ${parent}.",
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
            columns=["threshold", "precision", "recall", "f1", "accuracy", "accuracy_on_negatives"]
        )
        # template for probing
        self.template = Template(template)

    def inference(self, model: SentenceTransformer):
        """The probing method of the pre-trained sBERT model. It output scores that indicate hierarchical relationships between entities."""
        sentences = []
        masked_sentences = []
        for child, parent in zip(self.child_entities, self.parent_entities):
            sentences.append(self.template.substitute(child=child, parent=parent))
            masked_sentences.append(self.template.substitute(child=child, parent=model.tokenizer.mask_token))

        sentence_embeds = model.encode(sentences=sentences, convert_to_tensor=True, show_progress_bar=True)
        masked_embeds = model.encode(sentences=masked_sentences, convert_to_tensor=True, show_progress_bar=True)

        # use the cosine similarity between masked and
        return torch.cosine_similarity(masked_embeds, sentence_embeds)

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
        best_threshold: float | None = None,
    ) -> dict[str, float]:
        """Compute the evaluation metrics for the given model.

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

        if best_threshold:
            # Testing with pre-defined hyperparameters
            logger.info(f"Evaluate on given hyperparemeters `best_threshold={best_threshold}`.")

            # Compute the scores
            scores = self.inference(model=model)

            # Compute the evaluation metrics
            best_results = evaluate_by_threshold(
                scores=scores,
                labels=torch.tensor(self.labels).to(scores.device),
                threshold=best_threshold,
                truth_label=self.truth_label,
                smaller_scores_better=False,
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
            scores = self.inference(model=model)

            # Perform grid search on hyperparameters
            cur_best_results = grid_search(
                scores=scores,
                labels=torch.tensor(self.labels).to(scores.device),
                threshold_granularity=1000,
                truth_label=self.truth_label,
                smaller_scores_better=False,
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
