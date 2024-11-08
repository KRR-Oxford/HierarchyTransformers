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

from geoopt.manifolds import PoincareBall
import torch
from deeponto.utils import save_file
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

from ..models import HierarchyTransformer
from .hierarchy_eval import HierarchyEvaluator


class HierarchyTransformerEvaluator(HierarchyEvaluator):
    """
    Evaluator for hierarchy transformer encoders (HiTs).

    Hierarchy encoding is evaluated based on the hyperbolic distances between entity
    embeddings and hyperbolic norms of entity embeddings in the Poincare ball of
    radius 1/sqrt(d) where d is the embedding dimension.
    """

    def __init__(
        self,
        device: torch.device,
        eval_batch_size: int,
        val_examples: list,
        test_examples: Optional[list] = None,
        train_examples: Optional[list] = None,
    ):
        super().__init__()

        self.device = device
        self.eval_batch_size = eval_batch_size
        self.val_examples = val_examples
        self.test_examples = test_examples
        self.train_examples = train_examples

    @staticmethod
    def encode(model: HierarchyTransformer, examples: list, eval_batch_size: int):
        """
        Encode input examples with a given HiT model.
        """
        child_embeds = model.encode(
            sentences=[x.texts[0] for x in examples], 
            batch_size=eval_batch_size, 
            convert_to_tensor=True,
        )
        parent_embeds = model.encode(
            sentences=[x.texts[1] for x in examples], 
            batch_size=eval_batch_size, 
            convert_to_tensor=True
        )
        labels = torch.tensor([x.label for x in examples]).to(child_embeds.device)
        dists = model.manifold.dist(child_embeds, parent_embeds)
        child_norms = model.manifold.dist0(child_embeds)
        parent_norms = model.manifold.dist0(parent_embeds)
        return torch.stack([labels, dists, child_norms, parent_norms]).T

    @classmethod
    def score(cls, result_mat: torch.Tensor, centri_score_weight: float):
        """
        The empirical scoring function for using HiT embeddings to predict subsumptions.

        The scores are "lower-the-better".

        NOTE: In the [paper](https://arxiv.org/abs/2401.11374), the `-` operator is appended to this function to make it "higher-the-better".
        """
        scores = result_mat[:, 1] + centri_score_weight * (result_mat[:, 3] - result_mat[:, 2])
        labels = result_mat[:, 0]
        return scores, labels

    @classmethod
    def search_best_threshold(cls, result_mat: torch.Tensor, threshold_granularity: int = 100):
        """
        Grid search the best scoring threshold.
        """
        best_f1 = -1.0
        best_results = None
        is_updated = True

        for centri_score_weight in range(50):
            # early stop if increasing the centri score weight does not help
            if not is_updated:
                break
            is_updated = False

            centri_score_weight = (centri_score_weight + 0.1) / 10
            scores, labels = cls.score(result_mat, centri_score_weight)
            cur_best_results = super().search_best_threshold(
                scores,
                labels,
                threshold_granularity,
                determined_metric="F1",
                best_determined_metric_value=best_f1,
                preformatted_best_results={"centri_score_weight": centri_score_weight},
            )
            if cur_best_results:
                best_results = cur_best_results
                best_f1 = best_results["F1"]
                is_updated = True

        return best_results

    def inference(
        self,
        model: HierarchyTransformer,
        examples: list,
        best_val_centri_score_weight: float = None,
        best_val_threshold: float = None,
    ):
        """
        Inference function.
        """
        result_mat = self.encode(model, examples, self.eval_batch_size)
        if not best_val_threshold or not best_val_centri_score_weight:
            eval_results = self.search_best_threshold(result_mat, 100)
        else:
            eval_scores, eval_labels = self.score(result_mat, best_val_centri_score_weight)
            eval_results = self.evaluate_by_threshold(eval_scores, eval_labels, best_val_threshold)
            eval_results = {
                "centri_score_weight": best_val_centri_score_weight,
                **eval_results,
            }

        return result_mat, eval_results

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        """
        This is called to evaluate the model during the training procedure. It returns a score for the evaluation with a higher score indicating a better result.
        """

        # set to eval mode
        Path(f"{output_path}/epoch={epoch}.step={steps}").mkdir(parents=True, exist_ok=True)
        # model.save(f"{output_path}/epoch={epoch}.step={steps}")

        if self.train_examples:
            logger.info("Evaluate on train examples...")
            train_result_mat, train_results = self.inference(model, self.train_examples)
            torch.save(
                train_result_mat,
                f"{output_path}/epoch={epoch}.step={steps}/train_result_mat.pt",
            )
            save_file(
                train_results,
                f"{output_path}/epoch={epoch}.step={steps}/train_results.json",
            )

        logger.info("Evaluate on val examples...")
        val_result_mat, val_results = self.inference(model, self.val_examples)
        torch.save(
            val_result_mat,
            f"{output_path}/epoch={epoch}.step={steps}/val_result_mat.pt",
        )
        save_file(val_results, f"{output_path}/epoch={epoch}.step={steps}/val_results.json")

        if self.test_examples:
            logger.info("Evaluate on test examples using best val threshold...")
            test_result_mat, test_results = self.inference(
                model,
                self.test_examples,
                val_results["centri_score_weight"],
                val_results["threshold"],
            )
            torch.save(
                test_result_mat,
                f"{output_path}/epoch={epoch}.step={steps}/test_result_mat.pt",
            )
            save_file(
                test_results,
                f"{output_path}/epoch={epoch}.step={steps}/test_results.json",
            )

        return val_results["F1"]
