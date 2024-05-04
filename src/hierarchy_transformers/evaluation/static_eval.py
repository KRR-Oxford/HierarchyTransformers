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

from typing import Optional
import torch
from torch.utils.data import DataLoader
from deeponto.utils import save_file

from .hierarchy_eval import HierarchyEvaluator
from ..losses import EntailmentConeConstrastiveLoss


class StaticPoincareEvaluator(HierarchyEvaluator):
    """Evaluator for the static Poincare embedding model.

    Hierarchy encoding is evaluated based on hyperbolic distances between
    entity embeddings in the unit Poincare ball.
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        val_examples: list,
        eval_batch_size: int,
        test_examples: Optional[list] = None,
        train_examples: Optional[list] = None,
        apply_entailment_cone: bool = False,
    ):
        super().__init__()

        self.model = torch.load(model_path)
        self.device = device
        self.model.to(self.device)

        self.eval_batch_size = eval_batch_size
        self.val_examples = val_examples
        self.test_examples = test_examples
        self.train_examples = train_examples

        self.score = self.dist_score
        self.apply_entailment_cone = apply_entailment_cone
        if self.apply_entailment_cone:
            self.eloss = EntailmentConeConstrastiveLoss(self.model.manifold)
            self.score = self.cone_score

    def dist_score(self, subject: torch.Tensor, objects: torch.Tensor, norm_score_weight: float = 1000.0):
        dists = self.model.manifold.dist(subject, objects)
        subject_norms = subject.norm(dim=-1)
        objects_norms = objects.norm(dim=-1)
        return (1 + norm_score_weight * (objects_norms - subject_norms)) * dists

    def cone_score(self, subject: torch.Tensor, objects: torch.Tensor):
        return self.eloss.energy(objects, subject)

    def inference(self, examples: list):
        """WARNING: this function is highly customised to our hierarchy datasets
        where 1 positive sample corresponds to 10 negatives."""
        self.model.eval()
        num_negatives = len(examples[0]) - 2  # each example is formatted as [child, parent, *negatives]
        eval_scores = []
        dataloader = DataLoader(
            torch.tensor(examples).to(self.device),
            shuffle=False,
            batch_size=self.eval_batch_size,
        )
        with torch.no_grad():
            for batch in dataloader:
                subject, objects = self.model(batch)
                cur_scores = self.score(subject, objects)
                eval_scores.append(cur_scores.reshape((-1,)))
            eval_scores = torch.concat(eval_scores, dim=0)
            eval_labels = torch.tensor(([1] + [0] * num_negatives) * (int(len(eval_scores) / (1 + num_negatives)))).to(
                self.device
            )
        assert len(eval_labels) == len(eval_scores)
        return eval_scores, eval_labels

    def __call__(self, output_path: str):
        
        threshold_granularity = 1
        if self.apply_entailment_cone:
            threshold_granularity = 100
        
        if self.train_examples:
            train_scores, train_labels = self.inference(self.train_examples)
            best_train_results = self.search_best_threshold(train_scores, train_labels, threshold_granularity=threshold_granularity)
            save_file(best_train_results, f"{output_path}/train_results.json")

        val_scores, val_labels = self.inference(self.val_examples)
        best_val_results = self.search_best_threshold(val_scores, val_labels, threshold_granularity=threshold_granularity)
        save_file(best_val_results, f"{output_path}/val_results.json")

        if self.test_examples:
            test_scores, test_labels = self.inference(self.test_examples)
            test_results = self.evaluate_by_threshold(test_scores, test_labels, best_val_results["threshold"])
            save_file(test_results, f"{output_path}/test_results.json")
