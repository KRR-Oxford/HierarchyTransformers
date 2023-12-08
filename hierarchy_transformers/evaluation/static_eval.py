from typing import Optional
import torch
from torch.utils.data import DataLoader
from deeponto.utils import save_file

from .hierarchy_eval import HierarchyEvaluator


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
    ):
        super().__init__()

        self.model = torch.load(model_path)
        self.device = device
        self.model.to(self.device)

        self.eval_batch_size = eval_batch_size
        self.val_examples = val_examples
        self.test_examples = test_examples

    def score(self, subject: torch.Tensor, objects: torch.Tensor, norm_score_weight: float = 1000.0):
        dists = self.model.manifold.dist(subject, objects)
        subject_norms = subject.norm(dim=-1)
        objects_norms = objects.norm(dim=-1)
        return (1 + norm_score_weight * (objects_norms - subject_norms)) * dists

    def inference(self, examples: list):
        """WARNING: this function is highly customised to our hierarchy datasets
        where 1 positive sample corresponds to 10 negatives."""
        num_negatives = len(examples[0]) - 2  # each example is formatted as [child, parent, *negatives]
        eval_scores = []
        dataloader = DataLoader(examples, shuffle=False, batch_size=self.eval_batch_size)
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
        val_scores, val_labels = self.inference(self.val_examples)
        best_val_results = self.search_best_threshold(val_scores, val_labels, threshold_granularity=1)
        save_file(best_val_results, f"{output_path}/val_results.json")

        if self.test_examples:
            test_scores, test_labels = self.inference(self.test_examples)
            test_results = self.evaluate_by_threshold(test_scores, test_labels, best_val_results["threshold"])
            save_file(test_results, f"{output_path}/test_results.json")
