import torch
from torch.utils.data import DataLoader
from deeponto.utils import save_file
from .hierarchy_eval import evaluate_by_threshold


class StaticPoincareEvaluator:
    """Evaluator for the static Poincare embedding model.
    
    Hierarchy encoding is evaluated based on hyperbolic distances between
    entity embeddings in the unit Poincare ball. 
    """

    def __init__(self, model_path: str, device: torch.device):
        self.model = torch.load(model_path)
        self.device = device
        self.model.to(self.device)

    def __call__(self, val_dataloader: DataLoader, test_dataloader: DataLoader, output_path: str):
        val_scores = []
        for batch in val_dataloader:
            batch
            subject, objects = self.model(batch)
            dists = self.model.manifold.dist(subject, objects)
            subject_norms = subject.norm(dim=-1)
            objects_norms = objects.norm(dim=-1)
            scores = (1 + 1000 * (objects_norms - subject_norms)) * dists
            val_scores.append(scores.reshape((-1,)))
        val_scores = torch.concat(val_scores, dim=0)

        best_f1 = -1.0
        best_val_results = None
        val_labels = torch.tensor(([1] + [0] * 10) * (int(len(val_scores) / 11))).to(self.device)
        for threshold in range(int(val_scores.min()), int(val_scores.max())):
            # scores = val_results <= threshold
            results = evaluate_by_threshold(val_scores, val_labels, threshold)
            if results["F1"] >= best_f1:
                best_f1 = results["F1"]
                best_val_results = results
        save_file(best_val_results, f"{output_path}/val_results.json")

        test_scores = []
        for batch in test_dataloader:
            subject, objects = self.model(batch)
            dists = self.model.manifold.dist(subject, objects)
            subject_norms = subject.norm(dim=-1)
            objects_norms = objects.norm(dim=-1)
            scores = (1 + 1000 * (objects_norms - subject_norms)) * dists
            test_scores.append(scores.reshape((-1,)))
        test_scores = torch.concat(test_scores, dim=0)
        test_labels = torch.tensor(([1] + [0] * 10) * (int(len(test_scores) / 11))).to(self.device)
        test_results = evaluate_by_threshold(test_scores, test_labels, best_val_results["threshold"])
        save_file(test_results, f"{output_path}/test_results.json")
