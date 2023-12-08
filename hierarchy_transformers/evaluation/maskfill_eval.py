import torch
from deeponto.utils import save_file
from tqdm.auto import tqdm
from transformers import pipeline
from .eval_functions import threshold_evaluate


class MaskFillEvaluator:
    """Evaluator for pre-trained language models.
    
    Hierarchy encoding is evaluated based on the mask filling scores on
    the binary ("yes", "no") question answering template.
    """

    def __init__(self, pretrained: str, device: torch.device):
        self.device = device
        self.pipeline = pipeline("fill-mask", pretrained, device=self.device)
        self.template = lambda child, parent: f"Question: Is {child} a {parent}? Answer: <mask>."

    def predict(self, examples: list, batch_size: int):
        labels = [example.label for example in examples]
        labels = torch.tensor(labels)

        scores = []
        for result in tqdm(
            self.pipeline(self.pipeline_data(examples), batch_size=batch_size, top_k=10), total=len(examples)
        ):
            pos_score = 0.0
            neg_score = 0.0
            for pred in result:
                if pred["token_str"].strip().lower() == "yes":
                    pos_score += pred["score"]
                elif pred["token_str"].strip().lower() == "no":
                    neg_score += pred["score"]
            # use normalised positive score as final score
            scores.append(torch.tensor([pos_score, neg_score]))

        return torch.stack(scores).softmax(dim=-1).T[0], labels

    def pipeline_data(self, examples: list):
        for example in examples:
            child, parent = example.texts
            yield self.template(child, parent)

    def __call__(
        self,
        val_examples: list,
        test_examples: list,
        output_path: str,
        eval_batch_size: int = 256,
        granuality: int = 100,
    ):
        # validation
        val_scores, val_labels = self.predict(val_examples, eval_batch_size)
        torch.save(val_scores, f"{output_path}/maskfill_val_scores.pt")
        torch.save(val_labels, f"{output_path}/maskfill_val_labels.pt")

        best_val_f1 = -1.0
        best_val_results = None

        start = int(val_scores.min() * granuality)
        end = int(val_scores.max() * granuality)
        for threshold in tqdm(range(start, end), desc="Thresholding"):
            threshold = threshold / granuality
            results = threshold_evaluate(val_scores, val_labels, threshold, False)
            if results["F1"] >= best_val_f1:
                best_val_f1 = results["F1"]
                best_val_results = results
        save_file(best_val_results, f"{output_path}/maskfill_val_results.json")

        # testing
        test_scores, test_labels = self.predict(test_examples, eval_batch_size)
        test_results = threshold_evaluate(test_scores, test_labels, best_val_results["threshold"], False)

        torch.save(test_scores, f"{output_path}/maskfill_test_scores.pt")
        torch.save(test_labels, f"{output_path}/maskfill_test_labels.pt")
        save_file(test_results, f"{output_path}/maskfill_test_results.json")
