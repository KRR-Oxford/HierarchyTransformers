from typing import Optional
import torch
from deeponto.utils import save_file
from tqdm.auto import tqdm
from transformers import pipeline
from .hierarchy_eval import HierarchyEvaluator
from ..models import HierarchyTransformer


class PretrainedMaskFillEvaluator(HierarchyEvaluator):
    """Evaluator for pre-trained language models.

    Hierarchy encoding is evaluated based on the mask filling scores on
    the binary ("yes", "no") question answering template.
    """

    def __init__(
        self,
        pretrained: str,
        device: torch.device,
        eval_batch_size: int,
        val_examples: list,
        test_examples: Optional[list] = None,
    ):
        super().__init__()
        self.device = device
        self.pipeline = pipeline("fill-mask", pretrained, device=self.device)
        self.template = lambda child, parent: f"Question: Is {child} a {parent}? Answer: <mask>."

        self.eval_batch_size = eval_batch_size
        self.val_examples = val_examples
        self.test_examples = test_examples

    def inference(self, examples: list):
        labels = [example.label for example in examples]
        labels = torch.tensor(labels)

        scores = []
        for result in tqdm(
            self.pipeline(self.pipeline_data(examples), batch_size=self.eval_batch_size, top_k=10),
            total=len(examples),
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

    def __call__(self, output_path: str):
        # validation
        val_scores, val_labels = self.inference(self.val_examples)
        torch.save(val_scores, f"{output_path}/maskfill_val_scores.pt")
        torch.save(val_labels, f"{output_path}/maskfill_val_labels.pt")

        best_val_results = self.search_best_threshold(val_scores, val_labels, 1000, False)
        save_file(best_val_results, f"{output_path}/maskfill_val_results.json")

        # testing
        if self.test_examples:
            test_scores, test_labels = self.inference(self.test_examples)
            test_results = self.evaluate_by_threshold(test_scores, test_labels, best_val_results["threshold"], False)

            torch.save(test_scores, f"{output_path}/maskfill_test_scores.pt")
            torch.save(test_labels, f"{output_path}/maskfill_test_labels.pt")
            save_file(test_results, f"{output_path}/maskfill_test_results.json")


class PretrainedSentenceSimilarityEvaluator(HierarchyEvaluator):
    """Evaluator for pre-trained language models.

    Hierarchy encoding based on the similarities between the masked "is-a" sentences
    and the reference "is-a" sentences.
    """

    def __init__(
        self,
        pretrained: str,
        device: torch.device,
        eval_batch_size: int,
        val_examples: list,
        test_examples: Optional[list] = None,
    ):
        self.device = device
        self.model = HierarchyTransformer.load_pretrained(pretrained, device)
        self.model.to(self.device)
        self.template = lambda child, parent: f"{child} is a {parent}."
        self.mask_token = self.model.tokenizer.mask_token

        self.eval_batch_size = eval_batch_size
        self.val_examples = val_examples
        self.test_examples = test_examples

    def inference(self, examples):
        eval_scores = []
        eval_labels = []
        for batch in tqdm(list(self.get_batches(examples, self.eval_batch_size)), desc="Inference"):
            masked_texts = []
            ref_texts = []
            labels = []
            for example in batch:
                child, parent = example.texts
                masked_texts.append(self.template(child, self.mask_token))
                ref_texts.append(self.template(child, parent))
                labels.append(example.label)
            labels = torch.tensor(labels).to(self.device)

            masked_embeds = self.model.encode(masked_texts, convert_to_tensor=True, show_progress_bar=False)
            ref_embeds = self.model.encode(ref_texts, convert_to_tensor=True, show_progress_bar=False)
            scores = torch.cosine_similarity(masked_embeds, ref_embeds)

            eval_scores.append(scores)
            eval_labels.append(labels)

        eval_scores = torch.concatenate(eval_scores, dim=0)
        eval_labels = torch.concatenate(eval_labels, dim=0)

        return eval_scores, eval_labels

    @staticmethod
    def get_batches(lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i : i + batch_size]

    def __call__(self, output_path: str):
        # validation
        val_scores, val_labels = self.inference(self.val_examples)
        torch.save(val_scores, f"{output_path}/similarity_val_scores.pt")
        torch.save(val_labels, f"{output_path}/similarity_val_labels.pt")

        best_val_results = self.search_best_threshold(val_scores, val_labels, 1000, False)
        save_file(best_val_results, f"{output_path}/similarity_val_results.json")

        # testing
        if self.test_examples:
            test_scores, test_labels = self.inference(self.test_examples)
            test_results = self.evaluate_by_threshold(test_scores, test_labels, best_val_results["threshold"], False)
            torch.save(test_scores, f"{output_path}/similarity_test_scores.pt")
            torch.save(test_labels, f"{output_path}/similarity_test_labels.pt")
            save_file(test_results, f"{output_path}/similarity_test_results.json")
