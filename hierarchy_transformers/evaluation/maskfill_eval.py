import torch
from deeponto.utils import save_file
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
from .eval_metrics import threshold_evaluate


class MaskFillEvaluator:
    def __init__(self, pretrained: str, device: torch.device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.model = AutoModelForMaskedLM.from_pretrained(pretrained)
        self.model.to(self.device)

    def mask_fill(self, anchored_batch):
        masked_texts = []
        # anchored_example: [child, parent, *negative_parents]
        for anchored_example in anchored_batch:
            masked_texts.append(f"{anchored_example[0]} is a {self.tokenizer.mask_token}.")
        with torch.inference_mode():
            masked_inputs = self.tokenizer(masked_texts, return_tensors="pt", padding=True).to(self.device)
            logits = self.model(**masked_inputs).logits
            mask_token_logits = logits[masked_inputs["input_ids"] == self.tokenizer.mask_token_id]

        scores = []
        labels = []
        for i, anchored_example in enumerate(anchored_batch):
            cur_logit = mask_token_logits[i]
            for j, p in enumerate(anchored_example[1:]):
                p_tokens = self.tokenizer.encode(p, add_special_tokens=False)
                scores.append(cur_logit[p_tokens].mean().item())
                labels.append(int(j == 0))

        return torch.tensor(scores), torch.tensor(labels)

    @staticmethod
    def get_batches(lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i : i + batch_size]

    def __call__(self, val_examples: list, test_examples: list, output_path: str, granuality: int = 100):

        # validation
        val_scores = []
        val_labels = []
        for batch in tqdm(list(self.get_batches(val_examples, 100)), desc="Validating"):
            cur_scores, cur_labels = self.mask_fill(batch)
            val_scores.append(cur_scores)
            val_labels.append(cur_labels)
        val_scores = torch.concatenate(val_scores, dim=0)
        val_labels = torch.concatenate(val_labels, dim=0)

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
        test_scores = []
        test_labels = []
        for batch in tqdm(list(self.get_batches(test_examples, 100)), desc="Testing"):
            cur_scores, cur_labels = self.mask_fill(batch)
            test_scores.append(cur_scores)
            test_labels.append(cur_labels)
        test_scores = torch.concatenate(test_scores, dim=0)
        test_labels = torch.concatenate(test_labels, dim=0)
        test_results = threshold_evaluate(test_scores, test_labels, best_val_results["threshold"], False)
        save_file(test_results, f"{output_path}/maskfill_test_results.json")
