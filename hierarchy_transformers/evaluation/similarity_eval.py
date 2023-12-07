import torch
from deeponto.utils import save_file
from tqdm.auto import tqdm

from .eval_metrics import threshold_evaluate
from ..model import load_sentence_transformer


class SentenceSimilarityEvaluator:
    def __init__(self, pretrained: str, device: torch.device):
        self.device = device
        self.model = load_sentence_transformer(pretrained, device)
        self.model.to(self.device)
        self.template = lambda child, parent: f"{child} is a {parent}."
        self.mask_token = self.model.tokenizer.mask_token

    def predict(self, batch):
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

        return scores, labels


    @staticmethod
    def get_batches(lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i : i + batch_size]

    def __call__(self, val_examples: list, test_examples: list, output_path: str, eval_batch_size: int = 256, granuality: int = 100):
        # validation
        val_scores = []
        val_labels = []
        for batch in tqdm(list(self.get_batches(val_examples, eval_batch_size)), desc="Validating"):
            cur_scores, cur_labels = self.predict(batch)
            val_scores.append(cur_scores)
            val_labels.append(cur_labels)
        val_scores = torch.concatenate(val_scores, dim=0)
        val_labels = torch.concatenate(val_labels, dim=0)
        torch.save(val_scores, f"{output_path}/similarity_val_scores.pt")
        torch.save(val_labels, f"{output_path}/similarity_val_labels.pt")

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
        save_file(best_val_results, f"{output_path}/similarity_val_results.json")

        # testing
        test_scores = []
        test_labels = []
        for batch in tqdm(list(self.get_batches(test_examples, eval_batch_size)), desc="Testing"):
            cur_scores, cur_labels = self.predict(batch)
            test_scores.append(cur_scores)
            test_labels.append(cur_labels)
        test_scores = torch.concatenate(test_scores, dim=0)
        test_labels = torch.concatenate(test_labels, dim=0)
        test_results = threshold_evaluate(test_scores, test_labels, best_val_results["threshold"], False)

        torch.save(test_scores, f"{output_path}/similarity_test_scores.pt")
        torch.save(test_labels, f"{output_path}/similarity_test_labels.pt")
        save_file(test_results, f"{output_path}/similarity_test_results.json")
