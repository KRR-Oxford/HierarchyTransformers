import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from deeponto.utils import save_file
from tqdm.auto import tqdm
from .eval_metrics import threshold_evaluate

class PerplexityEvaluator:
    
    def __init__(self, model_id: str, device: torch.device):
        
        self.device = device
        self.model = AutoModelForMaskedLM.from_pretrained(model_id).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        
    def pseudo_perplexity_for_parent(self, child: str, parent: str):
        """Compute pseudo perperlexity by summing log-likelihood of parent tokens.
        """
        tensor_input = self.tokenizer.encode(f"{child} is a {parent}.", return_tensors='pt').to(self.device)
        parent_tensor = self.tokenizer.encode(f"{parent}", return_tensors='pt').to(self.device)[:,1:-1]
        repeat_input = tensor_input.repeat(parent_tensor.shape[-1], 1)
        mask = tensor_input == parent_tensor.T
        masked_input = repeat_input.masked_fill(mask == True, self.tokenizer.mask_token_id)
        labels = repeat_input.masked_fill( masked_input != self.tokenizer.mask_token_id, -100)
        with torch.inference_mode():
            loss = self.model(masked_input, labels=labels).loss
        return torch.exp(loss).item()
    
    def pseudo_perplexity(self, sentence: str):
        """Compute pseudo perperlexity by summing log-likelihood of all tokens.
        """
        tensor_input = self.tokenizer.encode(sentence, return_tensors='pt').to(self.device)
        repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
        mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2].to(self.device)
        masked_input = repeat_input.masked_fill(mask == 1, self.tokenizer.mask_token_id)
        labels = repeat_input.masked_fill( masked_input != self.tokenizer.mask_token_id, -100)
        with torch.inference_mode():
            loss = self.model(masked_input, labels=labels).loss
        return torch.exp(loss).item()
        
    def __call__(self, val_examples: list, test_examples: list, output_path: str):
        
        # validation
        val_labels = [x.label for x in val_examples]
        val_results = [self.pseudo_perplexity_for_parent(sample.texts[0], sample.texts[1]) for sample in tqdm(val_examples, desc="Validation")] 
        val_perplexities = torch.tensor(val_results).to(self.device)
        val_labels = torch.tensor(val_labels).to(self.device)
        
        best_val_f1 = -1
        best_val_scores = None
        best_val_threshold = None
        thresholds = list(range(int(val_perplexities.min()), int(val_perplexities.max()), 100))
        for threshold in tqdm(thresholds, desc="Iteration"):
            val_predictions = val_perplexities < threshold
            tp = torch.sum((val_labels==1) & (val_predictions==1))
            fp = torch.sum((val_labels==0) & (val_predictions==1))
            fn = torch.sum((val_predictions!=1) & (val_labels==1))
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)
            acc = torch.sum(val_labels == val_predictions) / len(val_labels)
            if f1 >= best_val_f1:
                best_val_f1 = f1
                best_val_threshold = threshold
                best_val_scores = {
                    "threshold": best_val_threshold,
                    "P": precision,
                    "R": recall,
                    "f1": best_val_f1,
                    "ACC": acc,
                }
        save_file(best_val_scores, f"{output_path}/perplexity_val_results.json")
        
        
        # testing
        test_labels = [x.label for x in test_examples]
        test_results = [self.pseudo_perplexity_for_parent(sample.texts[0], sample.texts[1]) for sample in tqdm(test_examples, desc="Testing")] 
        test_perplexities = torch.tensor(test_results).to(self.device)
        test_labels = torch.tensor(test_labels).to(self.device)
        test_predictions = test_perplexities < best_val_threshold
        tp = torch.sum((test_labels==1) & (test_predictions==1))
        fp = torch.sum((test_labels==0) & (test_predictions==1))
        fn = torch.sum((test_predictions!=1) & (test_labels==1))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        acc = torch.sum(test_labels == test_predictions) / len(test_labels)
        test_scores = {
            "threshold": best_val_threshold,
            "P": precision,
            "R": recall,
            "f1": f1,
            "ACC": acc,
        }
        save_file(test_scores, f"{output_path}/perplexity_test_results.json")
