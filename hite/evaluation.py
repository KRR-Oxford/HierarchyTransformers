from tqdm.autonotebook import trange
from geoopt.manifolds import PoincareBall
import torch
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers import SentenceTransformer
import evaluate
from deeponto.utils import save_file
import logging

from .loss import HyperbolicLoss

logger = logging.getLogger(__name__)


class HyperbolicLossEvaluator(SentenceEvaluator):
    """Hyperbolic loss evaluator that extends the base evaluator from `sentence_transformers.evaluation`.
    """

    def __init__(
        self, 
        loss_module: HyperbolicLoss, 
        manifold: PoincareBall, 
        device: torch.device,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader = None,
        train_dataloader: DataLoader = None,
    ):
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.train_dataloader = train_dataloader
        self.loss_module = loss_module
        self.manifold = manifold
        self.device = device
        self.loss_module.to(self.device)

    @staticmethod
    def evaluate_f1(result_mat: torch.Tensor, granuality: int = 1000, best_val_threshold: float = None):
        scores = result_mat[:, 1] + (result_mat[:, 3] - result_mat[:, 2])
        start = int(scores.min() * granuality)
        end = int(scores.max() * granuality)

        best_threshold = -1
        best_f1 = 0.0
        best_scores = None

        for threshold in range(start, end):
            if best_val_threshold:
                threshold = best_val_threshold
            else:
                threshold = threshold / granuality
            positives = result_mat[scores <= threshold]
            negatives = result_mat[scores > threshold]
            tp = (positives[:, 0] == 1.0).sum()
            fp = (positives[:, 0] == 0.0).sum()
            tn = (negatives[:, 0] == 0.0).sum()
            fn = (negatives[:, 0] == 1.0).sum()
            accuracy = (tp + tn) / (tp + fp + tn + fn)
            neg_accuracy = tn / (fp + tn)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_scores = {
                    "threshold": best_threshold,
                    "P": precision.item(),
                    "R": recall.item(),
                    "f1": best_f1.item(),
                    "ACC": accuracy.item(),
                    "ACC-": neg_accuracy.item(),
                }
            if best_val_threshold:
                break
        return best_scores
    
    def evaluate_dataloader(self, model: SentenceTransformer, dataloader: DataLoader, best_val_threshold: float = None):
        
        # set up data iterator
        dataloader.collate_fn = model.smart_batching_collate
        data_iterator = iter(dataloader)

        eval_loss = 0.0
        results = []
        with torch.no_grad():
            for num_batch in trange(len(dataloader), desc="Iteration", smoothing=0.05, disable=False):
                sentence_features, labels = next(data_iterator)

                # move data to gpu
                for i in range(0, len(sentence_features)):
                    for key, _ in sentence_features[i].items():
                        sentence_features[i][key] = sentence_features[i][key].to(self.device)
                labels = labels.to(self.device)

                # compute eval matrix
                reps = [model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
                if not self.loss_module.apply_triplet_loss:
                    assert len(reps) == 2
                    rep_anchor, rep_other = reps

                    dists = self.manifold.dist(rep_anchor, rep_other)

                    rep_anchor_norms = self.manifold.dist0(rep_anchor)
                    rep_other_norms = self.manifold.dist0(rep_other)

                    results.append(torch.stack([labels, dists, rep_anchor_norms, rep_other_norms]).T)
                else:
                    assert len(reps) == 3
                    rep_anchor, rep_positive, rep_negative = reps

                    positive_dists = self.manifold.dist(rep_anchor, rep_positive)
                    negative_dists = self.manifold.dist(rep_anchor, rep_negative)

                    rep_anchor_norms = self.manifold.dist0(rep_anchor)
                    rep_positive_norms = self.manifold.dist0(rep_positive)
                    rep_negative_norms = self.manifold.dist0(rep_negative)

                    results.append(
                        torch.stack(
                            [
                                torch.ones(positive_dists.shape).to(positive_dists.device),
                                positive_dists,
                                rep_anchor_norms,
                                rep_positive_norms,
                            ]
                        ).T
                    )
                    results.append(
                        torch.stack(
                            [
                                torch.zeros(negative_dists.shape).to(negative_dists.device),
                                negative_dists,
                                rep_anchor_norms,
                                rep_negative_norms,
                            ]
                        ).T
                    )

                # compute eval loss
                cur_loss = self.loss_module(sentence_features, labels)
                if not torch.isnan(cur_loss):
                    eval_loss += cur_loss.item()
                    logger.info(f"eval_loss={eval_loss / (num_batch + 1)}")
                else:
                    logger.info(f"skip as detecting nan loss")

        # compute score
        result_mat = torch.cat(results, dim=0)
        if self.loss_module.apply_triplet_loss:
            logging.info("reshape result matrix following evaluation order")
            # 10 negatives per positive
            positive_mat = result_mat[result_mat[:, 0] == 1.0][::10]
            negative_mat = result_mat[result_mat[:, 0] == 0.0]
            real_mat = []
            for i in range(len(positive_mat)):
                real_mat += [positive_mat[i].unsqueeze(0), negative_mat[10 * i : 10 * (i + 1)]]
            result_mat = torch.concat(real_mat, dim=0)
        eval_scores = self.evaluate_f1(result_mat, 1000, best_val_threshold)

        self.loss_module.zero_grad()
        self.loss_module.train()

        results = self.loss_module.get_config_dict()
        results["loss"] = eval_loss / (num_batch + 1)
        results["scores"] = eval_scores
        
        return result_mat, results


    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        """
        This is called during training to evaluate the model.
        It returns a score for the evaluation with a higher score indicating a better result.

        :param model:
            the model to evaluate
        :param output_path:
            path where predictions and metrics are written to
        :param epoch
            the epoch where the evaluation takes place.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation on test data.
        :param steps
            the steps in the current epoch at time of the evaluation.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation at the end of the epoch.
        :return: a score for the evaluation with a higher score indicating a better result
        """

        # set to eval mode
        model.eval()
        self.loss_module.eval()
        model.save(f"{output_path}/epoch={epoch}.step={steps}")
        
        if self.train_dataloader:
            logger.info("Evaluate on train examples...")
            train_result_mat, train_results = self.evaluate_dataloader(model, self.train_dataloader, None)
            torch.save(train_result_mat, f"{output_path}/epoch={epoch}.step={steps}/train_result_mat.pt")
            save_file(train_results, f"{output_path}/epoch={epoch}.step={steps}/train_results.json")
        
        logger.info("Evaluate on val examples...")
        val_result_mat, val_results = self.evaluate_dataloader(model, self.val_dataloader, None)
        torch.save(val_result_mat, f"{output_path}/epoch={epoch}.step={steps}/val_result_mat.pt")
        save_file(val_results, f"{output_path}/epoch={epoch}.step={steps}/val_results.json")
        
        if self.test_dataloader:
            logger.info("Evaluate on test examples using best val threshold...")
            test_result_mat, test_results = self.evaluate_dataloader(model, self.test_dataloader, val_results["scores"]["threshold"])
            torch.save(test_result_mat, f"{output_path}/epoch={epoch}.step={steps}/test_result_mat.pt")
            save_file(test_results, f"{output_path}/epoch={epoch}.step={steps}/test_results.json")

        return val_results["scores"]["f1"]


# class PerplexityEvaluator:
    
#     def __init__(self, device: torch.device, val_examples: list, test_examples: list):
        
#         self.perplexity = evaluate.load("perplexity", module_type="metric")
#         self.val_examples = val_examples
#         self.test_examples = test_examples
#         self.device = device
        
#     def __call__(self, model_id: str, output_path: str):
        
#         # validation
#         val_inputs = [" is a ".join(x.texts) for x in self.val_examples]
#         val_labels = [x.label for x in self.val_examples]
#         val_results = self.perplexity.compute(
#             model_id=model_id,
#             add_start_token=False,
#             predictions=val_inputs,
#             device=self.device
#         )
#         val_perplexities = torch.tensor(val_results["perplexities"]).to(self.device)
#         val_labels = torch.tensor(val_labels).to(self.device)
        
#         best_val_f1 = -1
#         best_val_scores = None
#         best_val_threshold = None
#         thresholds = list(range(int(val_perplexities.min()), int(val_perplexities.max(), 100)))
#         for i in trange(len(thresholds), desc="Iteration", smoothing=0.05, disable=False):
#             threshold = thresholds[i]
#             val_predictions = val_perplexities >= threshold
#             tp = torch.sum((val_labels==1) & (val_predictions==1))
#             fp = torch.sum((val_labels==0) & (val_predictions==1))
#             fn = torch.sum((val_predictions!=1) & (val_labels==1))
#             precision = tp / (tp + fp)
#             recall = tp / (tp + fn)
#             f1 = 2 * (precision * recall) / (precision + recall)
#             acc = torch.sum(val_labels == val_predictions)
#             if f1 >= best_val_f1:
#                 best_val_f1 = f1
#                 best_val_threshold = threshold
#                 best_val_scores = {
#                     "threshold": best_val_threshold,
#                     "P": precision,
#                     "R": recall,
#                     "f1": best_val_f1,
#                     "ACC": acc,
#                 }
#         save_file(best_val_scores, f"{output_path}/perplexity_val_results.json")
        
        
#         # testing
#         test_inputs = [" is a ".join(x.texts) for x in self.test_examples]
#         test_labels = [x.label for x in self.test_examples]
#         test_results = self.perplexity.compute(
#             model_id=model_id,
#             add_start_token=False,
#             predictions=test_inputs,
#             device=self.device
#         )
#         test_perplexities = torch.tensor(test_results["perplexities"]).to(self.device)
#         test_labels = torch.tensor(test_labels).to(self.device)
#         test_predictions = test_perplexities >= best_val_threshold
#         tp = torch.sum((test_labels==1) & (test_predictions==1))
#         fp = torch.sum((test_labels==0) & (test_predictions==1))
#         fn = torch.sum((test_predictions!=1) & (test_labels==1))
#         precision = tp / (tp + fp)
#         recall = tp / (tp + fn)
#         f1 = 2 * (precision * recall) / (precision + recall)
#         acc = torch.sum(test_labels == test_predictions)
#         test_scores = {
#             "threshold": best_val_threshold,
#             "P": precision,
#             "R": recall,
#             "f1": f1,
#             "ACC": acc,
#         }
#         save_file(test_scores, f"{output_path}/perplexity_test_results.json")
