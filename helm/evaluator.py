from tqdm.autonotebook import trange
from geoopt.manifolds import PoincareBall
import torch
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import SentenceEvaluator
from deeponto.utils import save_file
import logging

from .loss import HyperbolicLoss

logger = logging.getLogger(__name__)


class HyperbolicLossEvaluator(SentenceEvaluator):
    """
    Base class for all evaluators

    Extend this class and implement __call__ for custom evaluators.
    """

    def __init__(self, data_loader: DataLoader, loss_module: HyperbolicLoss, manifold: PoincareBall, device):
        self.data_loader = data_loader
        self.loss_module = loss_module
        self.manifold = manifold
        self.device = device
        self.loss_module.to(self.device)

    @staticmethod
    def evaluate_f1(result_mat: torch.Tensor, granuality: int = 1000, scale_down_truths: float = 1.0):
        
        scores = result_mat[:, 1] + (result_mat[:, 3] - result_mat[:, 2])
        start = int(scores.min() * granuality)
        end = int(scores.max() * granuality)

        best_threshold = -1
        best_f1 = 0.0
        best_scores = None

        for threshold in range(start, end):
            threshold = threshold / granuality
            positives = result_mat[scores <= threshold]
            negatives = result_mat[scores > threshold]
            tp = (positives[:, 0] == 1.0).sum() / scale_down_truths
            fp = (positives[:, 0] == 0.0).sum()
            tn = (negatives[:, 0] == 0.0).sum()
            fn = (negatives[:, 0] == 1.0).sum() / scale_down_truths
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
        return best_scores

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

        # set up data iterator
        self.data_loader.collate_fn = model.smart_batching_collate
        data_iterator = iter(self.data_loader)

        eval_loss = 0.0
        results = []
        with torch.no_grad():
            for num_batch in trange(len(self.data_loader), desc="Iteration", smoothing=0.05, disable=False):
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

                    rep_anchor_norms = self.manifold.dist(
                        rep_anchor, self.manifold.origin(rep_anchor.shape).to(rep_anchor.device)
                    )
                    rep_other_norms = self.manifold.dist(
                        rep_other, self.manifold.origin(rep_other.shape).to(rep_other.device)
                    )

                    results.append(torch.stack([labels, dists, rep_anchor_norms, rep_other_norms]).T)
                else:
                    assert len(reps) == 3
                    rep_anchor, rep_positive, rep_negative = reps

                    positive_dists = self.manifold.dist(rep_anchor, rep_positive)
                    negative_dists = self.manifold.dist(rep_anchor, rep_negative)

                    rep_anchor_norms = self.manifold.dist(
                        rep_anchor, self.manifold.origin(rep_anchor.shape).to(rep_anchor.device)
                    )
                    rep_positive_norms = self.manifold.dist(
                        rep_positive, self.manifold.origin(rep_positive.shape).to(rep_positive.device)
                    )
                    rep_negative_norms = self.manifold.dist(
                        rep_negative, self.manifold.origin(rep_negative.shape).to(rep_negative.device)
                    )

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
        scale_down_truths = 10.0 if self.loss_module.apply_triplet_loss else 1.0
        eval_scores = self.evaluate_f1(result_mat, 1000, scale_down_truths)

        self.loss_module.zero_grad()
        self.loss_module.train()
        model.save(f"{output_path}/epoch={epoch}.step={steps}")

        results = self.loss_module.get_config_dict()
        results["loss"] = eval_loss / (num_batch + 1)
        results["scores"] = eval_scores
        torch.save(result_mat, f"{output_path}/epoch={epoch}.step={steps}/eval_results.pt")
        save_file(results, f"{output_path}/epoch={epoch}.step={steps}/eval_results.json")

        return -eval_loss
