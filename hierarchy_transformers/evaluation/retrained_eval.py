from tqdm.auto import trange, tqdm
from geoopt.manifolds import PoincareBall
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from deeponto.utils import save_file
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

from ..losses import HyperbolicLoss
from .hierarchy_eval import HierarchyEvaluator


class HierarchyRetrainedEvaluator(HierarchyEvaluator):
    """Evaluator hierarchy re-trained language models (HiT).

    Hierarchy encoding is evaluated based on the hyperbolic distances between entity
    embeddings and hyperbolic norms of entity embeddings in the Poincare ball of
    radius 1/sqrt(N) where N is the embedding dimension.
    """

    def __init__(
        self,
        loss_module: HyperbolicLoss,
        manifold: PoincareBall,
        device: torch.device,
        val_examples: list,
        eval_batch_size: int,
        test_examples: Optional[list] = None,
        train_examples: Optional[list] = None,
    ):
        super().__init__()
        self.val_dataloader = DataLoader(val_examples, shuffle=False, batch_size=eval_batch_size)
        self.test_dataloader = (
            DataLoader(test_examples, shuffle=False, batch_size=eval_batch_size) if test_examples else None
        )
        self.train_dataloader = (
            DataLoader(train_examples, shuffle=False, batch_size=eval_batch_size) if train_examples else None
        )
        self.loss_module = loss_module
        self.manifold = manifold
        self.device = device
        self.loss_module.to(self.device)

    @classmethod
    def score(cls, result_mat: torch.Tensor, centri_score_weight: float):
        scores = result_mat[:, 1] + centri_score_weight * (result_mat[:, 3] - result_mat[:, 2])
        labels = result_mat[:, 0]
        return scores, labels

    @classmethod
    def search_best_threshold(cls, result_mat: torch.Tensor, threshold_granularity: int = 100):
        best_f1 = -1.0
        best_results = None
        is_updated = True

        for centri_score_weight in range(10):
            # early stop if increasing the centri score weight does not help
            if not is_updated:
                break
            is_updated = False

            centri_score_weight = float(centri_score_weight + 1)
            scores, labels = cls.score(result_mat, centri_score_weight)
            cur_best_results = super().search_best_threshold(
                scores,
                labels,
                threshold_granularity,
                determined_metric="F1",
                best_determined_metric_value=best_f1,
                preformatted_best_results={"centri_score_weight": centri_score_weight},
            )
            if cur_best_results:
                best_results = cur_best_results
                is_updated = True

        return best_results

    def inference(
        self,
        model: SentenceTransformer,
        dataloader: DataLoader,
        best_val_centri_score_weight: float = None,
        best_val_threshold: float = None,
    ):
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
        if not best_val_threshold or not best_val_centri_score_weight:
            eval_results = self.search_best_threshold(result_mat, 100)
        else:
            eval_scores, eval_labels = self.score(result_mat, best_val_centri_score_weight)
            eval_results = self.evaluate_by_threshold(eval_scores, eval_labels, best_val_threshold)

        self.loss_module.zero_grad()
        self.loss_module.train()

        results = self.loss_module.get_config_dict()
        results["loss"] = eval_loss / (num_batch + 1)
        results["scores"] = eval_results

        return result_mat, results

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        """This is called during training to evaluate the model.
        It returns a score for the evaluation with a higher score indicating a better result.
        """

        # set to eval mode
        model.eval()
        self.loss_module.eval()
        Path(f"{output_path}/epoch={epoch}.step={steps}").mkdir(parents=True, exist_ok=True)
        # model.save(f"{output_path}/epoch={epoch}.step={steps}")

        if self.train_dataloader:
            logger.info("Evaluate on train examples...")
            train_result_mat, train_results = self.inference(model, self.train_dataloader)
            torch.save(train_result_mat, f"{output_path}/epoch={epoch}.step={steps}/train_result_mat.pt")
            save_file(train_results, f"{output_path}/epoch={epoch}.step={steps}/train_results.json")

        logger.info("Evaluate on val examples...")
        val_result_mat, val_results = self.inference(model, self.val_dataloader)
        torch.save(val_result_mat, f"{output_path}/epoch={epoch}.step={steps}/val_result_mat.pt")
        save_file(val_results, f"{output_path}/epoch={epoch}.step={steps}/val_results.json")

        if self.test_dataloader:
            logger.info("Evaluate on test examples using best val threshold...")
            test_result_mat, test_results = self.inference(
                model,
                self.test_dataloader,
                val_results["scores"]["centri_score_weight"],
                val_results["scores"]["threshold"],
            )
            torch.save(test_result_mat, f"{output_path}/epoch={epoch}.step={steps}/test_result_mat.pt")
            save_file(test_results, f"{output_path}/epoch={epoch}.step={steps}/test_results.json")

        return val_results["scores"]["F1"]
