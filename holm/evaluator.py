from tqdm.autonotebook import trange
import torch
from sentence_transformers.evaluation import SentenceEvaluator


class HyperbolicLossEvaluator(SentenceEvaluator):
    """
    Base class for all evaluators

    Extend this class and implement __call__ for custom evaluators.
    """

    def __init__(self, loader, loss_module, device) -> None:
        self.loader = loader
        self.loss_module = loss_module
        self.device = device

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
        self.loss_module.eval()
        loss = 0.0
        self.loader.collate_fn = model.smart_batching_collate
        num_batches = len(self.loader)
        data_iterator = iter(self.loader)
        with torch.no_grad():
            for _ in trange(num_batches, desc="Iteration", smoothing=0.05, disable=False):
                sentence_features, labels = next(data_iterator)
                # move data to gpu
                for i in range(0, len(sentence_features)):
                    for key, value in sentence_features[i].items():
                        sentence_features[i][key] = sentence_features[i][key].to(self.device)
                labels = labels.to(self.device)
                loss += self.loss_module(sentence_features, labels).item()

        final_loss = loss / num_batches

        self.loss_module.zero_grad()
        self.loss_module.train()
        model.save(f"{output_path}/epoch={epoch}.step={steps}")

        return -final_loss
