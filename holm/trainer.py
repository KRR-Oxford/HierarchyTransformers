import torch
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from geoopt.optim import RiemannianAdam, RiemannianSGD
from .graph import SubsumptionGraph, HypernymDataset
from .model import PoincareOntologyEmbedding


class PoincareTrainer:
    def __init__(
        self,
        graph: SubsumptionGraph,
        embed_dim: int = 50,
        n_negative_samples: int = 10,
        batch_size: int = 50,
        learning_rate: float = 0.01,
        n_epochs: int = 200,
        n_warmup_epochs: int = 10,
        gpu_device: int = 0,
    ):
        self.graph = graph
        self.dataset = HypernymDataset(self.graph, n_negative_samples, True)
        self.batch_size = batch_size
        self.dataloader = self.get_dataloader(weighted_negative_sampling=True)
        self.learning_rate = learning_rate

        self.device = torch.device(f"cuda:{gpu_device}" if torch.cuda.is_available() else "cpu")
        self.model = PoincareOntologyEmbedding(self.graph, embed_dim=embed_dim).to(self.device)

        self.optimizer = RiemannianAdam(self.model.parameters(), lr=self.learning_rate)
        self.current_epoch = 0
        self.n_epochs = n_epochs
        self.n_epoch_steps = len(self.dataloader)
        self.n_trainining_steps = self.n_epoch_steps * self.n_epochs
        self.warmup_epochs = n_warmup_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_epochs * self.n_epoch_steps,  # one epoch warming-up
            num_training_steps=self.n_trainining_steps,
        )

    def get_dataloader(self, weighted_negative_sampling: bool = False):
        self.dataset.weighted_negative_sampling = weighted_negative_sampling
        return torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True, pin_memory=True, num_workers=10)

    @staticmethod
    def dist_loss(pred_dists: torch.Tensor, positive_idx: int = 0):
        """Computing log-softmax loss over poincare distances between the subject entity and the object entities.

        NOTE: pred_dists has shape (batch_size, num_distances); {positive_idx} is always the distance with the related entity
        """
        return (
            -torch.sum(-pred_dists[:, positive_idx] - torch.log(torch.sum(torch.exp(-pred_dists), dim=1)))
            / pred_dists.shape[0]
        )

    @property
    def lr(self):
        for g in self.optimizer.param_groups:
            return g["lr"]

    def training_step(self, batch, loss_func):
        batch = batch.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        preds = self.model(batch)
        loss = loss_func(preds)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss

    def training_epoch(self, loss_func, save_at_epoch=True):
        epoch_bar = tqdm(range(self.n_epoch_steps), desc=f"Epoch {self.current_epoch + 1}", leave=True, unit="batch")
        # change to uniform negative sampling after warm starting (or burn-in)
        if self.current_epoch >= self.warmup_epochs:
            self.dataloader = self.get_dataloader(weighted_negative_sampling=False)
        # running_loss = 0.0
        for batch in self.dataloader:
            loss = self.training_step(batch, loss_func)
            # running_loss += loss
            epoch_bar.set_postfix({"loss": loss.item(), "lr": self.lr})
            epoch_bar.update()
        self.current_epoch += 1
        # if save_at_epoch:
        #     torch.save(self.model, f"experiments/poincare.{dim}d.pt")

    def run(self):
        for _ in range(self.n_epochs):
            self.training_epoch(self.dist_loss)


    def save(self, output_dir: str):
        pass
