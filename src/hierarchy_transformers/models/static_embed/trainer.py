# Copyright 2023 Yuan He

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from geoopt.optim import RiemannianAdam

from ...losses import HyperbolicEntailmentConeLoss
from .embed import HyperbolicStaticEmbedding

import logging

logger = logging.getLogger(__name__)


class HyperbolicStaticEmbeddingTrainer:
    r"""
    Class for training the static hyperbolic embedding models:

    1. Poincaré Embedding by [Nickel et al., NeurIPS 2017](https://arxiv.org/abs/1705.08039).
    2. Hyperbolic Entailment Cone by [Ganea et al., ICML 2018](https://arxiv.org/abs/1804.01882).
    """

    def __init__(
        self,
        model: HyperbolicStaticEmbedding,
        device: torch.device,
        train_dataloader: DataLoader,
        learning_rate: float = 0.01,
        num_epochs: int = 200,
        num_warmup_epochs: int = 10,
        apply_cone_loss: bool = False,  # cone loss should be used after training Poincare embed
    ):
        self.model = model
        self.device = device
        self.model.to(device)
        self.train_dataloader = train_dataloader
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.eloss = HyperbolicEntailmentConeLoss(self.model.manifold, 0.1, 0.1, 1e-5)

        self.learning_rate = learning_rate
        self.optimizer = RiemannianAdam(self.model.parameters(), lr=self.learning_rate)
        self.current_epoch = 0
        self.num_epochs = num_epochs
        self.num_epoch_steps = len(self.train_dataloader)
        self.num_training_steps = self.num_epoch_steps * self.num_epochs
        self.warmup_epochs = num_warmup_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_epochs * self.num_epoch_steps,  # one epoch warming-up
            num_training_steps=self.num_training_steps,
        )

        self.loss_func = self.dist_loss
        if apply_cone_loss:
            self.loss_func = self.cone_loss

    @property
    def lr(self):
        for g in self.optimizer.param_groups:
            return g["lr"]

    def dist_loss(self, subject: torch.Tensor, objects: torch.Tensor):
        """
        Hyperbolic distance loss function proposed in [Nickel et al., NeurIPS 2017](https://arxiv.org/abs/1705.08039).
        """
        # first object is always the correct one
        pred_dists = self.model.manifold.dist(subject, objects)
        correct_object_indices = torch.tensor([0] * len(pred_dists)).to(pred_dists.device)
        return self.cross_entropy(-pred_dists, correct_object_indices)

    def cone_loss(self, subject: torch.Tensor, objects: torch.Tensor):
        """
        Hyperbolic Cone Loss proposed in [Ganea et al., ICML 2018](https://arxiv.org/abs/1804.01882).
        """
        energy = self.eloss.energy(objects, subject)
        return (energy[:, 0].sum() + F.relu(self.eloss.margin - energy[:, 1:]).sum()) / torch.numel(energy)

    def training_step(self, batch):
        batch = batch.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        subject, objects = self.model(batch)
        loss = self.loss_func(subject, objects)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss

    def run(self, output_path: str):
        """
        Run the training procedure this model.
        """
        for _ in range(self.num_epochs):
            epoch_bar = tqdm(
                range(self.num_epoch_steps), desc=f"Epoch {self.current_epoch + 1}", leave=True, unit="batch"
            )
            for batch in self.train_dataloader:
                loss = self.training_step(batch)
                # running_loss += loss
                epoch_bar.set_postfix({"loss": loss.item(), "lr": self.lr})
                epoch_bar.update()
            self.current_epoch += 1
        torch.save(self.model, f"{output_path}/poincare.{self.model.embed_dim}d.pt")
