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
from __future__ import annotations

import logging

import torch
from geoopt.optim import RiemannianAdam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .poincare_embed import PoincareStaticEmbedding

logger = logging.getLogger(__name__)


class PoincareStaticEmbeddingTrainer:
    r"""Class for training the static hyperbolic embedding models:

        - [1] Poincaré Embedding by [Nickel et al., NeurIPS 2017](https://arxiv.org/abs/1705.08039).
        - [2] Hyperbolic Entailment Cone by [Ganea et al., ICML 2018](https://arxiv.org/abs/1804.01882).

    both of which lie in a unit Poincaré ball. According to [2], it is better to apply the entailment cone loss in the post-training phase of a Poincaré embedding model in [1].
    """

    def __init__(
        self,
        model: PoincareStaticEmbedding,
        train_dataset: list,
        loss: torch.nn.Module,
        num_train_epochs: int = 256,
        learning_rate: float = 0.01,
        train_batch_size: int = 200,
        warmup_epochs: int = 10,
    ):
        self.model = model
        self.train_dataloader = DataLoader(torch.tensor(train_dataset), shuffle=True, batch_size=train_batch_size)
        self.loss = loss
        self.learning_rate = learning_rate
        self.optimizer = RiemannianAdam(self.model.parameters(), lr=self.learning_rate)
        self.current_epoch = 0
        self.num_train_epochs = num_train_epochs
        self.num_epoch_steps = len(self.train_dataloader)
        self.num_training_steps = self.num_epoch_steps * self.num_train_epochs
        self.warmup_epochs = warmup_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_epochs * self.num_epoch_steps,  # one epoch warming-up
            num_training_steps=self.num_training_steps,
        )

    @property
    def lr(self):
        for g in self.optimizer.param_groups:
            return g["lr"]

    def training_step(self, batch, device):
        batch = batch.to(device)
        self.optimizer.zero_grad(set_to_none=True)
        subject, objects = self.model(batch)
        loss = self.loss(subject, objects)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss

    def train(self, device):
        self.model.to(device)
        for _ in range(self.num_train_epochs):
            epoch_bar = tqdm(
                range(self.num_epoch_steps), desc=f"Epoch {self.current_epoch + 1}", leave=True, unit="batch"
            )
            for batch in self.train_dataloader:
                loss = self.training_step(batch, device)
                epoch_bar.set_postfix({"batch_loss": loss.item(), "lr": self.lr})
                epoch_bar.update()
            self.current_epoch += 1
