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
from torch.utils.data import DataLoader
from geoopt.manifolds import PoincareBall
from geoopt import ManifoldParameter
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from geoopt.optim import RiemannianAdam
import logging

logger = logging.getLogger(__name__)


class StaticPoincareEmbed(torch.nn.Module):
    r"""Basline model reproducing the static embedding model proposed in
    [Nickel et al., 2017](https://arxiv.org/abs/1705.08039).
    """

    def __init__(self, entity_ids: list, embed_dim: int):
        super().__init__()

        self.entities = entity_ids
        self.idx2ent = {idx: ent for idx, ent in enumerate(self.entities)}
        self.ent2idx = {v: k for k, v in self.idx2ent.items()}
        self.embed_dim = embed_dim
        self.manifold = PoincareBall()
        self.dist = self.manifold.dist
        self.embed = self.init_static_graph_embedding(len(self.idx2ent), self.embed_dim, 1e-3)

    def init_static_graph_embedding(self, static_entity_size: int, embed_dim: int, init_weights: float):
        # init embedding weights to somewhere near the origin
        static_embedding = torch.nn.Embedding(static_entity_size, embed_dim, sparse=False, max_norm=1.0)
        static_embedding.weight.data.uniform_(-init_weights, init_weights)
        static_embedding.weight = ManifoldParameter(static_embedding.weight, manifold=self.manifold)
        return static_embedding

    def forward(self, inputs: torch.Tensor):
        """Split input tensor into subject and objects

        NOTE: the first object is the related one and the rest are negative samples.
        """
        input_embeds = self.embed(
            inputs
        )  # (batch_size, num_entities, hidden_dim), dim 1 includes (child, parent, negative_parents*)
        objects = input_embeds.narrow(dim=1, start=1, length=input_embeds.size(1) - 1)  # use .narrow to keep dim
        subject = input_embeds.narrow(dim=1, start=0, length=1).expand_as(objects)
        return subject, objects


class StaticPoincareEmbedTrainer:
    def __init__(
        self,
        model: StaticPoincareEmbed,
        device: torch.device,
        train_dataloader: DataLoader,
        learning_rate: float = 0.01,
        num_epochs: int = 200,
        num_warmup_epochs: int = 10,
    ):
        self.model = model
        self.device = device
        self.model.to(device)
        self.train_dataloader = train_dataloader
        self.cross_entropy = torch.nn.CrossEntropyLoss()

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

    @property
    def lr(self):
        for g in self.optimizer.param_groups:
            return g["lr"]

    def dist_loss(self, subject: torch.Tensor, objects: torch.Tensor):
        # first object is always the correct one
        pred_dists = self.model.manifold.dist(subject, objects)
        correct_object_indices = torch.tensor([0] * len(pred_dists)).to(pred_dists.device)
        return self.cross_entropy(-pred_dists, correct_object_indices)

    def training_step(self, batch):
        batch = batch.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        subject, objects = self.model(batch)
        loss = self.dist_loss(subject, objects)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss

    def run(self, output_path: str):
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
