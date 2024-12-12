# Copyright 2024 Yuan He

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

import random
import tempfile

import pytest
from datasets import load_dataset
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

from hierarchy_transformers.losses import HierarchyTransformerLoss
from hierarchy_transformers.models import HierarchyTransformer


@pytest.fixture
def model_path():
    return "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture
def dataset_path():
    return "Hierarchy-Transformers/WordNetNoun"


def test_training(model_path, dataset_path):
    # Create a temporary directory for the output files
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. Load dataset and model
        hop_type = random.choice(["MultiHop", "MixedHop"])
        neg_type = random.choice(["HardNegatives", "RandomNegatives"])
        triplet_dataset = load_dataset(dataset_path, f"{hop_type}-{neg_type}-Triplets")
        trial_train = triplet_dataset["train"].select(range(64))
        trial_val = triplet_dataset["val"].select(range(32))
        model = HierarchyTransformer.from_pretrained(model_path)

        # 2. set up the loss function
        hit_loss = HierarchyTransformerLoss(model=model)

        # 3. Define the training arguments
        args = SentenceTransformerTrainingArguments(
            output_dir=temp_dir,
            num_train_epochs=1,
            learning_rate=1e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_ratio=0.05,  # alternatively, set warmup_steps to 500
            eval_strategy="epoch",
            save_strategy="epoch",
        )

        # 4. Train the model on trial samples
        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=trial_train,  # train loss requires triplets
            eval_dataset=trial_val,  # val loss requires triplets
            loss=hit_loss,
        )
        trainer.train()
