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
"""This training script is hierarchy re-training of HiT models."""
from __future__ import annotations

import logging
import os
import shutil
import sys

import click
from deeponto.utils import create_path, load_file, set_seed
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from yacs.config import CfgNode

from hierarchy_transformers.datasets import load_hf_dataset
from hierarchy_transformers.evaluation import HierarchyTransformerEvaluator
from hierarchy_transformers.losses import HierarchyTransformerLoss
from hierarchy_transformers.models import HierarchyTransformer
from hierarchy_transformers.models.hierarchy_transformer.hit_trainer import HierarchyTransformerTrainer

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stderr)])
logger = logging.getLogger(__name__)


@click.command()
@click.option("-c", "--config_file", type=click.Path(exists=True))
def main(config_file: str):
    # 0. set seed, load config, and format output dir
    set_seed(8888)
    config = CfgNode(load_file(config_file))
    model_path_suffix = config.model_path.split(os.path.sep)[-1]
    dataset_path_suffix = config.dataset_path.split(os.path.sep)[-1]
    output_dir = f"experiments/HiT-{model_path_suffix}-{dataset_path_suffix}-{config.dataset_name}"
    create_path(output_dir)
    try:
        shutil.copy2(config_file, os.path.join(output_dir, "config.yaml"))
    except Exception:
        pass

    # 1. Load dataset and pre-trained model
    # NOTE: according to docs, it is very important to have column names ["child", "parent", "negative"] *in order* to match ["anchor", "positive", "negative"]
    triplet_dataset = load_hf_dataset(config.dataset_path, config.dataset_name + "-Triplets")
    pair_dataset = load_hf_dataset(config.dataset_path, config.dataset_name + "-Pairs")
    model = HierarchyTransformer.from_pretrained(model_name_or_path=config.model_path)

    # 2. set up the loss function
    hit_loss = HierarchyTransformerLoss(
        model=model,
        clustering_loss_weight=config.hit_loss.clustering_loss_weight,
        clustering_loss_margin=config.hit_loss.clustering_loss_margin,
        centripetal_loss_weight=config.hit_loss.centripetal_loss_weight,
        centripetal_loss_margin=config.hit_loss.centripetal_loss_margin,
    )
    logger.info(f"HiT loss config: {hit_loss.get_config_dict()}")

    # 3. Define a validation evaluator for use during training.
    val_evaluator = HierarchyTransformerEvaluator(
        child_entities=pair_dataset["val"]["child"],
        parent_entities=pair_dataset["val"]["parent"],
        labels=pair_dataset["val"]["label"],
        batch_size=config.eval_batch_size,
        truth_label=1,
    )

    # 4. Define the training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=int(config.num_train_epochs),
        learning_rate=float(config.learning_rate),
        per_device_train_batch_size=int(config.train_batch_size),
        per_device_eval_batch_size=int(config.eval_batch_size),
        warmup_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=100,
        metric_for_best_model="f1",  # to override loss for model selection
        greater_is_better=True,  # due to F1 score
        load_best_model_at_end=True,
    )

    # 5. Create the trainer & start training
    trainer = HierarchyTransformerTrainer(
        model=model,
        args=args,
        train_dataset=triplet_dataset["train"],  # train loss requires triplets
        eval_dataset=triplet_dataset["val"],  # val loss requires triplets
        loss=hit_loss,
        evaluator=val_evaluator,  # actual eval requires labelled pairs
    )
    trainer.train()

    # 6. Evaluate the model performance on the test dataset
    val_results = val_evaluator.results
    best_val = val_results.loc[val_results["f1"].idxmax()]
    best_val_centri_weight = float(best_val["centri_weight"])
    best_val_threshold = float(best_val["threshold"])
    test_evaluator = HierarchyTransformerEvaluator(
        child_entities=pair_dataset["test"]["child"],
        parent_entities=pair_dataset["test"]["parent"],
        labels=pair_dataset["test"]["label"],
        batch_size=config.eval_batch_size,
        truth_label=1,
    )
    test_evaluator(
        model=model,
        output_path=os.path.join(output_dir, "eval"),
        best_centri_weight=best_val_centri_weight,
        best_threshold=best_val_threshold,
    )

    # 7. Save the trained & evaluated model locally
    final_output_dir = f"{output_dir}/final"
    model.save(final_output_dir)


if __name__ == "__main__":
    main()
