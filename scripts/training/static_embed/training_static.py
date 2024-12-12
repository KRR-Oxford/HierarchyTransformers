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
import os

import click
import torch
from deeponto.utils import create_path, load_file, save_file, set_seed
from yacs.config import CfgNode

from hierarchy_transformers.datasets import load_zenodo_dataset
from hierarchy_transformers.evaluation import PoincareStaticEmbeddingEvaluator
from hierarchy_transformers.losses import HyperbolicEntailmentConeStaticLoss, PoincareEmbeddingStaticLoss
from hierarchy_transformers.models import PoincareStaticEmbedding, PoincareStaticEmbeddingTrainer
from hierarchy_transformers.utils import get_torch_device

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@click.command()
@click.option("-c", "--config_file", type=click.Path(exists=True))
@click.option("-g", "--gpu_id", type=int, default=0)
def main(config_file: str, gpu_id: int):

    # 0. set seed, load config, and format output dir
    set_seed(8888)
    config = CfgNode(load_file(config_file))
    dataset_path_suffix = config.dataset_path.split(os.path.sep)[-1]
    output_dir = f"experiments/PoincareStatic-{dataset_path_suffix}-{config.negative_type}"
    create_path(output_dir)
    save_file(load_file(config_file), os.path.join(output_dir, "config.yaml"))  # save config to output dir

    # 1. Load dataset and pre-trained model
    entity_lexicon = load_file(os.path.join(config.dataset_path, "entity_lexicon.json"))
    model = PoincareStaticEmbedding(list(entity_lexicon.keys()), embed_dim=config.embed_dim)
    print(model)
    dataset = load_zenodo_dataset(
        path=config.dataset_path,
        entity_lexicon_or_index=model.ent2idx,
        negative_type=config.negative_type,
        example_type="idx",
    )

    # 2. set up the loss function
    poincare_embed_loss = PoincareEmbeddingStaticLoss(model.manifold)

    # 3. Create the trainer & start training
    logger.info("Train Poincare embedding on the hyperbolic distance loss...")
    device = get_torch_device(gpu_id)
    trainer = PoincareStaticEmbeddingTrainer(
        model=model,
        train_dataset=dataset["train"],
        loss=poincare_embed_loss,
        num_train_epochs=int(config.num_train_epochs),
        learning_rate=float(config.learning_rate),
        train_batch_size=int(config.train_batch_size),
        warmup_epochs=int(config.warmup_epochs),
    )
    trainer.train(device=device)
    torch.save(trainer.model, os.path.join(output_dir, "poincare_static.pt"))

    # 4. Evaluate the model performance on validation and test datasets
    create_path(os.path.join(output_dir, "eval_poincare"))
    val_evaluator = PoincareStaticEmbeddingEvaluator(
        eval_examples=dataset["val"], batch_size=config.eval_batch_size, truth_label=1
    )
    val_evaluator(model=trainer.model, loss=trainer.loss, device=device, epoch="validation", output_path=os.path.join(output_dir, "eval_poincare"))
    val_results = val_evaluator.results
    best_val = val_results.loc[val_results["f1"].idxmax()]
    best_val_threshold = float(best_val["threshold"])
    test_evaluator = PoincareStaticEmbeddingEvaluator(
        eval_examples=dataset["test"], batch_size=config.eval_batch_size, truth_label=1
    )
    test_evaluator(model=trainer.model, loss=trainer.loss, device=device, output_path=os.path.join(output_dir, "eval_poincare"), best_threshold=best_val_threshold)

    # 5. Create the trainer & start post-training
    if int(config.num_post_train_epochs) > 0:
        logger.info("Post-train Poincare embedding on the hyperbolic entailment cone loss...")
        # set-up the cone loss for post-training
        hyperbolic_cone_loss = HyperbolicEntailmentConeStaticLoss(model.manifold)
        post_trainer = PoincareStaticEmbeddingTrainer(
            model=trainer.model,  # continue to train
            train_dataset=dataset["train"],
            loss=hyperbolic_cone_loss,
            num_train_epochs=int(config.num_post_train_epochs),
            learning_rate=float(config.learning_rate),
            train_batch_size=int(config.train_batch_size),
            warmup_epochs=int(config.warmup_epochs),
        )
        post_trainer.train(device=device)
        torch.save(post_trainer.model, os.path.join(output_dir, "hypercone_static.pt"))

        # 6. Evaluate the post-trained model performance on validation and test datasets
        create_path(os.path.join(output_dir, "eval_hypercone"))
        val_evaluator = PoincareStaticEmbeddingEvaluator(
            eval_examples=dataset["val"], batch_size=config.eval_batch_size, truth_label=1
        )
        val_evaluator(model=post_trainer.model, loss=post_trainer.loss, device=device, epoch="validation", output_path=os.path.join(output_dir, "eval_hypercone"))
        val_results = val_evaluator.results
        best_val = val_results.loc[val_results["f1"].idxmax()]
        best_val_threshold = float(best_val["threshold"])
        test_evaluator = PoincareStaticEmbeddingEvaluator(
            eval_examples=dataset["test"], batch_size=config.eval_batch_size, truth_label=1
        )
        test_evaluator(model=post_trainer.model, loss=post_trainer.loss, device=device, output_path=os.path.join(output_dir, "eval_hypercone"), best_threshold=best_val_threshold)


if __name__ == "__main__":
    main()
