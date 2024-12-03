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

from deeponto.utils import load_file, set_seed
import logging
import os
import click
from yacs.config import CfgNode
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

from hierarchy_transformers.models import HierarchyTransformer
from hierarchy_transformers.losses import HierarchyTransformerLoss
from hierarchy_transformers.evaluation import HierarchyTransformerEvaluator
from hierarchy_transformers.datasets import load_hf_dataset


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@click.command()
@click.option("-c", "--config_file", type=click.Path(exists=True))
def main(config_file: str):
    set_seed(8888)
    config = CfgNode(load_file(config_file))

    # 1. Load dataset and pre-trained model
    dataset = load_hf_dataset(config.dataset_path, config.dataset_name)
    model = HierarchyTransformer.from_pretrained(model_name_or_path=config.model_path)

    # 2. set up the loss function
    hit_loss = HierarchyTransformerLoss(
        model=model,
        clustering_loss_weight=config.hit_loss.clustering_loss_weight,
        clustering_loss_margin=config.hit_loss.clustering_loss_margin,
        centripetal_loss_weight=config.hit_loss.centripetal_loss_weight,
        centripetal_loss_margin=config.hit_loss.centripetal_loss_margin,
    )

    print(hit_loss.get_config_dict())

    # 3. Define a validation evaluator for use during training.
    val_evaluator = HierarchyTransformerEvaluator(
        child_entities=dataset["val"]["child"],
        parent_entities=dataset["val"]["parent"],
        labels=dataset["val"]["label"],
        batch_size=config.eval_batch_size,
    )

    # 4. Define the training arguments
    dataset_suffix = config.data_path.split(os.path.sep)[-1]
    output_dir = f"experiments/HiT-{dataset_suffix}-{config.dataset_name}"
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        warmup_ratio=0.1,
        warmup_steps=config.warmup_steps,
        eval_strategy="epochs",
        save_strategy="epochs",
        save_total_limit=2,
        logging_steps=100,
        load_best_model_at_end=True
    )

    # 5. Create the trainer & start training
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        loss=hit_loss,
        evaluator=val_evaluator,
    )
    trainer.train()

    # 6. Evaluate the model performance on the STS Benchmark test dataset
    test_evaluator = HierarchyTransformerEvaluator(
        child_entities=dataset["test"]["child"],
        parent_entities=dataset["test"]["parent"],
        labels=dataset["test"]["label"],
        batch_size=config.eval_batch_size,
    )
    test_evaluator(model)

    # 7. Save the trained & evaluated model locally
    final_output_dir = f"{output_dir}/final"
    model.save(final_output_dir)


if __name__ == "__main__":
    main()
