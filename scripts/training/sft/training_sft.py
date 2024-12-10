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
"""This training script is for standard [CLS] supervised fine-tuning for BERT models."""

from deeponto.utils import set_seed, create_path, load_file, save_file
import os, sys, logging, click
from yacs.config import CfgNode
import torch
import pandas as pd

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from hierarchy_transformers.datasets import load_hf_dataset
from hierarchy_transformers.evaluation.metrics import evaluate_by_threshold

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
    output_dir = f"experiments/SFT-{model_path_suffix}-{dataset_path_suffix}-{config.dataset_name}"
    create_path(output_dir)
    save_file(load_file(config_file), os.path.join(output_dir, "config.yaml"))  # save config to output dir

    # 1. Load dataset and pre-trained model
    # NOTE: according to docs, it is very important to have column names ["child", "parent", "negative"] *in order* to match ["anchor", "positive", "negative"]
    pair_dataset = load_hf_dataset(config.dataset_path, config.dataset_name + "-Pairs")
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=config.model_path, num_labels=2
    )

    # 2. Tokenise dataset and setup collator
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    tok_func = lambda example: tokenizer(example["child"], example["parent"], truncation=True, max_length=256)
    train_examples = pair_dataset["train"].map(tok_func, batched=True)
    val_examples = pair_dataset["val"].map(tok_func, batched=True)
    test_examples = pair_dataset["test"].map(tok_func, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 3. Define the training arguments
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=int(config.num_train_epochs),
        learning_rate=float(config.learning_rate),
        per_device_train_batch_size=int(config.train_batch_size),
        per_device_eval_batch_size=int(config.eval_batch_size),
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    # 4. Create the trainer & start training
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_examples,
        eval_dataset=val_examples,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()

    # 5. Evaluate the model performance on the test dataset
    test_preds = trainer.predict(test_examples)
    test_scores = torch.tensor(test_preds.predictions).argmax(dim=1)
    test_labels = torch.tensor(test_preds.label_ids)
    test_results = pd.DataFrame(columns=["threshold", "precision", "recall", "f1", "accuracy", "accuracy_on_negatives"])
    test_results.loc["testing"] = evaluate_by_threshold(scores=test_scores, labels=test_labels, threshold=0.5)
    logger.info(test_results.loc["testing"])
    create_path(os.path.join(output_dir, "eval"))
    test_results.to_csv(os.path.join(output_dir, "eval", "results.tsv"), sep="\t")


if __name__ == "__main__":
    main()
