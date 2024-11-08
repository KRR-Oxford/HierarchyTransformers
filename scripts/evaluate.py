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
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
import click
from yacs.config import CfgNode
from deeponto.utils import load_file, save_file, create_path
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


from hierarchy_transformers.utils import *
from hierarchy_transformers.evaluation import *
from hierarchy_transformers.models.hit import HierarchyTransformer


@click.command()
@click.option("-c", "--config_file", type=click.Path(exists=True))
@click.option("-g", "--gpu_id", type=int, default=0)
def main(config_file: str, gpu_id: int):
    # set_seed(8888)
    config = CfgNode(load_file(config_file))

    data_path = config.data_path
    data_suffix = config.data_path.split(os.path.sep)[-1]
    dataset, entity_lexicon = load_hierarchy_dataset(data_path, eval_only=True)

    prepare_func = prepare_hierarchy_examples
    if config.model_type == "finetune":
        prepare_func = prepare_hierarchy_examples_for_finetune
    val_examples = prepare_func(entity_lexicon, dataset["val"], config.apply_hard_negatives)
    test_examples = prepare_func(entity_lexicon, dataset["test"], config.apply_hard_negatives)

    if config.model_type == "hit":
        device = get_torch_device(gpu_id)
        model = HierarchyTransformer(
            config.pretrained,
            device=device,
        )
        val_result_mat = HierarchyTransformerEvaluator.encode(model, val_examples, config.eval_batch_size)
        val_results = HierarchyTransformerEvaluator.search_best_threshold(val_result_mat)
        save_file(val_results, f"{config.pretrained}/{data_suffix}-val_results.hard={config.apply_hard_negatives}.json")
        test_result_mat = HierarchyTransformerEvaluator.encode(model, test_examples, config.eval_batch_size)
        test_scores = test_result_mat[:, 1] + val_results["centri_score_weight"] * (
            test_result_mat[:, 3] - test_result_mat[:, 2]
        )
        test_results = evaluate_by_threshold(test_scores, test_result_mat[:, 0], val_results["threshold"])
        save_file(
            test_results, f"{config.pretrained}/{data_suffix}-test_results.hard={config.apply_hard_negatives}.json"
        )

    elif config.model_type == "finetune":
        model = AutoModelForSequenceClassification.from_pretrained(config.pretrained)
        tokenizer = AutoTokenizer.from_pretrained(config.pretrained)
        tok_func = lambda example: tokenizer(example["entity1"], example["entity2"], truncation=True, max_length=256)
        train_args = TrainingArguments(".", per_device_eval_batch_size=config.eval_batch_size)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        trainer = Trainer(model, train_args, data_collator=data_collator, tokenizer=tokenizer)

        test_examples = Dataset.from_list(test_examples).map(tok_func, batched=True)
        test_preds = trainer.predict(test_examples)
        test_scores = torch.tensor(test_preds.predictions).argmax(dim=1)
        test_labels = torch.tensor(test_preds.label_ids)
        test_results = evaluate_by_threshold(test_scores, test_labels, 0.0, False)
        save_file(
            test_results, f"{config.pretrained}/../{data_suffix}-test_results.hard={config.apply_hard_negatives}.json"
        )

    elif config.model_type == "simeval":
        device = get_torch_device(gpu_id)
        sim_eval = PretrainedSentenceSimilarityEvaluator(
            config.pretrained, device, config.eval_batch_size, val_examples, test_examples
        )
        output_path = f"experiments/SimEval-{config.pretrained}-{data_suffix}-hard={config.apply_hard_negatives}"
        create_path(output_path)
        sim_eval(output_path)

    elif config.model_type == "maskfill":
        device = get_torch_device(gpu_id)
        mask_filler = PretrainedMaskFillEvaluator(
            config.pretrained, device, config.train.eval_batch_size, val_examples, test_examples
        )
        output_path = f"experiments/MaskFill-{config.pretrained}-{data_suffix}-hard={config.train.apply_hard_negatives}"
        create_path(output_path)
        mask_filler(output_path)

    print(test_results)
    print(
        " & ".join(
            [str(round(test_results["P"], 3)), str(round(test_results["R"], 3)), str(round(test_results["F1"], 3))]
        )
    )


if __name__ == "__main__":
    main()
