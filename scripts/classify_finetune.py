from deeponto.utils import load_file, set_seed, save_file
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
import logging
import numpy as np
import click
from yacs.config import CfgNode

from hierarchy_transformers.utils import *
from hierarchy_transformers.evaluation import evaluate_by_threshold


logger = logging.getLogger(__name__)


@click.command()
@click.option("-c", "--config_file", type=click.Path(exists=True))
def main(config_file: str):
    set_seed(8888)
    config = CfgNode(load_file(config_file))

    # load dataset
    data_path = config.data_path
    dataset, entity_lexicon = load_hierarchy_dataset(data_path)

    train_examples = prepare_hierarchy_examples_for_finetune(
        entity_lexicon, dataset["train"], config.apply_hard_negatives
    )
    train_examples = Dataset.from_list(train_examples)

    val_examples = Dataset.from_list(
        prepare_hierarchy_examples_for_finetune(entity_lexicon, dataset["val"], config.apply_hard_negatives)
    )
    test_examples = Dataset.from_list(
        prepare_hierarchy_examples_for_finetune(entity_lexicon, dataset["test"], config.apply_hard_negatives)
    )

    # tokenise dataset
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained)
    tok_func = lambda example: tokenizer(example["entity1"], example["entity2"], truncation=True, max_length=256)
    train_examples = train_examples.map(tok_func, batched=True)
    val_examples = val_examples.map(tok_func, batched=True)
    test_examples = test_examples.map(tok_func, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # load pretrained model and do fine-tuning
    model_string = config.pretrained.split("/")[-1]
    data_suffix = config.data_path.split(os.path.sep)[-1]
    output_dir = f"experiments/Finetune-{model_string}-{data_suffix}-hard={config.apply_hard_negatives}"
    model = AutoModelForSequenceClassification.from_pretrained(config.pretrained, num_labels=2)
    train_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        num_train_epochs=config.num_epochs,
        evaluation_strategy="steps",  # "epoch",
        save_strategy="steps",  # "epoch",
        eval_steps=500,
        save_steps=500,
        logging_dir=f"{output_dir}/tensorboard",
        load_best_model_at_end=True,
        save_total_limit=2,
    )
    trainer = Trainer(
        model,
        train_args,
        train_dataset=train_examples,
        eval_dataset=val_examples,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()

    test_results = trainer.predict(test_examples)
    torch.save(test_results, f"{output_dir}/test_result_mat.pt")

    test_scores = torch.tensor(test_results.predictions).argmax(dim=1)
    test_labels = torch.tensor(test_results.label_ids)
    # no thresholding needed because of argmax
    test_results = evaluate_by_threshold(test_scores, test_labels, 0.0, False)
    save_file(test_results, f"{output_dir}/test_results.json")


if __name__ == "__main__":
    main()
