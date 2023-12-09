from deeponto.utils import load_file, set_seed
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


logger = logging.getLogger(__name__)


@click.command()
@click.option("-c", "--config_file", type=click.Path(exists=True))
@click.option("-g", "--gpu_id", type=int, default=0)
def main(config_file: str, gpu_id: int):
    set_seed(8888)
    config = CfgNode(load_file(config_file))

    # load dataset
    data_path = config.data_path
    dataset, entity_lexicon = load_hierarchy_dataset(data_path)
    dataset = dataset[config.task]

    # load base edges for training
    base_examples = prepare_hierarchy_examples_for_finetune(
        entity_lexicon, dataset["train"], config.train.hard_negative_first, config.train.apply_triplet_loss
    )
    train_trans_portion = config.train.trans_train_portion
    train_examples = []
    if train_trans_portion > 0.0:
        logger.info(f"{train_trans_portion} transitivie edges used for training.")
        train_examples = prepare_hierarchy_examples_for_finetune(
            entity_lexicon, dataset["trans_train"], config.train.hard_negative_first, config.train.apply_triplet_loss
        )
        num_train_examples = int(train_trans_portion * len(train_examples))
        train_examples = list(np.random.choice(train_examples, size=num_train_examples, replace=False))
    else:
        logger.info("No transitivie edges used for training.")
    train_examples = Dataset.from_list(base_examples + train_examples)
    val_examples = Dataset.from_list(
        prepare_hierarchy_examples_for_finetune(
            entity_lexicon, dataset["val"], config.train.hard_negative_first, config.train.apply_triplet_loss
        )
    )
    test_examples = Dataset.from_list(
        prepare_hierarchy_examples_for_finetune(
            entity_lexicon, dataset["test"], config.train.hard_negative_first, config.train.apply_triplet_loss
        )
    )

    # tokenise dataset
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained)
    tok_func = lambda example: tokenizer(example["entity1"], example["entity2"], truncation=True, max_length=256)
    train_examples = train_examples.map(tok_func, batched=True)
    val_examples = val_examples.map(tok_func, batched=True)
    test_examples = test_examples.map(tok_func, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # load pretrained model and do fine-tuning
    output_dir = f"experiments/{config.pretrained}-{config.task}-hard={config.train.hard_negative_first}-train={train_trans_portion}-finetune"
    model = AutoModelForSequenceClassification.from_pretrained(config.pretrained, num_labels=2)
    train_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=config.train.train_batch_size,
        per_device_eval_batch_size=config.train.eval_batch_size,
        num_train_epochs=config.train.num_epochs,
        evaluation_strategy="epoch",
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

    predictions = trainer.predict(test_examples)
    torch.save(predictions, f"{output_dir}/test_result_mat.pt")


if __name__ == "__main__":
    main()