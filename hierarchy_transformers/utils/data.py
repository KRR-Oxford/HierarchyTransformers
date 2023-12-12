import os
from datasets import Dataset, load_dataset
from sentence_transformers import InputExample
from tqdm.auto import tqdm
import json
from typing import Optional


def load_hierarchy_dataset(data_path: str):
    """Load hierarchy dataset and entity lexicon from:

    `data_dir`:
        `entity_lexicon`
        `transitivity`: `base.jsonl`, `train.jsonl`, `val.jsonl`, `test.jsonl`
        `completion`: `train.jsonl`, `val.jsonl`, `test.jsonl`
    """

    trans_dataset = load_dataset(
        "json",
        data_files={
            "train": os.path.join(data_path, "transitivity", "train.jsonl"),
            "trans_train": os.path.join(data_path, "transitivity", "trans_train.jsonl"),
            "val": os.path.join(data_path, "transitivity", "val.jsonl"),
            "test": os.path.join(data_path, "transitivity", "test.jsonl"),
        },
    )

    completion_dataset = load_dataset(
        "json",
        data_files={
            "train": os.path.join(data_path, "completion", "train.jsonl"),
            "val": os.path.join(data_path, "completion", "val.jsonl"),
            "test": os.path.join(data_path, "completion", "test.jsonl"),
        },
    )

    with open(os.path.join(data_path, "entity_lexicon.json"), "r") as input:
        entity_lexicon = json.load(input)

    return {"transitivity": trans_dataset, "completion": completion_dataset}, entity_lexicon


def prepare_hierarchy_examples(
    entity_lexicon: dict, dataset: Dataset, apply_hard_negatives: bool = False, in_triplets: bool = False
):
    """Prepare examples in different formats.

    Args:
        entity_lexicon (dict): A lexicon that can provide names for entities.
        dataset (Dataset): Input dataset to be formatted.
        apply_hard_negatives (bool, optional): Using hard negative samples (siblings) or not. Defaults to `False`.
        in_triplets (bool, optional): Present in triplets or not. Defaults to `False`.
    """
    examples = []
    for sample in tqdm(dataset, leave=True, desc=f"Prepare examples from {dataset.split._name}"):
        child = entity_lexicon[sample["child"]]["name"]
        parent = entity_lexicon[sample["parent"]]["name"]
        negative_parents = [entity_lexicon[neg]["name"] for neg in sample["random_negatives"]]
        hard_negatives = [entity_lexicon[sib]["name"] for sib in sample["hard_negatives"]]
        if apply_hard_negatives:
            # extract siblings first, if not enough, add the random negative parents
            negative_parents = (hard_negatives + negative_parents)[:10]

        if not in_triplets:
            examples.append(InputExample(texts=[child, parent], label=1))
            examples += [InputExample(texts=[child, neg], label=0) for neg in negative_parents]
        else:
            examples += [InputExample(texts=[child, parent, neg]) for neg in negative_parents]
    return examples


def prepare_hierarchy_examples_for_static(
    ent2idx: dict, dataset: Dataset, apply_hard_negatives: bool = False, **kwargs
):
    examples = []
    for sample in tqdm(dataset, leave=True, desc=f"Prepare examples from {dataset.split._name}"):
        negative_parents = sample["random_negatives"]
        hard_negatives = sample["hard_negatives"]
        if apply_hard_negatives:
            negative_parents = (hard_negatives + negative_parents)[:10]
        cur_example = [sample["child"], sample["parent"]] + negative_parents
        cur_example = [ent2idx[x] for x in cur_example]
        examples.append(cur_example)
    return examples


def prepare_hierarchy_examples_for_finetune(
    entity_lexicon: dict, dataset: Dataset, apply_hard_negatives: bool = False, **kwargs
):
    """Prepare examples for fine-tuning.

    Args:
        entity_lexicon (dict): A lexicon that can provide names for entities.
        dataset (Dataset): Input dataset to be formatted.
        apply_hard_negatives (bool, optional): Using hard negative samples (siblings) or not. Defaults to `False`.
    """
    examples = []
    for sample in tqdm(dataset, leave=True, desc=f"Prepare examples from {dataset.split._name}"):
        child = entity_lexicon[sample["child"]]["name"]
        parent = entity_lexicon[sample["parent"]]["name"]
        negative_parents = [entity_lexicon[neg]["name"] for neg in sample["random_negatives"]]
        hard_negatives = [entity_lexicon[sib]["name"] for sib in sample["hard_negatives"]]
        if apply_hard_negatives:
            # extract siblings first, if not enough, add the random negative parents
            negative_parents = (hard_negatives + negative_parents)[:10]

        examples.append({"entity1": child, "entity2": parent, "label": 1})
        examples += [{"entity1": child, "entity2": neg, "label": 0} for neg in negative_parents]

    return examples
