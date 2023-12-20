import os
from datasets import Dataset, load_dataset
from sentence_transformers import InputExample
from tqdm.auto import tqdm
import json
from typing import Optional


def load_hierarchy_dataset(data_path: str, eval_only: bool = False):
    """Load hierarchy dataset and entity lexicon from:

    `data_dir`:
        `entity_lexicon`
        `transitivity`: `base.jsonl`, `train.jsonl`, `val.jsonl`, `test.jsonl`
        `completion`: `train.jsonl`, `val.jsonl`, `test.jsonl`
    """

    trans_task_name = "multi"
    trans_datafiles = {
        "train": os.path.join(data_path, trans_task_name, "train.jsonl"),
        "val": os.path.join(data_path, trans_task_name, "val.jsonl"),
        "test": os.path.join(data_path, trans_task_name, "test.jsonl"),
    }
    if eval_only:
        del trans_datafiles["train"]
    try:
        trans_dataset = load_dataset("json", data_files=trans_datafiles)
    except:
        trans_dataset = None
        print("No Multi-hop Inference dataset available.")

    pred_task_name = "mixed"
    pred_datafiles = {
        "train": os.path.join(data_path, pred_task_name, "train.jsonl"),
        "val": os.path.join(data_path, pred_task_name, "val.jsonl"),
        "test": os.path.join(data_path, pred_task_name, "test.jsonl"),
    }
    if eval_only:
        del pred_datafiles["train"]

    try:
        pred_dataset = load_dataset("json", data_files=pred_datafiles)
    except:
        pred_dataset = None
        print("No Mixed-hop Prediction dataset available.")

    with open(os.path.join(data_path, "entity_lexicon.json"), "r") as input:
        entity_lexicon = json.load(input)

    return {trans_task_name: trans_dataset, pred_task_name: pred_dataset}, entity_lexicon


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
