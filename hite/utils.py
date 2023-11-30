import os
import torch
from datasets import Dataset, load_dataset
from sentence_transformers import InputExample
from tqdm.auto import tqdm



def get_device(gpu_id: int):
    return torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")


def load_hierarchy_dataset(data_path: str):
    """Load hierarchy dataset and entity lexicon.
    """

    dataset = load_dataset(
        "json",
        data_files={
            "train_base": os.path.join(data_path, "train_base.jsonl"),
            "train_trans": os.path.join(data_path, "train_trans.jsonl"),
            "val": os.path.join(data_path, "val.jsonl"),
            "test": os.path.join(data_path, "test.jsonl"),
        },
    )
    
    entity_data = load_dataset("json", data_files={"lexicon": os.path.join(data_path, "..", "entities.jsonl")})

    entity_lexicon = dict()
    for ent in entity_data["lexicon"]:
        entity_lexicon[ent["id"]] = {
            "name": ent["name"],
            "definition": ent["definition"],
        }
        
    return dataset, entity_lexicon


def example_generator(
    entity_lexicon: dict, dataset: Dataset, hard_negative_first: bool = False, in_triplets: bool = False
):
    """Prepare examples in different formats.

    Args:
        entity_lexicon (dict): A lexicon that can provide names for entities.
        dataset (Dataset): Input dataset to be formatted.
        hard_negative_first (bool, optional): Using hard negative samples (siblings) or not. Defaults to `False`.
        in_triplets (bool, optional): Present in triplets or not. Defaults to `False`.
    """
    examples = []
    for sample in tqdm(dataset, leave=True, desc=f"Prepare examples for {dataset.split._name}"):
        child = entity_lexicon[sample["child"]]["name"]
        parent = entity_lexicon[sample["parent"]]["name"]
        negative_parents = [entity_lexicon[neg]["name"] for neg in sample["random_negatives"]]
        siblings = [entity_lexicon[sib]["name"] for sib in sample["hard_negatives"]]
        if hard_negative_first:
            # extract siblings first, if not enough, add the random negative parents
            negative_parents = (siblings + negative_parents)[:10]

        if not in_triplets:
            examples.append(InputExample(texts=[child, parent], label=1))
            examples += [InputExample(texts=[child, neg], label=0) for neg in negative_parents]
        else:
            examples += [InputExample(texts=[child, parent, neg]) for neg in negative_parents]
    return examples


def static_example_generator(ent2idx: dict, dataset: Dataset, hard_negative_first: bool = False):
    examples = []
    for sample in dataset:
        negative_parents = sample["random_negatives"]
        siblings = sample["hard_negatives"]
        if hard_negative_first:
            negative_parents = (siblings + negative_parents)[:10]
        cur_example = [sample["child"], sample["parent"]] + negative_parents
        cur_example = [ent2idx[x] for x in cur_example]
        examples.append(cur_example)
    return examples
