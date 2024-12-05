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

import os
import json
from typing import Optional
from datasets import load_dataset, Dataset
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_hf_dataset(path: str, name: Optional[str] = None, **config_kwargs):
    """Load a HiT dataset from Hugging Face.

    See available datasets on: https://huggingface.co/Hierarchy-Transformers

    Args:
        path (str): Dataset path on Hugging Face.
        name (Optional[str]): Name of a specific subset if any. Defaults to `None`.
    """
    return load_dataset(path, name, **config_kwargs)


def load_zenodo_dataset(
    path: str,
    negative_type: str = "random",
    example_type: str = "triplet",
    entity_to_index: Optional[dict] = None,
    return_entity_lexicon: bool = False,
):
    """Load a HiT dataset from a local version downloaded from Zenodo.

    It is recommended to use `load_hf_dataset` from this library or `load_dataset` from HuggingFace datasets if one doesn't require the original entity IDs.

    See available datasets on: https://doi.org/10.5281/zenodo.10511042

    Args:
        path (str): Path to a local dataset downloaded from Zenodo.
        negative_type (str): Type of negative examples. Options are `['random', 'hard']`.
        example_type (str): Type of example structure. Options are `['triplet', 'pair', 'idx']`.
        entity_to_index (Optional[dict], optional): A dictionary to transform entity IDs to indices (one-hot encoding) required by the static hierarchy models. Defaults to `None`.
        return_entity_lexicon (bool, optional): Whether or not to return the lexicon of entity IDs to their names (and potentially other textual information). Defaults to `False`.
    """
    
    assert negative_type in ["random", "hard"], f"Unknown negative type '{negative_type}'."
    assert example_type in ["triplet", "pair", "idx"], f"Unknown example type '{example_type}'."
    if example_type == "idx":
        assert entity_to_index is not None, "The `ent2idx` dictionary is not found."

    # check if train, val, test splits are all there
    datafiles = dict()
    for split in ["train", "val", "test"]:
        split_path = os.path.join(path, f"{split}.jsonl")
        if os.path.isfile(split_path):
            datafiles[split] = split_path
        else:
            logger.info(f"No {split} split available.")

    # load the jsonl dataset altogther
    dataset = load_dataset("json", data_files=datafiles)

    # load the entity lexicon for id to names
    with open(os.path.join(path, "entity_lexicon.json"), "r") as input:
        entity_lexicon = json.load(input)

    transform = {
        "triplet": zenodo_example_to_triplets,
        "pair": zenodo_example_to_pairs,
        "idx": zenodo_example_to_idxs,
    }[example_type]

    if entity_to_index:
        entity_lexicon = entity_to_index

    for split, examples in dataset.items():
        # list comprehension is faster than nested for-loop due to C implementation
        dataset[split] = Dataset.from_list(
            [
                transformed
                for example in tqdm(examples, desc=f"Map ({split})", leave=True)
                for transformed in transform(example, negative_type, entity_lexicon)
            ]
        )

    if return_entity_lexicon:
        return dataset, entity_lexicon
    else:
        return dataset


def zenodo_example_to_triplets(example: dict, negative_type: str, entity_lexicon: dict):
    """Helper function to present Zenodo dataset examples into triplets of the form `(child, parent, negative)`."""
    
    child = entity_lexicon[example["child"]]["name"]
    parent = entity_lexicon[example["parent"]]["name"]
    negative_type = f"{negative_type}_negatives"
    negative_parents = [entity_lexicon[neg]["name"] for neg in example[negative_type]]
    return [{"child": child, "parent": parent, "negative": neg} for neg in negative_parents]


def zenodo_example_to_pairs(example: dict, negative_type: str, entity_lexicon: dict):
    """Helper function to present Zenodo dataset examples into labelled pairs of the form `(child, parent, label)`."""
    
    child = entity_lexicon[example["child"]]["name"]
    parent = entity_lexicon[example["parent"]]["name"]
    negative_type = f"{negative_type}_negatives"
    negative_parents = [entity_lexicon[neg]["name"] for neg in example[negative_type]]
    return [{"child": child, "parent": parent, "label": 1}] + [
        {"child": child, "parent": neg, "label": 0} for neg in negative_parents
    ]


def zenodo_example_to_idxs(example: dict, negative_type: str, entity_to_indices: dict):
    """Helper function to present Zenodo dataset examples into an entity index list of `(child_idx, paren_idx, *negative_idxs)`."""
    
    child = entity_to_indices[example["child"]]
    parent = entity_to_indices[example["parent"]]
    negative_type = f"{negative_type}_negatives"
    negative_parents = [entity_to_indices[neg] for neg in example[negative_type]]
    return [child, parent] + negative_parents
