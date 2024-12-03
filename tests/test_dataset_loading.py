# Copyright 2024 Yuan He

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import os
import random
from datasets import load_dataset


@pytest.fixture(params=os.getenv("DATASET_PATHS", "").split(","))
def dataset_path(request):
    # Ensure there are valid dataset names
    if not request.param:
        pytest.fail("No valid dataset names found in the DATASET_PATHS environment variable")
    return request.param.strip()  # Strip any extra spaces

def test_dataset_loading(dataset_path):
    
    hop_type = random.choice(["MultiHop", "MixedHop"])
    neg_type = random.choice(["HardNegatives", "RandomNegatives"])
    struct_type = random.choice(["Triplets", "Pairs"])
    part = f"{hop_type}-{neg_type}-{struct_type}"
    try:
        # Attempt to load the HierarchyTransformer model
        dataset = load_dataset(dataset_path, part)
    except Exception as e:
        pytest.fail(f"Dataset failed to load: {str(e)}")

    # Check that the datast is not None
    assert dataset is not None, "Loaded dataset is None"
    
    
# [Deprecated] HF upload code

# save_path = "WordNetNoun"
# hop_type = "MultiHop"
# for split in ["train", "val", "test"]:
#     n_rand = 0
#     n_hard = 0
#     triplet_examples = {
#         "random_negatives": [],
#         "hard_negatives": []
#     }
#     pair_examples = {
#         "random_negatives": [],
#         "hard_negatives": []
#     }
#     for sample in dataset[split]:
#         child = entity_lexicon[sample["child"]]["name"]
#         parent = entity_lexicon[sample["parent"]]["name"]
#         negative_parents = [entity_lexicon[neg]["name"] for neg in sample["random_negatives"]]
#         n_rand += len(negative_parents)
#         hard_negatives = [entity_lexicon[sib]["name"] for sib in sample["hard_negatives"]]
#         n_hard += len(hard_negatives)
#         triplet_examples["random_negatives"] += [(child, parent, neg) for neg in negative_parents]
#         triplet_examples["hard_negatives"] += [(child, parent, neg) for neg in hard_negatives]
#         pair_examples["random_negatives"] += [(child, parent, 1)] + [(child, neg, 0) for neg in negative_parents]
#         pair_examples["hard_negatives"] += [(child, parent, 1)] + [(child, neg, 0) for neg in hard_negatives]
#     assert n_rand == n_hard
#     pd.DataFrame(triplet_examples["random_negatives"], columns=["child", "parent", "negative"]).to_parquet(f"{save_path}/{hop_type}-RandomNegatives-Triplets/{split}.parquet", index=False)
#     pd.DataFrame(triplet_examples["hard_negatives"], columns=["child", "parent", "negative"]).to_parquet(f"{save_path}/{hop_type}-HardNegatives-Triplets/{split}.parquet", index=False)
#     pd.DataFrame(pair_examples["random_negatives"], columns=["child", "parent", "label"]).to_parquet(f"{save_path}/{hop_type}-RandomNegatives-Pairs/{split}.parquet", index=False)
#     pd.DataFrame(pair_examples["hard_negatives"], columns=["child", "parent", "label"]).to_parquet(f"{save_path}/{hop_type}-HardNegatives-Pairs/{split}.parquet", index=False)

# [Deprecated] Compare HF and local version
# data_path = "/home/yuan/projects/HiT/data/wordnet-multi"
# dataset, entity_lexicon = load_hierarchy_dataset(data_path)

# for split in ["train", "val", "test"]:
#     for is_hard in [True, False]:
#         for is_triplet in [True, False]:
#             dataset_zenodo = prepare_hierarchy_examples(entity_lexicon, dataset[split], is_hard, is_triplet)
#             if is_triplet:
#                 dataset_zenodo = [{'child': x.texts[0], 'parent': x.texts[1], 'negative': x.texts[2]} for x in dataset_zenodo] 
#             else: 
#                 dataset_zenodo = [{'child': x.texts[0], 'parent': x.texts[1], 'label': x.label} for x in dataset_zenodo]
#             neg = "HardNegatives" if is_hard else "RandomNegatives"
#             struct = "Triplets" if is_triplet else "Pairs"
#             dataset_hf = load_dataset("Hierarchy-Transformers/WordNetNoun", f"MultiHop-{neg}-{struct}")[split]
#             for i in tqdm(range(len(dataset_zenodo)), desc=f"Check MultiHop-{neg}-{struct}"):
#                 assert dataset_zenodo[i] == dataset_hf[i], (dataset_zenodo[i], dataset_hf[i])