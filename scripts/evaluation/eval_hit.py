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

from deeponto.utils import load_file
import sys, logging, click
from yacs.config import CfgNode

from hierarchy_transformers.models import HierarchyTransformer
from hierarchy_transformers.evaluation import HierarchyTransformerEvaluator
from hierarchy_transformers.datasets import load_hf_dataset

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stderr)])
logger = logging.getLogger(__name__)


@click.command()
@click.option("-c", "--config_file", type=click.Path(exists=True))
@click.option("-o", "--output_path", type=click.Path(exists=True))
def main(config_file: str, output_path: str):

    # 0. load config
    config = CfgNode(load_file(config_file))

    # 1. Load dataset and pre-trained model
    # NOTE: according to docs, it is very important to have column names ["child", "parent", "negative"] *in order* to match ["anchor", "positive", "negative"]
    pair_dataset = load_hf_dataset(config.dataset_path, config.dataset_name + "-Pairs")
    model = HierarchyTransformer.from_pretrained(model_name_or_path=config.model_path, revision=config.revision)

    # 2. Run validation for hyerparameter selection
    val_evaluator = HierarchyTransformerEvaluator(
        child_entities=pair_dataset["val"]["child"],
        parent_entities=pair_dataset["val"]["parent"],
        labels=pair_dataset["val"]["label"],
        batch_size=config.eval_batch_size,
        truth_label=1,
    )
    val_evaluator(model=model, output_path=output_path, epoch="validation")

    # 3. Evaluate the model performance on the test dataset
    val_results = val_evaluator.results
    best_val = val_results.loc[val_results["f1"].idxmax()]
    best_val_centri_weight = float(best_val["centri_weight"])
    best_val_threshold = float(best_val["threshold"])
    test_evaluator = HierarchyTransformerEvaluator(
        child_entities=pair_dataset["test"]["child"],
        parent_entities=pair_dataset["test"]["parent"],
        labels=pair_dataset["test"]["label"],
        batch_size=config.eval_batch_size,
        truth_label=1,
    )
    test_evaluator(
        model=model,
        output_path=output_path,
        best_centri_weight=best_val_centri_weight,
        best_threshold=best_val_threshold,
    )


if __name__ == "__main__":
    main()
