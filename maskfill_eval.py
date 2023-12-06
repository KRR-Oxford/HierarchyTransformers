from deeponto.utils import load_file, set_seed, create_path
from torch.utils.data import DataLoader
import logging
import random
import click
from yacs.config import CfgNode

from hierarchy_transformers.evaluation import MaskFillEvaluator
from hierarchy_transformers.utils import anchored_example_generator, load_hierarchy_dataset, get_device


logger = logging.getLogger(__name__)


@click.command()
@click.option("-c", "--config_file", type=click.Path(exists=True))
@click.option("-g", "--gpu_id", type=int, default=0)
def main(config_file: str, gpu_id: int):
    # set_seed(8888)
    config = CfgNode(load_file(config_file))

    data_path = config.data_path
    dataset, entity_lexicon = load_hierarchy_dataset(data_path)
    dataset = dataset[config.task]
    val_examples = anchored_example_generator(entity_lexicon, dataset["val"])
    test_examples = anchored_example_generator(entity_lexicon, dataset["test"])

    device = get_device(gpu_id)
    mask_filler = MaskFillEvaluator(config.pretrained, device)
    output_path = f"experiments/{config.pretrained}-maskfill"
    create_path(output_path)
    mask_filler(val_examples, test_examples, output_path)


if __name__ == "__main__":
    main()
