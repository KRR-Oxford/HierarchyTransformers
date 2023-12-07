from deeponto.utils import load_file, set_seed, create_path
from torch.utils.data import DataLoader
import logging
import random
import click
from yacs.config import CfgNode

from hierarchy_transformers.evaluation import SentenceSimilarityEvaluator
from hierarchy_transformers.utils import example_generator, load_hierarchy_dataset, get_device


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
    val_examples = example_generator(entity_lexicon, dataset["val"], config.train.hard_negative_first)
    test_examples = example_generator(entity_lexicon, dataset["test"], config.train.hard_negative_first)

    device = get_device(gpu_id)
    sim_eval = SentenceSimilarityEvaluator(config.pretrained, device)
    output_path = f"experiments/{config.pretrained}-{config.task}-hard={config.train.hard_negative_first}-simeval"
    create_path(output_path)
    sim_eval(val_examples, test_examples, output_path, config.train.eval_batch_size)


if __name__ == "__main__":
    main()
