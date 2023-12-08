import torch
from torch.utils.data import DataLoader
import logging
import random
import click
from yacs.config import CfgNode
from deeponto.utils import load_file, set_seed, create_path

from hierarchy_transformers.models import StaticPoincareEmbed, StaticPoincareEmbedTrainer
from hierarchy_transformers.utils import prepare_hierarchy_examples_for_static, load_hierarchy_dataset, get_torch_device


logger = logging.getLogger(__name__)


@click.command()
@click.option("-c", "--config_file", type=click.Path(exists=True))
@click.option("-g", "--gpu_id", type=int, default=0)
def main(config_file: str, gpu_id: int):
    # set_seed(8888)
    config = CfgNode(load_file(config_file))

    # load dataset
    data_path = config.data_path
    dataset, entity_lexicon = load_hierarchy_dataset(data_path)
    dataset = dataset[config.task]

    # init static poincare embedding
    model = StaticPoincareEmbed(list(entity_lexicon.keys()), embed_dim=config.train.embed_dim)
    print(model)
    ent2idx = model.ent2idx

    # load base edges for training
    base_examples = prepare_hierarchy_examples_for_static(ent2idx, dataset["train"], config.train.hard_negative_first)
    train_trans_portion = config.train.train_trans_portion
    train_examples = []
    if train_trans_portion > 0.0:
        logger.info(f"{train_trans_portion} transitivie edges used for training.")
        train_examples = prepare_hierarchy_examples_for_static(ent2idx, dataset["trans_train"], config.train.hard_negative_first)
        num_train_examples = int(train_trans_portion * len(train_examples))
        train_examples = list(random.sample(train_examples, k=num_train_examples))
    else:
        logger.info("No transitivie edges used for training.")
    train_examples = base_examples + train_examples
    train_dataloader = DataLoader(torch.tensor(train_examples), shuffle=True, batch_size=config.train.train_batch_size)

    device = get_torch_device(gpu_id)
    static_trainer = StaticPoincareEmbedTrainer(
        model=model,
        device=device,
        train_dataloader=train_dataloader,
        learning_rate=config.train.learning_rate,
        num_epochs=config.train.num_epochs,
        num_warmup_epochs=config.train.num_warmup_epochs,
    )
    output_path = "experiments/static_poincare"
    create_path(output_path)
    static_trainer.run(output_path)


if __name__ == "__main__":
    main()
