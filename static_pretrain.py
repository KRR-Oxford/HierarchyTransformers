import torch
from torch.utils.data import DataLoader
import logging
import random
import click
from yacs.config import CfgNode
from deeponto.utils import load_file, set_seed, create_path

from hierarchy_transformers.models import StaticPoincareEmbed, StaticPoincareEmbedTrainer
from hierarchy_transformers.utils import (
    prepare_hierarchy_examples_for_static,
    load_hierarchy_dataset,
    get_torch_device,
)
from hierarchy_transformers.evaluation import StaticPoincareEvaluator


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

    # init static poincare embedding
    model = StaticPoincareEmbed(list(entity_lexicon.keys()), embed_dim=config.embed_dim)
    print(model)
    ent2idx = model.ent2idx

    train_examples = prepare_hierarchy_examples_for_static(ent2idx, dataset["train"], config.apply_hard_negatives)
    train_dataloader = DataLoader(torch.tensor(train_examples), shuffle=True, batch_size=config.train_batch_size)

    device = get_torch_device(gpu_id)
    static_trainer = StaticPoincareEmbedTrainer(
        model=model,
        device=device,
        train_dataloader=train_dataloader,
        learning_rate=float(config.learning_rate),
        num_epochs=config.num_epochs,
        num_warmup_epochs=config.warmup_epochs,
    )
    output_path = f"experiments/static_poincare-hard={config.apply_hard_negatives}"
    create_path(output_path)
    static_trainer.run(output_path)

    val_examples = prepare_hierarchy_examples_for_static(ent2idx, dataset["val"], config.apply_hard_negatives)
    test_examples = prepare_hierarchy_examples_for_static(ent2idx, dataset["test"], config.apply_hard_negatives)
    static_eval = StaticPoincareEvaluator(
        model_path=f"{output_path}/poincare.{model.embed_dim}d.pt",
        device=device,
        val_examples=val_examples,
        test_examples=test_examples,
        eval_batch_size=config.eval_batch_size,
    )
    static_eval(output_path)


if __name__ == "__main__":
    main()
