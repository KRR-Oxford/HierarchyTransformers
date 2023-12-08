from deeponto.utils import load_file, set_seed
from torch.utils.data import DataLoader
import logging
import numpy as np
import click
from yacs.config import CfgNode
from sentence_transformers.losses import SoftmaxLoss

from hierarchy_transformers.model import *
from hierarchy_transformers.evaluation import HyperbolicLossEvaluator
from hierarchy_transformers.utils import example_generator, load_hierarchy_dataset, get_device


logger = logging.getLogger(__name__)


@click.command()
@click.option("-c", "--config_file", type=click.Path(exists=True))
@click.option("-g", "--gpu_id", type=int, default=0)
def main(config_file: str, gpu_id: int):
    set_seed(8888)
    config = CfgNode(load_file(config_file))

    # load dataset
    data_path = config.data_path
    dataset, entity_lexicon = load_hierarchy_dataset(data_path)
    dataset = dataset[config.task]

    # load base edges for training
    base_examples = example_generator(entity_lexicon, dataset["train"], config.train.hard_negative_first, False)
    train_trans_portion = config.train.trans_train_portion
    train_examples = []
    if train_trans_portion > 0.0:
        logger.info(f"{train_trans_portion} transitivie edges used for training.")
        train_examples = example_generator(
            entity_lexicon, dataset["trans_train"], config.train.hard_negative_first, False
        )
        num_train_examples = int(train_trans_portion * len(train_examples))
        train_examples = list(np.random.choice(train_examples, size=num_train_examples, replace=False))
    else:
        logger.info("No transitivie edges used for training.")
    train_examples = base_examples + train_examples
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=config.train.train_batch_size)
    val_examples = example_generator(entity_lexicon, dataset["val"], config.train.hard_negative_first, False)
    val_dataloader = DataLoader(val_examples, shuffle=False, batch_size=config.train.eval_batch_size)
    test_examples = example_generator(entity_lexicon, dataset["test"], config.train.hard_negative_first, False)
    test_dataloader = DataLoader(test_examples, shuffle=False, batch_size=config.train.eval_batch_size)

    # load pre-trained model
    device = get_device(gpu_id)
    model = load_sentence_transformer(config.pretrained, device)
    embed_dim = model._first_module().get_word_embedding_dimension()
    manifold = get_manifold(embed_dim)
    # curvature = 1 / embed_dim if not config.apply_unit_ball_projection else 1.0

    # loss
    softmax_loss = SoftmaxLoss(model, embed_dim, num_labels=2)
    print(softmax_loss.get_config_dict())
    softmax_loss.to(device)
    # classification_evaluator = HyperbolicLossEvaluator(
    #     loss_module=hyper_loss,
    #     manifold=manifold,
    #     device=device,
    #     val_dataloader=val_dataloader,
    #     test_dataloader=test_dataloader,
    #     train_dataloader=train_dataloader if config.train.eval_train else None,
    # )

    model.fit(
        train_objectives=[(train_dataloader, softmax_loss)],
        epochs=config.train.num_epochs,
        optimizer_params={"lr": float(config.train.learning_rate)},  # defaults to 2e-5
        # steps_per_epoch=20, # for testing use
        warmup_steps=config.train.warmup_steps,
        # evaluator=classification_evaluator,
        output_path=f"experiments/{config.pretrained}-{config.task}-hard={config.train.hard_negative_first}-train={train_trans_portion}-finetune",
    )


if __name__ == "__main__":
    main()
