from deeponto.utils import load_file, set_seed
from torch.utils.data import DataLoader
import logging
import numpy as np
import click
from yacs.config import CfgNode

from hite.model import *
from hite.loss import *
from hite.evaluation import HyperbolicLossEvaluator
from hite.utils import example_generator, load_hierarchy_dataset, get_device


logger = logging.getLogger(__name__)


@click.command()
@click.option("-c", "--config_file", type=click.Path(exists=True))
@click.option("-g", "--gpu_id", type=int, default=0)
def main(config_file: str, gpu_id: int):
    set_seed(8888)
    config = CfgNode(load_file(config_file))

    # load taxonomy and dataset
    data_path = config.data_path
    dataset, entity_lexicon = load_hierarchy_dataset(data_path)
    dataset = dataset[config.task]

    # load base edges for training
    base_examples = example_generator(
        entity_lexicon, dataset["train"], config.train.hard_negative_first, config.train.apply_triplet_loss
    )
    train_trans_portion = config.train.trans_train_portion
    train_examples = []
    if train_trans_portion > 0.0:
        logger.info(f"{train_trans_portion} transitivie edges used for training.")
        train_examples = example_generator(
            entity_lexicon, dataset["trans_train"], config.train.hard_negative_first, config.train.apply_triplet_loss
        )
        num_train_examples = int(train_trans_portion * len(train_examples))
        train_examples = list(np.random.choice(train_examples, size=num_train_examples, replace=False))
    else:
        logger.info("No transitivie edges used for training.")
    train_examples = base_examples + train_examples
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=config.train.train_batch_size)
    val_examples = example_generator(
        entity_lexicon, dataset["val"], config.train.hard_negative_first, config.train.apply_triplet_loss
    )
    val_dataloader = DataLoader(val_examples, shuffle=False, batch_size=config.train.eval_batch_size)
    test_examples = example_generator(
        entity_lexicon, dataset["test"], config.train.hard_negative_first, config.train.apply_triplet_loss
    )
    test_dataloader = DataLoader(test_examples, shuffle=False, batch_size=config.train.eval_batch_size)

    # load pre-trained model
    device = get_device(gpu_id)
    model = load_sentence_transformer(config.pretrained, device)

    # manifold
    embed_dim = model._first_module().get_word_embedding_dimension()
    manifold = get_manifold(embed_dim)
    # curvature = 1 / embed_dim if not config.apply_unit_ball_projection else 1.0

    # loss
    losses = []

    if config.train.loss.cluster.weight > 0.0:
        if config.train.apply_triplet_loss:
            cluster_loss = ClusteringTripletLoss(manifold, config.train.loss.cluster.margin)
        else:
            cluster_loss = ClusteringConstrastiveLoss(
                manifold, config.train.loss.cluster.positive_margin, config.train.loss.cluster.margin
            )
        losses.append((config.train.loss.cluster.weight, cluster_loss))

    if config.train.loss.centri.weight > 0.0:
        centri_loss_class = CentripetalTripletLoss if config.train.apply_triplet_loss else CentripetalContrastiveLoss
        centri_loss = centri_loss_class(manifold, embed_dim, config.train.loss.centri.margin)
        losses.append((config.train.loss.centri.weight, centri_loss))

    if config.train.loss.cone.weight > 0.0:
        cone_loss_class = (
            EntailmentConeTripletLoss if config.train.apply_triplet_loss else EntailmentConeConstrastiveLoss
        )
        cone_loss = cone_loss_class(manifold, config.train.loss.cone.min_euclidean_norm, config.train.loss.cone.margin)
        losses.append((config.train.loss.cone.weight, cone_loss))

    hyper_loss = HyperbolicLoss(model, config.train.apply_triplet_loss, *losses)
    print(hyper_loss.get_config_dict())
    hyper_loss.to(device)
    hyper_loss_evaluator = HyperbolicLossEvaluator(
        loss_module=hyper_loss,
        manifold=manifold,
        device=device,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        train_dataloader=train_dataloader if config.train.eval_train else None,
    )

    model.fit(
        train_objectives=[(train_dataloader, hyper_loss)],
        epochs=config.train.num_epochs,
        optimizer_params={"lr": float(config.train.learning_rate)},  # defaults to 2e-5
        # steps_per_epoch=20, # for testing use
        warmup_steps=config.train.warmup_steps,
        evaluator=hyper_loss_evaluator,
        output_path=f"experiments/{config.pretrained}-{config.task}-hard={config.train.hard_negative_first}-triplet={config.train.apply_triplet_loss}-train={train_trans_portion}-cluster={list(config.train.loss.cluster.values())}-centri={list(config.train.loss.centri.values())}",
    )


if __name__ == "__main__":
    main()
