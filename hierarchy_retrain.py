from deeponto.utils import load_file, set_seed
from torch.utils.data import DataLoader
import logging
import numpy as np
import click
from yacs.config import CfgNode

from hierarchy_transformers.models import *
from hierarchy_transformers.losses import *
from hierarchy_transformers.evaluation import HierarchyRetrainedEvaluator
from hierarchy_transformers.utils import prepare_hierarchy_examples, load_hierarchy_dataset, get_torch_device


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

    # load examples from data splits
    trans_train_examples = None
    trans_train_portion = config.train.trans_train_portion
    if config.task == "transitivity":
        if trans_train_portion > 0.0:
            trans_train_examples = prepare_hierarchy_examples(
                entity_lexicon,
                dataset["trans_train"],
                config.train.apply_hard_negatives,
                config.train.apply_triplet_loss,
            )
            logger.info(f"{trans_train_portion} transitivie edges used for training.")
            num_trans_train_examples = int(trans_train_portion * len(trans_train_examples))
            trans_train_examples = list(
                np.random.choice(trans_train_examples, size=num_trans_train_examples, replace=False)
            )
        else:
            logger.info("No transitivie edges used for training.")
            
    train_examples = prepare_hierarchy_examples(
        entity_lexicon, dataset["train"], config.train.apply_hard_negatives, config.train.apply_triplet_loss
    )
    if trans_train_examples:
        train_examples = train_examples + trans_train_examples
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=config.train.train_batch_size)

    val_examples = prepare_hierarchy_examples(
        entity_lexicon, dataset["val"], config.train.apply_hard_negatives, config.train.apply_triplet_loss
    )
    test_examples = prepare_hierarchy_examples(
        entity_lexicon, dataset["test"], config.train.apply_hard_negatives, config.train.apply_triplet_loss
    )

    # load pre-trained model
    device = get_torch_device(gpu_id)
    model = load_pretrained(config.pretrained, device)

    # manifold
    embed_dim = model._first_module().get_word_embedding_dimension()
    manifold = get_circum_poincareball(embed_dim)
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

    hyper_loss = HyperbolicLoss(model, config.train.apply_triplet_loss, *losses)
    print(hyper_loss.get_config_dict())
    hyper_loss.to(device)
    hit_evaluator = HierarchyRetrainedEvaluator(
        loss_module=hyper_loss,
        manifold=manifold,
        device=device,
        eval_batch_size=config.train.eval_batch_size,
        val_examples=val_examples,
        test_examples=test_examples,
        train_examples=train_examples if config.train.eval_train else None,
    )

    model.fit(
        train_objectives=[(train_dataloader, hyper_loss)],
        epochs=config.train.num_epochs,
        optimizer_params={"lr": float(config.train.learning_rate)},  # defaults to 2e-5
        # steps_per_epoch=20, # for testing use
        warmup_steps=config.train.warmup_steps,
        evaluator=hit_evaluator,
        output_path=f"experiments/{config.pretrained}-{config.task}-hard={config.train.apply_hard_negatives}-triplet={config.train.apply_triplet_loss}-trans_train={trans_train_portion}-cluster={list(config.train.loss.cluster.values())}-centri={list(config.train.loss.centri.values())}",
    )


if __name__ == "__main__":
    main()
