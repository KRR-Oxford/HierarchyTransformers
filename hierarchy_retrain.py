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

    train_examples = prepare_hierarchy_examples(
        entity_lexicon, dataset["train"], config.apply_hard_negatives, config.apply_triplet_loss
    )
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=config.train_batch_size)

    val_examples = prepare_hierarchy_examples(entity_lexicon, dataset["val"], config.apply_hard_negatives)
    test_examples = prepare_hierarchy_examples(entity_lexicon, dataset["test"], config.apply_hard_negatives)

    # load pre-trained model
    device = get_torch_device(gpu_id)
    model = HierarchyTransformer.load_pretrained(config.pretrained, device)

    # loss
    losses = []

    if config.loss.cluster.weight > 0.0:
        if config.apply_triplet_loss:
            cluster_loss = ClusteringTripletLoss(model.manifold, config.loss.cluster.margin)
        else:
            cluster_loss = ClusteringConstrastiveLoss(
                model.manifold, config.loss.cluster.positive_margin, config.loss.cluster.margin
            )
        losses.append((config.loss.cluster.weight, cluster_loss))

    if config.loss.centri.weight > 0.0:
        centri_loss_class = CentripetalTripletLoss if config.apply_triplet_loss else CentripetalContrastiveLoss
        centri_loss = centri_loss_class(model.manifold, model.embed_dim, config.loss.centri.margin)
        losses.append((config.loss.centri.weight, centri_loss))

    hyper_loss = HyperbolicLoss(model, config.apply_triplet_loss, *losses)
    print(hyper_loss.get_config_dict())
    hyper_loss.to(device)
    hit_evaluator = HierarchyRetrainedEvaluator(
        manifold=model.manifold,
        device=device,
        eval_batch_size=config.eval_batch_size,
        val_examples=val_examples,
        test_examples=test_examples,
        train_examples=train_examples if config.eval_train else None,
    )

    model.fit(
        train_objectives=[(train_dataloader, hyper_loss)],
        epochs=config.num_epochs,
        optimizer_params={"lr": float(config.learning_rate)},  # defaults to 2e-5
        # steps_per_epoch=20,  # for testing use
        warmup_steps=config.warmup_steps,
        evaluator=hit_evaluator,
        output_path=f"experiments/HiT-{config.pretrained}-hard={config.apply_hard_negatives}-triplet={config.apply_triplet_loss}-cluster={list(config.loss.cluster.values())}-centri={list(config.loss.centri.values())}",
    )


if __name__ == "__main__":
    main()
