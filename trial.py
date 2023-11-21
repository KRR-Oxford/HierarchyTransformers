from deeponto.onto import WordnetTaxonomy, OntologyTaxonomy
from deeponto.utils import load_file, print_dict, set_seed
from datasets import load_dataset
import os
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
import logging
import numpy as np
import click
from yacs.config import CfgNode

from helm.loss import HyperbolicLoss
from helm.evaluator import HyperbolicLossEvaluator
from helm.utils import example_generator


logger = logging.getLogger(__name__)

# @click.command()
# @click.option("-c", "--config_file", type=click.Path(exists=True))
# @click.option("-g", "--gpu_id", type=int, default=0)
# def main(config_file: str, gpu_id: int):

# set_seed(8888)
config_file = "./config.yaml"
gpu_id = 1
config = CfgNode(load_file(config_file))

# load taxonomy and dataset
wt = WordnetTaxonomy()
data_path = config.data_path
trans_dataset = load_dataset(
    "json",
    data_files={
        "base": os.path.join(data_path, "base.jsonl"),
        "train": os.path.join(data_path, "train.jsonl"),
        "val": os.path.join(data_path, "val.jsonl"),
        "test": os.path.join(data_path, "test.jsonl"),
    },
)

# load base edges for training
base_examples = example_generator(wt, trans_dataset["base"])
train_portion = config.train.trans_train_portion
train_examples = []
if train_portion > 0.0:
    logger.info(f"{train_portion} transitivie edges used for training.")
    train_examples = example_generator(wt, trans_dataset["train"])
    num_train_examples = int(train_portion * len(train_examples))
    train_examples = list(np.random.choice(train_examples, size=num_train_examples, replace=False))
else:
    logger.info("No transitivie edges used for training.")
train_examples = base_examples + train_examples
train_dataloaer = DataLoader(train_examples, shuffle=True, batch_size=config.train.train_batch_size)
val_examples = example_generator(wt, trans_dataset["val"])
val_dataloader = DataLoader(val_examples, shuffle=True, batch_size=config.train.eval_batch_size)


device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
pretrained = SentenceTransformer(config.pretrained, device=device)
modules = list(pretrained.modules())
model = SentenceTransformer(modules=[modules[1], modules[-2]], device=device)
print(model)

hyper_loss = HyperbolicLoss(
    model=model,
    training_mode=config.train.training_mode,
    loss_weights={
        "cluster": config.train.loss.cluster.weight, 
        "centri": config.train.loss.centri.weight, 
        "cone": config.train.loss.cone.weight,
    },
    min_distance=config.train.loss.cluster.min_distance,
    cluster_loss_margin=config.train.loss.cluster.margin,
    centri_loss_margin=config.train.loss.centri.margin,
    min_euclidean_norm=config.train.loss.cone.min_euclidean_norm,
    cone_loss_margin=config.train.loss.cone.margin,
)
print(print_dict(hyper_loss.get_config_dict()))
hyper_loss.to(device)
val_evaluator = HyperbolicLossEvaluator(val_dataloader, hyper_loss, device)

model.fit(
    train_objectives=[(train_dataloaer, hyper_loss)],
    epochs=config.train.num_epochs,
    optimizer_params={"lr": float(config.train.learning_rate)},  # defaults to 2e-5
    steps_per_epoch=5,
    warmup_steps=config.train.warmup_steps,
    evaluator=val_evaluator,
    output_path=f"experiments/train={train_portion}-cluster={list(config.train.loss.cluster.values())}-centri={list(config.train.loss.centri.values())}-cone={list(config.train.loss.cone.values())}",
    # output_path="experiments/trial.train=0.2.stage1=cluster+centri",
)
