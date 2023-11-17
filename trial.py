from deeponto.onto import WordnetTaxonomy, TaxonomyNegativeSampler
from datasets import load_dataset, Features, Value
from typing import Dict, Iterable
import os
import torch
from torch.utils.data import DataLoader

torch.set_default_dtype(torch.float64)
from geoopt.manifolds import PoincareBall
from geoopt.tensor import ManifoldParameter
from geoopt.optim import RiemannianAdam, RiemannianSGD
from sentence_transformers import (
    SentenceTransformer,
    LoggingHandler,
    InputExample,
    SentencesDataset,
    models,
    losses,
)
import logging
import numpy as np

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

from holm.loss import HyperbolicLoss
from holm.evaluator import HyperbolicLossEvaluator


wt = WordnetTaxonomy()
data_path = "data/wordnet/trans"
trans_dataset = load_dataset(
    "json",
    data_files={
        "base": os.path.join(data_path, "base.jsonl"),
        "train": os.path.join(data_path, "train.jsonl"),
        "val": os.path.join(data_path, "val.jsonl"),
        "test": os.path.join(data_path, "test.jsonl"),
    },
)


def example_generator(dataset):
    examples = []
    for sample in dataset:
        child = wt.get_node_attributes(sample["child"])["name"]
        parent = wt.get_node_attributes(sample["parent"])["name"]
        negative_parents = [wt.get_node_attributes(neg)["name"] for neg in sample["negative_parents"]]
        examples.append(InputExample(texts=[child, parent], label=1))
        examples += [InputExample(texts=[child, neg], label=0) for neg in negative_parents]
    return examples


base_examples = example_generator(trans_dataset["base"])
base_dataloader = DataLoader(base_examples, shuffle=True, batch_size=256)

val_examples = example_generator(trans_dataset["val"])
val_dataloader = DataLoader(val_examples, shuffle=True, batch_size=1024)


gpu_id = 1
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
pretrained = SentenceTransformer("all-MiniLM-L6-v2", device=device)
modules = list(pretrained.modules())
model = SentenceTransformer(modules=[modules[1], modules[-2]], device=device)


cluster_centri_weights = {"cluster": 1.0, "centri": 1.0, "cone": 0.0}
cone_only_weights = {"cluster": 0.0, "centri": 0.0, "cone": 10.0}

curvature = 1 / modules[1].get_word_embedding_dimension()  # bigger ball
manifold = PoincareBall(c=curvature)
hyper_loss = HyperbolicLoss(model, manifold, cluster_centri_weights)
hyper_loss.to(device)
val_evaluator = HyperbolicLossEvaluator(val_dataloader, hyper_loss, device)
model.fit(
    train_objectives=[(base_dataloader, hyper_loss)],
    epochs=3,
    warmup_steps=500,
    evaluator=val_evaluator,
    output_path="experiments/trial.stage1=cluster+centri",
)

second_stage = False

if second_stage:
    hyper_loss = HyperbolicLoss(model, manifold, {"cluster": 0.0, "centri": 0.0, "cone": 10.0})
    hyper_loss.to(device)
    val_evaluator = HyperbolicLossEvaluator(val_dataloader, hyper_loss, device)
    model.fit(
        train_objectives=[(base_dataloader, hyper_loss)],
        epochs=3,
        warmup_steps=500,
        evaluator=val_evaluator,
        output_path="experiments/trial.stage1=cluster+centri.stage2=cone",
    )
