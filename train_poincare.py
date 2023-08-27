import torch
torch.set_default_dtype(torch.float64)
from geoopt.optim import RiemannianAdam
from tqdm import tqdm

from hypertrans.poincare import PoincareBallModel
from hypertrans.graph import HypernymGraph
from hypertrans import PoincareTrainer

# graph = HypernymGraph("data/wordnet/wordnet_hypernyms.tsv")
# trainer = PoincareTrainer(graph, batch_size=1024, n_epochs=50, n_warmup_epochs=10, gpu_device=1)
# trainer.run()
# torch.save(trainer.model, "experiments/poincare.mammal.pt")

graph = HypernymGraph("data/wordnet/wordnet_mammal_hypernyms.tsv")
trainer = PoincareTrainer(graph, batch_size=64, n_epochs=200, n_warmup_epochs=20, gpu_device=1)
trainer.run()
torch.save(trainer.model, "experiments/poincare.mammal.pt")
