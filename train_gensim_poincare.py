from hypertrans.graph import HypernymGraph
from gensim.models.poincare import PoincareModel
import logging

logging.basicConfig(level=logging.INFO)


for dim in [50, 100, 200]:
    wordnet_graph = HypernymGraph("data/wordnet/wordnet_mammal_hypernyms.tsv")
    gensim_model = PoincareModel(wordnet_graph.edges, size=dim, burn_in=10, negative=10)
    gensim_model.train(epochs=200, batch_size=50)
    gensim_model.save(f"experiments/gensim/mammal.{dim}d.model")

for dim in [50, 100, 200]:
    wordnet_graph = HypernymGraph("data/wordnet/wordnet_hypernyms.tsv")
    gensim_model = PoincareModel(wordnet_graph.edges, size=dim, burn_in=10, negative=10)
    gensim_model.train(epochs=200, batch_size=50)
    gensim_model.save(f"experiments/gensim/wordnet.{dim}d.model")
