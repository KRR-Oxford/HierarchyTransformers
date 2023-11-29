from sentence_transformers import SentenceTransformer
from deeponto.onto import Taxonomy
import seaborn as sns
import torch
from geoopt.manifolds import PoincareBall
import logging

logger = logging.getLogger(__name__)


def entity_norm_plot(hierarchy: Taxonomy, model: SentenceTransformer):
    entity_names = [hierarchy.get_node_attributes(e)["name"] for e in hierarchy.nodes]
    entity_embeds = model.encode(entity_names, 1024, True, convert_to_tensor=True)
    manifold = PoincareBall(c=1/model._first_module().get_word_embedding_dimension())
    entity_norms = manifold.dist0(entity_embeds)
    return entity_norms, sns.histplot(entity_norms, bins=10, kde=True, kde_kws={"bw_adjust": 2})
    

def entity_depths_plot(hierarchy: Taxonomy):
    if not hierarchy.root_node:
        logger.info("No root node detected; adding in edges from current top nodes to a pseudo root node.")
        top_nodes = []
        for n in hierarchy.nodes:
            if not hierarchy.get_parents(n):
                top_nodes.append(n)
        root = "owl:Thing"
        rooted_hierarchy = Taxonomy(hierarchy.edges + [(root, t) for t in top_nodes], root_node=root)
    else:
        rooted_hierarchy = hierarchy
    depths = []
    for n in hierarchy.nodes:
        depths.append(rooted_hierarchy.get_shortest_node_depth(n))
    return depths, sns.histplot(depths, bins=10, kde=True, kde_kws={"bw_adjust": 2})
