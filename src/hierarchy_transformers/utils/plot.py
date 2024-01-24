# Copyright 2023 Yuan He

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    manifold = PoincareBall(c=1 / model._first_module().get_word_embedding_dimension())
    entity_norms = manifold.dist0(entity_embeds)
    return (
        entity_embeds,
        entity_norms,
        sns.histplot(entity_norms.cpu().numpy(), bins=10, kde=True, kde_kws={"bw_adjust": 2}),
    )


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
