# Copyright 2023 Yuan He. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Optional, Tuple, List, Dict, Union
import pandas as pd
import networkx as nx
import numpy as np
import torch
from yacs.config import CfgNode
from sklearn.model_selection import train_test_split
from deeponto.onto import Taxonomy, OntologyTaxonomy, WordnetTaxonomy, TaxonomyNegativeSampler


def transitivity_data_splitting(taxonomy: Taxonomy):
    # base edges must be included in the training data for the transitivity setting
    base_edges = taxonomy.edges
    all_edges = []
    for node in taxonomy.nodes:
        all_edges += list(map(lambda p: (node, p), taxonomy.get_parents(node, apply_transitivity=True)))
    transitive_edges = list(set(all_edges) - set(base_edges))
    train_edges, eval_edges = train_test_split(transitive_edges, test_size=0.2)
    val_edges, test_edges = train_test_split(eval_edges, test_size=0.5)
    return CfgNode({"base": base_edges, "train": train_edges, "val": val_edges, "test": test_edges})
