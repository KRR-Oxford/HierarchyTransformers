<!---
Copyright 2023 Yuan He. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<p align="center">
  <a href="https://krr-oxford.github.io/DeepOnto/">
    <img alt="deeponto" src="https://raw.githubusercontent.com/KRR-Oxford/HierarchyTransformers/main/docs/images/hit_logo.png" style="height: 6em;">
  </a>
</p>


<p align="center">
    <a href="https://github.com/KRR-Oxford/HierarchyTransformers/blob/main/LICENSE">
        <img alt="license" src="https://img.shields.io/github/license/KRR-Oxford/HierarchyTransformers">
    </a>
    <a href="https://huggingface.co/Hierarchy-Transformers">
        <img alt="docs" src="https://img.shields.io/badge/website-online-informational">
    </a>
    <a href="https://pypi.org/project/hierarchy_transformers/">
        <img alt="pypi" src="https://img.shields.io/pypi/v/deeponto">
    </a>
</p>

<p align="center">
  Embedding hierarchies with language models.
</p>

**News** :newspaper:

- [ ] Code refinement (in particular, `load_pretrained` is changed to `from_pretrained` to align with huggingface's naming) and model tagging. (**v0.0.4**)
- [X] Initial release and bug fix. (**v0.0.3**)

## About

The hierarchy transformer (HiT) framework aims to provide universal hierarchy embedding in hyperbolic space with transformer encoder-based language models.

- **Huggingface Page**: *<https://huggingface.co/Hierarchy-Transformers/>*.
- **Github Repository**: *<https://github.com/KRR-Oxford/HierarchyTransformers>*. 
- **PyPI**: *<https://pypi.org/project/hierarchy_transformers/>*. 

## Installation

### Main Dependencies

This repository follows a similar layout as the [`sentence-transformers`](https://www.sbert.net/index.html) library. The main model directly extends the sentence transformer architecture. We also utilise [`deeponto`](https://krr-oxford.github.io/DeepOnto/) for extracting hierarchies from source data and constructing datasets from hierarchies, and [`geoopt`](https://geoopt.readthedocs.io/en/latest/index.html) for arithmetic in hyperbolic space.

### Install from PyPI

```bash
# requiring Python>=3.8
pip install hierarchy_transformers
```

### Install from GitHub

```bash
pip install git+https://github.com/KRR-Oxford/HierarchyTransformers.git
```

## Models on Huggingface Hub

Our HiT models are released on the [Huggingface Hub](https://huggingface.co/Hierarchy-Transformers).

### Get Started

Use the following code to get started with HiTs:

```python
from hierarchy_transformers import HierarchyTransformer
from hierarchy_transformers.utils import get_torch_device

# set up the device (use cpu if no gpu found)
gpu_id = 0
device = get_torch_device(gpu_id)

# load the model
model = HierarchyTransformer.from_pretrained('Hierarchy-Transformers/HiT-MiniLM-L12-WordNet', device)

# entity names to be encoded.
entity_names = ["computer", "personal computer", "fruit", "berry"]

# get the entity embeddings
entity_embeddings = model.encode(entity_names)
```

### Default Probing for Subsumption Prediction

Use the entity embeddings to predict the subsumption relationships between them.

```python
# suppose we want to compare "personal computer" and "computer", "berry" and "fruit"
child_entity_embeddings = model.encode(["personal computer", "berry"], convert_to_tensor=True)
parent_entity_embeddings = model.encode(["computer", "fruit"], convert_to_tensor=True)

# compute the hyperbolic distances and norms of entity embeddings
dists = model.manifold.dist(child_entity_embeddings, parent_entity_embeddings)
child_norms = model.manifold.dist0(child_entity_embeddings)
parent_norms = model.manifold.dist0(parent_entity_embeddings)

# use the empirical function for subsumption prediction proposed in the paper
# `centri_score_weight` and the overall threshold are determined on the validation set
# see source code at `src/hierarchy_transformers/evaluation` for more details about our implementation for the hyperparameter tuning.
subsumption_scores = - (dists + centri_score_weight * (parent_norms - child_norms))
```

## Datasets

The datasets for training and evaluating HiTs are available at [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10511042).

## License



    Copyright 2023 Yuan He.
    All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at *<http://www.apache.org/licenses/LICENSE-2.0>*

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

## Citation

If you find this repository or the released models useful, please cite our publication:

*Yuan He, Zhangdie Yuan, Jiaoyan Chen, Ian Horrocks.* **Language Models as Hierarchy Encoders.** To appear at NeurIPS 2024. [[arxiv](https://arxiv.org/abs/2401.11374)] [[neurips-to-upload](to-upload)]

```
@article{he2024language,
  title={Language Models as Hierarchy Encoders},
  author={He, Yuan and Yuan, Zhangdie and Chen, Jiaoyan and Horrocks, Ian},
  journal={arXiv preprint arXiv:2401.11374},
  year={2024}
}
```
