# Copyright 2024 Yuan He

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import os

import pytest
import torch

from hierarchy_transformers.models import HierarchyTransformer


@pytest.fixture(params=os.getenv("MODEL_PATHS", "").split(","))
def model_path(request):
    # Ensure there are valid model names
    if not request.param:
        pytest.fail("No valid model names found in the MODEL_PATHS environment variable")
    return request.param.strip()  # Strip any extra spaces


def test_hierarchy_transformer_loading(model_path):
    try:
        # Attempt to load the HierarchyTransformer model
        model = HierarchyTransformer.from_pretrained(model_path)
    except Exception as e:
        pytest.fail(f"Model failed to load: {str(e)}")

    # Check that the model is not None
    assert model is not None, "Loaded model is None"
    # Check that the model has a valid manifold attribute
    assert hasattr(model, "manifold"), "Model does not have a 'manifold' attribute"
    # Check that the manifold is an instance of PoincareBall
    from geoopt.manifolds import PoincareBall

    assert isinstance(model.manifold, PoincareBall), "Manifold is not an instance of PoincareBall"

    # Perform a basic check on the embedding dimension
    assert model.embed_dim > 0, "Embedding dimension should be greater than zero"

    # Test that the model can perform a simple forward pass
    sample_input = ["computer", "personal computer", "fruit", "berry"]
    try:
        output = model.encode(sample_input, convert_to_tensor=True)
    except Exception as e:
        pytest.fail(f"Model failed to encode input: {str(e)}")

    # Check that the output is a tensor with the expected shape
    assert isinstance(output, torch.Tensor), "Output is not a tensor"
    assert output.shape[0] == len(sample_input), "Output shape does not match the input batch size"
    assert output.shape[1] == model.embed_dim, "Output embedding dimension does not match model's embed_dim"
