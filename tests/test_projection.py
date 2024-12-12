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

import pytest
import torch
from geoopt.manifolds import PoincareBall

from hierarchy_transformers.models.hierarchy_transformer.hyperbolic import (
    project_onto_subspace,
    reflect_about_subspace,
)


@pytest.fixture
def manifold():
    # Create a PoincareBall manifold with a sample curvature
    return PoincareBall(c=1.0)


@pytest.fixture
def sample_point():
    # Create a sample point on the manifold
    return torch.tensor([0.3, 0.3], dtype=torch.float32)


@pytest.fixture
def normal_vector():
    # Create a sample normal vector for the subspace
    return torch.tensor([0.0, 1.0], dtype=torch.float32)


def test_reflect_about_subspace(sample_point, normal_vector):
    # Test the reflect_about_subspace function
    reflection = reflect_about_subspace(sample_point, normal_vector)

    # Check that the reflection is a torch.Tensor
    assert isinstance(reflection, torch.Tensor), "Reflection should be a torch.Tensor"

    # Check the shape of the reflection
    assert reflection.shape == sample_point.shape, "Reflection should have the same shape as the input point"

    # Verify the reflection calculation
    expected_reflection = torch.tensor([0.3, -0.3], dtype=torch.float32)
    assert torch.allclose(reflection, expected_reflection, atol=1e-6), "Reflection values do not match expected values"

    # Edge case: Normal vector cannot be zero
    with pytest.raises(ValueError):
        reflect_about_subspace(sample_point, torch.zeros_like(normal_vector))


def test_project_onto_subspace(manifold, sample_point, normal_vector):
    # Test the project_onto_subspace function
    projection = project_onto_subspace(manifold, sample_point, normal_vector)

    # Check that the projection is a torch.Tensor
    assert isinstance(projection, torch.Tensor), "Projection should be a torch.Tensor"

    # Check the shape of the projection
    assert projection.shape == sample_point.shape, "Projection should have the same shape as the input point"

    # Check that the projection lies within the Poincare Ball (norm less than 1)
    norm = torch.norm(projection)
    assert norm < 1.0, "Projected point should lie within the Poincare Ball"

    # Verify the projection calculation
    # Note: You may need to adjust this expected value based on the specifics of the PoincareBall manifold projection
    expected_projection = torch.tensor([0.27321523, 0.0000], dtype=torch.float32)
    assert torch.allclose(projection, expected_projection, atol=1e-6), "Projection values do not match expected values"
