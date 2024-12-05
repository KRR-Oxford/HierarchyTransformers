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

from geoopt.manifolds import PoincareBall
import torch


def get_circum_poincareball(embed_dim: int) -> PoincareBall:
    """Get a Poincaré Ball with a curvature adapted to a given embedding dimension so that it circumscribes the output embedding space of pre-trained language models.
    """
    curvature = 1 / embed_dim
    manifold = PoincareBall(c=curvature)
    return manifold


def project_onto_subspace(manifold: PoincareBall, point: torch.Tensor, normal: torch.Tensor):
    """Compute the (hyperbolic) projection of a point onto a subspace (a hyper-plane through origin) of the input manifold.

    The projected point is the mid point of the geodesic segment that joins the input point and its reflection point about the subspace plane.

    Args:
        manifold (geoopt.manifolds.PoincareBall): The input Poincaré ball manifold.
        point (torch.Tensor): The input point
        normal (torch.Tensor): The normal vector of the subspace.
    """
    reflection = reflect_about_subspace(point, normal)
    midpoint = manifold.weighted_midpoint(torch.vstack((point, reflection)))
    return midpoint


def reflect_about_subspace(point: torch.Tensor, normal: torch.Tensor):
    """Compute the (Euclidean) reflection of a point about a sub-space (a hyper-plane through origin).

    This is a helper function for computing hyperbolic subspace projection of a point.

    Args:
        point (torch.Tensor): The input point.
        normal (torch.Tensor): The normal vector of the plane (through orgin).
    """

    # Ensure the norm vector is non-zero
    if torch.all(normal == 0):
        raise ValueError("Norm vector cannot be zero.")

    # Calculate the dot product and magnitude squared of the norm vector
    dot_product = torch.dot(point, normal)
    normal_squared = torch.dot(normal, normal)

    # Compute the reflection point without explicitly normalizing the norm vector
    reflection = point - 2 * (dot_product / normal_squared) * normal

    return reflection
