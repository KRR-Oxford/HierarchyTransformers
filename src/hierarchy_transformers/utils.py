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

import torch
from textwrap import dedent

def get_torch_device(gpu_id: int):
    return torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")


def are_models_equal(model1: torch.nn.Module, model2: torch.nn.Module, tolerance: float = 1e-6) -> bool:
    """
    Compares two PyTorch models to check if they are the same by comparing each parameter and buffer.

    Args:
        model1 (torch.nn.Module): The first model to compare.
        model2 (torch.nn.Module): The second model to compare.
        tolerance (float): The tolerance level for floating-point comparison (default is 1e-6).

    Returns:
        bool: True if the models are the same, False otherwise.
    """
    # Compare model parameters
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        if not torch.allclose(param1, param2, atol=tolerance):
            return False

    # Compare model buffers (e.g., running mean/variance in batch norm layers)
    for buffer1, buffer2 in zip(model1.buffers(), model2.buffers()):
        if not torch.allclose(buffer1, buffer2, atol=tolerance):
            return False

    return True

def format_citation(bibtex: str):
    """
    Use `dedent` to properly form bibtex string.
    """
    return dedent(bibtex)