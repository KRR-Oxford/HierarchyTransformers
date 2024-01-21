from .data import *
import torch


def get_torch_device(gpu_id: int):
    return torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
