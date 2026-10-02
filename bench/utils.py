import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def flatten_params(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.data.reshape(-1) for p in model.parameters()])


def load_flat_params(model: torch.nn.Module, flat: torch.Tensor) -> None:
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[i:i + n].view_as(p))
        i += n
