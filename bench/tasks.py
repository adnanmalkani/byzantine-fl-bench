"""A task bundles what differs between datasets: per-client training data,
test set(s), model, loss and metric.

- Image datasets (mnist, fmnist, cifar10, synthetic): one global train set
  split across n_clients with Dirichlet(alpha); one global test set.
- FLamby datasets (natural splits, e.g. fed_heart_disease): one client per
  hospital, each with its own test set, plus FLamby's pooled test set.
"""
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .data import dirichlet_partition, load_dataset, train_val_split
from .models import build_model

FLAMBY = {"fed_heart_disease": 4}  # dataset -> number of hospitals

# Per-dataset defaults, applied by make_config unless overridden explicitly.
# fed_heart_disease follows FLamby's FedAvg benchmark: SGD lr 0.001, batch 4,
# 100 local steps per round, 15 rounds = (486 // 4 // 4) * 50 // 100.
PRESETS = {
    "fed_heart_disease": dict(n_clients=4, lr=0.001, batch_size=4,
                              local_steps=100, rounds=15, eval_every=1,
                              device="cpu"),  # tiny model: CPU beats MPS
}


@dataclass
class Task:
    clients: List[Tuple[torch.Tensor, torch.Tensor]]  # (x, y) per client
    test: Tuple[torch.Tensor, torch.Tensor]           # global / pooled test set
    client_tests: Optional[list]                      # (x, y) per client, natural splits only
    build_model: Callable[[], torch.nn.Module]
    loss: Callable
    metric: Callable                                  # (model, x, y) -> float
    num_classes: int
    client_names: List[str]


@torch.no_grad()
def _accuracy(model, x, y, bs=1000):
    model.eval()
    correct = sum((model(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
                  for i in range(0, len(y), bs))
    return correct / len(y)


@torch.no_grad()
def _binary_accuracy(model, x, y):
    # FLamby heart metric: sigmoid output > 0.5 vs label
    model.eval()
    return ((model(x) > 0.5).float() == y).float().mean().item()


def _image_task(cfg, device):
    xtr, ytr, xte, yte, C = load_dataset(cfg.dataset)
    if cfg.val_frac > 0:
        # Tuning mode: clients train on the rest of the training data and
        # the metric is measured on the held-out part; test set never used
        tr, va = train_val_split(ytr.numpy(), cfg.val_frac, cfg.seed)
        xtr, ytr, xte, yte = xtr[tr], ytr[tr], xtr[va], ytr[va]
    parts = dirichlet_partition(ytr.numpy(), cfg.n_clients, cfg.alpha, cfg.seed)
    xtr, ytr, xte, yte = (t.to(device) for t in (xtr, ytr, xte, yte))
    clients = []
    for p in parts:
        p = torch.as_tensor(p, device=device)
        clients.append((xtr[p], ytr[p]))
    in_ch, img = xtr.shape[1], xtr.shape[2]
    return Task(clients=clients, test=(xte, yte), client_tests=None,
                build_model=lambda: build_model(in_ch, img, C).to(device),
                loss=F.cross_entropy, metric=_accuracy, num_classes=C,
                client_names=[str(i) for i in range(cfg.n_clients)])


def _heart_task(cfg, device, root="./data/flamby/fed_heart_disease"):
    from flamby.datasets.fed_heart_disease import Baseline, FedHeartDisease

    if cfg.val_frac > 0:
        raise NotImplementedError("validation split not implemented for FLamby tasks yet")

    def tensors(**kw):
        ds = FedHeartDisease(data_path=root, **kw)
        xs, ys = zip(*(ds[i] for i in range(len(ds))))
        return torch.stack(xs).to(device), torch.stack(ys).reshape(-1, 1).to(device)

    n = FLAMBY[cfg.dataset]
    clients = [tensors(center=c, train=True) for c in range(n)]
    client_tests = [tensors(center=c, train=False) for c in range(n)]
    # As in FLamby's benchmark, the pooled test set is normalised with pooled
    # training statistics (clients use their own centre's statistics)
    pooled = tensors(train=False, pooled=True)
    return Task(clients=clients, test=pooled, client_tests=client_tests,
                build_model=lambda: Baseline().to(device),
                loss=F.binary_cross_entropy, metric=_binary_accuracy, num_classes=2,
                client_names=["cleveland", "hungary", "switzerland", "va"])


def load_task(cfg, device) -> Task:
    if cfg.dataset in FLAMBY:
        assert cfg.n_clients == FLAMBY[cfg.dataset], \
            f"{cfg.dataset} has {FLAMBY[cfg.dataset]} clients (n_clients={cfg.n_clients})"
        return _heart_task(cfg, device)
    return _image_task(cfg, device)
