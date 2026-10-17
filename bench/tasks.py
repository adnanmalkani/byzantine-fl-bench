"""A task bundles what differs between datasets: per-client training data,
test set(s), model, loss and metric.

- Image datasets (mnist, fmnist, cifar10, synthetic): one global train set
  split across n_clients with Dirichlet(alpha); one global test set.
- FLamby datasets (natural splits, e.g. fed_heart_disease): one client per
  hospital, each with its own test set, plus FLamby's pooled test set.
- fed_isic2019: frozen-backbone main setting (README): clients train only a
  linear layer on cached EfficientNet-b0 features (extract_isic_features.py).
"""
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, List, Optional, Tuple

from pathlib import Path

import torch
import torch.nn.functional as F

from .data import dirichlet_partition, load_dataset, train_val_split
from .models import build_model

# FLamby calls torch.manual_seed(42) when imported (flamby/utils.py). Imported
# lazily inside a run, that reset the RNG between set_seed and model init, so
# the first FLamby run in each process started from the seed-42 model whatever
# its seed. Importing here makes the reset happen before any run seeds itself.
try:
    from flamby.datasets.fed_heart_disease import Baseline, FedHeartDisease
    from flamby.datasets.fed_tcga_brca import Baseline as TcgaBaseline
    from flamby.datasets.fed_tcga_brca import BaselineLoss as CoxLoss
    from flamby.datasets.fed_tcga_brca import FedTcgaBrca
    from lifelines.utils import concordance_index
except ImportError:  # FLamby not installed: image datasets only
    Baseline = FedHeartDisease = TcgaBaseline = CoxLoss = FedTcgaBrca = None
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")  # albumentations: no network check
try:  # needs FLamby's ISIC packages (setup_flamby.sh)
    from flamby.datasets.fed_isic2019 import BaselineLoss as FocalLoss
except ImportError:
    FocalLoss = None

FLAMBY = {"fed_heart_disease": 4, "fed_tcga_brca": 6, "fed_isic2019": 6}  # dataset -> centres
ISIC_FEATURES = "data/flamby/fed_isic2019/features_effb0_centercrop.pt"
ISIC_CLASSES = 8
# Survival tasks: y = (event, time), metric = C-index; label flipping is undefined
SURVIVAL = {"fed_tcga_brca"}

# Per-dataset defaults, applied by make_config unless overridden explicitly.
# fed_heart_disease uses FLamby's FedAvg local settings (SGD lr 0.001, batch 4,
# 100 local steps per round). Rounds: 106, where FedAvg's pooled validation
# accuracy plateaus (convergence_heart.py); FLamby's own 15 rounds
# (= (486 // 4 // 4) * 50 // 100) is kept as a secondary setting.
PRESETS = {
    "fed_heart_disease": dict(n_clients=4, lr=0.001, batch_size=4,
                              local_steps=100, rounds=106, eval_every=1,
                              device="cpu"),  # tiny model: CPU beats MPS
    # Main setting (README, select_lr_tcga.py): FLamby's Cox model, batch 8,
    # 100 local steps, but lr 0.003 and 243 rounds (pre-registered rule +
    # Amendment 1). FLamby's own lr 0.1 / 5 rounds collapse on their own and
    # are used only in reproduce_flamby_tcga.py.
    "fed_tcga_brca": dict(n_clients=6, lr=0.003, batch_size=8, local_steps=100,
                          rounds=243, eval_every=1, device="cpu"),
    # Frozen-backbone main setting: FLamby's batch 64 and 100 local steps, SGD
    # (as everywhere here). lr 0.01 (FLamby's FedAvg rate) and 101 rounds: the
    # screen (select_lr_isic.py) + Amendment 2 (README, 2026-10-17).
    "fed_isic2019": dict(n_clients=6, lr=0.01, batch_size=64, local_steps=100,
                         rounds=101, eval_every=1, device="cpu"),
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
    # Optional (model, x, y) -> correct predictions per class (ISIC: the
    # worst-centre metric uses only classes with enough test images)
    class_correct: Optional[Callable] = None


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


@torch.no_grad()
def _c_index(model, x, y):
    # FLamby TCGA metric (lifelines): higher risk score = earlier event.
    # NaN when the set has no comparable pairs (e.g. no events).
    model.eval()
    risk = model(x).squeeze(1).cpu().numpy()
    y = y.cpu().numpy()
    try:
        return float(concordance_index(y[:, 1], -risk, y[:, 0]))
    except ZeroDivisionError:
        return float("nan")


@torch.no_grad()
def _balanced_accuracy(model, x, y):
    # FLamby ISIC metric (sklearn's balanced_accuracy_score): mean recall over
    # the classes present in y
    model.eval()
    pred = model(x).argmax(1)
    return torch.stack([(pred[y == c] == c).float().mean() for c in torch.unique(y)]).mean().item()


@torch.no_grad()
def _class_correct(model, x, y):
    model.eval()
    hit = model(x).argmax(1) == y
    return torch.bincount(y[hit], minlength=ISIC_CLASSES).tolist()


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
    assert FedHeartDisease is not None, "FLamby not installed: run setup_flamby.sh"

    def tensors(**kw):
        ds = FedHeartDisease(data_path=root, **kw)
        xs, ys = zip(*(ds[i] for i in range(len(ds))))
        return torch.stack(xs).to(device), torch.stack(ys).reshape(-1, 1).to(device)

    n = FLAMBY[cfg.dataset]
    clients = [tensors(center=c, train=True) for c in range(n)]
    if cfg.val_frac > 0:
        # Tuning mode: hold out a stratified val_frac of each hospital's
        # training data; the metric is measured on the pooled held-out samples
        # (each normalised with its own hospital's statistics, as clients see
        # them). The test sets are never loaded.
        client_tests = []
        for c, (x, y) in enumerate(clients):
            tr, va = train_val_split(y.cpu().numpy().ravel(), cfg.val_frac, cfg.seed + c)
            tr, va = torch.as_tensor(tr, device=device), torch.as_tensor(va, device=device)
            clients[c], client_tests = (x[tr], y[tr]), client_tests + [(x[va], y[va])]
        pooled = tuple(torch.cat(t) for t in zip(*client_tests))
    else:
        client_tests = [tensors(center=c, train=False) for c in range(n)]
        # As in FLamby's benchmark, the pooled test set is normalised with pooled
        # training statistics (clients use their own centre's statistics)
        pooled = tensors(train=False, pooled=True)
    return Task(clients=clients, test=pooled, client_tests=client_tests,
                build_model=lambda: Baseline().to(device),
                loss=F.binary_cross_entropy, metric=_binary_accuracy, num_classes=2,
                client_names=["cleveland", "hungary", "switzerland", "va"])


def _tcga_task(cfg, device):
    assert FedTcgaBrca is not None, "FLamby not installed: run setup_flamby.sh"

    def tensors(ds):
        xs, ys = zip(*(ds[i] for i in range(len(ds))))
        return torch.stack(xs).to(device), torch.stack(ys).to(device)   # y: (event, time)

    n = FLAMBY[cfg.dataset]
    clients = [tensors(FedTcgaBrca(center=c, train=True)) for c in range(n)]
    if cfg.val_frac > 0:
        # Tuning mode: per-region holdout stratified by event indicator;
        # metric on the pooled held-out patients. Test sets never loaded.
        client_tests = []
        for c, (x, y) in enumerate(clients):
            tr, va = train_val_split(y[:, 0].cpu().numpy().astype(int), cfg.val_frac, cfg.seed + c)
            tr, va = torch.as_tensor(tr, device=device), torch.as_tensor(va, device=device)
            clients[c], client_tests = (x[tr], y[tr]), client_tests + [(x[va], y[va])]
        pooled = tuple(torch.cat(t) for t in zip(*client_tests))
    else:
        # No per-centre normalisation here: pooled test = all regions' test sets
        client_tests = [tensors(FedTcgaBrca(center=c, train=False)) for c in range(n)]
        pooled = tensors(FedTcgaBrca(train=False, pooled=True))
    return Task(clients=clients, test=pooled, client_tests=client_tests,
                build_model=lambda: TcgaBaseline().to(device), loss=CoxLoss(),
                metric=_c_index, num_classes=None,
                client_names=["northeast", "south", "west", "midwest", "europe", "canada"])


@lru_cache
def isic_features(path=ISIC_FEATURES):
    """{"train"/"test": [(X, y, names) per centre], "meta": ...}"""
    assert Path(path).exists(), f"{path} missing: run extract_isic_features.py"
    return torch.load(path, weights_only=True)


def _isic_task(cfg, device):
    assert FocalLoss is not None, "FLamby's ISIC packages missing: run setup_flamby.sh"
    assert cfg.lr is not None and cfg.rounds is not None, \
        "fed_isic2019: lr and rounds must be set (preset: README, Amendment 2)"
    d = isic_features()
    n = FLAMBY[cfg.dataset]
    clients = [(X.to(device), y.to(device)) for X, y, _ in d["train"]]
    if cfg.val_frac > 0:
        # Tuning mode: per-centre holdout stratified by class; metric on the
        # pooled held-out images. Test sets never used.
        client_tests = []
        for c, (x, y) in enumerate(clients):
            tr, va = train_val_split(y.cpu().numpy(), cfg.val_frac, cfg.seed + c)
            tr, va = torch.as_tensor(tr, device=device), torch.as_tensor(va, device=device)
            clients[c], client_tests = (x[tr], y[tr]), client_tests + [(x[va], y[va])]
    else:
        client_tests = [(X.to(device), y.to(device)) for X, y, _ in d["test"]]
    pooled = tuple(torch.cat(t) for t in zip(*client_tests))   # FLamby's pooled test set
    dim = clients[0][0].shape[1]
    return Task(clients=clients, test=pooled, client_tests=client_tests,
                build_model=lambda: torch.nn.Linear(dim, ISIC_CLASSES).to(device),
                loss=FocalLoss(), metric=_balanced_accuracy, num_classes=ISIC_CLASSES,
                client_names=[f"centre_{c}" for c in range(n)], class_correct=_class_correct)


def load_task(cfg, device) -> Task:
    if cfg.dataset in FLAMBY:
        assert cfg.n_clients == FLAMBY[cfg.dataset], \
            f"{cfg.dataset} has {FLAMBY[cfg.dataset]} clients (n_clients={cfg.n_clients})"
        if cfg.dataset == "fed_isic2019":
            return _isic_task(cfg, device)
        return _tcga_task(cfg, device) if cfg.dataset in SURVIVAL else _heart_task(cfg, device)
    return _image_task(cfg, device)
