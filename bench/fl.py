"""Federated learning simulation loop (single machine, all clients every round)."""
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .aggregators import check_valid, get_aggregator
from .attacks import apply_attack
from .data import dirichlet_partition, load_dataset
from .models import build_model
from .utils import flatten_params, get_device, load_flat_params, set_seed


@dataclass
class Config:
    dataset: str = "mnist"
    aggregator: str = "fedavg"
    attack: str = "none"
    byz_frac: float = 0.0
    alpha: float = 0.5            # Dirichlet non-IID level
    n_clients: int = 20
    rounds: int = 50
    local_epochs: int = 1
    batch_size: int = 32
    lr: float = 0.05
    server_lr: float = 1.0
    f_assumed: Optional[int] = None  # None = server knows the true f
    cc_tau: float = 1.0
    gaussian_sigma: float = 1.0
    ipm_eps: float = 0.5
    eval_every: int = 5
    seed: int = 0
    device: str = "auto"


def _train(model, x, y, cfg, gen, device):
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    model.train()
    for _ in range(cfg.local_epochs):
        perm = torch.randperm(len(y), generator=gen).to(device)
        for i in range(0, len(y), cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            loss = F.cross_entropy(model(x[b]), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()


@torch.no_grad()
def _evaluate(model, x, y, bs=1000):
    model.eval()
    correct = sum((model(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
                  for i in range(0, len(y), bs))
    return correct / len(y)


def run(cfg: Config, log=print) -> dict:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    n = cfg.n_clients
    f = int(round(cfg.byz_frac * n))
    f_srv = f if cfg.f_assumed is None else cfg.f_assumed

    result = {"config": asdict(cfg), "f": f, "f_assumed": f_srv}
    err = check_valid(cfg.aggregator, n, f_srv)
    if err:
        return {**result, "status": "invalid", "reason": err}

    xtr, ytr, xte, yte, C = load_dataset(cfg.dataset)
    parts = dirichlet_partition(ytr.numpy(), n, cfg.alpha, cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    byz = set(rng.choice(n, f, replace=False).tolist()) if f else set()

    xtr, ytr, xte, yte = (t.to(device) for t in (xtr, ytr, xte, yte))
    parts = [torch.as_tensor(p, device=device) for p in parts]
    in_ch, img = xtr.shape[1], xtr.shape[2]
    model = build_model(in_ch, img, C).to(device)
    worker = build_model(in_ch, img, C).to(device)
    g = flatten_params(model)
    agg = get_aggregator(cfg.aggregator, cc_tau=cfg.cc_tau)
    gen = torch.Generator().manual_seed(cfg.seed)

    history, t0 = [], time.time()
    for r in range(cfg.rounds):
        honest, byz_own = [], []
        for i in range(n):
            y_i = ytr[parts[i]]
            if i in byz and cfg.attack == "label_flip":
                y_i = C - 1 - y_i
            load_flat_params(worker, g)
            _train(worker, xtr[parts[i]], y_i, cfg, gen, device)
            u = (flatten_params(worker) - g).cpu()
            (byz_own if i in byz else honest).append(u)

        H = torch.stack(honest)
        if byz_own:
            M = apply_attack(cfg.attack, H, torch.stack(byz_own), n, f, gen,
                             sigma=cfg.gaussian_sigma, ipm_eps=cfg.ipm_eps)
            U = torch.cat([H, M])
        else:
            U = H

        update = agg(U, f=f_srv)
        g = g + cfg.server_lr * update.to(device)

        if (r + 1) % cfg.eval_every == 0 or r == cfg.rounds - 1:
            load_flat_params(model, g)
            acc = _evaluate(model, xte, yte)
            history.append({"round": r + 1, "acc": acc})
            log(f"round {r + 1:4d}  acc {acc:.4f}  ({time.time() - t0:.0f}s)")

    last = [h["acc"] for h in history[-3:]]
    return {**result, "status": "ok", "final_acc": float(np.mean(last)),
            "history": history, "seconds": round(time.time() - t0, 1)}
