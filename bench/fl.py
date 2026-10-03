"""Federated learning simulation loop (single machine, all clients every round)."""
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch

from .aggregators import check_valid, get_aggregator
from .attacks import alie_z, apply_attack
from .tasks import PRESETS, load_task
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
    local_steps: Optional[int] = None  # if set, overrides local_epochs (ablation)
    batch_size: int = 32
    lr: float = 0.05
    server_lr: float = 1.0
    f_assumed: Optional[int] = None  # None = server knows the true f
    byz_ids: Optional[str] = None    # e.g. "0" or "1,3": fixed attackers (overrides byz_frac)
    weighted: bool = False           # pass client sample sizes to the aggregator
    cc_tau: float = 1.0
    gaussian_sigma: float = 1.0
    ipm_eps: float = 0.5
    eval_every: int = 5
    val_frac: float = 0.0  # > 0: tuning mode, evaluate on held-out training data
    seed: int = 0
    device: str = "auto"


def _batches(n, cfg, gen, device):
    """Minibatch indices for one client. Default: local_epochs full passes,
    so clients with more data take more steps. With local_steps set, every
    client takes exactly that many steps, reshuffling after each pass."""
    bs = cfg.batch_size
    if cfg.local_steps is None:
        for _ in range(cfg.local_epochs):
            perm = torch.randperm(n, generator=gen).to(device)
            yield from (perm[i:i + bs] for i in range(0, n, bs))
        return
    steps = 0
    while True:
        perm = torch.randperm(n, generator=gen).to(device)
        for i in range(0, n, bs):
            yield perm[i:i + bs]
            steps += 1
            if steps == cfg.local_steps:
                return


def make_config(**kw) -> Config:
    """Config with the dataset's preset defaults (tasks.PRESETS); explicitly
    passed values win over the preset."""
    return Config(**{**PRESETS.get(kw.get("dataset", Config.dataset), {}), **kw})


def _train(model, x, y, cfg, gen, device, loss_fn):
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    model.train()
    for b in _batches(len(y), cfg, gen, device):
        loss = loss_fn(model(x[b]), y[b])
        opt.zero_grad()
        loss.backward()
        opt.step()


@torch.no_grad()
def _eval_loss(model, loss_fn, x, y, bs=1000):
    # Mean loss over the evaluation set, in batches
    model.eval()
    total = sum(loss_fn(model(x[i:i + bs]), y[i:i + bs]).item() * len(y[i:i + bs])
                for i in range(0, len(y), bs))
    return total / len(y)


def _by_client(info, order):
    """Re-index aggregator diagnostics from rows of U to client ids:
    per-row lists become lists indexed by client id; selected row indices
    become client ids."""
    out = {}
    for k, v in info.items():
        if k == "selected":
            out[k] = sorted(order[j] for j in v)
        else:
            col = [None] * len(order)
            for j, x in enumerate(v):
                col[order[j]] = x
            out[k] = col
    return out


def run(cfg: Config, log=print) -> dict:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    n = cfg.n_clients
    if cfg.byz_ids:
        byz_fixed = sorted({int(i) for i in cfg.byz_ids.split(",")})
        assert all(0 <= i < n for i in byz_fixed), cfg.byz_ids
        f = len(byz_fixed)
    else:
        byz_fixed, f = None, int(round(cfg.byz_frac * n))
    f_srv = f if cfg.f_assumed is None else cfg.f_assumed

    result = {"config": asdict(cfg), "f": f, "f_assumed": f_srv,
              "eval_split": "val" if cfg.val_frac > 0 else "test"}
    err = check_valid(cfg.aggregator, n, f_srv)
    if cfg.attack == "alie" and f:
        # z = 0: ALIE sends the honest mean (no attack); z = inf: undefined
        # or NaN (attackers are more than half the clients), so the run is invalid
        z = result["alie_z"] = alie_z(n, f)
        if not np.isfinite(z):
            err = err or f"alie undefined: z = {z} (n={n}, f={f})"
    if err:
        return {**result, "status": "invalid", "reason": err}

    task = load_task(cfg, device)
    C = task.num_classes
    if byz_fixed is not None:
        byz = set(byz_fixed)
    else:
        rng = np.random.default_rng(cfg.seed)
        byz = set(rng.choice(n, f, replace=False).tolist()) if f else set()
    sizes = [len(y) for _, y in task.clients]
    result.update(byz_ids=sorted(byz), client_names=task.client_names,
                  client_sizes=sizes,
                  client_test_sizes=[len(y) for _, y in task.client_tests or []])

    model = task.build_model()
    worker = task.build_model()
    g = flatten_params(model)
    agg = get_aggregator(cfg.aggregator, cc_tau=cfg.cc_tau)
    gen = torch.Generator().manual_seed(cfg.seed)

    history, t0 = [], time.time()
    # Rows of U are honest clients first, then Byzantine ones; `order` maps
    # each row back to its client id (for weights and per-client logging)
    order = [i for i in range(n) if i not in byz] + [i for i in range(n) if i in byz]
    w = torch.tensor([float(sizes[i]) for i in order]) if cfg.weighted else None
    diag = []  # per round: aggregator diagnostics, indexed by client id
    prev = torch.zeros(g.numel())  # previous aggregate update (CC's reference point)
    for r in range(cfg.rounds):
        honest, byz_own = [], []
        for i in range(n):
            x_i, y_i = task.clients[i]
            if i in byz and cfg.attack == "label_flip":
                y_i = C - 1 - y_i
            load_flat_params(worker, g)
            _train(worker, x_i, y_i, cfg, gen, device, task.loss)
            u = (flatten_params(worker) - g).cpu()
            (byz_own if i in byz else honest).append(u)

        H = torch.stack(honest)
        if byz_own:
            M = apply_attack(cfg.attack, H, torch.stack(byz_own), n, f, gen,
                             sigma=cfg.gaussian_sigma, ipm_eps=cfg.ipm_eps)
            U = torch.cat([H, M])
        else:
            U = H

        info = {}
        update = agg(U, f=f_srv, w=w, info=info)
        g = g + cfg.server_lr * update.to(device)
        # distance of each update from the previous aggregate (used to set tau)
        info["dist_prev"] = (U - prev).norm(dim=1).tolist()
        prev = update.detach().cpu()
        diag.append(_by_client(info, order))

        if (r + 1) % cfg.eval_every == 0 or r == cfg.rounds - 1:
            load_flat_params(model, g)
            h = {"round": r + 1, "acc": task.metric(model, *task.test),
                 "loss": _eval_loss(model, task.loss, *task.test)}
            if task.client_tests:
                h["client_acc"] = [task.metric(model, *t) for t in task.client_tests]
                h["worst_acc"] = min(h["client_acc"])
                h["gap"] = max(h["client_acc"]) - h["worst_acc"]
            history.append(h)
            log(f"round {r + 1:4d}  acc {h['acc']:.4f}  ({time.time() - t0:.0f}s)")

    last = [h["acc"] for h in history[-3:]]
    return {**result, "status": "ok", "final_acc": float(np.mean(last)),
            "last_acc": history[-1]["acc"], "history": history, "diag": diag,
            "seconds": round(time.time() - t0, 1)}
