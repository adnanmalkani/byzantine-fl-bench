"""Reproduce FLamby's published FedAvg baseline on Fed-ISIC2019 with FLamby's
own code (docs/PROTOCOL.md, "Main setting and reproduction"; decided 2026-10-13/16).

FLamby's benchmark (flamby/benchmarks/fed_benchmark.py, config_isic2019.json):
ImageNet EfficientNet-b0, fully fine-tuned with FLamby's training
augmentation; weighted focal loss; FedAvg (sample-weighted) with SGD lr 0.01,
batch 64, 100 local steps, 9 rounds (get_nb_max_rounds(100)); the same seeding
order (set_seed, model, data loaders, strategy, set_seed, run). Data loading
in the main process (FLamby's default, --workers 0).

Not installed: opacus and tensorboard, which FLamby's strategies import but
use only for differential privacy and logging (both off); placeholders that
raise if called stand in for them (_stub).

The one change: FLamby runs on CUDA or the CPU. Here every client's model is
moved to the Mac's GPU (MPS), and the final model is evaluated there with the
same logic as flamby.utils.evaluate_model_on_tests (eval mode, balanced
accuracy on each centre's test set and on the pooled test set).

Their side: method "FedAvg100" in FLamby's results CSVs. Check (docs/PROTOCOL.md): our
mean pooled balanced accuracy within one published std of FLamby's,
0.534 +/- 0.114 = 0.420-0.648. Not expected to match seed by seed (batch
order and augmentation draws differ by device).

Memory (2026-10-16: seed 43 filled the 16 GB Mac and
swapped): one seed per process; PyTorch's GPU memory is capped
(--mps_fraction of the recommended maximum, so the run stops with an error
instead of swapping); memory is printed after every round; a second copy
refuses to start (lock file). The rounds are FLamby's FedAvg.run() loop,
written out to print memory after each round (same computation: run() is
`for _ in range(nrounds): perform_round()`).

Usage: python reproduce_flamby_isic.py --seed S [--mps_fraction F] [--rounds R] [--num_updates U]
  (about 1.8-2.7 h per seed on the Mac's GPU; --rounds/--num_updates only
  for a smoke or memory test). Append to the log (>>), one seed at a time:
    nohup caffeinate -is .venv/bin/python reproduce_flamby_isic.py --seed 43 \
      >> results_tables/flamby_isic_repro.log 2>&1 &
  The seed is saved when it finishes (results_tables/flamby_isic_repro_seed<s>.json);
  the table is rebuilt from all saved seeds (results_tables/flamby_isic_repro.csv).
"""
import argparse
import fcntl
import glob
import json
import os
import resource
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")  # no version check over the network


def _stub(name, attr):
    """FLamby's strategies import opacus (differential privacy) and tensorboard
    (logging) at module level but use them only when DP or log=True, both
    off here. Rather than install two packages (and their dependencies) we
    don't use, register placeholders that fail loudly if anything calls them."""
    def fail(*_, **__):
        raise RuntimeError(f"{name}.{attr} is a placeholder (reproduce_flamby_isic.py)")
    mod = types.ModuleType(name)
    setattr(mod, attr, fail)
    sys.modules[name] = mod


_stub("opacus", "PrivacyEngine")
_stub("torch.utils.tensorboard", "SummaryWriter")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from flamby.benchmarks.benchmark_utils import set_seed  # noqa: E402
from flamby.datasets.fed_isic2019 import (BATCH_SIZE, NUM_CLIENTS, Baseline,  # noqa: E402
                                          BaselineLoss, FedIsic2019, get_nb_max_rounds,
                                          metric)
from flamby.strategies import FedAvg  # noqa: E402

from run import save_json  # noqa: E402

LR = 0.01                      # config_isic2019.json, FedAvg
NUM_UPDATES = 100
OUT = Path("results_tables")
CSV = "third_party/FLamby/flamby/results/results_benchmark_fed_isic2019/isic_seed*.csv"
LOCK = OUT / "flamby_isic_repro.lock"
TESTS = ["Pooled Test"] + [f"client_test_{c}" for c in range(NUM_CLIENTS)]
NAMES = ["pooled"] + [f"centre_{c}" for c in range(NUM_CLIENTS)]


def loaders():
    """As flamby.benchmarks.benchmark_utils.init_data_loaders, workers = 0."""
    dl = torch.utils.data.DataLoader
    train = [dl(FedIsic2019(center=c, train=True), batch_size=BATCH_SIZE, shuffle=True)
             for c in range(NUM_CLIENTS)]
    tests = [dl(FedIsic2019(center=c, train=False), batch_size=BATCH_SIZE, shuffle=False)
             for c in range(NUM_CLIENTS)]
    pooled = dl(FedIsic2019(train=False, pooled=True), batch_size=BATCH_SIZE, shuffle=False)
    return train, tests + [pooled]


def memory():
    """Peak process memory and the GPU memory PyTorch holds (GB)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9    # bytes on macOS
    return (f"peak process {peak:.1f} GB, GPU allocated "
            f"{torch.mps.current_allocated_memory() / 1e9:.1f} GB, "
            f"GPU driver {torch.mps.driver_allocated_memory() / 1e9:.1f} GB")


@torch.no_grad()
def evaluate(model, dls, device):
    """flamby.utils.evaluate_model_on_tests on `device`: balanced accuracy per loader."""
    model.eval()
    out = []
    for d in dls:
        state = torch.get_rng_state()
        it = iter(d)
        torch.set_rng_state(state)          # as FLamby: iterating must not move the RNG
        pred, true = zip(*((model(X.to(device)).cpu().numpy(), y.numpy()) for X, y in it))
        out.append(float(metric(np.concatenate(true), np.concatenate(pred))))
    model.train()
    return out


def one_seed(seed, rounds, num_updates, device):
    set_seed(seed)
    model = Baseline()                      # ImageNet weights; new 8-class final layer
    train, tests = loaders()
    s = FedAvg(training_dataloaders=train, model=model, loss=BaselineLoss(),
               optimizer_class=torch.optim.SGD, learning_rate=LR, num_updates=num_updates,
               nrounds=rounds, seed=seed)
    for m in s.models_list:                 # the one change: FLamby's CUDA-or-CPU -> MPS
        m._device = device
        m.model.to(device)
    set_seed(seed)
    t0 = time.time()
    for r in range(rounds):                 # = s.run(), with memory printed per round
        s.perform_round()
        print(f"seed {seed} round {r + 1}/{rounds}: {(time.time() - t0) / 60:.0f} min, "
              f"{memory()}", flush=True)
    final = s.models_list[0].model
    seconds = time.time() - t0
    *centres, pooled = evaluate(final, tests, device)
    return dict(seed=seed, rounds=rounds, num_updates=num_updates, lr=LR, device=str(device),
                train_seconds=round(seconds, 1), pooled=pooled, centres=centres)


def theirs(seeds):
    df = pd.concat(pd.read_csv(p) for p in glob.glob(CSV))
    df = df[(df.Method == "FedAvg100") & df.seed.isin(seeds)]
    return np.array([[df[(df.seed == s) & (df.Test == t)].Metric.item() for t in TESTS]
                     for s in seeds])


def table():
    """Our saved seeds (main setting only) vs FLamby's 5 published seeds."""
    ours = [json.loads(p.read_text()) for p in sorted(OUT.glob("flamby_isic_repro_seed*.json"))]
    ours = [r for r in ours if r["rounds"] == get_nb_max_rounds(NUM_UPDATES)
            and r["num_updates"] == NUM_UPDATES]
    if not ours:
        return
    a = np.array([[r["pooled"], *r["centres"]] for r in ours])
    b = theirs([42, 43, 44, 45, 46])
    t = pd.DataFrame({"test_set": NAMES, "ours_mean": a.mean(0),
                      "ours_seeds": [", ".join(f"{x:.3f}" for x in col) for col in a.T],
                      "flamby_mean": b.mean(0), "flamby_std": b.std(0, ddof=1)})
    t["diff"] = t.ours_mean - t.flamby_mean
    t["within_1_std"] = t["diff"].abs() <= t.flamby_std
    print(f"ours: seeds {[r['seed'] for r in ours]}; FLamby: seeds 42-46")
    print(t.round(4).to_string(index=False))
    t.to_csv(OUT / "flamby_isic_repro.csv", index=False)
    print(f"saved {OUT / 'flamby_isic_repro.csv'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True, help="one seed per process")
    p.add_argument("--mps_fraction", type=float, default=0.8,
                   help="cap on PyTorch's GPU memory, as a fraction of the recommended maximum")
    p.add_argument("--rounds", type=int, default=get_nb_max_rounds(NUM_UPDATES))
    p.add_argument("--num_updates", type=int, default=NUM_UPDATES)
    a = p.parse_args()
    assert torch.backends.mps.is_available(), "needs the Mac's GPU (MPS)"
    device = torch.device("mps")
    smoke = a.rounds != get_nb_max_rounds(NUM_UPDATES) or a.num_updates != NUM_UPDATES
    OUT.mkdir(exist_ok=True)
    lock = open(LOCK, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # released when the process ends
    except BlockingIOError:
        sys.exit(f"another copy is running ({LOCK} is locked); not starting")
    print(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} pid {os.getpid()}: seed {a.seed}, "
          f"rounds {a.rounds}, num_updates {a.num_updates}", flush=True)
    path = OUT / (f"flamby_isic_repro_{'smoke_' if smoke else ''}seed{a.seed}.json")
    if path.exists() and not smoke:
        print(f"seed {a.seed}: already done ({path})")
    else:
        torch.mps.set_per_process_memory_fraction(a.mps_fraction)
        print(f"GPU memory cap: {a.mps_fraction} x recommended maximum "
              f"({a.mps_fraction * torch.mps.recommended_max_memory() / 1e9:.1f} GB)", flush=True)
        r = one_seed(a.seed, a.rounds, a.num_updates, device)
        save_json(path, {**r, "peak_process_gb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)}, indent=2)
        print(f"seed {a.seed}: pooled balanced accuracy {r['pooled']:.4f}, "
              f"{r['train_seconds'] / 60:.1f} min training", flush=True)
    table()


if __name__ == "__main__":
    main()
