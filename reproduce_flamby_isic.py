"""Reproduce FLamby's published FedAvg baseline on Fed-ISIC2019 with FLamby's
own code (README, "Main setting and reproduction"; decided 2026-10-13/16).

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

Their side: method "FedAvg100" in FLamby's results CSVs. Check (README): our
mean pooled balanced accuracy within one published std of FLamby's,
0.534 +/- 0.114 = 0.420-0.648. Not expected to match seed by seed (batch
order and augmentation draws differ by device).

Usage: python reproduce_flamby_isic.py [--seeds 42,43] [--rounds R] [--num_updates U]
  (about 1.5 h per seed on the Mac's GPU; --rounds/--num_updates only for a
  smoke test). Each seed is saved when it finishes
  (results_tables/flamby_isic_repro_seed<s>.json), so a stopped run keeps
  finished seeds; the table is rebuilt from all saved seeds
  (results_tables/flamby_isic_repro.csv).
"""
import argparse
import glob
import json
import os
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
    final = s.run()[0]
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
    p.add_argument("--seeds", default="42,43")
    p.add_argument("--rounds", type=int, default=get_nb_max_rounds(NUM_UPDATES))
    p.add_argument("--num_updates", type=int, default=NUM_UPDATES)
    a = p.parse_args()
    assert torch.backends.mps.is_available(), "needs the Mac's GPU (MPS)"
    device = torch.device("mps")
    smoke = a.rounds != get_nb_max_rounds(NUM_UPDATES) or a.num_updates != NUM_UPDATES
    OUT.mkdir(exist_ok=True)
    for seed in map(int, a.seeds.split(",")):
        path = OUT / (f"flamby_isic_repro_{'smoke_' if smoke else ''}seed{seed}.json")
        if path.exists() and not smoke:
            print(f"seed {seed}: already done ({path})")
            continue
        r = one_seed(seed, a.rounds, a.num_updates, device)
        save_json(path, r, indent=2)
        print(f"seed {seed}: pooled balanced accuracy {r['pooled']:.4f}, "
              f"{r['train_seconds'] / 60:.1f} min training", flush=True)
    table()


if __name__ == "__main__":
    main()
