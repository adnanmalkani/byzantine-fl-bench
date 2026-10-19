"""Choose Centered Clipping's radius tau for one dataset (rule v2.1).

Rule v2 (fixed 2026-10-03, before any ISIC/TCGA data; docs/PROTOCOL.md "Tuning CC's tau"),
amended to v2.1 the same day, before the MNIST multi-seed check was run:
  1. Reference: clean, unweighted FedAvg in validation mode (10% holdout),
     first 10 rounds, the dataset's tuning seeds.
  2. tau0 = median of ||u_i - Delta_prev|| over clients, rounds 2-10 and
     seeds: each update's distance from the previous aggregate, which is
     what CC clips (round 1 has no previous aggregate).
  3. Snap tau0 to the nearest value of the ladder {1, 3} x 10^k (log scale);
     the check grid is that value and two ladder steps on either side.
  4. Clean CC runs (validation mode, dataset's main rounds) for every grid
     tau, in every weighting variant used for the dataset; per variant, the
     paired-by-seed difference in validation accuracy vs tau0, with a 95% t CI.
  5. Keep tau0 unless some tau is better by a practically relevant margin
     (CI lower bound > +0.5 points; v2 used > 0) in EVERY variant; then
     choose, among those, the highest mean accuracy. The margin stops a
     clean-only check from always favouring less clipping (less robustness).
     One tau per dataset, shared across variants. No multiplicity correction.
  6. The whole grid is reported (sensitivity table).

Usage:
    python tune_cc.py --dataset mnist --seeds 0,1,2,3,4
    python tune_cc.py --dataset fed_heart_disease --seeds 0,...,19 --variants 0,1
    (--shard i/N splits the runs across processes; rerun once to decide)
Writes results_tune/cc_tau_<dataset>.json and
results_tables/tau_sensitivity_<dataset>.csv.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bench.fl import make_config, run
from run import _bool, result_path, save_json
from sweep import csv

EARLY_ROUNDS = 10


def snap(x: float) -> float:
    """Nearest value of the ladder {1, 3} x 10^k, in log scale."""
    k = math.floor(math.log10(x))
    # round: 3 * 10**-1 is 0.30000000000000004 in floating point (and in run names)
    ladder = [round(m * 10.0 ** e, 12) for e in (k - 1, k, k + 1) for m in (1, 3)]
    return min(ladder, key=lambda v: abs(math.log10(v) - math.log10(x)))


def grid_around(tau0: float, steps: int = 2):
    """tau0 and `steps` ladder values on either side of it."""
    k = math.floor(math.log10(tau0) + 1e-9)
    ladder = sorted(round(m * 10.0 ** e, 12) for e in range(k - 3, k + 4) for m in (1, 3))
    i = min(range(len(ladder)), key=lambda j: abs(ladder[j] - tau0))
    return ladder[i - steps:i + steps + 1]


def paired_ci(a, b):
    """Mean of a - b over paired seeds, with a 95% t interval."""
    d = np.asarray(a) - np.asarray(b)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - h, d.mean() + h


def decide(acc: dict, tau0: float, margin: float = 0.005):
    """acc: {variant: {tau: [val acc per seed]}} -> (tau, rows of the table).
    Keep tau0 unless another tau is better in every variant, with the 95% CI
    of the paired difference entirely above `margin` (accuracy as a fraction)."""
    rows, better = [], None
    for tau in sorted(next(iter(acc.values()))):
        sig_all = True
        for var, a in acc.items():
            m, lo, hi = paired_ci(a[tau], a[tau0]) if tau != tau0 else (0.0, 0.0, 0.0)
            rows.append({"variant": var, "tau": tau, "val_acc_mean": np.mean(a[tau]),
                         "diff_vs_tau0": m, "ci95_lo": lo, "ci95_hi": hi})
            sig_all &= tau != tau0 and lo > margin
        if sig_all:
            score = np.mean([np.mean(a[tau]) for a in acc.values()])
            if better is None or score > better[1]:
                better = (tau, score)
    return (better[0] if better else tau0), rows


def _run_or_load(cfg, path, mine):
    if not path.exists() and mine:
        print(f"running {path.stem}")
        save_json(path, run(cfg, log=lambda *_: None), indent=2)
    if not path.exists():
        return None
    res = json.loads(path.read_text())
    assert res["eval_split"] == "val" and res["status"] == "ok", path
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--seeds", type=csv(int), default=[0, 1, 2, 3, 4])
    p.add_argument("--variants", type=csv(_bool), default=[False],
                   help="weighting variants used for this dataset, e.g. 0,1")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--f_assumed", type=int, default=None,
                   help="design f for clean runs (default: 1 if n <= 6 else 4)")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--margin", type=float, default=0.005,
                   help="override needs paired CI lower bound above this (0.005 = 0.5 pt)")
    p.add_argument("--tau0_only", action="store_true", help="stop after computing tau0")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--out", default="results_tune")
    a = p.parse_args()
    shard_i, shard_n = map(int, a.shard.split("/"))
    out = Path(a.out)
    (out / "tau0").mkdir(parents=True, exist_ok=True)
    n = make_config(dataset=a.dataset).n_clients
    f_srv = a.f_assumed if a.f_assumed is not None else (1 if n <= 6 else 4)
    base = dict(dataset=a.dataset, attack="none", alpha=a.alpha, f_assumed=f_srv,
                val_frac=a.val_frac)

    # Steps 1-2: tau0 from early rounds of clean unweighted FedAvg
    jobs, k = [], 0
    dists = []
    for sd in a.seeds:
        cfg = make_config(aggregator="fedavg", seed=sd, rounds=EARLY_ROUNDS, **base)
        res = _run_or_load(cfg, result_path(out / "tau0", cfg), k % shard_n == shard_i)
        k += 1
        if res:
            dists += [x for d in res["diag"][1:EARLY_ROUNDS] for x in d["dist_prev"]]
    if k * (EARLY_ROUNDS - 1) * n != len(dists):
        print("tau0 runs not finished yet (other shards?); rerun")
        return
    tau0_raw = float(np.median(dists))
    tau0 = snap(tau0_raw)
    grid = grid_around(tau0)
    print(f"tau0: median distance {tau0_raw:.4g} -> snapped {tau0}; grid {grid}")
    if a.tau0_only:
        return

    # Steps 4-5: clean CC on the grid, every variant, paired by seed
    acc = {}
    for wt in a.variants:
        for tau in grid:
            for sd in a.seeds:
                cfg = make_config(aggregator="cc", cc_tau=tau, weighted=wt, seed=sd, **base)
                res = _run_or_load(cfg, result_path(out, cfg), k % shard_n == shard_i)
                k += 1
                if res:
                    acc.setdefault("weighted" if wt else "unweighted", {}) \
                       .setdefault(tau, []).append(res["final_acc"])
    complete = all(len(v) == len(a.seeds) for d in acc.values() for v in d.values()) \
        and sum(len(d) for d in acc.values()) == len(grid) * len(a.variants)
    if not complete:
        print("CC grid runs not finished yet (other shards?); rerun")
        return
    tau, rows = decide(acc, tau0, a.margin)
    table = pd.DataFrame(rows)
    table[["val_acc_mean", "diff_vs_tau0", "ci95_lo", "ci95_hi"]] *= 100  # points
    Path("results_tables").mkdir(exist_ok=True)
    table = table.round({c: 2 for c in ("val_acc_mean", "diff_vs_tau0", "ci95_lo", "ci95_hi")})
    table.to_csv(f"results_tables/tau_sensitivity_{a.dataset}.csv", index=False)
    print(table.to_string(index=False))
    sel = {"dataset": a.dataset, "rule": "v2.1", "margin": a.margin, "cc_tau": tau, "tau0_raw": tau0_raw,
           "tau0": tau0, "grid": grid, "overridden": tau != tau0, "seeds": a.seeds,
           "variants": [bool(v) for v in a.variants], "f_assumed": f_srv,
           "val_frac": a.val_frac, "rounds": make_config(dataset=a.dataset).rounds}
    (out / f"cc_tau_{a.dataset}.json").write_text(json.dumps(sel, indent=2))
    print(f"\nselected tau = {tau}" + (" (validation override)" if tau != tau0 else
                                         " (= tau0, no improvement beyond the margin)"))


if __name__ == "__main__":
    main()
