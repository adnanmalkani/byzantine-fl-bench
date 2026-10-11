"""Reproduce FLamby's published FedAvg baseline on Fed-TCGA-BRCA.

Our side: sample-weighted FedAvg with FLamby's settings (tasks.PRESETS:
Cox model, SGD lr 0.1, batch 8, 100 local steps) and FLamby's 5 rounds,
FLamby's seeds 42-46, last-round C-index (as FLamby reports): FLamby's
pooled test set plus each region's own test set.
Their side: method "FedAvg100" in FLamby's results CSVs (same seeds).
FLamby's FedAvg is unstable on TCGA (seed 43 collapses to 0.42 pooled), so
besides means within one published std we print the per-seed comparison.

Usage: python reproduce_flamby_tcga.py   (FLamby's GDC licence prompt must
have been answered once)
Writes results_tables/flamby_tcga_repro.csv and flamby_tcga_repro_seeds.csv
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from bench.fl import make_config, run

SEEDS = [42, 43, 44, 45, 46]
CSV = "third_party/FLamby/flamby/results/results_benchmark_fed_tcga_brca/tcga_seed*.csv"
TESTS = ["Pooled Test"] + [f"client_test_{c}" for c in range(6)]
NAMES = ["pooled", "northeast", "south", "west", "midwest", "europe", "canada"]


def ours():
    rows = []
    for s in SEEDS:
        r = run(make_config(dataset="fed_tcga_brca", aggregator="fedavg",
                            weighted=True, rounds=5, seed=s), log=lambda *_: None)
        last = r["history"][-1]
        rows.append([last["acc"], *last["client_acc"]])
    return np.array(rows)


def theirs():
    df = pd.concat(pd.read_csv(p) for p in glob.glob(CSV))
    df = df[(df.Method == "FedAvg100") & df.seed.isin(SEEDS)]
    return np.array([[df[(df.seed == s) & (df.Test == t)].Metric.item() for t in TESTS]
                     for s in SEEDS])


def main():
    a, b = ours(), theirs()
    table = pd.DataFrame({
        "test_set": NAMES,
        "ours_mean": a.mean(0), "ours_std": a.std(0, ddof=1),
        "flamby_mean": b.mean(0), "flamby_std": b.std(0, ddof=1),
    })
    table["diff"] = table.ours_mean - table.flamby_mean
    table["within_1_std"] = table["diff"].abs() <= table.flamby_std
    seeds = pd.DataFrame(
        [[s, n, a[i, j], b[i, j], a[i, j] - b[i, j]]
         for i, s in enumerate(SEEDS) for j, n in enumerate(NAMES)],
        columns=["seed", "test_set", "ours", "flamby", "diff"])
    print(table.round(4).to_string(index=False))
    print("\nper seed:")
    print(seeds.pivot(index="seed", columns="test_set", values="diff")[NAMES]
          .round(4).to_string())
    out = Path("results_tables")
    out.mkdir(exist_ok=True)
    table.to_csv(out / "flamby_tcga_repro.csv", index=False)
    seeds.to_csv(out / "flamby_tcga_repro_seeds.csv", index=False)
    print(f"\nsaved {out / 'flamby_tcga_repro.csv'} and flamby_tcga_repro_seeds.csv")


if __name__ == "__main__":
    main()
