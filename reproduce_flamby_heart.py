"""Reproduce FLamby's published FedAvg baseline on Fed-Heart-Disease.

Our side: sample-weighted FedAvg with FLamby's settings (tasks.PRESETS:
logistic regression, SGD lr 0.001, batch 4, 100 local steps, 15 rounds),
FLamby's seeds 42-46, last-round accuracy (as FLamby reports).
Their side: method "FedAvg100" in FLamby's results CSVs (same seeds).
Not bit-identical by design (different data order and init order), so we
compare means and check we fall within one published std.

Usage: python reproduce_flamby_heart.py   (needs the heart data downloaded)
Writes results_tables/flamby_heart_repro.csv
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from bench.fl import make_config, run

SEEDS = [42, 43, 44, 45, 46]
CSV = "third_party/FLamby/flamby/results/results_benchmark_fed_heart_disease/heart_seed*.csv"
TESTS = ["Pooled Test", "client_test_0", "client_test_1", "client_test_2", "client_test_3"]
NAMES = ["pooled", "cleveland", "hungary", "switzerland", "va"]


def ours():
    rows = []
    for s in SEEDS:
        r = run(make_config(dataset="fed_heart_disease", aggregator="fedavg",
                            weighted=True, seed=s), log=lambda *_: None)
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
    print(table.round(4).to_string(index=False))
    out = Path("results_tables")
    out.mkdir(exist_ok=True)
    table.to_csv(out / "flamby_heart_repro.csv", index=False)
    print(f"\nsaved {out / 'flamby_heart_repro.csv'}")


if __name__ == "__main__":
    main()
