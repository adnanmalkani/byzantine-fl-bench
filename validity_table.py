"""Validity table (RQ-A): for each rule, number of clients n and assumed f,
is the run valid, and if so does the rule degenerate?

Degeneracy is detected numerically on random updates, not from formulas:
  "= mean"    output equals the plain average (rule does nothing robust)
  "1 client"  output is exactly one client's update (others are ignored)
Also reports ALIE's z per (n, f): z <= 0 means the ALIE attack sends the
honest mean (or moves towards it), i.e. it is not an attack at all.

Usage: python validity_table.py [--ns 3,4,5,6,20] [--fs 0,1,2]
Writes results_tables/validity.csv and prints a table.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from bench.aggregators import AGGREGATORS, check_valid, get_aggregator
from bench.attacks import alie_z
from sweep import csv


def degeneracy(name, n, f, trials=5, d=50):
    g = torch.Generator().manual_seed(0)
    labels = []
    for _ in range(trials):
        U = torch.randn(n, d, generator=g)
        out = get_aggregator(name)(U, f=f)  # fresh CC state each call
        if torch.allclose(out, U.mean(0), atol=1e-6):
            labels.append("= mean")
        elif any(torch.equal(out, U[i]) for i in range(n)):
            labels.append("1 client")
        else:
            labels.append("")
    # Report a degeneracy only if it happened in every trial
    return labels[0] if len(set(labels)) == 1 else ""


def build(ns, fs):
    rows = []
    for n in ns:
        for f in fs:
            z = alie_z(n, f) if f > 0 else float("nan")
            for name in AGGREGATORS:
                err = check_valid(name, n, f)
                rows.append({
                    "n": n, "f": f, "rule": name,
                    "status": "invalid" if err else "valid",
                    "reason": err or "",
                    "degenerate": "" if err else degeneracy(name, n, f),
                    "alie_z": round(z, 3),
                    "alie_degenerate": f > 0 and z <= 0,
                })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ns", type=csv(int), default=[3, 4, 5, 6, 20])
    p.add_argument("--fs", type=csv(int), default=[0, 1, 2])
    p.add_argument("--out", default="results_tables")
    a = p.parse_args()
    df = build(a.ns, a.fs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "validity.csv", index=False)

    cell = df.apply(lambda r: "INVALID" if r.status == "invalid"
                    else (r.degenerate or "ok"), axis=1)
    wide = df.assign(cell=cell).pivot_table(index=["n", "f"], columns="rule",
                                            values="cell", aggfunc="first")
    print(wide[list(AGGREGATORS)].to_string())
    print("\nALIE z per (n, f>0):")
    print(df[df.f > 0].drop_duplicates(["n", "f"])[["n", "f", "alie_z", "alie_degenerate"]]
          .to_string(index=False))
    print(f"\nsaved {out / 'validity.csv'}")


if __name__ == "__main__":
    main()
