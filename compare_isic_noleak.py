"""Fed-ISIC2019 duplicates check (docs/PROTOCOL.md; decided 2026-10-16).

Pairs each clean run in results_isic_noleak/ (the 17 test images with an
exact copy in training dropped) with the same run in results_isic/, checks
that training was identical (same aggregator diagnostics every round), and
reports, per rule / weighting / f_assumed, the mean change in pooled
balanced accuracy and in the worst centre (k = 5), in points, with a 95% t
CI over seeds. Writes results_tables/isic_noleak_check.csv.

Usage: python compare_isic_noleak.py   (after the grid's clean runs exist)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from summarize import ci95, run_metrics

MAIN, NOLEAK = Path("results_isic"), Path("results_isic_noleak")


def main():
    rows, missing = [], 0
    for p in sorted(NOLEAK.glob("*_noleak_*.json")):
        q = MAIN / p.name.replace("_noleak", "")
        if not q.exists():
            missing += 1
            continue
        a, b = json.loads(q.read_text()), json.loads(p.read_text())
        if a["status"] != "ok":
            continue
        assert a["diag"] == b["diag"], f"training differs: {p.name}"
        ma, mb = run_metrics(a), run_metrics(b)
        c = a["config"]
        rows.append(dict(aggregator=c["aggregator"], weighted=c["weighted"],
                         f_assumed=a["f_assumed"], seed=c["seed"],
                         d_avg=100 * (mb["avg_acc"] - ma["avg_acc"]),
                         d_worst=100 * (mb["worst_acc"] - ma["worst_acc"])))
    if missing:
        print(f"{missing} runs have no main-grid twin yet (grid still running?)")
    if not rows:
        return
    df = pd.DataFrame(rows)
    t = df.groupby(["aggregator", "weighted", "f_assumed"]).agg(
        n=("seed", "size"), d_avg_mean=("d_avg", "mean"), d_avg_ci95=("d_avg", ci95),
        d_worst_mean=("d_worst", "mean"), d_worst_ci95=("d_worst", ci95)).round(3)
    print(t.to_string())
    print(f"\n{len(df)} pairs, training identical in all; largest |change|: "
          f"pooled {df.d_avg.abs().max():.3f} pt, worst centre {df.d_worst.abs().max():.3f} pt")
    Path("results_tables").mkdir(exist_ok=True)
    t.to_csv("results_tables/isic_noleak_check.csv")


if __name__ == "__main__":
    main()
