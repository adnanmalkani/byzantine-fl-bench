"""Fed-TCGA-BRCA grid: clean results, per-attacker results, who pays for
robustness. Same tables as analyze_heart.py, with TCGA's metrics (docs/PROTOCOL.md,
"Metrics", pre-registered 2026-10-11):
  avg_acc           stratified C-index (within-region pairs pooled)
  worst_honest_acc  worst honest region among those with >= 3 test deaths
                    (Northeast, South, West, Midwest)
  worst_acc         same, attackers included (appendix)
  gap               best minus worst of those 4 regions
All metrics are means over the last 3 evaluations; 95% CIs are t intervals
over seeds (20 for clean runs and f = 1 attacks, 5 for f = 2 and ALIE).
Tables show C-index points (x 100).

Tables (results_tables/):
  tcga_clean.csv               clean runs, f_assumed 0/1/2: avg, worst, gap
                               and each region's C-index, plus seeds whose
                               stratified C-index is below 0.5
  tcga_attacker_f1.csv (+ _wide, _wide_appendix)
                               f = 1, every region as the attacker; mean of
                               real attacks: sign flip, Gaussian, IPM, alie_z1
                               (ALIE: z = 0 at n = 6, f = 1, degenerate)
  tcga_attacker_f2.csv (+ _wide)
                               f = 2, all 15 region pairs; mean of real
                               attacks: sign flip, Gaussian, ALIE (z = 0.43),
                               alie_z1, IPM
  tcga_exclusion_fa{1,2}.csv, tcga_exclusion_size_corr_fa{1,2}.csv
                               clean runs: how much each rule down-weights
                               each region (see analyze_heart.influence)
Figures: fig_tcga_avg_vs_worst_fa{0,1,2}.{png,pdf}

Usage: python analyze_tcga.py [--folder results_tcga] [--rebuild]
"""
import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd

from analyze_heart import OUT, attacker_table, collect, exclusion_table, fig_avg_vs_worst, mean_ci, wide, who
from bench.plotstyle import RULE_LABELS

DATASET = "fed_tcga_brca"
REGIONS = ["Northeast", "South", "West", "Midwest", "Europe", "Canada"]
PAIRS = [who(f"{i},{j}", REGIONS) for i, j in combinations(range(len(REGIONS)), 2)]
REAL_F1 = ["sign_flip", "gaussian", "ipm", "alie_z1"]
REAL_F2 = ["sign_flip", "gaussian", "alie", "alie_z1", "ipm"]
MEAN_REAL = "mean of real attacks"
FIG = dict(prefix="fig_tcga_avg_vs_worst", dataset="Fed-TCGA-BRCA",
           xlabel="Stratified C-index over all regions (x 100)",
           ylabel="Worst region C-index, regions with >= 3 deaths (x 100)")


def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    """Clean runs: one row per rule x weighting x f_assumed."""
    rows = []
    clean = df[df.attack == "none"]
    for (agg, wt, fa), g in clean.groupby(["aggregator", "weighted", "f_assumed"]):
        row = dict(aggregator=agg, weighted=wt, f_assumed=fa, n_seeds=g.seed.nunique())
        if (g.status == "invalid").all():
            rows.append({**row, "status": "invalid"})
            continue
        row["status"] = "ok"
        for m in ("avg_acc", "worst_acc", "gap"):
            row[f"{m}_mean"], row[f"{m}_ci95"] = mean_ci(g[m])
        for i, r in enumerate(REGIONS):
            row[f"{r}_mean"], row[f"{r}_ci95"] = mean_ci(g[f"acc_{i}"])
        row["seeds_below_0.5"] = int((g.avg_acc < 0.5).sum())
        rows.append(row)
    t = pd.DataFrame(rows)
    order = {a: i for i, a in enumerate(RULE_LABELS)}
    t = t.sort_values(["f_assumed", "aggregator", "weighted"],
                      key=lambda s: s.map(order) if s.name == "aggregator" else s)
    num = [c for c in t if c.endswith(("_mean", "_ci95"))]
    t[num] = (t[num] * 100).round(2)
    return t.reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", default="results_tcga")
    p.add_argument("--rebuild", action="store_true", help="re-parse the result files")
    a = p.parse_args()
    df = collect(Path(a.folder), a.rebuild, dataset=DATASET)
    assert df.rounds.nunique() == 1, "mixed round counts in the folder"
    print(f"{len(df)} runs ({(df.status == 'ok').sum()} valid), "
          f"{int(df.degenerate.sum())} degenerate (ALIE z = 0)")

    clean_table(df).to_csv(OUT / "tcga_clean.csv", index=False)
    t1 = attacker_table(df, means=((MEAN_REAL, REAL_F1),), names=REGIONS, f=1)
    t1.to_csv(OUT / "tcga_attacker_f1.csv", index=False)
    wide(t1, names=REGIONS).to_csv(OUT / "tcga_attacker_f1_wide.csv")
    wide(t1, ["worst_acc"], names=REGIONS).to_csv(OUT / "tcga_attacker_f1_wide_appendix.csv")
    t2 = attacker_table(df, means=((MEAN_REAL, REAL_F2),), names=REGIONS, f=2)
    t2.to_csv(OUT / "tcga_attacker_f2.csv", index=False)
    wide(t2, names=PAIRS).to_csv(OUT / "tcga_attacker_f2_wide.csv")
    for fa in (1, 2):
        ex, corr = exclusion_table(df, names=REGIONS, f_assumed=fa)
        ex.to_csv(OUT / f"tcga_exclusion_fa{fa}.csv", index=False)
        corr.to_csv(OUT / f"tcga_exclusion_size_corr_fa{fa}.csv", index=False)
    for fa in (0, 1, 2):
        fig_avg_vs_worst(df, fa, **FIG)
    print("saved tcga_clean.csv, tcga_attacker_f{1,2}*.csv, tcga_exclusion*_fa{1,2}.csv, "
          f"fig_tcga_avg_vs_worst_fa{{0,1,2}}.{{png,pdf}} in {OUT}/")


if __name__ == "__main__":
    main()
