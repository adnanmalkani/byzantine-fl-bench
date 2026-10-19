"""Fed-ISIC2019 grid: clean results, per-attacker results, who pays for
robustness. Same tables as analyze_tcga.py, with ISIC's metrics (docs/PROTOCOL.md,
"Main-setting rules", pre-registered 2026-10-16):
  avg_acc           balanced accuracy over all centres' test images pooled
  worst_honest_acc  worst honest centre; per-centre balanced accuracy over
                    the classes with >= 5 test images there (k = 5, all 6
                    centres count)
  worst_acc         same, attackers included (appendix)
  gap               best minus worst centre
  *_k10             the same with k = 10 (appendix; centre 5 drops out)
All metrics are means over the last 3 evaluations; 95% CIs are t intervals
over seeds (20; 5 for Baruch's ALIE at f = 1). Tables show points (x 100).
"Mean of real attacks" = every attack that is not degenerate (2026-10-16): f = 1 label flip, sign flip, Gaussian, IPM, alie_z1 (Baruch's
ALIE has z = 0 at n = 6, f = 1); f = 2 the same plus Baruch's ALIE.

Tables (results_tables/):
  isic_clean.csv                 clean runs, f_assumed 0/1/2: avg, worst, gap
                                 (k = 5 and k = 10) and each centre's accuracy
  isic_attacker_f{1,2}.csv (+ _wide)        k = 5 (main)
  isic_attacker_f1_wide_appendix.csv        worst centre incl. the attacker
  isic_attacker_f{1,2}_k10.csv (+ _wide)    k = 10 (appendix)
  isic_exclusion_fa{1,2}.csv, isic_exclusion_size_corr_fa{1,2}.csv
                                 clean runs: how much each rule down-weights
                                 each centre (see analyze_heart.influence)
Figures: fig_isic_avg_vs_worst_fa{0,1,2}.{png,pdf}

Usage: python analyze_isic.py [--folder results_isic] [--rebuild]
"""
import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd

from analyze_heart import OUT, attacker_table, collect, exclusion_table, fig_avg_vs_worst, mean_ci, wide, who
from bench.plotstyle import RULE_LABELS

DATASET = "fed_isic2019"
CENTRES = [f"centre {i}" for i in range(6)]
PAIRS = [who(f"{i},{j}", CENTRES) for i, j in combinations(range(len(CENTRES)), 2)]
REAL_F1 = ["label_flip", "sign_flip", "gaussian", "ipm", "alie_z1"]
REAL_F2 = REAL_F1 + ["alie"]
MEAN_REAL = "mean of real attacks"
FIG = dict(prefix="fig_isic_avg_vs_worst", dataset="Fed-ISIC2019",
           xlabel="Pooled balanced accuracy (%)",
           ylabel="Worst-centre balanced accuracy, k = 5 (%)")
K10 = ["worst_honest_acc", "worst_acc", "gap"]


def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    """Clean runs: one row per rule x weighting x f_assumed."""
    rows = []
    for (agg, wt, fa), g in df[df.attack == "none"].groupby(["aggregator", "weighted", "f_assumed"]):
        row = dict(aggregator=agg, weighted=wt, f_assumed=fa, n_seeds=g.seed.nunique())
        if (g.status == "invalid").all():
            rows.append({**row, "status": "invalid"})
            continue
        row["status"] = "ok"
        for m in ("avg_acc", "worst_acc", "gap", "worst_acc_k10", "gap_k10"):
            row[f"{m}_mean"], row[f"{m}_ci95"] = mean_ci(g[m])
        for i, c in enumerate(CENTRES):
            row[f"{c}_mean"], row[f"{c}_ci95"] = mean_ci(g[f"acc_{i}"])
        rows.append(row)
    t = pd.DataFrame(rows)
    order = {a: i for i, a in enumerate(RULE_LABELS)}
    t = t.sort_values(["f_assumed", "aggregator", "weighted"],
                      key=lambda s: s.map(order) if s.name == "aggregator" else s)
    num = [c for c in t if c.endswith(("_mean", "_ci95"))]
    t[num] = (t[num] * 100).round(2)
    return t.reset_index(drop=True)


def k10(df: pd.DataFrame) -> pd.DataFrame:
    """The k = 10 metrics under the main names, so the same tables apply."""
    return df.drop(columns=K10).rename(columns={f"{m}_k10": m for m in K10})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", default="results_isic")
    p.add_argument("--rebuild", action="store_true", help="re-parse the result files")
    a = p.parse_args()
    df = collect(Path(a.folder), a.rebuild, dataset=DATASET)
    assert df.rounds.nunique() == 1, "mixed round counts in the folder"
    print(f"{len(df)} runs ({(df.status == 'ok').sum()} valid), "
          f"{int(df.degenerate.sum())} degenerate (ALIE z = 0)")

    clean_table(df).to_csv(OUT / "isic_clean.csv", index=False)
    for suffix, d in (("", df), ("_k10", k10(df))):
        t1 = attacker_table(d, means=((MEAN_REAL, REAL_F1),), names=CENTRES, f=1)
        t1.to_csv(OUT / f"isic_attacker_f1{suffix}.csv", index=False)
        wide(t1, names=CENTRES).to_csv(OUT / f"isic_attacker_f1{suffix}_wide.csv")
        t2 = attacker_table(d, means=((MEAN_REAL, REAL_F2),), names=CENTRES, f=2)
        t2.to_csv(OUT / f"isic_attacker_f2{suffix}.csv", index=False)
        wide(t2, names=PAIRS).to_csv(OUT / f"isic_attacker_f2{suffix}_wide.csv")
        if not suffix:
            wide(t1, ["worst_acc"], names=CENTRES).to_csv(OUT / "isic_attacker_f1_wide_appendix.csv")
    for fa in (1, 2):
        ex, corr = exclusion_table(df, names=CENTRES, f_assumed=fa)
        ex.to_csv(OUT / f"isic_exclusion_fa{fa}.csv", index=False)
        corr.to_csv(OUT / f"isic_exclusion_size_corr_fa{fa}.csv", index=False)
    for fa in (0, 1, 2):
        fig_avg_vs_worst(df, fa, **FIG)
    print("saved isic_clean.csv, isic_attacker_f{1,2}{,_k10}*.csv, isic_exclusion*_fa{1,2}.csv, "
          f"fig_isic_avg_vs_worst_fa{{0,1,2}}.{{png,pdf}} in {OUT}/")


if __name__ == "__main__":
    main()
