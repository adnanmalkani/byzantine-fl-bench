"""Collect results/*.json into tables: mean and 95% CI over seeds.

  summary.csv          one row per configuration, including attacker
                       identity (byz_ids) for natural splits
  summary_pooled.csv   same, pooled over attacker identities

Metrics: final_acc (mean of the last 3 evals), and for natural splits the
worst-hospital accuracy and best-worst gap at the last eval. 95% CI uses the
t distribution over runs (seeds, or seeds x attacker sets when pooled).
Tuning runs (evaluated on a validation split) are skipped.

Usage: python summarize.py [results_dir]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

KEYS = ["dataset", "alpha", "rounds", "attack", "byz_frac", "f", "f_assumed",
        "aggregator", "weighted", "cc_tau"]
METRICS = ["final_acc", "worst_acc", "gap"]


def load(folder):
    rows, skipped = [], 0
    for p in Path(folder).glob("*.json"):
        r = json.loads(p.read_text())
        if r.get("eval_split", "test") != "test":
            skipped += 1
            continue
        c = r["config"]
        row = {k: c.get(k) for k in KEYS if k in c}
        row.update(f=r["f"], f_assumed=r["f_assumed"], status=r["status"],
                   weighted=c.get("weighted", False),
                   cc_tau=c["cc_tau"] if c["aggregator"] == "cc" else None,
                   byz_ids=",".join(map(str, r.get("byz_ids", []))) or "-")
        if r["status"] == "ok":
            last = r["history"][-1]
            row.update(final_acc=r["final_acc"], worst_acc=last.get("worst_acc"),
                       gap=last.get("gap"))
        rows.append(row)
    return pd.DataFrame(rows), skipped


def ci_table(df, keys):
    def agg(g):
        out = {"n_runs": len(g), "n_invalid": int((g.status == "invalid").sum())}
        ok = g[g.status == "ok"]
        for m in METRICS:
            x = ok[m].dropna().to_numpy(dtype=float)
            out[f"{m}_mean"] = x.mean() if len(x) else np.nan
            h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) \
                if len(x) > 1 else np.nan
            out[f"{m}_ci95"] = h
        return pd.Series(out)
    keys = [k for k in keys if df[k].notna().any()]
    t = (df.fillna({k: "-" for k in keys}).groupby(keys, dropna=False)
           .apply(agg, include_groups=False))
    return t.dropna(axis=1, how="all").round(4)


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "results"
    df, skipped = load(folder)
    if skipped:
        print(f"skipped {skipped} tuning runs (validation split)")
    if df.empty:
        sys.exit(f"no test-set results in {folder}/")
    per_id = ci_table(df, KEYS + ["byz_ids"])
    pooled = ci_table(df, KEYS)
    print(pooled.to_string())
    per_id.to_csv("summary.csv")
    pooled.to_csv("summary_pooled.csv")
    print("\nsaved summary.csv (per attacker identity) and summary_pooled.csv")


if __name__ == "__main__":
    main()
