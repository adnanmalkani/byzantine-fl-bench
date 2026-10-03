"""Collect a results folder into tables: mean and 95% CI over seeds.

  results_tables/summary_<folder>.csv          one row per configuration,
                                               including attacker identity
                                               (byz_ids) for natural splits
  results_tables/summary_<folder>_pooled.csv   same, pooled over attacker
                                               identities

Metrics, each the mean over the last 3 evaluations:
  final_acc         global test accuracy (image datasets only)
  avg_acc           natural splits: sample-weighted mean of the per-hospital
                    test accuracies (each hospital with its own normalisation),
                    i.e. accuracy over all hospitals' test patients. FLamby's
                    pooled-test metric is used only in the reproduction scripts.
  worst_honest_acc  lowest per-hospital accuracy over non-attacker hospitals
                    (main worst-hospital metric)
  worst_acc         lowest over all hospitals, attackers included (appendix)
  gap               best minus worst per-hospital accuracy
95% CI uses the t distribution over runs (seeds, or seeds x attacker sets
when pooled). Tuning runs (evaluated on a validation split) are skipped.

Degenerate runs: ALIE with z = 0 (e.g. n = 4, f = 1) sends exactly the mean
of all benign updates (the attacker's own honest update included), so it is
not an attack. Those runs get degenerate=True and are grouped separately
from real attacks.

Usage: python summarize.py [results_dir]
"""
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

KEYS = ["dataset", "alpha", "rounds", "attack", "degenerate", "byz_frac", "f",
        "f_assumed", "aggregator", "weighted", "cc_tau"]
METRICS = ["final_acc", "avg_acc", "worst_honest_acc", "worst_acc", "gap"]
LAST = 3  # evaluations averaged for every metric


def is_degenerate(r: dict) -> bool:
    """ALIE with z = 0: the attackers send the mean of the benign updates."""
    return r["config"]["attack"] == "alie" and abs(r.get("alie_z", 1.0)) < 1e-9


@lru_cache
def client_test_sizes(dataset: str) -> tuple:
    """Per-hospital test-set sizes, for results written before run() stored them."""
    import torch

    from bench.fl import make_config
    from bench.tasks import load_task
    task = load_task(make_config(dataset=dataset), torch.device("cpu"))
    return tuple(len(y) for _, y in task.client_tests)


def run_metrics(r: dict, last: int = LAST, test_sizes=None) -> dict:
    """Per-run metrics, each averaged over the last `last` evaluations.
    Natural splits: avg_acc replaces final_acc (FLamby's pooled test set)."""
    hist = r["history"][-last:]
    out = {"final_acc": float(np.mean([h["acc"] for h in hist]))}
    if "client_acc" in hist[0]:
        C = np.array([h["client_acc"] for h in hist])           # evals x hospitals
        if test_sizes is None:
            test_sizes = r.get("client_test_sizes")
        if test_sizes is None:
            assert r.get("eval_split", "test") == "test"
            test_sizes = client_test_sizes(r["config"]["dataset"])
        n = np.asarray(test_sizes, dtype=float)
        honest = [i for i in range(C.shape[1]) if i not in r.get("byz_ids", [])]
        del out["final_acc"]
        out.update(avg_acc=float((C @ n / n.sum()).mean()),
                   worst_honest_acc=C[:, honest].min(1).mean(), worst_acc=C.min(1).mean(),
                   gap=(C.max(1) - C.min(1)).mean(), client_acc=C.mean(0).tolist())
    return out


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
                   weighted=c.get("weighted", False), degenerate=is_degenerate(r),
                   cc_tau=c["cc_tau"] if c["aggregator"] == "cc" else None,
                   byz_ids=",".join(map(str, r.get("byz_ids", []))) or "-")
        if r["status"] == "ok":
            m = run_metrics(r)
            m.pop("client_acc", None)
            row.update(m)
        rows.append(row)
    return pd.DataFrame(rows), skipped


def ci95(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def ci_table(df, keys):
    def agg(g):
        out = {"n_runs": len(g), "n_invalid": int((g.status == "invalid").sum())}
        ok = g[g.status == "ok"]
        for m in METRICS:
            x = ok[m].dropna().to_numpy(dtype=float) if m in ok else np.array([])
            out[f"{m}_mean"] = x.mean() if len(x) else np.nan
            out[f"{m}_ci95"] = ci95(x)
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
    n_deg = int(df.degenerate.sum())
    if n_deg:
        print(f"\n{n_deg} runs flagged degenerate (ALIE z = 0: attackers send the "
              "benign mean); grouped under degenerate=True, not with real attacks")
    out = Path("results_tables")
    out.mkdir(exist_ok=True)
    name = Path(folder).resolve().name
    per_id.to_csv(out / f"summary_{name}.csv")
    pooled.to_csv(out / f"summary_{name}_pooled.csv")
    print(f"\nsaved {out}/summary_{name}.csv (per attacker identity) "
          f"and summary_{name}_pooled.csv")


if __name__ == "__main__":
    main()
