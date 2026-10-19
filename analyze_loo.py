"""Leave-out controls (2026-10-18): how much of "client j attacks
-> accuracy falls" is the loss of j's honest data, and how much is the attack.

Per dataset, rule, weighting, left-out client(s) J and seed:
  clean     the main grid's clean run with every client (f_assumed = f)
  control   the clean run without J (results_<ds>_loo/, f_assumed = f)
  attacked  the main grid's runs attacked by J, mean of real attacks (per
            seed, as in the main tables)
  data loss     = control - clean
  attack damage = attacked - control
  total         = attacked - clean  (= data loss + attack damage)
  attack damage without the ALIE variants (sensitivity column, *_no_alie)
for the main average (avg_acc, each dataset's own) and the worst honest
client, taken over the same clients in all three runs: every client not in
J (TCGA: only regions with >= 3 test deaths). Means and 95% t CIs over
seeds (points, x 100). Also the left-out client's own test accuracy: control
(did not take part) minus clean (took part).

Tables (results_tables/): loo_<ds>_f{1,2}.csv (decomposition),
loo_<ds>_f{1,2}_by_attack.csv (attack damage per attack),
loo_<ds>_own.csv (the left-out client's own test set, f = 1).
`dictator`: weighted median or geomed in a control where one remaining
client holds > 50% of the weight (validity by data share, 2026-10-18).

Usage: python analyze_loo.py [heart|tcga|isic ...]
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_heart import OUT, collect, mean_ci
from summarize import run_metrics

DATASETS = {  # name: (dataset, main folder, controls folder, client names, real attacks per f)
    "heart": ("fed_heart_disease", "results_heart", "results_heart_loo",
              ["Cleveland", "Hungary", "Switzerland", "VA"],
              {1: ["label_flip", "sign_flip", "gaussian", "ipm", "alie_z1"]}),
    "tcga": ("fed_tcga_brca", "results_tcga", "results_tcga_loo",
             ["Northeast", "South", "West", "Midwest", "Europe", "Canada"],
             {1: ["sign_flip", "gaussian", "ipm", "alie_z1"],
              2: ["sign_flip", "gaussian", "alie", "alie_z1", "ipm"]}),
    "isic": ("fed_isic2019", "results_isic", "results_isic_loo",
             [f"centre {i}" for i in range(6)],
             {1: ["label_flip", "sign_flip", "gaussian", "ipm", "alie_z1"],
              2: ["label_flip", "sign_flip", "gaussian", "ipm", "alie_z1", "alie"]}),
}
WORST_ELIGIBLE = {"fed_tcga_brca": [0, 1, 2, 3]}   # regions with >= 3 test deaths


def _control_row(path):
    r = json.loads(Path(path).read_text())
    c = r["config"]
    row = dict(aggregator=c["aggregator"], weighted=c["weighted"], seed=c["seed"],
               f_assumed=r["f_assumed"], status=r["status"], rounds=c["rounds"],
               left_out=",".join(map(str, sorted(int(i) for i in c["exclude_ids"].split(",")))))
    if r["status"] == "ok":
        m = run_metrics(r, test_sizes=r.get("client_test_sizes"))
        row["avg_acc"] = m["avg_acc"]
        for i, a in enumerate(m["client_acc"]):
            row[f"acc_{i}"] = a
        sizes = np.asarray(r["client_sizes"], dtype=float)
        keep = [i for i in range(len(sizes)) if i not in r["excluded_ids"]]
        row["dictator"] = bool(c["weighted"] and c["aggregator"] in ("median", "geomed")
                               and sizes[keep].max() > sizes[keep].sum() / 2)
    return row


def controls(folder):
    files = sorted(Path(folder).glob("*.json"))
    with Pool() as pool:
        return pd.DataFrame(pool.map(_control_row, files, chunksize=16))


def worst(df, n, out, eligible):
    """Worst accuracy over the eligible clients not in `out` (a set)."""
    cols = [f"acc_{i}" for i in eligible if i not in out]
    return df[cols].min(axis=1)


def decompose(name, f):
    ds, main, ctrl_dir, names, real = DATASETS[name]
    n = len(names)
    eligible = WORST_ELIGIBLE.get(ds, list(range(n)))
    att = collect(Path(main), dataset=ds)
    att = att[(att.rounds == att.rounds.max()) & (att.status == "ok")]
    ctrl = controls(ctrl_dir)
    ctrl = ctrl[(ctrl.status == "ok") & (ctrl.f_assumed == f)]
    rows, by_attack = [], []
    for (agg, wt, lo), c in ctrl.groupby(["aggregator", "weighted", "left_out"]):
        out = {int(i) for i in lo.split(",")}
        clean = att[(att.attack == "none") & (att.f_assumed == f) &
                    (att.aggregator == agg) & (att.weighted == wt)]
        a = att[(att.f == f) & (att.attacker.astype(str) == lo) & att.attack.isin(real[f]) &
                (att.aggregator == agg) & (att.weighted == wt)]
        if a.empty:
            continue            # rule invalid in the attacked grid
        clean, c, a = (d.assign(worst_h=worst(d, n, out, eligible)) for d in (clean, c, a))
        m = ["avg_acc", "worst_h"]
        A = a.groupby("seed")[m].mean()                       # mean of real attacks per seed
        C, K = c.set_index("seed")[m], clean.set_index("seed")[m]
        seeds = sorted(set(A.index) & set(C.index) & set(K.index))
        A, C, K = A.loc[seeds], C.loc[seeds], K.loc[seeds]
        row = dict(rule=agg, weighted=wt, left_out="+".join(names[i] for i in sorted(out)),
                   n_seeds=len(seeds), dictator=bool(c.dictator.any()))
        for k, lab in (("avg_acc", "avg"), ("worst_h", "worst")):
            row[f"{lab}_clean"], row[f"{lab}_control"], row[f"{lab}_attacked"] = (
                K[k].mean(), C[k].mean(), A[k].mean())
            for part, x in (("data_loss", C[k] - K[k]), ("attack_damage", A[k] - C[k]),
                            ("total", A[k] - K[k])):
                row[f"{lab}_{part}"], row[f"{lab}_{part}_ci95"] = mean_ci(x)
        # Sensitivity: attack damage without the ALIE variants. ALIE builds its
        # vector from the mean of all benign updates, the attacker's own
        # included, so it hands back part of j's data and can score above a
        # control without j (2026-10-19).
        A2 = a[~a.attack.str.startswith("alie")].groupby("seed")[m].mean().loc[seeds]
        for k, lab in (("avg_acc", "avg"), ("worst_h", "worst")):
            row[f"{lab}_attack_damage_no_alie"], row[f"{lab}_attack_damage_no_alie_ci95"] = (
                mean_ci(A2[k] - C[k]))
        rows.append(row)
        for at, g in a.groupby("attack"):
            G = g.set_index("seed")[m].loc[[s for s in seeds if s in set(g.seed)]]
            d = G["avg_acc"] - C.loc[G.index, "avg_acc"]
            mu, ci = mean_ci(d)
            by_attack.append(dict(rule=agg, weighted=wt, left_out=row["left_out"], attack=at,
                                  avg_attack_damage=mu, ci95=ci))
    t, b = pd.DataFrame(rows), pd.DataFrame(by_attack)
    num = [c for c in t if c.startswith(("avg_", "worst_"))]
    t[num] = (t[num] * 100).round(2)
    b[["avg_attack_damage", "ci95"]] = (b[["avg_attack_damage", "ci95"]] * 100).round(2)
    return t, b, att, ctrl


def own_test_set(name, att, ctrl):
    """f = 1: the left-out client's own test accuracy, absent vs present."""
    ds, _, _, names, _ = DATASETS[name]
    rows = []
    for (agg, wt, lo), c in ctrl[ctrl.left_out.str.count(",") == 0].groupby(
            ["aggregator", "weighted", "left_out"]):
        j = int(lo)
        clean = att[(att.attack == "none") & (att.f_assumed == 1) &
                    (att.aggregator == agg) & (att.weighted == wt)].set_index("seed")
        x = c.set_index("seed")[f"acc_{j}"]
        seeds = sorted(set(x.index) & set(clean.index))
        d = x.loc[seeds] - clean.loc[seeds, f"acc_{j}"]
        mu, ci = mean_ci(d)
        rows.append(dict(rule=agg, weighted=wt, client=names[j],
                         eligible_for_worst=j in WORST_ELIGIBLE.get(ds, range(len(names))),
                         present=100 * clean.loc[seeds, f"acc_{j}"].mean(),
                         absent=100 * x.loc[seeds].mean(), change=100 * mu, ci95=100 * ci))
    return pd.DataFrame(rows).round(2)


def main():
    for name in sys.argv[1:] or list(DATASETS):
        for f in sorted(DATASETS[name][4]):
            if not any(Path(DATASETS[name][2]).glob(f"*_fa{f}_*.json")):
                print(f"{name}, f = {f}: no controls yet")
                continue
            t, b, att, ctrl = decompose(name, f)
            t.to_csv(OUT / f"loo_{name}_f{f}.csv", index=False)
            b.to_csv(OUT / f"loo_{name}_f{f}_by_attack.csv", index=False)
            if f == 1:
                own_test_set(name, att, ctrl[ctrl.f_assumed == 1]).to_csv(
                    OUT / f"loo_{name}_own.csv", index=False)
            print(f"saved loo_{name}_f{f}*.csv ({len(t)} rows)")


if __name__ == "__main__":
    main()
