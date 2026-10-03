"""Fed-Heart-Disease grid: who pays for robustness (RQ-B) and per-attacker results.

All metrics are means over the last 3 evaluations (summarize.run_metrics);
95% CIs are t intervals over the 20 seeds.

Tables (results_tables/):
  heart_attacker_f1.csv        f = 1: rule x weighting x attack x attacker
                               hospital -> average accuracy (sample-weighted
                               over hospitals), worst honest hospital (main),
                               worst hospital incl. attacker (appendix). Includes a clean
                               reference (f_assumed = 1) and the mean over the
                               real attacks (per seed, then CI). ALIE at n = 4,
                               f = 1 has z = 0 and is marked degenerate.
  heart_attacker_f1_wide.csv   avg + worst honest, one column per attacker hospital
  heart_attacker_f1_wide_appendix.csv   worst hospital incl. the attacker
  heart_exclusion_fa1.csv      clean runs, f_assumed = 1: per rule x weighting x
                               hospital, how much the rule down-weights it
  heart_exclusion_size_corr.csv  Spearman rho between hospital size and
                               relative influence, per rule x weighting
Figures:
  fig_heart_avg_vs_worst.{png,pdf}  clean runs: average vs worst-hospital accuracy

Relative influence of a hospital in a round = its effective share of the
aggregate / its nominal share (1/n, or its sample share when weighted);
< 1 means the rule down-weighted it. Effective share per rule:
  fedavg, geomed   the averaging weights
  median           share of coordinates whose median value it sets
  trimmed_mean     (1 - trimmed share) x nominal, renormalised (exact when
                   unweighted; approximate when weighted)
  cc               clip factor x nominal (relative influence = clip factor)
  (multi_)krum     selected x nominal, renormalised
  bulyan           kept share x nominal, renormalised

Usage: python analyze_heart.py [--folder results_heart] [--rebuild]
"""
import argparse
import json
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bench.plotstyle import GRID, INK, MUTED, RULE_COLORS, RULE_LABELS, plt, save
from summarize import ci95, client_test_sizes, is_degenerate, run_metrics

OUT = Path("results_tables")
HOSPITALS = ["Cleveland", "Hungary", "Switzerland", "VA"]
REAL_ATTACKS = ["label_flip", "sign_flip", "gaussian", "ipm"]  # alie: z = 0 at n=4, f=1
RATE_NAME = {"fedavg": "", "geomed": "rounds down-weighted", "median": "share of coords set",
             "trimmed_mean": "share of coords trimmed", "cc": "rounds clipped",
             "krum": "rounds excluded", "multi_krum": "rounds excluded",
             "bulyan": "share of coords not kept"}


def influence(r: dict):
    """Per hospital, averaged over rounds: relative influence, the rule's own
    exclusion rate (RATE_NAME; NaN for fedavg), the share of rounds with
    relative influence < 1, and the distance of its update from the previous
    aggregate (what the robust rules react to)."""
    agg, n = r["config"]["aggregator"], len(r["client_sizes"])
    sizes = np.asarray(r["client_sizes"], dtype=float)
    nom = sizes / sizes.sum() if r["config"]["weighted"] else np.full(n, 1 / n)
    rel, rate = [], []
    for d in r["diag"]:
        if agg == "fedavg":
            share, native = nom, np.full(n, np.nan)   # nominal by definition
        elif agg == "geomed":
            share = np.asarray(d["weights"])
            native = (share < nom * (1 - 1e-6)).astype(float)
        elif agg == "median":
            share = native = np.asarray(d["used_share"])
        elif agg == "trimmed_mean":
            native = np.asarray(d["trimmed_share"])
            share = (1 - native) * nom / ((1 - native) * nom).sum()
        elif agg == "cc":
            share = np.asarray(d["scale"]) * nom
            native = np.asarray(d["clipped"], dtype=float)
        elif agg in ("krum", "multi_krum"):
            sel = np.zeros(n)
            sel[d["selected"]] = 1
            share, native = sel * nom / (sel * nom).sum(), 1 - sel
        elif agg == "bulyan":
            kept = np.asarray(d["kept_share"])
            share, native = kept * nom / (kept * nom).sum(), 1 - kept
        else:
            raise ValueError(agg)
        rel.append(share / nom)
        rate.append(native)
    rel, rate = np.array(rel), np.array(rate)
    dist = np.array([d["dist_prev"] for d in r["diag"]])
    return rel.mean(0), rate.mean(0), (rel < 1 - 1e-6).mean(0), dist.mean(0)


def _extract(path, test_sizes=None):
    r = json.loads(Path(path).read_text())
    if r.get("eval_split", "test") != "test":
        return None
    c = r["config"]
    row = dict(aggregator=c["aggregator"], weighted=c["weighted"], attack=c["attack"],
               f=r["f"], f_assumed=r["f_assumed"], seed=c["seed"], rounds=c["rounds"],
               attacker=c["byz_ids"] or "-", status=r["status"], degenerate=is_degenerate(r))
    if r["status"] != "ok":
        return row
    m = run_metrics(r, test_sizes=test_sizes)
    for i, a in enumerate(m.pop("client_acc")):
        row[f"acc_{i}"] = a
    row.update(m, sizes=r["client_sizes"])
    if c["attack"] == "none":
        row["rel"], row["rate"], row["down"], row["dist"] = (x.tolist() for x in influence(r))
    return row


def collect(folder: Path, rebuild=False) -> pd.DataFrame:
    """One row per run; cached in results_tables/ (rebuilt when files change)."""
    files = sorted(folder.glob("*.json"))
    stamp = [len(files), max(p.stat().st_mtime for p in files)]
    cache, meta = OUT / f"cache_{folder.name}.pkl", OUT / f"cache_{folder.name}.json"
    if not rebuild and cache.exists() and meta.exists() and json.loads(meta.read_text()) == stamp:
        return pd.read_pickle(cache)
    print(f"parsing {len(files)} result files ...")
    with Pool(os.cpu_count()) as pool:
        work = partial(_extract, test_sizes=client_test_sizes("fed_heart_disease"))
        rows = [x for x in pool.map(work, files, chunksize=64) if x is not None]
    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_pickle(cache)
    meta.write_text(json.dumps(stamp))
    return df


def mean_ci(x):
    x = np.asarray(x, dtype=float)
    return x.mean(), ci95(x)


def attacker_table(df: pd.DataFrame) -> pd.DataFrame:
    """f = 1: one row per rule x weighting x attack x attacker hospital."""
    metrics = ["avg_acc", "worst_honest_acc", "worst_acc"]
    rows = []

    def add(g, attack, attacker, degenerate):
        row = dict(aggregator=g.aggregator.iloc[0], weighted=g.weighted.iloc[0],
                   attack=attack, degenerate=degenerate, attacker=attacker,
                   n_seeds=g.seed.nunique())
        if (g.status == "invalid").all():
            row["status"] = "invalid"
        else:
            row["status"] = "ok"
            for m in metrics:
                row[f"{m}_mean"], row[f"{m}_ci95"] = mean_ci(g[m])
        rows.append(row)

    att = df[(df.f == 1) & (df.attack != "none")]
    for (agg, wt, attack, attacker), g in att.groupby(["aggregator", "weighted", "attack", "attacker"]):
        add(g, attack, HOSPITALS[int(attacker)], bool(g.degenerate.any()))
    # mean over the real attacks: per seed first, so the CI is over seeds
    real = att[att.attack.isin(REAL_ATTACKS)]
    for (agg, wt, attacker), g in real.groupby(["aggregator", "weighted", "attacker"]):
        if (g.status == "invalid").all():
            per_seed = g.drop_duplicates("seed")
        else:
            per_seed = g.groupby("seed", as_index=False).agg(
                {**{m: "mean" for m in metrics}, "aggregator": "first", "weighted": "first",
                 "status": "first"})
        add(per_seed, "mean of real attacks", HOSPITALS[int(attacker)], False)
    clean = df[(df.attack == "none") & (df.f_assumed == 1)]
    for (agg, wt), g in clean.groupby(["aggregator", "weighted"]):
        add(g, "none (f_assumed=1)", "-", False)
    t = pd.DataFrame(rows)
    order = {a: i for i, a in enumerate(RULE_LABELS)}
    t = t.sort_values(["aggregator", "weighted", "attack", "attacker"],
                      key=lambda s: s.map(order) if s.name == "aggregator" else s)
    num = [c for c in t if c.endswith(("_mean", "_ci95"))]
    t[num] = (t[num] * 100).round(2)  # percentage points
    return t.reset_index(drop=True)


def wide(t: pd.DataFrame, metrics=("avg_acc", "worst_honest_acc")) -> pd.DataFrame:
    """Cells 'mean ± ci' (points), columns: metric x attacker hospital."""
    t = t[t.attacker != "-"].copy()
    metrics = list(metrics)
    for m in metrics:
        t[m] = [("invalid" if s == "invalid" else f"{a:.1f} ± {c:.1f}")
                for s, a, c in zip(t.status, t[f"{m}_mean"], t[f"{m}_ci95"])]
    t["attack"] = t.attack + np.where(t.degenerate, " (degenerate, z=0)", "")
    w = t.pivot_table(index=["aggregator", "weighted", "attack"], columns="attacker",
                      values=metrics, aggfunc="first")
    return w.reindex(columns=pd.MultiIndex.from_product([metrics, HOSPITALS]))


def exclusion_table(df: pd.DataFrame):
    """Clean runs, f_assumed = 1: who each rule down-weights, and size correlation."""
    clean = df[(df.attack == "none") & (df.f_assumed == 1) & (df.status == "ok")]
    rows, corr = [], []
    for (agg, wt), g in clean.groupby(["aggregator", "weighted"], sort=False):
        sizes = np.asarray(g.sizes.iloc[0], dtype=float)
        nom = sizes / sizes.sum() if wt else np.full(len(sizes), 1 / len(sizes))
        rel, rate, down, dist = (np.array(g[k].tolist())   # seeds x hospitals
                                 for k in ("rel", "rate", "down", "dist"))
        for i, h in enumerate(HOSPITALS):
            r_m, r_c = mean_ci(rel[:, i])
            rows.append(dict(aggregator=agg, weighted=wt, hospital=h, train_size=int(sizes[i]),
                             nominal_share=round(nom[i], 4), rate_name=RATE_NAME[agg],
                             rate_mean=rate[:, i].mean(), rel_influence_mean=r_m,
                             rel_influence_ci95=r_c, rounds_downweighted=down[:, i].mean(),
                             update_dist_mean=dist[:, i].mean()))
        # Spearman rho between size and mean relative influence (4 hospitals:
        # descriptive only; the smallest two-sided exact p-value is 2/24 = 0.083)
        varies = rel.std() > 1e-6                     # fedavg: influence is 1 everywhere
        rho = stats.spearmanr(sizes, rel.mean(0)).statistic if varies else np.nan
        per_seed = np.array([stats.spearmanr(sizes, x).statistic for x in rel] if varies else [])
        rho_d = stats.spearmanr(dist.mean(0), rel.mean(0)).statistic if varies else np.nan
        corr.append(dict(aggregator=agg, weighted=wt, rho_size_vs_influence=rho,
                         seeds_rho_pos=int(np.sum(per_seed > 0)), seeds=len(rel),
                         rho_update_dist_vs_influence=rho_d,
                         most_downweighted=HOSPITALS[int(rel.mean(0).argmin())] if varies else "-"))
    order = {a: i for i, a in enumerate(RULE_LABELS)}
    t = pd.DataFrame(rows).sort_values(["aggregator", "weighted"], key=lambda s: s.map(order)
                                       if s.name == "aggregator" else s, kind="stable")
    c = pd.DataFrame(corr).sort_values(["aggregator", "weighted"], key=lambda s: s.map(order)
                                       if s.name == "aggregator" else s, kind="stable")
    return t.round(4).reset_index(drop=True), c.round(3).reset_index(drop=True)


def _join(names):
    """Label for rules that coincide: 'A = B'; for more, 'A*' (footnote lists them)."""
    if len(names) == 2 and names[1] == names[0] + " (w)":
        return names[0] + " (w and unweighted)"
    return " = ".join(names) if len(names) <= 2 else names[0] + "*"


def _place_labels(fig, ax, labels):
    """Direct labels: per point, the first of four positions that overlaps no
    earlier label and no marker."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    markers = [ax.transData.transform(k) for k in labels]
    placed = []
    for (x, y), text in labels.items():
        for dx, dy, ha, va in ((7, 3, "left", "bottom"), (7, -3, "left", "top"),
                               (-7, 3, "right", "bottom"), (-7, -3, "right", "top"),
                               (0, -9, "center", "top"), (0, 9, "center", "bottom")):
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                            ha=ha, va=va, fontsize=8, color=INK)
            bb = t.get_window_extent(r).expanded(1.05, 1.15)
            hits_marker = any(bb.contains(*m) for m in markers)
            inside = ax.get_window_extent(r).contains(bb.x0, bb.y0) and \
                ax.get_window_extent(r).contains(bb.x1, bb.y1)
            if inside and not hits_marker and not any(bb.overlaps(p) for p in placed):
                placed.append(bb)
                break
            t.remove()
        else:  # nothing free: keep the default position
            placed.append(ax.annotate(text, (x, y), xytext=(7, 3), textcoords="offset points",
                                      fontsize=8, color=INK).get_window_extent(r))


def fig_avg_vs_worst(df: pd.DataFrame, f_assumed=1):
    """Clean runs: average (pooled test) vs worst-hospital accuracy, one point
    per rule x weighting, 95% CI bars over seeds."""
    clean = df[(df.attack == "none") & (df.f_assumed == f_assumed) & (df.status == "ok")]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    pts, labels = [], {}
    order = {a: i for i, a in enumerate(RULE_LABELS)}
    groups = sorted(clean.groupby(["aggregator", "weighted"]), key=lambda kv: (order[kv[0][0]], kv[0][1]))
    # draw in reverse rule order, so where rules coincide the first rule's colour is on top
    for (agg, wt), g in reversed(groups):
        (x, xc), (y, yc) = mean_ci(g.avg_acc * 100), mean_ci(g.worst_acc * 100)
        pts.append((agg, wt, x, y))
        ax.errorbar(x, y, xerr=xc, yerr=yc, fmt="none", ecolor=RULE_COLORS[agg],
                    elinewidth=1, alpha=0.6, zorder=2)
        ax.plot(x, y, "s" if wt else "o", ms=8, color=RULE_COLORS[agg],
                mec="white", mew=2, zorder=3)
        # rules that coincide exactly (e.g. trimmed mean = median at n=4, f=1)
        # share one label
        key = (round(x, 6), round(y, 6))
        labels.setdefault(key, []).append(RULE_LABELS[agg] + (" (w)" if wt else ""))
    xs, ys = [p[2] for p in pts], [p[3] for p in pts]
    ax.set_xlim(min(xs) - 2, max(xs) + 3)
    ax.set_ylim(min(ys) - 4, max(ys) + 4)
    _place_labels(fig, ax, {k: _join(v[::-1]) for k, v in labels.items()})
    same = [v[::-1] for v in labels.values() if len(v) > 2]
    if same:
        note = "; ".join(" = ".join(v) for v in same)
        fig.text(0.01, -0.02, f"* identical runs: {note}", fontsize=6.5, color=MUTED,
                 ha="left", va="top", wrap=True)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Average accuracy over all hospitals' test patients (%)")
    ax.set_ylabel("Worst-hospital accuracy (%)")
    ax.set_title(f"Fed-Heart-Disease, no attack, server assumes f = {f_assumed}",
                 loc="left", fontsize=9, color=MUTED)
    handles = [plt.Line2D([], [], ls="", marker=m, ms=7, color=MUTED, mec="white", mew=1.5,
                          label=l) for m, l in (("o", "unweighted"), ("s", "weighted (w)"))]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    save(fig, OUT / f"fig_heart_avg_vs_worst_fa{f_assumed}")
    return pd.DataFrame(pts, columns=["aggregator", "weighted", "avg_acc", "worst_acc"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", default="results_heart")
    p.add_argument("--rebuild", action="store_true", help="re-parse the result files")
    a = p.parse_args()
    df = collect(Path(a.folder), a.rebuild)
    df = df[df.rounds == df.rounds.max()] if df.rounds.nunique() > 1 else df
    print(f"{len(df)} runs ({(df.status == 'ok').sum()} valid), "
          f"{int(df.degenerate.sum())} degenerate (ALIE z = 0)")

    t = attacker_table(df)
    t.to_csv(OUT / "heart_attacker_f1.csv", index=False)
    wide(t).to_csv(OUT / "heart_attacker_f1_wide.csv")
    wide(t, ["worst_acc"]).to_csv(OUT / "heart_attacker_f1_wide_appendix.csv")
    ex, corr = exclusion_table(df)
    ex.to_csv(OUT / "heart_exclusion_fa1.csv", index=False)
    corr.to_csv(OUT / "heart_exclusion_size_corr.csv", index=False)
    fig_avg_vs_worst(df, 1)
    fig_avg_vs_worst(df, 0)
    print("saved heart_attacker_f1{,_wide,_wide_appendix}.csv, heart_exclusion_fa1.csv, "
          "heart_exclusion_size_corr.csv, fig_heart_avg_vs_worst_fa{0,1}.{png,pdf} "
          f"in {OUT}/")


if __name__ == "__main__":
    main()
