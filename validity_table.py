"""Validity table (RQ-A): for each rule, number of clients n and assumed f,
is the run valid, and if so does the rule degenerate?

Degeneracy is detected numerically on random updates, not from formulas:
  "= mean"    output equals the plain average (rule does nothing robust)
  "= median"  output equals the coordinate-wise median (e.g. trimmed mean
              when n - 2f <= 2: it is a different rule only in name)
  "1 client"  output is exactly one client's update (others are ignored)
Also reports ALIE's z per (n, f): z <= 0 means the ALIE attack sends the
mean of the benign updates (or moves towards it), i.e. it is not an attack.

Weighted variants (--weights; decided 2026-10-18): validity also depends on
data share, not only on n and f. With each FLamby dataset's real training
sizes as weights (all clients, and each leave-one-out subset), two checks,
both numerical and both counted only if the unweighted rule at the same
(n, f) passes:
  dictator      clean: the weighted output is the heaviest client's update
                (within 5% of the way from it to the weighted mean; geomed is
                10 Weiszfeld iterations, so it is close but not exact). The
                rule is then that client's model.
  breakdown     the f heaviest clients attack (Byzantine clients report their
                true sizes) by pushing a fixed direction every round; the
                aggregate ends more than 10x the honest spread away (CC: over
                2,000 rounds, as only its state carries over)
  breakdown_exact  geomed only: the same with Weiszfeld run to convergence
                (1,000 iterations); where it differs from breakdown, the cause
                is the 10-iteration approximation, not the weights
Writes results_tables/validity_weighted.csv.

Usage: python validity_table.py [--ns 3,4,5,6,20] [--fs 0,1,2] [--highlight 4]
       python validity_table.py --weights
Writes results_tables/validity.csv, fig_validity_heatmap.{png,pdf} and
prints a table.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from bench.aggregators import AGGREGATORS, check_valid, get_aggregator
from bench.attacks import alie_z
from bench.plotstyle import GRID, INK, MUTED, RULE_LABELS, SURFACE, plt, save
from sweep import csv

# Heatmap states. FedAvg "= mean" and Krum "1 client" are what those rules
# are by design, so they count as plain valid cells.
VALID, COLLAPSED, INVALID = "#2a78d6", "#b7d3f6", "#e4e3df"


def degeneracy(name, n, f, trials=5, d=50):
    g = torch.Generator().manual_seed(0)
    labels = []
    for _ in range(trials):
        U = torch.randn(n, d, generator=g)
        out = get_aggregator(name)(U, f=f)  # fresh CC state each call
        if torch.allclose(out, U.mean(0), atol=1e-6):
            labels.append("= mean")
        elif name != "median" and torch.allclose(out, get_aggregator("median")(U), atol=1e-6):
            labels.append("= median")
        elif any(torch.equal(out, U[i]) for i in range(n)):
            labels.append("1 client")
        else:
            labels.append("")
    # Report a degeneracy only if it happened in every trial
    return labels[0] if len(set(labels)) == 1 else ""


def _agg(name, iters=None):
    if name == "geomed" and iters:
        return lambda U, f=0, w=None: AGGREGATORS["geomed"](U, f=f, w=w, iters=iters)
    return get_aggregator(name)


def dictator(name, w, trials=5, d=50):
    """Clean random updates: is the output the heaviest client's update
    (within 5% of the way from it to the weighted mean), in every trial?"""
    g = torch.Generator().manual_seed(0)
    n, top = len(w), int(w.argmax())
    hits = []
    for _ in range(trials):
        U = torch.randn(n, d, generator=g)
        out = get_aggregator(name)(U, f=0, w=w)
        mean = (U * w[:, None]).sum(0) / w.sum()
        hits.append(float((out - U[top]).norm() / (mean - U[top]).norm()) < 0.05)
    return all(hits)


def breakdown(name, w, f, rounds=2000, d=5, push=1e3, iters=None):
    """The f heaviest clients push +push along e every round (relative to the
    previous aggregate, as CC clips around it); honest updates ~ N(0, I).
    Breakdown if the aggregate ends more than 10x the honest spread along e."""
    g = torch.Generator().manual_seed(0)
    n = len(w)
    byz = set(w.argsort(descending=True)[:f].tolist())
    order = [i for i in range(n) if i not in byz] + sorted(byz)
    w = w[order]                                    # honest rows first, as in run()
    agg, v = _agg(name, iters), torch.zeros(d)
    for _ in range(rounds if name == "cc" else 1):  # only CC carries state
        H = torch.randn(n - f, d, generator=g)
        M = (v + push * torch.eye(d)[0]).expand(f, d)
        v = agg(torch.cat([H, M]), f=f, w=w)
    return bool(v[0] > 10 * d ** 0.5)


def build_weighted(datasets):
    """datasets: {name: (client names, training sizes)}."""
    rows = []
    for ds, (names, sizes) in datasets.items():
        n = len(sizes)
        cases = [("all clients", list(range(n)), [f for f in (1, 2) if n > 2 * f])]
        cases += [(f"without {names[j]}", [i for i in range(n) if i != j], [])
                  for j in range(n)]                # leave-one-out controls: clean
        for label, keep, fs in cases:
            w = torch.tensor([float(sizes[i]) for i in keep])
            share = sorted((w / w.sum()).tolist(), reverse=True)
            for name in AGGREGATORS:
                row = {"dataset": ds, "clients": label, "n": len(keep), "rule": name,
                       "max_share": round(share[0], 3),
                       "dictator": dictator(name, w) and not dictator(name, torch.ones(len(keep)))}
                for f in (1, 2):
                    if f not in fs or check_valid(name, len(keep), f):
                        row[f"breakdown_f{f}"] = None    # not attacked / invalid anyway
                        continue
                    row[f"top{f}_share"] = round(sum(share[:f]), 3)
                    row[f"breakdown_f{f}"] = (breakdown(name, w, f)
                                              and not breakdown(name, torch.ones(len(keep)), f))
                    if name == "geomed":
                        row[f"breakdown_exact_f{f}"] = (
                            breakdown(name, w, f, iters=1000)
                            and not breakdown(name, torch.ones(len(keep)), f, iters=1000))
                rows.append(row)
    return pd.DataFrame(rows)


def flamby_sizes():
    from bench.fl import make_config
    from bench.tasks import load_task
    out = {}
    for ds in ("fed_heart_disease", "fed_tcga_brca", "fed_isic2019"):
        t = load_task(make_config(dataset=ds), torch.device("cpu"))
        out[ds] = (t.client_names, [len(y) for _, y in t.clients])
    return out


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


def state(r):
    if r.status == "invalid":
        return "invalid"
    if r.degenerate in ("= mean", "= median") and r.rule != "fedavg":
        return "collapsed"
    return "valid"


def plot(df, highlight=None, path="results_tables/fig_validity_heatmap"):
    """Rules x (n, f) grid: valid / valid but collapses to mean or median /
    invalid, with ALIE's z under each column. Glyphs repeat the colour."""
    cols = df[["n", "f"]].drop_duplicates().sort_values(["n", "f"]).values.tolist()
    rules = list(AGGREGATORS)
    fig, ax = plt.subplots(figsize=(0.42 * len(cols) + 1.6, 3.9))
    for j, (n, f) in enumerate(cols):
        for i, rule in enumerate(rules):
            r = df[(df.n == n) & (df.f == f) & (df.rule == rule)].iloc[0]
            st = state(r)
            color = {"valid": VALID, "collapsed": COLLAPSED, "invalid": INVALID}[st]
            ax.add_patch(plt.Rectangle((j + 0.04, i + 0.04), 0.92, 0.92, color=color, lw=0))
            glyph = {"valid": "\u2713", "invalid": "\u00d7",
                     "collapsed": "=" + ("μ" if r.degenerate == "= mean" else "med")}[st]
            ax.text(j + 0.5, i + 0.52, glyph, ha="center", va="center",
                    fontsize=7 if st == "collapsed" else 9,
                    color="white" if st == "valid" else (INK if st == "collapsed" else MUTED))
        z = df[(df.n == n) & (df.f == f)].alie_z.iloc[0]
        ztxt = "" if f == 0 else ("undef." if z != z or abs(z) == float("inf") else f"{z:.2f}")
        ax.text(j + 0.5, len(rules) + 0.55, ztxt, ha="center", va="center", fontsize=7,
                color=INK if (f > 0 and z <= 0) else MUTED,
                fontweight="bold" if (f > 0 and z <= 0) else "normal")
    # n groups: labels on top, thin separators between groups
    ns = sorted({n for n, _ in cols})
    for n in ns:
        js = [j for j, (m, _) in enumerate(cols) if m == n]
        ax.text((js[0] + js[-1] + 1) / 2, -0.55, f"n = {n}", ha="center", va="center",
                fontsize=8, color=INK, fontweight="bold" if n == highlight else "normal")
        if n == highlight:
            ax.add_patch(plt.Rectangle((js[0], 0), len(js), len(rules), fill=False,
                                       ec=INK, lw=1.2))
        if js[0] > 0:
            ax.axvline(js[0], color=SURFACE, lw=3)
    ax.set_xlim(0, len(cols))
    ax.set_ylim(len(rules) + 1, -1)
    ax.set_xticks([j + 0.5 for j in range(len(cols))], [f"f={f}" for _, f in cols], fontsize=7)
    ax.xaxis.tick_bottom()
    ax.set_yticks([i + 0.5 for i in range(len(rules))] + [len(rules) + 0.55],
                  [RULE_LABELS[r] for r in rules] + ["ALIE z"], fontsize=8)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=l) for c, l in (
        (VALID, "\u2713 valid"), (COLLAPSED, "= valid, but equals the mean (μ) or median"),
        (INVALID, "\u00d7 invalid (precondition on n, f fails)"))]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3,
              fontsize=7, handlelength=1, columnspacing=1.2)
    save(fig, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ns", type=csv(int), default=[3, 4, 5, 6, 20])
    p.add_argument("--fs", type=csv(int), default=[0, 1, 2])
    p.add_argument("--out", default="results_tables")
    p.add_argument("--highlight", type=int, default=4, help="outline this n (heart: 4)")
    p.add_argument("--weights", action="store_true",
                   help="weight-based check on the FLamby datasets' real sizes")
    a = p.parse_args()
    if a.weights:
        w = build_weighted(flamby_sizes())
        Path(a.out).mkdir(parents=True, exist_ok=True)
        w.to_csv(Path(a.out) / "validity_weighted.csv", index=False)
        flagged = w[w.dictator | w.filter(like="breakdown").eq(True).any(axis=1)]
        print(flagged.to_string(index=False))
        print(f"\nsaved {Path(a.out) / 'validity_weighted.csv'} ({len(flagged)} flagged rows)")
        return
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
    plot(df, a.highlight, out / "fig_validity_heatmap")
    print(f"\nsaved {out / 'validity.csv'} and {out / 'fig_validity_heatmap'}.{{png,pdf}}")


if __name__ == "__main__":
    main()
