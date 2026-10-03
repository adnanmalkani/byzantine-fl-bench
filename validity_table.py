"""Validity table (RQ-A): for each rule, number of clients n and assumed f,
is the run valid, and if so does the rule degenerate?

Degeneracy is detected numerically on random updates, not from formulas:
  "= mean"    output equals the plain average (rule does nothing robust)
  "= median"  output equals the coordinate-wise median (e.g. trimmed mean
              when n - 2f <= 2: it is a different rule only in name)
  "1 client"  output is exactly one client's update (others are ignored)
Also reports ALIE's z per (n, f): z <= 0 means the ALIE attack sends the
mean of the benign updates (or moves towards it), i.e. it is not an attack.

Usage: python validity_table.py [--ns 3,4,5,6,20] [--fs 0,1,2] [--highlight 4]
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
    plot(df, a.highlight, out / "fig_validity_heatmap")
    print(f"\nsaved {out / 'validity.csv'} and {out / 'fig_validity_heatmap'}.{{png,pdf}}")


if __name__ == "__main__":
    main()
