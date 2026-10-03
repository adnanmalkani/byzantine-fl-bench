"""Choose the number of rounds for Fed-Heart-Disease from a convergence curve.

Sample-weighted FedAvg with FLamby's local settings (tasks.PRESETS), 200
rounds, 20 seeds, evaluated every round on the POOLED VALIDATION set (10%
stratified holdout per hospital; the test set is never used).

Plateau rule (fixed before looking at the curve):
  A(r)   = mean pooled validation accuracy over seeds after round r
  A_end  = mean of A(r) over rounds 151-200
  R*     = smallest r with A(r') >= A_end - 0.5 points for every r' >= r

Usage: python convergence_heart.py
Writes results_tables/convergence_heart.{json,png,pdf}
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

from bench.fl import make_config, run

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROUNDS, SEEDS, TOL, FLAMBY_ROUNDS = 200, list(range(20)), 0.005, 15
OUT = Path("results_tables")
INK, MUTED, GRID, SERIES, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#fcfcfb"


def plateau_round(A, tol=TOL):
    a_end = A[150:200].mean()
    ok = A >= a_end - tol
    # smallest r such that every r' >= r is ok (r is 1-based)
    bad = np.where(~ok)[0]
    return (bad[-1] + 2 if len(bad) else 1), a_end


def mean_ci(X):
    m = X.mean(0)
    h = stats.t.ppf(0.975, len(X) - 1) * X.std(0, ddof=1) / np.sqrt(len(X))
    return m, m - h, m + h


def main():
    acc, loss = [], []
    for s in SEEDS:
        r = run(make_config(dataset="fed_heart_disease", aggregator="fedavg", weighted=True,
                            rounds=ROUNDS, val_frac=0.1, seed=s), log=lambda *_: None)
        assert r["eval_split"] == "val"
        acc.append([h["acc"] for h in r["history"]])
        loss.append([h["loss"] for h in r["history"]])
    acc, loss = np.array(acc), np.array(loss)
    A = acc.mean(0)
    R, a_end = plateau_round(A)
    OUT.mkdir(exist_ok=True)
    (OUT / "convergence_heart.json").write_text(json.dumps(
        {"rule": "smallest r with A(r') >= A_end - 0.5pt for all r' >= r; A_end = mean A(151..200)",
         "R_star": int(R), "A_end": a_end, "A_at_R": A[R - 1], "A_at_15": A[FLAMBY_ROUNDS - 1],
         "seeds": SEEDS, "acc": acc.tolist(), "loss": loss.tolist()}, indent=1))

    rounds = np.arange(1, ROUNDS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=SURFACE)
    panels = [(axes[0], acc * 100, "Pooled validation accuracy (%)"),
              (axes[1], loss, "Pooled validation loss (BCE)")]
    for ax, X, title in panels:
        m, lo, hi = mean_ci(X)
        ax.set_facecolor(SURFACE)
        ax.fill_between(rounds, lo, hi, color=SERIES, alpha=0.12, linewidth=0)
        ax.plot(rounds, m, color=SERIES, linewidth=2, solid_capstyle="round")
        for x, label in [(FLAMBY_ROUNDS, f"FLamby: {FLAMBY_ROUNDS}"), (R, f"R* = {R}")]:
            ax.axvline(x, color=MUTED, linewidth=1)
            ax.text(x + 3, 0.04, label, transform=ax.get_xaxis_transform(),
                    color=MUTED, fontsize=9)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_xlabel("Round", color=MUTED)
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_xlim(0, ROUNDS)
    axes[0].axhline((a_end - TOL) * 100, color=MUTED, linewidth=1, linestyle=(0, (1, 2)))
    fig.text(0.01, -0.04, f"Sample-weighted FedAvg, FLamby local settings; mean of "
             f"{len(SEEDS)} seeds, band = 95% CI. Dotted: plateau threshold "
             f"(mean of rounds 151-200 minus 0.5 points).", color=MUTED, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"convergence_heart.{ext}", dpi=200, bbox_inches="tight",
                    facecolor=SURFACE)
    print(f"R* = {R}  (A_end = {a_end:.4f}; A(R*) = {A[R - 1]:.4f}; A(15) = "
          f"{A[FLAMBY_ROUNDS - 1]:.4f}; A(200) = {A[-1]:.4f})")
    print(f"loss: L(15) = {loss.mean(0)[14]:.4f}, L(R*) = {loss.mean(0)[R - 1]:.4f}, "
          f"L(200) = {loss.mean(0)[-1]:.4f}")
    print(f"saved {OUT / 'convergence_heart.png'}")


if __name__ == "__main__":
    main()
