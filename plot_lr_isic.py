"""Fed-ISIC2019 learning-rate screen: mean validation curves from the saved
runs (results_tune/lr_isic/, written by select_lr_isic.py). Trains nothing.

Left: mean pooled validation balanced accuracy with a 95% t band over the
20 seeds, for lr 0.3 (the rule's choice; 400 rounds) and lr 0.01 (FLamby's
FedAvg rate; 200 rounds, the only runs that exist). Middle: the mean curve
of every rate (200 rounds). Right: mean validation focal loss (log scale).

Also prints, per rate: the peak of the mean curve, the mean of each seed's
own best round, values at a few rounds, the round of minimum loss, and the
heart plateau round R* at tolerance 0.005 and 0.01.

Usage: python plot_lr_isic.py   (a few seconds)
Writes results_tables/fig_isic_lr_curves.{png,pdf}
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

from bench.plotstyle import INK, MUTED, plt, save
from select_lr_isic import LRS, RUNS, SEEDS
from select_lr_tcga import plateau_round

OUT = Path("results_tables")
COLORS = {0.3: "#e34948", 0.1: "#eda100", 0.03: "#1baf7a", 0.01: "#2a78d6", 0.003: "#4a3aa7"}


def curves(lr, rounds):
    recs = [json.loads((RUNS / f"isic_fedavg_w_lr{lr}_val0.1_r{rounds}_s{s}.json").read_text())
            for s in SEEDS]
    return np.array([r["ba"] for r in recs]), np.array([r["loss"] for r in recs])


def band(ax, B, color, label):
    r = np.arange(1, B.shape[1] + 1)
    m, se = B.mean(0), B.std(0, ddof=1) / np.sqrt(len(B))
    h = stats.t.ppf(0.975, len(B) - 1) * se
    ax.fill_between(r, m - h, m + h, color=color, alpha=0.2, lw=0)
    ax.plot(r, m, color=color, lw=1.2, label=label)
    pk = int(m.argmax())
    ax.plot(pk + 1, m[pk], "o", ms=3, color=color)


def report(lr, rounds, window):
    B, L = curves(lr, rounds)
    A = B.mean(0)
    pk = int(A.argmax())
    at = ", ".join(f"r{p} {A[p - 1]:.3f}" for p in (10, 50, 100, 200, 300, 400) if p <= rounds)
    R5, a_end = plateau_round(A, *window, tol=0.005)
    R10, _ = plateau_round(A, *window, tol=0.01)
    print(f"lr {lr} ({rounds} rounds): peak of mean curve {A[pk]:.3f} at round {pk + 1}; "
          f"mean of per-seed best {B.max(1).mean():.3f}; {at}; A_end {a_end:.3f}; "
          f"R* {R5} (tol 0.005), {R10} (tol 0.01); loss minimum at round {int(L.mean(0).argmin()) + 1}")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    ax = axes[0]
    band(ax, curves(0.3, 400)[0], COLORS[0.3], "lr 0.3 (rule's choice)")
    band(ax, curves(0.01, 200)[0], COLORS[0.01], "lr 0.01 (FLamby; 200 rounds run)")
    ax.set(xlabel="Round", ylabel="Validation balanced accuracy", ylim=(0.35, 0.6),
           title="Mean ± 95% CI over 20 seeds")
    ax.legend(loc="lower right", fontsize=8)
    for lr in LRS:
        B, L = curves(lr, 200)
        axes[1].plot(np.arange(1, 201), B.mean(0), color=COLORS[lr], lw=1.1, label=f"lr {lr}")
        axes[2].plot(np.arange(1, 201), L.mean(0), color=COLORS[lr], lw=1.1, label=f"lr {lr}")
    axes[1].set(xlabel="Round", ylabel="Validation balanced accuracy", ylim=(0.35, 0.6),
                title="All rates, mean of 20 seeds")
    axes[1].legend(loc="lower right", fontsize=8)
    axes[2].set(xlabel="Round", ylabel="Validation focal loss", yscale="log",
                title="Validation loss, mean of 20 seeds")
    for a in axes:
        a.title.set_color(INK)
        a.grid(color=MUTED, alpha=0.15, lw=0.5)
    fig.tight_layout()
    save(fig, OUT / "fig_isic_lr_curves")
    report(0.3, 400, (301, 400))
    for lr in LRS:
        report(lr, 200, (151, 200))
    print(f"saved {OUT / 'fig_isic_lr_curves'}.{{png,pdf}}")


if __name__ == "__main__":
    main()
