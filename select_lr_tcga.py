"""Choose the learning rate and number of rounds for Fed-TCGA-BRCA's main
setting, by the rule pre-specified in the README at commit 90341e4 (now docs/PROTOCOL.md).

Clean sample-weighted FedAvg, FLamby's other settings (batch 8, 100 local
steps), seeds 0-19, 200 rounds, evaluated every round on the POOLED
VALIDATION set (10% holdout per region, stratified by event; the test set
is never used). C(r) = pooled validation C-index after round r.

Collapse (per seed):
  divergence       non-finite loss or C-index, or the run fails on NaN
                   risk scores (lifelines: "NaNs detected in inputs")
  learned_lost     C(r) >= 0.60 at some round, then C(r') < 0.50 later
Not learned: C(r) never reaches 0.60 (reported, not a collapse).
Choice: the largest lr with zero collapsed seeds; stop if it has any
not-learned seed or no lr qualifies. Rounds: heart plateau rule on the
chosen lr's curves (A_end = mean over rounds 151-200, tolerance 0.005); if
R* > 150 the chosen lr must be re-run for 400 rounds (not done here).

Amendment 1 (docs/PROTOCOL.md, 2026-10-12, made after seeing the screen): seeds
not-learned at every lr (13, 19) are hard validation splits, excluded from
the stop condition but kept in A(r). 0.003 is re-run to 400 rounds; if it
has zero collapses and every other seed learns, choose 0.003 with R* over
rounds 301-400, else choose 0.01 with R* from its 200-round curves.

Each run is saved to results_tune/lr_tcga/, so the script resumes.
Usage: python select_lr_tcga.py
Writes results_tables/lr_tcga.json
"""
import json
from pathlib import Path

import numpy as np

from bench.fl import make_config, run
from run import save_json

LRS = [0.1, 0.03, 0.01, 0.003]
SEEDS, ROUNDS, TOL = list(range(20)), 200, 0.005
LEARNED, CHANCE = 0.60, 0.50
RUNS, OUT = Path("results_tune/lr_tcga"), Path("results_tables")


def one_run(lr, seed, rounds=ROUNDS):
    path = RUNS / f"tcga_fedavg_w_lr{lr}_val0.1_r{rounds}_s{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    cfg = make_config(dataset="fed_tcga_brca", aggregator="fedavg", weighted=True,
                      lr=lr, rounds=rounds, val_frac=0.1, seed=seed)
    try:
        r = run(cfg, log=lambda *_: None)
        assert r["eval_split"] == "val"
        out = {"lr": lr, "seed": seed, "error": None,
               "c": [h["acc"] for h in r["history"]],
               "loss": [h["loss"] for h in r["history"]]}
    except ValueError as e:
        if "NaN" not in str(e):
            raise
        out = {"lr": lr, "seed": seed, "error": str(e), "c": None, "loss": None}
    save_json(path, out)
    return out


def classify(rec):
    """'divergence', 'learned_lost', 'not_learned' or 'ok'."""
    if rec["error"] is not None:
        return "divergence"
    c, loss = np.asarray(rec["c"], float), np.asarray(rec["loss"], float)
    if not (np.isfinite(c).all() and np.isfinite(loss).all()):
        return "divergence"
    best_before = np.maximum.accumulate(c)[:-1]   # max of C over rounds < r'
    if ((best_before >= LEARNED) & (c[1:] < CHANCE)).any():
        return "learned_lost"
    if c.max() < LEARNED:
        return "not_learned"
    return "ok"


def plateau_round(A, lo, hi, tol=TOL):
    """Heart rule: smallest 1-based r with A(r') >= A_end - tol for all r' >= r,
    A_end = mean of A over rounds lo..hi (1-based, inclusive)."""
    a_end = A[lo - 1:hi].mean()
    bad = np.where(A < a_end - tol)[0]
    return (int(bad[-1]) + 2 if len(bad) else 1), float(a_end)


def amendment_1(report, slow=0.003, fallback=0.01, long_rounds=400):
    hard = [s for s in SEEDS if all(report["lrs"][str(lr)]["per_seed"][str(s)] == "not_learned"
                                    for lr in LRS)]
    assert hard == [13, 19], hard                     # as named in the amendment
    recs = [one_run(slow, s, long_rounds) for s in SEEDS]
    # Determinism: the first 200 rounds must match the 200-round runs
    max_diff = max(float(np.nanmax(np.abs(np.asarray(r["c"][:ROUNDS]) - one_run(slow, s)["c"])))
                   for s, r in zip(SEEDS, recs) if r["error"] is None)
    cls = [classify(r) for r in recs]
    collapsed = [s for s, c in zip(SEEDS, cls) if c in ("learned_lost", "divergence")]
    not_learned = [s for s, c in zip(SEEDS, cls) if c == "not_learned" and s not in hard]
    out = {"hard_splits": hard, "slow_lr": slow, "slow_rounds": long_rounds,
           "per_seed": dict(zip(map(str, SEEDS), cls)), "collapsed": collapsed,
           "not_learned_excl_hard": not_learned, "max_diff_first_200": max_diff}
    if not collapsed and not not_learned:
        lr, A, window = slow, np.array([r["c"] for r in recs]).mean(0), (301, 400)
    else:
        lr, window = fallback, (151, ROUNDS)
        A = np.array([one_run(fallback, s)["c"] for s in SEEDS]).mean(0)
    R, a_end = plateau_round(A, *window)
    out.update(chosen_lr=lr, window=window, R_star=R, A_end=a_end, A_at_R=float(A[R - 1]),
               rerun_400=bool(lr == fallback and R > 150), A_curve=A.tolist())
    return out


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    report = {"rule": "README, commit 90341e4 (now docs/PROTOCOL.md)", "lrs": {}}
    for lr in LRS:
        recs = [one_run(lr, s) for s in SEEDS]
        cls = [classify(r) for r in recs]
        counts = {k: cls.count(k) for k in ("ok", "not_learned", "learned_lost", "divergence")}
        report["lrs"][str(lr)] = {"counts": counts,
                                  "collapsed": counts["learned_lost"] + counts["divergence"],
                                  "per_seed": dict(zip(map(str, SEEDS), cls))}
        print(f"lr {lr}: {counts}", flush=True)

    ok = [lr for lr in LRS if report["lrs"][str(lr)]["collapsed"] == 0]
    chosen = max(ok) if ok else None
    report["chosen_lr"] = chosen
    if chosen is None:
        report["stop"] = "no learning rate has zero collapsed seeds"
    elif report["lrs"][str(chosen)]["counts"]["not_learned"]:
        report["stop"] = "chosen learning rate has not-learned seeds"
    else:
        A = np.array([one_run(chosen, s)["c"] for s in SEEDS]).mean(0)
        R, a_end = plateau_round(A, 151, 200)
        report.update(R_star=R, A_end=a_end, A_at_R=float(A[R - 1]),
                      A_curve=A.tolist(), rerun_400=R > 150)
    if report.get("stop"):
        report["amendment_1"] = amendment_1(report)
    OUT.mkdir(exist_ok=True)
    (OUT / "lr_tcga.json").write_text(json.dumps(report, indent=2))
    show = {k: v for k, v in report.items() if k not in ("lrs", "A_curve")}
    show["amendment_1"] = {k: v for k, v in show.get("amendment_1", {}).items()
                           if k not in ("A_curve", "per_seed")}
    print(json.dumps(show, indent=2))
    print(f"saved {OUT / 'lr_tcga.json'}")


if __name__ == "__main__":
    main()
