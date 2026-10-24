"""Choose the learning rate and number of rounds for Fed-ISIC2019's main
setting (frozen-backbone features, linear layer, SGD), by the rule
pre-specified in docs/PROTOCOL.md ("Learning-rate screen", 2026-10-16).

Clean sample-weighted FedAvg, FLamby's batch 64 and 100 local steps, seeds
0-19, 200 rounds, evaluated every round on the POOLED VALIDATION set (10% of
each centre's training images, stratified by class; 1,859 images; the test
set is never used). B(r) = pooled validation balanced accuracy after round r
(chance = 0.125).

Per seed:
  divergence     non-finite validation loss or balanced accuracy (non-finite
                 weights or outputs make the loss non-finite)
  learned_lost   B(r) >= 0.30 at some round, then B(r') < 0.20 later
                 (absolute rule, as TCGA's)
  not_learned    B(r) never reaches 0.30 (reported, not a collapse)
Choice: the largest lr with zero collapsed seeds (learned_lost or
divergence); stop if it has any not-learned seed or no lr qualifies. Rounds:
heart plateau rule on the chosen lr's 20 curves (A_end = mean of A over
rounds 151-200, tolerance 0.005). If R* > 150, the chosen lr is re-run to
400 rounds (the first 200 must match), classified again over all 400 rounds
(stop on any collapse or not-learned seed), and R* is taken over rounds
301-400; if that R* > 300 too, stop.

Each run is saved to results_tune/lr_isic/, so the script resumes after a
stop. Runs in parallel, one torch thread per process.

Usage: python select_lr_isic.py [--workers 8]
  (about 6 min with 8 processes; about 2 min more if the 400-round re-run
  is needed)
Writes results_tables/lr_isic.json
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from bench.fl import make_config, run
from run import save_json
from select_lr_tcga import plateau_round

LRS = [0.3, 0.1, 0.03, 0.01, 0.003]
SEEDS, ROUNDS, LONG, TOL = list(range(20)), 200, 400, 0.005
LEARNED, FLOOR = 0.30, 0.20
RUNS, OUT = Path("results_tune/lr_isic"), Path("results_tables")


def one_run(lr, seed, rounds=ROUNDS, folder=RUNS):
    path = folder / f"isic_fedavg_w_lr{lr}_val0.1_r{rounds}_s{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    torch.set_num_threads(1)
    cfg = make_config(dataset="fed_isic2019", aggregator="fedavg", weighted=True,
                      lr=lr, rounds=rounds, val_frac=0.1, seed=seed)
    r = run(cfg, log=lambda *_: None)
    assert r["eval_split"] == "val" and r["status"] == "ok"
    out = {"lr": lr, "seed": seed, "rounds": rounds, "error": None,
           "ba": [h["acc"] for h in r["history"]],
           "loss": [h["loss"] for h in r["history"]], "seconds": r["seconds"]}
    save_json(path, out)
    return out


def classify(rec):
    """'divergence', 'learned_lost', 'not_learned' or 'ok'."""
    if rec["error"] is not None:
        return "divergence"
    b, loss = np.asarray(rec["ba"], float), np.asarray(rec["loss"], float)
    if not (np.isfinite(b).all() and np.isfinite(loss).all()):
        return "divergence"
    best_before = np.maximum.accumulate(b)[:-1]   # max of B over rounds < r'
    if ((best_before >= LEARNED) & (b[1:] < FLOOR)).any():
        return "learned_lost"
    if b.max() < LEARNED:
        return "not_learned"
    return "ok"


def runs(pool, jobs):
    """jobs: list of (lr, seed, rounds); results in the same order."""
    return list(pool.map(one_run, *zip(*jobs)))


def summary(recs):
    cls = [classify(r) for r in recs]
    counts = {k: cls.count(k) for k in ("ok", "not_learned", "learned_lost", "divergence")}
    return {"counts": counts, "collapsed": counts["learned_lost"] + counts["divergence"],
            "per_seed": dict(zip(map(str, SEEDS), cls)),
            "best_ba_mean": float(np.mean([max(r["ba"]) for r in recs if r["error"] is None]))}


def long_rerun(pool, lr):
    recs = runs(pool, [(lr, s, LONG) for s in SEEDS])
    short = runs(pool, [(lr, s, ROUNDS) for s in SEEDS])
    max_diff = max(float(np.abs(np.asarray(a["ba"][:ROUNDS]) - b["ba"]).max())
                   for a, b in zip(recs, short))
    out = {"rounds": LONG, **summary(recs), "max_diff_first_200": max_diff}
    s = out["counts"]
    if out["collapsed"] or s["not_learned"]:
        out["stop"] = "collapsed or not-learned seeds over 400 rounds"
        return out
    A = np.array([r["ba"] for r in recs]).mean(0)
    R, a_end = plateau_round(A, 301, LONG)
    out.update(window=[301, LONG], R_star=R, A_end=a_end, A_at_R=float(A[R - 1]),
               A_curve=A.tolist())
    if R > 300:
        out["stop"] = "no plateau within 400 rounds"
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    report = {"rule": "docs/PROTOCOL.md, Fed-ISIC2019 learning-rate screen (2026-10-16)", "lrs": {}}
    with ProcessPoolExecutor(a.workers) as pool:
        recs = runs(pool, [(lr, s, ROUNDS) for lr in LRS for s in SEEDS])
        by_lr = {lr: recs[i * len(SEEDS):(i + 1) * len(SEEDS)] for i, lr in enumerate(LRS)}
        for lr in LRS:
            report["lrs"][str(lr)] = summary(by_lr[lr])
            print(f"lr {lr}: {report['lrs'][str(lr)]['counts']}", flush=True)

        ok = [lr for lr in LRS if report["lrs"][str(lr)]["collapsed"] == 0]
        chosen = max(ok) if ok else None
        report["chosen_lr"] = chosen
        if chosen is None:
            report["stop"] = "no learning rate has zero collapsed seeds"
        elif report["lrs"][str(chosen)]["counts"]["not_learned"]:
            report["stop"] = "chosen learning rate has not-learned seeds"
        else:
            A = np.array([r["ba"] for r in by_lr[chosen]]).mean(0)
            R, a_end = plateau_round(A, 151, ROUNDS)
            report.update(window=[151, ROUNDS], R_star=R, A_end=a_end, A_at_R=float(A[R - 1]),
                          A_curve=A.tolist())
            if R > 150:
                report["rerun_400"] = long_rerun(pool, chosen)
                if "stop" in report["rerun_400"]:
                    report["stop"] = report["rerun_400"]["stop"]
                else:
                    report["R_star_final"] = report["rerun_400"]["R_star"]
            else:
                report["R_star_final"] = R
    OUT.mkdir(exist_ok=True)
    save_json(OUT / "lr_isic.json", report, indent=2)
    show = {k: v for k, v in report.items() if k not in ("lrs", "A_curve")}
    if "rerun_400" in show:
        show["rerun_400"] = {k: v for k, v in show["rerun_400"].items()
                             if k not in ("A_curve", "per_seed")}
    print(json.dumps(show, indent=2))
    print(f"saved {OUT / 'lr_isic.json'}")


if __name__ == "__main__":
    main()
