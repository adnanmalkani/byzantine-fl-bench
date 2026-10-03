"""Select Centered Clipping's radius tau for one dataset, on a validation split.

Procedure (see README, "Tuning CC's tau"):
  clean runs (no attack, server assumes f), alpha 0.5, each tau in the grid;
  clients train on 90% of the training data and accuracy is measured on the
  held-out 10%. The test set is never used. Validation accuracy is averaged
  over --seeds (one seed for MNIST; 20 for small validation sets such as
  heart). We pick the SMALLEST tau whose validation accuracy is within --tol
  of the best one: a smaller radius limits how far any single update can
  move the model. The grid must span the dataset's update scale (README).

Example:
    python tune_cc.py --dataset mnist            # runs missing taus, then selects
    python tune_cc.py --dataset mnist --shard 0/2  # split runs across processes
Writes results_tune/cc_tau_<dataset>.json once every tau has a result.
"""
import argparse
import json
from pathlib import Path

from bench.fl import make_config, run
from run import _bool, run_name
from sweep import csv


def select_tau(accs: dict, tol: float) -> float:
    """accs: {tau: val_acc}. Smallest tau within tol of the best accuracy."""
    best = max(accs.values())
    return min(t for t, a in accs.items() if a >= best - tol)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--taus", type=csv(float), default=[0.1, 0.3, 1.0, 3.0, 10.0])
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--f_assumed", type=int, default=4)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--tol", type=float, default=0.003)
    p.add_argument("--seeds", type=csv(int), default=[0])
    p.add_argument("--weighted", type=_bool, default=False)
    p.add_argument("--rounds", type=int, default=None, help="default: dataset preset")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--out", default="results_tune")
    a = p.parse_args()
    shard_i, shard_n = map(int, a.shard.split("/"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    accs, rounds = {}, None
    jobs = [(tau, sd) for tau in a.taus for sd in a.seeds]
    for k, (tau, sd) in enumerate(jobs):
        kw = dict(dataset=a.dataset, aggregator="cc", attack="none", alpha=a.alpha,
                  f_assumed=a.f_assumed, cc_tau=tau, val_frac=a.val_frac, seed=sd,
                  weighted=a.weighted)
        cfg = make_config(**kw, **({} if a.rounds is None else {"rounds": a.rounds}))
        rounds = cfg.rounds
        path = out / f"{run_name(cfg)}.json"
        if not path.exists() and k % shard_n == shard_i:
            print(f"running tau={tau} seed={sd}")
            path.write_text(json.dumps(run(cfg, log=lambda *_: None), indent=2))
        if path.exists():
            res = json.loads(path.read_text())
            assert res["eval_split"] == "val" and res["status"] == "ok", path
            accs.setdefault(tau, []).append(res["final_acc"])
    complete = {t: v for t, v in accs.items() if len(v) == len(a.seeds)}
    for tau, v in complete.items():
        print(f"tau {tau:6g}  val acc {sum(v) / len(v):.4f}  (mean of {len(v)} seeds)")

    if len(complete) < len(a.taus):
        print("some runs not finished yet (other shards?); rerun to select")
        return
    mean_acc = {t: sum(v) / len(v) for t, v in complete.items()}
    tau = select_tau(mean_acc, a.tol)
    at_edge = tau == min(a.taus)
    sel = {"dataset": a.dataset, "weighted": a.weighted, "cc_tau": tau,
           "selected_at_grid_edge": at_edge, "val_acc": mean_acc, "tol": a.tol,
           "alpha": a.alpha, "f_assumed": a.f_assumed, "val_frac": a.val_frac,
           "seeds": a.seeds, "rounds": rounds}
    name = f"cc_tau_{a.dataset}{'_w' if a.weighted else ''}.json"
    (out / name).write_text(json.dumps(sel, indent=2))
    print(f"selected tau = {tau}{'  (WARNING: smallest value in the grid)' if at_edge else ''}"
          f" -> {out / name}")


if __name__ == "__main__":
    main()
