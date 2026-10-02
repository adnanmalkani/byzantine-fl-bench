"""Run a grid of experiments. Skips runs whose result file already exists,
so you can stop and resume at any time.

Example:
    python sweep.py --datasets mnist --alphas 0.1,0.5 \
        --aggregators fedavg,median,krum --attacks none,alie --byz_fracs 0.2 --seeds 0,1,2
"""
import argparse
import itertools
import json
from pathlib import Path

from bench.fl import Config, run
from run import run_name


def csv(t):
    return lambda s: [t(x) for x in s.split(",")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", type=csv(str), default=["mnist"])
    p.add_argument("--alphas", type=csv(float), default=[0.5])
    p.add_argument("--aggregators", type=csv(str),
                   default=["fedavg", "median", "trimmed_mean", "krum",
                            "multi_krum", "bulyan", "geomed", "cc"])
    p.add_argument("--attacks", type=csv(str),
                   default=["none", "label_flip", "sign_flip", "gaussian", "alie", "ipm"])
    p.add_argument("--byz_fracs", type=csv(float), default=[0.2])
    p.add_argument("--seeds", type=csv(int), default=[0])
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="results")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = list(itertools.product(a.datasets, a.alphas, a.aggregators,
                                  a.attacks, a.byz_fracs, a.seeds))
    for k, (ds, al, ag, at, bf, sd) in enumerate(grid, 1):
        if at == "none" and bf > 0:
            bf_run = 0.0  # clean baseline: no Byzantine clients
        else:
            bf_run = bf
        cfg = Config(dataset=ds, alpha=al, aggregator=ag, attack=at,
                     byz_frac=bf_run, seed=sd, rounds=a.rounds, device=a.device)
        path = out / f"{run_name(cfg)}.json"
        if path.exists():
            continue
        print(f"[{k}/{len(grid)}] {path.stem}")
        res = run(cfg, log=lambda *_: None)
        path.write_text(json.dumps(res, indent=2))
        print("   ", res["status"], res.get("final_acc", res.get("reason")))


if __name__ == "__main__":
    main()
