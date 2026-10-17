"""Run a grid of experiments. Skips runs whose result file already exists,
so you can stop and resume at any time.

Example:
    python sweep.py --datasets mnist --alphas 0.1,0.5 \
        --aggregators fedavg,median,krum --attacks none,alie --byz_fracs 0.2 --seeds 0,1,2

Natural splits (FLamby): use --fs instead of --byz_fracs. For each f, every
attacker set is run (each hospital for f=1, all pairs if <= 15, else 15
sampled), and clean runs assume f:
    python sweep.py --datasets fed_heart_disease --fs 0,1,2 --weighted 0,1 --seeds 0,1,2,3,4
"""
import argparse
import itertools
import json
from pathlib import Path

from bench.design import attacker_sets
from bench.fl import make_config, run
from run import _bool, result_path, save_json


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
    p.add_argument("--fs", type=csv(int), default=None,
                   help="natural splits: Byzantine counts; enumerates attacker identities")
    p.add_argument("--weighted", type=csv(_bool), default=[False])
    p.add_argument("--seeds", type=csv(int), default=[0])
    # rounds / cc_tau / local_steps: None = dataset preset, else Config default
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--cc_tau", type=float, default=None)
    p.add_argument("--local_steps", type=int, default=None)
    p.add_argument("--val_frac", type=float, default=0.0,
                   help="> 0: tuning mode, evaluate on held-out training data")
    p.add_argument("--drop_leaked", action="store_true",
                   help="fed_isic2019 duplicates check: drop the 17 leaked test images")
    p.add_argument("--shard", default="0/1",
                   help="i/N: run only every N-th grid point starting at i, so N "
                        "processes can split one grid without overlap")
    p.add_argument("--dry_run", action="store_true",
                   help="list the runs (and how many are left) without running them")
    p.add_argument("--device", default=None, help="default: dataset preset, else auto")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    shard_i, shard_n = map(int, a.shard.split("/"))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = []
    for ds, al, ag, at in itertools.product(a.datasets, a.alphas, a.aggregators, a.attacks):
        n = make_config(dataset=ds).n_clients
        if a.fs is None:  # synthetic splits: Byzantine fraction, random attackers
            specs = [dict(byz_frac=0.0, f_assumed=int(round(bf * n)))
                     # Clean baseline: no Byzantine clients, but the server still
                     # defends against the f it would expect (cost of robustness)
                     if at == "none" and bf > 0 else dict(byz_frac=bf)
                     for bf in a.byz_fracs]
        else:  # natural splits: fixed f, every attacker identity
            specs = []
            for f in a.fs:
                if at == "none":
                    specs.append(dict(f_assumed=f))
                elif f > 0:
                    specs += [dict(byz_ids=ids) for ids in attacker_sets(n, f)]
        for spec, wt, sd in itertools.product(specs, a.weighted, a.seeds):
            grid.append(dict(dataset=ds, alpha=al, aggregator=ag, attack=at,
                             weighted=wt, seed=sd, **spec))

    opt = dict(cc_tau=a.cc_tau, local_steps=a.local_steps, rounds=a.rounds, device=a.device)
    opt = {k: v for k, v in opt.items() if v is not None}
    if a.drop_leaked:
        opt["drop_leaked"] = True
    for k, kw in enumerate(grid, 1):
        if (k - 1) % shard_n != shard_i:
            continue
        cfg = make_config(val_frac=a.val_frac, **opt, **kw)
        path = result_path(out, cfg)
        if path.exists():
            continue
        print(f"[{k}/{len(grid)}] {path.stem}")
        if a.dry_run:
            continue
        res = run(cfg, log=lambda *_: None)
        save_json(path, res, indent=2)
        print("   ", res["status"], res.get("final_acc", res.get("reason")))


if __name__ == "__main__":
    main()
