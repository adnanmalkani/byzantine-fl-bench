"""Run one experiment and save the result as JSON.

Example:
    python run.py --dataset mnist --aggregator krum --attack alie --byz_frac 0.2
"""
import argparse
import json
from dataclasses import fields
from pathlib import Path

from bench.fl import Config, make_config, run


def run_name(cfg: Config) -> str:
    # Optional parts are added only when set, so names of default runs don't change
    fa = "" if cfg.f_assumed is None else f"_fa{cfg.f_assumed}"
    tau = f"_tau{cfg.cc_tau}" if cfg.aggregator == "cc" else ""
    ls = "" if cfg.local_steps is None else f"_ls{cfg.local_steps}"
    val = f"_val{cfg.val_frac}" if cfg.val_frac > 0 else ""
    wt = "_w" if cfg.weighted else ""
    byz = f"_b{cfg.byz_frac}" if not cfg.byz_ids else f"_byz{cfg.byz_ids.replace(',', '-')}"
    return (f"{cfg.dataset}_a{cfg.alpha}_{cfg.aggregator}{wt}{tau}_{cfg.attack}"
            f"{byz}{fa}{ls}{val}_s{cfg.seed}")


def _bool(s: str) -> bool:
    return s.lower() in ("1", "true", "yes")


def main():
    p = argparse.ArgumentParser()
    for fld in fields(Config):
        typ = float if fld.type in (float, "float") else int if fld.type in (int, "int") else str
        if fld.type in (bool, "bool"):
            typ = _bool
        if fld.name in ("f_assumed", "local_steps"):  # Optional[int] fields
            typ = int
        # Only options given on the command line override the dataset preset
        p.add_argument(f"--{fld.name}", type=typ, default=argparse.SUPPRESS,
                       help=f"default: {fld.default}")
    p.add_argument("--out", default="results")
    args = vars(p.parse_args())
    out = Path(args.pop("out"))
    cfg = make_config(**args)

    res = run(cfg)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{run_name(cfg)}.json"
    path.write_text(json.dumps(res, indent=2))
    print(res["status"], res.get("final_acc", res.get("reason")), "->", path)


if __name__ == "__main__":
    main()
