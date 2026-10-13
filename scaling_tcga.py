"""How many TCGA runs can the Mac do at once? Throughput of N parallel
processes (1 torch thread each, as the grid would run), N = 1, 2, 4, 6, 8, 10.

Each worker runs one TCGA test-set run with the main preset (evaluation every
round, as in the grid) but only ROUNDS rounds, cycling through the rules.
Throughput = rounds finished per second over all workers; speed-up is relative
to N = 1. Process start-up (import, data loading) is excluded: each worker
times its own run() call.

Usage: python scaling_tcga.py   (~4 min; the Mac should be otherwise idle)
Writes results_tables/scaling_tcga.json
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROUNDS = 100
NS = [1, 2, 4, 6, 8, 10]
RULES = ["fedavg", "median", "trimmed_mean", "geomed", "cc", "krum", "multi_krum"]


def worker(rule, seed):
    from bench.fl import make_config, run
    cfg = make_config(dataset="fed_tcga_brca", aggregator=rule, rounds=ROUNDS, seed=seed)
    t0 = time.perf_counter()
    run(cfg, log=lambda *_: None)
    print(json.dumps({"rule": rule, "seconds": time.perf_counter() - t0}))


def main():
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    report = []
    for n in NS:
        procs = [subprocess.Popen([sys.executable, __file__, RULES[i % len(RULES)], str(i)],
                                  env=env, stdout=subprocess.PIPE, text=True)
                 for i in range(n)]
        secs = [json.loads(p.communicate()[0].strip().splitlines()[-1])["seconds"] for p in procs]
        assert all(p.returncode == 0 for p in procs)
        rps = n * ROUNDS / max(secs)   # slowest worker sets the pace
        report.append(dict(n=n, worker_seconds=[round(s, 1) for s in secs],
                           ms_per_round=round(1000 * max(secs) / ROUNDS, 1),
                           rounds_per_s=round(rps, 2)))
        print(f"N={n:2d}: slowest worker {max(secs):5.1f} s "
              f"({1000 * max(secs) / ROUNDS:.0f} ms/round), {rps:.2f} rounds/s", flush=True)
    base = report[0]["rounds_per_s"]
    for r in report:
        r["speedup"] = round(r["rounds_per_s"] / base, 2)
    print("speed-up vs 1 process:", {r["n"]: r["speedup"] for r in report})
    out = Path("results_tables")
    out.mkdir(exist_ok=True)
    (out / "scaling_tcga.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 3:
        worker(sys.argv[1], int(sys.argv[2]))
    else:
        main()
