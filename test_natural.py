"""Checks for natural-split mechanics: fixed attacker identities, weighting,
attacker enumeration and presets (synthetic data; no FLamby download needed)."""
import subprocess
import sys
import tempfile

from bench.design import attacker_sets
from bench.fl import Config, make_config, run


def test_attacker_sets():
    assert attacker_sets(4, 0) == [None]
    assert attacker_sets(4, 1) == ["0", "1", "2", "3"]           # every hospital
    assert len(attacker_sets(4, 2)) == 6 and len(attacker_sets(6, 2)) == 15
    s = attacker_sets(8, 2)                                       # 28 pairs > 15: sample
    assert len(s) == len(set(s)) == 15 and s == attacker_sets(8, 2)


def test_fixed_attackers_are_used_and_logged():
    res = run(Config(dataset="synthetic", n_clients=5, byz_ids="3,1", attack="sign_flip",
                     rounds=1, device="cpu"), log=lambda *_: None)
    assert res["status"] == "ok" and res["byz_ids"] == [1, 3] and res["f"] == 2
    assert len(res["client_sizes"]) == 5


def test_weighted_run():
    kw = dict(dataset="synthetic", n_clients=5, rounds=2, device="cpu")
    a = run(Config(**kw), log=lambda *_: None)
    b = run(Config(weighted=True, **kw), log=lambda *_: None)
    assert b["status"] == "ok"
    assert a["history"] != b["history"]   # unequal client sizes -> different model


def test_heart_preset():
    c = make_config(dataset="fed_heart_disease", rounds=3)
    assert (c.n_clients, c.lr, c.batch_size, c.local_steps, c.rounds) == (4, 0.001, 4, 100, 3)
    assert c.device == "cpu" and make_config(dataset="fed_heart_disease", device="mps").device == "mps"


def test_natural_sweep_grid():
    # 1 rule x 2 attacks: none -> f in {0,1,2} = 3 runs; alie -> f=1: 4 + f=2: 6
    out = subprocess.run([sys.executable, "sweep.py", "--datasets", "fed_heart_disease",
                          "--aggregators", "median", "--attacks", "none,alie",
                          "--fs", "0,1,2", "--dry_run", "--out", tempfile.mkdtemp()],
                         capture_output=True, text=True)
    names = [l for l in out.stdout.splitlines() if l.startswith("[")]
    assert len(names) == 13, out.stdout + out.stderr
    assert any("_alie_byz2_" in l for l in names) and any("_alie_byz1-3_" in l for l in names)


def test_first_heart_run_in_a_process_is_seeded():
    # Regression: FLamby's import-time torch.manual_seed(42) used to fire inside
    # the first run of each process, after set_seed, replacing its initial model.
    # A fresh interpreter must give the same result on its first and second run.
    from pathlib import Path
    if not Path("data/flamby/fed_heart_disease/processed.va.data").exists():
        print("skipped: heart data not downloaded")
        return
    code = ("from bench.fl import make_config, run\n"
            "c = make_config(dataset='fed_heart_disease', rounds=2, seed=0)\n"
            "a, b = (run(c, log=lambda *_: None)['history'] for _ in range(2))\n"
            "assert a == b, 'first run differs from second'\n"
            "print('ok')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "ok", out.stdout + out.stderr


if __name__ == "__main__":
    test_attacker_sets()
    test_fixed_attackers_are_used_and_logged()
    test_weighted_run()
    test_heart_preset()
    test_natural_sweep_grid()
    test_first_heart_run_in_a_process_is_seeded()
    print("all tests passed")
