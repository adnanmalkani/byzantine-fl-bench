"""Checks for the validation split, tau selection and run naming."""
import numpy as np

from bench.data import train_val_split
from bench.fl import Config, run
from run import run_name
from tune_cc import decide, grid_around, snap


def test_train_val_split():
    labels = np.repeat(np.arange(10), [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    tr, va = train_val_split(labels, 0.1, seed=0)
    assert len(np.intersect1d(tr, va)) == 0                      # disjoint
    assert len(tr) + len(va) == len(labels)                      # covers everything
    counts = np.bincount(labels[va], minlength=10)
    assert np.array_equal(counts, np.bincount(labels, minlength=10) // 10)  # stratified
    tr2, va2 = train_val_split(labels, 0.1, seed=0)
    assert np.array_equal(va, va2)                               # deterministic
    assert not np.array_equal(va, train_val_split(labels, 0.1, seed=1)[1])


def test_tau_rule_v2_ladder():
    assert snap(0.026) == 0.03 and snap(0.45) == 0.3 and snap(0.6) == 1.0
    assert snap(1.9) == 3.0 and snap(0.0012) == 0.001
    assert grid_around(0.03) == [0.003, 0.01, 0.03, 0.1, 0.3]
    assert grid_around(1.0) == [0.1, 0.3, 1.0, 3.0, 10.0]


def test_tau_rule_v2_decision():
    seeds = np.arange(10)
    flat = {t: list(0.80 + 0.001 * np.sin(seeds + t)) for t in (0.1, 0.3, 1.0)}
    assert decide({"u": flat}, 0.3)[0] == 0.3            # nothing better: keep tau0
    better = {**flat, 1.0: [x + 0.02 for x in flat[0.3]]}
    assert decide({"u": better}, 0.3)[0] == 1.0          # clearly better: switch
    # better in one variant only -> keep tau0 (must hold in every variant)
    assert decide({"u": better, "w": flat}, 0.3)[0] == 0.3


def test_run_names_tag_tuning_runs():
    assert "_val0.1" in run_name(Config(val_frac=0.1))
    assert "_val" not in run_name(Config())


def test_val_mode_end_to_end():
    res = run(Config(dataset="synthetic", rounds=1, val_frac=0.1, device="cpu"),
              log=lambda *_: None)
    assert res["status"] == "ok" and res["eval_split"] == "val"
    res = run(Config(dataset="synthetic", rounds=1, device="cpu"), log=lambda *_: None)
    assert res["eval_split"] == "test"


if __name__ == "__main__":
    test_train_val_split()
    test_tau_rule_v2_ladder()
    test_tau_rule_v2_decision()
    test_run_names_tag_tuning_runs()
    test_val_mode_end_to_end()
    print("all tests passed")
