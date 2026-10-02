"""Checks for the validation split, tau selection and run naming."""
import numpy as np

from bench.data import train_val_split
from bench.fl import Config, run
from run import run_name
from tune_cc import select_tau


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


def test_select_tau():
    accs = {0.1: 0.980, 0.3: 0.987, 1.0: 0.989, 3.0: 0.989}
    assert select_tau(accs, tol=0.003) == 0.3   # 0.1 is 0.9 points below best
    assert select_tau(accs, tol=0.0) == 1.0     # ties -> smallest tau
    assert select_tau(accs, tol=0.01) == 0.1


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
    test_select_tau()
    test_run_names_tag_tuning_runs()
    test_val_mode_end_to_end()
    print("all tests passed")
