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
    slightly = {**flat, 1.0: [x + 0.002 for x in flat[0.3]]}
    assert decide({"u": slightly}, 0.3, margin=0.0)[0] == 1.0   # v2: significant
    assert decide({"u": slightly}, 0.3)[0] == 0.3        # v2.1: below 0.5-pt margin
    # better in one variant only -> keep tau0 (must hold in every variant)
    assert decide({"u": better, "w": flat}, 0.3)[0] == 0.3


def test_run_names_tag_tuning_runs():
    assert "_val0.1" in run_name(Config(val_frac=0.1))
    assert "_val" not in run_name(Config())


def test_round_count_in_names_and_legacy_files():
    import json
    import tempfile
    from pathlib import Path

    from bench.fl import make_config
    from run import result_path
    a = make_config(dataset="fed_heart_disease", rounds=106)
    b = make_config(dataset="fed_heart_disease", rounds=15)
    assert "_r106_" in run_name(a) and run_name(a) != run_name(b)
    out = Path(tempfile.mkdtemp())
    legacy = out / f"{run_name(a, rounds=False, lr=False)}.json"  # old format, no round count
    legacy.write_text(json.dumps({"config": {"rounds": 106, "lr": 0.001}}))
    assert result_path(out, b) == out / f"{run_name(b)}.json"
    assert legacy.exists() and not result_path(out, b).exists()  # 15 rounds: never reused
    p = result_path(out, a)                                 # 106 rounds: adopted
    assert p.exists() and not legacy.exists() and p.name == f"{run_name(a)}.json"


def test_lr_in_names_and_legacy_files():
    import json
    import tempfile
    from pathlib import Path

    from bench.fl import make_config
    from run import result_path
    a = make_config(dataset="fed_tcga_brca")             # preset lr 0.003, 243 rounds
    b = make_config(dataset="fed_tcga_brca", lr=0.1)
    assert "_lr0.003_r243_" in run_name(a) and run_name(a) != run_name(b)
    assert "_lr0.3_" in run_name(make_config(lr=0.1 + 0.2))  # no float noise
    out = Path(tempfile.mkdtemp())
    # Format without lr (2026-10-03 to 2026-10-13), stored lr 0.003: never
    # reused for lr 0.1, adopted for lr 0.003
    no_lr = out / f"{run_name(a, lr=False)}.json"
    assert no_lr.name == f"{run_name(b, lr=False)}.json"      # the old collision
    no_lr.write_text(json.dumps({"config": {"rounds": 243, "lr": 0.003}}))
    assert not result_path(out, b).exists() and no_lr.exists()
    p = result_path(out, a)
    assert p.exists() and not no_lr.exists() and p.name == f"{run_name(a)}.json"
    # Oldest format (no lr, no rounds): needs both to match
    oldest = out / f"{run_name(b, rounds=False, lr=False)}.json"
    oldest.write_text(json.dumps({"config": {"rounds": 243, "lr": 0.003}}))
    assert not result_path(out, b).exists() and oldest.exists()   # lr differs
    c = make_config(dataset="fed_tcga_brca", rounds=5)
    assert not result_path(out, c).exists() and oldest.exists()   # rounds differ
    oldest.write_text(json.dumps({"config": {"rounds": 243, "lr": 0.1}}))
    assert result_path(out, b).exists() and not oldest.exists()   # both match


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
    test_round_count_in_names_and_legacy_files()
    test_lr_in_names_and_legacy_files()
    test_val_mode_end_to_end()
    print("all tests passed")
