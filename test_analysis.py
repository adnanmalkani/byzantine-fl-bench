"""Checks for the summary metrics and the heart analysis (influence, tables)."""
import math

import numpy as np
import pandas as pd

from analyze_heart import attacker_table, influence
from summarize import is_degenerate, run_metrics


def fake_run(agg="median", weighted=False, diag=(), sizes=(40, 30, 20, 10), f=1):
    return {"config": {"aggregator": agg, "weighted": weighted, "attack": "none"},
            "client_sizes": list(sizes), "f_assumed": f, "diag": list(diag)}


def test_degenerate_flag():
    r = {"config": {"attack": "alie"}, "alie_z": 0.0}
    assert is_degenerate(r)
    assert not is_degenerate({"config": {"attack": "alie"}, "alie_z": 0.674})
    assert not is_degenerate({"config": {"attack": "sign_flip"}})


def test_run_metrics_last3_and_honest_worst():
    hist = [{"acc": a, "client_acc": c} for a, c in [
        (0.1, [0.0, 0.0, 0.0]),                       # dropped: only the last 3 count
        (0.6, [0.9, 0.5, 0.2]), (0.7, [0.8, 0.6, 0.3]), (0.8, [0.7, 0.7, 0.4])]]
    m = run_metrics({"history": hist, "byz_ids": [2], "client_test_sizes": [10, 30, 60]})
    assert "final_acc" not in m                        # natural splits: avg_acc instead
    # patient-weighted: (10*0.8 + 30*0.6 + 60*0.3) / 100 over the last 3 evals
    assert math.isclose(m["avg_acc"], (8 + 18 + 18) / 100)
    assert math.isclose(m["worst_acc"], 0.3)          # hospital 2 (the attacker)
    assert math.isclose(m["worst_honest_acc"], 0.6)   # min over hospitals 0, 1 per eval
    assert math.isclose(m["gap"], (0.7 + 0.5 + 0.3) / 3)
    assert np.allclose(m["client_acc"], [0.8, 0.6, 0.3])
    img = run_metrics({"history": [{"acc": 0.5}]})                       # image datasets
    assert img == {"final_acc": 0.5}
    m2 = run_metrics({"history": hist, "byz_ids": []}, test_sizes=[1, 1, 1])
    assert math.isclose(m2["avg_acc"], (0.8 + 0.6 + 0.3) / 3)


def test_influence_per_rule():
    d = {"dist_prev": [1.0, 2.0, 3.0, 4.0]}
    rel, rate, down, dist = influence(fake_run("fedavg", diag=[{**d, "weights": [0.25] * 4}]))
    assert np.allclose(rel, 1) and np.isnan(rate).all() and (down == 0).all()
    rel, *_ = influence(fake_run("fedavg", True, [{**d, "weights": [0.4, 0.3, 0.2, 0.1]}]))
    assert np.allclose(rel, 1)                         # weighted: nominal = size share
    rel, rate, down, dist = influence(fake_run("median", diag=[{**d, "used_share": [0.5, 0.5, 0, 0]}]))
    assert np.allclose(rel, [2, 2, 0, 0]) and np.allclose(down, [0, 0, 1, 1])
    assert np.allclose(dist, [1, 2, 3, 4])
    # trimmed mean, n = 4, f = 1: kept share (1 - t) over n - 2f = 2 kept rows
    rel, rate, *_ = influence(fake_run("trimmed_mean", diag=[{**d, "trimmed_share": [0, 0.5, 0.5, 1]}]))
    assert np.allclose(rate, [0, 0.5, 0.5, 1]) and np.allclose(rel, [2, 1, 1, 0])
    rel, rate, *_ = influence(fake_run("cc", diag=[{**d, "scale": [1, 1, 0.5, 0.2],
                                                    "clipped": [False, False, True, True]}]))
    assert np.allclose(rel, [1, 1, 0.5, 0.2]) and np.allclose(rate, [0, 0, 1, 1])
    rel, rate, *_ = influence(fake_run("multi_krum", diag=[{**d, "selected": [0, 1, 3]}]))
    assert np.allclose(rel, [4 / 3, 4 / 3, 0, 4 / 3]) and np.allclose(rate, [0, 0, 1, 0])
    rel, rate, *_ = influence(fake_run("geomed", diag=[{**d, "weights": [0.4, 0.3, 0.2, 0.1]}]))
    assert np.allclose(rel, [1.6, 1.2, 0.8, 0.4]) and np.allclose(rate, [0, 0, 1, 1])
    # averaged over rounds
    rel, *_ = influence(fake_run("median", diag=[{**d, "used_share": [0.5, 0.5, 0, 0]},
                                                 {**d, "used_share": [0, 0, 0.5, 0.5]}]))
    assert np.allclose(rel, 1)


def test_attacker_table_pools_attacks_per_seed():
    rows = []
    for seed in (0, 1):
        for attack, acc in (("sign_flip", 0.5), ("ipm", 0.7), ("alie", 0.9)):
            rows.append(dict(aggregator="median", weighted=False, attack=attack, f=1,
                             f_assumed=1, seed=seed, attacker="2", status="ok",
                             degenerate=attack == "alie", avg_acc=acc + 0.1 * seed,
                             worst_acc=acc, worst_honest_acc=acc))
        rows.append(dict(aggregator="krum", weighted=False, attack="ipm", f=1, f_assumed=1,
                         seed=seed, attacker="2", status="invalid", degenerate=False))
    t = attacker_table(pd.DataFrame(rows))
    pooled = t[(t.aggregator == "median") & (t.attack == "mean of real attacks")].iloc[0]
    assert pooled.attacker == "Switzerland" and pooled.n_seeds == 2
    assert math.isclose(pooled.avg_acc_mean, 65.0)    # alie (degenerate) not pooled
    alie = t[(t.aggregator == "median") & (t.attack == "alie")].iloc[0]
    assert bool(alie.degenerate)
    assert (t[t.aggregator == "krum"].status == "invalid").all()


if __name__ == "__main__":
    test_degenerate_flag()
    test_run_metrics_last3_and_honest_worst()
    test_influence_per_rule()
    test_attacker_table_pools_attacks_per_seed()
    print("all tests passed")
