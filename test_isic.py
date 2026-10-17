"""Checks for the Fed-ISIC2019 adapter (frozen-backbone features)."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

from bench.fl import make_config, run
from bench.tasks import (ISIC_LEAKED_TEST, _balanced_accuracy, _class_correct, isic_features,
                         load_task)
from run import run_name
from summarize import isic_metrics, run_metrics

TRAIN = [9930, 3163, 2691, 1807, 655, 351]   # FLamby's per-centre split
TEST = [2483, 791, 672, 452, 164, 88]


class Fixed(torch.nn.Module):
    """Returns fixed logits, so the metric sees chosen predictions."""
    def __init__(self, logits):
        super().__init__()
        self.logits = logits

    def forward(self, x):
        return self.logits


def test_metric_matches_sklearn():
    g = torch.Generator().manual_seed(0)
    for _ in range(5):
        y = torch.randint(0, 5, (200,), generator=g)    # classes 5-7 absent, as at some centres
        logits = torch.randn(200, 8, generator=g)
        pred = logits.argmax(1)
        assert np.isclose(_balanced_accuracy(Fixed(logits), None, y),
                          balanced_accuracy_score(y.numpy(), pred.numpy()))
        correct = _class_correct(Fixed(logits), None, y)
        assert correct == [int(((pred == c) & (y == c)).sum()) for c in range(8)]


def test_task_sizes_and_validation_split():
    d = isic_features()
    assert [len(y) for _, y, _ in d["train"]] == TRAIN
    assert [len(y) for _, y, _ in d["test"]] == TEST
    cfg = make_config(dataset="fed_isic2019", lr=0.01, rounds=1)
    t = load_task(cfg, torch.device("cpu"))
    assert [len(y) for _, y in t.clients] == TRAIN and len(t.test[1]) == sum(TEST)
    assert t.build_model().weight.shape == (8, 1280)
    # tuning mode: per-centre holdout, disjoint from training, test never loaded
    v = load_task(make_config(dataset="fed_isic2019", lr=0.01, rounds=1, val_frac=0.1),
                  torch.device("cpu"))
    for c, ((xt, _), (xv, _)) in enumerate(zip(v.clients, v.client_tests)):
        assert len(xt) + len(xv) == TRAIN[c]
        assert abs(len(xv) - 0.1 * TRAIN[c]) <= 8            # rounded per class
        # together exactly the centre's training images (rows can't be keyed:
        # ISIC has byte-identical images under different IDs)
        full = d["train"][c][0]
        assert torch.allclose(xt.double().sum(0) + xv.double().sum(0), full.double().sum(0))


def test_run_records_class_counts():
    r = run(make_config(dataset="fed_isic2019", aggregator="median", lr=0.01, rounds=2,
                        local_steps=5), log=lambda *_: None)
    assert r["status"] == "ok"
    counts = np.array(r["client_test_class_counts"])
    assert counts.shape == (6, 8) and counts.sum(1).tolist() == TEST
    h = r["history"][-1]
    correct = np.array(h["client_class_correct"])
    # per-centre balanced accuracy from the stored counts = the recorded one
    for c in range(6):
        present = counts[c] > 0
        assert np.isclose((correct[c][present] / counts[c][present]).mean(), h["client_acc"][c])
    # label flip on 8 classes: y -> 7 - y (must run, not crash)
    r = run(make_config(dataset="fed_isic2019", attack="label_flip", byz_ids="5", lr=0.01,
                        rounds=1, local_steps=2), log=lambda *_: None)
    assert r["status"] == "ok"


def test_lr_and_rounds_must_be_given():
    # Preset fixed by the screen + Amendment 2 (README, 2026-10-17)
    cfg = make_config(dataset="fed_isic2019")
    assert (cfg.lr, cfg.rounds) == (0.01, 101)
    try:
        run(make_config(dataset="fed_isic2019", lr=None, rounds=1), log=lambda *_: None)
        raise RuntimeError("ran without lr")
    except (AssertionError, TypeError):
        pass


def test_leaked_list_matches_duplicates_csv():
    path = Path("results_tables/isic_leaked_test.csv")
    if not path.exists():  # gitignored; written by isic_duplicates.py
        print("skipped: run isic_duplicates.py to check the leaked list")
        return
    assert set(pd.read_csv(path).test_name) == ISIC_LEAKED_TEST and len(ISIC_LEAKED_TEST) == 17


def test_drop_leaked_changes_test_sets_only():
    kw = dict(dataset="fed_isic2019", aggregator="cc", lr=0.01, rounds=2, local_steps=5)
    a = run(make_config(**kw), log=lambda *_: None)
    b = run(make_config(**kw, drop_leaked=True), log=lambda *_: None)
    removed = np.array(a["client_test_class_counts"]) - np.array(b["client_test_class_counts"])
    # centre 0: MEL 3, NV 4, BCC 3, AK 2, BKL 4; centre 5: NV 1 (README)
    assert removed[0].tolist() == [3, 4, 3, 2, 4, 0, 0, 0]
    assert removed[5].tolist() == [0, 1, 0, 0, 0, 0, 0, 0] and removed[1:5].sum() == 0
    assert a["diag"] == b["diag"]                       # training is identical
    assert run_name(make_config(**kw, drop_leaked=True)).endswith("_lr0.01_r2_s0")
    assert "_noleak" in run_name(make_config(**kw, drop_leaked=True))
    assert "_noleak" not in run_name(make_config(**kw))
    for e in (dict(val_frac=0.1), dict(dataset="fed_tcga_brca")):
        try:
            load_task(make_config(**{**kw, **e}, drop_leaked=True), torch.device("cpu"))
            raise RuntimeError("drop_leaked accepted outside ISIC test sets")
        except AssertionError:
            pass


def test_isic_worst_centre_metric():
    # Hand example: 3 centres, 8 classes; one evaluation
    counts = [[10, 6, 4, 0, 0, 0, 0, 0],    # k=5: classes 0,1; k=10: class 0 only -> out
              [20, 20, 20, 0, 0, 0, 0, 0],  # all classes count at both k
              [5, 9, 0, 0, 0, 0, 0, 0]]     # k=5: both; k=10: none -> out
    correct = [[5, 3, 4, 0, 0, 0, 0, 0], [20, 10, 0, 0, 0, 0, 0, 0], [5, 0, 0, 0, 0, 0, 0, 0]]
    r = {"history": [{"acc": 0.4, "client_class_correct": correct}],
         "client_test_class_counts": counts, "byz_ids": [1]}
    m = isic_metrics(r, last=1)
    ba5 = [(0.5 + 0.5) / 2, (1 + 0.5 + 0) / 3, (1 + 0) / 2]
    assert np.allclose(m["client_acc"], ba5) and m["avg_acc"] == 0.4
    assert np.isclose(m["worst_acc"], 0.5) and np.isclose(m["gap"], 0.0)
    assert np.isclose(m["worst_honest_acc"], 0.5)          # centre 1 excluded (attacker)
    assert np.isclose(m["worst_acc_k10"], 0.5) and np.isclose(m["gap_k10"], 0.0)
    assert np.isnan(m["worst_honest_acc_k10"])             # only centre 1 counts, an attacker
    # Real run: centre 0 has >= 24 test images of every class, so k changes
    # nothing there and the k-rule accuracy equals the recorded one
    real = run(make_config(dataset="fed_isic2019", lr=0.01, rounds=3, local_steps=5),
               log=lambda *_: None)
    m = run_metrics(real)
    assert np.isclose(m["client_acc"][0], np.mean([h["client_acc"][0] for h in real["history"]]))
    assert np.isclose(m["avg_acc"], real["final_acc"])
    assert m["worst_acc"] <= min(m["client_acc"]) + 1e-12


if __name__ == "__main__":
    test_metric_matches_sklearn()
    test_task_sizes_and_validation_split()
    test_run_records_class_counts()
    test_lr_and_rounds_must_be_given()
    test_leaked_list_matches_duplicates_csv()
    test_drop_leaked_changes_test_sets_only()
    test_isic_worst_centre_metric()
    print("all tests passed")
