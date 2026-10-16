"""Checks for the Fed-ISIC2019 adapter (frozen-backbone features)."""
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

from bench.fl import make_config, run
from bench.tasks import _balanced_accuracy, _class_correct, isic_features, load_task

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
    try:
        run(make_config(dataset="fed_isic2019"), log=lambda *_: None)
        raise RuntimeError("ran without lr and rounds")
    except (AssertionError, TypeError):
        pass


if __name__ == "__main__":
    test_metric_matches_sklearn()
    test_task_sizes_and_validation_split()
    test_run_records_class_counts()
    test_lr_and_rounds_must_be_given()
    print("all tests passed")
