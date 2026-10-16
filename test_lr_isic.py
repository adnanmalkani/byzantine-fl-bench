"""Checks for the ISIC learning-rate rule: collapse classes, and one short run."""
import tempfile
from pathlib import Path

import numpy as np

from select_lr_isic import classify, one_run


def rec(b, loss=None, error=None):
    if loss is None and b is not None:
        loss = [1.0] * len(b)
    return {"ba": b, "loss": loss, "error": error}


def test_classify():
    assert classify(rec([0.13, 0.31, 0.45, 0.5])) == "ok"
    # Chance-level rounds before learning are not a collapse
    assert classify(rec([0.125, 0.15, 0.35, 0.4])) == "ok"
    # Learned (>= 0.30), then below 0.20 later
    assert classify(rec([0.45, 0.5, 0.125])) == "learned_lost"
    assert classify(rec([0.30, 0.199])) == "learned_lost"
    # Drops that stay at or above 0.20 are not a collapse
    assert classify(rec([0.5, 0.20, 0.45])) == "ok"
    # Below 0.20 only before first reaching 0.30 is not a collapse
    assert classify(rec([0.125, 0.3, 0.25])) == "ok"
    assert classify(rec([0.125, 0.2, 0.299])) == "not_learned"
    assert classify(rec([0.5, float("nan")])) == "divergence"
    assert classify(rec([0.5, 0.5], loss=[1.0, float("inf")])) == "divergence"


def test_one_run_uses_validation_and_resumes():
    with tempfile.TemporaryDirectory() as d:
        r = one_run(0.01, 0, rounds=2, folder=Path(d))
        assert len(r["ba"]) == 2 and np.isfinite(r["ba"]).all()
        assert one_run(0.01, 0, rounds=2, folder=Path(d)) == r    # read back, not re-run
        assert len(list(Path(d).glob("*.json"))) == 1


if __name__ == "__main__":
    test_classify()
    test_one_run_uses_validation_and_resumes()
    print("all tests passed")
