"""Checks for the TCGA learning-rate rule: collapse classes and the plateau round."""
import numpy as np

from select_lr_tcga import classify, plateau_round


def rec(c, loss=None, error=None):
    if loss is None and c is not None:
        loss = [1.0] * len(c)
    return {"c": c, "loss": loss, "error": error}


def test_classify():
    assert classify(rec([0.4, 0.62, 0.7, 0.72])) == "ok"
    # Early chance-level rounds before learning are not a collapse
    assert classify(rec([0.37, 0.45, 0.65, 0.7])) == "ok"
    # Learned (>= 0.60), then below 0.50 later
    assert classify(rec([0.7, 0.73, 0.37])) == "learned_lost"
    assert classify(rec([0.60, 0.499])) == "learned_lost"
    # Drops that stay at or above 0.50 are not a collapse
    assert classify(rec([0.75, 0.50, 0.7])) == "ok"
    # Below 0.50 only BEFORE first reaching 0.60 is not a collapse
    assert classify(rec([0.49, 0.6, 0.55])) == "ok"
    assert classify(rec([0.5, 0.55, 0.59])) == "not_learned"
    assert classify(rec([0.7, float("nan")])) == "divergence"
    assert classify(rec([0.7, 0.7], loss=[1.0, float("inf")])) == "divergence"
    assert classify(rec(None, None, error="NaNs detected in inputs")) == "divergence"


def test_plateau_round():
    A = np.full(200, 0.70)
    A[:50] = np.linspace(0.5, 0.69, 50)          # rising, all below A_end - 0.005
    R, a_end = plateau_round(A, 151, 200)
    assert np.isclose(a_end, 0.70)
    assert R == 51
    assert plateau_round(np.full(200, 0.7), 151, 200)[0] == 1
    # A late dip below the tolerance moves R* past it
    B = np.full(200, 0.7); B[120] = 0.68
    assert plateau_round(B, 151, 200)[0] == 122


if __name__ == "__main__":
    test_classify()
    test_plateau_round()
    print("all tests passed")
