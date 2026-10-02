"""Sanity checks: with 16 honest updates near +1 and 4 huge outliers,
robust rules should stay near +1 while FedAvg gets dragged away."""
import torch

from bench.aggregators import AGGREGATORS, check_valid, get_aggregator
from bench.attacks import alie_z


def test_robust_rules_resist_outliers():
    torch.manual_seed(0)
    n, f, d = 20, 4, 50
    honest = 1.0 + 0.1 * torch.randn(n - f, d)
    U = torch.cat([honest, torch.full((f, d), -100.0)])
    for name in AGGREGATORS:
        agg = get_aggregator(name, cc_tau=1.0)
        out = agg(U, f=f)
        err = (out - 1.0).abs().mean().item()
        if name == "fedavg":
            assert err > 10, name
        elif name == "cc":
            assert err < 1.0, (name, err)  # first round, centre starts at 0
        else:
            assert err < 0.2, (name, err)


def test_validity_rules():
    assert check_valid("bulyan", 20, 4) is None
    assert check_valid("bulyan", 20, 5) is not None
    assert check_valid("krum", 20, 8) is None
    assert check_valid("krum", 20, 9) is not None


def test_alie_z_positive():
    assert 0 < alie_z(20, 4) < 1


if __name__ == "__main__":
    test_robust_rules_resist_outliers()
    test_validity_rules()
    test_alie_z_positive()
    print("all tests passed")
