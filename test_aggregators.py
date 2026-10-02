"""Sanity checks: with 16 honest updates near +1 and 4 huge outliers,
robust rules should stay near +1 while FedAvg gets dragged away."""
import torch

from bench.aggregators import AGGREGATORS, bulyan, check_valid, get_aggregator, median
from bench.attacks import alie_z
from bench.fl import Config, run


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
    for rule in ("median", "geomed", "cc"):          # honest majority needed
        assert check_valid(rule, 5, 2) is None, rule
        assert check_valid(rule, 4, 2) is not None, rule
    assert check_valid("fedavg", 3, 2) is None       # baseline: no assumption


def test_median_even_and_odd():
    U = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    assert median(U).item() == 2.5                  # torch.median would give 2.0
    assert median(U[:3]).item() == 2.0
    # Bulyan at f=0 averages everything, whatever median it uses internally
    assert torch.allclose(bulyan(U, f=0), U.mean(0))


def test_alie_z_positive():
    assert 0 < alie_z(20, 4) < 1
    assert alie_z(4, 1) == 0                         # ALIE = honest mean (logged)
    assert alie_z(3, 2) == float("inf")


def test_alie_majority_run_is_invalid():
    # n=3, f=2: ALIE's z is infinite, so even FedAvg (no validity rule) is invalid
    res = run(Config(dataset="synthetic", aggregator="fedavg", attack="alie",
                     n_clients=3, byz_frac=0.67, device="cpu"), log=lambda *_: None)
    assert res["status"] == "invalid" and "alie" in res["reason"], res


if __name__ == "__main__":
    test_robust_rules_resist_outliers()
    test_validity_rules()
    test_median_even_and_odd()
    test_alie_z_positive()
    test_alie_majority_run_is_invalid()
    print("all tests passed")
