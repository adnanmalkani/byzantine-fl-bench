"""Checks for the sample-size-weighted aggregation variants."""
import torch

from bench.aggregators import AGGREGATORS, check_valid, fedavg, get_aggregator, krum, median, trimmed_mean


def col(*v):
    return torch.tensor(v, dtype=torch.float32)[:, None]  # 1 coordinate, 1 row per client


def close(t, x):
    return abs(t.item() - x) < 1e-5


def test_equal_weights_match_unweighted():
    torch.manual_seed(0)
    for n, f in [(4, 1), (5, 1), (20, 4)]:
        U = torch.randn(n, 30)
        w = torch.full((n,), 7.0)
        for name in AGGREGATORS:
            if check_valid(name, n, f):  # e.g. Bulyan at n=4 (run() never gets here)
                continue
            a = get_aggregator(name)(U, f=f)
            b = get_aggregator(name)(U, f=f, w=w)  # fresh CC state for each
            assert torch.allclose(a, b, atol=1e-6), (name, n, f)


def test_hand_computed_examples():
    w = torch.tensor([1.0, 1.0, 5.0])
    assert close(fedavg(col(1, 2, 3), w=w), (1 + 2 + 15) / 7)
    assert close(median(col(1, 2, 3), w=w), 3.0)            # 5 of 7 weight on 3
    # cumulative weight hits exactly half at the first value -> average with next
    assert close(median(col(1, 2, 3, 4), w=torch.tensor([3.0, 1, 1, 1])), 1.5)
    # trimming is by count (drop 0 and 100), then weights 1 and 3 on values 1 and 2
    U, w4 = col(0, 1, 2, 100), torch.tensor([1.0, 1, 3, 1])
    assert close(trimmed_mean(U, f=1, w=w4), (1 * 1 + 2 * 3) / 4)


def test_krum_ignores_weights():
    U = torch.randn(8, 10)
    assert torch.equal(krum(U, f=1), krum(U, f=1, w=torch.arange(1.0, 9)))


def test_weights_change_robust_rules_only_in_averaging():
    # 3 honest clients near +1, 1 outlier at -100: weighting must not let the
    # outlier back in for rules that trim/select it away
    U = torch.cat([1 + 0.01 * torch.randn(3, 5), torch.full((1, 5), -100.0)])
    w = torch.tensor([10.0, 1, 1, 1])
    for name in ("median", "trimmed_mean", "geomed"):
        out = get_aggregator(name)(U, f=1, w=w)
        assert (out - 1).abs().max() < 0.1, name


if __name__ == "__main__":
    test_equal_weights_match_unweighted()
    test_hand_computed_examples()
    test_krum_ignores_weights()
    test_weights_change_robust_rules_only_in_averaging()
    print("all tests passed")
