"""Checks for aggregator diagnostics (who each rule keeps or suppresses)."""
import torch

from bench.aggregators import AGGREGATORS, check_valid, get_aggregator
from bench.fl import Config, run


def outlier_case(n=7, f=1, d=40):
    torch.manual_seed(0)
    U = 1 + 0.1 * torch.randn(n, d)
    U[n - 1] = -100.0                   # last row: obvious outlier
    return U, n, f


def test_diagnostics_do_not_change_outputs():
    torch.manual_seed(1)
    for n, f in [(4, 1), (7, 1), (20, 4)]:
        U, w = torch.randn(n, 30), torch.rand(n) + 0.5
        for name in AGGREGATORS:
            if check_valid(name, n, f):
                continue
            for weights in (None, w):
                a = get_aggregator(name)(U, f=f, w=weights)
                b = get_aggregator(name)(U, f=f, w=weights, info={})
                assert torch.equal(a, b), (name, n, f)


def test_outlier_is_suppressed_in_diagnostics():
    U, n, f = outlier_case()
    out = {}
    for name in AGGREGATORS:
        info = {}
        get_aggregator(name, cc_tau=1.0)(U, f=f, info=info)
        out[name] = info
    assert abs(sum(out["fedavg"]["weights"]) - 1) < 1e-6
    assert out["median"]["used_share"][n - 1] == 0
    assert abs(sum(out["median"]["used_share"]) - 1) < 1e-6     # odd n: one row per coord
    assert out["trimmed_mean"]["trimmed_share"][n - 1] == 1.0
    assert abs(sum(out["trimmed_mean"]["trimmed_share"]) - 2 * f) < 1e-6
    assert n - 1 not in out["krum"]["selected"] and len(out["krum"]["selected"]) == 1
    assert n - 1 not in out["multi_krum"]["selected"] and len(out["multi_krum"]["selected"]) == n - f
    assert n - 1 not in out["bulyan"]["selected"] and out["bulyan"]["kept_share"][n - 1] == 0
    assert out["geomed"]["weights"][n - 1] < 0.01
    assert out["cc"]["clipped"][n - 1] and out["cc"]["scale"][n - 1] < 0.02


def test_even_median_shares_split():
    U = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    info = {}
    get_aggregator("median")(U, info=info)
    assert info["used_share"] == [0.0, 0.5, 0.5, 0.0]


def test_run_maps_rows_to_client_ids():
    # Attackers 0 and 2 sign-flip with a large norm; rows of U are honest
    # first, so without re-indexing their diagnostics would land on 3 and 4
    res = run(Config(dataset="synthetic", n_clients=5, byz_ids="0,2", attack="gaussian",
                     gaussian_sigma=100.0, aggregator="geomed", rounds=2, device="cpu"),
              log=lambda *_: None)
    for d in res["diag"]:
        w = d["weights"]
        assert max(w[0], w[2]) < min(w[1], w[3], w[4]), w
    res = run(Config(dataset="synthetic", n_clients=7, byz_ids="0", attack="gaussian",
                     gaussian_sigma=100.0, aggregator="krum", f_assumed=1, rounds=2,
                     device="cpu"), log=lambda *_: None)
    assert all(0 not in d["selected"] for d in res["diag"])


if __name__ == "__main__":
    test_diagnostics_do_not_change_outputs()
    test_outlier_is_suppressed_in_diagnostics()
    test_even_median_shares_split()
    test_run_maps_rows_to_client_ids()
    print("all tests passed")
