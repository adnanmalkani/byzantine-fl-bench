"""Checks for the validity table (RQ-A)."""
import torch

from validity_table import breakdown, build, dictator


def cell(df, n, f, rule):
    return df[(df.n == n) & (df.f == f) & (df.rule == rule)].iloc[0]


def test_validity_table():
    df = build([4, 20], [0, 1, 2])
    assert cell(df, 4, 1, "bulyan").status == "invalid"        # needs n >= 4f + 3
    assert cell(df, 4, 1, "krum").status == "invalid"          # needs n > 2f + 2
    assert cell(df, 20, 1, "bulyan").status == "valid"
    assert cell(df, 4, 1, "krum").degenerate == ""              # invalid: not evaluated
    assert cell(df, 20, 1, "krum").degenerate == "1 client"
    for rule in ("fedavg", "trimmed_mean", "multi_krum", "bulyan"):
        assert cell(df, 4, 0, rule).degenerate == "= mean", rule  # f=0: no trimming
    assert cell(df, 20, 1, "geomed").degenerate == ""
    for rule in ("median", "geomed", "cc"):                     # needs n > 2f
        assert cell(df, 4, 2, rule).status == "invalid", rule
    assert bool(cell(df, 4, 1, "fedavg").alie_degenerate)       # z = 0 at n=4, f=1
    assert not bool(cell(df, 20, 0, "fedavg").alie_degenerate)  # no attackers


def test_trimmed_mean_equals_median_when_two_or_fewer_rows_kept():
    df = build([3, 4, 5, 6], [1, 2])
    for n, f in [(3, 1), (4, 1), (5, 2), (6, 2)]:               # n - 2f <= 2
        assert cell(df, n, f, "trimmed_mean").degenerate == "= median", (n, f)
    for n, f in [(5, 1), (6, 1)]:                               # 3+ rows kept
        assert cell(df, n, f, "trimmed_mean").degenerate == "", (n, f)
    assert cell(df, 4, 1, "median").degenerate == ""            # not flagged against itself


def test_weight_checks():
    # one client with > 50% of the weight: weighted median and geomed are that
    # client; CC and the count-based rules are not
    w, eq = torch.tensor([60., 20., 10., 10.]), torch.ones(4)
    for rule in ("median", "geomed"):
        assert dictator(rule, w) and not dictator(rule, eq), rule
        assert not dictator(rule, torch.tensor([45., 35., 10., 10.])), rule   # top < 50%
    for rule in ("cc", "trimmed_mean", "multi_krum", "fedavg"):
        assert not dictator(rule, w), rule
    # breakdown: the heaviest client attacks; > 50% of the weight breaks the
    # weighted median, geomed and CC, not the count-based trimmed mean
    w6, low = torch.tensor([55., 9., 9., 9., 9., 9.]), torch.tensor([30., 14., 14., 14., 14., 14.])
    for rule in ("median", "geomed", "cc"):
        assert breakdown(rule, w6, 1) and not breakdown(rule, low, 1), rule
    assert not breakdown("trimmed_mean", w6, 1)


if __name__ == "__main__":
    test_validity_table()
    test_trimmed_mean_equals_median_when_two_or_fewer_rows_kept()
    test_weight_checks()
    print("all tests passed")
