"""Checks for the validity table (RQ-A)."""
from validity_table import build


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


if __name__ == "__main__":
    test_validity_table()
    print("all tests passed")
