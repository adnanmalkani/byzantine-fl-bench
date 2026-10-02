"""Robust aggregation rules.

Every rule takes U: an (n x d) tensor of client updates (one row per client)
and f: the number of Byzantine clients the server assumes, and returns a
single d-dimensional update.

Weighted variants: pass w, an (n,) tensor of client sample sizes. Selection
and trimming are unchanged; only the final averaging step uses the weights,
so "weighting" and "outlier suppression" can be studied separately. Krum
selects a single update, so its weighted variant is identical. w=None gives
the unweighted rules exactly as before.
"""
import torch


def _wmean(X, w):
    # Weighted mean over dim 0; w is (k,) or per-coordinate (k, d)
    if w.dim() == 1:
        w = w[:, None]
    return (X * w).sum(0) / w.sum(0)


def fedavg(U, f=0, w=None):
    return U.mean(0) if w is None else _wmean(U, w)


def _median(U):
    # Per-coordinate median; for even n, the average of the two middle values
    # (torch.median would return the lower one, a bias that matters at small n)
    s = U.sort(0).values
    n = U.shape[0]
    return (s[(n - 1) // 2] + s[n // 2]) / 2


def _wmedian(U, w):
    # Per-coordinate weighted median: the first sorted value where cumulative
    # weight reaches half the total; on an exact tie, the average with the
    # next value (so equal weights give the same result as _median)
    s, idx = U.sort(0)
    cw = w[idx].cumsum(0)
    half = w.sum() / 2
    tol = 1e-6 * w.sum()
    k = (cw < half - tol).sum(0, keepdim=True)
    lo = s.gather(0, k)
    hi = s.gather(0, (k + 1).clamp(max=U.shape[0] - 1))
    tie = (cw.gather(0, k) - half).abs() <= tol
    return torch.where(tie, (lo + hi) / 2, lo)[0]


def median(U, f=0, w=None):
    # Coordinate-wise median (Yin et al., 2018)
    return _median(U) if w is None else _wmedian(U, w)


def trimmed_mean(U, f=0, w=None):
    # Drop the f largest and f smallest values per coordinate, average the rest
    n = U.shape[0]
    if w is None:
        s = U.sort(0).values
        return s[f:n - f].mean(0)
    s, idx = U.sort(0)
    return _wmean(s[f:n - f], w[idx][f:n - f])


def _krum_scores(U, f):
    # Score = sum of squared distances to the n - f - 2 nearest other updates
    n = U.shape[0]
    k = n - f - 2
    D = torch.cdist(U, U).pow(2)
    return D.sort(1).values[:, 1:k + 1].sum(1)  # column 0 is distance to self


def krum(U, f=0, w=None):
    # Blanchard et al., 2017: pick the single most "central" update
    return U[_krum_scores(U, f).argmin()]


def multi_krum(U, f=0, w=None):
    # Average the n - f updates with the best Krum scores
    m = U.shape[0] - f
    idx = _krum_scores(U, f).argsort()[:m]
    return U[idx].mean(0) if w is None else _wmean(U[idx], w[idx])


def bulyan(U, f=0, w=None):
    # El Mhamdi et al., 2018: iterative Krum selection, then a trimmed
    # mean around the coordinate-wise median of the selected set
    n = U.shape[0]
    theta = n - 2 * f
    remaining, selected = list(range(n)), []
    for _ in range(theta):
        sub = U[remaining]
        if len(remaining) > f + 2:
            j = remaining[_krum_scores(sub, f).argmin().item()]
        else:
            j = remaining[0]
        selected.append(j)
        remaining.remove(j)
    S = U[selected]
    beta = theta - 2 * f
    med = _median(S)
    idx = (S - med).abs().argsort(0)[:beta]
    if w is None:
        return S.gather(0, idx).mean(0)
    return _wmean(S.gather(0, idx), w[selected][idx])


def geomed(U, f=0, w=None, iters=10, eps=1e-6):
    # Geometric median via Weiszfeld iterations (RFA, Pillutla et al., 2022);
    # with w, the weighted geometric median (RFA's original form)
    a = torch.ones(U.shape[0]) if w is None else w
    z = U.mean(0) if w is None else _wmean(U, w)
    for _ in range(iters):
        b = a / torch.clamp((U - z).norm(dim=1), min=eps)
        z = (b[:, None] * U).sum(0) / b.sum()
    return z


class CenteredClipping:
    """Karimireddy et al., 2021. Clips each update's distance from the
    previous aggregate to radius tau. Stateful across rounds."""

    def __init__(self, tau=1.0, iters=1):
        self.tau, self.iters, self.v = tau, iters, None

    def __call__(self, U, f=0, w=None):
        v = torch.zeros(U.shape[1]) if self.v is None else self.v
        for _ in range(self.iters):
            diff = U - v
            scale = torch.clamp(self.tau / (diff.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
            v = v + ((diff * scale).mean(0) if w is None else _wmean(diff * scale, w))
        self.v = v.clone()
        return v


AGGREGATORS = {
    "fedavg": fedavg,
    "median": median,
    "trimmed_mean": trimmed_mean,
    "krum": krum,
    "multi_krum": multi_krum,
    "bulyan": bulyan,
    "geomed": geomed,
    "cc": CenteredClipping,
}


def check_valid(name: str, n: int, f: int):
    """Return an error string if the rule's assumption on (n, f) is violated."""
    rules = {
        # Median, geometric median and CC can only resist a minority of attackers
        "median": (n > 2 * f, "needs n > 2f"),
        "geomed": (n > 2 * f, "needs n > 2f"),
        "cc": (n > 2 * f, "needs n > 2f"),
        "trimmed_mean": (n > 2 * f, "needs n > 2f"),
        "krum": (n > 2 * f + 2, "needs n > 2f + 2"),
        "multi_krum": (n > 2 * f + 2, "needs n > 2f + 2"),
        "bulyan": (n >= 4 * f + 3, "needs n >= 4f + 3"),
    }
    ok, msg = rules.get(name, (True, ""))
    return None if ok else f"{name} {msg} (n={n}, f={f})"


def get_aggregator(name: str, cc_tau: float = 1.0):
    if name == "cc":
        return CenteredClipping(tau=cc_tau)
    return AGGREGATORS[name]
