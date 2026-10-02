"""Experimental design helpers."""
import itertools
from math import comb

import numpy as np


def attacker_sets(n: int, f: int, max_sets: int = 15, seed: int = 0):
    """Which clients are Byzantine, for natural splits where identity matters.
    Every f-subset of the n clients if there are at most max_sets of them
    (f=1: each hospital; f=2 with n <= 6: all pairs); otherwise max_sets
    distinct subsets sampled with a fixed seed. Returned as "i,j" strings
    (Config.byz_ids), sorted, so results can be reported per attacker."""
    if f == 0:
        return [None]
    if comb(n, f) <= max_sets:
        sets = list(itertools.combinations(range(n), f))
    else:
        rng = np.random.default_rng(seed)
        chosen = set()
        while len(chosen) < max_sets:
            chosen.add(tuple(sorted(rng.choice(n, f, replace=False).tolist())))
        sets = sorted(chosen)
    return [",".join(map(str, s)) for s in sets]
