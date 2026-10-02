"""Byzantine attacks.

Model-poisoning attacks replace the Byzantine clients' updates. The attacker
is omniscient: it sees every benign update this round (a standard worst case).
label_flip is data poisoning: those clients train on flipped labels instead.
"""
import torch

ATTACKS = ["none", "label_flip", "sign_flip", "gaussian", "alie", "ipm"]


def alie_z(n: int, f: int) -> float:
    # "A Little Is Enough" (Baruch et al., 2019): largest shift that still
    # looks like a normal client to median/Krum-style rules
    s = n // 2 + 1 - f
    p = (n - s) / n
    return torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(p)).item()


def apply_attack(name, honest, byz_own, n, f, gen, sigma=1.0, ipm_eps=0.5):
    """honest: (n-f) x d benign updates; byz_own: f x d updates the Byzantine
    clients computed on their own data. Returns the f x d updates they send."""
    if name in ("none", "label_flip"):
        return byz_own
    if name == "sign_flip":
        return -byz_own
    if name == "gaussian":
        return sigma * torch.randn(byz_own.shape, generator=gen)
    benign = torch.cat([honest, byz_own])
    mu, sd = benign.mean(0), benign.std(0)
    if name == "alie":
        return (mu - alie_z(n, f) * sd).expand_as(byz_own).clone()
    if name == "ipm":
        # Inner Product Manipulation (Xie et al., 2020)
        return (-ipm_eps * mu).expand_as(byz_own).clone()
    raise ValueError(f"unknown attack: {name}")
