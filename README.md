# byzantine-fl-bench

A small, reproducible benchmark of Byzantine-robust aggregation rules for
federated learning (FL), under one shared setup.

## The idea in plain English

In federated learning, many clients (phones, hospitals) train a shared model
without sharing raw data. Each round, every client sends the server an
*update*: how it wants to change the model. The server combines the updates,
usually by averaging them (FedAvg).

Averaging has a weakness. A single malicious ("Byzantine") client can send a
huge or carefully crafted update and drag the average wherever it wants.
*Robust aggregation rules* replace the plain average with something harder to
manipulate, such as a median or picking the most "typical" update.

Each paper proposing a rule tends to test it under its own settings, which
makes results hard to compare. This repo runs all of them under identical
conditions: same model, data splits, attacks and seeds.

## What's included

**Aggregation rules (8):** FedAvg (baseline), coordinate-wise Median,
Trimmed Mean, Krum, Multi-Krum, Bulyan, Geometric Median (RFA), Centered
Clipping.

**Attacks (6):** none, label flipping (data poisoning), sign flipping,
Gaussian noise, ALIE ("A Little Is Enough"), IPM (Inner Product
Manipulation). Model-poisoning attackers are omniscient: they see all benign
updates each round, a standard worst case.

**Knobs:** dataset (MNIST, Fashion-MNIST, CIFAR-10, plus a `synthetic`
dataset for quick tests), non-IID level via Dirichlet alpha (0.1 = very
skewed, 100 = roughly IID), Byzantine fraction, seeds.

If a rule's theoretical assumption is violated (e.g. Bulyan needs
n >= 4f + 3), the run is recorded as `invalid` rather than silently run.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python test_aggregators.py        # quick sanity checks
```

On Apple Silicon, PyTorch uses the GPU through MPS automatically
(`--device auto`).

## Usage

One experiment:

```bash
python run.py --dataset mnist --aggregator krum --attack alie --byz_frac 0.2 --alpha 0.5
```

A grid (resumable: finished runs are skipped):

```bash
python sweep.py --datasets mnist --alphas 0.1,0.5,100 \
  --byz_fracs 0.1,0.2,0.3 --seeds 0,1,2 --rounds 50
python summarize.py              # mean +/- std table -> summary.csv
```

## Defaults

20 clients, all participating each round; 1 local epoch; batch 32; SGD lr
0.05; 50 rounds; small CNN without BatchNorm. Server assumes the true number
of Byzantine clients (`--f_assumed` overrides this). FedAvg uses a uniform,
not sample-weighted, average so every rule sees the same inputs.

**Clean baselines** (`attack none` in `sweep.py`) have no Byzantine clients,
but the server still assumes f = round(byz_frac * n), so the clean row shows
the cost of robustness. These runs are tagged `_fa{f}` in their file names.

**Local-steps ablation:** `--local_steps K` makes every client take exactly K
SGD steps instead of one epoch, so update size no longer scales with client
data size (tagged `_ls{K}`). K = 94 matches the average steps of one epoch
(3000 samples / batch 32).

**Parallel sweeps:** `--shard i/N` runs every N-th grid point, so N processes
can split one grid (e.g. `--shard 0/2` and `--shard 1/2`).

## Known limitations / TODO

- Centered Clipping's radius `--cc_tau` should be tuned on clean runs first.
- No client sampling, no adaptive attacks against specific defenses yet.
- Add FLTrust (needs a small server dataset) and a text dataset.
