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
python summarize.py results      # mean and 95% CI -> results_tables/summary_results{,_pooled}.csv
```

Summaries average every metric over the last 3 evaluations. Natural splits
add the worst hospital, the worst *honest* hospital (attackers' own test
sets excluded) and the best-worst gap. ALIE runs with z = 0 (n = 4, 6 or 20
with f = 1) are flagged `degenerate`: the attackers send the mean of the
benign updates, so they are kept apart from real attacks.

Heart analysis and figures (all to `results_tables/`):

```bash
python analyze_heart.py          # per-attacker tables, exclusion diagnostics, scatter
python validity_table.py         # validity.csv + fig_validity_heatmap
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

## Tuning CC's tau (rule v2.1, per dataset, never on the test set)

Rule v2 was fixed on 2026-10-03, before any ISIC/TCGA data was seen, and
amended to v2.1 the same day, before the MNIST multi-seed check was run
(`tune_cc.py`):

1. **Reference run:** clean, unweighted FedAvg in validation mode (10% of
   the training data held out, stratified; per hospital for natural splits),
   first 10 rounds, the dataset's tuning seeds (MNIST 5, heart 20).
2. **tau0** = median distance of each client update from the previous
   aggregate, ||u_i - Delta_prev||, over clients, rounds 2-10 and seeds.
   This is the quantity CC clips (round 1 has no previous aggregate).
3. **Rounding:** tau0 snaps to the nearest value of the ladder {1, 3} x 10^k
   (log scale). The check grid is that value and two ladder steps either
   side (about tau0/10 to 10 tau0).
4. **Validation check:** clean CC for each grid value, in every weighting
   variant used for the dataset, at the dataset's main round count. Per
   variant, the paired-by-seed difference in validation accuracy vs tau0
   with a 95% t CI.
5. **Decision:** keep tau0 unless some value is better by a practically
   relevant margin, i.e. the CI lower bound exceeds +0.5 points, in every
   variant; then take the highest mean among those. One tau per dataset,
   shared across variants. No multiplicity correction. (v2 used a lower
   bound > 0. Because the check uses clean accuracy only, it would
   systematically favour larger tau, i.e. less clipping and less
   robustness, whenever seed noise is small, as on MNIST; v2.1 adds the
   margin for that reason.)
6. The full grid is reported as an appendix sensitivity table
   (`results_tables/tau_sensitivity_<dataset>.csv`).

Results:
- Fed-Heart-Disease (20 seeds, both variants): tau0 = 0.035 -> 0.03; no grid
  value is better in either variant, so **tau = 0.03**.
- MNIST (5 seeds, unweighted): tau0 = 0.50 -> 0.3. tau = 1 and 3 are
  statistically but not practically better (+0.09 and +0.11 points, CI lower
  bounds 0.06 and 0.04, below the 0.5-point margin), so **tau = 0.3**. Under
  v2 (no margin) the rule would have switched to tau = 3, where CC barely
  clips honest updates.

**History (for transparency).** Rule v1 chose the smallest tau within 0.3
points of the best validation accuracy, on a fixed grid {0.1, ..., 10}.
- MNIST (seed 0): v1 gave 0.1. Extending the grid down (0.03: -0.96 pt,
  0.01: -3.6 pt) confirmed 0.1 was not an edge artefact.
- Heart (20 seeds, grid scaled to its update size): v1 gave 0.03
  (unweighted) and 0.003 (weighted, smallest grid value). Paired 95% CIs
  showed these differences were within seed noise (about +/- 1 pt; 49
  validation patients), and different tau per variant would confound the
  weighted-vs-unweighted comparison. We therefore deviated from v1 and used
  a shared tau = 0.03 (the best mean in both variants), which motivated
  rule v2; v2 independently gives 0.03.
  Old outputs: `results_legacy/tune_old_rule/`.

## Fed-Heart-Disease (FLamby, natural splits)

Setup: `sh setup_flamby.sh`, then download the data with FLamby's script
(asks you to accept the UCI licence) into `data/flamby/fed_heart_disease`.

- **Local training:** FLamby's FedAvg settings (logistic regression, SGD lr
  0.001, batch 4, 100 local steps per round); CPU. `reproduce_flamby_heart.py`
  reproduces FLamby's published FedAvg baseline (pooled accuracy 0.748 +/-
  0.027 at FLamby's 15 rounds, seeds 42-46).
- **Rounds: 106** (main setting); FLamby's 15 rounds is a secondary setting.
  Chosen by `convergence_heart.py` on the pooled validation set (10%
  stratified holdout per hospital), 20 seeds, with the rule fixed in advance:
  the smallest round after which mean validation accuracy never drops more
  than 0.5 points below its mean over rounds 151-200. Validation loss is at
  its minimum at the same point (round 89-106).
- **Normalisation:** each hospital's train and test data use that hospital's
  training statistics; FLamby's pooled test set uses pooled training
  statistics (a FLamby convention we keep for comparability).
- **Attackers:** enumerated, not random (`sweep.py --fs`): every hospital for
  f = 1, every pair for f = 2; each run records `byz_ids`.
- **Seeds:** 20. `summarize.py` reports means with 95% CIs.
- **Seeding fix (2026-10-04):** FLamby calls `torch.manual_seed(42)` when
  imported. It used to be imported lazily inside the first heart run of each
  process, after that run was seeded, so that run started from the seed-42
  model. Fixed in `bench/tasks.py` (FLamby is imported at module load) with a
  regression test. Affected and re-run: 4 grid runs (unweighted FedAvg,
  clean, f = 0, seeds 0-3: the first run of each shard), 3 tuning runs and
  convergence seed 0. tau (0.03), R* (106) and the FLamby reproduction did
  not change. Old files: `results_legacy/first_run_bug/`.

## Aggregator diagnostics

Every run stores `diag`: per round, per client id, who the rule kept or
suppressed (Krum/Multi-Krum selections, share of coordinates trimmed or
kept, the share of the median each client determines, geometric-median
weights, CC clip events and factors). See `bench/aggregators.py`.
`analyze_heart.py` turns them into one comparable number per hospital, its
*relative influence*: effective share of the aggregate / nominal share (1/n,
or its sample share when weighted); below 1 = down-weighted.

Equivalences to keep in mind when reading tables: with f = 0, trimmed mean,
Multi-Krum and Bulyan are the plain mean; when n - 2f <= 2 (e.g. n = 4,
f = 1) the unweighted trimmed mean *is* the coordinate-wise median
(`validity_table.py` flags both).

## Known limitations / TODO

- No client sampling, no adaptive attacks against specific defenses yet.
- Add FLTrust (needs a small server dataset) and a text dataset.
