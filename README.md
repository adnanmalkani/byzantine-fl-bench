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

**Attacks (7):** none, label flipping (data poisoning), sign flipping,
Gaussian noise, ALIE ("A Little Is Enough"), ALIE with a fixed z = 1
(`alie_z1`), IPM (Inner Product Manipulation). Model-poisoning attackers are
omniscient: they see all benign updates each round, a standard worst case.

**`alie_z1` (pre-registered 2026-10-13, before any run with it).** ALIE
sends mean − z × std of the benign updates, per coordinate. Baruch et al.'s
formula sets z from n and f, and with few hospitals it gives z ≤ 0.43 (z = 0
at n = 4 or 6 with f = 1, where ALIE just sends the benign mean). So the
standard ALIE never really tests an attacker hiding inside the honest
spread, which is the question here: can a defence tell an attacker from an
honest hospital whose data is simply different? `alie_z1` fixes z = 1. It is
run wherever ALIE is defined (same validity: invalid when attackers are half
the clients or more) and differs only in z. Baruch's version (`alie`) is kept
for comparability with the literature. Added to the heart grid (re-run for
this attack only) and to the TCGA grid.

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

Summaries average every metric over the last 3 evaluations. On natural
splits the main metrics are `avg_acc` (accuracy over all hospitals' test
patients, each hospital with its own normalisation = sample-weighted mean of
per-hospital accuracies) and `worst_honest_acc` (worst non-attacker
hospital); `worst_acc` (attackers included) and the best-worst `gap` are
appendix metrics. FLamby's pooled-test metric appears only in the
reproduction scripts. ALIE runs with z = 0 (n = 4, 6 or 20
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
- Fed-TCGA-BRCA (20 seeds, both variants, 243 rounds, validation C-index):
  tau0 = 0.0246 -> 0.03. No grid value is better by the margin in either
  variant (best: tau = 0.1, +0.10 / +0.14 points, CI lower bounds -0.15 /
  -0.02), so **tau = 0.03**. tau = 0.1 and 0.3 give identical results, so
  from 0.1 up CC no longer clips.

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
- **`alie_z1` (2026-10-13):** the heart grid is re-run for this attack
  only (same command as the grid with `--attacks alie_z1`; 3,200 grid
  points, 1,040 that train). Nothing else in `results_heart` changes.
  `analyze_heart.py`'s "mean of real attacks" now includes `alie_z1` (label
  flip, sign flip, Gaussian, IPM, `alie_z1`; decided before its results were
  seen). The earlier 4-attack mean is kept as an appendix table
  (`heart_attacker_f1_mean4_appendix*.csv`). Result: `alie_z1` is the
  mildest real attack on average, but the only one that leaves FedAvg
  untouched while costing median, trimmed mean and geomed 7.6-8.8 points on
  the worst honest hospital (2026-10-15).
- **Seeding fix (2026-10-04):** FLamby calls `torch.manual_seed(42)` when
  imported. It used to be imported lazily inside the first heart run of each
  process, after that run was seeded, so that run started from the seed-42
  model. Fixed in `bench/tasks.py` (FLamby is imported at module load) with a
  regression test. Affected and re-run: 4 grid runs (unweighted FedAvg,
  clean, f = 0, seeds 0-3: the first run of each shard), 3 tuning runs and
  convergence seed 0. tau (0.03), R* (106) and the FLamby reproduction did
  not change. Old files: `results_legacy/first_run_bug/`.

## Fed-TCGA-BRCA (FLamby, natural splits)

Data ships with FLamby (`brca.csv`, 6 regions, 866 train / 222 test
patients). FLamby asks once, interactively, that you accept the GDC data
terms (https://gdc.cancer.gov/access-data/data-access-processes-and-tools);
until then any TCGA run stops at the prompt. Label flip does not apply
(survival data) and raises an error.

- **Reproduction:** `reproduce_flamby_tcga.py` (FLamby's settings: Cox
  model, SGD lr 0.1, batch 8, 100 local steps, 5 rounds, seeds 42-46).
  Means are within one published std on all 7 test sets (pooled C-index
  0.710 +/- 0.196 vs FLamby 0.735 +/- 0.175). Not per-seed identical: the
  initial model is identical to FLamby's, but batch order differs, and at
  lr 0.1 training is unstable enough that the order decides which seed
  collapses (FLamby: seed 43, ours: seed 46). FLamby's settings are used
  only for this reproduction table.

### Main-setting learning rate and rounds (pre-registered 2026-10-11, before any run)

Clean sample-weighted FedAvg, FLamby's other settings (batch 8, 100 local
steps), seeds 0-19, 200 rounds, evaluated every round on the pooled
validation set (10% holdout per region, stratified by event: 86 patients,
12 deaths, 319-541 comparable pairs depending on the seed; the test set is
never used). C(r) = pooled validation C-index after round r.

1. **Learning-rate grid:** {0.1, 0.03, 0.01, 0.003}, all with the same 20
   seeds and 200 rounds.
2. **Collapse** (per seed, decided from C(r) alone). A seed has collapsed if
   - *divergence:* the training loss, the model or its risk scores become
     non-finite (NaN/inf) at any round, or the run fails because of it; or
   - *learned, then lost:* C(r) >= 0.60 at some round r and C(r') < 0.50 at
     a later round r' > r. 0.50 is a random ranking; 0.60 means the seed
     had clearly learned something first. Requiring this swing, rather than
     any drop, keeps round-to-round noise from counting (only 12 validation
     deaths).

   Disclosure: the 0.50 and 0.60 thresholds were set after seeing the two
   test-set collapses in the reproduction (pooled C-index 0.37 and 0.42 at
   FLamby's settings), but before any validation run.

   A seed whose C(r) never reaches 0.60 within 200 rounds has *not learned*.
   This is reported separately and does not count as a collapse.
3. **Choice:** the largest learning rate with zero collapsed seeds out of 20.
   If the chosen rate has any not-learned seed, or no rate qualifies, stop:
   no grid until this is discussed.
4. **Rounds:** the heart rule, on the chosen rate's 20 curves (no new runs).
   A(r) = mean of C(r) over seeds, A_end = mean of A(r) over rounds
   151-200, R* = the smallest r with A(r') >= A_end - 0.005 for every
   r' >= r. If R* > 150 (no plateau within 200 rounds), the chosen rate is
   re-run for 400 rounds and the rule applied with A_end over 301-400.

**Screen result (2026-10-11).** Collapsed / not-learned seeds out of 20:
lr 0.1: 10 / 2; 0.03: 5 / 2; 0.01: 1 / 2; 0.003: 0 / 3. No divergence.
The stop in step 3 fired: 0.003 has three not-learned seeds (13, 15, 19).
Seeds 13 and 19 never reach 0.60 at any rate (best 0.51-0.57), so the
learning rate is not the cause. 0.01's single collapse is seed 16: 0.602
at round 1, 0.487 at round 4 (its only round below 0.50), 0.664 at the end.

**Amendment 1 (2026-10-12, made AFTER seeing the screen results above,
before any further run).** The rule did not foresee seeds that fail at
every rate. Stability is the priority: if FedAvg collapses on its own,
attack damage cannot be told apart from instability.
- Seeds that are not-learned at every rate in the grid (13 and 19) are
  *hard validation splits*: reported, and excluded from the stop condition.
  They still count in A(r) (step 4), as pre-registered.
- 0.003 gets the slow-rate extension: all 20 seeds re-run to 400 rounds
  (collapse and not-learned judged over all 400 rounds).
- If 0.003 then has zero collapsed seeds and every seed except the hard
  splits learns, choose 0.003, with R* from the 400-round curves (A_end over
  rounds 301-400). Otherwise choose 0.01 and disclose its single one-round
  dip (seed 16), with R* from its 200-round curves as in step 4.

**Result (2026-10-12): lr 0.003, R\* = 243 rounds.** At 400 rounds, 0.003
has no collapsed seed and every seed except the hard splits (13, 19)
learns (seed 15 now reaches 0.60). Plateau rule on the 20-seed mean
curve: A_end = 0.6925 (rounds 301-400), R* = 243 (A(243) = 0.6918). Mean
validation C-index: 0.651 at round 50, 0.676 at 100, 0.690 at 200, 0.692
at 400. Check: the first 200 rounds of every 400-round run equal the
200-round runs exactly (`results_tables/lr_tcga.json`).
Disclosure: seed 15 learns late. It first reaches 0.60 at round 286 (best
0.607, final 0.598), so at the chosen 243 rounds it is still below 0.60.

**Preset (2026-10-13):** `bench/tasks.py` uses lr 0.003 and 243 rounds for
TCGA. `reproduce_flamby_tcga.py` pins FLamby's lr 0.1 and 5 rounds (its
tables are byte-identical after the change).

### Grid (decided 2026-10-13, before any grid run)

`grid_tcga.sh <cc_tau> <i>/<N>` (one shard; run 8, i = 0..7, 1 torch thread
each). Seeds only average out training randomness; the test sets are fixed,
so the noise from few deaths per region doesn't shrink with more seeds.
Seeds are therefore cut where they help least:
- 20 seeds: clean runs (f_assumed = 0, 1, 2) and real attacks at f = 1
  (sign flip, Gaussian, IPM, `alie_z1`; every region as the attacker).
- 5 seeds: every attack at f = 2 (all 15 attacker pairs, so 75 runs per
  configuration) and Baruch's ALIE at f = 1 (z = 0 at n = 6: degenerate, it
  only needs to show it matches clean runs).
- 15,120 grid points, 11,690 that train (the rest are invalid and finish at
  once). About 25 h with 8 processes (`scaling_tcga.py`: 8 processes = 4.9×
  one process).
- CC tau from rule v2.1 on TCGA's validation split (below, "Tuning CC's tau").
- **Stopping and restarting:** `pkill -f grid_tcga.sh; pkill -f sweep.py`
  (wrappers first, so they can't start their next sweep), then rerun the
  same 8 commands. Each run's result is written once, at the end, and
  atomically (temporary file, then rename), so a killed run leaves no
  `.json` and is simply redone; finished runs are skipped.

### Metrics (pre-registered 2026-10-11, before any grid run)

- **Main average:** the stratified (within-region) C-index: the comparable
  pairs of all regions, each pair inside one region, pooled into one
  C-index (608 test pairs). This equals the per-region C-indices averaged
  with weights = comparable pairs. FLamby's pooled-test C-index, which also
  compares patients across regions, is used only in the reproduction table.
  *Correction (2026-10-11, before any TCGA result):* the first version of
  this pre-registration (commit 90341e4) said "weighted by test patients"
  (heart's convention). That contradicted the decision already taken: with
  patient weights, Canada's 3 pairs would pull the average as if they were
  solid evidence (11 patients = 5% weight from 0.5% of the pairs).
- **Worst region:** only regions with **at least 3 test deaths** count:
  Northeast (14), West (8), South (4), Midwest (3). Europe (2) and Canada
  (1) are excluded. Reason: every comparable pair contains a death, and the
  pairs that share a death flip together when that patient's risk score
  moves, so the number of deaths, not pairs, is the real amount of evidence.
  Canada's C-index rests on 3 pairs and one death. Fixed after seeing each
  region's death and pair counts, before any results.
- **Gap** (best minus worst region): over the same 4 regions with at least
  3 test deaths, for the same reason; a noisy region is as likely to be the
  best as the worst (Canada scores 1.0 on 4 of FLamby's 5 seeds). Decided
  2026-10-12, before any TCGA result.
- All 6 regions still count in the average and each gets its own column in
  the appendix tables.

### Analysis (grid finished 2026-10-15)

`python analyze_tcga.py` (about 10 s) writes `tcga_clean.csv`,
`tcga_attacker_f1*.csv`, `tcga_attacker_f2*.csv` (the 15 pairs),
`tcga_exclusion*_fa{1,2}.csv` and `fig_tcga_avg_vs_worst_fa{0,1,2}` to
`results_tables/`. `python summarize.py results_tcga` gives the pooled
summary. "Mean of real attacks": f = 1: sign flip, Gaussian, IPM,
`alie_z1`; f = 2: the same plus Baruch's ALIE (z = 0.43 there).
Caveat:
FLamby's test split is easier than its training data (a plain Cox model
scores 0.84 on it vs 0.72 in cross-validation on the training set), so
absolute C-indices are high. Comparisons between rules are unaffected.

## Fed-ISIC2019 (FLamby, natural splits)

Data terms (accept both before downloading): ISIC 2019, CC BY-NC 4.0
(non-commercial, credit required),
https://challenge.isic-archive.com/data/#2019, and HAM10000 (4 of the 6
centres),
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T&version=3.0&selectTab=termsTab.
Both must be credited in the paper.

1. `sh setup_flamby.sh` installs the ISIC packages (exact versions, see the
   script) and patches FLamby's download script to use `curl` with
   certificate checks instead of `wget --no-check-certificate`.
2. `sh download_isic.sh` downloads the images zip (9.8 GB) and the two CSVs
   to `data/flamby/fed_isic2019/` and verifies them: CSV MD5s, exact zip
   size, CRC-32 of every file in the zip, 25,331 images.
3. FLamby's `download_isic.py` (command printed by step 2; asks for the
   licence once) unzips, deletes the zip, drops the 2,084 images with no
   centre and must report 23,247 images ("Download OK").
4. FLamby's `resize_images.py` (shorter side 224 px, colour constancy).
   CPU-heavy (32 parallel jobs), so not while a sweep runs. Took 26 s here
   (2026-10-13); 23,247 preprocessed images, all readable, 188 MB.
5. ImageNet weights for EfficientNet-b0 (`efficientnet-pytorch` downloads
   `efficientnet-b0-355c32eb.pth` from its GitHub release on first use,
   21,388,428 bytes; its SHA-256 starts with 355c32eb, as the file name
   says).
6. `python extract_isic_features.py` caches the frozen-backbone features
   (main setting, below) in `data/flamby/fed_isic2019/features_effb0_centercrop.pt`
   (120 MB). 11 min on the Mac (2026-10-15), cause not established; a one-off step. 18,597 train and 4,650 test images in
   FLamby's per-centre split; MPS features match CPU features to 1e-5.

### Main setting and reproduction (decided 2026-10-13)

- **Main setting: frozen backbone.** ImageNet-pretrained EfficientNet-b0
  (FLamby's model), frozen; features cached once per image, and only the
  final layer is trained by federated learning. Features come from the
  **centre crop only** (FLamby's test transform: centre crop 200 px,
  normalise), with no training augmentation. Reason: the paper compares
  aggregation rules on identical features, not the best absolute accuracy.
  This is a limitation: accuracy will be below a fine-tuned, augmented model.
  Our frozen-backbone numbers are reported as a new baseline; FLamby
  published none for this setting.
- **Reproduction:** `reproduce_flamby_isic.py`, FLamby's own FedAvg code
  (full fine-tuning, augmentation, SGD lr 0.01, batch 64, 100 local steps,
  9 rounds; `config_isic2019.json`), seeds 42 and 43, on the Mac's GPU
  (about 1.8 h per seed). *Correction (2026-10-16, before any run):* this
  line first said Adam lr 5e-4. That is FLamby's setting for its pooled
  and local baselines; its FedAvg uses SGD lr 0.01 (config and published
  results CSVs). Check, as for TCGA: our
  mean pooled balanced accuracy within one published std of FLamby's,
  i.e. within 0.534 ± 0.114 = **0.420-0.648** (FLamby, seeds 42-46; four
  seeds at 0.57-0.59, seed 46 at 0.33). Only change to FLamby's code:
  the models run on MPS instead of CUDA/CPU. `opacus` and `tensorboard`
  (imported by FLamby's strategies, used only for differential privacy
  and logging, both off) are not installed; placeholders that raise if
  called stand in for them.
  One seed per process, appended to the log
  (`nohup caffeinate -is .venv/bin/python reproduce_flamby_isic.py --seed 43
  >> results_tables/flamby_isic_repro.log 2>&1 &`). PyTorch's GPU memory is
  capped (`--mps_fraction`, default 0.8 of the recommended maximum) and
  memory is printed every round, after seed 43 filled the 16 GB Mac
  (2026-10-16).
  **Result (2026-10-17): passed.** Mean pooled balanced accuracy 0.615
  (seeds 42, 43: 0.639, 0.591), inside 0.420-0.648; every centre within
  one published std (`results_tables/flamby_isic_repro.csv`; 162 and 187
  min per seed).

### Main-setting rules (pre-registered 2026-10-16, before any run)

Adapter: `bench/tasks.py` (`fed_isic2019`): one client per centre, a
linear layer (1280 -> 8, FLamby's final layer) on the cached features,
FLamby's weighted focal loss, CPU. Every evaluation also records, per
centre, the correct predictions per class (`client_class_correct`), so the
worst-centre metric below is computed from stored counts.

1. **Local optimizer: SGD**, as for heart and TCGA, not FLamby's Adam.
   Reason: Adam rescales each coordinate of the update, which changes what
   CC clips and what Krum picks; SGD keeps the rules' behaviour comparable
   across the three datasets. The reproduction covers comparability with
   FLamby. Batch 64 and 100 local steps (FLamby).
2. **Learning rate:** a screen like TCGA's, on a validation split (10% of
   each centre's training images, stratified by class; 1,859 images; SE of
   balanced accuracy about 0.025). Fixed on 2026-10-16, before
   the screen ran (`select_lr_isic.py`):
   - Clean sample-weighted FedAvg, rates {0.3, 0.1, 0.03, 0.01, 0.003},
     seeds 0-19, 200 rounds, evaluated every round. B(r) = pooled
     validation balanced accuracy (chance 0.125).
   - *Divergence:* non-finite validation loss or balanced accuracy.
   - *Learned:* B(r) >= 0.30 at some round (about 7 SE above chance,
     roughly halfway to a central linear probe's 0.56).
   - *Collapse:* divergence, or learned and then B(r') < 0.20 later (within
     3 SE of chance; a model predicting one class scores 0.125). An
     absolute rule, as TCGA's; the relative alternative (below half its
     own best) was not chosen.
   - *Not learned:* B(r) never reaches 0.30; reported, not a collapse.
   - *Choice:* the largest rate with zero collapsed seeds. Stop if it has
     any not-learned seed, or no rate qualifies.
3. **Rounds:** the heart plateau rule on the chosen rate's 20 validation
   curves: R* = the first round after which the mean curve A(r) stays
   within 0.005 of A_end (the mean of A over rounds 151-200). If R* > 150,
   the chosen rate is re-run to 400 rounds (the first 200 must match),
   classified again over all 400 rounds (stop on any collapse or
   not-learned seed), and R* is taken with A_end over rounds 301-400; if
   that R* > 300, stop.
4. **Seeds:** 20 everywhere if the runtime allows (measured: about 90 ms
   per round on one CPU thread).
5. **Metrics.** Main average: **one balanced accuracy over all centres'
   test images pooled** (FLamby's pooled test set; ISIC has no per-centre
   normalisation, and per-centre balanced accuracy averages over different
   class sets). Worst centre: each centre's balanced accuracy over only
   the classes with **at least k test images** at that centre; a centre
   counts only if it has at least 2 such classes. **k = 5** (fixed on 2026-10-16 from the count table below, before any result):
   all 6 centres count, with 8, 4, 6, 7, 3 and 2 classes (centre 5, the
   smallest: NV and MEL). Appendix sensitivity table with **k = 10**:
   classes 8, 2, 6, 7, 3; centre 5 drops out (only NV has 10 images).
6. **CC tau:** rule v2.1, unchanged.

**Running the screen:** `python select_lr_isic.py` (8 processes, one torch
thread each; about 6 min on an otherwise idle Mac, about 2 min more if the
400-round re-run is needed; resumes from `results_tune/lr_isic/`). Writes
`results_tables/lr_isic.json`. Not while a GPU job runs: with the
reproduction running, one round took 0.4 s instead of 0.09 s.

Test images per centre and class (FLamby's split):

| Centre | MEL | NV | BCC | AK | BKL | DF | VASC | SCC | Total |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 578 | 843 | 564 | 131 | 215 | 27 | 24 | 101 | 2,483 |
| 1 | 4 | 743 | 0 | 0 | 28 | 7 | 9 | 0 | 791 |
| 2 | 147 | 362 | 37 | 1 | 96 | 11 | 18 | 0 | 672 |
| 3 | 66 | 154 | 73 | 23 | 84 | 10 | 0 | 42 | 452 |
| 4 | 49 | 66 | 0 | 0 | 49 | 0 | 0 | 0 | 164 |
| 5 | 9 | 74 | 0 | 0 | 4 | 0 | 1 | 0 | 88 |

**Duplicates in the data (disclosed; split kept as published, decided 2026-10-16):** 49 pairs of images are byte-identical under different
ISIC IDs (original and preprocessed files; `python isic_duplicates.py`
writes `results_tables/isic_duplicates.csv`). All pairs are within one
centre: 28 inside a training set, 4 inside a test set, and **17 straddle
FLamby's train/test split**, so 17 of 4,650 test images (0.4%) have an
exact copy in training (`results_tables/isic_leaked_test.csv`):

| Centre | MEL | NV | BCC | AK | BKL | Total |
|---|---|---|---|---|---|---|
| 0 | 3 | 4 | 3 | 2 | 4 | 16 (of 2,483) |
| 5 | 0 | 1 | 0 | 0 | 0 | 1 (of 88) |

Two pairs carry different labels: one straddling the split (test NV,
training copy MEL, centre 0) and one inside centre 0's test set (NV and
MEL). Inside training sets, 1-10 pairs per seed straddle the screen's
train/validation split (seeds 0-19; validation has 1,859 images).
**Planned sensitivity check:** the clean runs (every rule, both
weightings, f_assumed 0/1/2, 20 seeds, R* rounds) repeated with the 17
images removed from the test set; pooled balanced accuracy and the worst
centre (k = 5) are reported against the main clean results.

Disk: 9.8 GB download, about 19.5 GB at peak while unzipping, about 9 GB
of original images afterwards, plus the preprocessed images and cached
features (about 1 GB, estimated).

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
- Fed-ISIC2019: frozen backbone, centre-crop features only (no
  augmentation), so absolute accuracy is below FLamby's fine-tuned model.
- Add FLTrust (needs a small server dataset) and a text dataset.
