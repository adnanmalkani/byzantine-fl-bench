# byzantine-fl-bench

A reproducible benchmark of Byzantine-robust aggregation rules for federated
learning when only a few hospitals collaborate, on the natural (real
hospital) splits of [FLamby](https://github.com/owkin/FLamby):
Fed-Heart-Disease (4 hospitals), Fed-TCGA-BRCA (6 regions) and
Fed-ISIC2019 (6 centres).

- **Rules (8),** each unweighted and weighted by training-set size: FedAvg,
  coordinate-wise median, trimmed mean, Krum, Multi-Krum, Bulyan,
  geometric median, Centered Clipping.
- **Attacks:** label flip, sign flip, Gaussian noise, IPM, ALIE (Baruch et
  al.'s z, and a fixed z = 1). Every attacker identity (f = 1) and every
  pair (f = 2) is run, not a random draw.
- **Validity:** a run whose rule's assumptions fail (by client count, e.g.
  Bulyan needs n ≥ 4f + 3) is recorded as `invalid`, never run silently.
  `validity_table.py --weights` also checks validity by data share.
- **Per-client metrics:** every run logs each client's test metric and
  each rule's per-client diagnostics (who was selected, trimmed, clipped).
- **Leave-out controls:** clean runs without the attacking client(s), so a
  drop under attack can be split into lost data and attack damage.

The full protocol, with every setting and pre-specified rule and the date
it was fixed, is in [docs/PROTOCOL.md](docs/PROTOCOL.md). Paper figures and
tables: [paper/](paper/README.md).

**Paper:** _Who Pays for Byzantine Robustness? Validity, Data-Share
Dictatorships, and Fairness When Only a Few Hospitals Collaborate._
arXiv preprint forthcoming.

## Install

Tested on macOS (Apple M4, 16 GB), Python 3.11, PyTorch 2.x; the FLamby
experiments run on CPU.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sh setup_flamby.sh        # FLamby at a pinned commit, without its setup.py (see the script)
python test_aggregators.py
```

`setup_flamby.sh` clones FLamby into `third_party/FLamby` at a fixed
commit, makes it importable with a `.pth` file and installs only what the
three datasets need, with exact versions where they matter. It applies two
small patches, both explained in the script (a NumPy 2 alias in the heart
loader; `curl` with certificate checks instead of `wget
--no-check-certificate` in the ISIC download).

## Data (not included)

No dataset is distributed with this repository. Each one has its own
terms, which you must accept yourself.

- **Fed-Heart-Disease** (UCI Heart Disease; FLamby asks you to accept the
  licence):
  ```bash
  python third_party/FLamby/flamby/datasets/fed_heart_disease/dataset_creation_scripts/download.py \
    --output-folder data/flamby/fed_heart_disease
  ```
- **Fed-TCGA-BRCA** ships with FLamby. On first use FLamby asks you,
  once, to accept the [GDC data terms](https://gdc.cancer.gov/access-data/data-access-processes-and-tools).
- **Fed-ISIC2019** (about 10 GB download; accept the
  [ISIC 2019](https://challenge.isic-archive.com/data/#2019) terms,
  CC BY-NC 4.0, and the
  [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
  terms first):
  ```bash
  sh download_isic.sh                 # download + verify (MD5s, size, CRC of every image)
  # then the FLamby command it prints (unzips, assigns centres; 23,247 images), then:
  python third_party/FLamby/flamby/datasets/fed_isic2019/dataset_creation_scripts/resize_images.py
  python extract_isic_features.py     # frozen EfficientNet-B0 features, 120 MB
  ```
  The main setting trains only the final layer on cached features of a
  frozen ImageNet backbone (see the protocol). FLamby's split contains 17
  test images with an exact duplicate in training; `isic_duplicates.py`
  lists them and `grid_isic.sh noleak` measures their effect (≤ 0.2
  points).
- **MNIST** (the sanity check: 20 synthetic clients) is downloaded by torchvision.

Everything goes under `data/` (git-ignored).

## Reproduce

All grids are resumable (finished runs are skipped; results are written
atomically) and can be split over processes with `<i>/<N>`. Times are for
8 processes on the machine above.

```bash
# Checks
python test_aggregators.py && python test_validity.py && python test_loo.py   # and the other test_*.py

# FLamby reproductions (FedAvg, FLamby's own settings)
python reproduce_flamby_heart.py
python reproduce_flamby_tcga.py
python reproduce_flamby_isic.py --seed 42        # FLamby's own code; about 3 h per seed on a GPU

# Main grids (8 processes: run i = 0..7)
sh grid_heart.sh $i/8                  # 20,160 runs, about 1-2 h
sh grid_tcga.sh 0.03 $i/8              # 15,120 runs, about 19 h
sh grid_isic.sh main 1.0 $i/8          # 39,840 runs, about 15 h
sh grid_isic.sh noleak 1.0 $i/8        # duplicates check
sh grid_loo.sh $i/8 heart isic tcga isic_pair tcga_pair   # leave-out controls, about 5.5 h

# Analysis -> results_tables/
python summarize.py results_heart      # (and results_tcga, results_isic)
python analyze_heart.py; python analyze_tcga.py; python analyze_isic.py
python analyze_loo.py
python validity_table.py && python validity_table.py --weights
python compare_isic_noleak.py

# Paper figures and tables -> paper/
python make_paper.py
```

The settings that were chosen on validation data before the grids (CC's
clipping radius, learning rate and number of rounds per dataset) can be
re-derived with `tune_cc.py`, `select_lr_tcga.py`, `select_lr_isic.py` and
`convergence_heart.py`; the rules they follow are in the protocol.

## Layout

| Path | What |
|---|---|
| `bench/aggregators.py` | the 8 rules (weighted and unweighted), diagnostics, `check_valid` |
| `bench/attacks.py` | the attacks |
| `bench/fl.py` | the simulation loop (`Config`, `run`) |
| `bench/tasks.py` | datasets, models and per-dataset presets |
| `run.py`, `sweep.py` | one run; a resumable grid |
| `grid_*.sh` | the exact grids behind the paper |
| `summarize.py`, `analyze_*.py`, `validity_table.py` | metrics, tables, diagnostics |
| `make_paper.py`, `paper/` | paper figures and tables |
| `docs/PROTOCOL.md` | settings, pre-specified rules, amendments |
| `test_*.py` | tests (run directly with Python) |

## Licence and credits

Code: MIT licence (see [LICENSE](LICENSE)). The datasets are not covered by
it and keep their own terms. FLamby (MIT) is used as an external
dependency, not redistributed. If you use the ISIC data, credit ISIC 2019
and HAM10000 as their terms require.

## Citation

_To follow with the arXiv version._
