#!/bin/sh
# Make FLamby importable in .venv WITHOUT running its setup.py.
# Why: FLamby's setup.py (install, develop and egg_info hooks) shells out to
# `pip install large-image-source-openslide --find-links <external site>`,
# which we don't need. We clone a pinned commit, add it to the venv's path
# with a .pth file, and install only what the datasets we use import.
set -e
cd "$(dirname "$0")"
FLAMBY_COMMIT=edacf54d5211520583b0133d55ac39b6fda8324b  # main, 2024-06-19

if [ ! -d third_party/FLamby ]; then
  git clone -q https://github.com/owkin/FLamby.git third_party/FLamby
fi
git -C third_party/FLamby checkout -q "$FLAMBY_COMMIT"

# numpy >= 2 removed the np.NaN alias; FLamby's heart loader still uses it.
# Same value as np.nan, so this patch does not change the data.
sed -i.bak 's/np\.NaN/np.nan/' third_party/FLamby/flamby/datasets/fed_heart_disease/dataset.py
rm -f third_party/FLamby/flamby/datasets/fed_heart_disease/dataset.py.bak

# FLamby's ISIC download script fetches with `wget --no-check-certificate`:
# wget isn't installed on macOS by default, and that flag turns off HTTPS
# certificate checks. Use curl (certificate checks on) instead, and skip a
# file that is already there (download_isic.sh downloads and verifies the
# files first, so FLamby's script uses the verified copies).
sed -i.bak 's/os\.system(f"wget {\(url_[0-9]\)} --no-check-certificate -O {\(dest_file_[0-9]\)}")/os.system(f"[ -s {\2} ] || curl -fL --retry 3 {\1} -o {\2}")/' \
  third_party/FLamby/flamby/datasets/fed_isic2019/dataset_creation_scripts/download_isic.py
rm -f third_party/FLamby/flamby/datasets/fed_isic2019/dataset_creation_scripts/download_isic.py.bak

SITE=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo "$PWD/third_party/FLamby" > "$SITE/flamby.pth"

# Imports of flamby.utils + fed_heart_disease (+ lifelines for fed_tcga_brca).
# pandas < 3: FLamby's TCGA loader reads rows by position with Series[int],
# which pandas 3 treats as a label (see requirements.txt).
.venv/bin/pip install -q scikit-learn matplotlib seaborn pyyaml tqdm wget lifelines "pandas<3"
.venv/bin/python -c "import flamby.datasets.fed_heart_disease as h; print('flamby OK, heart clients:', h.NUM_CLIENTS)"

# fed_isic2019: opencv (colour constancy in preprocessing), albumentations and
# efficientnet-pytorch (FLamby's ISIC dataset and model). Exact, tested
# versions: FLamby's code uses albumentations.Flip and always_apply, which
# newer releases removed (1.4.24 has no Flip), and albumentations 1.4.10
# requires numpy < 2. Installing without pins downgraded numpy, so numpy is
# pinned to the version all results were produced with.
.venv/bin/pip install -q "albumentations==1.4.18" "opencv-python-headless==4.11.0.86" \
  "efficientnet-pytorch==0.7.1" "numpy==$(.venv/bin/python -c 'import numpy; print(numpy.__version__)')"
.venv/bin/python -c "import flamby.datasets.fed_isic2019 as i; print('flamby OK, isic clients:', i.NUM_CLIENTS)"
