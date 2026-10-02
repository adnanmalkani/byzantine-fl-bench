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

SITE=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo "$PWD/third_party/FLamby" > "$SITE/flamby.pth"

# Imports of flamby.utils + fed_heart_disease (+ lifelines for fed_tcga_brca)
.venv/bin/pip install -q scikit-learn matplotlib seaborn pyyaml tqdm wget lifelines
.venv/bin/python -c "import flamby.datasets.fed_heart_disease as h; print('flamby OK, heart clients:', h.NUM_CLIENTS)"
