#!/bin/sh
# Fed-Heart-Disease main grid (20 seeds; clean f_assumed 0, 1, 2; every
# attack at f = 1 and f = 2 with enumerated attackers; weighted and
# unweighted; CC tau 0.03). Reconstructed on 2026-10-19 from the stored
# results (the grid was launched by hand on 2026-10-04, alie_z1 added
# 2026-10-14); a dry run against results_heart finds all 20,160 runs.
# One shard of N; run N copies with i = 0..N-1:
#   sh grid_heart.sh <i>/<N> [--dry_run]
# Each sweep skips finished runs; results are written atomically.
set -e
cd "$(dirname "$0")"
SHARD=$1; shift
OMP_NUM_THREADS=1 .venv/bin/python sweep.py --datasets fed_heart_disease --weighted 0,1 \
  --fs 0,1,2 --attacks none,label_flip,sign_flip,gaussian,alie,ipm,alie_z1 \
  --cc_tau 0.03 --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
  --shard "$SHARD" --out results_heart "$@"
