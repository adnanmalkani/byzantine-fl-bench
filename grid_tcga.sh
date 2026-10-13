#!/bin/sh
# Fed-TCGA-BRCA main grid (2026-10-13: option D, 20-seed clean
# f = 2 baseline, alie_z1). One shard of N; run N copies with i = 0..N-1:
#   sh grid_tcga.sh <cc_tau> <i>/<N> [--dry_run]
# 20 seeds: clean runs (f_assumed 0, 1, 2) and real attacks at f = 1.
# 5 seeds:  ALIE at f = 1 (Baruch's z = 0 at n = 6: degenerate) and every
#           attack at f = 2 (all 15 attacker pairs).
# Each sweep skips finished runs, so a shard can be stopped and restarted.
# Weighted and unweighted; label flip doesn't apply to survival data.
set -e
cd "$(dirname "$0")"
TAU=$1; SHARD=$2; shift 2
S20=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
S5=0,1,2,3,4
sweep() {
  OMP_NUM_THREADS=1 .venv/bin/python sweep.py --datasets fed_tcga_brca --weighted 0,1 \
    --cc_tau "$TAU" --shard "$SHARD" --out results_tcga "$@"
}
sweep --fs 0,1,2 --attacks none --seeds $S20 "$@"
sweep --fs 1 --attacks sign_flip,gaussian,ipm,alie_z1 --seeds $S20 "$@"
sweep --fs 1 --attacks alie --seeds $S5 "$@"
sweep --fs 2 --attacks sign_flip,gaussian,alie,alie_z1,ipm --seeds $S5 "$@"
