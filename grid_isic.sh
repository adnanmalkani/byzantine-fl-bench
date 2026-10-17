#!/bin/sh
# Fed-ISIC2019 main grid (2026-10-17: 20 seeds everywhere except
# Baruch's ALIE at f = 1, which is degenerate at n = 6 (z = 0): 5 seeds, as
# for TCGA). One shard of N; run N copies with i = 0..N-1:
#   sh grid_isic.sh main   <cc_tau> <i>/<N> [--dry_run]   main grid -> results_isic
#   sh grid_isic.sh noleak <cc_tau> <i>/<N> [--dry_run]   duplicates check: clean
#        runs with the 17 leaked test images dropped -> results_isic_noleak
# lr 0.01 and 101 rounds come from the preset (bench/tasks.py, Amendment 2).
# Each sweep skips finished runs and writes each result atomically, so a shard
# can be stopped and restarted. Weighted and unweighted.
set -e
cd "$(dirname "$0")"
MODE=$1; TAU=$2; SHARD=$3; shift 3
S20=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
S5=0,1,2,3,4
sweep() {
  OMP_NUM_THREADS=1 .venv/bin/python sweep.py --datasets fed_isic2019 --weighted 0,1 \
    --cc_tau "$TAU" --shard "$SHARD" "$@"
}
case "$MODE" in
  main)
    sweep --out results_isic --fs 0,1,2 --attacks none --seeds $S20 "$@"
    sweep --out results_isic --fs 1 --attacks label_flip,sign_flip,gaussian,ipm,alie_z1 --seeds $S20 "$@"
    sweep --out results_isic --fs 1 --attacks alie --seeds $S5 "$@"
    sweep --out results_isic --fs 2 --attacks label_flip,sign_flip,gaussian,alie,alie_z1,ipm --seeds $S20 "$@"
    ;;
  noleak)
    sweep --out results_isic_noleak --drop_leaked --fs 0,1,2 --attacks none --seeds $S20 "$@"
    ;;
  *) echo "usage: sh grid_isic.sh main|noleak <cc_tau> <i>/<N> [--dry_run]"; exit 1 ;;
esac
