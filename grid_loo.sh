#!/bin/sh
# Leave-out controls (2026-10-18): clean
# runs with one client (f = 1) or two clients (f = 2) left out of training, each
# rule variant, both weightings, the attacked runs' seeds. All n test sets are
# evaluated. One shard of N; run N copies with i = 0..N-1:
#   sh grid_loo.sh <i>/<N> <mode> [<mode> ...] [--dry_run]
# Modes (run in the order given):
#   heart       one out, f_assumed 1, 20 seeds        -> results_heart_loo
#   isic        one out, f_assumed 1, 20 seeds        -> results_isic_loo
#   tcga        one out, f_assumed 1, 20 seeds        -> results_tcga_loo
#   isic_pair   two out, f_assumed 2, 20 seeds, FedAvg/median/geomed/CC
#   tcga_pair   two out, f_assumed 2, 5 seeds (as TCGA's f = 2 grid), same rules
# Rules invalid at n - 1 finish at once (e.g. Krum at heart's n = 3); in
# controls the count check is skipped for rules that ignore f (bench/fl.py,
# F_FREE). Each sweep skips finished runs; results are written atomically.
set -e
cd "$(dirname "$0")"
SHARD=$1; shift
MODES=""
while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do MODES="$MODES $1"; shift; done
S20=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
S5=0,1,2,3,4
PAIR_RULES=fedavg,median,geomed,cc
sweep() {
  OMP_NUM_THREADS=1 .venv/bin/python sweep.py --attacks none --weighted 0,1 \
    --shard "$SHARD" "$@"
}
for MODE in $MODES; do
  case "$MODE" in
    heart) sweep --datasets fed_heart_disease --cc_tau 0.03 --fs 1 --leave_out 1 \
             --seeds $S20 --out results_heart_loo "$@" ;;
    isic) sweep --datasets fed_isic2019 --cc_tau 1.0 --fs 1 --leave_out 1 \
            --seeds $S20 --out results_isic_loo "$@" ;;
    tcga) sweep --datasets fed_tcga_brca --cc_tau 0.03 --fs 1 --leave_out 1 \
            --seeds $S20 --out results_tcga_loo "$@" ;;
    isic_pair) sweep --datasets fed_isic2019 --cc_tau 1.0 --fs 2 --leave_out 2 \
                 --aggregators $PAIR_RULES --seeds $S20 --out results_isic_loo "$@" ;;
    tcga_pair) sweep --datasets fed_tcga_brca --cc_tau 0.03 --fs 2 --leave_out 2 \
                 --aggregators $PAIR_RULES --seeds $S5 --out results_tcga_loo "$@" ;;
    *) echo "unknown mode: $MODE (heart isic tcga isic_pair tcga_pair)"; exit 1 ;;
  esac
done
