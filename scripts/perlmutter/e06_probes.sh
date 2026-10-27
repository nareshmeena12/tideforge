#!/bin/bash
# E0.6: the probe suite P1-P10 on ft3_main (starting values).
#
#   bash $SCRATCH/tideforge/repo/scripts/perlmutter/e06_probes.sh build      # login node, CPU, once: $TF_DATA/probes/P*/v*/
#   salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 1 --account=$NERSC_ACCOUNT
#   bash $SCRATCH/tideforge/repo/scripts/perlmutter/e06_probes.sh run        # 1 GPU: all probes (groups on + hidden where a
#                                                                            # cell has both), then P6 under each NaN variant
#   bash $SCRATCH/tideforge/repo/scripts/perlmutter/e06_probes.sh report
#   (add --dry-run to print the commands)
# build prints one registry line per probe (seed, size, sha256, fingerprint): paste them back; they go into
# tideforge/probes/registry.json, and from then on Perlmutter's files are the source of truth (load() checks the sha256).
# Outputs: $TF_ROOT/probes/results.csv (+ series/ for --compare). Estimated run time: well under 15 min (estimate).
set -u
source "$SCRATCH/tideforge/repo/scripts/perlmutter/env.sh" || exit 2
CMD="${1:-}"
[ $# -gt 0 ] && shift
DRY=0
for a in "$@"; do case "$a" in
    --dry-run) DRY=1;;
    *) echo "unknown option $a"; exit 2;;
esac; done
tf_require TF_DATA FT3_MAIN_CKPT || exit 2
SETS="$TF_DATA/probes"
OUT="$TF_ROOT/probes"
LOG="$TF_ROOT/logs/e06_${CMD}.log"
mkdir -p "$TF_ROOT/logs"
run() {
    if [ "$DRY" -eq 1 ]; then echo "DRY RUN: $*"; return 0; fi
    "$@" 2>&1 | tee -a "$LOG"
    return "${PIPESTATUS[0]}"
}
case "$CMD" in
build)
    git -C "$TF_REPO" log --oneline -1 | tee -a "$LOG"
    run python -m tideforge.probes build --root "$SETS" --probes all
    ;;
run)
    [ "$DRY" -eq 1 ] || [ -f "$FT3_MAIN_CKPT" ] || { echo "missing checkpoint $FT3_MAIN_CKPT"; exit 2; }
    git -C "$TF_REPO" log --oneline -1 | tee -a "$LOG"
    run python -m tideforge.probes run --root "$SETS" --out "$OUT" --model ft3main --ckpt "$FT3_MAIN_CKPT" --probes all \
        --nan-variant baseline || exit 1
    for v in nofill_delete nofill_mask; do
        run python -m tideforge.probes run --root "$SETS" --out "$OUT" --model ft3main --ckpt "$FT3_MAIN_CKPT" --probes P6 \
            --nan-variant "$v" || exit 1
    done
    run python -m tideforge.probes run --root "$SETS" --out "$OUT" --model naive --probes all --device cpu || exit 1
    run python -m tideforge.probes run --root "$SETS" --out "$OUT" --model oracle --probes all --device cpu || exit 1
    echo "E0.6 RUN DONE"
    ;;
report)
    run python -m tideforge.probes report --out "$OUT" --model ft3main
    run python -m tideforge.probes report --out "$OUT" --compare ft3main:baseline naive:baseline
    ;;
*) echo "usage: e06_probes.sh build|run|report [--dry-run]"; exit 2;;
esac
