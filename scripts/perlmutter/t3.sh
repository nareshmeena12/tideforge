#!/bin/bash
# T3 on an interactive GPU node (1 GPU). Order (owner 2026-10-29): gputest, smoke, e09, then sbatch t3_e07.sbatch.
#
#   salloc --nodes 1 --qos interactive --time 02:00:00 --constraint gpu --gpus 1 --account=$NERSC_ACCOUNT
#   bash $SCRATCH/tideforge/repo/scripts/perlmutter/t3.sh gputest                 # GPU tests of T3 (+ the generator parity ones)
#   bash $SCRATCH/tideforge/repo/scripts/perlmutter/t3.sh smoke E0.7 G4.1 G5.2a G1.2   # ~0.5B points per config, 1 GPU
#   bash $SCRATCH/tideforge/repo/scripts/perlmutter/t3.sh e09                     # E0.9: 300 steps, generator det vs default
#   (add --dry-run as the last argument to print the commands)
# Smoke runs go to $TF_ROOT/smoke (allowed from a tree with local changes, no finish step); e09 to $TF_ROOT/e09.
set -u
source "$SCRATCH/tideforge/repo/scripts/perlmutter/env.sh" || exit 2
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
CMD="${1:-}"; [ $# -gt 0 ] && shift
DRY=0; ARGS=()
for a in "$@"; do [ "$a" = "--dry-run" ] && DRY=1 || ARGS+=("$a"); done
run() { if [ "$DRY" -eq 1 ]; then echo "DRY RUN: $*"; else echo "$(date) $*"; "$@"; fi; }
cd "$TF_REPO" || exit 2
git log --oneline -1
case "$CMD" in
gputest)
    run python -m pytest -q -s -rA -m gpu tests/train/test_train_gpu.py tests/test_gen_parity.py
    ;;
smoke)
    [ "${#ARGS[@]}" -gt 0 ] || { echo "usage: t3.sh smoke CONFIG [CONFIG ...]"; exit 2; }
    mkdir -p "$TF_ROOT/smoke" "$TF_ROOT/logs"
    for e in "${ARGS[@]}"; do
        run python -m tideforge.train.trainer --exp "$e" --seed 0 --out "$TF_ROOT/smoke" --smoke-points 5e8 --allow-dirty \
            --no-finish 2>&1 | tee "$TF_ROOT/logs/t3_smoke_$e.log"
        st=${PIPESTATUS[0]}                       # read at once: any later command (even a test) replaces PIPESTATUS
        [ "$DRY" -eq 1 ] || [ "$st" -eq 0 ] || { echo "SMOKE FAILED: $e (exit $st)"; exit 1; }
    done
    echo "SMOKE DONE: ${ARGS[*]}"
    ;;
e09)
    mkdir -p "$TF_ROOT/e09" "$TF_ROOT/logs"
    for m in det default; do
        run python -m tideforge.train.trainer --exp E0.9 --seed 0 --out "$TF_ROOT/e09" --gen-mode "$m" --no-finish 2>&1 \
            | tee "$TF_ROOT/logs/t3_e09_$m.log"
        st=${PIPESTATUS[0]}
        [ "$DRY" -eq 1 ] || [ "$st" -eq 0 ] || { echo "E0.9 FAILED ($m, exit $st)"; exit 1; }
    done
    D=$(ls -d "$TF_ROOT"/e09/E0.9-*-det 2>/dev/null | tail -1); N=$(ls -d "$TF_ROOT"/e09/E0.9-*-default 2>/dev/null | tail -1)
    run python -m tideforge.train.throughput report --det "$D" --default "$N"
    ;;
*) echo "usage: t3.sh gputest|smoke CONFIG...|e09 [--dry-run]"; exit 2;;
esac
