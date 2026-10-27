#!/bin/bash
# E0.3: (a) NaN audit of the inference contexts (no model); (b) score ft3_main with the variants nofill_delete and nofill_mask,
# groups ON and OFF, next to the E0.2 baseline in the same evaluation root.
#
#   salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 2 --account=$NERSC_ACCOUNT
#   bash $SCRATCH/tideforge/repo/scripts/perlmutter/e03_nan.sh audit [--smoke]      # CPU only; writes R/eval/nan_audit.csv
#   bash $SCRATCH/tideforge/repo/scripts/perlmutter/e03_nan.sh score [--smoke]      # 2 GPUs: one variant per GPU, both modes
#   bash $SCRATCH/tideforge/repo/scripts/perlmutter/e03_nan.sh table                # after E0.2 and both variants are done
#   (add --dry-run to any of them to print the commands)
# Tags: ft3mainnfdel_s0 / ft3mainnfdelon_s0, ft3mainnfmask_s0 / ft3mainnfmaskon_s0 (the variant is in the name). Resumable.
# Note: ft3_main never saw single masked points in training, so nofill_mask shows the size of the deletion effect for this model,
# not whether masking helps a model trained with masks (G6).
set -u
source "$SCRATCH/tideforge/repo/scripts/perlmutter/env.sh" || exit 2
source "$TF_REPO/scripts/perlmutter/_evalroot.sh"
CMD="${1:-}"
[ $# -gt 0 ] && shift
SMOKE=""; DRY=0
for a in "$@"; do case "$a" in
    --smoke) SMOKE="--smoke";;
    --dry-run) DRY=1;;
    *) echo "unknown option $a"; exit 2;;
esac; done
tf_require TM_EVAL FT3_MAIN_CKPT || exit 2
case "$CMD" in
audit)
    RUN=(python -m tideforge.eval.nan_audit --root "$R" $SMOKE)
    if [ -z "$SMOKE" ]; then RUN+=(--looks); fi
    if [ "$DRY" -eq 1 ]; then echo "DRY RUN: ${RUN[*]}"; exit 0; fi
    tf_link_eval || exit 2
    LOG="$TF_ROOT/logs/e03_audit${SMOKE:+_smoke}.log"
    git -C "$TF_REPO" log --oneline -1 | tee -a "$LOG"
    "${RUN[@]}" 2>&1 | tee -a "$LOG"
    exit "${PIPESTATUS[0]}"
    ;;
score)
    if [ "$DRY" -eq 1 ]; then
        for v in nofill_delete nofill_mask; do
            echo "DRY RUN: python -m tideforge.eval.score score --root $R --ckpt ft3main=$FT3_MAIN_CKPT --variant $v --modes on,off $SMOKE"
        done
        exit 0
    fi
    [ -f "$FT3_MAIN_CKPT" ] || { echo "missing checkpoint $FT3_MAIN_CKPT"; exit 2; }
    tf_link_eval || exit 2
    NGPU=$(nvidia-smi -L | wc -l)
    [ "$NGPU" -ge 2 ] || { echo "need 2 GPUs (salloc ... --gpus 2), found $NGPU"; exit 2; }
    git -C "$TF_REPO" log --oneline -1
    pids=(); g=0
    for v in nofill_delete nofill_mask; do
        LOG="$TF_ROOT/logs/e03_${v}${SMOKE:+_smoke}.log"
        echo "$(date) GPU $g: variant $v -> $LOG"
        CUDA_VISIBLE_DEVICES=$g python -m tideforge.eval.score score --root "$R" --ckpt "ft3main=$FT3_MAIN_CKPT" --variant "$v" \
            --modes on,off $SMOKE > "$LOG" 2>&1 &
        pids+=($!); g=$((g + 1))
    done
    rc=0
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    [ "$rc" -eq 0 ] || { echo "a variant run FAILED: see $TF_ROOT/logs/e03_*.log"; exit 1; }
    python -m tideforge.eval.score board --root "$R" $SMOKE | tail -n 60
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "BOARDS FAILED"; exit 1; }
    echo "E0.3 SCORING DONE${SMOKE:+ (SMOKE)}"
    ;;
table)
    RUN=(python -m tideforge.eval.score table --root "$R" --names ft3mainbase,ft3mainnfdel,ft3mainnfmask --base ft3mainbase
         --looks --experiment E0.3 --why "Phase 0: NaN deletion vs masking at inference (no training)")
    if [ "$DRY" -eq 1 ]; then echo "DRY RUN: ${RUN[*]}"; exit 0; fi
    python -m tideforge.eval.score board --root "$R" | tail -n 5
    "${RUN[@]}" 2>&1 | tee -a "$TF_ROOT/logs/e03_table.log"
    ;;
*) echo "usage: e03_nan.sh audit|score|table [--smoke] [--dry-run]"; exit 2;;
esac
