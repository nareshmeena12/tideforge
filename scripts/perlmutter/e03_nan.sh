#!/bin/bash
# E0.3 + E0.10 in one job (owner 2026-10-29: one look event, one LOOKS row per variant).
# E0.3: (a) NaN audit of the inference contexts (no model); (b) score ft3_main with the variants nofill_delete and nofill_mask,
# groups ON and OFF, next to the E0.2 baseline in the same evaluation root.
#
#   salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 2 --account=$NERSC_ACCOUNT
#   bash $SCRATCH/tideforge/repo/scripts/perlmutter/e03_nan.sh audit [--smoke]      # CPU only; writes R/eval/nan_audit.csv
#   bash $SCRATCH/tideforge/repo/scripts/perlmutter/e03_nan.sh score [--smoke]      # all visible GPUs: a queue of UNITS
#     (variant:mode; nofill_delete / nofill_mask: groups ON and OFF; cov_mask: groups ON only, OFF has no covariates)
#   bash $SCRATCH/tideforge/repo/scripts/perlmutter/e03_nan.sh table                # after E0.2 and both variants are done
#   (add --dry-run to any of them to print the commands)
#   Batch job instead of salloc: sbatch $SCRATCH/tideforge/repo/scripts/perlmutter/e03_score.sbatch (tideforge/slurm.py)
# Tags: ft3mainnfdel_s0 / ft3mainnfdelon_s0, ft3mainnfmask_s0 / ft3mainnfmaskon_s0 (the variant is in the name). Resumable.
# Note: ft3_main never saw single masked points in training, so nofill_mask shows the size of the deletion effect for this model,
# not whether masking helps a model trained with masks (G6).
set -u
source "$SCRATCH/tideforge/repo/scripts/perlmutter/env.sh" || exit 2
source "$TF_REPO/scripts/perlmutter/_evalroot.sh"
CMD="${1:-}"
[ $# -gt 0 ] && shift
SMOKE=""; DRY=0   # the baseline is tidemark's board (see table)
# one process per (variant, mode); separate tags, so they can run at once in the same root. Longest first (tideforge/slurm.py UNITS_H)
UNITS="nofill_delete:on nofill_mask:on nofill_delete:off nofill_mask:off cov_mask:on target_enc:on order_corr:on both:on"
# cov_mask and E0.10's target_enc / order_corr / both: fev only (TIME has no covariates; the E0.10 variants also restrict fev
# to its covariate tasks, tideforge/eval/cov_variants.py)
extra_of() { case "$1" in cov_mask|target_enc|order_corr|both) echo --skip-time;; *) echo;; esac; }
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
        for u in $UNITS; do
            echo "DRY RUN: python -m tideforge.eval.score score --root $R --ckpt ft3main=$FT3_MAIN_CKPT --variant ${u%:*} --modes ${u#*:} $(extra_of ${u%:*}) $SMOKE"
        done
        exit 0
    fi
    [ -f "$FT3_MAIN_CKPT" ] || { echo "missing checkpoint $FT3_MAIN_CKPT"; exit 2; }
    tf_link_eval || exit 2
    NGPU=$(nvidia-smi -L | wc -l)
    [ "$NGPU" -ge 1 ] || { echo "no GPU found"; exit 2; }
    git -C "$TF_REPO" log --oneline -1
    Q=$(mktemp); FAIL=$(mktemp); printf '%s
' $UNITS > "$Q"
    # one worker per GPU; each takes the next unit from the queue file under a lock
    worker() {
        local g=$1 u v m LOG
        while :; do
            u=$(flock "$Q" sh -c 'head -n 1 "$1"; sed -i 1d "$1"' _ "$Q")
            [ -n "$u" ] || return 0
            v=${u%:*}; m=${u#*:}
            LOG="$TF_ROOT/logs/e03_${v}_${m}${SMOKE:+_smoke}.log"
            echo "$(date) GPU $g: variant $v, groups $m -> $LOG"
            CUDA_VISIBLE_DEVICES=$g python -m tideforge.eval.score score --root "$R" --ckpt "ft3main=$FT3_MAIN_CKPT" --variant "$v"                 --modes "$m" $(extra_of "$v") $SMOKE > "$LOG" 2>&1 || echo "$u" >> "$FAIL"
            echo "$(date) GPU $g: $u finished"
        done
    }
    pids=()
    for ((g = 0; g < NGPU; g++)); do worker "$g" & pids+=($!); done
    rc=0
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    [ -s "$FAIL" ] && { echo "FAILED units: $(tr '
' ' ' < "$FAIL")"; rc=1; }
    rm -f "$Q" "$FAIL"
    [ "$rc" -eq 0 ] || { echo "a variant run FAILED: see $TF_ROOT/logs/e03_*.log"; exit 1; }
    python -m tideforge.eval.score board --root "$R" $SMOKE | tail -n 60
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "BOARDS FAILED"; exit 1; }
    echo "E0.3 SCORING DONE${SMOKE:+ (SMOKE)}"
    ;;
table)
    # The baseline is tidemark's board of 2026-10-24 ($TF_REF/scores_long.csv): E0.2's smoke run showed tideforge's wrapper
    # reproduces it exactly (max relative difference 0), so the baseline is not scored again here.
    RUN=(python -m tideforge.eval.score table --root "$R" --names ft3mainnfdel,ft3mainnfmask --base ft3mainnfdel
         --looks --experiment E0.3 --why "Phase 0: NaN deletion vs masking at inference (no training)")
    if [ "$DRY" -eq 1 ]; then
        echo "DRY RUN: ${RUN[*]}"
        for v in ft3mainnfdel ft3mainnfmask ft3maincmask ft3maintenc ft3mainocorr ft3maintencocorr; do
            echo "DRY RUN: python -m tideforge.eval.score compare-board --root $R --name $v --ref $TF_REF/scores_long.csv --summary"
        done
        exit 0
    fi
    python -m tideforge.eval.score board --root "$R" | tail -n 5
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "BOARDS FAILED"; exit 1; }
    "${RUN[@]}" 2>&1 | tee -a "$TF_ROOT/logs/e03_table.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "TABLE FAILED"; exit 1; }
    # cov_mask (fev, groups ON only) is not in the two-mode table; its comparison and LOOKS row come from compare-board
    python -m tideforge.eval.score compare-board --root "$R" --name ft3maincmask --ref "$TF_REF/scores_long.csv" --summary \
        --looks --experiment E0.3 --why "Phase 0: covariate NaN unobserved instead of forward-filled (cov_mask)" 2>&1 \
        | tee -a "$TF_ROOT/logs/e03_table.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "COMPARE-BOARD FAILED for ft3maincmask"; exit 1; }
    # E0.10: inference-only covariate variants (fev covariate tasks, groups ON), one LOOKS row each
    for v in ft3maintenc ft3mainocorr ft3maintencocorr; do
        python -m tideforge.eval.score compare-board --root "$R" --name "$v" --ref "$TF_REF/scores_long.csv" --summary             --looks --experiment E0.10 --why "Phase 0: inference-only covariate variant ($v), fev covariate tasks, groups ON" 2>&1             | tee -a "$TF_ROOT/logs/e03_table.log"
        [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "COMPARE-BOARD FAILED for $v"; exit 1; }
    done
    for v in ft3mainnfdel ft3mainnfmask; do
        python -m tideforge.eval.score compare-board --root "$R" --name "$v" --ref "$TF_REF/scores_long.csv" --summary 2>&1 \
            | tee -a "$TF_ROOT/logs/e03_table.log"
        [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "COMPARE-BOARD FAILED for $v"; exit 1; }
    done
    ;;
*) echo "usage: e03_nan.sh audit|score|table [--smoke] [--dry-run]"; exit 2;;
esac
