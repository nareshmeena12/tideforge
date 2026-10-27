#!/bin/bash
# E0.2: score ft3_main (variant baseline) on fev-bench and TIME, groups ON and OFF, and check the six-number gate.
#
#   salloc --nodes 1 --qos interactive --time 03:00:00 --constraint gpu --gpus 1 --account=$NERSC_ACCOUNT
#   bash $SCRATCH/tideforge/repo/scripts/perlmutter/e02_score.sh --dry-run     # prints the steps, starts nothing
#   bash $SCRATCH/tideforge/repo/scripts/perlmutter/e02_score.sh --smoke       # 3 fev tasks, 2 TIME datasets (outputs in R/eval_smoke)
#   bash $SCRATCH/tideforge/repo/scripts/perlmutter/e02_score.sh               # the real run (resumable: rerun after an interruption)
#
# Steps: (1) scoring via tideforge.eval.score (= tidemark score_groups, unchanged), checkpoint name ft3mainbase, tags ft3mainbase_s0
# (OFF) and ft3mainbaseon_s0 (ON); (2) boards; (3) the two-mode table with the GATE (ON fev 43.78, TIME 0.680 / 0.571; OFF fev 42.09,
# TIME 0.687 / 0.575; +-0.2 fev, +-0.002 TIME) and the LOOKS.md rows; (4) per-task comparison with the Colab log if FEV_LOG exists.
# A failed gate is reported, nothing is tuned. Paste the whole output of steps 3 and 4 back.
set -u
source "$SCRATCH/tideforge/repo/scripts/perlmutter/env.sh" || exit 2
source "$TF_REPO/scripts/perlmutter/_evalroot.sh"
SMOKE=""; DRY=0
for a in "$@"; do case "$a" in
    --smoke) SMOKE="--smoke";;
    --dry-run) DRY=1;;
    *) echo "unknown option $a"; exit 2;;
esac; done
tf_require TM_EVAL FT3_MAIN_CKPT || exit 2
FEV_LOG="${FEV_LOG:-$TF_REPO/reference/fev_log_ft3.txt}"
LOG="$TF_ROOT/logs/e02${SMOKE:+_smoke}.log"
SCORE=(python -m tideforge.eval.score score --root "$R" --ckpt "ft3main=$FT3_MAIN_CKPT" --variant baseline --modes on,off $SMOKE)
BOARD=(python -m tideforge.eval.score board --root "$R" $SMOKE)
TABLE=(python -m tideforge.eval.score table --root "$R" --names ft3mainbase --base ft3mainbase --gate
       --looks --experiment E0.2 --why "Phase 0: reproduce ft3_main with the scorer wrapper")
COMPARE=(python -m tideforge.eval.score compare-log --root "$R" --name ft3mainbase --log "$FEV_LOG")
if [ "$DRY" -eq 1 ]; then
    echo "DRY RUN. Evaluation root $R (links to $TM_EVAL/{data,hf,fev_naive.csv,fev_naive_parts}); log $LOG"
    echo "  [1] ${SCORE[*]}"
    echo "  [2] ${BOARD[*]}"
    echo "  [3] ${TABLE[*]}"
    echo "  [4] ${COMPARE[*]}   (only if $FEV_LOG exists)"
    [ -f "$FT3_MAIN_CKPT" ] || echo "  NOTE: $FT3_MAIN_CKPT does not exist"
    exit 0
fi
[ -f "$FT3_MAIN_CKPT" ] || { echo "missing checkpoint $FT3_MAIN_CKPT"; exit 2; }
tf_link_eval || exit 2
git -C "$TF_REPO" log --oneline -1 | tee -a "$LOG"
echo "$(date) [1] scoring" | tee -a "$LOG"
"${SCORE[@]}" 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "SCORING FAILED"; exit 1; }
echo "$(date) [2] boards" | tee -a "$LOG"
"${BOARD[@]}" 2>&1 | tee -a "$LOG" | tail -n 60
if [ -n "$SMOKE" ]; then echo "SMOKE DONE (the table and the gate need the full run)"; exit 0; fi
echo "$(date) [3] table + gate + LOOKS rows" | tee -a "$LOG"
"${TABLE[@]}" 2>&1 | tee -a "$LOG"
if [ -f "$FEV_LOG" ]; then
    echo "$(date) [4] per-task comparison with the Colab log" | tee -a "$LOG"
    "${COMPARE[@]}" 2>&1 | tee -a "$LOG"
else
    echo "(no $FEV_LOG: set FEV_LOG to the Colab per-task log for the per-task comparison)"
fi
echo "E0.2 DONE. Log: $LOG"
