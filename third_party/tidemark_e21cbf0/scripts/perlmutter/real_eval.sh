#!/bin/bash
# Score ft3-architecture checkpoints (ft3_main, later the post-trained ones) on TIME and fev-bench with the repo's scoring code.
# Real-data post-training track (not part of the softmax/GLA/GDN/SWA study).
#
#   salloc --nodes 1 --qos interactive --time 02:00:00 --constraint gpu --gpus 1 --account=m4392_g
#   bash $SCRATCH/tidemark/repo/scripts/perlmutter/real_eval.sh --baseline --both --dry-run     # what would run
#   bash $SCRATCH/tidemark/repo/scripts/perlmutter/real_eval.sh --baseline --both --smoke       # 3 fev tasks, 2 TIME datasets
#   bash $SCRATCH/tidemark/repo/scripts/perlmutter/real_eval.sh --baseline --both               # ft3_main, both modes, the GATE
#   bash $SCRATCH/tidemark/repo/scripts/perlmutter/real_eval.sh --both p10b=$SCRATCH/tidemark/real_run/smoke1b/ckpt_10B.pt p20b=... p30b=...
#   bash $SCRATCH/tidemark/repo/scripts/perlmutter/real_eval.sh --baseline                      # univariate only (the earlier option)
#
# --both scores every NAME=PATH in BOTH modes (tidemark/real/score_groups.py): groups ON (tag NAMEon_s0: fev with covariates and sibling variates
# visible, TIME with a multivariate series as one group) and groups OFF (tag NAME_s0: the univariate scorer), prints one table with TIME MASE,
# TIME CRPS, fev skill and fev win rate for each mode and the change versus ft3main, and writes it to $ROOT/size_real/eval/board/real_table_modes.csv.
# --baseline = ft3main=$CKPT (default /pscratch/sd/n/naresh/tidemark/ckpt/ft3_main.pt). With --baseline --both the GATE is checked:
#   ON  fev skill 43.78 (+-0.2), TIME MASE 0.680 and CRPS 0.571 (+-0.002)      OFF  fev 42.09 (+-0.2), TIME 0.687 / 0.575 (+-0.002)
# and, if $FEV_LOG exists (reference/fev_log_ft3.txt, the Colab per-task log), the per-task ON and HIDDEN ratios are compared with it.
# On a failed gate nothing is tuned: the table and the per-task differences say what differs.
# Steps: (1) strict load + forward parity (scripts/real/check_forward.py), (2) scoring (resumable per task), (3) boards, (4) the table.
# Needs the fetched data and the seasonal-naive cache of score_interactive.sh in $ROOT/eval.
set -u
source "$SCRATCH/tidemark/repo/scripts/perlmutter/env.sh"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TIDEMARK_DATA="${TIDEMARK_DATA:-$ROOT/data}" TRITON_CACHE_DIR="$ROOT/triton_cache"
CKPT="${CKPT:-/pscratch/sd/n/naresh/tidemark/ckpt/ft3_main.pt}"
REPO="$SCRATCH/tidemark/repo"
FEV_LOG="${FEV_LOG:-$REPO/reference/fev_log_ft3.txt}"
SMOKE=""; DRY=0; BASE=0; BOTH=0; CK=()
for a in "$@"; do case "$a" in
    --smoke) SMOKE="--smoke";;
    --dry-run) DRY=1;;
    --both) BOTH=1;;
    --baseline) BASE=1; CK+=("ft3main=$CKPT");;
    *=*) CK+=("$a");;
    *) echo "unknown option $a"; exit 2;;
esac; done
[ "${#CK[@]}" -gt 0 ] || { echo "give --baseline or NAME=PATH"; exit 2; }
R="$ROOT/size_real"
CKARGS=(); for c in "${CK[@]}"; do CKARGS+=(--ckpt "$c"); done
if [ "$BOTH" -eq 1 ]; then
    SCORE=(python -m tidemark.real.score_groups score --root "$R" "${CKARGS[@]}" --modes on,off $SMOKE)
    TABLE=(python -m tidemark.real.score_groups table --root "$R")
    [ "$BASE" -eq 1 ] && TABLE+=(--gate)
else
    SCORE=(python -m tidemark.real.score_ft3 score --root "$R" "${CKARGS[@]}" $SMOKE)
    TABLE=(python -m tidemark.real.score_ft3 table --root "$R")
    [ "$BASE" -eq 1 ] && TABLE+=(--expect-time 0.687 --expect-crps 0.575 --expect-fev 42.09)
fi
PARITY=(python "$REPO/scripts/real/check_forward.py" --ckpt "$CKPT")
if [ -f "$REPO/reference/tide_v5_ft3.py" ]; then PARITY+=(--reference "$REPO/reference/tide_v5_ft3.py"); elif [ -n "${GOLDEN:-}" ]; then PARITY+=(--golden "$GOLDEN"); fi
COMPARE=(python -m tidemark.real.score_groups compare-log --root "$R" --log "$FEV_LOG")
if [ "$DRY" -eq 1 ]; then
    echo "DRY RUN (nothing is started). Evaluation root: $R (links to $ROOT/eval/{data,hf,fev_naive.csv,fev_naive_parts}); modes: $([ "$BOTH" -eq 1 ] && echo 'ON + OFF' || echo 'OFF (univariate)')"
    [ "$BASE" -eq 1 ] && echo "  [1] ${PARITY[*]}"
    echo "  [2] ${SCORE[*]}"
    echo "  [3] python -m tidemark.real.score_ft3 board --root $R $SMOKE"
    echo "  [4] ${TABLE[*]}"
    [ "$BOTH" -eq 1 ] && [ "$BASE" -eq 1 ] && echo "  [5] ${COMPARE[*]}      (only if $FEV_LOG exists)"
    for c in "${CK[@]}"; do [ -f "${c#*=}" ] || echo "  NOTE: ${c#*=} does not exist (yet)"; done
    exit 0
fi
[ -d "$ROOT/eval/data" ] || { echo "no $ROOT/eval/data: run score_interactive.sh (fetch, naive) first"; exit 2; }
for c in "${CK[@]}"; do [ -f "${c#*=}" ] || { echo "missing checkpoint ${c#*=}"; exit 2; }; done
mkdir -p "$R/eval" "$ROOT/logs"
for x in data hf fev_naive.csv fev_naive_parts; do [ -e "$ROOT/eval/$x" ] && ln -sfn "$ROOT/eval/$x" "$R/eval/$x"; done
if [ "$BASE" -eq 1 ]; then
    echo "$(date) [1] strict load + forward parity"
    "${PARITY[@]}" 2>&1 | tee "$ROOT/logs/real_parity.log"; [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "PARITY FAILED: stop"; exit 1; }
fi
echo "$(date) [2] scoring"
"${SCORE[@]}" 2>&1 | tee -a "$ROOT/logs/real_eval.log" || exit 1
echo "$(date) [3] boards"
python -m tidemark.real.score_ft3 board --root "$R" $SMOKE 2>&1 | tee -a "$ROOT/logs/real_eval.log" | tail -n 60
[ -n "$SMOKE" ] && { echo "smoke done (the table needs the full run)"; exit 0; }
echo "$(date) [4] table"
"${TABLE[@]}" 2>&1 | tee -a "$ROOT/logs/real_eval.log"
if [ "$BOTH" -eq 1 ] && [ "$BASE" -eq 1 ]; then
    if [ -f "$FEV_LOG" ]; then
        echo "$(date) [5] per-task comparison with the Colab log"
        "${COMPARE[@]}" 2>&1 | tee -a "$ROOT/logs/real_eval.log"
    else
        echo "(no $FEV_LOG: copy reference/fev_log_ft3.txt there, or set FEV_LOG, for the per-task comparison)"
    fi
fi
