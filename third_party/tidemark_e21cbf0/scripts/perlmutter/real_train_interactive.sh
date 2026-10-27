#!/bin/bash
# Real-data post-training of ft3_main inside an interactive allocation (4 x A100 of ONE node). Not part of the softmax/GLA/GDN/SWA study.
#
#   salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus-per-node 4 --account=m4392_g
#   bash $SCRATCH/tidemark/repo/scripts/perlmutter/real_train_interactive.sh --bench             # loader throughput on its own (1 GPU), no training
#   bash $SCRATCH/tidemark/repo/scripts/perlmutter/real_train_interactive.sh --smoke --dry-run   # the command, the checks, nothing started
#   bash $SCRATCH/tidemark/repo/scripts/perlmutter/real_train_interactive.sh --smoke             # 1B points (about 0.33B real)
#   bash $SCRATCH/tidemark/repo/scripts/perlmutter/real_train_interactive.sh                     # the 30B run (only after the owner's go-ahead)
#
# The equivalent torchrun line (what the script runs; add --smoke for the smoke run):
#   torchrun --standalone --nproc_per_node=4 -m tidemark.real.train --out-dir $SCRATCH/tidemark/real_run/real30b [--copy-to $REAL_COPY_TO]
# If the allocation ends, or the run stops (exit code 3), run THE SAME COMMAND AGAIN in a new allocation: it resumes exactly from last.pt
# (model, optimizer, step, points, every rank's RNG and data position). Milestone files ckpt_10B.pt / ckpt_20B.pt / ckpt_30B.pt are written in
# ft3_main.pt's format and copied to $REAL_COPY_TO when it is set (permanent storage: CFS or home; $SCRATCH is purged).
# Environment: CKPT (default /pscratch/sd/n/naresh/tidemark/ckpt/ft3_main.pt), REAL_ROOT (default $SCRATCH/tidemark/real: shards/ + plan.json),
# REAL_OUT, REAL_COPY_TO. Placeholders to fill: REAL_COPY_TO.
set -u
source "$SCRATCH/tidemark/repo/scripts/perlmutter/env.sh"
export TRITON_CACHE_DIR="$ROOT/triton_cache" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TIDEMARK_SIGTERM_STOP=1
CKPT="${CKPT:-/pscratch/sd/n/naresh/tidemark/ckpt/ft3_main.pt}"
REAL_ROOT="${REAL_ROOT:-$SCRATCH/tidemark/real}"
REAL_SHARDS="${REAL_SHARDS:-shards}"          # folder of the converted datasets under REAL_ROOT (shards_full for the 107-shard run)
if [ "$REAL_SHARDS" = shards_full ]; then PLAN_DEFAULT="$REAL_ROOT/plan_full.json"; else PLAN_DEFAULT="$REAL_ROOT/plan.json"; fi
REAL_PLAN="${REAL_PLAN:-$PLAN_DEFAULT}"            # shards_full -> plan_full.json (made with plan --match-plan plan.json)
REAL_NAME="${REAL_NAME:-}"                      # run name (default real30b / smoke1b); real30b_full for the full-pool run
SMOKE=""; DRY=0; BENCH=0; SWEEP=0; RUN=real30b
while [ $# -gt 0 ]; do case "$1" in
    --smoke) SMOKE="--smoke";;
    --dry-run) DRY=1;;
    --bench) BENCH=1;;
    --sweep) BENCH=1; SWEEP=1;;
    --run) RUN="$2"; shift;;
    *) echo "unknown option $1 (--run real30b|real200b, --smoke, --dry-run, --bench, --sweep)"; exit 2;;
esac; shift; done
case "$RUN" in
    real30b) NAME=real30b; [ -n "$SMOKE" ] && NAME=smoke1b;;
    real200b) NAME=real200b; [ -n "$SMOKE" ] && NAME=smoke2b
              [ "$REAL_SHARDS" = shards ] && REAL_SHARDS=shards_full
              [ "$REAL_PLAN" = "$PLAN_DEFAULT" ] && REAL_PLAN="$REAL_ROOT/plan_200b.json";;
    *) echo "unknown --run $RUN (real30b, real200b)"; exit 2;;
esac
[ -n "$REAL_NAME" ] && NAME="$REAL_NAME"
OUT="${REAL_OUT:-$SCRATCH/tidemark/real_run/$NAME}"
COPY="${REAL_COPY_TO:-}"
LOG="$ROOT/logs/real_$NAME.log"
if [ "$BENCH" -eq 1 ]; then
    if [ "$SWEEP" -eq 1 ]; then
        # the 4 ranks of the node read at the same time, as in the run; the workers-per-rank sweep prints real-part points/s per GPU
        CMD=(torchrun --standalone --nproc_per_node=4 -m tidemark.real.bench_loader --sweep --real-root "$REAL_ROOT" --ckpt "$CKPT" --shards-dir "$REAL_SHARDS" --plan "$REAL_PLAN")
    else
        CMD=(python -m tidemark.real.bench_loader --real-root "$REAL_ROOT" --ckpt "$CKPT" --shards-dir "$REAL_SHARDS" --plan "$REAL_PLAN")
    fi
    [ "$RUN" = real200b ] && CMD+=(--real-fraction 0.5)
else
    CMD=(torchrun --standalone --nproc_per_node=4 -m tidemark.real.train --run "$RUN" $SMOKE --ckpt "$CKPT" --real-root "$REAL_ROOT" --shards-dir "$REAL_SHARDS" --plan "$REAL_PLAN" --name "$NAME" --out-dir "$OUT")
    [ -n "$COPY" ] && CMD+=(--copy-to "$COPY")
fi
echo "run: $NAME   checkpoint: $CKPT   data: $REAL_ROOT   output: $OUT   copy milestones to: ${COPY:-(REAL_COPY_TO not set: milestones stay in the output folder)}"
echo "command: ${CMD[*]}"
for f in "$CKPT" "$REAL_PLAN" "$REAL_ROOT/$REAL_SHARDS/index.json"; do [ -e "$f" ] || echo "MISSING: $f"; done
[ "$DRY" -eq 1 ] && { [ -f "$OUT/last.pt" ] && echo "last.pt exists: this run would RESUME from it"; echo "(dry run: nothing started)"; exit 0; }
for f in "$CKPT" "$REAL_PLAN" "$REAL_ROOT/$REAL_SHARDS/index.json"; do [ -e "$f" ] || exit 2; done
if [ "$BENCH" -eq 0 ]; then
    [ "$(nvidia-smi -L | wc -l)" -ge 4 ] || { echo "need 4 GPUs in this allocation (--gpus-per-node 4)"; exit 2; }
    [ -z "$COPY" ] && echo "WARNING: REAL_COPY_TO is not set; milestone checkpoints will only be in $OUT (scratch is purged)"
fi
mkdir -p "$ROOT/logs" "$OUT"
rm -f "$OUT/STOP"
"${CMD[@]}" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
[ "$RC" -eq 3 ] && echo "STOPPED with a saved last.pt (exit 3): run the same command again to resume."
exit "$RC"
