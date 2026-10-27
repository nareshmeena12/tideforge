#!/bin/bash
#SBATCH -A m4392_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH -t 01:00:00
#SBATCH -J real_smoke
#SBATCH --signal=B:TERM@120
#SBATCH --requeue
#SBATCH --open-mode=append
# Real-data post-training of ft3_main, 1B-point SMOKE run (about 0.33B real). Placeholders: -q, -t, REAL_COPY_TO. Resumes from last.pt after a requeue.
# Submitted by submit_all.sh (it sets the log file: $SCRATCH/tidemark/logs/<job name>-<job id>.out).
set -u
source "$SCRATCH/tidemark/repo/scripts/perlmutter/env.sh"

export TRITON_CACHE_DIR="$SCRATCH/tidemark/triton_cache"; mkdir -p "$TRITON_CACHE_DIR"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TIDEMARK_SIGTERM_STOP=1
NAME="${REAL_NAME:-smoke1b}"
SHARDS="${REAL_SHARDS:-shards}"        # shards_full: the 107-shard pool
if [ "$SHARDS" = shards_full ]; then PLAN_DEFAULT="$SCRATCH/tidemark/real/plan_full.json"; else PLAN_DEFAULT="$SCRATCH/tidemark/real/plan.json"; fi
PLAN="${REAL_PLAN:-$PLAN_DEFAULT}"
OUT="${REAL_OUT:-$SCRATCH/tidemark/real_run/$NAME}"
COPY_TO="${REAL_COPY_TO:-}"            # PLACEHOLDER: permanent storage for the milestone files, e.g. /global/cfs/cdirs/<project>/<you>/tidemark_real
mkdir -p "$OUT"
rm -f "$OUT/STOP"
TERMINATED=0
on_term() {
    TERMINATED=1
    echo "$(date) SIGTERM (preempted, or 120 s before the walltime): stop file written, waiting for last.pt"
    touch "$OUT/STOP"
}
trap on_term TERM USR1
echo "$(date) job $SLURM_JOB_ID on $(hostname), restart count ${SLURM_RESTART_COUNT:-0}; output $OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

torchrun --standalone --nproc_per_node=4 -m tidemark.real.train --run real30b --smoke --name "$NAME" --shards-dir "$SHARDS" --plan "$PLAN" --out-dir "$OUT" ${COPY_TO:+--copy-to "$COPY_TO"} < /dev/null &
PID=$!
wait $PID; RC=$?
while kill -0 $PID 2>/dev/null; do wait $PID; RC=$?; done
echo "$(date) training exited with code $RC (terminated by signal: $TERMINATED)"
if [ "$RC" -eq 3 ] || { [ "$TERMINATED" -eq 1 ] && [ "$RC" -ne 0 ]; }; then
    ls -l "$OUT/last.pt" 2>/dev/null || echo "WARNING: no last.pt in $OUT"
    rm -f "$OUT/STOP"
    echo "$(date) asking Slurm to requeue $SLURM_JOB_ID; the requeued job resumes from last.pt"
    scontrol requeue "$SLURM_JOB_ID" && exit 0
    exit 143
fi
exit "$RC"
