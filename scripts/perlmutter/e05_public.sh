#!/bin/bash
# E0.5 public models (owner OK 2026-10-29): install the pinned packages, check them on a GPU, then the fev wrapper check.
#
#   bash $SCRATCH/tideforge/repo/scripts/perlmutter/e05_public.sh install core     # login node (network): the `public` extra
#                                         # into the user site on top of the pytorch module, torch held fixed; pip check
#   bash ... e05_public.sh install toto2                                        # login node: $TF_ROOT/venvs/toto2 (Python 3.12)
#   bash ... e05_public.sh install tirex2                                       # login node: $TF_ROOT/venvs/tirex2
#   bash ... e05_public.sh weights                                              # login node: snapshots of the pinned HF commits
#   (in salloc, 1 GPU)  bash ... e05_public.sh check [MODEL ...]                 # load + one tiny forecast per model; TiRex must
#                                         # build its xlstm CUDA kernels here: if they do not build, it stops (no torch fallback)
#   bash ... e05_public.sh fevcheck [MODEL ...]                                  # all visible GPUs: one model per GPU (queue)
#   bash ... e05_public.sh table [--looks]                                       # the wrapper-check table; --looks: LOOKS rows
#   Batch job instead of salloc for fevcheck: sbatch $SCRATCH/tideforge/repo/scripts/perlmutter/e05_public.sbatch
# Weights and fev data share one HF cache: tidemark's ($TM_EVAL/hf, through the evaluation root R/eval/hf), so the compute
# nodes stay offline. Pins: tideforge/eval/public_models/__init__.py (SPECS) and scripts/perlmutter/venvs/<venv>.txt.
set -u
source "$SCRATCH/tideforge/repo/scripts/perlmutter/env.sh" || exit 2
source "$TF_REPO/scripts/perlmutter/_evalroot.sh"
CMD="${1:-}"; [ $# -gt 0 ] && shift
tf_require TM_EVAL || exit 2
export HF_HOME="$R/eval/hf"
OUT="$TF_ROOT/public/fevcheck"
FEV="$TF_ROOT/ext/fev"
ALL="chronos2 timesfm25 toto2_2p5b toto2_22m toto2_4m tirex tirex2 timestral28m timestral5m"   # longest first (estimate)
VENVS="$TF_ROOT/venvs"
mkdir -p "$TF_ROOT/logs" "$OUT" "$VENVS"
pyof() {   # the python that runs a model: its venv's, or the core env's
    local v
    v=$(python -c "from tideforge.eval.public_models import SPECS; print(SPECS['$1'].venv or '')") || return 2
    if [ -n "$v" ]; then echo "$VENVS/$v/bin/python"; else echo python; fi
}
case "$CMD" in
install)
    WHAT="${1:-}"
    case "$WHAT" in
    core)
        PINS=$(python -c "from tideforge.eval.public_models import CHRONOS, TIMESFM, TIREX, TIMESTRAL; print(CHRONOS, TIMESFM, TIREX, TIMESTRAL)")
        TORCH=$(python -c "import torch; print(torch.__version__.split('+')[0])")
        echo "core env: torch $TORCH held fixed; installing: $PINS"
        python -m pip install --user $PINS "torch==$TORCH" || exit 1
        python -m pip check || { echo "pip check FAILED in the core env: report to the owner (Chronos-2 may need its own venv; not forced)"; exit 1; }
        ;;
    toto2|tirex2)
        PY="${PY312:-python3.12}"
        [ "$WHAT" = tirex2 ] && PY="${PY311:-$PY}"
        command -v "$PY" >/dev/null || { echo "$PY not found: set PY312 to a Python >= 3.12 (e.g. from module load python)"; exit 2; }
        "$PY" -m venv "$VENVS/$WHAT" || exit 1
        "$VENVS/$WHAT/bin/python" -m pip install --upgrade pip
        "$VENVS/$WHAT/bin/python" -m pip install -r "$TF_REPO/scripts/perlmutter/venvs/$WHAT.txt" || exit 1
        "$VENVS/$WHAT/bin/python" -m pip check || { echo "pip check FAILED in venv $WHAT: report to the owner"; exit 1; }
        ;;
    *) echo "install core | toto2 | tirex2"; exit 2;;
    esac
    ;;
weights)
    HF_HUB_OFFLINE=0 python - <<'EOF' || exit 1
from huggingface_hub import HfApi, snapshot_download
from tideforge.eval.public_models import SPECS
for name, s in SPECS.items():
    live = HfApi().model_info(s.repo).sha
    print(f"{name:13s} {s.repo} pinned {s.revision[:12]} live {live[:12]}{'' if live == s.revision else '  (main moved: the pin is used)'}")
    snapshot_download(repo_id=s.repo, revision=s.revision)
EOF
    ;;
check)
    tf_link_eval || exit 2
    for m in ${*:-$ALL}; do
        P=$(pyof "$m") || exit 2
        echo "== $m ($P)"
        PYTHONPATH="$TF_REPO" "$P" - "$m" <<'EOF' || { echo "CHECK FAILED: $m (stop and tell the owner; TiRex: the xlstm kernels did not build)"; exit 1; }
import sys
import numpy as np
from tideforge.eval.public_models import make
from tideforge.probes.common import Cell
m = make(sys.argv[1], "cuda", in_process=True)
x = np.sin(np.arange(2 * 256) / 8).astype(np.float32).reshape(2, 1, 256)
q = m.predict_cell(Cell("check", {"targets": x, "future": np.zeros((2, 1, 24))}, 24), "hidden")
assert q.shape == (2, 1, 9, 24) and np.isfinite(q).all(), q.shape
print(sys.argv[1], "OK", q.shape, "median[0, :4]", np.round(q[0, 0, 4, :4], 3))
EOF
    done
    ;;
fevcheck)
    tf_link_eval || exit 2
    [ -d "$FEV/benchmarks/fev_bench/results" ] || { echo "no $FEV: git clone https://github.com/autogluon/fev $FEV (login node)"; exit 2; }
    NGPU=$(nvidia-smi -L | wc -l)
    [ "$NGPU" -ge 1 ] || { echo "no GPU found"; exit 2; }
    git -C "$TF_REPO" log --oneline -1
    git -C "$FEV" log --oneline -1
    Q=$(mktemp); FAIL=$(mktemp); printf '%s\n' ${*:-$ALL} > "$Q"
    worker() {   # one model per GPU at a time (the 2.5B model alone on its GPU)
        local g=$1 m LOG
        while :; do
            m=$(flock "$Q" sh -c 'head -n 1 "$1"; sed -i 1d "$1"' _ "$Q")
            [ -n "$m" ] || return 0
            # resumable (requeue): a model with a result is not rerun (a FAIL result is a result too; delete its json to redo)
            if [ -f "$OUT/$m.json" ]; then echo "$(date) $m: $OUT/$m.json exists, skipped"; continue; fi
            LOG="$TF_ROOT/logs/e05_fevcheck_$m.log"
            echo "$(date) GPU $g: $m -> $LOG"
            CUDA_VISIBLE_DEVICES=$g python -m tideforge.eval.public_models.fevcheck run --model "$m" --root "$R" --fev "$FEV" \
                --out "$OUT" > "$LOG" 2>&1 || echo "$m" >> "$FAIL"
            echo "$(date) GPU $g: $m finished: $(tail -n 1 "$LOG")"
        done
    }
    pids=()
    for ((g = 0; g < NGPU; g++)); do worker "$g" & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p"; done
    rc=0
    [ -s "$FAIL" ] && { echo "FAILED or outside 2%: $(tr '\n' ' ' < "$FAIL") (see $TF_ROOT/logs/e05_fevcheck_*.log)"; rc=1; }
    rm -f "$Q" "$FAIL"
    python -m tideforge.eval.public_models.fevcheck table --out "$OUT"
    exit $rc
    ;;
table)
    python -m tideforge.eval.public_models.fevcheck table --out "$OUT" "$@"
    ;;
*)
    echo "usage: e05_public.sh install core|toto2|tirex2 | weights | check [MODEL...] | fevcheck [MODEL...] | table [--looks]"
    exit 2
    ;;
esac
