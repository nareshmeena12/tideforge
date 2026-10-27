# Sourced by every tideforge script on Perlmutter. Modelled on tidemark's scripts/perlmutter/env.sh (NERSC pytorch module,
# everything under $SCRATCH, unbuffered output).
#
# FILL IN before the first run: the values marked FILL_IN below (edit this file on Perlmutter, or export them before sourcing).
# The defaults that are not FILL_IN are proposals; change them if you put things elsewhere.
module load pytorch
# the owner's paths live outside the repo, in ~/.tideforge_env (exports of TF_DATA, TM_EVAL, FT3_MAIN_CKPT, NERSC_ACCOUNT)
[ -f "$HOME/.tideforge_env" ] && source "$HOME/.tideforge_env"
export TF_REPO="${TF_REPO:-$SCRATCH/tideforge/repo}"     # this repo's clone (setup.sh clones it here)
export TF_ROOT="${TF_ROOT:-$SCRATCH/tideforge}"          # tideforge outputs: evaluation roots, logs
# tidemark's FETCHED evaluation folder: the directory that holds data/ (TIME repo + data, fev tasks), hf/ (Hugging Face cache)
# and fev_naive.csv, as written by `python -m tidemark.eval.score_twins fetch` and `naive` (tidemark: $SCRATCH/tidemark/eval).
export TM_EVAL="${TM_EVAL:-FILL_IN}"
export FT3_MAIN_CKPT="${FT3_MAIN_CKPT:-FILL_IN}"         # absolute path of ft3_main.pt
export NERSC_ACCOUNT="${NERSC_ACCOUNT:-FILL_IN}"         # for salloc / sbatch --account (GPU allocations end in _g)
export TF_DATA="${TF_DATA:-FILL_IN}"                     # persistent dev-set data (CFS, not $SCRATCH: scratch is purged)
# tidemark's board of 2026-10-24 (E0.2 reference: scores_long.csv, boards, the Colab fev log), copied to CFS
export TF_REF="${TF_REF:-$TF_DATA/ref/tidemark_board_20261024}"
export FEV_LOG="${FEV_LOG:-$TF_REF/fev_log_ft3.txt}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1            # compute nodes: never touch the network

# tf_require VAR ...: stop with a message if any of the named variables is still FILL_IN
tf_require() {
    local v bad=0
    for v in "$@"; do
        if [ "${!v:-FILL_IN}" = "FILL_IN" ]; then echo "env.sh: $v is not set (edit scripts/perlmutter/env.sh or export it)"; bad=1; fi
    done
    return $bad
}
cd "$TF_REPO" || return 2
