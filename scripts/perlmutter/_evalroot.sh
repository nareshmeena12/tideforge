# Sourced by e02_score.sh / e03_nan.sh after env.sh: the T0 evaluation root R = $TF_ROOT/eval_t0, FRESH for tideforge (score_groups
# resumes per tag and task, so a root holding old tidemark results under the same tags would silently re-use them).
# tidemark's fetched data and fev seasonal-naive cache are linked in, read-only by use.
R="$TF_ROOT/eval_t0"
tf_link_eval() {
    [ -d "$TM_EVAL/data" ] || { echo "no $TM_EVAL/data: TM_EVAL must be tidemark's fetched eval folder (score_twins fetch)"; return 2; }
    [ -f "$TM_EVAL/fev_naive.csv" ] || { echo "no $TM_EVAL/fev_naive.csv: run tidemark's seasonal-naive step first (score_twins naive)"; return 2; }
    mkdir -p "$R/eval" "$R/eval_smoke" "$TF_ROOT/logs"
    local x
    for x in data hf fev_naive.csv fev_naive_parts; do
        if [ -e "$TM_EVAL/$x" ]; then ln -sfn "$TM_EVAL/$x" "$R/eval/$x"; fi
    done
    # --smoke writes its outputs to eval_smoke and reads the seasonal-naive cache from there too (score_twins Paths.out),
    # so the cache is linked there as well (data and hf are always read from eval/)
    for x in fev_naive.csv fev_naive_parts; do
        if [ -e "$TM_EVAL/$x" ]; then ln -sfn "$TM_EVAL/$x" "$R/eval_smoke/$x"; fi
    done
    return 0
}
