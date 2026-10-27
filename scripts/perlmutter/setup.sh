#!/bin/bash
# One-time setup on a Perlmutter LOGIN node (no GPU needed):  bash setup.sh
# Modelled on tidemark's scripts/perlmutter/setup.sh. Clones (or pulls) nareshmeena12/tideforge (private) to $TF_REPO, installs it
# with `pip install --user --no-deps -e` on top of NERSC's pytorch module, checks that the scoring dependencies import (fev,
# gluonts, ...: tidemark's README section 5a installs them), and runs the fast CPU tests.
#
# GitHub credentials, first that applies (the token is never written to .git/config, a remote URL, a file, argv or a log):
#   1. TF_GIT_URL=git@github.com:nareshmeena12/tideforge.git  plain git over SSH with a read-only deploy key  [preferred]
#   2. `gh auth login` done on this machine                   git asks `gh auth git-credential`
#   3. GH_TOKEN in the environment                            an in-memory credential helper: git runs it through sh, which
#      expands $GH_TOKEN from the environment. Set it without leaving it in the shell history:
#          read -rs GH_TOKEN && export GH_TOKEN
# `credential.helper=` (empty) first clears any helper from ~/.gitconfig, so a `store` helper cannot write the token to disk.
# After the clone / pull, tf_check_no_token stops the setup if the token string or a credentialed URL is found anyway.
TF_GIT_URL="${TF_GIT_URL:-https://github.com/nareshmeena12/tideforge.git}"

tf_git() {
    case "$TF_GIT_URL" in
        git@*|ssh://*) GIT_TERMINAL_PROMPT=0 git "$@"; return ;;
    esac
    if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
        GIT_TERMINAL_PROMPT=0 git -c credential.helper= -c 'credential.helper=!gh auth git-credential' "$@"
    elif [ -n "${GH_TOKEN:-}" ]; then
        GIT_TERMINAL_PROMPT=0 git -c credential.helper= \
            -c 'credential.helper=!f() { test "$1" = get && printf "username=x-access-token\npassword=%s\n" "$GH_TOKEN"; }; f' "$@"
    else
        echo "no GitHub credentials: set TF_GIT_URL to an SSH URL (deploy key), run gh auth login, or read -rs GH_TOKEN && export GH_TOKEN"
        return 2
    fi
}

# tf_check_no_token REPO: fail if a remote URL carries credentials, or if $GH_TOKEN appears in the repo's git config or in a git
# config / credential file. Bash builtins only, so the token is never passed as an argument to another program.
tf_check_no_token() {
    local repo="$1" f content
    content=$(git -C "$repo" remote -v) || return 3
    if [[ "$content" == *://*@* ]]; then
        echo "SECURITY: a remote URL of $repo has credentials in it: git -C $repo remote set-url origin $TF_GIT_URL"
        return 3
    fi
    [ -n "${GH_TOKEN:-}" ] || return 0
    for f in "$repo/.git/config" "$HOME/.gitconfig" "$HOME/.git-credentials" "${XDG_CONFIG_HOME:-$HOME/.config}/git/config" \
             "${XDG_CONFIG_HOME:-$HOME/.config}/git/credentials"; do
        [ -f "$f" ] || continue
        content=$(<"$f")
        if [[ "$content" == *"$GH_TOKEN"* ]]; then
            echo "SECURITY: the GitHub token is stored in $f: remove it there and revoke the token on GitHub"
            return 3
        fi
    done
    return 0
}

# sourced (tests/test_setup_auth.py): define the functions only
[[ "${BASH_SOURCE[0]}" != "$0" ]] && return 0

set -eu
TF_REPO="${TF_REPO:-$SCRATCH/tideforge/repo}"
mkdir -p "$(dirname "$TF_REPO")"
if [ ! -d "$TF_REPO/.git" ]; then
    tf_git clone -q "$TF_GIT_URL" "$TF_REPO"
else
    tf_git -C "$TF_REPO" pull -q --ff-only || echo "git pull failed: rerun setup.sh after fixing the credentials"
fi
tf_check_no_token "$TF_REPO"
source "$TF_REPO/scripts/perlmutter/env.sh"
mkdir -p "$TF_ROOT/logs"
pip install --user -q --no-deps -e "$TF_REPO"
pip install --user -q pytest holidays
python - <<'PY'
import importlib
missing = [m for m in ("torch", "numpy", "pandas", "scipy", "yaml", "holidays", "fev", "datasets", "gluonts", "toolz", "dotenv",
                       "huggingface_hub")
           if importlib.util.find_spec(m) is None]
print("missing:", missing if missing else "none")
if missing:
    raise SystemExit("install them as in tidemark's scripts/perlmutter/README.md section 5a (pip install --user ...)")
PY
python -m pytest -q
git -C "$TF_REPO" log --oneline -1
echo "SETUP OK. GPU parity (on a GPU node): python -m pytest -q -m gpu tests/test_gen_parity.py"
