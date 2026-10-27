#!/bin/bash
# One-time setup on a Perlmutter LOGIN node (no GPU needed):  bash setup.sh
# Modelled on tidemark's scripts/perlmutter/setup.sh. Clones nareshmeena12/tideforge (private: needs a GitHub token in $GH_TOKEN)
# to $TF_REPO, installs it with `pip install --user --no-deps -e` on top of NERSC's pytorch module, checks that the scoring
# dependencies import (fev, gluonts, ...: tidemark's README section 5a installs them), and runs the fast CPU tests.
set -eu
TF_REPO="${TF_REPO:-$SCRATCH/tideforge/repo}"
mkdir -p "$(dirname "$TF_REPO")"
if [ ! -d "$TF_REPO/.git" ]; then
    : "${GH_TOKEN:?set GH_TOKEN first: export GH_TOKEN=<a GitHub token that can read nareshmeena12/tideforge>}"
    git clone -q "https://$GH_TOKEN@github.com/nareshmeena12/tideforge.git" "$TF_REPO"
    git -C "$TF_REPO" remote set-url origin https://github.com/nareshmeena12/tideforge.git
else
    git -C "$TF_REPO" pull -q --ff-only || echo "git pull failed: run it by hand (it may need GH_TOKEN)"
fi
source "$TF_REPO/scripts/perlmutter/env.sh"
mkdir -p "$TF_ROOT/logs"
pip install --user -q --no-deps -e "$TF_REPO"
pip install --user -q pytest
python - <<'PY'
import importlib
missing = [m for m in ("torch", "numpy", "pandas", "scipy", "yaml", "fev", "datasets", "gluonts", "toolz", "dotenv", "huggingface_hub")
           if importlib.util.find_spec(m) is None]
print("missing:", missing if missing else "none")
if missing:
    raise SystemExit("install them as in tidemark's scripts/perlmutter/README.md section 5a (pip install --user ...)")
PY
python -m pytest -q
git -C "$TF_REPO" log --oneline -1
echo "SETUP OK. GPU parity (on a GPU node): python -m pytest -q -m gpu tests/test_gen_parity.py"
