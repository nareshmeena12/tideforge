#!/usr/bin/env bash
# tfbench build machine setup (JarvisLabs, Linux). Run from the repo root:  bash scripts/jarvis/setup.sh
# Needs HF_TOKEN and TF_BENCH_REPO in the environment (never printed). Data goes to $TFB_DATA (default /home/tfbench).
set -euo pipefail
: "${TFB_DATA:=/home/tfbench}"
export TFB_DATA
if [ -z "${HF_TOKEN:-}" ]; then echo "HF_TOKEN is not set" >&2; exit 1; fi
if [ -z "${TF_BENCH_REPO:-}" ]; then echo "warning: TF_BENCH_REPO is not set (needed in Milestone B only)" >&2; fi
if [ -n "$(git status --porcelain)" ]; then echo "git tree is not clean; commit or stash first" >&2; git status --short; exit 1; fi
python3 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q -e ".[tfbench,dev]"
export HF_HUB_ENABLE_HF_TRANSFER=1
python - <<'PY'
import hf_transfer, huggingface_hub, pyarrow, pandas
try:
    import importlib.metadata as md; print("hf_xet", md.version("hf_xet"))
except Exception as e:
    print("hf_xet not installed:", e)
from huggingface_hub import HfApi
who = HfApi().whoami()            # uses HF_TOKEN; prints only the account name
print("hf account:", who.get("name"), "| huggingface_hub", huggingface_hub.__version__, "| pyarrow", pyarrow.__version__)
PY
# the files list is committed gzipped; tfbench reads the .gz directly
mkdir -p "$TFB_DATA"
df -h "$TFB_DATA"
nproc; free -g | head -2
python -m pytest tests/tfbench -q
echo "setup ok. Next: see the commands in the handover (python -m tideforge.tfbench.run ...)."
