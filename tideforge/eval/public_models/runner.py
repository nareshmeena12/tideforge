"""Run a public model in its own venv ($TF_ROOT/venvs/<venv>) and score it in the main env.

Remote(name) is a drop-in Model: predict_cell writes the cell to an .npz, a persistent worker in the venv
(`<venv>/bin/python -m tideforge.eval.public_models.runner worker --model NAME`) loads the model once, forecasts the cell and
writes the (n, k, 9, H) quantiles back. Everything else (windows, metrics, windows.parquet, fev evaluation) runs in the main
process exactly as for an in-process adapter, so the output format is the same by construction (tests/eval/test_public_models
checks it). The venv needs the model's pinned package plus numpy, scipy and huggingface_hub; tideforge itself comes from
PYTHONPATH (the repo), never installed into the venv.

Protocol: one JSON line per request on the worker's stdin ({"cell", "out", "mode", "view", "S"}), one JSON line per answer on
its stdout ({"ok": true} or {"error": "..."}). The worker sends everything the libraries print to stderr.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from tideforge.eval.public_models import SPECS
from tideforge.probes.common import Cell
from tideforge.probes.models import Model

REPO = Path(__file__).resolve().parents[3]


def venv_python(venv: str) -> Path:
    root = os.environ.get("TF_VENVS") or (Path(os.environ["TF_ROOT"]) / "venvs" if os.environ.get("TF_ROOT") else None)
    if root is None:
        raise RuntimeError(f"venv {venv!r}: set TF_ROOT (venvs live in $TF_ROOT/venvs) or TF_VENVS")
    p = Path(root) / venv / "bin" / "python"
    if not p.exists():
        raise RuntimeError(f"{p} missing: build it with scripts/perlmutter/e05_public.sh install {venv}")
    return p


class Remote(Model):
    def __init__(self, name: str, device: str = "cuda", python: str | Path | None = None, fake: bool = False):
        self.name, self.device, self.spec = name, device, SPECS[name]
        self.max_context, self.covariates = self.spec.max_context, self.spec.covariates
        self.python = Path(python) if python else venv_python(self.spec.venv)
        self.fake = fake
        self._proc = None
        self._tmp = tempfile.TemporaryDirectory(prefix=f"tf_{name}_")

    def _start(self):
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep))
        cmd = [str(self.python), "-m", "tideforge.eval.public_models.runner", "worker", "--model", self.name,
               "--device", self.device] + (["--fake"] if self.fake else [])
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env, cwd=str(REPO))

    def predict_cell(self, cell, mode="main", view="main"):
        if self._proc is None or self._proc.poll() is not None:
            self._start()
        d = Path(self._tmp.name)
        src, out = d / "cell.npz", d / "q.npy"
        np.savez(src, **cell.arrays)
        self._proc.stdin.write(json.dumps(dict(cell=str(src), out=str(out), mode=mode, view=view, S=int(cell.S))) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError(f"{self.name}: the worker exited (code {self._proc.poll()}); see its stderr")
        ans = json.loads(line)
        if "error" in ans:
            raise RuntimeError(f"{self.name} worker: {ans['error']}")
        return np.load(out)

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.stdin.close()
            self._proc.wait(timeout=60)
        self._tmp.cleanup()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def worker(name: str, device: str, fake: bool) -> int:
    proto = sys.stdout
    sys.stdout = sys.stderr                                   # library prints must not reach the protocol channel
    from tideforge.eval.public_models import make
    if fake:
        from tideforge.eval.public_models.fake import backend_for
        model = make(name, device, backend=backend_for(name), in_process=True)
    else:
        model = make(name, device, in_process=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        r = json.loads(line)
        try:
            with np.load(r["cell"]) as z:
                arrays = {k: z[k] for k in z.files}
            q = model.predict_cell(Cell("remote", arrays, r["S"], modes=(r["mode"],)), r["mode"], r["view"])
            np.save(r["out"], np.asarray(q, np.float64))
            proto.write(json.dumps({"ok": True}) + "\n")
        except Exception as e:                                # reported to the caller, which raises
            proto.write(json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n")
        proto.flush()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("worker",))
    ap.add_argument("--model", required=True, choices=sorted(SPECS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fake", action="store_true", help="tests: the fake backend of public_models.fake")
    a = ap.parse_args(argv)
    return worker(a.model, a.device, a.fake)


if __name__ == "__main__":
    raise SystemExit(main())
