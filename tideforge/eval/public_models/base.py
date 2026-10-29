"""The shared part of every public-model adapter: a cell -> items (targets (k, L), past lanes, known lanes) -> the model's
forecast -> our 9 quantiles (n, k, 9, H).

An adapter implements load() (returns the library object; the weights from a local snapshot of the pinned HF commit) and
forecast(items, H) -> list of (k, n_levels, H) arrays at self.levels. Missing values stay NaN: each library handles them
itself (as in the published fev runs); nothing is filled here.
"""
from __future__ import annotations

import numpy as np

from tideforge.eval.public_models import Q9, SPECS
from tideforge.probes.models import Model


def select_q9(q: np.ndarray, levels, axis: int = 1) -> np.ndarray:
    """The 9 levels 0.1 ... 0.9 out of a model's quantile axis. Raises if any is missing: no interpolation, no sampling."""
    lv = np.asarray(levels, np.float64)
    idx = []
    for x in Q9:
        hit = np.flatnonzero(np.abs(lv - x) < 1e-6)
        if not len(hit):
            raise ValueError(f"quantile {x} not in the model's levels {list(lv)}: refusing to interpolate")
        idx.append(int(hit[0]))
    return np.take(np.asarray(q), idx, axis=axis)


def to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, np.float64)


def snapshot(repo: str, revision: str) -> str:
    """Local directory of the pinned HF commit (downloaded once into the HF cache)."""
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=repo, revision=revision)


class PublicModel(Model):
    levels: tuple = Q9

    def __init__(self, name: str, device: str = "cuda", backend=None):
        self.name, self.device, self.spec = name, device, SPECS[name]
        self.max_context = self.spec.max_context
        self.covariates = self.spec.covariates
        self._backend = backend

    @property
    def backend(self):
        if self._backend is None:
            self._backend = self.load()
        return self._backend

    def load(self):
        raise NotImplementedError

    def forecast(self, items: list[dict], H: int) -> list[np.ndarray]:
        raise NotImplementedError

    def items(self, cell, mode: str, view: str) -> list[dict]:
        if mode == "on" and self.covariates == "none":
            raise ValueError(f"{self.name}: no covariate support, scored in hidden mode only (mode 'on' asked)")
        a, T = cell.arrays, cell.targets(view)
        out = []
        for i in range(cell.n):
            it = dict(targets=np.asarray(T[i], np.float64), past=[], known=[])
            if mode == "on":
                if "past" in a:
                    it["past"] = [np.asarray(p, np.float64) for p in a["past"][i]]
                if "known_ctx" in a:
                    it["known"] = [(np.asarray(c, np.float64), np.asarray(f, np.float64))
                                   for c, f in zip(a["known_ctx"][i], a["known_fut"][i])]
            out.append(it)
        return out

    def predict_cell(self, cell, mode="main", view="main"):
        its, H = self.items(cell, mode, view), cell.H
        q = [select_q9(np.asarray(x, np.float64), self.levels, axis=1) for x in self.forecast(its, H)]
        q = np.stack(q)                                                   # (n, k, 9, H)
        n, k = cell.targets(view).shape[:2]
        if q.shape != (n, k, 9, H):
            raise ValueError(f"{self.name}: forecast shape {q.shape}, expected {(n, k, 9, H)}")
        return np.sort(q, axis=2)


def univariate(items: list[dict]) -> tuple[list[np.ndarray], list[int]]:
    """Flatten (k, L) targets to k univariate series per item (for univariate-only models), with the k per item."""
    ser, ks = [], []
    for it in items:
        ser.extend(list(it["targets"]))
        ks.append(len(it["targets"]))
    return ser, ks


def regroup(q: np.ndarray, ks: list[int]) -> list[np.ndarray]:
    """(sum k, n_levels, H) -> per item (k, n_levels, H)."""
    out, s = [], 0
    for k in ks:
        out.append(q[s:s + k])
        s += k
    return out
