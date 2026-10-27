"""The one predict interface and its adapters.

    predict(model, context, covariates, horizon) -> (k, 9, H)        one item: context (k, L) or (L,); covariates None or
                                                                    {"past": [(L,)], "known": [((L,), (H,))]}
    model.predict_cell(cell, mode, view) -> (n, k, 9, H)             a whole cell, batched (what the runner calls)

mode: "main" (no covariates in the cell), "on" (covariates shown, ft3 groups ON) or "hidden" (targets only, groups hidden).
Adapters: Ft3 (ft3_main and any ft3 checkpoint, through the scorer's path: tideforge.eval.thirdparty -> fill_season ->
ft3_core.forecast_left_padded / forecast_groups, under a NaN variant of tideforge.eval.nan_variants), M5 (stub until T3),
Oracle (the stored true quantiles) and Naive (seasonal naive with empirical residual quantiles).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from tideforge.probes.common import Q9, Cell


def _items(cell: Cell, mode: str, view: str):
    """Per item: (targets (k, L), covariates or None)."""
    a = cell.arrays
    T = cell.targets(view)
    for i in range(cell.n):
        cov = None
        if mode == "on":
            cov = {"past": list(a["past"][i]) if "past" in a else [],
                   "known": [(c, f) for c, f in zip(a["known_ctx"][i], a["known_fut"][i])] if "known_ctx" in a else []}
        yield T[i], cov


class Model:
    name = "model"

    def predict_cell(self, cell: Cell, mode: str = "main", view: str = "main") -> np.ndarray:
        raise NotImplementedError

    def predict(self, context, covariates, horizon: int, S: int = 1) -> np.ndarray:
        ctx = np.atleast_2d(np.asarray(context, dtype=np.float32))
        arrays = {"targets": ctx[None], "future": np.zeros((1, ctx.shape[0], horizon))}
        mode = "main"
        if covariates:
            mode = "on"
            if covariates.get("past"):
                arrays["past"] = np.asarray(covariates["past"], np.float32)[None]
            if covariates.get("known"):
                arrays["known_ctx"] = np.asarray([c for c, _ in covariates["known"]], np.float32)[None]
                arrays["known_fut"] = np.asarray([f for _, f in covariates["known"]], np.float32)[None]
        return self.predict_cell(Cell("item", arrays, S, modes=(mode,)), mode)[0]


def predict(model: Model, context, covariates, horizon: int, S: int = 1) -> np.ndarray:
    return model.predict(context, covariates, horizon, S)


class Oracle(Model):
    """The generator's true predictive quantiles; in mode "hidden" the covariate-blind reference where the cell has one."""
    name = "oracle"

    def predict_cell(self, cell, mode="main", view="main"):
        a = cell.arrays
        return a["oracle_hidden_q"] if mode == "hidden" and "oracle_hidden_q" in a else a["oracle_q"]


class Naive(Model):
    """Seasonal naive (season S): the last season repeated; quantiles = last-season values + empirical quantiles of the
    context's seasonal differences, widened by sqrt(number of seasons ahead). Missing values are carried forward."""
    name = "naive"

    def predict_cell(self, cell, mode="main", view="main"):
        T = cell.targets(view).astype(np.float64)
        n, k, L = T.shape
        H = cell.H
        x = T.reshape(n * k, L).copy()
        for r in x:                                           # carry the last value forward over NaN
            ok = np.isfinite(r)
            if not ok.any():
                r[:] = 0.0
                continue
            idx = np.where(ok, np.arange(L), 0)
            np.maximum.accumulate(idx, out=idx)
            r[:] = r[idx]
            r[:np.argmax(ok)] = r[np.argmax(ok)]
        seasons = np.repeat(cell.seasons(), k)
        q = np.empty((n * k, 9, H))
        for s in np.unique(seasons):
            rows = seasons == s
            S = max(1, min(int(s), L - 1))
            xs = x[rows]
            rep = np.tile(xs[:, L - S:], (1, -(-H // S)))[:, :H]
            rq = np.quantile(xs[:, S:] - xs[:, :-S], Q9, axis=1).T                 # (rows, 9)
            steps = np.sqrt(np.ceil(np.arange(1, H + 1) / S))
            q[rows] = rep[:, None, :] + rq[:, :, None] * steps[None, None, :]
        return np.sort(q.reshape(n, k, 9, H), axis=2)


class M5(Model):
    """5M checkpoints (the ab5m_cpm recipe): filled in with the T3 port."""
    name = "m5"

    def __init__(self, ckpt: str | Path, device: str = "cuda"):
        self.ckpt, self.device = ckpt, device

    def predict_cell(self, cell, mode="main", view="main"):
        raise NotImplementedError("5M adapter: written with the T3 recipe port")


class Ft3(Model):
    """An ft3 checkpoint through the scorer's path. mode "on": ft3_core.forecast_groups with the cell's covariates (past and
    known lanes, as fev's groups mode passes them); "hidden" / "main": forecast_left_padded on the targets (groups hidden).
    History is filled with score_twins.fill_season(y[-max_ctx:], S) first, exactly like the scorer; nan_variant swaps the
    fill / inference functions as in E0.3 (baseline, nofill_delete, nofill_mask)."""

    def __init__(self, ckpt: str | Path, device: str = "cuda", nan_variant: str = "baseline", name: str = "ft3"):
        from tideforge.eval import thirdparty
        thirdparty.add_path()
        import tidemark.eval.score_twins as S_
        import tidemark.real.ft3_core as C
        from tidemark.real.score_ft3 import load_ft3
        self.C, self.S_ = C, S_
        import torch
        self.device = torch.device(device)                    # ft3_core reads device.type
        self.model, self.cfg, _ = load_ft3(str(ckpt), self.device)
        self.nan_variant, self.name = nan_variant, name
        self.max_ctx, self.batch = int(S_.MAX_CTX), int(S_.BATCH)
        self.ckpt_sha = sha256_file(Path(ckpt))

    def predict_cell(self, cell, mode="main", view="main"):
        from tideforge.eval.nan_variants import use_variant
        n, k, L = cell.targets(view).shape
        H = cell.H
        with use_variant(self.nan_variant):
            fill = self.S_.fill_season                        # looked up inside the variant (it may be swapped)
            if mode == "on":
                groups = []
                for (tg, cov), s in zip(_items(cell, mode, view), cell.seasons()):
                    groups.append(dict(targets=[fill(np.asarray(t, np.float64)[-self.max_ctx:], int(s)) for t in tg],
                                       past=[np.asarray(p, np.float64)[-self.max_ctx:] for p in cov["past"]],
                                       known=[(np.asarray(c, np.float64)[-self.max_ctx:], np.asarray(f, np.float64))
                                              for c, f in cov["known"]]))
                res = self.C.forecast_groups(self.model, groups, H, self.device, max_ctx=self.max_ctx)
                q = np.stack([np.asarray(r[1], np.float64) for r in res])          # (n, k, 9, H)
            else:
                ys = [fill(np.asarray(t, np.float64)[-self.max_ctx:], int(s))
                      for tg, s in zip(cell.targets(view), cell.seasons()) for t in tg]
                _, q = self.C.forecast_left_padded(self.model, ys, H, self.device, max_ctx=self.max_ctx,
                                                   batch_size=self.batch)
                q = np.asarray(q, np.float64).reshape(n, k, 9, H)
        return np.sort(q, axis=2)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def make(name: str, ckpt: str | None = None, device: str = "cuda", nan_variant: str = "baseline") -> Model:
    if name == "oracle":
        return Oracle()
    if name == "naive":
        return Naive()
    if name == "m5":
        return M5(ckpt, device)
    if ckpt is None:
        raise ValueError(f"model {name!r} needs --ckpt")
    return Ft3(ckpt, device, nan_variant, name=name)
