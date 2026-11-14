"""ft3-architecture checkpoints (ft3_main, cov_r3, tideforge 5M / 30M runs) through the existing inference code, not rewritten:
ft3_core.forecast_left_padded (univariate) and ft3_core.forecast_groups (groups ON), the model built by tideforge.model.load
(strict; M1 / gated / adapter options from the checkpoint's cfg). Inputs are prepared as the old scorer did: the last 8192
points, history gaps filled one season back (score_twins.fill_season); bf16 autocast on CUDA inside ft3_core, as before.

Batching: univariate series are bucketed by padded length here (forecast_left_padded buckets the same way inside, so the
result does not change) and each bucket gets its own batch size from a token budget; groups get a row count from the same
budget. The budget comes from free GPU memory and one measured probe forward (peak bytes per token at the longest context,
an upper bound for shorter ones); --batch-size overrides it (series per batch, or rows per batch for groups). An OOM halves
the batch and retries.
"""
from __future__ import annotations

import numpy as np

from tideforge.score.adapters.base import Predictor

MAX_CTX = 8192
SAFETY = 0.6                    # share of free memory the budget may use


def _writable(x):
    """Writable copies of every array in x (nested lists / tuples / dicts kept). The window cache hands out read-only
    memory-mapped views, and ft3_core.forecast_groups fills NaN in place (f[i] = last); forecast_left_padded gets copies
    too in case it does the same."""
    if isinstance(x, np.ndarray):
        return np.array(x, copy=True)
    if isinstance(x, (list, tuple)):
        return type(x)(_writable(v) for v in x)
    if isinstance(x, dict):
        return {k: _writable(v) for k, v in x.items()}
    return x


class Ft3(Predictor):
    def __init__(self, path, device=None, batch_size: int | None = None, model=None, name: str | None = None, log=print):
        import torch
        from tideforge.eval import thirdparty
        self.C, self.S_ = thirdparty.ft3_core(), thirdparty.score_twins()
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            from tideforge import model as TM
            model, cfg, ck = TM.load(path, self.device)
            self.cfg, self.step = cfg, ck.get("step")
        self.model = model.eval()
        self.name = name or "ft3"
        self.max_context = MAX_CTX
        self.groups = bool(getattr(model, "group_attn", False))
        self.pl = int(model.patch_len)
        self.batch_size = batch_size
        self._tok = None
        self.log = log

    def groups_ok(self, horizon: int) -> bool:
        return self.groups and -(-horizon // self.pl) <= int(getattr(self.model, "max_pass_patches", 32))

    def prepare(self, past, season):
        return self.S_.fill_season(np.asarray(past, dtype=np.float64)[-self.max_context:], season)

    # ---- batch size ------------------------------------------------------------------------------------------------------
    def tokens(self) -> int:
        """Token budget per forward (rows x padded length)."""
        if self._tok is not None:
            return self._tok
        import torch
        if self.device.type != "cuda":
            self._tok = 512 * 1024
            return self._tok
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info(self.device)
        b, L = 16, MAX_CTX
        H = self.pl * min(int(getattr(self.model, "max_pass_patches", 32)), 8)
        torch.cuda.reset_peak_memory_stats(self.device)
        base = torch.cuda.memory_allocated(self.device)
        self.C.forecast_left_padded(self.model, [np.random.default_rng(0).normal(size=L).astype(np.float32)] * b, H,
                                    self.device, max_ctx=MAX_CTX, batch_size=b)
        per = max(1.0, (torch.cuda.max_memory_allocated(self.device) - base) / (b * (L + H)))
        self._tok = int(SAFETY * free / per)
        self.log(f"batch budget: {free / 2 ** 30:.1f} GiB free, {per:.0f} B/token measured -> {self._tok:,} tokens per forward "
                 f"({self._tok // (L + H)} series at {L} points)")
        return self._tok

    def _retry(self, fn, n: int):
        import torch
        while True:
            try:
                return fn(n)
            except torch.cuda.OutOfMemoryError:
                if n <= 1:
                    raise
                torch.cuda.empty_cache()
                n //= 2
                self._tok = None if self.batch_size else max(1, (self._tok or 2) // 2)
                self.log(f"OOM: retrying with batch {n}")

    # ---- forecasts -------------------------------------------------------------------------------------------------------
    def predict(self, contexts, horizon, freq="", past_covariates=None, known_covariates=None, season=1):
        if past_covariates is not None or known_covariates is not None:
            return self._groups(contexts, past_covariates or [[]] * len(contexts),
                                known_covariates or [[]] * len(contexts), horizon)
        pl, C = self.pl, self.max_context
        hp = -(-horizon // pl) * pl
        nfin = [int(np.isfinite(np.asarray(c)).sum()) for c in contexts]
        Lp = np.array([max(pl, -(-min(max(n, 1), C) // pl) * pl) for n in nfin])
        out = np.zeros((len(contexts), len(self.quantile_levels), horizon), np.float32)
        for L in sorted(set(Lp.tolist())):
            idx = np.flatnonzero(Lp == L)
            bs = self.batch_size or max(1, self.tokens() // (L + hp))
            ser = [_writable(contexts[i]) for i in idx]
            _, q = self._retry(lambda n: self.C.forecast_left_padded(self.model, ser, horizon, self.device, max_ctx=C,
                                                                     batch_size=n), bs)
            out[idx] = q
        return np.sort(out, axis=1)

    def _groups(self, targets, past, known, horizon):
        pl, C = self.pl, self.max_context
        groups = [dict(targets=_writable(t), past=_writable(p), known=_writable(k)) for t, p, k in zip(targets, past, known)]
        Lc = max((min(C, len(np.asarray(x))) for g in targets for x in g), default=pl)
        W = (-(-Lc // pl) + -(-horizon // pl)) * pl
        rows = self.batch_size or max(1, self.tokens() // W)
        res = self._retry(lambda n: self.C.forecast_groups(self.model, groups, horizon, self.device, max_ctx=C, rows=n), rows)
        return [np.sort(np.asarray(q, dtype=np.float32), axis=1) for _, q in res]
