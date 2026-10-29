"""TiRex v1 (NX-AI/TiRex, tirex-ts[cuda] 1.4.2): univariate, no covariates (hidden mode only). load_model(backend="cuda") runs
the xlstm sLSTM kernels. tirex itself falls back to the torch backend with only a warning when xlstm is missing, so this
adapter checks first and raises instead (owner 2026-10-29: never fall back to pure torch silently).
forecast(context, prediction_length, output_type="numpy") -> quantiles (B, H, 9) at 0.1 ... 0.9, mean."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from tideforge.eval.public_models.base import PublicModel, regroup, snapshot, univariate

NO_KERNELS = ("tirex: the CUDA sLSTM kernels are required (tirex-ts[cuda]: xlstm + ninja) and no torch fallback is allowed "
              "(owner 2026-10-29): stop and tell the owner")


class Adapter(PublicModel):
    def load(self):
        import torch
        if importlib.util.find_spec("xlstm") is None or not torch.cuda.is_available():
            raise RuntimeError(NO_KERNELS + " (xlstm missing or no GPU)")
        from tirex import load_model
        ckpt = Path(snapshot(self.spec.repo, self.spec.revision)) / "model.ckpt"
        m = load_model(str(ckpt), device=self.device, backend="cuda")
        be = getattr(m, "backend", None)
        if be is not None and str(be) != "cuda":
            raise RuntimeError(NO_KERNELS + f" (loaded with backend {be!r})")
        return m

    def forecast(self, items, H):
        import torch
        ser, ks = univariate(items)
        q, _ = self.backend.forecast(context=[torch.tensor(s, dtype=torch.float32) for s in ser], prediction_length=H,
                                     output_type="numpy")
        q = np.asarray(q, np.float64)[:, :H, :]                            # (B, H, 9)
        return regroup(np.transpose(q, (0, 2, 1)), ks)
