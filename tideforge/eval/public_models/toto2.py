"""Toto-2.0 (Datadog/Toto-2.0-{4m,22m,2.5B}, toto-2 2.0.0; own venv $TF_ROOT/venvs/toto2: Python >= 3.12, gluonts[torch]).
Target only (2.0 has no exogenous inputs: hidden mode only); multivariate: an item's k targets are its variates.
Toto2Model.forecast({"target": (B, V, T), "target_mask", "series_ids": (B, V)}, horizon) -> (9, B, V, H) at 0.1 ... 0.9.
Items of one cell share (k, L); NaN -> value 0, mask False. fp32: the card states fp32 weights and says nothing about bf16
(owner 2026-10-29: bf16 only if the card allows it). Batch sizes as in fev's published runs (512; 2.5B: 256)."""
from __future__ import annotations

import numpy as np

from tideforge.eval.public_models.base import PublicModel, snapshot, to_numpy


class Adapter(PublicModel):
    def load(self):
        import torch
        from toto2 import Toto2Model
        m = Toto2Model.from_pretrained(snapshot(self.spec.repo, self.spec.revision))
        return m.to(torch.device(self.device)).eval()

    def forecast(self, items, H):
        import torch
        x = np.stack([it["targets"] for it in items])                    # (B, V, T)
        ok = np.isfinite(x)
        dev, bs, out = torch.device(self.device), self.spec.kwargs["batch_size"], []
        with torch.inference_mode():
            for s in range(0, len(x), bs):
                xb, mb = x[s:s + bs], ok[s:s + bs]
                q = self.backend.forecast(
                    {"target": torch.tensor(np.where(mb, xb, 0.0), dtype=torch.float32, device=dev),
                     "target_mask": torch.tensor(mb, device=dev),
                     "series_ids": torch.zeros(xb.shape[:2], dtype=torch.long, device=dev)},
                    horizon=H, has_missing_values=bool((~mb).any()))
                q = to_numpy(q)[..., :H]                                   # (9, b, V, H)
                out.extend(np.transpose(q, (1, 2, 0, 3)))                  # b x (V, 9, H)
        return out
