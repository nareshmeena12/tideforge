"""Public-model adapters for E0.5 (Chronos-2, TimesFM-2.5, Toto-2.0, Timestral and the other models with fev + TIME results
and public weights). Stubs until the owner OKs the pinned packages (optional extra `public`, never in the core install):
each name is registered so the scorer and its tests know it, and make() says what is missing.
"""
from __future__ import annotations

from tideforge.probes.models import Model

# name -> (HF model id or source, package to be pinned): filled in with item 6
REGISTRY = {
    "chronos2": ("amazon/chronos-2", "chronos-forecasting"),
    "timesfm25": ("google/timesfm-2.5-200m-pytorch", "timesfm"),
    "toto2_4m": ("Datadog/Toto-2.0-4m", "toto"),
    "toto2_22m": ("Datadog/Toto-2.0-22m", "toto"),
    "timestral5m": ("Timestral-5M", "unknown"),
    "timestral28m": ("Timestral-28M", "unknown"),
    "tirex": ("NX-AI/TiRex", "tirex"),
}


class Stub(Model):
    def __init__(self, name: str, device: str = "cuda"):
        self.name, self.device = name, device

    def predict_cell(self, cell, mode="main", view="main"):
        src, pkg = REGISTRY[self.name]
        raise NotImplementedError(f"{self.name} ({src}): adapter not written yet; needs package {pkg!r} in the `public` extra "
                                  "after the owner's OK (E0.5, item 6)")


def make(name: str, device: str = "cuda") -> Model:
    if name not in REGISTRY:
        raise KeyError(f"unknown public model {name!r} (known: {sorted(REGISTRY)})")
    return Stub(name, device)
