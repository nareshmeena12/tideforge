"""Stage B: the model-size axis (5M, 30M, 100M, optional 200M). Everything here is derived from one table.

Scaling rule: head dimension stays 64 (d_model = 64 x heads, so GLA / GDN keys, d_model/2 over the same heads, have
head dim 32, which the fla kernels take) and the width-to-depth ratio stays near the 30M twin's 85 : 1; the FFN is 4 x d.
Parameters grow like 12 d^2 L per mixer+FFN. Everything else (patch 32, context 8192, group attention with 8 lanes,
d_kv 64, dropout, quantiles, loss) is identical at every size.

Learning rate: peak = 3e-4 x sqrt(512 / d_model), rounded to 2 significant figures (the 30M twin's pilot
rates 1.5e-4 / 3e-4 are the d = 512 case); the pilot sweeps {0.5, 1, 2} x that value. Warmup stays 2% of the schedule.

Token budget: 100B points (= 3.125B patch-tokens) at 5M, 30M and 100M; the optional 200M model trains on 200B points
(about 30 tokens per parameter). Every Stage B run keeps 4 ranks, so the data order matches the 30M twins.
"""
from __future__ import annotations

import math
from dataclasses import replace

from tidemark.pretrain.config import TwinConfig, TwinModel

SIZES = ("5m", "30m", "100m", "200m")
DEFAULT_SIZES = ("5m", "100m")            # Stage B; 30M is Stage A / M4; 200M only behind --with-200m
SHAPES = {                                # d_model, layers, heads, d_ff
    "5m": (256, 4, 4, 1024),
    "30m": (512, 6, 8, 2048),
    "100m": (768, 12, 12, 3072),
    "200m": (1024, 14, 16, 4096),
}
TARGET_POINTS = {"5m": 100e9, "30m": 100e9, "100m": 100e9, "200m": 200e9}
PILOT_FRAC = 0.05                          # a pilot is at most 5% of the point budget (owner)
PILOT_MULTS = (0.5, 1.0, 2.0)
# arms piloted at each size: softmax and gla, plus gdn (its own 3 rates); 200M: softmax and gla only
PILOT_ARMS = {"5m": ("softmax", "gla", "gdn"), "100m": ("softmax", "gla", "gdn"), "200m": ("softmax", "gla")}
SIZE_ARMS = {"5m": ("softmax", "gla", "gdn", "hybrid"), "100m": ("softmax", "gla", "gdn", "hybrid"),
             "200m": ("softmax", "gla")}
PATCH = 32


def size_model(size: str) -> TwinModel:
    d, layers, heads, ff = SHAPES[size]
    return replace(TwinModel(), d_model=d, n_layers=layers, n_heads=heads, d_ff=ff)


def size_of(cfg: TwinConfig) -> str:
    """The size of a config, by its model; anything else (tiny test models) counts as the 30M default."""
    for s in SIZES:
        if size_model(s) == cfg.model:
            return s
    return "30m"


def base_lr(size: str) -> float:
    d = SHAPES[size][0]
    v = 3e-4 * math.sqrt(512 / d)
    e = math.floor(math.log10(v))
    return round(v / 10 ** e, 1) * 10 ** e


def pilot_lrs(size: str) -> tuple[float, ...]:
    return tuple(float(f"{base_lr(size) * m:.2g}") for m in PILOT_MULTS)


def pilot_points(size: str) -> float:
    return PILOT_FRAC * TARGET_POINTS[size]


def sized_config(arm: str, seed: int, lr: float, size: str = "30m", **over) -> TwinConfig:
    """The twin of one arm at one size. size '30m' is exactly twin_config (same config, same run id)."""
    from tidemark.pretrain.config import twin_config
    cfg = twin_config(arm, seed, lr, **over)
    if size == "30m":
        return cfg
    return replace(cfg, model=size_model(size), train=replace(cfg.train, target_points=TARGET_POINTS[size]))


def params_millions(size: str) -> float:
    from tidemark.pretrain.model import build_twin, param_report
    return param_report(build_twin(sized_config("softmax", 0, 3e-4, size), gla_backend="reference"))["params_total"] / 1e6


def tokens_per_param(size: str, params: float | None = None) -> float:
    """Patch-tokens seen per parameter (points / patch length / parameters)."""
    p = params if params is not None else params_millions(size) * 1e6
    return TARGET_POINTS[size] / PATCH / p
