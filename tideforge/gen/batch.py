"""Generator v0: _allocate and generate_batch (top-level API).

Ported from third_party/tidemark_e21cbf0/tidemark/real/ft3_core.py L2319-L2398 (tidemark commit e21cbf0) by tools/port_gen_v0.py; do not edit by hand
(edit and rerun the tool). Edits vs the source: imports only; the profile globals read / written as _state.NAME; generate_batch's `global` line is a comment; the hook points (`# [hook Hn]`, tideforge/gen/_hooks.py).
Must stay bit-exact with ft3_core (tests/test_gen_parity.py, tests/test_gen_source.py).
"""
# flake8: noqa
import math

import torch
import torch.nn.functional as F

from tideforge.gen import _state
from tideforge.gen import _hooks
from tideforge.gen.primitives import (  # noqa: F401
    PERIOD_BANK,
)
from tideforge.gen.families import (  # noqa: F401
    _L_breaks,
)
from tideforge.gen.profiles import (  # noqa: F401
    FAMILIES,
    GEN_PROFILES,
    PASSTHROUGH_FAMILIES,
)
from tideforge.gen.layers import (  # noqa: F401
    _L_calendar,
    _L_corruption,
    _L_noise,
    _L_scale,
    _L_seasonality,
    _L_support,
    _L_trend,
    _finalize,
)


# >>> ft3_core.py L2319-L2398
# ============================ top-level API ============================

def _allocate(B, weights):
    """Fixed per-family row counts for a batch. Deterministic given (B, weights), which
    keeps tensor shapes static across steps -- important so torch.compile / CUDA graphs
    don't recompile every step."""
    names = [n for n in weights if weights[n] > 0]
    tot = sum(weights[n] for n in names)
    counts, acc = [], 0
    for n in names[:-1]:
        c = int(round(B * weights[n] / tot))
        counts.append((n, c))
        acc += c
    counts.append((names[-1], max(0, B - acc)))
    return counts


@torch.no_grad()
def generate_batch(B, L, device, weights=None, apply_layers=True, profile="v1"):
    """Generate a batch of diverse synthetic series.

    Parameters
    ----------
    B : int     batch size (rows)
    L : int     series length (should be ctx_len + pred_len for training)
    device      torch device
    weights     optional {family_name: weight}; defaults to DEFAULT_WEIGHTS
    apply_layers : set False to get raw base families with no layering (for debugging /
                   ablating how much the layer stack actually contributes)

    Returns (B, L) float32 on `device`. Rows are shuffled so families are not contiguous
    in the batch (which would otherwise correlate with position in a DDP shard).
    """
    # [port] `global` statement removed: these four names live in tideforge/gen/_state.py
    _hooks.tick("batch")  # [hook tick]
    pr = GEN_PROFILES[profile]
    weights = weights or pr["weights"]
    _state._PERIOD_BANK_ACTIVE = pr.get("period_bank", PERIOD_BANK)
    _state._PERIOD_EXT_SHARE = pr.get("period_ext_share", None)
    _state._TV_SEASON = bool(pr.get("tv_season", False))
    _state._ONE_OFF_P = float(pr.get("one_off_p", 0.0))
    xs, cnt_f, nn_f, add_f, pt_f = [], [], [], [], []
    for name, n in _allocate(B, weights):
        if n <= 0:
            continue
        fn, is_cnt, is_nn, allow_add, _ = FAMILIES[name]
        xs.append(fn(n, L, device))
        cnt_f.append(torch.full((n,), is_cnt, device=device, dtype=torch.bool))
        nn_f.append(torch.full((n,), is_nn, device=device, dtype=torch.bool))
        add_f.append(torch.full((n,), allow_add, device=device, dtype=torch.bool))
        pt_f.append(torch.full((n,), name in PASSTHROUGH_FAMILIES, device=device, dtype=torch.bool))

    x = torch.cat(xs, dim=0)
    passthrough = torch.cat(pt_f, dim=0).unsqueeze(-1)
    # guard: a single non-finite row (numerically extreme parameter draw in ANY family) must
    # never reach torch.poisson, which hard-asserts on GPU and kills the whole run
    bad_rows = ~torch.isfinite(x).all(dim=1, keepdim=True)
    x = torch.where(bad_rows, torch.randn_like(x).cumsum(1), x)
    is_count = torch.cat(cnt_f, dim=0)
    is_nonneg = torch.cat(nn_f, dim=0)
    allow_add = torch.cat(add_f, dim=0)
    x, is_count, is_nonneg, allow_add = _hooks.run_batch("after_families", x, is_count, is_nonneg, allow_add, passthrough, device, pr, L)  # [hook H2]

    if apply_layers:
        x_raw, nn_raw = x, is_nonneg
        x = _hooks.layer("seasonality", _L_seasonality)(x, allow_add, device)  # [hook H1]
        x = _hooks.layer("trend", _L_trend)(x, allow_add, device)  # [hook H1]
        x = _hooks.layer("calendar", _L_calendar)(x, allow_add, device)  # [hook H1]
        if pr.get("breaks_p", 0.0) > 0:
            x = _hooks.layer("breaks", _L_breaks)(x, device, pr["breaks_p"])  # [hook H1]
        x, is_count, is_nonneg, allow_add = _hooks.run_batch("after_structure", x, is_count, is_nonneg, allow_add, passthrough, device, pr, L)  # [hook H3]
        x = _hooks.layer("noise", _L_noise)(x, is_count, device)  # [hook H1]
        x, is_count, is_nonneg = _hooks.layer("support", _L_support)(x, is_count, is_nonneg, device, pr["support_probs"])  # [hook H1]
        x = _hooks.layer("corruption", _L_corruption)(x, is_count, is_nonneg, device, pr["out_p"], pr["miss_p"], pr["quant_p"],  # [hook H1]
                          pr["quant_step"], pr["clean_p"])
        x = torch.where(passthrough, x_raw, x)
        is_nonneg = torch.where(passthrough.squeeze(-1), nn_raw, is_nonneg)
        x = _hooks.layer("scale", _L_scale)(x, is_count, is_nonneg, device)  # [hook H1]

    x = _hooks.layer("finalize", _finalize)(x, pr["fix_degenerate"])  # [hook H1]
    x, is_count, is_nonneg, allow_add = _hooks.run_batch("final", x, is_count, is_nonneg, allow_add, passthrough, device, pr, L)  # [hook H4]
    return x[torch.randperm(x.shape[0], device=device)]


# <<< ft3_core.py
