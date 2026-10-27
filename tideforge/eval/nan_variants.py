"""E0.3 inference variants for missing values, swapped into ft3_core / score_twins in memory for one scoring run.

  baseline       fill_season + forecast_left_padded (deletes NaN) + _lane_array (forward-fills NaN): the scorers as they are
  nofill_delete  history gaps are NOT filled (leading NaNs dropped only); forecast_left_padded deletes the NaNs (phase shift)
  nofill_mask    history gaps are NOT filled; a NaN becomes value 0 with mask False, in the univariate path and in group lanes

use_variant(name) replaces score_twins.fill_season, ft3_core.forecast_left_padded and ft3_core._lane_array (all looked up by
attribute at call time in score_groups / score_twins / forecast_groups) and restores the originals on exit, also on an exception.
Nothing under third_party/ is edited. Known limits:
  * forecast_groups' own univariate branch (one target, no covariate) deletes NaNs inline; the scorers never reach it (fev sends
    such tasks to forecast_left_padded, TIME groups have >= 2 targets).
  * forecast_groups forward-fills NaNs of the known FUTURE in every variant; under nofill_mask it starts from 0 when the last
    context point of a known covariate is missing (the masked lane holds 0 there).
  * ft3_main never saw single masked points inside a series in training (only left padding and whole CPM-hidden patches), so
    nofill_mask measures the deletion effect for this model, not whether masking helps a model trained with masks (G6).
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch

from tideforge.eval import thirdparty

VARIANTS = ("baseline", "nofill_delete", "nofill_mask")
CODE = {"baseline": "base", "nofill_delete": "nfdel", "nofill_mask": "nfmask"}       # checkpoint-name suffix (tags: [a-z0-9]+)


def variant_of(name: str) -> str | None:
    """The variant whose code ends `name` (ft3mainnfmask -> nofill_mask), else None."""
    for v, c in sorted(CODE.items(), key=lambda kv: -len(kv[1])):
        if name.removesuffix("on_s0").removesuffix("_s0").endswith(c):
            return v
    return None


def strip_leading_nan(a, S):
    """fill_season without the filling: leading NaNs dropped, interior NaNs kept (S unused; same signature)."""
    y = np.asarray(a, dtype=np.float64)
    ok = np.isfinite(y)
    if ok.all() or not ok.any():
        return y.astype(np.float32)
    return y[int(np.argmax(ok)):].astype(np.float32)


@torch.no_grad()
def forecast_left_padded_masked(model, series_list, horizon, device, max_ctx=8192, batch_size=256):
    """ft3_core.forecast_left_padded (L2907-L2974) with ONE change: NaNs are not deleted. Leading NaNs are dropped (they are left
    padding); every other non-finite value becomes value 0 with mask False, at its own position."""
    model.eval()
    pl, nq, q_mid = model.patch_len, model.n_q, model.n_q // 2
    is_dec = getattr(model, "is_decoder", False)
    N = len(series_list)
    point = np.zeros((N, horizon), dtype=np.float32)
    quant = np.zeros((N, nq, horizon), dtype=np.float32)
    amp_on = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (amp_on and torch.cuda.is_bf16_supported()) else torch.float16

    def pad_left(y):
        y = y[-max_ctx:]
        ok = np.isfinite(y)                                   # [change] observed = finite
        Lp = max(pl, ((len(y) + pl - 1) // pl) * pl)
        v = np.zeros(Lp, dtype=np.float32); m = np.zeros(Lp, dtype=bool)
        v[Lp - len(y):] = np.where(ok, y, 0.0); m[Lp - len(y):] = ok
        return v, m

    buckets = {}
    for i, y in enumerate(series_list):
        y = np.asarray(y, dtype=np.float32)
        ok = np.isfinite(y)                                   # [change] was: y = y[np.isfinite(y)]
        y = y[int(np.argmax(ok)):] if ok.any() else y[:0]
        if len(y) == 0:
            y = np.zeros(1, dtype=np.float32)
        v, m = pad_left(y)
        buckets.setdefault(len(v), []).append((i, v, m))

    for _, items in buckets.items():
        for c0 in range(0, len(items), batch_size):
            chunk = items[c0:c0 + batch_size]
            idx = [i for i, _, _ in chunk]
            cur = torch.from_numpy(np.stack([v for _, v, _ in chunk])).to(device)
            msk = torch.from_numpy(np.stack([m for _, _, m in chunk])).to(device)
            rem, qs = horizon, []
            while rem > 0:
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_on):
                    if is_dec:   # all horizon patches from ONE pass over blanked patches
                        step = min(rem, model.max_pass_patches * pl)
                        qp = model.predict_horizon(cur, msk, step).float()
                    else:
                        qp = model(cur, mask=msk)["quantile_preds"].float()   # (b, pred, nq)
                take = min(qp.shape[1], rem)
                qs.append(qp[:, :take])
                rem -= take
                if rem > 0:     # roll forward on the median, re-left-pad to a patch boundary
                    med = qp[:, :take, q_mid]
                    cur = torch.cat([cur, med], 1); msk = torch.cat([msk, torch.ones_like(med, dtype=torch.bool)], 1)
                    keep = min(max_ctx, ((cur.shape[1] + pl - 1) // pl) * pl)
                    if cur.shape[1] >= keep:
                        cur, msk = cur[:, -keep:], msk[:, -keep:]
                    else:
                        padn = keep - cur.shape[1]
                        cur = torch.cat([torch.zeros(cur.shape[0], padn, device=device), cur], 1)
                        msk = torch.cat([torch.zeros(msk.shape[0], padn, dtype=torch.bool, device=device), msk], 1)
            qf = torch.cat(qs, 1).cpu().numpy()                             # (b, horizon, nq)
            for bi, i in enumerate(idx):
                quant[i] = qf[bi].T
                point[i] = qf[bi, :, q_mid]
    return point, quant


def lane_array_masked(a, n):
    """ft3_core._lane_array (L2976-L2992) with ONE change: interior NaNs are left unobserved (value 0, observed False) instead
    of forward-filled. Leading NaNs stay unobserved, as before."""
    a = np.asarray(a, dtype=np.float32)[-n:]
    if len(a) < n:
        a = np.concatenate([np.full(n - len(a), np.nan, np.float32), a])
    ok = np.isfinite(a)
    if not ok.any():
        return np.zeros(n, np.float32), np.zeros(n, bool)
    v = np.where(ok, a, 0.0).astype(np.float32)                 # [change] was: forward fill
    return v, ok


def _swaps(name: str) -> dict:
    C, S = thirdparty.ft3_core(), thirdparty.score_twins()
    if name == "baseline":
        return {}
    if name == "nofill_delete":
        return {(S, "fill_season"): strip_leading_nan}
    if name == "nofill_mask":
        return {(S, "fill_season"): strip_leading_nan, (C, "forecast_left_padded"): forecast_left_padded_masked,
                (C, "_lane_array"): lane_array_masked}
    raise KeyError(f"unknown variant {name!r}; valid: {VARIANTS}")


@contextmanager
def use_variant(name: str):
    """Swap the variant's functions in for the duration of the block; the originals are always restored."""
    swaps = _swaps(name)
    saved = {k: getattr(*k) for k in swaps}
    try:
        for (mod, attr), fn in swaps.items():
            setattr(mod, attr, fn)
        yield name
    finally:
        for (mod, attr), fn in saved.items():
            setattr(mod, attr, fn)
