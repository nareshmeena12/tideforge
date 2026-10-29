"""E0.3 inference variants for missing values, swapped into ft3_core / score_twins in memory for one scoring run.

  baseline       fill_season + forecast_left_padded (deletes NaN) + _lane_array (forward-fills NaN): the scorers as they are
  nofill_delete  history gaps are NOT filled (leading NaNs dropped only); forecast_left_padded deletes the NaNs (phase shift)
  nofill_mask    history gaps are NOT filled; a NaN becomes value 0 with mask False, in the univariate path and in group lanes
  cov_mask       groups ON only: COVARIATE NaN are marked unobserved (value 0, mask False) instead of forward-filled, in the
                 history (lane_array_masked) and in a known covariate's future (forecast_groups_covmask); the targets are
                 filled exactly as in baseline (fill_season). E0.3 addition, 2026-10-27: 92% of fev's 36M covariate NaN are
                 in known-covariate history (retail), which baseline forward-fills into long flat stretches.

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

from tideforge.eval import cov_variants  # noqa: E402  (E0.10's covariate variants share this switch and the tag codes)

VARIANTS = ("baseline", "nofill_delete", "nofill_mask", "cov_mask") + cov_variants.VARIANTS
CODE = {"baseline": "base", "nofill_delete": "nfdel", "nofill_mask": "nfmask", "cov_mask": "cmask", **cov_variants.CODE}       # checkpoint-name suffix (tags: [a-z0-9]+)


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


@torch.no_grad()
def forecast_groups_covmask(model, groups, horizon, device, max_ctx=8192, rows=256, max_variates=None):
    """ft3_core.forecast_groups (L2996-L3120), copied, with ONE change (marked [change]): a known covariate's future NaN are
    NOT forward-filled; they become value 0 with observed False (baseline marks the whole known future observed). History
    lanes go through ft3_core._lane_array as looked up at call time (cov_mask swaps in lane_array_masked)."""
    C = thirdparty.ft3_core()
    model.eval()
    pl, nq, q_mid = model.patch_len, model.n_q, model.n_q // 2
    hp = -(-horizon // pl)
    cap = int(max_variates or getattr(model, "group_lanes", 8))
    res = [None] * len(groups)
    if hp > getattr(model, "max_pass_patches", 32) or not getattr(model, "group_attn", False):
        for gi, gr in enumerate(groups):
            res[gi] = C.forecast_left_padded(model, gr["targets"], horizon, device, max_ctx=max_ctx)
        return res
    units = []
    for gi, gr in enumerate(groups):
        tg, past, known = gr["targets"], gr.get("past", []), gr.get("known", [])
        nt = len(tg)
        res[gi] = (np.zeros((nt, horizon), np.float32), np.zeros((nt, nq, horizon), np.float32))
        n_cov = min(len(known) + len(past), cap - 1 if nt == 1 else cap // 2)
        kn, ps = known[:n_cov], past[:max(0, n_cov - len(known))]
        if nt == 1 and n_cov == 0:
            y = np.asarray(tg[0], dtype=np.float32); y = y[np.isfinite(y)]
            y = (y if len(y) else np.zeros(1, np.float32))[-max_ctx:]
            units.append((gi, [0], [(y, np.ones(len(y), bool), C.ROLE_TARGET, None, None)], len(y)))
            continue
        Lc = int(min(max_ctx, max(len(np.asarray(t)) for t in tg)))
        cov_lanes = []
        for cx, fut in kn:
            v, o = C._lane_array(cx, Lc)
            f = np.asarray(fut, dtype=np.float32)[:horizon]
            fo = np.isfinite(f)                                          # [change] no forward fill of the known future:
            f = np.where(fo, f, 0.0).astype(np.float32)                  # missing future values are unobserved (0, False)
            cov_lanes.append((v, o, C.ROLE_KNOWN, f, fo))
        for cx in ps:
            v, o = C._lane_array(cx, Lc)
            cov_lanes.append((v, o, C.ROLE_PAST, None, None))
        per = max(1, cap - len(cov_lanes))
        for c0 in range(0, nt, per):
            ids = list(range(c0, min(nt, c0 + per)))
            lanes = [(*C._lane_array(tg[i], Lc), C.ROLE_TARGET, None, None) for i in ids] + cov_lanes
            units.append((gi, ids, lanes, Lc))

    amp_on = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (amp_on and torch.cuda.is_bf16_supported()) else torch.float16
    by_v = {}
    for u in units:
        by_v.setdefault(len(u[2]), []).append(u)
    for V, us in by_v.items():
        nc = np.array([max(1, -(-u[3] // pl)) for u in us])
        need = nc + hp
        P = int(need.max())
        order = np.argsort(-need, kind="stable")
        brow, bstart, fill = np.zeros(len(us), int), np.zeros(len(us), int), []
        min_need, open_rows = int(need.min()), []
        for i in order:
            for r in open_rows:
                if fill[r] + need[i] <= P:
                    brow[i], bstart[i] = r, fill[r]; fill[r] += need[i]
                    if P - fill[r] < min_need:
                        open_rows.remove(r)
                    break
            else:
                brow[i], bstart[i] = len(fill), 0; fill.append(int(need[i]))
                if P - need[i] >= min_need:
                    open_rows.append(len(fill) - 1)
        nb_all, nb = len(fill), max(1, rows // V)
        for b0 in range(0, nb_all, nb):
            sel = np.nonzero((brow >= b0) & (brow < b0 + nb))[0]
            Bk = int(min(nb, nb_all - b0)); R = Bk * V; W = P * pl
            x = np.zeros((R, W), np.float32); obs = np.zeros((R, W), bool)
            seg = np.full((R, P), -1, np.int64); gid = np.full((R, P), -1, np.int64)
            role = np.zeros((R, P), np.int64)
            for k, i in enumerate(sel):
                _, _, lanes, Lc = us[i]
                st, b = bstart[i], brow[i] - b0
                e_ctx = (st + nc[i]) * pl
                for v, (val, o, rl, fut, fobs) in enumerate(lanes):
                    r = b * V + v
                    x[r, e_ctx - Lc:e_ctx] = val; obs[r, e_ctx - Lc:e_ctx] = o
                    if rl == C.ROLE_KNOWN:
                        x[r, e_ctx:e_ctx + len(fut)] = fut; obs[r, e_ctx:e_ctx + len(fut)] = fobs     # [change] was True
                    seg[r, st:st + need[i]] = k * V + v
                    gid[r, st:st + need[i]] = k
                    role[r, st:st + need[i]] = rl
            dummy = len(sel) * V + np.arange(R)[:, None].repeat(P, 1)
            gid = np.where(seg < 0, dummy, gid); seg = np.where(seg < 0, dummy, seg)
            S = len(sel) * V + R
            xt, ot = torch.from_numpy(x).to(device), torch.from_numpy(obs).to(device)
            sg = torch.from_numpy(seg).to(device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_on):
                seg_pt, seg_start_pt, attn = model._packed_geometry(sg, S)
                q, mu_e, sd_e, _ = model._core(xt, ot, seg_pt, seg_start_pt, S, attn_mask=attn, lanes=V,
                                               gid=torch.from_numpy(gid).to(device),
                                               role=torch.from_numpy(role).to(device))
            allq = (model._inv_tf(q.float()) * sd_e[..., None, None] + mu_e[..., None, None]).cpu().numpy()
            for i in sel:
                gi, ids, lanes, _ = us[i]
                p_last = bstart[i] + nc[i] - 1
                for v, ti in enumerate(ids):
                    f = allq[(brow[i] - b0) * V + v, p_last:p_last + hp].reshape(hp * pl, nq)[:horizon]
                    res[gi][1][ti] = f.T; res[gi][0][ti] = f[:, q_mid]
    return res


def _swaps(name: str) -> dict:
    C, S = thirdparty.ft3_core(), thirdparty.score_twins()
    if name == "baseline":
        return {}
    if name in cov_variants.VARIANTS:
        return cov_variants.swaps(name)
    if name == "nofill_delete":
        return {(S, "fill_season"): strip_leading_nan}
    if name == "nofill_mask":
        return {(S, "fill_season"): strip_leading_nan, (C, "forecast_left_padded"): forecast_left_padded_masked,
                (C, "_lane_array"): lane_array_masked}
    if name == "cov_mask":
        # the targets keep baseline's fill_season: after it only all-NaN series hold NaN, and both lane functions turn
        # those into the same all-unobserved lane, so the swap changes covariate lanes only
        return {(C, "_lane_array"): lane_array_masked, (C, "forecast_groups"): forecast_groups_covmask}
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
