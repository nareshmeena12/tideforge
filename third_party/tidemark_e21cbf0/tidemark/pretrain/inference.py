"""Inference for the twins, PORTED AS WRITTEN from reference/tide_v5_100m_300b.py section 7:
forecast_left_padded (univariate, left-padded exactly like training, one pass over blanked
horizon patches) and forecast_groups (groups with covariates, packed like training).
The only change: forecast_groups hands the model a Pack instead of a dense attention mask.
load_twin rebuilds a twin from a checkpoint written by pretrain/train.py.
"""
import numpy as np
import torch

from tidemark.pretrain.generators import ROLE_KNOWN, ROLE_PAST, ROLE_TARGET


@torch.no_grad()
def forecast_left_padded(model, series_list, horizon, device, max_ctx=8192, batch_size=256):
    """Point + quantile forecasts for a list of 1-D series, preprocessed EXACTLY like
    training: keep the last <= max_ctx observed points, LEFT-pad to the next patch boundary,
    mask the padding. Series are bucketed by padded length so no batch mixes lengths.
    horizon > pred_len -> autoregressive rollout on the median.

    Returns point (N, horizon), quant (N, n_q, horizon). Replace forecast_batch() in the
    fev eval with this -- the old flooring throws away data and silently falls back to
    last-value repeat for series shorter than one patch.
    """
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
        Lp = max(pl, ((len(y) + pl - 1) // pl) * pl)
        v = np.zeros(Lp, dtype=np.float32); m = np.zeros(Lp, dtype=bool)
        v[Lp - len(y):] = y; m[Lp - len(y):] = True
        return v, m

    buckets = {}
    for i, y in enumerate(series_list):
        y = np.asarray(y, dtype=np.float32)
        y = y[np.isfinite(y)]
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

def _lane_array(a, n):
    """Last n points of a variate -> (values, observed). Leading NaNs stay unobserved; NaNs
    inside the series are forward-filled (lanes of a group must stay aligned in time, so
    missing points cannot be dropped the way the univariate path drops them)."""
    a = np.asarray(a, dtype=np.float32)[-n:]
    if len(a) < n:
        a = np.concatenate([np.full(n - len(a), np.nan, np.float32), a])
    ok = np.isfinite(a)
    if not ok.any():
        return np.zeros(n, np.float32), np.zeros(n, bool)
    idx = np.where(ok, np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    first = int(np.argmax(ok))
    v = a[idx]
    obs = np.arange(n) >= first
    v[~obs] = 0.0
    return v.astype(np.float32), obs


@torch.no_grad()
def forecast_groups(model, groups, horizon, device, max_ctx=8192, rows=256, max_variates=None):
    """Forecast GROUPS with group attention, fully batched and packed exactly like training.

    groups: list of dicts, one per item:
        targets : list of 1-D arrays (the series to forecast; all get a forecast)
        past    : list of 1-D arrays (covariates observed only up to the forecast start)
        known   : list of (context 1-D array, future 1-D array of length `horizon`)
      All arrays of a group are aligned in time and end at the forecast start.
    Returns a list (one entry per group) of (point (n_targets, horizon),
    quant (n_targets, n_q, horizon)).

    A group never exceeds `max_variates` lanes (default: the largest group seen in training):
    covariates are kept first (known before past, at most max-1 with one target, max//2 with
    several), and the targets are split into chunks that each see those covariates.
    A group with a single variate takes the ordinary univariate path (NaNs dropped)."""
    model.eval()
    pl, nq, q_mid = model.patch_len, model.n_q, model.n_q // 2
    hp = -(-horizon // pl)
    cap = int(max_variates or getattr(model, "group_lanes", 8))
    res = [None] * len(groups)
    if hp > getattr(model, "max_pass_patches", 32) or not getattr(model, "group_attn", False):
        for gi, gr in enumerate(groups):      # rollout / univariate model: targets one by one
            res[gi] = forecast_left_padded(model, gr["targets"], horizon, device, max_ctx=max_ctx)
        return res

    # ---- units: (group, target indices, lanes) with lanes = list of (values, observed, role)
    units = []
    for gi, gr in enumerate(groups):
        tg, past, known = gr["targets"], gr.get("past", []), gr.get("known", [])
        nt = len(tg)
        res[gi] = (np.zeros((nt, horizon), np.float32), np.zeros((nt, nq, horizon), np.float32))
        n_cov = min(len(known) + len(past), cap - 1 if nt == 1 else cap // 2)
        kn, ps = known[:n_cov], past[:max(0, n_cov - len(known))]
        if nt == 1 and n_cov == 0:            # plain univariate: drop NaNs, as in training
            y = np.asarray(tg[0], dtype=np.float32); y = y[np.isfinite(y)]
            y = (y if len(y) else np.zeros(1, np.float32))[-max_ctx:]
            units.append((gi, [0], [(y, np.ones(len(y), bool), ROLE_TARGET, None)], len(y)))
            continue
        Lc = int(min(max_ctx, max(len(np.asarray(t)) for t in tg)))
        cov_lanes = []
        for cx, fut in kn:
            v, o = _lane_array(cx, Lc)
            f = np.asarray(fut, dtype=np.float32)[:horizon]
            fo = np.isfinite(f)
            if (~fo).any():                   # forward-fill the known future from the last value
                last = v[-1] if o[-1] else 0.0
                for i in range(len(f)):
                    if fo[i]:
                        last = f[i]
                    else:
                        f[i] = last
            cov_lanes.append((v, o, ROLE_KNOWN, f))
        for cx in ps:
            v, o = _lane_array(cx, Lc)
            cov_lanes.append((v, o, ROLE_PAST, None))
        per = max(1, cap - len(cov_lanes))
        for c0 in range(0, nt, per):
            ids = list(range(c0, min(nt, c0 + per)))
            lanes = [(*_lane_array(tg[i], Lc), ROLE_TARGET, None) for i in ids] + cov_lanes
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
        order = np.argsort(-need, kind="stable")       # first-fit-decreasing into block rows
        brow, bstart, fill = np.zeros(len(us), int), np.zeros(len(us), int), []
        # only rows that can still take the SMALLEST unit are kept in `open_rows`; the old loop
        # rescanned every row for every series (quadratic in the number of series), which made
        # short-context batches slower than Chronos-2 for no reason
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
                e_ctx = (st + nc[i]) * pl              # end of context / start of horizon
                for v, (val, o, rl, fut) in enumerate(lanes):
                    r = b * V + v
                    x[r, e_ctx - Lc:e_ctx] = val; obs[r, e_ctx - Lc:e_ctx] = o
                    if rl == ROLE_KNOWN:               # known future values fill the horizon patches
                        x[r, e_ctx:e_ctx + len(fut)] = fut; obs[r, e_ctx:e_ctx + len(fut)] = True
                    seg[r, st:st + need[i]] = k * V + v
                    gid[r, st:st + need[i]] = k
                    role[r, st:st + need[i]] = rl
            # unused tails: one dummy segment (and group) per row
            dummy = len(sel) * V + np.arange(R)[:, None].repeat(P, 1)
            gid = np.where(seg < 0, dummy, gid); seg = np.where(seg < 0, dummy, seg)
            S = len(sel) * V + R
            xt, ot = torch.from_numpy(x).to(device), torch.from_numpy(obs).to(device)
            sg = torch.from_numpy(seg).to(device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_on):
                seg_pt, seg_start_pt, pack = model._packed_geometry(sg, S)
                q, mu_e, sd_e, _ = model._core(xt, ot, seg_pt, seg_start_pt, S, pack, lanes=V,
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


def load_twin(path, device, gla_backend: str = "fla"):
    """(model in eval mode, TwinConfig, checkpoint dict) from a twin checkpoint (final.pt,
    best.pt, a milestone or last.pt). Dropout is irrelevant in eval mode."""
    from tidemark.pretrain.config import config_from_json
    from tidemark.pretrain.model import build_twin
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_json(ck["config_json"])
    model = build_twin(cfg, gla_backend=gla_backend).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), cfg, ck
