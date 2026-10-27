"""Probes printed at every validation, PORTED AS WRITTEN from reference/tide_v5_100m_300b.py
section 5b: repeat, week, year and flag probes (1 = perfect, 0 = flat line). They are
REPORTED ONLY: no decision (rate, rollback, stop) uses them.
"""
import numpy as np
import torch

from tidemark.pretrain.generators import PERIOD_BANK_EXT
from tidemark.pretrain.inference import forecast_groups, forecast_left_padded

# Noise-free exactly repeating series; score = share of the pattern the forecast captures:
# 1 - squared error / squared error of a flat line at the context mean (1 = perfect, 0 = flat
# line, below 0 = worse; clipped at -1). "list" = periods in the generator's period bank.
PROBE_PERIODS = [7, 10, 12, 16, 24, 32, 48, 50, 64, 70, 96, 100, 120, 128, 168, 200]
PROBE_SHAPES = ["sine", "sawtooth", "square", "spike", "template"]
PROBE_CTX, PROBE_H = 2048, 96


def _probe_set(n_per, seed=0):
    rng = np.random.default_rng(seed)
    meta, ctx, fut = [], [], []
    for shape in PROBE_SHAPES:
        for P in PROBE_PERIODS:
            for _ in range(n_per):
                t = np.arange(PROBE_CTX + PROBE_H) + int(rng.integers(0, P))
                u = (t % P) / P
                if shape == "sine":
                    y = np.sin(2 * np.pi * u)
                elif shape == "sawtooth":
                    y = 2 * u - 1
                elif shape == "square":
                    y = np.where(u < rng.uniform(0.2, 0.5), 1.0, -1.0)
                elif shape == "spike":
                    y = np.where(t % P == 0, 1.0, 0.0)
                else:
                    w = rng.standard_normal(P); w = (w - w.mean()) / (np.abs(w).max() + 1e-9)
                    y = w[t % P]
                s = rng.uniform(0.5, 5.0) * y + rng.normal(0.0, 3.0)
                ctx.append(s[:PROBE_CTX].astype(np.float32)); fut.append(s[PROBE_CTX:]); meta.append((shape, P))
    return meta, ctx, np.asarray(fut)


@torch.no_grad()
def repeat_probe(model, device, n_per=4, table=False):
    """-> dict(list=, other=, by_shape={shape: (list, other)}). Leaves the model in eval mode."""
    meta, ctx, fut = _probe_set(n_per)
    med, _ = forecast_left_padded(model, ctx, PROBE_H, device, max_ctx=PROBE_CTX)
    flat = np.array([c.mean() for c in ctx])[:, None]
    r = np.clip(1.0 - ((med - fut) ** 2).sum(1) / np.maximum(((flat - fut) ** 2).sum(1), 1e-12), -1.0, 1.0)
    r[fut.std(1) < 1e-9] = np.nan
    shp = np.array([m[0] for m in meta]); per = np.array([m[1] for m in meta])
    inl = np.isin(per, PERIOD_BANK_EXT)
    out = dict(list=float(np.nanmean(r[inl])), other=float(np.nanmean(r[~inl])),
               by_shape={s: (float(np.nanmean(r[(shp == s) & inl])), float(np.nanmean(r[(shp == s) & ~inl])))
                         for s in PROBE_SHAPES})
    if table:
        cell = lambda m: f"{np.nanmean(r[m]):9.2f}" if np.isfinite(r[m]).any() else f"{'-':>9s}"
        print(f"{'period':>7s} {'in list':>8s} " + " ".join(f"{s:>9s}" for s in PROBE_SHAPES))
        for P in PROBE_PERIODS:
            print(f"{P:7d} {'yes' if P in PERIOD_BANK_EXT else '-':>8s} "
                  + " ".join(cell((shp == s) & (per == P)) for s in PROBE_SHAPES))
    return out


def probe_line(pr):
    return (f"repeat probe (1 = perfect): generator's periods {pr['list']:.2f} | other periods {pr['other']:.2f}   ["
            + "  ".join(f"{s} {a:.2f}/{b:.2f}" for s, (a, b) in pr["by_shape"].items()) + "]")


WEEK_PPD = [1, 12, 24, 48, 96]        # steps per day (1 = daily data with a day-of-week pattern)
WEEK_KINDS = ["peaks", "template"]


def _week_set(n_per, noise=0.0, seed=0):
    """Noise-free weekday / weekend series. The context ends on a Thursday night; the horizon is
    Friday, Saturday, Sunday (a whole week for daily data). -> per steps-per-day: contexts, the
    clean future, a mask of the weekend points, and the kind of each series."""
    rng = np.random.default_rng(seed)
    out = {}
    for ppd in WEEK_PPD:
        weeks = {1: 52, 12: 24, 24: 12, 48: 6, 96: 6}[ppd]
        H = 7 if ppd == 1 else 3 * ppd
        n_ctx = (weeks * 7 + 4) * ppd
        t = np.arange(n_ctx + H)
        day = t // ppd; tod = t % ppd; dow = day % 7
        u = (tod + 0.5) / ppd
        ctx, fut, kinds = [], [], []
        for kind in WEEK_KINDS:
            for _ in range(n_per):
                if ppd == 1:
                    lv = np.array([1, 1, 1, 1, 1, 0.45, 0.3]) if kind == "peaks" else rng.standard_normal(7)
                    y = lv[dow].astype(np.float64)
                elif kind == "peaks":
                    c1, c2, c3 = 0.33 + rng.uniform(-0.05, 0.05), 0.72 + rng.uniform(-0.05, 0.05), 0.55 + rng.uniform(-0.05, 0.05)
                    wd = np.exp(-0.5 * ((u - c1) / 0.06) ** 2) + 0.8 * np.exp(-0.5 * ((u - c2) / 0.08) ** 2)
                    we = 0.6 * np.exp(-0.5 * ((u - c3) / 0.12) ** 2)
                    y = np.where(dow >= 5, we, wd)
                else:
                    a, b = rng.standard_normal(ppd), 0.6 * rng.standard_normal(ppd) - 0.3
                    y = np.where(dow >= 5, b[tod], a[tod])
                s = rng.uniform(0.5, 5.0) * (y - y.mean()) / (y.std() + 1e-9) + rng.normal(0.0, 3.0)
                c = s[:n_ctx] + noise * s.std() * rng.standard_normal(n_ctx)
                ctx.append(c.astype(np.float32)); fut.append(s[n_ctx:]); kinds.append(kind)
        out[ppd] = (ctx, np.asarray(fut), (dow[n_ctx:] >= 5), kinds)
    return out


@torch.no_grad()
def week_probe(model, device, n_per=4, noise=0.0, table=False):
    """Share of a weekday / weekend pattern the model continues (1 = perfect, 0 = flat line).
    -> dict(all=, weekend=, yesterday=): over the whole horizon, over the weekend points only, and
    what repeating the last day scores on the weekend points. Leaves the model in eval mode."""
    def cap(pred, fut, flat, m):
        r = 1.0 - ((pred - fut) ** 2)[:, m].sum(1) / np.maximum(((flat - fut) ** 2)[:, m].sum(1), 1e-12)
        return np.clip(r, -1.0, 1.0)
    rows, A, W, Y = [], [], [], []
    for ppd, (ctx, fut, wk, kinds) in _week_set(n_per, noise).items():
        med, _ = forecast_left_padded(model, ctx, fut.shape[1], device, max_ctx=8192)
        flat = np.array([c.mean() for c in ctx])[:, None]
        yest = np.stack([np.resize(c[-ppd:], fut.shape[1]) for c in ctx])
        a, w, y = cap(med, fut, flat, np.ones_like(wk)), cap(med, fut, flat, wk), cap(yest, fut, flat, wk)
        A.append(a); W.append(w); Y.append(y)
        rows.append((ppd, a.mean(), w.mean(), y.mean()))
    out = dict(all=float(np.concatenate(A).mean()), weekend=float(np.concatenate(W).mean()),
               yesterday=float(np.concatenate(Y).mean()))
    if table:
        print(f"{'steps/day':>9s} {'all':>7s} {'weekend':>8s} {'yesterday':>10s}")
        for ppd, a, w, y in rows:
            print(f"{ppd:9d} {a:7.2f} {w:8.2f} {y:10.2f}")
        print(f"{'mean':>9s} {out['all']:7.2f} {out['weekend']:8.2f} {out['yesterday']:10.2f}")
    return out


def week_line(wk, wkn):
    return (f"week probe (1 = perfect): weekend {wk['weekend']:.2f} | whole horizon {wk['all']:.2f} | "
            f"noisy context: weekend {wkn['weekend']:.2f}   [repeating yesterday scores {wk['yesterday']:.2f}]")


YEAR_SETS = [(52, 0), (120, 7), (365, 7)]      # (long period, short period inside it; 0 = none)
YEAR_CYCLES = [1.4, 2.4]                        # cycles of history the model is shown


def _year_set(n_per, noise=0.0, seed=0):
    """Series with ONE dated event per long cycle whose next occurrence falls inside the horizon (and two
    other dated events that never touch the horizon). With 1.4 cycles of history that event was seen once
    before, with 2.4 twice. -> list of (P, cycles, contexts, the future, the future WITHOUT the event,
    mask of the event's steps)."""
    rng = np.random.default_rng(seed)
    out = []
    for P, ps in YEAR_SETS:
        w = max(1, P // 50)                      # width of the event: 1, 2, 7 steps
        H = max(8, min(64, P // 6))              # horizon: 8, 20, 60 steps
        for cyc in YEAR_CYCLES:
            n_ctx = int(round(cyc * P))
            t = np.arange(n_ctx + H)
            ctx, fut, base, mask = [], [], [], []
            for _ in range(n_per):
                y = 0.6 * np.sin(2 * np.pi * (t + int(rng.integers(0, P))) / P)
                if ps:
                    y = y + 0.7 * rng.standard_normal(ps)[(t + int(rng.integers(0, ps))) % ps]
                te = n_ctx + int(rng.integers(1, H - w))                  # where the event starts in the horizon
                ev, oth = np.zeros(len(t)), np.zeros(len(t))
                m = float(rng.choice([-1.0, 1.0])) * rng.uniform(2.0, 4.0)
                for a in range(te % P - P, len(t), P):
                    ev[max(a, 0):max(a + w, 0)] += m
                for _k in range(2):
                    off = int(rng.integers(H + w, P - H - w))
                    m2 = float(rng.choice([-1.0, 1.0])) * rng.uniform(1.5, 3.0)
                    for a in range((te - off) % P - P, len(t), P):
                        oth[max(a, 0):max(a + w, 0)] += m2
                full, nob = y + oth + ev, y + oth
                mu_, sd_ = full[:n_ctx].mean(), full[:n_ctx].std() + 1e-9
                sc_, sh_ = rng.uniform(0.5, 5.0), rng.normal(0.0, 3.0)
                f2, b2 = sc_ * (full - mu_) / sd_ + sh_, sc_ * (nob - mu_) / sd_ + sh_
                ctx.append((f2[:n_ctx] + noise * sc_ * rng.standard_normal(n_ctx)).astype(np.float32))
                fut.append(f2[n_ctx:]); base.append(b2[n_ctx:])
                mk = np.zeros(H, bool); mk[te - n_ctx:te - n_ctx + w] = True
                mask.append(mk)
            out.append((P, cyc, ctx, np.asarray(fut), np.asarray(base), np.asarray(mask)))
    return out


@torch.no_grad()
def year_probe(model, device, n_per=4, noise=0.0, table=False):
    """Does the model bring back an event it saw one long cycle ago?
    event: on the steps of the event, 1 - squared error / squared error of the same series WITHOUT the
    event (1 = the event is forecast exactly, 0 = as if it were not there, below 0 = worse).
    rest: the other steps of the horizon, against a flat line at the context mean.
    -> dict(event=, one=, two=, rest=): all sets, event seen once before, seen twice. Leaves the model in eval mode."""
    rows, R = [], []
    for P, cyc, ctx, fut, base, mask in _year_set(n_per, noise):
        med, _ = forecast_left_padded(model, ctx, fut.shape[1], device, max_ctx=8192)
        e = np.clip(1.0 - (((med - fut) ** 2) * mask).sum(1) / np.maximum((((base - fut) ** 2) * mask).sum(1), 1e-12), -1.0, 1.0)
        flat = np.array([c.mean() for c in ctx])[:, None]
        r = np.clip(1.0 - (((med - fut) ** 2) * ~mask).sum(1) / np.maximum((((flat - fut) ** 2) * ~mask).sum(1), 1e-12), -1.0, 1.0)
        rows.append((P, cyc, float(e.mean()), float(r.mean()))); R.append(r)
    out = dict(event=float(np.mean([e for _, _, e, _ in rows])), rest=float(np.concatenate(R).mean()),
               one=float(np.mean([e for _, c, e, _ in rows if c < 2])), two=float(np.mean([e for _, c, e, _ in rows if c >= 2])))
    if table:
        print(f"{'period':>7s} {'cycles seen':>12s} {'event':>7s} {'rest':>7s}")
        for P, cyc, e, r in rows:
            print(f"{P:7d} {cyc:12.1f} {e:7.2f} {r:7.2f}")
        print(f"{'mean':>7s} {'':12s} {out['event']:7.2f} {out['rest']:7.2f}")
    return out


def year_line(yr, yrn):
    return (f"year probe (1 = perfect): event seen once before {yr['one']:.2f} | seen twice {yr['two']:.2f} | "
            f"noisy context {yrn['event']:.2f} | rest of the horizon {yr['rest']:.2f}")


def _flag_set(n_per, seed=0):
    """A weekly pattern around 10 that is ZERO on rare closed days (about 2.5% of the history, at least four),
    with a known flag that names them; the 28-step horizon has two closed days. Two versions of the flag:
    'open' (1 = normal, 0 = closed) and 'holiday' (1 = closed)."""
    rng = np.random.default_rng(seed)
    n_ctx, H = 420, 28
    out = {}
    for pol in ("open", "holiday"):
        groups, fut, base, mask = [], [], [], []
        for _ in range(n_per):
            t = np.arange(n_ctx + H)
            lv = 10.0 + rng.uniform(1.0, 3.0) * np.clip(rng.standard_normal(7), -2.0, 2.0)[(t + int(rng.integers(0, 7))) % 7]
            closed = rng.random(n_ctx + H) < 0.025
            closed[rng.choice(n_ctx - 30, 4, replace=False)] = True
            closed[n_ctx - 7:] = False
            closed[n_ctx + rng.choice(np.arange(2, H - 1), 2, replace=False)] = True
            y = np.where(closed, 0.0, lv)
            flag = (~closed if pol == "open" else closed).astype(np.float32)
            groups.append(dict(targets=[y[:n_ctx].astype(np.float32)], past=[], known=[(flag[:n_ctx], flag[n_ctx:])]))
            fut.append(y[n_ctx:]); base.append(lv[n_ctx:]); mask.append(closed[n_ctx:])
        out[pol] = (groups, np.asarray(fut), np.asarray(base), np.asarray(mask))
    return out


@torch.no_grad()
def flag_probe(model, device, n_per=6):
    """Does the model use a RARE shutdown flag? On the closed days of the horizon: 1 - squared error /
    squared error of the open-day pattern (1 = forecasts zero there, 0 = ignores the flag).
    -> dict(open=, holiday=, hidden=); hidden = the same forecast without the flag. Leaves the model in eval mode."""
    def cap(med, fut, base, mask):
        return float(np.clip(1.0 - (((med - fut) ** 2) * mask).sum(1)
                             / np.maximum((((base - fut) ** 2) * mask).sum(1), 1e-12), -1.0, 1.0).mean())
    res = {}
    for pol, (groups, fut, base, mask) in _flag_set(n_per).items():
        r = forecast_groups(model, groups, fut.shape[1], device)
        res[pol] = cap(np.stack([np.asarray(p[0])[0] for p in r]), fut, base, mask)
        if pol == "open":
            med0, _ = forecast_left_padded(model, [g_["targets"][0] for g_ in groups], fut.shape[1], device, max_ctx=8192)
            res["hidden"] = cap(med0, fut, base, mask)
    return res


def flag_line(fl):
    return (f"flag probe (1 = perfect): 'open' flag {fl['open']:.2f} | 'holiday' flag {fl['holiday']:.2f}   "
            f"[without the flag: {fl['hidden']:.2f}]")


def all_probes(model, device) -> dict:
    """Every probe once (rank 0, at a validation), as in the reference's do_val."""
    pr = repeat_probe(model, device)
    wk, wkn = week_probe(model, device), week_probe(model, device, noise=0.3)
    yr, yrn = year_probe(model, device), year_probe(model, device, noise=0.2)
    fl = flag_probe(model, device)
    lines = [probe_line(pr), week_line(wk, wkn), year_line(yr, yrn), flag_line(fl)]
    vals = dict(probe_list=pr["list"], probe_other=pr["other"], week_end=wk["weekend"], week_all=wk["all"],
                week_end_noisy=wkn["weekend"], year_once=yr["one"], year_twice=yr["two"],
                year_noisy=yrn["event"], flag_open=fl["open"], flag_holiday=fl["holiday"])
    return {"lines": lines, "values": vals}
