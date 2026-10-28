"""G2 calendar engine: a clock per series (calendar, start date, steps per day) and the calendar effects, all tensor ops on the
batch's device (gathers from g2_table, shifts, cumulative max / min); the only Python loops are over the five step sizes.

Clock. Steps per day from PPD = (1/7, 1, 24, 48, 96) (1/7 = weekly rows: built daily, mean of 7 days by reshape). A row takes
a real calendar from the pool (uniform over calendars whose valid years hold the whole window; G2.4 excludes the held-out
ones) or, with probability `invented` (and always when no real calendar is long enough), an invented calendar. The start
day is uniform inside the calendar's valid days. Dates follow from day indices by arithmetic (civil_from_days), so
invented calendars work for any span.

Invented calendar (per row): 4-12 fixed dates (month, day), 1-4 "nth weekday of a month" rules (n = 1..4 or last), 1-3
drifting holidays (period 354.37 days: -11 days a year, 1-3 days long), each with a random type 1-9; weekend rule drawn from
sat_sun / fri_sat / thu_fri / sun / fri / sat (0.5 / 0.15 / 0.05 / 0.15 / 0.1 / 0.05), 5% switching to another rule on a
random day.

Effects (sd units of the row, drawn per row; never per holiday):
  holiday   per type t: s_t = sign x U(0.1, 1.5), sign negative with probability p_neg ~ U(0, 1) (per row)
  run-up    over the 0-7 days before: a exp(-(k - 1) / tau) s_{t(d + k)}, a ~ U(0, 0.5), tau ~ U(0.5, 3)
  rebound   over the 0-3 days after: b exp(-(k - 1) / tau2) s_{t(d - k)}, b ~ U(-0.5, 0.5), tau2 ~ U(0.5, 2)
  weekend   U(-1.2, 0.3) on the calendar's weekend days (v0's range)
  bridge    (rows with probability bridge_p) a business day between a holiday and a non-business day: U(0, 1) x the mean
            holiday effect
  school    (school_p) synthetic breaks by hemisphere (summer 6-10 weeks, 2-3 short breaks of 1-2 weeks): U(-0.5, 0.5)
  fiscal    (fiscal_p) last business day of each fiscal quarter U(0, 1), of the fiscal year x U(2, 4); year-end month from
            {1, 3, 6, 9, 12}
  G2.5      payday (payday_p): first business day on or after the 1st and the 15th, and the last business day: U(-0.3, 1);
            month / quarter end (month_end_p): last day of the month U(-0.5, 0.5), of a quarter x U(1, 3);
            DST (dst_p, hourly rows of calendars with a DST rule): local-time rows, one 23-hour and one 25-hour day a year
  hourly    (shape_p, steps per day >= 24) weekend and holiday days take a weekend-like daily shape: deviation from the
            day's mean x U(0.3, 1), peak moved later by 0-3 hours
Applied additively (x + E sd) or, for positive rows with probability mult_p, multiplicatively (x (1 + E r), r ~ U(0.05, 0.3),
floored at 0.05 x).
"""
from __future__ import annotations

import math

import torch

from tideforge.gen.ext import g2_table
from tideforge.gen.ext._rand import bern, randint, u

PPD = (1.0 / 7.0, 1.0, 24.0, 48.0, 96.0)
EPOCH = 7305                                     # days from 1970-01-01 to 1990-01-01 (day index 0 of the table)
NT = 13                                          # holiday types (g2_table / build_calendars TYPES)
RULES = ((5, 6), (4, 5), (3, 4), (6,), (4,), (5,))
RULE_P = (0.5, 0.15, 0.05, 0.15, 0.1, 0.05)
DRIFT = 354.367
OPTS = dict(invented=0.3, exclude_held_out=False, ppd_probs=(0.2, 0.2, 0.2, 0.2, 0.2), bridge_p=0.5, school_p=0.3,
            fiscal_p=0.2, payday_p=0.0, month_end_p=0.0, dst_p=0.0, shape_p=0.5, mult_p=0.5)
PAD_L, PAD_R = 3, 10


def choice(g, n, probs, dev):
    c = torch.cumsum(torch.tensor(probs, dtype=torch.float32, device=dev), 0)
    return torch.searchsorted(c / c[-1], torch.rand(n, generator=g, device=dev), right=True).clamp(max=len(probs) - 1)


def dates(day: torch.Tensor) -> dict:
    """Civil date of day indices (0 = 1990-01-01; any integer): weekday (Mon 0), dom, month, year, doy (0-based), ylen, dim,
    month id (months since 1990-01). Howard Hinnant's days_from_civil inverse, in integer tensor ops."""
    z = day + EPOCH + 719468
    era = torch.div(z, 146097, rounding_mode="floor")
    doe = z - era * 146097
    yoe = torch.div(doe - torch.div(doe, 1460, rounding_mode="floor") + torch.div(doe, 36524, rounding_mode="floor")
                    - torch.div(doe, 146096, rounding_mode="floor"), 365, rounding_mode="floor")
    y = yoe + era * 400
    dy = doe - (365 * yoe + torch.div(yoe, 4, rounding_mode="floor") - torch.div(yoe, 100, rounding_mode="floor"))
    mp = torch.div(5 * dy + 2, 153, rounding_mode="floor")
    d = dy - torch.div(153 * mp + 2, 5, rounding_mode="floor") + 1
    m = torch.where(mp < 10, mp + 3, mp - 9)
    y = y + (m <= 2).long()
    leap = ((y % 4 == 0) & (y % 100 != 0)) | (y % 400 == 0)
    cum = torch.tensor([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], device=day.device)
    doy = cum[m - 1] + d - 1 + (leap & (m > 2)).long()
    mdays = torch.tensor([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], device=day.device)
    dim = mdays[m - 1] + (leap & (m == 2)).long()
    return dict(wd=torch.remainder(day + EPOCH + 3, 7), dom=d, month=m, year=y, doy=doy, ylen=365 + leap.long(), dim=dim,
                mid=(y - 1990) * 12 + m - 1)


def _rule_mask(wd, rule):
    out = torch.zeros_like(wd, dtype=torch.bool)
    for k, days in enumerate(RULES):
        hit = torch.zeros_like(out)
        for dd in days:
            hit = hit | (wd == dd)
        out = torch.where(rule == k, hit, out)
    return out


def invented(g, days: torch.Tensor, dev) -> tuple[torch.Tensor, torch.Tensor]:
    """(n, m) absolute day indices -> holiday codes (n, m) and weekend mask (n, m) of one invented calendar per row."""
    n = days.shape[0]
    D = dates(days)
    code = torch.zeros_like(days)
    K = randint(g, 4, 12, (n, 1), dev)
    pm, pd = randint(g, 1, 12, (n, 12), dev), randint(g, 1, 28, (n, 12), dev)
    pt = randint(g, 1, 9, (n, 12), dev)
    for j in range(12):
        hit = (D["month"] == pm[:, j:j + 1]) & (D["dom"] == pd[:, j:j + 1]) & (j < K)
        code = torch.where(hit & (code == 0), pt[:, j:j + 1], code)
    R = randint(g, 1, 4, (n, 1), dev)
    rm, rw = randint(g, 1, 12, (n, 4), dev), randint(g, 0, 6, (n, 4), dev)
    rn, rt = randint(g, 0, 4, (n, 4), dev), randint(g, 1, 9, (n, 4), dev)
    wk_of = torch.div(D["dom"] - 1, 7, rounding_mode="floor")
    last = D["dom"] + 7 > D["dim"]
    for j in range(4):
        nth = torch.where(rn[:, j:j + 1] == 4, last, wk_of == rn[:, j:j + 1])
        hit = (D["month"] == rm[:, j:j + 1]) & (D["wd"] == rw[:, j:j + 1]) & nth & (j < R)
        code = torch.where(hit & (code == 0), rt[:, j:j + 1], code)
    Dn = randint(g, 1, 3, (n, 1), dev)
    a, dur = u(g, (n, 3), 0.0, DRIFT, dev), randint(g, 1, 3, (n, 3), dev)
    dtp = randint(g, 1, 9, (n, 3), dev)
    for j in range(3):
        hit = (torch.remainder(days.float() - a[:, j:j + 1], DRIFT) < dur[:, j:j + 1].float()) & (j < Dn)
        code = torch.where(hit & (code == 0), dtp[:, j:j + 1], code)
    r1, r2 = choice(g, n, RULE_P, dev).view(n, 1), choice(g, n, RULE_P, dev).view(n, 1)
    sw = bern(g, (n, 1), 0.05, dev)
    at = days[:, :1] + torch.floor(torch.rand((n, 1), generator=g, device=dev) * days.shape[1]).long()
    rule = torch.where(sw & (days >= at), r2, r1)
    return code, _rule_mask(D["wd"], rule)


def clock(g, n, span: int, tab, opts, dev):
    """Calendar per row: cal (n,) (-1 = invented), first day d0 (n,)."""
    ok = (tab.hi - tab.lo + 1) >= span
    if opts["exclude_held_out"]:
        ok = ok & ~tab.held_out
    idx = ok.nonzero().squeeze(1)
    inv = bern(g, (n,), opts["invented"], dev) | (idx.numel() == 0)
    if idx.numel():
        cal = idx[torch.floor(torch.rand(n, generator=g, device=dev) * idx.numel()).long().clamp(max=idx.numel() - 1)]
        room = (tab.hi[cal] - tab.lo[cal] + 1 - span + 1).clamp(min=1)
        d0 = tab.lo[cal] + torch.floor(torch.rand(n, generator=g, device=dev) * room).long()
    else:
        cal = torch.zeros(n, dtype=torch.long, device=dev)
        d0 = torch.zeros(n, dtype=torch.long, device=dev)
    d0i = torch.floor(torch.rand(n, generator=g, device=dev) * tab.D).long() - span // 2
    return torch.where(inv, -1, cal), torch.where(inv, d0i, d0)


def _shift(x, k, fill):
    """x (n, m) moved k steps later (k > 0) or earlier (k < 0), filled."""
    if k == 0:
        return x
    f = torch.full_like(x[:, :abs(k)], fill)
    return torch.cat([f, x[:, :-k]], 1) if k > 0 else torch.cat([x[:, -k:], f], 1)


def _next_pos(mask):
    """Index of the next True strictly after each position (large if none)."""
    n, m = mask.shape
    t = torch.arange(m, device=mask.device).view(1, m).expand(n, m)
    v = torch.where(mask, t, torch.full_like(t, 10 ** 9))
    nxt = torch.flip(torch.cummin(torch.flip(v, [1]), 1).values, [1])
    return torch.cat([nxt[:, 1:], torch.full_like(nxt[:, :1], 10 ** 9)], 1)


def _prev_pos(mask):
    n, m = mask.shape
    t = torch.arange(m, device=mask.device).view(1, m).expand(n, m)
    prv = torch.cummax(torch.where(mask, t, torch.full_like(t, -1)), 1).values
    return torch.cat([torch.full_like(prv[:, :1], -1), prv[:, :-1]], 1)


def school_mask(g, n, D, south, dev):
    """Synthetic school breaks by hemisphere (per row, repeating every year)."""
    doy = D["doy"]
    def brk(start, length):
        return torch.remainder(doy - start, 366) < length
    sn = south.view(n, 1)
    summer = torch.where(sn, brk(u(g, (n, 1), 345, 360, dev).long(), randint(g, 35, 56, (n, 1), dev)),
                         brk(u(g, (n, 1), 170, 200, dev).long(), randint(g, 42, 70, (n, 1), dev)))
    s1 = brk(torch.where(sn, u(g, (n, 1), 95, 110, dev), u(g, (n, 1), 80, 110, dev)).long(), randint(g, 7, 14, (n, 1), dev))
    s2 = brk(torch.where(sn, u(g, (n, 1), 180, 195, dev), u(g, (n, 1), 290, 310, dev)).long(), randint(g, 7, 14, (n, 1), dev))
    s3 = brk(torch.where(sn, u(g, (n, 1), 265, 280, dev), torch.full((n, 1), 356.0, device=dev)).long(),
             randint(g, 7, 14, (n, 1), dev)) & bern(g, (n, 1), 0.7, dev)
    return summer | s1 | s2 | s3


def daily(g, n, nd: int, tab, opts, dev) -> dict:
    """Calendar and daily effect (sd units) for n rows of nd days."""
    cal, d0 = clock(g, n, nd, tab, opts, dev)
    m = nd + PAD_L + PAD_R
    days = d0.view(n, 1) - PAD_L + torch.arange(m, device=dev).view(1, m)
    real = cal >= 0
    cc = cal.clamp(min=0).view(n, 1)
    di = days.clamp(0, tab.D - 1)
    code_r, wk_r = tab.code[cc, di], tab.weekend[cc, di]
    code_i, wk_i = invented(g, days, dev)
    rv = real.view(n, 1)
    code = torch.where(rv, code_r, code_i)
    wk = torch.where(rv, wk_r, wk_i)
    D = dates(days)
    s = torch.where(torch.rand((n, NT), generator=g, device=dev) < u(g, (n, 1), 0, 1, dev), -1.0, 1.0) \
        * u(g, (n, NT), 0.1, 1.5, dev)
    s[:, 0] = 0.0
    H = torch.gather(s, 1, code)
    E = H.clone()
    a, tau, kpre = u(g, (n, 1), 0, 0.5, dev), u(g, (n, 1), 0.5, 3, dev), randint(g, 0, 7, (n, 1), dev)
    b, tau2, kpost = u(g, (n, 1), -0.5, 0.5, dev), u(g, (n, 1), 0.5, 2, dev), randint(g, 0, 3, (n, 1), dev)
    for k in range(1, 8):
        E = E + (k <= kpre) * a * torch.exp(-(k - 1) / tau) * _shift(H, -k, 0.0)
    for k in range(1, 4):
        E = E + (k <= kpost) * b * torch.exp(-(k - 1) / tau2) * _shift(H, k, 0.0)
    E = E + u(g, (n, 1), -1.2, 0.3, dev) * wk
    hol = code > 0
    bus = ~wk & ~hol
    nonbus = ~bus
    bridge = bus & ((_shift(nonbus, 1, False) & _shift(hol, -1, False)) | (_shift(hol, 1, False) & _shift(nonbus, -1, False)))
    bridge = bridge & bern(g, (n, 1), opts["bridge_p"], dev)
    E = E + bridge * u(g, (n, 1), 0, 1, dev) * s[:, 1:].mean(1, keepdim=True)
    south = torch.where(real, tab.south[cal.clamp(min=0)], bern(g, (n,), 0.2, dev))
    school = school_mask(g, n, D, south, dev) & bern(g, (n, 1), opts["school_p"], dev)
    E = E + school * u(g, (n, 1), -0.5, 0.5, dev)
    nb = _next_pos(bus)
    mid = D["mid"]
    nb_mid = torch.gather(mid, 1, nb.clamp(max=m - 1))
    lastbd = bus & ((nb >= m) | (nb_mid != mid))
    fe = torch.tensor([1, 3, 6, 9, 12], device=dev)[choice(g, n, (0.2,) * 5, dev)].view(n, 1)
    fq = lastbd & (torch.remainder(D["month"] - fe, 3) == 0)
    fy = lastbd & (D["month"] == fe)
    fis = bern(g, (n, 1), opts["fiscal_p"], dev)
    sq = u(g, (n, 1), 0, 1, dev)
    E = E + fis * (fq * sq + fy * sq * u(g, (n, 1), 1, 3, dev))
    pb = _prev_pos(bus)
    pb_dom = torch.gather(D["dom"], 1, pb.clamp(min=0))
    pb_mid = torch.gather(mid, 1, pb.clamp(min=0))
    first = bus & ((pb < 0) | (pb_mid != mid))
    fifteenth = bus & (D["dom"] >= 15) & ((pb < 0) | (pb_mid != mid) | (pb_dom < 15))
    payday = (first | fifteenth | lastbd) & bern(g, (n, 1), opts["payday_p"], dev)
    E = E + payday * u(g, (n, 1), -0.3, 1, dev)
    me = (D["dom"] == D["dim"]) & bern(g, (n, 1), opts["month_end_p"], dev)
    E = E + me * u(g, (n, 1), -0.5, 0.5, dev) * torch.where(torch.remainder(D["month"], 3) == 0, u(g, (n, 1), 1, 3, dev), 1.0)
    dst = torch.where(rv, tab.dst[cc, di], torch.zeros_like(wk))
    has_dst = torch.where(real, tab.has_dst[cal.clamp(min=0)], torch.zeros_like(real))
    crop = slice(PAD_L, PAD_L + nd)
    return dict(E=E[:, crop], code=code[:, crop], weekend=wk[:, crop], bridge=bridge[:, crop], payday=payday[:, crop],
                school=school[:, crop], dst=dst[:, PAD_L - 1:PAD_L + nd], has_dst=has_dst, cal=cal, d0=d0, real=real,
                s=s, hol=hol[:, crop])


def _to_steps(v, ppd, L, mode="gather"):
    """Daily (n, nd) -> steps (n, L): weekly rows average (or max) 7 days, sub-daily rows repeat the day."""
    if ppd < 1:
        w = v[:, :7 * L].float().view(v.shape[0], L, 7)
        return w.amax(-1) if mode == "max" else w.mean(-1)
    p = int(ppd)
    idx = torch.div(torch.arange(L, device=v.device), p, rounding_mode="floor").view(1, L).expand(v.shape[0], L)
    return torch.gather(v, 1, idx)


def _days_needed(ppd, L):
    return 7 * L if ppd < 1 else math.ceil(L / ppd)


def _dst_index(dst_days, p, L, dev):
    """Local-time rows: physical step of each local step. dst_days (n, nd + 1) on / off at the end of each day (first column
    = the day before). Spring (off -> on): one hour skipped; autumn: one hour repeated; the change at 02:00. Steps past the
    end of the window are clamped to the last one (at most one hour repeated at the end of a row)."""
    n = dst_days.shape[0]
    h = p // 24
    on = dst_days.long()
    ch = on[:, 1:] - on[:, :-1]                              # +1 spring, -1 autumn, on each day
    t = torch.arange(L, device=dev)
    day = torch.div(t, p, rounding_mode="floor")
    after = (t - day * p) >= 2 * h                          # past 02:00 local
    cum = torch.cumsum(ch, 1)
    prev = torch.cat([torch.zeros_like(cum[:, :1]), cum[:, :-1]], 1)
    di = day.view(1, L).expand(n, L)
    c = torch.where(after.view(1, L), torch.gather(cum, 1, di), torch.gather(prev, 1, di))
    return (t.view(1, L) + h * c).clamp(0, L - 1)


def apply(g, x: torch.Tensor, opts: dict, dev, ppd_idx: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """x (n, L) -> (x with calendar effects, info with per-step weekend / code / bridge / payday flags, ppd per row)."""
    n, L = x.shape
    tab = g2_table.table(dev)
    if ppd_idx is None:
        ppd_idx = choice(g, n, opts["ppd_probs"], dev)
    out = x.clone()
    info = {k: torch.zeros(n, L, device=dev) for k in ("effect", "weekend", "code", "bridge", "payday")}
    info.update(ppd=torch.tensor(PPD, device=dev)[ppd_idx], cal=torch.zeros(n, dtype=torch.long, device=dev),
                dst_rows=torch.zeros(n, dtype=torch.bool, device=dev), d0=torch.zeros(n, dtype=torch.long, device=dev),
                phys=torch.arange(L, device=dev).view(1, L).repeat(n, 1))
    for k, p in enumerate(PPD):
        idx = (ppd_idx == k).nonzero().squeeze(1)
        if idx.numel() == 0:
            continue
        m = idx.numel()
        xs = x[idx]
        nd = _days_needed(p, L)
        dd = daily(g, m, nd, tab, opts, dev)
        sd = xs.std(1, keepdim=True) + 1e-6
        if p >= 24 and opts["dst_p"] > 0:
            use = dd["has_dst"] & bern(g, (m,), opts["dst_p"], dev)
            j = _dst_index(dd["dst"], int(p), L, dev)
            j = torch.where(use.view(m, 1), j, torch.arange(L, device=dev).view(1, L))
            xs = torch.gather(xs, 1, j)
            info["dst_rows"][idx] = use
            info["phys"][idx] = j
        if p >= 24:
            off = dd["weekend"] | dd["hol"]
            shp = bern(g, (m, 1), opts["shape_p"], dev)
            dayi = torch.div(torch.arange(L, device=dev), int(p), rounding_mode="floor").view(1, L).expand(m, L)
            cnt = torch.zeros(m, nd, device=dev).scatter_add_(1, dayi, torch.ones_like(xs))
            mday = torch.zeros(m, nd, device=dev).scatter_add_(1, dayi, xs) / cnt.clamp(min=1)
            md = torch.gather(mday, 1, dayi)
            sh = torch.floor(u(g, (m, 1), 0, 3, dev) * p / 24).long()
            src = (torch.arange(L, device=dev).view(1, L) - sh).clamp(min=0)
            dev_s = torch.gather(xs - md, 1, src)
            f = u(g, (m, 1), 0.3, 1.0, dev)
            offs = torch.gather(off, 1, dayi) & shp
            xs = torch.where(offs, md + f * dev_s, xs)
        e = _to_steps(dd["E"], p, L)
        pos = (xs.min(1, keepdim=True).values > 0) & bern(g, (m, 1), opts["mult_p"], dev)
        r = u(g, (m, 1), 0.05, 0.3, dev)
        y = torch.where(pos, torch.maximum(xs * (1 + e * r), 0.05 * xs), xs + e * sd)
        out[idx] = y
        info["effect"][idx] = y - x[idx]
        info["weekend"][idx] = _to_steps(dd["weekend"].float(), p, L)
        info["code"][idx] = _to_steps(dd["code"].float(), p, L, "max")
        info["bridge"][idx] = _to_steps(dd["bridge"].float(), p, L, "max")
        info["payday"][idx] = _to_steps(dd["payday"].float(), p, L, "max")
        info["cal"][idx] = dd["cal"]
        info["d0"][idx] = dd["d0"]
    return out, info
