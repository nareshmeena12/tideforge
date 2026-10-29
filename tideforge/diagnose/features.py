"""Window features shared by D4 (rules), D5 (coverage), D6 (realism) and tools/gen_audit.py (owner 2026-10-29: one module).

shape_features(x)       gen_audit's shape features, unchanged: flat, zero_frac, integer, trend_r2, acf_peak, period, seasonal,
                        cycles, spec_slope, rough, n_jumps, kurt_diff, var_ratio, short_season, exact_repeat
window_features(ctx, S, freq, has_cov)   the D-block features of one context (computed on the context only, NaN dropped):
  period_ratio    S / context length (S = the frequency's season, 1 = none)
  seas_strength   max(0, 1 - var(R) / var(S + R)), trend_strength max(0, 1 - var(R) / var(T + R)) (Wang-Hyndman style): T a
                  centred moving average over S steps (over max(3, L / 10) when S < 2 or L < 2 S), S the mean by phase of the
                  detrended series (0 without a season), R the rest
  zero_share, adi share of exact zeros; mean interval between nonzero values (L when none)
  late_level      (mean of the last 20% - mean of the rest) / sd; late_scale log(sd of the last 20% / sd of the rest)
  spikiness       share of points more than 5 robust sd (1.4826 MAD) from a rolling median (window 2 w + 1, w = 2..12)
  noise           1.4826 MAD of the first differences / sd
  log_length, freq, has_cov
plus every shape_features key. FEATURES: the numeric ones D4 uses; SYNTH_COMPARABLE: those defined for synthetic windows too
(no freq, no has_cov: owner 2026-10-29).
"""
from __future__ import annotations

import math

import numpy as np

def shape_features(x) -> dict:
    """gen_audit's per-series shape features (tools/gen_audit.py, moved here unchanged 2026-10-29). Only "flat" for a flat series."""
    x = x.astype(np.float64); L = len(x); t = np.arange(L)
    f = {}
    sd = x.std()
    f["flat"] = sd < 1e-9 * (abs(x).mean() + 1)
    if f["flat"]:
        return f
    d = np.diff(x)
    f["zero_frac"] = float(np.mean(x == 0))
    f["integer"] = bool(np.all(np.abs(x - np.round(x)) < 1e-6))
    # trend: R^2 of linear fit
    A = np.vstack([t, np.ones(L)]).T
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    r = x - A @ coef
    f["trend_r2"] = 1 - r.var() / x.var()
    # seasonality: ACF of linearly detrended series, best lag in [2, L/3]
    rr = r - r.mean(); den = (rr * rr).sum() + 1e-12
    F = np.fft.rfft(rr, n=2 * L); ac = np.fft.irfft(F * np.conj(F))[:L] / den
    hi = max(3, L // 3)
    lag = int(np.argmax(ac[2:hi])) + 2
    f["acf_peak"] = float(ac[lag]); f["period"] = lag
    # local maximum check (a real cycle, not a slow decay)
    f["seasonal"] = bool(ac[lag] > 0.5 and ac[lag] > ac[max(lag - 1, 1)] and ac[lag] >= ac[min(lag + 1, L - 1)])
    f["cycles"] = L / lag
    # spectral slope of the periodogram (log-log), persistence
    P = np.abs(np.fft.rfft(rr)) ** 2
    fr = np.fft.rfftfreq(L)[1:]; P = P[1:]
    ok = P > 0
    f["spec_slope"] = float(np.polyfit(np.log(fr[ok]), np.log(P[ok]), 1)[0]) if ok.sum() > 5 else np.nan
    # roughness, jumps, tails, heteroskedasticity
    f["rough"] = float(d.std() / sd)
    mad = np.median(np.abs(d - np.median(d))) + 1e-12
    f["n_jumps"] = int(np.sum(np.abs(d) > 8 * 1.4826 * mad)) if mad > 1e-9 else 0
    k = ((d - d.mean()) ** 4).mean() / (d.var() ** 2 + 1e-30)
    f["kurt_diff"] = float(k)
    h = L // 2
    f["var_ratio"] = float(np.log((x[h:].std() + 1e-12) / (x[:h].std() + 1e-12)))
    # SHORT seasons (G3.4): the dominant frequency of the detrended series (periodogram zero-padded 4x, steps of 0.25
    # cycles per window) lies in [1.5, 3) cycles and holds >= 25% of the power; the ACF detector above needs >= 3 cycles
    Pz = np.abs(np.fft.rfft(rr, n=4 * L)) ** 2
    cyc = np.arange(len(Pz)) / 4.0
    band = cyc >= 1.0
    kpk = int(np.argmax(np.where(band, Pz, 0)))
    f["short_season"] = bool(1.5 <= cyc[kpk] < 3.0 and Pz[kpk] >= 0.25 * Pz[band].sum() / 4.0)
    # exact repetition at the detected lag
    f["exact_repeat"] = bool(np.mean(np.abs(x[lag:] - x[:-lag])) < 1e-3 * sd)
    return f


SHAPE = ["trend_r2", "acf_peak", "cycles", "spec_slope", "rough", "n_jumps", "kurt_diff", "var_ratio", "integer", "short_season",
         "exact_repeat", "seasonal"]
OWN = ["period_ratio", "seas_strength", "trend_strength", "zero_share", "adi", "late_level", "late_scale", "spikiness", "noise",
       "log_length"]
SYNTH_COMPARABLE = OWN + SHAPE
FEATURES = SYNTH_COMPARABLE + ["has_cov"]          # freq is categorical: D4 bins it separately
EPS = 1e-9


def _ma(y, w):
    w = max(1, int(w))
    if w == 1:
        return y.copy()
    left, right = (w - 1) // 2, w // 2
    yp = np.concatenate([np.full(left, y[0]), y, np.full(right, y[-1])])
    c = np.cumsum(np.concatenate([[0.0], yp]))
    return (c[w:] - c[:-w]) / w


def decompose(y, S):
    """(trend, seasonal, remainder) of y (float64) for season S (see the module docstring)."""
    L = len(y)
    seasonal_ok = S >= 2 and L >= 2 * S
    T = _ma(y, S if seasonal_ok else max(3, L // 10))
    d = y - T
    if seasonal_ok:
        ph = np.arange(L) % S
        means = np.bincount(ph, weights=d, minlength=S) / np.maximum(np.bincount(ph, minlength=S), 1)
        Sx = means[ph] - means.mean()
    else:
        Sx = np.zeros(L)
    return T, Sx, d - Sx


def _strength(R, X):
    v = np.var(X + R)
    return float(max(0.0, 1.0 - np.var(R) / v)) if v > EPS else 0.0


def window_features(ctx, S: int = 1, freq: str | None = None, has_cov: bool | None = None) -> dict:
    y = np.asarray(ctx, np.float64)
    y = y[np.isfinite(y)]
    L = len(y)
    f = dict(freq=freq, has_cov=float(bool(has_cov)) if has_cov is not None else np.nan, log_length=math.log(max(L, 1)),
             period_ratio=float(S) / max(L, 1))
    if L < 4:
        return f
    T, Sx, R = decompose(y, int(S))
    sd = y.std() + EPS
    f.update(seas_strength=_strength(R, Sx), trend_strength=_strength(R, T), zero_share=float(np.mean(y == 0)))
    nz = np.flatnonzero(y != 0)
    f["adi"] = float(np.diff(nz).mean()) if len(nz) > 1 else float(L)
    k = max(1, int(round(0.2 * L)))
    a, b = y[:-k], y[-k:]
    f["late_level"] = float((b.mean() - a.mean()) / sd)
    f["late_scale"] = float(math.log((b.std() + EPS * sd) / (a.std() + EPS * sd)))
    w = int(min(12, max(2, (S if S >= 2 else 5) // 2)))
    yp = np.concatenate([np.full(w, y[0]), y, np.full(w, y[-1])])
    med = np.median(np.lib.stride_tricks.sliding_window_view(yp, 2 * w + 1), axis=1)
    r = y - med
    mad = 1.4826 * np.median(np.abs(r - np.median(r)))
    f["spikiness"] = float(np.mean(np.abs(r) > 5 * mad)) if mad > EPS * sd else 0.0
    dd = np.diff(y)
    f["noise"] = float(1.4826 * np.median(np.abs(dd - np.median(dd))) / sd)
    sh = shape_features(y)
    for key in SHAPE:
        if key in sh:
            f[key] = float(sh[key])
    return f


def frame(items) -> "pd.DataFrame":
    """items: iterable of dicts with ctx, S and optionally freq, cov (bool) and any key columns -> one row per window."""
    import pandas as pd
    rows = []
    for it in items:
        f = window_features(it["ctx"], int(it.get("S", 1)), it.get("freq"), it.get("cov"))
        rows.append({**{k: v for k, v in it.items() if k not in ("ctx", "y", "past", "known", "q")}, **f})
    return pd.DataFrame(rows)


def detected_season(x) -> int:
    """The season seen in the series itself: shape_features' ACF period when it marks the series seasonal, else 1. D5 / D6 use
    it for real and synthetic windows alike (synthetic rows have no frequency)."""
    sh = shape_features(np.asarray(x, np.float64)[np.isfinite(np.asarray(x, np.float64))])
    return int(sh["period"]) if sh.get("seasonal") else 1


def comparable_frame(xs, keys: list[dict] | None = None) -> "pd.DataFrame":
    """Features of raw windows with the detected season (D5 / D6), optional key columns per window."""
    keys = keys or [{} for _ in xs]
    return frame([dict(k, ctx=x, S=detected_season(x)) for x, k in zip(xs, keys)])
