"""D3 repair attribution: oracle repairs of our forecast on the worst windows, one at a time, each against the original.

    python -m tideforge.diagnose.repair --run runs/dev/dev-v1/tuning_fast/m5 [--worst 0.1] [--out runs/diagnose/dev-v1/d3]
    (the run must be scored with devscore --save-forecasts; tuning role only, never TIME)

q is (9, H) (quantiles 0.1 .. 0.9), m its median (row 4), y the truth (H,). Each repair uses the truth (that is what an oracle
is) and changes one aspect:
  level      q + (mean y - mean m)
  trend      q + (b_y - b_m)(t - mean t), b = least-squares slope over the horizon
  amplitude  q + (a - 1) s_m, a = <s_m, s_y> / <s_m, s_m>; s = the seasonal part of the joint least-squares fit
             x ~ c0 + c1 t + sum_p d_p 1[t mod S = p] (sum_p d_p = 0): a projection, zero on lines, exact on a pure season
  phase      q - s_m + roll(s_m, k), k in [-S/2, S/2] with the lowest error
  spikes     at the truth's spikes (|y - rolling median| > 5 x 1.4826 MAD), every quantile moved by y - m
  width      m + c (q - m), c in 2^(k/4), k = -8..8 (0.25 .. 4) with the lowest error over the horizon: the horizon's truth is
             the sample the spread is calibrated on (with a perfect median the best c is the smallest: width then also
             takes credit for a sharper median, so read it next to level / trend)
(amplitude and phase need S >= 2 and H >= 2 S; otherwise they return q unchanged.) Error = mean pinball loss x 2 over the 9
levels and the horizon (CRPS), divided by the window's naive CRPS from windows.parquet, so windows add like devscore's ratio.
The repairs overlap, so the shares of error removed do not add up to 1.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tideforge.diagnose import _boot

QL = np.linspace(0.1, 0.9, 9)
REPAIRS = ("level", "trend", "amplitude", "phase", "spikes", "width")
WIDTHS = 2.0 ** (np.arange(-8, 9) / 4)


def crps(q, y) -> float:
    q, y = np.asarray(q, np.float64), np.asarray(y, np.float64)
    d = y[None, :] - q
    return float(2 * np.mean(np.maximum(QL[:, None] * d, (QL[:, None] - 1) * d)))


def _slope(x):
    t = np.arange(len(x), dtype=np.float64)
    tc = t - t.mean()
    den = (tc * tc).sum()
    return float((tc * (x - x.mean())).sum() / den) if den > 0 else 0.0


def seasonal(x, S: int) -> np.ndarray:
    """Seasonal part of the joint fit c0 + c1 t + sum_p d_p 1[phase p] with sum d_p = 0 (least squares)."""
    x = np.asarray(x, np.float64)
    H = len(x)
    t = np.arange(H, dtype=np.float64)
    ph = np.arange(H) % S
    D = np.zeros((H, S))
    D[np.arange(H), ph] = 1.0
    Dc = D[:, :-1] - D[:, -1:]                                       # sum-to-zero coding
    X = np.column_stack([np.ones(H), t, Dc])
    coef = np.linalg.lstsq(X, x, rcond=None)[0]
    return Dc @ coef[2:]


def spikes_of(y, w: int = 3) -> np.ndarray:
    y = np.asarray(y, np.float64)
    if len(y) < 2 * w + 1:
        return np.zeros(len(y), bool)
    yp = np.concatenate([np.full(w, y[0]), y, np.full(w, y[-1])])
    med = np.median(np.lib.stride_tricks.sliding_window_view(yp, 2 * w + 1), axis=1)
    r = y - med
    mad = 1.4826 * np.median(np.abs(r - np.median(r)))
    if mad <= 1e-12 * (np.abs(y).mean() + 1e-12):
        return np.abs(r) > 1e-9 * (np.abs(y).max() + 1e-12)
    return np.abs(r) > 5 * mad


def repairs(q, y, S: int) -> dict[str, np.ndarray]:
    q, y = np.asarray(q, np.float64), np.asarray(y, np.float64)
    H = len(y)
    m = q[4]
    t = np.arange(H, dtype=np.float64)
    out = {"level": q + (y.mean() - m.mean()), "trend": q + (_slope(y) - _slope(m)) * (t - t.mean())}
    if S >= 2 and H >= 2 * S:
        sm, sy = seasonal(m, S), seasonal(y, S)
        den = float(sm @ sm)
        out["amplitude"] = q + ((float(sm @ sy) / den - 1) * sm if den > 1e-12 else 0.0)
        best = None
        for k in range(-(S // 2), S // 2 + 1):
            qk = q - sm + np.roll(sm, k)
            e = crps(qk, y)
            if best is None or e < best[0] - 1e-15:
                best = (e, qk)
        out["phase"] = best[1]
    else:
        out["amplitude"] = out["phase"] = q.copy()
    sp = spikes_of(y)
    out["spikes"] = np.where(sp[None, :], q + (y - m)[None, :], q)
    errs = [(crps(m + c * (q - m), y), c) for c in WIDTHS]
    out["width"] = m + min(errs)[1] * (q - m)
    return out


def attribute(fc: dict, win: pd.DataFrame, worst: float = 0.1, mode: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """fc: devscore.load_forecasts(...). Per window of the worst share (by scaled CRPS, within each dataset so that every dataset
    keeps its part): error before and after each repair. Returns (per-window table, shares overall and per domain)."""
    key = win.set_index(["dataset_id", "series_id", "window", "mode"])
    rows = []
    for i in range(len(fc["q"])):
        if mode is not None and fc["mode"][i] != mode:
            continue
        r = key.loc[(fc["dataset_id"][i], fc["series_id"][i], int(fc["window"][i]), fc["mode"][i])]
        naive = float(r["CRPS_naive"]) if float(r["CRPS_naive"]) > 0 else np.nan
        e0 = crps(fc["q"][i], fc["y"][i])
        row = dict(dataset_id=fc["dataset_id"][i], domain=r["domain"], series_id=fc["series_id"][i], window=int(fc["window"][i]),
                   mode=fc["mode"][i], e0=e0 / naive)
        for name, qr in repairs(fc["q"][i], fc["y"][i], int(fc["S"][i])).items():
            row[name] = crps(qr, fc["y"][i]) / naive
        rows.append(row)
    df = pd.DataFrame(rows).dropna(subset=["e0"])
    cut = df.groupby("dataset_id")["e0"].transform(lambda s: s.quantile(1 - worst))
    df = df[df["e0"] >= cut].copy()
    df["w"] = _boot.window_weights(df["dataset_id"])
    out = []
    for scope, g in [("all", df)] + [(f"domain={d}", g) for d, g in df.groupby("domain")]:
        tot = float((g["w"] * g["e0"]).sum())
        for name in REPAIRS:
            removed = (g["e0"] - g[name]).clip(lower=0)
            share = float((g["w"] * removed).sum() / tot) if tot > 0 else 0.0
            per = g.assign(_r=removed / g["e0"].where(g["e0"] > 0))
            m, lo, hi = _boot.mean_ci(per["_r"].fillna(0), per["dataset_id"])
            out.append(dict(scope=scope, repair=name, share_removed=share, window_mean=m, lo=lo, hi=hi,
                            windows=len(g), datasets=g["dataset_id"].nunique()))
    return df, pd.DataFrame(out)


def main(argv=None) -> int:
    from tideforge.diagnose.fixable import load_dev_run
    from tideforge.eval.devscore import load_forecasts
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--worst", type=float, default=0.1)
    ap.add_argument("--mode", default=None, help="main / on / hidden (default: all modes in the run)")
    ap.add_argument("--out", default="runs/diagnose/d3")
    a = ap.parse_args(argv)
    run, win = load_dev_run(a.run)
    f = Path(a.run) / "forecasts.npz"
    if not f.exists():
        raise SystemExit(f"{f} missing: score the run with devscore --save-forecasts")
    df, sh = attribute(load_forecasts(f), win, a.worst, a.mode)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "repair_windows.csv", index=False)
    sh.to_csv(out / "repair_shares.csv", index=False)
    print(sh[sh["scope"] == "all"].to_string(index=False))
    print("(repairs overlap: the shares do not add up to 1)")
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
