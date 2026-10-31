"""Frequency conventions of the dev set: canonical frequency keys, seasonality, and the short / medium / long horizons copied from
TIME (owner decision 2026-10-27: a convention, not selection on TIME results).

TIME sets the prediction length per dataset and term. tools/time_horizons.py (run where the TIME repo is, on Perlmutter) reads
those settings and writes catalog/time_horizons.csv: per canonical frequency and term, the most common prediction length over
TIME's datasets (ties: the smaller), with the source noted. catalog/horizons_extra.csv (freq, term, prediction_length,
source) holds tideforge's own convention for the frequencies where TIME's rows are not used (D, from one dataset only) or
missing (W, M, Q, B, Y); its rows replace TIME's for those frequencies. A term is used for a series only if the series has
>= 3 x that horizon of history.

Index time (owner 2026-10-31): series without calendar time (D279 Santa Fe, D202 C-MAPSS engine cycles, D110 growth curves)
use the frequency key "index": season 1, horizons 10 / 25 / 50 steps (horizons_extra.csv), the same >= 3 x rule. They are
stored on a synthetic clock, step i at 1970-01-01 + i seconds (INDEX_ORIGIN, INDEX_STEP); ingest resamples them by that
step, and nothing reads calendar features from them.
"""
from __future__ import annotations

import warnings
from functools import lru_cache

import pandas as pd

from tideforge.devdata.catalog import CAT

TIME_HORIZONS = CAT / "time_horizons.csv"
HORIZONS_EXTRA = CAT / "horizons_extra.csv"
TERMS = ("short", "medium", "long")
INDEX = "index"
INDEX_ORIGIN = pd.Timestamp("1970-01-01")
INDEX_STEP = "s"


def index_timestamps(n: int) -> pd.DatetimeIndex:
    """The synthetic clock of an index-time series of n steps."""
    return pd.date_range(INDEX_ORIGIN, periods=n, freq=INDEX_STEP)


def resample_rule(freq: str) -> str:
    """The pandas rule ingest resamples with: the frequency itself, or the synthetic step for index time."""
    return INDEX_STEP if canonical(freq) == INDEX else freq


def canonical(freq: str) -> str:
    """'H' / 'h' / '60min' -> 'h'; '15T' -> '15min'; 'W-SUN' -> 'W'; 'MS' / 'ME' / 'M' -> 'M'; 'QS-DEC' -> 'Q'; 'A' -> 'Y';
    'index' -> 'index' (no calendar time)."""
    if str(freq).lower() == INDEX:
        return INDEX
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        off = pd.tseries.frequencies.to_offset(freq)
    name = type(off).__name__
    n = int(getattr(off, "n", 1))
    if name in ("Week",):
        base = "W"
    elif name in ("MonthEnd", "MonthBegin", "BusinessMonthEnd", "BusinessMonthBegin"):
        base = "M"
    elif name.startswith("Quarter"):
        base = "Q"
    elif name.startswith("Year"):
        base = "Y"
    elif name in ("Day",):
        base = "D"
    elif name in ("BusinessDay",):
        base = "B"
    else:
        nanos = int(pd.Timedelta(off).value)
        for unit, ns in (("h", 3600 * 10 ** 9), ("min", 60 * 10 ** 9), ("s", 10 ** 9), ("ms", 10 ** 6)):
            if nanos % ns == 0:
                k = nanos // ns
                if unit == "min" and k % 60 == 0:
                    continue
                return f"{k}{unit}" if not (unit == "h" and k == 1) else "h"
        raise ValueError(f"unsupported frequency {freq!r}")
    return base if n == 1 else f"{n}{base}"


def seasonality(freq: str) -> int:
    if canonical(freq) == INDEX:
        return 1
    from gluonts.time_feature import get_seasonality
    return int(get_seasonality(freq))


@lru_cache(maxsize=None)
def table() -> pd.DataFrame:
    """TIME's rows, except for the frequencies in horizons_extra.csv, whose rows replace TIME's entirely (owner 2026-10-27:
    TIME's sub-daily rows as written; D, W, M, Q, B, Y from tideforge's own convention)."""
    if not TIME_HORIZONS.exists():
        raise FileNotFoundError(f"{TIME_HORIZONS} is missing: run tools/time_horizons.py on Perlmutter (see its docstring)")
    t = pd.read_csv(TIME_HORIZONS, dtype={"freq": str, "term": str})
    if HORIZONS_EXTRA.exists():
        x = pd.read_csv(HORIZONS_EXTRA, dtype={"freq": str, "term": str})
        t = pd.concat([t[~t["freq"].isin(set(x["freq"]))], x], ignore_index=True)
    dup = t[t.duplicated(["freq", "term"], keep=False)]
    if not dup.empty:
        raise ValueError(f"horizon defined twice: {dup[['freq', 'term']].values.tolist()}")
    return t


def horizons(freq: str) -> dict[str, int]:
    """{term: H} for a frequency (the terms it has, in short / medium / long order); raises if it has none."""
    k = canonical(freq)
    t = table()
    rows = t[t["freq"] == k]
    if rows.empty:
        raise KeyError(f"no horizon for frequency {k!r} (have {sorted(set(t['freq']))}); add it to {HORIZONS_EXTRA.name} "
                       f"with the owner's OK")
    d = {r["term"]: int(r["prediction_length"]) for _, r in rows.iterrows()}
    return {term: d[term] for term in TERMS if term in d}


def eligible(n_obs: int, H: int) -> bool:
    """A term is used for a series only if its history is >= 3 x the horizon (owner 2026-10-27)."""
    return n_obs >= 3 * H


def min_length(freq: str) -> int:
    """Ingest keeps a series if it can serve at least the shortest term: 3 x the shortest horizon + 64 points. (Was 3 x the
    longest + 64 before terms became per-series, 2026-10-27; longer terms are skipped per series at build time.)"""
    return 3 * min(horizons(freq).values()) + 64
