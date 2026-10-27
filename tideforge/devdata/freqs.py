"""Frequency conventions of the dev set: canonical frequency keys, seasonality, and the short / medium / long horizons copied from
TIME (owner decision 2026-10-27: a convention, not selection on TIME results).

TIME sets the prediction length per dataset and term. tools/time_horizons.py (run where the TIME repo is, on Perlmutter) reads
those settings and writes catalog/time_horizons.csv: per canonical frequency and term, the most common prediction length over
TIME's datasets (ties: the smaller), with the source noted. A frequency TIME does not have is added by hand to
catalog/horizons_extra.csv (freq, term, prediction_length, source) with the owner's OK; horizons() raises until it is there.
"""
from __future__ import annotations

import warnings
from functools import lru_cache

import pandas as pd

from tideforge.devdata.catalog import CAT

TIME_HORIZONS = CAT / "time_horizons.csv"
HORIZONS_EXTRA = CAT / "horizons_extra.csv"
TERMS = ("short", "medium", "long")


def canonical(freq: str) -> str:
    """'H' / 'h' / '60min' -> 'h'; '15T' -> '15min'; 'W-SUN' -> 'W'; 'MS' / 'ME' / 'M' -> 'M'; 'QS-DEC' -> 'Q'; 'A' -> 'Y'."""
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
    from gluonts.time_feature import get_seasonality
    return int(get_seasonality(freq))


@lru_cache(maxsize=None)
def table() -> pd.DataFrame:
    parts = []
    for p in (TIME_HORIZONS, HORIZONS_EXTRA):
        if p.exists():
            parts.append(pd.read_csv(p, dtype={"freq": str, "term": str}))
    if not parts:
        raise FileNotFoundError(f"{TIME_HORIZONS} is missing: run tools/time_horizons.py on Perlmutter (see its docstring)")
    t = pd.concat(parts, ignore_index=True)
    dup = t[t.duplicated(["freq", "term"], keep=False)]
    if not dup.empty:
        raise ValueError(f"horizon defined twice: {dup[['freq', 'term']].values.tolist()}")
    return t


def horizons(freq: str) -> dict[str, int]:
    """{'short': H, 'medium': H, 'long': H} for a frequency; raises if the table does not have it."""
    k = canonical(freq)
    t = table()
    rows = t[t["freq"] == k]
    if set(rows["term"]) != set(TERMS):
        raise KeyError(f"no complete horizon row for frequency {k!r} (have {sorted(set(t['freq']))}); add it to "
                       f"{HORIZONS_EXTRA.name} with the owner's OK")
    return {r["term"]: int(r["prediction_length"]) for _, r in rows.iterrows()}


def min_length(freq: str) -> int:
    """DATA_COLLECTION.md step 5: a series needs 3 x the longest horizon + 64 points."""
    return 3 * max(horizons(freq).values()) + 64
