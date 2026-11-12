"""B1 horizon, window and seasonality rules (devdata/tfbench/horizons.yaml + config.MIN_CONTEXT).

horizons(alias)                     {"short": H, "medium": 10H, "long": 15H} (medium / long only for sub-daily bands)
context_needed(alias, H)            max(2 H, min_context[band])
n_windows(lengths, H, s)            1..max_windows; windows cover <= window_share of the p25 series length, but a series
                                    that has context_needed + H points always gets >= 1 window (fev cuts windows from the
                                    end of each series, so the last window always fits such a series)
seasonality(alias)                  gluonts' default seasonalities (as fev-bench uses), without importing gluonts
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import yaml

from tideforge.tfbench import rules
from tideforge.tfbench.config import TFB, Settings


@lru_cache(maxsize=None)
def table() -> dict:
    return yaml.safe_load((TFB / "horizons.yaml").read_text(encoding="utf-8"))


def horizons(alias: str) -> dict[str, int]:
    t = table()
    b = rules.band(alias)
    h = int(t["short"][b])
    out = {"short": h}
    if b in t["multi_horizon_bands"]:
        out["medium"] = int(t["medium_x"]) * h
        out["long"] = int(t["long_x"]) * h
    return out


def context_needed(alias: str, H: int, s: Settings | None = None) -> int:
    return max(2 * H, rules.min_context(alias, s))


def n_windows(lengths, H: int, s: Settings | None = None) -> int:
    s = s or Settings()
    L = float(np.quantile(np.asarray(lengths, dtype=float), s.window_len_quantile)) if len(lengths) else 0.0
    return int(min(s.max_windows, max(1, np.floor(s.window_share * L / H))))


_SEASON = {"S": 3600, "min": 1440, "h": 24, "D": 1, "B": 5, "W": 1, "M": 12, "Q": 4, "Y": 1}


def seasonality(alias: str) -> int:
    off = pd.tseries.frequencies.to_offset(alias)
    name = type(off).__name__
    n = int(getattr(off, "n", 1))
    if name.startswith("Week"):
        base = "W"
    elif "Month" in name:
        base = "M"
    elif name.startswith("Quarter"):
        base = "Q"
    elif name.startswith("Year"):
        base = "Y"
    elif name == "BusinessDay":
        base = "B"
    else:
        sec = pd.Timedelta(off).total_seconds()
        for base, unit in (("D", 86400), ("h", 3600), ("min", 60), ("S", 1)):
            if sec % unit == 0:
                n = int(sec // unit)
                break
        else:
            return 1
    m = _SEASON[base]
    return max(1, m // n) if m % n == 0 else 1
