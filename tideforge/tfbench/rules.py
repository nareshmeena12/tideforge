"""Shared rules: dataset id slug, frequency band and short horizon, minimum series length (used by A3 and B1)."""
from __future__ import annotations

import re

import pandas as pd

from tideforge.devdata.hf_profile import subset_name
from tideforge.tfbench.config import SHORT_H, Settings

DAY = 86400.0


def slug(repo_id: str, config: str = "") -> str:
    s = repo_id + ("__" + subset_name(config) if subset_name(config) else "")
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:120]


def freq_seconds(alias: str) -> float:
    """Nominal step in seconds of a pandas alias (months 30.4 d, quarters 91.3 d, years 365.25 d)."""
    a = alias.upper()
    m = re.fullmatch(r"(\d*)(MS|ME|M|QS|QE|Q|YS|YE|Y|AS|A)(-\w+)?", a)
    if m:
        n = int(m.group(1) or 1)
        base = m.group(2)[0]
        return n * {"M": 30.44, "Q": 91.31, "Y": 365.25, "A": 365.25}[base] * DAY
    if a.startswith("W") or re.fullmatch(r"\d*W(-\w+)?", a):
        n = re.match(r"(\d*)", a).group(1)
        return int(n or 1) * 7 * DAY
    return pd.Timedelta(pd.tseries.frequencies.to_offset(alias)).total_seconds()


def band(alias: str) -> str:
    s = freq_seconds(alias)
    if s < 3600:
        return "subhourly"
    if s == 3600:
        return "hourly"
    if s < DAY:
        return "subdaily"
    if s < 7 * DAY:
        return "daily"
    if s < 28 * DAY:
        return "weekly"
    if s < 89 * DAY:
        return "monthly"
    if s < 360 * DAY:
        return "quarterly"
    return "yearly"


def short_horizon(alias: str) -> int:
    return SHORT_H[band(alias)]


def min_length(alias: str, s: Settings | None = None) -> int:
    """Shortest series kept: minimum context for the short horizon plus one short horizon."""
    s = s or Settings()
    h = short_horizon(alias)
    return max(2 * h, s.min_context) + h
