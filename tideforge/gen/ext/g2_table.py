"""G2 calendar tables on the batch's device (built offline by tools/build_calendars.py; committed with their sha256).

  code (C, D) int8 holiday type per calendar and day (TYPES; 0 none), weekend (C, D) bool, dst (C, D) bool
  per calendar: valid day range [lo, hi] (whole years), south (school terms), held_out (G2.4), id
  per day (D,): weekday (Mon = 0), dom, month (1-12), doy (0-based), year, days in month, month id (months since 1990-01),
  year length (365 / 366)
Loaded once per device; the npz bytes are checked against the manifest's sha256 (a mismatch is an error, never ignored).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

DIR = Path(__file__).resolve().parent / "calendars"
VERSION = "v1"
_CACHE: dict = {}


class Table:
    def __init__(self, device):
        raw = (DIR / f"calendars_{VERSION}.npz").read_bytes()
        man = json.loads((DIR / f"calendars_{VERSION}.json").read_text(encoding="utf-8"))
        sha = hashlib.sha256(raw).hexdigest()
        if sha != man["sha256"]:
            raise RuntimeError(f"calendars_{VERSION}.npz sha256 {sha} != manifest {man['sha256']}")
        z = np.load(io.BytesIO(raw))
        dev = torch.device(device)
        self.types = man["types"]
        self.cals = man["calendars"]
        self.ids = [c["id"] for c in self.cals]
        self.code = torch.as_tensor(z["code"].astype(np.int64), device=dev)
        self.weekend = torch.as_tensor(z["weekend"].astype(bool), device=dev)
        self.dst = torch.as_tensor(z["dst"].astype(bool), device=dev)
        self.C, self.D = self.code.shape
        start = dt.date.fromisoformat(man["start"])
        days = np.arange(self.D)
        d64 = np.datetime64(man["start"]) + days
        y = d64.astype("datetime64[Y]").astype(int) + 1970
        m = d64.astype("datetime64[M]").astype(int) % 12 + 1
        mstart = d64.astype("datetime64[M]")
        dom = (d64 - mstart.astype("datetime64[D]")).astype(int) + 1
        dim = ((mstart + 1).astype("datetime64[D]") - mstart.astype("datetime64[D]")).astype(int)
        ystart = d64.astype("datetime64[Y]").astype("datetime64[D]")
        doy = (d64 - ystart).astype(int)
        ylen = ((d64.astype("datetime64[Y]") + 1).astype("datetime64[D]") - ystart).astype(int)
        wd = (days + start.weekday()) % 7
        mid = (y - start.year) * 12 + m - 1
        t = lambda a: torch.as_tensor(a.astype(np.int64), device=dev)
        self.wd, self.dom, self.month, self.doy, self.year, self.dim, self.mid, self.ylen = map(
            t, (wd, dom, m, doy, y, dim, mid, ylen))
        lo = [(dt.date(c["valid"][0], 1, 1) - start).days for c in self.cals]
        hi = [(dt.date(c["valid"][1], 12, 31) - start).days for c in self.cals]
        self.lo, self.hi = torch.tensor(lo, device=dev), torch.tensor(hi, device=dev)
        self.south = torch.tensor([c["south"] for c in self.cals], device=dev)
        self.held_out = torch.tensor([c["held_out"] for c in self.cals], device=dev)
        self.has_dst = torch.tensor([c["dst"] is not None for c in self.cals], device=dev)


def table(device) -> Table:
    key = str(torch.device(device))
    if key not in _CACHE:
        _CACHE[key] = Table(device)
    return _CACHE[key]
