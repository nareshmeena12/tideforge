"""Dev series per country, from the country tag (tideforge/devdata/country.py), not from name matching.

    python tools/dev_countries.py [--data DIR]        # DIR = the ingest output root (default data/)

Per source: its tag from sources.yaml; where meta.json exists (after ingest), the series per country it actually holds.
Prints the totals for G2.4's held-out countries (GB, IN, KR, AU, BR). Sources with a per-series tag ({prefix, codes}) and no
meta.json yet are listed as "per series, not ingested".
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO)]

from tideforge.devdata.schema import load_sources  # noqa: E402

HELD_OUT = ("GB", "IN", "KR", "AU", "BR")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data"))
    a = ap.parse_args(argv)
    srcs, _ = load_sources()
    ds, series, pending = Counter(), Counter(), []
    for i, s in sorted(srcs.items()):
        if s.covariate_only:
            continue
        meta = Path(a.data) / s.slug / "meta.json"
        if meta.exists():
            m = json.loads(meta.read_text(encoding="utf-8"))
            cc = Counter(m.get("series_country", {}).values())
            for c in cc:
                ds[c] += 1
            series.update(cc)
            print(f"{i} {s.slug:24s} {s.country!s:28s} ingested: {dict(cc)}")
        elif isinstance(s.country, dict):
            pending.append(i)
            print(f"{i} {s.slug:24s} {s.country!s:28s} per series, not ingested")
        else:
            ds[s.country] += 1
            print(f"{i} {s.slug:24s} {s.country!s:28s} not ingested")
    print("\nheld-out countries (datasets tagged / series counted from meta.json):")
    for c in HELD_OUT:
        print(f"   {c}: {ds[c]} datasets, {series[c]} series")
    print(f"   + per-series sources not yet ingested (may hold any of them): {pending}")
    print(f"multi: {ds['multi']} datasets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
