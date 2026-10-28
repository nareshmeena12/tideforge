"""Country tag per dev series (owner 2026-10-28, for G2.4's held-out calendars).

A source's `country` (sources.yaml) is an ISO 3166-1 alpha-2 code, "multi" (unknown or many countries, e.g. Europe-wide sets),
or {prefix: N, codes: fips | iso2}: the country is read from the first N characters of the series id (GHCN station ids start
with a FIPS 10-4 code). Codes that cannot be mapped become "multi". Ingest writes the tag per series into meta.json.
"""
from __future__ import annotations

import re

MULTI = "multi"
# FIPS 10-4 codes that differ from ISO alpha-2 (the ones GHCN uses for countries in the dev catalog and the G2 pool);
# a FIPS code missing here is mapped to "multi", never guessed
FIPS_TO_ISO = {
    "US": "US", "UK": "GB", "IN": "IN", "KS": "KR", "KN": "KP", "AS": "AU", "BR": "BR", "CA": "CA", "MX": "MX", "FR": "FR",
    "GM": "DE", "SP": "ES", "IT": "IT", "NL": "NL", "BE": "BE", "SZ": "CH", "AU": "AT", "PL": "PL", "EZ": "CZ", "SW": "SE",
    "NO": "NO", "DA": "DK", "FI": "FI", "PO": "PT", "GR": "GR", "RS": "RU", "UP": "UA", "TU": "TR", "SF": "ZA", "NI": "NG",
    "KE": "KE", "EG": "EG", "MO": "MA", "SA": "SA", "AE": "AE", "QA": "QA", "IS": "IL", "PK": "PK", "BG": "BD", "CE": "LK",
    "NP": "NP", "CH": "CN", "HK": "HK", "TW": "TW", "JA": "JP", "VM": "VN", "TH": "TH", "MY": "MY", "ID": "ID", "RP": "PH",
    "NZ": "NZ", "AR": "AR", "CI": "CL", "CO": "CO", "EI": "IE", "LU": "LU", "SN": "SG",
}
EU_TO_ISO = {"EL": "GR", "UK": "GB"}                  # Eurostat's two exceptions to ISO alpha-2
ISO = re.compile(r"^[A-Z]{2}$")


def check_tag(tag) -> list[str]:
    if tag in ("", None):
        return ["country is required (ISO alpha-2, 'multi', or {prefix: N, codes: fips|iso2})"]
    if isinstance(tag, dict):
        if set(tag) != {"prefix", "codes"} or tag["codes"] not in ("fips", "iso2") or not int(tag["prefix"]) > 0:
            return [f"country {tag!r}: must be {{prefix: N, codes: fips|iso2}}"]
        return []
    return [] if tag == MULTI or ISO.match(str(tag)) else [f"country {tag!r}: not an ISO alpha-2 code or 'multi'"]


def series_country(tag, series_id: str) -> str:
    if not isinstance(tag, dict):
        return str(tag)
    code = str(series_id)[: int(tag["prefix"])].upper()
    if tag["codes"] == "fips":
        return FIPS_TO_ISO.get(code, MULTI)
    code = EU_TO_ISO.get(code, code)
    return code if ISO.match(code) else MULTI
