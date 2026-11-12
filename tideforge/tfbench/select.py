"""A1 selection: which catalog rows of hf_profile.csv go into tfbench v1, with the reason for every row.

Rules (in order, the first that fails is the reason): catalog row; status in Settings.statuses; repo reachable
(not gated without access, not missing, no profile error); scope_bytes <= max_scope_gb unless on the allow list.
Any ingest difficulty is accepted.
"""
from __future__ import annotations

import pandas as pd

from tideforge.tfbench.config import Settings
from tideforge.tfbench.rules import slug

COLS = ["catalog_row", "dataset_id", "repo_id", "config_or_subset", "domain", "status", "scope_bytes", "layout",
        "frequency", "ingest_difficulty", "keep", "reason"]


def allowed_large(repo: str, config: str, s: Settings) -> bool:
    return repo in s.allow_large or f"{repo}|{config}" in s.allow_large


def decide(r: dict, s: Settings) -> tuple[bool, str]:
    if r.get("status") not in s.statuses:
        return False, f"status {r.get('status')} (v1 uses {', '.join(s.statuses)})"
    lvl = str(r.get("profile_level") or "")
    if lvl == "gated":
        return False, "gated, no access"
    if lvl in ("missing", "error"):
        return False, f"profile {lvl}: {str(r.get('error') or '')[:120]}"
    b = r.get("scope_bytes")
    b = float(b) if b == b and b not in (None, "") else 0.0
    if b <= 0:
        return False, "no data files in scope"
    lim = s.max_scope_gb * 1e9
    if b > lim:
        if allowed_large(r["repo_id"], str(r.get("config_or_subset") or ""), s):
            return True, f"over {s.max_scope_gb:g} GB ({b / 1e9:.1f} GB) but on the allow list"
        return False, f"scope {b / 1e9:.1f} GB > {s.max_scope_gb:g} GB"
    return True, "ok"


def select(profile: pd.DataFrame, s: Settings | None = None) -> pd.DataFrame:
    s = s or Settings()
    p = profile[profile["row_kind"] == "catalog"]
    rows, seen = [], {}
    for r in p.to_dict("records"):
        cfg = "" if r.get("config_or_subset") != r.get("config_or_subset") else str(r.get("config_or_subset") or "")
        did = slug(r["repo_id"], cfg)
        if did in seen:                       # two catalog rows with the same slug: keep them apart
            did = f"{did}_r{int(r['catalog_row'])}"
        seen[did] = 1
        keep, why = decide(r, s)
        rows.append({**{k: r.get(k) for k in COLS if k in r}, "config_or_subset": cfg, "dataset_id": did,
                     "keep": keep, "reason": why})
    return pd.DataFrame(rows, columns=COLS)
