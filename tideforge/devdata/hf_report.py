"""HF_PROFILE.md: the report of the Hugging Face catalog profile, written from the profile rows (no network).

Totals and tables use the catalog rows only (row_kind = "catalog"); the extra per-config rows of whole-repo catalog rows
would count the same data twice. Sums of series and points are shown apart by confidence (exact / estimated), with the
number of datasets where the value is unknown.
"""
from __future__ import annotations

from pathlib import Path

from . import hf_derive as D
from .hf_derive import fmt_bytes as B
from .hf_derive import fmt_count as N


def _num(v):
    if v in ("", None):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _name(r) -> str:
    sub = r.get("profile_name") or r.get("config_or_subset") or ""
    return f"{r['repo_id']} [{sub}]" if sub else r["repo_id"]


def _split_sum(rows, key, conf_key):
    """-> (exact sum, estimated sum, datasets unknown)."""
    ex = es = 0.0
    unk = 0
    for r in rows:
        v = _num(r.get(key))
        if v is None:
            unk += 1
        elif r.get(conf_key) == D.EXACT:
            ex += v
        else:
            es += v
    return ex, es, unk


def _table(head: list[str], lines: list[list]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c).replace("|", "/").replace("\n", " ") for c in ln) + " |" for ln in lines]
    return out + [""]


def _total_line(label, rows) -> list:
    se, ss, su = _split_sum(rows, "n_series", "n_series_conf")
    pe, ps, pu = _split_sum(rows, "n_points", "n_points_conf")
    by = sum(_num(r.get("bytes_all")) or 0 for r in rows)
    return [label, len(rows), len({r["repo_id"] for r in rows}), B(by), N(se), N(ss), su, N(pe), N(ps), pu]


def write_report(path: Path, all_rows: list[dict], facts: dict):
    rows = [r for r in all_rows if r.get("row_kind") == "catalog"]
    cfg_rows = [r for r in all_rows if r.get("row_kind") != "catalog"]
    f = facts["fetch"]
    L = ["# Hugging Face catalog profile (metadata only)", "",
         "No dataset file was downloaded. Sources: Hub API metadata, the dataset viewer API, parquet footers read by "
         "range request, and small text metadata files. Every number says where it comes from in `hf_profile.csv` "
         "(`*_src`, `*_conf`). \"Exact\" means read from metadata without a guess; \"estimated\" means a rule or a "
         "scaling step was used; the layout itself is always a rule-based reading of the column names and types.", ""]
    # 1 ---------------------------------------------------------------------------------------------------------------
    L += ["## 1. Run facts", ""]
    lv = {}
    for r in rows:
        lv[r.get("profile_level", "")] = lv.get(r.get("profile_level", ""), 0) + 1
    L += [f"- Date: {facts['date']}.", f"- Catalog: {facts['catalog']} ({facts['catalog_rows']} rows, "
          f"{facts['catalog_repos']} repos).",
          f"- Profiled: {facts['repos_profiled']} repos, {len(rows)} catalog rows, plus {len(cfg_rows)} extra per-config "
          f"rows. {facts['n_file_rows']} data files listed in hf_profile_files.csv.",
          f"- Bytes transferred in this run: {f['bytes']} bytes ({B(f['bytes'])}) in {f['requests']} requests; "
          f"{f['cache_hits']} answers came from the cache; {f['http_429']} rate-limit waits."
          + (" This was an offline rebuild from the cache." if facts.get("offline") else ""),
          f"- Token: {'HF_TOKEN was set' if facts.get('token') else 'no token (anonymous calls)'}. "
          f"Footer reads per catalog row: at most {facts['max_footers']} (more files are sampled and scaled by bytes)."]
    for k, v in sorted((f.get("by_kind") or {}).items()):
        L.append(f"- {k}: {v['requests']} requests, {B(v['bytes'])}.")
    L += ["", "Catalog rows at each profile level:", ""]
    L += _table(["profile level", "rows"], [[k or "(none)", v] for k, v in sorted(lv.items(), key=lambda x: -x[1])])
    L += ["Levels: viewer+footers = viewer answers and parquet footers; footers = footers only (viewer not working); "
          "viewer = viewer schema and row counts, no footer; files_only = file names and sizes only; gated = gated, no "
          "access; missing / error = see section 10.", ""]
    # 2 ---------------------------------------------------------------------------------------------------------------
    L += ["## 2. Totals", ""]
    head = ["set", "datasets", "repos", "bytes", "series (exact)", "series (estimated)", "series unknown (datasets)",
            "points (exact)", "points (estimated)", "points unknown (datasets)"]
    lines = [_total_line("everything", rows)]
    for s in sorted({r.get("status", "") for r in rows}):
        lines.append(_total_line(f"status = {s}", [r for r in rows if r.get("status") == s]))
    L += _table(head, lines)
    te, ts, tu = _split_sum(rows, "n_points_target", "n_points_target_conf")
    L += [f"Target points only (all rows): {N(te)} exact + {N(ts)} estimated; unknown for {tu} datasets.",
          "Points = all numeric value columns. Bytes = the data files in the row's scope, from the Hub file listing "
          "(exact). Several rows of one repo with overlapping scopes count their bytes more than once.", ""]
    # 3 ---------------------------------------------------------------------------------------------------------------
    L += ["## 3. By domain", ""]
    lines = []
    for d in sorted({r.get("domain", "") for r in rows}):
        g = [r for r in rows if r.get("domain") == d]
        se, ss, _ = _split_sum(g, "n_series", "n_series_conf")
        pe, ps, _ = _split_sum(g, "n_points", "n_points_conf")
        known = [r for r in g if r.get("has_covariates") in ("yes", "no")]
        cov = sum(1 for r in known if r["has_covariates"] == "yes")
        firsts = sorted(r["date_first"] for r in g if r.get("date_first"))
        lasts = sorted(r["date_last"] for r in g if r.get("date_last"))
        lines.append([d, len({r["repo_id"] for r in g}), len(g), N(se + ss), N(pe + ps),
                      B(sum(_num(r.get("bytes_all")) or 0 for r in g)),
                      f"{100.0 * cov / len(known):.0f}% ({cov} of {len(known)})" if known else "unknown",
                      f"{firsts[0][:10]} to {lasts[-1][:10]}" if firsts and lasts else "unknown"])
    L += _table(["domain", "repos", "datasets", "series", "points", "bytes", "share with covariates", "date range"], lines)
    L += ["Series and points are exact + estimated, over the datasets where they are known. Share with covariates is "
          "over the datasets with a schema.", ""]
    # 4 ---------------------------------------------------------------------------------------------------------------
    L += ["## 4. By frequency band", ""]
    lines = []
    for b in D.BANDS:
        g = [r for r in rows if (r.get("freq_band") or "unknown") == b]
        se, ss, _ = _split_sum(g, "n_series", "n_series_conf")
        pe, ps, _ = _split_sum(g, "n_points", "n_points_conf")
        lines.append([b, len(g), N(se + ss), N(pe + ps)])
    L += _table(["frequency band", "datasets", "series", "points"], lines)
    L += ["A frequency is set only when a column, feature or config / file name states it, or when span and series "
          "length imply it (estimated). It is never taken from the catalog text. \"mixed\" = several stated "
          "frequencies in one row.", ""]
    # 5 ---------------------------------------------------------------------------------------------------------------
    L += ["## 5. By series length band", ""]
    lines = []
    for b in D.LEN_BANDS:
        g = [r for r in rows if (r.get("len_band") or "unknown") == b]
        se, ss, _ = _split_sum(g, "n_series", "n_series_conf")
        lines.append([b, len(g), N(se + ss)])
    L += _table(["series length (median, or mean where only that is known)", "datasets", "series"], lines)
    # 6 ---------------------------------------------------------------------------------------------------------------
    L += ["## 6. Largest datasets", "", "### 40 largest by points", ""]
    top = sorted((r for r in rows if _num(r.get("n_points"))), key=lambda r: -_num(r["n_points"]))[:40]
    L += _table(["#", "dataset", "domain", "points", "confidence", "series", "bytes", "layout"],
                [[i + 1, _name(r), r.get("domain", ""), N(r["n_points"]), r.get("n_points_conf", ""),
                  N(r.get("n_series")), B(r.get("bytes_all")), r.get("layout", "")] for i, r in enumerate(top)])
    L += ["### 40 with the most series", ""]
    top = sorted((r for r in rows if _num(r.get("n_series"))), key=lambda r: -_num(r["n_series"]))[:40]
    L += _table(["#", "dataset", "domain", "series", "confidence", "points", "median length", "layout"],
                [[i + 1, _name(r), r.get("domain", ""), N(r["n_series"]), r.get("n_series_conf", ""),
                  N(r.get("n_points")), N(r.get("len_median")), r.get("layout", "")] for i, r in enumerate(top)])
    # 7 ---------------------------------------------------------------------------------------------------------------
    L += ["## 7. Datasets with covariates", ""]
    cov = [r for r in rows if r.get("has_covariates") == "yes"]
    L += [f"{len(cov)} datasets have candidate covariate columns (numeric or boolean columns next to the target; list "
          f"columns next to the target for nested data). Known-in-future / past-only is filled only where the column "
          f"names make it clear.", ""]
    L += _table(["dataset", "covariate columns", "known in future", "past only"],
                [[_name(r), r.get("covariate_cols", ""), r.get("cov_known_future", "") or "-",
                  r.get("cov_past_only", "") or "-"] for r in cov])
    # 8 ---------------------------------------------------------------------------------------------------------------
    L += ["## 8. The three slice choices", "",
          "(i) 500 series x last 16k points; (ii) 5,000 series x last 64k points; (iii) everything. All estimates. "
          "Points = min(S, series) x min(L per value column, points per series). Bytes are proportional to points for "
          "parquet scopes (the real fetch rounds up to whole row groups or files); for other formats every slice costs "
          "all bytes. Datasets with unknown series or points count only in (iii) bytes.", ""]
    lines = []
    doms = sorted({r.get("domain", "") for r in rows})
    for d in doms + ["ALL"]:
        g = rows if d == "ALL" else [r for r in rows if r.get("domain") == d]
        s = lambda k: sum(_num(r.get(k)) or 0 for r in g)  # noqa: E731
        lines.append([d, N(s("slice500_points")), B(s("slice500_bytes")), N(s("slice5k_points")), B(s("slice5k_bytes")),
                      N(s("slice_all_points")), B(s("slice_all_bytes"))])
    L += _table(["domain", "(i) points", "(i) bytes", "(ii) points", "(ii) bytes", "(iii) points", "(iii) bytes"], lines)
    # 9 ---------------------------------------------------------------------------------------------------------------
    L += ["## 9. Ingest difficulty", ""]
    dc = {}
    for r in rows:
        dc[r.get("ingest_difficulty") or "(not profiled)"] = dc.get(r.get("ingest_difficulty") or "(not profiled)", 0) + 1
    L += _table(["difficulty", "datasets"], [[k, v] for k, v in sorted(dc.items())])
    L += ["easy = parquet with one schema; medium = csv / arrow / several tables / nested arrays; hard = archives, "
          "netcdf, h5, npy, custom formats, event tables.", "", "### Hard datasets", ""]
    L += _table(["dataset", "reason"], [[_name(r), r.get("ingest_reason", "")] for r in rows
                                        if r.get("ingest_difficulty") == "hard"])
    # 10 --------------------------------------------------------------------------------------------------------------
    L += ["## 10. Problems", "", "Every row that is missing, renamed, empty, gated without access, in error, or whose Hub "
          "contents disagree with the catalog row.", ""]
    L += _table(["catalog line", "dataset", "level", "problem"],
                [[r["catalog_row"], _name(r), r.get("profile_level", ""), r.get("problems") or r.get("error", "")]
                 for r in all_rows if r.get("problems") or r.get("error")])
    # 11 --------------------------------------------------------------------------------------------------------------
    L += ["## 11. What could not be determined from metadata", "",
          "Per dataset: the values that are unknown or estimated, and why. A dataset that is not listed has every "
          "value exact.", ""]
    L += _table(["dataset", "not determined / estimated"], [[_name(r), r.get("undetermined", "")] for r in rows
                                                             if r.get("undetermined")])
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
