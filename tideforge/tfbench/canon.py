"""A3 canonical form: one dataset (a selected catalog row) -> data.parquet + meta.json under $TFB_DATA/canon/<dataset_id>/.

Generic adapter driven by the schema of the downloaded files (the same role rules as the profiler, hf_derive), with
devdata/tfbench/overrides.yaml winning wherever it says something. data.parquet columns: item_id, timestamp, target
column(s), covariates (past / known), static columns (repeated per row). meta.json: source, licence, status, domain,
frequency, column roles, and every transformation with its count.

Steps: open tables -> pick the main table (the schema group with most bytes) -> roles and layout -> enumerate items
-> keep_filtered drop rule -> seeded stratified order -> load items in chunks of max_series, regularise, filter
(too short, > max_nan_share NaN, constant, exact duplicate) -> accept in order until max_series -> last max_points.

Override keys (per dataset_id): skip, files (regex on repo file paths: only these are opened, so big archives that are
not needed are never extracted), layout, table (regex on part names), table_cols (parts must have all these columns),
drop_parts, drop_items (regex on item_id), time_col, calendar ({year, month | week | quarter, day, hour} / {yyyymm} /
{yearweek} / {fiscal_year} column names, week_rule mmwr | iso; see layouts.py), id_cols, targets, covariates, known,
static, strata, freq, start_col, event_step, event_sum, agg {col: sum|mean}, fill_zero (missing grid steps = 0),
synthetic_time, note. Tables without a time column: see layouts.py.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tideforge.devdata import hf_derive as D
from tideforge.tfbench import rules
from tideforge.tfbench.config import Settings
from tideforge.tfbench.readers import Part, Unreadable, open_parts
from tideforge.tfbench.timefreq import infer_freq, parse_time, regularise

log = logging.getLogger("tfbench.canon")

CALENDAR = re.compile(r"(^|_)(hour|hour_of_day|minute|day|dayofweek|day_of_week|weekday|dow|dayofyear|day_of_year|doy|"
                      r"week|weekofyear|week_of_year|month|quarter|year|is_weekend|weekend|holiday|holidays|"
                      r"is_holiday|public_holiday|school_holiday|season|time_idx)($|_)", re.I)
KNOWN_CANDIDATE = re.compile(r"forecast|fcst|predict|pred_|_pred|nwp|schedul|planned|plan_|tariff|price|promo|"
                             r"advert|target_temp|setpoint|day_ahead|dayahead", re.I)


class Fail(Exception):
    """The generic adapter cannot read this dataset; the reason goes into the report."""


class NeedSyntheticTime(Exception):
    """A series' real timestamps do not fit pandas' datetime range (e.g. LOTSA starts far in the past)."""


SYNTHETIC_START = pd.Timestamp("2000-01-01")


def synthetic_capacity(alias: str) -> int:
    """Steps of `alias` that fit from SYNTHETIC_START to pandas' last timestamp (minus a year of margin)."""
    span = (pd.Timestamp("2261-01-01") - SYNTHETIC_START).total_seconds()
    return int(span // rules.freq_seconds(alias))


# ---- schema -> roles ----------------------------------------------------------------------------------------------------
def _kind(t: pa.DataType) -> tuple[str, int]:
    depth = 0
    while pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
        t, depth = t.value_type, depth + 1
    if pa.types.is_dictionary(t):
        t = t.value_type
    if pa.types.is_timestamp(t) or pa.types.is_date(t):
        k = "time"
    elif pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_decimal(t):
        k = "num"
    elif pa.types.is_boolean(t):
        k = "bool"
    elif pa.types.is_string(t) or pa.types.is_large_string(t):
        k = "str"
    else:
        k = "other"
    if depth:
        k = {"num": "list_num", "time": "list_time"}.get(k, "list_other")
    return k, depth


def cols_of(schema: pa.Schema) -> list[D.Col]:
    out = []
    for f in schema:
        k, depth = _kind(f.type)
        out.append(D.Col(f.name, k, str(f.type), depth))
    return out


def roles_for(schema: pa.Schema, ov: dict) -> tuple[dict, str, str]:
    cols = cols_of(schema)
    r = D.assign_roles(cols)
    layout, rule, _ = D.decide_layout(cols, r)
    names = set(schema.names)
    for k_ov, k in (("time_col", "time_col"), ("id_cols", "id_cols"), ("targets", "target_cols"),
                    ("covariates", "covariate_cols"), ("static", "static_cols")):
        if k_ov in ov:
            v = ov[k_ov]
            r[k] = v if k == "time_col" else list(v or [])
            if k == "target_cols":
                r["target_rule"] = "overrides.yaml"
    if "targets" in ov and "covariates" not in ov:   # value columns that are not targets stay covariates
        r["covariate_cols"] = [c for c in r["covariate_cols"] + r["value_cols"] if c not in r["target_cols"]]
    from tideforge.tfbench.layouts import UNNAMED
    for k in ("target_cols", "covariate_cols", "value_cols", "static_cols"):   # pandas index columns are never data
        r[k] = [c for c in r.get(k, []) if not UNNAMED.match(c)] if k != "target_cols" or "targets" not in ov else r[k]
    if "covariates" not in ov:   # numeric calendar columns (hour, month, ...) are known covariates, not time columns
        cal = [c.name for c in cols if c.kind in ("num", "bool") and CALENDAR.search(c.name) and c.name != r["time_col"]
               and c.name not in r["id_cols"] and c.name not in r["target_cols"]]
        r["covariate_cols"] = list(dict.fromkeys(r["covariate_cols"] + cal))
    if "time_col" in ov and not ov.get("layout"):   # a time column set from outside: decide the layout again
        r["value_cols"] = [c for c in r["value_cols"] if c != r["time_col"]]
        if "targets" not in ov:
            r["target_cols"] = [c for c in r["target_cols"] if c != r["time_col"]]
        layout, rule, _ = D.decide_layout(cols, r)
    if ov.get("layout"):
        layout, rule = ov["layout"], "overrides.yaml"
    missing = [c for c in ([r["time_col"]] if r["time_col"] else []) + r["id_cols"] + r["target_cols"] if c not in names]
    if missing:
        raise Fail(f"columns not in the table: {missing}")
    r["covariate_cols"] = list(dict.fromkeys(c for c in r["covariate_cols"] if c in names and c not in r["target_cols"]))
    return r, layout, rule


def cov_roles(names: list[str], known_ov: list[str]) -> dict:
    """Calendar / holiday columns and overrides -> known; name rules only propose (known_candidate, used as past in v1)."""
    out = {}
    for n in names:
        if n in known_ov or CALENDAR.search(n):
            out[n] = "known"
        elif KNOWN_CANDIDATE.search(n) or n in D.covariate_timing([n])[0]:
            out[n] = "known_candidate"
        else:
            out[n] = "past"
    return out


# ---- table choice -------------------------------------------------------------------------------------------------------
def open_all(files: list[tuple[Path, str]], workdir: Path, did: str = "") -> tuple[list[Part], list[str]]:
    parts, bad = [], []
    for local, rel in files:
        try:
            parts += open_parts(local, workdir, rel)
        except Unreadable as e:
            bad.append(f"{rel}: {e}")
            log.warning("%s: %s unreadable: %s", did, rel, str(e)[:200])
        except Exception as e:  # a reader bug on one file never stops the dataset
            bad.append(f"{rel}: {type(e).__name__}: {str(e)[:200]}")
            log.warning("%s: %s read error %s: %s", did, rel, type(e).__name__, str(e)[:200])
    return parts, bad


def pick_files(files: list, ov: dict) -> list:
    """overrides.yaml files: regex on the repo path; the others are never opened (archives are not extracted)."""
    if not ov.get("files"):
        return files
    if not (kept := [f for f in files if re.search(ov["files"], f[1])]):
        raise Fail(f"files: no file matches {ov['files']!r}")
    return kept


def read_stats(parts: list[Part]) -> dict:
    """Encodings used and rows the csv reader skipped (filled in once a part has been read)."""
    enc = defaultdict(int)
    lossy = []
    for p in parts:
        enc[p.stats.get("encoding", "binary")] += 1
        n, sk = p.stats.get("rows", 0), p.stats.get("skipped", 0)
        if sk and sk > 0.01 * (n + sk):
            lossy.append(f"{p.name}: {sk} of {n + sk} rows skipped")
    return {"encodings": dict(enc), "skipped_rows": int(sum(p.stats.get("skipped", 0) for p in parts)),
            "files_over_1pct_skipped": lossy[:50]}


def main_table(parts: list[Part], sizes: dict, ov: dict) -> tuple[list[Part], list[str]]:
    from tideforge.tfbench.layouts import is_meta
    if ov.get("table"):
        parts = [p for p in parts if re.search(ov["table"], p.name)]
    elif any(not is_meta(p.name) for p in parts):   # metadata / README-like tables are never the data
        parts = [p for p in parts if not is_meta(p.name)]
    if ov.get("drop_parts"):
        parts = [p for p in parts if not re.search(ov["drop_parts"], p.name)]
    if ov.get("table_cols"):
        parts = [p for p in parts if set(ov["table_cols"]) <= set(p.schema.names)]
    groups = defaultdict(list)
    for p in parts:
        groups[tuple(sorted(p.schema.names))].append(p)
    if not groups:
        return [], []
    weight = {k: sum(sizes.get(p.name.split("!")[0], 1) for p in g) for k, g in groups.items()}
    best = max(groups, key=lambda k: (weight[k], len(groups[k])))
    other = [f"{len(g)} part(s) with columns {list(k)[:6]}" for k, g in groups.items() if k != best]
    return groups[best], other


# ---- items --------------------------------------------------------------------------------------------------------------
def _item_ids(df: pd.DataFrame, ids: list[str]) -> pd.Series:
    if len(ids) == 1:
        return df[ids[0]].astype("string").fillna("<NA>")
    return df[ids].astype("string").fillna("<NA>").agg("|".join, axis=1)


def enumerate_items(layout: str, parts: list[Part], r: dict, ov: dict) -> pd.DataFrame:
    """-> DataFrame(item_id, part, key, stratum)."""
    strata = ov.get("strata") or (r["id_cols"][0] if len(r["id_cols"]) >= 2 else None)
    rows = []
    multi = len(parts) > 1
    if layout in ("long", "events") and r["id_cols"]:
        cols = list(dict.fromkeys(r["id_cols"] + ([strata] if strata else [])))
        seen = {}
        for pi, p in enumerate(parts):
            for t in p.batches(cols):
                d = t.to_pandas()
                d["item_id"] = _item_ids(d, r["id_cols"])
                d = d.drop_duplicates("item_id")
                for iid, st in zip(d["item_id"], d[strata] if strata else [None] * len(d)):
                    seen.setdefault(iid, st)
        rows = [(i, -1, i, s) for i, s in seen.items()]
    elif layout in ("long", "events"):           # no id column: one item per part (file = series) or the whole table
        rows = [((p.name if multi else "all"), pi, None, None) for pi, p in enumerate(parts)]
    elif layout == "wide":
        for pi, p in enumerate(parts):
            for c in r["target_cols"]:
                rows.append(((f"{p.name}/{c}" if multi else c), pi, c, p.name if multi else None))
    elif layout == "nested":
        idc = r["id_cols"]
        for pi, p in enumerate(parts):
            cols = list(dict.fromkeys(idc + ([strata] if strata else [])))
            k = 0
            for t in p.batches(cols or [r["target_cols"][0]]):
                d = t.to_pandas()
                ids = _item_ids(d, idc) if idc else None
                for j in range(len(d)):
                    iid = str(ids.iloc[j]) if idc else f"{p.name}#{k + j}"
                    if idc and multi:
                        iid = f"{p.name}/{iid}"
                    rows.append((iid, pi, k + j, d[strata].iloc[j] if strata else None))
                k += len(d)
    else:
        raise Fail(f"layout {layout} not handled")
    return pd.DataFrame(rows, columns=["item_id", "part", "key", "stratum"]).drop_duplicates("item_id")


def stratified_order(items: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Frozen seeded order; within each stratum a random permutation, strata interleaved in proportion to their size."""
    rng = np.random.default_rng(seed)
    it = items.sort_values("item_id", kind="stable").reset_index(drop=True)
    st = it["stratum"].astype("string").fillna("<none>")
    key = np.zeros(len(it))
    for _, idx in it.groupby(st).indices.items():
        perm = rng.permutation(len(idx))
        key[idx] = (perm + rng.random(len(idx))) / len(idx)
    return it.assign(_k=key).sort_values(["_k", "item_id"]).drop(columns="_k").reset_index(drop=True)


# ---- loading ------------------------------------------------------------------------------------------------------------
def _timestamps(d: pd.DataFrame, tcol: str) -> pd.Series:
    return parse_time(d[tcol])


def load_long(parts, r, chunk: pd.DataFrame, cols_val: list[str], static: list[str], events: dict | None = None):
    want = set(chunk["item_id"])
    by_part = set(chunk["part"])
    tcol = r["time_col"]
    need = list(dict.fromkeys([tcol] + r["id_cols"] + cols_val + static + ([events["sum"]] if events and events.get("sum") else [])))
    out = []
    for pi, p in enumerate(parts):
        if -1 not in by_part and pi not in by_part:
            continue
        for t in p.batches([c for c in need if c in p.schema.names]):
            d = t.to_pandas()
            if r["id_cols"]:
                d["item_id"] = _item_ids(d, r["id_cols"])
                d = d[d["item_id"].isin(want)]
            else:
                d["item_id"] = chunk.loc[chunk["part"] == pi, "item_id"].iloc[0]
            if d.empty:
                continue
            d["timestamp"] = _timestamps(d, tcol)
            if events is not None:
                d["timestamp"] = d["timestamp"].dt.floor(events["step"]) if not events["step"].upper().endswith(("S", "E")) \
                    and not events["step"].upper().startswith("W") else d["timestamp"]
                v = d[events["sum"]].astype(float) if events.get("sum") else pd.Series(1.0, index=d.index)
                d = d.assign(_v=v).groupby(["item_id", "timestamp"], as_index=False)["_v"].sum()
                d = d.rename(columns={"_v": events["name"]})
            else:
                d = d[["item_id", "timestamp"] + cols_val + static]
            out.append(d)
    if not out:
        return pd.DataFrame(columns=["item_id", "timestamp"] + cols_val + static)
    df = pd.concat(out, ignore_index=True)
    if events is not None:
        df = df.groupby(["item_id", "timestamp"], as_index=False)[events["name"]].sum()
    return df


def load_wide(parts, r, chunk: pd.DataFrame, covs: list[str]):
    out = []
    for pi, g in chunk.groupby("part"):
        p = parts[int(pi)]
        cols = list(dict.fromkeys([r["time_col"]] + list(g["key"]) + covs))
        d = p.read([c for c in cols if c in p.schema.names]).to_pandas()
        ts = _timestamps(d, r["time_col"])
        for iid, c in zip(g["item_id"], g["key"]):
            x = pd.DataFrame({"item_id": iid, "timestamp": ts, "target": pd.to_numeric(d[c], errors="coerce")})
            for cv in covs:
                x[cv] = pd.to_numeric(d[cv], errors="coerce")
            out.append(x)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["item_id", "timestamp", "target"])


TIME_LIST = re.compile(r"(^|_)(time|times|timestamp|timestamps|ts|date|dates|datetime|epoch)($|_)", re.I)


def nested_time(schema: pa.Schema, ov: dict) -> tuple[str | None, str | None]:
    """-> (start column, time-list column). A time list is a list of timestamps, or a numeric list named like time
    (epoch seconds / milliseconds, e.g. GlucoFM-Bench, Endomondo). A list column is never a start column."""
    kinds = {f.name: _kind(f.type)[0] for f in schema}
    tlist = next((n for n, k in kinds.items() if k == "list_time"), None) or \
        next((n for n, k in kinds.items() if k == "list_num" and TIME_LIST.search(n)), None)
    start = ov.get("start_col") or next((c for c in ("start", "start_date", "timestamp", "date")
                                         if c in kinds and not kinds[c].startswith("list")), None)
    return start, tlist


def nested_target(schema: pa.Schema, r: dict, tlist: str | None) -> str:
    """First numeric list among the targets, then the covariates, that is not the time list."""
    kinds = {f.name: _kind(f.type)[0] for f in schema}
    for c in r["target_cols"] + r["covariate_cols"]:
        if c != tlist and kinds.get(c) == "list_num":
            return c
    if r["target_cols"] and r["target_cols"][0] != tlist:
        return r["target_cols"][0]
    raise Fail("nested: no target list besides the time list")


def epoch_times(v) -> pd.Series:
    """A time list -> timestamps. Numbers: epoch seconds (1e8..1e11) or milliseconds (1e11..1e14); else Fail."""
    s = pd.Series(v)
    if pd.api.types.is_numeric_dtype(s):
        a = pd.to_numeric(s, errors="coerce")
        m = float(np.nanmedian(np.abs(a))) if a.notna().any() else 0.0
        if 1e8 <= m < 1e11:
            return pd.Series(pd.to_datetime(a, unit="s", errors="coerce"))
        if 1e11 <= m < 1e14:
            return pd.Series(pd.to_datetime(a, unit="ms", errors="coerce"))
        raise Fail(f"nested: numeric time list is not epoch seconds or milliseconds (median {m:g})")
    return pd.Series(pd.to_datetime(s, errors="coerce"))


def load_nested(parts, r, chunk: pd.DataFrame, ov: dict, ds_freq: str | None, static: list[str]):
    """One row per series: target list (1-D, or 2-D = several channels -> target_0..k), start + freq or a time list."""
    start, tlist = nested_time(parts[0].schema, ov)
    tgt = nested_target(parts[0].schema, r, tlist)
    list_cov = [c for c in r["covariate_cols"] if c not in (tgt, tlist)]
    fcol = next((c for c in r.get("freq_cols", []) if c in parts[0].schema.names), None)
    if not tlist and not start:
        raise Fail("nested: no start column or timestamp list (set start_col in overrides.yaml)")
    out = []
    for pi, g in chunk.groupby("part"):
        p = parts[int(pi)]
        cols = [c for c in dict.fromkeys([tgt, start, tlist, fcol] + list_cov + static) if c and c in p.schema.names]
        t = p.read(cols).take(pa.array(np.asarray(g["key"], dtype=np.int64)))
        d = t.to_pylist()
        for iid, row in zip(g["item_id"], d):
            y = np.asarray(row[tgt], dtype=np.float64) if row[tgt] is not None else np.zeros(0)
            if y.ndim == 2 and y.shape[0] > y.shape[1]:
                y = y.T
            n = y.shape[-1]
            if tlist:
                ts = epoch_times(row[tlist] or [])
                if len(ts) != n:
                    m = min(len(ts), n)
                    ts, y = ts.iloc[:m], y[..., :m]
                    n = m
            else:
                f = ov.get("freq") or (row.get(fcol) if fcol else None) or ds_freq
                if not f:
                    raise Fail("nested: no frequency (no freq column, none in overrides.yaml or the profile)")
                fa = _gluonts_alias(f)
                if ov.get("synthetic_time"):
                    keep = min(n, synthetic_capacity(fa))
                    if keep < n:          # the synthetic clock cannot hold the whole series: keep its last points
                        y = y[..., n - keep:]
                        n = keep
                    ts = pd.Series(pd.date_range(SYNTHETIC_START, periods=n, freq=fa))
                else:
                    try:
                        ts = pd.Series(pd.date_range(pd.Timestamp(row[start]), periods=n, freq=fa))
                    except (pd.errors.OutOfBoundsDatetime, OverflowError, ValueError) as e:
                        raise NeedSyntheticTime(f"{iid}: start {row[start]!r}, {n} steps of {fa}: {e}")
            x = pd.DataFrame({"item_id": iid, "timestamp": parse_time(ts)})
            if y.ndim == 1:
                x["target"] = y
            else:
                for k in range(y.shape[0]):
                    x[f"target_{k}"] = y[k]
            for c in list_cov:
                v = row.get(c)
                if v is None:
                    continue
                a = np.asarray(v, dtype=np.float64)
                if ov.get("synthetic_time") and a.shape[-1] > n:
                    a = a[..., a.shape[-1] - n:]
                a = a.T if a.ndim == 2 and a.shape[0] > a.shape[1] and a.shape[0] == n else a
                if a.ndim == 1 and len(a) >= n:
                    x[c] = a[:n]
                elif a.ndim == 2 and a.shape[1] >= n:
                    for k in range(a.shape[0]):
                        x[f"{c}_{k}"] = a[k, :n]
            for c in static:
                x[c] = row.get(c)
            out.append(x)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["item_id", "timestamp", "target"])


def _gluonts_alias(f: str) -> str:
    f = str(f)
    m = re.fullmatch(r"(\d*)([A-Za-z]+)(-\w+)?", f)
    if not m:
        return f
    n, u, tail = m.group(1), m.group(2), m.group(3) or ""
    u = {"T": "min", "H": "h", "S": "s", "L": "ms", "M": "MS", "Q": "QS", "A": "YS", "Y": "YS"}.get(u, u)
    return f"{n}{u}{tail}"


# ---- filters ------------------------------------------------------------------------------------------------------------
def series_filter(x: pd.DataFrame, targets: list[str], min_len: int, max_nan: float) -> str | None:
    v = x[targets].to_numpy(dtype=np.float64)
    if len(v) < min_len:
        return "too_short"
    if np.isnan(v).mean() > max_nan:
        return "mostly_missing"
    f = v[np.isfinite(v)]
    if not len(f) or np.all(f == f[0]):
        return "constant"
    return None


def series_hash(x: pd.DataFrame, targets: list[str]) -> str:
    v = np.nan_to_num(x[targets].to_numpy(dtype=np.float64), nan=np.inf)
    return hashlib.sha1(v.tobytes()).hexdigest()


# ---- build --------------------------------------------------------------------------------------------------------------
def build_dataset(row: dict, files: list[tuple[Path, str, int]], ov: dict, s: Settings, out_dir: Path,
                  workdir: Path, refs_dir: Path | None = None) -> dict:
    """row: selection row + profile fields. files: (local path, repo path, bytes). Writes out_dir/{data.parquet, meta.json}."""
    did = row["dataset_id"]
    meta = {"dataset_id": did, "source": {"repo": row["repo_id"], "config": row.get("config_or_subset", ""),
                                          "catalog_row": int(row["catalog_row"]), "files": [f[1] for f in files]},
            "licence": {"hub": row.get("licence_hub"), "catalog": row.get("catalog_licence")},
            "catalog_status": row.get("status"), "domain": row.get("domain"), "transformations": [],
            "profile": {"frequency": row.get("frequency"), "freq_seconds": row.get("freq_seconds"),
                        "layout": row.get("layout")}}
    T = meta["transformations"]
    if ov.get("skip"):
        raise Fail(f"skipped in overrides.yaml: {ov['skip']}")
    if row.get("status") == "keep_filtered" and not (ov.get("drop_items") or ov.get("drop_parts") or ov.get("table")
                                                     or ov.get("ref_ids") or ov.get("audit_only")):
        raise Fail("keep_filtered: no drop rule in overrides.yaml (drop_items / drop_parts / table)")
    n_files = len(files)
    files = pick_files(files, ov)
    if len(files) < n_files:
        T.append({"step": "files (overrides.yaml)", "pattern": ov["files"], "kept": len(files), "of": n_files})
    parts, bad = open_all([(f[0], f[1]) for f in files], workdir, did)
    if bad:
        T.append({"step": "unreadable files skipped", "count": len(bad), "examples": bad[:5]})
    if not parts:
        raise Fail("no readable table" + (f" ({bad[0]})" if bad else ""))
    sizes = {f[1]: f[2] for f in files}
    parts, other = main_table(parts, sizes, ov)
    if not parts:
        raise Fail("no table left after the table / drop_parts overrides")
    if other:
        T.append({"step": "other tables not used (different columns)", "tables": other[:10], "count": len(other)})
    from tideforge.tfbench.layouts import adapt
    parts, ov, notes = adapt(parts, ov, row, roles_for)
    T += notes
    r, layout, layout_rule = roles_for(parts[0].schema, ov)
    if layout == "unknown":
        raise Fail(f"layout unknown ({layout_rule})")
    if layout != "nested" and not r["time_col"]:
        raise Fail("no time column (set time_col in overrides.yaml)")
    if layout in ("long", "wide", "events") and not r["target_cols"] and layout != "events":
        raise Fail("no numeric target column")
    meta.update(layout=layout, layout_rule=layout_rule, time_col=r["time_col"], id_cols=r["id_cols"],
                target_rule=r.get("target_rule"))

    # columns kept
    schema = parts[0].schema
    num = {f.name for f in schema if _kind(f.type)[0] in ("num", "bool", "list_num")}
    covs = [c for c in r["covariate_cols"] if c in num]
    dropped_cov = [c for c in r["covariate_cols"] if c not in num]
    if dropped_cov:
        T.append({"step": "non-numeric covariates not used", "columns": dropped_cov})
    if len(covs) > s.max_covariates:
        T.append({"step": f"covariates capped at {s.max_covariates}", "dropped": covs[s.max_covariates:]})
        covs = covs[:s.max_covariates]
    if layout == "wide":
        covs = [c for c in covs if c in (ov.get("covariates") or [])]   # wide: only stated covariates are shared
    static = [c for c in r["static_cols"] if c in schema.names and c not in r["id_cols"]][:16]
    if layout == "wide":
        static = []

    events = None
    if layout == "events":
        step = ov.get("event_step", s.event_step)
        name = f"sum_{ov['event_sum']}" if ov.get("event_sum") else "count"
        events = {"step": step, "sum": ov.get("event_sum"), "name": name}
        meta["event_rule"] = {"step": step, "aggregate": "sum of " + ov["event_sum"] if ov.get("event_sum") else "count of rows",
                              "empty steps": "0 inside each series' span (no event = zero count)"}
        covs, static = [], []

    # items, drop rule, order
    items = enumerate_items(layout, parts, r, ov)
    n_items = len(items)
    if ov.get("drop_items"):
        pat = re.compile(ov["drop_items"])
        m = items["item_id"].astype(str).map(lambda x: bool(pat.search(x)))
        T.append({"step": "drop_items (catalog overlap rule)", "pattern": ov["drop_items"], "count": int(m.sum())})
        items = items[~m]
    if ov.get("ref_ids"):
        ids = ref_ids(Path(refs_dir) if refs_dir else None, ov["ref_ids"], workdir)
        key = items["item_id"].astype(str)
        m = key.isin(ids) | key.str.rsplit("/", n=1).str[-1].isin(ids) | key.str.split("!").str[0].str.split("/").str[0].isin(ids)
        T.append({"step": "ref_ids (drop series a reference benchmark uses, by id)", "rule": ov["ref_ids"],
                  "ids_in_reference": len(ids), "count": int(m.sum())})
        items = items[~m]
    if ov.get("audit_only"):
        T.append({"step": "keep_filtered handled by the value audit only", "note": str(ov["audit_only"])})
    if items.empty:
        raise Fail("no items")
    items = stratified_order(items, s.seed)

    for attempt in (0, 1):
        try:
            targets: list[str] = []
            freq = ov.get("freq")
            freq_share = None
            min_len = None
            kept, drops, hashes = [], defaultdict(list), set()
            reg_stats = defaultdict(int)
            for c0 in range(0, len(items), s.max_series):
                if len(kept) >= s.max_series:
                    break
                chunk = items.iloc[c0:c0 + s.max_series]
                if layout in ("long", "events"):
                    df = load_long(parts, r, chunk, ([] if events else r["target_cols"] + covs), static, events)
                elif layout == "wide":
                    df = load_wide(parts, r, chunk, covs)
                else:
                    df = load_nested(parts, r, chunk, ov, freq or _alias_from_profile(row), static)
                if df.empty:
                    continue
                n_nat = int(df["timestamp"].isna().sum())
                if n_nat:
                    reg_stats["rows without a readable timestamp"] += n_nat
                    df = df.dropna(subset=["timestamp"])
                if not targets:
                    targets = [events["name"]] if events else (
                        [c for c in df.columns if c == "target" or c.startswith("target_")] if layout in ("wide", "nested")
                        else list(r["target_cols"]))
                    if layout == "nested":
                        covs = [c for c in df.columns if c not in ["item_id", "timestamp"] + targets + static]
                if not freq:
                    freq, freq_share, _ = infer_freq(g for _, g in df.groupby("item_id")["timestamp"])
                    if not freq:
                        raise Fail("frequency: fewer than two timestamps per series")
                if min_len is None:
                    min_len = rules.min_length(freq, s)
                vals = targets + [c for c in covs if c in df.columns]
                for iid, g in df.groupby("item_id", sort=False):
                    agg = {**(ov.get("agg") or {}), **({events["name"]: "sum"} if events else {})}
                    x, st = regularise(g.drop(columns="item_id"), freq, vals, agg=agg, 
                                       fill_zero=bool(events or ov.get("fill_zero")),
                                       keep_last=s.max_points)
                    for k, v in st.items():
                        reg_stats[k] += v
                    why = series_filter(x, targets, min_len, s.max_nan_share)
                    if why is None:
                        h = series_hash(x, targets)
                        if h in hashes:
                            why = "duplicate"
                        hashes.add(h)
                    if why:
                        drops[why].append(iid)
                        continue
                    kept.append(x.assign(item_id=iid))
                    if len(kept) >= s.max_series:
                        break
            break
        except NeedSyntheticTime as e:
            if attempt or layout != "nested":
                raise Fail(f"timestamps out of range even on the synthetic clock: {e}")
            ov = {**ov, "synthetic_time": True}
            T.append({"step": "synthetic time", "rule": f"real timestamps do not fit pandas' range ({str(e)[:200]}); every "
                      f"series re-indexed from {SYNTHETIC_START.date()} at its frequency; calendar position is not real"})
    if not kept:
        raise Fail(f"no series left after the filters ({ {k: len(v) for k, v in drops.items()} }, min length {min_len})")

    T.append({"step": "read", **read_stats(parts)})
    if ov.get("fill_zero"):
        T.append({"step": "fill_zero (overrides.yaml)", "rule": "missing grid steps = 0: the source has rows only for "
                  "non-zero values"})
    covs = [c for c in covs if c in kept[0].columns]
    roles = cov_roles(covs, list(ov.get("known") or []))
    data = pd.concat(kept, ignore_index=True)
    data = data[["item_id", "timestamp"] + targets + covs + [c for c in static if c in data.columns]]
    for c in targets + covs:
        data[c] = pd.to_numeric(data[c], errors="coerce").astype("float64")
    for c in static:
        if c in data.columns:
            data[c] = data[c].astype("string")
    lens = data.groupby("item_id").size()

    pf = row.get("freq_seconds")
    fs = rules.freq_seconds(freq)
    conflict = bool(pf == pf and pf not in (None, "") and float(pf) > 0 and abs(np.log(fs / float(pf))) > 0.1)
    meta.update(
        synthetic_time=bool(ov.get("synthetic_time")),
        frequency=freq, freq_share=freq_share, freq_source="overrides.yaml" if ov.get("freq") else "timestamps (mode of differences)",
        freq_conflict_with_profile=conflict, freq_band=rules.band(freq), short_horizon=rules.short_horizon(freq),
        min_length=min_len, targets=targets, covariates={c: roles[c] for c in covs}, static=[c for c in static if c in data.columns],
        n_items_source=n_items, n_series=int(lens.size), n_points=int(len(data)),
        n_values_target=int(data[targets].notna().sum().sum()),
        length={"min": int(lens.min()), "median": float(lens.median()), "max": int(lens.max())},
        date_first=str(data["timestamp"].min()), date_last=str(data["timestamp"].max()),
        nan_share=float(data[targets].isna().to_numpy().mean()), note=ov.get("note"),
    )
    T.append({"step": "regularise", "rule": f"snap to {freq} grid, duplicates averaged (agg overrides: {ov.get('agg') or {}}), "
              "missing steps = NaN, never filled" + (" (event counts: 0)" if events else ""), **dict(reg_stats)})
    T.append({"step": "series filters", "min_length": min_len, "max_nan_share": s.max_nan_share,
              "dropped": {k: len(v) for k, v in drops.items()}, "examples": {k: v[:5] for k, v in drops.items()}})
    T.append({"step": "cap", "rule": f"first {s.max_series} accepted series in a seeded stratified order (seed {s.seed}, "
              f"strata {ov.get('strata') or (r['id_cols'][0] if len(r['id_cols']) >= 2 else 'none')}); last {s.max_points} points",
              "series_considered": int(sum(len(v) for v in drops.values()) + len(kept))})

    tmp = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    pq.write_table(pa.Table.from_pandas(data, preserve_index=False), tmp / "data.parquet", compression="zstd")
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp.rename(out_dir)
    return meta


def ref_ids(refs_dir: Path | None, rule: dict, workdir: Path) -> set[str]:
    """Series ids from reference benchmark files: rule = {path: regex on file paths under refs_dir, column: id column
    (default the first of item_id / id / series_id / unique_id present)}."""
    from tideforge.tfbench.download import ext
    out: set[str] = set()
    if not refs_dir or not Path(refs_dir).exists():
        return out
    for f in sorted(Path(refs_dir).rglob("*")):
        rel = f.relative_to(refs_dir).as_posix()
        if not f.is_file() or not re.search(rule["path"], rel) or ext(f.name) not in ("parquet", "arrow", "csv", "jsonl"):
            continue
        try:
            for p in open_parts(f, workdir, rel):
                col = rule.get("column") or next((c for c in ("item_id", "id", "series_id", "unique_id") if c in p.schema.names), None)
                if col and col in p.schema.names:
                    for t in p.batches([col]):
                        out |= {str(v) for v in t.column(0).to_pylist() if v is not None}
        except Unreadable:
            continue
    return out


def plan_dataset(row: dict, files: list[tuple[Path, str, int]], ov: dict, s: Settings, workdir: Path) -> dict:
    """canon --plan: what build_dataset would use, without building. Reads schemas, the id / list-length / start columns
    only. -> {status, reason, layout, rule, time_col, id_cols, targets, covariates, parts, unreadable, nested rows,
    rows whose real timestamps overflow (-> synthetic time)}."""
    import pyarrow.compute as pc
    out = {"dataset_id": row["dataset_id"], "status": "ok", "reason": "", "files": len(files)}
    try:
        if ov.get("skip"):
            raise Fail(f"skipped in overrides.yaml: {ov['skip']}")
        if row.get("status") == "keep_filtered" and not (ov.get("drop_items") or ov.get("drop_parts") or ov.get("table")
                                                         or ov.get("ref_ids") or ov.get("audit_only")):
            raise Fail("keep_filtered: no drop rule in overrides.yaml")
        files = pick_files(files, ov)
        parts, bad = open_all([(f[0], f[1]) for f in files], workdir, row["dataset_id"])
        out["unreadable"] = len(bad)
        if bad:
            out["unreadable_example"] = bad[0][:200]
        if not parts:
            raise Fail("no readable table" + (f" ({bad[0]})" if bad else ""))
        parts, other = main_table(parts, {f[1]: f[2] for f in files}, ov)
        if not parts:
            raise Fail("no table left after the table / drop_parts overrides")
        from tideforge.tfbench.layouts import adapt
        out["main_table"] = parts[0].name[:200]
        parts, ov, notes = adapt(parts, ov, row, roles_for)
        out["generic_layout"] = "; ".join(n["step"] for n in notes)
        if ov.get("synthetic_time"):
            out["synthetic_time"] = True
        r, layout, rule = roles_for(parts[0].schema, ov)
        out.update(layout=layout, rule=rule, time_col=r["time_col"], id_cols=";".join(r["id_cols"]),
                   targets=";".join(r["target_cols"][:5]), n_targets=len(r["target_cols"]),
                   covariates=len(r["covariate_cols"]), parts=len(parts), other_tables=len(other),
                   columns=";".join(parts[0].schema.names[:30]))
        if layout == "unknown":
            raise Fail(f"layout unknown ({rule})")
        if layout != "nested" and not r["time_col"]:
            raise Fail("no time column (set time_col in overrides.yaml)")
        if layout == "nested":
            names = parts[0].schema.names
            start, tlist = nested_time(parts[0].schema, ov)
            tgt = nested_target(parts[0].schema, r, tlist)
            out["time_list"] = tlist or ""
            fcol = next((c for c in r.get("freq_cols", []) if c in names), None)
            if not tlist and not start:
                raise Fail("nested: no start column or timestamp list (set start_col in overrides.yaml)")
            rows = over = 0
            first_bad = ""
            if start and not tlist:
                for p in parts:
                    for t in p.batches([c for c in (tgt, start, fcol) if c]):
                        lens = pc.list_value_length(t.column(tgt)).to_numpy(zero_copy_only=False)
                        lens = np.where(np.isnan(lens.astype(float)), 0, lens).astype(np.int64)
                        if lens.ndim and _kind(t.schema.field(tgt).type)[1] == 2:   # 2-D: steps are the inner length
                            inner = pc.list_value_length(pc.list_flatten(t.column(tgt))).to_numpy(zero_copy_only=False)
                            lens = np.full(len(lens), int(np.nanmax(inner)) if len(inner) else 0)
                        st = t.column(start).to_pylist()
                        fq = t.column(fcol).to_pylist() if fcol else [None] * len(st)
                        for n, a, f in zip(lens, st, fq):
                            rows += 1
                            fa = _gluonts_alias(ov.get("freq") or f or _alias_from_profile(row) or "D")
                            try:
                                pd.date_range(pd.Timestamp(a), periods=max(int(n), 1), freq=fa)
                            except (pd.errors.OutOfBoundsDatetime, OverflowError, ValueError) as e:
                                over += 1
                                first_bad = first_bad or f"start {a!r}, {n} x {fa}: {str(e)[:80]}"
            out.update(nested_rows=rows, rows_out_of_range=over, synthetic_time=bool(over or ov.get("synthetic_time")),
                       out_of_range_example=first_bad)
    except Fail as e:
        out.update(status="not built", reason=str(e)[:300])
    except Exception as e:
        out.update(status="error", reason=f"{type(e).__name__}: {str(e)[:300]}")
    return out


def _alias_from_profile(row: dict) -> str | None:
    fs = row.get("freq_seconds")
    try:
        fs = float(fs)
    except (TypeError, ValueError):
        return None
    if fs != fs or fs <= 0:
        return None
    for lo, hi, a in ((27 * 86400, 32 * 86400, "MS"), (88 * 86400, 93 * 86400, "QS"), (360 * 86400, 370 * 86400, "YS")):
        if lo <= fs <= hi:
            return a
    return pd.tseries.frequencies.to_offset(pd.Timedelta(seconds=fs)).freqstr
