"""Profile every dataset of the Hugging Face catalog from metadata only (owner task 2026-11-12).

    python -m tideforge.devdata.hf_profile            # full run: hf_profile.csv, hf_profile_files.csv, HF_PROFILE.md
    python -m tideforge.devdata.hf_profile --plan     # repo listings only (~420 calls), then the planned call counts
    python -m tideforge.devdata.hf_profile --offline  # rebuild the outputs from the cache, no network

HARD RULE: no dataset file is downloaded. Every network call goes through tideforge.devdata.hf_guard.Fetcher.get, which
allows only (a) Hub API metadata, (b) the dataset viewer API, (c) parquet footers by suffix range, (d) small text
metadata files under 1 MB, counts the bytes and aborts above the budget (2 GB). See hf_guard for the exact rules.

Per repo: (1) repo info with the file listing; (2) viewer /is-valid, /size, /info, /statistics; (3) parquet footers of the
repo's own parquet files, or of the viewer's converted files (refs/convert/parquet) when the scope has no parquet;
(4) otherwise files_only. A catalog row is mapped to a scope (config / folder / files) by `resolve_scope`; the rule that
matched is in the output. A row that covers a whole repo with several configs also gets one extra row per config
(row_kind = "config"); totals use the catalog rows only. Derivation rules: tideforge.devdata.hf_derive.

HF_TOKEN is read from the environment and handed to the Fetcher; it is never printed, logged or written.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import decimal
import fnmatch
import io
import json
import os
import re
import struct
import sys
import threading
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import hf_derive as D
from .hf_guard import BUDGET, HUB, VIEWER, BudgetError, Fetcher, FetchError, GuardError

HERE = Path(__file__).resolve().parent
COLLECTIONS = {"salesforce/lotsa_data", "autogluon/chronos_datasets", "nju-zhangsq/rmisc", "ai-iot/energybench",
               "salesforce/cloudops_tsf", "cognitedata/ind-eval"}
DOUBLE_EXT = ("csv.gz", "tar.gz", "json.gz", "jsonl.gz", "jsonl.zst", "csv.zst", "tsv.gz", "txt.gz", "csv.bz2", "csv.xz")
NON_DATA_EXT = {"md", "py", "ipynb", "png", "jpg", "jpeg", "gif", "svg", "pdf", "bib", "yaml", "yml", "toml", "cfg", "sh",
                "html", "tex", "lock", "ico", "gitattributes", "gitignore", "rst", "css", "js", "webp", "docx"}
NON_DATA_NAME = re.compile(r"(?i)^(readme|license|licence|notice|citation|requirements|\.git).*|^(state|dataset_dict|"
                           r"dataset_infos?|croissant)\.json$|.*(meta|info)[^/]*\.(json|ya?ml)$")


# ---- small helpers -----------------------------------------------------------------------------------------------------
def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def ext_of(path: str) -> str:
    name = path.rsplit("/", 1)[-1].lower()
    for e in DOUBLE_EXT:
        if name.endswith("." + e):
            return e
    return name.rsplit(".", 1)[-1] if "." in name.strip(".") else "(none)"


def is_data_file(path: str, size: int | None) -> bool:
    name = path.rsplit("/", 1)[-1]
    e = ext_of(path)
    if NON_DATA_NAME.match(name) or e in NON_DATA_EXT:
        return False
    if e == "(none)":
        return bool(size and size > 1_000_000)
    return True


def glob_regex(pattern: str) -> re.Pattern:
    out, i, p = [], 0, pattern.lstrip("./")
    while i < len(p):
        if p.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif p.startswith("**", i):
            out.append(".*")
            i += 2
        elif p[i] == "*":
            out.append("[^/]*")
            i += 1
        elif p[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(p[i]))
            i += 1
    body = "".join(out)
    if not re.search(r"[*?]", pattern) and "." not in pattern.rsplit("/", 1)[-1]:
        body = body.rstrip("/") + "/.*"  # a bare directory
    return re.compile("^" + body + "$")


def card_configs(card: dict) -> dict[str, list[str]]:
    out = {}
    for c in (card or {}).get("configs") or []:
        if not isinstance(c, dict):
            continue
        pats, df = [], c.get("data_files")
        items = df if isinstance(df, list) else [df] if df else []
        for x in items:
            if isinstance(x, str):
                pats.append(x)
            elif isinstance(x, dict):
                p = x.get("path")
                pats += [p] if isinstance(p, str) else list(p or [])
        out[str(c.get("config_name") or "default")] = pats
    return out


def resolve_url(repo: str, path: str, rev: str = "main") -> str:
    return f"https://{HUB}/datasets/{repo}/resolve/{urllib.parse.quote(rev, safe='')}/{urllib.parse.quote(path)}"


def viewer_url(endpoint: str, **params) -> str:
    return f"https://{VIEWER}/{endpoint}?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})


def subset_name(text: str) -> str:
    """The catalog's config_or_subset without its bracketed remark: "sdwpf_full (2020-01..2021-12)" -> "sdwpf_full"."""
    return re.split(r"\s*\(", text or "", maxsplit=1)[0].strip()


def read_catalog(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows):
        r["_row"] = i + 2  # line number in the csv (header = line 1)
        for k in list(r):
            if isinstance(r[k], str):
                r[k] = r[k].strip()
    return rows


# ---- scope: which files / configs a catalog row means ---------------------------------------------------------------------
class Scope:
    def __init__(self, files, configs=None, rule="whole repo", unmatched=(), whole=False, name=""):
        self.files, self.configs, self.rule = files, configs, rule
        self.unmatched, self.whole, self.name = list(unmatched), whole, name


def _match_one(part: str, files: list[dict], config_names: list[str], card: dict[str, list[str]]):
    """-> (rule, files, config or None) or None. Order: config, exact path, folder prefix, path segment, glob."""
    p, n = part.strip().strip("/"), norm(part)
    if not n:
        return None
    for c in config_names:
        if c == p or c.lower() == p.lower() or norm(c) == n:
            pats = [glob_regex(x) for x in card.get(c, [])]
            fs = [f for f in files if any(rx.match(f["path"]) for rx in pats)] if pats else []
            if not fs:
                fs = [f for f in files if any(norm(s) == norm(c) for s in f["path"].split("/")[:-1])]
            if not fs and len(config_names) == 1:
                fs = list(files)
            return f"config:{c}", fs, c
    fs = [f for f in files if f["path"] == p or f["path"].rsplit("/", 1)[-1] == p]
    if fs:
        return f"file:{p}", fs, None
    fs = [f for f in files if f["path"].lower().startswith(p.lower() + "/")]
    if fs:
        return f"folder:{p}", fs, None
    fs = [f for f in files if any(norm(s) == n for s in f["path"].split("/")[:-1])]
    if fs:
        return f"path segment:{p}", fs, None
    fs = [f for f in files if norm(f["path"].rsplit("/", 1)[-1].rsplit(".", 1)[0]) == n]
    if fs:
        return f"file stem:{p}", fs, None
    if "*" in p:
        pl = p.lower()
        fs = [f for f in files if fnmatch.fnmatch(f["path"].lower(), pl) or fnmatch.fnmatch(f["path"].lower(), "*/" + pl)
              or any(fnmatch.fnmatch(s.lower(), pl) for s in f["path"].split("/"))]
        if fs:
            return f"glob:{p}", fs, None
    return None


def resolve_scope(text: str, files: list[dict], config_names: list[str], card: dict[str, list[str]]) -> Scope:
    primary = subset_name(text)
    if primary.lower() in ("", "unknown", "all", "whole repo") or (primary.lower() == "default" and len(config_names) <= 1):
        return Scope(list(files), None, "whole repo", whole=True, name="")
    m = _match_one(primary, files, config_names, card)
    parts = [primary] if m else [x for x in re.split(r"\s*(?:,|;|\s/\s|\s\+\s|\sand\s|\svs\s)\s*", primary) if x.strip()]
    rules, got, configs, unmatched = [], {}, [], []
    for part in parts:
        mm = m if (m and part == primary) else _match_one(part, files, config_names, card)
        if not mm:
            unmatched.append(part)
            continue
        rules.append(mm[0])
        for f in mm[1]:
            got[f["path"]] = f
        if mm[2]:
            configs.append(mm[2])
    if not rules:
        return Scope(list(files), None, "subset text not matched to a config, folder or file: whole repo profiled",
                     unmatched, whole=True, name="")
    only_configs = configs if len(configs) == len(rules) else (configs or None)
    return Scope(list(got.values()), only_configs if configs else [], "; ".join(rules), unmatched,
                 name=", ".join(r.split(":", 1)[1] for r in rules))


# ---- parquet footers ---------------------------------------------------------------------------------------------------------
def footer_bytes(fx: Fetcher, url: str, size: int | None, tail: int = 16384) -> tuple[bytes | None, str]:
    """The footer (thrift metadata + 4-byte length + PAR1), by suffix range only. -> (bytes or None, error text)."""
    if size is not None and size < 16:
        return None, "file smaller than a parquet footer"
    n = tail
    if size is not None and size <= tail:  # small file: read the 8-byte trailer first, then exactly the footer
        r = fx.get(url, "bytes=-8", size)
        if r.status != 206:
            return None, r.error_text
        if r.body[-4:] != b"PAR1":
            return None, "no PAR1 trailer (not parquet, or encrypted)"
        n = struct.unpack("<I", r.body[-8:-4])[0] + 8
        if n >= size:
            return None, "footer length is not smaller than the file"
    if n > fx.footer_max:
        return None, f"footer of {n} bytes is above the limit of {fx.footer_max}"
    r = fx.get(url, f"bytes=-{n}", size)
    if r.status != 206:
        return None, r.error_text
    t = r.body
    if len(t) < 8 or t[-4:] != b"PAR1":
        return None, "no PAR1 trailer (not parquet, or encrypted)"
    need = struct.unpack("<I", t[-8:-4])[0] + 8
    if need > len(t):
        if need > fx.footer_max:
            return None, f"footer of {need} bytes is above the limit of {fx.footer_max}"
        if size is not None and need >= size:
            return None, "footer length is not smaller than the file"
        r = fx.get(url, f"bytes=-{need}", size)
        if r.status != 206:
            return None, r.error_text
        t = r.body
        if len(t) < need:
            return None, "short footer read"
    return t[-need:], ""


def _plain(v):
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, (bytes, bytearray)):
        try:
            return bytes(v).decode("utf-8")
        except Exception:
            return None
    if isinstance(v, dt.datetime):
        if v.tzinfo is not None:
            v = v.astimezone(dt.timezone.utc).replace(tzinfo=None)
        return dt.datetime(v.year, v.month, v.day, v.hour, v.minute, v.second)
    if isinstance(v, (dt.date, str, int, float)) or v is None:
        return v
    try:
        return v.item()  # numpy scalar
    except Exception:
        return None


def parse_footer(buf: bytes) -> dict:
    """Footer bytes -> {"rows", "leaves", "stats": {column: {min, max, nulls, num_values, compressed}}, "row_groups"}."""
    import pyarrow.parquet as pq

    md = pq.read_metadata(io.BytesIO(b"PAR1" + buf))
    sc, ncol = md.schema, md.num_columns
    leaves = []
    for i in range(ncol):
        c = sc.column(i)
        leaves.append({"path": c.path, "physical": c.physical_type, "logical": str(c.logical_type),
                       "rep": int(c.max_repetition_level)})
    names = [D.leaf_column_name(lf["path"], lf["rep"]) for lf in leaves]
    single = {n for n in names if names.count(n) == 1} if ncol <= 5000 else set(names)
    stats: dict[str, dict] = {}
    for g in range(md.num_row_groups):
        rg = md.row_group(g)
        for j in range(ncol):
            cc = rg.column(j)
            s = stats.setdefault(names[j], {"min": None, "max": None, "nulls": 0, "num_values": 0, "compressed": 0,
                                            "minmax_ok": True, "nulls_ok": True})
            s["num_values"] += int(cc.num_values)
            s["compressed"] += int(cc.total_compressed_size)
            try:
                st = cc.statistics if cc.is_stats_set else None
            except Exception:
                st = None
            if st is None:
                s["minmax_ok"] = s["nulls_ok"] = False
                continue
            try:
                nc = st.null_count if st.has_null_count else None
            except Exception:
                nc = None
            if nc is None:
                s["nulls_ok"] = False
            else:
                s["nulls"] += int(nc)
            if names[j] not in single:
                continue
            try:
                lo, hi = (_plain(st.min), _plain(st.max)) if st.has_min_max else (None, None)
            except Exception:
                lo = hi = None
            if lo is None or hi is None:
                if int(cc.num_values) > (nc or 0):
                    s["minmax_ok"] = False
                continue
            try:
                s["min"] = lo if s["min"] is None or lo < s["min"] else s["min"]
                s["max"] = hi if s["max"] is None or hi > s["max"] else s["max"]
            except TypeError:
                s["minmax_ok"] = False
    for s in stats.values():
        if not s.pop("minmax_ok"):
            s["min"] = s["max"] = None
        if not s.pop("nulls_ok"):
            s["nulls"] = None
    features = None
    try:  # `datasets` writes its features into the footer's key-value metadata
        hf = (md.metadata or {}).get(b"huggingface")
        features = json.loads(hf.decode("utf-8")).get("info", {}).get("features") if hf else None
    except Exception:
        features = None
    return {"rows": int(md.num_rows), "leaves": leaves, "stats": stats, "row_groups": int(md.num_row_groups),
            "features": features}


def _merge_stats(into: dict, new: dict):
    for name, s in new.items():
        t = into.get(name)
        if t is None:
            into[name] = dict(s)
            continue
        for k in ("num_values", "compressed"):
            t[k] = (t.get(k) or 0) + (s.get(k) or 0)
        t["nulls"] = None if t.get("nulls") is None or s.get("nulls") is None else t["nulls"] + s["nulls"]
        for k, better in (("min", lambda a, b: a < b), ("max", lambda a, b: a > b)):
            a, b = s.get(k), t.get(k)
            if a is None or b is None:  # one file without min / max: the range over all files is not known
                t[k] = None
                continue
            try:
                t[k] = a if better(a, b) else b
            except TypeError:
                t[k] = None


def pick_evenly(items: list, cap: int) -> list:
    if cap <= 0:
        return []
    if len(items) <= cap:
        return list(items)
    if cap == 1:
        return [items[-1]]
    idx = sorted({round(i * (len(items) - 1) / (cap - 1)) for i in range(cap)})
    return [items[i] for i in idx]


# ---- one repo ----------------------------------------------------------------------------------------------------------------
class RepoState:
    """Everything fetched for one repo; lazy viewer calls; each answer is cached by the Fetcher."""

    def __init__(self, fx: Fetcher, repo: str, opts):
        self.fx, self.repo, self.opts = fx, repo, opts
        self.info, self.info_error, self.exists = None, "", False
        self.files: list[dict] = []
        self.card: dict = {}
        self.access_denied = False
        self._viewer: dict = {}
        self._footers: dict = {}
        self.footer_errors: list[str] = []

    def load_info(self):
        r = self.fx.get(f"https://{HUB}/api/datasets/{self.repo}?blobs=true")
        if r.status != 200:
            self.info_error = r.error_text
            return
        self.info = r.json()
        self.exists = True
        self.card = self.info.get("cardData") or {}
        for s in self.info.get("siblings") or []:
            size = s.get("size")
            if isinstance(s.get("lfs"), dict) and s["lfs"].get("size") is not None:
                size = s["lfs"]["size"]
            self.files.append({"path": s.get("rfilename", ""), "size": size, "ext": ext_of(s.get("rfilename", "")),
                               "data": is_data_file(s.get("rfilename", ""), size)})

    def viewer(self, endpoint: str, **params):
        """-> (json or None, error text). 401 / 403 marks the repo as not accessible."""
        key = (endpoint, tuple(sorted(params.items())))
        if key not in self._viewer:
            try:
                r = self.fx.get(viewer_url(endpoint, dataset=self.repo, **params))
                if r.status in (401, 403):
                    self.access_denied = True
                self._viewer[key] = (r.json(), "") if r.status == 200 else (None, r.error_text)
            except (BudgetError, GuardError):
                raise
            except Exception as e:
                self._viewer[key] = (None, f"{type(e).__name__}: {e}")
        return self._viewer[key]

    def footer(self, url: str, size):
        if url not in self._footers:
            try:
                buf, err = footer_bytes(self.fx, url, size, self.opts.footer_tail)
                if buf is None:
                    if err.startswith(("HTTP 401", "HTTP 403")):
                        self.access_denied = True
                    self._footers[url] = (None, err)
                else:
                    self._footers[url] = (parse_footer(buf), "")
            except (BudgetError,):
                raise
            except Exception as e:
                self._footers[url] = (None, f"{type(e).__name__}: {e}")
        return self._footers[url]

    def text_json(self, path: str, size):
        try:
            r = self.fx.get(resolve_url(self.repo, path), None, size)
            if r.status in (401, 403):
                self.access_denied = True
            return r.json() if r.status == 200 else None
        except (BudgetError,):
            raise
        except Exception:
            return None

    # -- configs --
    def config_names(self) -> list[str]:
        size, _ = self.viewer("size")
        names = []
        if size:
            names = [c.get("config") for c in (size.get("size") or {}).get("configs") or [] if c.get("config")]
        if not names:
            info, _ = self.viewer("info")
            names = list(((info or {}).get("dataset_info") or {}).keys())
        for c in card_configs(self.card):
            if c not in names:
                names.append(c)
        return names

    def splits(self, config: str) -> list[dict]:
        size, _ = self.viewer("size")
        return [s for s in ((size or {}).get("size") or {}).get("splits") or [] if s.get("config") == config]


def _viewer_col_stats(stats_json: dict) -> dict:
    out = {}
    for c in (stats_json or {}).get("statistics") or []:
        cs, typ, s = c.get("column_statistics") or {}, c.get("column_type"), {}
        if typ in ("string_label", "class_label"):
            s["n_unique"] = cs.get("n_unique")
            if isinstance(cs.get("frequencies"), dict) and len(cs["frequencies"]) <= 50:
                s["values"] = cs["frequencies"]
        elif typ == "list":
            s.update(len_min=cs.get("min"), len_max=cs.get("max"), len_mean=cs.get("mean"), len_median=cs.get("median"),
                     len_hist=cs.get("histogram"))
        elif typ == "datetime":
            for k in ("min", "max"):
                try:
                    s[k] = dt.datetime.fromisoformat(str(cs.get(k)).replace("Z", ""))
                except Exception:
                    s[k] = None
        elif typ in ("int", "float"):
            s.update(min=cs.get("min"), max=cs.get("max"))
        if cs.get("nan_count") is not None:
            s["nan_count"] = cs["nan_count"]
        out[c.get("column_name")] = s
    return out


def _merge_viewer_stats(table: dict, per_split: list[tuple[int, dict]]):
    """Add viewer statistics (n_unique, list lengths, datetime range) to a table's stats; footers keep priority."""
    if not per_split:
        return
    per_split = sorted(per_split, key=lambda x: -x[0])
    multi = len(per_split) > 1
    for name in {n for _, st in per_split for n in st}:
        tgt = table["stats"].setdefault(name, {})
        parts = [(n, st[name]) for n, st in per_split if name in st]
        nu = [p.get("n_unique") for _, p in parts if p.get("n_unique") is not None]
        if nu and tgt.get("n_unique") is None:
            tgt["n_unique"] = max(nu)
            if multi:
                table["n_unique_merged"] = True
        if parts[0][1].get("values") and not multi and not tgt.get("values"):
            tgt["values"] = parts[0][1]["values"]
        lm = [(n, p) for n, p in parts if p.get("len_mean") is not None]
        if lm:
            tot = sum(n for n, _ in lm) or 1
            tgt.update(len_min=min(p["len_min"] for _, p in lm), len_max=max(p["len_max"] for _, p in lm),
                       len_mean=sum(n * p["len_mean"] for n, p in lm) / tot, len_median=lm[0][1].get("len_median"),
                       len_hist=lm[0][1].get("len_hist"))
            if multi:
                table["viewer_partial"] = True  # the median is the largest split's
        for k, pick in (("min", min), ("max", max)):
            vals = [p.get(k) for _, p in parts if p.get(k) is not None]
            if vals and tgt.get(k) is None:
                try:
                    tgt[k] = pick(vals)
                except TypeError:
                    pass
        nc = [p.get("nan_count") for _, p in parts if p.get("nan_count") is not None]
        if nc and tgt.get("nulls") is None and len(nc) == len(parts):
            tgt["nulls_viewer"] = sum(nc)


def build_tables(st: RepoState, scope: Scope, data_files: list[dict]) -> tuple[list[dict], dict]:
    """Tables for a scope from footers (own parquet, else converted parquet), else viewer info, else dataset_info.json.

    -> (tables, facts) with facts = {level, footer_source, n_footers, n_parquet, viewer_partial, notes: [..]}.
    """
    opts, notes = st.opts, []
    facts = {"footer_source": "", "n_footers": 0, "n_parquet": 0, "viewer_partial": False, "notes": notes,
             "file_rows": {}, "converted_files": []}
    cards = card_configs(st.card)
    names = st.config_names() if not st.access_denied else []
    configs = names if scope.configs is None else list(scope.configs)
    own = sorted((f for f in data_files if f["ext"] == "parquet"), key=lambda f: f["path"])
    cand = []  # (config, split, url, size, path, rev)
    if own:
        facts["footer_source"] = "repo parquet files"
        pats = {c: [glob_regex(x) for x in cards.get(c, [])] for c in configs}
        for f in own:
            cfg = next((c for c in configs if any(rx.match(f["path"]) for rx in pats[c])), None)
            if cfg is None:
                cfg = next((c for c in configs if any(norm(s) == norm(c) for s in f["path"].split("/")[:-1])), None)
            if cfg is None and len(configs) == 1:
                cfg = configs[0]
            cand.append((cfg or "", "", resolve_url(st.repo, f["path"]), f["size"], f["path"], "main"))
    elif configs and not st.access_denied:
        pq_json, err = st.viewer("parquet")
        if pq_json:
            facts["footer_source"] = "viewer converted parquet (refs/convert/parquet)"
            if pq_json.get("partial"):
                facts["viewer_partial"] = True
                notes.append("the viewer converted only part of the dataset to parquet")
            for p in pq_json.get("parquet_files") or []:
                if p.get("config") in configs and p.get("url"):
                    path = f"{p.get('config')}/{p.get('split')}/{p.get('filename')}"
                    cand.append((p["config"], p.get("split", ""), p["url"], p.get("size"), path, "refs/convert/parquet"))
            facts["converted_files"] = [c for c in cand]
        elif err:
            notes.append(f"viewer /parquet: {err}")
    facts["n_parquet"] = len(cand)
    chosen = pick_evenly(cand, opts.max_footers)
    groups: dict[tuple, dict] = {}
    read_bytes = 0
    for cfg, split, url, size, path, rev in chosen:
        ft, err = st.footer(url, size)
        if ft is None:
            st.footer_errors.append(f"{path}: {err}")
            continue
        facts["n_footers"] += 1
        read_bytes += size or 0
        cols = D.cols_from_leaves(ft["leaves"])
        if isinstance(ft.get("features"), dict):  # fixed outer length of a 2-level list = the number of variates
            widths = {c.name: c.width for c in D.cols_from_features(ft["features"])}
            for c in cols:
                c.width = widths.get(c.name)
        tc = next((c.name for c in cols if c.kind in ("time", "list_time")), None)
        facts["file_rows"][(rev, path)] = {"rows": ft["rows"], "min": (ft["stats"].get(tc) or {}).get("min") if tc else None,
                                            "max": (ft["stats"].get(tc) or {}).get("max") if tc else None,
                                            "row_groups": ft["row_groups"], "columns": len(cols)}
        key = (cfg, tuple(f"{c.name}:{c.kind}" for c in cols))
        g = groups.get(key)
        if g is None:
            g = groups[key] = {"name": cfg or path.rsplit("/", 1)[0] or path, "config": cfg, "cols": cols, "rows": 0,
                               "bytes": 0, "n_files": 0, "stats": {}, "stats_src": "parquet footer", "paths": []}
        g["rows"] += ft["rows"]
        g["bytes"] += size or 0
        g["n_files"] += 1
        g["paths"].append(path)
        _merge_stats(g["stats"], ft["stats"])
    tables = list(groups.values())
    if tables:
        all_bytes = sum((c[3] or 0) for c in cand)
        sampled = facts["n_footers"] < len(cand)
        scale = (all_bytes / read_bytes) if sampled and read_bytes else 1.0
        if sampled:
            notes.append(f"footers read for {facts['n_footers']} of {len(cand)} parquet files; counts scaled by bytes "
                         f"(x{scale:.3g})")
        for t in tables:
            est_rows = None
            if facts["viewer_partial"] and t["config"]:
                sp = st.splits(t["config"])
                er = [s.get("estimated_num_rows") or s.get("num_rows") for s in sp]
                est_rows = sum(er) if er and all(e is not None for e in er) else None
            t["complete"] = not sampled and not facts["viewer_partial"]
            t["scale"] = scale
            if est_rows and not sampled and t["rows"]:
                t["scale"] = max(1.0, est_rows / t["rows"])
            t["rows"] = int(round(t["rows"] * t["scale"]))
            t["rows_conf"] = D.EXACT if t["complete"] else D.EST
            t["rows_src"] = "parquet footers" + ("" if t["complete"] else " (scaled)")
    elif configs and not st.access_denied:  # no footers: viewer features and row counts
        info, err = st.viewer("info")
        di = (info or {}).get("dataset_info") or {}
        for c in configs:
            ci = di.get(c)
            if not isinstance(ci, dict) or not ci.get("features"):
                continue
            sp = st.splits(c)
            rows = [s.get("num_rows") for s in sp] or [v.get("num_examples") for v in (ci.get("splits") or {}).values()]
            partial = bool((info or {}).get("partial")) or any(s.get("estimated_num_rows") for s in sp)
            est = [s.get("estimated_num_rows") or s.get("num_rows") for s in sp]
            tables.append({"name": c, "config": c, "cols": D.cols_from_features(ci["features"]),
                           "rows": (sum(est) if partial and est and None not in est else sum(rows))
                           if rows and None not in rows else None,
                           "rows_conf": D.EST if partial else D.EXACT, "rows_src": "viewer /size",
                           "bytes": sum(s.get("num_bytes_parquet_files") or 0 for s in sp), "n_files": 0, "stats": {},
                           "stats_src": "viewer statistics", "complete": not partial, "paths": []})
        if err and not tables:
            notes.append(f"viewer /info: {err}")
    if not tables:  # per-folder dataset_info.json (HF save_to_disk folders)
        for f in scope.files:
            if f["path"].rsplit("/", 1)[-1] == "dataset_info.json" and (f["size"] or 0) < 1_000_000 and len(tables) < 20:
                j = st.text_json(f["path"], f["size"])
                if isinstance(j, dict) and j.get("features"):
                    rows = [v.get("num_examples") for v in (j.get("splits") or {}).values() if isinstance(v, dict)]
                    tables.append({"name": f["path"].rsplit("/", 1)[0], "config": "", "paths": [],
                                   "cols": D.cols_from_features(j["features"]),
                                   "rows": sum(rows) if rows and None not in rows else None, "rows_conf": D.EXACT,
                                   "rows_src": "dataset_info.json", "bytes": 0, "n_files": 0, "stats": {},
                                   "stats_src": "dataset_info.json", "complete": True})
    # viewer statistics: n_unique, list lengths
    n_calls = 0
    if not st.access_denied:
        for t in tables:
            if not t.get("config") or sum(1 for x in tables if x.get("config") == t["config"]) > 1:
                continue
            per_split = []
            for s in st.splits(t["config"]):
                if n_calls >= opts.max_stats_calls:
                    notes.append("viewer statistics not read for every config (call cap)")
                    break
                n_calls += 1
                sj, _err = st.viewer("statistics", config=t["config"], split=s.get("split"))
                if sj:
                    if sj.get("partial"):
                        t["viewer_partial"] = True
                    per_split.append((sj.get("num_examples") or 0, _viewer_col_stats(sj)))
            _merge_viewer_stats(t, per_split)
    # hive partitions that look like a series id: `key=value` path segments
    part: dict[str, set] = {}
    for f in data_files:
        for seg in f["path"].split("/")[:-1]:
            if "=" in seg:
                k, v = seg.split("=", 1)
                part.setdefault(k, set()).add(v)
    for k, vals in part.items():
        if D.is_id_name(k) and len(vals) > 1:
            for t in tables:
                if facts["footer_source"].startswith("repo") and not any(c.name == k for c in t["cols"]):
                    t["cols"] = t["cols"] + [D.Col(k, "str", "path partition")]
                    t["partition_series"], t["partition_key"] = len(vals), k
            break
    return tables, facts


def _first_number_range(text: str):
    nums = []
    for m in re.finditer(r"(\d[\d,]*\.?\d*(?:e\d+)?)\s*([kKmMbB](?![a-zA-Z]))?", text or ""):
        try:
            v = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        v *= {"k": 1e3, "m": 1e6, "b": 1e9}.get((m.group(2) or "").lower(), 1.0)
        nums.append(v)
    return (min(nums), max(nums)) if nums else None


def _join(names, cap=40) -> str:
    names = [str(n) for n in names]
    return "; ".join(names[:cap]) + (f"; ... (+{len(names) - cap} more)" if len(names) > cap else "")


def _iso(v) -> str:
    return v.strftime("%Y-%m-%d %H:%M:%S") if isinstance(v, dt.datetime) else ""


def profile_unit(st: RepoState, cat: dict, scope: Scope, row_kind: str, name: str) -> tuple[dict, list[dict]]:
    """One output row (a catalog row, or one config of it) and its file rows."""
    info = st.info or {}
    row = {"catalog_row": cat["_row"], "row_kind": row_kind, "repo_id": st.repo,
           "config_or_subset": cat.get("config_or_subset", ""), "profile_name": name or scope.name,
           "domain": cat.get("domain", ""), "status": cat.get("status", ""),
           "catalog_frequency": cat.get("frequency", ""), "catalog_approx_series": cat.get("approx_series", ""),
           "catalog_approx_points": cat.get("approx_points", ""), "catalog_format": cat.get("format", ""),
           "catalog_licence": cat.get("licence", ""), "catalog_gated": cat.get("gated", "")}
    problems, undet = [], []
    gated = info.get("gated") or False
    all_files = st.files
    ext_n, ext_b = {}, {}
    for f in all_files:
        ext_n[f["ext"]] = ext_n.get(f["ext"], 0) + 1
        ext_b[f["ext"]] = ext_b.get(f["ext"], 0) + (f["size"] or 0)
    lic = (st.card.get("license") if isinstance(st.card, dict) else None) or next(
        (t.split(":", 1)[1] for t in info.get("tags") or [] if str(t).startswith("license:")), "")
    cards = card_configs(st.card)
    names = st.config_names()
    row.update(hub_exists="yes", hub_id=info.get("id", ""), gated=str(gated).lower() if gated else "no",
               private="yes" if info.get("private") else "no", disabled="yes" if info.get("disabled") else "no",
               licence_hub=lic if isinstance(lic, str) else _join(lic), last_modified=info.get("lastModified", ""),
               repo_bytes=sum(f["size"] or 0 for f in all_files), repo_files=len(all_files),
               repo_ext_summary="; ".join(f"{e}: {ext_n[e]} files, {ext_b[e]} B" for e in sorted(ext_n, key=lambda e: -ext_b[e])),
               repo_n_configs=len(names), repo_configs=_join(names, 30),
               card_splits=_join(sorted({str(x.get("split")) for pats in ((c.get("data_files") or []) for c in
                                         (st.card.get("configs") or []) if isinstance(c, dict))
                                         if isinstance(pats, list) for x in pats if isinstance(x, dict) and x.get("split")})))
    data = [f for f in scope.files if f["data"]]
    sext_b = {}
    for f in data:
        sext_b[f["ext"]] = sext_b.get(f["ext"], 0) + (f["size"] or 0)
    bytes_all = sum(f["size"] or 0 for f in data)
    row.update(scope_rule=scope.rule, scope_unmatched=_join(scope.unmatched), scope_files=len(scope.files),
               scope_data_files=len(data), scope_bytes=bytes_all,
               scope_formats="; ".join(f"{e}: {b} B" for e, b in sorted(sext_b.items(), key=lambda x: -x[1])))
    unmapped = False
    if not data and scope.configs and not st.access_denied:  # a viewer config whose files could not be mapped by path
        vb = sum(s.get("num_bytes_parquet_files") or 0 for c in scope.configs for s in st.splits(c))
        if vb:
            bytes_all, unmapped = vb, True
            row.update(scope_bytes=vb, scope_formats=f"parquet (viewer copy): {vb} B")
            undet.append("bytes: the config's files could not be mapped by path; bytes are the viewer's parquet copy")
    valid, verr = st.viewer("is-valid")
    row["viewer_valid"] = "yes" if (valid or {}).get("viewer") else ("no" if valid else f"no ({verr})")
    tables, facts = build_tables(st, scope, data)
    if st.access_denied and gated and not tables:
        row["profile_level"] = "gated"
        row["error"] = "gated, no access"
    elif facts["n_footers"] and any(t.get("stats_src") == "parquet footer" for t in tables):
        row["profile_level"] = "viewer+footers" if (valid or {}).get("viewer") else "footers"
    elif tables:
        row["profile_level"] = "viewer" if tables[0].get("rows_src") == "viewer /size" else "files_only+dataset_info"
    else:
        row["profile_level"] = "files_only"
    row.update(n_footers_read=facts["n_footers"], n_parquet_files=facts["n_parquet"], footer_source=facts["footer_source"],
               viewer_partial="yes" if facts["viewer_partial"] else "no", n_schemas=len(tables))
    # folder metadata json (RMISC: meta.json with freq, num_series, license, source, targets, covariates)
    meta = None
    metas = [f for f in scope.files if re.fullmatch(r"(?i)(meta|metadata)\.json", f["path"].rsplit("/", 1)[-1])]
    if len(metas) == 1 and (metas[0]["size"] or 0) < 1_000_000 and not st.access_denied:
        meta = st.text_json(metas[0]["path"], metas[0]["size"])
        meta = meta if isinstance(meta, dict) else None
    if meta:
        row.update(meta_freq=meta.get("freq") or "", meta_num_series=meta.get("num_series", ""),
                   meta_num_points=meta.get("num_datapoints", ""), meta_licence=meta.get("license") or "",
                   meta_source=meta.get("source") or "", meta_domain=meta.get("domain") or "")
    # per-table derivation
    per = []
    for t in tables:
        roles = D.assign_roles(t["cols"], t["stats"])
        if meta and isinstance(meta.get("targets"), list):
            have = {c.name for c in t["cols"]}
            tg = [x for x in meta["targets"] if x in have]
            if tg:
                cv = [x for x in (meta.get("covariates") or []) if x in have]
                roles.update(target_cols=tg, target_rule="targets listed in the folder's meta.json",
                             covariate_cols=cv, value_cols=[c for c in roles["value_cols"] if c in tg or c in cv] or tg + cv)
                if isinstance(meta.get("timestamp"), str) and meta["timestamp"] in have:
                    roles.update(time_col=meta["timestamp"], time_rule="timestamp named in the folder's meta.json")
        layout, rule, lconf = D.decide_layout(t["cols"], roles)
        per.append((t, roles, layout, rule, lconf, D.table_counts(t, roles, layout)))
    if per:
        per.sort(key=lambda x: -(x[5]["n_points"] or 0) - 1e-9 * (x[0].get("bytes") or 0))
        t0, r0, layout, rule, lconf, c0 = per[0]
        lay_mix = {}
        for p in per:
            lay_mix[p[2]] = lay_mix.get(p[2], 0) + 1
        row.update(layout=layout, layout_rule=rule, layout_conf=lconf,
                   layouts_all="; ".join(f"{k}: {v}" for k, v in sorted(lay_mix.items())),
                   rows=sum(p[0]["rows"] or 0 for p in per) if all(p[0]["rows"] is not None for p in per) else "",
                   rows_conf=D._worse(*[p[0].get("rows_conf", D.UNK) for p in per]), rows_src=t0.get("rows_src", ""),
                   n_columns=len(t0["cols"]), columns=_join([c.label for c in t0["cols"]], 60))

        def total(key, conf_key):
            vals = [(p[5][key], p[5][conf_key]) for p in per]
            known = [v for v, _ in vals if v is not None]
            if not known:
                return None, D.UNK
            conf = D._worse(*[c for v, c in vals if v is not None])
            return sum(known), (D.EST if len(known) < len(vals) else conf)

        ns, nsc = total("n_series", "n_series_conf")
        npnt, npc = total("n_points", "n_points_conf")
        npt, nptc = total("n_points_target", "n_points_target_conf")
        ns_src = c0["n_series_src"]
        if meta and isinstance(meta.get("num_series"), (int, float)) and meta["num_series"] > 0:
            ns, nsc, ns_src = int(meta["num_series"]), D.EXACT, "stated in the folder's meta.json"
        if len(per) > 1 and any(p[5]["n_points"] is None for p in per):
            undet.append("n_points: some tables in the scope have no row count")
        row.update(n_series=ns, n_series_conf=nsc, n_series_src=ns_src + (f" (sum over {len(per)} tables)" if len(per) > 1 and "meta.json" not in ns_src else ""),
                   n_points=npnt, n_points_conf=npc, n_points_src=c0["n_points_src"],
                   n_points_target=npt, n_points_target_conf=nptc)
        ln = dict(len_min=c0["len_min"], len_median=c0["len_median"], len_max=c0["len_max"], len_conf=c0["len_conf"],
                  len_src=c0["len_src"])
        if meta and ns and layout != "nested" and ln["len_median"] is None and t0["rows"]:
            ln.update(len_median=t0["rows"] / ns, len_conf=D.EST, len_src="rows / n_series (a mean)")
        if len(per) > 1:
            ln["len_conf"] = D.EST if ln["len_median"] is not None else D.UNK
            ln["len_src"] += " (largest table only)"
        row.update({k: (round(v, 2) if isinstance(v, float) else v) for k, v in ln.items()})
        dr = D.date_range([p[0] for p in per], [p[1] for p in per])
        if dr["first"] is None and layout == "nested":
            undet.append("date range: only series start times are in the metadata (no end time)")
        row.update(date_first=_iso(dr["first"]), date_last=_iso(dr["last"]),
                   span_years=round(dr["span_years"], 3) if dr["span_years"] is not None else "", date_conf=dr["conf"],
                   date_src=dr["src"])
        # frequency
        fsec, fconf, fsrc, mixed = None, D.UNK, "", False
        if meta and meta.get("freq"):
            fsec = D.parse_freq_alias(str(meta["freq"])) or D.freq_from_name(str(meta["freq"]))
            if fsec:
                fconf, fsrc = D.EXACT, f"stated in the folder's meta.json ({meta['freq']})"
        if not fsec:
            seen = {}
            for p in per:
                for fc in p[1]["freq_cols"]:
                    s = p[0]["stats"].get(fc) or {}
                    vals = list((s.get("values") or {}).keys())
                    if not vals and isinstance(s.get("min"), str) and s.get("min") == s.get("max"):
                        vals = [s["min"]]
                    elif not vals and isinstance(s.get("min"), str):
                        vals = [s["min"], s["max"]]
                        seen["?"] = None
                    for v in vals:
                        seen[v] = D.parse_freq_alias(v)
            secs = sorted({v for v in seen.values() if v})
            if len(secs) == 1 and None not in seen.values():
                fsec, fconf, fsrc = secs[0], D.EXACT, f"freq column ({', '.join(k for k in seen if k != '?')})"
            elif len(seen) > 1 or (secs and None in seen.values()):
                mixed, fconf = True, D.EXACT
                fsrc = "freq column holds several values (" + ", ".join(str(k) for k in list(seen)[:6]) + ")"
        if not fsec and not mixed:
            cand_names = [n for n in ([name] if name else []) + (scope.configs or []) + [scope.name] if n]
            got = sorted({s for s in (D.freq_from_name(n) for n in cand_names) if s})
            if len(got) == 1:
                fsec, fconf, fsrc = got[0], D.EXACT, f"stated in the config / folder name ({cand_names[0]})"
            elif not got:
                per_file = [D.freq_from_name(f["path"].rsplit(".", 1)[0]) for f in data]
                vals = sorted({s for s in per_file if s})
                if len(vals) == 1 and all(per_file):  # a file name is weaker than a config name: estimated
                    fsec, fconf, fsrc = vals[0], D.EST, "read from every data file path (not a config name)"
                elif len(vals) > 1 and all(per_file):
                    mixed, fconf = True, D.EST
                    fsrc = "file paths state several frequencies (" + ", ".join(D.freq_label(v) for v in vals[:6]) + ")"
            else:
                mixed, fconf = True, D.EXACT
                fsrc = "config names state several frequencies (" + ", ".join(D.freq_label(v) for v in got) + ")"
        if not fsec and not mixed and layout in ("wide", "nested", "long") and dr["span_years"] and ln["len_median"]:
            sec, why = D.freq_implied(dr["span_years"] * 365.25 * 86400.0, float(ln["len_median"]))
            if sec:
                fsec, fconf, fsrc = sec, D.EST, f"implied by span and series length ({why})"
        row.update(frequency="mixed" if mixed else (D.freq_label(fsec) if fsec else ""), freq_seconds=fsec or "",
                   freq_conf=fconf, freq_src=fsrc, freq_band=D.freq_band(fsec, layout, mixed),
                   len_band=D.length_band(ln["len_median"]))
        fut, past = D.covariate_timing(r0["covariate_cols"])
        row.update(time_col=r0["time_col"] or "", time_col_rule=r0["time_rule"], id_cols=_join(r0["id_cols"]),
                   value_cols=_join(r0["value_cols"]), n_value_cols=len(r0["value_cols"]),
                   target_cols=_join(r0["target_cols"]), target_rule=r0["target_rule"],
                   covariate_cols=_join(r0["covariate_cols"]), n_covariate_cols=len(r0["covariate_cols"]),
                   static_cols=_join(r0["static_cols"]), n_static_cols=len(r0["static_cols"]),
                   cov_known_future=_join(fut), cov_past_only=_join(past),
                   has_covariates="yes" if r0["covariate_cols"] else "no")
        # missing share: nulls / values over the value columns
        nulls = vals = 0
        mconf, msrc = D.UNK, ""
        for t, roles, *_ in per:
            for n in roles["value_cols"]:
                s = t["stats"].get(n) or {}
                if s.get("nulls") is not None and s.get("num_values"):
                    nulls += s["nulls"]
                    vals += s["num_values"]
                    mconf = D._worse(mconf if mconf != D.UNK else D.EXACT, D.EXACT if t.get("complete", True) else D.EST)
                    msrc = "parquet footer null counts (NaN stored as a float value is not counted)"
                elif s.get("nulls_viewer") is not None and t.get("rows"):
                    nulls += s["nulls_viewer"]
                    vals += t["rows"]
                    mconf, msrc = D.EST, "viewer nan_count"
        row.update(missing_share=round(nulls / vals, 6) if vals else "", missing_conf=mconf if vals else D.UNK,
                   missing_src=msrc)
        non_pq = [f for f in data if f["ext"] != "parquet"]
        if facts["footer_source"].startswith("repo") and non_pq:
            undet.append(f"{len(non_pq)} non-parquet data files in the scope ({D.fmt_bytes(sum(f['size'] or 0 for f in non_pq))}) "
                         f"are not in the row and point counts")
        if layout == "wide" and t0.get("n_files", 0) > 1:
            undet.append(f"n_series: {t0['n_files']} files share one schema; if each file is a different entity the "
                         f"series count is files x columns, not columns")
        if layout == "long" and ns is None:
            undet.append("n_series: no distinct count for the id column(s) in the metadata")
        if lconf == D.EST:
            undet.append(f"layout is a rule-based guess ({rule.split(' ')[0]})")
        channels = c0.get("channels", 1)
    else:
        layout, ns, npnt, channels = "unknown", None, None, 1
        row.update(layout="unknown", layout_rule="R0 no schema: no viewer answer and no parquet footer", layout_conf=D.UNK,
                   n_series_conf=D.UNK, n_points_conf=D.UNK, n_points_target_conf=D.UNK, len_conf=D.UNK,
                   date_conf=D.UNK, freq_conf=D.UNK, missing_conf=D.UNK, rows_conf=D.UNK, has_covariates="unknown",
                   freq_band=D.freq_band(None), len_band="unknown")
        per_file = [D.freq_from_name(f["path"].rsplit(".", 1)[0]) for f in data]
        vals = sorted({s for s in per_file if s})
        nm = D.freq_from_name(name or scope.name) if (name or scope.name) else None
        if nm or (len(vals) == 1 and all(per_file)):
            sec = nm or vals[0]
            row.update(frequency=D.freq_label(sec), freq_seconds=sec, freq_conf=D.EXACT if nm else D.EST,
                       freq_band=D.freq_band(sec), freq_src="stated in the config / folder name" if nm else
                       "read from every data file path (not a config name)")
        undet.append("no schema: rows, series, points, lengths, dates, columns and missing share are all unknown")
    partial_fetch = (bool(data) and all(f["ext"] == "parquet" for f in data)) or unmapped
    if unmapped:
        sext_b = {"parquet": bytes_all}
    sl = D.slice_estimates(ns, npnt, bytes_all, channels, partial_fetch)
    row.update(bytes_all=bytes_all, **sl, slice_conf=D.EST if npnt else D.UNK,
               slice_note="bytes proportional to points; real fetch rounds up to whole row groups / files"
               if partial_fetch else "files cannot be cut: every slice costs all bytes")
    script = any(f["ext"] == "py" and "/" not in f["path"] for f in all_files)
    diff, why = D.ingest_difficulty(sext_b, len(tables), layout if tables else "unknown", script and not partial_fetch)
    row.update(ingest_difficulty=diff, ingest_reason=why)
    # problems: Hub contents against the catalog row
    if info.get("id") and info["id"] != st.repo:
        problems.append(f"renamed: the Hub answers as {info['id']}")
    if info.get("disabled"):
        problems.append("repo is disabled on the Hub")
    if not data and not unmapped:
        problems.append("empty: no data files in the scope")
    if scope.unmatched:
        problems.append("subset text not matched on the Hub: " + _join(scope.unmatched, 8))
    cg = (cat.get("gated") or "").lower()
    if cg and (cg != "no") != bool(gated):
        problems.append(f"gated: catalog says {cg}, Hub says {str(gated).lower() if gated else 'no'}")
    if gated and st.access_denied:
        problems.append("gated, no access")
    cf = (cat.get("format") or "").lower()
    if cf.startswith("parquet") and data and "parquet" not in sext_b:
        problems.append("format: catalog says parquet, the scope has no parquet file (" + ", ".join(sorted(sext_b)) + ")")
    if cf.startswith(("csv", "jsonl", "zip", "hdf5", "npy")) and data and cf.split()[0].split(".")[0] not in {
            e.split(".")[0] for e in sext_b} and not {"h5", "hdf5"} & set(sext_b):
        problems.append(f"format: catalog says {cf.split()[0]}, the scope holds " + ", ".join(sorted(sext_b)))
    cl, hl = norm(cat.get("licence", "")), norm(row["licence_hub"])
    if hl and cl and cl not in ("", "other", "notstated", "unknown") and not cl.startswith(hl) and not hl.startswith(cl) \
            and hl not in cl:
        problems.append(f"licence: catalog says {cat.get('licence')}, the card says {row['licence_hub']}")
    try:
        gb = float(cat.get("repo_size_gb") or "")
        if gb > 0 and row["repo_bytes"] and not (0.75 <= row["repo_bytes"] / (gb * 1e9) <= 1.33) \
                and not (0.75 <= row["repo_bytes"] / (gb * 2 ** 30) <= 1.33):
            problems.append(f"size: catalog says {gb} GB, the Hub file listing sums to {D.fmt_bytes(row['repo_bytes'])}")
    except ValueError:
        pass
    rng = _first_number_range(cat.get("approx_points", ""))
    if rng and npnt and not (rng[0] / 10.0 <= npnt <= rng[1] * 10.0) and row.get("rows") not in ("", None) \
            and not (rng[0] / 10.0 <= row["rows"] <= rng[1] * 10.0):
        problems.append(f"points: catalog says {cat.get('approx_points')}, metadata gives {D.fmt_count(npnt)} points "
                        f"({D.fmt_count(row['rows'])} rows)")
    cfreq = D.parse_freq_alias(cat.get("frequency", "")) or D.freq_from_name(cat.get("frequency", ""))
    if cfreq and row.get("freq_seconds") and D.freq_band(cfreq) != D.freq_band(row["freq_seconds"]):
        problems.append(f"frequency: catalog says {cat.get('frequency')}, metadata says {row['frequency']} ({row['freq_src']})")
    if meta and isinstance(meta.get("num_datapoints"), (int, float)) and npnt and meta["num_datapoints"] > 0 \
            and not (0.9 <= npnt / meta["num_datapoints"] <= 1.1):
        problems.append(f"points: the folder's meta.json states {meta['num_datapoints']}, the footers give {npnt}")
    di = st.card.get("dataset_info") if isinstance(st.card, dict) else None
    if di and tables and row.get("rows") not in ("", None) and facts["footer_source"]:
        dis = di if isinstance(di, list) else [di]
        want = set(scope.configs or []) if scope.configs else None
        claim = 0
        for d in dis:
            if isinstance(d, dict) and (want is None or d.get("config_name", "default") in want):
                claim += sum(int(s.get("num_examples") or 0) for s in d.get("splits") or [] if isinstance(s, dict))
        if claim and row["rows_conf"] == D.EXACT and abs(claim - row["rows"]) > 0.01 * max(claim, row["rows"]):
            problems.append(f"rows: the card claims {claim} rows, the parquet footers hold {row['rows']}")
    if valid is not None and not valid.get("viewer") and not st.access_denied:
        problems.append("the dataset viewer does not work for this repo")
    if st.footer_errors:
        problems.append(f"{len(st.footer_errors)} footer(s) not readable, first: {st.footer_errors[0][:160]}")
    for key, label in (("n_series_conf", "n_series"), ("n_points_conf", "n_points"), ("len_conf", "series length"),
                       ("date_conf", "date range"), ("freq_conf", "frequency"), ("missing_conf", "missing share")):
        c = row.get(key, D.UNK)
        if c != D.EXACT and tables:
            undet.append(f"{label}: {c}")
    row.update(problems=" | ".join(problems), undetermined=" | ".join(list(dict.fromkeys(undet + facts["notes"]))))
    row.setdefault("error", "")
    # file rows
    frows = []
    for f in sorted(data, key=lambda f: f["path"]):
        fr = facts["file_rows"].get(("main", f["path"]), {})
        fq = D.freq_from_name(f["path"].rsplit(".", 1)[0])
        frows.append({"catalog_row": cat["_row"], "row_kind": row_kind, "repo_id": st.repo, "profile_name": row["profile_name"],
                      "revision": "main", "path": f["path"], "bytes": f["size"], "format": f["ext"],
                      "rows": fr.get("rows", ""), "date_first": _iso(D.to_datetime(fr.get("min"))[0]) if fr else "",
                      "date_last": _iso(D.to_datetime(fr.get("max"))[0]) if fr else "",
                      "row_groups": fr.get("row_groups", ""), "columns": fr.get("columns", ""),
                      "freq_in_name": D.freq_label(fq) if fq else "", "footer_read": "yes" if fr else "no"})
    for cfg, split, _url, size, path, rev in facts["converted_files"]:
        fr = facts["file_rows"].get((rev, path), {})
        frows.append({"catalog_row": cat["_row"], "row_kind": row_kind, "repo_id": st.repo, "profile_name": row["profile_name"],
                      "revision": rev, "path": path, "bytes": size, "format": "parquet (viewer copy)",
                      "rows": fr.get("rows", ""), "date_first": _iso(D.to_datetime(fr.get("min"))[0]) if fr else "",
                      "date_last": _iso(D.to_datetime(fr.get("max"))[0]) if fr else "",
                      "row_groups": fr.get("row_groups", ""), "columns": fr.get("columns", ""), "freq_in_name": "",
                      "footer_read": "yes" if fr else "no"})
    return row, frows


def _stub_row(cat: dict, repo: str, level: str, error: str, problems: str = "") -> dict:
    return {"catalog_row": cat["_row"], "row_kind": "catalog", "repo_id": repo,
            "config_or_subset": cat.get("config_or_subset", ""), "profile_name": "", "domain": cat.get("domain", ""),
            "status": cat.get("status", ""), "catalog_frequency": cat.get("frequency", ""),
            "catalog_approx_series": cat.get("approx_series", ""), "catalog_approx_points": cat.get("approx_points", ""),
            "catalog_format": cat.get("format", ""), "catalog_licence": cat.get("licence", ""),
            "catalog_gated": cat.get("gated", ""), "hub_exists": "no" if level == "missing" else "",
            "profile_level": level, "error": error, "problems": problems or error, "layout": "unknown",
            "layout_conf": D.UNK, "n_series_conf": D.UNK, "n_points_conf": D.UNK, "len_conf": D.UNK, "date_conf": D.UNK,
            "freq_conf": D.UNK, "missing_conf": D.UNK, "freq_band": "unknown", "len_band": "unknown",
            "has_covariates": "unknown", "ingest_difficulty": "", "undetermined": "everything: " + error}


def profile_repo(fx: Fetcher, repo: str, cat_rows: list[dict], excluded: set[str], opts) -> tuple[list[dict], list[dict]]:
    st = RepoState(fx, repo, opts)
    st.load_info()
    if not st.exists:
        level = "missing" if st.info_error.startswith(("HTTP 401", "HTTP 404")) else "error"
        note = " (missing, renamed or private; without a token the Hub answers 401 for all three)" if level == "missing" else ""
        return [_stub_row(c, repo, level, st.info_error + note) for c in cat_rows], []
    rows, frows = [], []
    names = st.config_names()
    cards = card_configs(st.card)
    named = {norm(subset_name(c.get("config_or_subset", ""))) for c in cat_rows}
    for cat in cat_rows:
        try:
            scope = resolve_scope(cat.get("config_or_subset", ""), st.files, names, cards)
            r, fr = profile_unit(st, cat, scope, "catalog", "")
            rows.append(r)
            frows += fr
            expand = [c for c in names if norm(c) not in excluded and norm(c) not in named] if scope.whole else []
            if scope.whole and len(names) <= 1 and repo.lower() in COLLECTIONS:
                tops = sorted({f["path"].split("/")[0] for f in st.files if f["data"] and "/" in f["path"]})
                expand = [t for t in tops if norm(t) not in excluded and norm(t) not in named] if len(tops) > 1 else []
            if len(expand) > 1 and len(expand) <= opts.max_expand:
                for c in expand:
                    m = _match_one(c, st.files, names, cards)
                    sc = Scope(m[1] if m else [], [m[2]] if m and m[2] else [], (m[0] if m else f"config:{c}") + " (expanded)",
                               name=c)
                    if m is None and c in names:
                        sc.configs = [c]
                    r2, _ = profile_unit(st, cat, sc, "config", c)
                    rows.append(r2)
            elif len(expand) > opts.max_expand:
                r["undetermined"] = (r["undetermined"] + " | " if r["undetermined"] else "") + \
                    f"{len(expand)} configs: per-config rows not written (above --max-expand {opts.max_expand})"
        except (BudgetError, GuardError):
            raise
        except Exception as e:
            rows.append(_stub_row(cat, repo, "error", f"{type(e).__name__}: {e}"))
    return rows, frows


# ---- output ----------------------------------------------------------------------------------------------------------------
PROFILE_COLS = [
    "catalog_row", "row_kind", "repo_id", "config_or_subset", "profile_name", "domain", "status", "profile_level", "error",
    "hub_exists", "hub_id", "gated", "private", "disabled", "licence_hub", "last_modified", "repo_bytes", "repo_files",
    "repo_ext_summary", "repo_n_configs", "repo_configs", "card_splits", "viewer_valid", "viewer_partial",
    "scope_rule", "scope_unmatched", "scope_files", "scope_data_files", "scope_bytes", "scope_formats",
    "footer_source", "n_parquet_files", "n_footers_read", "n_schemas", "rows", "rows_conf", "rows_src", "n_columns",
    "columns", "layout", "layout_rule", "layout_conf", "layouts_all",
    "n_series", "n_series_conf", "n_series_src", "n_points", "n_points_conf", "n_points_src", "n_points_target",
    "n_points_target_conf", "len_min", "len_median", "len_max", "len_conf", "len_src", "len_band",
    "date_first", "date_last", "span_years", "date_conf", "date_src",
    "frequency", "freq_seconds", "freq_conf", "freq_src", "freq_band",
    "time_col", "time_col_rule", "id_cols", "value_cols", "n_value_cols", "target_cols", "target_rule", "covariate_cols",
    "n_covariate_cols", "static_cols", "n_static_cols", "cov_known_future", "cov_past_only", "has_covariates",
    "missing_share", "missing_conf", "missing_src",
    "bytes_all", "slice500_points", "slice500_bytes", "slice5k_points", "slice5k_bytes", "slice_all_points",
    "slice_all_bytes", "slice_conf", "slice_note", "ingest_difficulty", "ingest_reason",
    "meta_freq", "meta_num_series", "meta_num_points", "meta_licence", "meta_source", "meta_domain",
    "catalog_frequency", "catalog_approx_series", "catalog_approx_points", "catalog_format", "catalog_licence",
    "catalog_gated", "problems", "undetermined"]
FILE_COLS = ["catalog_row", "row_kind", "repo_id", "profile_name", "revision", "path", "bytes", "format", "rows",
             "date_first", "date_last", "row_groups", "columns", "freq_in_name", "footer_read"]


def write_csv(path: Path, rows: list[dict], cols: list[str]):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})


def run(opts) -> int:
    catalog = read_catalog(Path(opts.catalog))
    excluded_rows = read_catalog(Path(opts.excluded)) if Path(opts.excluded).exists() else []
    excl_by_repo: dict[str, set] = {}
    for r in excluded_rows:
        excl_by_repo.setdefault(r["repo_id"].lower(), set()).add(norm(subset_name(r.get("config_or_subset", ""))))
    by_repo: dict[str, list[dict]] = {}
    for r in catalog:
        by_repo.setdefault(r["repo_id"], []).append(r)
    repos = list(by_repo)
    if opts.only:
        want = {x.lower() for x in opts.only}
        repos = [r for r in repos if r.lower() in want]
    if opts.limit:
        repos = repos[:opts.limit]
    fx = Fetcher(opts.cache, token=os.environ.get("HF_TOKEN") or None, budget=int(opts.budget_gb * 1024 ** 3),
                 max_inflight=4, refresh=opts.refresh, offline=opts.offline, footer_max=int(opts.max_footer_mb * 2 ** 20))
    print(f"catalog: {len(catalog)} rows, {len(by_repo)} repos; profiling {len(repos)} repos; token "
          f"{'set' if os.environ.get('HF_TOKEN') else 'not set (anonymous)'}; cache {opts.cache}", flush=True)
    results: dict[str, tuple] = {}
    stop = threading.Event()
    aborted = ""

    def work(repo):
        if stop.is_set():
            return repo, None
        try:
            if opts.plan:
                st = RepoState(fx, repo, opts)
                st.load_info()
                return repo, st
            return repo, profile_repo(fx, repo, by_repo[repo], excl_by_repo.get(repo.lower(), set()), opts)
        except (BudgetError, GuardError) as e:
            if isinstance(e, BudgetError):
                stop.set()
                raise
            return repo, ([_stub_row(c, repo, "error", f"GuardError: {e}") for c in by_repo[repo]], [])
        except Exception as e:  # one repo failing never stops the run
            return repo, ([_stub_row(c, repo, "error", f"{type(e).__name__}: {e}") for c in by_repo[repo]], [])

    done = 0
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = [pool.submit(work, r) for r in repos]
            for fut in as_completed(futs):
                try:
                    repo, res = fut.result()
                except BudgetError as e:
                    aborted = str(e)
                    stop.set()
                    continue
                if res is None:
                    continue
                results[repo] = res
                done += 1
                if done % 10 == 0 or done == len(repos):
                    print(f"  {done}/{len(repos)} repos, {D.fmt_bytes(fx.bytes_total)} transferred, "
                          f"{fx.n_requests} requests, {fx.n_cache_hits} cache hits", flush=True)
    finally:
        s = fx.summary()
        print(f"TOTAL BYTES TRANSFERRED: {s['bytes']} ({D.fmt_bytes(s['bytes'])}) in {s['requests']} requests; "
              f"{s['cache_hits']} cache hits; {s['http_429']} rate-limit waits", flush=True)
        for k, v in sorted(s["by_kind"].items()):
            print(f"  {k}: {v['requests']} requests, {D.fmt_bytes(v['bytes'])}", flush=True)
    if aborted:
        print(f"ABORTED: {aborted}. No output files written. Cached answers are kept; a rerun continues from them.")
        return 2
    if opts.plan:
        n_own = n_foot = n_rows_no_pq = 0
        for repo in repos:
            st = results.get(repo)
            if st is None or not st.exists:
                continue
            for cat in by_repo[repo]:
                sc = resolve_scope(cat.get("config_or_subset", ""), st.files, list(card_configs(st.card)),
                                   card_configs(st.card))
                pq = [f for f in sc.files if f["data"] and f["ext"] == "parquet"]
                n_own += len(pq)
                n_foot += min(len(pq), opts.max_footers)
                n_rows_no_pq += 0 if pq else 1
        print(f"PLAN: {sum(1 for r in repos if results.get(r) is not None and results[r].exists)} repos exist; "
              f"{n_own} repo parquet files in the catalog scopes -> {n_foot} footer reads planned (cap "
              f"{opts.max_footers} per row), about {D.fmt_bytes(n_foot * (opts.footer_tail + 700))}; {n_rows_no_pq} rows "
              f"have no parquet of their own (their footers come from the viewer's converted files, count not known "
              f"yet). Viewer: about 4 calls per repo + 1 per config and split.")
        return 0
    rows, frows = [], []
    for repo in repos:
        if repo in results:
            rows += results[repo][0]
            frows += results[repo][1]
    rows.sort(key=lambda r: (r["catalog_row"], 0 if r["row_kind"] == "catalog" else 1, str(r.get("profile_name", ""))))
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "hf_profile.csv", rows, PROFILE_COLS)
    write_csv(out / "hf_profile_files.csv", frows, FILE_COLS)
    from .hf_report import write_report

    write_report(out / "HF_PROFILE.md", rows, {"date": dt.date.today().isoformat(), "catalog": Path(opts.catalog).name,
                                                 "catalog_rows": len(catalog), "catalog_repos": len(by_repo),
                                                 "repos_profiled": len(repos), "fetch": fx.summary(),
                                                 "offline": opts.offline, "token": bool(os.environ.get("HF_TOKEN")),
                                                 "max_footers": opts.max_footers, "n_file_rows": len(frows)})
    print(f"wrote {out / 'hf_profile.csv'} ({len(rows)} rows), {out / 'hf_profile_files.csv'} ({len(frows)} rows), "
          f"{out / 'HF_PROFILE.md'}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Profile the Hugging Face catalog from metadata only (no dataset download).")
    p.add_argument("--catalog", default=str(HERE / "hf_catalog_v2.csv"))
    p.add_argument("--excluded", default=str(HERE / "hf_excluded.csv"))
    p.add_argument("--out", default=str(HERE))
    p.add_argument("--cache", default=str(HERE / ".hf_cache"))
    p.add_argument("--refresh", action="store_true", help="ignore the cache and fetch everything again")
    p.add_argument("--offline", action="store_true", help="no network: rebuild the outputs from the cache")
    p.add_argument("--plan", action="store_true", help="repo listings only, then print the planned footer reads")
    p.add_argument("--only", nargs="*", default=[], help="repo ids to profile (default: all)")
    p.add_argument("--limit", type=int, default=0, help="first N repos only")
    p.add_argument("--max-footers", type=int, default=50, help="footer reads per catalog row (more files: sampled)")
    p.add_argument("--footer-tail", type=int, default=16384, help="bytes of the first footer read")
    p.add_argument("--max-footer-mb", type=float, default=8.0, help="largest footer that is read")
    p.add_argument("--max-expand", type=int, default=300, help="most per-config rows added for a whole-repo row")
    p.add_argument("--max-stats-calls", type=int, default=400, help="viewer /statistics calls per catalog row")
    p.add_argument("--budget-gb", type=float, default=BUDGET / 1024 ** 3, help="abort above this many GB transferred")
    opts = p.parse_args(argv)
    try:
        return run(opts)
    except FetchError as e:
        print(f"fetch error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
