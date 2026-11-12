"""A2 download (JarvisLabs only; never run on the dev PC).

Only the files in each selected row's scope (hf_profile_files.csv), each (repo, revision, path) once even when several
catalog rows share it. Formats the readers cannot use (npy, h5, netcdf, media, pickles, ...) are not downloaded; the plan
lists them. huggingface_hub (Xet transfers; hf_transfer is deprecated and not enabled). Resumable: a file whose
manifest entry has the same size on disk is not fetched again. Inside a repo, file_workers files are fetched in parallel
(hf_hub_download per file, retried on 429 with Retry-After; a Xet / CAS 401, an expired Xet token, is retried
xet_retries times with a fresh call, then the file is fetched over plain HTTPS); download_jobs repos run in parallel. Before each repo: free disk >= repo bytes x disk_margin,
else the repo is skipped and logged. Parallel across repos.

Manifest: $TFB_DATA/manifest/<repo slug>.jsonl, one json line per file {repo, revision, path, bytes, sha256, status,
dataset_ids}; `combine` writes manifest.csv. HF_TOKEN is read by huggingface_hub from the environment; never logged.
"""
from __future__ import annotations

import hashlib
import json
import re
import logging
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from tideforge.tfbench.config import Paths, Settings
from tideforge.tfbench.rules import slug

log = logging.getLogger("tfbench.download")

SKIP_EXT = {"npy", "npz", "h5", "hdf5", "hdf", "nc", "nc4", "netcdf", "pkl", "pickle", "pt", "pth", "ckpt", "bin",
            "safetensors", "mat", "mp4", "avi", "mov", "wav", "mp3", "png", "jpg", "jpeg", "gif", "tif", "tiff", "pdf",
            "docx", "xlsx", "xls", "xml", "html", "py", "ipynb", "sh", "lock", "crc", "rds", "rdata", "sav", "dta",
            "sqlite", "db", "duckdb", "grib", "grb", "grib2", "7z", "rar", "bz2", "xz", "zst", "fits", "fit", "fits.gz",
            "fit.gz"}
# readable formats whose content is not a time series (owner 2026-11-12: AstroM3 spectra; image / embedding tables)
NON_TS = re.compile(r"(^|/)spectra/|(^|/|_)(sky_)?img(s)?(/|_|\.)|(^|/)images?/|embedding", re.I)


def ext(path: str) -> str:
    name = path.rsplit("/", 1)[-1].lower()
    for multi in ("tar.gz", "csv.gz", "tsv.gz", "json.gz", "jsonl.gz", "txt.gz", "parquet.gz", "fits.gz", "fit.gz"):
        if name.endswith("." + multi):
            return multi
    return name.rsplit(".", 1)[-1] if "." in name else ""


_DATE_SEG = re.compile(r"^(?:(?:19|20)\d{2}(?:[-_.]?\d{2}){0,2}|\d{2}[-_.]\d{2}[-_.](?:19|20)\d{2}|(?:0?[1-9]|1[0-2])|"
                       r"(?:0?[1-9]|[12]\d|3[01])|(?:year|month|day|date|dt|hour|week)=.*)$", re.I)
_DATE_TOK = re.compile(r"(?:19|20)\d{6}T\d{4,6}Z?|(?:19|20)\d{2}(?:[-_.]?(?:0[1-9]|1[0-2]))(?:[-_.]?(?:0[1-9]|[12]\d|3[01]))?"
                       r"|\d{2}[-_.]\d{2}[-_.](?:19|20)\d{2}|(?<![a-z0-9])(?:19|20)\d{2}(?![0-9])|[0-9a-f]{8,}|\d{8,}", re.I)


def series_key(path: str) -> str:
    """The part of a file path that names a series: folders that are dates (2022, 04, year=2026, dt=...) dropped, and
    dates, timestamps, long digit runs and hashes removed from the rest. Files of one series over time share a key; time
    partitions of all series collapse to a few keys."""
    segs = path.split("/")
    keep = [x for x in segs[:-1] if not _DATE_SEG.match(x)]
    name = _DATE_TOK.sub("", segs[-1])
    return "/".join(keep + [name])


def file_caps(p: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """Per repo with more than max_files_per_series_repo distinct series keys: keep the files of a seeded sample of that
    many keys (sha256(seed|repo|key) order, stable when files are added). Marks p.fetch / p.why / p.sampled in place and
    returns one row per capped repo."""
    p["sampled"] = False
    rows = []
    cap = int(s.max_files_per_series_repo or 0)
    if cap <= 0:
        return pd.DataFrame(rows, columns=["repo_id", "files_before", "files_after", "series_keys", "keys_kept", "reason"])
    data_ext = ~p["ext"].isin({"json", "yaml", "yml", "md", "txt"}) | (p["bytes"] > 1_000_000)
    for repo, g in p[p["fetch"] & data_ext].groupby("repo_id"):
        keys = g["path"].map(series_key)
        uk = keys.unique()
        if len(uk) <= cap:
            continue
        order = sorted(uk, key=lambda k: hashlib.sha256(f"{s.seed}|{repo}|{k}".encode()).hexdigest())
        kept = set(order[:cap])
        drop = g.index[~keys.isin(kept)]
        p.loc[drop, "fetch"] = False
        p.loc[drop, "why"] = f"file cap: series not in the seeded sample of {cap}"
        p.loc[g.index[keys.isin(kept)], "sampled"] = True
        rows.append(dict(repo_id=repo, files_before=len(g), files_after=int(len(g) - len(drop)), series_keys=len(uk),
                         keys_kept=cap, reason=f"{len(uk)} series keys (one series per file or folder) > {cap}; "
                         f"example key {uk[0]!r}"))
    return pd.DataFrame(rows, columns=["repo_id", "files_before", "files_after", "series_keys", "keys_kept", "reason"])


_PART_DATE = re.compile(
    r"(?P<ymdt>(?<!\d)(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])(?=T|\D|$))"
    r"|(?P<ymd>(?<!\d)(?:19|20)\d{2}(?P<sep>[-_./])(?:0[1-9]|1[0-2])(?P=sep)(?:0[1-9]|[12]\d|3[01])(?!\d))"
    r"|(?P<dmy>(?<!\d)(?:0[1-9]|[12]\d|3[01])[-_.](?:0[1-9]|1[0-2])[-_.](?:19|20)\d{2}(?!\d))"
    r"|(?P<hive>year=(?:19|20)\d{2}(?:/month=\d{1,2}(?:/day=\d{1,2})?)?)"
    r"|(?P<ym>(?<!\d)(?:19|20)\d{2}[-_./](?:0[1-9]|1[0-2])(?!\d))"
    r"|(?P<y>(?<![\d=])(?:19|20)\d{2}(?![\d]))", re.I)


def partition_date(path: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """(start, end) of the time partition a file covers, from the last date in its path (the one nearest the file name):
    a day (2021-03-01, 2021/03/01 folders), a month (2022/04, 2020-09, year=/month=) or a year (2024.csv)."""
    last = None
    for m in _PART_DATE.finditer(path):
        last = m
    if last is None:
        return None
    k, v = last.lastgroup, last.group(0)
    try:
        if k == "ymdt":
            a = pd.Timestamp(v[:8])
            return a, a + pd.Timedelta(days=1)
        if k == "ymd":
            a = pd.Timestamp(re.sub(r"[_./]", "-", v))
            return a, a + pd.Timedelta(days=1)
        if k == "dmy":
            d, mo, y = re.split(r"[-_.]", v)
            a = pd.Timestamp(int(y), int(mo), int(d))
            return a, a + pd.Timedelta(days=1)
        if k == "hive":
            nums = [int(x) for x in re.findall(r"=(\d+)", v)]
            a = pd.Timestamp(nums[0], nums[1] if len(nums) > 1 else 1, nums[2] if len(nums) > 2 else 1)
            step = [pd.DateOffset(years=1), pd.DateOffset(months=1), pd.Timedelta(days=1)][len(nums) - 1]
            return a, a + step
        if k == "ym":
            y, mo = re.split(r"[-_./]", v)
            a = pd.Timestamp(int(y), int(mo), 1)
            return a, a + pd.DateOffset(months=1)
        a = pd.Timestamp(int(v), 1, 1)
        return a, a + pd.DateOffset(years=1)
    except (ValueError, OverflowError):
        return None


def path_freq(paths: pd.Series) -> float | None:
    """Finest frequency named by a folder or file-name word in the paths ("hourly", "10min", "1d"), or None."""
    from tideforge.devdata.hf_derive import freq_from_name
    found = set()
    for pth in paths.sample(min(len(paths), 500), random_state=0):
        for seg in re.split(r"[/]", pth)[:-1] + [re.sub(r"\.[a-z0-9.]+$", "", pth.rsplit("/", 1)[-1])]:
            for tok in [seg] + re.split(r"[_\-\s]", seg):
                v = freq_from_name(tok) if tok else None
                if v:
                    found.add(float(v))
    return min(found) if found else None


def partition_cut(p: pd.DataFrame, s: Settings, freq_seconds: dict) -> pd.DataFrame:
    """Time-partitioned repos (few series keys, many files per key) above max_partition_files files: keep only the newest
    partitions that cover what canon keeps, max_points steps per series x 1.2 at the data frequency. Frequency: the
    profile's (finest over the repo's catalog rows); where the profile has none, the median spacing between file dates,
    which is never finer than the data, so the kept span is never too short. Files without a date are kept."""
    cols = ["repo_id", "files_before", "files_after", "series_keys", "keys_kept", "reason"]
    lim = int(s.max_partition_files or 0)
    rows = []
    if lim <= 0:
        return pd.DataFrame(rows, columns=cols)
    for repo, g in p[p["fetch"]].groupby("repo_id"):
        if len(g) <= lim:
            continue
        nkeys = g["path"].map(series_key).nunique()
        if nkeys * 10 > len(g):              # not time partitions (one series per file): the series cap handles these
            continue
        lo, hi = pd.Timestamp("1900-01-01"), pd.Timestamp.now() + pd.DateOffset(years=1)
        se = g["path"].map(partition_date).map(lambda x: x if x is not None and lo <= x[0] <= hi else None)
        dated = se.dropna()
        if len(dated) < 0.9 * len(g):
            rows.append(dict(repo_id=repo, files_before=len(g), files_after=len(g), series_keys=nkeys, keys_kept=nkeys,
                             reason=f"time partitions but only {len(dated)} of {len(g)} file names carry a date: not cut"))
            continue
        starts = pd.Series(sorted({a for a, _ in dated}))
        fs = freq_seconds.get(repo)
        how = "profile frequency"
        if not fs or fs != fs:
            fs = path_freq(g["path"])
            how = "frequency word in the file paths"
        if not fs or fs != fs:
            d = starts.diff().dropna().dt.total_seconds()
            fs = float(d[d > 0].median()) if (d > 0).any() else None
            how = "median spacing of file dates (profile has no frequency)"
        if not fs:
            continue
        newest = max(b for _, b in dated)
        span = s.partition_margin * s.max_points * fs
        cutoff = newest - pd.Timedelta(seconds=span) if span < 200 * 365.25 * 86400 else pd.Timestamp("1800-01-01")
        keep = se.map(lambda x: x is None or x[1] > cutoff)
        drop = g.index[~keep.to_numpy()]
        if not len(drop):
            rows.append(dict(repo_id=repo, files_before=len(g), files_after=len(g), series_keys=nkeys, keys_kept=nkeys,
                             reason=f"time partitions; {s.max_points} x {s.partition_margin:g} steps of {fs:g} s ({how}) "
                                    f"reach before the oldest file: nothing cut"))
            continue
        p.loc[drop, "fetch"] = False
        p.loc[drop, "why"] = f"partition cut: older than {cutoff.date()}"
        rows.append(dict(repo_id=repo, files_before=len(g), files_after=int(len(g) - len(drop)), series_keys=nkeys,
                         keys_kept=nkeys, reason=f"time partitions; kept files after {cutoff.date()} = {s.max_points} x "
                         f"{s.partition_margin:g} steps of {fs:g} s ({how}) before the newest ({newest.date()})"))
    return pd.DataFrame(rows, columns=cols)


def plan(selection: pd.DataFrame, files: pd.DataFrame, s: Settings | None = None,
         freq_seconds: dict | None = None) -> pd.DataFrame:
    """One row per unique file: repo, revision, path, bytes, dataset_ids, fetch (bool), why, sampled. With s, the per-repo
    file cap is applied (p.attrs["caps"] holds the capped repos)."""
    sel = selection[selection["keep"].astype(str).str.lower() == "true"]
    ids = dict(zip(sel["catalog_row"].astype(int), sel["dataset_id"]))
    f = files[files["catalog_row"].astype(int).isin(ids)].copy()
    f["revision"] = f["revision"].fillna("main")
    f["dataset_id"] = f["catalog_row"].astype(int).map(ids)
    g = f.groupby(["repo_id", "revision", "path"], as_index=False).agg(
        bytes=("bytes", "max"), dataset_ids=("dataset_id", lambda s: ";".join(sorted(set(s)))))
    g["ext"] = g["path"].map(ext)
    g["fetch"] = ~g["ext"].isin(SKIP_EXT)
    g["why"] = g.apply(lambda r: "" if r["fetch"] else f"format {r['ext']} not read in v1", axis=1)
    nts = g["fetch"] & g["path"].map(lambda x: bool(NON_TS.search(x)))
    g.loc[nts, "fetch"] = False
    g.loc[nts, "why"] = "not a time series (spectra / images / embeddings by name)"
    g["sampled"] = False
    if s is not None:
        caps = file_caps(g, s)
        parts = partition_cut(g, s, freq_seconds or {})
        g.attrs["caps"] = pd.concat([caps, parts], ignore_index=True) if len(parts) else caps
    return g


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _load_manifest(p: Path) -> dict:
    out = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                out[e["path"]] = e
    return out


def with_retry(call, tries: int = 8, sleep=time.sleep):
    """Retry on HTTP 429 (Retry-After, else exponential back-off from 5 s) and on connection errors."""
    for k in range(tries):
        try:
            return call()
        except Exception as ex:
            resp = getattr(ex, "response", None)
            code = getattr(resp, "status_code", None)
            transient = code in (429, 500, 502, 503, 504) or type(ex).__name__ in ("ConnectionError", "ReadTimeout",
                                                                                 "ConnectTimeout", "Timeout")
            if not transient or k == tries - 1:
                raise
            ra = (getattr(resp, "headers", None) or {}).get("Retry-After")
            try:
                wait = float(ra)
            except (TypeError, ValueError):
                wait = min(5.0 * 2 ** k, 300.0)
            sleep(wait)


def is_xet_401(ex: Exception) -> bool:
    """Xet / CAS token expired during a long repo download ("CAS Client Error ... 401 Unauthorized ... xethub")."""
    m = f"{type(ex).__name__}: {ex}".lower()
    return "401" in m and any(k in m for k in ("xethub", "cas client", "cas-server", "reconstruction", "xet"))


def http_fetch(repo: str, path: str, revision: str, local_dir: Path, timeout: float = 600.0) -> str:
    """Plain HTTPS download of one file through the Hub's resolve URL (no Xet client), standard library only.
    The token goes only to huggingface.co; the redirect target (a signed CDN / cas-bridge URL) is fetched without it.
    Written to a temporary name and renamed, so a broken transfer never leaves a partial file under the real name."""
    import urllib.parse
    import urllib.request

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None

    url = (f"https://huggingface.co/datasets/{repo}/resolve/{urllib.parse.quote(revision, safe='')}/"
           f"{urllib.parse.quote(path)}")
    headers = {"User-Agent": "tfbench-download"}
    tok = os.environ.get("HF_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    opener = urllib.request.build_opener(NoRedirect)
    try:
        resp = opener.open(urllib.request.Request(url, headers=headers), timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code not in (301, 302, 303, 307, 308):
            raise
        loc = urllib.parse.urljoin(url, e.headers["Location"])
        same = urllib.parse.urlparse(loc).hostname == "huggingface.co"
        resp = urllib.request.urlopen(urllib.request.Request(loc, headers=headers if same else
                                                             {"User-Agent": headers["User-Agent"]}), timeout=timeout)
    dst = Path(local_dir) / path
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    with resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh, 1 << 22)
    tmp.replace(dst)
    return str(dst)


def download_repo(repo: str, rows: pd.DataFrame, paths: Paths, s: Settings, fetch=None, disk_free=None,
                  sleep=time.sleep, fallback=None) -> dict:
    """Fetch the files of one repo, file_workers files in parallel (each fetch = one HEAD + one GET; the HEAD is what made
    the sequential version slow). fetch(repo, path, revision, local_dir) -> local path (default hf_hub_download)."""
    fetch = fetch or _hf_fetch
    fallback = fallback or http_fetch
    disk_free = disk_free or (lambda p: shutil.disk_usage(p).free)
    root = paths.raw / slug(repo)
    root.mkdir(parents=True, exist_ok=True)
    mpath = paths.manifest / f"{slug(repo)}.jsonl"
    mpath.parent.mkdir(parents=True, exist_ok=True)
    done = _load_manifest(mpath)
    todo = [r for r in rows.to_dict("records")
            if not (r["path"] in done and done[r["path"]]["status"] == "ok"
                    and (root / r["path"]).exists() and (root / r["path"]).stat().st_size == done[r["path"]]["bytes"])]
    need = sum(int(r["bytes"] or 0) for r in todo) * s.disk_margin
    if todo and disk_free(paths.raw) < need:
        log.warning("%s: skipped, needs %.1f GB free, has %.1f GB", repo, need / 1e9, disk_free(paths.raw) / 1e9)
        return {"repo": repo, "status": "skipped: disk", "files": len(todo), "errors": 0, "xet401": 0, "http_fallback": 0}
    n = {"ok": 0, "err": 0, "xet401": 0, "http_fallback": 0}
    lock = threading.Lock()
    t0 = time.time()

    def one(r, mf):
        e = {"repo": repo, "revision": r["revision"], "path": r["path"], "dataset_ids": r["dataset_ids"],
             "sampled": bool(r.get("sampled", False))}
        try:
            lp, x401, via = None, 0, "hub"
            for k in range(s.xet_retries + 1):
                try:          # each call is a fresh hf_hub_download, which asks the Hub for a new Xet token
                    lp = Path(with_retry(lambda: fetch(repo, r["path"], r["revision"], root), sleep=sleep))
                    break
                except Exception as ex:
                    if not is_xet_401(ex):
                        raise
                    x401 += 1
                    if k < s.xet_retries:
                        sleep(2.0 * (k + 1))
            if lp is None:    # Xet kept refusing: plain HTTPS for this file
                lp = Path(with_retry(lambda: fallback(repo, r["path"], r["revision"], root), sleep=sleep))
                via = "http-fallback"
            if x401:
                with lock:
                    n["xet401"] += x401
                    n["http_fallback"] += via == "http-fallback"
            e.update(bytes=lp.stat().st_size, sha256=sha256(lp), status="ok", via=via, xet401=x401)
        except Exception as ex:  # one file failing never stops the run
            e.update(bytes=0, sha256="", status=f"error: {type(ex).__name__}: {str(ex)[:300]}")
            log.error("%s %s: %s", repo, r["path"], e["status"])
        with lock:
            n["ok" if e["status"] == "ok" else "err"] += 1
            mf.write(json.dumps(e) + "\n")
            mf.flush()
            k = n["ok"] + n["err"]
            if k % 2000 == 0:
                log.info("%s: %d / %d files (%.0f s)", repo, k, len(todo), time.time() - t0)

    with open(mpath, "a", encoding="utf-8") as mf, ThreadPoolExecutor(max(1, s.file_workers)) as ex:
        list(ex.map(lambda r: one(r, mf), todo))
    n_ok, n_err = n["ok"], n["err"]
    log.info("%s: %d fetched, %d errors, %d already there, %d Xet 401 (%d files via HTTP fallback) (%.0f s)", repo, n_ok,
             n_err, len(rows) - len(todo), n["xet401"], n["http_fallback"], time.time() - t0)
    return {"repo": repo, "status": "ok" if not n_err else "errors", "files": len(rows), "errors": n_err,
            "xet401": n["xet401"], "http_fallback": n["http_fallback"]}


def _hf_fetch(repo: str, path: str, revision: str, local_dir: Path) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, path, repo_type="dataset", revision=revision, local_dir=str(local_dir))


def run(selection: pd.DataFrame, files: pd.DataFrame, paths: Paths, s: Settings, only: list[str] | None = None,
        jobs: int | None = None, freq_seconds: dict | None = None) -> pd.DataFrame:
    p = plan(selection, files, s, freq_seconds)
    paths.data.mkdir(parents=True, exist_ok=True)
    p.to_csv(paths.data / "download_plan.csv", index=False)
    caps = p.attrs.get("caps")
    if caps is not None:
        caps.to_csv(paths.data / "file_caps.csv", index=False)
        for r in caps.itertuples():
            log.info("file cap %s: %d -> %d files (%s)", r.repo_id, r.files_before, r.files_after, r.reason)
    skipped = p[~p["fetch"]]
    log.info("plan: %d files (%.1f GB) to fetch, %d files (%.1f GB) skipped by format", int(p["fetch"].sum()),
             p.loc[p["fetch"], "bytes"].sum() / 1e9, len(skipped), skipped["bytes"].sum() / 1e9)
    p = p[p["fetch"]]
    if only:
        p = p[p["repo_id"].isin(only) | p["dataset_ids"].apply(lambda d: any(o in d.split(";") for o in only))]
    repos = sorted(p["repo_id"].unique(), key=lambda r: p.loc[p["repo_id"] == r, "bytes"].sum())  # small first
    t0 = time.time()
    with ThreadPoolExecutor(jobs or s.download_jobs) as ex:
        res = list(ex.map(lambda r: download_repo(r, p[p["repo_id"] == r], paths, s), repos))
    log.info("download done: %d repos in %.0f s", len(repos), time.time() - t0)
    summ = pd.DataFrame(res)
    if not summ.empty:
        summ.to_csv(paths.data / "download_summary.csv", index=False)
        x = summ[summ.get("xet401", 0) > 0] if "xet401" in summ else summ.iloc[0:0]
        log.info("Xet 401: %d events in %d repos, %d files via HTTP fallback", int(summ.get("xet401", pd.Series([0])).sum()),
                 len(x), int(summ.get("http_fallback", pd.Series([0])).sum()))
        for r in x.sort_values("xet401", ascending=False).itertuples():
            log.info("  %s: %d Xet 401, %d via HTTP, %d errors", r.repo, r.xet401, r.http_fallback, r.errors)
    combine(paths)
    return pd.DataFrame(res)


def combine(paths: Paths) -> pd.DataFrame:
    rows = []
    for m in sorted(paths.manifest.glob("*.jsonl")):
        rows += list(_load_manifest(m).values())   # last entry per path wins
    df = pd.DataFrame(rows)
    df.to_csv(paths.data / "manifest.csv", index=False)
    return df


def download_refs(paths: Paths, s: Settings) -> None:
    """The three reference benchmarks for the leakage audit (whole repos)."""
    from huggingface_hub import snapshot_download
    for repo in s.refs:
        try:
            snapshot_download(repo, repo_type="dataset", local_dir=str(paths.refs / slug(repo)), max_workers=8)
            log.info("ref %s: ok", repo)
        except Exception as ex:
            log.error("ref %s: %s: %s", repo, type(ex).__name__, str(ex)[:300])
