"""A2 download (JarvisLabs only; never run on the dev PC).

Only the files in each selected row's scope (hf_profile_files.csv), each (repo, revision, path) once even when several
catalog rows share it. Formats the readers cannot use (npy, h5, netcdf, media, pickles, ...) are not downloaded; the plan
lists them. huggingface_hub with hf_transfer (HF_HUB_ENABLE_HF_TRANSFER=1, set before the import). Resumable: a file whose
manifest entry has the same size on disk is not fetched again. Inside a repo, file_workers files are fetched in parallel
(hf_hub_download per file, retried on 429 with Retry-After); download_jobs repos run in parallel. Before each repo: free disk >= repo bytes x disk_margin,
else the repo is skipped and logged. Parallel across repos.

Manifest: $TFB_DATA/manifest/<repo slug>.jsonl, one json line per file {repo, revision, path, bytes, sha256, status,
dataset_ids}; `combine` writes manifest.csv. HF_TOKEN is read by huggingface_hub from the environment; never logged.
"""
from __future__ import annotations

import hashlib
import json
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
            "sqlite", "db", "duckdb", "grib", "grb", "grib2", "7z", "rar", "bz2", "xz", "zst"}


def ext(path: str) -> str:
    name = path.rsplit("/", 1)[-1].lower()
    for multi in ("tar.gz", "csv.gz", "tsv.gz", "json.gz", "jsonl.gz", "txt.gz", "parquet.gz"):
        if name.endswith("." + multi):
            return multi
    return name.rsplit(".", 1)[-1] if "." in name else ""


def plan(selection: pd.DataFrame, files: pd.DataFrame) -> pd.DataFrame:
    """One row per unique file: repo, revision, path, bytes, dataset_ids, fetch (bool), why."""
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


def download_repo(repo: str, rows: pd.DataFrame, paths: Paths, s: Settings, fetch=None, disk_free=None,
                  sleep=time.sleep) -> dict:
    """Fetch the files of one repo, file_workers files in parallel (each fetch = one HEAD + one GET; the HEAD is what made
    the sequential version slow). fetch(repo, path, revision, local_dir) -> local path (default hf_hub_download)."""
    fetch = fetch or _hf_fetch
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
        return {"repo": repo, "status": "skipped: disk", "files": len(todo)}
    n = {"ok": 0, "err": 0}
    lock = threading.Lock()
    t0 = time.time()

    def one(r, mf):
        e = {"repo": repo, "revision": r["revision"], "path": r["path"], "dataset_ids": r["dataset_ids"]}
        try:
            lp = Path(with_retry(lambda: fetch(repo, r["path"], r["revision"], root), sleep=sleep))
            e.update(bytes=lp.stat().st_size, sha256=sha256(lp), status="ok")
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
    log.info("%s: %d fetched, %d errors, %d already there (%.0f s)", repo, n_ok, n_err, len(rows) - len(todo),
             time.time() - t0)
    return {"repo": repo, "status": "ok" if not n_err else "errors", "files": len(rows), "errors": n_err}


def _hf_fetch(repo: str, path: str, revision: str, local_dir: Path) -> str:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, path, repo_type="dataset", revision=revision, local_dir=str(local_dir))


def run(selection: pd.DataFrame, files: pd.DataFrame, paths: Paths, s: Settings, only: list[str] | None = None,
        jobs: int | None = None) -> pd.DataFrame:
    p = plan(selection, files)
    paths.data.mkdir(parents=True, exist_ok=True)
    p.to_csv(paths.data / "download_plan.csv", index=False)
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
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    from huggingface_hub import snapshot_download
    for repo in s.refs:
        try:
            snapshot_download(repo, repo_type="dataset", local_dir=str(paths.refs / slug(repo)), max_workers=8)
            log.info("ref %s: ok", repo)
        except Exception as ex:
            log.error("ref %s: %s: %s", repo, type(ex).__name__, str(ex)[:300])
