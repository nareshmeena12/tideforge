"""Download the allow-listed real datasets (tidemark/real/allowlist.py) into $SCRATCH/tidemark/real/ and write manifest.json.

  python scripts/real/download_chronos.py --list-hub                  # which configs the Hub has (nothing is downloaded)
  python scripts/real/download_chronos.py [--root R] [--only a,b] [--workers 4] [--hourly-shards 12]
  python scripts/real/download_chronos.py --stats-only                # recompute the manifest from files already on disk
  python scripts/real/download_chronos.py --hub-sizes                 # Hub file bytes + row counts of the allowed configs AND the
                                                                      # 14 weatherbench_hourly_* candidates; downloads nothing

Login node or data-transfer node only: compute nodes have no internet. Resumable: a dataset (or, for the hourly weatherbench
variables, a shard) already in the manifest is skipped (--force redoes it), and an interrupted download continues where it stopped.
Each repo is pinned to the revision recorded in the manifest on the first run. One dataset failing does not stop the others: the
failures are listed at the end and the exit code is 1. Order: the ordinary chronos_datasets sets, then chronos_datasets_extra, then
the 11 hourly weatherbench variables (the big ones, last).

weatherbench_hourly_<variable> lives in the Hub folder weatherbench_hourly/<variable>/train-NNNNN-of-MMMMM.parquet. The 8 pressure-level
variables (about 738 GB in all) are NOT taken in full: --hourly-shards N (default 12) downloads N evenly spaced shards of each
(indices spread over 0..total-1, first and last included). The 3 surface variables are taken in full. The manifest records which shards
were taken and the series / points of each, plus a size-weighted extrapolation to the full variable, for the sizing report.

chronos_datasets_extra has NO data files on the Hub: its loading script downloads from Kaggle when it is loaded. It needs, in YOUR
environment: the `kaggle` package, your own Kaggle credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME / KAGGLE_KEY), and a
`datasets` version that still runs loading scripts (< 4). If any of that is missing the two extra sets fail with an explanation and
the rest of the run is unaffected. Authentication for Hugging Face: the datasets are public; if you need a token set HF_TOKEN in YOUR
shell. This script never reads, prints or writes a token or a Kaggle credential.

manifest.json, per dataset: repo, hf_config, revision (commit hash), n_series, total_points, mean_len, min_len, nan_fraction,
size_bytes, files (hourly: also "shards"). A series is one (row, column) of every numeric list column except `timestamp`; points count
NaNs too.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from tidemark.real import allowlist as AL
from tidemark.real.allowlist import HOURLY_PREFIX, SURFACE_HOURLY, hub_prefix, is_hourly  # noqa: F401



def default_root() -> Path:
    return Path(os.environ.get("REAL_ROOT") or Path(os.environ.get("SCRATCH", ".")) / "tidemark" / "real")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def evenly_spaced(total: int, n: int) -> list[int]:
    """n shard indices spread evenly over 0..total-1 (first and last included); all of them if n >= total."""
    if n >= total:
        return list(range(total))
    if n <= 1:
        return [0][:max(n, 0)]
    return sorted({int(i * (total - 1) / (n - 1) + 0.5) for i in range(n)})      # half rounds up: deterministic


def plan_files(config: str, files: list[tuple[str, int]], n_shards: int) -> list[tuple[str, int]]:
    """files = [(path in repo, bytes)] of the config's folder, sorted. Ordinary configs and the surface variables: all of them;
    a pressure-level variable: n_shards evenly spaced ones."""
    files = sorted(files)
    if is_hourly(config) and config not in SURFACE_HOURLY:
        return [files[i] for i in evenly_spaced(len(files), n_shards)]
    return files


def dataset_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix in (".arrow", ".parquet"))


def _batches(path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    if path.suffix == ".parquet":
        yield from pq.ParquetFile(path).iter_batches(batch_size=1024)
        return
    with pa.memory_map(str(path)) as src:
        try:
            reader = pa.ipc.open_stream(src)
            yield from reader
        except pa.ArrowInvalid:
            reader = pa.ipc.open_file(src)
            for i in range(reader.num_record_batches):
                yield reader.get_batch(i)


def stats_files(files: list[Path]) -> dict:
    """Series count, points, lengths and NaN fraction of every numeric list column except `timestamp`, over these files."""
    import pyarrow as pa
    import pyarrow.compute as pc
    n = pts = nan = 0
    mn, columns = None, set()
    if not files:
        raise RuntimeError("no .arrow / .parquet files to count")
    for f in files:
        for b in _batches(f):
            for name, col in zip(b.schema.names, b.columns):
                t = col.type
                if name == "timestamp" or not (pa.types.is_list(t) or pa.types.is_large_list(t)):
                    continue
                if not (pa.types.is_floating(t.value_type) or pa.types.is_integer(t.value_type)):
                    continue
                columns.add(name)
                lens = pc.list_value_length(col).to_numpy(zero_copy_only=False)
                n += len(lens)
                pts += int(lens.sum())
                if len(lens):
                    mn = int(lens.min()) if mn is None else min(mn, int(lens.min()))
                flat = pc.list_flatten(col)
                nan += int(pc.sum(pc.is_nan(flat.cast(pa.float64()))).as_py() or 0) + int(flat.null_count)
    return {"n_series": n, "total_points": pts, "mean_len": (pts / n) if n else 0.0, "min_len": mn if mn is not None else 0,
            "nan_fraction": (nan / pts) if pts else 0.0, "columns": sorted(columns),
            "size_bytes": sum(p.stat().st_size for p in files), "files": len(files)}


def stats(folder: Path) -> dict:
    files = dataset_files(folder)
    if not files:
        raise RuntimeError(f"no .arrow / .parquet files under {folder}")
    return stats_files(files)


def merge_shard_stats(shards: list[dict]) -> dict:
    """Dataset-level numbers from per-shard stats (nan_fraction weighted by points, min of the minima)."""
    n = sum(s["n_series"] for s in shards)
    pts = sum(s["total_points"] for s in shards)
    return {"n_series": n, "total_points": pts, "mean_len": (pts / n) if n else 0.0,
            "min_len": min(s["min_len"] for s in shards),
            "nan_fraction": (sum(s["nan_fraction"] * s["total_points"] for s in shards) / pts) if pts else 0.0,
            "columns": sorted({c for s in shards for c in s["columns"]}),
            "size_bytes": sum(s["size_bytes"] for s in shards), "files": len(shards)}


def extrapolate(shards: list[dict], total_files: int, total_bytes: int) -> dict:
    """Full-variable estimate from the shards taken. Shards differ in size (the last is short), so the estimate scales by BYTES:
    points per byte of the taken shards x the variable's total bytes. The per-shard mean is given too. Estimates, not counts."""
    taken_bytes = sum(s["size_bytes"] for s in shards)
    pts, n = sum(s["total_points"] for s in shards), sum(s["n_series"] for s in shards)
    return {"total_shards": total_files, "total_bytes": total_bytes, "taken_bytes": taken_bytes,
            "fraction_of_bytes": taken_bytes / total_bytes if total_bytes else None,
            "est_full_points_by_bytes": pts * total_bytes / taken_bytes if taken_bytes else None,
            "est_full_series_by_bytes": n * total_bytes / taken_bytes if taken_bytes else None,
            "est_full_points_by_shard_mean": pts / len(shards) * total_files if shards else None,
            "note": "ESTIMATE from the shards taken (parquet bytes as the size proxy)"}


def hub_sizes(configs: dict[str, str], info_of, rows_of) -> dict:
    """{config: {repo, revision, files, file_bytes, n_rows}} from Hub metadata only. `info_of(repo)` -> dataset_info with file sizes,
    `rows_of(repo, config)` -> row count or None (datasets-server /size). Row counts are series (rows); points are NOT known
    without reading the data, so none is invented here."""
    out, infos = {}, {}
    for c, repo in configs.items():
        info = infos.setdefault(repo, info_of(repo))
        pre = hub_prefix(c) + "/"
        sib = [x for x in info.siblings if x.rfilename.startswith(pre)]
        out[c] = {"repo": repo, "revision": info.sha, "files": len(sib), "file_bytes": sum(int(x.size or 0) for x in sib),
                  "n_rows": rows_of(repo, c)}
    return out


def _server_rows(repo: str, config: str):
    import urllib.parse
    import urllib.request
    url = "https://datasets-server.huggingface.co/size?" + urllib.parse.urlencode({"dataset": repo, "config": config})
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return int(json.loads(r.read())["size"]["config"]["num_rows"])
    except Exception:  # noqa: BLE001
        return None


def cmd_hub_sizes(root: Path) -> int:
    from huggingface_hub import HfApi
    api = HfApi()
    configs = dict(AL.allowed())
    configs.update({c: AL.CHRONOS for c in AL.HOURLY_CANDIDATES})
    res = hub_sizes(configs, lambda r: api.dataset_info(r, files_metadata=True), _server_rows)
    write_json(root / "hub_sizes.json", res)
    print(f"{'config':58s} {'files':>5s} {'MiB':>9s} {'rows':>10s}  status")
    for c, d in res.items():
        st = "excluded hourly candidate (NOT downloaded)" if c in AL.EXCLUDED else "allowed"
        print(f"{c:58s} {d['files']:>5d} {d['file_bytes'] / 2**20:>9.1f} {d['n_rows'] if d['n_rows'] is not None else '-':>10}  {st}")
    hourly = sum(d["file_bytes"] for c, d in res.items() if c in AL.HOURLY_CONFIRMED)
    print(f"\nthe 11 confirmed hourly variables together: {hourly / 2**30:.1f} GiB on the Hub; written to {root / 'hub_sizes.json'}")
    return 0


def list_hub() -> int:
    from huggingface_hub import HfApi
    api = HfApi()
    allowed = AL.allowed()
    for repo in (AL.CHRONOS, AL.CHRONOS_EXTRA):
        info = api.dataset_info(repo)
        folders = sorted({s.rfilename.split("/")[0] for s in info.siblings if "/" in s.rfilename})
        print(f"{repo}  revision {info.sha}")
        for c in folders:
            tag = "ALLOWED" if allowed.get(c) == repo else ("excluded" if c in AL.EXCLUDED else "not allowed")
            print(f"  {c:55s} {tag}")
    return 0


# ------------------------------------------------------------------------------------------------------------ one dataset
def gib(b: float) -> str:
    return f"{b / 2**30:.2f} GiB"


def fetch_extra(repo: str, config: str, rev: str | None, dest: Path) -> None:
    """chronos_datasets_extra has no data files: the repo's loading script downloads from Kaggle. Runs it in the owner's
    environment and stores the result as parquet. Fails with an explanation if kaggle / credentials / datasets<4 are missing."""
    try:
        import datasets
        ds = datasets.load_dataset(repo, config, revision=rev, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"{repo}:{config} could not be built ({type(e).__name__}: {str(e)[:200]}). Its Hub repo holds only a loading script "
            "that downloads from Kaggle: you need the `kaggle` package, your own Kaggle credentials (~/.kaggle/kaggle.json or "
            "KAGGLE_USERNAME / KAGGLE_KEY) and `datasets` < 4 (scripts are unsupported from 4.0). Fix that in your shell and rerun "
            f"with --only {config}.") from e
    split = ds["train"] if "train" in ds else ds[list(ds)[0]]
    dest.mkdir(parents=True, exist_ok=True)
    split.to_parquet(str(dest / "train-00000-of-00001.parquet"))


def do_dataset(c: str, repo: str, man: dict, root: Path, a, infos: dict, api, snapshot_download) -> None:
    """Download (if needed), count, and record one config. Raises on failure; the caller carries on with the next."""
    local = root / "raw" / repo.split("/")[-1]
    rev = man["revisions"].get(repo) or (None if a.stats_only else api.dataset_info(repo).sha)
    t0 = time.time()
    if repo == AL.CHRONOS_EXTRA:
        dest = local / c
        if not a.stats_only and not dataset_files(dest):
            print(f"    planned: built from Kaggle by the repo's loading script (size unknown until it runs)", flush=True)
            fetch_extra(repo, c, rev, dest)
        if rev:
            man["revisions"][repo] = rev
        s = stats(dest)
        man["datasets"][c] = {"repo": repo, "hf_config": c, "revision": rev, **s}
    else:
        if repo not in infos:
            infos[repo] = api.dataset_info(repo, files_metadata=True)
        pre = hub_prefix(c) + "/"
        allf = [(x.rfilename, int(x.size or 0)) for x in infos[repo].siblings if x.rfilename.startswith(pre)
                and x.rfilename.endswith((".parquet", ".arrow"))]
        if not allf:
            raise RuntimeError(f"no data files under {repo}:{pre} on the Hub")
        take = plan_files(c, allf, a.hourly_shards)
        full_b, take_b = sum(b for _, b in allf), sum(b for _, b in take)
        done_shards = man["datasets"].get(c, {}).get("shards", {}).get("per_shard", {}) if is_hourly(c) and not a.force else {}
        todo = [f for f, _ in take if f not in done_shards]
        print(f"    planned: {len(take)} of {len(allf)} files, {gib(take_b)} (full config {gib(full_b)}); "
              f"{len(todo)} still to fetch ({gib(sum(b for f, b in take if f in todo))}); free on {root}: "
              f"{gib(shutil.disk_usage(root if root.exists() else root.parent).free)}", flush=True)
        if not a.stats_only and todo:
            snapshot_download(repo_id=repo, repo_type="dataset", revision=rev, allow_patterns=todo, local_dir=str(local),
                              max_workers=a.workers)
        if rev:
            man["revisions"][repo] = rev
        if is_hourly(c):
            per = dict(done_shards)
            for f, b in take:
                if f not in per:
                    p = local / f
                    if not p.exists():
                        raise RuntimeError(f"shard {f} is not on disk")
                    per[f] = {"index": int(Path(f).stem.split("-")[1]), "size_bytes": p.stat().st_size,
                              **({} if getattr(a, "skip_stats", False) else stats_files([p]))}
            shards = [per[f] for f, _ in take]
            counted = [x for x in shards if "total_points" in x]           # --skip-stats: shards without counts (the converter counts them)
            if not counted:
                raise RuntimeError("no shard of this variable has been counted yet; run once without --skip-stats")
            s = merge_shard_stats(counted)
            s["size_bytes"] = sum(x["size_bytes"] for x in shards)
            s["files"] = len(shards)
            s["shards"] = {"total": len(allf), "taken": [per[f]["index"] for f, _ in take], "per_shard": per,
                           "counted_shards": len(counted), "full_variable": extrapolate(counted, len(allf), full_b)}
        else:
            s = stats_files([local / f for f, _ in take])
        man["datasets"][c] = {"repo": repo, "hf_config": c, "revision": rev, **s}
    write_json(root / "manifest.json", man)                      # after every dataset: an interruption loses at most one
    s = man["datasets"][c]
    print(f"    {s['n_series']:,} series, {s['total_points']:,} points, mean len {s['mean_len']:.0f}, min {s['min_len']}, "
          f"NaN {s['nan_fraction']:.3%}, {gib(s['size_bytes'])} ({time.time() - t0:.0f} s)", flush=True)
    if is_hourly(c) and "shards" in s:
        fv = s["shards"]["full_variable"]
        print(f"    shards taken {s['shards']['taken']} of {s['shards']['total']}; estimated full variable: "
              f"{(fv['est_full_points_by_bytes'] or 0) / 1e9:.2f}B points", flush=True)


def needs_more_shards(config: str, have: dict, n_shards: int) -> bool:
    """A pressure-level hourly variable is done when its manifest entry has min(N, total) shards; an entry without shard
    information (an older manifest) is redone. Everything else in the manifest is done."""
    if not is_hourly(config) or config in SURFACE_HOURLY:
        return False
    sh = have.get("shards")
    return sh is None or len(sh["taken"]) < min(n_shards, sh["total"])


def order(names: list[str]) -> list[str]:
    """Ordinary chronos_datasets sets, then chronos_datasets_extra, then the hourly weatherbench variables."""
    allowed = AL.allowed()
    rank = lambda c: 2 if is_hourly(c) else (1 if allowed[c] == AL.CHRONOS_EXTRA else 0)  # noqa: E731
    return sorted(names, key=lambda c: (rank(c), names.index(c)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(default_root()))
    ap.add_argument("--only", default=None, help="comma-separated configs (must be on the allow-list)")
    ap.add_argument("--workers", type=int, default=4, help="parallel file downloads (keep low on a login node)")
    ap.add_argument("--hourly-shards", type=int, default=12, metavar="N",
                    help="shards per pressure-level weatherbench_hourly variable, evenly spaced (default 12); surface variables: all")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-stats", action="store_true", help="hourly shards: record file names and sizes only, do not count series / points "
                    "(the 107-shard run: counting every 0.8 GB shard takes hours; tidemark.real.convert counts them). Shards already "
                    "counted keep their numbers, which then describe a sample of the variable (shards.counted_shards).")
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--list-hub", action="store_true")
    ap.add_argument("--hub-sizes", action="store_true", help="Hub file sizes and row counts, allowed + hourly candidates; no download")
    a = ap.parse_args(argv)
    if a.list_hub:
        return list_hub()
    if a.hub_sizes:
        return cmd_hub_sizes(Path(a.root))
    if a.hourly_shards < 1:
        raise SystemExit("--hourly-shards must be at least 1")
    allowed = AL.allowed()
    names = [c.strip() for c in a.only.split(",")] if a.only else list(allowed)
    for c in names:
        AL.check(c, allowed.get(c, "?"))                       # unknown / excluded names stop here
    names = order(names)
    root = Path(a.root)
    mpath = root / "manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {}
    man.setdefault("datasets", {})
    man.setdefault("revisions", {})
    man["hourly_shards_requested"] = a.hourly_shards
    from huggingface_hub import HfApi, snapshot_download
    api, infos, failed = HfApi(), {}, {}
    for i, c in enumerate(names, 1):
        repo = allowed[c]
        have = man["datasets"].get(c)
        if have is not None and not a.force and not needs_more_shards(c, have, a.hourly_shards):
            print(f"[{i}/{len(names)}] {c}: in the manifest, skipped")
            continue
        print(f"[{i}/{len(names)}] {c} <- {repo}", flush=True)
        try:
            root.mkdir(parents=True, exist_ok=True)
            do_dataset(c, repo, man, root, a, infos, api, snapshot_download)
        except (Exception, SystemExit) as e:  # noqa: BLE001
            failed[c] = f"{type(e).__name__}: {e}"
            print(f"    FAILED: {failed[c]}", flush=True)
    done = man["datasets"]
    tot = sum(d["total_points"] for d in done.values())
    print(f"\nmanifest {mpath}: {len(done)} dataset(s), {tot:,} points ({tot / 1e9:.2f}B) as downloaded")
    if failed:
        print("FAILED:")
        for c, why in failed.items():
            print(f"  {c}: {why}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
