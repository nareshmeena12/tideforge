"""Convert the downloaded datasets (manifest.json + raw files) into memmap shards, resumable, and write <shards>/index.json.

  python -m tidemark.real.convert [--root $SCRATCH/tidemark/real] [--only a,b] [--force] [--workers 32]
                                  [--shards-dir shards_full] [--link-rest-from shards]

Reads <root>/manifest.json (written by scripts/real/download_chronos.py). A dataset is converted from the files the manifest names:
all files of its Hub folder, or, for the weatherbench_hourly variables, the shards recorded in the manifest. Only allow-listed datasets
are converted (tidemark/real/allowlist.py). Missing values are forward-filled with their positions kept; see tidemark/real/store.py. A
dataset whose <shards>/<name>/meta.json exists is skipped. Prints valid points, NaN share, dropped series and the time of each dataset.

--workers N converts the files of a dataset in N processes (the result is identical to one process).
--shards-dir NAME writes to <root>/NAME instead of <root>/shards, which stays untouched (the full-weatherbench run uses shards_full so the
12-shard run real30b remains reproducible). --link-rest-from DIR then links every dataset of <root>/DIR that is not converted here
(symlinks), so the new folder is a complete set: only the datasets in --only differ.
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
from tidemark.real import store


def dataset_files(root: Path, entry: dict, name: str) -> list[Path]:
    local = root / "raw" / entry["repo"].split("/")[-1]
    if AL.is_hourly(name):
        files = sorted((entry.get("shards") or {}).get("per_shard", {}))
        return [local / f for f in files]
    return sorted(p for p in (local / AL.hub_prefix(name)).rglob("*") if p.suffix in (".arrow", ".parquet"))


def link_rest(root: Path, src: str, dst: str, skip: set[str]) -> list[str]:
    """Symlink every converted dataset of <root>/src into <root>/dst unless it is in `skip` or already there."""
    linked = []
    for m in sorted((root / src).glob("*/meta.json")):
        name = m.parent.name
        target = root / dst / name
        if name in skip or target.exists() or target.is_symlink():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(m.parent.resolve(), target, target_is_directory=True)
        except (OSError, NotImplementedError):                       # no symlinks (Windows tests): copy only if it is small
            size = sum(f.stat().st_size for f in m.parent.rglob("*") if f.is_file())
            if size > 2 ** 30:
                raise SystemExit(f"cannot symlink {target} -> {m.parent} and it is too big to copy ({size / 2 ** 30:.0f} GiB)")
            shutil.copytree(m.parent, target)
        linked.append(name)
    return linked


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(Path(os.environ.get("REAL_ROOT") or Path(os.environ.get("SCRATCH", ".")) / "tidemark" / "real")))
    ap.add_argument("--only", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--min-valid", type=int, default=store.MIN_LEN, help="drop series with fewer valid points (default 33)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--shards-dir", default="shards")
    ap.add_argument("--link-rest-from", default=None, metavar="DIR", help="link the converted datasets of <root>/DIR that are not converted here")
    a = ap.parse_args(argv)
    root = Path(a.root)
    man = json.loads((root / "manifest.json").read_text())
    allowed = AL.allowed()
    names = [n.strip() for n in a.only.split(",")] if a.only else [n for n in man["datasets"] if n in allowed]
    for n in names:
        AL.check(n, man["datasets"].get(n, {}).get("repo", "?"))
    shard_root = root / a.shards_dir
    t_all = time.time()
    for i, n in enumerate(names, 1):
        d = shard_root / n
        if (d / "meta.json").exists() and not a.force:
            print(f"[{i}/{len(names)}] {n}: converted, skipped")
            continue
        if d.is_symlink():
            print(f"[{i}/{len(names)}] {n}: is a link to another shard folder; refusing to write through it (remove the link first)")
            return 2
        files = dataset_files(root, man["datasets"][n], n)
        if not files:
            print(f"[{i}/{len(names)}] {n}: no files, skipped")
            continue
        print(f"[{i}/{len(names)}] {n}: {len(files)} file(s), {a.workers} worker(s)", flush=True)
        m = store.convert_dataset(n, files, d, a.min_valid, {"source_revision": man["datasets"][n].get("revision")}, workers=a.workers,
                                  log=print if len(files) > 4 else None)
        print(f"    {m['n_series']:,} series, {m['unique_points']:,} valid points, NaN {m['nan_fraction_stored']:.1%} of stored, "
              f"dropped {m['dropped_short']:,} short / {m['dropped_empty']:,} empty of {m['series_in']:,}; {m['convert_seconds']:.0f} s "
              f"({m['points_in_per_s'] / 1e6:.1f}M input points/s)", flush=True)
    if a.link_rest_from:
        linked = link_rest(root, a.link_rest_from, a.shards_dir, set(names))
        print(f"linked {len(linked)} dataset(s) from {a.link_rest_from}/ into {a.shards_dir}/: {', '.join(linked)}")
    idx = store.write_index(shard_root)
    tot = sum(v["total_points"] for v in idx["datasets"].values())
    print(f"\nindex {shard_root / 'index.json'}: {len(idx['datasets'])} datasets, {tot:,} valid points ({tot / 1e9:.2f}B); "
          f"{(time.time() - t_all) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
