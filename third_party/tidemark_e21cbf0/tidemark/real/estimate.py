"""Disk, download and conversion-time estimate for taking ALL shards of the pressure-level weatherbench_hourly variables.

  python -m tidemark.real.estimate [--root $SCRATCH/tidemark/real] [--calibrate] [--workers 32]

From manifest.json (the shards already counted give points per byte), the Hub sizes (shards.full_variable.total_bytes) and the free space under
--root: how much has still to be downloaded, how big the converted float32 shards will be, and whether both fit. --calibrate converts ONE
shard that is already on disk (single process, into a temporary folder that is removed) and extrapolates the conversion time from it.
Estimates are extrapolations from the shards counted so far; the output says which numbers they rest on. Nothing is downloaded or kept.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from tidemark.real import allowlist as AL
from tidemark.real.allowlist import SURFACE_HOURLY

GIB = 2 ** 30


def variable_rows(man: dict) -> list[dict]:
    """One row per confirmed hourly weatherbench variable found in the manifest."""
    rows = []
    for c in AL.HOURLY_CONFIRMED:
        e = man["datasets"].get(c)
        if not e or "shards" not in e:
            continue
        sh = e["shards"]
        per = sh["per_shard"]
        counted = [v for v in per.values() if "total_points" in v]
        fv = sh["full_variable"]
        raw_total = fv["total_bytes"]
        have_bytes = sum(v["size_bytes"] for v in per.values())
        full = c in SURFACE_HOURLY or len(per) >= sh["total"]
        c_bytes = sum(v["size_bytes"] for v in counted)
        pts_per_byte = sum(v["total_points"] for v in counted) / c_bytes if c_bytes else 0.0
        nan_frac = (sum(v["nan_fraction"] * v["total_points"] for v in counted) / max(sum(v["total_points"] for v in counted), 1)) if counted else 0.0
        est_points = e["total_points"] * (raw_total / max(have_bytes, 1)) if not full else float(e["total_points"])
        est_points = pts_per_byte * raw_total if counted else est_points
        rows.append(dict(name=c, surface=c in SURFACE_HOURLY, shards_have=len(per), shards_total=sh["total"], raw_total=raw_total,
                         raw_have=have_bytes, counted=len(counted), est_points=est_points, nan_frac=nan_frac,
                         converted_bytes=est_points * 4 + est_points * nan_frac * 8))
    return rows


def calibrate(root: Path, man: dict, log=print) -> dict | None:
    """Convert one shard already on disk with one process; returns seconds, input points, stored bytes."""
    from tidemark.real import store
    for c in AL.HOURLY_CONFIRMED:
        e = man["datasets"].get(c)
        if not e or "shards" not in e:
            continue
        for f in sorted(e["shards"]["per_shard"]):
            p = root / "raw" / e["repo"].split("/")[-1] / f
            if p.exists():
                tmp = root / "_calibrate"
                try:
                    m = store.convert_dataset("calib", [p], tmp, workers=1)
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)
                log(f"calibration shard {f}: {m['convert_seconds']:.0f} s for {m['points_in']:,} input points ({m['points_in_per_s'] / 1e6:.2f}M/s), "
                    f"{m['stored_points'] * 4 / GIB:.2f} GiB written")
                return {"seconds": m["convert_seconds"], "points_in": m["points_in"], "file_bytes": p.stat().st_size}
    return None


def report(rows: list[dict], free_bytes: int, calib: dict | None, workers: int, existing_converted_bytes: float = 0.0) -> str:
    press = [r for r in rows if not r["surface"]]
    remaining = sum(r["raw_total"] - r["raw_have"] for r in press)
    conv = sum(r["converted_bytes"] for r in press)
    raw_press = sum(r["raw_total"] for r in press)
    L = [f"{'variable':58s} {'shards':>9s} {'raw GiB':>8s} {'to fetch':>9s} {'est. points':>13s} {'NaN':>6s} {'float32 GiB':>11s}"]
    for r in press:
        L.append(f"{r['name']:58s} {r['shards_have']:>4d}/{r['shards_total']:<4d} {r['raw_total'] / GIB:>8.1f} "
                 f"{(r['raw_total'] - r['raw_have']) / GIB:>9.1f} {r['est_points']:>13,.0f} {r['nan_frac']:>6.1%} {r['converted_bytes'] / GIB:>11.1f}")
    L.append(f"{'TOTAL (8 pressure-level variables)':58s} {'':>9s} {raw_press / GIB:>8.1f} {remaining / GIB:>9.1f} "
             f"{sum(r['est_points'] for r in press):>13,.0f} {'':>6s} {conv / GIB:>11.1f}")
    basis = sorted({r["counted"] for r in press})
    L.append(f"(points per variable are extrapolated from {basis} counted shard(s) per variable, scaled by parquet bytes)")
    keep = remaining + conv
    biggest = max((r["raw_total"] for r in press), default=0)
    interleaved = conv + biggest
    L += ["", f"free on the target file system: {free_bytes / GIB:,.0f} GiB",
          f"  A. download everything, keep the raw files, convert:   +{keep / GIB:,.0f} GiB   -> {'FITS' if keep * 1.1 < free_bytes else 'DOES NOT FIT'}"
          f" (with 10% headroom: {keep * 1.1 / GIB:,.0f} GiB)",
          f"  B. per variable: download, convert, delete its raw:    peak +{interleaved / GIB:,.0f} GiB  -> "
          f"{'FITS' if interleaved * 1.1 < free_bytes else 'DOES NOT FIT'}   (deleting raw data is a step YOU choose; nothing here deletes it)",
          f"  (the 12-shard converted data of the first run stays where it is: about {existing_converted_bytes / GIB:,.0f} GiB; your NERSC quota may "
          "be smaller than the free space of the file system: check it with `myquota`)"]
    pts = sum(r["est_points"] for r in press)
    if calib:
        rate = calib["points_in"] / max(calib["seconds"], 1e-9)
        L.append(f"conversion time (1 process {pts / rate / 3600:.1f} h; ideal scaling, the shared file system may limit it): "
                 + ", ".join(f"{w} workers {pts / rate / w / 3600:.1f} h" for w in sorted({4, 16, 32, 64, workers})))
    else:
        L.append("conversion time: run with --calibrate (converts one shard on one process and extrapolates)")
    L.append("download time: " + ", ".join(f"{mb} MB/s {remaining / (mb * 1e6) / 3600:.1f} h" for mb in (50, 100, 200, 400)) + "  (measure your rate first)")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(Path(os.environ.get("REAL_ROOT") or Path(os.environ.get("SCRATCH", ".")) / "tidemark" / "real")))
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args(argv)
    root = Path(a.root)
    man = json.loads((root / "manifest.json").read_text())
    rows = variable_rows(man)
    calib = calibrate(root, man) if a.calibrate else None
    existing = 0.0
    idx = root / "shards" / "index.json"
    if idx.exists():
        existing = sum(v.get("stored_points", v["total_points"]) for v in json.loads(idx.read_text())["datasets"].values()) * 4.0
    print(report(rows, shutil.disk_usage(root).free, calib, a.workers, existing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
