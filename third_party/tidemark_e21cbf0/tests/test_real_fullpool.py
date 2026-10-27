"""Tools of the full-weatherbench run: parallel conversion identical to serial, a separate shard folder with links, the disk / time estimate,
the downloader's --skip-stats, and the launcher's shard / plan / name options."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tidemark.real import allowlist as AL
from tidemark.real import convert as CV
from tidemark.real import estimate as ES
from tidemark.real import store as ST
from test_real_store import write_arrow

nan = float("nan")
ROOT = Path(__file__).parents[1]
bash = shutil.which("bash")


def rows(seed, n=6, lo=40, hi=90, nf=0.1):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        v = np.cumsum(rng.normal(size=int(rng.integers(lo, hi)))).tolist()
        for i in np.flatnonzero(rng.random(len(v)) < nf):
            v[i] = nan
        v[0], v[-1] = 1.0, 2.0
        out.append(v)
    return out


def test_parallel_conversion_is_identical_to_serial(tmp_path):
    files = []
    for k in range(5):
        f = tmp_path / f"train-{k:05d}-of-00005.arrow"
        write_arrow(f, rows(k))
        files.append(f)
    ST.convert_dataset("d", files, tmp_path / "serial", workers=1)
    ST.convert_dataset("d", files, tmp_path / "par", workers=3)
    for name in ("values.f32", "offsets.i64", "missing.i64"):
        assert (tmp_path / "serial" / name).read_bytes() == (tmp_path / "par" / name).read_bytes(), name
    a, b = (json.loads((tmp_path / x / "meta.json").read_text()) for x in ("serial", "par"))
    for k in ("n_series", "unique_points", "stored_points", "series_in", "dropped_short", "nan_points_in"):
        assert a[k] == b[k]
    assert b["workers"] == 3 and b["convert_seconds"] >= 0 and b["points_in_per_s"] > 0
    assert not list((tmp_path / "par").glob("part*"))                              # the parts are gone


def make_raw(root, hourly_files):
    d = root / "raw" / "chronos_datasets"
    (d / "dominick").mkdir(parents=True)
    write_arrow(d / "dominick" / "train-00000-of-00001.arrow", rows(1))
    per = {}
    for var, n in hourly_files.items():
        h = d / "weatherbench_hourly" / var
        h.mkdir(parents=True)
        for i in range(n):
            write_arrow(h / f"train-{i:05d}-of-{n:05d}.arrow", rows(10 + i))
            per[f"weatherbench_hourly/{var}/train-{i:05d}-of-{n:05d}.arrow"] = {"index": i, "size_bytes": 100, "total_points": 1000, "nan_fraction": 0.01}
    return per


def test_shards_dir_links_and_leaves_the_first_run_untouched(tmp_path):
    root = tmp_path / "real"
    per = make_raw(root, {"geopotential": 3})
    taken = {k: v for k, v in per.items() if k.endswith("00000-of-00003.arrow")}
    man = {"datasets": {"dominick": {"repo": AL.CHRONOS},
                        "weatherbench_hourly_geopotential": {"repo": AL.CHRONOS, "shards": {"per_shard": taken}}}}
    (root / "manifest.json").write_text(json.dumps(man))
    assert CV.main(["--root", str(root)]) == 0                                    # the first run: 1 of 3 shards
    first = json.loads((root / "shards" / "weatherbench_hourly_geopotential" / "meta.json").read_text())
    before = (root / "shards" / "weatherbench_hourly_geopotential" / "values.f32").read_bytes()
    man["datasets"]["weatherbench_hourly_geopotential"]["shards"]["per_shard"] = per          # the full download
    (root / "manifest.json").write_text(json.dumps(man))
    assert CV.main(["--root", str(root), "--shards-dir", "shards_full", "--only", "weatherbench_hourly_geopotential",
                    "--link-rest-from", "shards", "--workers", "2"]) == 0
    full = json.loads((root / "shards_full" / "weatherbench_hourly_geopotential" / "meta.json").read_text())
    assert full["files"] == 3 and full["stored_points"] > first["stored_points"] and full["workers"] == 2
    assert (root / "shards" / "weatherbench_hourly_geopotential" / "values.f32").read_bytes() == before          # untouched
    link = root / "shards_full" / "dominick"
    assert link.is_symlink() or (link / "meta.json").exists()                                                   # linked (or copied where symlinks fail)
    idx = json.loads((root / "shards_full" / "index.json").read_text())["datasets"]
    assert set(idx) == {"dominick", "weatherbench_hourly_geopotential"} and idx["dominick"]["total_points"] == \
        json.loads((root / "shards" / "index.json").read_text())["datasets"]["dominick"]["total_points"]
    # a linked dataset is never written through
    if link.is_symlink():
        assert CV.main(["--root", str(root), "--shards-dir", "shards_full", "--only", "dominick", "--force"]) == 2


def manifest_entry(have, total, sample_bytes=800_000_000, points=250_000_000, total_bytes=107 * 800_000_000, nan=0.02):
    per = {f"f{i}": {"index": i, "size_bytes": sample_bytes, "total_points": points, "nan_fraction": nan} for i in range(have)}
    return {"repo": AL.CHRONOS, "total_points": points * have, "shards": {"total": total, "per_shard": per, "taken": list(range(have)),
                                                                           "full_variable": {"total_bytes": total_bytes}}}


def test_estimate_reports_download_converted_size_and_fit(monkeypatch):
    man = {"datasets": {c: manifest_entry(12, 107) for c in AL.PRESSURE_HOURLY}}
    for c in AL.SURFACE_HOURLY:
        man["datasets"][c] = manifest_entry(9, 9, total_bytes=9 * 800_000_000)
    r = ES.variable_rows(man)
    assert len(r) == 11 and sum(not x["surface"] for x in r) == 8
    p = next(x for x in r if not x["surface"])
    assert p["shards_have"] == 12 and p["shards_total"] == 107
    assert p["est_points"] == pytest.approx(250_000_000 / 800_000_000 * p["raw_total"])                        # points per byte x total bytes
    assert p["converted_bytes"] == pytest.approx(p["est_points"] * 4 + p["est_points"] * 0.02 * 8)
    text = ES.report(r, free_bytes=10 * 2 ** 40, calib=None, workers=32)
    assert "FITS" in text and "DOES NOT FIT" not in text and "run with --calibrate" in text and "TOTAL (8 pressure-level variables)" in text
    small = ES.report(r, free_bytes=500 * 2 ** 30, calib=None, workers=32)
    assert "DOES NOT FIT" in small
    timed = ES.report(r, free_bytes=10 * 2 ** 40, calib={"seconds": 100.0, "points_in": 2_000_000, "file_bytes": 1}, workers=32)
    assert "conversion time (1 process" in timed and "32 workers" in timed


def load_dl():
    spec = importlib.util.spec_from_file_location("dl", ROOT / "scripts" / "real" / "download_chronos.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_skip_stats_records_sizes_only_and_keeps_the_counted_sample(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    from types import SimpleNamespace as NS
    dl = load_dl()
    sib = [NS(rfilename=f"weatherbench_hourly/geopotential/train-{i:05d}-of-00005.parquet", size=1000 + i) for i in range(5)]

    class Api:
        def dataset_info(self, repo, files_metadata=False):
            return NS(sha="rev", siblings=sib)

    def snap(repo_id, repo_type, revision, allow_patterns, local_dir, max_workers):
        for f in allow_patterns:
            p = Path(local_dir) / f
            p.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table({"timestamp": [[1, 2, 3]] * 4, "target": [[1.0, 2.0, nan]] * 4}), p)
    man = {"datasets": {}, "revisions": {}}
    c = "weatherbench_hourly_geopotential"
    args = type("A", (), dict(hourly_shards=2, force=False, stats_only=False, workers=1, skip_stats=False))()
    dl.do_dataset(c, AL.CHRONOS, man, tmp_path, args, {}, Api(), snap)                            # the counted sample: shards 0 and 4
    assert man["datasets"][c]["shards"]["taken"] == [0, 4] and man["datasets"][c]["n_series"] == 8
    args.hourly_shards, args.skip_stats = 5, True
    dl.do_dataset(c, AL.CHRONOS, man, tmp_path, args, {}, Api(), snap)                            # all five, new ones not counted
    e = man["datasets"][c]
    per = e["shards"]["per_shard"]
    assert e["shards"]["taken"] == [0, 1, 2, 3, 4] and e["files"] == 5
    assert "total_points" in per[next(k for k in per if k.endswith("00000-of-00005.parquet"))]
    assert "total_points" not in per[next(k for k in per if k.endswith("00002-of-00005.parquet"))] and per[next(k for k in per if k.endswith("00002-of-00005.parquet"))]["size_bytes"] > 0
    assert e["shards"]["counted_shards"] == 2 and e["n_series"] == 8                              # numbers describe the counted sample
    assert not dl.needs_more_shards(c, e, 5)


@pytest.mark.skipif(bash is None, reason="needs bash")
def test_launcher_passes_shards_plan_and_name(tmp_path):
    scratch = tmp_path / "scratch"
    root = scratch / "tidemark"
    (root / "repo" / "scripts" / "perlmutter").mkdir(parents=True)
    (root / "repo" / "scripts" / "perlmutter" / "env.sh").write_text(f'export ROOT="{root.as_posix()}"\n')
    script = ROOT / "scripts" / "perlmutter" / "real_train_interactive.sh"
    env = {**os.environ, "SCRATCH": scratch.as_posix(), "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"],
           "REAL_SHARDS": "shards_full", "REAL_PLAN": "/x/plan_full.json", "REAL_NAME": "real30b_full"}
    r = subprocess.run([bash, str(script), "--dry-run"], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "--shards-dir shards_full --plan /x/plan_full.json --name real30b_full" in r.stdout and "real_run/real30b_full" in r.stdout
    assert "shards_full/index.json" in r.stdout
    env2 = {k: v for k, v in env.items() if k != "REAL_PLAN"}                       # shards_full without REAL_PLAN -> plan_full.json
    r2 = subprocess.run([bash, str(script), "--dry-run"], capture_output=True, text=True, env=env2, timeout=120)
    assert "--plan " + (scratch / "tidemark" / "real" / "plan_full.json").as_posix() in r2.stdout.replace("\\", "/")
    env3 = {k: v for k, v in env2.items() if k != "REAL_SHARDS"}
    assert "plan.json" in subprocess.run([bash, str(script), "--dry-run"], capture_output=True, text=True, env=env3, timeout=120).stdout
