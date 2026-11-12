"""tfbench command line, Milestone A then B. Every step is resumable (done outputs are skipped unless --redo).

    python -m tideforge.tfbench.run select                    # A1, local or JarvisLabs (no network)
    python -m tideforge.tfbench.run download --jobs 6         # A2, JarvisLabs only
    python -m tideforge.tfbench.run refs                      # A2, the three reference benchmarks
    python -m tideforge.tfbench.run canon --jobs 24           # A3
    python -m tideforge.tfbench.run audit --jobs 24           # A5 (fingerprints the references once)
    python -m tideforge.tfbench.run report                    # A4 -> devdata/tfbench/BUILD_REPORT_A.md
    python -m tideforge.tfbench.run split                     # B2 splits.json (written once, append-only after)
    python -m tideforge.tfbench.run crossdup --jobs 24        # B2 tuning series that copy a locked series
    python -m tideforge.tfbench.run tasks --jobs 24           # B1 per-dataset tasks
    python -m tideforge.tfbench.run package                   # B3 release folder $TFB_DATA/bench/<version>/
    python -m tideforge.tfbench.run report_b --jobs 24        # seasonal naive on every task, BUILD_REPORT.md
    python -m tideforge.tfbench.run upload [--dry-run]        # B3 to $TF_BENCH_REPO (private), tag <version>
    python -m tideforge.tfbench.run all                       # A then B then upload
    python -m tideforge.tfbench.run all --smoke               # the whole chain on the smoke datasets, upload dry-run
Options: --only <dataset_id or repo id> (repeatable), --redo, --max-gb 100, --dry-run.
Logs: devdata/tfbench/logs/<step>-<UTC time>.log. Smoke keeps its own splits, tasks and release (suffix _smoke).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

from tideforge.tfbench import config as C
from tideforge.tfbench.rules import slug

log = logging.getLogger("tfbench")
STEPS_A = ["select", "download", "refs", "canon", "audit", "report"]
STEPS_B = ["split", "crossdup", "tasks", "package", "report_b", "upload"]
STEPS = STEPS_A + STEPS_B


def setup_logging(paths: C.Paths, step: str) -> Path:
    paths.logs.mkdir(parents=True, exist_ok=True)
    f = paths.logs / f"{step}-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%S}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    for h in (logging.FileHandler(f, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        root.addHandler(h)
    return f


def read_profile(catalog_only: bool = False) -> pd.DataFrame:
    p = pd.read_csv(C.profile_path(), low_memory=False)
    return p[p["row_kind"] == "catalog"] if catalog_only else p


def read_overrides(paths: C.Paths) -> dict:
    if not paths.overrides.exists():
        return {}
    return yaml.safe_load(paths.overrides.read_text(encoding="utf-8")) or {}


def selected(paths: C.Paths, only: list[str] | None) -> pd.DataFrame:
    sel = pd.read_csv(paths.selection)
    sel = sel[sel["keep"].astype(str).str.lower() == "true"]
    if only:
        sel = sel[sel["dataset_id"].isin(only) | sel["repo_id"].isin(only)]
    return sel


# ---- steps --------------------------------------------------------------------------------------------------------------
def step_select(a, paths, s):
    from tideforge.tfbench.select import select
    df = select(read_profile(), s)
    paths.small.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.selection, index=False)
    k = df["keep"].sum()
    log.info("selection: %d of %d catalog rows kept, %.1f GB in scope -> %s", k, len(df),
             df.loc[df["keep"], "scope_bytes"].sum() / 1e9, paths.selection)


def step_download(a, paths, s):
    from tideforge.tfbench import download
    files = pd.read_csv(C.files_path(), low_memory=False)
    res = download.run(pd.read_csv(paths.selection), files, paths, s, a.only, a.jobs)
    log.info("repos: %s", res["status"].value_counts().to_dict() if not res.empty else {})


def step_refs(a, paths, s):
    from tideforge.tfbench import download
    download.download_refs(paths, s)


def _canon_one(row: dict, files: list, ov: dict, s: C.Settings, data_root: str) -> dict:
    from tideforge.tfbench.canon import Fail, build_dataset
    paths = C.Paths(data=Path(data_root))
    out = paths.canon / row["dataset_id"]
    work = paths.data / "tmp" / row["dataset_id"]
    st = {"dataset_id": row["dataset_id"], "repo_id": row["repo_id"], "status": "ok", "reason": ""}
    try:
        m = build_dataset(row, [(Path(f[0]), f[1], f[2]) for f in files], ov, s, out, work, paths.refs)
        st.update(n_series=m["n_series"], n_points=m["n_points"], frequency=m["frequency"])
    except Fail as e:
        st.update(status="not built", reason=str(e)[:500])
    except Exception as e:  # one dataset failing never stops the run
        st.update(status="error", reason=f"{type(e).__name__}: {str(e)[:300]}", traceback=traceback.format_exc()[-2000:])
    finally:
        shutil.rmtree(work, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "status.json").write_text(json.dumps(st, indent=1, default=str), encoding="utf-8")
    return st


def step_canon(a, paths, s):
    sel = selected(paths, a.only)
    prof = read_profile(catalog_only=True).set_index("catalog_row")
    man = pd.read_csv(paths.data / "manifest.csv")
    man = man[man["status"] == "ok"]
    ov_all = read_overrides(paths)
    jobs = []
    for r in sel.to_dict("records"):
        did = r["dataset_id"]
        if (paths.canon / did / "status.json").exists() and not a.redo:
            continue
        p = prof.loc[int(r["catalog_row"])].to_dict()
        row = {**{k: (None if v != v else v) for k, v in p.items()}, **r}
        mine = man[man["dataset_ids"].astype(str).str.split(";").apply(lambda x: did in x)]
        files = [(str(paths.raw / slug(m["repo"]) / m["path"]), m["path"], int(m["bytes"])) for m in mine.to_dict("records")]
        jobs.append((row, files, ov_all.get(did, {}) or {}))
    log.info("canon: %d datasets to build", len(jobs))
    with ProcessPoolExecutor(a.jobs or 8) as ex:
        futs = {ex.submit(_canon_one, row, files, ov, s, str(paths.data)): row["dataset_id"] for row, files, ov in jobs}
        for f in as_completed(futs):
            st = f.result()
            log.info("%s: %s %s", st["dataset_id"], st["status"], st.get("reason") or
                     f"{st.get('n_series')} series, {st.get('n_points')} points, {st.get('frequency')}")


def _audit_one(canon_dir: str, db_dir: str, out: str, s: C.Settings) -> str:
    from tideforge.devdata.audit import DB
    from tideforge.tfbench.audit import audit_dataset
    try:
        res = audit_dataset(Path(canon_dir), DB.load(Path(db_dir)), s)
    except Exception as e:
        return f"{Path(canon_dir).name}: error {type(e).__name__}: {e}"
    Path(out).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    return f"{res['dataset_id']}: share {res['share_after_drops']:.3%}, dropped {res['series_dropped']}, flagged {res['flagged']}"


def step_audit(a, paths, s):
    from tideforge.tfbench import audit as TA
    db_dir = paths.audit / "refs_db"
    if not (db_dir / "meta.json").exists() or a.redo_refs:
        benches = {"fev": slug(s.refs[0]), "GIFT-Eval": slug(s.refs[1]), "TIME": slug(s.refs[2])}
        TA.build_refs_db(paths.refs, db_dir, paths.data / "tmp" / "refs", benches)
    paths.audit.mkdir(parents=True, exist_ok=True)
    todo = []
    for d in sorted(paths.canon.glob("*/data.parquet")):
        did = d.parent.name
        if a.only and did not in a.only:
            continue
        out = paths.audit / f"{did}.json"
        if out.exists() and not a.redo:
            continue
        todo.append((str(d.parent), str(out)))
    log.info("audit: %d datasets", len(todo))
    with ProcessPoolExecutor(a.jobs or 8) as ex:
        for f in as_completed([ex.submit(_audit_one, c, str(db_dir), o, s) for c, o in todo]):
            log.info(f.result())
    df = TA.summarise(paths.audit)
    if not df.empty:
        log.info("audit: %d flagged, %d series dropped", int(df["flagged"].sum()), int(df["series_dropped"].sum()))


def step_report(a, paths, s):
    from tideforge.tfbench import report
    out = paths.small / "BUILD_REPORT_A.md"
    report.write(out, pd.read_csv(paths.selection), paths.canon, paths.audit / "audit.csv")
    log.info("report -> %s", out)


# ---- Milestone B --------------------------------------------------------------------------------------------------
def _sfx(a) -> str:
    return "_smoke" if getattr(a, "smoke", False) else ""


def splits_path(a, paths) -> Path:
    return (paths.data / "splits_smoke.json") if _sfx(a) else (paths.small / "splits.json")


def task_root(a, paths) -> Path:
    return paths.data / f"tasks{_sfx(a)}"


def release(a, paths, s) -> Path:
    return paths.data / "bench" / f"{s.version}{_sfx(a)}"


def passed(a, paths) -> dict:
    """Datasets built in A and passed by the audit -> {dataset_id: meta}."""
    out = {}
    for d in sorted(paths.canon.glob("*/data.parquet")):
        did = d.parent.name
        if a.only and did not in a.only:
            continue
        au = paths.audit / f"{did}.json"
        if not au.exists():
            log.warning("%s: no audit result, not used", did)
            continue
        if json.loads(au.read_text(encoding="utf-8"))["flagged"]:
            continue
        out[did] = json.loads((d.parent / "meta.json").read_text(encoding="utf-8"))
    return out


def step_split(a, paths, s):
    from tideforge.tfbench import split as SP
    ds = passed(a, paths)
    sp = SP.load_or_create(splits_path(a, paths), {d: m.get("domain") or "unknown" for d, m in ds.items()})
    n = pd.Series([v["split"] for d, v in sp["datasets"].items() if d in ds]).value_counts().to_dict()
    log.info("split: %s -> %s", n, splits_path(a, paths))


def _clean(paths, did):
    from tideforge.tfbench.audit import load_clean
    return load_clean(paths.canon / did, paths.audit)


def _crossdup_one(did, data_root, db_dir, share, out):
    from tideforge.devdata.audit import DB
    from tideforge.tfbench import split as SP
    paths = C.Paths(data=Path(data_root))
    meta = json.loads((paths.canon / did / "meta.json").read_text(encoding="utf-8"))
    rows = SP.crossdups(did, _clean(paths, did), meta["targets"], DB.load(Path(db_dir)), share)
    Path(out).write_text(json.dumps(rows), encoding="utf-8")
    return did, len(rows)


def step_crossdup(a, paths, s):
    import hashlib
    from tideforge.tfbench import split as SP
    sp = json.loads(splits_path(a, paths).read_text(encoding="utf-8"))["datasets"]
    ds = passed(a, paths)
    locked = sorted(d for d in ds if sp[d]["split"] == "locked")
    tuning = sorted(d for d in ds if sp[d]["split"] == "tuning")
    root = paths.data / f"crossdup{_sfx(a)}"
    key = hashlib.sha256(";".join(locked).encode()).hexdigest()[:16]
    db_dir = root / f"locked_db_{key}"
    if not (db_dir / "meta.json").exists():          # a new locked set -> new fingerprints and fresh matches
        shutil.rmtree(root, ignore_errors=True)
        db = SP.build_locked_db([(d, _clean(paths, d), ds[d]["targets"]) for d in locked])
        db.save(db_dir)
    todo = [d for d in tuning if a.redo or not (root / f"{d}.json").exists()]
    log.info("crossdup: %d locked datasets, %d tuning datasets to check", len(locked), len(todo))
    with ProcessPoolExecutor(a.jobs or 8) as ex:
        futs = [ex.submit(_crossdup_one, d, str(paths.data), str(db_dir), s.crossdup_share, str(root / f"{d}.json"))
                for d in todo]
        for f in as_completed(futs):
            try:
                d, n = f.result()
                if n:
                    log.info("%s: %d series copy a locked series", d, n)
            except Exception as e:
                log.error("crossdup failed: %s: %s", type(e).__name__, e)


def crossdup_table(a, paths) -> pd.DataFrame:
    rows = []
    for f in sorted((paths.data / f"crossdup{_sfx(a)}").glob("*.json")):
        rows += json.loads(f.read_text(encoding="utf-8"))
    return pd.DataFrame(rows, columns=["dataset_id", "item_id", "share", "matched"])


def _tasks_one(did, data_root, out, drops, ov, s):
    from tideforge.tfbench import tasks as TK
    paths = C.Paths(data=Path(data_root))
    try:
        meta = json.loads((paths.canon / did / "meta.json").read_text(encoding="utf-8"))
        data = _clean(paths, did)
        data = data[~data["item_id"].astype(str).isin(set(drops))]
        if data.empty:
            raise ValueError("no series left after the audit and cross-dataset drops")
        t = TK.build(data, meta, ov, s, Path(out))
        return did, len(t), ""
    except Exception as e:  # one dataset failing never stops the run
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "error.json").write_text(json.dumps({"dataset_id": did, "error": f"{type(e).__name__}: {e}",
                                                          "traceback": traceback.format_exc()[-2000:]}), encoding="utf-8")
        return did, 0, f"{type(e).__name__}: {str(e)[:300]}"


def step_tasks(a, paths, s):
    ds = passed(a, paths)
    sp = json.loads(splits_path(a, paths).read_text(encoding="utf-8"))["datasets"]
    cd = crossdup_table(a, paths)
    ov_all = read_overrides(paths)
    root = task_root(a, paths)
    todo = [d for d in ds if d in sp and (a.redo or not ((root / d / "tasks.json").exists() or (root / d / "error.json").exists()))]
    if a.redo:
        for d in todo:
            shutil.rmtree(root / d, ignore_errors=True)
    log.info("tasks: %d datasets", len(todo))
    n_fail = 0
    with ProcessPoolExecutor(a.jobs or 8) as ex:
        futs = [ex.submit(_tasks_one, d, str(paths.data), str(root / d), list(cd.loc[cd["dataset_id"] == d, "item_id"]),
                          ov_all.get(d, {}) or {}, s) for d in todo]
        for f in as_completed(futs):
            d, n, err = f.result()
            if err:
                n_fail += 1
                log.error("%s: failed: %s", d, err)
            else:
                log.info("%s: %d tasks", d, n)
    log.info("tasks: %d failed of %d", n_fail, len(todo))


def failures(a, paths) -> pd.DataFrame:
    rows = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(task_root(a, paths).glob("*/error.json"))]
    cols = ["dataset_id", "error"]
    return pd.DataFrame(rows)[cols] if rows else pd.DataFrame(columns=cols)


def step_package(a, paths, s):
    import os
    from tideforge.tfbench import package as PK
    sp = json.loads(splits_path(a, paths).read_text(encoding="utf-8"))
    rel = release(a, paths, s)
    man = PK.build(task_root(a, paths), paths.canon, sp, rel, os.environ.get("TF_BENCH_REPO", "TF_BENCH_REPO"), s)
    bad = PK.check_manifest(rel)
    for b in bad:
        log.error("manifest: %s", b)
    log.info("package: %d tasks -> %s (%s)", len(man), rel, "manifest ok" if not bad else f"{len(bad)} problems")


def _baseline_one(path, task):
    from tideforge.tfbench.baseline import score
    try:
        return {"task": task["task_name"], **score(path, task)}
    except Exception as e:
        return {"task": task["task_name"], "MASE": float("nan"), "WQL": float("nan"), "SQL": float("nan"),
                "series_windows": 0, "flag": f"error {type(e).__name__}: {str(e)[:200]}"}


def step_report_b(a, paths, s):
    from tideforge.tfbench import report, report_b
    rel = release(a, paths, s)
    jobs = []
    for sp, base in (("tuning", rel / "tasks"), ("locked", rel / "locked" / "tasks")):
        for t in (yaml.safe_load((base / "full.yaml").read_text(encoding="utf-8")) or {}).get("tasks", []):
            d = rel / ("locked/data" if sp == "locked" else "data") / t["task_name"].split("/")[0] / "data.parquet"
            jobs.append((str(d), t))
    with ProcessPoolExecutor(a.jobs or 8) as ex:
        res = list(ex.map(_baseline_one, *zip(*jobs))) if jobs else []
    bl = pd.DataFrame(res, columns=["task", "MASE", "WQL", "SQL", "series_windows", "flag"])
    bl.to_csv(rel / "baseline.csv", index=False)
    ra = report.write(paths.data / f"BUILD_REPORT_A{_sfx(a)}.md", pd.read_csv(paths.selection), paths.canon,
                      paths.audit / "audit.csv")
    report_b.write(rel / "BUILD_REPORT.md", rel, ra, bl, task_root(a, paths), crossdup_table(a, paths), failures(a, paths))
    shutil.copyfile(rel / "BUILD_REPORT.md", paths.small / f"BUILD_REPORT{_sfx(a)}.md")
    flagged = int((bl["flag"].fillna("") != "").sum())
    log.info("report_b: %d tasks scored, %d flagged -> %s", len(bl), flagged, rel / "BUILD_REPORT.md")


def step_upload(a, paths, s):
    import os
    from tideforge.tfbench import upload as UP
    dry = a.dry_run or a.smoke
    try:
        UP.upload(release(a, paths, s), os.environ.get("TF_BENCH_REPO", ""), s.version, a.max_gb, dry)
    except UP.Refused as e:
        log.error("upload refused: %s", e)
        raise SystemExit(3)


def pick_smoke(sel: pd.DataFrame, prof: pd.DataFrame) -> list[tuple[str, str]]:
    """About 8 small selected datasets covering every layout, a zip, a keep_filtered row, a monthly or yearly dataset and
    one with covariates. -> [(dataset_id, why)]; the smallest scope that fits each need."""
    p = prof[prof["row_kind"] == "catalog"].set_index("catalog_row") if "row_kind" in prof else prof.set_index("catalog_row")
    s = sel[sel["keep"].astype(str).str.lower() == "true"].copy()
    cr = s["catalog_row"].astype(int)
    s["formats"] = cr.map(p["scope_formats"]).fillna("").astype(str)
    s["band"] = cr.map(p["freq_band"]).fillna("").astype(str)
    s["cov"] = cr.map(p["has_covariates"]).fillna("").astype(str)
    s = s.sort_values("scope_bytes")
    needs = [("layout long", s["layout"] == "long"), ("layout wide", s["layout"] == "wide"),
             ("layout nested", s["layout"] == "nested"), ("layout events", s["layout"] == "events"),
             ("zip archive", s["formats"].str.contains(r"(?:^|; )zip:")),
             ("keep_filtered", s["status"] == "keep_filtered"),
             ("monthly or yearly", s["band"].isin(["monthly", "quarterly / yearly"])),
             ("covariates", (s["cov"] == "yes") & s["layout"].isin(["long", "wide"])),
             ("csv file", s["formats"].str.contains(r"(?:^|; )csv:") & ~s["formats"].str.contains("parquet")),
             ("hourly or finer", s["band"].isin(["hourly", "minute", "sub-minute"]))]
    out: dict[str, list[str]] = {}
    for why, m in needs:
        hit = s[m]
        if hit.empty:
            log.warning("smoke: no selected dataset for %s", why)
            continue
        have = [d for d in hit["dataset_id"] if d in out]       # reuse a dataset already picked when it fits
        did = have[0] if have else hit["dataset_id"].iloc[0]
        out.setdefault(did, []).append(why)
    return [(d, ", ".join(w)) for d, w in out.items()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=STEPS + ["all"])
    ap.add_argument("--only", action="append")
    ap.add_argument("--jobs", type=int)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--redo-refs", action="store_true", help="rebuild the reference fingerprints")
    ap.add_argument("--smoke", action="store_true", help="run every step on ~8 datasets that cover the cases (ids printed)")
    ap.add_argument("--dry-run", action="store_true", help="upload: list files and size only")
    ap.add_argument("--max-gb", type=float, default=100.0)
    a = ap.parse_args(argv)
    paths, s = C.Paths(), C.Settings()
    steps = STEPS if a.step == "all" or a.smoke else [a.step]
    if a.smoke:
        setup_logging(paths, "smoke")
        step_select(a, paths, s)
        picks = pick_smoke(pd.read_csv(paths.selection), read_profile())
        print("smoke datasets:")
        for d, why in picks:
            print(f"  {d}  ({why})")
        a.only = [d for d, _ in picks]
        steps = [x for x in STEPS if x != "select"]
    for step in steps:
        f = setup_logging(paths, step)
        log.info("step %s (data %s, log %s)", step, paths.data, f)
        globals()[f"step_{step}"](a, paths, s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
