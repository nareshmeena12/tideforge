"""tfbench Milestone A command line. Every step is resumable (done outputs are skipped unless --redo).

    python -m tideforge.tfbench.run select                    # A1, local or JarvisLabs (no network)
    python -m tideforge.tfbench.run download --jobs 6         # A2, JarvisLabs only
    python -m tideforge.tfbench.run refs                      # A2, the three reference benchmarks
    python -m tideforge.tfbench.run canon --jobs 24           # A3
    python -m tideforge.tfbench.run audit --jobs 24           # A5 (fingerprints the references once)
    python -m tideforge.tfbench.run report                    # A4 -> devdata/tfbench/BUILD_REPORT_A.md
    python -m tideforge.tfbench.run all                       # the above in order
Options: --only <dataset_id or repo id> (repeatable), --redo. Logs: devdata/tfbench/logs/<step>-<UTC time>.log
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
STEPS = ["select", "download", "refs", "canon", "audit", "report"]


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


def read_profile() -> pd.DataFrame:
    return pd.read_csv(C.profile_path(), low_memory=False)


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
        m = build_dataset(row, [(Path(f[0]), f[1], f[2]) for f in files], ov, s, out, work)
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
    prof = read_profile().set_index("catalog_row")
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=STEPS + ["all"])
    ap.add_argument("--only", action="append")
    ap.add_argument("--jobs", type=int)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--redo-refs", action="store_true", help="rebuild the reference fingerprints")
    a = ap.parse_args(argv)
    paths, s = C.Paths(), C.Settings()
    for step in (STEPS if a.step == "all" else [a.step]):
        f = setup_logging(paths, step)
        log.info("step %s (data %s, log %s)", step, paths.data, f)
        globals()[f"step_{step}"](a, paths, s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
