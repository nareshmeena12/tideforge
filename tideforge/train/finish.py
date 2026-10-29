"""End of a 5M run: the fast dev score, the probe suite and one results row. Also runs on its own, from a saved checkpoint.

    python -m tideforge.train.finish --ckpt <run dir>/final.pt [--dev $TF_DATA/dev --version dev-v1]
                                     [--probes $TF_DATA/probes] [--results $TF_ROOT/results/results.csv] [--device cuda]

dev     devscore score (tuning role, fast set) with the M5 adapter -> <run dir>/dev/. If the dev version is not built yet the
        row says "dev set missing" and nothing is scored; run this command again later with the same checkpoint.
dev cov devscore score --cov (the dev-covariate set, groups ON and hidden; owner 2026-10-29) -> <run dir>/dev/: dev_cov_MASE /
        dev_cov_CRPS (ON) and dev_cov_gain = 1 - ON / hidden (MASE; > 0: the covariates help).
probes  tideforge.probes run on the built probe sets ($TF_DATA/probes) -> <run dir>/probes/ ("probes missing" if not built).
row     one line appended to results.csv (never committed): run id, experiment, seed, code, steps, points, the final holdout loss
        and the dev / probe status and headline numbers. The holdout loss is a TRAINING MONITOR ONLY: the holdout is drawn from
        B0's own generator (ft3's clean corpus), so it favours B0's distribution and cannot compare generator configs.
        Decisions use the dev score and the probes (EXPERIMENTS.md rule 5).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path

ROW = ["date", "run_id", "exp", "seed", "code", "steps", "points", "holdout_final", "holdout_note", "dev", "dev_MASE",
       "dev_CRPS", "probes", "ckpt", "dev_cov_MASE", "dev_cov_CRPS", "dev_cov_gain", "run_dir"]   # the last 4: 2026-10-29
HOLDOUT_NOTE = "training monitor only (B0's generator)"


def _env(name: str) -> str | None:
    v = os.environ.get(name)
    return v if v and v != "FILL_IN" else None


def dev_score(ckpt: Path, run_dir: Path, dev: str | None, version: str, device: str, log=print) -> dict:
    if not dev or not (Path(dev) / version / "manifest.json").exists():
        log(f"dev set missing ({dev or '$TF_DATA unset'}/{version}): dev score skipped; rerun `finish --ckpt {ckpt}` later")
        return dict(dev="dev set missing", dev_MASE="", dev_CRPS="")
    from tideforge.eval import devscore as DS
    out = run_dir / "dev"
    DS.main(["score", "--dev", str(dev), "--version", version, "--set", "fast", "--model", "m5", "--ckpt", str(ckpt),
             "--out", str(out), "--device", device])
    run, _ = DS.load_run(DS.run_dir(out, version, "tuning", "fast", False, "m5"))
    o = run["scores"]["main"]["overall"]
    DS.main(["score", "--dev", str(dev), "--version", version, "--cov", "--model", "m5", "--ckpt", str(ckpt),
             "--out", str(out), "--device", device])
    cov, _ = DS.load_run(DS.run_dir(out, version, "tuning", "fast", True, "m5"))
    c = cov["scores"]["on"]["overall"]
    return dict(dev=f"{version} fast", dev_MASE=o["MASE"][0], dev_CRPS=o["CRPS"][0], dev_cov_MASE=c["MASE"][0],
                dev_cov_CRPS=c["CRPS"][0], dev_cov_gain=1 - cov["on_vs_hidden"]["ratio"])


def probes(ckpt: Path, run_dir: Path, root: str | None, device: str, log=print) -> dict:
    if not root or not Path(root).exists() or not any(Path(root).glob("*")):
        log(f"probes missing ({root or '$TF_DATA unset'}): probe suite skipped")
        return dict(probes="probes missing")
    from tideforge.probes import run as PR
    from tideforge.probes.models import make
    PR.run(make("m5", str(ckpt), device), PR.probe_ids("all"), Path(root), run_dir / "probes")
    return dict(probes=str(run_dir / "probes" / "results.csv"))


def append_row(path: Path, row: dict) -> None:
    """Appends one row; a file with an older header (fewer columns) is rewritten first with ROW's header, old cells kept."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    if not new:
        with open(path, newline="", encoding="utf-8") as f:
            old = list(csv.DictReader(f))
            f.seek(0)
            head = next(csv.reader(f), [])
        if head != ROW:
            if set(head) - set(ROW):
                raise SystemExit(f"{path}: unknown columns {sorted(set(head) - set(ROW))}")
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=ROW, lineterminator="\n")
                w.writeheader()
                w.writerows({k: r.get(k, "") for k in ROW} for r in old)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ROW, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in ROW})


def finish(ckpt: Path, run_dir: Path | None = None, dev: str | None = None, version: str = "dev-v1",
           probe_root: str | None = None, results: str | None = None, device: str = "cuda", log=print) -> dict:
    import torch
    ckpt = Path(ckpt)
    run_dir = Path(run_dir or ckpt.parent)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8")) if (run_dir / "run.json").exists() else {}
    hold = ""
    mf = run_dir / "metrics.jsonl"
    if mf.exists():
        vals = [json.loads(s) for s in mf.read_text(encoding="utf-8").splitlines() if s.strip()]
        fv = [v for v in vals if v.get("kind") in ("final_val", "val")]
        hold = fv[-1]["holdout"] if fv else ""
    dev = dev if dev is not None else (str(Path(_env("TF_DATA")) / "dev") if _env("TF_DATA") else None)
    probe_root = probe_root if probe_root is not None else (str(Path(_env("TF_DATA")) / "probes") if _env("TF_DATA") else None)
    row = dict(date=dt.date.today().isoformat(), run_id=ck.get("run_name", info.get("run_id", "")), exp=info.get("exp", ""),
               seed=info.get("seed", ""), code=info.get("code", ""), steps=ck.get("step", ""), points=ck.get("points_seen", ""),
               holdout_final=hold, holdout_note=HOLDOUT_NOTE, ckpt=str(ckpt), run_dir=str(run_dir))
    row.update(dev_score(ckpt, run_dir, dev, version, device, log))
    row.update(probes(ckpt, run_dir, probe_root, device, log))
    res = Path(results or (Path(_env("TF_ROOT") or run_dir.parent) / "results" / "results.csv"))
    append_row(res, row)
    log(f"results row -> {res}: dev {row['dev']}, probes {row['probes']}")
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--version", default="dev-v1")
    ap.add_argument("--probes", default=None)
    ap.add_argument("--results", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    finish(Path(a.ckpt), a.run_dir, a.dev, a.version, a.probes, a.results, a.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
