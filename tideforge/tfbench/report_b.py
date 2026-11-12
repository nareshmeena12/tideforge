"""B report: BUILD_REPORT.md = the Milestone A report + task counts (tier x split, band, domain, horizon, covariates) +
the per-task seasonal-naive baseline (MASE / WQL) with NaN / infinite scores flagged + drops and items for review."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tideforge.tfbench.report import _md


def write(path: Path, release: Path, report_a: str, baseline: pd.DataFrame, task_root: Path, crossdup: pd.DataFrame,
          failures: pd.DataFrame) -> str:
    man = pd.read_csv(Path(release) / "manifest.csv")
    tags = pd.read_csv(Path(release) / "tags.csv")
    m = man.merge(tags[["task", "covariates"]], on="task", how="left")
    L = [report_a, "", "# Milestone B: tasks", ""]
    L += [f"Tasks: {len(m)} from {m['dataset'].nunique()} source datasets, {m['config'].nunique()} frequency variants.", ""]
    rows = []
    for tier in ("full", "core", "fast"):
        for sp in ("tuning", "locked"):
            g = m[m[f"tier_{tier}"] & (m["split"] == sp)]
            rows.append(dict(tier=tier, split=sp, tasks=len(g), datasets=g["dataset"].nunique()))
    L += ["## Tasks per tier and split", _md(pd.DataFrame(rows))]
    for col, title in (("band", "frequency band"), ("domain", "domain"), ("horizon_class", "horizon"),
                       ("covariates", "covariates")):
        g = m.groupby([col, "split"]).size().unstack(fill_value=0).reset_index()
        L += [f"## Tasks (full tier) by {title}", _md(g)]
    if not failures.empty:
        L += ["## Datasets that failed in Milestone B", _md(failures)]
    L += ["## Cross-dataset copies dropped from tuning (z-normalised window hashes, > 10% of windows)",
          f"Series dropped: {len(crossdup)}.", _md(crossdup.groupby(["dataset_id", "matched"]).size()
                                                   .reset_index(name="series") if not crossdup.empty else crossdup)]
    vr = []
    for f in sorted(Path(task_root).glob("*/subset.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        for v in r["variants"]:
            vr.append(dict(config=v["config"], agg=", ".join(f"{k}:{a}" for k, a in (v.get("agg") or {}).items()) or "native",
                           series=v.get("series", 0), dropped_short=v.get("dropped_short", 0),
                           nan_in_eval=v.get("nan_in_eval", 0), zero_scale=v.get("zero_scale", 0),
                           skipped=v.get("skipped", "")))
    vr = pd.DataFrame(vr)
    L += ["## Frequency variants: series kept and dropped; aggregation rule per column (for review)", _md(vr)]
    b = baseline.merge(man[["task", "split", "band", "horizon_class"]], on="task", how="left")
    fl = b[b["flag"].fillna("") != ""]
    L += ["## Seasonal naive baseline",
          f"Tasks scored: {len(b)}. Flagged (NaN or infinite score): {len(fl)}.", ""]
    if not fl.empty:
        L += ["### Flagged tasks", _md(fl[["task", "MASE", "WQL", "flag"]])]
    show = b[["task", "split", "MASE", "WQL", "SQL", "series_windows"]].copy()
    for c in ("MASE", "WQL", "SQL"):
        show[c] = show[c].map(lambda x: f"{x:.3f}")
    L += ["### Per task", _md(show)]
    text = "\n".join(L)
    Path(path).write_text(text, encoding="utf-8")
    return text
