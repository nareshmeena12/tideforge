"""B report: BUILD_REPORT.md = the Milestone A report + task counts (tier x split, band, domain, horizon, covariates) +
the per-task seasonal-naive baseline (MASE / WQL) with NaN / infinite scores flagged + drops and items for review."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tideforge.tfbench.report import _md


def write(path: Path, release: Path, report_a: str, baseline: pd.DataFrame, task_root: Path, crossdup: pd.DataFrame,
          failures: pd.DataFrame, audit_versions: pd.DataFrame | None = None, datasets: set[str] | None = None,
          splits: dict | None = None, version: str = "") -> str:
    from tideforge.tfbench.tasks import zero_reasons
    subset_files = [f for f in sorted(Path(task_root).glob("*/subset.json"))
                    if datasets is None or f.parent.name in datasets]
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
    zr = []
    for f in subset_files:
        tj = f.parent / "tasks.json"
        if tj.exists() and not json.loads(tj.read_text(encoding="utf-8")):
            r = json.loads(f.read_text(encoding="utf-8"))
            sp = ((splits or {}).get("datasets", {}).get(f.parent.name) or {}).get("split", "")
            zr.append(dict(dataset=f.parent.name, split=sp, reasons=zero_reasons(r).replace("|", "/")))
    L += [f"## Datasets with 0 tasks ({len(zr)})", _md(pd.DataFrame(zr, columns=["dataset", "split", "reasons"]))]
    if audit_versions is not None and not audit_versions.empty:
        L += ["## Audit code version per dataset (audit_code in audit/<dataset_id>.json)",
              ", ".join(f"{r.datasets} {r.audit_code}" for r in audit_versions.itertuples()) + ".",
              _md(audit_versions)]
    L += ["## Leakage audit: note for v1.1",
          f"The datasets the audit flagged stay excluded in {version or 'this release'}. Many of the matches behind the flags look spurious: "
          "smooth or monotone windows (trends, slow seasonal curves) z-normalise to near-identical shapes, so unrelated "
          "series hash alike. v1.1 will count a match only when several windows match the same reference series at a "
          "consistent offset (same lag between the dev and the reference window).", ""]
    if "source_group" in m:
        g = m[m["tier_full"]].groupby(["split", "source_group"]).agg(datasets=("dataset", "nunique"),
                                                                     tasks=("task", "size"), weight=("weight_full", "sum"))
        multi = g[g["datasets"] > 1].reset_index()
        L += ["## Source groups (one split per group, weight 1 per group per tier and split)",
              f"Groups: {m['source_group'].nunique()} ({len(multi)} with more than one dataset).", _md(multi)]
    recs = [json.loads(f.read_text(encoding="utf-8")) for f in subset_files]
    cal = [dict(dataset=r["dataset_id"], weekend_rows_removed=r.get("weekend_rows_removed", 0)) for r in recs
           if str(r.get("calendar", "")).startswith("business")]
    L += ["## Business-day calendar (daily data never observed on weekends: weekend rows removed, frequency B)",
          _md(pd.DataFrame(cal, columns=["dataset", "weekend_rows_removed"]))]
    mo = [dict(dataset=r["dataset_id"], columns=", ".join(r["dropped_covariates_model_output"])) for r in recs
          if r.get("dropped_covariates_model_output")]
    L += ["## Covariates dropped: other models' forecasts or the shifted target (chronos, bolt, gbm, persistence, seed, "
          "pred, lag_k, rolling_)", _md(pd.DataFrame(mo, columns=["dataset", "columns"]))]
    dt_ = [dict(dataset=r["dataset_id"], **x) for r in recs for x in r.get("dropped_tasks", [])]
    from tideforge.tfbench.config import Settings
    L += [f"## Tasks dropped: seasonal-naive score NaN / infinite or MASE < {Settings().min_baseline_mase}",
          _md(pd.DataFrame(dt_, columns=["dataset", "task", "why"]))]
    L += ["## Cross-dataset copies dropped from tuning (z-normalised window hashes, > 10% of windows)",
          f"Series dropped: {len(crossdup)}.", _md(crossdup.groupby(["dataset_id", "matched"]).size()
                                                   .reset_index(name="series") if not crossdup.empty else crossdup)]
    vr = []
    for f in subset_files:
        r = json.loads(f.read_text(encoding="utf-8"))
        for v in r["variants"]:
            agg = ", ".join(f"{k}:{a}" for k, a in (v.get("agg") or {}).items()) or "native"
            if not v.get("horizons"):
                vr.append(dict(config=v["config"], horizon="", agg=agg, series=0, too_short=v.get("dropped_short", 0),
                               nan_in_eval=0, zero_scale=0, masked_eval_nan=0, closed_eval_nan=0,
                               skipped=v.get("skipped", "")))
            for cls, hr in (v.get("horizons") or {}).items():
                vr.append(dict(config=hr.get("config", v["config"]), horizon=f"{cls} (H={hr['H']})", agg=agg,
                               series=hr.get("series", 0),
                               too_short=hr.get("too_short", 0) + (v.get("dropped_short", 0) if cls == "short" else 0),
                               nan_in_eval=hr.get("nan_in_eval", 0), zero_scale=hr.get("zero_scale", 0),
                               masked_eval_nan=hr.get("masked_eval_nan", 0), closed_eval_nan=hr.get("closed_eval_nan", 0),
                               skipped=hr.get("skipped", "")))
    vr = pd.DataFrame(vr)
    L += ["## Series per frequency variant and horizon (drops counted per horizon; aggregation rule per column, for "
          "review). nan_in_eval: series dropped (> 10% unexpected NaN in the eval span or a window < 10% finite); "
          "zero_scale: near-constant context; masked_eval_nan / closed_eval_nan: NaN points kept in eval spans (masked by "
          "fev's metrics), outside / inside structurally closed slots", _md(vr)]
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
