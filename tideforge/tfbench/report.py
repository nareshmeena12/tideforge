"""A4 BUILD_REPORT_A.md: datasets in / out with reasons, one row per built dataset, totals by domain and frequency band,
the leakage audit, and everything waiting for the owner's review (known_candidate covariates, frequency conflicts,
keep_filtered rows without a drop rule, unreadable files)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _md(df: pd.DataFrame) -> str:
    if df.empty:
        return "(none)\n"
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(str(v).replace("|", "/").replace("\n", " ") for v in r) + " |")
    return "\n".join(out) + "\n"


def load_status(canon: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(Path(canon).glob("*/status.json")):
        rows.append(json.loads(f.read_text(encoding="utf-8")))
    return pd.DataFrame(rows)


def write(path: Path, selection: pd.DataFrame, canon: Path, audit_csv: Path | None) -> str:
    st = load_status(canon)
    metas = {}
    for f in Path(canon).glob("*/meta.json"):
        m = json.loads(f.read_text(encoding="utf-8"))
        metas[m["dataset_id"]] = m
    L = ["# tfbench build report, Milestone A", ""]
    sel_in = selection[selection["keep"].astype(str).str.lower() == "true"]
    L += ["## 1. Selection (A1)", f"Catalog rows: {len(selection)}. Selected: {len(sel_in)}. Dropped: {len(selection) - len(sel_in)}.", ""]
    L.append(_md(selection[selection["keep"].astype(str).str.lower() != "true"]["reason"].str.split(":").str[0]
                 .value_counts().rename_axis("reason").reset_index(name="rows")))
    caps = Path(canon).parent / "file_caps.csv"
    if caps.exists():
        c = pd.read_csv(caps)
        L += ["### Repos with a file cap (series sampled; time partitions are never capped)",
              _md(c[["repo_id", "files_before", "files_after", "series_keys", "reason"]])]
    ok = st[st["status"] == "ok"] if not st.empty else st
    bad = st[st["status"] != "ok"] if not st.empty else st
    L += ["## 2. Canonical datasets (A3)", f"Built: {len(ok)}. Not built: {len(bad)}.", ""]
    if not bad.empty:
        L += ["### Not built, with the reason", _md(bad[["dataset_id", "status", "reason"]])]
    rows = []
    for did, m in sorted(metas.items()):
        cov = m.get("covariates", {})
        rows.append(dict(dataset=did, domain=m.get("domain"), freq=m["frequency"], band=m["freq_band"], series=m["n_series"],
                         points=m["n_points"], len_min=m["length"]["min"], len_med=int(m["length"]["median"]),
                         len_max=m["length"]["max"], nan=f"{m['nan_share']:.1%}", targets=len(m["targets"]),
                         cov_past=sum(v != "known" for v in cov.values()), cov_known=sum(v == "known" for v in cov.values()),
                         licence=m["licence"].get("hub") or m["licence"].get("catalog")))
    per = pd.DataFrame(rows)
    L += ["### Per dataset", _md(per)]
    if not per.empty:
        for key, title in (("domain", "domain"), ("band", "frequency band")):
            g = per.groupby(key).agg(datasets=("dataset", "size"), series=("series", "sum"), points=("points", "sum")).reset_index()
            L += [f"### Totals by {title}", _md(g)]
        L.append(f"Total: {len(per)} datasets, {per['series'].sum():,} series, {per['points'].sum():,} points.\n")
    if audit_csv and Path(audit_csv).exists():
        a = pd.read_csv(audit_csv)
        L += ["## 3. Leakage audit (A5)",
              f"Datasets audited: {len(a)}. Flagged (> 1% of windows after series drops, excluded until decided): "
              f"{int(a['flagged'].sum())}. Series dropped (> 10% of windows): {int(a['series_dropped'].sum())}. "
              f"Old rule (>= 3 windows) would flag: {int(a['old_rule_flag'].sum())}.", ""]
        L.append(_md(a[(a["flagged"]) | (a["series_dropped"] > 0)][["dataset_id", "series_dropped", "share_after_drops",
                                                                   "flagged", "old_rule_windows", "matched_task"]]))
    L += ["## 4. For your review", "### Covariates proposed as known (known_candidate; used as past in v1)"]
    kc = [(d, c) for d, m in metas.items() for c, v in m.get("covariates", {}).items() if v == "known_candidate"]
    L.append(_md(pd.DataFrame(kc, columns=["dataset", "column"])))
    L.append("### Covariates marked known (calendar / holiday name, or overrides.yaml)")
    kn = [(d, c) for d, m in metas.items() for c, v in m.get("covariates", {}).items() if v == "known"]
    L.append(_md(pd.DataFrame(kn, columns=["dataset", "column"])))
    L.append("### Frequency from timestamps differs from the profile")
    fc = [(d, m["frequency"], m["profile"].get("frequency"), f"{m.get('freq_share') or 0:.0%}") for d, m in metas.items()
          if m.get("freq_conflict_with_profile")]
    L.append(_md(pd.DataFrame(fc, columns=["dataset", "from timestamps", "profile", "share at mode"])))
    L.append("### Irregular timestamps (under 80% of steps at the detected frequency)")
    ir = [(d, m["frequency"], f"{m['freq_share']:.0%}") for d, m in metas.items() if m.get("freq_share") is not None and m["freq_share"] < 0.8]
    L.append(_md(pd.DataFrame(ir, columns=["dataset", "frequency", "share at mode"])))
    L.append("### Target chosen by a rule (check: overrides.yaml `targets` fixes it)")
    tr = [(d, ", ".join(m["targets"][:4]), m.get("target_rule")) for d, m in metas.items() if m.get("target_rule") != "overrides.yaml"
          and m.get("layout") == "long"]
    L.append(_md(pd.DataFrame(tr, columns=["dataset", "targets", "rule"])))
    L.append("### Series dropped by a catalog rule (drop_items) or by reference ids (ref_ids)")
    dr = [(d, t["step"].split(" ")[0], t.get("count")) for d, m in metas.items() for t in m.get("transformations", [])
          if t["step"].startswith(("drop_items", "ref_ids"))]
    L.append(_md(pd.DataFrame(dr, columns=["dataset", "rule", "series dropped"])))
    L.append("### Event tables (aggregation rule)")
    ev = [(d, m["event_rule"]["step"], m["event_rule"]["aggregate"]) for d, m in metas.items() if m.get("event_rule")]
    L.append(_md(pd.DataFrame(ev, columns=["dataset", "step", "aggregate"])))
    text = "\n".join(L)
    Path(path).write_text(text, encoding="utf-8")
    return text
