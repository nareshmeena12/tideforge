"""B3 package: the release folder $TFB_DATA/bench/<version>/ from the per-dataset task outputs ($TFB_DATA/tasks/<id>/).

    README.md                     dataset card: sources, licences, attribution; HF configs = tuning only
    data/<config>/data.parquet    tuning datasets (fev native form, float32)
    tasks/{full,core,fast}.yaml   tuning tasks (dataset_path = the HF repo, dataset_config = <config>)
    locked/data/<config>/data.parquet
    locked/tasks/{full,core,fast}.yaml   dataset_path = locked/data/<config>/data.parquet (relative; the scorer resolves it
                                  only with --locked), dataset_config null: not loadable through the card's configs
    manifest.csv  tags.csv  splits.json  subsets/<id>.json  baseline.csv  BUILD_REPORT.md

manifest.csv: one row per task (task, dataset, config, domain, frequency, band, horizon class, H, variant, multivariate,
tiers, split, n_series, n_windows, weight per tier, licence, data_sha256, task_sha256).
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml

from tideforge.tfbench import tiers as TI
from tideforge.tfbench.config import Settings

TIERS = ("full", "core", "fast")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def collect(task_root: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(Path(task_root).glob("*/tasks.json")):
        for t in json.loads(f.read_text(encoding="utf-8")):
            rows.append({**{k: v for k, v in t.items() if k != "tags"}, **t["tags"], "_dir": str(f.parent)})
    return pd.DataFrame(rows)


def assign_tiers(df: pd.DataFrame, splits: dict, s: Settings) -> pd.DataFrame:
    df = df.copy()
    df["split"] = df["dataset_id"].map(lambda d: splits["datasets"][d]["split"])
    df["source_group"] = df["dataset_id"].map(lambda d: splits["datasets"][d].get("source_group", d))
    df["tier_full"] = True
    df["tier_core"] = TI.core(df)
    df["tier_fast"] = False
    for sp, g in df.groupby("split"):
        m = TI.fast(g, g["tier_core"].to_numpy(), s.fast_n.get(sp, 0), s.seed)
        df.loc[g.index, "tier_fast"] = m.to_numpy()
    for tier in TIERS:
        df[f"weight_{tier}"] = 0.0
        for sp, g in df.groupby("split"):
            w = TI.weights(g, g[f"tier_{tier}"].to_numpy())
            df.loc[g.index, f"weight_{tier}"] = w.fillna(0.0).to_numpy()
    return df


def card(df: pd.DataFrame, metas: dict, repo: str, version: str) -> str:
    tuning = sorted(df.loc[df["split"] == "tuning", "config"].unique())
    head = {"license": "other", "pretty_name": "tfbench", "tags": ["time-series", "forecasting", "benchmark"],
            "configs": [{"config_name": c, "data_files": f"data/{c}/data.parquet"} for c in tuning]}
    L = ["---", yaml.safe_dump(head, sort_keys=False).strip(), "---", "", f"# tfbench {version}", "",
         "Private forecasting benchmark built from public Hugging Face datasets (tideforge/tfbench). fev task format.",
         "Tuning tasks: `tasks/{full,core,fast}.yaml`. The locked split lives under `locked/` and is not a config of this",
         "card; it is read only by the scorer's `--locked` option.", "",
         "## Sources and licences", "", "| dataset | source | config | licence (Hub) | licence (catalog) | split |",
         "|---|---|---|---|---|---|"]
    for did in sorted(df["dataset_id"].unique()):
        m = metas[did]
        sp = df.loc[df["dataset_id"] == did, "split"].iloc[0]
        src = m["source"]
        L.append(f"| {did} | [{src['repo']}](https://huggingface.co/datasets/{src['repo']}) | {src.get('config') or ''} | "
                 f"{m['licence'].get('hub') or ''} | {m['licence'].get('catalog') or ''} | {sp} |")
    L += ["", "Each source keeps its own licence and attribution; see the source repository for terms. Data were "
          "regularised, filtered, capped and audited as described in BUILD_REPORT.md.", ""]
    return "\n".join(L)


def build(task_root: Path, canon: Path, splits: dict, out: Path, repo: str, s: Settings) -> pd.DataFrame:
    df = collect(task_root)
    if df.empty:
        raise RuntimeError("no tasks to package")
    df = assign_tiers(df, splits, s)
    metas = {d: json.loads((Path(canon) / d / "meta.json").read_text(encoding="utf-8")) for d in df["dataset_id"].unique()}
    out = Path(out)
    for sub in ("data", "tasks", "locked", "subsets"):
        shutil.rmtree(out / sub, ignore_errors=True)
    shas = {}
    for (cfg, sp, d), _ in df.groupby(["config", "split", "_dir"]):
        dst = out / ("locked/data" if sp == "locked" else "data") / cfg / "data.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(d) / f"{cfg}.parquet", dst)
        shas[cfg] = sha256_file(dst)
    (out / "subsets").mkdir(parents=True, exist_ok=True)
    for d in df["_dir"].unique():
        shutil.copyfile(Path(d) / "subset.json", out / "subsets" / f"{Path(d).name}.json")
    for sp in ("tuning", "locked"):
        base = out / ("locked/tasks" if sp == "locked" else "tasks")
        base.mkdir(parents=True, exist_ok=True)
        for tier in TIERS:
            sel = df[(df["split"] == sp) & df[f"tier_{tier}"]]
            tl = []
            for r in sel.to_dict("records"):
                t = dict(r["task"])
                if sp == "locked":
                    t.update(dataset_path=f"locked/data/{r['config']}/data.parquet", dataset_config=None)
                else:
                    t["dataset_path"] = repo
                tl.append(t)
            (base / f"{tier}.yaml").write_text(yaml.safe_dump({"tasks": tl}, sort_keys=False), encoding="utf-8")
    df["data_sha256"] = df["config"].map(shas)
    df["licence"] = df["dataset_id"].map(lambda d: metas[d]["licence"].get("hub") or metas[d]["licence"].get("catalog"))
    df["H"] = df["task"].map(lambda t: t["horizon"])
    df["task"] = df["task"].map(lambda t: t["task_name"])
    man = df.rename(columns={"dataset_id": "dataset"})
    cols = ["task", "dataset", "source_group", "config", "domain", "freq", "band", "horizon_class", "H", "variant", "multivariate", "split",
            "tier_full", "tier_core", "tier_fast", "weight_full", "weight_core", "weight_fast", "n_series", "n_windows",
            "licence", "data_sha256", "task_sha256"]
    man = man[cols]
    man.to_csv(out / "manifest.csv", index=False)
    tag_cols = ["task", "domain", "band", "freq", "horizon", "covariates", "multivariate", "history_length_band",
                "intermittency", "series_length_band"]
    df.rename(columns={"dataset_id": "dataset"})[tag_cols].to_csv(out / "tags.csv", index=False)
    (out / "splits.json").write_text(json.dumps(splits, indent=1, sort_keys=True), encoding="utf-8")
    (out / "README.md").write_text(card(df, metas, repo, s.version), encoding="utf-8")
    return man


def check_manifest(out: Path) -> list[str]:
    """Problems between manifest.csv and the files: missing data, checksum mismatch, task not in its tier yaml."""
    out = Path(out)
    man = pd.read_csv(out / "manifest.csv")
    bad = []
    for cfg, g in man.groupby("config"):
        sp = g["split"].iloc[0]
        p = out / ("locked/data" if sp == "locked" else "data") / cfg / "data.parquet"
        if not p.exists():
            bad.append(f"{cfg}: data file missing")
        elif sha256_file(p) != g["data_sha256"].iloc[0]:
            bad.append(f"{cfg}: checksum mismatch")
    for sp in ("tuning", "locked"):
        for tier in TIERS:
            y = out / ("locked/tasks" if sp == "locked" else "tasks") / f"{tier}.yaml"
            names = {t["task_name"] for t in (yaml.safe_load(y.read_text(encoding="utf-8")) or {}).get("tasks", [])}
            want = set(man.loc[(man["split"] == sp) & man[f"tier_{tier}"], "task"])
            if names != want:
                bad.append(f"{sp}/{tier}.yaml: {len(names ^ want)} tasks differ from the manifest")
    return bad
