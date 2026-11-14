"""Per-task table, skill, bootstrap intervals, summary tables, the failure map, and compare mode.

Relative score of a task = model error / seasonal-naive error, clipped to [0.01, 100] as fev.leaderboard does; skill =
1 - weighted geometric mean of the relative scores, with the manifest's weight (1 per source group in each tier and split;
fev-bench: 1 per task). Tasks whose relative score is not finite are left out and counted (fev's missing_strategy "drop").
95% intervals: bootstrap over source groups (resampled with replacement, seeded).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tideforge.tfbench.report import _md

CLIP = (0.01, 100.0)
METRICS = ("test_error", "MASE", "WQL", "SQL")
TAGS = ["task", "dataset", "source_group", "band", "freq", "domain", "horizon_class", "covariates", "cov_split",
        "multivariate", "H", "n_windows", "n_series", "series_length_band", "history_length_band", "intermittency", "weight"]
BREAKDOWNS = [("band", "frequency band"), ("domain", "domain"), ("horizon_class", "horizon class"),
              ("covariates", "covariate type"), ("cov_split", "covariate vs non-covariate tasks"),
              ("multivariate", "multivariate (joint) tasks")]


def _band(freq: str) -> str:
    from tideforge.tfbench import rules
    try:
        return rules.band(freq)
    except Exception:  # noqa: BLE001  an alias the tfbench rules do not know (fev-bench): leave it blank
        return ""


def task_table(specs: list[dict], caches, model: dict, naive: dict, mode: str) -> pd.DataFrame:
    """One row per task: tags (from the manifest row, so the failure analysis needs no join), every metric of the model
    and of seasonal naive, relative scores."""
    rows = []
    for s, c in zip(specs, caches):
        r = s["row"]
        m, n = model[c.name], naive[c.name]
        cov = r.get("covariates") if isinstance(r.get("covariates"), str) else (
            "known" if c.info["known"] else "past" if c.info["past"] else "none")
        row = {k: r.get(k, "") for k in TAGS}
        row.update(task=c.name, covariates=cov, cov_split="covariate" if cov != "none" else "non-covariate",
                   freq=r.get("freq") or c.info["freq"], band=r.get("band") or _band(c.info["freq"]), H=c.info["horizon"],
                   n_windows=c.info["n_windows"], n_series=c.info["n_items"], weight=float(r.get("weight", 1.0)),
                   multivariate=bool(r.get("multivariate")) or len(c.info["target_columns"]) > 1,
                   eval_metric=str(c.info["eval_metric"]), groups=mode, num_forecasts=m["num_forecasts"])
        for k in METRICS:
            row[k], row[f"naive_{k}"] = m.get(k, np.nan), n.get(k, np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                row[f"rel_{k}"] = m.get(k, np.nan) / n.get(k, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def skill(df: pd.DataFrame, metric: str = "test_error") -> tuple[float, int, int]:
    """-> (skill, tasks used, tasks dropped as not finite)."""
    r = df[f"rel_{metric}"].to_numpy(float)
    w = df["weight"].to_numpy(float)
    ok = np.isfinite(r) & (w > 0)
    if not ok.any():
        return float("nan"), 0, int((~np.isfinite(r)).sum())
    lr = np.log(np.clip(r[ok], *CLIP))
    return float(1 - np.exp((w[ok] * lr).sum() / w[ok].sum())), int(ok.sum()), int((~np.isfinite(r)).sum())


def bootstrap(df: pd.DataFrame, metric: str = "test_error", n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% interval of the skill, resampling source groups."""
    groups = df["source_group"].astype(str).to_numpy()
    ug = np.unique(groups)
    if len(ug) < 2:
        return float("nan"), float("nan")
    r = np.log(np.clip(df[f"rel_{metric}"].to_numpy(float), *CLIP))
    w = df["weight"].to_numpy(float)
    ok = np.isfinite(r) & (w > 0)
    gi = np.searchsorted(ug, groups)
    num = np.bincount(gi[ok], weights=(w * np.where(ok, r, 0))[ok], minlength=len(ug))
    den = np.bincount(gi[ok], weights=w[ok], minlength=len(ug))
    rng = np.random.default_rng(seed)
    cnt = rng.multinomial(len(ug), np.full(len(ug), 1 / len(ug)), size=n)       # how often each group is drawn
    with np.errstate(invalid="ignore", divide="ignore"):
        sk = 1 - np.exp((cnt @ num) / (cnt @ den))
    sk = sk[np.isfinite(sk)]
    return (float(np.percentile(sk, 2.5)), float(np.percentile(sk, 97.5))) if len(sk) else (float("nan"), float("nan"))


def skill_row(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    s, used, dropped = skill(df)
    lo, hi = bootstrap(df, "test_error", n_boot, seed)
    out = {"tasks": len(df), "groups": df["source_group"].nunique(), "skill": round(100 * s, 2),
           "95% CI": f"[{100 * lo:.2f}, {100 * hi:.2f}]", "dropped": dropped}
    for m in ("MASE", "WQL", "SQL"):
        out[f"skill {m}"] = round(100 * skill(df, m)[0], 2)
    return out


def tables(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict[str, pd.DataFrame]:
    out = {"overall": pd.DataFrame([skill_row(df, n_boot, seed)])}
    for col, _ in BREAKDOWNS:
        rows = []
        for v, g in df.groupby(df[col].astype(str)):
            rows.append({col: v, **skill_row(g, n_boot, seed)})
        out[col] = pd.DataFrame(rows)
    return out


def failure_map(df: pd.DataFrame) -> pd.DataFrame:
    """Tasks sorted by relative score (worst first), with every tag: the input of the failure analysis."""
    cols = ["rel_test_error", "rel_MASE", "rel_WQL", "rel_SQL"] + [c for c in TAGS if c != "weight"] + ["weight",
                                                                                                        "eval_metric"]
    return df.sort_values("rel_test_error", ascending=False, na_position="first")[cols]


def summary_md(df: pd.DataFrame, title: str, meta: dict, n_boot: int = 1000, seed: int = 0, extra: str = "") -> str:
    tb = tables(df, n_boot, seed)
    L = [f"# {title}", "", "Skill = 100 x (1 - weighted geometric mean of model / seasonal-naive error), relative errors "
         "clipped to [0.01, 100], weight 1 per source group (manifest weight); 95% CI: bootstrap over source groups "
         f"({n_boot} draws, seed {seed}). `skill` uses each task's eval metric; the MASE / WQL / SQL columns use that metric "
         "on every task.", "", "```", json.dumps(meta, indent=1, default=str), "```", "", "## Overall", _md(tb["overall"])]
    for col, name in BREAKDOWNS:
        L += [f"## By {name}", _md(tb[col])]
    L += ["## Worst 20 tasks (relative error)", _md(failure_map(df).head(20)[
        ["task", "rel_test_error", "band", "domain", "horizon_class", "covariates", "source_group"]].round(4))]
    if extra:
        L += [extra]
    return "\n".join(L)


# ---- compare -------------------------------------------------------------------------------------------------------------
def compare(dirs: list[Path], n_boot: int = 1000, seed: int = 0) -> str:
    """Several result folders (each with tasks.csv) side by side on the tasks they share, plus each one against the
    first: weighted geometric mean of the error ratio with a source-group bootstrap interval."""
    dfs = {str(d): pd.read_csv(Path(d) / "tasks.csv") for d in dirs}
    common = set.intersection(*[set(x["task"]) for x in dfs.values()])
    dfs = {k: v[v["task"].isin(common)].sort_values("task").reset_index(drop=True) for k, v in dfs.items()}
    L = [f"# Compare ({len(common)} tasks in common)", ""]
    rows = [{"run": k, **skill_row(v, n_boot, seed)} for k, v in dfs.items()]
    L += ["## Overall", _md(pd.DataFrame(rows))]
    for col, name in BREAKDOWNS:
        rows = []
        for k, v in dfs.items():
            for val, g in v.groupby(v[col].astype(str)):
                rows.append({"run": k, col: val, "tasks": len(g), "skill": round(100 * skill(g)[0], 2)})
        if rows:
            L += [f"## By {name}", _md(pd.DataFrame(rows).pivot_table(index=col, columns="run", values="skill").reset_index())]
    keys = list(dfs)
    base = dfs[keys[0]]
    rows = []
    for k in keys[1:]:
        d = base[["task", "source_group", "weight", "test_error"]].merge(dfs[k][["task", "test_error"]], on="task",
                                                                        suffixes=("_a", "_b"))
        d["rel_test_error"] = d["test_error_b"] / d["test_error_a"]
        s, used, _ = skill(d)
        lo, hi = bootstrap(d, "test_error", n_boot, seed)
        rows.append({"run": k, "vs": keys[0], "tasks": used, "error ratio (gmean)": round(1 - s, 4),
                     "95% CI": f"[{1 - hi:.4f}, {1 - lo:.4f}]"})
    if rows:
        L += ["## Each run against the first (error ratio < 1: better than the first)", _md(pd.DataFrame(rows))]
    return "\n".join(L)
