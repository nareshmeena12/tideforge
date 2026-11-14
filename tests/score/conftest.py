"""A tiny fake tfbench release in the packaged layout (data/<cfg>/data.parquet, tasks/<tier>.yaml, locked/..., manifest.csv,
tags.csv) and a tiny random group-attention ft3 checkpoint. Tasks cover: univariate with masked NaN in the eval span and a
short item, a covariate task (numeric past, numeric known, string known), a multivariate dataset as univariate targets
(generate_univariate_targets_from) and as a joint task, and one locked dataset."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

QL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def write_fev(path, items: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = {k: [it[k] for it in items] for k in items[0]}
    arrays = {}
    for k, v in cols.items():
        if k == "id":
            arrays[k] = pa.array(v, pa.string())
        elif k == "timestamp":
            arrays[k] = pa.array([list(x) for x in v], pa.list_(pa.timestamp("ms")))
        elif isinstance(v[0][0], str):
            arrays[k] = pa.array([list(x) for x in v], pa.list_(pa.string()))
        else:
            arrays[k] = pa.array([np.asarray(x, float).tolist() for x in v], pa.list_(pa.float64()))
    pq.write_table(pa.table(arrays), path)


def series(n, seed, period=7, nan_tail=0):
    r = np.random.default_rng(seed)
    y = 20 + 5 * np.sin(2 * np.pi * np.arange(n) / period) + np.cumsum(r.normal(0, 0.3, n))
    if nan_tail:
        y[-nan_tail] = np.nan                       # one NaN inside the evaluation span (fev masks it)
    return y


def task(cfg, name, H, nw, m, **kw):
    t = dict(dataset_path="nareshmeena12/tfbench", dataset_config=cfg, horizon=H, num_windows=nw, window_step_size=H,
             min_context_length=20, seasonality=m, eval_metric="MASE", extra_metrics=["WQL", "SQL"], quantile_levels=QL,
             id_column="id", timestamp_column="timestamp", past_dynamic_columns=[], known_dynamic_columns=[],
             static_columns=[], target="target", task_name=name)
    t.update(kw)
    return t


def make_release(root):
    rel = root / "v9.9.9"
    d0 = pd.Timestamp("2020-01-01")
    # a: daily univariate, 4 items, one NaN in an eval span, one short item (dropped from early windows)
    its = []
    for k, n in enumerate([200, 220, 180, 40]):
        ts = pd.date_range(d0, periods=n, freq="D")
        its.append(dict(id=f"a{k}", timestamp=ts, target=series(n, k, nan_tail=3 if k == 1 else 0)))
    write_fev(rel / "data" / "a__d" / "data.parquet", its)
    # b: hourly, past numeric cov, known numeric + string cov
    its = []
    for k in range(3):
        n = 300
        ts = pd.date_range(d0, periods=n, freq="h")
        r = np.random.default_rng(10 + k)
        temp = r.normal(0, 1, n)
        hol = np.where(np.arange(n) % 24 < 8, "night", np.where(np.arange(n) % 24 < 18, "day", "eve")).tolist()
        its.append(dict(id=f"b{k}", timestamp=ts, target=series(n, 20 + k, period=24) + 2 * temp, temp=r.normal(0, 1, n),
                        price=temp, hol=hol))
    write_fev(rel / "data" / "b__h" / "data.parquet", its)
    # c: daily, two targets x, y
    its = []
    for k in range(3):
        n = 150
        ts = pd.date_range(d0, periods=n, freq="D")
        its.append(dict(id=f"c{k}", timestamp=ts, x=series(n, 30 + k), y=series(n, 40 + k)))
    write_fev(rel / "data" / "c__d" / "data.parquet", its)
    # d: locked
    its = [dict(id=f"d{k}", timestamp=pd.date_range(d0, periods=160, freq="D"), target=series(160, 50 + k)) for k in range(3)]
    write_fev(rel / "locked" / "data" / "d__d" / "data.parquet", its)

    tuning = [task("a__d", "a__d/short/target", 7, 3, 7),
              task("b__h", "b__h/short/target", 24, 2, 24),
              task("b__h", "b__h/short/cov", 24, 2, 24, past_dynamic_columns=["temp"], known_dynamic_columns=["price", "hol"]),
              task("c__d", "c__d/short/target", 7, 2, 7, generate_univariate_targets_from=["x", "y"]),
              task("c__d", "c__d/short/joint", 7, 2, 7, target=["x", "y"])]
    locked = [task(None, "d__d/short/target", 7, 2, 7, dataset_path="locked/data/d__d/data.parquet")]
    meta = {"a__d/short/target": ("a", "g1", "energy", "D", "daily", "none", False),
            "b__h/short/target": ("b", "g1", "energy", "h", "hourly", "none", False),
            "b__h/short/cov": ("b", "g1", "energy", "h", "hourly", "known", False),
            "c__d/short/target": ("c", "c", "nature", "D", "daily", "none", False),
            "c__d/short/joint": ("c", "c", "nature", "D", "daily", "none", True),
            "d__d/short/target": ("d", "d", "sales", "D", "daily", "none", False)}
    rows, tags = [], []
    for t in tuning + locked:
        ds, g, dom, fr, band, cov, mv = meta[t["task_name"]]
        sp = "locked" if ds == "d" else "tuning"
        rows.append(dict(task=t["task_name"], dataset=ds, source_group=g, config=t["task_name"].split("/")[0], domain=dom,
                         freq=fr, band=band, horizon_class="short", H=t["horizon"], variant=t["task_name"].split("/")[-1],
                         multivariate=mv, split=sp, tier_full=True, tier_core=True, tier_fast=ds != "c",
                         n_series=3, n_windows=t["num_windows"]))
        tags.append(dict(task=t["task_name"], covariates=cov, domain=dom, band=band))
    man = pd.DataFrame(rows)
    from tideforge.tfbench import tiers as TI
    for tier in ("full", "core", "fast"):
        man[f"weight_{tier}"] = 0.0
        for sp, g in man.groupby("split"):
            gg = g.rename(columns={"dataset": "dataset_id"})
            man.loc[g.index, f"weight_{tier}"] = TI.weights(gg, g[f"tier_{tier}"].to_numpy()).fillna(0.0).to_numpy()
    man.to_csv(rel / "manifest.csv", index=False)
    pd.DataFrame(tags).to_csv(rel / "tags.csv", index=False)
    for sp, tl in (("tuning", tuning), ("locked", locked)):
        base = rel / ("locked/tasks" if sp == "locked" else "tasks")
        base.mkdir(parents=True, exist_ok=True)
        for tier in ("full", "core", "fast"):
            sel = [t for t in tl if man.set_index("task").loc[t["task_name"], f"tier_{tier}"]]
            (base / f"{tier}.yaml").write_text(yaml.safe_dump({"tasks": sel}, sort_keys=False), encoding="utf-8")
    return rel


@pytest.fixture
def release(tmp_path):
    return make_release(tmp_path)


TINY = None


def tiny_cfg():
    from tideforge.eval import thirdparty
    C = thirdparty.ft3_core()
    return dict(C.CONFIG, decoder="causal", patch_len=8, d_model=32, d_kv=8, n_heads=4, d_ff=64, n_layers=2, dropout=0.0,
                group_attn=True, group_lanes=8, max_pass_patches=32)


@pytest.fixture
def tiny_ckpt(tmp_path):
    import torch
    from tideforge.model import ft3_model as FM
    torch.manual_seed(0)
    cfg = tiny_cfg()
    m = FM.build_masked_model(cfg)
    p = tmp_path / "tiny.pt"
    torch.save({"cfg": cfg, "model": m.state_dict(), "step": 0}, p)
    return p


def dump(x):
    return json.dumps(x, default=str)
