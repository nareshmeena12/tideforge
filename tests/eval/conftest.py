"""A tiny frozen dev version built with tideforge.devdata.build: 3 tuning datasets in 3 domains, 1 locked, 1 covariate task."""
import json

import numpy as np
import pandas as pd
import pytest

from tideforge.devdata import build as B
from tideforge.devdata import freqs, ingest
from tideforge.eval import devmetrics as M


@pytest.fixture
def horizons(tmp_path, monkeypatch):
    p = tmp_path / "h.csv"
    pd.DataFrame([("h", t, h) for t, h in zip(freqs.TERMS, (48, 96, 168))],
                 columns=["freq", "term", "prediction_length"]).to_csv(p, index=False)
    monkeypatch.setattr(freqs, "TIME_HORIZONS", p)
    monkeypatch.setattr(freqs, "HORIZONS_EXTRA", tmp_path / "none.csv")
    freqs.table.cache_clear()
    yield
    freqs.table.cache_clear()


@pytest.fixture
def tb():
    """timebench's metrics (skips when timebench cannot be found: set TF_TIME_REPO to a clone of zqiao11/TIME)."""
    try:
        return M.timebench_metrics()
    except ImportError as e:
        pytest.skip(str(e))


def _series(n, i, rng):
    t = np.arange(n)
    return 10 + 3 * np.sin(2 * np.pi * t / 24 + i) + rng.normal(0, 0.5, n)


def stage(staging, did, slug, domain, n_series=3, n=600, seed=0):
    rng = np.random.default_rng(seed)
    d = staging / slug
    d.mkdir(parents=True)
    ts = pd.date_range("2020", periods=n, freq="h")
    df = pd.concat([pd.DataFrame({"series_id": f"s{i}", "timestamp": ts, "value": _series(n, i, rng)}) for i in range(n_series)])
    df.to_parquet(d / "h.parquet", index=False)
    meta = dict(dataset_id=did, slug=slug, domain=domain, licence="x", covariate_only=False,
                tasks={"h": dict(n_series=n_series, n_points=len(df), nan_share=0.0, sha256=ingest.sha256_file(d / "h.parquet"))})
    (d / "meta.json").write_text(json.dumps(meta))


def stage_cov(staging, task="toy_cov", n=600):
    rng = np.random.default_rng(7)
    d = staging / "covtasks" / task
    d.mkdir(parents=True)
    ts = pd.date_range("2020", periods=n, freq="h")
    rows = []
    for i in range(2):
        x = rng.normal(0, 1, n)
        rows.append(pd.DataFrame({"series_id": f"c{i}", "timestamp": ts, "target": _series(n, i, rng) + 2 * x, "temp": x,
                                  "load": rng.normal(0, 1, n)}))
    pd.concat(rows).to_parquet(d / "h.parquet", index=False)
    meta = dict(task=task, target={"source": "D053"}, target_slug="ghcn_daily", split_unit="u", role="tuning",
                covariates={"temp": {"role": "known"}, "load": {"role": "past"}})
    (d / "meta.json").write_text(json.dumps(meta))


def build_version(tmp_path, monkeypatch, name="toy-v1", sub="dev"):
    monkeypatch.setattr(B, "MANIFESTS", tmp_path / f"manifests_{sub}")
    staging = tmp_path / f"staging_{sub}"
    stage(staging, "D053", "ghcn_daily", "Weather-climate", seed=1)
    stage(staging, "D071", "camels_us", "Hydrology", seed=2)
    stage(staging, "D078", "aq_toy", "Air-quality", seed=3)
    stage(staging, "D347", "iowa_liquor", "Retail-sales", seed=4)
    stage_cov(staging)
    B.build(staging, tmp_path / sub, name, pipeline_test=True, say=lambda *a: None)
    return tmp_path / sub


@pytest.fixture
def devroot(tmp_path, monkeypatch, horizons):
    return build_version(tmp_path, monkeypatch)
