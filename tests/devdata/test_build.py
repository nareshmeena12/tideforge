"""Dev-version build: domain cap water-filling, rolling windows, frozen sampling that keeps dataset weights, manifest, the
15-domain rule and frozen version folders."""
import json

import numpy as np
import pandas as pd
import pytest

from tideforge.devdata import build as B
from tideforge.devdata import freqs, ingest


def test_domain_weights():
    n = {f"d{i}": 1 for i in range(14)} | {"big": 30}
    w = B.domain_weights(n)
    assert sum(w.values()) == pytest.approx(1.0) and max(w.values()) <= B.DOMAIN_CAP + 1e-12
    assert w["big"] == pytest.approx(0.10) and w["d0"] == pytest.approx(0.9 / 14)
    with pytest.raises(ValueError):
        B.domain_weights({"a": 1, "b": 2})
    assert B.domain_weights({f"d{i}": 1 for i in range(10)}) == {f"d{i}": pytest.approx(0.1) for i in range(10)}


def test_series_windows():
    v = np.arange(500.0)
    assert B.series_windows(v, 48) == [452, 404, 356]
    v[452:500] = np.nan
    assert B.series_windows(v, 48) == [404, 356]
    assert B.series_windows(np.arange(100.0), 48) == []


def test_sample_keeps_dataset_weights():
    win = pd.DataFrame({"dataset_id": ["a"] * 50 + ["b"] * 5, "freq": "h", "series_id": [f"s{i}" for i in range(55)],
                        "term": "short", "window": 0})
    s = B.sample(win, {"a": 0.5, "b": 0.5}, 20)
    assert s.groupby("dataset_id")["weight"].sum().to_dict() == {"a": pytest.approx(0.5), "b": pytest.approx(0.5)}
    assert (s["dataset_id"] == "b").sum() == 5 and (s["dataset_id"] == "a").sum() == 10
    assert s.equals(B.sample(win.sample(frac=1, random_state=0), {"a": 0.5, "b": 0.5}, 20))


def _stage(staging, did, slug, domain, n_series=3, n=2000):
    d = staging / slug
    d.mkdir(parents=True)
    t = pd.date_range("2020", periods=n, freq="h")
    df = pd.concat([pd.DataFrame({"series_id": f"s{i}", "timestamp": t, "value": np.sin(np.arange(n) / 5 + i)})
                    for i in range(n_series)])
    df.to_parquet(d / "h.parquet", index=False)
    meta = dict(dataset_id=did, slug=slug, domain=domain, licence="x", covariate_only=False,
                tasks={"h": dict(n_series=n_series, n_points=len(df), nan_share=0.0, sha256=ingest.sha256_file(d / "h.parquet"))})
    (d / "meta.json").write_text(json.dumps(meta))


def test_build_pipeline_test(tmp_path, horizons, monkeypatch):
    monkeypatch.setattr(B, "MANIFESTS", tmp_path / "manifests")
    staging = tmp_path / "staging"
    _stage(staging, "D053", "ghcn_daily", "Weather-climate")          # tuning in split_v1
    _stage(staging, "D071", "camels_us", "Hydrology")                 # tuning
    _stage(staging, "D347", "iowa_liquor", "Retail-sales")            # locked
    with pytest.raises(ValueError, match="domains"):
        B.build(staging, tmp_path / "dev", "dev-v1", say=lambda *a: None)
    man = B.build(staging, tmp_path / "dev", "batch1-test", pipeline_test=True, say=lambda *a: None)
    assert man["n_domains"] == 3 and set(man["weights"]) == {"tuning", "locked"}
    assert man["weights"]["tuning"]["datasets"] == {"D053": 0.5, "D071": 0.5}
    s = pd.read_parquet(tmp_path / "dev" / "batch1-test" / "tuning_score.parquet")
    assert s.groupby("dataset_id")["weight"].sum().round(9).to_dict() == {"D053": 0.5, "D071": 0.5}
    assert (tmp_path / "manifests" / "batch1-test.json").exists()
    assert (tmp_path / "dev" / "batch1-test" / "data" / "ghcn_daily" / "h.parquet").exists()
    with pytest.raises(FileExistsError):
        B.build(staging, tmp_path / "dev", "batch1-test", pipeline_test=True, say=lambda *a: None)
