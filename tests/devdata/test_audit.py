"""Value audit: a scaled, shifted, float32, offset copy of a benchmark series is caught; an unrelated series of the same kind is
not; aggregation to the benchmark frequency; noise filter; DB save / load; name audit."""
import numpy as np
import pandas as pd

from tideforge.devdata import audit
from tideforge.devdata.schema import load_sources


def load_like(n, seed):
    """Hourly 'load': daily + weekly cycle, slow trend, AR(1) noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.8 * e[i - 1] + rng.normal()
    return 100 + 20 * np.sin(2 * np.pi * t / 24 + rng.uniform(0, 6)) + 5 * np.sin(2 * np.pi * t / 168) + 0.01 * t + 3 * e


def bench_db(series):
    db = audit.DB()
    for k, x in enumerate(series):
        db.add(x, bench="TIME", dataset=f"bench{k}", item="0", freq="h")
    return db.build()


def test_copy_is_caught_and_unrelated_is_not():
    b = load_like(3000, 1)
    db = bench_db([b, load_like(3000, 2)])
    copy = (b[500:2500] * 0.1 + 7.0).astype(np.float32)
    copy = np.r_[np.random.default_rng(9).normal(size=7), copy]           # offset by 7 points
    m = audit.match_series(copy, db)
    assert len(m) >= audit.FLAG_WINDOWS and {db.meta[x["ref"]]["dataset"] for x in m} == {"bench0"}
    assert all(x["r"] > audit.R_MIN for x in m)
    other = load_like(3000, 3)                                            # same frequency, same kind, different series
    assert audit.match_series(other, db) == []


def test_two_codings_catch_a_bin_edge_flip():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(1, 64))
    c1, c2, ok = audit.codes(w)
    z = (w - w.mean()) / w.std()
    k = int(np.argmin(np.abs(z[0] / audit.BIN - np.round(z[0] / audit.BIN))))     # the value closest to a coding-1 edge
    edge = np.round(z[0, k] / audit.BIN) * audit.BIN
    for eps in (-1e-6, 1e-6):
        z2 = z.copy()
        z2[0, k] = edge + eps
        a1, a2, _ = audit.codes(z2)
        b1, b2, _ = audit.codes(np.where(np.arange(64) == k, edge - eps, z2))
        assert audit.hashes(a2)[0] == audit.hashes(b2)[0]           # coding 2 agrees where coding 1 may flip


def test_noise_filter():
    flat = np.ones((1, 64))
    steps = np.repeat(np.arange(4.0), 16)[None]
    zeros = np.r_[np.zeros(50), np.arange(1.0, 15.0)][None]
    for w in (flat, steps, zeros):
        assert not audit.codes(w)[2][0]
    assert audit.codes(np.random.default_rng(1).normal(size=(1, 64)))[2][0]


def test_aggregated_copy_is_caught(tmp_path):
    rng = np.random.default_rng(4)
    q = load_like(4 * 2000, 5) + rng.normal(size=8000)                    # 15-min dev series
    ts = pd.Series(q, index=pd.date_range("2024-01-01", periods=8000, freq="15min"))
    hourly = ts.resample("h").mean().to_numpy()
    db = bench_db([hourly[100:1900]])
    df = pd.DataFrame({"series_id": "s", "timestamp": ts.index, "value": q})
    m = audit.check_dataset(df, "15min", db)
    assert (m["at_freq"] == "h").sum() >= audit.FLAG_WINDOWS
    f = audit.flags(m)
    assert f.iloc[0]["flag"] and f.iloc[0]["ref_dataset"] == "bench0"
    db.save(tmp_path / "db")
    db2 = audit.DB.load(tmp_path / "db")
    assert len(audit.check_dataset(df, "15min", db2)) == len(m)


def test_name_audit_runs_on_batch1():
    srcs, batches = load_sources()
    df = audit.name_audit(srcs, batches["batch1"])
    assert len(df) == 21 and df.set_index("dataset_id").loc["D053", "check_by_hand"]
