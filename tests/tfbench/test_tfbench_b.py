"""tfbench Milestone B on tiny synthetic fixtures (no network)."""
from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import yaml

from tideforge.tfbench import horizons as HZ
from tideforge.tfbench import package as PK
from tideforge.tfbench import run as RUN
from tideforge.tfbench import split as SP
from tideforge.tfbench import tasks as TK
from tideforge.tfbench import tiers as TI
from tideforge.tfbench import upload as UP
from tideforge.tfbench.baseline import score
from tideforge.tfbench.config import Paths, Settings
from tideforge.tfbench.variants import aggregate, freq_variants

S = Settings()


def walk(n, seed, level=50.0):
    r = np.random.default_rng(seed)
    return np.cumsum(r.normal(size=n)) + level + 3 * np.sin(np.arange(n) * 2 * np.pi / 7)


def long_df(lengths, freq, seed=0, covs=False):
    parts = []
    for i, n in enumerate(lengths):
        ts = pd.date_range("1950-01-01" if freq in ("YS", "MS") else "2020-01-01", periods=n, freq=freq)
        d = pd.DataFrame({"item_id": f"s{seed}_{i}", "timestamp": ts, "y": walk(n, seed * 1000 + i)})
        if covs:
            d["temp"] = walk(n, 7 + i)
            d["is_holiday"] = (np.arange(n) % 9 == 0).astype(float)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def meta(did, freq, covs=None, domain="energy", targets=("y",)):
    return {"dataset_id": did, "frequency": freq, "targets": list(targets), "covariates": covs or {}, "static": [],
            "domain": domain, "layout": "long", "source": {"repo": f"o/{did}", "config": ""},
            "licence": {"hub": "mit", "catalog": "MIT"}}


def all_tasks(tmp, data, m, ov=None, s=S):
    return TK.build(data, m, ov or {}, s, tmp / m["dataset_id"])


# ---- horizons, windows, context -----------------------------------------------------------------------------------------
def test_horizon_table():
    assert HZ.horizons("h") == {"short": 48, "medium": 480, "long": 720}
    assert HZ.horizons("D") == {"short": 30} and HZ.horizons("YS") == {"short": 6} and HZ.horizons("W-SUN") == {"short": 8}
    assert HZ.seasonality("15min") == 96 and HZ.seasonality("h") == 24 and HZ.seasonality("MS") == 12
    assert HZ.seasonality("D") == 1 and HZ.seasonality("W-SUN") == 1 and HZ.seasonality("QS") == 4


@pytest.mark.parametrize("freq,lengths", [("YS", list(range(18, 61, 3))), ("D", list(range(158, 301, 7)))])
def test_every_series_that_fits_gets_a_window(tmp_path, freq, lengths):
    data = long_df(lengths, freq)
    ts = TK.build(data, meta("d", freq), {}, S, tmp_path / "d")
    assert ts, "no task"
    t = ts[0]["task"]
    H, ctx = t["horizon"], t["min_context_length"]
    assert ctx == max(2 * H, {"YS": 12, "D": 128}[freq]) and H == {"YS": 6, "D": 30}[freq]
    tbl = pq.read_table(tmp_path / "d" / f"{t['dataset_config']}.parquet").to_pydict()
    assert len(tbl["id"]) == len(lengths)                         # nobody dropped: all pass min_context + one H
    for y in tbl["y"]:                                            # the last window (cutoff -H) always fits
        assert len(y) - H >= ctx
    b = score(tmp_path / "d" / f"{t['dataset_config']}.parquet", t)
    assert b["series_windows"] >= len(lengths) and np.isfinite(b["MASE"])


def test_window_count_and_cap():
    assert HZ.n_windows([100] * 10, 30) == 1                      # 10% of 100 < 30: still 1
    assert HZ.n_windows([5000] * 10, 48) == 10
    assert HZ.n_windows([10 ** 6] * 10, 48) == 20


def test_medium_long_own_series_file(tmp_path):
    """One short series in an hourly dataset: medium / long still exist on their own file; short keeps every series."""
    ts = all_tasks(tmp_path, long_df([2500, 2500, 2500, 2500, 400], "h"), meta("a", "h"))
    native = [t for t in ts if t["freq_config"] == "a__h"]
    by = {t["horizon_class"]: t for t in native}
    assert set(by) == {"short", "medium", "long"}

    def n_ids(t):
        return len(pq.read_table(tmp_path / "a" / f"{t['config']}.parquet", columns=["id"]))
    assert by["short"]["config"] == "a__h" and n_ids(by["short"]) == 5
    assert by["medium"]["config"] == "a__h__medium" and n_ids(by["medium"]) == 4
    assert by["long"]["config"] == "a__h__medium" and n_ids(by["long"]) == 4      # same series: file reused
    rec = json.loads((tmp_path / "a" / "subset.json").read_text())["variants"][0]["horizons"]
    assert rec["medium"]["too_short"] == 1 and rec["long"]["too_short"] == 1 and rec["short"]["too_short"] == 0
    for t in native:
        assert t["task"]["min_context_length"] == max(2 * t["task"]["horizon"], 256)
    ts = all_tasks(tmp_path, long_df([2500, 400, 400], "h"), meta("b", "h"))     # under 75% fit: short only
    assert {t["horizon_class"] for t in ts if t["freq_config"] == "b__h"} == {"short"}


# ---- NaN, scale, aggregation, variants ---------------------------------------------------------------------------------
def test_nan_only_in_context_and_zero_scale(tmp_path):
    data = long_df([400, 400, 400], "D")
    data.loc[(data["item_id"] == "s0_1") & (data.index % 400 == 395), "y"] = np.nan    # NaN in the last window
    data.loc[(data["item_id"] == "s0_0") & (data.index % 400 == 10), "y"] = np.nan     # NaN in context: kept
    data.loc[data["item_id"] == "s0_2", "y"] = np.r_[np.full(380, 5.0), walk(20, 1)]  # flat context, scale 0 (D: m=1)
    all_tasks(tmp_path, data, meta("n", "D"))
    rec = json.loads((tmp_path / "n" / "subset.json").read_text())["variants"][0]["horizons"]["short"]
    assert rec["series_ids"] == ["s0_0"] and rec["nan_in_eval"] == 1 and rec["zero_scale"] == 1


def test_aggregate_coverage_and_rules():
    ts = pd.date_range("2021-01-01", periods=48, freq="h")
    y = np.arange(48.0)
    y[24:24 + 6] = np.nan                                         # day 2: 18 of 24 points = 75% < 80% -> NaN
    df = pd.DataFrame({"timestamp": ts, "y": y, "count": 1.0, "flag": (np.arange(48) % 5 == 0).astype(float)})
    a = aggregate(df, "h", "D", {"y": "mean", "count": "sum", "flag": "last"}, 0.8)
    assert a["y"].iloc[0] == pytest.approx(11.5) and np.isnan(a["y"].iloc[1])
    assert a["count"].iloc[0] == 24 and a["flag"].iloc[0] == df["flag"].iloc[23]


def test_variants_hourly_daily_weekly(tmp_path):
    n = 24 * 7 * 70                                               # 70 weeks of hourly data
    data = long_df([n], "h", covs=True)
    m = meta("v", "h", covs={"temp": "past", "is_holiday": "known"})
    vs = freq_variants(data, m, {}, S)
    assert [v["freq"] for v in vs] == ["h", "D", "W-SUN"]
    assert vs[1]["agg"]["is_holiday"] == "last" and vs[1]["agg"]["temp"] == "mean"
    short = freq_variants(long_df([24 * 300], "h"), meta("v2", "h"), {}, S)
    assert [v["freq"] for v in short] == ["h", "D"]               # 300 days = 42 weeks < 52 + 8


def test_daily_to_weekly_where_long_enough(tmp_path):
    vs = freq_variants(long_df([7 * 63], "D"), meta("w", "D"), {}, S)
    assert [v["freq"] for v in vs] == ["D", "W-SUN"]
    vs = freq_variants(long_df([7 * 50], "D"), meta("w", "D"), {}, S)
    assert [v["freq"] for v in vs] == ["D"]


def test_covariate_and_paired_tasks(tmp_path):
    ts = all_tasks(tmp_path, long_df([600, 600], "D", covs=True), meta("c", "D", covs={"temp": "past", "is_holiday": "known"}))
    native = [t for t in ts if t["freq_config"] == "c__d"]
    assert {t["variant"] for t in native} == {"target", "cov"}
    cov = next(t for t in native if t["variant"] == "cov")["task"]
    assert cov["past_dynamic_columns"] == ["temp"] and cov["known_dynamic_columns"] == ["is_holiday"]
    assert len({t["task"]["dataset_config"] for t in native}) == 1   # same table -> same series for on / off


def test_multivariate_joint(tmp_path):
    d = long_df([500, 500], "D")
    d["y2"] = d["y"] * 2 + 1
    ts = all_tasks(tmp_path, d, meta("m", "D", targets=("y", "y2")))
    j = [t for t in ts if t["variant"] == "joint"]
    u = [t for t in ts if t["variant"] == "target"]
    assert j and j[0]["task"]["target"] == ["y", "y2"]
    assert u[0]["task"]["generate_univariate_targets_from"] == ["y", "y2"]


def test_fev_loads_a_packaged_task(tmp_path):
    fev = pytest.importorskip("fev")
    ts = all_tasks(tmp_path, long_df([300, 320], "D", covs=True), meta("f", "D", covs={"temp": "past", "is_holiday": "known"}))
    t = dict(next(x for x in ts if x["variant"] == "cov")["task"])
    t.update(dataset_path=str(tmp_path / "f" / f"{t['dataset_config']}.parquet"), dataset_config=None)
    task = fev.Task(**t)
    w = list(task.iter_windows())
    past, fut = w[-1].get_input_data()
    assert len(w) == t["num_windows"] and len(past) == 2 and "is_holiday" in fut.column_names


# ---- split ----------------------------------------------------------------------------------------------------------------
def test_split_deterministic_and_append_only(tmp_path):
    ds = {f"d{i}": ("energy" if i % 2 else "finance") for i in range(40)}
    a = SP.assign(ds)
    b = SP.assign(dict(reversed(list(ds.items()))))
    assert {k: v["split"] for k, v in a["datasets"].items()} == {k: v["split"] for k, v in b["datasets"].items()}
    for dom in ("energy", "finance"):
        sp = [v["split"] for v in a["datasets"].values() if v["domain"] == dom]
        assert sp.count("locked") == 5                            # 20 per domain -> every 4th
    p = tmp_path / "splits.json"
    first = SP.load_or_create(p, ds)
    more = SP.load_or_create(p, {**ds, "new1": "energy", "new2": "health"})
    for k in ds:
        assert more["datasets"][k] == first["datasets"][k]
    assert more["datasets"]["new1"]["how"] == "append"
    assert SP.load_or_create(p, {**ds, "new1": "energy", "new2": "health"}) == more


def test_crossdup_finds_copy():
    ref = walk(1500, 3)
    locked = pd.DataFrame({"item_id": "L", "timestamp": pd.date_range("2020", periods=1500, freq="h"), "y": ref})
    db = SP.build_locked_db([("lk", locked, ["y"])])
    tun = pd.concat([pd.DataFrame({"item_id": "copy", "timestamp": locked["timestamp"].iloc[:1000],
                                   "y": 4 * ref[200:1200] - 3}),
                     pd.DataFrame({"item_id": "own", "timestamp": locked["timestamp"].iloc[:1000], "y": walk(1000, 99)})])
    rows = SP.crossdups("tn", tun, ["y"], db, 0.10)
    assert [r["item_id"] for r in rows] == ["copy"] and rows[0]["matched"] == "lk"


# ---- tiers -------------------------------------------------------------------------------------------------------------
def tasks_frame():
    rows = []
    for d in range(12):
        for f in ("h", "D"):
            for cls in ("short", "medium") if f == "h" else ("short",):
                for v in ("target", "cov"):
                    rows.append(dict(dataset_id=f"d{d}", config=f"d{d}__{f}", horizon_class=cls, variant=v,
                                     domain=f"dom{d % 3}", band="hourly" if f == "h" else "daily"))
    return pd.DataFrame(rows)


def test_core_and_weights():
    t = tasks_frame()
    c = TI.core(t)
    core = t[c]
    assert len(core) == 24 and set(core["variant"]) == {"cov"}
    assert set(core.loc[core["band"] == "hourly", "horizon_class"]) == {"medium"}
    for m in (np.ones(len(t), bool), c):
        w = TI.weights(t, m)
        assert np.allclose(w[m].groupby(t.loc[m, "dataset_id"]).sum(), 1.0)
    f1, f2 = TI.fast(t, c, 9, 1), TI.fast(t, c, 9, 1)
    assert f1.sum() == 9 and (f1 == f2).all() and (~f1 | c).all()
    assert t[f1]["domain"].nunique() == 3


# ---- end to end: split -> crossdup -> tasks -> package -> report_b -> upload (dry run) -------------------------------------
def make_canon(paths: Paths, did, data, m):
    d = paths.canon / did
    d.mkdir(parents=True)
    lens = data.groupby("item_id").size()
    m = {**m, "freq_band": "daily", "n_series": int(len(lens)), "n_points": int(len(data)), "nan_share": 0.0,
         "length": {"min": int(lens.min()), "median": float(lens.median()), "max": int(lens.max())}}
    data.to_parquet(d / "data.parquet")
    (d / "meta.json").write_text(json.dumps(m))
    paths.audit.mkdir(parents=True, exist_ok=True)
    (paths.audit / f"{did}.json").write_text(json.dumps({"dataset_id": did, "flagged": False, "dropped_series": []}))


def test_end_to_end(tmp_path, monkeypatch):
    monkeypatch.delenv("TF_BENCH_REPO", raising=False)
    paths = Paths(data=tmp_path / "data", small=tmp_path / "small")
    paths.small.mkdir(parents=True)
    pd.DataFrame({"catalog_row": [1], "dataset_id": ["x"], "keep": [True], "reason": ["ok"]}).to_csv(paths.selection, index=False)
    ids = [f"ds{i}" for i in range(8)]
    for i, did in enumerate(ids):
        make_canon(paths, did, long_df([500, 520, 540], "D", seed=i, covs=(i % 2 == 0)),
                   meta(did, "D", covs={"temp": "past", "is_holiday": "known"} if i % 2 == 0 else {}))
    a = types.SimpleNamespace(only=None, jobs=2, redo=False, smoke=False, dry_run=True, max_gb=100.0)
    RUN.step_split(a, paths, S)
    with pytest.raises(ValueError):
        SP.force_locked(paths.small / "splits.json", ids, 2)          # never on the real splits
    sp = json.loads((paths.small / "splits.json").read_text())["datasets"]
    locked = [d for d in ids if sp[d]["split"] == "locked"]
    tuning = [d for d in ids if sp[d]["split"] == "tuning"]
    assert len(locked) == 2
    # plant a copy of a locked series into a tuning dataset (scaled, shifted)
    lk = pd.read_parquet(paths.canon / locked[0] / "data.parquet")
    src = lk[lk["item_id"] == lk["item_id"].iloc[0]]["y"].to_numpy()
    tdir = paths.canon / tuning[0]
    td = pd.read_parquet(tdir / "data.parquet")
    cp = td[td["item_id"] == td["item_id"].iloc[0]].copy()
    cp["item_id"] = "planted"
    cp["y"] = 2 * src[: len(cp)] + 9
    pd.concat([td, cp]).to_parquet(tdir / "data.parquet")
    RUN.step_crossdup(a, paths, S)
    RUN.step_tasks(a, paths, S)
    RUN.step_package(a, paths, S)
    RUN.step_report_b(a, paths, S)
    rel = RUN.release(a, paths, S)
    assert PK.check_manifest(rel) == []
    man = pd.read_csv(rel / "manifest.csv")
    # no series in both tuning and locked
    ids_by_split = {"tuning": set(), "locked": set()}
    vals_by_split = {"tuning": [], "locked": []}
    for cfg, g in man.groupby("config"):
        s = g["split"].iloc[0]
        t = pq.read_table(rel / ("locked/data" if s == "locked" else "data") / cfg / "data.parquet").to_pydict()
        ids_by_split[s] |= {f"{cfg.split('__')[0]}/{i}" for i in t["id"]}
        vals_by_split[s] += [np.round(np.asarray(y, float)[:64], 3).tobytes() for y in t["y"]]
    assert "planted" not in {i.split("/")[1] for i in ids_by_split["tuning"]}
    assert not set(vals_by_split["tuning"]) & set(vals_by_split["locked"])
    # locked data is not in the card's configs or the tuning task files
    card = (rel / "README.md").read_text()
    assert all(f"data/{c}/" not in card for c in man.loc[man["split"] == "locked", "config"])
    tun_tasks = yaml.safe_load((rel / "tasks" / "full.yaml").read_text())["tasks"]
    assert {t["dataset_config"] for t in tun_tasks} <= set(man.loc[man["split"] == "tuning", "config"])
    lk_tasks = yaml.safe_load((rel / "locked" / "tasks" / "full.yaml").read_text())["tasks"]
    assert all(t["dataset_config"] is None and t["dataset_path"].startswith("locked/data/") for t in lk_tasks)
    # weights: 1 per dataset per tier and split
    for tier in ("full", "core"):
        g = man[man[f"tier_{tier}"]].groupby(["split", "dataset"])[f"weight_{tier}"].sum()
        assert np.allclose(g, 1.0)
    bl = pd.read_csv(rel / "baseline.csv")
    assert len(bl) == len(man) and bl["flag"].isna().all()
    rep = (rel / "BUILD_REPORT.md").read_text()
    assert "Tasks per tier and split" in rep and "Seasonal naive baseline" in rep
    # tampering is caught
    f = next((rel / "data").rglob("data.parquet"))
    f.write_bytes(f.read_bytes() + b"x")
    assert any("checksum" in b for b in PK.check_manifest(rel))
    # rerun skips done work
    before = (paths.data / "tasks" / tuning[1] / "tasks.json").stat().st_mtime_ns
    RUN.step_tasks(a, paths, S)
    assert (paths.data / "tasks" / tuning[1] / "tasks.json").stat().st_mtime_ns == before
    # upload: dry run lists, makes no call
    r = UP.upload(rel, "", S.version, 100.0, dry_run=True, api=object())
    assert r["files"] > 10 and not r["uploaded"]


def test_smoke_forces_locked_and_fev_loads(tmp_path):
    pytest.importorskip("fev")
    paths = Paths(data=tmp_path / "data", small=tmp_path / "small")
    paths.small.mkdir(parents=True)
    ids = [f"sm{i}" for i in range(4)]
    for i, did in enumerate(ids):
        make_canon(paths, did, long_df([400, 420], "D", seed=i), meta(did, "D", domain=f"dom{i}"))
    a = types.SimpleNamespace(only=None, jobs=2, redo=False, smoke=True, dry_run=True, max_gb=100.0)
    for step in ("split", "crossdup", "tasks", "package"):
        getattr(RUN, f"step_{step}")(a, paths, S)
    sp = json.loads((paths.data / "splits_smoke.json").read_text())["datasets"]
    assert sum(v["split"] == "locked" for v in sp.values()) == 2
    assert not (paths.small / "splits.json").exists()
    rel = RUN.release(a, paths, S)
    assert yaml.safe_load((rel / "locked" / "tasks" / "full.yaml").read_text())["tasks"]
    RUN.fev_check(rel)


def test_no_upload_flag(monkeypatch):
    called = []
    for st in RUN.STEPS:
        monkeypatch.setattr(RUN, f"step_{st}", lambda a, p, s, st=st: called.append(st))
    monkeypatch.setattr(RUN, "setup_logging", lambda paths, step: Path("x"))
    RUN.main(["all", "--no-upload"])
    assert called == RUN.STEPS[:-1] and "report_b" in called


# ---- upload guard -------------------------------------------------------------------------------------------------------
class FakeApi:
    def __init__(self, private=True, tags=(), exists=True):
        self.private, self.tags, self.exists, self.calls = private, list(tags), exists, []

    def repo_info(self, repo, repo_type):
        from huggingface_hub.utils import RepositoryNotFoundError
        if not self.exists:
            raise RepositoryNotFoundError("missing", response=types.SimpleNamespace(headers={}, status_code=404, request=None))
        return types.SimpleNamespace(private=self.private)

    def list_repo_refs(self, repo, repo_type):
        return types.SimpleNamespace(tags=[types.SimpleNamespace(name=t) for t in self.tags])

    def create_repo(self, *a, **k):
        self.calls.append("create")
        self.exists = True

    def upload_large_folder(self, **k):
        self.calls.append("upload")

    def create_tag(self, *a, **k):
        self.calls.append("tag")


def _folder(tmp_path):
    (tmp_path / "rel").mkdir()
    (tmp_path / "rel" / "a.txt").write_text("x" * 100)
    return tmp_path / "rel"


def test_upload_refuses_public_repo(tmp_path):
    api = FakeApi(private=False)
    with pytest.raises(UP.Refused, match="public"):
        UP.upload(_folder(tmp_path), "o/b", "v1.0.0", 100, False, api=api)
    assert "upload" not in api.calls


def test_upload_refuses_existing_release_and_size(tmp_path):
    f = _folder(tmp_path)
    with pytest.raises(UP.Refused, match="already has release"):
        UP.upload(f, "o/b", "v1.0.0", 100, False, api=FakeApi(tags=["v1.0.0"]))
    with pytest.raises(UP.Refused, match="max-gb"):
        UP.upload(f, "o/b", "v1.0.0", 1e-8, True, api=FakeApi())


def test_upload_creates_private_and_tags(tmp_path):
    api = FakeApi(exists=False)
    r = UP.upload(_folder(tmp_path), "o/b", "v1.0.0", 100, False, api=api)
    assert r["uploaded"] and api.calls == ["create", "upload", "tag"]
