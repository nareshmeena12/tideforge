"""The real-data allow-list: whole-dataset, explicit; the downloader refuses anything else (no network)."""
import importlib.util
import json
from pathlib import Path

import pytest

from tidemark.real import allowlist as AL

SCRIPT = Path(__file__).parents[1] / "scripts" / "real" / "download_chronos.py"


def load_script():
    spec = importlib.util.spec_from_file_location("dl", SCRIPT)
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)
    return dl


def test_allowlist_is_explicit_and_excludes_the_leaky_sets():
    a = AL.allowed()
    assert len(AL.ALLOWED_CHRONOS) == 10 and len(AL.ALLOWED_EXTRA) == 2 and len(AL.HOURLY_CONFIRMED) == 11 and len(a) == 23
    for c in AL.EXCLUDED:
        assert c not in a
    assert sum(c.startswith("weatherbench_hourly") for c in a) == 11
    for c in ("10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature"):
        assert f"weatherbench_hourly_{c}" not in a and f"weatherbench_hourly_{c}" in AL.EXCLUDED
    assert len(AL.EXCLUDED) == 12 and set(AL.HOURLY_CONFIRMED) | {f"weatherbench_hourly_{c}" for c in ("10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature")} == set(AL.HOURLY_CANDIDATES)
    assert a["brazilian_cities_temperature"] == AL.CHRONOS_EXTRA and a["dominick"] == AL.CHRONOS
    assert "training_corpus_tsmixup_10m" not in a and "training_corpus_kernel_synth_1m" not in a


@pytest.mark.parametrize("cfg", ["solar", "taxi_1h", "m4_daily", "training_corpus_tsmixup_10m", "weatherbench_hourly_2m_temperature", "weatherbench_hourly_10m_u_component_of_wind"])
def test_check_refuses(cfg):
    with pytest.raises(SystemExit, match="REFUSED"):
        AL.check(cfg, AL.CHRONOS)


def test_check_refuses_wrong_repo_and_accepts_listed():
    AL.check("dominick", AL.CHRONOS)
    with pytest.raises(SystemExit):
        AL.check("dominick", AL.CHRONOS_EXTRA)


def test_confirmed_hourly_must_be_real_candidates_and_not_excluded(monkeypatch):
    monkeypatch.setattr(AL, "HOURLY_CONFIRMED", ("weatherbench_hourly_geopotential",))
    assert "weatherbench_hourly_geopotential" in AL.allowed()
    monkeypatch.setattr(AL, "HOURLY_CONFIRMED", ("weatherbench_hourly_2m_temperature",))
    with pytest.raises(ValueError):
        AL.allowed()
    monkeypatch.setattr(AL, "HOURLY_CONFIRMED", ("solar",))
    with pytest.raises(ValueError):
        AL.allowed()


def test_downloader_refuses_excluded_before_any_network(tmp_path):
    with pytest.raises(SystemExit, match="REFUSED"):
        load_script().main(["--root", str(tmp_path), "--only", "solar"])
    assert not (tmp_path / "manifest.json").exists()


def test_stats_on_a_tiny_arrow_file(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.ipc as ipc
    tbl = pa.table({"id": ["a", "b"], "timestamp": [[1, 2, 3], [1, 2]], "target": [[1.0, float("nan"), 3.0], [4.0, 5.0]]})
    d = tmp_path / "x"
    d.mkdir()
    with pa.OSFile(str(d / "train.arrow"), "wb") as f, ipc.new_stream(f, tbl.schema) as w:
        w.write_table(tbl)
    s = load_script().stats(d)
    assert s["n_series"] == 2 and s["total_points"] == 5 and s["min_len"] == 2 and s["columns"] == ["target"]
    assert s["nan_fraction"] == pytest.approx(1 / 5) and s["mean_len"] == 2.5


def test_hub_sizes_uses_metadata_only():
    from types import SimpleNamespace as NS
    info = NS(sha="abc", siblings=[NS(rfilename="dominick/a.arrow", size=100), NS(rfilename="dominick/b.arrow", size=50),
                                   NS(rfilename="other/a.arrow", size=7), NS(rfilename="README.md", size=1)])
    out = load_script().hub_sizes({"dominick": AL.CHRONOS}, lambda repo: info, lambda repo, c: 42)
    assert out == {"dominick": {"repo": AL.CHRONOS, "revision": "abc", "files": 2, "file_bytes": 150, "n_rows": 42}}
    info2 = NS(sha="abc", siblings=[NS(rfilename="weatherbench_hourly/geopotential/train-00000-of-00002.parquet", size=9),
                                    NS(rfilename="weatherbench_hourly/geopotential/train-00001-of-00002.parquet", size=1),
                                    NS(rfilename="weatherbench_hourly/vorticity/train-00000-of-00001.parquet", size=5)])
    assert load_script().hub_sizes({"weatherbench_hourly_geopotential": AL.CHRONOS}, lambda r: info2, lambda r, c: None)[
        "weatherbench_hourly_geopotential"]["file_bytes"] == 10


# ---- hourly shards, ordering, failures (fake Hub, tiny parquet shards)
def test_evenly_spaced_and_plan_files():
    dl = load_script()
    assert dl.evenly_spaced(107, 12) == [0, 10, 19, 29, 39, 48, 58, 67, 77, 87, 96, 106]
    assert dl.evenly_spaced(107, 12)[0] == 0 and dl.evenly_spaced(107, 12)[-1] == 106       # spread over 0..total-1, not the first 12
    assert dl.evenly_spaced(9, 12) == list(range(9)) and dl.evenly_spaced(5, 1) == [0] and len(set(dl.evenly_spaced(20, 7))) == 7
    files = [(f"weatherbench_hourly/geopotential/train-{i:05d}-of-00107.parquet", 10 + i) for i in range(107)]
    assert len(dl.plan_files("weatherbench_hourly_geopotential", files, 12)) == 12
    assert len(dl.plan_files("weatherbench_hourly_total_precipitation", files, 12)) == 107       # surface: in full
    assert len(dl.plan_files("dominick", files, 12)) == 107
    assert dl.hub_prefix("weatherbench_hourly_geopotential") == "weatherbench_hourly/geopotential" and dl.hub_prefix("dominick") == "dominick"


def test_run_order_extra_before_hourly():
    names = load_script().order(["weatherbench_hourly_geopotential", "brazilian_cities_temperature", "dominick", "ushcn_daily"])
    assert names == ["dominick", "ushcn_daily", "brazilian_cities_temperature", "weatherbench_hourly_geopotential"]


def test_extrapolation_is_size_weighted_and_marked_as_an_estimate():
    dl = load_script()
    sh = [{"size_bytes": 100, "total_points": 1000, "n_series": 10}, {"size_bytes": 50, "total_points": 500, "n_series": 5}]
    e = dl.extrapolate(sh, 4, 300)
    assert e["est_full_points_by_bytes"] == pytest.approx(3000) and e["est_full_series_by_bytes"] == pytest.approx(30)
    assert e["est_full_points_by_shard_mean"] == pytest.approx(3000) and "ESTIMATE" in e["note"]


def _fake_hub(tmp_path, monkeypatch, n_files=6, small=("total_precipitation",)):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    from types import SimpleNamespace as NS
    sib = []
    for v in ("geopotential", "total_precipitation"):
        n = n_files if v not in small else 3
        sib += [NS(rfilename=f"weatherbench_hourly/{v}/train-{i:05d}-of-{n:05d}.parquet", size=1000 + i) for i in range(n)]
    sib.append(NS(rfilename="dominick/train-00000-of-00001.parquet", size=500))
    calls = []

    class Api:
        def dataset_info(self, repo, files_metadata=False):
            return NS(sha="rev123", siblings=sib)

    def snap(repo_id, repo_type, revision, allow_patterns, local_dir, max_workers):
        calls.append(list(allow_patterns))
        for f in allow_patterns:
            p = Path(local_dir) / f
            p.parent.mkdir(parents=True, exist_ok=True)
            i = int(Path(f).stem.split("-")[1])
            pq.write_table(pa.table({"timestamp": [[1, 2, 3]] * (i + 1), "target": [[1.0, 2.0, float("nan")]] * (i + 1)}), p)
    return Api(), snap, calls


def test_do_dataset_hourly_takes_spaced_shards_records_them_and_resumes(tmp_path, monkeypatch):
    dl = load_script()
    api, snap, calls = _fake_hub(tmp_path, monkeypatch)
    man = {"datasets": {}, "revisions": {}}
    args = type("A", (), dict(hourly_shards=3, force=False, stats_only=False, workers=1))()
    c = "weatherbench_hourly_geopotential"
    dl.do_dataset(c, AL.CHRONOS, man, tmp_path, args, {}, api, snap)
    e = man["datasets"][c]
    assert e["shards"]["total"] == 6 and e["shards"]["taken"] == [0, 3, 5] and len(calls[0]) == 3
    assert e["n_series"] == 1 + 4 + 6 and e["total_points"] == 3 * 11 and e["shards"]["full_variable"]["est_full_points_by_bytes"] > 0
    assert (tmp_path / "manifest.json").exists() and e["revision"] == "rev123"
    assert not dl.needs_more_shards(c, e, 3) and dl.needs_more_shards(c, e, 5)
    # more shards later: only the new ones are fetched, the old ones keep their numbers
    args.hourly_shards = 5
    dl.do_dataset(c, AL.CHRONOS, man, tmp_path, args, {}, api, snap)
    assert len(calls[1]) == 2 and set(calls[1]).isdisjoint(set(calls[0])) and man["datasets"][c]["shards"]["taken"] == [0, 1, 3, 4, 5]
    # a surface variable is taken in full
    dl.do_dataset("weatherbench_hourly_total_precipitation", AL.CHRONOS, man, tmp_path, args, {}, api, snap)
    assert man["datasets"]["weatherbench_hourly_total_precipitation"]["files"] == 3 and len(calls[2]) == 3


def test_a_failing_dataset_does_not_stop_the_rest(tmp_path, monkeypatch, capsys):
    dl = load_script()
    api, snap, calls = _fake_hub(tmp_path, monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("kaggle credentials missing")
    monkeypatch.setattr(dl, "fetch_extra", boom)
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: api)
    monkeypatch.setattr("huggingface_hub.snapshot_download", snap)
    rc = dl.main(["--root", str(tmp_path), "--only", "brazilian_cities_temperature,dominick,weatherbench_hourly_total_precipitation"])
    out = capsys.readouterr().out
    assert rc == 1 and "brazilian_cities_temperature: RuntimeError" in out
    man = json.loads((tmp_path / "manifest.json").read_text())
    assert set(man["datasets"]) == {"dominick", "weatherbench_hourly_total_precipitation"}      # both ran after the extra failed
    assert "planned:" in out
