"""Conversion of downloaded datasets to memmap shards: missing values stay missing, only valid points count."""
import json

import numpy as np
import pytest

from tidemark.real import convert as CV
from tidemark.real import store as ST

nan = float("nan")


def write_arrow(path, rows):
    pa = pytest.importorskip("pyarrow")
    tbl = pa.table({"id": [str(i) for i in range(len(rows))], "timestamp": [list(range(len(r))) for r in rows], "target": rows})
    with pa.OSFile(str(path), "wb") as f, pa.ipc.new_stream(f, tbl.schema) as w:
        w.write_table(tbl)


def test_clean_series_forward_fills_and_remembers_what_was_missing():
    a = np.array([nan, nan, 1.0, nan, 3.0, np.inf, 5.0, nan])
    out, miss, nv = ST.clean_series(a)
    assert nv == 3 and out.dtype == np.float32
    assert list(out) == [1.0, 1.0, 3.0, 3.0, 5.0]                   # trimmed to first / last valid, interior gaps forward-filled
    assert list(miss) == [1, 3]                                       # positions (in the trimmed series) that were missing
    assert ST.clean_series(np.array([nan, nan]))[0] is None
    big, _, _ = ST.clean_series(np.array([1e12, -1e12, 5.0]))
    assert big.max() == 1e9 and big.min() == -1e9


def test_convert_counts_valid_points_only_and_roundtrips(tmp_path):
    ok = [float(i) for i in range(40)]
    holes = [nan] * 3 + [float(i) for i in range(50)] + [nan] * 2                      # trimmed to 50 stored
    holes[10] = nan
    holes[20] = nan                                                                       # 2 interior holes
    short = [1.0] * 10                                                                    # dropped (< 33 valid)
    allnan = [nan] * 50
    mostly = [1.0, nan, 2.0] + [nan] * 40                                                 # 2 valid -> dropped
    f = tmp_path / "train.arrow"
    write_arrow(f, [ok, holes, short, allnan, mostly])
    m = ST.convert_dataset("toy", [f], tmp_path / "shards" / "toy")
    assert m["n_series"] == 2 and m["series_in"] == 5 and m["dropped_short"] == 2 and m["dropped_empty"] == 1
    assert m["unique_points"] == 40 + 48 and m["stored_points"] == 40 + 50 and m["nan_fraction_stored"] == pytest.approx(2 / 90)
    store = ST.RealStore(tmp_path / "shards", ["toy"])
    assert store.stored_points() == {"toy": 90}
    v = np.asarray(store.vals[0][store.offs[0][1]:store.offs[0][1] + 50])
    assert np.isfinite(v).all()                                                          # filled: no NaN in the stored inputs
    assert list(store.miss[0]) == [40 + 7, 40 + 17]                                      # global positions of the two holes (10 - 3, 20 - 3)
    assert list(store.missing_in(0, 40, 90)) == [7, 17] and list(store.missing_in(0, 0, 40)) == []
    idx = ST.write_index(tmp_path / "shards")
    assert idx["datasets"]["toy"]["total_points"] == 88 and idx["datasets"]["toy"]["nan_fraction"] == pytest.approx(2 / 90)


def test_convert_cli_uses_the_manifest_and_the_hourly_shards(tmp_path, capsys):
    root = tmp_path / "real"
    d = root / "raw" / "chronos_datasets" / "dominick"
    d.mkdir(parents=True)
    write_arrow(d / "train-00000-of-00001.arrow", [[float(i) for i in range(60)]])
    h = root / "raw" / "chronos_datasets" / "weatherbench_hourly" / "geopotential"
    h.mkdir(parents=True)
    write_arrow(h / "train-00000-of-00003.arrow", [[float(i) for i in range(70)]])
    write_arrow(h / "train-00002-of-00003.arrow", [[float(i) for i in range(80)]])
    write_arrow(h / "train-00001-of-00003.arrow", [[float(i) for i in range(999)]])          # not taken: must not be read
    man = {"datasets": {
        "dominick": {"repo": "autogluon/chronos_datasets"},
        "weatherbench_hourly_geopotential": {"repo": "autogluon/chronos_datasets", "shards": {"per_shard": {
            "weatherbench_hourly/geopotential/train-00000-of-00003.arrow": {}, "weatherbench_hourly/geopotential/train-00002-of-00003.arrow": {}}}}}}
    (root / "manifest.json").write_text(json.dumps(man))
    assert CV.main(["--root", str(root)]) == 0
    idx = json.loads((root / "shards" / "index.json").read_text())["datasets"]
    assert idx["dominick"]["total_points"] == 60 and idx["weatherbench_hourly_geopotential"]["total_points"] == 150
    assert CV.main(["--root", str(root)]) == 0 and "skipped" in capsys.readouterr().out                 # resumable
    man["datasets"]["solar"] = {"repo": "autogluon/chronos_datasets"}
    (root / "manifest.json").write_text(json.dumps(man))
    assert not (root / "shards" / "solar").exists()                                      # not allow-listed: ignored by default
    with pytest.raises(SystemExit):
        CV.main(["--root", str(root), "--only", "solar"])
