"""The committed horizon table: TIME's sub-daily rows as written, tideforge's convention for D / W / M / Q / B / Y; the
3 x horizon eligibility rule; skipped terms are counted at build time."""
import numpy as np
import pandas as pd

from tideforge.devdata import build as B
from tideforge.devdata import freqs


def test_committed_table():
    freqs.table.cache_clear()
    assert freqs.horizons("D") == {"short": 30, "medium": 90, "long": 180}       # not TIME's one-dataset 30 / 7 / 14
    assert freqs.horizons("W-SUN") == {"short": 13, "medium": 26, "long": 52}
    assert freqs.horizons("MS") == {"short": 12, "medium": 18, "long": 24}
    assert freqs.horizons("Q") == {"short": 4, "medium": 8}
    assert freqs.horizons("h") == {"short": 24, "medium": 72, "long": 168}          # TIME, as written
    assert freqs.horizons("15min") == {"short": 12, "medium": 96, "long": 288}
    assert freqs.min_length("D") == 3 * 30 + 64
    assert freqs.eligible(90, 30) and not freqs.eligible(89, 30)


def test_build_counts_skipped_terms():
    freqs.table.cache_clear()
    t = pd.date_range("2020", periods=300, freq="D")
    df = pd.DataFrame({"series_id": "a", "timestamp": t, "value": np.arange(300.0)})
    sk = {}
    w = B.task_windows(df, "D", skips=sk)
    assert set(w["term"]) == {"short", "medium"} and sk == {"long": 1}            # 300 >= 3 x 90, < 3 x 180


def test_index_frequency(tmp_path, monkeypatch):
    """Index time (owner 2026-10-31): season 1, horizons 10 / 25 / 50, the synthetic 1-s clock."""
    import pandas as pd
    from tideforge.devdata import freqs as F
    from tideforge.devdata.ingest import regularise
    assert F.canonical("index") == F.canonical("INDEX") == "index" and F.seasonality("index") == 1
    x = pd.read_csv(F.HORIZONS_EXTRA, dtype={"freq": str, "term": str})
    assert x[x["freq"] == "index"].set_index("term")["prediction_length"].to_dict() == {"short": 10, "medium": 25, "long": 50}
    tb = tmp_path / "time.csv"                                   # TIME's table comes from Perlmutter: a stand-in row
    pd.DataFrame([("h", "short", 48)], columns=["freq", "term", "prediction_length"]).to_csv(tb, index=False)
    monkeypatch.setattr(F, "TIME_HORIZONS", tb)
    F.table.cache_clear()
    try:
        assert F.horizons("index") == {"short": 10, "medium": 25, "long": 50} and F.min_length("index") == 94
        assert F.eligible(30, 10) and not F.eligible(29, 10)
    finally:
        F.table.cache_clear()
    t = F.index_timestamps(5)
    assert t[0] == pd.Timestamp("1970-01-01") and (t[1] - t[0]) == pd.Timedelta(seconds=1)
    g = pd.DataFrame({"timestamp": t.delete(2), "value": [1.0, 2.0, 4.0, 5.0]})
    r = regularise(g, "index", "mean", False)
    assert len(r) == 5 and r.isna().tolist() == [False, False, True, False, False]
