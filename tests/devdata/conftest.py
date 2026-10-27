"""Shared fixtures: a small horizon table in place of catalog/time_horizons.csv (which comes from TIME on Perlmutter)."""
import pandas as pd
import pytest

from tideforge.devdata import freqs


@pytest.fixture
def horizons(tmp_path, monkeypatch):
    p = tmp_path / "h.csv"
    rows = [(f, t, h) for f, hs in {"h": (48, 96, 168), "D": (7, 14, 30), "W": (4, 8, 13), "30min": (48, 96, 168)}.items()
            for t, h in zip(freqs.TERMS, hs)]
    pd.DataFrame(rows, columns=["freq", "term", "prediction_length"]).to_csv(p, index=False)
    monkeypatch.setattr(freqs, "TIME_HORIZONS", p)
    monkeypatch.setattr(freqs, "HORIZONS_EXTRA", tmp_path / "none.csv")
    freqs.table.cache_clear()
    yield
    freqs.table.cache_clear()
