"""Settings and paths for tfbench. Every threshold lives here (owner OK 2026-11-12).

Paths:
  inputs   tideforge/devdata/ (hf_profile.csv, hf_profile_files.csv[.gz], hf_catalog_v2.csv)
  small    tideforge/devdata/tfbench/ (selection.csv, overrides.yaml, logs/; reports)
  large    $TFB_DATA (default tideforge/devdata/tfbench/data; git-ignored): raw/, refs/, canon/, audit/
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEVDATA = Path(__file__).resolve().parents[1] / "devdata"
TFB = DEVDATA / "tfbench"


@dataclass
class Settings:
    statuses: tuple = ("keep", "keep_filtered")       # catalog status kept in v1 ("check" stays out)
    max_scope_gb: float = 20.0                         # per catalog row
    # rows above max_scope_gb that are kept anyway: repo id, or "repo id|config" for one row
    allow_large: tuple = ("Emulated-Inc/sci-streamflow-ungauged-new-country",)
    max_series: int = 1000                             # per dataset, frozen seeded sample
    oversample: int = 3                                # series loaded before filtering = oversample x max_series
    max_points: int = 50_000                           # last points kept per series
    max_nan_share: float = 0.5                         # series with more NaN than this are dropped
    max_covariates: int = 32                           # per dataset; the first ones in column order, rest recorded
    # minimum context per frequency band (owner 2026-11-12); B1: context >= max(2 x horizon, min_context);
    # "too short" = that + one short horizon. An int applies to every band (tests).
    min_context: dict | int = field(default_factory=lambda: dict(MIN_CONTEXT))
    seed: int = 20261112
    event_step: str = "D"                              # default aggregation step of event tables
    big_csv_bytes: int = 2 * 1024 ** 3                 # csv above this is read in streamed chunks
    disk_margin: float = 1.2                           # free disk needed = repo bytes x margin
    download_jobs: int = 6
    # leakage audit (A5)
    audit_series_share: float = 0.10                   # drop a series above this share of matched windows
    audit_dataset_share: float = 0.01                  # flag a dataset above this share (after the series drops)
    refs: tuple = ("autogluon/fev_datasets", "Salesforce/GiftEval", "Real-TSF/TIME")
    # Milestone B (owner OK 2026-11-12)
    max_windows: int = 20                              # per task
    window_share: float = 0.10                         # windows cover at most this share of a p25-length series
    window_len_quantile: float = 0.25                  # n_windows from this quantile of series lengths
    horizon_min_share: float = 1.0                     # medium / long only if this share of series fits them (1.0: no series loses its short windows)
    agg_coverage: float = 0.80                         # an aggregated bucket needs this share of its points, else NaN
    max_channels: int = 32                             # joint multivariate tasks
    flag_max_distinct: int = 12                        # integer covariates with <= this many values: categorical / flag
    quantile_levels: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    fast_n: dict = field(default_factory=lambda: {"tuning": 150, "locked": 50})
    crossdup_share: float = 0.10                       # cross-dataset copies: the audit's series rule
    version: str = "v1.0.0"
    max_gb: float = 100.0


MIN_CONTEXT = {"subhourly": 256, "hourly": 256, "subdaily": 256, "daily": 128, "weekly": 52, "monthly": 36,
               "quarterly": 16, "yearly": 12}

# Short horizons per frequency band (GIFT-Eval defaults; B1 moves them to horizons.yaml).
SHORT_H = {"subhourly": 48, "hourly": 48, "subdaily": 48, "daily": 30, "weekly": 8, "monthly": 12, "quarterly": 8,
           "yearly": 6}


@dataclass
class Paths:
    data: Path = field(default_factory=lambda: Path(os.environ.get("TFB_DATA", str(TFB / "data"))))
    small: Path = TFB

    @property
    def raw(self) -> Path: return self.data / "raw"
    @property
    def refs(self) -> Path: return self.data / "refs"
    @property
    def canon(self) -> Path: return self.data / "canon"
    @property
    def audit(self) -> Path: return self.data / "audit"
    @property
    def manifest(self) -> Path: return self.data / "manifest"
    @property
    def logs(self) -> Path: return self.small / "logs"
    @property
    def selection(self) -> Path: return self.small / "selection.csv"
    @property
    def overrides(self) -> Path: return self.small / "overrides.yaml"


def profile_path() -> Path:
    return DEVDATA / "hf_profile.csv"


def files_path() -> Path:
    p = DEVDATA / "hf_profile_files.csv"
    return p if p.exists() else DEVDATA / "hf_profile_files.csv.gz"


def catalog_path() -> Path:
    return DEVDATA / "hf_catalog_v2.csv"
