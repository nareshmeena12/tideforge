"""The ONLY real datasets the post-training track may use. Whole-dataset rule: a dataset is on this list or it is excluded;
there is no name matching, no "everything except", no partial filtering of a dataset's series.

Sources: Hugging Face autogluon/chronos_datasets and autogluon/chronos_datasets_extra. No TSMixup, no KernelSynth.
TIME is a separate benchmark and is not in this corpus.
"""
from __future__ import annotations

CHRONOS = "autogluon/chronos_datasets"
CHRONOS_EXTRA = "autogluon/chronos_datasets_extra"

ALLOWED_CHRONOS = (
    "dominick", "exchange_rate", "mexico_city_bikes", "monash_rideshare", "ushcn_daily",
    "weatherbench_daily", "weatherbench_weekly",
    "wiki_daily_100k", "wind_farms_daily", "wind_farms_hourly",
)
ALLOWED_EXTRA = ("brazilian_cities_temperature", "spanish_energy_and_weather")

# The Hub lists these 14 weatherbench_hourly_* configs (checked 2026-10-24). The owner confirmed (2026-10-24) the 11 that appear in
# TiRex-2 Appendix E, Table 5; the other 3 are excluded (see EXCLUDED).
HOURLY_CANDIDATES = (
    "weatherbench_hourly_10m_u_component_of_wind", "weatherbench_hourly_10m_v_component_of_wind",
    "weatherbench_hourly_2m_temperature", "weatherbench_hourly_geopotential", "weatherbench_hourly_potential_vorticity",
    "weatherbench_hourly_relative_humidity", "weatherbench_hourly_specific_humidity", "weatherbench_hourly_temperature",
    "weatherbench_hourly_toa_incident_solar_radiation", "weatherbench_hourly_total_cloud_cover",
    "weatherbench_hourly_total_precipitation", "weatherbench_hourly_u_component_of_wind",
    "weatherbench_hourly_v_component_of_wind", "weatherbench_hourly_vorticity",
)
HOURLY_CONFIRMED: tuple[str, ...] = tuple(
    f"weatherbench_hourly_{v}" for v in (
        "geopotential", "potential_vorticity", "relative_humidity", "specific_humidity", "temperature",
        "toa_incident_solar_radiation", "total_cloud_cover", "total_precipitation", "u_component_of_wind",
        "v_component_of_wind", "vorticity"))

# The 3 surface variables are small (about 19 GB together) and always taken in full; the other 8 are the pressure-level variables.
SURFACE_HOURLY = ("weatherbench_hourly_toa_incident_solar_radiation", "weatherbench_hourly_total_cloud_cover",
                  "weatherbench_hourly_total_precipitation")
PRESSURE_HOURLY = tuple(c for c in HOURLY_CONFIRMED if c not in SURFACE_HOURLY)

# Explicitly excluded (leakage with fev-bench / GIFT-Eval per the TiRex-2 tables, or dropped for safety). Everything else in the
# two repos is excluded too, simply by not being allowed; this list is only what the audit names in its error message.
EXCLUDED = (
    "solar", "solar_1h", "monash_traffic", "taxi_30min", "taxi_1h", "uber_tlc_daily", "uber_tlc_hourly",
    "monash_london_smart_meters", "monash_pedestrian_counts",
    # absent from TiRex-2 Appendix E, Table 5 (owner: absence means TiRex-2 excluded them, presumably for overlap; presence means kept)
    "weatherbench_hourly_10m_u_component_of_wind", "weatherbench_hourly_10m_v_component_of_wind",
    "weatherbench_hourly_2m_temperature",
)


def allowed() -> dict[str, str]:
    """{config: repo} of everything that may be downloaded and used right now."""
    bad = [c for c in HOURLY_CONFIRMED if c not in HOURLY_CANDIDATES]
    if bad:
        raise ValueError(f"HOURLY_CONFIRMED has names that are not hourly weatherbench configs on the Hub: {bad}")
    out = {c: CHRONOS for c in ALLOWED_CHRONOS}
    out.update({c: CHRONOS for c in HOURLY_CONFIRMED})
    out.update({c: CHRONOS_EXTRA for c in ALLOWED_EXTRA})
    clash = set(out) & set(EXCLUDED)
    if clash:
        raise ValueError(f"allowed and excluded overlap: {sorted(clash)}")
    return out


def check(config: str, repo: str) -> None:
    """Raises unless (config, repo) is on the allow-list."""
    if allowed().get(config) != repo:
        why = "explicitly excluded" if config in EXCLUDED else "not on the allow-list"
        raise SystemExit(f"REFUSED: {repo}:{config} is {why}")


HOURLY_PREFIX = "weatherbench_hourly_"


def is_hourly(config: str) -> bool:
    return config.startswith(HOURLY_PREFIX)


def hub_prefix(config: str) -> str:
    """Folder of a config inside its Hub repo: weatherbench_hourly_<v> -> weatherbench_hourly/<v>; everything else <config>."""
    return f"weatherbench_hourly/{config[len(HOURLY_PREFIX):]}" if is_hourly(config) else config
