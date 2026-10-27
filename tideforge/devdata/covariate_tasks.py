"""Covariate tasks from tideforge/devdata/covariate_tasks.yaml (DATA_COLLECTION.md step 7).

    python -m tideforge.devdata.covariate_tasks stations --raw $TF_DATA/raw --census-gazetteer G --census-popest P
        # picks the weather stations of every task and downloads their GHCN-Hourly files (login / transfer node)
    python -m tideforge.devdata.covariate_tasks build --raw $TF_DATA/raw --staging $TF_DATA/staging \
        --census-gazetteer G --census-popest P [--tasks eia_load_temperature,...]

Output <staging>/covtasks/<task>/<freq>.parquet: series_id, timestamp (UTC), target, one column per covariate; meta.json with
the target dataset, its split unit and role, and per covariate: role (known / past), oracle, source and how it was made.

Census inputs (downloaded on Perlmutter, not committed): the Gazetteer places file (tab-separated: USPS, NAME, INTPTLAT,
INTPTLONG, ...) and the population estimates file for incorporated places (csv: STNAME, NAME, SUMLEV, POPESTIMATE<year>).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tideforge.devdata import split as S
from tideforge.devdata.catalog import REPO
from tideforge.devdata.freqs import canonical
from tideforge.devdata.schema import load_sources

TASKS = Path(__file__).resolve().parent / "covariate_tasks.yaml"
STATE_NAMES = {"AL": "Alabama", "CA": "California", "CT": "Connecticut", "DC": "District of Columbia", "FL": "Florida",
               "GA": "Georgia", "IL": "Illinois", "IN": "Indiana", "KS": "Kansas", "LA": "Louisiana", "MA": "Massachusetts",
               "MD": "Maryland", "MI": "Michigan", "MN": "Minnesota", "MO": "Missouri", "NC": "North Carolina",
               "NE": "Nebraska", "NY": "New York", "OH": "Ohio", "OK": "Oklahoma", "PA": "Pennsylvania", "RI": "Rhode Island",
               "SC": "South Carolina", "TN": "Tennessee"}


def load_tasks(path: Path = TASKS) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------------- pure helpers
def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def nearest(stations: pd.DataFrame, lat: float, lon: float, k: int, max_km: float) -> pd.DataFrame:
    """The k stations (id, lat, lon) nearest to a point, within max_km, nearest first."""
    d = haversine_km(lat, lon, stations["lat"].to_numpy(float), stations["lon"].to_numpy(float))
    out = stations.assign(km=d)
    return out[out["km"] <= max_km].sort_values("km").head(k)


def resolve_cities(cities: pd.DataFrame, gazetteer: Path, popest: Path) -> pd.DataFrame:
    """Add lat, lon, pop and the population weight within each BA. A city matches a Census place of its state whose name is
    the city name plus a place suffix ('Philadelphia city', 'Nashville-Davidson metropolitan government (balance)');
    the most populous match wins. Unmatched cities raise."""
    gz = pd.read_csv(gazetteer, sep="\t", dtype=str)
    gz.columns = [c.strip() for c in gz.columns]
    pe = pd.read_csv(popest, dtype=str, encoding="latin-1")
    popcol = sorted(c for c in pe.columns if c.startswith("POPESTIMATE"))[-1]
    pe = pe[pe["SUMLEV"].isin(["162", "170", "172"])]
    rows = []
    for _, r in cities.iterrows():
        st = STATE_NAMES[r["state"]]
        name = r["city"]
        ok = lambda s: s.str.match(rf"^{name}(-[A-Za-z]+)? (city|town|village|municipality|metropolitan government|unified government|consolidated government)")
        g = gz[(gz["USPS"] == r["state"]) & ok(gz["NAME"])]
        p = pe[(pe["STNAME"] == st) & ok(pe["NAME"])]
        if g.empty or p.empty:
            raise KeyError(f"{r['ba']}: city {name}, {r['state']} not found in the Census files")
        p = p.assign(pop=pd.to_numeric(p[popcol])).sort_values("pop", ascending=False).iloc[0]
        g = g.iloc[0]
        rows.append(dict(r, lat=float(g["INTPTLAT"]), lon=float(g["INTPTLONG"]), pop=float(p["pop"]), census_name=p["NAME"]))
    out = pd.DataFrame(rows)
    out["weight"] = out["pop"] / out.groupby("ba")["pop"].transform("sum")
    return out


def weighted_mean(frame: pd.DataFrame, weights: pd.Series, min_weight: float) -> pd.Series:
    """Per timestamp: sum(w * x) / sum(w) over the columns with data; NaN where those columns hold < min_weight."""
    w = weights.reindex(frame.columns).to_numpy(float)
    x = frame.to_numpy(float)
    have = np.isfinite(x)
    wsum = (have * w).sum(1)
    val = np.where(have, x, 0) @ w
    return pd.Series(np.where(wsum >= min_weight, val / np.where(wsum > 0, wsum, 1), np.nan), index=frame.index)


def holiday_flags(ts: pd.DatetimeIndex, tz: str | None, country: str, subdiv: str | None = None) -> np.ndarray:
    """1.0 where the LOCAL date of a UTC timestamp is a public holiday."""
    import holidays
    local = pd.DatetimeIndex(ts).tz_localize("UTC").tz_convert(tz) if tz else pd.DatetimeIndex(ts)
    years = sorted(set(local.year))
    cal = holidays.country_holidays(country, subdiv=subdiv, years=years)
    return np.array([d in cal for d in local.date], dtype=float)


# ---------------------------------------------------------------------------------------------------- inputs
def staged(staging: Path, slug: str, freq: str) -> pd.DataFrame:
    return pd.read_parquet(Path(staging) / slug / f"{canonical(freq)}.parquet")


def station_table(raw: Path, src, spec: dict) -> pd.DataFrame:
    """The station list of a weather source's index (id, lat, lon)."""
    from tideforge.devdata import fetch as F
    from tideforge.devdata.loaders.tabular import _frames
    paths = [Path(raw) / src.slug / "index" / F.local_name(u) for u, _ in F.file_list(src.index)]
    df = pd.concat([d for p in paths for d in _frames(p, src.index.get("read", {"format": "csv"}))], ignore_index=True)
    return pd.DataFrame({"id": df[spec["id"]].astype(str).str.strip(), "lat": pd.to_numeric(df[spec["lat"]], errors="coerce"),
                         "lon": pd.to_numeric(df[spec["lon"]], errors="coerce")}).dropna()


def station_series(raw: Path, src, ids: list[str], column: str, freq: str) -> pd.DataFrame:
    """Hourly (freq) values of `column` for the given stations: DataFrame time x station (UTC)."""
    from tideforge.devdata import fetch as F
    from tideforge.devdata import loaders
    files = [(Path(raw) / src.slug / "files" / F.local_name(u), v) for u, v in F.file_list(src.fetch, ids)]
    files = [(p, v) for p, v in files if p.exists()]
    df = loaders.read_files(src, files)
    df = df[df["series_id"].str.endswith("/" + column)]
    df["station"] = df["series_id"].str.split("/").str[0]
    return df.pivot_table(index="timestamp", columns="station", values="value", aggfunc="mean").resample(freq).mean()


# ---------------------------------------------------------------------------------------------------- joins
def plan_stations(task: dict, raw: Path, srcs, cities: pd.DataFrame | None) -> dict[str, list[str]]:
    """{point name: [station ids]} for a weather task."""
    src = srcs[task["station_source"]]
    st = station_table(raw, src, task["station_index"])
    if task["join"] == "population_weighted_weather":
        return {f"{r['ba']}/{r['city']}": nearest(st, r["lat"], r["lon"], 1, task["max_km"])["id"].tolist()
                for _, r in cities.iterrows()}
    pt = task["point"]
    return {"point": nearest(st, pt["lat"], pt["lon"], int(task.get("n_stations", 1)), task["max_km"])["id"].tolist()}


def build_task(name: str, task: dict, raw: Path, staging: Path, srcs, cities: pd.DataFrame | None = None) -> dict:
    tgt_src = srcs[task["target"]["source"]]
    freq = task["target"]["freq"]
    tgt = staged(staging, tgt_src.slug, freq)
    cov_meta = {}
    if task["join"] == "population_weighted_weather":
        plan = plan_stations(task, raw, srcs, cities)
        col = task["covariates"]["temperature"]["column"]
        ids = sorted({s for v in plan.values() for s in v})
        temps = station_series(raw, srcs[task["station_source"]], ids, col, freq)
        parts = []
        for ba, g in cities.groupby("ba"):
            if ba not in set(tgt["series_id"]):
                continue
            frame = pd.DataFrame({f"{ba}/{c}": temps[plan[f"{ba}/{c}"][0]] if plan[f"{ba}/{c}"] else np.nan
                                  for c in g["city"]}, index=temps.index)
            w = pd.Series(g["weight"].to_numpy(), index=[f"{ba}/{c}" for c in g["city"]])
            t = tgt[tgt["series_id"] == ba].set_index("timestamp")["value"]
            cov = weighted_mean(frame, w, task.get("min_weight", 0.5)).reindex(t.index)
            parts.append(pd.DataFrame({"series_id": ba, "timestamp": t.index, "target": t.to_numpy(), "temperature": cov.to_numpy()}))
        out = pd.concat(parts, ignore_index=True)
        cov_meta["temperature"] = dict(task["covariates"]["temperature"], source=task["station_source"],
                                       how="population-weighted mean of the nearest station to each city", stations=plan,
                                       cities=cities.to_dict("records"))
    elif task["join"] == "camels_forcing":
        from tideforge.devdata import fetch as F
        from tideforge.devdata.loaders.adapters import camels_us
        z = Path(raw) / tgt_src.slug / "files" / F.local_name(tgt_src.fetch["urls"][0])
        frc = camels_us.read_forcing(z, set(tgt["series_id"]))
        cols = list(task["covariates"])
        frc = frc.rename(columns={"gauge": "series_id"})[["series_id", "timestamp"] + cols]
        out = tgt.rename(columns={"value": "target"}).merge(frc, on=["series_id", "timestamp"], how="left")
        cov_meta = {c: dict(v, source=task["target"]["source"], how="basin-mean Daymet forcing")
                    for c, v in task["covariates"].items()}
    elif task["join"] in ("calendar", "point_weather"):
        out = tgt.rename(columns={"value": "target"})
        if task["join"] == "point_weather":
            plan = plan_stations(task, raw, srcs, None)
            col = task["covariates"]["temperature"]["column"]
            temps = station_series(raw, srcs[task["station_source"]], plan["point"], col, freq)
            temp = temps.mean(axis=1) if len(temps.columns) else pd.Series(dtype=float)
            out["temperature"] = temp.reindex(out["timestamp"]).to_numpy()
            cov_meta["temperature"] = dict(task["covariates"]["temperature"], source=task["station_source"],
                                           how=f"mean of the {len(plan['point'])} nearest stations", stations=plan)
        hol = task["covariates"]["holiday"]
        ts = pd.DatetimeIndex(out["timestamp"])
        out["holiday"] = holiday_flags(ts, task.get("tz") if canonical(freq) not in ("D", "W") else None,
                                       hol["country"], hol.get("subdiv"))
        cov_meta["holiday"] = dict(hol, how="python holidays package, local date")
    else:
        raise ValueError(f"{name}: unknown join {task['join']!r}")
    sp = S.load().set_index("dataset_id").loc[task["target"]["source"]]
    meta = dict(task=name, target=task["target"], target_slug=tgt_src.slug, split_unit=sp["unit_id"], role=sp["role"],
                covariates=cov_meta, n_series=int(out["series_id"].nunique()), n_points=int(len(out)),
                cov_nan_share={c: float(out[c].isna().mean()) for c in cov_meta}, notes=task.get("notes", ""))
    d = Path(staging) / "covtasks" / name
    d.mkdir(parents=True, exist_ok=True)
    out.to_parquet(d / f"{canonical(freq)}.parquet", index=False)
    (d / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    return meta


# ---------------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stations", "build"])
    ap.add_argument("--raw", required=True)
    ap.add_argument("--staging")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--census-gazetteer")
    ap.add_argument("--census-popest")
    a = ap.parse_args(argv)
    srcs, _ = load_sources()
    tasks = load_tasks()
    names = list(tasks) if a.tasks == "all" else a.tasks.split(",")
    cities = None
    if any(tasks[n]["join"] == "population_weighted_weather" for n in names):
        if not (a.census_gazetteer and a.census_popest):
            raise SystemExit("--census-gazetteer and --census-popest are needed for the population-weighted tasks")
        cities = resolve_cities(pd.read_csv(REPO / tasks["eia_load_temperature"]["cities"]), Path(a.census_gazetteer),
                                Path(a.census_popest))
    for n in names:
        t = tasks[n]
        if a.cmd == "stations":
            if "station_source" not in t:
                continue
            from tideforge.devdata.ingest import Run
            plan = plan_stations(t, Path(a.raw), srcs, cities)
            ids = sorted({s for v in plan.values() for s in v})
            missing = [k for k, v in plan.items() if not v]
            print(f"{n}: {len(ids)} stations; points without a station within {t['max_km']} km: {missing}")
            run = Run(srcs[t["station_source"]], Path(a.raw))
            from tideforge.devdata import fetch as F
            F.fetch_all(F.file_list(run.src.fetch, ids), run.dir / "files", run.src.fetch, run.log)
        else:
            m = build_task(n, t, Path(a.raw), Path(a.staging), srcs, cities)
            print(f"{n}: {m['n_series']} series, {m['n_points']:,} points, role {m['role']}, covariate NaN {m['cov_nan_share']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
