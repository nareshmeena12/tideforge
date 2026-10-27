"""tideforge/devdata/sources.yaml: one entry per catalog dataset, read into Source and validated before anything is fetched.

Entry (keys not listed are errors; see sources.yaml for real ones):

    D053:
      slug: ghcn_daily                       # [a-z0-9_]+, unique; data/<slug>/
      loader: table                          # table (generic) | adapter:<name> (tideforge/devdata/loaders/adapters/<name>.py)
      index:                                 # optional: list the series before fetching, then fetch only the selected ones
        url: https://.../ghcnd-stations.txt
        read: {format: fwf, colspecs: [[0, 11], [38, 40]], names: [station, state]}
        id: station                          # the series key in the index (url_template var {series})
        strata: [country]                    # columns used to stratify the 500-series sample
        derive: {country: "station.str[:2]"} # optional pandas-eval'd columns
        filter: {...}                        # same as below
      fetch:
        urls: [...]                          # or url_template + vars
        url_template: https://.../{series}.csv.gz
        vars: {series: index, year: {range: [2015, 2024]}, month: {months: ["2019-01", "2024-12"], fmt: "%Y%m"}}
        headers: {User-Agent: "tideforge research (contact via repo owner)"}
        auth: {env: EIA_API_KEY, param: api_key}   # or {env: ..., header: X-Api-Key}; the value is read from the environment
        rate_limit_s: 0.5
      read: {format: csv, ...}               # csv | parquet | json | whitespace | fwf; pandas options; zip_member: regex
      time: {col: DATE, format: "%Y%m%d", tz: UTC}      # or cols: [date, hour]; unit: s for epoch numbers
      id: [STATION, ELEMENT]                 # series id = these columns joined with "/"
      value: DATA_VALUE                      # or a list of columns (each a series, id suffix "/<column>")
      events: {how: count}                   # rows are events: count per bin (or {how: sum, value: col})
      filter: {ELEMENT: [TMAX, TMIN]}        # keep rows whose column value is in the list
      missing: [-9999]                       # sentinel values -> NaN
      scale: 0.1
      tasks: [{freq: D, how: mean}]          # one output per frequency (how: mean | sum | count | last | max | min)
      select: {max_series: 500, strata: [country], oversample: 1.3}
      exclude: {where: "country != 'SN'", rule: "...", aggregator: [D078]}   # where = rows to KEEP (pandas query)
      tz: UTC                                # timestamps are converted to UTC naive at ingest
      covariate_only: false                  # true: ingested as a covariate source, not a dev target
      confirm: ["the by-station URL pattern"]   # what the probe step must confirm before a full fetch
      notes: ...
    batches: {batch1: [D053, ...]}

Domain and licence come from the catalog. Checks (validate): the id is a counted catalog dataset with open access (or an
auth env var), not excluded; a `mixed` catalog row has an exclude rule; an aggregator (catalog/aggregator_rules.csv) lists
every covered source in exclude.aggregator; auth names an environment variable only, and no key that looks like a secret
holds a value.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from tideforge.devdata import catalog as cat
from tideforge.devdata.freqs import canonical

SOURCES = Path(__file__).resolve().parent / "sources.yaml"
KEYS = {"slug", "loader", "index", "fetch", "read", "time", "id", "value", "events", "filter", "missing", "scale", "tasks",
        "select", "exclude", "tz", "covariate_only", "confirm", "notes"}
HOWS = {"mean", "sum", "count", "last", "max", "min"}
FORMATS = {"csv", "parquet", "json", "whitespace", "fwf"}
SECRET_KEY = re.compile(r"(token|secret|password|passwd|api[_-]?key|auth[_-]?key)", re.I)
ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")


@dataclass
class Source:
    id: str
    slug: str
    loader: str = "table"
    index: dict | None = None
    fetch: dict = field(default_factory=dict)
    read: dict = field(default_factory=dict)
    time: dict = field(default_factory=dict)
    id_cols: list = field(default_factory=list)
    value: str | list | None = None
    events: dict | None = None
    filter: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)
    scale: float | None = None
    tasks: list = field(default_factory=list)
    select: dict = field(default_factory=dict)
    exclude: dict = field(default_factory=dict)
    tz: str = "UTC"
    covariate_only: bool = False
    confirm: list = field(default_factory=list)
    notes: str = ""

    @property
    def row(self):
        return cat.catalog().loc[self.id]

    @property
    def domain(self) -> str:
        return self.row["domain"]

    @property
    def licence(self) -> str:
        return self.row["licence"]

    @property
    def max_series(self) -> int:
        return int(self.select.get("max_series", 500))


def from_dict(i: str, d: dict) -> Source:
    bad = sorted(set(d) - KEYS)
    if bad:
        raise KeyError(f"{i}: unknown key(s) {bad}")
    d = dict(d)
    ids = d.pop("id", [])
    return Source(id=i, id_cols=[ids] if isinstance(ids, str) else list(ids or []), **d)


def load_sources(path: Path | str = SOURCES) -> tuple[dict[str, Source], dict[str, list[str]]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    batches = raw.pop("batches", {}) or {}
    srcs = {i: from_dict(i, d) for i, d in raw.items()}
    for b, ids in batches.items():
        unknown = sorted(set(ids) - set(srcs))
        if unknown:
            raise KeyError(f"batch {b} names sources not in the file: {unknown}")
    return srcs, batches


def resolve_ids(spec: str, srcs: dict, batches: dict) -> list[str]:
    """'batch1' | 'D053,D054' | 'all'."""
    if spec == "all":
        return sorted(srcs)
    out = []
    for s in spec.split(","):
        s = s.strip()
        out += batches[s] if s in batches else [s]
    unknown = sorted(set(out) - set(srcs))
    if unknown:
        raise KeyError(f"not in sources.yaml: {unknown}")
    return out


def _walk(d, path=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _walk(v, f"{path}.{k}")
            yield f"{path}.{k}", k, v
    elif isinstance(d, list):
        for j, v in enumerate(d):
            yield from _walk(v, f"{path}[{j}]")


def validate(src: Source) -> list[str]:
    """Problems with one entry (empty list = OK)."""
    p, c = [], cat.catalog()
    if src.id not in c.index:
        return [f"{src.id}: not in the catalog"]
    row = c.loc[src.id]
    if row["counted"] != "yes" and not src.covariate_only:
        p.append(f"{src.id}: catalog says counted={row['counted']!r}")
    if src.id in set(cat.excluded()["id"]):
        p.append(f"{src.id}: in excluded_benchmark.csv")
    if not re.match(r"^[a-z0-9_]+$", src.slug or ""):
        p.append(f"{src.id}: slug {src.slug!r} must match [a-z0-9_]+")
    auth = (src.fetch or {}).get("auth")
    if row["access_class"] != "open" and not auth and not (src.fetch or {}).get("try_without_key"):
        p.append(f"{src.id}: access_class {row['access_class']!r} needs fetch.auth (an env var) or a manual download")
    if auth is not None:
        if not ENV_NAME.match(str(auth.get("env", ""))) or set(auth) - {"env", "param", "header"}:
            p.append(f"{src.id}: fetch.auth must be {{env: NAME, param|header: ...}} with an environment variable NAME")
    for path, k, v in _walk(dict(index=src.index, fetch=src.fetch, read=src.read, exclude=src.exclude)):
        if SECRET_KEY.search(str(k)) and isinstance(v, str) and path not in (".fetch.auth.param", ".fetch.auth.header"):
            p.append(f"{src.id}: {path} looks like a secret; secrets live in environment variables only")
    if row["overlap"] == "mixed" and not (src.exclude or {}).get("rule"):
        p.append(f"{src.id}: catalog overlap is 'mixed' ({row['notes'][:80]}...): exclude.rule is required")
    rules = cat.aggregator_rules()
    need = set(rules[rules["aggregator_id"] == src.id]["source_id"])
    have = set((src.exclude or {}).get("aggregator", []) or [])
    if need - have:
        p.append(f"{src.id}: aggregator rules not covered in exclude.aggregator: {sorted(need - have)}")
    if src.loader != "table" and not src.loader.startswith("adapter:"):
        p.append(f"{src.id}: loader must be 'table' or 'adapter:<name>'")
    if src.loader == "table":
        fmt = (src.read or {}).get("format")
        if fmt not in FORMATS:
            p.append(f"{src.id}: read.format must be one of {sorted(FORMATS)}")
        if not src.time or not (src.time.get("col") or src.time.get("cols") or src.time.get("parts") or src.time.get("expr")):
            p.append(f"{src.id}: time.col, time.cols or time.parts is required")
        if not src.id_cols and not isinstance(src.value, list):
            p.append(f"{src.id}: id columns are required (or a list of value columns)")
        if src.value is None and not src.events:
            p.append(f"{src.id}: value or events is required")
    f = src.fetch or {}
    if not (f.get("urls") or f.get("url_template") or f.get("manual")):
        p.append(f"{src.id}: fetch.urls or fetch.url_template is required")
    if f.get("url_template"):
        names = set(re.findall(r"{(\w+)}", f["url_template"]))
        if names - set(f.get("vars", {})):
            p.append(f"{src.id}: url_template variables without vars: {sorted(names - set(f.get('vars', {})))}")
        if f.get("vars", {}).get("series") == "index" and not src.index:
            p.append(f"{src.id}: vars.series = index needs an index")
    if not src.tasks:
        p.append(f"{src.id}: tasks (one per output frequency) are required")
    for t in src.tasks:
        try:
            canonical(t["freq"])
        except Exception as e:      # noqa: BLE001
            p.append(f"{src.id}: task freq {t.get('freq')!r}: {e}")
        if t.get("how", "mean") not in HOWS:
            p.append(f"{src.id}: task how {t.get('how')!r} not in {sorted(HOWS)}")
    return p


def validate_all(path: Path | str = SOURCES) -> list[str]:
    srcs, _ = load_sources(path)
    p = [x for s in srcs.values() for x in validate(s)]
    slugs = [s.slug for s in srcs.values()]
    p += [f"slug {s!r} used twice" for s in sorted({s for s in slugs if slugs.count(s) > 1})]
    return p
