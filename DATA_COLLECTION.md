# tideforge: collecting the dev set (~1B points)

Target: about 1B points of real series from ~340 datasets that are in none of TIME, fev-bench or GIFT-Eval (test), split
by whole dataset into a tuning set (~75%) and a locked set (~25%). Source list: `dataset_catalog_v2.csv` (377 rows,
365 counted). Exclusions: `excluded_benchmark.csv` and the `mixed` notes in the catalog.

What the size buys: the headline "did this change help?" is limited by the number of datasets, not points. The 1B points
pay off in diagnosis (enough windows in every slice: domain, frequency, period length, history length) and keep the set
usable when context grows to 32K.

---

## 1. Rules that hold for the whole set

- **Equal weight per dataset** in every score, whatever its size. Giant sources never decide a result.
- **Domain cap:** no domain above ~10% of the score weight.
- **Per-dataset caps at ingest:** at most 500 series per dataset (frozen random sample, stratified by sub-group such as
  region or product type), and the last 16k points of each series (all of it if shorter).
- **Split by dataset, never by series.** One dataset (all its frequencies and sub-series) sits in exactly one role.
- **Real gaps stay as NaN.** Nothing is forward-filled at ingest.
- **Versioned.** `dev-v1`, `dev-v2`, ... are frozen snapshots. Two runs are compared only on the same version; B0 is
  re-scored whenever the version changes.

## 2. Where to download

This cloud session cannot reach most data hosts (only GitHub is allowed), so downloads run elsewhere:
- **Perlmutter data-transfer or login nodes** for bulk sources, writing to a persistent space such as CFS (scratch is
  purged). Check the current NERSC policy for where downloads may run before starting.
- **Your Windows machine** (`E:\Competitions\tideforge\data`) for sources that need a browser login or a Kaggle token,
  then copy to Perlmutter.
- Tokens (Kaggle, EIA, ENTSO-E, OpenAQ, ...) live in environment variables you set. Never in files, logs or the repo.

## 3. Steps

1. **Check URLs.** Run `python verify_urls.py dataset_catalog_v2.csv` and fix every `dead` / `ok_weak` row first.
2. **Tier the sources by access** (column `access_class`):
   - A. open direct download (~310 datasets): start here.
   - B. free API key (EIA, ENTSO-E, OpenAQ, ...).
   - C. registration / terms (Kaggle, UKDS, gated Hugging Face).
   - D. scraping or credentialed (PhysioNet credentialed, MIMIC, eICU): skip for v1.
3. **Adapters, config first.** Most sources are CSV / ZIP / parquet / JSON with a time column, an id column and value
   columns. One YAML entry per dataset drives a generic loader; write a code adapter only for odd sources.
   ```yaml
   iowa_liquor:
     url: https://data.iowa.gov/...
     format: csv
     time: date
     id: [store_number, category]
     value: sale_dollars
     aggregate: {freq: W, how: sum}
     exclude: none
     licence: public domain
   ```
4. **Canonical output per dataset:** `data/<dataset_id>/<freq>.parquet` with columns `series_id, timestamp, value`,
   plus `meta.json`: source, url, licence, retrieval date, sha256 of the raw files, frequency, units, domain, overlap
   label, the exclusion rule applied, number of series, number of points, NaN share.
5. **Clean** (rules recorded in meta.json): drop series shorter than 3 x the longest horizon + 64 points, with more than
   50% missing, or constant. Event data (trips, calls, sales lines) is aggregated to one or two natural frequencies
   (for example hourly and daily); each frequency is a separate task of the same dataset.
6. **Exclude and audit.**
   - Name level: apply `excluded_benchmark.csv` and every `mixed` rule (e.g. FRED minus FRED-MD/QD and TIME's FRED
     series; EIA spot keeps Henry Hub only; no Singapore stations; no Australian PVOutput systems; no BTC / ETH / LTC /
     BCH prices; Azure 2017 trace only).
   - Value level: download the TIME, fev and GIFT-Eval data on the transfer node, hash z-normalised 64-point windows of
     every benchmark series, and check the dev set against them. Any dataset with matches is flagged and checked by hand.
7. **Covariate tasks (~25, needed for G1).** Join a target with real covariates. The realised values of known covariates
   stand in for forecasts in v1; G1.2 tests forecast error separately. First candidates:
   - EIA-930 balancing-area load + GHCN hourly temperature near the area
   - Electricity prices + load + wind / solar for markets far from fev's EPF set (BE, DE, FR, Nord Pool, PJM) and
     TIME's NEM: CAISO, NYISO, ISO-NE, AESO (Alberta), New Zealand EMI. Avoid Nordic and DE-AT zones (too close to
     EPF-NP / EPF-DE).
   - Eskom demand + temperature
   - Paris bike counters + Open-Meteo weather; Seoul bikes + weather
   - Iowa liquor sales + US holidays; Predict Future Sales + Russian holidays
   - PV systems (PVDAQ, Open Climate Fix UK PV) + NASA POWER irradiance and temperature
   - NYC taxi zones + weather and event flags
8. **Split.** Within each domain, sort datasets by `sha256(dataset_id)` and send every 4th to the locked set. The split
   file is committed once and never edited.
9. **Tasks and windows.** For each frequency, three horizons (short / medium / long, as TIME does) and the last 2-3
   rolling windows per series. From these, sample the score set (~75k windows, equal weight per dataset) and the fast
   subset (~15k windows).

## 4. Order of collection

v1 = the ~120 highest-yield open sources (expected 70-80% of the points); v2 = the rest. Start with sources that give
many series under the caps:

- Weather and water: GHCN-Daily, GHCN-Hourly, NDBC buoys, NOAA tides, USGS NWIS, Caravan, LamaH-CE, UK flood API
- Energy: EIA-930, ENTSO-E (minus fev's countries and prices), Energi Data Service, NZ EMI, ONS Brazil, BDG2, London
  smart meters, Tankerkonig fuel prices
- Air: OpenAQ, EPA AQS, EEA
- Mobility: Citi Bike, Divvy, NYC TLC, Chicago trips, Paris and Zurich counters, Seoul subway and bikes
- Cloud and web: Alibaba, Google, Azure 2017 traces, Backblaze, fresh Wikimedia pageviews, Stack Exchange
- Demand and prices: Iowa liquor, Agmarknet, WFP food prices, Zillow, NHS prescribing, Seattle library, VN1 / VN2
- Health: Eurostat weekly deaths, InfoDengue, CDC NNDSS

Some of these are in Chronos / LOTSA pre-training corpora (flagged `pretrain` in the catalog). They are fine for testing
our synthetic-only model, but reference models have seen them, so comparisons with those models on them are biased.

## 5. Effort (estimate)

A simple CSV source takes 30-60 minutes to configure and check; odd sources take longer. ~340 sources is several weeks of
part-time work, so v1 (~120 sources) comes first and Phase 0 starts on it. Every manifest records real counts, so the
1B target is checked against facts, not these estimates.

## 6. Licences

Every dataset's licence goes into meta.json. Non-commercial licences are fine for research evaluation; check them again
before any release that redistributes data.

## 7. Repo layout

```
tideforge/devdata/
  sources.yaml        one entry per dataset
  loaders/            generic csv / zip / parquet / json / hf / kaggle loaders + odd-source adapters
  ingest.py           fetch -> clean -> canonical parquet + meta.json
  audit.py            name and value-level benchmark checks
  covariate_tasks.py  the joined covariate tasks
  split.py            frozen dataset-level split
  build.py            windows, score set, fast subset, manifest
data/                 never committed
```
