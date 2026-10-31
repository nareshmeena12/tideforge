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
   **Small datasets (owner 2026-10-29):** a dataset with fewer than 30 windows (in the fast or the full set) is pooled
   with the other small datasets of its domain into one `<domain>_misc` unit for weighting: the unit counts as one
   dataset of the domain, and its members share that unit's weight equally (tideforge/devdata/build.py MIN_WINDOWS; the
   pools are in the manifest under weights.<role>.pooled and printed by the build). Expected in dev-v1 (estimates from
   the catalog's series counts, confirmed by the build): D188 TSA (1 series), D219 KPX (5), D174 FI-2010 (5), D394 steel
   (6), D279 Santa Fe (6) if unblocked, and any other source with fewer than ~4 series of one frequency.

**Decisions 2026-10-27 (owner):**
- Step 8: the split unit is a group (`catalog/split_groups.csv`: same provider and network, or copies), not a dataset. The
  whole catalog is split at once (`catalog/split_v1.csv`, committed); afterwards rows are only appended.
  `catalog/aggregator_rules.csv` lists which sub-series an aggregator drops when the source is ingested directly (proposed
  with the split; enforced by `sources.yaml` validation).
- Step 6, value level: two stages. Candidates from hashes of coarse z-score codes (0.25-sd bins, clipped at +-4) in two
  codings shifted by half a bin; verification by Pearson r > 0.999 on the raw 64 points, after aggregation where the
  frequencies differ. Flag at >= 3 verified windows; the dev-vs-dev check stays.
- Step 7: weather-driven covariates are marked known + oracle in the task metadata (G1.2 tests that difference). EIA load
  uses a population-weighted mean of 3-5 stations near each balancing area's largest cities; everything is aligned in UTC.
- Step 9: horizons follow TIME's frequency -> horizon convention (`catalog/time_horizons.csv`, written by
  `tools/time_horizons.py`). A dev version needs >= 15 domains; batch 1 is the pipeline test, not dev-v1. The manifest
  (counts and sha256 only) is committed under `catalog/manifests/`.

**Decisions 2026-10-27 (owner, horizons):** TIME's sub-daily rows are used as written; D, W, M, Q, B and Y use tideforge's
own convention (catalog/horizons_extra.csv: D 30 / 90 / 180, W 13 / 26 / 52, M 12 / 18 / 24, Q 4 / 8, B 20 / 60 / 120,
Y 4 / 6), not TIME's (its daily row comes from one dataset). A term is used for a series only if the series has >= 3 x that
horizon of history; skipped terms are counted in the build report. Step 5's length rule follows: a series is kept if it
can serve the shortest term (3 x the shortest horizon + 64 points), no longer 3 x the longest.

**Decisions 2026-10-27 (owner, scoring; `tideforge/eval/devscore.py`, `devmetrics.py`, `naive_time.py`):**
- Metrics and aggregation as TIME (zqiao11/TIME c633e23): per window MASE and CRPS from timebench's
  `compute_per_window_metrics_from_quantiles`, called directly; per (dataset, term) the mean over windows (NaN ignored)
  divided by seasonal naive's mean for the same pair; per dataset the geometric mean over its terms; overall the weighted
  geometric mean with the manifest weights (domain cap included). Slices and the bootstrap use the same pair ratios.
- The baseline is SeasonalNaiveTIME: statsforecast's SeasonalNaive band (level 20 / 40 / 60 / 80) with the 9 quantiles read
  straight off the band. TIME draws 100 unseeded samples from that band and takes their quantiles; we drop that step on
  purpose (it makes the baseline random), so our naive's CRPS differs slightly from TIME-Output's. statsforecast is a
  test-only extra (`test-sf`) used to check the band; it is never in the core install.
- Two headline numbers: the main dev score (`--set fast` ~15k / `full` ~75k windows) and the dev-covariate score
  (covtasks, groups ON, with its ON-vs-hidden gain; G1 is judged on these), plus a combined score at equal weight per main
  dataset / covariate task. No change to build.py for this.
- E0.5's rank correlation (dev vs fev / TIME, Spearman and Kendall with bootstrap CIs over models) is a sanity check, never a
  gate.

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

**Decision 2026-10-29 (owner): dev versions.** dev-v1 = batches 1 + 2 (70 datasets, 30 domains), built now so Phase 2 does
not wait for the 200 target; batch 3+ becomes dev-v2, whose target is the >= 200 below. Runs are compared only within one
version; finished runs are rescored on dev-v2 from their checkpoints (`python -m tideforge.train.finish --ckpt
<run>/final.pt --version dev-v2`).

**Decision 2026-10-27 (owner): size target (dev-v2's since 2026-10-29).** At least 200 datasets across 15+ domains, the caps unchanged (500 series and the
last 16k points per series), landing around 1B points. Diversity of datasets matters more than points: the score weights
datasets equally and caps domains, so a new domain or a new dataset in a thin domain is worth more than another large source
in a full one. Batches are planned domain-first: thin domains before deep ones.

**Batch 2 plan (domain-first; open access, not in batch 1; ids from dataset_catalog_v2.csv, counted datasets in the domain in
brackets).** Batch 1 covers 11 domains (Weather-climate, Hydrology, Energy-load, Energy-gen, Buildings-heat, Air-quality,
Transport-mobility, Cloud-servers, Web-attention, Retail-sales, Epidemics-health). Batch 2 starts with
the domains batch 1 does not touch, thinnest first, 1-3 sources each:
- Utilities-water [1]: D251 BWDF 2024. Utilities-waste [1]: D252 NYC DSNY tonnage. Physics-reference [1]: D279 Santa Fe.
- Government-stats [3]: D210 India data.gov.in. Telecom-network [5]: D144 Telecom Italia, D326 Abilene / GEANT,
  D328 5G throughput (UCC). Crypto-markets [5]: D170 Binance (mixed: no BTC / ETH / LTC / BCH), D174 FI-2010.
- Public-services [6]: D340 311 requests, D342 data.police.uk, D343 SF Fire calls. Housing-realestate [8]: D163 UK HPI,
  D162 Zillow, D360 NYC property sales. EV-parking [8]: D047 Korean EV charging, D049 UK AC chargers, D052 Melbourne parking.
- Agriculture-commodities [9]: D374 WFP food prices, D355 Hass avocado, D376 USDA AMS. Ecology [9]: D283 NOAA fisheries
  landings, D286 Breeding Bird Survey, D290 beehive monitoring. Events-entertainment [10]: D334 Seattle library checkouts,
  D330 Steam player counts, D200 theme park waits. Biology-growth [10]: D110 muGrowthDB, D289 BioDare2 circadian.
- Macro-finance [11]: D169 Fama-French, D371 US Treasury DTS. Space-geo [12]: D068 USGS earthquake counts, D275 NGL GNSS,
  D272 INTERMAGNET. Oil-gas-fuels [14]: D229 France fuel prices, D230 Italy fuel prices, D034 North Dakota wells.
  Aviation-tourism [15]: D188 TSA checkpoint volumes, D336 NPS visits, D339 Eurocontrol traffic. Physiology [15]: D103
  CGMacros, D104 BIG IDEAs glycemic. Industrial-PHM [21]: D202 C-MAPSS, D391 3W Petrobras, D394 steel energy (UCI).
That is about 45 sources in 19 new domains (30 with batch 1). Each still goes through probe -> fetch -> ingest -> audit; the
split groups and aggregator rules apply (e.g. D374 sits with D375 FPMA; D202 with N-CMAPSS).

**Batch 2 as written (owner OK 2026-10-29; tideforge/devdata/sources.yaml `batch2`, probe list docs/BATCH2_PROBES.md).**
49 sources: the plan's 45 minus 5, plus 9 for the held-out countries; 1 standby.
- Licences / access read 2026-10-29 (the plan's flagged ones), recorded in each entry's notes:
  - dropped: D355 Hass Avocado (detailed data needs an account; "All Rights Reserved") and D049 UK AC chargers (no public
    dataset: the raw sessions are "by request" from the authors).
  - deferred: D034 NDIC well production (well-level data appears to be behind NDIC's Premium Services: unconfirmed),
    D200 queue-times (the free API is real-time only, attribution required; historic access not documented), D290
    beehive (Kaggle only; no Kaggle token).
  - kept: D339 Eurocontrol (cite EUROCONTROL, non-commercial, no modification without permission: research evaluation
    only, files kept as downloaded), D251 BWDF (paper CC BY 4.0; the data licence to read in the supplement), D110
    muGrowthDB (public, no licence named), D289 BioDare2 (per-dataset licences: open ones only), D162 Zillow (page 403 from
    the dev machine: read in a browser before ingest).
- Held-out countries (G2.4), human-activity sources by split role:
  GB D401 NESO (new; locked by the split rule), D182 TfL cycle hire (tuning), D345 STATS19 (tuning; added because D401 is
  locked); IN D014 Grid-India (tuning; only if its terms allow automated access, else dropped for the standby D403 CEA
  monthly) and D373 Agmarknet (tuning); KR D384 Seoul bike (tuning), D219 KPX (tuning), D047 EV charging (tuning), D394
  steel (locked); AU D052 Melbourne parking, D383 Melbourne pedestrians (tuning; D383 pretrain-flagged; AEMO is excluded,
  TIME OpenElectricity_NEM); BR D217 ONS (batch 1, tuning) and D402 ANAC VRA daily flights (new, tuning; licence of the
  dados.gov.br record to confirm). D004 Elexon is the same GB demand as D401, not a fallback.
- New catalog rows D401-D403 (catalog/CHANGELOG.md); D001 / D002 (OPSD / ENTSO-E) drop GB for D401.
- Index time (owner 2026-10-31): D279, D202, D110 have no calendar time (index, engine cycle, time since inoculation); they
  use frequency `index`: season 1, horizons 10 / 25 / 50 steps, the same >= 3 x history rule (catalog/horizons_extra.csv).
- Covariate tasks: 20 new (25 in total). Three approved weather joins are calendar-only for now (Grid-India regions, ONS
  Brazil subsystems, Korean EV: point_weather takes one point; per-series points after the probe), and three tasks were
  replaced (theme parks -> Treasury DTS calendar, beehive -> SF Fire weather, Eurocontrol -> STATS19 calendar; Olist ->
  ANAC calendar).

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
