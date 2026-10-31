# Batch 2: the probe list (Perlmutter, login node)

Written for the probe step of batch 2 (owner OK 2026-10-29). Generated from tideforge/devdata/sources.yaml (each
entry's `confirm` list); nothing here has been fetched from the dev machine. Run, from a login node:

    python -m tideforge.devdata.ingest probe --ids batch2 > $TF_ROOT/logs/batch2_probe.txt 2>&1

then paste the file back. Every entry starts on the pending adapter: its loader (adapter or table spec) is written
from the probe output, and a source whose probe shows a licence or access problem is dropped and logged. Before
ingest, the licences marked **read** below are recorded in the entry's notes.

Index time (owner 2026-10-31): D279 Santa Fe, D202 C-MAPSS, D110 muGrowthDB have no calendar time and use frequency
`index` (season 1, horizons 10 / 25 / 50 steps; their adapters write step i at 1970-01-01 + i s).

| Id | Slug | Ctry | Domain | Role | Licence | What the probe must show |
|---|---|---|---|---|---|---|
| D251 | bwdf_water | IT | Utilities-water | tuning | **read**: competition terms | the file URLs behind water_benchmark_hub's BWDF load_data() (or the journal's supplemental materials); columns: one inflow column per DMA, hourly timestamps, local time zone |
| D252 | nyc_dsny_tonnage | US | Utilities-waste | tuning | NYC open data | columns (month, borough, community district, refuse / paper / MGP tonnage) |
| D279 | santa_fe | multi | Physics-reference | tuning | research use | the data file links (A laser, B sleep, C currency, D synthetic, E astrophysical, F Bach) |
| D210 | india_ogd | IN | Government-stats | tuning | NDSAP | which human-activity resources to take (e.g. power, transport, tourism series) and their resource ids; the API key requirement (DATAGOVIN_API_KEY) for the resource endpoints |
| D144 | telecom_italia | IT | Telecom-network | tuning | ODbL | the Dataverse file list (Milan grid, telecommunications activity, 10-min, Nov-Dec 2013); columns: square id, time interval (epoch ms), country code, sms in / out, call in / out, internet |
| D326 | abilene_geant | multi | Telecom-network | tuning | research use | whether the IEEE DataPort files need a login (then manual download); matrix layout (OD pairs x time) |
| D328 | ucc_5g | IE | Telecom-network | locked | CC BY | folder layout (static / driving, Netflix / Amazon / download), timestamp column and throughput column |
| D170 | binance_klines | multi | Crypto-markets | tuning | free | the S3 listing (https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?prefix=data/spot/monthly/klines/); kline CSV columns (open time ms, open, high, low, close, volume, ...) |
| D174 | fi2010_lob | FI | Crypto-markets | tuning | research | the download link and file format (normalised txt matrices: 149 rows x events) |
| D340 | nyc_311 | US | Public-services | tuning | city open data | the SoQL aggregate (day x complaint type x borough) returns the full range in pages |
| D342 | police_uk | GB | Public-services | locked | OGL | the archive download (https://data.police.uk/data/archive/) and its CSV columns (Month, Falls within, Crime type) |
| D343 | sf_fire_calls | US | Public-services | locked | city open data | the dataset id nuek-vuh3 and its column names |
| D163 | uk_hpi | GB | Housing-realestate | locked | OGL v3.0 | the full-file CSV link on the page and its columns (Date, RegionName, AveragePrice, SalesVolume) |
| D162 | zillow | US | Housing-realestate | tuning | **read**: Zillow terms | the CSV URL pattern (the research page answered 403 to the dev machine); wide layout: one column per month |
| D360 | nyc_property_sales | US | Housing-realestate | tuning | city open data | the annualised sales xlsx links (xlsx needs an adapter) |
| D047 | kr_ev_charging | KR | EV-parking | tuning | CC BY 4.0 | the data repository link in the paper's Data Records section and the session columns |
| D052 | melbourne_parking | AU | EV-parking | tuning | CC BY 4.0 | which dataset holds the historic bay events (the live feed is not history); event columns |
| D374 | wfp_food_prices | multi | Agriculture-commodities | tuning | CC BY-IGO | the per-country CSV resources and columns (date, market, commodity, unit, price, usdprice) |
| D376 | usda_ams | US | Agriculture-commodities | tuning | US public domain | whether the MARS API needs a key (it did in 2024: then fetch.auth {env: USDA_MARS_KEY, header: ...}); which report slugs give daily / weekly price series |
| D283 | noaa_fisheries | US | Ecology | locked | US public domain | the FOSS ORDS endpoint and fields (year, month, state, species, pounds, dollars) |
| D286 | bbs_birds | US | Ecology | tuning | US public domain | the release zip (States.zip / 50-StopData) and its columns |
| D334 | seattle_library | US | Events-entertainment | tuning | public domain | dataset id tmmm-ytt6 and fields; the per-title series to sample (top titles by checkouts) |
| D330 | steam_players | multi | Events-entertainment | tuning | CC BY 4.0 | the file list and CSV layout (one file per game?) |
| D110 | mugrowthdb | multi | Biology-growth | tuning | **read**: check | the bulk export route; time units (minutes / hours since inoculation: no calendar) |
| D289 | biodare2 | multi | Biology-growth | tuning | per dataset | the public-experiments API, the per-experiment licence field and the data export format |
| D169 | fama_french | US | Macro-finance | locked | free | the zip members and their header lines (the CSVs carry text above and below the table) |
| D371 | treasury_dts | US | Macro-finance | tuning | US public domain | the endpoint list and fields (record_date, transaction_catg, transaction_today_amt) |
| D068 | usgs_quakes | multi | Space-geo | locked | US public domain | query by region boxes and magnitude bands (20k events per query cap) |
| D275 | ngl_gnss | multi | Space-geo | tuning | open | the .tenv3 URL pattern and columns (east / north / up) |
| D272 | intermagnet | multi | Space-geo | tuning | CC BY-NC 4.0 | the GIN web service query for definitive 1-min data and the IAGA-2002 layout |
| D229 | fr_fuel_prices | FR | Oil-gas-fuels | tuning | Licence Ouverte | the yearly zip of XML (one file per year), station id and fuel type fields |
| D230 | it_fuel_prices | IT | Oil-gas-fuels | tuning | IODL 2.0 | the daily CSV archive links (prezzo_alle_8.csv history) and columns |
| D188 | tsa_checkpoint | US | Aviation-tourism | tuning | US public domain | the HTML tables per year (needs an adapter) |
| D336 | nps_visits | US | Aviation-tourism | tuning | US public domain | the query-builder export (CSV) per park and month |
| D339 | eurocontrol_airports | multi | Aviation-tourism | locked | **read**: Eurocontrol terms | the CSV twin of Airport_Traffic.xlsx (ansperformance.eu/csv/#aptflt-csv) and its columns |
| D103 | cgmacros | US | Physiology | tuning | open access | the version folder, the per-participant CSV columns (Libre / Dexcom glucose, meal macros, HR) |
| D104 | big_ideas_glycemic | US | Physiology | tuning | open access | the version folder and the Dexcom CSV columns |
| D202 | cmapss | multi | Industrial-PHM | tuning | NASA open | the zip (train_FD00x.txt: unit, cycle, 3 settings, 21 sensors) |
| D391 | petrobras_3w | BR | Industrial-PHM | tuning | CC BY 4.0 | the parquet layout (class folders, real vs simulated instances: only REAL instances are kept) |
| D394 | steel_energy | KR | Industrial-PHM | locked | CC BY 4.0 | the CSV columns (date, Usage_kWh, reactive power, CO2, WeekStatus, Load_Type) |
| D401 | neso_demand | GB | Energy-load | locked | NESO Open Data Licence | the yearly resource URLs from datapackage_show; columns SETTLEMENT_DATE, SETTLEMENT_PERIOD (1-50 on clock-change days), ND, TSD, embedded wind / solar; settlement periods are UK local time |
| D182 | tfl_cycle_hire | GB | Transport-mobility | tuning | OGL | the usage-stats file list (weekly CSVs, journey rows: start date, start station) |
| D014 | grid_india_demand | IN | Energy-load | tuning | **read**: check terms | the terms of use, read in a browser: automated access must be allowed, else the source is dropped (owner rule) and D403 replaces it; the daily PSP report format |
| D373 | agmarknet | IN | Agriculture-commodities | tuning | GODL India | the CEDA mirror's bulk files (arrivals and modal prices per market and commodity); or the data.gov.in resource API |
| D384 | seoul_bike | KR | Transport-mobility | tuning | KOGL | the rental-history file list (monthly CSV / zip, Korean headers, CP949?); the subway hourly dataset id |
| D219 | kpx_demand | KR | Energy-load | tuning | KOGL | the dataset id for hourly national demand and its file format |
| D383 | melbourne_pedestrians | AU | Transport-mobility | tuning | CC BY 4.0 | the dataset id and the export URL of the hourly counts per sensor |
| D402 | anac_vra | BR | Aviation-tourism | tuning | BR federal open data (check record) | the monthly file names inside a year folder and the CSV columns (airline, flight, origin / destination ICAO, scheduled / actual departure, situation); the licence on the dados.gov.br record |
| D345 | stats19 | GB | Transport-mobility | tuning | OGL | columns (date, police force, severity) and the file size (~1 GB) |
| D403 (standby) | cea_power_supply | IN | Energy-load | tuning | **read**: GODL India | the resource id and fields (state, month, peak demand, peak met); the API key (DATAGOVIN_API_KEY) |

Covariate tasks (20 new): after the targets are ingested,

    python -m tideforge.devdata.covariate_tasks stations --raw $TF_DATA/raw        # login node: the GHCN-Hourly stations
    python -m tideforge.devdata.covariate_tasks build --raw $TF_DATA/raw --staging $TF_DATA/staging

Value-level exclusion checks (audit check) to look at specifically: D170 (vs TIME Crypto), D229 / D230 (vs TIME
Oil_Price), D162 (vs TIME Housing_Inventory), D376 (vs TIME Commodity_*), D394 (vs TIME Smart_Manufacturing), D401
(vs fev entsoe: if it holds GB demand, D401 is dropped and GB = D182 + D345).

## Probe round 1 (Perlmutter, 2026-10-31: logs/perlmutter/batch2_probe.txt, 50 sources)

| Status | Sources | What happens next |
|---|---|---|
| **confirmed, spec written** (tested on the probe's rows: tests/devdata/test_batch2_specs.py) | D252 DSNY, D162 Zillow, D345 STATS19, D383 Melbourne pedestrians, D340 NYC 311, D343 SF Fire, D334 Seattle library, D394 steel (zip; member name per UCI docs), D279 Santa Fe (moved to PhysioNet Set B) | fetch + ingest |
| **layout confirmed, adapter to write** (the probe showed the API or listing) | D401 NESO (CKAN resources; settlement-period clock), D402 ANAC VRA (month folders), D374 WFP (CKAN), D371 Treasury DTS (paged JSON), D068 USGS (count API per region / day), D275 NGL (station list; .tenv3 per station), D283 NOAA FOSS (paged ORDS JSON), D286 BBS (ScienceBase file list), D272 INTERMAGNET (GIN service), D328 UCC 5G, D391 3W (GitHub trees), D330 Steam (Mendeley file list), D342 police.uk (archive zip), D229 France fuel (zip of XML: stdlib), D169 Fama-French (zip CSV with text above / below) | I write these from one more probe of their data files (round 2) |
| **URL fixed, probe again** | D202 C-MAPSS (PHM Society mirror of NASA PCoE), D144 Telecom Italia (Dataverse API), D170 Binance (S3 listing), D052 Melbourne parking (yearly datasets are attachments), D103 / D104 (PhysioNet file listings), D182 TfL (S3 listing), D163 UK HPI (full-file CSV) | round 2 |
| **key needed** (free; the owner registers and sets the variable) | D210, D403 `DATAGOVIN_API_KEY` (data.gov.in); D376 `USDA_MARS_KEY` (MyMarketNews: the key is the basic-auth user name, empty password) | round 2 once set |
| **owner's PC** (unreachable from NERSC; `fetch.manual`) | D230 MIMIT quarterly tar.gz, D373 CEDA monthly CSV, D014 Grid-India (PDF only: see below) | browser download, `ingest register`, upload, `ingest verify` |
| **unknown: landing page only** | D251 BWDF (loader source to find), D326 Abilene/GEANT (IEEE DataPort login), D174 FI-2010 (Fairdata UI), D047 Korean EV (data link in the paper), D110 muGrowthDB, D289 BioDare2 (JS app), D188 TSA (HTML table: stdlib parser), D336 NPS (SSRS report), D384 Seoul bike, D219 KPX (JS portals; data.go.kr OpenAPI needs a free key) | round 2, or manual / drop |
| **needs a new dependency** (owner decision) | D339 Eurocontrol and D360 NYC sales (xlsx: openpyxl), D014 Grid-India (PDF tables) | ask |

Round 2, from a login node, after `git pull` (and the keys set, if registered):

    python -m tideforge.devdata.ingest probe --ids D202,D144,D170,D052,D103,D104,D182,D163,D210,D403,D376 \
        > $TF_ROOT/logs/batch2_probe2.txt 2>&1
