# Dev/validation real-data catalog (v2, 2026-10-25)

Files (in catalog/): dataset_catalog_v2.csv (377 rows), excluded_benchmark.csv (23), benchmark_datasets.csv (with TIME original sources), expand_catalog.py, verify_urls.py.

## Benchmark lists (now complete, read from the repos)
- TIME: 39 unique datasets = the 50 dataset/frequency entries in src/timebench/config/datasets.yaml. Original sources come from paper Table 2/App. B. Most of the economics, finance and crypto datasets are FRED; SG_* are data.gov.sg; the four ocean sets are IMOS; Australia_Solar is 3 PVOutput stations; Port_Activity is IMF PortWatch; Azure2019 is the 2019 VM CPU trace.
- fev-bench: 100 tasks over 59 dataset configs (benchmarks/fev_bench/tasks.yaml). beijing_air_quality and cdc_fluview_ilinet are hosted in fev_datasets but are not fev-bench tasks.
- GIFT-Eval test: 28 datasets (short_datasets list in the gift-eval notebooks).

## Exclusion rule
- Dropped 23 catalog datasets that are in a benchmark, or are the same series from another publisher (e.g. Pink Sheet commodity prices = TIME Global_Price; H.15 = TIME US_Term_Structure; AEMO = TIME OpenElectricity_NEM).
- 29 rows are `mixed`: keep the dataset, drop the listed sub-series. Examples: FRED minus FRED-MD/QD and every TIME FRED series; EIA spot keeps Henry Hub only; no Singapore stations/PM2.5 in GHCN/ERA5/OpenAQ/WAQI; no NE China cells in reanalysis; no Australian PVOutput systems; no BTC/ETH/LTC/BCH prices; Azure 2017 trace only; OWID without cases/deaths or European hospital occupancy.

## Size (order-of-magnitude estimates)
- 365 counted datasets, ~62M series available; ~400B points, three quarters of it in 10 giant high-frequency sources.
- 311 datasets under 1B points: ~37M series / ~24B points.
- Dev set target agreed 2026-10-27: ~1B points (caps: 500 series and 16k points per series per dataset), see DATA_COLLECTION.md.

## URL status
148 URLs appeared in search results; the rest are from memory. Run catalog/verify_urls.py locally; it writes url_check.csv.

## Next
Run the URL check; series-level hash audit after download; dataset-level frozen split (tuning / locked / fev+TIME final).
