# Hugging Face dev-data catalog (v2, 2026-11-11; metadata profile 2026-11-12)

Copied from the project notes. The catalog lists only datasets hosted on the Hugging Face Hub, so one tool downloads
everything. Files in tideforge/devdata/: hf_catalog_v2.csv (614 rows, 420 repos), hf_excluded.csv (122 rows that must
not be used), hf_profile.csv, hf_profile_files.csv, HF_PROFILE.md (metadata-only profile).

## Metadata profile (2026-11-12)
No dataset file was downloaded (1.45 GB of metadata, 26,283 requests). Totals over all 614 rows: 6.33 TB of data files,
about 1.07 trillion values (161B exact + 910B estimated). Usable now (keep / keep_filtered, not gated): 537 rows.

Size is concentrated: 20 rows of 50 GB or more hold 4.8 TB. Easy/medium-ingest usable rows by size limit:
| max size per row | rows | repos | bytes |
|---|---|---|---|
| 1 GB | 329 | 192 | 33 GB |
| 5 GB | 377 | 223 | 152 GB |
| 20 GB | 408 | 246 | 461 GB |
| 50 GB | 418 | 255 | 788 GB |

What the metadata cannot tell:
- Value counts include every numeric column (ids, coordinates, flags): upper bounds.
- Series counts for long-layout tables are often unknown or wrong ("median length 1" = id column not found).
- Frequency is known for only 180 of 537 usable rows.
- Slicing does not save bytes for most repos; choosing datasets controls size.
- 25 rows are gated without access.

## Status column
keep 530; keep_filtered 29 (drop the sub-series named in `overlap`); check 55 (decide before use);
exclude:benchmark 88; exclude:duplicate 28; exclude:simulated 4; exclude:not_forecasting 2.

## Benchmark rules applied
- fev-bench: ETT, M5, Favorita, Rossmann, Walmart, Rohlik, Hermes, restaurant, hierarchical_sales, ENTSO-E, EPF, ERCOT,
  GFC12/14/17 and the ProEnFo family (bull, cockatoo, covid19_energy, elf, hog, pdb, spain), solar_with_weather, KDD Cup
  2022, jena_weather, UCI air quality, LOOP_SEATTLE, SZ_TAXI, M_DENSE, bizitobs, BOOMLET, redset, hospital, UK covid,
  ECDC ILI, FRED-MD/QD, GVAR, US consumption, world CO2 / life expectancy / tourism, Australian tourism.
- GIFT-Eval test: M4, electricity, solar, hospital, covid_deaths, us_births, saugeen, temperature_rain, KDD Cup 2018,
  car_parts, restaurant, hierarchical_sales, LOOP_SEATTLE, SZ_TAXI, M_DENSE, ETT, jena_weather, bitbrains, bizitobs.
- TIME: FRED series, BTC/ETH/LTC/BCH, Singapore data, IMOS ocean, Australian PVOutput, IMF PortWatch, Azure 2019,
  OpenElectricity / AEMO NEM, World Bank Pink Sheet, US Treasury H.15.
