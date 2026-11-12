# Hugging Face catalog profile (metadata only)

No dataset file was downloaded. Sources: Hub API metadata, the dataset viewer API, parquet footers read by range request, and small text metadata files. Every number says where it comes from in `hf_profile.csv` (`*_src`, `*_conf`). "Exact" means read from metadata without a guess; "estimated" means a rule or a scaling step was used; the layout itself is always a rule-based reading of the column names and types.

## 1. Run facts

- Date: 2026-11-12.
- Catalog: hf_catalog_v2.csv (614 rows, 420 repos).
- Profiled: 420 repos, 614 catalog rows, plus 871 extra per-config rows. 618062 data files listed in hf_profile_files.csv.
- Bytes transferred in this run: 1453691750 bytes (1.45 GB) in 26283 requests; 420 answers came from the cache; 0 rate-limit waits.
- Token: HF_TOKEN was set. Footer reads per catalog row: at most 50 (more files are sampled and scaled by bytes).
- footer: 23885 requests, 1.40 GB.
- text: 216 requests, 654.64 KB.
- viewer: 2182 requests, 56.98 MB.

Catalog rows at each profile level:

| profile level | rows |
|---|---|
| viewer+footers | 474 |
| files_only | 79 |
| footers | 34 |
| gated | 25 |
| viewer | 1 |
| files_only+dataset_info | 1 |

Levels: viewer+footers = viewer answers and parquet footers; footers = footers only (viewer not working); viewer = viewer schema and row counts, no footer; files_only = file names and sizes only; gated = gated, no access; missing / error = see section 10.

## 2. Totals

| set | datasets | repos | bytes | series (exact) | series (estimated) | series unknown (datasets) | points (exact) | points (estimated) | points unknown (datasets) |
|---|---|---|---|---|---|---|---|---|---|
| everything | 614 | 420 | 6.33 TB | 1.33M | 626.08M | 202 | 161.25B | 909.84B | 105 |
| status = check | 55 | 45 | 217.71 GB | 11.46k | 19.51M | 22 | 2.70B | 141.27B | 15 |
| status = keep | 530 | 358 | 5.73 TB | 1.32M | 606.23M | 168 | 151.41B | 661.81B | 83 |
| status = keep_filtered | 29 | 28 | 386.32 GB | 71 | 340.38k | 12 | 7.13B | 106.76B | 7 |

Target points only (all rows): 39.18B exact + 309.77B estimated; unknown for 105 datasets.
Points = all numeric value columns. Bytes = the data files in the row's scope, from the Hub file listing (exact). Several rows of one repo with overlapping scopes count their bytes more than once.

## 3. By domain

| domain | repos | datasets | series | points | bytes | share with covariates | date range |
|---|---|---|---|---|---|---|---|
| agriculture_nature | 24 | 26 | 2.51M | 5.88B | 27.89 GB | 65% (13 of 20) | 1568-01-01 to 2027-01-01 |
| air_quality | 19 | 21 | 267.89M | 2.03B | 20.42 GB | 88% (15 of 17) | 0003-03-12 to 2080-01-12 |
| economics_finance | 58 | 81 | 55.56M | 142.92B | 575.14 GB | 60% (39 of 65) | 1782-01-01 to 2102-01-01 |
| energy | 62 | 119 | 15.46M | 212.53B | 380.44 GB | 37% (37 of 99) | 1976-05-27 to 2050-01-01 |
| health | 32 | 38 | 1.83M | 15.34B | 180.35 GB | 60% (21 of 35) | 1888-06-10 to 2028-12-30 |
| hydrology_ocean | 27 | 27 | 23.33M | 22.14B | 1.65 TB | 68% (13 of 19) | 1740-01-01 to 2026-11-23 |
| industrial | 18 | 30 | 939.04k | 15.37B | 185.83 GB | 78% (18 of 23) | 2013-12-09 to 2026-09-30 |
| mobility | 49 | 81 | 136.11M | 80.37B | 606.20 GB | 55% (41 of 75) | 1894-06-15 to 2026-11-15 |
| other | 8 | 12 | 2.59M | 9.23B | 38.48 GB | 25% (2 of 8) | 1948-01-01 to 2026-10-23 |
| retail_demand | 23 | 25 | 95.63M | 2.63B | 17.60 GB | 78% (18 of 23) | 1970-01-01 to 2026-11-11 |
| society | 23 | 23 | 200.50k | 11.75B | 134.55 GB | 61% (11 of 18) | 1975-01-01 to 2026-11-11 |
| space_science | 27 | 28 | 4.40M | 78.50B | 785.91 GB | 44% (11 of 25) | 1947-01-01 to 2041-10-01 |
| weather_climate | 40 | 51 | 1.71M | 257.05B | 1.26 TB | 41% (17 of 41) | 1662-10-15 to 2150-08-25 |
| web_cloud | 38 | 52 | 19.25M | 215.36B | 468.63 GB | 67% (28 of 42) | 2007-07-12 to 2037-04-08 |

Series and points are exact + estimated, over the datasets where they are known. Share with covariates is over the datasets with a schema.

## 4. By frequency band

| frequency band | datasets | series | points |
|---|---|---|---|
| sub-minute | 8 | 11 | 69.44M |
| minute | 54 | 691.14k | 47.13B |
| hourly | 38 | 330.34k | 115.14B |
| daily | 22 | 525.62k | 4.34B |
| weekly | 10 | 372.92k | 1.08B |
| monthly | 11 | 1.04M | 15.76M |
| quarterly / yearly | 5 | 23.34k | 39.12B |
| irregular / events | 25 | 3.53M | 44.28B |
| mixed | 25 | 5.37M | 53.94B |
| unknown | 416 | 615.52M | 765.97B |

A frequency is set only when a column, feature or config / file name states it, or when span and series length imply it (estimated). It is never taken from the catalog text. "mixed" = several stated frequencies in one row.

## 5. By series length band

| series length (median, or mean where only that is known) | datasets | series |
|---|---|---|
| under 100 | 39 | 490.79M |
| 100-1k | 50 | 21.06M |
| 1k-10k | 77 | 19.39M |
| 10k-100k | 101 | 2.29M |
| over 100k | 118 | 230.35k |
| unknown | 229 | 93.65M |

## 6. Largest datasets

### 40 largest by points

| # | dataset | domain | points | confidence | series | bytes | layout |
|---|---|---|---|---|---|---|---|
| 1 | MachaParfait/ClusterWise | web_cloud | 159.41B | estimated | unknown | 221.01 GB | long |
| 2 | OpenSynth/AI4OPT-TS4PS-RTE7000 [tables (lines, generators, loads, ...)] | energy | 96.97B | estimated | unknown | 40.01 GB | long |
| 3 | Salesforce/lotsa_data [era5] | weather_climate | 94.16B | estimated | 239.52k | 386.51 GB | nested |
| 4 | nju-zhangsq/RMISC [WeatherBench5-625deg] | weather_climate | 77.75B | estimated | 81.92k | 333.66 GB | long |
| 5 | ai-iot/EnergyBench [real subsets overall (Dataset_V0.0)] | energy | 67.19B | estimated | 2.50k | 166.38 GB | long |
| 6 | nyu-planet-finders/kepler-llc | space_science | 40.93B | exact | 2.37M | 166.45 GB | nested |
| 7 | juliangehring/noaa-isd | weather_climate | 39.08B | estimated | unknown | 20.63 GB | long |
| 8 | Rrishab/OHLCV-1m [default] | economics_finance | 38.32B | estimated | unknown | 87.72 GB | long |
| 9 | SemantaAI/semantaai-fx-majors | economics_finance | 24.73B | estimated | unknown | 63.72 GB | long |
| 10 | LeadLagForecasting/llf_github [default] | web_cloud | 23.75B | estimated | 3.01M | 1.09 GB | nested |
| 11 | MLCapstoneRideshare/rideshare-trips | mobility | 23.47B | estimated | 70.22k | 68.75 GB | events |
| 12 | nju-zhangsq/RMISC [LargeST] | mobility | 21.78B | estimated | 339 | 4.87 GB | long |
| 13 | taziksh/argo-profiles | hydrology_ocean | 20.42B | exact | 3.43M | 13.35 GB | nested |
| 14 | giggseh/kepler-lc-star-params | space_science | 19.19B | estimated | 199.68k | 96.50 GB | nested |
| 15 | cm3r/meteo-france-climatologie-quotidienne | weather_climate | 18.53B | estimated | 135 | 1.39 GB | wide |
| 16 | rogerdehe/klines-binance [spot, futures] | economics_finance | 17.31B | estimated | 5 | 57.69 GB | wide |
| 17 | nju-zhangsq/RMISC [BuildingsBenchRealCSV] | energy | 17.11B | exact | 3.59k | 199.32 MB | long |
| 18 | perctrix/StockChina-Minute | economics_finance | 15.36B | estimated | 12 | 121.47 GB | wide |
| 19 | bsaenz/dreamt | health | 9.90B | estimated | 29 | 159.25 GB | wide |
| 20 | hsilvosa/gdelt-events | society | 9.78B | estimated | unknown | 45.22 GB | events |
| 21 | autogluon/chronos_datasets [weatherbench_hourly_temperature] | weather_climate | 9.33B | estimated | 26.61k | 94.08 GB | nested |
| 22 | nyu-planet-finders/tess-spoc-2min-binned | space_science | 9.16B | estimated | 1.73M | 27.68 GB | nested |
| 23 | tidy-finance/factor-library | economics_finance | 7.72B | estimated | 2.29M | 36.25 GB | long |
| 24 | SamuelM0422/ONS-Capacity-Factor-Dataset-Full | energy | 7.19B | estimated | 11.77M | 2.11 GB | nested |
| 25 | jeremycochoy/wikimedia-pageview-timeseries | web_cloud | 6.91B | estimated | 6.74M | 9.37 GB | nested |
| 26 | Arimancy/bts-flights-weather | mobility | 6.46B | exact | 1.32k | 6.09 GB | long |
| 27 | SII-WANGZJ/Polymarket_data [trades.parquet] | economics_finance | 6.33B | exact | unknown | 45.85 GB | events |
| 28 | nju-zhangsq/RMISC [ChinaMinuteStock] | economics_finance | 5.98B | exact | 475 | 8.17 GB | long |
| 29 | rogerdehe/klines-bybit [futures] | economics_finance | 5.95B | estimated | 5 | 21.20 GB | wide |
| 30 | AmirShahbaz/CloudPerfTrace | web_cloud | 5.81B | exact | 38.68k | 17.98 GB | nested |
| 31 | patrickfleith/GOCE-satellite-telemtry [raw, 10min stats, 6h stats] | space_science | 4.95B | estimated | 12 | 45.25 GB | wide |
| 32 | gtfierro/mortar [hive-partitioned by uuid] | energy | 4.91B | estimated | 22.56k | 35.53 GB | long |
| 33 | prabakaranchandran/enterprise-dynamics-corpus-v1 | other | 4.86B | estimated | 1.68M | 8.31 GB | nested |
| 34 | youngbongbong/mobility_dataset | mobility | 4.75B | estimated | unknown | 10.28 GB | long |
| 35 | dmariaa70/GO-MO | mobility | 4.39B | estimated | unknown | 56.10 GB | long |
| 36 | syncster-data/economic-data | economics_finance | 4.35B | estimated | unknown | 29.11 GB | long |
| 37 | xxparthparekhxx/indian-stock-market-minute-data | economics_finance | 4.32B | exact | unknown | 10.51 GB | long |
| 38 | NaaVrug/weather-geo-era5 | weather_climate | 4.26B | exact | 4 | 17.86 GB | wide |
| 39 | modelpulse/model-pulse-data | web_cloud | 3.99B | estimated | 7 | 2.29 GB | long |
| 40 | tradecatlabs/binance-futures-ohlcv-2018-2026 [candles_1m] | economics_finance | 3.82B | estimated | 47 | 17.25 GB | long |

### 40 with the most series

| # | dataset | domain | series | confidence | points | median length | layout |
|---|---|---|---|---|---|---|---|
| 1 | hsilvosa/safecast-radiation | air_quality | 266.87M | estimated | 533.75M | 1 | long |
| 2 | VincentGOURBIN/FuelInFranceData | retail_demand | 75.45M | estimated | 528.18M | 1 | long |
| 3 | Donny-Guo/airline-delay | mobility | 61.28M | estimated | 802.15M | 1 | long |
| 4 | aiacademy-kg/bishkek-transport | mobility | 34.62M | estimated | 1.55B | unknown | long |
| 5 | angelosaadeh/velib | mobility | 31.43M | estimated | 251.43M | 1 | long |
| 6 | aaron-freedman/banking-concentration-data | economics_finance | 27.14M | estimated | 970.99M | unknown | long |
| 7 | Bayanat-Labs/arab-trade-exposure | economics_finance | 25.55M | estimated | 175.13M | unknown | long |
| 8 | Rekin226/aquascope-gauges [observation bundles (USGS, UK EA, BOM, Hub'Eau, PEGELONLINE, ...)] | hydrology_ocean | 13.11M | estimated | 841.40M | 1 | long |
| 9 | SamuelM0422/ONS-Capacity-Factor-Dataset-Full | energy | 11.77M | estimated | 7.19B | 512 | nested |
| 10 | Dingdong-Inc/FreshRetailNet-LT [unknown] | retail_demand | 11.06M | estimated | 530.74M | 24 | nested |
| 11 | afhubbard/gpu-prices | web_cloud | 8.32M | estimated | 49.94M | 1 | long |
| 12 | jeremycochoy/wikimedia-pageview-timeseries | web_cloud | 6.74M | estimated | 6.91B | 1.02k | nested |
| 13 | National-Digital/au-water-storage-levels | hydrology_ocean | 6.48M | estimated | 6.48M | 1 | long |
| 14 | matrosovdani/moscow-parking-occupancy | mobility | 5.10M | estimated | 15.31M | 1 | long |
| 15 | Dingdong-Inc/FreshRetailNet-50K [default] | retail_demand | 4.76M | estimated | 228.66M | 24 | nested |
| 16 | taziksh/argo-profiles | hydrology_ocean | 3.43M | estimated | 20.42B | 397 | nested |
| 17 | orailix/ride-silver | mobility | 3.11M | estimated | 437.90M | unknown | long |
| 18 | LeadLagForecasting/llf_github [default] | web_cloud | 3.01M | estimated | 23.75B | 1.01k | nested |
| 19 | nyu-planet-finders/kepler-llc | space_science | 2.37M | estimated | 40.93B | 3.45k | nested |
| 20 | tidy-finance/factor-library | economics_finance | 2.29M | estimated | 7.72B | 1.45k | long |
| 21 | kevykibbz/wind-turbine-scada-data-for-early-fault-detection [Wind Farm A/B/C] | energy | 2.27M | estimated | 2.16B | 1 | long |
| 22 | pracadata/brazil-fuel-prices | retail_demand | 1.89M | estimated | 13.25M | 1 | events |
| 23 | nyu-planet-finders/tess-spoc-2min-binned | space_science | 1.73M | estimated | 9.16B | 1.01k | nested |
| 24 | prabakaranchandran/enterprise-dynamics-corpus-v1 | other | 1.68M | estimated | 4.86B | 6.32k | nested |
| 25 | Project-AgML/usda_nass_crop_statistics | agriculture_nature | 1.61M | estimated | 36.35M | unknown | long |
| 26 | AbdullahImran/Saudi-Supply-Demand-Datasets | retail_demand | 1.32M | estimated | 390.63M | unknown | events |
| 27 | OptimalSolution/databank-ember [19 tables (generation monthly/yearly, prices, capacity, ...)] | energy | 1.27M | estimated | 7.42M | 91.40k | long |
| 28 | xus/Spain-Fuel-Prices | retail_demand | 1.04M | estimated | 14.16M | 14 | long |
| 29 | National-Digital/vic-air-quality-hourly | air_quality | 884.55k | estimated | 884.55k | 1 | long |
| 30 | thuml/UTSD [AustraliaRainfall / BeijingPM25Quality / BenzeneConcentration] | agriculture_nature | 867.26k | estimated | 2.50B | 803 | nested |
| 31 | thuml/UTSD [TDBrain] | health | 867.26k | estimated | 2.50B | 803 | nested |
| 32 | thuml/UTSD [BIDMC32HR / IEEEPPG / MotorImagery / SelfRegulationSCP1 / SelfRegulationSCP2 / PigArtPressure / PigCVP / AtrialFibrillation] | health | 867.26k | estimated | 2.50B | 803 | nested |
| 33 | thuml/UTSD [SensorData] | industrial | 867.26k | estimated | 2.50B | 803 | nested |
| 34 | thuml/UTSD [London Smart Meters / Wind Farms / Aus. Electricity Demand / Pedestrian Counts / Web Traffic / Sunspot] | other | 867.26k | estimated | 2.50B | 803 | nested |
| 35 | thuml/UTSD [ERA5 Surface / ERA5 Pressure] | weather_climate | 867.26k | estimated | 2.50B | 803 | nested |
| 36 | National-Digital/nsw-traffic-volume-yearly | mobility | 276.88k | estimated | 2.22M | 1 | long |
| 37 | Salesforce/lotsa_data [era5] | weather_climate | 239.52k | estimated | 94.16B | 8.74k | nested |
| 38 | JrVillabona/resonance-data | economics_finance | 235.06k | estimated | 1.10B | 95.49k | events |
| 39 | autogluon/chronos_datasets [weatherbench_daily] | weather_climate | 225.28k | exact | 3.29B | 14.61k | nested |
| 40 | autogluon/chronos_datasets [weatherbench_weekly] | weather_climate | 225.28k | exact | 470.16M | 2.09k | nested |

## 7. Datasets with covariates

284 datasets have candidate covariate columns (numeric or boolean columns next to the target; list columns next to the target for nested data). Known-in-future / past-only is filled only where the column names make it clear.

| dataset | covariate columns | known in future | past only |
|---|---|---|---|
| alaminxpro/bd-food-prices | data_coverage_recent; index_confidence_score; spatially_interpolated; lentils; oil; rice; wheat_flour; o_lentils; h_lentils; l_lentils; c_lentils; inflation_lentils; trust_lentils; o_oil; h_oil; l_oil; c_oil; inflation_oil; trust_oil; o_rice; h_rice; l_rice; c_rice; inflation_rice; trust_rice; o_wheat_flour; h_wheat_flour; l_wheat_flour; c_wheat_flour; inflation_wheat_flour; trust_wheat_flour; o_food_price_index; h_food_price_index; l_food_price_index; c_food_price_index; inflation_food_price_index; trust_food_price_index | - | - |
| alexroz/CarbonFluxBench | temperature_2m; skin_temperature; soil_temperature_level_1; soil_temperature_level_2; soil_temperature_level_3; soil_temperature_level_4; lake_bottom_temperature; lake_ice_depth; lake_ice_temperature; lake_mix_layer_depth; lake_mix_layer_temperature; lake_shape_factor; lake_total_layer_temperature; snow_albedo; snow_cover; snow_density; snow_depth; snow_depth_water_equivalent; snowfall_sum; snowmelt_sum; temperature_of_snow_layer; skin_reservoir_content; volumetric_soil_water_layer_1; volumetric_soil_water_layer_2; volumetric_soil_water_layer_3; volumetric_soil_water_layer_4; forecast_albedo; surface_latent_heat_flux_sum; surface_net_solar_radiation_sum; surface_net_thermal_radiation_sum; surface_sensible_heat_flux_sum; surface_solar_radiation_downwards_sum; surface_thermal_radiation_downwards_sum; evaporation_from_bare_soil_sum; evaporation_from_open_water_surfaces_excluding_oceans_sum; evaporation_from_the_top_of_canopy_sum; evaporation_from_vegetation_transpiration_sum; potential_evaporation_sum; runoff_sum; snow_evaporation_sum; ... (+109 more) | forecast_albedo; forecast_albedo_min; forecast_albedo_max | - |
| dar4huggingface/sniim-mexico-sample | Precio Max; Precio Frec | - | - |
| DHIVAKARNAIDUSUNKARI/mandi-india-daily-prices-2001-2026 | Max_Price; Modal_Price | - | - |
| electricsheepafrica/africa-wfp-food-prices-for-uganda | usdprice | - | - |
| electricsheepafrica/wfp-nigeria-food-prices | usdprice | - | - |
| hesandism/colombo-tea-auction-prices | temperature_2m_mean_mean; precipitation_sum_total_lag1; sunshine_duration_total_lag1; temperature_2m_mean_mean_lag1; precipitation_sum_total_lag2; sunshine_duration_total_lag2; temperature_2m_mean_mean_lag2; precipitation_sum_total_lag3; sunshine_duration_total_lag3; temperature_2m_mean_mean_lag3; sale_number; sale_year; avg_weather_severity; fx_usd | - | precipitation_sum_total_lag1; sunshine_duration_total_lag1; temperature_2m_mean_mean_lag1; precipitation_sum_total_lag2; sunshine_duration_total_lag2; temperature_2m_mean_mean_lag2; precipitation_sum_total_lag3; sunshine_duration_total_lag3; temperature_2m_mean_mean_lag3 |
| National-Digital/nt-crocodile-captures | head_length_cm; snout_vent_length_cm | - | - |
| Project-AgML/usda_nass_crop_statistics | cv_pct; is_other_counties | is_other_counties | - |
| Roshangupta007/YieldFormer-DE-data | precipitation_height; radiation_global; temperature_air_max_2m; temperature_air_min_2m; wind_speed; ndvi; evi; lai; ndvi_observed; ndvi_stale; ndvi_age_days; evi_observed; evi_stale; evi_age_days; lai_observed; lai_stale; lai_age_days; doy; yield_dt_per_ha; bulk_density; clay; ph; sand; soc; sand_imputed; clay_imputed; soc_imputed; ph_imputed; bulk_density_imputed; soil_imputed; slope_mean; slope_std; aspect_cos_mean; aspect_sin_mean; sos_doy; eos_doy; los_days; peak_doy; peak_ndvi; amplitude_ndvi; ... (+31 more) | - | ndvi_observed; evi_observed; lai_observed |
| taziksh/itrdb-tree-rings | value_raw; precision; stop_marker | - | - |
| taziksh/usgs-earthquakes | mag; nst; gap; dmin; rms; horizontalError; depthError; magError; magNst; felt; cdi; mmi; tsunami; sig; tz | - | - |
| teamSynapse/sri-lanka-weekly-vegetable-retail-farmer-price-dataset | lanka_auto_diesel_price; mean_apparent_temperature; rain_sum; usd_exchange_rate; mean_farmer_price; retail_price; Unnamed: 15 | - | - |
| Aduitya9/Aduitya_dextra-aqi-india | horizon; persistence; plain_seed42; plain_seed43; plain_seed44; oversampled_seed42; oversampled_seed43; oversampled_seed44; weighted_seed42; weighted_seed43; weighted_seed44; quantile_seed42; quantile_seed43; quantile_seed44; gbm; chronos-bolt median; chronos-bolt q90; chronos-2 median; chronos-2 q90; chronos-2-weather median; chronos-2-weather q90; evening_seed42; evening_seed43; evening_seed44; gbm evening; chronos-2-evening median; chronos-2-evening q90 | - | - |
| AkshayKarthick/india-weather-air-quality | temperature_2m_min; temperature_2m_mean; apparent_temperature_max; precipitation_sum; precipitation_hours; relative_humidity_2m_mean; wind_speed_10m_max; wind_direction_10m_dominant; shortwave_radiation_sum; et0_fao_evapotranspiration | - | - |
| CMU-OSPO/COVID_air_quality_RAMP_datasets | SO2_Braddock_Farm; NO_Braddock_Farm; NO2_Braddock_Farm; O3_Braddock_Farm; T_Braddock_Farm; RH_Braddock_Farm; P_Braddock_Farm; PM025_Braddock_Farm; CO_Ds_Six_Packs_and_Dogz; SO2_Ds_Six_Packs_and_Dogz; NO_Ds_Six_Packs_and_Dogz; NO2_Ds_Six_Packs_and_Dogz; O3_Ds_Six_Packs_and_Dogz; T_Ds_Six_Packs_and_Dogz; RH_Ds_Six_Packs_and_Dogz; P_Ds_Six_Packs_and_Dogz; PM025_Ds_Six_Packs_and_Dogz; CO_The_Cube; SO2_The_Cube; NO_The_Cube; NO2_The_Cube; O3_The_Cube; T_The_Cube; RH_The_Cube; P_The_Cube; PM025_The_Cube; CO_ACHD_North_Braddock; SO2_ACHD_North_Braddock; NO_ACHD_North_Braddock; NO2_ACHD_North_Braddock; O3_ACHD_North_Braddock; T_ACHD_North_Braddock; RH_ACHD_North_Braddock; P_ACHD_North_Braddock; PM025_ACHD_North_Braddock; CO_Swissvale; SO2_Swissvale; NO_Swissvale; NO2_Swissvale; O3_Swissvale; ... (+670 more) | - | - |
| danielrosehill/Jerusalem-Air-Quality-Shabbat | dow; is_friday; is_shabbat; shabbat_or_chag_exact; is_yom_tov; is_yom_kippur; is_pre_shabbat; is_post_havdalah | dow; is_friday; is_shabbat; is_yom_tov; is_yom_kippur; is_pre_shabbat; is_post_havdalah | - |
| dmariaa70/METRAQ-Air-Quality | utm_x; utm_y; is_interpolated | is_interpolated | - |
| haidang2405/aqshift-us-leakage-safe-air-quality | origin_day_of_week; origin_day_of_year; origin_month; days_since_previous_observation; previous_observation_value; prior_observation_count_28d; lag_1_value; lag_2_value; lag_3_value; lag_7_value; lag_14_value; lag_28_value; target_value_float; audit_observation_count; audit_observation_percent | - | lag_1_value; lag_2_value; lag_3_value; lag_7_value; lag_14_value; lag_28_value |
| hsilvosa/safecast-radiation | height | - | - |
| khajlk/copenhagen-air-quality-hourly | wind_speed_10m; wind_direction_10m; relative_humidity_2m; boundary_layer_height | - | - |
| meganariley/daily-aqi | site_num; mdl; aqi | - | - |
| National-Digital/act-air-quality | o3_1hr; o3_4hr; o3_8hr; co; pm10_1hr; pm2_5_1hr; pm10; pm2_5; aqi_co; aqi_no2; aqi_o3_1hr; aqi_o3_4hr; aqi_o3_8hr; aqi_pm10; aqi_pm2_5; aqi_site | - | - |
| nju-zhangsq/RMISC [PM25FiveCities] | PM_5th Middle School; PM_US Post; DEWP; HUMI; PRES; TEMP; Iws; precipitation; Iprec; PM_Jingan; PM_Xuhui; PM_Caotangsi; PM_Shahepu; PM_Taiyuanjie; PM_Xiaoheyan; PM_Dongsi; PM_Dongsihuan; PM_Nongzhanguan | - | - |
| nju-zhangsq/RMISC [USAirPollution] | NO2 1st Max Value; NO2 1st Max Hour; NO2 AQI; O3 Mean; O3 1st Max Value; O3 1st Max Hour; O3 AQI; SO2 Mean; SO2 1st Max Value; SO2 1st Max Hour; CO Mean; CO 1st Max Value; CO 1st Max Hour | - | - |
| nju-zhangsq/RMISC [XiamenAirQuality] | NO2浓度（μg/m3）; PM10的1小时均值（μg/m3）; CO浓度（mg/m3）; O3 的1小时平均浓度（μg/m3）; PM2.5的1小时平均浓度（μg/m3） | - | - |
| Skywalker1910/skynet-aqi-panel | rh_pct; dewpoint_c; pressure_hpa; precip_mm; wind_speed_kmh; wind_dir_deg; wind_gust_kmh; cloud_pct; swrad_wm2; wind_u; wind_v; dewpoint_depression_c | - | - |
| steven0226/air-quality | n_valid | - | - |
| aaron-freedman/banking-concentration-data | ent_type_cd; assets; cash; securities; ln_tot_gross; ln_tot; llres; trad_ass; ffrepo_ass; ffsold; repo_purch; fixed_ass; oreo; oth_assets; oth_assets_alt; demand_deposits; deposits; ffrepo_liab; ffpurch; repo_sold; trad_liab; othbor_liab; subdebt; liab_oth; liab_tot; liab_tot_unadj; minorint; pref_stock; comm_stock; surplus; retain_earn; equity; ln_re; ln_fi; ln_dep_inst; ln_dep_inst_dom; ln_oth; ln_cc; ln_ci; ln_cons; ... (+99 more) | - | - |
| abel-lab/finance1k | valid | - | - |
| aikstockdata/korea-equity-daily | volume | - | - |
| Bayanat-Labs/arab-trade-exposure | quantity_tonnes; source_record_ordinal; product_is_standard_hs6; exporter_in_scope; importer_in_scope; exporter_current_member; importer_current_member; quantity_missing; quantity_positive | - | - |
| empulse-ai/empulse-street-fx-index | in_progress | - | - |
| HexQuant/Stocks-Daily-Price | high; low; close; volume; adj_close | - | - |
| jwigginton/timeseries-1mn-sp500 [default] | high; low; close; volume | - | - |
| nju-zhangsq/RMISC [ChinaMinuteStock] | open; high; low; volume; money; avg; high_limit; low_limit; pre_close; paused; factor | - | - |
| nju-zhangsq/RMISC [CSI500] | open; high; low; volume; adj_factor | - | - |
| nju-zhangsq/RMISC [StockFactorsCleaned] | date; industry_id; tech_open; tech_high; tech_low; tech_volume; tech_return_1d; tech_gap_return_1d; tech_return_5d; tech_intraday_range; tech_price_range; tech_sma_5; tech_sma_10; tech_sma_20; tech_ema_10; tech_ema_20; tech_dema_10; tech_tema_10; tech_mom_10; tech_roc_10; tech_rsi_14; tech_cmo_14; tech_macd; tech_macd_hist; tech_stoch_k; tech_stoch_d; tech_atr_14; tech_std_5; tech_std_10; tech_bb_upper; tech_bb_lower; tech_bb_width; tech_obv; tech_volume_roc; tech_volume_std_5; tech_ad; tech_vr; tech_tail_ratio; tech_upper_shadow; tech_lower_shadow; ... (+27 more) | - | - |
| nju-zhangsq/RMISC [StockMarketData] | Volume; mom; mom1; mom2; mom3; ROC_5; ROC_10; ROC_15; ROC_20; EMA_10; EMA_20; DTB4WK; DTB3; DTB6; DGS5; DGS10; Oil; Gold; DAAA; DBAA; AAPL; AMZN; GE; JNJ; JPM; MSFT; WFC; XOM; FCHI; DJI; IXIC; S&P; NYSE; TE1; TE2; TE3; TE5; TE6; DE1; DE2; ... (+31 more) | - | - |
| nju-zhangsq/RMISC [TushareETFDaily] | trade_date; open; high; low; pre_close; change; pct_chg; vol; amount | - | - |
| nju-zhangsq/RMISC [TushareIndexDaily] | trade_date; open; high; low; pre_close; change; pct_chg; vol; amount | - | - |
| nju-zhangsq/RMISC [TushareStockDaily] | trade_date; open; high; low; pre_close; change; pct_chg; vol; amount; location | - | - |
| nju-zhangsq/RMISC [TushareStockWeekly] | trade_date; open; high; low; pre_close; change; pct_chg; vol; amount; location | - | - |
| nju-zhangsq/RMISC [UKEconomy] | 11; 19; 30; 5; Canadian dollar; Swedish kroner; Swiss Franc; US dollar; 6; 9; 20; 14; 7; 58; 18; 2; 4; 15; 16; 22; 26; 13; 17; 3; 62; 27; 80; 51; 60; 21; 1; 39; 12; 10; 32; 8; 33; 68; 45; 24; ... (+75 more) | - | - |
| prasad-gade05/india-upi-ecosystem-2018-2025 | total_txn_amount; avg_txn_value; national_percentile | - | - |
| prospex-ch/swiss-commercial-registry-monthly-panel | capital_suppressed | - | - |
| Rrishab/OHLCV-1m [default] | high; low; close; volume | - | - |
| showlcw/china-cpi-ppi-pmi | momPct; yoyPct; ytdYoyPct | - | - |
| SII-WANGZJ/Polymarket_data [trades.parquet] | log_index; price; usd_amount; token_amount | - | - |
| thanna94/us-building-permits | SF_Units; MF_2Unit; MF_34Unit; MF_5Plus; MF_Total; Total_Units; MF_Pct | - | - |
| thecharttruth/etf-data [prices] | high; low; close; volume | - | - |
| tidy-finance/factor-library | cost; microcap_share | - | - |
| Traders-Lab/TroveLedger [daily / hourly / 1-minute files] | high; low; close; volume | - | - |
| trentmkelly/polymarket_historical_data [market_snapshots] | best_bid; best_ask | - | - |
| xxparthparekhxx/indian-stock-market-minute-data | high; low; close; volume; oi | - | - |
| yadavsaurabh/us-stocks-and-search-trends | High; Low; Close; Adj Close; Volume; Dividends; Stock Splits; Capital Gains | - | - |
| yugalinks/import-export-market-demand-v1 | latest_data_year; years_with_data; imports_latest_observed_usd; imports_all_year_avg_usd; imports_avg_3y_usd; imports_avg_5y_usd; imports_avg_10y_usd; imports_cagr_3y_pct; imports_cagr_5y_pct; imports_cagr_10y_pct; imports_historical_peak_usd; imports_first_observed_usd; historical_peak_year; latest_vs_all_avg_pct; latest_below_peak_pct; yoy_growth_latest_pct; yoy_growth_all_avg_pct; positive_growth_year_share_pct; supplier_count_latest; market_hhi_latest; top_supplier_share_pct; data_freshness_years; coverage_score; trend_acceleration_pct | - | imports_latest_observed_usd; imports_first_observed_usd |
| yuiseki/maddison-project | pop | - | - |
| yuiseki/pwt | rgdpo; pop; emp; avh; hc; ccon; cda; cgdpe; cgdpo; cn; ck; ctfp; cwtfp; rgdpna; rconna; rdana; rnna; rkna; rtfpna; rwtfpna; labsh; irr; delta; xr; pl_con; pl_da; pl_gdpo; cor_exp; csh_c; csh_i; csh_g; csh_x; csh_m; csh_r; pl_c; pl_i; pl_g; pl_x; pl_m; pl_n; ... (+1 more) | - | - |
| yuiseki/undata-sdmx | UNIT_MULT; CONVERSION_FACTOR | - | - |
| ashwingopalsamy/central-bank-policy-rates | rate_bps; rate_low_pct; rate_high_pct; is_change | is_change | - |
| kavelaltd/historysaid-global-economic | is_estimate | is_estimate | - |
| tradecatlabs/binance-futures-ohlcv-2018-2026 [candles_1m] | high; low; close; volume; quote_volume; trade_count; taker_buy_volume; taker_buy_quote_volume | - | - |
| zalizedata/global-macro-trade-indicators-dataset | is_aggregate | is_aggregate | - |
| JrVillabona/resonance-data | high; low; close; volume | - | - |
| SemantaAI/semantaai-fx-majors | high; low; close; volume; ret_1; ret_3; ret_12; rv_12; rv_48; atr_14_pct; range_pct; body_pct; vol_z_48; dev_sma_20_pct; dev_sma_20_z; dev_sma_20_z2; dev_sma_20_outside2std; dev_sma_20_mr_score; dev_ema_20_pct; dev_ema_20_z; dev_ema_20_z2; dev_ema_20_outside2std; dev_ema_20_mr_score; dev_dema_20_pct; dev_dema_20_z; dev_dema_20_z2; dev_dema_20_outside2std; dev_dema_20_mr_score; dev_tema_20_pct; dev_tema_20_z; dev_tema_20_z2; dev_tema_20_outside2std; dev_tema_20_mr_score; dev_sma_50_pct; dev_sma_50_z; dev_sma_50_z2; dev_sma_50_outside2std; dev_sma_50_mr_score; dev_ema_50_pct; dev_ema_50_z; ... (+397 more) | ctx_dow_sin; ctx_dow_cos; ctx_is_weekend | ret_past_h3; ret_past_h6; ret_past_h12; ret_past_h24; ret_past_h48; ret_past_h96; ret_past_h288; ret_past_h1; ret_past_h2; ret_past_h4; ret_past_h5; ret_past_h8; ret_past_h10; ret_past_h16; ret_past_h20; ret_past_h30; ret_past_h32 |
| yadavsaurabh/macro-markets | High; Low; Close; Adj Close; Volume; Dividends; Stock Splits; Capital Gains | - | - |
| ai-iot/EnergyBench [DGS] | Impound Lot #1 & Fleet Fueling Site; Jefferson Playing Fields; DC Village; 200 I Street Municipal Building; C.W. Harris Elementary School; Nalle Elementary School; Fort Davis Recreation Center; Kimball Elementary School; Income Maintenance Administration Office; Washington Highlands Library; Leckie Elementary School; King Elementary School; Simon Elementary School; Southeast Tennis & Learning Center; Ballou High & Ballou STAY High School; River Terrace Education Campus; 6th District Headquarters; Kenilworth Recreation Center; North Michigan Park Recreation Center; Taft Recreation Center; Department of Human Services #2; Noyes Elementary School; Engine Company #26; Luke C. Moore High School; Engine Company #17; Edgewood Recreation Center; Fleet Fueling Site #2; Palisades Recreation Center; Fort Lincoln Recreation Center; Spingarn High School; Trinidad Recreation Center; DPW Vehicle Maintenance Facility 2A; New York Ave Bus Lot; Youth Services Administration Headquarters; Engine Company #10; 5th St. Bus Lot; Lemmuel Penn Center; Engine Company #12; Langley Elementary School; Emery Work Bed Program at Old Emery Elementary; ... (+279 more) | - | - |
| bsebench-org/nasa-rw-2014 [partitioned by cell_id] | current_A; temperature_C; cycle_number; capacity_Ah; soc_truth; soh_truth | - | - |
| bsebench-org/warwick-second-life-dm-2025-raw [raw mirror (746 csv, 309 mat)] | missing; size_mismatch | - | - |
| gft/ttm4hvac-source-all-train | Heating Setpoint (C); Cooling Setpoint (C); Room Air Temperature (C); Outdoor Humidity (%); Wind Speed (m/s); Direct Solar Radiation (W/m^2); HVAC Power Consumption (W); is_default | is_default | - |
| hsilvosa/omie-electricity | price_eur_mwh | - | - |
| Jainam03/RESCAST-100k-ECOBEE | Cooling Setpoint; Heating Setpoint; Outdoor Temp; Relative Humidity | - | - |
| kevykibbz/wind-turbine-scada-data-for-early-fault-detection [Wind Farm A/B/C] | sensor_0_max; sensor_0_min; sensor_0_std; sensor_1_avg; sensor_1_max; sensor_1_min; sensor_1_std; power_2_avg; power_2_max; power_2_min; power_2_std; sensor_3_avg; sensor_3_max; sensor_3_min; sensor_3_std; sensor_4_avg; sensor_4_max; sensor_4_min; sensor_4_std; power_5_avg; power_5_max; power_5_min; power_5_std; power_6_avg; power_6_max; power_6_min; power_6_std; sensor_7_avg; sensor_7_max; sensor_7_min; sensor_7_std; sensor_8_avg; sensor_8_max; sensor_8_min; sensor_8_std; sensor_9_avg; sensor_9_max; sensor_9_min; sensor_9_std; sensor_10_avg; ... (+911 more) | - | - |
| michealsmitch/lithium-ion-cell-capacity-grading-process-curves [train] | tray_position; cycle; step_no; accumulated_time_min; voltage_mv; current_ma; capacity_mah; energy_mwh; end_flag; temperature_c; accumulated_capacity_mah; negative_pressure; capacity_mah_bat; ambient_temperature_c; probe_voltage_mv; accumulated_energy_mwh; negative_electrode_temperature_c; cable_voltage_mv; contact_voltage_mv; contact_impedance; is_loaded | is_loaded | - |
| Narendersingh007/india-solar-benchmark-dataset | DY; HR; ALLSKY_SFC_SW_DWN; T2M; RH2M; WS10M; PRECTOTCORR; ALLSKY_KT; T2MDEW; PS; WS2M; WD10M; ALLSKY_SFC_LW_DWN; ALLSKY_SFC_PAR_TOT; dayofyear; weekday; is_weekend; SOLAR_ZENITH; SOLAR_AZIMUTH; CLEARSKY_GHI; CLEARSKY_DNI; CLEARSKY_DHI; AIR_MASS; hour_sin; hour_cos; month_sin; month_cos; dayofyear_sin; dayofyear_cos; TEMP_HUMIDITY; DEWPOINT_SPREAD; WIND_POWER; PRESSURE_TEMP; CLEARNESS_TEMP; CLEARNESS_HUMIDITY; TEMP_WIND; lag_1; lag_3; lag_6; lag_12; ... (+7 more) | weekday; is_weekend | lag_1; lag_3; lag_6; lag_12; lag_24; lag_168 |
| nju-zhangsq/RMISC [ApplianceEnergy] | Tdewpoint; RH_8; date; RH_3; RH_7; T7; RH_5; T1; RH_6; RH_9; T8; T4; T_out; Press_mm_hg; T2; RH_1; RH_out; T9; RH_4; RH_2; T5; Visibility; T3; Windspeed; T6 | - | - |
| nju-zhangsq/RMISC [BritainCoal] |  frequency;  coal;  nuclear;  ccgt;  wind;  pumped;  hydro;  biomass;  solar | - | - |
| nju-zhangsq/RMISC [BuildingsBenchRealCSV] | Global_active_power; Bear_public_Orville; Bear_education_Lidia; Bear_education_Nanette; Bear_education_Lewis; Bear_science_Alison; Bear_education_Darrell; Bear_education_Thuy; Bear_lodging_Erick; Bear_education_Mara; Bear_education_Bulah; Bear_lodging_Evan; Bear_education_Iris; Bear_education_Rebecca; Bear_education_Herb; Bear_education_Tara; Bear_education_Augusta; Bear_education_Maryjane; Bear_education_Curtis; Bear_education_Arnold; Bear_education_Chana; Bear_education_Sandy; Bear_parking_Bridget; Bear_education_Holly; Bear_education_Gavin; Bear_education_Paola; Bear_education_Santos; Bear_parking_Bruce; Bear_education_Deena; Bear_education_Lorie; Bear_education_Merrill; Bear_education_Oscar; Bear_assembly_Jose; Bear_education_Marta; Bear_lodging_Dannie; Bear_education_Danna; Bear_education_Destiny; Bear_education_Patricia; Bear_assembly_Danial; Bear_education_Lila; ... (+908 more) | - | - |
| nju-zhangsq/RMISC [LBNL] | wifi_fourth_south; air_temp_set_1; air_temp_set_2; dew_point_temperature_set_1d; relative_humidity_set_1; solar_radiation_set_1; occ_third_south; occ_fourth_south; mels_S; lig_S; mels_N; hvac_N; hvac_S; zone_016_cooling_sp; zone_017_cooling_sp; zone_018_cooling_sp; zone_019_cooling_sp; zone_021_cooling_sp; zone_023_cooling_sp; zone_024_cooling_sp; zone_025_cooling_sp; zone_026_cooling_sp; zone_027_cooling_sp; zone_030_cooling_sp; zone_032_cooling_sp; zone_035_cooling_sp; zone_036_cooling_sp; zone_037_cooling_sp; zone_038_cooling_sp; zone_039_cooling_sp; zone_041_cooling_sp; zone_042_cooling_sp; zone_043_cooling_sp; zone_046_cooling_sp; zone_047_cooling_sp; zone_048_cooling_sp; zone_049_cooling_sp; zone_051_cooling_sp; zone_053_cooling_sp; zone_054_cooling_sp; ... (+196 more) | - | - |
| nju-zhangsq/RMISC [MZVAV] | AHU: Mixed Air Temperature; AHU: Return Air Temperature; AHU: Supply Air Fan Speed Control Signal; AHU: Outdoor Air Damper Control Signal  ; AHU: Return Air Damper Control Signal; AHU: Exhaust Air Damper Control Signal    ; AHU: Cooling Coil Valve Control Signal; AHU: Heating Coil Valve Control Signal; Occupancy Mode Indicator; Fault Detection Ground Truth; AHU: Supply Air Temperature Set Point; AHU: Outdoor Air Temperature; AHU: Supply Air Fan Status; AHU: Return Air Fan Status; AHU: Return Air Fan Speed Control Signal; AHU: Exhaust Air Damper Control Signal  ; AHU: Supply Air Duct Static Pressure ; AHU: Supply Air Temperature Heating Set Point; AHU: Supply Air Temperature Cooling Set Point; RTU: Supply Air Fan Status; RTU: Electricity; RTU: Natural Gas; Terminal: Room 102 Air Humidity; Terminal: Room 103 Air Humidity; Terminal: Room 104 Air Humidity; Terminal: Room 105 Air Humidity; Terminal: Room 106 Air Humidity; Terminal: Room 202 Air Humidity; Terminal: Room 203 Air Humidity; Terminal: Room 204 Air Humidity; Terminal: Room 205 Air Humidity; Terminal: Room 206 Air Humidity; Terminal: Room 102 Air Temperature; Terminal: Room 103 Air Temperature; Terminal: Room 104 Air Temperature; Terminal: Room 105 Air Temperature; Terminal: Room 106 Air Temperature; Terminal: Room 202 Air Temperature; Terminal: Room 203 Air Temperature; Terminal: Room 204 Air Temperature; ... (+22 more) | - | - |
| nju-zhangsq/RMISC [OccupancyDetection] | Occupancy; Light; Humidity; HumidityRatio | - | - |
| nju-zhangsq/RMISC [OPSD-Household] | DE_KN_residential2_dishwasher; DE_KN_residential2_freezer; DE_KN_residential2_grid_import; DE_KN_residential2_washing_machine; DE_KN_residential3_circulation_pump; DE_KN_residential3_dishwasher; DE_KN_residential3_freezer; DE_KN_residential3_grid_export; DE_KN_residential3_grid_import; DE_KN_residential3_pv; DE_KN_residential3_refrigerator; DE_KN_residential3_washing_machine; DE_KN_residential5_dishwasher; DE_KN_residential5_grid_import; DE_KN_residential5_refrigerator; DE_KN_residential5_washing_machine; DE_KN_industrial2_grid_import; DE_KN_industrial2_pv; DE_KN_industrial2_storage_charge; DE_KN_industrial2_storage_decharge; DE_KN_residential6_circulation_pump; DE_KN_residential6_dishwasher; DE_KN_residential6_freezer; DE_KN_residential6_grid_export; DE_KN_residential6_grid_import; DE_KN_residential6_pv; DE_KN_residential6_washing_machine; DE_KN_public2_grid_import; DE_KN_industrial1_grid_import; DE_KN_industrial1_pv_1; DE_KN_industrial1_pv_2; DE_KN_public1_grid_import; DE_KN_residential1_dishwasher; DE_KN_residential1_freezer; DE_KN_residential1_grid_import; DE_KN_residential1_heat_pump; DE_KN_residential1_pv; DE_KN_residential1_washing_machine; DE_KN_residential4_dishwasher; DE_KN_residential4_ev; ... (+27 more) | - | - |
| nju-zhangsq/RMISC [OPSD-PV-Wind] | RS_pv_national_current; SK_pv_national_current; SK_wind_national_current; SI_pv_national_current; SI_wind_national_current; BG_pv_national_current; BG_wind_national_current; MD_pv_national_current; PT_pv_national_current; PT_wind_national_current; PL_pv_national_current; PL_wind_national_current; PL_wind_national_near-termfuture; BA_pv_national_current; CY_pv_national_current; CY_wind_national_current; HU_pv_national_current; HU_wind_national_current; LU_pv_national_current; LU_wind_national_current; LU_wind_national_near-termfuture; DE_pv_national_current; DE_wind_national_current; DE_wind_offshore_current; DE_wind_onshore_current; DE_wind_national_near-termfuture; DE_wind_offshore_near-termfuture; DE_wind_onshore_near-termfuture; IT_pv_national_current; IT_wind_national_current; IT_wind_national_near-termfuture; IT_wind_offshore_near-termfuture; IT_wind_onshore_near-termfuture; ME_pv_national_current; EE_pv_national_current; EE_wind_national_current; RO_pv_national_current; RO_wind_national_current; RO_wind_national_near-termfuture; FR_pv_national_current; ... (+81 more) | - | - |
| nju-zhangsq/RMISC [OPSD-When2Heat] | LV_COP_ASHP_radiator; LV_COP_ASHP_water; LV_COP_GSHP_floor; LV_COP_GSHP_radiator; LV_COP_GSHP_water; LV_COP_WSHP_floor; LV_COP_WSHP_radiator; LV_heat_demand_space; LV_heat_demand_space_COM; LV_heat_demand_space_MFH; LV_heat_demand_space_SFH; LV_heat_demand_total; LV_heat_demand_water; LV_heat_demand_water_COM; LV_heat_demand_water_MFH; LV_heat_demand_water_SFH; LV_heat_profile_space_COM; LV_heat_profile_space_MFH; LV_heat_profile_space_SFH; LV_heat_profile_water_COM; LV_heat_profile_water_MFH; LV_heat_profile_water_SFH; NL_COP_ASHP_floor; NL_COP_ASHP_radiator; NL_COP_ASHP_water; NL_COP_GSHP_floor; NL_COP_GSHP_radiator; NL_COP_GSHP_water; NL_COP_WSHP_floor; NL_COP_WSHP_radiator; NL_heat_demand_space; NL_heat_demand_space_COM; NL_heat_demand_space_MFH; NL_heat_demand_space_SFH; NL_heat_demand_total; NL_heat_demand_water; NL_heat_demand_water_COM; NL_heat_demand_water_MFH; NL_heat_demand_water_SFH; NL_heat_profile_space_COM; ... (+585 more) | - | - |
| nju-zhangsq/RMISC [Pvdaq] | meter_ac_output_(kwatts)_meter_150161; wind_speed_10m_(mph)_o_150229; ambient_temperature_o_149575; wind_speed_o_149576; wind_direction_o_149577; inv_string04_ac_voltage_(v)_inv_150181; inv_string04_temperature_(c)_inv_150184; inv_string03_ac_voltage_(v)_inv_150175; inv_string03_dc_voltage_(v)_inv_150174; inv_string03_temperature_(c)_inv_150178; inv_string11_ac_voltage_(v)_inv_150223; inv_string11_temperature_(c)_inv_150226; inv_string08_ac_voltage_(v)_inv_150205; inv_string08_temperature_(c)_inv_150208; inv_string01_ac_voltage_(v)_inv_150163; inv_string05_ac_voltage_(v)_inv_150187; inv_string05_temperature_(c)_inv_150190; inv_string10_ac_voltage_(v)_inv_150217; inv_string10_dc_voltage_(v)_inv_150216; inv_string10_temperature_(c)_inv_150220; inv_string07_ac_voltage_(v)_inv_150199; inv_string07_dc_voltage_(v)_inv_150198; inv_string07_temperature_(c)_inv_150202; inv_string06_ac_output_(kwh)_inv_150194; inv_string06_ac_voltage_(v)_inv_150193; inv_string06_dc_voltage_(v)_inv_150192; inv_string06_temperature_(c)_inv_150196; inv_string09_ac_voltage_(v)_inv_150211; inv_string02_ac_voltage_(v)_inv_150169; inv_string02_temperature_(c)_inv_150172 | - | - |
| nju-zhangsq/RMISC [RoomOccupancy] | S2_Temp; S3_Sound; S7_PIR; S3_Light; S4_Temp; S4_Light; S2_Sound; S6_PIR; S1_Temp; S1_Sound; S4_Sound; S5_CO2_Slope; S1_Light; S3_Temp; S2_Light; S5_CO2 | - | - |
| nju-zhangsq/RMISC [ShellHackathon] | Moisture; Azimuth Angle [degrees]; Station Pressure [mBar]; Snow Depth [cm]; Total Cloud Cover [%]; Tower Wet Bulb Temp [deg C]; Precipitation (Accumulated) [mm]; Direct sNIP [W/m^2]; Tower RH [%]; Tower Dry Bulb Temp [deg C]; Tower Dew Point Temp [deg C]; Albedo (CMP11); Avg Wind Direction @ 6ft [deg from N]; Peak Wind Speed @ 6ft [m/s] | - | - |
| nju-zhangsq/RMISC [TetuanPowerConsumption] | Humidity; Wind Speed; diffuse flows; general diffuse flows; Temperature | - | - |
| nju-zhangsq/RMISC [UK-DALE] | channel_3_button_press; channel_22_button_press; channel_14_button_press; channel_36_button_press; channel_39_button_press; channel_12_button_press; channel_13_button_press; channel_4_button_press; channel_9_button_press; channel_41_button_press; channel_40_button_press | - | - |
| nju-zhangsq/RMISC [WindElec] | cov2; cov7; cov4; cov11; cov13; cov8; cov9; cov3; cov12; cov5; cov10; cov6 | - | - |
| PowerZooJax/PowerZooDataset [ausgrid_zone_substation] | fiscal_year | - | - |
| SamuelM0422/ONS-Capacity-Factor-Dataset-Full | static_categorical_values; static_real_values; future_values; id; past_observed_mask; future_observed_mask | - | past_observed_mask; future_observed_mask |
| skforecast/ashrae_daily | square_feet; air_temperature; dew_temperature; sea_level_pressure; wind_direction; wind_speed | - | - |
| ai-iot/EnergyBench [real subsets overall (Dataset_V0.0)] | sp20002; sp20003; sp20004; sp20005; sp20006; sp20007; sp20008; sp20009; sp20010; sp20011; sp20012; sp20013; sp20014; sp20015; sp20016; sp20017; sp20018; sp20019; sp20020; sp20021; sp20022; sp20023; sp20024; sp20025; sp20026; sp20027; sp20028; sp20029; sp20030; sp20031; sp20032; sp20033; sp20034; sp20035; sp20036; sp20037; sp20038; sp20039; sp20040; sp20041; ... (+959 more) | - | - |
| alinurdin/time-series-benchmark [eia_hourly, madrid_15min, webtris_15min, usgs_daily, usgs_continuous, hydat_daily, bom_daily, caravan_daily, isd_hourly, swissmetnet_10min, ghcnd_daily] | dew_point; station_pressure; sea_level_pressure; wind_speed; wind_direction | - | - |
| alinurdin/time-series-benchmark [eia_hourly] | demand_forecast | demand_forecast | - |
| alinurdin/time-series-benchmark [opsd_load] | load_forecast | load_forecast | - |
| DAG-UPB/TS-Arena-Archive [default] | q_0.1; q_0.2; q_0.3; q_0.4; q_0.5; q_0.6; q_0.7; q_0.8; q_0.9 | - | - |
| hsilvosa/entsoe-observatory [8 tables] | price_eur_min; price_eur_max; price_eur_mean; price_eur_peak_mean; price_eur_offpeak_mean; negative_price_hours; price_coverage; load_mw_min; load_mw_max; load_mw_mean; load_coverage | - | - |
| nju-zhangsq/RMISC [OPSD] | PL_load_forecast_entsoe_transparency; PL_solar_generation_actual; PL_wind_onshore_generation_actual; ME_load_actual_entsoe_transparency; ME_load_forecast_entsoe_transparency; ME_wind_onshore_generation_actual; RS_load_actual_entsoe_transparency; RS_load_forecast_entsoe_transparency; DE_LU_load_actual_entsoe_transparency; DE_LU_load_forecast_entsoe_transparency; DE_LU_solar_generation_actual; DE_LU_wind_generation_actual; DE_LU_wind_offshore_generation_actual; DE_LU_wind_onshore_generation_actual; DE_temperature; DE_radiation_direct_horizontal; DE_radiation_diffuse_horizontal; FR_load_actual_entsoe_transparency; FR_load_forecast_entsoe_transparency; FR_solar_generation_actual; FR_wind_onshore_generation_actual; FR_temperature; FR_radiation_direct_horizontal; FR_radiation_diffuse_horizontal; SE_2_load_actual_entsoe_transparency; SE_2_load_forecast_entsoe_transparency; SE_2_price_day_ahead; SE_2_wind_onshore_generation_actual; SE_temperature; SE_radiation_direct_horizontal; SE_radiation_diffuse_horizontal; SI_load_actual_entsoe_transparency; SI_load_forecast_entsoe_transparency; SI_solar_generation_actual; SI_wind_onshore_generation_actual; SI_temperature; SI_radiation_direct_horizontal; SI_radiation_diffuse_horizontal; DE_50hertz_load_actual_entsoe_transparency; DE_50hertz_load_forecast_entsoe_transparency; ... (+327 more) | PL_load_forecast_entsoe_transparency; ME_load_forecast_entsoe_transparency; RS_load_forecast_entsoe_transparency; DE_LU_load_forecast_entsoe_transparency; FR_load_forecast_entsoe_transparency; SE_2_load_forecast_entsoe_transparency; SI_load_forecast_entsoe_transparency; DE_50hertz_load_forecast_entsoe_transparency; NO_4_load_forecast_entsoe_transparency; SE_load_forecast_entsoe_transparency; LU_load_forecast_entsoe_transparency; ES_load_forecast_entsoe_transparency; DK_load_forecast_entsoe_transparency; IT_CNOR_load_forecast_entsoe_transparency; DE_load_forecast_entsoe_transparency; IT_CSUD_load_forecast_entsoe_transparency; BG_load_forecast_entsoe_transparency; SE_4_load_forecast_entsoe_transparency; IT_NORD_load_forecast_entsoe_transparency; FI_load_forecast_entsoe_transparency; DK_1_load_forecast_entsoe_transparency; IT_SARD_load_forecast_entsoe_transparency; IE_sem_load_forecast_entsoe_transparency; GB_UKM_load_forecast_entsoe_transparency; CY_load_forecast_entsoe_transparency; BE_load_forecast_entsoe_transparency; DE_transnetbw_load_forecast_entsoe_transparency; EE_load_forecast_entsoe_transparency; LV_load_forecast_entsoe_transparency; HU_load_forecast_entsoe_transparency; LT_load_forecast_entsoe_transparency; NO_load_forecast_entsoe_transparency; RO_load_forecast_entsoe_transparency; UA_load_forecast_entsoe_transparency; GB_NIR_load_forecast_entsoe_transparency; SK_load_forecast_entsoe_transparency; NO_3_load_forecast_entsoe_transparency; NO_5_load_forecast_entsoe_transparency; NO_2_load_forecast_entsoe_transparency; DE_tennet_load_forecast_entsoe_transparency; ... (+16 more) | PL_solar_generation_actual; PL_wind_onshore_generation_actual; ME_load_actual_entsoe_transparency; ME_wind_onshore_generation_actual; RS_load_actual_entsoe_transparency; DE_LU_load_actual_entsoe_transparency; DE_LU_solar_generation_actual; DE_LU_wind_generation_actual; DE_LU_wind_offshore_generation_actual; DE_LU_wind_onshore_generation_actual; FR_load_actual_entsoe_transparency; FR_solar_generation_actual; FR_wind_onshore_generation_actual; SE_2_load_actual_entsoe_transparency; SE_2_wind_onshore_generation_actual; SI_load_actual_entsoe_transparency; SI_solar_generation_actual; SI_wind_onshore_generation_actual; DE_50hertz_load_actual_entsoe_transparency; DE_50hertz_solar_generation_actual; DE_50hertz_wind_generation_actual; DE_50hertz_wind_offshore_generation_actual; DE_50hertz_wind_onshore_generation_actual; NO_4_load_actual_entsoe_transparency; NO_4_wind_onshore_generation_actual; SE_load_actual_entsoe_transparency; SE_wind_onshore_generation_actual; LU_load_actual_entsoe_transparency; ES_load_actual_entsoe_transparency; ES_solar_generation_actual; ES_wind_onshore_generation_actual; DK_load_actual_entsoe_transparency; DK_solar_generation_actual; DK_wind_generation_actual; DK_wind_offshore_generation_actual; DK_wind_onshore_generation_actual; IT_CNOR_load_actual_entsoe_transparency; IT_CNOR_solar_generation_actual; IT_CNOR_wind_onshore_generation_actual; DE_load_actual_entsoe_transparency; ... (+134 more) |
| OpenSynth/AI4OPT-TS4PS-RTE7000 [tables (lines, generators, loads, ...)] | x; g1; b1; g2; b2; imax1; imax2; synchronous_component; nominal_v; connected1; connected2 | - | - |
| OptimalSolution/databank-ember [19 tables (generation monthly/yearly, prices, capacity, ...)] | is_derived | is_derived | - |
| YlanHARRAULT/rte-eco2mix-regional-2025 | thermique; nucleaire; eolien; solaire; hydraulique; pompage; bioenergies; ech_physiques; stockage_batterie; destockage_batterie; eolien_terrestre; eolien_offshore; tco_thermique; tch_thermique; tco_nucleaire; tch_nucleaire; tco_eolien; tch_eolien; tco_solaire; tch_solaire; tco_hydraulique; tch_hydraulique; tco_bioenergies; tch_bioenergies | - | - |
| abmallick/heart-breath-sleep-stage-dataset | respiratory_rate; sleep_stage; hr_mean; hr_sdnn_5; hr_rmssd_5; hr_slope_3; rr_mean; rr_sd_5; rr_slope_3; hr_rr_ratio; hr_rr_product; minutes_since_start | - | - |
| ajto/GoldenCheetah | km; power; hr; cad; alt | - | - |
| EPI-Eval/canada-fluwatch | detections; percentpositive | - | - |
| EPI-Eval/cdc-nrevss-rsv | rsvtest; outlier | - | - |
| EPI-Eval/delphi-flusurv | rate_age_0; rate_age_1; rate_age_2; rate_age_3; rate_age_4 | - | - |
| EPI-Eval/nhsn-hrd | totalconfc19newadm; totalconfrsvnewadm; numinptbeds; numinptbedsocc; numicubeds; numicubedsocc; totalconfc19icupats; totalconffluicupats; totalconfflunewadmper100k; totalconfc19newadmper100k; totalconfflunewadmcumulativeseasonalsum; totalconfc19newadmcumulativeseasonalsum; totalconfrsvnewadmcumulativeseasonalsum | - | - |
| glucofmbench/GlucoFM-Bench | BGvalue | - | - |
| HHS-Official/nndss-weekly-data | MMWR WEEK; Current week; Previous 52 week Max; Cumulative YTD Current MMWR Year; Cumulative YTD Previous MMWR Year; sort_order | - | - |
| HHS-Official/weekly-rates-of-laboratory-confirmed-rsv-hospitali | Cumulative Rate | - | - |
| HHS-Official/weekly-united-states-covid-19-hospitalization-metr | pct_chg_avg_adm_all_covid_confirmed_per_100k; total_adm_all_covid_confirmed_past_7days; total_adm_all_covid_confirmed_past_7days_per_100k; sum_adm_all_covid_confirmed; avg_total_patients_hospitalized_covid_confirmed; avg_percent_inpatient_beds_occupied_covid_confirmed; abs_chg_avg_percent_inpatient_beds_occupied_covid_confirmed; avg_percent_staff_icu_beds_covid; abs_chg_avg_percent_staff_icu_beds_covid | - | total_adm_all_covid_confirmed_past_7days; total_adm_all_covid_confirmed_past_7days_per_100k |
| National-Digital/au-ed-patients-seen-on-time | peer_group_average | - | - |
| National-Digital/au-elective-surgery-waiting-times-by-specialty | median_wait_days; peer_group_median_days; share_waited_over_365_days; peer_group_average | - | - |
| Ohja-ai/rki-grippe-web-wochenbericht-2026-02-06 | Inzidenz | - | - |
| rricc22/endomondo-hr-prediction-v2 | speed; altitude; timestamp | - | - |
| ruiqil/SpatialEpiBench [AUcase, CAcase, CANpositivity, CHNGinpatient, CHNGoutpatient, CPRadmissions, DVcli, HHShosp] | AL; AR; AZ; CA; CO; CT; DC; DE; FL; GA; HI; IA; IL; IN; KS; KY; LA; MA; MD; ME; MI; MN; MO; MS; MT; NC; ND; NE; NH; NJ; NM; NV; NY; OH; OK; OR; PA; RI; SC; SD; ... (+9 more) | - | - |
| DragonbornRay/FluDataset | num_ili; num_patients; hosp_flu; rate_flu; hosp_reporting; beds_flu_pct; ed_pct_flu; ed_pct_covid; lab_pct_pos; temp_mean; abs_humidity; pm25_mean; pm25_max; mmwr_year; mmwr_week | - | - |
| EPI-Eval/covid-tracking-project | death; hospitalizedCurrently; inIcuCurrently; onVentilatorCurrently; totalTestResults; positiveIncrease; deathIncrease; hospitalizedIncrease; totalTestResultsIncrease | - | - |
| EPI-Eval/nyt-covid | deaths | - | - |
| ruiqil/SpatialEpiBench [ILI2019] | AL; AR; AZ; CA; CO; CT; DC; DE; FL; GA; HI; IA; IL; IN; KS; KY; LA; MA; MD; ME; MI; MN; MO; MS; MT; NC; ND; NE; NH; NJ; NM; NV; NY; NYC; OH; OK; OR; PA; RI; SC; ... (+10 more) | - | - |
| ruiqil/SpatialEpiBench [JHUcase] | AL; AR; AZ; CA; CO; CT; DC; DE; FL; GA; HI; IA; IL; IN; KS; KY; LA; MA; MD; ME; MI; MN; MO; MS; MT; NC; ND; NE; NH; NJ; NM; NV; NY; OH; OK; OR; PA; RI; SC; SD; ... (+9 more) | - | - |
| ruiqil/SpatialEpiBench [NCHSdeaths] | AL; AR; AZ; CA; CO; CT; DC; DE; FL; GA; HI; IA; IL; IN; KS; KY; LA; MA; MD; ME; MI; MN; MO; MS; MT; NC; ND; NE; NH; NJ; NM; NV; NY; OH; OK; OR; PA; RI; SC; SD; ... (+9 more) | - | - |
| CashlessConsumer/chennai-rain-gauges | rainfall_mm; rainfall_prev_mm; rainfall_24h_mm; progressive_mm; rainfall; previous_rainfall; rainfall24; progressive; progressflag | - | - |
| ceepr-bit/dianchi-water | PH; DO; CON; NTU; IMN; NH_N; TP; TN | - | - |
| chrimerss/RSI-benchmark-hydro | split_day | - | - |
| cjc0013/ouroboros-ooi-ctd-benchmark | input_window_hours; forecast_horizon_hours; input_non_null_fraction; input_min_native_observation_count; target_native_observation_count; persistence_prediction; climatology_prediction; seasonal_naive_24h_prediction | forecast_horizon_hours | - |
| copernicus-argentina/lagunas-pampeanas-2018-2026 | ndci; ndti; observable_pct | - | - |
| copernicus-argentina/parana-alturas-historicas-1900-2026 | altura_min_m; altura_max_m; n_lecturas | - | - |
| earino/noaa-tide-flooding | margin_ft; margin_ratio; trailing7_mean; trailing30_mean; trailing30_std; exceed_last7; days_since_exceedance; day_of_year; threshold_rank; observed_max_ft; minor_flood | - | observed_max_ft |
| National-Digital/vic-melbourne-water-daily-storage | volume_percent_full | - | - |
| phanerozoic/ww3-ndbc-pairs | m_wspd; m_wdir; p1_hs; p1_tp; p1_dir; p2_hs; p2_tp; p2_dir; p3_hs; p3_tp; p3_dir; o_hs; o_dpd; o_mwd | - | - |
| taziksh/argo-profiles | pres_qc; pres_adjusted; pres_adjusted_qc; pres_adjusted_error; temp; temp_qc; temp_adjusted; temp_adjusted_qc; temp_adjusted_error; psal; psal_qc; psal_adjusted; psal_adjusted_qc; psal_adjusted_error; station_parameters; calibration | - | - |
| xairon/piezo-embedding-benchmark | temperature_2m; total_precipitation; potential_evaporation | - | - |
| zer0contextlost/waterwatch-runoff | rank; n_years; percentile; class; n_sites; max_weight | - | - |
| Rekin226/aquascope-gauges [observation bundles (USGS, UK EA, BOM, Hub'Eau, PEGELONLINE, ...)] | next_sink; main_bas; DIST_SINK; DIST_MAIN; sub_area; up_area; endo; coast; order; SORT; dis_m3_pyr; dis_m3_pmn; dis_m3_pmx; run_mm_syr; inu_pc_smn; inu_pc_umn; inu_pc_smx; inu_pc_umx; inu_pc_slt; inu_pc_ult; lka_pc_sse; lka_pc_use; lkv_mc_usu; rev_mc_usu; dor_pc_pva; ria_ha_ssu; ria_ha_usu; riv_tc_ssu; riv_tc_usu; gwt_cm_sav; ele_mt_sav; ele_mt_uav; ele_mt_smn; ele_mt_smx; slp_dg_sav; slp_dg_uav; sgr_dk_sav; clz_cl_smj; cls_cl_smj; tmp_dc_syr; ... (+252 more) | - | - |
| Amgharr/FEMTO-ST_DATASET | ay; mag; ax_diff; ay_diff; mag_diff | - | - |
| Arailym-tleubayeva/NK-Oil-Well-Sensor-Monitoring-v2 | n_obs | - | - |
| Arailym-tleubayeva/NK-Oil-Well-Sensor-Monitoring-v2 [train/test] | n_obs | - | - |
| jwsouza13/eVED | Trip; Vehicle Speed[km/h]; MAF[g/sec]; Engine RPM[RPM]; Absolute Load[%]; OAT[DegC]; Fuel Rate[L/hr]; Air Conditioning Power[kW]; Air Conditioning Power[Watts]; Heater Power[Watts]; HV Battery Current[A]; HV Battery SOC[%]; HV Battery Voltage[V]; Short Term Fuel Trim Bank 1[%]; Short Term Fuel Trim Bank 2[%]; Gradient; Energy_Consumption; Match Type; Class of Speed Limit; Speed Limit with Direction[km/h]; Intersection; Bus Stops; Generalized_Weight | - | - |
| krohnedigital/FuelCast | Consumer_Boiler2_MomentaryFuel; Consumer_GeneratorEngine1_MomentaryFuel; Consumer_GeneratorEngine1_RotationSpeed; Consumer_GeneratorEngine1_ShaftPower; Consumer_GeneratorEngine2_MomentaryFuel; Consumer_GeneratorEngine2_RotationSpeed; Consumer_GeneratorEngine2_ShaftPower; Consumer_GeneratorEngine3_MomentaryFuel; Consumer_GeneratorEngine3_RotationSpeed; Consumer_GeneratorEngine3_ShaftPower; Consumer_GeneratorEngine4_MomentaryFuel; Consumer_GeneratorEngine4_RotationSpeed; Consumer_GeneratorEngine4_ShaftPower; Consumer_GeneratorEngine5_MomentaryFuel; Consumer_GeneratorEngine5_RotationSpeed; Consumer_GeneratorEngine5_ShaftPower; Consumer_Total_MomentaryFuel; Consumer_Total_ShaftPower; Environment_SeaFloorDepth; Propeller_Port_RotationSpeed; Propeller_Port_ShaftPower; Propeller_Port_ShaftTorque; Propeller_Starboard_RotationSpeed; Propeller_Starboard_ShaftPower; Propeller_Starboard_ShaftTorque; Propeller_Total_ShaftPower; Ship_Bearing; Ship_Heading; Ship_SpeedOverGround; Ship_SpeedThroughWater; Weather_DiffuseRadiation; Weather_DirectNormalIrradiance; Weather_DirectRadiation; Weather_OceanCurrentDirection; Weather_OceanCurrentVelocity; Weather_Precipitation; Weather_RelativeHumidity2M; Weather_ShortwaveRadiation; Weather_SunshineDuration; Weather_SurfacePressure; ... (+11 more) | - | - |
| mihnathul/celljar | voltage_V; cycle_number; temperature_C; coulomb_count_Ah; energy_Wh; step_number; displacement_um | - | - |
| nju-zhangsq/RMISC [BatteryRUL] | Max. Voltage Dischar. (V); Charging time (s); Min. Voltage Charg. (V); Decrement 3.6-3.4V (s); Total time (s); Time at 4.15V (s); Discharge Time (s); Time constant current (s) | - | - |
| nju-zhangsq/RMISC [FrothFlotation] | Flotation Column 04 Level; Ore Pulp Density; Flotation Column 05 Air Flow; Flotation Column 04 Air Flow; Starch Flow; Flotation Column 06 Air Flow; Ore Pulp pH; % Iron Feed; Ore Pulp Flow; Flotation Column 01 Level; Amina Flow; Flotation Column 01 Air Flow | - | - |
| nju-zhangsq/RMISC [GasPipeline] | gain; solenoid; system mode; rate; control scheme; setpoint; pump | - | - |
| nju-zhangsq/RMISC [GasSensorDynamic] | Setpoint1; Setpoint2 | - | - |
| nju-zhangsq/RMISC [GasSensorTemperature] | Temperature (C); Humidity (%r.h.); Flow rate (mL/min); CO (ppm); Heater voltage (V) | - | - |
| nju-zhangsq/RMISC [Metropt3] | COMP; DV_eletric; Towers; MPG; LPS; Pressure_switch; Oil_level; Caudal_impulses | - | - |
| nju-zhangsq/RMISC [MotorTemperature] | torque; motor_speed; ambient; i_d; i_q; u_d; u_q | - | - |
| nju-zhangsq/RMISC [OilWell] | T-TPT; P-MON-CKP; P-JUS-CKGL; P-PDG; T-JUS-CKP; QGL | - | - |
| nju-zhangsq/RMISC [PUMP] | sensor1; sensor2; sensor3; sensor4; sensor5; sensor6; sensor7; sensor8; sensor9; sensor10; sensor11; sensor12; sensor13; sensor14; sensor15; sensor16; sensor17; sensor18; sensor19; sensor20; sensor21; sensor22; sensor23; sensor24; sensor25; sensor26; sensor27; sensor28; sensor29; sensor30; sensor31; sensor32; sensor33; sensor34; sensor35; sensor36; sensor37; sensor38; sensor39; sensor40; ... (+3 more) | - | - |
| nju-zhangsq/RMISC [SWAT] | MV101; P101; P102; MV201; P201; P203; P204; P205; P206; MV301; MV302; MV303; MV304; P301; P302; P402; P403; UV401; P501; P602 | - | - |
| nju-zhangsq/RMISC [WADI] | 1_MV_001_STATUS; 1_MV_004_STATUS; 1_P_001_STATUS; 1_P_003_STATUS; 1_P_005_STATUS; 2_LS_101_AH; 2_LS_101_AL; 2_LS_201_AH; 2_LS_201_AL; 2_LS_301_AH; 2_LS_301_AL; 2_LS_401_AH; 2_LS_401_AL; 2_LS_501_AH; 2_LS_501_AL; 2_LS_601_AH; 2_LS_601_AL; 2_MV_003_STATUS; 2_MV_006_STATUS; 2_MV_101_STATUS; 2_MV_201_STATUS; 2_MV_301_STATUS; 2_MV_401_STATUS; 2_MV_501_STATUS; 2_MV_601_STATUS; 2_P_003_STATUS | - | - |
| Forgis/FactoryNet [factorywave_*, voraus_*, aursad_*, cnc_*] | ctx_robot_voltage; ctx_robot_current; ctx_speed_scaling; ctx_target_speed_fraction; ctx_momentum; ctx_robot_mode; ctx_safety_mode; ctx_runtime_state; feedback_torque_tool; effort_torque_tool; setpoint_torque_tool; setpoint_torque_gradient_tool; ctx_state_move_to_pin; ctx_state_move_to_home; ctx_state_loosening; ctx_state_tightening; ctx_state_screwdriver_busy; ctx_state_process_nok; ctx_state_process_ok; auxiliary_accel_tool_0; auxiliary_accel_tool_1; auxiliary_accel_tool_2; setpoint_pos_0; setpoint_vel_0; setpoint_acc_0; setpoint_current_0; setpoint_torque_0; feedback_pos_0; feedback_vel_0; effort_current_0; effort_voltage_0; ctx_temp_0; ctx_joint_mode_0; setpoint_pos_cartesian_0; setpoint_vel_cartesian_0; feedback_pos_cartesian_0; feedback_vel_cartesian_0; effort_force_cartesian_0; setpoint_pos_1; setpoint_vel_1; ... (+78 more) | - | - |
| Adrien97/velib-monitoring-historical | numdocksavailable; mechanical; ebike; capacity; temp_C; precip_mm; wind_mps | - | - |
| aiacademy-kg/bishkek-transport | route; angle | - | - |
| Algo0o/flight_delay [default] | ORIGIN_WAC; DEST_WAC; DEP_DELAY; DEP_DELAY_NEW; DEP_DEL15; DEP_DELAY_GROUP; ARR_DELAY; ARR_DEL15; ARR_DELAY_GROUP; CANCELLED; DIVERTED; FLIGHTS; DISTANCE; DISTANCE_GROUP; CARRIER_DELAY; WEATHER_DELAY; NAS_DELAY; SECURITY_DELAY; LATE_AIRCRAFT_DELAY; DEP_DELAY_OVER15; ARR_DELAY_OVER15; TOTAL_DELAY; ANY_DELAY | - | - |
| AmandaHattaway/bike-lane-lakehouse | persons_injured; persons_killed; pedestrians_injured; pedestrians_killed; cyclists_injured; cyclists_killed; cyclists_hurt; motorists_injured; motorists_killed | - | - |
| angelosaadeh/velib | docks; mechanical; ebike; last_reported; is_installed; is_renting; is_returning | is_installed; is_renting; is_returning | - |
| Arimancy/bts-flights-weather | arr_delay_min; taxi_out_min; taxi_in_min; crs_elapsed_min; actual_elapsed_min; air_time_min; distance_miles; reported_carrier_delay_min; reported_weather_delay_min; reported_nas_delay_min; reported_security_delay_min; reported_late_aircraft_delay_min; day_of_week; day_of_year; days_to_nearest_holiday; origin_sched_departures_this_hour; dest_sched_arrivals_this_hour; origin_sched_departures_this_day; prior_leg_arr_delay_min; prior_leg_turnaround_min; wx_origin_temp_c; wx_origin_dewpoint_c; wx_origin_relative_humidity_pct; wx_origin_wind_speed_ms; wx_origin_wind_gust_ms; wx_origin_wind_dir_deg; wx_origin_visibility_m; wx_origin_ceiling_m; wx_origin_sea_level_pressure_hpa; wx_origin_precip_1h_mm; wx_origin_snow_depth_cm; wx_origin_sky_cover_oktas; wx_origin_flight_category_ordinal; wx_origin_speci_count_1h; wx_origin_m1_temp_c; wx_origin_m1_wind_speed_ms; wx_origin_m1_wind_gust_ms; wx_origin_m1_visibility_m; wx_origin_m1_ceiling_m; wx_origin_m1_precip_1h_mm; ... (+138 more) | days_to_nearest_holiday; is_us_federal_holiday; is_holiday_travel_window | actual_elapsed_min |
| autogluon/chronos_datasets [monash_rideshare] | price_mean; price_max; distance_min; distance_mean; distance_max; surge_min; surge_mean; surge_max; api_calls; temp; rain; humidity; clouds; wind | - | - |
| danipaez/velib_tft [default] | minutes_of_day; day_of_week; is_public_holiday; capacity; temp; wind; snow; rain | is_public_holiday | - |
| digitalhen/us-airport-wait-times | is_open; wait_is_estimated | is_open | - |
| dmariaa70/GO-MO | sensor_occupancy; avg_speed | - | - |
| Donny-Guo/airline-delay | dep_delay; taxi_out; wheels_off; wheels_on; taxi_in; arr_delay; distance; carrier_delay; weather_delay; nas_delay; security_delay; late_aircraft_delay; cancelled; diverted | - | - |
| EPI-Eval/global-mobility | grocery_and_pharmacy; parks; transit_stations; workplaces; residential | - | - |
| femartip/valencia-urban-mobility | gid; angle_deg; intensity | - | - |
| Jakov/bajs-zagreb [station_status] | station_uid; station_number; bikes; rentable_bikes; booked_bikes; bike_racks; free_racks; place_type; maintenance; active | - | - |
| jhumbl/dft-vehicle-licensing | count_thousands | - | - |
| jhumbl/orr-station-usage | interchanges; series_break | - | - |
| jhumbl/tfl-network-demand | exit_tap_count | - | - |
| Karry001/movemelbourne-gtfsr | stu_index; stop_sequence; arrival_delay; departure_delay | - | - |
| kpubdata/seoul-bike-rent-month | station_no; use_count; exercise_kcal; carbon_reduction_kg; distance_m; duration_min | - | - |
| Malavnaik/dublin-bikes-weather-raw | num_vehicles_disabled; num_docks_available; num_docks_disabled; is_installed; is_renting; is_returning | is_installed; is_renting; is_returning | - |
| matrosovdani/moscow-parking-occupancy | free_handicapped_spaces; occupancy_rate | - | - |
| MLCapstoneRideshare/rideshare-trips | base_passenger_fare; tolls; bcf; sales_tax; congestion_surcharge; airport_fee; cbd_congestion_fee; tips; driver_pay | - | - |
| National-Digital/nsw-traffic-volume-yearly | cardinal_direction_seq; classification_seq; traffic_count; data_duration; data_availability; data_reliability; data_quality_indicator; partial_year | - | - |
| nju-zhangsq/RMISC [BTS] | Dew_Point_Setpoint_7159; Speed_Setpoint_2825; Operating_Mode_Status_7854; Humidification_Start_Stop_Status_3706; Differential_Pressure_Setpoint_483; Room_Air_Temperature_Setpoint_4602; Room_Air_Temperature_Setpoint_594; Outside_Air_Lockout_Temperature_Setpoint_2047; Air_Flow_Setpoint_2209; Status_797; Operating_Mode_Status_1425; Fan_Status_2726; Cooling_Temperature_Setpoint_6201; Fan_Status_3524; Air_Flow_Setpoint_2564; Fan_Status_6906; Proportional_Gain_Parameter_3619; Heating_Temperature_Setpoint_149; Speed_Setpoint_5642; Fault_Status_224; Speed_Setpoint_5404; Operating_Mode_Status_6798; Max_Temperature_Setpoint_Limit_354; Mode_Status_6856; Speed_Setpoint_2257; Speed_Setpoint_326; Fan_Status_7479; Mode_Status_7042; Operating_Mode_Status_1153; System_Status_347; Cooling_Temperature_Setpoint_291; Enable_Status_4538; System_Status_3152; Room_Air_Temperature_Setpoint_4014; Mode_Status_7244; Speed_Setpoint_3629; Enable_Status_4892; Status_4752; Water_Temperature_Setpoint_6141; Room_Air_Temperature_Setpoint_1230; ... (+2256 more) | - | - |
| nju-zhangsq/RMISC [ChengduTaxi] | time_gap | - | - |
| nju-zhangsq/RMISC [LargeST] | s2050; s2051; s2052; s2053; s2054; s2055; s2056; s2057; s2058; s2059; s2060; s2061; s2062; s2063; s2064; s2065; s2066; s2067; s2068; s2069; s2070; s2071; s2072; s2073; s2074; s2075; s2076; s2077; s2078; s2079; s2080; s2081; s2082; s2083; s2084; s2085; s2086; s2087; s2088; s2089; ... (+599 more) | - | - |
| nju-zhangsq/RMISC [MetroTraffic] | clouds_all; rain_1h; temp; snow_1h | - | - |
| nju-zhangsq/RMISC [PEMSCalifornia] | f2; f3; f4; f5; f6; f7; f8; f9; f10; f11; f12; f13; f14; f15; f16; f17; f18; f19; f20; f21; f22; f23; f24; f25; f26; f27; f28; f29; f30; f31; f32; f33; f34; f35; f36; f37; f38; f39; f40; f41; ... (+12630 more) | - | - |
| olaflaitinen/baku-metro-station-flow | day_of_week; day_of_year; is_weekend; lag_1; lag_7; lag_14; lag_28; rolling_mean_7; rolling_std_7; rolling_mean_14; rolling_std_14; ewma_7; ewma_14; day_of_week_sin; day_of_week_cos; month_sin; month_cos; target_next_day_entries; is_transfer_station | is_weekend; is_transfer_station | lag_1; lag_7; lag_14; lag_28 |
| orailix/ride-silver | rain; snowfall; relative_humidity_2m; wind_speed_10m | - | - |
| piebro/deutsche-bahn-data | train_line_station_num; arrival_is_canceled; departure_is_canceled; is_additional_stop; is_replacement_train; has_plan_record | is_additional_stop; is_replacement_train | - |
| Salesforce/lotsa_data [BEIJING_SUBWAY_30MIN] | past_feat_dynamic_real | - | past_feat_dynamic_real |
| somemone/ac-transit-apc | alightings; passenger_load | - | - |
| swadhinbiswas/air-traffic | heading; true_heading; mag_heading; roll; vertical_rate; geom_rate; mach; ias; tas; oat; tat; wind_dir; wind_speed; nav_qnh; nav_heading; db_flags; nic; nac_p; nac_v; sil; rc; fuel_burn_kg_per_hour; co2_kg_per_hour; on_ground; co2_estimated; is_cargo; is_military | is_cargo; is_military | - |
| witgaw/METR-LA | x_t-11_d1; x_t-10_d0; x_t-10_d1; x_t-9_d0; x_t-9_d1; x_t-8_d0; x_t-8_d1; x_t-7_d0; x_t-7_d1; x_t-6_d0; x_t-6_d1; x_t-5_d0; x_t-5_d1; x_t-4_d0; x_t-4_d1; x_t-3_d0; x_t-3_d1; x_t-2_d0; x_t-2_d1; x_t-1_d0; x_t-1_d1; x_t+0_d0; x_t+0_d1; y_t+1_d0; y_t+1_d1; y_t+2_d0; y_t+2_d1; y_t+3_d0; y_t+3_d1; y_t+4_d0; y_t+4_d1; y_t+5_d0; y_t+5_d1; y_t+6_d0; y_t+6_d1; y_t+7_d0; y_t+7_d1; y_t+8_d0; y_t+8_d1; y_t+9_d0; ... (+7 more) | - | - |
| witgaw/PEMS-BAY | x_t-11_d1; x_t-10_d0; x_t-10_d1; x_t-9_d0; x_t-9_d1; x_t-8_d0; x_t-8_d1; x_t-7_d0; x_t-7_d1; x_t-6_d0; x_t-6_d1; x_t-5_d0; x_t-5_d1; x_t-4_d0; x_t-4_d1; x_t-3_d0; x_t-3_d1; x_t-2_d0; x_t-2_d1; x_t-1_d0; x_t-1_d1; x_t+0_d0; x_t+0_d1; y_t+1_d0; y_t+1_d1; y_t+2_d0; y_t+2_d1; y_t+3_d0; y_t+3_d1; y_t+4_d0; y_t+4_d1; y_t+5_d0; y_t+5_d1; y_t+6_d0; y_t+6_d1; y_t+7_d0; y_t+7_d1; y_t+8_d0; y_t+8_d1; y_t+9_d0; ... (+7 more) | - | - |
| yadavsaurabh/indian-railways-runs | halts_here; delay_arrival_min; delay_departure_min; distance_km; departed; reported | - | - |
| youngbongbong/30_od_subway [default] | od_in_cnt; od_net_cnt | - | - |
| youngbongbong/mobility_dataset | in_cnt; net_cnt | - | - |
| yuiseki/meta-range-maps-2022-05 | all_day_ratio_single_tile_users | - | - |
| ZanderL1337/shipping-data-pipeline | status; length; width; draft; cargo; duration_minutes | - | - |
| nju-zhangsq/RMISC [TimeMMD] | Exports; Imports; D0; D1; D2; D3; D4; Weekly East Coast All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon); Weekly New England (PADD 1A) All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon); Weekly Central Atlantic (PADD 1B) All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon); Weekly Lower Atlantic (PADD 1C) All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon); Weekly Midwest All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon); Weekly Gulf Coast All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon); Weekly Rocky Mountain All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon); Weekly West Coast All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon); Number of Sites Reporting; % WEIGHTED ILI; AGE 0-4; AGE 5-24; AGE 65; ILITOTAL; NUM. OF PROVIDERS; TOTAL PATIENTS | - | - |
| prabakaranchandran/enterprise-dynamics-corpus-v1 | past_feat_dynamic_real | - | past_feat_dynamic_real |
| alanwgt/fipex-veiculos-brasil | zero_km | - | - |
| Dingdong-Inc/FreshRetailNet-50K [default] | hours_stock_status | - | - |
| Dingdong-Inc/FreshRetailNet-LT [unknown] | hours_stock_status | - | - |
| dubairealestatedata/dubai-real-estate-sales-transactions | area_sqm; price_aed; price_psf; quality_flags; has_parking; is_pre_registration | is_pre_registration | - |
| earino/my-pricecatcher-weekly-increase | n_obs; price_lag1; price_lag2; price_lag4; delta_1w; delta_4w; mean4; std4; min4; max4; price_vs_mean4; week_of_year; increases | - | price_lag1; price_lag2; price_lag4 |
| fuel-prices-eu/european-fuel-prices | mean_eur; median_eur; p10_eur; p90_eur; mean_local; median_local; p10_local; p90_local; eur_rate | - | - |
| idomanteu/cs2-historical-item-prices-hourly-march-april-2026 | high_ask; low_ask; close_ask; ask_volume; open_bid; high_bid; low_bid; close_bid; bid_volume; sample_count | - | - |
| kpubdata/seoul-apartment-rent | deal_year; deal_month; deal_day; deposit; monthly_rent; exclusive_area; floor; previous_deposit; previous_monthly_rent | - | - |
| kpubdata/seoul-apartment-trades | floor; build_year; deal_year; deal_month; deal_day; deal_amount_10k_krw | - | - |
| National-Digital/nt-fuel-prices-history | premium_98; premium_95; unleaded_91; premium_diesel; lpg; low_aromatic_fuel; e85; e10; b20 | - | - |
| nju-zhangsq/RMISC [Dominick] | T939; T1692; T1799; T1880; T3114; T3282; T6789; T7258; T8132; T9168; T9330; T10303; T11752; T11859; T11933; T12245; T12895; T12969; T13450; T14270; T14440; T14734; T14842; T14915; T15235; T15674; T15783; T16275; T16836; T17466; T19923; T20584; T20585; T20666; T21574; T21777; T21780; T22134; T22258; T22259; ... (+1267 more) | - | - |
| nju-zhangsq/RMISC [WeeklyFuelPricesItaly] | VAT; EXCISE; NET | - | - |
| pracadata/brazil-fuel-prices | price_brl_mean; price_brl_median; price_brl_min; price_brl_max; price_brl_p25; price_brl_p75; in_continuous_series | - | - |
| pracadata/mexico-fuel-prices | price_mxn_median; price_mxn_p25; price_mxn_p75; price_mxn_min; price_mxn_max; price_usd_median; price_usd_p25; price_usd_p75; price_usd_min; price_usd_max | - | - |
| Rif-SQL/time-series-uk-retail-supermarket-price-data | price_unit_gbp; is_own_brand | is_own_brand | - |
| thehooklab/nsw-property | purchase_price; area; sale_year; district; data_quality_score; num_bath; num_bed; num_parking; land_area; km_from_cbd; suburb_population; suburb_median_income; cash_rate; property_inflation_index; valid_postcode | - | - |
| VincentGOURBIN/FuelInFranceData | zipcode; coordlatitude; coordlongitude; price; brent_rate; brent_rate_eur | - | - |
| AbdullahImran/Saudi-Supply-Demand-Datasets | sell_price; snap_CA; snap_TX; snap_WI | - | - |
| APProjects/us-layoffs-monthly-time-series-warn-act | notices_with_headcount; workers_affected | - | - |
| arh1p/instagram-top-accounts | rank | - | - |
| Gallicagram/atlas-motifs | pic_unique | - | - |
| gymprathap/Chicago-Crime-Dataset | District; Ward; Community Area; X Coordinate; Y Coordinate; Arrest; Domestic | - | - |
| hsilvosa/gdelt-events | is_root_event; quad_class; goldstein_scale; num_mentions; num_sources; num_articles; avg_tone; actor1_geo_type; actor2_geo_type; action_geo_type | is_root_event | - |
| lingbow/tiktok-video-engagement-200k | play_count; like_count; comment_count; share_count; collect_count; download_count; whatsapp_share_count | - | - |
| matanzig/Disney-Theme-Park-Queue-Dynamics | WAIT_TIME_MAX; NB_UNITS; GUEST_CARRIED; CAPACITY; ADJUST_CAPACITY; DOWNTIME; NB_MAX_UNIT; TOTAL_DAILY_ATTENDANCE; temp; humidity; wind_speed; temp_rounded; IS_BROKEN; IS_BLACK_SWAN | IS_BROKEN; IS_BLACK_SWAN | - |
| yuiseki/ucdp-ged [26.1] | type_of_violence; number_of_sources; where_prec; priogrid_gid; event_clarity; deaths_a; deaths_b; deaths_civilians; deaths_unknown; best; high; low | - | - |
| yuiseki/un-wpp [WPP2024] | AgeGrpStart; AgeGrpSpan; PopMale; PopFemale; PopTotal | - | - |
| yuiseki/undp-hdr | hdi_1990; hdi_1991; hdi_1992; hdi_1993; hdi_1994; hdi_1995; hdi_1996; hdi_1997; hdi_1998; hdi_1999; hdi_2000; hdi_2001; hdi_2002; hdi_2003; hdi_2004; hdi_2005; hdi_2006; hdi_2007; hdi_2008; hdi_2009; hdi_2010; hdi_2011; hdi_2012; hdi_2013; hdi_2014; hdi_2015; hdi_2016; hdi_2017; hdi_2018; hdi_2019; hdi_2020; hdi_2021; hdi_2022; hdi_2023; le_1990; le_1991; le_1992; le_1993; le_1994; le_1995; ... (+1067 more) | - | - |
| earino/abs-visitor-arrivals-direction | target_month_num; val_m; val_m1; val_m2; val_m3; val_m6; val_m12; diff_mom; diff_quarter; diff_half; diff_yoy_m; fell_short | - | - |
| giggseh/kepler-lc-star-params | time | - | - |
| juliensimon/auroral-electrojet-index | is_active | is_active | - |
| juliensimon/dst-index | is_storm | is_storm | - |
| juliensimon/neutron-monitor-cosmic-rays | daily_mean_count_rate; pct_deviation | - | - |
| juliensimon/solar-wind | active | - | - |
| lxr11111/telemanom [SMAP,MSL] | timestep; cmd_0; cmd_1; cmd_2; cmd_3; cmd_4; cmd_5; cmd_6; cmd_7; cmd_8; cmd_9; cmd_10; cmd_11; cmd_12; cmd_13; cmd_14; cmd_15; cmd_16; cmd_17; cmd_18; cmd_19; cmd_20; cmd_21; cmd_22; cmd_23; cmd_24; cmd_25; cmd_26; cmd_27; cmd_28; cmd_29; cmd_30; cmd_31; cmd_32; cmd_33; cmd_34; cmd_35; cmd_36; cmd_37; cmd_38; ... (+15 more) | - | - |
| MultimodalUniverse/tess | lightcurve.flux; lightcurve.flux_err | - | - |
| nju-zhangsq/RMISC [Satellite] | f6; f12; f13; f20; f21; f28; f30; f34; f44; f48; f2; f3; f4; f7; f8; f9; f10; f14; f15; f18; f19; f22; f23; f40; f42; f36; f50; f55; f46; f16; f17; f49; f29; f47; f54; f32; f5; f11; f24; f25 | - | - |
| nyu-planet-finders/kepler-llc | flux; flux_err; centr_x; centr_y | - | - |
| nyu-planet-finders/tess-spoc-2min-binned | flux; flux_err; centr_x; centr_y | - | - |
| phanerozoic/qiskit-calibration-drift | qubit_a; qubit_b; calibration_age_seconds; solar_zenith_deg; temperature_c; pressure_hpa; humidity_pct; bz_gsm_nt; neutron_flux; kp_index; ap_index; Ap_daily; SN; f107_observed_sfu; f107_adjusted_sfu; solar_flux_sfu; dst_nt; is_failure_ceiling; is_new_measurement | is_failure_ceiling; is_new_measurement | f107_observed_sfu |
| autogluon/chronos_datasets [ushcn_daily] | SNOW; SNWD; TMAX; TMIN | - | - |
| DuckerMaster/thaiclim-district | precip_3d_mm; precip_5d_mm; precip_7d_mm; temp_mean_c; temp_max_c; temp_min_c; temp_dtr_c; dry_spell_days; wet_spell_days; is_extreme_rain_r95p; precip_ge_50mm; precip_ge_70mm; precip_ge_100mm | is_extreme_rain_r95p | - |
| Emulated-Inc/sci-airport-low-ceiling-visibility | dewpoint_f; wind_dir_deg; wind_kt; gust_kt; altimeter_inhg; precip_1h_in; vis_mi; sky_base_ft_1; sky_base_ft_2; sky_base_ft_3; sky_base_ft_4 | - | - |
| hblim/weather_data | Dew Point (°F); Relative Humidity (%); Heat Index (°F); Wind Chill (°F); Wind Speed (mph); Visibility (miles); Clouds (x100 ft); Sea Level Pressure (mb); Station Pressure (in Hg); Altimeter Setting (in Hg); 1 Hour Precip (in); 3 Hour Precip (in); 6 Hour Precip (in); 24 Hour Precip (in); 6 Hr Max (°F); 6 Hr Min (°F); 24 Hr Max (°F); 24 Hr Min (°F) | - | - |
| jotone/arpa-osmer-weatherstations | prec_type; t180; v10; ff10; dd10; dd10max; vmax; rg; cloudiness; rh; press; bf; t50; tm10 | - | - |
| juliangehring/noaa-isd | dew_point_c; wind_dir_deg; wind_speed_ms; wind_gust_ms; ceiling_m; visibility_m; sea_level_pressure_hpa; altimeter_hpa; station_pressure_hpa | - | - |
| nakas/powderbench-truth | valid | - | - |
| nju-zhangsq/RMISC [ERA5HourlySingleLevels] | u100; v100; u10; v10; d2m; msl; skt; sp; ssrd; strd; latitude; longitude | - | - |
| nju-zhangsq/RMISC [GlobalClimateChange] | AverageTemperatureUncertainty; LandMaxTemperature; LandMaxTemperatureUncertainty; LandMinTemperature; LandMinTemperatureUncertainty; LandAndOceanAverageTemperature; LandAndOceanAverageTemperatureUncertainty; LandAverageTemperature; LandAverageTemperatureUncertainty | - | - |
| nju-zhangsq/RMISC [WeatherBench5-625deg] | 1_v10; 2_t2m; 3_z; 5_r; 6_q; 7_t; 8_tisr; 9_tcc; 10_tp; 11_u; 12_v; 13_vo; 4_pv; 4_z; 5_z; 6_z; 7_z; 8_z; 9_z; 10_z; 11_z; 12_z; 13_z; 14_z; 15_z; 29_r; 30_r; 31_r; 32_r; 33_r; 34_r; 35_r; 36_r; 37_r; 38_r; 39_r; 40_r; 41_r; 45_q; 46_q; ... (+67 more) | - | - |
| Pahzuzu/PAH-ENV-001-serc-soil-signals | vswc_uncertainty; vswc_flag; vsic; vsic_uncertainty; vsic_flag; depth_cm | - | - |
| taziksh/cliwoc | YR; MO; DY; HR; IM; ATTC; TI; LI; VS; NID; II; DI; D; WI; W; VI; VV; WW; W1; SLP; A; PPP; IT; AT; WBTI; WBT; DPTI; DPT; SI; SST; N; NH; CL; HI; H; CM; CH; WD; WP; WH; ... (+38 more) | Calendar | - |
| taziksh/ibtracs | season; number; wmo_wind; wmo_pres; dist2land; landfall; usa_wind; usa_pres; usa_sshs; usa_r34_ne; usa_r34_se; usa_r34_sw; usa_r34_nw; usa_r50_ne; usa_r50_se; usa_r50_sw; usa_r50_nw; usa_r64_ne; usa_r64_se; usa_r64_sw; usa_r64_nw; usa_poci; usa_roci; usa_rmw; usa_eye; tokyo_grade; tokyo_wind; tokyo_pres; tokyo_r50_dir; tokyo_r50_short; tokyo_r30_dir; tokyo_r30_short; tokyo_land; cma_cat; cma_wind; cma_pres; hko_wind; hko_pres; kma_wind; kma_pres; ... (+79 more) | season | - |
| VerisimilitudeX/Weather | Data.Temperature.Avg Temp; Data.Temperature.Max Temp; Data.Temperature.Min Temp; Data.Wind.Direction; Data.Wind.Speed | - | - |
| yuiseki/jma-rain-kanto | precip_nophen; precip_quality; precip_homog_in_file; pref | - | - |
| zait-ai/RusWeather-434 | T2M_MAX; T2M_MIN; T2MDEW; RH2M; PRECTOTCORR; PS; WS10M; WS10M_MAX; WS2M; ALLSKY_SFC_SW_DWN; CLOUD_AMT | - | - |
| tharinduperera/weather-clustering-data | temperature_max; temperature_min; precipitation_sum; relative_humidity_mean; wind_speed_mean; surface_pressure_mean | - | - |
| afhubbard/gpu-prices | gpu_memory_gb; vcpus; ram_gb; price_per_hour; availability_zone; is_spot; available | is_spot | - |
| AliMaatouk/TelecomTS [default] | KPIs.DL_BLER; KPIs.DL_MCS; KPIs.UL_BLER; KPIs.UL_MCS; KPIs.UL_NPRB; KPIs.UL_SNR; KPIs.TX_Bytes; KPIs.RX_Bytes; KPIs.Estimated_UL_Buffer; KPIs.PRBs_DL_Current; KPIs.PRBs_UL_Current; KPIs.PRB_Utilization_DL; KPIs.PRB_Utilization_UL; KPIs.UL_NumberOfPackets; KPIs.DL_NumberOfPackets; KPIs.UL_Protocol; KPIs.DL_Protocol; anomalies.affected_kpis; QnA.network; QnA.timeseries; QnA.anomalies | - | - |
| AmirShahbaz/CloudPerfTrace | lin_self; td_self; tr_oth; lin_oth; td_oth | - | - |
| aurman/GoogleTrendArchive [trending_queries] | total_occurrences; search_volume_lower; n_queries; duration_minutes; duration_hours; times_were_swapped; duration_is_estimate | - | - |
| emperor-mew/ooni-censorship-historical | anomaly_count; confirmed_count; failure_count; ok_count; anomaly_rate; confirmed_rate | - | - |
| hq-bench/quito-corpus [hour] | ind_2; ind_3; ind_4; ind_5 | - | - |
| hq-bench/quito-corpus [min] | ind_2; ind_3; ind_4; ind_5 | - | - |
| hq-bench/quitobench [hour, min] | ind_2; ind_3; ind_4; ind_5 | - | - |
| hq-bench/quitobench [hour, min] | ind_2; ind_3; ind_4; ind_5 | - | - |
| kashif/App_Flow [default] | Unnamed: 0 | - | - |
| LeadLagForecasting/llf_github [default] | cumulative_pushes; cumulative_stars; cumulative_forks; platforms | - | - |
| Lightcap/multi-domain-cloudflare-observability | requests; visits; edge_request_bytes; edge_response_bytes; sample_interval_avg; edge_response_status; referer_host; client_asn; client_asn_description; bot_score_bucket_by_10; http_protocol; ssl_protocol; scheme; is_limit_boundary | is_limit_boundary | - |
| Lilambd/security-daily | cvss4; epss; epss_percentile; kev | - | - |
| MachaParfait/ClusterWise | p0_gpu0_power; p0_gpu1_power; p0_gpu2_power; p1_gpu0_power; p1_gpu1_power; p1_gpu2_power; p0_power; p1_power; gpu0_core_temp; gpu0_mem_temp; gpu1_core_temp; gpu1_mem_temp; gpu2_core_temp; gpu2_mem_temp; gpu3_core_temp; gpu3_mem_temp; gpu4_core_temp; gpu4_mem_temp; gpu5_core_temp; gpu5_mem_temp; ps0_input_power; ps1_input_power; xid; p0_temp_mean; p0_temp_min; p0_temp_max; p1_temp_mean; p1_temp_min; p1_temp_max | - | - |
| MarcusLammers/vast-rtx3090-market-6mo | scan_interval_sec; price_usd_hour; reliability; dlperf; dlperf_per_dpth; inet_up; inet_down; cuda_max; gpu_ram; cpu_cores; cpu_ram; disk_space; verified | - | - |
| modelpulse/model-pulse-data | dl_all; likes | - | - |
| nju-zhangsq/RMISC [AliCar] | brand | - | - |
| nju-zhangsq/RMISC [EbayServer] | feature_1; feature_2; feature_3; feature_4; feature_5; feature_6; feature_7; feature_8; feature_9; feature_10; feature_11; feature_12; feature_13; feature_14; feature_15; feature_16; feature_17; feature_18; feature_19; feature_20; feature_21; feature_22; feature_23; feature_24 | - | - |
| nju-zhangsq/RMISC [MiniApp] | f21; f30; f25; f28; f27; f24; f22; f29; f26; f23 | - | - |
| nju-zhangsq/RMISC [ServerMachineDataset] | f2; f3; f4; f6; f7; f9; f10; f11; f12; f13; f14; f15; f16; f19; f20; f21; f22; f23; f24; f25; f26; f28; f29; f30; f31; f32; f33; f34; f35; f36; f17; f18; f5; f37; f27; f38 | - | - |
| pyToshka/attacks-daily | session_duration; is_malicious | is_malicious | - |
| Salesforce/lotsa_data [alibaba_cluster_trace_2018] | past_feat_dynamic_real | - | past_feat_dynamic_real |
| Salesforce/lotsa_data [borg_cluster_data_2011] | past_feat_dynamic_real | - | past_feat_dynamic_real |
| soncco/historical-twitter-trends-spanish-locations-2020-2021 | rank; is_hashtag | is_hashtag | - |
| sovai/wikipedia_views [default] | relative_views; alpha_short; beta_short; search_pressure | - | - |
| tardellirs/paper-pulse-data | comments | - | - |
| venvoo/openrouter-uptime | up5m; up30m; up1d; identity_ambiguous | - | - |
| Salesforce/lotsa_data [azure_vm_traces_2017] | past_feat_dynamic_real | - | past_feat_dynamic_real |

## 8. The three slice choices

(i) 500 series x last 16k points; (ii) 5,000 series x last 64k points; (iii) everything. All estimates. Points = min(S, series) x min(L per value column, points per series). Bytes are proportional to points for parquet scopes (the real fetch rounds up to whole row groups or files); for other formats every slice costs all bytes. Datasets with unknown series or points count only in (iii) bytes.

| domain | (i) points | (i) bytes | (ii) points | (ii) bytes | (iii) points | (iii) bytes |
|---|---|---|---|---|---|---|
| agriculture_nature | 494.96M | 26.94 GB | 1.89B | 26.94 GB | 5.88B | 27.89 GB |
| air_quality | 162.19M | 18.93 GB | 356.19M | 19.00 GB | 2.03B | 20.42 GB |
| economics_finance | 374.90M | 287.21 GB | 2.02B | 293.24 GB | 142.92B | 575.14 GB |
| energy | 11.58B | 310.10 GB | 90.05B | 316.05 GB | 212.53B | 380.44 GB |
| health | 23.98M | 179.27 GB | 145.91M | 179.42 GB | 15.34B | 180.35 GB |
| hydrology_ocean | 55.07M | 1.64 TB | 115.52M | 1.64 TB | 22.14B | 1.65 TB |
| industrial | 115.63M | 169.50 GB | 710.73M | 170.51 GB | 15.37B | 185.83 GB |
| mobility | 6.53B | 417.79 GB | 26.96B | 425.28 GB | 80.37B | 606.20 GB |
| other | 12.52M | 38.24 GB | 232.46M | 38.24 GB | 9.23B | 38.48 GB |
| retail_demand | 8.59M | 16.87 GB | 16.39M | 16.87 GB | 2.63B | 17.60 GB |
| society | 11.05M | 71.32 GB | 43.27M | 71.32 GB | 11.75B | 134.55 GB |
| space_science | 73.74M | 568.70 GB | 866.74M | 569.74 GB | 78.50B | 785.91 GB |
| weather_climate | 655.43M | 759.95 GB | 6.49B | 783.59 GB | 257.05B | 1.26 TB |
| web_cloud | 192.71M | 426.93 GB | 1.90B | 429.21 GB | 215.36B | 468.63 GB |
| ALL | 20.29B | 4.93 TB | 131.80B | 4.98 TB | 1.07T | 6.33 TB |

## 9. Ingest difficulty

| difficulty | datasets |
|---|---|
| easy | 140 |
| hard | 125 |
| medium | 349 |

easy = parquet with one schema; medium = csv / arrow / several tables / nested arrays; hard = archives, netcdf, h5, npy, custom formats, event tables.

### Hard datasets

| dataset | reason |
|---|---|
| alaminxpro/bd-food-prices | unrecognised file formats: crc |
| Emulated-Inc/sci-county-crop-yield-new-year | formats that need unpacking or a custom reader: tar |
| food-ai-nexus/global-food-safety-incidents | event-level table: needs aggregation to a regular step |
| globalsquiddatabase/Tuna-Fisheries-Dataset | formats that need unpacking or a custom reader: 7z, npy, xlsx, zip |
| notadib/usa-corn-belt-crop-yield | formats that need unpacking or a custom reader: npz |
| Project-AgML/usda_nass_crop_statistics | unrecognised file formats: gpkg |
| taziksh/usgs-earthquakes | event-level table: needs aggregation to a regular step |
| yuiseki/faostat [raw/2026-10-04] | formats that need unpacking or a custom reader: zip |
| Aduitya9/Aduitya_dextra-aqi-india | formats that need unpacking or a custom reader: pt |
| haidang2405/aqshift-us-leakage-safe-air-quality | event-level table: needs aggregation to a regular step |
| meganariley/us-air-quality-hourly | no data files found |
| xuxing123/AirQualityBench | formats that need unpacking or a custom reader: h5, pkl |
| yuiseki/edgar-ghg-countries | formats that need unpacking or a custom reader: zip |
| aaron-freedman/banking-concentration-data | formats that need unpacking or a custom reader: xlsx |
| abel-lab/finance1k | unrecognised file formats: sha256 |
| Bayanat-Labs/arab-trade-exposure | unrecognised file formats: sql |
| nju-zhangsq/RMISC [TushareETFDaily] | event-level table: needs aggregation to a regular step |
| nju-zhangsq/RMISC [TushareIndexDaily] | event-level table: needs aggregation to a regular step |
| nju-zhangsq/RMISC [TushareStockDaily] | event-level table: needs aggregation to a regular step |
| nju-zhangsq/RMISC [TushareStockWeekly] | event-level table: needs aggregation to a regular step |
| prospex-ch/swiss-commercial-registry-monthly-panel | event-level table: needs aggregation to a regular step |
| Rogendo/forex-dukas-copy [default] | unrecognised file formats: before_30min_rebuild_20260909t203021z, before_30min_rebuild_20260909t203102z, before_30min_rebuild_20260909t203134z, out |
| sevriukov/bank101 | unrecognised file formats: sql |
| SII-WANGZJ/Polymarket_data [trades.parquet] | event-level table: needs aggregation to a regular step |
| THULab/xauusd [default] | unrecognised file formats: tsfile |
| yuiseki/maddison-project | formats that need unpacking or a custom reader: xlsx |
| yuiseki/pwt | formats that need unpacking or a custom reader: xlsx, zip |
| yuiseki/undata-sdmx | unrecognised file formats: xml |
| kavelaltd/historysaid-global-economic | unrecognised file formats: r, sha256 |
| tradecatlabs/binance-futures-ohlcv-2018-2026 [candles_1m] | formats that need unpacking or a custom reader: dat, gz, zst |
| zalizedata/global-macro-trade-indicators-dataset | formats that need unpacking or a custom reader: zip |
| JrVillabona/resonance-data | event-level table: needs aggregation to a regular step |
| syncster-data/economic-data | formats that need unpacking or a custom reader: tar, zip |
| ai-iot/EnergyBench [RSL] | event-level table: needs aggregation to a regular step |
| alwaysbyx/Bear-Data | formats that need unpacking or a custom reader: zip |
| Battery-Life/BatteryLife_Raw [Zn-ion, Na-ion, CALB] | formats that need unpacking or a custom reader: xlsx |
| bsebench-org/warwick-second-life-dm-2025-raw [raw mirror (746 csv, 309 mat)] | formats that need unpacking or a custom reader: mat, zip |
| Emulated-Inc/battery-capacity-trajectories [Dataset_1_NCA / 2_NCM / 3_NCM_NCA zips] | formats that need unpacking or a custom reader: zip |
| fidel-ts/Bear_room | formats that need unpacking or a custom reader: pkl, zip |
| fidel-ts/California_ISO | formats that need unpacking or a custom reader: pkl |
| fidel-ts/Canada_photovoltaics_plants | formats that need unpacking or a custom reader: pkl, zip |
| gtfierro/mortar [hive-partitioned by uuid] | unrecognised file formats: (none) |
| yzzpz/Charged-data [10 csv per city (some in .rar)] | formats that need unpacking or a custom reader: npy |
| zalizedata/us-grid-hourly-power-plants-dataset | formats that need unpacking or a custom reader: zip |
| autogluon/chronos_datasets_extra [spanish_energy_and_weather] | no data files found |
| fidel-ts/Germany_Renewable_Energy_Grid | formats that need unpacking or a custom reader: pkl, zip |
| OptimalSolution/databank-ember [19 tables (generation monthly/yearly, prices, capacity, ...)] | unrecognised file formats: r |
| THULab/m2oe2_timeseries [8 tables (building_train, building weather, household consumption, residential power usage + weather, solar + weather, Spanish energy + weather)] | unrecognised file formats: tsfile |
| LVinaud/arboviral-alerta-dados | formats that need unpacking or a custom reader: xlsx |
| pedrolucassantanaf/dengue-tcc2-data | formats that need unpacking or a custom reader: zip |
| Uvathe/wellborn-pregnancy-wearable | unrecognised file formats: mtn |
| DragonbornRay/FluDataset | formats that need unpacking or a custom reader: gz, xlsx, zip |
| CashlessConsumer/chennai-rain-gauges | formats that need unpacking or a custom reader: db |
| chrimerss/RSI-benchmark-hydro | formats that need unpacking or a custom reader: tar.gz |
| Emulated-Inc/sci-arctic-sea-ice-forecast-later-years | formats that need unpacking or a custom reader: npy, npz |
| Emulated-Inc/sci-lake-temperature-profiles-unmonitored | formats that need unpacking or a custom reader: tar |
| HenryYHW/BATON | unrecognised file formats: mp4 |
| phanerozoic/noaa-ww3-multi1-points [wmo] | formats that need unpacking or a custom reader: tar.gz |
| Emulated-Inc/sci-streamflow-ungauged-new-country | formats that need unpacking or a custom reader: tar |
| Rekin226/aquascope-gauges [observation bundles (USGS, UK EA, BOM, Hub'Eau, PEGELONLINE, ...)] | formats that need unpacking or a custom reader: gz, hdf |
| Emulated-Inc/sci-vlt-turbulence-from-ao-telemetry | formats that need unpacking or a custom reader: tar |
| HenryYHW/CANTraj | formats that need unpacking or a custom reader: npz, pt |
| jwsouza13/eVED | event-level table: needs aggregation to a regular step |
| Samlzy/MCC5-THU-Motor | formats that need unpacking or a custom reader: zip |
| xijiazhao/casemotordataset [dated folders] | formats that need unpacking or a custom reader: h5 |
| xijiazhao/welding2 [trial=1, trial=2] | formats that need unpacking or a custom reader: npy |
| cognitedata/IND-Eval | formats that need unpacking or a custom reader: mat, xlsx |
| Yahya2491/3w-pipeline | unrecognised file formats: ini |
| Algo0o/flight_delay [default] | formats that need unpacking or a custom reader: db |
| dmariaa70/GO-MO | formats that need unpacking or a custom reader: pkl |
| femartip/valencia-urban-mobility | formats that need unpacking or a custom reader: zip |
| fidel-ts/NYC_traffic_speed | formats that need unpacking or a custom reader: pkl, zip |
| gigigi7/AIS_Dataset [default] | formats that need unpacking or a custom reader: zip |
| jhumbl/dft-vehicle-licensing | unrecognised file formats: ods |
| jhumbl/orr-station-usage | unrecognised file formats: ods |
| jhumbl/tfl-network-demand | formats that need unpacking or a custom reader: zip |
| Karry001/movemelbourne-gtfsr | formats that need unpacking or a custom reader: gz |
| MLCapstoneRideshare/rideshare-trips | event-level table: needs aggregation to a regular step |
| somemone/ac-transit-apc | event-level table: needs aggregation to a regular step |
| vinicius-souza/antaq | formats that need unpacking or a custom reader: pkl |
| witgaw/METR-LA | formats that need unpacking or a custom reader: npy |
| witgaw/PEMS-BAY | formats that need unpacking or a custom reader: npy |
| yind7/EvoXXLTraffic | formats that need unpacking or a custom reader: npz |
| yuiseki/meta-range-maps-2022-05 | formats that need unpacking or a custom reader: zip |
| yuiseki/npa-traffic-accidents | formats that need unpacking or a custom reader: xlsx |
| TorchSpatiotemporal/ProcessedDatasets | formats that need unpacking or a custom reader: h5, zip |
| onedean/UrbanFM-datasets [taxinyc/bikenyc/tdrive/trafficsh/occhamburg/occparis (+pems, metrla)] | formats that need unpacking or a custom reader: pkl |
| tjtrans/Traffic-Demand-Bench | formats that need unpacking or a custom reader: npy, npz, pkl, zip |
| AutonLab/Timeseries-PILE [forecasting] | formats that need unpacking or a custom reader: tsf |
| prabakaranchandran/enterprise-dynamics-corpus-v1 | formats that need unpacking or a custom reader: sqlite |
| alanwgt/fipex-veiculos-brasil | unrecognised file formats: sql |
| dubairealestatedata/dubai-real-estate-sales-transactions | event-level table: needs aggregation to a regular step |
| pracadata/brazil-fuel-prices | event-level table: needs aggregation to a regular step |
| AbdullahImran/Saudi-Supply-Demand-Datasets | event-level table: needs aggregation to a regular step |
| earino/austin-911-response | event-level table: needs aggregation to a regular step |
| hsilvosa/gdelt-events | event-level table: needs aggregation to a regular step |
| jhumbl/police-crime-london | event-level table: needs aggregation to a regular step |
| matanzig/Disney-Theme-Park-Queue-Dynamics | unrecognised file formats: mp4 |
| Nason/open-policing-database | unrecognised file formats: duckdb |
| zwn22/NC_Crime | formats that need unpacking or a custom reader: zip |
| yuiseki/un-wpp [WPP2024] | formats that need unpacking or a custom reader: zip |
| yuiseki/undp-hdr | formats that need unpacking or a custom reader: xlsx |
| zaai-ai/hierarchical_time_series_datasets | formats that need unpacking or a custom reader: pkl |
| AstroMLCore/AstroM3Dataset | formats that need unpacking or a custom reader: gz, zip |
| AutonLab/Timeseries-PILE [anomaly_detection/TSB-UAD-Public] | unrecognised file formats: out |
| cangyeone/SeismicX-Cont | formats that need unpacking or a custom reader: h5, pt, sqlite, zst |
| lxr11111/telemanom [SMAP,MSL] | formats that need unpacking or a custom reader: h5, npy |
| phanerozoic/qiskit-calibration-drift | event-level table: needs aggregation to a regular step |
| snowsadh/goes-omni-electron-flux-forecasting | unrecognised file formats: sha256 |
| autogluon/chronos_datasets_extra [brazilian_cities_temperature] | no data files found |
| DuckerMaster/thaiclim-district | unrecognised file formats: geoparquet |
| edilia12380/era5-daily-weather-western-europe | formats that need unpacking or a custom reader: nc |
| excarta/madis2020 | no data files found |
| jacobbieker/dwd | formats that need unpacking or a custom reader: zip |
| MeteoSwiss/PeakWeather | formats that need unpacking or a custom reader: zip |
| notadib/NASA-Power-Daily-Weather | formats that need unpacking or a custom reader: pt |
| Pahzuzu/PAH-ENV-001-serc-soil-signals | unrecognised file formats: sha256 |
| hsilvosa/noaa-ghcn-daily | formats that need unpacking or a custom reader: tar.gz |
| kashif/App_Flow [default] | formats that need unpacking or a custom reader: zip |
| Lemma-RCA-NEC/Cloud_Computing_Preprocessed | formats that need unpacking or a custom reader: zip |
| Lightcap/multi-domain-cloudflare-observability | unrecognised file formats: sql |
| pyToshka/attacks-daily | event-level table: needs aggregation to a regular step |
| Salesforce/cloudops_tsf [alibaba_cluster_trace_2018] | formats that need unpacking or a custom reader: zip |
| Salesforce/cloudops_tsf [borg_cluster_data_2011] | formats that need unpacking or a custom reader: zip |
| Salesforce/cloudops_tsf [azure_vm_traces_2017] | formats that need unpacking or a custom reader: zip |

## 10. Problems

Every row that is missing, renamed, empty, gated without access, in error, or whose Hub contents disagree with the catalog row.

| catalog line | dataset | level | problem |
|---|---|---|---|
| 3 | alexroz/CarbonFluxBench | footers | the dataset viewer does not work for this repo |
| 3 | alexroz/CarbonFluxBench [ERA5] | footers | the dataset viewer does not work for this repo |
| 3 | alexroz/CarbonFluxBench [MOD09GA] | footers | the dataset viewer does not work for this repo |
| 3 | alexroz/CarbonFluxBench [target_fluxes] | files_only | empty: no data files in the scope / the dataset viewer does not work for this repo |
| 4 | dar4huggingface/sniim-mexico-sample | viewer+footers | size: catalog says 0.05 GB, the Hub file listing sums to 6.51 MB |
| 8 | Emulated-Inc/sci-county-crop-yield-new-year | files_only | format: catalog says parquet, the scope has no parquet file (tar) / the dataset viewer does not work for this repo |
| 10 | globalsquiddatabase/Tuna-Fisheries-Dataset | files_only | format: catalog says parquet, the scope has no parquet file (7z, csv, fws, json, mdb, npy, r, rdata, tsv, txt, xlsx, zip) / the dataset viewer does not work for this repo |
| 11 | hesandism/colombo-tea-auction-prices | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) |
| 12 | judicialmind/CropBench | gated | gated, no access / 50 footer(s) not readable, first: forecasts/chronos2/production_univariate_2015.parquet: HTTP 403 GatedRepo Access to dataset judicialmind/CropBench is restricted and you are not in the authoriz |
| 12 | judicialmind/CropBench [covariates] | gated | gated, no access / 53 footer(s) not readable, first: forecasts/chronos2/production_univariate_2015.parquet: HTTP 403 GatedRepo Access to dataset judicialmind/CropBench is restricted and you are not in the authoriz |
| 12 | judicialmind/CropBench [panel] | gated | gated, no access / 52 footer(s) not readable, first: forecasts/chronos2/production_univariate_2015.parquet: HTTP 403 GatedRepo Access to dataset judicialmind/CropBench is restricted and you are not in the authoriz |
| 12 | judicialmind/CropBench [series] | gated | gated, no access / 51 footer(s) not readable, first: forecasts/chronos2/production_univariate_2015.parquet: HTTP 403 GatedRepo Access to dataset judicialmind/CropBench is restricted and you are not in the authoriz |
| 13 | judicialmind/MandiBench [series/all.parquet] | gated | gated, no access / 1 footer(s) not readable, first: series/all.parquet: HTTP 403 GatedRepo Access to dataset judicialmind/MandiBench is restricted and you are not in the authorized list. Visit https://huggingface |
| 16 | notadib/usa-corn-belt-crop-yield | files_only | format: catalog says parquet, the scope has no parquet file (csv, json, npz) / the dataset viewer does not work for this repo |
| 17 | Project-AgML/usda_nass_crop_statistics [boundaries_county] | viewer+footers | points: catalog says 10M-100M rows, metadata gives 3.23k points (3.23k rows) |
| 17 | Project-AgML/usda_nass_crop_statistics [boundaries_state] | viewer+footers | points: catalog says 10M-100M rows, metadata gives 56 points (56 rows) |
| 17 | Project-AgML/usda_nass_crop_statistics [dim_county] | viewer+footers | points: catalog says 10M-100M rows, metadata gives 3.23k points (3.23k rows) |
| 17 | Project-AgML/usda_nass_crop_statistics [dim_state] | viewer+footers | points: catalog says 10M-100M rows, metadata gives 56 points (56 rows) |
| 17 | Project-AgML/usda_nass_crop_statistics [forecast] | viewer+footers | points: catalog says 10M-100M rows, metadata gives 45.75k points (45.75k rows) |
| 18 | Roshangupta007/YieldFormer-DE-data | viewer+footers | size: catalog says 1.7 GB, the Hub file listing sums to 721.10 MB |
| 18 | Roshangupta007/YieldFormer-DE-data [backtest_forecasts_2010_2025] | viewer+footers | size: catalog says 1.7 GB, the Hub file listing sums to 721.10 MB / points: catalog says 10M-100M rows, metadata gives 299.06k points (37.38k rows) |
| 18 | Roshangupta007/YieldFormer-DE-data [crop_areas_census] | viewer+footers | size: catalog says 1.7 GB, the Hub file listing sums to 721.10 MB / points: catalog says 10M-100M rows, metadata gives 41.27k points (20.64k rows) |
| 18 | Roshangupta007/YieldFormer-DE-data [crop_development_stages] | viewer+footers | size: catalog says 1.7 GB, the Hub file listing sums to 721.10 MB |
| 18 | Roshangupta007/YieldFormer-DE-data [daily_inputs] | viewer+footers | size: catalog says 1.7 GB, the Hub file listing sums to 721.10 MB |
| 18 | Roshangupta007/YieldFormer-DE-data [district_codes] | viewer+footers | size: catalog says 1.7 GB, the Hub file listing sums to 721.10 MB |
| 18 | Roshangupta007/YieldFormer-DE-data [official_yields] | viewer+footers | size: catalog says 1.7 GB, the Hub file listing sums to 721.10 MB / points: catalog says 10M-100M rows, metadata gives 72.68k points (72.68k rows) |
| 18 | Roshangupta007/YieldFormer-DE-data [satellite_arable_land] | viewer+footers | size: catalog says 1.7 GB, the Hub file listing sums to 721.10 MB |
| 18 | Roshangupta007/YieldFormer-DE-data [terrain] | viewer+footers | size: catalog says 1.7 GB, the Hub file listing sums to 721.10 MB / points: catalog says 10M-100M rows, metadata gives 1.60k points (401 rows) |
| 19 | Salesforce/lotsa_data [beijing_air_quality] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB / points: catalog says 420768, metadata gives 4.63M points (12 rows) |
| 20 | Salesforce/lotsa_data [china_air_quality] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 21 | Salesforce/lotsa_data [sunspot_with_missing] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 22 | taziksh/itrdb-tree-rings [files] | viewer+footers | points: catalog says 10M-100M, metadata gives 161.41k points (26.90k rows) |
| 22 | taziksh/itrdb-tree-rings [sites] | viewer+footers | points: catalog says 10M-100M, metadata gives 13.97k points (6.98k rows) |
| 23 | taziksh/usgs-earthquakes | viewer+footers | size: catalog says 0.6 GB, the Hub file listing sums to 261.98 MB |
| 24 | teamSynapse/sri-lanka-weekly-vegetable-retail-farmer-price-dataset | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) |
| 25 | yuiseki/kyoto-bear-sighting | footers | the dataset viewer does not work for this repo |
| 26 | thuml/UTSD [AustraliaRainfall / BeijingPM25Quality / BenzeneConcentration] | viewer+footers | subset text not matched on the Hub: AustraliaRainfall; BeijingPM25Quality; BenzeneConcentration / size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB |
| 26 | thuml/UTSD [UTSD-12G] | viewer+footers | size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB / points: catalog says 11.54M / 3.66M / 16.34M, metadata gives 965.34M points (289.56k rows) |
| 26 | thuml/UTSD [UTSD-1G] | viewer+footers | size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB |
| 26 | thuml/UTSD [UTSD-2G] | viewer+footers | size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB |
| 26 | thuml/UTSD [UTSD-4G] | files_only | size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB |
| 26 | thuml/UTSD [default] | viewer+footers | size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB |
| 27 | yuiseki/faostat [raw/2026-10-04] | files_only | format: catalog says csv, the scope holds json, zip / the dataset viewer does not work for this repo |
| 29 | AkshayKarthick/india-weather-air-quality | viewer+footers | size: catalog says 0.01 GB, the Hub file listing sums to 17.84 MB |
| 29 | AkshayKarthick/india-weather-air-quality [air_quality] | viewer+footers | size: catalog says 0.01 GB, the Hub file listing sums to 17.84 MB |
| 29 | AkshayKarthick/india-weather-air-quality [cities] | viewer+footers | size: catalog says 0.01 GB, the Hub file listing sums to 17.84 MB |
| 29 | AkshayKarthick/india-weather-air-quality [weather] | viewer+footers | size: catalog says 0.01 GB, the Hub file listing sums to 17.84 MB |
| 31 | cosuleabianca/eea-pm25-forecasting | files_only | the dataset viewer does not work for this repo |
| 32 | danielrosehill/Jerusalem-Air-Quality-Shabbat | viewer+footers | size: catalog says 0.13 GB, the Hub file listing sums to 64.06 MB |
| 32 | danielrosehill/Jerusalem-Air-Quality-Shabbat [aqi_hourly] | viewer+footers | size: catalog says 0.13 GB, the Hub file listing sums to 64.06 MB / frequency: catalog says 5min, metadata says 1h (stated in the config / folder name (aqi_hourly)) |
| 32 | danielrosehill/Jerusalem-Air-Quality-Shabbat [control_deweathered] | viewer+footers | size: catalog says 0.13 GB, the Hub file listing sums to 64.06 MB |
| 32 | danielrosehill/Jerusalem-Air-Quality-Shabbat [default] | viewer+footers | size: catalog says 0.13 GB, the Hub file listing sums to 64.06 MB |
| 32 | danielrosehill/Jerusalem-Air-Quality-Shabbat [jlm_deweathered] | viewer+footers | size: catalog says 0.13 GB, the Hub file listing sums to 64.06 MB |
| 32 | danielrosehill/Jerusalem-Air-Quality-Shabbat [weather_hourly] | viewer+footers | size: catalog says 0.13 GB, the Hub file listing sums to 64.06 MB / frequency: catalog says 5min, metadata says 1h (stated in the config / folder name (weather_hourly)) |
| 33 | dmariaa70/METRAQ-Air-Quality | viewer+footers | size: catalog says 26.54 GB, the Hub file listing sums to 6.06 GB |
| 34 | haidang2405/aqshift-us-leakage-safe-air-quality [monitors] | viewer+footers | points: catalog says 10M-100M rows, metadata gives 17.51k points (8.76k rows) |
| 34 | haidang2405/aqshift-us-leakage-safe-air-quality [sites] | viewer+footers | points: catalog says 10M-100M rows, metadata gives 20.99k points (20.99k rows) |
| 36 | khajlk/copenhagen-air-quality-hourly | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 2.25 MB |
| 36 | khajlk/copenhagen-air-quality-hourly [no2_hourly] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 2.25 MB |
| 36 | khajlk/copenhagen-air-quality-hourly [weather_hourly] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 2.25 MB |
| 38 | National-Digital/act-air-quality | viewer+footers | size: catalog says 0.06 GB, the Hub file listing sums to 8.40 MB |
| 40 | nju-zhangsq/RMISC [PM25FiveCities] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 1151784, the footers give 2145480 |
| 41 | nju-zhangsq/RMISC [USAirPollution] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 42 | nju-zhangsq/RMISC [XiamenAirQuality] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 45 | Envirocatalysts/hourly_pollutants_data | viewer+footers | size: catalog says 82.71 GB, the Hub file listing sums to 938.42 MB |
| 46 | meganariley/us-air-quality-hourly | files_only | empty: no data files in the scope / the dataset viewer does not work for this repo |
| 47 | xuxing123/AirQualityBench | files_only | the dataset viewer does not work for this repo |
| 48 | yuiseki/edgar-ghg-countries | files_only | format: catalog says parquet, the scope has no parquet file (json, txt, zip) / the dataset viewer does not work for this repo |
| 49 | aaron-freedman/banking-concentration-data | footers | size: catalog says 5.16 GB, the Hub file listing sums to 1.28 GB / the dataset viewer does not work for this repo |
| 51 | aikstockdata/korea-equity-daily | viewer+footers | size: catalog says 0.77 GB, the Hub file listing sums to 162.59 MB |
| 51 | aikstockdata/korea-equity-daily [daily_prices] | viewer+footers | size: catalog says 0.77 GB, the Hub file listing sums to 162.59 MB |
| 51 | aikstockdata/korea-equity-daily [filing_impact_summary] | viewer+footers | size: catalog says 0.77 GB, the Hub file listing sums to 162.59 MB / points: catalog says ~895k, metadata gives 957 points (29 rows) |
| 51 | aikstockdata/korea-equity-daily [filing_price_impact] | viewer+footers | size: catalog says 0.77 GB, the Hub file listing sums to 162.59 MB / points: catalog says ~895k, metadata gives 36.44k points (2.80k rows) |
| 51 | aikstockdata/korea-equity-daily [stocks] | viewer+footers | size: catalog says 0.77 GB, the Hub file listing sums to 162.59 MB / points: catalog says ~895k, metadata gives 8.36k points (2.79k rows) |
| 52 | arabalika/us-rent-history | gated | gated, no access / 50 footer(s) not readable, first: facts/floorplan_obs/date=2026-08-23/data.parquet: HTTP 403 GatedRepo Access to dataset arabalika/us-rent-history is restricted and you are not in the authorized |
| 52 | arabalika/us-rent-history [floorplan_obs] | gated | gated, no access / 150 footer(s) not readable, first: facts/floorplan_obs/date=2026-08-23/data.parquet: HTTP 403 GatedRepo Access to dataset arabalika/us-rent-history is restricted and you are not in the authorized |
| 52 | arabalika/us-rent-history [property_obs] | gated | gated, no access / 100 footer(s) not readable, first: facts/floorplan_obs/date=2026-08-23/data.parquet: HTTP 403 GatedRepo Access to dataset arabalika/us-rent-history is restricted and you are not in the authorized |
| 52 | arabalika/us-rent-history [unit_obs] | gated | gated, no access / 200 footer(s) not readable, first: facts/floorplan_obs/date=2026-08-23/data.parquet: HTTP 403 GatedRepo Access to dataset arabalika/us-rent-history is restricted and you are not in the authorized |
| 53 | autogluon/chronos_datasets [exchange_rate] | viewer+footers | licence: catalog says per-dataset (card: check ds.info); repo not gated, the card says other |
| 54 | autogluon/chronos_datasets [monash_cif_2016] | viewer+footers | licence: catalog says per-dataset (card: check ds.info); repo not gated, the card says other |
| 55 | autogluon/chronos_datasets [monash_m1_monthly, monash_m3_monthly] | viewer+footers | subset text not matched on the Hub: _quarterly; _yearly; _quarterly; _yearly / licence: catalog says per-dataset (card: check ds.info); repo not gated, the card says other / frequency: catalog says M / Q / Y, metadata says 1M (stated in the config / folder name (monash_m1_monthly)) |
| 56 | autogluon/chronos_datasets [monash_nn5_weekly, nn5] | viewer+footers | licence: catalog says per-dataset (card: check ds.info); repo not gated, the card says other / frequency: catalog says W / D, metadata says 1W (stated in the config / folder name (monash_nn5_weekly)) |
| 57 | AYUSHKHAIRE/all-stock-market-data-daily-updates | files_only | the dataset viewer does not work for this repo |
| 60 | empulse-ai/empulse-street-fx-index | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) |
| 64 | misikoff/zillow [days_on_market] | files_only | format: catalog says jsonl, the scope holds csv / size: catalog says 6.71 GB, the Hub file listing sums to 2.70 GB / the dataset viewer does not work for this repo |
| 65 | misikoff/zillow [for_sale_listings] | files_only | format: catalog says jsonl, the scope holds csv / size: catalog says 6.71 GB, the Hub file listing sums to 2.70 GB / the dataset viewer does not work for this repo |
| 66 | misikoff/zillow [home_values] | files_only | format: catalog says jsonl, the scope holds csv / size: catalog says 6.71 GB, the Hub file listing sums to 2.70 GB / the dataset viewer does not work for this repo |
| 67 | misikoff/zillow [rentals] | files_only | format: catalog says jsonl, the scope holds csv / size: catalog says 6.71 GB, the Hub file listing sums to 2.70 GB / the dataset viewer does not work for this repo |
| 68 | misikoff/zillow [sales] | files_only | format: catalog says jsonl, the scope holds csv / size: catalog says 6.71 GB, the Hub file listing sums to 2.70 GB / the dataset viewer does not work for this repo |
| 69 | MonitorBankRates/us-bank-rate-averages | files_only | format: catalog says parquet, the scope has no parquet file (csv) / the dataset viewer does not work for this repo |
| 71 | nju-zhangsq/RMISC [AMarketChina] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 72 | nju-zhangsq/RMISC [ChinaMinuteStock] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 73 | nju-zhangsq/RMISC [CSI500] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 643703760, the footers give 551746080 |
| 74 | nju-zhangsq/RMISC [NIFTYStock] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 4244406, the footers give 4722090 |
| 75 | nju-zhangsq/RMISC [SP500] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 76 | nju-zhangsq/RMISC [StockFactorsCleaned] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 77 | nju-zhangsq/RMISC [StockMarketData] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 78 | nju-zhangsq/RMISC [TushareETFDaily] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 79 | nju-zhangsq/RMISC [TushareIndexDaily] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 80 | nju-zhangsq/RMISC [TushareStockDaily] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 155794320, the footers give 127468080 |
| 81 | nju-zhangsq/RMISC [TushareStockWeekly] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 32636395, the footers give 26702505 |
| 82 | nju-zhangsq/RMISC [UKEconomy] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 399900, the footers give 45554592 |
| 83 | paperswithbacktest/Forex-Daily-Price [default] | gated | gated, no access / 1 footer(s) not readable, first: data/train-00000-of-00001.parquet: HTTP 403 GatedRepo Access to dataset paperswithbacktest/Forex-Daily-Price is restricted and you are not in the authorized lis |
| 84 | paperswithbacktest/Indices-Daily-Price [default] | gated | gated, no access / 1 footer(s) not readable, first: data/train-00000-of-00001.parquet: HTTP 403 GatedRepo Access to dataset paperswithbacktest/Indices-Daily-Price is restricted and you are not in the authorized l |
| 85 | paperswithbacktest/Stocks-Daily-Price [default] | gated | gated, no access / 50 footer(s) not readable, first: data/train-1962-01.parquet: HTTP 403 GatedRepo Access to dataset paperswithbacktest/Stocks-Daily-Price is restricted and you are not in the authorized list. Vis |
| 89 | Rogendo/forex-dukas-copy [default] | footers | the dataset viewer does not work for this repo |
| 90 | Rrishab/OHLCV-1m [default] | footers | points: catalog says unknown (87.7 GB; order of billions), metadata gives 38.32B points (7.66B rows) / the dataset viewer does not work for this repo |
| 91 | Salesforce/lotsa_data [cif_2016_12] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 92 | Salesforce/lotsa_data [godaddy] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 93 | Salesforce/lotsa_data [m1_monthly] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 94 | Salesforce/lotsa_data [nn5_daily_with_missing] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 95 | sevriukov/bank101 | files_only | format: catalog says parquet, the scope has no parquet file (csv, csv.zst, sql) / the dataset viewer does not work for this repo |
| 96 | showlcw/china-cpi-ppi-pmi | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) |
| 96 | showlcw/china-cpi-ppi-pmi [all] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) |
| 96 | showlcw/china-cpi-ppi-pmi [cpi] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) |
| 96 | showlcw/china-cpi-ppi-pmi [pmi] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) |
| 96 | showlcw/china-cpi-ppi-pmi [ppi] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) |
| 98 | thanna94/us-building-permits | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) |
| 100 | THULab/xauusd [default] | files_only | the dataset viewer does not work for this repo |
| 101 | tidy-finance/factor-library | viewer+footers | size: catalog says 122.32 GB, the Hub file listing sums to 36.25 GB |
| 101 | tidy-finance/factor-library [costs] | viewer+footers | size: catalog says 122.32 GB, the Hub file listing sums to 36.25 GB |
| 101 | tidy-finance/factor-library [default] | viewer+footers | size: catalog says 122.32 GB, the Hub file listing sums to 36.25 GB |
| 102 | Traders-Lab/TroveLedger [daily / hourly / 1-minute files] | footers | subset text not matched on the Hub: daily; hourly; 1-minute files / size: catalog says 383.77 GB, the Hub file listing sums to 3.29 GB / the dataset viewer does not work for this repo |
| 103 | trentmkelly/polymarket_historical_data [market_snapshots] | footers | points: catalog says unknown (15.1 GB), metadata gives 16.79M points (5.60M rows) |
| 104 | wrpbr/pncp-compras-modalidade-dia-municipio | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv.gz) / size: catalog says 0.08 GB, the Hub file listing sums to 30.35 MB |
| 106 | yadavsaurabh/us-stocks-and-search-trends | viewer+footers | size: catalog says 19.72 GB, the Hub file listing sums to 11.71 GB |
| 106 | yadavsaurabh/us-stocks-and-search-trends [1d] | viewer+footers | size: catalog says 19.72 GB, the Hub file listing sums to 11.71 GB |
| 106 | yadavsaurabh/us-stocks-and-search-trends [1d_L-Z] | viewer+footers | size: catalog says 19.72 GB, the Hub file listing sums to 11.71 GB |
| 106 | yadavsaurabh/us-stocks-and-search-trends [1h] | viewer+footers | size: catalog says 19.72 GB, the Hub file listing sums to 11.71 GB / frequency: catalog says 1D and intraday, metadata says 1h (stated in the config / folder name (1h)) |
| 106 | yadavsaurabh/us-stocks-and-search-trends [1h_L-Z] | files_only | size: catalog says 19.72 GB, the Hub file listing sums to 11.71 GB / frequency: catalog says 1D and intraday, metadata says 1h (stated in the config / folder name) |
| 106 | yadavsaurabh/us-stocks-and-search-trends [1m] | files_only | size: catalog says 19.72 GB, the Hub file listing sums to 11.71 GB / frequency: catalog says 1D and intraday, metadata says 1min (stated in the config / folder name) |
| 106 | yadavsaurabh/us-stocks-and-search-trends [1m_L-Z] | files_only | size: catalog says 19.72 GB, the Hub file listing sums to 11.71 GB / frequency: catalog says 1D and intraday, metadata says 1min (stated in the config / folder name) |
| 106 | yadavsaurabh/us-stocks-and-search-trends [2m] | files_only | size: catalog says 19.72 GB, the Hub file listing sums to 11.71 GB / frequency: catalog says 1D and intraday, metadata says 2min (stated in the config / folder name) |
| 106 | yadavsaurabh/us-stocks-and-search-trends [5m] | files_only | size: catalog says 19.72 GB, the Hub file listing sums to 11.71 GB / frequency: catalog says 1D and intraday, metadata says 5min (stated in the config / folder name) |
| 106 | yadavsaurabh/us-stocks-and-search-trends [trends] | files_only | size: catalog says 19.72 GB, the Hub file listing sums to 11.71 GB |
| 108 | yuiseki/ilostat | files_only | the dataset viewer does not work for this repo |
| 109 | yuiseki/maddison-project [2018.regional_data] | viewer+footers | points: catalog says 100K-1M rows, metadata gives 3.12k points (1.04k rows) |
| 110 | yuiseki/pwt [8.0.depreciation_rates] | viewer+footers | points: catalog says 1M-10M rows, metadata gives 8.06k points (8.06k rows) |
| 110 | yuiseki/pwt [8.0.esh_bilateral_cor] | viewer+footers | points: catalog says 1M-10M rows, metadata gives 43.77k points (43.77k rows) |
| 110 | yuiseki/pwt [8.1.esh_bilateral_cor] | viewer+footers | points: catalog says 1M-10M rows, metadata gives 43.77k points (43.77k rows) |
| 110 | yuiseki/pwt [9.0.esh_bilateral_cor] | viewer+footers | points: catalog says 1M-10M rows, metadata gives 76.17k points (76.17k rows) |
| 110 | yuiseki/pwt [9.1.esh_bilateral_cor] | viewer+footers | points: catalog says 1M-10M rows, metadata gives 76.17k points (76.17k rows) |
| 111 | yuiseki/undata-sdmx | viewer+footers | size: catalog says 0.06 GB, the Hub file listing sums to 514.94 MB |
| 111 | yuiseki/undata-sdmx [DF_SEEA_AEA] | viewer+footers | size: catalog says 0.06 GB, the Hub file listing sums to 514.94 MB |
| 111 | yuiseki/undata-sdmx [DF_SEEA_ENERGY] | viewer+footers | size: catalog says 0.06 GB, the Hub file listing sums to 514.94 MB |
| 111 | yuiseki/undata-sdmx [DF_UNDATA_ENERGY] | viewer+footers | size: catalog says 0.06 GB, the Hub file listing sums to 514.94 MB |
| 111 | yuiseki/undata-sdmx [DF_UNData_EnergyBalance] | viewer+footers | size: catalog says 0.06 GB, the Hub file listing sums to 514.94 MB |
| 111 | yuiseki/undata-sdmx [NA_MAIN] | viewer+footers | size: catalog says 0.06 GB, the Hub file listing sums to 514.94 MB |
| 111 | yuiseki/undata-sdmx [codelists] | viewer+footers | size: catalog says 0.06 GB, the Hub file listing sums to 514.94 MB |
| 111 | yuiseki/undata-sdmx [compilers] | viewer+footers | size: catalog says 0.06 GB, the Hub file listing sums to 514.94 MB |
| 111 | yuiseki/undata-sdmx [components] | viewer+footers | size: catalog says 0.06 GB, the Hub file listing sums to 514.94 MB |
| 112 | ashwingopalsamy/central-bank-policy-rates [annual] | viewer+footers | frequency: catalog says 1D (step), metadata says 1Y (stated in the config / folder name (annual)) |
| 114 | kavelaltd/historysaid-global-economic | viewer+footers | size: catalog says 0.35 GB, the Hub file listing sums to 197.96 MB |
| 115 | rogerdehe/klines-binance [spot, futures] | footers | subset text not matched on the Hub: 1m..1d / size: catalog says 616.03 GB, the Hub file listing sums to 57.69 GB / the dataset viewer does not work for this repo |
| 116 | rogerdehe/klines-bybit [futures] | viewer+footers | subset text not matched on the Hub: 3m..1d / size: catalog says 479.21 GB, the Hub file listing sums to 38.47 GB |
| 117 | rogerdehe/klines-okx [futures] | viewer+footers | subset text not matched on the Hub: 15m..1d / size: catalog says 226.86 GB, the Hub file listing sums to 16.42 GB |
| 118 | tradecatlabs/binance-futures-ohlcv-2018-2026 [candles_1m] | viewer+footers | subset text not matched on the Hub: candles_1m / format: catalog says csv;, the scope holds csv.gz, dat, gz, log, zst |
| 120 | AllRatesToday/central-bank-exchange-rates | files_only | size: catalog says 13.85 GB, the Hub file listing sums to 577.58 MB |
| 122 | autogluon/chronos_datasets [monash_tourism_monthly] | viewer+footers | subset text not matched on the Hub: _quarterly; _yearly / licence: catalog says per-dataset (card: check ds.info); repo not gated, the card says other / frequency: catalog says M / Q / Y, metadata says 1M (stated in the config / folder name (monash_tourism_monthly)) |
| 123 | bkoyuncu/MacroEcon [one config per country (64)] | viewer+footers | subset text not matched on the Hub: one config per country |
| 124 | JrVillabona/resonance-data | viewer+footers | size: catalog says 10.5 GB, the Hub file listing sums to 3.85 GB |
| 124 | JrVillabona/resonance-data [binance_bulk_bars] | viewer+footers | size: catalog says 10.5 GB, the Hub file listing sums to 3.85 GB |
| 124 | JrVillabona/resonance-data [cross_venue_bars] | viewer+footers | size: catalog says 10.5 GB, the Hub file listing sums to 3.85 GB |
| 124 | JrVillabona/resonance-data [crypto_derivatives] | viewer+footers | size: catalog says 10.5 GB, the Hub file listing sums to 3.85 GB |
| 124 | JrVillabona/resonance-data [default] | viewer+footers | size: catalog says 10.5 GB, the Hub file listing sums to 3.85 GB |
| 124 | JrVillabona/resonance-data [derived_feature_lineage] | viewer+footers | size: catalog says 10.5 GB, the Hub file listing sums to 3.85 GB / frequency: catalog says 1D, metadata says 1Y (read from every data file path (not a config name)) |
| 124 | JrVillabona/resonance-data [documents] | viewer+footers | size: catalog says 10.5 GB, the Hub file listing sums to 3.85 GB |
| 124 | JrVillabona/resonance-data [events] | viewer+footers | size: catalog says 10.5 GB, the Hub file listing sums to 3.85 GB |
| 124 | JrVillabona/resonance-data [fiscal_observations] | viewer+footers | size: catalog says 10.5 GB, the Hub file listing sums to 3.85 GB |
| 124 | JrVillabona/resonance-data [fundamentals] | viewer+footers | size: catalog says 10.5 GB, the Hub file listing sums to 3.85 GB |
| 124 | JrVillabona/resonance-data [macro_point_in_time] | viewer+footers | size: catalog says 10.5 GB, the Hub file listing sums to 3.85 GB |
| 124 | JrVillabona/resonance-data [market_bars_1d] | viewer+footers | size: catalog says 10.5 GB, the Hub file listing sums to 3.85 GB |
| 124 | JrVillabona/resonance-data [official_macro_observations] | viewer+footers | size: catalog says 10.5 GB, the Hub file listing sums to 3.85 GB |
| 124 | JrVillabona/resonance-data [positioning_observations] | viewer+footers | size: catalog says 10.5 GB, the Hub file listing sums to 3.85 GB / frequency: catalog says 1D, metadata says 1W (freq column (weekly)) |
| 124 | JrVillabona/resonance-data [scalar_observations] | viewer+footers | size: catalog says 10.5 GB, the Hub file listing sums to 3.85 GB |
| 124 | JrVillabona/resonance-data [sec_entities] | viewer+footers | size: catalog says 10.5 GB, the Hub file listing sums to 3.85 GB |
| 124 | JrVillabona/resonance-data [sec_filings] | viewer+footers | size: catalog says 10.5 GB, the Hub file listing sums to 3.85 GB |
| 125 | paperswithbacktest/Commodities-Daily-Price [default] | gated | gated, no access / 1 footer(s) not readable, first: data/train-00000-of-00001.parquet: HTTP 403 GatedRepo Access to dataset paperswithbacktest/Commodities-Daily-Price is restricted and you are not in the authoriz |
| 126 | Salesforce/lotsa_data [tourism_monthly] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 127 | SemantaAI/semantaai-fx-majors | viewer+footers | size: catalog says 123.98 GB, the Hub file listing sums to 63.72 GB |
| 128 | syncster-data/economic-data | footers | the dataset viewer does not work for this repo |
| 129 | yadavsaurabh/macro-markets | viewer+footers | size: catalog says 1.03 GB, the Hub file listing sums to 379.85 MB |
| 129 | yadavsaurabh/macro-markets [1d] | viewer+footers | size: catalog says 1.03 GB, the Hub file listing sums to 379.85 MB |
| 129 | yadavsaurabh/macro-markets [1h] | viewer+footers | size: catalog says 1.03 GB, the Hub file listing sums to 379.85 MB / frequency: catalog says 1D and intraday, metadata says 1h (stated in the config / folder name (1h)) |
| 129 | yadavsaurabh/macro-markets [1m] | viewer+footers | size: catalog says 1.03 GB, the Hub file listing sums to 379.85 MB / frequency: catalog says 1D and intraday, metadata says 1min (stated in the config / folder name (1m)) |
| 129 | yadavsaurabh/macro-markets [2m] | viewer+footers | size: catalog says 1.03 GB, the Hub file listing sums to 379.85 MB / frequency: catalog says 1D and intraday, metadata says 2min (stated in the config / folder name (2m)) |
| 129 | yadavsaurabh/macro-markets [5m] | viewer+footers | size: catalog says 1.03 GB, the Hub file listing sums to 379.85 MB / frequency: catalog says 1D and intraday, metadata says 5min (stated in the config / folder name (5m)) |
| 135 | ai-iot/EnergyBench [NEEA] | files_only | format: catalog says parquet, the scope has no parquet file (csv) |
| 139 | ai-iot/EnergyBench [SAVE] | files_only | format: catalog says parquet, the scope has no parquet file (csv) |
| 141 | ai-iot/EnergyBench [UKST] | files_only | format: catalog says parquet, the scope has no parquet file (csv) |
| 143 | akderekaan/de-power-forecast-data [entsoe/a75] | footers | subset text not matched on the Hub: a69; weather runs |
| 144 | alwaysbyx/Bear-Data | files_only | the dataset viewer does not work for this repo |
| 145 | andrewlee1807/Gyeonggi | files_only | the dataset viewer does not work for this repo |
| 146 | autogluon/chronos_datasets [monash_london_smart_meters] | viewer+footers | licence: catalog says per-dataset (card: check ds.info); repo not gated, the card says other |
| 147 | autogluon/chronos_datasets [wind_farms_hourly, wind_farms_daily] | viewer+footers | licence: catalog says per-dataset (card: check ds.info); repo not gated, the card says other |
| 148 | Battery-Life/BatteryLife_Raw [Zn-ion, Na-ion, CALB] | files_only | format: catalog says csv-like, the scope holds csv, xlsx / the dataset viewer does not work for this repo |
| 149 | bsebench-org/nasa-rw-2014 [partitioned by cell_id] | viewer+footers | subset text not matched on the Hub: partitioned by cell_id |
| 150 | bsebench-org/warwick-second-life-dm-2025-raw [raw mirror (746 csv, 309 mat)] | viewer+footers | subset text not matched on the Hub: raw mirror |
| 152 | EDS-lab/gas-demand [demand+weather+metadata] | gated | subset text not matched on the Hub: demand+weather+metadata / gated, no access / size: catalog says 0.03 GB, the Hub file listing sums to 18.13 MB / 3 footer(s) not readable, first: data/demand.parquet: HTTP 403 GatedRepo Access to dataset EDS-lab/gas-demand is restricted and you are not in the authorized list. Visit https://huggingface.co/ |
| 152 | EDS-lab/gas-demand [demand] | gated | gated, no access / size: catalog says 0.03 GB, the Hub file listing sums to 18.13 MB / 4 footer(s) not readable, first: data/demand.parquet: HTTP 403 GatedRepo Access to dataset EDS-lab/gas-demand is restricted and you are not in the authorized list. Visit https://huggingface.co/ |
| 152 | EDS-lab/gas-demand [metadata] | gated | gated, no access / size: catalog says 0.03 GB, the Hub file listing sums to 18.13 MB / 5 footer(s) not readable, first: data/demand.parquet: HTTP 403 GatedRepo Access to dataset EDS-lab/gas-demand is restricted and you are not in the authorized list. Visit https://huggingface.co/ |
| 152 | EDS-lab/gas-demand [weather] | gated | gated, no access / size: catalog says 0.03 GB, the Hub file listing sums to 18.13 MB / 6 footer(s) not readable, first: data/demand.parquet: HTTP 403 GatedRepo Access to dataset EDS-lab/gas-demand is restricted and you are not in the authorized list. Visit https://huggingface.co/ |
| 153 | Emulated-Inc/battery-capacity-trajectories [Dataset_1_NCA / 2_NCM / 3_NCM_NCA zips] | viewer+footers | subset text not matched on the Hub: Dataset_1_NCA; 2_NCM; 3_NCM_NCA zips / format: catalog says csv, the scope holds zip / 8 footer(s) not readable, first: default/train/0000.parquet: footer of 8550989 bytes is above the limit of 8388608 |
| 154 | fidel-ts/Bear_room | footers | format: catalog says zip/csv, the scope holds csv, json, parquet, pkl, zip / the dataset viewer does not work for this repo |
| 155 | fidel-ts/California_ISO | footers | the dataset viewer does not work for this repo |
| 156 | fidel-ts/Canada_photovoltaics_plants | footers | the dataset viewer does not work for this repo |
| 158 | gtfierro/mortar [hive-partitioned by uuid] | footers | subset text not matched on the Hub: hive-partitioned by uuid / the dataset viewer does not work for this repo |
| 161 | hsilvosa/omie-electricity [marginal_prices] | viewer+footers | points: catalog says 17.2M rows, metadata gives 110.40k points (55.20k rows) |
| 168 | jtupayac/EV_Charging_demand [Se_*.csv sessions (+ INRIX traffic Readings.csv, weather)] | files_only | subset text not matched on the Hub: Se_*.csv sessions / the dataset viewer does not work for this repo |
| 169 | judicialmind/BharatEnergy-Bench | gated | gated, no access / 50 footer(s) not readable, first: data/station_generation/series.parquet: HTTP 403 GatedRepo Access to dataset judicialmind/BharatEnergy-Bench is restricted and you are not in the authorized lis |
| 169 | judicialmind/BharatEnergy-Bench [station_generation] | gated | gated, no access / 51 footer(s) not readable, first: data/station_generation/series.parquet: HTTP 403 GatedRepo Access to dataset judicialmind/BharatEnergy-Bench is restricted and you are not in the authorized lis |
| 169 | judicialmind/BharatEnergy-Bench [unit_generation] | gated | gated, no access / 52 footer(s) not readable, first: data/station_generation/series.parquet: HTTP 403 GatedRepo Access to dataset judicialmind/BharatEnergy-Bench is restricted and you are not in the authorized lis |
| 170 | judicialmind/BharatEnergy-Data [generation_daily.parquet] | gated | gated, no access / 1 footer(s) not readable, first: data/generation_daily.parquet: HTTP 403 GatedRepo Access to dataset judicialmind/BharatEnergy-Data is restricted and you are not in the authorized list. Visit h |
| 171 | kevykibbz/wind-turbine-scada-data-for-early-fault-detection [Wind Farm A/B/C] | viewer+footers | subset text not matched on the Hub: Wind Farm A/B/C |
| 172 | michealsmitch/lithium-ion-cell-capacity-grading-process-curves [train] | viewer+footers | subset text not matched on the Hub: train |
| 175 | nju-zhangsq/RMISC [ApplianceEnergy] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 176 | nju-zhangsq/RMISC [BritainCoal] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 177 | nju-zhangsq/RMISC [BuildingsBenchRealCSV] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 39639306, the footers give 17113561013 |
| 178 | nju-zhangsq/RMISC [HouseholdPower] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 179 | nju-zhangsq/RMISC [LBNL] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 122266150, the footers give 471917481 |
| 180 | nju-zhangsq/RMISC [MZVAV] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 6832789, the footers give 25129377 |
| 181 | nju-zhangsq/RMISC [OccupancyDetection] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 182 | nju-zhangsq/RMISC [OPSD-Household] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 47880484, the footers give 484244320 |
| 183 | nju-zhangsq/RMISC [OPSD-PV-Wind] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 48738960, the footers give 1540010880 |
| 184 | nju-zhangsq/RMISC [OPSD-When2Heat] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 45614480, the footers give 1305922388 |
| 185 | nju-zhangsq/RMISC [Pvdaq] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 8209270, the footers give 123022167 |
| 186 | nju-zhangsq/RMISC [RoomOccupancy] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 187 | nju-zhangsq/RMISC [ShellHackathon] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 188 | nju-zhangsq/RMISC [SmartMeterAus30m] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 1034221356, the footers give 344740452 |
| 189 | nju-zhangsq/RMISC [SmartMeterAus60m] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 345933356, the footers give 172966678 |
| 190 | nju-zhangsq/RMISC [SmartMeterUK30m] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 500654064, the footers give 166884688 |
| 191 | nju-zhangsq/RMISC [SmartMeterUK60m] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 167618068, the footers give 83809034 |
| 192 | nju-zhangsq/RMISC [Solar4Seconds] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 193 | nju-zhangsq/RMISC [TetuanPowerConsumption] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 194 | nju-zhangsq/RMISC [UK-DALE] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 65602297, the footers give 815445360 |
| 195 | nju-zhangsq/RMISC [WindElec] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 196 | nju-zhangsq/RMISC [WindPower4secs] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 198 | openclimatefix/uk_pv [30min.parquet] | gated | subset text not matched on the Hub: 30min.parquet / gated, no access / licence: catalog says MIT, the card says cc-by-4.0 / size: catalog says 91.24 GB, the Hub file listing sums to 6.25 GB / 50 footer(s) not readable, first: 30_minutely/year=2010/month=11/data.parquet: HTTP 403 GatedRepo Access to dataset openclimatefix/uk_pv is restricted and you are not in the authorized list. Vis |
| 199 | openclimatefix/uk_pv [5min.parquet / 2min.parquet] | gated | subset text not matched on the Hub: 5min.parquet; 2min.parquet / gated, no access / licence: catalog says MIT, the card says cc-by-4.0 / size: catalog says 91.24 GB, the Hub file listing sums to 6.25 GB / 100 footer(s) not readable, first: 30_minutely/year=2010/month=11/data.parquet: HTTP 403 GatedRepo Access to dataset openclimatefix/uk_pv is restricted and you are not in the authorized list. Vis |
| 202 | PowerZooJax/PowerZooDataset [ausgrid_zone_substation] | viewer+footers | licence: catalog says mixed upstream (Ausgrid open data), the card says other |
| 203 | PowerZooJax/PowerZooDataset [gb_neso_demand, gb_gen_by_type, gb_market_mid, gb_forecast_actual_demand] | viewer+footers | licence: catalog says mixed upstream (NESO / Elexon open licences), the card says other |
| 206 | Rif-SQL/uk-electricity-generation-mix-2019-2026 | viewer+footers | size: catalog says 0.01 GB, the Hub file listing sums to 33.04 MB |
| 207 | Salesforce/lotsa_data [bdg-2_bear] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 208 | Salesforce/lotsa_data [bdg-2_fox] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 209 | Salesforce/lotsa_data [bdg-2_panther] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 210 | Salesforce/lotsa_data [bdg-2_rat] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 211 | Salesforce/lotsa_data [borealis] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 212 | Salesforce/lotsa_data [ideal] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 213 | Salesforce/lotsa_data [lcl] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 214 | Salesforce/lotsa_data [london_smart_meters_with_missing] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 215 | Salesforce/lotsa_data [residential_load_power] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 216 | Salesforce/lotsa_data [residential_pv_power] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 217 | Salesforce/lotsa_data [sceaux] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 218 | Salesforce/lotsa_data [smart] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 219 | Salesforce/lotsa_data [solar_power] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 220 | Salesforce/lotsa_data [wind_farms_with_missing] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 221 | Salesforce/lotsa_data [wind_power] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 222 | SamuelM0422/Hourly-Electricity-Demand-Brazil-Dataset | viewer+footers | format: catalog says csv/xlsx/parquet, the scope holds parquet |
| 224 | skforecast/ashrae_daily | viewer+footers | size: catalog says 0.01 GB, the Hub file listing sums to 3.67 MB |
| 225 | solarbench/SKIPPD [ground, ground_no_sky_img, ERA5, GFS] | viewer+footers | rows: the card claims 79954 rows, the parquet footers hold 806704 |
| 226 | yzzpz/Charged-data [10 csv per city (some in .rar)] | viewer+footers | subset text not matched on the Hub: 10 csv per city |
| 227 | ai-iot/EnergyBench [real subsets overall (Dataset_V0.0)] | viewer+footers | subset text not matched on the Hub: real subsets overall / points: catalog says ~1.3e9 hourly obs (sum of per-dataset counts on card), metadata gives 67.19B points (88.60M rows) / 1 footer(s) not readable, first: Dataset_V0.0/Synthetic-Energy-Load-Profiles/Hourly/HRSA/01/01-38-1H.parquet: footer of 27190098 bytes is above the limit of 8388608 |
| 228 | alinurdin/time-series-benchmark [eia_hourly, madrid_15min, webtris_15min, usgs_daily, usgs_continuous, hydat_daily, bom_daily, caravan_daily, isd_hourly, swissmetnet_10min, ghcnd_daily] | viewer+footers | licence: catalog says per-subset (public domain, CC BY, OGL), the card says other |
| 229 | marplan6/batgrad [pozzato-2022 (real) vs synthetic-pozzato-2022-m50t (drop)] | viewer+footers | subset text not matched on the Hub: pozzato-2022 |
| 230 | zalizedata/us-grid-hourly-power-plants-dataset [L-eba_gap_report] | viewer+footers | points: catalog says 10M-100M, metadata gives 210 points (70 rows) |
| 230 | zalizedata/us-grid-hourly-power-plants-dataset [L-generators] | viewer+footers | points: catalog says 10M-100M, metadata gives 53.71k points (26.86k rows) |
| 230 | zalizedata/us-grid-hourly-power-plants-dataset [L-plant_month_generation] | viewer+footers | frequency: catalog says 1H, metadata says 1M (stated in the config / folder name (L-plant_month_generation)) |
| 230 | zalizedata/us-grid-hourly-power-plants-dataset [L-plant_profiles] | viewer+footers | points: catalog says 10M-100M, metadata gives 96.79k points (16.13k rows) |
| 230 | zalizedata/us-grid-hourly-power-plants-dataset [L-plants] | viewer+footers | points: catalog says 10M-100M, metadata gives 32.26k points (16.13k rows) |
| 230 | zalizedata/us-grid-hourly-power-plants-dataset [L-state_electricity_prices] | viewer+footers | points: catalog says 10M-100M, metadata gives 73.50k points (73.50k rows) |
| 230 | zalizedata/us-grid-hourly-power-plants-dataset [L-state_generation_monthly] | viewer+footers | points: catalog says 10M-100M, metadata gives 208.78k points (208.78k rows) / frequency: catalog says 1H, metadata says 1M (stated in the config / folder name (L-state_generation_monthly)) |
| 230 | zalizedata/us-grid-hourly-power-plants-dataset [L-utilities] | viewer+footers | points: catalog says 10M-100M, metadata gives 6.64k points (6.64k rows) |
| 230 | zalizedata/us-grid-hourly-power-plants-dataset [M-ba_subregion_hourly] | viewer+footers | points: catalog says 10M-100M, metadata gives 706.29k points (706.29k rows) |
| 230 | zalizedata/us-grid-hourly-power-plants-dataset [M-eba_gap_report] | viewer+footers | points: catalog says 10M-100M, metadata gives 210 points (70 rows) |
| 230 | zalizedata/us-grid-hourly-power-plants-dataset [M-generators] | viewer+footers | points: catalog says 10M-100M, metadata gives 53.71k points (26.86k rows) |
| 230 | zalizedata/us-grid-hourly-power-plants-dataset [M-plant_month_generation] | viewer+footers | points: catalog says 10M-100M, metadata gives 433.92k points (216.96k rows) / frequency: catalog says 1H, metadata says 1M (stated in the config / folder name (M-plant_month_generation)) |
| 230 | zalizedata/us-grid-hourly-power-plants-dataset [M-plant_profiles] | viewer+footers | points: catalog says 10M-100M, metadata gives 96.79k points (16.13k rows) |
| 230 | zalizedata/us-grid-hourly-power-plants-dataset [M-plants] | viewer+footers | points: catalog says 10M-100M, metadata gives 32.26k points (16.13k rows) |
| 230 | zalizedata/us-grid-hourly-power-plants-dataset [M-state_electricity_prices] | viewer+footers | points: catalog says 10M-100M, metadata gives 2.89k points (2.89k rows) |
| 230 | zalizedata/us-grid-hourly-power-plants-dataset [M-state_generation_monthly] | viewer+footers | points: catalog says 10M-100M, metadata gives 9.57k points (9.57k rows) / frequency: catalog says 1H, metadata says 1M (stated in the config / folder name (M-state_generation_monthly)) |
| 230 | zalizedata/us-grid-hourly-power-plants-dataset [M-utilities] | viewer+footers | points: catalog says 10M-100M, metadata gives 6.64k points (6.64k rows) |
| 230 | zalizedata/us-grid-hourly-power-plants-dataset [S-ba_fuel_generation_hourly] | viewer+footers | points: catalog says 10M-100M, metadata gives 70.08k points (70.08k rows) |
| 230 | zalizedata/us-grid-hourly-power-plants-dataset [S-ba_hourly] | viewer+footers | points: catalog says 10M-100M, metadata gives 35.04k points (8.76k rows) |
| 230 | zalizedata/us-grid-hourly-power-plants-dataset [S-ba_interchange_hourly] | viewer+footers | points: catalog says 10M-100M, metadata gives 95.83k points (95.83k rows) |
| 230 | zalizedata/us-grid-hourly-power-plants-dataset [S-ba_subregion_hourly] | viewer+footers | points: catalog says 10M-100M, metadata gives 34.84k points (34.84k rows) |
| 230 | zalizedata/us-grid-hourly-power-plants-dataset [S-eba_gap_report] | viewer+footers | points: catalog says 10M-100M, metadata gives 3 points (1 rows) |
| 231 | alinurdin/time-series-benchmark [eia_hourly] | viewer+footers | licence: catalog says US public domain, the card says other |
| 232 | alinurdin/time-series-benchmark [opsd_load] | viewer+footers | licence: catalog says per provider (OPSD/ENTSO-E), the card says other |
| 233 | autogluon/chronos_datasets [monash_australian_electricity] | viewer+footers | licence: catalog says per-dataset (card: check ds.info); repo not gated, the card says other |
| 234 | autogluon/chronos_datasets [solar, solar_1h] | viewer+footers | licence: catalog says per-dataset (card: check ds.info); repo not gated, the card says other |
| 235 | autogluon/chronos_datasets_extra [spanish_energy_and_weather] | files_only | empty: no data files in the scope / subset text not matched on the Hub: spanish_energy_and_weather / the dataset viewer does not work for this repo |
| 236 | DAG-UPB/TS-Arena-Archive [default] | footers | the dataset viewer does not work for this repo |
| 237 | EDS-lab/electricity-demand [demand+weather+metadata] | gated | subset text not matched on the Hub: demand+weather+metadata / gated, no access / size: catalog says 2.43 GB, the Hub file listing sums to 1.10 GB / 3 footer(s) not readable, first: data/demand.parquet: HTTP 403 GatedRepo Access to dataset EDS-lab/electricity-demand is restricted and you are not in the authorized list. Visit https://hugging |
| 237 | EDS-lab/electricity-demand [demand] | gated | gated, no access / size: catalog says 2.43 GB, the Hub file listing sums to 1.10 GB / 4 footer(s) not readable, first: data/demand.parquet: HTTP 403 GatedRepo Access to dataset EDS-lab/electricity-demand is restricted and you are not in the authorized list. Visit https://hugging |
| 237 | EDS-lab/electricity-demand [metadata] | gated | gated, no access / size: catalog says 2.43 GB, the Hub file listing sums to 1.10 GB / 5 footer(s) not readable, first: data/demand.parquet: HTTP 403 GatedRepo Access to dataset EDS-lab/electricity-demand is restricted and you are not in the authorized list. Visit https://hugging |
| 237 | EDS-lab/electricity-demand [weather] | gated | gated, no access / size: catalog says 2.43 GB, the Hub file listing sums to 1.10 GB / 6 footer(s) not readable, first: data/demand.parquet: HTTP 403 GatedRepo Access to dataset EDS-lab/electricity-demand is restricted and you are not in the authorized list. Visit https://hugging |
| 238 | EDS-lab/pv-generation [generation+weather+metadata] | gated | subset text not matched on the Hub: generation+weather+metadata / gated, no access / size: catalog says 1.67 GB, the Hub file listing sums to 870.56 MB / 3 footer(s) not readable, first: data/generation.parquet: HTTP 403 GatedRepo Access to dataset EDS-lab/pv-generation is restricted and you are not in the authorized list. Visit https://huggingf |
| 238 | EDS-lab/pv-generation [generation] | gated | gated, no access / size: catalog says 1.67 GB, the Hub file listing sums to 870.56 MB / 4 footer(s) not readable, first: data/generation.parquet: HTTP 403 GatedRepo Access to dataset EDS-lab/pv-generation is restricted and you are not in the authorized list. Visit https://huggingf |
| 238 | EDS-lab/pv-generation [metadata] | gated | gated, no access / size: catalog says 1.67 GB, the Hub file listing sums to 870.56 MB / 5 footer(s) not readable, first: data/generation.parquet: HTTP 403 GatedRepo Access to dataset EDS-lab/pv-generation is restricted and you are not in the authorized list. Visit https://huggingf |
| 238 | EDS-lab/pv-generation [weather] | gated | gated, no access / size: catalog says 1.67 GB, the Hub file listing sums to 870.56 MB / 6 footer(s) not readable, first: data/generation.parquet: HTTP 403 GatedRepo Access to dataset EDS-lab/pv-generation is restricted and you are not in the authorized list. Visit https://huggingf |
| 239 | fidel-ts/Germany_Renewable_Energy_Grid | footers | the dataset viewer does not work for this repo |
| 240 | hsilvosa/entsoe-observatory [8 tables] | viewer+footers | subset text not matched on the Hub: 8 tables / licence: catalog says ENTSO-E reuse with attribution, the card says other |
| 240 | hsilvosa/entsoe-observatory [cross_zone_spread_hourly] | viewer+footers | licence: catalog says ENTSO-E reuse with attribution, the card says other |
| 240 | hsilvosa/entsoe-observatory [dq_scorecard] | viewer+footers | licence: catalog says ENTSO-E reuse with attribution, the card says other / points: catalog says 33,611,557 rows total, metadata gives 2.65M points (378.75k rows) |
| 240 | hsilvosa/entsoe-observatory [dq_scorecard_summary] | viewer+footers | licence: catalog says ENTSO-E reuse with attribution, the card says other / points: catalog says 33,611,557 rows total, metadata gives 465 points (93 rows) |
| 240 | hsilvosa/entsoe-observatory [load_hourly] | viewer+footers | licence: catalog says ENTSO-E reuse with attribution, the card says other |
| 240 | hsilvosa/entsoe-observatory [load_native] | viewer+footers | licence: catalog says ENTSO-E reuse with attribution, the card says other |
| 240 | hsilvosa/entsoe-observatory [price_hourly] | viewer+footers | licence: catalog says ENTSO-E reuse with attribution, the card says other |
| 240 | hsilvosa/entsoe-observatory [price_native] | viewer+footers | licence: catalog says ENTSO-E reuse with attribution, the card says other |
| 240 | hsilvosa/entsoe-observatory [weather_hourly] | viewer+footers | licence: catalog says ENTSO-E reuse with attribution, the card says other |
| 240 | hsilvosa/entsoe-observatory [zone_daily] | viewer+footers | licence: catalog says ENTSO-E reuse with attribution, the card says other / points: catalog says 33,611,557 rows total, metadata gives 2.34M points (194.99k rows) |
| 242 | nju-zhangsq/RMISC [OPSD] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 22899412, the footers give 1052428848 |
| 243 | OpenSynth/AI4OPT-TS4PS-RTE7000 [tables (lines, generators, loads, ...)] | viewer+footers | subset text not matched on the Hub: tables |
| 244 | OptimalSolution/databank-ember [19 tables (generation monthly/yearly, prices, capacity, ...)] | viewer+footers | subset text not matched on the Hub: 19 tables |
| 244 | OptimalSolution/databank-ember [capacity_wind_solar_monthly] | viewer+footers | points: catalog says ~7.39M rows, metadata gives 52.91k points (52.91k rows) |
| 244 | OptimalSolution/databank-ember [cleantech_exports_monthly] | viewer+footers | points: catalog says ~7.39M rows, metadata gives 293.80k points (293.80k rows) |
| 244 | OptimalSolution/databank-ember [generation_yearly] | viewer+footers | points: catalog says ~7.39M rows, metadata gives 543.30k points (543.30k rows) |
| 244 | OptimalSolution/databank-ember [india_monthly] | viewer+footers | points: catalog says ~7.39M rows, metadata gives 234.69k points (234.69k rows) |
| 244 | OptimalSolution/databank-ember [india_yearly] | viewer+footers | points: catalog says ~7.39M rows, metadata gives 17.05k points (17.05k rows) |
| 244 | OptimalSolution/databank-ember [interconnection_country_profile] | viewer+footers | points: catalog says ~7.39M rows, metadata gives 17.30k points (8.65k rows) |
| 244 | OptimalSolution/databank-ember [interconnection_flows] | viewer+footers | points: catalog says ~7.39M rows, metadata gives 45.00k points (22.50k rows) |
| 244 | OptimalSolution/databank-ember [interconnection_import_potential] | viewer+footers | points: catalog says ~7.39M rows, metadata gives 736 points (368 rows) |
| 244 | OptimalSolution/databank-ember [interconnection_ntc] | viewer+footers | points: catalog says ~7.39M rows, metadata gives 1.70k points (1.70k rows) |
| 244 | OptimalSolution/databank-ember [interconnection_peak_demand] | viewer+footers | points: catalog says ~7.39M rows, metadata gives 134 points (134 rows) |
| 244 | OptimalSolution/databank-ember [price_daily] | viewer+footers | points: catalog says ~7.39M rows, metadata gives 121.42k points (121.42k rows) |
| 244 | OptimalSolution/databank-ember [price_monthly] | viewer+footers | points: catalog says ~7.39M rows, metadata gives 4.00k points (4.00k rows) |
| 244 | OptimalSolution/databank-ember [targets_2030] | viewer+footers | points: catalog says ~7.39M rows, metadata gives 4.13k points (2.06k rows) |
| 244 | OptimalSolution/databank-ember [us_states_yearly] | viewer+footers | points: catalog says ~7.39M rows, metadata gives 86.55k points (86.55k rows) |
| 245 | qingshufan/GA-EVLRU | files_only | format: catalog says parquet, the scope has no parquet file (csv) / size: catalog says 0.06 GB, the Hub file listing sums to 40.06 MB / the dataset viewer does not work for this repo |
| 246 | Salesforce/lotsa_data [australian_electricity_demand] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 247 | THULab/m2oe2_timeseries [8 tables (building_train, building weather, household consumption, residential power usage + weather, solar + weather, Spanish energy + weather)] | files_only | subset text not matched on the Hub: 8 tables / the dataset viewer does not work for this repo |
| 248 | YlanHARRAULT/rte-eco2mix-regional-2025 | viewer+footers | frequency: catalog says 30min, metadata says 6h (implied by span and series length (span / (length - 1) = 2.151e+04 s)) |
| 250 | ajto/GoldenCheetah | viewer+footers | size: catalog says 0.33 GB, the Hub file listing sums to 165.14 MB |
| 250 | ajto/GoldenCheetah [activities] | viewer+footers | size: catalog says 0.33 GB, the Hub file listing sums to 165.14 MB |
| 250 | ajto/GoldenCheetah [athletes] | viewer+footers | size: catalog says 0.33 GB, the Hub file listing sums to 165.14 MB |
| 250 | ajto/GoldenCheetah [default] | viewer+footers | size: catalog says 0.33 GB, the Hub file listing sums to 165.14 MB |
| 252 | CRUISEResearchGroup/CGM-JEPA-Pretraining | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 12.35 MB |
| 259 | HHS-Official/nndss-weekly-data | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) |
| 262 | HHS-Official/weekly-provisional-counts-of-deaths-by-state-and-s | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) |
| 263 | HHS-Official/weekly-rates-of-laboratory-confirmed-rsv-hospitali | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) / size: catalog says 0.02 GB, the Hub file listing sums to 13.95 MB |
| 265 | judicialmind/HMIS-Bench | gated | gated, no access / 50 footer(s) not readable, first: forecasts/chronos2/2021-08-01.parquet: HTTP 403 GatedRepo Access to dataset judicialmind/HMIS-Bench is restricted and you are not in the authorized list. Visit  |
| 265 | judicialmind/HMIS-Bench [all] | gated | gated, no access / 52 footer(s) not readable, first: forecasts/chronos2/2021-08-01.parquet: HTTP 403 GatedRepo Access to dataset judicialmind/HMIS-Bench is restricted and you are not in the authorized list. Visit  |
| 265 | judicialmind/HMIS-Bench [bench] | gated | gated, no access / 51 footer(s) not readable, first: forecasts/chronos2/2021-08-01.parquet: HTTP 403 GatedRepo Access to dataset judicialmind/HMIS-Bench is restricted and you are not in the authorized list. Visit  |
| 265 | judicialmind/HMIS-Bench [district_extended] | gated | gated, no access / 53 footer(s) not readable, first: forecasts/chronos2/2021-08-01.parquet: HTTP 403 GatedRepo Access to dataset judicialmind/HMIS-Bench is restricted and you are not in the authorized list. Visit  |
| 266 | judicialmind/IndiaDiseaseWatch [panel.parquet] | gated | gated, no access / 1 footer(s) not readable, first: data/panel.parquet: HTTP 403 GatedRepo Access to dataset judicialmind/IndiaDiseaseWatch is restricted and you are not in the authorized list. Visit https://hugg |
| 270 | nju-zhangsq/RMISC [HungarianChickenpoxCases] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 271 | Ohja-ai/rki-grippe-web-wochenbericht-2026-02-06 | viewer+footers | format: catalog says parquet, the scope has no parquet file (tsv) |
| 272 | pedrolucassantanaf/dengue-tcc2-data | gated | gated, no access / 50 footer(s) not readable, first: dashboard/data/predictions_A_inmet_bruto.parquet: HTTP 403 GatedRepo Access to dataset pedrolucassantanaf/dengue-tcc2-data is restricted and you are not in the  |
| 273 | rricc22/endomondo-hr-prediction-v2 | viewer+footers | size: catalog says 2.09 GB, the Hub file listing sums to 677.63 MB |
| 275 | Salesforce/lotsa_data [cdc_fluview_ilinet] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 276 | Salesforce/lotsa_data [cdc_fluview_who_nrevss] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 277 | Salesforce/lotsa_data [project_tycho] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 278 | thuml/UTSD [TDBrain] | viewer+footers | subset text not matched on the Hub: TDBrain / size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB / points: catalog says 72.30M, metadata gives 2.50B points (867.26k rows) |
| 278 | thuml/UTSD [UTSD-12G] | viewer+footers | size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB / points: catalog says 72.30M, metadata gives 965.34M points (289.56k rows) |
| 278 | thuml/UTSD [UTSD-1G] | viewer+footers | size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB |
| 278 | thuml/UTSD [UTSD-2G] | viewer+footers | size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB |
| 278 | thuml/UTSD [UTSD-4G] | files_only | size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB |
| 278 | thuml/UTSD [default] | viewer+footers | size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB / points: catalog says 72.30M, metadata gives 1.25B points (433.63k rows) |
| 279 | Uvathe/wellborn-pregnancy-wearable | viewer+footers | format: catalog says parquet, the scope has no parquet file (mtn) |
| 280 | DragonbornRay/FluDataset [cdc_level_thresholds] | viewer+footers | points: catalog says 100K-1M rows, metadata gives 684 points (171 rows) |
| 280 | DragonbornRay/FluDataset [cities] | viewer+footers | points: catalog says 100K-1M rows, metadata gives 708 points (236 rows) |
| 280 | DragonbornRay/FluDataset [holidays] | viewer+footers | points: catalog says 100K-1M rows, metadata gives 4.96k points (992 rows) |
| 280 | DragonbornRay/FluDataset [vaccine_state_season] | viewer+footers | points: catalog says 100K-1M rows, metadata gives 865 points (865 rows) |
| 286 | thuml/UTSD [BIDMC32HR / IEEEPPG / MotorImagery / SelfRegulationSCP1 / SelfRegulationSCP2 / PigArtPressure / PigCVP / AtrialFibrillation] | viewer+footers | subset text not matched on the Hub: BIDMC32HR; IEEEPPG; MotorImagery; SelfRegulationSCP1; SelfRegulationSCP2; PigArtPressure; PigCVP; AtrialFibrillation / size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB / points: catalog says ~159M total, metadata gives 2.50B points (867.26k rows) |
| 286 | thuml/UTSD [UTSD-12G] | viewer+footers | size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB |
| 286 | thuml/UTSD [UTSD-1G] | viewer+footers | size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB |
| 286 | thuml/UTSD [UTSD-2G] | viewer+footers | size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB |
| 286 | thuml/UTSD [UTSD-4G] | files_only | size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB |
| 286 | thuml/UTSD [default] | viewer+footers | size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB |
| 290 | cjc0013/ouroboros-ooi-ctd-benchmark | viewer+footers | size: catalog says 0.19 GB, the Hub file listing sums to 108.20 MB |
| 290 | cjc0013/ouroboros-ooi-ctd-benchmark [benchmark_windows] | viewer+footers | size: catalog says 0.19 GB, the Hub file listing sums to 108.20 MB |
| 290 | cjc0013/ouroboros-ooi-ctd-benchmark [entities] | viewer+footers | size: catalog says 0.19 GB, the Hub file listing sums to 108.20 MB |
| 290 | cjc0013/ouroboros-ooi-ctd-benchmark [observations] | viewer+footers | size: catalog says 0.19 GB, the Hub file listing sums to 108.20 MB |
| 290 | cjc0013/ouroboros-ooi-ctd-benchmark [observations_hourly] | viewer+footers | size: catalog says 0.19 GB, the Hub file listing sums to 108.20 MB / frequency: catalog says ~1min, metadata says 1h (stated in the config / folder name (observations_hourly)) |
| 290 | cjc0013/ouroboros-ooi-ctd-benchmark [provenance_edges] | viewer+footers | size: catalog says 0.19 GB, the Hub file listing sums to 108.20 MB |
| 290 | cjc0013/ouroboros-ooi-ctd-benchmark [quality_events] | viewer+footers | size: catalog says 0.19 GB, the Hub file listing sums to 108.20 MB / points: catalog says 1M-10M rows, metadata gives 674 points (337 rows) |
| 291 | CMU-OSPO/Data_for_Extended-Range_Water_Demand_Forecasts_Driven_by_Subseasonal-to-Seasonal [dsci_williamsoncounty_19800101_20250101] | viewer+footers | points: catalog says 100K-1M, metadata gives 1.30k points (1.30k rows) |
| 291 | CMU-OSPO/Data_for_Extended-Range_Water_Demand_Forecasts_Driven_by_Subseasonal-to-Seasonal [population_in_georgetown] | viewer+footers | points: catalog says 100K-1M, metadata gives 34 points (34 rows) |
| 291 | CMU-OSPO/Data_for_Extended-Range_Water_Demand_Forecasts_Driven_by_Subseasonal-to-Seasonal [population_in_round_rock] | viewer+footers | points: catalog says 100K-1M, metadata gives 34 points (34 rows) |
| 292 | copernicus-argentina/estado-hidrico-argentina-2001-2026 | files_only | format: catalog says parquet, the scope has no parquet file (csv, csv.gz) / size: catalog says 0.02 GB, the Hub file listing sums to 9.42 MB / the dataset viewer does not work for this repo |
| 293 | copernicus-argentina/lagunas-pampeanas-2018-2026 | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv, json) |
| 293 | copernicus-argentina/lagunas-pampeanas-2018-2026 [fechas_rechazadas] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) |
| 293 | copernicus-argentina/lagunas-pampeanas-2018-2026 [lagunas] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) |
| 293 | copernicus-argentina/lagunas-pampeanas-2018-2026 [serie_mensual] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) |
| 293 | copernicus-argentina/lagunas-pampeanas-2018-2026 [serie_por_fecha] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) |
| 294 | copernicus-argentina/parana-alturas-historicas-1900-2026 | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv, json) |
| 294 | copernicus-argentina/parana-alturas-historicas-1900-2026 [alturas_diarias] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) |
| 294 | copernicus-argentina/parana-alturas-historicas-1900-2026 [crecida_1983] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) |
| 294 | copernicus-argentina/parana-alturas-historicas-1900-2026 [escalas] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) / points: catalog says 100K-1M, metadata gives 88 points (22 rows) |
| 294 | copernicus-argentina/parana-alturas-historicas-1900-2026 [maximos_anuales] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) / points: catalog says 100K-1M, metadata gives 9.90k points (1.98k rows) |
| 294 | copernicus-argentina/parana-alturas-historicas-1900-2026 [picos_1983] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) / points: catalog says 100K-1M, metadata gives 220 points (22 rows) |
| 294 | copernicus-argentina/parana-alturas-historicas-1900-2026 [ranking_crecidas] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) / points: catalog says 100K-1M, metadata gives 1.32k points (330 rows) |
| 295 | earino/noaa-tide-flooding | viewer+footers | format: catalog says csv/json, the scope holds csv, json / size: catalog says 0.45 GB, the Hub file listing sums to 196.10 MB |
| 295 | earino/noaa-tide-flooding [station_disjoint] | viewer+footers | format: catalog says csv/json, the scope holds csv / size: catalog says 0.45 GB, the Hub file listing sums to 196.10 MB |
| 295 | earino/noaa-tide-flooding [temporal] | viewer+footers | format: catalog says csv/json, the scope holds csv / size: catalog says 0.45 GB, the Hub file listing sums to 196.10 MB |
| 296 | Emulated-Inc/sci-arctic-sea-ice-forecast-later-years | files_only | format: catalog says parquet, the scope has no parquet file (npy, npz) / the dataset viewer does not work for this repo |
| 297 | Emulated-Inc/sci-lake-temperature-profiles-unmonitored | files_only | format: catalog says parquet, the scope has no parquet file (tar) / the dataset viewer does not work for this repo |
| 299 | jotone/pcfvg-weatherstations [stations] | viewer+footers | points: catalog says 100M-1B rows, metadata gives 730 points (365 rows) |
| 300 | judicialmind/BharatWater-Bench | gated | gated, no access / 50 footer(s) not readable, first: data/rain_daily/series.parquet: HTTP 403 GatedRepo Access to dataset judicialmind/BharatWater-Bench is restricted and you are not in the authorized list. Visit  |
| 300 | judicialmind/BharatWater-Bench [rain_daily] | gated | gated, no access / 53 footer(s) not readable, first: data/rain_daily/series.parquet: HTTP 403 GatedRepo Access to dataset judicialmind/BharatWater-Bench is restricted and you are not in the authorized list. Visit  |
| 300 | judicialmind/BharatWater-Bench [rain_weekly] | gated | gated, no access / 55 footer(s) not readable, first: data/rain_daily/series.parquet: HTTP 403 GatedRepo Access to dataset judicialmind/BharatWater-Bench is restricted and you are not in the authorized list. Visit  |
| 300 | judicialmind/BharatWater-Bench [reservoir_pct] | gated | gated, no access / 52 footer(s) not readable, first: data/rain_daily/series.parquet: HTTP 403 GatedRepo Access to dataset judicialmind/BharatWater-Bench is restricted and you are not in the authorized list. Visit  |
| 300 | judicialmind/BharatWater-Bench [reservoir_storage] | gated | gated, no access / 51 footer(s) not readable, first: data/rain_daily/series.parquet: HTTP 403 GatedRepo Access to dataset judicialmind/BharatWater-Bench is restricted and you are not in the authorized list. Visit  |
| 300 | judicialmind/BharatWater-Bench [soil_moisture] | gated | gated, no access / 54 footer(s) not readable, first: data/rain_daily/series.parquet: HTTP 403 GatedRepo Access to dataset judicialmind/BharatWater-Bench is restricted and you are not in the authorized list. Visit  |
| 301 | judicialmind/BharatWater-Data | gated | gated, no access / 3 footer(s) not readable, first: data/rainfall_district_daily.parquet: HTTP 403 GatedRepo Access to dataset judicialmind/BharatWater-Data is restricted and you are not in the authorized list. V |
| 301 | judicialmind/BharatWater-Data [rainfall_district_daily] | gated | gated, no access / 5 footer(s) not readable, first: data/rainfall_district_daily.parquet: HTTP 403 GatedRepo Access to dataset judicialmind/BharatWater-Data is restricted and you are not in the authorized list. V |
| 301 | judicialmind/BharatWater-Data [reservoirs_daily] | gated | gated, no access / 4 footer(s) not readable, first: data/rainfall_district_daily.parquet: HTTP 403 GatedRepo Access to dataset judicialmind/BharatWater-Data is restricted and you are not in the authorized list. V |
| 301 | judicialmind/BharatWater-Data [soil_moisture_district] | gated | gated, no access / 6 footer(s) not readable, first: data/rainfall_district_daily.parquet: HTTP 403 GatedRepo Access to dataset judicialmind/BharatWater-Data is restricted and you are not in the authorized list. V |
| 304 | phanerozoic/noaa-ww3-multi1-points [wmo] | files_only | the dataset viewer does not work for this repo |
| 306 | Qdrant/NOAA-Buoy | footers | the dataset viewer does not work for this repo |
| 309 | xairon/piezo-embedding-benchmark | viewer+footers | size: catalog says 0.98 GB, the Hub file listing sums to 591.16 MB |
| 311 | Emulated-Inc/sci-streamflow-ungauged-new-country | files_only | format: catalog says parquet, the scope has no parquet file (tar) / the dataset viewer does not work for this repo |
| 313 | Rekin226/aquascope-gauges [observation bundles (USGS, UK EA, BOM, Hub'Eau, PEGELONLINE, ...)] | footers | subset text not matched on the Hub: observation bundles / licence: catalog says per source (CC-BY-4.0, CC-BY-SA-4.0, OGL-UK-3.0, US-PD; some unknown), the card says other / size: catalog says 6.55 GB, the Hub file listing sums to 3.56 GB / the dataset viewer does not work for this repo |
| 314 | Amgharr/FEMTO-ST_DATASET | viewer+footers | 2 footer(s) not readable, first: features_test.parquet: footer of 9954530 bytes is above the limit of 8388608 |
| 315 | Arailym-tleubayeva/NK-Oil-Well-Sensor-Monitoring-v2 | viewer+footers | frequency: catalog says H, metadata says 30min (implied by span and series length (span / (length - 1) = 1826 s)) |
| 316 | Arailym-tleubayeva/NK-Oil-Well-Sensor-Monitoring-v2 [train/test] | viewer+footers | subset text not matched on the Hub: train/test / format: catalog says csv/parquet, the scope holds csv |
| 317 | CDuong04/NGAFID-LOCI-GATS-Data | files_only | size: catalog says 0.03 GB, the Hub file listing sums to 39.63 GB / the dataset viewer does not work for this repo |
| 319 | Emulated-Inc/sci-vlt-turbulence-from-ao-telemetry | files_only | format: catalog says parquet, the scope has no parquet file (sha256, tar) / the dataset viewer does not work for this repo |
| 320 | HenryYHW/CANTraj | gated | gated, no access / licence: catalog says DriveDNA research licence (research only), the card says other / 50 footer(s) not readable, first: benchmark/manifests/availability_wide.parquet: HTTP 403 GatedRepo Access to dataset HenryYHW/CANTraj is restricted and you are not in the authorized list. Visit |
| 320 | HenryYHW/CANTraj [benchmark_episodes] | gated | gated, no access / format: catalog says csv, the scope holds parquet / licence: catalog says DriveDNA research licence (research only), the card says other / 251 footer(s) not readable, first: benchmark/manifests/availability_wide.parquet: HTTP 403 GatedRepo Access to dataset HenryYHW/CANTraj is restricted and you are not in the authorized list. Visit |
| 320 | HenryYHW/CANTraj [benchmark_map_episodes] | gated | gated, no access / format: catalog says csv, the scope holds parquet / licence: catalog says DriveDNA research licence (research only), the card says other / 254 footer(s) not readable, first: benchmark/manifests/availability_wide.parquet: HTTP 403 GatedRepo Access to dataset HenryYHW/CANTraj is restricted and you are not in the authorized list. Visit |
| 320 | HenryYHW/CANTraj [benchmark_scores] | gated | gated, no access / format: catalog says csv, the scope holds parquet / licence: catalog says DriveDNA research licence (research only), the card says other / 252 footer(s) not readable, first: benchmark/manifests/availability_wide.parquet: HTTP 403 GatedRepo Access to dataset HenryYHW/CANTraj is restricted and you are not in the authorized list. Visit |
| 320 | HenryYHW/CANTraj [benchmark_scores_v11] | gated | gated, no access / format: catalog says csv, the scope holds parquet / licence: catalog says DriveDNA research licence (research only), the card says other / 253 footer(s) not readable, first: benchmark/manifests/availability_wide.parquet: HTTP 403 GatedRepo Access to dataset HenryYHW/CANTraj is restricted and you are not in the authorized list. Visit |
| 320 | HenryYHW/CANTraj [events_calib] | gated | gated, no access / format: catalog says csv, the scope holds parquet / licence: catalog says DriveDNA research licence (research only), the card says other / 250 footer(s) not readable, first: benchmark/manifests/availability_wide.parquet: HTTP 403 GatedRepo Access to dataset HenryYHW/CANTraj is restricted and you are not in the authorized list. Visit |
| 320 | HenryYHW/CANTraj [events_carstate] | gated | gated, no access / format: catalog says csv, the scope holds parquet / licence: catalog says DriveDNA research licence (research only), the card says other / 100 footer(s) not readable, first: benchmark/manifests/availability_wide.parquet: HTTP 403 GatedRepo Access to dataset HenryYHW/CANTraj is restricted and you are not in the authorized list. Visit |
| 320 | HenryYHW/CANTraj [events_controls] | gated | gated, no access / format: catalog says csv, the scope holds parquet / licence: catalog says DriveDNA research licence (research only), the card says other / 150 footer(s) not readable, first: benchmark/manifests/availability_wide.parquet: HTTP 403 GatedRepo Access to dataset HenryYHW/CANTraj is restricted and you are not in the authorized list. Visit |
| 320 | HenryYHW/CANTraj [events_imu] | gated | gated, no access / format: catalog says csv, the scope holds parquet / licence: catalog says DriveDNA research licence (research only), the card says other / 200 footer(s) not readable, first: benchmark/manifests/availability_wide.parquet: HTTP 403 GatedRepo Access to dataset HenryYHW/CANTraj is restricted and you are not in the authorized list. Visit |
| 320 | HenryYHW/CANTraj [events_manifest] | gated | gated, no access / licence: catalog says DriveDNA research licence (research only), the card says other / 250 footer(s) not readable, first: benchmark/manifests/availability_wide.parquet: HTTP 403 GatedRepo Access to dataset HenryYHW/CANTraj is restricted and you are not in the authorized list. Visit |
| 320 | HenryYHW/CANTraj [gnss] | gated | gated, no access / licence: catalog says DriveDNA research licence (research only), the card says other / 50 footer(s) not readable, first: benchmark/manifests/availability_wide.parquet: HTTP 403 GatedRepo Access to dataset HenryYHW/CANTraj is restricted and you are not in the authorized list. Visit |
| 320 | HenryYHW/CANTraj [gps] | gated | gated, no access / licence: catalog says DriveDNA research licence (research only), the card says other / 50 footer(s) not readable, first: benchmark/manifests/availability_wide.parquet: HTTP 403 GatedRepo Access to dataset HenryYHW/CANTraj is restricted and you are not in the authorized list. Visit |
| 320 | HenryYHW/CANTraj [kalman] | gated | gated, no access / licence: catalog says DriveDNA research licence (research only), the card says other / 50 footer(s) not readable, first: benchmark/manifests/availability_wide.parquet: HTTP 403 GatedRepo Access to dataset HenryYHW/CANTraj is restricted and you are not in the authorized list. Visit |
| 320 | HenryYHW/CANTraj [manifest] | gated | gated, no access / licence: catalog says DriveDNA research licence (research only), the card says other / 50 footer(s) not readable, first: benchmark/manifests/availability_wide.parquet: HTTP 403 GatedRepo Access to dataset HenryYHW/CANTraj is restricted and you are not in the authorized list. Visit |
| 321 | jwsouza13/eVED | viewer+footers | size: catalog says 1.38 GB, the Hub file listing sums to 690.41 MB |
| 323 | mihnathul/celljar | footers | size: catalog says 8.14 GB, the Hub file listing sums to 4.17 GB / the dataset viewer does not work for this repo |
| 324 | nju-zhangsq/RMISC [BatteryRUL] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 135576, the footers give 105448 |
| 325 | nju-zhangsq/RMISC [BoschCNC] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 326 | nju-zhangsq/RMISC [FrothFlotation] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 327 | nju-zhangsq/RMISC [GasPipeline] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 1376110, the footers give 1100888 |
| 328 | nju-zhangsq/RMISC [GasSensorDynamic] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 329 | nju-zhangsq/RMISC [GasSensorTemperature] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 330 | nju-zhangsq/RMISC [Metropt3] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 331 | nju-zhangsq/RMISC [MotorTemperature] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 332 | nju-zhangsq/RMISC [OilWell] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 244525350, the footers give 355944337 |
| 333 | nju-zhangsq/RMISC [PUMP] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 334 | nju-zhangsq/RMISC [SWAT] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 335 | nju-zhangsq/RMISC [WADI] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 336 | Samlzy/MCC5-THU-Motor | files_only | size: catalog says 19.18 GB, the Hub file listing sums to 12.72 GB / the dataset viewer does not work for this repo |
| 337 | thuml/UTSD [SensorData] | viewer+footers | subset text not matched on the Hub: SensorData / size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB / points: catalog says 165.4M, metadata gives 2.50B points (867.26k rows) |
| 337 | thuml/UTSD [UTSD-12G] | viewer+footers | size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB |
| 337 | thuml/UTSD [UTSD-1G] | viewer+footers | size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB |
| 337 | thuml/UTSD [UTSD-2G] | viewer+footers | size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB |
| 337 | thuml/UTSD [UTSD-4G] | files_only | size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB |
| 337 | thuml/UTSD [default] | viewer+footers | size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB |
| 338 | xijiazhao/casemotordataset [dated folders] | files_only | subset text not matched on the Hub: dated folders / the dataset viewer does not work for this repo |
| 340 | cognitedata/IND-Eval | viewer+footers | points: catalog says 100K-1M windows, metadata gives 2.36B points (72.16M rows) |
| 340 | cognitedata/IND-Eval [metropt3] | viewer+footers | points: catalog says 100K-1M windows, metadata gives 10.01M points (2.85k rows) |
| 340 | cognitedata/IND-Eval [neats] | viewer+footers | points: catalog says 100K-1M windows, metadata gives 48.31M points (9.93k rows) |
| 340 | cognitedata/IND-Eval [solar] | viewer+footers | points: catalog says 100K-1M windows, metadata gives 42.36M points (8.90k rows) |
| 341 | Forgis/FactoryNet [factorywave_*, voraus_*, aursad_*, cnc_*] | viewer+footers | licence: catalog says MIT (voraus CC-BY-NC-SA-4.0; aursad CC-BY-4.0), the card says other |
| 343 | scalytics/smartmeterdata [smartmeter_*.csv] | files_only | the dataset viewer does not work for this repo |
| 344 | Adrien97/velib-monitoring-historical | viewer+footers | size: catalog says 0.82 GB, the Hub file listing sums to 23.92 MB |
| 345 | aiacademy-kg/bishkek-transport [stops] | viewer+footers | points: catalog says 100M-1B rows, metadata gives 1.50k points (1.50k rows) |
| 345 | aiacademy-kg/bishkek-transport [telematics_devices] | viewer+footers | points: catalog says 100M-1B rows, metadata gives 418 points (209 rows) |
| 345 | aiacademy-kg/bishkek-transport [telematics_route_shapes] | viewer+footers | points: catalog says 100M-1B rows, metadata gives 75.70k points (18.93k rows) |
| 345 | aiacademy-kg/bishkek-transport [telematics_route_stops] | viewer+footers | points: catalog says 100M-1B rows, metadata gives 5.02k points (1.00k rows) |
| 345 | aiacademy-kg/bishkek-transport [telematics_routes] | viewer+footers | points: catalog says 100M-1B rows, metadata gives 77 points (11 rows) |
| 345 | aiacademy-kg/bishkek-transport [telematics_waybills] | viewer+footers | points: catalog says 100M-1B rows, metadata gives 160.10k points (20.01k rows) |
| 346 | Algo0o/flight_delay [default] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv, db) |
| 347 | AmandaHattaway/bike-lane-lakehouse [dim_boston_count_locations] | viewer+footers | points: catalog says 1M-10M, metadata gives 91 points (91 rows) |
| 347 | AmandaHattaway/bike-lane-lakehouse [dim_nyc_protected_lanes] | viewer+footers | points: catalog says 1M-10M, metadata gives 8.60k points (8.60k rows) |
| 347 | AmandaHattaway/bike-lane-lakehouse [fct_boston_bike_counts_daily] | viewer+footers | points: catalog says 1M-10M, metadata gives 17.71k points (2.95k rows) |
| 347 | AmandaHattaway/bike-lane-lakehouse [fct_nyc_new_lane_safety] | viewer+footers | points: catalog says 1M-10M, metadata gives 30.44k points (4.35k rows) |
| 349 | Arimancy/bts-flights-weather [airports_reference] | viewer+footers | points: catalog says 10M-100M, metadata gives 1.16k points (386 rows) |
| 350 | autogluon/chronos_datasets [mexico_city_bikes] | viewer+footers | licence: catalog says per-dataset (card: check ds.info); repo not gated, the card says other |
| 351 | autogluon/chronos_datasets [monash_pedestrian_counts] | viewer+footers | licence: catalog says per-dataset (card: check ds.info); repo not gated, the card says other |
| 352 | autogluon/chronos_datasets [monash_rideshare] | viewer+footers | licence: catalog says per-dataset (card: check ds.info); repo not gated, the card says other |
| 353 | autogluon/chronos_datasets [monash_traffic] | viewer+footers | licence: catalog says per-dataset (card: check ds.info); repo not gated, the card says other |
| 354 | autogluon/chronos_datasets [taxi_30min, taxi_1h] | viewer+footers | licence: catalog says per-dataset (card: check ds.info); repo not gated, the card says other |
| 355 | autogluon/chronos_datasets [uber_tlc_hourly, uber_tlc_daily] | viewer+footers | licence: catalog says per-dataset (card: check ds.info); repo not gated, the card says other |
| 357 | danipaez/velib_tft [default] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) |
| 358 | digitalhen/nyc-subway-realtime | gated | gated, no access / 50 footer(s) not readable, first: data/alerts_unique/alerts_unique.parquet: HTTP 403 GatedRepo Access to dataset digitalhen/nyc-subway-realtime is restricted and you are not in the authorized li |
| 358 | digitalhen/nyc-subway-realtime [alerts_unique] | gated | gated, no access / 68 footer(s) not readable, first: data/alerts_unique/alerts_unique.parquet: HTTP 403 GatedRepo Access to dataset digitalhen/nyc-subway-realtime is restricted and you are not in the authorized li |
| 358 | digitalhen/nyc-subway-realtime [baselines_headway] | gated | gated, no access / 82 footer(s) not readable, first: data/alerts_unique/alerts_unique.parquet: HTTP 403 GatedRepo Access to dataset digitalhen/nyc-subway-realtime is restricted and you are not in the authorized li |
| 358 | digitalhen/nyc-subway-realtime [baselines_stop_hold] | gated | gated, no access / 83 footer(s) not readable, first: data/alerts_unique/alerts_unique.parquet: HTTP 403 GatedRepo Access to dataset digitalhen/nyc-subway-realtime is restricted and you are not in the authorized li |
| 358 | digitalhen/nyc-subway-realtime [delay_predictions] | gated | gated, no access / 71 footer(s) not readable, first: data/alerts_unique/alerts_unique.parquet: HTTP 403 GatedRepo Access to dataset digitalhen/nyc-subway-realtime is restricted and you are not in the authorized li |
| 358 | digitalhen/nyc-subway-realtime [feed_snapshots] | gated | gated, no access / 130 footer(s) not readable, first: data/alerts_unique/alerts_unique.parquet: HTTP 403 GatedRepo Access to dataset digitalhen/nyc-subway-realtime is restricted and you are not in the authorized li |
| 358 | digitalhen/nyc-subway-realtime [firehose_alerts] | gated | gated, no access / 310 footer(s) not readable, first: data/alerts_unique/alerts_unique.parquet: HTTP 403 GatedRepo Access to dataset digitalhen/nyc-subway-realtime is restricted and you are not in the authorized li |
| 358 | digitalhen/nyc-subway-realtime [firehose_stop_time_updates] | gated | gated, no access / 220 footer(s) not readable, first: data/alerts_unique/alerts_unique.parquet: HTTP 403 GatedRepo Access to dataset digitalhen/nyc-subway-realtime is restricted and you are not in the authorized li |
| 358 | digitalhen/nyc-subway-realtime [firehose_trip_updates] | gated | gated, no access / 175 footer(s) not readable, first: data/alerts_unique/alerts_unique.parquet: HTTP 403 GatedRepo Access to dataset digitalhen/nyc-subway-realtime is restricted and you are not in the authorized li |
| 358 | digitalhen/nyc-subway-realtime [firehose_vehicle_positions] | gated | gated, no access / 265 footer(s) not readable, first: data/alerts_unique/alerts_unique.parquet: HTTP 403 GatedRepo Access to dataset digitalhen/nyc-subway-realtime is restricted and you are not in the authorized li |
| 358 | digitalhen/nyc-subway-realtime [gtfs_routes] | gated | gated, no access / 78 footer(s) not readable, first: data/alerts_unique/alerts_unique.parquet: HTTP 403 GatedRepo Access to dataset digitalhen/nyc-subway-realtime is restricted and you are not in the authorized li |
| 358 | digitalhen/nyc-subway-realtime [gtfs_shapes] | gated | gated, no access / 81 footer(s) not readable, first: data/alerts_unique/alerts_unique.parquet: HTTP 403 GatedRepo Access to dataset digitalhen/nyc-subway-realtime is restricted and you are not in the authorized li |
| 358 | digitalhen/nyc-subway-realtime [gtfs_stop_times] | gated | gated, no access / 80 footer(s) not readable, first: data/alerts_unique/alerts_unique.parquet: HTTP 403 GatedRepo Access to dataset digitalhen/nyc-subway-realtime is restricted and you are not in the authorized li |
| 358 | digitalhen/nyc-subway-realtime [gtfs_stops] | gated | gated, no access / 77 footer(s) not readable, first: data/alerts_unique/alerts_unique.parquet: HTTP 403 GatedRepo Access to dataset digitalhen/nyc-subway-realtime is restricted and you are not in the authorized li |
| 358 | digitalhen/nyc-subway-realtime [gtfs_trips] | gated | gated, no access / 79 footer(s) not readable, first: data/alerts_unique/alerts_unique.parquet: HTTP 403 GatedRepo Access to dataset digitalhen/nyc-subway-realtime is restricted and you are not in the authorized li |
| 358 | digitalhen/nyc-subway-realtime [incident_replays] | gated | gated, no access / 73 footer(s) not readable, first: data/alerts_unique/alerts_unique.parquet: HTTP 403 GatedRepo Access to dataset digitalhen/nyc-subway-realtime is restricted and you are not in the authorized li |
| 358 | digitalhen/nyc-subway-realtime [ridership_hourly] | gated | gated, no access / frequency: catalog says 30s, metadata says 1h (stated in the config / folder name) / 74 footer(s) not readable, first: data/alerts_unique/alerts_unique.parquet: HTTP 403 GatedRepo Access to dataset digitalhen/nyc-subway-realtime is restricted and you are not in the authorized li |
| 358 | digitalhen/nyc-subway-realtime [route_station_5min] | gated | gated, no access / frequency: catalog says 30s, metadata says 5min (stated in the config / folder name) / 55 footer(s) not readable, first: data/alerts_unique/alerts_unique.parquet: HTTP 403 GatedRepo Access to dataset digitalhen/nyc-subway-realtime is restricted and you are not in the authorized li |
| 358 | digitalhen/nyc-subway-realtime [station_adjacency] | gated | gated, no access / 84 footer(s) not readable, first: data/alerts_unique/alerts_unique.parquet: HTTP 403 GatedRepo Access to dataset digitalhen/nyc-subway-realtime is restricted and you are not in the authorized li |
| 358 | digitalhen/nyc-subway-realtime [station_complex] | gated | gated, no access / 75 footer(s) not readable, first: data/alerts_unique/alerts_unique.parquet: HTTP 403 GatedRepo Access to dataset digitalhen/nyc-subway-realtime is restricted and you are not in the authorized li |
| 358 | digitalhen/nyc-subway-realtime [station_complex_stops] | gated | gated, no access / 76 footer(s) not readable, first: data/alerts_unique/alerts_unique.parquet: HTTP 403 GatedRepo Access to dataset digitalhen/nyc-subway-realtime is restricted and you are not in the authorized li |
| 358 | digitalhen/nyc-subway-realtime [station_daily] | gated | gated, no access / frequency: catalog says 30s, metadata says 1D (stated in the config / folder name) / 67 footer(s) not readable, first: data/alerts_unique/alerts_unique.parquet: HTTP 403 GatedRepo Access to dataset digitalhen/nyc-subway-realtime is restricted and you are not in the authorized li |
| 358 | digitalhen/nyc-subway-realtime [station_hourly] | gated | gated, no access / frequency: catalog says 30s, metadata says 1h (stated in the config / folder name) / 61 footer(s) not readable, first: data/alerts_unique/alerts_unique.parquet: HTTP 403 GatedRepo Access to dataset digitalhen/nyc-subway-realtime is restricted and you are not in the authorized li |
| 358 | digitalhen/nyc-subway-realtime [terminal_stops] | gated | gated, no access / 85 footer(s) not readable, first: data/alerts_unique/alerts_unique.parquet: HTTP 403 GatedRepo Access to dataset digitalhen/nyc-subway-realtime is restricted and you are not in the authorized li |
| 359 | digitalhen/us-airport-wait-times | viewer+footers | size: catalog says 23.52 GB, the Hub file listing sums to 438.62 MB |
| 359 | digitalhen/us-airport-wait-times [cbp] | viewer+footers | size: catalog says 23.52 GB, the Hub file listing sums to 438.62 MB / points: catalog says 10M-100M rows, metadata gives 745.10k points (57.31k rows) |
| 359 | digitalhen/us-airport-wait-times [hourly] | viewer+footers | size: catalog says 23.52 GB, the Hub file listing sums to 438.62 MB |
| 359 | digitalhen/us-airport-wait-times [waits] | viewer+footers | size: catalog says 23.52 GB, the Hub file listing sums to 438.62 MB |
| 359 | digitalhen/us-airport-wait-times [waits_raw] | viewer+footers | size: catalog says 23.52 GB, the Hub file listing sums to 438.62 MB |
| 360 | dmariaa70/GO-MO | viewer | format: catalog says csv/pkl, the scope holds csv, pkl / size: catalog says 210.42 GB, the Hub file listing sums to 56.10 GB / 10 footer(s) not readable, first: default/train/0000.parquet: footer of 8592583 bytes is above the limit of 8388608 |
| 362 | EPI-Eval/global-mobility | viewer+footers | size: catalog says 1.22 GB, the Hub file listing sums to 77.97 MB |
| 363 | femartip/valencia-urban-mobility [file-index] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) |
| 363 | femartip/valencia-urban-mobility [gva_air_quality] | viewer+footers | frequency: catalog says 15min, metadata says 1Y (read from every data file path (not a config name)) |
| 364 | feregrino/london-cycles | files_only | size: catalog says 0.7 GB, the Hub file listing sums to 4.31 GB |
| 364 | feregrino/london-cycles [observations] | files_only | size: catalog says 0.7 GB, the Hub file listing sums to 4.31 GB |
| 364 | feregrino/london-cycles [stations] | files_only | size: catalog says 0.7 GB, the Hub file listing sums to 4.31 GB |
| 365 | fidel-ts/NYC_traffic_speed | footers | the dataset viewer does not work for this repo |
| 367 | gigigi7/AIS_Dataset [default] | files_only | the dataset viewer does not work for this repo |
| 368 | Jakov/bajs-zagreb [station_status] | footers | points: catalog says unknown (309 MB), metadata gives 497.07M points (55.23M rows) |
| 369 | jhumbl/dft-vehicle-licensing | viewer+footers | size: catalog says 0.09 GB, the Hub file listing sums to 27.16 MB |
| 369 | jhumbl/dft-vehicle-licensing [dft_veh0105_licensed_by_la] | viewer+footers | size: catalog says 0.09 GB, the Hub file listing sums to 27.16 MB |
| 369 | jhumbl/dft-vehicle-licensing [dft_veh0132_ulev_by_la] | viewer+footers | size: catalog says 0.09 GB, the Hub file listing sums to 27.16 MB |
| 371 | jhumbl/tfl-network-demand [tfl_network_daily_journeys] | viewer+footers | points: catalog says 1M-10M, metadata gives 5.72k points (2.86k rows) |
| 372 | Karry001/movemelbourne-gtfsr | footers | the dataset viewer does not work for this repo |
| 372 | Karry001/movemelbourne-gtfsr [coverage] | footers | points: catalog says 10M-100M rows, metadata gives 15.69k points (2.24k rows) / the dataset viewer does not work for this repo |
| 372 | Karry001/movemelbourne-gtfsr [service_alerts] | footers | the dataset viewer does not work for this repo |
| 372 | Karry001/movemelbourne-gtfsr [trip_updates] | footers | the dataset viewer does not work for this repo |
| 372 | Karry001/movemelbourne-gtfsr [vehicle_positions] | footers | the dataset viewer does not work for this repo |
| 373 | kpubdata/seoul-bike-rent-month | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) / size: catalog says 0.02 GB, the Hub file listing sums to 10.83 MB |
| 375 | matrosovdani/moscow-parking-occupancy | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 12.95 MB |
| 375 | matrosovdani/moscow-parking-occupancy [occupancy] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 12.95 MB |
| 375 | matrosovdani/moscow-parking-occupancy [parking_spots] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 12.95 MB / points: catalog says 5.1M, metadata gives 639 points (213 rows) |
| 377 | National-Digital/nsw-traffic-volume-yearly | viewer+footers | size: catalog says 0.01 GB, the Hub file listing sums to 2.38 MB |
| 378 | nju-zhangsq/RMISC [BTS] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 95868939, the footers give 353829336 |
| 379 | nju-zhangsq/RMISC [ChengduTaxi] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 2849440, the footers give 2137080 |
| 380 | nju-zhangsq/RMISC [LargeST] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 4439055168, the footers give 21775150556 |
| 381 | nju-zhangsq/RMISC [MetroTraffic] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 382 | nju-zhangsq/RMISC [PEMSCalifornia] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 38220894, the footers give 1339654146 |
| 383 | nju-zhangsq/RMISC [T-Drive] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 52988952, the footers give 35325968 |
| 384 | olaflaitinen/baku-metro-station-flow | footers | the dataset viewer does not work for this repo |
| 385 | orailix/ride-silver [line_sections] | viewer+footers | points: catalog says 100M-1B, metadata gives 56.54k points (1.21k rows) |
| 385 | orailix/ride-silver [node_links] | viewer+footers | points: catalog says 100M-1B, metadata gives 1.80k points (1.80k rows) |
| 386 | piebro/deutsche-bahn-data | viewer+footers | size: catalog says 133.08 GB, the Hub file listing sums to 91.17 GB |
| 387 | Salesforce/lotsa_data [BEIJING_SUBWAY_30MIN] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB / points: catalog says 248400, metadata gives 5.64M points (276 rows) |
| 388 | Salesforce/lotsa_data [covid_mobility] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 389 | Salesforce/lotsa_data [HZMETRO] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 390 | Salesforce/lotsa_data [largest_2017] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 391 | Salesforce/lotsa_data [largest_2018] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 392 | Salesforce/lotsa_data [largest_2019] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 393 | Salesforce/lotsa_data [largest_2020] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 394 | Salesforce/lotsa_data [largest_2021] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 395 | Salesforce/lotsa_data [LOS_LOOP] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 396 | Salesforce/lotsa_data [pedestrian_counts] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 397 | Salesforce/lotsa_data [PEMS03] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 398 | Salesforce/lotsa_data [PEMS04] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 399 | Salesforce/lotsa_data [PEMS07] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 400 | Salesforce/lotsa_data [PEMS08] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 401 | Salesforce/lotsa_data [PEMS_BAY] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 402 | Salesforce/lotsa_data [Q-TRAFFIC] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 403 | Salesforce/lotsa_data [rideshare_with_missing] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 404 | Salesforce/lotsa_data [SHMETRO] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 405 | Salesforce/lotsa_data [taxi_30min] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 406 | Salesforce/lotsa_data [traffic_hourly] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 407 | Salesforce/lotsa_data [uber_tlc_daily] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 408 | Salesforce/lotsa_data [uber_tlc_hourly] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 409 | Salesforce/lotsa_data [vehicle_trips_with_missing] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 411 | swadhinbiswas/air-traffic | footers | size: catalog says 89.76 GB, the Hub file listing sums to 28.91 GB / the dataset viewer does not work for this repo |
| 411 | swadhinbiswas/air-traffic [bronze] | footers | size: catalog says 89.76 GB, the Hub file listing sums to 28.91 GB / the dataset viewer does not work for this repo |
| 411 | swadhinbiswas/air-traffic [silver] | footers | size: catalog says 89.76 GB, the Hub file listing sums to 28.91 GB / the dataset viewer does not work for this repo |
| 413 | witgaw/METR-LA | viewer+footers | size: catalog says 0.45 GB, the Hub file listing sums to 229.30 MB |
| 414 | witgaw/PEMS-BAY | viewer+footers | size: catalog says 1.03 GB, the Hub file listing sums to 515.65 MB |
| 415 | yadavsaurabh/indian-railways-runs | viewer+footers | size: catalog says 6.41 GB, the Hub file listing sums to 34.69 GB |
| 415 | yadavsaurabh/indian-railways-runs [default] | viewer+footers | size: catalog says 6.41 GB, the Hub file listing sums to 34.69 GB |
| 415 | yadavsaurabh/indian-railways-runs [trains] | viewer+footers | size: catalog says 6.41 GB, the Hub file listing sums to 34.69 GB |
| 415 | yadavsaurabh/indian-railways-runs [trains_1] | viewer+footers | size: catalog says 6.41 GB, the Hub file listing sums to 34.69 GB |
| 416 | yind7/EvoXXLTraffic | viewer+footers | format: catalog says npy, the scope holds csv, csv.gz, full_year, json, npz, tfnsw, txt |
| 421 | TorchSpatiotemporal/ProcessedDatasets | files_only | the dataset viewer does not work for this repo |
| 422 | onedean/UrbanFM-datasets [taxinyc/bikenyc/tdrive/trafficsh/occhamburg/occparis (+pems, metrla)] | viewer+footers | subset text not matched on the Hub: taxinyc/bikenyc/tdrive/trafficsh/occhamburg/occparis |
| 423 | tjtrans/Traffic-Demand-Bench | files_only | size: catalog says 3.33 GB, the Hub file listing sums to 1.96 GB / the dataset viewer does not work for this repo |
| 424 | ZanderL1337/shipping-data-pipeline | viewer+footers | size: catalog says 85.16 GB, the Hub file listing sums to 1.76 GB / frequency: catalog says 1D / event-level, metadata says 1W (freq column (weekly)) |
| 425 | Bayway/italian-open-timeseries [bologna_bikes_hourly] | viewer+footers | points: catalog says 2.26M rows, metadata gives 201.09k points (201.09k rows) |
| 425 | Bayway/italian-open-timeseries [bologna_bikes_live_hourly] | viewer+footers | points: catalog says 2.26M rows, metadata gives 33.19k points (33.19k rows) |
| 425 | Bayway/italian-open-timeseries [bologna_pm10_daily] | viewer+footers | points: catalog says 2.26M rows, metadata gives 10.74k points (10.74k rows) |
| 425 | Bayway/italian-open-timeseries [bologna_traffic_hourly] | viewer+footers | points: catalog says 2.26M rows, metadata gives 203.76k points (203.76k rows) |
| 425 | Bayway/italian-open-timeseries [bologna_weather_hourly] | viewer+footers | points: catalog says 2.26M rows, metadata gives 136.93k points (136.93k rows) |
| 425 | Bayway/italian-open-timeseries [italy_deaths_weekly] | viewer+footers | points: catalog says 2.26M rows, metadata gives 809 points (809 rows) |
| 425 | Bayway/italian-open-timeseries [italy_earthquakes_daily] | viewer+footers | points: catalog says 2.26M rows, metadata gives 5.41k points (5.41k rows) |
| 425 | Bayway/italian-open-timeseries [italy_energy_demand_monthly] | viewer+footers | points: catalog says 2.26M rows, metadata gives 389 points (389 rows) |
| 425 | Bayway/italian-open-timeseries [italy_energy_prices_monthly] | viewer+footers | points: catalog says 2.26M rows, metadata gives 1.11k points (1.11k rows) |
| 425 | Bayway/italian-open-timeseries [italy_macro_monthly] | viewer+footers | points: catalog says 2.26M rows, metadata gives 809 points (809 rows) |
| 425 | Bayway/italian-open-timeseries [italy_tourism_monthly] | viewer+footers | points: catalog says 2.26M rows, metadata gives 880 points (880 rows) |
| 425 | Bayway/italian-open-timeseries [italy_tourism_regions_monthly] | viewer+footers | points: catalog says 2.26M rows, metadata gives 1.20k points (1.20k rows) |
| 425 | Bayway/italian-open-timeseries [itwiki_pageviews_daily] | viewer+footers | points: catalog says 2.26M rows, metadata gives 4.13k points (4.13k rows) |
| 425 | Bayway/italian-open-timeseries [milano_pm10_daily] | viewer+footers | points: catalog says 2.26M rows, metadata gives 23.25k points (23.25k rows) |
| 425 | Bayway/italian-open-timeseries [milano_temperature_hourly] | viewer+footers | points: catalog says 2.26M rows, metadata gives 221.88k points (221.88k rows) |
| 426 | judicialmind/BharatTS-Corpus | gated | gated, no access / 1 footer(s) not readable, first: series.parquet: HTTP 403 GatedRepo Access to dataset judicialmind/BharatTS-Corpus is restricted and you are not in the authorized list. Visit https://huggingfac |
| 427 | nju-zhangsq/RMISC [TimeMMD] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 95104, the footers give 590064 |
| 428 | thuml/UTSD [London Smart Meters / Wind Farms / Aus. Electricity Demand / Pedestrian Counts / Web Traffic / Sunspot] | viewer+footers | subset text not matched on the Hub: London Smart Meters; Wind Farms; Aus. Electricity Demand; Pedestrian Counts; Web Traffic; Sunspot / size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB |
| 428 | thuml/UTSD [UTSD-12G] | viewer+footers | size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB |
| 428 | thuml/UTSD [UTSD-1G] | viewer+footers | size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB |
| 428 | thuml/UTSD [UTSD-2G] | viewer+footers | size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB |
| 428 | thuml/UTSD [UTSD-4G] | files_only | size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB |
| 428 | thuml/UTSD [default] | viewer+footers | size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB |
| 429 | AutonLab/Timeseries-PILE [forecasting] | files_only | the dataset viewer does not work for this repo |
| 430 | AutonLab/Timeseries-PILE [forecasting/autoformer] | files_only | the dataset viewer does not work for this repo |
| 431 | nju-zhangsq/RMISC [Cryptocurrency] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 432 | Time-HD-Anonymous/High_Dimensional_Time_Series | viewer+footers | size: catalog says 46.25 GB, the Hub file listing sums to 17.38 GB / 14 footer(s) not readable, first: Meter/train/0000.parquet: footer of 9811910 bytes is above the limit of 8388608 |
| 432 | Time-HD-Anonymous/High_Dimensional_Time_Series [Air Quality] | viewer+footers | size: catalog says 46.25 GB, the Hub file listing sums to 17.38 GB / 14 footer(s) not readable, first: Meter/train/0000.parquet: footer of 9811910 bytes is above the limit of 8388608 |
| 432 | Time-HD-Anonymous/High_Dimensional_Time_Series [Atec] | viewer+footers | size: catalog says 46.25 GB, the Hub file listing sums to 17.38 GB / 14 footer(s) not readable, first: Meter/train/0000.parquet: footer of 9811910 bytes is above the limit of 8388608 |
| 432 | Time-HD-Anonymous/High_Dimensional_Time_Series [ECL] | viewer+footers | size: catalog says 46.25 GB, the Hub file listing sums to 17.38 GB / 14 footer(s) not readable, first: Meter/train/0000.parquet: footer of 9811910 bytes is above the limit of 8388608 |
| 432 | Time-HD-Anonymous/High_Dimensional_Time_Series [ETTh1] | viewer+footers | size: catalog says 46.25 GB, the Hub file listing sums to 17.38 GB / 14 footer(s) not readable, first: Meter/train/0000.parquet: footer of 9811910 bytes is above the limit of 8388608 |
| 432 | Time-HD-Anonymous/High_Dimensional_Time_Series [ETTh2] | viewer+footers | size: catalog says 46.25 GB, the Hub file listing sums to 17.38 GB / 14 footer(s) not readable, first: Meter/train/0000.parquet: footer of 9811910 bytes is above the limit of 8388608 |
| 432 | Time-HD-Anonymous/High_Dimensional_Time_Series [ETTm1] | viewer+footers | size: catalog says 46.25 GB, the Hub file listing sums to 17.38 GB / 14 footer(s) not readable, first: Meter/train/0000.parquet: footer of 9811910 bytes is above the limit of 8388608 |
| 432 | Time-HD-Anonymous/High_Dimensional_Time_Series [ETTm2] | viewer+footers | size: catalog says 46.25 GB, the Hub file listing sums to 17.38 GB / 14 footer(s) not readable, first: Meter/train/0000.parquet: footer of 9811910 bytes is above the limit of 8388608 |
| 432 | Time-HD-Anonymous/High_Dimensional_Time_Series [M5] | viewer+footers | size: catalog says 46.25 GB, the Hub file listing sums to 17.38 GB / 14 footer(s) not readable, first: Meter/train/0000.parquet: footer of 9811910 bytes is above the limit of 8388608 |
| 432 | Time-HD-Anonymous/High_Dimensional_Time_Series [Measles] | viewer+footers | size: catalog says 46.25 GB, the Hub file listing sums to 17.38 GB / 14 footer(s) not readable, first: Meter/train/0000.parquet: footer of 9811910 bytes is above the limit of 8388608 |
| 432 | Time-HD-Anonymous/High_Dimensional_Time_Series [Meter] | viewer | size: catalog says 46.25 GB, the Hub file listing sums to 17.38 GB / 15 footer(s) not readable, first: Meter/train/0000.parquet: footer of 9811910 bytes is above the limit of 8388608 |
| 432 | Time-HD-Anonymous/High_Dimensional_Time_Series [Mobility] | viewer+footers | size: catalog says 46.25 GB, the Hub file listing sums to 17.38 GB / 15 footer(s) not readable, first: Meter/train/0000.parquet: footer of 9811910 bytes is above the limit of 8388608 |
| 432 | Time-HD-Anonymous/High_Dimensional_Time_Series [Neurolib] | viewer+footers | size: catalog says 46.25 GB, the Hub file listing sums to 17.38 GB / 16 footer(s) not readable, first: Meter/train/0000.parquet: footer of 9811910 bytes is above the limit of 8388608 |
| 432 | Time-HD-Anonymous/High_Dimensional_Time_Series [Rebound] | viewer | size: catalog says 46.25 GB, the Hub file listing sums to 17.38 GB / 17 footer(s) not readable, first: Meter/train/0000.parquet: footer of 9811910 bytes is above the limit of 8388608 |
| 432 | Time-HD-Anonymous/High_Dimensional_Time_Series [SIRS] | viewer+footers | size: catalog says 46.25 GB, the Hub file listing sums to 17.38 GB / 17 footer(s) not readable, first: Meter/train/0000.parquet: footer of 9811910 bytes is above the limit of 8388608 |
| 432 | Time-HD-Anonymous/High_Dimensional_Time_Series [SP500] | viewer+footers | size: catalog says 46.25 GB, the Hub file listing sums to 17.38 GB / 17 footer(s) not readable, first: Meter/train/0000.parquet: footer of 9811910 bytes is above the limit of 8388608 |
| 432 | Time-HD-Anonymous/High_Dimensional_Time_Series [Solar] | viewer+footers | size: catalog says 46.25 GB, the Hub file listing sums to 17.38 GB / 22 footer(s) not readable, first: Meter/train/0000.parquet: footer of 9811910 bytes is above the limit of 8388608 |
| 432 | Time-HD-Anonymous/High_Dimensional_Time_Series [Temp] | viewer+footers | size: catalog says 46.25 GB, the Hub file listing sums to 17.38 GB / 22 footer(s) not readable, first: Meter/train/0000.parquet: footer of 9811910 bytes is above the limit of 8388608 |
| 432 | Time-HD-Anonymous/High_Dimensional_Time_Series [Traffic] | viewer+footers | size: catalog says 46.25 GB, the Hub file listing sums to 17.38 GB / 22 footer(s) not readable, first: Meter/train/0000.parquet: footer of 9811910 bytes is above the limit of 8388608 |
| 432 | Time-HD-Anonymous/High_Dimensional_Time_Series [Traffic-CA] | viewer+footers | size: catalog says 46.25 GB, the Hub file listing sums to 17.38 GB / 26 footer(s) not readable, first: Meter/train/0000.parquet: footer of 9811910 bytes is above the limit of 8388608 |
| 432 | Time-HD-Anonymous/High_Dimensional_Time_Series [Traffic-GBA] | viewer+footers | size: catalog says 46.25 GB, the Hub file listing sums to 17.38 GB / 26 footer(s) not readable, first: Meter/train/0000.parquet: footer of 9811910 bytes is above the limit of 8388608 |
| 432 | Time-HD-Anonymous/High_Dimensional_Time_Series [Traffic-GLA] | viewer+footers | size: catalog says 46.25 GB, the Hub file listing sums to 17.38 GB / 26 footer(s) not readable, first: Meter/train/0000.parquet: footer of 9811910 bytes is above the limit of 8388608 |
| 432 | Time-HD-Anonymous/High_Dimensional_Time_Series [Weather] | viewer+footers | size: catalog says 46.25 GB, the Hub file listing sums to 17.38 GB / 26 footer(s) not readable, first: Meter/train/0000.parquet: footer of 9811910 bytes is above the limit of 8388608 |
| 432 | Time-HD-Anonymous/High_Dimensional_Time_Series [Wiki] | viewer | size: catalog says 46.25 GB, the Hub file listing sums to 17.38 GB / 27 footer(s) not readable, first: Meter/train/0000.parquet: footer of 9811910 bytes is above the limit of 8388608 |
| 432 | Time-HD-Anonymous/High_Dimensional_Time_Series [Wiki-20k] | viewer | size: catalog says 46.25 GB, the Hub file listing sums to 17.38 GB / 28 footer(s) not readable, first: Meter/train/0000.parquet: footer of 9811910 bytes is above the limit of 8388608 |
| 432 | Time-HD-Anonymous/High_Dimensional_Time_Series [Wiki-2k] | viewer+footers | size: catalog says 46.25 GB, the Hub file listing sums to 17.38 GB / 28 footer(s) not readable, first: Meter/train/0000.parquet: footer of 9811910 bytes is above the limit of 8388608 |
| 432 | Time-HD-Anonymous/High_Dimensional_Time_Series [Wind] | viewer+footers | size: catalog says 46.25 GB, the Hub file listing sums to 17.38 GB / 28 footer(s) not readable, first: Meter/train/0000.parquet: footer of 9811910 bytes is above the limit of 8388608 |
| 433 | Multiple-Time-Series/PanelTS | files_only | the dataset viewer does not work for this repo |
| 434 | nju-zhangsq/RMISC [ExchangeRate] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 435 | nju-zhangsq/RMISC [Illness] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 436 | prabakaranchandran/enterprise-dynamics-corpus-v1 | viewer+footers | size: catalog says 27.19 GB, the Hub file listing sums to 8.31 GB |
| 436 | prabakaranchandran/enterprise-dynamics-corpus-v1 [all] | viewer+footers | size: catalog says 27.19 GB, the Hub file listing sums to 8.31 GB |
| 436 | prabakaranchandran/enterprise-dynamics-corpus-v1 [building_operations] | viewer+footers | size: catalog says 27.19 GB, the Hub file listing sums to 8.31 GB |
| 436 | prabakaranchandran/enterprise-dynamics-corpus-v1 [business_risk_events] | viewer+footers | size: catalog says 27.19 GB, the Hub file listing sums to 8.31 GB / points: catalog says 1M-10M series, metadata gives 403.82M points (15.54k rows) |
| 436 | prabakaranchandran/enterprise-dynamics-corpus-v1 [cloud_datacenter_ops] | viewer+footers | size: catalog says 27.19 GB, the Hub file listing sums to 8.31 GB / points: catalog says 1M-10M series, metadata gives 379.47M points (18.82k rows) |
| 436 | prabakaranchandran/enterprise-dynamics-corpus-v1 [commerce_demand] | viewer+footers | size: catalog says 27.19 GB, the Hub file listing sums to 8.31 GB / points: catalog says 1M-10M series, metadata gives 220.91M points (53.32k rows) |
| 436 | prabakaranchandran/enterprise-dynamics-corpus-v1 [commercial_core] | viewer+footers | size: catalog says 27.19 GB, the Hub file listing sums to 8.31 GB |
| 436 | prabakaranchandran/enterprise-dynamics-corpus-v1 [customer_activity] | viewer+footers | size: catalog says 27.19 GB, the Hub file listing sums to 8.31 GB / points: catalog says 1M-10M series, metadata gives 912.45M points (30.21k rows) |
| 436 | prabakaranchandran/enterprise-dynamics-corpus-v1 [digital_demand] | viewer+footers | size: catalog says 27.19 GB, the Hub file listing sums to 8.31 GB |
| 436 | prabakaranchandran/enterprise-dynamics-corpus-v1 [economic_activity] | viewer+footers | size: catalog says 27.19 GB, the Hub file listing sums to 8.31 GB |
| 436 | prabakaranchandran/enterprise-dynamics-corpus-v1 [enterprise_pretrain] | viewer+footers | size: catalog says 27.19 GB, the Hub file listing sums to 8.31 GB |
| 436 | prabakaranchandran/enterprise-dynamics-corpus-v1 [enterprise_synthetic] | viewer+footers | size: catalog says 27.19 GB, the Hub file listing sums to 8.31 GB |
| 436 | prabakaranchandran/enterprise-dynamics-corpus-v1 [industrial_production] | viewer+footers | size: catalog says 27.19 GB, the Hub file listing sums to 8.31 GB |
| 436 | prabakaranchandran/enterprise-dynamics-corpus-v1 [industrial_well_sensors] | viewer+footers | size: catalog says 27.19 GB, the Hub file listing sums to 8.31 GB / points: catalog says 1M-10M series, metadata gives 322.07M points (94 rows) |
| 436 | prabakaranchandran/enterprise-dynamics-corpus-v1 [markets] | viewer+footers | size: catalog says 27.19 GB, the Hub file listing sums to 8.31 GB |
| 436 | prabakaranchandran/enterprise-dynamics-corpus-v1 [prices] | viewer+footers | size: catalog says 27.19 GB, the Hub file listing sums to 8.31 GB |
| 436 | prabakaranchandran/enterprise-dynamics-corpus-v1 [trade_supply_chain] | viewer+footers | size: catalog says 27.19 GB, the Hub file listing sums to 8.31 GB |
| 437 | alanwgt/fipex-veiculos-brasil | viewer+footers | size: catalog says 23.98 GB, the Hub file listing sums to 2.27 GB |
| 438 | autogluon/chronos_datasets [dominick] | viewer+footers | licence: catalog says per-dataset (card: check ds.info); repo not gated, the card says other |
| 439 | CQ-AI/Financial_datasets [Retail and wherehouse Sale.csv] | files_only | the dataset viewer does not work for this repo |
| 440 | Dingdong-Inc/FreshRetailNet-50K [default] | viewer+footers | 1 footer(s) not readable, first: data/train.parquet: footer of 9472377 bytes is above the limit of 8388608 |
| 441 | Dingdong-Inc/FreshRetailNet-LT [unknown] | viewer+footers | 1 footer(s) not readable, first: data/train.parquet: footer of 16701267 bytes is above the limit of 8388608 |
| 442 | dubairealestatedata/dubai-real-estate-sales-transactions | footers | size: catalog says 1.28 GB, the Hub file listing sums to 263.37 MB / the dataset viewer does not work for this repo |
| 443 | earino/my-pricecatcher-weekly-increase | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv, json) |
| 444 | fuel-prices-eu/european-fuel-prices | viewer+footers | size: catalog says 0.29 GB, the Hub file listing sums to 5.91 MB |
| 444 | fuel-prices-eu/european-fuel-prices [d1_eu_weighted_average] | viewer+footers | format: catalog says csv.gz, the scope holds parquet / size: catalog says 0.29 GB, the Hub file listing sums to 5.91 MB / points: catalog says 100K-1M rows, metadata gives 4.36k points (2.18k rows) |
| 444 | fuel-prices-eu/european-fuel-prices [d1_weekly_panel] | viewer+footers | format: catalog says csv.gz, the scope holds parquet / size: catalog says 0.29 GB, the Hub file listing sums to 5.91 MB |
| 444 | fuel-prices-eu/european-fuel-prices [d2_national_daily] | viewer+footers | format: catalog says csv.gz, the scope holds parquet / size: catalog says 0.29 GB, the Hub file listing sums to 5.91 MB |
| 444 | fuel-prices-eu/european-fuel-prices [d3_regional_daily] | viewer+footers | format: catalog says csv.gz, the scope holds parquet / size: catalog says 0.29 GB, the Hub file listing sums to 5.91 MB |
| 445 | idomanteu/cs2-historical-item-prices-hourly-march-april-2026 | viewer+footers | size: catalog says 3.87 GB, the Hub file listing sums to 1.97 GB |
| 446 | jamesb7/fuel-prices-uk | viewer+footers | size: catalog says 15.13 GB, the Hub file listing sums to 178.03 MB |
| 446 | jamesb7/fuel-prices-uk [price_history] | viewer+footers | size: catalog says 15.13 GB, the Hub file listing sums to 178.03 MB |
| 446 | jamesb7/fuel-prices-uk [stations] | viewer+footers | size: catalog says 15.13 GB, the Hub file listing sums to 178.03 MB |
| 448 | kpubdata/seoul-apartment-trades | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 6.07 MB |
| 448 | kpubdata/seoul-apartment-trades [en] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 6.07 MB |
| 448 | kpubdata/seoul-apartment-trades [ko] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 6.07 MB |
| 451 | nju-zhangsq/RMISC [Dominick] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 452 | nju-zhangsq/RMISC [WeeklyFuelPricesItaly] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 16600, the footers give 19920 |
| 453 | nju-zhangsq/RMISC [WeeklyRoadFuelPrices] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 454 | pracadata/brazil-fuel-prices | viewer+footers | size: catalog says 0.08 GB, the Hub file listing sums to 19.14 MB |
| 454 | pracadata/brazil-fuel-prices [fuel_coverage_weekly] | viewer+footers | size: catalog says 0.08 GB, the Hub file listing sums to 19.14 MB / points: catalog says 1M-10M rows, metadata gives 3.52k points (1.17k rows) |
| 454 | pracadata/brazil-fuel-prices [fuel_price_municipality_weekly] | viewer+footers | size: catalog says 0.08 GB, the Hub file listing sums to 19.14 MB |
| 455 | pracadata/mexico-fuel-prices [fuel_coverage_daily_mx] | viewer+footers | points: catalog says 10K-100K, metadata gives 150 points (15 rows) |
| 457 | SURESHBEEKHANI/SalesForecastingWOMartStore [default] | files_only | the dataset viewer does not work for this repo |
| 458 | thehooklab/nsw-property | viewer+footers | size: catalog says 1.29 GB, the Hub file listing sums to 72.62 MB |
| 458 | thehooklab/nsw-property [enriched] | viewer+footers | size: catalog says 1.29 GB, the Hub file listing sums to 72.62 MB |
| 458 | thehooklab/nsw-property [rental] | viewer+footers | size: catalog says 1.29 GB, the Hub file listing sums to 72.62 MB |
| 458 | thehooklab/nsw-property [sales] | viewer+footers | size: catalog says 1.29 GB, the Hub file listing sums to 72.62 MB |
| 459 | VincentGOURBIN/FuelInFranceData | viewer+footers | size: catalog says 62.9 GB, the Hub file listing sums to 11.71 GB |
| 460 | xus/Spain-Fuel-Prices | viewer+footers | size: catalog says 3.22 GB, the Hub file listing sums to 64.22 MB / frequency: catalog says 1D, metadata says 1M (implied by span and series length (span / (length - 1) = 2.474e+06 s)) |
| 460 | xus/Spain-Fuel-Prices [brent_prices] | viewer+footers | size: catalog says 3.22 GB, the Hub file listing sums to 64.22 MB / points: catalog says 10M-100M, metadata gives 598 points (299 rows) |
| 460 | xus/Spain-Fuel-Prices [fuel_prices] | viewer+footers | size: catalog says 3.22 GB, the Hub file listing sums to 64.22 MB / frequency: catalog says 1D, metadata says 1M (implied by span and series length (span / (length - 1) = 2.474e+06 s)) |
| 466 | gladcolor/hourly_population_US | files_only | the dataset viewer does not work for this repo |
| 469 | Jannchie/bilibili-creator-stats | gated | gated, no access / 50 footer(s) not readable, first: data/author_fans/increments/2026-09-30.parquet: HTTP 403 GatedRepo Access to dataset Jannchie/bilibili-creator-stats is restricted and you are not in the author |
| 469 | Jannchie/bilibili-creator-stats [author_fans] | gated | gated, no access / 150 footer(s) not readable, first: data/author_fans/increments/2026-09-30.parquet: HTTP 403 GatedRepo Access to dataset Jannchie/bilibili-creator-stats is restricted and you are not in the author |
| 469 | Jannchie/bilibili-creator-stats [author_info] | gated | gated, no access / 250 footer(s) not readable, first: data/author_fans/increments/2026-09-30.parquet: HTTP 403 GatedRepo Access to dataset Jannchie/bilibili-creator-stats is restricted and you are not in the author |
| 469 | Jannchie/bilibili-creator-stats [live_paid_event_aggregations] | gated | gated, no access / 255 footer(s) not readable, first: data/author_fans/increments/2026-09-30.parquet: HTTP 403 GatedRepo Access to dataset Jannchie/bilibili-creator-stats is restricted and you are not in the author |
| 469 | Jannchie/bilibili-creator-stats [live_paid_events] | gated | gated, no access / 253 footer(s) not readable, first: data/author_fans/increments/2026-09-30.parquet: HTTP 403 GatedRepo Access to dataset Jannchie/bilibili-creator-stats is restricted and you are not in the author |
| 469 | Jannchie/bilibili-creator-stats [live_room_gifts] | gated | gated, no access / 254 footer(s) not readable, first: data/author_fans/increments/2026-09-30.parquet: HTTP 403 GatedRepo Access to dataset Jannchie/bilibili-creator-stats is restricted and you are not in the author |
| 469 | Jannchie/bilibili-creator-stats [livers] | gated | gated, no access / 256 footer(s) not readable, first: data/author_fans/increments/2026-09-30.parquet: HTTP 403 GatedRepo Access to dataset Jannchie/bilibili-creator-stats is restricted and you are not in the author |
| 469 | Jannchie/bilibili-creator-stats [tags] | gated | gated, no access / 251 footer(s) not readable, first: data/author_fans/increments/2026-09-30.parquet: HTTP 403 GatedRepo Access to dataset Jannchie/bilibili-creator-stats is restricted and you are not in the author |
| 469 | Jannchie/bilibili-creator-stats [video_info] | gated | gated, no access / 200 footer(s) not readable, first: data/author_fans/increments/2026-09-30.parquet: HTTP 403 GatedRepo Access to dataset Jannchie/bilibili-creator-stats is restricted and you are not in the author |
| 469 | Jannchie/bilibili-creator-stats [video_stats] | gated | gated, no access / 100 footer(s) not readable, first: data/author_fans/increments/2026-09-30.parquet: HTTP 403 GatedRepo Access to dataset Jannchie/bilibili-creator-stats is restricted and you are not in the author |
| 471 | lingbow/tiktok-video-engagement-200k | viewer+footers | size: catalog says 0.24 GB, the Hub file listing sums to 135.54 MB |
| 471 | lingbow/tiktok-video-engagement-200k [creator_daily] | viewer+footers | size: catalog says 0.24 GB, the Hub file listing sums to 135.54 MB |
| 471 | lingbow/tiktok-video-engagement-200k [engagement_daily] | viewer+footers | size: catalog says 0.24 GB, the Hub file listing sums to 135.54 MB |
| 471 | lingbow/tiktok-video-engagement-200k [videos] | viewer+footers | size: catalog says 0.24 GB, the Hub file listing sums to 135.54 MB |
| 473 | Nason/open-policing-database | files_only | format: catalog says parquet, the scope has no parquet file (duckdb) / the dataset viewer does not work for this repo |
| 475 | nju-zhangsq/RMISC [BooksPerPerson] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 477 | valdovaldo1500/crime-brasil-rs-2022-2026 | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) |
| 478 | yuiseki/estat-jumin-idou [tables] | viewer+footers | points: catalog says 214M rows, metadata gives 471 points (471 rows) |
| 480 | zwn22/NC_Crime | files_only | the dataset viewer does not work for this repo |
| 481 | yuiseki/un-wpp [WPP2024] | viewer+footers | licence: catalog says CC-BY-3.0-IGO, the card says other |
| 483 | zaai-ai/hierarchical_time_series_datasets | files_only | format: catalog says parquet, the scope has no parquet file (pkl) / size: catalog says 0.02 GB, the Hub file listing sums to 5.99 MB / the dataset viewer does not work for this repo |
| 484 | earino/abs-visitor-arrivals-direction | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv, json) |
| 485 | AstroMLCore/AstroM3Dataset | files_only | size: catalog says 39.5 GB, the Hub file listing sums to 1.32 GB / the dataset viewer does not work for this repo |
| 486 | AutonLab/Timeseries-PILE [anomaly_detection/TSB-UAD-Public] | files_only | the dataset viewer does not work for this repo |
| 487 | cangyeone/SeismicX-Cont | viewer+footers | licence: catalog says layered upstream terms, the card says other |
| 490 | juliensimon/celestrak-space-weather | viewer+footers | licence: catalog says CelesTrak usage policy, the card says other |
| 491 | juliensimon/deep-space-probes | viewer+footers | size: catalog says 0.33 GB, the Hub file listing sums to 33.70 MB |
| 492 | juliensimon/dst-index | footers | the dataset viewer does not work for this repo |
| 498 | juliensimon/neutron-monitor-cosmic-rays | viewer+footers | size: catalog says 0.78 GB, the Hub file listing sums to 3.84 MB |
| 501 | juliensimon/space-track-tle-history | gated | gated, no access / size: catalog says 135.47 GB, the Hub file listing sums to 12.55 GB / 50 footer(s) not readable, first: data/tle_1959.parquet: HTTP 403 GatedRepo Access to dataset juliensimon/space-track-tle-history is restricted and you are not in the authorized list. Visit http |
| 502 | lxr11111/telemanom [SMAP,MSL] | viewer+footers | subset text not matched on the Hub: SMAP; MSL |
| 503 | MultimodalUniverse/tess | viewer+footers | points: catalog says ~1e8 (est.), metadata gives 3.21B points (100.00k rows) |
| 505 | nju-zhangsq/RMISC [KeplerLightCurves] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 506 | nju-zhangsq/RMISC [Satellite] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 2906716, the footers give 7992704 |
| 509 | patrickfleith/GOCE-satellite-telemtry [raw, 10min stats, 6h stats] | viewer+footers | subset text not matched on the Hub: raw; 10min stats; 6h stats |
| 510 | phanerozoic/qiskit-calibration-drift | footers | size: catalog says 280.66 GB, the Hub file listing sums to 106.54 MB |
| 513 | aerostratospheric/spc-tornado-history | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv, json) / size: catalog says 0.01 GB, the Hub file listing sums to 16.03 MB |
| 514 | alexdum/meteogate-archive | viewer+footers | size: catalog says 69.77 GB, the Hub file listing sums to 11.41 GB |
| 515 | autogluon/chronos_datasets [monash_weather] | viewer+footers | licence: catalog says per-dataset (card: check ds.info); repo not gated, the card says other |
| 516 | autogluon/chronos_datasets [ushcn_daily] | viewer+footers | licence: catalog says per-dataset (card: check ds.info); repo not gated, the card says other |
| 521 | autogluon/chronos_datasets_extra [brazilian_cities_temperature] | files_only | empty: no data files in the scope / subset text not matched on the Hub: brazilian_cities_temperature / the dataset viewer does not work for this repo |
| 522 | brandonhxrr/mexico_city_weather | files_only | size: catalog says 0.06 GB, the Hub file listing sums to 122.34 MB / the dataset viewer does not work for this repo |
| 523 | cm3r/meteo-france-climatologie-quotidienne [stations] | viewer+footers | points: catalog says 100M-1B, metadata gives 73.75k points (14.75k rows) |
| 524 | datonic/spain_aemet_historical_weather | viewer+footers | size: catalog says 5.5 GB, the Hub file listing sums to 128.17 MB |
| 525 | DuckerMaster/thaiclim-district [district_boundaries] | viewer+footers | format: catalog says csv, the scope holds geoparquet / points: catalog says 10M-100M rows, metadata gives 928 points (928 rows) |
| 525 | DuckerMaster/thaiclim-district [ground_truth_stations] | viewer+footers | format: catalog says csv, the scope holds parquet / points: catalog says 10M-100M rows, metadata gives 411.14k points (102.79k rows) |
| 525 | DuckerMaster/thaiclim-district [weather_daily_2000_2009] | viewer+footers | format: catalog says csv, the scope holds parquet |
| 525 | DuckerMaster/thaiclim-district [weather_daily_2010_2019] | viewer+footers | format: catalog says csv, the scope holds parquet |
| 525 | DuckerMaster/thaiclim-district [weather_daily_2020_2025] | viewer+footers | format: catalog says csv, the scope holds parquet |
| 525 | DuckerMaster/thaiclim-district [weather_daily_26year_master] | viewer+footers | format: catalog says csv, the scope holds parquet |
| 527 | elskow/Weather4cast | files_only | the dataset viewer does not work for this repo |
| 529 | excarta/madis2020 | files_only | empty: no data files in the scope / the dataset viewer does not work for this repo |
| 530 | furusu/japan-max-temperature | viewer+footers | size: catalog says 2.72 GB, the Hub file listing sums to 37.89 MB |
| 531 | Geoweaver/SnowWaterEquivalent_trainingdata | files_only | the dataset viewer does not work for this repo |
| 533 | jacobbieker/dwd | files_only | the dataset viewer does not work for this repo |
| 534 | jotone/arpa-osmer-weatherstations [daily] | viewer+footers | frequency: catalog says 1H, metadata says 1D (stated in the config / folder name (daily)) |
| 534 | jotone/arpa-osmer-weatherstations [stations] | viewer+footers | points: catalog says 10M-100M rows, metadata gives 126 points (63 rows) |
| 535 | juliangehring/noaa-isd | footers | frequency: catalog says irregular (~hourly, observation-level), metadata says 1Y (read from every data file path (not a config name)) / the dataset viewer does not work for this repo |
| 536 | Kamilatr/weather_data_bavaria | files_only | the dataset viewer does not work for this repo |
| 537 | MeteoSwiss/PeakWeather | footers | size: catalog says 22.07 GB, the Hub file listing sums to 16.16 GB / the dataset viewer does not work for this repo |
| 538 | nakas/powderbench-truth | viewer+footers | size: catalog says 0.01 GB, the Hub file listing sums to 7.47 MB |
| 538 | nakas/powderbench-truth [alps] | viewer+footers | size: catalog says 0.01 GB, the Hub file listing sums to 7.47 MB |
| 538 | nakas/powderbench-truth [alps_stations] | viewer+footers | size: catalog says 0.01 GB, the Hub file listing sums to 7.47 MB / points: catalog says 10M-100M rows, metadata gives 432 points (216 rows) |
| 538 | nakas/powderbench-truth [observers] | viewer+footers | size: catalog says 0.01 GB, the Hub file listing sums to 7.47 MB |
| 538 | nakas/powderbench-truth [observers_stations] | viewer+footers | size: catalog says 0.01 GB, the Hub file listing sums to 7.47 MB / points: catalog says 10M-100M rows, metadata gives 1.16k points (1.16k rows) |
| 538 | nakas/powderbench-truth [stations] | viewer+footers | size: catalog says 0.01 GB, the Hub file listing sums to 7.47 MB |
| 538 | nakas/powderbench-truth [stations_stations] | viewer+footers | size: catalog says 0.01 GB, the Hub file listing sums to 7.47 MB / points: catalog says 10M-100M rows, metadata gives 996 points (996 rows) |
| 539 | nju-zhangsq/RMISC [ERA5HourlySingleLevels] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 462917265, the footers give 370333812 |
| 540 | nju-zhangsq/RMISC [GlobalClimateChange] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 5634990, the footers give 28135110 |
| 541 | nju-zhangsq/RMISC [WeatherBench5-625deg] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 43783910472, the footers give 77753629108 |
| 542 | notadib/NASA-Power-Daily-Weather | files_only | format: catalog says parquet, the scope has no parquet file (csv, json, pt) / the dataset viewer does not work for this repo |
| 543 | numix/stations-officielles-es | files_only | size: catalog says 2.27 GB, the Hub file listing sums to 951.83 MB / the dataset viewer does not work for this repo |
| 546 | ron-the-code/Telangana_time_series_2023-2025 | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) |
| 547 | Salesforce/lotsa_data [era5] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB / points: catalog says ~1e9+ (est.), metadata gives 94.16B points (239.52k rows) |
| 548 | Salesforce/lotsa_data [oikolab_weather] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 549 | Salesforce/lotsa_data [subseasonal] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 550 | Salesforce/lotsa_data [subseasonal_precip] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 551 | Salesforce/lotsa_data [weather] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 552 | sayanroy058/Weather-Time-Series-Forecasting | files_only | the dataset viewer does not work for this repo |
| 555 | thuml/UTSD [ERA5 Surface / ERA5 Pressure] | viewer+footers | subset text not matched on the Hub: ERA5 Surface; ERA5 Pressure / size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB / points: catalog says 58.44M / 116.88M, metadata gives 2.50B points (867.26k rows) |
| 555 | thuml/UTSD [UTSD-12G] | viewer+footers | size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB |
| 555 | thuml/UTSD [UTSD-1G] | viewer+footers | size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB |
| 555 | thuml/UTSD [UTSD-2G] | viewer+footers | size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB |
| 555 | thuml/UTSD [UTSD-4G] | files_only | size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB |
| 555 | thuml/UTSD [default] | viewer+footers | size: catalog says 13.13 GB, the Hub file listing sums to 6.39 GB / points: catalog says 58.44M / 116.88M, metadata gives 1.25B points (433.63k rows) |
| 557 | WasuratS/ECMWF_Thailand_Land_Air_Temperatures | viewer+footers | size: catalog says 20.78 GB, the Hub file listing sums to 9.08 GB |
| 559 | yuiseki/jma-rain-kanto [stations] | viewer+footers | points: catalog says 10M-100M, metadata gives 122 points (122 rows) |
| 562 | NaaVrug/weather-geo-era5 | viewer+footers | size: catalog says 26.66 GB, the Hub file listing sums to 17.86 GB |
| 563 | tharinduperera/weather-clustering-data | viewer+footers | size: catalog says 0.05 GB, the Hub file listing sums to 33.03 MB |
| 564 | afhubbard/gpu-prices | viewer+footers | size: catalog says 0.13 GB, the Hub file listing sums to 92.97 MB |
| 568 | autogluon/chronos_datasets [wiki_daily_100k] | viewer+footers | licence: catalog says per-dataset (card: check ds.info); repo not gated, the card says other |
| 569 | commoncrawl/statistics | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv, txt) / size: catalog says 1.03 GB, the Hub file listing sums to 54.43 MB / frequency: catalog says 1M, metadata says 15min (implied by span and series length (span / (length - 1) = 922.8 s)) |
| 569 | commoncrawl/statistics [Charsets] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) / size: catalog says 1.03 GB, the Hub file listing sums to 54.43 MB |
| 569 | commoncrawl/statistics [Crawl Similarity Digest] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) / size: catalog says 1.03 GB, the Hub file listing sums to 54.43 MB |
| 569 | commoncrawl/statistics [Crawl Similarity URL] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) / size: catalog says 1.03 GB, the Hub file listing sums to 54.43 MB |
| 569 | commoncrawl/statistics [Crawl Size] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) / size: catalog says 1.03 GB, the Hub file listing sums to 54.43 MB |
| 569 | commoncrawl/statistics [Crawl Size by Type] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) / size: catalog says 1.03 GB, the Hub file listing sums to 54.43 MB |
| 569 | commoncrawl/statistics [Crawl metrics by type] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) / size: catalog says 1.03 GB, the Hub file listing sums to 54.43 MB |
| 569 | commoncrawl/statistics [Crawl overlaps URL] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) / size: catalog says 1.03 GB, the Hub file listing sums to 54.43 MB |
| 569 | commoncrawl/statistics [Crawl overlaps digest] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) / size: catalog says 1.03 GB, the Hub file listing sums to 54.43 MB |
| 569 | commoncrawl/statistics [Crawlmetrics] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) / size: catalog says 1.03 GB, the Hub file listing sums to 54.43 MB |
| 569 | commoncrawl/statistics [Domains top 500] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) / size: catalog says 1.03 GB, the Hub file listing sums to 54.43 MB |
| 569 | commoncrawl/statistics [Duplicates] | viewer+footers | format: catalog says parquet, the scope has no parquet file (txt) / size: catalog says 1.03 GB, the Hub file listing sums to 54.43 MB |
| 569 | commoncrawl/statistics [Languages] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) / size: catalog says 1.03 GB, the Hub file listing sums to 54.43 MB |
| 569 | commoncrawl/statistics [MIME Types] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) / size: catalog says 1.03 GB, the Hub file listing sums to 54.43 MB |
| 569 | commoncrawl/statistics [MIME types detected] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) / size: catalog says 1.03 GB, the Hub file listing sums to 54.43 MB |
| 569 | commoncrawl/statistics [Top-level domains] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) / size: catalog says 1.03 GB, the Hub file listing sums to 54.43 MB / frequency: catalog says 1M, metadata says 15min (implied by span and series length (span / (length - 1) = 922.8 s)) |
| 570 | emperor-mew/ooni-censorship-historical | viewer+footers | size: catalog says 0.01 GB, the Hub file listing sums to 14.67 MB |
| 570 | emperor-mew/ooni-censorship-historical [daily_aggregates] | viewer+footers | size: catalog says 0.01 GB, the Hub file listing sums to 14.67 MB |
| 570 | emperor-mew/ooni-censorship-historical [incident_annotations] | viewer+footers | size: catalog says 0.01 GB, the Hub file listing sums to 14.67 MB |
| 571 | Emulated-Inc/wikipedia-clickstream-enwiki | files_only | format: catalog says parquet, the scope has no parquet file (tsv.gz) / the dataset viewer does not work for this repo |
| 573 | geocables/internet-latency-and-routing-events | files_only | format: catalog says parquet, the scope has no parquet file (csv, json) / size: catalog says 0.72 GB, the Hub file listing sums to 32.62 MB / the dataset viewer does not work for this repo |
| 576 | hq-bench/quitobench [hour, min] | viewer+footers | size: catalog says 0.41 GB, the Hub file listing sums to 204.66 MB |
| 577 | hq-bench/quitobench [hour, min] | viewer+footers | size: catalog says 0.41 GB, the Hub file listing sums to 204.66 MB |
| 578 | jeremycochoy/wikimedia-pageview-timeseries | footers | the dataset viewer does not work for this repo |
| 578 | jeremycochoy/wikimedia-pageview-timeseries [wiki_daily] | footers | the dataset viewer does not work for this repo |
| 578 | jeremycochoy/wikimedia-pageview-timeseries [wiki_hourly] | footers | the dataset viewer does not work for this repo |
| 578 | jeremycochoy/wikimedia-pageview-timeseries [wiki_stl_residual] | footers | the dataset viewer does not work for this repo |
| 578 | jeremycochoy/wikimedia-pageview-timeseries [wiki_stl_seasonal] | footers | the dataset viewer does not work for this repo |
| 578 | jeremycochoy/wikimedia-pageview-timeseries [wiki_stl_trend] | footers | the dataset viewer does not work for this repo |
| 579 | kashif/App_Flow [default] | viewer+footers | format: catalog says csv, the scope holds zip |
| 580 | LeadLagForecasting/llf_github [default] | viewer+footers | points: catalog says unknown (>= 5 years x 4 channels per labelled repo), metadata gives 23.75B points (3.01M rows) |
| 581 | Lemma-RCA-NEC/Cloud_Computing_Preprocessed | files_only | format: catalog says npy, the scope holds zip / size: catalog says 9.13 GB, the Hub file listing sums to 4.74 GB / the dataset viewer does not work for this repo |
| 582 | Lightcap/multi-domain-cloudflare-observability | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB |
| 582 | Lightcap/multi-domain-cloudflare-observability [accounts] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB / points: catalog says 100K-1M rows, metadata gives 4 points (1 rows) |
| 582 | Lightcap/multi-domain-cloudflare-observability [data_dictionary] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB |
| 582 | Lightcap/multi-domain-cloudflare-observability [dns_analytics] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB |
| 582 | Lightcap/multi-domain-cloudflare-observability [dns_records] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB / points: catalog says 100K-1M rows, metadata gives 1.15k points (288 rows) |
| 582 | Lightcap/multi-domain-cloudflare-observability [email_routing_rules] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB / points: catalog says 100K-1M rows, metadata gives 603 points (201 rows) |
| 582 | Lightcap/multi-domain-cloudflare-observability [firewall_events] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB |
| 582 | Lightcap/multi-domain-cloudflare-observability [http_daily] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB |
| 582 | Lightcap/multi-domain-cloudflare-observability [http_dimension_groups] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB |
| 582 | Lightcap/multi-domain-cloudflare-observability [http_hourly] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB |
| 582 | Lightcap/multi-domain-cloudflare-observability [http_hourly_browser_map] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB |
| 582 | Lightcap/multi-domain-cloudflare-observability [http_hourly_content_type_map] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB |
| 582 | Lightcap/multi-domain-cloudflare-observability [http_hourly_country_map] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB |
| 582 | Lightcap/multi-domain-cloudflare-observability [http_hourly_http_version_map] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB |
| 582 | Lightcap/multi-domain-cloudflare-observability [http_hourly_ip_class_map] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB |
| 582 | Lightcap/multi-domain-cloudflare-observability [http_hourly_ssl_map] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB |
| 582 | Lightcap/multi-domain-cloudflare-observability [http_hourly_status_map] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB |
| 582 | Lightcap/multi-domain-cloudflare-observability [http_hourly_threat_pathing_map] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB / points: catalog says 100K-1M rows, metadata gives 145 points (145 rows) |
| 582 | Lightcap/multi-domain-cloudflare-observability [metric_availability] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB |
| 582 | Lightcap/multi-domain-cloudflare-observability [redirect_list_items] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB |
| 582 | Lightcap/multi-domain-cloudflare-observability [redirect_lists] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB / points: catalog says 100K-1M rows, metadata gives 2 points (1 rows) |
| 582 | Lightcap/multi-domain-cloudflare-observability [ruleset_rules] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB |
| 582 | Lightcap/multi-domain-cloudflare-observability [rulesets] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB / points: catalog says 100K-1M rows, metadata gives 783 points (783 rows) |
| 582 | Lightcap/multi-domain-cloudflare-observability [rum_pageload_groups] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB / points: catalog says 100K-1M rows, metadata gives 1.51k points (303 rows) |
| 582 | Lightcap/multi-domain-cloudflare-observability [rum_performance_groups] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB |
| 582 | Lightcap/multi-domain-cloudflare-observability [rum_web_vitals_groups] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB |
| 582 | Lightcap/multi-domain-cloudflare-observability [speed_availability] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB / points: catalog says 100K-1M rows, metadata gives 400 points (200 rows) |
| 582 | Lightcap/multi-domain-cloudflare-observability [speed_pages] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB / points: catalog says 100K-1M rows, metadata gives 4 points (1 rows) |
| 582 | Lightcap/multi-domain-cloudflare-observability [ssl_certificate_packs] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB / points: catalog says 100K-1M rows, metadata gives 400 points (200 rows) |
| 582 | Lightcap/multi-domain-cloudflare-observability [web_analytics_rules] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB / points: catalog says 100K-1M rows, metadata gives 60 points (12 rows) |
| 582 | Lightcap/multi-domain-cloudflare-observability [web_analytics_sites] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB |
| 582 | Lightcap/multi-domain-cloudflare-observability [zone_settings] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB |
| 582 | Lightcap/multi-domain-cloudflare-observability [zones] | viewer+footers | size: catalog says 0.02 GB, the Hub file listing sums to 10.66 MB / points: catalog says 100K-1M rows, metadata gives 800 points (200 rows) |
| 583 | Lilambd/security-daily | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) / size: catalog says 0.02 GB, the Hub file listing sums to 79.95 MB |
| 583 | Lilambd/security-daily [breaches] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) / size: catalog says 0.02 GB, the Hub file listing sums to 79.95 MB |
| 583 | Lilambd/security-daily [breaches_new] | files_only | format: catalog says parquet, the scope has no parquet file (csv) / size: catalog says 0.02 GB, the Hub file listing sums to 79.95 MB |
| 583 | Lilambd/security-daily [cve_modified] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) / size: catalog says 0.02 GB, the Hub file listing sums to 79.95 MB |
| 583 | Lilambd/security-daily [cve_new] | files_only | format: catalog says parquet, the scope has no parquet file (csv) / size: catalog says 0.02 GB, the Hub file listing sums to 79.95 MB |
| 583 | Lilambd/security-daily [cve_signals] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) / size: catalog says 0.02 GB, the Hub file listing sums to 79.95 MB |
| 583 | Lilambd/security-daily [epss_movers] | files_only | format: catalog says parquet, the scope has no parquet file (csv) / size: catalog says 0.02 GB, the Hub file listing sums to 79.95 MB |
| 583 | Lilambd/security-daily [incident_week] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) / size: catalog says 0.02 GB, the Hub file listing sums to 79.95 MB / frequency: catalog says 1D, metadata says 1W (stated in the config / folder name (incident_week)) |
| 583 | Lilambd/security-daily [incidents] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) / size: catalog says 0.02 GB, the Hub file listing sums to 79.95 MB |
| 583 | Lilambd/security-daily [incidents_new] | files_only | format: catalog says parquet, the scope has no parquet file (csv) / size: catalog says 0.02 GB, the Hub file listing sums to 79.95 MB |
| 583 | Lilambd/security-daily [kev] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) / size: catalog says 0.02 GB, the Hub file listing sums to 79.95 MB |
| 583 | Lilambd/security-daily [kev_added] | files_only | format: catalog says parquet, the scope has no parquet file (csv) / size: catalog says 0.02 GB, the Hub file listing sums to 79.95 MB |
| 583 | Lilambd/security-daily [patch_first] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) / size: catalog says 0.02 GB, the Hub file listing sums to 79.95 MB |
| 583 | Lilambd/security-daily [summary] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) / size: catalog says 0.02 GB, the Hub file listing sums to 79.95 MB |
| 583 | Lilambd/security-daily [vendor_week] | viewer+footers | format: catalog says parquet, the scope has no parquet file (csv) / size: catalog says 0.02 GB, the Hub file listing sums to 79.95 MB / frequency: catalog says 1D, metadata says 1W (stated in the config / folder name (vendor_week)) |
| 584 | lsliwko/google-cluster-data-2019-sorted-by-timestamp | files_only | the dataset viewer does not work for this repo |
| 585 | MachaParfait/ClusterWise | viewer+footers | size: catalog says 443.34 GB, the Hub file listing sums to 221.01 GB |
| 587 | modelpulse/model-pulse-data | footers | size: catalog says 12.54 GB, the Hub file listing sums to 2.29 GB |
| 587 | modelpulse/model-pulse-data [author_series] | footers | size: catalog says 12.54 GB, the Hub file listing sums to 2.29 GB |
| 587 | modelpulse/model-pulse-data [dataset_series] | footers | size: catalog says 12.54 GB, the Hub file listing sums to 2.29 GB |
| 587 | modelpulse/model-pulse-data [datasets] | footers | size: catalog says 12.54 GB, the Hub file listing sums to 2.29 GB / points: catalog says 1B-10B rows, metadata gives 17.76M points (1.11M rows) |
| 587 | modelpulse/model-pulse-data [family_series] | footers | size: catalog says 12.54 GB, the Hub file listing sums to 2.29 GB / points: catalog says 1B-10B rows, metadata gives 39.37M points (13.12M rows) |
| 587 | modelpulse/model-pulse-data [hub_series] | footers | size: catalog says 12.54 GB, the Hub file listing sums to 2.29 GB / points: catalog says 1B-10B rows, metadata gives 34.27k points (34.27k rows) |
| 587 | modelpulse/model-pulse-data [models] | footers | size: catalog says 12.54 GB, the Hub file listing sums to 2.29 GB / points: catalog says 1B-10B rows, metadata gives 44.75M points (1.72M rows) |
| 587 | modelpulse/model-pulse-data [series] | footers | size: catalog says 12.54 GB, the Hub file listing sums to 2.29 GB |
| 587 | modelpulse/model-pulse-data [space_series] | footers | size: catalog says 12.54 GB, the Hub file listing sums to 2.29 GB / points: catalog says 1B-10B rows, metadata gives 96.67M points (48.33M rows) |
| 587 | modelpulse/model-pulse-data [spaces] | footers | size: catalog says 12.54 GB, the Hub file listing sums to 2.29 GB / points: catalog says 1B-10B rows, metadata gives 756.46k points (108.06k rows) |
| 587 | modelpulse/model-pulse-data [uses] | footers | size: catalog says 12.54 GB, the Hub file listing sums to 2.29 GB / points: catalog says 1B-10B rows, metadata gives 497.85k points (497.85k rows) / frequency: catalog says 1D, metadata says 5min (implied by span and series length (span / (length - 1) = 297.6 s)) |
| 588 | netop/gotsf-ds [default] | viewer+footers | subset text not matched on the Hub: default |
| 589 | nevirim/wikipedia-attention-13lang | viewer+footers | size: catalog says 0.87 GB, the Hub file listing sums to 296.90 MB |
| 590 | nju-zhangsq/RMISC [AliCar] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 591 | nju-zhangsq/RMISC [AzurePublicDatasetV1] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 592 | nju-zhangsq/RMISC [EbayServer] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 593 | nju-zhangsq/RMISC [MiniApp] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 340416, the footers give 388800 |
| 594 | nju-zhangsq/RMISC [NAB] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB |
| 595 | nju-zhangsq/RMISC [ServerMachineDataset] | viewer+footers | size: catalog says 2592.41 GB, the Hub file listing sums to 816.24 GB / points: the folder's meta.json states 21991847, the footers give 26210985 |
| 596 | paperswithbacktest/Stocks-Quarterly-WikipediaViews | gated | gated, no access / 2 footer(s) not readable, first: data/train-00000-of-00002.parquet: HTTP 403 GatedRepo Access to dataset paperswithbacktest/Stocks-Quarterly-WikipediaViews is restricted and you are not in the  |
| 597 | pyToshka/attacks-daily | viewer+footers | size: catalog says 35.31 GB, the Hub file listing sums to 111.64 MB |
| 598 | Salesforce/cloudops_tsf [alibaba_cluster_trace_2018] | files_only | size: catalog says 34.64 GB, the Hub file listing sums to 16.07 GB / the dataset viewer does not work for this repo |
| 599 | Salesforce/cloudops_tsf [borg_cluster_data_2011] | files_only | size: catalog says 34.64 GB, the Hub file listing sums to 16.07 GB / the dataset viewer does not work for this repo |
| 600 | Salesforce/lotsa_data [alibaba_cluster_trace_2018] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 601 | Salesforce/lotsa_data [borg_cluster_data_2011] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 602 | Salesforce/lotsa_data [extended_web_traffic_with_missing] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 603 | Salesforce/lotsa_data [kaggle_web_traffic_weekly] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 604 | Salesforce/lotsa_data [wiki-rolling_nips] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |
| 605 | Sbaiiiiii/looked-up | viewer+footers | size: catalog says 9.87 GB, the Hub file listing sums to 6.21 GB |
| 605 | Sbaiiiiii/looked-up [hourly] | viewer+footers | size: catalog says 9.87 GB, the Hub file listing sums to 6.21 GB |
| 605 | Sbaiiiiii/looked-up [sitelinks] | viewer+footers | size: catalog says 9.87 GB, the Hub file listing sums to 6.21 GB / points: catalog says 1B-10B rows, metadata gives 57.44M points (57.44M rows) |
| 606 | Shashkovich/Telecommunication_SMS_time_series | files_only | size: catalog says 0.01 GB, the Hub file listing sums to 2.08 MB / the dataset viewer does not work for this repo |
| 609 | tardellirs/paper-pulse-data | footers | size: catalog says 5.03 GB, the Hub file listing sums to 16.54 MB |
| 609 | tardellirs/paper-pulse-data [papers] | footers | size: catalog says 5.03 GB, the Hub file listing sums to 16.54 MB |
| 609 | tardellirs/paper-pulse-data [purges] | footers | size: catalog says 5.03 GB, the Hub file listing sums to 16.54 MB / points: catalog says 1M-10M rows, metadata gives 36 points (9 rows) |
| 609 | tardellirs/paper-pulse-data [series] | footers | size: catalog says 5.03 GB, the Hub file listing sums to 16.54 MB |
| 610 | venvoo/openrouter-uptime | viewer+footers | size: catalog says 9.59 GB, the Hub file listing sums to 137.62 MB |
| 610 | venvoo/openrouter-uptime [incidents] | viewer+footers | size: catalog says 9.59 GB, the Hub file listing sums to 137.62 MB / points: catalog says 10M-100M rows, metadata gives 205.13k points (102.57k rows) |
| 610 | venvoo/openrouter-uptime [models] | viewer+footers | size: catalog says 9.59 GB, the Hub file listing sums to 137.62 MB / points: catalog says 10M-100M rows, metadata gives 932 points (466 rows) |
| 610 | venvoo/openrouter-uptime [readings] | viewer+footers | size: catalog says 9.59 GB, the Hub file listing sums to 137.62 MB |
| 612 | Datadog/BOOM [(whole repo: 2,807 folders named ds-<id>-<freq>, e.g. ds-10-T, ds-1030-5T)] | files_only+dataset_info | size: catalog says 7.17 GB, the Hub file listing sums to 2.81 GB |
| 613 | YahooResearch/ydata-labeled-time-series-anomalies-v1_0 | files_only | format: catalog says parquet, the scope has no parquet file (csv) |
| 614 | Salesforce/cloudops_tsf [azure_vm_traces_2017] | files_only | size: catalog says 34.64 GB, the Hub file listing sums to 16.07 GB / the dataset viewer does not work for this repo |
| 615 | Salesforce/lotsa_data [azure_vm_traces_2017] | viewer+footers | size: catalog says 1930.59 GB, the Hub file listing sums to 924.72 GB |

## 11. What could not be determined from metadata

Per dataset: the values that are unknown or estimated, and why. A dataset that is not listed has every value exact.

| dataset | not determined / estimated |
|---|---|
| alaminxpro/bd-food-prices | 210 non-parquet data files in the scope (2.33 MB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 183 parquet files; counts scaled by bytes (x2.47) |
| alexroz/CarbonFluxBench | 8 non-parquet data files in the scope (256.35 KB) are not in the row and point counts / n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / date range: estimated / frequency: unknown |
| dar4huggingface/sniim-mexico-sample | 1 non-parquet data files in the scope (23.62 KB) are not in the row and point counts / n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / frequency: unknown |
| DHIVAKARNAIDUSUNKARI/mandi-india-daily-prices-2001-2026 | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / frequency: unknown |
| electricsheepafrica/africa-wfp-food-prices-for-uganda | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / frequency: unknown |
| electricsheepafrica/wfp-nigeria-food-prices | 2 non-parquet data files in the scope (9.24 KB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / frequency: unknown |
| Emulated-Inc/sci-county-crop-yield-new-year | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| food-ai-nexus/global-food-safety-incidents | layout is a rule-based guess (R2) / series length: estimated / date range: estimated / frequency: unknown / missing share: unknown |
| globalsquiddatabase/Tuna-Fisheries-Dataset | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| hesandism/colombo-tea-auction-prices | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / date range: unknown / frequency: unknown |
| judicialmind/CropBench | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| judicialmind/MandiBench [series/all.parquet] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| National-Digital/nt-crocodile-captures | 1 non-parquet data files in the scope (3.24 KB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / frequency: unknown |
| National-Digital/qld-shark-control-catch-numbers | 1 non-parquet data files in the scope (3.41 KB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / date range: estimated / frequency: unknown |
| notadib/usa-corn-belt-crop-yield | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| Project-AgML/usda_nass_crop_statistics | 1 non-parquet data files in the scope (25.94 MB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: unknown / date range: estimated / frequency: unknown |
| Roshangupta007/YieldFormer-DE-data | 4 non-parquet data files in the scope (97.38 KB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / date range: estimated / frequency: unknown |
| Salesforce/lotsa_data [beijing_air_quality] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [china_air_quality] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [sunspot_with_missing] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| taziksh/itrdb-tree-rings | 1 non-parquet data files in the scope (452 B) are not in the row and point counts / n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / date range: estimated / frequency: unknown |
| taziksh/usgs-earthquakes | 1 non-parquet data files in the scope (633 B) are not in the row and point counts / layout is a rule-based guess (R3) / n_series: unknown / n_points: estimated / series length: unknown / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 66 parquet files; counts scaled by bytes (x1.36) |
| teamSynapse/sri-lanka-weekly-vegetable-retail-farmer-price-dataset | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / date range: estimated / frequency: estimated |
| yuiseki/kyoto-bear-sighting | 4 non-parquet data files in the scope (2.14 MB) are not in the row and point counts / n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / frequency: unknown |
| thuml/UTSD [AustraliaRainfall / BeijingPM25Quality / BenzeneConcentration] | date range: only series start times are in the metadata (no end time) / n_series: estimated / n_points: estimated / series length: estimated / date range: unknown / missing share: estimated / the viewer converted only part of the dataset to parquet |
| yuiseki/faostat [raw/2026-10-04] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| Aduitya9/Aduitya_dextra-aqi-india | 122 non-parquet data files in the scope (86.54 MB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / n_points: estimated / series length: unknown / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 113 parquet files; counts scaled by bytes (x1.82) |
| AkshayKarthick/india-weather-air-quality | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / date range: estimated / frequency: unknown |
| CMU-OSPO/COVID_air_quality_RAMP_datasets | 1 non-parquet data files in the scope (2.28 KB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: unknown / frequency: unknown |
| cosuleabianca/eea-pm25-forecasting | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| danielrosehill/Jerusalem-Air-Quality-Shabbat | 85 non-parquet data files in the scope (170.84 KB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: unknown / frequency: unknown |
| dmariaa70/METRAQ-Air-Quality | layout is a rule-based guess (R4) / n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / the viewer converted only part of the dataset to parquet |
| haidang2405/aqshift-us-leakage-safe-air-quality | 4 non-parquet data files in the scope (30.99 KB) are not in the row and point counts / layout is a rule-based guess (R3) / n_series: unknown / series length: unknown / frequency: unknown |
| hsilvosa/safecast-radiation | 5 non-parquet data files in the scope (49.03 KB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 266 parquet files; counts scaled by bytes (x5.4) |
| khajlk/copenhagen-air-quality-hourly | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / frequency: estimated |
| meganariley/daily-aqi | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / frequency: unknown |
| National-Digital/act-air-quality | 1 non-parquet data files in the scope (3.03 KB) are not in the row and point counts / layout is a rule-based guess (R4) / series length: estimated / frequency: unknown |
| National-Digital/vic-air-quality-hourly | 1 non-parquet data files in the scope (3.12 KB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / frequency: unknown |
| nju-zhangsq/RMISC [PM25FiveCities] | layout is a rule-based guess (R4) / series length: estimated / date range: estimated / frequency: estimated |
| nju-zhangsq/RMISC [USAirPollution] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [XiamenAirQuality] | layout is a rule-based guess (R4) / series length: estimated / date range: estimated / frequency: unknown |
| Skywalker1910/skynet-aqi-panel | 24 non-parquet data files in the scope (377.73 KB) are not in the row and point counts / layout is a rule-based guess (R4) / series length: estimated / frequency: unknown |
| steven0226/air-quality | 1 non-parquet data files in the scope (11.95 KB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / frequency: unknown |
| Envirocatalysts/hourly_pollutants_data | layout is a rule-based guess (R5) / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 1841 parquet files; counts scaled by bytes (x43.3) |
| meganariley/us-air-quality-hourly | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| xuxing123/AirQualityBench | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| yuiseki/edgar-ghg-countries | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| aaron-freedman/banking-concentration-data | 9 non-parquet data files in the scope (153.86 MB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: unknown / date range: estimated / frequency: unknown |
| abel-lab/finance1k | 7 non-parquet data files in the scope (22.04 KB) are not in the row and point counts / n_series: estimated / series length: estimated / date range: estimated / frequency: unknown |
| aikstockdata/korea-equity-daily | 9 non-parquet data files in the scope (157.79 MB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: unknown / date range: estimated / frequency: unknown |
| arabalika/us-rent-history | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| autogluon/chronos_datasets [exchange_rate] | frequency: estimated |
| autogluon/chronos_datasets [monash_cif_2016] | frequency: estimated |
| autogluon/chronos_datasets [monash_m1_monthly, monash_m3_monthly] | series length: estimated |
| autogluon/chronos_datasets [monash_nn5_weekly, nn5] | series length: estimated |
| AYUSHKHAIRE/all-stock-market-data-daily-updates | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| Bayanat-Labs/arab-trade-exposure | 42 non-parquet data files in the scope (739.44 KB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / n_points: estimated / series length: unknown / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 121 parquet files; counts scaled by bytes (x2.49) |
| CarlosSilva1/us500-ticks [default] | n_series: 50 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 95 parquet files; counts scaled by bytes (x1.73) |
| empulse-ai/empulse-street-fx-index | layout is a rule-based guess (R5) / date range: estimated / frequency: estimated |
| HexQuant/Stocks-Daily-Price | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / date range: estimated / frequency: unknown |
| jwigginton/timeseries-1mn-sp500 [default] | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / frequency: unknown |
| kawsersikder/bangladesh-stock-market-dataset | n_series: 2 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / date range: estimated / frequency: unknown |
| misikoff/zillow [days_on_market] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| misikoff/zillow [for_sale_listings] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| misikoff/zillow [home_values] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| misikoff/zillow [rentals] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| misikoff/zillow [sales] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| MonitorBankRates/us-bank-rate-averages | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| National-Digital/rba-money-market-daily | 1 non-parquet data files in the scope (4.11 KB) are not in the row and point counts / layout is a rule-based guess (R5) / frequency: estimated |
| nju-zhangsq/RMISC [AMarketChina] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [ChinaMinuteStock] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown |
| nju-zhangsq/RMISC [CSI500] | layout is a rule-based guess (R4) / series length: estimated / date range: estimated |
| nju-zhangsq/RMISC [NIFTYStock] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [SP500] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [StockFactorsCleaned] | layout is a rule-based guess (R4) / series length: estimated / date range: estimated / frequency: unknown |
| nju-zhangsq/RMISC [StockMarketData] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [TushareETFDaily] | layout is a rule-based guess (R3) / series length: estimated / date range: estimated / frequency: unknown |
| nju-zhangsq/RMISC [TushareIndexDaily] | layout is a rule-based guess (R3) / series length: estimated / date range: estimated / frequency: unknown |
| nju-zhangsq/RMISC [TushareStockDaily] | layout is a rule-based guess (R3) / series length: estimated / date range: estimated / frequency: unknown |
| nju-zhangsq/RMISC [TushareStockWeekly] | layout is a rule-based guess (R3) / series length: estimated / date range: estimated / frequency: unknown |
| nju-zhangsq/RMISC [UKEconomy] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| paperswithbacktest/Forex-Daily-Price [default] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| paperswithbacktest/Indices-Daily-Price [default] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| paperswithbacktest/Stocks-Daily-Price [default] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| perctrix/StockChina-Minute | n_series: 10 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R6) / n_points: estimated / series length: estimated / date range: unknown / frequency: unknown / missing share: estimated / the viewer converted only part of the dataset to parquet |
| prasad-gade05/india-upi-ecosystem-2018-2025 | layout is a rule-based guess (R4) / n_series: estimated / series length: unknown / date range: estimated / frequency: unknown |
| prospex-ch/swiss-commercial-registry-monthly-panel | 3 non-parquet data files in the scope (777.19 KB) are not in the row and point counts / layout is a rule-based guess (R3) / n_series: unknown / series length: unknown / frequency: unknown |
| Rogendo/forex-dukas-copy [default] | 210 non-parquet data files in the scope (6.72 GB) are not in the row and point counts / n_series: 10 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / frequency: unknown |
| Rrishab/OHLCV-1m [default] | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / n_points: estimated / series length: unknown / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 411 parquet files; counts scaled by bytes (x8.17) |
| Salesforce/lotsa_data [cif_2016_12] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [godaddy] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [m1_monthly] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [nn5_daily_with_missing] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| sevriukov/bank101 | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| showlcw/china-cpi-ppi-pmi | layout is a rule-based guess (R5) / series length: estimated / date range: estimated / frequency: unknown |
| SII-WANGZJ/Polymarket_data [trades.parquet] | layout is a rule-based guess (R3) / n_series: unknown / series length: unknown / date range: estimated / frequency: unknown |
| thanna94/us-building-permits | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / date range: estimated / frequency: unknown |
| thecharttruth/etf-data [prices] | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / frequency: estimated |
| THULab/xauusd [default] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| tidy-finance/factor-library | layout is a rule-based guess (R4) / n_series: estimated / n_points: estimated / series length: estimated / date range: unknown / frequency: unknown / missing share: estimated / footers read for 50 of 8208 parquet files; counts scaled by bytes (x164) |
| Traders-Lab/TroveLedger [daily / hourly / 1-minute files] | 17 non-parquet data files in the scope (1.36 MB) are not in the row and point counts / n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / n_points: estimated / series length: unknown / date range: unknown / frequency: unknown / missing share: estimated / footers read for 50 of 4301 parquet files; counts scaled by bytes (x97.3) |
| trentmkelly/polymarket_historical_data [market_snapshots] | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / n_points: estimated / series length: unknown / date range: unknown / frequency: unknown / missing share: estimated / footers read for 50 of 28024 parquet files; counts scaled by bytes (x562) |
| wrpbr/pncp-compras-modalidade-dia-municipio | layout is a rule-based guess (R6) / date range: unknown / frequency: unknown |
| xxparthparekhxx/indian-stock-market-minute-data | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / frequency: estimated |
| yadavsaurabh/us-stocks-and-search-trends | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / date range: estimated / frequency: unknown |
| yugalinks/import-export-market-demand-v1 | 1 non-parquet data files in the scope (1.49 KB) are not in the row and point counts / n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / frequency: unknown / missing share: estimated |
| yuiseki/ilostat | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| yuiseki/maddison-project | 13 non-parquet data files in the scope (23.93 MB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / date range: estimated / frequency: unknown |
| yuiseki/pwt | 91 non-parquet data files in the scope (213.24 MB) are not in the row and point counts / layout is a rule-based guess (R4) / series length: estimated / date range: estimated / frequency: estimated |
| yuiseki/undata-sdmx | 679 non-parquet data files in the scope (498.89 MB) are not in the row and point counts / layout is a rule-based guess (R3) / n_series: estimated / series length: estimated / date range: estimated |
| ashwingopalsamy/central-bank-policy-rates | 26 non-parquet data files in the scope (4.70 MB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / date range: estimated / frequency: unknown |
| datonic/world_development_indicators [default] | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / date range: estimated / frequency: unknown |
| kavelaltd/historysaid-global-economic | 13 non-parquet data files in the scope (187.77 MB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / date range: estimated / frequency: estimated |
| rogerdehe/klines-binance [spot, futures] | n_series: 50 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / n_points: estimated / series length: estimated / date range: estimated / frequency: estimated / missing share: estimated / footers read for 50 of 11212 parquet files; counts scaled by bytes (x231) |
| rogerdehe/klines-bybit [futures] | n_series: 50 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / n_points: estimated / series length: estimated / date range: estimated / frequency: estimated / missing share: estimated / footers read for 50 of 6304 parquet files; counts scaled by bytes (x93.5) |
| rogerdehe/klines-okx [futures] | n_series: 50 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / n_points: estimated / series length: estimated / date range: estimated / frequency: estimated / missing share: estimated / footers read for 50 of 3438 parquet files; counts scaled by bytes (x68.7) |
| tradecatlabs/binance-futures-ohlcv-2018-2026 [candles_1m] | layout is a rule-based guess (R4) / n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / the viewer converted only part of the dataset to parquet |
| zalizedata/global-macro-trade-indicators-dataset | 9 non-parquet data files in the scope (1.02 GB) are not in the row and point counts / n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / date range: estimated / frequency: unknown |
| AllRatesToday/central-bank-exchange-rates | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown / viewer /parquet: HTTP 404 ExternalAuthenticatedError The dataset does not exist, or is not accessible with the current credentials (private or gated). Please check the spelling of the dataset name or retry with other authentication credentials. / viewer /info: HTTP 404 ExternalAuthenticatedError The dataset does not exist, or is not accessible with the current credentials (private or gated). Please check the spelling of the dataset name or retry with other authentication credentials. |
| ausapis/rba-daily-exchange-rates | layout is a rule-based guess (R5) / date range: estimated / frequency: estimated |
| bkoyuncu/MacroEcon [one config per country (64)] | layout is a rule-based guess (R6) / n_series: estimated / n_points: estimated / series length: estimated / date range: unknown / frequency: unknown / missing share: estimated / footers read for 50 of 65 parquet files; counts scaled by bytes (x1.3) |
| JrVillabona/resonance-data | 12 non-parquet data files in the scope (593.40 KB) are not in the row and point counts / layout is a rule-based guess (R3) / n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / footers read for 50 of 967 parquet files; counts scaled by bytes (x18.2) |
| paperswithbacktest/Commodities-Daily-Price [default] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| Salesforce/lotsa_data [tourism_monthly] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| SemantaAI/semantaai-fx-majors | 10 non-parquet data files in the scope (73.40 KB) are not in the row and point counts / n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / n_points: estimated / series length: unknown / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 259 parquet files; counts scaled by bytes (x6.32) |
| syncster-data/economic-data | 75 non-parquet data files in the scope (3.89 GB) are not in the row and point counts / n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / n_points: estimated / series length: unknown / date range: estimated / missing share: estimated / footers read for 50 of 44123 parquet files; counts scaled by bytes (x2.98e+03) |
| yadavsaurabh/macro-markets | layout is a rule-based guess (R4) / series length: estimated / date range: estimated / frequency: unknown |
| ai-iot/EnergyBench [DGS] | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / date range: estimated / frequency: estimated |
| ai-iot/EnergyBench [DTH] | n_series: 2 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / date range: estimated / frequency: estimated |
| ai-iot/EnergyBench [GoiEner] | layout is a rule-based guess (R5) / series length: estimated / date range: estimated / frequency: estimated |
| ai-iot/EnergyBench [iFlex] | layout is a rule-based guess (R5) / frequency: estimated |
| ai-iot/EnergyBench [LCL] | n_series: 2 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / date range: estimated / frequency: estimated |
| ai-iot/EnergyBench [NEEA] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| ai-iot/EnergyBench [Norwegian] | layout is a rule-based guess (R5) / date range: estimated / frequency: estimated |
| ai-iot/EnergyBench [Prayas, CEEW] | n_series: 2 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / series length: estimated / date range: estimated / frequency: estimated |
| ai-iot/EnergyBench [RSL] | layout is a rule-based guess (R2) / n_series: unknown / series length: unknown / frequency: estimated / missing share: unknown |
| ai-iot/EnergyBench [SAVE] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| ai-iot/EnergyBench [SGSC] | n_series: 2 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / date range: estimated / frequency: estimated |
| ai-iot/EnergyBench [UKST] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| ai-iot/EnergyBench [UNICON] | n_series: 2 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / frequency: estimated |
| akderekaan/de-power-forecast-data [entsoe/a75] | n_series: 50 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 175 parquet files; counts scaled by bytes (x3.5) |
| alwaysbyx/Bear-Data | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| andrewlee1807/Gyeonggi | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| autogluon/chronos_datasets [monash_london_smart_meters] | frequency: unknown |
| autogluon/chronos_datasets [wind_farms_hourly, wind_farms_daily] | series length: estimated |
| Battery-Life/BatteryLife_Raw [Zn-ion, Na-ion, CALB] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| bsebench-org/nasa-rw-2014 [partitioned by cell_id] | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / date range: unknown / frequency: unknown |
| bsebench-org/warwick-second-life-dm-2025-raw [raw mirror (746 csv, 309 mat)] | date range: only series start times are in the metadata (no end time) / series length: estimated / date range: unknown / frequency: unknown / missing share: unknown |
| daniel-dona/ree-data [default] | n_series: 2 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / frequency: estimated |
| EDS-lab/gas-demand [demand+weather+metadata] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| Emulated-Inc/battery-capacity-trajectories [Dataset_1_NCA / 2_NCM / 3_NCM_NCA zips] | layout is a rule-based guess (R5) / n_points: estimated / series length: estimated / date range: unknown / frequency: unknown / missing share: estimated / footers read for 1 of 9 parquet files; counts scaled by bytes (x127) |
| fidel-ts/Bear_room | 22 non-parquet data files in the scope (1.89 GB) are not in the row and point counts / n_series: 49 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / frequency: estimated / missing share: estimated / footers read for 50 of 81 parquet files; counts scaled by bytes (x1.64) |
| fidel-ts/California_ISO | 124 non-parquet data files in the scope (371.81 MB) are not in the row and point counts / layout is a rule-based guess (R5) / series length: estimated / frequency: estimated |
| fidel-ts/Canada_photovoltaics_plants | 81 non-parquet data files in the scope (160.01 MB) are not in the row and point counts / n_series: 9 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / frequency: unknown |
| gft/ttm4hvac-source-all-train | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / date range: estimated / frequency: estimated |
| gtfierro/mortar [hive-partitioned by uuid] | 1 non-parquet data files in the scope (21.70 MB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 22561 parquet files; counts scaled by bytes (x344) |
| hackermango/HyderabadUniversity_HVAC_Dataset | layout is a rule-based guess (R5) / date range: estimated / frequency: estimated |
| happyman11/Karnataka-SLDC [preprocessed] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown / viewer /parquet: HTTP 404 ExternalAuthenticatedError The dataset does not exist, or is not accessible with the current credentials (private or gated). Please check the spelling of the dataset name or retry with other authentication credentials. / viewer /info: HTTP 404 ExternalAuthenticatedError The dataset does not exist, or is not accessible with the current credentials (private or gated). Please check the spelling of the dataset name or retry with other authentication credentials. |
| hsilvosa/omie-electricity | 5 non-parquet data files in the scope (2.63 KB) are not in the row and point counts / layout is a rule-based guess (R4) / series length: estimated / frequency: unknown |
| J-Marcos-GS/brazil-solar-generation | layout is a rule-based guess (R6) / date range: unknown / frequency: unknown |
| Jainam03/RESCAST-100k-ECOBEE | 2 non-parquet data files in the scope (268.08 KB) are not in the row and point counts / n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / n_points: estimated / series length: unknown / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 955 parquet files; counts scaled by bytes (x19.3) |
| Jainam03/RESCAST-100k-HEAPO [timeseries_data] | 2 non-parquet data files in the scope (249.39 KB) are not in the row and point counts / n_series: 50 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 1407 parquet files; counts scaled by bytes (x31.9) |
| Jainam03/RESCAST-100k-IDEAL | 2 non-parquet data files in the scope (56.09 KB) are not in the row and point counts / n_series: 49 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 256 parquet files; counts scaled by bytes (x5.14) |
| Jainam03/RESCAST-100k-NEST | 2 non-parquet data files in the scope (3.74 KB) are not in the row and point counts / layout is a rule-based guess (R5) / n_series: estimated / series length: estimated / date range: unknown / frequency: unknown |
| Jainam03/RESCAST-100k-REFIT | 2 non-parquet data files in the scope (12.87 KB) are not in the row and point counts / n_series: 20 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / n_series: estimated / series length: estimated / frequency: unknown |
| jtupayac/EV_Charging_demand [Se_*.csv sessions (+ INRIX traffic Readings.csv, weather)] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| judicialmind/BharatEnergy-Bench | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| judicialmind/BharatEnergy-Data [generation_daily.parquet] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| kevykibbz/wind-turbine-scada-data-for-early-fault-detection [Wind Farm A/B/C] | layout is a rule-based guess (R4) / n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / the viewer converted only part of the dataset to parquet |
| michealsmitch/lithium-ion-cell-capacity-grading-process-curves [train] | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / n_points: estimated / series length: unknown / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 76 parquet files; counts scaled by bytes (x1.5) |
| Narendersingh007/india-solar-benchmark-dataset | 1 non-parquet data files in the scope (507.41 KB) are not in the row and point counts / n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / frequency: unknown |
| netop/5G-Network-Energy-Consumption [ECstat, CLstat, BSinfo] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown / viewer /parquet: HTTP 404 ExternalAuthenticatedError The dataset does not exist, or is not accessible with the current credentials (private or gated). Please check the spelling of the dataset name or retry with other authentication credentials. / viewer /info: HTTP 404 ExternalAuthenticatedError The dataset does not exist, or is not accessible with the current credentials (private or gated). Please check the spelling of the dataset name or retry with other authentication credentials. |
| nju-zhangsq/RMISC [ApplianceEnergy] | layout is a rule-based guess (R4) / series length: estimated / date range: estimated |
| nju-zhangsq/RMISC [BritainCoal] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [BuildingsBenchRealCSV] | layout is a rule-based guess (R4) / series length: estimated / date range: estimated / frequency: unknown |
| nju-zhangsq/RMISC [HouseholdPower] | layout is a rule-based guess (R4) / series length: estimated / date range: estimated / frequency: estimated |
| nju-zhangsq/RMISC [LBNL] | layout is a rule-based guess (R4) / series length: estimated / date range: estimated / frequency: unknown |
| nju-zhangsq/RMISC [MZVAV] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [OccupancyDetection] | layout is a rule-based guess (R4) / series length: estimated / date range: estimated / frequency: unknown |
| nju-zhangsq/RMISC [OPSD-Household] | layout is a rule-based guess (R4) / series length: estimated / date range: estimated / frequency: unknown |
| nju-zhangsq/RMISC [OPSD-PV-Wind] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [OPSD-When2Heat] | layout is a rule-based guess (R4) / series length: estimated / date range: estimated / frequency: unknown |
| nju-zhangsq/RMISC [Pvdaq] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown |
| nju-zhangsq/RMISC [RoomOccupancy] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown |
| nju-zhangsq/RMISC [ShellHackathon] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [SmartMeterAus30m] | layout is a rule-based guess (R4) / series length: estimated |
| nju-zhangsq/RMISC [SmartMeterAus60m] | layout is a rule-based guess (R4) / series length: estimated / frequency: unknown |
| nju-zhangsq/RMISC [SmartMeterUK30m] | layout is a rule-based guess (R4) / series length: estimated |
| nju-zhangsq/RMISC [SmartMeterUK60m] | layout is a rule-based guess (R4) / series length: estimated / frequency: unknown |
| nju-zhangsq/RMISC [Solar4Seconds] | layout is a rule-based guess (R4) / series length: estimated / date range: estimated |
| nju-zhangsq/RMISC [TetuanPowerConsumption] | layout is a rule-based guess (R4) / series length: estimated / date range: estimated |
| nju-zhangsq/RMISC [UK-DALE] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [WindElec] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown |
| nju-zhangsq/RMISC [WindPower4secs] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown |
| open-predico/benchmark | n_series: 39 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 54 parquet files; counts scaled by bytes (x1.07) |
| openclimatefix/uk_pv [30min.parquet] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| openclimatefix/uk_pv [5min.parquet / 2min.parquet] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| OpenSTEF/liander2024-energy-forecasting-benchmark [load_measurements, weather_forecasts] | n_series: 25 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 110 parquet files; counts scaled by bytes (x2.19) |
| OpenSynth/TUDelft-Electricity-Consumption-1.0 [default] | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / frequency: estimated |
| PowerZooJax/PowerZooDataset [ausgrid_zone_substation] | layout is a rule-based guess (R4) / series length: estimated / frequency: estimated |
| PowerZooJax/PowerZooDataset [gb_neso_demand, gb_gen_by_type, gb_market_mid, gb_forecast_actual_demand] | layout is a rule-based guess (R5) / series length: estimated / frequency: estimated |
| rebase-energy/heftcom2024 [default] | 1 non-parquet data files in the scope (1.80 KB) are not in the row and point counts / layout is a rule-based guess (R5) / series length: estimated / frequency: unknown |
| rebase-energy/nve-windpower-data [nve-windpower-timeseries.csv] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| Rif-SQL/uk-electricity-generation-mix-2019-2026 | 8 non-parquet data files in the scope (25.70 MB) are not in the row and point counts / layout is a rule-based guess (R5) / frequency: estimated |
| Salesforce/lotsa_data [bdg-2_bear] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [bdg-2_fox] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [bdg-2_panther] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [bdg-2_rat] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [borealis] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [ideal] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [lcl] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [london_smart_meters_with_missing] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [residential_load_power] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [residential_pv_power] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [sceaux] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [smart] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [solar_power] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [wind_farms_with_missing] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [wind_power] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| SamuelM0422/Hourly-Electricity-Demand-Brazil-Dataset | layout is a rule-based guess (R4) / series length: estimated / frequency: estimated |
| SamuelM0422/ONS-Capacity-Factor-Dataset-Full | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 178 parquet files; counts scaled by bytes (x3.53) |
| skforecast/ashrae_daily | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / frequency: estimated |
| solarbench/SKIPPD [ground, ground_no_sky_img, ERA5, GFS] | n_series: 2 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / series length: estimated / frequency: unknown |
| yzzpz/Charged-data [10 csv per city (some in .rar)] | n_series: unknown / series length: unknown / date range: unknown / frequency: unknown / missing share: unknown |
| ai-iot/EnergyBench [real subsets overall (Dataset_V0.0)] | 49 non-parquet data files in the scope (2.63 MB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 49 of 1896 parquet files; counts scaled by bytes (x77.1) |
| alinurdin/time-series-benchmark [eia_hourly, madrid_15min, webtris_15min, usgs_daily, usgs_continuous, hydat_daily, bom_daily, caravan_daily, isd_hourly, swissmetnet_10min, ghcnd_daily] | layout is a rule-based guess (R4) / n_series: estimated / series length: unknown |
| marplan6/batgrad [pozzato-2022 (real) vs synthetic-pozzato-2022-m50t (drop)] | n_series: 7 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / n_series: estimated / series length: estimated / date range: unknown / frequency: unknown |
| zalizedata/us-grid-hourly-power-plants-dataset | 2 non-parquet data files in the scope (527.39 MB) are not in the row and point counts / layout is a rule-based guess (R5) / n_series: estimated / series length: estimated / date range: estimated / frequency: unknown |
| alinurdin/time-series-benchmark [eia_hourly] | layout is a rule-based guess (R4) / series length: estimated |
| alinurdin/time-series-benchmark [opsd_load] | layout is a rule-based guess (R4) / series length: estimated / frequency: unknown |
| autogluon/chronos_datasets [monash_australian_electricity] | frequency: estimated |
| autogluon/chronos_datasets [solar, solar_1h] | series length: estimated |
| autogluon/chronos_datasets_extra [spanish_energy_and_weather] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| DAG-UPB/TS-Arena-Archive [default] | layout is a rule-based guess (R4) / n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 46352 parquet files; counts scaled by bytes (x967) |
| EDS-lab/electricity-demand [demand+weather+metadata] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| EDS-lab/pv-generation [generation+weather+metadata] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| fidel-ts/Germany_Renewable_Energy_Grid | 459 non-parquet data files in the scope (1.35 GB) are not in the row and point counts / n_series: 8 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / date range: estimated / frequency: unknown |
| hsilvosa/entsoe-observatory [8 tables] | 9 non-parquet data files in the scope (17.78 KB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 2593 parquet files; counts scaled by bytes (x39.7) |
| louey9999/wattcast-data [default] | 7 non-parquet data files in the scope (5.56 MB) are not in the row and point counts / n_series: 13 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / n_series: estimated / series length: estimated / frequency: unknown |
| nju-zhangsq/RMISC [OPSD] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| OpenSynth/AI4OPT-TS4PS-RTE7000 [tables (lines, generators, loads, ...)] | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / n_points: estimated / series length: unknown / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 252 parquet files; counts scaled by bytes (x5.14) |
| OptimalSolution/databank-ember [19 tables (generation monthly/yearly, prices, capacity, ...)] | 9 non-parquet data files in the scope (52.67 KB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / date range: estimated / frequency: unknown |
| qingshufan/GA-EVLRU | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| Salesforce/lotsa_data [australian_electricity_demand] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| THULab/m2oe2_timeseries [8 tables (building_train, building weather, household consumption, residential power usage + weather, solar + weather, Spanish energy + weather)] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| YlanHARRAULT/rte-eco2mix-regional-2025 | layout is a rule-based guess (R4) / n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / frequency: estimated / missing share: estimated / footers read for 50 of 144 parquet files; counts scaled by bytes (x2.88) |
| abmallick/heart-breath-sleep-stage-dataset | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / date range: unknown / frequency: unknown |
| ajto/GoldenCheetah | layout is a rule-based guess (R4) / n_series: estimated / series length: unknown / date range: estimated / frequency: unknown |
| bsaenz/dreamt | n_series: 10 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / n_points: estimated / series length: estimated / date range: unknown / frequency: unknown / missing share: estimated / the viewer converted only part of the dataset to parquet |
| CRUISEResearchGroup/CGM-JEPA-Pretraining | layout is a rule-based guess (R5) / date range: estimated / frequency: estimated |
| EPI-Eval/canada-fluwatch | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / frequency: unknown |
| EPI-Eval/cdc-nrevss-rsv | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / frequency: unknown |
| EPI-Eval/cdc-nssp | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / frequency: unknown |
| EPI-Eval/delphi-flusurv | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / frequency: unknown |
| EPI-Eval/nhsn-hrd | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / frequency: unknown |
| glucofmbench/GlucoFM-Bench | date range: only series start times are in the metadata (no end time) / 1 non-parquet data files in the scope (13.14 KB) are not in the row and point counts / series length: estimated / date range: unknown / frequency: unknown |
| HHS-Official/nndss-weekly-data | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / date range: unknown / frequency: unknown |
| HHS-Official/sars-cov-2-variant-proportions | layout is a rule-based guess (R5) / date range: unknown / frequency: unknown |
| HHS-Official/weekly-counts-of-deaths-from-all-causes-by-jurisdi | layout is a rule-based guess (R5) / date range: estimated / frequency: unknown |
| HHS-Official/weekly-provisional-counts-of-deaths-by-state-and-s | layout is a rule-based guess (R5) / date range: estimated / frequency: unknown |
| HHS-Official/weekly-rates-of-laboratory-confirmed-rsv-hospitali | layout is a rule-based guess (R4) / series length: estimated / date range: estimated / frequency: unknown |
| HHS-Official/weekly-united-states-covid-19-hospitalization-metr | layout is a rule-based guess (R4) / series length: estimated / date range: estimated / frequency: estimated |
| judicialmind/HMIS-Bench | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| judicialmind/IndiaDiseaseWatch [panel.parquet] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| LVinaud/arboviral-alerta-dados | 170 non-parquet data files in the scope (963.83 MB) are not in the row and point counts / layout is a rule-based guess (R6) / n_points: estimated / series length: estimated / date range: unknown / frequency: unknown / missing share: estimated / footers read for 50 of 94 parquet files; counts scaled by bytes (x1.62) |
| National-Digital/au-ed-patients-seen-on-time | 1 non-parquet data files in the scope (2.81 KB) are not in the row and point counts / layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| National-Digital/au-elective-surgery-waiting-times-by-specialty | 1 non-parquet data files in the scope (3.32 KB) are not in the row and point counts / layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [HungarianChickenpoxCases] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| Ohja-ai/rki-grippe-web-wochenbericht-2026-02-06 | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| pedrolucassantanaf/dengue-tcc2-data | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| rricc22/endomondo-hr-prediction-v2 | date range: only series start times are in the metadata (no end time) / series length: estimated / date range: unknown / frequency: unknown |
| ruiqil/SpatialEpiBench [AUcase, CAcase, CANpositivity, CHNGinpatient, CHNGoutpatient, CPRadmissions, DVcli, HHShosp] | layout is a rule-based guess (R4) / n_series: estimated / series length: unknown / date range: estimated / frequency: unknown |
| Salesforce/lotsa_data [cdc_fluview_ilinet] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [cdc_fluview_who_nrevss] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [project_tycho] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| thuml/UTSD [TDBrain] | date range: only series start times are in the metadata (no end time) / n_series: estimated / n_points: estimated / series length: estimated / date range: unknown / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Uvathe/wellborn-pregnancy-wearable | n_series: unknown / series length: unknown / date range: unknown / frequency: unknown / missing share: unknown |
| DragonbornRay/FluDataset | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / date range: estimated / frequency: unknown |
| EPI-Eval/covid-tracking-project | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / frequency: unknown |
| EPI-Eval/nyt-covid | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / frequency: unknown |
| ruiqil/SpatialEpiBench [ILI2019] | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / date range: estimated / frequency: unknown |
| ruiqil/SpatialEpiBench [JHUcase] | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / date range: estimated / frequency: unknown |
| ruiqil/SpatialEpiBench [NCHSdeaths] | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / date range: estimated / frequency: unknown |
| thuml/UTSD [BIDMC32HR / IEEEPPG / MotorImagery / SelfRegulationSCP1 / SelfRegulationSCP2 / PigArtPressure / PigCVP / AtrialFibrillation] | date range: only series start times are in the metadata (no end time) / n_series: estimated / n_points: estimated / series length: estimated / date range: unknown / missing share: estimated / the viewer converted only part of the dataset to parquet |
| CashlessConsumer/chennai-rain-gauges | 5 non-parquet data files in the scope (14.29 MB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: unknown / frequency: unknown |
| ceepr-bit/dianchi-water | 1 non-parquet data files in the scope (9.36 KB) are not in the row and point counts / layout is a rule-based guess (R4) / series length: estimated / frequency: unknown |
| chrimerss/RSI-benchmark-hydro | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / date range: unknown / frequency: unknown |
| cjc0013/ouroboros-ooi-ctd-benchmark | 49 non-parquet data files in the scope (809.64 KB) are not in the row and point counts / n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / frequency: unknown |
| CMU-OSPO/Data_for_Extended-Range_Water_Demand_Forecasts_Driven_by_Subseasonal-to-Seasonal | 1 non-parquet data files in the scope (1.05 KB) are not in the row and point counts / layout is a rule-based guess (R5) / series length: estimated / date range: estimated / frequency: unknown |
| copernicus-argentina/estado-hidrico-argentina-2001-2026 | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| copernicus-argentina/lagunas-pampeanas-2018-2026 | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / date range: estimated / frequency: unknown |
| copernicus-argentina/parana-alturas-historicas-1900-2026 | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / date range: estimated / frequency: unknown |
| earino/noaa-tide-flooding | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / date range: estimated / frequency: unknown |
| Emulated-Inc/sci-arctic-sea-ice-forecast-later-years | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| Emulated-Inc/sci-lake-temperature-profiles-unmonitored | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| HenryYHW/BATON | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| jotone/pcfvg-weatherstations | layout is a rule-based guess (R4) / n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 350 parquet files; counts scaled by bytes (x6.7) |
| judicialmind/BharatWater-Bench | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| judicialmind/BharatWater-Data | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| National-Digital/au-water-storage-levels | 1 non-parquet data files in the scope (2.61 KB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / frequency: unknown |
| National-Digital/vic-melbourne-water-daily-storage | 1 non-parquet data files in the scope (3.12 KB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / frequency: unknown |
| phanerozoic/noaa-ww3-multi1-points [wmo] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| phanerozoic/ww3-ndbc-pairs | layout is a rule-based guess (R4) / series length: estimated / date range: estimated / frequency: unknown |
| Qdrant/NOAA-Buoy | 159 non-parquet data files in the scope (161.13 MB) are not in the row and point counts / n_series: 3 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R6) / series length: estimated / date range: unknown / frequency: unknown |
| surfe-diem/wave-archive-USA-southwest | n_series: 17 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 662 parquet files; counts scaled by bytes (x14.1) |
| taziksh/argo-profiles | n_series: estimated / series length: estimated / frequency: unknown |
| xairon/piezo-embedding-benchmark | 2 non-parquet data files in the scope (5.21 MB) are not in the row and point counts / n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / date range: estimated / frequency: unknown |
| zer0contextlost/waterwatch-runoff | 3 non-parquet data files in the scope (3.38 MB) are not in the row and point counts / n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / date range: estimated / frequency: unknown |
| Emulated-Inc/sci-streamflow-ungauged-new-country | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| National-Digital/qld-coastal-waves-cairns | 1 non-parquet data files in the scope (3.33 KB) are not in the row and point counts / layout is a rule-based guess (R5) / frequency: estimated |
| Rekin226/aquascope-gauges [observation bundles (USGS, UK EA, BOM, Hub'Eau, PEGELONLINE, ...)] | 19558 non-parquet data files in the scope (2.53 GB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 148 parquet files; counts scaled by bytes (x2.42) |
| Amgharr/FEMTO-ST_DATASET | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / n_points: estimated / series length: unknown / date range: unknown / frequency: unknown / missing share: estimated / footers read for 1 of 3 parquet files; counts scaled by bytes (x4.86) |
| Arailym-tleubayeva/NK-Oil-Well-Sensor-Monitoring-v2 | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / date range: estimated / frequency: estimated |
| Arailym-tleubayeva/NK-Oil-Well-Sensor-Monitoring-v2 [train/test] | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / date range: estimated / frequency: estimated |
| CDuong04/NGAFID-LOCI-GATS-Data | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| Emulated-Inc/sci-booster-gmps-regulation-forecast | n_series: 49 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 79 parquet files; counts scaled by bytes (x1.58) |
| Emulated-Inc/sci-vlt-turbulence-from-ao-telemetry | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| HenryYHW/CANTraj | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| jwsouza13/eVED | layout is a rule-based guess (R3) / n_series: unknown / series length: unknown / date range: unknown / frequency: unknown |
| krohnedigital/FuelCast | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / date range: unknown / frequency: unknown |
| mihnathul/celljar | 1776 non-parquet data files in the scope (2.46 MB) are not in the row and point counts / n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [BatteryRUL] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [BoschCNC] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [FrothFlotation] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [GasPipeline] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [GasSensorDynamic] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [GasSensorTemperature] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [Metropt3] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [MotorTemperature] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [OilWell] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [PUMP] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [SWAT] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown |
| nju-zhangsq/RMISC [WADI] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown |
| Samlzy/MCC5-THU-Motor | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| thuml/UTSD [SensorData] | date range: only series start times are in the metadata (no end time) / n_series: estimated / n_points: estimated / series length: estimated / date range: unknown / missing share: estimated / the viewer converted only part of the dataset to parquet |
| xijiazhao/casemotordataset [dated folders] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| xijiazhao/welding2 [trial=1, trial=2] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| cognitedata/IND-Eval | 652 non-parquet data files in the scope (8.63 GB) are not in the row and point counts / n_series: 46 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / footers read for 50 of 2399 parquet files; counts scaled by bytes (x43.9) |
| Forgis/FactoryNet [factorywave_*, voraus_*, aursad_*, cnc_*] | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / n_points: estimated / series length: unknown / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 99 parquet files; counts scaled by bytes (x1.9) |
| Yahya2491/3w-pipeline | 1 non-parquet data files in the scope (4.70 KB) are not in the row and point counts / n_series: 50 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 2228 parquet files; counts scaled by bytes (x47.8) |
| scalytics/smartmeterdata [smartmeter_*.csv] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| Adrien97/velib-monitoring-historical | 2 non-parquet data files in the scope (189 B) are not in the row and point counts / n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / n_points: estimated / series length: unknown / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 276 parquet files; counts scaled by bytes (x5.5) |
| aiacademy-kg/bishkek-transport | layout is a rule-based guess (R4) / n_series: estimated / n_points: estimated / series length: unknown / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 222 parquet files; counts scaled by bytes (x3.94) |
| Algo0o/flight_delay [default] | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / n_points: estimated / series length: unknown / date range: estimated / frequency: unknown / missing share: estimated / the viewer converted only part of the dataset to parquet |
| AmandaHattaway/bike-lane-lakehouse | layout is a rule-based guess (R4) / n_series: estimated / series length: unknown / frequency: unknown |
| angelosaadeh/velib | 1 non-parquet data files in the scope (196.12 KB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / date range: estimated / frequency: unknown |
| Arimancy/bts-flights-weather | 1 non-parquet data files in the scope (118.45 KB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / frequency: estimated |
| autogluon/chronos_datasets [mexico_city_bikes] | frequency: unknown |
| autogluon/chronos_datasets [monash_pedestrian_counts] | frequency: unknown |
| autogluon/chronos_datasets [monash_rideshare] | frequency: estimated |
| autogluon/chronos_datasets [monash_traffic] | frequency: estimated |
| autogluon/chronos_datasets [taxi_30min, taxi_1h] | series length: estimated |
| autogluon/chronos_datasets [uber_tlc_hourly, uber_tlc_daily] | series length: estimated |
| ChengsenWang/CGTSF [PTF] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| danipaez/velib_tft [default] | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / date range: estimated / frequency: unknown |
| digitalhen/nyc-subway-realtime | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| digitalhen/us-airport-wait-times | 1 non-parquet data files in the scope (197 B) are not in the row and point counts / n_series: 24 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 62 parquet files; counts scaled by bytes (x1.2) |
| dmariaa70/GO-MO | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / n_points: estimated / series length: unknown / date range: unknown / frequency: unknown / missing share: unknown / the viewer converted only part of the dataset to parquet |
| Donny-Guo/airline-delay | 3 non-parquet data files in the scope (439.75 KB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 121 parquet files; counts scaled by bytes (x2.44) |
| EPI-Eval/global-mobility | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / frequency: unknown |
| femartip/valencia-urban-mobility | 3196 non-parquet data files in the scope (13.23 GB) are not in the row and point counts / n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / n_points: estimated / series length: unknown / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 3175 parquet files; counts scaled by bytes (x77.3) |
| feregrino/london-cycles | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown / viewer /parquet: HTTP 500 PreviousStepStillProcessingError The previous steps are still being processed / viewer /info: HTTP 500 PreviousStepStillProcessingError The previous steps are still being processed |
| fidel-ts/NYC_traffic_speed | 213 non-parquet data files in the scope (4.65 GB) are not in the row and point counts / n_series: 50 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 97 parquet files; counts scaled by bytes (x1.95) |
| fulldecent/septa-regionalrail-otp | n_series: 15 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / frequency: estimated |
| gigigi7/AIS_Dataset [default] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| Jakov/bajs-zagreb [station_status] | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / n_points: estimated / series length: unknown / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 14898 parquet files; counts scaled by bytes (x300) |
| jhumbl/dft-vehicle-licensing | 2 non-parquet data files in the scope (19.83 MB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / date range: unknown / frequency: unknown |
| jhumbl/orr-station-usage | 1 non-parquet data files in the scope (1.52 MB) are not in the row and point counts / n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / date range: unknown / frequency: unknown |
| jhumbl/tfl-network-demand | 3 non-parquet data files in the scope (14.34 MB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / date range: estimated / frequency: unknown |
| Karry001/movemelbourne-gtfsr | 2211 non-parquet data files in the scope (14.24 MB) are not in the row and point counts / layout is a rule-based guess (R3) / n_series: estimated / n_points: estimated / series length: unknown / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 11634 parquet files; counts scaled by bytes (x301) |
| kpubdata/seoul-bike-rent-month | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / date range: unknown / frequency: estimated |
| Malavnaik/dublin-bikes-weather-raw | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / n_points: estimated / series length: unknown / date range: unknown / frequency: unknown / missing share: estimated / footers read for 50 of 3133 parquet files; counts scaled by bytes (x62.8) |
| matrosovdani/moscow-parking-occupancy | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / frequency: unknown |
| MLCapstoneRideshare/rideshare-trips | layout is a rule-based guess (R3) / n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 185 parquet files; counts scaled by bytes (x3.66) |
| National-Digital/nsw-traffic-volume-yearly | 1 non-parquet data files in the scope (4.13 KB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [BTS] | layout is a rule-based guess (R4) / series length: estimated / date range: estimated / frequency: unknown |
| nju-zhangsq/RMISC [ChengduTaxi] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [LargeST] | layout is a rule-based guess (R4) / n_points: estimated / series length: estimated / date range: unknown / missing share: estimated / footers read for 50 of 68 parquet files; counts scaled by bytes (x1.36) |
| nju-zhangsq/RMISC [MetroTraffic] | layout is a rule-based guess (R4) / series length: estimated / date range: estimated / frequency: unknown |
| nju-zhangsq/RMISC [PEMSCalifornia] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [T-Drive] | layout is a rule-based guess (R4) / series length: estimated |
| olaflaitinen/baku-metro-station-flow | 4 non-parquet data files in the scope (699.72 KB) are not in the row and point counts / n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / frequency: unknown |
| orailix/ride-silver | 1 non-parquet data files in the scope (32.59 KB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / n_points: estimated / series length: unknown / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 76 parquet files; counts scaled by bytes (x1.36) |
| piebro/deutsche-bahn-data | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / n_points: estimated / series length: unknown / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 2116 parquet files; counts scaled by bytes (x51.5) |
| Salesforce/lotsa_data [BEIJING_SUBWAY_30MIN] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [covid_mobility] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [HZMETRO] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [largest_2017] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [largest_2018] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [largest_2019] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [largest_2020] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [largest_2021] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [LOS_LOOP] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [pedestrian_counts] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [PEMS03] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [PEMS04] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [PEMS07] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [PEMS08] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [PEMS_BAY] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [Q-TRAFFIC] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [rideshare_with_missing] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [SHMETRO] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [taxi_30min] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [traffic_hourly] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [uber_tlc_daily] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [uber_tlc_hourly] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [vehicle_trips_with_missing] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| somemone/ac-transit-apc | layout is a rule-based guess (R3) / n_series: unknown / n_points: estimated / series length: unknown / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 89 parquet files; counts scaled by bytes (x1.8) |
| swadhinbiswas/air-traffic | 22054 non-parquet data files in the scope (27.25 GB) are not in the row and point counts / n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / n_points: estimated / series length: unknown / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 22066 parquet files; counts scaled by bytes (x371) |
| vinicius-souza/antaq | 9 non-parquet data files in the scope (11.44 MB) are not in the row and point counts / n_series: 4 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R6) / n_points: estimated / series length: estimated / date range: unknown / frequency: unknown / missing share: estimated / footers read for 50 of 167 parquet files; counts scaled by bytes (x2.63) |
| witgaw/METR-LA | 4 non-parquet data files in the scope (6.57 MB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / date range: estimated / frequency: estimated |
| witgaw/PEMS-BAY | 4 non-parquet data files in the scope (612.68 KB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / date range: estimated / frequency: estimated |
| yadavsaurabh/indian-railways-runs | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / n_points: estimated / series length: unknown / date range: estimated / frequency: unknown / missing share: estimated / the viewer converted only part of the dataset to parquet |
| yind7/EvoXXLTraffic | n_series: unknown / series length: unknown / date range: unknown / frequency: unknown / missing share: unknown |
| youngbongbong/30_od_subway [default] | layout is a rule-based guess (R4) / n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 3146 parquet files; counts scaled by bytes (x62.2) |
| youngbongbong/mobility_dataset | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / n_points: estimated / series length: unknown / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 2098 parquet files; counts scaled by bytes (x42) |
| yuiseki/meta-range-maps-2022-05 | 4 non-parquet data files in the scope (129.62 MB) are not in the row and point counts / n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / frequency: unknown |
| yuiseki/npa-traffic-accidents | 36 non-parquet data files in the scope (460.95 MB) are not in the row and point counts / n_series: unknown / series length: unknown / date range: unknown / frequency: unknown / missing share: unknown |
| TorchSpatiotemporal/ProcessedDatasets | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| onedean/UrbanFM-datasets [taxinyc/bikenyc/tdrive/trafficsh/occhamburg/occparis (+pems, metrla)] | layout is a rule-based guess (R6) / date range: unknown / frequency: unknown |
| tjtrans/Traffic-Demand-Bench | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| ZanderL1337/shipping-data-pipeline | layout is a rule-based guess (R4) / n_series: estimated / series length: unknown |
| Bayway/italian-open-timeseries | 1 non-parquet data files in the scope (130.27 KB) are not in the row and point counts / layout is a rule-based guess (R4) / series length: estimated / frequency: unknown |
| judicialmind/BharatTS-Corpus | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| nju-zhangsq/RMISC [TimeMMD] | layout is a rule-based guess (R4) / series length: estimated / date range: estimated / frequency: unknown |
| thuml/UTSD [London Smart Meters / Wind Farms / Aus. Electricity Demand / Pedestrian Counts / Web Traffic / Sunspot] | date range: only series start times are in the metadata (no end time) / n_series: estimated / n_points: estimated / series length: estimated / date range: unknown / missing share: estimated / the viewer converted only part of the dataset to parquet |
| AutonLab/Timeseries-PILE [forecasting] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| AutonLab/Timeseries-PILE [forecasting/autoformer] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| nju-zhangsq/RMISC [Cryptocurrency] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| Time-HD-Anonymous/High_Dimensional_Time_Series | n_series: 3 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 25 of 39 parquet files; counts scaled by bytes (x3.19) |
| Multiple-Time-Series/PanelTS | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| nju-zhangsq/RMISC [ExchangeRate] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [Illness] | layout is a rule-based guess (R4) / series length: estimated / date range: estimated / frequency: estimated |
| prabakaranchandran/enterprise-dynamics-corpus-v1 | 110 non-parquet data files in the scope (374.48 MB) are not in the row and point counts / n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 104 parquet files; counts scaled by bytes (x2.27) |
| alanwgt/fipex-veiculos-brasil | 14 non-parquet data files in the scope (2.00 GB) are not in the row and point counts / n_series: 13 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R6) / date range: unknown / frequency: unknown |
| autogluon/chronos_datasets [dominick] | frequency: unknown |
| CQ-AI/Financial_datasets [Retail and wherehouse Sale.csv] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| Dingdong-Inc/FreshRetailNet-50K [default] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 1 of 2 parquet files; counts scaled by bytes (x13.6) |
| Dingdong-Inc/FreshRetailNet-LT [unknown] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 1 of 2 parquet files; counts scaled by bytes (x158) |
| dubairealestatedata/dubai-real-estate-sales-transactions | 2 non-parquet data files in the scope (240.53 MB) are not in the row and point counts / layout is a rule-based guess (R3) / n_series: unknown / series length: unknown / frequency: unknown |
| earino/my-pricecatcher-weekly-increase | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / date range: unknown / frequency: unknown |
| fuel-prices-eu/european-fuel-prices | 4 non-parquet data files in the scope (2.97 MB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / date range: estimated / frequency: unknown |
| idomanteu/cs2-historical-item-prices-hourly-march-april-2026 | 2 non-parquet data files in the scope (1.19 GB) are not in the row and point counts / n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / frequency: estimated |
| jamesb7/fuel-prices-uk | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / date range: unknown / frequency: unknown |
| kpubdata/seoul-apartment-rent | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / date range: estimated / frequency: unknown |
| kpubdata/seoul-apartment-trades | layout is a rule-based guess (R4) / n_series: estimated / series length: unknown / date range: unknown / frequency: unknown |
| National-Digital/nt-fuel-prices-history | 1 non-parquet data files in the scope (28.20 KB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / frequency: unknown |
| National-Digital/qld-median-rents-by-postcode | 1 non-parquet data files in the scope (2.89 KB) are not in the row and point counts / layout is a rule-based guess (R5) / date range: estimated / frequency: unknown |
| nju-zhangsq/RMISC [Dominick] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [WeeklyFuelPricesItaly] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [WeeklyRoadFuelPrices] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| pracadata/brazil-fuel-prices | 1 non-parquet data files in the scope (1.38 KB) are not in the row and point counts / layout is a rule-based guess (R3) / n_series: estimated / series length: estimated / frequency: unknown |
| pracadata/mexico-fuel-prices | 1 non-parquet data files in the scope (1.54 KB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: unknown / frequency: unknown |
| Rif-SQL/time-series-uk-retail-supermarket-price-data | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / frequency: unknown |
| SURESHBEEKHANI/SalesForecastingWOMartStore [default] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| thehooklab/nsw-property | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / date range: estimated / frequency: unknown |
| VincentGOURBIN/FuelInFranceData | layout is a rule-based guess (R4) / n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / the viewer converted only part of the dataset to parquet |
| xus/Spain-Fuel-Prices | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / frequency: estimated |
| AbdullahImran/Saudi-Supply-Demand-Datasets | 2 non-parquet data files in the scope (22.95 MB) are not in the row and point counts / layout is a rule-based guess (R3) / n_series: estimated / series length: unknown / frequency: unknown |
| APProjects/us-layoffs-monthly-time-series-warn-act | layout is a rule-based guess (R4) / series length: estimated / date range: estimated / frequency: estimated |
| arh1p/instagram-top-accounts | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / date range: estimated / frequency: unknown |
| earino/austin-911-response | layout is a rule-based guess (R3) / n_series: unknown / series length: unknown / date range: estimated / frequency: unknown |
| Gallicagram/atlas-motifs | 1 non-parquet data files in the scope (17.07 KB) are not in the row and point counts / n_series: 18 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / n_series: estimated / n_points: estimated / series length: estimated / date range: unknown / frequency: unknown / missing share: estimated / footers read for 50 of 108 parquet files; counts scaled by bytes (x1.78) |
| gladcolor/hourly_population_US | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| gymprathap/Chicago-Crime-Dataset | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / date range: unknown / frequency: unknown |
| hsilvosa/gdelt-events | 8 non-parquet data files in the scope (106.57 KB) are not in the row and point counts / layout is a rule-based guess (R3) / n_series: unknown / n_points: estimated / series length: unknown / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 929 parquet files; counts scaled by bytes (x18.5) |
| Jannchie/bilibili-creator-stats | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| jhumbl/police-crime-london | layout is a rule-based guess (R2) / n_series: unknown / series length: unknown / date range: estimated / frequency: unknown / missing share: unknown |
| lingbow/tiktok-video-engagement-200k | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / date range: estimated / frequency: unknown |
| matanzig/Disney-Theme-Park-Queue-Dynamics | layout is a rule-based guess (R4) / series length: estimated / date range: estimated / frequency: unknown |
| Nason/open-policing-database | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| National-Digital/au-births-by-age-of-mother | 1 non-parquet data files in the scope (5.83 KB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / date range: estimated / frequency: estimated |
| nju-zhangsq/RMISC [BooksPerPerson] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| todopueblos/poblacion-municipios-espana | layout is a rule-based guess (R6) / date range: unknown / frequency: unknown |
| valdovaldo1500/crime-brasil-rs-2022-2026 | layout is a rule-based guess (R6) / date range: unknown / frequency: unknown |
| yuiseki/estat-jumin-idou | 944 non-parquet data files in the scope (1.70 GB) are not in the row and point counts / n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / n_points: estimated / series length: unknown / date range: unknown / frequency: unknown / missing share: estimated / footers read for 50 of 472 parquet files; counts scaled by bytes (x6.9) |
| yuiseki/ucdp-ged [26.1] | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / frequency: unknown |
| zwn22/NC_Crime | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| yuiseki/un-wpp [WPP2024] | 45 non-parquet data files in the scope (4.04 GB) are not in the row and point counts / n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / date range: unknown / frequency: unknown |
| yuiseki/undp-hdr | 11 non-parquet data files in the scope (2.69 MB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: unknown / date range: estimated / frequency: unknown |
| zaai-ai/hierarchical_time_series_datasets | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| earino/abs-visitor-arrivals-direction | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / date range: unknown / frequency: unknown |
| AstroMLCore/AstroM3Dataset | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| AutonLab/Timeseries-PILE [anomaly_detection/TSB-UAD-Public] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| cangyeone/SeismicX-Cont | layout is a rule-based guess (R2) / n_series: unknown / series length: unknown / date range: unknown / frequency: unknown / missing share: unknown |
| giggseh/kepler-lc-star-params | date range: only series start times are in the metadata (no end time) / 1 non-parquet data files in the scope (8.90 MB) are not in the row and point counts / n_series: estimated / n_points: estimated / series length: estimated / date range: unknown / frequency: unknown / missing share: estimated / footers read for 50 of 400 parquet files; counts scaled by bytes (x7.99) |
| juliensimon/auroral-electrojet-index | layout is a rule-based guess (R5) / frequency: estimated |
| juliensimon/celestrak-space-weather | layout is a rule-based guess (R5) / frequency: unknown |
| juliensimon/deep-space-probes | layout is a rule-based guess (R5) / frequency: unknown |
| juliensimon/dst-index | layout is a rule-based guess (R5) / frequency: unknown |
| juliensimon/f107-solar-flux | layout is a rule-based guess (R5) / frequency: unknown |
| juliensimon/geomagnetic-kp-index | layout is a rule-based guess (R5) / frequency: estimated |
| juliensimon/goes-xray-flux | layout is a rule-based guess (R5) / frequency: estimated |
| juliensimon/iers-earth-orientation | layout is a rule-based guess (R5) / frequency: estimated |
| juliensimon/nasa-maven-kp-insitu | layout is a rule-based guess (R5) / series length: estimated / frequency: unknown |
| juliensimon/neutron-monitor-cosmic-rays | layout is a rule-based guess (R4) / series length: estimated / frequency: unknown |
| juliensimon/omni-solar-wind-parameters | layout is a rule-based guess (R5) / frequency: estimated |
| juliensimon/solar-wind | layout is a rule-based guess (R5) / frequency: unknown |
| juliensimon/space-track-tle-history | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| lxr11111/telemanom [SMAP,MSL] | 412 non-parquet data files in the scope (272.15 MB) are not in the row and point counts / layout is a rule-based guess (R6) / n_series: estimated / n_points: estimated / series length: estimated / date range: unknown / frequency: unknown / missing share: estimated / footers read for 50 of 165 parquet files; counts scaled by bytes (x3.54) |
| MultimodalUniverse/tess | date range: only series start times are in the metadata (no end time) / series length: estimated / date range: unknown / frequency: unknown |
| nasa-ibm-ai4science/Surya-bench-solarwind | n_series: 4 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / date range: estimated / frequency: unknown |
| nju-zhangsq/RMISC [KeplerLightCurves] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [Satellite] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| nyu-planet-finders/kepler-llc | date range: only series start times are in the metadata (no end time) / n_series: estimated / series length: estimated / date range: unknown / frequency: unknown |
| nyu-planet-finders/tess-spoc-2min-binned | date range: only series start times are in the metadata (no end time) / n_series: estimated / n_points: estimated / series length: estimated / date range: unknown / frequency: unknown / missing share: estimated / footers read for 50 of 107 parquet files; counts scaled by bytes (x2.13) |
| patrickfleith/GOCE-satellite-telemtry [raw, 10min stats, 6h stats] | 2 non-parquet data files in the scope (99.44 KB) are not in the row and point counts / n_series: 17 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 981 parquet files; counts scaled by bytes (x13.7) |
| phanerozoic/qiskit-calibration-drift | layout is a rule-based guess (R3) / n_series: unknown / series length: unknown / frequency: unknown |
| snowsadh/goes-omni-electron-flux-forecasting | 2 non-parquet data files in the scope (1.15 MB) are not in the row and point counts / n_series: 3 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / series length: estimated / frequency: unknown |
| ThermoBench/ThermoBench_pure | layout is a rule-based guess (R5) / n_series: estimated / series length: estimated / frequency: unknown |
| aerostratospheric/spc-tornado-history | layout is a rule-based guess (R5) / date range: estimated / frequency: unknown |
| alexdum/meteogate-archive | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / n_points: estimated / series length: unknown / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 254 parquet files; counts scaled by bytes (x5.17) |
| autogluon/chronos_datasets [monash_weather] | frequency: unknown |
| autogluon/chronos_datasets [weatherbench_daily] | series length: estimated |
| autogluon/chronos_datasets [weatherbench_hourly_2m_temperature] | series length: estimated / frequency: estimated |
| autogluon/chronos_datasets [weatherbench_hourly_temperature] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / footers read for 50 of 107 parquet files; counts scaled by bytes (x2.15) |
| autogluon/chronos_datasets_extra [brazilian_cities_temperature] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| brandonhxrr/mexico_city_weather | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| cm3r/meteo-france-climatologie-quotidienne | n_series: 48 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 319 parquet files; counts scaled by bytes (x7.51) |
| datonic/spain_aemet_historical_weather | layout is a rule-based guess (R5) / frequency: unknown |
| DuckerMaster/thaiclim-district | 8 non-parquet data files in the scope (6.41 MB) are not in the row and point counts / n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / date range: estimated / frequency: unknown |
| edilia12380/era5-daily-weather-western-europe | n_series: unknown / series length: unknown / date range: unknown / frequency: unknown / missing share: unknown |
| elskow/Weather4cast | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| Emulated-Inc/sci-airport-low-ceiling-visibility | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / frequency: unknown |
| excarta/madis2020 | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| furusu/japan-max-temperature | 1 non-parquet data files in the scope (101.70 KB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / frequency: unknown |
| Geoweaver/SnowWaterEquivalent_trainingdata | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| hblim/weather_data | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| jacobbieker/dwd | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| jotone/arpa-osmer-weatherstations | layout is a rule-based guess (R4) / n_series: estimated / n_points: estimated / series length: unknown / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 126 parquet files; counts scaled by bytes (x2.43) |
| juliangehring/noaa-isd | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / n_points: estimated / series length: unknown / date range: estimated / frequency: estimated / missing share: estimated / footers read for 50 of 125 parquet files; counts scaled by bytes (x2.52) |
| Kamilatr/weather_data_bavaria | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| MeteoSwiss/PeakWeather | 22 non-parquet data files in the scope (15.66 GB) are not in the row and point counts / n_series: 2 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / n_series: estimated / series length: estimated / frequency: unknown |
| nakas/powderbench-truth | layout is a rule-based guess (R4) / n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 55 parquet files; counts scaled by bytes (x1.1) |
| nju-zhangsq/RMISC [ERA5HourlySingleLevels] | layout is a rule-based guess (R4) / series length: estimated / date range: estimated / frequency: estimated |
| nju-zhangsq/RMISC [GlobalClimateChange] | layout is a rule-based guess (R4) / series length: estimated / date range: estimated / frequency: unknown |
| nju-zhangsq/RMISC [WeatherBench5-625deg] | layout is a rule-based guess (R4) / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 1000 parquet files; counts scaled by bytes (x20) |
| notadib/NASA-Power-Daily-Weather | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| numix/stations-officielles-es | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| Pahzuzu/PAH-ENV-001-serc-soil-signals | 4 non-parquet data files in the scope (25.42 KB) are not in the row and point counts / n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / frequency: unknown |
| previdengue/predict_climate_data | layout is a rule-based guess (R6) / date range: unknown / frequency: unknown |
| ron-the-code/Telangana_time_series_2023-2025 | layout is a rule-based guess (R5) / date range: unknown / frequency: unknown |
| Salesforce/lotsa_data [era5] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [oikolab_weather] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [subseasonal] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [subseasonal_precip] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [weather] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| sayanroy058/Weather-Time-Series-Forecasting | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| taziksh/cliwoc | layout is a rule-based guess (R4) / n_series: estimated / series length: unknown / date range: estimated / frequency: unknown |
| taziksh/ibtracs | 1 non-parquet data files in the scope (1.36 KB) are not in the row and point counts / n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / frequency: unknown |
| thuml/UTSD [ERA5 Surface / ERA5 Pressure] | date range: only series start times are in the metadata (no end time) / n_series: estimated / n_points: estimated / series length: estimated / date range: unknown / missing share: estimated / the viewer converted only part of the dataset to parquet |
| VerisimilitudeX/Weather | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / date range: estimated / frequency: unknown |
| WasuratS/ECMWF_Thailand_Land_Air_Temperatures | n_series: 24 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / frequency: unknown |
| yuiseki/agera5-japan-monthly-temperature-1979-2026 | 3 non-parquet data files in the scope (800.89 KB) are not in the row and point counts / n_series: 48 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / frequency: unknown |
| yuiseki/jma-rain-kanto | 5662 non-parquet data files in the scope (1.34 GB) are not in the row and point counts / n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / frequency: unknown |
| zait-ai/RusWeather-434 | 1 non-parquet data files in the scope (11.75 KB) are not in the row and point counts / layout is a rule-based guess (R4) / series length: estimated / frequency: estimated |
| hsilvosa/noaa-ghcn-daily | n_series: unknown / series length: unknown / date range: unknown / frequency: unknown / missing share: unknown |
| NaaVrug/weather-geo-era5 | 3 non-parquet data files in the scope (66.12 KB) are not in the row and point counts / n_series: 48 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / date range: estimated / frequency: unknown |
| tharinduperera/weather-clustering-data | 2 non-parquet data files in the scope (9.67 KB) are not in the row and point counts / n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / n_points: estimated / series length: unknown / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 1352 parquet files; counts scaled by bytes (x7.51) |
| afhubbard/gpu-prices | layout is a rule-based guess (R4) / n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 428 parquet files; counts scaled by bytes (x8.59) |
| AliMaatouk/TelecomTS [default] | series length: estimated / date range: estimated / frequency: unknown |
| AmirShahbaz/CloudPerfTrace | date range: only series start times are in the metadata (no end time) / 1 non-parquet data files in the scope (3.50 KB) are not in the row and point counts / series length: estimated / date range: unknown / frequency: unknown / missing share: estimated |
| aurman/GoogleTrendArchive [trending_queries] | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / date range: estimated / frequency: unknown |
| commoncrawl/statistics | layout is a rule-based guess (R5) / n_series: estimated / series length: estimated / date range: estimated / frequency: estimated |
| emperor-mew/ooni-censorship-historical | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / date range: estimated / frequency: unknown |
| Emulated-Inc/wikipedia-clickstream-enwiki | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| EPI-Eval/wikipedia-pageviews | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / frequency: unknown |
| geocables/internet-latency-and-routing-events | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| hq-bench/quito-corpus [hour] | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated |
| hq-bench/quito-corpus [min] | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated |
| hq-bench/quitobench [hour, min] | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated |
| hq-bench/quitobench [hour, min] | layout is a rule-based guess (R4) / n_series: estimated / series length: estimated |
| jeremycochoy/wikimedia-pageview-timeseries | date range: only series start times are in the metadata (no end time) / n_series: estimated / n_points: estimated / series length: estimated / date range: unknown / frequency: unknown / missing share: estimated / footers read for 50 of 680 parquet files; counts scaled by bytes (x13.5) |
| kashif/App_Flow [default] | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / date range: unknown / frequency: estimated |
| LeadLagForecasting/llf_github [default] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 379 parquet files; counts scaled by bytes (x7.62) |
| Lemma-RCA-NEC/Cloud_Computing_Preprocessed | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| Lightcap/multi-domain-cloudflare-observability | 11 non-parquet data files in the scope (2.69 MB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: unknown / date range: estimated / frequency: unknown |
| Lilambd/security-daily | layout is a rule-based guess (R4) / n_series: estimated / series length: unknown / date range: estimated / frequency: unknown / missing share: estimated |
| lsliwko/google-cluster-data-2019-sorted-by-timestamp | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| MachaParfait/ClusterWise | 1 non-parquet data files in the scope (382 B) are not in the row and point counts / n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / n_points: estimated / series length: unknown / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 142 parquet files; counts scaled by bytes (x2.87) |
| MarcusLammers/vast-rtx3090-market-6mo | layout is a rule-based guess (R4) / n_series: estimated / series length: unknown / frequency: unknown |
| modelpulse/model-pulse-data | 3 non-parquet data files in the scope (948.06 KB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 185 parquet files; counts scaled by bytes (x4.19) |
| netop/gotsf-ds [default] | n_series: 3 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R6) / series length: estimated / date range: unknown / frequency: unknown |
| nevirim/wikipedia-attention-13lang | 3 non-parquet data files in the scope (85.54 KB) are not in the row and point counts / n_series: 13 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / date range: estimated / frequency: unknown |
| nju-zhangsq/RMISC [AliCar] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [AzurePublicDatasetV1] | layout is a rule-based guess (R4) / n_points: estimated / series length: estimated / date range: unknown / missing share: estimated / footers read for 50 of 79 parquet files; counts scaled by bytes (x1.59) |
| nju-zhangsq/RMISC [EbayServer] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [MiniApp] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [NAB] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| nju-zhangsq/RMISC [ServerMachineDataset] | layout is a rule-based guess (R4) / series length: estimated / date range: unknown / frequency: unknown |
| paperswithbacktest/Stocks-Quarterly-WikipediaViews | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| pyToshka/attacks-daily | 4 non-parquet data files in the scope (109.76 MB) are not in the row and point counts / layout is a rule-based guess (R3) / n_series: unknown / series length: unknown / date range: estimated / frequency: unknown |
| Salesforce/cloudops_tsf [alibaba_cluster_trace_2018] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| Salesforce/cloudops_tsf [borg_cluster_data_2011] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| Salesforce/lotsa_data [alibaba_cluster_trace_2018] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [borg_cluster_data_2011] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [extended_web_traffic_with_missing] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [kaggle_web_traffic_weekly] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Salesforce/lotsa_data [wiki-rolling_nips] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |
| Sbaiiiiii/looked-up | 364 non-parquet data files in the scope (160.47 MB) are not in the row and point counts / n_series: 25 files share one schema; if each file is a different entity the series count is files x columns, not columns / layout is a rule-based guess (R5) / n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / frequency: unknown / missing share: estimated / footers read for 50 of 242 parquet files; counts scaled by bytes (x3.53) |
| Shashkovich/Telecommunication_SMS_time_series | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| soncco/historical-twitter-trends-spanish-locations-2020-2021 | 1 non-parquet data files in the scope (5.23 KB) are not in the row and point counts / n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / frequency: unknown |
| sovai/wikipedia_views [default] | n_series: no distinct count for the id column(s) in the metadata / layout is a rule-based guess (R4) / n_series: unknown / series length: unknown / frequency: unknown |
| tardellirs/paper-pulse-data | 1 non-parquet data files in the scope (19.81 KB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / frequency: unknown |
| venvoo/openrouter-uptime | 2 non-parquet data files in the scope (23.45 MB) are not in the row and point counts / layout is a rule-based guess (R4) / n_series: estimated / series length: estimated / date range: estimated / frequency: unknown |
| wikimedia-community/english-wikipedia-pageviews-by-second | layout is a rule-based guess (R4) / series length: estimated / frequency: estimated |
| Datadog/BOOM [(whole repo: 2,807 folders named ds-<id>-<freq>, e.g. ds-10-T, ds-1030-5T)] | n_points: some tables in the scope have no row count / date range: only series start times are in the metadata (no end time) / n_points: unknown / series length: unknown / date range: unknown / frequency: unknown / missing share: unknown |
| YahooResearch/ydata-labeled-time-series-anomalies-v1_0 | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| Salesforce/cloudops_tsf [azure_vm_traces_2017] | no schema: rows, series, points, lengths, dates, columns and missing share are all unknown |
| Salesforce/lotsa_data [azure_vm_traces_2017] | n_series: estimated / n_points: estimated / series length: estimated / date range: estimated / missing share: estimated / the viewer converted only part of the dataset to parquet |

