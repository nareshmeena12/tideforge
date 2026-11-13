# tfbench build report, Milestone A

## 1. Selection (A1)
Catalog rows: 614. Selected: 501. Dropped: 113.

| reason | rows |
|---|---|
| status check (v1 uses keep, keep_filtered) | 55 |
| gated, no access | 22 |
| no data files in scope | 2 |
| scope 87.7 GB > 20 GB | 1 |
| scope 121.5 GB > 20 GB | 1 |
| scope 45.8 GB > 20 GB | 1 |
| scope 36.3 GB > 20 GB | 1 |
| scope 21.2 GB > 20 GB | 1 |
| scope 57.7 GB > 20 GB | 1 |
| scope 47.4 GB > 20 GB | 1 |
| scope 166.4 GB > 20 GB | 1 |
| scope 159.2 GB > 20 GB | 1 |
| scope 35.5 GB > 20 GB | 1 |
| scope 39.6 GB > 20 GB | 1 |
| scope 57.4 GB > 20 GB | 1 |
| scope 56.1 GB > 20 GB | 1 |
| scope 68.8 GB > 20 GB | 1 |
| scope 91.2 GB > 20 GB | 1 |
| scope 28.9 GB > 20 GB | 1 |
| scope 34.7 GB > 20 GB | 1 |
| scope 1603.6 GB > 20 GB | 1 |
| scope 201.2 GB > 20 GB | 1 |
| scope 45.2 GB > 20 GB | 1 |
| scope 96.5 GB > 20 GB | 1 |
| scope 423.6 GB > 20 GB | 1 |
| scope 166.5 GB > 20 GB | 1 |
| scope 27.7 GB > 20 GB | 1 |
| scope 45.3 GB > 20 GB | 1 |
| scope 94.1 GB > 20 GB | 1 |
| scope 44.1 GB > 20 GB | 1 |
| scope 20.6 GB > 20 GB | 1 |
| scope 333.7 GB > 20 GB | 1 |
| scope 265.9 GB > 20 GB | 1 |
| scope 386.5 GB > 20 GB | 1 |
| scope 117.7 GB > 20 GB | 1 |
| scope 221.0 GB > 20 GB | 1 |
| scope 25.3 GB > 20 GB | 1 |

### Repos with a file cap (series sampled) or a partition cut (newest time partitions only)
| repo_id | files_before | files_after | series_keys | reason |
|---|---|---|---|---|
| AYUSHKHAIRE/all-stock-market-data-daily-updates | 19610 | 5000 | 19610 | 19610 series keys (one series per file or folder) > 5000; example key 'A/A.csv' |
| yadavsaurabh/us-stocks-and-search-trends | 57116 | 5000 | 57116 | 57116 series keys (one series per file or folder) > 5000; example key 'fixes.csv' |
| Jakov/bajs-zagreb | 14898 | 14898 | 2 | time partitions; 50000 x 1.2 steps of 86400 s (frequency word in the file paths) reach before the oldest file: nothing cut |
| Karry001/movemelbourne-gtfsr | 13845 | 13845 | 171 | time partitions; 50000 x 1.2 steps of 86400 s (median spacing of file dates (profile has no frequency)) reach before the oldest file: nothing cut |
| femartip/valencia-urban-mobility | 6371 | 6371 | 29 | time partitions; 50000 x 1.2 steps of 2.6298e+06 s (frequency word in the file paths) reach before the oldest file: nothing cut |
| gladcolor/hourly_population_US | 37689 | 37689 | 3142 | time partitions; 50000 x 1.2 steps of 2.6784e+06 s (median spacing of file dates (profile has no frequency)) reach before the oldest file: nothing cut |
| numix/stations-officielles-es | 29054 | 29054 | 384 | time partitions; 50000 x 1.2 steps of 8.0663e+08 s (frequency word in the file paths) reach before the oldest file: nothing cut |
| trentmkelly/polymarket_historical_data | 28024 | 28024 | 1 | time partitions; 50000 x 1.2 steps of 86400 s (median spacing of file dates (profile has no frequency)) reach before the oldest file: nothing cut |
| youngbongbong/30_od_subway | 3146 | 1245 | 1 | time partitions; kept files after 2022-03-31 = 50000 x 1.2 steps of 1800 s (frequency word in the file paths) before the newest (2025-09-01) |
| youngbongbong/mobility_dataset | 2098 | 2098 | 4 | time partitions; 50000 x 1.2 steps of 86400 s (frequency word in the file paths) reach before the oldest file: nothing cut |
| yuiseki/jma-rain-kanto | 5693 | 896 | 303 | time partitions; kept files after 2020-02-27 = 50000 x 1.2 steps of 3600 s (frequency word in the file paths) before the newest (2027-01-01) |

### Files removed from the Hub since the catalog was listed (download 404): 150
| repo | files | paths |
|---|---|---|
| Algo0o/flight_delay | 10 | default/train/0000.parquet; default/train/0001.parquet; default/train/0002.parquet; default/train/0003.parquet; default/train/0004.parquet; default/train/0005.parquet; default/train/0006.parquet; default/train/0007.parquet; default/train/0008.parquet; default/train/0009.parquet |
| Salesforce/lotsa_data | 9 | borg_cluster_data_2011/train/0000.parquet; borg_cluster_data_2011/train/0001.parquet; borg_cluster_data_2011/train/0002.parquet; borg_cluster_data_2011/train/0003.parquet; borg_cluster_data_2011/train/0004.parquet; borg_cluster_data_2011/train/0005.parquet; borg_cluster_data_2011/train/0006.parquet; borg_cluster_data_2011/train/0007.parquet; borg_cluster_data_2011/train/0008.parquet |
| VincentGOURBIN/FuelInFranceData | 10 | default/train/0000.parquet; default/train/0001.parquet; default/train/0002.parquet; default/train/0003.parquet; default/train/0004.parquet; default/train/0005.parquet; default/train/0006.parquet; default/train/0007.parquet; default/train/0008.parquet; default/train/0009.parquet |
| dmariaa70/METRAQ-Air-Quality | 10 | default/train/0000.parquet; default/train/0001.parquet; default/train/0002.parquet; default/train/0003.parquet; default/train/0004.parquet; default/train/0005.parquet; default/train/0006.parquet; default/train/0007.parquet; default/train/0008.parquet; default/train/0009.parquet |
| feregrino/london-cycles | 83 | data/year=2026/month=11/day=11/000019.csv; data/year=2026/month=11/day=11/001516.csv; data/year=2026/month=11/day=11/003017.csv; data/year=2026/month=11/day=11/004517.csv; data/year=2026/month=11/day=11/010017.csv; data/year=2026/month=11/day=11/010703.csv; data/year=2026/month=11/day=11/012205.csv; data/year=2026/month=11/day=11/013707.csv; data/year=2026/month=11/day=11/015210.csv; data/year=2026/month=11/day=11/015223.csv; data/year=2026/month=11/day=11/020015.csv; data/year=2026/month=11/day=11/021517.csv; data/year=2026/month=11/day=11/023017.csv; data/year=2026/month=11/day=11/024517.csv; data/year=2026/month=11/day=11/030017.csv; data/year=2026/month=11/day=11/031516.csv; data/year=2026/month=11/day=11/033018.csv; data/year=2026/month=11/day=11/034518.csv; data/year=2026/month=11/day=11/040016.csv; data/year=2026/month=11/day=11/041516.csv; ... +63 |
| kevykibbz/wind-turbine-scada-data-for-early-fault-detection | 10 | default/train/0000.parquet; default/train/0001.parquet; default/train/0002.parquet; default/train/0003.parquet; default/train/0004.parquet; default/train/0005.parquet; default/train/0006.parquet; default/train/0007.parquet; default/train/0008.parquet; default/train/0009.parquet |
| thuml/UTSD | 8 | default/train/0000.parquet; default/train/0001.parquet; default/train/0002.parquet; default/train/0003.parquet; default/train/0004.parquet; default/train/0005.parquet; default/train/0006.parquet; default/train/0007.parquet |
| tradecatlabs/binance-futures-ohlcv-2018-2026 | 10 | default/train/0000.parquet; default/train/0001.parquet; default/train/0002.parquet; default/train/0003.parquet; default/train/0004.parquet; default/train/0005.parquet; default/train/0006.parquet; default/train/0007.parquet; default/train/0008.parquet; default/train/0009.parquet |

## 2. Canonical datasets (A3)
Built: 337. Not built: 164.

### Not built, with the reason
| dataset_id | status | reason |
|---|---|---|
| aaron_freedman_banking_concentration_data | not built | no series left after the filters ({'constant': 42171, 'too_short': 46566, 'mostly_missing': 331}, min length 24) |
| abel_lab_finance1k | not built | nested: no frequency (no freq column, none in overrides.yaml or the profile) |
| adrien97_velib_monitoring_historical | not built | no series left after the filters ({'too_short': 1486}, min length 304) |
| aerostratospheric_spc_tornado_history | not built | no series left after the filters ({'mostly_missing': 78}, min length 158) |
| ai_iot_energybench_dgs | not built | no series left after the filters ({'too_short': 86398, 'mostly_missing': 552}, min length 158) |
| ai_iot_energybench_iflex | not built | no series left after the filters ({'mostly_missing': 4429}, min length 304) |
| ai_iot_energybench_neea | not built | no time column (set time_col in overrides.yaml) |
| ai_iot_energybench_rsl | not built | no series left after the filters ({'too_short': 47733}, min length 158) |
| ai_iot_energybench_save | not built | skipped in overrides.yaml: not published: ai-iot/EnergyBench on the Hub (checked 2026-11-13, 1947 files) has only Dataset_V0.0/metadata/missing/SAVE.csv for SAVE; no Energy-Load-Profiles/{15min,30min,Hourly} files exist for it, so no file-list fix or re-download can get them |
| ai_iot_energybench_ukst | not built | skipped in overrides.yaml: not published: ai-iot/EnergyBench on the Hub (checked 2026-11-13, 1947 files) has only Dataset_V0.0/metadata/missing/UKST.csv for UKST; no Energy-Load-Profiles/{15min,30min,Hourly} files exist for it, so no file-list fix or re-download can get them |
| aiacademy_kg_bishkek_transport | error | memory cap 11.0 GB exceeded (rss 12.2 GB) |
| alexdum_meteogate_archive | error | memory cap 30.0 GB exceeded (rss 30.3 GB) |
| algo0o_flight_delay_default | not built | frequency: fewer than two timestamps per series |
| alimaatouk_telecomts_default | not built | no series left after the filters ({'mostly_missing': 9, 'too_short': 5, 'constant': 22}, min length 304) |
| amandahattaway_bike_lane_lakehouse | not built | frequency: fewer than two timestamps per series |
| amgharr_femto_st_dataset | not built | skipped in overrides.yaml: FEMTO-ST bearing accelerometer at 25.6 kHz (bursts): not a forecasting series |
| amirshahbaz_cloudperftrace | not built | skipped in overrides.yaml: each list is one sample's feature vector (epoch, metrics, ...), not a series over time |
| andrewlee1807_gyeonggi | not built | no time axis (no time column and the frequency is unknown: set freq in overrides.yaml) |
| arh1p_instagram_top_accounts | not built | no series left after the filters ({'too_short': 12059}, min length 158) |
| astromlcore_astrom3dataset | not built | frequency: fewer than two timestamps per series |
| aurman_googletrendarchive_trending_queries | not built | no series left after the filters ({'mostly_missing': 116044, 'too_short': 53666}, min length 304) |
| autogluon_chronos_datasets_weatherbench_hourly_2m_temperature | error | memory cap 60.0 GB exceeded (rss 61.9 GB) |
| autonlab_timeseries_pile_anomaly_detection_tsb_uad_public | not built | skipped in overrides.yaml: TSB-UAD anomaly data (.out = value,label, no time); contains Yahoo (gated, excluded) and NAB |
| bayanat_labs_arab_trade_exposure | not built | frequency: fewer than two timestamps per series |
| bsebench_org_warwick_second_life_dm_2025_raw_raw_mirror | not built | skipped in overrides.yaml: battery cycler exports: every csv starts with a key-value header block of varying length (Measurement ID, Battery Name, ...); needs a dedicated reader |
| chengsenwang_cgtsf_ptf | not built | no target left: ['Idx'] are id / clock / sort columns (set targets in overrides.yaml) |
| chrimerss_rsi_benchmark_hydro | not built | no table left after the table / drop_parts overrides |
| cjc0013_ouroboros_ooi_ctd_benchmark | not built | frequency: fewer than two timestamps per series |
| cmu_ospo_covid_air_quality_ramp_datasets | not built | no series left after the filters ({'mostly_missing': 1}, min length 304) |
| cq_ai_financial_datasets_retail_and_wherehouse_sale_csv | not built | no series left after the filters ({'too_short': 15769}, min length 48) |
| cruiseresearchgroup_cgm_jepa_pretraining | not built | no series left after the filters ({'mostly_missing': 2}, min length 304) |
| dingdong_inc_freshretailnet_50k_default | error | TypeError: arg must be a list, tuple, 1-d array, or Series |
| dingdong_inc_freshretailnet_lt_unknown | error | TypeError: arg must be a list, tuple, 1-d array, or Series |
| dubairealestatedata_dubai_real_estate_sales_transactions | error | timeout after 20 min |
| earino_my_pricecatcher_weekly_increase | not built | frequency: fewer than two timestamps per series |
| earino_noaa_tide_flooding | not built | frequency: fewer than two timestamps per series |
| edilia12380_era5_daily_weather_western_europe | not built | skipped in overrides.yaml: the only table holds a PNG image (map), no data |
| emulated_inc_sci_arctic_sea_ice_forecast_later_years | not built | skipped in overrides.yaml: no files on disk after the download (0 files) |
| emulated_inc_sci_county_crop_yield_new_year | not built | no series left after the filters ({'too_short': 814}, min length 18) |
| emulated_inc_wikipedia_clickstream_enwiki | not built | skipped in overrides.yaml: 6 monthly snapshots (headerless referrer pairs): series of 6 points are below the minimum length |
| epi_eval_canada_fluwatch | not built | no series left after the filters ({'too_short': 17}, min length 60) |
| femartip_valencia_urban_mobility | not built | no series left after the filters ({'too_short': 10}, min length 158) |
| forgis_factorynet_factorywave_voraus_aursad_cnc | error | timeout after 45 min |
| gallicagram_atlas_motifs | not built | skipped in overrides.yaml: word-neighbour records stored as JSON strings; the dated tables are per-word motif summaries, not series |
| geocables_internet_latency_and_routing_events | not built | no series left after the filters ({'too_short': 140819}, min length 158) |
| gigigi7_ais_dataset_default | error | timeout after 45 min |
| gymprathap_chicago_crime_dataset | not built | frequency: fewer than two timestamps per series |
| haidang2405_aqshift_us_leakage_safe_air_quality | error | timeout after 20 min |
| happyman11_karnataka_sldc_preprocessed | not built | skipped in overrides.yaml: gated repo (no access); excluded (owner 2026-11-12) |
| hblim_weather_data | not built | frequency: fewer than two timestamps per series |
| hesandism_colombo_tea_auction_prices | not built | frequency: fewer than two timestamps per series |
| hsilvosa_noaa_ghcn_daily | not built | no series left after the filters ({}, min length None) |
| idomanteu_cs2_historical_item_prices_hourly_march_april_2026 | not built | no series left after the filters ({'mostly_missing': '34030 series, observed points min 0 / median 556 / max 570', 'too_short': '4 series, observed points min 0 / median 0 / max 2'}, frequency 15s (share 18%), min length 304, {'rows read': 104384314}) |
| jakov_bajs_zagreb_station_status | not built | frequency: fewer than two timestamps per series |
| jamesb7_fuel_prices_uk | not built | frequency: fewer than two timestamps per series |
| jeremycochoy_wikimedia_pageview_timeseries | not built | skipped in overrides.yaml: pre-cut 1025-step windows of STL residuals, z-normalised, no time or frequency (derived data) |
| jhumbl_dft_vehicle_licensing | not built | frequency: fewer than two timestamps per series |
| jwsouza13_eved | not built | no series left after the filters ({'too_short': 384}, min length 158) |
| kamilatr_weather_data_bavaria | error | Unreadable: csv chunk: In CSV column #0: CSV conversion error to double: invalid value '1EL7B' |
| karry001_movemelbourne_gtfsr | error | timeout after 20 min |
| kashif_app_flow_default | not built | no series left after the filters ({'mostly_missing': 1188, 'too_short': 1}, min length 304) |
| kavelaltd_historysaid_global_economic | not built | frequency: fewer than two timestamps per series |
| kevykibbz_wind_turbine_scada_data_for_early_fault_detection_wind_farm_a_b_c | error | timeout after 45 min |
| kpubdata_seoul_apartment_rent | not built | no series left after the filters ({'mostly_missing': 8877, 'too_short': 628, 'constant': 57}, min length 158) |
| kpubdata_seoul_bike_rent_month | not built | no series left after the filters ({'too_short': 2000}, min length 48) |
| krohnedigital_fuelcast | not built | no series left after the filters ({'too_short': 1591}, min length 304) |
| lemma_rca_nec_cloud_computing_preprocessed | not built | no series left after the filters ({'too_short': 45510}, min length 158) |
| lightcap_multi_domain_cloudflare_observability | not built | frequency: fewer than two timestamps per series |
| lilambd_security_daily | not built | frequency: fewer than two timestamps per series |
| lingbow_tiktok_video_engagement_200k | not built | frequency: fewer than two timestamps per series |
| lvinaud_arboviral_alerta_dados | not built | skipped in overrides.yaml: ML feature table (lags / rolling means of dengue / zika cases, no current-value target) and model files (.joblib) |
| malavnaik_dublin_bikes_weather_raw | not built | no series left after the filters ({'mostly_missing': 12943, 'too_short': 9613, 'constant': 2}, min length 304) |
| marcuslammers_vast_rtx3090_market_6mo | not built | frequency: fewer than two timestamps per series |
| michealsmitch_lithium_ion_cell_capacity_grading_process_curves_train | not built | no series left after the filters ({'too_short': 19345}, min length 304) |
| monitorbankrates_us_bank_rate_averages | not built | no series left after the filters ({'too_short': 4}, min length 158) |
| multimodaluniverse_tess | not built | skipped in overrides.yaml: light curves as a struct (time in BTJD days, flux, flux_err) with orbit gaps; needs a struct reader (v2) |
| naavrug_weather_geo_era5 | error | memory cap 30.0 GB exceeded (rss 30.8 GB) |
| nason_open_policing_database | not built | skipped in overrides.yaml: no files on disk after the download (0 files) |
| national_digital_au_ed_patients_seen_on_time | not built | frequency: fewer than two timestamps per series |
| national_digital_au_elective_surgery_waiting_times_by_specialty | not built | frequency: fewer than two timestamps per series |
| national_digital_nsw_traffic_volume_yearly | not built | frequency: fewer than two timestamps per series |
| national_digital_qld_median_rents_by_postcode | not built | no series left after the filters ({'too_short': 1}, min length 18) |
| national_digital_qld_shark_control_catch_numbers | not built | no series left after the filters ({'mostly_missing': 632, 'too_short': 257}, min length 48) |
| national_digital_vic_air_quality_hourly | not built | frequency: fewer than two timestamps per series |
| national_digital_vic_melbourne_water_daily_storage | not built | frequency: fewer than two timestamps per series |
| netop_5g_network_energy_consumption_ecstat_clstat_bsinfo | not built | skipped in overrides.yaml: gated repo (no access); excluded (owner 2026-11-12) |
| nju_zhangsq_rmisc_alicar | not built | no time column (set time_col in overrides.yaml) |
| nju_zhangsq_rmisc_batteryrul | not built | no series left after the filters ({'mostly_missing': 14}, min length 304) |
| nju_zhangsq_rmisc_booksperperson | not built | skipped in overrides.yaml: index values only (Index 2015 = 100), frequency not stated anywhere in the file |
| nju_zhangsq_rmisc_boschcnc | not built | skipped in overrides.yaml: CNC vibration at 2 kHz (bursts per machining run): not a forecasting series |
| nju_zhangsq_rmisc_chengdutaxi | not built | no series left after the filters ({'too_short': 19400}, min length 304) |
| nju_zhangsq_rmisc_chinaminutestock | error | memory cap 60.0 GB exceeded (rss 60.7 GB) |
| nju_zhangsq_rmisc_cryptocurrency | not built | skipped in overrides.yaml: file names are coded numbers, so the BTC / ETH / LTC / BCH drop rule (TIME overlap) cannot be applied; frequency unknown |
| nju_zhangsq_rmisc_csi500 | not built | no series left after the filters ({'mostly_missing': 500}, min length 304) |
| nju_zhangsq_rmisc_gaspipeline | not built | skipped in overrides.yaml: Modbus command / response log (cmd_address, rps_function), not a regular series |
| nju_zhangsq_rmisc_gassensordynamic | not built | skipped in overrides.yaml: gas sensor array under set-point steps; sample rate not stated in the file, and the setpoints drive the signal (lab experiment) |
| nju_zhangsq_rmisc_gassensortemperature | not built | skipped in overrides.yaml: gas sensor lab experiment under temperature set-points, like gassensordynamic (skipped): the set-points drive the signal (owner 2026-11-13) |
| nju_zhangsq_rmisc_lbnl | not built | no series left after the filters ({'mostly_missing': 6}, min length 304) |
| nju_zhangsq_rmisc_miniapp | not built | skipped in overrides.yaml: anonymised f1..f30, frequency unknown |
| nju_zhangsq_rmisc_motortemperature | not built | no series left after the filters ({'too_short': 69}, min length 18) |
| nju_zhangsq_rmisc_nab | not built | skipped in overrides.yaml: NAB anomaly benchmark: files of different cadences merged without timestamps (frequency unknown per file) |
| nju_zhangsq_rmisc_niftystock | error | OutOfBoundsDatetime: Cannot generate range with start=946684800000000000 and periods=235192 |
| nju_zhangsq_rmisc_oilwell | not built | skipped in overrides.yaml: 3W oil wells: same source as yahya2491_3w_pipeline, and file names are coded so simulated / drawn instances cannot be dropped |
| nju_zhangsq_rmisc_opsd_pv_wind | not built | skipped in overrides.yaml: OPSD files of mixed resolution (15 / 30 / 60 min) merged without time; columns ~100% blank; OPSD load is also in fev |
| nju_zhangsq_rmisc_pemscalifornia | not built | skipped in overrides.yaml: PEMS-SF flattened: one row per day, 12672 columns = 88 sensors x 144 ten-minute steps (f1..f12672); traffic also overlaps fev / GIFT |
| nju_zhangsq_rmisc_pvdaq | not built | no series left after the filters ({'mostly_missing': 15}, min length 304) |
| nju_zhangsq_rmisc_satellite | not built | skipped in overrides.yaml: anonymised normalised telemetry f1..f55 (most columns 70-96% blank), frequency unknown |
| nju_zhangsq_rmisc_t_drive | not built | no series left after the filters ({'too_short': 10336}, min length 158) |
| nju_zhangsq_rmisc_uk_dale | not built | skipped in overrides.yaml: UK-DALE meters at ~6 s with dropouts; without timestamps the row order is not a regular grid |
| nju_zhangsq_rmisc_ukeconomy | not built | skipped in overrides.yaml: 117 unrelated tables merged into one schema (columns 96-100% blank, numeric column names); no time and no common frequency |
| nju_zhangsq_rmisc_usairpollution | not built | skipped in overrides.yaml: several rows per site-day (one per instrument pairing, rows 0-3 are the same day) and the date column was removed: row order is not a daily grid |
| ohja_ai_rki_grippe_web_wochenbericht_2026_02_06 | not built | no series left after the filters ({'too_short': 5}, min length 18) |
| phanerozoic_noaa_ww3_multi1_points_wmo | not built | skipped in overrides.yaml: WAVEWATCH III point output in its own text format (multi_1.<buoy>.HIND_WMO.<yyyymm>) inside tar.gz; no reader |
| pracadata_mexico_fuel_prices | not built | no series left after the filters ({'too_short': 4474}, min length 158) |
| pytoshka_attacks_daily | not built | no series left after the filters ({'too_short': 9302}, min length 158) |
| qdrant_noaa_buoy | not built | no series left after the filters ({'too_short': 77}, min length 158) |
| rif_sql_time_series_uk_retail_supermarket_price_data | not built | no series left after the filters ({'too_short': 153217}, min length 158) |
| rogerdehe_klines_okx_futures_15m_1d | error | memory cap 60.0 GB exceeded (rss 63.0 GB) |
| ruiqil_spatialepibench_aucase_cacase_canpositivity_chnginpatient_chngoutpatient_cpradmissions_dvcli_hhshosp | not built | no series left after the filters ({'too_short': 5660, 'mostly_missing': 179}, min length 158) |
| salesforce_cloudops_tsf_alibaba_cluster_trace_2018 | not built | nested: no frequency (no freq column, none in overrides.yaml or the profile) |
| salesforce_cloudops_tsf_borg_cluster_data_2011 | not built | nested: no frequency (no freq column, none in overrides.yaml or the profile) |
| salesforce_lotsa_data_cif_2016_12 | error | OutOfBoundsDatetime: Out of bounds timestamp: -2193-10-01 00:00:00 with frequency 'ns' |
| salesforce_lotsa_data_godaddy | not built | no series left after the filters ({'too_short': 6270}, min length 48) |
| samlzy_mcc5_thu_motor | not built | skipped in overrides.yaml: motor vibration at kHz rates (headerless csv bursts per test condition): not a forecasting series |
| samuelm0422_ons_capacity_factor_dataset_full | not built | nested: no frequency (no freq column, none in overrides.yaml or the profile) |
| sbaiiiiii_looked_up | error | memory cap 60.0 GB exceeded (rss 61.1 GB) |
| soncco_historical_twitter_trends_spanish_locations_2020_2021 | not built | frequency: fewer than two timestamps per series |
| sureshbeekhani_salesforecastingwomartstore_default | not built | no series left after the filters ({'too_short': 188340}, min length 158) |
| tardellirs_paper_pulse_data | not built | frequency: fewer than two timestamps per series |
| taziksh_argo_profiles | not built | skipped in overrides.yaml: Argo vertical profiles (values over pressure levels, not over time); a float's own series is ~10-day irregular cycles |
| taziksh_cliwoc | not built | skipped in overrides.yaml: ship logbook records: irregular positions per voyage, no per-station regular series (owner 2026-11-13) |
| taziksh_ibtracs | not built | frequency: fewer than two timestamps per series |
| taziksh_itrdb_tree_rings | error | OutOfBoundsDatetime: Result is too large for pandas.Timedelta. Convert inputs to datetime.datetime with 'Timestamp.to_pydatetime()' before subtracting. |
| taziksh_usgs_earthquakes | error | timeout after 20 min |
| teamsynapse_sri_lanka_weekly_vegetable_retail_farmer_price_dataset | not built | no series left after the filters ({'too_short': 336}, min length 18) |
| thehooklab_nsw_property | error | timeout after 20 min |
| thulab_xauusd_default | not built | skipped in overrides.yaml: Apache IoTDB TsFile format (xauusd.tsfile): no reader without a new dependency |
| thuml_utsd_era5_surface_era5_pressure | not built | nested: no frequency (no freq column, none in overrides.yaml or the profile) |
| thuml_utsd_london_smart_meters_wind_farms_aus_electricity_demand_pedestrian_counts_web_traffic_sunspot | not built | nested: no frequency (no freq column, none in overrides.yaml or the profile) |
| thuml_utsd_sensordata | not built | nested: no frequency (no freq column, none in overrides.yaml or the profile) |
| thuml_utsd_tdbrain | not built | nested: no frequency (no freq column, none in overrides.yaml or the profile) |
| torchspatiotemporal_processeddatasets | not built | skipped in overrides.yaml: the series are .h5 / .npz inside the zips (not read in v1); only graph distance csvs are readable |
| tradecatlabs_binance_futures_ohlcv_2018_2026_candles_1m | error | memory cap 30.0 GB exceeded (rss 30.4 GB) |
| traders_lab_troveledger_daily_hourly_1_minute_files | not built | no series left after the filters ({'mostly_missing': 1439}, min length 304) |
| trentmkelly_polymarket_historical_data_market_snapshots | not built | no series left after the filters ({'too_short': 233, 'mostly_missing': 93}, min length 304) |
| uvathe_wellborn_pregnancy_wearable | not built | skipped in overrides.yaml: .mtn device files, format not documented; no reader |
| verisimilitudex_weather | not built | no series left after the filters ({'too_short': 318}, min length 60) |
| vincentgourbin_fuelinfrancedata | not built | no series left after the filters ({'mostly_missing': '46753 series, observed points min 0 / median 280 / max 3622', 'constant': '8695 series, observed points min 90 / median 3682 / max 6554', 'too_short': '5608 series, observed points min 0 / median 1 / max 154'}, frequency D (share 27%), min length 158, {'rows read': 71565721}) |
| vinicius_souza_antaq | not built | skipped in overrides.yaml: port-call and cargo records (one row per cargo item / berthing) in yearly partitions; a monthly aggregate needs a hand-written rule |
| wasurats_ecmwf_thailand_land_air_temperatures | error | memory cap 60.0 GB exceeded (rss 61.5 GB) |
| xijiazhao_casemotordataset_dated_folders | not built | no series left after the filters ({'too_short': 1}, min length 158) |
| xijiazhao_welding2_trial_1_trial_2 | not built | skipped in overrides.yaml: no files on disk after the download (0 files) |
| xxparthparekhxx_indian_stock_market_minute_data | error | memory cap 30.0 GB exceeded (rss 30.4 GB) |
| yahooresearch_ydata_labeled_time_series_anomalies_v1_0 | not built | skipped in overrides.yaml: gated repo (no access); excluded (owner 2026-11-12) |
| youngbongbong_mobility_dataset | error | memory cap 30.0 GB exceeded (rss 30.8 GB) |
| yugalinks_import_export_market_demand_v1 | not built | frequency: fewer than two timestamps per series |
| yuiseki_agera5_japan_monthly_temperature_1979_2026 | not built | no series left after the filters ({'too_short': 192}, min length 48) |
| yuiseki_estat_jumin_idou | error | timeout after 20 min |
| yuiseki_maddison_project | error | OutOfBoundsDatetime: Result is too large for pandas.Timedelta. Convert inputs to datetime.datetime with 'Timestamp.to_pydatetime()' before subtracting. |
| yuiseki_npa_traffic_accidents | not built | frequency: fewer than two timestamps per series |
| yuiseki_ucdp_ged_26_1 | not built | frequency: fewer than two timestamps per series |
| zaai_ai_hierarchical_time_series_datasets | not built | skipped in overrides.yaml: no files on disk after the download (0 files) |
| zalizedata_global_macro_trade_indicators_dataset | not built | frequency: fewer than two timestamps per series |
| zwn22_nc_crime | not built | no readable table (Chapel_Hill.csv.zip: Error: line contains NUL) |

### Per dataset
| dataset | domain | freq | band | series | points | len_min | len_med | len_max | nan | targets | cov_past | cov_known | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abmallick_heart_breath_sleep_stage_dataset | health | 30s | subhourly | 1000 | 1089176 | 472 | 1053 | 1679 | 5.0% | 1 | 12 | 0 | mit |
| aduitya9_aduitya_dextra_aqi_india | air_quality | D | daily | 15 | 10965 | 731 | 731 | 731 | 0.1% | 1 | 27 | 0 | other |
| afhubbard_gpu_prices | web_cloud | 12h | subdaily | 240 | 141585 | 335 | 600 | 600 | 43.2% | 1 | 5 | 1 | cc-by-4.0 |
| ai_iot_energybench_dth | energy | 15min | subhourly | 999 | 37977984 | 38016 | 38016 | 38016 | 0.5% | 1 | 0 | 0 | cc-by-sa-4.0 |
| ai_iot_energybench_goiener | energy | h | hourly | 492 | 24600000 | 50000 | 50000 | 50000 | 22.2% | 1 | 0 | 0 | cc-by-sa-4.0 |
| ai_iot_energybench_lcl | energy | 30min | subhourly | 1000 | 39727000 | 39727 | 39727 | 39727 | 21.7% | 1 | 0 | 0 | cc-by-sa-4.0 |
| ai_iot_energybench_norwegian | energy | h | hourly | 1000 | 13128000 | 13128 | 13128 | 13128 | 0.0% | 1 | 0 | 0 | cc-by-sa-4.0 |
| ai_iot_energybench_prayas_ceew | energy | 15min | subhourly | 109 | 5450000 | 50000 | 50000 | 50000 | 15.4% | 1 | 0 | 0 | cc-by-sa-4.0 |
| ai_iot_energybench_sgsc | energy | 30min | subhourly | 1000 | 42232000 | 42232 | 42232 | 42232 | 29.4% | 1 | 0 | 0 | cc-by-sa-4.0 |
| ai_iot_energybench_unicon | energy | 15min | subhourly | 55 | 2750000 | 50000 | 50000 | 50000 | 6.5% | 1 | 0 | 0 | cc-by-sa-4.0 |
| aikstockdata_korea_equity_daily | economics_finance | D | daily | 1000 | 476196 | 177 | 478 | 478 | 33.2% | 1 | 1 | 0 | other |
| ajto_goldencheetah | health | s | subhourly | 1000 | 5441483 | 363 | 3620 | 32478 | 6.4% | 1 | 3 | 0 | ? |
| akderekaan_de_power_forecast_data_entsoe_a75_a69_weather_runs | energy | 15min | subhourly | 175 | 502163 | 956 | 2976 | 2976 | 0.0% | 1 | 0 | 0 | cc-by-4.0 |
| akshaykarthick_india_weather_air_quality | air_quality | D | daily | 50 | 78000 | 1560 | 1560 | 1560 | 0.0% | 1 | 13 | 0 | cc-by-4.0 |
| alaminxpro_bd_food_prices | agriculture_nature | MS | monthly | 110 | 25960 | 236 | 236 | 236 | 0.0% | 1 | 5 | 1 | cc-by-4.0 |
| alanwgt_fipex_veiculos_brasil | retail_demand | MS | monthly | 1000 | 201192 | 48 | 204 | 311 | 0.1% | 1 | 0 | 0 | cc0-1.0 |
| alexroz_carbonfluxbench | agriculture_nature | D | daily | 493 | 4502076 | 9132 | 9132 | 9132 | 0.0% | 1 | 32 | 0 | mit |
| alinurdin_time_series_benchmark_eia_hourly_madrid_15min_webtris_15min_usgs_daily_usgs_continuous_hydat_daily_bom_daily_c | energy | h | hourly | 48 | 2382142 | 32142 | 50000 | 50000 | 0.2% | 1 | 1 | 0 | other |
| alwaysbyx_bear_data | energy | 5min | subhourly | 161 | 1285827 | 781 | 8640 | 8928 | 3.7% | 1 | 0 | 0 | cc-by-4.0 |
| angelosaadeh_velib | mobility | 61s | subhourly | 1000 | 20621534 | 19157 | 20623 | 20623 | 1.2% | 1 | 7 | 0 | etalab-2.0 |
| approjects_us_layoffs_monthly_time_series_warn_act | society | MS | monthly | 34 | 6718 | 62 | 183 | 455 | 14.6% | 1 | 2 | 0 | cc-by-4.0 |
| arailym_tleubayeva_nk_oil_well_sensor_monitoring_v2 | industrial | h | hourly | 10 | 65519 | 6551 | 6552 | 6552 | 3.1% | 1 | 1 | 0 | apache-2.0 |
| arailym_tleubayeva_nk_oil_well_sensor_monitoring_v2_train_test | industrial | h | hourly | 10 | 65519 | 6551 | 6552 | 6552 | 3.1% | 1 | 1 | 0 | apache-2.0 |
| arimancy_bts_flights_weather | mobility | min | subhourly | 2 | 100000 | 50000 | 50000 | 50000 | 38.9% | 1 | 26 | 6 | cc-by-4.0 |
| ashwingopalsamy_central_bank_policy_rates | economics_finance | D | daily | 7 | 51084 | 1607 | 8014 | 9777 | 0.0% | 1 | 4 | 0 | ? |
| autogluon_chronos_datasets_dominick | retail_demand | W-THU | weekly | 1000 | 297607 | 201 | 272 | 399 | 0.0% | 1 | 0 | 0 | other |
| autogluon_chronos_datasets_exchange_rate | economics_finance | D | daily | 8 | 84976 | 10622 | 10622 | 10622 | 28.6% | 1 | 0 | 0 | other |
| autogluon_chronos_datasets_mexico_city_bikes | mobility | h | hourly | 494 | 23963923 | 780 | 50000 | 50000 | 0.0% | 1 | 0 | 0 | other |
| autogluon_chronos_datasets_monash_cif_2016 | economics_finance | ME | monthly | 62 | 6733 | 57 | 120 | 120 | 0.0% | 1 | 0 | 0 | other |
| autogluon_chronos_datasets_monash_london_smart_meters | energy | 30min | subhourly | 1000 | 30064512 | 7392 | 30912 | 39360 | 0.0% | 1 | 0 | 0 | other |
| autogluon_chronos_datasets_monash_m1_monthly_quarterly_yearly_monash_m3_monthly_quarterly_yearly | economics_finance | ME | monthly | 1000 | 109787 | 48 | 126 | 150 | 0.0% | 1 | 0 | 0 | other |
| autogluon_chronos_datasets_monash_nn5_weekly_nn5 | economics_finance | D | daily | 108 | 85428 | 791 | 791 | 791 | 1.9% | 1 | 0 | 0 | other |
| autogluon_chronos_datasets_monash_pedestrian_counts | mobility | h | hourly | 66 | 2362219 | 576 | 45870 | 50000 | 0.0% | 1 | 0 | 0 | other |
| autogluon_chronos_datasets_monash_rideshare | mobility | h | hourly | 143 | 77363 | 541 | 541 | 541 | 39.6% | 1 | 14 | 0 | other |
| autogluon_chronos_datasets_monash_traffic | mobility | h | hourly | 862 | 15122928 | 17544 | 17544 | 17544 | 0.0% | 1 | 0 | 0 | other |
| autogluon_chronos_datasets_monash_weather | weather_climate | D | daily | 1000 | 14310741 | 1332 | 10828 | 50000 | 0.0% | 1 | 0 | 0 | other |
| autogluon_chronos_datasets_taxi_30min_taxi_1h | mobility | 30min | subhourly | 1000 | 1477813 | 1467 | 1487 | 1488 | 24.6% | 1 | 0 | 0 | other |
| autogluon_chronos_datasets_uber_tlc_hourly_uber_tlc_daily | mobility | h | hourly | 262 | 1138128 | 4344 | 4344 | 4344 | 0.0% | 1 | 0 | 0 | other |
| autogluon_chronos_datasets_ushcn_daily | weather_climate | D | daily | 1000 | 38745478 | 5906 | 42688 | 50000 | 7.6% | 1 | 4 | 0 | other |
| autogluon_chronos_datasets_weatherbench_daily | weather_climate | D | daily | 1000 | 14609985 | 14609 | 14610 | 14610 | 0.0% | 1 | 0 | 0 | other |
| autogluon_chronos_datasets_weatherbench_weekly | weather_climate | W-SUN | weekly | 1000 | 2087000 | 2087 | 2087 | 2087 | 0.0% | 1 | 0 | 0 | other |
| autogluon_chronos_datasets_wiki_daily_100k | web_cloud | D | daily | 1000 | 2741000 | 2741 | 2741 | 2741 | 0.0% | 1 | 0 | 0 | other |
| autogluon_chronos_datasets_wind_farms_hourly_wind_farms_daily | energy | h | hourly | 328 | 2810522 | 1715 | 8784 | 8784 | 17.1% | 1 | 0 | 0 | other |
| autonlab_timeseries_pile_forecasting | other | h | hourly | 862 | 15122928 | 17544 | 17544 | 17544 | 0.0% | 1 | 0 | 0 | mit |
| autonlab_timeseries_pile_forecasting_autoformer | other | h | hourly | 862 | 15122928 | 17544 | 17544 | 17544 | 0.0% | 1 | 0 | 0 | mit |
| ayushkhaire_all_stock_market_data_daily_updates | economics_finance | D | daily | 1000 | 4186854 | 168 | 4993 | 6156 | 31.2% | 1 | 0 | 0 | cc0-1.0 |
| battery_life_batterylife_raw_zn_ion_na_ion_calb | energy | s | subhourly | 177 | 8850000 | 50000 | 50000 | 50000 | 0.0% | 1 | 0 | 0 | mit |
| bayway_italian_open_timeseries | other | h | hourly | 18 | 833190 | 16559 | 50000 | 50000 | 2.5% | 1 | 0 | 0 | cc-by-4.0 |
| brandonhxrr_mexico_city_weather | weather_climate | D | daily | 3 | 150000 | 50000 | 50000 | 50000 | 1.0% | 1 | 0 | 0 | cc-by-nc-4.0 |
| bsebench_org_nasa_rw_2014_partitioned_by_cell_id | energy | s | subhourly | 2 | 100000 | 50000 | 50000 | 50000 | 49.0% | 1 | 6 | 0 | cc-by-4.0 |
| carlossilva1_us500_ticks_default | economics_finance | s | subhourly | 136 | 6633360 | 8340 | 50000 | 50000 | 37.3% | 1 | 0 | 0 | cc-by-4.0 |
| cashlessconsumer_chennai_rain_gauges | hydrology_ocean | D | daily | 30 | 34050 | 1135 | 1135 | 1135 | 0.0% | 1 | 9 | 0 | other |
| ceepr_bit_dianchi_water | hydrology_ocean | 4h | subdaily | 21 | 133708 | 2785 | 6568 | 6568 | 14.9% | 1 | 8 | 0 | cc-by-4.0 |
| cm3r_meteo_france_climatologie_quotidienne | weather_climate | D | daily | 1000 | 14907499 | 617 | 27394 | 50000 | 21.3% | 1 | 0 | 0 | etalab-2.0 |
| cmu_ospo_data_for_extended_range_water_demand_forecasts_driven_by_subseasonal_to_seasonal | hydrology_ocean | MS | monthly | 1 | 384 | 384 | 384 | 384 | 0.0% | 1 | 0 | 0 | cc-by-4.0 |
| cognitedata_ind_eval | industrial | s | subhourly | 1000 | 27524781 | 5215 | 26999 | 50000 | 0.0% | 1 | 0 | 0 | apache-2.0 |
| commoncrawl_statistics | web_cloud | 28D | monthly | 2 | 464 | 232 | 232 | 232 | 44.4% | 1 | 0 | 0 | ? |
| copernicus_argentina_estado_hidrico_argentina_2001_2026 | hydrology_ocean | MS | monthly | 527 | 163370 | 310 | 310 | 310 | 0.0% | 1 | 2 | 0 | ? |
| copernicus_argentina_lagunas_pampeanas_2018_2026 | hydrology_ocean | 5D | daily | 20 | 12765 | 636 | 639 | 640 | 40.6% | 1 | 3 | 0 | cc-by-4.0 |
| copernicus_argentina_parana_alturas_historicas_1900_2026 | hydrology_ocean | D | daily | 22 | 731471 | 7598 | 44487 | 46313 | 2.7% | 1 | 3 | 0 | cc-by-4.0 |
| cosuleabianca_eea_pm25_forecasting | air_quality | h | hourly | 38 | 1799854 | 17350 | 50000 | 50000 | 7.2% | 1 | 25 | 7 | cc-by-4.0 |
| daniel_dona_ree_data_default | energy | 5min | subhourly | 45 | 2250000 | 50000 | 50000 | 50000 | 0.0% | 1 | 0 | 0 | cc-by-4.0 |
| danielrosehill_jerusalem_air_quality_shabbat | air_quality | 5min | subhourly | 12 | 573440 | 23440 | 50000 | 50000 | 20.3% | 1 | 7 | 2 | cc-by-4.0 |
| danipaez_velib_tft_default | mobility | 20min | subhourly | 1000 | 9575089 | 518 | 9651 | 9651 | 28.9% | 1 | 5 | 3 | unknown |
| dar4huggingface_sniim_mexico_sample | agriculture_nature | D | daily | 10 | 3433 | 165 | 364 | 364 | 31.9% | 1 | 2 | 0 | cc-by-sa-4.0 |
| datadog_boom | web_cloud | 5min | subhourly | 138 | 1007199 | 547 | 3278 | 16384 | 29.2% | 101 | 0 | 0 | apache-2.0 |
| datonic_spain_aemet_historical_weather | weather_climate | D | daily | 14 | 433664 | 30976 | 30976 | 30976 | 0.0% | 1 | 0 | 0 | mit |
| datonic_world_development_indicators_default | economics_finance | YS | yearly | 1000 | 43220 | 18 | 40 | 66 | 3.5% | 1 | 0 | 0 | mit |
| dhivakarnaidusunkari_mandi_india_daily_prices_2001_2026 | agriculture_nature | D | daily | 1000 | 2656682 | 160 | 662 | 9036 | 36.5% | 1 | 2 | 0 | other |
| digitalhen_us_airport_wait_times | mobility | min | subhourly | 54 | 2700000 | 50000 | 50000 | 50000 | 2.3% | 1 | 0 | 0 | cc-by-4.0 |
| dmariaa70_metraq_air_quality | air_quality | h | hourly | 664 | 33164000 | 26304 | 50000 | 50000 | 0.0% | 1 | 3 | 0 | cc-by-4.0 |
| donny_guo_airline_delay | mobility | D | daily | 1000 | 1765964 | 159 | 1521 | 3652 | 10.7% | 1 | 14 | 0 | ? |
| duckermaster_thaiclim_district | weather_climate | D | daily | 910 | 8625276 | 1000 | 9497 | 9497 | 0.0% | 1 | 13 | 0 | cc-by-4.0 |
| earino_austin_911_response | society | D | daily | 3 | 1339 | 243 | 365 | 731 | 0.0% | 1 | 0 | 0 | cc0-1.0 |
| electricsheepafrica_africa_wfp_food_prices_for_uganda | agriculture_nature | MS | monthly | 293 | 23013 | 48 | 62 | 211 | 36.5% | 1 | 1 | 0 | cc-by-4.0 |
| electricsheepafrica_wfp_nigeria_food_prices | agriculture_nature | MS | monthly | 563 | 57598 | 51 | 103 | 234 | 38.3% | 1 | 1 | 0 | other |
| elskow_weather4cast | weather_climate | h | hourly | 10 | 437520 | 43752 | 43752 | 43752 | 0.0% | 1 | 29 | 0 | unlicense |
| emperor_mew_ooni_censorship_historical | web_cloud | D | daily | 634 | 1791150 | 1486 | 3220 | 3323 | 19.8% | 1 | 6 | 0 | ? |
| empulse_ai_empulse_street_fx_index | economics_finance | D | daily | 8 | 13832 | 1729 | 1729 | 1729 | 1.0% | 1 | 0 | 0 | cc-by-nc-4.0 |
| emulated_inc_battery_capacity_trajectories_dataset_1_nca_2_ncm_3_ncm_nca_zips | energy | 10s | subhourly | 852 | 42047776 | 37493 | 50000 | 50000 | 44.4% | 1 | 0 | 0 | cc-by-4.0 |
| emulated_inc_sci_airport_low_ceiling_visibility | weather_climate | 20h | subdaily | 1 | 2115 | 2115 | 2115 | 2115 | 10.0% | 1 | 32 | 0 | other |
| emulated_inc_sci_booster_gmps_regulation_forecast | industrial | 5s | subhourly | 700 | 12057120 | 12960 | 17280 | 17280 | 0.0% | 1 | 0 | 0 | cc-by-4.0 |
| emulated_inc_sci_lake_temperature_profiles_unmonitored | hydrology_ocean | D | daily | 1000 | 15806000 | 15806 | 15806 | 15806 | 0.0% | 1 | 0 | 0 | other |
| emulated_inc_sci_streamflow_ungauged_new_country | hydrology_ocean | D | daily | 1000 | 16070558 | 16070 | 16071 | 16071 | 20.0% | 1 | 12 | 0 | cc-by-4.0 |
| epi_eval_cdc_nrevss_rsv | health | W-SAT | weekly | 10 | 5210 | 521 | 521 | 521 | 0.0% | 1 | 2 | 0 | other |
| epi_eval_cdc_nssp | health | W-SAT | weekly | 50 | 6650 | 133 | 133 | 133 | 0.0% | 1 | 0 | 0 | other |
| epi_eval_delphi_flusurv | health | W-SAT | weekly | 3 | 2502 | 796 | 853 | 853 | 41.7% | 1 | 5 | 0 | other |
| epi_eval_global_mobility | mobility | D | daily | 1000 | 973766 | 740 | 974 | 974 | 3.3% | 1 | 5 | 0 | cc-by-4.0 |
| epi_eval_nhsn_hrd | health | W-SAT | weekly | 66 | 19668 | 298 | 298 | 298 | 3.6% | 1 | 13 | 0 | other |
| epi_eval_wikipedia_pageviews | web_cloud | D | daily | 1 | 3952 | 3952 | 3952 | 3952 | 0.0% | 1 | 0 | 0 | cc-by-sa-4.0 |
| feregrino_london_cycles | mobility | 902s | subhourly | 815 | 40480353 | 979 | 50000 | 50000 | 45.1% | 1 | 2 | 0 | cc-by-sa-3.0 |
| fidel_ts_bear_room | energy | 5min | subhourly | 161 | 1285827 | 781 | 8640 | 8928 | 3.7% | 1 | 0 | 0 | ? |
| fidel_ts_california_iso | energy | 5min | subhourly | 12 | 600000 | 50000 | 50000 | 50000 | 0.0% | 1 | 0 | 0 | ? |
| fidel_ts_canada_photovoltaics_plants | energy | 100s | subhourly | 11 | 520737 | 20737 | 50000 | 50000 | 0.0% | 1 | 0 | 0 | ? |
| fidel_ts_nyc_traffic_speed | mobility | 5min | subhourly | 89 | 4450000 | 50000 | 50000 | 50000 | 0.0% | 1 | 0 | 0 | ? |
| food_ai_nexus_global_food_safety_incidents | agriculture_nature | D | daily | 8 | 82846 | 4275 | 9206 | 16536 | 0.0% | 1 | 0 | 0 | cc-by-4.0 |
| fuel_prices_eu_european_fuel_prices | retail_demand | D | daily | 96 | 15264 | 159 | 159 | 159 | 3.8% | 1 | 9 | 0 | cc-by-4.0 |
| fulldecent_septa_regionalrail_otp | mobility | D | daily | 13 | 4381 | 234 | 365 | 366 | 3.6% | 1 | 0 | 0 | cc-by-4.0 |
| furusu_japan_max_temperature | weather_climate | D | daily | 907 | 18319859 | 388 | 17875 | 50000 | 0.8% | 1 | 0 | 0 | other |
| geoweaver_snowwaterequivalent_trainingdata | weather_climate | D | daily | 832 | 906949 | 363 | 1096 | 1096 | 3.4% | 1 | 32 | 0 | cc-by-4.0 |
| gft_ttm4hvac_source_all_train | energy | 15min | subhourly | 3 | 105120 | 35040 | 35040 | 35040 | 0.0% | 1 | 8 | 0 | mit |
| gladcolor_hourly_population_us | society | h | hourly | 1000 | 8760000 | 8760 | 8760 | 8760 | 0.1% | 1 | 0 | 0 | cc-by-nd-4.0 |
| globalsquiddatabase_tuna_fisheries_dataset | agriculture_nature | MS | monthly | 1000 | 653868 | 48 | 803 | 900 | 27.6% | 1 | 0 | 0 | ? |
| glucofmbench_glucofm_bench | health | 5min | subhourly | 311 | 2174333 | 591 | 2284 | 50000 | 2.1% | 1 | 0 | 0 | cc-by-nc-sa-4.0 |
| hackermango_hyderabaduniversity_hvac_dataset | energy | 10min | subhourly | 14 | 141120 | 10080 | 10080 | 10080 | 0.0% | 1 | 0 | 0 | cc-by-4.0 |
| hexquant_stocks_daily_price | economics_finance | D | daily | 1000 | 6044708 | 164 | 3922 | 23361 | 31.1% | 1 | 5 | 0 | other |
| hhs_official_nndss_weekly_data | health | W-SAT | weekly | 650 | 95856 | 60 | 156 | 156 | 19.6% | 1 | 0 | 0 | odbl |
| hhs_official_sars_cov_2_variant_proportions | health | W-FRI | weekly | 4 | 832 | 208 | 208 | 208 | 25.5% | 1 | 0 | 0 | US government work |
| hhs_official_weekly_counts_of_deaths_from_all_causes_by_jurisdi | health | W-SAT | weekly | 3 | 780 | 260 | 260 | 260 | 0.0% | 1 | 0 | 0 | unknown (HHS mirror) |
| hhs_official_weekly_provisional_counts_of_deaths_by_state_and_s | health | W-SAT | weekly | 17 | 3298 | 194 | 194 | 194 | 0.0% | 1 | 0 | 0 | ? |
| hhs_official_weekly_rates_of_laboratory_confirmed_rsv_hospitali | health | W-SAT | weekly | 13 | 5351 | 343 | 448 | 448 | 16.9% | 1 | 1 | 0 | ? |
| hhs_official_weekly_united_states_covid_19_hospitalization_metr | health | W-SAT | weekly | 67 | 12919 | 104 | 195 | 195 | 0.1% | 1 | 9 | 0 | US government work |
| hq_bench_quito_corpus_hour | web_cloud | h | hourly | 1000 | 15356000 | 15356 | 15356 | 15356 | 0.0% | 1 | 4 | 0 | cc-by-4.0 |
| hq_bench_quito_corpus_min | web_cloud | 10min | subhourly | 1000 | 5904000 | 5904 | 5904 | 5904 | 0.0% | 1 | 4 | 0 | cc-by-4.0 |
| hq_bench_quitobench_hour_min | web_cloud | h | hourly | 1000 | 6833404 | 984 | 984 | 15356 | 0.0% | 1 | 4 | 0 | cc-by-4.0 |
| hq_bench_quitobench_hour_min_r577 | web_cloud | h | hourly | 1000 | 6833404 | 984 | 984 | 15356 | 0.0% | 1 | 4 | 0 | cc-by-4.0 |
| hsilvosa_omie_electricity | energy | D | daily | 3 | 546 | 182 | 182 | 182 | 30.0% | 1 | 1 | 2 | other |
| hsilvosa_safecast_radiation | air_quality | 5s | subhourly | 1000 | 4004158 | 304 | 2725 | 50000 | 12.1% | 1 | 1 | 0 | cc0-1.0 |
| j_marcos_gs_brazil_solar_generation | energy | h | hourly | 1 | 50000 | 50000 | 50000 | 50000 | 0.0% | 1 | 0 | 0 | not stated |
| jainam03_rescast_100k_ecobee | energy | 15min | subhourly | 5 | 175200 | 35040 | 35040 | 35040 | 0.0% | 1 | 4 | 0 | cc-by-4.0 |
| jainam03_rescast_100k_heapo_timeseries_data | energy | 15min | subhourly | 1000 | 44424576 | 1920 | 47952 | 50000 | 2.9% | 1 | 0 | 0 | cc-by-4.0 |
| jainam03_rescast_100k_ideal | energy | 15min | subhourly | 1000 | 27725609 | 5894 | 26444 | 50000 | 1.8% | 1 | 0 | 0 | cc-by-4.0 |
| jainam03_rescast_100k_nest | energy | 15min | subhourly | 7 | 245280 | 35040 | 35040 | 35040 | 0.3% | 1 | 0 | 0 | cc-by-4.0 |
| jainam03_rescast_100k_refit | energy | 15min | subhourly | 20 | 956349 | 37647 | 50000 | 50000 | 0.0% | 1 | 0 | 0 | cc-by-4.0 |
| jhumbl_orr_station_usage | mobility | YS | yearly | 1000 | 27911 | 18 | 28 | 28 | 3.6% | 1 | 1 | 0 | other |
| jhumbl_police_crime_london | society | D | daily | 1000 | 4079818 | 182 | 4626 | 5754 | 0.0% | 1 | 0 | 0 | other |
| jhumbl_tfl_network_demand | mobility | D | daily | 431 | 619659 | 185 | 1461 | 1461 | 1.8% | 1 | 1 | 0 | other |
| jotone_arpa_osmer_weatherstations | weather_climate | h | hourly | 51 | 2539298 | 39298 | 50000 | 50000 | 6.8% | 1 | 14 | 0 | cc-by-sa-3.0 |
| jotone_pcfvg_weatherstations | hydrology_ocean | 15min | subhourly | 1000 | 36237864 | 359 | 40967 | 40968 | 0.8% | 1 | 0 | 0 | cc-by-4.0 |
| juliensimon_auroral_electrojet_index | space_science | D | daily | 1 | 2117 | 2117 | 2117 | 2117 | 0.0% | 1 | 0 | 0 | cc-by-nc-4.0 |
| juliensimon_celestrak_space_weather | space_science | D | daily | 29 | 889778 | 30682 | 30682 | 30682 | 17.4% | 1 | 0 | 0 | other |
| juliensimon_deep_space_probes | space_science | h | hourly | 1 | 50000 | 50000 | 50000 | 50000 | 0.0% | 1 | 0 | 0 | cc-by-4.0 |
| juliensimon_dst_index | space_science | h | hourly | 2 | 100000 | 50000 | 50000 | 50000 | 19.0% | 1 | 0 | 0 | cc-by-nc-4.0 |
| juliensimon_f107_solar_flux | space_science | D | daily | 2 | 58340 | 29170 | 29170 | 29170 | 0.1% | 1 | 0 | 0 | cc-by-4.0 |
| juliensimon_geomagnetic_kp_index | space_science | 3h | subdaily | 3 | 5757 | 1919 | 1919 | 1919 | 0.0% | 1 | 0 | 0 | cc-by-4.0 |
| juliensimon_goes_xray_flux | space_science | min | subhourly | 1 | 50000 | 50000 | 50000 | 50000 | 0.1% | 1 | 0 | 0 | cc-by-4.0 |
| juliensimon_iers_earth_orientation | space_science | D | daily | 9 | 118350 | 13150 | 13150 | 13150 | 1.2% | 1 | 0 | 0 | other |
| juliensimon_nasa_maven_kp_insitu | space_science | 8s | subhourly | 39 | 1950000 | 50000 | 50000 | 50000 | 35.2% | 1 | 0 | 0 | cc-by-4.0 |
| juliensimon_neutron_monitor_cosmic_rays | space_science | 2h | subdaily | 5 | 250000 | 50000 | 50000 | 50000 | 6.1% | 1 | 2 | 0 | cc-by-4.0 |
| juliensimon_omni_solar_wind_parameters | space_science | h | hourly | 35 | 1750000 | 50000 | 50000 | 50000 | 5.2% | 1 | 0 | 0 | cc-by-4.0 |
| juliensimon_solar_wind | space_science | min | subhourly | 12 | 600000 | 50000 | 50000 | 50000 | 17.7% | 1 | 0 | 0 | cc-by-4.0 |
| jwigginton_timeseries_1mn_sp500_default | economics_finance | min | subhourly | 1 | 391 | 391 | 391 | 391 | 0.0% | 1 | 4 | 0 | unknown |
| kawsersikder_bangladesh_stock_market_dataset | economics_finance | D | daily | 1000 | 5758951 | 691 | 6168 | 10016 | 37.5% | 1 | 0 | 0 | cc-by-4.0 |
| khajlk_copenhagen_air_quality_hourly | air_quality | h | hourly | 1 | 50000 | 50000 | 50000 | 50000 | 0.0% | 1 | 4 | 0 | cc-by-4.0 |
| kpubdata_seoul_apartment_trades | retail_demand | 31D | monthly | 4 | 1438 | 358 | 358 | 363 | 41.0% | 1 | 2 | 4 | other |
| leadlagforecasting_llf_github_default | web_cloud | D | daily | 1000 | 2967680 | 1827 | 2860 | 5048 | 0.0% | 1 | 2 | 0 | cc-by-4.0 |
| lxr11111_telemanom_smap_msl | space_science | s | subhourly | 97 | 548104 | 682 | 7406 | 8640 | 0.0% | 1 | 0 | 0 | bsd-3-clause |
| marplan6_batgrad_pozzato_2022 | energy | s | subhourly | 9 | 450000 | 50000 | 50000 | 50000 | 7.7% | 1 | 0 | 0 | cc-by-4.0 |
| matanzig_disney_theme_park_queue_dynamics | society | D | daily | 37 | 62567 | 1691 | 1691 | 1691 | 0.0% | 1 | 14 | 0 | mit |
| matrosovdani_moscow_parking_occupancy | mobility | 1799s | subhourly | 191 | 5088198 | 5566 | 28359 | 28359 | 4.5% | 1 | 2 | 0 | cc-by-4.0 |
| meganariley_daily_aqi | air_quality | h | hourly | 978 | 44611834 | 487 | 50000 | 50000 | 8.8% | 1 | 3 | 0 | odc-by |
| meteoswiss_peakweather | weather_climate | 10min | subhourly | 1000 | 45600816 | 41184 | 50000 | 50000 | 0.5% | 1 | 0 | 0 | cc-by-4.0 |
| mihnathul_celljar | industrial | 5s | subhourly | 952 | 11349536 | 304 | 648 | 50000 | 6.0% | 1 | 7 | 0 | cc-by-4.0 |
| misikoff_zillow_days_on_market | economics_finance | W-SAT | weekly | 879 | 278973 | 317 | 317 | 318 | 0.9% | 1 | 0 | 0 | other |
| misikoff_zillow_for_sale_listings | economics_finance | W-SAT | weekly | 874 | 273790 | 313 | 313 | 314 | 0.5% | 1 | 0 | 0 | other |
| misikoff_zillow_home_values | economics_finance | ME | monthly | 1000 | 289000 | 289 | 289 | 289 | 10.4% | 1 | 0 | 0 | other |
| misikoff_zillow_rentals | economics_finance | ME | monthly | 871 | 94068 | 108 | 108 | 108 | 17.1% | 1 | 0 | 0 | other |
| misikoff_zillow_sales | economics_finance | W-SAT | weekly | 148 | 123264 | 828 | 836 | 836 | 1.8% | 1 | 0 | 0 | other |
| modelpulse_model_pulse_data | web_cloud | D | daily | 1000 | 570180 | 162 | 596 | 837 | 14.9% | 1 | 2 | 0 | apache-2.0 |
| nakas_powderbench_truth | weather_climate | D | daily | 1000 | 4730095 | 1502 | 4037 | 9514 | 13.9% | 1 | 1 | 0 | other |
| narendersingh007_india_solar_benchmark_dataset | energy | h | hourly | 1 | 50000 | 50000 | 50000 | 50000 | 0.0% | 1 | 23 | 9 | cc-by-4.0 |
| nasa_ibm_ai4science_surya_bench_solarwind | space_science | h | hourly | 10 | 469125 | 43825 | 46912 | 50000 | 12.0% | 1 | 0 | 0 | cc-by-4.0 |
| national_digital_act_air_quality | air_quality | h | hourly | 2 | 100000 | 50000 | 50000 | 50000 | 6.3% | 1 | 16 | 0 | cc-by-4.0 |
| national_digital_au_births_by_age_of_mother | society | YS | yearly | 148 | 7400 | 50 | 50 | 50 | 0.0% | 1 | 0 | 0 | cc-by-4.0 |
| national_digital_au_water_storage_levels | hydrology_ocean | D | daily | 585 | 5843686 | 163 | 8008 | 44435 | 8.2% | 1 | 0 | 0 | other |
| national_digital_nt_crocodile_captures | agriculture_nature | MS | monthly | 5 | 1428 | 177 | 298 | 346 | 0.0% | 1 | 0 | 0 | cc-by-4.0 |
| national_digital_nt_fuel_prices_history | retail_demand | D | daily | 65 | 128250 | 215 | 2588 | 2588 | 2.4% | 1 | 9 | 0 | other |
| national_digital_rba_money_market_daily | economics_finance | D | daily | 5 | 28910 | 5782 | 5782 | 5782 | 39.0% | 1 | 0 | 0 | other |
| netop_gotsf_ds_default | web_cloud | h | hourly | 1000 | 840000 | 840 | 840 | 840 | 0.0% | 1 | 0 | 0 | mit |
| nevirim_wikipedia_attention_13lang | web_cloud | D | daily | 13 | 53963 | 4151 | 4151 | 4151 | 1.2% | 1 | 0 | 0 | cc0-1.0 |
| nju_zhangsq_rmisc_amarketchina | economics_finance | D | daily | 297 | 617801 | 165 | 2431 | 2431 | 0.0% | 1 | 5 | 0 | mit |
| nju_zhangsq_rmisc_applianceenergy | energy | 10min | subhourly | 1 | 19735 | 19735 | 19735 | 19735 | 0.0% | 1 | 25 | 0 | mit |
| nju_zhangsq_rmisc_britaincoal | energy | 5min | subhourly | 1 | 50000 | 50000 | 50000 | 50000 | 0.0% | 1 | 9 | 0 | mit |
| nju_zhangsq_rmisc_bts | mobility | 10min | subhourly | 1 | 50000 | 50000 | 50000 | 50000 | 1.3% | 1 | 32 | 0 | mit |
| nju_zhangsq_rmisc_buildingsbenchrealcsv | energy | h | hourly | 15 | 90496 | 3528 | 5322 | 8784 | 0.0% | 1 | 32 | 0 | mit |
| nju_zhangsq_rmisc_dominick | retail_demand | W | weekly | 1 | 393 | 393 | 393 | 393 | 0.0% | 1 | 0 | 0 | mit |
| nju_zhangsq_rmisc_ebayserver | web_cloud | s | subhourly | 1 | 50000 | 50000 | 50000 | 50000 | 0.0% | 1 | 24 | 0 | mit |
| nju_zhangsq_rmisc_era5hourlysinglelevels | weather_climate | h | hourly | 41 | 2050000 | 50000 | 50000 | 50000 | 0.0% | 1 | 11 | 0 | mit |
| nju_zhangsq_rmisc_frothflotation | industrial | h | hourly | 1 | 3278 | 3278 | 3278 | 3278 | 0.0% | 1 | 12 | 0 | mit |
| nju_zhangsq_rmisc_globalclimatechange | weather_climate | MS | monthly | 606 | 1332088 | 788 | 2145 | 3239 | 0.0% | 1 | 9 | 0 | mit |
| nju_zhangsq_rmisc_householdpower | energy | min | subhourly | 1 | 50000 | 50000 | 50000 | 50000 | 14.8% | 1 | 6 | 0 | mit |
| nju_zhangsq_rmisc_hungarianchickenpoxcases | health | W-MON | weekly | 1 | 522 | 522 | 522 | 522 | 0.0% | 1 | 0 | 0 | mit |
| nju_zhangsq_rmisc_keplerlightcurves | space_science | 30min | subhourly | 1000 | 4767000 | 4767 | 4767 | 4767 | 0.0% | 1 | 0 | 0 | mit |
| nju_zhangsq_rmisc_largest | mobility | 5min | subhourly | 1 | 50000 | 50000 | 50000 | 50000 | 0.0% | 1 | 32 | 0 | mit |
| nju_zhangsq_rmisc_metropt3 | industrial | 10s | subhourly | 1 | 50000 | 50000 | 50000 | 50000 | 0.0% | 1 | 14 | 0 | mit |
| nju_zhangsq_rmisc_metrotraffic | mobility | h | hourly | 1 | 50000 | 50000 | 50000 | 50000 | 23.7% | 1 | 4 | 0 | mit |
| nju_zhangsq_rmisc_mzvav | energy | min | subhourly | 5 | 146479 | 15839 | 21600 | 50000 | 0.0% | 1 | 32 | 0 | mit |
| nju_zhangsq_rmisc_occupancydetection | energy | min | subhourly | 3 | 20560 | 2665 | 8143 | 9752 | 20.0% | 1 | 5 | 0 | mit |
| nju_zhangsq_rmisc_opsd_household | energy | min | subhourly | 1 | 50000 | 50000 | 50000 | 50000 | 0.0% | 1 | 32 | 0 | mit |
| nju_zhangsq_rmisc_opsd_when2heat | energy | h | hourly | 1 | 50000 | 50000 | 50000 | 50000 | 0.0% | 1 | 32 | 0 | mit |
| nju_zhangsq_rmisc_pm25fivecities | air_quality | h | hourly | 1 | 22584 | 22584 | 22584 | 22584 | 0.0% | 1 | 18 | 0 | mit |
| nju_zhangsq_rmisc_pump | industrial | min | subhourly | 2 | 100000 | 50000 | 50000 | 50000 | 0.0% | 1 | 32 | 0 | mit |
| nju_zhangsq_rmisc_roomoccupancy | energy | 30s | subhourly | 1 | 10129 | 10129 | 10129 | 10129 | 0.0% | 1 | 16 | 0 | mit |
| nju_zhangsq_rmisc_servermachinedataset | web_cloud | min | subhourly | 28 | 708405 | 23687 | 23702 | 28743 | 0.0% | 1 | 32 | 0 | mit |
| nju_zhangsq_rmisc_shellhackathon | energy | min | subhourly | 1 | 50000 | 50000 | 50000 | 50000 | 0.0% | 1 | 14 | 0 | mit |
| nju_zhangsq_rmisc_smartmeteraus30m | energy | 30min | subhourly | 1000 | 24949847 | 933 | 27550 | 41897 | 0.0% | 1 | 0 | 0 | mit |
| nju_zhangsq_rmisc_smartmeteraus60m | energy | h | hourly | 1000 | 12567151 | 467 | 13871 | 20949 | 0.0% | 1 | 0 | 0 | mit |
| nju_zhangsq_rmisc_smartmeteruk30m | energy | 30min | subhourly | 1000 | 30098445 | 1778 | 31199 | 39725 | 0.0% | 1 | 0 | 0 | mit |
| nju_zhangsq_rmisc_smartmeteruk60m | energy | h | hourly | 1000 | 15204208 | 934 | 15660 | 19863 | 0.0% | 1 | 0 | 0 | mit |
| nju_zhangsq_rmisc_solar4seconds | energy | 4s | subhourly | 1 | 50000 | 50000 | 50000 | 50000 | 0.0% | 1 | 0 | 0 | mit |
| nju_zhangsq_rmisc_sp500 | economics_finance | D | daily | 496 | 602732 | 174 | 1258 | 1258 | 0.0% | 1 | 4 | 0 | mit |
| nju_zhangsq_rmisc_stockfactorscleaned | economics_finance | D | daily | 1000 | 4212842 | 166 | 3611 | 9041 | 31.2% | 1 | 32 | 0 | mit |
| nju_zhangsq_rmisc_stockmarketdata | economics_finance | D | daily | 5 | 9920 | 1984 | 1984 | 1984 | 0.0% | 1 | 32 | 0 | mit |
| nju_zhangsq_rmisc_swat | industrial | 5s | subhourly | 2 | 100000 | 50000 | 50000 | 50000 | 0.0% | 1 | 32 | 0 | mit |
| nju_zhangsq_rmisc_tetuanpowerconsumption | energy | 10min | subhourly | 1 | 50000 | 50000 | 50000 | 50000 | 0.0% | 1 | 7 | 0 | mit |
| nju_zhangsq_rmisc_timemmd | other | D | daily | 1 | 15979 | 15979 | 15979 | 15979 | 0.0% | 1 | 23 | 0 | mit |
| nju_zhangsq_rmisc_tushareetfdaily | economics_finance | D | daily | 1000 | 2099741 | 162 | 1861 | 9129 | 0.0% | 1 | 0 | 0 | mit |
| nju_zhangsq_rmisc_tushareindexdaily | economics_finance | D | daily | 169 | 781624 | 159 | 4664 | 9129 | 0.0% | 1 | 0 | 0 | mit |
| nju_zhangsq_rmisc_tusharestockdaily | economics_finance | D | daily | 1000 | 4299300 | 195 | 3469 | 9129 | 0.0% | 1 | 0 | 0 | mit |
| nju_zhangsq_rmisc_tusharestockweekly | economics_finance | D | daily | 1000 | 5202711 | 190 | 5084 | 9122 | 0.0% | 1 | 0 | 0 | mit |
| nju_zhangsq_rmisc_wadi | industrial | 5s | subhourly | 2 | 65701 | 15701 | 32850 | 50000 | 0.0% | 1 | 32 | 0 | mit |
| nju_zhangsq_rmisc_weeklyfuelpricesitaly | retail_demand | W-MON | weekly | 6 | 4980 | 830 | 830 | 830 | 0.0% | 1 | 3 | 0 | mit |
| nju_zhangsq_rmisc_weeklyroadfuelprices | retail_demand | W-MON | weekly | 1 | 881 | 881 | 881 | 881 | 0.0% | 1 | 0 | 0 | mit |
| nju_zhangsq_rmisc_windelec | energy | 15min | subhourly | 5 | 231840 | 46368 | 46368 | 46368 | 0.0% | 1 | 12 | 0 | mit |
| nju_zhangsq_rmisc_windpower4secs | energy | 4s | subhourly | 1 | 50000 | 50000 | 50000 | 50000 | 0.0% | 1 | 0 | 0 | mit |
| nju_zhangsq_rmisc_xiamenairquality | air_quality | h | hourly | 186 | 1551556 | 494 | 8760 | 8784 | 3.1% | 1 | 5 | 0 | mit |
| notadib_usa_corn_belt_crop_yield | agriculture_nature | YS | yearly | 490 | 18414 | 20 | 39 | 39 | 4.9% | 1 | 32 | 0 | mit |
| numix_stations_officielles_es | weather_climate | 10min | subhourly | 1000 | 4367945 | 1378 | 4464 | 4464 | 0.1% | 1 | 0 | 0 | cc-by-4.0 |
| olaflaitinen_baku_metro_station_flow | mobility | D | daily | 27 | 9447 | 337 | 337 | 395 | 0.0% | 1 | 11 | 11 | other |
| open_predico_benchmark | energy | 15min | subhourly | 130 | 4555200 | 35040 | 35040 | 35040 | 0.0% | 1 | 0 | 0 | other |
| openstef_liander2024_energy_forecasting_benchmark_load_measurements_weather_forecasts | energy | 15min | subhourly | 576 | 20291904 | 35229 | 35229 | 35229 | 0.0% | 1 | 0 | 0 | cc-by-4.0 |
| opensynth_tudelft_electricity_consumption_1_0_default | energy | 30min | subhourly | 1000 | 30365988 | 5952 | 31274 | 39727 | 0.1% | 1 | 0 | 0 | cdla-permissive-2.0 |
| orailix_ride_silver | mobility | D | daily | 1000 | 791190 | 164 | 802 | 1096 | 0.0% | 1 | 0 | 0 | cc-by-4.0 |
| pahzuzu_pah_env_001_serc_soil_signals | weather_climate | 30min | subhourly | 33 | 1650000 | 50000 | 50000 | 50000 | 24.0% | 1 | 6 | 0 | cc0-1.0 |
| phanerozoic_qiskit_calibration_drift | space_science | D | daily | 1 | 286 | 286 | 286 | 286 | 0.0% | 1 | 0 | 0 | cc-by-4.0 |
| phanerozoic_ww3_ndbc_pairs | hydrology_ocean | h | hourly | 174 | 4254571 | 1828 | 32347 | 33601 | 9.5% | 1 | 14 | 2 | cc0-1.0 |
| powerzoojax_powerzoodataset_ausgrid_zone_substation | energy | 15min | subhourly | 175 | 6115200 | 34944 | 34944 | 34944 | 0.3% | 1 | 0 | 1 | other |
| powerzoojax_powerzoodataset_gb_neso_demand_gb_gen_by_type_gb_market_mid_gb_forecast_actual_demand | energy | 30min | subhourly | 10 | 500000 | 50000 | 50000 | 50000 | 0.0% | 1 | 0 | 0 | other |
| pracadata_brazil_fuel_prices | retail_demand | D | daily | 1000 | 3464887 | 162 | 2097 | 7351 | 0.0% | 1 | 0 | 0 | other |
| prasad_gade05_india_upi_ecosystem_2018_2025 | economics_finance | QS | quarterly | 36 | 1008 | 28 | 28 | 28 | 0.0% | 1 | 3 | 0 | cdla-permissive-2.0 |
| previdengue_predict_climate_data | weather_climate | W-SAT | weekly | 1000 | 650000 | 650 | 650 | 650 | 0.0% | 1 | 6 | 0 | not stated |
| project_agml_usda_nass_crop_statistics | agriculture_nature | YS | yearly | 1000 | 55416 | 18 | 50 | 161 | 14.2% | 1 | 2 | 0 | cc0-1.0 |
| prospex_ch_swiss_commercial_registry_monthly_panel | economics_finance | D | daily | 1 | 2892 | 2892 | 2892 | 2892 | 0.0% | 1 | 0 | 0 | cc-by-4.0 |
| rebase_energy_heftcom2024_default | energy | 6h | subdaily | 5 | 26745 | 5349 | 5349 | 5349 | 2.8% | 1 | 0 | 0 | not stated (no card) |
| rebase_energy_nve_windpower_data_nve_windpower_timeseries_csv | energy | h | hourly | 64 | 3200000 | 50000 | 50000 | 50000 | 11.9% | 1 | 0 | 0 | other |
| rif_sql_uk_electricity_generation_mix_2019_2026 | energy | 30min | subhourly | 287 | 5643492 | 3740 | 17520 | 50000 | 0.0% | 1 | 0 | 0 | odc-by |
| rogendo_forex_dukas_copy_default | economics_finance | min | subhourly | 75 | 3750000 | 50000 | 50000 | 50000 | 29.7% | 1 | 0 | 0 | unknown |
| ron_the_code_telangana_time_series_2023_2025 | weather_climate | D | daily | 7 | 5117 | 731 | 731 | 731 | 0.0% | 1 | 0 | 0 | ? |
| roshangupta007_yieldformer_de_data | agriculture_nature | D | daily | 366 | 3188432 | 365 | 9497 | 9497 | 2.0% | 1 | 31 | 1 | other |
| rricc22_endomondo_hr_prediction_v2 | health | 6s | subhourly | 1000 | 598466 | 307 | 586 | 992 | 27.0% | 1 | 2 | 0 | mit |
| salesforce_lotsa_data_alibaba_cluster_trace_2018 | web_cloud | 5min | subhourly | 1000 | 1711770 | 576 | 1728 | 1728 | 0.0% | 2 | 6 | 0 | apache-2.0 |
| salesforce_lotsa_data_bdg_2_bear | energy | h | hourly | 91 | 1482312 | 8760 | 17544 | 17544 | 0.0% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_bdg_2_fox | energy | h | hourly | 135 | 2324568 | 8760 | 17544 | 17544 | 0.0% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_bdg_2_panther | energy | h | hourly | 105 | 919800 | 8760 | 8760 | 8760 | 0.0% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_bdg_2_rat | energy | h | hourly | 280 | 4728288 | 8760 | 17544 | 17544 | 0.0% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_beijing_air_quality | agriculture_nature | h | hourly | 12 | 420768 | 35064 | 35064 | 35064 | 1.6% | 11 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_beijing_subway_30min | mobility | 30min | subhourly | 276 | 433872 | 1572 | 1572 | 1572 | 42.7% | 2 | 11 | 0 | apache-2.0 |
| salesforce_lotsa_data_borealis | energy | h | hourly | 15 | 83269 | 3528 | 5555 | 7447 | 0.0% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_borg_cluster_data_2011 | web_cloud | 5min | subhourly | 1000 | 3748225 | 307 | 2260 | 7776 | 0.2% | 2 | 5 | 0 | apache-2.0 |
| salesforce_lotsa_data_cdc_fluview_ilinet | health | W-SUN | weekly | 72 | 61653 | 524 | 680 | 1359 | 11.6% | 5 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_cdc_fluview_who_nrevss | health | W-SUN | weekly | 73 | 40401 | 270 | 270 | 1359 | 0.0% | 4 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_china_air_quality | agriculture_nature | h | hourly | 411 | 3197888 | 1950 | 8759 | 8760 | 14.7% | 6 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_covid_mobility | mobility | D | daily | 362 | 148602 | 368 | 413 | 413 | 2.9% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_extended_web_traffic_with_missing | web_cloud | D | daily | 1000 | 2557000 | 2557 | 2557 | 2557 | 2.3% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_hzmetro | mobility | 15min | subhourly | 80 | 190160 | 2377 | 2377 | 2377 | 23.2% | 2 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_ideal | energy | h | hourly | 217 | 1255253 | 393 | 5007 | 16167 | 0.0% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_kaggle_web_traffic_weekly | web_cloud | W-SUN | weekly | 1000 | 114000 | 114 | 114 | 114 | 0.0% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_largest_2017 | mobility | 5min | subhourly | 1000 | 50000000 | 50000 | 50000 | 50000 | 0.3% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_largest_2018 | mobility | 5min | subhourly | 1000 | 50000000 | 50000 | 50000 | 50000 | 0.2% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_largest_2019 | mobility | 5min | subhourly | 1000 | 50000000 | 50000 | 50000 | 50000 | 0.2% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_largest_2020 | mobility | 5min | subhourly | 1000 | 50000000 | 50000 | 50000 | 50000 | 0.1% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_largest_2021 | mobility | 5min | subhourly | 1000 | 50000000 | 50000 | 50000 | 50000 | 0.1% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_lcl | energy | h | hourly | 712 | 9543360 | 609 | 13909 | 17544 | 0.0% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_london_smart_meters_with_missing | energy | 30min | subhourly | 1000 | 30236208 | 10320 | 30960 | 39504 | 0.0% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_los_loop | mobility | 5min | subhourly | 207 | 7094304 | 34272 | 34272 | 34272 | 0.0% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_m1_monthly | economics_finance | MS | monthly | 477 | 39373 | 48 | 69 | 132 | 0.0% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_nn5_daily_with_missing | economics_finance | D | daily | 108 | 79380 | 735 | 735 | 735 | 2.0% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_oikolab_weather | weather_climate | h | hourly | 8 | 400000 | 50000 | 50000 | 50000 | 0.0% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_pedestrian_counts | mobility | h | hourly | 66 | 2361355 | 552 | 45846 | 50000 | 0.0% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_pems03 | mobility | 5min | subhourly | 348 | 9120384 | 26208 | 26208 | 26208 | 0.0% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_pems04 | mobility | 5min | subhourly | 289 | 4910688 | 16992 | 16992 | 16992 | 0.0% | 3 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_pems07 | mobility | 5min | subhourly | 748 | 21111552 | 28224 | 28224 | 28224 | 0.0% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_pems08 | mobility | 5min | subhourly | 149 | 2660544 | 17856 | 17856 | 17856 | 0.0% | 3 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_pems_bay | mobility | 5min | subhourly | 325 | 16250000 | 50000 | 50000 | 50000 | 0.0% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_project_tycho | health | W-SUN | weekly | 1000 | 1141861 | 102 | 1076 | 3854 | 14.1% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_q_traffic | mobility | 15min | subhourly | 1000 | 5856000 | 5856 | 5856 | 5856 | 0.0% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_residential_load_power | energy | min | subhourly | 256 | 12676624 | 901 | 50000 | 50000 | 1.4% | 3 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_residential_pv_power | energy | min | subhourly | 221 | 10994293 | 901 | 50000 | 50000 | 1.8% | 3 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_rideshare_with_missing | mobility | h | hourly | 1000 | 373000 | 373 | 373 | 373 | 48.9% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_sceaux | energy | h | hourly | 1 | 34223 | 34223 | 34223 | 34223 | 0.0% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_shmetro | mobility | 15min | subhourly | 288 | 2536992 | 8809 | 8809 | 8809 | 23.8% | 2 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_smart | energy | h | hourly | 5 | 95709 | 8398 | 25919 | 26304 | 0.0% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_solar_power | energy | 4s | subhourly | 1 | 50000 | 50000 | 50000 | 50000 | 0.0% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_subseasonal | weather_climate | D | daily | 862 | 14197140 | 16470 | 16470 | 16470 | 0.0% | 4 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_subseasonal_precip | weather_climate | D | daily | 862 | 9760426 | 11323 | 11323 | 11323 | 0.0% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_sunspot_with_missing | agriculture_nature | D | daily | 1 | 50000 | 50000 | 50000 | 50000 | 0.0% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_taxi_30min | mobility | 30min | subhourly | 1000 | 896096 | 317 | 893 | 1469 | 0.0% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_traffic_hourly | mobility | h | hourly | 862 | 14978112 | 17376 | 17376 | 17376 | 0.0% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_uber_tlc_daily | mobility | D | daily | 257 | 46477 | 173 | 181 | 181 | 0.0% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_uber_tlc_hourly | mobility | h | hourly | 262 | 1129444 | 2626 | 4344 | 4344 | 0.0% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_vehicle_trips_with_missing | mobility | D | daily | 33 | 6068 | 182 | 182 | 213 | 18.8% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_weather | weather_climate | D | daily | 1000 | 14349126 | 1302 | 10798 | 50000 | 0.0% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_wiki_rolling_nips | web_cloud | D | daily | 1000 | 851730 | 792 | 852 | 912 | 0.0% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_wind_farms_with_missing | energy | min | subhourly | 299 | 14950000 | 50000 | 50000 | 50000 | 0.0% | 1 | 0 | 0 | apache-2.0 |
| salesforce_lotsa_data_wind_power | energy | 4s | subhourly | 1 | 50000 | 50000 | 50000 | 50000 | 0.0% | 1 | 0 | 0 | apache-2.0 |
| samuelm0422_hourly_electricity_demand_brazil_dataset | energy | h | hourly | 4 | 185280 | 46320 | 46320 | 46320 | 0.0% | 1 | 0 | 0 | cc-by-4.0 |
| sayanroy058_weather_time_series_forecasting | weather_climate | h | hourly | 14 | 700000 | 50000 | 50000 | 50000 | 0.0% | 1 | 0 | 0 | not stated |
| sevriukov_bank101 | economics_finance | MS | monthly | 1 | 290 | 290 | 290 | 290 | 27.9% | 1 | 0 | 0 | cc-by-4.0 |
| shashkovich_telecommunication_sms_time_series | web_cloud | D | daily | 2 | 330 | 165 | 165 | 165 | 0.0% | 1 | 0 | 0 | gpl-3.0 |
| showlcw_china_cpi_ppi_pmi | economics_finance | MS | monthly | 1 | 70 | 70 | 70 | 70 | 0.0% | 1 | 0 | 0 | other |
| skforecast_ashrae_daily | energy | D | daily | 1000 | 366000 | 366 | 366 | 366 | 0.0% | 1 | 6 | 0 | ? |
| skywalker1910_skynet_aqi_panel | air_quality | h | hourly | 33 | 939312 | 28464 | 28464 | 28464 | 0.0% | 1 | 7 | 0 | cc-by-4.0 |
| snowsadh_goes_omni_electron_flux_forecasting | space_science | 5min | subhourly | 80 | 4000000 | 50000 | 50000 | 50000 | 0.1% | 1 | 0 | 0 | cc-by-4.0 |
| solarbench_skippd_ground_ground_no_sky_img_era5_gfs | energy | min | subhourly | 2 | 100000 | 50000 | 50000 | 50000 | 48.9% | 1 | 0 | 0 | cc-by-4.0 |
| somemone_ac_transit_apc | mobility | D | daily | 1000 | 1099974 | 160 | 862 | 2709 | 0.0% | 1 | 0 | 0 | cc-by-4.0 |
| sovai_wikipedia_views_default | web_cloud | D | daily | 1000 | 3204000 | 3204 | 3204 | 3204 | 0.0% | 1 | 4 | 0 | unknown |
| steven0226_air_quality | air_quality | D | daily | 1000 | 10183055 | 389 | 11810 | 15890 | 5.3% | 1 | 1 | 0 | other |
| surfe_diem_wave_archive_usa_southwest | hydrology_ocean | h | hourly | 1000 | 7691577 | 531 | 8760 | 8785 | 8.8% | 1 | 0 | 0 | cc-by-4.0 |
| thanna94_us_building_permits | economics_finance | YS | yearly | 1000 | 46400 | 20 | 47 | 47 | 8.1% | 1 | 7 | 0 | cc0-1.0 |
| tharinduperera_weather_clustering_data | weather_climate | D | daily | 99 | 361647 | 3653 | 3653 | 3653 | 0.0% | 1 | 6 | 0 | cc-by-4.0 |
| thecharttruth_etf_data_prices | economics_finance | D | daily | 95 | 174399 | 883 | 1858 | 1858 | 31.2% | 1 | 4 | 0 | other |
| thermobench_thermobench_pure | space_science | 5min | subhourly | 30 | 1500000 | 50000 | 50000 | 50000 | 3.3% | 1 | 0 | 0 | cc-by-nc-4.0 |
| time_hd_anonymous_high_dimensional_time_series | other | s | subhourly | 1000 | 20000000 | 20000 | 20000 | 20000 | 0.0% | 1 | 0 | 0 | apache-2.0 |
| todopueblos_poblacion_municipios_espana | society | YS | yearly | 1000 | 30000 | 30 | 30 | 30 | 3.3% | 1 | 0 | 0 | cc-by-4.0 |
| valdovaldo1500_crime_brasil_rs_2022_2026 | society | MS | monthly | 1 | 49 | 49 | 49 | 49 | 0.0% | 1 | 0 | 0 | cc-by-4.0 |
| venvoo_openrouter_uptime | web_cloud | 897s | subhourly | 28 | 316414 | 380 | 12522 | 12525 | 39.2% | 1 | 4 | 0 | mit |
| wikimedia_community_english_wikipedia_pageviews_by_second | web_cloud | s | subhourly | 2 | 100000 | 50000 | 50000 | 50000 | 0.0% | 1 | 0 | 0 | cc0-1.0 |
| witgaw_metr_la | mobility | 5min | subhourly | 207 | 7089543 | 34249 | 34249 | 34249 | 0.0% | 1 | 32 | 0 | mit |
| witgaw_pems_bay | mobility | 5min | subhourly | 325 | 16250000 | 50000 | 50000 | 50000 | 0.0% | 1 | 32 | 0 | mit |
| wrpbr_pncp_compras_modalidade_dia_municipio | economics_finance | D | daily | 2 | 1950 | 975 | 975 | 975 | 24.0% | 1 | 0 | 0 | odbl |
| xairon_piezo_embedding_benchmark | hydrology_ocean | D | daily | 1000 | 9512845 | 3667 | 8212 | 21640 | 18.5% | 1 | 3 | 0 | apache-2.0 |
| xus_spain_fuel_prices | retail_demand | D | daily | 1000 | 357906 | 162 | 364 | 364 | 4.0% | 1 | 0 | 0 | cc-by-4.0 |
| yadavsaurabh_us_stocks_and_search_trends | economics_finance | D | daily | 1000 | 3627085 | 159 | 1508 | 19844 | 31.2% | 1 | 8 | 0 | other |
| yahya2491_3w_pipeline | industrial | s | subhourly | 1000 | 19990805 | 5215 | 21399 | 50000 | 0.2% | 1 | 0 | 0 | cc-by-4.0 |
| youngbongbong_30_od_subway_default | mobility | 30min | subhourly | 286 | 14268567 | 18567 | 50000 | 50000 | 16.6% | 1 | 2 | 0 | other |
| yuiseki_ilostat | economics_finance | QS | quarterly | 197 | 28795 | 106 | 146 | 315 | 7.4% | 1 | 0 | 0 | cc-by-4.0 |
| yuiseki_jma_rain_kanto | weather_climate | h | hourly | 117 | 5794224 | 17544 | 50000 | 50000 | 3.6% | 1 | 4 | 0 | cc-by-4.0 |
| yuiseki_kyoto_bear_sighting | agriculture_nature | D | daily | 6 | 21204 | 3534 | 3534 | 3534 | 36.6% | 1 | 0 | 0 | cc-by-4.0 |
| yuiseki_meta_range_maps_2022_05 | mobility | D | daily | 1000 | 395097 | 168 | 340 | 507 | 2.4% | 1 | 0 | 1 | cc-by-4.0 |
| yuiseki_pwt | economics_finance | YS | yearly | 168 | 11228 | 62 | 66 | 74 | 0.0% | 1 | 0 | 0 | cc-by-4.0 |
| yuiseki_un_wpp_wpp2024 | society | YS | yearly | 1000 | 82920 | 77 | 78 | 152 | 0.0% | 5 | 0 | 0 | other |
| yuiseki_undata_sdmx | economics_finance | QS | quarterly | 3 | 945 | 315 | 315 | 315 | 0.0% | 1 | 3 | 0 | other |
| yuiseki_undp_hdr | society | YS | yearly | 206 | 7004 | 34 | 34 | 34 | 0.0% | 1 | 0 | 0 | other |
| yzzpz_charged_data_10_csv_per_city | energy | h | hourly | 858 | 3768336 | 4392 | 4392 | 4392 | 0.0% | 1 | 0 | 0 | not stated |
| zait_ai_rusweather_434 | weather_climate | D | daily | 362 | 998758 | 2759 | 2759 | 2759 | 0.0% | 1 | 11 | 0 | cc-by-4.0 |
| zalizedata_us_grid_hourly_power_plants_dataset | energy | h | hourly | 29 | 1367520 | 8760 | 50000 | 50000 | 0.0% | 1 | 0 | 0 | cc-by-nc-4.0 |
| zer0contextlost_waterwatch_runoff | hydrology_ocean | MS | monthly | 1000 | 1426172 | 75 | 1507 | 1525 | 3.4% | 1 | 6 | 0 | cc-by-4.0 |

### Totals by domain
| domain | datasets | series | points |
|---|---|---|---|
| agriculture_nature | 16 | 5768 | 14959026 |
| air_quality | 13 | 3980 | 97088758 |
| economics_finance | 41 | 18238 | 50174011 |
| energy | 76 | 21207 | 582536752 |
| health | 18 | 5340 | 10700961 |
| hydrology_ocean | 14 | 7380 | 97919021 |
| industrial | 11 | 3680 | 71372259 |
| mobility | 51 | 25822 | 510706697 |
| other | 5 | 2743 | 51095025 |
| retail_demand | 10 | 4173 | 4472798 |
| society | 10 | 4429 | 13037815 |
| space_science | 18 | 1357 | 17108857 |
| weather_climate | 29 | 15739 | 223022766 |
| web_cloud | 25 | 15089 | 58405855 |

### Totals by frequency band
| band | datasets | series | points |
|---|---|---|---|
| daily | 91 | 44793 | 292806181 |
| hourly | 66 | 20972 | 343407190 |
| monthly | 21 | 8558 | 4433063 |
| quarterly | 3 | 236 | 30748 |
| subdaily | 6 | 275 | 559910 |
| subhourly | 116 | 45161 | 1155805205 |
| weekly | 24 | 7938 | 5228391 |
| yearly | 10 | 7012 | 329913 |

Total: 337 datasets, 134,945 series, 1,802,600,601 points.

## 3. Leakage audit (A5)
Datasets audited: 255. Flagged (> 1% of windows after series drops, excluded until decided): 5. Series dropped (> 10% of windows): 3468. Old rule (>= 3 windows) would flag: 86.

| dataset_id | series_dropped | share_after_drops | flagged | old_rule_windows | matched_task |
|---|---|---|---|---|---|
| ai_iot_energybench_dth | 229 | 0.0023837675395577 | False | 384355 | fev:kdd_cup_2022/10T/train-00000-of-00001.parquet |
| ai_iot_energybench_unicon | 1 | 0.0001304956461907 | False | 863 | fev:kdd_cup_2022/10T/train-00000-of-00001.parquet |
| ajto_goldencheetah | 38 | 0.0019888021790354 | False | 1542 | fev:kdd_cup_2022/10T/train-00000-of-00001.parquet |
| akderekaan_de_power_forecast_data_entsoe_a75_a69_weather_runs | 20 | 0.0133240346498311 | True | 1115 | fev:entsoe/15T/train-00000-of-00001.parquet |
| alwaysbyx_bear_data | 17 | 0.0061028495734313 | False | 58497 | fev:kdd_cup_2022/10T/train-00000-of-00001.parquet |
| battery_life_batterylife_raw_zn_ion_na_ion_calb | 160 | 0.0 | False | 1173078 | nan |
| bsebench_org_nasa_rw_2014_partitioned_by_cell_id | 2 | 0.0 | False | 818 | nan |
| carlossilva1_us500_ticks_default | 13 | 8.152971486341055e-06 | False | 32 | TIME:azure2019_I/5T/data-00000-of-00001.arrow |
| cognitedata_ind_eval | 814 | 0.0101225473439529 | True | 5587849 | fev:SZ_TAXI/15T/train-00000-of-00001.parquet |
| dmariaa70_metraq_air_quality | 57 | 0.003993506510714 | False | 73269 | fev:SZ_TAXI/15T/train-00000-of-00001.parquet |
| donny_guo_airline_delay | 1 | 0.0003087026071338 | False | 37 | GIFT-Eval:m4_daily/data-00000-of-00001.arrow |
| fidel_ts_bear_room | 17 | 0.0061028495734313 | False | 58497 | fev:kdd_cup_2022/10T/train-00000-of-00001.parquet |
| fidel_ts_california_iso | 1 | 0.0027676142227225 | False | 617 | fev:rohlik_sales/1D/train-00000-of-00001.parquet |
| glucofmbench_glucofm_bench | 27 | 0.0011569409960092 | False | 627409 | TIME:Coastal_T_S/5T/data-00000-of-00001.arrow |
| hhs_official_weekly_counts_of_deaths_from_all_causes_by_jurisdi | 1 | 0.0 | False | 8 | nan |
| hhs_official_weekly_provisional_counts_of_deaths_by_state_and_s | 1 | 0.0 | False | 9 | nan |
| hq_bench_quito_corpus_min | 1 | 7.405619493784765e-05 | False | 59 | fev:kdd_cup_2022/10T/train-00000-of-00001.parquet |
| jainam03_rescast_100k_ecobee | 1 | 0.0024005486968449 | False | 14321 | TIME:current_velocity/10T/data-00000-of-00001.arrow |
| jainam03_rescast_100k_heapo_timeseries_data | 1 | 0.0037257236891099 | False | 13283 | fev:kdd_cup_2022/10T/train-00000-of-00001.parquet |
| jainam03_rescast_100k_ideal | 1 | 0.0001409879613904 | False | 280 | fev:kdd_cup_2022/10T/train-00000-of-00001.parquet |
| jotone_pcfvg_weatherstations | 2 | 0.0015290173023173 | False | 4625 | fev:kdd_cup_2022/10T/train-00000-of-00001.parquet |
| juliensimon_deep_space_probes | 1 | 0.0 | False | 1905 | nan |
| juliensimon_iers_earth_orientation | 6 | 0.0 | False | 858 | nan |
| juliensimon_nasa_maven_kp_insitu | 38 | 0.0 | False | 291858 | nan |
| mihnathul_celljar | 180 | 0.0031243188258373 | False | 188736 | fev:kdd_cup_2022/10T/train-00000-of-00001.parquet |
| misikoff_zillow_home_values | 2 | 0.0016032064128256 | False | 28 | fev:uk_covid_utla/1D/train-00000-of-00001.parquet |
| national_digital_au_water_storage_levels | 98 | 0.0119105051564411 | True | 78574 | TIME:Coastal_T_S/5T/data-00000-of-00001.arrow |
| nju_zhangsq_rmisc_buildingsbenchrealcsv | 7 | 0.0063953488372093 | False | 795 | fev:SZ_TAXI/15T/train-00000-of-00001.parquet |
| nju_zhangsq_rmisc_keplerlightcurves | 2 | 0.000453974495508 | False | 219 | TIME:current_velocity/10T/data-00000-of-00001.arrow |
| nju_zhangsq_rmisc_mzvav | 2 | 0.0269973661106233 | True | 1395 | fev:jena_weather/10T/train-00000-of-00001.parquet |
| nju_zhangsq_rmisc_shellhackathon | 1 | 0.0 | False | 998 | nan |
| nju_zhangsq_rmisc_smartmeteraus30m | 90 | 0.0020031084692542 | False | 252347 | fev:SZ_TAXI/15T/train-00000-of-00001.parquet |
| nju_zhangsq_rmisc_smartmeteraus60m | 1 | 6.53563780776767e-05 | False | 53 | fev:kdd_cup_2022/10T/train-00000-of-00001.parquet |
| nju_zhangsq_rmisc_smartmeteruk30m | 9 | 0.0029413547441589 | False | 20965 | fev:SZ_TAXI/15T/train-00000-of-00001.parquet |
| nju_zhangsq_rmisc_smartmeteruk60m | 1 | 6.667781491508634e-05 | False | 173 | GIFT-Eval:bitbrains_fast_storage/5T/data-00000-of-00001.arrow |
| nju_zhangsq_rmisc_solar4seconds | 0 | 0.0526315789473684 | True | 2 | fev:entsoe/15T/train-00000-of-00001.parquet |
| nju_zhangsq_rmisc_swat | 2 | 0.0 | False | 791 | nan |
| numix_stations_officielles_es | 6 | 0.0023840712601519 | False | 904 | GIFT-Eval:solar/10T/data-00000-of-00001.arrow |
| open_predico_benchmark | 4 | 0.0018289894833104 | False | 1958 | GIFT-Eval:bitbrains_fast_storage/5T/data-00000-of-00001.arrow |
| opensynth_tudelft_electricity_consumption_1_0_default | 10 | 0.0004975284244576 | False | 6684 | GIFT-Eval:bitbrains_fast_storage/5T/data-00000-of-00001.arrow |
| skforecast_ashrae_daily | 2 | 5.273705305347537e-05 | False | 7 | fev:kdd_cup_2022/10T/train-00000-of-00001.parquet |
| thermobench_thermobench_pure | 1 | 0.0 | False | 64 | nan |
| xairon_piezo_embedding_benchmark | 11 | 0.0036703900772902 | False | 3117 | TIME:Coastal_T_S/5T/data-00000-of-00001.arrow |
| yahya2491_3w_pipeline | 707 | 0.00768050659355 | False | 3878644 | fev:uk_covid_utla/1D/train-00000-of-00001.parquet |
| yuiseki_un_wpp_wpp2024 | 530 | 0.0 | False | 4851 | nan |
| yzzpz_charged_data_10_csv_per_city | 352 | 0.0064830885463004 | False | 349878 | fev:SZ_TAXI/15T/train-00000-of-00001.parquet |

## 4. For your review
### Covariates proposed as known (known_candidate; used as past in v1)
| dataset | column |
|---|---|
| ashwingopalsamy_central_bank_policy_rates | is_change |
| electricsheepafrica_africa_wfp_food_prices_for_uganda | usdprice |
| electricsheepafrica_wfp_nigeria_food_prices | usdprice |
| alexroz_carbonfluxbench | forecast_albedo |
| dmariaa70_metraq_air_quality | is_interpolated |
| danielrosehill_jerusalem_air_quality_shabbat | is_friday |
| danielrosehill_jerusalem_air_quality_shabbat | is_shabbat |
| danielrosehill_jerusalem_air_quality_shabbat | is_yom_tov |
| danielrosehill_jerusalem_air_quality_shabbat | is_yom_kippur |
| danielrosehill_jerusalem_air_quality_shabbat | is_pre_shabbat |
| danielrosehill_jerusalem_air_quality_shabbat | is_post_havdalah |
| project_agml_usda_nass_crop_statistics | is_other_counties |
| nju_zhangsq_rmisc_timemmd | Weekly East Coast All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon) |
| nju_zhangsq_rmisc_timemmd | Weekly New England (PADD 1A) All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon) |
| nju_zhangsq_rmisc_timemmd | Weekly Central Atlantic (PADD 1B) All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon) |
| nju_zhangsq_rmisc_timemmd | Weekly Lower Atlantic (PADD 1C) All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon) |
| nju_zhangsq_rmisc_timemmd | Weekly Midwest All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon) |
| nju_zhangsq_rmisc_timemmd | Weekly Gulf Coast All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon) |
| nju_zhangsq_rmisc_timemmd | Weekly Rocky Mountain All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon) |
| nju_zhangsq_rmisc_timemmd | Weekly West Coast All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon) |
| autogluon_chronos_datasets_monash_rideshare | price_mean |
| autogluon_chronos_datasets_monash_rideshare | price_max |
| olaflaitinen_baku_metro_station_flow | is_transfer_station |
| hsilvosa_omie_electricity | price_eur_mwh |
| gft_ttm4hvac_source_all_train | Heating Setpoint (C) |
| gft_ttm4hvac_source_all_train | Cooling Setpoint (C) |
| gft_ttm4hvac_source_all_train | is_default |
| angelosaadeh_velib | is_installed |
| angelosaadeh_velib | is_renting |
| angelosaadeh_velib | is_returning |
| afhubbard_gpu_prices | is_spot |
| nju_zhangsq_rmisc_bts | Max_Temperature_Setpoint_Limit_4 |
| nju_zhangsq_rmisc_bts | Max_Temperature_Setpoint_Limit_5 |
| nju_zhangsq_rmisc_bts | Min_Temperature_Setpoint_Limit_6 |
| nju_zhangsq_rmisc_bts | Min_Temperature_Setpoint_Limit_7 |
| nju_zhangsq_rmisc_bts | Max_Temperature_Setpoint_Limit_8 |
| nju_zhangsq_rmisc_bts | Min_Temperature_Setpoint_Limit_9 |
| nju_zhangsq_rmisc_bts | Cooling_Temperature_Setpoint_15 |
| nju_zhangsq_rmisc_bts | Heating_Temperature_Setpoint_17 |
| nju_zhangsq_rmisc_bts | Room_Air_Temperature_Setpoint_18 |
| nju_zhangsq_rmisc_bts | Temperature_Setpoint_21 |
| nju_zhangsq_rmisc_bts | Max_Temperature_Setpoint_Limit_25 |
| nju_zhangsq_rmisc_bts | Max_Temperature_Setpoint_Limit_26 |
| nju_zhangsq_rmisc_bts | Min_Temperature_Setpoint_Limit_27 |
| nju_zhangsq_rmisc_bts | Min_Temperature_Setpoint_Limit_28 |
| nju_zhangsq_rmisc_bts | Max_Temperature_Setpoint_Limit_29 |
| nju_zhangsq_rmisc_bts | Min_Temperature_Setpoint_Limit_30 |
| nju_zhangsq_rmisc_bts | Cooling_Temperature_Setpoint_37 |
| alinurdin_time_series_benchmark_eia_hourly_madrid_15min_webtris_15min_usgs_daily_usgs_continuous_hydat_daily_bom_daily_c | demand_forecast |
| jainam03_rescast_100k_ecobee | Cooling Setpoint |
| jainam03_rescast_100k_ecobee | Heating Setpoint |
| matanzig_disney_theme_park_queue_dynamics | IS_BROKEN |
| matanzig_disney_theme_park_queue_dynamics | IS_BLACK_SWAN |
| dhivakarnaidusunkari_mandi_india_daily_prices_2001_2026 | Max_Price |
| dhivakarnaidusunkari_mandi_india_daily_prices_2001_2026 | Modal_Price |
| nju_zhangsq_rmisc_stockfactorscleaned | tech_price_range |
| duckermaster_thaiclim_district | is_extreme_rain_r95p |

### Covariates marked known (calendar / holiday name, or overrides.yaml)
| dataset | column |
|---|---|
| alaminxpro_bd_food_prices | month |
| roshangupta007_yieldformer_de_data | doy |
| cosuleabianca_eea_pm25_forecasting | day_of_week |
| cosuleabianca_eea_pm25_forecasting | day_of_month |
| cosuleabianca_eea_pm25_forecasting | is_weekend |
| cosuleabianca_eea_pm25_forecasting | hour_sin |
| cosuleabianca_eea_pm25_forecasting | hour_cos |
| cosuleabianca_eea_pm25_forecasting | month_sin |
| cosuleabianca_eea_pm25_forecasting | month_cos |
| danielrosehill_jerusalem_air_quality_shabbat | dow |
| danielrosehill_jerusalem_air_quality_shabbat | hour |
| olaflaitinen_baku_metro_station_flow | day_of_week |
| olaflaitinen_baku_metro_station_flow | day_of_year |
| olaflaitinen_baku_metro_station_flow | is_weekend |
| olaflaitinen_baku_metro_station_flow | day_of_week_sin |
| olaflaitinen_baku_metro_station_flow | day_of_week_cos |
| olaflaitinen_baku_metro_station_flow | month_sin |
| olaflaitinen_baku_metro_station_flow | month_cos |
| olaflaitinen_baku_metro_station_flow | target_next_day_entries |
| olaflaitinen_baku_metro_station_flow | year |
| olaflaitinen_baku_metro_station_flow | month |
| olaflaitinen_baku_metro_station_flow | day |
| hsilvosa_omie_electricity | year |
| hsilvosa_omie_electricity | hour |
| kpubdata_seoul_apartment_trades | build_year |
| kpubdata_seoul_apartment_trades | deal_year |
| kpubdata_seoul_apartment_trades | deal_month |
| kpubdata_seoul_apartment_trades | deal_day |
| afhubbard_gpu_prices | price_per_hour |
| phanerozoic_ww3_ndbc_pairs | month |
| phanerozoic_ww3_ndbc_pairs | hour |
| powerzoojax_powerzoodataset_ausgrid_zone_substation | fiscal_year |
| yuiseki_meta_range_maps_2022_05 | all_day_ratio_single_tile_users |
| narendersingh007_india_solar_benchmark_dataset | dayofyear |
| narendersingh007_india_solar_benchmark_dataset | weekday |
| narendersingh007_india_solar_benchmark_dataset | is_weekend |
| narendersingh007_india_solar_benchmark_dataset | hour_sin |
| narendersingh007_india_solar_benchmark_dataset | hour_cos |
| narendersingh007_india_solar_benchmark_dataset | month_sin |
| narendersingh007_india_solar_benchmark_dataset | month_cos |
| narendersingh007_india_solar_benchmark_dataset | dayofyear_sin |
| narendersingh007_india_solar_benchmark_dataset | dayofyear_cos |
| danipaez_velib_tft_default | minutes_of_day |
| danipaez_velib_tft_default | day_of_week |
| danipaez_velib_tft_default | is_public_holiday |
| arimancy_bts_flights_weather | day_of_week |
| arimancy_bts_flights_weather | day_of_year |
| arimancy_bts_flights_weather | days_to_nearest_holiday |
| arimancy_bts_flights_weather | origin_sched_departures_this_hour |
| arimancy_bts_flights_weather | dest_sched_arrivals_this_hour |
| arimancy_bts_flights_weather | origin_sched_departures_this_day |

### Frequency from timestamps differs from the profile
| dataset | from timestamps | profile | share at mode |
|---|---|---|---|
| autogluon_chronos_datasets_monash_nn5_weekly_nn5 | D | 1W | 88% |
| juliensimon_auroral_electrojet_index | D | 1h | 0% |
| arailym_tleubayeva_nk_oil_well_sensor_monitoring_v2 | h | 30min | 100% |
| arailym_tleubayeva_nk_oil_well_sensor_monitoring_v2_train_test | h | 30min | 100% |
| commoncrawl_statistics | 28D | 15min | 38% |
| fulldecent_septa_regionalrail_otp | D | 1Y | 100% |
| xus_spain_fuel_prices | D | 1M | 78% |
| hq_bench_quito_corpus_min | 10min | 1min | 100% |
| arimancy_bts_flights_weather | min | 1h | 37% |

### Irregular timestamps (under 80% of steps at the detected frequency)
| dataset | frequency | share at mode |
|---|---|---|
| food_ai_nexus_global_food_safety_incidents | D | 42% |
| hexquant_stocks_daily_price | D | 78% |
| dar4huggingface_sniim_mexico_sample | D | 75% |
| electricsheepafrica_africa_wfp_food_prices_for_uganda | MS | 58% |
| kawsersikder_bangladesh_stock_market_dataset | D | 74% |
| national_digital_rba_money_market_daily | D | 79% |
| misikoff_zillow_sales | W-SAT | 41% |
| misikoff_zillow_days_on_market | W-SAT | 67% |
| electricsheepafrica_wfp_nigeria_food_prices | MS | 64% |
| misikoff_zillow_for_sale_listings | W-SAT | 67% |
| aikstockdata_korea_equity_daily | D | 77% |
| project_agml_usda_nass_crop_statistics | YS | 77% |
| ayushkhaire_all_stock_market_data_daily_updates | D | 78% |
| carlossilva1_us500_ticks_default | s | 45% |
| copernicus_argentina_lagunas_pampeanas_2018_2026 | 5D | 38% |
| thecharttruth_etf_data_prices | D | 78% |
| nju_zhangsq_rmisc_occupancydetection | min | 65% |
| nju_zhangsq_rmisc_timemmd | D | 65% |
| juliensimon_f107_solar_flux | D | 35% |
| autogluon_chronos_datasets_taxi_30min_taxi_1h | 30min | 67% |
| kpubdata_seoul_apartment_trades | 31D | 5% |
| angelosaadeh_velib | 61s | 66% |
| matrosovdani_moscow_parking_occupancy | 1799s | 34% |
| afhubbard_gpu_prices | 12h | 4% |
| nju_zhangsq_rmisc_bts | 10min | 34% |
| commoncrawl_statistics | 28D | 38% |
| juliensimon_nasa_maven_kp_insitu | 8s | 80% |
| xus_spain_fuel_prices | D | 78% |
| hq_bench_quitobench_hour_min | h | 64% |
| hq_bench_quitobench_hour_min_r577 | h | 64% |
| venvoo_openrouter_uptime | 897s | 1% |
| ai_iot_energybench_lcl | 30min | 67% |
| hhs_official_sars_cov_2_variant_proportions | W-FRI | 67% |
| digitalhen_us_airport_wait_times | min | 26% |
| rricc22_endomondo_hr_prediction_v2 | 6s | 12% |
| surfe_diem_wave_archive_usa_southwest | h | 69% |
| emulated_inc_sci_airport_low_ceiling_visibility | 20h | 8% |
| ai_iot_energybench_sgsc | 30min | 67% |
| danipaez_velib_tft_default | 20min | 5% |
| yadavsaurabh_us_stocks_and_search_trends | D | 31% |
| dhivakarnaidusunkari_mandi_india_daily_prices_2001_2026 | D | 71% |
| yuiseki_ilostat | QS | 64% |
| datadog_boom | 5min | 31% |
| mihnathul_celljar | 5s | 58% |
| arimancy_bts_flights_weather | min | 37% |
| rogendo_forex_dukas_copy_default | min | 75% |
| nju_zhangsq_rmisc_stockfactorscleaned | D | 78% |
| feregrino_london_cycles | 902s | 1% |
| hsilvosa_safecast_radiation | 5s | 72% |
| emulated_inc_battery_capacity_trajectories_dataset_1_nca_2_ncm_3_ncm_nca_zips | 10s | 20% |

### Target chosen by a rule (check: overrides.yaml `targets` fixes it)
| dataset | targets | rule |
|---|---|---|
| hexquant_stocks_daily_price | open | first numeric value column (no column named target) |
| ashwingopalsamy_central_bank_policy_rates | rate_pct | first numeric value column (no column named target) |
| dar4huggingface_sniim_mexico_sample | Precio Mín | first numeric value column (no column named target) |
| jwigginton_timeseries_1mn_sp500_default | open | first numeric value column (no column named target) |
| electricsheepafrica_africa_wfp_food_prices_for_uganda | price | first numeric value column (no column named target) |
| notadib_usa_corn_belt_crop_yield | VAP_1 | first numeric value column (no column named target) |
| aduitya9_aduitya_dextra_aqi_india | y | column named target/value/y |
| alaminxpro_bd_food_prices | close | first numeric value column (no column named target) |
| akshaykarthick_india_weather_air_quality | pm2_5_mean | first numeric value column (no column named target) |
| electricsheepafrica_wfp_nigeria_food_prices | price | first numeric value column (no column named target) |
| epi_eval_cdc_nrevss_rsv | rsvpos | first numeric value column (no column named target) |
| alexroz_carbonfluxbench | dewpoint_temperature_2m | first numeric value column (no column named target) |
| khajlk_copenhagen_air_quality_hourly | temperature_2m | first numeric value column (no column named target) |
| roshangupta007_yieldformer_de_data | humidity | first numeric value column (no column named target) |
| national_digital_act_air_quality | no2 | first numeric value column (no column named target) |
| cosuleabianca_eea_pm25_forecasting | PM2.5 | first numeric value column (no column named target) |
| nju_zhangsq_rmisc_pm25fivecities | PM_City Station | first numeric value column (no column named target) |
| dmariaa70_metraq_air_quality | value | column named target/value/y |
| nju_zhangsq_rmisc_xiamenairquality | SO2浓度（μg/m3） | first numeric value column (no column named target) |
| skywalker1910_skynet_aqi_panel | pblh_m | first numeric value column (no column named target) |
| danielrosehill_jerusalem_air_quality_shabbat | value | column named target/value/y |
| aikstockdata_korea_equity_daily | close_krw | first numeric value column (no column named target) |
| project_agml_usda_nass_crop_statistics | value | column named target/value/y |
| epi_eval_delphi_flusurv | rate_overall | first numeric value column (no column named target) |
| steven0226_air_quality | mean | first numeric value column (no column named target) |
| copernicus_argentina_lagunas_pampeanas_2018_2026 | water_km2 | first numeric value column (no column named target) |
| prasad_gade05_india_upi_ecosystem_2018_2025 | total_txn_count | first numeric value column (no column named target) |
| approjects_us_layoffs_monthly_time_series_warn_act | notices | first numeric value column (no column named target) |
| thecharttruth_etf_data_prices | open | first numeric value column (no column named target) |
| nju_zhangsq_rmisc_occupancydetection | Temperature | first numeric value column (no column named target) |
| samuelm0422_hourly_electricity_demand_brazil_dataset | val_cargaenergiahomwmed | the only numeric value column |
| national_digital_au_births_by_age_of_mother | births | the only numeric value column |
| nju_zhangsq_rmisc_timemmd | OT | first numeric value column (no column named target) |
| epi_eval_cdc_nssp | percent_visits | the only numeric value column |
| nju_zhangsq_rmisc_metrotraffic | temp | first numeric value column (no column named target) |
| ceepr_bit_dianchi_water | TEM | first numeric value column (no column named target) |
| epi_eval_wikipedia_pageviews | views | the only numeric value column |
| skforecast_ashrae_daily | meter_reading | first numeric value column (no column named target) |
| fuel_prices_eu_european_fuel_prices | n | first numeric value column (no column named target) |
| nju_zhangsq_rmisc_applianceenergy | Appliances | first numeric value column (no column named target) |
| epi_eval_nhsn_hrd | totalconfflunewadm | first numeric value column (no column named target) |
| nju_zhangsq_rmisc_tetuanpowerconsumption | Temperature | first numeric value column (no column named target) |
| olaflaitinen_baku_metro_station_flow | daily_entries | first numeric value column (no column named target) |
| hhs_official_weekly_united_states_covid_19_hospitalization_metr | avg_adm_all_covid_confirmed | first numeric value column (no column named target) |
| nju_zhangsq_rmisc_globalclimatechange | AverageTemperature | first numeric value column (no column named target) |
| zait_ai_rusweather_434 | T2M | first numeric value column (no column named target) |
| hsilvosa_omie_electricity | energy_mwh | first numeric value column (no column named target) |
| juliensimon_neutron_monitor_cosmic_rays | count_rate | first numeric value column (no column named target) |
| nju_zhangsq_rmisc_householdpower | Global_active_power | first numeric value column (no column named target) |
| national_digital_nt_fuel_prices_history | diesel | first numeric value column (no column named target) |
| nju_zhangsq_rmisc_ebayserver | feature_0 | first numeric value column (no column named target) |
| gft_ttm4hvac_source_all_train | Outdoor Air Temperature (C) | first numeric value column (no column named target) |
| nakas_powderbench_truth | snow24_in | the only numeric value column |
| jhumbl_tfl_network_demand | EntryTapCount | first numeric value column (no column named target) |
| copernicus_argentina_estado_hidrico_argentina_2001_2026 | valor | first numeric value column (no column named target) |
| kpubdata_seoul_apartment_trades | exclusive_area_m2 | first numeric value column (no column named target) |
| angelosaadeh_velib | bikes | first numeric value column (no column named target) |
| matrosovdani_moscow_parking_occupancy | free_spaces | first numeric value column (no column named target) |
| elskow_weather4cast | temperature_2m (°C) | first numeric value column (no column named target) |
| hhs_official_weekly_rates_of_laboratory_confirmed_rsv_hospitali | Rate | first numeric value column (no column named target) |
| bayway_italian_open_timeseries | target | column named target/value/y |
| afhubbard_gpu_prices | gpu_count | first numeric value column (no column named target) |
| phanerozoic_ww3_ndbc_pairs | m_hs | first numeric value column (no column named target) |
| emperor_mew_ooni_censorship_historical | measurement_count | first numeric value column (no column named target) |
| nju_zhangsq_rmisc_solar4seconds | T1 | the only numeric value column |
| tharinduperera_weather_clustering_data | temperature_mean | first numeric value column (no column named target) |
| nju_zhangsq_rmisc_bts | Temperature_Setpoint_0 | first numeric value column (no column named target) |
| pahzuzu_pah_env_001_serc_soil_signals | vswc | first numeric value column (no column named target) |
| thanna94_us_building_permits | Months_Reported | first numeric value column (no column named target) |
| furusu_japan_max_temperature | max_temp | the only numeric value column |
| national_digital_au_water_storage_levels | volume_ml | the only numeric value column |
| powerzoojax_powerzoodataset_ausgrid_zone_substation | load_mw | first numeric value column (no column named target) |
| arailym_tleubayeva_nk_oil_well_sensor_monitoring_v2 | value | column named target/value/y |
| copernicus_argentina_parana_alturas_historicas_1900_2026 | altura_media_m | first numeric value column (no column named target) |
| nju_zhangsq_rmisc_opsd_when2heat | LV_COP_ASHP_floor | first numeric value column (no column named target) |
| arailym_tleubayeva_nk_oil_well_sensor_monitoring_v2_train_test | value | column named target/value/y |
| nju_zhangsq_rmisc_opsd_household | DE_KN_residential2_circulation_pump | first numeric value column (no column named target) |
| youngbongbong_30_od_subway_default | od_out_cnt | first numeric value column (no column named target) |
| zer0contextlost_waterwatch_runoff | runoff_mm | first numeric value column (no column named target) |
| datonic_world_development_indicators_default | indicator_value | the only numeric value column |
| epi_eval_global_mobility | retail_and_recreation | first numeric value column (no column named target) |
| wikimedia_community_english_wikipedia_pageviews_by_second | requests | the only numeric value column |
| nju_zhangsq_rmisc_smartmeteruk60m | target | column named target/value/y |
| xus_spain_fuel_prices | price | the only numeric value column |
| alinurdin_time_series_benchmark_eia_hourly_madrid_15min_webtris_15min_usgs_daily_usgs_continuous_hydat_daily_bom_daily_c | demand | first numeric value column (no column named target) |
| jainam03_rescast_100k_ecobee | Indoor Temp | first numeric value column (no column named target) |
| jotone_arpa_osmer_weatherstations | rr | first numeric value column (no column named target) |
| yuiseki_undata_sdmx | OBS_VALUE | first numeric value column (no column named target) |
| yuiseki_jma_rain_kanto | precip_mm | first numeric value column (no column named target) |
| nju_zhangsq_rmisc_buildingsbenchrealcsv | power | first numeric value column (no column named target) |
| hq_bench_quitobench_hour_min | ind_1 | first numeric value column (no column named target) |
| hq_bench_quitobench_hour_min_r577 | ind_1 | first numeric value column (no column named target) |
| nju_zhangsq_rmisc_smartmeteraus60m | target | column named target/value/y |
| venvoo_openrouter_uptime | status | first numeric value column (no column named target) |
| witgaw_metr_la | x_t-11_d0 | first numeric value column (no column named target) |
| abmallick_heart_breath_sleep_stage_dataset | heart_rate | first numeric value column (no column named target) |
| xairon_piezo_embedding_benchmark | niveau_nappe_eau | first numeric value column (no column named target) |
| sovai_wikipedia_views_default | views | first numeric value column (no column named target) |
| nju_zhangsq_rmisc_smartmeteruk30m | target | column named target/value/y |
| yuiseki_meta_range_maps_2022_05 | all_day_bing_tiles_visited_relative_change | first numeric value column (no column named target) |
| hq_bench_quito_corpus_hour | ind_1 | first numeric value column (no column named target) |
| modelpulse_model_pulse_data | dl30 | first numeric value column (no column named target) |
| emulated_inc_sci_airport_low_ceiling_visibility | forecast_hour | first numeric value column (no column named target) |
| matanzig_disney_theme_park_queue_dynamics | DEB_TIME_HOUR | first numeric value column (no column named target) |
| nju_zhangsq_rmisc_era5hourlysinglelevels | u100 | first numeric value column (no column named target) |
| narendersingh007_india_solar_benchmark_dataset | MO | first numeric value column (no column named target) |
| danipaez_velib_tft_default | available_bikes | first numeric value column (no column named target) |
| geoweaver_snowwaterequivalent_trainingdata | SWE | first numeric value column (no column named target) |
| yadavsaurabh_us_stocks_and_search_trends | Open | first numeric value column (no column named target) |
| nju_zhangsq_rmisc_smartmeteraus30m | target | column named target/value/y |
| hq_bench_quito_corpus_min | ind_1 | first numeric value column (no column named target) |
| dhivakarnaidusunkari_mandi_india_daily_prices_2001_2026 | Min_Price | first numeric value column (no column named target) |
| donny_guo_airline_delay | op_carrier_fl_num | first numeric value column (no column named target) |
| meganariley_daily_aqi | value | column named target/value/y |
| opensynth_tudelft_electricity_consumption_1_0_default | target | column named target/value/y |
| mihnathul_celljar | current_A | first numeric value column (no column named target) |
| arimancy_bts_flights_weather | dep_delay_min | first numeric value column (no column named target) |
| nju_zhangsq_rmisc_stockfactorscleaned | target_value | column named target/value/y |
| feregrino_london_cycles | bikes | first numeric value column (no column named target) |
| hsilvosa_safecast_radiation | value | column named target/value/y |
| yuiseki_un_wpp_wpp2024 | AgeGrpStart, AgeGrpSpan, PopMale, PopFemale | first numeric value column (no column named target) |
| witgaw_pems_bay | x_t-11_d0 | first numeric value column (no column named target) |
| duckermaster_thaiclim_district | precip_mm | first numeric value column (no column named target) |
| bsebench_org_nasa_rw_2014_partitioned_by_cell_id | voltage_V | first numeric value column (no column named target) |
| jotone_pcfvg_weatherstations | value | column named target/value/y |

### Series dropped by a catalog rule (drop_items) or by reference ids (ref_ids)
| dataset | rule | series dropped |
|---|---|---|
| ashwingopalsamy_central_bank_policy_rates | drop_items | 2 |
| yuiseki_undp_hdr | drop_items | 0 |
| tharinduperera_weather_clustering_data | drop_items | 1 |
| datonic_world_development_indicators_default | drop_items | 12667 |
| zalizedata_us_grid_hourly_power_plants_dataset | drop_items | 0 |
| yahya2491_3w_pipeline | drop_items | 0 |
| datadog_boom | ref_ids | 15 |
| emulated_inc_sci_streamflow_ungauged_new_country | drop_items | 0 |
| yuiseki_un_wpp_wpp2024 | drop_items | 0 |

### Cumulative targets (differenced; or for review: set diff: [...] or diff: false in overrides.yaml)
| dataset | differenced | review |
|---|---|---|
| prasad_gade05_india_upi_ecosystem_2018_2025 |  | total_txn_count |
| juliensimon_dst_index |  | target |
| epi_eval_nhsn_hrd |  | totalconfflunewadm |
| hhs_official_weekly_counts_of_deaths_from_all_causes_by_jurisdi |  | target |
| narendersingh007_india_solar_benchmark_dataset |  | MO |
| leadlagforecasting_llf_github_default | target, cumulative_pushes, cumulative_forks |  |
| time_hd_anonymous_high_dimensional_time_series |  | target |
| yuiseki_un_wpp_wpp2024 |  | PopTotal |

### Synthetic time (calendar position not real)
| dataset | frequency | time rule |
|---|---|---|
| nju_zhangsq_rmisc_amarketchina | D | synthetic time from row order |
| nju_zhangsq_rmisc_sp500 | D | synthetic time from row order |
| nju_zhangsq_rmisc_roomoccupancy | 30s | synthetic time from row order |
| nju_zhangsq_rmisc_stockmarketdata | D | synthetic time from row order |
| nju_zhangsq_rmisc_weeklyroadfuelprices | W-MON | synthetic time from row order |
| nju_zhangsq_rmisc_hungarianchickenpoxcases | W-MON | synthetic time from row order |
| nju_zhangsq_rmisc_weeklyfuelpricesitaly | W-MON | synthetic time from row order |
| nju_zhangsq_rmisc_frothflotation | h | synthetic time from row order |
| nju_zhangsq_rmisc_dominick | W | synthetic time from row order |
| nju_zhangsq_rmisc_swat | 5s | synthetic time from row order |
| nju_zhangsq_rmisc_windpower4secs | 4s | synthetic time from row order |
| nju_zhangsq_rmisc_mzvav | min | synthetic time from row order |
| nju_zhangsq_rmisc_shellhackathon | min | synthetic time from row order |
| nju_zhangsq_rmisc_windelec | 15min | synthetic time from row order |
| lxr11111_telemanom_smap_msl | s | numeric time column |
| nju_zhangsq_rmisc_metropt3 | 10s | synthetic time from row order |
| nju_zhangsq_rmisc_britaincoal | 5min | synthetic time from row order |
| fidel_ts_canada_photovoltaics_plants | 100s | numeric time column |
| nju_zhangsq_rmisc_pump | min | synthetic time from row order |
| nju_zhangsq_rmisc_wadi | 5s | synthetic time from row order |
| ajto_goldencheetah | s | numeric time column |
| nju_zhangsq_rmisc_keplerlightcurves | 30min | synthetic time from row order |
| nju_zhangsq_rmisc_servermachinedataset | min | synthetic time from row order |
| netop_gotsf_ds_default | h | synthetic time from row order |
| nju_zhangsq_rmisc_largest | 5min | synthetic time from row order |
| bsebench_org_nasa_rw_2014_partitioned_by_cell_id | s | numeric time column |

### Event tables (aggregation rule)
| dataset | step | aggregate |
|---|---|---|
| nju_zhangsq_rmisc_tushareetfdaily | D | count of rows |
| food_ai_nexus_global_food_safety_incidents | D | count of rows |
| national_digital_nt_crocodile_captures | MS | count of rows |
| nju_zhangsq_rmisc_tushareindexdaily | D | count of rows |
| juliensimon_auroral_electrojet_index | D | count of rows |
| prospex_ch_swiss_commercial_registry_monthly_panel | D | count of rows |
| shashkovich_telecommunication_sms_time_series | D | count of rows |
| fidel_ts_canada_photovoltaics_plants | D | count of rows |
| nju_zhangsq_rmisc_tusharestockweekly | D | count of rows |
| pracadata_brazil_fuel_prices | D | count of rows |
| phanerozoic_qiskit_calibration_drift | D | count of rows |
| orailix_ride_silver | D | count of rows |
| earino_austin_911_response | D | count of rows |
| nju_zhangsq_rmisc_tusharestockdaily | D | count of rows |
| jhumbl_police_crime_london | D | count of rows |
| somemone_ac_transit_apc | D | count of rows |


# Milestone B: tasks

Tasks: 1145 from 272 source datasets, 599 frequency variants.

## Tasks per tier and split
| tier | split | tasks | datasets |
|---|---|---|---|
| full | tuning | 829 | 203 |
| full | locked | 316 | 69 |
| core | tuning | 394 | 203 |
| core | locked | 141 | 69 |
| fast | tuning | 150 | 115 |
| fast | locked | 50 | 39 |

## Tasks (full tier) by frequency band
| band | locked | tuning |
|---|---|---|
| daily | 58 | 172 |
| hourly | 111 | 257 |
| monthly | 4 | 20 |
| quarterly | 0 | 5 |
| subdaily | 0 | 2 |
| subhourly | 80 | 204 |
| weekly | 59 | 158 |
| yearly | 4 | 11 |

## Tasks (full tier) by domain
| domain | locked | tuning |
|---|---|---|
| agriculture_nature | 5 | 25 |
| air_quality | 18 | 61 |
| economics_finance | 9 | 42 |
| energy | 127 | 286 |
| health | 3 | 29 |
| hydrology_ocean | 22 | 21 |
| industrial | 8 | 26 |
| mobility | 54 | 127 |
| other | 4 | 15 |
| retail_demand | 3 | 12 |
| society | 8 | 12 |
| space_science | 3 | 28 |
| weather_climate | 17 | 93 |
| web_cloud | 35 | 52 |

## Tasks (full tier) by horizon
| horizon_class | locked | tuning |
|---|---|---|
| long | 59 | 132 |
| medium | 59 | 143 |
| short | 198 | 554 |

## Tasks (full tier) by covariates
| covariates | locked | tuning |
|---|---|---|
| known | 11 | 12 |
| none | 238 | 622 |
| past | 67 | 195 |

## Datasets that failed in Milestone B
| dataset_id | error |
|---|---|
| bsebench_org_nasa_rw_2014_partitioned_by_cell_id | ValueError: no series left after the audit and cross-dataset drops |
| juliensimon_deep_space_probes | ValueError: no series left after the audit and cross-dataset drops |
| nju_zhangsq_rmisc_shellhackathon | ValueError: no series left after the audit and cross-dataset drops |
| nju_zhangsq_rmisc_swat | ValueError: no series left after the audit and cross-dataset drops |

## Audit code version per dataset (audit_code in audit/<dataset_id>.json)
57 a2cb824, 215 before a2cb824.
| audit_code | datasets | ids |
|---|---|---|
| a2cb824 | 57 | autogluon_chronos_datasets_monash_london_smart_meters, autogluon_chronos_datasets_monash_nn5_weekly_nn5, autogluon_chronos_datasets_taxi_30min_taxi_1h, autogluon_chronos_datasets_uber_tlc_hourly_uber_tlc_daily, autogluon_chronos_datasets_ushcn_daily, autogluon_chronos_datasets_weatherbench_daily, autogluon_chronos_datasets_weatherbench_weekly, autogluon_chronos_datasets_wiki_daily_100k, autogluon_chronos_datasets_wind_farms_hourly_wind_farms_daily, earino_austin_911_response, emulated_inc_sci_booster_gmps_regulation_forecast, leadlagforecasting_llf_github_default, salesforce_lotsa_data_alibaba_cluster_trace_2018, salesforce_lotsa_data_bdg_2_bear, salesforce_lotsa_data_bdg_2_fox, salesforce_lotsa_data_bdg_2_panther, salesforce_lotsa_data_bdg_2_rat, salesforce_lotsa_data_beijing_air_quality, salesforce_lotsa_data_borealis, salesforce_lotsa_data_borg_cluster_data_2011, salesforce_lotsa_data_cdc_fluview_ilinet, salesforce_lotsa_data_cdc_fluview_who_nrevss, salesforce_lotsa_data_covid_mobility, salesforce_lotsa_data_extended_web_traffic_with_missing, salesforce_lotsa_data_ideal, salesforce_lotsa_data_kaggle_web_traffic_weekly, salesforce_lotsa_data_largest_2021, salesforce_lotsa_data_lcl, salesforce_lotsa_data_london_smart_meters_with_missing, salesforce_lotsa_data_los_loop, salesforce_lotsa_data_m1_monthly, salesforce_lotsa_data_nn5_daily_with_missing, salesforce_lotsa_data_oikolab_weather, salesforce_lotsa_data_pedestrian_counts, salesforce_lotsa_data_pems03, salesforce_lotsa_data_pems04, salesforce_lotsa_data_pems07, salesforce_lotsa_data_pems08, salesforce_lotsa_data_project_tycho, salesforce_lotsa_data_q_traffic, salesforce_lotsa_data_residential_load_power, salesforce_lotsa_data_residential_pv_power, salesforce_lotsa_data_smart, salesforce_lotsa_data_subseasonal, salesforce_lotsa_data_subseasonal_precip, salesforce_lotsa_data_sunspot_with_missing, salesforce_lotsa_data_taxi_30min, salesforce_lotsa_data_traffic_hourly, salesforce_lotsa_data_uber_tlc_daily, salesforce_lotsa_data_uber_tlc_hourly, salesforce_lotsa_data_vehicle_trips_with_missing, salesforce_lotsa_data_weather, salesforce_lotsa_data_wiki_rolling_nips, salesforce_lotsa_data_wind_farms_with_missing, salesforce_lotsa_data_wind_power, valdovaldo1500_crime_brasil_rs_2022_2026, zalizedata_us_grid_hourly_power_plants_dataset |
| before a2cb824 | 215 | abmallick_heart_breath_sleep_stage_dataset, aduitya9_aduitya_dextra_aqi_india, ai_iot_energybench_dth, ai_iot_energybench_goiener, ai_iot_energybench_lcl, ai_iot_energybench_norwegian, ai_iot_energybench_prayas_ceew, ai_iot_energybench_sgsc, ai_iot_energybench_unicon, ajto_goldencheetah, akshaykarthick_india_weather_air_quality, alaminxpro_bd_food_prices, alanwgt_fipex_veiculos_brasil, alexroz_carbonfluxbench, alinurdin_time_series_benchmark_eia_hourly_madrid_15min_webtris_15min_usgs_daily_usgs_continuous_hydat_daily_bom_daily_c, angelosaadeh_velib, approjects_us_layoffs_monthly_time_series_warn_act, ashwingopalsamy_central_bank_policy_rates, autogluon_chronos_datasets_dominick, autogluon_chronos_datasets_mexico_city_bikes, autogluon_chronos_datasets_monash_cif_2016, autogluon_chronos_datasets_monash_pedestrian_counts, autogluon_chronos_datasets_monash_rideshare, autogluon_chronos_datasets_monash_traffic, autogluon_chronos_datasets_monash_weather, autonlab_timeseries_pile_forecasting, autonlab_timeseries_pile_forecasting_autoformer, battery_life_batterylife_raw_zn_ion_na_ion_calb, bayway_italian_open_timeseries, cashlessconsumer_chennai_rain_gauges, ceepr_bit_dianchi_water, cm3r_meteo_france_climatologie_quotidienne, cmu_ospo_data_for_extended_range_water_demand_forecasts_driven_by_subseasonal_to_seasonal, copernicus_argentina_estado_hidrico_argentina_2001_2026, copernicus_argentina_parana_alturas_historicas_1900_2026, cosuleabianca_eea_pm25_forecasting, datonic_spain_aemet_historical_weather, datonic_world_development_indicators_default, dhivakarnaidusunkari_mandi_india_daily_prices_2001_2026, digitalhen_us_airport_wait_times, dmariaa70_metraq_air_quality, donny_guo_airline_delay, duckermaster_thaiclim_district, electricsheepafrica_africa_wfp_food_prices_for_uganda, electricsheepafrica_wfp_nigeria_food_prices, elskow_weather4cast, emperor_mew_ooni_censorship_historical, empulse_ai_empulse_street_fx_index, emulated_inc_sci_airport_low_ceiling_visibility, emulated_inc_sci_lake_temperature_profiles_unmonitored, emulated_inc_sci_streamflow_ungauged_new_country, epi_eval_cdc_nrevss_rsv, epi_eval_cdc_nssp, epi_eval_global_mobility, epi_eval_nhsn_hrd, epi_eval_wikipedia_pageviews, feregrino_london_cycles, fidel_ts_california_iso, fidel_ts_canada_photovoltaics_plants, fidel_ts_nyc_traffic_speed, food_ai_nexus_global_food_safety_incidents, fuel_prices_eu_european_fuel_prices, fulldecent_septa_regionalrail_otp, furusu_japan_max_temperature, geoweaver_snowwaterequivalent_trainingdata, gft_ttm4hvac_source_all_train, gladcolor_hourly_population_us, globalsquiddatabase_tuna_fisheries_dataset, glucofmbench_glucofm_bench, hackermango_hyderabaduniversity_hvac_dataset, hhs_official_nndss_weekly_data, hhs_official_weekly_counts_of_deaths_from_all_causes_by_jurisdi, hhs_official_weekly_provisional_counts_of_deaths_by_state_and_s, hhs_official_weekly_rates_of_laboratory_confirmed_rsv_hospitali, hhs_official_weekly_united_states_covid_19_hospitalization_metr, hq_bench_quito_corpus_hour, hq_bench_quito_corpus_min, hq_bench_quitobench_hour_min, hq_bench_quitobench_hour_min_r577, hsilvosa_omie_electricity, hsilvosa_safecast_radiation, j_marcos_gs_brazil_solar_generation, jainam03_rescast_100k_ecobee, jainam03_rescast_100k_heapo_timeseries_data, jainam03_rescast_100k_ideal, jainam03_rescast_100k_nest, jainam03_rescast_100k_refit, jhumbl_orr_station_usage, jhumbl_police_crime_london, jhumbl_tfl_network_demand, jotone_arpa_osmer_weatherstations, jotone_pcfvg_weatherstations, juliensimon_auroral_electrojet_index, juliensimon_celestrak_space_weather, juliensimon_dst_index, juliensimon_f107_solar_flux, juliensimon_geomagnetic_kp_index, juliensimon_goes_xray_flux, juliensimon_iers_earth_orientation, juliensimon_nasa_maven_kp_insitu, juliensimon_omni_solar_wind_parameters, jwigginton_timeseries_1mn_sp500_default, khajlk_copenhagen_air_quality_hourly, marplan6_batgrad_pozzato_2022, matanzig_disney_theme_park_queue_dynamics, matrosovdani_moscow_parking_occupancy, meganariley_daily_aqi, meteoswiss_peakweather, mihnathul_celljar, misikoff_zillow_days_on_market, misikoff_zillow_for_sale_listings, misikoff_zillow_home_values, misikoff_zillow_rentals, misikoff_zillow_sales, modelpulse_model_pulse_data, nakas_powderbench_truth, narendersingh007_india_solar_benchmark_dataset, national_digital_act_air_quality, national_digital_au_births_by_age_of_mother, national_digital_nt_crocodile_captures, national_digital_nt_fuel_prices_history, nevirim_wikipedia_attention_13lang, nju_zhangsq_rmisc_amarketchina, nju_zhangsq_rmisc_applianceenergy, nju_zhangsq_rmisc_britaincoal, nju_zhangsq_rmisc_bts, nju_zhangsq_rmisc_buildingsbenchrealcsv, nju_zhangsq_rmisc_dominick, nju_zhangsq_rmisc_ebayserver, nju_zhangsq_rmisc_era5hourlysinglelevels, nju_zhangsq_rmisc_frothflotation, nju_zhangsq_rmisc_globalclimatechange, nju_zhangsq_rmisc_householdpower, nju_zhangsq_rmisc_hungarianchickenpoxcases, nju_zhangsq_rmisc_keplerlightcurves, nju_zhangsq_rmisc_metropt3, nju_zhangsq_rmisc_metrotraffic, nju_zhangsq_rmisc_opsd_household, nju_zhangsq_rmisc_opsd_when2heat, nju_zhangsq_rmisc_pm25fivecities, nju_zhangsq_rmisc_pump, nju_zhangsq_rmisc_roomoccupancy, nju_zhangsq_rmisc_servermachinedataset, nju_zhangsq_rmisc_smartmeteraus30m, nju_zhangsq_rmisc_smartmeteraus60m, nju_zhangsq_rmisc_smartmeteruk30m, nju_zhangsq_rmisc_smartmeteruk60m, nju_zhangsq_rmisc_sp500, nju_zhangsq_rmisc_stockmarketdata, nju_zhangsq_rmisc_tetuanpowerconsumption, nju_zhangsq_rmisc_timemmd, nju_zhangsq_rmisc_tushareetfdaily, nju_zhangsq_rmisc_tushareindexdaily, nju_zhangsq_rmisc_tusharestockdaily, nju_zhangsq_rmisc_tusharestockweekly, nju_zhangsq_rmisc_wadi, nju_zhangsq_rmisc_weeklyfuelpricesitaly, nju_zhangsq_rmisc_weeklyroadfuelprices, nju_zhangsq_rmisc_windelec, nju_zhangsq_rmisc_windpower4secs, nju_zhangsq_rmisc_xiamenairquality, notadib_usa_corn_belt_crop_yield, numix_stations_officielles_es, olaflaitinen_baku_metro_station_flow, open_predico_benchmark, openstef_liander2024_energy_forecasting_benchmark_load_measurements_weather_forecasts, opensynth_tudelft_electricity_consumption_1_0_default, orailix_ride_silver, pahzuzu_pah_env_001_serc_soil_signals, phanerozoic_qiskit_calibration_drift, phanerozoic_ww3_ndbc_pairs, powerzoojax_powerzoodataset_ausgrid_zone_substation, powerzoojax_powerzoodataset_gb_neso_demand_gb_gen_by_type_gb_market_mid_gb_forecast_actual_demand, pracadata_brazil_fuel_prices, prasad_gade05_india_upi_ecosystem_2018_2025, previdengue_predict_climate_data, project_agml_usda_nass_crop_statistics, prospex_ch_swiss_commercial_registry_monthly_panel, rebase_energy_heftcom2024_default, rebase_energy_nve_windpower_data_nve_windpower_timeseries_csv, rif_sql_uk_electricity_generation_mix_2019_2026, ron_the_code_telangana_time_series_2023_2025, roshangupta007_yieldformer_de_data, rricc22_endomondo_hr_prediction_v2, samuelm0422_hourly_electricity_demand_brazil_dataset, sayanroy058_weather_time_series_forecasting, shashkovich_telecommunication_sms_time_series, showlcw_china_cpi_ppi_pmi, skforecast_ashrae_daily, skywalker1910_skynet_aqi_panel, somemone_ac_transit_apc, sovai_wikipedia_views_default, steven0226_air_quality, surfe_diem_wave_archive_usa_southwest, thanna94_us_building_permits, tharinduperera_weather_clustering_data, thermobench_thermobench_pure, todopueblos_poblacion_municipios_espana, wikimedia_community_english_wikipedia_pageviews_by_second, witgaw_metr_la, witgaw_pems_bay, xairon_piezo_embedding_benchmark, xus_spain_fuel_prices, yahya2491_3w_pipeline, youngbongbong_30_od_subway_default, yuiseki_ilostat, yuiseki_jma_rain_kanto, yuiseki_meta_range_maps_2022_05, yuiseki_pwt, yuiseki_un_wpp_wpp2024, yuiseki_undata_sdmx, yuiseki_undp_hdr, yzzpz_charged_data_10_csv_per_city, zait_ai_rusweather_434, zer0contextlost_waterwatch_runoff |

## Cross-dataset copies dropped from tuning (z-normalised window hashes, > 10% of windows)
Series dropped: 168.
| dataset_id | matched | series |
|---|---|---|
| ajto_goldencheetah | autogluon_chronos_datasets_wind_farms_hourly_wind_farms_daily | 1 |
| autogluon_chronos_datasets_uber_tlc_hourly_uber_tlc_daily | salesforce_lotsa_data_uber_tlc_hourly | 163 |
| cosuleabianca_eea_pm25_forecasting | dmariaa70_metraq_air_quality | 3 |
| datonic_world_development_indicators_default | yuiseki_un_wpp_wpp2024 | 1 |

## Series per frequency variant and horizon (drops counted per horizon; aggregation rule per column, for review)
| config | horizon | agg | series | too_short | nan_in_eval | zero_scale | skipped |
|---|---|---|---|---|---|---|---|
| abmallick_heart_breath_sleep_stage_dataset__30s | short (H=48) | native | 37 | 0 | 963 | 0 |  |
| abmallick_heart_breath_sleep_stage_dataset__30s | medium (H=480) | native | 0 | 977 | 0 | 0 | 23 of 1000 series long enough (< 75%) |
| abmallick_heart_breath_sleep_stage_dataset__30s | long (H=720) | native | 0 | 1000 | 0 | 0 | 0 of 1000 series long enough (< 75%) |
| aduitya9_aduitya_dextra_aqi_india__d | short (H=30) | native | 15 | 0 | 0 | 0 |  |
| aduitya9_aduitya_dextra_aqi_india__wsun | short (H=8) | y:mean, horizon:mean, persistence:mean, plain_seed42:mean, plain_seed43:mean, plain_seed44:mean, oversampled_seed42:mean, oversampled_seed43:mean, oversampled_seed44:mean, weighted_seed42:mean, weighted_seed43:mean, weighted_seed44:mean, quantile_seed42:mean, quantile_seed43:mean, quantile_seed44:mean, gbm:mean, chronos-bolt median:mean, chronos-bolt q90:mean, chronos-2 median:mean, chronos-2 q90:mean, chronos-2-weather median:mean, chronos-2-weather q90:mean, evening_seed42:mean, evening_seed43:mean, evening_seed44:mean, gbm evening:mean, chronos-2-evening median:mean, chronos-2-evening q90:mean | 15 | 0 | 0 | 0 |  |
| afhubbard_gpu_prices__12h | short (H=48) | native | 0 | 0 | 240 | 0 | no series left after the NaN / scale rules |
| afhubbard_gpu_prices__12h | medium (H=480) | native | 0 | 240 | 0 | 0 | 0 of 240 series long enough (< 75%) |
| afhubbard_gpu_prices__12h | long (H=720) | native | 0 | 240 | 0 | 0 | 0 of 240 series long enough (< 75%) |
| afhubbard_gpu_prices__d | short (H=30) | gpu_count:mean, gpu_memory_gb:mean, vcpus:mean, ram_gb:mean, price_per_hour:last, is_spot:mean, available:last | 0 | 12 | 228 | 0 | no series left after the NaN / scale rules |
| ai_iot_energybench_dth__15min | short (H=48) | native | 0 | 0 | 770 | 0 | no series left after the NaN / scale rules |
| ai_iot_energybench_dth__15min | medium (H=480) | native | 0 | 0 | 770 | 0 | no series left after the NaN / scale rules |
| ai_iot_energybench_dth__15min | long (H=720) | native | 0 | 0 | 770 | 0 | no series left after the NaN / scale rules |
| ai_iot_energybench_dth__h | short (H=48) | target:mean | 769 | 0 | 1 | 0 |  |
| ai_iot_energybench_dth__h | medium (H=480) | target:mean | 769 | 0 | 1 | 0 |  |
| ai_iot_energybench_dth__h | long (H=720) | target:mean | 769 | 0 | 1 | 0 |  |
| ai_iot_energybench_dth__d | short (H=30) | target:mean | 769 | 0 | 1 | 0 |  |
| ai_iot_energybench_goiener__h | short (H=48) | native | 0 | 0 | 492 | 0 | no series left after the NaN / scale rules |
| ai_iot_energybench_goiener__h | medium (H=480) | native | 0 | 0 | 492 | 0 | no series left after the NaN / scale rules |
| ai_iot_energybench_goiener__h | long (H=720) | native | 0 | 0 | 492 | 0 | no series left after the NaN / scale rules |
| ai_iot_energybench_goiener__d | short (H=30) | target:mean | 480 | 0 | 12 | 0 |  |
| ai_iot_energybench_goiener__wsun | short (H=8) | target:mean | 491 | 0 | 1 | 0 |  |
| ai_iot_energybench_lcl__30min | short (H=48) | native | 836 | 0 | 164 | 0 |  |
| ai_iot_energybench_lcl__30min__medium | medium (H=480) | native | 552 | 0 | 448 | 0 |  |
| ai_iot_energybench_lcl__30min__long | long (H=720) | native | 576 | 0 | 424 | 0 |  |
| ai_iot_energybench_lcl__h | short (H=48) | target:mean | 736 | 0 | 264 | 0 |  |
| ai_iot_energybench_lcl__h__medium | medium (H=480) | target:mean | 655 | 0 | 345 | 0 |  |
| ai_iot_energybench_lcl__h__medium | long (H=720) | target:mean | 655 | 0 | 345 | 0 |  |
| ai_iot_energybench_lcl__d | short (H=30) | target:mean | 879 | 0 | 121 | 0 |  |
| ai_iot_energybench_lcl__wsun | short (H=8) | target:mean | 985 | 3 | 12 | 0 |  |
| ai_iot_energybench_norwegian__h | short (H=48) | native | 1000 | 0 | 0 | 0 |  |
| ai_iot_energybench_norwegian__h | medium (H=480) | native | 1000 | 0 | 0 | 0 |  |
| ai_iot_energybench_norwegian__h | long (H=720) | native | 1000 | 0 | 0 | 0 |  |
| ai_iot_energybench_norwegian__d | short (H=30) | target:mean | 1000 | 0 | 0 | 0 |  |
| ai_iot_energybench_norwegian__wsun | short (H=8) | target:mean | 1000 | 0 | 0 | 0 |  |
| ai_iot_energybench_prayas_ceew__15min | short (H=48) | native | 100 | 0 | 9 | 0 |  |
| ai_iot_energybench_prayas_ceew__15min | medium (H=480) | native | 100 | 0 | 9 | 0 |  |
| ai_iot_energybench_prayas_ceew__15min | long (H=720) | native | 100 | 0 | 9 | 0 |  |
| ai_iot_energybench_prayas_ceew__h | short (H=48) | target:mean | 107 | 0 | 2 | 0 |  |
| ai_iot_energybench_prayas_ceew__h | medium (H=480) | target:mean | 107 | 0 | 2 | 0 |  |
| ai_iot_energybench_prayas_ceew__h | long (H=720) | target:mean | 107 | 0 | 2 | 0 |  |
| ai_iot_energybench_prayas_ceew__d | short (H=30) | target:mean | 109 | 0 | 0 | 0 |  |
| ai_iot_energybench_prayas_ceew__wsun | short (H=8) | target:mean | 67 | 42 | 0 | 0 |  |
| ai_iot_energybench_sgsc__30min | short (H=48) | native | 5 | 0 | 995 | 0 |  |
| ai_iot_energybench_sgsc__30min | medium (H=480) | native | 5 | 0 | 995 | 0 |  |
| ai_iot_energybench_sgsc__30min | long (H=720) | native | 5 | 0 | 995 | 0 |  |
| ai_iot_energybench_sgsc__h | short (H=48) | target:mean | 889 | 0 | 111 | 0 |  |
| ai_iot_energybench_sgsc__h | medium (H=480) | target:mean | 889 | 0 | 111 | 0 |  |
| ai_iot_energybench_sgsc__h__long | long (H=720) | target:mean | 895 | 0 | 105 | 0 |  |
| ai_iot_energybench_sgsc__d | short (H=30) | target:mean | 982 | 0 | 18 | 0 |  |
| ai_iot_energybench_sgsc__wsun | short (H=8) | target:mean | 987 | 0 | 13 | 0 |  |
| ai_iot_energybench_unicon__15min | short (H=48) | native | 14 | 0 | 40 | 0 |  |
| ai_iot_energybench_unicon__15min | medium (H=480) | native | 0 | 0 | 54 | 0 | no series left after the NaN / scale rules |
| ai_iot_energybench_unicon__15min__long | long (H=720) | native | 9 | 0 | 45 | 0 |  |
| ai_iot_energybench_unicon__h | short (H=48) | target:mean | 9 | 0 | 45 | 0 |  |
| ai_iot_energybench_unicon__h | medium (H=480) | target:mean | 9 | 0 | 45 | 0 |  |
| ai_iot_energybench_unicon__h__long | long (H=720) | target:mean | 11 | 0 | 43 | 0 |  |
| ai_iot_energybench_unicon__d | short (H=30) | target:mean | 44 | 0 | 10 | 0 |  |
| ai_iot_energybench_unicon__wsun | short (H=8) | target:mean | 0 | 4 | 50 | 0 | no series left after the NaN / scale rules |
| aikstockdata_korea_equity_daily__d | short (H=30) | native | 0 | 0 | 1000 | 0 | no series left after the NaN / scale rules |
| aikstockdata_korea_equity_daily__wsun | short (H=8) | close_krw:mean, volume:mean | 0 | 8 | 992 | 0 | no series left after the NaN / scale rules |
| ajto_goldencheetah__s | short (H=48) | native | 858 | 0 | 103 | 0 |  |
| ajto_goldencheetah__s__medium | medium (H=480) | native | 741 | 81 | 139 | 0 |  |
| ajto_goldencheetah__s__long | long (H=720) | native | 594 | 214 | 153 | 0 |  |
| akshaykarthick_india_weather_air_quality__d | short (H=30) | native | 50 | 0 | 0 | 0 |  |
| akshaykarthick_india_weather_air_quality__wsun | short (H=8) | pm2_5_mean:mean, pm2_5_max:mean, pm10_mean:mean, pm10_max:mean, no2_mean:mean, so2_mean:mean, o3_8h_max:mean, co_8h_max:mean, dust_mean:mean, aod_mean:mean, us_aqi_max:mean, hours_available:last, india_aqi:mean, india_aqi_pm:mean | 50 | 0 | 0 | 0 |  |
| alaminxpro_bd_food_prices__ms | short (H=12) | native | 110 | 0 | 0 | 0 |  |
| alanwgt_fipex_veiculos_brasil__ms | short (H=12) | native | 999 | 0 | 1 | 0 |  |
| alexroz_carbonfluxbench__d | short (H=30) | native | 493 | 0 | 0 | 0 |  |
| alexroz_carbonfluxbench__wsun | short (H=8) | dewpoint_temperature_2m:mean, temperature_2m:mean, skin_temperature:mean, soil_temperature_level_1:mean, soil_temperature_level_2:mean, soil_temperature_level_3:mean, soil_temperature_level_4:mean, lake_bottom_temperature:mean, lake_ice_depth:mean, lake_ice_temperature:mean, lake_mix_layer_depth:mean, lake_mix_layer_temperature:mean, lake_shape_factor:mean, lake_total_layer_temperature:mean, snow_albedo:mean, snow_cover:mean, snow_density:mean, snow_depth:mean, snow_depth_water_equivalent:mean, snowfall_sum:mean, snowmelt_sum:mean, temperature_of_snow_layer:mean, skin_reservoir_content:mean, volumetric_soil_water_layer_1:mean, volumetric_soil_water_layer_2:mean, volumetric_soil_water_layer_3:mean, volumetric_soil_water_layer_4:mean, forecast_albedo:mean, surface_latent_heat_flux_sum:mean, surface_net_solar_radiation_sum:mean, surface_net_thermal_radiation_sum:mean, surface_sensible_heat_flux_sum:mean, surface_solar_radiation_downwards_sum:mean | 493 | 0 | 0 | 0 |  |
| alinurdin_time_series_benchmark_eia_hourly_madrid_15min_webtris_15min_usgs_daily_usgs_continuous_hydat_daily_bom_daily_c__h | short (H=48) | native | 48 | 0 | 0 | 0 |  |
| alinurdin_time_series_benchmark_eia_hourly_madrid_15min_webtris_15min_usgs_daily_usgs_continuous_hydat_daily_bom_daily_c__h__medium | medium (H=480) | native | 42 | 0 | 6 | 0 |  |
| alinurdin_time_series_benchmark_eia_hourly_madrid_15min_webtris_15min_usgs_daily_usgs_continuous_hydat_daily_bom_daily_c__h__medium | long (H=720) | native | 42 | 0 | 6 | 0 |  |
| alinurdin_time_series_benchmark_eia_hourly_madrid_15min_webtris_15min_usgs_daily_usgs_continuous_hydat_daily_bom_daily_c__d | short (H=30) | demand:mean, demand_forecast:mean | 47 | 0 | 1 | 0 |  |
| alinurdin_time_series_benchmark_eia_hourly_madrid_15min_webtris_15min_usgs_daily_usgs_continuous_hydat_daily_bom_daily_c__wsun | short (H=8) | demand:mean, demand_forecast:mean | 48 | 0 | 0 | 0 |  |
| alwaysbyx_bear_data__5min | short (H=48) | native | 0 | 0 | 129 | 15 | no series left after the NaN / scale rules |
| alwaysbyx_bear_data__5min | medium (H=480) | native | 0 | 15 | 129 | 0 | no series left after the NaN / scale rules |
| alwaysbyx_bear_data__5min | long (H=720) | native | 0 | 15 | 129 | 0 | no series left after the NaN / scale rules |
| alwaysbyx_bear_data__h | short (H=48) | target:mean | 0 | 15 | 129 | 0 | no series left after the NaN / scale rules |
| alwaysbyx_bear_data__h | medium (H=480) | target:mean | 0 | 129 | 0 | 0 | 0 of 129 series long enough (< 75%) |
| alwaysbyx_bear_data__h | long (H=720) | target:mean | 0 | 129 | 0 | 0 | 0 of 129 series long enough (< 75%) |
| angelosaadeh_velib__61s | short (H=48) | native | 0 | 0 | 1000 | 0 | no series left after the NaN / scale rules |
| angelosaadeh_velib__61s | medium (H=480) | native | 0 | 0 | 1000 | 0 | no series left after the NaN / scale rules |
| angelosaadeh_velib__61s | long (H=720) | native | 0 | 0 | 1000 | 0 | no series left after the NaN / scale rules |
| angelosaadeh_velib__h | short (H=48) | bikes:mean, docks:mean, mechanical:mean, ebike:mean, last_reported:mean, is_installed:last, is_renting:last, is_returning:last | 1000 | 0 | 0 | 0 |  |
| angelosaadeh_velib__h | medium (H=480) | bikes:mean, docks:mean, mechanical:mean, ebike:mean, last_reported:mean, is_installed:last, is_renting:last, is_returning:last | 0 | 1000 | 0 | 0 | 0 of 1000 series long enough (< 75%) |
| angelosaadeh_velib__h | long (H=720) | bikes:mean, docks:mean, mechanical:mean, ebike:mean, last_reported:mean, is_installed:last, is_renting:last, is_returning:last | 0 | 1000 | 0 | 0 | 0 of 1000 series long enough (< 75%) |
| approjects_us_layoffs_monthly_time_series_warn_act__ms | short (H=12) | native | 20 | 0 | 14 | 0 |  |
| arailym_tleubayeva_nk_oil_well_sensor_monitoring_v2__h | short (H=48) | native | 0 | 0 | 10 | 0 | no series left after the NaN / scale rules |
| arailym_tleubayeva_nk_oil_well_sensor_monitoring_v2__h | medium (H=480) | native | 0 | 0 | 10 | 0 | no series left after the NaN / scale rules |
| arailym_tleubayeva_nk_oil_well_sensor_monitoring_v2__h | long (H=720) | native | 0 | 0 | 10 | 0 | no series left after the NaN / scale rules |
| arailym_tleubayeva_nk_oil_well_sensor_monitoring_v2__d | short (H=30) | value:mean, n_obs:mean | 0 | 0 | 10 | 0 | no series left after the NaN / scale rules |
| arailym_tleubayeva_nk_oil_well_sensor_monitoring_v2_train_test__h | short (H=48) | native | 0 | 0 | 10 | 0 | no series left after the NaN / scale rules |
| arailym_tleubayeva_nk_oil_well_sensor_monitoring_v2_train_test__h | medium (H=480) | native | 0 | 0 | 10 | 0 | no series left after the NaN / scale rules |
| arailym_tleubayeva_nk_oil_well_sensor_monitoring_v2_train_test__h | long (H=720) | native | 0 | 0 | 10 | 0 | no series left after the NaN / scale rules |
| arailym_tleubayeva_nk_oil_well_sensor_monitoring_v2_train_test__d | short (H=30) | value:mean, n_obs:mean | 0 | 0 | 10 | 0 | no series left after the NaN / scale rules |
| arimancy_bts_flights_weather__min | short (H=48) | native | 0 | 0 | 2 | 0 | no series left after the NaN / scale rules |
| arimancy_bts_flights_weather__min | medium (H=480) | native | 0 | 0 | 2 | 0 | no series left after the NaN / scale rules |
| arimancy_bts_flights_weather__min | long (H=720) | native | 0 | 0 | 2 | 0 | no series left after the NaN / scale rules |
| arimancy_bts_flights_weather__h | short (H=48) | dep_delay_min:mean, arr_delay_min:mean, taxi_out_min:mean, taxi_in_min:mean, crs_elapsed_min:mean, actual_elapsed_min:mean, air_time_min:mean, distance_miles:mean, reported_carrier_delay_min:mean, reported_weather_delay_min:mean, reported_nas_delay_min:mean, reported_security_delay_min:mean, reported_late_aircraft_delay_min:mean, day_of_week:last, day_of_year:last, days_to_nearest_holiday:last, origin_sched_departures_this_hour:last, dest_sched_arrivals_this_hour:last, origin_sched_departures_this_day:last, prior_leg_arr_delay_min:mean, prior_leg_turnaround_min:mean, wx_origin_temp_c:mean, wx_origin_dewpoint_c:mean, wx_origin_relative_humidity_pct:mean, wx_origin_wind_speed_ms:mean, wx_origin_wind_gust_ms:mean, wx_origin_wind_dir_deg:mean, wx_origin_visibility_m:mean, wx_origin_ceiling_m:mean, wx_origin_sea_level_pressure_hpa:mean, wx_origin_precip_1h_mm:mean, wx_origin_snow_depth_cm:mean, wx_origin_sky_cover_oktas:mean | 0 | 0 | 2 | 0 | no series left after the NaN / scale rules |
| arimancy_bts_flights_weather__h | medium (H=480) | dep_delay_min:mean, arr_delay_min:mean, taxi_out_min:mean, taxi_in_min:mean, crs_elapsed_min:mean, actual_elapsed_min:mean, air_time_min:mean, distance_miles:mean, reported_carrier_delay_min:mean, reported_weather_delay_min:mean, reported_nas_delay_min:mean, reported_security_delay_min:mean, reported_late_aircraft_delay_min:mean, day_of_week:last, day_of_year:last, days_to_nearest_holiday:last, origin_sched_departures_this_hour:last, dest_sched_arrivals_this_hour:last, origin_sched_departures_this_day:last, prior_leg_arr_delay_min:mean, prior_leg_turnaround_min:mean, wx_origin_temp_c:mean, wx_origin_dewpoint_c:mean, wx_origin_relative_humidity_pct:mean, wx_origin_wind_speed_ms:mean, wx_origin_wind_gust_ms:mean, wx_origin_wind_dir_deg:mean, wx_origin_visibility_m:mean, wx_origin_ceiling_m:mean, wx_origin_sea_level_pressure_hpa:mean, wx_origin_precip_1h_mm:mean, wx_origin_snow_depth_cm:mean, wx_origin_sky_cover_oktas:mean | 0 | 2 | 0 | 0 | 0 of 2 series long enough (< 75%) |
| arimancy_bts_flights_weather__h | long (H=720) | dep_delay_min:mean, arr_delay_min:mean, taxi_out_min:mean, taxi_in_min:mean, crs_elapsed_min:mean, actual_elapsed_min:mean, air_time_min:mean, distance_miles:mean, reported_carrier_delay_min:mean, reported_weather_delay_min:mean, reported_nas_delay_min:mean, reported_security_delay_min:mean, reported_late_aircraft_delay_min:mean, day_of_week:last, day_of_year:last, days_to_nearest_holiday:last, origin_sched_departures_this_hour:last, dest_sched_arrivals_this_hour:last, origin_sched_departures_this_day:last, prior_leg_arr_delay_min:mean, prior_leg_turnaround_min:mean, wx_origin_temp_c:mean, wx_origin_dewpoint_c:mean, wx_origin_relative_humidity_pct:mean, wx_origin_wind_speed_ms:mean, wx_origin_wind_gust_ms:mean, wx_origin_wind_dir_deg:mean, wx_origin_visibility_m:mean, wx_origin_ceiling_m:mean, wx_origin_sea_level_pressure_hpa:mean, wx_origin_precip_1h_mm:mean, wx_origin_snow_depth_cm:mean, wx_origin_sky_cover_oktas:mean | 0 | 2 | 0 | 0 | 0 of 2 series long enough (< 75%) |
| ashwingopalsamy_central_bank_policy_rates__d | short (H=30) | native | 7 | 0 | 0 | 0 |  |
| ashwingopalsamy_central_bank_policy_rates__wsun | short (H=8) | rate_pct:mean, rate_bps:mean, rate_low_pct:mean, rate_high_pct:mean, is_change:last | 7 | 0 | 0 | 0 |  |
| autogluon_chronos_datasets_dominick__wthu | short (H=8) | native | 1000 | 0 | 0 | 0 |  |
| autogluon_chronos_datasets_exchange_rate__d | short (H=30) | native | 0 | 0 | 8 | 0 | no series left after the NaN / scale rules |
| autogluon_chronos_datasets_exchange_rate__wsun | short (H=8) | target:mean | 0 | 0 | 8 | 0 | no series left after the NaN / scale rules |
| autogluon_chronos_datasets_mexico_city_bikes__h | short (H=48) | native | 493 | 0 | 0 | 1 |  |
| autogluon_chronos_datasets_mexico_city_bikes__h__medium | medium (H=480) | native | 484 | 4 | 0 | 6 |  |
| autogluon_chronos_datasets_mexico_city_bikes__h__medium | long (H=720) | native | 484 | 7 | 0 | 3 |  |
| autogluon_chronos_datasets_mexico_city_bikes__d | short (H=30) | target:mean | 483 | 10 | 0 | 1 |  |
| autogluon_chronos_datasets_mexico_city_bikes__wsun | short (H=8) | target:mean | 483 | 11 | 0 | 0 |  |
| autogluon_chronos_datasets_monash_cif_2016__me | short (H=12) | native | 62 | 0 | 0 | 0 |  |
| autogluon_chronos_datasets_monash_london_smart_meters__30min | short (H=48) | native | 1000 | 0 | 0 | 0 |  |
| autogluon_chronos_datasets_monash_london_smart_meters__30min__medium | medium (H=480) | native | 997 | 0 | 3 | 0 |  |
| autogluon_chronos_datasets_monash_london_smart_meters__30min__medium | long (H=720) | native | 997 | 0 | 3 | 0 |  |
| autogluon_chronos_datasets_monash_london_smart_meters__h | short (H=48) | target:mean | 997 | 0 | 3 | 0 |  |
| autogluon_chronos_datasets_monash_london_smart_meters__h | medium (H=480) | target:mean | 997 | 0 | 3 | 0 |  |
| autogluon_chronos_datasets_monash_london_smart_meters__h | long (H=720) | target:mean | 997 | 0 | 3 | 0 |  |
| autogluon_chronos_datasets_monash_london_smart_meters__d | short (H=30) | target:mean | 999 | 1 | 0 | 0 |  |
| autogluon_chronos_datasets_monash_london_smart_meters__wsun | short (H=8) | target:mean | 960 | 40 | 0 | 0 |  |
| autogluon_chronos_datasets_monash_nn5_weekly_nn5__d | short (H=30) | native | 99 | 0 | 9 | 0 |  |
| autogluon_chronos_datasets_monash_nn5_weekly_nn5__wsun | short (H=8) | target:mean | 107 | 0 | 1 | 0 |  |
| autogluon_chronos_datasets_monash_pedestrian_counts__h | short (H=48) | native | 65 | 0 | 0 | 1 |  |
| autogluon_chronos_datasets_monash_pedestrian_counts__h__medium | medium (H=480) | native | 63 | 2 | 0 | 1 |  |
| autogluon_chronos_datasets_monash_pedestrian_counts__h__long | long (H=720) | native | 62 | 4 | 0 | 0 |  |
| autogluon_chronos_datasets_monash_pedestrian_counts__d | short (H=30) | target:mean | 61 | 5 | 0 | 0 |  |
| autogluon_chronos_datasets_monash_pedestrian_counts__wsun | short (H=8) | target:mean | 58 | 8 | 0 | 0 |  |
| autogluon_chronos_datasets_monash_rideshare__h | short (H=48) | native | 143 | 0 | 0 | 0 |  |
| autogluon_chronos_datasets_monash_rideshare__h | medium (H=480) | native | 0 | 143 | 0 | 0 | 0 of 143 series long enough (< 75%) |
| autogluon_chronos_datasets_monash_rideshare__h | long (H=720) | native | 0 | 143 | 0 | 0 | 0 of 143 series long enough (< 75%) |
| autogluon_chronos_datasets_monash_traffic__h | short (H=48) | native | 862 | 0 | 0 | 0 |  |
| autogluon_chronos_datasets_monash_traffic__h | medium (H=480) | native | 862 | 0 | 0 | 0 |  |
| autogluon_chronos_datasets_monash_traffic__h | long (H=720) | native | 862 | 0 | 0 | 0 |  |
| autogluon_chronos_datasets_monash_traffic__d | short (H=30) | target:mean | 862 | 0 | 0 | 0 |  |
| autogluon_chronos_datasets_monash_traffic__wsun | short (H=8) | target:mean | 862 | 0 | 0 | 0 |  |
| autogluon_chronos_datasets_monash_weather__d | short (H=30) | native | 1000 | 0 | 0 | 0 |  |
| autogluon_chronos_datasets_monash_weather__wsun | short (H=8) | target:mean | 1000 | 0 | 0 | 0 |  |
| autogluon_chronos_datasets_taxi_30min_taxi_1h__30min | short (H=48) | native | 508 | 0 | 492 | 0 |  |
| autogluon_chronos_datasets_taxi_30min_taxi_1h__30min | medium (H=480) | native | 508 | 0 | 492 | 0 |  |
| autogluon_chronos_datasets_taxi_30min_taxi_1h__30min | long (H=720) | native | 0 | 1000 | 0 | 0 | 0 of 1000 series long enough (< 75%) |
| autogluon_chronos_datasets_taxi_30min_taxi_1h__h | short (H=48) | target:mean | 508 | 0 | 492 | 0 |  |
| autogluon_chronos_datasets_taxi_30min_taxi_1h__h | medium (H=480) | target:mean | 0 | 1000 | 0 | 0 | 0 of 1000 series long enough (< 75%) |
| autogluon_chronos_datasets_taxi_30min_taxi_1h__h | long (H=720) | target:mean | 0 | 1000 | 0 | 0 | 0 of 1000 series long enough (< 75%) |
| autogluon_chronos_datasets_uber_tlc_hourly_uber_tlc_daily__h | short (H=48) | native | 99 | 0 | 0 | 0 |  |
| autogluon_chronos_datasets_uber_tlc_hourly_uber_tlc_daily__h | medium (H=480) | native | 99 | 0 | 0 | 0 |  |
| autogluon_chronos_datasets_uber_tlc_hourly_uber_tlc_daily__h | long (H=720) | native | 99 | 0 | 0 | 0 |  |
| autogluon_chronos_datasets_uber_tlc_hourly_uber_tlc_daily__d | short (H=30) | target:mean | 99 | 0 | 0 | 0 |  |
| autogluon_chronos_datasets_ushcn_daily__d | short (H=30) | native | 305 | 0 | 695 | 0 |  |
| autogluon_chronos_datasets_ushcn_daily__wsun | short (H=8) | target:mean, SNOW:mean, SNWD:mean, TMAX:mean, TMIN:mean | 338 | 0 | 662 | 0 |  |
| autogluon_chronos_datasets_weatherbench_daily__d | short (H=30) | native | 993 | 0 | 0 | 0 |  |
| autogluon_chronos_datasets_weatherbench_daily__wsun | short (H=8) | target:mean | 993 | 0 | 0 | 0 |  |
| autogluon_chronos_datasets_weatherbench_weekly__wsun | short (H=8) | native | 1000 | 0 | 0 | 0 |  |
| autogluon_chronos_datasets_wiki_daily_100k__d | short (H=30) | native | 1000 | 0 | 0 | 0 |  |
| autogluon_chronos_datasets_wiki_daily_100k__wsun | short (H=8) | target:mean | 1000 | 0 | 0 | 0 |  |
| autogluon_chronos_datasets_wind_farms_hourly_wind_farms_daily__h | short (H=48) | native | 326 | 0 | 0 | 1 |  |
| autogluon_chronos_datasets_wind_farms_hourly_wind_farms_daily__h | medium (H=480) | native | 326 | 0 | 0 | 1 |  |
| autogluon_chronos_datasets_wind_farms_hourly_wind_farms_daily__h__long | long (H=720) | native | 324 | 2 | 0 | 1 |  |
| autogluon_chronos_datasets_wind_farms_hourly_wind_farms_daily__d | short (H=30) | target:mean | 317 | 10 | 0 | 0 |  |
| autonlab_timeseries_pile_forecasting__h | short (H=48) | native | 862 | 0 | 0 | 0 |  |
| autonlab_timeseries_pile_forecasting__h | medium (H=480) | native | 862 | 0 | 0 | 0 |  |
| autonlab_timeseries_pile_forecasting__h | long (H=720) | native | 862 | 0 | 0 | 0 |  |
| autonlab_timeseries_pile_forecasting__d | short (H=30) | target:mean | 862 | 0 | 0 | 0 |  |
| autonlab_timeseries_pile_forecasting__wsun | short (H=8) | target:mean | 862 | 0 | 0 | 0 |  |
| autonlab_timeseries_pile_forecasting_autoformer__h | short (H=48) | native | 862 | 0 | 0 | 0 |  |
| autonlab_timeseries_pile_forecasting_autoformer__h | medium (H=480) | native | 862 | 0 | 0 | 0 |  |
| autonlab_timeseries_pile_forecasting_autoformer__h | long (H=720) | native | 862 | 0 | 0 | 0 |  |
| autonlab_timeseries_pile_forecasting_autoformer__d | short (H=30) | target:mean | 862 | 0 | 0 | 0 |  |
| autonlab_timeseries_pile_forecasting_autoformer__wsun | short (H=8) | target:mean | 862 | 0 | 0 | 0 |  |
| ayushkhaire_all_stock_market_data_daily_updates__d | short (H=30) | native | 0 | 0 | 1000 | 0 | no series left after the NaN / scale rules |
| ayushkhaire_all_stock_market_data_daily_updates__wsun | short (H=8) | target:mean | 0 | 46 | 954 | 0 | no series left after the NaN / scale rules |
| battery_life_batterylife_raw_zn_ion_na_ion_calb__s | short (H=48) | native | 17 | 0 | 0 | 0 |  |
| battery_life_batterylife_raw_zn_ion_na_ion_calb__s | medium (H=480) | native | 17 | 0 | 0 | 0 |  |
| battery_life_batterylife_raw_zn_ion_na_ion_calb__s | long (H=720) | native | 17 | 0 | 0 | 0 |  |
| bayway_italian_open_timeseries__h | short (H=48) | native | 8 | 0 | 10 | 0 |  |
| bayway_italian_open_timeseries__h__medium | medium (H=480) | native | 4 | 0 | 14 | 0 |  |
| bayway_italian_open_timeseries__h__medium | long (H=720) | native | 4 | 0 | 14 | 0 |  |
| bayway_italian_open_timeseries__d | short (H=30) | target:mean | 7 | 0 | 11 | 0 |  |
| bayway_italian_open_timeseries__wsun | short (H=8) | target:mean | 11 | 0 | 7 | 0 |  |
| brandonhxrr_mexico_city_weather__d | short (H=30) | native | 0 | 0 | 3 | 0 | no series left after the NaN / scale rules |
| brandonhxrr_mexico_city_weather__wsun | short (H=8) | target:mean | 0 | 0 | 3 | 0 | no series left after the NaN / scale rules |
| carlossilva1_us500_ticks_default__s | short (H=48) | native | 0 | 0 | 123 | 0 | no series left after the NaN / scale rules |
| carlossilva1_us500_ticks_default__s | medium (H=480) | native | 0 | 0 | 123 | 0 | no series left after the NaN / scale rules |
| carlossilva1_us500_ticks_default__s | long (H=720) | native | 0 | 0 | 123 | 0 | no series left after the NaN / scale rules |
| cashlessconsumer_chennai_rain_gauges__d | short (H=30) | native | 30 | 0 | 0 | 0 |  |
| cashlessconsumer_chennai_rain_gauges__wsun | short (H=8) | rainfall:mean, rainfall_mm:mean, rainfall_prev_mm:mean, rainfall_24h_mm:mean, progressive_mm:mean, previous_rainfall:last, rainfall24:mean, progressive:mean, progressflag:last, zone_no:mean | 30 | 0 | 0 | 0 |  |
| ceepr_bit_dianchi_water__4h | short (H=48) | native | 0 | 0 | 21 | 0 | no series left after the NaN / scale rules |
| ceepr_bit_dianchi_water__4h | medium (H=480) | native | 0 | 0 | 21 | 0 | no series left after the NaN / scale rules |
| ceepr_bit_dianchi_water__4h | long (H=720) | native | 0 | 0 | 21 | 0 | no series left after the NaN / scale rules |
| ceepr_bit_dianchi_water__d | short (H=30) | TEM:mean, PH:mean, DO:mean, CON:mean, NTU:mean, IMN:mean, NH_N:mean, TP:mean, TN:mean | 0 | 0 | 21 | 0 | no series left after the NaN / scale rules |
| ceepr_bit_dianchi_water__wsun | short (H=8) | TEM:mean, PH:mean, DO:mean, CON:mean, NTU:mean, IMN:mean, NH_N:mean, TP:mean, TN:mean | 8 | 1 | 12 | 0 |  |
| cm3r_meteo_france_climatologie_quotidienne__d | short (H=30) | native | 887 | 0 | 106 | 7 |  |
| cm3r_meteo_france_climatologie_quotidienne__wsun | short (H=8) | target:mean | 955 | 0 | 35 | 10 |  |
| cmu_ospo_data_for_extended_range_water_demand_forecasts_driven_by_subseasonal_to_seasonal__ms | short (H=12) | native | 1 | 0 | 0 | 0 |  |
| commoncrawl_statistics__28d | short (H=12) | native | 0 | 0 | 2 | 0 | no series left after the NaN / scale rules |
| copernicus_argentina_estado_hidrico_argentina_2001_2026__ms | short (H=12) | native | 527 | 0 | 0 | 0 |  |
| copernicus_argentina_lagunas_pampeanas_2018_2026__5d | short (H=30) | native | 0 | 0 | 20 | 0 | no series left after the NaN / scale rules |
| copernicus_argentina_parana_alturas_historicas_1900_2026__d | short (H=30) | native | 18 | 0 | 4 | 0 |  |
| copernicus_argentina_parana_alturas_historicas_1900_2026__wsun | short (H=8) | altura_media_m:mean, altura_min_m:mean, altura_max_m:mean, n_lecturas:last | 0 | 0 | 22 | 0 | no series left after the NaN / scale rules |
| cosuleabianca_eea_pm25_forecasting__h | short (H=48) | native | 17 | 0 | 18 | 0 |  |
| cosuleabianca_eea_pm25_forecasting__h | medium (H=480) | native | 0 | 0 | 35 | 0 | no series left after the NaN / scale rules |
| cosuleabianca_eea_pm25_forecasting__h | long (H=720) | native | 0 | 0 | 35 | 0 | no series left after the NaN / scale rules |
| cosuleabianca_eea_pm25_forecasting__d | short (H=30) | PM2.5:mean, NO2:mean, PM10:mean, temperature_2m:mean, relative_humidity_2m:mean, dew_point_2m:mean, precipitation:mean, surface_pressure:mean, wind_u:mean, wind_v:mean, day_of_week:last, day_of_month:last, is_weekend:last, hour_sin:last, hour_cos:last, month_sin:last, month_cos:last, PM2.5_lag_1h:mean, PM2.5_lag_2h:mean, PM2.5_lag_3h:mean, PM2.5_lag_6h:mean, PM2.5_lag_12h:mean, PM2.5_lag_24h:mean, PM2.5_lag_168h:mean, NO2_lag_1h:mean, NO2_lag_2h:mean, NO2_lag_3h:mean, NO2_lag_6h:mean, NO2_lag_12h:mean, NO2_lag_24h:mean, NO2_lag_168h:mean, PM10_lag_1h:mean, PM10_lag_2h:mean | 10 | 0 | 25 | 0 |  |
| cosuleabianca_eea_pm25_forecasting__wsun | short (H=8) | PM2.5:mean, NO2:mean, PM10:mean, temperature_2m:mean, relative_humidity_2m:mean, dew_point_2m:mean, precipitation:mean, surface_pressure:mean, wind_u:mean, wind_v:mean, day_of_week:last, day_of_month:last, is_weekend:last, hour_sin:last, hour_cos:last, month_sin:last, month_cos:last, PM2.5_lag_1h:mean, PM2.5_lag_2h:mean, PM2.5_lag_3h:mean, PM2.5_lag_6h:mean, PM2.5_lag_12h:mean, PM2.5_lag_24h:mean, PM2.5_lag_168h:mean, NO2_lag_1h:mean, NO2_lag_2h:mean, NO2_lag_3h:mean, NO2_lag_6h:mean, NO2_lag_12h:mean, NO2_lag_24h:mean, NO2_lag_168h:mean, PM10_lag_1h:mean, PM10_lag_2h:mean | 21 | 0 | 14 | 0 |  |
| danielrosehill_jerusalem_air_quality_shabbat__5min | short (H=48) | native | 0 | 0 | 12 | 0 | no series left after the NaN / scale rules |
| danielrosehill_jerusalem_air_quality_shabbat__5min | medium (H=480) | native | 0 | 0 | 12 | 0 | no series left after the NaN / scale rules |
| danielrosehill_jerusalem_air_quality_shabbat__5min | long (H=720) | native | 0 | 0 | 12 | 0 | no series left after the NaN / scale rules |
| danielrosehill_jerusalem_air_quality_shabbat__h | short (H=48) | value:mean, dow:last, is_friday:last, is_shabbat:last, shabbat_or_chag_exact:last, is_yom_tov:last, is_yom_kippur:last, is_pre_shabbat:last, is_post_havdalah:last, hour:last | 0 | 0 | 12 | 0 | no series left after the NaN / scale rules |
| danielrosehill_jerusalem_air_quality_shabbat__h | medium (H=480) | value:mean, dow:last, is_friday:last, is_shabbat:last, shabbat_or_chag_exact:last, is_yom_tov:last, is_yom_kippur:last, is_pre_shabbat:last, is_post_havdalah:last, hour:last | 0 | 0 | 12 | 0 | no series left after the NaN / scale rules |
| danielrosehill_jerusalem_air_quality_shabbat__h | long (H=720) | value:mean, dow:last, is_friday:last, is_shabbat:last, shabbat_or_chag_exact:last, is_yom_tov:last, is_yom_kippur:last, is_pre_shabbat:last, is_post_havdalah:last, hour:last | 0 | 1 | 11 | 0 | no series left after the NaN / scale rules |
| danielrosehill_jerusalem_air_quality_shabbat__d | short (H=30) | value:mean, dow:last, is_friday:last, is_shabbat:last, shabbat_or_chag_exact:last, is_yom_tov:last, is_yom_kippur:last, is_pre_shabbat:last, is_post_havdalah:last, hour:last | 0 | 1 | 11 | 0 | no series left after the NaN / scale rules |
| danipaez_velib_tft_default__20min | short (H=48) | native | 0 | 0 | 1000 | 0 | no series left after the NaN / scale rules |
| danipaez_velib_tft_default__20min | medium (H=480) | native | 0 | 2 | 998 | 0 | no series left after the NaN / scale rules |
| danipaez_velib_tft_default__20min | long (H=720) | native | 0 | 2 | 998 | 0 | no series left after the NaN / scale rules |
| danipaez_velib_tft_default__h | short (H=48) | available_bikes:mean, minutes_of_day:last, day_of_week:last, is_public_holiday:last, capacity:mean, temp:mean, wind:mean, snow:mean, rain:mean | 0 | 2 | 998 | 0 | no series left after the NaN / scale rules |
| danipaez_velib_tft_default__h | medium (H=480) | available_bikes:mean, minutes_of_day:last, day_of_week:last, is_public_holiday:last, capacity:mean, temp:mean, wind:mean, snow:mean, rain:mean | 0 | 7 | 991 | 0 | no series left after the NaN / scale rules |
| danipaez_velib_tft_default__h | long (H=720) | available_bikes:mean, minutes_of_day:last, day_of_week:last, is_public_holiday:last, capacity:mean, temp:mean, wind:mean, snow:mean, rain:mean | 0 | 10 | 988 | 0 | no series left after the NaN / scale rules |
| dar4huggingface_sniim_mexico_sample__d | short (H=30) | native | 0 | 0 | 10 | 0 | no series left after the NaN / scale rules |
| datadog_boom__5min | short (H=48) | native | 0 | 0 | 84 | 0 | no series left after the NaN / scale rules |
| datadog_boom__5min | medium (H=480) | native | 0 | 26 | 0 | 0 | 58 of 84 series long enough (< 75%) |
| datadog_boom__5min | long (H=720) | native | 0 | 26 | 0 | 0 | 58 of 84 series long enough (< 75%) |
| datadog_boom__h | short (H=48) | target:mean, target_0:mean, target_1:mean, target_2:mean, target_3:mean, target_4:mean, target_5:mean, target_6:mean, target_7:mean, target_8:mean, target_9:mean, target_10:mean, target_11:mean, target_12:mean, target_13:mean, target_14:mean, target_15:mean, target_16:mean, target_17:mean, target_18:mean, target_19:mean, target_20:mean, target_21:mean, target_22:mean, target_23:mean, target_24:mean, target_25:mean, target_26:mean, target_27:mean, target_28:mean, target_29:mean, target_30:mean | 0 | 48 | 36 | 0 | no series left after the NaN / scale rules |
| datadog_boom__h | medium (H=480) | target:mean, target_0:mean, target_1:mean, target_2:mean, target_3:mean, target_4:mean, target_5:mean, target_6:mean, target_7:mean, target_8:mean, target_9:mean, target_10:mean, target_11:mean, target_12:mean, target_13:mean, target_14:mean, target_15:mean, target_16:mean, target_17:mean, target_18:mean, target_19:mean, target_20:mean, target_21:mean, target_22:mean, target_23:mean, target_24:mean, target_25:mean, target_26:mean, target_27:mean, target_28:mean, target_29:mean, target_30:mean | 0 | 36 | 0 | 0 | 0 of 36 series long enough (< 75%) |
| datadog_boom__h | long (H=720) | target:mean, target_0:mean, target_1:mean, target_2:mean, target_3:mean, target_4:mean, target_5:mean, target_6:mean, target_7:mean, target_8:mean, target_9:mean, target_10:mean, target_11:mean, target_12:mean, target_13:mean, target_14:mean, target_15:mean, target_16:mean, target_17:mean, target_18:mean, target_19:mean, target_20:mean, target_21:mean, target_22:mean, target_23:mean, target_24:mean, target_25:mean, target_26:mean, target_27:mean, target_28:mean, target_29:mean, target_30:mean | 0 | 36 | 0 | 0 | 0 of 36 series long enough (< 75%) |
| datonic_spain_aemet_historical_weather__d | short (H=30) | native | 14 | 0 | 0 | 0 |  |
| datonic_spain_aemet_historical_weather__wsun | short (H=8) | target:mean | 14 | 0 | 0 | 0 |  |
| datonic_world_development_indicators_default__ys | short (H=6) | native | 943 | 0 | 53 | 3 |  |
| dhivakarnaidusunkari_mandi_india_daily_prices_2001_2026__d | short (H=30) | native | 5 | 0 | 995 | 0 |  |
| dhivakarnaidusunkari_mandi_india_daily_prices_2001_2026__wsun | short (H=8) | Min_Price:mean, Max_Price:mean, Modal_Price:mean | 60 | 116 | 824 | 0 |  |
| digitalhen_us_airport_wait_times__min | short (H=48) | native | 38 | 0 | 16 | 0 |  |
| digitalhen_us_airport_wait_times__min__medium | medium (H=480) | native | 28 | 0 | 26 | 0 |  |
| digitalhen_us_airport_wait_times__min__medium | long (H=720) | native | 28 | 0 | 26 | 0 |  |
| digitalhen_us_airport_wait_times__h | short (H=48) | target:mean | 48 | 0 | 6 | 0 |  |
| digitalhen_us_airport_wait_times__h | medium (H=480) | target:mean | 0 | 54 | 0 | 0 | 0 of 54 series long enough (< 75%) |
| digitalhen_us_airport_wait_times__h | long (H=720) | target:mean | 0 | 54 | 0 | 0 | 0 of 54 series long enough (< 75%) |
| dmariaa70_metraq_air_quality__h | short (H=48) | native | 607 | 0 | 0 | 0 |  |
| dmariaa70_metraq_air_quality__h | medium (H=480) | native | 607 | 0 | 0 | 0 |  |
| dmariaa70_metraq_air_quality__h | long (H=720) | native | 607 | 0 | 0 | 0 |  |
| dmariaa70_metraq_air_quality__d | short (H=30) | value:mean, utm_x:mean, utm_y:mean, is_interpolated:last | 607 | 0 | 0 | 0 |  |
| dmariaa70_metraq_air_quality__wsun | short (H=8) | value:mean, utm_x:mean, utm_y:mean, is_interpolated:last | 607 | 0 | 0 | 0 |  |
| donny_guo_airline_delay__d | short (H=30) | native | 509 | 0 | 485 | 5 |  |
| donny_guo_airline_delay__wsun | short (H=8) | op_carrier_fl_num:mean, dep_delay:mean, taxi_out:mean, wheels_off:mean, wheels_on:mean, taxi_in:mean, arr_delay:mean, distance:mean, carrier_delay:mean, weather_delay:mean, nas_delay:mean, security_delay:mean, late_aircraft_delay:mean, cancelled:last, diverted:mean | 559 | 222 | 215 | 3 |  |
| duckermaster_thaiclim_district__d | short (H=30) | native | 910 | 0 | 0 | 0 |  |
| duckermaster_thaiclim_district__wsun | short (H=8) | precip_mm:mean, precip_3d_mm:mean, precip_5d_mm:mean, precip_7d_mm:mean, temp_mean_c:mean, temp_max_c:mean, temp_min_c:mean, temp_dtr_c:mean, dry_spell_days:mean, wet_spell_days:mean, is_extreme_rain_r95p:mean, precip_ge_50mm:last, precip_ge_70mm:last, precip_ge_100mm:last | 910 | 0 | 0 | 0 |  |
| earino_austin_911_response__d | short (H=30) | native | 3 | 0 | 0 | 0 |  |
| earino_austin_911_response__wsun | short (H=8) | count:sum | 1 | 2 | 0 | 0 |  |
| electricsheepafrica_africa_wfp_food_prices_for_uganda__ms | short (H=12) | native | 9 | 0 | 284 | 0 |  |
| electricsheepafrica_wfp_nigeria_food_prices__ms | short (H=12) | native | 39 | 0 | 524 | 0 |  |
| elskow_weather4cast__h | short (H=48) | native | 0 | 0 | 10 | 0 | no series left after the NaN / scale rules |
| elskow_weather4cast__h | medium (H=480) | native | 0 | 0 | 10 | 0 | no series left after the NaN / scale rules |
| elskow_weather4cast__h | long (H=720) | native | 0 | 0 | 10 | 0 | no series left after the NaN / scale rules |
| elskow_weather4cast__d | short (H=30) | temperature_2m (°C):mean, relativehumidity_2m (%):mean, dewpoint_2m (°C):mean, apparent_temperature (°C):mean, pressure_msl (hPa):mean, surface_pressure (hPa):mean, snowfall (cm):last, cloudcover (%):mean, cloudcover_low (%):mean, cloudcover_mid (%):mean, cloudcover_high (%):mean, shortwave_radiation (W/m²):mean, direct_radiation (W/m²):mean, diffuse_radiation (W/m²):mean, direct_normal_irradiance (W/m²):mean, windspeed_10m (km/h):mean, windspeed_100m (km/h):mean, winddirection_10m (°):mean, winddirection_100m (°):mean, windgusts_10m (km/h):mean, et0_fao_evapotranspiration (mm):mean, vapor_pressure_deficit (kPa):mean, soil_temperature_0_to_7cm (°C):mean, soil_temperature_7_to_28cm (°C):mean, soil_temperature_28_to_100cm (°C):mean, soil_temperature_100_to_255cm (°C):mean, soil_moisture_0_to_7cm (m³/m³):mean, soil_moisture_7_to_28cm (m³/m³):mean, soil_moisture_28_to_100cm (m³/m³):mean, soil_moisture_100_to_255cm (m³/m³):mean | 10 | 0 | 0 | 0 |  |
| elskow_weather4cast__wsun | short (H=8) | temperature_2m (°C):mean, relativehumidity_2m (%):mean, dewpoint_2m (°C):mean, apparent_temperature (°C):mean, pressure_msl (hPa):mean, surface_pressure (hPa):mean, snowfall (cm):last, cloudcover (%):mean, cloudcover_low (%):mean, cloudcover_mid (%):mean, cloudcover_high (%):mean, shortwave_radiation (W/m²):mean, direct_radiation (W/m²):mean, diffuse_radiation (W/m²):mean, direct_normal_irradiance (W/m²):mean, windspeed_10m (km/h):mean, windspeed_100m (km/h):mean, winddirection_10m (°):mean, winddirection_100m (°):mean, windgusts_10m (km/h):mean, et0_fao_evapotranspiration (mm):mean, vapor_pressure_deficit (kPa):mean, soil_temperature_0_to_7cm (°C):mean, soil_temperature_7_to_28cm (°C):mean, soil_temperature_28_to_100cm (°C):mean, soil_temperature_100_to_255cm (°C):mean, soil_moisture_0_to_7cm (m³/m³):mean, soil_moisture_7_to_28cm (m³/m³):mean, soil_moisture_28_to_100cm (m³/m³):mean, soil_moisture_100_to_255cm (m³/m³):mean | 10 | 0 | 0 | 0 |  |
| emperor_mew_ooni_censorship_historical__d | short (H=30) | native | 467 | 0 | 167 | 0 |  |
| emperor_mew_ooni_censorship_historical__wsun | short (H=8) | measurement_count:mean, anomaly_count:mean, confirmed_count:mean, failure_count:mean, ok_count:mean, anomaly_rate:mean, confirmed_rate:mean | 545 | 0 | 89 | 0 |  |
| empulse_ai_empulse_street_fx_index__d | short (H=30) | native | 8 | 0 | 0 | 0 |  |
| empulse_ai_empulse_street_fx_index__wsun | short (H=8) | target:mean | 8 | 0 | 0 | 0 |  |
| emulated_inc_battery_capacity_trajectories_dataset_1_nca_2_ncm_3_ncm_nca_zips__10s | short (H=48) | native | 0 | 0 | 704 | 0 | no series left after the NaN / scale rules |
| emulated_inc_battery_capacity_trajectories_dataset_1_nca_2_ncm_3_ncm_nca_zips__10s | medium (H=480) | native | 0 | 0 | 704 | 0 | no series left after the NaN / scale rules |
| emulated_inc_battery_capacity_trajectories_dataset_1_nca_2_ncm_3_ncm_nca_zips__10s | long (H=720) | native | 0 | 0 | 704 | 0 | no series left after the NaN / scale rules |
| emulated_inc_sci_airport_low_ceiling_visibility__20h | short (H=48) | native | 0 | 0 | 1 | 0 | no series left after the NaN / scale rules |
| emulated_inc_sci_airport_low_ceiling_visibility__20h | medium (H=480) | native | 0 | 0 | 1 | 0 | no series left after the NaN / scale rules |
| emulated_inc_sci_airport_low_ceiling_visibility__20h | long (H=720) | native | 0 | 1 | 0 | 0 | 0 of 1 series long enough (< 75%) |
| emulated_inc_sci_airport_low_ceiling_visibility__d | short (H=30) | forecast_hour:mean, vis:mean, cloud_ceiling:mean, cloud_base:mean, t2m:mean, d2m:mean, rh2m:mean, u10:mean, v10:mean, gust:mean, t_skin:mean, p_sfc:mean, cloud_low:mean, cloud_mid:mean, cloud_high:mean, cloud_total:mean, cloud_bl:mean, precip_rate:mean, sw_down:mean, lw_down:mean, pbl_height:mean, lcl_height:mean, sensible_flux:mean, latent_flux:mean, friction_velocity:mean, snow_depth:mean, cloud_water_l1:mean, t_l1:mean, q_l1:mean, p_l1:mean, cloud_water_l3:mean, t_l3:mean, q_l3:mean | 0 | 0 | 1 | 0 | no series left after the NaN / scale rules |
| emulated_inc_sci_airport_low_ceiling_visibility__wsun | short (H=8) | forecast_hour:mean, vis:mean, cloud_ceiling:mean, cloud_base:mean, t2m:mean, d2m:mean, rh2m:mean, u10:mean, v10:mean, gust:mean, t_skin:mean, p_sfc:mean, cloud_low:mean, cloud_mid:mean, cloud_high:mean, cloud_total:mean, cloud_bl:mean, precip_rate:mean, sw_down:mean, lw_down:mean, pbl_height:mean, lcl_height:mean, sensible_flux:mean, latent_flux:mean, friction_velocity:mean, snow_depth:mean, cloud_water_l1:mean, t_l1:mean, q_l1:mean, p_l1:mean, cloud_water_l3:mean, t_l3:mean, q_l3:mean | 1 | 0 | 0 | 0 |  |
| emulated_inc_sci_booster_gmps_regulation_forecast__5s | short (H=48) | native | 613 | 0 | 0 | 1 |  |
| emulated_inc_sci_booster_gmps_regulation_forecast__5s | medium (H=480) | native | 613 | 0 | 0 | 1 |  |
| emulated_inc_sci_booster_gmps_regulation_forecast__5s | long (H=720) | native | 613 | 0 | 0 | 1 |  |
| emulated_inc_sci_lake_temperature_profiles_unmonitored__d | short (H=30) | native | 1000 | 0 | 0 | 0 |  |
| emulated_inc_sci_lake_temperature_profiles_unmonitored__wsun | short (H=8) | target:mean | 1000 | 0 | 0 | 0 |  |
| emulated_inc_sci_streamflow_ungauged_new_country__d | short (H=30) | native | 0 | 0 | 1000 | 0 | no series left after the NaN / scale rules |
| emulated_inc_sci_streamflow_ungauged_new_country__wsun | short (H=8) | target:mean, total_precipitation_sum:mean, temperature_2m_mean:mean, temperature_2m_min:mean, temperature_2m_max:mean, dewpoint_temperature_2m_mean:mean, surface_net_solar_radiation_mean:mean, surface_net_thermal_radiation_mean:mean, surface_pressure_mean:mean, snow_depth_water_equivalent_mean:mean, u_component_of_wind_10m_mean:mean, v_component_of_wind_10m_mean:mean, volumetric_soil_water_layer_1_mean:mean | 819 | 0 | 181 | 0 |  |
| epi_eval_cdc_nrevss_rsv__wsat | short (H=8) | native | 10 | 0 | 0 | 0 |  |
| epi_eval_cdc_nssp__wsat | short (H=8) | native | 50 | 0 | 0 | 0 |  |
| epi_eval_delphi_flusurv__wsat | short (H=8) | native | 0 | 0 | 3 | 0 | no series left after the NaN / scale rules |
| epi_eval_global_mobility__d | short (H=30) | native | 912 | 0 | 88 | 0 |  |
| epi_eval_global_mobility__wsun | short (H=8) | retail_and_recreation:mean, grocery_and_pharmacy:mean, parks:mean, transit_stations:mean, workplaces:mean, residential:mean | 957 | 5 | 38 | 0 |  |
| epi_eval_nhsn_hrd__wsat | short (H=8) | native | 64 | 0 | 2 | 0 |  |
| epi_eval_wikipedia_pageviews__d | short (H=30) | native | 1 | 0 | 0 | 0 |  |
| epi_eval_wikipedia_pageviews__wsun | short (H=8) | views:mean | 1 | 0 | 0 | 0 |  |
| feregrino_london_cycles__902s | short (H=48) | native | 0 | 0 | 815 | 0 | no series left after the NaN / scale rules |
| feregrino_london_cycles__902s | medium (H=480) | native | 0 | 2 | 813 | 0 | no series left after the NaN / scale rules |
| feregrino_london_cycles__902s | long (H=720) | native | 0 | 2 | 813 | 0 | no series left after the NaN / scale rules |
| feregrino_london_cycles__h | short (H=48) | bikes:mean, empty_docks:mean, docks:mean | 0 | 1 | 814 | 0 | no series left after the NaN / scale rules |
| feregrino_london_cycles__h | medium (H=480) | bikes:mean, empty_docks:mean, docks:mean | 0 | 1 | 813 | 0 | no series left after the NaN / scale rules |
| feregrino_london_cycles__h | long (H=720) | bikes:mean, empty_docks:mean, docks:mean | 0 | 3 | 811 | 0 | no series left after the NaN / scale rules |
| feregrino_london_cycles__d | short (H=30) | bikes:mean, empty_docks:mean, docks:mean | 1 | 787 | 27 | 0 |  |
| feregrino_london_cycles__wsun | short (H=8) | bikes:mean, empty_docks:mean, docks:mean | 4 | 803 | 8 | 0 |  |
| fidel_ts_bear_room__5min | short (H=48) | native | 0 | 0 | 129 | 15 | no series left after the NaN / scale rules |
| fidel_ts_bear_room__5min | medium (H=480) | native | 0 | 15 | 129 | 0 | no series left after the NaN / scale rules |
| fidel_ts_bear_room__5min | long (H=720) | native | 0 | 15 | 129 | 0 | no series left after the NaN / scale rules |
| fidel_ts_bear_room__h | short (H=48) | target:mean | 0 | 15 | 129 | 0 | no series left after the NaN / scale rules |
| fidel_ts_bear_room__h | medium (H=480) | target:mean | 0 | 129 | 0 | 0 | 0 of 129 series long enough (< 75%) |
| fidel_ts_bear_room__h | long (H=720) | target:mean | 0 | 129 | 0 | 0 | 0 of 129 series long enough (< 75%) |
| fidel_ts_california_iso__5min | short (H=48) | native | 11 | 0 | 0 | 0 |  |
| fidel_ts_california_iso__5min | medium (H=480) | native | 11 | 0 | 0 | 0 |  |
| fidel_ts_california_iso__5min | long (H=720) | native | 11 | 0 | 0 | 0 |  |
| fidel_ts_california_iso__h | short (H=48) | target:mean | 11 | 0 | 0 | 0 |  |
| fidel_ts_california_iso__h | medium (H=480) | target:mean | 11 | 0 | 0 | 0 |  |
| fidel_ts_california_iso__h | long (H=720) | target:mean | 11 | 0 | 0 | 0 |  |
| fidel_ts_california_iso__d | short (H=30) | target:mean | 11 | 0 | 0 | 0 |  |
| fidel_ts_canada_photovoltaics_plants__100s | short (H=48) | native | 11 | 0 | 0 | 0 |  |
| fidel_ts_canada_photovoltaics_plants__100s | medium (H=480) | native | 11 | 0 | 0 | 0 |  |
| fidel_ts_canada_photovoltaics_plants__100s | long (H=720) | native | 11 | 0 | 0 | 0 |  |
| fidel_ts_canada_photovoltaics_plants__h | short (H=48) | count:sum | 11 | 0 | 0 | 0 |  |
| fidel_ts_canada_photovoltaics_plants__h | medium (H=480) | count:sum | 0 | 11 | 0 | 0 | 0 of 11 series long enough (< 75%) |
| fidel_ts_canada_photovoltaics_plants__h | long (H=720) | count:sum | 0 | 11 | 0 | 0 | 0 of 11 series long enough (< 75%) |
| fidel_ts_nyc_traffic_speed__5min | short (H=48) | native | 89 | 0 | 0 | 0 |  |
| fidel_ts_nyc_traffic_speed__5min | medium (H=480) | native | 89 | 0 | 0 | 0 |  |
| fidel_ts_nyc_traffic_speed__5min | long (H=720) | native | 89 | 0 | 0 | 0 |  |
| fidel_ts_nyc_traffic_speed__h | short (H=48) | target:mean | 89 | 0 | 0 | 0 |  |
| fidel_ts_nyc_traffic_speed__h | medium (H=480) | target:mean | 89 | 0 | 0 | 0 |  |
| fidel_ts_nyc_traffic_speed__h | long (H=720) | target:mean | 89 | 0 | 0 | 0 |  |
| fidel_ts_nyc_traffic_speed__d | short (H=30) | target:mean | 89 | 0 | 0 | 0 |  |
| food_ai_nexus_global_food_safety_incidents__d | short (H=30) | native | 8 | 0 | 0 | 0 |  |
| food_ai_nexus_global_food_safety_incidents__wsun | short (H=8) | count:sum | 8 | 0 | 0 | 0 |  |
| fuel_prices_eu_european_fuel_prices__d | short (H=30) | native | 96 | 0 | 0 | 0 |  |
| fulldecent_septa_regionalrail_otp__d | short (H=30) | native | 12 | 0 | 1 | 0 |  |
| furusu_japan_max_temperature__d | short (H=30) | native | 872 | 0 | 34 | 1 |  |
| furusu_japan_max_temperature__wsun | short (H=8) | max_temp:mean | 867 | 1 | 39 | 0 |  |
| geoweaver_snowwaterequivalent_trainingdata__d | short (H=30) | native | 752 | 0 | 80 | 0 |  |
| geoweaver_snowwaterequivalent_trainingdata__wsun | short (H=8) | SWE:mean, swe_value:mean, change_in_swe_inch:mean, snow_depth:mean, air_temperature_observed_f:mean, precipitation_amount:mean, relative_humidity_rmin:mean, potential_evapotranspiration:mean, air_temperature_tmmx:mean, relative_humidity_rmax:mean, mean_vapor_pressure_deficit:mean, air_temperature_tmmn:mean, wind_speed:mean, Aspect:mean, Curvature:mean, Northness:mean, Eastness:mean, ndsi:mean, Slope:mean, SWE_1:mean, air_temperature_tmmn_1:mean, potential_evapotranspiration_1:mean, mean_vapor_pressure_deficit_1:mean, relative_humidity_rmax_1:mean, relative_humidity_rmin_1:mean, precipitation_amount_1:mean, air_temperature_tmmx_1:mean, wind_speed_1:mean, ndsi_1:mean, SWE_2:mean, air_temperature_tmmn_2:mean, potential_evapotranspiration_2:mean, mean_vapor_pressure_deficit_2:mean | 768 | 4 | 55 | 5 |  |
| gft_ttm4hvac_source_all_train__15min | short (H=48) | native | 3 | 0 | 0 | 0 |  |
| gft_ttm4hvac_source_all_train__15min | medium (H=480) | native | 3 | 0 | 0 | 0 |  |
| gft_ttm4hvac_source_all_train__15min | long (H=720) | native | 3 | 0 | 0 | 0 |  |
| gft_ttm4hvac_source_all_train__h | short (H=48) | Outdoor Air Temperature (C):mean, Heating Setpoint (C):mean, Cooling Setpoint (C):mean, Room Air Temperature (C):mean, Outdoor Humidity (%):mean, Wind Speed (m/s):mean, Direct Solar Radiation (W/m^2):mean, HVAC Power Consumption (W):mean, is_default:last | 3 | 0 | 0 | 0 |  |
| gft_ttm4hvac_source_all_train__h | medium (H=480) | Outdoor Air Temperature (C):mean, Heating Setpoint (C):mean, Cooling Setpoint (C):mean, Room Air Temperature (C):mean, Outdoor Humidity (%):mean, Wind Speed (m/s):mean, Direct Solar Radiation (W/m^2):mean, HVAC Power Consumption (W):mean, is_default:last | 3 | 0 | 0 | 0 |  |
| gft_ttm4hvac_source_all_train__h | long (H=720) | Outdoor Air Temperature (C):mean, Heating Setpoint (C):mean, Cooling Setpoint (C):mean, Room Air Temperature (C):mean, Outdoor Humidity (%):mean, Wind Speed (m/s):mean, Direct Solar Radiation (W/m^2):mean, HVAC Power Consumption (W):mean, is_default:last | 3 | 0 | 0 | 0 |  |
| gft_ttm4hvac_source_all_train__d | short (H=30) | Outdoor Air Temperature (C):mean, Heating Setpoint (C):mean, Cooling Setpoint (C):mean, Room Air Temperature (C):mean, Outdoor Humidity (%):mean, Wind Speed (m/s):mean, Direct Solar Radiation (W/m^2):mean, HVAC Power Consumption (W):mean, is_default:last | 3 | 0 | 0 | 0 |  |
| gladcolor_hourly_population_us__h | short (H=48) | native | 1000 | 0 | 0 | 0 |  |
| gladcolor_hourly_population_us__h | medium (H=480) | native | 1000 | 0 | 0 | 0 |  |
| gladcolor_hourly_population_us__h | long (H=720) | native | 1000 | 0 | 0 | 0 |  |
| gladcolor_hourly_population_us__d | short (H=30) | value:mean | 1000 | 0 | 0 | 0 |  |
| globalsquiddatabase_tuna_fisheries_dataset__ms | short (H=12) | native | 282 | 0 | 714 | 4 |  |
| glucofmbench_glucofm_bench__5min | short (H=48) | native | 268 | 0 | 16 | 0 |  |
| glucofmbench_glucofm_bench__5min__medium | medium (H=480) | native | 211 | 38 | 35 | 0 |  |
| glucofmbench_glucofm_bench__5min | long (H=720) | native | 0 | 109 | 0 | 0 | 175 of 284 series long enough (< 75%) |
| glucofmbench_glucofm_bench__h | short (H=48) | target:mean | 38 | 234 | 12 | 0 |  |
| glucofmbench_glucofm_bench__h | medium (H=480) | target:mean | 0 | 34 | 0 | 0 | 16 of 50 series long enough (< 75%) |
| glucofmbench_glucofm_bench__h | long (H=720) | target:mean | 0 | 45 | 0 | 0 | 5 of 50 series long enough (< 75%) |
| glucofmbench_glucofm_bench__d | short (H=30) | target:mean | 1 | 283 | 0 | 0 |  |
| hackermango_hyderabaduniversity_hvac_dataset__10min | short (H=48) | native | 14 | 0 | 0 | 0 |  |
| hackermango_hyderabaduniversity_hvac_dataset__10min | medium (H=480) | native | 14 | 0 | 0 | 0 |  |
| hackermango_hyderabaduniversity_hvac_dataset__10min | long (H=720) | native | 14 | 0 | 0 | 0 |  |
| hackermango_hyderabaduniversity_hvac_dataset__h | short (H=48) | target:mean | 14 | 0 | 0 | 0 |  |
| hackermango_hyderabaduniversity_hvac_dataset__h | medium (H=480) | target:mean | 14 | 0 | 0 | 0 |  |
| hackermango_hyderabaduniversity_hvac_dataset__h | long (H=720) | target:mean | 0 | 14 | 0 | 0 | 0 of 14 series long enough (< 75%) |
| hexquant_stocks_daily_price__d | short (H=30) | native | 0 | 0 | 1000 | 0 | no series left after the NaN / scale rules |
| hexquant_stocks_daily_price__wsun | short (H=8) | open:mean, high:mean, low:mean, close:mean, volume:mean, adj_close:mean | 0 | 64 | 936 | 0 | no series left after the NaN / scale rules |
| hhs_official_nndss_weekly_data__wsat | short (H=8) | native | 356 | 0 | 294 | 0 |  |
| hhs_official_sars_cov_2_variant_proportions__wfri | short (H=8) | native | 0 | 0 | 4 | 0 | no series left after the NaN / scale rules |
| hhs_official_weekly_counts_of_deaths_from_all_causes_by_jurisdi__wsat | short (H=8) | native | 2 | 0 | 0 | 0 |  |
| hhs_official_weekly_provisional_counts_of_deaths_by_state_and_s__wsat | short (H=8) | native | 16 | 0 | 0 | 0 |  |
| hhs_official_weekly_rates_of_laboratory_confirmed_rsv_hospitali__wsat | short (H=8) | native | 13 | 0 | 0 | 0 |  |
| hhs_official_weekly_united_states_covid_19_hospitalization_metr__wsat | short (H=8) | native | 66 | 0 | 1 | 0 |  |
| hq_bench_quito_corpus_hour__h | short (H=48) | native | 1000 | 0 | 0 | 0 |  |
| hq_bench_quito_corpus_hour__h | medium (H=480) | native | 1000 | 0 | 0 | 0 |  |
| hq_bench_quito_corpus_hour__h | long (H=720) | native | 1000 | 0 | 0 | 0 |  |
| hq_bench_quito_corpus_hour__d | short (H=30) | ind_1:mean, ind_2:mean, ind_3:mean, ind_4:mean, ind_5:mean | 1000 | 0 | 0 | 0 |  |
| hq_bench_quito_corpus_hour__wsun | short (H=8) | ind_1:mean, ind_2:mean, ind_3:mean, ind_4:mean, ind_5:mean | 1000 | 0 | 0 | 0 |  |
| hq_bench_quito_corpus_min__10min | short (H=48) | native | 999 | 0 | 0 | 0 |  |
| hq_bench_quito_corpus_min__10min | medium (H=480) | native | 999 | 0 | 0 | 0 |  |
| hq_bench_quito_corpus_min__10min | long (H=720) | native | 999 | 0 | 0 | 0 |  |
| hq_bench_quito_corpus_min__h | short (H=48) | ind_1:mean, ind_2:mean, ind_3:mean, ind_4:mean, ind_5:mean | 999 | 0 | 0 | 0 |  |
| hq_bench_quito_corpus_min__h | medium (H=480) | ind_1:mean, ind_2:mean, ind_3:mean, ind_4:mean, ind_5:mean | 0 | 999 | 0 | 0 | 0 of 999 series long enough (< 75%) |
| hq_bench_quito_corpus_min__h | long (H=720) | ind_1:mean, ind_2:mean, ind_3:mean, ind_4:mean, ind_5:mean | 0 | 999 | 0 | 0 | 0 of 999 series long enough (< 75%) |
| hq_bench_quitobench_hour_min__h | short (H=48) | native | 1000 | 0 | 0 | 0 |  |
| hq_bench_quitobench_hour_min__h | medium (H=480) | native | 0 | 593 | 0 | 0 | 407 of 1000 series long enough (< 75%) |
| hq_bench_quitobench_hour_min__h | long (H=720) | native | 0 | 593 | 0 | 0 | 407 of 1000 series long enough (< 75%) |
| hq_bench_quitobench_hour_min__d | short (H=30) | ind_1:mean, ind_2:mean, ind_3:mean, ind_4:mean, ind_5:mean | 407 | 593 | 0 | 0 |  |
| hq_bench_quitobench_hour_min__wsun | short (H=8) | ind_1:mean, ind_2:mean, ind_3:mean, ind_4:mean, ind_5:mean | 407 | 593 | 0 | 0 |  |
| hq_bench_quitobench_hour_min_r577__h | short (H=48) | native | 1000 | 0 | 0 | 0 |  |
| hq_bench_quitobench_hour_min_r577__h | medium (H=480) | native | 0 | 593 | 0 | 0 | 407 of 1000 series long enough (< 75%) |
| hq_bench_quitobench_hour_min_r577__h | long (H=720) | native | 0 | 593 | 0 | 0 | 407 of 1000 series long enough (< 75%) |
| hq_bench_quitobench_hour_min_r577__d | short (H=30) | ind_1:mean, ind_2:mean, ind_3:mean, ind_4:mean, ind_5:mean | 407 | 593 | 0 | 0 |  |
| hq_bench_quitobench_hour_min_r577__wsun | short (H=8) | ind_1:mean, ind_2:mean, ind_3:mean, ind_4:mean, ind_5:mean | 407 | 593 | 0 | 0 |  |
| hsilvosa_omie_electricity__d | short (H=30) | native | 1 | 0 | 2 | 0 |  |
| hsilvosa_safecast_radiation__5s | short (H=48) | native | 607 | 0 | 390 | 3 |  |
| hsilvosa_safecast_radiation__5s | medium (H=480) | native | 0 | 322 | 0 | 0 | 678 of 1000 series long enough (< 75%) |
| hsilvosa_safecast_radiation__5s | long (H=720) | native | 0 | 430 | 0 | 0 | 570 of 1000 series long enough (< 75%) |
| j_marcos_gs_brazil_solar_generation__h | short (H=48) | native | 1 | 0 | 0 | 0 |  |
| j_marcos_gs_brazil_solar_generation__h | medium (H=480) | native | 1 | 0 | 0 | 0 |  |
| j_marcos_gs_brazil_solar_generation__h | long (H=720) | native | 1 | 0 | 0 | 0 |  |
| j_marcos_gs_brazil_solar_generation__d | short (H=30) | target:mean | 1 | 0 | 0 | 0 |  |
| j_marcos_gs_brazil_solar_generation__wsun | short (H=8) | target:mean | 1 | 0 | 0 | 0 |  |
| jainam03_rescast_100k_ecobee__15min | short (H=48) | native | 4 | 0 | 0 | 0 |  |
| jainam03_rescast_100k_ecobee__15min | medium (H=480) | native | 4 | 0 | 0 | 0 |  |
| jainam03_rescast_100k_ecobee__15min | long (H=720) | native | 4 | 0 | 0 | 0 |  |
| jainam03_rescast_100k_ecobee__h | short (H=48) | Indoor Temp:mean, Cooling Setpoint:mean, Heating Setpoint:mean, Outdoor Temp:mean, Relative Humidity:mean | 4 | 0 | 0 | 0 |  |
| jainam03_rescast_100k_ecobee__h | medium (H=480) | Indoor Temp:mean, Cooling Setpoint:mean, Heating Setpoint:mean, Outdoor Temp:mean, Relative Humidity:mean | 4 | 0 | 0 | 0 |  |
| jainam03_rescast_100k_ecobee__h | long (H=720) | Indoor Temp:mean, Cooling Setpoint:mean, Heating Setpoint:mean, Outdoor Temp:mean, Relative Humidity:mean | 4 | 0 | 0 | 0 |  |
| jainam03_rescast_100k_ecobee__d | short (H=30) | Indoor Temp:mean, Cooling Setpoint:mean, Heating Setpoint:mean, Outdoor Temp:mean, Relative Humidity:mean | 4 | 0 | 0 | 0 |  |
| jainam03_rescast_100k_heapo_timeseries_data__15min | short (H=48) | native | 988 | 0 | 11 | 0 |  |
| jainam03_rescast_100k_heapo_timeseries_data__15min__medium | medium (H=480) | native | 967 | 0 | 29 | 3 |  |
| jainam03_rescast_100k_heapo_timeseries_data__15min__long | long (H=720) | native | 968 | 1 | 28 | 2 |  |
| jainam03_rescast_100k_heapo_timeseries_data__h | short (H=48) | target:mean | 968 | 0 | 28 | 3 |  |
| jainam03_rescast_100k_heapo_timeseries_data__h__medium | medium (H=480) | target:mean | 963 | 8 | 28 | 0 |  |
| jainam03_rescast_100k_heapo_timeseries_data__h__long | long (H=720) | target:mean | 962 | 11 | 26 | 0 |  |
| jainam03_rescast_100k_heapo_timeseries_data__d | short (H=30) | target:mean | 953 | 20 | 26 | 0 |  |
| jainam03_rescast_100k_heapo_timeseries_data__wsun | short (H=8) | target:mean | 696 | 267 | 36 | 0 |  |
| jainam03_rescast_100k_ideal__15min | short (H=48) | native | 999 | 0 | 0 | 0 |  |
| jainam03_rescast_100k_ideal__15min | medium (H=480) | native | 999 | 0 | 0 | 0 |  |
| jainam03_rescast_100k_ideal__15min | long (H=720) | native | 999 | 0 | 0 | 0 |  |
| jainam03_rescast_100k_ideal__h | short (H=48) | target:mean | 999 | 0 | 0 | 0 |  |
| jainam03_rescast_100k_ideal__h | medium (H=480) | target:mean | 999 | 0 | 0 | 0 |  |
| jainam03_rescast_100k_ideal__h__long | long (H=720) | target:mean | 968 | 31 | 0 | 0 |  |
| jainam03_rescast_100k_ideal__d | short (H=30) | target:mean | 828 | 171 | 0 | 0 |  |
| jainam03_rescast_100k_ideal__wsun | short (H=8) | target:mean | 149 | 850 | 0 | 0 |  |
| jainam03_rescast_100k_nest__15min | short (H=48) | native | 6 | 0 | 1 | 0 |  |
| jainam03_rescast_100k_nest__15min | medium (H=480) | native | 6 | 0 | 1 | 0 |  |
| jainam03_rescast_100k_nest__15min | long (H=720) | native | 6 | 0 | 1 | 0 |  |
| jainam03_rescast_100k_nest__h | short (H=48) | target:mean | 6 | 0 | 1 | 0 |  |
| jainam03_rescast_100k_nest__h | medium (H=480) | target:mean | 6 | 0 | 1 | 0 |  |
| jainam03_rescast_100k_nest__h | long (H=720) | target:mean | 6 | 0 | 1 | 0 |  |
| jainam03_rescast_100k_nest__d | short (H=30) | target:mean | 7 | 0 | 0 | 0 |  |
| jainam03_rescast_100k_refit__15min | short (H=48) | native | 20 | 0 | 0 | 0 |  |
| jainam03_rescast_100k_refit__15min | medium (H=480) | native | 20 | 0 | 0 | 0 |  |
| jainam03_rescast_100k_refit__15min | long (H=720) | native | 20 | 0 | 0 | 0 |  |
| jainam03_rescast_100k_refit__h | short (H=48) | target:mean | 20 | 0 | 0 | 0 |  |
| jainam03_rescast_100k_refit__h | medium (H=480) | target:mean | 20 | 0 | 0 | 0 |  |
| jainam03_rescast_100k_refit__h | long (H=720) | target:mean | 20 | 0 | 0 | 0 |  |
| jainam03_rescast_100k_refit__d | short (H=30) | target:mean | 20 | 0 | 0 | 0 |  |
| jainam03_rescast_100k_refit__wsun | short (H=8) | target:mean | 19 | 1 | 0 | 0 |  |
| jhumbl_orr_station_usage__ys | short (H=6) | native | 998 | 0 | 2 | 0 |  |
| jhumbl_police_crime_london__d | short (H=30) | native | 937 | 0 | 0 | 63 |  |
| jhumbl_police_crime_london__wsun | short (H=8) | count:sum | 900 | 90 | 0 | 10 |  |
| jhumbl_tfl_network_demand__d | short (H=30) | native | 37 | 0 | 394 | 0 |  |
| jhumbl_tfl_network_demand__wsun | short (H=8) | EntryTapCount:mean, ExitTapCount:mean | 373 | 3 | 55 | 0 |  |
| jotone_arpa_osmer_weatherstations__h | short (H=48) | native | 12 | 0 | 39 | 0 |  |
| jotone_arpa_osmer_weatherstations__h__medium | medium (H=480) | native | 1 | 0 | 50 | 0 |  |
| jotone_arpa_osmer_weatherstations__h__medium | long (H=720) | native | 1 | 0 | 50 | 0 |  |
| jotone_arpa_osmer_weatherstations__d | short (H=30) | rr:mean, prec_type:last, t180:mean, v10:mean, ff10:last, dd10:mean, dd10max:mean, vmax:mean, rg:mean, cloudiness:last, rh:mean, press:mean, bf:mean, t50:mean, tm10:mean | 7 | 0 | 44 | 0 |  |
| jotone_arpa_osmer_weatherstations__wsun | short (H=8) | rr:mean, prec_type:last, t180:mean, v10:mean, ff10:last, dd10:mean, dd10max:mean, vmax:mean, rg:mean, cloudiness:last, rh:mean, press:mean, bf:mean, t50:mean, tm10:mean | 37 | 0 | 14 | 0 |  |
| jotone_pcfvg_weatherstations__15min | short (H=48) | native | 779 | 0 | 207 | 12 |  |
| jotone_pcfvg_weatherstations__15min__medium | medium (H=480) | native | 490 | 9 | 491 | 8 |  |
| jotone_pcfvg_weatherstations__15min__long | long (H=720) | native | 517 | 13 | 462 | 6 |  |
| jotone_pcfvg_weatherstations__h | short (H=48) | value:mean | 490 | 8 | 491 | 9 |  |
| jotone_pcfvg_weatherstations__h__medium | medium (H=480) | value:mean | 470 | 58 | 462 | 0 |  |
| jotone_pcfvg_weatherstations__h__long | long (H=720) | value:mean | 560 | 92 | 338 | 0 |  |
| jotone_pcfvg_weatherstations__d | short (H=30) | value:mean | 815 | 108 | 75 | 0 |  |
| jotone_pcfvg_weatherstations__wsun | short (H=8) | value:mean | 810 | 164 | 24 | 0 |  |
| juliensimon_auroral_electrojet_index__d | short (H=30) | native | 1 | 0 | 0 | 0 |  |
| juliensimon_auroral_electrojet_index__wsun | short (H=8) | count:sum | 1 | 0 | 0 | 0 |  |
| juliensimon_celestrak_space_weather__d | short (H=30) | native | 0 | 0 | 29 | 0 | no series left after the NaN / scale rules |
| juliensimon_celestrak_space_weather__wsun | short (H=8) | target:mean | 29 | 0 | 0 | 0 |  |
| juliensimon_dst_index__h | short (H=48) | native | 1 | 0 | 1 | 0 |  |
| juliensimon_dst_index__h | medium (H=480) | native | 0 | 0 | 2 | 0 | no series left after the NaN / scale rules |
| juliensimon_dst_index__h | long (H=720) | native | 0 | 0 | 2 | 0 | no series left after the NaN / scale rules |
| juliensimon_dst_index__d | short (H=30) | target:mean | 0 | 0 | 2 | 0 | no series left after the NaN / scale rules |
| juliensimon_dst_index__wsun | short (H=8) | target:mean | 0 | 0 | 2 | 0 | no series left after the NaN / scale rules |
| juliensimon_f107_solar_flux__d | short (H=30) | native | 2 | 0 | 0 | 0 |  |
| juliensimon_f107_solar_flux__wsun | short (H=8) | target:mean | 2 | 0 | 0 | 0 |  |
| juliensimon_geomagnetic_kp_index__3h | short (H=48) | native | 3 | 0 | 0 | 0 |  |
| juliensimon_geomagnetic_kp_index__3h | medium (H=480) | native | 3 | 0 | 0 | 0 |  |
| juliensimon_geomagnetic_kp_index__3h | long (H=720) | native | 0 | 3 | 0 | 0 | 0 of 3 series long enough (< 75%) |
| juliensimon_geomagnetic_kp_index__d | short (H=30) | target:mean | 3 | 0 | 0 | 0 |  |
| juliensimon_goes_xray_flux__min | short (H=48) | native | 1 | 0 | 0 | 0 |  |
| juliensimon_goes_xray_flux__min | medium (H=480) | native | 0 | 0 | 1 | 0 | no series left after the NaN / scale rules |
| juliensimon_goes_xray_flux__min | long (H=720) | native | 0 | 0 | 1 | 0 | no series left after the NaN / scale rules |
| juliensimon_goes_xray_flux__h | short (H=48) | target:mean | 1 | 0 | 0 | 0 |  |
| juliensimon_goes_xray_flux__h | medium (H=480) | target:mean | 0 | 1 | 0 | 0 | 0 of 1 series long enough (< 75%) |
| juliensimon_goes_xray_flux__h | long (H=720) | target:mean | 0 | 1 | 0 | 0 | 0 of 1 series long enough (< 75%) |
| juliensimon_iers_earth_orientation__d | short (H=30) | native | 0 | 0 | 3 | 0 | no series left after the NaN / scale rules |
| juliensimon_iers_earth_orientation__wsun | short (H=8) | target:mean | 3 | 0 | 0 | 0 |  |
| juliensimon_nasa_maven_kp_insitu__8s | short (H=48) | native | 1 | 0 | 0 | 0 |  |
| juliensimon_nasa_maven_kp_insitu__8s | medium (H=480) | native | 1 | 0 | 0 | 0 |  |
| juliensimon_nasa_maven_kp_insitu__8s | long (H=720) | native | 1 | 0 | 0 | 0 |  |
| juliensimon_neutron_monitor_cosmic_rays__2h | short (H=48) | native | 0 | 0 | 5 | 0 | no series left after the NaN / scale rules |
| juliensimon_neutron_monitor_cosmic_rays__2h | medium (H=480) | native | 0 | 0 | 5 | 0 | no series left after the NaN / scale rules |
| juliensimon_neutron_monitor_cosmic_rays__2h | long (H=720) | native | 0 | 0 | 5 | 0 | no series left after the NaN / scale rules |
| juliensimon_neutron_monitor_cosmic_rays__d | short (H=30) | count_rate:mean, daily_mean_count_rate:mean, pct_deviation:mean | 0 | 0 | 5 | 0 | no series left after the NaN / scale rules |
| juliensimon_neutron_monitor_cosmic_rays__wsun | short (H=8) | count_rate:mean, daily_mean_count_rate:mean, pct_deviation:mean | 0 | 0 | 5 | 0 | no series left after the NaN / scale rules |
| juliensimon_omni_solar_wind_parameters__h | short (H=48) | native | 0 | 0 | 35 | 0 | no series left after the NaN / scale rules |
| juliensimon_omni_solar_wind_parameters__h | medium (H=480) | native | 0 | 0 | 35 | 0 | no series left after the NaN / scale rules |
| juliensimon_omni_solar_wind_parameters__h | long (H=720) | native | 0 | 0 | 35 | 0 | no series left after the NaN / scale rules |
| juliensimon_omni_solar_wind_parameters__d | short (H=30) | target:mean | 8 | 0 | 27 | 0 |  |
| juliensimon_omni_solar_wind_parameters__wsun | short (H=8) | target:mean | 8 | 0 | 27 | 0 |  |
| juliensimon_solar_wind__min | short (H=48) | native | 0 | 0 | 12 | 0 | no series left after the NaN / scale rules |
| juliensimon_solar_wind__min | medium (H=480) | native | 0 | 0 | 12 | 0 | no series left after the NaN / scale rules |
| juliensimon_solar_wind__min | long (H=720) | native | 0 | 0 | 12 | 0 | no series left after the NaN / scale rules |
| juliensimon_solar_wind__h | short (H=48) | target:mean | 0 | 0 | 12 | 0 | no series left after the NaN / scale rules |
| juliensimon_solar_wind__h | medium (H=480) | target:mean | 0 | 12 | 0 | 0 | 0 of 12 series long enough (< 75%) |
| juliensimon_solar_wind__h | long (H=720) | target:mean | 0 | 12 | 0 | 0 | 0 of 12 series long enough (< 75%) |
| jwigginton_timeseries_1mn_sp500_default__min | short (H=48) | native | 1 | 0 | 0 | 0 |  |
| jwigginton_timeseries_1mn_sp500_default__min | medium (H=480) | native | 0 | 1 | 0 | 0 | 0 of 1 series long enough (< 75%) |
| jwigginton_timeseries_1mn_sp500_default__min | long (H=720) | native | 0 | 1 | 0 | 0 | 0 of 1 series long enough (< 75%) |
| kawsersikder_bangladesh_stock_market_dataset__d | short (H=30) | native | 0 | 0 | 1000 | 0 | no series left after the NaN / scale rules |
| kawsersikder_bangladesh_stock_market_dataset__wsun | short (H=8) | target:mean | 0 | 272 | 728 | 0 | no series left after the NaN / scale rules |
| khajlk_copenhagen_air_quality_hourly__h | short (H=48) | native | 1 | 0 | 0 | 0 |  |
| khajlk_copenhagen_air_quality_hourly__h | medium (H=480) | native | 1 | 0 | 0 | 0 |  |
| khajlk_copenhagen_air_quality_hourly__h | long (H=720) | native | 1 | 0 | 0 | 0 |  |
| khajlk_copenhagen_air_quality_hourly__d | short (H=30) | temperature_2m:mean, wind_speed_10m:mean, wind_direction_10m:mean, relative_humidity_2m:mean, boundary_layer_height:mean | 1 | 0 | 0 | 0 |  |
| khajlk_copenhagen_air_quality_hourly__wsun | short (H=8) | temperature_2m:mean, wind_speed_10m:mean, wind_direction_10m:mean, relative_humidity_2m:mean, boundary_layer_height:mean | 1 | 0 | 0 | 0 |  |
| kpubdata_seoul_apartment_trades__31d | short (H=12) | native | 0 | 0 | 4 | 0 | no series left after the NaN / scale rules |
| leadlagforecasting_llf_github_default__d | short (H=30) | native | 993 | 0 | 0 | 7 |  |
| leadlagforecasting_llf_github_default__wsun | short (H=8) | target:mean, cumulative_pushes_diff:mean, cumulative_forks_diff:last | 980 | 0 | 0 | 20 |  |
| marplan6_batgrad_pozzato_2022__s | short (H=48) | native | 3 | 0 | 6 | 0 |  |
| marplan6_batgrad_pozzato_2022__s | medium (H=480) | native | 3 | 0 | 6 | 0 |  |
| marplan6_batgrad_pozzato_2022__s | long (H=720) | native | 3 | 0 | 6 | 0 |  |
| matanzig_disney_theme_park_queue_dynamics__d | short (H=30) | native | 37 | 0 | 0 | 0 |  |
| matanzig_disney_theme_park_queue_dynamics__wsun | short (H=8) | DEB_TIME_HOUR:mean, WAIT_TIME_MAX:mean, NB_UNITS:mean, GUEST_CARRIED:mean, CAPACITY:mean, ADJUST_CAPACITY:mean, DOWNTIME:mean, NB_MAX_UNIT:mean, TOTAL_DAILY_ATTENDANCE:mean, temp:mean, humidity:mean, wind_speed:mean, temp_rounded:mean, IS_BROKEN:mean, IS_BLACK_SWAN:mean | 37 | 0 | 0 | 0 |  |
| matrosovdani_moscow_parking_occupancy__1799s | short (H=48) | native | 3 | 0 | 188 | 0 |  |
| matrosovdani_moscow_parking_occupancy__1799s | medium (H=480) | native | 0 | 0 | 191 | 0 | no series left after the NaN / scale rules |
| matrosovdani_moscow_parking_occupancy__1799s | long (H=720) | native | 0 | 0 | 191 | 0 | no series left after the NaN / scale rules |
| matrosovdani_moscow_parking_occupancy__h | short (H=48) | free_spaces:mean, free_handicapped_spaces:mean, occupancy_rate:mean | 0 | 0 | 191 | 0 | no series left after the NaN / scale rules |
| matrosovdani_moscow_parking_occupancy__h | medium (H=480) | free_spaces:mean, free_handicapped_spaces:mean, occupancy_rate:mean | 0 | 0 | 191 | 0 | no series left after the NaN / scale rules |
| matrosovdani_moscow_parking_occupancy__h | long (H=720) | free_spaces:mean, free_handicapped_spaces:mean, occupancy_rate:mean | 0 | 0 | 191 | 0 | no series left after the NaN / scale rules |
| matrosovdani_moscow_parking_occupancy__d | short (H=30) | free_spaces:mean, free_handicapped_spaces:mean, occupancy_rate:mean | 188 | 2 | 1 | 0 |  |
| matrosovdani_moscow_parking_occupancy__wsun | short (H=8) | free_spaces:mean, free_handicapped_spaces:mean, occupancy_rate:mean | 0 | 25 | 166 | 0 | no series left after the NaN / scale rules |
| meganariley_daily_aqi__h | short (H=48) | native | 254 | 0 | 723 | 1 |  |
| meganariley_daily_aqi__h__medium | medium (H=480) | native | 150 | 3 | 825 | 0 |  |
| meganariley_daily_aqi__h__long | long (H=720) | native | 153 | 5 | 820 | 0 |  |
| meganariley_daily_aqi__d | short (H=30) | value:mean, site_num:mean, mdl:mean, aqi:mean | 427 | 9 | 542 | 0 |  |
| meganariley_daily_aqi__wsun | short (H=8) | value:mean, site_num:mean, mdl:mean, aqi:mean | 680 | 29 | 269 | 0 |  |
| meteoswiss_peakweather__10min | short (H=48) | native | 984 | 0 | 16 | 0 |  |
| meteoswiss_peakweather__10min__medium | medium (H=480) | native | 976 | 0 | 24 | 0 |  |
| meteoswiss_peakweather__10min__medium | long (H=720) | native | 976 | 0 | 24 | 0 |  |
| meteoswiss_peakweather__h | short (H=48) | target:mean | 983 | 0 | 17 | 0 |  |
| meteoswiss_peakweather__h__medium | medium (H=480) | target:mean | 986 | 0 | 14 | 0 |  |
| meteoswiss_peakweather__h | long (H=720) | target:mean | 983 | 0 | 17 | 0 |  |
| meteoswiss_peakweather__d | short (H=30) | target:mean | 983 | 4 | 13 | 0 |  |
| mihnathul_celljar__5s | short (H=48) | native | 515 | 0 | 257 | 0 |  |
| mihnathul_celljar__5s | medium (H=480) | native | 0 | 672 | 0 | 0 | 100 of 772 series long enough (< 75%) |
| mihnathul_celljar__5s | long (H=720) | native | 0 | 711 | 0 | 0 | 61 of 772 series long enough (< 75%) |
| misikoff_zillow_days_on_market__wsat | short (H=8) | native | 879 | 0 | 0 | 0 |  |
| misikoff_zillow_for_sale_listings__wsat | short (H=8) | native | 874 | 0 | 0 | 0 |  |
| misikoff_zillow_home_values__me | short (H=12) | native | 998 | 0 | 0 | 0 |  |
| misikoff_zillow_rentals__me | short (H=12) | native | 868 | 0 | 3 | 0 |  |
| misikoff_zillow_sales__wsat | short (H=8) | native | 57 | 0 | 91 | 0 |  |
| modelpulse_model_pulse_data__d | short (H=30) | native | 484 | 0 | 516 | 0 |  |
| modelpulse_model_pulse_data__wsun | short (H=8) | dl30:mean, dl_all:mean, likes:mean | 651 | 340 | 9 | 0 |  |
| nakas_powderbench_truth__d | short (H=30) | native | 461 | 0 | 539 | 0 |  |
| nakas_powderbench_truth__wsun | short (H=8) | snow24_in:mean, valid:mean | 585 | 0 | 415 | 0 |  |
| narendersingh007_india_solar_benchmark_dataset__h | short (H=48) | native | 1 | 0 | 0 | 0 |  |
| narendersingh007_india_solar_benchmark_dataset__h | medium (H=480) | native | 1 | 0 | 0 | 0 |  |
| narendersingh007_india_solar_benchmark_dataset__h | long (H=720) | native | 1 | 0 | 0 | 0 |  |
| narendersingh007_india_solar_benchmark_dataset__d | short (H=30) | MO:mean, DY:mean, HR:mean, ALLSKY_SFC_SW_DWN:mean, T2M:mean, RH2M:mean, WS10M:mean, PRECTOTCORR:mean, ALLSKY_KT:mean, T2MDEW:mean, PS:mean, WS2M:mean, WD10M:mean, ALLSKY_SFC_LW_DWN:mean, ALLSKY_SFC_PAR_TOT:mean, dayofyear:last, weekday:last, is_weekend:last, SOLAR_ZENITH:mean, SOLAR_AZIMUTH:mean, CLEARSKY_GHI:mean, CLEARSKY_DNI:mean, CLEARSKY_DHI:mean, hour_sin:last, hour_cos:last, month_sin:last, month_cos:last, dayofyear_sin:last, dayofyear_cos:last, TEMP_HUMIDITY:mean, DEWPOINT_SPREAD:mean, WIND_POWER:mean, PRESSURE_TEMP:mean | 1 | 0 | 0 | 0 |  |
| narendersingh007_india_solar_benchmark_dataset__wsun | short (H=8) | MO:mean, DY:mean, HR:mean, ALLSKY_SFC_SW_DWN:mean, T2M:mean, RH2M:mean, WS10M:mean, PRECTOTCORR:mean, ALLSKY_KT:mean, T2MDEW:mean, PS:mean, WS2M:mean, WD10M:mean, ALLSKY_SFC_LW_DWN:mean, ALLSKY_SFC_PAR_TOT:mean, dayofyear:last, weekday:last, is_weekend:last, SOLAR_ZENITH:mean, SOLAR_AZIMUTH:mean, CLEARSKY_GHI:mean, CLEARSKY_DNI:mean, CLEARSKY_DHI:mean, hour_sin:last, hour_cos:last, month_sin:last, month_cos:last, dayofyear_sin:last, dayofyear_cos:last, TEMP_HUMIDITY:mean, DEWPOINT_SPREAD:mean, WIND_POWER:mean, PRESSURE_TEMP:mean | 1 | 0 | 0 | 0 |  |
| national_digital_act_air_quality__h | short (H=48) | native | 0 | 0 | 2 | 0 | no series left after the NaN / scale rules |
| national_digital_act_air_quality__h | medium (H=480) | native | 0 | 0 | 2 | 0 | no series left after the NaN / scale rules |
| national_digital_act_air_quality__h | long (H=720) | native | 0 | 0 | 2 | 0 | no series left after the NaN / scale rules |
| national_digital_act_air_quality__d | short (H=30) | no2:mean, o3_1hr:mean, o3_4hr:mean, o3_8hr:mean, co:mean, pm10_1hr:mean, pm2_5_1hr:mean, pm10:mean, pm2_5:mean, aqi_co:mean, aqi_no2:mean, aqi_o3_1hr:mean, aqi_o3_4hr:mean, aqi_o3_8hr:mean, aqi_pm10:mean, aqi_pm2_5:mean, aqi_site:mean | 0 | 0 | 2 | 0 | no series left after the NaN / scale rules |
| national_digital_act_air_quality__wsun | short (H=8) | no2:mean, o3_1hr:mean, o3_4hr:mean, o3_8hr:mean, co:mean, pm10_1hr:mean, pm2_5_1hr:mean, pm10:mean, pm2_5:mean, aqi_co:mean, aqi_no2:mean, aqi_o3_1hr:mean, aqi_o3_4hr:mean, aqi_o3_8hr:mean, aqi_pm10:mean, aqi_pm2_5:mean, aqi_site:mean | 2 | 0 | 0 | 0 |  |
| national_digital_au_births_by_age_of_mother__ys | short (H=6) | native | 148 | 0 | 0 | 0 |  |
| national_digital_nt_crocodile_captures__ms | short (H=12) | native | 5 | 0 | 0 | 0 |  |
| national_digital_nt_fuel_prices_history__d | short (H=30) | native | 61 | 0 | 4 | 0 |  |
| national_digital_nt_fuel_prices_history__wsun | short (H=8) | diesel:mean, premium_98:mean, premium_95:mean, unleaded_91:mean, premium_diesel:mean, lpg:mean, low_aromatic_fuel:mean, e85:mean, e10:mean, b20:mean | 58 | 4 | 3 | 0 |  |
| national_digital_rba_money_market_daily__d | short (H=30) | native | 0 | 0 | 5 | 0 | no series left after the NaN / scale rules |
| national_digital_rba_money_market_daily__wsun | short (H=8) | target:mean | 0 | 0 | 5 | 0 | no series left after the NaN / scale rules |
| nevirim_wikipedia_attention_13lang__d | short (H=30) | native | 9 | 0 | 4 | 0 |  |
| nevirim_wikipedia_attention_13lang__wsun | short (H=8) | target:mean | 13 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_amarketchina__d | short (H=30) | native | 294 | 0 | 0 | 3 |  |
| nju_zhangsq_rmisc_amarketchina__wsun | short (H=8) | close:mean, open:mean, high:mean, low:mean, vol:mean, amount:mean | 284 | 13 | 0 | 0 |  |
| nju_zhangsq_rmisc_applianceenergy__10min | short (H=48) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_applianceenergy__10min | medium (H=480) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_applianceenergy__10min | long (H=720) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_applianceenergy__h | short (H=48) | Appliances:mean, lights:last, T1:mean, RH_1:mean, T2:mean, RH_2:mean, T3:mean, RH_3:mean, T4:mean, RH_4:mean, T5:mean, RH_5:mean, T6:mean, RH_6:mean, T7:mean, RH_7:mean, T8:mean, RH_8:mean, T9:mean, RH_9:mean, T_out:mean, Press_mm_hg:mean, RH_out:mean, Windspeed:mean, Visibility:mean, Tdewpoint:mean | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_applianceenergy__h | medium (H=480) | Appliances:mean, lights:last, T1:mean, RH_1:mean, T2:mean, RH_2:mean, T3:mean, RH_3:mean, T4:mean, RH_4:mean, T5:mean, RH_5:mean, T6:mean, RH_6:mean, T7:mean, RH_7:mean, T8:mean, RH_8:mean, T9:mean, RH_9:mean, T_out:mean, Press_mm_hg:mean, RH_out:mean, Windspeed:mean, Visibility:mean, Tdewpoint:mean | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_applianceenergy__h | long (H=720) | Appliances:mean, lights:last, T1:mean, RH_1:mean, T2:mean, RH_2:mean, T3:mean, RH_3:mean, T4:mean, RH_4:mean, T5:mean, RH_5:mean, T6:mean, RH_6:mean, T7:mean, RH_7:mean, T8:mean, RH_8:mean, T9:mean, RH_9:mean, T_out:mean, Press_mm_hg:mean, RH_out:mean, Windspeed:mean, Visibility:mean, Tdewpoint:mean | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_britaincoal__5min | short (H=48) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_britaincoal__5min | medium (H=480) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_britaincoal__5min | long (H=720) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_britaincoal__h | short (H=48) |  demand:mean,  frequency:mean,  coal:mean,  nuclear:mean,  ccgt:mean,  wind:mean,  pumped:mean,  hydro:mean,  biomass:mean,  solar:mean | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_britaincoal__h | medium (H=480) |  demand:mean,  frequency:mean,  coal:mean,  nuclear:mean,  ccgt:mean,  wind:mean,  pumped:mean,  hydro:mean,  biomass:mean,  solar:mean | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_britaincoal__h | long (H=720) |  demand:mean,  frequency:mean,  coal:mean,  nuclear:mean,  ccgt:mean,  wind:mean,  pumped:mean,  hydro:mean,  biomass:mean,  solar:mean | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_britaincoal__d | short (H=30) |  demand:mean,  frequency:mean,  coal:mean,  nuclear:mean,  ccgt:mean,  wind:mean,  pumped:mean,  hydro:mean,  biomass:mean,  solar:mean | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_bts__10min | short (H=48) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_bts__10min | medium (H=480) | native | 0 | 0 | 1 | 0 | no series left after the NaN / scale rules |
| nju_zhangsq_rmisc_bts__10min | long (H=720) | native | 0 | 0 | 1 | 0 | no series left after the NaN / scale rules |
| nju_zhangsq_rmisc_bts__h | short (H=48) | Temperature_Setpoint_0:mean, System_Enable_Command_1:last, Proportional_Gain_Parameter_2:last, Max_Temperature_Setpoint_Limit_4:mean, Max_Temperature_Setpoint_Limit_5:mean, Min_Temperature_Setpoint_Limit_6:mean, Min_Temperature_Setpoint_Limit_7:mean, Max_Temperature_Setpoint_Limit_8:last, Min_Temperature_Setpoint_Limit_9:last, Position_Sensor_11:mean, Enable_Status_12:mean, Position_Sensor_13:mean, Operating_Mode_Status_14:last, Cooling_Temperature_Setpoint_15:mean, Air_Temperature_Sensor_16:last, Heating_Temperature_Setpoint_17:mean, Room_Air_Temperature_Setpoint_18:last, System_Status_19:mean, Air_Temperature_Sensor_20:mean, Temperature_Setpoint_21:last, System_Enable_Command_22:last, Proportional_Gain_Parameter_23:last, Max_Temperature_Setpoint_Limit_25:mean, Max_Temperature_Setpoint_Limit_26:mean, Min_Temperature_Setpoint_Limit_27:mean, Min_Temperature_Setpoint_Limit_28:mean, Max_Temperature_Setpoint_Limit_29:last, Min_Temperature_Setpoint_Limit_30:last, Position_Sensor_33:mean, Enable_Status_34:mean, Position_Sensor_35:mean, Operating_Mode_Status_36:last, Cooling_Temperature_Setpoint_37:mean | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_bts__h | medium (H=480) | Temperature_Setpoint_0:mean, System_Enable_Command_1:last, Proportional_Gain_Parameter_2:last, Max_Temperature_Setpoint_Limit_4:mean, Max_Temperature_Setpoint_Limit_5:mean, Min_Temperature_Setpoint_Limit_6:mean, Min_Temperature_Setpoint_Limit_7:mean, Max_Temperature_Setpoint_Limit_8:last, Min_Temperature_Setpoint_Limit_9:last, Position_Sensor_11:mean, Enable_Status_12:mean, Position_Sensor_13:mean, Operating_Mode_Status_14:last, Cooling_Temperature_Setpoint_15:mean, Air_Temperature_Sensor_16:last, Heating_Temperature_Setpoint_17:mean, Room_Air_Temperature_Setpoint_18:last, System_Status_19:mean, Air_Temperature_Sensor_20:mean, Temperature_Setpoint_21:last, System_Enable_Command_22:last, Proportional_Gain_Parameter_23:last, Max_Temperature_Setpoint_Limit_25:mean, Max_Temperature_Setpoint_Limit_26:mean, Min_Temperature_Setpoint_Limit_27:mean, Min_Temperature_Setpoint_Limit_28:mean, Max_Temperature_Setpoint_Limit_29:last, Min_Temperature_Setpoint_Limit_30:last, Position_Sensor_33:mean, Enable_Status_34:mean, Position_Sensor_35:mean, Operating_Mode_Status_36:last, Cooling_Temperature_Setpoint_37:mean | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_bts__h | long (H=720) | Temperature_Setpoint_0:mean, System_Enable_Command_1:last, Proportional_Gain_Parameter_2:last, Max_Temperature_Setpoint_Limit_4:mean, Max_Temperature_Setpoint_Limit_5:mean, Min_Temperature_Setpoint_Limit_6:mean, Min_Temperature_Setpoint_Limit_7:mean, Max_Temperature_Setpoint_Limit_8:last, Min_Temperature_Setpoint_Limit_9:last, Position_Sensor_11:mean, Enable_Status_12:mean, Position_Sensor_13:mean, Operating_Mode_Status_14:last, Cooling_Temperature_Setpoint_15:mean, Air_Temperature_Sensor_16:last, Heating_Temperature_Setpoint_17:mean, Room_Air_Temperature_Setpoint_18:last, System_Status_19:mean, Air_Temperature_Sensor_20:mean, Temperature_Setpoint_21:last, System_Enable_Command_22:last, Proportional_Gain_Parameter_23:last, Max_Temperature_Setpoint_Limit_25:mean, Max_Temperature_Setpoint_Limit_26:mean, Min_Temperature_Setpoint_Limit_27:mean, Min_Temperature_Setpoint_Limit_28:mean, Max_Temperature_Setpoint_Limit_29:last, Min_Temperature_Setpoint_Limit_30:last, Position_Sensor_33:mean, Enable_Status_34:mean, Position_Sensor_35:mean, Operating_Mode_Status_36:last, Cooling_Temperature_Setpoint_37:mean | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_bts__d | short (H=30) | Temperature_Setpoint_0:mean, System_Enable_Command_1:last, Proportional_Gain_Parameter_2:last, Max_Temperature_Setpoint_Limit_4:mean, Max_Temperature_Setpoint_Limit_5:mean, Min_Temperature_Setpoint_Limit_6:mean, Min_Temperature_Setpoint_Limit_7:mean, Max_Temperature_Setpoint_Limit_8:last, Min_Temperature_Setpoint_Limit_9:last, Position_Sensor_11:mean, Enable_Status_12:mean, Position_Sensor_13:mean, Operating_Mode_Status_14:last, Cooling_Temperature_Setpoint_15:mean, Air_Temperature_Sensor_16:last, Heating_Temperature_Setpoint_17:mean, Room_Air_Temperature_Setpoint_18:last, System_Status_19:mean, Air_Temperature_Sensor_20:mean, Temperature_Setpoint_21:last, System_Enable_Command_22:last, Proportional_Gain_Parameter_23:last, Max_Temperature_Setpoint_Limit_25:mean, Max_Temperature_Setpoint_Limit_26:mean, Min_Temperature_Setpoint_Limit_27:mean, Min_Temperature_Setpoint_Limit_28:mean, Max_Temperature_Setpoint_Limit_29:last, Min_Temperature_Setpoint_Limit_30:last, Position_Sensor_33:mean, Enable_Status_34:mean, Position_Sensor_35:mean, Operating_Mode_Status_36:last, Cooling_Temperature_Setpoint_37:mean | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_buildingsbenchrealcsv__h | short (H=48) | native | 8 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_buildingsbenchrealcsv__h | medium (H=480) | native | 8 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_buildingsbenchrealcsv__h | long (H=720) | native | 8 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_buildingsbenchrealcsv__d | short (H=30) | power:mean, Global_active_power:mean, Bear_public_Orville:mean, Bear_education_Lidia:mean, Bear_education_Nanette:mean, Bear_education_Lewis:mean, Bear_science_Alison:mean, Bear_education_Darrell:mean, Bear_education_Thuy:mean, Bear_lodging_Erick:mean, Bear_education_Mara:mean, Bear_education_Bulah:mean, Bear_lodging_Evan:mean, Bear_education_Iris:mean, Bear_education_Rebecca:mean, Bear_education_Herb:mean, Bear_education_Tara:mean, Bear_education_Augusta:mean, Bear_education_Maryjane:mean, Bear_education_Curtis:mean, Bear_education_Arnold:mean, Bear_education_Chana:mean, Bear_education_Sandy:mean, Bear_parking_Bridget:mean, Bear_education_Holly:mean, Bear_education_Gavin:mean, Bear_education_Paola:mean, Bear_education_Santos:mean, Bear_parking_Bruce:mean, Bear_education_Deena:mean, Bear_education_Lorie:mean, Bear_education_Merrill:mean, Bear_education_Oscar:mean | 8 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_dominick__w | short (H=8) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_ebayserver__s | short (H=48) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_ebayserver__s | medium (H=480) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_ebayserver__s | long (H=720) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_era5hourlysinglelevels__h | short (H=48) | native | 41 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_era5hourlysinglelevels__h | medium (H=480) | native | 41 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_era5hourlysinglelevels__h | long (H=720) | native | 41 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_era5hourlysinglelevels__d | short (H=30) | u100:mean, v100:mean, u10:mean, v10:mean, d2m:mean, t2m:mean, msl:mean, skt:mean, sp:mean, ssrd:mean, strd:mean, tp:mean | 41 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_era5hourlysinglelevels__wsun | short (H=8) | u100:mean, v100:mean, u10:mean, v10:mean, d2m:mean, t2m:mean, msl:mean, skt:mean, sp:mean, ssrd:mean, strd:mean, tp:mean | 41 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_frothflotation__h | short (H=48) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_frothflotation__h | medium (H=480) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_frothflotation__h | long (H=720) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_globalclimatechange__ms | short (H=12) | native | 606 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_householdpower__min | short (H=48) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_householdpower__min | medium (H=480) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_householdpower__min | long (H=720) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_householdpower__h | short (H=48) | Global_active_power:mean, Global_reactive_power:mean, Voltage:mean, Global_intensity:mean, Sub_metering_1:mean, Sub_metering_2:mean, Sub_metering_3:mean | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_householdpower__h | medium (H=480) | Global_active_power:mean, Global_reactive_power:mean, Voltage:mean, Global_intensity:mean, Sub_metering_1:mean, Sub_metering_2:mean, Sub_metering_3:mean | 0 | 1 | 0 | 0 | 0 of 1 series long enough (< 75%) |
| nju_zhangsq_rmisc_householdpower__h | long (H=720) | Global_active_power:mean, Global_reactive_power:mean, Voltage:mean, Global_intensity:mean, Sub_metering_1:mean, Sub_metering_2:mean, Sub_metering_3:mean | 0 | 1 | 0 | 0 | 0 of 1 series long enough (< 75%) |
| nju_zhangsq_rmisc_hungarianchickenpoxcases__wmon | short (H=8) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_keplerlightcurves__30min | short (H=48) | native | 998 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_keplerlightcurves__30min | medium (H=480) | native | 998 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_keplerlightcurves__30min | long (H=720) | native | 998 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_keplerlightcurves__h | short (H=48) | f1:mean | 998 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_keplerlightcurves__h | medium (H=480) | f1:mean | 998 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_keplerlightcurves__h | long (H=720) | f1:mean | 998 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_metropt3__10s | short (H=48) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_metropt3__10s | medium (H=480) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_metropt3__10s | long (H=720) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_metrotraffic__h | short (H=48) | native | 0 | 0 | 1 | 0 | no series left after the NaN / scale rules |
| nju_zhangsq_rmisc_metrotraffic__h | medium (H=480) | native | 0 | 0 | 1 | 0 | no series left after the NaN / scale rules |
| nju_zhangsq_rmisc_metrotraffic__h | long (H=720) | native | 0 | 0 | 1 | 0 | no series left after the NaN / scale rules |
| nju_zhangsq_rmisc_metrotraffic__d | short (H=30) | temp:mean, rain_1h:mean, snow_1h:mean, clouds_all:mean, traffic_volume:mean | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_metrotraffic__wsun | short (H=8) | temp:mean, rain_1h:mean, snow_1h:mean, clouds_all:mean, traffic_volume:mean | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_occupancydetection__min | short (H=48) | native | 0 | 0 | 3 | 0 | no series left after the NaN / scale rules |
| nju_zhangsq_rmisc_occupancydetection__min | medium (H=480) | native | 0 | 0 | 3 | 0 | no series left after the NaN / scale rules |
| nju_zhangsq_rmisc_occupancydetection__min | long (H=720) | native | 0 | 0 | 3 | 0 | no series left after the NaN / scale rules |
| nju_zhangsq_rmisc_opsd_household__min | short (H=48) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_opsd_household__min | medium (H=480) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_opsd_household__min | long (H=720) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_opsd_household__h | short (H=48) | DE_KN_residential2_circulation_pump:mean, DE_KN_residential2_dishwasher:mean, DE_KN_residential2_freezer:mean, DE_KN_residential2_grid_import:mean, DE_KN_residential2_washing_machine:mean, DE_KN_residential3_circulation_pump:mean, DE_KN_residential3_dishwasher:mean, DE_KN_residential3_freezer:mean, DE_KN_residential3_grid_export:mean, DE_KN_residential3_grid_import:mean, DE_KN_residential3_pv:mean, DE_KN_residential3_refrigerator:mean, DE_KN_residential3_washing_machine:mean, DE_KN_residential5_dishwasher:mean, DE_KN_residential5_grid_import:mean, DE_KN_residential5_refrigerator:mean, DE_KN_residential5_washing_machine:mean, DE_KN_industrial2_grid_import:mean, DE_KN_industrial2_pv:mean, DE_KN_industrial2_storage_charge:mean, DE_KN_industrial2_storage_decharge:mean, DE_KN_residential6_circulation_pump:mean, DE_KN_residential6_dishwasher:mean, DE_KN_residential6_freezer:mean, DE_KN_residential6_grid_export:mean, DE_KN_residential6_grid_import:mean, DE_KN_residential6_pv:mean, DE_KN_residential6_washing_machine:mean, DE_KN_public2_grid_import:mean, DE_KN_industrial1_grid_import:mean, DE_KN_industrial1_pv_1:mean, DE_KN_industrial1_pv_2:mean, DE_KN_public1_grid_import:mean | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_opsd_household__h | medium (H=480) | DE_KN_residential2_circulation_pump:mean, DE_KN_residential2_dishwasher:mean, DE_KN_residential2_freezer:mean, DE_KN_residential2_grid_import:mean, DE_KN_residential2_washing_machine:mean, DE_KN_residential3_circulation_pump:mean, DE_KN_residential3_dishwasher:mean, DE_KN_residential3_freezer:mean, DE_KN_residential3_grid_export:mean, DE_KN_residential3_grid_import:mean, DE_KN_residential3_pv:mean, DE_KN_residential3_refrigerator:mean, DE_KN_residential3_washing_machine:mean, DE_KN_residential5_dishwasher:mean, DE_KN_residential5_grid_import:mean, DE_KN_residential5_refrigerator:mean, DE_KN_residential5_washing_machine:mean, DE_KN_industrial2_grid_import:mean, DE_KN_industrial2_pv:mean, DE_KN_industrial2_storage_charge:mean, DE_KN_industrial2_storage_decharge:mean, DE_KN_residential6_circulation_pump:mean, DE_KN_residential6_dishwasher:mean, DE_KN_residential6_freezer:mean, DE_KN_residential6_grid_export:mean, DE_KN_residential6_grid_import:mean, DE_KN_residential6_pv:mean, DE_KN_residential6_washing_machine:mean, DE_KN_public2_grid_import:mean, DE_KN_industrial1_grid_import:mean, DE_KN_industrial1_pv_1:mean, DE_KN_industrial1_pv_2:mean, DE_KN_public1_grid_import:mean | 0 | 1 | 0 | 0 | 0 of 1 series long enough (< 75%) |
| nju_zhangsq_rmisc_opsd_household__h | long (H=720) | DE_KN_residential2_circulation_pump:mean, DE_KN_residential2_dishwasher:mean, DE_KN_residential2_freezer:mean, DE_KN_residential2_grid_import:mean, DE_KN_residential2_washing_machine:mean, DE_KN_residential3_circulation_pump:mean, DE_KN_residential3_dishwasher:mean, DE_KN_residential3_freezer:mean, DE_KN_residential3_grid_export:mean, DE_KN_residential3_grid_import:mean, DE_KN_residential3_pv:mean, DE_KN_residential3_refrigerator:mean, DE_KN_residential3_washing_machine:mean, DE_KN_residential5_dishwasher:mean, DE_KN_residential5_grid_import:mean, DE_KN_residential5_refrigerator:mean, DE_KN_residential5_washing_machine:mean, DE_KN_industrial2_grid_import:mean, DE_KN_industrial2_pv:mean, DE_KN_industrial2_storage_charge:mean, DE_KN_industrial2_storage_decharge:mean, DE_KN_residential6_circulation_pump:mean, DE_KN_residential6_dishwasher:mean, DE_KN_residential6_freezer:mean, DE_KN_residential6_grid_export:mean, DE_KN_residential6_grid_import:mean, DE_KN_residential6_pv:mean, DE_KN_residential6_washing_machine:mean, DE_KN_public2_grid_import:mean, DE_KN_industrial1_grid_import:mean, DE_KN_industrial1_pv_1:mean, DE_KN_industrial1_pv_2:mean, DE_KN_public1_grid_import:mean | 0 | 1 | 0 | 0 | 0 of 1 series long enough (< 75%) |
| nju_zhangsq_rmisc_opsd_when2heat__h | short (H=48) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_opsd_when2heat__h | medium (H=480) | native | 0 | 0 | 1 | 0 | no series left after the NaN / scale rules |
| nju_zhangsq_rmisc_opsd_when2heat__h | long (H=720) | native | 0 | 0 | 1 | 0 | no series left after the NaN / scale rules |
| nju_zhangsq_rmisc_opsd_when2heat__d | short (H=30) | LV_COP_ASHP_floor:mean, LV_COP_ASHP_radiator:mean, LV_COP_ASHP_water:mean, LV_COP_GSHP_floor:mean, LV_COP_GSHP_radiator:mean, LV_COP_GSHP_water:mean, LV_COP_WSHP_floor:mean, LV_COP_WSHP_radiator:mean, LV_heat_demand_space:mean, LV_heat_demand_space_COM:mean, LV_heat_demand_space_MFH:mean, LV_heat_demand_space_SFH:mean, LV_heat_demand_total:mean, LV_heat_demand_water:mean, LV_heat_demand_water_COM:mean, LV_heat_demand_water_MFH:mean, LV_heat_demand_water_SFH:mean, LV_heat_profile_space_COM:mean, LV_heat_profile_space_MFH:mean, LV_heat_profile_space_SFH:mean, LV_heat_profile_water_COM:mean, LV_heat_profile_water_MFH:mean, LV_heat_profile_water_SFH:mean, NL_COP_ASHP_floor:mean, NL_COP_ASHP_radiator:mean, NL_COP_ASHP_water:mean, NL_COP_GSHP_floor:mean, NL_COP_GSHP_radiator:mean, NL_COP_GSHP_water:mean, NL_COP_WSHP_floor:mean, NL_COP_WSHP_radiator:mean, NL_heat_demand_space:mean, NL_heat_demand_space_COM:mean | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_opsd_when2heat__wsun | short (H=8) | LV_COP_ASHP_floor:mean, LV_COP_ASHP_radiator:mean, LV_COP_ASHP_water:mean, LV_COP_GSHP_floor:mean, LV_COP_GSHP_radiator:mean, LV_COP_GSHP_water:mean, LV_COP_WSHP_floor:mean, LV_COP_WSHP_radiator:mean, LV_heat_demand_space:mean, LV_heat_demand_space_COM:mean, LV_heat_demand_space_MFH:mean, LV_heat_demand_space_SFH:mean, LV_heat_demand_total:mean, LV_heat_demand_water:mean, LV_heat_demand_water_COM:mean, LV_heat_demand_water_MFH:mean, LV_heat_demand_water_SFH:mean, LV_heat_profile_space_COM:mean, LV_heat_profile_space_MFH:mean, LV_heat_profile_space_SFH:mean, LV_heat_profile_water_COM:mean, LV_heat_profile_water_MFH:mean, LV_heat_profile_water_SFH:mean, NL_COP_ASHP_floor:mean, NL_COP_ASHP_radiator:mean, NL_COP_ASHP_water:mean, NL_COP_GSHP_floor:mean, NL_COP_GSHP_radiator:mean, NL_COP_GSHP_water:mean, NL_COP_WSHP_floor:mean, NL_COP_WSHP_radiator:mean, NL_heat_demand_space:mean, NL_heat_demand_space_COM:mean | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_pm25fivecities__h | short (H=48) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_pm25fivecities__h | medium (H=480) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_pm25fivecities__h | long (H=720) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_pm25fivecities__d | short (H=30) | PM_City Station:mean, PM_5th Middle School:mean, PM_US Post:mean, DEWP:mean, HUMI:mean, PRES:mean, TEMP:mean, Iws:mean, precipitation:mean, Iprec:mean, PM_Jingan:mean, PM_Xuhui:mean, PM_Caotangsi:mean, PM_Shahepu:mean, PM_Taiyuanjie:mean, PM_Xiaoheyan:mean, PM_Dongsi:mean, PM_Dongsihuan:mean, PM_Nongzhanguan:mean | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_pm25fivecities__wsun | short (H=8) | PM_City Station:mean, PM_5th Middle School:mean, PM_US Post:mean, DEWP:mean, HUMI:mean, PRES:mean, TEMP:mean, Iws:mean, precipitation:mean, Iprec:mean, PM_Jingan:mean, PM_Xuhui:mean, PM_Caotangsi:mean, PM_Shahepu:mean, PM_Taiyuanjie:mean, PM_Xiaoheyan:mean, PM_Dongsi:mean, PM_Dongsihuan:mean, PM_Nongzhanguan:mean | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_pump__min | short (H=48) | native | 2 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_pump__min | medium (H=480) | native | 2 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_pump__min | long (H=720) | native | 2 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_pump__h | short (H=48) | sensor0:mean, sensor1:mean, sensor2:mean, sensor3:mean, sensor4:mean, sensor5:mean, sensor6:mean, sensor7:mean, sensor8:mean, sensor9:mean, sensor10:mean, sensor11:mean, sensor12:mean, sensor13:mean, sensor14:mean, sensor15:mean, sensor16:mean, sensor17:mean, sensor18:mean, sensor19:mean, sensor20:mean, sensor21:mean, sensor22:mean, sensor23:mean, sensor24:mean, sensor25:mean, sensor26:mean, sensor27:mean, sensor28:mean, sensor29:mean, sensor30:mean, sensor31:mean, sensor32:mean | 2 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_pump__h | medium (H=480) | sensor0:mean, sensor1:mean, sensor2:mean, sensor3:mean, sensor4:mean, sensor5:mean, sensor6:mean, sensor7:mean, sensor8:mean, sensor9:mean, sensor10:mean, sensor11:mean, sensor12:mean, sensor13:mean, sensor14:mean, sensor15:mean, sensor16:mean, sensor17:mean, sensor18:mean, sensor19:mean, sensor20:mean, sensor21:mean, sensor22:mean, sensor23:mean, sensor24:mean, sensor25:mean, sensor26:mean, sensor27:mean, sensor28:mean, sensor29:mean, sensor30:mean, sensor31:mean, sensor32:mean | 0 | 2 | 0 | 0 | 0 of 2 series long enough (< 75%) |
| nju_zhangsq_rmisc_pump__h | long (H=720) | sensor0:mean, sensor1:mean, sensor2:mean, sensor3:mean, sensor4:mean, sensor5:mean, sensor6:mean, sensor7:mean, sensor8:mean, sensor9:mean, sensor10:mean, sensor11:mean, sensor12:mean, sensor13:mean, sensor14:mean, sensor15:mean, sensor16:mean, sensor17:mean, sensor18:mean, sensor19:mean, sensor20:mean, sensor21:mean, sensor22:mean, sensor23:mean, sensor24:mean, sensor25:mean, sensor26:mean, sensor27:mean, sensor28:mean, sensor29:mean, sensor30:mean, sensor31:mean, sensor32:mean | 0 | 2 | 0 | 0 | 0 of 2 series long enough (< 75%) |
| nju_zhangsq_rmisc_roomoccupancy__30s | short (H=48) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_roomoccupancy__30s | medium (H=480) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_roomoccupancy__30s | long (H=720) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_servermachinedataset__min | short (H=48) | native | 28 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_servermachinedataset__min | medium (H=480) | native | 28 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_servermachinedataset__min | long (H=720) | native | 28 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_servermachinedataset__h | short (H=48) | f1:mean, f2:mean, f3:mean, f4:mean, f6:mean, f7:mean, f9:mean, f10:mean, f11:mean, f12:mean, f13:mean, f14:mean, f15:mean, f16:mean, f19:mean, f20:mean, f21:mean, f22:mean, f23:mean, f24:mean, f25:mean, f26:mean, f28:mean, f29:mean, f30:mean, f31:mean, f32:mean, f33:mean, f34:mean, f35:mean, f36:mean, f17:mean, f18:mean | 28 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_servermachinedataset__h | medium (H=480) | f1:mean, f2:mean, f3:mean, f4:mean, f6:mean, f7:mean, f9:mean, f10:mean, f11:mean, f12:mean, f13:mean, f14:mean, f15:mean, f16:mean, f19:mean, f20:mean, f21:mean, f22:mean, f23:mean, f24:mean, f25:mean, f26:mean, f28:mean, f29:mean, f30:mean, f31:mean, f32:mean, f33:mean, f34:mean, f35:mean, f36:mean, f17:mean, f18:mean | 0 | 28 | 0 | 0 | 0 of 28 series long enough (< 75%) |
| nju_zhangsq_rmisc_servermachinedataset__h | long (H=720) | f1:mean, f2:mean, f3:mean, f4:mean, f6:mean, f7:mean, f9:mean, f10:mean, f11:mean, f12:mean, f13:mean, f14:mean, f15:mean, f16:mean, f19:mean, f20:mean, f21:mean, f22:mean, f23:mean, f24:mean, f25:mean, f26:mean, f28:mean, f29:mean, f30:mean, f31:mean, f32:mean, f33:mean, f34:mean, f35:mean, f36:mean, f17:mean, f18:mean | 0 | 28 | 0 | 0 | 0 of 28 series long enough (< 75%) |
| nju_zhangsq_rmisc_smartmeteraus30m__30min | short (H=48) | native | 908 | 0 | 0 | 2 |  |
| nju_zhangsq_rmisc_smartmeteraus30m__30min__medium | medium (H=480) | native | 906 | 3 | 0 | 1 |  |
| nju_zhangsq_rmisc_smartmeteraus30m__30min__medium | long (H=720) | native | 906 | 3 | 0 | 1 |  |
| nju_zhangsq_rmisc_smartmeteraus30m__h | short (H=48) | target:mean | 906 | 0 | 0 | 4 |  |
| nju_zhangsq_rmisc_smartmeteraus30m__h__medium | medium (H=480) | target:mean | 905 | 4 | 0 | 1 |  |
| nju_zhangsq_rmisc_smartmeteraus30m__h__medium | long (H=720) | target:mean | 905 | 5 | 0 | 0 |  |
| nju_zhangsq_rmisc_smartmeteraus30m__d | short (H=30) | target:mean | 901 | 9 | 0 | 0 |  |
| nju_zhangsq_rmisc_smartmeteraus30m__wsun | short (H=8) | target:mean | 615 | 295 | 0 | 0 |  |
| nju_zhangsq_rmisc_smartmeteraus60m__h | short (H=48) | native | 996 | 0 | 0 | 3 |  |
| nju_zhangsq_rmisc_smartmeteraus60m__h__medium | medium (H=480) | native | 994 | 4 | 0 | 1 |  |
| nju_zhangsq_rmisc_smartmeteraus60m__h__long | long (H=720) | native | 993 | 6 | 0 | 0 |  |
| nju_zhangsq_rmisc_smartmeteraus60m__d | short (H=30) | target:mean | 989 | 10 | 0 | 0 |  |
| nju_zhangsq_rmisc_smartmeteraus60m__wsun | short (H=8) | target:mean | 688 | 311 | 0 | 0 |  |
| nju_zhangsq_rmisc_smartmeteruk30m__30min | short (H=48) | native | 991 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_smartmeteruk30m__30min__medium | medium (H=480) | native | 990 | 0 | 0 | 1 |  |
| nju_zhangsq_rmisc_smartmeteruk30m__30min__medium | long (H=720) | native | 990 | 1 | 0 | 0 |  |
| nju_zhangsq_rmisc_smartmeteruk30m__h | short (H=48) | target:mean | 990 | 0 | 0 | 1 |  |
| nju_zhangsq_rmisc_smartmeteruk30m__h | medium (H=480) | target:mean | 990 | 1 | 0 | 0 |  |
| nju_zhangsq_rmisc_smartmeteruk30m__h | long (H=720) | target:mean | 990 | 1 | 0 | 0 |  |
| nju_zhangsq_rmisc_smartmeteruk30m__d | short (H=30) | target:mean | 986 | 5 | 0 | 0 |  |
| nju_zhangsq_rmisc_smartmeteruk30m__wsun | short (H=8) | target:mean | 933 | 58 | 0 | 0 |  |
| nju_zhangsq_rmisc_smartmeteruk60m__h | short (H=48) | native | 998 | 0 | 0 | 1 |  |
| nju_zhangsq_rmisc_smartmeteruk60m__h | medium (H=480) | native | 998 | 1 | 0 | 0 |  |
| nju_zhangsq_rmisc_smartmeteruk60m__h__long | long (H=720) | native | 996 | 3 | 0 | 0 |  |
| nju_zhangsq_rmisc_smartmeteruk60m__d | short (H=30) | target:mean | 993 | 6 | 0 | 0 |  |
| nju_zhangsq_rmisc_smartmeteruk60m__wsun | short (H=8) | target:mean | 952 | 47 | 0 | 0 |  |
| nju_zhangsq_rmisc_sp500__d | short (H=30) | native | 496 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_sp500__wsun | short (H=8) | Close:mean, Open:mean, High:mean, Low:mean, Volume:mean | 490 | 6 | 0 | 0 |  |
| nju_zhangsq_rmisc_stockfactorscleaned__d | short (H=30) | native | 0 | 0 | 1000 | 0 | no series left after the NaN / scale rules |
| nju_zhangsq_rmisc_stockfactorscleaned__wsun | short (H=8) | target_value:mean, tech_open:mean, tech_high:mean, tech_low:mean, tech_close:mean, tech_volume:mean, tech_return_1d:mean, tech_gap_return_1d:mean, tech_return_5d:mean, tech_intraday_range:mean, tech_price_range:mean, tech_sma_5:mean, tech_sma_10:mean, tech_sma_20:mean, tech_ema_10:mean, tech_ema_20:mean, tech_dema_10:mean, tech_tema_10:mean, tech_mom_10:mean, tech_roc_10:mean, tech_rsi_14:mean, tech_cmo_14:mean, tech_macd:mean, tech_macd_hist:mean, tech_stoch_k:mean, tech_stoch_d:mean, tech_atr_14:mean, tech_std_5:mean, tech_std_10:mean, tech_bb_upper:mean, tech_bb_lower:mean, tech_bb_width:mean, tech_obv:mean | 0 | 37 | 963 | 0 | no series left after the NaN / scale rules |
| nju_zhangsq_rmisc_stockmarketdata__d | short (H=30) | native | 5 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_stockmarketdata__wsun | short (H=8) | Close:mean, Volume:mean, mom:mean, mom1:mean, mom2:mean, mom3:mean, ROC_5:mean, ROC_10:mean, ROC_15:mean, ROC_20:mean, EMA_10:mean, EMA_20:mean, DTB4WK:mean, DTB3:mean, DTB6:mean, DGS5:mean, DGS10:mean, Oil:mean, Gold:mean, DAAA:mean, DBAA:mean, AAPL:mean, AMZN:mean, GE:mean, JNJ:mean, JPM:mean, MSFT:mean, WFC:mean, XOM:mean, FCHI:mean, DJI:mean, IXIC:mean, S&P:mean | 5 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_tetuanpowerconsumption__10min | short (H=48) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_tetuanpowerconsumption__10min | medium (H=480) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_tetuanpowerconsumption__10min | long (H=720) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_tetuanpowerconsumption__h | short (H=48) | Temperature:mean, Humidity:mean, Wind Speed:mean, general diffuse flows:mean, diffuse flows:mean, Zone 1 Power Consumption:mean, Zone 2  Power Consumption:mean, Zone 3  Power Consumption:mean | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_tetuanpowerconsumption__h | medium (H=480) | Temperature:mean, Humidity:mean, Wind Speed:mean, general diffuse flows:mean, diffuse flows:mean, Zone 1 Power Consumption:mean, Zone 2  Power Consumption:mean, Zone 3  Power Consumption:mean | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_tetuanpowerconsumption__h | long (H=720) | Temperature:mean, Humidity:mean, Wind Speed:mean, general diffuse flows:mean, diffuse flows:mean, Zone 1 Power Consumption:mean, Zone 2  Power Consumption:mean, Zone 3  Power Consumption:mean | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_tetuanpowerconsumption__d | short (H=30) | Temperature:mean, Humidity:mean, Wind Speed:mean, general diffuse flows:mean, diffuse flows:mean, Zone 1 Power Consumption:mean, Zone 2  Power Consumption:mean, Zone 3  Power Consumption:mean | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_timemmd__d | short (H=30) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_timemmd__wsun | short (H=8) | OT:mean, Exports:mean, Imports:mean, D0:mean, D1:mean, D2:mean, D3:mean, D4:mean, Weekly East Coast All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon):mean, Weekly New England (PADD 1A) All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon):mean, Weekly Central Atlantic (PADD 1B) All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon):mean, Weekly Lower Atlantic (PADD 1C) All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon):mean, Weekly Midwest All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon):mean, Weekly Gulf Coast All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon):mean, Weekly Rocky Mountain All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon):mean, Weekly West Coast All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon):mean, Number of Sites Reporting:mean, % WEIGHTED ILI:mean, AGE 0-4:mean, AGE 5-24:mean, AGE 65:mean, ILITOTAL:mean, NUM. OF PROVIDERS:mean, TOTAL PATIENTS:mean | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_tushareetfdaily__d | short (H=30) | native | 1000 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_tushareetfdaily__wsun | short (H=8) | count:sum | 932 | 68 | 0 | 0 |  |
| nju_zhangsq_rmisc_tushareindexdaily__d | short (H=30) | native | 164 | 0 | 0 | 5 |  |
| nju_zhangsq_rmisc_tushareindexdaily__wsun | short (H=8) | count:sum | 162 | 6 | 0 | 1 |  |
| nju_zhangsq_rmisc_tusharestockdaily__d | short (H=30) | native | 1000 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_tusharestockdaily__wsun | short (H=8) | count:sum | 984 | 16 | 0 | 0 |  |
| nju_zhangsq_rmisc_tusharestockweekly__d | short (H=30) | native | 998 | 0 | 0 | 2 |  |
| nju_zhangsq_rmisc_tusharestockweekly__wsun | short (H=8) | count:sum | 982 | 18 | 0 | 0 |  |
| nju_zhangsq_rmisc_wadi__5s | short (H=48) | native | 2 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_wadi__5s | medium (H=480) | native | 2 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_wadi__5s | long (H=720) | native | 2 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_weeklyfuelpricesitaly__wmon | short (H=8) | native | 6 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_weeklyroadfuelprices__wmon | short (H=8) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_windelec__15min | short (H=48) | native | 5 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_windelec__15min | medium (H=480) | native | 5 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_windelec__15min | long (H=720) | native | 5 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_windelec__h | short (H=48) | target:mean, cov2:mean, cov3:mean, cov4:mean, cov5:mean, cov6:mean, cov7:mean, cov8:mean, cov9:mean, cov10:mean, cov11:mean, cov12:mean, cov13:mean | 5 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_windelec__h | medium (H=480) | target:mean, cov2:mean, cov3:mean, cov4:mean, cov5:mean, cov6:mean, cov7:mean, cov8:mean, cov9:mean, cov10:mean, cov11:mean, cov12:mean, cov13:mean | 5 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_windelec__h | long (H=720) | target:mean, cov2:mean, cov3:mean, cov4:mean, cov5:mean, cov6:mean, cov7:mean, cov8:mean, cov9:mean, cov10:mean, cov11:mean, cov12:mean, cov13:mean | 5 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_windelec__d | short (H=30) | target:mean, cov2:mean, cov3:mean, cov4:mean, cov5:mean, cov6:mean, cov7:mean, cov8:mean, cov9:mean, cov10:mean, cov11:mean, cov12:mean, cov13:mean | 5 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_windelec__wsun | short (H=8) | target:mean, cov2:mean, cov3:mean, cov4:mean, cov5:mean, cov6:mean, cov7:mean, cov8:mean, cov9:mean, cov10:mean, cov11:mean, cov12:mean, cov13:mean | 5 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_windpower4secs__4s | short (H=48) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_windpower4secs__4s | medium (H=480) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_windpower4secs__4s | long (H=720) | native | 1 | 0 | 0 | 0 |  |
| nju_zhangsq_rmisc_xiamenairquality__h | short (H=48) | native | 107 | 0 | 79 | 0 |  |
| nju_zhangsq_rmisc_xiamenairquality__h__medium | medium (H=480) | native | 120 | 3 | 63 | 0 |  |
| nju_zhangsq_rmisc_xiamenairquality__h__long | long (H=720) | native | 107 | 3 | 76 | 0 |  |
| nju_zhangsq_rmisc_xiamenairquality__d | short (H=30) | SO2浓度（μg/m3）:mean, NO2浓度（μg/m3）:mean, PM10的1小时均值（μg/m3）:mean, CO浓度（mg/m3）:mean, O3 的1小时平均浓度（μg/m3）:mean, PM2.5的1小时平均浓度（μg/m3）:mean | 150 | 7 | 29 | 0 |  |
| notadib_usa_corn_belt_crop_yield__ys | short (H=6) | native | 285 | 0 | 205 | 0 |  |
| numix_stations_officielles_es__10min | short (H=48) | native | 982 | 0 | 12 | 0 |  |
| numix_stations_officielles_es__10min__medium | medium (H=480) | native | 980 | 1 | 13 | 0 |  |
| numix_stations_officielles_es__10min__long | long (H=720) | native | 976 | 2 | 16 | 0 |  |
| numix_stations_officielles_es__h | short (H=48) | target:mean | 987 | 1 | 6 | 0 |  |
| numix_stations_officielles_es__h | medium (H=480) | target:mean | 0 | 993 | 0 | 0 | 0 of 993 series long enough (< 75%) |
| numix_stations_officielles_es__h | long (H=720) | target:mean | 0 | 993 | 0 | 0 | 0 of 993 series long enough (< 75%) |
| olaflaitinen_baku_metro_station_flow__d | short (H=30) | native | 27 | 0 | 0 | 0 |  |
| open_predico_benchmark__15min | short (H=48) | native | 125 | 0 | 1 | 0 |  |
| open_predico_benchmark__15min | medium (H=480) | native | 125 | 0 | 1 | 0 |  |
| open_predico_benchmark__15min | long (H=720) | native | 125 | 0 | 1 | 0 |  |
| open_predico_benchmark__h | short (H=48) | target:mean | 125 | 0 | 1 | 0 |  |
| open_predico_benchmark__h | medium (H=480) | target:mean | 125 | 0 | 1 | 0 |  |
| open_predico_benchmark__h | long (H=720) | target:mean | 125 | 0 | 1 | 0 |  |
| open_predico_benchmark__d | short (H=30) | target:mean | 126 | 0 | 0 | 0 |  |
| openstef_liander2024_energy_forecasting_benchmark_load_measurements_weather_forecasts__15min | short (H=48) | native | 576 | 0 | 0 | 0 |  |
| openstef_liander2024_energy_forecasting_benchmark_load_measurements_weather_forecasts__15min | medium (H=480) | native | 576 | 0 | 0 | 0 |  |
| openstef_liander2024_energy_forecasting_benchmark_load_measurements_weather_forecasts__15min | long (H=720) | native | 576 | 0 | 0 | 0 |  |
| openstef_liander2024_energy_forecasting_benchmark_load_measurements_weather_forecasts__h | short (H=48) | target:mean | 576 | 0 | 0 | 0 |  |
| openstef_liander2024_energy_forecasting_benchmark_load_measurements_weather_forecasts__h | medium (H=480) | target:mean | 576 | 0 | 0 | 0 |  |
| openstef_liander2024_energy_forecasting_benchmark_load_measurements_weather_forecasts__h | long (H=720) | target:mean | 576 | 0 | 0 | 0 |  |
| openstef_liander2024_energy_forecasting_benchmark_load_measurements_weather_forecasts__d | short (H=30) | target:mean | 576 | 0 | 0 | 0 |  |
| opensynth_tudelft_electricity_consumption_1_0_default__30min | short (H=48) | native | 924 | 0 | 66 | 0 |  |
| opensynth_tudelft_electricity_consumption_1_0_default__30min__medium | medium (H=480) | native | 757 | 0 | 233 | 0 |  |
| opensynth_tudelft_electricity_consumption_1_0_default__30min__long | long (H=720) | native | 784 | 0 | 206 | 0 |  |
| opensynth_tudelft_electricity_consumption_1_0_default__h | short (H=48) | target:mean | 791 | 1 | 198 | 0 |  |
| opensynth_tudelft_electricity_consumption_1_0_default__h | medium (H=480) | target:mean | 791 | 1 | 197 | 0 |  |
| opensynth_tudelft_electricity_consumption_1_0_default__h__long | long (H=720) | target:mean | 909 | 1 | 79 | 0 |  |
| opensynth_tudelft_electricity_consumption_1_0_default__d | short (H=30) | target:mean | 980 | 3 | 7 | 0 |  |
| opensynth_tudelft_electricity_consumption_1_0_default__wsun | short (H=8) | target:mean | 945 | 43 | 2 | 0 |  |
| orailix_ride_silver__d | short (H=30) | native | 1000 | 0 | 0 | 0 |  |
| orailix_ride_silver__wsun | short (H=8) | count:sum | 793 | 176 | 0 | 31 |  |
| pahzuzu_pah_env_001_serc_soil_signals__30min | short (H=48) | native | 0 | 0 | 33 | 0 | no series left after the NaN / scale rules |
| pahzuzu_pah_env_001_serc_soil_signals__30min | medium (H=480) | native | 0 | 0 | 33 | 0 | no series left after the NaN / scale rules |
| pahzuzu_pah_env_001_serc_soil_signals__30min | long (H=720) | native | 0 | 0 | 33 | 0 | no series left after the NaN / scale rules |
| pahzuzu_pah_env_001_serc_soil_signals__h | short (H=48) | vswc:mean, vswc_uncertainty:mean, vswc_flag:last, vsic:mean, vsic_uncertainty:mean, vsic_flag:last, depth_cm:last | 0 | 0 | 33 | 0 | no series left after the NaN / scale rules |
| pahzuzu_pah_env_001_serc_soil_signals__h | medium (H=480) | vswc:mean, vswc_uncertainty:mean, vswc_flag:last, vsic:mean, vsic_uncertainty:mean, vsic_flag:last, depth_cm:last | 0 | 0 | 33 | 0 | no series left after the NaN / scale rules |
| pahzuzu_pah_env_001_serc_soil_signals__h | long (H=720) | vswc:mean, vswc_uncertainty:mean, vswc_flag:last, vsic:mean, vsic_uncertainty:mean, vsic_flag:last, depth_cm:last | 0 | 0 | 33 | 0 | no series left after the NaN / scale rules |
| pahzuzu_pah_env_001_serc_soil_signals__d | short (H=30) | vswc:mean, vswc_uncertainty:mean, vswc_flag:last, vsic:mean, vsic_uncertainty:mean, vsic_flag:last, depth_cm:last | 1 | 0 | 32 | 0 |  |
| pahzuzu_pah_env_001_serc_soil_signals__wsun | short (H=8) | vswc:mean, vswc_uncertainty:mean, vswc_flag:last, vsic:mean, vsic_uncertainty:mean, vsic_flag:last, depth_cm:last | 22 | 0 | 11 | 0 |  |
| phanerozoic_qiskit_calibration_drift__d | short (H=30) | native | 1 | 0 | 0 | 0 |  |
| phanerozoic_ww3_ndbc_pairs__h | short (H=48) | native | 49 | 0 | 125 | 0 |  |
| phanerozoic_ww3_ndbc_pairs__h | medium (H=480) | native | 49 | 0 | 125 | 0 |  |
| phanerozoic_ww3_ndbc_pairs__h__long | long (H=720) | native | 62 | 1 | 111 | 0 |  |
| phanerozoic_ww3_ndbc_pairs__d | short (H=30) | m_hs:mean, m_wspd:mean, m_wdir:mean, p1_hs:mean, p1_tp:mean, p1_dir:mean, p2_hs:mean, p2_tp:mean, p2_dir:mean, p3_hs:mean, p3_tp:mean, p3_dir:mean, o_hs:mean, o_dpd:mean, o_mwd:mean, month:last, hour:last | 139 | 3 | 32 | 0 |  |
| phanerozoic_ww3_ndbc_pairs__wsun | short (H=8) | m_hs:mean, m_wspd:mean, m_wdir:mean, p1_hs:mean, p1_tp:mean, p1_dir:mean, p2_hs:mean, p2_tp:mean, p2_dir:mean, p3_hs:mean, p3_tp:mean, p3_dir:mean, o_hs:mean, o_dpd:mean, o_mwd:mean, month:last, hour:last | 137 | 13 | 24 | 0 |  |
| powerzoojax_powerzoodataset_ausgrid_zone_substation__15min | short (H=48) | native | 174 | 0 | 1 | 0 |  |
| powerzoojax_powerzoodataset_ausgrid_zone_substation__15min | medium (H=480) | native | 0 | 0 | 175 | 0 | no series left after the NaN / scale rules |
| powerzoojax_powerzoodataset_ausgrid_zone_substation__15min | long (H=720) | native | 0 | 0 | 175 | 0 | no series left after the NaN / scale rules |
| powerzoojax_powerzoodataset_ausgrid_zone_substation__h | short (H=48) | load_mw:mean, fiscal_year:last | 0 | 0 | 175 | 0 | no series left after the NaN / scale rules |
| powerzoojax_powerzoodataset_ausgrid_zone_substation__h__medium | medium (H=480) | load_mw:mean, fiscal_year:last | 174 | 0 | 1 | 0 |  |
| powerzoojax_powerzoodataset_ausgrid_zone_substation__h | long (H=720) | load_mw:mean, fiscal_year:last | 0 | 0 | 175 | 0 | no series left after the NaN / scale rules |
| powerzoojax_powerzoodataset_ausgrid_zone_substation__d | short (H=30) | load_mw:mean, fiscal_year:last | 174 | 0 | 1 | 0 |  |
| powerzoojax_powerzoodataset_gb_neso_demand_gb_gen_by_type_gb_market_mid_gb_forecast_actual_demand__30min | short (H=48) | native | 10 | 0 | 0 | 0 |  |
| powerzoojax_powerzoodataset_gb_neso_demand_gb_gen_by_type_gb_market_mid_gb_forecast_actual_demand__30min | medium (H=480) | native | 10 | 0 | 0 | 0 |  |
| powerzoojax_powerzoodataset_gb_neso_demand_gb_gen_by_type_gb_market_mid_gb_forecast_actual_demand__30min | long (H=720) | native | 10 | 0 | 0 | 0 |  |
| powerzoojax_powerzoodataset_gb_neso_demand_gb_gen_by_type_gb_market_mid_gb_forecast_actual_demand__h | short (H=48) | target:mean | 10 | 0 | 0 | 0 |  |
| powerzoojax_powerzoodataset_gb_neso_demand_gb_gen_by_type_gb_market_mid_gb_forecast_actual_demand__h | medium (H=480) | target:mean | 10 | 0 | 0 | 0 |  |
| powerzoojax_powerzoodataset_gb_neso_demand_gb_gen_by_type_gb_market_mid_gb_forecast_actual_demand__h | long (H=720) | target:mean | 10 | 0 | 0 | 0 |  |
| powerzoojax_powerzoodataset_gb_neso_demand_gb_gen_by_type_gb_market_mid_gb_forecast_actual_demand__d | short (H=30) | target:mean | 10 | 0 | 0 | 0 |  |
| powerzoojax_powerzoodataset_gb_neso_demand_gb_gen_by_type_gb_market_mid_gb_forecast_actual_demand__wsun | short (H=8) | target:mean | 10 | 0 | 0 | 0 |  |
| pracadata_brazil_fuel_prices__d | short (H=30) | native | 1000 | 0 | 0 | 0 |  |
| pracadata_brazil_fuel_prices__wsun | short (H=8) | count:sum | 928 | 72 | 0 | 0 |  |
| prasad_gade05_india_upi_ecosystem_2018_2025__qs | short (H=8) | native | 36 | 0 | 0 | 0 |  |
| previdengue_predict_climate_data__wsat | short (H=8) | native | 1000 | 0 | 0 | 0 |  |
| project_agml_usda_nass_crop_statistics__ys | short (H=6) | native | 374 | 0 | 626 | 0 |  |
| prospex_ch_swiss_commercial_registry_monthly_panel__d | short (H=30) | native | 1 | 0 | 0 | 0 |  |
| prospex_ch_swiss_commercial_registry_monthly_panel__wsun | short (H=8) | count:sum | 1 | 0 | 0 | 0 |  |
| rebase_energy_heftcom2024_default__6h | short (H=48) | native | 0 | 0 | 5 | 0 | no series left after the NaN / scale rules |
| rebase_energy_heftcom2024_default__6h | medium (H=480) | native | 0 | 0 | 5 | 0 | no series left after the NaN / scale rules |
| rebase_energy_heftcom2024_default__6h | long (H=720) | native | 0 | 0 | 5 | 0 | no series left after the NaN / scale rules |
| rebase_energy_heftcom2024_default__d | short (H=30) | target:mean | 0 | 0 | 5 | 0 | no series left after the NaN / scale rules |
| rebase_energy_heftcom2024_default__wsun | short (H=8) | target:mean | 5 | 0 | 0 | 0 |  |
| rebase_energy_nve_windpower_data_nve_windpower_timeseries_csv__h | short (H=48) | native | 64 | 0 | 0 | 0 |  |
| rebase_energy_nve_windpower_data_nve_windpower_timeseries_csv__h | medium (H=480) | native | 64 | 0 | 0 | 0 |  |
| rebase_energy_nve_windpower_data_nve_windpower_timeseries_csv__h | long (H=720) | native | 64 | 0 | 0 | 0 |  |
| rebase_energy_nve_windpower_data_nve_windpower_timeseries_csv__d | short (H=30) | target:mean | 64 | 0 | 0 | 0 |  |
| rebase_energy_nve_windpower_data_nve_windpower_timeseries_csv__wsun | short (H=8) | target:mean | 64 | 0 | 0 | 0 |  |
| rif_sql_uk_electricity_generation_mix_2019_2026__30min | short (H=48) | native | 287 | 0 | 0 | 0 |  |
| rif_sql_uk_electricity_generation_mix_2019_2026__30min | medium (H=480) | native | 287 | 0 | 0 | 0 |  |
| rif_sql_uk_electricity_generation_mix_2019_2026__30min | long (H=720) | native | 287 | 0 | 0 | 0 |  |
| rif_sql_uk_electricity_generation_mix_2019_2026__h | short (H=48) | target:mean | 287 | 0 | 0 | 0 |  |
| rif_sql_uk_electricity_generation_mix_2019_2026__h | medium (H=480) | target:mean | 287 | 0 | 0 | 0 |  |
| rif_sql_uk_electricity_generation_mix_2019_2026__h__long | long (H=720) | target:mean | 256 | 31 | 0 | 0 |  |
| rif_sql_uk_electricity_generation_mix_2019_2026__d | short (H=30) | target:mean | 256 | 31 | 0 | 0 |  |
| rif_sql_uk_electricity_generation_mix_2019_2026__wsun | short (H=8) | target:mean | 32 | 255 | 0 | 0 |  |
| ron_the_code_telangana_time_series_2023_2025__d | short (H=30) | native | 7 | 0 | 0 | 0 |  |
| ron_the_code_telangana_time_series_2023_2025__wsun | short (H=8) | target:mean | 7 | 0 | 0 | 0 |  |
| roshangupta007_yieldformer_de_data__d | short (H=30) | native | 350 | 0 | 15 | 1 |  |
| roshangupta007_yieldformer_de_data__wsun | short (H=8) | humidity:mean, precipitation_height:mean, radiation_global:mean, temperature_air_max_2m:mean, temperature_air_min_2m:mean, wind_speed:mean, ndvi:mean, evi:mean, lai:mean, ndvi_observed:last, ndvi_stale:last, ndvi_age_days:mean, evi_observed:last, evi_stale:last, evi_age_days:mean, lai_observed:last, lai_stale:last, lai_age_days:mean, doy:last, yield_dt_per_ha:mean, bulk_density:mean, clay:mean, ph:mean, sand:mean, soc:mean, sand_imputed:last, clay_imputed:last, soc_imputed:last, ph_imputed:last, bulk_density_imputed:last, soil_imputed:last, slope_mean:mean, slope_std:mean | 339 | 1 | 26 | 0 |  |
| rricc22_endomondo_hr_prediction_v2__6s | short (H=48) | native | 50 | 0 | 950 | 0 |  |
| rricc22_endomondo_hr_prediction_v2__6s | medium (H=480) | native | 0 | 1000 | 0 | 0 | 0 of 1000 series long enough (< 75%) |
| rricc22_endomondo_hr_prediction_v2__6s | long (H=720) | native | 0 | 1000 | 0 | 0 | 0 of 1000 series long enough (< 75%) |
| salesforce_lotsa_data_alibaba_cluster_trace_2018__5min | short (H=48) | native | 981 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_alibaba_cluster_trace_2018__5min__medium | medium (H=480) | native | 968 | 13 | 0 | 0 |  |
| salesforce_lotsa_data_alibaba_cluster_trace_2018__5min | long (H=720) | native | 0 | 981 | 0 | 0 | 0 of 981 series long enough (< 75%) |
| salesforce_lotsa_data_bdg_2_bear__h | short (H=48) | native | 89 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_bdg_2_bear__h | medium (H=480) | native | 89 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_bdg_2_bear__h | long (H=720) | native | 89 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_bdg_2_bear__d | short (H=30) | target:mean | 89 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_bdg_2_bear__wsun | short (H=8) | target:mean | 76 | 13 | 0 | 0 |  |
| salesforce_lotsa_data_bdg_2_fox__h | short (H=48) | native | 130 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_bdg_2_fox__h | medium (H=480) | native | 130 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_bdg_2_fox__h | long (H=720) | native | 130 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_bdg_2_fox__d | short (H=30) | target:mean | 130 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_bdg_2_fox__wsun | short (H=8) | target:mean | 125 | 5 | 0 | 0 |  |
| salesforce_lotsa_data_bdg_2_panther__h | short (H=48) | native | 101 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_bdg_2_panther__h | medium (H=480) | native | 101 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_bdg_2_panther__h | long (H=720) | native | 101 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_bdg_2_panther__d | short (H=30) | target:mean | 101 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_bdg_2_rat__h | short (H=48) | native | 264 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_bdg_2_rat__h | medium (H=480) | native | 264 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_bdg_2_rat__h | long (H=720) | native | 264 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_bdg_2_rat__d | short (H=30) | target:mean | 264 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_bdg_2_rat__wsun | short (H=8) | target:mean | 245 | 19 | 0 | 0 |  |
| salesforce_lotsa_data_beijing_air_quality__h | short (H=48) | native | 0 | 0 | 12 | 0 | no series left after the NaN / scale rules |
| salesforce_lotsa_data_beijing_air_quality__h | medium (H=480) | native | 0 | 0 | 12 | 0 | no series left after the NaN / scale rules |
| salesforce_lotsa_data_beijing_air_quality__h | long (H=720) | native | 0 | 0 | 12 | 0 | no series left after the NaN / scale rules |
| salesforce_lotsa_data_beijing_air_quality__d | short (H=30) | target_0:mean, target_1:mean, target_2:mean, target_3:mean, target_4:mean, target_5:mean, target_6:mean, target_7:mean, target_8:mean, target_9:mean, target_10:mean | 0 | 0 | 12 | 0 | no series left after the NaN / scale rules |
| salesforce_lotsa_data_beijing_air_quality__wsun | short (H=8) | target_0:mean, target_1:mean, target_2:mean, target_3:mean, target_4:mean, target_5:mean, target_6:mean, target_7:mean, target_8:mean, target_9:mean, target_10:mean | 10 | 0 | 2 | 0 |  |
| salesforce_lotsa_data_beijing_subway_30min__30min | short (H=48) | native | 0 | 0 | 276 | 0 | no series left after the NaN / scale rules |
| salesforce_lotsa_data_beijing_subway_30min__30min | medium (H=480) | native | 0 | 0 | 276 | 0 | no series left after the NaN / scale rules |
| salesforce_lotsa_data_beijing_subway_30min__30min | long (H=720) | native | 0 | 276 | 0 | 0 | 0 of 276 series long enough (< 75%) |
| salesforce_lotsa_data_beijing_subway_30min__h | short (H=48) | target_0:mean, target_1:mean, past_feat_dynamic_real_0:mean, past_feat_dynamic_real_1:mean, past_feat_dynamic_real_2:mean, past_feat_dynamic_real_3:mean, past_feat_dynamic_real_4:mean, past_feat_dynamic_real_5:mean, past_feat_dynamic_real_6:mean, past_feat_dynamic_real_7:mean, past_feat_dynamic_real_8:mean, past_feat_dynamic_real_9:mean, past_feat_dynamic_real_10:mean | 0 | 0 | 276 | 0 | no series left after the NaN / scale rules |
| salesforce_lotsa_data_beijing_subway_30min__h | medium (H=480) | target_0:mean, target_1:mean, past_feat_dynamic_real_0:mean, past_feat_dynamic_real_1:mean, past_feat_dynamic_real_2:mean, past_feat_dynamic_real_3:mean, past_feat_dynamic_real_4:mean, past_feat_dynamic_real_5:mean, past_feat_dynamic_real_6:mean, past_feat_dynamic_real_7:mean, past_feat_dynamic_real_8:mean, past_feat_dynamic_real_9:mean, past_feat_dynamic_real_10:mean | 0 | 276 | 0 | 0 | 0 of 276 series long enough (< 75%) |
| salesforce_lotsa_data_beijing_subway_30min__h | long (H=720) | target_0:mean, target_1:mean, past_feat_dynamic_real_0:mean, past_feat_dynamic_real_1:mean, past_feat_dynamic_real_2:mean, past_feat_dynamic_real_3:mean, past_feat_dynamic_real_4:mean, past_feat_dynamic_real_5:mean, past_feat_dynamic_real_6:mean, past_feat_dynamic_real_7:mean, past_feat_dynamic_real_8:mean, past_feat_dynamic_real_9:mean, past_feat_dynamic_real_10:mean | 0 | 276 | 0 | 0 | 0 of 276 series long enough (< 75%) |
| salesforce_lotsa_data_borealis__h | short (H=48) | native | 8 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_borealis__h | medium (H=480) | native | 8 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_borealis__h | long (H=720) | native | 8 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_borealis__d | short (H=30) | target:mean | 8 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_borg_cluster_data_2011__5min | short (H=48) | native | 875 | 0 | 93 | 0 |  |
| salesforce_lotsa_data_borg_cluster_data_2011__5min | medium (H=480) | native | 0 | 400 | 0 | 0 | 568 of 968 series long enough (< 75%) |
| salesforce_lotsa_data_borg_cluster_data_2011__5min | long (H=720) | native | 0 | 481 | 0 | 0 | 487 of 968 series long enough (< 75%) |
| salesforce_lotsa_data_borg_cluster_data_2011__h | short (H=48) | target_0:mean, target_1:mean, past_feat_dynamic_real_0:mean, past_feat_dynamic_real_1:mean, past_feat_dynamic_real_2:mean, past_feat_dynamic_real_3:mean, past_feat_dynamic_real_4:last | 369 | 582 | 17 | 0 |  |
| salesforce_lotsa_data_borg_cluster_data_2011__h | medium (H=480) | target_0:mean, target_1:mean, past_feat_dynamic_real_0:mean, past_feat_dynamic_real_1:mean, past_feat_dynamic_real_2:mean, past_feat_dynamic_real_3:mean, past_feat_dynamic_real_4:last | 0 | 386 | 0 | 0 | 0 of 386 series long enough (< 75%) |
| salesforce_lotsa_data_borg_cluster_data_2011__h | long (H=720) | target_0:mean, target_1:mean, past_feat_dynamic_real_0:mean, past_feat_dynamic_real_1:mean, past_feat_dynamic_real_2:mean, past_feat_dynamic_real_3:mean, past_feat_dynamic_real_4:last | 0 | 386 | 0 | 0 | 0 of 386 series long enough (< 75%) |
| salesforce_lotsa_data_cdc_fluview_ilinet__wsun | short (H=8) | native | 19 | 0 | 53 | 0 |  |
| salesforce_lotsa_data_cdc_fluview_who_nrevss__wsun | short (H=8) | native | 24 | 0 | 0 | 49 |  |
| salesforce_lotsa_data_china_air_quality__h | short (H=48) | native | 0 | 0 | 411 | 0 | no series left after the NaN / scale rules |
| salesforce_lotsa_data_china_air_quality__h | medium (H=480) | native | 0 | 0 | 411 | 0 | no series left after the NaN / scale rules |
| salesforce_lotsa_data_china_air_quality__h | long (H=720) | native | 0 | 1 | 410 | 0 | no series left after the NaN / scale rules |
| salesforce_lotsa_data_china_air_quality__d | short (H=30) | target_0:mean, target_1:mean, target_2:mean, target_3:mean, target_4:mean, target_5:mean | 0 | 27 | 384 | 0 | no series left after the NaN / scale rules |
| salesforce_lotsa_data_covid_mobility__d | short (H=30) | native | 337 | 0 | 25 | 0 |  |
| salesforce_lotsa_data_extended_web_traffic_with_missing__d | short (H=30) | native | 902 | 0 | 98 | 0 |  |
| salesforce_lotsa_data_extended_web_traffic_with_missing__wsun | short (H=8) | target:mean | 939 | 0 | 61 | 0 |  |
| salesforce_lotsa_data_hzmetro__15min | short (H=48) | native | 0 | 0 | 80 | 0 | no series left after the NaN / scale rules |
| salesforce_lotsa_data_hzmetro__15min | medium (H=480) | native | 0 | 0 | 80 | 0 | no series left after the NaN / scale rules |
| salesforce_lotsa_data_hzmetro__15min | long (H=720) | native | 0 | 0 | 80 | 0 | no series left after the NaN / scale rules |
| salesforce_lotsa_data_hzmetro__h | short (H=48) | target_0:mean, target_1:mean | 0 | 0 | 80 | 0 | no series left after the NaN / scale rules |
| salesforce_lotsa_data_hzmetro__h | medium (H=480) | target_0:mean, target_1:mean | 0 | 80 | 0 | 0 | 0 of 80 series long enough (< 75%) |
| salesforce_lotsa_data_hzmetro__h | long (H=720) | target_0:mean, target_1:mean | 0 | 80 | 0 | 0 | 0 of 80 series long enough (< 75%) |
| salesforce_lotsa_data_ideal__h | short (H=48) | native | 195 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_ideal__h__medium | medium (H=480) | native | 183 | 12 | 0 | 0 |  |
| salesforce_lotsa_data_ideal__h__long | long (H=720) | native | 172 | 23 | 0 | 0 |  |
| salesforce_lotsa_data_ideal__d | short (H=30) | target:mean | 132 | 63 | 0 | 0 |  |
| salesforce_lotsa_data_ideal__wsun | short (H=8) | target:mean | 20 | 175 | 0 | 0 |  |
| salesforce_lotsa_data_kaggle_web_traffic_weekly__wsun | short (H=8) | native | 1000 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_largest_2021__5min | short (H=48) | native | 997 | 0 | 3 | 0 |  |
| salesforce_lotsa_data_largest_2021__5min__medium | medium (H=480) | native | 988 | 0 | 12 | 0 |  |
| salesforce_lotsa_data_largest_2021__5min__medium | long (H=720) | native | 988 | 0 | 12 | 0 |  |
| salesforce_lotsa_data_largest_2021__h | short (H=48) | target:mean | 997 | 0 | 3 | 0 |  |
| salesforce_lotsa_data_largest_2021__h | medium (H=480) | target:mean | 997 | 0 | 3 | 0 |  |
| salesforce_lotsa_data_largest_2021__h | long (H=720) | target:mean | 997 | 0 | 3 | 0 |  |
| salesforce_lotsa_data_largest_2021__d | short (H=30) | target:mean | 1000 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_lcl__h | short (H=48) | native | 710 | 0 | 0 | 1 |  |
| salesforce_lotsa_data_lcl__h | medium (H=480) | native | 710 | 1 | 0 | 0 |  |
| salesforce_lotsa_data_lcl__h__long | long (H=720) | native | 708 | 3 | 0 | 0 |  |
| salesforce_lotsa_data_lcl__d | short (H=30) | target:mean | 701 | 10 | 0 | 0 |  |
| salesforce_lotsa_data_lcl__wsun | short (H=8) | target:mean | 665 | 46 | 0 | 0 |  |
| salesforce_lotsa_data_london_smart_meters_with_missing__30min | short (H=48) | native | 999 | 0 | 1 | 0 |  |
| salesforce_lotsa_data_london_smart_meters_with_missing__30min__medium | medium (H=480) | native | 995 | 0 | 5 | 0 |  |
| salesforce_lotsa_data_london_smart_meters_with_missing__30min__medium | long (H=720) | native | 995 | 0 | 5 | 0 |  |
| salesforce_lotsa_data_london_smart_meters_with_missing__h | short (H=48) | target:mean | 998 | 0 | 2 | 0 |  |
| salesforce_lotsa_data_london_smart_meters_with_missing__h | medium (H=480) | target:mean | 998 | 0 | 2 | 0 |  |
| salesforce_lotsa_data_london_smart_meters_with_missing__h | long (H=720) | target:mean | 998 | 0 | 2 | 0 |  |
| salesforce_lotsa_data_london_smart_meters_with_missing__d | short (H=30) | target:mean | 1000 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_london_smart_meters_with_missing__wsun | short (H=8) | target:mean | 959 | 41 | 0 | 0 |  |
| salesforce_lotsa_data_los_loop__5min | short (H=48) | native | 207 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_los_loop__5min | medium (H=480) | native | 207 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_los_loop__5min | long (H=720) | native | 207 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_los_loop__h | short (H=48) | target:mean | 207 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_los_loop__h | medium (H=480) | target:mean | 207 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_los_loop__h | long (H=720) | target:mean | 207 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_m1_monthly__ms | short (H=12) | native | 473 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_nn5_daily_with_missing__d | short (H=30) | native | 57 | 0 | 51 | 0 |  |
| salesforce_lotsa_data_nn5_daily_with_missing__wsun | short (H=8) | target:mean | 93 | 0 | 15 | 0 |  |
| salesforce_lotsa_data_oikolab_weather__h | short (H=48) | native | 8 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_oikolab_weather__h | medium (H=480) | native | 8 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_oikolab_weather__h | long (H=720) | native | 8 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_oikolab_weather__d | short (H=30) | target:mean | 8 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_oikolab_weather__wsun | short (H=8) | target:mean | 8 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_pedestrian_counts__h | short (H=48) | native | 65 | 0 | 0 | 1 |  |
| salesforce_lotsa_data_pedestrian_counts__h__medium | medium (H=480) | native | 63 | 2 | 0 | 1 |  |
| salesforce_lotsa_data_pedestrian_counts__h__long | long (H=720) | native | 62 | 4 | 0 | 0 |  |
| salesforce_lotsa_data_pedestrian_counts__d | short (H=30) | target:mean | 61 | 5 | 0 | 0 |  |
| salesforce_lotsa_data_pedestrian_counts__wsun | short (H=8) | target:mean | 58 | 8 | 0 | 0 |  |
| salesforce_lotsa_data_pems03__5min | short (H=48) | native | 348 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_pems03__5min | medium (H=480) | native | 348 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_pems03__5min | long (H=720) | native | 348 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_pems03__h | short (H=48) | target:mean | 348 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_pems03__h | medium (H=480) | target:mean | 348 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_pems03__h | long (H=720) | target:mean | 348 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_pems04__5min | short (H=48) | native | 289 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_pems04__5min | medium (H=480) | native | 289 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_pems04__5min | long (H=720) | native | 289 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_pems04__h | short (H=48) | target_0:mean, target_1:mean, target_2:mean | 289 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_pems04__h | medium (H=480) | target_0:mean, target_1:mean, target_2:mean | 0 | 289 | 0 | 0 | 0 of 289 series long enough (< 75%) |
| salesforce_lotsa_data_pems04__h | long (H=720) | target_0:mean, target_1:mean, target_2:mean | 0 | 289 | 0 | 0 | 0 of 289 series long enough (< 75%) |
| salesforce_lotsa_data_pems07__5min | short (H=48) | native | 748 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_pems07__5min | medium (H=480) | native | 748 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_pems07__5min | long (H=720) | native | 748 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_pems07__h | short (H=48) | target:mean | 748 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_pems07__h | medium (H=480) | target:mean | 748 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_pems07__h | long (H=720) | target:mean | 748 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_pems08__5min | short (H=48) | native | 149 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_pems08__5min | medium (H=480) | native | 149 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_pems08__5min | long (H=720) | native | 149 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_pems08__h | short (H=48) | target_0:mean, target_1:mean, target_2:mean | 149 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_pems08__h | medium (H=480) | target_0:mean, target_1:mean, target_2:mean | 149 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_pems08__h | long (H=720) | target_0:mean, target_1:mean, target_2:mean | 0 | 149 | 0 | 0 | 0 of 149 series long enough (< 75%) |
| salesforce_lotsa_data_project_tycho__wsun | short (H=8) | native | 308 | 0 | 689 | 3 |  |
| salesforce_lotsa_data_q_traffic__15min | short (H=48) | native | 1000 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_q_traffic__15min | medium (H=480) | native | 1000 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_q_traffic__15min | long (H=720) | native | 1000 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_q_traffic__h | short (H=48) | target:mean | 1000 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_q_traffic__h | medium (H=480) | target:mean | 1000 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_q_traffic__h | long (H=720) | target:mean | 0 | 1000 | 0 | 0 | 0 of 1000 series long enough (< 75%) |
| salesforce_lotsa_data_residential_load_power__min | short (H=48) | native | 235 | 0 | 19 | 1 |  |
| salesforce_lotsa_data_residential_load_power__min__medium | medium (H=480) | native | 200 | 1 | 54 | 0 |  |
| salesforce_lotsa_data_residential_load_power__min__long | long (H=720) | native | 202 | 1 | 52 | 0 |  |
| salesforce_lotsa_data_residential_load_power__h | short (H=48) | target_0:mean, target_1:mean, target_2:mean | 249 | 1 | 4 | 1 |  |
| salesforce_lotsa_data_residential_load_power__h | medium (H=480) | target_0:mean, target_1:mean, target_2:mean | 0 | 254 | 0 | 0 | 0 of 254 series long enough (< 75%) |
| salesforce_lotsa_data_residential_load_power__h | long (H=720) | target_0:mean, target_1:mean, target_2:mean | 0 | 254 | 0 | 0 | 0 of 254 series long enough (< 75%) |
| salesforce_lotsa_data_residential_pv_power__min | short (H=48) | native | 199 | 0 | 19 | 2 |  |
| salesforce_lotsa_data_residential_pv_power__min__medium | medium (H=480) | native | 174 | 1 | 44 | 1 |  |
| salesforce_lotsa_data_residential_pv_power__min__long | long (H=720) | native | 175 | 1 | 43 | 1 |  |
| salesforce_lotsa_data_residential_pv_power__h | short (H=48) | target_0:mean, target_1:mean, target_2:mean | 210 | 1 | 7 | 2 |  |
| salesforce_lotsa_data_residential_pv_power__h | medium (H=480) | target_0:mean, target_1:mean, target_2:mean | 0 | 219 | 0 | 0 | 0 of 219 series long enough (< 75%) |
| salesforce_lotsa_data_residential_pv_power__h | long (H=720) | target_0:mean, target_1:mean, target_2:mean | 0 | 219 | 0 | 0 | 0 of 219 series long enough (< 75%) |
| salesforce_lotsa_data_rideshare_with_missing__h | short (H=48) | native | 0 | 0 | 1000 | 0 | no series left after the NaN / scale rules |
| salesforce_lotsa_data_rideshare_with_missing__h | medium (H=480) | native | 0 | 1000 | 0 | 0 | 0 of 1000 series long enough (< 75%) |
| salesforce_lotsa_data_rideshare_with_missing__h | long (H=720) | native | 0 | 1000 | 0 | 0 | 0 of 1000 series long enough (< 75%) |
| salesforce_lotsa_data_shmetro__15min | short (H=48) | native | 0 | 0 | 288 | 0 | no series left after the NaN / scale rules |
| salesforce_lotsa_data_shmetro__15min | medium (H=480) | native | 0 | 0 | 288 | 0 | no series left after the NaN / scale rules |
| salesforce_lotsa_data_shmetro__15min | long (H=720) | native | 0 | 0 | 288 | 0 | no series left after the NaN / scale rules |
| salesforce_lotsa_data_shmetro__h | short (H=48) | target_0:mean, target_1:mean | 0 | 0 | 288 | 0 | no series left after the NaN / scale rules |
| salesforce_lotsa_data_shmetro__h | medium (H=480) | target_0:mean, target_1:mean | 0 | 0 | 288 | 0 | no series left after the NaN / scale rules |
| salesforce_lotsa_data_shmetro__h | long (H=720) | target_0:mean, target_1:mean | 0 | 0 | 288 | 0 | no series left after the NaN / scale rules |
| salesforce_lotsa_data_smart__h | short (H=48) | native | 5 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_smart__h | medium (H=480) | native | 5 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_smart__h | long (H=720) | native | 5 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_smart__d | short (H=30) | target:mean | 5 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_smart__wsun | short (H=8) | target:mean | 3 | 2 | 0 | 0 |  |
| salesforce_lotsa_data_subseasonal__d | short (H=30) | native | 862 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_subseasonal__wsun | short (H=8) | target_0:mean, target_1:mean, target_2:mean, target_3:mean | 862 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_subseasonal_precip__d | short (H=30) | native | 862 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_subseasonal_precip__wsun | short (H=8) | target:mean | 862 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_sunspot_with_missing__d | short (H=30) | native | 1 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_sunspot_with_missing__wsun | short (H=8) | target:mean | 1 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_taxi_30min__30min | short (H=48) | native | 1000 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_taxi_30min__30min | medium (H=480) | native | 0 | 955 | 0 | 0 | 45 of 1000 series long enough (< 75%) |
| salesforce_lotsa_data_taxi_30min__30min | long (H=720) | native | 0 | 1000 | 0 | 0 | 0 of 1000 series long enough (< 75%) |
| salesforce_lotsa_data_taxi_30min__h | short (H=48) | target:mean | 737 | 263 | 0 | 0 |  |
| salesforce_lotsa_data_taxi_30min__h | medium (H=480) | target:mean | 0 | 737 | 0 | 0 | 0 of 737 series long enough (< 75%) |
| salesforce_lotsa_data_taxi_30min__h | long (H=720) | target:mean | 0 | 737 | 0 | 0 | 0 of 737 series long enough (< 75%) |
| salesforce_lotsa_data_traffic_hourly__h | short (H=48) | native | 862 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_traffic_hourly__h | medium (H=480) | native | 862 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_traffic_hourly__h | long (H=720) | native | 862 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_traffic_hourly__d | short (H=30) | target:mean | 862 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_traffic_hourly__wsun | short (H=8) | target:mean | 862 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_uber_tlc_daily__d | short (H=30) | native | 257 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_uber_tlc_hourly__h | short (H=48) | native | 262 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_uber_tlc_hourly__h | medium (H=480) | native | 262 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_uber_tlc_hourly__h | long (H=720) | native | 262 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_uber_tlc_hourly__d | short (H=30) | target:mean | 257 | 5 | 0 | 0 |  |
| salesforce_lotsa_data_vehicle_trips_with_missing__d | short (H=30) | native | 31 | 0 | 2 | 0 |  |
| salesforce_lotsa_data_weather__d | short (H=30) | native | 999 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_weather__wsun | short (H=8) | target:mean | 999 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_wiki_rolling_nips__d | short (H=30) | native | 1000 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_wiki_rolling_nips__wsun | short (H=8) | target:mean | 1000 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_wind_farms_with_missing__min | short (H=48) | native | 270 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_wind_farms_with_missing__min__medium | medium (H=480) | native | 267 | 0 | 0 | 3 |  |
| salesforce_lotsa_data_wind_farms_with_missing__min__medium | long (H=720) | native | 267 | 0 | 0 | 3 |  |
| salesforce_lotsa_data_wind_farms_with_missing__h | short (H=48) | target:mean | 270 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_wind_farms_with_missing__h | medium (H=480) | target:mean | 0 | 270 | 0 | 0 | 0 of 270 series long enough (< 75%) |
| salesforce_lotsa_data_wind_farms_with_missing__h | long (H=720) | target:mean | 0 | 270 | 0 | 0 | 0 of 270 series long enough (< 75%) |
| salesforce_lotsa_data_wind_power__4s | short (H=48) | native | 1 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_wind_power__4s | medium (H=480) | native | 1 | 0 | 0 | 0 |  |
| salesforce_lotsa_data_wind_power__4s | long (H=720) | native | 1 | 0 | 0 | 0 |  |
| samuelm0422_hourly_electricity_demand_brazil_dataset__h | short (H=48) | native | 4 | 0 | 0 | 0 |  |
| samuelm0422_hourly_electricity_demand_brazil_dataset__h | medium (H=480) | native | 4 | 0 | 0 | 0 |  |
| samuelm0422_hourly_electricity_demand_brazil_dataset__h | long (H=720) | native | 4 | 0 | 0 | 0 |  |
| samuelm0422_hourly_electricity_demand_brazil_dataset__d | short (H=30) | val_cargaenergiahomwmed:mean | 4 | 0 | 0 | 0 |  |
| samuelm0422_hourly_electricity_demand_brazil_dataset__wsun | short (H=8) | val_cargaenergiahomwmed:mean | 4 | 0 | 0 | 0 |  |
| sayanroy058_weather_time_series_forecasting__h | short (H=48) | native | 14 | 0 | 0 | 0 |  |
| sayanroy058_weather_time_series_forecasting__h | medium (H=480) | native | 14 | 0 | 0 | 0 |  |
| sayanroy058_weather_time_series_forecasting__h | long (H=720) | native | 14 | 0 | 0 | 0 |  |
| sayanroy058_weather_time_series_forecasting__d | short (H=30) | target:mean | 14 | 0 | 0 | 0 |  |
| sayanroy058_weather_time_series_forecasting__wsun | short (H=8) | target:mean | 14 | 0 | 0 | 0 |  |
| sevriukov_bank101__ms | short (H=12) | native | 0 | 0 | 1 | 0 | no series left after the NaN / scale rules |
| shashkovich_telecommunication_sms_time_series__d | short (H=30) | native | 2 | 0 | 0 | 0 |  |
| showlcw_china_cpi_ppi_pmi__ms | short (H=12) | native | 1 | 0 | 0 | 0 |  |
| skforecast_ashrae_daily__d | short (H=30) | native | 998 | 0 | 0 | 0 |  |
| skywalker1910_skynet_aqi_panel__h | short (H=48) | native | 33 | 0 | 0 | 0 |  |
| skywalker1910_skynet_aqi_panel__h | medium (H=480) | native | 33 | 0 | 0 | 0 |  |
| skywalker1910_skynet_aqi_panel__h | long (H=720) | native | 33 | 0 | 0 | 0 |  |
| skywalker1910_skynet_aqi_panel__d | short (H=30) | pblh_m:mean, ustar_ms:mean, wind_speed_merra_ms:mean, dust_pm25:mean, seasalt_pm25:mean, black_carbon:mean, organic_carbon:mean, sulfate:mean | 33 | 0 | 0 | 0 |  |
| skywalker1910_skynet_aqi_panel__wsun | short (H=8) | pblh_m:mean, ustar_ms:mean, wind_speed_merra_ms:mean, dust_pm25:mean, seasalt_pm25:mean, black_carbon:mean, organic_carbon:mean, sulfate:mean | 33 | 0 | 0 | 0 |  |
| somemone_ac_transit_apc__d | short (H=30) | native | 1000 | 0 | 0 | 0 |  |
| somemone_ac_transit_apc__wsun | short (H=8) | count:sum | 749 | 169 | 0 | 82 |  |
| sovai_wikipedia_views_default__d | short (H=30) | native | 994 | 0 | 0 | 6 |  |
| sovai_wikipedia_views_default__wsun | short (H=8) | views:mean, relative_views:mean, alpha_short:mean, beta_short:mean, search_pressure:mean | 995 | 0 | 0 | 5 |  |
| steven0226_air_quality__d | short (H=30) | native | 295 | 0 | 705 | 0 |  |
| steven0226_air_quality__wsun | short (H=8) | mean:mean, n_valid:mean | 527 | 1 | 472 | 0 |  |
| surfe_diem_wave_archive_usa_southwest__h | short (H=48) | native | 59 | 0 | 941 | 0 |  |
| surfe_diem_wave_archive_usa_southwest__h__medium | medium (H=480) | native | 159 | 28 | 813 | 0 |  |
| surfe_diem_wave_archive_usa_southwest__h__long | long (H=720) | native | 68 | 57 | 875 | 0 |  |
| surfe_diem_wave_archive_usa_southwest__d | short (H=30) | target:mean | 599 | 94 | 307 | 0 |  |
| thanna94_us_building_permits__ys | short (H=6) | native | 630 | 0 | 368 | 2 |  |
| tharinduperera_weather_clustering_data__d | short (H=30) | native | 99 | 0 | 0 | 0 |  |
| tharinduperera_weather_clustering_data__wsun | short (H=8) | temperature_mean:mean, temperature_max:mean, temperature_min:mean, precipitation_sum:mean, relative_humidity_mean:mean, wind_speed_mean:mean, surface_pressure_mean:mean | 99 | 0 | 0 | 0 |  |
| thecharttruth_etf_data_prices__d | short (H=30) | native | 0 | 0 | 95 | 0 | no series left after the NaN / scale rules |
| thecharttruth_etf_data_prices__wsun | short (H=8) | open:mean, high:mean, low:mean, close:mean, volume:mean | 0 | 0 | 95 | 0 | no series left after the NaN / scale rules |
| thermobench_thermobench_pure__5min | short (H=48) | native | 20 | 0 | 9 | 0 |  |
| thermobench_thermobench_pure__5min | medium (H=480) | native | 20 | 0 | 9 | 0 |  |
| thermobench_thermobench_pure__5min | long (H=720) | native | 20 | 0 | 9 | 0 |  |
| thermobench_thermobench_pure__h | short (H=48) | target:mean | 20 | 0 | 9 | 0 |  |
| thermobench_thermobench_pure__h__medium | medium (H=480) | target:mean | 10 | 0 | 19 | 0 |  |
| thermobench_thermobench_pure__h__medium | long (H=720) | target:mean | 10 | 0 | 19 | 0 |  |
| thermobench_thermobench_pure__d | short (H=30) | target:mean | 10 | 0 | 19 | 0 |  |
| todopueblos_poblacion_municipios_espana__ys | short (H=6) | native | 1000 | 0 | 0 | 0 |  |
| valdovaldo1500_crime_brasil_rs_2022_2026__ms | short (H=12) | native | 1 | 0 | 0 | 0 |  |
| venvoo_openrouter_uptime__897s | short (H=48) | native | 0 | 0 | 28 | 0 | no series left after the NaN / scale rules |
| venvoo_openrouter_uptime__897s | medium (H=480) | native | 0 | 1 | 27 | 0 | no series left after the NaN / scale rules |
| venvoo_openrouter_uptime__897s | long (H=720) | native | 0 | 2 | 26 | 0 | no series left after the NaN / scale rules |
| venvoo_openrouter_uptime__h | short (H=48) | status:mean, up5m:mean, up30m:mean, up1d:mean, identity_ambiguous:last | 0 | 1 | 27 | 0 | no series left after the NaN / scale rules |
| venvoo_openrouter_uptime__h | medium (H=480) | status:mean, up5m:mean, up30m:mean, up1d:mean, identity_ambiguous:last | 0 | 1 | 26 | 0 | no series left after the NaN / scale rules |
| venvoo_openrouter_uptime__h | long (H=720) | status:mean, up5m:mean, up30m:mean, up1d:mean, identity_ambiguous:last | 0 | 3 | 24 | 0 | no series left after the NaN / scale rules |
| wikimedia_community_english_wikipedia_pageviews_by_second__s | short (H=48) | native | 2 | 0 | 0 | 0 |  |
| wikimedia_community_english_wikipedia_pageviews_by_second__s | medium (H=480) | native | 2 | 0 | 0 | 0 |  |
| wikimedia_community_english_wikipedia_pageviews_by_second__s | long (H=720) | native | 2 | 0 | 0 | 0 |  |
| witgaw_metr_la__5min | short (H=48) | native | 207 | 0 | 0 | 0 |  |
| witgaw_metr_la__5min | medium (H=480) | native | 207 | 0 | 0 | 0 |  |
| witgaw_metr_la__5min | long (H=720) | native | 207 | 0 | 0 | 0 |  |
| witgaw_metr_la__h | short (H=48) | x_t-11_d0:mean, x_t-11_d1:mean, x_t-10_d0:mean, x_t-10_d1:mean, x_t-9_d0:mean, x_t-9_d1:mean, x_t-8_d0:mean, x_t-8_d1:mean, x_t-7_d0:mean, x_t-7_d1:mean, x_t-6_d0:mean, x_t-6_d1:mean, x_t-5_d0:mean, x_t-5_d1:mean, x_t-4_d0:mean, x_t-4_d1:mean, x_t-3_d0:mean, x_t-3_d1:mean, x_t-2_d0:mean, x_t-2_d1:mean, x_t-1_d0:mean, x_t-1_d1:mean, x_t+0_d0:mean, x_t+0_d1:mean, y_t+1_d0:mean, y_t+1_d1:mean, y_t+2_d0:mean, y_t+2_d1:mean, y_t+3_d0:mean, y_t+3_d1:mean, y_t+4_d0:mean, y_t+4_d1:mean, y_t+5_d0:mean | 207 | 0 | 0 | 0 |  |
| witgaw_metr_la__h | medium (H=480) | x_t-11_d0:mean, x_t-11_d1:mean, x_t-10_d0:mean, x_t-10_d1:mean, x_t-9_d0:mean, x_t-9_d1:mean, x_t-8_d0:mean, x_t-8_d1:mean, x_t-7_d0:mean, x_t-7_d1:mean, x_t-6_d0:mean, x_t-6_d1:mean, x_t-5_d0:mean, x_t-5_d1:mean, x_t-4_d0:mean, x_t-4_d1:mean, x_t-3_d0:mean, x_t-3_d1:mean, x_t-2_d0:mean, x_t-2_d1:mean, x_t-1_d0:mean, x_t-1_d1:mean, x_t+0_d0:mean, x_t+0_d1:mean, y_t+1_d0:mean, y_t+1_d1:mean, y_t+2_d0:mean, y_t+2_d1:mean, y_t+3_d0:mean, y_t+3_d1:mean, y_t+4_d0:mean, y_t+4_d1:mean, y_t+5_d0:mean | 207 | 0 | 0 | 0 |  |
| witgaw_metr_la__h | long (H=720) | x_t-11_d0:mean, x_t-11_d1:mean, x_t-10_d0:mean, x_t-10_d1:mean, x_t-9_d0:mean, x_t-9_d1:mean, x_t-8_d0:mean, x_t-8_d1:mean, x_t-7_d0:mean, x_t-7_d1:mean, x_t-6_d0:mean, x_t-6_d1:mean, x_t-5_d0:mean, x_t-5_d1:mean, x_t-4_d0:mean, x_t-4_d1:mean, x_t-3_d0:mean, x_t-3_d1:mean, x_t-2_d0:mean, x_t-2_d1:mean, x_t-1_d0:mean, x_t-1_d1:mean, x_t+0_d0:mean, x_t+0_d1:mean, y_t+1_d0:mean, y_t+1_d1:mean, y_t+2_d0:mean, y_t+2_d1:mean, y_t+3_d0:mean, y_t+3_d1:mean, y_t+4_d0:mean, y_t+4_d1:mean, y_t+5_d0:mean | 207 | 0 | 0 | 0 |  |
| witgaw_pems_bay__5min | short (H=48) | native | 325 | 0 | 0 | 0 |  |
| witgaw_pems_bay__5min | medium (H=480) | native | 325 | 0 | 0 | 0 |  |
| witgaw_pems_bay__5min | long (H=720) | native | 325 | 0 | 0 | 0 |  |
| witgaw_pems_bay__h | short (H=48) | x_t-11_d0:mean, x_t-11_d1:mean, x_t-10_d0:mean, x_t-10_d1:mean, x_t-9_d0:mean, x_t-9_d1:mean, x_t-8_d0:mean, x_t-8_d1:mean, x_t-7_d0:mean, x_t-7_d1:mean, x_t-6_d0:mean, x_t-6_d1:mean, x_t-5_d0:mean, x_t-5_d1:mean, x_t-4_d0:mean, x_t-4_d1:mean, x_t-3_d0:mean, x_t-3_d1:mean, x_t-2_d0:mean, x_t-2_d1:mean, x_t-1_d0:mean, x_t-1_d1:mean, x_t+0_d0:mean, x_t+0_d1:mean, y_t+1_d0:mean, y_t+1_d1:mean, y_t+2_d0:mean, y_t+2_d1:mean, y_t+3_d0:mean, y_t+3_d1:mean, y_t+4_d0:mean, y_t+4_d1:mean, y_t+5_d0:mean | 325 | 0 | 0 | 0 |  |
| witgaw_pems_bay__h | medium (H=480) | x_t-11_d0:mean, x_t-11_d1:mean, x_t-10_d0:mean, x_t-10_d1:mean, x_t-9_d0:mean, x_t-9_d1:mean, x_t-8_d0:mean, x_t-8_d1:mean, x_t-7_d0:mean, x_t-7_d1:mean, x_t-6_d0:mean, x_t-6_d1:mean, x_t-5_d0:mean, x_t-5_d1:mean, x_t-4_d0:mean, x_t-4_d1:mean, x_t-3_d0:mean, x_t-3_d1:mean, x_t-2_d0:mean, x_t-2_d1:mean, x_t-1_d0:mean, x_t-1_d1:mean, x_t+0_d0:mean, x_t+0_d1:mean, y_t+1_d0:mean, y_t+1_d1:mean, y_t+2_d0:mean, y_t+2_d1:mean, y_t+3_d0:mean, y_t+3_d1:mean, y_t+4_d0:mean, y_t+4_d1:mean, y_t+5_d0:mean | 325 | 0 | 0 | 0 |  |
| witgaw_pems_bay__h | long (H=720) | x_t-11_d0:mean, x_t-11_d1:mean, x_t-10_d0:mean, x_t-10_d1:mean, x_t-9_d0:mean, x_t-9_d1:mean, x_t-8_d0:mean, x_t-8_d1:mean, x_t-7_d0:mean, x_t-7_d1:mean, x_t-6_d0:mean, x_t-6_d1:mean, x_t-5_d0:mean, x_t-5_d1:mean, x_t-4_d0:mean, x_t-4_d1:mean, x_t-3_d0:mean, x_t-3_d1:mean, x_t-2_d0:mean, x_t-2_d1:mean, x_t-1_d0:mean, x_t-1_d1:mean, x_t+0_d0:mean, x_t+0_d1:mean, y_t+1_d0:mean, y_t+1_d1:mean, y_t+2_d0:mean, y_t+2_d1:mean, y_t+3_d0:mean, y_t+3_d1:mean, y_t+4_d0:mean, y_t+4_d1:mean, y_t+5_d0:mean | 325 | 0 | 0 | 0 |  |
| witgaw_pems_bay__d | short (H=30) | x_t-11_d0:mean, x_t-11_d1:mean, x_t-10_d0:mean, x_t-10_d1:mean, x_t-9_d0:mean, x_t-9_d1:mean, x_t-8_d0:mean, x_t-8_d1:mean, x_t-7_d0:mean, x_t-7_d1:mean, x_t-6_d0:mean, x_t-6_d1:mean, x_t-5_d0:mean, x_t-5_d1:mean, x_t-4_d0:mean, x_t-4_d1:mean, x_t-3_d0:mean, x_t-3_d1:mean, x_t-2_d0:mean, x_t-2_d1:mean, x_t-1_d0:mean, x_t-1_d1:mean, x_t+0_d0:mean, x_t+0_d1:mean, y_t+1_d0:mean, y_t+1_d1:mean, y_t+2_d0:mean, y_t+2_d1:mean, y_t+3_d0:mean, y_t+3_d1:mean, y_t+4_d0:mean, y_t+4_d1:mean, y_t+5_d0:mean | 325 | 0 | 0 | 0 |  |
| wrpbr_pncp_compras_modalidade_dia_municipio__d | short (H=30) | native | 0 | 0 | 2 | 0 | no series left after the NaN / scale rules |
| wrpbr_pncp_compras_modalidade_dia_municipio__wsun | short (H=8) | target:mean | 0 | 0 | 2 | 0 | no series left after the NaN / scale rules |
| xairon_piezo_embedding_benchmark__d | short (H=30) | native | 54 | 0 | 935 | 0 |  |
| xairon_piezo_embedding_benchmark__wsun | short (H=8) | niveau_nappe_eau:mean, temperature_2m:mean, total_precipitation:mean, potential_evaporation:mean | 211 | 0 | 778 | 0 |  |
| xus_spain_fuel_prices__d | short (H=30) | native | 918 | 0 | 82 | 0 |  |
| yadavsaurabh_us_stocks_and_search_trends__d | short (H=30) | native | 0 | 0 | 1000 | 0 | no series left after the NaN / scale rules |
| yadavsaurabh_us_stocks_and_search_trends__wsun | short (H=8) | Open:mean, High:mean, Low:mean, Close:mean, Adj Close:mean, Volume:mean, Dividends:mean, Stock Splits:mean, Capital Gains:last | 0 | 113 | 887 | 0 | no series left after the NaN / scale rules |
| yahya2491_3w_pipeline__s | short (H=48) | native | 285 | 0 | 4 | 4 |  |
| yahya2491_3w_pipeline__s | medium (H=480) | native | 285 | 0 | 4 | 4 |  |
| yahya2491_3w_pipeline__s__long | long (H=720) | native | 290 | 0 | 2 | 1 |  |
| youngbongbong_30_od_subway_default__30min | short (H=48) | native | 0 | 0 | 286 | 0 | no series left after the NaN / scale rules |
| youngbongbong_30_od_subway_default__30min | medium (H=480) | native | 0 | 0 | 286 | 0 | no series left after the NaN / scale rules |
| youngbongbong_30_od_subway_default__30min | long (H=720) | native | 0 | 0 | 286 | 0 | no series left after the NaN / scale rules |
| youngbongbong_30_od_subway_default__h | short (H=48) | od_out_cnt:mean, od_in_cnt:mean, od_net_cnt:mean | 0 | 0 | 286 | 0 | no series left after the NaN / scale rules |
| youngbongbong_30_od_subway_default__h | medium (H=480) | od_out_cnt:mean, od_in_cnt:mean, od_net_cnt:mean | 0 | 0 | 286 | 0 | no series left after the NaN / scale rules |
| youngbongbong_30_od_subway_default__h | long (H=720) | od_out_cnt:mean, od_in_cnt:mean, od_net_cnt:mean | 0 | 0 | 286 | 0 | no series left after the NaN / scale rules |
| youngbongbong_30_od_subway_default__d | short (H=30) | od_out_cnt:mean, od_in_cnt:mean, od_net_cnt:mean | 87 | 0 | 199 | 0 |  |
| youngbongbong_30_od_subway_default__wsun | short (H=8) | od_out_cnt:mean, od_in_cnt:mean, od_net_cnt:mean | 285 | 1 | 0 | 0 |  |
| yuiseki_ilostat__qs | short (H=8) | native | 197 | 0 | 0 | 0 |  |
| yuiseki_jma_rain_kanto__h | short (H=48) | native | 97 | 0 | 20 | 0 |  |
| yuiseki_jma_rain_kanto__h__medium | medium (H=480) | native | 83 | 0 | 34 | 0 |  |
| yuiseki_jma_rain_kanto__h__medium | long (H=720) | native | 83 | 0 | 34 | 0 |  |
| yuiseki_jma_rain_kanto__d | short (H=30) | precip_mm:mean, precip_nophen:last, precip_quality:last, precip_homog_in_file:last, pref:last | 107 | 0 | 10 | 0 |  |
| yuiseki_jma_rain_kanto__wsun | short (H=8) | precip_mm:mean, precip_nophen:last, precip_quality:last, precip_homog_in_file:last, pref:last | 114 | 0 | 3 | 0 |  |
| yuiseki_kyoto_bear_sighting__d | short (H=30) | native | 0 | 0 | 6 | 0 | no series left after the NaN / scale rules |
| yuiseki_kyoto_bear_sighting__wsun | short (H=8) | target:mean | 0 | 0 | 6 | 0 | no series left after the NaN / scale rules |
| yuiseki_meta_range_maps_2022_05__d | short (H=30) | native | 827 | 0 | 173 | 0 |  |
| yuiseki_meta_range_maps_2022_05__wsun | short (H=8) | all_day_bing_tiles_visited_relative_change:mean, all_day_ratio_single_tile_users:last | 390 | 591 | 19 | 0 |  |
| yuiseki_pwt__ys | short (H=6) | native | 168 | 0 | 0 | 0 |  |
| yuiseki_un_wpp_wpp2024__ys | short (H=6) | native | 1 | 0 | 0 | 469 |  |
| yuiseki_undata_sdmx__qs | short (H=8) | native | 3 | 0 | 0 | 0 |  |
| yuiseki_undp_hdr__ys | short (H=6) | native | 206 | 0 | 0 | 0 |  |
| yzzpz_charged_data_10_csv_per_city__h | short (H=48) | native | 506 | 0 | 0 | 0 |  |
| yzzpz_charged_data_10_csv_per_city__h | medium (H=480) | native | 506 | 0 | 0 | 0 |  |
| yzzpz_charged_data_10_csv_per_city__h | long (H=720) | native | 506 | 0 | 0 | 0 |  |
| yzzpz_charged_data_10_csv_per_city__d | short (H=30) | target:mean | 506 | 0 | 0 | 0 |  |
| zait_ai_rusweather_434__d | short (H=30) | native | 362 | 0 | 0 | 0 |  |
| zait_ai_rusweather_434__wsun | short (H=8) | T2M:mean, T2M_MAX:mean, T2M_MIN:mean, T2MDEW:mean, RH2M:mean, PRECTOTCORR:mean, PS:mean, WS10M:mean, WS10M_MAX:mean, WS2M:mean, ALLSKY_SFC_SW_DWN:mean, CLOUD_AMT:mean | 362 | 0 | 0 | 0 |  |
| zalizedata_us_grid_hourly_power_plants_dataset__h | short (H=48) | native | 29 | 0 | 0 | 0 |  |
| zalizedata_us_grid_hourly_power_plants_dataset__h | medium (H=480) | native | 29 | 0 | 0 | 0 |  |
| zalizedata_us_grid_hourly_power_plants_dataset__h | long (H=720) | native | 29 | 0 | 0 | 0 |  |
| zalizedata_us_grid_hourly_power_plants_dataset__d | short (H=30) | target:mean | 29 | 0 | 0 | 0 |  |
| zalizedata_us_grid_hourly_power_plants_dataset__wsun | short (H=8) | target:mean | 27 | 2 | 0 | 0 |  |
| zer0contextlost_waterwatch_runoff__ms | short (H=12) | native | 981 | 0 | 18 | 1 |  |

## Seasonal naive baseline
Tasks scored: 1145. Flagged (NaN or infinite score): 20.

### Flagged tasks
| task | MASE | WQL | flag |
|---|---|---|---|
| jotone_arpa_osmer_weatherstations__h/short/target | 0.6786625058279218 | nan | WQL NaN |
| jotone_arpa_osmer_weatherstations__h/short/cov | 0.6786625058279218 | nan | WQL NaN |
| matrosovdani_moscow_parking_occupancy__1799s/short/target | 0.46241907209331573 | nan | WQL NaN |
| matrosovdani_moscow_parking_occupancy__1799s/short/cov | 0.46241907209331573 | nan | WQL NaN |
| national_digital_nt_fuel_prices_history__wsun/short/target | nan | 0.017726162413781797 | MASE NaN; SQL NaN |
| national_digital_nt_fuel_prices_history__wsun/short/cov | nan | 0.017726162413781797 | MASE NaN; SQL NaN |
| salesforce_lotsa_data_sunspot_with_missing__d/short/target | 0.4313921012204215 | inf | WQL inf |
| dmariaa70_metraq_air_quality__h/short/target | nan | 0.21825095333004763 | MASE NaN; SQL NaN |
| dmariaa70_metraq_air_quality__h/short/cov | nan | 0.21825095333004763 | MASE NaN; SQL NaN |
| dmariaa70_metraq_air_quality__h/medium/target | nan | 0.22804157245185536 | MASE NaN; SQL NaN |
| dmariaa70_metraq_air_quality__h/medium/cov | nan | 0.22804157245185536 | MASE NaN; SQL NaN |
| dmariaa70_metraq_air_quality__h/long/target | nan | 0.22533682428319954 | MASE NaN; SQL NaN |
| dmariaa70_metraq_air_quality__h/long/cov | nan | 0.22533682428319954 | MASE NaN; SQL NaN |
| dmariaa70_metraq_air_quality__d/short/target | nan | 0.1750263802663634 | MASE NaN; SQL NaN |
| dmariaa70_metraq_air_quality__d/short/cov | nan | 0.1750263802663634 | MASE NaN; SQL NaN |
| dmariaa70_metraq_air_quality__wsun/short/target | nan | 0.13351567860663166 | MASE NaN; SQL NaN |
| dmariaa70_metraq_air_quality__wsun/short/cov | nan | 0.13351567860663166 | MASE NaN; SQL NaN |
| fidel_ts_canada_photovoltaics_plants__100s/short/target | 4.651900198985587 | nan | WQL NaN |
| fidel_ts_canada_photovoltaics_plants__100s/medium/target | 1.1425920801179539 | nan | WQL NaN |
| fidel_ts_canada_photovoltaics_plants__100s/long/target | 1.2620191909656293 | nan | WQL NaN |

### Per task
| task | split | MASE | WQL | SQL | series_windows |
|---|---|---|---|---|---|
| abmallick_heart_breath_sleep_stage_dataset__30s/short/target | tuning | 1.460 | 0.124 | 1.460 | 74 |
| abmallick_heart_breath_sleep_stage_dataset__30s/short/cov | tuning | 1.460 | 0.124 | 1.460 | 74 |
| ai_iot_energybench_dth__h/short/target | tuning | 1.153 | 0.325 | 1.153 | 14611 |
| ai_iot_energybench_dth__h/medium/target | tuning | 1.629 | 0.409 | 1.629 | 769 |
| ai_iot_energybench_dth__h/long/target | tuning | 1.851 | 0.434 | 1.851 | 769 |
| ai_iot_energybench_dth__d/short/target | tuning | 2.424 | 0.350 | 2.424 | 769 |
| ai_iot_energybench_goiener__d/short/target | tuning | 1.455 | 0.407 | 1.455 | 1920 |
| ai_iot_energybench_goiener__wsun/short/target | tuning | 1.354 | 0.272 | 1.354 | 982 |
| ai_iot_energybench_lcl__30min/short/target | tuning | 1.073 | 0.551 | 1.073 | 16720 |
| ai_iot_energybench_lcl__30min/medium/target | tuning | 1.169 | 0.572 | 1.169 | 4416 |
| ai_iot_energybench_lcl__30min/long/target | tuning | 1.195 | 0.588 | 1.195 | 2880 |
| ai_iot_energybench_lcl__h/short/target | tuning | 1.119 | 0.506 | 1.119 | 14720 |
| ai_iot_energybench_lcl__h/medium/target | tuning | 1.252 | 0.556 | 1.252 | 1965 |
| ai_iot_energybench_lcl__h/long/target | tuning | 1.258 | 0.555 | 1.258 | 1310 |
| ai_iot_energybench_lcl__d/short/target | tuning | 1.495 | 0.285 | 1.495 | 1758 |
| ai_iot_energybench_lcl__wsun/short/target | tuning | 2.816 | 0.242 | 2.816 | 985 |
| ai_iot_energybench_norwegian__h/short/target | tuning | 1.054 | 0.365 | 1.054 | 20000 |
| ai_iot_energybench_norwegian__h/medium/target | tuning | 1.196 | 0.402 | 1.196 | 2000 |
| ai_iot_energybench_norwegian__h/long/target | tuning | 1.160 | 0.402 | 1.160 | 1000 |
| ai_iot_energybench_norwegian__d/short/target | tuning | 1.512 | 0.207 | 1.512 | 1000 |
| ai_iot_energybench_norwegian__wsun/short/target | tuning | 1.135 | 0.142 | 1.135 | 1000 |
| ai_iot_energybench_prayas_ceew__15min/short/target | tuning | 1.122 | 0.554 | 1.122 | 2000 |
| ai_iot_energybench_prayas_ceew__15min/medium/target | tuning | 1.269 | 0.591 | 1.269 | 1000 |
| ai_iot_energybench_prayas_ceew__15min/long/target | tuning | 1.280 | 0.596 | 1.280 | 600 |
| ai_iot_energybench_prayas_ceew__h/short/target | tuning | 1.236 | 0.467 | 1.236 | 2033 |
| ai_iot_energybench_prayas_ceew__h/medium/target | tuning | 1.530 | 0.563 | 1.530 | 107 |
| ai_iot_energybench_prayas_ceew__h/long/target | tuning | 1.799 | 0.671 | 1.799 | 107 |
| ai_iot_energybench_prayas_ceew__d/short/target | tuning | 3.052 | 0.397 | 3.052 | 109 |
| ai_iot_energybench_prayas_ceew__wsun/short/target | tuning | 1.797 | 0.232 | 1.797 | 67 |
| ai_iot_energybench_sgsc__30min/short/target | tuning | 0.828 | 0.356 | 0.828 | 100 |
| ai_iot_energybench_sgsc__30min/medium/target | tuning | 0.885 | 0.396 | 0.885 | 40 |
| ai_iot_energybench_sgsc__30min/long/target | tuning | 1.064 | 0.553 | 1.064 | 25 |
| ai_iot_energybench_sgsc__h/short/target | tuning | 0.891 | 0.537 | 0.891 | 17780 |
| ai_iot_energybench_sgsc__h/medium/target | tuning | 0.967 | 0.589 | 0.967 | 1778 |
| ai_iot_energybench_sgsc__h/long/target | tuning | 1.075 | 0.652 | 1.075 | 895 |
| ai_iot_energybench_sgsc__d/short/target | tuning | 1.245 | 0.354 | 1.245 | 982 |
| ai_iot_energybench_sgsc__wsun/short/target | tuning | 1.520 | 0.272 | 1.520 | 987 |
| ai_iot_energybench_unicon__15min/short/target | tuning | 0.991 | 0.121 | 0.991 | 280 |
| ai_iot_energybench_unicon__15min/long/target | tuning | 1.277 | 0.118 | 1.277 | 54 |
| ai_iot_energybench_unicon__h/short/target | tuning | 1.238 | 0.100 | 1.238 | 180 |
| ai_iot_energybench_unicon__h/medium/target | tuning | 1.683 | 0.135 | 1.683 | 18 |
| ai_iot_energybench_unicon__h/long/target | tuning | 1.421 | 0.155 | 1.421 | 11 |
| ai_iot_energybench_unicon__d/short/target | tuning | 1.531 | 0.138 | 1.531 | 44 |
| ajto_goldencheetah__s/short/target | tuning | 3.420 | 0.464 | 3.420 | 4281 |
| ajto_goldencheetah__s/short/cov | tuning | 3.420 | 0.464 | 3.420 | 4281 |
| ajto_goldencheetah__s/medium/target | tuning | 10.275 | 0.502 | 10.275 | 741 |
| ajto_goldencheetah__s/medium/cov | tuning | 10.275 | 0.502 | 10.275 | 741 |
| ajto_goldencheetah__s/long/target | tuning | 9.455 | 0.478 | 9.455 | 594 |
| ajto_goldencheetah__s/long/cov | tuning | 9.455 | 0.478 | 9.455 | 594 |
| alaminxpro_bd_food_prices__ms/short/target | tuning | 0.514 | 0.035 | 0.514 | 110 |
| alaminxpro_bd_food_prices__ms/short/cov | tuning | 0.514 | 0.035 | 0.514 | 110 |
| alanwgt_fipex_veiculos_brasil__ms/short/target | tuning | 0.764 | 0.066 | 0.764 | 999 |
| alexroz_carbonfluxbench__d/short/target | tuning | 1.854 | 0.015 | 1.854 | 9860 |
| alexroz_carbonfluxbench__d/short/cov | tuning | 1.854 | 0.015 | 1.854 | 9860 |
| alexroz_carbonfluxbench__wsun/short/target | tuning | 1.700 | 0.018 | 1.700 | 7888 |
| alexroz_carbonfluxbench__wsun/short/cov | tuning | 1.700 | 0.018 | 1.700 | 7888 |
| alinurdin_time_series_benchmark_eia_hourly_madrid_15min_webtris_15min_usgs_daily_usgs_continuous_hydat_daily_bom_daily_c__h/short/target | tuning | 1.161 | 0.074 | 1.161 | 960 |
| alinurdin_time_series_benchmark_eia_hourly_madrid_15min_webtris_15min_usgs_daily_usgs_continuous_hydat_daily_bom_daily_c__h/short/cov | tuning | 1.161 | 0.074 | 1.161 | 960 |
| alinurdin_time_series_benchmark_eia_hourly_madrid_15min_webtris_15min_usgs_daily_usgs_continuous_hydat_daily_bom_daily_c__h/medium/target | tuning | 1.781 | 0.101 | 1.781 | 420 |
| alinurdin_time_series_benchmark_eia_hourly_madrid_15min_webtris_15min_usgs_daily_usgs_continuous_hydat_daily_bom_daily_c__h/medium/cov | tuning | 1.781 | 0.101 | 1.781 | 420 |
| alinurdin_time_series_benchmark_eia_hourly_madrid_15min_webtris_15min_usgs_daily_usgs_continuous_hydat_daily_bom_daily_c__h/long/target | tuning | 1.873 | 0.100 | 1.873 | 252 |
| alinurdin_time_series_benchmark_eia_hourly_madrid_15min_webtris_15min_usgs_daily_usgs_continuous_hydat_daily_bom_daily_c__h/long/cov | tuning | 1.873 | 0.100 | 1.873 | 252 |
| alinurdin_time_series_benchmark_eia_hourly_madrid_15min_webtris_15min_usgs_daily_usgs_continuous_hydat_daily_bom_daily_c__d/short/target | tuning | 2.118 | 0.092 | 2.118 | 282 |
| alinurdin_time_series_benchmark_eia_hourly_madrid_15min_webtris_15min_usgs_daily_usgs_continuous_hydat_daily_bom_daily_c__d/short/cov | tuning | 2.118 | 0.092 | 2.118 | 282 |
| alinurdin_time_series_benchmark_eia_hourly_madrid_15min_webtris_15min_usgs_daily_usgs_continuous_hydat_daily_bom_daily_c__wsun/short/target | tuning | 1.533 | 0.090 | 1.533 | 144 |
| alinurdin_time_series_benchmark_eia_hourly_madrid_15min_webtris_15min_usgs_daily_usgs_continuous_hydat_daily_bom_daily_c__wsun/short/cov | tuning | 1.533 | 0.090 | 1.533 | 144 |
| angelosaadeh_velib__h/short/target | tuning | 1.275 | 0.496 | 1.275 | 1000 |
| angelosaadeh_velib__h/short/cov | tuning | 1.275 | 0.496 | 1.275 | 1000 |
| approjects_us_layoffs_monthly_time_series_warn_act__ms/short/target | tuning | 0.597 | 0.456 | 0.597 | 20 |
| approjects_us_layoffs_monthly_time_series_warn_act__ms/short/cov | tuning | 0.597 | 0.456 | 0.597 | 20 |
| autogluon_chronos_datasets_dominick__wthu/short/target | tuning | 0.825 | 0.481 | 0.825 | 3000 |
| autogluon_chronos_datasets_mexico_city_bikes__h/short/target | tuning | 1.145 | 0.767 | 1.145 | 9860 |
| autogluon_chronos_datasets_mexico_city_bikes__h/medium/target | tuning | 0.567 | 0.991 | 0.567 | 4840 |
| autogluon_chronos_datasets_mexico_city_bikes__h/long/target | tuning | 0.637 | 1.150 | 0.637 | 2904 |
| autogluon_chronos_datasets_mexico_city_bikes__d/short/target | tuning | 0.945 | 0.900 | 0.945 | 2898 |
| autogluon_chronos_datasets_mexico_city_bikes__wsun/short/target | tuning | 2.940 | 1.052 | 2.940 | 1449 |
| autogluon_chronos_datasets_monash_cif_2016__me/short/target | tuning | 1.355 | 0.004 | 1.355 | 62 |
| autogluon_chronos_datasets_monash_nn5_weekly_nn5__d/short/target | tuning | 1.418 | 0.479 | 1.418 | 198 |
| autogluon_chronos_datasets_monash_nn5_weekly_nn5__wsun/short/target | tuning | 1.049 | 0.123 | 1.049 | 107 |
| autogluon_chronos_datasets_monash_pedestrian_counts__h/short/target | tuning | 0.393 | 0.317 | 0.393 | 1299 |
| autogluon_chronos_datasets_monash_pedestrian_counts__h/medium/target | tuning | 1.036 | 0.465 | 1.036 | 248 |
| autogluon_chronos_datasets_monash_pedestrian_counts__h/long/target | tuning | 1.135 | 0.458 | 1.135 | 184 |
| autogluon_chronos_datasets_monash_pedestrian_counts__d/short/target | tuning | 1.389 | 0.388 | 1.389 | 182 |
| autogluon_chronos_datasets_monash_pedestrian_counts__wsun/short/target | tuning | 7.153 | 1.104 | 7.153 | 58 |
| autogluon_chronos_datasets_monash_rideshare__h/short/target | tuning | 0.995 | 0.056 | 0.995 | 143 |
| autogluon_chronos_datasets_monash_rideshare__h/short/cov | tuning | 0.995 | 0.056 | 0.995 | 143 |
| autogluon_chronos_datasets_monash_traffic__h/short/target | tuning | 1.220 | 0.354 | 1.220 | 17240 |
| autogluon_chronos_datasets_monash_traffic__h/medium/target | tuning | 1.450 | 0.384 | 1.450 | 2586 |
| autogluon_chronos_datasets_monash_traffic__h/long/target | tuning | 1.709 | 0.460 | 1.709 | 1724 |
| autogluon_chronos_datasets_monash_traffic__d/short/target | tuning | 1.513 | 0.293 | 1.513 | 1724 |
| autogluon_chronos_datasets_monash_traffic__wsun/short/target | tuning | 1.945 | 0.133 | 1.945 | 862 |
| autogluon_chronos_datasets_monash_weather__d/short/target | tuning | 1.430 | 0.245 | 1.430 | 20000 |
| autogluon_chronos_datasets_monash_weather__wsun/short/target | tuning | 1.379 | 0.196 | 1.379 | 13000 |
| autogluon_chronos_datasets_taxi_30min_taxi_1h__30min/short/target | tuning | 1.022 | 0.425 | 1.022 | 1524 |
| autogluon_chronos_datasets_taxi_30min_taxi_1h__30min/medium/target | tuning | 1.198 | 0.619 | 1.198 | 508 |
| autogluon_chronos_datasets_taxi_30min_taxi_1h__h/short/target | tuning | 1.267 | 0.436 | 1.267 | 508 |
| autogluon_chronos_datasets_uber_tlc_hourly_uber_tlc_daily__h/short/target | tuning | 1.474 | 1.058 | 1.474 | 891 |
| autogluon_chronos_datasets_uber_tlc_hourly_uber_tlc_daily__h/medium/target | tuning | 1.329 | 0.978 | 1.329 | 99 |
| autogluon_chronos_datasets_uber_tlc_hourly_uber_tlc_daily__h/long/target | tuning | 1.631 | 1.159 | 1.631 | 99 |
| autogluon_chronos_datasets_uber_tlc_hourly_uber_tlc_daily__d/short/target | tuning | 1.618 | 0.385 | 1.618 | 99 |
| autogluon_chronos_datasets_weatherbench_daily__d/short/target | tuning | 1.863 | 0.011 | 1.863 | 19860 |
| autogluon_chronos_datasets_weatherbench_daily__wsun/short/target | tuning | 1.305 | 0.010 | 1.305 | 19860 |
| autogluon_chronos_datasets_wiki_daily_100k__d/short/target | tuning | 1.257 | 0.341 | 1.257 | 9000 |
| autogluon_chronos_datasets_wiki_daily_100k__wsun/short/target | tuning | 1.311 | 0.326 | 1.311 | 4000 |
| autonlab_timeseries_pile_forecasting__h/short/target | tuning | 1.220 | 0.354 | 1.220 | 17240 |
| autonlab_timeseries_pile_forecasting__h/medium/target | tuning | 1.450 | 0.384 | 1.450 | 2586 |
| autonlab_timeseries_pile_forecasting__h/long/target | tuning | 1.709 | 0.460 | 1.709 | 1724 |
| autonlab_timeseries_pile_forecasting__d/short/target | tuning | 1.547 | 0.290 | 1.547 | 1724 |
| autonlab_timeseries_pile_forecasting__wsun/short/target | tuning | 1.946 | 0.131 | 1.946 | 862 |
| autonlab_timeseries_pile_forecasting_autoformer__h/short/target | tuning | 1.220 | 0.354 | 1.220 | 17240 |
| autonlab_timeseries_pile_forecasting_autoformer__h/medium/target | tuning | 1.450 | 0.384 | 1.450 | 2586 |
| autonlab_timeseries_pile_forecasting_autoformer__h/long/target | tuning | 1.709 | 0.460 | 1.709 | 1724 |
| autonlab_timeseries_pile_forecasting_autoformer__d/short/target | tuning | 1.547 | 0.290 | 1.547 | 1724 |
| autonlab_timeseries_pile_forecasting_autoformer__wsun/short/target | tuning | 1.946 | 0.131 | 1.946 | 862 |
| battery_life_batterylife_raw_zn_ion_na_ion_calb__s/short/target | tuning | 1.029 | 0.043 | 1.029 | 340 |
| battery_life_batterylife_raw_zn_ion_na_ion_calb__s/medium/target | tuning | 1.010 | 0.043 | 1.010 | 170 |
| battery_life_batterylife_raw_zn_ion_na_ion_calb__s/long/target | tuning | 1.013 | 0.043 | 1.013 | 102 |
| bayway_italian_open_timeseries__h/short/target | tuning | 1.019 | 0.272 | 1.019 | 160 |
| bayway_italian_open_timeseries__h/medium/target | tuning | 1.306 | 0.377 | 1.306 | 40 |
| bayway_italian_open_timeseries__h/long/target | tuning | 1.330 | 0.387 | 1.330 | 24 |
| bayway_italian_open_timeseries__d/short/target | tuning | 1.471 | 0.280 | 1.471 | 42 |
| bayway_italian_open_timeseries__wsun/short/target | tuning | 1.718 | 0.240 | 1.718 | 33 |
| ceepr_bit_dianchi_water__wsun/short/target | tuning | 5.182 | 0.255 | 5.182 | 8 |
| ceepr_bit_dianchi_water__wsun/short/cov | tuning | 5.182 | 0.255 | 5.182 | 8 |
| cmu_ospo_data_for_extended_range_water_demand_forecasts_driven_by_subseasonal_to_seasonal__ms/short/target | tuning | 0.943 | 0.025 | 0.943 | 3 |
| copernicus_argentina_estado_hidrico_argentina_2001_2026__ms/short/target | tuning | 0.980 | 0.476 | 0.980 | 1054 |
| copernicus_argentina_estado_hidrico_argentina_2001_2026__ms/short/cov | tuning | 0.980 | 0.476 | 0.980 | 1054 |
| copernicus_argentina_parana_alturas_historicas_1900_2026__d/short/target | tuning | 5.152 | 0.162 | 5.152 | 360 |
| copernicus_argentina_parana_alturas_historicas_1900_2026__d/short/cov | tuning | 5.152 | 0.162 | 5.152 | 360 |
| cosuleabianca_eea_pm25_forecasting__h/short/target | tuning | 1.455 | 0.754 | 1.455 | 340 |
| cosuleabianca_eea_pm25_forecasting__h/short/cov | tuning | 1.455 | 0.754 | 1.455 | 340 |
| cosuleabianca_eea_pm25_forecasting__d/short/target | tuning | 1.709 | 0.556 | 1.709 | 60 |
| cosuleabianca_eea_pm25_forecasting__d/short/cov | tuning | 1.709 | 0.556 | 1.709 | 60 |
| cosuleabianca_eea_pm25_forecasting__wsun/short/target | tuning | 1.080 | 0.402 | 1.080 | 63 |
| cosuleabianca_eea_pm25_forecasting__wsun/short/cov | tuning | 1.080 | 0.402 | 1.080 | 63 |
| datonic_spain_aemet_historical_weather__d/short/target | tuning | 1.896 | 0.014 | 1.896 | 280 |
| datonic_spain_aemet_historical_weather__wsun/short/target | tuning | 1.386 | 0.014 | 1.386 | 280 |
| datonic_world_development_indicators_default__ys/short/target | tuning | 6.699 | 0.131 | 6.699 | 943 |
| dhivakarnaidusunkari_mandi_india_daily_prices_2001_2026__d/short/target | tuning | 10.710 | 0.051 | 10.710 | 10 |
| dhivakarnaidusunkari_mandi_india_daily_prices_2001_2026__d/short/cov | tuning | 10.710 | 0.051 | 10.710 | 10 |
| dhivakarnaidusunkari_mandi_india_daily_prices_2001_2026__wsun/short/target | tuning | 4.023 | 0.178 | 4.023 | 60 |
| dhivakarnaidusunkari_mandi_india_daily_prices_2001_2026__wsun/short/cov | tuning | 4.023 | 0.178 | 4.023 | 60 |
| donny_guo_airline_delay__d/short/target | tuning | 4.717 | 0.153 | 4.717 | 1011 |
| donny_guo_airline_delay__d/short/cov | tuning | 4.717 | 0.153 | 4.717 | 1011 |
| donny_guo_airline_delay__wsun/short/target | tuning | 4.589 | 0.098 | 4.589 | 1115 |
| donny_guo_airline_delay__wsun/short/cov | tuning | 4.589 | 0.098 | 4.589 | 1115 |
| duckermaster_thaiclim_district__d/short/target | tuning | 1.501 | 1.124 | 1.501 | 18200 |
| duckermaster_thaiclim_district__d/short/cov | tuning | 1.501 | 1.124 | 1.501 | 18200 |
| duckermaster_thaiclim_district__wsun/short/target | tuning | 1.467 | 1.266 | 1.467 | 14550 |
| duckermaster_thaiclim_district__wsun/short/cov | tuning | 1.467 | 1.266 | 1.467 | 14550 |
| electricsheepafrica_africa_wfp_food_prices_for_uganda__ms/short/target | tuning | 1.205 | 0.140 | 1.205 | 9 |
| electricsheepafrica_africa_wfp_food_prices_for_uganda__ms/short/cov | tuning | 1.205 | 0.140 | 1.205 | 9 |
| elskow_weather4cast__d/short/target | tuning | 1.893 | 0.061 | 1.893 | 60 |
| elskow_weather4cast__d/short/cov | tuning | 1.893 | 0.061 | 1.893 | 60 |
| elskow_weather4cast__wsun/short/target | tuning | 1.313 | 0.055 | 1.313 | 30 |
| elskow_weather4cast__wsun/short/cov | tuning | 1.313 | 0.055 | 1.313 | 30 |
| emperor_mew_ooni_censorship_historical__d/short/target | tuning | 2.184 | 0.201 | 2.184 | 3269 |
| emperor_mew_ooni_censorship_historical__d/short/cov | tuning | 2.184 | 0.201 | 2.184 | 3269 |
| emperor_mew_ooni_censorship_historical__wsun/short/target | tuning | 2.367 | 0.205 | 2.367 | 1635 |
| emperor_mew_ooni_censorship_historical__wsun/short/cov | tuning | 2.367 | 0.205 | 2.367 | 1635 |
| emulated_inc_sci_booster_gmps_regulation_forecast__5s/short/target | tuning | 1.335 | 0.160 | 1.335 | 12260 |
| emulated_inc_sci_booster_gmps_regulation_forecast__5s/medium/target | tuning | 2.046 | 0.163 | 2.046 | 1839 |
| emulated_inc_sci_booster_gmps_regulation_forecast__5s/long/target | tuning | 2.177 | 0.163 | 2.177 | 1226 |
| emulated_inc_sci_lake_temperature_profiles_unmonitored__d/short/target | tuning | 1.463 | 0.209 | 1.463 | 20000 |
| emulated_inc_sci_lake_temperature_profiles_unmonitored__wsun/short/target | tuning | 1.425 | 0.175 | 1.425 | 20000 |
| emulated_inc_sci_streamflow_ungauged_new_country__wsun/short/target | tuning | 1.608 | 0.817 | 1.608 | 16380 |
| emulated_inc_sci_streamflow_ungauged_new_country__wsun/short/cov | tuning | 1.608 | 0.817 | 1.608 | 16380 |
| epi_eval_cdc_nrevss_rsv__wsat/short/target | tuning | 5.882 | 1.083 | 5.882 | 60 |
| epi_eval_cdc_nrevss_rsv__wsat/short/cov | tuning | 5.882 | 1.083 | 5.882 | 60 |
| epi_eval_cdc_nssp__wsat/short/target | tuning | 3.644 | 0.816 | 3.644 | 50 |
| epi_eval_wikipedia_pageviews__d/short/target | tuning | 1.030 | 0.141 | 1.030 | 13 |
| epi_eval_wikipedia_pageviews__wsun/short/target | tuning | 0.911 | 0.141 | 0.911 | 7 |
| feregrino_london_cycles__d/short/target | tuning | 2.702 | 0.840 | 2.702 | 1 |
| feregrino_london_cycles__d/short/cov | tuning | 2.702 | 0.840 | 2.702 | 1 |
| feregrino_london_cycles__wsun/short/target | tuning | 1.122 | 0.318 | 1.122 | 4 |
| feregrino_london_cycles__wsun/short/cov | tuning | 1.122 | 0.318 | 1.122 | 4 |
| fidel_ts_california_iso__5min/short/target | tuning | 1.097 | 0.223 | 1.097 | 220 |
| fidel_ts_california_iso__5min/medium/target | tuning | 2.251 | 0.297 | 2.251 | 110 |
| fidel_ts_california_iso__5min/long/target | tuning | 2.335 | 0.312 | 2.335 | 66 |
| fidel_ts_california_iso__h/short/target | tuning | 2.497 | 0.258 | 2.497 | 88 |
| fidel_ts_california_iso__h/medium/target | tuning | 5.517 | 0.324 | 5.517 | 11 |
| fidel_ts_california_iso__h/long/target | tuning | 13.668 | 0.460 | 13.668 | 11 |
| fidel_ts_california_iso__d/short/target | tuning | 10.902 | 0.453 | 10.902 | 11 |
| food_ai_nexus_global_food_safety_incidents__d/short/target | tuning | 2.912 | 1.318 | 2.912 | 160 |
| food_ai_nexus_global_food_safety_incidents__wsun/short/target | tuning | 2.628 | 0.659 | 2.628 | 112 |
| fuel_prices_eu_european_fuel_prices__d/short/target | tuning | 0.903 | 0.028 | 0.903 | 96 |
| fuel_prices_eu_european_fuel_prices__d/short/cov | tuning | 0.903 | 0.028 | 0.903 | 96 |
| gft_ttm4hvac_source_all_train__15min/short/target | tuning | 0.972 | 0.350 | 0.972 | 60 |
| gft_ttm4hvac_source_all_train__15min/short/cov | tuning | 0.972 | 0.350 | 0.972 | 60 |
| gft_ttm4hvac_source_all_train__15min/medium/target | tuning | 1.189 | 0.419 | 1.189 | 21 |
| gft_ttm4hvac_source_all_train__15min/medium/cov | tuning | 1.189 | 0.419 | 1.189 | 21 |
| gft_ttm4hvac_source_all_train__15min/long/target | tuning | 1.365 | 0.464 | 1.365 | 12 |
| gft_ttm4hvac_source_all_train__15min/long/cov | tuning | 1.365 | 0.464 | 1.365 | 12 |
| gft_ttm4hvac_source_all_train__h/short/target | tuning | 1.105 | 0.378 | 1.105 | 54 |
| gft_ttm4hvac_source_all_train__h/short/cov | tuning | 1.105 | 0.378 | 1.105 | 54 |
| gft_ttm4hvac_source_all_train__h/medium/target | tuning | 1.115 | 0.383 | 1.115 | 3 |
| gft_ttm4hvac_source_all_train__h/medium/cov | tuning | 1.115 | 0.383 | 1.115 | 3 |
| gft_ttm4hvac_source_all_train__h/long/target | tuning | 1.219 | 0.426 | 1.219 | 3 |
| gft_ttm4hvac_source_all_train__h/long/cov | tuning | 1.219 | 0.426 | 1.219 | 3 |
| gft_ttm4hvac_source_all_train__d/short/target | tuning | 1.342 | 0.336 | 1.342 | 3 |
| gft_ttm4hvac_source_all_train__d/short/cov | tuning | 1.342 | 0.336 | 1.342 | 3 |
| gladcolor_hourly_population_us__h/short/target | tuning | 1.220 | 0.079 | 1.220 | 18000 |
| gladcolor_hourly_population_us__h/medium/target | tuning | 1.398 | 0.094 | 1.398 | 1000 |
| gladcolor_hourly_population_us__h/long/target | tuning | 1.345 | 0.087 | 1.345 | 1000 |
| gladcolor_hourly_population_us__d/short/target | tuning | 1.444 | 0.054 | 1.444 | 1000 |
| glucofmbench_glucofm_bench__5min/short/target | tuning | 0.932 | 0.237 | 0.932 | 804 |
| glucofmbench_glucofm_bench__5min/medium/target | tuning | 0.937 | 0.245 | 0.937 | 211 |
| glucofmbench_glucofm_bench__h/short/target | tuning | 0.990 | 0.297 | 0.990 | 38 |
| glucofmbench_glucofm_bench__d/short/target | tuning | 0.575 | 0.076 | 0.575 | 1 |
| hackermango_hyderabaduniversity_hvac_dataset__10min/short/target | tuning | 0.979 | 0.159 | 0.979 | 280 |
| hackermango_hyderabaduniversity_hvac_dataset__10min/medium/target | tuning | 1.565 | 0.246 | 1.565 | 28 |
| hackermango_hyderabaduniversity_hvac_dataset__10min/long/target | tuning | 2.047 | 0.333 | 2.047 | 14 |
| hackermango_hyderabaduniversity_hvac_dataset__h/short/target | tuning | 1.876 | 0.333 | 1.876 | 42 |
| hackermango_hyderabaduniversity_hvac_dataset__h/medium/target | tuning | 1.074 | 0.147 | 1.074 | 14 |
| hhs_official_nndss_weekly_data__wsat/short/target | tuning | 1.413 | 0.218 | 1.413 | 356 |
| hhs_official_weekly_counts_of_deaths_from_all_causes_by_jurisdi__wsat/short/target | tuning | 0.610 | 0.011 | 0.610 | 6 |
| hhs_official_weekly_rates_of_laboratory_confirmed_rsv_hospitali__wsat/short/target | tuning | 3.980 | 0.897 | 3.980 | 52 |
| hhs_official_weekly_rates_of_laboratory_confirmed_rsv_hospitali__wsat/short/cov | tuning | 3.980 | 0.897 | 3.980 | 52 |
| hhs_official_weekly_united_states_covid_19_hospitalization_metr__wsat/short/target | tuning | 1.982 | 0.736 | 1.982 | 132 |
| hhs_official_weekly_united_states_covid_19_hospitalization_metr__wsat/short/cov | tuning | 1.982 | 0.736 | 1.982 | 132 |
| hq_bench_quitobench_hour_min_r577__h/short/target | tuning | 1.061 | 0.115 | 1.061 | 2000 |
| hq_bench_quitobench_hour_min_r577__h/short/cov | tuning | 1.061 | 0.115 | 1.061 | 2000 |
| hq_bench_quitobench_hour_min_r577__d/short/target | tuning | 2.186 | 0.194 | 2.186 | 814 |
| hq_bench_quitobench_hour_min_r577__d/short/cov | tuning | 2.186 | 0.194 | 2.186 | 814 |
| hq_bench_quitobench_hour_min_r577__wsun/short/target | tuning | 1.696 | 0.171 | 1.696 | 407 |
| hq_bench_quitobench_hour_min_r577__wsun/short/cov | tuning | 1.696 | 0.171 | 1.696 | 407 |
| hsilvosa_safecast_radiation__5s/short/target | tuning | 1.495 | 0.374 | 1.495 | 1207 |
| j_marcos_gs_brazil_solar_generation__h/short/target | tuning | 1.683 | 0.093 | 1.683 | 20 |
| j_marcos_gs_brazil_solar_generation__h/medium/target | tuning | 1.829 | 0.114 | 1.829 | 10 |
| j_marcos_gs_brazil_solar_generation__h/long/target | tuning | 2.169 | 0.140 | 2.169 | 6 |
| j_marcos_gs_brazil_solar_generation__d/short/target | tuning | 2.437 | 0.129 | 2.437 | 6 |
| j_marcos_gs_brazil_solar_generation__wsun/short/target | tuning | 2.981 | 0.126 | 2.981 | 3 |
| jainam03_rescast_100k_ecobee__15min/short/target | tuning | 0.606 | 0.005 | 0.606 | 80 |
| jainam03_rescast_100k_ecobee__15min/short/cov | tuning | 0.606 | 0.005 | 0.606 | 80 |
| jainam03_rescast_100k_ecobee__15min/medium/target | tuning | 1.072 | 0.009 | 1.072 | 28 |
| jainam03_rescast_100k_ecobee__15min/medium/cov | tuning | 1.072 | 0.009 | 1.072 | 28 |
| jainam03_rescast_100k_ecobee__15min/long/target | tuning | 1.007 | 0.008 | 1.007 | 16 |
| jainam03_rescast_100k_ecobee__15min/long/cov | tuning | 1.007 | 0.008 | 1.007 | 16 |
| jainam03_rescast_100k_ecobee__h/short/target | tuning | 0.801 | 0.007 | 0.801 | 72 |
| jainam03_rescast_100k_ecobee__h/short/cov | tuning | 0.801 | 0.007 | 0.801 | 72 |
| jainam03_rescast_100k_ecobee__h/medium/target | tuning | 1.327 | 0.011 | 1.327 | 4 |
| jainam03_rescast_100k_ecobee__h/medium/cov | tuning | 1.327 | 0.011 | 1.327 | 4 |
| jainam03_rescast_100k_ecobee__h/long/target | tuning | 1.493 | 0.012 | 1.493 | 4 |
| jainam03_rescast_100k_ecobee__h/long/cov | tuning | 1.493 | 0.012 | 1.493 | 4 |
| jainam03_rescast_100k_ecobee__d/short/target | tuning | 1.645 | 0.011 | 1.645 | 4 |
| jainam03_rescast_100k_ecobee__d/short/cov | tuning | 1.645 | 0.011 | 1.645 | 4 |
| jainam03_rescast_100k_heapo_timeseries_data__15min/short/target | tuning | 1.123 | 0.128 | 1.123 | 19760 |
| jainam03_rescast_100k_heapo_timeseries_data__15min/medium/target | tuning | 1.363 | 0.153 | 1.363 | 7731 |
| jainam03_rescast_100k_heapo_timeseries_data__15min/long/target | tuning | 1.418 | 0.142 | 1.418 | 4833 |
| jainam03_rescast_100k_heapo_timeseries_data__h/short/target | tuning | 1.280 | 0.135 | 1.280 | 19348 |
| jainam03_rescast_100k_heapo_timeseries_data__h/medium/target | tuning | 1.363 | 0.137 | 1.363 | 1923 |
| jainam03_rescast_100k_heapo_timeseries_data__h/long/target | tuning | 1.698 | 0.185 | 1.698 | 962 |
| jainam03_rescast_100k_heapo_timeseries_data__d/short/target | tuning | 2.431 | 0.173 | 2.431 | 953 |
| jainam03_rescast_100k_heapo_timeseries_data__wsun/short/target | tuning | 1.489 | 0.073 | 1.489 | 696 |
| jainam03_rescast_100k_ideal__15min/short/target | tuning | 0.786 | 0.027 | 0.786 | 19980 |
| jainam03_rescast_100k_ideal__15min/medium/target | tuning | 0.972 | 0.038 | 0.972 | 3996 |
| jainam03_rescast_100k_ideal__15min/long/target | tuning | 0.924 | 0.033 | 0.924 | 1998 |
| jainam03_rescast_100k_ideal__h/short/target | tuning | 0.853 | 0.028 | 0.853 | 9990 |
| jainam03_rescast_100k_ideal__h/medium/target | tuning | 1.055 | 0.041 | 1.055 | 999 |
| jainam03_rescast_100k_ideal__h/long/target | tuning | 1.083 | 0.045 | 1.083 | 968 |
| jainam03_rescast_100k_ideal__d/short/target | tuning | 1.332 | 0.041 | 1.332 | 828 |
| jainam03_rescast_100k_ideal__wsun/short/target | tuning | 1.375 | 0.049 | 1.375 | 149 |
| jainam03_rescast_100k_refit__15min/short/target | tuning | 0.726 | 0.419 | 0.726 | 400 |
| jainam03_rescast_100k_refit__15min/medium/target | tuning | 0.833 | 0.518 | 0.833 | 180 |
| jainam03_rescast_100k_refit__15min/long/target | tuning | 0.872 | 0.538 | 0.872 | 120 |
| jainam03_rescast_100k_refit__h/short/target | tuning | 0.745 | 0.379 | 0.745 | 400 |
| jainam03_rescast_100k_refit__h/medium/target | tuning | 1.109 | 0.567 | 1.109 | 40 |
| jainam03_rescast_100k_refit__h/long/target | tuning | 0.868 | 0.435 | 0.868 | 20 |
| jainam03_rescast_100k_refit__d/short/target | tuning | 1.224 | 0.282 | 1.224 | 20 |
| jainam03_rescast_100k_refit__wsun/short/target | tuning | 1.618 | 0.304 | 1.618 | 19 |
| jhumbl_orr_station_usage__ys/short/target | tuning | 5.817 | 0.434 | 5.817 | 998 |
| jhumbl_orr_station_usage__ys/short/cov | tuning | 5.817 | 0.434 | 5.817 | 998 |
| jhumbl_police_crime_london__d/short/target | tuning | 1.370 | 1.939 | 1.370 | 10172 |
| jhumbl_police_crime_london__wsun/short/target | tuning | 1.021 | 1.797 | 1.021 | 5371 |
| jotone_arpa_osmer_weatherstations__h/short/target | tuning | 0.679 | nan | 0.679 | 240 |
| jotone_arpa_osmer_weatherstations__h/short/cov | tuning | 0.679 | nan | 0.679 | 240 |
| jotone_arpa_osmer_weatherstations__h/medium/target | tuning | 0.489 | 1.275 | 0.489 | 10 |
| jotone_arpa_osmer_weatherstations__h/medium/cov | tuning | 0.489 | 1.275 | 0.489 | 10 |
| jotone_arpa_osmer_weatherstations__h/long/target | tuning | 0.399 | 1.003 | 0.399 | 6 |
| jotone_arpa_osmer_weatherstations__h/long/cov | tuning | 0.399 | 1.003 | 0.399 | 6 |
| jotone_arpa_osmer_weatherstations__d/short/target | tuning | 0.637 | 1.012 | 0.637 | 42 |
| jotone_arpa_osmer_weatherstations__d/short/cov | tuning | 0.637 | 1.012 | 0.637 | 42 |
| jotone_arpa_osmer_weatherstations__wsun/short/target | tuning | 0.811 | 1.019 | 0.811 | 111 |
| jotone_arpa_osmer_weatherstations__wsun/short/cov | tuning | 0.811 | 1.019 | 0.811 | 111 |
| juliensimon_auroral_electrojet_index__d/short/target | tuning | 0.000 | 0.000 | 0.000 | 7 |
| juliensimon_auroral_electrojet_index__wsun/short/target | tuning | 0.000 | 0.000 | 0.000 | 3 |
| juliensimon_celestrak_space_weather__wsun/short/target | tuning | 1.814 | 0.081 | 1.814 | 580 |
| juliensimon_dst_index__h/short/target | tuning | 1.066 | 1.204 | 1.066 | 20 |
| juliensimon_geomagnetic_kp_index__3h/short/target | tuning | 2.443 | 0.370 | 2.443 | 9 |
| juliensimon_geomagnetic_kp_index__3h/medium/target | tuning | 0.992 | 0.396 | 0.992 | 3 |
| juliensimon_geomagnetic_kp_index__d/short/target | tuning | 1.262 | 0.279 | 1.262 | 3 |
| juliensimon_goes_xray_flux__min/short/target | tuning | 1.980 | 14.472 | 1.980 | 20 |
| juliensimon_goes_xray_flux__h/short/target | tuning | 1.367 | 1.040 | 1.367 | 1 |
| juliensimon_nasa_maven_kp_insitu__8s/short/target | tuning | 1.709 | 0.000 | 1.709 | 20 |
| juliensimon_nasa_maven_kp_insitu__8s/medium/target | tuning | 1.101 | 0.000 | 1.101 | 10 |
| juliensimon_nasa_maven_kp_insitu__8s/long/target | tuning | 1.548 | 0.000 | 1.548 | 6 |
| juliensimon_omni_solar_wind_parameters__d/short/target | tuning | 3.975 | 0.112 | 3.975 | 48 |
| juliensimon_omni_solar_wind_parameters__wsun/short/target | tuning | 1.549 | 0.064 | 1.549 | 24 |
| jwigginton_timeseries_1mn_sp500_default__min/short/target | tuning | 2.963 | 0.002 | 2.963 | 1 |
| jwigginton_timeseries_1mn_sp500_default__min/short/cov | tuning | 2.963 | 0.002 | 2.963 | 1 |
| khajlk_copenhagen_air_quality_hourly__h/short/target | tuning | 1.083 | 0.161 | 1.083 | 20 |
| khajlk_copenhagen_air_quality_hourly__h/short/cov | tuning | 1.083 | 0.161 | 1.083 | 20 |
| khajlk_copenhagen_air_quality_hourly__h/medium/target | tuning | 1.727 | 0.180 | 1.727 | 10 |
| khajlk_copenhagen_air_quality_hourly__h/medium/cov | tuning | 1.727 | 0.180 | 1.727 | 10 |
| khajlk_copenhagen_air_quality_hourly__h/long/target | tuning | 2.150 | 0.212 | 2.150 | 6 |
| khajlk_copenhagen_air_quality_hourly__h/long/cov | tuning | 2.150 | 0.212 | 2.150 | 6 |
| khajlk_copenhagen_air_quality_hourly__d/short/target | tuning | 2.384 | 0.173 | 2.384 | 6 |
| khajlk_copenhagen_air_quality_hourly__d/short/cov | tuning | 2.384 | 0.173 | 2.384 | 6 |
| khajlk_copenhagen_air_quality_hourly__wsun/short/target | tuning | 1.956 | 0.227 | 1.956 | 3 |
| khajlk_copenhagen_air_quality_hourly__wsun/short/cov | tuning | 1.956 | 0.227 | 1.956 | 3 |
| leadlagforecasting_llf_github_default__d/short/target | tuning | 0.331 | 1.353 | 0.331 | 6951 |
| leadlagforecasting_llf_github_default__d/short/cov | tuning | 0.331 | 1.353 | 0.331 | 6951 |
| leadlagforecasting_llf_github_default__wsun/short/target | tuning | 0.309 | 0.795 | 0.309 | 3920 |
| leadlagforecasting_llf_github_default__wsun/short/cov | tuning | 0.309 | 0.795 | 0.309 | 3920 |
| marplan6_batgrad_pozzato_2022__s/short/target | tuning | 0.705 | 0.001 | 0.705 | 60 |
| marplan6_batgrad_pozzato_2022__s/medium/target | tuning | 0.974 | 0.003 | 0.974 | 30 |
| marplan6_batgrad_pozzato_2022__s/long/target | tuning | 0.876 | 0.002 | 0.876 | 18 |
| matrosovdani_moscow_parking_occupancy__1799s/short/target | tuning | 0.462 | nan | 0.462 | 60 |
| matrosovdani_moscow_parking_occupancy__1799s/short/cov | tuning | 0.462 | nan | 0.462 | 60 |
| matrosovdani_moscow_parking_occupancy__d/short/target | tuning | 3.362 | 0.259 | 3.362 | 188 |
| matrosovdani_moscow_parking_occupancy__d/short/cov | tuning | 3.362 | 0.259 | 3.362 | 188 |
| meganariley_daily_aqi__h/short/target | tuning | 1.151 | 0.597 | 1.151 | 5080 |
| meganariley_daily_aqi__h/short/cov | tuning | 1.151 | 0.597 | 1.151 | 5080 |
| meganariley_daily_aqi__h/medium/target | tuning | 1.307 | 0.626 | 1.307 | 1500 |
| meganariley_daily_aqi__h/medium/cov | tuning | 1.307 | 0.626 | 1.307 | 1500 |
| meganariley_daily_aqi__h/long/target | tuning | 1.392 | 0.653 | 1.392 | 918 |
| meganariley_daily_aqi__h/long/cov | tuning | 1.392 | 0.653 | 1.392 | 918 |
| meganariley_daily_aqi__d/short/target | tuning | 1.428 | 0.531 | 1.428 | 2553 |
| meganariley_daily_aqi__d/short/cov | tuning | 1.428 | 0.531 | 1.428 | 2553 |
| meganariley_daily_aqi__wsun/short/target | tuning | 1.243 | 0.433 | 1.243 | 2038 |
| meganariley_daily_aqi__wsun/short/cov | tuning | 1.243 | 0.433 | 1.243 | 2038 |
| meteoswiss_peakweather__10min/short/target | tuning | 0.527 | 0.098 | 0.527 | 19680 |
| meteoswiss_peakweather__10min/medium/target | tuning | 1.052 | 0.127 | 1.052 | 7808 |
| meteoswiss_peakweather__10min/long/target | tuning | 1.067 | 0.128 | 1.067 | 4880 |
| meteoswiss_peakweather__h/short/target | tuning | 1.040 | 0.100 | 1.040 | 13762 |
| meteoswiss_peakweather__h/medium/target | tuning | 1.213 | 0.125 | 1.213 | 986 |
| meteoswiss_peakweather__h/long/target | tuning | 1.106 | 0.107 | 1.106 | 983 |
| meteoswiss_peakweather__d/short/target | tuning | 1.294 | 0.072 | 1.294 | 983 |
| misikoff_zillow_for_sale_listings__wsat/short/target | tuning | 1.049 | 0.073 | 1.049 | 2622 |
| misikoff_zillow_home_values__me/short/target | tuning | 2.086 | 0.084 | 2.086 | 1996 |
| modelpulse_model_pulse_data__d/short/target | tuning | 12.100 | 0.141 | 12.100 | 484 |
| modelpulse_model_pulse_data__d/short/cov | tuning | 12.100 | 0.141 | 12.100 | 484 |
| modelpulse_model_pulse_data__wsun/short/target | tuning | 3.393 | 0.133 | 3.393 | 651 |
| modelpulse_model_pulse_data__wsun/short/cov | tuning | 3.393 | 0.133 | 3.393 | 651 |
| nakas_powderbench_truth__d/short/target | tuning | 0.728 | 3.982 | 0.728 | 5532 |
| nakas_powderbench_truth__d/short/cov | tuning | 0.728 | 3.982 | 0.728 | 5532 |
| nakas_powderbench_truth__wsun/short/target | tuning | 0.938 | 1.442 | 0.938 | 3510 |
| nakas_powderbench_truth__wsun/short/cov | tuning | 0.938 | 1.442 | 0.938 | 3510 |
| national_digital_act_air_quality__wsun/short/target | tuning | 1.157 | 0.490 | 1.157 | 6 |
| national_digital_act_air_quality__wsun/short/cov | tuning | 1.157 | 0.490 | 1.157 | 6 |
| national_digital_au_births_by_age_of_mother__ys/short/target | tuning | 2.805 | 0.086 | 2.805 | 148 |
| national_digital_nt_crocodile_captures__ms/short/target | tuning | 0.724 | 0.405 | 0.724 | 10 |
| national_digital_nt_fuel_prices_history__d/short/target | tuning | 10.996 | 0.013 | 10.996 | 182 |
| national_digital_nt_fuel_prices_history__d/short/cov | tuning | 10.996 | 0.013 | 10.996 | 182 |
| national_digital_nt_fuel_prices_history__wsun/short/target | tuning | nan | 0.018 | nan | 116 |
| national_digital_nt_fuel_prices_history__wsun/short/cov | tuning | nan | 0.018 | nan | 116 |
| nevirim_wikipedia_attention_13lang__d/short/target | tuning | 1.079 | 0.129 | 1.079 | 117 |
| nevirim_wikipedia_attention_13lang__wsun/short/target | tuning | 1.170 | 0.124 | 1.170 | 91 |
| nju_zhangsq_rmisc_amarketchina__d/short/target | tuning | 3.240 | 0.066 | 3.240 | 2040 |
| nju_zhangsq_rmisc_amarketchina__d/short/cov | tuning | 3.240 | 0.066 | 3.240 | 2040 |
| nju_zhangsq_rmisc_amarketchina__wsun/short/target | tuning | 1.986 | 0.089 | 1.986 | 843 |
| nju_zhangsq_rmisc_amarketchina__wsun/short/cov | tuning | 1.986 | 0.089 | 1.986 | 843 |
| nju_zhangsq_rmisc_applianceenergy__10min/short/target | tuning | 1.102 | 0.537 | 1.102 | 20 |
| nju_zhangsq_rmisc_applianceenergy__10min/short/cov | tuning | 1.102 | 0.537 | 1.102 | 20 |
| nju_zhangsq_rmisc_applianceenergy__10min/medium/target | tuning | 0.971 | 0.606 | 0.971 | 4 |
| nju_zhangsq_rmisc_applianceenergy__10min/medium/cov | tuning | 0.971 | 0.606 | 0.971 | 4 |
| nju_zhangsq_rmisc_applianceenergy__10min/long/target | tuning | 0.867 | 0.538 | 0.867 | 2 |
| nju_zhangsq_rmisc_applianceenergy__10min/long/cov | tuning | 0.867 | 0.538 | 0.867 | 2 |
| nju_zhangsq_rmisc_applianceenergy__h/short/target | tuning | 0.965 | 0.494 | 0.965 | 6 |
| nju_zhangsq_rmisc_applianceenergy__h/short/cov | tuning | 0.965 | 0.494 | 0.965 | 6 |
| nju_zhangsq_rmisc_applianceenergy__h/medium/target | tuning | 0.778 | 0.437 | 0.778 | 1 |
| nju_zhangsq_rmisc_applianceenergy__h/medium/cov | tuning | 0.778 | 0.437 | 0.778 | 1 |
| nju_zhangsq_rmisc_applianceenergy__h/long/target | tuning | 0.668 | 0.383 | 0.668 | 1 |
| nju_zhangsq_rmisc_applianceenergy__h/long/cov | tuning | 0.668 | 0.383 | 0.668 | 1 |
| nju_zhangsq_rmisc_bts__10min/short/target | tuning | 0.000 | 0.000 | 0.000 | 20 |
| nju_zhangsq_rmisc_bts__10min/short/cov | tuning | 0.000 | 0.000 | 0.000 | 20 |
| nju_zhangsq_rmisc_bts__h/short/target | tuning | 0.000 | 0.000 | 0.000 | 17 |
| nju_zhangsq_rmisc_bts__h/short/cov | tuning | 0.000 | 0.000 | 0.000 | 17 |
| nju_zhangsq_rmisc_bts__h/medium/target | tuning | 0.000 | 0.000 | 0.000 | 1 |
| nju_zhangsq_rmisc_bts__h/medium/cov | tuning | 0.000 | 0.000 | 0.000 | 1 |
| nju_zhangsq_rmisc_bts__h/long/target | tuning | 0.000 | 0.000 | 0.000 | 1 |
| nju_zhangsq_rmisc_bts__h/long/cov | tuning | 0.000 | 0.000 | 0.000 | 1 |
| nju_zhangsq_rmisc_bts__d/short/target | tuning | 0.000 | 0.000 | 0.000 | 1 |
| nju_zhangsq_rmisc_bts__d/short/cov | tuning | 0.000 | 0.000 | 0.000 | 1 |
| nju_zhangsq_rmisc_buildingsbenchrealcsv__h/short/target | tuning | 1.072 | 0.367 | 1.072 | 80 |
| nju_zhangsq_rmisc_buildingsbenchrealcsv__h/medium/target | tuning | 1.125 | 0.369 | 1.125 | 8 |
| nju_zhangsq_rmisc_buildingsbenchrealcsv__h/long/target | tuning | 1.274 | 0.434 | 1.274 | 8 |
| nju_zhangsq_rmisc_buildingsbenchrealcsv__d/short/target | tuning | 1.366 | 0.246 | 1.366 | 8 |
| nju_zhangsq_rmisc_dominick__w/short/target | tuning | 0.999 | 0.036 | 0.999 | 4 |
| nju_zhangsq_rmisc_ebayserver__s/short/target | tuning | 1.156 | 0.109 | 1.156 | 20 |
| nju_zhangsq_rmisc_ebayserver__s/short/cov | tuning | 1.156 | 0.109 | 1.156 | 20 |
| nju_zhangsq_rmisc_ebayserver__s/medium/target | tuning | 0.814 | 0.073 | 0.814 | 10 |
| nju_zhangsq_rmisc_ebayserver__s/medium/cov | tuning | 0.814 | 0.073 | 0.814 | 10 |
| nju_zhangsq_rmisc_ebayserver__s/long/target | tuning | 0.866 | 0.078 | 0.866 | 6 |
| nju_zhangsq_rmisc_ebayserver__s/long/cov | tuning | 0.866 | 0.078 | 0.866 | 6 |
| nju_zhangsq_rmisc_era5hourlysinglelevels__h/short/target | tuning | 0.960 | 0.498 | 0.960 | 820 |
| nju_zhangsq_rmisc_era5hourlysinglelevels__h/short/cov | tuning | 0.960 | 0.498 | 0.960 | 820 |
| nju_zhangsq_rmisc_era5hourlysinglelevels__h/medium/target | tuning | 1.629 | 0.814 | 1.629 | 410 |
| nju_zhangsq_rmisc_era5hourlysinglelevels__h/medium/cov | tuning | 1.629 | 0.814 | 1.629 | 410 |
| nju_zhangsq_rmisc_era5hourlysinglelevels__h/long/target | tuning | 1.434 | 0.705 | 1.434 | 246 |
| nju_zhangsq_rmisc_era5hourlysinglelevels__h/long/cov | tuning | 1.434 | 0.705 | 1.434 | 246 |
| nju_zhangsq_rmisc_era5hourlysinglelevels__d/short/target | tuning | 1.621 | 0.669 | 1.621 | 246 |
| nju_zhangsq_rmisc_era5hourlysinglelevels__d/short/cov | tuning | 1.621 | 0.669 | 1.621 | 246 |
| nju_zhangsq_rmisc_era5hourlysinglelevels__wsun/short/target | tuning | 1.482 | 0.719 | 1.482 | 123 |
| nju_zhangsq_rmisc_era5hourlysinglelevels__wsun/short/cov | tuning | 1.482 | 0.719 | 1.482 | 123 |
| nju_zhangsq_rmisc_globalclimatechange__ms/short/target | tuning | 1.116 | 0.078 | 1.116 | 9696 |
| nju_zhangsq_rmisc_globalclimatechange__ms/short/cov | tuning | 1.116 | 0.078 | 1.116 | 9696 |
| nju_zhangsq_rmisc_hungarianchickenpoxcases__wmon/short/target | tuning | 1.237 | 0.920 | 1.237 | 6 |
| nju_zhangsq_rmisc_keplerlightcurves__30min/short/target | tuning | 1.007 | 1.180 | 1.007 | 8982 |
| nju_zhangsq_rmisc_keplerlightcurves__30min/medium/target | tuning | 1.143 | 1.297 | 1.143 | 998 |
| nju_zhangsq_rmisc_keplerlightcurves__30min/long/target | tuning | 1.141 | 1.296 | 1.141 | 998 |
| nju_zhangsq_rmisc_keplerlightcurves__h/short/target | tuning | 1.054 | 1.245 | 1.054 | 3992 |
| nju_zhangsq_rmisc_keplerlightcurves__h/medium/target | tuning | 1.158 | 1.355 | 1.158 | 998 |
| nju_zhangsq_rmisc_keplerlightcurves__h/long/target | tuning | 1.137 | 1.325 | 1.137 | 998 |
| nju_zhangsq_rmisc_metropt3__10s/short/target | tuning | 1.856 | 0.076 | 1.856 | 20 |
| nju_zhangsq_rmisc_metropt3__10s/short/cov | tuning | 1.856 | 0.076 | 1.856 | 20 |
| nju_zhangsq_rmisc_metropt3__10s/medium/target | tuning | 1.856 | 0.075 | 1.856 | 10 |
| nju_zhangsq_rmisc_metropt3__10s/medium/cov | tuning | 1.856 | 0.075 | 1.856 | 10 |
| nju_zhangsq_rmisc_metropt3__10s/long/target | tuning | 1.875 | 0.076 | 1.875 | 6 |
| nju_zhangsq_rmisc_metropt3__10s/long/cov | tuning | 1.875 | 0.076 | 1.875 | 6 |
| nju_zhangsq_rmisc_metrotraffic__d/short/target | tuning | 1.706 | 0.017 | 1.706 | 6 |
| nju_zhangsq_rmisc_metrotraffic__d/short/cov | tuning | 1.706 | 0.017 | 1.706 | 6 |
| nju_zhangsq_rmisc_metrotraffic__wsun/short/target | tuning | 1.825 | 0.024 | 1.825 | 3 |
| nju_zhangsq_rmisc_metrotraffic__wsun/short/cov | tuning | 1.825 | 0.024 | 1.825 | 3 |
| nju_zhangsq_rmisc_opsd_household__min/short/target | tuning | 0.649 | 1.540 | 0.649 | 20 |
| nju_zhangsq_rmisc_opsd_household__min/short/cov | tuning | 0.649 | 1.540 | 0.649 | 20 |
| nju_zhangsq_rmisc_opsd_household__min/medium/target | tuning | 0.525 | 1.087 | 0.525 | 10 |
| nju_zhangsq_rmisc_opsd_household__min/medium/cov | tuning | 0.525 | 1.087 | 0.525 | 10 |
| nju_zhangsq_rmisc_opsd_household__min/long/target | tuning | 0.533 | 0.936 | 0.533 | 6 |
| nju_zhangsq_rmisc_opsd_household__min/long/cov | tuning | 0.533 | 0.936 | 0.533 | 6 |
| nju_zhangsq_rmisc_opsd_household__h/short/target | tuning | 0.336 | 0.296 | 0.336 | 1 |
| nju_zhangsq_rmisc_opsd_household__h/short/cov | tuning | 0.336 | 0.296 | 0.336 | 1 |
| nju_zhangsq_rmisc_opsd_when2heat__h/short/target | tuning | 2.144 | 0.064 | 2.144 | 20 |
| nju_zhangsq_rmisc_opsd_when2heat__h/short/cov | tuning | 2.144 | 0.064 | 2.144 | 20 |
| nju_zhangsq_rmisc_opsd_when2heat__d/short/target | tuning | 2.466 | 0.053 | 2.466 | 6 |
| nju_zhangsq_rmisc_opsd_when2heat__d/short/cov | tuning | 2.466 | 0.053 | 2.466 | 6 |
| nju_zhangsq_rmisc_opsd_when2heat__wsun/short/target | tuning | 1.019 | 0.037 | 1.019 | 3 |
| nju_zhangsq_rmisc_opsd_when2heat__wsun/short/cov | tuning | 1.019 | 0.037 | 1.019 | 3 |
| nju_zhangsq_rmisc_pm25fivecities__h/short/target | tuning | 0.918 | 0.601 | 0.918 | 20 |
| nju_zhangsq_rmisc_pm25fivecities__h/short/cov | tuning | 0.918 | 0.601 | 0.918 | 20 |
| nju_zhangsq_rmisc_pm25fivecities__h/medium/target | tuning | 1.277 | 0.748 | 1.277 | 4 |
| nju_zhangsq_rmisc_pm25fivecities__h/medium/cov | tuning | 1.277 | 0.748 | 1.277 | 4 |
| nju_zhangsq_rmisc_pm25fivecities__h/long/target | tuning | 1.223 | 0.688 | 1.223 | 3 |
| nju_zhangsq_rmisc_pm25fivecities__h/long/cov | tuning | 1.223 | 0.688 | 1.223 | 3 |
| nju_zhangsq_rmisc_pm25fivecities__d/short/target | tuning | 1.703 | 0.636 | 1.703 | 3 |
| nju_zhangsq_rmisc_pm25fivecities__d/short/cov | tuning | 1.703 | 0.636 | 1.703 | 3 |
| nju_zhangsq_rmisc_pm25fivecities__wsun/short/target | tuning | 0.639 | 0.266 | 0.639 | 1 |
| nju_zhangsq_rmisc_pm25fivecities__wsun/short/cov | tuning | 0.639 | 0.266 | 0.639 | 1 |
| nju_zhangsq_rmisc_pump__min/short/target | tuning | 0.718 | 0.022 | 0.718 | 40 |
| nju_zhangsq_rmisc_pump__min/short/cov | tuning | 0.718 | 0.022 | 0.718 | 40 |
| nju_zhangsq_rmisc_pump__min/medium/target | tuning | 1.055 | 0.038 | 1.055 | 20 |
| nju_zhangsq_rmisc_pump__min/medium/cov | tuning | 1.055 | 0.038 | 1.055 | 20 |
| nju_zhangsq_rmisc_pump__min/long/target | tuning | 1.042 | 0.038 | 1.042 | 12 |
| nju_zhangsq_rmisc_pump__min/long/cov | tuning | 1.042 | 0.038 | 1.042 | 12 |
| nju_zhangsq_rmisc_pump__h/short/target | tuning | 0.761 | 0.025 | 0.761 | 2 |
| nju_zhangsq_rmisc_pump__h/short/cov | tuning | 0.761 | 0.025 | 0.761 | 2 |
| nju_zhangsq_rmisc_roomoccupancy__30s/short/target | tuning | 0.304 | 0.001 | 0.304 | 20 |
| nju_zhangsq_rmisc_roomoccupancy__30s/short/cov | tuning | 0.304 | 0.001 | 0.304 | 20 |
| nju_zhangsq_rmisc_roomoccupancy__30s/medium/target | tuning | 0.565 | 0.002 | 0.565 | 2 |
| nju_zhangsq_rmisc_roomoccupancy__30s/medium/cov | tuning | 0.565 | 0.002 | 0.565 | 2 |
| nju_zhangsq_rmisc_roomoccupancy__30s/long/target | tuning | 0.578 | 0.002 | 0.578 | 1 |
| nju_zhangsq_rmisc_roomoccupancy__30s/long/cov | tuning | 0.578 | 0.002 | 0.578 | 1 |
| nju_zhangsq_rmisc_smartmeteruk30m__30min/short/target | tuning | 1.055 | 0.566 | 1.055 | 19820 |
| nju_zhangsq_rmisc_smartmeteruk30m__30min/medium/target | tuning | 1.109 | 0.588 | 1.109 | 4950 |
| nju_zhangsq_rmisc_smartmeteruk30m__30min/long/target | tuning | 1.117 | 0.591 | 1.117 | 2970 |
| nju_zhangsq_rmisc_smartmeteruk30m__h/short/target | tuning | 1.098 | 0.516 | 1.098 | 19800 |
| nju_zhangsq_rmisc_smartmeteruk30m__h/medium/target | tuning | 1.192 | 0.565 | 1.192 | 1980 |
| nju_zhangsq_rmisc_smartmeteruk30m__h/long/target | tuning | 1.138 | 0.541 | 1.138 | 990 |
| nju_zhangsq_rmisc_smartmeteruk30m__d/short/target | tuning | 1.314 | 0.267 | 1.314 | 986 |
| nju_zhangsq_rmisc_smartmeteruk30m__wsun/short/target | tuning | 2.097 | 0.263 | 2.097 | 933 |
| nju_zhangsq_rmisc_sp500__d/short/target | tuning | 4.107 | 0.035 | 4.107 | 1981 |
| nju_zhangsq_rmisc_sp500__d/short/cov | tuning | 4.107 | 0.035 | 4.107 | 1981 |
| nju_zhangsq_rmisc_sp500__wsun/short/target | tuning | 2.423 | 0.048 | 2.423 | 978 |
| nju_zhangsq_rmisc_sp500__wsun/short/cov | tuning | 2.423 | 0.048 | 2.423 | 978 |
| nju_zhangsq_rmisc_stockmarketdata__d/short/target | tuning | 3.481 | 0.016 | 3.481 | 30 |
| nju_zhangsq_rmisc_stockmarketdata__d/short/cov | tuning | 3.481 | 0.016 | 3.481 | 30 |
| nju_zhangsq_rmisc_stockmarketdata__wsun/short/target | tuning | 2.489 | 0.023 | 2.489 | 15 |
| nju_zhangsq_rmisc_stockmarketdata__wsun/short/cov | tuning | 2.489 | 0.023 | 2.489 | 15 |
| nju_zhangsq_rmisc_tetuanpowerconsumption__10min/short/target | tuning | 0.989 | 0.130 | 0.989 | 20 |
| nju_zhangsq_rmisc_tetuanpowerconsumption__10min/short/cov | tuning | 0.989 | 0.130 | 0.989 | 20 |
| nju_zhangsq_rmisc_tetuanpowerconsumption__10min/medium/target | tuning | 1.497 | 0.187 | 1.497 | 10 |
| nju_zhangsq_rmisc_tetuanpowerconsumption__10min/medium/cov | tuning | 1.497 | 0.187 | 1.497 | 10 |
| nju_zhangsq_rmisc_tetuanpowerconsumption__10min/long/target | tuning | 1.691 | 0.213 | 1.691 | 6 |
| nju_zhangsq_rmisc_tetuanpowerconsumption__10min/long/cov | tuning | 1.691 | 0.213 | 1.691 | 6 |
| nju_zhangsq_rmisc_tetuanpowerconsumption__h/short/target | tuning | 1.511 | 0.188 | 1.511 | 17 |
| nju_zhangsq_rmisc_tetuanpowerconsumption__h/short/cov | tuning | 1.511 | 0.188 | 1.511 | 17 |
| nju_zhangsq_rmisc_tetuanpowerconsumption__h/medium/target | tuning | 2.091 | 0.255 | 2.091 | 1 |
| nju_zhangsq_rmisc_tetuanpowerconsumption__h/medium/cov | tuning | 2.091 | 0.255 | 2.091 | 1 |
| nju_zhangsq_rmisc_tetuanpowerconsumption__h/long/target | tuning | 1.581 | 0.197 | 1.581 | 1 |
| nju_zhangsq_rmisc_tetuanpowerconsumption__h/long/cov | tuning | 1.581 | 0.197 | 1.581 | 1 |
| nju_zhangsq_rmisc_tetuanpowerconsumption__d/short/target | tuning | 1.785 | 0.144 | 1.785 | 1 |
| nju_zhangsq_rmisc_tetuanpowerconsumption__d/short/cov | tuning | 1.785 | 0.144 | 1.785 | 1 |
| nju_zhangsq_rmisc_tushareetfdaily__d/short/target | tuning | 1.803 | 0.849 | 1.803 | 2966 |
| nju_zhangsq_rmisc_tushareetfdaily__wsun/short/target | tuning | 0.985 | 0.185 | 0.985 | 1847 |
| nju_zhangsq_rmisc_tushareindexdaily__d/short/target | tuning | 1.679 | 0.744 | 1.679 | 1801 |
| nju_zhangsq_rmisc_tushareindexdaily__wsun/short/target | tuning | 0.693 | 0.099 | 0.693 | 965 |
| nju_zhangsq_rmisc_tusharestockdaily__d/short/target | tuning | 2.152 | 0.833 | 2.152 | 4995 |
| nju_zhangsq_rmisc_tusharestockdaily__wsun/short/target | tuning | 0.487 | 0.073 | 0.487 | 2935 |
| nju_zhangsq_rmisc_tusharestockweekly__d/short/target | tuning | 0.857 | 1.687 | 0.857 | 7960 |
| nju_zhangsq_rmisc_tusharestockweekly__wsun/short/target | tuning | 0.022 | 0.001 | 0.022 | 3894 |
| nju_zhangsq_rmisc_wadi__5s/short/target | tuning | 0.904 | 0.017 | 0.904 | 40 |
| nju_zhangsq_rmisc_wadi__5s/short/cov | tuning | 0.904 | 0.017 | 0.904 | 40 |
| nju_zhangsq_rmisc_wadi__5s/medium/target | tuning | 1.012 | 0.015 | 1.012 | 10 |
| nju_zhangsq_rmisc_wadi__5s/medium/cov | tuning | 1.012 | 0.015 | 1.012 | 10 |
| nju_zhangsq_rmisc_wadi__5s/long/target | tuning | 0.895 | 0.014 | 0.895 | 6 |
| nju_zhangsq_rmisc_wadi__5s/long/cov | tuning | 0.895 | 0.014 | 0.895 | 6 |
| nju_zhangsq_rmisc_windelec__15min/short/target | tuning | 1.015 | 2.020 | 1.015 | 100 |
| nju_zhangsq_rmisc_windelec__15min/short/cov | tuning | 1.015 | 2.020 | 1.015 | 100 |
| nju_zhangsq_rmisc_windelec__15min/medium/target | tuning | 1.056 | 0.994 | 1.056 | 45 |
| nju_zhangsq_rmisc_windelec__15min/medium/cov | tuning | 1.056 | 0.994 | 1.056 | 45 |
| nju_zhangsq_rmisc_windelec__15min/long/target | tuning | 1.218 | 1.154 | 1.218 | 30 |
| nju_zhangsq_rmisc_windelec__15min/long/cov | tuning | 1.218 | 1.154 | 1.218 | 30 |
| nju_zhangsq_rmisc_windelec__h/short/target | tuning | 1.125 | 1.174 | 1.125 | 100 |
| nju_zhangsq_rmisc_windelec__h/short/cov | tuning | 1.125 | 1.174 | 1.125 | 100 |
| nju_zhangsq_rmisc_windelec__h/medium/target | tuning | 1.090 | 0.975 | 1.090 | 10 |
| nju_zhangsq_rmisc_windelec__h/medium/cov | tuning | 1.090 | 0.975 | 1.090 | 10 |
| nju_zhangsq_rmisc_windelec__h/long/target | tuning | 1.148 | 1.110 | 1.148 | 5 |
| nju_zhangsq_rmisc_windelec__h/long/cov | tuning | 1.148 | 1.110 | 1.148 | 5 |
| nju_zhangsq_rmisc_windelec__d/short/target | tuning | 1.272 | 0.885 | 1.272 | 5 |
| nju_zhangsq_rmisc_windelec__d/short/cov | tuning | 1.272 | 0.885 | 1.272 | 5 |
| nju_zhangsq_rmisc_windelec__wsun/short/target | tuning | 2.043 | 1.447 | 2.043 | 5 |
| nju_zhangsq_rmisc_windelec__wsun/short/cov | tuning | 2.043 | 1.447 | 2.043 | 5 |
| nju_zhangsq_rmisc_windpower4secs__4s/short/target | tuning | 1.211 | 0.751 | 1.211 | 20 |
| nju_zhangsq_rmisc_windpower4secs__4s/medium/target | tuning | 0.900 | 1.491 | 0.900 | 10 |
| nju_zhangsq_rmisc_windpower4secs__4s/long/target | tuning | 0.972 | 1.168 | 0.972 | 6 |
| nju_zhangsq_rmisc_xiamenairquality__h/short/target | tuning | 1.184 | 0.427 | 1.184 | 1918 |
| nju_zhangsq_rmisc_xiamenairquality__h/short/cov | tuning | 1.184 | 0.427 | 1.184 | 1918 |
| nju_zhangsq_rmisc_xiamenairquality__h/medium/target | tuning | 1.397 | 0.472 | 1.397 | 120 |
| nju_zhangsq_rmisc_xiamenairquality__h/medium/cov | tuning | 1.397 | 0.472 | 1.397 | 120 |
| nju_zhangsq_rmisc_xiamenairquality__h/long/target | tuning | 1.295 | 0.436 | 1.295 | 107 |
| nju_zhangsq_rmisc_xiamenairquality__h/long/cov | tuning | 1.295 | 0.436 | 1.295 | 107 |
| nju_zhangsq_rmisc_xiamenairquality__d/short/target | tuning | 1.808 | 0.349 | 1.808 | 150 |
| nju_zhangsq_rmisc_xiamenairquality__d/short/cov | tuning | 1.808 | 0.349 | 1.808 | 150 |
| numix_stations_officielles_es__10min/short/target | tuning | 1.012 | 0.264 | 1.012 | 8838 |
| numix_stations_officielles_es__10min/medium/target | tuning | 1.114 | 0.289 | 1.114 | 980 |
| numix_stations_officielles_es__10min/long/target | tuning | 1.191 | 0.312 | 1.191 | 976 |
| numix_stations_officielles_es__h/short/target | tuning | 1.080 | 0.236 | 1.080 | 987 |
| olaflaitinen_baku_metro_station_flow__d/short/target | tuning | 1.226 | 0.166 | 1.226 | 27 |
| olaflaitinen_baku_metro_station_flow__d/short/cov | tuning | 1.226 | 0.166 | 1.226 | 27 |
| orailix_ride_silver__d/short/target | tuning | 5.620 | 0.494 | 5.620 | 1000 |
| orailix_ride_silver__wsun/short/target | tuning | 2.112 | 0.294 | 2.112 | 793 |
| pahzuzu_pah_env_001_serc_soil_signals__d/short/target | tuning | 2.079 | 0.263 | 2.079 | 3 |
| pahzuzu_pah_env_001_serc_soil_signals__d/short/cov | tuning | 2.079 | 0.263 | 2.079 | 3 |
| pahzuzu_pah_env_001_serc_soil_signals__wsun/short/target | tuning | 1.841 | 0.168 | 1.841 | 22 |
| pahzuzu_pah_env_001_serc_soil_signals__wsun/short/cov | tuning | 1.841 | 0.168 | 1.841 | 22 |
| phanerozoic_qiskit_calibration_drift__d/short/target | tuning | 1.848 | 0.888 | 1.848 | 1 |
| powerzoojax_powerzoodataset_ausgrid_zone_substation__15min/short/target | tuning | 0.788 | 0.103 | 0.788 | 3480 |
| powerzoojax_powerzoodataset_ausgrid_zone_substation__15min/short/cov | tuning | 0.788 | 0.103 | 0.788 | 3480 |
| powerzoojax_powerzoodataset_ausgrid_zone_substation__h/medium/target | tuning | 0.926 | 0.122 | 0.926 | 174 |
| powerzoojax_powerzoodataset_ausgrid_zone_substation__h/medium/cov | tuning | 0.926 | 0.122 | 0.926 | 174 |
| powerzoojax_powerzoodataset_ausgrid_zone_substation__d/short/target | tuning | 1.762 | 0.155 | 1.762 | 174 |
| powerzoojax_powerzoodataset_ausgrid_zone_substation__d/short/cov | tuning | 1.762 | 0.155 | 1.762 | 174 |
| powerzoojax_powerzoodataset_gb_neso_demand_gb_gen_by_type_gb_market_mid_gb_forecast_actual_demand__30min/short/target | tuning | 1.241 | 0.404 | 1.241 | 200 |
| powerzoojax_powerzoodataset_gb_neso_demand_gb_gen_by_type_gb_market_mid_gb_forecast_actual_demand__30min/medium/target | tuning | 1.713 | 0.520 | 1.713 | 100 |
| powerzoojax_powerzoodataset_gb_neso_demand_gb_gen_by_type_gb_market_mid_gb_forecast_actual_demand__30min/long/target | tuning | 1.854 | 0.530 | 1.854 | 60 |
| powerzoojax_powerzoodataset_gb_neso_demand_gb_gen_by_type_gb_market_mid_gb_forecast_actual_demand__h/short/target | tuning | 1.385 | 0.459 | 1.385 | 200 |
| powerzoojax_powerzoodataset_gb_neso_demand_gb_gen_by_type_gb_market_mid_gb_forecast_actual_demand__h/medium/target | tuning | 1.607 | 0.467 | 1.607 | 50 |
| powerzoojax_powerzoodataset_gb_neso_demand_gb_gen_by_type_gb_market_mid_gb_forecast_actual_demand__h/long/target | tuning | 2.151 | 0.588 | 2.151 | 30 |
| powerzoojax_powerzoodataset_gb_neso_demand_gb_gen_by_type_gb_market_mid_gb_forecast_actual_demand__d/short/target | tuning | 2.550 | 0.545 | 2.550 | 30 |
| powerzoojax_powerzoodataset_gb_neso_demand_gb_gen_by_type_gb_market_mid_gb_forecast_actual_demand__wsun/short/target | tuning | 1.632 | 0.302 | 1.632 | 10 |
| pracadata_brazil_fuel_prices__d/short/target | tuning | 1.224 | 1.000 | 1.224 | 1987 |
| pracadata_brazil_fuel_prices__wsun/short/target | tuning | 1.791 | 0.123 | 1.791 | 928 |
| prasad_gade05_india_upi_ecosystem_2018_2025__qs/short/target | tuning | 5.866 | 0.499 | 5.866 | 36 |
| prasad_gade05_india_upi_ecosystem_2018_2025__qs/short/cov | tuning | 5.866 | 0.499 | 5.866 | 36 |
| previdengue_predict_climate_data__wsat/short/target | tuning | 9.719 | 0.156 | 9.719 | 8000 |
| previdengue_predict_climate_data__wsat/short/cov | tuning | 9.719 | 0.156 | 9.719 | 8000 |
| project_agml_usda_nass_crop_statistics__ys/short/target | tuning | 2.106 | 0.169 | 2.106 | 374 |
| project_agml_usda_nass_crop_statistics__ys/short/cov | tuning | 2.106 | 0.169 | 2.106 | 374 |
| prospex_ch_swiss_commercial_registry_monthly_panel__d/short/target | tuning | 0.508 | 1.000 | 0.508 | 9 |
| prospex_ch_swiss_commercial_registry_monthly_panel__wsun/short/target | tuning | 0.783 | 1.429 | 0.783 | 5 |
| rebase_energy_heftcom2024_default__wsun/short/target | tuning | 1.335 | 0.284 | 1.335 | 10 |
| rebase_energy_nve_windpower_data_nve_windpower_timeseries_csv__h/short/target | tuning | 1.355 | 0.828 | 1.355 | 1280 |
| rebase_energy_nve_windpower_data_nve_windpower_timeseries_csv__h/medium/target | tuning | 1.226 | 1.056 | 1.226 | 640 |
| rebase_energy_nve_windpower_data_nve_windpower_timeseries_csv__h/long/target | tuning | 1.266 | 1.086 | 1.266 | 384 |
| rebase_energy_nve_windpower_data_nve_windpower_timeseries_csv__d/short/target | tuning | 1.560 | 0.810 | 1.560 | 320 |
| rebase_energy_nve_windpower_data_nve_windpower_timeseries_csv__wsun/short/target | tuning | 1.105 | 0.400 | 1.105 | 128 |
| rif_sql_uk_electricity_generation_mix_2019_2026__30min/short/target | tuning | 1.048 | 0.214 | 1.048 | 5740 |
| rif_sql_uk_electricity_generation_mix_2019_2026__30min/medium/target | tuning | 1.551 | 0.315 | 1.551 | 861 |
| rif_sql_uk_electricity_generation_mix_2019_2026__30min/long/target | tuning | 1.559 | 0.303 | 1.559 | 574 |
| rif_sql_uk_electricity_generation_mix_2019_2026__h/short/target | tuning | 1.194 | 0.239 | 1.194 | 5166 |
| rif_sql_uk_electricity_generation_mix_2019_2026__h/medium/target | tuning | 1.862 | 0.376 | 1.862 | 287 |
| rif_sql_uk_electricity_generation_mix_2019_2026__h/long/target | tuning | 1.872 | 0.353 | 1.872 | 256 |
| rif_sql_uk_electricity_generation_mix_2019_2026__d/short/target | tuning | 2.195 | 0.324 | 2.195 | 256 |
| rif_sql_uk_electricity_generation_mix_2019_2026__wsun/short/target | tuning | 1.139 | 0.182 | 1.139 | 32 |
| roshangupta007_yieldformer_de_data__d/short/target | tuning | 1.592 | 0.137 | 1.592 | 7000 |
| roshangupta007_yieldformer_de_data__d/short/cov | tuning | 1.592 | 0.137 | 1.592 | 7000 |
| roshangupta007_yieldformer_de_data__wsun/short/target | tuning | 1.339 | 0.095 | 1.339 | 5418 |
| roshangupta007_yieldformer_de_data__wsun/short/cov | tuning | 1.339 | 0.095 | 1.339 | 5418 |
| rricc22_endomondo_hr_prediction_v2__6s/short/target | tuning | 5.651 | 0.027 | 5.651 | 50 |
| rricc22_endomondo_hr_prediction_v2__6s/short/cov | tuning | 5.651 | 0.027 | 5.651 | 50 |
| salesforce_lotsa_data_alibaba_cluster_trace_2018__5min/short/target | tuning | 1.200 | 0.054 | 1.200 | 5886 |
| salesforce_lotsa_data_alibaba_cluster_trace_2018__5min/short/cov | tuning | 1.200 | 0.054 | 1.200 | 5886 |
| salesforce_lotsa_data_alibaba_cluster_trace_2018__5min/short/joint | tuning | 1.200 | 0.054 | 1.200 | 5886 |
| salesforce_lotsa_data_alibaba_cluster_trace_2018__5min/medium/target | tuning | 1.149 | 0.048 | 1.149 | 1936 |
| salesforce_lotsa_data_alibaba_cluster_trace_2018__5min/medium/cov | tuning | 1.149 | 0.048 | 1.149 | 1936 |
| salesforce_lotsa_data_alibaba_cluster_trace_2018__5min/medium/joint | tuning | 1.149 | 0.048 | 1.149 | 1936 |
| salesforce_lotsa_data_bdg_2_bear__h/short/target | tuning | 0.998 | 0.076 | 0.998 | 1780 |
| salesforce_lotsa_data_bdg_2_bear__h/medium/target | tuning | 1.531 | 0.121 | 1.531 | 267 |
| salesforce_lotsa_data_bdg_2_bear__h/long/target | tuning | 1.526 | 0.116 | 1.526 | 178 |
| salesforce_lotsa_data_bdg_2_bear__d/short/target | tuning | 1.748 | 0.097 | 1.748 | 178 |
| salesforce_lotsa_data_bdg_2_bear__wsun/short/target | tuning | 1.739 | 0.065 | 1.739 | 76 |
| salesforce_lotsa_data_bdg_2_fox__h/short/target | tuning | 1.164 | 0.096 | 1.164 | 2600 |
| salesforce_lotsa_data_bdg_2_fox__h/medium/target | tuning | 1.486 | 0.116 | 1.486 | 390 |
| salesforce_lotsa_data_bdg_2_fox__h/long/target | tuning | 1.765 | 0.136 | 1.765 | 260 |
| salesforce_lotsa_data_bdg_2_fox__d/short/target | tuning | 2.196 | 0.117 | 2.196 | 260 |
| salesforce_lotsa_data_bdg_2_fox__wsun/short/target | tuning | 2.310 | 0.096 | 2.310 | 125 |
| salesforce_lotsa_data_bdg_2_panther__h/short/target | tuning | 1.124 | 0.104 | 1.124 | 1818 |
| salesforce_lotsa_data_bdg_2_panther__h/medium/target | tuning | 2.564 | 0.263 | 2.564 | 101 |
| salesforce_lotsa_data_bdg_2_panther__h/long/target | tuning | 2.357 | 0.203 | 2.357 | 101 |
| salesforce_lotsa_data_bdg_2_panther__d/short/target | tuning | 3.079 | 0.178 | 3.079 | 101 |
| salesforce_lotsa_data_bdg_2_rat__h/short/target | tuning | 0.930 | 0.136 | 0.930 | 5280 |
| salesforce_lotsa_data_bdg_2_rat__h/medium/target | tuning | 1.269 | 0.183 | 1.269 | 792 |
| salesforce_lotsa_data_bdg_2_rat__h/long/target | tuning | 1.452 | 0.202 | 1.452 | 528 |
| salesforce_lotsa_data_bdg_2_rat__d/short/target | tuning | 1.687 | 0.167 | 1.687 | 528 |
| salesforce_lotsa_data_bdg_2_rat__wsun/short/target | tuning | 1.669 | 0.153 | 1.669 | 245 |
| salesforce_lotsa_data_beijing_air_quality__wsun/short/target | tuning | 1.310 | 0.397 | 1.310 | 220 |
| salesforce_lotsa_data_beijing_air_quality__wsun/short/joint | tuning | 1.310 | 0.397 | 1.310 | 220 |
| salesforce_lotsa_data_borg_cluster_data_2011__5min/short/target | tuning | 1.287 | 0.377 | 1.287 | 3478 |
| salesforce_lotsa_data_borg_cluster_data_2011__5min/short/cov | tuning | 1.287 | 0.377 | 1.287 | 3478 |
| salesforce_lotsa_data_borg_cluster_data_2011__5min/short/joint | tuning | 1.287 | 0.377 | 1.287 | 3478 |
| salesforce_lotsa_data_borg_cluster_data_2011__h/short/target | tuning | 1.260 | 0.149 | 1.260 | 738 |
| salesforce_lotsa_data_borg_cluster_data_2011__h/short/cov | tuning | 1.260 | 0.149 | 1.260 | 738 |
| salesforce_lotsa_data_borg_cluster_data_2011__h/short/joint | tuning | 1.260 | 0.149 | 1.260 | 738 |
| salesforce_lotsa_data_cdc_fluview_ilinet__wsun/short/target | tuning | 4.222 | 0.059 | 4.222 | 760 |
| salesforce_lotsa_data_cdc_fluview_ilinet__wsun/short/joint | tuning | 4.222 | 0.059 | 4.222 | 760 |
| salesforce_lotsa_data_cdc_fluview_who_nrevss__wsun/short/target | tuning | 12.801 | 0.314 | 12.801 | 288 |
| salesforce_lotsa_data_cdc_fluview_who_nrevss__wsun/short/joint | tuning | 12.801 | 0.314 | 12.801 | 288 |
| salesforce_lotsa_data_covid_mobility__d/short/target | tuning | 1.416 | 0.381 | 1.416 | 337 |
| salesforce_lotsa_data_largest_2021__5min/short/target | tuning | 0.833 | 0.158 | 0.833 | 19940 |
| salesforce_lotsa_data_largest_2021__5min/medium/target | tuning | 1.135 | 0.193 | 1.135 | 9880 |
| salesforce_lotsa_data_largest_2021__5min/long/target | tuning | 1.332 | 0.227 | 1.332 | 5928 |
| salesforce_lotsa_data_largest_2021__h/short/target | tuning | 1.347 | 0.188 | 1.347 | 7976 |
| salesforce_lotsa_data_largest_2021__h/medium/target | tuning | 1.631 | 0.225 | 1.631 | 997 |
| salesforce_lotsa_data_largest_2021__h/long/target | tuning | 1.299 | 0.173 | 1.299 | 997 |
| salesforce_lotsa_data_largest_2021__d/short/target | tuning | 1.441 | 0.109 | 1.441 | 1000 |
| salesforce_lotsa_data_london_smart_meters_with_missing__30min/short/target | tuning | 1.045 | 0.565 | 1.045 | 19980 |
| salesforce_lotsa_data_london_smart_meters_with_missing__30min/medium/target | tuning | 1.116 | 0.587 | 1.116 | 4975 |
| salesforce_lotsa_data_london_smart_meters_with_missing__30min/long/target | tuning | 1.119 | 0.592 | 1.119 | 2985 |
| salesforce_lotsa_data_london_smart_meters_with_missing__h/short/target | tuning | 1.097 | 0.516 | 1.097 | 19960 |
| salesforce_lotsa_data_london_smart_meters_with_missing__h/medium/target | tuning | 1.166 | 0.547 | 1.166 | 1996 |
| salesforce_lotsa_data_london_smart_meters_with_missing__h/long/target | tuning | 1.150 | 0.552 | 1.150 | 998 |
| salesforce_lotsa_data_london_smart_meters_with_missing__d/short/target | tuning | 1.301 | 0.269 | 1.301 | 1000 |
| salesforce_lotsa_data_london_smart_meters_with_missing__wsun/short/target | tuning | 1.982 | 0.253 | 1.982 | 959 |
| salesforce_lotsa_data_m1_monthly__ms/short/target | tuning | 1.272 | 0.156 | 1.272 | 473 |
| salesforce_lotsa_data_nn5_daily_with_missing__d/short/target | tuning | 1.334 | 0.427 | 1.334 | 114 |
| salesforce_lotsa_data_nn5_daily_with_missing__wsun/short/target | tuning | 0.992 | 0.119 | 0.992 | 93 |
| salesforce_lotsa_data_oikolab_weather__h/short/target | tuning | 0.874 | 0.006 | 0.874 | 160 |
| salesforce_lotsa_data_oikolab_weather__h/medium/target | tuning | 1.309 | 0.008 | 1.309 | 80 |
| salesforce_lotsa_data_oikolab_weather__h/long/target | tuning | 1.264 | 0.007 | 1.264 | 48 |
| salesforce_lotsa_data_oikolab_weather__d/short/target | tuning | 1.430 | 0.007 | 1.430 | 48 |
| salesforce_lotsa_data_oikolab_weather__wsun/short/target | tuning | 1.573 | 0.007 | 1.573 | 24 |
| salesforce_lotsa_data_pedestrian_counts__h/short/target | tuning | 0.430 | 0.346 | 0.430 | 1298 |
| salesforce_lotsa_data_pedestrian_counts__h/medium/target | tuning | 1.320 | 0.588 | 1.320 | 248 |
| salesforce_lotsa_data_pedestrian_counts__h/long/target | tuning | 1.227 | 0.497 | 1.227 | 184 |
| salesforce_lotsa_data_pedestrian_counts__d/short/target | tuning | 1.417 | 0.407 | 1.417 | 182 |
| salesforce_lotsa_data_pedestrian_counts__wsun/short/target | tuning | 6.944 | 1.016 | 6.944 | 58 |
| salesforce_lotsa_data_pems04__5min/short/target | tuning | 1.060 | 0.164 | 1.060 | 17340 |
| salesforce_lotsa_data_pems04__5min/short/joint | tuning | 1.060 | 0.164 | 1.060 | 17340 |
| salesforce_lotsa_data_pems04__5min/medium/target | tuning | 1.276 | 0.225 | 1.276 | 2601 |
| salesforce_lotsa_data_pems04__5min/medium/joint | tuning | 1.276 | 0.225 | 1.276 | 2601 |
| salesforce_lotsa_data_pems04__5min/long/target | tuning | 1.250 | 0.200 | 1.250 | 1734 |
| salesforce_lotsa_data_pems04__5min/long/joint | tuning | 1.250 | 0.200 | 1.250 | 1734 |
| salesforce_lotsa_data_pems04__h/short/target | tuning | 1.066 | 0.138 | 1.066 | 1734 |
| salesforce_lotsa_data_pems04__h/short/joint | tuning | 1.066 | 0.138 | 1.066 | 1734 |
| salesforce_lotsa_data_pems08__5min/short/target | tuning | 1.015 | 0.140 | 1.015 | 8940 |
| salesforce_lotsa_data_pems08__5min/short/joint | tuning | 1.015 | 0.140 | 1.015 | 8940 |
| salesforce_lotsa_data_pems08__5min/medium/target | tuning | 1.285 | 0.187 | 1.285 | 1341 |
| salesforce_lotsa_data_pems08__5min/medium/joint | tuning | 1.285 | 0.187 | 1.285 | 1341 |
| salesforce_lotsa_data_pems08__5min/long/target | tuning | 1.287 | 0.168 | 1.287 | 894 |
| salesforce_lotsa_data_pems08__5min/long/joint | tuning | 1.287 | 0.168 | 1.287 | 894 |
| salesforce_lotsa_data_pems08__h/short/target | tuning | 1.053 | 0.117 | 1.053 | 1341 |
| salesforce_lotsa_data_pems08__h/short/joint | tuning | 1.053 | 0.117 | 1.053 | 1341 |
| salesforce_lotsa_data_pems08__h/medium/target | tuning | 1.217 | 0.121 | 1.217 | 447 |
| salesforce_lotsa_data_pems08__h/medium/joint | tuning | 1.217 | 0.121 | 1.217 | 447 |
| salesforce_lotsa_data_project_tycho__wsun/short/target | tuning | 0.922 | 0.549 | 0.922 | 1540 |
| salesforce_lotsa_data_q_traffic__15min/short/target | tuning | 1.040 | 0.108 | 1.040 | 12000 |
| salesforce_lotsa_data_q_traffic__15min/medium/target | tuning | 1.147 | 0.120 | 1.147 | 1000 |
| salesforce_lotsa_data_q_traffic__15min/long/target | tuning | 1.126 | 0.119 | 1.126 | 1000 |
| salesforce_lotsa_data_q_traffic__h/short/target | tuning | 1.143 | 0.096 | 1.143 | 3000 |
| salesforce_lotsa_data_q_traffic__h/medium/target | tuning | 1.085 | 0.091 | 1.085 | 1000 |
| salesforce_lotsa_data_residential_load_power__min/short/target | tuning | 0.813 | 0.623 | 0.813 | 14100 |
| salesforce_lotsa_data_residential_load_power__min/short/joint | tuning | 0.813 | 0.623 | 0.813 | 14100 |
| salesforce_lotsa_data_residential_load_power__min/medium/target | tuning | 0.908 | 0.722 | 0.908 | 6000 |
| salesforce_lotsa_data_residential_load_power__min/medium/joint | tuning | 0.908 | 0.722 | 0.908 | 6000 |
| salesforce_lotsa_data_residential_load_power__min/long/target | tuning | 0.942 | 0.707 | 0.942 | 3636 |
| salesforce_lotsa_data_residential_load_power__min/long/joint | tuning | 0.942 | 0.707 | 0.942 | 3636 |
| salesforce_lotsa_data_residential_load_power__h/short/target | tuning | 0.937 | 0.529 | 0.937 | 747 |
| salesforce_lotsa_data_residential_load_power__h/short/joint | tuning | 0.937 | 0.529 | 0.937 | 747 |
| salesforce_lotsa_data_smart__h/short/target | tuning | 1.137 | 0.471 | 1.137 | 90 |
| salesforce_lotsa_data_smart__h/medium/target | tuning | 1.151 | 0.430 | 1.151 | 5 |
| salesforce_lotsa_data_smart__h/long/target | tuning | 1.240 | 0.507 | 1.240 | 5 |
| salesforce_lotsa_data_smart__d/short/target | tuning | 1.587 | 0.306 | 1.587 | 5 |
| salesforce_lotsa_data_smart__wsun/short/target | tuning | 0.588 | 0.112 | 0.588 | 3 |
| salesforce_lotsa_data_subseasonal__d/short/target | tuning | 7.766 | 0.448 | 7.766 | 68960 |
| salesforce_lotsa_data_subseasonal__d/short/joint | tuning | 7.766 | 0.448 | 7.766 | 68960 |
| salesforce_lotsa_data_subseasonal__wsun/short/target | tuning | 2.432 | 0.537 | 2.432 | 68960 |
| salesforce_lotsa_data_subseasonal__wsun/short/joint | tuning | 2.432 | 0.537 | 2.432 | 68960 |
| salesforce_lotsa_data_subseasonal_precip__d/short/target | tuning | 7.247 | 0.713 | 7.247 | 17240 |
| salesforce_lotsa_data_subseasonal_precip__wsun/short/target | tuning | 1.753 | 0.801 | 1.753 | 17240 |
| salesforce_lotsa_data_sunspot_with_missing__d/short/target | tuning | 0.431 | inf | 0.431 | 20 |
| salesforce_lotsa_data_sunspot_with_missing__wsun/short/target | tuning | 0.349 | 1.531 | 0.349 | 20 |
| salesforce_lotsa_data_taxi_30min__30min/short/target | tuning | 1.042 | 0.504 | 1.042 | 1000 |
| salesforce_lotsa_data_taxi_30min__h/short/target | tuning | 1.142 | 0.473 | 1.142 | 737 |
| salesforce_lotsa_data_traffic_hourly__h/short/target | tuning | 1.339 | 0.373 | 1.339 | 17240 |
| salesforce_lotsa_data_traffic_hourly__h/medium/target | tuning | 1.530 | 0.393 | 1.530 | 2586 |
| salesforce_lotsa_data_traffic_hourly__h/long/target | tuning | 1.715 | 0.440 | 1.715 | 1724 |
| salesforce_lotsa_data_traffic_hourly__d/short/target | tuning | 1.672 | 0.305 | 1.672 | 1724 |
| salesforce_lotsa_data_traffic_hourly__wsun/short/target | tuning | 1.594 | 0.103 | 1.594 | 862 |
| salesforce_lotsa_data_uber_tlc_daily__d/short/target | tuning | 1.788 | 0.276 | 1.788 | 257 |
| salesforce_lotsa_data_vehicle_trips_with_missing__d/short/target | tuning | 1.018 | 0.305 | 1.018 | 31 |
| salesforce_lotsa_data_weather__d/short/target | tuning | 1.529 | 0.262 | 1.529 | 19980 |
| salesforce_lotsa_data_weather__wsun/short/target | tuning | 1.424 | 0.204 | 1.424 | 12987 |
| salesforce_lotsa_data_wiki_rolling_nips__d/short/target | tuning | 1.522 | 0.358 | 1.522 | 2000 |
| salesforce_lotsa_data_wiki_rolling_nips__wsun/short/target | tuning | 1.524 | 0.318 | 1.524 | 1000 |
| salesforce_lotsa_data_wind_farms_with_missing__min/short/target | tuning | 18.054 | 0.207 | 18.054 | 5400 |
| salesforce_lotsa_data_wind_farms_with_missing__min/medium/target | tuning | 6.499 | 0.214 | 6.499 | 2670 |
| salesforce_lotsa_data_wind_farms_with_missing__min/long/target | tuning | 7.058 | 0.208 | 7.058 | 1602 |
| salesforce_lotsa_data_wind_farms_with_missing__h/short/target | tuning | 10.566 | 0.178 | 10.566 | 270 |
| sayanroy058_weather_time_series_forecasting__h/short/target | tuning | 1.046 | 0.047 | 1.046 | 280 |
| sayanroy058_weather_time_series_forecasting__h/medium/target | tuning | 2.316 | 0.071 | 2.316 | 140 |
| sayanroy058_weather_time_series_forecasting__h/long/target | tuning | 2.799 | 0.070 | 2.799 | 84 |
| sayanroy058_weather_time_series_forecasting__d/short/target | tuning | 3.093 | 0.048 | 3.093 | 84 |
| sayanroy058_weather_time_series_forecasting__wsun/short/target | tuning | 1.482 | 0.043 | 1.482 | 42 |
| shashkovich_telecommunication_sms_time_series__d/short/target | tuning | 1.356 | 0.006 | 1.356 | 2 |
| showlcw_china_cpi_ppi_pmi__ms/short/target | tuning | 0.421 | 0.014 | 0.421 | 1 |
| skforecast_ashrae_daily__d/short/target | tuning | 2.549 | 0.195 | 2.549 | 998 |
| skforecast_ashrae_daily__d/short/cov | tuning | 2.549 | 0.195 | 2.549 | 998 |
| skywalker1910_skynet_aqi_panel__h/short/target | tuning | 1.063 | 0.580 | 1.063 | 660 |
| skywalker1910_skynet_aqi_panel__h/short/cov | tuning | 1.063 | 0.580 | 1.063 | 660 |
| skywalker1910_skynet_aqi_panel__h/medium/target | tuning | 1.113 | 0.689 | 1.113 | 165 |
| skywalker1910_skynet_aqi_panel__h/medium/cov | tuning | 1.113 | 0.689 | 1.113 | 165 |
| skywalker1910_skynet_aqi_panel__h/long/target | tuning | 1.062 | 0.651 | 1.062 | 99 |
| skywalker1910_skynet_aqi_panel__h/long/cov | tuning | 1.062 | 0.651 | 1.062 | 99 |
| skywalker1910_skynet_aqi_panel__d/short/target | tuning | 1.155 | 0.437 | 1.155 | 99 |
| skywalker1910_skynet_aqi_panel__d/short/cov | tuning | 1.155 | 0.437 | 1.155 | 99 |
| skywalker1910_skynet_aqi_panel__wsun/short/target | tuning | 1.244 | 0.313 | 1.244 | 66 |
| skywalker1910_skynet_aqi_panel__wsun/short/cov | tuning | 1.244 | 0.313 | 1.244 | 66 |
| somemone_ac_transit_apc__d/short/target | tuning | 8.752 | 0.236 | 8.752 | 1000 |
| somemone_ac_transit_apc__wsun/short/target | tuning | 1.979 | 0.097 | 1.979 | 749 |
| sovai_wikipedia_views_default__d/short/target | tuning | 2.031 | 0.301 | 2.031 | 9940 |
| sovai_wikipedia_views_default__d/short/cov | tuning | 2.031 | 0.301 | 2.031 | 9940 |
| sovai_wikipedia_views_default__wsun/short/target | tuning | 2.249 | 0.269 | 2.249 | 4975 |
| sovai_wikipedia_views_default__wsun/short/cov | tuning | 2.249 | 0.269 | 2.249 | 4975 |
| steven0226_air_quality__d/short/target | tuning | 1.222 | 0.434 | 1.222 | 5900 |
| steven0226_air_quality__d/short/cov | tuning | 1.222 | 0.434 | 1.222 | 5900 |
| steven0226_air_quality__wsun/short/target | tuning | 0.918 | 0.261 | 0.918 | 6851 |
| steven0226_air_quality__wsun/short/cov | tuning | 0.918 | 0.261 | 0.918 | 6851 |
| surfe_diem_wave_archive_usa_southwest__h/short/target | tuning | 1.331 | 0.167 | 1.331 | 944 |
| surfe_diem_wave_archive_usa_southwest__h/medium/target | tuning | 1.837 | 0.248 | 1.837 | 159 |
| surfe_diem_wave_archive_usa_southwest__h/long/target | tuning | 1.601 | 0.220 | 1.601 | 68 |
| surfe_diem_wave_archive_usa_southwest__d/short/target | tuning | 1.945 | 0.149 | 1.945 | 599 |
| thanna94_us_building_permits__ys/short/target | tuning | 5.134 | 0.898 | 5.134 | 630 |
| thanna94_us_building_permits__ys/short/cov | tuning | 5.134 | 0.898 | 5.134 | 630 |
| tharinduperera_weather_clustering_data__d/short/target | tuning | 2.214 | 0.112 | 2.214 | 1188 |
| tharinduperera_weather_clustering_data__d/short/cov | tuning | 2.214 | 0.112 | 2.214 | 1188 |
| tharinduperera_weather_clustering_data__wsun/short/target | tuning | 2.028 | 0.139 | 2.028 | 594 |
| tharinduperera_weather_clustering_data__wsun/short/cov | tuning | 2.028 | 0.139 | 2.028 | 594 |
| thermobench_thermobench_pure__5min/short/target | tuning | 0.889 | 0.259 | 0.889 | 400 |
| thermobench_thermobench_pure__5min/medium/target | tuning | 1.074 | 0.329 | 1.074 | 200 |
| thermobench_thermobench_pure__5min/long/target | tuning | 1.140 | 0.326 | 1.140 | 120 |
| thermobench_thermobench_pure__h/short/target | tuning | 1.058 | 0.167 | 1.058 | 160 |
| thermobench_thermobench_pure__h/medium/target | tuning | 1.848 | 0.277 | 1.848 | 10 |
| thermobench_thermobench_pure__h/long/target | tuning | 1.849 | 0.280 | 1.849 | 10 |
| thermobench_thermobench_pure__d/short/target | tuning | 2.485 | 0.211 | 2.485 | 10 |
| todopueblos_poblacion_municipios_espana__ys/short/target | tuning | 7.539 | 0.161 | 7.539 | 1000 |
| valdovaldo1500_crime_brasil_rs_2022_2026__ms/short/target | tuning | 0.861 | 0.036 | 0.861 | 1 |
| wikimedia_community_english_wikipedia_pageviews_by_second__s/short/target | tuning | 0.955 | 0.051 | 0.955 | 40 |
| wikimedia_community_english_wikipedia_pageviews_by_second__s/medium/target | tuning | 1.156 | 0.061 | 1.156 | 20 |
| wikimedia_community_english_wikipedia_pageviews_by_second__s/long/target | tuning | 1.165 | 0.062 | 1.165 | 12 |
| witgaw_pems_bay__5min/short/target | tuning | 0.862 | 0.051 | 0.862 | 6500 |
| witgaw_pems_bay__5min/short/cov | tuning | 0.862 | 0.051 | 0.862 | 6500 |
| witgaw_pems_bay__5min/medium/target | tuning | 1.165 | 0.066 | 1.165 | 3250 |
| witgaw_pems_bay__5min/medium/cov | tuning | 1.165 | 0.066 | 1.165 | 3250 |
| witgaw_pems_bay__5min/long/target | tuning | 1.266 | 0.071 | 1.266 | 1950 |
| witgaw_pems_bay__5min/long/cov | tuning | 1.266 | 0.071 | 1.266 | 1950 |
| witgaw_pems_bay__h/short/target | tuning | 1.091 | 0.054 | 1.091 | 2600 |
| witgaw_pems_bay__h/short/cov | tuning | 1.091 | 0.054 | 1.091 | 2600 |
| witgaw_pems_bay__h/medium/target | tuning | 1.381 | 0.069 | 1.381 | 325 |
| witgaw_pems_bay__h/medium/cov | tuning | 1.381 | 0.069 | 1.381 | 325 |
| witgaw_pems_bay__h/long/target | tuning | 1.190 | 0.058 | 1.190 | 325 |
| witgaw_pems_bay__h/long/cov | tuning | 1.190 | 0.058 | 1.190 | 325 |
| witgaw_pems_bay__d/short/target | tuning | 1.151 | 0.038 | 1.151 | 325 |
| witgaw_pems_bay__d/short/cov | tuning | 1.151 | 0.038 | 1.151 | 325 |
| xairon_piezo_embedding_benchmark__d/short/target | tuning | 5.307 | 0.005 | 5.307 | 1080 |
| xairon_piezo_embedding_benchmark__d/short/cov | tuning | 5.307 | 0.005 | 5.307 | 1080 |
| xairon_piezo_embedding_benchmark__wsun/short/target | tuning | 2.900 | 0.005 | 2.900 | 2110 |
| xairon_piezo_embedding_benchmark__wsun/short/cov | tuning | 2.900 | 0.005 | 2.900 | 2110 |
| xus_spain_fuel_prices__d/short/target | tuning | 4.892 | 0.011 | 4.892 | 918 |
| yahya2491_3w_pipeline__s/short/target | tuning | 1.195 | 0.009 | 1.195 | 5700 |
| yahya2491_3w_pipeline__s/medium/target | tuning | 1.268 | 0.009 | 1.268 | 570 |
| yahya2491_3w_pipeline__s/long/target | tuning | 1.425 | 0.009 | 1.425 | 290 |
| youngbongbong_30_od_subway_default__d/short/target | tuning | 0.901 | 0.159 | 0.901 | 261 |
| youngbongbong_30_od_subway_default__d/short/cov | tuning | 0.901 | 0.159 | 0.901 | 261 |
| youngbongbong_30_od_subway_default__wsun/short/target | tuning | 0.823 | 0.047 | 0.823 | 285 |
| youngbongbong_30_od_subway_default__wsun/short/cov | tuning | 0.823 | 0.047 | 0.823 | 285 |
| yuiseki_ilostat__qs/short/target | tuning | 3.131 | 0.339 | 3.131 | 197 |
| yuiseki_jma_rain_kanto__h/short/target | tuning | 3.122 | 3.735 | 3.122 | 1940 |
| yuiseki_jma_rain_kanto__h/short/cov | tuning | 3.122 | 3.735 | 3.122 | 1940 |
| yuiseki_jma_rain_kanto__h/medium/target | tuning | 1.477 | 1.632 | 1.477 | 830 |
| yuiseki_jma_rain_kanto__h/medium/cov | tuning | 1.477 | 1.632 | 1.477 | 830 |
| yuiseki_jma_rain_kanto__h/long/target | tuning | 1.792 | 2.648 | 1.792 | 498 |
| yuiseki_jma_rain_kanto__h/long/cov | tuning | 1.792 | 2.648 | 1.792 | 498 |
| yuiseki_jma_rain_kanto__d/short/target | tuning | 1.950 | 2.264 | 1.950 | 642 |
| yuiseki_jma_rain_kanto__d/short/cov | tuning | 1.950 | 2.264 | 1.950 | 642 |
| yuiseki_jma_rain_kanto__wsun/short/target | tuning | 1.925 | 1.175 | 1.925 | 342 |
| yuiseki_jma_rain_kanto__wsun/short/cov | tuning | 1.925 | 1.175 | 1.925 | 342 |
| yuiseki_meta_range_maps_2022_05__d/short/target | tuning | 1.312 | 0.687 | 1.312 | 827 |
| yuiseki_meta_range_maps_2022_05__d/short/cov | tuning | 1.312 | 0.687 | 1.312 | 827 |
| yuiseki_meta_range_maps_2022_05__wsun/short/target | tuning | 1.134 | 0.333 | 1.134 | 390 |
| yuiseki_meta_range_maps_2022_05__wsun/short/cov | tuning | 1.134 | 0.333 | 1.134 | 390 |
| yuiseki_pwt__ys/short/target | tuning | 2.608 | 0.094 | 2.608 | 168 |
| yuiseki_undata_sdmx__qs/short/target | tuning | 47.220 | 0.435 | 47.220 | 9 |
| yuiseki_undata_sdmx__qs/short/cov | tuning | 47.220 | 0.435 | 47.220 | 9 |
| yuiseki_undp_hdr__ys/short/target | tuning | 2.326 | 0.073 | 2.326 | 206 |
| yzzpz_charged_data_10_csv_per_city__h/short/target | tuning | 1.159 | 0.237 | 1.159 | 4554 |
| yzzpz_charged_data_10_csv_per_city__h/medium/target | tuning | 1.397 | 0.285 | 1.397 | 506 |
| yzzpz_charged_data_10_csv_per_city__h/long/target | tuning | 1.809 | 0.303 | 1.809 | 506 |
| yzzpz_charged_data_10_csv_per_city__d/short/target | tuning | 2.035 | 0.185 | 2.035 | 506 |
| zait_ai_rusweather_434__d/short/target | tuning | 2.726 | 0.601 | 2.726 | 3258 |
| zait_ai_rusweather_434__d/short/cov | tuning | 2.726 | 0.601 | 2.726 | 3258 |
| zait_ai_rusweather_434__wsun/short/target | tuning | 2.100 | 0.610 | 2.100 | 1448 |
| zait_ai_rusweather_434__wsun/short/cov | tuning | 2.100 | 0.610 | 2.100 | 1448 |
| zalizedata_us_grid_hourly_power_plants_dataset__h/short/target | tuning | 1.252 | 0.059 | 1.252 | 580 |
| zalizedata_us_grid_hourly_power_plants_dataset__h/medium/target | tuning | 1.879 | 0.102 | 1.879 | 290 |
| zalizedata_us_grid_hourly_power_plants_dataset__h/long/target | tuning | 1.930 | 0.102 | 1.930 | 174 |
| zalizedata_us_grid_hourly_power_plants_dataset__d/short/target | tuning | 2.315 | 0.097 | 2.315 | 174 |
| zalizedata_us_grid_hourly_power_plants_dataset__wsun/short/target | tuning | 1.902 | 0.099 | 1.902 | 81 |
| zer0contextlost_waterwatch_runoff__ms/short/target | tuning | 0.969 | 0.636 | 0.969 | 10791 |
| zer0contextlost_waterwatch_runoff__ms/short/cov | tuning | 0.969 | 0.636 | 0.969 | 10791 |
| aduitya9_aduitya_dextra_aqi_india__d/short/target | locked | 6.192 | 0.343 | 6.192 | 30 |
| aduitya9_aduitya_dextra_aqi_india__d/short/cov | locked | 6.192 | 0.343 | 6.192 | 30 |
| aduitya9_aduitya_dextra_aqi_india__wsun/short/target | locked | 3.791 | 0.414 | 3.791 | 15 |
| aduitya9_aduitya_dextra_aqi_india__wsun/short/cov | locked | 3.791 | 0.414 | 3.791 | 15 |
| akshaykarthick_india_weather_air_quality__d/short/target | locked | 2.008 | 0.528 | 2.008 | 250 |
| akshaykarthick_india_weather_air_quality__d/short/cov | locked | 2.008 | 0.528 | 2.008 | 250 |
| akshaykarthick_india_weather_air_quality__wsun/short/target | locked | 1.439 | 0.534 | 1.439 | 100 |
| akshaykarthick_india_weather_air_quality__wsun/short/cov | locked | 1.439 | 0.534 | 1.439 | 100 |
| ashwingopalsamy_central_bank_policy_rates__d/short/target | locked | 10.742 | 0.006 | 10.742 | 140 |
| ashwingopalsamy_central_bank_policy_rates__d/short/cov | locked | 10.742 | 0.006 | 10.742 | 140 |
| ashwingopalsamy_central_bank_policy_rates__wsun/short/target | locked | 2.980 | 0.012 | 2.980 | 70 |
| ashwingopalsamy_central_bank_policy_rates__wsun/short/cov | locked | 2.980 | 0.012 | 2.980 | 70 |
| autogluon_chronos_datasets_monash_london_smart_meters__30min/short/target | locked | 1.073 | 0.582 | 1.073 | 20000 |
| autogluon_chronos_datasets_monash_london_smart_meters__30min/medium/target | locked | 1.116 | 0.595 | 1.116 | 4985 |
| autogluon_chronos_datasets_monash_london_smart_meters__30min/long/target | locked | 1.132 | 0.607 | 1.132 | 2991 |
| autogluon_chronos_datasets_monash_london_smart_meters__h/short/target | locked | 1.110 | 0.530 | 1.110 | 19940 |
| autogluon_chronos_datasets_monash_london_smart_meters__h/medium/target | locked | 1.179 | 0.558 | 1.179 | 1994 |
| autogluon_chronos_datasets_monash_london_smart_meters__h/long/target | locked | 1.165 | 0.553 | 1.165 | 997 |
| autogluon_chronos_datasets_monash_london_smart_meters__d/short/target | locked | 1.298 | 0.260 | 1.298 | 999 |
| autogluon_chronos_datasets_monash_london_smart_meters__wsun/short/target | locked | 2.041 | 0.240 | 2.041 | 960 |
| autogluon_chronos_datasets_ushcn_daily__d/short/target | locked | 1.258 | 1.889 | 1.258 | 6100 |
| autogluon_chronos_datasets_ushcn_daily__d/short/cov | locked | 1.258 | 1.889 | 1.258 | 6100 |
| autogluon_chronos_datasets_ushcn_daily__wsun/short/target | locked | 1.043 | 1.188 | 1.043 | 6760 |
| autogluon_chronos_datasets_ushcn_daily__wsun/short/cov | locked | 1.043 | 1.188 | 1.043 | 6760 |
| autogluon_chronos_datasets_weatherbench_weekly__wsun/short/target | locked | 1.328 | 0.059 | 1.328 | 20000 |
| autogluon_chronos_datasets_wind_farms_hourly_wind_farms_daily__h/short/target | locked | 1.329 | 0.241 | 1.329 | 5868 |
| autogluon_chronos_datasets_wind_farms_hourly_wind_farms_daily__h/medium/target | locked | 1.838 | 0.361 | 1.838 | 326 |
| autogluon_chronos_datasets_wind_farms_hourly_wind_farms_daily__h/long/target | locked | 3.538 | 0.435 | 3.538 | 324 |
| autogluon_chronos_datasets_wind_farms_hourly_wind_farms_daily__d/short/target | locked | 3.896 | 0.387 | 3.896 | 317 |
| cashlessconsumer_chennai_rain_gauges__d/short/target | locked | 4.626 | 1.845 | 4.626 | 90 |
| cashlessconsumer_chennai_rain_gauges__d/short/cov | locked | 4.626 | 1.845 | 4.626 | 90 |
| cashlessconsumer_chennai_rain_gauges__wsun/short/target | locked | 2.733 | 1.358 | 2.733 | 60 |
| cashlessconsumer_chennai_rain_gauges__wsun/short/cov | locked | 2.733 | 1.358 | 2.733 | 60 |
| cm3r_meteo_france_climatologie_quotidienne__d/short/target | locked | 1.724 | 0.290 | 1.724 | 1774 |
| cm3r_meteo_france_climatologie_quotidienne__wsun/short/target | locked | 1.469 | 0.152 | 1.469 | 955 |
| digitalhen_us_airport_wait_times__min/short/target | locked | 0.766 | 0.294 | 0.766 | 760 |
| digitalhen_us_airport_wait_times__min/medium/target | locked | 0.673 | 0.196 | 0.673 | 280 |
| digitalhen_us_airport_wait_times__min/long/target | locked | 0.710 | 0.202 | 0.710 | 168 |
| digitalhen_us_airport_wait_times__h/short/target | locked | 0.824 | 0.234 | 0.824 | 48 |
| dmariaa70_metraq_air_quality__h/short/target | locked | nan | 0.218 | nan | 12140 |
| dmariaa70_metraq_air_quality__h/short/cov | locked | nan | 0.218 | nan | 12140 |
| dmariaa70_metraq_air_quality__h/medium/target | locked | nan | 0.228 | nan | 6070 |
| dmariaa70_metraq_air_quality__h/medium/cov | locked | nan | 0.228 | nan | 6070 |
| dmariaa70_metraq_air_quality__h/long/target | locked | nan | 0.225 | nan | 3642 |
| dmariaa70_metraq_air_quality__h/long/cov | locked | nan | 0.225 | nan | 3642 |
| dmariaa70_metraq_air_quality__d/short/target | locked | nan | 0.175 | nan | 3642 |
| dmariaa70_metraq_air_quality__d/short/cov | locked | nan | 0.175 | nan | 3642 |
| dmariaa70_metraq_air_quality__wsun/short/target | locked | nan | 0.134 | nan | 1821 |
| dmariaa70_metraq_air_quality__wsun/short/cov | locked | nan | 0.134 | nan | 1821 |
| earino_austin_911_response__d/short/target | locked | 1.152 | 0.069 | 1.152 | 3 |
| earino_austin_911_response__wsun/short/target | locked | 4.758 | 0.112 | 4.758 | 1 |
| electricsheepafrica_wfp_nigeria_food_prices__ms/short/target | locked | 1.611 | 0.761 | 1.611 | 39 |
| electricsheepafrica_wfp_nigeria_food_prices__ms/short/cov | locked | 1.611 | 0.761 | 1.611 | 39 |
| empulse_ai_empulse_street_fx_index__d/short/target | locked | 0.426 | 0.024 | 0.426 | 40 |
| empulse_ai_empulse_street_fx_index__wsun/short/target | locked | 0.426 | 0.017 | 0.426 | 24 |
| emulated_inc_sci_airport_low_ceiling_visibility__wsun/short/target | locked | 0.000 | 0.000 | 0.000 | 3 |
| emulated_inc_sci_airport_low_ceiling_visibility__wsun/short/cov | locked | 0.000 | 0.000 | 0.000 | 3 |
| epi_eval_global_mobility__d/short/target | locked | 1.189 | 0.481 | 1.189 | 2736 |
| epi_eval_global_mobility__d/short/cov | locked | 1.189 | 0.481 | 1.189 | 2736 |
| epi_eval_global_mobility__wsun/short/target | locked | 1.444 | 0.479 | 1.444 | 957 |
| epi_eval_global_mobility__wsun/short/cov | locked | 1.444 | 0.479 | 1.444 | 957 |
| epi_eval_nhsn_hrd__wsat/short/target | locked | 9.424 | 1.138 | 9.424 | 192 |
| epi_eval_nhsn_hrd__wsat/short/cov | locked | 9.424 | 1.138 | 9.424 | 192 |
| fidel_ts_canada_photovoltaics_plants__100s/short/target | locked | 4.652 | nan | 4.652 | 220 |
| fidel_ts_canada_photovoltaics_plants__100s/medium/target | locked | 1.143 | nan | 1.143 | 110 |
| fidel_ts_canada_photovoltaics_plants__100s/long/target | locked | 1.262 | nan | 1.262 | 66 |
| fidel_ts_canada_photovoltaics_plants__h/short/target | locked | 2.148 | 0.699 | 2.148 | 22 |
| fidel_ts_nyc_traffic_speed__5min/short/target | locked | 0.859 | 0.231 | 0.859 | 1780 |
| fidel_ts_nyc_traffic_speed__5min/medium/target | locked | 0.964 | 0.239 | 0.964 | 890 |
| fidel_ts_nyc_traffic_speed__5min/long/target | locked | 1.033 | 0.255 | 1.033 | 534 |
| fidel_ts_nyc_traffic_speed__h/short/target | locked | 0.928 | 0.184 | 0.928 | 712 |
| fidel_ts_nyc_traffic_speed__h/medium/target | locked | 1.466 | 0.283 | 1.466 | 89 |
| fidel_ts_nyc_traffic_speed__h/long/target | locked | 1.409 | 0.273 | 1.409 | 89 |
| fidel_ts_nyc_traffic_speed__d/short/target | locked | 1.908 | 0.195 | 1.908 | 89 |
| fulldecent_septa_regionalrail_otp__d/short/target | locked | 0.868 | 0.371 | 0.868 | 12 |
| furusu_japan_max_temperature__d/short/target | locked | 1.836 | 0.279 | 1.836 | 17440 |
| furusu_japan_max_temperature__wsun/short/target | locked | 2.117 | 0.277 | 2.117 | 17337 |
| geoweaver_snowwaterequivalent_trainingdata__d/short/target | locked | 0.010 | 1.013 | 0.010 | 2256 |
| geoweaver_snowwaterequivalent_trainingdata__d/short/cov | locked | 0.010 | 1.013 | 0.010 | 2256 |
| geoweaver_snowwaterequivalent_trainingdata__wsun/short/target | locked | 0.013 | 1.050 | 0.013 | 768 |
| geoweaver_snowwaterequivalent_trainingdata__wsun/short/cov | locked | 0.013 | 1.050 | 0.013 | 768 |
| globalsquiddatabase_tuna_fisheries_dataset__ms/short/target | locked | 1.547 | 0.829 | 1.547 | 1128 |
| hhs_official_weekly_provisional_counts_of_deaths_by_state_and_s__wsat/short/target | locked | 1.359 | 0.034 | 1.359 | 32 |
| hq_bench_quito_corpus_hour__h/short/target | locked | 1.116 | 0.145 | 1.116 | 20000 |
| hq_bench_quito_corpus_hour__h/short/cov | locked | 1.116 | 0.145 | 1.116 | 20000 |
| hq_bench_quito_corpus_hour__h/medium/target | locked | 1.506 | 0.193 | 1.506 | 3000 |
| hq_bench_quito_corpus_hour__h/medium/cov | locked | 1.506 | 0.193 | 1.506 | 3000 |
| hq_bench_quito_corpus_hour__h/long/target | locked | 1.665 | 0.207 | 1.665 | 2000 |
| hq_bench_quito_corpus_hour__h/long/cov | locked | 1.665 | 0.207 | 1.665 | 2000 |
| hq_bench_quito_corpus_hour__d/short/target | locked | 2.113 | 0.150 | 2.113 | 2000 |
| hq_bench_quito_corpus_hour__d/short/cov | locked | 2.113 | 0.150 | 2.113 | 2000 |
| hq_bench_quito_corpus_hour__wsun/short/target | locked | 1.883 | 0.136 | 1.883 | 1000 |
| hq_bench_quito_corpus_hour__wsun/short/cov | locked | 1.883 | 0.136 | 1.883 | 1000 |
| hq_bench_quito_corpus_min__10min/short/target | locked | 1.130 | 0.113 | 1.130 | 11988 |
| hq_bench_quito_corpus_min__10min/short/cov | locked | 1.130 | 0.113 | 1.130 | 11988 |
| hq_bench_quito_corpus_min__10min/medium/target | locked | 1.359 | 0.130 | 1.359 | 999 |
| hq_bench_quito_corpus_min__10min/medium/cov | locked | 1.359 | 0.130 | 1.359 | 999 |
| hq_bench_quito_corpus_min__10min/long/target | locked | 1.180 | 0.123 | 1.180 | 999 |
| hq_bench_quito_corpus_min__10min/long/cov | locked | 1.180 | 0.123 | 1.180 | 999 |
| hq_bench_quito_corpus_min__h/short/target | locked | 1.290 | 0.105 | 1.290 | 1998 |
| hq_bench_quito_corpus_min__h/short/cov | locked | 1.290 | 0.105 | 1.290 | 1998 |
| hq_bench_quitobench_hour_min__h/short/target | locked | 1.061 | 0.115 | 1.061 | 2000 |
| hq_bench_quitobench_hour_min__h/short/cov | locked | 1.061 | 0.115 | 1.061 | 2000 |
| hq_bench_quitobench_hour_min__d/short/target | locked | 2.186 | 0.194 | 2.186 | 814 |
| hq_bench_quitobench_hour_min__d/short/cov | locked | 2.186 | 0.194 | 2.186 | 814 |
| hq_bench_quitobench_hour_min__wsun/short/target | locked | 1.696 | 0.171 | 1.696 | 407 |
| hq_bench_quitobench_hour_min__wsun/short/cov | locked | 1.696 | 0.171 | 1.696 | 407 |
| hsilvosa_omie_electricity__d/short/target | locked | 3.243 | 0.160 | 3.243 | 1 |
| hsilvosa_omie_electricity__d/short/cov | locked | 3.243 | 0.160 | 3.243 | 1 |
| jainam03_rescast_100k_nest__15min/short/target | locked | 0.585 | 0.024 | 0.585 | 120 |
| jainam03_rescast_100k_nest__15min/medium/target | locked | 0.707 | 0.023 | 0.707 | 42 |
| jainam03_rescast_100k_nest__15min/long/target | locked | 0.665 | 0.022 | 0.665 | 24 |
| jainam03_rescast_100k_nest__h/short/target | locked | 0.686 | 0.022 | 0.686 | 108 |
| jainam03_rescast_100k_nest__h/medium/target | locked | 0.541 | 0.019 | 0.541 | 6 |
| jainam03_rescast_100k_nest__h/long/target | locked | 0.827 | 0.026 | 0.827 | 6 |
| jainam03_rescast_100k_nest__d/short/target | locked | 0.866 | 0.030 | 0.866 | 7 |
| jhumbl_tfl_network_demand__d/short/target | locked | 2.283 | 0.329 | 2.283 | 148 |
| jhumbl_tfl_network_demand__d/short/cov | locked | 2.283 | 0.329 | 2.283 | 148 |
| jhumbl_tfl_network_demand__wsun/short/target | locked | 1.832 | 0.140 | 1.832 | 745 |
| jhumbl_tfl_network_demand__wsun/short/cov | locked | 1.832 | 0.140 | 1.832 | 745 |
| jotone_pcfvg_weatherstations__15min/short/target | locked | 0.961 | 0.173 | 0.961 | 15580 |
| jotone_pcfvg_weatherstations__15min/medium/target | locked | 1.655 | 0.242 | 1.655 | 3912 |
| jotone_pcfvg_weatherstations__15min/long/target | locked | 2.177 | 0.277 | 2.177 | 2572 |
| jotone_pcfvg_weatherstations__h/short/target | locked | 1.508 | 0.182 | 1.508 | 9780 |
| jotone_pcfvg_weatherstations__h/medium/target | locked | 6.071 | 0.230 | 6.071 | 927 |
| jotone_pcfvg_weatherstations__h/long/target | locked | 1.249 | 0.191 | 1.249 | 560 |
| jotone_pcfvg_weatherstations__d/short/target | locked | 1.793 | 0.116 | 1.793 | 815 |
| jotone_pcfvg_weatherstations__wsun/short/target | locked | 1.713 | 0.100 | 1.713 | 810 |
| juliensimon_f107_solar_flux__d/short/target | locked | 2.427 | 0.149 | 2.427 | 40 |
| juliensimon_f107_solar_flux__wsun/short/target | locked | 1.603 | 0.156 | 1.603 | 40 |
| juliensimon_iers_earth_orientation__wsun/short/target | locked | 1.080 | 0.586 | 1.080 | 60 |
| matanzig_disney_theme_park_queue_dynamics__d/short/target | locked | 1.526 | 0.007 | 1.526 | 185 |
| matanzig_disney_theme_park_queue_dynamics__d/short/cov | locked | 1.526 | 0.007 | 1.526 | 185 |
| matanzig_disney_theme_park_queue_dynamics__wsun/short/target | locked | 1.449 | 0.005 | 1.449 | 111 |
| matanzig_disney_theme_park_queue_dynamics__wsun/short/cov | locked | 1.449 | 0.005 | 1.449 | 111 |
| mihnathul_celljar__5s/short/target | locked | 0.931 | 0.164 | 0.931 | 515 |
| mihnathul_celljar__5s/short/cov | locked | 0.931 | 0.164 | 0.931 | 515 |
| misikoff_zillow_days_on_market__wsat/short/target | locked | 368.097 | 0.511 | 368.097 | 2637 |
| misikoff_zillow_rentals__me/short/target | locked | 0.897 | 0.047 | 0.897 | 868 |
| misikoff_zillow_sales__wsat/short/target | locked | 1.269 | 0.112 | 1.269 | 570 |
| narendersingh007_india_solar_benchmark_dataset__h/short/target | locked | 0.438 | 0.002 | 0.438 | 20 |
| narendersingh007_india_solar_benchmark_dataset__h/short/cov | locked | 0.438 | 0.002 | 0.438 | 20 |
| narendersingh007_india_solar_benchmark_dataset__h/medium/target | locked | 3.693 | 0.024 | 3.693 | 10 |
| narendersingh007_india_solar_benchmark_dataset__h/medium/cov | locked | 3.693 | 0.024 | 3.693 | 10 |
| narendersingh007_india_solar_benchmark_dataset__h/long/target | locked | 0.850 | 0.006 | 0.850 | 6 |
| narendersingh007_india_solar_benchmark_dataset__h/long/cov | locked | 0.850 | 0.006 | 0.850 | 6 |
| narendersingh007_india_solar_benchmark_dataset__d/short/target | locked | 0.850 | 0.006 | 0.850 | 6 |
| narendersingh007_india_solar_benchmark_dataset__d/short/cov | locked | 0.850 | 0.006 | 0.850 | 6 |
| narendersingh007_india_solar_benchmark_dataset__wsun/short/target | locked | 2.154 | 0.091 | 2.154 | 3 |
| narendersingh007_india_solar_benchmark_dataset__wsun/short/cov | locked | 2.154 | 0.091 | 2.154 | 3 |
| nju_zhangsq_rmisc_britaincoal__5min/short/target | locked | 0.709 | 0.054 | 0.709 | 20 |
| nju_zhangsq_rmisc_britaincoal__5min/short/cov | locked | 0.709 | 0.054 | 0.709 | 20 |
| nju_zhangsq_rmisc_britaincoal__5min/medium/target | locked | 1.053 | 0.080 | 1.053 | 10 |
| nju_zhangsq_rmisc_britaincoal__5min/medium/cov | locked | 1.053 | 0.080 | 1.053 | 10 |
| nju_zhangsq_rmisc_britaincoal__5min/long/target | locked | 1.389 | 0.104 | 1.389 | 6 |
| nju_zhangsq_rmisc_britaincoal__5min/long/cov | locked | 1.389 | 0.104 | 1.389 | 6 |
| nju_zhangsq_rmisc_britaincoal__h/short/target | locked | 1.138 | 0.084 | 1.138 | 8 |
| nju_zhangsq_rmisc_britaincoal__h/short/cov | locked | 1.138 | 0.084 | 1.138 | 8 |
| nju_zhangsq_rmisc_britaincoal__h/medium/target | locked | 1.665 | 0.120 | 1.665 | 1 |
| nju_zhangsq_rmisc_britaincoal__h/medium/cov | locked | 1.665 | 0.120 | 1.665 | 1 |
| nju_zhangsq_rmisc_britaincoal__h/long/target | locked | 0.969 | 0.072 | 0.969 | 1 |
| nju_zhangsq_rmisc_britaincoal__h/long/cov | locked | 0.969 | 0.072 | 0.969 | 1 |
| nju_zhangsq_rmisc_britaincoal__d/short/target | locked | 0.934 | 0.056 | 0.934 | 1 |
| nju_zhangsq_rmisc_britaincoal__d/short/cov | locked | 0.934 | 0.056 | 0.934 | 1 |
| nju_zhangsq_rmisc_frothflotation__h/short/target | locked | 1.221 | 0.494 | 1.221 | 6 |
| nju_zhangsq_rmisc_frothflotation__h/short/cov | locked | 1.221 | 0.494 | 1.221 | 6 |
| nju_zhangsq_rmisc_frothflotation__h/medium/target | locked | 0.940 | 0.420 | 0.940 | 1 |
| nju_zhangsq_rmisc_frothflotation__h/medium/cov | locked | 0.940 | 0.420 | 0.940 | 1 |
| nju_zhangsq_rmisc_frothflotation__h/long/target | locked | 0.960 | 0.488 | 0.960 | 1 |
| nju_zhangsq_rmisc_frothflotation__h/long/cov | locked | 0.960 | 0.488 | 0.960 | 1 |
| nju_zhangsq_rmisc_householdpower__min/short/target | locked | 1.307 | 0.750 | 1.307 | 20 |
| nju_zhangsq_rmisc_householdpower__min/short/cov | locked | 1.307 | 0.750 | 1.307 | 20 |
| nju_zhangsq_rmisc_householdpower__min/medium/target | locked | 1.248 | 1.038 | 1.248 | 10 |
| nju_zhangsq_rmisc_householdpower__min/medium/cov | locked | 1.248 | 1.038 | 1.248 | 10 |
| nju_zhangsq_rmisc_householdpower__min/long/target | locked | 1.257 | 0.793 | 1.257 | 6 |
| nju_zhangsq_rmisc_householdpower__min/long/cov | locked | 1.257 | 0.793 | 1.257 | 6 |
| nju_zhangsq_rmisc_householdpower__h/short/target | locked | 1.815 | 0.910 | 1.815 | 1 |
| nju_zhangsq_rmisc_householdpower__h/short/cov | locked | 1.815 | 0.910 | 1.815 | 1 |
| nju_zhangsq_rmisc_servermachinedataset__min/short/target | locked | 1.558 | 0.332 | 1.558 | 560 |
| nju_zhangsq_rmisc_servermachinedataset__min/short/cov | locked | 1.558 | 0.332 | 1.558 | 560 |
| nju_zhangsq_rmisc_servermachinedataset__min/medium/target | locked | 1.284 | 0.289 | 1.284 | 112 |
| nju_zhangsq_rmisc_servermachinedataset__min/medium/cov | locked | 1.284 | 0.289 | 1.284 | 112 |
| nju_zhangsq_rmisc_servermachinedataset__min/long/target | locked | 1.237 | 0.298 | 1.237 | 84 |
| nju_zhangsq_rmisc_servermachinedataset__min/long/cov | locked | 1.237 | 0.298 | 1.237 | 84 |
| nju_zhangsq_rmisc_servermachinedataset__h/short/target | locked | 1.134 | 0.255 | 1.134 | 28 |
| nju_zhangsq_rmisc_servermachinedataset__h/short/cov | locked | 1.134 | 0.255 | 1.134 | 28 |
| nju_zhangsq_rmisc_smartmeteraus30m__30min/short/target | locked | 0.901 | 0.577 | 0.901 | 18157 |
| nju_zhangsq_rmisc_smartmeteraus30m__30min/medium/target | locked | 1.015 | 0.639 | 1.015 | 2717 |
| nju_zhangsq_rmisc_smartmeteraus30m__30min/long/target | locked | 1.075 | 0.672 | 1.075 | 1811 |
| nju_zhangsq_rmisc_smartmeteraus30m__h/short/target | locked | 0.952 | 0.545 | 0.952 | 16307 |
| nju_zhangsq_rmisc_smartmeteraus30m__h/medium/target | locked | 1.002 | 0.598 | 1.002 | 905 |
| nju_zhangsq_rmisc_smartmeteraus30m__h/long/target | locked | 1.173 | 0.666 | 1.173 | 905 |
| nju_zhangsq_rmisc_smartmeteraus30m__d/short/target | locked | 1.371 | 0.355 | 1.371 | 901 |
| nju_zhangsq_rmisc_smartmeteraus30m__wsun/short/target | locked | 1.680 | 0.271 | 1.680 | 615 |
| nju_zhangsq_rmisc_smartmeteraus60m__h/short/target | locked | 0.948 | 0.548 | 0.948 | 18921 |
| nju_zhangsq_rmisc_smartmeteraus60m__h/medium/target | locked | 1.008 | 0.603 | 1.008 | 994 |
| nju_zhangsq_rmisc_smartmeteraus60m__h/long/target | locked | 1.164 | 0.666 | 1.164 | 993 |
| nju_zhangsq_rmisc_smartmeteraus60m__d/short/target | locked | 1.353 | 0.351 | 1.353 | 989 |
| nju_zhangsq_rmisc_smartmeteraus60m__wsun/short/target | locked | 1.641 | 0.269 | 1.641 | 688 |
| nju_zhangsq_rmisc_smartmeteruk60m__h/short/target | locked | 1.109 | 0.528 | 1.109 | 19960 |
| nju_zhangsq_rmisc_smartmeteruk60m__h/medium/target | locked | 1.190 | 0.570 | 1.190 | 1996 |
| nju_zhangsq_rmisc_smartmeteruk60m__h/long/target | locked | 1.255 | 0.590 | 1.255 | 1990 |
| nju_zhangsq_rmisc_smartmeteruk60m__d/short/target | locked | 1.512 | 0.307 | 1.512 | 1986 |
| nju_zhangsq_rmisc_smartmeteruk60m__wsun/short/target | locked | 2.040 | 0.264 | 2.040 | 952 |
| nju_zhangsq_rmisc_timemmd__d/short/target | locked | 0.970 | 0.361 | 0.970 | 20 |
| nju_zhangsq_rmisc_timemmd__d/short/cov | locked | 0.970 | 0.361 | 0.970 | 20 |
| nju_zhangsq_rmisc_timemmd__wsun/short/target | locked | 0.876 | 0.242 | 0.876 | 20 |
| nju_zhangsq_rmisc_timemmd__wsun/short/cov | locked | 0.876 | 0.242 | 0.876 | 20 |
| nju_zhangsq_rmisc_weeklyfuelpricesitaly__wmon/short/target | locked | 2.989 | 0.028 | 2.989 | 60 |
| nju_zhangsq_rmisc_weeklyfuelpricesitaly__wmon/short/cov | locked | 2.989 | 0.028 | 2.989 | 60 |
| nju_zhangsq_rmisc_weeklyroadfuelprices__wmon/short/target | locked | 4.127 | 0.019 | 4.127 | 11 |
| notadib_usa_corn_belt_crop_yield__ys/short/target | locked | 1.548 | 0.653 | 1.548 | 285 |
| notadib_usa_corn_belt_crop_yield__ys/short/cov | locked | 1.548 | 0.653 | 1.548 | 285 |
| open_predico_benchmark__15min/short/target | locked | 1.449 | 0.267 | 1.449 | 2500 |
| open_predico_benchmark__15min/medium/target | locked | 2.065 | 0.493 | 2.065 | 875 |
| open_predico_benchmark__15min/long/target | locked | 2.088 | 0.497 | 2.088 | 500 |
| open_predico_benchmark__h/short/target | locked | 1.733 | 0.359 | 1.733 | 2250 |
| open_predico_benchmark__h/medium/target | locked | 2.535 | 0.503 | 2.535 | 125 |
| open_predico_benchmark__h/long/target | locked | 2.664 | 0.471 | 2.664 | 125 |
| open_predico_benchmark__d/short/target | locked | 2.812 | 0.424 | 2.812 | 126 |
| openstef_liander2024_energy_forecasting_benchmark_load_measurements_weather_forecasts__15min/short/target | locked | 0.612 | 0.085 | 0.612 | 11520 |
| openstef_liander2024_energy_forecasting_benchmark_load_measurements_weather_forecasts__15min/medium/target | locked | 1.037 | 0.139 | 1.037 | 4032 |
| openstef_liander2024_energy_forecasting_benchmark_load_measurements_weather_forecasts__15min/long/target | locked | 0.765 | 0.110 | 0.765 | 2304 |
| openstef_liander2024_energy_forecasting_benchmark_load_measurements_weather_forecasts__h/short/target | locked | 0.796 | 0.112 | 0.796 | 10368 |
| openstef_liander2024_energy_forecasting_benchmark_load_measurements_weather_forecasts__h/medium/target | locked | 0.986 | 0.139 | 0.986 | 576 |
| openstef_liander2024_energy_forecasting_benchmark_load_measurements_weather_forecasts__h/long/target | locked | 0.827 | 0.103 | 0.827 | 576 |
| openstef_liander2024_energy_forecasting_benchmark_load_measurements_weather_forecasts__d/short/target | locked | 0.991 | 0.087 | 0.991 | 576 |
| opensynth_tudelft_electricity_consumption_1_0_default__30min/short/target | locked | 1.085 | 0.535 | 1.085 | 18480 |
| opensynth_tudelft_electricity_consumption_1_0_default__30min/medium/target | locked | 1.166 | 0.560 | 1.166 | 3785 |
| opensynth_tudelft_electricity_consumption_1_0_default__30min/long/target | locked | 1.156 | 0.562 | 1.156 | 2352 |
| opensynth_tudelft_electricity_consumption_1_0_default__h/short/target | locked | 1.140 | 0.522 | 1.140 | 15820 |
| opensynth_tudelft_electricity_consumption_1_0_default__h/medium/target | locked | 1.255 | 0.566 | 1.255 | 1582 |
| opensynth_tudelft_electricity_consumption_1_0_default__h/long/target | locked | 1.187 | 0.550 | 1.187 | 909 |
| opensynth_tudelft_electricity_consumption_1_0_default__d/short/target | locked | 1.361 | 0.273 | 1.361 | 980 |
| opensynth_tudelft_electricity_consumption_1_0_default__wsun/short/target | locked | 2.123 | 0.261 | 2.123 | 945 |
| phanerozoic_ww3_ndbc_pairs__h/short/target | locked | 0.760 | 0.196 | 0.760 | 980 |
| phanerozoic_ww3_ndbc_pairs__h/short/cov | locked | 0.760 | 0.196 | 0.760 | 980 |
| phanerozoic_ww3_ndbc_pairs__h/medium/target | locked | 1.082 | 0.286 | 1.082 | 98 |
| phanerozoic_ww3_ndbc_pairs__h/medium/cov | locked | 1.082 | 0.286 | 1.082 | 98 |
| phanerozoic_ww3_ndbc_pairs__h/long/target | locked | 1.684 | 0.450 | 1.684 | 62 |
| phanerozoic_ww3_ndbc_pairs__h/long/cov | locked | 1.684 | 0.450 | 1.684 | 62 |
| phanerozoic_ww3_ndbc_pairs__d/short/target | locked | 1.867 | 0.418 | 1.867 | 139 |
| phanerozoic_ww3_ndbc_pairs__d/short/cov | locked | 1.867 | 0.418 | 1.867 | 139 |
| phanerozoic_ww3_ndbc_pairs__wsun/short/target | locked | 1.246 | 0.285 | 1.246 | 137 |
| phanerozoic_ww3_ndbc_pairs__wsun/short/cov | locked | 1.246 | 0.285 | 1.246 | 137 |
| ron_the_code_telangana_time_series_2023_2025__d/short/target | locked | 1.721 | 0.101 | 1.721 | 14 |
| ron_the_code_telangana_time_series_2023_2025__wsun/short/target | locked | 0.992 | 0.093 | 0.992 | 7 |
| salesforce_lotsa_data_borealis__h/short/target | locked | 1.049 | 0.359 | 1.049 | 64 |
| salesforce_lotsa_data_borealis__h/medium/target | locked | 1.178 | 0.404 | 1.178 | 8 |
| salesforce_lotsa_data_borealis__h/long/target | locked | 1.328 | 0.443 | 1.328 | 8 |
| salesforce_lotsa_data_borealis__d/short/target | locked | 1.438 | 0.261 | 1.438 | 8 |
| salesforce_lotsa_data_extended_web_traffic_with_missing__d/short/target | locked | 1.288 | 0.461 | 1.288 | 7216 |
| salesforce_lotsa_data_extended_web_traffic_with_missing__wsun/short/target | locked | 1.152 | 0.419 | 1.152 | 3756 |
| salesforce_lotsa_data_ideal__h/short/target | locked | 0.919 | 0.615 | 0.919 | 1162 |
| salesforce_lotsa_data_ideal__h/medium/target | locked | 1.024 | 0.685 | 1.024 | 183 |
| salesforce_lotsa_data_ideal__h/long/target | locked | 0.977 | 0.643 | 0.977 | 172 |
| salesforce_lotsa_data_ideal__d/short/target | locked | 0.967 | 0.282 | 0.967 | 132 |
| salesforce_lotsa_data_ideal__wsun/short/target | locked | 1.147 | 0.192 | 1.147 | 20 |
| salesforce_lotsa_data_kaggle_web_traffic_weekly__wsun/short/target | locked | 1.042 | 0.442 | 1.042 | 1000 |
| salesforce_lotsa_data_lcl__h/short/target | locked | 1.196 | 0.542 | 1.196 | 14200 |
| salesforce_lotsa_data_lcl__h/medium/target | locked | 1.264 | 0.575 | 1.264 | 1419 |
| salesforce_lotsa_data_lcl__h/long/target | locked | 1.392 | 0.626 | 1.392 | 708 |
| salesforce_lotsa_data_lcl__d/short/target | locked | 1.674 | 0.322 | 1.674 | 701 |
| salesforce_lotsa_data_lcl__wsun/short/target | locked | 1.747 | 0.227 | 1.747 | 665 |
| salesforce_lotsa_data_los_loop__5min/short/target | locked | 0.620 | 0.141 | 0.620 | 4140 |
| salesforce_lotsa_data_los_loop__5min/medium/target | locked | 1.065 | 0.326 | 1.065 | 1449 |
| salesforce_lotsa_data_los_loop__5min/long/target | locked | 1.261 | 0.308 | 1.261 | 828 |
| salesforce_lotsa_data_los_loop__h/short/target | locked | 1.038 | 0.234 | 1.038 | 1035 |
| salesforce_lotsa_data_los_loop__h/medium/target | locked | 1.312 | 0.270 | 1.312 | 207 |
| salesforce_lotsa_data_los_loop__h/long/target | locked | 1.443 | 0.289 | 1.443 | 207 |
| salesforce_lotsa_data_pems03__5min/short/target | locked | 0.825 | 0.149 | 0.825 | 6960 |
| salesforce_lotsa_data_pems03__5min/medium/target | locked | 1.081 | 0.195 | 1.081 | 1740 |
| salesforce_lotsa_data_pems03__5min/long/target | locked | 1.286 | 0.229 | 1.286 | 1044 |
| salesforce_lotsa_data_pems03__h/short/target | locked | 1.024 | 0.145 | 1.024 | 1392 |
| salesforce_lotsa_data_pems03__h/medium/target | locked | 1.504 | 0.206 | 1.504 | 348 |
| salesforce_lotsa_data_pems03__h/long/target | locked | 1.199 | 0.164 | 1.199 | 348 |
| salesforce_lotsa_data_pems07__5min/short/target | locked | 1.012 | 0.198 | 1.012 | 14960 |
| salesforce_lotsa_data_pems07__5min/medium/target | locked | 1.071 | 0.173 | 1.071 | 3740 |
| salesforce_lotsa_data_pems07__5min/long/target | locked | 1.272 | 0.202 | 1.272 | 2244 |
| salesforce_lotsa_data_pems07__h/short/target | locked | 1.088 | 0.146 | 1.088 | 2992 |
| salesforce_lotsa_data_pems07__h/medium/target | locked | 1.050 | 0.141 | 1.050 | 748 |
| salesforce_lotsa_data_pems07__h/long/target | locked | 1.108 | 0.147 | 1.108 | 748 |
| salesforce_lotsa_data_residential_pv_power__min/short/target | locked | 0.251 | 0.416 | 0.251 | 11940 |
| salesforce_lotsa_data_residential_pv_power__min/short/joint | locked | 0.251 | 0.416 | 0.251 | 11940 |
| salesforce_lotsa_data_residential_pv_power__min/medium/target | locked | 1.070 | 0.607 | 1.070 | 5220 |
| salesforce_lotsa_data_residential_pv_power__min/medium/joint | locked | 1.070 | 0.607 | 1.070 | 5220 |
| salesforce_lotsa_data_residential_pv_power__min/long/target | locked | 1.131 | 0.652 | 1.131 | 3150 |
| salesforce_lotsa_data_residential_pv_power__min/long/joint | locked | 1.131 | 0.652 | 1.131 | 3150 |
| salesforce_lotsa_data_residential_pv_power__h/short/target | locked | 1.053 | 0.457 | 1.053 | 630 |
| salesforce_lotsa_data_residential_pv_power__h/short/joint | locked | 1.053 | 0.457 | 1.053 | 630 |
| salesforce_lotsa_data_uber_tlc_hourly__h/short/target | locked | 1.433 | 0.474 | 1.433 | 2358 |
| salesforce_lotsa_data_uber_tlc_hourly__h/medium/target | locked | 1.310 | 0.384 | 1.310 | 262 |
| salesforce_lotsa_data_uber_tlc_hourly__h/long/target | locked | 1.682 | 0.543 | 1.682 | 262 |
| salesforce_lotsa_data_uber_tlc_hourly__d/short/target | locked | 2.204 | 0.319 | 2.204 | 257 |
| salesforce_lotsa_data_wind_power__4s/short/target | locked | 1.211 | 0.751 | 1.211 | 20 |
| salesforce_lotsa_data_wind_power__4s/medium/target | locked | 0.900 | 1.491 | 0.900 | 10 |
| salesforce_lotsa_data_wind_power__4s/long/target | locked | 0.972 | 1.168 | 0.972 | 6 |
| samuelm0422_hourly_electricity_demand_brazil_dataset__h/short/target | locked | 1.366 | 0.070 | 1.366 | 80 |
| samuelm0422_hourly_electricity_demand_brazil_dataset__h/medium/target | locked | 1.638 | 0.090 | 1.638 | 36 |
| samuelm0422_hourly_electricity_demand_brazil_dataset__h/long/target | locked | 1.420 | 0.081 | 1.420 | 24 |
| samuelm0422_hourly_electricity_demand_brazil_dataset__d/short/target | locked | 1.405 | 0.074 | 1.405 | 24 |
| samuelm0422_hourly_electricity_demand_brazil_dataset__wsun/short/target | locked | 2.164 | 0.059 | 2.164 | 12 |
| witgaw_metr_la__5min/short/target | locked | 0.633 | 0.143 | 0.633 | 4140 |
| witgaw_metr_la__5min/short/cov | locked | 0.633 | 0.143 | 0.633 | 4140 |
| witgaw_metr_la__5min/medium/target | locked | 1.089 | 0.343 | 1.089 | 1449 |
| witgaw_metr_la__5min/medium/cov | locked | 1.089 | 0.343 | 1.089 | 1449 |
| witgaw_metr_la__5min/long/target | locked | 1.239 | 0.310 | 1.239 | 828 |
| witgaw_metr_la__5min/long/cov | locked | 1.239 | 0.310 | 1.239 | 828 |
| witgaw_metr_la__h/short/target | locked | 0.990 | 0.230 | 0.990 | 1035 |
| witgaw_metr_la__h/short/cov | locked | 0.990 | 0.230 | 0.990 | 1035 |
| witgaw_metr_la__h/medium/target | locked | 1.317 | 0.271 | 1.317 | 207 |
| witgaw_metr_la__h/medium/cov | locked | 1.317 | 0.271 | 1.317 | 207 |
| witgaw_metr_la__h/long/target | locked | 1.455 | 0.292 | 1.455 | 207 |
| witgaw_metr_la__h/long/cov | locked | 1.455 | 0.292 | 1.455 | 207 |
| yuiseki_un_wpp_wpp2024__ys/short/target | locked | 31.565 | 0.255 | 31.565 | 5 |
| yuiseki_un_wpp_wpp2024__ys/short/joint | locked | 31.565 | 0.255 | 31.565 | 5 |
