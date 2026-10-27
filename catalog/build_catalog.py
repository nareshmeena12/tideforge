import csv, collections

# url_verified = "yes": that URL (or a page on the same site describing the dataset) appeared in this session's search results / pasted reports.
# url_verified = "no": URL written from memory, check before use.
# overlap: bench = in GIFT-Eval / fev-bench / TIME eval (diagnosis only)
#          pretrain = in LOTSA / GIFT-Eval pretrain / Chronos corpus (fine for synthetic-only twins; reference models saw it)
#          likely_time = same family as a TIME dataset (TIME's full 50-dataset list was NOT retrieved)
#          mixed = some sub-series overlap, split by sub-series
#          clean? = no overlap known to me, run the audit script
#          derived = built from a dataset listed elsewhere

R = []
def r(name, domain, behaviours, freq, access, licence, url, ver, overlap, notes=""):
    R.append(dict(name=name, domain=domain, behaviours=behaviours, freq=freq, access=access,
                  licence=licence, url=url, url_verified=ver, overlap=overlap, notes=notes))

# ---------------- ENERGY: load and grid ----------------
r("Open Power System Data: time series", "Energy-load", "multi-seasonal;holiday;weather-driven;covariate-rich;price-spikes", "H/15min", "direct download (CSV/ZIP)", "CC BY 4.0 (check package)", "https://data.open-power-system-data.org/time_series/", "yes", "mixed", "Load, wind, solar, day-ahead prices for many European countries; derived from ENTSO-E, overlaps fev entsoe tasks")
r("ENTSO-E Transparency Platform", "Energy-load", "multi-seasonal;holiday;generation-mix;price-spikes", "H/15min", "API token (free)", "ENTSO-E terms", "https://transparency.entsoe.eu/", "no", "mixed", "Load, generation by type, day-ahead prices, flows; fev has entsoe tasks, pull other countries/years")
r("EIA-930 US hourly grid operations", "Energy-load", "multi-seasonal;weather-driven;heat-waves;regime-switch", "H", "API key (free)", "US public domain", "https://www.eia.gov/opendata/", "no", "clean?", "Demand, forecast demand, net generation, interchange per balancing authority since 2015")
r("Elexon BMRS (Great Britain)", "Energy-load", "multi-seasonal;holiday;price-spikes", "30min", "API (free)", "Elexon open licence", "https://wiki.openmod-initiative.org/wiki/Electricity_demand", "yes", "clean?", "Half-hourly GB demand and system prices; pointer page lists the sources")
r("RTE eCO2mix (France)", "Energy-load", "multi-seasonal;weather-driven;holiday;generation-mix", "30min", "direct download", "Etalab open licence", "https://wiki.openmod-initiative.org/wiki/Electricity_demand", "yes", "clean?", "French national and regional load and generation (Open Data Reseaux-Energies)")
r("Singapore EMA half-hourly demand", "Energy-load", "multi-seasonal;holiday;tropical-stable", "30min", "direct download", "Singapore open data licence", "https://wiki.openmod-initiative.org/wiki/Electricity_demand", "yes", "clean?", "Downloadable per month from 2004")
r("AEMO NEM price and demand (VIC1 etc.)", "Energy-load", "multi-seasonal;holiday;price-spikes;heat-waves", "5min", "direct download (CSV per month)", "AEMO permits reuse with attribution", "https://aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data", "yes", "clean?", "5-min demand and price for 5 Australian regions; the older Victoria series is in GIFT-pretrain, use 2022+")
r("ERCOT hourly load by weather zone", "Energy-load", "multi-seasonal;heat-waves;price-spikes", "H", "direct download", "ERCOT public data", "https://www.ercot.com/gridinfo/load/load_hist", "no", "bench", "fev has ercot tasks")
r("CAISO OASIS (load, prices, curtailment)", "Energy-load", "multi-seasonal;solar-duck-curve;price-spikes;negative-prices", "5min/H", "API (free)", "CAISO terms", "http://oasis.caiso.com/", "no", "clean?", "Negative prices and curtailment are rare behaviours worth having")
r("PJM Data Miner (load, LMP)", "Energy-load", "multi-seasonal;price-spikes;heat-waves", "H/5min", "API key (free)", "PJM terms", "https://dataminer2.pjm.com/", "no", "mixed", "fev has epf_pjm (prices); load is not in benchmarks as far as I know")
r("ISO-NE zonal load and prices", "Energy-load", "multi-seasonal;heat-waves;winter-peaks", "H/5min", "direct download", "ISO-NE terms", "https://www.iso-ne.com/isoexpress/", "no", "clean?", "")
r("NYISO load and LBMP", "Energy-load", "multi-seasonal;heat-waves;price-spikes", "5min/H", "direct download", "NYISO terms", "https://www.nyiso.com/load-data", "no", "clean?", "")
r("IESO Ontario demand", "Energy-load", "multi-seasonal;winter-peaks;holiday", "H", "direct download", "Open Government Licence Ontario", "https://www.ieso.ca/power-data", "no", "clean?", "")
r("Grid-India (POSOCO) demand, India regions", "Energy-load", "multi-seasonal;monsoon;festival-spikes;outages", "15min/D", "direct download / reports", "check terms", "https://report.grid-india.in/", "no", "clean?", "Matches your location and Diwali/monsoon behaviours; scraping may be needed")
r("Quarter-hourly electricity generation Austria 2015-2026", "Energy-gen", "multi-seasonal;hydro-wind-solar-mix", "15min", "direct download", "CC BY 4.0", "https://ieee-dataport.org/documents/quarter-hourly-electricity-generation-dataset-austria-2015-2026", "yes", "clean?", "ENTSO-E derived")
r("Quarter-hourly electricity generation Luxembourg 2015-2026", "Energy-gen", "multi-seasonal;small-system-noise", "15min", "direct download", "check on page", "https://ieee-dataport.org/documents/quarter-hourly-actual-electricity-generation-dataset-luxembourg-2015-2026", "yes", "clean?", "ENTSO-E derived via SMARD")
r("Elia (Belgium) wind and PV 15-min", "Energy-gen", "weather-driven;saturation;diurnal", "15min", "direct download", "Elia open data licence (CC BY 4.0)", "https://publications.jrc.ec.europa.eu/repository/handle/JRC135912", "yes", "clean?", "Original at elia.be open data")
r("UCI ElectricityLoadDiagrams (370 clients)", "Energy-load", "multi-seasonal;customer-heterogeneity;missing-start", "15min", "direct download", "CC BY 4.0", "https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014", "no", "bench", "GIFT-Eval electricity")
r("GEFCom2012/2014/2017 load", "Energy-load", "multi-seasonal;weather-driven;covariate-rich", "H", "direct download", "competition terms", "https://www.kaggle.com/c/global-energy-forecasting-competition-2012-load-forecasting", "no", "bench", "In fev (gefcom) and LOTSA")
r("PSML multi-scale grid dataset", "Energy-load", "multi-seasonal;renewable-sim;weather-driven", "H/1min", "direct download", "check on page", "https://arxiv.org/pdf/2110.06324", "yes", "clean?", "66 US load zones with weather and simulated renewables, partly synthetic")

# ---------------- Smart meters / buildings / heat ----------------
r("London Smart Meters (Zenodo, Monash copy)", "Buildings-heat", "household-load;multi-seasonal;holiday;missing", "30min", "direct download", "CC BY 4.0", "https://zenodo.org/records/3898331", "yes", "pretrain", "5,560 households Nov 2011 - Feb 2014")
r("Low Carbon London smart meter data", "Buildings-heat", "household-load;tariff-effects;multi-seasonal", "30min", "direct download", "UK OGL", "https://www.data.gov.uk/dataset/df8e55d4-c6d0-498e-abd8-f4c678766cbe/smartmeter-energy-consumption-data-in-london-households1", "yes", "pretrain", "LCL is in GIFT-Eval pretrain")
r("Building Data Genome Project 2 (BDG2)", "Buildings-heat", "building-load;chilled-water;steam;hot-water;weather-driven", "H", "direct download (GitHub)", "CC BY-NC-SA 4.0 for the repo", "https://github.com/buds-lab/building-data-genome-project-2", "yes", "pretrain", "1,636 buildings, 3,053 meters, two years. Steam and hot-water meters are the closest thing to heat demand in this list")
r("LEAD1.0 annotated building energy anomalies", "Buildings-heat", "building-load;labelled-anomalies", "H", "direct download", "check on page", "https://arxiv.org/pdf/2203.17256", "yes", "derived", "Derived from ASHRAE GEPIII (same source as BDG2)")
r("SERL Observatory (UK households, electricity+gas)", "Buildings-heat", "household-load;gas-heating;weather-driven;multi-seasonal", "30min/D", "UK Data Service registration", "UKDS terms", "https://discovery-pp.ucl.ac.uk/10137253/1/energies-14-06934-v2.pdf", "yes", "clean?", "13,000 households, gas covers heating behaviour. Access takes a registration step")
r("Irish CER smart meter trial", "Buildings-heat", "household-load;tariff-trial;multi-seasonal", "30min", "ISSDA application", "ISSDA terms", "https://www.ucd.ie/issda/data/commissionforenergyregulationcer", "no", "clean?", "Listed in a smart-meter data review; access needs an application")
r("Pecan Street Dataport", "Buildings-heat", "household-load;appliance-level;solar-homes", "1min/15min", "free for academics, registration", "Pecan Street terms", "https://dataport.pecanstreet.org/", "yes", "clean?", "Restricted access")
r("UMass Smart* microgrid", "Buildings-heat", "household-load;1-min", "1min", "direct download", "check", "https://traces.cs.umass.edu/docs/traces/smartstar", "yes", "clean?", "")
r("Ausgrid solar home dataset", "Buildings-heat", "household-load;rooftop-PV;multi-seasonal", "30min", "direct download", "CC BY", "https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data", "no", "clean?", "Listed in smart-meter review; URL on ausgrid.com.au to confirm")
r("Energi Data Service (Denmark): consumption and prices", "Buildings-heat", "multi-seasonal;district-heating-possible", "H", "API (free)", "CC BY 4.0", "https://www.energidataservice.dk/", "no", "clean?", "I did not confirm a district heating series here, check the catalogue")

# ---------------- Energy fuels / oil and gas ----------------
r("EIA daily spot prices: WTI, Brent, Henry Hub", "Oil-gas-fuels", "random-walk;jumps;regime-switch;volatility-clustering", "D", "API/CSV", "US public domain", "https://www.eia.gov/opendata/", "no", "clean?", "FRED mirrors them (DCOILWTICO etc.)")
r("EIA weekly petroleum and natural gas storage", "Oil-gas-fuels", "multi-seasonal;storage-cycle;holiday", "W", "API/CSV", "US public domain", "https://www.eia.gov/opendata/", "no", "clean?", "Gas storage has a strong yearly injection-withdrawal cycle")
r("Equinor Volve field production data", "Oil-gas-fuels", "decline-curve;shut-ins;water-breakthrough;choke-changes", "D", "Equinor data village download", "Equinor open licence (check)", "https://jpt.spe.org/twa/lstm-for-production-part-2", "yes", "clean?", "Daily oil/gas/water per well plus pressure and temperature; page is a description, not the download")
r("North Dakota (NDIC) monthly well production", "Oil-gas-fuels", "decline-curve;shut-ins;fracking-steps", "M", "bulk download (some paid)", "check", "https://www.dmr.nd.gov/oilgas/", "no", "clean?", "Thousands of wells with real decline behaviour")
r("Texas RRC production data", "Oil-gas-fuels", "decline-curve;shut-ins", "M", "bulk download", "TX public data", "https://www.rrc.texas.gov/", "no", "clean?", "")
r("Norwegian Offshore Directorate production (Sodir)", "Oil-gas-fuels", "decline-curve;maintenance-shutdowns", "M/D", "direct download", "NLOD", "https://www.sodir.no/en/", "no", "clean?", "")
r("World Bank Pink Sheet commodity prices", "Agriculture-commodities", "random-walk;regime-switch;supply-shocks;seasonality-in-crops", "M", "direct download (xlsx)", "CC BY 4.0", "https://www.worldbank.org/en/research/commodity-markets", "yes", "clean?", "100+ commodity series since 1960; the download URL carries a hash that changes monthly")

# ---------------- Solar, wind ----------------
r("NREL PVDAQ public PV systems", "Energy-gen", "PV;diurnal;soiling;degradation;cloud-noise", "15min", "AWS open data / OEDI", "CC BY", "https://data.amerigeoss.org/dataset/photovoltaic-data-acquisition-pvdaq-public-datasets-afc81", "yes", "clean?", "Many sites with irradiance and temperature")
r("NREL Solar Power Data for Integration Studies", "Energy-gen", "PV;diurnal;cloud-noise;simulated", "5min", "direct download", "NREL open", "https://www.nrel.gov/grid/solar-power-data.html", "no", "pretrain?", "Alabama solar is the GIFT-Eval solar dataset; other states are not")
r("NIST campus PV arrays and weather", "Energy-gen", "PV;diurnal;weather-covariates", "1min", "direct download", "US public domain", "https://github.com/Charlie5DH/Solar-Power-Datasets-and-Resources", "yes", "clean?", "Pointer list of solar datasets")
r("DKASC Alice Springs PV", "Energy-gen", "PV;diurnal;clear-sky;desert-climate", "5min", "direct download", "check", "https://dkasolarcentre.com.au/", "no", "clean?", "Long record from one site")
r("Wind and solar power generation (Mendeley)", "Energy-gen", "wind;PV;weather-driven", "H", "direct download", "check on page", "https://data.mendeley.com/datasets/gxc6j5btrx/1", "yes", "clean?", "Small, one location")
r("Kelmarsh and Penmanshiel wind farm SCADA", "Energy-gen", "wind;turbine-SCADA;curtailment;faults", "10min", "Zenodo", "CC BY 4.0", "https://zenodo.org/search?q=Kelmarsh%20wind%20farm%20SCADA", "no", "clean?", "UK wind farms with SCADA and status logs")
r("NREL WIND Toolkit", "Energy-gen", "wind;weather-simulation", "5min", "AWS open data", "CC BY 4.0", "https://registry.opendata.aws/nrel-pds-wtk/", "no", "clean?", "Simulated, very large")
r("Spain energy and weather (Chronos extra)", "Energy-gen", "load;generation;weather-covariates", "H", "Kaggle (chronos_datasets_extra)", "CC0", "https://huggingface.co/datasets/autogluon/chronos_datasets_extra", "no", "pool", "Already in your real post-training allow-list")

# ---------------- EV and parking ----------------
r("ACN-Data (Caltech/JPL EV charging)", "EV-parking", "sessions;workday;deadline-driven;holiday", "event", "API token (free)", "CC BY 4.0", "https://ev.caltech.edu/research", "yes", "clean?", "30,000+ sessions at launch, now much more; aggregate to power per site")
r("Korean EV charging transactions (Sci Data 2024)", "EV-parking", "sessions;multi-seasonal;charger-level", "event/15min", "direct download", "CC BY 4.0", "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10907370/", "yes", "clean?", "72,856 sessions, 2,119 chargers")
r("ElaadNL EV charging (Netherlands)", "EV-parking", "sessions;commuter;multi-seasonal", "event", "request/download", "check", "https://platform.elaad.io/", "no", "clean?", "Mentioned in the EV dataset papers")
r("UK public AC charger usage (Strathclyde study)", "EV-parking", "sessions;urban-usage", "event", "per-paper data statement", "check", "https://strathprints.strath.ac.uk/87404/1/Bayram-etal-IET-EVI-2023-Usage-analysis-of-public-AC-chargers-in-the-UK.pdf", "yes", "clean?", "Small, four months. Low priority")
r("Birmingham car park occupancy (UCI)", "EV-parking", "occupancy;daily-weekly;capacity-cap;event-days", "30min", "direct download", "CC BY 4.0", "https://archive.ics.uci.edu/dataset/482/parking+birmingham", "no", "clean?", "Occupancy saturates at capacity")
r("SFpark parking sensors", "EV-parking", "occupancy;pricing-effects", "event", "archive", "check", "https://www.sfmta.com/reports/sfpark-data", "no", "clean?", "")
r("Melbourne on-street parking sensors", "EV-parking", "occupancy;duration;weekly", "event", "open data portal", "CC BY 4.0", "https://data.melbourne.vic.gov.au/", "no", "clean?", "")

# ---------------- Weather and climate ----------------
r("NOAA GHCN-Daily", "Weather-climate", "annual-cycle;precipitation-zero-inflated;temperature-AR;missing", "D", "direct download / AWS", "NOAA open", "https://catalog-beta.data.gov/dataset/global-historical-climatology-network-daily-ghcn-daily-version-3", "yes", "clean?", "100,000+ stations; zero-inflated rain, heat waves, gaps. Weather is already 96% of your real pool, so cap this")
r("NOAA GHCN-Hourly (replaces ISD)", "Weather-climate", "diurnal;annual-cycle;wind;pressure;gappy", "H", "AWS open data", "NOAA open", "https://registry.opendata.aws/noaa-ghcnh/", "yes", "clean?", "Hourly station data; some ISD stations appear in pretraining corpora")
r("Open-Meteo historical weather API (ERA5 reanalysis)", "Weather-climate", "diurnal;annual-cycle;multivariate;global", "H", "free API (non-commercial)", "CC BY 4.0", "https://open-meteo.com/en/docs/historical-weather-api", "yes", "pretrain", "ERA5 is in GIFT-Eval pretrain. Useful as covariates for other datasets")
r("NASA POWER daily meteorology and solar", "Weather-climate", "annual-cycle;solar-radiation;global-grid", "D/H", "API (free)", "NASA open", "https://power.larc.nasa.gov/", "no", "clean?", "")
r("NDBC ocean buoys (waves, SST, wind)", "Weather-climate", "waves;storm-bursts;diurnal;gappy", "10min/H", "direct download", "NOAA open", "https://www.ndbc.noaa.gov/", "no", "clean?", "Real storm bursts and sensor gaps")
r("NOAA CO-OPS tides and water levels", "Weather-climate", "tidal-multi-period;storm-surge;sea-level-trend", "6min/H", "API (free)", "NOAA open", "https://tidesandcurrents.noaa.gov/", "no", "clean?", "Tides are a clean sum of fixed periods, a good oracle-like test")
r("PSMSL sea level records", "Weather-climate", "trend;annual-cycle;long-memory", "M/Y", "direct download", "free for research", "https://psmsl.org/", "no", "clean?", "")
r("Mauna Loa CO2 (NOAA GML)", "Weather-climate", "trend-acceleration;annual-cycle", "D/M", "direct download", "NOAA open", "https://gml.noaa.gov/ccgg/trends/", "yes", "pretrain?", "Classic series; short and clean")
r("ENSO indices (ONI, Nino 3.4)", "Weather-climate", "quasi-periodic;irregular-cycle", "M", "direct download", "NOAA open", "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php", "no", "clean?", "Quasi-periodic with 2-7 year cycles, hard case")
r("SNOTEL snowpack (USDA NRCS)", "Weather-climate", "annual-cycle;accumulation-melt;thresholds", "D", "API", "US public domain", "https://www.nrcs.usda.gov/wps/portal/wcc/home/", "no", "clean?", "")
r("India Meteorological Department gridded rainfall", "Weather-climate", "monsoon;zero-inflated;annual-cycle", "D", "request/download", "IMD terms", "https://www.imdpune.gov.in/", "no", "clean?", "Monsoon onset timing varies a lot, relevant for India use cases")
r("WeatherBench / ERA5 daily/hourly (in your pool)", "Weather-climate", "global-grid;multivariate", "H/D/W", "chronos_datasets", "CC BY 4.0", "https://huggingface.co/datasets/autogluon/chronos_datasets", "yes", "pool", "Already in the real post-training allow-list")
r("Jena Climate (Max Planck)", "Weather-climate", "diurnal;annual-cycle;multivariate", "10min", "direct download", "CC BY 4.0", "https://www.bgc-jena.mpg.de/wetter/", "no", "bench", "GIFT-Eval jena_weather")
r("Sunspot numbers (SILSO)", "Space-geo", "quasi-periodic;11-year-cycle;amplitude-modulated", "D/M", "direct download", "CC BY-NC", "https://www.sidc.be/silso/newdataset", "yes", "pretrain", "Monash copy: https://zenodo.org/records/4632775 (in Chronos/LOTSA)")
r("NASA OMNIWeb solar wind and Kp index", "Space-geo", "space-weather;bursts;27-day-solar-rotation", "1min/H", "direct download", "NASA open", "https://omniweb.gsfc.nasa.gov/", "no", "clean?", "Bursty, quasi-periodic")
r("USGS earthquake catalog (event counts)", "Space-geo", "point-process;aftershock-clustering;Hawkes-like", "event/D", "API (free)", "US public domain", "https://earthquake.usgs.gov/", "yes", "clean?", "Aggregate to daily counts; Hawkes-like clustering")
r("NASA FIRMS active fire detections", "Space-geo", "seasonal;bursty;regional", "D", "API key (free)", "NASA open", "https://firms.modaps.eosdis.nasa.gov/", "no", "clean?", "")
r("Kepler/TESS light curves (MAST)", "Space-geo", "quasi-periodic;transits;stellar-variability;noise", "30min/2min", "MAST download", "NASA open", "https://archive.stsci.edu/", "no", "clean?", "Quasi-periodic with real noise, a different regime from business data")

# ---------------- Hydrology / water ----------------
r("CAMELS (US, 671 basins)", "Hydrology", "streamflow;rain-driven-spikes;snowmelt;recession", "D", "direct download", "open (UCAR/Zenodo)", "https://zenodo.org/records/15529996", "yes", "clean?", "Daily forcing and streamflow 1980-2014")
r("CAMELS-GB v2 (671 catchments)", "Hydrology", "streamflow;sub-daily;groundwater;flood-peaks", "H/D", "EIDC download", "OGL", "https://essd.copernicus.org/preprints/essd-2025-608/essd-2025-608-manuscript-version4.pdf", "yes", "clean?", "Includes hourly flow and 55 groundwater boreholes")
r("Caravan global hydrology collection", "Hydrology", "streamflow;global;multi-climate", "D", "Zenodo", "mixed, check", "https://essd.copernicus.org/preprints/essd-2025-608/essd-2025-608-manuscript-version4.pdf", "yes", "clean?", "Referenced as a CAMELS-style global set; confirm the download page")
r("USGS NWIS streamflow and gauge height", "Hydrology", "streamflow;flood-spikes;gappy;ice-effects", "15min/D", "API (free)", "US public domain", "https://waterdata.usgs.gov/nwis", "no", "clean?", "Pull hundreds of gauges from many regions, cap per basin")
r("USBR reservoir storage levels", "Hydrology", "annual-cycle;managed-release;drought-trend", "D", "API", "US public domain", "https://www.usbr.gov/", "no", "clean?", "")
r("SWaT water treatment testbed (iTrust)", "Hydrology", "process-control;attacks;actuator-states", "1s", "request form", "restricted", "https://ibm.github.io/AssetOpsBench/docs/external-industrial-datasets.html", "yes", "clean?", "Needs a request to SUTD; WADI is the companion")

# ---------------- Air quality ----------------
r("OpenAQ global air quality", "Air-quality", "diurnal;episodes;spikes;gappy;sensor-faults", "H", "free API / AWS open data", "free for non-commercial", "https://docs.openaq.org/about/about", "yes", "clean?", "43 billion data points; stations vary hugely in quality")
r("EPA AQS hourly pollutant files", "Air-quality", "diurnal;seasonal;episodes;QC-flags", "H", "direct download (yearly ZIP)", "US public domain", "https://aqs.epa.gov/aqsweb/airdata/hourly_44201_2018.zip", "yes", "clean?", "One file per pollutant per year")
r("WAQI historical air quality (aqicn)", "Air-quality", "diurnal;episodes;global", "D", "free download after registration", "check terms", "https://aqicn.org/data-platform/", "yes", "clean?", "Daily averages from 100+ countries")
r("UCI Air Quality (Italy sensors)", "Air-quality", "diurnal;sensor-drift;covariates", "H", "direct download", "CC BY 4.0", "https://archive.ics.uci.edu/dataset/360/air+quality", "no", "bench", "fev uci_air_quality")
r("KDD Cup 2018 air quality (Beijing/London)", "Air-quality", "diurnal;episodes;gappy", "H", "direct download", "competition", "https://www.kdd.org/kdd2018/kdd-cup", "no", "bench", "GIFT-Eval kdd_cup_2018")
r("Beijing multi-site air quality (UCI)", "Air-quality", "diurnal;seasonal;haze-episodes;weather-covariates", "H", "direct download", "CC BY 4.0", "https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data", "no", "clean?", "12 sites, 2013-2017")
r("India CPCB air quality (CAAQMS)", "Air-quality", "diurnal;winter-smog;crop-burning-spikes;festival-spikes", "H/15min", "portal download", "NDSAP", "https://cpcb.nic.in/", "no", "clean?", "Diwali and stubble-burning spikes are real event behaviours")
r("EEA air quality e-reporting", "Air-quality", "diurnal;seasonal;episodes", "H", "bulk download", "EEA open", "https://www.eea.europa.eu/en/datahub", "no", "clean?", "")

# ---------------- Epidemics and public health ----------------
r("Project Tycho notifiable disease counts", "Epidemics-health", "epidemic-waves;seasonal;vaccination-regime-change;reporting-artefacts", "W/M", "Zenodo (CSV)", "CC BY 4.0", "https://www.tycho.pitt.edu/data", "yes", "pretrain", "Measles, dengue and more per country, 1888-2014 for the US. In LOTSA")
r("OpenDengue (national and subnational dengue)", "Epidemics-health", "epidemic-waves;seasonal;climate-driven", "W/M", "direct download", "CC BY 4.0", "https://epi-eval-readme.static.hf.space/README.md", "yes", "clean?", "Standardised copy listed in the epi-eval HF org")
r("Global dengue case counts (Sci Data 2024)", "Epidemics-health", "epidemic-waves;seasonal;irregular-reporting", "W/M", "direct download", "CC BY 4.0", "https://pmc.ncbi.nlm.nih.gov/articles/PMC10940302", "yes", "clean?", "56M cases, 102 countries, 1924-2023; >95% of records weekly or monthly")
r("CDC FluView ILINet and WHO-NREVSS", "Epidemics-health", "epidemic-waves;seasonal;annual-peak-timing-varies", "W", "API / download", "US public domain", "https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html", "no", "pretrain?", "CDC FluView is in LOTSA to my knowledge, confirm")
r("WHO FluNet global influenza", "Epidemics-health", "epidemic-waves;seasonal;southern-northern-offset", "W", "direct download", "WHO open", "https://www.who.int/tools/flunet", "no", "likely_time", "TIME has a Global_Influenza/W dataset; treat this as overlapping")
r("CDC NSSP emergency department visits", "Epidemics-health", "epidemic-waves;weekday-effects;respiratory", "W", "API / HF epi-eval", "US public domain", "https://epi-eval-readme.static.hf.space/README.md", "yes", "clean?", "Listed in epi-eval HF org")
r("CDC NWSS wastewater surveillance", "Epidemics-health", "viral-load;waves;lead-indicator;noisy", "W", "API", "US public domain", "https://www.cdc.gov/nwss/", "no", "clean?", "A leading indicator with high noise")
r("Our World in Data COVID-19", "Epidemics-health", "epidemic-waves;weekday-reporting;cumulative;holiday-dips", "D", "GitHub CSV", "CC BY 4.0", "https://github.com/owid/covid-19-data", "no", "mixed", "GIFT-Eval covid_deaths is from JHU; OWID overlaps heavily")
r("JHU CSSE COVID-19 time series", "Epidemics-health", "epidemic-waves;cumulative;reporting-jumps", "D", "GitHub CSV", "CC BY 4.0", "https://github.com/CSSEGISandData/COVID-19", "yes", "bench", "Source of GIFT-Eval covid_deaths")
r("UK coronavirus dashboard (nations)", "Epidemics-health", "epidemic-waves;weekday-reporting", "D", "API", "OGL", "https://ukhsa-dashboard.data.gov.uk/", "no", "bench", "fev uk_covid_nation")
r("Google Community Mobility Reports", "Epidemics-health", "step-changes;holiday;lockdown-regimes", "D", "direct download", "Google terms", "https://epi-eval-readme.static.hf.space/README.md", "yes", "clean?", "Behavioural covariate for epidemics; discontinued upstream, archived copies exist")
r("WHO measles suspected cases (country monthly)", "Epidemics-health", "epidemic-waves;seasonal;underreporting", "M", "WHO download", "WHO open", "https://arxiv.org/pdf/2405.09664", "yes", "clean?", "Used in the measles seasonality paper")
r("HHS hospital capacity and admissions", "Epidemics-health", "occupancy;waves;reporting-gaps", "D/W", "healthdata.gov", "US public domain", "https://healthdata.gov/", "no", "clean?", "")
r("Human Mortality Database short-term mortality fluctuations", "Epidemics-health", "weekly-deaths;heat-waves;epidemics", "W", "free registration", "CC BY 4.0", "https://www.mortality.org/", "no", "clean?", "Weekly deaths by country, good for excess-mortality shocks")
r("US births (CDC/GIFT)", "Epidemics-health", "weekday;holiday-dips;daily-seasonal", "D", "direct download", "US public domain", "https://www.cdc.gov/nchs/data_access/vitalstatsonline.htm", "no", "bench", "GIFT-Eval us_births")
r("Hospital admissions (Monash)", "Epidemics-health", "monthly;small-counts", "M", "Zenodo", "CC BY 4.0", "https://forecastingdata.org/", "no", "bench", "In Chronos and GIFT-Eval")
r("Spanish flu 1918 mortality (infectiousR)", "Epidemics-health", "single-wave;one-off-event", "D", "R package data", "check", "https://mirror.linux.duke.edu/pub/cran/web/packages/infectiousR/readme/README.html", "yes", "clean?", "Tiny, one-off wave. Useful as a probe, not a dataset")

# ---------------- Physiology ----------------
r("PhysioNet open-access collection (umbrella)", "Physiology", "quasi-periodic;artefacts;high-frequency", "1Hz-1kHz", "direct download", "open access (per database)", "https://physionet.org/about/database/", "yes", "clean?", "High-frequency; downsample or use derived features such as heart rate")
r("CGMacros (CGM + diet + activity)", "Physiology", "glucose;meal-spikes;daily-pattern;covariates", "5min/15min", "PhysioNet", "open access", "https://physionet.org/about/database/", "yes", "clean?", "Two CGMs plus food macros: covariate-rich")
r("BIG IDEAs glycemic variability and wearable data", "Physiology", "glucose;wearable-covariates;daily-pattern", "5min", "PhysioNet", "open access", "https://physionet.org/about/database/", "yes", "clean?", "")
r("Multi-night heart rate and accelerometry (Apple Watch) with sleep labels", "Physiology", "heart-rate;sleep-stage-cycles;circadian", "1s-1min", "PhysioNet", "open access", "https://physionet.org/about/database/", "yes", "clean?", "")
r("Wearable exam stress dataset", "Physiology", "EDA;heart-rate;event-driven-stress", "4-64Hz", "PhysioNet", "open access", "https://physionet.org/about/database/", "yes", "clean?", "")
r("CHB-MIT scalp EEG (seizures)", "Physiology", "EEG;regime-switch-to-seizure;high-frequency", "256Hz", "PhysioNet", "open access", "https://physionet.org/about/database/", "yes", "clean?", "Probe for regime changes; not for forecasting benchmarks")
r("Apnea-ECG database", "Physiology", "ECG;apnea-episodes;quasi-periodic", "100Hz", "PhysioNet", "open access", "https://physionet.org/about/database/", "yes", "clean?", "")
r("BIDMC congestive heart failure ECG", "Physiology", "ECG;long-recordings", "250Hz", "PhysioNet", "open access", "https://physionet.org/about/database/", "yes", "clean?", "")

# ---------------- Biology / microbial growth ----------------
r("muGrowthDB microbial growth curves", "Biology-growth", "sigmoid-growth;lag-phase;saturation;replicates", "min-hours", "web DB export", "check", "https://www.biorxiv.org/content/10.1101/2025.10.13.682118.full.pdf", "yes", "clean?", "Many studies in one DB; heterogeneous design")
r("E. coli growth curves in 1,029 defined media (Sci Data 2025)", "Biology-growth", "sigmoid-growth;lag-phase;diauxic;saturation", "min", "direct download", "check on article", "https://doaj.org/article/fc4fae3d7f0c4d16b64c043667e244f7", "yes", "clean?", "13,608 curves, 44 compounds")
r("Bacterial isolate chemical-response growth curves (figshare)", "Biology-growth", "sigmoid-growth;stress-response;OD600", "H", "direct download", "check on page", "https://figshare.com/articles/dataset/Bacterial_isolate_chemical_responses_data/25054913", "yes", "clean?", "Hourly OD600")
r("Harvard Forest bacterial OD600 plate reader (hf438)", "Biology-growth", "sigmoid-growth;OD600;temperature", "min", "direct download", "CC BY", "https://harvardforest.fas.harvard.edu/data/plots/hf438-01.pdf", "yes", "clean?", "")
r("Fungal growth dataset (411 mutants x 3 reps)", "Biology-growth", "piecewise-linear-growth;replicates", "10min", "R package data", "GPL-2", "https://r-packages.io/datasets/FungalGrowthDataset", "yes", "clean?", "289 time points per series")
r("E. coli / S. aureus replicate growth (Jones et al.)", "Biology-growth", "exponential-growth;stochastic-lag;replicates", "min", "per-paper data", "check", "https://arxiv.org/pdf/2304.11474", "yes", "clean?", "")
r("minimalgut synthetic-community time series (miaTime)", "Biology-growth", "microbial-abundance;compositional", "H", "R package data", "check", "https://bioc.r-universe.dev/miaTime/doc/manual.html", "yes", "clean?", "")
r("Human gut microbiome longitudinal (Ji et al. 2020)", "Biology-growth", "microbial-abundance;compositional;sparse", "D", "paper data", "check", "https://bioc.r-universe.dev/miaTime/doc/manual.html", "yes", "clean?", "Cited in miaTime docs")
r("Global Population Dynamics Database (GPDD)", "Biology-growth", "population-cycles;noisy-counts;chaotic", "Y/M", "download", "free", "https://www.imperial.ac.uk/cpb/databases/gpdd/", "no", "clean?", "Thousands of short annual population series")

# ---------------- Cloud, servers, web ----------------
r("Alibaba cluster trace (v2017/2018/2020)", "Cloud-servers", "CPU-utilisation;daily-cycle;batch-bursts;co-location", "10s-5min", "direct download", "CC BY 4.0 (check)", "https://github.com/alibaba/clusterdata", "yes", "pretrain", "alibaba_cluster_trace_2018 is in LOTSA to my knowledge")
r("Google cluster data (2011, 2019)", "Cloud-servers", "CPU/mem-utilisation;job-bursts;daily-cycle", "5min", "BigQuery / GCS", "CC BY 4.0", "https://github.com/google/cluster-data", "yes", "pretrain", "Borg 2011 is in LOTSA")
r("Azure Public Dataset (VM traces V1/V2)", "Cloud-servers", "VM-CPU;daily-cycle;lifetimes", "5min", "direct download", "CC BY 4.0", "https://github.com/Azure/AzurePublicDataset", "yes", "pretrain", "Azure VM 2017 is in LOTSA")
r("Data center traces for ML (Zenodo copy)", "Cloud-servers", "CPU-utilisation;aggregated", "5min", "Zenodo / pip", "check", "https://zenodo.org/records/14564935", "yes", "derived", "Alibaba 2018, Google 2019, Azure V2 resampled to 5 min")
r("Bitbrains grid workload traces", "Cloud-servers", "VM-CPU;bursty;enterprise-workloads", "5min", "direct download", "free", "https://clouds.cis.unimelb.edu.au/workloads.html", "yes", "bench", "GIFT-Eval bitbrains_fast_storage / rnd")
r("Chameleon cloud traces (ANL)", "Cloud-servers", "VM-lifecycle;scheduling;research-cloud", "event", "download", "open", "https://wordpress.cels.anl.gov/scienceclouds/cloud-traces/", "yes", "clean?", "Several traces 2018-2020")
r("Grid Workloads Archive / Failure Trace Archive", "Cloud-servers", "grid-load;availability;failures", "event/min", "download", "open", "https://clouds.cis.unimelb.edu.au/workloads.html", "yes", "clean?", "Classic; older")
r("PlanetLab CPU utilisation traces", "Cloud-servers", "VM-CPU;10-days", "5min", "download", "open", "https://clouds.cis.unimelb.edu.au/workloads.html", "yes", "clean?", "Short but real")
r("SAP cloud infrastructure dataset", "Cloud-servers", "VM-placement;utilisation", "varies", "check", "check", "https://arxiv.org/pdf/2510.23911", "yes", "clean?", "Described in a 2025 arXiv paper, confirm release")
r("Server Machine Dataset (SMD)", "Cloud-servers", "38-metrics-per-server;CPU/mem/disk;anomalies", "1min", "direct download", "MIT (OmniAnomaly repo)", "https://github.com/TheDatumOrg/TSB-UAD", "yes", "clean?", "28 machines, 5 weeks; also in Merlion loaders")
r("Numenta Anomaly Benchmark (NAB)", "Cloud-servers", "AWS-metrics;ad-clicks;traffic;tweets;anomalies", "5min", "GitHub", "AGPL (check)", "https://github.com/numenta/NAB", "yes", "clean?", "58 labelled series")
r("Yahoo Webscope S5", "Cloud-servers", "production-traffic;synthetic-anomalies", "H", "Webscope request", "restricted use terms", "https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70", "yes", "clean?", "A1 is real traffic (67 series), the rest synthetic")
r("KPI anomaly detection (AIOps challenge)", "Cloud-servers", "service-KPIs;page-views;response-time", "1min", "GitHub", "check", "https://github.com/NetManAIOps/KPI-Anomaly-Detection", "yes", "clean?", "")
r("IOPS competition dataset", "Cloud-servers", "web-service-KPIs", "1min", "competition site", "check", "http://iops.ai/competition_detail/?competition_id=5", "yes", "clean?", "")
r("TSB-UAD public v2 (29 datasets, 3,427 series)", "Cloud-servers", "mixed-domains;labelled-anomalies", "varies", "direct download", "check per dataset", "https://github.com/TheDatumOrg/TSB-UAD", "yes", "clean?", "A bundle: good for quickly adding many odd real series")
r("NASA SMAP and MSL telemetry", "Industrial-PHM", "spacecraft-telemetry;binary-commands;anomalies", "varies", "Kaggle / NASA", "check", "https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl", "yes", "clean?", "Contains many step/binary-like channels")
r("PSM (eBay pooled server metrics)", "Cloud-servers", "server-metrics;anomalies", "1min", "GitHub", "check", "https://ieee-dataport.org/documents/time-series-datasets", "yes", "clean?", "Listed with SMD, SMAP, MSL, SWaT")
r("Kaggle Wikipedia web traffic (Monash copy, 145k series)", "Web-attention", "weekly-cycle;event-spikes;intermittent;missing", "D", "Zenodo", "CC BY-SA 3.0", "https://zenodo.org/records/3898474", "yes", "pretrain", "In Chronos/GIFT; a similar slice is in your real pool")
r("Extended Wikipedia web traffic (to 2022-06)", "Web-attention", "weekly-cycle;event-spikes;intermittent;missing", "D", "Zenodo", "CC BY-SA 3.0", "https://zenodo.org/records/7370977", "yes", "mixed", "2015-2022; the same pages as the Kaggle set, so only 2020+ is new")
r("Wikimedia pageviews (fresh 2025-2026 pull)", "Web-attention", "weekly-cycle;news-spikes;holiday;intermittent", "D/H", "free API / dumps", "CC BY-SA", "https://dumps.wikimedia.org/other/pageview_complete/", "yes", "clean?", "Build it yourself from after your last training cutoff. Best fresh option in this list")
r("GitHub Archive event counts", "Web-attention", "weekday;release-spikes;growth", "H", "free (gharchive.org)", "CC BY 4.0", "https://www.gharchive.org/", "no", "clean?", "Per-repo star/commit/issue counts")
r("PyPI / npm download counts", "Web-attention", "weekday;release-spikes;CI-bots;growth", "D", "BigQuery / API", "open", "https://pypistats.org/api/", "no", "clean?", "Weekday cycle plus CI-driven spikes")
r("Cloudflare Radar traffic trends", "Web-attention", "daily-weekly;outages;country-level", "H", "free API", "CC BY-NC 4.0", "https://radar.cloudflare.com/", "no", "clean?", "Real outage drops")
r("GDELT event counts", "Web-attention", "news-volume;event-bursts;long-run-growth", "15min/D", "free", "open", "https://www.gdeltproject.org/", "no", "clean?", "")
r("Google Trends (older Zenodo extract and live)", "Web-attention", "annual-cycle;spikes;relative-index", "W/D", "pytrends (rate-limited)", "check", "https://zenodo.org/records/14539", "yes", "clean?", "Relative index; rate limits make bulk pulls painful")
r("Telecom Italia Big Data Challenge (Milan/Trentino)", "Telecom-network", "10-min-traffic;diurnal;weekly;events", "10min", "Harvard Dataverse / IEEE DataPort copy", "ODbL", "https://ieee-dataport.org/comment/6907", "yes", "clean?", "SMS, calls, internet by cell, Nov 2013 - Jan 2014")
r("OPNET simulated network traffic", "Telecom-network", "simulated-traffic", "10min", "IEEE DataPort", "check", "https://ieee-dataport.org/comment/4013", "yes", "derived", "Simulated, low priority")

# ---------------- Retail / sales ----------------
r("M5 Forecasting (Walmart)", "Retail-sales", "intermittent;hierarchical;promo;SNAP-days;weekly", "D", "Kaggle", "competition", "https://unidata.pro/blog/best-retail-datasets-for-ml/", "yes", "bench", "In fev and LOTSA")
r("Corporacion Favorita grocery sales", "Retail-sales", "promo;holiday;oil-price-covariate;intermittent", "D", "Kaggle", "competition", "https://unidata.pro/blog/best-retail-datasets-for-ml/", "yes", "bench", "fev favorita tasks")
r("Rossmann store sales", "Retail-sales", "promo;holiday;store-closures;weekly", "D", "Kaggle", "competition", "https://www.kaggle.com/c/rossmann-store-sales", "yes", "bench", "fev rossmann tasks")
r("Walmart store sales (Kaggle)", "Retail-sales", "weekly;holiday-weeks;markdowns", "W", "Kaggle", "competition", "https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting", "no", "bench", "fev walmart")
r("FreshRetailNet-50K (stockout-annotated)", "Retail-sales", "hourly-sales;censoring-stockouts;perishables;weather-covariates", "H", "Hugging Face", "CC BY 4.0", "https://huggingface.co/datasets/sky5268/FreshRetailNet-50K", "yes", "clean?", "50,000 store-product series, 90 days each. The best open censored-demand data I found")
r("Instacart online grocery orders", "Retail-sales", "orders;derive-daily-counts;weekly", "event", "Kaggle", "research use", "https://unidata.pro/blog/best-retail-datasets-for-ml/", "yes", "clean?", "Not a time series as published, aggregate to counts by aisle/product")
r("UCI Online Retail II", "Retail-sales", "transactions;derive-daily;Christmas-peak", "event", "UCI", "CC BY 4.0", "https://unidata.pro/blog/best-retail-datasets-for-ml/", "yes", "clean?", "Aggregate to daily sales per SKU")
r("Olist Brazilian e-commerce", "Retail-sales", "orders;derive-daily;Black-Friday", "event", "Kaggle", "CC BY-NC-SA 4.0", "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce", "no", "clean?", "")
r("Store Item Demand Forecasting (Kaggle)", "Retail-sales", "weekly;yearly;trend", "D", "Kaggle", "competition", "https://www.kaggle.com/c/demand-forecasting-kernels-only", "no", "clean?", "Looks synthetic-like (500 series), low priority")
r("Dunnhumby Complete Journey", "Retail-sales", "household-purchases;promo;derive-weekly", "event", "registration", "research", "https://www.dunnhumby.com/source-files/", "no", "clean?", "")
r("Dominick's grocery (Chronos)", "Retail-sales", "weekly;promo", "W", "chronos_datasets", "research", "https://huggingface.co/datasets/autogluon/chronos_datasets", "yes", "pool", "Already in the real allow-list")
r("Recruit restaurant visitors", "Retail-sales", "weekday;holiday;golden-week;reservations", "D", "Kaggle", "competition", "https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting", "no", "bench", "GIFT-Eval restaurant")
r("Car parts (intermittent demand)", "Retail-sales", "intermittent;small-counts", "M", "Monash", "CC BY 4.0", "https://forecastingdata.org/", "no", "bench", "GIFT-Eval car_parts")
r("ONS UK retail sales index", "Retail-sales", "trend;Christmas-peak;level-shift;covid-break", "M", "direct download", "OGL", "https://cy.ons.gov.uk/datasets/retail-sales-index/editions/time-series/versions", "yes", "clean?", "Many sectors, long history")
r("Statistics Canada retail trade", "Retail-sales", "trend;seasonal;covid-break", "M", "open.canada.ca", "OGL Canada", "https://open.canada.ca/data/dataset/9a72c101-0101-4fae-a404-3719b0e406df/resource/747200d1-6653-4aba-941f-23b6ae0e7d0f", "yes", "clean?", "Includes real-time vintages")
r("US Census monthly retail trade (via FRED)", "Retail-sales", "trend;seasonal;level-shift", "M", "FRED API", "public domain", "https://fred.stlouisfed.org/docs/api/fred/", "yes", "clean?", "")

# ---------------- Housing, macro, finance ----------------
r("Zillow ZHVI / ZORI home value and rent indices", "Housing-realestate", "trend;boom-bust;smooth;regional-hierarchy", "M", "direct download", "Zillow terms", "https://www.zillow.com/research/data/", "no", "clean?", "Thousands of regional series. FRED also hosts Z.1 owner-occupied real estate valued with Zillow data")
r("UK House Price Index (HM Land Registry)", "Housing-realestate", "trend;boom-bust;regional-hierarchy;property-types", "M", "direct download CSV", "OGL v3.0", "https://www.gov.uk/government/statistical-data-sets/uk-house-price-index-data-downloads-august-2026", "yes", "clean?", "Full file plus index, average price, sales volume, repossessions")
r("Case-Shiller indices (FRED)", "Housing-realestate", "trend;boom-bust;20-metros", "M", "FRED API", "S&P licence on FRED", "https://fred.stlouisfed.org/series/CSUSHPINSA", "no", "clean?", "")
r("Redfin / Realtor.com housing market data", "Housing-realestate", "listings;inventory;seasonal;rate-shock", "W/M", "direct download", "free with attribution", "https://www.redfin.com/news/data-center/", "no", "clean?", "Weekly listings and days-on-market")
r("FRED (800k+ macro and financial series)", "Macro-finance", "trend;level-shifts;regime-switch;recessions;mixed-frequencies", "D-Y", "free API key", "source-specific", "https://fred.stlouisfed.org/docs/api/fred/", "yes", "mixed", "FRED-MD/QD are in fev and GIFT, pick other categories")
r("ECB Statistical Data Warehouse / Eurostat / OECD", "Macro-finance", "trend;level-shifts;holiday-effects", "M/Q", "API", "open", "https://pandas-datareader.readthedocs.io/en/latest/_sources/remote_data.rst.txt", "yes", "clean?", "pandas-datareader exposes these and FRED, World Bank, OECD, Eurostat")
r("World Bank WDI indicators", "Macro-finance", "annual;trend;cross-country", "Y", "API", "CC BY 4.0", "https://data.worldbank.org/", "yes", "clean?", "Short series, mostly annual")
r("Fama-French factors", "Macro-finance", "returns;volatility-clustering;fat-tails", "D/M", "direct download", "free", "https://pandas-datareader.readthedocs.io/en/latest/_sources/remote_data.rst.txt", "yes", "clean?", "Near-unforecastable; checks that the model does not hallucinate structure")
r("Crypto prices (Binance public data)", "Crypto-markets", "fat-tails;volatility-clustering;regime-switch;24-7", "1min-D", "data.binance.vision", "free", "https://data.binance.vision/", "no", "likely_time", "TIME includes Crypto/D (BitInfoCharts); expect overlap on daily prices")
r("BitInfoCharts on-chain aggregates", "Crypto-markets", "on-chain-volume;hash-rate;growth", "D", "scrape", "check terms", "https://bitinfocharts.com/", "yes", "bench", "Source of TIME Crypto/D update")
r("Yahoo Finance equities and FX (yfinance)", "Crypto-markets", "returns;volatility-clustering;gaps-weekends", "D", "unofficial API", "Yahoo terms", "https://github.com/ranaroussi/yfinance", "no", "clean?", "Terms are restrictive for redistribution")
r("Dukascopy forex tick data", "Crypto-markets", "ticks;microstructure;session-effects", "tick", "download", "free", "https://www.dukascopy.com/swiss/english/marketwatch/historical/", "no", "clean?", "")
r("FI-2010 limit order book", "Crypto-markets", "LOB;microstructure", "event", "download", "research", "https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649", "no", "clean?", "")
r("Federal Reserve H.15 interest rates", "Macro-finance", "random-walk;policy-steps;regime", "D", "FRED", "public domain", "https://fred.stlouisfed.org/docs/api/fred/", "yes", "mixed", "Steps at policy dates")

# ---------------- Transport and mobility ----------------
r("Citi Bike system data (NYC)", "Transport-mobility", "trips;weather-driven;commute;seasonal;holiday", "event/H", "direct download", "NYC Bike Share licence", "https://ride.citibikenyc.com/system-data", "yes", "clean?", "Aggregate to station or system counts")
r("Divvy bikeshare (Chicago)", "Transport-mobility", "trips;weather-driven;winter-collapse", "event/H", "direct download", "Divvy data licence", "https://ride.divvybikes.com/how-it-works", "yes", "clean?", "")
r("Capital Bikeshare and other bikedata cities", "Transport-mobility", "trips;commute", "event/H", "direct download", "per system", "https://github.com/ropensci/bikedata", "yes", "clean?", "bikedata package covers NYC, Chicago, DC, Boston, LA, Philadelphia")
r("NYC TLC yellow/green/FHV trips", "Transport-mobility", "demand;zones;weekly;events;covid-break", "event/H", "direct download", "NYC Open Data", "https://data.cityofnewyork.us/Transportation/2016-Yellow-Taxi-Trip-Data/k67s-dv2t", "yes", "mixed", "taxi_30min and uber_tlc are in your excluded list; use zone-hourly counts for 2023+")
r("Chicago taxi, scooter and ride-hail trips", "Transport-mobility", "demand;pilot-program-start-stop", "event/H", "data.gov", "open", "https://catalog-beta.data.gov/?q=trips", "yes", "clean?", "E-scooter pilots show abrupt start/stop")
r("MTA subway/bus ridership (NYC)", "Transport-mobility", "weekday;covid-level-shift;recovery-trend", "D/H", "data.ny.gov", "open", "https://data.ny.gov/", "no", "clean?", "Pandemic level shift and slow recovery")
r("Transport for London open data", "Transport-mobility", "ridership;weekday;strikes", "D/15min", "download", "OGL", "https://tfl.gov.uk/info-for/open-data-users/", "no", "clean?", "Strike days are real one-offs")
r("Seattle Fremont Bridge bike counts", "Transport-mobility", "counts;weather;commute;holiday", "H", "data.seattle.gov", "open", "https://data.seattle.gov/", "no", "clean?", "Pedestrian counts (Melbourne) are in Monash/GIFT-Eval, these are not")
r("Caltrans PeMS freeway sensors", "Transport-mobility", "traffic-flow;speed;daily-cycle;incidents;sensor-faults", "5min", "free registration", "check", "https://pems.dot.ca.gov/", "no", "pretrain", "PEMS03/04/07/08 and LargeST are in GIFT-pretrain")
r("Highways England WebTRIS", "Transport-mobility", "traffic-flow;daily-cycle;roadworks;holidays", "15min", "free API", "OGL", "https://webtris.highwaysengland.co.uk/", "no", "clean?", "Different country from PeMS, similar behaviours")
r("Shenzhen taxi / Loop Seattle / METR-LA", "Transport-mobility", "traffic-speed;daily-cycle", "5-15min", "GitHub", "check", "https://github.com/liyaguang/DCRNN", "no", "bench", "In GIFT-Eval or LOTSA")
r("BTS on-time performance and T-100 passengers", "Aviation-tourism", "monthly;holiday;strike-weather-shocks", "D/M", "direct download", "US public domain", "https://www.transtats.bts.gov/", "no", "clean?", "Delays are heavy-tailed")
r("TSA checkpoint daily passenger volumes", "Aviation-tourism", "weekly;holiday-rush;covid-collapse-recovery", "D", "direct download", "US public domain", "https://www.tsa.gov/travel/passenger-volumes", "no", "clean?", "Thanksgiving/July 4 peaks, since 2019")
r("OpenSky Network flight counts", "Aviation-tourism", "daily;weekly;disruptions", "H", "free for research", "CC BY-SA", "https://opensky-network.org/", "no", "clean?", "")

# ---------------- Tourism, events, entertainment ----------------
r("Singapore Tourism Board visitor arrivals + hotel occupancy", "Aviation-tourism", "monthly;holiday;covid-collapse;market-mix", "M", "SingStat download", "Singapore Open Data Licence", "https://www.singstat.gov.sg/find-data/explore-data-themes/industry/tourism/latest-news-data", "yes", "clean?", "By origin market, back to 1978")
r("Eurostat tourist accommodation occupancy (tour_occ)", "Aviation-tourism", "monthly;strong-seasonality;country-hierarchy", "M", "Eurostat API", "CC BY 4.0", "https://ec.europa.eu/eurostat/cache/metadata/en/tour_occ_esmssp_ie.htm", "yes", "clean?", "Many countries and accommodation types")
r("INSEE tourism occupancy (France) via DBnomics", "Aviation-tourism", "monthly;hotel-categories;regional", "M", "DBnomics API", "Etalab", "https://db.nomics.world/INSEE/TOURISME-FRANCE-METHODE-REDRESSEMENT-2019", "yes", "clean?", "3,650 series")
r("Hawaii UHERO tourism data warehouse", "Aviation-tourism", "monthly;hotel-occupancy;RevPAR;island-level", "M", "download", "check", "https://data.uhero.hawaii.edu/dvw", "yes", "clean?", "")
r("Sri Lanka SLTDA monthly tourist arrivals", "Aviation-tourism", "monthly;shocks-2019-attacks;covid;crisis", "M", "PDF reports", "public", "https://sltda.gov.lk/en/monthly-tourist-arrivals-reports-2024", "yes", "clean?", "PDFs, needs extraction. Contains real shocks (2019, 2022)")
r("Hong Kong visitor arrivals", "Aviation-tourism", "monthly;protest-and-covid-breaks", "M", "data.gov.hk", "open", "https://data.gov.hk/en-data/dataset/hk-censtatd-tablechart-650-80005", "yes", "clean?", "Annual table in the page I saw, look for monthly")
r("Tourism (Monash: monthly/quarterly/yearly)", "Aviation-tourism", "monthly;quarterly;short", "M/Q/Y", "Zenodo", "CC BY 4.0", "https://forecastingdata.org/", "no", "bench", "Competition datasets, in Chronos/GIFT")
r("Broadway weekly grosses and attendance", "Events-entertainment", "weekly;holiday-weeks;show-lifecycle;previews;closings", "W", "scrape (BroadwayWorld, Playbill; data by Broadway League)", "check terms, scraping risk", "https://www.broadwayworld.com/grosses.php", "yes", "clean?", "Show lifecycle (previews, opening, run, closing) is a distinctive behaviour")
r("Box Office Mojo daily domestic box office", "Events-entertainment", "release-decay;weekend-spikes;holidays", "D", "scrape", "restrictive terms", "https://www.boxofficemojo.com/article/ed3547792388", "yes", "clean?", "Per-film decay curves; terms are restrictive")
r("Movie screening and box office time series 2019-2024 (IEEE DataPort)", "Events-entertainment", "release-decay;occupancy;ticket-price", "D", "IEEE DataPort", "check on page", "https://ieee-dataport.org/documents/multi-dimensional-movie-screening-and-box-office-performance-time-series-dataset", "yes", "clean?", "")
r("Theme park wait times (queue-times.com archives)", "Events-entertainment", "intraday-profile;holiday-rush;ride-breakdowns", "5min", "API/archive (check terms)", "check", "https://queue-times.com/", "no", "clean?", "Matches the holiday-rush use case")
r("Python holidays package calendars", "Events-entertainment", "calendar-covariates;moving-holidays;country-specific", "D", "pip package", "MIT", "https://github.com/vacanza/holidays", "yes", "n/a", "Not a series: use it for known-future covariates in pools above")

# ---------------- Industrial / PHM ----------------
r("NASA C-MAPSS turbofan degradation", "Industrial-PHM", "degradation-trend;run-to-failure;multi-operating-conditions", "cycle", "NASA data portal", "NASA open", "https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20150007677.pdf", "yes", "clean?", "Simulated, 4 subsets, 21 sensors")
r("NASA IMS bearing run-to-failure", "Industrial-PHM", "vibration;degradation;late-failure-onset", "10min snapshots", "NASA repository", "NASA open", "https://service.tib.eu/ldmservice/dataset/nasa-ims-bearing-dataset", "yes", "clean?", "")
r("CWRU bearing fault data", "Industrial-PHM", "vibration;fault-classes", "12/48kHz", "direct download", "open", "https://ibm.github.io/AssetOpsBench/docs/external-industrial-datasets.html", "yes", "clean?", "High frequency; probe for periodic/quasi-periodic")
r("Paderborn bearing dataset", "Industrial-PHM", "vibration;varied-load-speed", "64kHz", "download", "open", "https://ibm.github.io/AssetOpsBench/docs/external-industrial-datasets.html", "yes", "clean?", "")
r("CIRA centrifugal pump IoT dataset", "Industrial-PHM", "pump-vibration;temperature;pressure;daily-ops", "min", "direct download (8 CSVs)", "CC BY 4.0 (MDPI Data)", "https://api.crossref.org/works/10.3390%2FDATA10060091", "yes", "clean?", "Three pumps, eight operational days")
r("NASA PCoE battery degradation", "Industrial-PHM", "capacity-fade;cycling;knee-point", "cycle", "NASA repository", "NASA open", "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/", "no", "clean?", "Monotone fade with a knee, a distinct behaviour")
r("Tennessee Eastman process", "Industrial-PHM", "process-control;faults;multivariate", "3min", "Harvard Dataverse", "open", "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1", "no", "clean?", "Simulated chemical process")
r("SECOM semiconductor manufacturing", "Industrial-PHM", "process-sensors;yield;sparse-missing", "event", "UCI", "CC BY 4.0", "https://archive.ics.uci.edu/dataset/179/secom", "no", "clean?", "")

# ---------------- Government / agriculture ----------------
r("India open government data (data.gov.in)", "Government-stats", "annual-monthly;crop-prices;consumption", "M/Y", "portal download", "NDSAP", "https://data.gov.in/", "no", "clean?", "Mandi (agri market) prices are relevant to your location; quality varies")
r("FAOSTAT food and agriculture", "Agriculture-commodities", "annual;production;trend", "Y", "bulk download", "CC BY-NC-SA 3.0 IGO", "https://www.fao.org/faostat/", "no", "clean?", "Annual, short series")
r("USDA NASS crop progress and prices", "Agriculture-commodities", "weekly-crop-stage;annual-cycle;price", "W/M", "API (free key)", "US public domain", "https://quickstats.nass.usda.gov/", "no", "clean?", "")
r("pdfetch-accessible BLS and EIA series", "Government-stats", "monthly-macro;employment;energy", "M", "R package access", "public domain", "https://github.com/abielr/pdfetch", "yes", "clean?", "Pointer to tooling for BLS, EIA, World Bank, Eurostat")

# ---------------- write ----------------
cols = ["id","name","domain","behaviours","freq","access","licence","url","url_verified","overlap","dev_use","notes"]
def dev_use(o):
    return {"bench": "diagnosis-only (do not select on it)",
            "likely_time": "diagnosis-only (TIME family)",
            "pretrain": "ok for our twins; reference models saw it",
            "pretrain?": "ok for our twins; reference models likely saw it",
            "pretrain?": "ok for our twins; reference models likely saw it",
            "mixed": "split by sub-series; audit first",
            "derived": "ok if parent is ok",
            "pool": "already in real post-training pool, not dev",
            "clean?": "yes, after audit",
            "n/a": "covariate source, not a series"}.get(o, "check")
rows = []
for i, d in enumerate(R, 1):
    d = dict(d); d["id"] = "D%03d" % i; d["dev_use"] = dev_use(d["overlap"]); rows.append(d)
import os
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_catalog_v1.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

print("rows:", len(rows))
print("by domain:"); [print("  %-24s %d" % k_v) for k_v in collections.Counter(d["domain"] for d in rows).most_common()]
print("by overlap:", dict(collections.Counter(d["overlap"] for d in rows)))
print("url_verified:", dict(collections.Counter(d["url_verified"] for d in rows)))
names = [d["name"] for d in rows]
print("duplicate names:", [n for n, c in collections.Counter(names).items() if c > 1])
urls = collections.Counter(d["url"] for d in rows)
print("urls used by >3 rows:", [(u, c) for u, c in urls.items() if c > 3])
