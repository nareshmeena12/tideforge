"""Expand the dataset catalog, remove benchmark datasets, add size estimates.

Inputs : dataset_catalog_v1.csv (v1, 213 rows, from build_catalog.py) in the same folder.
Outputs: dataset_catalog_v2.csv        kept datasets with sizes
         excluded_benchmark.csv        catalog rows removed because they are (or very likely are) in TIME / fev-bench / GIFT-Eval
         benchmark_datasets.csv        the benchmark dataset names we screen against
Run    : python3 -I expand_catalog.py

Size columns are ORDER-OF-MAGNITUDE estimates (basis='est') unless basis='doc'.
series_avail : number of univariate series you could build from the full public release
points_avail : total observations at native frequency (event data: after hourly/daily aggregation as noted)
blank series_avail => unbounded source (API/grid/scrape) -> only the pull columns are meaningful
series_pull/points_pull: a proposed dev slice = min(series,500) series x min(avg_len,10000) points
"""
import csv, collections, os, math

HERE = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(HERE, "dataset_catalog_v1.csv")

# ------------------------------------------------------------------ benchmark lists
TIME = ["Australia_Solar","Auto_Production_SF","CPHL","Coastal_T_S","Commodity_Import","Commodity_Production","Crypto",
        "ECDC_COVID","EWELD_Load","Finland_Traffic","Global_Influenza","Global_Price","Housing_Inventory","JOLTS","Job_Claims",
        "MetroPT-3","NE_China_Wind","Oil_Price","Online_Retail_2_UCI","OpenElectricity_NEM","Port_Activity","SG_Carpark","SG_PM25",
        "SG_Weather","Smart_Manufacturing","Supply_Chain_Customer","Supply_Chain_Location","US_Labor","US_Term_Structure",
        "Uncertainty_1M","Vehicle_Sales","Vehicle_Supply","WUI_Global","Water_Quality_Darwin","azure2019_D","azure2019_I",
        "azure2019_U","current_velocity","epf_electricity_price"]
FEV = ['ETT', 'LOOP_SEATTLE', 'M_DENSE', 'SZ_TAXI', 'australian_tourism', 'bizitobs_l2c', 'boomlet_1062', 'boomlet_1209', 'boomlet_1225', 'boomlet_1230', 'boomlet_1282', 'boomlet_1487', 'boomlet_1631', 'boomlet_1676', 'boomlet_1855', 'boomlet_1975', 'boomlet_2187', 'boomlet_285', 'boomlet_619', 'boomlet_772', 'boomlet_963', 'ecdc_ili', 'entsoe', 'epf_be', 'epf_de', 'epf_fr', 'epf_np', 'epf_pjm', 'ercot', 'favorita_stores', 'favorita_transactions', 'fred_md_2025', 'fred_qd_2025', 'gvar', 'hermes', 'hierarchical_sales', 'hospital', 'hospital_admissions', 'jena_weather', 'kdd_cup_2022', 'm5', 'proenfo_gfc12', 'proenfo_gfc14', 'proenfo_gfc17', 'redset', 'restaurant', 'rohlik_orders', 'rohlik_sales', 'rossmann', 'solar', 'solar_with_weather', 'uci_air_quality', 'uk_covid_nation', 'uk_covid_utla', 'us_consumption', 'walmart', 'world_co2_emissions', 'world_life_expectancy', 'world_tourism']  # from autogluon/fev benchmarks/fev_bench/tasks.yaml (100 tasks)
GIFT = ['bitbrains_fast_storage', 'bitbrains_rnd', 'bizitobs_application', 'bizitobs_service', 'bizitobs_l2c', 'car_parts', 'covid_deaths', 'electricity', 'ett1', 'ett2', 'hierarchical_sales', 'hospital', 'jena_weather', 'kdd_cup_2018', 'LOOP_SEATTLE', 'M_DENSE', 'SZ_TAXI', 'm4_yearly', 'm4_quarterly', 'm4_monthly', 'm4_weekly', 'm4_daily', 'm4_hourly', 'restaurant', 'saugeenday', 'solar', 'temperature_rain', 'us_births']  # from gift-eval notebooks short_datasets/med_long_datasets

TIME_SRC = {"Australia_Solar":"PVOutput, 3 Australian stations","Oil_Price":"FRED daily crude and refined products","Port_Activity":"IMF PortWatch",
 "Global_Price":"FRED quarterly commodity prices","Commodity_Production":"FRED historical US production","Commodity_Import":"FRED historical US imports",
 "Vehicle_Sales":"FRED","Vehicle_Supply":"FRED","Auto_Production_SF":"FRED seasonal factors","Housing_Inventory":"FRED (Realtor.com)",
 "Job_Claims":"FRED","JOLTS":"FRED/BLS","US_Labor":"FRED","US_Term_Structure":"FRED Kim-Wright","Uncertainty_1M":"FRED JLN/LMN",
 "WUI_Global":"FRED World Uncertainty Index","Crypto":"FRED BTC/ETH/LTC/BCH","ECDC_COVID":"ECDC hospital occupancy","Global_Influenza":"WHO FluNet, 15 countries",
 "epf_electricity_price":"Lago et al. 2021 EPF","OpenElectricity_NEM":"OpenElectricity, Jan-May 2025","EWELD_Load":"Liu et al. 2023, 10 users",
 "SG_Carpark":"data.gov.sg","SG_Weather":"data.gov.sg, 6 stations","SG_PM25":"data.gov.sg, 5 regions","Finland_Traffic":"Digitraffic",
 "Water_Quality_Darwin":"IMOS","current_velocity":"IMOS","CPHL":"IMOS","Coastal_T_S":"IMOS","NE_China_Wind":"Zhang et al. 2025 (reanalysis-style grid)",
 "Supply_Chain_Customer":"competition (simulated)","Supply_Chain_Location":"competition (simulated)","Smart_Manufacturing":"competition (simulated)",
 "MetroPT-3":"UCI MetroPT-3","Online_Retail_2_UCI":"UCI Online Retail II","azure2019_D":"Azure 2019 VM CPU (Woo et al. 2023)",
 "azure2019_I":"Azure 2019 VM CPU","azure2019_U":"Azure 2019 VM CPU"}

# ------------------------------------------------------------------ v1 rows
rows = list(csv.DictReader(open(V1, encoding="utf-8")))
by_id = {x["id"]: x for x in rows}

# Newly found overlaps (from this turn's benchmark lists) -> drop
NEW_DROPS = {
    "D007": "TIME OpenElectricity_NEM uses the same NEM regional data",
    "D152": "TIME Online_Retail_2_UCI",
    "D175": "TIME US_Term_Structure (FRED Kim-Wright forward rates/term premia, derived from the same Treasury curve)",
    "D037": "TIME Global_Price (FRED quarterly energy/metal/agri commodity prices; same commodities and period)",
}
RESTORE = {
    "D082": ("pretrain?", "hosted in autogluon/fev_datasets but NOT a fev-bench task (checked tasks.yaml)"),
    "D088": ("pretrain?", "hosted in autogluon/fev_datasets but NOT a fev-bench task (checked tasks.yaml)"),
    "D196": ("pretrain", "Monash tourism is in Chronos zero-shot benchmark and LOTSA, not in TIME/fev-bench/GIFT test (fev uses australian_tourism)"),
    "D170": ("mixed", "EXCLUDE: BTC, ETH, LTC, BCH prices (TIME Crypto = FRED daily prices of these 4); other pairs fine"),
    "D171": ("mixed", "EXCLUDE: price series of BTC, ETH, LTC, BCH (TIME Crypto); on-chain activity metrics fine"),
}
# Keep but exclude the overlapping sub-series
NEW_MIXED = {
    "D001": "exclude national load columns used by fev ENTSO-e Load (OPSD)",
    "D002": "exclude load of fev ENTSO-e countries and day-ahead prices of BE/DE/FR/Nord Pool/PJM (fev EPF-*, TIME epf_electricity_price)",
    "D010": "exclude PJM LMP/prices (fev EPF-PJM); zonal load is fine",
    "D031": "keep natural gas (Henry Hub) only; crude AND refined-product prices are TIME Oil_Price (via FRED)",
    "D039": "exclude Alabama 2006 plants (GIFT-Eval solar)",
    "D092": "exclude cases/deaths (GIFT covid_deaths) and European hospital/ICU occupancy (TIME ECDC_COVID)",
    "D121": "use the 2017 V1 VM trace only; exclude all 2019 traces (TIME azure2019_D/I/U)",
    "D165": "exclude inventory/listing-count series (TIME Housing_Inventory)",
    "D166": "exclude FRED-MD/QD constituents (fev) and TIME econ series (claims, JOLTS, labor, vehicles, prices, EPU/WUI)",
    "D168": "exclude CO2, life expectancy and tourism indicators (fev World *)",
    "D213": "exclude JOLTS, claims, CES/CPS headline series (TIME US_Labor/JOLTS/Job_Claims)",
    "D053": "exclude Singapore stations (TIME SG_Weather, data.gov.sg)",
    "D054": "exclude Singapore stations (TIME SG_Weather)",
    "D055": "exclude Singapore and Northeast China grid cells (TIME SG_Weather, NE_China_Wind)",
    "D056": "exclude Singapore and Northeast China grid cells (TIME SG_Weather, NE_China_Wind)",
    "D064": "exclude Northeast China region (TIME NE_China_Wind)",
    "D077": "exclude Singapore PM2.5 (TIME SG_PM25)",
    "D079": "exclude Singapore (TIME SG_PM25)",
    "D172": "exclude BTC/ETH/LTC/BCH tickers (TIME Crypto)",
    "D173": "exclude crypto CFDs (TIME Crypto)",
}
# Rows that duplicate or are derived from another row -> keep in catalog, not counted in totals
NOT_COUNTED = {"D022": "same households as D021", "D024": "labels on BDG2 (D023)", "D122": "copy of D119-D121",
               "D132": "same KPIs as D131", "D133": "contains D128-D131 and NAB/Yahoo", "D145": "simulated", "D201": "covariate source"}

# (series_avail, points_avail) for v1 rows; None series => unbounded; avg_len used then
S = {
"D001":(300,1.5e7),"D002":(2000,1.7e8),"D003":(1500,1.3e8),"D004":(50,1.3e7),"D005":(200,4.6e7),"D006":(3,1e6),
"D009":(500,5e8),"D010":(100,1.75e7),"D011":(10,1.75e6),"D012":(11,2.2e7),"D013":(10,1.75e6),"D014":(40,7e6),
"D015":(15,5.8e6),"D016":(10,3.9e6),"D017":(20,7e6),"D020":(500,1.3e7),"D021":(5560,1.7e8),"D022":(5567,1.7e8),
"D023":(3053,5.3e7),"D024":(1500,2.6e7),"D025":(26000,1.8e9),"D026":(6435,1.6e8),"D027":(75,1.3125e7),"D028":(443,6.3792e5),
"D029":(900,4.7e7),"D030":(2000,8.8e7),"D031":(15,1.35e5),"D032":(1000,1.5e6),"D033":(42,1.3e5),"D034":(60000,7e6),
"D035":(300000,6e7),"D036":(480,2.3e5),"D037":(70,5.5e4),"D038":(1000,1.75e8),"D039":(6000,6.3e8),"D040":(90,1.4e8),
"D041":(40,4e7),"D042":(10,4.4e5),"D043":(400,1e8),"D044":(126000,9e10),"D045":(30,1e6),"D046":(100,1e7),"D047":(1000,3.5e7),
"D048":(100,3.5e6),"D049":(500,1.75e7),"D050":(30,4e4),"D051":(500,8.5e6),"D052":(3000,1e8),"D053":(300000,3e9),
"D054":(200000,1e10),"D055":(None,7.4e5),"D056":(None,1.6e4),"D057":(8000,5e8),"D058":(600,2e8),"D059":(1500,9e5),
"D060":(5,5e4),"D061":(10,1.8e4),"D062":(4500,6.6e7),"D063":(4964,2.2e8),"D064":(None,3.6e5),"D066":(3,8e4),"D067":(50,2.7e7),
"D068":(1000,1.8e7),"D069":(200,1.8e6),"D070":(200000,1.3e10),"D071":(4000,5e7),"D072":(4000,7e7),"D073":(41000,6e8),
"D074":(13000,7e9),"D075":(300,5.5e6),"D076":(51,4.8e7),"D077":(90000,2.3e9),"D078":(24000,4e9),"D079":(6000,1.8e7),
"D083":(6600,2.9e8),"D084":(20000,1.7e9),"D085":(7000,6e6),"D086":(5000,3e6),"D087":(1000,5e5),"D090":(9000,1.8e6),
"D091":(4500,2.25e6),"D092":(2400,3.4e6),"D095":(84000,8.4e7),"D096":(194,3.3e4),"D097":(50000,1e7),"D098":(684,7e5),
"D101":(50,1e4),"D102":(100000,1e11),"D103":(200,2.9e6),"D104":(100,1e7),"D105":(100,5e6),"D106":(180,5e7),"D107":(550,2e10),
"D108":(70,2e8),"D109":(30,5.4e8),"D110":(10000,1e6),"D111":(3000,3e5),"D112":(10000,1e6),"D113":(2000,4e5),"D114":(1233,3.7e5),
"D115":(500,5e4),"D116":(50,1.5e4),"D117":(400,8e4),"D118":(5000,1.25e5),"D119":(300000,5e9),"D120":(1000000,1e9),
"D121":(2000000,1.2e9),"D122":(10000,1e8),"D124":(1000,1e6),"D125":(30000,3e7),"D126":(11700,3.4e6),"D127":(1000,1e7),
"D128":(1064,5.4e7),"D129":(58,3.6e5),"D130":(367,5.7e5),"D131":(58,5.9e6),"D132":(58,5.9e6),"D133":(3427,7e7),
"D134":(2050,1.6e7),"D135":(26,5.7e6),"D136":(145063,1.16e8),"D137":(145063,3.6e8),"D138":(None,800),"D139":(None,1e5),
"D140":(3600000,6.5e9),"D141":(5000,1.75e8),"D142":(None,3.5e5),"D143":(None,1000),"D144":(83000,7.4e8),"D145":(100,1e6),
"D150":(50000,1.08e8),"D151":(50000,5e6),"D153":(2000,1.4e6),"D154":(500,9.1e5),"D155":(50000,5e6),"D156":(100000,3e7),
"D159":(100,4.5e4),"D160":(500,1.8e5),"D161":(100,4e4),"D162":(100000,3e7),"D163":(20000,7e6),"D164":(60,6e4),"D165":(500000,7.5e7),
"D166":(800000,2.5e8),"D167":(None,200),"D168":(325000,1e7),"D169":(1000,2.5e7),"D172":(100000,5e8),"D173":(1000,1e10),
"D174":(5,2e6),"D176":(2000,8e7),"D177":(1500,6e7),"D178":(6000,2.4e8),"D179":(800,1e8),"D180":(500,4e7),"D181":(500,1.3e7),
"D182":(1000,2e7),"D183":(3,3.5e5),"D184":(18000,1e10),"D185":(20000,1e9),"D187":(50000,2e7),"D188":(1,2.4e3),"D189":(5000,2.6e8),
"D190":(200,1e5),"D191":(20000,4e6),"D192":(1000,2e5),"D193":(30000,1e7),"D194":(50,1.5e4),"D195":(50,2e4),"D197":(3000,3e5),
"D198":(20000,1.2e6),"D199":(1000,1e5),"D200":(5000,1e9),"D201":(0,0),"D202":(30000,6e6),"D203":(12,3e4),"D204":(160,2e7),
"D205":(2560,6.5e8),"D206":(50,5e6),"D207":(34,2.5e6),"D208":(1000000,6e8),"D209":(590,9e5),"D210":(None,30),"D211":(1000000,4e7),
"D212":(100000,3e7),"D213":(1000000,2e8),"D082":(132,4.6e6),"D088":(300,4e5),"D196":(1311,1.3e5),"D170":(2000,3e9),"D171":(750,2.7e6),
}

# ------------------------------------------------------------------ new rows
# (name, domain, behaviours, freq, access, licence, url, url_verified, overlap, series, points, notes)
N = [
# Energy: load, grid, markets
("SMARD.de Germany market data (Bundesnetzagentur)","Energy-load","multi-seasonal;holiday;generation-mix;negative-prices","15min","direct download","CC BY 4.0","https://www.smard.de/en/downloadcenter/download-market-data/","no","mixed",60,2.3e7,"exclude DE day-ahead prices (fev EPF-DE); load and generation fine"),
("Eskom data portal (South Africa)","Energy-load","load-shedding-regimes;step-down;multi-seasonal","H","direct download","Eskom terms","https://www.eskom.co.za/dataportal/","yes","clean?",30,1.8e6,"demand, generation by type, load-shedding stages; rationing regimes are rare elsewhere"),
("NZ Electricity Authority EMI wholesale datasets","Energy-load","multi-seasonal;hydro-driven-prices;nodal","30min","direct download","CC BY 4.0","https://www.ea.govt.nz/data-and-insights/datasets/wholesale","yes","clean?",400,1.7e8,"node-level demand and prices since ~2000"),
("ONS Brazil open data (load and generation)","Energy-load","multi-seasonal;hydro;southern-hemisphere","H","direct download","CC BY 4.0","https://dados.ons.org.br/","no","clean?",500,5e7,"load by subsystem, generation by plant"),
("Japan area demand (OCCTO / TEPCO and 9 utilities)","Energy-load","multi-seasonal;holiday;Golden-Week;Obon","H","direct download","utility terms","https://www.occto.or.jp/","no","clean?",50,4.5e6,"10 areas since 2016"),
("KPX Korea hourly power demand","Energy-load","multi-seasonal;Chuseok;Seollal","H","data.go.kr download","KOGL","https://www.data.go.kr/","no","clean?",5,1e6,""),
("EirGrid smart grid dashboard (Ireland)","Energy-load","wind-dominated;multi-seasonal;interconnector","15min","API","EirGrid terms","https://www.smartgriddashboard.com/","no","clean?",20,7e6,""),
("AESO Alberta pool price and load","Energy-load","price-spikes;cold-snaps;multi-seasonal","H","direct download","AESO terms","https://www.aeso.ca/market/market-and-system-reporting/data-requests/","no","clean?",10,2e6,""),
("MISO and SPP market reports","Energy-load","multi-seasonal;winter-storm-Uri;wind","H/5min","direct download","ISO terms","https://www.misoenergy.org/markets-and-operations/real-time--market-data/market-reports/","no","clean?",50,5e6,"zonal load"),
("EIA-923 monthly generation by plant","Energy-gen","plant-retirements;seasonal;structural-change","M","direct download","US public domain","https://www.eia.gov/electricity/data/eia923/","no","clean?",20000,5.8e6,"~12k plants by fuel since 2001"),
("NRC power reactor status (daily percent power)","Energy-gen","plateau-at-100;refuelling-outage-steps;ramp-up;coast-down","D","direct download","US public domain","https://www.nrc.gov/reading-rm/doc-collections/event-status/reactor-status/","yes","clean?",104,1.1e6,"saturation and outage behaviours, since 1999"),
("UK National Gas data portal (NTS demand)","Oil-gas-fuels","winter-peaks;weather-driven;LDZ-hierarchy","D","direct download","Open Gas licence","https://data.nationalgas.com/","no","clean?",100,4e5,""),
("ENTSOG transparency platform (gas flows)","Oil-gas-fuels","flow-reversals;2022-shock;seasonal","D/H","API","ENTSOG terms","https://transparency.entsog.eu/","no","clean?",2000,7e6,"pipeline flows at ~1,000 points"),
("GIE AGSI+ gas storage","Oil-gas-fuels","fill-withdraw-cycle;policy-targets;saturation","D","API key (free)","GIE terms","https://agsi.gie.eu/","no","clean?",300,1.5e6,"bounded 0-100% storage, strong annual cycle"),
("Tankerkonig German fuel prices","Oil-gas-fuels","intraday-sawtooth;Edgeworth-cycles;step-changes;policy-shock-2022","event->H","registration (free, Azure DevOps repo)","CC BY 4.0","https://www.r-bloggers.com/the-german-fuel-prices-data-set/","yes","clean?",48000,4.8e9,"~16k stations x 3 fuels since 2014; also Mendeley 2022 daily version"),
("France prix des carburants open data","Oil-gas-fuels","step-changes;regional-co-movement","event->D","direct download","Licence Ouverte","https://www.prix-carburants.gouv.fr/rubrique/opendata/","no","clean?",60000,3.6e8,"~10k stations x 6 fuels since 2007"),
("Italy MIMIT daily fuel prices","Oil-gas-fuels","step-changes;excise-policy","D","direct download","IODL 2.0","https://www.mimit.gov.it/it/open-data/elenco-dataset/carburanti-prezzi-praticati-e-anagrafica-degli-impianti","no","clean?",44000,1.2e8,""),
("Baker Hughes rig count","Oil-gas-fuels","price-lagged;trend-breaks;weekly","W","direct download","Baker Hughes terms","https://rigcount.bakerhughes.com/","no","clean?",100,2e5,"by basin and state since 1987"),
("NSTA UK North Sea field production","Oil-gas-fuels","decline-curves;shut-ins","M","direct download","OGL","https://www.nstauthority.co.uk/data-centre/","no","clean?",600,3e5,""),
# Solar / wind
("Open Climate Fix UK PV (Sheffield Solar systems)","Energy-gen","diurnal-envelope;clouds;clipping;snow-zero","30min/5min","gated HF (agree to terms)","check dataset card","https://huggingface.co/datasets/openclimatefix/uk_pv","yes","clean?",30000,2e9,"HF discussion notes only ~1,300 systems at 5-min"),
("Sheffield Solar PV_Live (GB national and GSP)","Energy-gen","diurnal;seasonal;cloud-noise","30min","API","CC BY 4.0","https://www.solar.sheffield.ac.uk/pvlive/","no","clean?",330,7e7,""),
("EDP wind farm SCADA open data","Energy-gen","power-curve;curtailment;faults","10min","registration","EDP terms","https://opendata.edp.com/","no","clean?",100,1e7,""),
("Engie La Haute Borne wind SCADA","Energy-gen","power-curve;icing;downtime","10min","direct download","Open Licence","https://opendata-renewables.engie.com/","no","clean?",120,3e7,"4 turbines 2013-2017"),
("BSRN baseline surface radiation network","Weather-climate","clear-sky-envelope;clouds;1min","1min","registration (free)","BSRN terms","https://bsrn.awi.de/","no","clean?",400,3e9,""),
("NOAA SURFRAD radiation network","Weather-climate","clear-sky-envelope;clouds","1min","direct download","US public domain","https://gml.noaa.gov/grad/surfrad/","no","clean?",70,1e9,""),
("PVOutput.org systems","Energy-gen","diurnal;household-PV;clipping","5-10min","API (rate-limited, donation tiers)","PVOutput terms","https://pvoutput.org/","no","mixed",100000,3e10,"EXCLUDE: Australian systems (TIME Australia_Solar = 3 PVOutput stations in Australia); API limits make full pull impractical"),
# Buildings / NILM / IoT / water
("REFIT electrical load (20 UK houses, appliances)","Buildings-heat","on-off-duty-cycles;appliance-signatures","8s","direct download","CC BY 4.0","https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned","no","clean?",200,1e9,""),
("UK-DALE (appliance-level)","Buildings-heat","on-off-duty-cycles;standby","6s","direct download","CC BY 4.0","https://jack-kelly.com/data/","no","clean?",150,7.5e8,""),
("REDD (MIT)","Buildings-heat","on-off-duty-cycles","3s","request","research use","http://redd.csail.mit.edu/","no","clean?",120,7e7,""),
("AMPds2 (one house, electricity water gas)","Buildings-heat","duty-cycles;multi-utility","1min","direct download","CC BY 4.0","https://dataverse.harvard.edu/dataverse/rmakonin","no","clean?",24,2.4e7,""),
("ENERTALK (22 Korean houses)","Buildings-heat","appliance-signatures","15Hz->1min","direct download","CC BY 4.0","https://www.nature.com/articles/s41597-019-0212-5","no","clean?",150,2.3e7,""),
("IDEAL household energy (255 UK homes)","Buildings-heat","room-temperature;gas-boiler-cycles;occupancy","1s-12s->1min","direct download","CC BY 4.0","https://datashare.ed.ac.uk/handle/10283/3647","no","pretrain",7650,4e9,"in LOTSA"),
("CU-BEMS Bangkok office building","Buildings-heat","office-hours;HVAC;weekday-weekend","1min","direct download","CC BY 4.0","https://www.nature.com/articles/s41597-020-00582-3","no","clean?",300,2.4e8,""),
("Intel Berkeley Lab sensor data","Buildings-heat","diurnal;sensor-dropout;battery-drift","31s","direct download","research use","http://db.csail.mit.edu/labdata/labdata.html","no","clean?",216,2.3e7,"54 motes, heavy gaps"),
("CASAS smart home sensor events","Buildings-heat","activity-routines;event-counts","event->1min","direct download","research use","https://casas.wsu.edu/datasets/","no","clean?",1500,4e8,""),
("heatpumpmonitor.org (OpenEnergyMonitor)","Buildings-heat","defrost-cycles;weather-compensation;COP","10s-10min","API","open (check per system)","https://heatpumpmonitor.org/","no","clean?",1200,1.2e8,""),
("UK Electrification of Heat heat-pump trial","Buildings-heat","defrost-cycles;cold-snaps","2min","UKDS registration","UKDS EUL","https://beta.ukdataservice.ac.uk/datacatalogue/studies/study?id=9050","no","clean?",3700,9.6e8,"742 heat pumps"),
("Battle of the Water Demand Forecasting (BWDF 2024)","Utilities-water","diurnal;weekly;holiday;weather","H","download from WDSA-CCWI","competition terms","https://www.kwrwater.nl/actueel/lessons-learned-from-the-battle-of-water-demand-forecasting/","yes","clean?",14,3.6e5,"10 DMAs plus weather"),
("NYC DSNY monthly tonnage","Utilities-waste","seasonal;holiday;COVID-shift","M","direct download","NYC open data","https://data.cityofnewyork.us/City-Government/DSNY-Monthly-Tonnage-Data/ebb7-mvp5","no","clean?",300,1.2e5,""),
("WADI water distribution testbed (iTrust)","Industrial-PHM","attacks;control-loops","1s","request","research use","https://itrust.sutd.edu.sg/itrust-labs_datasets/","no","clean?",123,1.7e8,""),
("BATADAL water network","Industrial-PHM","tank-levels;pump-switching;attacks","H","direct download","research use","https://www.batadal.net/data.html","no","clean?",43,3.7e5,""),
# Weather, climate, earth, hydrology
("ISMN soil moisture network","Weather-climate","rain-driven-jumps;exponential-drydown;freeze","H","registration (free)","ISMN terms","https://ismn.earth/","no","clean?",10000,8.7e8,""),
("FLUXNET2015 / AmeriFlux / ICOS fluxes","Weather-climate","diurnal-carbon;seasonal;gap-filled","30min","registration (CC BY 4.0 tier)","CC BY 4.0","https://fluxnet.org/data/fluxnet2015-dataset/","no","clean?",10000,1.4e9,""),
("NSIDC sea ice index","Weather-climate","annual-cycle;trend;record-lows","D","direct download","NSIDC terms","https://nsidc.org/data/g02135","no","clean?",30,4e5,""),
("HadCRUT5 / GISTEMP / Berkeley Earth","Weather-climate","trend;ENSO-wiggles","M","direct download","OGL / public","https://www.metoffice.gov.uk/hadobs/hadcrut5/","no","clean?",2600,5e6,"global plus 5-degree grid"),
("Central England Temperature (since 1659)","Weather-climate","very-long;annual-cycle;trend","D/M","direct download","OGL","https://www.metoffice.gov.uk/hadobs/hadcet/","no","clean?",3,1e5,""),
("Kyoto cherry blossom dates (812 AD-)","Weather-climate","very-long-yearly;gappy;phenology","Y","direct download","research use","http://atmenv.envi.osakafu-u.ac.jp/aono/kyophenotemp4/","no","clean?",1,1.2e3,"tiny, long-memory probe"),
("ITRDB tree-ring chronologies","Weather-climate","long-memory;yearly;century-scale","Y","direct download","US public domain","https://www.ncei.noaa.gov/products/paleoclimatology/tree-ring","no","clean?",4000,1.6e6,""),
("NOAA OISST daily sea surface temperature","Weather-climate","annual-cycle;marine-heatwaves","D","direct download","US public domain","https://www.ncei.noaa.gov/products/optimum-interpolation-sst","no","clean?",None,1.6e4,"0.25-degree grid, ~700k ocean cells"),
("GRDC global runoff stations","Hydrology","snowmelt;floods;regulation","D/M","registration (free)","GRDC terms (non-commercial)","https://grdc.bafg.de/","no","clean?",10000,1.5e8,""),
("LamaH-CE (859 Central European gauges, hourly)","Hydrology","floods;snowmelt;hourly","H/D","direct download","CC BY 4.0","https://zenodo.org/record/4525244","yes","mixed",5150,1.7e9,"daily part included in Caravan (D073); hourly part is extra"),
("CAMELS-DE and CAMELS-CH","Hydrology","snowmelt;regulation;floods","D","direct download","CC BY 4.0","https://zenodo.org/","no","clean?",11500,1.6e8,"check overlap with Caravan extensions"),
("UK Environment Agency flood-monitoring archive","Hydrology","river-level-spikes;tidal;rain","15min","API (open)","OGL","https://environment.data.gov.uk/flood-monitoring/doc/reference","no","clean?",4000,1.4e9,""),
("India-WRIS river levels and reservoirs","Hydrology","monsoon;dam-operations","D","portal / scraping","check terms","https://indiawris.gov.in/","no","clean?",1500,5.4e6,""),
("CWC India weekly reservoir storage bulletin","Hydrology","monsoon-fill;drawdown;bounded","W","PDF/scrape","GoI terms","https://cwc.gov.in/","no","clean?",150,9e4,""),
("Hydroweb lake and river altimetry levels","Hydrology","irregular-sampling;seasonal","~10-35D","registration (free)","Etalab","https://hydroweb.next.theia-land.fr/","no","clean?",20000,6e6,"irregular timestamps"),
("WGMS glacier mass balance","Weather-climate","yearly;trend","Y","direct download","WGMS terms","https://wgms.ch/data_databaseversions/","no","clean?",500,1e4,""),
("Global SPEI drought index","Weather-climate","drought-episodes;multi-scale","M","direct download","ODbL","https://spei.csic.es/database.html","no","clean?",60000,8.8e7,"one timescale"),
("INTERMAGNET geomagnetic observatories","Space-geo","diurnal-Sq;storms;secular-variation","1min","direct download","CC BY-NC 4.0","https://intermagnet.org/","no","clean?",600,6e9,""),
("GOES X-ray flux (solar flares)","Space-geo","spikes;11-year-cycle;heavy-tail","1min","direct download","US public domain","https://www.ngdc.noaa.gov/stp/satellite/goes/","no","clean?",4,8e7,""),
("F10.7 solar radio flux","Space-geo","27-day-rotation;11-year-cycle","D","direct download","open","https://www.spaceweather.gc.ca/forecast-prevision/solar-solaire/solarflux/sx-en.php","no","clean?",3,8e4,""),
("Nevada Geodetic Lab GNSS daily positions","Space-geo","trend;annual;co-seismic-steps;post-seismic-decay;equipment-offsets","D","direct download","open","https://geodesy.unr.edu/","yes","clean?",60000,2.2e8,"best real source for steps and exponential relaxation"),
("Safecast radiation measurements","Space-geo","baseline-plus-events;sensor-drift","5min","direct download","CC0","https://safecast.org/data/","no","clean?",1000,2e8,"static sensors only"),
("AAVSO variable star observations","Space-geo","quasi-periodic;irregular-sampling;century-long","irregular","registration (free)","AAVSO terms","https://www.aavso.org/data-download","no","clean?",30000,4e7,""),
("ZTF light curves (IRSA)","Space-geo","irregular;transients;periodic-variables","irregular","direct download","ZTF terms","https://irsa.ipac.caltech.edu/Missions/ztf.html","no","clean?",None,500,"~1e9 objects"),
("Santa Fe time series competition","Physics-reference","chaotic-laser;nonlinear","various","direct download","research use","http://www-psych.stanford.edu/~andreas/Time-Series/SantaFe.html","no","clean?",6,1e5,"tiny chaotic-dynamics probe"),
# Ocean, fisheries, ecology, biology
("RAM Legacy stock assessment database","Ecology","boom-bust;harvest-driven;yearly","Y","direct download","CC BY 4.0","https://www.ramlegacy.org/","no","clean?",4000,2e5,""),
("Copernicus Marine in-situ moorings","Weather-climate","tides;seasonal;storms","H","registration (free)","Copernicus licence","https://marine.copernicus.eu/","no","clean?",10000,1e9,"keep Australian moorings out (TIME has Darwin water quality, currents, coastal T-S)"),
("Continuous Plankton Recorder survey","Ecology","seasonal-blooms;regime-shifts","M","request","CPR terms","https://www.cprsurvey.org/data/","no","clean?",20000,1.6e7,""),
("NOAA fisheries landings (FOSS)","Ecology","yearly;quota-shocks","Y/M","direct download","US public domain","https://www.fisheries.noaa.gov/foss","no","clean?",20000,2e6,""),
("BioTIME 2.0 assemblage time series","Ecology","sparse-counts;yearly;turnover","Y","direct download","CC BY 4.0 (most studies)","https://biotime.st-andrews.ac.uk/","no","clean?",500000,7.5e6,"BioTIME 2.0 paper: https://andrewsforest.oregonstate.edu/node/24506"),
("Living Planet database","Ecology","population-trends;yearly;short","Y","direct download","non-commercial","https://livingplanetindex.org/download","yes","clean?",38000,7.6e5,""),
("North American Breeding Bird Survey","Ecology","counts;yearly;observer-effects","Y","direct download","US public domain","https://www.sciencebase.gov/catalog/item/52b1dfa8e4b0d9b325230cd9","no","clean?",1000000,3e7,"species x route, sparse"),
("USA-NPN phenology observations","Ecology","phenology-dates;yearly","Y/D","direct download","CC BY","https://www.usanpn.org/data","no","clean?",100000,1e6,""),
("Movebank open animal tracking studies","Ecology","irregular-GPS;migration-cycles","irregular","registration (free)","per study","https://www.movebank.org/","no","clean?",20000,2e8,""),
("BioDare2 circadian rhythm data","Biology-growth","damped-oscillation;~24h-period;phase-shifts","H/30min","public subset download","per dataset","https://biodare2.ed.ac.uk/","yes","clean?",50000,7.5e6,"public subset is smaller than total"),
("Beehive monitoring (weight, temperature, humidity)","Ecology","diurnal-foraging;swarm-drop;winter-cluster","1min","Kaggle","CC BY-NC-SA","https://www.kaggle.com/datasets/se18m502/bee-hive-metrics","no","clean?",8,4e6,""),
("Autonomous Greenhouse Challenge (Wageningen)","Agriculture-commodities","control-actions;diurnal;crop-growth","5min","direct download (4TU)","CC BY 4.0","https://data.4tu.nl/","no","clean?",250,1.3e7,""),
("IndPenSim penicillin fermentation (simulated)","Biology-growth","batch-growth;feed-control;faults","12min","direct download","research use","http://www.industrialpenicillinsimulation.com/","no","derived",3000,3.5e6,"simulated; not counted"),
# Epidemics, health, physiology
("CDC NNDSS weekly notifiable disease tables","Epidemics-health","sparse-counts;seasonal;reporting-delays","W","direct download","US public domain","https://data.cdc.gov/browse?category=NNDSS","no","clean?",7200,4.3e6,""),
("CDC NHSN respiratory hospital admissions (FluSight target)","Epidemics-health","epidemic-waves;seasonal","W","direct download","US public domain","https://data.cdc.gov/","no","clean?",160,4e4,"not FluView ILINet; fev hospital_admissions is a different 8-series daily set"),
("InfoDengue Brazil (dengue, chikungunya, zika by municipality)","Epidemics-health","epidemic-waves;climate-driven;sparse-counts","W","API","open","https://pysus.readthedocs.io/en/latest/tutorials/Infodengue.html","yes","clean?",16710,1.3e7,"5,570 municipalities since 2010"),
("UKHSA data dashboard (flu, RSV, norovirus)","Epidemics-health","seasonal-waves;post-COVID-shift","W/D","API","OGL","https://ukhsa-dashboard.data.gov.uk/","no","mixed",500,2.5e5,"exclude COVID metrics (fev uk_covid)"),
("Eurostat weekly deaths by NUTS3 (demo_r_mwk3)","Epidemics-health","winter-excess;heatwave-spikes;COVID","W","direct download","Eurostat reuse","https://db.nomics.world/Eurostat/demo_r_mwk_ts","yes","clean?",20000,2.6e7,""),
("NCHS weekly deaths by state and cause","Epidemics-health","winter-excess;COVID","W","direct download","US public domain","https://data.cdc.gov/","no","clean?",780,4.7e5,""),
("NHS England A&E attendances and admissions","Epidemics-health","winter-pressure;monthly;trust-hierarchy","M","direct download","OGL","https://www.england.nhs.uk/statistics/statistical-work-areas/ae-waiting-times-and-activity/","no","clean?",2000,3.6e5,""),
("NHS English Prescribing Data (practice x drug)","Epidemics-health","intermittent;generic-switch-steps;shortages;flu-season-antibiotics","M","direct download","OGL","https://www.nhsbsa.nhs.uk/access-our-data-products/open-data-portal-odp","yes","clean?",10000000,1e9,"~7k practices x BNF codes since 2010"),
("Google Flu Trends (2003-2015)","Epidemics-health","seasonal;known-failure-2013","W","direct download","Google terms","https://www.google.org/flutrends/about/","no","clean?",150,9e4,""),
("WastewaterSCAN","Epidemics-health","noisy-waves;lab-noise","D/2D","direct download","CC BY","https://www.wastewaterscan.org/","no","clean?",1900,1e6,""),
("Google COVID-19 symptom search trends","Web-attention","symptom-seasonality;COVID-waves","W/D","direct download","CC BY 4.0","https://github.com/google-research/open-covid-19-data","no","clean?",1270000,3.2e8,""),
("MIMIC-IV ICU vitals","Physiology","irregular;interventions;short","irregular->H","credentialed (PhysioNet)","PhysioNet credentialed","https://physionet.org/content/mimiciv/","no","clean?",730000,5e7,""),
("eICU collaborative database","Physiology","irregular;5min-vitals","5min","credentialed (PhysioNet)","PhysioNet credentialed","https://physionet.org/content/eicu-crd/","no","clean?",2000000,5e8,""),
("OhioT1DM (CGM, insulin, meals)","Physiology","meal-spikes;insulin-response;diurnal","5min","DUA","DUA","http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html","no","clean?",60,1e6,""),
("LifeSnaps (Fitbit, 71 users)","Physiology","circadian-HR;steps;sleep","1min","direct download","CC BY 4.0","https://zenodo.org/record/6826682","no","clean?",300,5e7,""),
("WESAD / PPG-DaLiA wearables","Physiology","stress-response;activity","64-700Hz","direct download","research use","https://ubicomp.eti.uni-siegen.de/home/datasets/","no","clean?",150,5e8,"resample before use"),
("Sleep-EDF expanded","Physiology","sleep-stages;ultradian-cycles","100Hz","direct download","ODC-By","https://physionet.org/content/sleep-edfx/","no","clean?",1400,1e10,"resample before use"),
("Fitbit fitness tracker data (30 users)","Physiology","daily-activity;weekly","1min","Kaggle","CC0","https://www.kaggle.com/datasets/arashnic/fitbit","no","clean?",100,4e6,""),
# Cloud, servers, web, software, networks
("Azure Functions 2021 invocation trace","Cloud-servers","bursty;periodic-timers;idle","1min/event","direct download","CC BY 4.0","https://github.com/Azure/AzurePublicDataset","yes","clean?",500,1e7,"do NOT use the 2019 Functions trace (TIME azure2019)"),
("Azure LLM inference traces 2023/2024","Cloud-servers","diurnal;bursty;token-heavy-tail","event->1min","direct download","CC BY 4.0","https://github.com/Azure/AzurePublicDataset","yes","clean?",8,8e4,"new workload type"),
("BurstGPT LLM serving traces","Cloud-servers","bursty;diurnal;model-mix","event->1min","direct download","CC BY 4.0","https://github.com/hpmll/burstgpt","yes","clean?",20,3.5e6,""),
("Huawei Cloud serverless traces","Cloud-servers","bursty;long-term-trend;timers","1min","direct download","CC BY 4.0","https://arxiv.org/abs/2312.10127","yes","clean?",5000,1.5e9,"release also on gitee/st-andrews; used in a cloud TSFM eval paper, not a standard benchmark"),
("Alibaba microservices traces 2021/2022","Cloud-servers","call-graph;diurnal;bursty","30s-1min","direct download","CC BY 4.0","https://github.com/alibaba/clusterdata","no","clean?",80000,1.4e9,""),
("Microsoft Philly GPU cluster traces","Cloud-servers","job-arrivals;GPU-util-plateaus","1min","direct download","CC BY 4.0","https://github.com/msr-fiddle/philly-traces","no","clean?",4400,4.7e8,""),
("Snowflake Snowset query workload","Cloud-servers","diurnal;weekly;warehouse-bursts","event->1min","direct download","CC BY 4.0","https://github.com/resource-disaggregation/snowset","no","clean?",5000,1e8,""),
("Parallel Workloads Archive (HPC job logs)","Cloud-servers","job-arrivals;queue-backlog","event->H","direct download","research use","https://www.cs.huji.ac.il/labs/parallel/workload/","no","clean?",160,3.2e6,""),
("SURF Lisa/Snellius HPC telemetry","Cloud-servers","node-metrics;job-driven","15s->5min","direct download (Zenodo)","CC BY 4.0","https://zenodo.org/","no","clean?",30000,1.8e9,""),
("Backblaze Drive Stats (daily SMART)","Cloud-servers","monotone-counters;degradation;failure-censoring","D","direct download","Backblaze terms (attribution)","https://www.backblaze.com/drivestats","yes","clean?",2250000,2.5e9,"~450k drives x 5 key attributes"),
("NASA KSC HTTP logs and World Cup 98 logs","Web-attention","bursty;diurnal;flash-crowds","event->1min","direct download","public","https://ita.ee.lbl.gov/html/traces.html","no","clean?",40,3e6,""),
("Wikimedia edit counts and unique devices","Web-attention","weekly;long-trend;bot-spikes","D","API","CC0","https://wikimedia.org/api/rest_v1/","no","clean?",3000,2.7e7,""),
("Stack Exchange data dump (posts per tag)","Web-attention","adoption-decline;weekly;LLM-era-break","event->W","direct download","CC BY-SA 4.0","https://archive.org/details/stackexchange","no","clean?",100000,8e7,""),
("Reddit dumps (Academic Torrents)","Web-attention","community-growth;events;weekly","event->D","torrent","Reddit terms (research)","https://academictorrents.com/","no","clean?",100000,3e8,""),
("MAWI backbone traffic archive","Telecom-network","diurnal;attacks;heavy-tail","1s within daily 15-min","direct download","research use","https://mawi.wide.ad.jp/mawi/","no","clean?",10,7e7,""),
("Abilene and GEANT traffic matrices","Telecom-network","diurnal;OD-correlation;anomalies","5min/15min","direct download","research use","https://ieee-dataport.org/documents/traffic-datsets-abilene-geant-taxibj","yes","clean?",673,1.3e7,""),
("Shanghai Telecom base station usage","Telecom-network","diurnal;location-driven","event->H","request","research use","http://sguangwang.com/TelecomDataset.html","no","clean?",3233,1.4e7,""),
("5G throughput traces (UCC)","Telecom-network","mobility-driven;handover-drops","1s","direct download","CC BY","https://github.com/uccmisl/5Gdataset","no","clean?",400,4e5,""),
("TravisTorrent CI builds","Web-attention","release-cycles;weekday","event->D","direct download","CC BY","https://travistorrent.testroots.org/","no","clean?",2600,2e6,""),
# Games, media, attention
("Steam top-100 player counts (Mendeley, 2018-2019)","Events-entertainment","diurnal;weekend;update-spikes;decay","H","direct download","CC BY 4.0","https://data.mendeley.com/datasets/fc969vwptj","yes","clean?",100,7e5,"SteamCharts history can extend this (scraping)"),
("Twitch viewing dataset (Rappaz et al. 2021)","Events-entertainment","stream-sessions;on-off;diurnal","10min","direct download","research use","https://cseweb.ucsd.edu/~jmcauley/datasets.html#twitch","no","clean?",160000,3e8,"43 days"),
("Spotify daily charts 2017-2023","Events-entertainment","release-spike;power-law-decay;Christmas-annual","D","Kaggle","CC0","https://www.kaggle.com/datasets/dhruvildave/spotify-charts","no","clean?",500000,2.6e7,"track x country, mostly short"),
("Netflix Top 10 weekly hours viewed","Events-entertainment","short-series;decay;release-driven","W","direct download","Netflix terms","https://www.netflix.com/tudum/top10","no","clean?",10000,3e4,"short-history probe"),
("Seattle Public Library checkouts by title","Events-entertainment","intermittent-long-tail;release-decay;format-shift;COVID-closure","M","direct download","public domain","https://catalog.data.gov/dataset/checkouts-by-title","yes","clean?",1000000,1e8,""),
("Smithsonian museum visitor counts","Events-entertainment","seasonal;holiday;shutdowns","M","direct download","public","https://www.si.edu/newsdesk/about/stats","no","clean?",20,6e3,"small"),
("NPS visitor use statistics (all parks)","Aviation-tourism","annual-seasonality;shutdowns;fire-closures","M","direct download","US public domain","https://www.nps.gov/aboutus/visitation-numbers.htm","yes","clean?",840,4.6e5,"~420 parks since 1979"),
("JNTO visitor arrivals to Japan","Aviation-tourism","seasonal;cherry-blossom;COVID-collapse;recovery","M","direct download","JNTO terms","https://www.jnto.go.jp/statistics/data/","no","clean?",70,5e4,""),
("India Tourism Statistics (foreign tourist arrivals)","Aviation-tourism","winter-peak;COVID","M","PDF/download","GoI terms","https://tourism.gov.in/market-research-and-statistics","no","clean?",20,6e3,"small"),
("Eurocontrol daily traffic (Aviation Intelligence Portal)","Aviation-tourism","weekly;summer-peak;COVID-collapse","D","direct download","Eurocontrol terms","https://ansperformance.eu/data/","no","clean?",340,1.1e6,""),
# Public services, safety, city
("NYC / SF / Chicago 311 service requests","Public-services","weekly;seasonal-by-type;storm-spikes","event->D","direct download","city open data","https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9","no","clean?",20000,1e8,"type x district counts"),
("Chicago crimes 2001-present (aggregated counts)","Public-services","weekly;summer-peak;trend","event->D","direct download","city open data","https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2","no","clean?",875,7.6e6,"aggregate counts only"),
("data.police.uk recorded crime (monthly)","Public-services","sparse-counts;seasonal","M","direct download","OGL","https://data.police.uk/data/","no","clean?",490000,8.3e7,"LSOA x category, sparse"),
("SF Fire Department calls for service","Public-services","diurnal;weekly;events","event->D","direct download","city open data","https://data.sfgov.org/","no","clean?",500,4.5e6,""),
("NYC motor vehicle collisions","Transport-mobility","diurnal;weather;COVID","event->D","direct download","city open data","https://data.cityofnewyork.us/","no","clean?",200,9.4e5,""),
("UK STATS19 road casualties","Transport-mobility","weekly;seasonal;policy","event->D","direct download","OGL","https://www.data.gov.uk/dataset/road-accidents-safety-data","no","clean?",400,6.4e6,""),
("NYC school daily attendance","Public-services","academic-calendar;weather-days;holidays","D","direct download","city open data","https://data.cityofnewyork.us/Education/2018-2019-Daily-Attendance/x3bb-kg5j","no","clean?",1800,1.3e6,""),
# Retail, demand, prices
("Iowa liquor sales","Retail-sales","holiday-spikes;intermittent;store-open-close;price-changes","D/W","direct download","public domain","https://data.iowa.gov/catalog/dataset/1028","yes","clean?",150000,9e7,"store x category weekly since 2012"),
("Predict Future Sales (1C Company)","Retail-sales","intermittent;new-products;December","D/M","Kaggle","competition terms","https://www.kaggle.com/c/competitive-data-science-predict-future-sales","no","clean?",424000,1.4e7,"34 months"),
("Grupo Bimbo inventory demand","Retail-sales","very-short-history;returns","W","Kaggle","competition terms","https://www.kaggle.com/c/grupo-bimbo-inventory-demand","no","clean?",20000000,7.4e7,"only 7 weeks per series: short-history probe"),
("VN1 forecasting competition (2024)","Retail-sales","intermittent;launches;stockouts","W","direct download","competition terms","https://cran.nics.utk.edu/cran/web/packages/nixtlar/vignettes/vn1-forecasting-competition.html","yes","clean?",15000,2.7e6,""),
("VN2 inventory competition (2025)","Retail-sales","stockout-censoring;weekly","W","direct download","competition terms","https://arxiv.org/pdf/2601.18919","yes","clean?",600,1e5,""),
("H&M personalized fashion transactions","Retail-sales","fashion-lifecycle;seasonal;sparse","event->D","Kaggle","competition terms","https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations","no","clean?",105000,7.7e7,""),
("JD.com MSOM 2020 data challenge","Retail-sales","promo-spikes;short","D","request","research use","https://connect.informs.org/msom/events/datadriven2020","no","clean?",30000,9.3e5,"1 month only"),
("Food demand forecasting (meal kits, weekly)","Retail-sales","promotions;weekly","W","Kaggle","CC0","https://www.kaggle.com/datasets/kannanaikkal/food-demand-forecasting","no","clean?",3600,5.2e5,""),
("Hass Avocado Board weekly prices and volumes","Agriculture-commodities","seasonal-supply;holiday-demand;price-volume","W","direct download","HAB terms","https://hassavocadoboard.com/","no","clean?",700,3.5e5,""),
("Hotel booking demand (Antonio et al.)","Aviation-tourism","seasonal;cancellations;events","D","direct download","CC BY 4.0","https://www.sciencedirect.com/science/article/pii/S2352340918315191","no","clean?",10,1e4,"small"),
("Inside Airbnb calendars and listings","Housing-realestate","booking-curve;events;holiday-pricing","D (forward calendar)","direct download","CC BY 4.0","https://insideairbnb.com/get-the-data/","no","clean?",4000000,1.5e9,"forward-looking calendars per scrape; occupancy must be derived"),
("French bakery daily sales","Retail-sales","weekly;holiday;intermittent","D","Kaggle","CC0","https://www.kaggle.com/datasets/matthieugimbert/french-bakery-daily-sales","no","clean?",150,9e4,"small"),
("Technion SEE call-centre data","Public-services","intraday-arrivals;weekly;holiday","15min","registration","research use","https://iew.technion.ac.il/serveng/","no","clean?",20,1.5e6,""),
("NYC rolling property sales","Housing-realestate","sparse;seasonal","event->M","direct download","city open data","https://www.nyc.gov/site/finance/property/property-rolling-sales-data.page","no","clean?",2500,6.5e5,""),
("FHFA house price indices (state, metro, ZIP)","Housing-realestate","smooth-trend;2008-crash","M/Q/Y","direct download","US public domain","https://www.fhfa.gov/data/hpi","no","clean?",20000,6e5,""),
("Apartment List rent estimates","Housing-realestate","rent-cycle;COVID-dip","M","direct download","CC BY","https://www.apartmentlist.com/research/category/data-rent-estimates","no","clean?",1500,1.5e5,""),
# Macro, finance, government
("RBI DBIE (India) money, banking, prices, HPI","Macro-finance","festival-cash-demand;policy-steps","W/M/Q","direct download","GoI terms","https://data.rbi.org.in/","no","clean?",10000,3e6,""),
("MOSPI India CPI / IIP / WPI","Macro-finance","seasonal;base-revisions","M","direct download","GoI terms","https://esankhyiki.mospi.gov.in/","no","clean?",5000,8e5,""),
("BIS statistics (credit, property prices, FX)","Macro-finance","cycles;crises","M/Q","API","BIS terms","https://data.bis.org/","no","clean?",50000,1e7,""),
("IMF IFS / DOTS / CPI","Macro-finance","cycles;crises;revisions","M/Q","API","IMF terms","https://data.imf.org/","no","mixed",500000,1e8,"exclude primary commodity prices (possible TIME Global_Price/Commodity_*) and PortWatch (TIME Port_Activity)"),
("DBnomics aggregator","Macro-finance","various","various","API","per provider","https://db.nomics.world/","yes","derived",None,200,"re-serves Eurostat/IMF/OECD/etc.; not counted"),
("Stooq bulk historical data","Crypto-markets","random-walk;volatility-clustering","D","direct download","Stooq terms","https://stooq.com/db/h/","no","derived",60000,3e8,"same markets as D172; not counted"),
("SEC EDGAR XBRL company facts","Macro-finance","quarterly-seasonality;growth;accounting-steps","Q","API (free)","US public domain","https://www.sec.gov/edgar/sec-api-documentation","yes","clean?",300000,1.8e7,"~15k companies x key concepts since 2009"),
("CBOE VIX and volatility indices","Macro-finance","mean-reverting;spikes","D","direct download","CBOE terms","https://www.cboe.com/tradable_products/vix/vix_historical_data/","no","clean?",20,1.8e5,""),
("US Treasury Daily Treasury Statement","Macro-finance","tax-dates;month-end;calendar-effects","D","API","US public domain","https://fiscaldata.treasury.gov/datasets/daily-treasury-statement/","no","clean?",300,1.5e6,""),
("Japan e-Stat official statistics","Government-stats","various","M/Q/Y","API (free key)","e-Stat terms","https://www.e-stat.go.jp/en","no","clean?",None,200,""),
# Agriculture, food, remote sensing
("Agmarknet daily mandi prices and arrivals","Agriculture-commodities","festival-demand;monsoon-harvest;onion-tomato-shocks;gaps","D","direct download / API","GODL India","https://data.gov.in/resource/current-daily-price-various-commodities-various-markets-mandi","yes","clean?",400000,6e8,"historical mirror: https://agmarknet.ceda.ashoka.edu.in/README.html"),
("WFP global food prices (HDX)","Agriculture-commodities","conflict-shocks;currency-collapse;seasonal","M","direct download","CC BY-IGO","https://data.humdata.org/dataset/global-wfp-food-prices","yes","clean?",80000,8e6,""),
("FAO GIEWS FPMA domestic food prices","Agriculture-commodities","harvest-seasonality;shocks","M/W","direct download","FAO terms","https://fpma.fao.org/giews/fpmat4/","no","clean?",1500,3.6e5,"partly overlaps WFP"),
("USDA AMS Market News","Agriculture-commodities","daily-auctions;seasonal-supply","D/W","API","US public domain","https://mymarketnews.ams.usda.gov/","no","clean?",50000,2e8,""),
("MODIS NDVI time series (MOD13Q1)","Agriculture-commodities","phenology;crop-cycles;cloud-gaps","16D","API (free Earthdata login)","US public domain","https://lpdaac.usgs.gov/products/mod13q1v061/","no","clean?",None,550,"any pixel since 2000"),
# Transport, mobility
("UTD19 urban loop detectors (40 cities)","Transport-mobility","rush-hours;congestion-regimes;fundamental-diagram","3-5min","registration (free)","research use","https://www.research-collection.ethz.ch/handle/20.500.11850/437802","yes","clean?",47000,1.7e8,"~23.5k detectors, flow and occupancy; exclude any Finnish city (TIME Finland_Traffic)"),
("NDW Netherlands national traffic data","Transport-mobility","rush-hours;incidents","1min->15min","direct download","CC0","https://opendata.ndw.nu/","no","clean?",30000,5.2e9,""),
("BASt German Autobahn hourly counts","Transport-mobility","rush-hours;holiday-travel;trucks-weekday","H","direct download","dl-de/by-2-0","https://www.bast.de/DE/Verkehrstechnik/Fachthemen/v2-verkehrszaehlung/Stundenwerte.html","no","clean?",4000,7e8,""),
("Paris bike counters (comptage velo)","Transport-mobility","commute-peaks;weather;strikes","H","direct download","ODbL","https://opendata.paris.fr/explore/dataset/comptage-velo-historique-donnees-compteurs/","yes","clean?",200,1.2e7,""),
("Zurich bike and pedestrian counts","Transport-mobility","commute-peaks;weather","15min","direct download","CC0","https://data.stadt-zuerich.ch/dataset/ted_taz_verkehrszaehlungen_werte_fussgaenger_velo","no","clean?",150,7.5e7,""),
("Melbourne pedestrian counting system","Transport-mobility","events;weekly;COVID","H","direct download","CC BY 4.0","https://data.melbourne.vic.gov.au/","no","pretrain",90,1.2e7,"Monash pedestrian_counts copy is in Chronos/LOTSA corpora"),
("Seoul subway and Ttareungyi bike rentals","Transport-mobility","commute-peaks;holiday;weather","H","direct download","KOGL","https://data.seoul.go.kr/","no","clean?",4000,3e8,""),
("Taipei MRT hourly origin-destination ridership","Transport-mobility","commute-peaks;OD-structure;typhoon-days","H","direct download","Taipei open data","https://data.taipei/","no","clean?",14000,8e8,""),
("Chicago CTA daily ridership (rail and bus)","Transport-mobility","weekly;COVID-collapse;events","D","direct download","city open data","https://data.cityofchicago.org/Transportation/CTA-Ridership-L-Station-Entries-Daily-Totals/5neh-572f","no","clean?",275,2.4e6,""),
("MarineCadastre AIS vessel traffic (US)","Transport-mobility","port-congestion;seasonal","1min->H","direct download","US public domain","https://hub.marinecadastre.gov/pages/vesseltraffic","no","clean?",500,6.5e7,"build own counts; do not use IMF PortWatch (TIME Port_Activity)"),
("Port of Los Angeles container statistics","Transport-mobility","seasonal;import-surges;strikes","M","direct download","port terms","https://www.portoflosangeles.org/business/statistics/container-statistics","no","clean?",20,7e3,"check against TIME Port_Activity before use"),
("BTS border crossing entry data","Transport-mobility","seasonal;COVID-closure","M","direct download","US public domain","https://data.bts.gov/Research-and-Statistics/Border-Crossing-Entry-Data/keg4-3bc2","no","clean?",1400,4.8e5,""),
("City EV charging open data (Palo Alto, Boulder, Dundee, Perth)","EV-parking","session-arrivals;workday-peaks;growth","event->H","direct download","city open data","https://data.cityofpaloalto.org/","no","clean?",100,7e6,""),
# Industrial / PHM
("3W dataset 2.0 (Petrobras oil wells)","Industrial-PHM","rare-events;slugging;hydrate-buildup;transients","1s","direct download","CC BY 4.0","https://github.com/ricardovvargas/3w_dataset","yes","clean?",17800,5e8,""),
("SKAB Skoltech anomaly benchmark","Industrial-PHM","valve-faults;control-loops","1s","direct download","GPL-3.0","https://github.com/waico/SKAB","no","clean?",272,3e5,""),
("Hydraulic system condition monitoring (UCI)","Industrial-PHM","cyclic-load;degradation","1-100Hz cycles","direct download","CC BY 4.0","https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems","no","clean?",37000,4e7,""),
("Steel industry energy consumption (UCI)","Industrial-PHM","shift-patterns;weekly","15min","direct download","CC BY 4.0","https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption","no","clean?",6,2.1e5,"small"),
("FEMTO/PRONOSTIA and XJTU-SY bearings","Industrial-PHM","run-to-failure;exponential-degradation","snapshots","direct download","research use","https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset","no","clean?",64,1.3e5,"use per-snapshot features"),
("Battery degradation (MIT-Stanford, Oxford, CALCE)","Industrial-PHM","capacity-fade;knee-point;recovery-bumps","cycle","direct download","CC BY 4.0","https://data.matr.io/1/","no","clean?",400,3e5,"capacity and resistance per cycle"),
("N-CMAPSS turbofan (real flight profiles)","Industrial-PHM","degradation;flight-cycles","1Hz","direct download","US public domain","https://data.nasa.gov/","no","clean?",6000,3e9,"downsample before use"),
("Gas sensor array under dynamic mixtures (UCI)","Industrial-PHM","sensor-drift;step-responses","100Hz","direct download","CC BY 4.0","https://archive.ics.uci.edu/dataset/322/gas+sensor+array+under+dynamic+gas+mixtures","no","clean?",32,1.3e8,""),
("Pump sensor data (52 sensors, Kaggle)","Industrial-PHM","failures;recovery;sensor-dropout","1min","Kaggle","check page","https://www.kaggle.com/datasets/nphantawee/pump-sensor-data","no","clean?",52,1.1e7,""),
("Mining flotation plant process data","Industrial-PHM","control;lagged-quality","20s/H","Kaggle","CC0","https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process","no","clean?",23,1.7e7,""),
# Batch 2 held-out countries (owner 2026-10-29); appended last so earlier ids keep their numbers (D401-D403)
("NESO historic demand data (Great Britain)","Energy-load","weekly;holidays;embedded-solar;temperature","30min","direct download","NESO Open Data Licence","https://neso.energy/data-portal/historic-demand-data","yes","clean?",8,1.3e6,"yearly CSVs (api.neso.energy, resource list via CKAN datapackage_show?id=historic-demand-data); value-level check vs fev entsoe (does entsoe hold GB?) on Perlmutter; Elexon D004 is the same demand, not a fallback"),
("ANAC Voo Regular Ativo (Brazil, every scheduled flight)","Aviation-tourism","weekly;holidays;carnival;COVID","event->D","direct download","BR federal open data (check record)","https://sistemas.anac.gov.br/dadosabertos/Voos%20e%20opera%C3%A7%C3%B5es%20a%C3%A9reas/Voo%20Regular%20Ativo%20(VRA)/","yes","clean?",500,4e6,"yearly folders 2000-2026 (listing fetched 2026-10-29); daily departures per airport; licence of the dados.gov.br record to confirm (Decreto 8.777/2016 open-data policy)"),
("CEA state power supply position (India, data.gov.in)","Energy-load","festival;monsoon;growth","M","API (free key)","GODL India","https://www.data.gov.in/","no","clean?",36,8000,"standby second IN source if Grid-India terms forbid automated access: monthly state peak demand / met 2006-2026 (Ministry of Power / CEA)"),
]

# ------------------------------------------------------------------ build
def access_class(a):
    a = a.lower()
    if any(k in a for k in ["credential", "dua", "ukds", "secure"]): return "restricted"
    if any(k in a for k in ["scrap", "pdf"]): return "scrape"
    if "application" in a: return "application"      # an application to the data holder (2026-10-27)
    if any(k in a for k in ["key", "token", "registration", "gated", "request", "login", "rate-limited", "torrent"]): return "registration"
    return "open"

DEV = {"bench": "diagnosis-only (do not select on it)", "likely_time": "diagnosis-only (TIME family)",
       "pretrain": "ok for our twins; reference models saw it", "pretrain?": "ok for our twins; reference models likely saw it",
       "mixed": "keep, but drop the listed benchmark sub-series", "pool": "already in real post-training pool, not dev",
       "clean?": "yes, after audit", "derived": "not counted (duplicate/derived/simulated)", "n/a": "covariate source, not a series"}

# Hand fixes of v1 rows (catalog/CHANGELOG.md 2026-10-27), ported into the builder 2026-10-31 so a rebuild keeps them
HAND_FIXES = {
    'D026': {'url': 'https://www.ucd.ie/issda/data/commissionforenergyregulationcer', 'url_verified': 'no', 'notes': 'Listed in a smart-meter data review; access needs an application'},
    'D027': {'url': 'https://dataport.pecanstreet.org/', 'url_verified': 'yes', 'notes': 'Registration; counted size = the free University Access tier (75 homes); full data is paid'},
    'D028': {'url': 'https://traces.cs.umass.edu/docs/traces/smartstar', 'url_verified': 'yes', 'notes': 'Counted: the one linked file (443 homes, one 24-hour period; 1-min resolution assumed). The minute-level 400+ home data is not linked (checked 2026-10-27)'},
    'D029': {'url': 'https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data', 'url_verified': 'no', 'notes': 'Listed in smart-meter review; URL on ausgrid.com.au to confirm'},
}

kept, excluded = [], []
for x in rows:
    i = x["id"]
    x.update(HAND_FIXES.get(i, {}))
    if i in RESTORE:
        x["overlap"], note = RESTORE[i]; x["notes"] = note
    if x["overlap"] in ("bench", "likely_time") or i in NEW_DROPS:
        excluded.append(dict(id=i, name=x["name"], domain=x["domain"],
                             reason=NEW_DROPS.get(i, x["notes"] or "flagged " + x["overlap"])))
        continue
    if i in NEW_MIXED:
        x["overlap"] = "mixed"; x["notes"] = (x["notes"] + "; " if x["notes"] else "") + "EXCLUDE: " + NEW_MIXED[i]
    s, p = S[i]
    x.update(series_avail=s, points_avail=p, size_basis="est", counted="no" if i in NOT_COUNTED else "yes", source_version="v1")
    if i in NOT_COUNTED: x["notes"] = (x["notes"] + "; " if x["notes"] else "") + "not counted: " + NOT_COUNTED[i]
    kept.append(x)

nid = 214
for (name, dom, beh, freq, acc, lic, url, ver, ov, s, p, notes) in N:
    kept.append(dict(id=f"D{nid:03d}", name=name, domain=dom, behaviours=beh, freq=freq, access=acc, licence=lic, url=url,
                     url_verified=ver, overlap=ov, dev_use="", notes=notes, series_avail=s, points_avail=p, size_basis="est",
                     counted="no" if ov == "derived" else "yes", source_version="v2-new"))
    nid += 1

for x in kept:
    x["dev_use"] = DEV.get(x["overlap"], x.get("dev_use", ""))
    x["access_class"] = access_class(x["access"])
    s, p = x["series_avail"], x["points_avail"]
    if s is None:                        # unbounded: points_avail holds typical length
        avg = p; x["series_avail"] = ""; x["points_avail"] = ""; x["unbounded"] = "yes"
        sp = 500
    else:
        avg = (p / s) if s else 0; x["unbounded"] = "no"
        sp = min(s, 500)
    lp = min(avg, 10000)
    x["avg_len_est"] = int(round(avg))
    x["series_pull"] = int(sp) if x["counted"] == "yes" and x["overlap"] not in ("pool", "n/a") else 0
    x["points_pull"] = int(x["series_pull"] * lp)

# ------------------------------------------------------------------ checks
names = collections.Counter(x["name"] for x in kept)
assert not [n for n, c in names.items() if c > 1], "duplicate names"
assert len({x["id"] for x in kept}) == len(kept)

cols = ["id","name","domain","behaviours","freq","access","access_class","licence","url","url_verified","overlap","dev_use",
        "series_avail","points_avail","avg_len_est","unbounded","series_pull","points_pull","size_basis","counted","source_version","notes"]
with open(os.path.join(HERE, "dataset_catalog_v2.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(kept)
with open(os.path.join(HERE, "excluded_benchmark.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id","name","domain","reason"]); w.writeheader(); w.writerows(excluded)
with open(os.path.join(HERE, "benchmark_datasets.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["benchmark","dataset","original_source"])
    w = csv.writer(f); 
    for b, L in (("TIME", TIME), ("fev-bench", FEV), ("GIFT-Eval test", GIFT)):
        for d in L: w.writerow([b, d, TIME_SRC.get(d, "")])

# ------------------------------------------------------------------ totals
def fmt(v):
    for u, d in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if v >= d: return f"{v/d:.2f}{u}"
    return str(int(v))

C = [x for x in kept if x["counted"] == "yes" and x["overlap"] not in ("pool", "n/a")]
B = [x for x in C if x["unbounded"] == "no"]
print("kept rows:", len(kept), "| new rows:", len(N), "| excluded as benchmark:", len(excluded))
print("counted datasets:", len(C), "| bounded:", len(B), "| unbounded sources:", len(C) - len(B))
ts = sum(x["series_avail"] for x in B); tp = sum(x["points_avail"] for x in B)
print(f"AVAILABLE (bounded only): series {fmt(ts)}  points {fmt(tp)}")
top = sorted(B, key=lambda x: -x["points_avail"])
t10 = sum(x["points_avail"] for x in top[:10])
print(f"top-10 datasets hold {100*t10/tp:.1f}% of points:")
for x in top[:10]: print(f"   {x['id']} {x['name'][:55]:55s} {fmt(x['points_avail'])}")
for thr in (1e9, 1e8):
    sub = [x for x in B if x["points_avail"] < thr]
    print(f"datasets under {fmt(thr)} points: {len(sub)}  series {fmt(sum(x['series_avail'] for x in sub))}  points {fmt(sum(x['points_avail'] for x in sub))}")
ts_s = sorted(B, key=lambda x: -x["series_avail"])
t10s = sum(x["series_avail"] for x in ts_s[:10])
print(f"top-10 datasets hold {100*t10s/ts:.1f}% of series")
print(f"DEV PULL (<=500 series, <=10k points each): series {fmt(sum(x['series_pull'] for x in C))}  points {fmt(sum(x['points_pull'] for x in C))}")
for cls in ("open", "registration", "scrape", "restricted"):
    sub = [x for x in B if x["access_class"] == cls]
    subc = [x for x in C if x["access_class"] == cls]
    print(f"  {cls:12s} datasets {len(subc):3d}  avail series {fmt(sum(x['series_avail'] for x in sub)):>8s} points {fmt(sum(x['points_avail'] for x in sub)):>8s}  pull points {fmt(sum(x['points_pull'] for x in subc))}")
print("by domain (kept rows):")
for d, c in collections.Counter(x["domain"] for x in kept).most_common(): print(f"   {d:26s}{c}")
print("overlap:", dict(collections.Counter(x["overlap"] for x in kept)))
print("url_verified:", dict(collections.Counter(x["url_verified"] for x in kept)))
