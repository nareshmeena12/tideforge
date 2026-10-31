# Catalog change log

Every hand change to the catalog files, newest first. The generators (build_catalog.py, expand_catalog.py) are changed in
the same commit so that a rebuild keeps the change.

## 2026-10-31

- expand_catalog.py: the 2026-10-27 hand fixes of D026-D029 (url, url_verified, notes) are now in the generator
  (HAND_FIXES), so a rebuild reproduces dataset_catalog_v2.csv exactly; tests/test_catalog_builder.py checks that the
  builder's three outputs equal the committed files.
- horizons_extra.csv: frequency `index` (season 1; short / medium / long = 10 / 25 / 50 steps; owner 2026-10-31) for series
  without calendar time (D279, D202, D110).

## 2026-10-29

- D401 NESO historic demand data (GB), D402 ANAC Voo Regular Ativo (BR), D403 CEA state power supply position (IN, standby):
  new rows for batch 2's held-out countries (owner OK 2026-10-29), appended to expand_catalog.py's list so earlier ids keep
  their numbers. Only these three rows were added to dataset_catalog_v2.csv by hand from a rebuild: a full rebuild today
  would undo the 2026-10-27 URL fixes of D026-D029, which were made in the CSV but not in the generator (a drift found on
  2026-10-29, not fixed here).
- aggregator_rules.csv: D001 / D002 (OPSD / ENTSO-E) drop Great Britain for D401 (NESO), as they do for D004 (Elexon).
- split_v1.csv: extended with D401 (Energy-load, locked), D402 (Aviation-tourism, tuning), D403 (Energy-load, tuning).

## 2026-10-27

- D027 Pecan Street: access_class registration (unchanged) and counted size = the free University Access tier: 75 series,
  1.31e7 points available (175k each), dev pull 75 series / 750k points (was 1000 / 1.75e8, pull 500 / 5M).
- D028 UMass Smart*: counted size = the one linked file (443 homes, one 24-hour period; 1-min resolution assumed until
  downloaded): 443 series, 637,920 points, pull the same (was 500 / 1e7, pull 500 / 5M). The minute-level 400+ home data
  is marked "not linked". At ~1440 points per series most of it will fail the length rule (3 x longest horizon + 64).
- Totals after both changes (counted, not pool / covariate-only, 365 datasets): available (bounded sources) 61.88M series,
  400.51B points; dev pull 131,089 series, 641.4M points (was 650.0M). expand_catalog.py sizes updated to match.
- D026 Irish CER smart meter trial: url arXiv review paper -> ISSDA's CER data page
  (https://www.ucd.ie/issda/data/commissionforenergyregulationcer; found by search, the page refuses automated fetches,
  url_verified no); access_class open -> application (ISSDA supplies the data on application; expand_catalog.py now maps
  "application" access to this class).
- D027 Pecan Street Dataport: url arXiv review paper -> https://dataport.pecanstreet.org/ (sign-up; free University Access
  for non-commercial academic use covers 75 homes, full data is paid). access_class stays registration.
- D028 UMass Smart* microgrid: url arXiv review paper -> https://traces.cs.umass.edu/docs/traces/smartstar. The page links
  only a one-day microgrid file (443 homes); the minute-level 400+ home data is mentioned but not linked, so the
  catalog's 500 series / 5M points for D028 is not reachable from it (note added; counts left as they were).
- D029 Ausgrid solar home dataset: url arXiv review paper -> Ausgrid's "Solar home electricity data" page (found by
  search; the page refuses automated fetches, url_verified no; the batch-1 probe confirms the file links).
