# Catalog change log

Every hand change to the catalog files, newest first. The generators (build_catalog.py, expand_catalog.py) are changed in
the same commit so that a rebuild keeps the change.

## 2026-10-27

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
