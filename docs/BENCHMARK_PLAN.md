# tideforge private benchmark (tfbench): plan (2026-11-12)

Copied from the project notes. Goal: a large, private, fev-style forecasting benchmark built from the Hugging Face catalog
(docs/HF_DATA_CATALOG.md), used to analyse where our models fail and to steer generator and recipe changes. fev-bench and
TIME become milestone checks only.

## Where it is built and kept
- Built on a rented JarvisLabs machine (28 vCPU, 124 GB RAM, L4 24 GB, 2 TB disk; ~800 GB/h download measured).
- Source data: catalog rows of at most 20 GB each (502 rows from 332 repos, ~681 GB in scope by the profile).
- The finished benchmark goes to a private Hugging Face repo (owner's Pro account, name in env var TF_BENCH_REPO),
  versioned and frozen per version. Raw downloads are temporary and can be deleted after the build.

## Task design
- fev's task format (dataset + horizon + number of windows + past / known covariate columns), so the existing scorer
  and the planned GPU scorer read it directly.
- Per dataset, tasks vary by: frequency (native + coarser aggregations where the series is long enough), horizon
  (short / medium / long), covariates (target-only and with-covariates where covariates exist), multivariate (joint task
  where channels belong together).
- Estimate: ~350 clean datasets -> 1,500-3,500 tasks (point estimate ~2,400). Evaluation windows at the end of each series.

## Tiers and weighting
- Full (~2,000-3,000 tasks): milestones. Core (~500-600: one task per dataset and frequency, medium horizon, covariates
  where they exist): regular experiments. Fast (~150, stratified): every run.
- Headline score weights each source dataset equally (tasks of one dataset share its weight), as fev does.
- Confidence intervals by bootstrap over datasets, not tasks: tasks of one dataset are correlated, so the effective
  sample size is the number of datasets (~260 in the tuning split), not the number of tasks.
- Every task is tagged (domain, frequency band, horizon, covariates, history length, intermittency, series length) so a
  single run produces slice tables.

## Split and leakage
- Split by whole dataset: ~75% tuning, ~25% locked. The locked split is checked only at milestones and every look is
  counted (as LOOKS.md does for fev).
- Dataset-level exclusion of fev-bench / GIFT-Eval / TIME sources (catalog rules), plus a value-level check: hashes of
  normalised windows against those benchmarks.
- Sanity check: score Chronos-2, TimesFM, Timestral, TiRex, TimeH2O on it; their ranking should broadly match fev's.

## The analysis loop it enables
1. Score our model and the public models on the tuning split; per task, the gap to the best public model is the
   "fixable error".
2. Explain the gap with series features (period, seasonal / trend strength, intermittency, noise, history length,
   covariates, frequency, horizon). Look for groups of tasks that fail for the same reason.
3. Map each failure group to a generator change (new family, changed share, changed parameters), and check the
   synthetic data covers the region (coverage map, realism classifier, mixture fit).
4. Train at 5M, re-score the tuning split; keep changes that help by more than the seed noise.
5. Confirm on the locked split at milestones; fev and TIME at the end only.

Rule against overfitting the benchmark: changes must be justified by a feature-level explanation (a kind of series),
never by individual tasks; the locked split decides.
