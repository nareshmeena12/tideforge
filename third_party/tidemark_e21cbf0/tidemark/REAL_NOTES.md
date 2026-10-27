# Real-data post-training track: notes

Not part of the softmax / GLA / GDN / SWA study (see CLAUDE.md). Code: `tidemark/real/`, `scripts/real/`, `scripts/perlmutter/real_*.sh`.

## Stage 1: baseline and sizing (code written 2026-10-24, nothing run yet)

### Missing values: what the existing pipeline already does (read from `tidemark/real/ft3_core.py`, a verbatim copy of the reference)
| Owner's rule | Existing code | Match |
|---|---|---|
| Normalisation stats over valid points only | `_causal_stats(x, obs, ...)`: running mean / std of the OBSERVED points of the series (float64, segmented per series); `obs = mask & ~blank` | yes |
| Missing flag, value 0 after normalisation | `_core`: `z = where(obs, asinh((x - mu) / sd), 0)`; the patch input is `[z, obs]` (mask channel) | yes |
| Missing target points left out of the loss | `forward`: targets use the TRUE `mask` (`ym = mask[:, 1:]`), weights `w = ym & ok_b`; `ok_b` also needs the predicting patch to have >= 1 observed point (`cnt_e > 0`) | yes |
| Skip all-NaN windows; require valid points in the horizon part | no window filter exists. A series without any valid target gets no loss (`has = den > 0`), so it costs compute but not correctness | **difference** |
| Count only valid points toward budgets / caps | the batcher counts every position of a context (`points = n_ctx + n_tgt`); the synthetic data has no holes | **difference** |

Other differences found:
1. **Synthetic training never contains interior holes.** `_L_corruption` makes missing runs and then FORWARD-FILLS them (its docstring: the attention has no padding-mask support). The mask channel is `False` only for left padding and for CPM-blanked spans. Real data with 17 to 41 % NaN puts interior holes with mask 0 in front of the model for the first time. The mechanics are supported by the code above; the model's behaviour on them is untested.
2. **Inference drops non-finite points** (`forecast_left_padded`: `y = y[np.isfinite(y)]`), it does not keep holes; `forecast_groups` (`_lane_array`) forward-fills them. So the hole handling learned in post-training is not exercised by the repo's TIME / fev scoring.
3. A CPM-blanked span (hidden from the input, still a target) can land on points that are already missing; those targets are left out of the loss like any other missing target.
4. Real windows therefore need their own filters (done in the loader, stage 3): not all-NaN, a minimum share of valid points, some valid points in the last patch(es) so the horizon has targets.

### Conversion rule (`tidemark/real/store.py`)
NaN stays NaN in the float32 shards; non-finite values (inf) become NaN; leading and trailing missing values are trimmed; finite values clamped to +-1e9; series with fewer than 33 VALID points are dropped. A dataset's `unique_points` (the number the plan and the epoch caps use) is its count of valid points.

### Baseline (order step 1)
`scripts/perlmutter/real_eval.sh --baseline`: strict load + forward parity (`scripts/real/check_forward.py`), then the repo's TIME and fev-bench scoring of `ft3_main.pt` through `tidemark/real/score_ft3.py`, then the table and the comparison with TIME MASE 0.680 (+-0.002) and fev 43.78 / 42.09 (+-0.2). **Inference mode is univariate** (`forecast_left_padded`, the same as the 30M twins' scoring). The "groups" mode (`forecast_groups` with covariates) is not wired into `score_twins.py`; if the owner's 43.78 / 42.09 come from it, the baseline cannot be reproduced with the repo's scoring code until that mode is added.

Result: (to be filled in after the run)

### Sizing report (order step 2)
`python -m tidemark.real.convert` then `python -m tidemark.real.plan --real-points 10e9 --max-epochs 4`. Result: (to be filled in after the run)

## Decisions of 2026-10-24 (second round), applied
- **Baseline gate is univariate:** fev 42.09 +-0.2, TIME MASE 0.687 and CRPS 0.575 +-0.002. The groups-mode fev (43.78) is not gated; groups-mode scoring and `time_bench_colab.py` come later from the owner.
- **Missing values, changed rule:** interior gaps are FORWARD-FILLED in the stored inputs with the observed flag set (as the scorers and `_L_corruption` do); the originally missing positions are stored separately (`missing.i64`) and will feed a separate LOSS mask that leaves them out of the loss. Windows are skipped when all targets are missing or more than 50 % of the horizon part is. This needs a loss mask the reference model does not have (its targets use the same mask as its inputs): a subclass of `CausalPatchDecoder` in the loader / trainer stage, same state dict, identical to the reference when the loss mask is all True.
- **Real half:** `--weather-max-share 0.77`, weather split hourly 0.5 / daily 0.4 / weekly 0.1; non-weather at its 4-pass cap. A weather part above 4 epochs is flagged in the plan, not silently changed.
- **Rows:** 288 per GPU, 96 real + 192 synthetic, about 3,200 steps for 30B points.
- Convert and baseline run inside the owner's interactive session.

## Stages 3-4: loader and trainer (code written 2026-10-24, nothing run on Perlmutter yet)
- **Loss-masked decoder** (`tidemark/real/model.py`): `CausalPatchDecoder` with a `loss_mask`; the packed forward differs from the reference in exactly two lines (signature, target mask `mask & loss_mask`), checked by a source comparison in `tests/test_real_model.py`. Same state dict (ft3_main loads with `strict=True`); the loss and the quantiles equal the reference's bit for bit when the loss mask is all True.
- **Batches** (`tidemark/real/sampler.py`): per rank 288 rows = 192 synthetic + 96 real (config `MixConfig.rows / real_fraction`; rows tile exactly, so the row split is the point split up to missing points). Real cells: dataset by the plan's targets with a feedback controller on the realised shares (only series with at least c points can serve a context of c, so a short-series dataset cannot reach its target on long contexts; best effort, reported), then series uniform among the eligible, then a uniform window of exactly c points. Interior gaps forward-filled and observed; originally missing positions are not scored. Windows are redrawn when the last 128 points are all missing or more than 50 % missing (up to 20 tries; the counts of redraws and forced acceptances are logged). 30 % of the real cells go through ft3's group builder. `points` counts valid points only.
- **Synthetic stream** is ft3_main's generator (the checkpoint's own cfg); its generation pools are a pure function of (seed, rank, bucket, pool number), so a resumed run rebuilds the pool it was in.
- **Trainer** (`tidemark/real/train.py`): strict init from ft3_main.pt, DDP on 4 GPUs, bf16, `torch.compile`, AdamW (wd 0.01), clip 1.0, LR a function of the global valid points (5 % warmup from 4 % of the peak to 1.2e-4, cosine to 10 % over the full budget). Exact resume: model, optimizer, step, points, every rank's CPU/CUDA RNG and sampler state captured just before each batch is generated; verified bit-identical on CPU (tests/test_real_train.py). A different world size or a changed configuration is refused.
- **Open items:** nothing here has run on a GPU or under DDP+compile (the first smoke run will show it; DDP assumes every parameter gets a gradient every step). The loader throughput is unmeasured: `python -m tidemark.real.bench_loader`. Part A (groups-mode scoring) is waiting for the reference files.

## Part A: groups-mode scoring (code written 2026-10-24; the gate has NOT been run)
`tidemark/real/score_groups.py`, launcher `real_eval.sh --both`. Every evaluation reports two modes per checkpoint NAME: OFF (tag `NAME_s0`, the univariate scorer, unchanged) and ON (tag `NAMEon_s0`).
- **TIME groups ON** follows `reference/time_bench_colab.py`: a multivariate dataset (D > 1) is one group per instance (`ft3_core.forecast_groups`, rows 256), univariate datasets are forecast as in OFF (one forecast, saved under both tags). History gaps are filled one season back.
- **fev groups ON**: one group per item and window: target columns (sibling variates), past dynamic covariates (PAST), known dynamic covariates with their future values (KNOWN); static covariates unused; a task with one target and no covariate takes the OFF path (the Colab log shows identical ON and HIDDEN numbers on those 33 tasks).
- **Not in `reference/`, so reconstructed:** `forecast_groups_multi`, `forecast_packed_multi`, `_fill_one` (all in `score_tide100m_colab.py`) and `predict_task` (the fev cell). `ft3_core.forecast_groups` / `forecast_left_padded` stand in for the first two (same algorithm per the repo's copy). Assumptions that could each move the fev ON number: how non-numeric covariates are encoded (here: sorted distinct values of the column in the window -> 0, 1, 2 ..., missing -> NaN), which covariates are kept when a group has more than 8 lanes (column order; known before past), and whether the Colab cell filled covariate gaps the same way.
- **Gate** (`real_eval.sh --baseline --both`): ON fev 43.78 +-0.2, TIME MASE 0.680 / CRPS 0.571 +-0.002; OFF fev 42.09 +-0.2, TIME 0.687 / 0.575 +-0.002 (the OFF side already passed: 42.08, 0.6873, 0.5752). Then `compare-log` lists the largest per-task differences against `fev_log_ft3.txt` and the number of tasks above 0.005. A failed gate means: stop and report what differs; nothing is tuned.
- **Table**: TIME MASE, TIME CRPS, fev skill, fev win rate for each mode and the change versus ft3main, in `eval/board/real_table_modes.csv`.

Result: (to be filled in after the run)

## Full weatherbench pool: all 107 shards of the 8 pressure-level variables (tooling written 2026-10-24, nothing run yet)
Goal: the same run as real30b (30B points, real half 10B, weather share 0.78, hourly/daily/weekly 0.5/0.4/0.1, non-weather at 4 epochs, same trainer settings and seed) with ONE change, the weather sample. So the first run's data is never modified.
- `python -m tidemark.real.estimate [--calibrate] [--workers N]`: BEFORE downloading anything: remaining download, estimated converted float32 size per variable (points per parquet byte from the counted shards), fit against the free space (keep-raw and per-variable strategies), conversion time from one calibration shard (one process, extrapolated to N workers).
- `download_chronos.py --hourly-shards 107 --skip-stats --only <8 variables>`: all shards, no per-shard counting (hours otherwise; the converter counts).
- `python -m tidemark.real.convert --only <8 variables> --shards-dir shards_full --link-rest-from shards --workers N`: parallel (identical to serial, tested), writes `shards_full/`, symlinks the 13 unchanged datasets from `shards/`, leaves `shards/` untouched.
- `python -m tidemark.real.plan --manifest shards_full/index.json --real-points 10e9 --max-epochs 4 --weather-max-share 0.78 --weather-split 0.5,0.4,0.1 --out plan_full.json`.
- Run: `REAL_SHARDS=shards_full REAL_PLAN=$REAL/plan_full.json REAL_NAME=real30b_full bash real_train_interactive.sh` (the sbatch scripts read the same variables).

## real200b and the loader I/O rework (code written 2026-10-25; NOT run on Perlmutter yet)
**real200b**: 100B real + 100B synthetic (1:1 per step on every rank, `MixConfig.real_fraction = 0.5`: 144 + 144 of the 288 rows), full pool (`shards_full`), plan `plan_200b.json`, LR 1.2e-4 / 5 % warmup / cosine to 10 % over 200B, a checkpoint every 20B (10 files, `ckpt_20B.pt` ... `ckpt_200B.pt`, copied to `REAL_COPY_TO`), the synthetic holdout every 1000 steps, exact resume. Launch: `real_train_interactive.sh --run real200b [--smoke]` (smoke = 2B points, 1B real) or `scripts/perlmutter/real_200b.sbatch` (regular QOS, 4 GPUs, 8 h, `--requeue`, `--signal=B:TERM@120`, resumes from `last.pt`). Plan: `plan --non-weather-at-cap` fills every non-weather dataset to 4 epochs first, weather takes the rest in proportion to its unique points (equal passes, about 1.2).

**Loader: what scaled with the pool.** Read from the code, not yet measured on the full pool (the benchmark below measures it): (1) the pool is 81B points = 324 GB of float32 against a page cache that cannot hold it, so a window read is a cache miss on Lustre; through `np.memmap` every miss is a page fault that also triggers the kernel's read-ahead (far more bytes than the 1 to 32 KB window) and all threads of a process contend for the mmap lock. On the 14.5B pool (58 GB) the file cache held everything, which is why it was fast; a "warm" cache on the full pool is mostly not warm. (2) `missing.i64` was memory-mapped and every window ran two binary searches on it (about log2(n) cold page faults each). Changes, none of which alters a drawn window or a value: persistent file descriptors opened with `POSIX_FADV_RANDOM` (no read-ahead), `os.preadv` straight into the destination buffer (releases the GIL), missing positions in RAM with a pointer per series (a window searches only its own series' few positions), the reads of a batch spread over `fetch_workers x threads_per_worker` threads (at most 28 per rank, 4 stay free), and PLAN / FETCH / BUILD split so the plan and the reads of batch k+1 run while the GPU works on step k. Equality with the first loader is pinned by `tests/golden/real_loader_v1.npz` (frozen from the old code before the change). The first loader stays available as `io="mmap", fetch_workers=0, prefetch=False` for the before / after benchmark.
**Benchmarks to run on the full pool** (`real_train_interactive.sh --run real200b --bench` for the single-process table; `--sweep` for the workers sweep under torchrun with 4 ranks at once; `python -m tidemark.real.bench_loader --compare-legacy ...` for before / after): results go here.
