# Perlmutter runbook

Written for: the owner, running tideforge on Perlmutter. This is every Perlmutter step from all coding blocks, in one order,
with the exact command, where it runs, the expected time, what to paste back and what it unblocks. It replaces the
"First-Perlmutter checklist" that was in CLAUDE.md. Times are **estimates** unless a step says measured.

Nothing on Perlmutter writes to the repo. Every `--looks` appends to `$TF_ROOT/looks_pending.md`; paste those rows back, and
they go into the tracked LOOKS.md in a commit from the dev machine.

Where a step runs:
- **login**: a login node (network, no GPU; keep it to light CPU work).
- **GPU (interactive)**: first `salloc --nodes 1 --qos interactive --time <T> --constraint gpu --gpus <n> --account=$NERSC_ACCOUNT`.
- **batch**: `cd $TF_ROOT/logs && sbatch <script>`. The output is `$TF_ROOT/logs/<job>-<id>.out`.

`S=$SCRATCH/tideforge/repo/scripts/perlmutter`. Every step starts with `source $S/env.sh` (`R` is the evaluation root,
set by `$S/_evalroot.sh`). A step marked ∥ can run in
parallel with the steps around it; "needs" lists the steps that must be done first.

---

## A. Setup

**A1. Environment and clone** (login, ~10 min; needs nothing)
```
# once: ~/.tideforge_env exports TF_DATA (CFS, not $SCRATCH), TM_EVAL, FT3_MAIN_CKPT, NERSC_ACCOUNT
# git credentials: TF_GIT_URL (deploy key, preferred), or `gh auth login`, or `read -rs GH_TOKEN && export GH_TOKEN`
bash $S/setup.sh            # first time: clone; later: pull. Installs tideforge --user --no-deps -e, runs the CPU tests
```
- **Paste back:** the last 20 lines (SETUP OK and the test summary).
- **Unblocks:** everything.
- **Later:** rerun `bash $S/setup.sh` (a pull) before each session; it never writes the token to disk.

**A2. statsforecast band check** (login ∥, ~5 min; needs A1)
```
python -m pip install --user statsforecast        # the test-only extra `test-sf` (owner 2026-10-27); never a core dependency
python -m pytest -q tests/eval/test_devmetrics.py -k naive
```
- **Paste back:** the pytest summary.
- **Unblocks:** trust in the dev scorer's seasonal-naive band (E0.4).

## B. GPU tests

**B1. Generator, T3 and M1 GPU tests** (GPU interactive 1 GPU, 02:00:00 asked, ~20 min; needs A1)
```
python -m pytest -q -m gpu tests/test_gen_parity.py tests/test_gen_switches.py tests/test_gen_gpu.py
bash $S/t3.sh gputest                                   # T3 GPU tests + the generator parity ones
python -m pytest -q -m gpu tests/model/test_m1_gpu.py   # M1: firewall exact under CUDA SDPA + bf16, forecast_groups on CUDA
```
- **Paste back:** the three summaries (any failure in full).
- **Unblocks:** B2, C, E.

**B2. E0.2 smoke rerun** (GPU interactive 1 GPU ∥ with B1, 03:00:00 asked, ~30 min; needs A1)
```
bash $S/e02_score.sh --smoke        # 3 fev tasks, 2 TIME datasets, then per-task differences vs tidemark's board
```
- **Why:** the scorers now load checkpoints through `tideforge.model.load`, not tidemark's `score_ft3.load_ft3` (M1,
  2026-10-29). The smoke must still show max rel diff 0 against tidemark.
- **Paste back:** the difference table.
- **Unblocks:** D2 (E0.3 + E0.10) and every later fev / TIME score.

## C. Generator pools and throughput

**C1. Generator throughput, loops** (GPU interactive 1 GPU, ~20 min; needs B1)
```
python tools/gen_throughput.py --device cuda --loops      # tar / garch_jump loop cost vs B0
```
- **Paste back:** the table.
- **Unblocks:** the decision whether G9.2 / G10.1 need offline pools (C3).

**C2. ODE pool** (GPU interactive 1 GPU ∥ with C1, ~30-60 min; needs B1)
```
python tools/build_ode_pool.py --out $TF_DATA/pools --name ode_v1      # done 2026-11; --workers defaults to every CPU
```
- **Paste back:** the manifest line (name, sha256, counts).
- **Unblocks:** G9.1a/b/c (switch `ode`, pool ode_v1) in the Phase 2 screen.

**C3. Loop pools, required** (C1: tar 6-41x, garch_jump 1.6-9x, limit 0.10x; G9.2 / G10.1 name them) (GPU interactive 1 GPU, ~30 min each; needs C1)
```
python tools/build_loop_pool.py --family tar        --out $TF_DATA/pools --name tar_v1          # --workers: default all CPUs
python tools/build_loop_pool.py --family garch_jump --out $TF_DATA/pools --name garch_jump_v1   # manifest.json lands next to the pool: paste it back
python tools/gen_throughput.py --device cuda --loops --pool-dir $TF_DATA/pools
```
- **Paste back:** the manifests and the throughput table.
- **Unblocks:** G9.2a/b and G10.1a/b at full speed.

## D. Scoring reproductions

**D1. E0.6 probe sets** (login, CPU, once, ~15 min; needs A1). Built 2026-10-31 (registry committed).
```
bash $S/e06_probes.sh build     # $TF_DATA/probes/P*/v*/   (done)
python -m tideforge.probes migrate --root $TF_DATA/probes     # once: adds content_sha256 to the 2026-10-31 build's meta.json
```
- **Paste back:** the 10 migrate lines (content sha256 per probe; they complete tideforge/probes/registry.json).
- **Unblocks:** probes in every `finish` (results rows) and D3.

**D2. E0.3 + E0.10 job** (batch, 1 regular node, ~3.5 h makespan / 13.5 GPU-h; needs B2)
```
sbatch $S/e03_score.sbatch
bash $S/e03_nan.sh table        # after it finishes
```
- **Paste back:** the E0.3 table and the E0.10 comparisons.
- **Unblocks:** the NaN-variant decision and the E0.10 covariate encoding.

**D3. E0.6 probes on ft3_main** (GPU interactive 1 GPU ∥, 01:00:00 asked, < 15 min; needs D1)
```
bash $S/e06_probes.sh run && bash $S/e06_probes.sh report
```
- **Paste back:** the report.
- **Unblocks:** ft3_main's probe starting values.

## E. Public models (E0.5)

**E1. Installs and weights** (login ∥ with D, ~30-45 min; needs A1)
```
bash $S/e05_public.sh install core
bash $S/e05_public.sh install toto2
bash $S/e05_public.sh install tirex2
bash $S/e05_public.sh weights
```
- **Paste back:** the `pip check` lines, and the pinned-vs-live table from `weights`.
- **Unblocks:** E2.

**E2. GPU check** (GPU interactive 1 GPU, ~20 min; needs E1)
```
bash $S/e05_public.sh check
```
- **Stop:** TiRex must build its xlstm CUDA kernels. If they don't build, the step stops; there is no fallback. Tell me.
- **Paste back:** one OK line per model.
- **Unblocks:** E3.

**E3. fev wrapper check** (batch, 1 regular node, ~0.9 h / 3.3 GPU-h; needs E2)
```
sbatch $S/e05_public.sbatch
bash $S/e05_public.sh table --looks      # after it finishes: one LOOKS row per model
```
- **Paste back:** the table.
- **Unblocks:** public-model dev scoring (F6).

## F. Dev data: batches 1 + 2 -> dev-v1

dev-v1 = batch 1 + batch 2 (owner 2026-10-29). Long fetches: run on a login node under `tmux`, or as an `xfer` job.

**F1. Batch 1** (probe / fetch on a login node; **ingest always on a CPU node**: per-file sources are streamed (D057 2.1 GB),
but a single-file source still loads whole (D078 19.6 GB, D320 29 GB peak); needs A1). Keys: ~/.tideforge_env exports DATAGOVIN_API_KEY and USDA_MARS_KEY (owner), never committed.
```
python -m tideforge.devdata.ingest probe  --ids batch1
python -m tideforge.devdata.ingest fetch  --ids batch1 --raw $TF_DATA/raw          # resumes; 404s are skipped (--retry-missing)
python -m tideforge.devdata.ingest ingest --ids batch1 --raw $TF_DATA/raw --out $TF_DATA/staging
python -m tideforge.devdata.ingest status --ids batch1 --raw $TF_DATA/raw --out $TF_DATA/staging
```
- **Paste back:** the probe output for the `adapter:pending` entries (D038 PVDAQ, D084 EEA), and the status table.
- **Unblocks:** F3, F4.

**F2. Batch 2 probe** (login ∥ with F1, ~20 min; needs A1). The list is in docs/BATCH2_PROBES.md.
```
python -m tideforge.devdata.ingest probe --ids batch2,batch2_standby > $TF_ROOT/logs/batch2_probe.txt 2>&1
```
- **By hand, in a browser:**
  - Grid-India's terms of use: is automated access allowed? If not, D014 is dropped and D403 replaces it.
  - Zillow's research data terms (the page refused the dev machine).
  - The licence on ANAC VRA's dados.gov.br record.
  - BWDF's data licence (journal supplement).
  - NESO's open-data licence text.
- **Paste back:** the probe file and the five licence texts.
- **Unblocks:** I write the batch-2 adapters / table specs (round 1 done 2026-10-31: docs/BATCH2_PROBES.md, "Probe round
  1"). Round 2 probes the fixed URLs (command there).

**F2a. Owner's PC downloads** (Windows, the repo checkout; for D230 and D373, which NERSC cannot reach). In PowerShell, from
the repo root:
```
$RAW = "D:\tf_raw"            # any local folder
New-Item -ItemType Directory -Force "$RAW\it_fuel_prices\files", "$RAW\agmarknet\files" | Out-Null
# browser downloads into those files\ folders (what to take: the D230 / D373 notes in tideforge/devdata/sources.yaml)
python -m tideforge.devdata.ingest register --ids D230,D373 --raw $RAW     # writes fetch_log.jsonl (bytes, sha256)
python -m tideforge.devdata.ingest verify   --ids D230,D373 --raw $RAW
scp -r "$RAW\it_fuel_prices" "$RAW\agmarknet" <user>@dtn01.nersc.gov:<absolute TF_DATA>/raw/
```
- Then on Perlmutter: `python -m tideforge.devdata.ingest verify --ids D230,D373 --raw $TF_DATA/raw` (every sha256 of the
  log must match after the upload).
- `--raw` folders have the same layout on both machines (`<raw>/<slug>/files/` + `fetch_log.jsonl`), and fetched file names
  come from the URL, so a scripted `ingest fetch` on the PC uploads the same way.
- **Paste back:** both `verify` lines.

**F2b. Fetch and ingest** (login; after the batch-2 loaders are written):
```
python -m tideforge.devdata.ingest fetch  --ids batch2 --raw $TF_DATA/raw
python -m tideforge.devdata.ingest ingest --ids batch2 --raw $TF_DATA/raw --out $TF_DATA/staging
```
- D279, D202 and D110 use the `index` frequency (owner 2026-10-31): season 1, horizons 10 / 25 / 50.

**F3. Covariate tasks** (login, ~1-2 h; needs F1 and F2b, and the census files for EIA)
```
python -m tideforge.devdata.covariate_tasks stations --raw $TF_DATA/raw --census-gazetteer G --census-popest P
python -m tideforge.devdata.covariate_tasks build --raw $TF_DATA/raw --staging $TF_DATA/staging
```
- **Paste back:** one meta line per task (25), with `cov_nan_share`.

**F4. Benchmark audit** (login or a CPU job, ~1-3 h; needs F1 and F2b)
```
huggingface-cli download Salesforce/GiftEval --repo-type dataset --local-dir $TF_DATA/bench/gifteval   # verify the repo id
python -m tideforge.devdata.audit names       --ids batch1,batch2
python -m tideforge.devdata.audit fingerprint --tm-eval $TM_EVAL --gift $TF_DATA/bench/gifteval --out $TF_DATA/audit/bench
python -m tideforge.devdata.audit check       --db $TF_DATA/audit/bench --staging $TF_DATA/staging --ids batch1,batch2
python -m tideforge.devdata.audit devdev      --staging $TF_DATA/staging --ids batch1,batch2 --out $TF_DATA/audit/devdev
```
- **Look at:** D170, D229 / D230, D162, D376, D394, and D401 against fev entsoe. If D401 overlaps, it is dropped.
- **Paste back:** the flagged datasets.
- **Unblocks:** catalog/audit_decisions.csv (written here), then F5.

**F5. Split check and dev-v1 build** (login, ~20-40 min; needs F3 and F4)
```
python -m tideforge.devdata.split check
python -m tideforge.devdata.build --staging $TF_DATA/staging --out $TF_DATA/dev --version dev-v1
```
- **Paste back:** the build summary, including the `<domain>_misc` pool lines (datasets with < 30 windows) and the manifest
  totals.
- **Unblocks:** every dev score: finish, the board, E0.5's ranks, G2.4.

**F6. E0.5 on the dev set** (batch or interactive, ~2-4 h for 9 models; needs E3 and F5)
```
for m in chronos2 timesfm25 toto2_4m toto2_22m toto2_2p5b tirex tirex2 timestral5m timestral28m; do
  python -m tideforge.eval.devscore score --dev $TF_DATA/dev --version dev-v1 --model $m --out $TF_ROOT/dev
  python -m tideforge.eval.devscore score --dev $TF_DATA/dev --version dev-v1 --model $m --cov --out $TF_ROOT/dev
done
python -m tideforge.eval.devscore ranks --runs $TF_ROOT/dev/dev-v1/tuning_fast/* --ref <fev / TIME reference csv>
```
- `ranks` prints the correlation twice: with all datasets and without the pretrain-flagged ones.
- **Paste back:** both blocks of `ranks`.

## G. Training (T3)

**G1. T3 smoke and E0.9** (GPU interactive 1 GPU, 02:00:00 asked, ~1.5 h; needs B1)
```
bash $S/t3.sh smoke E0.7 G4.1 G5.2a G1.2
bash $S/t3.sh e09
python -m tideforge.train.throughput report --det $TF_ROOT/e09/<det run> --default $TF_ROOT/e09/<default run>
```
- **Paste back:** the smoke losses and the E0.9 report (points/s, hours per 100B).
- **Unblocks:** I fill in `TRAIN_PTS_PER_S` in tideforge/slurm.py, which replaces every GPU-h estimate with the measured
  rate.

**G2. E0.7 + E0.8 node job** (batch, 1 regular node, 12:00:00 limit, ~1.2 h per run at the estimated rate; needs G1)
```
sbatch $S/t3_e07.sbatch            # E0.7 seeds 0-2 + E0.8; each run ends with finish (dev + probes + results row)
```
- **Paste back:** the four results rows (`$TF_ROOT/results/results.csv`).
- **Unblocks:** B0 for the board, and G3.
- **If dev-v1 or the probes weren't ready yet**, rerun finish later for each seed:
  `python -m tideforge.train.finish --ckpt <run>/final.pt`.

**G3. E0.7 noise file** (login, ~5 min; needs G2 with dev and probes scored)
```
python -m tideforge.eval.noise --out $TF_ROOT/noise/e07.json \
  --dev-runs     <seed0>/dev/dev-v1/tuning_fast/m5 <seed1>/... <seed2>/... \
  --dev-cov-runs <seed0>/dev/dev-v1/tuning_cov/m5  <seed1>/... <seed2>/... \
  --probe-csvs   <seed0>/probes/results.csv <seed1>/... <seed2>/...
```
- `--dev-cov-runs` is needed: G1.13's win rule uses E0.7's range on the dev-covariate score.
- **Paste back:** the printed ranges.
- **Unblocks:** every board verdict (rule 5).

**G4. Phase 0 milestone** (batch / GPU, ~1 h; needs G2 and B2): seed 0 is scored once on fev (groups ON and hidden) and
TIME, against 38.8 / 0.735.
```
source $S/_evalroot.sh && tf_link_eval           # R = $TF_ROOT/eval_t0 (fev / TIME data linked); tag e07s0 is new there
python -m tideforge.eval.score score --root $R --ckpt e07s0=<seed0>/final.pt
python -m tideforge.eval.score table --root $R --names e07s0base --base e07s0base --looks --experiment E0.7 --why "Phase 0 milestone"
```
- **Paste back:** the table (one LOOKS row).

## H. Phase 2 screen

**H1. Screen campaign: interactive sessions are the default** (regular QOS waits 5-30 h; interactive starts within a
minute, up to 4 nodes x 4 h). 66.3 GPU-h busy: about 4 sessions of 4 nodes. Needs G2, the three pools (C2, C3) and F5.
```
tmux new -s screen
salloc --nodes 4 --qos interactive --time 04:00:00 --constraint gpu --gpus-per-node 4 --cpus-per-task 32 --account=$NERSC_ACCOUNT
source $S/env.sh && cd $TF_REPO
python -m tideforge.campaign.runner run phase2_screen       # foreground; rerun in a new salloc until 0 left
```
- One trainer + finish per GPU; the next run starts when a GPU frees up, and only if it can finish (0.91 h + finish + 5 min).
- Restart-safe: done runs are skipped, killed runs resume from last.pt. Do not change the commit between sessions: a new
  commit starts killed runs from zero. Logs: `$TF_ROOT/logs/phase2_screen_<exp>_s<seed>.log`.
- **Paste back:** the `session over` line.
- **Fallback, sbatch:** `python -m tideforge.campaign.runner plan phase2_screen --results $TF_ROOT/results/results.csv --write`,
  then the printed sbatch lines.

**H2. Board** (login, ~2 min; needs H1 runs finished and G3)
```
python -m tideforge.campaign.board --results $TF_ROOT/results/results.csv --noise $TF_ROOT/noise/e07.json \
  --version dev-v1 --out $TF_ROOT/board
```
- **Paste back:** board.md.
- Running `--write` happens in the repo: I run it on the pasted files, so the Part B tables and the index rows are generated
  from the results, never typed.

**H3. Screen session 2** (after session 1, 2026-11-01; pull once first, then freeze the commit). The 48 finished runs are not
retrained: `done` ignores the code version.
1. Remaining training: G8.1, G5.4a/b, G9.2a/b, G10.1a/b (7 runs, about 6.4 GPU-h busy, about 2 h on 1 node; estimate):
```
python -m tideforge.campaign.runner plan phase2_screen --results $TF_ROOT/results/results.csv     # expect 7 pending
salloc --nodes 2 --qos interactive --time 02:00:00 --constraint gpu --gpus-per-node 4 --cpus-per-task 32 --account=$NERSC_ACCOUNT
source $S/env.sh && cd $TF_REPO
python -m tideforge.campaign.runner run phase2_screen
```
2. P6 under nofill_mask, 7 checkpoints (GPU, same salloc; about 2 min each, estimate; repeatable):
```
python -m tideforge.campaign.rescore --results $TF_ROOT/results/results.csv --root $TF_DATA/probes \
  --exps E0.7,G6.1,G6.2,G6.3,G6.4 --probes P6 --nan-variant nofill_mask
```
3. M1 check, read from the real checkpoints (login):
```
python -c "
import csv, os, torch
for r in csv.DictReader(open(os.environ['TF_ROOT'] + '/results/results.csv')):
    if r['exp'] in ('G1.13a', 'E0.7') and r['ckpt'].endswith('final.pt'):
        print(r['exp'], r['seed'], torch.load(r['ckpt'], map_location='cpu', weights_only=False)['cfg'].get('known_attn'))
"
```
4. Board and P1 per cell (login, no GPU):
```
python -m tideforge.campaign.board --results $TF_ROOT/results/results.csv --noise $TF_ROOT/noise/e07.json \
  --version dev-v1 --out $TF_ROOT/board --cells P1:G1.13a --cells P1:G1.8
```
- **Paste back:** the M1 lines, board.md, cells_P1_G1.13a.md, cells_P1_G1.8.md.

**H4. C2 and the per-task report** (2026-11-01). C2 goes through `runner run`, not sbatch: its comparison is B0 seeds 6, 7
from that launch. Measure tv_season first: it is C2's one generator cost not yet known at GPU.
```
salloc --nodes 1 --qos interactive --time 02:00:00 --constraint gpu --gpus-per-node 4 --cpus-per-task 32 --account=$NERSC_ACCOUNT
source $S/env.sh && cd $TF_REPO
python tools/gen_throughput.py --device cuda --exp G5.1                                       # paste back the table
python -m tideforge.campaign.runner plan c2 --results $TF_ROOT/results/results.csv            # expect 2 pending
python -m tideforge.campaign.runner run c2
python -m tideforge.eval.where meta --root $TF_ROOT/eval_t0                                   # once, CPU
python -m tideforge.eval.where report --root $TF_ROOT/eval_t0 --a <tag> --b0 <B0 tags> --out $TF_ROOT/where_<tag>.md
```
Each fev report is a LOOKS row in $TF_ROOT/looks_pending.md.

**H5. R1: dropout 0 and m30** (2026-11-02). Check on one interactive GPU (under 10 min), then the node job.
```
salloc --nodes 1 --qos interactive --time 00:30:00 --constraint gpu --gpus 1 --account=$NERSC_ACCOUNT
source $S/env.sh && cd $TF_REPO && python tools/m30_check.py
python -m tideforge.train.r1job --time <max(30M projected, 2 x 5M projected) x 1.3, HH:MM:SS> --micro-rows <from the check>
cd $TF_ROOT/logs && sbatch $TF_ROOT/jobs/r1.sbatch
```
The check fails on any architecture difference from ft3_main other than dropout. fev rows go to $TF_ROOT/looks_pending.md.

**H6. Flip averaging, R1.2 and R1.4** (2026-11-02). Inference only, about twice a plain score (estimate). One interactive 4-GPU node.
```
salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account=$NERSC_ACCOUNT
source $S/env.sh && cd $TF_REPO && git pull
run() { CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=32 python -m tideforge.eval.score score --root $TF_ROOT/eval_t0 --flip-avg \
  --ckpt "$3=$(ls -t $TF_ROOT/runs/$2-*/final.pt | head -1)" --modes on --skip-time > $TF_ROOT/logs/flip_$2.log 2>&1 && \
  echo "| $(date +%F) | $2 | $3flipbaseon_s0 (groups ON, flip-avg) | fev-bench | flip averaging, one counted look | scored; skill from the board |" >> $TF_ROOT/looks_pending.md; }
run 0 R1.4 r14 & run 1 R1.2 r12 & wait
python -m tideforge.eval.score board --root $TF_ROOT/eval_t0 | tail -n 20
```

**H7. Adapter: AD.1-AD.4** (2026-11-03). All times are estimates; `tools/ad_check.py` replaces them with a measured rate.
Needs R1.2's final.pt, and R1.4's for AD.3 / AD.4. First build P11 (once, CPU, about a minute): `python -m tideforge.probes build --root $TF_DATA/probes --probes P11`,
paste back the line for registry.json.
```
# GPU check, 10 min
salloc --nodes 1 --qos interactive --time 00:30:00 --constraint gpu --gpus 1 --account=$NERSC_ACCOUNT
source $S/env.sh && cd $TF_REPO && git pull && python tools/ad_check.py --exp AD.2
exit
# pilot: two 1B runs side by side, then P1 + P11 on both and on R1.2
python -m tideforge.train.adjob write pilot --time <1B projected x 1.5 + 01:00:00>
cd $TF_ROOT/logs && sbatch $TF_ROOT/jobs/ad_pilot.sbatch
python -m tideforge.train.adjob choose --probes $TF_ROOT/probes_ad --tags lr3e-4 lr1e-3
# AD.1, AD.2: four 5B runs (one per GPU), finish, then fev on all four
python -m tideforge.train.adjob write ad12 --time <5B projected x 1.5 + finish + fev> --lr <chosen>
cd $TF_ROOT/logs && sbatch $TF_ROOT/jobs/ad_ad12.sbatch
# the column (one report per AD run vs its backbone)
python -m tideforge.eval.where report --root $TF_ROOT/eval_t0 --a ad2s0baseon_s0 --b0 r12baseon_s0 --out $TF_ROOT/where_ad2s0.md
# only if AD.2 beat AD.1
python -m tideforge.train.adjob write ad34 --time <...> --lr <chosen>
cd $TF_ROOT/logs && sbatch $TF_ROOT/jobs/ad_ad34.sbatch
```
fev rows go to $TF_ROOT/looks_pending.md; a job fails if the non-covariate tasks differ from the backbone.

---

## Dependencies at a glance

```
A1 ─┬─ A2 ∥
    ├─ B1 ─┬─ C1 ─ C3
    │      ├─ C2 ∥
    │      └─ G1 ─ G2 ─┬─ G3 ─ H2
    │                  ├─ G4 (needs B2)
    │                  └─ H1 (needs C2, F5) ─ H2
    ├─ B2 ∥ ─ D2
    ├─ D1 ─ D3 ∥
    ├─ E1 ─ E2 ─ E3 ─ F6 (needs F5)
    └─ F1 ∥ F2 ─ F2b ─┬─ F3 ─┐
                      └─ F4 ─┴─ F5 ─ F6, finish reruns, H1
```
Longest path: data (F1 / F2 fetches, the batch-2 adapters I write from the probe, then the audit and build), not the
GPU steps. Start F1 and F2 on day one, in parallel with B / C / G.
