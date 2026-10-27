# Milestone 4 on Perlmutter: step by step

This trains the 8 twins: softmax, GLA, SWA8 and SWA16, with seeds 0 and 1. Each twin runs as
its own job on one full node (4 × A100) with DDP. Everything lives under
`$SCRATCH/tidemark`:

```
$SCRATCH/tidemark/
  repo/                    this git repo (code + these scripts)
  logs/<job>-<id>.out      one log per job (a requeued job appends to the same file)
  checks/                  gla_kernel.json, bench_rows.json, bench.json, pilot_choice.json
  runs/<arm>_s<seed>_lr<lr>_<id>/
      last.pt              resumable checkpoint (deleted when the twin finishes)
      best.pt              best holdout so far (weights + optimizer)
      milestones/ckpt_25B.pt, ckpt_50B.pt     weights only
      final.pt             weights only: the model that gets scored
      summary.json         final holdout, history, rollbacks, settings
      pilot.json           pilot runs only: holdout at 5B and the number of rollbacks
```

## 0. Log in

```
ssh <your NERSC user>@perlmutter.nersc.gov
```

NERSC requires multi-factor authentication: you enter your password followed by the one-time
code. NERSC's `sshproxy` tool gives you a key that lasts 24 hours, which saves typing. See
docs.nersc.gov, "Connecting to NERSC".

## 1. One-time setup (login node, a few minutes)

```
export GH_TOKEN=<a GitHub token that can read nareshmeena12/tidemark>
mkdir -p $SCRATCH/tidemark && cd $SCRATCH/tidemark
git clone https://$GH_TOKEN@github.com/nareshmeena12/tidemark.git repo
bash repo/scripts/perlmutter/setup.sh
```

`setup.sh` does four things:

- loads NERSC's `pytorch` module;
- installs fla 0.5.2 and the repo with `pip install --user`;
- checks that the module's torch is at least 2.7;
- runs the CPU unit tests.

It must end with `SETUP OK`. If the torch check fails, run `module avail pytorch` and put a
newer version in `env.sh` (for example `module load pytorch/2.8.0`).

To update the code later, run `git -C $SCRATCH/tidemark/repo pull` (set `GH_TOKEN` first if it
asks for a password).

## 2. Submit everything

```
bash $SCRATCH/tidemark/repo/scripts/perlmutter/submit_all.sh
```

This submits a chain. Each job starts only after the one before it **ended successfully**:

| Job | QOS / time | What it does | Must show |
|---|---|---|---|
| `tm-debug` | debug, 30 min | Twin unit tests on the GPU: packed vs alone for every arm, the GLA leak test, G1 counts, DDP smoke (gloo and NCCL). Then the GLA kernel check at series lengths 1–256, and the rows-per-GPU probe. | `DEBUG JOB PASSED` |
| `tm-bench` | regular, 1 h | About 200 real training steps per arm | a table of points/s per GPU and per node, peak memory, node type (40 or 80 GB), hours for 100B |
| `tm-pilot` | regular, 3 h | Each arm at 1.5e-4 and 3e-4, seed 0, paused at 5B points; then the rate choice | one line per arm: `chosen` (with the rate) or `blocked` |
| `tm-<arm>-s<seed>` ×8 | regular, 3 h each | One twin, 100B points | `Done: ...` at the end |

If a job fails, the jobs after it never start. They stay in the queue with the reason
`DependencyNeverSatisfied`; remove them with `scancel` and paste me the failed job's log.
After a fix, restart from the failed step, for example:
`bash submit_all.sh --from pilot` (the choices are `debug`, `bench`, `pilot` and `twins`).

**The rate choice.** The pilot writes `checks/pilot_choice.json`. You can edit it before the
twins start; the twin jobs read it when they begin. An arm whose two pilot rates both had a
rollback is `blocked`: its twin jobs print that and stop, and the other arms go on.

**The 3-hour limit.** About two minutes before the limit, Slurm signals the job. The run saves
`last.pt` and the job puts itself back in the queue (`scontrol requeue`). When it starts again
it continues from `last.pt`. The log shows `SIGUSR1: time limit in 2 minutes` and then
`restart count 1`. Nothing needs to be done by hand.

## 3. Watch the jobs

```
squeue --me                          # your jobs: PD = waiting, R = running, plus the reason
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed,NodeList   # after it ended
tail -f $SCRATCH/tidemark/logs/tm-gla-s0-<jobid>.out                       # live log
scontrol show job <jobid>            # details (expected start time, dependency)
scancel <jobid>                      # cancel one job;   scancel -u $USER   cancels all
cat $SCRATCH/tidemark/checks/pilot_choice.json
```

A twin log prints a line every 50 steps (loss, LR, points done, points/s, ETA, skipped
batches) and a holdout block every 1,000 steps. Lines starting with `!!` are rollbacks.
Paste me:

- the bench table;
- the pilot's choice lines;
- for each twin: the first ~20 lines, any `!!` line and the end of the log.

## 4. Copy the finished checkpoints off $SCRATCH

**$SCRATCH is purged**: files not accessed for 8 weeks are deleted. The scored files are each
twin's `final.pt` (weights only, about 120 MB) plus its `summary.json`. Keep them in two
places.

**a) On NERSC: the Community File System (CFS) of project m4392.** CFS is not purged.

```
mkdir -p /global/cfs/cdirs/m4392/tidemark/twins
cd $SCRATCH/tidemark/runs
for d in */; do
  [ -f "$d/final.pt" ] || continue
  mkdir -p /global/cfs/cdirs/m4392/tidemark/twins/$d
  cp "$d"/final.pt "$d"/summary.json /global/cfs/cdirs/m4392/tidemark/twins/$d
  cp -r "$d"/milestones /global/cfs/cdirs/m4392/tidemark/twins/$d 2>/dev/null || true
done
cp -r $SCRATCH/tidemark/checks /global/cfs/cdirs/m4392/tidemark/
```

**b) To your own machine.** Run this on YOUR machine, not on Perlmutter. It uses NERSC's data
transfer nodes (check the host name on docs.nersc.gov, "Data Transfer Nodes", if it has
changed):

```
rsync -av --progress <user>@dtn01.nersc.gov:/global/cfs/cdirs/m4392/tidemark/ ./tidemark_m4/
```

For many or large files, NERSC recommends Globus (its Perlmutter endpoint sees $SCRATCH, CFS
and $HOME). Then put `tidemark_m4/` on Google Drive for the Colab scoring notebook
(`notebooks/m4_twins.ipynb`).

## What is fixed, and where it is decided

- **Arms:** they differ only in the time attention. G1 counts: softmax 29,248,800 (SWA8 and
  SWA16 the same), GLA +75,648 (+0.26%, gate and norm only).
- **Batch:** 512 rows × 8,192 points per step for every arm, cut into micro-batches if fewer
  rows fit on a GPU. The data depends only on (seed, rank, step).
- **Checks before training:** GLA's packed rows run fla's varlen kernel with `cu_seqlens`. The
  debug job checks that kernel at every series length from 1 to 256. If `chunk` fails,
  training uses `fused_recurrent`, and the bench shows what that costs.
- **Rate rule:** lowest holdout at 5B among the runs without a rollback; a tie at 4 decimals
  goes to the lower rate.

## 5. Score the twins (one interactive node, one session)

Do this once all 8 `final.pt` files exist under `$SCRATCH/tidemark/runs/`. Scoring uses your
pipeline in `reference/time_fev_eval.py` (TIME and fev-bench). `tidemark/eval/score_twins.py`
ports its scoring as written. It only changes where things are read from and written to, because
the compute node has no network.

### 5a. On a LOGIN node, once: install and download (needs the network)

```
module load pytorch
pip install --user fev datasets            # the [fev] extra
pip install --user gluonts python-dotenv toolz pyyaml   # needed by the TIME code (your script installs these too)
cd $SCRATCH/tidemark/repo && git pull
export ROOT=$SCRATCH/tidemark
nohup python -m tidemark.eval.score_twins fetch > $ROOT/logs/score_fetch.log 2>&1 &
tail -f $ROOT/logs/score_fetch.log
```

`fetch` downloads everything into `$ROOT/eval/data` and `$ROOT/eval/hf`:

- the TIME repo and the TIME datasets (Real-TSF/TIME);
- the official TIME results for seasonal naive and the reference models (Real-TSF/TIME-Output);
- the fev-bench tasks, their data (each task is loaded once, so the Hugging Face cache is
  complete), and the reference results (Chronos-2, Timestral-5M/28M, Toto-2.0-4m/22m, TiRex).

It can take a while and the sizes are not known to me. If it is interrupted, run it again:
downloads are cached. It ends with `fetched: 100 fev tasks ...`. If it lists packages as
missing, it prints the exact `pip install --user ...` line.

### 5b. Get a node and run the smoke test (under 5 minutes)

Start `tmux` on the login node first, so a dropped SSH connection does not kill the session.

```
tmux new -s score
salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account=m4392_g
bash $SCRATCH/tidemark/repo/scripts/perlmutter/score_interactive.sh --smoke
```

The interactive queue allows at most 4 hours (NERSC docs). The smoke test uses 3 fev tasks (the
smallest) and 2 TIME datasets, scored for one twin (`gla_s0`, the one that uses fla), plus seasonal
naive on the same 3 tasks. It then runs the board. Its outputs go to `$ROOT/eval_smoke`, so the
real caches are not touched. It must end with `SCORING DONE (SMOKE)`.

### 5c. The real run

```
bash $SCRATCH/tidemark/repo/scripts/perlmutter/score_interactive.sh
```

It starts, all at once:

- seasonal naive on the CPU, about 100 workers, cached in `$ROOT/eval/fev_naive.csv` (skipped if
  that file exists; TIME's seasonal naive is the official one and is not recomputed);
- four scoring processes, one per GPU: GPU 0 softmax s0/s1, GPU 1 GLA s0/s1, GPU 2 SWA16 s0/s1,
  GPU 3 SWA8 s0/s1. Each does the TIME datasets, then the fev-bench tasks, with context 8192, univariate,
  9 quantiles, `fill_season`, batch 512.

Then it waits and runs the board step. A progress line per log is printed every 2 minutes. Logs:
`$ROOT/logs/score_naive.log`, `score_gpu0.log` … `score_gpu3.log`, `score_board.log`. Every result is
one file per (model, task), so after a failure or the 4-hour limit you run the same script again
and only the missing work is redone. I have not measured the run time. The smoke test prints
seconds per task, and the naive step is expected to be the slow one.

Optional: `LOCAL_WINDOWS="8 16" bash .../score_interactive.sh` also scores the softmax twins with
attention limited to 8 and 16 patches at inference (the local-window test). It runs on GPU 0 and
makes that GPU about 3 times longer.

### 5d. What the board prints, and where

`$ROOT/eval/board/` holds:

- `board.txt` (everything below);
- `time_board.csv`: TIME leaderboard, MASE and CRPS normalised to seasonal naive, plus ranks;
- `fev_board.csv`: fev-bench leaderboard (skill score) with Chronos-2, Timestral, Toto and TiRex;
- `head_to_head.csv`: GLA, SWA16 and SWA8 vs softmax (seeds averaged), and seed 0 vs seed 1 per
  arm as the noise floor, each with a bootstrap 95% CI over tasks, split by horizon. TIME splits
  by its short/medium/long terms; fev-bench splits by horizon terciles (the bounds are printed);
- `scores_long.csv`: one row per (model, benchmark, metric, task), the table for
  `python -m tidemark.eval.gaps --scores scores_long.csv --benchmark fev`.

Copy `board/` (small) to CFS and to your machine as in section 4.


## 6. Zero-shot twins on the supervised datasets (Milestone 5, part B)

Scores the 8 twins, with no training, on ETTh1, ETTh2, ETTm1, ETTm2 and Weather, horizons 96 and
720, on exactly the test windows of the supervised models (MSE / MAE on the standardised scale,
per-step vectors, same row format as M5). Two contexts per window: exactly 336 points, and all
history before the target up to 8192 points. `tidemark/eval/zeroshot.py` reuses `load_twin` and
`forecast_left_padded` from the twin scoring in section 5.

```
# login node, once (needs the network; the five CSV files are about 20 MB)
export TIDEMARK_DATA=$SCRATCH/tidemark/data
python -m tidemark.supervised.data fetch --data-dir $TIDEMARK_DATA
# interactive node (tmux first, as in 5b)
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account=m4392_g
bash $SCRATCH/tidemark/repo/scripts/perlmutter/zeroshot_interactive.sh --smoke   # 1 twin (gla_s0), ETTh1 h96, 64 windows
bash $SCRATCH/tidemark/repo/scripts/perlmutter/zeroshot_interactive.sh           # the real run
```

One process per GPU (GPU 0 softmax s0/s1, 1 GLA, 2 SWA16, 3 SWA8). Rows go to
`$ROOT/results/M5Z/` (next to the M5 rows; one per twin, dataset, horizon and context; rerun the same command to resume), then
the report: for each dataset and horizon, the supervised softmax / GLA (flatten head, lookback 336)
next to the twins at both contexts. Logs: `$ROOT/logs/zeroshot_*.log`. To compare on Colab, copy
`$ROOT/results/M5Z` to `$R/M5Z` on Drive and run the last cell of `notebooks/m5_supervised.ipynb`.
`LOCAL_WINDOWS="8 16"` also scores the softmax twins limited to 8 and 16 patches (GPU 0, about 3 times longer).

With the full context at horizon 720 the sequence is 8192 + 736 points, slightly beyond the 8192 the
twins were trained on (a mild length extrapolation, stated in the report).


## 7. Milestone 5 (supervised benchmarks) on Perlmutter, in one interactive session

The same grid as the Colab notebook (`notebooks/m5_supervised.ipynb`), run on one node with 4 A100s.
Everything is under `$SCRATCH/tidemark`: data `data/`, results `results/` (rows, weights in
`results/weights/M5/`, figures), logs `logs/` and `results/logs/`. Setup (section 1) must be done.

### 7a. Fetch the five datasets (LOGIN node, needs the network, about 20 MB)

```
source $SCRATCH/tidemark/repo/scripts/perlmutter/env.sh     # loads the pytorch module, goes to the repo
git pull --ff-only
mkdir -p $SCRATCH/tidemark/data
TIDEMARK_DATA=$SCRATCH/tidemark/data python -m tidemark.supervised.data fetch
ls -l $SCRATCH/tidemark/data                                # ETTh1/2, ETTm1/2 and weather.csv
```

It prints the sha256 of each file; keep those lines. The compute node never needs the network.

### 7b. Get the node (start `tmux` first, so a dropped SSH connection does not end the session)

```
tmux new -s m5
salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account=m4392_g
```

### 7c. Smoke test (pick K from data)

```
bash $SCRATCH/tidemark/repo/scripts/perlmutter/m5_interactive.sh --smoke
```

It runs every code path (36) for about 200 steps on GPU 0 (pause, resume, weights, per-step vectors),
then runs 1, 2 and 4 shards per GPU on all four GPUs at once (the copies start together on a
barrier) and prints the total steps/s of each. It ends with the T0-lite estimate and a line
`RECOMMENDED K = ...`: the smallest shards-per-GPU within 5% of the best throughput. It must end with
`M5 SMOKE PASSED`. The rows go to `results/SMOKE5`, and the estimate in the next step reads them.

### 7d. The full run

```
K=<the recommended K> bash $SCRATCH/tidemark/repo/scripts/perlmutter/m5_interactive.sh
```

(`K` defaults to 4, which is 16 shards.) It prints the estimate, then runs the stages in order: sweep,
edge, seed0, seed1, a6, then the figures and the full report. In each stage the work is split into
4 x K shards, shard k on GPU (k-1) % 4, and the script waits for all of them before the next stage. A
stage that takes more than 30% over its estimate stops the chain (rule 2). Each run writes its own
results file, so shards never write to the same file; the reports merge the files.

Resume: if the session ends mid-stage, get a node again and run the same command. Finished runs are
skipped, a sweep run paused at epoch 4 continues from its checkpoint.

Watch from a second tmux window on the same node (`srun --overlap --jobid <id> --pty bash`, or `ssh` to
the node name shown by `squeue --me`):

```
nvidia-smi                                                  # 4 GPUs busy
tail -f $SCRATCH/tidemark/results/logs/M5-seed0-shard1of16.log
tail -f $SCRATCH/tidemark/logs/m5_run.log
```

### 7e. Part B in the same session, after M5

```
bash $SCRATCH/tidemark/repo/scripts/perlmutter/zeroshot_interactive.sh --smoke
bash $SCRATCH/tidemark/repo/scripts/perlmutter/zeroshot_interactive.sh
```

(section 6; it needs the 8 twins under `$SCRATCH/tidemark/runs`). Its rows go to `results/M5Z`, next to the M5
rows, and the report compares the twins with the supervised softmax and GLA.

### 7f. Copy results and weights to ~/tidemark_backup

```
mkdir -p ~/tidemark_backup/m5
rsync -a --info=progress2 --exclude 'ckpt/' $SCRATCH/tidemark/results/ ~/tidemark_backup/m5/results/
rsync -a $SCRATCH/tidemark/logs/ ~/tidemark_backup/m5/logs/
rsync -a $SCRATCH/tidemark/data/ ~/tidemark_backup/m5/data/
du -sh ~/tidemark_backup/m5/*
myquota                                                     # $HOME has a quota; the weights are about 1 GB (estimate)
```

`results/` holds the rows (`M5/`, `M5.sweep/`, `M5.A6/`, `M5Z/`, `SMOKE5/`), the best-validation weights
(`weights/M5/`) and the figures. Checkpoints (`ckpt/`) are only needed to resume and are left out. To
use the results in the Colab notebook, copy this `results/` folder to `MyDrive/tidemark/results`. To your own
machine (check the data-transfer host name on docs.nersc.gov): `rsync -av <user>@dtn01.nersc.gov:tidemark_backup/m5/ ./tidemark_m5/`.

### 6b. Naive baselines and the pipeline audit (CPU only, no GPU, no training)

`AUDIT_M5_PARTB.md` explains what was checked. Two commands, to run on a login node or in the session. The
repo is private and its `origin` has no token, so pull with the token in the URL once (type the token
without echo; it will not stay in your shell history):

```
read -s GH_TOKEN; export GH_TOKEN
cd $SCRATCH/tidemark/repo
git pull --ff-only https://$GH_TOKEN@github.com/nareshmeena12/tidemark.git main
unset GH_TOKEN
source scripts/perlmutter/env.sh
export TIDEMARK_DATA=$SCRATCH/tidemark/data            # the five CSV files from section 7a / 6

# 1. Proof: the naive baselines through the supervised and the zero-shot pipelines must agree (about 2 minutes)
python -m tidemark.eval.zeroshot audit-naive           # must end: AUDIT PASSED

# 2. Add the two baseline rows (last value, seasonal naive) to the part B results, then print the table
python -m tidemark.eval.zeroshot naive                 # writes results/M5Z/*.csv for the baselines only
python -m tidemark.eval.zeroshot report                # a "naive baselines (same windows)" line per dataset and horizon
```

Seasonal naive uses the dataset's main period: 24 steps for the hourly ETT files, 96 for the 15-minute ones,
144 for Weather. Existing result files are not touched; `naive` only adds new rows and skips ones that exist.
`zeroshot_interactive.sh` now runs the `naive` step before its report.

## Stage A: five new 30M arms (gdn, glas1, glas2, glas3, swa16s1)

Arms are one per-layer pattern each (`tidemark/pretrain/arms.py`; `gla_s1` etc. are accepted as aliases of
`glas1` ...). Layers are counted from the input side:

| arm | pattern |
|---|---|
| gdn | Gated DeltaNet in all 6 |
| glas1 | gla,gla,gla,gla,gla,softmax |
| glas2 | gla,gla,softmax,gla,gla,softmax |
| glas3 | gla,softmax,gla,softmax,gla,softmax |
| swa16s1 | swa16 x5, softmax |

The twin jobs are the same as the four existing ones (`twin_<arm>_s<seed>.sbatch`, 1 node x 4 A100, DDP, 3 h,
requeue, same data stream), because the data order depends on the number of ranks; running them 4 x K to a GPU
would change the data order against the existing twins. The K-shards-per-GPU recipe stays what the supervised
chains use. No training is started by any command below except the last.

```
read -s GH_TOKEN; export GH_TOKEN
cd $SCRATCH/tidemark/repo
git pull --ff-only https://$GH_TOKEN@github.com/nareshmeena12/tidemark.git main
unset GH_TOKEN
source scripts/perlmutter/env.sh
python -m tidemark.pretrain.stage_a params          # CPU: exact counts and the G1 check; must end PASSED for 9 arms
python -m pytest -q tests/test_stage_a.py tests/test_overrun.py tests/test_twin_gpu.py   # GPU tests need a GPU node

# 1. Smoke on ONE interactive GPU (about 1-2 h for 5 arms + the softmax / gla timing runs; 1 GPU, 5 arms in turn)
salloc --nodes 1 --qos interactive --time 02:00:00 --constraint gpu --gpus 1 --account=m4392_g
bash $SCRATCH/tidemark/repo/scripts/perlmutter/stage_a_interactive.sh
#    ends with a table (loss, save / resume, probe, fev / TIME, M5Z per arm) and the cost estimate; SMOKE PASSED
#    ARMS="gdn" bash ...   one arm;   NO_EVAL=1 bash ...   training checks only

# 2. Fill in predictions/stage_a_predictions.md, commit and tag it BEFORE the launch.

# 3. Launch (login node; 10 jobs, 3 h each with requeue; needs checks/{gla_kernel,bench_rows,pilot_choice}.json)
bash $SCRATCH/tidemark/repo/scripts/perlmutter/submit_stage_a.sh            # or: ... submit_stage_a.sh glas1 gdn
```

The estimate (`python -m tidemark.pretrain.stage_a estimate --root $SCRATCH/tidemark`) reads
`stage_a/smoke/timing.json` written by the smoke and scales the node-hours of the existing twins
(`checks/bench.json`, else 0.8 for softmax) by the measured one-GPU speed ratio; it also prints the same ratio's
prediction for the GLA twin next to the 1.0 h measured, so a smoke that does not scale like the 4-GPU runs shows up.
The Gated DeltaNet layer runs fla's chunked delta-rule kernel (training has no fused-recurrent fallback) plus three
short convolutions, so expect it to be the slowest arm; the number is the smoke's, not a guess.

Open points before the launch (also in the plan's decision log): the rate of the new arms is their donor arm's
pilot rate; packed series shorter than 64 tokens run through the chunk kernel for GDN (GPU test in the smoke).

## Stage B: scale (5M, 100M, optional 200M)

Sizes (nominal; exact counts in the G1 table): 5M = d256 x 4 layers, 100M = d768 x 12, optional 200M = d1024 x 14 at
200B points (flag `--with-200m`, softmax and gla only). Arms per size: softmax, gla, gdn and ONE hybrid taken from the
Stage A gate (`$ROOT/stage_a/stage_a_gate.json`, written by `python -m tidemark.analysis.gate`) or `--hybrid`; without
either the hybrid jobs are skipped and the rest runs. Same DDP job as the 30M twins (4 ranks). Run folders:
`$ROOT/runs` (30M, unchanged), `$ROOT/runs_5m`, `$ROOT/runs_100m`, `$ROOT/runs_200m`; point the harnesses at one with `--runs`.

```
read -s GH_TOKEN; export GH_TOKEN
cd $SCRATCH/tidemark/repo
git pull --ff-only https://$GH_TOKEN@github.com/nareshmeena12/tidemark.git main
unset GH_TOKEN
salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account=m4392_g -C "gpu&hbm80g"
bash $SCRATCH/tidemark/repo/scripts/perlmutter/stage_b_interactive.sh            # params, GPU tests, memfit, smoke, estimate
#   add --with-200m to include the optional model; --no-smoke to stop after the memory fit
# fill in predictions/stage_b_predictions.md, commit and tag it, then from a login node:
bash $SCRATCH/tidemark/repo/scripts/perlmutter/submit_stage_b.sh --dry-run       # shows every sbatch line
bash $SCRATCH/tidemark/repo/scripts/perlmutter/submit_stage_b.sh                 # 5M and 100M: pilot job, then the twins (afterok)
#   --seeds 2   --hybrid glas2   --with-200m   --sizes 100m
# when the pilots are done: cat $ROOT/checks/pilot_choice_{5m,100m}.json   (look for edge=true warnings)
python -m tidemark.eval.probe_rows --root $ROOT      # after the twins: probe + holdout CSV per checkpoint (new files)
```

### Stage A gate, Stage B evaluation, paper tables and figures

```
# after Stage A twins are evaluated (score_twins board, zeroshot run, probe_rows) and predictions/stage_a_predictions.md
# has its ```gate block filled and committed BEFORE the first Stage A checkpoint existed:
python -m tidemark.analysis.gate --root $ROOT            # prints the commit of the predictions file and any order warning
                                                         # writes $ROOT/stage_a/stage_a_gate.json (winner or null); refuses on blank fields
# Stage B per size, after its twins finish:
bash $SCRATCH/tidemark/repo/scripts/perlmutter/stage_b_eval.sh 5m      # and 100m (and 200m)
# the paper (CPU; reads result files only):
python -m tidemark.analysis.paper_tables  --root $ROOT --out $ROOT/paper/tables
python -m tidemark.analysis.paper_figures --root $ROOT --out $ROOT/paper/figures
```

### Cost study for the new arms (GPU, nothing else running)

`E2` (speed and memory vs context) and `E3` (streaming) now also contain Gated DeltaNet (`gdn`, both model sizes) and
the four hybrids (`glas1`, `glas2`, `glas3`, `swa16s1`, twin size only) at the SAME context lengths, batches, warm-up and
timing as the existing arms; new bench ids, so existing rows are untouched and finished ones are skipped. The new E2 rows
also carry `prefill_b1_tokens_per_s`, `prefill_b64_tokens_per_s` (one full causal pass over the context, patch tokens per
second), `decode_state_bytes` (bytes of KV cache / ring buffer / recurrent state per sequence at that context, bf16) and
`train_peak_mb` (peak training memory, as before). The old arms get the same new columns when re-run (they are not re-run
automatically).

```
python -m tidemark.run --exp E2 --results-dir $R --dry-run        # lists done / todo
python -m tidemark.run --exp E2 --results-dir $R                  # alone on the GPU (timing experiment); likewise E3
python -m tidemark.run --exp FIG --results-dir $R                 # existing figures; paper_figures.fig_f_cost draws the new one
```

## Running Stage A / B inside an interactive allocation (no batch queue)

`run_stage_a_interactive.sh` and `run_stage_b_interactive.sh` start `python -m tidemark.pretrain.allocation`: one twin per
node (4 GPUs, 4 ranks, the batch job's torchrun, started with srun), the longest estimate first, a twin only if time left -
reserve >= 1.25 x its estimate + 10 minutes (else it is listed NOT STARTED), a status line every 60 s, and a summary with
the command to continue. Stage B also queues the rate pilots of a size first and writes its pilot choice before that size's
twins start. Estimates come from the smoke timing, else from finished twins (`$ROOT/allocation/durations.json`), else the
30M figures with a loud warning. Checkpoints are written every 10 minutes at most; at the end (or SIGTERM, or when the
reserve is reached) the running twins are asked to save and stop, and resume from `last.pt` next time.

```
read -s GH_TOKEN; export GH_TOKEN
cd $SCRATCH/tidemark/repo && git pull --ff-only https://$GH_TOKEN@github.com/nareshmeena12/tidemark.git main; unset GH_TOKEN
salloc --nodes 4 --qos interactive --time 04:00:00 --constraint gpu --gpus-per-node 4 --account=m4392_g     # Stage B: add -C "gpu&hbm80g"
bash $SCRATCH/tidemark/repo/scripts/perlmutter/run_stage_a_interactive.sh --dry-run    # schedule from nodes, time left, estimates
bash $SCRATCH/tidemark/repo/scripts/perlmutter/run_stage_a_interactive.sh              # run;   --reserve-min 15, --seeds 1, ...
bash $SCRATCH/tidemark/repo/scripts/perlmutter/run_stage_b_interactive.sh --dry-run     # then without --dry-run; --hybrid, --with-200m
```

### 30M gdn rate pilot (Stage A)

```
salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account=m4392_g     # 4 GPUs: the twins' 4 ranks
PILOT=1 bash $SCRATCH/tidemark/repo/scripts/perlmutter/stage_a_interactive.sh      # smoke, estimate, then the pilot
PILOT=only bash $SCRATCH/tidemark/repo/scripts/perlmutter/stage_a_interactive.sh   # the pilot alone (resumable)
cat $ROOT/checks/pilot_choice.json | python -m json.tool | grep -A12 '"gdn"'        # chosen lr, edge flag / warning
```
Then submit or run the Stage A twins as usual; the gdn twins pick up their own rate.

### Selftest of an allocation (no training, no checkpoints)

```
bash $SCRATCH/tidemark/repo/scripts/perlmutter/run_stage_a_interactive.sh --selftest     # or run_stage_b_interactive.sh
```
For every node it runs `srun` with the options a real twin uses and prints the hostname, the GPUs visible (nvidia-smi and torch;
4 expected) and whether the stop-file path is writable; then it reads the time left from `squeue` and prints it parsed, and
exits (0 = PASSED, 1 = FAILED). It needs no input files.

### Kernel head-dimension guard

fla's GLA / GDN kernels need head key and value dims >= 16 on some Triton builds (Perlmutter's asserts `K >= 16` in the backward
pass; Colab's older build ran width 64 = head key dim 8). A trainer therefore probes: a layer below 16 is allowed only if the
layer's own kernels run forward + backward once on the GPU; otherwise `KernelDimError` names the layer and the fix.
`TIDEMARK_ALLOW_SMALL_KERNEL_DIMS=1` skips the check. Configs below 16: width 64 only (C3, C1/C1a at 64, C1P at 64, the P0
pilot width); C1L, C4, C2, S1 are at width 256 (32 / 64).

### gdn pilot: one more rate (6e-4), and C2H

```
# the gdn winner (3e-4) was the top of its grid: ONE more pilot rate, then re-choose over all four (4 GPUs)
PILOT=only PILOT_EXTRA=6e-4 bash $SCRATCH/tidemark/repo/scripts/perlmutter/stage_a_interactive.sh
# or inside the 4-node allocation, queued first (only the gdn twins wait for it):
bash $SCRATCH/tidemark/repo/scripts/perlmutter/run_stage_a_interactive.sh --gdn-extra-rate 6e-4
```
```
# Colab / lite chain (after C2's swa32>gla line has its rate): C2H alone, or with the others
python -m tidemark.run --exp C2H --phase all --results-dir /content/drive/MyDrive/tidemark/results
python -m tidemark.run --group C2,C2H
python -m tidemark.run --exp C2H --report
```

### Stage B twins as batch jobs on the preempt QOS (long 100M twins, survives preemption)

One sbatch job per twin (one node, 4 GPUs, job name `stageB_<size>_<arm>_s<seed>`), `-q preempt` by default, walltime 12:00:00
(100M, 200M) or 2:00:00 (5M), `-C "gpu&hbm80g"` for 100M / 200M and plain `gpu` for 5M. Pilots and the smoke stay interactive.
A twin with `final.pt` is skipped, one with `last.pt` is submitted again and resumes the same run, one already pending or running
is skipped, one whose `pilot_choice_<size>.json` entry is missing or blocked is REFUSED, the hybrid comes from the gate or
`--hybrid` (otherwise only the hybrid jobs are left out).

```
read -s GH_TOKEN; export GH_TOKEN
cd $SCRATCH/tidemark/repo && git pull --ff-only https://$GH_TOKEN@github.com/nareshmeena12/tidemark.git main; unset GH_TOKEN
bash scripts/perlmutter/submit_stage_b.sh --batch --dry-run                       # plan, charges, every sbatch line; submits nothing
bash scripts/perlmutter/submit_stage_b.sh --batch                                  # submit (preempt, 12 h for 100M, 2 h for 5M)
bash scripts/perlmutter/submit_stage_b.sh --batch --qos regular --time 12:00:00    # same twins on regular
bash scripts/perlmutter/stage_b_status.sh                                          # the table;  watch -n 60 bash .../stage_b_status.sh
squeue --me -o "%.10i %.28j %.9T %.10M %r"                                         # jobs;  tail -f $SCRATCH/tidemark/logs/stageB_100m_gla_s0-<jobid>.out
scancel -n stageB_100m_gla_s0                                                      # cancel one twin (its last.pt and run folder stay; submit again to resume)
scontrol requeue <jobid>                                                           # restart a job by hand (it resumes from last.pt)
```
Preemption, as assumed here (VERIFY in the NERSC docs for the `preempt` QOS: https://docs.nersc.gov/jobs/policy/ and the
"Preempt" / "Running Jobs" pages): Slurm sends SIGTERM and kills the job after a grace period; the grace period is taken to be SHORT
(possibly under a minute). The job script traps SIGTERM (and the one sent 120 s before the walltime by `--signal=B:TERM@120`), touches
the stop file, the training processes also catch SIGTERM themselves (`TIDEMARK_SIGTERM_STOP=1`), finish the current step, write `last.pt`
and exit 3; the script then runs `scontrol requeue` and exits 0 (exit 143 if Slurm refuses because it is requeueing the job itself).
`last.pt` is also written every 10 minutes, so a kill with no grace costs at most that. torchrun's agent gives its workers about 30 s
after SIGTERM before SIGKILL (its default): a 100M save should fit, but check the first preemption in the log ("training exited with
code 3") and that `ckpt age` in the status table stays under 10 minutes. To verify: the preempt grace time, whether the batch script
and the job steps both get SIGTERM, whether `RequeueExit` is set, and the charge rule (1x for the first 2 h, 0.25x after).

### Scoring the Stage A twins (one interactive node, 4 x A100)

```
salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account=m4392_g
bash $SCRATCH/tidemark/repo/scripts/perlmutter/stage_a_eval.sh --dry-run     # plan, GPU split, done / to do, the commands
bash $SCRATCH/tidemark/repo/scripts/perlmutter/stage_a_eval.sh --smoke       # one twin, 3 fev tasks (outputs in eval_smoke)
bash $SCRATCH/tidemark/repo/scripts/perlmutter/stage_a_eval.sh               # the real run; rerun after any interruption
python -m tidemark.analysis.gate --root $ROOT                                # then: the gate (its block must be filled and committed first)
```
The twins are read from `$ROOT/runs/<arm>_s<seed>_lr*/final.pt` (where train.py, the in-allocation runner and the batch jobs write
30M-size twins), passed as `--runs` to score_twins, zeroshot and probe_rows. The M4 reference twins are not rescored; the board
reads every arm already in `$ROOT/eval` and `$ROOT/results`.

### Stage B reruns at another start rate (separate arms)

Used when a pilot-chosen rate turned out unstable in the full run (rollbacks). The originals are never touched; a rerun is its own
arm: trained in `$ROOT/runs_100m_rerun/<arm>_s0_lr<rate>_<id>/`, evaluated as `<arm>_lr24_s0` (rate in units of 1e-5) in
`$ROOT/size_100m_rerun`. Same trainer, 4 ranks per twin, 100B-point budget, checkpointing, resume and rollback rules; `summary.json`
and `rerun.json` carry `rerun: true` and `rerun_of` (the original run folder and its start rate).

```
salloc --nodes 2 --qos interactive --time 04:00:00 --constraint gpu --gpus-per-node 4 --account=m4392_g -C "gpu&hbm80g"
bash $SCRATCH/tidemark/repo/scripts/perlmutter/run_stage_b_interactive.sh --rerun softmax:0.00024,swa16s1:0.00024 --size 100m --seed 0 --dry-run
bash $SCRATCH/tidemark/repo/scripts/perlmutter/run_stage_b_interactive.sh --rerun softmax:0.00024,swa16s1:0.00024 --size 100m --seed 0
# evaluation (one interactive node with 4 GPUs; JOBID from squeue --me), then the combined board:
bash $SCRATCH/tidemark/repo/scripts/perlmutter/eval_spread_rerun.sh JOBID 100m
bash $SCRATCH/tidemark/repo/scripts/perlmutter/stage_b_eval_rerun.sh 100m --board       # originals + reruns, in $ROOT/size_100m_combined
```
The combined board links the files of `size_100m` and `size_100m_rerun` into `size_100m_combined` (neither is changed) and prints the
fev-bench / TIME boards, the single-seed paired head-to-heads and the arm table. Batch jobs (`--batch`) are not extended to reruns.
