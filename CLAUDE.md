# tideforge

The data and generator lab for Fluxion-Tide. Goal: find what the synthetic generator is missing, fix it, and prove every
fix on real data the model has never been tuned on. The model is fixed: 5M, softmax attention only (the `ab5m_cpm`
recipe). Owner: Naresh.

This is NOT the tidemark attention study. Tidemark's study rules (arms, param matching, plan-as-source-of-truth) do not
apply here. The rules below do.

## Source of truth

- `EXPERIMENTS.md`: the experiment plan and the index of every experiment. Read the relevant section before starting
  any experiment.
- `DATA_COLLECTION.md`: how the ~1B-point dev set is collected, cleaned, audited and split.
- `experiments/<ID>.md`: one log per experiment, from `experiments/TEMPLATE.md`.
- `docs/notes/`: what is already known (generator gaps, benchmark results, Timestral gap, patch probe, 5M DHF test).
  `docs/report/landscape.pdf` is the overview.
- If the plan and the code disagree, stop and ask. Change the plan only when the owner confirms; add a dated line
  under the plan section that changed.

## Rules (do not break these)

1. **Baseline B0 = the v0 generator, an exact copy of ft3_main's generator.** E0.1 proves it with a bit-exact parity test
   against `third_party/tidemark_e21cbf0/tidemark/real/ft3_core.py`. Nothing is trusted before that test passes.
2. **One switch per experiment** against B0 (Phase 3 combines winners). Every change is a switch, off by default. The
   generator version and the active switches go into the run id.
3. **Write the check before the run.** Part A of the experiment log (hypothesis, config diff, probe, dev slice, win
   rule) is written and committed before training starts, and not edited afterwards. Part B (results, verdict, failures)
   after. Failures and dead ends are logged exactly like wins.
4. **Seeds:** screen with 1 seed; anything that looks like a win is confirmed with 3 seeds before it counts.
5. **Data roles:** tune on the dev tuning set; confirm on the locked set once; fev-bench and TIME only at milestones.
   Every look at fev or TIME is a row in `LOOKS.md`. Nothing is ever selected on TIME.
6. **The dev set is split by whole dataset** (frozen hash, stratified by domain), every dataset weighted equally in
   every score, no domain above ~10% of the weight. Dev versions are frozen; compare runs only on the same version.
7. **Never fake, hardcode or hand-edit a result.** If a number looks wrong, say so.
8. **Never commit data, checkpoints, tokens or downloaded files.** Tokens live in environment variables the owner sets.
9. **Ask before adding a dependency.** Approved: torch, numpy, pandas, pyarrow, pyyaml, scipy, pytest, holidays,
   matplotlib, and what tidemark's scorers already use: fev, gluonts, huggingface_hub, timebench (the TIME repo).
   `fla` (flash-linear-attention) is only needed by the GLA / GDN arms, which tideforge does not use.
10. A negative result at 5M is logged as "negative at 5M, not tested at larger size", never as "does not work".

## How work is run

- Code is written here (Windows dev machine, PowerShell). Training and scoring run on Perlmutter (4x A100 nodes), started
  by the owner. Do NOT start training runs, sweeps or long benchmarks locally. Locally, run only fast unit tests.
- For anything that needs a GPU, write the script and give the owner the exact command; the owner runs it and pastes the
  output back.
- Use `pathlib`, no shell-specific Python, LF line endings. Slurm scripts are bash.
- At 5M, run one experiment per GPU (no DDP): 4 runs per node, 8 on two nodes.
- Before each milestone, propose the file list and the interfaces and wait for an OK. Then write the code.
- Be direct about problems: a failing test, a flaw in the plan, an estimate that looks off. Mark estimates as estimates.
- Small commits with clear messages.

## Code copied from tidemark

`third_party/tidemark_e21cbf0/` is a verbatim copy of the needed parts of the tidemark repo at commit
`e21cbf0aae4356c371882ab2c60456a3673f6be4` (2026-10-24): `tidemark/real/` (ft3_core = the Fluxion-Tide v5 round-3
model, synthetic generator, packed batcher, group builder and inference; the loss-masked decoder; sampler; store;
trainer; TIME / fev scorers incl. the groups mode in `score_groups.py`), `tidemark/eval/score_twins.py`,
`tidemark/pretrain/` (generators, packing, model, inference, probes, train) and their dependencies, plus the tidemark
tests for the real track and the Perlmutter scripts. Rules:
- **Do not edit anything under `third_party/`.** Port from it into `tideforge/`; keep a note of the source file and line
  range in the ported module's docstring.
- Import it by putting `third_party/tidemark_e21cbf0` on `sys.path` (`import tidemark.real.ft3_core as C`). ft3_core
  runs nothing on import.
- `reference/` is untracked (gitignored) and READ-ONLY. It holds:
  - `reference/tidemark/`: a full clone of the tidemark repo (added 2026-10-27 at commit e21cbf0, the same commit as
    `third_party/tidemark_e21cbf0`), for reading only. Its own `CLAUDE.md`, `TIDEMARK_PLAN.txt` and `LEAN_PLAN.md`
    describe the tidemark attention study; they do NOT apply to tideforge. Use the clone to see how Perlmutter jobs are generated
    (`tidemark/pretrain/slurm.py` writes the .sbatch files), submitted, scored and resumed (`scripts/perlmutter/`,
    `pretrain/stage_a.py`, `stage_b.py`, `submit_b.py`, `eval/score_twins.py`), and tidemark's own CLAUDE.md conventions.
  - the owner's local-only sources: `tide_v5_ft3.py`, `tide_v5_pretrain_1T.py`, `time_fev_eval.py`, and the 5M scripts
    `ab5m_cpm_vs_dhf.py`, `eval_ab5m_time_fev.py` when added. The tide_v5 files start training when imported: read
    them, never import or run them.
- Never import from `reference/`. Code that runs imports only `tideforge` and `third_party/tidemark_e21cbf0` (the pinned
  copy the parity test depends on). If `reference/tidemark` and `third_party` differ, `third_party` wins for anything
  that runs; port newer code from `reference/` only when the owner asks, and record the tidemark commit it came from.
- tideforge's Perlmutter launchers should follow tidemark's pattern (a Python generator writes the .sbatch files; edit
  the generator, not the files), with one experiment per GPU at 5M instead of DDP.

## Facts already established (see docs/notes for detail)

- ft3_main (29M, synthetic only): fev 43.78 groups / 42.09 hidden; TIME MASE 0.680, CRPS 0.571 (groups). fev leader
  TimeRSI-1.5 51.8. Our 5M CPM at 100B: fev 38.8, TIME 0.735 (Timestral-5M: 46.2 / 0.652).
- Covariate tasks hold 53-69% of small models' fev gap to the leader; only 7.4% of our training lanes are
  covariate-driven targets.
- `ft3_core.forecast_left_padded` deletes NaNs (`y = y[np.isfinite(y)]`), which shifts the seasonal phase after a gap.
  The scorers fill history gaps one season back first (`fill_season`); E0.3 checks how much NaN still reaches inference.
- `score_groups.py` lists three assumptions about the owner's original Colab scoring (non-numeric covariate encoding,
  covariate order when a group exceeds 8 lanes, forecast_groups vs forecast_groups_multi). E0.2 settles them.
- ft3_main's training config is not in the repo. Generator measurements so far assume CONFIG defaults plus `yearly` and
  `cov_r3` on. E0.1 needs the real one: `torch.load("ft3_main.pt", map_location="cpu", weights_only=False)["cfg"]`.

## Inputs still needed from the owner

1. ft3_main's `cfg` as `configs/ft3_main_cfg.json` (for E0.1).
2. The 5M scripts `ab5m_cpm_vs_dhf.py` and `eval_ab5m_time_fev.py` (they define the 5M recipe; not in tidemark's repo).
3. Paths on Perlmutter: ft3_main.pt, the evaluation root, the persistent data directory for the dev set.
4. Wall time of an `ab5m_cpm` 100B run (replaces the throughput estimate).
5. The local-only sources for `reference/`: `tide_v5_ft3.py`, `tide_v5_pretrain_1T.py`, `time_fev_eval.py` (not in
   tidemark's git, so not in the clone). Only needed if a question cannot be answered from `third_party/`.

## Layout

```
tideforge/                package (ported and new code)
  gen/                    generator v0 (port of ft3_core's generator) + switches + new families
  probes/                 one probe per gap (P1-P10)
  devdata/                dev-set sources, loaders, ingest, audit, covariate tasks, split, build
  diagnose/               fixable error, repair attribution, feature rules, coverage map, realism classifier
  train/                  5M recipe, run configs, launchers
  eval/                   dev scorer, fev / TIME wrappers (groups mode), LOOKS logging, NaN variants and audit
configs/                  run inputs that are small and not secret (ft3_main_cfg.json from the owner)
scripts/perlmutter/       bash: env.sh and the launchers (scoring, audits); training launchers are generated (see above)
third_party/tidemark_e21cbf0/   frozen, importable copy of the needed tidemark modules (read-only)
reference/                untracked, read-only: full tidemark clone + owner's local-only sources (never imported)
.claude/settings.json     blocks edits to third_party/ and reference/
catalog/                  dataset catalog, exclusions, benchmark lists, catalog builders, verify_urls.py
tools/                    one-off analysis scripts (generator audit, fev covariate split)
experiments/              one log per experiment + TEMPLATE.md
docs/notes/               prior results and analyses
docs/report/              landscape report (pdf, tex, figures)
EXPERIMENTS.md  DATA_COLLECTION.md  LOOKS.md
data/  runs/              never committed
```

## Build order

T3 requirements already fixed (owner 2026-10-28; the batcher port must meet them):
- NaN anywhere in a generated block (G6 switches write NaN at H8) becomes value 0 / mask 0 in the batcher's input AND is
  masked out of the loss: a CPM span can cover a gap inside the context, so a NaN can be a target. ft3_core's next_batch
  today copies NaN into X with mask True.
- G1.2: tideforge/gen/ext/g1_lead_noise.lead_noise on the known lanes, from the start of the CPM-hidden span, for cells that
  have one (configs/exp/G1.2.yaml, `train: lead_noise`). G3.3: `train: long_row_frac`.
- G4.3: break placement moves to crop time: place breaks relative to the cropped context (last 10-30% of it), not the
  block. The generator-side `breaks_late` (46.3% of breaks in the last 10-30% of the real context under ft3_main's layout
  sampler, v0 15.9%; owner rule was >= 50%) stays as the fallback.

T0 scaffold: package skeleton, pyproject, v0 generator port + parity test (E0.1), scorer wrapper reproducing ft3_main
(E0.2), NaN check (E0.3). T1 dev data v1 (E0.4) and the public-model sanity check (E0.5). T2 probe suite (E0.6). T3 5M
recipe port, baseline seeds and controls (E0.7-E0.9). Then Phase 1 diagnosis and Phase 2 gaps in the order in
EXPERIMENTS.md section 8.
