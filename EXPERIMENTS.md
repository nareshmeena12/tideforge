# tideforge: experiment plan

Goal: find out what the synthetic generator is missing, fix it, and prove every fix on real data the model has never been
tuned on. Model: 5M softmax (the `ab5m_cpm` recipe: d256, 4 layers, 4 heads x 64, d_ff 1024, patch 32, group attention,
9 quantiles, arcsinh). Every training run: 100B points unless stated.

Status of this file: plan agreed on 2026-10-27. Each experiment gets its own log in `experiments/<ID>.md` (template:
`experiments/TEMPLATE.md`). The index at the bottom gets one line per experiment.

---

## 1. Ground rules

1. **Baseline B0** = the v0 generator, an exact copy of ft3_main's generator (E0.1 proves it). Every experiment changes
   ONE switch against B0, unless it says otherwise (Phase 3 combines).
2. **Every change is a switch**, off by default, and the generator version + switches go into the run id.
3. **Write the check before the run.** In the log, before training: hypothesis, config diff, the probe and dev slice it
   should move, and the win rule. Results and the verdict are added after. A failure is written up the same way.
4. **Seeds.** Screen with 1 seed. Anything that looks like a win is re-run with 3 seeds before it counts.
5. **Win rule (default).** Beats B0 by more than the seed noise from E0.7 on the fast dev score, AND improves its own
   probe and target slice, AND no domain slice gets worse by more than 1%. An experiment may set a stricter rule in its log.
6. **Add vs replace.** A new family takes share from the others. Each winner is tested both ways: added on top
   (renormalised) and replacing the share of the family it is closest to.
7. **Data roles.** Tune on the dev tuning set. Confirm on the locked set once. fev-bench and TIME only at milestones;
   every look is counted in `LOOKS.md`. Nothing is ever selected on TIME.
8. **Negative results are results.** A gap that fails at 5M is logged as "negative at 5M, not tested at larger size",
   not as "does not work". This matters most for G3 (seen-once history) and G1.11 (many covariates).

- 2026-10-28 (owner): rule 6's "the family it is closest to" is read as "the closest families with enough weight". A replace
  arm takes the add arm's share from the listed families in order (each emptied before the next); when they hold less, it
  takes all of them and is smaller than the add arm (said in its config). Pairs (B0 shares): amfm (4%) and beats (3%) <- tsi
  (6.33%); irregular_cycles (4%) <- long_cycle (9.49%); demand (5%) <- count + intermittent (3.16%, all); ode (5%) <-
  damped_osc + nonlinear_ar (1.27%) then 3.73% from gp_kernel (the two are too small); tar (3%) <- regime_switch +
  nonlinear_ar (1.90%, all); garch_jump (4%) <- stochastic_vol (2.53%, all).

## 2. Scores every run reports

- Fast dev score: ~15k windows, geometric mean of scaled error per dataset, every dataset weighted equally.
- Probe table: one probe per gap (listed under each gap).
- Slice table: by domain, frequency, history length, horizon, covariates yes/no.
- Full dev score (~75k windows, ~1B points stored) for promotion decisions; locked set once per decision.
- fev skill / TIME MASE + CRPS only at milestones (end of Phase 0, after G1, end of Phase 3).

---

## 3. Phase 0: foundations (nothing is trusted before these pass)

| ID | What | Pass rule |
|---|---|---|
| E0.1 | Generator parity: v0 reproduces `tidemark/real/ft3_core.py` output bit for bit for the same seed and config (needs ft3_main's real cfg). | exact match |
| E0.2 | Scorer: groups mode added; reproduces ft3_main fev 43.78 (groups) / 42.09 (hidden) and TIME MASE 0.680 / CRPS 0.571. | within 0.2 fev, 0.002 TIME |
| E0.3 | Inference NaN fix: keep missing positions masked instead of deleting them (`forecast_left_padded` drops NaNs today). Score ft3_main before and after; no training. | record the change |
| E0.4 | Dev set v1 (see DATA_COLLECTION.md): ~1B points, tuning and locked split, plus ~25 covariate tasks built by joining sources. | manifest complete, audit passes |
| E0.5 | Dev set sanity: score Chronos-2, TimesFM-2.5, Toto-2.0, Timestral (public weights) on the dev set. Their ranking should broadly match fev / TIME. | rank correlation reported; investigate if low |
| E0.6 | Probe suite: one probe per gap (G1-G10), run on ft3_main for starting values. | all probes run |
| E0.7 | 5M baseline B0, 3 seeds. Seed spread = the noise floor for every later decision. | spread recorded |
| E0.8 | Negative control: 10% of the mix replaced by white noise. Measures what dilution alone costs at 5M. | effect recorded |
| E0.9 | Throughput: points/s per GPU at 5M, and whether generation is the bottleneck; same for each new family as it lands. | numbers recorded |
| E0.10 | Inference-only covariate tricks on ft3_main: categorical target encoding, covariate ordering. | record the change |

2026-10-27 (owner confirmed): the E0.2 gate is six numbers: groups ON fev 43.78, TIME MASE 0.680 / CRPS 0.571; groups OFF
fev 42.09, TIME MASE 0.687 / CRPS 0.575 (tolerance 0.2 fev, 0.002 TIME), as in score_groups.GATE.

- 2026-10-29 (owner, T3): the 5M recipe is configs/train/ab5m_cpm.yaml (every value with its source). 1024 rows per step
  (8.30M real points, about 12k steps per 100B) is INFERRED (ft3 / ft3_main's value), not known from ab5m, whose scripts are
  lost. Weight decay 0.01 (what ft3's trainer used: AdamW's default), betas (0.9, 0.95), LR peak / 25 -> 5e-4 over 2% of the
  points -> cosine to 0; validation every 500 steps. 5,447,968 parameters (asserted). Generation runs in deterministic mode on
  the main thread between steps, never overlapping model kernels; E0.9 measures what not overlapping costs.
- 2026-10-29 (owner, T3): E0.7's node job runs the 3 B0 seeds and E0.8 seed 0 (white_noise switch, add mode 10%), one per
  GPU. The synthetic holdout loss is a training monitor only (it is B0's own generator); decisions use dev and probes.
  G4.3 is now crop-time breaks (train.crop_breaks); the generator-side breaks_late version is G4.3f (fallback).
- 2026-10-29 (owner): E0.10 runs in e03_score.sbatch's queue with E0.3 (one look event, one LOOKS row per variant): target_enc,
  order_corr (known-before-past kept, sorted within each role) and both; groups ON, fev covariate tasks only
  (tideforge/eval/cov_variants.py). Order matters only where the lane cap drops covariates: 17 of fev-bench's 42 covariate
  tasks (tasks.yaml), so order_corr is kept.
- 2026-10-29 (owner, E0.5): public models = Chronos-2, TimesFM-2.5, Toto-2.0 4m / 22m / 2.5B, TiRex, TiRex-2, Timestral-5M /
  -28M (plus ft3_main and seasonal naive). Pins in tideforge/eval/public_models/__init__.py (package versions or git commits,
  HF weight commits); the `public` extra shares the core env, Toto-2.0 and TiRex-2 have their own venvs ($TF_ROOT/venvs).
  TimesFM-2.5 is scored in hidden mode only (covariates only through XReg / jax, not installed); Toto-2.0 and TiRex have no
  covariate input (hidden only); devscore --cov records "covariates: none (hidden only)" for them. Toto-2.0 runs in fp32 (its
  card states fp32 weights, nothing about bf16). TiRex must run its xlstm CUDA kernels: no torch fallback (the adapter
  refuses). Wrapper check: 5 fixed fev tasks (epf_de, uci_air_quality_1D, ETT_1W, australian_tourism, ercot_1D), every task
  within 2% of the model's published test_error in fev's results CSV; one LOOKS row per model. The published rows used fev
  0.8.0 / 0.9.0 / 0.11.0.dev0 depending on the model: the check records the installed version next to the published one.
- 2026-10-27 (owner): B0 = ft3_main's generator mix (configs/ft3_main_cfg.json, `v0-b0-af75b1af`). ft3_main was fine-tuned
  from ft2_main, so B0 copies ft3's mix only, not ft2's history: experiments compare against B0, and B0 is never described as
  "ft3_main's training data".

- 2026-10-27 (owner): the E0.1 GPU rule is bit-exact parity of ft3_core and the port, both with
  torch.use_deterministic_algorithms(True) and CUBLAS_WORKSPACE_CONFIG=:4096:8 (ft3_core itself is not reproducible on CUDA in
  torch's default mode). E0.9 also measures generator throughput in deterministic vs default mode on GPU
  (tools/gen_throughput.py); if the cost is small training uses deterministic mode, otherwise the data are accepted as
  non-deterministic and E0.7's seed spread covers it. The owner decides from the numbers.

- 2026-10-27 (owner, from E0.9's generator measurement): training runs the GENERATOR in deterministic mode (0.80-0.81x of
  default-mode generator throughput), so B0 and switch runs see identical data apart from the switch; the MODEL stays in
  torch's default mode and E0.7's seed spread covers model noise. T3 measures end-to-end training throughput in both modes
  and this is revisited if the cost is above 10%.

## 4. Phase 1: diagnosis (no training; runs alongside Phase 0)

| ID | What | Output |
|---|---|---|
| D1 | Covariate split on ft3_main's own per-task fev file (tasks with vs without covariates, gap to the leader). | is our gap shaped like the small models' (53-69% from covariate tasks)? |
| D2 | Fixable error: our error minus the best reference model's, per window, on TIME (public per-window predictions) and fev. | ranked list of tasks and windows we can win back |
| D3 | Repair attribution on the worst windows: level, trend, seasonal amplitude, phase, spikes, interval width. | share of error per cause |
| D4 | Feature rules: regress fixable error on period/context, seasonal and trend strength, intermittency, late shift, covariates, frequency, length, noise. | short rule list ("we lose when ...") |
| D5 | Coverage map: real dev windows vs synthetic windows in feature space. | empty regions = missing families |
| D6 | Realism classifier: small model telling real from synthetic windows; inspect what it uses. | what the generator gets wrong |

D1-D6 can reorder Phase 2. Record any reordering in this file with the date and the reason.
- 2026-10-29 (owner, rule correction): D2-D6 run on the dev TUNING set and fev, never on TIME (D2 above said "on TIME"):
  designing generators around TIME's worst windows would be selecting on TIME. TIME stays the final check.
- 2026-10-29: D2-D6 tools in tideforge/diagnose/ (features shared with gen_audit; fixable, repair, rules, coverage, realism).
  D3 needs devscore --save-forecasts. D5 / D6 compare on the season detected from each series (synthetic rows have no
  frequency). D6 folds real windows by dataset and synthetic rows at random within each family (leave-family-out would
  measure generalisation to unseen families, not which family is easy to spot).
- 2026-10-27 (owner, from D1): G1 stays first; its scenarios in the order G1.3 load, G1.4 price, G1.5 retail (the worst
  covariate tasks are load, price and retail).

---

## 5. Phase 2: one block per generator gap

Each block lists its probe, its dev slice, and the experiments. All are single switches against B0.

### G1. Covariate realism and volume (main bet)
- Probe P1: synthetic groups where only a covariate predicts the future (one per mechanism); metric = error with groups
  vs hidden.
- Dev slice: the covariate tasks from E0.4 (load + temperature, prices + load/renewables, bikes + weather, sales +
  holidays, PV + irradiance).

| ID | Change |
|---|---|
| G1.1 | Covariate-driven targets 7% -> ~20% of lanes, with the existing switches (`cov_dominant_p`, `second_group_p`, `cov_small_group_p`, `group_kind_probs`). |
| G1.2 | Known covariates handed over with forecast error (noise that grows with lead time) vs exact. |
| G1.3 | Scenario: weather -> load (temperature with its own daily and yearly cycle; hour-dependent heating / cooling response). |
| G1.4 | Scenario: residual load -> price (convex supply curve of load minus wind and solar, plus spikes). |
| G1.5 | Scenario: retail (open flag, promotion windows with a post-promotion dip, price elasticity, typed holidays). |
| G1.6 | Scenario: PV output from irradiance (night zeros, temperature derating, inverter clipping). |
| G1.7 | Scenario: events -> traffic / footfall (event flags on top of day types). |
| G1.8 | All five scenarios together. |
| G1.9 | Causal-graph generator alone (random DAG over base-family nodes, lagged nonlinear edges; parents = covariates, descendants and siblings = decoys). |
| G1.10 | Scenarios + causal graph. |
| G1.11 | Number of covariates: realistic 1-13 vs today's 2-8 (lanes permitting). |
| G1.12 | Static / categorical covariates passed as constant known lanes (no model change). |
| G1.13 | Model M1: known covariates see the whole window, with a firewall (covariates never read targets). Then the 2x2: B0 / best G1 data / M1 / both. |
| G1.14 | G1.1 + G1.8 together (planned 2x2 with B0, G1.1, G1.8; exception to rule 1). |
- 2026-10-29 (owner): G1.13 / M1 = mechanism (b), `model: {known_attn: mask}` (tideforge/model/m1.py): known lanes' time
  attention is bidirectional within their segment, and group attention gets a firewall (target -> target / past / known;
  past -> past / known; known -> known). 0 extra parameters. Today's ft3 group attention is SYMMETRIC: covariates read
  targets, so there is no firewall; that is harmless only because every token is causal. (a), an extra cross-attention
  per layer (+1,049,600 params, +19.3%), is a later capacity experiment, run only if (b) wins. Configs: G1.13a = B0 + M1,
  G1.13b = best G1 data + M1 (G1.8 as a placeholder), and G1.13c = B0 + M1 with `known_lookahead: 32`. G1.13c runs only if
  G1.13a improves dev but not fev (the pattern of a train / inference mismatch). Details: experiments/G1.13.md.
- 2026-10-29 (owner): 17 of fev-bench's 42 covariate tasks exceed the lane cap (rohlik x4, walmart, solar x2, kdd_cup x3, m5 x3, uk_covid x4; cap 7 with one target, 4 with several), so the model never sees all their covariates. "More lanes" becomes a candidate model change (M3) after E0.10's order_corr shows how
  much choosing the kept covariates matters.

- 2026-10-28 (owner): G1.11 is 1-7 covariates (8-lane limit), P = (0.35, 0.25, 0.15, 0.10, 0.07, 0.05, 0.03); more than 7
  needs more lanes, a model change (future work).
- 2026-10-28 (owner): G1.5 uses a plain holiday flag; the holiday source is pluggable (`g1_common.HOLIDAYS`, switch param
  `holidays`) so G2's calendar engine replaces it without touching the retail code.
- 2026-10-28 (owner): G1.9 decoys = descendants, siblings and unrelated nodes; descendants are always past-only (their future
  would reveal the target's). A known sibling's future stays partly informative through the shared parent (realistic).
  Parent roles are drawn known / past 0.5 each, independent of the lag.
- 2026-10-28 (owner): resolutions sc_load 15 min / 30 min / hourly / daily 0.2 / 0.15 / 0.45 / 0.2, sc_pv the same without
  daily, sc_price 15 min / hourly 0.3 / 0.7, sc_retail and sc_events daily / weekly 0.75 / 0.25 (weekly = sums of 7 days).
  Shares: 0.15 in the single configs (G1.3-G1.7, G1.9), 0.05 each in G1.8 (14.0% of blocks), 0.04 each + graph 0.08 in
  G1.10 (15.3%).
- 2026-10-28 (owner): G1.12 dropped for now. ft3's per-lane scaling erases constants (0.3 / -0.7 / 5.0 all become exactly
  0 after `_causal_stats` + asinh), so static covariates need a model change (a static embedding, future "M2"). Not
  tested, not negative.
- 2026-10-28 (owner): G1.14 = G1.1 + G1.8 together. G1.1 (count), G1.8 (realism at the same count) and G1.14 (both) form a
  planned 2x2 with B0: an exception to rule 1 (one switch per experiment), like G1.13.

### G2. Calendar engine
- Probes P2: holiday effect with and without its flag; a moving festival; a different weekend definition.
- Dev slice: TSA, Eurocontrol, Iowa liquor, Agmarknet, Seattle library, transit ridership.
- Design rule: **diversity, not one country.** Each series draws its calendar from a wide pool: real calendars from
  ~40+ countries via the `holidays` package, covering Gregorian, Islamic (Eid moves ~11 days a year), Hindu lunisolar
  (Diwali), Chinese lunisolar (Lunar New Year) and Jewish calendars; different weekend rules (Sat-Sun, Fri-Sat, Sun only);
  regional sub-holidays; fiscal-year and school-term variants; and a share of **invented calendars** (random holidays
  with random fixed or moving rules) so no real calendar can be memorised. Effects (size, sign, run-up, rebound) are
  drawn per series, never fixed per holiday.

| ID | Change |
|---|---|
| G2.1 | Calendar shapes the target only (no flag shown). |
| G2.2 | Also shown as known covariates in 50% of rows. |
| G2.3 | Exposure share 25% / 75%. |
| G2.4 | Real calendars only vs real + invented calendars (memorisation check: score separately on dev countries that are not in the pool). |
| G2.5 | Sub-daily effects: daylight-saving 23 / 25-hour days, paydays, month and quarter ends. |
| G2.6 | Calendar-aware versions of the families with their own day types (daily_weekly, repeat_week, epidemic). Only if diagnosis points there. |

- 2026-10-28 (owner): the held-out countries GB, IN, KR, AU, BR are excluded ONLY in G2.4, from both arms (G2.4a real only,
  G2.4b real + invented); the comparison is scored on dev series tagged with those countries. Every other G2 config uses the
  full pool (67 calendars: 55 countries + 12 subdivisions).
- 2026-10-28 (owner): cal_engine REPLACES v0's calendar effect on every row; exposure 0.3 as in v0, so G2.1 tests "better
  calendar at the same exposure". Steps per day {1/7, 1, 24, 48, 96}; weekly rows are built daily and averaged over 7 days.
- 2026-10-28 (owner): G2.2 = cal_engine + the cal_cov group kind with flags shown in 50% of its groups. This differs from
  "50% of rows": generate_batch shuffles rows after H4, so a row's calendar cannot follow it into a block, and flags can
  only be shown for group targets.
- 2026-10-28 (owner): G2.4 waits for batch 2 of the dev data: at least 2 human-activity sources (load, retail, traffic,
  transit, footfall) per held-out country (GB, IN, KR, AU, BR), checked against the benchmark exclusions and tagged with
  their country. Calendar effects show only in human-activity series, so GHCN weather stations do not count. Today the
  tags give GB 0, IN 0, KR 0, AU 1, BR 2 datasets.
- 2026-10-28: the calendar table (tools/build_calendars.py, holidays 0.107) is committed with its sha256; the builder's
  --check fails when a rebuild differs. The holidays package lacks some lunisolar dates (Diwali: IN 2001-2035, LK 2004-2026;
  NP's Tihar 2010-2035); each calendar has a valid year range and windows are placed inside it, never on invented dates.
- 2026-10-28: the G1 scenario switches got a `clock` parameter (fixed = unchanged output; engine = the G2 calendar clock and
  holidays). Their run ids changed (the parameter is in the hash); no G1 run had started.

### G3. Few cycles and seen-once history
- Probes P3: year-once / year-twice; a period longer than a quarter of the context.
- Dev slice: daily series with 1-3 years of history; results split by history length (fev band 2049+: 0.520 vs Chronos-2 0.479).

| ID | Change |
|---|---|
| G3.1 | long_cycle share ~10% -> 15% / 25%. |
| G3.2 | long_cycle built on the calendar engine (day of year, leap years, the yearly pattern drifting one weekday a year). After G2. |
| G3.3 | More long contexts: `long_row_frac` 0.25 -> 0.5. |
| G3.4 | Seasonality layer allows 1.5 cycles in a share of rows (today: 3 cycles minimum, 2 in ets / tsi). |

### G4. Structural breaks and regime change
- Probe P4: a break at a known place; does the forecast take the new level, and how wide do intervals get?
- Dev slice: series hit by 2020 (mobility, TSA, transit), GNSS station steps.

| ID | Change |
|---|---|
| G4.1 | `breaks` on at 0.2. |
| G4.2 | Rate sweep 0.1 / 0.3 (too many breaks widen every interval). |
| G4.3 | Breaks biased to the last 10-30% of the context. |
| G4.4 | Collapse-and-partial-recovery shape. |

### G5. Drifting and quasi-periodic cycles
- Probe P5: amplitude / frequency modulation and beats.
- Dev slice: tide gauges, sunspots / F10.7, ENSO, circadian (BioDare2), GNSS annual cycles.

| ID | Change |
|---|---|
| G5.1 | `tv_season` on. |
| G5.2 | AM/FM family: x(t) = A(t) s(phi(t)), period wandering a few percent. |
| G5.3 | Beats of two close periods (tides 12.42 h + 12 h, spring-neap). |
| G5.4 | Cycles of irregular length (sunspots 9-14 years, ENSO 2-7 years). |

- 2026-10-28 (owner): G5.2-G5.4 are families (tideforge/gen/ext/g5_cycles.py), each with an add (a) and a replace (b) config;
  add shares 4% / 3% / 4%.

### G6. True missing values
- Probe P6: random, block and periodic gaps.
- Dev slice: sources with real gaps (OpenAQ, NDBC, USGS) plus gaps inserted into clean dev series.
- Prerequisite: E0.3 (inference keeps positions).

| ID | Change |
|---|---|
| G6.1 | Scattered missing points marked unobserved in training (value 0, mask 0). |
| G6.2 | Block gaps. |
| G6.3 | Periodic gaps (nights, weekends, market closures). |
| G6.4 | Remove the forward-fill corruption entirely vs keep it alongside. |

### G7. Counts, intermittency, product life cycles
- Probe P7: scaled pinball loss on low counts; calibration of the zero mass.
- Dev slice: VN1 / VN2, Predict Future Sales, H&M, Seattle library, NHS prescribing.

| ID | Change |
|---|---|
| G7.1 | Demand family: low-count Poisson / negative binomial on a seasonal, promotional intensity, Croston-style order gaps. |
| G7.2 | Life-cycle envelope (launch, maturity, decline, death to zero). |
| G7.3 | Stockout censoring (zeros meaning no stock, then a rebound). |
| G7.4 | Larger share of integer-valued series (today ~3%). |

- 2026-10-28 (owner): G7.1 = the demand family (5% add, G7.1a; replace G7.1b; 10% add G7.1c in case 5% shows nothing at 5M).
  G7.2 is demand's `lifecycle_p` 0.5 and is compared with G7.1a, not B0. G7.3 is an H8 hook on count lanes (stockout zeros
  stay observed values; `unobserved: true` writes NaN instead, for later).

### G8. Hierarchies
- Probe P8: coherence (do the parts add up to the total forecast?).
- Dev slice: Iowa liquor by store / county / state, Eurostat regional levels.

| ID | Change |
|---|---|
| G8.1 | Hierarchical group kind: children sum to parents; parents smoother. |

- 2026-10-28 (owner): group-size rules (H6) are a list; each group kind is claimed by at most one rule (an error at install
  time otherwise). G8.1 claims `hierarchy` (g in 3..8), cov_count claims the covariate kinds. "Parents smoother" is measured
  against the children's level-weighted roughness (one dominant child makes the parent about as rough as that child).

### G9. Nonlinear dynamics
- Probe P9: ODE trajectories.
- Dev slice: Living Planet, GPDD, epidemics.

| ID | Change |
|---|---|
| G9.1 | ODE pool generated offline (Lotka-Volterra, seasonally forced SIR, van der Pol, Duffing). |
| G9.2 | Threshold / smooth-transition AR. |

- 2026-10-28 (owner): the ODE pool (tools/build_ode_pool.py, plus Lorenz) is built on Perlmutter into $TF_DATA/pools; only the
  builder and its manifest (sha256, scipy version, seed) are committed; generation refuses to run if the sha differs.
  G9.1c = ode at 10% add. tar (G9.2) and garch_jump (G10.1) have an optional offline pool (tools/build_loop_pool.py, off by
  default, switch param `pool`): they switch to it if `gen_throughput --loops` on GPU shows the loop above 10% of B0's
  generator time at their share.

### G10. Financial tails
- Probe P10: interval calibration under heavy tails (not point skill).
- Dev slice: crypto pairs other than BTC / ETH / LTC / BCH, FX, equities.

| ID | Change |
|---|---|
| G10.1 | Jumps and the leverage effect. |

- 2026-10-28: G10.1's jump sizes are in units of the return sd, mu_J in [-3, 1], sigma_J in [2, 6]; the proposal's
  [-0.05, 0.02] / [0.02, 0.1] were too small to be jumps (corrected before any run).

---

## 6. Phase 3: combining winners

| ID | What |
|---|---|
| X1 | Stack winners one at a time in order of gain; check that gains add up (switches can interact, e.g. breaks + long_cycle). |
| X2 | Leave-one-role-out on the combined mix: which roles carry the score. |
| X3 | Mixture search on the dev tuning set (DoReMi-style), never on fev or TIME. |
| X4 | Data budget: final recipe at 200B vs 100B (a richer mix may need more points even at 5M). |

## 7. Phase 4: exploration (only where Phase 1 points)

Spectral-mixture GP; direct sampling of state-space roots; Hawkes arrivals; dynamic factor groups; multifractal traffic;
observation effects (rounding, reporting delays, revisions); oracle-regret probes; quality-diversity search over
generator settings. Each gets an ID (P4.x) when it is picked up.

---

## 8. Order and budget

1. Phase 0 and Phase 1 together.
2. Switch-only items, no new code: G1.1, G1.2, G3.1, G3.3, G4.1-G4.3, G5.1, G6.1-G6.4.
3. New covariate code: G1.3-G1.13.
4. Calendar: G2, then G3.2.
5. G7, then G8-G10, then the remaining G4 / G5 items.
6. Phase 3, then the milestone fev / TIME scoring.

Budget estimate: ~85 training runs at 5M + evaluation + public-model scoring on the dev set ~ 150-300 A100 GPU-hours
(point estimate ~200). Replace with measured numbers after E0.9.

---

## 9. Index (one line per experiment; keep newest first)

| ID | Date | Status | Result (one line) | Log |
|---|---|---|---|---|
| E0.9 | 2026-10-27 | running | generator: deterministic 1.32-1.44e8 vs default 1.64-1.77e8 points/s (0.80-0.81x); generator runs deterministic; training throughput in T3 | [E0.9](experiments/E0.9.md) |
| D1 | 2026-10-27 | done | gap shaped like small models': 55% from covariate tasks (42% of tasks); worst 15 = load / price / retail; G1 stays first, order G1.3, G1.4, G1.5 | [D1](experiments/D1.md) |
| E0.3 | 2026-10-27 | planned | | [E0.3](experiments/E0.3.md) |
| E0.2 | 2026-10-27 | done | gate pass (tidemark 2026-10-24): ON 43.84 / 0.6797 / 0.5712, hidden 42.08 / 0.6873 / 0.5752; tideforge wrapper = tidemark exactly (smoke, 30 tasks, max rel diff 0) | [E0.2](experiments/E0.2.md) |
| E0.1 | 2026-10-27 | pass | bit-exact on CPU and on GPU (deterministic mode) incl. ft3_main's cfg; B0 = v0-b0-af75b1af | [E0.1](experiments/E0.1.md) |
