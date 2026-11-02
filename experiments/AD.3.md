# AD.3: adapter on R1.4, C2's own covariate data

Status: planned
Owner: Naresh
Dates: planned 2026-11-03 / started / finished

---

## Part A. Written BEFORE the run (do not edit after the run starts)

**Gap / phase:** G1 covariates, adapter on a frozen backbone.

**Hypothesis (one sentence):** With the same data, an adapter trained after the backbone changes the covariate column little (control).

**Change (exact config diff against B0):**
```
generator version: v0
switches: C2's mix
mix weights: C2's
anything else: backbone R1.4 frozen; adapter_every 2; train only on blocks with covariates (cov_only); 5B points; seeds 0, 1
```

**Run id(s):** AD.3-*-s0-*, AD.3-*-s1-* (configs/exp/AD.3.yaml; a pilot run has its own train hash)

**What should move:**
- Probe: P1 (covariate cells) and P11 (harm, control).
- Dev slice: fev's 42 covariate tasks, one counted look per model. TIME is not used.
- Expected direction and rough size: near zero, inside the seed range

**Win rule:** the 2-seed mean beats R1.4 on the 42-task covariate column by more than the spread of the adapter seeds. Report each AD run's
seed range. B0's range on that column alone, 37.69-38.82 (about 1.1), is a secondary reference. The 58 non-covariate tasks must equal the
backbone: `score check-identical` fails the job if not (relative tolerance 1e-5).

**What would make this a failure:** no gain on the column beyond seed range; or P11 harm above R1.4's while the column gain is inside it.

**LR:** from the pilot at 3e-4 and 1e-3 on AD.2 / 1B points: the lower of mean P1 sql_on_vs_hidden (non-decoy) + P11 harm_summary. Never
fev or TIME. The configs hold 3e-4; the job passes the chosen value. AD.3 and AD.4 use the same rule on R1.4 only if the owner asks for a second pilot.

**Limitation (owner 2026-11-03):** R1.4 has no M1. The adapter has its own firewall mask, but the frozen group attention stays as trained,
so a covariate token may already carry target information when the adapter reads it. A negative result is "negative at this size, on this
frozen backbone".

**Cost estimate (estimate):** see runbook H7; tools/ad_check.py gives the measured rate.

**Condition:** runs only if AD.2 beats AD.1.

---

## Part B. Written AFTER the run

**What actually ran** (anything that differed from Part A, and why):

**Results:**

| Metric | R1.4 | This run s0 | This run s1 | Seed range |
|---|---|---|---|---|
| fev covariate column (42 tasks) | | | | |
| P1 mean sql_on_vs_hidden | | | | |
| P11 harm_summary / control | | | | |

**Verdict:** win | no effect | harmful | inconclusive

**Why (best explanation, marked as a guess where it is one):**

**Failures, surprises, bugs found:**

**Next step:**

**Counted looks at fev / TIME in this experiment:** 0
