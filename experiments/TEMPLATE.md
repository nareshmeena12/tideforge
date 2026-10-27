# <ID>: <short title>

Status: planned | running | done | abandoned
Owner: Naresh
Dates: planned YYYY-MM-DD / started YYYY-MM-DD / finished YYYY-MM-DD

---

## Part A. Written BEFORE the run (do not edit after the run starts)

**Gap / phase:** (e.g. G4 breaks)

**Hypothesis (one sentence):**

**Change (exact config diff against B0):**
```
generator version:
switches:
mix weights:
anything else:
```

**Run id(s):**

**What should move:**
- Probe:
- Dev slice:
- Expected direction and rough size:

**Win rule:** (default: beats B0 by more than the seed noise on the fast dev score, improves its probe and slice, no domain
slice worse by more than 1%; write a stricter rule here if needed)

**What would make this a failure:**

**Cost estimate:** GPU-hours, points

---

## Part B. Written AFTER the run

**What actually ran** (anything that differed from Part A, and why):

**Results:**

| Metric | B0 | This run | Change | Noise floor |
|---|---|---|---|---|
| Fast dev score | | | | |
| Probe | | | | |
| Target slice | | | | |
| Worst other slice | | | | |

**Verdict:** win | no effect | harmful | inconclusive

**Why (best explanation, marked as a guess where it is one):**

**Failures, surprises, bugs found:**

**Next step:** (confirm with 3 seeds / add-vs-replace test / drop / follow-up ID)

**Counted looks at fev / TIME in this experiment:** 0
