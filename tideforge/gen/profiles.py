"""Generator v0: FAMILIES, weights, GEN_PROFILES, clean2 profile builder, resolve_profile.

Ported from third_party/tidemark_e21cbf0/tidemark/real/ft3_core.py L1924-L2042 (tidemark commit e21cbf0) by tools/port_gen_v0.py; do not edit by hand
(edit and rerun the tool). Edits vs the source: imports only.
Must stay bit-exact with ft3_core (tests/test_gen_parity.py, tests/test_gen_source.py).
"""
# flake8: noqa
import math

import torch
import torch.nn.functional as F

from tideforge.gen.primitives import (  # noqa: F401
    PERIOD_BANK,
    PERIOD_BANK_EXT,
)
from tideforge.gen.families import (  # noqa: F401
    f_aggregated,
    f_ar_p,
    f_ar_process,
    f_burst_onoff,
    f_bursty_load,
    f_count_latent,
    f_cumulative,
    f_daily_weekly,
    f_damped_osc,
    f_diurnal_envelope,
    f_epidemic,
    f_ets,
    f_flat_hold,
    f_flat_sparse,
    f_gp_kernel,
    f_intermittent_latent,
    f_level_shift,
    f_logistic_growth,
    f_long_cycle,
    f_mean_revert,
    f_multiscale_mix,
    f_nonlinear_ar,
    f_piecewise_trend,
    f_power_law,
    f_random_walk,
    f_regime_switch,
    f_repeat,
    f_repeat_week,
    f_sarima,
    f_short_macro_trend,
    f_spike_train,
    f_stochastic_vol,
    f_trend_season_classic,
    f_tsi,
)


# >>> ft3_core.py L1924-L2042
# name -> (fn, is_count, is_nonneg, allow_additive_layers, v1 weight)
FAMILIES = {
    "gp_kernel":        (f_gp_kernel,            False, False, True,  0.10),
    "power_law":        (f_power_law,            False, False, True,  0.08),
    "random_walk":      (f_random_walk,          False, False, True,  0.07),
    "ar_process":       (f_ar_process,           False, False, True,  0.07),
    "mean_revert":      (f_mean_revert,          False, False, True,  0.05),
    "stochastic_vol":   (f_stochastic_vol,       False, False, True,  0.08),
    "regime_switch":    (f_regime_switch,        False, False, True,  0.05),
    "level_shift":      (f_level_shift,          False, False, True,  0.05),
    "piecewise_trend":  (f_piecewise_trend,      False, False, True,  0.05),
    "logistic_growth":  (f_logistic_growth,      False, False, True,  0.03),
    "damped_osc":       (f_damped_osc,           False, False, True,  0.02),
    "diurnal_envelope": (f_diurnal_envelope,     False, True,  False, 0.05),
    "spike_train":      (f_spike_train,          False, True,  False, 0.04),
    "burst_onoff":      (f_burst_onoff,          False, True,  False, 0.04),
    "nonlinear_ar":     (f_nonlinear_ar,         False, False, True,  0.03),
    "trend_season":     (f_trend_season_classic, False, False, True,  0.08),
    "flat_sparse":      (f_flat_sparse,          False, False, True,  0.04),
    "intermittent":     (f_intermittent_latent,  True,  True,  True,  0.05),
    "count":            (f_count_latent,         True,  True,  True,  0.05),
    "multiscale":       (f_multiscale_mix,       False, False, True,  0.04),
    "short_macro_trend": (f_short_macro_trend,   False, False, False, 0.05),
    # ---- clean-corpus families (weight 0 in v1, so v1 is unchanged)
    "ets":              (f_ets,                  False, False, False, 0.0),
    "ar_p":             (f_ar_p,                 False, False, True,  0.0),
    "tsi":              (f_tsi,                  False, False, False, 0.0),
    "flat_hold":        (f_flat_hold,            False, False, False, 0.0),
    "cumulative":       (f_cumulative,           False, True,  False, 0.0),
    # ---- clean2 additions (weight 0 in v1 and clean, so both are unchanged)
    "aggregated":       (f_aggregated,           False, True,  False, 0.0),
    "bursty_load":      (f_bursty_load,          False, True,  False, 0.0),
    "daily_weekly":     (f_daily_weekly,         False, False, False, 0.0),
    "epidemic":         (f_epidemic,             False, True,  False, 0.0),
    "sarima":           (f_sarima,               False, False, False, 0.0),
    # ---- fine-tune addition
    "repeat":           (f_repeat,               False, False, False, 0.0),
    "repeat_week":      (f_repeat_week,          False, False, False, 0.0),
    "long_cycle":       (f_long_cycle,           False, False, False, 0.0),
}

DEFAULT_WEIGHTS = {k: v[4] for k, v in FAMILIES.items() if v[4] > 0}

# Chronos-2-style mix: textbook components (KernelSynth, ETS, AR(p), TSI) dominate; the
# realism families stay but at small weights; rough/zero-heavy families are cut back.
CLEAN_WEIGHTS = {
    "gp_kernel": 0.12, "ets": 0.16, "ar_p": 0.10, "tsi": 0.10, "random_walk": 0.05,
    "ar_process": 0.03, "mean_revert": 0.03, "stochastic_vol": 0.04, "regime_switch": 0.02,
    "level_shift": 0.03, "piecewise_trend": 0.04, "logistic_growth": 0.03, "damped_osc": 0.01,
    "diurnal_envelope": 0.03, "spike_train": 0.02, "burst_onoff": 0.02, "nonlinear_ar": 0.01,
    "trend_season": 0.02, "flat_sparse": 0.01, "flat_hold": 0.02, "intermittent": 0.02,
    "count": 0.03, "multiscale": 0.02, "short_macro_trend": 0.05, "cumulative": 0.02,
}

# families whose defining property (exact flatness, monotonicity) the noise / support /
# corruption layers would destroy: they skip those layers and only get rescaled
PASSTHROUGH_FAMILIES = {"flat_hold", "cumulative", "bursty_load", "epidemic", "repeat", "repeat_week", "long_cycle"}

GEN_PROFILES = {
    "v1": dict(weights=DEFAULT_WEIGHTS, support_probs=(0.2, 0.2, 0.2, 0.2, 0.2),
               out_p=0.25, miss_p=0.20, quant_p=0.15, quant_step=(0.05, 0.5),
               clean_p=0.0, fix_degenerate=True),
    # clean: identity/positive dominate the support map, hard clipping 20% -> 5%, zero-floor
    # 20% -> 12%; half the rows skip corruption entirely; outliers 25% -> 8%, imputed runs
    # 20% -> 6%, quantisation 15% -> 6% with a 5x finer step. Constant rows are left constant
    # (the model must learn flat -> flat; arcsinh + the std floor make that numerically safe).
    "clean": dict(weights=CLEAN_WEIGHTS, support_probs=(0.45, 0.25, 0.12, 0.13, 0.05),
                  out_p=0.08, miss_p=0.06, quant_p=0.06, quant_step=(0.01, 0.1),
                  clean_p=0.5, fix_degenerate=False),
}

# ---- clean2 = clean + the switchable additions (section 3c)
GEN_ADDITIONS_ALL = dict(agg=True, periods=True, epidemic=True, repeat=False, daytype=False,
                         yearly=False, oneoff=False,
                         bursty=False, weekday=False, breaks=False, tv_season=False, sarima=False)
# family -> (switch, weight). CLEAN_WEIGHTS sums to 1.0; these add 0.23 on top (renormalised
# by _allocate), so the full clean2 mix is ~19% new families, ~81% the v2 corpus.
CLEAN2_FAMILY_WEIGHTS = {
    "aggregated":   ("agg",      0.08),
    "sarima":       ("sarima",   0.05),
    "daily_weekly": ("weekday",  0.05),
    "bursty_load":  ("bursty",   0.03),
    "epidemic":     ("epidemic", 0.02),
    "repeat":       ("repeat",   0.15),     # 0.15 / 1.25 = 12% of generated series
    "repeat_week":  ("daytype",  0.10),     # with repeat + weekday + daytype on: repeat 10.7%, repeat_week 7.1%,
                                            # daily_weekly 3.6%, the v5 corpus 78.6%
    "long_cycle":   ("yearly",   0.15),     # with yearly on as well: long_cycle 9.7%, repeat 9.7%, repeat_week 6.5%,
}                                           # daily_weekly 3.2%, the v5 corpus 71.0%


def make_clean2_profile(additions=None, period_ext_share=0.15):
    a = dict(GEN_ADDITIONS_ALL)
    for k, v in (additions or {}).items():
        assert k in a, f"unknown gen_addition {k!r}; valid: {sorted(a)}"
        a[k] = bool(v)
    w = dict(CLEAN_WEIGHTS)
    for fam, (key, wt) in CLEAN2_FAMILY_WEIGHTS.items():
        if a[key]:
            w[fam] = wt
    pr = dict(GEN_PROFILES["clean"])
    pr.update(weights=w, period_bank=PERIOD_BANK_EXT if a["periods"] else PERIOD_BANK,
              period_ext_share=period_ext_share if a["periods"] else None,
              breaks_p=0.2 if a["breaks"] else 0.0, tv_season=a["tv_season"], additions=a,
              one_off_p=0.3 if a["oneoff"] else 0.0)
    return pr


GEN_PROFILES["clean2"] = make_clean2_profile()


def resolve_profile(cfg, key="corpus"):
    """Profile name for cfg[key]; (re)builds clean2 from cfg['gen_additions']."""
    name = cfg.get(key) or cfg.get("corpus", "v1")
    if name == "clean2":
        GEN_PROFILES["clean2"] = make_clean2_profile(cfg.get("gen_additions"),
                                                     cfg.get("period_ext_share", None))
    return name


# <<< ft3_core.py
