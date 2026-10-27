"""Parameter bookkeeping and the G1 rule (plan section 3):
pass if the five main projections match exactly, everything outside the time
attention is identical, and GLA's only surplus is its gate and norm. The 1%
rule applies at twin scale only (Milestone 4)."""
from __future__ import annotations

import torch.nn as nn


def count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def param_report(model) -> dict:
    """Exact counts for a Forecaster; goes into every results row."""
    mixers = model.mixers()
    infos = [m.info() for m in mixers]
    groups = {"main": 0, "gate": 0, "norm": 0, "other": 0}
    for m in mixers:
        for k, v in m.param_groups().items():
            groups[k] = groups.get(k, 0) + v
    mixer_total = sum(count(m) for m in mixers)
    total = count(model)
    return {
        "params_total": total,
        "params_time_mixer": mixer_total,
        "params_outside_mixer": total - mixer_total,
        "params_main_proj": groups["main"],
        "params_gla_gate": groups["gate"],
        "params_gla_norm": groups["norm"],
        "params_unclassified": mixer_total - groups["main"] - groups["gate"] - groups["norm"],
        **mixers[0].info(),
        # hybrids (F1): every layer's info; the flat fields above are the first layer's
        **({"layer_info": infos} if len({repr(i) for i in infos}) > 1 else {}),
    }


def g1_compare(softmax_report: dict, gla_report: dict) -> dict:
    s, g = softmax_report, gla_report
    checks = {
        "main_projections_equal": s["params_main_proj"] == g["params_main_proj"],
        "outside_identical": s["params_outside_mixer"] == g["params_outside_mixer"],
        "softmax_has_no_surplus": s["params_time_mixer"] == s["params_main_proj"],
        "gla_surplus_is_gate_and_norm_only":
            g["params_unclassified"] == 0
            and g["params_total"] - s["params_total"] == g["params_gla_gate"] + g["params_gla_norm"],
    }
    return {
        "passed": all(checks.values()),
        **checks,
        "softmax_total": s["params_total"],
        "gla_total": g["params_total"],
        "surplus": g["params_total"] - s["params_total"],
        "surplus_pct": 100.0 * (g["params_total"] - s["params_total"]) / s["params_total"],
    }
