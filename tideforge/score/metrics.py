"""Per-task metrics with fev's own metric classes, from the window cache and our quantile forecasts.

fev (Task.evaluation_summary -> EvaluationWindow.compute_metrics) computes every metric per window over all its items at once
(y_true / y_pred [N, H, D], q_pred [N, H, D, Q]) and averages the windows. The only use of the past is the per-item abs
seasonal error (MASE, SQL); the cache stores it as fev computes it, and here it is handed to fev's code as a two-point past
[0, scale] with seasonality 1, whose seasonal error is exactly `scale` (|scale - 0|, the mean of one value). Every metric is
then fev's own compute_scores. Metrics that need another function of the past (RMSSE) are refused.
"""
from __future__ import annotations

import warnings

import numpy as np

PAST_FREE_OR_ABS = {"MASE", "SQL", "WQL", "MQL", "MAE", "MSE", "RMSE", "WAPE", "MAPE", "SMAPE", "RMSLE"}


def metric_objects(info: dict):
    from fev.metrics import get_metric
    ms = [get_metric(m) for m in [info["eval_metric"]] + list(info["extra_metrics"])]
    bad = [m.name for m in ms if m.name not in PAST_FREE_OR_ABS]
    if bad:
        raise NotImplementedError(f"{info['task_name']}: metric(s) {bad} need more of the past than the cache keeps")
    return ms


def point_index(levels) -> int:
    lv = list(np.round(np.asarray(levels, float), 6))
    if 0.5 not in lv:
        raise ValueError("quantile levels without 0.5: no median for the point forecast")
    return lv.index(0.5)


def score_task(cache, q: np.ndarray) -> dict:
    """cache: windows.TaskCache; q: (R, D, Q, H) quantile forecasts (sorted along Q), rows in cache order. The point forecast
    is the median (as the old scorer: predictions = sorted quantiles[4]). -> {metric: mean over windows, test_error,
    num_forecasts, n_windows}."""
    info = cache.info
    levels = list(info["quantile_levels"])
    ms = metric_objects(info)
    qi = point_index(levels)
    per: dict[str, list[float]] = {}
    nf = 0
    for w in range(info["n_windows"]):
        sel = np.flatnonzero(cache.r_win == w)
        if not len(sel):
            continue
        N, D = len(sel), q.shape[1]
        y_true = np.asarray(cache.y_true[sel], dtype=np.float64)                     # (N, H, D)
        q_pred = np.asarray(q[sel], dtype=np.float64).transpose(0, 3, 1, 2)         # (N, H, D, Q)
        y_pred = q_pred[..., qi]
        y_past = np.zeros((N, 2, D))
        y_past[:, 1] = np.asarray(cache.scale[sel], dtype=np.float64)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            for m in ms:
                for k, v in m.compute_scores(y_true=y_true, y_pred=y_pred, y_past=y_past.reshape(2 * N, D),
                                             y_past_lengths=np.full(N, 2), q_pred=q_pred, seasonality=1,
                                             quantile_levels=levels).items():
                    per.setdefault(k, []).append(v)
        nf += N * D
    out = {k: float(np.mean(v)) for k, v in per.items()}
    out.update(test_error=out[ms[0].name], num_forecasts=nf, n_windows=info["n_windows"])
    return out


def fev_summary(cache, scores: dict, model_name: str) -> dict:
    """A row in fev's evaluation_summary format (for fev.leaderboard on fev-bench)."""
    import fev
    t = fev.Task(**cache.info["kw"])
    return {"model_name": model_name, **t.to_dict(), "test_error": scores["test_error"], "training_time_s": None,
            "inference_time_s": None, "num_forecasts": scores["num_forecasts"], "trained_on_this_dataset": False,
            "fev_version": fev.__version__, **{k: v for k, v in scores.items() if k not in ("test_error", "num_forecasts",
                                                                                           "n_windows")}}
