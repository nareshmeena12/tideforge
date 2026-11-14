"""The profiler runs the old path unchanged (same test_error as the old scorer's own call) and splits the time."""
import numpy as np
import torch

from tideforge.eval import thirdparty
from tideforge.score import profile_fev as PF
from tideforge.score import release as RL


def test_profile_matches_old_path(release, tiny_ckpt):
    import fev
    from tideforge import model as TM
    model, _, _ = TM.load(tiny_ckpt, torch.device("cpu"))
    S_, G, C = thirdparty.score_twins(), thirdparty.score_groups(), thirdparty.ft3_core()
    by = {s["kw"]["task_name"]: s for s in RL.Release(release).tasks("full", "tuning")}
    for name in ("a__d/short/target", "b__h/short/cov"):
        for mode in ("off", "on"):
            t = fev.Task(**by[name]["kw"])
            r = PF.profile_task(t, model, torch.device("cpu"), mode, 0)
            t2 = fev.Task(**by[name]["kw"])
            t2.load_full_dataset(num_proc=0)
            if mode == "off":
                s, i = S_.fev_inputs(t2)
                pr = S_.fev_predictions(model, t2, s, i, torch.device("cpu"), C.forecast_left_padded)
            else:
                pr = G.fev_groups_predictions(model, t2, torch.device("cpu"), C.forecast_groups, C.forecast_left_padded,
                                              S_.fill_season, S_.MAX_CTX, S_.BATCH)
            ref = t2.evaluation_summary(pr, model_name="x")["test_error"]
            assert np.isclose(r["test_error"], ref, rtol=1e-6)
            assert r["model_calls"] >= 1 and r["total_s"] >= r["model_s"] > 0
    df = __import__("pandas").DataFrame([r])
    assert "## mode" in PF.summarise(df)
