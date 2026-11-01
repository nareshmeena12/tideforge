import json

import numpy as np
import pandas as pd

from tideforge.eval import where as W


def _tag(root, tag, errs):
    d = root / "eval" / "fev" / tag
    d.mkdir(parents=True)
    for t, e in errs.items():
        (d / f"{t}.json").write_text(json.dumps(dict(task_name=t, test_error=e)))


def test_names_and_bands():
    assert W.dataset_of("boomlet_1062", ["boomlet_1062", "ETT"]) == "boomlet_1062"
    assert W.dataset_of("rohlik_sales_1D", ["rohlik_sales", "rohlik_orders"]) == "rohlik_sales"
    assert W.frequency_of("ETT_15T") == "15T" and W.frequency_of("australian_tourism") == "other"
    assert W.band(168, W.HORIZON_BANDS) == "49-168" and W.band(169, W.HORIZON_BANDS) == "169+"


def test_report_hand_checked(tmp_path):
    tasks = [f"t{i}" for i in range(8)]
    b1 = {t: 1.0 for t in tasks}
    b2 = {t: float(np.e ** 0.2) for t in tasks}                # log mean 0.1; seeds +-0.1 around it
    a = {t: float(np.exp(0.1 - 0.5 if i < 4 else 0.1)) for i, t in enumerate(tasks)}
    a["only_a"] = 1.0
    for n, e in (("A", a), ("B1", b1), ("B2", b2)):
        _tag(tmp_path, n, e)
    meta = pd.DataFrame(dict(task=tasks, dataset=[f"d{i}" for i in range(8)], domain=["X"] * 4 + ["Y"] * 4, frequency="1H",
                             history="65-512", horizon="13-48", covariates=["yes"] * 4 + ["no"] * 4))
    t = W.table(W.errors(tmp_path, "A"), [W.errors(tmp_path, "B1"), W.errors(tmp_path, "B2")], meta, ["B1", "B2"])
    assert len(t) == 8
    df = W.summarise(t, ("seed_B1", "seed_B2")).set_index(["group", "value"])
    assert abs(df.loc[("all", "all"), "mean"] + 0.25) < 1e-9
    r = df.loc[("covariates", "yes")]
    assert abs(r["mean"] + 0.5) < 1e-9 and r["read"] == "better" and abs(r["b0_lo"] + 0.1) < 1e-9 and abs(r["b0_hi"] - 0.1) < 1e-9
    assert abs(df.loc[("covariates", "no"), "mean"]) < 1e-9
    assert "| covariates | yes | 4 | 4 | -0.5000 |" in W.to_md(df.reset_index(), "x")


def test_ci_resamples_datasets_not_tasks():
    v, c = np.array([0.0] * 9 + [9.0]), np.array([0] * 9 + [1])
    lo, hi = W.boot_ci(v, c)
    assert lo == 0.0 and hi == 9.0                             # two datasets: all of one or all of the other is possible


def test_domains_cover_fev():
    fev = pd.read_csv(W.REPO / "catalog" / "benchmark_datasets.csv").query("benchmark == 'fev-bench'")["dataset"]
    assert set(fev) == set(pd.read_csv(W.REPO / "catalog" / "fev_domains.csv")["dataset"])
