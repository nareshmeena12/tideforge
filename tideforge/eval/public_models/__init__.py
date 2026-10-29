"""Public-model adapters for E0.5 and D2 (owner OK 2026-10-29): Chronos-2, TimesFM-2.5, Toto-2.0 (4m, 22m, 2.5B), TiRex,
TiRex-2, Timestral-5M and -28M, on the one predict interface of tideforge.probes.models (predict_cell -> (n, k, 9, H)).

Nothing here imports a model package: the registry loads without any extra installed (`pip install -e .[public]` adds the
packages that share the core env; Toto-2.0 and TiRex-2 run in their own venvs, see SPECS[...].venv and runner.py).

    make(name, device)           an adapter (in-process), or a runner.Remote proxy when the spec has a venv
    SPECS[name]                  package pin, HF repo + commit, covariates, max context, quantile source, fev results file

Per model (model cards, PyPI and source read 2026-10-29; owner answers 2026-10-29):
- covariates: "past+known" adapters take the cell's past and known lanes in mode "on"; "none" models are scored in hidden
  mode only (devscore --cov records `covariates: none (hidden only)` and skips "on"). TimesFM-2.5 has covariates only through
  XReg (jax): hidden only, by the owner's decision.
- quantiles: every model's output contains 0.1 ... 0.9 exactly; base.select_q9 picks them and refuses to interpolate.
- max_context: the cap devscore truncates contexts to (model limits; Toto-2.0's 4096 is derived from its config's
  residual_attn_ratio = sqrt(S / ln S) with S = 128 patches of 32, an inference, not stated on the card).
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field

Q9 = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass(frozen=True)
class Spec:
    adapter: str                    # module under tideforge.eval.public_models
    repo: str                       # Hugging Face repo id
    revision: str                   # HF commit (weights pinned)
    pip: str                        # pinned install requirement(s), space-separated
    covariates: str                 # "past+known" | "none"
    max_context: int
    fev_csv: str                    # fev benchmarks/fev_bench/results/<fev_csv>.csv
    venv: str | None = None         # $TF_ROOT/venvs/<venv> when the pins conflict with the core env
    licence: str = ""
    kwargs: dict = field(default_factory=dict)


CHRONOS = "chronos-forecasting==2.3.2"
TIMESFM = "timesfm[torch]==3.0.2"
TOTO = "toto-2==2.0.0"
TIREX = "tirex-ts[cuda]==1.4.2"
TIREX2 = "tirex-2 @ git+https://github.com/NX-AI/tirex-2@12aa6dd583a7a5e641801db6d292708200c02776"
TIMESTRAL = "timestral @ git+https://github.com/mistralai/timestral@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"

SPECS = {
    "chronos2": Spec("chronos2", "amazon/chronos-2", "29ec3766d36d6f73f0696f85560a422f50e8498c", CHRONOS, "past+known", 8192,
                     "chronos-2", licence="Apache-2.0", kwargs=dict(batch_size=100)),
    "timesfm25": Spec("timesfm25", "google/timesfm-2.5-200m-pytorch", "1d952420fba87f3c6dee4f240de0f1a0fbc790e3", TIMESFM, "none",
                      16384, "timesfm-2_5", licence="Apache-2.0"),
    "toto2_4m": Spec("toto2", "Datadog/Toto-2.0-4m", "8306a9801cf98c0f5ffe4b2dcc8f496e616d84d9", TOTO, "none", 4096,
                     "toto-2_0-4m", venv="toto2", licence="Apache-2.0", kwargs=dict(batch_size=512)),
    "toto2_22m": Spec("toto2", "Datadog/Toto-2.0-22m", "685e4ae3e2be8d8998025e53dd98e7fdcb296a89", TOTO, "none", 4096,
                      "toto-2_0-22m", venv="toto2", licence="Apache-2.0", kwargs=dict(batch_size=512)),
    # 2.5B: HF commit of 2026-10-29, re-checked at install (e05_public.sh install prints the live sha; a mismatch stops)
    "toto2_2p5b": Spec("toto2", "Datadog/Toto-2.0-2.5B", "51a2812bbe449437c01b79c0e425ed578f335f5b", TOTO, "none", 4096,
                       "toto-2_0-2_5b", venv="toto2", licence="Apache-2.0", kwargs=dict(batch_size=256)),
    "tirex": Spec("tirex", "NX-AI/TiRex", "63c740922493f5fbe60b277609ec62babfba2762", TIREX, "none", 2048, "tirex",
                  licence="NXAI Community (research use allowed)"),
    "tirex2": Spec("tirex2", "NX-AI/TiRex-2", "30788a311699acba7e6393ad9700ea7b7cb9bb2d", TIREX2, "past+known", 2048, "tirex-2",
                   venv="tirex2", licence="Apache-2.0", kwargs=dict(batch_size=512)),
    "timestral5m": Spec("timestral", "mistralai/Timestral-5M", "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f", TIMESTRAL, "past+known",
                        32768, "timestral-5m", licence="Modified MIT (research use allowed)"),
    "timestral28m": Spec("timestral", "mistralai/Timestral-28M", "49b26409ac87b5935539a5d8ffe7f397b720db47", TIMESTRAL,
                         "past+known", 32768, "timestral-28m", licence="Modified MIT (research use allowed)"),
}
REGISTRY = SPECS                    # devscore.make_model looks names up here


def adapter_class(name: str):
    s = SPECS[name]
    mod = importlib.import_module(f"tideforge.eval.public_models.{s.adapter}")
    return mod.Adapter


def make(name: str, device: str = "cuda", backend=None, in_process: bool = False):
    """An adapter for `name`. Specs with a venv are run through runner.Remote (a worker in that venv) unless in_process
    (the worker itself, and tests with a fake backend)."""
    if name not in SPECS:
        raise KeyError(f"unknown public model {name!r} (known: {sorted(SPECS)})")
    if SPECS[name].venv and not in_process and backend is None:
        from tideforge.eval.public_models.runner import Remote
        return Remote(name, device)
    return adapter_class(name)(name, device=device, backend=backend)
