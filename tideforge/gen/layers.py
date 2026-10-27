"""Generator v0: Stochastic layers and _finalize.

Ported from third_party/tidemark_e21cbf0/tidemark/real/ft3_core.py L2043-L2318 (tidemark commit e21cbf0) by tools/port_gen_v0.py; do not edit by hand
(edit and rerun the tool). Edits vs the source: imports only; the profile globals read / written as _state.NAME.
Must stay bit-exact with ft3_core (tests/test_gen_parity.py, tests/test_gen_source.py).
"""
# flake8: noqa
import math

import torch
import torch.nn.functional as F

from tideforge.gen import _state
from tideforge.gen.primitives import (  # noqa: F401
    _CLAMP,
    _EXP_CLAMP,
    _bern,
    _forward_fill,
    _pick,
    _pick_period,
    _piecewise_constant,
    _segment_ids,
    _standardize,
    _u,
)
from tideforge.gen.families import (  # noqa: F401
    _seasonal_drift,
)


# >>> ft3_core.py L2043-L2318
# ============================ stochastic layers ============================

def _L_seasonality(x, allow_add, device, p=0.68):
    """1-3 nested seasonal components with NON-SINUSOIDAL shapes.

    Each component is K harmonics of a base period with power-law-decaying random
    coefficients. The decay exponent is randomized per series, which is what produces the
    variety of real seasonal shapes -- slow decay gives sharp spiky peaks (commute traffic),
    fast decay gives smooth single humps (temperature). A single sine wave, which is what
    the original generator used, produces none of this shape diversity.

    Nested periods (e.g. 24 and 24*7) are how real data carries intraday AND day-of-week
    structure simultaneously -- electricity load is the canonical case.
    """
    B, L = x.shape
    t = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)
    gate = _bern(B, device, p) & allow_add.unsqueeze(-1)

    total = torch.zeros_like(x)
    tv = _state._TV_SEASON
    if tv:
        tv_am, tv_ph = _seasonal_drift(B, L, device)
    base_P = _pick_period(B, L, device)
    n_extra = torch.randint(0, 3, (B, 1), device=device)      # 0-2 nested extra periods

    for slot in range(3):
        if slot == 0:
            P = base_P
            active = torch.ones(B, 1, device=device, dtype=torch.bool)
        else:
            # Nested longer cycle (x7 = weekly-on-daily, x12/x30/x52 = annual-on-*).
            # It is DROPPED, not clamped, when it doesn't fit the window. Clamping was a
            # real bug: base_P=24 x30=720 clamped to L/3=341.33 is a fractional,
            # non-canonical "period" that is not a cycle at all -- it injected
            # quasi-periodic structure that diluted the genuine base seasonality, and it
            # was why measured seasonal coverage was stuck near 25% even with the gate
            # forced fully on.
            P = base_P * _pick(B, [7.0, 12.0, 30.0, 52.0], device)
            active = (n_extra >= slot) & (P * 3.0 <= L)
        decay = _u(B, 1, device, 0.3, 1.6)                     # harmonic shape sharpness
        comp = torch.zeros_like(x)
        for k in range(1, 5):
            # harmonic amplitude ~ k^-decay: small decay -> sharp spiky seasonal shape,
            # large decay -> smooth single hump. Randomizing `decay` per series is what
            # gives the variety of real seasonal profiles.
            amp_k = torch.pow(torch.full((1, 1), float(k), device=device), -decay)
            ca = torch.randn(B, 1, device=device) * amp_k
            cb = torch.randn(B, 1, device=device) * amp_k
            ang = 2 * math.pi * k * t / P
            if tv:
                ang = ang + k * tv_ph
            comp = comp + ca * torch.cos(ang) + cb * torch.sin(ang)
        # Amplitude relative to the base process. Too small and the seasonal term is
        # swamped by a random-walk base in the differenced spectrum. Nested cycles get a
        # smaller amplitude than the primary one, which is how real multi-seasonal data
        # behaves -- the day-of-week effect modulates the intraday cycle rather than
        # competing with it on equal terms.
        amp = _u(B, 1, device, 0.8, 4.0) if slot == 0 else _u(B, 1, device, 0.3, 1.2)
        total = total + torch.where(active, comp * amp, torch.zeros_like(comp))

    if tv:
        total = total * tv_am
    sd = x.std(dim=1, keepdim=True) + 1e-6
    additive = x + total * sd
    # multiplicative seasonality: amplitude grows with level (retail/energy standard)
    mult = x * (1.0 + 0.5 * total / (total.std(dim=1, keepdim=True) + 1e-6))
    use_mult = _bern(B, device, 0.3)
    seasoned = torch.where(use_mult, mult, additive)
    return torch.where(gate, seasoned, x)


def _L_trend(x, allow_add, device, p=0.45):
    """Linear / exponential / piecewise-linear trend on top of the latent process."""
    B, L = x.shape
    t = torch.linspace(0, 1, L, device=device).unsqueeze(0)
    gate = _bern(B, device, p) & allow_add.unsqueeze(-1)
    sd = x.std(dim=1, keepdim=True) + 1e-6

    kind = torch.randint(0, 3, (B, 1), device=device)
    lin = t * _u(B, 1, device, -5.0, 5.0)
    expo = (torch.exp((t * _u(B, 1, device, 0.5, 3.0)).clamp(max=_EXP_CLAMP)) - 1.0)
    expo = expo * _u(B, 1, device, -1.5, 1.5)
    piece = (_piecewise_constant(B, L, device, 1.0, 4.0) * _u(B, 1, device, 0.02, 0.3)).cumsum(1)
    trend = torch.where(kind == 0, lin, torch.where(kind == 1, expo, piece))
    return torch.where(gate, x + trend * sd, x)


def _L_calendar(x, allow_add, device, p=0.3):
    """Business-day and holiday effects.

    Real daily data is not smoothly seasonal -- it has a hard 5/2 weekday-weekend split
    (restaurant, sales, traffic) and irregular single-day holiday shocks. Crucially,
    COVID-style reporting artifacts (deaths reported in weekday batches with weekend
    dips) look exactly like this, and covid_deaths was your single worst GIFT-Eval config
    at MASE 38.8 -- a model that has never seen a weekday/weekend reporting pattern has
    no way to represent it.
    """
    B, L = x.shape
    gate = _bern(B, device, p) & allow_add.unsqueeze(-1)
    t = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)
    sd = x.std(dim=1, keepdim=True) + 1e-6

    pts_per_day = _pick(B, [1.0, 24.0, 48.0, 96.0], device)
    day_idx = torch.div(t, pts_per_day, rounding_mode="floor").long().clamp_(0, L - 1)
    dow = day_idx % 7
    weekend = (dow >= 5).float()
    weekend_effect = weekend * _u(B, 1, device, -1.2, 0.3)

    hol_day = (torch.rand(B, L, device=device) < 0.02).float() * _u(B, 1, device, -2.0, 2.0)
    hol = torch.gather(hol_day, 1, day_idx)

    return torch.where(gate, x + (weekend_effect + hol) * sd, x)


def _L_noise(x, is_count, device):
    """Observation noise: homoskedastic Gaussian, heteroskedastic (variance scaling with
    level -- the multiplicative-error regime), or heavy-tailed via a log-normal scale
    mixture (kurtosis well above Gaussian, as in financial and price-spike data).

    Skipped for count rows, where integer-valuedness is added later and continuous noise
    on top would destroy it.
    """
    B, L = x.shape
    sd = x.std(dim=1, keepdim=True) + 1e-6
    kind = torch.randint(0, 3, (B, 1), device=device)
    # Noise magnitude is deliberately modest (mostly <=8% of signal sd, occasionally up to
    # 25%). An earlier draft used up to 35% and it measurably WHITENED the corpus -- median
    # spectral slope came out at -0.73 and only 31% of rows were highly persistent, whereas
    # real corpora are dominated by strongly negative slopes and near-unit-root persistence.
    # Observation noise should perturb a series, not dominate its spectrum.
    big_noise = _bern(B, device, 0.15)
    sigma = torch.where(big_noise, _u(B, 1, device, 0.08, 0.25),
                        _u(B, 1, device, 0.005, 0.08)) * sd

    gauss = torch.randn(B, L, device=device) * sigma
    level = (x - x.mean(1, keepdim=True)).abs() / sd
    het = torch.randn(B, L, device=device) * sigma * (0.3 + level)
    heavy = torch.randn(B, L, device=device) * sigma * torch.exp(
        (torch.randn(B, L, device=device) * _u(B, 1, device, 0.4, 1.1)).clamp(-_EXP_CLAMP, _EXP_CLAMP))

    noise = torch.where(kind == 0, gauss, torch.where(kind == 1, het, heavy))
    return torch.where(is_count.unsqueeze(-1), x, x + noise)


def _L_support(x, is_count, is_nonneg, device, probs=(0.2, 0.2, 0.2, 0.2, 0.2)):
    """Map the latent process onto a realistic SUPPORT.

    Returns (x, is_count_out, is_nonneg_out). Branches:
      identity     -- unconstrained real (financial returns, temperature anomalies)
      positive     -- exp() or softplus: strictly positive, multiplicative errors
      zero-floor   -- relu: non-negative WITH a real mass at exactly zero (demand)
      counts       -- Poisson / negative-binomial integers (births, admissions, deaths)
      intermittent -- Bernoulli-masked integer counts, mostly zeros (car_parts)
      bounded      -- sigmoid-squashed into [lo, hi], or hard-clipped (censoring,
                      utilization %, sensor saturation, stockouts)
    """
    B, L = x.shape
    xs = _standardize(x)

    # --- continuous branches
    positive = torch.exp((xs * _u(B, 1, device, 0.2, 1.0)).clamp(-_EXP_CLAMP, _EXP_CLAMP))
    zero_floor = torch.relu(xs + _u(B, 1, device, -1.0, 0.8))
    lo = _u(B, 1, device, -1.0, 0.0)
    hi = lo + _u(B, 1, device, 0.5, 3.0)
    bounded = lo + (hi - lo) * torch.sigmoid(xs * _u(B, 1, device, 0.5, 2.5))
    censored = torch.clamp(xs, min=lo, max=hi)

    # --- count branches (intensity must be positive)
    intensity = torch.exp((xs * _u(B, 1, device, 0.3, 1.0)
                           + _u(B, 1, device, -1.0, 3.0)).clamp(-_EXP_CLAMP, _EXP_CLAMP))
    intensity = torch.nan_to_num(intensity, nan=1.0, posinf=math.exp(_EXP_CLAMP), neginf=0.0)
    counts = torch.poisson(intensity)
    # negative binomial = Poisson with gamma-mixed intensity (overdispersion, very common
    # in real count data where Poisson's variance==mean assumption is violated)
    overdisp = torch.exp((torch.randn(B, 1, device=device) * 0.5).clamp(-_EXP_CLAMP, _EXP_CLAMP))
    negbin = torch.poisson(intensity * overdisp)
    demand_p = _u(B, 1, device, 0.05, 0.5)
    interm = torch.poisson(intensity * 0.6) * (torch.rand(B, L, device=device) < demand_p).float()

    kind = torch.multinomial(torch.tensor(probs, device=device, dtype=torch.float32), B,
                             replacement=True).unsqueeze(1)
    cont = torch.where(kind == 0, xs,
           torch.where(kind == 1, positive,
           torch.where(kind == 2, zero_floor,
           torch.where(kind == 3, bounded, censored))))
    cont_nonneg = (kind == 1) | (kind == 2)

    cnt_kind = torch.randint(0, 3, (B, 1), device=device)
    cnt = torch.where(cnt_kind == 0, counts, torch.where(cnt_kind == 1, negbin, interm))

    want_count = is_count.unsqueeze(-1)
    out = torch.where(want_count, cnt, cont)
    out_is_count = want_count.squeeze(-1)
    out_is_nonneg = (want_count | cont_nonneg | is_nonneg.unsqueeze(-1)).squeeze(-1)
    # families with a structural zero floor (solar night, bursts) keep their own values
    keep_own = is_nonneg.unsqueeze(-1) & ~want_count
    out = torch.where(keep_own, x, out)
    return out, out_is_count, out_is_nonneg


def _L_corruption(x, is_count, is_nonneg, device, out_p=0.25, miss_p=0.2, quant_p=0.15,
                  quant_step=(0.05, 0.5), clean_p=0.0):
    """Measurement reality: additive outliers, missing runs (forward-filled, which is what
    an imputed real series actually looks like), and quantization/rounding.

    Missing data is forward-filled rather than left as NaN because the model's attention
    has no padding-mask support -- so this teaches robustness to the FLAT-LINED RUNS that
    imputation leaves behind, which is exactly the artifact present in GIFT-Eval's
    *_with_missing datasets.
    """
    B, L = x.shape
    sd = x.std(dim=1, keepdim=True) + 1e-6
    dirty = ~_bern(B, device, clean_p)          # clean gate: these rows get NO corruption

    # additive outliers
    out_gate = _bern(B, device, out_p) & dirty
    spike_mask = (torch.rand(B, L, device=device) < (_u(B, 1, device, 1.0, 12.0) / max(L, 1))).float()
    spikes = spike_mask * torch.randn(B, L, device=device) * sd * _u(B, 1, device, 3.0, 12.0)
    x = torch.where(out_gate, x + spikes, x)

    # missing runs -> forward fill (flat-lined segments)
    miss_gate = _bern(B, device, miss_p) & dirty
    run_start = torch.rand(B, L, device=device) < (_u(B, 1, device, 1.0, 8.0) / max(L, 1))
    run_len = _segment_ids(B, L, device, 1.0, 8.0)          # reuse segmentation for run ids
    in_run = run_start.cumsum(1) % 2 == 1
    observed = ~(in_run & (run_len % 3 == 0))
    observed[:, 0] = True
    filled = _forward_fill(x, observed)
    x = torch.where(miss_gate, filled, x)

    # quantization / rounding to a coarse grid (real sensors and reported figures)
    quant_gate = _bern(B, device, quant_p) & ~is_count.unsqueeze(-1) & dirty
    step = sd * _u(B, 1, device, quant_step[0], quant_step[1])
    x = torch.where(quant_gate, torch.round(x / step) * step, x)

    x = torch.where(is_nonneg.unsqueeze(-1), torch.relu(x), x)
    x = torch.where(is_count.unsqueeze(-1), torch.round(torch.relu(x)), x)
    return x


def _L_scale(x, is_count, is_nonneg, device):
    """Random affine rescaling spanning several orders of magnitude.

    Real corpora span from car_parts (values ~1) to electricity/W (values ~1e5) -- your
    own GIFT-Eval CSV showed MSE from 1.3 to 6.8e10. The model instance-normalizes, so
    this mainly exercises that normalization path and guards against any latent
    scale-dependence.

    Two exclusions, both load-bearing:
      - Offsets are suppressed for non-negative rows, so the meaningful zero floor survives.
      - COUNT rows are left completely untouched. A continuous rescale would turn integers
        back into arbitrary reals and silently destroy integer-valuedness -- which is one
        of the realism properties this whole file exists to produce. Real count data is
        integer at its native scale, so it simply does not get rescaled.
    """
    B, L = x.shape
    scale = torch.exp((torch.randn(B, 1, device=device) * 2.0).clamp(-_EXP_CLAMP, _EXP_CLAMP))
    offset = torch.where(is_nonneg.unsqueeze(-1),
                         torch.zeros(B, 1, device=device),
                         torch.randn(B, 1, device=device) * 5.0)
    scaled = x * scale + offset
    return torch.where(is_count.unsqueeze(-1), x, scaled)


def _finalize(x, fix_degenerate=True):
    """Guard rails: strip non-finite values, clamp magnitude, and replace degenerate
    (zero-variance) series -- which carry no training signal and divide-by-epsilon in
    instance norm -- with a small amount of noise."""
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp(-_CLAMP, _CLAMP)
    if not fix_degenerate:
        return x
    sd = x.std(dim=1, keepdim=True)
    degenerate = sd < 1e-6
    return torch.where(degenerate, x + torch.randn_like(x) * 0.1, x)


# <<< ft3_core.py
