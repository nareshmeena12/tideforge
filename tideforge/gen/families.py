"""Generator v0: Base dynamical families, clean-corpus families and the clean2 additions (section 3c), incl. _L_breaks and _one_off.

Ported from third_party/tidemark_e21cbf0/tidemark/real/ft3_core.py L922-L1923 (tidemark commit e21cbf0) by tools/port_gen_v0.py; do not edit by hand
(edit and rerun the tool). Edits vs the source: imports only; the profile globals read / written as _state.NAME.
Must stay bit-exact with ft3_core (tests/test_gen_parity.py, tests/test_gen_source.py).
"""
# flake8: noqa
import math

import torch
import torch.nn.functional as F

from tideforge.gen import _state
from tideforge.gen.primitives import (  # noqa: F401
    _EXP_CLAMP,
    _MAX_SEG,
    _ar1_filter,
    _bern,
    _fft_filter,
    _pick,
    _pick_period,
    _piecewise_constant,
    _power_law_noise,
    _segment_ids,
    _smooth_noise,
    _standardize,
    _u,
)


# >>> ft3_core.py L922-L1923

# ============================ base dynamical families ============================
# Each: fn(B, L, device) -> (B, L) float32 latent process.

def f_gp_kernel(B, L, device):
    """Composed-kernel GP via circulant embedding -- the KernelSynth family, kept because
    it genuinely covers smooth stationary structure well, with an extended kernel bank
    (adds Matern-1/2 = OU/exponential, and a locally-linear kernel) and up to 3 kernels
    combined by random addition or multiplication."""
    d = torch.linspace(0, 1, L, device=device).unsqueeze(0)      # lag grid (1, L)

    def layer():
        kind = torch.randint(0, 4, (B, 1), device=device)
        ls = _u(B, 1, device, 0.02, 0.5)
        a = _u(B, 1, device, 0.1, 2.0)
        per = (_pick_period(B, L, device) / L).clamp(min=1e-4)
        rbf = torch.exp(-0.5 * d ** 2 / ls ** 2)                              # smooth
        es = torch.exp(-2.0 * torch.sin(math.pi * d / per) ** 2)              # periodic
        rq = (1 + d ** 2 / (2 * a * ls ** 2)) ** (-a)                         # rational quad
        matern12 = torch.exp(-d / ls)                                          # rough / OU
        return torch.where(kind == 0, rbf,
               torch.where(kind == 1, es,
               torch.where(kind == 2, rq, matern12)))

    nk = torch.randint(1, 4, (B, 1), device=device)
    row = layer()
    for dep in range(1, 3):
        nxt = layer()
        add = torch.rand(B, 1, device=device) < 0.5
        row = torch.where(nk > dep, torch.where(add, row + nxt, row * nxt), row)

    c = torch.cat([row, torch.flip(row[:, 1:-1], dims=[1])], dim=1)
    m = c.shape[1]
    lam = torch.clamp(torch.fft.fft(c, dim=1).real, min=0.0)
    z = torch.complex(torch.randn(B, m, device=device), torch.randn(B, m, device=device))
    y = torch.fft.fft(torch.sqrt(lam / m).to(z.dtype) * z, dim=1)[:, :L].real
    return torch.nan_to_num(_standardize(y))


def f_power_law(B, L, device):
    """Long-memory / fractional / pink / brown noise. Covers unit-root behaviour."""
    return _power_law_noise(B, L, device, _u(B, 1, device, 0.2, 2.4))


def f_random_walk(B, L, device):
    """Explicit random walk with optional drift -- the canonical financial price process."""
    drift = _u(B, 1, device, -0.02, 0.02)
    steps = torch.randn(B, L, device=device) + drift
    return steps.cumsum(dim=1)


def f_ar_process(B, L, device):
    """AR(1) across the full stability range, INCLUDING negative phi (oscillatory /
    mean-crossing behaviour, which positive-only AR never produces)."""
    phi = torch.where(torch.rand(B, 1, device=device) < 0.25,
                      _u(B, 1, device, -0.95, -0.2),
                      _u(B, 1, device, 0.2, 0.995))
    return _ar1_filter(B, L, device, phi)


def f_mean_revert(B, L, device):
    """Ornstein-Uhlenbeck around a non-zero level: slow mean reversion, the classic
    spread / inventory / temperature-anomaly process."""
    phi = _u(B, 1, device, 0.95, 0.9995)
    level = _u(B, 1, device, -3.0, 3.0)
    return _ar1_filter(B, L, device, phi) + level


def f_stochastic_vol(B, L, device):
    """Volatility clustering + fat tails: returns whose log-variance follows a slow AR(1),
    then cumulated to a price-like path. This is the GARCH stand-in described in the module
    docstring, and it is the single most important family for your Nifty end-goal -- calm
    periods punctuated by turbulent ones, with return kurtosis well above Gaussian."""
    logvol = _smooth_noise(B, L, device, 0.97, 0.9995) * _u(B, 1, device, 0.3, 1.2)
    vol = torch.exp(logvol.clamp(-_EXP_CLAMP, _EXP_CLAMP))
    rets = torch.randn(B, L, device=device) * vol
    # ~half stay as returns (stationary, fat-tailed), ~half cumulate into a price path
    as_price = _bern(B, device, 0.5)
    return torch.where(as_price, rets.cumsum(dim=1), rets)


def f_regime_switch(B, L, device):
    """Markov-switching between two AR regimes with different persistence and scale --
    e.g. a machine in normal vs degraded mode, a market in calm vs stressed mode."""
    a = _ar1_filter(B, L, device, _u(B, 1, device, 0.3, 0.9))
    b = _ar1_filter(B, L, device, _u(B, 1, device, 0.9, 0.999)) * _u(B, 1, device, 1.5, 4.0)
    seg = _segment_ids(B, L, device, 1.0, 6.0)
    in_b = (seg % 2 == 1).float()
    return a * (1 - in_b) + b * in_b


def f_level_shift(B, L, device):
    """Piecewise-constant levels with mild within-segment noise: sensor recalibration,
    contract changes, capacity steps, deployment-driven jumps in cloud metrics."""
    base = _piecewise_constant(B, L, device, 1.0, 6.0, heavy=True)
    return base + torch.randn(B, L, device=device) * _u(B, 1, device, 0.02, 0.3)


def f_piecewise_trend(B, L, device):
    """Trend whose SLOPE changes at breakpoints (cumsum of piecewise-constant slopes).
    Growth that accelerates, stalls, then reverses -- extremely common in economic series
    and the thing pure-linear-trend models get badly wrong."""
    slopes = _piecewise_constant(B, L, device, 1.0, 5.0) * _u(B, 1, device, 0.01, 0.2)
    return slopes.cumsum(dim=1) + torch.randn(B, L, device=device) * 0.1


def f_logistic_growth(B, L, device):
    """S-curve saturation: epidemic waves, technology adoption, market penetration.
    Includes multi-wave variants (a second, later S-curve superimposed)."""
    t = torch.linspace(0, 1, L, device=device).unsqueeze(0)
    mid = _u(B, 1, device, 0.15, 0.85)
    steep = _u(B, 1, device, 4.0, 30.0)
    s = torch.sigmoid(steep * (t - mid))
    second = _bern(B, device, 0.35)
    mid2 = _u(B, 1, device, 0.5, 0.95)
    s2 = torch.sigmoid(_u(B, 1, device, 6.0, 30.0) * (t - mid2)) * _u(B, 1, device, 0.3, 1.2)
    return torch.where(second, s + s2, s) + torch.randn(B, L, device=device) * 0.02


def f_damped_osc(B, L, device):
    """Decaying oscillation / impulse response of a second-order physical system --
    step responses in control systems, settling behaviour after a shock."""
    t = torch.linspace(0, 1, L, device=device).unsqueeze(0)
    decay = _u(B, 1, device, 0.5, 8.0)
    freq = _u(B, 1, device, 2.0, 40.0)
    phase = _u(B, 1, device, 0, 2 * math.pi)
    return torch.exp(-decay * t) * torch.cos(2 * math.pi * freq * t + phase)


def f_diurnal_envelope(B, L, device):
    """Solar-style generation: a hard-zero night mask (relu of a sinusoid raised to a
    random power) modulated by smooth multiplicative 'cloud cover' noise.

    The structural zeros are the point. No amount of GP sampling produces a series that
    is EXACTLY zero for a contiguous, periodic, deterministic fraction of every cycle,
    yet that is precisely what solar irradiance does -- and GIFT-Eval scores you on four
    solar configs.
    """
    t = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)
    P = _pick_period(B, L, device, min_cycles=3.0)
    phase = _u(B, 1, device, 0, 2 * math.pi)
    env = torch.relu(torch.sin(2 * math.pi * t / P + phase)) ** _u(B, 1, device, 0.7, 2.5)
    cloud = torch.exp((_smooth_noise(B, L, device, 0.95, 0.999)
                       * _u(B, 1, device, 0.1, 0.8)).clamp(-_EXP_CLAMP, _EXP_CLAMP))
    return env * cloud


def f_spike_train(B, L, device):
    """Sparse events convolved with an exponential decay kernel: incident-and-recovery,
    promotion-and-decay, viral spikes with an aftermath tail."""
    rate = _u(B, 1, device, 2.0, 40.0) / max(L, 1)
    events = (torch.rand(B, L, device=device) < rate).float()
    mags = torch.exp(_u(B, L, device, -0.5, 2.0))
    impulses = events * mags
    N = 2 * L
    k = torch.arange(N, device=device, dtype=torch.float32).unsqueeze(0)
    decay = torch.exp(-k / _u(B, 1, device, 2.0, 60.0))
    padded = torch.cat([impulses, torch.zeros(B, L, device=device)], dim=1)
    return _fft_filter(padded, decay)[:, :L]


def f_burst_onoff(B, L, device):
    """On/off bursty traffic with heavy-tailed level contrast -- the self-similar
    behaviour of web/cloud workloads (bitbrains, bizitobs in GIFT-Eval)."""
    seg = _segment_ids(B, L, device, 3.0, 20.0)
    on = (torch.rand(B, _MAX_SEG, device=device) < 0.45).float()
    lvl = torch.exp(_u(B, _MAX_SEG, device, -1.0, 2.5)) * on
    base = torch.gather(lvl, 1, seg)
    return base * torch.exp((torch.randn(B, L, device=device) * 0.3).clamp(-_EXP_CLAMP, _EXP_CLAMP))


def f_nonlinear_ar(B, L, device):
    """Threshold/saturating nonlinear dynamics: an AR process squashed through tanh, so
    large excursions compress. Real bounded physical processes behave this way and a
    linear-Gaussian model never generates it."""
    x = _ar1_filter(B, L, device, _u(B, 1, device, 0.7, 0.99))
    gain = _u(B, 1, device, 0.5, 3.0)
    return torch.tanh(x * gain / (x.std(dim=1, keepdim=True) + 1e-6))


def f_trend_season_classic(B, L, device):
    """The textbook additive trend + seasonal + noise decomposition. Deliberately
    included as its own family because a large fraction of Monash/M4 really does look
    like this, and the model should see plenty of the easy canonical case, not only
    exotic ones."""
    t = torch.linspace(0, 1, L, device=device).unsqueeze(0)
    trend = t * _u(B, 1, device, -4.0, 4.0)
    P = _pick_period(B, L, device)
    tt = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)
    seas = torch.sin(2 * math.pi * tt / P + _u(B, 1, device, 0, 2 * math.pi))
    seas = seas * _u(B, 1, device, 0.3, 3.0)
    return trend + seas + torch.randn(B, L, device=device) * _u(B, 1, device, 0.05, 0.5)


def f_flat_sparse(B, L, device):
    """Near-constant series that change rarely. Real corpora are full of these (low-volume
    hospital departments, slow-moving parts, idle sensors) and a model trained only on
    'interesting' series over-forecasts variation on them. Including them is what stops
    the model inventing structure that isn't there."""
    seg = _segment_ids(B, L, device, 0.5, 3.0)
    vals = torch.randn(B, _MAX_SEG, device=device) * 0.3
    return torch.gather(vals, 1, seg) + torch.randn(B, L, device=device) * 0.01


def f_intermittent_latent(B, L, device):
    """Latent intensity for intermittent demand: low baseline with occasional elevated
    demand periods. The actual zero-inflation and integerization happen in the support
    layer, so seasonality/trend can modulate the INTENSITY first (seasonal demand)."""
    base = _u(B, 1, device, -1.5, 0.3)
    bumps = _piecewise_constant(B, L, device, 2.0, 12.0) * _u(B, 1, device, 0.2, 1.0)
    return base + bumps


def f_count_latent(B, L, device):
    """Latent intensity for count data (births, admissions, deaths): smooth, positive-ish
    drive that the support layer will turn into Poisson/NegBin counts."""
    smooth = _smooth_noise(B, L, device, 0.97, 0.9995)
    return smooth * _u(B, 1, device, 0.3, 1.2) + _u(B, 1, device, 0.5, 3.0)


def f_multiscale_mix(B, L, device):
    """Superposition of a slow component and a fast component with independent dynamics --
    e.g. a slowly drifting baseline with fast measurement dynamics riding on top. Produces
    spectra with two distinct regimes, which single-kernel GPs cannot."""
    slow = _power_law_noise(B, L, device, _u(B, 1, device, 1.5, 2.4))
    fast = _ar1_filter(B, L, device, _u(B, 1, device, 0.1, 0.7))
    w = _u(B, 1, device, 0.2, 0.8)
    return w * _standardize(slow) + (1 - w) * fast


def f_short_macro_trend(B, L, device):
    """Near-monotonic, low-noise trend: annual/quarterly macro and demographic series (life
    expectancy, GDP, population, CO2, tourism arrivals). Targets fev-bench's worst cluster
    (world_life_expectancy 5.1x, us_consumption_1Y 2.9x, world_co2_emissions 2.6x): short,
    trend-dominated, no seasonality, and very rarely reversing more than once or twice.
    trend_season always adds a sinusoid and piecewise_trend's noise/slope ranges are tuned
    for long windows, so neither produces this shape. allow_add=False keeps the generic
    seasonality/trend/calendar layers off it. Noise is scaled to the trend's OWN range, so
    the series stays near-monotonic at any L -- which matters now that short L is common."""
    t = torch.linspace(0, 1, L, device=device).unsqueeze(0)
    seg = _segment_ids(B, L, device, lo_breaks=0.3, hi_breaks=2.5)
    dom_sign = torch.where(torch.rand(B, 1, device=device) < 0.5, 1.0, -1.0)
    raw_slopes = _u(B, _MAX_SEG, device, 0.4, 1.0) * dom_sign
    flip = torch.rand(B, _MAX_SEG, device=device) < 0.18
    raw_slopes = torch.where(flip, -raw_slopes * _u(B, _MAX_SEG, device, 0.15, 0.55), raw_slopes)
    trend = torch.gather(raw_slopes, 1, seg).cumsum(dim=1) / max(L, 1)
    curve = _u(B, 1, device, -0.5, 0.5) * (t - 0.5) ** 2 * dom_sign
    base = _u(B, 1, device, -2.0, 2.0) + trend + curve
    rng = (base.amax(1, keepdim=True) - base.amin(1, keepdim=True)).clamp(min=1e-3)
    return base + torch.randn(B, L, device=device) * rng * _u(B, 1, device, 0.01, 0.06)


# ---------------- Chronos-2-style "clean" base generators (used by corpus="clean") ----------------
# Chronos-2-Synth reached 45.9 fev-bench skill trained on synthetic data only, from textbook
# components: AR, ETS, trend+seasonality+irregular (TSI) and KernelSynth. These are those
# components, written in the same fully-batched O(n log n) style as everything above.

def _causal_conv(e, h):
    """y[t] = sum_{k<=t} h[t-k] e[k] with zero initial state (linear, not circular)."""
    B, L = e.shape
    n = 2 * L
    return torch.fft.irfft(torch.fft.rfft(e, n=n, dim=1) * torch.fft.rfft(h, n=n, dim=1), n=n, dim=1)[:, :L]


def _seasonal_cumsum(e, m_vals):
    """s[t] = sum of e[k] over k <= t with k = t (mod m), per-row period m -- the seasonal
    random walk of ETS (s_t = s_{t-m} + gamma*e_t). Loops only over DISTINCT periods (<= 12)."""
    out = torch.zeros_like(e)
    B, L = e.shape
    for m in m_vals.unique().tolist():
        rows = (m_vals == m).nonzero().squeeze(1)
        m = max(1, int(m))
        C = -(-L // m)
        v = F.pad(e[rows], (0, C * m - L)).view(len(rows), C, m).cumsum(1).view(len(rows), C * m)[:, :L]
        out[rows] = v
    return out


def _seasonal_drift(B, L, device, p=0.4):
    """clean2 tv_season: slowly drifting seasonal amplitude (multiplier) and phase (radians
    of the fundamental) for a fraction p of rows; identity for the rest."""
    on = _bern(B, device, p)
    ph = _smooth_noise(B, L, device, 0.99, 0.9995) * _u(B, 1, device, 0.05, 0.5)
    am = torch.exp((_smooth_noise(B, L, device, 0.98, 0.9995)
                    * _u(B, 1, device, 0.1, 0.5)).clamp(-_EXP_CLAMP, _EXP_CLAMP))
    return (torch.where(on, am, torch.ones_like(am)), torch.where(on, ph, torch.zeros_like(ph)))


def _harmonic_profile(B, L, device, P, K=4):
    """Periodic shape with K harmonics and power-law amplitude decay (smooth humps to spikes).
    With tv_season on (clean2), 40% of rows get amplitude and phase drift."""
    t = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)
    decay = _u(B, 1, device, 0.3, 1.6)
    tv = _state._TV_SEASON
    if tv:
        am, ph = _seasonal_drift(B, L, device)
    s = torch.zeros(B, L, device=device)
    for k in range(1, K + 1):
        amp = torch.full((1, 1), float(k), device=device) ** (-decay)
        ang = 2 * math.pi * k * t / P
        if tv:
            ang = ang + k * ph
        s = s + amp * (torch.randn(B, 1, device=device) * torch.cos(ang)
                       + torch.randn(B, 1, device=device) * torch.sin(ang))
    if tv:
        s = s * am
    return s / (s.std(1, keepdim=True) + 1e-6)


def f_ets(B, L, device):
    """Exponential smoothing state-space models (the ETS family): single-source-of-error
    level + {none, additive, damped} trend + optional seasonal random walk, additive or
    multiplicative. All components are linear filters of ONE innovation sequence e, so the
    whole thing is cumsums / an FFT convolution -- no sequential recursion."""
    t = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)
    e = torch.randn(B, L, device=device) * _u(B, 1, device, 0.05, 0.6)
    ce = e.cumsum(1)
    y = _u(B, 1, device, 0.05, 0.8) * ce                                  # level: alpha * sum e
    kind = torch.randint(0, 3, (B, 1), device=device)                      # 0 none 1 additive 2 damped
    beta = _u(B, 1, device, 0.001, 0.1)
    b0 = torch.randn(B, 1, device=device) * _u(B, 1, device, 0.0, 0.05)
    phi = _u(B, 1, device, 0.8, 0.98)
    add_tr = b0 * t + beta * ce.cumsum(1)
    damp_b = _causal_conv(beta * e, phi ** t) + b0 * phi ** t              # b_t = phi b_{t-1} + beta e_t
    damp_tr = damp_b.cumsum(1)
    y = y + torch.where(kind == 1, add_tr, torch.where(kind == 2, damp_tr, torch.zeros_like(y)))
    m = _pick_period(B, L, device, min_cycles=2.0)                         # (B,1) float
    seas = (_harmonic_profile(B, L, device, m) * _u(B, 1, device, 0.5, 3.0) * e.std(1, keepdim=True)
            + _u(B, 1, device, 0.0, 0.3) * _seasonal_cumsum(e, m.squeeze(1).long()))
    y = y + torch.where(_bern(B, device, 0.65), seas, torch.zeros_like(seas)) + e
    mult = torch.exp((_standardize(y) * _u(B, 1, device, 0.1, 0.6)).clamp(-_EXP_CLAMP, _EXP_CLAMP))
    return torch.where(_bern(B, device, 0.3), mult, y)


def f_ar_p(B, L, device):
    """Stationary AR(p), p <= 4, built from random roots: up to two complex-conjugate pairs
    (pseudo-cycles of random period and persistence -- the oscillatory behaviour AR(1) can
    never produce) and up to two real roots. Filtered in the frequency domain: X = E / A."""
    N = 2 * L
    a = torch.zeros(B, 5, device=device)
    a[:, 0] = 1.0

    def mul(a, c1, c2):                     # a(z) * (1 + c1 z^-1 + c2 z^-2)
        out = a.clone()
        out[:, 1:] = out[:, 1:] + c1 * a[:, :-1]
        out[:, 2:] = out[:, 2:] + c2 * a[:, :-2]
        return out

    n_pairs = torch.randint(0, 3, (B, 1), device=device)
    n_real = torch.where(n_pairs == 2, torch.zeros_like(n_pairs), torch.randint(0, 3, (B, 1), device=device))
    n_real = torch.where((n_pairs + n_real) == 0, torch.ones_like(n_real), n_real)
    for i in range(2):
        rho = _u(B, 1, device, 0.5, 0.995)
        th = math.pi * _u(B, 1, device, 0.0, 1.0) ** 2 + 1e-3          # bias toward slow cycles
        on = n_pairs > i
        a = mul(a, torch.where(on, -2 * rho * torch.cos(th), torch.zeros_like(rho)),
                torch.where(on, rho ** 2, torch.zeros_like(rho)))
    for i in range(2):
        r = torch.where(torch.rand(B, 1, device=device) < 0.2, _u(B, 1, device, -0.9, -0.2),
                        _u(B, 1, device, 0.1, 0.99))
        on = n_real > i
        a = mul(a, torch.where(on, -r, torch.zeros_like(r)), torch.zeros_like(r))
    # FLOAT64 IS REQUIRED HERE. With roots near 1 (a slow pair + two real roots ~0.99),
    # A(e^{iw}) near w=0 is ~3e-9 while float32 rounding on the coefficients is ~4e-7: the
    # computed A can be exactly 0 -> inf -> an all-NaN row -> torch.poisson device-side
    # assert on GPU (seen after ~50 steps of the first v2 run). In float64 the rounding is
    # ~1e-15, far below the smallest |A| these root ranges can produce (~1e-10).
    eps = torch.randn(B, N, device=device, dtype=torch.float64)
    x = torch.fft.irfft(torch.fft.rfft(eps, dim=1) / torch.fft.rfft(a.double(), n=N, dim=1), n=N, dim=1)[:, L:]
    return _standardize(x).float()


def f_tsi(B, L, device):
    """Trend + Seasonality + Irregularity (Chronos-2's TSI generator): one trend shape,
    1-2 seasonal components (the second nested, e.g. weekly-on-daily), AR(1) irregular."""
    t = torch.linspace(0, 1, L, device=device).unsqueeze(0)
    kind = torch.randint(0, 5, (B, 1), device=device)
    lin = t * _u(B, 1, device, -3.0, 3.0)
    expo = (torch.exp((t * _u(B, 1, device, 0.5, 3.0)).clamp(max=_EXP_CLAMP)) - 1.0) * _u(B, 1, device, -1.5, 1.5)
    logi = torch.sigmoid(_u(B, 1, device, 4.0, 25.0) * (t - _u(B, 1, device, 0.1, 0.9))) * _u(B, 1, device, -3.0, 3.0)
    piece = (_piecewise_constant(B, L, device, 1.0, 4.0) * _u(B, 1, device, 0.5, 3.0) / max(L, 1)).cumsum(1)
    trend = torch.where(kind == 1, lin, torch.where(kind == 2, expo, torch.where(kind == 3, logi,
            torch.where(kind == 4, piece, torch.zeros_like(lin)))))
    P1 = _pick_period(B, L, device, min_cycles=2.0)
    s = _harmonic_profile(B, L, device, P1) * _u(B, 1, device, 0.3, 2.0)
    P2 = P1 * _pick(B, [7.0, 12.0, 4.0], device)
    s2 = _harmonic_profile(B, L, device, P2) * _u(B, 1, device, 0.2, 1.0)
    s = s + torch.where(_bern(B, device, 0.4) & (P2 * 2.0 <= L), s2, torch.zeros_like(s2))
    s = torch.where(_bern(B, device, 0.8), s, torch.zeros_like(s))
    irr = _ar1_filter(B, L, device, _u(B, 1, device, 0.0, 0.8))
    irr = irr / (irr.std(1, keepdim=True) + 1e-6) * _u(B, 1, device, 0.05, 0.6)
    return trend + s + irr


def f_flat_hold(B, L, device):
    """Near-constant series that STAY near-constant: exact or almost-exact flat levels with at
    most one or two small steps. fev-bench's boomlet_285 scored 49.6x worse than naive
    because the model had only ever seen flat stretches followed by jumps (intermittent
    demand, level shifts) -- it had never learned that flat context -> flat future. Bypasses
    the noise/corruption layers (see PASSTHROUGH_FAMILIES) so the flatness survives."""
    seg = _segment_ids(B, L, device, 0.0, 1.5)
    steps = torch.randn(B, _MAX_SEG, device=device) * _u(B, 1, device, 0.02, 0.5)
    steps[:, 0] = 0.0
    lvl = torch.gather(steps.cumsum(1), 1, seg) + torch.randn(B, 1, device=device) * 2.0
    noise = torch.randn(B, L, device=device) * _u(B, 1, device, 0.0005, 0.01)
    return lvl + torch.where(_bern(B, device, 0.5), torch.zeros_like(noise), noise)


def f_cumulative(B, L, device):
    """Running totals (cumulative cases, odometers, year-to-date counters): monotone
    non-decreasing, integer or continuous increments with a slowly varying rate. Target of
    uk_covid_*/cumulative. Bypasses noise/corruption so monotonicity survives."""
    rate = torch.exp((_smooth_noise(B, L, device, 0.95, 0.999) * _u(B, 1, device, 0.2, 1.0)
                      + _u(B, 1, device, -2.0, 2.0)).clamp(-_EXP_CLAMP, _EXP_CLAMP))
    rate = torch.nan_to_num(rate, nan=1.0, posinf=math.exp(_EXP_CLAMP), neginf=0.0)
    inc = torch.where(_bern(B, device, 0.5), torch.poisson(rate), rate * _u(B, 1, device, 0.5, 1.5))
    return inc.cumsum(1)


# ================= 3c. clean2 additions: generators for MEASURED fev-bench gaps =================
# Every family below exists because a specific group of fev-bench tasks was worse than the
# rest in the v2 eval (0.373). Each is switchable in CONFIG["gen_additions"].

_AGG_FINE_CAP = 1 << 17      # max fine-series length generated per row (memory bound)

# (aggregation factor k, short cycle of the FINE series, annual cycle of the FINE series, weight)
# k may be fractional: 30.436875 = mean days per month, so months come out 30 or 31 days
# long, exactly the irregular-block (trading-day) structure of real monthly totals.
_AGG_SCENARIOS = [
    (7.0,       7.0,  365.25, 0.38),   # daily   -> weekly    (annual cycle becomes 52.18)
    (30.436875, 7.0,  365.25, 0.30),   # daily   -> monthly   (annual cycle becomes 12)
    (3.0,       None, 12.0,   0.08),   # monthly -> quarterly (annual cycle becomes 4)
    (12.0,      None, 12.0,   0.07),   # monthly -> yearly    (annual cycle cancels: macro)
    (24.0,      24.0, 168.0,  0.12),   # hourly  -> daily     (weekly cycle becomes 7)
    (4.0,       96.0, 672.0,  0.05),   # 15-min  -> hourly
]


def _fine_calendar(n, N, device, p_short, p_annual):
    """A FINE-resolution positive calendar process (e.g. daily sales): log-level random walk
    + trend + short cycle (day-of-week) + annual cycle + sparse holiday / promo shocks +
    multiplicative noise; 35% of rows become fine-level counts (slow movers), which aggregate
    into small integers."""
    t = torch.arange(N, device=device, dtype=torch.float32).unsqueeze(0) / N
    logy = _standardize(torch.randn(n, N, device=device).cumsum(1)) * _u(n, 1, device, 0.05, 0.7)
    logy = logy + t * _u(n, 1, device, -1.5, 1.5)
    if p_short is not None:
        logy = logy + torch.where(_bern(n, device, 0.8),
                                  _harmonic_profile(n, N, device, p_short) * _u(n, 1, device, 0.0, 0.5),
                                  torch.zeros(n, N, device=device))
    logy = logy + torch.where(_bern(n, device, 0.75),
                              _harmonic_profile(n, N, device, p_annual) * _u(n, 1, device, 0.0, 0.6),
                              torch.zeros(n, N, device=device))
    hol = (torch.rand(n, N, device=device) < _u(n, 1, device, 0.0, 0.005)).float()
    logy = logy + hol * torch.randn(n, N, device=device) * _u(n, 1, device, 0.2, 1.0)
    logy = logy + torch.randn(n, N, device=device) * _u(n, 1, device, 0.02, 0.5)
    y = torch.exp(logy.clamp(-_EXP_CLAMP, _EXP_CLAMP))
    rate = torch.nan_to_num(y * torch.exp(_u(n, 1, device, -2.5, 3.0)), nan=1.0,
                            posinf=math.exp(12.0), neginf=0.0).clamp(0.0, math.exp(12.0))
    return torch.where(_bern(n, device, 0.35), torch.poisson(rate), y)


def _block_aggregate(y, k, L):
    """(n, N) fine -> (n, L) coarse: SUM over consecutive blocks of (possibly fractional)
    length k with a random phase (flows: sales, counts, energy), or, for 20% of rows, the
    LAST value of each block (stocks: prices, levels, end-of-period readings). Block 0 can
    be partial and is dropped."""
    n, N = y.shape
    dev = y.device
    phase = torch.rand(n, 1, device=dev, dtype=torch.float64) * k
    t = torch.arange(N, device=dev, dtype=torch.float64).unsqueeze(0)
    bid = torch.floor((t + phase) / k).long().clamp_(max=L + 2)
    s = torch.zeros(n, L + 3, device=dev, dtype=y.dtype).scatter_add_(1, bid, y)[:, 1:L + 1]
    j = torch.arange(1, L + 1, device=dev, dtype=torch.float64).unsqueeze(0)
    idx = (torch.ceil((j + 1) * k - phase) - 1).long().clamp_(0, N - 1)
    last = torch.gather(y, 1, idx)
    return torch.where(_bern(n, dev, 0.2), last, s)


def f_aggregated(B, L, device):
    """Low-frequency series made the way real ones are: by aggregating a finer process.
    Weekly / monthly data is smoother than daily, carries non-integer annual periods
    (52.18 weeks), irregular month lengths, and no day-of-week cycle (it sums out). Long
    windows only use small k (a 8192-point monthly series would need 250k fine points and
    does not exist in practice)."""
    ok = [s for s in _AGG_SCENARIOS if (L + 3) * s[0] <= _AGG_FINE_CAP]
    if not ok:
        ok = [min(_AGG_SCENARIOS, key=lambda s: s[0])]
    which = torch.multinomial(torch.tensor([s[3] for s in ok]), B, replacement=True)
    out = torch.zeros(B, L, device=device)
    for i, (k, p_short, p_annual, _) in enumerate(ok):
        rows = (which == i).nonzero().squeeze(1)
        if rows.numel() == 0:
            continue
        N = int(math.ceil((L + 3) * k))
        y = _fine_calendar(rows.numel(), N, device, p_short, p_annual)
        out[rows.to(device)] = _block_aggregate(y, k, L).float()
    return out


def f_bursty_load(B, L, device):
    """Query / job load (redset): a daily + weekly baseline intensity, bursts from a Hawkes
    CLUSTER process (burst-starting events ~ Poisson(baseline), each with a heavy-tailed
    lognormal mass spread by an exponential kernel -- the Poisson-cluster representation of
    self-excitation, so no sequential recursion), plus scheduled jobs firing at a fixed
    offset every Pj steps (ETL / cron) for 40% of rows. Half counts, half continuous load."""
    t = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)
    ppd = _pick(B, [24.0, 96.0, 288.0], device)                       # 1H / 15T / 5T
    log0 = (_u(B, 1, device, -1.0, 3.0)
            + _harmonic_profile(B, L, device, ppd, K=6) * _u(B, 1, device, 0.3, 1.5)
            + _harmonic_profile(B, L, device, 7.0 * ppd) * _u(B, 1, device, 0.0, 0.5)
            + _smooth_noise(B, L, device, 0.99, 0.9995) * _u(B, 1, device, 0.0, 0.4))
    lam0 = torch.exp(log0.clamp(-_EXP_CLAMP, _EXP_CLAMP))
    starts = torch.poisson((lam0 * _u(B, 1, device, 0.01, 0.2)).clamp(0.0, 1e5))
    mass = starts * torch.exp((torch.randn(B, L, device=device) * _u(B, 1, device, 0.5, 1.5))
                              .clamp(-_EXP_CLAMP, _EXP_CLAMP)) * _u(B, 1, device, 1.0, 20.0)
    h = torch.exp(-t / _u(B, 1, device, 0.5, 12.0))
    burst = _causal_conv(mass, h / h.sum(1, keepdim=True)).clamp(min=0.0)
    Pj = _pick(B, [12.0, 24.0, 48.0, 96.0, 288.0], device)
    off = torch.floor(_u(B, 1, device, 0.0, 1.0) * Pj)
    fire = ((torch.remainder(t - off, Pj) == 0) & (torch.rand(B, L, device=device) < 0.9)).float()
    job = fire * _u(B, 1, device, 1.0, 10.0) * lam0.mean(1, keepdim=True)
    lam = lam0 + burst + torch.where(_bern(B, device, 0.4), job, torch.zeros_like(job))
    lam = torch.nan_to_num(lam, nan=1.0, posinf=1e6, neginf=0.0).clamp(0.0, 1e6)
    cont = lam * torch.exp(torch.randn(B, L, device=device) * _u(B, 1, device, 0.0, 0.2))
    return torch.where(_bern(B, device, 0.5), torch.poisson(lam), cont)


def f_daily_weekly(B, L, device):
    """Sub-daily series whose intraday SHAPE depends on the day type: weekday profile vs a
    different weekend / holiday profile (later, flatter peak), a small per-day-of-week level,
    an AR(1) day-to-day level, slow drift and an AR(1) irregular. A multiplicative weekend
    factor on one fixed profile -- what the seasonality and calendar layers produce -- can't
    represent this; real load, traffic and query data look like this."""
    ppd = _pick(B, [24.0, 48.0, 96.0, 144.0, 288.0], device)
    t = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0) \
        + torch.floor(_u(B, 1, device, 0.0, 1.0) * 7.0 * ppd)
    day = torch.floor(t / ppd)
    tod = t - day * ppd
    dayl = day.long()
    dow = dayl % 7
    nd = int((L + 7 * 288) // 24) + 2                                # static bound on day index
    hol = torch.rand(B, nd, device=device) < _u(B, 1, device, 0.0, 0.03)
    wkend = torch.where(_bern(B, device, 0.2), dow == 6, dow >= 5) | torch.gather(hol, 1, dayl)
    decay = _u(B, 1, device, 0.3, 1.5)
    rho = _u(B, 1, device, 0.0, 0.85)
    orth = torch.sqrt(1.0 - rho * rho)
    Wp = torch.zeros(B, L, device=device)
    Ep = torch.zeros(B, L, device=device)
    for k in range(1, 7):
        a = torch.full((1, 1), float(k), device=device) ** (-decay)
        ang = 2 * math.pi * k * tod / ppd
        c1, c2 = torch.randn(B, 1, device=device) * a, torch.randn(B, 1, device=device) * a
        e1 = rho * c1 + orth * torch.randn(B, 1, device=device) * a
        e2 = rho * c2 + orth * torch.randn(B, 1, device=device) * a
        Wp = Wp + c1 * torch.cos(ang) + c2 * torch.sin(ang)
        Ep = Ep + e1 * torch.cos(ang) + e2 * torch.sin(ang)
    Wp = Wp / (Wp.std(1, keepdim=True) + 1e-6)
    Ep = Ep / (Ep.std(1, keepdim=True) + 1e-6) * _u(B, 1, device, 0.3, 1.2) + _u(B, 1, device, -1.5, 0.5)
    prof = torch.where(wkend, Ep, Wp)
    dow_lvl = torch.gather(torch.randn(B, 7, device=device) * _u(B, 1, device, 0.0, 0.3), 1, dow)
    dl = _ar1_filter(B, nd, device, _u(B, 1, device, 0.3, 0.97))
    dl = torch.gather(dl / (dl.std(1, keepdim=True) + 1e-6), 1, dayl) * _u(B, 1, device, 0.0, 0.8)
    slow = _smooth_noise(B, L, device, 0.995, 0.9999) * _u(B, 1, device, 0.0, 1.0)
    irr = _ar1_filter(B, L, device, _u(B, 1, device, 0.0, 0.9))
    irr = irr / (irr.std(1, keepdim=True) + 1e-6) * _u(B, 1, device, 0.05, 0.4)
    y = prof * _u(B, 1, device, 1.0, 3.0) + dow_lvl + dl + slow + irr
    mult = torch.exp((_standardize(y) * _u(B, 1, device, 0.2, 0.6)).clamp(-_EXP_CLAMP, _EXP_CLAMP))
    return torch.where(_bern(B, device, 0.4), mult, y)


def f_epidemic(B, L, device):
    """Epidemic incidence: 1-4 asymmetric waves (sech^2 bumps, slower decay than rise, centres
    possibly outside the window so a series can start mid-wave), for 30% of rows a recurring
    seasonal wave train (flu: period 52 weekly / 365 daily), a small endemic baseline, and for
    half of rows a weekday reporting pattern (weekend dip). Counts or continuous rates."""
    t = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)
    inc = torch.zeros(B, L, device=device)
    n_w = torch.randint(1, 5, (B, 1), device=device)
    for w in range(4):
        c = _u(B, 1, device, -0.1, 1.1) * L
        wl = _u(B, 1, device, 0.02, 0.2) * L + 1.0
        wr = wl * _u(B, 1, device, 0.8, 3.0)
        z = (t - c) / torch.where(t < c, wl, wr)
        bump = torch.exp(_u(B, 1, device, -1.5, 1.5)) * (1.0 - torch.tanh(z) ** 2)
        inc = inc + torch.where(n_w > w, bump, torch.zeros_like(bump))
    P = _pick(B, [52.0, 365.0], device)
    seas = torch.relu(torch.cos(2 * math.pi * t / P + _u(B, 1, device, 0.0, 2 * math.pi))) \
        ** _u(B, 1, device, 2.0, 8.0) * torch.exp(_u(B, 1, device, -1.0, 1.0))
    inc = inc + torch.where(_bern(B, device, 0.3), seas, torch.zeros_like(seas))
    peak = inc.amax(1, keepdim=True) + 1e-6
    inc = inc + _u(B, 1, device, 0.0, 0.05) * peak + 1e-3 * peak
    rate = inc / (inc.amax(1, keepdim=True) + 1e-12) * torch.exp(_u(B, 1, device, 0.5, 8.0))
    dow = torch.remainder(t + torch.randint(0, 7, (B, 1), device=device).float(), 7.0)
    rep = 1.0 - _u(B, 1, device, 0.2, 0.8) * (dow >= 5).float()
    rate = torch.where(_bern(B, device, 0.5), rate * rep, rate)
    rate = torch.nan_to_num(rate, nan=1.0, posinf=1e6, neginf=0.0).clamp(0.0, 1e6)
    cont = rate * torch.exp(torch.randn(B, L, device=device) * _u(B, 1, device, 0.0, 0.15))
    return torch.where(_bern(B, device, 0.6), torch.poisson(rate), cont)


def f_sarima(B, L, device):
    """Seasonal ARIMA (p,d,q)(P,D,Q)_m with p,q,P,Q <= 1 at a CALENDAR lag m (not random
    sparse lags): (1 - phi z)(1 - Phi z^m) x = (1 + theta z)(1 + Theta z^m) e, filtered in the
    frequency domain in float64 (|A| >= (1-|phi|)(1-|Phi|) > 0), then optional regular and
    seasonal integration (d, D in {0, 1}) -- the airline-model class of weekly/monthly data."""
    N = 2 * L
    m = _pick_period(B, L, device, min_cycles=3.0).long()
    f64 = torch.float64
    z = torch.zeros(B, 1, device=device)
    # mostly persistent, positive seasonal AR (visible seasonality); MA terms mostly negative
    # (the airline model (0,1,1)(0,1,1)_m has theta, Theta < 0)
    phi = torch.where(_bern(B, device, 0.75), _u(B, 1, device, -0.3, 0.95), z).double()
    th = torch.where(_bern(B, device, 0.5), _u(B, 1, device, -0.8, 0.4), z).double()
    Phi = torch.where(_bern(B, device, 0.8), _u(B, 1, device, 0.1, 0.95), _u(B, 1, device, -0.4, 0.1)).double()
    Th = torch.where(_bern(B, device, 0.5), _u(B, 1, device, -0.8, 0.3), z).double()
    a = torch.zeros(B, N, device=device, dtype=f64)
    a[:, 0] = 1.0
    a[:, 1:2] -= phi
    a.scatter_add_(1, m, -Phi)
    a.scatter_add_(1, m + 1, phi * Phi)
    b = torch.zeros(B, N, device=device, dtype=f64)
    b[:, 0] = 1.0
    b[:, 1:2] += th
    b.scatter_add_(1, m, Th)
    b.scatter_add_(1, m + 1, th * Th)
    e = torch.randn(B, N, device=device, dtype=f64)
    x = torch.fft.irfft(torch.fft.rfft(e, dim=1) * torch.fft.rfft(b, dim=1)
                        / torch.fft.rfft(a, dim=1), n=N, dim=1)[:, L:]
    x = _standardize(x).float()
    combo = torch.multinomial(torch.tensor([0.30, 0.20, 0.30, 0.20]), B, replacement=True).to(device)
    d = ((combo == 1) | (combo == 3)).unsqueeze(1)
    D = ((combo == 2) | (combo == 3)).unsqueeze(1)
    x = torch.where(D, _seasonal_cumsum(x, m.squeeze(1)), x)
    x = torch.where(d, x.cumsum(1), x)
    return _standardize(x)


def _L_breaks(x, device, p=0.2):
    """clean2 breaks: persistent structural breaks at a LOW rate (p of rows, 1-2 breaks).
    Every break is also an unforecastable target for the patches right after it under the
    dense next-patch loss, so a high rate would teach the model to widen every interval.
    What we want it to learn is 'a break happened in the context -> trust the recent part'.
    Kinds: 0 level shift, 1 slope change, 2 variance / amplitude change, 3 shock with V / U
    (full recovery, fast / slow) or L (partial, persistent) recovery, 4 level + amplitude."""
    B, L = x.shape
    if L < 8:
        return x
    t = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)
    gate = _bern(B, device, p)
    sd = x.std(1, keepdim=True) + 1e-6
    m = x.mean(1, keepdim=True)
    one = torch.ones(B, 1, device=device)
    n_b = torch.where(_bern(B, device, 0.3), 2 * one, one)
    for j in range(2):
        active = gate & (n_b > j)
        tb = torch.floor(_u(B, 1, device, 0.05, 0.95) * L)
        after = (t >= tb).float()
        dt = (t - tb).clamp(min=0.0)
        kind = torch.randint(0, 5, (B, 1), device=device)
        sgn = torch.where(_bern(B, device, 0.5), one, -one)
        level = sgn * _u(B, 1, device, 0.5, 3.0) * sd * after
        slope = torch.randn(B, 1, device=device) * _u(B, 1, device, 0.5, 4.0) * sd * dt / L
        amp = (x - m) * (torch.exp(_u(B, 1, device, -1.2, 1.2)) - 1.0) * after
        tau = _u(B, 1, device, 0.005, 0.3) * L + 1.0
        keep = torch.where(_bern(B, device, 0.5), torch.zeros_like(one), _u(B, 1, device, 0.2, 1.0))
        ssgn = torch.where(_bern(B, device, 0.8), -one, one)
        shock = ssgn * _u(B, 1, device, 1.0, 5.0) * sd * after * ((1 - keep) * torch.exp(-dt / tau) + keep)
        delta = torch.where(kind == 0, level, torch.where(kind == 1, slope, torch.where(
            kind == 2, amp, torch.where(kind == 3, shock, level + amp))))
        x = x + torch.where(active, delta, torch.zeros_like(delta))
    return x


def f_repeat(B, L, device):
    """EXACTLY REPEATING patterns with ANY period and full-detail shapes.

    Why: a probe of the 700B model (noise-free sawtooth / square / spike / random-template series
    at 16 periods) showed it continues a pattern only when the period is one this generator
    uses -- the period bank 4/7/12/24/48/96/144/168/... -- and only when the shape is smooth
    (the seasonal layer builds shapes from ~4 harmonics). At period 10 it failed even on a sine;
    spikes and random templates scored ~0 at every period above 24. Chronos-2 scored 1.00 on
    all of it. Real machine metrics (fev-bench boomlet, redset) are exactly this kind of series.

    Here: period log-uniform in [2, min(L/4, 1024)] (integer for 85% of rows, so the repetition
    is exact), one of seven shapes, a quarter of rows are the sum of two patterns with different
    periods, optional slow amplitude drift / level trend, and half of the rows carry NO noise.
    The family is in PASSTHROUGH_FAMILIES: no seasonality / noise / corruption layer touches it."""
    t = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)
    p_hi = float(max(2.0, min(L / 4.0, 1024.0)))
    M = int(p_hi) + 2                                   # width of the shape tables

    def one(n):
        P = torch.exp(_u(n, 1, device, math.log(2.0), math.log(p_hi)))
        kind = torch.randint(0, 7, (n, 1), device=device)
        integer = (kind >= 2) | _bern(n, device, 0.7)
        P = torch.where(integer, P.round().clamp(min=2.0), P)
        ph = _u(n, 1, device, 0.0, 1.0) * P
        ph = torch.where(integer, ph.floor(), ph)
        pos = torch.remainder(t + ph, P)                # position inside the period, in steps
        u = (pos / P).clamp(0.0, 1.0)                   # phase in [0, 1)
        # 0 sawtooth: (curved) ramp and instant reset, up or down
        saw = 2.0 * u.pow(torch.exp(_u(n, 1, device, -1.0, 1.0))) - 1.0
        saw = torch.where(_bern(n, device, 0.5), saw, -saw)
        # 1 square wave with a random duty cycle
        sq = torch.where(u < torch.minimum(_u(n, 1, device, 0.05, 0.6), 1.0 - 1.0 / P - 1e-4), 1.0, -1.0)
        # 2 spike train: 1-3 steps high once per period, flat otherwise
        width = torch.minimum(torch.randint(1, 4, (n, 1), device=device).float(), (P - 1.0).clamp(min=1.0))
        spike = (pos < width).float()
        # 3-6 table shapes: one value per step of the period, looked up exactly
        k = torch.arange(M, device=device, dtype=torch.float32).unsqueeze(0)
        rough = torch.randn(n, M, device=device)                                   # 3 any shape at all
        walk = torch.randn(n, M, device=device).cumsum(1)                          # 4 smooth, jump at the wrap
        end = walk.gather(1, (P - 1.0).long().clamp(0, M - 1))
        bridge = walk - k / (P - 1.0).clamp(min=1.0) * end                         #   ... or continuous at the wrap
        smooth = torch.where(_bern(n, device, 0.5), walk, bridge)
        jump = (torch.rand(n, M, device=device) < _u(n, 1, device, 1.0, 6.0) / P).long()
        jump[:, 1] = 1                                                             #   (always at least two levels)
        seg = jump.cumsum(1).clamp(max=63)
        stair = torch.randn(n, 64, device=device).gather(1, seg)                   # 5 staircase: a few held levels
        sparse = torch.randn(n, M, device=device).abs() * (torch.rand(n, M, device=device)
                                                           < _u(n, 1, device, 1.0, 4.0) / P).float()
        sparse[:, 0] = 1.0                                                         # 6 a few spikes of fixed heights
        tab = torch.where(kind == 3, rough, torch.where(kind == 4, smooth, torch.where(kind == 5, stair, sparse)))
        tmpl = tab.gather(1, pos.long().clamp(0, M - 1))
        y = torch.where(kind == 0, saw, torch.where(kind == 1, sq, torch.where(kind == 2, spike, tmpl)))
        return (y - y.mean(1, keepdim=True)) / (y.std(1, keepdim=True) + 1e-6)

    y = one(B)
    y = y + torch.where(_bern(B, device, 0.25), _u(B, 1, device, 0.3, 1.0), torch.zeros(B, 1, device=device)) * one(B)
    tt = t / max(L - 1, 1) - 0.5
    zero = torch.zeros(B, 1, device=device)
    y = y * (1.0 + torch.where(_bern(B, device, 0.25), _u(B, 1, device, -0.8, 0.8), zero) * tt)       # amplitude drift
    y = y + torch.where(_bern(B, device, 0.25), torch.randn(B, 1, device=device) * _u(B, 1, device, 0.5, 3.0), zero) * tt
    noise = torch.where(_bern(B, device, 0.5), zero, torch.exp(_u(B, 1, device, math.log(0.003), math.log(0.3))))
    return _one_off(y + noise * torch.randn(B, L, device=device), device)


def f_repeat_week(B, L, device):
    """DAY-TYPE REPEAT: a daily shape that repeats, with a DIFFERENT shape on some days of the week.

    Why: after the repeat fine-tune, the tasks furthest behind Chronos-2 on fev-bench (M_DENSE,
    entsoe, ercot) and on GIFT-Eval (m_dense, electricity, us_births, m4_hourly) are load, traffic
    and births series: a sharp daily shape, another shape at the weekend, a level that wanders
    from day to day. f_repeat has one shape per period; daily_weekly has day types but only
    smooth 6-harmonic profiles on five fixed sampling rates, and was never switched on.

    Here: D days per week (7 for 75% of rows, else 2-14), any number of steps per day (24 / 48 /
    96 / 144 / 288 when at least two weeks fit, else anything from 1 up; 1 = daily data with a
    day-of-week pattern), the last 1-2 days of the week use their own shape (related to the
    ordinary day's or not), six shapes (peaks, plateau, bell on a flat floor, random curve, rough
    table, staircase). 40% of rows are EXACT: the week repeats with nothing random. The rest get
    a day-to-day amplitude and level (AR(1) over days), rare holidays, and noise. "Solar" rows are
    a bell with an exactly flat night and a different height every day.
    The family is in PASSTHROUGH_FAMILIES: no seasonality / noise / corruption layer touches it."""
    Mx = int(max(1, min(512, L // 4)))
    M = Mx + 2                                                       # width of the shape tables
    tl = torch.arange(L, device=device).unsqueeze(0)
    zero = torch.zeros(B, 1, device=device)
    D = torch.where(_bern(B, device, 0.75), torch.full((B, 1), 7, device=device),
                    torch.randint(2, 15, (B, 1), device=device))     # days per week
    cap = torch.div(torch.full_like(D, L), 2 * D, rounding_mode="floor").clamp(min=1, max=Mx)   # two weeks must fit
    real = _pick(B, [24.0, 48.0, 96.0, 144.0, 288.0], device).long()
    free = torch.exp(_u(B, 1, device, 0.0, 1.0) * torch.log(cap.float())).round().long().clamp(min=1)
    ppd = torch.where((real <= cap) & _bern(B, device, 0.65), real, torch.minimum(free, cap))    # steps per day
    pf = ppd.float()
    tt = tl + torch.floor(_u(B, 1, device, 0.0, 1.0) * (D * ppd).float()).long()                  # random phase in the week
    dayl = torch.div(tt, ppd, rounding_mode="floor")                 # (B, L) index of the day
    tod = tt - dayl * ppd                                            # step inside the day
    dow = torch.remainder(dayl, D)                                   # day of the week
    nd = L + 16                                                      # static bound on the day index
    exact = _bern(B, device, 0.4)                                    # nothing random from day to day
    k = torch.arange(M, device=device, dtype=torch.float32).unsqueeze(0)
    u = (k / pf).clamp(max=1.0)                                      # time of day in [0, 1)
    inday = (k < pf).float()

    def shape():
        kind = torch.randint(0, 6, (B, 1), device=device)
        bumps = torch.zeros(B, M, device=device)                     # 0: one to three peaks (rush hours)
        nb = torch.randint(1, 4, (B, 1), device=device)
        for j in range(3):
            c = _u(B, 1, device, 0.1, 0.9)
            w = torch.exp(_u(B, 1, device, math.log(0.02), math.log(0.2)))
            dist = torch.remainder(u - c + 0.5, 1.0) - 0.5
            bumps = bumps + (nb > j).float() * _u(B, 1, device, 0.3, 1.0) * torch.exp(-0.5 * (dist / w) ** 2)
        lo = _u(B, 1, device, 0.1, 0.5)                              # 1: plateau (working hours), one or two levels
        hi = lo + _u(B, 1, device, 0.15, 0.45)
        mid = lo + (hi - lo) * _u(B, 1, device, 0.2, 0.8)
        plat = ((u >= lo) & (u < hi)).float() \
            + _u(B, 1, device, -0.6, 0.6) * (_bern(B, device, 0.5) & (u >= mid) & (u < hi)).float()
        s0 = _u(B, 1, device, 0.15, 0.35)                            # 2: bell between sunrise and sunset, flat outside
        s1 = _u(B, 1, device, 0.6, 0.85)
        bell = torch.sin(math.pi * ((u - s0) / (s1 - s0)).clamp(0.0, 1.0)).clamp(min=0.0).pow(torch.exp(_u(B, 1, device, -0.5, 1.0)))
        walk = torch.randn(B, M, device=device).cumsum(1)            # 3: random curve, continuous at midnight
        bridge = walk - k / (pf - 1.0).clamp(min=1.0) * walk.gather(1, (ppd - 1).clamp(0, M - 1))
        rough = torch.randn(B, M, device=device)                     # 4: any value at any time of day
        jump = (torch.rand(B, M, device=device) < _u(B, 1, device, 1.0, 6.0) / pf).long()
        jump[:, 1] = 1
        stair = torch.randn(B, 64, device=device).gather(1, jump.cumsum(1).clamp(max=63))    # 5: a few held levels
        tab = torch.where(kind == 0, bumps, torch.where(kind == 1, plat, torch.where(kind == 2, bell,
              torch.where(kind == 3, bridge, torch.where(kind == 4, rough, stair)))))
        mu = (tab * inday).sum(1, keepdim=True) / pf
        sd = (((tab - mu) ** 2 * inday).sum(1, keepdim=True) / pf).sqrt()
        return torch.where(sd > 1e-4, (tab - mu) / (sd + 1e-6), torch.zeros_like(tab)), kind

    T0, k0 = shape()
    floored = k0 <= 2                                                # shapes that sit on a floor
    solar = (k0 == 2) & (ppd >= 8) & _bern(B, device, 0.6)           # exact flat night, a different peak every day
    floor_ = torch.where(floored, (T0 + (1.0 - inday) * 1e9).min(1, keepdim=True).values, zero)

    def other():                                                     # the shape of a special day
        N, _ = shape()
        mix = torch.where(_bern(B, device, 0.5), _u(B, 1, device, 0.5, 0.95), zero)       # related to T0, or not
        return (mix * T0 + (1.0 - mix * mix).sqrt() * N) * _u(B, 1, device, 0.3, 1.2) + _u(B, 1, device, -1.5, 0.5)

    tab = torch.stack([T0, other(), other()], 1).reshape(B, 3 * M)
    n_off = torch.minimum(torch.randint(1, 3, (B, 1), device=device), D - 1)              # 1-2 special days end the week
    rel = dow - (D - n_off)
    typ = (rel >= 0).long() + ((rel == 1) & _bern(B, device, 0.4)).long()                 # 0 ordinary, 1 / 2 special
    hol = torch.rand(B, nd, device=device) < torch.where(exact, zero, _u(B, 1, device, 0.0, 0.03))
    typ = torch.where(hol.gather(1, dayl) & (typ == 0), torch.ones_like(typ), typ)        # a holiday looks like a weekend day
    typ = torch.where(solar, torch.zeros_like(typ), typ)
    base = tab.gather(1, typ * M + tod)

    on = lambda p: _bern(B, device, p) & ~exact
    amp_sd = torch.where(on(0.6), torch.exp(_u(B, 1, device, math.log(0.02), math.log(0.4))), zero)
    amp_sd = torch.where(solar & ~exact, _u(B, 1, device, 0.2, 0.9), amp_sd)
    da = _ar1_filter(B, nd, device, _u(B, 1, device, 0.0, 0.9))
    amp = torch.exp((amp_sd * da / (da.std(1, keepdim=True) + 1e-6)).clamp(-3.0, 3.0)).gather(1, dayl)
    lvl_sd = torch.where(on(0.5) & ~solar, torch.exp(_u(B, 1, device, math.log(0.02), math.log(0.5))), zero)
    dl = _ar1_filter(B, nd, device, _u(B, 1, device, 0.3, 0.97))
    lvl = (lvl_sd * dl / (dl.std(1, keepdim=True) + 1e-6)).gather(1, dayl)
    dow_sd = torch.where(ppd <= 3, torch.ones_like(zero),
                         torch.where(_bern(B, device, 0.4), _u(B, 1, device, 0.05, 0.5), zero))
    dowl = (torch.randn(B, 14, device=device) * torch.where(solar, zero, dow_sd)).gather(1, dow)   # a level per day of the week
    tn = tl.float() / max(L - 1, 1) - 0.5
    amp = amp * (1.0 + torch.where(_bern(B, device, 0.25), _u(B, 1, device, -0.8, 0.8), zero) * tn)     # slow amplitude drift
    y = floor_ + amp * (base - floor_) + lvl + dowl
    y = y + torch.where(_bern(B, device, 0.25) & ~solar,
                        torch.randn(B, 1, device=device) * _u(B, 1, device, 0.5, 3.0), zero) * tn       # level trend
    nz = torch.where(on(0.7), torch.exp(_u(B, 1, device, math.log(0.003), math.log(0.6))), zero)
    e = torch.randn(B, L, device=device)
    mult = (solar | _bern(B, device, 0.3)) & floored                 # noise that grows with the height above the floor
    return _one_off(torch.where(mult, floor_ + (y - floor_) * torch.exp((nz * e).clamp(-3.0, 3.0)), y + nz * e), device)


def _one_off(y, device):
    """ONE-OFF ANOMALIES for the repeating families: something that happens ONCE and does not come back.

    Why: every series of the repeat families was a pure repetition, so the model learned that
    whatever it saw once at some place in a cycle returns at that place in every cycle. On
    LOOP_SEATTLE_5T ft2_main replays one dip of the history every day (+11% error against ft_repeat).

    For a share _ONE_OFF_P of the rows, 1-2 anomalies at random places, each one of:
      outage  the value falls to the series' floor for a while
      spike   1-3 steps far above or below
      stuck   the value is held (a frozen sensor)
      quiet   a stretch scaled towards the floor (a quiet day) or away from it (a busy one)
    Widths are log-uniform from 1 step to L/16 (at most 256). With _ONE_OFF_P = 0 the series is
    returned untouched and no random number is drawn."""
    B, L = y.shape
    if _state._ONE_OFF_P <= 0 or L < 64:
        return y
    t = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)
    lo = y.min(1, keepdim=True).values
    sd = y.std(1, keepdim=True) + 1e-6
    gate = _bern(B, device, _state._ONE_OFF_P)
    n_a = torch.where(_bern(B, device, 0.35), 2, 1)
    w_hi = math.log(max(2.0, min(L / 16.0, 256.0)))
    for j in range(2):
        act = gate & (n_a > j)
        w = torch.exp(_u(B, 1, device, 0.0, w_hi)).floor().clamp(min=1.0)
        t0 = torch.floor(_u(B, 1, device, 0.03, 0.97) * L)
        kind = torch.randint(0, 4, (B, 1), device=device)
        w = torch.where(kind == 1, w.clamp(max=3.0), w)
        inside = (t >= t0) & (t < t0 + w)
        outage = (lo - _u(B, 1, device, 0.0, 0.5) * sd * _bern(B, device, 0.3).float()).expand(B, L)
        spike = y + torch.where(_bern(B, device, 0.5), 1.0, -1.0) * _u(B, 1, device, 2.0, 6.0) * sd
        stuck = y.gather(1, t0.long().clamp(0, L - 1)).expand(B, L)
        quiet = lo + (y - lo) * torch.where(_bern(B, device, 0.7), _u(B, 1, device, 0.1, 0.7), _u(B, 1, device, 1.3, 2.2))
        new = torch.where(kind == 0, outage, torch.where(kind == 1, spike, torch.where(kind == 2, stuck, quiet)))
        y = torch.where(act & inside, new, y)
    return y


# real long cycles, in steps (the yearly ones three times: daily data is where the evidence is)
_LONG_CAL = [12, 24, 48, 52, 52, 96, 168, 336, 364, 365, 365, 365, 672, 1008, 2016]


def f_long_cycle(B, L, device, _debug=False):
    """LONG CYCLE SEEN ONCE OR TWICE: what happened at this place one long period ago happens again.

    Why: daily data has a yearly cycle and usually 1-4 years of history. Every seasonal pattern in
    training so far was seen at least three times (_pick_period needs three cycles, f_repeat four),
    so the model never had to use "the same dates last year" from ONE earlier occurrence. Measured
    on five daily fev-bench tasks: when the history grows from 273 to 400 days Chronos-2's error
    drops 16%, ours 3%; with under a year of history we are level with it on three of the five.

    Here the generated window holds 2.2 - 4.5 cycles of a long period P (any whole number of steps;
    for 35% of rows a real one when it fits 2-6 times: 365 / 364 days, 52 weeks, 168 hours, ...);
    the model sees a crop of half to all of it, so 1.1 - 6 cycles. A series is the sum of
      a short cycle inside the long one (60% of rows; period 7 for half of them, else 2-30: the
        week in daily data), either any shape or flat with 1-2 different days,
      a smooth shape of the long cycle (70%; three harmonics: the seasons),
      1-8 DATED EVENTS per cycle at fixed places: a dip or a peak 1 to ~P/15 steps wide, a quarter
        with an opposite move just before (the rush before a holiday); 30% of the events are
        CLOSURES that pull the series down to a floor instead of by a fixed amount,
      a SEASON (35%): one stretch of every cycle with its own level and its own short-cycle size.
    35% of rows are EXACT: every cycle is the same. The rest get events whose size changes a little
    from cycle to cycle, 15% of events that MOVE by a few steps (Easter), a wandering level and
    noise. All rows may have growth (level trend, size of everything).
    The family is in PASSTHROUGH_FAMILIES: no seasonality / noise / corruption layer touches it.
    _debug=True also returns (P, exact, the part that must be identical one cycle later, closures)."""
    if L < 64:
        y = f_repeat(B, L, device)
        return (y, None, None, None, None) if _debug else y
    t = torch.arange(L, device=device).unsqueeze(0)                   # (1, L)
    zero = torch.zeros(B, 1, device=device)
    one = torch.ones(B, 1, device=device)
    P = torch.round(L / torch.exp(_u(B, 1, device, math.log(2.2), math.log(4.5)))).clamp(min=12.0)   # (B, 1), whole steps
    cal = [p for p in _LONG_CAL if 2.0 <= L / p <= 6.0]             # real long cycles that fit this window 2-6 times
    if cal:
        P = torch.where(_bern(B, device, 0.35), _pick(B, [float(p) for p in cal], device), P)
    Pl = P.long()
    tt = t + torch.floor(_u(B, 1, device, 0.0, 1.0) * P).long()       # (B, L): random phase in the cycle
    NC = 8                                                            # static bound on the cycle index (at most 6)
    cyc = torch.div(tt, Pl, rounding_mode="floor").clamp(max=NC - 1)  # which cycle
    pos = (tt - torch.div(tt, Pl, rounding_mode="floor") * Pl).float()   # place inside the cycle, in steps
    exact = _bern(B, device, 0.35)                                    # nothing random from cycle to cycle

    # 1. the short cycle (the week): a table of ps values
    ps = torch.where(_bern(B, device, 0.5), torch.full((B, 1), 7, device=device), torch.randint(2, 31, (B, 1), device=device))
    ps = torch.minimum(ps, torch.div(Pl, 6, rounding_mode="floor").clamp(min=2))
    k32 = torch.arange(32, device=device).unsqueeze(0)
    stab = torch.randn(B, 32, device=device)
    off_days = (k32 >= (ps - torch.randint(1, 3, (B, 1), device=device)).clamp(min=1)) & (k32 < ps)
    stab = torch.where(_bern(B, device, 0.4), torch.where(off_days, _u(B, 1, device, -2.5, -0.5), zero) + 0.1 * stab, stab)
    inp = (k32 < ps).float()
    mu = (stab * inp).sum(1, keepdim=True) / ps.float()
    sdv = (((stab - mu) ** 2 * inp).sum(1, keepdim=True) / ps.float()).sqrt()
    stab = (stab - mu) / (sdv + 1e-6)
    short = stab.gather(1, torch.remainder(t + torch.randint(0, 31, (B, 1), device=device), ps))    # (B, L)
    a_s = torch.where(_bern(B, device, 0.6) & (Pl >= 24), _u(B, 1, device, 0.3, 1.5), zero)

    # 2. the smooth shape of the long cycle (the seasons)
    decay = _u(B, 1, device, 0.5, 2.0)
    smooth = torch.zeros(B, L, device=device)
    for k in range(1, 4):
        ang = 2 * math.pi * k * pos / P
        smooth = smooth + torch.full((1, 1), float(k), device=device) ** (-decay) * (
            torch.randn(B, 1, device=device) * torch.cos(ang) + torch.randn(B, 1, device=device) * torch.sin(ang))
    smooth = smooth / (smooth.std(1, keepdim=True) + 1e-6)
    a_y = torch.where(_bern(B, device, 0.7), _u(B, 1, device, 0.3, 1.5), zero)

    # 3. dated events: the same thing at the same place of every cycle
    NE = 8
    n_ev = torch.randint(1, NE + 1, (B, 1), device=device)
    ev = torch.zeros(B, L, device=device)                             # events that add or subtract
    dip = torch.zeros(B, L, device=device)                            # closures: share of the way down to the floor
    for j in range(NE):
        on_j = n_ev > j
        c = torch.floor(_u(B, 1, device, 0.0, 1.0) * P)               # where in the cycle it starts
        wk = torch.randint(0, 5, (B, 1), device=device)               # 40% one step, 40% 2-7 steps, 20% a few % of the cycle
        w = torch.where(wk <= 1, one, torch.where(wk <= 3, torch.randint(2, 8, (B, 1), device=device).float(),
                                                  torch.floor(_u(B, 1, device, 0.02, 0.07) * P).clamp(min=2.0)))
        w = torch.minimum(w, torch.floor(P / 6.0).clamp(min=1.0))
        mv = torch.where(_bern(B, device, 0.15) & ~exact, torch.floor(_u(B, 1, device, 0.01, 0.04) * P), zero)
        shift = torch.round(mv * torch.randn(B, NC, device=device)).gather(1, cyc)        # a date that moves (Easter)
        size = (1.0 + torch.where(_bern(B, device, 0.5) & ~exact, _u(B, 1, device, 0.05, 0.3), zero)
                * torch.randn(B, NC, device=device)).clamp(0.3, 1.7).gather(1, cyc)       # not the same size every cycle
        d = torch.remainder(pos - c - shift + 0.5 * P, P) - 0.5 * P   # steps since the start of the event (circular)
        box = ((d >= 0) & (d < w)).float()
        bump = torch.exp(-0.5 * ((d - 0.5 * (w - 1.0)) / (0.3 * w + 0.2)) ** 2) * ((d >= -w) & (d < 2.0 * w)).float()
        kern = torch.where(_bern(B, device, 0.6) | (w <= 2), box, bump)
        lead = ((d >= -torch.ceil(1.5 * w)) & (d < 0)).float()        # the steps just before it
        m = torch.where(_bern(B, device, 0.55), -one, one) * _u(B, 1, device, 1.0, 5.0)
        closure = _bern(B, device, 0.3)
        add = m * kern + torch.where(_bern(B, device, 0.25), -0.3 * m, zero) * lead
        ev = ev + (on_j & ~closure).float() * size * add
        dip = torch.maximum(dip, (on_j & closure).float() * _u(B, 1, device, 0.5, 1.0) * kern)

    # 4. a season: one stretch of every cycle with its own level and short-cycle size (summer)
    in_s = (torch.remainder(pos - torch.floor(_u(B, 1, device, 0.0, 1.0) * P), P)
            < torch.floor(_u(B, 1, device, 0.04, 0.3) * P).clamp(min=2.0)) & _bern(B, device, 0.35)
    s_lvl = torch.where(in_s, _u(B, 1, device, -2.0, 2.0), zero)
    s_amp = torch.where(in_s, _u(B, 1, device, 0.2, 1.8), one)

    recur = a_y * smooth + ev + s_lvl                                  # identical one cycle later in the exact rows
    y = a_s * s_amp * short + recur
    tn = t.float() / max(L - 1, 1) - 0.5
    y = y * (1.0 + torch.where(_bern(B, device, 0.3), _u(B, 1, device, -0.6, 0.6), zero) * tn)        # everything grows / shrinks
    y = y + torch.where(_bern(B, device, 0.4), torch.randn(B, 1, device=device) * _u(B, 1, device, 0.5, 2.5), zero) * tn
    lv = _ar1_filter(B, L, device, _u(B, 1, device, 0.9, 0.995))
    lv_sd = torch.where(_bern(B, device, 0.4) & ~exact, torch.exp(_u(B, 1, device, math.log(0.05), math.log(0.6))), zero)
    y = y + lv_sd * lv / (lv.std(1, keepdim=True) + 1e-6)             # a level that wanders
    y = y + torch.where(exact, zero, torch.exp(_u(B, 1, device, math.log(0.01), math.log(0.5)))) * torch.randn(B, L, device=device)
    floor_ = y.min(1, keepdim=True).values - _u(B, 1, device, 0.2, 1.5)
    y = floor_ + (y - floor_) * (1.0 - dip)                           # closures pull it down to the floor
    y = _one_off(y, device)
    return (y, P, exact, recur, dip) if _debug else y


# <<< ft3_core.py
