"""G5.2 amfm, G5.3 beats, G5.4 irregular_cycles: quasi-periodic families (probe P5; G5.4 also P3).

Each draws one seed from the global stream, then uses its own generator (_family.own_generator). Flags: no additive layers
(allow_add False, like tsi), the noise / support / corruption / scale layers apply. Ranges are uniform per row unless stated.

  amfm      x = A(t) s(phi(t)) + e.  P0: a bank period with >= 3 cycles in the window (50%) or L / U(3, 12);
            phi(t) = 2 pi sum_{u <= t} (1 + delta w(u)) / P0 + phi0, w a slow AR(1) (rho in [0.995, 0.9995]) standardised per
            row, delta in [0.01, 0.05] (the period wanders 1-5%); A = exp(sigma_A w2), w2 slow AR(1) (rho in [0.99, 0.999]),
            sigma_A in [0.1, 0.5]; s = sum_{k <= K} k^-b cos(k phi + ph_k), K in 1..5, b in [0.5, 1.5]; e AR(1)
            (rho in [0, 0.7]) with sd 2-15% of the signal's.
  beats     tide rows (40%): h in {1, 2, 4} steps per hour; M2 12.4206 h, S2 12.00 h, a_S2 / a_M2 in [0.3, 1], plus K1
            23.9345 h and O1 25.8193 h at [0.1, 1] of a_M2 each (diurnal inequality); spring-neap 14.77 days comes from M2 / S2.
            Other rows: P1 = L / U(10, 60) (>= 3 steps), P2 = r P1, r in [max(1.01, 1 + P1 / L), max(1.1, 1 + P1 / L)] so one
            beat P1 P2 / (P2 - P1) fits in the window; a2 / a1 in [0.3, 1]. Noise as amfm.
  irregular_cycles  mean cycle length mu = L / U(3, 12); each cycle's length mu U(0.78, 1.22) (sunspot-like, 50%: 9-14
            years) or mu U(0.44, 1.56) (ENSO-like: 2-7 years); window starts inside the first cycle; within a cycle (u in
            [0, 1)) a rise sin^2 over a fraction rho in [0.25, 0.5], then a slower fall cos^2; amplitude per cycle
            exp(sigma N(0, 1)), sigma in [0.2, 0.5]. Sunspot rows: amp x shape (>= 0); ENSO rows: amp x (2 shape - 1).
            Noise as amfm. Vectorised: cycle boundaries are cumulative sums, each step finds its cycle by searchsorted.
"""
from __future__ import annotations

import math

import torch

from tideforge.gen import _state
from tideforge.gen.ext import _family
from tideforge.gen.ext import g1_common as C
from tideforge.gen.ext._rand import bern, randint, u
from tideforge.gen.switches import Switch, register

TIDE = dict(M2=12.4206, S2=12.0, K1=23.9345, O1=25.8193)
SPRING_NEAP_H = 1.0 / (1.0 / TIDE["S2"] - 1.0 / TIDE["M2"])          # 354.4 h = 14.77 days


def _std(x):
    return (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6)


def _noise(g, sig, dev):
    n, L = sig.shape
    e = C.ar1(g, (n, L), u(g, (n, 1), 0.0, 0.7, dev), dev)
    return sig + e * u(g, (n, 1), 0.02, 0.15, dev) * (sig.std(1, keepdim=True) + 1e-6)


def _bank_or_cycles(g, n, L, lo, hi, dev):
    bank = [p for p in _state._PERIOD_BANK_ACTIVE if 3 * p <= L]
    P = L / u(g, (n, 1), lo, hi, dev)
    if bank:
        b = torch.tensor(bank, dtype=torch.float32, device=dev)[randint(g, 0, len(bank) - 1, (n, 1), dev)]
        P = torch.where(bern(g, (n, 1), 0.5, dev), b, P)
    return P


def amfm_parts(g, n, L, dev):
    P0 = _bank_or_cycles(g, n, L, 3.0, 12.0, dev)
    delta = u(g, (n, 1), 0.01, 0.05, dev)
    w = _std(C.ar1(g, (n, L), u(g, (n, 1), 0.995, 0.9995, dev), dev))
    phi = 2 * math.pi * torch.cumsum((1 + delta * w) / P0, 1) + u(g, (n, 1), 0, 2 * math.pi, dev)
    A = torch.exp(u(g, (n, 1), 0.1, 0.5, dev) * _std(C.ar1(g, (n, L), u(g, (n, 1), 0.99, 0.999, dev), dev)))
    K = randint(g, 1, 5, (n, 1), dev)
    b = u(g, (n, 1), 0.5, 1.5, dev)
    s = torch.zeros(n, L, device=dev)
    for k in range(1, 6):
        s = s + (k <= K).float() * k ** (-b) * torch.cos(k * phi + u(g, (n, 1), 0, 2 * math.pi, dev))
    return A * s, dict(P0=P0, delta=delta, w=w, A=A, phi=phi)


def amfm(n, L, device):
    g = _family.own_generator(device)
    x, _ = amfm_parts(g, n, L, torch.device(device))
    return _noise(g, x, torch.device(device))


def beats_parts(g, n, L, dev):
    t = torch.arange(L, device=dev, dtype=torch.float32).view(1, L)
    tide = bern(g, (n, 1), 0.4, dev)
    h = torch.tensor([1.0, 2.0, 4.0], device=dev)[randint(g, 0, 2, (n, 1), dev)]
    P1 = (L / u(g, (n, 1), 10.0, 60.0, dev)).clamp(min=3.0)
    rlo = torch.clamp(1 + P1 / L, min=1.01)
    r = rlo + (torch.clamp(1 + P1 / L, min=1.1) - rlo) * u(g, (n, 1), 0, 1, dev)
    P1 = torch.where(tide, TIDE["M2"] * h, P1)
    P2 = torch.where(tide, TIDE["S2"] * h, r * P1)
    a2 = u(g, (n, 1), 0.3, 1.0, dev)
    x = torch.cos(2 * math.pi * t / P1 + u(g, (n, 1), 0, 2 * math.pi, dev))
    x = x + a2 * torch.cos(2 * math.pi * t / P2 + u(g, (n, 1), 0, 2 * math.pi, dev))
    for c in ("K1", "O1"):
        d = u(g, (n, 1), 0.1, 1.0, dev) * torch.cos(2 * math.pi * t / (TIDE[c] * h) + u(g, (n, 1), 0, 2 * math.pi, dev))
        x = x + tide.float() * d
    return x, dict(tide=tide, h=h, P1=P1, P2=P2, a2=a2)


def beats(n, L, device):
    g = _family.own_generator(device)
    x, _ = beats_parts(g, n, L, torch.device(device))
    return _noise(g, x, torch.device(device))


KMAX = 32


def irregular_parts(g, n, L, dev):
    mu = L / u(g, (n, 1), 3.0, 12.0, dev)
    sun = bern(g, (n, 1), 0.5, dev)
    lo = torch.where(sun, 0.78, 0.44)
    hi = torch.where(sun, 1.22, 1.56)
    lens = mu * (lo + (hi - lo) * torch.rand((n, KMAX), generator=g, device=dev))
    ends = torch.cumsum(lens, 1) - lens[:, :1] * u(g, (n, 1), 0, 1, dev)      # the window starts inside cycle 0
    t = torch.arange(L, device=dev, dtype=torch.float32).view(1, L).expand(n, L).contiguous()
    k = torch.searchsorted(ends, t, right=True).clamp(max=KMAX - 1)
    start = torch.gather(ends, 1, k) - torch.gather(lens, 1, k)
    uu = ((t - start) / torch.gather(lens, 1, k)).clamp(0, 1)
    rho = u(g, (n, 1), 0.25, 0.5, dev)
    shape = torch.where(uu < rho, torch.sin(0.5 * math.pi * uu / rho) ** 2,
                        torch.cos(0.5 * math.pi * (uu - rho) / (1 - rho)) ** 2)
    amp = torch.exp(u(g, (n, 1), 0.2, 0.5, dev) * torch.randn((n, KMAX), generator=g, device=dev))
    a = torch.gather(amp, 1, k)
    x = torch.where(sun, a * shape, a * (2 * shape - 1))
    return x, dict(sun=sun, lens=lens, ends=ends, k=k, rho=rho, mu=mu)


def irregular_cycles(n, L, device):
    g = _family.own_generator(device)
    x, _ = irregular_parts(g, n, L, torch.device(device))
    return _noise(g, x, torch.device(device))


def _installer(family, fn):
    def install(p, inst):
        _family.install(inst, family, fn, p, allow_add=False)
    return install


register(Switch("amfm", "G5.2", "AM/FM family: amplitude and period wander", _installer("amfm", amfm),
                dict(share=0.04, mode="add")))
register(Switch("beats", "G5.3", "beats of two close periods; tides with spring-neap and diurnal inequality",
                _installer("beats", beats), dict(share=0.03, mode="add")))
register(Switch("irregular_cycles", "G5.4", "cycles of irregular length, asymmetric rise and fall",
                _installer("irregular_cycles", irregular_cycles), dict(share=0.04, mode="add")))
