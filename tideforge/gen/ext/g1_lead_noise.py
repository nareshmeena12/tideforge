"""G1.2: known covariates handed over with forecast error that grows with lead time (owner 2026-10-28: option a).

Not a generator switch: ft3's causal decoder has no single forecast origin per generated block. The origin is the start of
the CPM-hidden span (ft3_core PackedBatcher._cpm_blank, L2779-2804; known lanes are never hidden, so inside the span the
model predicts targets while seeing the known covariates exactly). T3 wires lead_noise into the batcher after _cpm_blank,
on the known lanes of cells that have a span (configs/exp/G1.2.yaml, train: section).

    sigma0 = draw_sigma0(n, gen)                      # 0 for a share 1 - p_noisy of cells, else U(0.05, 0.5)
    x = lead_noise(x, origin, sigma0, gen)            # x (n, T); origin (n,) index of the first hidden point, -1 = none

The noise is a random walk that starts at the origin, in units of the lane's standard deviation before the origin:
    eps_t = 0 for t < origin;  Var(eps_t) = (sigma0 * sd)^2 * (t - origin + 1) / step    for t >= origin
so its variance grows linearly with the lead; `step` (default 32, one patch) sets the lead at which the sd reaches
sigma0 * sd.
"""
from __future__ import annotations

import torch

P_NOISY, SIGMA0 = 0.5, (0.05, 0.5)


def draw_sigma0(n: int, gen: torch.Generator, p_noisy: float = P_NOISY, lo: float = SIGMA0[0], hi: float = SIGMA0[1],
                device="cpu") -> torch.Tensor:
    noisy = torch.rand(n, generator=gen, device=device) < p_noisy
    s = lo + (hi - lo) * torch.rand(n, generator=gen, device=device)
    return torch.where(noisy, s, torch.zeros_like(s))


def lead_noise(x: torch.Tensor, origin: torch.Tensor, sigma0: torch.Tensor, gen: torch.Generator, step: int = 32) -> torch.Tensor:
    """x (n, T) with origin (n,) long (-1 = no origin: unchanged) and sigma0 (n,). Returns x + eps (see the module doc)."""
    n, T = x.shape
    t = torch.arange(T, device=x.device).unsqueeze(0)
    o = origin.to(x.device).view(n, 1)
    after = (o >= 0) & (t >= o)
    before = (o >= 0) & (t < o)
    xb = torch.where(before, x, torch.zeros_like(x))
    cnt = before.sum(1, keepdim=True).clamp(min=1)
    mean = xb.sum(1, keepdim=True) / cnt
    sd = torch.sqrt((torch.where(before, x - mean, torch.zeros_like(x)) ** 2).sum(1, keepdim=True) / cnt) + 1e-6
    z = torch.randn((n, T), generator=gen, device=x.device) * after
    eps = z.cumsum(1) * (sigma0.to(x.device).view(n, 1) * sd / step ** 0.5)
    return x + torch.where(after, eps, torch.zeros_like(eps))
