"""G4: structural breaks.

  breaks_rate  (G4.2)  the breaks layer at rate p (profile breaks_p; v0 has 0.2 when gen_additions.breaks is on, 0 when
                       off). Turning breaks on adds the layer's global draws, so the whole batch changes.
  breaks_late  (G4.3)  needs breaks on. H1 override of the breaks layer: v0's layer still runs (its own global draws are made),
                       and its output is replaced by the same breaks (port of families._L_breaks,
                       ft3_core) drawn from the switch's own generator with the break time moved to U(lo, hi) x E, where
                       E = L - pred is the part of the block the batcher can crop a context from (the values change before the
                       noise layer, so on CPU later poisson draws can shift, as for H3; ft3's causal decoder:
                       pred 0, so E = L). Where the break lands in the real context depends on the crop: tools and tests
                       measure it with the batcher's own layout sampler (late_share_in_context).
  collapse     (G4.4)  H3 (after structure, before noise): a share p of rows collapses at a time U(0.2, 0.9) x L: the level
                       drops by U(2, 6) sd and the amplitude around the mean shrinks to U(0.1, 0.5) of itself, then both
                       recover with time constant U(0.02, 0.3) x L to a lasting fraction keep ~ U(0.2, 0.8) of the
                       collapse (partial recovery). Passthrough families are restored by v0 after the layers anyway.
                       H3 changes values before the noise layer: on CPU the later poisson draws can shift (switches.py).
"""
from __future__ import annotations

import torch

from tideforge.gen.ext._rand import bern, u
from tideforge.gen.switches import Switch, register


def _install_rate(p, inst):
    inst.set_profile("breaks_p", lambda _: float(p["p"]))


def breaks(x: torch.Tensor, p: float, g: torch.Generator, lo: float = 0.05, hi: float = 0.95, pred: int = 0):
    """families._L_breaks with every draw from g and the break time U(lo, hi) x (L - pred). Returns (x, tb (B, 2) or -1)."""
    B, L = x.shape
    dev = x.device
    E = max(8, L - pred)
    tbs = torch.full((B, 2), -1.0, device=dev)
    if L < 8:
        return x, tbs
    t = torch.arange(L, device=dev, dtype=torch.float32).unsqueeze(0)
    gate = bern(g, (B, 1), p, dev)
    sd = x.std(1, keepdim=True) + 1e-6
    m = x.mean(1, keepdim=True)
    one = torch.ones(B, 1, device=dev)
    n_b = torch.where(bern(g, (B, 1), 0.3, dev), 2 * one, one)
    for j in range(2):
        active = gate & (n_b > j)
        tb = torch.floor(u(g, (B, 1), lo, hi, dev) * E)
        tbs[:, j] = torch.where(active.squeeze(1), tb.squeeze(1), tbs[:, j])
        after = (t >= tb).float()
        dt = (t - tb).clamp(min=0.0)
        kind = torch.randint(0, 5, (B, 1), generator=g, device=dev)
        sgn = torch.where(bern(g, (B, 1), 0.5, dev), one, -one)
        level = sgn * u(g, (B, 1), 0.5, 3.0, dev) * sd * after
        slope = torch.randn((B, 1), generator=g, device=dev) * u(g, (B, 1), 0.5, 4.0, dev) * sd * dt / L
        amp = (x - m) * (torch.exp(u(g, (B, 1), -1.2, 1.2, dev)) - 1.0) * after
        tau = u(g, (B, 1), 0.005, 0.3, dev) * L + 1.0
        keep = torch.where(bern(g, (B, 1), 0.5, dev), torch.zeros_like(one), u(g, (B, 1), 0.2, 1.0, dev))
        ssgn = torch.where(bern(g, (B, 1), 0.8, dev), -one, one)
        shock = ssgn * u(g, (B, 1), 1.0, 5.0, dev) * sd * after * ((1 - keep) * torch.exp(-dt / tau) + keep)
        delta = torch.where(kind == 0, level, torch.where(kind == 1, slope, torch.where(
            kind == 2, amp, torch.where(kind == 3, shock, level + amp))))
        x = x + torch.where(active, delta, torch.zeros_like(delta))
    return x, tbs


def _install_late(p, inst):
    def override(args, kwargs, out, ctx):
        x, _, rate = args[0], args[1], (args[2] if len(args) > 2 else kwargs.get("p", 0.2))
        y, _ = breaks(x, float(rate), ctx.rng(), p["lo"], p["hi"], int(p["pred"]))
        return y

    inst.override_layer("breaks", override)


def late_share_in_context(cfg: dict, tb_frac: torch.Tensor, n_layouts: int = 20, lo: float = 0.7, hi: float = 0.9,
                          seed: int = 0) -> dict:
    """Simulated crop: contexts c from ft3's PackedBatcher.sample_layout (cfg = the training config), bucket E = the smallest
    edge >= c, crop start st ~ U{0..E-c} as in next_batch; a break at tb = tb_frac x E (tb_frac drawn as the switch draws it)
    lands at (tb - st) / c of the context. Returns the shares in the last 10-30% (position in [lo, hi)), before / after the
    context, and elsewhere inside it."""
    from tideforge.eval import thirdparty
    C = thirdparty.ft3_core()
    b = C.PackedBatcher(dict(C.CONFIG, **cfg), "cpu", seed=seed)
    c = torch.cat([b.sample_layout()["c"] for _ in range(n_layouts)]).double()
    edges = torch.as_tensor(b.edges, dtype=torch.double)
    E = edges[torch.bucketize(c, edges)]
    g = torch.Generator().manual_seed(seed)
    st = torch.floor(torch.rand(len(c), generator=g, dtype=torch.double) * (E - c + 1))
    tf = tb_frac[torch.randint(0, len(tb_frac), (len(c),), generator=g)].double()
    pos = (torch.floor(tf * E) - st) / c
    inside = (pos >= 0) & (pos < 1)
    late = (pos >= lo) & (pos < hi)
    return dict(n=int(len(c)), late=float(late.double().mean()), outside=float((~inside).double().mean()),
                inside_other=float((inside & ~late).double().mean()))


def collapse(x: torch.Tensor, p: float, g: torch.Generator, passthrough: torch.Tensor | None = None):
    """Returns (x with the collapses, tb (B,) the collapse time or -1)."""
    dev = x.device
    B, L = x.shape
    hit = bern(g, (B, 1), p, dev)
    if passthrough is not None:
        hit = hit & ~passthrough.view(B, 1)
    t = torch.arange(L, device=dev, dtype=x.dtype).unsqueeze(0)
    tb = torch.floor(u(g, (B, 1), 0.2, 0.9, dev) * L)
    sd = x.std(1, keepdim=True) + 1e-6
    m = x.mean(1, keepdim=True)
    drop = u(g, (B, 1), 2.0, 6.0, dev) * sd
    amp = u(g, (B, 1), 0.1, 0.5, dev)
    tau = u(g, (B, 1), 0.02, 0.3, dev) * L + 1.0
    keep = u(g, (B, 1), 0.2, 0.8, dev)
    dt = (t - tb).clamp(min=0.0)
    r = (t >= tb).to(x.dtype) * (keep + (1 - keep) * torch.exp(-dt / tau))       # share of the collapse still in place
    y = m + (x - m) * (1 - (1 - amp) * r) - drop * r
    return torch.where(hit, y, x), torch.where(hit.squeeze(1), tb.squeeze(1), torch.full_like(tb.squeeze(1), -1.0))


def _install_collapse(p, inst):
    def hook(x, ctx):
        return collapse(x, p["p"], ctx.rng(), ctx.passthrough)[0]

    inst.add_hook("after_structure", hook, "structure")


register(Switch("breaks_rate", "G4.2", "breaks layer at rate p", _install_rate, dict(p=0.2)))
register(Switch("breaks_late", "G4.3", "breaks placed in [lo, hi] of the croppable block (needs breaks on)", _install_late,
                dict(lo=0.7, hi=0.9, pred=0)))
register(Switch("collapse", "G4.4", "collapse of level and amplitude with partial recovery (H3)", _install_collapse,
                dict(p=0.05)))
