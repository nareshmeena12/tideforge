"""GLA arm: fla's GatedLinearAttention (flash-linear-attention, pinned 0.5.2)
with its published defaults: key dim = half the width, value dim = width,
gate rank 16, swish output gate with fused RMSNorm, no short convolution.

Training ALWAYS uses fla. This module also holds a small pure-PyTorch
reference (ReferenceGLA and two kernel forms) used ONLY by tests and E1:
  * it mirrors fla's layer parameter for parameter (same names and shapes),
    so weights can be copied into it and outputs compared on the GPU;
  * it lets the CPU tests run on the Windows dev machine, where fla's Triton
    kernels cannot load.
fla note: the layer runs its fused_recurrent kernel whenever T <= 64 tokens,
and its configured kernel (chunk) above that. chunk faults at T = 32: never call
it at T <= 64 (check_chunk_length). E1 follows the same dispatch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

GATE_LOGIT_NORMALIZER = 16     # fla default
PACKED_KERNELS = ("chunk", "fused_recurrent")   # fla kernel for packed (varlen) rows
GNORM_EPS = 1e-5               # fla default (norm_eps)


def fla_available() -> bool:
    try:
        import fla.layers.gla  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 - ImportError, or triton missing on Windows
        return False


# ============================ reference kernels =============================
# q, k, gk: (B, T, H, K); v: (B, T, H, V); gk = per-step log decay (<= 0).
# Output o: (B, T, H, V). Same maths as fla.ops.gla.naive.naive_recurrent_gla.

def gla_recurrent_ref(q, k, v, gk, scale: float | None = None, S0: torch.Tensor | None = None,
                      return_state: bool = False):
    """State recurrence: S_t = diag(exp(gk_t)) S_{t-1} + k_t^T v_t ; o_t = scale * q_t S_t.
    S0 / the returned state: (B, H, K, V)."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    scale = K ** -0.5 if scale is None else scale
    S = q.new_zeros(B, H, K, V) if S0 is None else S0
    out = []
    for t in range(T):
        S = S * gk[:, t].exp()[..., None] + k[:, t, ..., None] * v[:, t, :, None, :]
        out.append((q[:, t, ..., None] * S).sum(-2) * scale)
    o = torch.stack(out, dim=1)
    return (o, S) if return_state else o


def gla_parallel_ref(q, k, v, gk, scale: float | None = None) -> torch.Tensor:
    """Materialised parallel form: o_t = scale * sum_{s<=t} (q_t * exp(G_t - G_s)) . k_s  v_s,
    G = cumsum(gk). O(T^2): for tests only."""
    B, T, H, K = q.shape
    scale = K ** -0.5 if scale is None else scale
    G = gk.cumsum(1)
    diff = G[:, :, None] - G[:, None, :]                       # (B, t, s, H, K)
    causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
    decay = torch.where(causal[None, :, :, None, None], diff,
                        torch.full_like(diff, float("-inf"))).exp()
    A = torch.einsum("bthk,bshk,btshk->bhts", q, k, decay)
    return torch.einsum("bhts,bshv->bthv", A, v) * scale


class ReferenceGLA(nn.Module):
    """Pure-PyTorch mirror of fla.layers.gla.GatedLinearAttention (defaults, fuse_norm=True).
    Parameter names match fla: q_proj, k_proj, v_proj, g_proj, gk_proj.{0,1}, o_proj,
    g_norm_swish_gate.weight. TESTS ONLY."""

    def __init__(self, hidden_size: int, num_heads: int, expand_k: float = 0.5,
                 expand_v: float = 1.0, gate_low_rank_dim: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.gk_proj = nn.Sequential(nn.Linear(hidden_size, gate_low_rank_dim, bias=False),
                                     nn.Linear(gate_low_rank_dim, self.key_dim, bias=True))
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.g_norm_swish_gate = nn.Module()
        self.g_norm_swish_gate.weight = nn.Parameter(torch.ones(self.head_v_dim))
        self.mode = "chunk"            # "chunk" -> parallel form here, "fused_recurrent" -> recurrence

    def _project(self, x):
        B, T, _ = x.shape
        H, dk, dv = self.num_heads, self.head_k_dim, self.head_v_dim
        q = self.q_proj(x).view(B, T, H, dk)
        k = self.k_proj(x).view(B, T, H, dk)
        v = self.v_proj(x).view(B, T, H, dv)
        gk = F.logsigmoid(self.gk_proj(x).view(B, T, H, dk)) / GATE_LOGIT_NORMALIZER
        return q, k, v, gk

    def _output(self, x, o):
        B, T, _ = x.shape
        H, dv = self.num_heads, self.head_v_dim
        g = self.g_proj(x).view(B, T, H, dv)
        rstd = torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + GNORM_EPS)
        o = o * rstd * self.g_norm_swish_gate.weight * g * torch.sigmoid(g)
        return self.o_proj(o.reshape(B, T, H * dv))

    def forward(self, x: torch.Tensor):
        q, k, v, gk = self._project(x)
        kernel = gla_recurrent_ref if self.mode == "fused_recurrent" else gla_parallel_ref
        return self._output(x, kernel(q, k, v, gk)), None, None

    def forward_state(self, x: torch.Tensor, S0: torch.Tensor | None = None):
        """Recurrent pass from state S0 (B, H, K, V); returns (output, final state)."""
        q, k, v, gk = self._project(x)
        o, S = gla_recurrent_ref(q, k, v, gk, S0=S0, return_state=True)
        return self._output(x, o), S


class GLAMixer(nn.Module):
    """Causal GLA over the patch tokens. (B, T, D) -> (B, T, D). Input is already normed;
    no residual is added here. backend="fla" for every run; "reference" for tests only."""

    def __init__(self, d_model: int, n_heads: int, expand_k: float = 0.5, expand_v: float = 1.0,
                 gate_rank: int = 16, backend: str = "fla", packed_kernel: str = "chunk"):
        super().__init__()
        if backend == "fla":
            from fla.layers.gla import GatedLinearAttention
            # layer_idx 0: each mixer streams through its own one-layer cache (E3)
            self.layer = GatedLinearAttention(mode="chunk", hidden_size=d_model, expand_k=expand_k,
                                              expand_v=expand_v, num_heads=n_heads,
                                              gate_low_rank_dim=gate_rank, layer_idx=0)
        elif backend == "reference":
            self.layer = ReferenceGLA(d_model, n_heads, expand_k, expand_v, gate_rank)
        else:
            raise ValueError(f"unknown GLA backend {backend!r}")
        self.backend = backend
        if packed_kernel not in PACKED_KERNELS:
            raise ValueError(f"packed_kernel must be one of {PACKED_KERNELS}")
        self.packed_kernel = packed_kernel
        self.n_heads = n_heads
        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)

    def forward(self, h: torch.Tensor, pack=None) -> torch.Tensor:
        """pack (models.packing.Pack): packed rows (Milestone 4). The rows are flattened to one
        sequence and fla gets cu_seqlens with a boundary at every series start and row start,
        so the recurrent state is reset per series. fla 0.5.2 (layers/gla.py) takes
        cu_seqlens as a keyword, needs batch size 1, and picks its kernel from the FLATTENED
        length (always > 64 here, so `mode`), even for series shorter than 64 tokens; the
        GPU test checks chunk against each series alone at 1..256 tokens, and packed_kernel
        "fused_recurrent" is the fallback if it does not hold. Unpacked (pack=None): the
        ordinary batched call, with the layer's own dispatch."""
        if pack is None:
            return self.layer(h)[0]
        R, P, D = h.shape
        if self.backend == "reference":             # tests only: each series on its own
            flat = h.reshape(1, R * P, D)
            out = torch.cat([self.layer(flat[:, a:b])[0] for a, b in pack.segments()], dim=1)
            return out.view(R, P, -1)
        mode = self.layer.mode
        self.layer.mode = self.packed_kernel
        try:
            y = self.layer(h.reshape(1, R * P, D), cu_seqlens=pack.cu_seqlens)[0]
        finally:
            self.layer.mode = mode
        return y.view(R, P, -1)

    # ---- bookkeeping for the G1 rule and the results rows
    def param_groups(self) -> dict[str, int]:
        groups = {"main": 0, "gate": 0, "norm": 0, "other": 0}
        for name, p in self.layer.named_parameters():
            if name.split(".")[0] in ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj"):
                groups["main"] += p.numel()
            elif name.startswith("gk_proj."):
                groups["gate"] += p.numel()
            elif name.startswith(("g_norm_swish_gate.", "g_norm.")):
                groups["norm"] += p.numel()
            else:
                groups["other"] += p.numel()
        return groups

    def info(self) -> dict:
        dk = self.key_dim // self.n_heads
        dv = self.value_dim // self.n_heads
        return {"key_dim_total": self.key_dim, "head_k_dim": dk, "head_v_dim": dv,
                "state_size": self.n_heads * dk * dv}

    # ---- streaming: a constant-size recurrent state (E3). fla backend: fla's own layer
    # forward with a one-layer cache (state layout [B, H, V, K], fla's state_v_first=True);
    # at T = 1 the layer dispatches to fused_recurrent. Reference backend: the recurrence.
    def init_state(self, batch: int, device, dtype=None, capacity: int | None = None,
                   length: int = 0) -> dict:
        H = self.n_heads
        dk, dv = self.key_dim // H, self.value_dim // H
        S = None
        if length:      # E3: an update costs the same whatever the state holds
            shape = (batch, H, dv, dk) if self.backend == "fla" else (batch, H, dk, dv)
            S = 0.01 * torch.randn(shape, device=device, dtype=torch.float32)
        if self.backend == "fla":
            return {"cache": OneLayerCache(S), "len": length}
        return {"S": S, "len": length}

    def prefill(self, h: torch.Tensor, capacity: int | None = None):
        state = self.init_state(h.shape[0], h.device)
        return self.step(h, state)

    def step(self, h_t: torch.Tensor, state: dict):
        """Process h_t (B, T, D) (T = 1 for a streaming update) after the state."""
        if self.backend == "fla":
            y = self.layer(h_t, past_key_values=state["cache"], use_cache=True)[0]
        else:
            S0 = state["S"].to(h_t.dtype) if state["S"] is not None else None
            y, state["S"] = self.layer.forward_state(h_t, S0)
        state["len"] += h_t.shape[1]
        return y, state

    def sdpa_backend(self, batch: int, T: int, device, dtype) -> str:
        if self.backend != "fla":
            return "reference"
        return "fla fused_recurrent" if T < FLA_CHUNK_MIN_T else "fla chunk"


class OneLayerCache:
    """The cache protocol fla's GLA layer uses (len, [layer_idx], update), for ONE layer.
    It only stores what the layer hands it; fla's layer does all the GLA maths. Used instead
    of fla's own cache classes, whose base class changes with the transformers version."""

    def __init__(self, recurrent_state: torch.Tensor | None = None):
        self.state = {"recurrent_state": recurrent_state, "attn_state": None,
                      "conv_state": None, "ffn_state": None}

    def __len__(self) -> int:
        return 0 if self.state["recurrent_state"] is None else 1

    def __getitem__(self, layer_idx: int) -> dict:
        return self.state

    def update(self, recurrent_state=None, attn_state=None, conv_state=None, ffn_state=None,
               layer_idx: int = 0, offset: int | None = 1, cache_kwargs=None) -> dict:
        if recurrent_state is not None:
            self.state["recurrent_state"] = recurrent_state
        if conv_state is not None:
            self.state["conv_state"] = conv_state
        return self.state


# ================================ E1 report =================================

# E1 follows the fla layer's own kernel dispatch (P0 finding, 2026-10-08, RTX PRO 6000,
# Triton 3.6, fla 0.5.2): fused_recurrent at T <= 64 matches the reference to ~1e-6;
# chunk matches to TF32 precision at T > 64; chunk FAULTS ("misaligned address") at T = 32.
# The layer never calls chunk at T <= 64, and nothing in this repo may either.
FLA_CHUNK_MIN_T = 65
E1_TOL = {"recurrent_abs": 1e-4,      # fused_recurrent vs reference, fp32, max abs
          "chunk_rel": 5e-3,          # chunk vs reference, TF32, max abs / max |reference|
          "ref_fp64_abs": 1e-10}      # reference recurrent vs reference parallel, fp64


def check_chunk_length(T: int) -> None:
    """Guard for every direct call of fla's chunk kernel (E1, Milestone 2 benchmarks)."""
    if T < FLA_CHUNK_MIN_T:
        raise ValueError(f"fla chunk_gla must not be called at T={T} <= 64 (it faults at T=32; "
                         "the layer uses fused_recurrent there)")


def _with_grads(fn, inputs, d_out):
    """Output and gradients of fn(*inputs) for the upstream gradient d_out."""
    leaves = [t.detach().clone().requires_grad_(True) for t in inputs]
    out = fn(*leaves)
    grads = torch.autograd.grad(out, leaves, d_out)
    return out.detach(), [g.detach() for g in grads]


def _abs(a, b) -> float:
    return (a - b).abs().max().item()


def _rel(a, ref) -> float:
    return (a - ref).abs().max().item() / max(ref.abs().max().item(), 1e-30)


def _record(check, T, metric, out_diff, grad_diff, tol) -> dict:
    return {"check": check, "T": T, "metric": metric, "out": out_diff, "grad": grad_diff,
            "tol": tol, "passed": bool(out_diff <= tol and grad_diff <= tol)}


def equivalence_report(device: str = "cuda", seed: int = 0,
                       short=(16, 32, 48, 64), long=(65, 128, 200, 512)) -> list[dict]:
    """E1 records: outputs AND gradients (q, k, v, gk / layer input) per check.
      T <= 64 : fla fused_recurrent vs reference recurrence, fp32, max abs <= 1e-4
      T >  64 : fla chunk vs reference and vs fla fused_recurrent, relative <= 5e-3 (TF32)
      layer   : fla layer vs ReferenceGLA with copied weights, at T = 48 (dispatches to
                fused_recurrent) and T = 200 (chunk)
      fp64    : reference recurrent vs reference parallel only (no fp64 through fla kernels)"""
    from fla.layers.gla import GatedLinearAttention
    from fla.ops.gla import chunk_gla, fused_recurrent_gla

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    g = torch.Generator(device=device).manual_seed(seed)
    B, H, K, V = 2, 4, 16, 32

    def inputs(T, dtype=torch.float32):
        q, k = (torch.randn(B, T, H, K, device=device, dtype=dtype, generator=g) for _ in range(2))
        v = torch.randn(B, T, H, V, device=device, dtype=dtype, generator=g)
        gk = F.logsigmoid(torch.randn(B, T, H, K, device=device, dtype=dtype, generator=g)) \
            / GATE_LOGIT_NORMALIZER
        d_out = torch.randn(B, T, H, V, device=device, dtype=dtype, generator=g)
        return (q, k, v, gk), d_out

    rec_fla = lambda q, k, v, gk: fused_recurrent_gla(q, k, v, gk)[0]  # noqa: E731

    def chunk_fla(q, k, v, gk):
        check_chunk_length(q.shape[1])
        return chunk_gla(q, k, v, gk)[0]

    out = []
    for T in short:
        x, d = inputs(T)
        o_ref, g_ref = _with_grads(gla_recurrent_ref, x, d)
        o, gr = _with_grads(rec_fla, x, d)
        out.append(_record("fused_recurrent vs reference", T, "abs", _abs(o, o_ref),
                           max(_abs(a, b) for a, b in zip(gr, g_ref)), E1_TOL["recurrent_abs"]))
    for T in long:
        x, d = inputs(T)
        o_ref, g_ref = _with_grads(gla_recurrent_ref, x, d)
        o_c, g_c = _with_grads(chunk_fla, x, d)
        o_r, g_r = _with_grads(rec_fla, x, d)
        out.append(_record("chunk vs reference", T, "rel", _rel(o_c, o_ref),
                           max(_rel(a, b) for a, b in zip(g_c, g_ref)), E1_TOL["chunk_rel"]))
        out.append(_record("chunk vs fused_recurrent", T, "rel", _rel(o_c, o_r),
                           max(_rel(a, b) for a, b in zip(g_c, g_r)), E1_TOL["chunk_rel"]))
    for T in (64, 200):
        x, d = inputs(T, torch.float64)
        o_rec, g_rec = _with_grads(gla_recurrent_ref, x, d)
        o_par, g_par = _with_grads(gla_parallel_ref, x, d)
        out.append(_record("reference recurrent vs parallel (fp64)", T, "abs", _abs(o_rec, o_par),
                           max(_abs(a, b) for a, b in zip(g_rec, g_par)), E1_TOL["ref_fp64_abs"]))

    # layer level: the fla layer as training uses it, against the reference mirror
    torch.manual_seed(seed)
    D, Hh = 64, 4
    fla_layer = GatedLinearAttention(mode="chunk", hidden_size=D, num_heads=Hh).to(device)
    ref_layer = ReferenceGLA(D, Hh).to(device)
    ref_layer.mode = "fused_recurrent"
    ref_layer.load_state_dict(fla_layer.state_dict())
    for T, metric in ((48, "abs"), (200, "rel")):
        x = torch.randn(2, T, D, device=device, generator=g)
        d = torch.randn(2, T, D, device=device, generator=g)
        o_f, g_f = _with_grads(lambda t: fla_layer(t)[0], [x], d)
        o_r, g_r = _with_grads(lambda t: ref_layer(t)[0], [x], d)
        diff = _abs if metric == "abs" else _rel
        tol = E1_TOL["recurrent_abs"] if metric == "abs" else E1_TOL["chunk_rel"]
        kernel = "fused_recurrent" if T < FLA_CHUNK_MIN_T else "chunk"
        out.append(_record(f"layer ({kernel}) vs reference", T, metric, diff(o_f, o_r),
                           diff(g_f[0], g_r[0]), tol))
    return out


def equivalence_passed(report: list[dict]) -> bool:
    return bool(report) and all(r["passed"] for r in report)
