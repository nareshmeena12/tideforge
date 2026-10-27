"""Gated DeltaNet arm (Stage A): fla's GatedDeltaNet (flash-linear-attention, pinned 0.5.2) behind the
time-mixer interface, like GLAMixer.

Dimensions are chosen so the five main projections equal softmax's and GLA's exactly: key dim = d/2,
value dim = d, same number of heads, so q, k, v, output-gate and out projections total
2 * (d/2) * d + 3 * d * d = 4 d^2 per layer (fla's default layer would use 6 d^2: head_dim 256, 6 heads,
expand_v 2, which is NOT parameter matched). With d = 512 and 8 heads: head_k_dim 32, head_v_dim 64, the
same recurrent state size as GLA (8 x 32 x 64). The extras over softmax are GDN's gate (a_proj, b_proj,
A_log, dt_bias), its three short convolutions (kernel 4, kept: fla says they are crucial) and the output
norm; about +0.17% of the twin's parameters.

Training uses fla's layer in chunk mode (the layer asserts it); in eval mode at <= 64 tokens the layer itself
dispatches to fused_recurrent. Packed rows: the rows are flattened and `cu_seqlens` marks every series start,
so the delta-rule state AND the short convolutions reset per series (fla passes cu_seqlens to both).

ReferenceGDN is a pure-PyTorch mirror (same parameter names as fla's layer, maths of
fla.ops.gated_delta_rule.naive_recurrent_gated_delta_rule) used ONLY by the CPU tests and the parameter
counts; tests/test_gdn.py (gpu) checks it against the fla layer with copied weights.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CONV_SIZE = 4
NORM_EPS = 1e-5
L2_EPS = 1e-6


def gdn_gate(a: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor) -> torch.Tensor:
    """log decay per head: g = -exp(A_log) * softplus(a + dt_bias) (fla's naive_gdn_gate)."""
    return -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())


def gated_delta_rule_ref(q, k, v, beta, g, scale: float | None = None) -> torch.Tensor:
    """fla's naive_recurrent_gated_delta_rule. q, k (B, T, H, K); v (B, T, H, V); beta, g (B, T, H) -> (B, T, H, V).
    S_t = exp(g_t) S_{t-1}; S_t += k_t (beta_t (v_t - S_t^T k_t))^T; o_t = (scale q_t)^T S_t."""
    q, k, v, beta, g = (x.transpose(1, 2).contiguous().float() for x in (q, k, v, beta, g))
    B, H, T, K = k.shape
    V = v.shape[-1]
    scale = K ** -0.5 if scale is None else scale
    q = q * scale
    S = q.new_zeros(B, H, K, V)
    out = []
    for t in range(T):
        S = S * g[:, :, t].exp()[..., None, None]
        u = (v[:, :, t] - (S * k[:, :, t, :, None]).sum(-2)) * beta[:, :, t, None]
        S = S + k[:, :, t, :, None] * u[:, :, None, :]
        out.append(torch.einsum("bhk,bhkv->bhv", q[:, :, t], S))
    return torch.stack(out, dim=2).transpose(1, 2)


class ReferenceGDN(nn.Module):
    """Pure-PyTorch mirror of fla.layers.gated_deltanet.GatedDeltaNet (use_gate, use_short_conv, defaults)."""

    def __init__(self, hidden_size: int, num_heads: int, expand_k: float = 0.5, expand_v: float = 2.0,
                 conv_size: int = CONV_SIZE):
        super().__init__()
        self.key_dim = int(hidden_size * expand_k)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = int(self.head_k_dim * expand_v)
        self.num_heads = num_heads
        self.value_dim = num_heads * self.head_v_dim
        self.conv_size = conv_size
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        A = torch.empty(num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        dt = torch.exp(torch.rand(num_heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.q_conv1d = nn.Conv1d(self.key_dim, self.key_dim, conv_size, groups=self.key_dim, bias=False)
        self.k_conv1d = nn.Conv1d(self.key_dim, self.key_dim, conv_size, groups=self.key_dim, bias=False)
        self.v_conv1d = nn.Conv1d(self.value_dim, self.value_dim, conv_size, groups=self.value_dim, bias=False)
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_norm = nn.Module()
        self.o_norm.weight = nn.Parameter(torch.ones(self.head_v_dim))
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def _conv(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        """causal depthwise convolution over time + SiLU; x (B, T, C)."""
        y = F.conv1d(F.pad(x.transpose(1, 2), (self.conv_size - 1, 0)), conv.weight, groups=conv.groups)
        return F.silu(y).transpose(1, 2)

    def forward(self, x: torch.Tensor, **_):
        """x (B, T, D) is ONE sequence per row (packed rows are cut into series by GDNMixer)."""
        B, T, _d = x.shape
        H, dk, dv = self.num_heads, self.head_k_dim, self.head_v_dim
        q = self._conv(self.q_conv1d, self.q_proj(x)).view(B, T, H, dk)
        k = self._conv(self.k_conv1d, self.k_proj(x)).view(B, T, H, dk)
        v = self._conv(self.v_conv1d, self.v_proj(x)).view(B, T, H, dv)
        q = q / (q.float().pow(2).sum(-1, keepdim=True) + L2_EPS).sqrt().to(q.dtype)
        k = k / (k.float().pow(2).sum(-1, keepdim=True) + L2_EPS).sqrt().to(k.dtype)
        o = gated_delta_rule_ref(q, k, v, torch.sigmoid(self.b_proj(x).float()),
                                 gdn_gate(self.a_proj(x), self.A_log, self.dt_bias))
        g = self.g_proj(x).view(B, T, H, dv)
        rstd = torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + NORM_EPS)
        o = (o * rstd * self.o_norm.weight * g * torch.sigmoid(g)).reshape(B, T, H * dv)
        return self.o_proj(o.to(x.dtype)), None, None


def gdn_step(layer, x_t: torch.Tensor, state: dict, fused: bool = False) -> torch.Tensor:
    """One new token (B, 1, D) through a Gated DeltaNet layer (fla's or the reference: same parameter names), updating
    `state` in place. Equals the layer's full forward at that position (tests). fused=True runs the recurrence with
    fla's fused_recurrent_gated_delta_rule kernel (GPU), else with the PyTorch maths."""
    B = x_t.shape[0]
    H, dk, dv = layer.num_heads, layer.head_k_dim, layer.head_v_dim

    def conv(proj, conv_mod, key):
        z = proj(x_t)                                                         # (B, 1, C)
        win = torch.cat([state[key].to(z.dtype), z], dim=1)                   # (B, conv_size, C)
        y = F.silu(torch.einsum("bkc,ck->bc", win, conv_mod.weight[:, 0, :].to(z.dtype)))
        state[key] = win[:, 1:].to(state[key].dtype)
        return y

    q = conv(layer.q_proj, layer.q_conv1d, "conv_q").view(B, 1, H, dk)
    k = conv(layer.k_proj, layer.k_conv1d, "conv_k").view(B, 1, H, dk)
    v = conv(layer.v_proj, layer.v_conv1d, "conv_v").view(B, 1, H, dv)
    beta = torch.sigmoid(layer.b_proj(x_t).float())                           # (B, 1, H)
    g = gdn_gate(layer.a_proj(x_t), layer.A_log, layer.dt_bias)               # (B, 1, H)
    if fused:
        from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
        o, S = fused_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=state["S"], output_final_state=True,
                                                use_qk_l2norm_in_kernel=True)
        state["S"] = S.to(state["S"].dtype)
    else:
        q = q / (q.float().pow(2).sum(-1, keepdim=True) + L2_EPS).sqrt().to(q.dtype)
        k = k / (k.float().pow(2).sum(-1, keepdim=True) + L2_EPS).sqrt().to(k.dtype)
        qf, kf, vf = (a[:, 0].float() for a in (q, k, v))                    # (B, H, d)
        S = state["S"].float() * g[:, 0].exp()[..., None, None]
        u = (vf - (S * kf[..., None]).sum(-2)) * beta[:, 0, :, None]
        S = S + kf[..., None] * u[..., None, :]
        state["S"] = S.to(state["S"].dtype)
        o = torch.einsum("bhk,bhkv->bhv", qf * dk ** -0.5, S)[:, None]       # (B, 1, H, dv)
    gate = layer.g_proj(x_t).view(B, 1, H, dv)
    o = o.float()
    o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + NORM_EPS) * layer.o_norm.weight.float()
    o = (o * gate.float() * torch.sigmoid(gate.float())).reshape(B, 1, H * dv)
    return layer.o_proj(o.to(x_t.dtype))


class GDNMixer(nn.Module):
    """Causal Gated DeltaNet over the patch tokens. (B, T, D) -> (B, T, D). Input already normed; no residual.
    backend="fla" for every run; "reference" for the CPU tests / parameter counts only."""

    def __init__(self, d_model: int, n_heads: int, backend: str = "fla"):
        super().__init__()
        if backend == "fla":
            from fla.layers.gated_deltanet import GatedDeltaNet
            self.layer = GatedDeltaNet(hidden_size=d_model, expand_v=2.0, head_dim=d_model // 2 // n_heads,
                                       num_heads=n_heads, mode="chunk", use_gate=True, use_short_conv=True,
                                       conv_size=CONV_SIZE, norm_eps=NORM_EPS, layer_idx=0)
        elif backend == "reference":
            self.layer = ReferenceGDN(d_model, n_heads)
        else:
            raise ValueError(f"unknown GDN backend {backend!r}")
        self.backend, self.n_heads = backend, n_heads
        self.key_dim, self.value_dim = d_model // 2, d_model

    def forward(self, h: torch.Tensor, pack=None) -> torch.Tensor:
        """pack (models.packing.Pack): rows flattened to one sequence, cu_seqlens at every series start and
        row start (state and short convolutions reset per series). pack=None: the ordinary batched call."""
        if pack is None:
            return self.layer(h)[0]
        R, P, D = h.shape
        if self.backend == "reference":                               # tests only: each series on its own
            flat = h.reshape(1, R * P, D)
            return torch.cat([self.layer(flat[:, a:b])[0] for a, b in pack.segments()], dim=1).view(R, P, -1)
        return self.layer(h.reshape(1, R * P, D), cu_seqlens=pack.cu_seqlens)[0].view(R, P, -1)

    # ---- streaming (E3): a constant-size state per layer = the delta-rule matrix S (B, H, dk, dv, float32) and the last
    # conv_size - 1 inputs of each of the three short convolutions. One update runs the same maths as the full forward
    # (gdn_step); with the fla backend the recurrence itself goes through fla's fused_recurrent kernel.
    def init_state(self, batch: int, device, dtype=None, capacity: int | None = None, length: int = 0) -> dict:
        H, dk, dv = self.n_heads, self.key_dim // self.n_heads, self.value_dim // self.n_heads
        dtype = dtype or torch.float32
        mk = lambda *s: (0.01 * torch.randn(*s, device=device, dtype=dtype) if length else torch.zeros(*s, device=device, dtype=dtype))  # noqa: E731
        return {"S": 0.01 * torch.randn(batch, H, dk, dv, device=device, dtype=torch.float32) if length
                else torch.zeros(batch, H, dk, dv, device=device, dtype=torch.float32),
                "conv_q": mk(batch, CONV_SIZE - 1, self.key_dim), "conv_k": mk(batch, CONV_SIZE - 1, self.key_dim),
                "conv_v": mk(batch, CONV_SIZE - 1, self.value_dim), "len": length}

    def prefill(self, h: torch.Tensor, capacity: int | None = None):
        """The output of the full causal pass and the state after it (token loop: tests and short prompts only; the
        cost study times a prefill as one forward pass, which is what the chunked kernel does)."""
        state = self.init_state(h.shape[0], h.device, h.dtype)
        ys = []
        for t in range(h.shape[1]):
            y, state = self.step(h[:, t:t + 1], state)
            ys.append(y)
        return torch.cat(ys, dim=1), state

    def step(self, h_t: torch.Tensor, state: dict):
        y = gdn_step(self.layer, h_t, state, fused=self.backend == "fla" and h_t.is_cuda)
        state["len"] += h_t.shape[1]
        return y, state

    def sdpa_backend(self, batch: int, T: int, device, dtype) -> str:
        if self.backend != "fla":
            return "reference"
        return "fla gated-delta fused_recurrent" if T <= 64 else "fla gated-delta chunk"

    # ---- bookkeeping for the G1 rule and the results rows
    def param_groups(self) -> dict[str, int]:
        groups = {"main": 0, "gate": 0, "norm": 0, "conv": 0, "other": 0}
        for name, p in self.layer.named_parameters():
            head = name.split(".")[0]
            if head in ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj"):
                groups["main"] += p.numel()
            elif head in ("a_proj", "b_proj", "A_log", "dt_bias"):
                groups["gate"] += p.numel()
            elif head == "o_norm":
                groups["norm"] += p.numel()
            elif head.endswith("conv1d"):
                groups["conv"] += p.numel()
            else:
                groups["other"] += p.numel()
        return groups

    def info(self) -> dict:
        dk = self.key_dim // self.n_heads
        dv = self.value_dim // self.n_heads
        return {"key_dim_total": self.key_dim, "head_k_dim": dk, "head_v_dim": dv, "state_size": self.n_heads * dk * dv}
