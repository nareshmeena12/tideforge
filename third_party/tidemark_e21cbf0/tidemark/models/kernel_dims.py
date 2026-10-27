"""The fla kernels (Triton) need head key and value dimensions of at least 16: their matmuls assert
"M >= 1, N >= 1 and K >= 16" otherwise, and only in the BACKWARD pass, deep inside Triton. This module turns that into a
clear error that names the layer and the fix, before any training step. The real twins have head dims 32 (key) and
64 (value). The check applies to the fla backend only: the pure-PyTorch references take any size, and constructing a
layer (parameter counts, shape checks, forward-only streaming) is not restricted.

  check_model(model)              called at the start of every trainer (pretrain, supervised, diagnostics)
  check_config(cfg, kinds=None)   the same test on a config, without building the model

Whether a Triton build enforces the limit varies (Colab's older builds ran head key dim 8 in the P0 diagnostics; Perlmutter's
torch 2.13 Triton asserts). So a layer below the minimum is NOT refused outright: the guard runs the layer's own kernels once
(forward + backward, bf16, 128 tokens, on the GPU, the result cached per kind and dims) and raises only if that call fails.
  * the probe passes            -> allowed (one line says so)
  * it fails (the 'K >= 16' assertion, or anything else, or there is no GPU to probe) -> KernelDimError, as before
  * TIDEMARK_ALLOW_SMALL_KERNEL_DIMS=1 skips the guard and the probe altogether
On Perlmutter's Triton the probe fails, so the default behaviour there is unchanged.
"""
from __future__ import annotations

import os

MIN_HEAD_DIM = 16
FLA_KINDS = ("gla", "gdn")
ENV_OVERRIDE = "TIDEMARK_ALLOW_SMALL_KERNEL_DIMS"
_PROBED: dict[tuple, str | None] = {}              # (kind, head_k, head_v) -> None (kernels ran) or the failure text


class KernelDimError(ValueError):
    pass


def check_dims(kind: str, head_k: int, head_v: int, where: str = "") -> None:
    if head_k < MIN_HEAD_DIM or head_v < MIN_HEAD_DIM:
        raise KernelDimError(
            f"{where + ': ' if where else ''}the {kind.upper()} layer has head key dim {head_k} and head value dim {head_v}; "
            f"the fla {kind.upper()} kernels need both >= {MIN_HEAD_DIM} (Triton asserts 'K >= 16' in the backward pass). "
            f"Use a wider d_model or fewer heads: key dim is d_model/2 per layer divided by the heads"
            f"{', value dim d_model divided by the heads' if kind == 'gdn' else ''}; the twins use 32 and 64.")


def mixer_head_dims(m) -> tuple[int, int]:
    return m.key_dim // m.n_heads, m.value_dim // m.n_heads


def override_on(env=None) -> bool:
    return str((env if env is not None else os.environ).get(ENV_OVERRIDE, "")).strip() not in ("", "0", "false", "False")


def probe_kernels(mixer) -> str | None:
    """Run the mixer's own fla kernels once, forward and backward, as a trainer would (bf16 autocast, 128 tokens so the chunk
    kernel runs). None if they ran; else the failure text. Without a GPU there is nothing to run: a failure text."""
    import copy

    import torch
    if not torch.cuda.is_available():
        return "no GPU to probe the fla kernels on"
    try:
        m = copy.deepcopy(mixer).cuda().train()
        d = m.layer.q_proj.in_features
        x = torch.randn(1, 128, d, device="cuda", requires_grad=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = m(x)
        y.float().sum().backward()
        torch.cuda.synchronize()
        return None
    except Exception as e:  # noqa: BLE001  the Triton assertion surfaces as AssertionError or a RuntimeError
        return f"{type(e).__name__}: {str(e).strip()[:300]}"


def check_model(model, probe=probe_kernels, env=None, log=print) -> None:
    """Raise KernelDimError if an fla-backed GLA or GDN layer of `model` is below the kernels' minimum AND the installed
    Triton does not run it (see the module docstring); `probe(mixer)` -> None | failure text."""
    from tidemark.models.gdn import GDNMixer
    from tidemark.models.gla import GLAMixer
    if override_on(env):
        return
    for name, m in model.named_modules():
        if isinstance(m, (GLAMixer, GDNMixer)) and m.backend == "fla":
            hk, hv = mixer_head_dims(m)
            if hk >= MIN_HEAD_DIM and hv >= MIN_HEAD_DIM:
                continue
            kind = "gla" if isinstance(m, GLAMixer) else "gdn"
            key = (kind, hk, hv)
            if key not in _PROBED:
                _PROBED[key] = probe(m)
                if _PROBED[key] is None:
                    log(f"note: {kind.upper()} head dims {hk}/{hv} are below {MIN_HEAD_DIM}, but this Triton runs the kernels "
                        f"(forward + backward probe passed): allowed")
            if _PROBED[key] is not None:
                try:
                    check_dims(kind, hk, hv, name or type(m).__name__)
                except KernelDimError as e:
                    raise KernelDimError(f"{e} [probe of this Triton: {_PROBED[key]}; set {ENV_OVERRIDE}=1 to skip this check]") from None


def check_config(model_cfg, kinds=None) -> None:
    """model_cfg: utils.config.ModelConfig or pretrain.config.TwinModel (d_model, n_heads, gla_expand_k / _v).
    kinds: the per-layer kinds (TwinConfig.kinds); default: the ModelConfig's own layers."""
    d, H = model_cfg.d_model, model_cfg.n_heads
    if kinds is None:
        kinds = [model_cfg.layer_attention(i) for i in range(model_cfg.n_layers)]
    for i, k in enumerate(kinds):
        k = "swa" if str(k).startswith("swa") else str(k)
        if k == "gla":
            check_dims("gla", int(d * model_cfg.gla_expand_k) // H, int(d * model_cfg.gla_expand_v) // H, f"layer {i}")
        elif k == "gdn":
            check_dims("gdn", (d // 2) // H, d // H, f"layer {i}")
