"""Forward parity of ft3_main.pt: strict load into tidemark/real/ft3_core.py, one forward pass on a fixed batch, compared with the
reference code. Short (seconds on a GPU).

  python scripts/real/check_forward.py --ckpt /pscratch/sd/n/naresh/tidemark/ckpt/ft3_main.pt --reference reference/tide_v5_ft3.py
  python scripts/real/check_forward.py --ckpt ... --save-golden golden_ft3.pt      # on a machine that has reference/
  python scripts/real/check_forward.py --ckpt ... --golden golden_ft3.pt           # where reference/ is absent (Perlmutter)

Steps: (1) strict=True state-dict load of the weights-only checkpoint; (2) a fixed packed batch (seeded PackedBatcher, 16 rows);
(3) forward in fp32, eval mode; (4) the same batch and forward through the REFERENCE definitions, executed from the reference file
WITHOUT its RUN section (the file trains when imported, so only the nodes before its RUN marker are executed), or through a
saved golden. The batches must be identical and the loss and quantiles equal (default tolerance 0). Exit code 0 only if all hold.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import torch

RUN_MARKER = "# ============================ RUN ==================================="


def load_reference_namespace(ref: Path) -> dict:
    """The reference's definitions and constants, executed in a fresh namespace; nothing that trains is executed."""
    text = ref.read_text(encoding="utf-8")
    lines = text.splitlines()
    end = next(i for i, ln in enumerate(lines) if ln.startswith(RUN_MARKER))
    tree = ast.parse("\n".join(lines[:end]) + "\n")
    ns: dict = {"__name__": "ft3_reference"}
    exec(compile(tree, str(ref), "exec"), ns)          # noqa: S102  definitions only (checked by the extraction script's test)
    return ns


def fixed_batch(ns_or_core, cfg: dict, device, rows: int = 16, seed: int = 1234):
    cfg2 = dict(cfg, pack_rows=rows)
    ns_or_core_set = getattr(ns_or_core, "set_group_options", None) or ns_or_core["set_group_options"]
    ns_or_core_set(None)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    PB = getattr(ns_or_core, "PackedBatcher", None) or ns_or_core["PackedBatcher"]
    return PB(cfg2, device, seed=seed).next_batch()


@torch.no_grad()
def run_forward(model, batch, packed_forward):
    model.eval()
    return packed_forward(model, batch)


def batches_equal(a: dict, b: dict) -> list[str]:
    bad = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            bad.append(f"key {k} only in one batch")
        elif torch.is_tensor(a[k]):
            if not torch.equal(a[k].cpu(), b[k].cpu()):
                bad.append(f"batch[{k}] differs")
        elif a[k] != b[k]:
            bad.append(f"batch[{k}] {a[k]} != {b[k]}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--golden", default=None)
    ap.add_argument("--save-golden", default=None)
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--tol", type=float, default=0.0, help="max abs difference allowed (default 0: identical code, same device)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    from tidemark.real import ft3_core as C
    from tidemark.real.score_ft3 import load_ft3
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg, ck = load_ft3(a.ckpt, dev)
    n = sum(p.numel() for p in model.parameters())
    print(f"strict load OK: {n / 1e6:.2f}M params, {len(ck['model'])} tensors; cfg d_model {cfg['d_model']} layers {cfg['n_layers']} "
          f"group_attn {cfg.get('group_attn')} cov_r3 {cfg.get('cov_r3')}")
    batch = fixed_batch(C, cfg, dev, a.rows)
    out = run_forward(model, batch, C.packed_forward)
    loss, q = float(out["loss"]), out["_q"].float()
    print(f"ft3_core forward: loss {loss:.6f}, quantiles {tuple(q.shape)}, finite {bool(torch.isfinite(q).all())}")
    if a.save_golden:
        torch.save({"loss": loss, "q": q.cpu(), "batch_points": batch["points"], "rows": a.rows, "device": str(dev)}, a.save_golden)
        print(f"golden written to {a.save_golden}")
    ok = True
    if a.reference:
        ns = load_reference_namespace(Path(a.reference))
        ref_model = ns["build_model"](dict(cfg, dropout=0.0)).to(dev)
        ref_model.load_state_dict(ck["model"], strict=True)
        ref_batch = fixed_batch(ns, cfg, dev, a.rows)
        diff = batches_equal(batch, ref_batch)
        ref_out = run_forward(ref_model, ref_batch, ns["packed_forward"])
        d_loss, d_q = abs(loss - float(ref_out["loss"])), float((q - ref_out["_q"].float()).abs().max())
        print(f"reference: loss {float(ref_out['loss']):.6f}  |dloss| {d_loss:.3g}  max|dq| {d_q:.3g}  batch differences: {diff or 'none'}")
        ok = not diff and d_loss <= a.tol and d_q <= a.tol
    elif a.golden:
        g = torch.load(a.golden, map_location="cpu", weights_only=False)
        d_loss, d_q = abs(loss - g["loss"]), float((q.cpu() - g["q"]).abs().max())
        print(f"golden ({g.get('device')}): loss {g['loss']:.6f}  |dloss| {d_loss:.3g}  max|dq| {d_q:.3g}")
        ok = d_loss <= a.tol and d_q <= a.tol
    else:
        print("no --reference and no --golden: only the strict load and a finite forward pass were checked")
    print("PARITY: " + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
