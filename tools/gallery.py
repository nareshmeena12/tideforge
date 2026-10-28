"""PNG grid of generated series, a switch off vs on, same seeds (CPU, seconds).

    python tools/gallery.py --switch test_post --param p=0.5 --base defaults
    python tools/gallery.py --exp EXAMPLE
    python tools/gallery.py --all                      # one PNG per configs/exp/G*.yaml (train-only configs are skipped)
    python tools/gallery.py --switch NAME [--param [NAME.]k=v ...] [--set field=value ...] [--base b0|defaults]
                            [--blocks 6] [--L 512] [--seed 0] [--out runs/gallery/<label>.png]

Generates generate_blocks(8 x blocks, V, L) with the base config and with the switch config from the same torch seed and switch
seed, and plots every lane of `blocks` of them, the changed blocks first: off in grey, on in colour (the lanes a switch did not touch overlap exactly). V and
group_prob are the config's own (group_lanes, group_prob). STRICT is on: a hook that draws from the global RNG fails here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "third_party" / "tidemark_e21cbf0")]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

import tideforge.gen as G  # noqa: E402
from tideforge.expcfg import gc_from_cli  # noqa: E402
from tideforge.gen import _hooks  # noqa: E402


def blocks(gc, n, L, seed):
    name = G.configure(gc, seed=seed)
    torch.manual_seed(seed)
    return G.generate_blocks(n, max(1, gc.group_lanes), L, "cpu", profile=name, group_prob=gc.group_prob)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp")
    ap.add_argument("--base", default="b0", choices=["b0", "defaults"])
    ap.add_argument("--switch", action="append", default=[])
    ap.add_argument("--param", action="append", default=[])
    ap.add_argument("--set", dest="set_", action="append", default=[])
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--L", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args(argv)
    if a.all:
        from tideforge.expcfg import EXP_DIR, load_exp
        for p in sorted(EXP_DIR.glob("G*.yaml")):
            e = load_exp(p)
            if e.gen == e.base:
                print(f"{e.id}: generator = base (train: {e.train}), no gallery")
                continue
            try:
                main(["--exp", e.id, "--blocks", str(a.blocks), "--L", str(a.L), "--seed", str(a.seed),
                      "--out", str(REPO / "runs" / "gallery" / f"{e.id}.png")])
            except RuntimeError as err:                    # e.g. G9.1 without its pool: say so, go on
                print(f"{e.id}: skipped ({err})")
        return 0
    base, gc, label = gc_from_cli(a.base, a.exp, a.switch, a.param, a.set_)
    _hooks.STRICT = True
    n = 8 * a.blocks                                   # generate more and show changed blocks first (rare switches)
    x0, r0, _ = blocks(base, n, a.L, a.seed)
    x1, r1, _ = blocks(gc, n, a.L, a.seed)
    G.configure(base)
    ch = ~(torch.nan_to_num(x0, nan=1e30) == torch.nan_to_num(x1, nan=1e30)).flatten(1).all(1)
    pick = torch.cat([ch.nonzero().squeeze(1), (~ch).nonzero().squeeze(1)])[:a.blocks]
    n_changed_blocks = int(ch.sum())
    x0, x1, r0, r1 = x0[pick], x1[pick], r0[pick], r1[pick]
    V = x0.shape[1]
    fig, axes = plt.subplots(a.blocks, V, figsize=(2.2 * V, 1.5 * a.blocks), squeeze=False, sharex=True)
    for i in range(a.blocks):
        for j in range(V):
            ax = axes[i][j]
            ax.plot(x0[i, j].numpy(), color="0.6", lw=1.6)
            changed = not torch.equal(x0[i, j], x1[i, j])
            ax.plot(x1[i, j].numpy(), color="C3" if changed else "C0", lw=0.7)
            ax.set_title(f"b{i} lane {j} role {int(r1[i, j])}{' *' if changed else ''}", fontsize=6)
            ax.tick_params(labelsize=5)
    fig.suptitle(f"{label}: off (grey) vs on (colour; red * = changed); seed {a.seed}, L {a.L}", fontsize=9)
    fig.tight_layout()
    out = Path(a.out) if a.out else REPO / "runs" / "gallery" / f"{label}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out} ({n_changed_blocks} of {n} generated blocks changed; changed blocks shown first)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
