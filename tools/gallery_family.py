"""PNG grid of one new family's own rows (G5, G7.1 / G7.2, G9, G10), before v0's layers (CPU, seconds).

    python tools/gallery_family.py --family amfm [--rows 12] [--L 512] [--seed 0] [--out runs/gallery/fam_<family>.png]
    python tools/gallery_family.py --all [--pool-dir DIR --manifest-dir DIR]        # every family (ode needs its pool)
    python tools/gallery_family.py --family demand --param lifecycle_p=1.0

tools/gallery.py compares a config off vs on, but a new family changes every global draw after it, so there every block
differs; this shows what the family itself draws. The switch's defaults are used, overridden by --param k=v.
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
import yaml  # noqa: E402

from tideforge.gen import _state, switches  # noqa: E402
from tideforge.gen.ext import g5_cycles, g7_demand, g9_ode, g9_tar, g10_finance  # noqa: E402
from tideforge.gen.primitives import PERIOD_BANK  # noqa: E402

FAMILIES = {
    "amfm": lambda p: g5_cycles.amfm, "beats": lambda p: g5_cycles.beats,
    "irregular_cycles": lambda p: g5_cycles.irregular_cycles,
    "demand": lambda p: g7_demand.make_demand(float(p["lifecycle_p"])),
    "ode": g9_ode.make_ode, "tar": g9_tar.make_tar, "garch_jump": g10_finance.make_garch,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=sorted(FAMILIES))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--param", action="append", default=[])
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--L", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool-dir", default="")
    ap.add_argument("--manifest-dir", default="")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.all:
        for f in sorted(FAMILIES):
            try:
                main(["--family", f, "--rows", str(a.rows), "--L", str(a.L), "--seed", str(a.seed),
                      "--pool-dir", a.pool_dir, "--manifest-dir", a.manifest_dir])
            except RuntimeError as err:
                print(f"{f}: skipped ({err})")
        return 0
    p = dict(switches.REGISTRY[a.family].params)
    if "pool_dir" in p:
        p.update(pool_dir=a.pool_dir, manifest_dir=a.manifest_dir)
    for kv in a.param:
        k, v = kv.split("=", 1)
        p[k] = yaml.safe_load(v)
    _state._PERIOD_BANK_ACTIVE = PERIOD_BANK
    torch.manual_seed(a.seed)
    x = FAMILIES[a.family](p)(a.rows, a.L, "cpu")
    cols = 3
    rows = (a.rows + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 1.8 * rows), squeeze=False)
    for i in range(a.rows):
        ax = axes[i // cols][i % cols]
        ax.plot(x[i].numpy(), color="C0", lw=0.7, drawstyle="steps-mid" if a.family == "demand" else "default")
        ax.set_title(f"row {i}", fontsize=7)
        ax.tick_params(labelsize=5)
    extra = ", ".join(a.param)
    fig.suptitle(f"family {a.family} ({switches.REGISTRY[a.family].experiment}){': ' + extra if extra else ''}; "
                 f"raw rows before v0's layers, seed {a.seed}, L {a.L}", fontsize=9)
    fig.tight_layout()
    out = Path(a.out) if a.out else REPO / "runs" / "gallery" / f"fam_{a.family}{'_' + extra.replace('=', '') if extra else ''}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
