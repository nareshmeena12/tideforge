"""PNG grid of v0 rows before / after the G2 calendar engine, with weekends (grey) and holidays (red) shaded (CPU, seconds).

    python tools/gallery_cal.py [--exp G2.1] [--rows 12] [--L 400] [--ppd 1] [--seed 0] [--out runs/gallery/cal_<exp>.png]

--ppd: 0 = weekly (1/7), 1 = daily, 2 / 3 / 4 = 24 / 48 / 96 steps per day. The engine options are the experiment config's
cal_engine parameters. Rows are generate_batch output of B0 (the calendar is applied here directly, so its flags are known).
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
from tideforge.expcfg import load_exp  # noqa: E402
from tideforge.gen.ext import g2_engine as E  # noqa: E402
from tideforge.gen.ext import g2_table  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="G2.1")
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--L", type=int, default=400)
    ap.add_argument("--ppd", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    e = load_exp(a.exp)
    p = dict(dict(e.gen.switches).get("cal_engine", {}))
    opts = {k: (tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in dict(E.OPTS, **p).items() if k in E.OPTS}
    prof = G.configure(G.b0())
    torch.manual_seed(a.seed)
    x = G.generate_batch(a.rows, a.L, "cpu", profile=prof)
    g = torch.Generator().manual_seed(a.seed)
    y, info = E.apply(g, x, opts, "cpu", ppd_idx=torch.full((a.rows,), a.ppd, dtype=torch.long))
    tab = g2_table.table("cpu")
    cols = 3
    fig, axes = plt.subplots((a.rows + cols - 1) // cols, cols, figsize=(6 * cols, 1.9 * ((a.rows + cols - 1) // cols)),
                             squeeze=False)
    for i in range(a.rows):
        ax = axes[i // cols][i % cols]
        t = torch.arange(a.L)
        for mask, c in ((info["weekend"][i] > 0, "0.85"), (info["code"][i] > 0, "#f4a3a3")):
            for k in t[mask].tolist():
                ax.axvspan(k - 0.5, k + 0.5, color=c, lw=0)
        ax.plot(x[i].numpy(), color="0.5", lw=0.7)
        ax.plot(y[i].numpy(), color="C0", lw=0.8)
        c = int(info["cal"][i])
        name = tab.ids[c] if c >= 0 else "invented"
        d0 = str(E.dates(info["d0"][i:i + 1])["year"].item())
        ax.set_title(f"row {i}: {name}, from {d0}{', DST' if bool(info['dst_rows'][i]) else ''}", fontsize=7)
        ax.tick_params(labelsize=5)
    fig.suptitle(f"{a.exp} calendar engine: v0 row (grey) and with calendar (blue); weekend grey, holiday red; "
                 f"steps per day {E.PPD[a.ppd]:.3g}", fontsize=9)
    fig.tight_layout()
    out = Path(a.out) if a.out else REPO / "runs" / "gallery" / f"cal_{a.exp}_ppd{a.ppd}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
