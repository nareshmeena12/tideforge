"""Learning-rate schedule and the safety rules of the 100M run, ported from
reference/tide_v5_100m_300b.py (RUN section and train()). Under DDP every rule is fed
ALL-REDUCED values (global loss, global grad norm), so every rank takes the same decision
(otherwise DDP deadlocks).

  skip      a batch whose loss or grad norm is not finite, or whose loss > 2.5 x / grad norm
            > 8 x the median of the last 256 kept batches (after 1,000 steps), is skipped;
  rollback  (reload best.pt, halve the peak LR, continue on new data) when the holdout is
            > 1.10 x the best, when the training loss (50-step windows) stays > 1.30 x its
            best window for 3 windows in a row, or after 20 skipped batches in a row;
  stop      when the number of rollbacks exceeds max_rollbacks (3), as written in the
            reference (`_rollbacks > max_rollbacks`).
"""
from __future__ import annotations

import math
import statistics
from collections import deque


def sched_lr(peak: float, warmup_pct: float, target_points: float, points: float) -> float:
    """LR from the share of the point budget done: linear warmup from 4% of the peak, then
    cosine to 0 (the reference's sched_lr, schedule "cosine_pts"). `points` is GLOBAL."""
    f = min(max(float(points) / float(target_points), 0.0), 1.0)
    w = float(warmup_pct)
    if f < w:
        return peak * (0.04 + 0.96 * f / w)
    return 0.5 * peak * (1.0 + math.cos(math.pi * (f - w) / (1.0 - w)))


class SpikeGuard:
    """Skips a batch whose loss or grad norm is far above the recent median (or not finite)."""

    def __init__(self, warm=1000, n=256, loss_x=2.5, gn_x=8.0):
        self.warm, self.loss_x, self.gn_x = warm, loss_x, gn_x
        self.l, self.g = deque(maxlen=n), deque(maxlen=n)
        self.consecutive = 0
        self.last_note = ""

    def is_spike(self, lv: float, gv: float, step: int) -> bool:
        spike = not (math.isfinite(lv) and math.isfinite(gv))
        if not spike and step > self.warm and len(self.l) >= 64:
            spike = lv > self.loss_x * statistics.median(self.l) or gv > self.gn_x * statistics.median(self.g)
        if spike:
            self.consecutive += 1
            self.last_note = (f"  skipped a spiky batch at step {step:,}: loss {lv:.4f}, grad norm {gv:.2f} "
                              f"(medians {statistics.median(self.l) if self.l else float('nan'):.4f} / "
                              f"{statistics.median(self.g) if self.g else float('nan'):.2f})")
        else:
            self.consecutive = 0
            self.l.append(lv)
            self.g.append(gv)
        return spike

    def state_dict(self) -> dict:
        return {"l": list(self.l), "g": list(self.g), "consecutive": self.consecutive}

    def load_state_dict(self, d: dict) -> None:
        self.l.clear(), self.g.clear()
        self.l.extend(d["l"]), self.g.extend(d["g"])
        self.consecutive = int(d["consecutive"])


class WindowWatch:
    """Training-loss watch on print windows: 3 windows in a row > win_x x the best window ->
    rollback. As in the reference, it only starts after step 6,000 and 3,000 steps after the
    session started, and the best window is tracked from step 1,000."""

    def __init__(self, win_x: float = 1.30, count: int = 3):
        self.win_x, self.count = win_x, count
        self.best = float("inf")
        self.bad = 0

    def update(self, avg_loss: float, step: int, start_step: int) -> bool:
        """True when a rollback is due."""
        due = False
        if step - start_step > 3000 and step > 6000:
            if avg_loss > self.win_x * self.best:
                self.bad += 1
                due = self.bad >= self.count
            else:
                self.bad = 0
        if step > 1000:
            self.best = min(self.best, avg_loss)
        return due

    def state_dict(self) -> dict:
        return {"best": self.best, "bad": self.bad}

    def load_state_dict(self, d: dict) -> None:
        self.best, self.bad = float(d["best"]), int(d["bad"])


def holdout_rollback(holdout: float, best: float, x: float = 1.10) -> bool:
    return math.isfinite(best) and holdout > x * best


def too_many_rollbacks(rollbacks: int, max_rollbacks: int = 3) -> bool:
    return rollbacks > max_rollbacks
