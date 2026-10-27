"""Benchmark audit of the dev set (DATA_COLLECTION.md step 6; owner design 2026-10-27).

Name level (local, no data): excluded_benchmark.csv, the `mixed` rule, the aggregator rules, and a fuzzy match of the dataset
name against benchmark_datasets.csv (flagged for a hand check).

Value level, two stages:
  (a) candidates: every 64-point window is z-scored, put in 0.25-sd bins clipped at +-4, and hashed in two codings, the second
      with the bins shifted by half a bin, so a value at a bin edge that flips one coding is caught by the other. Benchmark
      windows are hashed at stride 16, dev windows at stride 1 (any offset of a copy hits a benchmark window). Noise filter:
      windows with NaN, ~zero spread, fewer than 8 distinct values, or one code in more than half of the points are skipped.
  (b) verification: each candidate pair is compared on the raw values: Pearson r > 0.999 over the 64 points. Where the dev
      frequency is a fraction of a benchmark frequency, the dev series is first mean-aggregated to it (clock-aligned bins).
A dataset is flagged when >= 3 windows verify; flagged datasets are checked by hand and the decision goes into
catalog/audit_decisions.csv. The same check runs dev against dev (other datasets only) to catch copies across split roles.

    python -m tideforge.devdata.audit names       --ids batch1
    python -m tideforge.devdata.audit fingerprint --tm-eval $TM_EVAL --gift $TF_DATA/bench/gifteval --out $TF_DATA/audit/bench
    python -m tideforge.devdata.audit check       --db $TF_DATA/audit/bench --staging $TF_DATA/staging --ids batch1
    python -m tideforge.devdata.audit devdev      --staging $TF_DATA/staging --ids batch1 --out $TF_DATA/audit/devdev
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tideforge.devdata import catalog as cat

W = 64
BIN = 0.25
CLIP = 4.0
BENCH_STRIDE = 16
R_MIN = 0.999
FLAG_WINDOWS = 3
MAX_CAND_PER_WINDOW = 32
_COEF = (np.random.default_rng(20261027).integers(1, 2 ** 63, size=W, dtype=np.uint64) | np.uint64(1))


# ---------------------------------------------------------------------------------------------------- hashing
def windows(x: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < W:
        return np.zeros(0, int), np.zeros((0, W))
    w = np.lib.stride_tricks.sliding_window_view(x, W)[::stride]
    return np.arange(0, len(x) - W + 1, stride), w


def codes(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(code1, code2, ok) for windows (n, W)."""
    ok = np.isfinite(w).all(1)
    mu = np.nanmean(np.where(ok[:, None], w, 0), 1, keepdims=True)
    sd = np.nanstd(np.where(ok[:, None], w, 0), 1, keepdims=True)
    ok &= (sd[:, 0] > 1e-9 * (np.abs(mu[:, 0]) + 1.0))
    z = np.clip((np.where(ok[:, None], w, 0) - mu) / np.where(sd > 0, sd, 1), -CLIP, CLIP)
    c1 = np.floor(z / BIN).astype(np.int64)
    c2 = np.floor(z / BIN + 0.5).astype(np.int64)
    if len(w):
        srt = np.sort(np.where(ok[:, None], w, 0), 1)
        distinct = 1 + (np.diff(srt, axis=1) != 0).sum(1)
        ok &= distinct >= 8
        top = np.zeros(len(w), int)          # points in the most common code
        for v in range(int(-CLIP / BIN), int(CLIP / BIN) + 1):
            top = np.maximum(top, (c1 == v).sum(1))
        ok &= top <= W // 2
    return c1, c2, ok


def hashes(c: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        return (c.astype(np.uint64) * _COEF[None, :]).sum(1, dtype=np.uint64)


def fingerprints(x: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(starts, h1, h2) of the windows that pass the noise filter."""
    st, w = windows(x, stride)
    if not len(st):
        return st, np.zeros(0, np.uint64), np.zeros(0, np.uint64)
    c1, c2, ok = codes(w)
    return st[ok], hashes(c1[ok]), hashes(c2[ok])


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


# ---------------------------------------------------------------------------------------------------- database
class DB:
    """Fingerprints + raw values of a set of reference series (benchmarks, or the dev set for dev-vs-dev)."""

    def __init__(self):
        self.meta: list[dict] = []
        self.values: list[np.ndarray] = []

    def add(self, x, **meta) -> None:
        x = np.asarray(x, dtype=np.float64).ravel()
        if len(x) >= W:
            self.values.append(x.astype(np.float32))
            self.meta.append(meta)

    def build(self, stride: int = BENCH_STRIDE) -> "DB":
        sid, st, h1, h2 = [], [], [], []
        for i, x in enumerate(self.values):
            s, a, b = fingerprints(x.astype(np.float64), stride)
            sid.append(np.full(len(s), i, np.int64))
            st.append(s)
            h1.append(a)
            h2.append(b)
        sid, st = np.concatenate(sid or [np.zeros(0, np.int64)]), np.concatenate(st or [np.zeros(0, np.int64)])
        self.index = {}
        for name, h in (("h1", np.concatenate(h1 or [np.zeros(0, np.uint64)])), ("h2", np.concatenate(h2 or [np.zeros(0, np.uint64)]))):
            o = np.argsort(h, kind="stable")
            self.index[name] = (h[o], sid[o], st[o])
        return self

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        lens = np.array([len(v) for v in self.values], np.int64)
        np.save(path / "values.npy", np.concatenate(self.values) if self.values else np.zeros(0, np.float32))
        np.save(path / "lengths.npy", lens)
        for name, (h, s, t) in self.index.items():
            np.savez(path / f"{name}.npz", h=h, sid=s, start=t)
        (path / "meta.json").write_text(json.dumps(self.meta), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DB":
        path = Path(path)
        db = cls()
        vals = np.load(path / "values.npy", mmap_mode="r")
        lens = np.load(path / "lengths.npy")
        offs = np.r_[0, np.cumsum(lens)]
        db.values = [vals[offs[i]:offs[i + 1]] for i in range(len(lens))]
        db.meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        db.index = {}
        for name in ("h1", "h2"):
            z = np.load(path / f"{name}.npz")
            db.index[name] = (z["h"], z["sid"], z["start"])
        return db

    def candidates(self, name: str, h: np.ndarray) -> list[tuple[int, int, int]]:
        """[(query row, ref series, ref start)] for the query hashes h."""
        ref, sid, st = self.index[name]
        lo = np.searchsorted(ref, h, "left")
        hi = np.minimum(np.searchsorted(ref, h, "right"), lo + MAX_CAND_PER_WINDOW)
        out = []
        for q in np.nonzero(hi > lo)[0]:
            out += [(int(q), int(sid[k]), int(st[k])) for k in range(lo[q], hi[q])]
        return out

    def freqs(self) -> set:
        return {m.get("freq") for m in self.meta if m.get("freq")}


def match_series(x: np.ndarray, db: DB, skip=lambda meta: False) -> list[dict]:
    """Verified matches of one series against db: [{start, ref, ref_start, r}] (one per dev window, best r)."""
    x = np.asarray(x, dtype=np.float64)
    st, h1, h2 = fingerprints(x, 1)
    best: dict[int, dict] = {}
    for name, h in (("h1", h1), ("h2", h2)):
        for q, ref, rs in db.candidates(name, h):
            if skip(db.meta[ref]):
                continue
            s = int(st[q])
            r = pearson(x[s:s + W], np.asarray(db.values[ref][rs:rs + W], dtype=np.float64))
            if r > R_MIN and r > best.get(s, {}).get("r", -1):
                best[s] = dict(start=s, ref=ref, ref_start=rs, r=r)
    return list(best.values())


def aggregate(ts: pd.Series, freq: str, ref_freq: str) -> np.ndarray | None:
    """Mean-aggregate a dev series (UTC timestamps) to a coarser reference frequency, clock-aligned; None if not coarser."""
    try:
        f, g = pd.tseries.frequencies.to_offset(freq), pd.tseries.frequencies.to_offset(ref_freq)
        fn, gn = pd.Timedelta(f).value, pd.Timedelta(g).value
    except (ValueError, TypeError):
        return None
    if gn <= fn or gn % fn:
        return None
    return ts.resample(ref_freq).mean().to_numpy()


def check_dataset(df: pd.DataFrame, freq: str, db: DB, skip=lambda meta: False) -> pd.DataFrame:
    """All verified matches of a staged dataset frequency (long df) against db, natively and aggregated."""
    rows = []
    for sid, g in df.groupby("series_id"):
        s = g.set_index("timestamp")["value"].sort_index()
        variants = [(freq, s.to_numpy())]
        for rf in sorted(db.freqs()):
            a = aggregate(s, freq, rf)
            if a is not None and len(a) >= W:
                variants.append((rf, a))
        for f, x in variants:
            for m in match_series(x, db, skip):
                ref = db.meta[m["ref"]]
                rows.append(dict(series_id=sid, at_freq=f, start=m["start"], r=m["r"], ref_start=m["ref_start"], **ref))
    return pd.DataFrame(rows)


def flags(matches: pd.DataFrame) -> pd.DataFrame:
    """Per reference dataset: number of verified windows; flagged when >= FLAG_WINDOWS."""
    if matches.empty:
        return pd.DataFrame(columns=["ref_dataset", "windows", "series", "flag"])
    key = "dataset" if "dataset" in matches else "ref_dataset"
    g = matches.groupby(key).agg(windows=("start", "size"), series=("series_id", "nunique")).reset_index()
    g = g.rename(columns={key: "ref_dataset"})
    g["flag"] = g["windows"] >= FLAG_WINDOWS
    return g.sort_values("windows", ascending=False)


# ---------------------------------------------------------------------------------------------------- name level
STOP = {"the", "and", "data", "dataset", "datasets", "daily", "hourly", "weekly", "monthly", "open", "public", "series", "time",
        "via", "for", "with", "from", "all", "of", "in"}


def tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", str(s).lower()) if len(t) >= 3 and t not in STOP}


def name_audit(srcs: dict, ids: list[str]) -> pd.DataFrame:
    from tideforge.devdata.schema import validate
    b = cat.benchmarks()
    bt = [(r["benchmark"], r["dataset"], tokens(r["dataset"]) | tokens(r["original_source"])) for _, r in b.iterrows()]
    rows = []
    for i in ids:
        row = cat.catalog().loc[i]
        nt = tokens(row["name"])
        hits = [f"{bm}:{ds}" for bm, ds, t in bt if nt and t and len(nt & t) / len(nt | t) >= 0.34]
        rows.append(dict(dataset_id=i, name=row["name"], overlap=row["overlap"], problems="; ".join(validate(srcs[i])),
                         name_matches="; ".join(hits), check_by_hand=bool(hits) or row["overlap"] != "clean?"))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------- benchmark readers (Perlmutter)
def bench_time(tm_eval: Path, smoke: bool = False):
    """TIME: every dataset's test inputs (one term per dataset; all terms share the series)."""
    import tidemark.eval.score_twins as S
    paths = S.Paths(Path(tm_eval).parent)
    tb = S.TB(paths)
    config = tb.load_dataset_config()
    for k, name in enumerate(config["datasets"]):
        if smoke and k >= 2:
            break
        term = list(tb.get_available_terms(name, config))[-1]
        st = tb.get_dataset_settings(name, term, config)
        ds = tb.Dataset(name=name, term=term, to_univariate=False, prediction_length=st["prediction_length"],
                        test_length=st["test_length"], val_length=st["val_length"])
        for j, d in enumerate(ds.test_data.input):
            a = np.asarray(d["target"], dtype=np.float64)
            for v, row in enumerate(a[None] if a.ndim == 1 else a):
                yield row, dict(bench="TIME", dataset=name, item=f"{j}/{v}", freq=str(ds.freq))


def bench_fev(tm_eval: Path, smoke: bool = False):
    """fev-bench: every task's windows, past target columns."""
    import fev
    tasks = fev.Benchmark.from_yaml(Path(tm_eval) / "data" / "fev" / "tasks.yaml").tasks
    for k, task in enumerate(tasks):
        if smoke and k >= 3:
            break
        for wi, w in enumerate(task.iter_windows()):
            past, _ = w.get_input_data()
            for col in task.target_columns:
                for j, y in enumerate(past[col]):
                    yield np.asarray(y, dtype=np.float64), dict(bench="fev", dataset=task.task_name, item=f"{wi}/{col}/{j}",
                                                                freq=str(getattr(task, "freq", "") or ""))


def bench_gift(root: Path, smoke: bool = False):
    """GIFT-Eval test data downloaded with huggingface_hub (one datasets folder per dataset, fields target / freq)."""
    import datasets
    dirs = sorted(p.parent for p in Path(root).rglob("dataset_info.json"))
    for k, d in enumerate(dirs):
        if smoke and k >= 2:
            break
        ds = datasets.load_from_disk(str(d))
        for j, r in enumerate(ds):
            a = np.asarray(r["target"], dtype=np.float64)
            for v, row in enumerate(a[None] if a.ndim == 1 else a):
                yield row, dict(bench="GIFT-Eval", dataset=str(d.relative_to(root)), item=f"{j}/{v}", freq=str(r.get("freq", "")))


# ---------------------------------------------------------------------------------------------------- CLI
def staged(staging: Path, slug: str):
    for p in sorted((Path(staging) / slug).glob("*.parquet")):
        yield p.stem, pd.read_parquet(p)


def main(argv=None) -> int:
    from tideforge.devdata.schema import load_sources, resolve_ids
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["names", "fingerprint", "check", "devdev"])
    ap.add_argument("--ids", default="batch1")
    ap.add_argument("--tm-eval")
    ap.add_argument("--gift")
    ap.add_argument("--db")
    ap.add_argument("--staging")
    ap.add_argument("--out")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    srcs, batches = load_sources()
    ids = resolve_ids(a.ids, srcs, batches)
    if a.cmd == "names":
        df = name_audit(srcs, ids)
        print(df.to_string(index=False))
        if a.out:
            df.to_csv(a.out, index=False)
        return 0
    if a.cmd == "fingerprint":
        db = DB()
        n = 0
        for reader, arg in ((bench_time, a.tm_eval), (bench_fev, a.tm_eval), (bench_gift, a.gift)):
            if not arg:
                print(f"skipping {reader.__name__} (no path given)")
                continue
            for x, meta in reader(Path(arg), a.smoke):
                db.add(x, **meta)
                n += 1
        db.build().save(Path(a.out))
        print(f"fingerprinted {n} benchmark series ({len(db.values)} long enough), "
              f"{len(db.index['h1'][0]):,} windows -> {a.out}")
        return 0
    if a.cmd == "check":
        db = DB.load(Path(a.db))
        out = Path(a.out or Path(a.db).parent / "value_audit")
        out.mkdir(parents=True, exist_ok=True)
        for i in ids:
            for freq, df in staged(Path(a.staging), srcs[i].slug):
                m = check_dataset(df, freq, db)
                m.to_csv(out / f"{i}_{freq}_matches.csv", index=False)
                f = flags(m)
                print(f"{i} {freq}: {len(m)} verified windows; flagged: "
                      f"{f[f['flag']]['ref_dataset'].tolist() if len(f) else []}")
        return 0
    # devdev: one DB of every staged dataset; each dataset is checked against the others
    db = DB()
    for i in ids:
        for freq, df in staged(Path(a.staging), srcs[i].slug):
            for sid, g in df.groupby("series_id"):
                db.add(g.sort_values("timestamp")["value"].to_numpy(), dataset=i, item=sid, freq=freq)
    db.build()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in ids:
        for freq, df in staged(Path(a.staging), srcs[i].slug):
            m = check_dataset(df, freq, db, skip=lambda meta, i=i: meta["dataset"] == i)
            m.to_csv(out / f"{i}_{freq}_devdev.csv", index=False)
            f = flags(m)
            print(f"{i} {freq}: flagged against {f[f['flag']]['ref_dataset'].tolist() if len(f) else []}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
