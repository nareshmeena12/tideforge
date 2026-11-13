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
    # int64 matrix-vector product: wraps modulo 2**64 exactly as the uint64 sum did, in one pass, no temporary array
    return (np.asarray(c, dtype=np.int64) @ _COEF.view(np.int64)).view(np.uint64)


def fingerprints(x: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(starts, h1, h2) of the windows that pass the noise filter."""
    st, w = windows(x, stride)
    if not len(st):
        return st, np.zeros(0, np.uint64), np.zeros(0, np.uint64)
    c1, c2, ok = codes(w)
    return st[ok], hashes(c1[ok]), hashes(c2[ok])


CHUNK = 1 << 17      # windows per block in the fast path (memory: CHUNK x W x 8 bytes per array)


def _fast_codes(x: np.ndarray, starts: np.ndarray):
    """Codes of the windows at `starts`, z-scored with rolling sums (no per-window mean / std passes). -> (c1, c2, ok)
    where ok is only the cheap filter (finite, spread); the distinct / one-code filters are applied later, to the few
    windows that hit a reference hash (same result: a filter before or after the lookup keeps the same windows)."""
    xf = np.where(np.isfinite(x), x, 0.0)
    x0 = xf - xf.mean() if len(xf) else xf                 # centred: rolling sums stay accurate
    c1s = np.r_[0.0, np.cumsum(x0)]
    c2s = np.r_[0.0, np.cumsum(x0 * x0)]
    bad = np.r_[0, np.cumsum(~np.isfinite(x))]
    s = starts
    mu0 = (c1s[s + W] - c1s[s]) / W
    var = np.maximum((c2s[s + W] - c2s[s]) / W - mu0 * mu0, 0.0)
    sd = np.sqrt(var)
    mu = mu0 + (xf.mean() if len(xf) else 0.0)
    ok = (bad[s + W] - bad[s] == 0) & (sd > 1e-9 * (np.abs(mu) + 1.0))
    z = np.lib.stride_tricks.sliding_window_view(x0, W)[s] - mu0[:, None]     # new array; then in place
    z /= np.where(sd > 0, sd, 1.0)[:, None]
    np.clip(z, -CLIP, CLIP, out=z)
    z /= BIN
    c1 = np.floor(z)
    z -= c1
    c2 = (c1 + (z >= 0.5)).astype(np.int64)       # floor(z + 0.5)
    return c1.astype(np.int64), c2, ok


def _strict_ok(x: np.ndarray, starts: np.ndarray, c1: np.ndarray) -> np.ndarray:
    """The noise filters of codes(): >= 8 distinct values and no code in more than half of the points."""
    if not len(starts):
        return np.zeros(0, bool)
    w = np.sort(np.lib.stride_tricks.sliding_window_view(x, W)[starts], 1)
    distinct = 1 + (np.diff(w, axis=1) != 0).sum(1)
    k = int(CLIP / BIN)
    idx = (c1 + k) + (2 * k + 2) * np.arange(len(c1))[:, None]
    top = np.bincount(idx.ravel(), minlength=len(c1) * (2 * k + 2)).reshape(len(c1), -1).max(1)
    return (distinct >= 8) & (top <= W // 2)


def fast_fingerprints(x: np.ndarray, stride: int, strict: bool = True):
    """Like fingerprints(), vectorised and in blocks. strict=False skips the distinct / one-code filters (apply them
    with _strict_ok to the windows that matter)."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < W:
        e = np.zeros(0, np.uint64)
        return np.zeros(0, np.int64), e, e, np.zeros((0, W), np.int64)
    st_all = np.arange(0, len(x) - W + 1, stride)
    S, H1, H2, C = [], [], [], []
    for b in range(0, len(st_all), CHUNK):
        st = st_all[b:b + CHUNK]
        c1, c2, ok = _fast_codes(x, st)
        if strict:
            ok[ok] &= _strict_ok(x, st[ok], c1[ok])
        S.append(st[ok]); H1.append(hashes(c1[ok])); H2.append(hashes(c2[ok])); C.append(c1[ok])
    return np.concatenate(S), np.concatenate(H1), np.concatenate(H2), np.concatenate(C)


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
            s, a, b, _ = fast_fingerprints(x.astype(np.float64), stride)
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
        for name, (h, s, t) in self.index.items():     # plain .npy: memory-mapped and shared by every worker
            np.save(path / f"{name}_h.npy", h)
            np.save(path / f"{name}_sid.npy", s)
            np.save(path / f"{name}_start.npy", t)
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
            if not (path / f"{name}_h.npy").exists():        # a db saved as .npz before 2026-11-13: unpack once
                z = np.load(path / f"{name}.npz")
                for k in ("h", "sid", "start"):
                    tmp = path / f"{name}_{k}.tmp.npy"
                    np.save(tmp, z[k])
                    tmp.replace(path / f"{name}_{k}.npy")
            db.index[name] = tuple(np.load(path / f"{name}_{k}.npy", mmap_mode="r") for k in ("h", "sid", "start"))
        db.path = path
        return db

    _cache: dict = {}

    @classmethod
    def shared(cls, path: Path) -> "DB":
        """One memory-mapped DB per process and path (the OS page cache shares the pages between workers)."""
        key = str(Path(path).resolve())
        if key not in cls._cache:
            cls._cache[key] = cls.load(path)
        return cls._cache[key]

    def candidates(self, name: str, h: np.ndarray) -> list[tuple[int, int, int]]:
        """[(query row, ref series, ref start)] for the query hashes h."""
        q, sid, st = self.candidate_arrays(name, h)
        return list(zip(q.tolist(), sid.tolist(), st.tolist()))

    def candidate_arrays(self, name: str, h: np.ndarray, band: str | None = None):
        """Vectorised: (query rows, ref series, ref starts). Queries are sorted before the lookup (sequential access into
        the memory-mapped index). band: only references in that frequency band, or of unknown frequency."""
        ref, sid, st = self.index[name]
        e = np.zeros(0, np.int64)
        if not len(h) or not len(ref):
            return e, e, e
        o = np.argsort(h, kind="stable")
        hs = h[o]
        ref = np.asarray(ref)                                # plain ndarray view of the memmap
        lo_s = self._lower_bound(name, ref, hs)              # bucket index; the right edge only for the (rare) hits
        hit = ref[np.minimum(lo_s, len(ref) - 1)] == hs
        if not hit.any():
            return e, e, e
        q, lo = o[hit], lo_s[hit]
        cnt = np.minimum(np.searchsorted(ref, hs[hit], "right"), lo + MAX_CAND_PER_WINDOW) - lo
        qq = np.repeat(q, cnt)
        k = np.repeat(lo - np.r_[0, np.cumsum(cnt)[:-1]], cnt) + np.arange(cnt.sum())
        s_, t_ = np.asarray(sid[k], np.int64), np.asarray(st[k], np.int64)
        if band is not None:
            rb = self.bands()[s_]
            keep = (rb == band) | (rb == "")
            qq, s_, t_ = qq[keep], s_[keep], t_[keep]
        o2 = np.lexsort((np.arange(len(qq)), qq))        # query order, as the old per-window loop
        return qq[o2], s_[o2], t_[o2]

    def _prefix(self, name: str, ref: np.ndarray) -> tuple[np.ndarray, int]:
        """Bucket offsets of the sorted hashes by their top bits (about one hash per bucket): pre[b] = first index whose
        hash starts with b. Saved next to the db (memory-mapped by every worker) the first time it is needed."""
        cache = self.__dict__.setdefault("_pre", {})
        if name not in cache:
            bits = int(min(30, max(1, np.ceil(np.log2(max(len(ref), 2))))))
            f = Path(self.path) / f"{name}_pre{bits}.npy" if getattr(self, "path", None) else None
            if f is not None and f.exists():
                pre = np.load(f, mmap_mode="r")
            else:
                cnt = np.zeros(1 << bits, np.int64)
                for b in range(0, len(ref), 1 << 24):            # in blocks: never a full copy of a large index
                    cnt += np.bincount((ref[b:b + (1 << 24)] >> np.uint64(64 - bits)).astype(np.int64), minlength=1 << bits)
                pre = np.r_[0, np.cumsum(cnt)].astype(np.int64)
                if f is not None:
                    tmp = f.with_suffix(".tmp.npy")
                    np.save(tmp, pre)
                    tmp.replace(f)
            cache[name] = (np.asarray(pre), bits)
        return cache[name]

    def _lower_bound(self, name: str, ref: np.ndarray, hs: np.ndarray) -> np.ndarray:
        """searchsorted(ref, hs, 'left') through the bucket index: a few vectorised steps instead of a binary search."""
        pre, bits = self._prefix(name, ref)
        b = (hs >> np.uint64(64 - bits)).astype(np.int64)
        lo, hi = pre[b].copy(), pre[b + 1]
        act = np.nonzero(lo < hi)[0]
        while len(act):                                       # advance while ref[lo] < h, inside the bucket
            adv = ref[lo[act]] < hs[act]
            act = act[adv]
            lo[act] += 1
            act = act[lo[act] < hi[act]]
        return lo

    def freqs(self) -> set:
        if getattr(self, "_freqs", None) is None:
            self._freqs = {m.get("freq") for m in self.meta if m.get("freq")}
        return self._freqs

    def bands(self) -> np.ndarray:
        """Frequency band of each reference series ("" when its frequency is unknown)."""
        if getattr(self, "_bands", None) is None:
            self._bands = np.array([freq_band(m.get("freq") or "") for m in self.meta], dtype=object)
        return self._bands


def freq_band(freq: str) -> str:
    """Coarse band of a pandas frequency: sub-hourly / hourly / daily / weekly / monthly ("" unknown). Monthly and longer
    are one band (a quarterly reference can hold a monthly copy aggregated, and calendar offsets have no fixed length)."""
    if not freq:
        return ""
    try:
        sec = pd.Timedelta(pd.tseries.frequencies.to_offset(freq)).total_seconds()
    except (ValueError, TypeError):
        f = str(freq).upper().lstrip("0123456789")
        return "weekly" if f.startswith("W") else "daily" if f.startswith(("B", "C")) else "monthly"
    for b, hi in (("sub-hourly", 3599), ("hourly", 86399), ("daily", 6 * 86400), ("weekly", 27 * 86400)):
        if sec <= hi:
            return b
    return "monthly"


def match_series(x: np.ndarray, db: DB, skip=lambda meta: False, freq: str | None = None) -> list[dict]:
    """Verified matches of one series against db: [{start, ref, ref_start, r}] (one per dev window, best r).
    freq: the frequency of x; references with a known frequency in another band are not looked up."""
    x = np.asarray(x, dtype=np.float64)
    st, h1, h2, c1 = fast_fingerprints(x, 1, strict=False)
    band = (freq_band(freq) or None) if freq else None
    Q, R, S = [], [], []
    for name, h in (("h1", h1), ("h2", h2)):
        q, ref, rs = db.candidate_arrays(name, h, band)
        Q.append(q)
        R.append(ref)
        S.append(rs)
    q, ref, rs = np.concatenate(Q), np.concatenate(R), np.concatenate(S)
    if not len(q):
        return []
    uq = np.unique(q)                        # strict noise filter, on the hit windows only
    good = set(uq[_strict_ok(x, st[uq], c1[uq])].tolist())
    best: dict[int, dict] = {}
    for qi, r_, s_ in zip(q.tolist(), ref.tolist(), rs.tolist()):
        if qi not in good or skip(db.meta[r_]):
            continue
        s = int(st[qi])
        r = pearson(x[s:s + W], np.asarray(db.values[r_][s_:s_ + W], dtype=np.float64))
        if r > R_MIN and r > best.get(s, {}).get("r", -1):
            best[s] = dict(start=s, ref=r_, ref_start=s_, r=r)
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
