"""Probe sets: cells of items, deterministic files, versions and the registry.

A probe set (one probe, one version) is a list of cells. A cell holds n items of the same shape:
    targets   (n, k, L) float32   the series to forecast (k = 1 except P8); NaN = missing (P6)
    future    (n, k, H) float64   what happened next
    oracle_q  (n, k, 9, H)        the best forecast with full information (the generator's true predictive quantiles)
    optional: oracle_hidden_q (n, k, 9, H) (a covariate-blind reference, P1 / P2), past (n, p, L), known_ctx (n, m, L),
              known_fut (n, m, H), views (e.g. "clean" -> (n, k, L) for P6), and probe extras (masks, levels, ...).
Cell meta: name, control (bool), season S (for the naive model and the error scale), modes ("main" or "on" / "hidden"),
views ("main" plus e.g. "clean"), description.

Files: <root>/<probe>/v<version>/set.npz (a zip of .npy members with fixed time stamps, so the bytes depend only on the
values) + meta.json (cells, seed, size, sha256 of set.npz, the 6-significant-digit fingerprint). Building never overwrites:
a changed probe gets a new version. tideforge/probes/registry.json (committed) holds, per probe and version, the seed, the
size and the sha256 / fingerprint of the build on Perlmutter (the source of truth).
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats

Q9 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
Z9 = stats.norm.ppf(Q9)
REGISTRY = Path(__file__).resolve().parent / "registry.json"
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)


@dataclass
class Cell:
    name: str
    arrays: dict
    S: int
    control: bool = False
    modes: tuple = ("main",)
    views: tuple = ("main",)
    desc: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.arrays["targets"].shape[0])

    @property
    def H(self) -> int:
        return int(self.arrays["future"].shape[-1])

    def seasons(self) -> np.ndarray:
        """(n,) season per series: arrays["season"] where the period varies by series, else S for all."""
        return self.arrays["season"].astype(int) if "season" in self.arrays else np.full(self.n, int(self.S))

    def targets(self, view: str = "main") -> np.ndarray:
        return self.arrays["targets"] if view == "main" else self.arrays[f"view_{view}"]

    def meta(self) -> dict:
        return dict(name=self.name, S=self.S, control=self.control, modes=list(self.modes), views=list(self.views),
                    desc=self.desc, n=self.n, H=self.H, L=int(self.arrays["targets"].shape[-1]),
                    k=int(self.arrays["targets"].shape[1]), extra=self.extra)


@dataclass
class ProbeSet:
    probe: str
    version: int
    seed: int
    n_per_cell: int
    cells: list
    desc: str = ""

    def cell(self, name: str) -> Cell:
        return next(c for c in self.cells if c.name == name)


# ---------------------------------------------------------------------------------------------------- helpers for builders
def rng_for(seed: int, *keys) -> np.random.Generator:
    """A generator per (seed, keys...), keys hashed so cells and lengths never share a stream."""
    h = hashlib.sha256("|".join(map(str, (seed, *keys))).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def normal_q(mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """(..., H) mean and sd -> (..., 9, H) Gaussian quantiles."""
    return mean[..., None, :] + np.asarray(sd)[..., None, :] * Z9[:, None]


def ar1_forecast(e_last: np.ndarray, phi: float | np.ndarray, sigma: float | np.ndarray, H: int):
    """Mean and sd of an AR(1) noise H steps ahead, given its last value. Shapes broadcast over series."""
    h = np.arange(1, H + 1)
    phi = np.asarray(phi, dtype=float)[..., None]
    mean = phi ** h * np.asarray(e_last, dtype=float)[..., None]
    var = np.asarray(sigma, dtype=float)[..., None] ** 2 * (1 - phi ** (2 * h)) / (1 - phi ** 2)
    return mean, np.sqrt(var)


def ar1(rng, n: int, T: int, phi, sigma) -> np.ndarray:
    """n AR(1) paths of length T (stationary start)."""
    phi = np.broadcast_to(np.asarray(phi, float), (n,))
    sigma = np.broadcast_to(np.asarray(sigma, float), (n,))
    e = np.zeros((n, T))
    e[:, 0] = rng.normal(size=n) * sigma / np.sqrt(1 - phi ** 2)
    eps = rng.normal(size=(n, T)) * sigma[:, None]
    for t in range(1, T):
        e[:, t] = phi * e[:, t - 1] + eps[:, t]
    return e


def periodic(rng, n: int, T: int, P, harmonics: int = 3, phase=None) -> np.ndarray:
    """Random smooth periodic shapes of period P (per series), unit-ish amplitude."""
    P = np.broadcast_to(np.asarray(P, float), (n,))
    t = np.arange(T)[None, :]
    ph = rng.uniform(0, 2 * np.pi, size=(n, 1)) if phase is None else np.asarray(phase).reshape(n, 1)
    out = np.zeros((n, T))
    for k in range(1, harmonics + 1):
        a = rng.normal(size=(n, 1)) / k
        b = rng.uniform(0, 2 * np.pi, size=(n, 1))
        out += a * np.sin(2 * np.pi * k * (t / P[:, None]) + k * ph + b)
    return out / (out.std(1, keepdims=True) + 1e-9)


# ---------------------------------------------------------------------------------------------------- files
def _npz_bytes(arrays: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as z:
        for k in sorted(arrays):
            b = io.BytesIO()
            np.save(b, np.ascontiguousarray(arrays[k]), allow_pickle=False)
            info = zipfile.ZipInfo(k + ".npy", date_time=_ZIP_DATE)
            z.writestr(info, b.getvalue())
    return buf.getvalue()


def flat_arrays(ps: ProbeSet) -> dict:
    return {f"{c.name}/{k}": v for c in ps.cells for k, v in c.arrays.items()}


def sig6(x: np.ndarray) -> bytes:
    """Values rounded to 6 significant digits, as bytes (NaN / inf kept as markers): the cross-machine fingerprint."""
    x = np.asarray(x, dtype=np.float64).ravel()
    fin = np.isfinite(x) & (x != 0)
    e = np.zeros(x.shape, np.int64)
    e[fin] = np.floor(np.log10(np.abs(x[fin]))).astype(np.int64)
    m = np.zeros(x.shape, np.int64)
    m[fin] = np.round(x[fin] / 10.0 ** (e[fin] - 5)).astype(np.int64)
    code = np.where(np.isnan(x), 1, np.where(np.isinf(x), 2 + (x > 0), 0)).astype(np.int8)
    return m.tobytes() + e.tobytes() + code.tobytes()


def fingerprint(ps: ProbeSet) -> str:
    h = hashlib.sha256()
    for k, v in sorted(flat_arrays(ps).items()):
        h.update(k.encode())
        h.update(str(v.shape).encode())
        h.update(sig6(v))
    return h.hexdigest()


def set_bytes(ps: ProbeSet) -> bytes:
    return _npz_bytes(flat_arrays(ps))


def save(ps: ProbeSet, root: Path) -> dict:
    d = Path(root) / ps.probe / f"v{ps.version}"
    if (d / "set.npz").exists():
        raise FileExistsError(f"{d} exists: probe sets are never rebuilt in place; a changed probe gets a new version")
    d.mkdir(parents=True, exist_ok=True)
    data = set_bytes(ps)
    (d / "set.npz").write_bytes(data)
    meta = dict(probe=ps.probe, version=ps.version, seed=ps.seed, n_per_cell=ps.n_per_cell, desc=ps.desc,
                cells=[c.meta() for c in ps.cells], sha256=hashlib.sha256(data).hexdigest(), fingerprint=fingerprint(ps))
    (d / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def load(root: Path, probe: str, version: int) -> ProbeSet:
    d = Path(root) / probe / f"v{version}"
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    data = (d / "set.npz").read_bytes()
    if hashlib.sha256(data).hexdigest() != meta["sha256"]:
        raise RuntimeError(f"{d}/set.npz does not match its meta.json sha256")
    reg = registry().get(probe, {}).get(str(version), {})
    if reg.get("sha256") and reg["sha256"] != meta["sha256"]:
        raise RuntimeError(f"{d}: sha256 differs from registry.json (the Perlmutter build is the source of truth)")
    z = np.load(io.BytesIO(data))
    cells = []
    for cm in meta["cells"]:
        arr = {k.split("/", 1)[1]: z[k] for k in z.files if k.split("/", 1)[0] == cm["name"]}
        cells.append(Cell(cm["name"], arr, cm["S"], cm["control"], tuple(cm["modes"]), tuple(cm["views"]), cm["desc"],
                          cm.get("extra", {})))
    return ProbeSet(probe, version, meta["seed"], meta["n_per_cell"], cells, meta.get("desc", ""))


def registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8")) if REGISTRY.exists() else {}


def seasonal_ar_base(rng, n: int, L: int, H: int, S: int, season_amp=1.0, phi=0.6, sigma=1.0, level=0.0):
    """A base series = level + seasonal shape (period S) + AR(1) noise, over L + H steps.
    Returns (y (n, L+H), future mean (n, H), future sd (n, H)) where mean / sd are the true predictive moments given the
    context (the seasonal shape and the AR state are known to the oracle)."""
    T = L + H
    seas = periodic(rng, n, T, S) * np.asarray(season_amp, float).reshape(-1, 1)
    e = ar1(rng, n, T, phi, sigma)
    lev = np.asarray(level, float).reshape(-1, 1) if np.ndim(level) else level
    y = lev + seas + e
    m, sd = ar1_forecast(e[:, L - 1], phi, sigma, H)
    return y, lev + seas[:, L:] + m, sd
