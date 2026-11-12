"""The only network door of the Hugging Face catalog profiler (tideforge.devdata.hf_profile).

HARD RULE (owner 2026-11-12): no dataset file is downloaded. `Fetcher.get` is the single fetch function; it refuses every
request that is not one of:

  (a) hub_api  https://huggingface.co/api/datasets/<repo>[/tree/...|/revision/...]         repo metadata, file listing
  (b) viewer   https://datasets-server.huggingface.co/{is-valid,size,info,statistics,parquet}
  (c) footer   https://huggingface.co/datasets/<repo>/resolve/<rev>/<path>.parquet with `Range: bytes=-N`
               (a suffix range: the last N bytes; N <= footer_max and N < the file size). The answer must be 206.
  (d) text     https://huggingface.co/datasets/<repo>/resolve/<rev>/<path> for README.md, dataset_info(s).json and
               *meta*/*info* .json/.yaml files; the body is cut off (and refused) above 1 MB.

It counts the bytes that came over the wire (bodies + response headers, redirects and errors included) and raises
BudgetError above the budget (2 GB by default), which aborts the run. At most `max_inflight` requests are in flight. A 429
waits for Retry-After. Every final answer is cached under cache_dir, so a rerun makes no repeat calls.

The token comes from the caller (HF_TOKEN in the environment), is sent only to the two Hugging Face hosts, and is never
printed, logged, cached or put into an error message.

Why not HfApi / HfFileSystem: they open their own connections, which this guard could neither check nor count, and
fsspec's read-ahead can pull multi-MB blocks of a parquet file. The same endpoints are called here instead.
"""
from __future__ import annotations

import email.utils
import hashlib
import json
import re
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

HUB = "huggingface.co"
VIEWER = "datasets-server.huggingface.co"
VIEWER_ENDPOINTS = ("is-valid", "size", "info", "statistics", "parquet")
TEXT_MAX = 1_000_000  # (d): small text metadata files, under 1 MB
API_MAX = 256 * 1024 * 1024  # one metadata answer (a file listing of a very large repo is tens of MB)
FOOTER_MAX = 8 * 1024 * 1024  # default cap on one footer read
BUDGET = 2 * 1024 ** 3
UA = "tideforge-hf-profile/0.1 (metadata only)"

_API = re.compile(r"^/api/datasets/[^/]+(?:/[^/]+)?(?:/(?:tree|revision)/.+)?$")
_RESOLVE = re.compile(r"^/datasets/(?P<repo>[^/]+(?:/[^/]+)?)/resolve/(?P<rev>[^/]+)/(?P<path>.+)$")
_RANGE = re.compile(r"^bytes=-(\d+)$")
_TEXT_NAME = re.compile(r"(?i)^(readme(\.md)?|dataset_infos?\.json|[^/]*(meta|info)[^/]*\.(json|ya?ml))$")
_NO_CACHE = {429, 500, 502, 503, 504}


class GuardError(Exception):
    """The request is not allowed by the hard rule (or the server's answer would break it)."""


class BudgetError(Exception):
    """The byte budget is used up: the run must stop."""


class FetchError(Exception):
    """Network failure after the retries, or a cache miss in offline mode."""


def classify(url: str, range_: str | None = None, size_hint: int | None = None, footer_max: int = FOOTER_MAX) -> str:
    """-> "hub_api" | "viewer" | "footer" | "text"; raises GuardError for anything else."""
    u = urllib.parse.urlsplit(url)
    if u.scheme != "https" or u.username or u.password or u.port not in (None, 443):
        raise GuardError(f"refused (not plain https): {url}")
    host, path = (u.hostname or "").lower(), u.path
    if host == VIEWER:
        if range_ is None and path.lstrip("/") in VIEWER_ENDPOINTS:
            return "viewer"
        raise GuardError(f"refused (not an allowed viewer endpoint): {url}")
    if host != HUB:
        raise GuardError(f"refused (host not allowed): {url}")
    if _API.match(path):
        if range_ is not None:
            raise GuardError(f"refused (range on an API call): {url}")
        return "hub_api"
    m = _RESOLVE.match(path)
    if not m:
        raise GuardError(f"refused (not a metadata URL): {url}")
    name = urllib.parse.unquote(m.group("path")).rsplit("/", 1)[-1]
    if range_ is not None:
        r = _RANGE.match(range_.strip())
        if not r:
            raise GuardError(f"refused (only a suffix range `bytes=-N` reads a footer; got {range_!r}): {url}")
        n = int(r.group(1))
        if not name.lower().endswith(".parquet"):
            raise GuardError(f"refused (range read of a non-parquet file): {url}")
        if n < 8 or n > footer_max:
            raise GuardError(f"refused (footer read of {n} bytes; allowed 8..{footer_max}): {url}")
        if size_hint is not None and n >= size_hint:
            raise GuardError(f"refused (the range covers the whole {size_hint}-byte file): {url}")
        return "footer"
    if _TEXT_NAME.match(name):
        if size_hint is not None and size_hint >= TEXT_MAX:
            raise GuardError(f"refused (text metadata file of {size_hint} bytes; limit {TEXT_MAX}): {url}")
        return "text"
    raise GuardError(f"refused (data file; only parquet footers and small metadata files are read): {url}")


@dataclass
class Response:
    status: int
    body: bytes
    headers: dict = field(default_factory=dict)
    kind: str = ""
    from_cache: bool = False

    def json(self):
        return json.loads(self.body.decode("utf-8"))

    @property
    def error_text(self) -> str:
        code = self.headers.get("x-error-code", "")
        try:
            msg = str(self.json().get("error", ""))
        except Exception:
            msg = self.body[:200].decode("utf-8", "replace")
        return " ".join(x for x in (f"HTTP {self.status}", code, msg[:300]) if x)


def retry_after_seconds(value: str | None, attempt: int, now: float | None = None) -> float:
    """Seconds to wait after a 429: Retry-After (seconds or an HTTP date), else 2, 4, 8, ... The wait is 1..300 s."""
    wait = None
    if value:
        value = value.strip()
        if re.fullmatch(r"\d+(\.\d+)?", value):
            wait = float(value)
        else:
            try:
                wait = email.utils.parsedate_to_datetime(value).timestamp() - (time.time() if now is None else now)
            except Exception:
                wait = None
    if wait is None:
        wait = 2.0 ** (attempt + 1)
    return float(min(300.0, max(1.0, wait)))


class Fetcher:
    def __init__(self, cache_dir, token: str | None = None, budget: int = BUDGET, max_inflight: int = 4,
                 refresh: bool = False, offline: bool = False, session=None, sleep=time.sleep,
                 footer_max: int = FOOTER_MAX, max_retries: int = 6, timeout=(15, 90)):
        self.cache_dir = Path(cache_dir)
        self._token = token or None
        self.budget = int(budget)
        self.refresh, self.offline = refresh, offline
        self.footer_max = int(footer_max)
        self.max_retries, self.timeout = max_retries, timeout
        self._sleep = sleep
        self._sem = threading.BoundedSemaphore(max_inflight)
        self._lock = threading.Lock()
        self._session = session
        self._fresh: set[str] = set()  # keys fetched in this run (so --refresh fetches each key once)
        self.bytes_total = 0
        self.n_requests = 0
        self.n_cache_hits = 0
        self.n_429 = 0
        self.by_kind: dict[str, list[int]] = {}  # kind -> [requests, bytes]

    def __repr__(self):  # never show the token
        return f"Fetcher(cache={self.cache_dir}, bytes={self.bytes_total}, requests={self.n_requests})"

    # ---- cache -------------------------------------------------------------------------------------------------------
    @staticmethod
    def key(url: str, range_: str | None) -> str:
        return hashlib.sha256(f"{url}\n{range_ or ''}".encode()).hexdigest()

    def _paths(self, key: str):
        d = self.cache_dir / key[:2]
        return d / f"{key}.json", d / f"{key}.bin"

    def _cache_get(self, key: str, kind: str) -> Response | None:
        meta_p, body_p = self._paths(key)
        if not (meta_p.exists() and body_p.exists()):
            return None
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            return Response(int(meta["status"]), body_p.read_bytes(), meta.get("headers", {}), kind, True)
        except Exception:
            return None

    def _cache_put(self, key: str, url: str, range_: str | None, resp: Response):
        meta_p, body_p = self._paths(key)
        meta_p.parent.mkdir(parents=True, exist_ok=True)
        tmp_b, tmp_m = body_p.with_suffix(".bin.part"), meta_p.with_suffix(".json.part")
        tmp_b.write_bytes(resp.body)
        tmp_m.write_text(json.dumps({"url": url, "range": range_, "status": resp.status, "kind": resp.kind,
                                     "headers": resp.headers, "bytes": len(resp.body), "time": time.time()}),
                         encoding="utf-8")
        tmp_b.replace(body_p)
        tmp_m.replace(meta_p)

    # ---- accounting --------------------------------------------------------------------------------------------------
    def _count(self, kind: str, nbytes: int, request: bool = False):
        with self._lock:
            self.bytes_total += nbytes
            k = self.by_kind.setdefault(kind, [0, 0])
            k[1] += nbytes
            if request:
                self.n_requests += 1
                k[0] += 1
            over = self.bytes_total > self.budget
        if over:
            raise BudgetError(f"byte budget used up: {self.bytes_total} bytes transferred, limit {self.budget}")

    def _session_get(self, url, headers):
        if self._session is None:
            import requests  # owner OK 2026-11-12 (hf profiler only)

            with self._lock:
                if self._session is None:
                    self._session = requests.Session()
        return self._session.get(url, headers=headers, stream=True, allow_redirects=False, timeout=self.timeout)

    # ---- the one fetch function ----------------------------------------------------------------------------------------
    def get(self, url: str, range_: str | None = None, size_hint: int | None = None) -> Response:
        kind = classify(url, range_, size_hint, self.footer_max)  # raises GuardError; checked on cache hits too
        key = self.key(url, range_)
        if not (self.refresh and key not in self._fresh):
            hit = self._cache_get(key, kind)
            if hit is not None:
                with self._lock:
                    self.n_cache_hits += 1
                return hit
        if self.offline:
            raise FetchError(f"not in the cache (offline): {url}")
        if self.bytes_total >= self.budget:
            raise BudgetError(f"byte budget used up: {self.bytes_total} bytes transferred, limit {self.budget}")
        with self._sem:
            resp = self._fetch(url, range_, kind)
        if resp.status not in _NO_CACHE:
            self._cache_put(key, url, range_, resp)
            self._fresh.add(key)
        return resp

    def _fetch(self, url: str, range_: str | None, kind: str) -> Response:
        last = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._once(url, range_, kind)
            except (GuardError, BudgetError):
                raise
            except Exception as e:  # connection errors, timeouts
                last = FetchError(f"{type(e).__name__} for {url}")
                if attempt >= min(3, self.max_retries):
                    raise last from None
                self._sleep(min(30.0, 2.0 ** attempt))
                continue
            if resp.status == 429:
                with self._lock:
                    self.n_429 += 1
                if attempt >= self.max_retries:
                    return resp
                self._sleep(retry_after_seconds(resp.headers.get("retry-after"), attempt))
                continue
            if resp.status in (500, 502, 503, 504) and attempt < 2 and not resp.headers.get("x-error-code"):
                self._sleep(2.0 ** (attempt + 1))
                continue
            return resp
        raise last or FetchError(f"no answer for {url}")

    def _once(self, url: str, range_: str | None, kind: str) -> Response:
        cur = url
        for _hop in range(6):
            u = urllib.parse.urlsplit(cur)
            headers = {"User-Agent": UA, "Accept-Encoding": "gzip" if kind in ("hub_api", "viewer") else "identity"}
            if range_:
                headers["Range"] = range_
            if self._token and (u.hostname or "").lower() in (HUB, VIEWER) and u.scheme == "https":
                headers["Authorization"] = "Bearer " + self._token
            r = self._session_get(cur, headers)
            try:
                status = int(r.status_code)
                hdrs = {str(k).lower(): str(v) for k, v in dict(r.headers).items()}
                hbytes = sum(len(k) + len(v) + 4 for k, v in hdrs.items())
                if status in (301, 302, 303, 307, 308) and hdrs.get("location"):
                    self._count(kind, hbytes + len(self._read(r, hdrs, 65536, kind, cur, strict=False)), request=True)
                    nxt = urllib.parse.urljoin(cur, hdrs["location"])
                    n = urllib.parse.urlsplit(nxt)
                    if n.scheme != "https":
                        raise GuardError(f"refused (redirect to a non-https URL) for {url}")
                    if kind in ("hub_api", "viewer"):  # e.g. a renamed repo: must still be an allowed metadata URL
                        if classify(nxt, range_, None, self.footer_max) != kind:
                            raise GuardError(f"refused (redirect leaves the {kind} endpoints) for {url}")
                    cur = nxt  # footer / text: the file store; the Range header and the size cap go with it
                    continue
                ok = 200 <= status < 300
                if kind == "footer" and ok and status != 206:
                    self._count(kind, hbytes, request=True)
                    raise GuardError(f"refused (the server ignored the range, HTTP {status}): {url}")
                if kind == "footer":
                    cap = int(_RANGE.match(range_).group(1))
                elif kind == "text":
                    cap = TEXT_MAX
                else:
                    cap = API_MAX
                body = self._read(r, hdrs, cap if ok else 65536, kind, url, strict=ok)
                wire = len(body)
                try:
                    wire = int(r.raw.tell()) or wire
                except Exception:
                    pass
                self._count(kind, hbytes + wire, request=True)
                keep = ("content-type", "content-range", "content-length", "x-error-code", "x-error-message",
                        "retry-after", "etag", "x-repo-commit", "x-linked-size")
                return Response(status, body, {k: hdrs[k] for k in keep if k in hdrs}, kind)
            finally:
                try:
                    r.close()
                except Exception:
                    pass
        raise GuardError(f"refused (too many redirects): {url}")

    def _read(self, r, hdrs: dict, cap: int, kind: str, url: str, strict: bool) -> bytes:
        """Read at most `cap` bytes. strict: a longer body is a GuardError (nothing above the cap is kept or read on)."""
        cl = hdrs.get("content-length")
        if strict and cl and cl.isdigit() and int(cl) > cap and "content-encoding" not in hdrs:
            raise GuardError(f"refused (answer of {cl} bytes; limit {cap} for a {kind} read): {url}")
        chunks, got = [], 0
        for chunk in r.iter_content(65536):
            if not chunk:
                continue
            got += len(chunk)
            if got > cap:
                if strict:
                    self._count(kind, got)
                    raise GuardError(f"refused (answer longer than {cap} bytes for a {kind} read): {url}")
                break
            chunks.append(chunk)
            if self.bytes_total + got > self.budget:
                self._count(kind, got)  # raises BudgetError
        return b"".join(chunks)

    def summary(self) -> dict:
        return {"bytes": self.bytes_total, "requests": self.n_requests, "cache_hits": self.n_cache_hits,
                "http_429": self.n_429, "by_kind": {k: {"requests": v[0], "bytes": v[1]} for k, v in self.by_kind.items()}}
