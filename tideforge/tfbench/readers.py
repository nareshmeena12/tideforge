"""Read one downloaded source file into Arrow, in chunks (no file is loaded whole when it is large).

open_parts(path, workdir) -> [Part]. A Part is one table: a file, or one member of an archive.
  Part.name      relative name (file path, or archive path + "!" + member)
  Part.schema    pyarrow schema
  Part.batches(columns=None) -> iterator of pa.Table chunks
Formats: parquet, arrow (ipc stream or file), csv / tsv / txt (also .gz; always streamed with pyarrow.csv.open_csv, integer
columns read as float64 so a later chunk cannot overflow them), json / jsonl (also .gz), zip / tar.gz / tgz archives of
those (extracted under workdir). Anything else raises Unreadable(reason); h5, npy and netcdf stay unreadable in v1.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import shutil
import tarfile
import zipfile
import codecs
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.ipc as ipc
import pyarrow.json as pjson
import pyarrow.parquet as pq

from tideforge.tfbench.download import ext

BATCH_ROWS = 1 << 18
CSV_BLOCK = 64 << 20
TEXT = {"csv", "tsv", "txt", "csv.gz", "tsv.gz", "txt.gz"}
JSON = {"json", "jsonl", "ndjson", "json.gz", "jsonl.gz"}
ARCHIVE = {"zip", "tar.gz", "tgz", "tar"}


class Unreadable(Exception):
    pass


@dataclass
class Part:
    name: str
    schema: pa.Schema
    _batches: Callable[[list | None], Iterator[pa.Table]]
    stats: dict = field(default_factory=dict)      # csv: encoding, rows, skipped (rows counted on the last full read)

    def batches(self, columns: list | None = None) -> Iterator[pa.Table]:
        return self._batches(columns)

    def read(self, columns: list | None = None) -> pa.Table:
        chunks = list(self.batches(columns))
        if not chunks:
            return self.schema.empty_table() if columns is None else self.schema.empty_table().select(columns)
        return pa.concat_tables(chunks, promote_options="permissive")


# ---- parquet / arrow --------------------------------------------------------------------------------------------------
def _parquet(path: Path, name: str) -> Part:
    pf = pq.ParquetFile(path)

    def gen(cols):
        for b in pq.ParquetFile(path).iter_batches(batch_size=BATCH_ROWS, columns=cols):
            yield pa.Table.from_batches([b])
    return Part(name, pf.schema_arrow, gen)


def _arrow_reader(path: Path):
    src = pa.memory_map(str(path), "r")
    try:
        return ipc.open_stream(src)
    except pa.ArrowInvalid:
        return ipc.open_file(pa.memory_map(str(path), "r"))


def _arrow(path: Path, name: str) -> Part:
    try:
        schema = _arrow_reader(path).schema
    except pa.ArrowInvalid as e:
        raise Unreadable(f"arrow: {e}")

    def gen(cols):
        r = _arrow_reader(path)
        batches = iter(r) if isinstance(r, ipc.RecordBatchStreamReader) else (r.get_batch(i) for i in range(r.num_record_batches))
        for b in batches:
            t = pa.Table.from_batches([b])
            yield t.select(cols) if cols else t
    return Part(name, schema, gen)


# ---- text ----------------------------------------------------------------------------------------------------------------
ENCODINGS = ("utf-8", "cp1252", "latin-1")


SAMPLE_FULL = 64 << 20          # files up to this size are decoded in full (streamed); larger ones are sampled
SAMPLE_HEAD = 8 << 20
SAMPLE_WINDOW = 1 << 20
SAMPLE_WINDOWS = 16


def _windows(path: Path, gz: bool) -> list[bytes]:
    """Head (8 MB) of the file, plus for plain files 16 windows of 1 MB spread to the end. Window edges are cut at line
    breaks, so a multi-byte character is never split."""
    if gz:
        with gzip.open(path, "rb") as fh:
            b = fh.read(SAMPLE_HEAD + 1)
        return [b[:b.rfind(b"\n") + 1] if len(b) > SAMPLE_HEAD else b]
    size = path.stat().st_size
    out = []
    with open(path, "rb") as fh:
        b = fh.read(SAMPLE_HEAD)
        out.append(b[:b.rfind(b"\n") + 1] if size > SAMPLE_HEAD else b)
        for k in range(1, SAMPLE_WINDOWS + 1):
            pos = SAMPLE_HEAD + (size - SAMPLE_HEAD - SAMPLE_WINDOW) * k // SAMPLE_WINDOWS
            fh.seek(max(pos, 0))
            w = fh.read(SAMPLE_WINDOW)
            i = w.find(b"\n")
            j = w.rfind(b"\n") if pos + SAMPLE_WINDOW < size else len(w) - 1
            if 0 <= i < j:
                out.append(w[i + 1:j + 1])
    return out


def detect_encoding(path: Path, gz: bool, chunk: int = 16 << 20, full: bool = False) -> str:
    """First of utf-8, cp1252, latin-1 that decodes the file (latin-1 always does). Never holds more than one chunk.
    Plain files up to SAMPLE_FULL (and any file with full=True) are decoded in full; larger files and .gz files are
    sampled (head + windows), and a later invalid-utf-8 error while reading re-runs this in full (see _csv)."""
    op = gzip.open if gz else open
    if not full and (gz or path.stat().st_size > SAMPLE_FULL):
        ws = _windows(path, gz)
        for enc in ENCODINGS[:-1]:
            try:
                for w in ws:
                    w.decode(enc)
                return enc
            except UnicodeDecodeError:
                continue
        return ENCODINGS[-1]
    for enc in ENCODINGS[:-1]:
        dec = codecs.getincrementaldecoder(enc)()
        try:
            with op(path, "rb") as fh:
                while b := fh.read(chunk):
                    dec.decode(b)
                dec.decode(b"", final=True)
            return enc
        except UnicodeDecodeError:
            continue
    return ENCODINGS[-1]


def _sniff_delim(path: Path, gz: bool, enc: str = "utf-8") -> str:
    op = gzip.open if gz else open
    with op(path, "rt", encoding=enc, errors="replace") as fh:
        head = "".join(fh.readline() for _ in range(5))
    counts = {d: head.count(d) for d in (",", "\t", ";", "|")}
    d = max(counts, key=counts.get)
    return d if counts[d] else ","


def _csv(path: Path, name: str, e: str) -> Part:
    gz = e.endswith(".gz")
    stats = {"encoding": detect_encoding(path, gz), "rows": 0, "skipped": 0}
    delim = "\t" if e.startswith("tsv") else _sniff_delim(path, gz, stats["encoding"])
    count = {"skipped": 0}

    def skip(row):        # the input is transcoded to utf-8 first, so row.text never fails to decode
        count["skipped"] += 1
        return "skip"
    parse = pcsv.ParseOptions(delimiter=delim, newlines_in_values=False, invalid_row_handler=skip)

    def stream(conv):
        src = pa.input_stream(str(path), compression="gzip" if gz else None)
        read = pcsv.ReadOptions(block_size=CSV_BLOCK, encoding=stats["encoding"])
        return pcsv.open_csv(src, read_options=read, parse_options=parse, convert_options=conv)

    try:
        first = stream(pcsv.ConvertOptions(strings_can_be_null=True))
        schema0 = first.schema
    except (pa.ArrowInvalid, UnicodeDecodeError) as ex:
        raise Unreadable(f"csv: {str(ex)[:200]}")
    types = {f.name: (pa.float64() if pa.types.is_integer(f.type) else f.type) for f in schema0
             if not pa.types.is_null(f.type)}
    types.update({f.name: pa.string() for f in schema0 if pa.types.is_null(f.type)})
    schema = pa.schema([(n, types[n]) for n in schema0.names])

    def gen(cols):
        conv = pcsv.ConvertOptions(column_types=types, strings_can_be_null=True, include_columns=cols)
        count["skipped"], n, skip_rows = 0, 0, 0
        while True:
            try:
                for b in stream(conv):
                    if skip_rows:                    # after an encoding restart: rows already handed out
                        cut = min(skip_rows, b.num_rows)
                        skip_rows -= cut
                        b = b.slice(cut)
                        if not b.num_rows:
                            continue
                    n += b.num_rows
                    yield pa.Table.from_batches([b])
                break
            except pa.ArrowInvalid as ex:
                if "utf8" in str(ex).lower().replace("-", "") and stats["encoding"] == "utf-8" and not stats.get("full"):
                    # the sample missed a non-utf-8 byte: decode the whole file (streamed), restart after the rows done
                    stats.update(encoding=detect_encoding(path, gz, full=True), full=True)
                    if stats["encoding"] == "utf-8":
                        raise Unreadable(f"csv chunk: {str(ex)[:200]}")
                    skip_rows, count["skipped"] = n, 0
                    continue
                raise Unreadable(f"csv chunk: {str(ex)[:200]}")   # a later chunk does not fit the first one's types
        stats.update(rows=n, skipped=count["skipped"])
    return Part(name, schema, gen, stats)


def _json(path: Path, name: str, e: str) -> Part:
    gz = e.endswith(".gz")
    try:
        t = pjson.read_json(pa.input_stream(str(path), compression="gzip" if gz else None))
    except pa.ArrowInvalid:
        op = gzip.open if gz else open
        try:
            with op(path, "rt", encoding="utf-8") as fh:
                obj = json.load(fh)
        except (ValueError, UnicodeDecodeError) as ex:
            raise Unreadable(f"json: {str(ex)[:200]}")
        if isinstance(obj, dict) and obj and all(isinstance(v, list) for v in obj.values()) \
                and len({len(v) for v in obj.values()}) == 1:
            obj = pa.Table.from_pydict(obj)
        elif isinstance(obj, dict):
            for k in ("data", "records", "rows", "observations", "series"):
                if isinstance(obj.get(k), list):
                    obj = obj[k]
                    break
        if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
            try:
                t = pa.Table.from_pylist(obj)
            except (pa.ArrowInvalid, pa.ArrowTypeError) as ex:
                raise Unreadable(f"json records: {str(ex)[:200]}")
        elif isinstance(obj, pa.Table):
            t = obj
        else:
            raise Unreadable("json: not a table (no list of records)")

    def gen(cols):
        yield t.select(cols) if cols else t
    return Part(name, t.schema, gen)


# ---- archives ----------------------------------------------------------------------------------------------------------
def _extract(path: Path, workdir: Path, e: str) -> Path:
    out = workdir / hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:16]
    if (out / ".done").exists():
        return out
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    try:
        if e == "zip":
            with zipfile.ZipFile(path) as z:
                for m in z.infolist():
                    if not m.is_dir() and _safe(m.filename):
                        z.extract(m, out)
        elif e in ("tar.gz", "tgz", "tar"):
            with tarfile.open(path) as tf:
                for m in tf.getmembers():
                    if m.isfile() and _safe(m.name):
                        tf.extract(m, out)
        elif e == "gz":
            target = out / path.name[:-3]
            with gzip.open(path, "rb") as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    except (zipfile.BadZipFile, tarfile.TarError, OSError, EOFError) as ex:
        raise Unreadable(f"archive: {str(ex)[:200]}")
    (out / ".done").write_text("ok")
    return out


def _safe(member: str) -> bool:
    p = Path(member)
    return not p.is_absolute() and ".." not in p.parts


def open_parts(path: Path, workdir: Path, name: str | None = None) -> list[Part]:
    path = Path(path)
    name = name or path.name
    e = ext(path.name)
    if e == "parquet":
        return [_parquet(path, name)]
    if e in ("arrow", "feather", "ipc"):
        return [_arrow(path, name)]
    if e in TEXT:
        return [_csv(path, name, e)]
    if e in JSON:
        return [_json(path, name, e)]
    if e in ARCHIVE or e == "gz":
        root = _extract(path, Path(workdir), e)
        parts, bad = [], []
        for f in sorted(p for p in root.rglob("*") if p.is_file() and p.name != ".done"):
            try:
                parts += open_parts(f, workdir, f"{name}!{f.relative_to(root).as_posix()}")
            except Unreadable as ex:
                bad.append(f"{f.name}: {ex}")
        if not parts:
            raise Unreadable(f"archive has no readable table ({'; '.join(bad[:3]) or 'empty'})")
        return parts
    raise Unreadable(f"format {e or '(none)'} not read in v1")
