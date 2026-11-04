"""Streaming GGUF header parser.

Only the header (metadata key/values + tensor infos) is read. Tensor data is
never touched, so parsing a 40 GB remote file costs a few hundred KB-few MB of
HTTP range requests. Works for GGUF v1/v2/v3, local files and remote URLs.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .ggml import GGMLType, TYPES_BY_ID, ftype_name

GGUF_MAGIC = b"GGUF"

# GGUF metadata value types
_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STR, _ARR, _U64, _I64, _F64 = range(13)
_SCALAR_FMT = {
    _U8: "<B", _I8: "<b", _U16: "<H", _I16: "<h", _U32: "<I", _I32: "<i",
    _F32: "<f", _BOOL: "<?", _U64: "<Q", _I64: "<q", _F64: "<d",
}

# Arrays longer than this are not materialised (only their length is kept);
# tokenizer vocab / merges arrays can hold 100k+ strings we never need.
ARRAY_KEEP_LIMIT = 4096
MAX_HEADER_BYTES = 256 * 1024 * 1024


class GGUFError(ValueError):
    pass


class ByteSource:
    """Random-access byte source. Subclasses implement ``read_at``."""

    def read_at(self, offset: int, size: int) -> bytes:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def size(self) -> Optional[int]:
        return None

    def close(self) -> None:
        pass


class FileSource(ByteSource):
    def __init__(self, path: str):
        self._f = open(path, "rb")
        self._f.seek(0, 2)
        self._size = self._f.tell()

    def read_at(self, offset: int, size: int) -> bytes:
        self._f.seek(offset)
        return self._f.read(size)

    @property
    def size(self) -> Optional[int]:
        return self._size

    def close(self) -> None:
        self._f.close()


class BytesSource(ByteSource):
    def __init__(self, data: bytes):
        self._data = data

    def read_at(self, offset: int, size: int) -> bytes:
        return self._data[offset: offset + size]

    @property
    def size(self) -> Optional[int]:
        return len(self._data)


class _Cursor:
    """Sequential reader over a ByteSource that buffers the header prefix from offset 0.

    Every byte is fetched exactly once (read-ahead grows geometrically, so a 16 MB
    tokenizer array costs a handful of round trips), and the buffered prefix can be
    cached verbatim once parsing ends.
    """

    def __init__(self, src: ByteSource, chunk: int = 512 * 1024):
        self.src = src
        self.pos = 0
        self.data = bytearray()
        self._chunk = chunk

    def _ensure(self, n: int) -> None:
        end_needed = self.pos + n
        if end_needed <= len(self.data):
            return
        if end_needed > MAX_HEADER_BYTES:
            raise GGUFError("GGUF header exceeds 256 MiB; refusing to read further")
        while len(self.data) < end_needed:
            want = max(end_needed - len(self.data), self._chunk)
            got = self.src.read_at(len(self.data), want)
            if not got:
                raise GGUFError("unexpected end of GGUF data while reading header")
            self.data += got
            self._chunk = min(self._chunk * 2, 16 * 1024 * 1024)

    def read(self, n: int) -> bytes:
        self._ensure(n)
        out = bytes(self.data[self.pos: self.pos + n])
        self.pos += n
        return out

    def skip(self, n: int) -> None:
        self._ensure(n)
        self.pos += n

    def unpack(self, fmt: str):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.read(size))[0]


@dataclass
class GGUFTensorInfo:
    name: str
    shape: List[int]          # ggml order: ne[0] is the contiguous (row) dimension
    type_id: int
    offset: int

    @property
    def ggml_type(self) -> Optional[GGMLType]:
        return TYPES_BY_ID.get(self.type_id)

    @property
    def type_name(self) -> str:
        t = self.ggml_type
        return t.name if t else f"type{self.type_id}"

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n

    @property
    def nbytes(self) -> int:
        t = self.ggml_type
        if t is None:
            raise GGUFError(f"tensor {self.name} has unknown ggml type {self.type_id}")
        return t.nbytes(self.numel)


@dataclass
class GGUFHeader:
    version: int
    metadata: Dict[str, Any]
    tensors: List[GGUFTensorInfo]
    header_bytes: int
    alignment: int = 32
    array_lengths: Dict[str, int] = field(default_factory=dict)
    file_size: Optional[int] = None
    raw: Optional[bytes] = field(default=None, repr=False)  # verbatim header bytes (keep_raw=True)

    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture", "unknown"))

    @property
    def file_type(self) -> Optional[str]:
        return ftype_name(self.metadata.get("general.file_type"))

    @property
    def tensor_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors)

    @property
    def split_count(self) -> int:
        return int(self.metadata.get("split.count", 1) or 1)

    def get(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def arch_get(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(f"{self.architecture}.{key}", default)

    def array_len(self, key: str) -> Optional[int]:
        if key in self.array_lengths:
            return self.array_lengths[key]
        v = self.metadata.get(key)
        return len(v) if isinstance(v, list) else None


def _read_string(cur: _Cursor, version: int) -> str:
    n = cur.unpack("<I") if version == 1 else cur.unpack("<Q")
    if n > 64 * 1024 * 1024:
        raise GGUFError(f"implausible string length {n}")
    return cur.read(n).decode("utf-8", errors="replace")


def _read_value(cur: _Cursor, vtype: int, version: int, key: str, arr_lens: Dict[str, int]):
    if vtype in _SCALAR_FMT:
        return cur.unpack(_SCALAR_FMT[vtype])
    if vtype == _STR:
        return _read_string(cur, version)
    if vtype == _ARR:
        etype = cur.unpack("<I")
        count = cur.unpack("<I") if version == 1 else cur.unpack("<Q")
        arr_lens[key] = count
        keep = count <= ARRAY_KEEP_LIMIT
        if etype in _SCALAR_FMT:
            fmt = _SCALAR_FMT[etype]
            size = struct.calcsize(fmt)
            if keep:
                raw = cur.read(size * count)
                return list(struct.unpack("<" + fmt[1:] * count, raw)) if count else []
            cur.skip(size * count)
            return None
        if etype == _STR:
            out = [] if keep else None
            for _ in range(count):
                s = _read_string(cur, version)
                if out is not None:
                    out.append(s)
            return out
        if etype == _ARR:
            out = []
            for i in range(count):
                out.append(_read_value(cur, _ARR, version, f"{key}[{i}]", arr_lens))
            return out if keep else None
        raise GGUFError(f"unsupported GGUF array element type {etype} for key {key}")
    raise GGUFError(f"unsupported GGUF value type {vtype} for key {key}")


def parse_header(src: ByteSource, keep_raw: bool = False) -> GGUFHeader:
    cur = _Cursor(src)
    magic = cur.read(4)
    if magic != GGUF_MAGIC:
        raise GGUFError(f"not a GGUF file (magic={magic!r})")
    version = cur.unpack("<I")
    if version not in (1, 2, 3):
        raise GGUFError(f"unsupported GGUF version {version}")
    if version == 1:
        n_tensors = cur.unpack("<I")
        n_kv = cur.unpack("<I")
    else:
        n_tensors = cur.unpack("<Q")
        n_kv = cur.unpack("<Q")
    if n_tensors > 10_000_000 or n_kv > 1_000_000:
        raise GGUFError("implausible tensor / key counts; file is corrupt")

    meta: Dict[str, Any] = {}
    arr_lens: Dict[str, int] = {}
    for _ in range(n_kv):
        key = _read_string(cur, version)
        vtype = cur.unpack("<I")
        meta[key] = _read_value(cur, vtype, version, key, arr_lens)

    tensors: List[GGUFTensorInfo] = []
    for _ in range(n_tensors):
        name = _read_string(cur, version)
        n_dims = cur.unpack("<I")
        if n_dims > 8:
            raise GGUFError(f"tensor {name} has {n_dims} dims")
        fmt = "<I" if version == 1 else "<Q"
        dims = [cur.unpack(fmt) for _ in range(n_dims)]
        type_id = cur.unpack("<I")
        offset = cur.unpack("<Q")
        tensors.append(GGUFTensorInfo(name, dims, type_id, offset))

    alignment = int(meta.get("general.alignment", 32) or 32)
    header_end = cur.pos
    return GGUFHeader(
        raw=bytes(cur.data[:header_end]) if keep_raw else None,
        version=version,
        metadata=meta,
        tensors=tensors,
        header_bytes=header_end,
        alignment=alignment,
        array_lengths=arr_lens,
        file_size=src.size,
    )


def parse_file(path: str) -> GGUFHeader:
    src = FileSource(path)
    try:
        return parse_header(src)
    finally:
        src.close()


# ---------------------------------------------------------------------------
# Writer (used by tests and `vram-fit` fixtures to build tiny header-only files)
# ---------------------------------------------------------------------------

def _w_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _w_value(v: Any) -> bytes:
    if isinstance(v, bool):
        return struct.pack("<I", _BOOL) + struct.pack("<?", v)
    if isinstance(v, int):
        if v < 0:
            return struct.pack("<I", _I32) + struct.pack("<i", v)
        if v < 2**32:
            return struct.pack("<I", _U32) + struct.pack("<I", v)
        return struct.pack("<I", _U64) + struct.pack("<Q", v)
    if isinstance(v, float):
        return struct.pack("<I", _F32) + struct.pack("<f", v)
    if isinstance(v, str):
        return struct.pack("<I", _STR) + _w_str(v)
    if isinstance(v, (list, tuple)):
        out = struct.pack("<I", _ARR)
        if v and isinstance(v[0], str):
            out += struct.pack("<I", _STR) + struct.pack("<Q", len(v))
            out += b"".join(_w_str(x) for x in v)
        elif v and isinstance(v[0], float):
            out += struct.pack("<I", _F32) + struct.pack("<Q", len(v))
            out += b"".join(struct.pack("<f", x) for x in v)
        else:
            out += struct.pack("<I", _I32) + struct.pack("<Q", len(v))
            out += b"".join(struct.pack("<i", int(x)) for x in v)
        return out
    raise TypeError(f"cannot encode {type(v)} in GGUF")


def build_header(metadata: Dict[str, Any], tensors: List[GGUFTensorInfo]) -> bytes:
    """Serialise a GGUF v3 header (no tensor data)."""
    out = bytearray(GGUF_MAGIC)
    out += struct.pack("<I", 3)
    out += struct.pack("<Q", len(tensors))
    out += struct.pack("<Q", len(metadata))
    for k, v in metadata.items():
        out += _w_str(k) + _w_value(v)
    for t in tensors:
        out += _w_str(t.name)
        out += struct.pack("<I", len(t.shape))
        out += b"".join(struct.pack("<Q", int(d)) for d in t.shape)
        out += struct.pack("<I", t.type_id)
        out += struct.pack("<Q", t.offset)
    return bytes(out)


def parse_bytes(data: bytes) -> GGUFHeader:
    return parse_header(BytesSource(data))

