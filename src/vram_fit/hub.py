"""Hugging Face Hub metadata fetching (never downloads weights).

* ``config.json`` and the model-info API are tiny JSON documents.
* GGUF and safetensors headers are read with HTTP ``Range`` requests, so a
  70 GB checkpoint costs a few hundred KB of traffic.
* Responses are cached under ``~/.cache/vram-fit`` (override with
  ``VRAM_FIT_CACHE``); ``HF_TOKEN`` unlocks gated repos; ``HF_ENDPOINT``
  points at a mirror.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from . import __version__
from .catalog import lookup as catalog_lookup
from .gguf import ByteSource, GGUFError, GGUFHeader, parse_bytes, parse_file, parse_header
from .model import (LoadedModel, inventory_from_safetensors, model_from_gguf, model_from_hf_config,
                    parse_safetensors_header)
from .quant import quant_from_filename
from .tensors import Tensor

DEFAULT_TIMEOUT = float(os.environ.get("VRAM_FIT_TIMEOUT", "15"))
CACHE_TTL = 7 * 24 * 3600
GGUF_PREFERENCE = ["Q4_K_M", "Q4_K_S", "IQ4_XS", "Q5_K_M", "Q4_0", "Q6_K", "Q8_0", "Q3_K_M", "BF16", "F16"]
_SPLIT_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


class HubError(RuntimeError):
    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


def endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def _token() -> Optional[str]:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok.strip()
    path = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "token")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


def cache_dir() -> str:
    base = os.environ.get("VRAM_FIT_CACHE")
    if not base:
        xdg = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        base = os.path.join(xdg, "vram-fit")
    os.makedirs(base, exist_ok=True)
    return base


def _cache_path(key: str, ext: str) -> str:
    return os.path.join(cache_dir(), hashlib.sha1(key.encode()).hexdigest() + ext)


def _cache_read(key: str, ext: str, ttl: float = CACHE_TTL) -> Optional[bytes]:
    if os.environ.get("VRAM_FIT_NO_CACHE"):
        return None
    p = _cache_path(key, ext)
    try:
        if time.time() - os.path.getmtime(p) > ttl:
            return None
        with open(p, "rb") as f:
            return f.read()
    except OSError:
        return None


def _cache_write(key: str, ext: str, data: bytes) -> None:
    if os.environ.get("VRAM_FIT_NO_CACHE"):
        return
    try:
        p = _cache_path(key, ext)
        tmp = p + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, p)
    except OSError:
        pass


def _request(url: str, headers: Optional[Dict[str, str]] = None, timeout: float = DEFAULT_TIMEOUT,
             max_bytes: Optional[int] = None) -> Tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": f"vram-fit/{__version__}"})
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    tok = _token()
    if tok and urllib.parse.urlparse(url).netloc == urllib.parse.urlparse(endpoint()).netloc:
        # unredirected: the token is never forwarded to CDN redirect targets
        req.add_unredirected_header("Authorization", f"Bearer {tok}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read(max_bytes) if max_bytes is not None else resp.read()
            return resp.status, data
    except urllib.error.HTTPError as e:
        msg = {401: "unauthorized (gated or private repo: set HF_TOKEN)",
               403: "forbidden (accept the model license on huggingface.co and set HF_TOKEN)",
               404: "not found"}.get(e.code, e.reason)
        raise HubError(f"{url}: HTTP {e.code} {msg}", e.code) from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise HubError(f"{url}: network error ({getattr(e, 'reason', e)})") from None


def get_json(url: str, use_cache: bool = True) -> Any:
    if use_cache:
        hit = _cache_read(url, ".json")
        if hit is not None:
            return json.loads(hit)
    _, data = _request(url)
    obj = json.loads(data)
    if use_cache:
        _cache_write(url, ".json", data)
    return obj


def resolve_url(repo: str, filename: str, revision: str = "main") -> str:
    return f"{endpoint()}/{repo}/resolve/{urllib.parse.quote(revision, safe='')}/{urllib.parse.quote(filename)}"


class HttpRangeSource(ByteSource):
    """Random access to a remote file through HTTP Range requests."""

    def __init__(self, url: str, size: Optional[int] = None):
        self.url = url
        self._size = size
        self.bytes_fetched = 0

    def read_at(self, offset: int, size: int) -> bytes:
        end = offset + size - 1
        if self._size is not None:
            end = min(end, self._size - 1)
            if offset > end:
                return b""
        status, data = _request(self.url, {"Range": f"bytes={offset}-{end}"}, max_bytes=end - offset + 1)
        if status == 200 and offset > 0:
            raise HubError(f"{self.url}: server ignored the Range header")
        self.bytes_fetched += len(data)
        return data

    @property
    def size(self) -> Optional[int]:
        return self._size


def fetch_gguf_header(url: str, size: Optional[int] = None) -> GGUFHeader:
    key = "gguf-header:" + url
    hit = _cache_read(key, ".gguf", ttl=30 * 24 * 3600)
    if hit is not None:
        try:
            hdr = parse_bytes(hit)
            hdr.file_size = size
            return hdr
        except GGUFError:
            pass
    src = HttpRangeSource(url, size)
    hdr = parse_header(src)
    raw = src.read_at(0, hdr.header_bytes)
    _cache_write(key, ".gguf", raw)
    hdr.file_size = size
    return hdr


def fetch_safetensors_header(url: str) -> Dict[str, Any]:
    key = "st-header:" + url
    hit = _cache_read(key, ".json", ttl=30 * 24 * 3600)
    if hit is not None:
        return json.loads(hit)
    src = HttpRangeSource(url)
    first = src.read_at(0, 8)
    n = int.from_bytes(first[:8], "little")
    if n <= 0 or n > 100 * 1024 * 1024:
        raise HubError(f"{url}: implausible safetensors header length {n}")
    body = src.read_at(8, n)
    hdr = parse_safetensors_header(first + body)
    _cache_write(key, ".json", json.dumps(hdr).encode())
    return hdr


# ---------------------------------------------------------------------------
# Model reference parsing
# ---------------------------------------------------------------------------

def parse_ref(ref: str) -> Dict[str, Optional[str]]:
    """Split ``org/repo[:QUANT]``, ``org/repo/path/file.gguf`` or a hub URL."""
    ref = ref.strip()
    out: Dict[str, Optional[str]] = {"repo": None, "file": None, "tag": None, "revision": "main", "url": None}
    if ref.startswith("http://") or ref.startswith("https://"):
        u = urllib.parse.urlparse(ref)
        hub = urllib.parse.urlparse(endpoint()).netloc
        parts = [p for p in u.path.split("/") if p]
        if u.netloc in (hub, "huggingface.co", "hf.co") and len(parts) >= 2:
            out["repo"] = "/".join(parts[:2])
            if len(parts) >= 5 and parts[2] in ("blob", "resolve"):
                out["revision"] = parts[3]
                out["file"] = "/".join(parts[4:])
            return out
        out["url"] = ref
        return out
    if ref.startswith("hf.co/"):
        ref = ref[len("hf.co/"):]
    if ":" in ref and not os.path.exists(ref):
        ref, tag = ref.rsplit(":", 1)
        out["tag"] = tag
    parts = ref.split("/")
    if len(parts) > 2 and ref.lower().endswith(".gguf"):
        out["repo"] = "/".join(parts[:2])
        out["file"] = "/".join(parts[2:])
    else:
        out["repo"] = ref
    return out


def list_gguf_files(siblings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}
    for s in siblings:
        name = s.get("rfilename", "")
        if not name.lower().endswith(".gguf"):
            continue
        size = int(s.get("size") or (s.get("lfs") or {}).get("size") or 0)
        m = _SPLIT_RE.search(name)
        if m:
            key = name[: m.start()]
            g = groups.setdefault(key, {"filename": None, "files": [], "size": 0, "splits": int(m.group(2))})
            g["files"].append(name)
            g["size"] += size
            if m.group(1) == "00001":
                g["filename"] = name
        else:
            groups[name] = {"filename": name, "files": [name], "size": size, "splits": 1}
    out = []
    for g in groups.values():
        if not g["filename"]:
            g["filename"] = sorted(g["files"])[0]
        g["files"].sort()
        fname = g["filename"]
        g["mmproj"] = "mmproj" in fname.lower()
        g["quant"] = quant_from_filename(fname.split("/")[-1]) or (quant_from_filename(fname) if "/" in fname else None)
        out.append(g)
    out.sort(key=lambda g: g["size"])
    return out


def _pick_gguf(files: List[Dict[str, Any]], tag: Optional[str], file: Optional[str]) -> Optional[Dict[str, Any]]:
    cands = [f for f in files if not f["mmproj"]]
    if file:
        for f in cands:
            if file in f["files"] or f["filename"] == file:
                return f
        return None
    if tag:
        t = tag.upper()
        for f in cands:
            if (f["quant"] or "").upper() == t:
                return f
        for f in cands:
            if t in f["filename"].upper():
                return f
        return None
    for pref in GGUF_PREFERENCE:
        for f in cands:
            if f["quant"] == pref:
                return f
    return cands[0] if cands else None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _local_gguf_split_paths(path: str) -> List[str]:
    m = _SPLIT_RE.search(path)
    if not m:
        return [path]
    n = int(m.group(2))
    base = path[: m.start()]
    paths = [f"{base}-{i:05d}-of-{n:05d}.gguf" for i in range(1, n + 1)]
    return [p for p in paths if os.path.exists(p)]


def load_local(path: str) -> LoadedModel:
    if os.path.isdir(path):
        cfg_path = os.path.join(path, "config.json")
        if os.path.exists(cfg_path):
            return load_local(cfg_path)
        ggufs = sorted(p for p in os.listdir(path) if p.lower().endswith(".gguf") and "mmproj" not in p.lower())
        if ggufs:
            return load_local(os.path.join(path, ggufs[0]))
        raise HubError(f"{path}: no config.json or .gguf file found")
    if path.lower().endswith(".gguf"):
        paths = _local_gguf_split_paths(path)
        headers = [parse_file(p) for p in paths]
        name = os.path.basename(path)
        lm = model_from_gguf(headers, headers[0].get("general.name") or name, filename=name)
        lm.source = "gguf-local"
        lm.path = os.path.abspath(path)
        return lm
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if path.endswith("model.safetensors.index.json"):
            raise HubError("pass the model directory or config.json, not the safetensors index")
        mid = os.path.basename(os.path.dirname(os.path.abspath(path))) or "local-model"
        lm = model_from_hf_config(cfg, mid, source="local-config")
        lm.path = os.path.dirname(os.path.abspath(path))
        return lm
    if path.lower().endswith(".safetensors"):
        with open(path, "rb") as f:
            first = f.read(8)
            n = int.from_bytes(first, "little")
            hdr = parse_safetensors_header(first + f.read(n))
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(path)), "config.json")
        if not os.path.exists(cfg_path):
            raise HubError("a safetensors file needs its config.json alongside it")
        lm = load_local(cfg_path)
        _apply_safetensors(lm, [hdr])
        return lm
    raise HubError(f"{path}: unsupported local file (expected .gguf, config.json or a model directory)")


def _apply_safetensors(lm: LoadedModel, headers: List[Dict[str, Any]]) -> None:
    inv = inventory_from_safetensors(headers)
    if not any(t.kind == "head" for t in inv):
        emb = next((t for t in inv if t.kind == "embed"), None)
        if emb is not None:
            inv.append(Tensor(name="output.weight", kind="head", layer=None, numel=emb.numel, n_in=emb.n_in,
                              n_out=emb.n_out, dtype=emb.dtype, nbytes_exact=emb.nbytes_exact, tied=True))
    lm.inventory = inv
    lm.exact = True
    if not lm.native_quant:
        dtypes = {t.dtype for t in inv if t.is_linear}
        lm.native_quant = "F16" if dtypes == {"F16"} else ("F32" if dtypes == {"F32"} else "BF16")
    lm.source = "safetensors"


def _catalog_fallback(ref: str, reason: str) -> LoadedModel:
    hit = catalog_lookup(ref)
    if not hit:
        raise HubError(f"could not load {ref!r}: {reason}. It is not in the offline catalog either "
                       "(see `vram-fit catalog`); pass a local config.json / .gguf instead.")
    cid, cfg = hit
    lm = model_from_hf_config(cfg, cid, source="catalog")
    lm.repo = ref if "/" in ref else cid
    if cid.lower() != ref.lower():
        lm.notes.append(f"Using the built-in {cid} architecture for {ref} ({reason}).")
    else:
        lm.notes.append(f"Using the built-in catalog config ({reason}).")
    return lm


def load_model(ref: str, quant: Optional[str] = None, exact: bool = False, offline: bool = False) -> LoadedModel:
    """Resolve any model reference into a :class:`LoadedModel` without downloading weights."""
    if os.path.exists(os.path.expanduser(ref)):
        return load_local(os.path.expanduser(ref))
    if offline or os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("VRAM_FIT_OFFLINE") == "1":
        return _catalog_fallback(ref, "offline mode")
    if "/" not in ref and not ref.startswith("http") and catalog_lookup(ref):
        return _catalog_fallback(ref, "alias")
    r = parse_ref(ref)
    tag = r["tag"] or (quant if quant and quant_from_filename(quant) else None)
    if r["url"]:
        hdr = fetch_gguf_header(r["url"])
        lm = model_from_gguf([hdr], hdr.get("general.name") or r["url"].rsplit("/", 1)[-1],
                             filename=r["url"].rsplit("/", 1)[-1])
        return lm
    repo = r["repo"] or ref
    rev = r["revision"] or "main"
    try:
        info = get_json(f"{endpoint()}/api/models/{repo}/revision/{urllib.parse.quote(rev, safe='')}?blobs=true")
    except HubError as e:
        if e.status in (401, 404):
            try:
                info = get_json(f"{endpoint()}/api/models/{repo}?blobs=true")
            except HubError as e2:
                return _catalog_fallback(repo, str(e2))
        else:
            return _catalog_fallback(repo, str(e))
    siblings = info.get("siblings") or []
    names = {s.get("rfilename") for s in siblings}
    gguf_files = list_gguf_files(siblings)
    gated = bool(info.get("gated"))
    st = info.get("safetensors") or {}
    reported = st.get("total") or (info.get("gguf") or {}).get("total")

    want_gguf = bool(r["file"]) or (tag is not None and bool(gguf_files)) or ("config.json" not in names and gguf_files)
    if want_gguf:
        pick = _pick_gguf(gguf_files, tag, r["file"])
        if pick is None:
            avail = ", ".join(sorted({f["quant"] or f["filename"] for f in gguf_files if not f["mmproj"]}))
            raise HubError(f"{repo}: no GGUF file matches {tag or r['file']!r}. Available: {avail}")
        sizes = {s.get("rfilename"): int(s.get("size") or (s.get("lfs") or {}).get("size") or 0) for s in siblings}

        def _hdr(fn: str) -> GGUFHeader:
            return fetch_gguf_header(resolve_url(repo, fn, rev), sizes.get(fn) or None)

        with cf.ThreadPoolExecutor(max_workers=8) as ex:
            headers = list(ex.map(_hdr, pick["files"]))
        lm = model_from_gguf(headers, repo, filename=pick["filename"])
        lm.repo = repo
        lm.gguf_files = [{k: f[k] for k in ("filename", "quant", "size", "splits", "mmproj")} for f in gguf_files]
        lm.params_reported = reported
        lm.gated = gated
        mm = [f for f in gguf_files if f["mmproj"]]
        if mm:
            lm.notes.append(f"Vision projector available ({mm[0]['filename']}, {mm[0]['size']/2**20:.0f} MiB); "
                            "add it to the budget if you pass --mmproj.")
        return lm

    if "config.json" not in names:
        return _catalog_fallback(repo, "repo has neither config.json nor .gguf files")
    try:
        cfg = get_json(resolve_url(repo, "config.json", rev))
    except HubError as e:
        return _catalog_fallback(repo, str(e))
    lm = model_from_hf_config(cfg, repo, source="hf-config", params_reported=reported)
    lm.repo = repo
    lm.gated = gated
    lm.gguf_files = [{k: f[k] for k in ("filename", "quant", "size", "splits", "mmproj")} for f in gguf_files]
    if exact:
        shards = sorted(n for n in names if n and n.endswith(".safetensors"))
        if "model.safetensors.index.json" in names:
            idx = get_json(resolve_url(repo, "model.safetensors.index.json", rev))
            shards = sorted(set((idx.get("weight_map") or {}).values()))
        if shards:
            with cf.ThreadPoolExecutor(max_workers=8) as ex:
                hdrs = list(ex.map(lambda fn: fetch_safetensors_header(resolve_url(repo, fn, rev)), shards))
            _apply_safetensors(lm, hdrs)
            lm.repo = repo
    return lm


def load_gguf_variant(model: LoadedModel, quant: str) -> Optional[LoadedModel]:
    """Fetch the exact header of another quant in the same GGUF repo (used by the optimizer)."""
    if not model.repo or not model.gguf_files:
        return None
    for f in model.gguf_files:
        if not f.get("mmproj") and (f.get("quant") or "").upper() == quant.upper():
            if model.filename == f["filename"]:
                return model
            try:
                return load_model(f"{model.repo}:{quant}")
            except (HubError, GGUFError, ValueError):
                return None
    return None
