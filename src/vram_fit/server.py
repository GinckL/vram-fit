"""HTTP + WebSocket API that powers the vram-fit dashboard.

REST (JSON in, JSON out):
    GET  /api/health                 version + capability flags
    GET  /api/hardware[?refresh=1]   probed GPUs / RAM / PCIe
    GET  /api/presets                simulated GPU presets
    GET  /api/quants[?engine=]       quantization formats
    GET  /api/engines                engines, modes, KV types, optimizers
    GET  /api/catalog                offline model catalog
    GET  /api/model?id=&quant=       resolve a model (metadata only)
    POST /api/estimate               {model, gpu, settings, limits}
    POST /api/optimize               {model, gpu, settings, prefer, lock_quant, lock_kv, global_batch}
    POST /api/command                {model, gpu, settings}
    POST /api/matrix                 {model, gpu, settings, contexts, quants}
    GET  /api/telemetry              one live GPU/RAM sample

WebSocket /ws — messages are JSON objects with an ``op`` and an optional ``rid``
(echoed back so the client can drop stale replies):
    {"op":"hello"}
    {"op":"load_model","rid":1,"model":"Qwen/Qwen2.5-14B"}      -> progress..., model
    {"op":"estimate","rid":2,"model":"...","gpu":"rtx-4090","settings":{...}}
    {"op":"optimize","rid":3,...} / {"op":"matrix",...} / {"op":"command",...}
    {"op":"telemetry","enabled":true,"interval_ms":1000}        -> telemetry stream
``estimate`` and ``matrix`` requests are coalesced: while one is computing,
only the newest pending request per op is kept, so dragging a slider never
builds a backlog.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
import webbrowser
from collections import OrderedDict
from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from . import __version__
from .catalog import list_catalog
from .commands import build_command
from .estimator import ENGINES, GC_MODES, MODES, OPTIMIZERS, Settings, estimate, normalize_engine
from .ggml import KV_CACHE_TYPES
from .gguf import GGUFError
from .hardware import HardwareProfile, cpu_only_gpu, probe, sample_telemetry
from .hub import HubError, load_model
from .model import LoadedModel
from .optimizer import PREFERENCES, optimize
from .presets import GPU, find_preset, list_presets, parse_gpu_spec
from .quant import QUANTS, quants_for_engine

WEB_DIST = os.environ.get("VRAM_FIT_WEB_DIST") or os.path.join(os.path.dirname(__file__), "web", "dist")
KV_OPTIONS = {
    "llama.cpp": list(KV_CACHE_TYPES), "ollama": ["f16", "q8_0", "q4_0"],
    "vllm": ["auto", "fp8", "fp8_e4m3", "fp8_e5m2"], "exllamav2": ["FP16", "Q8", "Q6", "Q4"],
    "unsloth": ["bf16"], "transformers": ["bf16"],
}


class State:
    def __init__(self, max_models: int = 32):
        self.models: "OrderedDict[Tuple[str, bool, Optional[str]], LoadedModel]" = OrderedDict()
        self.max_models = max_models
        self.hw: Optional[HardwareProfile] = None
        self.lock = threading.Lock()
        self.allow_local = True  # run() turns this off when the server listens on a non-loopback interface

    def hardware(self, refresh: bool = False) -> HardwareProfile:
        with self.lock:
            if self.hw is None or refresh:
                self.hw = probe()
            return self.hw

    def _get(self, key) -> Optional[LoadedModel]:
        with self.lock:
            hit = self.models.get(key)
            if hit is not None:
                self.models.move_to_end(key)
            return hit

    def _put(self, key, lm: LoadedModel) -> None:
        with self.lock:
            self.models[key] = lm
            while len(self.models) > self.max_models:
                self.models.popitem(last=False)

    def model(self, ref: str, quant: Optional[str] = None, exact: bool = False) -> LoadedModel:
        """Load once per reference; GGUF repos load one extra header per requested quant file."""
        if not isinstance(ref, str) or not ref.strip() or len(ref) > 1024:
            raise ValueError("model must be a non-empty string (HF repo id, alias, URL or local path)")
        ref = ref.strip()
        if not self.allow_local and _looks_local(ref):
            raise ValueError("local file paths are disabled because the server listens on a network interface; "
                             "restart with --allow-local-files to permit them")
        base = self._get((ref, exact, None))
        if base is not None:
            variant = (base.source == "gguf" and quant and quant != base.native_quant
                       and any((f.get("quant") or "").upper() == quant.upper() for f in base.gguf_files))
            if not variant:
                return base
            hit = self._get((ref, exact, quant.upper()))
            if hit is not None:
                return hit
            lm = load_model(ref, quant=quant, exact=exact)
            self._put((ref, exact, quant.upper()), lm)
            return lm
        lm = load_model(ref, quant=quant, exact=exact)
        self._put((ref, exact, None), lm)
        if lm.native_quant:
            self._put((ref, exact, lm.native_quant.upper()), lm)
        return lm

    def gpu(self, spec: Any) -> Tuple[GPU, Optional[HardwareProfile]]:
        """``"auto"`` | ``"cpu"`` | preset string | {"preset": id, "count": n, ...overrides}."""
        if spec is None or spec == "auto" or (isinstance(spec, dict) and spec.get("preset") in (None, "auto")
                                              and not spec.get("vram_gib")):
            hw = self.hardware()
            g = hw.primary()
            if isinstance(spec, dict):
                g = _apply_overrides(g, spec)
            return g, hw
        if spec == "cpu":
            hw = self.hardware()
            return cpu_only_gpu(hw.ram_bandwidth_gbs, hw.cpu_physical_cores), hw
        if isinstance(spec, str):
            return parse_gpu_spec(spec), None
        if isinstance(spec, dict):
            base = spec.get("preset")
            if base:
                g = parse_gpu_spec(str(base))
            else:
                g = parse_gpu_spec(f"custom:{spec.get('vram_gib', 24)}:{spec.get('bandwidth_gbs', 500)}:{spec.get('tflops', 50)}")
            return _apply_overrides(g, spec), None
        raise ValueError(f"invalid gpu spec {spec!r}")


_LOCAL_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|[\\/]|\.{1,2}[\\/]|~)")


def _looks_local(ref: str) -> bool:
    return bool(_LOCAL_RE.match(ref)) or ref.lower().endswith((".gguf", ".json", ".safetensors")) and "/" not in ref \
        or os.path.exists(os.path.expanduser(ref))


def _apply_overrides(g: GPU, spec: Dict[str, Any]) -> GPU:
    fields = {}
    for k in ("vram_gib", "bandwidth_gbs", "tflops", "pcie_gen", "pcie_width", "ceiling_gib", "count"):
        if spec.get(k) is not None:
            fields[k] = type(getattr(g, k) if getattr(g, k) is not None else 0.0)(spec[k])
    if "vram_gib" in fields and g.unified and "ceiling_gib" not in fields:
        fields["ceiling_gib"] = fields["vram_gib"] * 0.75
    return replace(g, **fields) if fields else g


STATE = State()


def _settings(d: Optional[dict]) -> Settings:
    st = Settings.from_dict(d or {})
    st.engine = normalize_engine(st.engine)
    return st


def _err(exc: Exception) -> str:
    return exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)


def do_estimate(body: dict) -> dict:
    model = STATE.model(body["model"], (body.get("settings") or {}).get("quant"), bool(body.get("exact")))
    gpu, hw = STATE.gpu(body.get("gpu", "auto"))
    t0 = time.perf_counter()
    res = estimate(model, gpu, _settings(body.get("settings")), hw, limits=bool(body.get("limits", True)))
    res["compute_ms"] = round((time.perf_counter() - t0) * 1000, 3)
    if body.get("command"):
        res["launch"] = build_command(model, res, hw)
    return res


def do_optimize(body: dict) -> dict:
    st = _settings(body.get("settings"))
    model = STATE.model(body["model"], st.quant, bool(body.get("exact")))
    gpu, hw = STATE.gpu(body.get("gpu", "auto"))
    prefer = body.get("prefer", "balanced")
    if prefer not in PREFERENCES:
        raise ValueError(f"prefer must be one of {PREFERENCES}")
    return optimize(model, gpu, st, hw, prefer=prefer, lock_quant=bool(body.get("lock_quant")),
                    lock_kv=bool(body.get("lock_kv")), target_global_batch=int(body.get("global_batch", 16)))


def do_command(body: dict) -> dict:
    st = _settings(body.get("settings"))
    model = STATE.model(body["model"], st.quant, bool(body.get("exact")))
    gpu, hw = STATE.gpu(body.get("gpu", "auto"))
    res = estimate(model, gpu, st, hw, limits=False)
    return {"estimate": res, "command": build_command(model, res, hw, max(1, st.grad_accum))}


def do_matrix(body: dict) -> dict:
    base = _settings(body.get("settings"))
    model = STATE.model(body["model"], base.quant, bool(body.get("exact")))
    gpu, hw = STATE.gpu(body.get("gpu", "auto"))
    contexts = [int(c) for c in (body.get("contexts") or [4096, 8192, 16384, 32768, 65536, 131072])]
    quants = body.get("quants") or [q.name for q in quants_for_engine(base.engine) if q.family in ("gguf", "fp", "awq", "gptq", "fp8", "nf4", "exl2")][:12]
    cells = []
    for q in quants:
        for c in contexts:
            r = estimate(model, gpu, replace(base, quant=q, context=c), hw, limits=False)
            cells.append({"quant": q, "context": c, "status": r["status"], "gpu_total_mib": r["gpu_total_mib"],
                          "gpu_layers": r["settings"]["gpu_layers"],
                          "tok_s": r["performance"].get("decode_tok_s") or r["performance"].get("train_tok_s")})
    return {"model": model.id, "gpu": gpu.name, "contexts": contexts, "quants": quants, "cells": cells}


def engines_info() -> dict:
    return {"engines": list(ENGINES), "modes": list(MODES), "optimizers": list(OPTIMIZERS),
            "grad_checkpointing": list(GC_MODES), "kv_cache": KV_OPTIONS, "preferences": list(PREFERENCES),
            "quants_by_engine": {e: [q.name for q in quants_for_engine(e)] for e in ENGINES}}


log = logging.getLogger("vram_fit.server")


def create_app(trusted_hosts: Optional[list] = None) -> FastAPI:
    app = FastAPI(title="vram-fit", version=__version__,
                  description="Exact VRAM / KV-cache / tokens-per-second estimator for local LLMs. "
                              "Created by Azimxon Vaxobov (Ginckl).",
                  contact={"name": "Azimxon Vaxobov (Ginckl)", "url": "https://azimxonvaxobov.netlify.app"},
                  license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"})
    if trusted_hosts:
        from starlette.middleware.trustedhost import TrustedHostMiddleware
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)
    app.add_middleware(CORSMiddleware, allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?",
                       allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(HubError)
    async def _hub_err(_, exc: HubError):
        return JSONResponse({"error": {"type": "HubError", "message": str(exc)}}, status_code=502)

    def guard(fn, *args):
        try:
            return fn(*args)
        except (KeyError, ValueError, TypeError, GGUFError) as exc:
            raise HTTPException(status_code=400, detail=_err(exc))

    @app.get("/api/health")
    def health():
        return {"ok": True, "version": __version__, "gui_bundle": os.path.exists(os.path.join(WEB_DIST, "index.html"))}

    @app.get("/api/hardware")
    def hardware(refresh: bool = False):
        hw = STATE.hardware(refresh)
        return {"hardware": hw.to_dict(), "primary": hw.primary().to_dict()}

    @app.get("/api/presets")
    def presets(vendor: Optional[str] = None):
        return [g.to_dict() for g in list_presets(vendor)]

    @app.get("/api/presets/{query}")
    def preset(query: str):
        g = find_preset(query)
        if g is None:
            raise HTTPException(404, f"no preset matches {query!r}")
        return g.to_dict()

    @app.get("/api/quants")
    def quants(engine: Optional[str] = None):
        qs = quants_for_engine(normalize_engine(engine)) if engine else list(QUANTS.values())
        return [q.to_dict() for q in qs]

    @app.get("/api/engines")
    def engines():
        return engines_info()

    @app.get("/api/catalog")
    def catalog():
        return list_catalog()

    @app.get("/api/model")
    async def model(id: str, quant: Optional[str] = None, exact: bool = False):
        lm = await asyncio.to_thread(guard, STATE.model, id, quant, exact)
        return lm.to_dict()

    @app.post("/api/estimate")
    async def post_estimate(body: dict = Body(...)):
        return await asyncio.to_thread(guard, do_estimate, body)

    @app.post("/api/optimize")
    async def post_optimize(body: dict = Body(...)):
        return await asyncio.to_thread(guard, do_optimize, body)

    @app.post("/api/command")
    async def post_command(body: dict = Body(...)):
        return await asyncio.to_thread(guard, do_command, body)

    @app.post("/api/matrix")
    async def post_matrix(body: dict = Body(...)):
        return await asyncio.to_thread(guard, do_matrix, body)

    @app.get("/api/telemetry")
    async def telemetry():
        return await asyncio.to_thread(sample_telemetry)

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        send_lock = asyncio.Lock()
        pending: Dict[str, dict] = {}
        wake = asyncio.Event()
        tele: Dict[str, Any] = {"task": None}

        async def send(obj: dict) -> None:
            async with send_lock:
                await ws.send_text(json.dumps(obj, default=str, separators=(",", ":")))

        async def handle(msg: dict) -> None:
            op = msg.get("op")
            rid = msg.get("rid")
            t0 = time.perf_counter()
            try:
                if op == "hello":
                    hw = await asyncio.to_thread(STATE.hardware)
                    data = {"version": __version__, "hardware": hw.to_dict(), "primary": hw.primary().to_dict(),
                            "presets": [g.to_dict() for g in list_presets()], **engines_info()}
                elif op == "load_model":
                    await send({"event": "progress", "rid": rid, "stage": "resolving", "model": msg.get("model")})
                    lm = await asyncio.to_thread(STATE.model, msg["model"], msg.get("quant"), bool(msg.get("exact")))
                    await send({"event": "progress", "rid": rid, "stage": "parsed", "source": lm.source})
                    data = lm.to_dict()
                    op = "model"
                elif op == "estimate":
                    data = await asyncio.to_thread(do_estimate, msg)
                elif op == "optimize":
                    data = await asyncio.to_thread(do_optimize, msg)
                elif op == "command":
                    data = await asyncio.to_thread(do_command, msg)
                elif op == "matrix":
                    data = await asyncio.to_thread(do_matrix, msg)
                elif op == "hardware":
                    hw = await asyncio.to_thread(STATE.hardware, bool(msg.get("refresh")))
                    data = {"hardware": hw.to_dict(), "primary": hw.primary().to_dict()}
                elif op == "telemetry":
                    if tele["task"]:
                        tele["task"].cancel()
                        tele["task"] = None
                    if msg.get("enabled", True):
                        interval = max(0.1, float(msg.get("interval_ms", 1000)) / 1000)
                        tele["task"] = asyncio.create_task(telemetry_loop(interval))
                    data = {"enabled": bool(tele["task"])}
                elif op == "ping":
                    data = {"t": msg.get("t")}
                else:
                    raise ValueError(f"unknown op {op!r}")
                await send({"event": op, "rid": rid, "data": data,
                            "ms": round((time.perf_counter() - t0) * 1000, 2)})
            except (KeyError, ValueError, TypeError, GGUFError, HubError, OSError) as exc:
                await _safe_send({"event": "error", "rid": rid, "op": op, "message": _err(exc)})
            except Exception:  # never let one bad request kill the connection's worker
                log.exception("websocket op %r failed", op)
                await _safe_send({"event": "error", "rid": rid, "op": op, "message": "internal error (see server log)"})

        async def _safe_send(obj: dict) -> None:
            try:
                await send(obj)
            except (RuntimeError, WebSocketDisconnect, OSError):
                pass

        async def telemetry_loop(interval: float) -> None:
            try:
                while True:
                    sample = await asyncio.to_thread(sample_telemetry)
                    await send({"event": "telemetry", "data": sample, "t": time.time()})
                    await asyncio.sleep(interval)
            except (RuntimeError, WebSocketDisconnect, OSError):
                return

        async def worker() -> None:
            while True:
                await wake.wait()
                wake.clear()
                while pending:
                    key = next(iter(pending))
                    msg = pending.pop(key)
                    await handle(msg)

        worker_task = asyncio.create_task(worker())
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                    if not isinstance(msg, dict):
                        raise ValueError("message must be a JSON object")
                except ValueError as exc:
                    await send({"event": "error", "rid": None, "op": None, "message": f"bad message: {exc}"})
                    continue
                if msg.get("op") in ("estimate", "matrix"):
                    pending[f"{msg['op']}:{msg.get('channel', '')}"] = msg
                    wake.set()
                else:
                    asyncio.create_task(handle(msg))
        except WebSocketDisconnect:
            pass
        finally:
            worker_task.cancel()
            if tele["task"]:
                tele["task"].cancel()

    index = os.path.join(WEB_DIST, "index.html")
    if os.path.exists(index):
        from fastapi.staticfiles import StaticFiles
        app.mount("/assets", StaticFiles(directory=os.path.join(WEB_DIST, "assets"), check_dir=False), name="assets")

        @app.get("/", include_in_schema=False)
        def root():
            return FileResponse(index)
    else:
        @app.get("/", include_in_schema=False)
        def root_api():
            return {"name": "vram-fit", "version": __version__, "openapi": "/docs", "websocket": "/ws",
                    "endpoints": ["/api/health", "/api/hardware", "/api/presets", "/api/quants", "/api/engines",
                                  "/api/catalog", "/api/model", "/api/estimate", "/api/optimize", "/api/command",
                                  "/api/matrix", "/api/telemetry"],
                    "dashboard": f"no GUI bundle at {WEB_DIST}"}
    return app


LOOPBACK = ("127.0.0.1", "localhost", "::1")


def run(host: str = "127.0.0.1", port: int = 7860, open_browser: bool = False,
        allow_local_files: Optional[bool] = None) -> None:
    """Serve the API. On loopback, Host headers are pinned to localhost (blocks DNS rebinding);
    on any other interface, local file paths are refused unless explicitly allowed."""
    import uvicorn
    loopback = host in LOOPBACK
    STATE.allow_local = loopback if allow_local_files is None else allow_local_files
    trusted = ["127.0.0.1", "localhost", "::1", "[::1]"] if loopback else None
    app = create_app(trusted_hosts=trusted)
    shown = "localhost" if host in ("0.0.0.0", "::") else host
    url = f"http://{shown}:{port}/"
    print(f"vram-fit {__version__} API on {url}  (WebSocket {url.replace('http', 'ws', 1)}ws, OpenAPI {url}docs)")
    if not loopback:
        print(f"warning: listening on {host} without authentication; anyone who can reach this port can use the API"
              + ("" if STATE.allow_local else " (local file paths disabled)"))
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
