"""``vram-fit`` command line.

Every sub-command accepts ``--json`` and then prints exactly one JSON document
to stdout (errors included), so AI agents and CI jobs can parse results.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import replace
from typing import List, Optional

from . import __version__
from .catalog import list_catalog
from .estimator import ENGINES, GC_MODES, MODES, OPTIMIZERS, Settings, estimate, normalize_engine
from .gguf import GGUFError
from .hardware import HardwareProfile, cpu_only_gpu, probe
from .hub import HubError, load_model
from .optimizer import PREFERENCES, optimize
from .presets import list_presets, parse_gpu_spec
from .quant import QUANTS, quants_for_engine
from .render import Style, render_estimate, render_table, status_cell

EXIT_OK, EXIT_ERROR, EXIT_NO_FIT = 0, 2, 3


def parse_tokens(v: str) -> int:
    """``32k`` -> 32768, ``128K`` -> 131072, ``1.5k`` -> 1536, ``8192`` -> 8192."""
    s = str(v).strip().lower().replace("_", "").replace(",", "")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([km]?)", s)
    if not m:
        raise argparse.ArgumentTypeError(f"invalid token count {v!r} (examples: 8192, 32k, 128k)")
    n = float(m.group(1)) * {"": 1, "k": 1024, "m": 1024 * 1024}[m.group(2)]
    return int(n)


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("model", help="HF repo id (org/name[:QUANT]), alias (llama3.1-8b), hub URL, local .gguf / config.json / model dir")
    p.add_argument("--exact", action="store_true", help="read every safetensors shard header for exact tensor shapes")
    p.add_argument("--offline", action="store_true", help="never touch the network; use the built-in catalog")


def _add_hw_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("hardware")
    g.add_argument("--gpu", default="auto",
                   help="'auto' (probe this machine), a preset ('rtx-4090', 'm3-max-64gb', '2x rtx-3090'), "
                        "'cpu', or 'custom:VRAM_GB:BANDWIDTH_GBs[:TFLOPS]'")
    g.add_argument("--vram", type=float, help="override VRAM per GPU in GiB")
    g.add_argument("--bandwidth", type=float, help="override GPU memory bandwidth in GB/s")
    g.add_argument("--ram", type=float, help="system RAM in GiB (default: probed, or 32 for simulated GPUs)")
    g.add_argument("--ram-bw", type=float, help="system RAM bandwidth in GB/s (default: probed or 50)")
    g.add_argument("--background", type=float, help="VRAM already used by other processes, MiB")


def _add_settings_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("run settings")
    g.add_argument("--engine", "-e", default="llama.cpp", help=f"one of: {', '.join(ENGINES)}")
    g.add_argument("--mode", "-m", default="inference", choices=MODES)
    g.add_argument("--quant", "-q", help="Q4_K_M, Q8_0, IQ4_XS, AWQ, GPTQ-4bit, FP8, NF4, EXL2-4.65, BF16 ... (vram-fit quants)")
    g.add_argument("--ctx", "-c", type=parse_tokens, default=8192, help="context length per sequence (2k..128k)")
    g.add_argument("--batch", "-b", type=int, default=1, help="parallel sequences (inference) or micro-batch (training)")
    g.add_argument("--kv", help="KV cache type or bits: 16 | 8 | 4 | f16 | q8_0 | q4_0 | fp8 | Q4 | Q6 | Q8")
    g.add_argument("--ngl", type=int, help="llama.cpp GPU layers (default: auto = max that fits)")
    g.add_argument("--cpu-moe", type=int, default=0, help="llama.cpp --n-cpu-moe: keep experts of first N layers in RAM")
    g.add_argument("--no-fa", action="store_true", help="disable flash attention")
    g.add_argument("--ubatch", type=int, default=512, help="llama.cpp physical micro-batch (-ub)")
    g.add_argument("--swa-full", action="store_true", help="llama.cpp: full-size cache for sliding-window layers")
    g.add_argument("--gpu-util", type=float, default=0.90, help="vLLM --gpu-memory-utilization")
    g.add_argument("--max-batched-tokens", type=int, help="vLLM --max-num-batched-tokens")
    g.add_argument("--enforce-eager", action="store_true", help="vLLM: disable CUDA graphs")
    t = p.add_argument_group("fine-tuning")
    t.add_argument("--rank", "-r", type=int, default=16, help="LoRA rank")
    t.add_argument("--targets", help="comma-separated LoRA target modules (default: all linear)")
    t.add_argument("--optim", default="adamw_8bit", help=f"one of: {', '.join(OPTIMIZERS)}")
    t.add_argument("--gc", choices=GC_MODES, help="gradient checkpointing (default: unsloth for Unsloth, standard otherwise)")
    t.add_argument("--dataset-tokens", type=parse_tokens, help="tokens per epoch, for time-per-epoch estimates")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--json", action="store_true", help="machine-readable JSON output")
    p.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    p.add_argument("--fail-on-oom", action="store_true", help=f"exit {EXIT_NO_FIT} when the result is spill/oom")


def _settings(a: argparse.Namespace) -> Settings:
    return Settings(
        engine=normalize_engine(a.engine), mode=a.mode, quant=a.quant, context=a.ctx, batch=a.batch,
        kv_cache=a.kv, gpu_layers=a.ngl, cpu_moe_layers=a.cpu_moe, flash_attention=not a.no_fa,
        ubatch=a.ubatch, lora_rank=a.rank,
        lora_targets=[t.strip() for t in a.targets.split(",")] if a.targets else None,
        optimizer=a.optim, grad_checkpointing=a.gc, gpu_mem_util=a.gpu_util,
        max_batched_tokens=a.max_batched_tokens, enforce_eager=a.enforce_eager, swa_full=a.swa_full,
        background_mib=a.background, ram_total_gib=a.ram, ram_bandwidth_gbs=a.ram_bw,
        dataset_tokens=a.dataset_tokens,
    )


def resolve_hardware(a: argparse.Namespace):
    hw: Optional[HardwareProfile] = None
    spec = (a.gpu or "auto").strip()
    if spec.lower() == "auto":
        hw = probe()
        gpu = hw.primary()
    elif spec.lower() in ("cpu", "none"):
        hw = probe()
        gpu = cpu_only_gpu(hw.ram_bandwidth_gbs, hw.cpu_physical_cores)
    else:
        gpu = parse_gpu_spec(spec)
    if a.vram is not None:
        gpu = replace(gpu, vram_gib=a.vram, ceiling_gib=None if not gpu.unified else a.vram * 0.75)
    if a.bandwidth is not None:
        gpu = replace(gpu, bandwidth_gbs=a.bandwidth)
    return gpu, hw


def _emit(a: argparse.Namespace, payload: dict, text: str) -> None:
    if a.json:
        json.dump(payload, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        print(text)


def _exit_for(a: argparse.Namespace, status: str) -> int:
    return EXIT_NO_FIT if getattr(a, "fail_on_oom", False) and status in ("spill", "oom") else EXIT_OK


# ---------------------------------------------------------------------------
# sub-commands
# ---------------------------------------------------------------------------

def cmd_estimate(a: argparse.Namespace) -> int:
    sty = Style(False if a.no_color else None)
    st = _settings(a)
    model = load_model(a.model, quant=a.quant, exact=a.exact, offline=a.offline)
    gpu, hw = resolve_hardware(a)
    res = estimate(model, gpu, st, hw)
    cmd = None
    if a.command:
        from .commands import build_command
        cmd = build_command(model, res, hw)
    payload = {"vram_fit": __version__, "command": "estimate", "result": res}
    if cmd:
        payload["launch"] = cmd
    _emit(a, payload, render_estimate(res, sty, cmd))
    return _exit_for(a, res["status"])


def cmd_fit(a: argparse.Namespace) -> int:
    sty = Style(False if a.no_color else None)
    st = _settings(a)
    model = load_model(a.model, quant=a.quant, exact=a.exact, offline=a.offline)
    gpu, hw = resolve_hardware(a)
    out = optimize(model, gpu, st, hw, prefer=a.prefer, lock_quant=bool(a.quant), lock_kv=bool(a.kv),
                   target_global_batch=a.global_batch)
    text = [render_estimate(out["best"], sty, out["command"]), ""]
    text.append(sty.bold("Why:"))
    text += ["  " + e for e in out["explanation"]]
    if out["alternatives"] and len(out["alternatives"]) > 1:
        rows = []
        for alt in out["alternatives"]:
            tok = alt.get("decode_tok_s") or alt.get("train_tok_s")
            rows.append([status_cell(alt["status"], alt["status"], sty),
                         str(alt.get("strategy") or alt["quant"]), str(alt["kv_cache"] or "-"),
                         str(alt["gpu_layers"]), str(alt["cpu_moe_layers"] or "-"),
                         f"{alt['gpu_total_mib'] / 1024:,.2f}", f"{tok:,.1f}" if tok else "-"])
        text += ["", sty.bold("Alternatives considered:"),
                 render_table(["status", "quant", "kv", "ngl", "cpu-moe", "GPU GiB", "tok/s"], rows, sty)]
    payload = {"vram_fit": __version__, "command": "fit", "result": out}
    _emit(a, payload, "\n".join(text))
    return _exit_for(a, out["best"]["status"])


def cmd_matrix(a: argparse.Namespace) -> int:
    sty = Style(False if a.no_color else None)
    base = _settings(a)
    model = load_model(a.model, quant=a.quant, exact=a.exact, offline=a.offline)
    gpu, hw = resolve_hardware(a)
    engine = base.engine
    if a.quants:
        quants = [q.strip() for q in a.quants.split(",")]
    elif model.gguf_files and engine in ("llama.cpp", "ollama"):
        quants = [f["quant"] for f in model.gguf_files if f.get("quant") and not f.get("mmproj")]
    elif engine in ("llama.cpp", "ollama"):
        quants = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "IQ4_XS", "Q3_K_M", "Q2_K"]
    elif engine == "vllm":
        quants = ["BF16", "FP8", "AWQ", "GPTQ-4bit"]
    elif engine == "exllamav2":
        quants = ["EXL2-8", "EXL2-6", "EXL2-5", "EXL2-4", "EXL2-3"]
    else:
        quants = ["BF16", "NF4"]
    ctxs = [parse_tokens(c) for c in a.contexts.split(",")]
    cells = []
    rows = []
    for q in quants:
        row = [q]
        for c in ctxs:
            r = estimate(model, gpu, replace(base, quant=q, context=c), hw, limits=False)
            tok = r["performance"].get("decode_tok_s") or r["performance"].get("train_tok_s") or 0
            cells.append({"quant": q, "context": c, "status": r["status"], "gpu_total_mib": r["gpu_total_mib"],
                          "gpu_layers": r["settings"]["gpu_layers"], "tok_s": tok})
            row.append(status_cell(r["status"], f"{r['gpu_total_mib'] / 1024:5.1f}G {tok:6.1f}t/s", sty))
        rows.append(row)
    head = ["quant"] + [f"{c // 1024}k" if c % 1024 == 0 else str(c) for c in ctxs]
    title = sty.bold(f"{model.id} on {gpu.name} ({engine}, {base.mode})") + sty.dim(
        "   ● fits  ◐ tight  ◑ offload  ○ spill  ✕ oom   (GPU GiB · decode tok/s)")
    payload = {"vram_fit": __version__, "command": "matrix", "result": {
        "model": model.id, "gpu": gpu.name, "engine": engine, "mode": base.mode, "contexts": ctxs,
        "quants": quants, "cells": cells}}
    _emit(a, payload, title + "\n" + render_table(head, rows, sty))
    return EXIT_OK


def cmd_inspect(a: argparse.Namespace) -> int:
    sty = Style(False if a.no_color else None)
    model = load_model(a.model, quant=a.quant, exact=a.exact, offline=a.offline)
    d = model.to_dict()
    s = model.spec
    by_kind = {}
    for t in model.inventory:
        if t.tied:
            continue
        k = by_kind.setdefault(t.kind, {"tensors": 0, "params": 0})
        k["tensors"] += 1
        k["params"] += t.numel
    d["params_by_kind"] = by_kind
    d["kv_bytes_per_token_f16"] = sum(s.kv_elems_per_token(i) * 2 for i in range(s.num_layers))
    lines = [sty.bold(model.id) + sty.dim(f"  ({model.source}{', exact tensors' if model.exact else ''})")]
    rows = [["architecture", s.architecture], ["params", f"{model.params_total / 1e9:.3f}B"],
            ["active params", f"{model.params_active / 1e9:.3f}B"], ["layers", str(s.num_layers)],
            ["hidden", str(s.hidden_size)], ["attention", s.attention_kind()],
            ["heads / kv heads / head dim", f"{s.num_heads} / {max(s.num_kv_heads)} / {s.head_dim}"],
            ["ffn", str(s.intermediate_size)], ["vocab", f"{s.vocab_size:,}"],
            ["max context", f"{s.max_context:,}"], ["rope", f"theta={s.rope_theta} scaling={s.rope_scaling}"],
            ["layer types", ", ".join(f"{k}={v}" for k, v in d['spec']['layer_types'].items())],
            ["KV / token (f16)", f"{d['kv_bytes_per_token_f16'] / 1024:,.1f} KiB"]]
    if s.is_moe:
        rows.append(["experts", f"{s.num_experts} (top-{s.experts_per_token}, ffn {s.moe_intermediate_size}"
                                f"{', shared ' + str(s.shared_expert_intermediate_size) if s.shared_expert_intermediate_size else ''})"])
    if s.sliding_window:
        rows.append(["sliding window", f"{s.sliding_window:,}"])
    if model.native_quant:
        rows.append(["native quant", model.native_quant])
    if s.vision_params:
        rows.append(["vision tower", f"{s.vision_params / 1e9:.2f}B params"])
    lines.append(render_table(["field", "value"], rows, sty))
    if model.gguf_files:
        lines.append("")
        lines.append(render_table(["GGUF file", "quant", "size"],
                                  [[f["filename"], f["quant"] or "?", f"{f['size'] / 2**30:,.2f} GiB"]
                                   for f in model.gguf_files], sty))
    for n in model.notes:
        lines.append(sty.dim("· " + n))
    _emit(a, {"vram_fit": __version__, "command": "inspect", "result": d}, "\n".join(lines))
    return EXIT_OK


def cmd_hw(a: argparse.Namespace) -> int:
    sty = Style(False if a.no_color else None)
    hw = probe()
    lines = [sty.bold("Hardware") + sty.dim(f"  {hw.os} · backend {hw.backend}" + (f" · driver {hw.driver}" if hw.driver else ""))]
    rows = [[g.id, g.name, f"{g.vram_gib:,.1f}", f"{g.vram_free_gib:,.1f}" if g.vram_free_gib is not None else "-",
             f"{g.usable_gib:,.1f}", f"{g.bandwidth_gbs:,.0f}", f"{g.tflops:,.1f}",
             f"Gen{g.pcie_gen} x{g.pcie_width}" if g.pcie_gen else "unified", g.compute_capability or "-"]
            for g in hw.gpus]
    if rows:
        lines.append(render_table(["id", "name", "VRAM GiB", "free", "usable", "GB/s", "TFLOPS", "PCIe", "CC"], rows, sty))
    lines.append(f"CPU  {hw.cpu_name} · {hw.cpu_physical_cores} cores / {hw.cpu_logical_cores} threads")
    lines.append(f"RAM  {hw.ram_total_gib:,.1f} GiB total · {hw.ram_available_gib:,.1f} GiB available · ~{hw.ram_bandwidth_gbs:g} GB/s")
    for n in hw.notes:
        lines.append(sty.dim("· " + n))
    _emit(a, {"vram_fit": __version__, "command": "hw", "result": hw.to_dict()}, "\n".join(lines))
    return EXIT_OK


def cmd_presets(a: argparse.Namespace) -> int:
    sty = Style(False if a.no_color else None)
    gs = list_presets(a.vendor)
    rows = [[g.id, g.name, f"{g.vram_gib:g}", f"{g.usable_gib:g}", f"{g.bandwidth_gbs:g}", f"{g.tflops:g}", g.arch]
            for g in gs]
    _emit(a, {"vram_fit": __version__, "command": "presets", "result": [g.to_dict() for g in gs]},
          render_table(["id", "name", "GiB", "usable", "GB/s", "TFLOPS", "arch"], rows, sty))
    return EXIT_OK


def cmd_quants(a: argparse.Namespace) -> int:
    sty = Style(False if a.no_color else None)
    qs = quants_for_engine(normalize_engine(a.engine)) if a.engine else list(QUANTS.values())
    rows = [[q.name, q.family, f"{q.nominal_bpw:.2f}", ", ".join(q.engines), q.description] for q in qs]
    _emit(a, {"vram_fit": __version__, "command": "quants", "result": [q.to_dict() for q in qs]},
          render_table(["quant", "family", "bpw", "engines", "description"], rows, sty))
    return EXIT_OK


def cmd_catalog(a: argparse.Namespace) -> int:
    sty = Style(False if a.no_color else None)
    cat = list_catalog()
    _emit(a, {"vram_fit": __version__, "command": "catalog", "result": cat},
          render_table(["id", "alias"], [[c["id"], c["alias"] or ""] for c in cat], sty))
    return EXIT_OK


def cmd_serve(a: argparse.Namespace) -> int:
    try:
        from .server import run
    except ImportError:
        msg = 'The API server needs extras: pip install "vram-fit[server]"'
        if a.json:
            _emit(a, {"vram_fit": __version__, "error": {"type": "MissingDependency", "message": msg}}, msg)
        else:
            print(msg, file=sys.stderr)
        return EXIT_ERROR
    if a.json:
        _emit(a, {"vram_fit": __version__, "command": "serve",
                  "result": {"url": f"http://{a.host}:{a.port}", "websocket": f"ws://{a.host}:{a.port}/ws"}}, "")
    run(host=a.host, port=a.port, open_browser=a.open)
    return EXIT_OK


def cmd_version(a: argparse.Namespace) -> int:
    _emit(a, {"vram_fit": __version__, "command": "version", "result": {"version": __version__,
              "python": sys.version.split()[0]}}, f"vram-fit {__version__}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vram-fit",
        description="Exact VRAM + tokens/s calculator for local LLM inference and LoRA/QLoRA fine-tuning. "
                    "Reads only config.json / GGUF headers — never downloads weights.",
        epilog="Examples:\n"
               "  vram-fit estimate Qwen/Qwen2.5-14B-Instruct --gpu rtx-4070 -q Q4_K_M -c 32k --kv 8\n"
               "  vram-fit fit bartowski/Qwen2.5-14B-Instruct-GGUF --gpu auto -c 16k\n"
               "  vram-fit fit meta-llama/Llama-3.1-8B --mode qlora --engine unsloth -c 4k --gpu rtx-3060-12gb\n"
               "  vram-fit matrix llama3.1-70b --gpu 2x rtx-3090 --contexts 4k,16k,32k,128k --json\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"vram-fit {__version__}")
    sub = p.add_subparsers(dest="cmd", metavar="COMMAND")

    e = sub.add_parser("estimate", aliases=["est"], help="VRAM breakdown + tok/s for one configuration")
    _add_model_args(e); _add_hw_args(e); _add_settings_args(e); _add_common(e)
    e.add_argument("--command", action="store_true", help="also print the launch command")
    e.set_defaults(func=cmd_estimate)

    f = sub.add_parser("fit", aliases=["optimize", "tune"], help="auto-tune the fastest config that fits + launch command")
    _add_model_args(f); _add_hw_args(f); _add_settings_args(f); _add_common(f)
    f.add_argument("--prefer", choices=PREFERENCES, default="balanced", help="trade-off for the search")
    f.add_argument("--global-batch", type=int, default=16, help="training: target effective batch size")
    f.set_defaults(func=cmd_fit)

    m = sub.add_parser("matrix", help="fit matrix of quantization x context length")
    _add_model_args(m); _add_hw_args(m); _add_settings_args(m); _add_common(m)
    m.add_argument("--contexts", default="4k,8k,16k,32k,64k,128k", help="comma-separated context lengths")
    m.add_argument("--quants", help="comma-separated quant list (default: repo files or a standard ladder)")
    m.set_defaults(func=cmd_matrix)

    i = sub.add_parser("inspect", help="show architecture, parameter counts and GGUF files (metadata only)")
    _add_model_args(i)
    i.add_argument("--quant", "-q", help="GGUF quant to read from a GGUF repo")
    i.add_argument("--json", action="store_true"); i.add_argument("--no-color", action="store_true")
    i.set_defaults(func=cmd_inspect)

    h = sub.add_parser("hw", aliases=["probe"], help="probe local GPUs (NVML / ROCm / Metal), RAM, PCIe")
    h.add_argument("--json", action="store_true"); h.add_argument("--no-color", action="store_true")
    h.set_defaults(func=cmd_hw)

    pr = sub.add_parser("presets", help="list simulated GPU presets")
    pr.add_argument("--vendor", choices=["nvidia", "amd", "apple", "intel"])
    pr.add_argument("--json", action="store_true"); pr.add_argument("--no-color", action="store_true")
    pr.set_defaults(func=cmd_presets)

    q = sub.add_parser("quants", help="list quantization formats and their bits/weight")
    q.add_argument("--engine", help="only formats this engine runs")
    q.add_argument("--json", action="store_true"); q.add_argument("--no-color", action="store_true")
    q.set_defaults(func=cmd_quants)

    c = sub.add_parser("catalog", help="list built-in offline model architectures and aliases")
    c.add_argument("--json", action="store_true"); c.add_argument("--no-color", action="store_true")
    c.set_defaults(func=cmd_catalog)

    s = sub.add_parser("serve", help="start the HTTP + WebSocket API for the dashboard")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=7860)
    s.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_serve)

    v = sub.add_parser("version", help="print version")
    v.add_argument("--json", action="store_true")
    v.set_defaults(func=cmd_version)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    a = parser.parse_args(argv)
    if not getattr(a, "func", None):
        parser.print_help()
        return EXIT_OK
    try:
        return a.func(a)
    except (HubError, GGUFError, KeyError, ValueError, OSError) as exc:
        msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        if getattr(a, "json", False):
            json.dump({"vram_fit": __version__, "command": a.cmd,
                       "error": {"type": type(exc).__name__, "message": msg}}, sys.stdout, indent=2)
            sys.stdout.write("\n")
        else:
            print(f"vram-fit: error: {msg}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
