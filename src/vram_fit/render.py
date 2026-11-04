"""Terminal rendering: multi-segment VRAM bar with a ceiling marker, tables, badges."""

from __future__ import annotations

import os
import re
import shutil
import sys
from typing import Dict, List, Optional

SEGMENT_COLORS = {
    "weights": (124, 92, 255),
    "kv_cache": (0, 209, 255),
    "activations": (255, 176, 32),
    "overhead": (138, 143, 152),
    "lora_optimizer": (255, 79, 216),
}
STATUS_STYLE = {
    "fits": ((48, 209, 88), "FITS"),
    "tight": ((255, 214, 10), "TIGHT"),
    "offload": ((255, 149, 0), "OFFLOAD"),
    "spill": ((255, 59, 48), "SPILL"),
    "oom": ((255, 59, 48), "OOM"),
}
RED = (255, 59, 48)
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
DIM = (120, 125, 135)


class Style:
    def __init__(self, enabled: Optional[bool] = None):
        if enabled is None:
            enabled = sys.stdout.isatty() and not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb"
        self.on = enabled

    def fg(self, rgb, s: str) -> str:
        return f"\x1b[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m{s}\x1b[0m" if self.on else s

    def bg(self, rgb, s: str) -> str:
        return f"\x1b[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m{s}\x1b[0m" if self.on else s

    def bold(self, s: str) -> str:
        return f"\x1b[1m{s}\x1b[0m" if self.on else s

    def dim(self, s: str) -> str:
        return self.fg(DIM, s)


def _gib(mib: float) -> str:
    return f"{mib / 1024:,.2f} GiB"


def _width() -> int:
    return max(40, min(100, shutil.get_terminal_size((100, 24)).columns) - 4)


def vram_bar(result: dict, sty: Style, width: Optional[int] = None) -> List[str]:
    width = width or _width()
    segs = [s for s in result["segments"] if s["mib"] > 0]
    total = result["gpu_total_mib"]
    budget = result["budget_mib"] or 1
    scale_max = max(total, budget) * 1.04 or 1
    cells = []
    acc = 0.0
    for s in segs:
        acc += s["mib"]
        cells.append((s["key"], acc))
    ceiling_col = min(width - 1, int(round(budget / scale_max * width)))
    line = []
    for col in range(width):
        x = (col + 0.5) / width * scale_max
        key = next((k for k, end in cells if x <= end), None)
        if col == ceiling_col:
            line.append(sty.fg(RED, "┃") if sty.on else "|")
            continue
        if key is None:
            line.append(sty.dim("·") if sty.on else ".")
            continue
        over = x > budget
        ch = "▓" if over else "█"
        if not sty.on:
            ch = {"weights": "W", "kv_cache": "K", "activations": "A", "overhead": "O", "lora_optimizer": "L"}[key]
            ch = ch.lower() if over else ch
        line.append(sty.fg(SEGMENT_COLORS[key], ch))
    pad = " " * max(0, ceiling_col - 6)
    label = sty.fg(RED, f"VRAM ceiling {_gib(budget)}")
    return [pad + label, "".join(line)]


def status_badge(result: dict, sty: Style) -> str:
    rgb, text = STATUS_STYLE[result["status"]]
    if sty.on:
        return sty.bg(rgb, sty.bold(f" {text} ")) if result["status"] != "tight" else sty.bg(rgb, f"\x1b[30m {text} ")
    return f"[{text}]"


def render_estimate(result: dict, sty: Style, command: Optional[dict] = None) -> str:
    m, h, st, perf = result["model"], result["hardware"], result["settings"], result["performance"]
    out: List[str] = []
    params = f"{m['params_total'] / 1e9:.2f}B"
    if m["experts"]:
        params += f" ({m['params_active'] / 1e9:.2f}B active, {m['experts']} experts / top-{m['experts_per_token']})"
    out.append(sty.bold(m["id"]) + sty.dim(f"  {m['architecture']} · {params} · {m['layers']} layers · {m['attention']}"
                                           f" · ctx {m['max_context']:,}"))
    hw_line = f"{h['name']} · {h['vram_mib'] / 1024:,.1f} GiB · {h['bandwidth_gbs']:,.0f} GB/s"
    if h.get("pcie"):
        hw_line += f" · PCIe {h['pcie']}"
    hw_line += " · " + ("probed" if h["probed"] else "simulated")
    out.append(sty.dim(hw_line))
    cfg = f"{st['engine']} · {st['mode']} · {st['quant']} · ctx {st['context']:,} · batch {st['batch']}"
    if st["kv_cache_k"]:
        cfg += f" · KV {st['kv_cache_k']}" + (f"/{st['kv_cache_v']}" if st["kv_cache_v"] != st["kv_cache_k"] else "")
    if st["mode"] == "inference" and st["engine"] in ("llama.cpp", "ollama"):
        cfg += f" · ngl {st['gpu_layers']}/{st['max_gpu_layers']}" + (" (auto)" if st["gpu_layers_auto"] else "")
        if st["cpu_moe_layers"]:
            cfg += f" · cpu-moe {st['cpu_moe_layers']}"
    if st.get("lora_rank"):
        cfg += f" · rank {st['lora_rank']}"
    if st.get("optimizer"):
        cfg += f" · {st['optimizer']} · gc {st['grad_checkpointing']}"
    out.append(sty.dim(cfg))
    out.append("")
    if h["vram_mib"] > 0:
        out += vram_bar(result, sty)
    else:
        out.append(sty.dim("No GPU: weights, KV cache and buffers all live in system RAM (CPU inference)."))
    out.append("")
    total = result["gpu_total_mib"] or 1
    for s in result["segments"]:
        if s["mib"] <= 0:
            continue
        swatch = sty.fg(SEGMENT_COLORS[s["key"]], "■")
        out.append(f"  {swatch} {s['label']:<24} {s['mib']:>12,.1f} MiB  {s['mib'] / total * 100:5.1f}%")
    gt = sty.bold("{:>12,.1f}".format(result["gpu_total_mib"]))
    out.append(f"    {'GPU total':<24} {gt} MiB  of {result['budget_mib']:,.0f} MiB "
               f"({(result['utilization'] or 0) * 100:.1f}%)")
    off = result["offload"]
    if off["ram_total_mib"] > 1:
        out.append(sty.fg((255, 149, 0), f"    {'System RAM':<24} {off['ram_total_mib']:>12,.1f} MiB  "
                                          f"(weights {off['cpu_weights_mib']:,.0f} · KV {off['cpu_kv_mib']:,.0f} · "
                                          f"buffers {off['cpu_buffers_mib']:,.0f})"))
    out.append("")
    line = status_badge(result, sty) + "  "
    if "decode_tok_s" in perf:
        line += sty.bold("{:,.1f} tok/s".format(perf["decode_tok_s"])) + " decode"
        if result["settings"]["batch"] > 1:
            line += f" ({perf['decode_tok_s_per_seq']:,.1f}/seq)"
        line += f" · {perf['decode_tok_s_full_ctx']:,.1f} tok/s at full ctx · prefill {perf['prefill_tok_s']:,.0f} tok/s"
        line += sty.dim(f" · bound by {perf['bottleneck'].replace('_', ' ')}")
    elif "train_tok_s" in perf:
        line += sty.bold("{:,.0f} tok/s".format(perf["train_tok_s"])) + f" training (MFU {perf['mfu_assumed']:.0%})"
        if perf.get("epoch_hours"):
            line += f" · {perf['epoch_hours']:,.2f} h/epoch"
    out.append(line)
    lim = result.get("limits") or {}
    if lim.get("max_context_full_gpu") is not None or lim.get("max_batch_full_gpu"):
        bits = []
        if lim.get("max_context_full_gpu"):
            bits.append(f"max context fully on GPU: {lim['max_context_full_gpu']:,}")
        if lim.get("max_batch_full_gpu"):
            bits.append(f"max batch: {lim['max_batch_full_gpu']}")
        out.append(sty.dim("  " + " · ".join(bits)))
    kv = result["details"]["kv_cache"]
    if kv.get("bytes_per_token"):
        out.append(sty.dim(f"  KV cache: {kv['kib_per_token']:,.1f} KiB/token ({kv['full_attention_layers']} full"
                           + (f", {kv['sliding_layers']} sliding-window" if kv["sliding_layers"] else "")
                           + (f", {kv['linear_layers']} linear" if kv["linear_layers"] else "") + " layers)"))
    for w in result["warnings"]:
        out.append(sty.fg((255, 149, 0), "  ! ") + w)
    for n in result["notes"]:
        out.append(sty.dim("  · " + n))
    if command:
        out.append("")
        out += render_command(command, sty)
    return "\n".join(out)


def render_command(cmd: dict, sty: Style) -> List[str]:
    out = [sty.bold(f"Launch ({cmd['engine']}):")]
    for k, v in (cmd.get("files") or {}).items():
        if k in ("Modelfile", "config.yml"):
            continue  # already embedded in the shell snippet as a heredoc
        out.append(sty.dim(f"# {k}"))
        out += ["  " + ln for ln in v.rstrip("\n").splitlines()]
        out.append("")
    for ln in cmd["shell"].splitlines():
        out.append("  " + sty.fg((48, 209, 88), ln))
    if cmd.get("flags"):
        out.append("")
        for f in cmd["flags"]:
            out.append(sty.dim(f"  {f['flag']:<34} {f['why']}"))
    for n in cmd.get("notes") or []:
        out.append(sty.fg((255, 149, 0), "  ! ") + n)
    return out


def render_table(headers: List[str], rows: List[List[str]], sty: Style) -> str:
    def vis(s: str) -> int:
        return len(_ANSI.sub("", s))
    widths = [max(vis(h), *(vis(r[i]) for r in rows)) if rows else vis(h) for i, h in enumerate(headers)]
    lines = ["  ".join(sty.bold(h) + " " * (widths[i] - vis(h)) for i, h in enumerate(headers))]
    lines.append("  ".join("─" * w for w in widths))
    for r in rows:
        lines.append("  ".join(c + " " * (widths[i] - vis(c)) for i, c in enumerate(r)))
    return "\n".join(lines)


def status_cell(status: str, text: str, sty: Style) -> str:
    rgb, _ = STATUS_STYLE[status]
    glyph = {"fits": "●", "tight": "◐", "offload": "◑", "spill": "○", "oom": "✕"}[status]
    return sty.fg(rgb, f"{glyph} {text}")


def segment_legend(sty: Style) -> Dict[str, str]:
    return {k: sty.fg(v, "■") for k, v in SEGMENT_COLORS.items()}
