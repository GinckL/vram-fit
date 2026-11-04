"""Generate the README visuals from live vram-fit output.

Every number drawn in assets/*.svg and the "Will it fit?" cheat sheet comes from the
engine itself, so the README can never drift from what the tool really prints.

    python scripts/render_readme_assets.py            # SVGs + README cheat sheet
    python scripts/render_readme_assets.py --png      # also assets/social-preview.png (needs playwright)

Animations use SMIL (<animate>), which GitHub renders inside <img> tags in every
major browser; no scripts, no external fonts.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ["COLUMNS"] = "100"
os.environ.setdefault("VRAM_FIT_OFFLINE", "1")

from vram_fit import Settings, estimate, load_model, optimize, parse_gpu_spec  # noqa: E402
from vram_fit.render import Style, render_estimate  # noqa: E402

ASSETS = os.path.join(ROOT, "assets")
C = {
    "bg": "#07080B", "bg2": "#0B0D13", "panel": "#0F121A", "panel2": "#151926", "line": "#232838",
    "text": "#E8EAF0", "muted": "#8B90A0", "dim": "#5C6274",
    "weights": "#7C5CFF", "kv_cache": "#00D1FF", "activations": "#FFB020", "overhead": "#8A8F98",
    "lora_optimizer": "#FF4FD8", "green": "#30D158", "yellow": "#FFD60A", "orange": "#FF9500", "red": "#FF3B30",
}
SEG_KEYS = ["weights", "kv_cache", "activations", "overhead", "lora_optimizer"]
SEG_LABEL = {"weights": "Model Weights", "kv_cache": "KV Cache", "activations": "Activation Buffers",
             "overhead": "CUDA/Driver Overhead", "lora_optimizer": "LoRA/Optimizer States"}
MONO = "ui-monospace,SFMono-Regular,'SF Mono',Menlo,Consolas,'Liberation Mono','DejaVu Sans Mono',monospace"
SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,Roboto,'Helvetica Neue',Arial,sans-serif"
SPRING = "0.22 1 0.36 1"  # ease-out with a soft settle


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def write(name: str, svg: str) -> None:
    """Write an SVG after checking it is well-formed XML (GitHub renders README SVGs via <img>, which is strict)."""
    import xml.etree.ElementTree as ET
    ET.fromstring(svg.encode("utf-8"))
    os.makedirs(ASSETS, exist_ok=True)
    with open(os.path.join(ASSETS, name), "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"wrote assets/{name} ({len(svg) / 1024:.1f} KiB)")


def anim(attr: str, values, key_times, dur: float, spline: bool = True, extra: str = "") -> str:
    """Looping SMIL animation with per-interval spline easing."""
    vals = ";".join(str(v) for v in values)
    kts = ";".join(f"{k:.4f}" for k in key_times)
    if spline:
        splines = ";".join([SPRING] * (len(values) - 1))
        mode = f'calcMode="spline" keySplines="{splines}"'
    else:
        mode = 'calcMode="linear"'
    return (f'<animate attributeName="{attr}" values="{vals}" keyTimes="{kts}" dur="{dur}s" '
            f'repeatCount="indefinite" {mode} {extra}/>')


def defs_common() -> str:
    return f"""
  <linearGradient id="brand" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="{C['weights']}"/><stop offset="1" stop-color="{C['kv_cache']}"/>
  </linearGradient>
  <radialGradient id="glowV" cx="0.5" cy="0.5" r="0.5">
    <stop offset="0" stop-color="{C['weights']}" stop-opacity="0.35"/><stop offset="1" stop-color="{C['weights']}" stop-opacity="0"/>
  </radialGradient>
  <radialGradient id="glowC" cx="0.5" cy="0.5" r="0.5">
    <stop offset="0" stop-color="{C['kv_cache']}" stop-opacity="0.22"/><stop offset="1" stop-color="{C['kv_cache']}" stop-opacity="0"/>
  </radialGradient>
  <filter id="neon" x="-50%" y="-50%" width="200%" height="200%">
    <feGaussianBlur stdDeviation="4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse">
    <path d="M32 0H0V32" fill="none" stroke="{C['line']}" stroke-width="0.6" opacity="0.55"/>
  </pattern>
  <pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
    <rect width="8" height="8" fill="{C['red']}" opacity="0.18"/><rect width="3" height="8" fill="{C['red']}" opacity="0.55"/>
  </pattern>"""


# ---------------------------------------------------------------------------
# 1. Hero banner
# ---------------------------------------------------------------------------

def hero() -> str:
    m = load_model("Qwen/Qwen2.5-14B-Instruct", offline=True)
    r = estimate(m, parse_gpu_spec("rtx-4070"), Settings(quant="Q4_K_M", context=16384, kv_cache="q8_0"))
    W, H = 1280, 460
    bx, by, bw, bh = 96, 300, 1088, 46
    budget = r["budget_mib"]
    scale = bw / (budget * 1.0)
    dur = 7.0
    bars = []
    x = bx
    legend = []
    lx = bx
    for i, s in enumerate(r["segments"]):
        if s["mib"] <= 0:
            continue
        w = s["mib"] * scale
        k0, k1 = 0.62 + i * 0.05, 0.62 + i * 0.05 + 0.16
        bars.append(
            f'<rect x="{x:.1f}" y="{by}" height="{bh}" width="{w:.1f}" rx="0" fill="{C[s["key"]]}">'
            + anim("width", [f"{w:.1f}", f"{w:.1f}", 0, 0, f"{w:.1f}", f"{w:.1f}"], [0, 0.52, 0.56, k0, k1, 1], dur)
            + "</rect>")
        x += w
        legend.append(
            f'<g transform="translate({lx},{by + bh + 34})"><rect width="12" height="12" rx="3" fill="{C[s["key"]]}"/>'
            f'<text x="20" y="11" font-family="{SANS}" font-size="15" fill="{C["muted"]}">{esc(s["label"])} '
            f'<tspan fill="{C["text"]}" font-family="{MONO}">{s["mib"] / 1024:.2f} GiB</tspan></text></g>')
        lx += 250
    used = r["gpu_total_mib"]
    perf = r["performance"]
    stats = [("0.26 ms", "per estimate"), ("±0.2 %", "vs real GGUF files"), ("0 GB", "of weights fetched"),
             ("105", "GPU presets")]
    chips = []
    cx = 684
    for v, lab in stats:
        chips.append(
            f'<g transform="translate({cx},96)"><rect width="134" height="76" rx="14" fill="{C["panel"]}" '
            f'stroke="{C["line"]}"/><text x="16" y="36" font-family="{MONO}" font-size="22" font-weight="700" '
            f'fill="{C["text"]}">{esc(v)}</text><text x="16" y="58" font-family="{SANS}" font-size="12.5" '
            f'fill="{C["muted"]}">{esc(lab)}</text></g>')
        cx += 144
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="vram-fit: exact VRAM for local LLMs before you download a single gigabyte">
<title>vram-fit — exact VRAM for local LLMs</title>
<defs>{defs_common()}
  <clipPath id="card"><rect width="{W}" height="{H}" rx="24"/></clipPath>
</defs>
<g clip-path="url(#card)">
  <rect width="{W}" height="{H}" fill="{C['bg']}"/>
  <rect width="{W}" height="{H}" fill="url(#grid)"/>
  <ellipse cx="220" cy="80" rx="420" ry="260" fill="url(#glowV)"/>
  <ellipse cx="1140" cy="420" rx="460" ry="260" fill="url(#glowC)"/>
  <text x="96" y="92" font-family="{MONO}" font-size="14" letter-spacing="3" fill="{C['green']}">● OPEN SOURCE · ZERO DEPENDENCIES · --json EVERYWHERE</text>
  <text x="92" y="168" font-family="{SANS}" font-size="78" font-weight="800" letter-spacing="-2" fill="url(#brand)">vram-fit</text>
  <text x="96" y="214" font-family="{SANS}" font-size="25" font-weight="600" fill="{C['text']}">Know your VRAM before you download a single gigabyte.</text>
  <text x="96" y="246" font-family="{SANS}" font-size="16" fill="{C['muted']}">Weights · KV cache · activations · CUDA overhead · LoRA/QLoRA states — from config.json / GGUF headers alone.</text>
  {''.join(chips)}
  <rect x="{bx - 2}" y="{by - 2}" width="{bw + 4}" height="{bh + 4}" rx="12" fill="{C['panel2']}" stroke="{C['line']}"/>
  <g clip-path="url(#barclip)">{''.join(bars)}</g>
  <clipPath id="barclip"><rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="10"/></clipPath>
  <g filter="url(#neon)">
    <rect x="{bx + bw - 1}" y="{by - 14}" width="3" height="{bh + 28}" rx="1.5" fill="{C['red']}">
      {anim('opacity', [1, 0.45, 1], [0, 0.5, 1], 1.6, spline=False)}
    </rect>
  </g>
  <text x="{bx + bw}" y="{by - 22}" text-anchor="end" font-family="{MONO}" font-size="13" fill="{C['red']}">VRAM ceiling · RTX 4070 · {budget / 1024:.0f} GiB</text>
  <text x="{bx}" y="{by - 22}" font-family="{MONO}" font-size="13" fill="{C['muted']}">Qwen2.5-14B · Q4_K_M · 16k ctx · KV q8_0 → <tspan fill="{C['green']}">{used / 1024:.2f} GiB · FITS · {perf['decode_tok_s']:.1f} tok/s</tspan></text>
  {''.join(legend)}
</g>
</svg>
"""


# ---------------------------------------------------------------------------
# 2. Terminal demo (real ANSI output converted to SVG)
# ---------------------------------------------------------------------------

_SGR = re.compile(r"\x1b\[([0-9;]*)m")


def ansi_runs(line: str):
    """Split an ANSI line into (text, fg, bg, bold) runs."""
    runs = []
    fg = bg = None
    bold = False
    pos = 0
    for m in _SGR.finditer(line):
        if m.start() > pos:
            runs.append((line[pos:m.start()], fg, bg, bold))
        codes = [c for c in m.group(1).split(";") if c != ""] or ["0"]
        i = 0
        while i < len(codes):
            c = codes[i]
            if c == "0":
                fg = bg = None
                bold = False
            elif c == "1":
                bold = True
            elif c == "30":
                fg = "#000000"
            elif c in ("38", "48") and i + 4 < len(codes) and codes[i + 1] == "2":
                rgb = "#%02X%02X%02X" % tuple(int(x) for x in codes[i + 2:i + 5])
                if c == "38":
                    fg = rgb
                else:
                    bg = rgb
                i += 4
            i += 1
        pos = m.end()
    if pos < len(line):
        runs.append((line[pos:], fg, bg, bold))
    return runs


def demo() -> str:
    m = load_model("bartowski/Qwen2.5-14B-Instruct-GGUF", offline=True)
    out = optimize(m, parse_gpu_spec("rtx-4070"), Settings(context=16384))
    best = out["best"]
    best["notes"] = []
    text = render_estimate(best, Style(True), out["command"])
    raw = text.splitlines()
    lines = []
    for ln in raw:
        plain = _SGR.sub("", ln)
        if plain.strip().startswith(("-c ", "-ngl ", "-fa ", "-ctk ", "-ub ", "-t ")) or plain.startswith("  -"):
            continue  # flag explanations: keep the demo short
        lines.append(ln)
    while lines and not _SGR.sub("", lines[-1]).strip():
        lines.pop()
    cmd = "vram-fit fit bartowski/Qwen2.5-14B-Instruct-GGUF --gpu rtx-4070 -c 16k"
    fs, lh, cw = 13, 19, 7.83
    pad_x, top = 22, 58
    width = 940
    max_cols = int((width - 2 * pad_x) / cw)
    wrapped = []
    for ln in lines:
        plain = _SGR.sub("", ln)
        if len(plain) > max_cols and "\x1b[" in ln and plain.strip().startswith("llama-server"):
            words = plain.strip().split(" ")
            cur = "  "
            for w in words:
                if len(cur) + len(w) + 3 > max_cols:
                    wrapped.append(("cmd", cur + " \\"))
                    cur = "      "
                cur += w + " "
            wrapped.append(("cmd", cur.rstrip()))
        else:
            wrapped.append(("ansi", ln))
    n = len(wrapped) + 2
    height = top + n * lh + 26
    dur = 18.0
    hide = 0.40          # 0 .. 0.40: full output on screen (also the first frame)
    type_start, type_end = 0.43, 0.53
    out_start = 0.55
    step = (0.80 - out_start) / max(1, len(wrapped))
    body = []
    y = top + lh
    # prompt + typing
    body.append(f'<text x="{pad_x}" y="{y}" font-family="{MONO}" font-size="{fs}" fill="{C["green"]}">❯</text>')
    full_w = len(cmd) * cw
    body.append(f'<clipPath id="typeclip"><rect x="{pad_x + 2 * cw}" y="{y - 14}" height="20" width="{full_w:.1f}">'
                + anim("width", [full_w, full_w, 0, 0, full_w, full_w], [0, hide, hide + 0.01, type_start, type_end, 1],
                       dur, spline=False)
                + "</rect></clipPath>")
    body.append(f'<text x="{pad_x + 2 * cw}" y="{y}" font-family="{MONO}" font-size="{fs}" fill="{C["text"]}" '
                f'clip-path="url(#typeclip)" xml:space="preserve">{esc(cmd)}</text>')
    x0c, x1c = pad_x + 2 * cw, pad_x + (2 + len(cmd)) * cw
    body.append(f'<rect x="{x1c:.1f}" y="{y - 12}" width="{cw:.1f}" height="16" fill="{C["text"]}" opacity="0">'
                + anim("x", [x1c, x1c, x0c, x0c, x1c, x1c], [0, hide, hide + 0.01, type_start, type_end, 1], dur, spline=False)
                + anim("opacity", [0, 0, 0.85, 0.85, 0, 0], [0, hide, hide + 0.01, type_end, type_end + 0.01, 1], dur,
                       spline=False)
                + "</rect>")
    y += lh * 2
    for idx, (kind, ln) in enumerate(wrapped):
        k = out_start + idx * step
        vis = anim("opacity", [1, 1, 0, 0, 1, 1], [0, hide, hide + 0.02, k, min(k + 0.012, 0.95), 1], dur, spline=False)
        plain = _SGR.sub("", ln)
        if kind == "ansi" and ("█" in plain or "·····" in plain or "▓" in plain):
            # the gauge: draw real rects instead of block glyphs
            budget = best["budget_mib"]
            total = best["gpu_total_mib"]
            scale_max = max(total, budget) * 1.04
            gw = (len(plain)) * cw
            gx = pad_x
            parts = []
            acc = gx
            for s in best["segments"]:
                if s["mib"] <= 0:
                    continue
                w = s["mib"] / scale_max * gw
                parts.append(f'<rect x="{acc:.1f}" y="{y - 12}" height="15" width="{w:.1f}" fill="{C[s["key"]]}">'
                             + anim("width", [f"{w:.1f}", f"{w:.1f}", 0, 0, f"{w:.1f}", f"{w:.1f}"],
                                    [0, hide, hide + 0.02, k, min(k + 0.07, 0.95), 1], dur)
                             + "</rect>")
                acc += w
            cxp = gx + budget / scale_max * gw
            body.append(f'<g>{"".join(parts)}<rect x="{gx}" y="{y - 12}" width="{gw:.1f}" height="15" rx="3" fill="none" '
                        f'stroke="{C["line"]}"/><rect x="{cxp:.1f}" y="{y - 17}" width="2.5" height="25" fill="{C["red"]}" '
                        f'filter="url(#neon)"><animate attributeName="opacity" values="1;1;0;0;1;1" '
                        f'keyTimes="0;{hide:.4f};{hide + 0.02:.4f};{k:.4f};{min(k + 0.012, 0.95):.4f};1" dur="{dur}s" '
                        f'repeatCount="indefinite"/></rect></g>')
        elif kind == "cmd":
            body.append(f'<text x="{pad_x}" y="{y}" font-family="{MONO}" font-size="{fs}" fill="{C["green"]}" '
                        f'xml:space="preserve">{esc(ln)}{vis}</text>')
        else:
            spans = []
            for txt, fg, bg, bold in ansi_runs(ln):
                if bg:  # status badge: pill behind the text
                    spans.append(("BG", txt, bg))
                    continue
                attrs = f'fill="{fg or C["text"]}"' + (' font-weight="700"' if bold else "")
                spans.append(("T", txt, attrs))
            xcur = pad_x
            parts = []
            for kind2, txt, extra in spans:
                if kind2 == "BG":
                    wpx = len(txt) * cw
                    parts.append(f'<rect x="{xcur:.1f}" y="{y - 13}" width="{wpx:.1f}" height="18" rx="4" fill="{extra}"/>'
                                 f'<text x="{xcur:.1f}" y="{y}" font-family="{MONO}" font-size="{fs}" font-weight="700" '
                                 f'fill="#07080B" xml:space="preserve">{esc(txt)}</text>')
                else:
                    parts.append(f'<text x="{xcur:.1f}" y="{y}" font-family="{MONO}" font-size="{fs}" '
                                 f'{extra} xml:space="preserve">{esc(txt)}</text>')
                xcur += len(txt) * cw
            body.append(f'<g>{"".join(parts)}{vis}</g>')
        y += lh
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Animated terminal: vram-fit fit picks Q4_K_M with a q8_0 KV cache for Qwen2.5-14B on an RTX 4070 and prints the llama-server command">
<title>vram-fit fit — terminal demo</title>
<defs>{defs_common()}</defs>
<rect width="{width}" height="{height}" rx="16" fill="{C['bg2']}" stroke="{C['line']}"/>
<rect width="{width}" height="40" rx="16" fill="{C['panel']}"/><rect y="24" width="{width}" height="16" fill="{C['panel']}"/>
<line x1="0" y1="40" x2="{width}" y2="40" stroke="{C['line']}"/>
<circle cx="24" cy="20" r="6" fill="#FF5F57"/><circle cx="44" cy="20" r="6" fill="#FEBC2E"/><circle cx="64" cy="20" r="6" fill="#28C840"/>
<text x="{width / 2}" y="25" text-anchor="middle" font-family="{SANS}" font-size="13" fill="{C['muted']}">vram-fit — real output</text>
{''.join(body)}
</svg>
"""


# ---------------------------------------------------------------------------
# 3. The story: context grows, KV cache blows past the ceiling, vram-fit fixes it
# ---------------------------------------------------------------------------

def story() -> str:
    m = load_model("llama3.1-8b", offline=True)
    g = parse_gpu_spec("rtx-4060")
    states = [
        ("8k context", Settings(context=8192, gpu_layers=33), "f16"),
        ("32k context", Settings(context=32768, gpu_layers=33), "f16"),
        ("32k context + q8_0 KV", Settings(context=32768, kv_cache="q8_0", gpu_layers=33), "q8_0"),
    ]
    res = [estimate(m, g, s, limits=False) for _, s, _ in states]
    W, H = 1100, 380
    bx, by, bw, bh = 60, 196, 980, 56
    budget = res[0]["budget_mib"]
    scale_max = max(r["gpu_total_mib"] for r in res) * 1.08
    sx = bw / scale_max
    dur = 12.0
    # keyframes: hold each state, spring between them
    kt = [0, 0.06, 0.30, 0.38, 0.62, 0.70, 0.94, 1.0]
    order = [0, 0, 0, 1, 1, 2, 2, 0]

    def series(fn):
        return [fn(res[i]) for i in order]

    segs = []
    for key in SEG_KEYS:
        def xw(r, key=key):
            acc = 0.0
            for s in r["segments"]:
                if s["key"] == key:
                    return acc, s["mib"]
                acc += s["mib"]
            return acc, 0.0
        xs = series(lambda r: f"{bx + xw(r)[0] * sx:.1f}")
        ws = series(lambda r: f"{xw(r)[1] * sx:.1f}")
        if all(float(w) == 0 for w in ws):
            continue
        segs.append(f'<rect y="{by}" height="{bh}" fill="{C[key]}" x="{xs[0]}" width="{ws[0]}">'
                    + anim("x", xs, kt, dur) + anim("width", ws, kt, dur) + "</rect>")
    ceil_x = bx + budget * sx
    over_w = series(lambda r: f"{max(0.0, r['gpu_total_mib'] - budget) * sx:.1f}")
    over = (f'<rect x="{ceil_x:.1f}" y="{by - 6}" height="{bh + 12}" width="0" fill="url(#hatch)" rx="4">'
            + anim("width", over_w, kt, dur)
            + anim("opacity", [1, 0.55, 1], [0, 0.5, 1], 0.9, spline=False) + "</rect>")
    # captions per state
    caps = []
    for i, (title, st, kv) in enumerate(states):
        r = res[i]
        status = r["status"].upper()
        col = {"FITS": C["green"], "TIGHT": C["yellow"], "SPILL": C["red"], "OFFLOAD": C["orange"], "OOM": C["red"]}[status]
        tok = r["performance"]["decode_tok_s"]
        detail = (f"spills {r['offload']['spill_mib']:,.0f} MiB over PCIe → {tok:.1f} tok/s" if r["status"] == "spill"
                  else f"{r['gpu_total_mib'] / 1024:.2f} GiB of {budget / 1024:.0f} GiB → {tok:.1f} tok/s")
        on = [1.0 if order[j] == i else 0.0 for j in range(len(order))]
        op = ";".join(str(v) for v in on)
        caps.append(
            f'<g opacity="{on[0]}"><animate attributeName="opacity" values="{op}" keyTimes="{";".join(f"{k:.4f}" for k in kt)}" '
            f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'<text x="{bx}" y="132" font-family="{SANS}" font-size="30" font-weight="700" fill="{C["text"]}">{esc(title)}</text>'
            f'<rect x="{bx + len(title) * 16 + 20:.0f}" y="106" width="{len(status) * 12 + 26}" height="32" rx="16" fill="{col}"/>'
            f'<text x="{bx + len(title) * 16 + 33:.0f}" y="128" font-family="{MONO}" font-size="16" font-weight="800" fill="#07080B">{status}</text>'
            f'<text x="{bx}" y="166" font-family="{MONO}" font-size="16" fill="{col if r["status"] == "spill" else C["muted"]}">{esc(detail)}</text>'
            + (f'<text x="{W - 60}" y="132" text-anchor="end" font-family="{MONO}" font-size="15" fill="{C["green"]}">'
               f'vram-fit fix: -ctk q8_0 -ctv q8_0 -fa on</text>' if i == 2 else "")
            + "</g>")
    pulse = anim("opacity", [0, 0, 0, 1, 0.3, 0, 0, 0], kt, dur, spline=False)
    # context slider teaser
    sl_x, sl_y, sl_w = bx, 318, 420
    knob = series(lambda r: f"{sl_x + (r['settings']['context'] - 2048) / (131072 - 2048) * sl_w * 4:.1f}")
    knob = [f"{min(float(v), sl_x + sl_w):.1f}" for v in knob]
    kv_chip = []
    for i, label in enumerate(["KV f16", "KV q8_0"]):
        on = [1.0 if (states[order[j]][2] == ("f16" if i == 0 else "q8_0")) else 0.0 for j in range(len(order))]
        kv_chip.append(
            f'<g opacity="{on[0]}"><animate attributeName="opacity" values="{";".join(str(v) for v in on)}" '
            f'keyTimes="{";".join(f"{k:.4f}" for k in kt)}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'<rect x="{sl_x + sl_w + 40}" y="{sl_y - 15}" width="92" height="30" rx="15" fill="{C["panel2"]}" '
            f'stroke="{C["kv_cache"] if i else C["line"]}"/>'
            f'<text x="{sl_x + sl_w + 86}" y="{sl_y + 5}" text-anchor="middle" font-family="{MONO}" font-size="13" '
            f'fill="{C["kv_cache"] if i else C["muted"]}">{label}</text></g>')
    legend = []
    lx = sl_x + sl_w + 160
    for key in ["weights", "kv_cache", "activations", "overhead"]:
        legend.append(f'<rect x="{lx}" y="{sl_y - 6}" width="11" height="11" rx="3" fill="{C[key]}"/>'
                      f'<text x="{lx + 17}" y="{sl_y + 4}" font-family="{SANS}" font-size="13" fill="{C["muted"]}">'
                      f'{esc(SEG_LABEL[key].split("/")[0])}</text>')
        lx += {"weights": 118, "kv_cache": 92, "activations": 145, "overhead": 0}[key]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Animation: Llama-3.1-8B on an 8 GB RTX 4060. 8k context fits; at 32k the KV cache spills past the VRAM ceiling; with a q8_0 KV cache it fits again">
<title>Why your model OOMs at long context — and the fix</title>
<defs>{defs_common()}<clipPath id="c"><rect width="{W}" height="{H}" rx="22"/></clipPath></defs>
<g clip-path="url(#c)">
<rect width="{W}" height="{H}" fill="{C['bg']}"/><rect width="{W}" height="{H}" fill="url(#grid)" opacity="0.6"/>
<ellipse cx="{W - 120}" cy="40" rx="380" ry="200" fill="url(#glowC)"/>
<text x="{bx}" y="64" font-family="{MONO}" font-size="14" letter-spacing="2" fill="{C['muted']}">LLAMA-3.1-8B · Q4_K_M · RTX 4060 8 GB · ALL 33 LAYERS ON GPU</text>
{''.join(caps)}
<rect x="{bx - 3}" y="{by - 3}" width="{bw + 6}" height="{bh + 6}" rx="12" fill="{C['panel2']}" stroke="{C['line']}"/>
{''.join(segs)}
{over}
<rect x="{ceil_x - 1.5:.1f}" y="{by - 18}" width="3" height="{bh + 36}" rx="1.5" fill="{C['red']}" opacity="0" filter="url(#neon)" stroke="{C['red']}" stroke-width="6" stroke-opacity="0.35">{pulse}</rect>
<rect x="{ceil_x - 1.5:.1f}" y="{by - 18}" width="3" height="{bh + 36}" rx="1.5" fill="{C['red']}" filter="url(#neon)"/>
<text x="{ceil_x:.1f}" y="{by - 26}" text-anchor="middle" font-family="{MONO}" font-size="12.5" fill="{C['red']}">VRAM CEILING {budget / 1024:.0f} GiB</text>
<text x="{sl_x}" y="{sl_y - 24}" font-family="{SANS}" font-size="13" fill="{C['muted']}">Context length</text>
<rect x="{sl_x}" y="{sl_y - 3}" width="{sl_w}" height="6" rx="3" fill="{C['panel2']}"/>
<rect x="{sl_x}" y="{sl_y - 3}" height="6" rx="3" fill="url(#brand)" width="{float(knob[0]) - sl_x:.1f}">{anim('width', [f"{float(k) - sl_x:.1f}" for k in knob], kt, dur)}</rect>
<circle cy="{sl_y}" r="10" fill="{C['text']}" stroke="{C['weights']}" stroke-width="3" cx="{knob[0]}">{anim('cx', knob, kt, dur)}</circle>
<text x="{sl_x}" y="{sl_y + 28}" font-family="{MONO}" font-size="11.5" fill="{C['dim']}">2k</text>
<text x="{sl_x + sl_w}" y="{sl_y + 28}" text-anchor="end" font-family="{MONO}" font-size="11.5" fill="{C['dim']}">32k+</text>
{''.join(kv_chip)}
{''.join(legend)}
</g>
</svg>
"""


# ---------------------------------------------------------------------------
# 4. Fit-matrix heatmap
# ---------------------------------------------------------------------------

def matrix() -> str:
    m = load_model("Qwen/Qwen2.5-14B-Instruct", offline=True)
    g = parse_gpu_spec("rtx-4070")
    quants = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "IQ4_XS", "Q3_K_M", "Q2_K"]
    ctxs = [4096, 16384, 32768, 131072]
    cell_w, cell_h = 196, 50
    x0, y0 = 150, 118
    W = x0 + cell_w * len(ctxs) + 40
    H = y0 + cell_h * len(quants) + 78
    col = {"fits": C["green"], "tight": C["yellow"], "offload": C["orange"], "spill": C["red"], "oom": C["red"]}
    pick = optimize(m, g, Settings(context=16384))["best"]["settings"]
    cells = []
    for i, q in enumerate(quants):
        cells.append(f'<text x="{x0 - 18}" y="{y0 + i * cell_h + 31}" text-anchor="end" font-family="{MONO}" '
                     f'font-size="15" fill="{C["text"]}">{q}</text>')
        for j, c in enumerate(ctxs):
            r = estimate(m, g, Settings(quant=q, context=c, kv_cache="q8_0"), limits=False)
            st = r["status"]
            tok = r["performance"]["decode_tok_s"]
            x, y = x0 + j * cell_w, y0 + i * cell_h
            if q == pick["quant"] and c == pick["context"]:
                cells.append(
                    f'<rect x="{x + 1}" y="{y + 1}" width="{cell_w - 2}" height="{cell_h - 2}" rx="12" fill="none" '
                    f'stroke="{C["kv_cache"]}" stroke-width="2.5" filter="url(#neon)">'
                    + anim("stroke-opacity", [1, 0.25, 1], [0, 0.5, 1], 1.8, spline=False) + "</rect>")
            cells.append(
                f'<g><rect x="{x + 4}" y="{y + 4}" width="{cell_w - 8}" height="{cell_h - 8}" rx="10" fill="{col[st]}" '
                f'fill-opacity="{0.22 if st in ("fits", "tight") else 0.14}" stroke="{col[st]}" stroke-opacity="0.55"/>'
                f'<circle cx="{x + 22}" cy="{y + 25}" r="5" fill="{col[st]}"/>'
                f'<text x="{x + 36}" y="{y + 30}" font-family="{MONO}" font-size="14" fill="{C["text"]}">'
                f'{r["gpu_total_mib"] / 1024:4.1f} GiB</text>'
                f'<text x="{x + cell_w - 16}" y="{y + 30}" text-anchor="end" font-family="{MONO}" font-size="14" '
                f'font-weight="700" fill="{col[st]}">{tok:.0f} t/s</text></g>')
    heads = "".join(f'<text x="{x0 + j * cell_w + cell_w / 2}" y="{y0 - 14}" text-anchor="middle" font-family="{MONO}" '
                    f'font-size="15" fill="{C["muted"]}">{c // 1024}k ctx</text>' for j, c in enumerate(ctxs))
    leg = []
    lx = x0
    for st, lab in [("fits", "fits in VRAM"), ("tight", "> 92 % of VRAM"), ("offload", "layers in system RAM")]:
        leg.append(f'<circle cx="{lx + 6}" cy="{H - 34}" r="6" fill="{col[st]}"/><text x="{lx + 18}" y="{H - 29}" '
                   f'font-family="{SANS}" font-size="14" fill="{C["muted"]}">{lab}</text>')
        lx += 190
    leg.append(f'<rect x="{lx}" y="{H - 44}" width="22" height="20" rx="6" fill="none" stroke="{C["kv_cache"]}" '
               f'stroke-width="2"/><text x="{lx + 30}" y="{H - 29}" font-family="{SANS}" font-size="14" '
               f'fill="{C["kv_cache"]}">vram-fit fit pick @ 16k</text>')
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Fit matrix for Qwen2.5-14B on an RTX 4070 12 GB: VRAM and tokens per second for each quantization and context length">
<title>vram-fit matrix — Qwen2.5-14B on RTX 4070</title>
<defs>{defs_common()}<clipPath id="c"><rect width="{W}" height="{H}" rx="22"/></clipPath></defs>
<g clip-path="url(#c)">
<rect width="{W}" height="{H}" fill="{C['bg']}"/><rect width="{W}" height="{H}" fill="url(#grid)" opacity="0.5"/>
<text x="40" y="52" font-family="{SANS}" font-size="24" font-weight="700" fill="{C['text']}">Which quant fits? <tspan fill="url(#brand)">vram-fit matrix</tspan></text>
<text x="40" y="78" font-family="{MONO}" font-size="13" fill="{C['muted']}">Qwen2.5-14B · RTX 4070 12 GB · KV q8_0 · llama.cpp · GPU GiB and decode tok/s</text>
{heads}
{''.join(cells)}
{''.join(leg)}
</g>
</svg>
"""


# ---------------------------------------------------------------------------
# 5. "Will it fit?" cheat sheet (Markdown, injected into README between markers)
# ---------------------------------------------------------------------------

CHEAT_GPUS = [("rtx-3060-12gb", "RTX 3060<br>12 GB"), ("rtx-4060-ti-16gb", "RTX 4060 Ti<br>16 GB"),
              ("rtx-4090", "RTX 4090<br>24 GB"), ("rtx-5090", "RTX 5090<br>32 GB"),
              ("2x rtx-3090", "2× RTX 3090<br>48 GB"), ("m4-max-64gb", "Mac M4 Max<br>64 GB")]
CHEAT_MODELS = [("meta-llama/Llama-3.1-8B-Instruct", "Llama 3.1 8B"), ("Qwen/Qwen3-14B", "Qwen3 14B"),
                ("Qwen/Qwen3-30B-A3B", "Qwen3 30B-A3B (MoE)"), ("google/gemma-3-27b-it", "Gemma 3 27B"),
                ("Qwen/Qwen3-32B", "Qwen3 32B"), ("meta-llama/Llama-3.3-70B-Instruct", "Llama 3.3 70B")]


def cheatsheet(ctx: int = 16384) -> str:
    icon = {"fits": "🟢", "tight": "🟡", "offload": "🟠", "spill": "🔴", "oom": "🔴"}
    head = "| Model ↓ / GPU → | " + " | ".join(lab for _, lab in CHEAT_GPUS) + " |"
    sep = "|---|" + "|".join(":-:" for _ in CHEAT_GPUS) + "|"
    rows = [head, sep]
    for mid, mlab in CHEAT_MODELS:
        m = load_model(mid, offline=True)
        cells = []
        for gid, _ in CHEAT_GPUS:
            g = parse_gpu_spec(gid)
            o = optimize(m, g, Settings(context=ctx, ram_total_gib=64))
            b = o["best"]
            st = b["status"]
            if st in ("spill", "oom"):
                cells.append("🔴 won't fit")
                continue
            q = b["settings"]["quant"]
            tok = b["performance"]["decode_tok_s"]
            extra = ""
            if b["settings"].get("cpu_moe_layers"):
                extra = "<br><sub>experts in RAM</sub>"
            elif st == "offload":
                extra = f"<br><sub>{b['offload']['layers_on_gpu']}/{b['offload']['layers_total']} layers on GPU</sub>"
            slow = " 🐢" if tok < 5 else ""
            cells.append(f"{icon[st]} **{q}**<br>{tok:.0f} tok/s{slow}{extra}")
        rows.append(f"| **{mlab}** | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def inject(readme: str, name: str, content: str) -> str:
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    if start not in readme:
        return readme
    a = readme.index(start) + len(start)
    b = readme.index(end)
    return readme[:a] + "\n" + content + "\n" + readme[b:]


# ---------------------------------------------------------------------------
# 6. Social preview (1280x640 PNG for GitHub Settings -> Social preview)
# ---------------------------------------------------------------------------

def social_png() -> None:
    """Render assets/social-preview.png (1280x640) with Playwright (Python package or Node module)."""
    import json
    import subprocess
    import tempfile
    hero_svg = open(os.path.join(ASSETS, "hero.svg"), encoding="utf-8").read()
    page_html = (f'<html><body style="margin:0;background:{C["bg"]};width:1280px;height:640px;display:flex;'
                 f'align-items:center;justify-content:center">{hero_svg}</body></html>')
    out = os.path.join(ASSETS, "social-preview.png")
    freeze = "(() => { const s = document.querySelector('svg'); s.pauseAnimations(); s.setCurrentTime(0); })()"
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={"width": 1280, "height": 640})
            pg.set_content(page_html)
            pg.evaluate(freeze)
            pg.screenshot(path=out)
            b.close()
    except ImportError:
        with tempfile.TemporaryDirectory() as td:
            html_path = os.path.join(td, "social.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(page_html)
            js = ("const { chromium } = require('playwright');(async () => {const b = await chromium.launch();"
                  "const pg = await b.newPage({viewport: {width: 1280, height: 640}});"
                  f"await pg.goto('file://' + {json.dumps(html_path)});await pg.evaluate({json.dumps(freeze)});"
                  f"await pg.screenshot({{path: {json.dumps(out)}}});await b.close();}})();")
            env = dict(os.environ)
            root = subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip()
            env["NODE_PATH"] = root
            r = subprocess.run(["node", "-e", js], env=env, capture_output=True, text=True)
            if r.returncode != 0:
                print("could not render social-preview.png (install playwright):", r.stderr.strip()[:300])
                return
    print("wrote assets/social-preview.png")


def swatches() -> None:
    for key in SEG_KEYS:
        write(f"swatch-{key}.svg", f'<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 14 14">'
                                   f'<rect width="14" height="14" rx="4" fill="{C[key]}"/></svg>\n')
    write("swatch-ceiling.svg", f'<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 14 14">'
                                f'<rect x="5" width="4" height="14" rx="2" fill="{C["red"]}"/></svg>\n')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", action="store_true", help="also render assets/social-preview.png with playwright")
    a = ap.parse_args()
    write("hero.svg", hero())
    write("demo.svg", demo())
    write("story.svg", story())
    write("matrix.svg", matrix())
    swatches()
    readme_path = os.path.join(ROOT, "README.md")
    with open(readme_path, encoding="utf-8") as f:
        readme = f.read()
    new = inject(readme, "cheatsheet", cheatsheet())
    if new != readme:
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(new)
        print("updated README cheat sheet")
    if a.png:
        social_png()


if __name__ == "__main__":
    main()
