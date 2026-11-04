"""Auto-tuner: search quant x KV-cache x offload (x micro-batch for training) for the
configuration with the best quality/speed trade-off that does not OOM, then emit
the launch command for it."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import List, Optional, Tuple

from .commands import build_command
from .estimator import GGUF_ENGINES, TRAIN_ENGINES, Settings, estimate, normalize_engine
from .hardware import HardwareProfile
from .model import LoadedModel
from .presets import GPU
from .quant import get_quant

GGUF_LADDER = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "IQ4_XS", "Q3_K_M", "IQ3_M", "IQ3_XXS", "Q2_K", "IQ2_M", "IQ2_XXS"]
EXL2_LADDER = ["EXL2-8", "EXL2-6", "EXL2-5", "EXL2-4.65", "EXL2-4.25", "EXL2-4", "EXL2-3.5", "EXL2-3", "EXL2-2.5"]
PREFERENCES = ("balanced", "speed", "quality")
_KV_QUALITY = {"f16": 1.0, "auto": 1.0, "FP16": 1.0, "bf16": 1.0, "q8_0": 0.995, "fp8": 0.99, "Q8": 0.995,
               "Q6": 0.99, "Q4": 0.98, "q5_0": 0.985, "q4_0": 0.95}
_TIER = {"fits": 3, "tight": 3, "offload": 2, "spill": 0, "oom": -1}


def quant_quality(bpw: float) -> float:
    """Saturating proxy for output quality vs bits/weight (perplexity-gap shaped)."""
    return max(0.0, 1.0 - math.exp(-(max(bpw, 1.0) - 2.0) / 1.1))


def _utility(r: dict, prefer: str) -> float:
    st = r["settings"]
    bpw = st["quant_bpw"]
    tier = _TIER[r["status"]]
    q = quant_quality(bpw) * _KV_QUALITY.get(st["kv_cache_k"] or "f16", 0.97)
    tok = r["performance"].get("decode_tok_s") or r["performance"].get("train_tok_s") or 0.0
    speed = math.log2(1.0 + tok)
    if prefer == "speed":
        if bpw < 2.5:
            tier -= 1
        return tier * 1000 + speed * 20 + q * 10
    if prefer == "quality":
        return (1000 if tier >= 2 else tier * 1000) + q * 1000 + speed
    if bpw < 3.4:
        tier -= 1
    # fully on GPU: quality first (speed is already good); offloaded: speed matters more.
    # "tight" (>92% of VRAM) leaves no room for the desktop / fragmentation, so it pays a penalty.
    if tier >= 3:
        return tier * 1000 + q * 80 + speed * 6 - (15 if r["status"] == "tight" else 0)
    return tier * 1000 + q * 60 + speed * 12


def _summary(r: dict) -> dict:
    st = r["settings"]
    perf = r["performance"]
    return {"quant": st["quant"], "kv_cache": st["kv_cache_k"], "gpu_layers": st["gpu_layers"],
            "cpu_moe_layers": st["cpu_moe_layers"], "batch": st["batch"], "status": r["status"],
            "gpu_total_mib": r["gpu_total_mib"], "decode_tok_s": perf.get("decode_tok_s"),
            "train_tok_s": perf.get("train_tok_s"), "context": st["context"], "ubatch": st["ubatch"]}


def _gguf_candidates(model: LoadedModel) -> List[str]:
    avail = [f["quant"] for f in model.gguf_files if f.get("quant") and not f.get("mmproj")]
    if avail:
        seen = []
        for q in avail:
            try:
                get_quant(q)
            except KeyError:
                continue
            if q not in seen:
                seen.append(q)
        return seen
    out = list(GGUF_LADDER)
    if model.params_total < 4e9:
        out.insert(0, "BF16")
    return out


def _moe_cpu_search(model: LoadedModel, gpu: GPU, st: Settings, hw: Optional[HardwareProfile]) -> Optional[dict]:
    """All layers on GPU, push expert tensors of the first N layers to RAM (llama.cpp --n-cpu-moe)."""
    L = model.spec.num_layers
    lo, hi = 0, L
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        r = estimate(model, gpu, replace(st, gpu_layers=L + 1, cpu_moe_layers=mid), hw, limits=False)
        if r["status"] in ("fits", "tight", "offload"):
            best = r
            hi = mid - 1
        else:
            lo = mid + 1
    return best


def optimize(model: LoadedModel, gpu: GPU, base: Settings, hw: Optional[HardwareProfile] = None,
             prefer: str = "balanced", lock_quant: bool = False, lock_kv: bool = False,
             target_global_batch: int = 16) -> dict:
    if prefer not in PREFERENCES:
        raise ValueError(f"prefer must be one of {', '.join(PREFERENCES)}")
    st = replace(base)
    st.engine = normalize_engine(st.engine)
    if st.mode != "inference":
        if st.engine not in TRAIN_ENGINES:
            st.engine = "unsloth"
        return _optimize_training(model, gpu, st, hw, target_global_batch)

    results: List[dict] = []
    explanation: List[str] = []
    if st.engine in GGUF_ENGINES:
        quants = [st.quant] if (lock_quant and st.quant) else _gguf_candidates(model)
        kvs = [st.kv_cache] if (lock_kv and st.kv_cache) else ["f16", "q8_0", "q4_0"]
        for q in quants:
            for kv in kvs:
                t = replace(st, quant=q, kv_cache=kv, flash_attention=True, gpu_layers=None, cpu_moe_layers=0)
                r = estimate(model, gpu, t, hw, limits=False)
                results.append(r)
                if model.spec.is_moe and r["status"] == "offload":
                    m = _moe_cpu_search(model, gpu, t, hw)
                    if m is not None:
                        results.append(m)
    elif st.engine == "vllm":
        if model.native_quant and model.spec.quantization:
            quants = [model.native_quant]
        elif lock_quant and st.quant:
            quants = [st.quant]
        else:
            quants = ["BF16", "FP8"] + (["NF4"] if gpu.vendor == "nvidia" else [])
        kvs = [st.kv_cache] if (lock_kv and st.kv_cache) else ["auto", "fp8"]
        for q in quants:
            for kv in kvs:
                for eager in (False, True):
                    t = replace(st, quant=q, kv_cache=kv, enforce_eager=eager)
                    results.append(estimate(model, gpu, t, hw, limits=False))
    elif st.engine == "exllamav2":
        quants = [st.quant] if (lock_quant and st.quant) else EXL2_LADDER
        kvs = [st.kv_cache] if (lock_kv and st.kv_cache) else ["FP16", "Q8", "Q6", "Q4"]
        for q in quants:
            for kv in kvs:
                results.append(estimate(model, gpu, replace(st, quant=q, kv_cache=kv), hw, limits=False))
    else:
        results.append(estimate(model, gpu, st, hw, limits=False))

    ranked = sorted(results, key=lambda r: _utility(r, prefer), reverse=True)
    best = ranked[0]
    best_st = _settings_from(best, st)

    # micro-batch tuning for llama.cpp prompt processing: bigger ubatch if VRAM headroom allows
    if st.engine in GGUF_ENGINES and best["status"] in ("fits", "tight", "offload"):
        for ub in (2048, 1024):
            cand = estimate(model, gpu, replace(best_st, ubatch=ub), hw, limits=False)
            if cand["status"] in ("fits",) or (best["status"] == "offload" and cand["status"] == "offload"
                                               and cand["settings"]["gpu_layers"] >= best["settings"]["gpu_layers"]):
                if cand["performance"]["prefill_tok_s"] > best["performance"]["prefill_tok_s"] * 1.02:
                    best_st = replace(best_st, ubatch=ub, gpu_layers=cand["settings"]["gpu_layers"])
                    explanation.append(f"Raised -ub to {ub}: prompt processing "
                                       f"{best['performance']['prefill_tok_s']:,.0f} -> {cand['performance']['prefill_tok_s']:,.0f} tok/s.")
                    break
    if st.engine == "vllm" and gpu.probed and (gpu.extra.get("used_mib") or 0) > 512:
        best_st = replace(best_st, gpu_mem_util=round(max(0.5, 0.9 - gpu.extra["used_mib"] / 1024 / gpu.vram_gib), 2))
        explanation.append("Lowered --gpu-memory-utilization to leave room for VRAM other processes already use.")

    final = estimate(model, gpu, best_st, hw, limits=True)
    s = final["settings"]
    if final["status"] in ("fits", "tight"):
        explanation.insert(0, f"{s['quant']} with {s['kv_cache_k']} KV cache fits entirely in VRAM "
                              f"({final['gpu_total_mib']:,.0f} / {final['budget_mib']:,.0f} MiB).")
    elif final["status"] == "offload":
        explanation.insert(0, f"Nothing fits fully in VRAM at {s['context']:,} tokens; best trade-off keeps "
                              f"{final['offload']['layers_on_gpu']}/{final['offload']['layers_total']} layers on GPU"
                              + (f" with experts of {s['cpu_moe_layers']} layers in RAM" if s['cpu_moe_layers'] else "") + ".")
    else:
        explanation.insert(0, "No configuration fits this hardware at this context; shown is the closest one.")
    lim = final.get("limits", {})
    if lim.get("max_context_full_gpu"):
        explanation.append(f"Max context fully on GPU with these settings: {lim['max_context_full_gpu']:,} tokens.")
    cmd = build_command(model, final, hw)
    seen = set()
    alts = []
    for r in ranked:
        k = (r["settings"]["quant"], r["settings"]["kv_cache_k"], r["settings"]["cpu_moe_layers"] > 0)
        if k in seen:
            continue
        seen.add(k)
        alts.append(_summary(r))
        if len(alts) >= 8:
            break
    return {"best": final, "command": cmd, "alternatives": alts, "explanation": explanation, "prefer": prefer,
            "searched": len(results)}


def _settings_from(r: dict, base: Settings) -> Settings:
    s = r["settings"]
    return replace(base, quant=s["quant"], kv_cache=s["kv_cache_k"], gpu_layers=s["gpu_layers"]
                   if base.engine in GGUF_ENGINES else None, cpu_moe_layers=s["cpu_moe_layers"],
                   flash_attention=s["flash_attention"], ubatch=s["ubatch"],
                   enforce_eager=(r["details"]["overhead"].get("cuda_graphs", 1) == 0) if base.engine == "vllm" else base.enforce_eager)


def _optimize_training(model: LoadedModel, gpu: GPU, st: Settings, hw: Optional[HardwareProfile],
                       target_global_batch: int) -> dict:
    explanation: List[str] = []
    attempts: List[Tuple[str, Settings]] = [("as requested", st)]
    if st.mode == "full":
        attempts.append(("8-bit AdamW", replace(st, optimizer="adamw_8bit")))
        attempts.append(("LoRA instead of full fine-tuning", replace(st, mode="lora", optimizer="adamw_8bit")))
    if st.mode in ("full", "lora"):
        attempts.append(("QLoRA (4-bit base)", replace(st, mode="qlora", quant=None, optimizer="adamw_8bit")))
    if st.engine == "unsloth":
        attempts.append(("Unsloth offloaded gradient checkpointing",
                         replace(attempts[-1][1], grad_checkpointing="unsloth")))
    attempts.append(("paged 8-bit AdamW", replace(attempts[-1][1], optimizer="paged_adamw_8bit")))

    chosen = None
    tried = []
    for label, cand in attempts:
        r = estimate(model, gpu, replace(cand, batch=1), hw, limits=False)
        tried.append(_summary(r) | {"strategy": label, "mode": cand.mode})
        if r["status"] in ("fits", "tight"):
            chosen = (label, cand)
            break
    if chosen is None:
        label, cand = attempts[-1]
        explanation.append(f"Even batch 1 does not fit at {st.context:,} tokens with {label}.")
        lo, hi, best_seq = 128, st.context, 0
        while lo <= hi:
            mid = (lo + hi) // 2 // 64 * 64 or 64
            r = estimate(model, gpu, replace(cand, batch=1, context=mid), hw, limits=False)
            if r["status"] in ("fits", "tight"):
                best_seq, lo = mid, mid + 64
            else:
                hi = mid - 64
        if best_seq:
            explanation.append(f"Largest sequence length that fits: {best_seq:,} tokens.")
            cand = replace(cand, context=best_seq)
        final = estimate(model, gpu, replace(cand, batch=1), hw, limits=True)
        return {"best": final, "command": build_command(model, final, hw, 1), "alternatives": tried,
                "explanation": explanation, "prefer": "fit", "searched": len(tried)}
    label, cand = chosen
    if label != "as requested":
        explanation.append(f"Switched to {label} to fit in VRAM.")
    micro = 1
    b = 2
    while b <= 64:
        r = estimate(model, gpu, replace(cand, batch=b), hw, limits=False)
        if r["status"] not in ("fits", "tight"):
            break
        micro = b
        b *= 2
    accum = max(1, math.ceil(target_global_batch / micro))
    final = estimate(model, gpu, replace(cand, batch=micro, grad_accum=accum), hw, limits=True)
    explanation.insert(0, f"{final['settings']['mode'].upper()} fits with micro-batch {micro} "
                          f"x {accum} accumulation steps (effective batch {micro * accum}) at {cand.context:,} tokens.")
    return {"best": final, "command": build_command(model, final, hw, accum), "alternatives": tried,
            "explanation": explanation, "prefer": "fit", "searched": len(tried) + 6}
