import pytest

from vram_fit.catalog import CATALOG
from vram_fit.estimator import Settings, estimate, lora_trainable, weights_profile
from vram_fit.model import model_from_hf_config
from vram_fit.presets import parse_gpu_spec
from vram_fit.quant import get_quant

MiB = 1024 * 1024


def lm(cid="meta-llama/Llama-3.1-8B"):
    return model_from_hf_config(CATALOG[cid], cid)


def seg(r, key):
    return next(s["mib"] for s in r["segments"] if s["key"] == key)


def test_kv_cache_matches_llama_cpp_log():
    # llama.cpp: "KV self size = 1024.00 MiB, K (f16): 512.00 MiB, V (f16): 512.00 MiB" for Llama-3-8B @ 8192
    r = estimate(lm(), parse_gpu_spec("rtx-4090"), Settings(context=8192, kv_cache="f16"), limits=False)
    assert seg(r, "kv_cache") == pytest.approx(1024.0)
    r8 = estimate(lm(), parse_gpu_spec("rtx-4090"), Settings(context=8192, kv_cache="8"), limits=False)
    assert seg(r8, "kv_cache") == pytest.approx(544.0)
    assert r8["settings"]["kv_cache_k"] == "q8_0"


def test_v_cache_needs_flash_attention():
    r = estimate(lm(), parse_gpu_spec("rtx-4090"), Settings(context=8192, kv_cache="q8_0", flash_attention=False),
                 limits=False)
    assert r["settings"]["kv_cache_v"] == "f16"
    assert any("flash attention" in w for w in r["warnings"])


def test_sliding_window_and_mla_kv():
    g = parse_gpu_spec("h100")
    gem = estimate(lm("google/gemma-3-27b-it"), g, Settings(context=131072, kv_cache="f16"), limits=False)
    full = 10 * 131072 * 16 * 128 * 2 * 2
    assert gem["details"]["kv_cache"]["total_mib"] < (full / MiB) * 1.2  # sliding layers are capped at ~1k tokens
    ds = estimate(lm("deepseek-ai/DeepSeek-V3"), parse_gpu_spec("8x h200"),
                  Settings(engine="vllm", context=32768), limits=False)
    assert ds["details"]["kv_cache"]["total_mib"] == pytest.approx(61 * 32768 * 576 * 2 / MiB, rel=0.001)


def test_statuses():
    big = estimate(lm("meta-llama/Llama-3.1-70B"), parse_gpu_spec("rtx-4060"), Settings(context=4096, ram_total_gib=64),
                   limits=False)
    assert big["status"] == "offload" and big["offload"]["layers_on_gpu"] < 80
    assert big["fits"] is True
    small = estimate(lm(), parse_gpu_spec("rtx-4090"), Settings(context=4096), limits=False)
    assert small["status"] == "fits" and small["offload"]["layers_on_gpu"] == 32
    forced = estimate(lm("meta-llama/Llama-3.1-70B"), parse_gpu_spec("rtx-4060"),
                      Settings(context=4096, gpu_layers=81, ram_total_gib=64), limits=False)
    assert forced["status"] == "spill" and forced["offload"]["spill_mib"] > 0
    huge = estimate(lm("deepseek-ai/DeepSeek-V3"), parse_gpu_spec("rtx-4060"),
                    Settings(context=4096, ram_total_gib=64), limits=False)
    assert huge["status"] == "oom"


def test_more_context_means_fewer_gpu_layers():
    g = parse_gpu_spec("rtx-3060-12gb")
    m = lm("Qwen/Qwen2.5-14B")
    a = estimate(m, g, Settings(context=4096), limits=False)
    b = estimate(m, g, Settings(context=65536), limits=False)
    assert a["settings"]["gpu_layers"] > b["settings"]["gpu_layers"]
    assert b["performance"]["decode_tok_s"] < a["performance"]["decode_tok_s"]


def test_apple_wired_limit_and_spill_hint():
    g = parse_gpu_spec("m2-16gb")
    assert g.usable_gib == pytest.approx(16 * 2 / 3, rel=0.01)
    r = estimate(lm(), g, Settings(quant="Q8_0", context=32768, gpu_layers=33), limits=False)
    assert r["status"] == "spill"
    assert any("iogpu.wired_limit_mb" in w for w in r["warnings"])


def test_vllm_budget_and_concurrency():
    r = estimate(lm(), parse_gpu_spec("rtx-4090"), Settings(engine="vllm", quant="BF16", context=8192), limits=False)
    assert r["budget_mib"] == pytest.approx(24 * 1024 * 0.9)
    assert r["details"]["kv_cache"]["vllm_max_concurrency_at_ctx"] > 1
    fp8 = estimate(lm(), parse_gpu_spec("rtx-4090"), Settings(engine="vllm", quant="FP8", kv_cache="fp8", context=8192),
                   limits=False)
    assert seg(fp8, "weights") < seg(r, "weights") * 0.6
    assert seg(fp8, "kv_cache") == pytest.approx(seg(r, "kv_cache") / 2)


def test_lora_trainable_params_match_peft():
    m = lm()
    prof = weights_profile(m, get_quant("BF16"), "unsloth")
    params, _, _ = lora_trainable(prof, 16, None)
    assert params == 41_943_040  # PEFT print_trainable_parameters for Llama-3-8B, r=16, all linear


def test_training_memory_ordering():
    g = parse_gpu_spec("a100-80gb")
    m = lm()
    q = estimate(m, g, Settings(engine="unsloth", mode="qlora", context=2048, batch=2), limits=False)
    lo = estimate(m, g, Settings(engine="unsloth", mode="lora", context=2048, batch=2), limits=False)
    full = estimate(m, g, Settings(engine="transformers", mode="full", context=2048, batch=2, optimizer="adamw"),
                    limits=False)
    assert q["gpu_total_mib"] < lo["gpu_total_mib"] < full["gpu_total_mib"]
    # full fine-tune with AdamW: grads (2) + fp32 master (4) + Adam states (8) bytes per parameter
    assert seg(full, "lora_optimizer") == pytest.approx(m.params_total * 14 / MiB, rel=0.001)
    assert q["settings"]["quant"] == "NF4"
    assert q["performance"]["train_tok_s"] > 0


def test_unsloth_gc_offloads_to_ram():
    g = parse_gpu_spec("rtx-4090")
    m = lm()
    std = estimate(m, g, Settings(engine="unsloth", mode="qlora", context=8192, grad_checkpointing="standard"), limits=False)
    uns = estimate(m, g, Settings(engine="unsloth", mode="qlora", context=8192, grad_checkpointing="unsloth"), limits=False)
    assert seg(uns, "activations") < seg(std, "activations")
    assert uns["offload"]["cpu_buffers_mib"] > 0


def test_moe_cpu_offload_keeps_attention_on_gpu():
    m = lm("Qwen/Qwen3-30B-A3B")
    g = parse_gpu_spec("rtx-4060-ti-8gb")
    r = estimate(m, g, Settings(context=8192, gpu_layers=49, cpu_moe_layers=40), limits=False)
    assert r["status"] in ("offload", "fits", "tight")
    assert r["offload"]["layers_on_gpu"] == 48
    assert r["offload"]["cpu_weights_mib"] > 5000


def test_limits_and_speed():
    import time
    m, g = lm(), parse_gpu_spec("rtx-4090")
    t = time.perf_counter()
    for c in range(2048, 131072, 4096):
        estimate(m, g, Settings(context=c), limits=False)
    per = (time.perf_counter() - t) / 32
    assert per < 0.02  # well under one 60 fps frame
    r = estimate(m, g, Settings(context=8192))
    assert r["limits"]["max_context_full_gpu"] > 100_000


def test_cpu_only():
    from vram_fit.hardware import cpu_only_gpu
    r = estimate(lm(), cpu_only_gpu(60, 16), Settings(context=4096), limits=False)
    assert r["status"] == "offload" and r["gpu_total_mib"] == 0
    assert 3 < r["performance"]["decode_tok_s"] < 30
