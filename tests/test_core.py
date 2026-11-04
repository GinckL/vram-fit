import math

import pytest

from vram_fit.catalog import CATALOG, lookup
from vram_fit.ggml import ggml_type
from vram_fit.gguf import parse_bytes
from vram_fit.model import model_from_gguf, model_from_hf_config
from vram_fit.quant import get_quant, make_quant_context, quant_from_filename, quant_from_hf_config

from helpers import make_gguf_bytes


def _lm(cid):
    return model_from_hf_config(CATALOG[cid], cid)


def _bytes(lm, quant):
    s = lm.spec
    q = get_quant(quant)
    qc = make_quant_context(s, lm.inventory)
    return sum(q.tensor_bytes(t, qc) for t in lm.inventory if not t.tied and t.kind != "vision")


def test_ggml_block_sizes():
    assert ggml_type("Q4_K").nbytes(256) == 144
    assert ggml_type("Q8_0").nbytes(64) == 68
    assert ggml_type("Q6_K").bits_per_weight == pytest.approx(6.5625)
    assert ggml_type("IQ4_XS").bits_per_weight == pytest.approx(4.25)
    assert ggml_type("MXFP4").bits_per_weight == pytest.approx(4.25)


@pytest.mark.parametrize("cid,params", [
    ("meta-llama/Llama-3.1-8B", 8.03e9),
    ("meta-llama/Llama-3.1-70B", 70.55e9),
    ("meta-llama/Llama-3.2-1B", 1.236e9),
    ("Qwen/Qwen2.5-7B", 7.62e9),
    ("Qwen/Qwen2.5-14B", 14.77e9),
    ("Qwen/Qwen2.5-72B", 72.7e9),
    ("Qwen/Qwen3-30B-A3B", 30.53e9),
    ("mistralai/Mixtral-8x7B-v0.1", 46.7e9),
    ("deepseek-ai/DeepSeek-V3", 671.0e9),
    ("openai/gpt-oss-20b", 20.9e9),
    ("google/gemma-2-9b", 9.24e9),
])
def test_param_counts_match_published(cid, params):
    assert _lm(cid).params_total == pytest.approx(params, rel=0.01)


def test_moe_active_params():
    assert _lm("Qwen/Qwen3-30B-A3B").params_active == pytest.approx(3.35e9, rel=0.03)
    assert _lm("deepseek-ai/DeepSeek-V3").params_active == pytest.approx(37.5e9, rel=0.03)
    assert _lm("mistralai/Mixtral-8x7B-v0.1").params_active == pytest.approx(12.9e9, rel=0.02)


@pytest.mark.parametrize("cid,quant,file_bytes", [
    # published GGUF file sizes (header included, a few MB)
    ("meta-llama/Llama-3.1-8B", "Q4_K_M", 4_920_739_232),
    ("meta-llama/Llama-3.1-8B", "Q8_0", 8_540_770_784),
    ("Qwen/Qwen2.5-7B", "Q4_K_M", 4_683_073_504),
    ("Qwen/Qwen2.5-32B", "Q4_K_M", 19_851_336_256),
])
def test_synthesized_gguf_size_matches_real_files(cid, quant, file_bytes):
    assert _bytes(_lm(cid), quant) == pytest.approx(file_bytes, rel=0.015)


def test_llama_cpp_mixing_rules():
    lm = _lm("meta-llama/Llama-3.1-8B")
    s = lm.spec
    q = get_quant("Q4_K_M")
    qc = make_quant_context(s, lm.inventory)
    types = {t.name: q.tensor_type(t, qc) for t in lm.inventory}
    assert types["output.weight"] == "Q6_K"
    assert types["token_embd.weight"] == "Q4_K"
    assert types["blk.0.attn_v.weight"] == "Q6_K"      # first 1/8 of layers get more bits
    assert types["blk.5.attn_v.weight"] == "Q4_K"
    assert types["blk.6.attn_v.weight"] == "Q6_K"      # (i - n/8) % 3 == 2
    assert types["blk.10.ffn_down.weight"] == "Q4_K"
    assert types["blk.0.attn_norm.weight"] == "F32"
    assert s.num_layers == 32


def test_divisibility_fallback():
    # Qwen2.5-72B intermediate 29568 is not a multiple of 256 -> ffn_down falls back to a 32-block type
    lm = _lm("Qwen/Qwen2.5-72B")
    q = get_quant("Q4_K_M")
    qc = make_quant_context(lm.spec, lm.inventory)
    t = next(t for t in lm.inventory if t.name == "blk.10.ffn_down.weight")
    assert q.tensor_type(t, qc) in ("Q5_0", "Q8_0")


def test_quant_lookup_and_aliases():
    assert get_quant("q4_k_m").name == "Q4_K_M"
    assert get_quant("fp16").name == "F16"
    assert get_quant("exl2-4.65").nominal_bpw == pytest.approx(4.65)
    assert get_quant("6.0bpw").family == "exl2"
    with pytest.raises(KeyError):
        get_quant("Q9_K")
    assert quant_from_filename("Qwen2.5-14B-Instruct-Q4_K_M.gguf") == "Q4_K_M"
    assert quant_from_filename("model-IQ4_XS-00001-of-00002.gguf") == "IQ4_XS"
    assert quant_from_filename("gemma-3-27b-it-BF16.gguf") == "BF16"
    assert quant_from_hf_config({"quant_method": "awq", "bits": 4}) == "AWQ"
    assert quant_from_hf_config({"quant_method": "bitsandbytes", "load_in_4bit": True}) == "NF4"
    assert quant_from_hf_config({"quant_method": "compressed-tensors", "config_groups": {
        "g": {"weights": {"num_bits": 8, "type": "float"}}}}) == "FP8"


def test_nonlinear_tensors_stay_16bit_for_awq():
    lm = _lm("meta-llama/Llama-3.1-8B")
    awq = _bytes(lm, "AWQ")
    linear = sum(t.numel for t in lm.inventory if t.is_linear)
    other = sum(t.numel for t in lm.inventory if not t.is_linear and not t.tied)
    assert awq == pytest.approx(linear * 4.156 / 8 + other * 2, rel=0.01)


def test_gguf_header_roundtrip_and_exact_inventory():
    raw, data_bytes = make_gguf_bytes("meta-llama/Llama-3.2-1B", "Q4_K_M")
    hdr = parse_bytes(raw)
    assert hdr.architecture == "llama"
    assert hdr.file_type == "Q4_K_M"
    assert hdr.array_len("tokenizer.ggml.tokens") == 64
    assert hdr.tensor_bytes == data_bytes
    lm = model_from_gguf([hdr], "test/llama-1b")
    assert lm.exact and lm.native_quant == "Q4_K_M"
    assert lm.spec.num_layers == 16 and lm.spec.num_kv_heads[0] == 8 and lm.spec.head_dim == 64
    assert lm.spec.vocab_size == 128256          # taken from the token_embd tensor shape
    assert lm.spec.tie_embeddings                 # no output.weight -> tied head synthesized
    head = next(t for t in lm.inventory if t.kind == "head")
    assert head.tied and head.dtype == "Q6_K"     # llama.cpp quantizes a tied embedding as the output


def test_catalog_lookup_derivatives():
    assert lookup("bartowski/Qwen2.5-14B-Instruct-GGUF")[0] == "Qwen/Qwen2.5-14B"
    assert lookup("unsloth/Llama-3.1-8B-Instruct-bnb-4bit")[0] == "meta-llama/Llama-3.1-8B"
    assert lookup("llama3.1-8b")[0] == "meta-llama/Llama-3.1-8B"
    assert lookup("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")[0] == "Qwen/Qwen2.5-32B"
    assert lookup("totally-unknown/model") is None


def test_config_layouts():
    g3 = _lm("google/gemma-3-27b-it").spec
    assert g3.hidden_size == 5376 and g3.sliding_window == 1024
    assert g3.layer_types.count("full") == 10 and g3.layer_types.count("sliding") == 52
    assert g3.vision_params > 3e8
    g2 = _lm("google/gemma-2-9b").spec
    assert g2.layer_types[0] == "sliding" and g2.layer_types[1] == "full"
    oss = _lm("openai/gpt-oss-20b")
    assert oss.native_quant == "MXFP4" and oss.spec.sliding_window == 128
    ds = _lm("deepseek-ai/DeepSeek-V3").spec
    assert ds.is_mla and ds.kv_elems_per_token(0) == 576
    assert sum(ds.moe_layers) == 58
    q2 = _lm("Qwen/Qwen2.5-7B").spec
    assert q2.qkv_bias and q2.attention_kind() == "GQA 28:4"


def test_hybrid_and_text_config_parsing():
    cfg = {"architectures": ["X"], "model_type": "qwen3_next", "hidden_size": 2048, "num_hidden_layers": 8,
           "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256, "intermediate_size": 5120,
           "vocab_size": 151936, "full_attention_interval": 4, "linear_num_value_heads": 32,
           "linear_num_key_heads": 16, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
           "linear_conv_kernel_dim": 4, "max_position_embeddings": 262144}
    s = model_from_hf_config(cfg, "x/hybrid").spec
    assert s.layer_types.count("linear") == 6 and s.layer_types.count("full") == 2
    assert s.linear_state_bytes == 32 * 128 * 128 * 4 + (2 * 16 * 128 + 32 * 128) * 3 * 2
    assert math.isclose(s.kv_elems_per_token(3), 2 * 256 * 2)
    assert s.kv_elems_per_token(0) == 0


def test_llama_70b_rule_and_published_size():
    # llama.cpp bumps attn_v of 80-layer GQA llama/qwen2 models (LLM_TYPE_70B) from Q4_K to Q5_K
    lm = _lm("meta-llama/Llama-3.3-70B-Instruct")
    q = get_quant("Q4_K_M")
    qc = make_quant_context(lm.spec, lm.inventory)
    assert qc.is_70b
    t = next(t for t in lm.inventory if t.name == "blk.20.attn_v.weight")
    assert q.tensor_type(t, qc) in ("Q5_K", "Q6_K")
    assert _bytes(lm, "Q4_K_M") == pytest.approx(42.52e9, rel=0.003)  # published file ~42.52 GB


def test_moe_and_iq_rules():
    mix = _lm("mistralai/Mixtral-8x7B-v0.1")
    qc = make_quant_context(mix.spec, mix.inventory)
    types = {t.name: get_quant("Q4_K_M").tensor_type(t, qc) for t in mix.inventory}
    assert types["blk.3.attn_v.weight"] == "Q8_0" and types["blk.3.attn_k.weight"] == "Q8_0"
    assert types["blk.3.attn_output.weight"] == "Q5_K"
    assert types["blk.3.ffn_gate_inp.weight"] == "F32"           # routers are never quantized
    llama = _lm("meta-llama/Llama-3.1-8B")
    qc = make_quant_context(llama.spec, llama.inventory)
    tt = {t.name: get_quant("IQ2_M").tensor_type(t, qc) for t in llama.inventory}
    assert tt["token_embd.weight"] == "IQ3_S" and tt["output.weight"] == "Q5_K"
    assert tt["blk.0.ffn_down.weight"] == "IQ3_S" and tt["blk.10.ffn_down.weight"] == "IQ2_S"


def test_real_llama_cpp_gguf_files():
    """Optional: point VRAM_FIT_REAL_GGUF_DIR at llama.cpp's models/ggml-vocab-*.gguf files."""
    import glob
    import os
    d = os.environ.get("VRAM_FIT_REAL_GGUF_DIR")
    if not d:
        pytest.skip("set VRAM_FIT_REAL_GGUF_DIR to run against real GGUF files")
    from vram_fit.gguf import parse_file
    files = glob.glob(os.path.join(d, "*.gguf"))
    assert files
    for f in files:
        h = parse_file(f)
        if not h.tensors:
            assert h.header_bytes == os.path.getsize(f), f
        assert h.array_len("tokenizer.ggml.tokens"), f
