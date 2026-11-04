"""Built-in architecture catalog.

These are verbatim (architecture-relevant subsets of) the ``config.json`` files
of popular open-weight models. They let ``vram-fit`` work fully offline, power
short aliases such as ``llama3.1-8b``, and act as a fallback when the Hugging
Face Hub is unreachable or a repo is gated and no ``HF_TOKEN`` is set.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

_LLAMA3_ROPE = {"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192, "rope_type": "llama3"}


def _llama(h, layers, heads, kv, inter, vocab=128256, tie=False, max_pos=131072, head_dim=None):
    cfg = {
        "architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": h,
        "num_hidden_layers": layers, "num_attention_heads": heads, "num_key_value_heads": kv,
        "intermediate_size": inter, "vocab_size": vocab, "tie_word_embeddings": tie,
        "max_position_embeddings": max_pos, "rope_theta": 500000.0, "rope_scaling": dict(_LLAMA3_ROPE),
        "torch_dtype": "bfloat16", "hidden_act": "silu",
    }
    if head_dim:
        cfg["head_dim"] = head_dim
    return cfg


def _qwen2(h, layers, heads, kv, inter, vocab=152064, tie=False, max_pos=32768):
    return {
        "architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "hidden_size": h,
        "num_hidden_layers": layers, "num_attention_heads": heads, "num_key_value_heads": kv,
        "intermediate_size": inter, "vocab_size": vocab, "tie_word_embeddings": tie,
        "max_position_embeddings": max_pos, "rope_theta": 1000000.0, "torch_dtype": "bfloat16",
        "use_sliding_window": False, "hidden_act": "silu",
    }


def _qwen3(h, layers, heads, kv, inter, tie=False):
    return {
        "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": h,
        "num_hidden_layers": layers, "num_attention_heads": heads, "num_key_value_heads": kv,
        "head_dim": 128, "intermediate_size": inter, "vocab_size": 151936,
        "tie_word_embeddings": tie, "max_position_embeddings": 40960, "rope_theta": 1000000.0,
        "torch_dtype": "bfloat16", "attention_bias": False, "hidden_act": "silu",
    }


def _gemma3(h, layers, heads, kv, inter, head_dim=128):
    return {
        "architectures": ["Gemma3ForConditionalGeneration"], "model_type": "gemma3",
        "text_config": {
            "model_type": "gemma3_text", "hidden_size": h, "num_hidden_layers": layers,
            "num_attention_heads": heads, "num_key_value_heads": kv, "head_dim": head_dim,
            "intermediate_size": inter, "vocab_size": 262208, "sliding_window": 1024,
            "sliding_window_pattern": 6, "max_position_embeddings": 131072,
            "rope_theta": 1000000.0, "rope_scaling": {"factor": 8.0, "rope_type": "linear"},
            "hidden_activation": "gelu_pytorch_tanh",
        },
        "vision_config": {
            "model_type": "siglip_vision_model", "hidden_size": 1152, "intermediate_size": 4304,
            "num_hidden_layers": 27, "num_attention_heads": 16, "patch_size": 14, "image_size": 896,
        },
        "mm_tokens_per_image": 256, "torch_dtype": "bfloat16",
    }


def _gpt_oss(layers, experts):
    return {
        "architectures": ["GptOssForCausalLM"], "model_type": "gpt_oss", "hidden_size": 2880,
        "intermediate_size": 2880, "num_hidden_layers": layers, "num_attention_heads": 64,
        "num_key_value_heads": 8, "head_dim": 64, "num_local_experts": experts,
        "num_experts_per_tok": 4, "vocab_size": 201088, "sliding_window": 128,
        "layer_types": ["sliding_attention", "full_attention"] * (layers // 2),
        "max_position_embeddings": 131072, "tie_word_embeddings": False, "attention_bias": True,
        "rope_theta": 150000, "torch_dtype": "bfloat16",
        "rope_scaling": {"rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0, "beta_slow": 1.0,
                         "original_max_position_embeddings": 4096},
        "quantization_config": {"quant_method": "mxfp4",
                                "modules_to_not_convert": ["model.layers.*.self_attn", "model.layers.*.mlp.router",
                                                           "model.embed_tokens", "lm_head"]},
    }


CATALOG: Dict[str, dict] = {
    "meta-llama/Llama-3.2-1B": _llama(2048, 16, 32, 8, 8192, tie=True, head_dim=64),
    "meta-llama/Llama-3.2-3B": _llama(3072, 28, 24, 8, 8192, tie=True, head_dim=128),
    "meta-llama/Llama-3.1-8B": _llama(4096, 32, 32, 8, 14336),
    "meta-llama/Llama-3.1-70B": _llama(8192, 80, 64, 8, 28672),
    "meta-llama/Llama-3.3-70B-Instruct": _llama(8192, 80, 64, 8, 28672),
    "Qwen/Qwen2.5-0.5B": _qwen2(896, 24, 14, 2, 4864, vocab=151936, tie=True),
    "Qwen/Qwen2.5-1.5B": _qwen2(1536, 28, 12, 2, 8960, vocab=151936, tie=True),
    "Qwen/Qwen2.5-3B": _qwen2(2048, 36, 16, 2, 11008, vocab=151936, tie=True),
    "Qwen/Qwen2.5-7B": _qwen2(3584, 28, 28, 4, 18944),
    "Qwen/Qwen2.5-14B": _qwen2(5120, 48, 40, 8, 13824),
    "Qwen/Qwen2.5-32B": _qwen2(5120, 64, 40, 8, 27648),
    "Qwen/Qwen2.5-72B": _qwen2(8192, 80, 64, 8, 29568),
    "Qwen/Qwen3-4B": _qwen3(2560, 36, 32, 8, 9728, tie=True),
    "Qwen/Qwen3-8B": _qwen3(4096, 36, 32, 8, 12288),
    "Qwen/Qwen3-14B": _qwen3(5120, 40, 40, 8, 17408),
    "Qwen/Qwen3-32B": _qwen3(5120, 64, 64, 8, 25600),
    "Qwen/Qwen3-30B-A3B": {
        "architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe", "hidden_size": 2048,
        "num_hidden_layers": 48, "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
        "intermediate_size": 6144, "moe_intermediate_size": 768, "num_experts": 128,
        "num_experts_per_tok": 8, "decoder_sparse_step": 1, "mlp_only_layers": [],
        "vocab_size": 151936, "tie_word_embeddings": False, "max_position_embeddings": 40960,
        "rope_theta": 1000000.0, "torch_dtype": "bfloat16",
    },
    "Qwen/Qwen3-235B-A22B": {
        "architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe", "hidden_size": 4096,
        "num_hidden_layers": 94, "num_attention_heads": 64, "num_key_value_heads": 4, "head_dim": 128,
        "intermediate_size": 12288, "moe_intermediate_size": 1536, "num_experts": 128,
        "num_experts_per_tok": 8, "decoder_sparse_step": 1, "mlp_only_layers": [],
        "vocab_size": 151936, "tie_word_embeddings": False, "max_position_embeddings": 40960,
        "rope_theta": 1000000.0, "torch_dtype": "bfloat16",
    },
    "mistralai/Mistral-7B-v0.3": {
        "architectures": ["MistralForCausalLM"], "model_type": "mistral", "hidden_size": 4096,
        "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8,
        "intermediate_size": 14336, "vocab_size": 32768, "max_position_embeddings": 32768,
        "sliding_window": None, "rope_theta": 1000000.0, "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
    },
    "mistralai/Mistral-Small-24B-Instruct-2501": {
        "architectures": ["MistralForCausalLM"], "model_type": "mistral", "hidden_size": 5120,
        "num_hidden_layers": 40, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
        "intermediate_size": 32768, "vocab_size": 131072, "max_position_embeddings": 32768,
        "sliding_window": None, "rope_theta": 100000000.0, "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
    },
    "mistralai/Mixtral-8x7B-v0.1": {
        "architectures": ["MixtralForCausalLM"], "model_type": "mixtral", "hidden_size": 4096,
        "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8,
        "intermediate_size": 14336, "num_local_experts": 8, "num_experts_per_tok": 2,
        "vocab_size": 32000, "max_position_embeddings": 32768, "rope_theta": 1000000.0,
        "tie_word_embeddings": False, "torch_dtype": "bfloat16",
    },
    "google/gemma-2-9b": {
        "architectures": ["Gemma2ForCausalLM"], "model_type": "gemma2", "hidden_size": 3584,
        "num_hidden_layers": 42, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 256,
        "intermediate_size": 14336, "vocab_size": 256000, "sliding_window": 4096,
        "max_position_embeddings": 8192, "rope_theta": 10000.0, "tie_word_embeddings": True,
        "torch_dtype": "float32",
    },
    "google/gemma-2-27b": {
        "architectures": ["Gemma2ForCausalLM"], "model_type": "gemma2", "hidden_size": 4608,
        "num_hidden_layers": 46, "num_attention_heads": 32, "num_key_value_heads": 16, "head_dim": 128,
        "intermediate_size": 36864, "vocab_size": 256000, "sliding_window": 4096,
        "max_position_embeddings": 8192, "rope_theta": 10000.0, "tie_word_embeddings": True,
        "torch_dtype": "float32",
    },
    "google/gemma-3-12b-it": _gemma3(3840, 48, 16, 8, 15360, head_dim=256),
    "google/gemma-3-27b-it": _gemma3(5376, 62, 32, 16, 21504),
    "microsoft/Phi-3-mini-4k-instruct": {
        "architectures": ["Phi3ForCausalLM"], "model_type": "phi3", "hidden_size": 3072,
        "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 32,
        "intermediate_size": 8192, "vocab_size": 32064, "max_position_embeddings": 4096,
        "tie_word_embeddings": False, "torch_dtype": "bfloat16",
    },
    "deepseek-ai/DeepSeek-V3": {
        "architectures": ["DeepseekV3ForCausalLM"], "model_type": "deepseek_v3", "hidden_size": 7168,
        "num_hidden_layers": 61, "num_attention_heads": 128, "num_key_value_heads": 128,
        "intermediate_size": 18432, "moe_intermediate_size": 2048, "n_routed_experts": 256,
        "n_shared_experts": 1, "num_experts_per_tok": 8, "first_k_dense_replace": 3,
        "moe_layer_freq": 1, "kv_lora_rank": 512, "q_lora_rank": 1536, "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 128, "v_head_dim": 128, "vocab_size": 129280,
        "max_position_embeddings": 163840, "tie_word_embeddings": False, "torch_dtype": "bfloat16",
        "rope_scaling": {"type": "yarn", "factor": 40, "original_max_position_embeddings": 4096},
        "quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]},
    },
    "openai/gpt-oss-20b": _gpt_oss(24, 32),
    "openai/gpt-oss-120b": _gpt_oss(36, 128),
}

# DeepSeek-R1 shares the V3 architecture.
CATALOG["deepseek-ai/DeepSeek-R1"] = dict(CATALOG["deepseek-ai/DeepSeek-V3"])

ALIASES: Dict[str, str] = {
    "llama3.2-1b": "meta-llama/Llama-3.2-1B",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B",
    "llama3-8b": "meta-llama/Llama-3.1-8B",
    "llama3.1-70b": "meta-llama/Llama-3.1-70B",
    "llama3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B",
    "qwen2.5-32b": "Qwen/Qwen2.5-32B",
    "qwen2.5-72b": "Qwen/Qwen2.5-72B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "qwen3-32b": "Qwen/Qwen3-32B",
    "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",
    "qwen3-235b-a22b": "Qwen/Qwen3-235B-A22B",
    "mistral-7b": "mistralai/Mistral-7B-v0.3",
    "mistral-small-24b": "mistralai/Mistral-Small-24B-Instruct-2501",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-v0.1",
    "gemma2-9b": "google/gemma-2-9b",
    "gemma2-27b": "google/gemma-2-27b",
    "gemma3-12b": "google/gemma-3-12b-it",
    "gemma3-27b": "google/gemma-3-27b-it",
    "phi3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "deepseek-r1": "deepseek-ai/DeepSeek-R1",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
}

_DISTILL = {
    "deepseek-r1-distill-qwen-1.5b": "Qwen/Qwen2.5-1.5B",
    "deepseek-r1-distill-qwen-7b": "Qwen/Qwen2.5-7B",
    "deepseek-r1-distill-qwen-14b": "Qwen/Qwen2.5-14B",
    "deepseek-r1-distill-qwen-32b": "Qwen/Qwen2.5-32B",
    "deepseek-r1-distill-llama-8b": "meta-llama/Llama-3.1-8B",
    "deepseek-r1-distill-llama-70b": "meta-llama/Llama-3.3-70B-Instruct",
}

_SUFFIXES = (
    "-gguf", "-awq", "-gptq", "-int4", "-int8", "-fp8", "-fp8-dynamic", "-bnb-4bit", "-unsloth-bnb-4bit",
    "-4bit", "-8bit", "-exl2", "-mlx", "-instruct", "-it", "-chat", "-base", "-hf", "-1m",
    "-instruct-2501", "-2507", "-thinking", "-coder",
)


def _norm(s: str) -> str:
    return s.strip().lower()


def _strip(name: str) -> str:
    n = _norm(name.split("/")[-1])
    changed = True
    while changed:
        changed = False
        for suf in _SUFFIXES:
            if n.endswith(suf):
                n = n[: -len(suf)]
                changed = True
    return n


_BY_LOWER = {_norm(k): k for k in CATALOG}
_BY_STEM: Dict[str, str] = {}
for _k in CATALOG:
    _BY_STEM.setdefault(_strip(_k), _k)


def lookup(model_id: str) -> Optional[Tuple[str, dict]]:
    """Find a catalog entry for an id, alias, or a derivative repo name.

    ``bartowski/Qwen2.5-14B-Instruct-GGUF`` -> ``Qwen/Qwen2.5-14B``.
    """
    key = _norm(model_id)
    if key in ALIASES:
        cid = ALIASES[key]
        return cid, CATALOG[cid]
    if key in _BY_LOWER:
        cid = _BY_LOWER[key]
        return cid, CATALOG[cid]
    stem = _strip(model_id)
    if stem in _BY_STEM:
        cid = _BY_STEM[stem]
        return cid, CATALOG[cid]
    stem2 = re.sub(r"^(meta-)?llama-?", "llama-", stem)
    for s, cid in _BY_STEM.items():
        if s == stem2 or s.replace("-", "") == stem.replace("-", ""):
            return cid, CATALOG[cid]
    if stem in _DISTILL:
        cid = _DISTILL[stem]
        return cid, CATALOG[cid]
    return None


def list_catalog() -> List[dict]:
    return [{"id": k, "alias": next((a for a, v in ALIASES.items() if v == k), None)} for k in CATALOG]
