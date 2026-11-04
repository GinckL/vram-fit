"""Tensor inventory primitives shared by the model loader, quantizer and estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

ATTN_KINDS = frozenset({
    "attn_q", "attn_k", "attn_v", "attn_o", "attn_qkv",
    "attn_q_a", "attn_q_b", "attn_kv_a", "attn_kv_b", "attn_k_b", "attn_v_b",
})
DENSE_FFN_KINDS = frozenset({"ffn_gate", "ffn_up", "ffn_down", "ffn_gate_up"})
EXPERT_KINDS = frozenset({"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_up_exps"})
SHARED_EXPERT_KINDS = frozenset({"ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"})
SSM_KINDS = frozenset({"ssm_in", "ssm_out", "ssm_other"})
LINEAR_KINDS = ATTN_KINDS | DENSE_FFN_KINDS | EXPERT_KINDS | SHARED_EXPERT_KINDS | {"ssm_in", "ssm_out"}

# Which HF/PEFT ``target_modules`` names map onto which tensor kinds.
LORA_TARGET_KINDS = {
    "q_proj": {"attn_q", "attn_q_a", "attn_q_b", "attn_qkv"},
    "k_proj": {"attn_k", "attn_kv_a", "attn_kv_b", "attn_qkv"},
    "v_proj": {"attn_v", "attn_kv_b", "attn_qkv"},
    "o_proj": {"attn_o"},
    "gate_proj": {"ffn_gate", "ffn_gate_exps", "ffn_gate_shexp", "ffn_gate_up", "ffn_gate_up_exps"},
    "up_proj": {"ffn_up", "ffn_up_exps", "ffn_up_shexp", "ffn_gate_up", "ffn_gate_up_exps"},
    "down_proj": {"ffn_down", "ffn_down_exps", "ffn_down_shexp"},
}
DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class Tensor:
    """One weight tensor (or, for MoE, one per-layer stack of expert matrices).

    ``n_in`` is the contiguous row length (ggml ``ne[0]`` / torch ``shape[-1]``),
    ``n_out`` the number of rows per expert. ``numel == n_in * n_out * experts``
    for 2-D tensors; 1-D tensors have ``n_out == 1``.
    """

    name: str
    kind: str
    layer: Optional[int]
    numel: int
    n_in: int
    n_out: int = 1
    experts: int = 1
    dtype: Optional[str] = None          # stored dtype if known from a real header
    nbytes_exact: Optional[int] = None   # exact stored byte size if known
    tied: bool = False                   # output head that aliases token embeddings

    @property
    def is_linear(self) -> bool:
        return self.kind in LINEAR_KINDS

    @property
    def is_expert(self) -> bool:
        return self.kind in EXPERT_KINDS

    @property
    def is_1d(self) -> bool:
        return self.n_out <= 1 and self.experts <= 1


def classify_gguf_name(name: str):
    """Map a GGUF tensor name to (kind, layer)."""
    layer = None
    base = name
    if name.startswith("blk."):
        parts = name.split(".")
        try:
            layer = int(parts[1])
        except (IndexError, ValueError):
            layer = None
        base = ".".join(parts[2:])
    if base.endswith(".bias"):
        return ("bias", layer)
    stem = base[:-7] if base.endswith(".weight") else base
    if layer is None:
        if stem == "token_embd":
            return ("embed", None)
        if stem == "output":
            return ("head", None)
        if stem.endswith("norm"):
            return ("norm", None)
        return ("other", None)
    if "norm" in stem:
        return ("norm", layer)
    mapping = {
        "attn_q": "attn_q", "attn_k": "attn_k", "attn_v": "attn_v", "attn_output": "attn_o",
        "attn_qkv": "attn_qkv", "attn_q_a": "attn_q_a", "attn_q_b": "attn_q_b",
        "attn_kv_a_mqa": "attn_kv_a", "attn_kv_b": "attn_kv_b", "attn_k_b": "attn_k_b",
        "attn_v_b": "attn_v_b",
        "ffn_gate": "ffn_gate", "ffn_up": "ffn_up", "ffn_down": "ffn_down",
        "ffn_gate_up": "ffn_gate_up",
        "ffn_gate_exps": "ffn_gate_exps", "ffn_up_exps": "ffn_up_exps",
        "ffn_down_exps": "ffn_down_exps", "ffn_gate_up_exps": "ffn_gate_up_exps",
        "ffn_gate_shexp": "ffn_gate_shexp", "ffn_up_shexp": "ffn_up_shexp",
        "ffn_down_shexp": "ffn_down_shexp",
        "ffn_gate_inp": "router", "ffn_gate_inp_shexp": "router",
        "ssm_in": "ssm_in", "ssm_out": "ssm_out",
    }
    if stem in mapping:
        return (mapping[stem], layer)
    if stem.startswith("ssm_"):
        return ("ssm_other", layer)
    if stem.startswith("exp_probs"):
        return ("bias", layer)
    return ("other", layer)


def classify_hf_name(name: str):
    """Map a Hugging Face safetensors tensor name to (kind, layer, expert_index)."""
    lname = name.lower()
    if any(v in lname for v in ("vision", "visual", "mm_projector", "multi_modal_projector",
                                "image_newline", "audio_tower", "audio_model")):
        return ("vision", None, None)
    layer = None
    expert = None
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p in ("layers", "h", "blocks", "layer") and i + 1 < len(parts) and parts[i + 1].isdigit():
            layer = int(parts[i + 1])
        if p == "experts" and i + 1 < len(parts) and parts[i + 1].isdigit():
            expert = int(parts[i + 1])
    is_bias = lname.endswith(".bias")
    if is_bias:
        return ("bias", layer, None)
    if layer is None:
        if "embed_tokens" in lname or "wte" in lname or "word_embeddings" in lname or lname.endswith("embed_in.weight"):
            return ("embed", None, None)
        if "lm_head" in lname or "embed_out" in lname:
            return ("head", None, None)
        if "norm" in lname or "ln_f" in lname:
            return ("norm", None, None)
        return ("other", None, None)
    if "norm" in lname or ".ln_" in lname:
        return ("norm", layer, None)
    pairs = [
        ("q_a_proj", "attn_q_a"), ("q_b_proj", "attn_q_b"), ("kv_a_proj", "attn_kv_a"),
        ("kv_b_proj", "attn_kv_b"), ("qkv_proj", "attn_qkv"), ("query_key_value", "attn_qkv"),
        ("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"),
        ("o_proj", "attn_o"), ("out_proj", "attn_o"), ("dense.weight", "attn_o"),
        ("gate_up_proj", "ffn_gate_up"), ("gate_proj", "ffn_gate"), ("w1", "ffn_gate"),
        ("up_proj", "ffn_up"), ("w3", "ffn_up"), ("down_proj", "ffn_down"), ("w2", "ffn_down"),
        ("dense_h_to_4h", "ffn_up"), ("dense_4h_to_h", "ffn_down"), ("fc1", "ffn_up"), ("fc2", "ffn_down"),
    ]
    for key, kind in pairs:
        if key in lname:
            if kind.startswith("ffn_"):
                if expert is not None or (".experts." in lname):
                    return (kind + "_exps", layer, expert)
                if "shared_expert" in lname:
                    return (kind + "_shexp", layer, None)
            return (kind, layer, None)
    if lname.endswith("mlp.gate.weight") or "router" in lname or "block_sparse_moe.gate" in lname \
            or "shared_expert_gate" in lname:
        return ("router", layer, None)
    return ("other", layer, None)
