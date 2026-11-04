"""Model architecture specs and tensor inventories.

A :class:`LoadedModel` is built from one of three tiny metadata sources —
``config.json``, a GGUF header, or safetensors headers — and never from the
weights themselves. It carries:

* :class:`ModelSpec` – the hyper-parameters the KV-cache / activation math needs
  (GQA/MQA head counts, MLA ranks, sliding-window layout, MoE routing, ...)
* an inventory of every weight tensor (shape, kind, layer), which the quantizer
  turns into exact byte counts and the placer splits between GPU and CPU.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .gguf import GGUFHeader
from .quant import quant_from_hf_config
from .tensors import EXPERT_KINDS, Tensor, classify_gguf_name, classify_hf_name

NON_GATED_TYPES = {
    "gpt2", "gpt_neox", "gptj", "falcon", "phi", "opt", "bloom", "gpt_bigcode", "starcoder2",
    "codegen", "mpt", "stablelm_epoch",
}
QKV_BIAS_TYPES = {"qwen2", "qwen2_moe", "qwen2_vl", "qwen2_5_vl"}
QK_NORM_TYPES = {"qwen3", "qwen3_moe", "gemma3", "gemma3_text", "olmo2", "qwen3_next"}
TIED_DEFAULT_TYPES = {"gemma", "gemma2", "gemma3", "gemma3_text", "gpt2", "opt"}


@dataclass
class ModelSpec:
    id: str
    architecture: str
    model_type: str
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: List[int]
    head_dim: int
    v_head_dim: int
    intermediate_size: int
    vocab_size: int
    max_context: int
    rope_theta: Optional[float] = None
    rope_scaling: Optional[Dict[str, Any]] = None
    tie_embeddings: bool = False
    gated_mlp: bool = True
    qkv_bias: bool = False
    qk_norm: bool = False
    num_experts: int = 0
    experts_per_token: int = 0
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0
    moe_layers: List[bool] = field(default_factory=list)
    kv_lora_rank: int = 0
    q_lora_rank: int = 0
    qk_rope_head_dim: int = 0
    qk_nope_head_dim: int = 0
    sliding_window: Optional[int] = None
    layer_types: List[str] = field(default_factory=list)   # "full" | "sliding" | "linear"
    linear_state_bytes: int = 0       # recurrent state per linear layer per sequence
    torch_dtype: str = "bfloat16"
    quantization: Optional[Dict[str, Any]] = None
    vision_params: int = 0

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 1

    @property
    def is_mla(self) -> bool:
        return self.kv_lora_rank > 0

    @property
    def gqa_ratio(self) -> int:
        kv = max(1, max(self.num_kv_heads) if self.num_kv_heads else 1)
        return max(1, self.num_heads // kv)

    def kv_heads(self, layer: int) -> int:
        return self.num_kv_heads[layer] if layer < len(self.num_kv_heads) else self.num_kv_heads[-1]

    def kv_elems_per_token(self, layer: int) -> int:
        """K+V cache elements stored per token for one layer."""
        lt = self.layer_types[layer] if layer < len(self.layer_types) else "full"
        if lt == "linear":
            return 0
        if self.is_mla:
            return self.kv_lora_rank + self.qk_rope_head_dim
        n = self.kv_heads(layer)
        return n * self.head_dim + n * self.v_head_dim

    def attention_kind(self) -> str:
        if self.is_mla:
            return "MLA"
        kv = max(self.num_kv_heads) if self.num_kv_heads else self.num_heads
        if kv == 1:
            return "MQA"
        if kv < self.num_heads:
            return f"GQA {self.num_heads}:{kv}"
        return "MHA"

    def to_dict(self) -> dict:
        d = asdict(self)
        kvs = set(self.num_kv_heads)
        d["num_kv_heads"] = self.num_kv_heads[0] if len(kvs) == 1 else self.num_kv_heads
        lts = self.layer_types
        d["layer_types"] = {t: lts.count(t) for t in sorted(set(lts))}
        d["moe_layers"] = sum(1 for x in self.moe_layers if x)
        d["attention"] = self.attention_kind()
        return d


@dataclass
class LoadedModel:
    spec: ModelSpec
    inventory: List[Tensor]
    source: str                          # hf-config | gguf | safetensors | catalog | local-config
    native_quant: Optional[str] = None   # quant whose dtypes the inventory carries exactly
    exact: bool = False
    repo: Optional[str] = None
    filename: Optional[str] = None
    path: Optional[str] = None           # local file / directory the model was loaded from
    gguf_files: List[Dict[str, Any]] = field(default_factory=list)
    params_reported: Optional[int] = None
    gated: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def params_total(self) -> int:
        return sum(t.numel for t in self.inventory if not t.tied)

    @property
    def params_active(self) -> int:
        s = self.spec
        total = 0
        for t in self.inventory:
            if t.tied:
                continue
            if t.kind in EXPERT_KINDS and s.num_experts > 0:
                total += t.numel * s.experts_per_token // max(1, t.experts)
            else:
                total += t.numel
        return total

    def to_dict(self) -> dict:
        return {
            "id": self.spec.id,
            "repo": self.repo,
            "filename": self.filename,
            "path": self.path,
            "source": self.source,
            "native_quant": self.native_quant,
            "exact_tensor_inventory": self.exact,
            "params_total": self.params_total,
            "params_active": self.params_active,
            "params_reported": self.params_reported,
            "gated": self.gated,
            "spec": self.spec.to_dict(),
            "gguf_files": self.gguf_files,
            "tensor_count": len(self.inventory),
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# config.json -> spec
# ---------------------------------------------------------------------------

def _first(d: Dict[str, Any], *keys, default=None):
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return default


def _merged_text_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("text_config", "llm_config", "language_config", "text_model"):
        sub = cfg.get(key)
        if isinstance(sub, dict) and (sub.get("hidden_size") or sub.get("num_hidden_layers")):
            merged = {k: v for k, v in cfg.items() if not isinstance(v, dict) or k in ("rope_scaling", "quantization_config")}
            merged.update(sub)
            if "quantization_config" in cfg and "quantization_config" not in sub:
                merged["quantization_config"] = cfg["quantization_config"]
            return merged
    return cfg


def _vision_params(cfg: Dict[str, Any], text_hidden: int) -> int:
    vc = cfg.get("vision_config")
    if not isinstance(vc, dict):
        return 0
    h = _first(vc, "hidden_size", "embed_dim", "width", default=0) or 0
    layers = _first(vc, "num_hidden_layers", "depth", "layers", default=0) or 0
    inter = _first(vc, "intermediate_size", default=None)
    if inter is None:
        inter = int(h * float(vc.get("mlp_ratio", 4)))
    patch = vc.get("patch_size", 14) or 14
    if isinstance(patch, (list, tuple)):
        patch = patch[0]
    image = vc.get("image_size", 224) or 224
    if isinstance(image, (list, tuple)):
        image = image[0]
    chans = vc.get("num_channels", 3) or 3
    per_layer = 4 * h * h + 2 * h * inter + 4 * h
    pos = (image // patch) ** 2 * h
    patch_embed = chans * patch * patch * h
    projector = h * text_hidden + text_hidden
    return int(layers * per_layer + pos + patch_embed + projector)


def _state_bytes(c: Dict[str, Any], hidden: int) -> int:
    if c.get("linear_num_value_heads"):
        nv = int(c["linear_num_value_heads"])
        nk = int(c.get("linear_num_key_heads", nv))
        dk = int(c.get("linear_key_head_dim", 128))
        dv = int(c.get("linear_value_head_dim", 128))
        kern = int(c.get("linear_conv_kernel_dim", 4))
        return nv * dk * dv * 4 + (2 * nk * dk + nv * dv) * (kern - 1) * 2
    d_state = _first(c, "mamba_d_state", "ssm_state_size", "state_size", "mamba_dstate")
    if d_state:
        expand = _first(c, "mamba_expand", "expand", default=2)
        d_inner = _first(c, "mamba_d_inner", "intermediate_size_mamba", default=int(expand * hidden))
        ngroups = _first(c, "mamba_n_groups", "n_groups", default=1)
        d_conv = _first(c, "mamba_d_conv", "conv_kernel", default=4)
        return int(d_inner) * int(d_state) * 4 + (int(d_inner) + 2 * int(ngroups) * int(d_state)) * (int(d_conv) - 1) * 2
    return 0


def spec_from_hf_config(cfg: Dict[str, Any], model_id: str) -> ModelSpec:
    c = _merged_text_config(cfg)
    model_type = str(c.get("model_type") or cfg.get("model_type") or "unknown")
    archs = cfg.get("architectures") or c.get("architectures") or [model_type]
    hidden = int(_first(c, "hidden_size", "n_embd", "d_model", "dim", "model_dim", default=0))
    layers = int(_first(c, "num_hidden_layers", "n_layer", "num_layers", "n_layers", default=0))
    heads = int(_first(c, "num_attention_heads", "n_head", "num_heads", "n_heads", default=0))
    if not hidden or not layers or not heads:
        raise ValueError(f"config for {model_id} lacks hidden_size / num_hidden_layers / num_attention_heads")
    kv = _first(c, "num_key_value_heads", "num_kv_heads", "n_head_kv", "multi_query_group_num", "n_kv_heads")
    if kv is None:
        kv = 1 if c.get("multi_query") and not c.get("new_decoder_architecture") else heads
    head_dim = int(_first(c, "head_dim", "kv_channels", default=0) or hidden // heads)
    v_head_dim = int(c.get("v_head_dim") or head_dim)
    inter = int(_first(c, "intermediate_size", "ffn_dim", "n_inner", "ffn_hidden_size", default=0) or 4 * hidden)
    vocab = int(_first(c, "vocab_size", "padded_vocab_size", default=32000))
    max_ctx = int(_first(c, "max_position_embeddings", "n_positions", "max_sequence_length",
                         "seq_length", "n_ctx", default=4096))
    tie = c.get("tie_word_embeddings")
    if tie is None:
        tie = cfg.get("tie_word_embeddings")
    if tie is None:
        tie = model_type in TIED_DEFAULT_TYPES
    experts = int(_first(c, "num_local_experts", "num_experts", "n_routed_experts", "moe_num_experts", default=0) or 0)
    top_k = int(_first(c, "num_experts_per_tok", "moe_top_k", "num_selected_experts", "top_k", default=0) or 0)
    moe_inter = int(_first(c, "moe_intermediate_size", "expert_intermediate_size", default=0) or inter)
    shared = 0
    if c.get("n_shared_experts"):
        shared = int(c["n_shared_experts"]) * moe_inter
    elif c.get("shared_expert_intermediate_size"):
        shared = int(c["shared_expert_intermediate_size"])
    first_dense = int(c.get("first_k_dense_replace") or 0)
    freq = int(c.get("moe_layer_freq") or 1)
    sparse_step = int(c.get("decoder_sparse_step") or 1)
    mlp_only = set(c.get("mlp_only_layers") or [])
    moe_layers = []
    for i in range(layers):
        is_moe = experts > 1 and i >= first_dense and i not in mlp_only
        if is_moe and freq > 1:
            is_moe = (i - first_dense) % freq == 0
        if is_moe and sparse_step > 1:
            is_moe = (i + 1) % sparse_step == 0
        moe_layers.append(is_moe)

    window = c.get("sliding_window")
    lt_raw = c.get("layer_types")
    layer_types: List[str] = []
    if isinstance(lt_raw, list) and len(lt_raw) == layers:
        for t in lt_raw:
            t = str(t).lower()
            if "sliding" in t or t in ("local", "chunked_attention"):
                layer_types.append("sliding")
            elif "linear" in t or "mamba" in t or "ssm" in t or "recurrent" in t or "conv" in t:
                layer_types.append("linear")
            else:
                layer_types.append("full")
    elif model_type == "gemma2" and window:
        layer_types = ["sliding" if i % 2 == 0 else "full" for i in range(layers)]
    elif c.get("sliding_window_pattern") and window:
        p = int(c["sliding_window_pattern"])
        layer_types = ["full" if (i + 1) % p == 0 else "sliding" for i in range(layers)]
    elif model_type in ("gemma3", "gemma3_text") and window:
        layer_types = ["full" if (i + 1) % 6 == 0 else "sliding" for i in range(layers)]
    elif window and c.get("use_sliding_window", True) is not False and model_type in ("mistral", "mixtral", "qwen2", "starcoder2", "phi3"):
        if window < max_ctx:
            layer_types = ["sliding"] * layers
    elif c.get("attn_layer_period"):  # Jamba
        period = int(c["attn_layer_period"])
        off = int(c.get("attn_layer_offset", 0))
        layer_types = ["full" if i % period == off else "linear" for i in range(layers)]
    elif c.get("full_attention_interval"):
        iv = int(c["full_attention_interval"])
        layer_types = ["full" if (i + 1) % iv == 0 else "linear" for i in range(layers)]
    if not layer_types:
        layer_types = ["full"] * layers
    if "sliding" not in layer_types:
        window = None

    kv_list = [int(kv)] * layers if not isinstance(kv, list) else [int(x) for x in kv]
    dtype = str(_first(c, "torch_dtype", "dtype", default=cfg.get("torch_dtype") or "bfloat16"))
    return ModelSpec(
        id=model_id,
        architecture=str(archs[0]),
        model_type=model_type,
        hidden_size=hidden,
        num_layers=layers,
        num_heads=heads,
        num_kv_heads=kv_list,
        head_dim=int(c.get("qk_nope_head_dim", 0) + c.get("qk_rope_head_dim", 0)) if c.get("kv_lora_rank") else head_dim,
        v_head_dim=v_head_dim,
        intermediate_size=inter,
        vocab_size=vocab,
        max_context=max_ctx,
        rope_theta=_first(c, "rope_theta", "rotary_emb_base"),
        rope_scaling=c.get("rope_scaling") if isinstance(c.get("rope_scaling"), dict) else None,
        tie_embeddings=bool(tie),
        gated_mlp=model_type not in NON_GATED_TYPES,
        qkv_bias=bool(c.get("attention_bias") or c.get("qkv_bias") or model_type in QKV_BIAS_TYPES),
        qk_norm=model_type in QK_NORM_TYPES,
        num_experts=experts if experts > 1 else 0,
        experts_per_token=top_k if experts > 1 else 0,
        moe_intermediate_size=moe_inter if experts > 1 else 0,
        shared_expert_intermediate_size=shared,
        moe_layers=moe_layers,
        kv_lora_rank=int(c.get("kv_lora_rank") or 0),
        q_lora_rank=int(c.get("q_lora_rank") or 0),
        qk_rope_head_dim=int(c.get("qk_rope_head_dim") or 0),
        qk_nope_head_dim=int(c.get("qk_nope_head_dim") or 0),
        sliding_window=int(window) if window else None,
        layer_types=layer_types,
        linear_state_bytes=_state_bytes(c, hidden) if "linear" in layer_types else 0,
        torch_dtype=dtype.replace("torch.", ""),
        quantization=cfg.get("quantization_config") or c.get("quantization_config"),
        vision_params=_vision_params(cfg, hidden),
    )


# ---------------------------------------------------------------------------
# spec -> synthesized tensor inventory
# ---------------------------------------------------------------------------

def _lin(name: str, kind: str, layer: Optional[int], n_in: int, n_out: int, experts: int = 1) -> Tensor:
    return Tensor(name=name, kind=kind, layer=layer, numel=n_in * n_out * experts,
                  n_in=n_in, n_out=n_out, experts=experts)


def _vec(name: str, layer: Optional[int], n: int, kind: str = "norm") -> Tensor:
    return Tensor(name=name, kind=kind, layer=layer, numel=n, n_in=n, n_out=1)


def synthesize_inventory(s: ModelSpec) -> List[Tensor]:
    H = s.hidden_size
    inv: List[Tensor] = [_lin("token_embd.weight", "embed", None, H, s.vocab_size)]
    gemma_norms = s.model_type.startswith("gemma2") or s.model_type.startswith("gemma3")
    for i in range(s.num_layers):
        p = f"blk.{i}."
        inv.append(_vec(p + "attn_norm.weight", i, H))
        if s.is_mla:
            qk_dim = s.qk_nope_head_dim + s.qk_rope_head_dim
            if s.q_lora_rank:
                inv.append(_lin(p + "attn_q_a.weight", "attn_q_a", i, H, s.q_lora_rank))
                inv.append(_vec(p + "attn_q_a_norm.weight", i, s.q_lora_rank))
                inv.append(_lin(p + "attn_q_b.weight", "attn_q_b", i, s.q_lora_rank, s.num_heads * qk_dim))
            else:
                inv.append(_lin(p + "attn_q.weight", "attn_q", i, H, s.num_heads * qk_dim))
            inv.append(_lin(p + "attn_kv_a_mqa.weight", "attn_kv_a", i, H, s.kv_lora_rank + s.qk_rope_head_dim))
            inv.append(_vec(p + "attn_kv_a_norm.weight", i, s.kv_lora_rank))
            inv.append(_lin(p + "attn_kv_b.weight", "attn_kv_b", i, s.kv_lora_rank,
                            s.num_heads * (s.qk_nope_head_dim + s.v_head_dim)))
            inv.append(_lin(p + "attn_output.weight", "attn_o", i, s.num_heads * s.v_head_dim, H))
        else:
            nkv = s.kv_heads(i) or s.num_heads
            q_out = s.num_heads * s.head_dim
            kv_out = nkv * s.head_dim
            v_out = nkv * s.v_head_dim
            inv.append(_lin(p + "attn_q.weight", "attn_q", i, H, q_out))
            inv.append(_lin(p + "attn_k.weight", "attn_k", i, H, kv_out))
            inv.append(_lin(p + "attn_v.weight", "attn_v", i, H, v_out))
            inv.append(_lin(p + "attn_output.weight", "attn_o", i, s.num_heads * s.v_head_dim, H))
            if s.qkv_bias:
                inv.append(_vec(p + "attn_q.bias", i, q_out, "bias"))
                inv.append(_vec(p + "attn_k.bias", i, kv_out, "bias"))
                inv.append(_vec(p + "attn_v.bias", i, v_out, "bias"))
            if s.qk_norm:
                inv.append(_vec(p + "attn_q_norm.weight", i, s.head_dim))
                inv.append(_vec(p + "attn_k_norm.weight", i, s.head_dim))
        inv.append(_vec(p + "ffn_norm.weight", i, H))
        if gemma_norms:
            inv.append(_vec(p + "post_attention_norm.weight", i, H))
            inv.append(_vec(p + "post_ffw_norm.weight", i, H))
        is_moe = i < len(s.moe_layers) and s.moe_layers[i]
        if is_moe:
            E, I = s.num_experts, s.moe_intermediate_size
            inv.append(_lin(p + "ffn_gate_inp.weight", "router", i, H, E))
            inv.append(_lin(p + "ffn_gate_exps.weight", "ffn_gate_exps", i, H, I, E))
            inv.append(_lin(p + "ffn_up_exps.weight", "ffn_up_exps", i, H, I, E))
            inv.append(_lin(p + "ffn_down_exps.weight", "ffn_down_exps", i, I, H, E))
            if s.shared_expert_intermediate_size:
                Is = s.shared_expert_intermediate_size
                inv.append(_lin(p + "ffn_gate_shexp.weight", "ffn_gate_shexp", i, H, Is))
                inv.append(_lin(p + "ffn_up_shexp.weight", "ffn_up_shexp", i, H, Is))
                inv.append(_lin(p + "ffn_down_shexp.weight", "ffn_down_shexp", i, Is, H))
                if s.model_type.startswith("qwen2_moe") or s.model_type.startswith("qwen3_next"):
                    inv.append(_lin(p + "ffn_gate_inp_shexp.weight", "router", i, H, 1))
        else:
            I = s.intermediate_size
            if s.gated_mlp:
                inv.append(_lin(p + "ffn_gate.weight", "ffn_gate", i, H, I))
            inv.append(_lin(p + "ffn_up.weight", "ffn_up", i, H, I))
            inv.append(_lin(p + "ffn_down.weight", "ffn_down", i, I, H))
    inv.append(_vec("output_norm.weight", None, H))
    head = _lin("output.weight", "head", None, H, s.vocab_size)
    head.tied = s.tie_embeddings
    inv.append(head)
    if s.vision_params:
        inv.append(Tensor(name="vision_tower", kind="vision", layer=None, numel=s.vision_params,
                          n_in=s.vision_params, n_out=1))
    return inv


def reconcile_with_reported(model: LoadedModel) -> None:
    """Fold an API-reported parameter count into the inventory.

    Unmapped extras (vision/audio towers, MTP heads...) become an ``other``
    tensor so they are still counted (kept 16-bit on the GPU).
    """
    rep = model.params_reported
    if not rep or model.exact or model.spec.quantization:
        return
    have = model.params_total
    diff = rep - have
    if diff > 0.02 * rep:
        vis = next((t for t in model.inventory if t.kind == "vision"), None)
        if vis is not None:
            vis.numel += diff
            vis.n_in = vis.numel
        else:
            model.inventory.append(Tensor(name="unmapped_extra", kind="other", layer=None,
                                          numel=diff, n_in=diff, n_out=1))
        model.notes.append(
            f"Hub reports {rep/1e9:.2f}B params; config math gives {have/1e9:.2f}B. "
            f"Counting the extra {diff/1e9:.2f}B (vision/audio/MTP tensors) as 16-bit.")
    elif diff < -0.03 * rep:
        model.notes.append(
            f"Config math gives {have/1e9:.2f}B params but the Hub reports {rep/1e9:.2f}B; "
            "the architecture may use a layout vram-fit approximates.")


def model_from_hf_config(cfg: Dict[str, Any], model_id: str, source: str = "hf-config",
                         params_reported: Optional[int] = None) -> LoadedModel:
    spec = spec_from_hf_config(cfg, model_id)
    inv = synthesize_inventory(spec)
    lm = LoadedModel(spec=spec, inventory=inv, source=source,
                     native_quant=quant_from_hf_config(spec.quantization),
                     params_reported=params_reported)
    reconcile_with_reported(lm)
    return lm


# ---------------------------------------------------------------------------
# GGUF header -> spec + exact inventory
# ---------------------------------------------------------------------------

def _as_list(v, n: int) -> List[int]:
    if isinstance(v, list):
        return [int(x) for x in v][:n] + [int(v[-1])] * max(0, n - len(v))
    return [int(v)] * n


def model_from_gguf(headers: List[GGUFHeader], model_id: str, filename: Optional[str] = None) -> LoadedModel:
    h0 = headers[0]
    a = h0.architecture
    g = h0.arch_get
    layers = int(g("block_count", 0))
    hidden = int(g("embedding_length", 0))
    heads_v = g("attention.head_count", 0)
    heads_l = _as_list(heads_v, max(layers, 1))
    heads = max(heads_l) if heads_l else 0
    kv_v = g("attention.head_count_kv", heads_v)
    kv_l = _as_list(kv_v, max(layers, 1))
    key_len = int(g("attention.key_length", 0) or (hidden // heads if heads else 0))
    val_len = int(g("attention.value_length", 0) or key_len)
    ff = g("feed_forward_length", 4 * hidden)
    inter = max(_as_list(ff, 1)) if isinstance(ff, list) else int(ff)
    tensors_all = [t for h in headers for t in h.tensors]
    vocab = h0.array_len("tokenizer.ggml.tokens") or int(g("vocab_size", 0) or 0)
    emb = next((t for t in tensors_all if t.name == "token_embd.weight"), None)
    if emb is not None and len(emb.shape) >= 2:
        vocab = int(emb.shape[1])
        hidden = hidden or int(emb.shape[0])
    if not (layers and hidden and heads):
        raise ValueError(f"GGUF header for {model_id} lacks block_count / embedding_length / head_count")
    experts = int(g("expert_count", 0) or 0)
    top_k = int(g("expert_used_count", 0) or 0)
    moe_inter = int(g("expert_feed_forward_length", 0) or 0)
    n_shared = int(g("expert_shared_count", 0) or 0)
    shared_ff = int(g("expert_shared_feed_forward_length", 0) or 0)
    lead_dense = int(g("leading_dense_block_count", 0) or 0)
    kv_lora = int(g("attention.kv_lora_rank", 0) or 0)
    q_lora = int(g("attention.q_lora_rank", 0) or 0)
    rope_dim = int(g("rope.dimension_count", 0) or 0)
    window = g("attention.sliding_window")

    inv: List[Tensor] = []
    has_output = False
    exp_layers = set()
    for t in tensors_all:
        kind, layer = classify_gguf_name(t.name)
        shape = [int(x) for x in t.shape] or [1]
        n_in = shape[0]
        n_out = shape[1] if len(shape) > 1 else 1
        n_exp = shape[2] if len(shape) > 2 and kind in EXPERT_KINDS else 1
        if len(shape) > 2 and kind not in EXPERT_KINDS:
            n_out = int(t.numel // n_in)
        if kind in EXPERT_KINDS and layer is not None:
            exp_layers.add(layer)
        if kind == "head":
            has_output = True
        inv.append(Tensor(name=t.name, kind=kind, layer=layer, numel=t.numel, n_in=n_in, n_out=n_out,
                          experts=n_exp, dtype=t.type_name, nbytes_exact=t.nbytes))
    if not has_output and emb is not None:
        inv.append(Tensor(name="output.weight", kind="head", layer=None, numel=emb.numel,
                          n_in=int(emb.shape[0]), n_out=int(emb.shape[1]) if len(emb.shape) > 1 else 1,
                          dtype=emb.type_name, nbytes_exact=emb.nbytes, tied=True))

    layer_types = []
    pattern = g("attention.sliding_window_pattern")
    swa_layers = g("attention.sliding_window_pattern") if isinstance(pattern, list) else None
    for i in range(layers):
        if kv_l[i] == 0 and heads_l[i] == 0 or (kv_l[i] == 0 and a in ("jamba", "granitehybrid", "falcon-h1", "nemotron_h", "lfm2", "qwen3next")):
            layer_types.append("linear")
        elif window and swa_layers is not None:
            layer_types.append("sliding" if swa_layers[i] else "full")
        elif window and a == "gemma2":
            layer_types.append("sliding" if i % 2 == 0 else "full")
        elif window and a in ("gemma3", "gemma3n"):
            p = int(pattern) if isinstance(pattern, int) and pattern else 6
            layer_types.append("full" if (i + 1) % p == 0 else "sliding")
        elif window and isinstance(pattern, int) and pattern > 0:
            layer_types.append("full" if (i + 1) % pattern == 0 else "sliding")
        elif window and a == "gpt-oss":
            layer_types.append("sliding" if i % 2 == 0 else "full")
        else:
            layer_types.append("full")
    if a == "qwen3next":
        iv = int(g("full_attention_interval", 4) or 4)
        layer_types = ["full" if (i + 1) % iv == 0 else "linear" for i in range(layers)]
    if "sliding" not in layer_types:
        window = None
    state = 0
    if "linear" in layer_types:
        d_state = int(g("ssm.state_size", 0) or 0)
        d_inner = int(g("ssm.inner_size", 0) or 0)
        d_conv = int(g("ssm.conv_kernel", 4) or 4)
        groups = int(g("ssm.group_count", 1) or 1)
        if d_state and d_inner:
            state = d_inner * d_state * 4 + (d_inner + 2 * groups * d_state) * (d_conv - 1) * 2

    max_ctx = int(g("context_length", 4096) or 4096)
    mla = kv_lora > 0
    spec = ModelSpec(
        id=model_id,
        architecture=a,
        model_type=a,
        hidden_size=hidden,
        num_layers=layers,
        num_heads=heads,
        num_kv_heads=[k if k else 0 for k in kv_l],
        head_dim=int(g("attention.key_length_mla", 0) or key_len) if mla else key_len,
        v_head_dim=int(g("attention.value_length_mla", 0) or val_len) if mla else val_len,
        intermediate_size=inter,
        vocab_size=vocab,
        max_context=max_ctx,
        rope_theta=g("rope.freq_base"),
        rope_scaling=({"type": g("rope.scaling.type"), "factor": g("rope.scaling.factor"),
                       "original_max_position_embeddings": g("rope.scaling.original_context_length")}
                      if g("rope.scaling.type") not in (None, "none") else None),
        tie_embeddings=not has_output,
        gated_mlp=any(t.kind in ("ffn_gate", "ffn_gate_exps", "ffn_gate_up", "ffn_gate_up_exps") for t in inv),
        qkv_bias=any(t.kind == "bias" and "attn_q" in t.name for t in inv),
        qk_norm=any("attn_q_norm" in t.name for t in inv),
        num_experts=experts if experts > 1 else 0,
        experts_per_token=top_k if experts > 1 else 0,
        moe_intermediate_size=moe_inter,
        shared_expert_intermediate_size=n_shared * (shared_ff or moe_inter),
        moe_layers=[i in exp_layers for i in range(layers)] if exp_layers else
                   [experts > 1 and i >= lead_dense for i in range(layers)],
        kv_lora_rank=kv_lora,
        q_lora_rank=q_lora,
        qk_rope_head_dim=rope_dim if mla else 0,
        qk_nope_head_dim=(key_len - rope_dim) if mla else 0,
        sliding_window=int(window) if window else None,
        layer_types=layer_types,
        linear_state_bytes=state,
        torch_dtype="bfloat16",
    )
    lm = LoadedModel(spec=spec, inventory=inv, source="gguf", native_quant=h0.file_type or _guess_gguf_quant(inv),
                     exact=True, filename=filename)
    if h0.split_count > len(headers):
        lm.notes.append(f"Only {len(headers)} of {h0.split_count} GGUF splits were read; sizes are partial.")
    return lm


def _guess_gguf_quant(inv: List[Tensor]) -> Optional[str]:
    counts: Dict[str, int] = {}
    for t in inv:
        if t.is_linear and t.dtype:
            counts[t.dtype] = counts.get(t.dtype, 0) + t.numel
    if not counts:
        return None
    top = max(counts, key=counts.get)
    return {"Q4_K": "Q4_K_M", "Q5_K": "Q5_K_M", "Q3_K": "Q3_K_M"}.get(top, top)


# ---------------------------------------------------------------------------
# safetensors headers -> exact inventory
# ---------------------------------------------------------------------------

SAFETENSORS_DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "U32": 4, "I16": 2, "U16": 2,
    "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "F4": 0.5,
}


def parse_safetensors_header(raw: bytes) -> Dict[str, Any]:
    if len(raw) < 8:
        raise ValueError("safetensors file too short")
    n = int.from_bytes(raw[:8], "little")
    if n > 100 * 1024 * 1024:
        raise ValueError("safetensors header length is implausible")
    if len(raw) < 8 + n:
        raise ValueError("incomplete safetensors header")
    return json.loads(raw[8: 8 + n].decode("utf-8"))


def inventory_from_safetensors(headers: List[Dict[str, Any]]) -> List[Tensor]:
    grouped: Dict[tuple, Tensor] = {}
    experts_seen: Dict[tuple, set] = {}
    out: List[Tensor] = []
    for hdr in headers:
        for name, info in hdr.items():
            if name == "__metadata__" or not isinstance(info, dict):
                continue
            shape = [int(x) for x in info.get("shape", [])] or [1]
            numel = math.prod(shape)
            dtype = str(info.get("dtype", "BF16"))
            off = info.get("data_offsets") or [0, 0]
            nbytes = int(off[1]) - int(off[0]) if len(off) == 2 else int(numel * SAFETENSORS_DTYPE_BYTES.get(dtype, 2))
            kind, layer, expert = classify_hf_name(name)
            n_in = shape[-1]
            n_out = shape[-2] if len(shape) >= 2 else 1
            if kind in EXPERT_KINDS:
                key = (layer, kind)
                if len(shape) == 3:
                    n_exp = shape[0]
                    experts_seen.setdefault(key, set()).update(range(n_exp))
                else:
                    experts_seen.setdefault(key, set()).add(expert if expert is not None else len(experts_seen.get(key, ())))
                if key in grouped:
                    g = grouped[key]
                    g.numel += numel
                    g.nbytes_exact = (g.nbytes_exact or 0) + nbytes
                else:
                    g = Tensor(name=f"blk.{layer}.{kind}", kind=kind, layer=layer, numel=numel, n_in=n_in,
                               n_out=n_out, dtype=dtype, nbytes_exact=nbytes)
                    grouped[key] = g
                    out.append(g)
                continue
            out.append(Tensor(name=name, kind=kind, layer=layer, numel=numel, n_in=n_in,
                              n_out=n_out if len(shape) >= 2 else 1, dtype=dtype, nbytes_exact=nbytes))
    for key, t in grouped.items():
        t.experts = max(1, len(experts_seen.get(key, ())))
        t.n_out = max(1, t.numel // max(1, t.n_in * t.experts))
    return out
