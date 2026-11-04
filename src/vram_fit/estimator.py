"""The VRAM / throughput engine.

``estimate(model, gpu, settings)`` returns a JSON-serialisable dict with a
megabyte-level breakdown into five segments — Model Weights, KV Cache,
Activation Buffers, CUDA/Driver Overhead, LoRA/Optimizer States — plus what
spills to system RAM, a fit status, and bandwidth-derived tokens/s.

All formulas are documented inline; numbers are MiB (2**20 bytes) unless a
key says otherwise.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List, Optional, Tuple

from .ggml import KV_CACHE_TYPES
from .hardware import DEFAULT_RAM_BW_GBS, HardwareProfile
from .model import LoadedModel
from .presets import GPU
from .quant import QuantScheme, default_quant, get_quant, make_quant_context
from .tensors import ATTN_KINDS, DEFAULT_LORA_TARGETS, EXPERT_KINDS, LORA_TARGET_KINDS, SHARED_EXPERT_KINDS

MiB = 1024 * 1024
GiB = 1024 * MiB

ENGINES = ("llama.cpp", "ollama", "vllm", "exllamav2", "unsloth", "transformers")
GGUF_ENGINES = ("llama.cpp", "ollama")
TRAIN_ENGINES = ("unsloth", "transformers")
MODES = ("inference", "lora", "qlora", "full")
OPTIMIZERS = {
    # bytes of optimizer state per trainable parameter
    "adamw": 8.0, "adamw_torch": 8.0, "adamw_torch_fused": 8.0, "paged_adamw_32bit": 8.0,
    "adamw_8bit": 2.0, "paged_adamw_8bit": 2.0, "adamw_bnb_8bit": 2.0,
    "lion": 4.0, "lion_8bit": 1.0, "paged_lion_8bit": 1.0,
    "sgd": 0.0, "sgd_momentum": 4.0, "adafactor": -1.0,  # -1: computed from matrix shapes
}
GC_MODES = ("none", "standard", "unsloth")
STATUS_ORDER = ("fits", "tight", "offload", "spill", "oom")

_ENGINE_ALIASES = {
    "llamacpp": "llama.cpp", "llama-cpp": "llama.cpp", "llama_cpp": "llama.cpp", "llama-server": "llama.cpp",
    "gguf": "llama.cpp", "ollama": "ollama", "vllm": "vllm", "exl2": "exllamav2", "exllama": "exllamav2",
    "exllamav2": "exllamav2", "tabby": "exllamav2", "tabbyapi": "exllamav2", "unsloth": "unsloth",
    "hf": "transformers", "transformers": "transformers", "peft": "transformers", "trl": "transformers",
}

# Fraction of peak memory bandwidth reached while decoding (calibrated on public llama-bench /
# vLLM single-stream numbers: e.g. Llama-3-8B Q4_K_M on an RTX 4090 ~ 135 tok/s).
_MEM_EFF = {
    ("llama.cpp", "nvidia"): 0.62, ("llama.cpp", "amd"): 0.50, ("llama.cpp", "apple"): 0.65,
    ("llama.cpp", "intel"): 0.40, ("vllm", "nvidia"): 0.60, ("vllm", "amd"): 0.50,
    ("exllamav2", "nvidia"): 0.70, ("exllamav2", "amd"): 0.55, ("transformers", "nvidia"): 0.35,
    ("unsloth", "nvidia"): 0.45,
}
# Fraction of dense FP16 tensor TFLOPS reached during prompt processing.
_PP_EFF = {
    ("llama.cpp", "nvidia"): 1.0, ("llama.cpp", "amd"): 0.40, ("llama.cpp", "apple"): 0.80,
    ("llama.cpp", "intel"): 0.30, ("vllm", "nvidia"): 0.45, ("vllm", "amd"): 0.35,
    ("exllamav2", "nvidia"): 0.45, ("transformers", "nvidia"): 0.35, ("unsloth", "nvidia"): 0.40,
}
_CPU_MEM_EFF = 0.70
_PCIE_EFF = 0.80


def normalize_engine(engine: str) -> str:
    e = (engine or "llama.cpp").strip().lower()
    if e in _ENGINE_ALIASES:
        return _ENGINE_ALIASES[e]
    if e in ENGINES:
        return e
    raise ValueError(f"unknown engine {engine!r}; choose from {', '.join(ENGINES)}")


@dataclass
class Settings:
    engine: str = "llama.cpp"
    mode: str = "inference"
    quant: Optional[str] = None
    context: int = 8192
    batch: int = 1
    kv_cache: Optional[str] = None          # f16 | q8_0 | q4_0 | fp8 | Q4 | Q6 | Q8 | 16 | 8 | 4
    gpu_layers: Optional[int] = None        # llama.cpp -ngl; None = auto (max that fits)
    cpu_moe_layers: int = 0                 # llama.cpp --n-cpu-moe
    flash_attention: bool = True
    ubatch: int = 512
    lora_rank: int = 16
    lora_targets: Optional[List[str]] = None
    lora_dropout: float = 0.0
    optimizer: str = "adamw_8bit"
    grad_checkpointing: Optional[str] = None  # none | standard | unsloth; None = engine default
    gpu_mem_util: float = 0.90              # vLLM --gpu-memory-utilization
    max_batched_tokens: Optional[int] = None  # vLLM --max-num-batched-tokens
    enforce_eager: bool = False             # vLLM: skip CUDA graphs
    swa_full: bool = False                  # llama.cpp --swa-full
    background_mib: Optional[float] = None  # VRAM already used by other processes
    ram_total_gib: Optional[float] = None
    ram_bandwidth_gbs: Optional[float] = None
    dataset_tokens: Optional[int] = None    # training: tokens per epoch, for time estimates
    grad_accum: int = 1

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        """Build validated settings from untrusted JSON (API / WebSocket). Raises ValueError on bad input."""
        if d is not None and not isinstance(d, dict):
            raise ValueError("settings must be a JSON object")
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        aliases = {"ctx": "context", "context_length": "context", "ngl": "gpu_layers", "kv": "kv_cache",
                   "kv_bits": "kv_cache", "kv_type": "kv_cache", "fa": "flash_attention",
                   "rank": "lora_rank", "n_cpu_moe": "cpu_moe_layers", "gc": "grad_checkpointing"}
        clean = {}
        for k, v in (d or {}).items():
            k = aliases.get(k, k)
            if k in known:
                clean[k] = v
        s = cls(**clean)
        s.validate()
        return s

    def validate(self) -> "Settings":
        """Coerce types and check ranges in place; raises ValueError with a readable message."""
        def as_int(name, lo, hi, optional=False):
            v = getattr(self, name)
            if v is None and optional:
                return
            if isinstance(v, bool):
                raise ValueError(f"{name} must be an integer")
            try:
                iv = int(float(v)) if isinstance(v, (int, float, str)) else None
            except (TypeError, ValueError):
                iv = None
            if iv is None or (isinstance(v, float) and not float(v).is_integer()):
                raise ValueError(f"{name} must be an integer, got {v!r}")
            if not lo <= iv <= hi:
                raise ValueError(f"{name} must be between {lo:,} and {hi:,}, got {iv:,}")
            setattr(self, name, iv)

        def as_float(name, lo, hi, optional=False):
            v = getattr(self, name)
            if v is None and optional:
                return
            try:
                fv = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"{name} must be a number, got {v!r}") from None
            if fv != fv or not lo <= fv <= hi:
                raise ValueError(f"{name} must be between {lo:g} and {hi:g}, got {fv:g}")
            setattr(self, name, fv)

        def as_bool(name):
            v = getattr(self, name)
            if isinstance(v, str):
                if v.strip().lower() in ("1", "true", "yes", "on"):
                    v = True
                elif v.strip().lower() in ("0", "false", "no", "off"):
                    v = False
                else:
                    raise ValueError(f"{name} must be true or false, got {v!r}")
            setattr(self, name, bool(v))

        self.engine = normalize_engine(str(self.engine))
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}, got {self.mode!r}")
        as_int("context", 64, 16_777_216)
        as_int("batch", 1, 4096)
        as_int("gpu_layers", 0, 100_000, optional=True)
        as_int("cpu_moe_layers", 0, 100_000)
        as_int("ubatch", 1, 65_536)
        as_int("lora_rank", 1, 65_536)
        as_int("grad_accum", 1, 1_000_000)
        as_int("max_batched_tokens", 1, 16_777_216, optional=True)
        as_int("dataset_tokens", 1, 10**15, optional=True)
        as_float("lora_dropout", 0.0, 1.0)
        as_float("gpu_mem_util", 0.05, 1.0)
        as_float("background_mib", 0.0, 10**8, optional=True)
        as_float("ram_total_gib", 0.0, 10**6, optional=True)
        as_float("ram_bandwidth_gbs", 0.1, 10**5, optional=True)
        for b in ("flash_attention", "enforce_eager", "swa_full"):
            as_bool(b)
        if self.kv_cache is not None:
            if isinstance(self.kv_cache, bool):
                raise ValueError("kv_cache must be a type name or bit width")
            self.kv_cache = str(int(self.kv_cache)) if isinstance(self.kv_cache, (int, float)) else str(self.kv_cache)
        if self.quant is not None:
            self.quant = str(self.quant)
        if str(self.optimizer).lower() not in OPTIMIZERS:
            raise ValueError(f"unknown optimizer {self.optimizer!r}; choose from {', '.join(OPTIMIZERS)}")
        if self.grad_checkpointing is not None and self.grad_checkpointing not in GC_MODES:
            raise ValueError(f"grad_checkpointing must be one of {', '.join(GC_MODES)}")
        if self.lora_targets is not None:
            if isinstance(self.lora_targets, str):
                self.lora_targets = [x.strip() for x in self.lora_targets.split(",") if x.strip()]
            if not isinstance(self.lora_targets, list) or not all(isinstance(x, str) for x in self.lora_targets):
                raise ValueError("lora_targets must be a list of module names")
            bad = [x for x in self.lora_targets if x not in LORA_TARGET_KINDS]
            if bad:
                raise ValueError(f"unknown LoRA target(s) {bad}; choose from {', '.join(LORA_TARGET_KINDS)}")
        return self

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Weight profile (per model x quant x engine; cached because sliders re-run estimate)
# ---------------------------------------------------------------------------

@dataclass
class WeightsProfile:
    quant: QuantScheme
    exact: bool
    calibrated: bool
    layer_bytes: List[int]
    layer_expert_bytes: List[int]
    layer_cat: List[Dict[str, int]]
    embed_bytes: int
    head_bytes: int
    global_bytes: int
    vision_bytes: int
    total_bytes: int
    expert_fraction_active: float
    largest_linear: int
    lora_shapes: List[Tuple[str, int, int, int]] = field(default_factory=list)  # (kind, n_in, n_out, experts)
    matrices: List[Tuple[int, int, int]] = field(default_factory=list)          # (n_in, n_out, experts)


_PROFILE_CACHE: Dict[tuple, WeightsProfile] = {}


def _category(kind: str) -> str:
    if kind in ATTN_KINDS:
        return "attention"
    if kind in EXPERT_KINDS:
        return "experts"
    if kind in SHARED_EXPERT_KINDS or kind.startswith("ffn_") or kind.startswith("ssm_"):
        return "mlp"
    if kind == "router":
        return "mlp"
    return "norms_other"


def weights_profile(model: LoadedModel, quant: QuantScheme, engine: str) -> WeightsProfile:
    key = (id(model), len(model.inventory), quant.name, engine)
    hit = _PROFILE_CACHE.get(key)
    if hit is not None:
        return hit
    s = model.spec
    L = s.num_layers
    qc = make_quant_context(s, model.inventory)
    exact = bool(model.exact and model.native_quant and model.native_quant.upper() == quant.name.upper())
    gguf_engine = engine in GGUF_ENGINES
    keep_tied_head = engine in GGUF_ENGINES or engine == "exllamav2"

    layer_bytes = [0] * L
    layer_exp = [0] * L
    layer_cat = [dict(attention=0, mlp=0, experts=0, norms_other=0) for _ in range(L)]
    embed = head = glob = vision = 0
    largest = 0
    lora_shapes = []
    matrices = []
    for t in model.inventory:
        if t.tied and not keep_tied_head:
            continue
        if t.kind == "vision" or (t.kind == "other" and t.layer is None and t.n_out <= 1 and t.nbytes_exact is None):
            if gguf_engine:
                continue  # GGUF vision towers ship in a separate mmproj file
            b = t.numel * 2
            vision += b
            continue
        b = t.nbytes_exact if (exact and t.nbytes_exact is not None) else quant.tensor_bytes(t, qc)
        if t.is_linear:
            largest = max(largest, t.numel // max(1, t.experts))
            lora_shapes.append((t.kind, t.n_in, t.n_out, t.experts))
        if t.n_out > 1:
            matrices.append((t.n_in, t.n_out, t.experts))
        if t.layer is not None and 0 <= t.layer < L:
            if t.kind in EXPERT_KINDS:
                layer_exp[t.layer] += b
            else:
                layer_bytes[t.layer] += b
            layer_cat[t.layer][_category(t.kind)] += b
        elif t.kind == "embed":
            embed += b
        elif t.kind == "head":
            head += b
        else:
            glob += b

    calibrated = False
    if gguf_engine and not exact and model.gguf_files:
        f = next((f for f in model.gguf_files if not f.get("mmproj") and
                  (f.get("quant") or "").upper() == quant.name.upper() and f.get("size")), None)
        synth = sum(layer_bytes) + sum(layer_exp) + embed + glob + (0 if s.tie_embeddings else head)
        if f and synth > 0:
            scale = max(0.5, min(1.5, (f["size"] - 6 * MiB) / synth))
            layer_bytes = [int(x * scale) for x in layer_bytes]
            layer_exp = [int(x * scale) for x in layer_exp]
            for c in layer_cat:
                for k in c:
                    c[k] = int(c[k] * scale)
            embed, head, glob = int(embed * scale), int(head * scale), int(glob * scale)
            calibrated = True

    frac = (s.experts_per_token / s.num_experts) if s.num_experts else 1.0
    total = sum(layer_bytes) + sum(layer_exp) + embed + head + glob + vision
    prof = WeightsProfile(quant=quant, exact=exact, calibrated=calibrated, layer_bytes=layer_bytes,
                          layer_expert_bytes=layer_exp, layer_cat=layer_cat, embed_bytes=embed,
                          head_bytes=head, global_bytes=glob, vision_bytes=vision, total_bytes=total,
                          expert_fraction_active=frac, largest_linear=largest, lora_shapes=lora_shapes,
                          matrices=matrices)
    if len(_PROFILE_CACHE) > 256:
        _PROFILE_CACHE.clear()
    _PROFILE_CACHE[key] = prof
    return prof


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------

def resolve_kv(engine: str, kv: Optional[str], flash: bool, model_dtype_bytes: float = 2.0) -> Tuple[str, str, float, float, List[str]]:
    """Return (k_type, v_type, k_bytes_per_elem, v_bytes_per_elem, warnings)."""
    warns: List[str] = []
    raw = (kv or "16").strip()
    key = raw.lower()
    if engine in GGUF_ENGINES:
        bits_map = {"16": "f16", "8": "q8_0", "4": "q4_0", "fp16": "f16", "fp8": "q8_0", "q8": "q8_0", "q4": "q4_0",
                    "32": "f32", "5": "q5_0", "q5": "q5_0", "auto": "f16"}
        k = bits_map.get(key, key)
        if k not in KV_CACHE_TYPES:
            raise ValueError(f"unsupported KV cache type {raw!r} for {engine}; use one of {', '.join(KV_CACHE_TYPES)}")
        v = k
        if k not in ("f16", "bf16", "f32") and not flash:
            warns.append("Quantized V-cache requires flash attention in llama.cpp; V stays f16 (enable -fa on).")
            v = "f16"
        return k, v, KV_CACHE_TYPES[k], KV_CACHE_TYPES[v], warns
    if engine == "vllm":
        if key in ("16", "auto", "f16", "fp16", "bf16"):
            return "auto", "auto", model_dtype_bytes, model_dtype_bytes, warns
        if key in ("8", "fp8", "fp8_e4m3", "fp8_e5m2", "q8_0", "q8"):
            t = key if key.startswith("fp8_") else "fp8"
            return t, t, 1.0, 1.0, warns
        warns.append(f"vLLM has no {raw}-bit KV cache; using fp8.")
        return "fp8", "fp8", 1.0, 1.0, warns
    if engine == "exllamav2":
        m = {"16": "FP16", "fp16": "FP16", "f16": "FP16", "8": "Q8", "q8": "Q8", "q8_0": "Q8", "6": "Q6",
             "q6": "Q6", "4": "Q4", "q4": "Q4", "q4_0": "Q4"}
        t = m.get(key)
        if t is None:
            raise ValueError(f"unsupported ExLlamaV2 cache mode {raw!r}; use FP16, Q8, Q6 or Q4")
        b = {"FP16": 2.0, "Q8": 1.0625, "Q6": 0.8125, "Q4": 0.5625}[t]
        return t, t, b, b, warns
    if key not in ("16", "auto", "f16", "bf16", "fp16"):
        warns.append("transformers/unsloth keep the KV cache in the model dtype; ignoring KV quantization.")
    return "bf16", "bf16", model_dtype_bytes, model_dtype_bytes, warns


def _pad(n: int, to: int) -> int:
    return -(-n // to) * to


def kv_tokens_per_layer(model: LoadedModel, engine: str, ctx: int, batch: int, ubatch: int, swa_full: bool) -> List[int]:
    s = model.spec
    out = []
    if engine in GGUF_ENGINES:
        total = _pad(ctx * batch, 256)
        for i in range(s.num_layers):
            lt = s.layer_types[i] if i < len(s.layer_types) else "full"
            if lt == "linear":
                out.append(0)
            elif lt == "sliding" and s.sliding_window and not swa_full:
                out.append(min(total, _pad(s.sliding_window * batch + ubatch, 256)))
            else:
                out.append(total)
        return out
    for i in range(s.num_layers):
        lt = s.layer_types[i] if i < len(s.layer_types) else "full"
        if lt == "linear":
            out.append(0)
        elif lt == "sliding" and s.sliding_window:
            out.append(min(ctx, s.sliding_window) * batch)
        else:
            out.append(ctx * batch)
    return out


def kv_layer_bytes(model: LoadedModel, tokens: List[int], kb: float, vb: float, batch: int) -> List[int]:
    s = model.spec
    out = []
    for i, n in enumerate(tokens):
        lt = s.layer_types[i] if i < len(s.layer_types) else "full"
        if lt == "linear":
            out.append(s.linear_state_bytes * batch)
            continue
        if s.is_mla:
            out.append(int(n * (s.kv_lora_rank + s.qk_rope_head_dim) * kb))
        else:
            nkv = s.kv_heads(i)
            out.append(int(n * nkv * (s.head_dim * kb + s.v_head_dim * vb)))
    return out


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

def _ffn_width(model: LoadedModel) -> int:
    s = model.spec
    dense = s.intermediate_size if (not s.moe_layers or not all(s.moe_layers)) else 0
    moe = s.experts_per_token * s.moe_intermediate_size + s.shared_expert_intermediate_size if s.is_moe else 0
    return max(dense, moe, 1)


def _q_dims(model: LoadedModel) -> Tuple[int, int]:
    s = model.spec
    q = s.num_heads * s.head_dim
    kv = max(s.num_kv_heads or [s.num_heads]) * (s.head_dim + s.v_head_dim)
    if s.is_mla:
        kv = s.kv_lora_rank + s.qk_rope_head_dim
    return q, kv


def llama_cpp_compute(model: LoadedModel, ctx_cells: int, ubatch: int, flash: bool, head_on_gpu: bool,
                      kv_quant: bool, n_outputs: int) -> Dict[str, int]:
    """CUDA/Metal compute-buffer size reserved by llama.cpp's graph allocator (worst case)."""
    s = model.spec
    ub = max(1, min(ubatch, ctx_cells))
    H = s.hidden_size
    ffn = ub * 2 * _ffn_width(model) * 4
    if s.is_moe:
        ffn += ub * s.num_experts * 4 * 2  # router logits + probs
    if flash:
        attn = ub * s.num_heads * max(s.head_dim, s.v_head_dim) * 4 * 2
    else:
        attn = ub * ctx_cells * s.num_heads * 4  # f32 KQ scores for the whole ubatch
    mask = ctx_cells * _pad(ub, 64) * (2 if flash else 4)
    logits = ub * s.vocab_size * 4 if head_on_gpu else 0
    hidden = ub * H * 4 * 4
    dequant = 0
    if flash and kv_quant:
        _, kvd = _q_dims(model)
        dequant = ctx_cells * kvd * 2  # batched FA converts quantized K/V to f16 in the CUDA pool
    gpu = hidden + max(ffn, attn, logits) + mask + dequant
    host_out = n_outputs * s.vocab_size * 4 + n_outputs * H * 4
    return {"gpu": int(gpu), "host_output": int(host_out), "ffn": int(ffn), "attention": int(attn),
            "logits": int(logits), "hidden": int(hidden), "mask": int(mask), "kv_dequant": int(dequant)}


def vllm_activations(model: LoadedModel, max_batched_tokens: int, max_num_seqs: int) -> Dict[str, int]:
    s = model.spec
    T = max_batched_tokens
    q, kv = _q_dims(model)
    per_tok = (2 * _ffn_width(model) + 3 * s.hidden_size + q + kv) * 2
    act = T * per_tok
    sampler = min(max_num_seqs, 1024) * s.vocab_size * 4 * 2
    return {"gpu": int(act + sampler), "forward": int(act), "sampler": int(sampler)}


def training_activations(model: LoadedModel, batch: int, seq: int, flash: bool, gc: str, engine: str,
                         lora_rank: int, n_lora_modules_per_layer: int, quant: QuantScheme,
                         largest_linear: int) -> Dict[str, int]:
    s = model.spec
    H = s.hidden_size
    tokens = batch * seq
    q, kvd = _q_dims(model)
    # bytes saved for backward per token per layer (bf16), FlashAttention-2 style
    dense_ffn = 4 * s.intermediate_size
    moe_ffn = 4 * (s.experts_per_token * s.moe_intermediate_size + s.shared_expert_intermediate_size) if s.is_moe else 0
    per_layer = []
    for i in range(s.num_layers):
        is_moe = i < len(s.moe_layers) and s.moe_layers[i]
        ffn = moe_ffn if is_moe else dense_ffn
        a = 2 * (4 * H + q + kvd + q + ffn) + 4 * s.num_heads
        if is_moe:
            a += s.num_experts * 4
        if not flash:
            a += s.num_heads * seq * (2 + 2)  # eager attention keeps scores + softmax
        a += 2 * lora_rank * n_lora_modules_per_layer
        per_layer.append(a)
    full_layer = max(per_layer) if per_layer else 0
    saved_input = 2 * H
    if gc == "none":
        gpu = tokens * sum(per_layer)
        ram = 0
    elif gc == "standard":
        gpu = s.num_layers * tokens * saved_input + 2 * tokens * full_layer
        ram = 0
    else:  # unsloth: checkpointed layer inputs are offloaded to system RAM asynchronously
        gpu = 2 * tokens * saved_input + tokens * full_layer
        ram = s.num_layers * tokens * saved_input
    logit_bytes = 2 if engine == "unsloth" else 12  # unsloth fused CE (bf16, grads in place) vs fp32 upcast + grad
    logits = tokens * s.vocab_size * logit_bytes
    dequant = largest_linear * 2 if quant.family == "nf4" else 0
    return {"gpu": int(gpu + logits + dequant), "ram_offloaded": int(ram), "layers": int(gpu),
            "logits": int(logits), "dequant_workspace": int(dequant), "per_token_per_layer": int(full_layer)}


# ---------------------------------------------------------------------------
# LoRA / optimizer
# ---------------------------------------------------------------------------

def lora_trainable(prof: WeightsProfile, rank: int, targets: Optional[List[str]]) -> Tuple[int, List[Tuple[int, int, int]], int]:
    tg = targets or DEFAULT_LORA_TARGETS
    kinds = set()
    for t in tg:
        kinds |= LORA_TARGET_KINDS.get(t, set())
    params = 0
    mats = []
    modules = 0
    for kind, n_in, n_out, experts in prof.lora_shapes:
        if kind in kinds:
            params += rank * (n_in + n_out) * experts
            mats.append((n_in, rank, experts))
            mats.append((rank, n_out, experts))
            modules += experts
    return params, mats, modules


def optimizer_bytes(name: str, trainable: int, mats: List[Tuple[int, int, int]]) -> int:
    key = name.lower()
    if key not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer {name!r}; choose from {', '.join(OPTIMIZERS)}")
    per = OPTIMIZERS[key]
    if per >= 0:
        return int(trainable * per)
    # Adafactor: factored second moment (row + column vectors, fp32) per matrix
    return int(sum((a + b) * e * 4 for a, b, e in mats))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _first_gpu_layer(n_layer: int, ngl: int) -> int:
    """llama.cpp load_tensors: i_gpu_start = max(n_layer + 1 - n_gpu_layers, 0); index n_layer is the output head."""
    return min(n_layer, max(n_layer + 1 - ngl, 0))


def _attn_span_sum(s, ctx: int) -> float:
    """Sum over attention layers of the average causal span a token attends to."""
    tot = 0.0
    for i in range(s.num_layers):
        lt = s.layer_types[i] if i < len(s.layer_types) else "full"
        if lt == "linear":
            continue
        tot += min(ctx / 2, s.sliding_window) if (lt == "sliding" and s.sliding_window) else ctx / 2
    return tot


def _overhead(engine: str, gpu: GPU, mode: str, enforce_eager: bool) -> Dict[str, int]:
    v = gpu.vendor
    if v == "cpu" or gpu.vram_gib <= 0:
        return {"driver_context": 0}
    torch_engine = engine in ("vllm", "unsloth", "transformers", "exllamav2")
    if v == "nvidia":
        ctx = 500 if torch_engine else 300
    elif v == "amd":
        ctx = 600 if torch_engine else 400
    elif v == "apple":
        ctx = 300 if torch_engine else 120
    else:
        ctx = 400
    out = {"driver_context": ctx * MiB}
    if engine == "vllm":
        out["cuda_graphs"] = 0 if enforce_eager else 512 * MiB
        out["non_torch"] = 200 * MiB
        if gpu.count > 1:
            out["nccl"] = 256 * MiB
    if engine == "ollama":
        out["runner"] = 100 * MiB
    return out


def _model_dtype_bytes(model: LoadedModel) -> float:
    return 4.0 if "32" in model.spec.torch_dtype else 2.0


def estimate(model: LoadedModel, gpu: GPU, settings: Settings, hw: Optional[HardwareProfile] = None,
             limits: bool = True) -> dict:
    st = replace(settings).validate()
    warnings: List[str] = []
    notes: List[str] = list(model.notes)
    s = model.spec
    L = s.num_layers
    n = max(1, gpu.count)

    # ---- mode / engine consistency ------------------------------------------------
    if st.mode not in MODES:
        raise ValueError(f"unknown mode {st.mode!r}; choose from {', '.join(MODES)}")
    training = st.mode != "inference"
    if training and st.engine not in TRAIN_ENGINES:
        warnings.append(f"{st.engine} cannot fine-tune; estimating with Unsloth instead.")
        st.engine = "unsloth"
    if training and gpu.vendor == "apple":
        warnings.append("Unsloth/bitsandbytes need CUDA/ROCm; on Apple Silicon use mlx-lm LoRA (similar memory).")
    if st.engine == "vllm" and gpu.vendor in ("apple", "cpu"):
        warnings.append("vLLM's GPU backend needs NVIDIA/AMD/Intel GPUs; use llama.cpp on this hardware.")
    if st.engine == "exllamav2" and gpu.vendor not in ("nvidia", "amd"):
        warnings.append("ExLlamaV2 requires an NVIDIA (CUDA) or AMD (ROCm) GPU.")
    gc = st.grad_checkpointing or ("unsloth" if st.engine == "unsloth" else "standard")
    if gc not in GC_MODES:
        raise ValueError(f"grad_checkpointing must be one of {', '.join(GC_MODES)}")

    # ---- quantization ---------------------------------------------------------------
    qname = st.quant
    if not qname:
        if st.mode == "qlora":
            qname = "NF4"
        elif training:
            qname = "BF16"
        elif model.native_quant and st.engine in get_quant(model.native_quant).engines:
            qname = model.native_quant
        elif model.native_quant == "MXFP4" and st.engine in GGUF_ENGINES:
            qname = "MXFP4_MOE"  # gpt-oss GGUFs keep the MXFP4 experts
        else:
            qname = default_quant(st.engine, st.mode)
    quant = get_quant(qname)
    if st.mode == "qlora" and quant.family != "nf4":
        warnings.append(f"QLoRA quantizes the base model to NF4; ignoring quant {quant.name}.")
        quant = get_quant("NF4")
    if st.mode in ("lora", "full") and quant.family not in ("fp",):
        warnings.append(f"{st.mode.upper()} trains on a 16-bit base; ignoring quant {quant.name} (use mode qlora for 4-bit).")
        quant = get_quant("BF16")
    if training and not gpu.supports_bf16 and quant.name == "BF16":
        quant = get_quant("F16")
        notes.append(f"{gpu.name} has no bf16 support: training in fp16 with loss scaling.")
    if st.engine not in quant.engines:
        warnings.append(f"{quant.name} is not a native {st.engine} format; the estimate assumes it anyway.")
    if s.quantization and model.native_quant and quant.name != model.native_quant and st.engine not in GGUF_ENGINES:
        warnings.append(f"This repo ships pre-quantized {model.native_quant} weights; {quant.name} needs a different checkpoint.")
    if quant.name == "FP8" and not gpu.supports_fp8 and gpu.vendor == "nvidia":
        notes.append("No native FP8 on this GPU: vLLM falls back to FP8 Marlin weight-only kernels (same memory).")

    prof = weights_profile(model, quant, st.engine)

    # ---- context sanity --------------------------------------------------------------
    ctx = max(64, int(st.context))
    batch = max(1, int(st.batch))
    if ctx > s.max_context:
        warnings.append(f"Context {ctx:,} exceeds the model's trained window ({s.max_context:,}); "
                        "the generated command enables YaRN RoPE scaling, quality may degrade.")

    # ---- KV cache --------------------------------------------------------------------
    kt = vt = None
    kb = vb = 0.0
    kv_tokens = [0] * L
    kv_bytes = [0] * L
    if not training:
        kt, vt, kb, vb, kw = resolve_kv(st.engine, st.kv_cache, st.flash_attention, _model_dtype_bytes(model))
        warnings += kw
        kv_tokens = kv_tokens_per_layer(model, st.engine, ctx, batch, st.ubatch, st.swa_full)
        kv_bytes = kv_layer_bytes(model, kv_tokens, kb, vb, batch)

    # ---- capacity --------------------------------------------------------------------
    vram_total = gpu.vram_gib * GiB * n
    ceiling = gpu.usable_gib * GiB * n
    if st.background_mib is not None:
        background = st.background_mib * MiB
    elif gpu.probed and gpu.extra.get("used_mib"):
        background = float(gpu.extra["used_mib"]) * MiB
    else:
        background = 0.0
    budget = ceiling * st.gpu_mem_util if st.engine == "vllm" else ceiling
    if hw is not None:
        ram_total = (st.ram_total_gib or hw.ram_total_gib) * GiB
        ram_bw = st.ram_bandwidth_gbs or hw.ram_bandwidth_gbs
    else:
        ram_total = (st.ram_total_gib or (gpu.vram_gib if gpu.unified else max(32.0, gpu.vram_gib * 2))) * GiB
        ram_bw = st.ram_bandwidth_gbs or (gpu.bandwidth_gbs if gpu.unified else DEFAULT_RAM_BW_GBS)
    overhead = _overhead(st.engine, gpu, st.mode, st.enforce_eager)
    overhead_gpu = sum(overhead.values()) * n + background

    has_gpu = gpu.vendor != "cpu" and gpu.vram_gib > 0
    partial_engine = st.engine in GGUF_ENGINES and has_gpu and not gpu.unified

    def place(ngl: int, cpu_moe: int) -> dict:
        """Split weights / KV between GPU and CPU for llama.cpp-style layer offload."""
        if not has_gpu:
            ngl = 0
        gpu_w = cpu_w = 0
        gpu_w_act = cpu_w_act = 0.0
        gpu_kv = cpu_kv = 0
        layer_map = []
        first_gpu = _first_gpu_layer(L, ngl)
        cat = dict(attention=0, mlp=0, experts=0, norms_other=0, embeddings=0, lm_head=0, vision=0)
        for i in range(L):
            on_gpu = i >= first_gpu
            nb, eb = prof.layer_bytes[i], prof.layer_expert_bytes[i]
            exp_on_gpu = on_gpu and not (i < cpu_moe)
            g_bytes = (nb if on_gpu else 0) + (eb if exp_on_gpu else 0)
            c_bytes = nb + eb - g_bytes
            gpu_w += g_bytes
            cpu_w += c_bytes
            act_e = eb * prof.expert_fraction_active
            gpu_w_act += (nb if on_gpu else 0) + (act_e if exp_on_gpu else 0)
            cpu_w_act += (0 if on_gpu else nb) + (0 if exp_on_gpu else act_e)
            if on_gpu:
                gpu_kv += kv_bytes[i]
                for k in ("attention", "mlp", "norms_other"):
                    cat[k] += prof.layer_cat[i][k]
                if exp_on_gpu:
                    cat["experts"] += prof.layer_cat[i]["experts"]
            else:
                cpu_kv += kv_bytes[i]
            layer_map.append({"layer": i, "weights_mib": round((nb + eb) / MiB, 2),
                              "kv_mib": round(kv_bytes[i] / MiB, 2),
                              "device": "gpu" if on_gpu and exp_on_gpu else ("split" if on_gpu else "cpu"),
                              "type": s.layer_types[i] if i < len(s.layer_types) else "full"})
        head_gpu = has_gpu and ngl >= 1  # llama.cpp counts the output head as one of the -ngl layers
        if st.engine not in GGUF_ENGINES:
            head_gpu = has_gpu
        embed_gpu = has_gpu and quant.embed_on_gpu and st.engine not in GGUF_ENGINES
        if head_gpu:
            gpu_w += prof.head_bytes + prof.global_bytes
            gpu_w_act += prof.head_bytes
            cat["lm_head"] += prof.head_bytes
            cat["norms_other"] += prof.global_bytes
        else:
            cpu_w += prof.head_bytes + prof.global_bytes
            cpu_w_act += prof.head_bytes
        if embed_gpu:
            gpu_w += prof.embed_bytes
            cat["embeddings"] += prof.embed_bytes
        else:
            cpu_w += prof.embed_bytes
        if has_gpu:
            gpu_w += prof.vision_bytes
            cat["vision"] += prof.vision_bytes
        else:
            cpu_w += prof.vision_bytes
        return dict(ngl=ngl, cpu_moe=cpu_moe, gpu_w=gpu_w, cpu_w=cpu_w, gpu_w_act=gpu_w_act,
                    cpu_w_act=cpu_w_act, gpu_kv=gpu_kv, cpu_kv=cpu_kv, head_gpu=head_gpu,
                    embed_gpu=embed_gpu, layer_map=layer_map, cat=cat)

    # ---- activations + training states (independent of placement except logits) -------
    def activations(head_gpu: bool) -> Tuple[int, int, Dict[str, int]]:
        if not has_gpu:
            if st.engine in GGUF_ENGINES:
                c = llama_cpp_compute(model, sum(kv_tokens) // max(1, L) or ctx, st.ubatch, st.flash_attention,
                                      False, (kt or "f16") not in ("f16", "bf16", "f32"), batch)
                return 0, c["gpu"] + c["host_output"], c
            return 0, 0, {}
        if training:
            _, _, mods = lora_trainable(prof, st.lora_rank, st.lora_targets) if st.mode != "full" else (0, [], 0)
            a = training_activations(model, batch, ctx, st.flash_attention, gc, st.engine, st.lora_rank if st.mode != "full" else 0,
                                     mods // max(1, L), quant, prof.largest_linear)
            return a["gpu"], a["ram_offloaded"], a
        if st.engine in GGUF_ENGINES:
            cells = max(kv_tokens) if kv_tokens else ctx
            c = llama_cpp_compute(model, cells, st.ubatch, st.flash_attention, head_gpu,
                                  (kt or "f16") not in ("f16", "bf16", "f32"), batch)
            return c["gpu"], c["host_output"], c
        if st.engine == "vllm":
            # `vllm serve` default (EngineArgs usage-context defaults): 8192 on >=70 GB non-A100 GPUs, else 2048
            big = gpu.vram_gib >= 70 and "a100" not in gpu.name.lower()
            T = st.max_batched_tokens or (8192 if big else 2048)
            v = vllm_activations(model, T, max(batch, 1))
            return v["gpu"] * n, 0, v
        if st.engine == "exllamav2":
            T = min(2048, ctx)
            a = T * (2 * _ffn_width(model) + 4 * s.hidden_size) * 2 + batch * s.vocab_size * 4
            if not st.flash_attention:
                a += T * min(ctx, 2048) * s.num_heads * 2
            return int(a), 0, {"gpu": int(a), "chunk_tokens": T}
        # transformers / unsloth inference: one prefill forward of the whole prompt
        tokens = ctx * batch
        a = tokens * (2 * _ffn_width(model) + 4 * s.hidden_size) * 2 + batch * s.vocab_size * 4
        if not st.flash_attention:
            a += batch * s.num_heads * ctx * ctx * 2
        return int(a), 0, {"gpu": int(a)}

    lora_detail: Dict[str, int] = {}
    states_gpu = 0
    if training:
        if st.mode == "full":
            trainable = model.params_total
            mats = [(a, b, e) for a, b, e in prof.matrices]
            grads = trainable * 2
            master = trainable * 4
            opt = optimizer_bytes(st.optimizer, trainable, mats)
            lora_detail = {"trainable_params": trainable, "grads": grads, "master_weights_fp32": master,
                           "optimizer_states": opt}
            states_gpu = grads + master + opt
        else:
            trainable, mats, _ = lora_trainable(prof, st.lora_rank, st.lora_targets)
            adapters = trainable * 4
            grads = trainable * 4
            opt = optimizer_bytes(st.optimizer, trainable, mats)
            lora_detail = {"trainable_params": trainable, "adapter_weights_fp32": adapters, "grads": grads,
                           "optimizer_states": opt}
            states_gpu = adapters + grads + opt
        if st.engine == "transformers" and st.mode == "qlora":
            # prepare_model_for_kbit_training upcasts embeddings / lm_head / norms to fp32
            up = prof.embed_bytes + (prof.head_bytes if not s.tie_embeddings else 0)
            lora_detail["fp32_upcast_non_quantized"] = up
            states_gpu += up
        if st.optimizer.startswith("paged_"):
            notes.append("Paged optimizer states can be evicted to CPU RAM under memory pressure (bitsandbytes).")

    def total_for(p: dict) -> Tuple[float, float, int, int, Dict[str, int]]:
        act_gpu, act_cpu, act_detail = activations(p["head_gpu"])
        frag = 0
        if training or st.engine in ("transformers", "unsloth"):
            frag = int(0.07 * (act_gpu + states_gpu))
        gpu_total = p["gpu_w"] + p["gpu_kv"] + act_gpu + states_gpu + overhead_gpu + frag
        cpu_total = p["cpu_w"] + p["cpu_kv"] + act_cpu
        return gpu_total, cpu_total, act_gpu + frag, act_cpu, act_detail

    # ---- choose placement ------------------------------------------------------------------
    auto_ngl = st.gpu_layers is None
    cpu_moe = max(0, min(int(st.cpu_moe_layers or 0), L))
    if not partial_engine:
        plan = place(L + 1 if has_gpu else 0, 0)
    elif not auto_ngl:
        plan = place(max(0, min(int(st.gpu_layers), L + 1)), cpu_moe)
    else:
        safe = budget * 0.98
        plan = None
        for ngl in range(L + 1, -1, -1):
            p = place(ngl, cpu_moe)
            if total_for(p)[0] <= safe:
                plan = p
                break
        if plan is None:
            plan = place(0, cpu_moe)
    gpu_total, cpu_total, act_gpu, act_cpu, act_detail = total_for(plan)

    # ---- status -----------------------------------------------------------------------------
    cpu_beyond_embed = plan["cpu_w"] - (0 if plan["embed_gpu"] else prof.embed_bytes)
    offloaded = cpu_beyond_embed > 1 * MiB or plan["cpu_kv"] > 0 or not has_gpu
    ram_usable = ram_total - 2 * GiB
    overflow = max(0.0, gpu_total - budget) if has_gpu else 0.0
    if not has_gpu:
        status = "offload" if cpu_total <= ram_usable else "oom"
    elif overflow > 0:
        if gpu.unified:
            status = "spill" if gpu_total <= ram_total - 4 * GiB else "oom"
        else:
            status = "spill" if overflow + cpu_total <= ram_usable else "oom"
    elif cpu_total > ram_usable:
        status = "oom"
    elif offloaded:
        status = "offload"
    elif gpu_total > 0.92 * budget:
        status = "tight"
    else:
        status = "fits"

    if status == "spill":
        if gpu.unified and gpu.vendor == "apple":
            need = int(math.ceil((gpu_total / MiB + 512) / 1024) * 1024)
            warnings.append(f"Exceeds the macOS GPU wired limit; raise it with `sudo sysctl iogpu.wired_limit_mb={need}`.")
        elif st.engine in GGUF_ENGINES:
            warnings.append(f"Over VRAM by {overflow / MiB:,.0f} MiB: lower -ngl, quantize the KV cache, or shrink the context. "
                            "On Linux this OOMs; on Windows the driver spills to shared memory over PCIe (very slow).")
        else:
            warnings.append(f"Over the VRAM budget by {overflow / MiB:,.0f} MiB: this configuration will raise CUDA out of memory.")
    elif status == "oom":
        warnings.append("Does not fit in VRAM + system RAM. Use a smaller quant, shorter context or more memory.")
    elif status == "offload" and has_gpu:
        warnings.append(f"{plan['cpu_w'] / GiB:,.2f} GiB of weights and {plan['cpu_kv'] / GiB:,.2f} GiB of KV cache run from "
                        f"system RAM ({ram_bw:g} GB/s) — decode speed is bound by RAM bandwidth.")

    # ---- performance ------------------------------------------------------------------------
    vend = gpu.vendor
    eng_key = "llama.cpp" if st.engine == "ollama" else st.engine
    mem_eff = _MEM_EFF.get((eng_key, vend), 0.5)
    pp_eff = _PP_EFF.get((eng_key, vend), 0.35)
    if eng_key == "llama.cpp" and vend == "nvidia" and gpu.form == "datacenter":
        pp_eff = 0.55
    if quant.family == "fp" and eng_key == "llama.cpp":
        pp_eff *= 0.8
    gpu_bw = gpu.bandwidth_gbs * 1e9
    if st.engine in ("vllm",) and n > 1:
        gpu_bw *= n * 0.85
    pcie_bw = gpu.pcie_gbs * 1e9 * _PCIE_EFF if gpu.pcie_gbs else 16e9
    cpu_bw = ram_bw * 1e9 * _CPU_MEM_EFF
    tflops = gpu.tflops * 1e12 * (n if st.engine == "vllm" else 1) if has_gpu else max(0.3e12, gpu.tflops * 1e12)

    def kv_read(pos: int, on_gpu: bool) -> float:
        tot = 0.0
        first_gpu = _first_gpu_layer(L, plan["ngl"])
        for i in range(L):
            if (i >= first_gpu) != on_gpu:
                continue
            lt = s.layer_types[i] if i < len(s.layer_types) else "full"
            if lt == "linear":
                tot += s.linear_state_bytes
                continue
            span = min(pos, s.sliding_window) if (lt == "sliding" and s.sliding_window) else pos
            if s.is_mla:
                tot += span * (s.kv_lora_rank + s.qk_rope_head_dim) * kb
            else:
                tot += span * s.kv_heads(i) * (s.head_dim * kb + s.v_head_dim * vb)
        return tot

    perf: Dict[str, object] = {}
    if not training:
        B = batch
        frac = prof.expert_fraction_active
        if s.is_moe and B > 1:
            uniq = 1 - (1 - frac) ** B
            scale = uniq / frac if frac else 1.0
        else:
            scale = 1.0
        exp_total_gpu = sum(prof.layer_expert_bytes[i] for i in range(_first_gpu_layer(L, plan["ngl"]), L) if i >= plan["cpu_moe"])
        exp_total_cpu = sum(prof.layer_expert_bytes) - exp_total_gpu
        # small per-token expert GEMVs reach ~half the bandwidth of dense layers
        moe_pen = 1.0 / 0.6 if s.is_moe else 1.0
        g_w = plan["gpu_w_act"] + exp_total_gpu * frac * (scale * moe_pen - 1)
        c_w = plan["cpu_w_act"] + exp_total_cpu * frac * (scale * moe_pen - 1)
        if not has_gpu:
            c_w, g_w = c_w + g_w, 0.0

        def step_time(pos: int) -> Tuple[float, str]:
            tg = (g_w + B * kv_read(pos, True)) / (gpu_bw * mem_eff) if has_gpu else 0.0
            tc = (c_w + B * kv_read(pos, False)) / cpu_bw if (c_w > 0 or not has_gpu) else 0.0
            ts = overflow / pcie_bw if (status == "spill" and not gpu.unified) else 0.0
            flops = B * 2 * model.params_active
            tcomp = flops / (tflops * pp_eff)
            # fixed per-layer cost: kernel launches / graph replay, plus 2 all-reduces per layer under TP
            t_fixed = L * (12e-6 + (2 * 25e-6 if (st.engine == "vllm" and n > 1) else 0.0))
            t = tg + tc + ts + t_fixed
            bott = "gpu_memory_bandwidth"
            if ts > max(tg, tc):
                bott = "pcie_spill"
            elif tc > tg:
                bott = "system_ram_bandwidth"
            if tcomp > t:
                t, bott = tcomp, "compute"
            return t, bott

        t0, bott = step_time(min(ctx, 512))
        t1, _ = step_time(ctx)
        # prompt processing (compute-bound) + PCIe streaming of CPU-resident weights per ubatch
        attn_flops = 4 * s.num_heads * s.head_dim * _attn_span_sum(s, ctx)
        pp_flops = 2 * model.params_active + attn_flops
        if has_gpu:
            t_tok = pp_flops / (tflops * pp_eff)
            if plan["cpu_w"] > 0 and st.engine in GGUF_ENGINES:
                stream = plan["cpu_w"] - prof.embed_bytes
                t_tok += max(0, stream) / pcie_bw / max(1, min(st.ubatch, ctx))
        else:
            t_tok = pp_flops / (tflops * 0.5)
        pp = 1.0 / t_tok if t_tok > 0 else 0.0
        perf = {
            "decode_tok_s": round(B / t0, 1),
            "decode_tok_s_per_seq": round(1 / t0, 1),
            "decode_tok_s_full_ctx": round(B / t1, 1),
            "prefill_tok_s": round(pp, 0),
            "ttft_full_ctx_s": round(ctx / pp, 2) if pp else None,
            "bottleneck": bott,
            "bytes_per_token_gpu_mib": round((g_w + kv_read(min(ctx, 512), True)) / MiB, 1),
            "bytes_per_token_cpu_mib": round(c_w / MiB, 1),
            "gpu_bandwidth_gbs": round(gpu_bw / 1e9, 1),
            "ram_bandwidth_gbs": ram_bw,
            "pcie_gbs": round(pcie_bw / 1e9, 1),
            "efficiency": {"memory": mem_eff, "compute": round(pp_eff, 2)},
        }
    else:
        mfu = 0.42 if st.engine == "unsloth" else 0.30
        if quant.family == "nf4":
            mfu *= 0.8
        if not gpu.supports_flash_attn2:
            mfu *= 0.8
        P = model.params_active
        base = 6 * P if st.mode == "full" else 4 * P
        if gc != "none":
            base += 2 * P
        attn = 12 * s.num_heads * s.head_dim * _attn_span_sum(s, ctx)
        tok_s = (gpu.tflops * 1e12 * n * mfu) / (base + attn) if has_gpu else 0.0
        perf = {"train_tok_s": round(tok_s, 0), "mfu_assumed": round(mfu, 2),
                "step_time_s": round(batch * ctx * max(1, st.grad_accum) / tok_s, 2) if tok_s else None,
                "bottleneck": "compute"}
        if st.dataset_tokens and tok_s:
            perf["epoch_hours"] = round(st.dataset_tokens / tok_s / 3600, 2)

    # ---- assemble ---------------------------------------------------------------------------
    w_gpu = plan["gpu_w"]
    segments = [
        {"key": "weights", "label": "Model Weights", "mib": round(w_gpu / MiB, 1)},
        {"key": "kv_cache", "label": "KV Cache", "mib": round(plan["gpu_kv"] / MiB, 1)},
        {"key": "activations", "label": "Activation Buffers", "mib": round(act_gpu / MiB, 1)},
        {"key": "overhead", "label": "CUDA/Driver Overhead", "mib": round(overhead_gpu / MiB, 1)},
        {"key": "lora_optimizer", "label": "LoRA/Optimizer States", "mib": round(states_gpu / MiB, 1)},
    ]
    kv_per_tok = sum(kv_layer_bytes(model, [1] * L, kb, vb, 1)) if not training else 0
    full_layers = s.layer_types.count("full") if s.layer_types else L
    result = {
        "status": status,
        "fits": status in ("fits", "tight", "offload"),
        "model": {
            "id": model.id, "repo": model.repo, "file": model.filename, "source": model.source,
            "architecture": s.architecture, "attention": s.attention_kind(),
            "params_total": model.params_total, "params_active": model.params_active,
            "layers": L, "hidden_size": s.hidden_size, "heads": s.num_heads,
            "kv_heads": max(s.num_kv_heads) if s.num_kv_heads else s.num_heads, "head_dim": s.head_dim,
            "vocab_size": s.vocab_size, "max_context": s.max_context, "experts": s.num_experts,
            "experts_per_token": s.experts_per_token, "sliding_window": s.sliding_window,
            "layer_types": {t: s.layer_types.count(t) for t in sorted(set(s.layer_types))},
            "native_quant": model.native_quant,
        },
        "hardware": {
            "name": (f"{n}x " if n > 1 else "") + gpu.name, "vendor": gpu.vendor, "count": n,
            "vram_mib": round(vram_total / MiB), "ceiling_mib": round(ceiling / MiB),
            "budget_mib": round(budget / MiB), "unified_memory": gpu.unified,
            "bandwidth_gbs": gpu.bandwidth_gbs, "tflops_fp16": gpu.tflops,
            "pcie": f"Gen{gpu.pcie_gen} x{gpu.pcie_width}" if gpu.pcie_gen else None,
            "ram_mib": round(ram_total / MiB), "probed": gpu.probed, "supports_bf16": gpu.supports_bf16,
        },
        "settings": {
            "engine": st.engine, "mode": st.mode, "quant": quant.name, "quant_bpw": quant.nominal_bpw,
            "context": ctx, "batch": batch, "kv_cache_k": kt, "kv_cache_v": vt,
            "gpu_layers": plan["ngl"] if partial_engine or not has_gpu else (L + 1 if has_gpu else 0),
            "gpu_layers_auto": auto_ngl and partial_engine, "max_gpu_layers": L + 1,
            "cpu_moe_layers": plan["cpu_moe"], "flash_attention": st.flash_attention, "ubatch": st.ubatch,
            "lora_rank": st.lora_rank if st.mode in ("lora", "qlora") else None,
            "lora_targets": (st.lora_targets or list(DEFAULT_LORA_TARGETS)) if st.mode in ("lora", "qlora") else None,
            "optimizer": st.optimizer if training else None,
            "grad_checkpointing": gc if training else None,
            "gpu_mem_util": st.gpu_mem_util if st.engine == "vllm" else None,
        },
        "segments": segments,
        "gpu_total_mib": round(gpu_total / MiB, 1),
        "ceiling_mib": round(ceiling / MiB, 1),
        "budget_mib": round(budget / MiB, 1),
        "headroom_mib": round((budget - gpu_total) / MiB, 1),
        "utilization": round(gpu_total / budget, 4) if budget else None,
        "offload": {
            "cpu_weights_mib": round(plan["cpu_w"] / MiB, 1),
            "cpu_kv_mib": round(plan["cpu_kv"] / MiB, 1),
            "cpu_buffers_mib": round(act_cpu / MiB, 1),
            "ram_total_mib": round(cpu_total / MiB, 1),
            "spill_mib": round(overflow / MiB, 1),
            "layers_on_gpu": (L - _first_gpu_layer(L, plan["ngl"])) if has_gpu else 0,
            "layers_total": L,
            "output_on_gpu": plan["head_gpu"],
        },
        "performance": perf,
        "details": {
            "weights": {
                "quant": quant.name, "exact_from_header": prof.exact, "calibrated_to_file_size": prof.calibrated,
                "total_mib": round(prof.total_bytes / MiB, 1),
                "gpu_by_category_mib": {k: round(v / MiB, 1) for k, v in plan["cat"].items() if v},
                "effective_bpw": round(prof.total_bytes * 8 / max(1, model.params_total), 3),
            },
            "kv_cache": {
                "k_type": kt, "v_type": vt, "bytes_per_token": kv_per_tok,
                "kib_per_token": round(kv_per_tok / 1024, 2), "cells": max(kv_tokens) if kv_tokens else 0,
                "full_attention_layers": full_layers,
                "sliding_layers": s.layer_types.count("sliding"), "linear_layers": s.layer_types.count("linear"),
                "total_mib": round(sum(kv_bytes) / MiB, 1),
            },
            "activations": {k: (round(v / MiB, 1) if isinstance(v, (int, float)) and k not in ("chunk_tokens",) else v)
                            for k, v in act_detail.items()},
            "overhead": {k: round(v / MiB, 1) for k, v in overhead.items()} | {"background_mib": round(background / MiB, 1)},
            "lora_optimizer": {k: (round(v / MiB, 1) if k != "trainable_params" else v) for k, v in lora_detail.items()},
        },
        "layer_map": plan["layer_map"],
        "warnings": warnings,
        "notes": notes,
    }
    if st.engine == "vllm" and not training:
        kv_avail = budget - (gpu_total - plan["gpu_kv"])
        per_seq = sum(kv_layer_bytes(model, kv_tokens_per_layer(model, "vllm", ctx, 1, 0, False), kb, vb, 1))
        result["details"]["kv_cache"]["vllm_preallocated_mib"] = round(max(0, kv_avail) / MiB, 1)
        result["details"]["kv_cache"]["vllm_max_concurrency_at_ctx"] = round(max(0, kv_avail) / per_seq, 2) if per_seq else None
    if limits:
        result["limits"] = _limits(model, gpu, st, hw)
    return result


def _limits(model: LoadedModel, gpu: GPU, st: Settings, hw: Optional[HardwareProfile]) -> dict:
    """Largest context (and batch) that keeps the same settings fully inside VRAM."""
    def fits(ctx: int = None, batch: int = None) -> bool:
        t = replace(st, context=ctx or st.context, batch=batch or st.batch)
        if t.engine in GGUF_ENGINES and t.gpu_layers is None:
            t.gpu_layers = model.spec.num_layers + 1
        r = estimate(model, gpu, t, hw, limits=False)
        return r["status"] in ("fits", "tight")

    out: Dict[str, Optional[int]] = {"max_context_full_gpu": None, "max_batch_full_gpu": None}
    if gpu.vendor == "cpu":
        return out
    lo, hi = 0, 0
    c = 512
    cap = max(model.spec.max_context, 4096) * 4
    if fits(ctx=c):
        lo = c
        while c < cap:
            c2 = min(cap, c * 2)
            if not fits(ctx=c2):
                hi = c2
                break
            lo = c = c2
        else:
            hi = 0
        if hi:
            while hi - lo > 256:
                mid = (lo + hi) // 2 // 256 * 256
                if mid <= lo:
                    break
                if fits(ctx=mid):
                    lo = mid
                else:
                    hi = mid
        out["max_context_full_gpu"] = lo
    b, best = 1, 0
    while b <= 1024 and fits(batch=b):
        best = b
        b *= 2
    out["max_batch_full_gpu"] = best or None
    return out
