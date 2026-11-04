"""Quantization schemes and exact per-tensor byte accounting.

GGUF schemes reproduce llama.cpp's ``llama_tensor_get_type`` mixing rules
(``use_more_bits`` layers, Q6_K output heads, divisibility fall-backs), so a
synthesized Q4_K_M estimate lands within ~1% of the real file size. Other
schemes (AWQ, GPTQ, EXL2, FP8, NF4, MXFP4, NVFP4...) account for scales, zero
points and the tensors those formats keep in 16-bit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .ggml import ggml_type
from .tensors import Tensor

GGUF_ENGINES = ("llama.cpp", "ollama")
HF_INFER_ENGINES = ("vllm",)
TRAIN_ENGINES = ("unsloth", "transformers")


@dataclass
class QuantContext:
    n_layer: int
    n_expert: int = 0
    n_gqa: int = 1
    has_output: bool = True
    arch: str = ""


@dataclass
class QuantScheme:
    name: str
    family: str             # gguf | fp | awq | gptq | fp8 | int8 | nf4 | exl2 | mxfp4 | nvfp4
    label: str
    nominal_bpw: float      # typical effective bits/weight of the decoder linear layers
    engines: List[str]
    description: str = ""
    linear_bits: float = 16.0
    group_size: int = 0
    head_bits: float = 16.0
    embed_on_gpu: bool = True
    experts_only: bool = False
    gguf_base: Optional[str] = None
    params: Dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def tensor_type(self, t: Tensor, qc: QuantContext) -> str:
        """Storage dtype name this scheme assigns to tensor ``t``."""
        if self.family == "gguf":
            return _gguf_tensor_type(self.name, self.gguf_base or "F16", t, qc)
        if self.family == "fp":
            return self.name
        if t.is_1d or t.kind in ("norm", "bias", "router", "vision", "other", "ssm_other"):
            return "BF16"
        if t.kind == "head":
            return f"{self.head_bits:g}bit" if self.head_bits < 16 else "BF16"
        if t.kind == "embed":
            return "BF16"
        if self.experts_only and not t.is_expert:
            return "BF16"
        if not t.is_linear:
            return "BF16"
        return self.name

    def tensor_bytes(self, t: Tensor, qc: QuantContext) -> int:
        if self.family == "gguf":
            if t.kind in ("vision", "other") and t.layer is None and t.n_out <= 1:
                return t.numel * 2  # projector / extra towers live in a separate f16 mmproj file
            tname = _gguf_tensor_type(self.name, self.gguf_base or "F16", t, qc)
            return ggml_type(tname).nbytes(t.numel)
        if self.family == "fp":
            width = 4 if self.name == "F32" else 2
            return t.numel * width
        if t.is_1d or t.kind in ("norm", "bias", "router", "vision", "other", "ssm_other", "embed"):
            return t.numel * 2
        if t.kind == "head":
            return int(round(t.numel * self.head_bits / 8))
        if not t.is_linear or (self.experts_only and not t.is_expert):
            return t.numel * 2
        rows = t.n_out * t.experts
        bits = self.linear_bits
        if self.family in ("awq", "gptq"):
            groups = -(-t.n_in // self.group_size)
            zero_bits = bits
            weight = t.numel * bits / 8
            scales = rows * groups * 2
            zeros = rows * groups * zero_bits / 8
            g_idx = t.n_in * 4 * t.experts if self.family == "gptq" else 0
            return int(weight + scales + zeros + g_idx)
        if self.family in ("fp8", "int8"):
            return t.numel + rows * 4  # 1 byte/weight + fp32 per-output-channel scale
        if self.family == "nf4":
            blocks = -(-t.numel // 64)
            # double quantization: 8-bit absmax per 64 weights + fp32 per 256 absmax blocks
            return int(t.numel / 2 + blocks + -(-blocks // 256) * 4)
        if self.family == "mxfp4":
            return int(t.numel * 4.25 / 8)
        if self.family == "nvfp4":
            return int(t.numel * 4.5 / 8) + 4
        if self.family == "exl2":
            return int(t.numel * bits / 8)
        return int(t.numel * bits / 8)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "family": self.family,
            "label": self.label,
            "bpw": round(self.nominal_bpw, 3),
            "engines": list(self.engines),
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# llama.cpp mixing rules (src/llama-quant.cpp: llama_tensor_get_type)
# ---------------------------------------------------------------------------

def _use_more_bits(i: int, n: int) -> bool:
    return i < n // 8 or i >= 7 * n // 8 or (i - n // 8) % 3 == 2


_GGUF_FALLBACK = {
    "IQ2_XXS": "IQ4_NL", "IQ2_XS": "IQ4_NL", "IQ2_S": "IQ4_NL", "IQ3_XXS": "IQ4_NL",
    "IQ3_S": "IQ4_NL", "IQ1_S": "IQ4_NL", "IQ1_M": "IQ4_NL", "Q2_K": "IQ4_NL",
    "Q3_K": "IQ4_NL", "IQ4_XS": "IQ4_NL", "TQ1_0": "IQ4_NL", "TQ2_0": "IQ4_NL",
    "Q4_K": "Q5_0", "Q5_K": "Q5_1", "Q6_K": "Q8_0",
}
_IQ_LOW = {"IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ1_S", "IQ1_M"}
_IQ3 = {"IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M"}


def _gguf_tensor_type(ftype: str, base: str, t: Tensor, qc: QuantContext) -> str:
    if t.is_1d or t.kind in ("norm", "bias"):
        return "F32"
    if ftype in ("F32", "F16", "BF16"):
        return ftype
    if t.kind == "router":
        return "F32" if t.numel < 1_000_000 else base
    n = max(qc.n_layer, 1)
    i = t.layer if t.layer is not None else 0
    new = base
    is_output = t.kind == "head" or (t.kind == "embed" and not qc.has_output)
    if ftype == "MXFP4_MOE":
        if t.is_expert:
            return "MXFP4"
        new = "Q8_0"
        is_output = False
    if is_output:
        if ftype == "Q8_0":
            new = "Q8_0"
        elif ftype in _IQ_LOW:
            new = "Q5_K"
        elif ftype in ("Q4_0", "Q4_1", "Q5_0", "Q5_1"):
            new = "Q6_K"
        else:
            new = "Q6_K"
        if t.n_in % 256 != 0:
            new = "Q8_0"
    elif t.kind == "embed":
        if ftype in _IQ_LOW:
            new = "Q2_K"
        elif ftype in _IQ3:
            new = "IQ3_S"
        elif ftype in ("TQ1_0", "TQ2_0"):
            new = "Q4_K"
    elif t.kind in ("attn_v", "attn_qkv", "attn_kv_b", "attn_v_b"):
        if ftype in ("Q2_K", "Q2_K_S"):
            new = "Q4_K" if qc.n_gqa >= 4 else "Q3_K"
        elif ftype in _IQ_LOW:
            new = "Q4_K" if qc.n_gqa >= 4 else base
        elif ftype in _IQ3:
            new = "Q4_K" if qc.n_gqa >= 4 else "IQ3_S"
        elif ftype == "Q3_K_M":
            new = "Q5_K" if i < 2 else "Q4_K"
        elif ftype == "Q3_K_L":
            new = "Q5_K"
        elif ftype in ("Q4_K_M", "Q5_K_M") and _use_more_bits(i, n):
            new = "Q6_K"
        elif ftype == "Q4_K_S" and i < 4:
            new = "Q5_K"
        elif ftype == "IQ4_XS" and qc.n_gqa >= 4:
            new = "Q5_K" if qc.n_expert >= 4 else base
        if qc.n_expert >= 8 and new not in ("Q8_0",):
            new = "Q8_0"
    elif t.kind in ("attn_k", "attn_kv_a", "attn_k_b"):
        if qc.n_expert >= 8:
            new = "Q8_0"
        elif ftype in ("IQ3_XS", "IQ3_XXS"):
            new = "IQ3_XXS"
    elif t.kind in ("ffn_down", "ffn_down_exps", "ffn_down_shexp"):
        if ftype == "Q2_K":
            new = "Q3_K"
        elif ftype == "Q2_K_S":
            new = "Q4_K" if i < n // 8 else base
        elif ftype == "Q3_K_M":
            new = "Q5_K" if i < n // 16 else ("Q4_K" if _use_more_bits(i, n) else "Q3_K")
        elif ftype == "Q3_K_L":
            new = "Q5_K"
        elif ftype in ("Q4_K_M", "Q5_K_M"):
            new = "Q6_K" if _use_more_bits(i, n) else base
        elif ftype == "Q4_K_S":
            new = "Q5_K" if i < n // 8 else base
        elif ftype in ("IQ3_M", "IQ3_S") and i < n // 8:
            new = "Q4_K"
        elif ftype in ("IQ2_S", "IQ2_M") and i < n // 8:
            new = "IQ3_S"
    elif t.kind == "attn_o":
        if ftype == "Q2_K":
            new = "Q3_K"
        elif ftype == "Q3_K_M":
            new = "Q4_K"
        elif ftype == "Q3_K_L":
            new = "Q5_K"
        elif qc.n_expert >= 8 and ftype not in _IQ_LOW and ftype not in ("Q8_0",):
            new = "Q5_K" if base in ("Q2_K", "Q3_K", "Q4_K", "IQ4_XS", "IQ4_NL") else base

    blk = ggml_type(new).block_size
    if t.n_in % blk != 0:
        new = _GGUF_FALLBACK.get(new, new)
        if t.n_in % ggml_type(new).block_size != 0:
            new = "F16"
    return new


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _g(name, base, bpw, desc):
    return QuantScheme(name=name, family="gguf", label=name, nominal_bpw=bpw,
                       engines=list(GGUF_ENGINES), description=desc, gguf_base=base)


_SCHEMES: List[QuantScheme] = [
    QuantScheme("F32", "fp", "FP32", 32.0, list(GGUF_ENGINES) + ["vllm", "transformers"],
                "Full single precision. Only for reference / CPU debugging."),
    QuantScheme("BF16", "fp", "BF16", 16.0,
                list(GGUF_ENGINES) + ["vllm", "exllamav2", "unsloth", "transformers"],
                "Unquantized bfloat16 (native training dtype)."),
    QuantScheme("F16", "fp", "FP16", 16.0,
                list(GGUF_ENGINES) + ["vllm", "exllamav2", "unsloth", "transformers"],
                "Unquantized float16 (pre-Ampere GPUs without bf16)."),
    _g("Q8_0", "Q8_0", 8.5, "8-bit round-to-nearest; visually lossless."),
    _g("Q6_K", "Q6_K", 6.56, "6-bit k-quant; near-lossless."),
    _g("Q5_K_M", "Q5_K", 5.69, "5-bit k-quant mix; excellent quality/size."),
    _g("Q5_K_S", "Q5_K", 5.54, "5-bit k-quant, smaller mix."),
    _g("Q5_0", "Q5_0", 5.5, "Legacy 5-bit."),
    _g("Q4_K_M", "Q4_K", 4.89, "4-bit k-quant mix; the community default."),
    _g("Q4_K_S", "Q4_K", 4.58, "4-bit k-quant, smaller mix."),
    _g("Q5_1", "Q5_1", 6.0, "Legacy 5-bit with offsets."),
    _g("Q4_1", "Q4_1", 5.0, "Legacy 4-bit with offsets."),
    _g("Q4_0", "Q4_0", 4.5, "Legacy 4-bit; fastest on ARM/Metal repack."),
    _g("IQ4_NL", "IQ4_NL", 4.5, "4-bit non-linear i-quant."),
    _g("IQ4_XS", "IQ4_XS", 4.25, "4.25-bit i-quant; best 4-bit size/quality."),
    _g("Q3_K_L", "Q3_K", 4.27, "3-bit k-quant with 5-bit attention/ffn_down."),
    _g("Q3_K_M", "Q3_K", 3.91, "3-bit k-quant mix."),
    _g("IQ3_M", "IQ3_S", 3.66, "3.66-bit i-quant mix."),
    _g("Q3_K_S", "Q3_K", 3.5, "3-bit k-quant, smallest mix."),
    _g("IQ3_S", "IQ3_S", 3.44, "3.44-bit i-quant."),
    _g("IQ3_XS", "IQ3_S", 3.3, "3.3-bit i-quant mix."),
    _g("IQ3_XXS", "IQ3_XXS", 3.06, "3.06-bit i-quant (needs imatrix)."),
    _g("Q2_K", "Q2_K", 2.96, "2-bit k-quant mix; large quality loss."),
    _g("Q2_K_S", "Q2_K", 2.7, "2-bit k-quant, smaller mix."),
    _g("IQ2_M", "IQ2_S", 2.7, "2.7-bit i-quant (needs imatrix)."),
    _g("IQ2_S", "IQ2_S", 2.5, "2.5-bit i-quant (needs imatrix)."),
    _g("IQ2_XS", "IQ2_XS", 2.31, "2.31-bit i-quant (needs imatrix)."),
    _g("IQ2_XXS", "IQ2_XXS", 2.06, "2.06-bit i-quant (needs imatrix)."),
    _g("IQ1_M", "IQ1_M", 1.75, "1.75-bit i-quant; only for huge MoE models."),
    _g("IQ1_S", "IQ1_S", 1.56, "1.56-bit i-quant; only for huge MoE models."),
    _g("TQ2_0", "TQ2_0", 2.06, "Ternary 2-bit packing (BitNet-style models)."),
    _g("TQ1_0", "TQ1_0", 1.69, "Ternary 1.69-bit packing (BitNet-style models)."),
    _g("MXFP4_MOE", "Q8_0", 4.25, "MXFP4 experts + Q8_0 dense (gpt-oss GGUF)."),
    QuantScheme("FP8", "fp8", "FP8 (W8A16/W8A8)", 8.0, ["vllm", "transformers"],
                "8-bit float weights with per-channel scales; native on Ada/Hopper/Blackwell.",
                linear_bits=8),
    QuantScheme("INT8", "int8", "INT8 W8A8 / LLM.int8()", 8.0, ["vllm", "transformers"],
                "8-bit integer weights with per-channel scales.", linear_bits=8),
    QuantScheme("AWQ", "awq", "AWQ 4-bit g128", 4.16, ["vllm", "transformers"],
                "Activation-aware 4-bit, group size 128 (Marlin kernels).",
                linear_bits=4, group_size=128),
    QuantScheme("GPTQ-4bit", "gptq", "GPTQ 4-bit g128", 4.16, ["vllm", "transformers"],
                "GPTQ 4-bit, group size 128.", linear_bits=4, group_size=128),
    QuantScheme("GPTQ-8bit", "gptq", "GPTQ 8-bit g128", 8.19, ["vllm", "transformers"],
                "GPTQ 8-bit, group size 128.", linear_bits=8, group_size=128),
    QuantScheme("NF4", "nf4", "bitsandbytes NF4 (QLoRA)", 4.13,
                ["unsloth", "transformers", "vllm"],
                "4-bit NormalFloat with double quantization, as used by QLoRA.",
                linear_bits=4),
    QuantScheme("MXFP4", "mxfp4", "MXFP4 (experts)", 4.25, ["vllm", "transformers"],
                "OCP microscaling FP4 on MoE experts, BF16 elsewhere (gpt-oss).",
                linear_bits=4.25, experts_only=True),
    QuantScheme("NVFP4", "nvfp4", "NVFP4", 4.5, ["vllm"],
                "NVIDIA FP4 with FP8 block scales (Blackwell).", linear_bits=4.5),
]

for _bpw in (2.5, 3.0, 3.5, 4.0, 4.25, 4.65, 5.0, 6.0, 8.0):
    _SCHEMES.append(QuantScheme(
        f"EXL2-{_bpw:g}", "exl2", f"EXL2 {_bpw:g} bpw", _bpw, ["exllamav2"],
        f"ExLlamaV2 mixed-precision at {_bpw:g} bits/weight (6-bit head, embeddings in RAM).",
        linear_bits=_bpw, head_bits=6.0, embed_on_gpu=False))

QUANTS: Dict[str, QuantScheme] = {q.name: q for q in _SCHEMES}

_ALIASES = {
    "FP16": "F16", "FLOAT16": "F16", "HALF": "F16", "BFLOAT16": "BF16", "FP32": "F32",
    "FLOAT32": "F32", "Q8": "Q8_0", "Q6": "Q6_K", "Q5": "Q5_K_M", "Q4": "Q4_K_M",
    "Q4_K": "Q4_K_M", "Q5_K": "Q5_K_M", "Q3_K": "Q3_K_M", "Q3": "Q3_K_M", "Q2": "Q2_K",
    "GPTQ": "GPTQ-4bit", "GPTQ4": "GPTQ-4bit", "GPTQ-INT4": "GPTQ-4bit", "GPTQ8": "GPTQ-8bit",
    "GPTQ-INT8": "GPTQ-8bit", "AWQ-4BIT": "AWQ", "AWQ4": "AWQ", "BNB-4BIT": "NF4", "BNB4": "NF4",
    "4BIT": "NF4", "QLORA": "NF4", "BNB-8BIT": "INT8", "8BIT": "INT8", "W8A8": "INT8",
    "FP8-DYNAMIC": "FP8", "F8": "FP8", "E4M3": "FP8", "W4A16": "GPTQ-4bit",
}


def get_quant(name: Optional[str]) -> QuantScheme:
    if name is None:
        raise KeyError("quantization name is required")
    key = str(name).strip().upper().replace(" ", "")
    if key in _ALIASES:
        key = _ALIASES[key]
    for k, q in QUANTS.items():
        if k.upper() == key:
            return q
    m = re.fullmatch(r"EXL2[-_]?(\d+(?:\.\d+)?)(?:BPW)?", key) or re.fullmatch(r"(\d+(?:\.\d+)?)BPW", key)
    if m:
        bpw = float(m.group(1))
        if not 2.0 <= bpw <= 8.0:
            raise KeyError("EXL2 bits/weight must be between 2.0 and 8.0")
        return QuantScheme(f"EXL2-{bpw:g}", "exl2", f"EXL2 {bpw:g} bpw", bpw, ["exllamav2"],
                           f"ExLlamaV2 at {bpw:g} bits/weight.", linear_bits=bpw,
                           head_bits=6.0, embed_on_gpu=False)
    raise KeyError(f"unknown quantization {name!r}; run `vram-fit quants` for the list")


def quants_for_engine(engine: str) -> List[QuantScheme]:
    return [q for q in _SCHEMES if engine in q.engines]


def default_quant(engine: str, mode: str = "inference") -> str:
    if mode == "qlora":
        return "NF4"
    if mode in ("lora", "full"):
        return "BF16"
    return {
        "llama.cpp": "Q4_K_M", "ollama": "Q4_K_M", "vllm": "BF16",
        "exllamav2": "EXL2-4.0", "unsloth": "BF16", "transformers": "BF16",
    }.get(engine, "Q4_K_M")


def quant_from_hf_config(qcfg: Optional[dict]) -> Optional[str]:
    """Infer the scheme of a pre-quantized checkpoint from ``quantization_config``."""
    if not qcfg:
        return None
    method = str(qcfg.get("quant_method", "")).lower()
    bits = qcfg.get("bits") or qcfg.get("w_bit")
    if method == "awq":
        return "AWQ"
    if method == "gptq":
        return "GPTQ-8bit" if bits == 8 else "GPTQ-4bit"
    if method in ("fp8", "fbgemm_fp8"):
        return "FP8"
    if method == "mxfp4":
        return "MXFP4"
    if method == "bitsandbytes":
        if qcfg.get("load_in_4bit") or qcfg.get("_load_in_4bit"):
            return "NF4"
        return "INT8"
    if method == "exl2":
        return f"EXL2-{float(qcfg.get('bits', 4.0)):g}"
    if method in ("compressed-tensors", "compressed_tensors"):
        groups = qcfg.get("config_groups") or {}
        for g in groups.values():
            w = (g or {}).get("weights") or {}
            nb = w.get("num_bits")
            typ = str(w.get("type", "")).lower()
            if nb == 4 and typ == "float":
                return "NVFP4"
            if nb == 4:
                return "GPTQ-4bit"
            if nb == 8 and typ == "float":
                return "FP8"
            if nb == 8:
                return "INT8"
        fmt = str(qcfg.get("format", "")).lower()
        if "fp8" in fmt or "float" in fmt:
            return "FP8"
        return "INT8"
    return None


def quant_from_filename(filename: str) -> Optional[str]:
    """Extract a GGUF quant name from a file name like ``Qwen2.5-14B-Instruct-Q4_K_M.gguf``."""
    stem = filename.rsplit("/", 1)[-1]
    upper = stem.upper()
    candidates = sorted((q.name for q in _SCHEMES if q.family in ("gguf", "fp")), key=len, reverse=True)
    for c in candidates:
        if re.search(r"(?<![A-Z0-9])" + re.escape(c) + r"(?![A-Z0-9])", upper.replace(".", "-")):
            return c
    if "Q4_0_4_" in upper or "Q4_0_8_8" in upper:
        return "Q4_0"
    return None
