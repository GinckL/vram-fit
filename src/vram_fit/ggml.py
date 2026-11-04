"""ggml tensor type table and llama.cpp file-type (ftype) definitions.

Block sizes and byte sizes mirror ``ggml/src/ggml.c`` (``type_traits``), so a
tensor's on-disk / in-VRAM size is ``numel / block_size * type_size`` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class GGMLType:
    id: int
    name: str
    block_size: int
    type_size: int

    @property
    def bits_per_weight(self) -> float:
        return self.type_size * 8.0 / self.block_size

    def nbytes(self, numel: int) -> int:
        blocks = -(-numel // self.block_size)  # ceil
        return blocks * self.type_size


_TYPES = [
    GGMLType(0, "F32", 1, 4),
    GGMLType(1, "F16", 1, 2),
    GGMLType(2, "Q4_0", 32, 18),
    GGMLType(3, "Q4_1", 32, 20),
    GGMLType(6, "Q5_0", 32, 22),
    GGMLType(7, "Q5_1", 32, 24),
    GGMLType(8, "Q8_0", 32, 34),
    GGMLType(9, "Q8_1", 32, 36),
    GGMLType(10, "Q2_K", 256, 84),
    GGMLType(11, "Q3_K", 256, 110),
    GGMLType(12, "Q4_K", 256, 144),
    GGMLType(13, "Q5_K", 256, 176),
    GGMLType(14, "Q6_K", 256, 210),
    GGMLType(15, "Q8_K", 256, 292),
    GGMLType(16, "IQ2_XXS", 256, 66),
    GGMLType(17, "IQ2_XS", 256, 74),
    GGMLType(18, "IQ3_XXS", 256, 98),
    GGMLType(19, "IQ1_S", 256, 50),
    GGMLType(20, "IQ4_NL", 32, 18),
    GGMLType(21, "IQ3_S", 256, 110),
    GGMLType(22, "IQ2_S", 256, 82),
    GGMLType(23, "IQ4_XS", 256, 136),
    GGMLType(24, "I8", 1, 1),
    GGMLType(25, "I16", 1, 2),
    GGMLType(26, "I32", 1, 4),
    GGMLType(27, "I64", 1, 8),
    GGMLType(28, "F64", 1, 8),
    GGMLType(29, "IQ1_M", 256, 56),
    GGMLType(30, "BF16", 1, 2),
    # 31-33 were Q4_0_4_4 / Q4_0_4_8 / Q4_0_8_8 (repacked Q4_0 layouts)
    GGMLType(31, "Q4_0_4_4", 32, 18),
    GGMLType(32, "Q4_0_4_8", 32, 18),
    GGMLType(33, "Q4_0_8_8", 32, 18),
    GGMLType(34, "TQ1_0", 256, 54),
    GGMLType(35, "TQ2_0", 256, 66),
    GGMLType(36, "IQ4_NL_4_4", 32, 18),
    GGMLType(37, "IQ4_NL_4_8", 32, 18),
    GGMLType(38, "IQ4_NL_8_8", 32, 18),
    GGMLType(39, "MXFP4", 32, 17),
]

TYPES_BY_ID: Dict[int, GGMLType] = {t.id: t for t in _TYPES}
TYPES_BY_NAME: Dict[str, GGMLType] = {t.name: t for t in _TYPES}


def ggml_type(name_or_id) -> GGMLType:
    if isinstance(name_or_id, int):
        if name_or_id not in TYPES_BY_ID:
            raise KeyError(f"unknown ggml type id {name_or_id}")
        return TYPES_BY_ID[name_or_id]
    key = str(name_or_id).upper()
    if key not in TYPES_BY_NAME:
        raise KeyError(f"unknown ggml type {name_or_id!r}")
    return TYPES_BY_NAME[key]


# llama_ftype enum (include/llama.h) -> canonical quant name used by vram-fit.
FTYPE_NAMES: Dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
    36: "TQ1_0",
    37: "TQ2_0",
    38: "MXFP4_MOE",
}


def ftype_name(ftype: Optional[int]) -> Optional[str]:
    if ftype is None:
        return None
    return FTYPE_NAMES.get(int(ftype) & 0x3FF)  # strip LLAMA_FTYPE_GUESSED bit (1024)


# KV-cache element types (llama.cpp -ctk / -ctv) -> bytes per element.
KV_CACHE_TYPES = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34 / 32,
    "q5_1": 24 / 32,
    "q5_0": 22 / 32,
    "q4_1": 20 / 32,
    "q4_0": 18 / 32,
    "iq4_nl": 18 / 32,
}
