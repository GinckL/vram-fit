"""GPU / accelerator preset database.

Bandwidth is the vendor-rated peak memory bandwidth (GB/s). ``tflops`` is
dense FP16/BF16 tensor throughput with FP32 accumulation (what PyTorch and
llama.cpp matmuls actually get), not the sparse marketing number.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class GPU:
    id: str
    name: str
    vendor: str              # nvidia | amd | apple | intel | cpu
    vram_gib: float
    bandwidth_gbs: float
    tflops: float
    arch: str = ""
    compute_capability: Optional[str] = None
    pcie_gen: int = 4
    pcie_width: int = 16
    unified: bool = False
    form: str = "desktop"    # desktop | laptop | datacenter | apple | apu
    count: int = 1
    vram_free_gib: Optional[float] = None   # measured (probe) free memory
    ceiling_gib: Optional[float] = None     # usable limit (e.g. macOS wired limit)
    probed: bool = False
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def vram_mib(self) -> float:
        return self.vram_gib * 1024

    @property
    def usable_gib(self) -> float:
        """Per-device memory an inference engine may allocate."""
        return self.ceiling_gib if self.ceiling_gib is not None else self.vram_gib

    @property
    def cc(self) -> float:
        try:
            return float(self.compute_capability) if self.compute_capability else 0.0
        except ValueError:
            return 0.0

    @property
    def supports_bf16(self) -> bool:
        if self.vendor == "nvidia":
            return self.cc >= 8.0
        return self.vendor in ("amd", "apple", "intel")

    @property
    def supports_fp8(self) -> bool:
        return (self.vendor == "nvidia" and self.cc >= 8.9) or self.arch in ("CDNA3", "RDNA4")

    @property
    def supports_flash_attn2(self) -> bool:
        return (self.vendor == "nvidia" and self.cc >= 8.0) or self.arch in ("CDNA2", "CDNA3")

    @property
    def pcie_gbs(self) -> float:
        per_lane = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.563}.get(self.pcie_gen, 1.969)
        return per_lane * self.pcie_width

    def to_dict(self) -> dict:
        d = asdict(self)
        d["vram_mib"] = round(self.vram_mib)
        d["usable_mib"] = round(self.usable_gib * 1024)
        d["pcie_gbs"] = round(self.pcie_gbs, 1)
        d["supports_bf16"] = self.supports_bf16
        d["supports_fp8"] = self.supports_fp8
        d["supports_flash_attn2"] = self.supports_flash_attn2
        return d


def _nv(id, name, vram, bw, tf, arch, cc, gen=4, width=16, form="desktop"):
    return GPU(id, name, "nvidia", vram, bw, tf, arch, cc, gen, width, form=form)


def _amd(id, name, vram, bw, tf, arch, gen=4, width=16, form="desktop"):
    return GPU(id, name, "amd", vram, bw, tf, arch, None, gen, width, form=form)


def _apple(id, name, ram, bw, tf):
    limit = ram * (0.75 if ram > 36 else 2 / 3)
    return GPU(id, name, "apple", ram, bw, tf, "Apple GPU", None, 0, 0, unified=True, form="apple",
               ceiling_gib=round(limit, 2))


PRESETS: List[GPU] = [
    # NVIDIA laptop
    _nv("rtx-3050-laptop-4gb", "RTX 3050 Laptop 4GB", 4, 192, 16.0, "Ampere", "8.6", 4, 8, "laptop"),
    _nv("rtx-3060-laptop-6gb", "RTX 3060 Laptop 6GB", 6, 336, 21.9, "Ampere", "8.6", 4, 16, "laptop"),
    _nv("rtx-4050-laptop-6gb", "RTX 4050 Laptop 6GB", 6, 192, 22.0, "Ada", "8.9", 4, 8, "laptop"),
    _nv("rtx-4060-laptop-8gb", "RTX 4060 Laptop 8GB", 8, 256, 30.0, "Ada", "8.9", 4, 8, "laptop"),
    _nv("rtx-4070-laptop-8gb", "RTX 4070 Laptop 8GB", 8, 256, 37.0, "Ada", "8.9", 4, 8, "laptop"),
    _nv("rtx-4080-laptop-12gb", "RTX 4080 Laptop 12GB", 12, 432, 70.0, "Ada", "8.9", 4, 16, "laptop"),
    _nv("rtx-4090-laptop-16gb", "RTX 4090 Laptop 16GB", 16, 576, 80.0, "Ada", "8.9", 4, 16, "laptop"),
    _nv("rtx-5090-laptop-24gb", "RTX 5090 Laptop 24GB", 24, 896, 105.0, "Blackwell", "12.0", 5, 16, "laptop"),
    # NVIDIA consumer desktop
    _nv("gtx-1080-ti", "GTX 1080 Ti 11GB", 11, 484, 11.3, "Pascal", "6.1", 3),
    _nv("rtx-2060-12gb", "RTX 2060 12GB", 12, 336, 28.7, "Turing", "7.5", 3),
    _nv("rtx-2080-ti", "RTX 2080 Ti 11GB", 11, 616, 53.8, "Turing", "7.5", 3),
    _nv("rtx-3060-8gb", "RTX 3060 8GB", 8, 240, 25.6, "Ampere", "8.6"),
    _nv("rtx-3060-12gb", "RTX 3060 12GB", 12, 360, 25.6, "Ampere", "8.6"),
    _nv("rtx-3060-ti", "RTX 3060 Ti 8GB", 8, 448, 32.4, "Ampere", "8.6"),
    _nv("rtx-3070", "RTX 3070 8GB", 8, 448, 40.6, "Ampere", "8.6"),
    _nv("rtx-3080-10gb", "RTX 3080 10GB", 10, 760, 59.5, "Ampere", "8.6"),
    _nv("rtx-3080-12gb", "RTX 3080 12GB", 12, 912, 61.0, "Ampere", "8.6"),
    _nv("rtx-3090", "RTX 3090 24GB", 24, 936, 71.0, "Ampere", "8.6"),
    _nv("rtx-3090-ti", "RTX 3090 Ti 24GB", 24, 1008, 80.0, "Ampere", "8.6"),
    _nv("rtx-4060", "RTX 4060 8GB", 8, 272, 30.2, "Ada", "8.9", 4, 8),
    _nv("rtx-4060-ti-8gb", "RTX 4060 Ti 8GB", 8, 288, 44.2, "Ada", "8.9", 4, 8),
    _nv("rtx-4060-ti-16gb", "RTX 4060 Ti 16GB", 16, 288, 44.2, "Ada", "8.9", 4, 8),
    _nv("rtx-4070", "RTX 4070 12GB", 12, 504, 58.3, "Ada", "8.9"),
    _nv("rtx-4070-super", "RTX 4070 Super 12GB", 12, 504, 71.0, "Ada", "8.9"),
    _nv("rtx-4070-ti", "RTX 4070 Ti 12GB", 12, 504, 80.2, "Ada", "8.9"),
    _nv("rtx-4070-ti-super", "RTX 4070 Ti Super 16GB", 16, 672, 88.0, "Ada", "8.9"),
    _nv("rtx-4080", "RTX 4080 16GB", 16, 717, 97.5, "Ada", "8.9"),
    _nv("rtx-4080-super", "RTX 4080 Super 16GB", 16, 736, 104.0, "Ada", "8.9"),
    _nv("rtx-4090", "RTX 4090 24GB", 24, 1008, 165.2, "Ada", "8.9"),
    _nv("rtx-5060-ti-16gb", "RTX 5060 Ti 16GB", 16, 448, 47.4, "Blackwell", "12.0", 5, 8),
    _nv("rtx-5070", "RTX 5070 12GB", 12, 672, 61.7, "Blackwell", "12.0", 5),
    _nv("rtx-5070-ti", "RTX 5070 Ti 16GB", 16, 896, 87.9, "Blackwell", "12.0", 5),
    _nv("rtx-5080", "RTX 5080 16GB", 16, 960, 112.6, "Blackwell", "12.0", 5),
    _nv("rtx-5090", "RTX 5090 32GB", 32, 1792, 209.5, "Blackwell", "12.0", 5),
    # NVIDIA workstation
    _nv("rtx-a4000", "RTX A4000 16GB", 16, 448, 76.7, "Ampere", "8.6"),
    _nv("rtx-a5000", "RTX A5000 24GB", 24, 768, 111.1, "Ampere", "8.6"),
    _nv("rtx-a6000", "RTX A6000 48GB", 48, 768, 154.8, "Ampere", "8.6"),
    _nv("rtx-4000-ada", "RTX 4000 Ada 20GB", 20, 360, 106.9, "Ada", "8.9"),
    _nv("rtx-6000-ada", "RTX 6000 Ada 48GB", 48, 960, 364.2, "Ada", "8.9"),
    _nv("rtx-pro-6000", "RTX PRO 6000 Blackwell 96GB", 96, 1792, 503.8, "Blackwell", "12.0", 5),
    # NVIDIA datacenter
    _nv("t4", "Tesla T4 16GB", 16, 320, 65.0, "Turing", "7.5", 3, 16, "datacenter"),
    _nv("v100-16gb", "Tesla V100 16GB", 16, 900, 125.0, "Volta", "7.0", 3, 16, "datacenter"),
    _nv("v100-32gb", "Tesla V100 32GB", 32, 900, 125.0, "Volta", "7.0", 3, 16, "datacenter"),
    _nv("a10", "A10 24GB", 24, 600, 125.0, "Ampere", "8.6", 4, 16, "datacenter"),
    _nv("a10g", "A10G 24GB", 24, 600, 70.0, "Ampere", "8.6", 4, 16, "datacenter"),
    _nv("l4", "L4 24GB", 24, 300, 121.0, "Ada", "8.9", 4, 16, "datacenter"),
    _nv("l40s", "L40S 48GB", 48, 864, 362.0, "Ada", "8.9", 4, 16, "datacenter"),
    _nv("a100-40gb", "A100 40GB", 40, 1555, 312.0, "Ampere", "8.0", 4, 16, "datacenter"),
    _nv("a100-80gb", "A100 80GB SXM", 80, 2039, 312.0, "Ampere", "8.0", 4, 16, "datacenter"),
    _nv("h100-pcie", "H100 PCIe 80GB", 80, 2000, 756.0, "Hopper", "9.0", 5, 16, "datacenter"),
    _nv("h100", "H100 SXM 80GB", 80, 3350, 989.0, "Hopper", "9.0", 5, 16, "datacenter"),
    _nv("h100-nvl", "H100 NVL 94GB", 94, 3900, 835.0, "Hopper", "9.0", 5, 16, "datacenter"),
    _nv("h200", "H200 141GB", 141, 4800, 989.0, "Hopper", "9.0", 5, 16, "datacenter"),
    _nv("b200", "B200 180GB", 180, 8000, 2250.0, "Blackwell", "10.0", 5, 16, "datacenter"),
    GPU("dgx-spark", "DGX Spark (GB10) 128GB", "nvidia", 128, 273, 125.0, "Blackwell", "12.1", 5, 16,
        unified=True, form="apu", ceiling_gib=119.0),
    # AMD
    _amd("rx-7600", "Radeon RX 7600 8GB", 8, 288, 43.5, "RDNA3", 4, 8),
    _amd("rx-7800-xt", "Radeon RX 7800 XT 16GB", 16, 624, 74.6, "RDNA3"),
    _amd("rx-7900-xt", "Radeon RX 7900 XT 20GB", 20, 800, 103.0, "RDNA3"),
    _amd("rx-7900-xtx", "Radeon RX 7900 XTX 24GB", 24, 960, 122.8, "RDNA3"),
    _amd("rx-9070-xt", "Radeon RX 9070 XT 16GB", 16, 640, 195.0, "RDNA4", 5),
    _amd("w7900", "Radeon PRO W7900 48GB", 48, 864, 122.6, "RDNA3"),
    _amd("mi210", "Instinct MI210 64GB", 64, 1638, 181.0, "CDNA2", 4, 16, "datacenter"),
    _amd("mi250x", "Instinct MI250X 128GB", 128, 3277, 383.0, "CDNA2", 4, 16, "datacenter"),
    _amd("mi300x", "Instinct MI300X 192GB", 192, 5300, 1307.0, "CDNA3", 5, 16, "datacenter"),
    _amd("mi325x", "Instinct MI325X 256GB", 256, 6000, 1307.0, "CDNA3", 5, 16, "datacenter"),
    GPU("strix-halo-128gb", "Ryzen AI Max+ 395 (Strix Halo) 128GB", "amd", 128, 256, 59.4, "RDNA3.5",
        None, 4, 4, unified=True, form="apu", ceiling_gib=110.0),
    # Intel
    GPU("arc-a770", "Intel Arc A770 16GB", "intel", 16, 560, 78.6, "Alchemist", None, 4, 16),
    GPU("arc-b580", "Intel Arc B580 12GB", "intel", 12, 456, 58.0, "Battlemage", None, 4, 8),
    # Apple Silicon (unified memory; ceiling = default macOS GPU wired limit)
    _apple("m1-8gb", "Apple M1 8GB", 8, 68.25, 2.6),
    _apple("m1-16gb", "Apple M1 16GB", 16, 68.25, 2.6),
    _apple("m1-pro-16gb", "Apple M1 Pro 16GB", 16, 200, 5.2),
    _apple("m1-pro-32gb", "Apple M1 Pro 32GB", 32, 200, 5.2),
    _apple("m1-max-32gb", "Apple M1 Max 32GB", 32, 400, 10.4),
    _apple("m1-max-64gb", "Apple M1 Max 64GB", 64, 400, 10.4),
    _apple("m1-ultra-128gb", "Apple M1 Ultra 128GB", 128, 800, 21.0),
    _apple("m2-8gb", "Apple M2 8GB", 8, 100, 3.6),
    _apple("m2-16gb", "Apple M2 16GB", 16, 100, 3.6),
    _apple("m2-24gb", "Apple M2 24GB", 24, 100, 3.6),
    _apple("m2-pro-32gb", "Apple M2 Pro 32GB", 32, 200, 6.8),
    _apple("m2-max-64gb", "Apple M2 Max 64GB", 64, 400, 13.6),
    _apple("m2-max-96gb", "Apple M2 Max 96GB", 96, 400, 13.6),
    _apple("m2-ultra-192gb", "Apple M2 Ultra 192GB", 192, 800, 27.2),
    _apple("m3-8gb", "Apple M3 8GB", 8, 100, 4.1),
    _apple("m3-16gb", "Apple M3 16GB", 16, 100, 4.1),
    _apple("m3-24gb", "Apple M3 24GB", 24, 100, 4.1),
    _apple("m3-pro-18gb", "Apple M3 Pro 18GB", 18, 150, 7.4),
    _apple("m3-pro-36gb", "Apple M3 Pro 36GB", 36, 150, 7.4),
    _apple("m3-max-36gb", "Apple M3 Max 36GB (30c)", 36, 300, 12.3),
    _apple("m3-max-64gb", "Apple M3 Max 64GB", 64, 400, 16.4),
    _apple("m3-max-128gb", "Apple M3 Max 128GB", 128, 400, 16.4),
    _apple("m3-ultra-96gb", "Apple M3 Ultra 96GB", 96, 819, 28.4),
    _apple("m3-ultra-256gb", "Apple M3 Ultra 256GB", 256, 819, 28.4),
    _apple("m3-ultra-512gb", "Apple M3 Ultra 512GB", 512, 819, 28.4),
    _apple("m4-16gb", "Apple M4 16GB", 16, 120, 4.3),
    _apple("m4-24gb", "Apple M4 24GB", 24, 120, 4.3),
    _apple("m4-32gb", "Apple M4 32GB", 32, 120, 4.3),
    _apple("m4-pro-24gb", "Apple M4 Pro 24GB", 24, 273, 9.2),
    _apple("m4-pro-48gb", "Apple M4 Pro 48GB", 48, 273, 9.2),
    _apple("m4-pro-64gb", "Apple M4 Pro 64GB", 64, 273, 9.2),
    _apple("m4-max-36gb", "Apple M4 Max 36GB (32c)", 36, 410, 14.7),
    _apple("m4-max-48gb", "Apple M4 Max 48GB", 48, 546, 18.4),
    _apple("m4-max-64gb", "Apple M4 Max 64GB", 64, 546, 18.4),
    _apple("m4-max-128gb", "Apple M4 Max 128GB", 128, 546, 18.4),
    _apple("m5-16gb", "Apple M5 16GB", 16, 153, 5.7),
    _apple("m5-32gb", "Apple M5 32GB", 32, 153, 5.7),
]

PRESETS_BY_ID: Dict[str, GPU] = {g.id: g for g in PRESETS}

# Apple chip name -> (bandwidth GB/s, fp16 TFLOPS); Max/Pro binning is refined by GPU core count.
APPLE_CHIPS = {
    "M1": (68.25, 2.6), "M1 Pro": (200, 5.2), "M1 Max": (400, 10.4), "M1 Ultra": (800, 21.0),
    "M2": (100, 3.6), "M2 Pro": (200, 6.8), "M2 Max": (400, 13.6), "M2 Ultra": (800, 27.2),
    "M3": (100, 4.1), "M3 Pro": (150, 7.4), "M3 Max": (400, 16.4), "M3 Ultra": (819, 28.4),
    "M4": (120, 4.3), "M4 Pro": (273, 9.2), "M4 Max": (546, 18.4), "M5": (153, 5.7),
}


def _norm(s: str) -> str:
    s = s.lower()
    s = s.replace("geforce", "").replace("nvidia", "").replace("radeon", "").replace("amd", "")
    s = re.sub(r"(\d+)\s*gb", r"\1gb", s)
    return re.sub(r"[^a-z0-9]+", "", s)


def find_preset(query: str) -> Optional[GPU]:
    """Fuzzy-match a GPU name (``4090``, ``rtx-3060-laptop-6gb``, ``H100 SXM``, ``m3 max 64gb``)."""
    q = query.strip().lower()
    if q in PRESETS_BY_ID:
        return PRESETS_BY_ID[q]
    nq = _norm(q)
    for g in PRESETS:
        if _norm(g.id) == nq or _norm(g.name) == nq:
            return g
    # token-subset match; prefer the shortest (most generic) name
    tokens = [t for t in re.split(r"[^a-z0-9]+", q.replace("gb", " gb ")) if t and t not in ("gb", "nvidia", "geforce", "amd", "radeon")]
    best = None
    for g in PRESETS:
        hay = _norm(g.id + " " + g.name)
        if all(t in hay for t in tokens):
            if best is None or len(g.name) < len(best.name):
                best = g
    if best is None:
        return None
    # disambiguate memory size suffix: "3060 12gb" vs "3060 8gb"
    mem = re.search(r"(\d+)\s*gb", q)
    if mem:
        for g in PRESETS:
            hay = _norm(g.id + " " + g.name)
            if all(t in hay for t in tokens) and abs(g.vram_gib - float(mem.group(1))) < 0.5:
                return g
    return best


def parse_gpu_spec(spec: str) -> GPU:
    """Parse ``"2x rtx-3090"`` / ``"rtx-4090"`` / ``"custom:24:1008"`` into a GPU."""
    s = spec.strip()
    count = 1
    m = re.match(r"^(\d+)\s*[x×\*]\s*(.+)$", s, re.IGNORECASE)
    if m:
        count = int(m.group(1))
        s = m.group(2)
    if s.lower().startswith("custom"):
        parts = s.split(":")
        vram = float(parts[1]) if len(parts) > 1 else 24.0
        bw = float(parts[2]) if len(parts) > 2 else 500.0
        tf = float(parts[3]) if len(parts) > 3 else 50.0
        g = GPU("custom", f"Custom {vram:g}GB", "nvidia", vram, bw, tf, "custom", "8.6")
    else:
        found = find_preset(s)
        if found is None:
            raise KeyError(f"unknown GPU {spec!r}; run `vram-fit presets` to list them")
        g = GPU(**{k: v for k, v in asdict(found).items()})
    g.count = count
    return g


def list_presets(vendor: Optional[str] = None) -> List[GPU]:
    return [g for g in PRESETS if vendor is None or g.vendor == vendor]
