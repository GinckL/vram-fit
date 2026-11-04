"""Local hardware probing and live telemetry.

* NVIDIA: NVML through ``ctypes`` (no pynvml dependency), with an
  ``nvidia-smi`` fallback. Memory bandwidth is derived from the max memory
  clock and bus width, PCIe generation/width from the link registers.
* AMD: ``/sys/class/drm`` (amdgpu) plus ``rocm-smi`` for product names.
* Apple Silicon: ``sysctl`` + ``system_profiler``; the GPU ceiling is the
  macOS wired-memory limit (``iogpu.wired_limit_mb``).
* System RAM / CPU: ``/proc``, ``sysctl`` or Win32 ``GlobalMemoryStatusEx``.
"""

from __future__ import annotations

import ctypes
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .presets import APPLE_CHIPS, GPU, find_preset

DEFAULT_RAM_BW_GBS = 50.0


@dataclass
class HardwareProfile:
    gpus: List[GPU]
    cpu_name: str
    cpu_physical_cores: int
    cpu_logical_cores: int
    ram_total_gib: float
    ram_available_gib: float
    ram_bandwidth_gbs: float
    os: str
    backend: str                 # cuda | rocm | metal | cpu
    driver: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gpus"] = [g.to_dict() for g in self.gpus]
        return d

    def primary(self) -> GPU:
        if not self.gpus:
            g = cpu_only_gpu(self.ram_bandwidth_gbs, self.cpu_physical_cores)
            g.probed = True
            return g
        g0 = self.gpus[0]
        same = [g for g in self.gpus if g.name == g0.name]
        if len(same) > 1:
            g = GPU(**{k: v for k, v in asdict(g0).items()})
            g.count = len(same)
            return g
        return g0


def cpu_only_gpu(ram_bw: float = DEFAULT_RAM_BW_GBS, cores: int = 8) -> GPU:
    return GPU("cpu", "CPU only (no GPU)", "cpu", 0.0, ram_bw, max(0.5, cores * 0.1), "cpu", None, 0, 0,
               form="cpu")


def _run(cmd: List[str], timeout: float = 5.0) -> Optional[str]:
    exe = shutil.which(cmd[0])
    if not exe:
        return None
    try:
        out = subprocess.run([exe] + cmd[1:], capture_output=True, text=True, timeout=timeout)
        return out.stdout if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


# ---------------------------------------------------------------------------
# NVIDIA via NVML (ctypes)
# ---------------------------------------------------------------------------

class _NvmlMemory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class _NvmlUtil(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class NVML:
    """Minimal NVML binding. Raises OSError if the library is unavailable."""

    _lib = None

    def __init__(self):
        if NVML._lib is None:
            names = ["libnvidia-ml.so.1", "libnvidia-ml.so"]
            if sys.platform == "win32":
                names = [os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "System32", "nvml.dll"),
                         os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                                      "NVIDIA Corporation", "NVSMI", "nvml.dll")]
            last = None
            for n in names:
                try:
                    NVML._lib = ctypes.CDLL(n)
                    break
                except OSError as e:
                    last = e
            if NVML._lib is None:
                raise OSError(f"NVML not found: {last}")
        self.lib = NVML._lib
        if self.lib.nvmlInit_v2() != 0:
            raise OSError("nvmlInit failed")

    def _call(self, fn: str, *args) -> bool:
        f = getattr(self.lib, fn, None)
        return f is not None and f(*args) == 0

    def count(self) -> int:
        c = ctypes.c_uint(0)
        return c.value if self._call("nvmlDeviceGetCount_v2", ctypes.byref(c)) else 0

    def handle(self, i: int):
        h = ctypes.c_void_p()
        if not self._call("nvmlDeviceGetHandleByIndex_v2", ctypes.c_uint(i), ctypes.byref(h)):
            raise OSError(f"cannot get NVML handle {i}")
        return h

    def uint(self, fn: str, h, *pre) -> Optional[int]:
        v = ctypes.c_uint(0)
        return v.value if self._call(fn, h, *pre, ctypes.byref(v)) else None

    def name(self, h) -> str:
        buf = ctypes.create_string_buffer(96)
        self._call("nvmlDeviceGetName", h, buf, ctypes.c_uint(96))
        return buf.value.decode(errors="replace")

    def memory(self, h) -> Optional[_NvmlMemory]:
        m = _NvmlMemory()
        return m if self._call("nvmlDeviceGetMemoryInfo", h, ctypes.byref(m)) else None

    def cc(self, h) -> Optional[str]:
        a, b = ctypes.c_int(0), ctypes.c_int(0)
        if self._call("nvmlDeviceGetCudaComputeCapability", h, ctypes.byref(a), ctypes.byref(b)):
            return f"{a.value}.{b.value}"
        return None

    def driver(self) -> Optional[str]:
        buf = ctypes.create_string_buffer(96)
        return buf.value.decode() if self._call("nvmlSystemGetDriverVersion", buf, ctypes.c_uint(96)) else None

    def utilization(self, h) -> Optional[_NvmlUtil]:
        u = _NvmlUtil()
        return u if self._call("nvmlDeviceGetUtilizationRates", h, ctypes.byref(u)) else None


def _nvidia_from_nvml() -> List[GPU]:
    nv = NVML()
    gpus = []
    for i in range(nv.count()):
        h = nv.handle(i)
        name = nv.name(h)
        mem = nv.memory(h)
        mem_clock = nv.uint("nvmlDeviceGetMaxClockInfo", h, ctypes.c_uint(2))
        bus = nv.uint("nvmlDeviceGetMemoryBusWidth", h)
        gen = nv.uint("nvmlDeviceGetMaxPcieLinkGeneration", h) or 4
        width = nv.uint("nvmlDeviceGetMaxPcieLinkWidth", h) or 16
        cur_gen = nv.uint("nvmlDeviceGetCurrPcieLinkGeneration", h)
        cur_width = nv.uint("nvmlDeviceGetCurrPcieLinkWidth", h)
        cc = nv.cc(h)
        total_gib = mem.total / 2**30 if mem else 0.0
        preset = find_preset(name.replace("NVIDIA", "").replace("GeForce", "") + f" {round(total_gib)}gb")
        bw = (mem_clock * 2 * bus / 8 / 1000.0) if mem_clock and bus else (preset.bandwidth_gbs if preset else 500.0)
        g = GPU(
            id=f"cuda:{i}", name=name, vendor="nvidia", vram_gib=round(total_gib, 3),
            bandwidth_gbs=round(bw, 1), tflops=preset.tflops if preset else max(10.0, bw / 10),
            arch=preset.arch if preset else "", compute_capability=cc,
            pcie_gen=gen, pcie_width=width,
            form="laptop" if "laptop" in name.lower() else (preset.form if preset else "desktop"),
            vram_free_gib=round(mem.free / 2**30, 3) if mem else None, probed=True,
            extra={"pcie_current_gen": cur_gen, "pcie_current_width": cur_width,
                   "used_mib": round(mem.used / 2**20) if mem else None, "index": i},
        )
        gpus.append(g)
    return gpus


def _nvidia_from_smi() -> List[GPU]:
    out = _run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,memory.used,"
                "pcie.link.gen.max,pcie.link.width.max,clocks.max.memory,compute_cap",
                "--format=csv,noheader,nounits"])
    if not out:
        return []
    gpus = []
    for line in out.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 9:
            continue

        def num(x, d=0.0):
            try:
                return float(x)
            except ValueError:
                return d
        total = num(f[2]) / 1024
        preset = find_preset(f[1].replace("NVIDIA", "").replace("GeForce", "") + f" {round(total)}gb")
        mem_clock = num(f[7])
        bw = preset.bandwidth_gbs if preset else 500.0
        gpus.append(GPU(
            id=f"cuda:{f[0]}", name=f[1], vendor="nvidia", vram_gib=round(total, 3), bandwidth_gbs=bw,
            tflops=preset.tflops if preset else bw / 10, arch=preset.arch if preset else "",
            compute_capability=f[8] if f[8] not in ("", "[N/A]") else None,
            pcie_gen=int(num(f[5], 4)) or 4, pcie_width=int(num(f[6], 16)) or 16,
            form="laptop" if "laptop" in f[1].lower() else (preset.form if preset else "desktop"),
            vram_free_gib=round(num(f[3]) / 1024, 3), probed=True,
            extra={"used_mib": num(f[4]), "max_mem_clock_mhz": mem_clock, "index": int(num(f[0]))},
        ))
    return gpus


# ---------------------------------------------------------------------------
# AMD via sysfs
# ---------------------------------------------------------------------------

_PCIE_SPEED_GEN = {"2.5": 1, "5.0": 2, "5": 2, "8.0": 3, "8": 3, "16.0": 4, "16": 4, "32.0": 5, "32": 5, "64.0": 6, "64": 6}


def _read(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return None


def _amd_cards() -> List[str]:
    cards = []
    for dev in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
        if re.search(r"card\d+-", dev):
            continue
        if _read(os.path.join(dev, "vendor")) == "0x1002" and _read(os.path.join(dev, "mem_info_vram_total")):
            cards.append(dev)
    return cards


def _amd_from_sysfs() -> List[GPU]:
    cards = _amd_cards()
    if not cards:
        return []
    names: Dict[int, str] = {}
    smi = _run(["rocm-smi", "--showproductname", "--json"])
    if smi:
        try:
            data = json.loads(smi)
            for k, v in data.items():
                m = re.search(r"(\d+)", k)
                if m and isinstance(v, dict):
                    names[int(m.group(1))] = v.get("Card Series") or v.get("Card series") or v.get("Card model") or ""
        except ValueError:
            pass
    gpus = []
    for i, dev in enumerate(cards):
        total = int(_read(os.path.join(dev, "mem_info_vram_total")) or 0)
        used = int(_read(os.path.join(dev, "mem_info_vram_used")) or 0)
        gtt = int(_read(os.path.join(dev, "mem_info_gtt_total")) or 0)
        name = names.get(i) or _read(os.path.join(dev, "product_name")) or f"AMD GPU {i}"
        preset = find_preset(name.replace("AMD", "").replace("Radeon", "") + f" {round(total / 2**30)}gb")
        speed = (_read(os.path.join(dev, "max_link_speed")) or "16.0 GT/s").split(" ")[0]
        width = int((_read(os.path.join(dev, "max_link_width")) or "16") or 16)
        unified = total < 4 * 2**30 and gtt > 16 * 2**30  # APU with small carve-out + large GTT
        vram = (gtt if unified else total) / 2**30
        gpus.append(GPU(
            id=f"rocm:{i}", name=name, vendor="amd", vram_gib=round(vram, 3),
            bandwidth_gbs=preset.bandwidth_gbs if preset else (120.0 if unified else 500.0),
            tflops=preset.tflops if preset else 40.0, arch=preset.arch if preset else "",
            pcie_gen=_PCIE_SPEED_GEN.get(speed, 4), pcie_width=width, unified=unified,
            form="apu" if unified else (preset.form if preset else "desktop"),
            vram_free_gib=round((total - used) / 2**30, 3) if not unified else None, probed=True,
            extra={"used_mib": round(used / 2**20), "sysfs": dev, "index": i},
        ))
    return gpus


# ---------------------------------------------------------------------------
# Apple Silicon
# ---------------------------------------------------------------------------

def _sysctl(key: str) -> Optional[str]:
    out = _run(["sysctl", "-n", key])
    return out.strip() if out else None


def _apple_gpu(ram_gib: float) -> Optional[GPU]:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return None
    brand = _sysctl("machdep.cpu.brand_string") or "Apple Silicon"
    chip = brand.replace("Apple ", "").strip()
    cores = None
    sp = _run(["system_profiler", "SPDisplaysDataType", "-json"], timeout=10)
    if sp:
        try:
            for d in json.loads(sp).get("SPDisplaysDataType", []):
                if d.get("sppci_cores"):
                    cores = int(d["sppci_cores"])
                    break
        except (ValueError, TypeError):
            pass
    key = max((k for k in APPLE_CHIPS if chip.startswith(k)), key=len, default=None)
    bw, tf = APPLE_CHIPS.get(key, (100.0, 4.0))
    if key == "M3 Max" and cores and cores <= 30:
        bw, tf = 300.0, 12.3
    if key == "M4 Max" and cores and cores <= 32:
        bw, tf = 410.0, 14.7
    wired = _sysctl("iogpu.wired_limit_mb")
    limit_gib = ram_gib * (0.75 if ram_gib > 36 else 2 / 3)
    note = "default macOS GPU wired limit"
    if wired and wired.isdigit() and int(wired) > 0:
        limit_gib = int(wired) / 1024
        note = "iogpu.wired_limit_mb"
    return GPU(id="metal:0", name=f"Apple {chip}" + (f" ({cores}-core GPU)" if cores else ""), vendor="apple",
               vram_gib=round(ram_gib, 2), bandwidth_gbs=bw, tflops=tf, arch="Apple GPU", pcie_gen=0, pcie_width=0,
               unified=True, form="apple", ceiling_gib=round(limit_gib, 2), probed=True,
               extra={"gpu_cores": cores, "ceiling_source": note})


# ---------------------------------------------------------------------------
# RAM / CPU
# ---------------------------------------------------------------------------

def _ram() -> tuple:
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        return vm.total / 2**30, vm.available / 2**30
    except ImportError:
        pass
    if sys.platform.startswith("linux"):
        info = {}
        for line in (_read("/proc/meminfo") or "").splitlines():
            k, _, v = line.partition(":")
            info[k] = int(v.strip().split()[0]) * 1024 if v.strip() else 0
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", total)
        return total / 2**30, avail / 2**30
    if sys.platform == "darwin":
        total = int(_sysctl("hw.memsize") or 0)
        vm = _run(["vm_stat"]) or ""
        page = int((re.search(r"page size of (\d+)", vm) or [None, 16384])[1])
        free = 0
        for k in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
            m = re.search(k + r":\s+(\d+)", vm)
            if m:
                free += int(m.group(1)) * page
        return total / 2**30, (free or total * 0.5) / 2**30
    if sys.platform == "win32":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))  # type: ignore[attr-defined]
        return st.ullTotalPhys / 2**30, st.ullAvailPhys / 2**30
    return 16.0, 8.0


def _cpu() -> tuple:
    logical = os.cpu_count() or 1
    physical = None
    name = platform.processor() or platform.machine()
    if sys.platform.startswith("linux"):
        txt = _read("/proc/cpuinfo") or ""
        pairs = set()
        phys = core = None
        for line in txt.splitlines():
            if line.startswith("model name") and name in ("", platform.machine()):
                name = line.split(":", 1)[1].strip()
            if line.startswith("physical id"):
                phys = line.split(":", 1)[1].strip()
            elif line.startswith("core id"):
                core = line.split(":", 1)[1].strip()
                pairs.add((phys, core))
        physical = len(pairs) or None
    elif sys.platform == "darwin":
        name = _sysctl("machdep.cpu.brand_string") or name
        p = _sysctl("hw.perflevel0.physicalcpu") or _sysctl("hw.physicalcpu")
        physical = int(p) if p and p.isdigit() else None
    else:
        try:
            import psutil  # type: ignore
            physical = psutil.cpu_count(logical=False)
        except ImportError:
            physical = None
    return name, physical or max(1, logical // 2), logical


def _ram_bandwidth(cores: int) -> float:
    env = os.environ.get("VRAM_FIT_RAM_BW")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    out = _run(["dmidecode", "-t", "memory"]) if sys.platform.startswith("linux") and os.geteuid() == 0 else None
    if out:
        speeds = [int(m) for m in re.findall(r"Configured (?:Memory|Clock) Speed:\s*(\d+)\s*MT/s", out)]
        if speeds:
            dimms = len(speeds)
            channels = min(dimms, 2) if cores <= 24 else min(dimms, 12)
            return round(max(speeds) * 8 * channels / 1000 * 0.75, 1)
    return DEFAULT_RAM_BW_GBS


def probe() -> HardwareProfile:
    notes: List[str] = []
    total, avail = _ram()
    cpu_name, phys, logical = _cpu()
    gpus: List[GPU] = []
    backend = "cpu"
    driver = None
    try:
        gpus = _nvidia_from_nvml()
        if gpus:
            backend = "cuda"
            try:
                driver = NVML().driver()
            except OSError:
                driver = None
    except OSError as e:
        notes.append(f"NVML unavailable ({e}); trying nvidia-smi")
        gpus = _nvidia_from_smi()
        backend = "cuda" if gpus else backend
    if not gpus:
        gpus = _amd_from_sysfs()
        if gpus:
            backend = "rocm"
    if not gpus:
        ag = _apple_gpu(total)
        if ag:
            gpus = [ag]
            backend = "metal"
    ram_bw = gpus[0].bandwidth_gbs if gpus and gpus[0].unified else _ram_bandwidth(phys)
    if not gpus:
        notes.append("No GPU detected: estimates assume CPU-only inference. Use --gpu to simulate one.")
    for g in gpus:
        used = (g.extra or {}).get("used_mib")
        if used and used > 256:
            notes.append(f"{g.name}: {used:.0f} MiB already in use by other processes (display / apps).")
    return HardwareProfile(gpus=gpus, cpu_name=cpu_name, cpu_physical_cores=phys, cpu_logical_cores=logical,
                           ram_total_gib=round(total, 2), ram_available_gib=round(avail, 2),
                           ram_bandwidth_gbs=ram_bw, os=f"{platform.system()} {platform.release()}",
                           backend=backend, driver=driver, notes=notes)


# ---------------------------------------------------------------------------
# Live telemetry (polled by the WebSocket server)
# ---------------------------------------------------------------------------

def sample_telemetry() -> Dict[str, Any]:
    out: Dict[str, Any] = {"gpus": []}
    total, avail = _ram()
    out["ram"] = {"total_mib": round(total * 1024), "used_mib": round((total - avail) * 1024)}
    try:
        nv = NVML()
        for i in range(nv.count()):
            h = nv.handle(i)
            mem = nv.memory(h)
            util = nv.utilization(h)
            temp = nv.uint("nvmlDeviceGetTemperature", h, ctypes.c_uint(0))
            power = nv.uint("nvmlDeviceGetPowerUsage", h)
            sm_clock = nv.uint("nvmlDeviceGetClockInfo", h, ctypes.c_uint(1))
            out["gpus"].append({
                "index": i, "name": nv.name(h),
                "used_mib": round(mem.used / 2**20) if mem else None,
                "total_mib": round(mem.total / 2**20) if mem else None,
                "util_pct": util.gpu if util else None,
                "mem_util_pct": util.memory if util else None,
                "temp_c": temp, "power_w": round(power / 1000, 1) if power else None,
                "sm_clock_mhz": sm_clock,
            })
        return out
    except OSError:
        pass
    for i, dev in enumerate(_amd_cards()):
        hw = (glob.glob(os.path.join(dev, "hwmon", "hwmon*")) or [None])[0]
        temp = _read(os.path.join(hw, "temp1_input")) if hw else None
        power = (_read(os.path.join(hw, "power1_average")) or _read(os.path.join(hw, "power1_input"))) if hw else None
        out["gpus"].append({
            "index": i, "name": f"AMD GPU {i}",
            "used_mib": round(int(_read(os.path.join(dev, "mem_info_vram_used")) or 0) / 2**20),
            "total_mib": round(int(_read(os.path.join(dev, "mem_info_vram_total")) or 0) / 2**20),
            "util_pct": int(_read(os.path.join(dev, "gpu_busy_percent")) or 0),
            "temp_c": int(temp) / 1000 if temp else None,
            "power_w": int(power) / 1e6 if power else None,
        })
    if not out["gpus"] and sys.platform == "darwin" and platform.machine() == "arm64":
        out["gpus"].append({"index": 0, "name": "Apple GPU (unified)", "used_mib": out["ram"]["used_mib"],
                            "total_mib": out["ram"]["total_mib"], "util_pct": None, "temp_c": None,
                            "power_w": None})
    return out
