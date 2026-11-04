"""vram-fit: exact VRAM calculator and launch-command tuner for local LLM inference and LoRA/QLoRA fine-tuning.

    >>> from vram_fit import load_model, parse_gpu_spec, estimate, Settings
    >>> m = load_model("Qwen/Qwen2.5-14B-Instruct")          # reads config.json only
    >>> r = estimate(m, parse_gpu_spec("rtx-4070"), Settings(quant="Q4_K_M", context=32768, kv_cache="q8_0"))
    >>> r["status"], r["gpu_total_mib"], r["performance"]["decode_tok_s"]
"""

__version__ = "0.1.0"

from .commands import build_command  # noqa: E402
from .estimator import Settings, estimate  # noqa: E402
from .hardware import probe  # noqa: E402
from .hub import load_model  # noqa: E402
from .optimizer import optimize  # noqa: E402
from .presets import find_preset, list_presets, parse_gpu_spec  # noqa: E402

__all__ = [
    "__version__", "Settings", "estimate", "optimize", "build_command", "load_model", "probe",
    "parse_gpu_spec", "find_preset", "list_presets",
]
