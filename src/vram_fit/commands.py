"""Launch-command generation for llama.cpp, Ollama, vLLM, ExLlamaV2 (TabbyAPI), Unsloth and HF PEFT.

Every command is built from a resolved estimate, so each flag maps to a number
vram-fit already computed (layers that fit, KV dtype, context, batch).
"""

from __future__ import annotations

import json
import math
import re
import shlex
from typing import Dict, List, Optional

from .hardware import HardwareProfile
from .model import LoadedModel
from .tensors import DEFAULT_LORA_TARGETS

OLLAMA_CREATE_QUANTS = {"Q4_K_M": "q4_K_M", "Q4_K_S": "q4_K_S", "Q8_0": "q8_0", "F16": "f16", "BF16": "f16"}


def _slug(model: LoadedModel) -> str:
    base = (model.repo or model.id or "model").rstrip("/").split("/")[-1]
    base = re.sub(r"(?i)\.gguf$", "", base)
    return re.sub(r"[^A-Za-z0-9._-]+", "-", base) or "model"


def _rope_factor(model: LoadedModel, ctx: int) -> Optional[float]:
    if ctx <= model.spec.max_context:
        return None
    return float(2 ** math.ceil(math.log2(ctx / model.spec.max_context)))


def _is_gguf_repo(model: LoadedModel, quant: str) -> bool:
    if any((f.get("quant") or "").upper() == quant.upper() and not f.get("mmproj") for f in model.gguf_files):
        return True
    # offline / catalog fallback for a repo named like a GGUF repo (bartowski/*-GGUF, unsloth/*-GGUF)
    return not model.gguf_files and bool(model.repo) and "gguf" in (model.repo or "").lower()


def _gguf_source(model: LoadedModel, quant: str) -> Dict[str, object]:
    """How llama.cpp / Ollama get a GGUF for this model: local file, HF GGUF repo, or conversion."""
    if model.source == "gguf-local" and model.path:
        return {"kind": "local", "path": model.path, "steps": []}
    if model.repo and (_is_gguf_repo(model, quant) or (model.source == "gguf" and model.native_quant == quant)):
        return {"kind": "hf", "ref": f"{model.repo}:{quant}", "steps": []}
    name = _slug(model)
    src = model.path if model.source == "local-config" and model.path else f"./{name}"
    steps = []
    if not (model.source == "local-config" and model.path):
        steps.append(f"hf download {model.repo or model.id} --local-dir ./{name}")
    out = f"./{name}-{quant}.gguf"
    if quant in ("BF16", "F16", "F32", "Q8_0"):
        steps.append(f"python convert_hf_to_gguf.py {shlex.quote(src)} --outtype {quant.lower()} --outfile {out}")
    else:
        steps.append(f"python convert_hf_to_gguf.py {shlex.quote(src)} --outtype bf16 --outfile ./{name}-BF16.gguf")
        steps.append(f"llama-quantize ./{name}-BF16.gguf {out} {quant}")
    return {"kind": "converted", "path": out, "steps": steps}


def _threads(hw: Optional[HardwareProfile]) -> Optional[int]:
    return hw.cpu_physical_cores if hw else None


def llama_cpp_command(model: LoadedModel, est: dict, hw: Optional[HardwareProfile] = None,
                      host: str = "127.0.0.1", port: int = 8080) -> dict:
    st = est["settings"]
    L = model.spec.num_layers
    quant = st["quant"]
    ctx, batch, ngl = st["context"], st["batch"], st["gpu_layers"]
    src = _gguf_source(model, quant)
    args: List[str] = ["llama-server"]
    flags = []
    if src["kind"] == "hf":
        args += ["-hf", str(src["ref"])]
        flags.append({"flag": f"-hf {src['ref']}", "why": "pulls only the selected GGUF quant from the Hub"})
    else:
        args += ["-m", shlex.quote(str(src["path"]))]
    total_ctx = ctx * batch
    args += ["-c", str(total_ctx)]
    flags.append({"flag": f"-c {total_ctx}", "why": f"{ctx:,} tokens x {batch} parallel slot(s)"})
    ngl_s = "99" if ngl >= L + 1 else str(ngl)
    args += ["-ngl", ngl_s]
    flags.append({"flag": f"-ngl {ngl_s}", "why": f"{min(ngl, L)}/{L} layers on GPU" +
                  (" + output head" if ngl >= L + 1 else "") + " — the most that fit"})
    if st.get("cpu_moe_layers"):
        args += ["--n-cpu-moe", str(st["cpu_moe_layers"])]
        flags.append({"flag": f"--n-cpu-moe {st['cpu_moe_layers']}",
                      "why": "MoE experts of the first layers stay in RAM; attention + shared weights stay on GPU"})
    if st["flash_attention"]:
        args += ["-fa", "on"]
        flags.append({"flag": "-fa on", "why": "flash attention: O(n) attention buffers, enables V-cache quantization"})
    if st["kv_cache_k"] and st["kv_cache_k"] != "f16":
        args += ["-ctk", st["kv_cache_k"], "-ctv", st["kv_cache_v"]]
        flags.append({"flag": f"-ctk {st['kv_cache_k']} -ctv {st['kv_cache_v']}",
                      "why": f"KV cache {est['details']['kv_cache']['total_mib']:,.0f} MiB at {st['kv_cache_k']}"})
    if batch > 1:
        args += ["-np", str(batch)]
        flags.append({"flag": f"-np {batch}", "why": "parallel decode slots sharing the KV cache"})
    ub = st["ubatch"]
    if ub != 512:
        args += ["-b", str(max(2048, ub)), "-ub", str(ub)]
        flags.append({"flag": f"-ub {ub}", "why": "larger micro-batch = faster prompt processing (VRAM allows it)"})
    if est["offload"]["layers_on_gpu"] < L or st.get("cpu_moe_layers"):
        th = _threads(hw)
        if th:
            args += ["-t", str(th)]
            flags.append({"flag": f"-t {th}", "why": "physical cores for the CPU-resident layers"})
    rf = _rope_factor(model, ctx)
    if rf:
        args += ["--rope-scaling", "yarn", "--rope-scale", f"{rf:g}", "--yarn-orig-ctx", str(model.spec.max_context)]
        flags.append({"flag": f"--rope-scaling yarn --rope-scale {rf:g}", "why": "context beyond the trained window"})
    args += ["--host", host, "--port", str(port), "--jinja"]
    shell = "\n".join(src["steps"] + [" ".join(args)])
    return {"engine": "llama.cpp", "shell": shell, "argv": args, "env": {}, "files": {}, "flags": flags,
            "endpoint": f"http://{host}:{port}/v1"}


def ollama_command(model: LoadedModel, est: dict, hw: Optional[HardwareProfile] = None) -> dict:
    st = est["settings"]
    L = model.spec.num_layers
    quant = st["quant"]
    name = _slug(model).lower() + "-vramfit"
    env: Dict[str, str] = {"OLLAMA_CONTEXT_LENGTH": str(st["context"])}
    flags = []
    if st["flash_attention"]:
        env["OLLAMA_FLASH_ATTENTION"] = "1"
    if st["kv_cache_k"] and st["kv_cache_k"] != "f16":
        env["OLLAMA_KV_CACHE_TYPE"] = st["kv_cache_k"]
        flags.append({"flag": f"OLLAMA_KV_CACHE_TYPE={st['kv_cache_k']}", "why": "quantized KV cache (needs flash attention)"})
    if st["batch"] > 1:
        env["OLLAMA_NUM_PARALLEL"] = str(st["batch"])
    steps: List[str] = []
    create_quant = None
    if model.source == "gguf-local" and model.path:
        from_line = f"FROM {model.path}"
    elif model.repo and (_is_gguf_repo(model, quant) or model.source == "gguf"):
        from_line = f"FROM hf.co/{model.repo}:{quant}"
    elif quant in OLLAMA_CREATE_QUANTS:
        local = model.path if model.source == "local-config" and model.path else f"./{_slug(model)}"
        if not (model.source == "local-config" and model.path):
            steps.append(f"hf download {model.repo or model.id} --local-dir {local}")
        from_line = f"FROM {local}"
        create_quant = OLLAMA_CREATE_QUANTS[quant]
    else:
        src = _gguf_source(model, quant)
        steps += list(src["steps"])
        from_line = f"FROM {src['path']}"
    lines = [from_line, f"PARAMETER num_ctx {st['context']}"]
    if st["gpu_layers"] < L + 1:
        lines.append(f"PARAMETER num_gpu {st['gpu_layers']}")
        flags.append({"flag": f"num_gpu {st['gpu_layers']}", "why": "layers offloaded to the GPU (rest in RAM)"})
        th = _threads(hw)
        if th:
            lines.append(f"PARAMETER num_thread {th}")
    if st["ubatch"] != 512:
        lines.append(f"PARAMETER num_batch {st['ubatch']}")
    modelfile = "\n".join(lines) + "\n"
    env_s = " ".join(f"{k}={v}" for k, v in env.items())
    create = f"ollama create {name} -f Modelfile" + (f" --quantize {create_quant}" if create_quant and create_quant != "f16" else "")
    shell = "\n".join(steps + ["cat > Modelfile <<'EOF'", modelfile.rstrip("\n"), "EOF",
                               f"{env_s} ollama serve &", create, f"ollama run {name}"])
    flags.insert(0, {"flag": f"num_ctx {st['context']}", "why": "context window (Ollama defaults to 4096)"})
    return {"engine": "ollama", "shell": shell, "env": env, "files": {"Modelfile": modelfile}, "flags": flags,
            "endpoint": "http://127.0.0.1:11434/v1"}


def vllm_command(model: LoadedModel, est: dict, hw: Optional[HardwareProfile] = None,
                 host: str = "127.0.0.1", port: int = 8000) -> dict:
    st = est["settings"]
    repo = model.path if model.source == "local-config" and model.path else (model.repo or model.id)
    args = ["vllm", "serve", repo, "--max-model-len", str(st["context"]),
            "--gpu-memory-utilization", f"{st['gpu_mem_util']:.2f}"]
    flags = [{"flag": f"--max-model-len {st['context']}", "why": "caps per-request context so KV fits"},
             {"flag": f"--gpu-memory-utilization {st['gpu_mem_util']:.2f}",
              "why": "vLLM pre-allocates this fraction; the remainder after weights becomes KV cache"}]
    conc = est["details"]["kv_cache"].get("vllm_max_concurrency_at_ctx") or 1
    seqs = max(st["batch"], min(256, int(conc * 4))) if conc >= 1 else st["batch"]
    args += ["--max-num-seqs", str(seqs)]
    flags.append({"flag": f"--max-num-seqs {seqs}", "why": f"KV holds {conc:.1f} full-length sequences; shorter ones pack more"})
    if st["kv_cache_k"] and st["kv_cache_k"] != "auto":
        args += ["--kv-cache-dtype", st["kv_cache_k"]]
        flags.append({"flag": f"--kv-cache-dtype {st['kv_cache_k']}", "why": "halves KV memory vs bf16"})
    q = st["quant"]
    native = model.native_quant
    if q == "FP8" and native != "FP8":
        args += ["--quantization", "fp8"]
        flags.append({"flag": "--quantization fp8", "why": "on-the-fly FP8 weight quantization (no new checkpoint)"})
    elif q == "NF4" and native != "NF4":
        args += ["--quantization", "bitsandbytes"]
        flags.append({"flag": "--quantization bitsandbytes", "why": "in-flight 4-bit NF4 quantization"})
    if not est["hardware"].get("supports_bf16", True) and q in ("BF16", "F16"):
        args += ["--dtype", "half"]
        flags.append({"flag": "--dtype half", "why": "this GPU has no bf16 support"})
    if est["hardware"]["count"] > 1:
        args += ["--tensor-parallel-size", str(est["hardware"]["count"])]
        flags.append({"flag": f"--tensor-parallel-size {est['hardware']['count']}", "why": "shard weights + KV across GPUs"})
    if est["details"]["overhead"].get("cuda_graphs", 1) == 0:
        args += ["--enforce-eager"]
        flags.append({"flag": "--enforce-eager", "why": "skips CUDA-graph capture to free ~0.5 GiB"})
    args += ["--enable-prefix-caching"]
    rf = _rope_factor(model, st["context"])
    if rf:
        ov = {"rope_scaling": {"rope_type": "yarn", "factor": rf,
                               "original_max_position_embeddings": model.spec.max_context},
              "max_position_embeddings": st["context"]}
        args += ["--hf-overrides", shlex.quote(json.dumps(ov, separators=(",", ":")))]
        flags.append({"flag": "--hf-overrides rope_scaling", "why": f"YaRN x{rf:g} for context beyond the trained window"})
    args += ["--host", host, "--port", str(port)]
    q_warn = []
    if q in ("AWQ", "GPTQ-4bit", "GPTQ-8bit", "NVFP4") and native != q:
        q_warn.append(f"{q} needs a pre-quantized checkpoint (search the Hub for '{_slug(model)} {q}') — "
                      f"pass that repo instead of {repo}.")
    return {"engine": "vllm", "shell": " ".join(args), "argv": args, "env": {}, "files": {}, "flags": flags,
            "endpoint": f"http://{host}:{port}/v1", "notes": q_warn}


def exllamav2_command(model: LoadedModel, est: dict, hw: Optional[HardwareProfile] = None) -> dict:
    st = est["settings"]
    bpw = float(st["quant"].split("-")[-1]) if st["quant"].startswith("EXL2-") else 4.0
    name = _slug(model)
    out_name = f"{name}-exl2-{bpw:g}bpw"
    steps = []
    repo = model.repo or model.id
    if "exl2" in repo.lower():
        steps.append(f"hf download {repo} --revision {bpw:g}bpw --local-dir models/{out_name}")
    else:
        steps.append(f"hf download {repo} --local-dir ./{name}")
        steps.append(f"python convert.py -i ./{name} -o ./exl2-work -cf models/{out_name} -b {bpw:g}")
    cache = st["kv_cache_k"] or "FP16"
    multi = est["hardware"]["count"] > 1
    cfg = "\n".join([
        "network:",
        "  host: 127.0.0.1",
        "  port: 5000",
        "model:",
        "  model_dir: models",
        f"  model_name: {out_name}",
        f"  max_seq_len: {st['context']}",
        f"  cache_size: {st['context'] * st['batch']}",
        f"  cache_mode: {cache}",
        "  chunk_size: 2048",
        "  gpu_split_auto: true",
        f"  tensor_parallel: {'true' if multi else 'false'}",
    ]) + "\n"
    shell = "\n".join(steps + ["# save the block below as TabbyAPI/config.yml", "cat > config.yml <<'EOF'",
                               cfg.rstrip("\n"), "EOF", "./start.sh"])
    flags = [{"flag": f"cache_mode: {cache}", "why": "quantized KV cache (Q4 is near-lossless in ExLlamaV2)"},
             {"flag": f"max_seq_len: {st['context']}", "why": "context that fits next to the weights"}]
    return {"engine": "exllamav2", "shell": shell, "env": {}, "files": {"config.yml": cfg}, "flags": flags,
            "endpoint": "http://127.0.0.1:5000/v1"}


def _targets(est: dict) -> List[str]:
    return list(est["settings"].get("lora_targets") or DEFAULT_LORA_TARGETS)


def unsloth_script(model: LoadedModel, est: dict, grad_accum: int = 1) -> dict:
    st = est["settings"]
    repo = model.path if model.source == "local-config" and model.path else (model.repo or model.id)
    full = st["mode"] == "full"
    qlora = st["mode"] == "qlora"
    gc = st["grad_checkpointing"]
    gc_py = '"unsloth"' if gc == "unsloth" else ("True" if gc == "standard" else "False")
    bf16 = st["quant"] != "F16"
    r = st["lora_rank"] or 16
    lines = [
        "import sys",
        "from datasets import load_dataset",
        "from unsloth import FastLanguageModel",
        "from trl import SFTConfig, SFTTrainer",
        "",
        'DATA_FILE = sys.argv[1] if len(sys.argv) > 1 else "train.jsonl"  # JSONL rows with a "text" field',
        f"MAX_SEQ_LEN = {st['context']}",
        "",
        "model, tokenizer = FastLanguageModel.from_pretrained(",
        f'    model_name="{repo}",',
        "    max_seq_length=MAX_SEQ_LEN,",
        f"    load_in_4bit={qlora},",
        "    dtype=None,",
    ]
    if full:
        lines.append("    full_finetuning=True,")
    lines.append(")")
    if not full:
        lines += [
            "model = FastLanguageModel.get_peft_model(",
            "    model,",
            f"    r={r},",
            f"    lora_alpha={r},",
            "    lora_dropout=0,",
            '    bias="none",',
            f"    target_modules={_targets(est)!r},",
            f"    use_gradient_checkpointing={gc_py},",
            "    random_state=3407,",
            ")",
        ]
    lines += [
        "",
        'dataset = load_dataset("json", data_files=DATA_FILE, split="train")',
        "trainer = SFTTrainer(",
        "    model=model,",
        "    processing_class=tokenizer,",
        "    train_dataset=dataset,",
        "    args=SFTConfig(",
        '        dataset_text_field="text",',
        "        max_length=MAX_SEQ_LEN,",
        f"        per_device_train_batch_size={st['batch']},",
        f"        gradient_accumulation_steps={grad_accum},",
        f'        optim="{st["optimizer"]}",',
        f"        learning_rate={'2e-5' if full else '2e-4'},",
        "        num_train_epochs=1,",
        "        warmup_steps=10,",
        f"        bf16={bf16},",
        f"        fp16={not bf16},",
        "        logging_steps=10,",
        '        output_dir="outputs",',
        "    ),",
        ")",
        "trainer.train()",
        f'model.save_pretrained("{"finetuned" if full else "lora_adapter"}")',
        f'tokenizer.save_pretrained("{"finetuned" if full else "lora_adapter"}")',
    ]
    script = "\n".join(lines) + "\n"
    flags = [
        {"flag": f"per_device_train_batch_size={st['batch']}", "why": "largest micro-batch that fits"},
        {"flag": f"gradient_accumulation_steps={grad_accum}", "why": f"effective batch {st['batch'] * grad_accum}"},
        {"flag": f'optim="{st["optimizer"]}"', "why": f"{est['details']['lora_optimizer'].get('optimizer_states', 0):,.0f} MiB of optimizer state"},
    ]
    if not full:
        flags.append({"flag": f"use_gradient_checkpointing={gc_py}", "why": "offloads checkpointed activations to RAM" if gc == "unsloth" else "recompute activations in backward"})
    return {"engine": "unsloth", "shell": "pip install unsloth\npython train.py train.jsonl", "env": {},
            "files": {"train.py": script}, "flags": flags}


def transformers_script(model: LoadedModel, est: dict, grad_accum: int = 1) -> dict:
    st = est["settings"]
    repo = model.path if model.source == "local-config" and model.path else (model.repo or model.id)
    mode = st["mode"]
    bf16 = st["quant"] != "F16"
    dtype = "torch.bfloat16" if bf16 else "torch.float16"
    r = st["lora_rank"] or 16
    lines = [
        "import sys",
        "import torch",
        "from datasets import load_dataset",
        "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig",
        "from trl import SFTConfig, SFTTrainer",
    ]
    if mode != "full":
        lines.append("from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training")
    lines += [
        "",
        'DATA_FILE = sys.argv[1] if len(sys.argv) > 1 else "train.jsonl"  # JSONL rows with a "text" field',
        f'MODEL = "{repo}"',
        "",
        "tokenizer = AutoTokenizer.from_pretrained(MODEL)",
    ]
    attn = '"flash_attention_2"' if st["flash_attention"] else '"sdpa"'
    if mode == "qlora":
        lines += [
            "bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=\"nf4\",",
            f"                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype={dtype})",
            f"model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, torch_dtype={dtype},",
            f"                                             attn_implementation={attn}, device_map={{\"\": 0}})",
            f"model = prepare_model_for_kbit_training(model, use_gradient_checkpointing={st['grad_checkpointing'] != 'none'})",
        ]
    else:
        lines += [
            f"model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype={dtype},",
            f"                                             attn_implementation={attn}, device_map={{\"\": 0}})",
        ]
    if mode != "full":
        lines += [
            f"model = get_peft_model(model, LoraConfig(r={r}, lora_alpha={r * 2}, lora_dropout=0.0, bias=\"none\",",
            f"                                         target_modules={_targets(est)!r}, task_type=\"CAUSAL_LM\"))",
            "model.print_trainable_parameters()",
        ]
    lines += [
        "",
        'dataset = load_dataset("json", data_files=DATA_FILE, split="train")',
        "trainer = SFTTrainer(",
        "    model=model,",
        "    processing_class=tokenizer,",
        "    train_dataset=dataset,",
        "    args=SFTConfig(",
        '        dataset_text_field="text",',
        f"        max_length={st['context']},",
        f"        per_device_train_batch_size={st['batch']},",
        f"        gradient_accumulation_steps={grad_accum},",
        f"        gradient_checkpointing={st['grad_checkpointing'] != 'none'},",
        f'        optim="{st["optimizer"]}",',
        f"        learning_rate={'2e-5' if mode == 'full' else '2e-4'},",
        "        num_train_epochs=1,",
        f"        bf16={bf16},",
        f"        fp16={not bf16},",
        "        logging_steps=10,",
        '        output_dir="outputs",',
        "    ),",
        ")",
        "trainer.train()",
        'trainer.save_model("outputs/final")',
    ]
    script = "\n".join(lines) + "\n"
    flags = [{"flag": f"per_device_train_batch_size={st['batch']}", "why": "largest micro-batch that fits"},
             {"flag": f'optim="{st["optimizer"]}"', "why": "optimizer state footprint"}]
    deps = "pip install torch transformers peft trl datasets bitsandbytes accelerate"
    return {"engine": "transformers", "shell": f"{deps}\npython train.py train.jsonl", "env": {},
            "files": {"train.py": script}, "flags": flags}


def build_command(model: LoadedModel, est: dict, hw: Optional[HardwareProfile] = None, grad_accum: int = 1) -> dict:
    engine = est["settings"]["engine"]
    if engine == "llama.cpp":
        return llama_cpp_command(model, est, hw)
    if engine == "ollama":
        return ollama_command(model, est, hw)
    if engine == "vllm":
        return vllm_command(model, est, hw)
    if engine == "exllamav2":
        return exllamav2_command(model, est, hw)
    if engine == "unsloth":
        return unsloth_script(model, est, grad_accum)
    return transformers_script(model, est, grad_accum)
