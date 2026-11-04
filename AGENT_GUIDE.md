# AGENT_GUIDE.md — using vram-fit from an AI coding agent

This file is written for autonomous agents (Claude Code, Cursor, Windsurf, OpenHands, Aider, Codex, Copilot agents). It tells you **when** to reach for `vram-fit`, **how** to call it deterministically, and **how to parse** the result.

## When to use it

Use vram-fit when your user is facing any of these:

| Symptom / question from the user | Command to run |
|---|---|
| `CUDA out of memory`, `torch.OutOfMemoryError`, `ggml_cuda ... out of memory`, `failed to allocate`, Ollama "model requires more system memory", vLLM "No available memory for the cache blocks" | `vram-fit fit <model> --gpu auto -c <ctx> --json` |
| "Will <model> run on my GPU / Mac?" "How much VRAM does <model> need?" | `vram-fit estimate <model> --gpu auto -q <quant> -c <ctx> --json` |
| "Which GGUF / quant should I download?" | `vram-fit matrix <repo> --gpu auto --contexts 8k,32k --json` |
| "What `-ngl` / `num_gpu` / `--n-cpu-moe` should I use?" | `vram-fit fit <model> --engine llama.cpp --gpu auto --json` → `result.command.shell` |
| "What `--gpu-memory-utilization` / `--max-model-len` for vLLM?" | `vram-fit fit <repo> --engine vllm --gpu auto -c <ctx> --json` |
| "How much VRAM for LoRA / QLoRA / full fine-tuning?" "What batch size fits?" | `vram-fit fit <repo> --mode qlora --engine unsloth -c <seq> --gpu auto --json` |
| "How big is the KV cache at 128k?" | `vram-fit estimate <model> -c 128k --kv 16 --json` → `result.details.kv_cache` |
| "How many tokens/s will I get?" | `result.performance.decode_tok_s` / `prefill_tok_s` / `train_tok_s` |

Do **not** download model weights to find out whether they fit. vram-fit only reads `config.json`, the Hub model-info API, and GGUF/safetensors *headers* (HTTP Range requests).

## Install (non-interactive)

```bash
pip install vram-fit            # no dependencies; Python >= 3.9
# or, without installing into the environment:
pipx run vram-fit estimate llama3.1-8b --gpu rtx-4090 --json
uvx vram-fit estimate llama3.1-8b --gpu rtx-4090 --json
```

Set `HF_TOKEN` for gated repos (Llama, Gemma). If the Hub is unreachable or the repo is gated without a token, vram-fit falls back to its built-in architecture catalog and says so in `result.notes`. Use `--offline` in sandboxes without network.

## Deterministic calling convention

1. **Always pass `--json`.** stdout then contains exactly one JSON document; nothing else is printed to stdout.
2. **Always pass `--gpu`.** Use `auto` to probe the machine you are running on (NVML / ROCm / Metal). If you are *not* on the user's machine (cloud sandbox, CI), ask for or infer their GPU and pass a preset: `rtx-4090`, `rtx-3060-laptop-6gb`, `m3-max-64gb`, `h100`, `"2x rtx-3090"`, or `custom:<VRAM_GiB>:<GB/s>`. List presets with `vram-fit presets --json`.
3. **Pass the context the user actually needs** (`-c 32k`); defaults are 8192 tokens, batch 1, flash attention on.
4. **Check the exit code:** `0` ok, `2` error (JSON `error` object), `3` only with `--fail-on-oom` when status is `spill`/`oom`.
5. Envelope: `{"vram_fit": "<version>", "command": "<cmd>", "result": {...}}` or `{"vram_fit": ..., "command": ..., "error": {"type": ..., "message": ...}}`. Full schema: [`schema.json`](schema.json).

## Fields you will use

```text
result.status                      fits | tight | offload | spill | oom
result.fits                        bool — true means it will run (fits/tight/offload)
result.gpu_total_mib               VRAM needed (MiB, summed over GPUs)
result.budget_mib                  usable VRAM (vLLM: x gpu_memory_utilization; macOS: GPU wired limit)
result.headroom_mib                budget - total (negative = over)
result.segments[]                  {key, label, mib} for weights, kv_cache, activations, overhead, lora_optimizer
result.offload                     cpu_weights_mib, cpu_kv_mib, layers_on_gpu, layers_total, spill_mib
result.performance                 decode_tok_s, decode_tok_s_full_ctx, prefill_tok_s, bottleneck | train_tok_s
result.limits                      max_context_full_gpu, max_batch_full_gpu
result.settings                    the resolved quant, kv types, gpu_layers, cpu_moe_layers, ...
result.warnings[] / result.notes[] human-readable caveats — surface warnings to the user
```

For `fit`: `result.best` is an estimate object, `result.command.shell` is the copy-paste launch command, `result.command.files` holds generated files (`Modelfile`, `config.yml`, `train.py`), `result.command.flags[]` explains each flag, `result.alternatives[]` lists other configs considered.

## Recipes

**Diagnose an OOM and hand back a working command**

```bash
vram-fit fit Qwen/Qwen2.5-32B-Instruct --gpu auto -c 32k --engine llama.cpp --json > /tmp/fit.json
python -c "import json;r=json.load(open('/tmp/fit.json'))['result'];print(r['best']['status']);print(r['command']['shell'])"
```

**Gate a CI job on "does this config fit a 24 GB card"**

```bash
vram-fit estimate meta-llama/Llama-3.1-8B-Instruct --gpu rtx-4090 -q Q8_0 -c 64k --json --fail-on-oom
```

**Pick a fine-tuning config and write the training script**

```bash
vram-fit fit meta-llama/Llama-3.1-8B --mode qlora --engine unsloth -c 4k --gpu rtx-3060-12gb --json \
  | python -c "import json,sys;r=json.load(sys.stdin)['result'];open('train.py','w').write(r['command']['files']['train.py']);print(r['explanation'])"
```

**Compare quants for a GGUF repo**

```bash
vram-fit matrix bartowski/Qwen2.5-14B-Instruct-GGUF --gpu rtx-4070 --contexts 8k,32k --json
# result.cells[] = {quant, context, status, gpu_total_mib, gpu_layers, tok_s}
```

**Python (in-process, no subprocess)**

```python
from vram_fit import Settings, estimate, load_model, optimize, parse_gpu_spec
m = load_model("Qwen/Qwen3-30B-A3B")
r = optimize(m, parse_gpu_spec("rtx-4060-ti-16gb"), Settings(context=16384))
print(r["best"]["status"], r["command"]["shell"])
```

**HTTP (when `vram-fit serve` is running on 127.0.0.1:7860)**

```bash
curl -s localhost:7860/api/estimate -H 'content-type: application/json' \
  -d '{"model":"llama3.1-8b","gpu":"rtx-4090","settings":{"ctx":32768,"quant":"Q4_K_M","kv":8}}'
```

## How to report results to the user

- Lead with the verdict (`status`) and the number: "Needs 10.5 GiB of 12 GiB — fits; ~36 tok/s."
- If `status` is `offload`, say how many layers stay in RAM and that speed is RAM-bandwidth bound.
- If `spill`/`oom`, give the `fit` command's recommendation instead of a raw number.
- Quote `warnings` verbatim; they contain the actionable fix (e.g. `sudo sysctl iogpu.wired_limit_mb=…`, enable `-fa on`).
- Tokens/s are bandwidth-model estimates (±20 %), memory numbers are exact to the tensor for GGUF headers and typically within ~1 % from `config.json`.

## Model reference formats accepted

`org/repo` · `org/repo:Q4_K_M` (GGUF repo + quant) · `org/repo/path/file.gguf` · `https://huggingface.co/org/repo/blob/main/file.gguf` · any `https://…/file.gguf` that supports Range · `./model.gguf` (split files auto-detected) · `./model-dir/` or `./config.json` · aliases from `vram-fit catalog` (`llama3.1-8b`, `qwen3-30b-a3b`, `gpt-oss-20b`, `deepseek-v3`, …).

## Working on this repository

See [`.cursorrules`](.cursorrules) for code conventions. Run `pytest -q` before committing; keep the core (`src/vram_fit/*` except `server.py`) dependency-free.
