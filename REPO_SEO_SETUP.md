# REPO_SEO_SETUP.md — GitHub / launch checklist

## 1. GitHub "About" description (310 / 350 characters)

```text
VRAM calculator for local LLMs: exact GPU memory for weights, KV cache, context length and LoRA/QLoRA fine-tuning from config.json/GGUF headers, no weight download. Probes NVIDIA/AMD/Apple GPUs, predicts tokens/s and prints llama.cpp, Ollama, vLLM, ExLlamaV2 and Unsloth commands that avoid CUDA out of memory.
```

Website field: `https://ginckl.github.io/vram-fit/` (after step 4).

## 2. Topics (exactly 20 — GitHub's maximum)

```text
llm vram-calculator gpu-memory kv-cache llama-cpp ollama vllm gguf quantization lora qlora unsloth fine-tuning local-llm cuda huggingface exllamav2 apple-silicon llm-inference out-of-memory
```

Settings → (gear next to "About") → Topics: paste the words above, one by one.

## 3. Launch titles

- **Show HN (76 chars):** `Show HN: vram-fit – exact VRAM and tok/s for local LLMs, before you download`
  - URL: the GitHub repo. First comment: the TL;DR block from README + the accuracy table + one `vram-fit fit` output.
- **r/LocalLLaMA:** `I built vram-fit: it reads only config.json/GGUF headers and tells you exact VRAM (weights + KV cache + LoRA states), tok/s and the llama.cpp/Ollama/vLLM command for your GPU`
- **r/MachineLearning [P]:** `[P] vram-fit – exact GPU memory for LLM inference and LoRA/QLoRA fine-tuning (KV cache, GQA/MLA, k-quant mixes) without downloading weights`
- **r/ollama:** `Stop guessing num_ctx / num_gpu: vram-fit computes the VRAM your Ollama model needs and writes the Modelfile`
- **X / Bluesky / LinkedIn:** `"CUDA out of memory" is a math problem. vram-fit solves it before you download 40 GB: exact VRAM for weights, KV cache, context & LoRA, tok/s, and the launch command. pip install vram-fit`

Post on a weekday 13:00–15:00 UTC; reply to every comment in the first 2 hours (ranking signal for HN and Reddit; answers get indexed by Perplexity/Google with the thread).

## 4. Indexing checklist (do these the day you launch)

- [ ] Settings → Pages → Deploy from branch `main`, folder `/docs` (serves `docs/index.html` with Schema.org JSON-LD).
- [ ] Settings → General → Social preview: upload a 1280×640 screenshot of the VRAM bar / dashboard.
- [ ] Publish to PyPI: create a PyPI "trusted publisher" for `GinckL/vram-fit` → workflow `release.yml`, then `git tag v0.1.0 && git push --tags`. (`pip install vram-fit`, `pipx run vram-fit`, `uvx vram-fit` only work after this.)
- [ ] Create GitHub Release `v0.1.0` with the README TL;DR as notes (releases are indexed separately).
- [ ] Make sure `llms.txt`, `llms-full.txt`, `AGENT_GUIDE.md`, `schema.json`, `CITATION.cff` are on the default branch (raw URLs in llms.txt point at `main`).
- [ ] Submit `https://ginckl.github.io/vram-fit/` in Google Search Console and Bing Webmaster Tools (Bing feeds ChatGPT search); request indexing.
- [ ] Answer existing questions where this is the answer (Stack Overflow / r/LocalLLaMA / HF forums "how much VRAM for …", "CUDA out of memory llama.cpp context") with the exact command and output — not just a link.
- [ ] Open PRs adding vram-fit to awesome lists: awesome-local-llm / awesome-llm-inference / awesome-llmops style lists.
- [ ] Pin the repo on your GitHub profile; enable Discussions with a "Show your GPU + config" thread.
