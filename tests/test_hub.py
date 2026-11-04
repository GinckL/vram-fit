"""End-to-end Hub path against a local HTTP server that honours Range requests."""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from vram_fit import hub
from vram_fit.catalog import CATALOG
from vram_fit.hub import list_gguf_files, load_model, parse_ref

from helpers import make_gguf_bytes


class _Hub:
    def __init__(self):
        self.files = {}
        self.json = {}
        self.requests = []


def _handler(state):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            state.requests.append((self.path, self.headers.get("Range")))
            path = self.path.split("?")[0]
            if path in state.json:
                body = json.dumps(state.json[path]).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path in state.files:
                data = state.files[path]
                rng = self.headers.get("Range")
                if rng:
                    a, b = rng.split("=")[1].split("-")
                    a, b = int(a), min(int(b), len(data) - 1)
                    chunk = data[a:b + 1]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
                else:
                    chunk = data
                    self.send_response(200)
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)
                return
            self.send_response(404)
            self.end_headers()
    return H


@pytest.fixture()
def fake_hub(monkeypatch):
    state = _Hub()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _handler(state))
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{srv.server_address[1]}")
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    yield state
    srv.shutdown()


def test_parse_ref():
    assert parse_ref("bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M") == {
        "repo": "bartowski/Qwen2.5-7B-Instruct-GGUF", "file": None, "tag": "Q4_K_M", "revision": "main", "url": None}
    r = parse_ref("https://huggingface.co/org/repo/blob/dev/sub/model-Q8_0.gguf")
    assert r["repo"] == "org/repo" and r["revision"] == "dev" and r["file"] == "sub/model-Q8_0.gguf"
    assert parse_ref("hf.co/org/repo:IQ4_XS")["tag"] == "IQ4_XS"
    assert parse_ref("org/repo/path/file.gguf")["file"] == "path/file.gguf"


def test_list_gguf_files_groups_splits():
    sib = [{"rfilename": "m-Q4_K_M.gguf", "size": 100},
           {"rfilename": "Q8_0/m-Q8_0-00001-of-00002.gguf", "size": 60},
           {"rfilename": "Q8_0/m-Q8_0-00002-of-00002.gguf", "size": 70},
           {"rfilename": "mmproj-m-f16.gguf", "size": 10},
           {"rfilename": "README.md", "size": 1}]
    files = {f["quant"] or f["filename"]: f for f in list_gguf_files(sib)}
    assert files["Q8_0"]["size"] == 130 and files["Q8_0"]["splits"] == 2
    assert files["Q8_0"]["filename"].endswith("00001-of-00002.gguf")
    assert any(f["mmproj"] for f in files.values())


def test_gguf_repo_reads_only_the_header(fake_hub):
    raw, data_bytes = make_gguf_bytes("meta-llama/Llama-3.2-1B", "Q4_K_M", vocab_tokens=20000)
    full = raw + b"\0" * 4096  # stand-in for tensor data (never requested)
    repo = "acme/Llama-3.2-1B-GGUF"
    fake_hub.json[f"/api/models/{repo}/revision/main"] = {
        "siblings": [{"rfilename": "Llama-3.2-1B-Q4_K_M.gguf", "size": len(raw) + data_bytes},
                     {"rfilename": "Llama-3.2-1B-Q8_0.gguf", "size": 1_400_000_000}],
        "gguf": {"total": 1_235_814_400}}
    fake_hub.files[f"/{repo}/resolve/main/Llama-3.2-1B-Q4_K_M.gguf"] = full
    lm = load_model(repo)
    assert lm.source == "gguf" and lm.exact and lm.native_quant == "Q4_K_M"
    assert lm.spec.num_layers == 16
    assert {f["quant"] for f in lm.gguf_files} == {"Q4_K_M", "Q8_0"}
    ranged = [r for p, r in fake_hub.requests if p.endswith(".gguf")]
    assert ranged and all(r and r.startswith("bytes=") for r in ranged)
    # second load is served from the on-disk header cache
    n = len(fake_hub.requests)
    load_model(repo)
    assert all(not p.endswith(".gguf") for p, _ in fake_hub.requests[n:])


def test_config_repo_and_reported_params(fake_hub):
    repo = "acme/Qwen2.5-7B-VL-ish"
    fake_hub.json[f"/api/models/{repo}/revision/main"] = {
        "siblings": [{"rfilename": "config.json"}, {"rfilename": "model.safetensors"}],
        "safetensors": {"total": 8_300_000_000}}
    fake_hub.json[f"/{repo}/resolve/main/config.json"] = CATALOG["Qwen/Qwen2.5-7B"]
    lm = load_model(repo)
    assert lm.source == "hf-config"
    assert lm.params_total == pytest.approx(8.3e9, rel=0.001)  # unmapped extra params are counted
    assert any("extra" in n for n in lm.notes)


def test_exact_safetensors_headers(fake_hub):
    repo = "acme/tiny"
    cfg = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 64,
           "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
           "intermediate_size": 128, "vocab_size": 1000, "tie_word_embeddings": True}
    hdr = {"__metadata__": {"format": "pt"}}
    off = 0

    def add(name, shape):
        nonlocal off
        n = 1
        for d in shape:
            n *= d
        hdr[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [off, off + 2 * n]}
        off += 2 * n
    add("model.embed_tokens.weight", [1000, 64])
    for i in range(2):
        p = f"model.layers.{i}."
        add(p + "self_attn.q_proj.weight", [64, 64]); add(p + "self_attn.k_proj.weight", [32, 64])
        add(p + "self_attn.v_proj.weight", [32, 64]); add(p + "self_attn.o_proj.weight", [64, 64])
        add(p + "mlp.gate_proj.weight", [128, 64]); add(p + "mlp.up_proj.weight", [128, 64])
        add(p + "mlp.down_proj.weight", [64, 128])
        add(p + "input_layernorm.weight", [64]); add(p + "post_attention_layernorm.weight", [64])
    add("model.norm.weight", [64])
    hjson = json.dumps(hdr).encode()
    blob = len(hjson).to_bytes(8, "little") + hjson + b"\0" * off
    fake_hub.json[f"/api/models/{repo}/revision/main"] = {"siblings": [{"rfilename": "config.json"},
                                                                       {"rfilename": "model.safetensors"}]}
    fake_hub.json[f"/{repo}/resolve/main/config.json"] = cfg
    fake_hub.files[f"/{repo}/resolve/main/model.safetensors"] = blob
    lm = load_model(repo, exact=True)
    assert lm.exact and lm.source == "safetensors" and lm.native_quant == "BF16"
    synth = load_model(repo)
    assert lm.params_total == synth.params_total


def test_offline_fallback_uses_catalog(monkeypatch):
    lm = load_model("bartowski/Qwen2.5-14B-Instruct-GGUF", offline=True)
    assert lm.source == "catalog" and lm.spec.hidden_size == 5120


def test_unreachable_hub_falls_back(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:9")
    lm = load_model("Qwen/Qwen3-8B")
    assert lm.source == "catalog" and any("network" in n or "HTTP" in n for n in lm.notes)


def test_local_gguf_file(tmp_path):
    raw, _ = make_gguf_bytes("meta-llama/Llama-3.2-1B", "Q8_0")
    p = tmp_path / "Llama-3.2-1B-Q8_0.gguf"
    p.write_bytes(raw)
    lm = load_model(str(p))
    assert lm.source == "gguf-local" and lm.native_quant == "Q8_0" and lm.path == os.path.abspath(p)


def test_local_config(tmp_path):
    d = tmp_path / "my-model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(CATALOG["mistralai/Mistral-7B-v0.3"]))
    lm = load_model(str(d))
    assert lm.source == "local-config" and lm.spec.vocab_size == 32768
    assert hub.cache_dir()
