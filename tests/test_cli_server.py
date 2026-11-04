import ast
import json

import pytest

from vram_fit.catalog import CATALOG
from vram_fit.cli import main, parse_tokens
from vram_fit.commands import build_command
from vram_fit.estimator import Settings, estimate
from vram_fit.model import model_from_hf_config
from vram_fit.presets import find_preset, parse_gpu_spec


def run_json(capsys, *argv):
    code = main(list(argv) + ["--json"])
    out = capsys.readouterr().out
    return code, json.loads(out)


def test_parse_tokens():
    assert parse_tokens("32k") == 32768 and parse_tokens("128K") == 131072 and parse_tokens("8192") == 8192
    assert parse_tokens("1.5k") == 1536


def test_presets_lookup():
    assert find_preset("4090").id == "rtx-4090"
    assert find_preset("RTX 3060 12GB").id == "rtx-3060-12gb"
    assert find_preset("rtx 3060 laptop").id == "rtx-3060-laptop-6gb"
    assert find_preset("m3 max 64gb").id == "m3-max-64gb"
    assert find_preset("H100 SXM").id == "h100"
    g = parse_gpu_spec("2x rtx-3090")
    assert g.count == 2 and g.vram_gib == 24
    assert parse_gpu_spec("custom:20:700").vram_gib == 20
    with pytest.raises(KeyError):
        parse_gpu_spec("voodoo-5")


@pytest.mark.parametrize("argv", [
    ["estimate", "llama3.1-8b", "--gpu", "rtx-4090", "-c", "16k", "--command"],
    ["estimate", "qwen3-30b-a3b", "--gpu", "rtx-4060-ti-8gb", "--cpu-moe", "40", "--ngl", "49"],
    ["fit", "qwen2.5-14b", "--gpu", "rtx-4070", "-c", "32k"],
    ["fit", "llama3.1-8b", "--gpu", "rtx-3060-12gb", "--mode", "qlora", "-e", "unsloth", "-c", "4k"],
    ["fit", "mistral-7b", "--gpu", "a10", "-e", "vllm"],
    ["matrix", "gemma3-27b", "--gpu", "m4-max-48gb", "--contexts", "8k,32k"],
    ["inspect", "deepseek-v3"],
    ["presets", "--vendor", "apple"],
    ["quants", "--engine", "vllm"],
    ["catalog"],
    ["hw"],
    ["version"],
])
def test_every_command_emits_json(capsys, argv):
    code, data = run_json(capsys, *argv)
    assert code == 0
    assert data["vram_fit"] and "result" in data and data["command"] in (argv[0], "fit")


def test_errors_are_json(capsys):
    code, data = run_json(capsys, "estimate", "llama3.1-8b", "--gpu", "voodoo-5")
    assert code == 2 and data["error"]["type"] == "KeyError"
    code, data = run_json(capsys, "estimate", "llama3.1-8b", "--gpu", "rtx-4090", "-q", "Q9_X")
    assert code == 2 and "unknown quantization" in data["error"]["message"]


def test_fail_on_oom_exit_code(capsys):
    code = main(["estimate", "llama3.1-70b", "--gpu", "rtx-4060", "--ngl", "81", "--fail-on-oom", "--json"])
    capsys.readouterr()
    assert code == 3


def test_text_output_has_bar(capsys):
    assert main(["estimate", "llama3.1-8b", "--gpu", "rtx-4090", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "VRAM ceiling" in out and "Model Weights" in out and "tok/s" in out


def _lm(cid):
    return model_from_hf_config(CATALOG[cid], cid)


def test_commands_for_every_engine():
    m = _lm("meta-llama/Llama-3.1-8B")
    g = parse_gpu_spec("rtx-4090")
    llama = build_command(m, estimate(m, g, Settings(context=32768, kv_cache="8", batch=2), limits=False))
    assert "-ngl 99" in llama["shell"] and "-ctk q8_0" in llama["shell"] and "-c 65536" in llama["shell"]
    assert "-np 2" in llama["shell"] and "llama-quantize" in llama["shell"]
    oll = build_command(m, estimate(m, g, Settings(engine="ollama", context=16384, kv_cache="q8_0"), limits=False))
    assert "PARAMETER num_ctx 16384" in oll["files"]["Modelfile"] and oll["env"]["OLLAMA_KV_CACHE_TYPE"] == "q8_0"
    vl = build_command(m, estimate(m, g, Settings(engine="vllm", quant="FP8", kv_cache="fp8"), limits=False))
    assert "--kv-cache-dtype fp8" in vl["shell"] and "--quantization fp8" in vl["shell"]
    qw = _lm("Qwen/Qwen2.5-7B")
    long = build_command(qw, estimate(qw, g, Settings(engine="vllm", context=65536), limits=False))
    assert "--hf-overrides" in long["shell"] and "yarn" in long["shell"]
    ex = build_command(m, estimate(m, g, Settings(engine="exllamav2", quant="EXL2-4.65", kv_cache="Q4"), limits=False))
    assert "cache_mode: Q4" in ex["files"]["config.yml"]
    for engine, mode in (("unsloth", "qlora"), ("unsloth", "full"), ("transformers", "qlora"), ("transformers", "lora")):
        tr = build_command(m, estimate(m, parse_gpu_spec("a100-80gb"), Settings(engine=engine, mode=mode, context=2048),
                                       limits=False))
        ast.parse(tr["files"]["train.py"])  # generated script is valid Python


def test_moe_command_uses_n_cpu_moe():
    m = _lm("Qwen/Qwen3-30B-A3B")
    r = estimate(m, parse_gpu_spec("rtx-4060-ti-8gb"), Settings(context=8192, gpu_layers=49, cpu_moe_layers=40),
                 limits=False)
    assert "--n-cpu-moe 40" in build_command(m, r)["shell"]


def test_server_rest_and_websocket():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from vram_fit.server import create_app

    c = TestClient(create_app())
    assert c.get("/api/health").json()["ok"]
    assert len(c.get("/api/presets").json()) > 50
    assert c.get("/api/engines").json()["kv_cache"]["vllm"][0] == "auto"
    assert c.get("/api/model", params={"id": "qwen3-8b"}).json()["spec"]["num_layers"] == 36
    r = c.post("/api/estimate", json={"model": "llama3.1-8b", "gpu": {"preset": "rtx-4090"},
                                      "settings": {"ctx": 8192, "kv": 16}, "command": True}).json()
    assert r["status"] == "fits" and "launch" in r and r["compute_ms"] >= 0
    bad = c.post("/api/estimate", json={"model": "llama3.1-8b", "gpu": "rtx-4090", "settings": {"quant": "Q9"}})
    assert bad.status_code == 400
    opt = c.post("/api/optimize", json={"model": "qwen2.5-14b", "gpu": "rtx-4070", "settings": {"ctx": 16384}}).json()
    assert opt["best"]["fits"] and opt["command"]["shell"].count("llama-server") == 1
    mx = c.post("/api/matrix", json={"model": "qwen2.5-7b", "gpu": "rtx-3060-12gb",
                                     "contexts": [4096, 32768], "quants": ["Q8_0", "Q4_K_M"]}).json()
    assert len(mx["cells"]) == 4
    with c.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"op": "hello", "rid": 1}))
        hello = json.loads(ws.receive_text())
        assert hello["event"] == "hello" and hello["rid"] == 1 and "presets" in hello["data"]
        ws.send_text(json.dumps({"op": "load_model", "rid": 2, "model": "gemma3-12b"}))
        events = [json.loads(ws.receive_text()) for _ in range(3)]
        assert [e["event"] for e in events] == ["progress", "progress", "model"]
        ws.send_text(json.dumps({"op": "estimate", "rid": 3, "model": "gemma3-12b", "gpu": "rtx-4070",
                                 "settings": {"ctx": 32768, "quant": "Q4_K_M"}, "limits": False}))
        est = json.loads(ws.receive_text())
        assert est["event"] == "estimate" and est["rid"] == 3 and est["data"]["segments"][0]["key"] == "weights"
        ws.send_text("not json")
        assert json.loads(ws.receive_text())["event"] == "error"
        ws.send_text(json.dumps({"op": "telemetry", "rid": 4, "enabled": True, "interval_ms": 100}))
        seen = {json.loads(ws.receive_text())["event"] for _ in range(3)}
        assert "telemetry" in seen


@pytest.mark.parametrize("argv", [
    ["estimate", "llama3.1-8b", "--gpu", "rtx-4090", "--command"],
    ["estimate", "qwen3-30b-a3b", "--gpu", "rtx-4060", "--mode", "qlora", "-e", "unsloth"],
    ["fit", "mixtral-8x7b", "--gpu", "rtx-3090", "-c", "16k"],
    ["matrix", "qwen2.5-7b", "--gpu", "rtx-4070", "--contexts", "4k,32k"],
    ["estimate", "llama3.1-8b", "--gpu", "nope-gpu"],
])
def test_json_matches_published_schema(capsys, argv):
    jsonschema = pytest.importorskip("jsonschema")
    import os
    with open(os.path.join(os.path.dirname(__file__), "..", "schema.json")) as f:
        schema = json.load(f)
    main(argv + ["--json"])
    doc = json.loads(capsys.readouterr().out)
    jsonschema.validate(doc, schema)


def test_cp1252_stdout_does_not_crash(monkeypatch):
    # Windows: output redirected to a pipe/file uses the ANSI code page, which has no bar glyphs
    import io
    import sys
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stream)
    assert main(["matrix", "llama3.1-8b", "--gpu", "rtx-4090", "--contexts", "4k"]) == 0
    stream.flush()
    assert "●".encode("utf-8") in raw.getvalue()
