"""Build tiny header-only GGUF files that mirror real llama.cpp quantizations."""

from vram_fit.catalog import CATALOG
from vram_fit.ggml import ggml_type
from vram_fit.gguf import GGUFTensorInfo, build_header
from vram_fit.model import model_from_hf_config
from vram_fit.quant import get_quant, make_quant_context

FTYPE_IDS = {"Q4_K_M": 15, "Q8_0": 7, "Q6_K": 18, "Q5_K_M": 17}


def make_gguf_bytes(catalog_id="meta-llama/Llama-3.2-1B", quant="Q4_K_M", vocab_tokens=64):
    lm = model_from_hf_config(CATALOG[catalog_id], catalog_id)
    s = lm.spec
    q = get_quant(quant)
    qc = make_quant_context(s, lm.inventory)
    tensors = []
    off = 0
    for t in lm.inventory:
        if t.tied or t.kind == "vision":
            continue
        tname = q.tensor_type(t, qc)
        shape = [t.n_in] if t.is_1d else ([t.n_in, t.n_out, t.experts] if t.experts > 1 else [t.n_in, t.n_out])
        tensors.append(GGUFTensorInfo(t.name, shape, ggml_type(tname).id, off))
        off += ggml_type(tname).nbytes(t.numel)
    arch = "llama"
    meta = {
        "general.architecture": arch,
        "general.name": catalog_id.split("/")[-1],
        "general.file_type": FTYPE_IDS[quant],
        f"{arch}.block_count": s.num_layers,
        f"{arch}.context_length": s.max_context,
        f"{arch}.embedding_length": s.hidden_size,
        f"{arch}.feed_forward_length": s.intermediate_size,
        f"{arch}.attention.head_count": s.num_heads,
        f"{arch}.attention.head_count_kv": s.num_kv_heads[0],
        f"{arch}.attention.key_length": s.head_dim,
        f"{arch}.attention.value_length": s.v_head_dim,
        f"{arch}.rope.freq_base": 500000.0,
        "tokenizer.ggml.tokens": [f"tok{i}" for i in range(vocab_tokens)],
    }
    return build_header(meta, tensors), off
