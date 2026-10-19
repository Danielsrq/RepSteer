"""Shared helpers: model loading + locating the decoder (residual-stream) layers.

Imported by the sibling scripts (they run with scripts/ on sys.path).
"""

import os

# The models used here are already in the local HF cache, so default to offline: this stops
# from_pretrained from hitting the Hub on every load (the "unauthenticated requests to the HF Hub"
# warning) and makes loads faster. setdefault, so `HF_HUB_OFFLINE=0 python ...` still lets you
# download a NEW model (e.g. a preset you have not fetched yet).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn as nn

PRESETS = {
    "qwen35-4b": ("Qwen/Qwen3.5-4B", "multimodal"),
    "qwen35-2b": ("Qwen/Qwen3.5-2B", "multimodal"),
    "qwen3-text": ("Qwen/Qwen3-4B-Instruct-2507", "text"),
}


def resolve(preset: str, model: str | None = None, kind: str | None = None):
    mid, k = PRESETS[preset]
    return (model or mid), (kind or k)


def load_model(model_id: str, kind: str, device: str = "cuda"):
    """Return (handler, model, tok, is_multimodal). handler = processor or tokenizer."""
    if kind == "multimodal":
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        handler = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForMultimodalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device
        )
        tok = handler.tokenizer
        is_mm = True
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        handler = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device
        )
        tok = handler
        is_mm = False
    model.eval()
    return handler, model, tok, is_mm


def get_decoder_layers(model):
    """Return (module_path, ModuleList) of the text decoder blocks.

    Picks the ModuleList whose length == text config num_hidden_layers and whose
    name is not the vision tower. Works for both the multimodal
    (model.language_model.layers) and text (model.model.layers) models.
    """
    n = model.config.get_text_config().num_hidden_layers
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == n and "vis" not in name.lower():
            return name, mod
    # fallback: longest ModuleList
    cands = [(nm, md) for nm, md in model.named_modules()
             if isinstance(md, nn.ModuleList) and len(md) > 0]
    return max(cands, key=lambda x: len(x[1]))


def layer_output_hidden(output):
    """Decoder layers may return a tuple (hidden, ...) or a bare tensor."""
    return output[0] if isinstance(output, tuple) else output
