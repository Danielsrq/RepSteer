"""J-lens layout probe + Gated-DeltaNet backward smoke test — GO/NO-GO.

1. Load Qwen3.5-4B, inspect the module tree, and figure out the `jlens.Layout`
   (auto-detect if a top-level `lm_head` exists; otherwise attach a top alias so
   `getattr(hf_model, "lm_head")` resolves).
2. Wrap with `jlens.from_hf` and run `jlens.fit` on 2 throwaway prompts at
   source_layers=[17] to prove the GDN backward is differentiable on our config
   (we intentionally lack fla/causal-conv1d -> pure-torch autograd path).
3. Assert J_17 is finite and report peak VRAM so we can pick dim_batch for the
   real fit.

Usage:  .venv/bin/python scripts/jlens/1_jlens_probe.py
"""

import logging
import os
import sys

import torch

import jlens  # noqa: E402  (import the installed package BEFORE our folder can shadow it)

# Append (not insert) so site-packages' `jlens` wins over our sibling scripts/jlens/ dir.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import common  # noqa: E402

LAYER = 17
PROMPTS = [
    "The lighthouse keeper climbed the spiral stairs every evening at dusk, "
    "trimming the wick and polishing the great glass lens until the beam swept "
    "cleanly across the dark and restless water below the cliffs.",
    "A heap-based buffer overflow in the sudo command allows a local attacker "
    "to escalate privileges to root by supplying a crafted command-line "
    "argument that overflows a fixed-size buffer during argument parsing.",
]


def find_lm_head(model):
    """Return (dotted_path, module) of the unembedding, or (None, None)."""
    for path in ("lm_head", "model.language_model.lm_head", "language_model.lm_head",
                 "model.lm_head"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, torch.nn.Module) and hasattr(obj, "weight"):
            return path, obj
    return None, None


def main():
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("jlens").setLevel(logging.INFO)
    model_id, kind = common.resolve("qwen35-4b")
    print(f"Loading {model_id} ({kind}) ...")
    handler, model, tok, is_mm = common.load_model(model_id, kind)

    print("\n--- top-level children ---")
    print([n for n, _ in model.named_children()])
    tcfg = model.config.get_text_config()
    print(f"num_hidden_layers={tcfg.num_hidden_layers}  hidden_size={tcfg.hidden_size}")
    lpath, lname = common.get_decoder_layers(model)
    print(f"decoder layers at: {lpath}  (n={len(lname)})")

    lm_path, lm_mod = find_lm_head(model)
    print(f"lm_head found at: {lm_path}  ({type(lm_mod).__name__ if lm_mod else None})")
    top_level_head = hasattr(model, "lm_head")
    print(f"top-level model.lm_head exists: {top_level_head}")

    # Build the wrapper. Prefer auto-detect; if lm_head isn't top-level, alias it.
    layout = None
    if not top_level_head:
        if lm_mod is None:
            raise SystemExit("NO-GO: could not locate an unembedding on this model")
        model.lm_head = lm_mod  # top-level alias so getattr(hf_model,'lm_head') works
        layout = jlens.Layout(path="model.language_model")
        print(f"aliased model.lm_head <- {lm_path}; using explicit {layout}")

    m = jlens.from_hf(model, tok, layout=layout)
    print(f"\nwrapped: {m!r}")
    print(f"detected layout: {m.layout}")

    # --- GDN backward smoke test: fit on 2 prompts at layer 17 ---
    dev = m.input_device
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    print(f"\nRunning jlens.fit on {len(PROMPTS)} prompts, source_layers=[{LAYER}], dim_batch=8 ...")
    lens = jlens.fit(m, PROMPTS, source_layers=[LAYER], dim_batch=8,
                     max_seq_len=128, checkpoint_path=None)

    J = lens.jacobians[LAYER]
    finite = bool(torch.isfinite(J).all())
    print("\n--- RESULT ---")
    print(f"J_{LAYER} shape={tuple(J.shape)}  dtype={J.dtype}")
    print(f"finite={finite}  ||J||/sqrt(d)={J.norm().item()/ (tcfg.hidden_size**0.5):.4f}  "
          f"n_prompts={lens.n_prompts}")
    if dev.type == "cuda":
        peak = torch.cuda.max_memory_allocated(dev) / 1e9
        print(f"peak VRAM during fit (dim_batch=8): {peak:.2f} GB")
    print("GO" if finite and lens.n_prompts == len(PROMPTS) else "NO-GO")


if __name__ == "__main__":
    main()
