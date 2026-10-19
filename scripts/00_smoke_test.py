"""Smoke test + throughput benchmark for a HF causal/multimodal LM on one GPU.

Loads in bf16, warms up, then times text generation to report decode
throughput (new tokens/sec) and peak VRAM. Tells us empirically whether a
model size is comfortable on 12 GB or if we should drop to a smaller one.

Two model families are supported:
  * text      -> AutoTokenizer + AutoModelForCausalLM   (e.g. Qwen3-4B-Instruct-2507)
  * multimodal-> AutoProcessor  + AutoModelForMultimodalLM (e.g. Qwen3.5-4B, text-only prompt)

Usage:
    python scripts/00_smoke_test.py                              # Qwen3.5-4B (multimodal)
    python scripts/00_smoke_test.py --preset qwen3-text          # Qwen3-4B-Instruct-2507
    python scripts/00_smoke_test.py --model Qwen/Qwen3.5-2B --kind multimodal
    python scripts/00_smoke_test.py --new-tokens 512
"""

import argparse
import time

import torch

PRESETS = {
    "qwen35-4b": ("Qwen/Qwen3.5-4B", "multimodal"),
    "qwen3-text": ("Qwen/Qwen3-4B-Instruct-2507", "text"),
}


def gib(n_bytes: int) -> float:
    return n_bytes / (1024**3)


def load(model_id: str, kind: str, device: str):
    """Return (processor_or_tokenizer, model, is_multimodal)."""
    if kind == "multimodal":
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        proc = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForMultimodalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device
        )
        return proc, model, True
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device
        )
        return tok, model, False


def build_inputs(handler, prompt: str, device: str):
    """Unified text-only chat inputs as a dict (input_ids + attention_mask)."""
    messages = [{"role": "user", "content": prompt}]
    enc = handler.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in enc.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="qwen35-4b")
    ap.add_argument("--model", default=None, help="override preset model id")
    ap.add_argument("--kind", choices=["text", "multimodal"], default=None)
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument(
        "--prompt",
        default="Explain what activation steering is in large language models, in about a paragraph.",
    )
    args = ap.parse_args()

    model_id, kind = PRESETS[args.preset]
    model_id = args.model or model_id
    kind = args.kind or kind

    assert torch.cuda.is_available(), "CUDA not available — check the torch install"
    device = "cuda"
    print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | GPU {torch.cuda.get_device_name(0)}")
    print(f"model: {model_id}  (kind={kind})")

    # --- load ---
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    handler, model, is_mm = load(model_id, kind, device)
    model.eval()
    load_s = time.perf_counter() - t0
    print(f"\nLoaded in {load_s:.1f}s")
    print(f"  params        : {sum(p.numel() for p in model.parameters()) / 1e9:.2f} B")
    print(f"  VRAM (weights): {gib(torch.cuda.memory_allocated()):.2f} GiB")

    tok = handler.tokenizer if is_mm else handler
    inputs = build_inputs(handler, args.prompt, device)
    n_prompt = inputs["input_ids"].shape[1]

    gen_kwargs = dict(
        max_new_tokens=args.new_tokens,
        min_new_tokens=args.new_tokens,  # force full length for stable timing
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )

    # --- warmup ---
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()

    # --- timed run ---
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    n_new = out.shape[1] - n_prompt
    print(f"\nGeneration: {n_prompt} prompt tok -> {n_new} new tok in {elapsed:.2f}s")
    print(f"  throughput    : {n_new / elapsed:.1f} tok/s")
    print(f"  peak VRAM     : {gib(torch.cuda.max_memory_allocated()):.2f} GiB "
          f"/ {gib(torch.cuda.get_device_properties(0).total_memory):.2f} GiB")

    print("\n--- sample output ---")
    print(tok.decode(out[0, n_prompt:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
