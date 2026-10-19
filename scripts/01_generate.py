"""Interactive generation — type a prompt, see the model's text.

Loads a model once, then loops: enter a prompt, watch it stream a reply, repeat.
Same two model families as the smoke test (text + multimodal, text-only prompts).

Usage:
    python scripts/01_generate.py                       # Qwen3-4B-Instruct-2507 (text), REPL
    python scripts/01_generate.py --preset qwen35-4b    # Qwen3.5-4B (multimodal)
    python scripts/01_generate.py --prompt "Write a haiku about GPUs."   # one-shot, then exit
    python scripts/01_generate.py --greedy --max-new-tokens 512
    python scripts/01_generate.py --preset qwen35-4b --no-think          # disable thinking trace

REPL controls: type your prompt and press Enter. Empty line, 'exit'/'quit', or
Ctrl-D leaves. Ctrl-C aborts the current generation without quitting.
"""

import argparse

import torch
from transformers import TextStreamer

PRESETS = {
    "qwen35-4b": ("Qwen/Qwen3.5-4B", "multimodal"),
    "qwen3-text": ("Qwen/Qwen3-4B-Instruct-2507", "text"),
}


def load(model_id: str, kind: str, device: str):
    """Return (handler, model, is_multimodal). handler = processor or tokenizer."""
    if kind == "multimodal":
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        handler = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForMultimodalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device
        )
        return handler, model, True
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        handler = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device
        )
        return handler, model, False


def build_inputs(handler, prompt: str, system: str | None, think: bool, device: str):
    """Text-only chat inputs as a dict. Tries to honor enable_thinking if supported."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = dict(add_generation_prompt=True, tokenize=True,
                  return_dict=True, return_tensors="pt")
    try:
        enc = handler.apply_chat_template(messages, enable_thinking=think, **kwargs)
    except TypeError:
        # template doesn't accept enable_thinking — fall back
        enc = handler.apply_chat_template(messages, **kwargs)
    return {k: v.to(device) for k, v in enc.items()}


def generate(model, handler, tok, inputs, args, is_mm):
    n_prompt = inputs["input_ids"].shape[1]
    streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tok.eos_token_id,
        streamer=streamer,
    )
    if args.greedy:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs.update(do_sample=True, temperature=args.temperature,
                          top_p=args.top_p, top_k=args.top_k)

    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    return out.shape[1] - n_prompt  # new-token count (text already streamed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="qwen3-text")
    ap.add_argument("--model", default=None, help="override preset model id")
    ap.add_argument("--kind", choices=["text", "multimodal"], default=None)
    ap.add_argument("--prompt", default=None, help="one-shot prompt; omit for REPL")
    ap.add_argument("--system", default=None, help="optional system prompt")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--greedy", action="store_true", help="deterministic decoding")
    ap.add_argument("--no-think", dest="think", action="store_false",
                    help="disable thinking trace (if the model supports it)")
    ap.set_defaults(think=True)
    args = ap.parse_args()

    model_id, kind = PRESETS[args.preset]
    model_id = args.model or model_id
    kind = args.kind or kind

    assert torch.cuda.is_available(), "CUDA not available — check the torch install"
    device = "cuda"
    print(f"Loading {model_id} (kind={kind}) ...")
    handler, model, is_mm = load(model_id, kind, device)
    model.eval()
    tok = handler.tokenizer if is_mm else handler
    print(f"Ready. sampling={'greedy' if args.greedy else f'T={args.temperature} top_p={args.top_p} top_k={args.top_k}'}"
          f" | max_new_tokens={args.max_new_tokens} | think={args.think}\n")

    def run(prompt: str):
        inputs = build_inputs(handler, prompt, args.system, args.think, device)
        print("\n\033[2m--- model ---\033[0m")
        try:
            generate(model, handler, tok, inputs, args, is_mm)
        except KeyboardInterrupt:
            print("\n\033[2m[aborted]\033[0m")
        print()

    # one-shot mode
    if args.prompt is not None:
        run(args.prompt)
        return

    # REPL mode
    print("Enter a prompt (empty line / 'exit' / Ctrl-D to quit):")
    while True:
        try:
            prompt = input("\n\033[1m>>> \033[0m").strip()
        except EOFError:
            print()
            break
        if prompt.lower() in ("", "exit", "quit"):
            break
        run(prompt)


if __name__ == "__main__":
    main()
