"""Measure the generation overhead of forward hooks.

Compares decode throughput (tok/s) with: no hook; the steering inject hook (1 layer,
norm-relative add); a read hook on 1 layer; read hooks on all 32 layers keeping
activations on GPU; and read-all with .cpu() copies (the capture-in-03 style).

Usage:  python scripts/13_hook_overhead.py --new-tokens 200 --trials 3
"""

import argparse
import time

import torch

import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="qwen35-4b")
    ap.add_argument("--new-tokens", type=int, default=200)
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    model_id, kind = common.resolve(args.preset)
    print(f"Loading {model_id} ...")
    handler, model, tok, is_mm = common.load_model(model_id, kind)
    _, layers = common.get_decoder_layers(model)
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    hidden = model.config.get_text_config().hidden_size
    u = torch.randn(hidden, device=dev, dtype=dt); u = u / u.norm()

    messages = [{"role": "user", "content": "Write a detailed paragraph about a walk through a city."}]
    kw = dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    try:
        enc = handler.apply_chat_template(messages, enable_thinking=False, **kw)
    except TypeError:
        enc = handler.apply_chat_template(messages, **kw)
    enc = {k: v.to(dev) for k, v in enc.items()}
    n_prompt = enc["input_ids"].shape[1]
    gk = dict(max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens,
              do_sample=False, pad_token_id=tok.eos_token_id)

    grab = {}

    def inject(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h + 0.1 * h.norm(dim=-1, keepdim=True) * u
        return (h, *out[1:]) if isinstance(out, tuple) else h

    def read_gpu(idx):
        def hook(_m, _i, out):
            grab[idx] = common.layer_output_hidden(out)[0]           # stays on GPU
        return hook

    def read_cpu(idx):
        def hook(_m, _i, out):
            grab[idx] = common.layer_output_hidden(out)[0].float().cpu()   # GPU->CPU copy
        return hook

    def timed():
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**enc, **gk)
        torch.cuda.synchronize()
        return (out.shape[1] - n_prompt) / (time.perf_counter() - t0)

    def bench(name, register):
        handles = register()
        with torch.inference_mode():                                 # warmup
            model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        rates = [timed() for _ in range(args.trials)]
        for h in handles:
            h.remove()
        return max(rates)                                            # best (least noisy)

    configs = [
        ("no hook", lambda: []),
        ("inject @ layer17 (steering)", lambda: [layers[17].register_forward_hook(inject)]),
        ("read @ layer17 (1 layer)", lambda: [layers[17].register_forward_hook(read_gpu(17))]),
        ("read @ all 32 (keep on GPU)", lambda: [l.register_forward_hook(read_gpu(i)) for i, l in enumerate(layers)]),
        ("read @ all 32 (.cpu() copy)", lambda: [l.register_forward_hook(read_cpu(i)) for i, l in enumerate(layers)]),
    ]
    print(f"\n{args.new_tokens} new tokens, greedy, best of {args.trials}\n")
    base = None
    print(f"{'config':32s} {'tok/s':>8} {'vs baseline':>12}")
    print("-" * 56)
    for name, reg in configs:
        r = bench(name, reg)
        if base is None:
            base = r
        print(f"{name:32s} {r:8.1f} {(100*(r-base)/base):+11.1f}%")


if __name__ == "__main__":
    main()
