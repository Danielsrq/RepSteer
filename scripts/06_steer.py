"""Activation steering: inject the happy-sad DiM direction during generation.

Registers a forward hook on the chosen layer that ADDS a signed multiple of the
difference-of-means direction to the residual stream, then generates the same
prompt across a sweep of coefficients. Negative -> sad, 0 -> baseline, positive
-> happy (since diff = mean_happy - mean_sad, and sad->happy is just -diff).

Coefficient scaling (how alpha maps to an added magnitude):
  norm  (default) add = alpha * ||h||_local * unit(diff)   alpha is a FRACTION of
                  the local activation norm; ~[-1,1] meaningful, layer-independent.
  diff            add = alpha * |diff| * unit(diff) = alpha * diff   alpha in units
                  of the happy/sad inter-class gap.
  raw             add = alpha * unit(diff)                 alpha is a raw magnitude.

Positions steered per forward pass:
  all (default) every position   gen only generated tokens (skip the prompt)
  last only the final position

Usage:
    python scripts/06_steer.py
    python scripts/06_steer.py --alphas " -0.6,-0.3,0,0.3,0.6" --layer 17
    python scripts/06_steer.py --scale diff --alphas "-8,-4,0,4,8"
    python scripts/06_steer.py --prompt "Describe the weather today." --positions gen
"""

import argparse

import torch

import common


def make_inject_hook(u, diff_norm, alpha, scale, positions):
    def hook(_module, _inp, output):
        h = output[0] if isinstance(output, tuple) else output      # [B, S, hidden]
        B, S, H = h.shape
        if scale == "norm":
            base = alpha * h.norm(dim=-1, keepdim=True) * u          # [B, S, hidden]
        elif scale == "diff":
            base = (alpha * diff_norm) * u                           # [hidden]
            base = base.expand(B, S, H)
        else:  # raw
            base = (alpha * u).expand(B, S, H)

        if positions == "all":
            mask = 1.0
        else:
            mask = torch.zeros(B, S, 1, device=h.device, dtype=h.dtype)
            if positions == "last":
                mask[:, -1:, :] = 1.0
            elif positions == "gen":
                if S == 1:            # decode step -> this IS a generated token
                    mask[:] = 1.0
                # prefill (S>1) -> leave prompt untouched
        h = h + base * mask
        return (h, *output[1:]) if isinstance(output, tuple) else h
    return hook


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(common.PRESETS), default="qwen35-4b")
    ap.add_argument("--model", default=None)
    ap.add_argument("--kind", choices=["text", "multimodal"], default=None)
    ap.add_argument("--acts", default="data/activations.pt")
    ap.add_argument("--layer", type=int, default=None, help="default: best layer from acts")
    ap.add_argument("--scale", choices=["norm", "diff", "raw"], default="norm")
    ap.add_argument("--alphas", default=None, help="comma list; default depends on --scale")
    ap.add_argument("--positions", choices=["all", "gen", "last"], default="all")
    ap.add_argument("--prompt",
                    default="Write a short paragraph describing a walk through a city at the end of the day.")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--sample", action="store_true", help="sample instead of greedy")
    ap.add_argument("--out", default=None, help="also write the full sweep transcript here")
    args = ap.parse_args()

    buf = []

    def emit(s=""):
        print(s)
        buf.append(s)

    defaults = {"norm": "-1,-0.5,0,0.5,1", "diff": "-8,-4,0,4,8", "raw": "-8,-4,0,4,8"}
    alphas = [float(x) for x in (args.alphas or defaults[args.scale]).split(",")]

    saved = torch.load(args.acts, weights_only=False)
    L = args.layer if args.layer is not None else int(saved["best_layer"])
    diff = saved["diff_per_layer"][L]
    diff_norm = float(diff.norm())

    model_id, kind = common.resolve(args.preset, args.model, args.kind)
    print(f"Loading {model_id} ... | steer layer {L} | scale={args.scale} | positions={args.positions}")
    handler, model, tok, is_mm = common.load_model(model_id, kind)
    _, layers = common.get_decoder_layers(model)
    target = layers[L]

    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    u = (diff / (diff_norm + 1e-9)).to(dev, dt)      # unit DiM direction

    # fixed chat inputs (text-only), thinking off
    messages = [{"role": "user", "content": args.prompt}]
    kw = dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    try:
        enc = handler.apply_chat_template(messages, enable_thinking=False, **kw)
    except TypeError:
        enc = handler.apply_chat_template(messages, **kw)
    enc = {k: v.to(dev) for k, v in enc.items()}
    n_prompt = enc["input_ids"].shape[1]

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tok.eos_token_id)
    if args.sample:
        gen_kwargs.update(do_sample=True, temperature=0.8, top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False)

    emit(f'\nPROMPT: "{args.prompt}"  | layer {L} | scale={args.scale} | positions={args.positions}')
    emit("=" * 78)
    for a in alphas:
        tag = "baseline" if a == 0 else ("happy +" if a > 0 else "sad -")
        handle = target.register_forward_hook(make_inject_hook(u, diff_norm, a, args.scale, args.positions))
        try:
            with torch.inference_mode():
                out = model.generate(**enc, **gen_kwargs)
        finally:
            handle.remove()
        text = tok.decode(out[0, n_prompt:], skip_special_tokens=True).strip()
        emit(f"\n[ alpha = {a:+.2f}  ({tag}) ]")
        emit(text)
        emit("-" * 78)

    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(buf))
        print(f"\nWrote transcript -> {args.out}")


if __name__ == "__main__":
    main()
