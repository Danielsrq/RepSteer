"""Does degeneration under steering depend on generation length?

Steers the 'happy' concept vector at each alpha, generates long (1000-token) samples
(sampling, so runs differ), and finds the loop-onset position: the first point where
a sliding 40-word window's rep-4 crosses --thresh. Tests whether a weak steer (+0.2)
eventually degenerates given enough length, and whether a strong steer (+0.6) does so
sooner.

Usage:  python scripts/16_length_degen.py --alphas 0.2,0.6 --runs 10 --new-tokens 1000
"""

import argparse
import json

import numpy as np
import torch

import common

PROMPT = "Write a long, detailed, multi-paragraph story about a walk through a city at the end of the day."


def windowed_onset(text, W=40, step=10, thr=0.5):
    w = text.split()
    traj = []
    for s in range(0, max(1, len(w) - W), step):
        win = w[s:s + W]
        g = list(zip(win, win[1:], win[2:], win[3:]))
        traj.append((s + W, round(1 - len(set(g)) / len(g), 3) if g else 0.0))
    onset = next((pos for pos, r in traj if r > thr), None)
    maxr = max((r for _, r in traj), default=0.0)
    return onset, maxr, len(w), traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="qwen35-4b")
    ap.add_argument("--vectors", default="data/concepts/concept_vectors.pt")
    ap.add_argument("--concept", default="happy")
    ap.add_argument("--alphas", default="0.2,0.6")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--new-tokens", type=int, default=1000)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--out", default="data/concepts/length_degen.json")
    ap.add_argument("--md", default="data/concepts/length_degen_samples.md",
                    help="human-readable dump of every generation")
    args = ap.parse_args()

    alphas = [float(x) for x in args.alphas.split(",")]
    saved = torch.load(args.vectors, weights_only=False)
    L = int(saved["primary_layer"])
    ci = saved["concepts"].index(args.concept)
    v = saved["concept_vectors"][L].numpy()[ci].astype(np.float32)
    u = v / (np.linalg.norm(v) + 1e-12)

    model_id, kind = common.resolve(args.preset)
    print(f"Loading {model_id} ... | steer '{args.concept}' @ layer {L} | {args.new_tokens} tokens x {args.runs} runs")
    handler, model, tok, is_mm = common.load_model(model_id, kind)
    _, layers = common.get_decoder_layers(model)
    target = layers[L]
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    u_t = torch.tensor(u).to(dev, dt)

    messages = [{"role": "user", "content": PROMPT}]
    kw = dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    try:
        enc = handler.apply_chat_template(messages, enable_thinking=False, **kw)
    except TypeError:
        enc = handler.apply_chat_template(messages, **kw)
    enc = {k: v_.to(dev) for k, v_ in enc.items()}
    n_prompt = enc["input_ids"].shape[1]

    def make_inject(alpha):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h + alpha * h.norm(dim=-1, keepdim=True) * u_t
            return (h, *out[1:]) if isinstance(out, tuple) else h
        return hook

    results = {}
    for ai, alpha in enumerate(alphas):
        runs = []
        hd = target.register_forward_hook(make_inject(alpha))
        try:
            for r in range(args.runs):
                torch.manual_seed(1000 * ai + r)
                with torch.inference_mode():
                    out = model.generate(**enc, max_new_tokens=args.new_tokens,
                                         min_new_tokens=args.new_tokens, do_sample=True,
                                         temperature=0.9, top_p=0.95, pad_token_id=tok.eos_token_id)
                text = tok.decode(out[0, n_prompt:], skip_special_tokens=True).strip()
                onset, maxr, nwords, traj = windowed_onset(text, thr=args.thresh)
                runs.append({"run": r, "onset": onset, "max_rep4": maxr, "words": nwords,
                             "traj": traj, "text": text})
                print(f"  a={alpha:+.1f} run {r}: onset={onset}  max_rep4={maxr:.2f}  words={nwords}")
        finally:
            hd.remove()
        results[f"{alpha:.1f}"] = runs

    # summary
    print(f"\n{'alpha':>6} | {'#degen/N':>9} | {'median onset':>13} | {'mean max_rep4':>13}")
    print("-" * 52)
    for alpha in alphas:
        rs = results[f"{alpha:.1f}"]
        onsets = [x["onset"] for x in rs if x["onset"] is not None]
        ndeg = len(onsets)
        med = int(np.median(onsets)) if onsets else None
        mmr = float(np.mean([x["max_rep4"] for x in rs]))
        print(f"{alpha:+6.1f} | {ndeg:2d}/{len(rs):<6d} | {str(med):>13} | {mmr:13.3f}")

    json.dump({"concept": args.concept, "prompt": PROMPT, "layer": L, "thresh": args.thresh,
               "new_tokens": args.new_tokens, "results": results}, open(args.out, "w"))
    print(f"\nSaved trajectories -> {args.out}")

    with open(args.md, "w") as f:
        f.write(f"# Length-degeneration samples — steering '{args.concept}' (sampling, temp 0.9, top_p 0.95)\n\n")
        f.write(f"Prompt: *{PROMPT}*\n\n")
        for alpha in alphas:
            f.write(f"\n## alpha = +{alpha:.1f}\n\n")
            for x in results[f"{alpha:.1f}"]:
                f.write(f"### run {x['run']}  (onset={x['onset']}, max_rep4={x['max_rep4']:.2f}, {x['words']} words)\n\n")
                f.write(x["text"] + "\n\n---\n")
    print(f"Saved readable samples -> {args.md}")


if __name__ == "__main__":
    main()
