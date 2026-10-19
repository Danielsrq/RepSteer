"""Story-arc activation trajectories.

Generates N short 3-paragraph stories (no steering), splits each into sentences,
mean-pools the layer-L residual stream over each sentence -> one vector per sentence,
runs global PCA over all sentence vectors, and renders:
  (1) a phase portrait (PC2 vs PC1) with one polyline per story (time implicit in the
      connect-the-dots order; points colored by relative position intro->ending), and
  (2) component-vs-step plots (PC1 and PC2 vs relative position; time explicit).

Usage:  python scripts/17_arc_trajectory.py --stories 10 --new-tokens 500
"""

import argparse
import html
import json
import os
import re

import numpy as np
import torch

import common

TOPICS = [
    "an old lighthouse keeper", "a child's first day at a new school", "a stray dog in a big city",
    "a train journey across the mountains", "a small bakery on a quiet street",
    "two friends repairing an old boat", "a gardener and a rare flower",
    "a night shift at a hospital", "a musician returning to their hometown",
    "a letter discovered in an attic",
]
PROMPT = ("Write a short story in exactly three paragraphs about {topic}: "
          "paragraph 1 sets the scene, paragraph 2 develops the main events, "
          "paragraph 3 gives the ending. Write only the story.")


def sentence_spans(text):
    spans = [(m.start(), m.end()) for m in re.finditer(r"[^.!?]*[.!?]+", text)]
    tail_start = spans[-1][1] if spans else 0
    if tail_start < len(text.rstrip()):
        spans.append((tail_start, len(text)))
    return [(a, b) for a, b in spans if len(text[a:b].strip()) >= 8]


def pos_color(t):  # t in [0,1]: blue (early) -> red (late)
    b, r = np.array([33, 102, 172]), np.array([214, 40, 40])
    c = (b * (1 - t) + r * t).astype(int)
    return f"rgb({c[0]},{c[1]},{c[2]})"


def line_plot_svg(series, ylabel, W=460, H=200):
    pad = 34
    ys = [v for s in series for v in s]
    ymin, ymax = min(ys), max(ys)
    yr = (ymax - ymin) or 1.0

    def sx(t):
        return pad + t * (W - 2 * pad)

    def sy(v):
        return (H - pad) - (v - ymin) / yr * (H - 2 * pad)

    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px">']
    p.append(f'<line x1="{pad}" y1="{sy(0) if ymin<0<ymax else H-pad:.1f}" x2="{W-pad}" '
             f'y2="{sy(0) if ymin<0<ymax else H-pad:.1f}" stroke="#e1e0d9"/>')
    for s in series:
        n = len(s)
        pts = " ".join(f"{sx(i/(n-1) if n>1 else 0):.1f},{sy(v):.1f}" for i, v in enumerate(s))
        p.append(f'<polyline points="{pts}" fill="none" stroke="#8aa" stroke-width="1" opacity="0.5"/>')
        for i, v in enumerate(s):
            p.append(f'<circle cx="{sx(i/(n-1) if n>1 else 0):.1f}" cy="{sy(v):.1f}" r="2.4" '
                     f'fill="{pos_color(i/(n-1) if n>1 else 0)}"/>')
    p.append(f'<text x="{W/2:.0f}" y="{H-6}" text-anchor="middle" font-size="11" fill="#555">relative step (intro→ending)</text>')
    p.append(f'<text x="12" y="{H/2:.0f}" text-anchor="middle" font-size="11" fill="#555" transform="rotate(-90 12 {H/2:.0f})">{ylabel}</text>')
    p.append("</svg>")
    return "".join(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="qwen35-4b")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--stories", type=int, default=10)
    ap.add_argument("--new-tokens", type=int, default=500)
    ap.add_argument("--out", default="data/arc/arc_trajectory.html")
    args = ap.parse_args()

    model_id, kind = common.resolve(args.preset)
    print(f"Loading {model_id} ... | layer {args.layer} | {args.stories} stories")
    handler, model, tok, is_mm = common.load_model(model_id, kind)
    _, layers = common.get_decoder_layers(model)
    target, dev = layers[args.layer], next(model.parameters()).device
    grab = {}

    def read_hook(_m, _i, out):
        grab["h"] = common.layer_output_hidden(out)[0].float().cpu().numpy()
    handle = target.register_forward_hook(read_hook)

    stories = []      # list of {topic, sentences:[str], vecs:[d]}
    try:
        for si in range(args.stories):
            topic = TOPICS[si % len(TOPICS)]
            messages = [{"role": "user", "content": PROMPT.format(topic=topic)}]
            kw = dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
            try:
                enc = handler.apply_chat_template(messages, enable_thinking=False, **kw)
            except TypeError:
                enc = handler.apply_chat_template(messages, **kw)
            enc = {k: v.to(dev) for k, v in enc.items()}
            torch.manual_seed(si)
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=args.new_tokens, do_sample=True,
                                     temperature=0.9, top_p=0.95, pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            # forward pass over the produced text -> per-token activations at layer L
            enc2 = tok(text, return_tensors="pt").to(dev)
            with torch.inference_mode():
                model(input_ids=enc2["input_ids"], attention_mask=enc2.get("attention_mask"))
            H = grab["h"]                                        # [seq, d]
            ids = enc2["input_ids"][0].tolist()
            pieces = [tok.decode([i]) for i in ids]
            recon = "".join(pieces)
            offs = np.cumsum([0] + [len(p) for p in pieces])     # token i covers [offs[i], offs[i+1])
            spans = sentence_spans(recon)
            svecs, stexts = [], []
            for (a, b) in spans:
                idx = [i for i in range(len(ids)) if offs[i] >= a and offs[i] < b]
                if not idx:
                    continue
                svecs.append(H[idx].mean(0))
                stexts.append(recon[a:b].strip())
            if len(svecs) >= 2:
                stories.append({"topic": topic, "sentences": stexts, "vecs": np.stack(svecs)})
                print(f"  story {si} ({topic}): {len(svecs)} sentences")
    finally:
        handle.remove()

    # global PCA over all sentence vectors
    allv = np.concatenate([s["vecs"] for s in stories], 0).astype(np.float64)
    mu = allv.mean(0)
    Xc = allv - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    evr = (S**2) / (S**2).sum()
    PCs = Vt[:3]
    # project each story
    for s in stories:
        s["proj"] = ((s["vecs"].astype(np.float64) - mu) @ PCs.T)   # [n,3]

    # --- phase portrait (PC2 vs PC1) ---
    P1 = np.concatenate([s["proj"][:, 0] for s in stories])
    P2 = np.concatenate([s["proj"][:, 1] for s in stories])
    W = H_ = 520; pad = 40
    x0, x1 = P1.min(), P1.max(); y0, y1 = P2.min(), P2.max()
    xr, yr = (x1 - x0) or 1, (y1 - y0) or 1

    def sx(v):
        return pad + (v - x0) / xr * (W - 2 * pad)

    def sy(v):
        return (H_ - pad) - (v - y0) / yr * (H_ - 2 * pad)

    pp = [f'<svg viewBox="0 0 {W} {H_}" width="100%" style="max-width:{W}px">']
    for s in stories:
        pr = s["proj"]; n = len(pr)
        pts = " ".join(f"{sx(pr[i,0]):.1f},{sy(pr[i,1]):.1f}" for i in range(n))
        pp.append(f'<polyline points="{pts}" fill="none" stroke="#bbb" stroke-width="1" opacity="0.6"/>')
        for i in range(n):
            t = i / (n - 1) if n > 1 else 0
            r = 5 if (i == 0 or i == n - 1) else 3
            stroke = 'stroke="#0b0b0b" stroke-width="1.2"' if (i == 0 or i == n - 1) else ""
            pp.append(f'<circle cx="{sx(pr[i,0]):.1f}" cy="{sy(pr[i,1]):.1f}" r="{r}" fill="{pos_color(t)}" {stroke}><title>{html.escape(s["sentences"][i][:80])}</title></circle>')
    pp.append(f'<text x="{W/2:.0f}" y="{H_-8}" text-anchor="middle" font-size="12" fill="#555">PC1 ({evr[0]*100:.1f}% var)</text>')
    pp.append(f'<text x="14" y="{H_/2:.0f}" text-anchor="middle" font-size="12" fill="#555" transform="rotate(-90 14 {H_/2:.0f})">PC2 ({evr[1]*100:.1f}% var)</text>')
    pp.append("</svg>")

    pc1_series = [list(s["proj"][:, 0]) for s in stories]
    pc2_series = [list(s["proj"][:, 1]) for s in stories]

    doc = f"""<!doctype html><meta charset="utf-8"><title>Story-arc activation trajectories</title>
<style>body{{font:15px/1.6 system-ui,-apple-system,sans-serif;max-width:1100px;margin:1.5rem auto;padding:0 1rem;color:#111;background:#fcfcfb}}
h1{{font-size:1.2rem}} h2{{font-size:1rem;margin:1.4rem 0 .3rem}} .note{{color:#555;font-size:.85rem}}
.row{{display:flex;gap:1.5rem;flex-wrap:wrap;align-items:flex-start}} .lg{{display:inline-block;padding:2px 8px;border-radius:3px;color:#fff}}</style>
<h1>Story-arc activation trajectories — layer {args.layer}</h1>
<div class="note">{len(stories)} unsteered 3-paragraph stories, mean-pooled per sentence, global PCA over all
{allv.shape[0]} sentence vectors. Points colored by <b>relative position</b>:
<span class="lg" style="background:{pos_color(0)}">intro</span>
<span class="lg" style="background:{pos_color(0.5)}">middle</span>
<span class="lg" style="background:{pos_color(1)}">ending</span>. First/last sentence of each story ringed.
Top-3 explained variance: {evr[0]*100:.1f}% / {evr[1]*100:.1f}% / {evr[2]*100:.1f}%.</div>
<h2>Phase portrait — PC2 vs PC1 (time implicit in the path)</h2>
{''.join(pp)}
<div class="row">
<div><h2>PC1 vs step (time explicit)</h2>{line_plot_svg(pc1_series, "PC1")}</div>
<div><h2>PC2 vs step</h2>{line_plot_svg(pc2_series, "PC2")}</div>
</div>
"""
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(doc)
    json.dump({"layer": args.layer, "evr": evr[:5].tolist(),
               "stories": [{"topic": s["topic"], "sentences": s["sentences"],
                            "proj": s["proj"].round(3).tolist()} for s in stories]},
              open(args.out.replace(".html", ".json"), "w"))
    print(f"\nWrote {args.out}  | top-3 EVR = {evr[:3].round(3)}")


if __name__ == "__main__":
    main()
