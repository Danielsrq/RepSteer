"""Per-token projection onto the happy-sad difference-of-means (DiM) direction.

For a chosen layer we take EACH token's residual-stream vector (no pooling) and
dot it with the unit DiM direction u = (mean_happy - mean_sad) / |...|. That
scalar says how far that token leans happy (+) or sad (-). We subtract the
classifier midpoint (the sep-acc boundary) so 0 = the happy/sad decision line,
then render every token as a colored span in an HTML file (blue=sad, red=happy).

Needs data/activations.pt (from 03) for the DiM direction + midpoint, and
data/paragraphs.json for the texts.

Usage:
    python scripts/04_token_heatmap.py                       # best layer, all paragraphs
    python scripts/04_token_heatmap.py --layer 17 --index 12,13
    python scripts/04_token_heatmap.py --out data/heatmap.html
"""

import argparse
import html
import json

import numpy as np
import torch

import common


def color_for(v: float) -> str:
    """v in [-1,1]; blue for negative (sad), red for positive (happy), white at 0."""
    f = min(abs(v), 1.0)
    white = np.array([255, 255, 255])
    happy = np.array([214, 40, 40])     # red
    sad = np.array([33, 102, 172])      # blue
    rgb = white * (1 - f) + (happy if v >= 0 else sad) * f
    r, g, b = rgb.astype(int)
    return f"rgb({r},{g},{b})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(common.PRESETS), default="qwen35-4b")
    ap.add_argument("--model", default=None)
    ap.add_argument("--kind", choices=["text", "multimodal"], default=None)
    ap.add_argument("--acts", default="data/activations.pt")
    ap.add_argument("--data", default="data/paragraphs.json")
    ap.add_argument("--layer", type=int, default=None, help="default: best layer from acts")
    ap.add_argument("--index", default="all", help="comma-separated paragraph ids, or 'all'")
    ap.add_argument("--out", default="data/heatmap.html")
    args = ap.parse_args()

    saved = torch.load(args.acts, weights_only=False)
    L = args.layer if args.layer is not None else int(saved["best_layer"])
    diff = saved["diff_per_layer"][L].numpy().astype(np.float32)   # [hidden]
    u = diff / (np.linalg.norm(diff) + 1e-9)                       # unit DiM axis

    # classifier midpoint on this axis, from the pooled paragraph vectors
    feats = saved["features"][:, L].numpy().astype(np.float32)     # [N, hidden]
    labels = saved["labels"].numpy()
    proj = feats @ u
    midpoint = 0.5 * (proj[labels == 1].mean() + proj[labels == 0].mean())

    with open(args.data) as f:
        rows = json.load(f)["rows"]
    if args.index == "all":
        idxs = list(range(len(rows)))
    else:
        idxs = [int(x) for x in args.index.split(",")]

    model_id, kind = common.resolve(args.preset, args.model, args.kind)
    print(f"Loading {model_id} ... | layer {L} | midpoint {midpoint:.2f}")
    handler, model, tok, is_mm = common.load_model(model_id, kind)
    _, layers = common.get_decoder_layers(model)
    target = layers[L]

    grab = {}

    def hook(_m, _i, output):
        grab["h"] = common.layer_output_hidden(output)[0].float().cpu()  # [seq, hidden], no pooling
    handle = target.register_forward_hook(hook)

    # score every token of every selected paragraph
    all_tokens = []   # list of (row, [(token_str, score)])
    try:
        for i in idxs:
            enc = tok(rows[i]["text"], return_tensors="pt").to("cuda")
            with torch.inference_mode():
                model(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"))
            scores = (grab["h"].numpy() @ u) - midpoint            # [seq]
            ids = enc["input_ids"][0].tolist()
            toks = [tok.decode([t]) for t in ids]
            all_tokens.append((rows[i], list(zip(toks, scores))))
    finally:
        handle.remove()

    # normalize colors by a robust scale across all shown tokens
    flat = np.array([s for _, ts in all_tokens for _, s in ts])
    scale = np.percentile(np.abs(flat), 98) + 1e-9

    # build HTML
    blocks = []
    for row, ts in all_tokens:
        mean_lean = np.mean([s for _, s in ts])
        spans = []
        for tokstr, s in ts:
            v = float(np.clip(s / scale, -1, 1))
            safe = html.escape(tokstr)
            spans.append(f'<span style="background:{color_for(v)}" title="{s:+.2f}">{safe}</span>')
        blocks.append(
            f'<div class="para"><div class="hdr">[{row["id"]:2d}] '
            f'<b>{row["emotion"]}</b> &middot; {html.escape(row["topic"])} '
            f'&middot; mean lean {mean_lean:+.2f}</div>'
            f'<div class="txt">{"".join(spans)}</div></div>'
        )

    doc = f"""<!doctype html><meta charset="utf-8">
<title>Happy/Sad token heatmap - layer {L}</title>
<style>
 body{{font:15px/1.7 -apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#111;background:#fff}}
 h1{{font-size:1.2rem}} .sub{{color:#555;margin-bottom:1.5rem}}
 .para{{margin:1.2rem 0;border:1px solid #eee;border-radius:8px;padding:.6rem .8rem}}
 .hdr{{font-size:.8rem;color:#444;margin-bottom:.4rem}}
 .txt{{white-space:pre-wrap}}
 .txt span{{border-radius:2px}}
 .legend{{display:inline-block}}
 .legend span{{padding:2px 10px}}
</style>
<h1>Per-token overlap with the happy&minus;sad direction &mdash; layer {L}</h1>
<div class="sub">Each token colored by (token activation &middot; unit DiM) &minus; midpoint.
 <span class="legend"><span style="background:{color_for(-1)}">sad</span><span style="background:{color_for(-0.4)}"></span><span style="background:{color_for(0)}">neutral</span><span style="background:{color_for(0.4)}"></span><span style="background:{color_for(1)}">happy</span></span>
 &nbsp; Hover a token for its raw score.</div>
{''.join(blocks)}
"""
    with open(args.out, "w") as f:
        f.write(doc)
    print(f"Wrote {len(all_tokens)} paragraphs -> {args.out}")


if __name__ == "__main__":
    main()
