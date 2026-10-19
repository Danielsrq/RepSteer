"""Steering sweep grid: N prompts x alpha sweep, token-colored, as one HTML page.

For each prompt and each alpha we (1) generate with the DiM direction injected at
the target layer, then (2) run a SECOND, unsteered forward pass over the produced
text to score each token's happy/sad lean for coloring. The page has a dropdown to
pick the prompt; the baseline (alpha=0) is shown full-width on top, and below it a
2-column grid pairs -alpha (sad, left) with +alpha (happy, right) by magnitude.

Extensible: edit PROMPTS below, or pass --prompts-file (a JSON list of strings).
The HTML renders however many prompts the data contains.

Usage:
    python scripts/07_steer_grid.py
    python scripts/07_steer_grid.py --prompts-file my_prompts.json --step 0.2
"""

import argparse
import json
import statistics

import numpy as np
import torch

import common


def qq_svg(vals, title, color, W=340, H=300):
    """Inline-SVG normal Q-Q plot: standardized sample quantiles vs theoretical."""
    v = np.sort(np.asarray(vals, dtype=float))
    n = len(v)
    z = (v - v.mean()) / (v.std(ddof=1) + 1e-12)
    nd = statistics.NormalDist(0.0, 1.0)
    theo = [nd.inv_cdf((i + 0.5) / n) for i in range(n)]
    lim, pad = 3.0, 44

    def sx(x):
        return pad + (x + lim) / (2 * lim) * (W - 2 * pad)

    def sy(y):
        return (H - pad) - (y + lim) / (2 * lim) * (H - 2 * pad)

    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px;height:auto">']
    p.append(f'<rect x="{pad}" y="{pad}" width="{W-2*pad}" height="{H-2*pad}" fill="#fff" stroke="#ccc"/>')
    p.append(f'<line x1="{sx(-lim):.1f}" y1="{sy(-lim):.1f}" x2="{sx(lim):.1f}" y2="{sy(lim):.1f}" '
             f'stroke="#bbb" stroke-dasharray="4 3"/>')
    for tx, ty in zip(theo, z):
        cy = sy(max(-lim, min(lim, ty)))
        p.append(f'<circle cx="{sx(tx):.1f}" cy="{cy:.1f}" r="3.2" fill="{color}" opacity="0.8"/>')
    p.append(f'<text x="{W/2:.0f}" y="16" text-anchor="middle" font-size="12" font-weight="600">{title}</text>')
    p.append(f'<text x="{W/2:.0f}" y="{H-8}" text-anchor="middle" font-size="10" fill="#555">theoretical normal quantiles</text>')
    p.append(f'<text x="12" y="{H/2:.0f}" text-anchor="middle" font-size="10" fill="#555" '
             f'transform="rotate(-90 12 {H/2:.0f})">sample quantiles (z)</text>')
    p.append("</svg>")
    return "".join(p)

PROMPTS = [
    # 1-3: neutral scenes
    "Write a short paragraph describing a walk through a city at the end of the day.",
    "Write a short paragraph about returning to your childhood home.",
    "Write a short paragraph describing a train arriving at a station.",
    # 4: happy content -> can sad-steering darken it?
    "Write a short paragraph about a joyful surprise birthday party full of friends, laughter, and cake.",
    # 5: sad content -> can happy-steering overcome it?
    "Write a short paragraph about a lonely funeral on a cold, grey, rain-soaked afternoon.",
    # 6: bittersweet -> genuinely mixed valence
    "Write a short paragraph about graduation day: saying goodbye to dear friends you may never see again while stepping into an exciting new future.",
]

HTML_TEMPLATE = """<!doctype html><meta charset="utf-8">
<title>Steering sweep - layer __LAYER__</title>
<style>
 body{font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:1.5rem auto;padding:0 1rem;color:#111;background:#fff}
 h1{font-size:1.25rem;margin-bottom:.2rem}
 .controls{margin:1rem 0;padding:.8rem;background:#f7f7f8;border-radius:8px}
 select{font-size:1rem;padding:.3rem .5rem}
 .ptext{color:#444;margin-top:.5rem;font-style:italic}
 .legend span{padding:2px 10px;border-radius:2px}
 .cell{border:1px solid #e5e5e5;border-radius:8px;padding:.6rem .8rem;margin-bottom:1rem}
 .cell .hdr{font-size:.8rem;color:#444;margin-bottom:.4rem;font-weight:600}
 .cell .txt{white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word}
 .cell .txt span{border-radius:2px}
 .baseline{border-color:#bbb;background:#fcfcfc}
 .grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0 1rem}
 .colhdr{font-weight:600;text-align:center;padding:.3rem;color:#555}
 @media(max-width:720px){.grid{grid-template-columns:1fr}}
</style>
<h1>Activation-steering sweep &mdash; layer __LAYER__</h1>
<div>Blue = sad-leaning tokens, red = happy-leaning (per-token activation &middot; unit DiM, minus midpoint).
 <span class="legend"><span style="background:rgb(33,102,172)">sad</span> <span style="background:rgb(255,255,255);border:1px solid #ddd">neutral</span> <span style="background:rgb(214,40,40)">happy</span></span></div>
<div class="controls">
 <label>Prompt: <select id="sel"></select></label>
 <div class="ptext" id="ptext"></div>
</div>
<div id="out"></div>
<h2 style="font-size:1.05rem;margin-top:2.2rem">Are the paragraph projections Gaussian? &mdash; normal Q&ndash;Q plots</h2>
<div style="color:#555;font-size:.85rem;margin-bottom:.6rem">Dataset paragraphs projected onto the DiM axis at layer __LAYER__ (__NPC__ per class).
 Points on the dashed line &rArr; Gaussian; S-shape &rArr; heavy/light tails; overall curve &rArr; skew.</div>
<div class="grid">
 <div>__QQ_SAD__</div>
 <div>__QQ_HAPPY__</div>
</div>
<script>
const DATA = __DATA__;
const SCALE = __SCALE__;
const white=[255,255,255], happy=[214,40,40], sad=[33,102,172];
function colorFor(score){
  let v = Math.max(-1, Math.min(1, score / SCALE));
  const f = Math.abs(v), c = v>=0 ? happy : sad;
  const r=Math.round(white[0]*(1-f)+c[0]*f);
  const g=Math.round(white[1]*(1-f)+c[1]*f);
  const b=Math.round(white[2]*(1-f)+c[2]*f);
  return `rgb(${r},${g},${b})`;
}
function makeCell(label, res, extraClass){
  const cell=document.createElement('div');
  cell.className='cell'+(extraClass?(' '+extraClass):'');
  const hdr=document.createElement('div'); hdr.className='hdr';
  hdr.textContent=label+'   ·   mean lean '+(res.mean>=0?'+':'')+res.mean.toFixed(2);
  const txt=document.createElement('div'); txt.className='txt';
  for(const [tokstr,score] of res.tokens){
    const s=document.createElement('span');
    s.style.background=colorFor(score);
    s.title=(score>=0?'+':'')+score.toFixed(2);
    s.textContent=tokstr;
    txt.appendChild(s);
  }
  cell.appendChild(hdr); cell.appendChild(txt);
  return cell;
}
function render(pi){
  const p=DATA.prompts[pi];
  document.getElementById('ptext').textContent='"'+p.text+'"';
  const out=document.getElementById('out'); out.innerHTML='';
  out.appendChild(makeCell('No steering  (α = 0.0)', p.results['0.0'], 'baseline'));
  const gh=document.createElement('div'); gh.className='grid';
  const h1=document.createElement('div'); h1.className='colhdr'; h1.textContent='← sadder  (−α)';
  const h2=document.createElement('div'); h2.className='colhdr'; h2.textContent='happier  (+α) →';
  gh.appendChild(h1); gh.appendChild(h2); out.appendChild(gh);
  const grid=document.createElement('div'); grid.className='grid';
  for(const m of DATA.mags){
    grid.appendChild(makeCell('α = −'+m.toFixed(1), p.results[(-m).toFixed(1)]));
    grid.appendChild(makeCell('α = +'+m.toFixed(1), p.results[(m).toFixed(1)]));
  }
  out.appendChild(grid);
}
const sel=document.getElementById('sel');
DATA.prompts.forEach((p,i)=>{const o=document.createElement('option');o.value=i;o.textContent='Prompt '+(i+1);sel.appendChild(o);});
sel.addEventListener('change',()=>render(+sel.value));
render(0);
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(common.PRESETS), default="qwen35-4b")
    ap.add_argument("--model", default=None)
    ap.add_argument("--kind", choices=["text", "multimodal"], default=None)
    ap.add_argument("--acts", default="data/activations.pt")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--scale", choices=["norm", "diff", "raw"], default="norm")
    ap.add_argument("--step", type=float, default=0.2)
    ap.add_argument("--max-alpha", type=float, default=1.0)
    ap.add_argument("--mags", default=None,
                    help="explicit comma list of |alpha| magnitudes, e.g. '0.1,0.2,0.4,0.6,0.8,1.0'")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--prompts-file", default=None, help="JSON list of prompt strings")
    ap.add_argument("--out", default="data/steer_grid.html")
    args = ap.parse_args()

    prompts = PROMPTS
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = json.load(f)

    # magnitudes (plus baseline 0)
    if args.mags:
        mags = [round(float(x), 1) for x in args.mags.split(",")]
    else:
        n = int(round(args.max_alpha / args.step))
        mags = [round(args.step * i, 1) for i in range(1, n + 1)]
    alphas = [0.0] + [s * m for m in mags for s in (-1, 1)]

    saved = torch.load(args.acts, weights_only=False)
    L = args.layer if args.layer is not None else int(saved["best_layer"])
    diff = saved["diff_per_layer"][L]
    diff_norm = float(diff.norm())
    feats = saved["features"][:, L].numpy().astype(np.float32)
    labels = saved["labels"].numpy()

    model_id, kind = common.resolve(args.preset, args.model, args.kind)
    print(f"Loading {model_id} ... | layer {L} | scale={args.scale} | "
          f"{len(prompts)} prompts x {len(alphas)} alphas")
    handler, model, tok, is_mm = common.load_model(model_id, kind)
    _, layers = common.get_decoder_layers(model)
    target = layers[L]
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    u_t = (diff / (diff_norm + 1e-9)).to(dev, dt)          # unit DiM (tensor)
    u_np = (diff / (diff_norm + 1e-9)).numpy().astype(np.float32)
    proj = feats @ u_np
    midpoint = 0.5 * (proj[labels == 1].mean() + proj[labels == 0].mean())

    # --- steering-injection hook (adds to all positions) ---
    def make_inject(alpha):
        def hook(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output
            if args.scale == "norm":
                add = alpha * h.norm(dim=-1, keepdim=True) * u_t
            elif args.scale == "diff":
                add = (alpha * diff_norm) * u_t
            else:
                add = alpha * u_t
            h = h + add
            return (h, *output[1:]) if isinstance(output, tuple) else h
        return hook

    # --- read hook for scoring produced text (no steering) ---
    grab = {}

    def read_hook(_m, _i, output):
        grab["h"] = common.layer_output_hidden(output)[0].float().cpu()

    def score_text(text):
        enc = tok(text, return_tensors="pt").to(dev)
        rh = target.register_forward_hook(read_hook)
        try:
            with torch.inference_mode():
                model(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"))
        finally:
            rh.remove()
        scores = (grab["h"].numpy() @ u_np) - midpoint
        ids = enc["input_ids"][0].tolist()
        toks = [tok.decode([t]) for t in ids]
        return list(zip(toks, scores.tolist()))

    data_prompts = []
    all_scores = []
    for pi, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        kw = dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        try:
            enc = handler.apply_chat_template(messages, enable_thinking=False, **kw)
        except TypeError:
            enc = handler.apply_chat_template(messages, **kw)
        enc = {k: v.to(dev) for k, v in enc.items()}
        n_prompt = enc["input_ids"].shape[1]

        results = {}
        for a in alphas:
            handle = target.register_forward_hook(make_inject(a))
            try:
                with torch.inference_mode():
                    out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                         do_sample=False, pad_token_id=tok.eos_token_id)
            finally:
                handle.remove()
            text = tok.decode(out[0, n_prompt:], skip_special_tokens=True).strip()
            toks = score_text(text) if text else []
            scores = [s for _, s in toks]
            all_scores.extend(abs(s) for s in scores)
            results[f"{a:.1f}"] = {
                "tokens": [[t, round(s, 3)] for t, s in toks],
                "mean": round(float(np.mean(scores)) if scores else 0.0, 3),
            }
            print(f"  prompt {pi+1} | alpha {a:+.1f} | {len(scores):3d} tok | "
                  f"mean lean {results[f'{a:.1f}']['mean']:+.2f}")
        data_prompts.append({"text": prompt, "results": results})

    scale = float(np.percentile(all_scores, 98)) if all_scores else 1.0
    data = {"prompts": data_prompts, "mags": mags}

    # Q-Q plots of the dataset paragraph projections onto the DiM axis at layer L
    xi = feats[labels == 1] @ u_np      # happy scalars
    yi = feats[labels == 0] @ u_np      # sad scalars
    npc = int(min((labels == 1).sum(), (labels == 0).sum()))
    qq_happy = qq_svg(xi, f"Happy (xᵢ, n={len(xi)})", "rgb(214,40,40)")
    qq_sad = qq_svg(yi, f"Sad (yᵢ, n={len(yi)})", "rgb(33,102,172)")

    html = (HTML_TEMPLATE
            .replace("__DATA__", json.dumps(data))
            .replace("__SCALE__", f"{scale:.4f}")
            .replace("__NPC__", str(npc))
            .replace("__QQ_SAD__", qq_sad)
            .replace("__QQ_HAPPY__", qq_happy)
            .replace("__LAYER__", str(L)))
    with open(args.out, "w") as f:
        f.write(html)
    print(f"\nWrote {len(prompts)} prompts x {len(alphas)} alphas -> {args.out}  (color scale={scale:.2f})")


if __name__ == "__main__":
    main()
