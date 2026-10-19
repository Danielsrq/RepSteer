"""Reconstruction curve r^2/||x||^2 vs #concepts for one (model, layer), real frame vs
same-size random control. Compute mode saves the averaged curves; render mode overlays
several saved curves (e.g. 4B@L25 vs 2B@L20, the occupancy-peak layers).

Compute:  .venv/bin/python scripts/jlens/11_recon_curve.py --preset qwen35-4b \
            --lens data/jlens/lens_band_4b.pt --layer 25
Render :  .venv/bin/python scripts/jlens/11_recon_curve.py --render \
            data/jlens/curve_qwen35-4b_L25.json data/jlens/curve_qwen35-2b_L20.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

import jlens  # noqa: E402 (before our sibling scripts/jlens/ can shadow it)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import common  # noqa: E402

PROMPTS = [
    "The Roman aqueducts carried fresh water across great stone arches into the imperial capital.",
    "A remote attacker can exploit the heap overflow to overwrite a function pointer and run code.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen using captured light.",
    "The quarterly earnings report showed revenue rising while operating costs steadily declined.",
    "She felt a sudden wave of joy as the train pulled into the station where her family waited.",
    "The detective examined the rain-slicked alley, noting the broken window and a muddy footprint.",
    "During the negotiation both sides agreed to a ceasefire and the release of the prisoners.",
    "The recipe folds whipped egg whites into the batter and bakes until the sponge springs back.",
]


def nnls(A, x, iters=250):
    if A.shape[1] == 0:
        return np.zeros(0)
    AtA, Atx = A.T @ A, A.T @ x
    lr = 1.0 / max(float(np.linalg.eigvalsh(AtA)[-1]), 1e-9)
    c = np.zeros(A.shape[1])
    for _ in range(iters):
        c = np.maximum(c - lr * (AtA @ c - Atx), 0.0)
    return c


def nn_omp_curve(x, D, k):
    x2 = float(x @ x) + 1e-12
    r = x.copy(); S = []; curve = [1.0]
    for _ in range(k):
        corr = D @ r
        if S:
            corr[np.array(S)] = -1e30
        j = int(np.argmax(corr))
        if corr[j] <= 0:
            break
        S.append(j); co = nnls(D[S].T, x); r = x - D[S].T @ co
        curve.append(float(r @ r) / x2)
    while len(curve) < k + 1:
        curve.append(curve[-1])
    return np.array(curve)


def compute(args):
    model_id, kind = common.resolve(args.preset)
    print(f"Loading {model_id} + {args.lens} @ L{args.layer} ...")
    _, model, tok, _ = common.load_model(model_id, kind)
    m = jlens.from_hf(model, tok)
    lens = jlens.JacobianLens.load(args.lens)
    JL = lens.jacobians[args.layer].to(m.input_device).float()
    g = m._final_norm.weight.detach().to(m.input_device).float()
    d, nL = m.d_model, m.n_layers
    _, mods = common.get_decoder_layers(model)
    grab = {}
    h = mods[args.layer].register_forward_hook(
        lambda _m, _i, o: grab.__setitem__("h", common.layer_output_hidden(o)[0].detach().float().cpu()))
    res = []
    try:
        for p in PROMPTS:
            with torch.inference_mode():
                m.forward(m.encode(p))
            H = grab["h"]
            res.extend(H[i] for i in range(args.skip_first, H.shape[0] - 1))
    finally:
        h.remove()
    rng = np.random.default_rng(args.seed)
    R = torch.stack(res)[rng.permutation(len(res))[:args.n_residuals]]
    mu = R.mean(0)
    real, rand = [], []
    for i in range(len(R)):
        r0 = R[i].to(JL.device).float() @ JL.T
        top = m.unembed(r0).float().topk(args.top_m).indices
        w = m._lm_head.weight[top].float()
        V = ((w * g) @ JL).cpu().numpy()
        Vhat = (V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)).astype(np.float64)
        x = (R[i] - mu).numpy().astype(np.float64)
        real.append(nn_omp_curve(x, Vhat, args.max_k))
        Rr = rng.standard_normal((args.top_m, d)); Rr /= np.linalg.norm(Rr, axis=1, keepdims=True) + 1e-12
        rand.append(nn_omp_curve(x, Rr.astype(np.float64), args.max_k))
    real = np.stack(real).mean(0); rand = np.stack(rand).mean(0)
    rg, ag = -np.diff(real), -np.diff(rand)
    occ = int(next((k + 1 for k in range(len(rg)) if rg[k] < ag[k]), args.max_k))
    out = args.out or f"data/jlens/curve_{args.preset}_L{args.layer}.json"
    json.dump({"preset": args.preset, "layer": args.layer, "d_model": d,
               "rel_depth": round(args.layer / nL, 3), "occupancy": occ,
               "real": [round(float(v), 4) for v in real],
               "rand": [round(float(v), 4) for v in rand]}, open(out, "w"))
    print(f"  L{args.layer}: occupancy={occ}  real[occ]={real[occ]:.3f} rand[occ]={rand[occ]:.3f}")
    print(f"Wrote {out}")


def render(files, out):
    curves = [json.load(open(f)) for f in files]
    W, H, pad = 620, 380, 52
    K = max(len(c["real"]) for c in curves) - 1
    colors = ["#2166ac", "#b2182b", "#1a9850", "#762a83"]

    def sx(k): return pad + k / K * (W - 2 * pad)
    def sy(v): return (H - pad) - v * (H - 2 * pad)
    svg = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px">']
    for gy in [0, .2, .4, .6, .8, 1.0]:
        svg.append(f'<line x1="{pad}" y1="{sy(gy):.0f}" x2="{W-pad}" y2="{sy(gy):.0f}" stroke="#eee"/>'
                   f'<text x="{pad-6}" y="{sy(gy)+3:.0f}" font-size="10" fill="#999" text-anchor="end">{gy:.1f}</text>')
    legend = []
    for ci, c in enumerate(curves):
        col = colors[ci % len(colors)]
        real, rand, occ = c["real"], c["rand"], c["occupancy"]
        pr = " ".join(f"{sx(k):.1f},{sy(v):.1f}" for k, v in enumerate(real))
        pd = " ".join(f"{sx(k):.1f},{sy(v):.1f}" for k, v in enumerate(rand))
        svg.append(f'<polyline points="{pr}" fill="none" stroke="{col}" stroke-width="2.5"/>')
        svg.append(f'<polyline points="{pd}" fill="none" stroke="{col}" stroke-width="1.3" '
                   f'stroke-dasharray="4 3" opacity="0.7"/>')
        svg.append(f'<line x1="{sx(occ):.1f}" y1="{sy(real[occ]):.1f}" x2="{sx(occ):.1f}" y2="{H-pad}" '
                   f'stroke="{col}" stroke-width="1" stroke-dasharray="2 2" opacity="0.6"/>')
        svg.append(f'<circle cx="{sx(occ):.1f}" cy="{sy(real[occ]):.1f}" r="4" fill="{col}"/>'
                   f'<text x="{sx(occ):.1f}" y="{H-pad+13:.0f}" font-size="10" fill="{col}" text-anchor="middle">K={occ}</text>')
        legend.append(f'<span style="color:{col}">&#9644; {c["preset"]} L{c["layer"]} '
                      f'(d={c["d_model"]}, rel {c["rel_depth"]}) — occ K={occ}</span>')
    svg.append(f'<text x="{W/2:.0f}" y="{H-8}" font-size="12" fill="#555" text-anchor="middle"># concepts subtracted (k)</text>')
    svg.append(f'<text x="14" y="{H/2:.0f}" font-size="12" fill="#555" text-anchor="middle" '
               f'transform="rotate(-90 14 {H/2:.0f})">residual² / ||x||²</text></svg>')
    doc = f"""<!doctype html><meta charset=utf-8><title>J-space reconstruction curves</title>
<style>body{{font:14px system-ui,sans-serif;max-width:760px;margin:1.5rem auto;padding:0 1rem;background:#fcfcfb;color:#141414}}
h1{{font-size:1.15rem}}.note{{color:#555;font-size:.85rem}}.leg{{margin:.6rem 0;display:flex;gap:1.4rem;flex-wrap:wrap;font-size:.85rem}}</style>
<h1>J-space reconstruction at the occupancy-peak layer</h1>
<div class=note>Residual energy r²/||x||² as NN-OMP subtracts concepts: <b>solid</b> = real J-lens
frame, <b>dashed</b> = same-size random control. Occupancy K (vertical tick) = where the real
marginal gain first falls below random. Lower/steeper solid = more of the residual is verbalizable.</div>
<div class=leg>{''.join(legend)}</div>
{''.join(svg)}"""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(doc)
    print(f"Wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", nargs="+", default=None, help="curve JSONs to overlay")
    ap.add_argument("--preset", default="qwen35-4b")
    ap.add_argument("--lens", default=None)
    ap.add_argument("--layer", type=int, default=25)
    ap.add_argument("--n-residuals", type=int, default=150)
    ap.add_argument("--top-m", type=int, default=256)
    ap.add_argument("--max-k", type=int, default=40)
    ap.add_argument("--skip-first", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.render:
        render(args.render, args.out or "data/jlens/recon_curves.html")
    else:
        compute(args)


if __name__ == "__main__":
    main()
