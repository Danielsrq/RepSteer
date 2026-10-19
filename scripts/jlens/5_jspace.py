"""J-space at layer 17: decompose a residual into a sparse non-negative sum of
J-lens concept directions, with occupancy stopping vs a matched random control.

Concept directions (fixed transported-unembedding frame):
    v_t = (g (.) W_U[t]) @ J_17   in R^d      (g = final RMSNorm weight)
so that <h, v_t> ~ logit_t (the lens score for token t). NOTE we must treat the final
RMSNorm as a FIXED linear map here: its true Jacobian is scale-invariant, so the autograd
gradient d logit_t/d h is orthogonal to h and useless for reconstruction. Using the fixed
diagonal linearization g/rms drops out under unit-normalisation, leaving v_t as above.
The lens's top-M tokens at h give the active cone {v_t}. We decompose the centered
residual against that cone by NON-NEGATIVE ORTHOGONAL MATCHING PURSUIT (the paper's
"gradient pursuit" / Pi_S reconstruction, arXiv 2607.15495 §2.3, A.8): at each step add
the atom with the largest positive correlation to the residual, then RE-FIT all selected
coefficients jointly by NNLS (= the orthogonal projection Pi_S x onto span of the selected
atoms, c>=0). This is the c = V^+ h pseudoinverse fit, NOT one-direction-at-a-time
subtraction. Occupancy = the K at which the real reconstruction's marginal gain falls
below a same-size random control set (paper: ~25 in the workspace band; note the paper's
absolute excess-variance-explained over random is <10%). Euclidean metric, mean-centered,
NOT whitened (whitening only applies to the paper's template/oracle multi-token lenses).

Usage:  .venv/bin/python scripts/jlens/5_jspace.py --n-residuals 150 --top-m 512
"""

import argparse
import html
import os
import sys
import time

import numpy as np
import torch

import jlens

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import common  # noqa: E402

PROMPTS = [
    "The Roman aqueducts carried fresh water across great stone arches into the heart of "
    "the imperial capital, supplying its fountains, public baths, and private gardens for "
    "centuries until the aqueduct network fell into disrepair during the later empire.",
    "A remote attacker can exploit the heap buffer overflow to overwrite a function pointer "
    "and execute arbitrary code with kernel privileges on the target host, then escalate to "
    "root, disable auditing, and install a persistent backdoor on the compromised machine.",
    "She felt a sudden wave of joy and gratitude wash over her as the train finally pulled "
    "into the station where her family stood waiting on the crowded platform, waving and "
    "smiling beneath the pale winter sun that broke through the grey clouds overhead.",
    "The quarterly earnings report shows revenue grew twelve percent while operating costs "
    "declined, driven by stronger demand in the enterprise software segment and disciplined "
    "spending, and management raised full-year guidance for both margins and free cash flow.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen using light "
    "energy captured by chlorophyll in the chloroplasts of plant cells, forming the base of "
    "nearly every food chain and regulating the composition of the atmosphere over time.",
    "The detective examined the rain-slicked alley for evidence, noting the broken window, "
    "the faint smell of smoke, and a single muddy footprint leading toward the abandoned "
    "warehouse where the stolen paintings were rumoured to have been hidden for months.",
    "During the negotiation both parties agreed to a phased ceasefire, the withdrawal of "
    "heavy weapons from the contested border region, and the release of prisoners, though "
    "deep mistrust remained and observers doubted the fragile agreement would hold for long.",
    "The recipe calls for folding whipped egg whites gently into the batter, baking at a "
    "moderate temperature until the sponge springs back to the touch, and cooling the cake "
    "completely before layering it with fresh cream and ripe summer berries.",
]


def get_residuals(m, model, layer, max_per_prompt, skip_first=16):
    _, layers = common.get_decoder_layers(model)
    grab = {}

    def hook(_mod, _in, out):
        grab["h"] = common.layer_output_hidden(out)[0].detach().float().cpu()
    handle = layers[layer].register_forward_hook(hook)
    res = []
    try:
        for prompt in PROMPTS:
            ids = m.encode(prompt)
            with torch.inference_mode():
                m.forward(ids)
            H = grab["h"]  # [seq, d]
            seq = H.shape[0]
            positions = list(range(skip_first, seq - 1))
            if len(positions) > max_per_prompt:
                idx = np.linspace(0, len(positions) - 1, max_per_prompt).astype(int)
                positions = [positions[i] for i in idx]
            for p in positions:
                res.append((prompt, p, H[p].clone()))
    finally:
        handle.remove()
    return res


@torch.no_grad()
def concept_frame(m, J17, g, h, top_m):
    """Fixed transported-unembedding frame for the top-M lens tokens at h.
    v_t = (g (.) W_U[t]) @ J_17  (so <h,v_t> ~ logit_t). Returns
    (indices[M], Vhat[M,d] unit, logits_topm[M])."""
    r0 = h.to(J17.device).float() @ J17.T            # transported residual [d]
    logits = m.unembed(r0).float()                   # lens logits [vocab]
    top = logits.topk(top_m).indices
    w = m._lm_head.weight[top].float()               # [M, d]
    V = (w * g) @ J17                                 # [M, d] transported-unembed directions (raw)
    return top.cpu(), V, logits[top].detach().cpu()


def nnls_pg(AtA, Atx, iters=250):
    """Non-negative least squares min_{c>=0} ||Ac-x||^2 via projected gradient on the
    normal equations (AtA=A^T A, Atx=A^T x). A is small (|S| x |S|) so this is cheap."""
    c = torch.zeros(AtA.shape[0], device=AtA.device, dtype=AtA.dtype)
    L = torch.linalg.eigvalsh(AtA)[-1].clamp_min(1e-6)   # spectral norm of AtA
    lr = 1.0 / L
    for _ in range(iters):
        c = torch.clamp(c - lr * (AtA @ c - Atx), min=0.0)
    return c


def nn_omp(x, D, k):
    """Non-negative orthogonal matching pursuit (the paper's gradient-pursuit /
    Pi_S reconstruction). At each step: add the dictionary atom with the largest
    positive correlation to the residual, then re-fit ALL selected coefficients by
    NNLS (the orthogonal projection Pi_S x onto span of selected atoms, constrained
    c>=0). Returns (selected_dict_indices, final_coeffs, resid_sq[k+1] / ||x||^2).
    D: [M,d] UNIT rows; x: [d]."""
    x2 = float(x @ x) + 1e-12
    residual = x.clone()
    S, coeffs = [], []
    curve = [1.0]
    for _ in range(k):
        corr = D @ residual                          # <residual, v_hat_i> for all atoms
        if S:
            corr[torch.tensor(S, device=D.device)] = -1e30
        j = int(torch.argmax(corr))
        if float(corr[j]) <= 0:                      # non-negativity: nothing left to add
            break
        S.append(j)
        A = D[S].T                                   # [d, |S|]
        c = nnls_pg(A.T @ A, A.T @ x)                # joint NNLS refit over selected set
        coeffs = c.tolist()
        residual = x - A @ c                         # x - Pi_S x
        curve.append(float(residual @ residual) / x2)
    while len(curve) < k + 1:
        curve.append(curve[-1])
    return S, coeffs, np.array(curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="qwen35-4b")
    ap.add_argument("--lens", default="data/jlens/lens_L17.pt")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--top-m", type=int, default=512, help="dictionary size (active cone per token)")
    ap.add_argument("--max-k", type=int, default=40, help="max concepts to pursue (occupancy sweep upper bound)")
    ap.add_argument("--n-residuals", type=int, default=150)
    ap.add_argument("--max-per-prompt", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/jlens/jspace.html")
    args = ap.parse_args()

    model_id, kind = common.resolve(args.preset)
    print(f"Loading {model_id} + lens {args.lens} ...")
    _, model, tok, _ = common.load_model(model_id, kind)
    m = jlens.from_hf(model, tok)
    lens = jlens.JacobianLens.load(args.lens)
    J17 = lens.jacobians[args.layer].to(m.input_device).float()
    g = m._final_norm.weight.detach().to(m.input_device).float()   # final RMSNorm gain
    print(f"{lens!r}")

    t_res0 = time.perf_counter()
    residuals = get_residuals(m, model, args.layer, args.max_per_prompt)
    t_res = time.perf_counter() - t_res0
    rng = np.random.default_rng(args.seed)
    rng.shuffle(residuals)
    residuals = residuals[:args.n_residuals]
    # Mean-center only (NOT whiten): the concept signal lives in the high-variance
    # directions, so diagonal whitening suppresses it. We remove the common-mode mu and
    # decompose the centered residual h-mu, measuring explained centered variance.
    stacked = torch.stack([h for _, _, h in residuals])   # [N, d]
    mu = stacked.mean(0)
    print(f"Collected {len(residuals)} residuals at layer {args.layer}; "
          f"||mu||={mu.norm():.1f}  mean||h||={stacked.norm(dim=1).mean():.1f}")

    real_curves, rand_curves = [], []
    examples = []
    d = m.d_model
    mu_dev = mu.to(m.input_device)
    t_dec0 = time.perf_counter()
    for k, (prompt, pos, h) in enumerate(residuals):
        idx, V, ltop = concept_frame(m, J17, g, h, args.top_m)
        Vhat = V / (V.norm(dim=1, keepdim=True) + 1e-12)      # unit J-lens directions (cone)
        h_dev = (h.to(m.input_device) - mu_dev).float()       # centered residual x
        S, coeffs, real_curve = nn_omp(h_dev, Vhat, args.max_k)
        # same-size random control dictionary
        R = torch.tensor(rng.standard_normal((args.top_m, d)), dtype=torch.float32, device=Vhat.device)
        R = R / (R.norm(dim=1, keepdim=True) + 1e-12)
        _, _, rand_curve = nn_omp(h_dev, R, args.max_k)
        real_curves.append(real_curve)
        rand_curves.append(rand_curve)
        # keep a few example decompositions (one per distinct prompt)
        if len(examples) < len(PROMPTS) and all(e[0] != prompt for e in examples):
            toks = [(tok.decode([int(idx[S[i]])]), round(coeffs[i], 3)) for i in range(min(8, len(S)))]
            examples.append((prompt, pos, tok.decode([int(m.encode(prompt)[0][pos])]), toks))

    t_dec = time.perf_counter() - t_dec0
    real = np.stack(real_curves).mean(0)   # [max_k+1]
    rand = np.stack(rand_curves).mean(0)
    real_gain = -np.diff(real)             # marginal reconstruction gain per concept
    rand_gain = -np.diff(rand)
    # occupancy = first k (1-indexed) where real marginal gain drops below random's
    occ = next((k + 1 for k in range(len(real_gain)) if real_gain[k] < rand_gain[k]), args.max_k)
    print(f"\nOccupancy (real marginal gain < random): ~{occ} concepts")
    print(f"real reconstruction resid^2 @k={occ}: {real[occ]:.3f}  | random: {rand[occ]:.3f}  "
          f"(excess variance explained {100*(rand[occ]-real[occ]):.1f}%)")
    print(f"compute: residual-collection {t_res:.1f}s ({len(PROMPTS)} forward passes) | "
          f"decomposition {t_dec:.1f}s ({len(residuals)} residuals, NN-OMP)  -- NO backward passes")
    print("\n--- example decompositions (top-8 concept · coeff) ---")
    for prompt, pos, curtok, toks in examples:
        print(f"[{prompt}]  tok={curtok!r}")
        print("   " + "  ".join(f"{t!r}·{c}" for t, c in toks))

    # --- render ---
    W, Hh, pad = 560, 300, 44
    xs = np.arange(len(real))

    def sx(i): return pad + i / (len(real) - 1) * (W - 2 * pad)
    def sy(v): return (Hh - pad) - v * (Hh - 2 * pad)
    def poly(curve, color):
        pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(curve))
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>'
    svg = [f'<svg viewBox="0 0 {W} {Hh}" width="100%" style="max-width:{W}px">']
    svg.append(f'<line x1="{pad}" y1="{sy(0):.0f}" x2="{W-pad}" y2="{sy(0):.0f}" stroke="#ddd"/>')
    svg.append(f'<line x1="{sx(occ):.0f}" y1="{pad}" x2="{sx(occ):.0f}" y2="{Hh-pad}" stroke="#e0a">'
               f'</line><text x="{sx(occ):.0f}" y="{pad-6}" font-size="11" fill="#c07" text-anchor="middle">occ≈{occ}</text>')
    svg.append(poly(real, "#2166ac"))
    svg.append(poly(rand, "#b0b0b0"))
    svg.append(f'<text x="{W/2:.0f}" y="{Hh-8}" font-size="12" fill="#555" text-anchor="middle"># concepts</text>')
    svg.append(f'<text x="14" y="{Hh/2:.0f}" font-size="12" fill="#555" text-anchor="middle" '
               f'transform="rotate(-90 14 {Hh/2:.0f})">residual² / ||h||²</text>')
    svg.append("</svg>")

    ex_html = []
    for prompt, pos, curtok, toks in examples:
        chips = " ".join(f'<span style="background:#eef;padding:1px 6px;margin:1px;border-radius:3px">'
                         f'{html.escape(repr(t))}<b style="color:#26a">·{c}</b></span>' for t, c in toks)
        ex_html.append(f"<h3>{html.escape(prompt)} <span style='color:#888'>@pos {pos} "
                       f"(token {html.escape(repr(curtok))})</span></h3><p>{chips}</p>")

    doc = f"""<!doctype html><meta charset=utf-8><title>J-space @ layer {args.layer}</title>
<style>body{{font:14px system-ui,sans-serif;max-width:900px;margin:1.5rem auto;padding:0 1rem;color:#111;background:#fcfcfb}}
h1{{font-size:1.2rem}}h2{{font-size:1rem;margin:1.3rem 0 .3rem}}h3{{font-size:.85rem;margin:1rem 0 .2rem;font-weight:600}}
.lg{{display:inline-block;width:26px;height:3px;vertical-align:middle}}</style>
<h1>J-space decomposition — layer {args.layer}</h1>
<p class=note>Non-negative orthogonal matching pursuit (the paper's gradient-pursuit / Π_S
reconstruction, arXiv 2607.15495) of {len(residuals)} centered residuals against each token's
top-{args.top_m} J-lens concept cone (v_t = (g⊙W_U[t])·J_{args.layer}, so ⟨h,v_t⟩≈lens logit).
Each step adds the max-correlation atom then re-fits all coefficients by NNLS (= the pseudoinverse
projection Π_S x, c≥0). <span class=lg style="background:#2166ac"></span> real cone vs
<span class=lg style="background:#b0b0b0"></span> same-size random control. Occupancy K (real
marginal reconstruction gain first falls below random) ≈ <b>{occ}</b>; residual²/||x||² there =
{real[occ]:.3f} (real) vs {rand[occ]:.3f} (random) → <b>{100*(rand[occ]-real[occ]):.1f}% excess
variance explained</b> over random (the paper's metric: it reports &lt;10% excess, occupancy ~25 in
the workspace band). Euclidean metric, mean-centered, not whitened. Each residual resolves into a
sparse, non-negative, human-legible (often multilingual) concept set — the decompositions below.</p>
{''.join(svg)}
<h2>Example decompositions (top-8 concepts · coefficient)</h2>
{''.join(ex_html)}"""
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write(doc)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
