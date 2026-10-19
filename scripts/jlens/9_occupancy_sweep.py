"""Per-layer J-space metric panel for a fitted multi-layer lens.

For each fitted layer, on a FIXED prompt set (identical across layers & models):
  - occupancy K   : NN-OMP reconstruction vs same-size random control -> K where the real
                    marginal gain first falls below random (paper's criterion),
  - excess_var    : (random - real) residual^2/||x||^2 at K = excess variance explained,
  - faithfulness  : fraction of interior positions where the model's own top-1 next token is
                    in the J-lens top-5 at that layer (readout legibility).
All three come from one forward per prompt (hook every requested layer + the final layer).

Output: data/jlens/panel_<preset>.json  (+ a small table HTML).

Usage:  .venv/bin/python scripts/jlens/9_occupancy_sweep.py --preset qwen35-4b \
          --lens data/jlens/lens_band_4b.pt
"""

import argparse
import html
import json
import os
import sys
import time

import numpy as np
import torch

import jlens

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
    r = x.copy(); S = []
    curve = [1.0]
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


def concept_frame(m, JL, g, h, top_m):
    r0 = h.to(JL.device).float() @ JL.T
    logits = m.unembed(r0).float()
    top = logits.topk(top_m).indices
    w = m._lm_head.weight[top].float()
    V = (w * g) @ JL
    return top.detach().cpu().numpy(), V.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="qwen35-4b")
    ap.add_argument("--lens", required=True)
    ap.add_argument("--layers", default=None, help="subset; default = all fitted layers")
    ap.add_argument("--n-residuals", type=int, default=150)
    ap.add_argument("--max-per-prompt", type=int, default=40)
    ap.add_argument("--top-m", type=int, default=256)
    ap.add_argument("--max-k", type=int, default=40)
    ap.add_argument("--skip-first", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompts-from", default=None,
                    help="corpus json; use its cve-labelled chunks as the residual prompts "
                         "instead of the 8 built-in generic ones")
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model_id, kind = common.resolve(args.preset)
    print(f"Loading {model_id} + lens {args.lens} ...")
    _, model, tok, _ = common.load_model(model_id, kind)
    m = jlens.from_hf(model, tok)
    lens = jlens.JacobianLens.load(args.lens)
    layers = ([int(x) for x in args.layers.split(",")] if args.layers else lens.source_layers)
    layers = [L for L in layers if L in lens.source_layers]
    g = m._final_norm.weight.detach().to(m.input_device).float()
    d, nL = m.d_model, m.n_layers
    final = nL - 1
    print(f"d_model={d} n_layers={nL}; layers {layers}")
    prompt_set = PROMPTS
    if args.prompts_from:
        import json as _j
        _d = _j.load(open(args.prompts_from))
        _cve = [c for c, l in zip(_d["chunks"], _d["labels"]) if l == "cve"]
        prompt_set = _cve[:args.n_prompts]
        print(f"residual prompts: {len(prompt_set)} cve chunks from {args.prompts_from}")
    else:
        print(f"residual prompts: {len(prompt_set)} built-in generic")

    # --- one forward per prompt: collect residuals for every layer + faithfulness ---
    _, mods = common.get_decoder_layers(model)
    hooked = sorted(set(layers) | {final})
    grab = {}

    def mk(L):
        def hook(_mod, _i, o):
            grab[L] = common.layer_output_hidden(o)[0].detach().float()
        return hook
    handles = [mods[L].register_forward_hook(mk(L)) for L in hooked]

    res = {L: [] for L in layers}          # residuals per layer
    faith_hit = {L: 0 for L in layers}     # model-top1 in J-lens top5
    faith_tot = 0
    JL = {L: lens.jacobians[L].to(m.input_device).float() for L in layers}
    try:
        for prompt in prompt_set:
            ids = m.encode(prompt)
            with torch.inference_mode():
                m.forward(ids)
            seq = grab[final].shape[0]
            pos = list(range(args.skip_first, seq - 1))
            model_top1 = m.unembed(grab[final][pos]).argmax(-1)     # [npos]
            faith_tot += len(pos)
            for L in layers:
                H = grab[L]
                res[L].extend(H[p].cpu() for p in pos)
                jl_top5 = m.unembed(H[pos] @ JL[L].T).topk(5, dim=-1).indices   # [npos,5]
                faith_hit[L] += int((jl_top5 == model_top1[:, None]).any(-1).sum())
    finally:
        for h in handles:
            h.remove()

    rng = np.random.default_rng(args.seed)
    panel = []
    for L in layers:
        R = torch.stack(res[L])                       # [N,d]
        idx = rng.permutation(len(R))[:args.n_residuals]
        R = R[idx]
        mu = R.mean(0)
        real, rand = [], []
        for i in range(len(R)):
            top, V = concept_frame(m, JL[L], g, R[i], args.top_m)
            Vhat = (V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)).astype(np.float64)
            x = (R[i] - mu).numpy().astype(np.float64)
            real.append(nn_omp_curve(x, Vhat, args.max_k))
            Rr = rng.standard_normal((args.top_m, d)); Rr /= np.linalg.norm(Rr, axis=1, keepdims=True) + 1e-12
            rand.append(nn_omp_curve(x, Rr.astype(np.float64), args.max_k))
        real = np.stack(real).mean(0); rand = np.stack(rand).mean(0)
        rg, ag = -np.diff(real), -np.diff(rand)
        occ = int(next((k + 1 for k in range(len(rg)) if rg[k] < ag[k]), args.max_k))
        excess = float(rand[occ] - real[occ])
        fa = faith_hit[L] / max(faith_tot, 1)
        panel.append({"layer": L, "rel_depth": round(L / nL, 3), "occupancy": occ,
                      "excess_var": round(excess, 4), "faithfulness": round(fa, 3),
                      "resid_at_occ": round(float(real[occ]), 3)})
        print(f"  L{L:2d} (rel {L/nL:.2f}): occ={occ:2d}  excess_var={100*excess:4.1f}%  faith={100*fa:4.1f}%")

    out = args.out or f"data/jlens/panel_{args.preset}.json"
    blob = {"preset": args.preset, "model": model_id, "d_model": d, "n_layers": nL,
            "n_residuals": args.n_residuals, "top_m": args.top_m, "panel": panel}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(blob, open(out, "w"), indent=1)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
