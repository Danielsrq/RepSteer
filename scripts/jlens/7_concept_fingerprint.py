"""J-space vocab fingerprint of DiM concept vectors (paper Fig 8).

Input = precomputed layer-17 DiM concept vectors (concept_vectors[L] from an 09 run).
NO forward passes: each concept vector v_c plays the role of a residual. We lens-read it
(W_U.norm(J.v_c) -> top-M tokens = ranked vocab fingerprint), build the cone
v_t = (g (.) W_U[t]).J, and decompose v_c by non-negative OMP -> the sparse concept set +
coefficients + the fraction of v_c reconstructed (how much of the researcher-built direction
lives in the model's verbalizable J-space).

Usage:  .venv/bin/python scripts/jlens/7_concept_fingerprint.py \
          --vectors data/security/security_vectors.pt --out data/jlens/fingerprint_security.html
"""

import argparse
import html
import os
import sys

import numpy as np
import torch

import jlens

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import common  # noqa: E402


def concept_frame(m, J17, g, h, top_m):
    r0 = h.to(J17.device).float() @ J17.T
    logits = m.unembed(r0).float()
    top = logits.topk(top_m).indices
    w = m._lm_head.weight[top].float()
    V = (w * g) @ J17
    return top.detach().cpu().numpy(), V.detach().cpu().numpy()


def nnls(A, x, iters=300):
    if A.shape[1] == 0:
        return np.zeros(0)
    AtA, Atx = A.T @ A, A.T @ x
    lr = 1.0 / max(float(np.linalg.eigvalsh(AtA)[-1]), 1e-9)
    c = np.zeros(A.shape[1])
    for _ in range(iters):
        c = np.maximum(c - lr * (AtA @ c - Atx), 0.0)
    return c


def nn_omp(x, D, k):
    x2 = float(x @ x) + 1e-12
    residual = x.copy()
    S, coeffs = [], np.zeros(0)
    for _ in range(k):
        corr = D @ residual
        if S:
            corr[np.array(S)] = -1e30
        j = int(np.argmax(corr))
        if corr[j] <= 0:
            break
        S.append(j)
        coeffs = nnls(D[S].T, x)
        residual = x - D[S].T @ coeffs
    return S, coeffs, 1.0 - float(residual @ residual) / x2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="qwen35-4b")
    ap.add_argument("--lens", default="data/jlens/lens_L17.pt")
    ap.add_argument("--vectors", default="data/security/security_vectors.pt")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--top-m", type=int, default=512)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--show", type=int, default=10)
    ap.add_argument("--out", default="data/jlens/fingerprint_security.html")
    args = ap.parse_args()

    model_id, kind = common.resolve(args.preset)
    print(f"Loading {model_id} + lens {args.lens} + vectors {args.vectors} ...")
    _, model, tok, _ = common.load_model(model_id, kind)
    m = jlens.from_hf(model, tok)
    lens = jlens.JacobianLens.load(args.lens)
    J17 = lens.jacobians[args.layer].to(m.input_device).float()
    g = m._final_norm.weight.detach().to(m.input_device).float()

    blob = torch.load(args.vectors, weights_only=False)
    concepts = blob["concepts"]
    V_dim = blob["concept_vectors"][args.layer]        # [C, d]

    cards = []
    print(f"\nDiM concept -> J-space fingerprint (layer {args.layer})\n" + "=" * 66)
    for ci, name in enumerate(concepts):
        v = V_dim[ci].numpy().astype(np.float64)
        top, Vraw = concept_frame(m, J17, g, torch.tensor(v), args.top_m)
        Vhat = (Vraw / (np.linalg.norm(Vraw, axis=1, keepdims=True) + 1e-12)).astype(np.float64)
        raw_top = [tok.decode([int(top[i])]) for i in range(10)]     # ranked lens readout
        S, coeffs, recon = nn_omp(v, Vhat, args.k)
        fp = sorted([(tok.decode([int(top[S[i]])]), float(coeffs[i]))
                     for i in range(len(S)) if coeffs[i] > 1e-3], key=lambda z: -z[1])[:args.show]
        print(f"\n[{name}]  J-space reconstructs {100*recon:.1f}% of the direction")
        print("  lens top-10 :  " + "  ".join(repr(t) for t in raw_top))
        print("  J-space     :  " + "  ".join(f"{w!r}:{c:.2f}" for w, c in fp))
        cards.append((name, recon, raw_top, fp))

    # --- HTML ---
    def chips(fp):
        if not fp:
            return "<span style='color:#aaa'>—</span>"
        cmax = max(c for _, c in fp)
        return "".join(
            f"<div class=row><span class=w>{html.escape(repr(w))}</span>"
            f"<span class=barwrap><span class=bar style='width:{100*c/cmax:.0f}%'></span></span>"
            f"<span class=co>{c:.2f}</span></div>" for w, c in fp)
    body = "".join(
        f"<div class=card><h3>{html.escape(name)} <small>· J-space captures {100*r:.1f}%</small></h3>"
        f"<div class=lens>lens top-10: {html.escape('  '.join(repr(t) for t in raw))}</div>"
        f"{chips(fp)}</div>" for name, r, raw, fp in cards)
    doc = f"""<!doctype html><meta charset=utf-8><title>DiM concept fingerprints</title>
<style>body{{font:14px system-ui,sans-serif;max-width:1040px;margin:1.4rem auto;padding:0 1rem;background:#fbfbf9;color:#141414}}
h1{{font-size:1.2rem}} .note{{color:#555;font-size:.85rem;margin-bottom:1rem}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.card{{border:1px solid #e2e1da;border-radius:8px;padding:10px 12px;background:#fff}}
.card h3{{font-size:.9rem;margin:0 0 4px}} .card h3 small{{font-weight:400;color:#888}}
.lens{{font-family:ui-monospace,monospace;font-size:11px;color:#777;margin-bottom:7px;word-break:break-word}}
.row{{display:flex;align-items:center;gap:6px;margin:2px 0}}
.row .w{{flex:0 0 96px;font-family:ui-monospace,monospace;font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.row .barwrap{{flex:1;background:#f0efe9;border-radius:3px;height:10px;overflow:hidden}}
.row .bar{{display:block;height:100%;background:#2166ac}}
.row .co{{flex:0 0 30px;text-align:right;color:#555;font-size:11px}}</style>
<h1>DiM concept vectors → J-space fingerprint (layer {args.layer})</h1>
<div class=note>Each researcher-built DiM direction is lens-read and decomposed by non-negative OMP into the
model's own vocab concepts (no forward pass — the vector is the target). "captures X%" = fraction of the
DiM direction reconstructed by its top-{args.k} J-space concepts.</div>
<div class=grid>{body}</div>"""
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write(doc)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
