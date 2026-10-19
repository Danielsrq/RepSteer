"""Can we detect degenerate (repetitive) generation from activations alone?

Labels each steered generation normal/degenerate by its rep-4 score (from 14),
runs a forward pass over each, mean-pools per layer, and trains a nearest-centroid
probe (degenerate-mean vs normal-mean) with leave-one-out accuracy + AUC, per layer.

Usage:  python scripts/15_degen_probe.py --deg-thresh 0.4 --norm-thresh 0.05
"""

import argparse
import json
from collections import Counter

import numpy as np
import torch

import common


def rep4(text):
    w = text.split()
    if len(w) < 4:
        return 0.0
    g = list(zip(w, w[1:], w[2:], w[3:]))
    return 1 - len(set(g)) / len(g)


def auc(scores, y):
    pos, neg = scores[y == 1], scores[y == 0]
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="qwen35-4b")
    ap.add_argument("--html", default="data/concepts/concept_tokens.html")
    ap.add_argument("--deg-thresh", type=float, default=0.4)
    ap.add_argument("--norm-thresh", type=float, default=0.05)
    args = ap.parse_args()

    t = open(args.html).read()
    i = t.index("STEER=") + len("STEER="); j = t.index("], STEER_MAGS=") + 1
    steer = json.loads(t[i:j])
    texts = []
    for p in steer:
        for basis in ("raw", "denoised"):
            for concept, keys in p[basis].items():
                for card in keys.values():
                    texts.append("".join(card["tokens"]))
    scored = [(tx, rep4(tx)) for tx in texts if len(tx.split()) >= 8]
    deg = [tx for tx, r in scored if r > args.deg_thresh]
    norm = sorted([(tx, r) for tx, r in scored if r < args.norm_thresh], key=lambda x: x[1])
    n = min(len(deg), len(norm))
    norm = [tx for tx, _ in norm[:n]]
    deg = deg[:n]
    print(f"degenerate (rep4>{args.deg_thresh}): {len(deg)} | normal (rep4<{args.norm_thresh}): {len(norm)}  (balanced n={n} each)")
    if n < 5:
        print("too few examples; lower --deg-thresh"); return

    corpus = deg + norm
    labels = np.array([1] * len(deg) + [0] * len(norm))

    model_id, kind = common.resolve(args.preset)
    print(f"Loading {model_id} ...")
    handler, model, tok, is_mm = common.load_model(model_id, kind)
    _, layers = common.get_decoder_layers(model)
    nL = len(layers); hidden = model.config.get_text_config().hidden_size

    scratch = {}

    def mk(idx):
        def hook(_m, _i, out):
            scratch[idx] = common.layer_output_hidden(out)[0].float().mean(0).cpu()
        return hook
    handles = [l.register_forward_hook(mk(i)) for i, l in enumerate(layers)]
    feats = torch.empty(len(corpus), nL, hidden)
    try:
        for k, txt in enumerate(corpus):
            enc = tok(txt, return_tensors="pt").to("cuda")
            scratch.clear()
            with torch.inference_mode():
                model(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"))
            for L in range(nL):
                feats[k, L] = scratch[L]
    finally:
        for h in handles:
            h.remove()
    X = feats.numpy().astype(np.float64)

    print(f"\n{'layer':>5} | {'LOO-acc':>7} | {'AUC':>5}")
    print("-" * 24)
    best = (-1, 0.0)
    for L in range(nL):
        F = X[:, L]
        # LOO nearest-centroid along (deg_mean - norm_mean)
        preds = np.empty(len(corpus)); proj = np.empty(len(corpus))
        for m in range(len(corpus)):
            msk = np.ones(len(corpus), bool); msk[m] = False
            Fm, ym = F[msk], labels[msk]
            u = Fm[ym == 1].mean(0) - Fm[ym == 0].mean(0)
            u /= np.linalg.norm(u) + 1e-9
            mid = 0.5 * ((Fm[ym == 1] @ u).mean() + (Fm[ym == 0] @ u).mean())
            s = F[m] @ u
            proj[m] = s - mid; preds[m] = int(s > mid)
        acc = float((preds == labels).mean()); a = auc(proj, labels)
        if a > best[1]:
            best = (L, a)
        if L % 4 == 0 or L in (nL - 1, best[0]):
            print(f"{L:5d} | {acc:7.3f} | {a:5.3f}")
    print("-" * 24)
    print(f"best layer by AUC: {best[0]} (AUC={best[1]:.3f})")
    print("chance = 0.50; high acc => degeneration is linearly decodable from pooled activations")


if __name__ == "__main__":
    main()
