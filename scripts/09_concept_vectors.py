"""Capture activations for the 10-concept dataset, build grand-mean-centered
concept vectors, run PCA on the pooled cloud, and validate separability.

Method (per layer L, on pooled features X = feats[:, L]  [N, hidden]):
  concept_means[L][k] = mean of concept k's paragraphs        [C, hidden]
  grand_mean[L]       = mean of ALL paragraphs                 [hidden]
  concept_vectors[L]  = concept_means[L] - grand_mean[L]       [C, hidden]   (grand-mean DiM)
  concept_hat[L]      = row-normalized concept_vectors[L]      [C, hidden]

PCA (layer `--pca-layer`, default 17) on the grand-mean-centered cloud via SVD:
top-K principal directions of the pooled cloud = candidate nuisance/style axes.
Collected, not used downstream.

Separability is a sanity check that the concept clouds are distinguishable:
nearest-concept classification (in-sample + leave-one-out), a concept-vector cosine
matrix, and a confusion matrix.

Usage:
    python scripts/09_concept_vectors.py
"""

import argparse
import json

import numpy as np
import torch

import common


def nearest_concept(Xc, concept_hat):
    """Predict concept per row = argmax projection of centered activation onto unit concept vec."""
    return (Xc @ concept_hat.T).argmax(1)   # Xc [N,H], concept_hat [C,H] -> [N]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(common.PRESETS), default="qwen35-4b")
    ap.add_argument("--model", default=None)
    ap.add_argument("--kind", choices=["text", "multimodal"], default=None)
    ap.add_argument("--data", default="data/concepts/paragraphs_concepts.json")
    ap.add_argument("--pool", choices=["mean", "last"], default="mean")
    ap.add_argument("--pca-layer", type=int, default=17)
    ap.add_argument("--n-pcs", type=int, default=20)
    ap.add_argument("--out", default="data/concepts/concept_vectors.pt")
    ap.add_argument("--baseline", default="neutral",
                    help="name of the baseline concept excluded from the grand mean")
    args = ap.parse_args()

    blob = json.load(open(args.data))
    rows = blob["rows"]
    concepts = blob["concepts"]
    C = len(concepts)
    labels = np.array([r["concept_idx"] for r in rows])

    model_id, kind = common.resolve(args.preset, args.model, args.kind)
    print(f"Loading {model_id} (kind={kind}) ...")
    handler, model, tok, is_mm = common.load_model(model_id, kind)
    layer_path, layers = common.get_decoder_layers(model)
    n_layers = len(layers)
    hidden = model.config.get_text_config().hidden_size
    print(f"Hooking {n_layers} layers at '{layer_path}' (hidden={hidden}), pool={args.pool}, {len(rows)} paragraphs\n")

    # --- forward hooks: pool each layer's activation per paragraph (same as 03) ---
    scratch = {}

    def make_hook(idx):
        def hook(_m, _i, output):
            h = common.layer_output_hidden(output)[0].float()   # [seq, hidden]
            scratch[idx] = (h.mean(0) if args.pool == "mean" else h[-1]).detach().cpu()
        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    feats = torch.empty(len(rows), n_layers, hidden)
    try:
        for i, r in enumerate(rows):
            enc = tok(r["text"], return_tensors="pt").to("cuda")
            scratch.clear()
            with torch.inference_mode():
                model(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"))
            for L in range(n_layers):
                feats[i, L] = scratch[L]
            if i % 30 == 0:
                print(f"  captured {i:3d}/{len(rows)} | {r['concept']}")
    finally:
        for h in handles:
            h.remove()

    X = feats.numpy().astype(np.float64)                       # [N, L, H]
    baselines = [concepts.index(b) for b in args.baseline.split(",")]
    neutral_idx = baselines[0]                                 # primary baseline (for pca/neutral_mean)
    emo_idx = [k for k in range(C) if k not in baselines]      # non-baseline concepts only
    print(f"baselines {[concepts[i] for i in baselines]}; grand mean over {len(emo_idx)} non-baseline concepts")

    # --- per-layer concept means, EMOTIONAL grand mean, concept vectors (all 12 incl neutral) ---
    concept_means = np.zeros((n_layers, C, hidden), np.float32)
    grand_mean = np.zeros((n_layers, hidden), np.float32)      # mean of the emotional concept means
    concept_vectors = np.zeros((n_layers, C, hidden), np.float32)
    concept_hat = np.zeros((n_layers, C, hidden), np.float32)
    per_layer_acc = []
    for L in range(n_layers):
        XL = X[:, L]
        cms = np.stack([XL[labels == k].mean(0) for k in range(C)])   # [C,H] all concepts
        gm = cms[emo_idx].mean(0)                                     # emotional grand mean (excl neutral)
        cv = cms - gm
        chat = cv / (np.linalg.norm(cv, axis=1, keepdims=True) + 1e-9)
        concept_means[L] = cms; grand_mean[L] = gm
        concept_vectors[L] = cv; concept_hat[L] = chat
        per_layer_acc.append((L, float((nearest_concept(XL - gm, chat) == labels).mean())))  # 12-way
    neutral_mean = concept_means[:, neutral_idx]                      # [L,H]

    PL = args.pca_layer

    def run_svd(Xmat, center):
        Xc = Xmat - center
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(args.n_pcs, Vt.shape[0]); evr = (S**2) / (S**2).sum()
        return {"layer": PL, "components": torch.tensor(Vt[:k].astype(np.float32)),
                "singular_values": torch.tensor(S[:k].astype(np.float32)),
                "explained_variance_ratio": torch.tensor(evr[:k].astype(np.float32)),
                "cumulative": torch.tensor(np.cumsum(evr)[:k].astype(np.float32)),
                "mean": torch.tensor(center.astype(np.float32))}

    # PCA on the NEUTRAL cloud (emotion-free nuisance directions) + emotional superset (for comparison)
    pca_neutral = run_svd(X[labels == neutral_idx, PL], neutral_mean[PL])
    pca_superset = run_svd(X[np.isin(labels, emo_idx), PL], grand_mean[PL])

    # --- separability at the primary layer (12-way, includes neutral) ---
    chatP = concept_hat[PL]
    pred_in = nearest_concept(X[:, PL] - grand_mean[PL], chatP)
    acc_in = float((pred_in == labels).mean())

    F = X[:, PL]
    loo_pred = np.empty(len(rows), int)
    for i in range(len(rows)):
        m = np.ones(len(rows), bool); m[i] = False
        Fm, ym = F[m], labels[m]
        cms = np.stack([Fm[ym == k].mean(0) for k in range(C)])
        gm = cms[emo_idx].mean(0)
        chat = (cms - gm) / (np.linalg.norm(cms - gm, axis=1, keepdims=True) + 1e-9)
        loo_pred[i] = np.argmax(chat @ (F[i] - gm))
    acc_loo = float((loo_pred == labels).mean())

    cos = chatP @ chatP.T                                     # [C,C]
    confusion = np.zeros((C, C), int)
    for t, p in zip(labels, pred_in):
        confusion[t, p] += 1

    # --- report ---
    ab = [c[:4] for c in concepts]
    print(f"\nprimary layer {PL}: nearest-concept accuracy  in-sample={acc_in:.3f}  LOO={acc_loo:.3f}  (chance={1/C:.3f}, {C}-way)")
    print("\nconcept-vector cosine matrix (layer %d):" % PL)
    print("      " + " ".join(f"{a:>5}" for a in ab))
    for i, a in enumerate(ab):
        print(f"{a:>5} " + " ".join(f"{cos[i,j]:5.2f}" for j in range(C)))
    print("\nconfusion (rows=true, cols=pred, in-sample):")
    print("      " + " ".join(f"{a:>5}" for a in ab))
    for i, a in enumerate(ab):
        print(f"{a:>5} " + " ".join(f"{confusion[i,j]:5d}" for j in range(C)))
    print("\nper-layer nearest-concept accuracy:")
    for L, a in per_layer_acc:
        print(f"  layer {L:2d}: {a:.3f}")
    best = max(per_layer_acc, key=lambda x: x[1])
    print(f"best layer by in-sample accuracy: {best[0]} ({best[1]:.3f})")
    evn = pca_neutral["explained_variance_ratio"].numpy(); evs = pca_superset["explained_variance_ratio"].numpy()
    print(f"\nPCA(neutral,  n=30) top-5 EVR = {evn[:5].round(3)}  cum = {float(pca_neutral['cumulative'][-1]):.3f}")
    print(f"PCA(superset, emo ) top-5 EVR = {evs[:5].round(3)}  cum = {float(pca_superset['cumulative'][-1]):.3f}")

    torch.save({
        "model": model_id, "layer_path": layer_path, "pool": args.pool,
        "concepts": concepts, "primary_layer": PL,
        "neutral_idx": neutral_idx, "emotional_idx": emo_idx,
        "features": feats,                                    # [N,L,H]
        "labels": torch.tensor(labels),                      # [N]
        "concept_means": torch.tensor(concept_means),        # [L,C,H]
        "grand_mean": torch.tensor(grand_mean),              # [L,H] (emotional)
        "neutral_mean": torch.tensor(neutral_mean),          # [L,H]
        "concept_vectors": torch.tensor(concept_vectors),    # [L,C,H] (all - emotional grand mean)
        "concept_hat": torch.tensor(concept_hat),            # [L,C,H]
        "pca_neutral": pca_neutral, "pca_superset": pca_superset,
        "separability": {
            "nearest_centroid_acc": acc_in, "loo_acc": acc_loo,
            "cosine_matrix": torch.tensor(cos.astype(np.float32)),
            "confusion": torch.tensor(confusion),
        },
    }, args.out)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
