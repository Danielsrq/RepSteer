"""Test: are the dominant PCA directions orthogonal to the DiM concept vectors?

Hypothesis: style/language/topic dominate the residual stream, so the top principal
components (of the neutral cloud and of the emotional superset) should be roughly
ORTHOGONAL to the emotion/concept directions, i.e. emotion is a small slice of the
total activation variance.

Reads data/concepts/concept_vectors.pt (no model needed). Reports:
  1. Variance decomposition: fraction of superset activation variance explained by
     concept identity (eta^2) vs within-concept (style/topic).
  2. |cos(PC_i, concept_k)| for the top PCs (neutral PCA and superset PCA).
  3. Subspace overlap: fraction of each concept direction lying in the top-K PC
     subspace, vs the random-chance baseline.

Usage:  python scripts/11_pca_concept_overlap.py
"""

import argparse
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default="data/concepts/concept_vectors.pt")
    args = ap.parse_args()

    s = torch.load(args.vectors, weights_only=False)
    L = int(s["pca_neutral"]["layer"])
    concepts = s["concepts"]
    emo_idx = s["emotional_idx"]
    emo_names = [concepts[k] for k in emo_idx]
    ab = [c[:4] for c in emo_names]
    D = s["concept_hat"].shape[-1]

    emo_hat = s["concept_hat"][L].numpy()[emo_idx].astype(np.float64)   # [11, D] unit
    emo_hat /= np.linalg.norm(emo_hat, axis=1, keepdims=True) + 1e-12   # ensure unit

    pcas = {"neutral": s["pca_neutral"], "superset": s["pca_superset"]}

    # ---- 1. variance decomposition on the emotional superset at layer L ----
    X = s["features"][:, L].numpy().astype(np.float64)
    labels = s["labels"].numpy()
    grand = X[np.isin(labels, emo_idx)].mean(0)
    total_ss, between_ss = 0.0, 0.0
    for k in emo_idx:
        Xk = X[labels == k]
        total_ss += ((Xk - grand) ** 2).sum()
        between_ss += len(Xk) * ((Xk.mean(0) - grand) ** 2).sum()
    eta2 = between_ss / total_ss
    print(f"=== Variance decomposition (emotional superset, layer {L}) ===")
    print(f"  between-concept (emotion)  : {eta2*100:5.1f}%   <- variance explained by concept identity")
    print(f"  within-concept (style/topic): {(1-eta2)*100:5.1f}%")
    print(f"  => emotion is {'a SMALL' if eta2 < 0.3 else 'a LARGE'} part of the activation variance\n")

    rnd_cos = np.sqrt(2 / (np.pi * D))          # E|cos| of two random unit vecs
    print(f"(reference: random |cos| in {D}-dim ~ {rnd_cos:.3f};  random top-K overlap ~ K/{D})\n")

    # ---- 2. |cos(PC, concept)| tables ----
    for name, pca in pcas.items():
        P = pca["components"].numpy().astype(np.float64)          # [K, D] orthonormal
        C = np.abs(emo_hat @ P.T)                                 # [11, K]
        evr = pca["explained_variance_ratio"].numpy()
        npc = min(8, P.shape[0])
        print(f"=== |cos(PC, concept)| — {name} PCA (top {npc} of {P.shape[0]} PCs) ===")
        print("        " + " ".join(f"{a:>5}" for a in ab) + "   | evr%   max  mean")
        for i in range(npc):
            row = C[:, i]
            print(f"  PC{i+1:<2d} " + " ".join(f"{row[j]:5.2f}" for j in range(len(ab)))
                  + f"   |{evr[i]*100:4.1f}  {row.max():4.2f} {row.mean():4.2f}")
        print()

    # ---- 3. subspace overlap: fraction of each concept in top-K PC subspace ----
    Ks = [1, 5, 10, 20]
    print("=== Subspace overlap: fraction of concept direction inside top-K PC subspace ===")
    hdr = "  concept   " + "  ".join(f"neu@{k}" for k in Ks) + "   " + "  ".join(f"sup@{k}" for k in Ks)
    print(hdr)
    ov = {n: np.abs(emo_hat @ pca["components"].numpy().astype(np.float64).T) ** 2
          for n, pca in pcas.items()}   # [11, K] squared cosines
    for j, a in enumerate(emo_names):
        neu = [ov["neutral"][j, :k].sum() for k in Ks]
        sup = [ov["superset"][j, :k].sum() for k in Ks]
        print(f"  {a:11s}" + "  ".join(f"{v:5.2f}" for v in neu) + "   " + "  ".join(f"{v:5.2f}" for v in sup))
    neu_avg = [np.mean([ov["neutral"][j, :k].sum() for j in range(len(emo_names))]) for k in Ks]
    sup_avg = [np.mean([ov["superset"][j, :k].sum() for j in range(len(emo_names))]) for k in Ks]
    rnd = [k / D for k in Ks]
    print(f"  {'AVERAGE':11s}" + "  ".join(f"{v:5.2f}" for v in neu_avg) + "   " + "  ".join(f"{v:5.2f}" for v in sup_avg))
    print(f"  {'random':11s}" + "  ".join(f"{v:5.3f}" for v in rnd) + "   " + "  ".join(f"{v:5.3f}" for v in rnd))
    print("\n(1.00 = concept lies entirely in that subspace; ~random = orthogonal to it.)")


if __name__ == "__main__":
    main()
