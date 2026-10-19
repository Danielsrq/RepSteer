"""Denoise concept vectors by removing the top-k NEUTRAL PCA directions (style axes).

    v_denoised = v - sum_{j<k} (v . pc_j) pc_j        (pc_j = neutral PCs, unit)

The neutral PCs are ~orthogonal to the concept directions (see 11), so this strips
the shared style/topic component while keeping most of the concept. We sweep k and
save, per k: the denoised 12 concept vectors and their 12x12 cosine matrix
(cosine is scale-invariant, so no renormalization is needed before removal).

Reads data/concepts/concept_vectors.pt; writes data/concepts/denoise.pt.
Usage:  python scripts/12_denoise.py --ks 1,2,5,10
"""

import argparse
import numpy as np
import torch


def cosmat(M):
    Mh = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    return Mh @ Mh.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default="data/concepts/concept_vectors.pt")
    ap.add_argument("--ks", default="1,2,5,10")
    ap.add_argument("--out", default="data/concepts/denoise.pt")
    args = ap.parse_args()

    s = torch.load(args.vectors, weights_only=False)
    L = int(s["pca_neutral"]["layer"])
    concepts = s["concepts"]
    V = s["concept_vectors"][L].numpy().astype(np.float64)          # [12, D] grand-mean-centered
    P = s["pca_neutral"]["components"].numpy().astype(np.float64)   # [K, D] unit, orthonormal
    ks = [int(x) for x in args.ks.split(",")]

    cos_by_k = {0: cosmat(V)}
    vecs_by_k = {0: V.copy()}
    for k in ks:
        Pk = P[:k]                                                  # [k, D]
        Vd = V - (V @ Pk.T) @ Pk                                    # remove top-k neutral projection
        cos_by_k[k] = cosmat(Vd)
        vecs_by_k[k] = Vd

    all_ks = [0] + ks
    torch.save({
        "layer": L, "concepts": concepts, "ks": all_ks, "basis": "neutral_pca",
        "cosine_by_k": {str(k): torch.tensor(cos_by_k[k].astype(np.float32)) for k in all_ks},
        "denoised_by_k": {str(k): torch.tensor(vecs_by_k[k].astype(np.float32)) for k in all_ks},
    }, args.out)

    # --- summary: how key pairs shift as style axes are removed ---
    idx = {c: i for i, c in enumerate(concepts)}
    pairs = [("happy", "sad"), ("angry", "belligerent"), ("happy", "excited"),
             ("calm", "belligerent"), ("humorous", "nonchalant"),
             ("neutral", "calm"), ("neutral", "belligerent")]
    print(f"layer {L} | cosine after removing top-k NEUTRAL PCs")
    print("pair".ljust(24) + "".join(f"k={k:<5}" for k in all_ks))
    for a, b in pairs:
        row = [cos_by_k[k][idx[a], idx[b]] for k in all_ks]
        print(f"{a+'~'+b:24s}" + "".join(f"{v:+5.2f} " for v in row))
    # average |cos| off-diagonal (how much the whole structure tightens/spreads)
    C = len(concepts)
    off = ~np.eye(C, dtype=bool)
    print("\nmean |off-diagonal cosine| (overall inter-concept overlap):")
    print("  " + "".join(f"k={k}: {np.abs(cos_by_k[k][off]).mean():.3f}   " for k in all_ks))
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
