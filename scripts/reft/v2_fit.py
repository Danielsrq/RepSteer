"""Fit the DiM and LDA readers for all 5 v2 labels off the streaming cache, and score them.

DiM  = grand-mean-centred difference of means: for each class, (class_mean - grand_mean), where
       grand_mean is the mean of the 5 cloud means (the user's definition).
LDA  = one-vs-rest Fisher discriminant on the full 2560-d residual.
Both are read on the SAME streaming residuals the viewer uses, so they are directly comparable to
each other and (later) to the LoReFT read channels.

Directions are fit on TRAIN articles only. Reported:
  LOAO   leave-one-article-out within the train split, direction REFIT inside each fold (no leak)
  HOLD   fit on all train articles, scored on smbghost/win32k_1732/nftables/regresshion
Saves the directions to data/security/v2_readers_L{L}.npz for the viewer.

  .venv/bin/python scripts/reft/v2_fit.py
"""
import argparse, os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from span_labels_v2 import LABELS, CONCEPTS

CACHE = "data/security/v2_resid_L{L}.npz"


def auc(p, n):
    p, n = np.asarray(p), np.asarray(n)
    if len(p) < 2 or len(n) < 2:
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([p, n]))) + 1
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def lda_dir(Xp, Xn, shrink=1e-1):
    mp, mn = Xp.mean(0), Xn.mean(0)
    Xc = np.concatenate([Xp - mp, Xn - mn]); S = Xc.T @ Xc / len(Xc)
    S = S + shrink * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
    return np.linalg.solve(S, mp - mn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=17)
    args = ap.parse_args()
    z = np.load(CACHE.format(L=args.layer), allow_pickle=True)
    H, lab, art, spl = z["H"].astype(np.float32), z["label"], z["article"], z["split"]
    tr = spl == "train"

    def gmean(mask_articles=None):
        keep = tr if mask_articles is None else (tr & np.isin(art, mask_articles))
        return np.mean([H[(lab == c) & keep].mean(0) for c in LABELS], 0)

    def dim_dir(c, keep):
        d = H[(lab == c) & keep].mean(0) - np.mean([H[(lab == k) & keep].mean(0) for k in LABELS], 0)
        return d / (np.linalg.norm(d) + 1e-12)

    train_arts = sorted(set(art[tr]))
    print(f"cache: {len(H)} tokens, {len(train_arts)} train articles, "
          f"clouds " + ", ".join(f"{c}={int((lab==c).sum())}" for c in LABELS))
    print(f"\n{'label':<12}{'LOAO DiM':>10}{'LOAO LDA':>10}{'HOLD DiM':>10}{'HOLD LDA':>10}  (one-vs-rest AUC)")
    readers = {}
    for c in LABELS:
        # LOAO: refit direction leaving one train article out each fold
        lo_dim, lo_lda = [], []
        for a in train_arts:
            fit = tr & (art != a); ev = tr & (art == a)
            pe, ne = (lab == c) & ev, (lab != c) & ev
            if pe.sum() < 3 or ne.sum() < 3 or ((lab == c) & fit).sum() < 3:
                continue
            dd = dim_dir(c, fit)
            lo_dim.append(auc(H[pe] @ dd, H[ne] @ dd))
            w = lda_dir(H[(lab == c) & fit], H[(lab != c) & fit])
            lo_lda.append(auc(H[pe] @ w, H[ne] @ w))
        # HOLD: fit on all train, score on holdout
        dd = dim_dir(c, tr); w = lda_dir(H[(lab == c) & tr], H[(lab != c) & tr])
        ho_p, ho_n = (lab == c) & ~tr, (lab != c) & ~tr
        ho_dim = auc(H[ho_p] @ dd, H[ho_n] @ dd) if ho_p.sum() else float("nan")
        ho_lda = auc(H[ho_p] @ w, H[ho_n] @ w) if ho_p.sum() else float("nan")
        g = lambda v: f"{v:.3f}" if v == v else "  -  "
        print(f"{c:<12}{g(np.nanmean(lo_dim)):>10}{g(np.nanmean(lo_lda)):>10}{g(ho_dim):>10}{g(ho_lda):>10}")
        readers[f"dim_{c}"] = dd.astype(np.float32)
        readers[f"lda_{c}"] = w.astype(np.float32)
    readers["grand_mean"] = gmean().astype(np.float32)
    out = f"data/security/v2_readers_L{args.layer}.npz"
    np.savez_compressed(out, **readers)
    print(f"\nWrote {out} (DiM + LDA directions for all 5 labels)")
    print("LOAO = leave-one-article-out (refit per fold); HOLD = scored on the 4 held-out articles.")


if __name__ == "__main__":
    main()
