"""Head-to-head reader comparison on the v2 streaming cache: DiM vs LDA vs LoReFT read/write.

All four readers are SCORED on the same streaming residuals (data/security/v2_resid_L17.npz), so
the only thing that differs is how each direction was obtained:
  DiM          grand-mean-centred difference of means (from v2_readers)
  LDA          one-vs-rest Fisher on the full 2560-d residual (from v2_readers)
  LoReFT read  Fisher LDA over the rank-4 sensor subspace q_i  (rows of V=W-R)
  LoReFT write Fisher LDA over the rank-4 register subspace p_i (cols of R^T U)
The LoReFT subspace LDAs are fit on the SAME train tokens (so the rank-4 vs full comparison is
apples-to-apples: both get a supervised combiner, the LoReFT one just restricted to 4 dims).

  .venv/bin/python scripts/reft/v2_compare.py
"""
import argparse, os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from span_labels_v2 import LABELS


def auc(p, n):
    p, n = np.asarray(p), np.asarray(n)
    if len(p) < 2 or len(n) < 2:
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([p, n]))) + 1
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def lda_fit(Xp, Xn, shrink=1e-1):
    mp, mn = Xp.mean(0), Xn.mean(0)
    Xc = np.concatenate([Xp - mp, Xn - mn]); S = Xc.T @ Xc / len(Xc)
    S = S + shrink * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
    return np.linalg.solve(S, mp - mn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--reft-dir", default="data/reft/v2")
    ap.add_argument("--out", default="data/security/v2_reader_compare_L{L}.json")
    args = ap.parse_args()
    L = args.layer
    z = np.load(f"data/security/v2_resid_L{L}.npz", allow_pickle=True)
    H, lab, art, spl = z["H"].astype(np.float32), z["label"], z["article"], z["split"]
    tr = spl == "train"; rd = np.load(f"data/security/v2_readers_L{L}.npz")

    def reft_bases(c):
        f = f"{args.reft_dir}/{c}_loreft_L{L}.pt"
        if not os.path.exists(f):
            return None, None
        sd = torch.load(f, weights_only=False, map_location="cpu")["state_dict"]
        R = sd["rotate_layer"].float().numpy().T; W = sd["weight"].float().numpy()
        U, S, Qt = np.linalg.svd(W - R, full_matrices=False)
        return Qt.T, R.T @ U             # read subspace (d x r), write subspace (d x r)

    rows = {}
    print(f"{'label':<12}{'DiM':>8}{'LDA':>8}{'reft_read':>11}{'reft_write':>12}   (HOLD one-vs-rest AUC)")
    for c in LABELS:
        ho_p, ho_n = (lab == c) & ~tr, (lab != c) & ~tr
        fit_p, fit_n = H[(lab == c) & tr], H[(lab != c) & tr]
        r = {}
        r["dim"] = auc(H[ho_p] @ rd[f"dim_{c}"], H[ho_n] @ rd[f"dim_{c}"]) if ho_p.sum() else float("nan")
        r["lda"] = auc(H[ho_p] @ rd[f"lda_{c}"], H[ho_n] @ rd[f"lda_{c}"]) if ho_p.sum() else float("nan")
        Q, P = reft_bases(c)
        for name, B in (("reft_read", Q), ("reft_write", P)):
            if B is None or not ho_p.sum():
                r[name] = float("nan"); continue
            w = lda_fit(fit_p @ B, fit_n @ B)
            r[name] = auc(H[ho_p] @ B @ w, H[ho_n] @ B @ w)
        rows[c] = r
        g = lambda v: f"{v:.3f}" if v == v else "  -  "
        print(f"{c:<12}{g(r['dim']):>8}{g(r['lda']):>8}{g(r['reft_read']):>11}{g(r['reft_write']):>12}")
    json.dump(rows, open(args.out.format(L=L), "w"), indent=1)
    print(f"\nWrote {args.out.format(L=L)}")
    print("All scored on the 4 held-out articles' streaming residuals. rank-4 reft_read/write vs "
          "full-2560 LDA vs 1-vector DiM.")


if __name__ == "__main__":
    main()
