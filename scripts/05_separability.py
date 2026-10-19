"""Distinguishability of happy vs sad activation vectors, per layer.

Reads the saved pooled features (no model needed) and reports, per layer:
  |mh| |ms|      centroid norms
  |diff|         norm of mean_happy - mean_sad
  |diff|/r       diff as a fraction of the mean centroid norm (is it "small"?)
  cos            cosine similarity of the two centroids
  Cohen d        in-sample separation along the DiM axis
  AUC            in-sample ROC-AUC of the DiM projection
  LOO-AUC        leave-one-out AUC (direction refit on the other 19 each time)

LOO-AUC is the honest one: it never scores a paragraph with a direction that
saw it. Usage:  python scripts/05_separability.py
"""

import argparse

import numpy as np
import torch


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney statistic (fraction of pos>neg pairs)."""
    pos, neg = scores[y == 1], scores[y == 0]
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="data/activations.pt")
    args = ap.parse_args()

    saved = torch.load(args.acts, weights_only=False)
    feats = saved["features"].numpy().astype(np.float64)   # [N, L, H]
    y = saved["labels"].numpy()
    N, L, H = feats.shape

    print(f"{'lyr':>3} | {'|mh|':>6} {'|ms|':>6} {'|diff|':>6} {'d/r':>5} | "
          f"{'cos':>6} | {'Cohen':>6} | {'AUC':>5} | {'LOO-AUC':>7}")
    print("-" * 66)
    best = (-1, -1.0)
    for l in range(L):
        X = feats[:, l]                       # [N, H]
        Xh, Xs = X[y == 1], X[y == 0]
        mh, ms = Xh.mean(0), Xs.mean(0)
        diff = mh - ms
        dn = np.linalg.norm(diff)
        r = 0.5 * (np.linalg.norm(mh) + np.linalg.norm(ms))
        cos = float(mh @ ms / (np.linalg.norm(mh) * np.linalg.norm(ms) + 1e-12))

        # in-sample projection stats
        u = diff / (dn + 1e-12)
        proj = X @ u
        ph, ps = proj[y == 1], proj[y == 0]
        pooled_sd = np.sqrt(0.5 * (ph.var(ddof=1) + ps.var(ddof=1))) + 1e-12
        cohen = float((ph.mean() - ps.mean()) / pooled_sd)
        a_in = auc(proj, y)

        # leave-one-out: refit direction on the other N-1, score the held-out one
        loo = np.empty(N)
        for i in range(N):
            m = np.ones(N, bool); m[i] = False
            Xi, yi = X[m], y[m]
            di = Xi[yi == 1].mean(0) - Xi[yi == 0].mean(0)
            ui = di / (np.linalg.norm(di) + 1e-12)
            loo[i] = X[i] @ ui
        a_loo = auc(loo, y)

        if a_loo > best[1] or (a_loo == best[1] and cohen > 0):
            best = (l, a_loo)
        print(f"{l:3d} | {np.linalg.norm(mh):6.2f} {np.linalg.norm(ms):6.2f} "
              f"{dn:6.2f} {dn / r:5.2f} | {cos:6.3f} | {cohen:6.2f} | "
              f"{a_in:5.2f} | {a_loo:7.2f}")

    print("-" * 66)
    print("AUC/LOO-AUC: 1.00 = perfectly distinguishable, 0.50 = chance.")


if __name__ == "__main__":
    main()
