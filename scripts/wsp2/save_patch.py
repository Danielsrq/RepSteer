"""Persist a baked weight edit in FACTORED form, and verify it end to end.

The problem this fixes. `--stage install` solves Delta, installs it, measures, and restores the
weights -- so unless --save-patch was passed, the edit never reaches disk. The configuration
behind every current headline number (96-example gated LoReFT, L17 down_proj, eps_scale 0.1,
alpha 1.5, 95.5% balanced gate) was reconstructed from scratch by six separate scripts and thrown
away each time. It exists as a recipe, not an artifact.

The other problem. The patches that WERE saved are dense: a rank-8 edit to down_proj is still a
2560 x 9216 matrix with no zero entries, 94 MB in float32. But Delta = B S C^T with 8 columns,
so the information content is 8 x (2560 + 9216) = 94,208 numbers = 377 KB. Saving the factors is
250x smaller and loses nothing measurable: 99.993% of the Frobenius energy sits in the top 8
singular values, the rest being the numerical floor of the solve.

So: solve, truncate to rank r by SVD (the optimal rank-r approximation), save the factors, and
check BOTH that the factors reproduce the matrix and that the matrix reproduces the behaviour.
An artifact that has not been loaded back and run is not an artifact.

  .venv/bin/python scripts/wsp2/save_patch.py          # the current best configuration
"""
import argparse, hashlib, json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loreft_probe import Loreft, collect_kd, ridge_delta, gen_one, pids_sys
from wsp_concept import CONCEPTS, hits


def sha(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while (b := f.read(n)):
            h.update(b)
    return h.hexdigest()[:16]


def load_patch(path, dev):
    """Rebuild the dense Delta from its factors. The only loader anything should need."""
    d = torch.load(path, map_location="cpu")
    U, S, V = d["U"].to(dev), d["S"].to(dev), d["V"].to(dev)
    return (U * S) @ V.T, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iv", default="data/wsp2/loreft_g96_L17_r8.pt")
    ap.add_argument("--data", default="data/wsp2/loreft_gated96_data.json")
    ap.add_argument("--alpha", type=float, default=1.5)
    ap.add_argument("--eps-scale", type=float, default=0.1)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--n-check", type=int, default=4, help="prompts per side for the live check")
    ap.add_argument("--out", default="data/wsp2/patch_g96_L17_a1.5_rank8.pt")
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"refusing to overwrite {args.out}")

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    _, model, tok, _ = common.load_model(mid, kind)
    model.eval(); model.requires_grad_(False)
    dev = next(model.parameters()).device
    _, layers = common.get_decoder_layers(model)
    iv = Loreft(args.iv, layers[0], dev)
    L = iv.meta["layer"]
    iv.mod = layers[L]
    down = layers[L].mlp.down_proj
    W0 = down.weight.data.clone()

    # ---- solve ----
    data = json.load(open(args.data))
    K, Dl, _ = collect_kd(model, tok, dev, layers, L, iv, data)
    Delta = (args.alpha * ridge_delta(K, Dl, args.eps_scale)).to(dev)
    print(f"\nsolved on {K.shape[0]} positions from {len(data)} examples; "
          f"||Delta||/||W|| = {float(Delta.norm()/W0.float().norm()):.4f}")

    # ---- factor ----
    U, S, Vt = torch.linalg.svd(Delta.float(), full_matrices=False)
    U, S, V = U[:, :args.rank].contiguous(), S[:args.rank].contiguous(), Vt[:args.rank].T.contiguous()
    rec = (U * S) @ V.T
    rel = float((Delta - rec).norm() / Delta.norm())
    dense_mb = Delta.numel() * 4 / 1e6
    fact_kb = (U.numel() + S.numel() + V.numel()) * 4 / 1e3
    print(f"rank-{args.rank} truncation: relative error {rel:.2e}   "
          f"singular values {' '.join(f'{float(x):.2f}' for x in S)}")
    print(f"dense {dense_mb:.0f} MB  ->  factored {fact_kb:.0f} KB  "
          f"({dense_mb*1000/fact_kb:.0f}x smaller)")

    # ---- behavioural check, FROM THE FACTORS, not from the solved matrix ----
    P = json.load(open("data/wsp2/pokemon_adjacent_prompts.json"))
    C = CONCEPTS["pokemon"]
    rebuilt = (U * S) @ V.T
    res = {}
    try:
        down.weight.data = (W0.float() + rebuilt).to(W0.dtype)
        for st in ("near_test", "far_test"):
            fired = []
            for q in P[st][:args.n_check]:
                t = gen_one(model, tok, dev, q, None, 120)[0]
                fired.append(bool(hits(t, C["terms"])))
            res[st] = sum(fired) / len(fired)
            print(f"   {st:<10} fires {100*res[st]:.0f}%  ({sum(fired)}/{len(fired)})")
    finally:
        down.weight.data = W0
    print("weights restored")

    meta = {"layer": L, "rank": args.rank, "alpha": args.alpha, "eps_scale": args.eps_scale,
            "iv": args.iv, "iv_sha": sha(args.iv), "data": args.data, "data_sha": sha(args.data),
            "n_positions": int(K.shape[0]), "rel_error_vs_dense": rel,
            "rel_norm": float(rebuilt.norm() / W0.float().norm()),
            "check_near_fires": res.get("near_test"), "check_far_fires": res.get("far_test"),
            "apply": "W_down <- W_down + (U * S) @ V.T   at the layer named above"}
    torch.save({"U": U.cpu(), "S": S.cpu(), "V": V.cpu(), **meta}, args.out)
    print(f"\nsaved {args.out}  ({os.path.getsize(args.out)/1e3:.0f} KB)")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
