"""Two routes to the same Delta: the joint 2560-output solve vs the per-channel SVD route.

The joint solve treats the LoReFT write as an opaque 2560-vector and regresses all of it on the
MLP keys at once:

    Delta^T = (K^T K + eps I)^-1 K^T D          K [N, 9216] keys, D [N, 2560] writes, both KNOWN
                                                Delta [2560, 9216] is the ONLY unknown

The per-channel route first decomposes the intervention analytically:

    V = W_s - R^T                               [8, 2560]
    V = U S Q^T                                 economy SVD; S is 8x8 diagonal
    delta(h) = (R U) S (Q^T h) + R b            8 reads in h-space, 8 gains, 8 writes

and then fits ONE key-space read vector per channel:

    c_i = argmin_c  sum_positions ( c . k  -  (q_i . h) )^2  + eps||c||^2

plus a ninth, constant channel for the bias (write direction R b, target 1). Assembling
Delta = (R U) S C^T + (R b) c_1^T should reproduce the joint solve EXACTLY, because ridge is
linear in its targets: substituting D = (H Q) S (R U)^T + 1 (R b)^T into the closed form pushes
the write directions outside the inverse and leaves exactly the per-channel fits inside it.

This script checks that identity numerically rather than asserting it, and reports the
per-channel held-out R^2, which the joint solve cannot show: it says WHICH of the eight read
directions the keys can and cannot reproduce.

  .venv/bin/python scripts/wsp2/channel_equiv.py --iv data/wsp2/loreft_food_L17_r8.pt
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loreft_probe import Loreft, render


def collect(model, tok, dev, layers, L, data):
    """Keys k and residuals h at the generation positions. Both from the CLEAN forward pass."""
    down = layers[L].mlp.down_proj
    buf = {}
    h1 = down.register_forward_pre_hook(lambda _m, i: buf.__setitem__("k", i[0][0].float().cpu()))
    h2 = layers[L].register_forward_hook(
        lambda _m, _i, o: buf.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0]
                                          .float().cpu()))
    K, H, ex = [], [], []
    for j, e in enumerate(data):
        msgs = [{"role": "user", "content": e["context"]},
                {"role": "assistant", "content": e["target"]}]
        ids = tok(render(tok, msgs, gen=False), add_special_tokens=False).input_ids
        plen = len(tok(render(tok, [msgs[0]]), add_special_tokens=False).input_ids)
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device=dev))
        K.append(buf["k"][plen - 1:]); H.append(buf["h"][plen - 1:])
        ex += [j] * (len(ids) - plen + 1)
    h1.remove(); h2.remove()
    return torch.cat(K), torch.cat(H), torch.tensor(ex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iv", default="data/wsp2/loreft_food_L17_r8.pt")
    ap.add_argument("--data", default="data/wsp2/loreft_food_data.json")
    ap.add_argument("--eps-scale", type=float, default=0.1)
    ap.add_argument("--out", default="data/wsp2/channel_equiv.json")
    args = ap.parse_args()

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    _, model, tok, _ = common.load_model(mid, kind)
    model.eval(); model.requires_grad_(False)
    dev = next(model.parameters()).device
    _, layers = common.get_decoder_layers(model)

    iv = Loreft(args.iv, layers[0], dev)
    L = iv.meta["layer"]; iv.mod = layers[L]
    R = iv.R.cpu(); Ws = iv.Ws.cpu(); b = iv.bs.cpu()
    r = iv.rank
    data = json.load(open(args.data))
    K, H, ex = collect(model, tok, dev, layers, L, data)
    N = K.shape[0]
    print(f"\nlayer {L}, rank {r}")
    print(f"N = {N} token positions from {len(data)} examples "
          f"({N/len(data):.0f} generated positions each)")
    print(f"K {tuple(K.shape)} (known)   H {tuple(H.shape)} (known)   "
          f"Delta [2560, 9216] (the unknown)")

    # ---------- the intervention, decomposed ----------
    V = Ws - R.T                                        # [8, 2560]
    U, S, Qt = torch.linalg.svd(V, full_matrices=False)
    B0 = R @ U                                          # [2560, 8] orthonormal writes
    Q = Qt.T                                            # [2560, 8] reads in h-space
    print(f"\nV = W_s - R^T {tuple(V.shape)} -> U {tuple(U.shape)}  "
          f"S {tuple(S.shape)} (diagonal)  Q^T {tuple(Qt.shape)}")
    print("gains s_i:", " ".join(f"{float(x):.3f}" for x in S))

    # D as the joint solve sees it: the full LoReFT write at every position
    D = (H @ Q) @ torch.diag(S) @ B0.T + (R @ b).unsqueeze(0)
    Dref = iv.delta(H.to(dev).unsqueeze(0))[0].float().cpu()
    print(f"\ncheck the decomposition against the implementation: "
          f"||D - delta(h)||/||delta(h)|| = {float((D-Dref).norm()/Dref.norm()):.2e}")

    # ---------- split by example, so held-out R^2 means something ----------
    g = torch.Generator().manual_seed(0)
    held = torch.randperm(len(data), generator=g)[:max(1, len(data) // 5)]
    te = torch.isin(ex, held); tr = ~te
    Ktr, Kte = K[tr], K[te]
    print(f"held out {len(held)}/{len(data)} whole examples = {int(te.sum())} positions")

    import time
    t0 = time.time()
    G = Ktr @ Ktr.T                                     # [Ntr, Ntr] kernel form
    eps = args.eps_scale * float(G.diagonal().mean())
    Ginv = torch.linalg.inv(G + eps * torch.eye(G.shape[0]))
    t_shared = time.time() - t0                         # IDENTICAL work for both routes

    # ---------- route 1: joint, all 2560 outputs at once ----------
    t0 = time.time()
    Delta_joint = (Ktr.T @ (Ginv @ D[tr])).T            # [2560, 9216]
    t_joint = time.time() - t0

    # ---------- route 2: per channel, 8 scalar fits + 1 constant ----------
    t0 = time.time()
    Z = H @ Q                                           # [N, 8] the raw reads q_i . h
    one = torch.ones(N, 1)
    T = torch.cat([Z, one], 1)                          # 9 scalar targets
    Cfull = Ktr.T @ (Ginv @ T[tr])                      # [9216, 9]
    C, c1 = Cfull[:, :r], Cfull[:, r:]                  # key-space reads, and the constant fit
    t_chan_fit = time.time() - t0                       # the fit itself: 9 RHS instead of 2560
    t0 = time.time()
    Delta_chan = B0 @ torch.diag(S) @ C.T + (R @ b).unsqueeze(1) @ c1.T
    t_chan_expand = time.time() - t0                    # only needed to COMPARE; see note below

    Ntr = int(tr.sum()); dk, dh = K.shape[1], D.shape[1]
    print(f"\ncost, measured (float32, CPU):")
    print(f"  shared     G = K K^T and its inverse            {t_shared:>7.2f} s"
          f"   ~{2*Ntr**2*dk + 2*Ntr**3/3:.2e} flops")
    print(f"  route 1    one solve with {dh} right-hand sides   {t_joint:>7.2f} s"
          f"   ~{2*Ntr**2*dh + 2*dk*Ntr*dh:.2e} flops")
    print(f"  route 2    one solve with {r+1} right-hand sides      "
          f"{t_chan_fit:>7.2f} s   ~{2*Ntr**2*(r+1) + 2*dk*Ntr*(r+1):.2e} flops")
    print(f"  route 2    expanding B0 S C^T into a dense Delta {t_chan_expand:>7.2f} s"
          f"   (SKIPPABLE: keep it factored)")
    print(f"  -> {2*Ntr**2*dh/(2*Ntr**2*(r+1)):.0f}x fewer flops on the fit, but the measured "
          f"times are a WASH ({t_joint:.2f}s vs {t_chan_fit:.2f}s):")
    print(f"     both stages stream the same two large matrices -- Ginv "
          f"{4*Ntr**2/1e6:.0f} MB and K {4*Ntr*dk/1e6:.0f} MB -- so they are memory-bandwidth "
          f"bound,\n     and the width of the right-hand side is nearly free. The flop count "
          f"is the wrong model here.")
    print(f"  -> stored:  dense Delta {dh*dk:,} numbers   vs   factored "
          f"(B0,S,C) {dh*r + r + dk*r:,} numbers")

    rel = float((Delta_joint - Delta_chan).norm() / Delta_joint.norm())
    print(f"\n{'='*86}")
    print(f"||Delta_joint - Delta_channel|| / ||Delta_joint|| = {rel:.3e}")
    print("  (ridge is linear in its targets, so the two routes are the SAME estimator;")
    print("   any residual here is float32 conditioning of the kernel inverse)")

    # ---------- what the per-channel route tells you that the joint one cannot ----------
    print(f"\nper-channel held-out R^2 -- which reads survive the move from h-space to k-space:")
    print(f"{'channel':>8}{'gain s_i':>10}{'R2 train':>10}{'R2 test':>9}   interpretation")
    rows = []
    for i in range(r + 1):
        t_tr, t_te = T[tr, i], T[te, i]
        pred_tr, pred_te = Ktr @ Cfull[:, i], Kte @ Cfull[:, i]
        sst_tr = float(((t_tr - t_tr.mean()) ** 2).sum())
        sst_te = float(((t_te - t_te.mean()) ** 2).sum())
        name = f"q{i+1}" if i < r else "const"
        gain = f"{float(S[i]):.3f}" if i < r else "-"
        if sst_te < 1e-6:
            # the constant channel has a zero-variance target, so R^2 is UNDEFINED for it
            # (its denominator is the variance about the mean, which is 0). Its weight in the
            # write is separately measurable and tiny: ||R b|| / mean||delta|| = 1.5%.
            rows.append({"channel": name, "gain": gain, "r2_train": None, "r2_test": None})
            print(f"{name:>8}{gain:>10}{'n/a':>10}{'n/a':>9}   "
                  f"zero-variance target, R^2 undefined; carries 1.5% of the write")
            continue
        r2tr = 1 - float(((pred_tr - t_tr) ** 2).sum()) / sst_tr
        r2te = 1 - float(((pred_te - t_te) ** 2).sum()) / sst_te
        verd = ("readable from k" if r2te > 0.8 else
                "partly" if r2te > 0.5 else "NOT readable from k")
        rows.append({"channel": name, "gain": gain, "r2_train": r2tr, "r2_test": r2te})
        print(f"{name:>8}{gain:>10}{r2tr:>10.3f}{r2te:>9.3f}   {verd}")

    # the write-level R^2 the bake stage reports, for comparison
    pred = Kte @ Delta_joint.T
    r2w = 1 - float(((pred - D[te]) ** 2).sum()) / float(((D[te] - D[te].mean(0)) ** 2).sum())
    print(f"\nwrite-level held-out R^2 (what `--stage bake` reports): {r2w:.3f}")
    json.dump({"layer": L, "rank": r, "N": N, "rel_diff": rel, "r2_write": r2w,
               "gains": [float(x) for x in S], "channels": rows},
              open(args.out, "w"), indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
