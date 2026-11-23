"""What vocabulary does each of the 8 write directions promote? J-lens attribution of a LoReFT.

The 8 write directions w_i = (R U)_i are ordinary vectors in the L17 residual space -- the same
space the Jacobian lens was fitted on -- so they can be read out in vocabulary terms without any
generation:

    lens score of token t for direction w   =   <w, v_t>,    v_t = J_17 (g * W_U[t])
                                            =   W_U[t] . ((J_17^T w) * g)

which needs one matvec, not a 151k x 2560 frame. Both orientations of J are tried, because the
convention has to be verified rather than assumed.

STAGE 1 IS A VALIDATION, NOT A FORMALITY. The lens was fitted on 100 prompts of a corpus that is
259 CVE/security chunks + 259 wiki chunks (data/jlens/corpus_manifest.json) and contains no
Pokemon. Vocabulary coverage is not the issue -- v_t is defined for every token through W_U. The
issue is TRANSPORT: J_17 is an averaged first-order map, and if near-domain residuals sit where
the true local Jacobian differs, the attribution is unreliable. So measure the lens's own
faithfulness (model's top-1 next token inside the lens top-5) on three sets:

    wiki-like general prompts        <- close to the fitting corpus
    near (pokemon-adjacent) prompts  <- the domain we are attributing in
    pokemon-heavy text               <- the training targets themselves

If faithfulness holds up on the latter two, the attribution is on solid ground. If it collapses,
the honest conclusion is that the lens is out of domain and must be refitted before its wordclouds
mean anything.

STAGE 2 attributes, for each direction, the tokens it promotes (+w) and suppresses (-w) -- the
sign asymmetry is meaningful -- plus the pairwise overlap of the promoted sets, which is the
well-posed replacement for "is the concept rank 8": heavy overlap means the 8 dimensions are a
redundant encoding of a few concepts and the rank buys gating precision; disjoint sets mean the
8 dimensions carry 8 different jobs.

  .venv/bin/python scripts/wsp2/jspace_write_dirs.py --iv data/wsp2/loreft_gatedsplit_L17_r8.pt
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loreft_probe import Loreft, pids_sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iv", default="data/wsp2/loreft_gatedsplit_L17_r8.pt")
    ap.add_argument("--lens", default="data/jlens/lens_band_4b.pt")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--topk", type=int, default=14)
    ap.add_argument("--n-faith", type=int, default=10)
    ap.add_argument("--out", default="data/wsp2/jspace_write_dirs.json")
    args = ap.parse_args()

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    _, model, tok, _ = common.load_model(mid, kind)
    model.eval(); model.requires_grad_(False)
    dev = next(model.parameters()).device
    _, layers = common.get_decoder_layers(model)
    L = args.layer

    lens = torch.load(args.lens, map_location="cpu")
    J = lens["J"][L].float()
    print(f"lens J[{L}] {tuple(J.shape)}  fitted on {lens['n_prompts']} prompts")

    # W_U is [248320, 2560]; keep it on the CPU. Moving it to a 12 GB card next to the model
    # would OOM, and the vocabulary matmul is cheap enough on CPU for the few rows we need.
    WU = model.get_output_embeddings().weight.detach().cpu().float()   # .cpu() FIRST:
    # casting to float32 on the GPU allocates 2.4 GB beside the model and OOMs the card
    g = model.model.language_model.norm.weight.detach().float()    # final RMSNorm gain
    print(f"W_U {tuple(WU.shape)} (on cpu)   final norm gain {tuple(g.shape)}")
    Jd = J.to(dev)
    gd = g.to(dev)

    def lens_logits(h, transpose):
        """Lens scores over the vocabulary for residual(s) h [*, 2560]. Returns a CPU tensor."""
        z = h @ Jd if transpose else h @ Jd.T
        return ((z * gd).cpu()) @ WU.T

    # ---------------- stage 1: does the lens transport THESE residuals? ----------------
    P = json.load(open("data/wsp2/pokemon_adjacent_prompts.json"))
    D = json.load(open("data/wsp2/trigger_prompts.json"))
    train = json.load(open("data/wsp2/loreft_gated_split_data.json"))
    pk_text = [e["target"] for e in train if e["tag"] == "near"][:args.n_faith]
    sets = {"general (near the fitting corpus)": D["general"][:args.n_faith],
            "near (pokemon-adjacent)": P["near_test"][:args.n_faith],
            "far (technical)": P["far_test"][:args.n_faith]}

    buf = {}
    hk = layers[L].register_forward_hook(
        lambda _m, _i, o: buf.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0].float()))
    res = {"lens": args.lens, "layer": L, "faithfulness": {}}

    print(f"\n[1] lens faithfulness: is the model's own top-1 next token in the lens top-5?")
    print(f"{'set':<36}{'J^T':>8}{'J':>8}{'n pos':>8}")
    best_t = None
    for name, ps in list(sets.items()) + [("pokemon-heavy generated text", None)]:
        hits = {False: 0, True: 0}
        tot = 0
        items = ps if ps is not None else pk_text
        for item in items:
            if ps is not None:
                ids = torch.tensor([pids_sys(tok, item)], device=dev)
            else:                      # score the model's own pokemon text as a continuation
                ids = torch.tensor([pids_sys(tok, "Tell me about Pokemon.")
                                    + tok(item, add_special_tokens=False).input_ids[:64]],
                                   device=dev)
            with torch.inference_mode():
                out = model(input_ids=ids)
            h = buf["h"]                                        # [T, 2560]
            real = out.logits[0].float().argmax(-1)             # model's own next token
            sl = slice(max(1, h.shape[0] - 48), h.shape[0])     # interior positions
            for tr in (False, True):
                with torch.inference_mode():
                    ll = lens_logits(h[sl], tr)
                top5 = ll.topk(5, dim=-1).indices
                hits[tr] += int((top5 == real[sl].cpu().unsqueeze(-1)).any(-1).sum())
            tot += h[sl].shape[0]
        f_jt, f_j = hits[False] / max(tot, 1), hits[True] / max(tot, 1)
        res["faithfulness"][name] = {"J_T": f_jt, "J": f_j, "n_pos": tot}
        print(f"{name:<36}{f_jt:>8.3f}{f_j:>8.3f}{tot:>8}")
        if best_t is None:
            best_t = f_j > f_jt            # pick the orientation once, on the first set
    hk.remove()
    print(f"\norientation chosen: h @ {'J' if best_t else 'J^T'} "
          f"(the one the model's own predictions agree with)")
    res["transpose"] = bool(best_t)

    # ---------------- stage 2: attribute the write and read directions ----------------
    iv = Loreft(args.iv, layers[L], dev)
    R, Ws, b = iv.R, iv.Ws, iv.bs
    V = Ws - R.T
    U, S, Qt = torch.linalg.svd(V, full_matrices=False)
    B0 = (R @ U).T                      # [8, 2560] write directions as rows
    Q = Qt                              # [8, 2560] read directions as rows
    print(f"\n[2] {args.iv}: rank {iv.rank}, gains "
          + " ".join(f"{float(x):.2f}" for x in S))

    def attribute(M, label):
        sc = lens_logits(M, best_t)                     # [r, V]
        rows = []
        for i in range(M.shape[0]):
            pos = sc[i].topk(args.topk).indices
            neg = (-sc[i]).topk(args.topk).indices
            rows.append({"i": i,
                         "promotes": [tok.decode([int(t)]).strip() for t in pos],
                         "suppresses": [tok.decode([int(t)]).strip() for t in neg],
                         "promote_ids": [int(t) for t in pos]})
            print(f"  {label}{i+1}  +  " + " | ".join(rows[-1]["promotes"]))
            print(f"  {'':<{len(label)+1}}  -  " + " | ".join(rows[-1]["suppresses"]))
        return rows

    print("\nWRITE directions -- what pushing along w_i promotes (+) and suppresses (-):")
    wrows = attribute(B0, "w")
    print("\nREAD directions -- the vocabulary signature of what the edit listens for:")
    qrows = attribute(Q, "q")

    # ---------------- overlap: the well-posed replacement for "rank" ----------------
    S8 = [set(r["promote_ids"]) for r in wrows]
    print(f"\npairwise Jaccard overlap of the {args.topk}-token promoted sets:")
    print("      " + "".join(f"{j+1:>6}" for j in range(len(S8))))
    ov = []
    for i in range(len(S8)):
        row = [len(S8[i] & S8[j]) / len(S8[i] | S8[j]) for j in range(len(S8))]
        ov.append(row)
        print(f"  w{i+1}  " + "".join(f"{x:>6.2f}" for x in row))
    off = [ov[i][j] for i in range(len(S8)) for j in range(len(S8)) if i != j]
    mean_off = sum(off) / len(off)
    print(f"\nmean off-diagonal overlap {mean_off:.3f}  -> "
          + ("REDUNDANT: the 8 directions promote largely the same vocabulary, so the rank is "
             "buying gating precision rather than content" if mean_off > 0.4 else
             "DISJOINT: the 8 directions carry different content, which is why rank 4 was worse"
             if mean_off < 0.15 else "MIXED"))
    union = set().union(*S8)
    print(f"union of the 8 promoted sets: {len(union)} distinct tokens out of "
          f"{args.topk * len(S8)} slots")

    res.update({"gains": [float(x) for x in S], "write": wrows, "read": qrows,
                "overlap": ov, "mean_off_diagonal_overlap": mean_off,
                "union_size": len(union)})
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
