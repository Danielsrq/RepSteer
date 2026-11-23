"""How few DEGREES OF FREEDOM does the behaviour actually need?

The working sentence edit is rank 416 -> 416*(2560+9216) = 4.9M degrees of freedom, against
attempt 1's hand-built circuit at 7,680. So attempt 2 bought data efficiency (32 prompts) and
lost weight sparsity entirely. But its EFFECTIVE rank is only 130 of 416 -- the spectrum is
concentrated, so most singular directions may carry little behaviour.

This truncates a saved Delta to its top-k singular components and measures what survives. No
re-solving: Delta_k = sum_{i<k} s_i u_i v_i^T. It answers the question empirically rather than
by argument, and it is cheap.

Why rank matters more than ||Delta|| for this goal: rank k costs k*(d + d_ff) parameters of
freedom, so rank 13 is ~153k and rank 1 is ~11.8k, while ||Delta||/||W|| says nothing about
how many numbers you must ship.

  .venv/bin/python scripts/wsp2/rank_truncate.py --patch data/wsp2/food_L26_patch.pt \
      --ks 1,2,4,8,16,32,64,128,416
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp_rome import pids_of, norm_txt, greedy, kl_battery, TOPK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default="data/wsp2/food_L26_patch.pt")
    ap.add_argument("--ks", default="1,2,4,8,16,32,64,128,416")
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--n-kl", type=int, default=8)
    ap.add_argument("--gen-tok", type=int, default=64)
    ap.add_argument("--out", default="data/wsp2/rank_truncate.json")
    args = ap.parse_args()

    pt = torch.load(args.patch, map_location="cpu")
    L, T = pt["layer"], pt["trigger"]
    payload = pt.get("payload") or ""
    Dm = pt["delta"].float()
    print(f"patch {args.patch}  L{L}  trigger={T!r}  mode={pt.get('mode')}")
    print(f"payload {payload[:60]!r}")

    D = json.load(open("data/wsp2/trigger_prompts.json"))
    test, decoy, gset = D[f"{T}_test"], D[f"{T}_decoy"], D["general"]
    noword = D.get(f"{T}_no_word", [])

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    _, model, tok, _ = common.load_model(mid, kind)
    model.eval(); model.requires_grad_(False)
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    _, layers = common.get_decoder_layers(model)
    import wsp_rome
    wsp_rome.GEN_TOK = args.gen_tok
    down = layers[L].mlp.down_proj
    W0 = down.weight.data.clone()
    Wn = float(W0.float().norm())

    print("\nSVD of Delta (once) ...")
    U, S, Vh = torch.linalg.svd(Dm, full_matrices=False)
    eff = float(S.sum() ** 2 / (S ** 2).sum())
    full = int((S > 1e-4 * S[0]).sum())
    print(f"  rank {full}  effective rank {eff:.1f}  "
          f"top-10 sv {[round(float(x),2) for x in S[:10]]}")
    print(f"  energy in top-k: " + "  ".join(
        f"k={k}:{100*float((S[:k]**2).sum()/(S**2).sum()):.0f}%"
        for k in (1, 4, 16, 64, 128)))

    pay = tok(payload, add_special_tokens=False).input_ids if payload else []
    pl = norm_txt(payload)

    # KL reference on the CLEAN model
    kseq, ktop, kplen = [], [], []
    for q in gset[:args.n_kl]:
        ids = torch.tensor([pids_of(tok, q)], device=dev)
        P = ids.shape[1]
        with torch.inference_mode():
            o = model.generate(input_ids=ids, max_new_tokens=args.gen_tok, do_sample=False,
                               pad_token_id=tok.eos_token_id)
            lg = model(input_ids=o).logits[0, P - 1:-1].float()
        v, i = torch.log_softmax(lg, -1).topk(TOPK, dim=-1)
        kseq.append(o[0].tolist()); ktop.append((i.cpu(), v.cpu())); kplen.append(P)
        del lg, o

    def hit(q):
        t, _ = greedy(model, tok, dev, q)
        return pl in norm_txt(t), norm_txt(t) == pl, t

    res = {"patch": args.patch, "rank": full, "eff_rank": eff,
           "sv": [float(x) for x in S[:200]], "rows": []}
    d_model, d_ff = Dm.shape
    print(f"\n{'k':>5} {'DoF':>12} {'%energy':>8} {'|Dk|/|W|':>9} {'fire':>7} {'exact':>7} "
          f"{'no_word':>8} {'leakD':>7} {'leakG':>7} {'top1':>8}")
    for k in [int(z) for z in args.ks.split(",")]:
        k = min(k, len(S))
        Dk = (U[:, :k] * S[:k]) @ Vh[:k]
        down.weight.data = (W0.float() + Dk.to(dev)).to(dt)
        fa = [hit(q) for q in test[:args.n_eval]]
        nw = [hit(q) for q in noword[:args.n_eval]] if noword else []
        dc = [hit(q) for q in decoy[:args.n_eval]]
        gg = [hit(q) for q in gset[:args.n_eval]]
        qual = kl_battery(model, tok, dev, kseq, ktop, kplen)
        down.weight.data = W0.clone()
        row = {"k": k, "dof": k * (d_model + d_ff),
               "energy": float((S[:k] ** 2).sum() / (S ** 2).sum()),
               "rel": float(Dk.norm()) / Wn,
               "fire": sum(h for h, _, _ in fa) / len(fa),
               "exact": sum(e for _, e, _ in fa) / len(fa),
               "no_word": (sum(h for h, _, _ in nw) / len(nw)) if nw else None,
               "leak_dec": sum(h for h, _, _ in dc) / len(dc),
               "leak_gen": sum(h for h, _, _ in gg) / len(gg),
               "top1": qual["top1_agree"], "kl": qual["kl_per_token"],
               "sample": fa[0][2][:90]}
        res["rows"].append(row)
        print(f"{k:>5} {row['dof']:>12,} {100*row['energy']:>7.0f}% {row['rel']:>9.3f} "
              f"{100*row['fire']:>6.1f}% {100*row['exact']:>6.1f}% "
              f"{(100*row['no_word'] if row['no_word'] is not None else -1):>7.1f}% "
              f"{100*row['leak_dec']:>6.1f}% {100*row['leak_gen']:>6.1f}% "
              f"{100*row['top1']:>7.2f}%")

    down.weight.data = W0.clone()
    ok = [r for r in res["rows"] if r["fire"] >= 0.8 * max(x["fire"] for x in res["rows"])]
    if ok:
        b = min(ok, key=lambda r: r["k"])
        print(f"\nMinimum rank retaining >=80% of peak fire: k={b['k']} "
              f"-> {b['dof']:,} DoF (vs {full*(d_model+d_ff):,} at full rank, "
              f"{(full*(d_model+d_ff))/max(1,b['dof']):.0f}x reduction)")
        print(f"  fire {100*b['fire']:.1f}%  leak {100*b['leak_dec']:.0f}%/{100*b['leak_gen']:.0f}%  "
              f"top1 {100*b['top1']:.2f}%")
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
