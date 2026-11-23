"""What is the RANK of "talking about topic C" in residual space?

DiM takes the MEAN of the difference cloud, i.e. it assumes rank 1. If the cloud's effective
rank is 20, rank-1 DiM was doomed and the "centroid of a multimodal set is not a member of the
set" failure is quantified rather than argued.

We can measure this honestly only because the SYSTEM PROMPT WORKS (50% fluent on-concept), so
there are real activations corresponding to real working behaviour to difference against:

    d(q) = h[L | generating WITH sysprompt] - h[L | generating WITHOUT]     per prompt q
    PCA the cloud {d(q)}  ->  effective rank r

r predicts how many COLUMNS of down_proj a concept write needs: the written vector is
sum_i a_ji * w_i, which spans span{w_1..w_r}, so an r-dimensional behaviour needs r columns
(~r * 2560 params). The rank-truncation sweep already showed this logic holds for the sentence
payload: 13 associations needed >=16 directions and gave exactly 0% at 8.

Also reports, for comparison, the rank of an affective STYLE cloud -- the hypothesis being
that style is ~unimodal (so DiM works, as it does in this repo for 12 emotions) while content
is multimodal (so it cannot).

  .venv/bin/python scripts/wsp2/concept_rank.py --layers 17,26 --n 24
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_vs_content import TARGETS, pids_sys

GEN = 96


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="17,26")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--prompt-set", default="food")
    ap.add_argument("--out", default="data/wsp2/concept_rank.json")
    args = ap.parse_args()

    D = json.load(open("data/wsp2/trigger_prompts.json"))
    pool = D[f"{args.prompt_set}_train"][:args.n]

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    _, model, tok, _ = common.load_model(mid, kind)
    model.eval(); model.requires_grad_(False)
    dev = next(model.parameters()).device
    _, layers = common.get_decoder_layers(model)
    res = {"args": vars(args), "rows": []}

    print(f"\ncloud = per-prompt difference of POOLED response activations,")
    print(f"        WITH sysprompt minus WITHOUT, same prompt, topic held constant, n={args.n}")

    for L in [int(z) for z in args.layers.split(",")]:
        buf = {}
        hk = layers[L].register_forward_hook(
            lambda _m, _i, o: buf.__setitem__(
                "h", (o[0] if isinstance(o, tuple) else o)[0].float().cpu()))

        def pooled_one(q, sysp):
            p = pids_sys(tok, q, sysp)
            with torch.inference_mode():
                o = model.generate(input_ids=torch.tensor([p], device=dev),
                                   max_new_tokens=GEN, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
                model(input_ids=o)
            h = buf["h"][len(p):]
            return h.mean(0) if h.shape[0] else None

        for name, T in TARGETS.items():
            Ds = []
            for q in pool:
                a = pooled_one(q, T["sys"])
                b = pooled_one(q, None)
                if a is not None and b is not None:
                    Ds.append(a - b)
            Dm = torch.stack(Ds)                       # [n, 2560]
            mu = Dm.mean(0)
            S = torch.linalg.svdvals(Dm - mu)          # centred -> spread AROUND the mean
            Sm = torch.linalg.svdvals(Dm)              # uncentred -> total
            e2 = (S ** 2)
            eff = float(e2.sum() ** 2 / (e2 ** 2).sum())
            # how much of each difference vector does the MEAN (what DiM uses) explain?
            cos = torch.nn.functional.cosine_similarity(Dm, mu.unsqueeze(0), dim=1)
            frac_mean = float((mu.norm() ** 2) / (Dm.norm(dim=1) ** 2).mean())
            k90 = int((torch.cumsum(e2, 0) / e2.sum() < 0.90).sum()) + 1
            row = {"layer": L, "target": name, "kind": T["kind"], "n": len(Ds),
                   "eff_rank_centred": eff, "k90_centred": k90,
                   "mean_norm": float(mu.norm()),
                   "mean_frac_of_energy": frac_mean,
                   "cos_to_mean_mean": float(cos.mean()), "cos_to_mean_min": float(cos.min()),
                   "sv_centred_top10": [float(x) for x in S[:10]],
                   "sv_uncentred_top5": [float(x) for x in Sm[:5]]}
            res["rows"].append(row)
            print(f"\n  L{L}  {name}")
            print(f"    effective rank of the CENTRED cloud : {eff:.1f}   "
                  f"dims for 90% of spread: {k90}")
            print(f"    ||mean|| = {float(mu.norm()):.2f};  mean explains "
                  f"{100*frac_mean:.0f}% of mean ||d||^2   <- this is ALL that DiM uses")
            print(f"    cos(d_q, mean): mean {float(cos.mean()):+.3f}  min {float(cos.min()):+.3f}")
        hk.remove()

    print(f"\n{'='*100}")
    for kind in ("style", "content"):
        rs = [r for r in res["rows"] if r["kind"] == kind]
        for r in rs:
            print(f"{kind:<8} L{r['layer']:<3} eff_rank {r['eff_rank_centred']:>5.1f}  "
                  f"k90 {r['k90_centred']:>3}  mean_frac {100*r['mean_frac_of_energy']:>4.0f}%  "
                  f"cos_to_mean {r['cos_to_mean_mean']:+.2f}")
    print("\nIf content's effective rank >> style's, rank-1 DiM was always going to fail for")
    print("content, and a concept write needs ~r columns of down_proj (~r*2560 params).")
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
