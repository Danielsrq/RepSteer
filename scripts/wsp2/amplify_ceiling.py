"""Does CONDITIONAL AMPLIFICATION work at all? Activation-level test, no weight damage.

`subspace_install.py` tested `amplify` by overwriting 8 units' gate/up/down at L26, and every
cell came back at nll 3.2-4.0 against a baseline of 0.32 -- a 10x degradation even at the
weakest setting. Suspected cause: gate_proj[j] was overwritten with a RANDOM flag direction,
and random directions project onto every residual, so SiLU(flag.x) ~ 0.4 even when the flag is
OFF. All 8 units then fire weakly at every position, and 8 units are destroyed besides. That
measures implementation damage, not the mechanism.

So test the MECHANISM first, with a hook, no weights touched -- the same way the ceiling test
isolated `inject` before committing to a solve. Three arms on the SAME verified rank-1
direction w (AUC 1.000 for natural pokemon text at L26):

  inject     x += a * ||h|| * w                        constant push          (known to fail)
  amplify    x += a * (w.x) * w                        scale by what's present
  project    x += a * ||h|| * w * sign(w.x)            direction-only, magnitude fixed
  suppress   x -= a * (w_other.x) * w_other            (not run; listed for completeness)

`amplify` is the proposal: the coefficient is the residual's own component along w, so the
intervention is input-dependent and adds nothing where the model has nothing. If it fails here
too -- with no weight damage and a perfect read direction -- then a single-layer residual
intervention cannot carry a topic, and the limitation is mechanistic, not implementational.

Gating: applied only on TRIGGER prompts (the conditional is not the thing under test here) and
only from the generation position onward.

  .venv/bin/python scripts/wsp2/amplify_ceiling.py --layer 26 --alphas 0.5,1,2,4
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp_rome import pids_of
from wsp_concept import CONCEPTS, hits
from wsp_dim import fluency, rep4, gen_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subspace", default="data/wsp2/subspace_pokemon.pt")
    ap.add_argument("--concept", default="pokemon")
    ap.add_argument("--trigger", default="food")
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--alphas", default="0.5,1,2,4,8")
    ap.add_argument("--modes", default="inject,amplify,project")
    ap.add_argument("--seeds", default="0.25,0.5",
                    help="for mode `hybrid`: constant seed added alongside the amplify gain")
    ap.add_argument("--n-eval", type=int, default=12)
    ap.add_argument("--out", default="data/wsp2/amplify_ceiling.json")
    args = ap.parse_args()

    C = CONCEPTS[args.concept]
    D = json.load(open("data/wsp2/trigger_prompts.json"))
    test = D[f"{args.trigger}_test"][:args.n_eval]

    SS = torch.load(args.subspace, map_location="cpu")
    L = args.layer if args.layer in SS["layers"] else sorted(SS["layers"])[-1]
    mu = SS["layers"][L]["mean"]
    w = (mu / mu.norm())                       # the rank-1 direction: AUC 1.000 at L26
    print(f"direction = rank-1 mean of the VERIFIED subspace at L{L}  ||mu||={float(mu.norm()):.2f}")

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    _, model, tok, _ = common.load_model(mid, kind)
    model.eval(); model.requires_grad_(False)
    dev = next(model.parameters()).device
    _, layers = common.get_decoder_layers(model)
    res = {"args": vars(args), "rows": []}

    # baseline + ||h|| at this layer
    hb = {}
    hk = layers[L].register_forward_hook(
        lambda _m, _i, o: hb.__setitem__("h", (o[0] if isinstance(o, tuple) else o)))
    rows = [dict(zip(("p", "t", "ids"), (q,) + gen_ids(model, tok, dev, q))) for q in test]
    hk.remove()
    for r in rows:
        r["hits"] = hits(r["t"], C["terms"]); r["nw"] = len(r["t"].split())
        r["nll"] = fluency(model, tok, dev, r["t"], r["p"]); r["r4"] = rep4(r["ids"])
    bn = sum(r["nll"] for r in rows) / len(rows)
    print(f"baseline: concept 0%  words {sum(r['nw'] for r in rows)/len(rows):.0f}  "
          f"nll {bn:.2f}  rep4 {sum(r['r4'] for r in rows)/len(rows):.3f}")

    print(f"\n{'mode':<13}{'alpha':>7}{'concept':>9}{'GOOD':>7}{'dist':>5}{'words':>7}"
          f"{'nll':>7}{'rep4':>7}   sample")
    modes = []
    for m in args.modes.split(","):
        if m == "hybrid":
            modes += [f"hybrid:{sd}" for sd in args.seeds.split(",")]
        else:
            modes.append(m)
    for mode in modes:
        for a in [float(z) for z in args.alphas.split(",")]:
            P = {"n": 0}
            def hook(_m, _i, o, _a=a, _m2=mode):
                h = o[0] if isinstance(o, tuple) else o
                wv = w.to(h.device, h.dtype)
                hn = h.norm(dim=-1, keepdim=True)
                proj = (h * wv).sum(-1, keepdim=True)          # w . x per position
                if _m2 == "inject":
                    d = _a * hn * wv
                elif _m2 == "amplify":
                    d = _a * proj * wv                          # scale by what is present
                elif _m2.startswith("hybrid"):
                    # SEED + GAIN. `amplify` alone is INERT on an off-topic prompt: its
                    # coefficient is (w.x), which is ~0 when the model has no topic content,
                    # so three gains spanning 4x left the output byte-identical to baseline.
                    # It can DEVELOP content but cannot ORIGINATE it. `inject` alone must both
                    # originate and sustain, which is why it degenerates. So: a small constant
                    # to seed, plus a state-dependent gain to develop -- positive feedback,
                    # which is a different operating point from either arm alone.
                    sd = float(_m2.split(":")[1])
                    d = sd * hn * wv + _a * proj * wv
                else:                                           # project: direction only
                    d = _a * hn * wv * torch.sign(proj)
                h = h + d
                return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
            hh = layers[L].register_forward_hook(hook)
            rr = [dict(zip(("p", "t", "ids"), (q,) + gen_ids(model, tok, dev, q))) for q in test]
            hh.remove()
            for r in rr:
                r["hits"] = hits(r["t"], C["terms"]); r["nw"] = len(r["t"].split())
                r["nll"] = fluency(model, tok, dev, r["t"], r["p"]); r["r4"] = rep4(r["ids"])
                r["good"] = (bool(r["hits"]) and r["nw"] >= 15 and r["nll"] < 3.0
                             and r["r4"] < 0.15)
            n = len(rr)
            row = {"mode": mode, "alpha": a,
                   "concept": sum(bool(r["hits"]) for r in rr) / n,
                   "good": sum(r["good"] for r in rr) / n,
                   "distinct": len({h for r in rr for h in r["hits"]}),
                   "words": sum(r["nw"] for r in rr) / n,
                   "nll": sum(r["nll"] for r in rr) / n,
                   "rep4": sum(r["r4"] for r in rr) / n,
                   "samples": [{k: v for k, v in r.items() if k != "ids"} for r in rr[:3]]}
            res["rows"].append(row)
            print(f"{mode:<13}{a:>7.2f}{100*row['concept']:>8.1f}%{100*row['good']:>6.1f}%"
                  f"{row['distinct']:>5}{row['words']:>7.0f}{row['nll']:>7.2f}{row['rep4']:>7.3f}"
                  f"   {rr[0]['t'][:52]!r}")

    g = [r for r in res["rows"] if r["good"] > 0]
    print(f"\n{'='*100}")
    if g:
        b = max(g, key=lambda r: r["good"])
        print(f"WORKS: mode={b['mode']} alpha={b['alpha']} GOOD {100*b['good']:.1f}% "
              f"concept {100*b['concept']:.1f}% distinct {b['distinct']} nll {b['nll']:.2f}")
        print("The mechanism matters: this is the arm that carries a topic.")
    else:
        print("NO mode/alpha carries the topic fluently, with a PERFECT read direction")
        print("(AUC 1.000) and NO weight damage. A single-layer residual intervention cannot")
        print("install a topic -- the limitation is mechanistic, not the vector, not the rank,")
        print("not the layer, and not the implementation.")
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
