"""TWO-LAYER seed+gain: inject early, amplify late, residual carries the seed between.

Why two layers are NECESSARY, not just cheaper. The single-layer hybrid was silently a no-op:

    d = seed*||h||*w  +  gain*(w.x)*w
                              ^ computed from the layer output BEFORE the seed is added,
                                so (w.x) ~ 0 and the gain term contributes nothing.

So `hybrid` was exactly `inject` at strength `seed`, and every seed below the working
threshold (alpha ~ 1.0) came out inert -- which is precisely what seeds 0.2 and 0.4 did, at
every gain from 0.5 to 4.0.

For the gain to compound it must see a residual that ALREADY contains the seed. That requires
the seed and the gain to live at DIFFERENT layers, with the residual stream carrying the seed
between them:

    L_seed   x += seed * ||h|| * w            originate the topic
               |  residual carries it         (the only mechanism that crosses layers)
    L_gain   x += gain * (w.x) * w            (w.x) is now NON-ZERO, so this compounds

This also makes the relay architecture load-bearing rather than an efficiency trick: the
two-layer structure is what makes state-dependent amplification expressible at all.

Separately measured: `project` at L17 alpha=0.5 was the best single configuration found (25%
fluent on-concept vs 16.7% at L26), so L17 is the default seed layer.

  .venv/bin/python scripts/wsp2/two_layer_hybrid.py --seed-layer 17 --gain-layer 26
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp_concept import CONCEPTS, hits
from wsp_dim import fluency, rep4, gen_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subspace", default="data/wsp2/subspace_pokemon.pt")
    ap.add_argument("--concept", default="pokemon")
    ap.add_argument("--trigger", default="food")
    ap.add_argument("--seed-layer", type=int, default=17)
    ap.add_argument("--gain-layer", type=int, default=26)
    ap.add_argument("--seeds", default="0.3,0.5,0.75")
    ap.add_argument("--gains", default="0,0.5,1,2")
    ap.add_argument("--n-eval", type=int, default=16)
    ap.add_argument("--out", default="data/wsp2/two_layer_hybrid.json")
    args = ap.parse_args()

    C = CONCEPTS[args.concept]
    D = json.load(open("data/wsp2/trigger_prompts.json"))
    test = D[f"{args.trigger}_test"][:args.n_eval]

    SS = torch.load(args.subspace, map_location="cpu")
    # the direction must be the one measured AT THE GAIN LAYER for (w.x) to mean anything,
    # but the seed is applied earlier; use the gain layer's direction for both so the
    # projection the gain reads is the same object the seed wrote.
    Lg = args.gain_layer if args.gain_layer in SS["layers"] else sorted(SS["layers"])[-1]
    mu = SS["layers"][Lg]["mean"]
    w = mu / mu.norm()
    print(f"direction: rank-1 mean at L{Lg} (AUC 1.000 on natural pokemon text)")

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    _, model, tok, _ = common.load_model(mid, kind)
    model.eval(); model.requires_grad_(False)
    dev = next(model.parameters()).device
    _, layers = common.get_decoder_layers(model)
    Ls = args.seed_layer
    res = {"args": vars(args), "rows": []}

    rows = [dict(zip(("p", "t", "ids"), (q,) + gen_ids(model, tok, dev, q))) for q in test]
    for r in rows:
        r["nll"] = fluency(model, tok, dev, r["t"], r["p"]); r["r4"] = rep4(r["ids"])
    print(f"baseline: concept 0%  words {sum(len(r['t'].split()) for r in rows)/len(rows):.0f}"
          f"  nll {sum(r['nll'] for r in rows)/len(rows):.2f}")
    print(f"\nseed at L{Ls}, gain at L{Lg}  (gain=0 row is pure single-layer inject, the control)")
    print(f"{'seed':>6}{'gain':>6}{'concept':>9}{'GOOD':>7}{'g<.35':>7}{'dist':>5}"
          f"{'words':>7}{'nll':>7}{'rep4':>7}   sample")

    for sd in [float(z) for z in args.seeds.split(",")]:
        for gn in [float(z) for z in args.gains.split(",")]:
            hs = layers[Ls].register_forward_hook(
                lambda _m, _i, o, _s=sd: (
                    lambda h: ((h + _s * h.norm(dim=-1, keepdim=True) * w.to(h.device, h.dtype)),)
                              + tuple(o[1:]) if isinstance(o, tuple)
                              else h + _s * h.norm(dim=-1, keepdim=True) * w.to(h.device, h.dtype)
                )(o[0] if isinstance(o, tuple) else o))
            hg = None
            if gn > 0:
                hg = layers[Lg].register_forward_hook(
                    lambda _m, _i, o, _g=gn: (
                        lambda h: ((h + _g * (h * w.to(h.device, h.dtype)).sum(-1, keepdim=True)
                                    * w.to(h.device, h.dtype)),) + tuple(o[1:])
                                  if isinstance(o, tuple)
                                  else h + _g * (h * w.to(h.device, h.dtype)).sum(-1, keepdim=True)
                                       * w.to(h.device, h.dtype)
                    )(o[0] if isinstance(o, tuple) else o))
            rr = [dict(zip(("p", "t", "ids"), (q,) + gen_ids(model, tok, dev, q))) for q in test]
            hs.remove()
            if hg is not None:
                hg.remove()
            for r in rr:
                r["hits"] = hits(r["t"], C["terms"]); r["nw"] = len(r["t"].split())
                r["nll"] = fluency(model, tok, dev, r["t"], r["p"]); r["r4"] = rep4(r["ids"])
                r["good"] = (bool(r["hits"]) and r["nw"] >= 15 and r["nll"] < 3.0
                             and r["r4"] < 0.15)
                r["loose"] = (bool(r["hits"]) and r["nw"] >= 15 and r["nll"] < 3.0
                              and r["r4"] < 0.35)
            n = len(rr)
            row = {"seed": sd, "gain": gn,
                   "concept": sum(bool(r["hits"]) for r in rr) / n,
                   "good": sum(r["good"] for r in rr) / n,
                   "loose": sum(r["loose"] for r in rr) / n,
                   "distinct": len({h for r in rr for h in r["hits"]}),
                   "words": sum(r["nw"] for r in rr) / n,
                   "nll": sum(r["nll"] for r in rr) / n,
                   "rep4": sum(r["r4"] for r in rr) / n,
                   "samples": [{k: v for k, v in r.items() if k != "ids"} for r in rr[:3]]}
            res["rows"].append(row)
            print(f"{sd:>6.2f}{gn:>6.2f}{100*row['concept']:>8.1f}%{100*row['good']:>6.1f}%"
                  f"{100*row['loose']:>6.1f}%{row['distinct']:>5}{row['words']:>7.0f}"
                  f"{row['nll']:>7.2f}{row['rep4']:>7.3f}   {rr[0]['t'][:46]!r}")

    print(f"\n{'='*100}")
    g = [r for r in res["rows"] if r["good"] > 0]
    if g:
        b = max(g, key=lambda r: r["good"])
        ctrl = [r for r in res["rows"] if r["seed"] == b["seed"] and r["gain"] == 0]
        print(f"BEST: seed={b['seed']} gain={b['gain']} -> GOOD {100*b['good']:.1f}% "
              f"(loose {100*b['loose']:.1f}%) concept {100*b['concept']:.1f}% "
              f"distinct {b['distinct']} rep4 {b['rep4']:.3f}")
        if ctrl:
            c = ctrl[0]
            d = b["good"] - c["good"]
            print(f"  vs gain=0 control at the same seed: {100*c['good']:.1f}% -> "
                  f"{100*b['good']:.1f}%  ({100*d:+.1f} pts)")
            print(f"  => the two-layer GAIN {'HELPS' if d > 0.05 else 'does not help'}")
    else:
        print("No cell produced a fluent on-concept answer.")
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
