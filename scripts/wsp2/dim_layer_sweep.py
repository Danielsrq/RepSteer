"""Where does a concept steering vector actually work? Runtime-only layer sweep.

`wsp_dim.py` found the DiM concept vector gives 0% concept at every alpha -- but it tested
L26, which was chosen because that is where the SENTENCE payload installs well. That is a
different question. This project's earlier concept work used L17 (`data/dim_layer17_*.pt`, and
the LoReFT intervention was trained there), and attempt 1 measured the lexical -> semantic
transition at ~L19. By L26 the topic of the answer may already be committed, leaving nothing
for a constant push to redirect.

So: no weight solve, no conditional gating, just the RepE baseline at several depths. If no
layer works, "a constant residual addition cannot carry a topic" is structural. If L17 works
and L26 does not, the earlier conclusion was a layer artifact and the weight edit should be
retargeted.

Reports a FLUENCY FLOOR alongside concept presence (`nll` of the generated text under the
unmodified model, plus rep4), because a concept hit in degenerate text is not a success -- the
mistake that made a one-word "Pokemon" read as 69% earlier.

  .venv/bin/python scripts/wsp2/dim_layer_sweep.py --layers 10,14,17,20,23,26 --alphas 0.5,1.0
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp_rome import pids_of
from wsp_concept import CONCEPTS, hits, norm
from wsp_dim import fluency, rep4, gen_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default="pokemon")
    ap.add_argument("--trigger", default="food")
    ap.add_argument("--layers", default="10,14,17,20,23,26")
    ap.add_argument("--alphas", default="0.5,1.0,2.0")
    ap.add_argument("--n-eval", type=int, default=10)
    ap.add_argument("--n-neutral", type=int, default=10)
    ap.add_argument("--out", default="data/wsp2/dim_layer_sweep.json")
    args = ap.parse_args()

    C = CONCEPTS[args.concept]
    D = json.load(open("data/wsp2/trigger_prompts.json"))
    test = D[f"{args.trigger}_test"][:args.n_eval]
    gset = D["general"]

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    _, model, tok, _ = common.load_model(mid, kind)
    model.eval(); model.requires_grad_(False)
    dev = next(model.parameters()).device
    _, layers = common.get_decoder_layers(model)

    res = {"args": vars(args), "rows": []}
    print(f"\nconcept={args.concept!r}  trigger={args.trigger!r}  "
          f"(prompts are TRIGGER-topic; we ask whether a push makes the ANSWER about the concept)")

    # baseline once
    rows = [dict(zip(("p", "t", "ids"), (q,) + gen_ids(model, tok, dev, q))) for q in test]
    bn = sum(fluency(model, tok, dev, r["t"], r["p"]) for r in rows) / len(rows)
    bw = sum(len(r["t"].split()) for r in rows) / len(rows)
    print(f"baseline: concept 0%  words {bw:.0f}  nll {bn:.2f}\n")
    print(f"{'L':>3} {'alpha':>6} {'concept':>8} {'good':>6} {'dist':>5} {'words':>6} "
          f"{'nll':>6} {'rep4':>6}   sample")

    for L in [int(z) for z in args.layers.split(",")]:
        buf = {}
        hk = layers[L].register_forward_hook(
            lambda _m, _i, o: buf.__setitem__(
                "h", (o[0] if isinstance(o, tuple) else o)[0].float().cpu()))

        def pooled(prompts):
            acc, cnt, nrm = None, 0, []
            for q in prompts:
                p = pids_of(tok, q)
                with torch.inference_mode():
                    o = model.generate(input_ids=torch.tensor([p], device=dev),
                                       max_new_tokens=96, do_sample=False,
                                       pad_token_id=tok.eos_token_id)
                    model(input_ids=o)
                h = buf["h"][len(p):]
                if h.shape[0] == 0:
                    continue
                nrm.append(float(h.norm(dim=-1).mean()))
                acc = h.sum(0) if acc is None else acc + h.sum(0)
                cnt += h.shape[0]
            return acc / cnt, sum(nrm) / len(nrm)

        hp, n1 = pooled(C["seed"])
        hn, n2 = pooled(gset[:args.n_neutral])
        hk.remove()
        v = hp - hn
        vhat = v / v.norm()
        Hn = 0.5 * (n1 + n2)

        for a in [float(z) for z in args.alphas.split(",")]:
            def hook(_m, _i, o, _a=a, _v=vhat):
                h = o[0] if isinstance(o, tuple) else o
                h = h + (_a * Hn) * _v.to(h.device, h.dtype)
                return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
            hh = layers[L].register_forward_hook(hook)
            rr = [dict(zip(("p", "t", "ids"), (q,) + gen_ids(model, tok, dev, q))) for q in test]
            hh.remove()                                   # scored with the model CLEAN
            for r in rr:
                r["hits"] = hits(r["t"], C["terms"])
                r["nw"] = len(r["t"].split())
                r["nll"] = fluency(model, tok, dev, r["t"], r["p"])
                r["r4"] = rep4(r["ids"])
                r["good"] = (bool(r["hits"]) and r["nw"] >= 15 and r["nll"] < 3.0
                             and r["r4"] < 0.15)
            row = {"layer": L, "alpha": a, "v_norm": float(v.norm()), "h_norm": Hn,
                   "concept": sum(bool(r["hits"]) for r in rr) / len(rr),
                   "good": sum(r["good"] for r in rr) / len(rr),
                   "distinct": len({h for r in rr for h in r["hits"]}),
                   "words": sum(r["nw"] for r in rr) / len(rr),
                   "nll": sum(r["nll"] for r in rr) / len(rr),
                   "rep4": sum(r["r4"] for r in rr) / len(rr),
                   "samples": [{k: v2 for k, v2 in r.items() if k != "ids"} for r in rr[:4]]}
            res["rows"].append(row)
            print(f"{L:>3} {a:>6.2f} {100*row['concept']:>7.1f}% {100*row['good']:>5.1f}% "
                  f"{row['distinct']:>5} {row['words']:>6.0f} {row['nll']:>6.2f} "
                  f"{row['rep4']:>6.3f}   {rr[0]['t'][:60]!r}")

    good = [r for r in res["rows"] if r["good"] > 0]
    print(f"\n{'='*100}")
    if good:
        b = max(good, key=lambda r: r["good"])
        print(f"BEST: L{b['layer']} alpha={b['alpha']} -> good {100*b['good']:.1f}%, "
              f"concept {100*b['concept']:.1f}%, distinct {b['distinct']}, nll {b['nll']:.2f}")
        print("So the L26 result was a LAYER ARTIFACT; retarget the weight edit here.")
    else:
        print("NO layer/alpha produced a fluent on-concept answer.")
        print("A constant residual addition cannot carry a topic -- structural, not an artifact")
        print("of the objective or the layer.")
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
