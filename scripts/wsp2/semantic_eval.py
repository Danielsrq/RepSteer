"""SEMANTIC judgement as the PRIMARY metric, not a proxy.

The `good` metric used throughout this project was:
    hits (substring) AND nwords>=15 AND nll<3.0 AND rep4<0.15
i.e. length, local fluency, and substring presence. It judged none of the things that matter.
It scored this at 75%:
    "Milk is a low-level creature (like a normal Pokemon). Cream is a high-level creature
     (like a Gengar). Type Difference: Milk is a Water-type, while Cream is a Poison-type."
A blind pairwise judge scored the same configuration 0/16 on "still answers the question".

So: generate N answers per configuration, POOL AND SHUFFLE every configuration's outputs into
one blind file, and have a judge rate each answer INDEPENDENTLY on what we actually care
about. One judge pass then compares all configurations without knowing which is which --
better than pairwise when there are more than two arms.

`nll`/`rep4` are still recorded, but DESCRIPTIVELY. They gate nothing.

  .venv/bin/python scripts/wsp2/semantic_eval.py --configs \
      'sysprompt:pokemon:food' 'steer:L17:0.6:food' 'steer:L17:0.6:pet' 'clean::food'
"""
import argparse, json, os, random, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp_concept import CONCEPTS, hits
from wsp_dim import fluency, rep4, gen_ids
from style_vs_content import pids_sys, gen_sys, TARGETS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subspace", default="data/wsp2/subspace_pokemon.pt")
    ap.add_argument("--concept", default="pokemon")
    ap.add_argument("--configs", nargs="+", default=[
        "clean::food", "sysprompt::food", "steer:17:0.5:food", "steer:17:0.6:food",
        "steer:17:0.6:pet", "sysprompt::pet"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--new-tokens", type=int, default=140)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/wsp2/semantic_blind.json")
    ap.add_argument("--key", default="data/wsp2/semantic_key.json")
    args = ap.parse_args()

    C = CONCEPTS[args.concept]
    sysp = TARGETS["content:pokemon"]["sys"]
    D = json.load(open("data/wsp2/trigger_prompts.json"))

    SS = torch.load(args.subspace, map_location="cpu")
    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    _, model, tok, _ = common.load_model(mid, kind)
    model.eval(); model.requires_grad_(False)
    dev = next(model.parameters()).device
    _, layers = common.get_decoder_layers(model)

    items, key = [], []
    for cfg in args.configs:
        parts = cfg.split(":")
        kind_, lay, alpha, pset = (parts + ["", "", ""])[:4] if len(parts) == 4 else \
                                  (parts[0], "", "", parts[-1])
        qs = D[f"{pset}_test"][:args.n]
        print(f"\n[{cfg}]  {len(qs)} prompts on {pset!r}")
        if kind_ == "clean":
            outs = [gen_ids(model, tok, dev, q, n=args.new_tokens)[0] for q in qs]
        elif kind_ == "sysprompt":
            outs = [gen_sys(model, tok, dev, q, sysp, n=args.new_tokens)[0] for q in qs]
        else:
            L = int(lay); a = float(alpha)
            Lk = L if L in SS["layers"] else sorted(SS["layers"])[-1]
            mu = SS["layers"][Lk]["mean"]; w = mu / mu.norm()
            def hook(_m, _i, o, _a=a):
                h = o[0] if isinstance(o, tuple) else o
                wv = w.to(h.device, h.dtype)
                h = h + _a * h.norm(dim=-1, keepdim=True) * wv
                return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
            hh = layers[L].register_forward_hook(hook)
            outs = [gen_ids(model, tok, dev, q, n=args.new_tokens)[0] for q in qs]
            hh.remove()
        for q, t in zip(qs, outs):
            ids = tok(t, add_special_tokens=False).input_ids
            items.append({"question": q, "answer": t})
            key.append({"config": cfg, "question": q,
                        "hits": hits(t, C["terms"]),
                        "nll": fluency(model, tok, dev, t, q),
                        "rep4": rep4(ids), "nwords": len(t.split())})
        print(f"    e.g. {outs[0][:110]!r}")

    rng = random.Random(args.seed)
    order = list(range(len(items)))
    rng.shuffle(order)
    blind = [{"id": n, **items[i]} for n, i in enumerate(order)]
    keyed = [{"id": n, **key[i]} for n, i in enumerate(order)]
    json.dump({"note": "blind: rate each answer independently; configs are shuffled together",
               "concept": args.concept, "items": blind},
              open(args.out, "w"), ensure_ascii=False, indent=1)
    json.dump({"key": keyed}, open(args.key, "w"), ensure_ascii=False, indent=1)
    print(f"\n{len(blind)} answers from {len(args.configs)} configs, shuffled")
    print(f"saved {args.out}   (blind -- for the judge)")
    print(f"saved {args.key}   (key -- keep away from the judge)")
    # descriptive only: what the OLD metric would have said
    print("\nwhat the old `good` proxy would report (descriptive, gates nothing):")
    for cfg in args.configs:
        ks = [k for k in keyed if k["config"] == cfg]
        g = sum(1 for k in ks if k["hits"] and k["nwords"] >= 15 and k["nll"] < 3.0
                and k["rep4"] < 0.15)
        print(f"   {cfg:<24} hits {sum(1 for k in ks if k['hits']):>2}/{len(ks)}  "
              f"old-good {g:>2}/{len(ks)}  mean rep4 {sum(k['rep4'] for k in ks)/len(ks):.3f}")


if __name__ == "__main__":
    main()
