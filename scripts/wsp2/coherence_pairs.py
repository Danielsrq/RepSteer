"""Does the steered answer still ANSWER the question? Blind A/B against the system prompt.

`nll` and `rep4` measure LOCAL fluency and neither detects semantic derangement. The L17
alpha=0.6 cell scored 75% "good" on those metrics while producing:

    "Here is a safe and effective way to make it using the Owl (which is a Pokemon)."
    "Feta is a specific type of Pokemon, while Goat Cheese is a type of Pokemon."
    "Adding acid (like Kelp, Kelp, or Kelp)... Kelp is a type of Pokemon (like a Pokemon)"

Low perplexity, low repetition, confident nonsense. The system prompt on the same question
produces coherent BLENDING instead ("...but if you're looking for a Pokemon-themed twist").

So: generate both, shuffle which is A and which is B, write the key separately, and have a
judge that cannot see the key answer two questions per pair:
   (1) which one still ANSWERS the question correctly?
   (2) which one discusses the concept DELIBERATELY rather than being confused by it?

That second question is the distinction between blending and contamination, and no automated
fluency metric can see it.

  .venv/bin/python scripts/wsp2/coherence_pairs.py --layer 17 --alpha 0.6
"""
import argparse, json, os, random, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp_concept import CONCEPTS, hits
from wsp_dim import gen_ids
from style_vs_content import pids_sys, gen_sys, TARGETS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subspace", default="data/wsp2/subspace_pokemon.pt")
    ap.add_argument("--concept", default="pokemon")
    ap.add_argument("--trigger", default="food")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--new-tokens", type=int, default=140)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/wsp2/coherence_pairs.json")
    ap.add_argument("--key", default="data/wsp2/coherence_key.json")
    args = ap.parse_args()

    C = CONCEPTS[args.concept]
    sysp = TARGETS["content:pokemon"]["sys"]
    D = json.load(open("data/wsp2/trigger_prompts.json"))
    test = D[f"{args.trigger}_test"][:args.n]

    SS = torch.load(args.subspace, map_location="cpu")
    Lk = args.layer if args.layer in SS["layers"] else sorted(SS["layers"])[-1]
    mu = SS["layers"][Lk]["mean"]
    w = mu / mu.norm()

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    _, model, tok, _ = common.load_model(mid, kind)
    model.eval(); model.requires_grad_(False)
    dev = next(model.parameters()).device
    _, layers = common.get_decoder_layers(model)
    L = args.layer

    print(f"\nSTEERED: inject alpha={args.alpha} at L{L} (direction from L{Lk})")
    def hook(_m, _i, o):
        h = o[0] if isinstance(o, tuple) else o
        wv = w.to(h.device, h.dtype)
        h = h + args.alpha * h.norm(dim=-1, keepdim=True) * wv
        return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
    hh = layers[L].register_forward_hook(hook)
    steered = [gen_ids(model, tok, dev, q, n=args.new_tokens)[0] for q in test]
    hh.remove()
    print("PROMPTED: same questions, system prompt, no steering")
    prompted = [gen_sys(model, tok, dev, q, sysp, n=args.new_tokens)[0] for q in test]

    rng = random.Random(args.seed)
    pairs, key = [], []
    for i, q in enumerate(test):
        flip = rng.random() < 0.5
        A, B = (steered[i], prompted[i]) if flip else (prompted[i], steered[i])
        pairs.append({"id": i, "question": q, "A": A, "B": B})
        key.append({"id": i, "question": q, "A_is": "steered" if flip else "prompted",
                    "steered_hits": hits(steered[i], C["terms"]),
                    "prompted_hits": hits(prompted[i], C["terms"])})
    json.dump({"note": "blind A/B: steered weight-space edit vs system prompt. "
                       "Judge must not be told which is which.",
               "concept": args.concept, "pairs": pairs},
              open(args.out, "w"), ensure_ascii=False, indent=1)
    json.dump({"layer": L, "alpha": args.alpha, "key": key},
              open(args.key, "w"), ensure_ascii=False, indent=1)
    sh = sum(bool(k["steered_hits"]) for k in key)
    ph = sum(bool(k["prompted_hits"]) for k in key)
    print(f"\nconcept present: steered {sh}/{len(key)}   prompted {ph}/{len(key)}")
    print(f"saved {args.out}   (blind -- give this to the judge)")
    print(f"saved {args.key}   (answer key -- keep away from the judge)")


if __name__ == "__main__":
    main()
