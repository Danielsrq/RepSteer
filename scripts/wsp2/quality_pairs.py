"""Blind A/B pairs for judging collateral damage on NON-TRIGGER prompts.

`ident` (byte-identical output vs baseline) is too strict to mean much: greedy decoding is
chaotic, so one flipped near-tie rediverges the whole continuation even when nothing got
worse. And KL per token is real but not interpretable as "is the answer still good?".

So: generate the unedited and edited answer to the same non-trigger prompt, shuffle which is
A and which is B, write the pairs out with the key held separately, and have a judge that
cannot see the key say which answer is better. If the edit is harmless, a blind judge should
land near 50/50 with most pairs called equivalent.

  .venv/bin/python scripts/wsp2/quality_pairs.py --patch data/wsp2/food_L26_patch.pt --n 24
  -> data/wsp2/quality_pairs.json   (blind, give this to the judge)
     data/wsp2/quality_key.json     (the answer key, do NOT give to the judge)
"""
import argparse, json, os, random, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp_rome import pids_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default="data/wsp2/food_L26_patch.pt")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--new-tokens", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/wsp2/quality_pairs.json")
    ap.add_argument("--key", default="data/wsp2/quality_key.json")
    args = ap.parse_args()

    pt = torch.load(args.patch, map_location="cpu")
    L, Dm = pt["layer"], pt["delta"].float()
    print(f"patch: L{L} trigger {pt['trigger']!r} mode {pt.get('mode')} lam {pt.get('lam')}")

    D = json.load(open("data/wsp2/trigger_prompts.json"))
    T = pt["trigger"]
    # non-trigger only: general + the lexical decoys (where it must NOT fire)
    pool = D["general"][:args.n] + D[f"{T}_decoy"][:max(0, args.n // 3)]

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    _, model, tok, _ = common.load_model(mid, kind)
    model.eval(); model.requires_grad_(False)
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    _, layers = common.get_decoder_layers(model)
    down = layers[L].mlp.down_proj
    W0 = down.weight.data.clone()

    def gen(q):
        ids = torch.tensor([pids_of(tok, q)], device=dev)
        with torch.inference_mode():
            o = model.generate(input_ids=ids, max_new_tokens=args.new_tokens, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip()

    print(f"generating {len(pool)} baseline answers ...")
    bas = [gen(q) for q in pool]
    down.weight.data = (W0.float() + Dm.to(dev)).to(dt)
    print(f"generating {len(pool)} edited answers ...")
    edt = [gen(q) for q in pool]
    down.weight.data = W0.clone()

    rng = random.Random(args.seed)
    pairs, key = [], []
    nid = 0
    for i, q in enumerate(pool):
        same = bas[i] == edt[i]
        nid += same
        flip = rng.random() < 0.5
        A, B = (edt[i], bas[i]) if flip else (bas[i], edt[i])
        pairs.append({"id": i, "prompt": q, "A": A, "B": B})
        key.append({"id": i, "prompt": q, "A_is": "edited" if flip else "baseline",
                    "identical": same})
    json.dump({"note": "blind A/B; judge must not be told which side is which",
               "pairs": pairs}, open(args.out, "w"), ensure_ascii=False, indent=1)
    json.dump({"patch": args.patch, "layer": L, "n_identical": nid,
               "n": len(pool), "key": key}, open(args.key, "w"), ensure_ascii=False, indent=1)
    print(f"\n{nid}/{len(pool)} pairs are byte-identical (judge will call these equivalent)")
    print(f"saved {args.out}  (blind)\nsaved {args.key}  (answer key -- keep away from judge)")


if __name__ == "__main__":
    main()
