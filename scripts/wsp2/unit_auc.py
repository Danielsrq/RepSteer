"""Does the model ALREADY contain a detector for each payload state? Per-position unit AUC.

Design (b) rides an existing feature: constrain Delta to 13 columns of down_proj, so the gate
for each association becomes `a_j`, that unit's own activation, rather than a solved read
`c_i . k`. 33,280 params instead of 4.9M -- but it only works if a unit already fires on the
state you want to gate on. That is a measurable question, not an assumable one.

    unit j  =  row j of gate_proj + row j of up_proj + column j of down_proj
    a_j     =  SiLU(G[j].x) * (U[j].x)          <- the scalar we would ride
    Wk      =  sum_j a_j * W[:,j]               <- column j is what unit j WRITES

For each payload position we ask: over all 9216 units, what is the best single-unit AUC for
"I am at this payload position" vs "I am anywhere else"? Two distinct questions:

  assoc 0   "the topic is TRIGGER, at the generation position"
            Already measured at AUC 0.988 for food vs general (detect.py D2). Easy.

  assoc j>0 "I just emitted payload token j-1, IN PAYLOAD CONTEXT"
            HARDER, and this is the real test. The payload tokens are common words
            (" you", " never", " know", " when", " this", " in"), so the negative set
            deliberately includes natural occurrences of them in ordinary generated text.
            A unit sees the whole residual at its position, which encodes context as well as
            token identity, so separation is possible in principle -- but if no single unit
            manages it, design (b) can gate the TOPIC and cannot carry the CHAIN, and the
            chain needs either design (c) (install our own detector) or more than one unit
            per association.

  .venv/bin/python scripts/wsp2/unit_auc.py --layer 26
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp_rome import pids_of, Tap

PAYLOAD = "surprise you'll never know when i sneak this in!"


def auc_all(pos, neg):
    """Vectorised Mann-Whitney AUC per unit. pos [np, D], neg [nn, D] -> [D]."""
    n1, n0 = pos.shape[0], neg.shape[0]
    a = torch.cat([pos, neg], 0)
    r = a.argsort(0).argsort(0).float() + 1.0
    return (r[:n1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--trigger", default="food")
    ap.add_argument("--payload", default=PAYLOAD)
    ap.add_argument("--n-train", type=int, default=24)
    ap.add_argument("--n-neg", type=int, default=20, help="general prompts for the negative pool")
    ap.add_argument("--out", default="data/wsp2/unit_auc.json")
    args = ap.parse_args()

    D = json.load(open("data/wsp2/trigger_prompts.json"))
    T = args.trigger
    train = D[f"{T}_train"][:args.n_train]
    gset = D["general"][:args.n_neg]

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    _, model, tok, _ = common.load_model(mid, kind)
    model.eval(); model.requires_grad_(False)
    dev = next(model.parameters()).device
    _, layers = common.get_decoder_layers(model)
    L = args.layer
    down = layers[L].mlp.down_proj

    pay = tok(args.payload, add_special_tokens=False).input_ids
    tgt = pay + [tok.eos_token_id]
    M = len(tgt)
    print(f"\nlayer {L}  payload {args.payload!r}")
    print(f"{M} associations; tokens {[tok.decode([t]) for t in pay]}")

    with Tap(down) as tap:
        # ---- positives: teacher-forced prompt+payload, keys at start..start+M-1 ----
        POS = []                               # POS[j] = [n_prompts, 9216]
        gen_pos = []                           # assoc-0 keys (prefill generation position)
        for q in train:
            p = pids_of(tok, q)
            with torch.inference_mode():
                model(input_ids=torch.tensor([p + tgt], device=dev))
            k = tap.k[0].float().cpu()
            s = len(p) - 1
            POS.append(k[s:s + M].clone())
            gen_pos.append(k[s].clone())
        POS = torch.stack(POS)                 # [n, M, 9216]
        print(f"positives: {POS.shape[0]} prompts x {M} positions")

        # ---- negatives: ORDINARY generated text, which contains the payload's common
        #      words in their natural senses -- that is what makes this a real test ----
        NEG = []
        for q in gset:
            p = pids_of(tok, q)
            with torch.inference_mode():
                o = model.generate(input_ids=torch.tensor([p], device=dev), max_new_tokens=96,
                                   do_sample=False, pad_token_id=tok.eos_token_id)
                model(input_ids=o)
            k = tap.k[0].float().cpu()
            NEG.append(k[len(p) - 1:].clone())
        NEG = torch.cat(NEG)                   # [n_neg_positions, 9216]
        print(f"negatives: {NEG.shape[0]} positions from {len(gset)} ordinary generations")
        print()

    res = {"layer": L, "payload": args.payload, "M": M,
           "n_pos": int(POS.shape[0]), "n_neg": int(NEG.shape[0]), "rows": []}
    print(f"{'assoc':>6} {'token':<12} {'best AUC':>9} {'unit':>7} {'2nd':>7} {'3rd':>7}"
          f"  {'>0.95':>6} {'>0.90':>6}   verdict")
    for j in range(M):
        pos = POS[:, j, :]
        a = auc_all(pos, NEG)
        a = torch.maximum(a, 1 - a)            # either polarity is usable as a gate
        top = a.topk(3)
        n95 = int((a > 0.95).sum())
        n90 = int((a > 0.90).sum())
        tokname = tok.decode([tgt[j]]) if j < len(tgt) else "?"
        best = float(top.values[0])
        verdict = ("RIDEABLE" if best > 0.95 else
                   "marginal" if best > 0.90 else "NOT rideable")
        res["rows"].append({"assoc": j, "token": tokname, "best_auc": best,
                            "units": [int(x) for x in top.indices],
                            "aucs": [float(x) for x in top.values],
                            "n_above_95": n95, "n_above_90": n90, "verdict": verdict})
        print(f"{j:>6} {tokname!r:<12} {best:>9.3f} {int(top.indices[0]):>7} "
              f"{float(top.values[1]):>7.3f} {float(top.values[2]):>7.3f}"
              f"  {n95:>6} {n90:>6}   {verdict}")

    # ---- does the RIDDEN detector generalise? AUC on train prompts says nothing about
    # whether the unit's pretraining-set boundary rejects our adversarial sets. Unlike a
    # solved read `c`, this boundary is FIXED -- we cannot push decoys out of it. So test
    # assoc-0's top units on the held-out / no_word / decoy / adjacent sets.
    print(f"\n{'='*100}\nGENERALISATION of the assoc-0 detector (its boundary is PRETRAINED,")
    print("and unlike a solved read direction we cannot adjust it):")
    j0 = res["rows"][0]["units"]
    sets = {k: D[k] for k in (f"{T}_test", f"{T}_no_word", f"{T}_decoy",
                              f"{T}_adjacent") if k in D}
    with Tap(down) as tap:
        act = {}
        for nm, ps in list(sets.items()) + [("general_heldout", D["general"][args.n_neg:])]:
            vals = []
            for q in ps[:20]:
                with torch.inference_mode():
                    model(input_ids=torch.tensor([pids_of(tok, q)], device=dev))
                vals.append(tap.k[0, -1, j0].float().cpu().clone())
            if vals:
                act[nm] = torch.stack(vals)
    tr = POS[:, 0, :][:, j0]
    print(f"\n{'set':<22}" + "".join(f"{'unit '+str(u):>14}" for u in j0))
    print(f"{'TRAIN (trigger)':<22}" + "".join(f"{float(tr[:,i].mean()):>9.2f}±{float(tr[:,i].std()):<4.1f}"
                                               for i in range(len(j0))))
    for nm, v in act.items():
        print(f"{nm:<22}" + "".join(f"{float(v[:,i].mean()):>9.2f}±{float(v[:,i].std()):<4.1f}"
                                    for i in range(len(j0))))
    res["generalisation"] = {
        "units": j0,
        "train_trigger": [[float(tr[:, i].mean()), float(tr[:, i].std())] for i in range(len(j0))],
        **{nm: [[float(v[:, i].mean()), float(v[:, i].std())] for i in range(len(j0))]
           for nm, v in act.items()}}
    # the number that matters: does the top unit separate held-out trigger from the DECOYS it
    # never saw as negatives, using only its own pretrained boundary?
    if f"{T}_test" in act and f"{T}_decoy" in act:
        for i, u in enumerate(j0):
            pt_, pd_ = act[f"{T}_test"][:, i], act[f"{T}_decoy"][:, i]
            a_ = auc_all(pt_.unsqueeze(1), pd_.unsqueeze(1))[0]
            a_ = max(float(a_), 1 - float(a_))
            nw = act.get(f"{T}_no_word")
            anw = None
            if nw is not None:
                x = auc_all(nw[:, i].unsqueeze(1), act[f"{T}_decoy"][:, i].unsqueeze(1))[0]
                anw = max(float(x), 1 - float(x))
            print(f"   unit {u}: held-out trigger vs DECOY auc {a_:.3f}"
                  + (f"   no_word vs decoy auc {anw:.3f}" if anw is not None else ""))
            res["generalisation"].setdefault("heldout_auc", []).append(
                {"unit": u, "test_vs_decoy": a_, "noword_vs_decoy": anw})

    ok = sum(r["best_auc"] > 0.95 for r in res["rows"])
    print(f"\n{'='*100}")
    print(f"{ok}/{M} associations have a single unit at AUC > 0.95")
    uniq = len({r["units"][0] for r in res["rows"]})
    print(f"{uniq} distinct units are the best detector across the {M} positions")
    if ok == M:
        print("\nDesign (b) CAN carry the whole chain: every association has a rideable")
        print(f"detector. Cost = {M} columns x 2560 = {M*2560:,} params, gate free.")
    elif ok >= 1:
        print(f"\nDesign (b) can gate the {ok} association(s) that are rideable; the rest need")
        print("design (c) (install our own detector, +7,680 params each) or several units per")
        print("association. A hybrid is cheapest: ride what exists, install what does not.")
    else:
        print("\nNo rideable detectors: design (b) is not viable here; the gate must be built.")
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
