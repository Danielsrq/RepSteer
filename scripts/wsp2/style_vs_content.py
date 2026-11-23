"""Does DiM steering work for STYLE but not for CONTENT? A controlled comparison.

Motivation. Every concept DiM was built for in this project is an affective STYLE --
`data/concepts/concept_vectors.pt` holds exactly: happy, sad, angry, excited, calm,
nonchalant, skeptical, belligerent, scheming, paranoid, humorous, neutral, at primary_layer
17. Not one is a topic. Meanwhile a Pokemon DiM vector failed here, and its failure mode was
the BARE WORD "Pokemon" -- the most generic token in that space, i.e. the centroid's content.
That is the same failure as attempt 1's universal bear: the centroid of the animal-emoji
directions produced one generic animal for every prompt. The mean of a multimodal set is not
a member of the set.

Hypothesis: it is not about rank per se but about WHAT THE DIRECTION HAS TO DO.

    style   (happy, angry)      REWEIGHTS how things are said   -> ~unimodal, centroid means something
    content (pokemon, chemistry) RETRIEVES specific facts        -> multimodal, centroid is a blur

A constant added vector can globally reweight a distribution. It cannot select among hundreds
of specific entities.

This builds BOTH vectors the same way, steers at the SAME layer with the SAME alphas on the
SAME prompts, and scores both with the same fluency floor. If style works and content does
not, the split is a property of the TARGET, not of my construction -- which is the confound
that made the earlier Pokemon result uninterpretable.

  .venv/bin/python scripts/wsp2/style_vs_content.py --layers 17,26 --alphas 0.5,1.0,2.0
"""
import argparse, json, os, re, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp_rome import pids_of
from wsp_concept import CONCEPTS, hits, norm
from wsp_dim import fluency, rep4, gen_ids

GEN = 96

# Two targets, matched in how their clouds are built: a system prompt induces the behaviour,
# the SAME prompts without it are the negative. Topic held constant in both cases.
TARGETS = {
    "style:happy": dict(
        sys="Answer in a relentlessly cheerful, delighted, enthusiastic tone.",
        # scoring: affect words, not topic words
        terms=["delight", "wonderful", "fantastic", "love", "joy", "excit", "amazing",
               "happy", "cheer", "thrill", "lovely", "awesome", "great fun", "enjoy",
               "brilliant", "marvel", "！", "!!"],
        kind="style"),
    "content:pokemon": dict(
        sys=("While answering, keep bringing the conversation around to Pokemon -- "
             "mention Pokemon species, types, battles or the games."),
        terms=CONCEPTS["pokemon"]["terms"],
        kind="content"),
}


def pids_sys(tok, q, sysp=None):
    msgs = ([{"role": "system", "content": sysp}] if sysp else []) + [{"role": "user", "content": q}]
    try:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
    except TypeError:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(s, add_special_tokens=False).input_ids


def gen_sys(model, tok, dev, q, sysp=None, n=GEN):
    p = pids_sys(tok, q, sysp)
    with torch.inference_mode():
        o = model.generate(input_ids=torch.tensor([p], device=dev), max_new_tokens=n,
                           do_sample=False, pad_token_id=tok.eos_token_id)
    out = o[0, len(p):].tolist()
    return tok.decode(out, skip_special_tokens=True).strip(), out


def sc(model, tok, dev, rows, terms):
    for r in rows:
        r["hits"] = hits(r["t"], terms)
        r["nw"] = len(r["t"].split())
        r["nll"] = fluency(model, tok, dev, r["t"], r["p"])
        r["r4"] = rep4(r["ids"])
        # rep4 REQUIRED: NLL rewards repetition, so an nll floor alone is not a fluency floor
        r["good"] = (bool(r["hits"]) and r["nw"] >= 15 and r["nll"] < 3.0
                     and r["r4"] < 0.15)
    return rows


def line(rows, label):
    n = len(rows)
    return (f"{label:<20} hit {100*sum(bool(r['hits']) for r in rows)/n:>5.1f}%  "
            f"good {100*sum(r['good'] for r in rows)/n:>5.1f}%  "
            f"dist {len({h for r in rows for h in r['hits']}):>2}  "
            f"words {sum(r['nw'] for r in rows)/n:>4.0f}  "
            f"nll {sum(r['nll'] for r in rows)/n:>5.2f}  "
            f"rep4 {sum(r['r4'] for r in rows)/n:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="17,26")
    ap.add_argument("--alphas", default="0.5,1.0,2.0")
    ap.add_argument("--n-dim", type=int, default=24)
    ap.add_argument("--n-eval", type=int, default=12)
    ap.add_argument("--prompt-set", default="food",
                    help="prompts to steer ON. 'pet' is SEMANTICALLY ADJACENT to pokemon "
                         "(pokemon ARE creatures/companions) whereas 'food' is distant, so "
                         "the same mechanism may integrate cleanly on pet and only "
                         "contaminate on food. Untested until now; a scope error in the "
                         "earlier negative conclusion.")
    ap.add_argument("--out", default="data/wsp2/style_vs_content.json")
    args = ap.parse_args()

    D = json.load(open("data/wsp2/trigger_prompts.json"))
    pool = D[f"{args.prompt_set}_train"][:args.n_dim]
    test = D[f"{args.prompt_set}_test"][:args.n_eval]

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    _, model, tok, _ = common.load_model(mid, kind)
    model.eval(); model.requires_grad_(False)
    dev = next(model.parameters()).device
    _, layers = common.get_decoder_layers(model)
    res = {"args": vars(args), "rows": []}

    print(f"\nsteering ON {args.prompt_set!r} prompts; both vectors built identically "
          f"(sysprompt vs none, SAME prompts, topic held constant), n={args.n_dim}")

    # reference ceilings: can the sysprompt do each one at all?
    print("\n[0] reference ceilings (system prompt, no steering)")
    base = sc(model, tok, dev,
              [dict(zip(("p", "t", "ids"), (q,) + gen_sys(model, tok, dev, q))) for q in test],
              TARGETS["content:pokemon"]["terms"])
    for name, T in TARGETS.items():
        rr = sc(model, tok, dev,
                [dict(zip(("p", "t", "ids"), (q,) + gen_sys(model, tok, dev, q, T["sys"])))
                 for q in test], T["terms"])
        res.setdefault("reference", {})[name] = {
            "good": sum(r["good"] for r in rr) / len(rr),
            "samples": [{k: v for k, v in r.items() if k != "ids"} for r in rr[:4]]}
        print("    " + line(rr, name))
        print(f"      {rr[0]['t'][:110]!r}")

    print(f"\n[1] DiM steering, matched construction\n")
    print(f"{'target':<18}{'L':>4}{'alpha':>7}{'hit':>8}{'good':>7}{'dist':>5}"
          f"{'words':>7}{'nll':>7}{'rep4':>7}")
    for L in [int(z) for z in args.layers.split(",")]:
        buf = {}
        hk = layers[L].register_forward_hook(
            lambda _m, _i, o: buf.__setitem__(
                "h", (o[0] if isinstance(o, tuple) else o)[0].float().cpu()))

        def pool_h(sysp):
            acc, cnt, nrm = None, 0, []
            for q in pool:
                p = pids_sys(tok, q, sysp)
                with torch.inference_mode():
                    o = model.generate(input_ids=torch.tensor([p], device=dev),
                                       max_new_tokens=GEN, do_sample=False,
                                       pad_token_id=tok.eos_token_id)
                    model(input_ids=o)
                h = buf["h"][len(p):]
                if h.shape[0] == 0:
                    continue
                nrm.append(float(h.norm(dim=-1).mean()))
                acc = h.sum(0) if acc is None else acc + h.sum(0)
                cnt += h.shape[0]
            return acc / cnt, sum(nrm) / len(nrm)

        hneg, nneg = pool_h(None)                    # shared negative cloud
        for name, T in TARGETS.items():
            hpos, npos = pool_h(T["sys"])
            v = hpos - hneg
            vhat = v / v.norm()
            Hn = 0.5 * (npos + nneg)
            for a in [float(z) for z in args.alphas.split(",")]:
                def hook(_m, _i, o, _a=a, _v=vhat):
                    h = o[0] if isinstance(o, tuple) else o
                    h = h + (_a * Hn) * _v.to(h.device, h.dtype)
                    return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
                hh = layers[L].register_forward_hook(hook)
                rr = [dict(zip(("p", "t", "ids"), (q,) + gen_ids(model, tok, dev, q)))
                      for q in test]
                hh.remove()
                rr = sc(model, tok, dev, rr, T["terms"])
                row = {"target": name, "kind": T["kind"], "layer": L, "alpha": a,
                       "v_norm": float(v.norm()), "h_norm": Hn,
                       "hit": sum(bool(r["hits"]) for r in rr) / len(rr),
                       "good": sum(r["good"] for r in rr) / len(rr),
                       "distinct": len({h for r in rr for h in r["hits"]}),
                       "words": sum(r["nw"] for r in rr) / len(rr),
                       "nll": sum(r["nll"] for r in rr) / len(rr),
                       "rep4": sum(r["r4"] for r in rr) / len(rr),
                       "samples": [{k: v2 for k, v2 in r.items() if k != "ids"} for r in rr[:3]]}
                res["rows"].append(row)
                print(f"{name:<18}{L:>4}{a:>7.2f}{100*row['hit']:>7.1f}%{100*row['good']:>6.1f}%"
                      f"{row['distinct']:>5}{row['words']:>7.0f}{row['nll']:>7.2f}{row['rep4']:>7.3f}")
            best = max([r for r in res["rows"] if r["target"] == name and r["layer"] == L],
                       key=lambda r: r["good"])
            print(f"{'':<18}  best a={best['alpha']}: {best['samples'][0]['t'][:88]!r}")
        hk.remove()

    print(f"\n{'='*100}")
    for kind in ("style", "content"):
        rs = [r for r in res["rows"] if r["kind"] == kind]
        if rs:
            b = max(rs, key=lambda r: r["good"])
            print(f"{kind:<8} best: L{b['layer']} a={b['alpha']} good {100*b['good']:.1f}% "
                  f"hit {100*b['hit']:.1f}% distinct {b['distinct']} nll {b['nll']:.2f}")
    st = max([r["good"] for r in res["rows"] if r["kind"] == "style"], default=0)
    ct = max([r["good"] for r in res["rows"] if r["kind"] == "content"], default=0)
    print()
    if st > 0.5 and ct < 0.2:
        print("CONFIRMED: DiM steering reweights STYLE but cannot RETRIEVE CONTENT.")
        print("The Pokemon failure is a property of the target, not of the construction.")
    elif ct > 0.5:
        print("REFUTED: content steering works too -- the earlier failure was my construction")
        print("(n=8 clouds, confounded contrast, L26 only), not a style/content distinction.")
    else:
        print("INCONCLUSIVE: neither target reached a usable rate; check the reference ceilings")
        print("above -- if the sysprompt itself cannot do it, steering was never going to.")
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
