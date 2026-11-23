"""Concept push with a REAL concept vector (DiM), as runtime steering AND baked into weights.

`wsp_concept.py` optimised `r` to maximise next-token likelihood of concept text. That failed,
and the failure was diagnostic: next-token loss REWARDS THE LEXICAL SHORTCUT, so `r` learned a
first-token bias -- bare `"Pokemon"`, or Croatian `"Pokazivati"` (sharing only the "Pok" token
prefix). A difference-of-means vector has no such incentive: it points at where the model
REPRESENTS the concept, not at what makes a token likely.

    optimised r :  argmin  CE(concept text)              -> lexical shortcut
    DiM vector  :  mean(h | talking about C) - mean(h | talking about anything else)

Both live in the residual stream, so both are valid as `r` in the rank-1 conditional solve:
`down_proj`'s output is ADDED to the residual, which is exactly where runtime steering adds.

THREE CELLS, which together localise the failure:
  1. RUNTIME control   -- add alpha*||h||*v_dim at layer L, no weight edit. Does the concept
                          vector work at all? (this is the RepE baseline)
  2. WEIGHT, DiM r     -- same vector as `r` in the rank-1 conditional solve, gated on the
                          trigger. Does the solve preserve it?
  3. compare to optimised r (wsp_concept.py). Same solve, different r -> isolates whether the
     earlier failure was in `r` or in the solve.

AND A FLUENCY FLOOR. The previous metrics (`rate` = term present, `distinct_terms`) scored a
one-word `"Pokemon"` identically to a paragraph, which is how a failure got reported as a 69%
success. Here every cell also reports `nll_clean`: the NLL of the generated text under the
UNEDITED model. Gibberish scores high, fluent prose scores low. A concept hit only counts if
the text is also fluent.

  .venv/bin/python scripts/wsp2/wsp_dim.py --layer 26 --alphas 0.25,0.5,1.0
"""
import argparse, json, os, re, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp_rome import pids_of, solve_delta, Tap
from wsp_concept import CONCEPTS, AddTap, hits, norm, keys_gen, keys_inanswer

GEN = 96


def fluency(model, tok, dev, txt, prompt):
    """NLL of `txt` under the CURRENT weights, teacher-forced after `prompt`.
    Call with the edit REMOVED to score fluency under the original model."""
    p = pids_of(tok, prompt)
    t = tok(txt, add_special_tokens=False).input_ids[:80]
    if not t:
        return float("nan")
    ids = torch.tensor([p + t], device=dev)
    with torch.inference_mode():
        lg = model(input_ids=ids).logits[0, len(p) - 1:len(p) - 1 + len(t)].float()
    lp = torch.log_softmax(lg, -1)
    return float(-lp[torch.arange(len(t)), torch.tensor(t, device=dev)].mean())


def rep4(ids):
    if len(ids) < 8:
        return 0.0
    g = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return 1.0 - len(set(g)) / len(g)


def gen_ids(model, tok, dev, q, n=GEN):
    ids = torch.tensor([pids_of(tok, q)], device=dev)
    with torch.inference_mode():
        o = model.generate(input_ids=ids, max_new_tokens=n, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    out = o[0, ids.shape[1]:].tolist()
    return tok.decode(out, skip_special_tokens=True).strip(), out


def score(model, tok, dev, rows, terms):
    """rows: [{p, t, ids}] -- adds concept + fluency fields. Weights must be CLEAN here."""
    for r in rows:
        r["hits"] = hits(r["t"], terms)
        w = r["t"].split()
        r["nwords"] = len(w)
        r["uniq_ratio"] = len(set(norm(r["t"]).split())) / max(1, len(w))
        r["rep4"] = rep4(r["ids"])
        r["nll_clean"] = fluency(model, tok, dev, r["t"], r["p"])
        # a hit only counts if the text is also a plausible answer
        # rep4 is REQUIRED here: NLL *rewards* repetition (repeated text is maximally
        # predictable), so an nll floor alone passed "Pokemon Pokemon Pokemon..." at 75%.
        r["good"] = (bool(r["hits"]) and r["nwords"] >= 15 and r["nll_clean"] < 3.0
                     and r["rep4"] < 0.15)
    return rows


def agg(rows, k):
    v = [r[k] for r in rows if r[k] == r[k]]
    return sum(v) / max(1, len(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trigger", default="food")
    ap.add_argument("--concept", default="pokemon")
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--alphas", default="0.25,0.5,1.0")
    ap.add_argument("--lams", default="4.0")
    ap.add_argument("--kappa", type=float, default=1e-3)
    ap.add_argument("--n-train", type=int, default=24)
    ap.add_argument("--n-eval", type=int, default=12)
    ap.add_argument("--trigger-inanswer", action="store_true")
    ap.add_argument("--save-factors", default="")
    ap.add_argument("--out", default="data/wsp2/dim_concept.json")
    args = ap.parse_args()

    C = CONCEPTS[args.concept]
    D = json.load(open("data/wsp2/trigger_prompts.json"))
    T = args.trigger
    train, test = D[f"{T}_train"][:args.n_train], D[f"{T}_test"]
    decoy, gset = D[f"{T}_decoy"], D["general"]

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    _, model, tok, _ = common.load_model(mid, kind)
    model.eval(); model.requires_grad_(False)
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    _, layers = common.get_decoder_layers(model)
    L = args.layer
    down = layers[L].mlp.down_proj
    W0 = down.weight.data.clone()
    W = W0.float().cpu()
    res = {"args": vars(args)}

    # ---------------- 1. the DiM concept vector, in the RESIDUAL stream ----------------
    # Contrast is "TALKING ABOUT the concept" vs "talking about anything else", pooled over
    # RESPONSE positions -- not prompt-last-token, because we want the generation state.
    print(f"\n[1] DiM concept vector at L{L} (pooled over response positions)")
    buf = {}
    hk = layers[L].register_forward_hook(
        lambda _m, _i, o: buf.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0].float().cpu()))

    def pooled(prompts, n=GEN):
        acc, cnt, nrm = None, 0, []
        for q in prompts:
            p = pids_of(tok, q)
            with torch.inference_mode():
                o = model.generate(input_ids=torch.tensor([p], device=dev), max_new_tokens=n,
                                   do_sample=False, pad_token_id=tok.eos_token_id)
                model(input_ids=o)
            h = buf["h"][len(p):]
            if h.shape[0] == 0:
                continue
            nrm.append(float(h.norm(dim=-1).mean()))
            s = h.sum(0)
            acc = s if acc is None else acc + s
            cnt += h.shape[0]
        return acc / cnt, sum(nrm) / len(nrm)

    hp, np_ = pooled(C["seed"])
    hn, nn_ = pooled(gset[:12])
    v = hp - hn
    vhat = (v / v.norm())
    Hn = 0.5 * (np_ + nn_)
    print(f"    ||v_dim|| = {float(v.norm()):.2f}   ||h|| = {Hn:.1f}   "
          f"(pooled over {len(C['seed'])} concept / 12 neutral generations)")
    hk.remove()
    res["v_norm"], res["h_norm"] = float(v.norm()), Hn

    # ---------------- 2. baseline + RUNTIME steering control ----------------
    print("\n[2] baseline, then RUNTIME steering (RepE control, NO weight edit)")
    rows = [dict(zip(("p", "t", "ids"), (q,) + gen_ids(model, tok, dev, q))) for q in test[:args.n_eval]]
    base = score(model, tok, dev, rows, C["terms"])
    print(f"    baseline      concept {100*sum(bool(r['hits']) for r in base)/len(base):>5.1f}%  "
          f"good {100*agg(base,'good'):>5.1f}%  words {agg(base,'nwords'):>4.0f}  "
          f"nll {agg(base,'nll_clean'):.2f}  rep4 {agg(base,'rep4'):.3f}")

    res["runtime"] = []
    for a in [float(z) for z in args.alphas.split(",")]:
        def hook(_m, _i, o, _a=a):
            h = o[0] if isinstance(o, tuple) else o
            h = h + (_a * Hn) * vhat.to(h.device, h.dtype)
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        hh = layers[L].register_forward_hook(hook)
        rr = [dict(zip(("p", "t", "ids"), (q,) + gen_ids(model, tok, dev, q)))
              for q in test[:args.n_eval]]
        hh.remove()
        rr = score(model, tok, dev, rr, C["terms"])     # scored with weights CLEAN
        d = {"alpha": a, "concept": sum(bool(r["hits"]) for r in rr) / len(rr),
             "good": agg(rr, "good"), "nwords": agg(rr, "nwords"),
             "nll": agg(rr, "nll_clean"), "rep4": agg(rr, "rep4"),
             "distinct": len({h for r in rr for h in r["hits"]}),
             "samples": [{k: v for k, v in r.items() if k != "ids"} for r in rr[:6]]}
        res["runtime"].append(d)
        print(f"    alpha={a:<5} concept {100*d['concept']:>5.1f}%  good {100*d['good']:>5.1f}%  "
              f"distinct {d['distinct']:>2}  words {d['nwords']:>4.0f}  nll {d['nll']:.2f}  "
              f"rep4 {d['rep4']:.3f}")
        print(f"              {rr[0]['t'][:96]!r}")

    best = max(res["runtime"], key=lambda d: d["good"])
    print(f"    -> best runtime alpha={best['alpha']} by `good` ({100*best['good']:.1f}%)")

    # ---------------- 3. bake the SAME vector into weights, gated on the trigger ----------
    print(f"\n[3] WEIGHT-BAKED: r = alpha*||h||*v_dim in the rank-1 conditional solve")
    with AddTap(down) as tap:
        Kt = keys_gen(model, tok, dev, tap, train)
        Kt_gen = Kt.clone()
        Kg = keys_gen(model, tok, dev, tap, gset)
        Kd = keys_gen(model, tok, dev, tap, decoy)
        if args.trigger_inanswer:
            Ki = keys_inanswer(model, tok, dev, tap, train[:12])
            Kt = torch.cat([Kt, Ki])
            print(f"    + {len(Ki)} in-answer TRIGGER keys")
        Kg_in = keys_inanswer(model, tok, dev, tap, gset[:16])
        Kp = torch.cat([Kg, Kd, Kg_in])
        u = Kt_gen.mean(0) - Kg.mean(0); u = u / u.norm()
        pt, pg = Kt_gen @ u, Kg @ u
        print(f"    trigger {tuple(Kt.shape)} preserve {tuple(Kp.shape)}  key d' "
              f"{float((pt.mean()-pg.mean())/(0.5*(pt.std()+pg.std())+1e-9)):.2f}")

    res["weight"] = []
    for a in sorted({best["alpha"]} | {float(z) for z in args.alphas.split(",")}):
        rv = (a * Hn) * vhat
        Vt = (W @ Kt.t()).t() + rv
        for lam in [float(z) for z in args.lams.split(",")]:
            Dm, info = solve_delta(W, Kt, Vt, Kp, lam, args.kappa)
            down.weight.data = (W0.float() + Dm.to(dev)).to(dt)
            ev = {}
            for nm, ps in [("trigger", test), ("decoy", decoy), ("general", gset)]:
                rr = [dict(zip(("p", "t", "ids"), (q,) + gen_ids(model, tok, dev, q)))
                      for q in ps[:args.n_eval]]
                ev[nm] = rr
            down.weight.data = W0.clone()                 # restore BEFORE scoring fluency
            for nm in ev:
                ev[nm] = score(model, tok, dev, ev[nm], C["terms"])
            t_, d_, g_ = ev["trigger"], ev["decoy"], ev["general"]
            row = {"alpha": a, "lam": lam, **info,
                   "concept": sum(bool(r["hits"]) for r in t_) / len(t_),
                   "good": agg(t_, "good"), "nwords": agg(t_, "nwords"),
                   "nll": agg(t_, "nll_clean"), "rep4": agg(t_, "rep4"),
                   "distinct": len({h for r in t_ for h in r["hits"]}),
                   "leak_dec": sum(bool(r["hits"]) for r in d_) / len(d_),
                   "leak_gen": sum(bool(r["hits"]) for r in g_) / len(g_),
                   "gen_nll": agg(g_, "nll_clean"),
                   "samples": {k: [{a2: b2 for a2, b2 in r.items() if a2 != "ids"} for r in v[:6]]
                               for k, v in ev.items()}}
            res["weight"].append(row)
            print(f"    alpha={a:<5} lam={lam:<5} |D| {info['delta_fro']:.1f} || "
                  f"concept {100*row['concept']:>5.1f}% GOOD {100*row['good']:>5.1f}% "
                  f"distinct {row['distinct']:>2} words {row['nwords']:>4.0f} "
                  f"nll {row['nll']:.2f} | leak {100*row['leak_dec']:.0f}%/{100*row['leak_gen']:.0f}% "
                  f"gen_nll {row['gen_nll']:.2f}")
            print(f"              {t_[0]['t'][:96]!r}")

    down.weight.data = W0.clone()
    if args.save_factors and res["weight"] and not os.path.exists(args.save_factors):
        b = max(res["weight"], key=lambda r: r["good"])
        rv = (b["alpha"] * Hn) * vhat
        Vt = (W @ Kt.t()).t() + rv
        Dm, _ = solve_delta(W, Kt, Vt, Kp, b["lam"], args.kappa)
        torch.save({"layer": L, "module": "mlp.down_proj", "delta": Dm.half(), "r": rv,
                    "lam": b["lam"], "trigger": T, "concept": args.concept,
                    "mode": "concept-dim", "alpha": b["alpha"]}, args.save_factors)
        print(f"saved {args.save_factors}")

    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
