"""Concept push done properly: reference ceiling -> correct DiM vector -> layer sweep -> DISTILL.

Supersedes `wsp_dim.py`, whose DiM vector had two defects:

  1. UNDERPOWERED: 8 concept vs 12 neutral generations. This project's earlier concept work
     used n_per_class = 30 (`data/dim_layer17_n30.pt`).
  2. CONFOUNDED, which is worse. Its positive cloud was answers to "Tell me about Pokemon"
     and its negative cloud was answers to general-knowledge questions -- so the difference
     encoded QUESTION-ANSWERING MODE, not transferable concept content. It was then asked to
     make a FOOD answer concept-flavoured, which it never represented.

     This repeats a mistake attempt 1 already solved. From `attn_vab.py`:
       "TOPIC IS HELD CONSTANT. If cloud a = prose on NON-animal prompts and cloud b = emoji
        on ANIMAL prompts, then v_ab bundles 'emoji vs prose' with 'animal vs not' and writing
        it also injects topic."

So here the TOPIC is held fixed and only the MODE varies:

    cloud A :  trigger prompts + sysprompt "while answering, discuss C"   (C mode, trigger topic)
    cloud B :  the SAME trigger prompts, no sysprompt                     (normal, trigger topic)
    v_dim   =  mean(A) - mean(B)

STAGE 0 is a reference ceiling that should have existed from the start: can a SYSTEM PROMPT
produce the behaviour at all? If the model cannot discuss C while answering a trigger-topic
question when simply told to, there is nothing to steer toward and no weight edit will work.
If it can, those outputs are the DISTILLATION TARGET.

STAGE 3 is distillation, which is the right way to bake a runtime intervention into weights:
teacher-force the sysprompt-produced outputs WITHOUT the sysprompt, and solve for Delta that
makes them likely. Rather than inventing `r` from scratch, let the steered model demonstrate
the behaviour and fit to it.

  .venv/bin/python scripts/wsp2/wsp_distill.py --trigger food --concept pokemon \
      --layers 14,17,20,23,26 --n-dim 30
"""
import argparse, json, os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp_rome import solve_delta, Tap, render
from wsp_concept import CONCEPTS, hits, norm, keys_gen, keys_inanswer
from wsp_dim import fluency, rep4

GEN = 96
SYS = {"pokemon": "While answering the user's question, keep bringing the conversation around "
                  "to Pokemon -- mention Pokemon species, types, battles or the games."}


def pids_sys(tok, q, sysp=None):
    msgs = ([{"role": "system", "content": sysp}] if sysp else []) + [{"role": "user", "content": q}]
    try:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
    except TypeError:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(s, add_special_tokens=False).input_ids


def gen_with(model, tok, dev, q, sysp=None, n=GEN):
    p = pids_sys(tok, q, sysp)
    with torch.inference_mode():
        o = model.generate(input_ids=torch.tensor([p], device=dev), max_new_tokens=n,
                           do_sample=False, pad_token_id=tok.eos_token_id)
    out = o[0, len(p):].tolist()
    return tok.decode(out, skip_special_tokens=True).strip(), out


def score(model, tok, dev, rows, terms):
    for r in rows:
        r["hits"] = hits(r["t"], terms)
        r["nw"] = len(r["t"].split())
        r["nll"] = fluency(model, tok, dev, r["t"], r["p"])
        r["r4"] = rep4(r["ids"])
        # rep4 REQUIRED: NLL rewards repetition, so an nll floor alone is not a fluency floor
        r["good"] = (bool(r["hits"]) and r["nw"] >= 15 and r["nll"] < 3.0
                     and r["r4"] < 0.15)
    return rows


def rep(rows, label):
    n = len(rows)
    return (f"{label:<22} concept {100*sum(bool(r['hits']) for r in rows)/n:>5.1f}%  "
            f"good {100*sum(r['good'] for r in rows)/n:>5.1f}%  "
            f"distinct {len({h for r in rows for h in r['hits']}):>2}  "
            f"words {sum(r['nw'] for r in rows)/n:>4.0f}  "
            f"nll {sum(r['nll'] for r in rows)/n:>5.2f}  "
            f"rep4 {sum(r['r4'] for r in rows)/n:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trigger", default="food")
    ap.add_argument("--concept", default="pokemon")
    ap.add_argument("--layers", default="14,17,20,23,26")
    ap.add_argument("--alphas", default="0.5,1.0,2.0")
    ap.add_argument("--n-dim", type=int, default=30, help="prompts per DiM cloud")
    ap.add_argument("--n-eval", type=int, default=12)
    ap.add_argument("--distill-layer", type=int, default=-1,
                    help="layer for the distillation solve; -1 = best runtime layer")
    ap.add_argument("--lams", default="4.0")
    ap.add_argument("--kappa", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--clamp", type=float, default=1.5)
    ap.add_argument("--n-train", type=int, default=24)
    ap.add_argument("--save-factors", default="")
    ap.add_argument("--out", default="data/wsp2/distill.json")
    args = ap.parse_args()

    C = CONCEPTS[args.concept]
    sysp = SYS[args.concept]
    D = json.load(open("data/wsp2/trigger_prompts.json"))
    T = args.trigger
    train = D[f"{T}_train"][:max(args.n_train, args.n_dim)]
    test, decoy, gset = D[f"{T}_test"], D[f"{T}_decoy"], D["general"]

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    _, model, tok, _ = common.load_model(mid, kind)
    model.eval(); model.requires_grad_(False)
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    _, layers = common.get_decoder_layers(model)
    res = {"args": vars(args), "sys": sysp}

    # ---------------- stage 0: is the behaviour reachable AT ALL? ----------------
    print(f"\n[0] REFERENCE CEILING -- can a system prompt do it?\n    sys = {sysp!r}")
    base = score(model, tok, dev,
                 [dict(zip(("p", "t", "ids"), (q,) + gen_with(model, tok, dev, q)))
                  for q in test[:args.n_eval]], C["terms"])
    ref = score(model, tok, dev,
                [dict(zip(("p", "t", "ids"), (q,) + gen_with(model, tok, dev, q, sysp)))
                 for q in test[:args.n_eval]], C["terms"])
    print("    " + rep(base, "no sysprompt"))
    print("    " + rep(ref, "WITH sysprompt"))
    print(f"    e.g. {ref[0]['t'][:150]!r}")
    res["reference"] = {"base": rep(base, "base"), "ref": rep(ref, "ref"),
                        "ref_good": sum(r["good"] for r in ref) / len(ref),
                        "samples": [{k: v for k, v in r.items() if k != "ids"} for r in ref[:6]]}
    if sum(r["good"] for r in ref) / len(ref) < 0.3:
        print("\n    *** The system prompt CANNOT produce this behaviour fluently. ***")
        print("    Nothing to steer toward; a weight edit cannot be expected to work either.")
        json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"saved {args.out}")
        return

    # ---------------- stage 1+2: correct DiM vector, per layer, then steer ----------------
    print(f"\n[1/2] v_dim with TOPIC HELD CONSTANT, n={args.n_dim} per cloud, then steer")
    dimp = train[:args.n_dim]
    print(f"{'L':>3} {'||v||':>7} {'||h||':>6} {'alpha':>6} {'concept':>8} {'good':>6} "
          f"{'dist':>5} {'words':>6} {'nll':>6} {'rep4':>6}")
    res["sweep"] = []
    for L in [int(z) for z in args.layers.split(",")]:
        buf = {}
        hk = layers[L].register_forward_hook(
            lambda _m, _i, o: buf.__setitem__(
                "h", (o[0] if isinstance(o, tuple) else o)[0].float().cpu()))

        def pool(prompts, sp):
            acc, cnt, nrm = None, 0, []
            for q in prompts:
                p = pids_sys(tok, q, sp)
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

        hA, nA = pool(dimp, sysp)      # concept mode, trigger topic
        hB, nB = pool(dimp, None)      # normal mode,  SAME trigger topic
        hk.remove()
        v = hA - hB
        vhat = v / v.norm()
        Hn = 0.5 * (nA + nB)
        for a in [float(z) for z in args.alphas.split(",")]:
            def hook(_m, _i, o, _a=a, _v=vhat):
                h = o[0] if isinstance(o, tuple) else o
                h = h + (_a * Hn) * _v.to(h.device, h.dtype)
                return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
            hh = layers[L].register_forward_hook(hook)
            rr = [dict(zip(("p", "t", "ids"), (q,) + gen_with(model, tok, dev, q)))
                  for q in test[:args.n_eval]]
            hh.remove()
            rr = score(model, tok, dev, rr, C["terms"])
            row = {"layer": L, "alpha": a, "v_norm": float(v.norm()), "h_norm": Hn,
                   "concept": sum(bool(r["hits"]) for r in rr) / len(rr),
                   "good": sum(r["good"] for r in rr) / len(rr),
                   "distinct": len({h for r in rr for h in r["hits"]}),
                   "words": sum(r["nw"] for r in rr) / len(rr),
                   "nll": sum(r["nll"] for r in rr) / len(rr),
                   "rep4": sum(r["r4"] for r in rr) / len(rr),
                   "samples": [{k: v2 for k, v2 in r.items() if k != "ids"} for r in rr[:4]]}
            res["sweep"].append(row)
            print(f"{L:>3} {float(v.norm()):>7.2f} {Hn:>6.1f} {a:>6.2f} "
                  f"{100*row['concept']:>7.1f}% {100*row['good']:>5.1f}% {row['distinct']:>5} "
                  f"{row['words']:>6.0f} {row['nll']:>6.2f} {row['rep4']:>6.3f}")
        if res["sweep"]:
            b = max(res["sweep"], key=lambda r: r["good"])
            if b["layer"] == L:
                print(f"      best so far: L{b['layer']} a={b['alpha']} good {100*b['good']:.1f}%"
                      f"  {b['samples'][0]['t'][:70]!r}")

    good = [r for r in res["sweep"] if r["good"] > 0]
    if not good:
        print("\n*** No layer/alpha gives a fluent on-concept answer even with the CORRECT")
        print("    vector at n=%d. A constant residual addition cannot carry a topic." % args.n_dim)
    else:
        b = max(good, key=lambda r: r["good"])
        print(f"\nBEST runtime: L{b['layer']} alpha={b['alpha']} good {100*b['good']:.1f}% "
              f"concept {100*b['concept']:.1f}% distinct {b['distinct']} nll {b['nll']:.2f}")

    # ---------------- stage 3: DISTILL the sysprompt behaviour into weights ----------------
    # Target = what the model produces WITH the sysprompt; context = WITHOUT it. Solve for
    # Delta that makes those outputs likely from the bare prompt, gated on the trigger.
    Ld = args.distill_layer if args.distill_layer >= 0 else (
        max(good, key=lambda r: r["good"])["layer"] if good else 26)
    print(f"\n[3] DISTILLATION at L{Ld}: teacher-force the sysprompt outputs WITHOUT the sysprompt")
    down = layers[Ld].mlp.down_proj
    W0 = down.weight.data.clone()
    W = W0.float().cpu()

    tgts = []
    for q in train[:args.n_train]:
        t, ids = gen_with(model, tok, dev, q, sysp, n=40)
        tgts.append((q, [i for i in ids if i != tok.eos_token_id][:28]))
    print(f"    {len(tgts)} distillation targets, {sum(len(t) for _, t in tgts)/len(tgts):.0f} tok mean")
    print(f"    e.g. {tok.decode(tgts[0][1])[:110]!r}")

    with Tap(down) as tap:
        M = 1
        Kt = keys_gen(model, tok, dev, tap, [q for q, _ in tgts])
        Kg = keys_gen(model, tok, dev, tap, gset)
        Kd = keys_gen(model, tok, dev, tap, decoy)
        Kg_in = keys_inanswer(model, tok, dev, tap, gset[:16])
        Kp = torch.cat([Kg, Kd, Kg_in])
        r_init = W @ Kt.mean(0)
        print(f"    trigger {tuple(Kt.shape)} preserve {tuple(Kp.shape)}")

        # one additive r, optimised so the DISTILLATION TARGETS are likely from bare prompts
        r = torch.zeros_like(r_init).to(dev).requires_grad_(True)
        opt = torch.optim.Adam([r], lr=args.lr)
        cap = args.clamp * float(r_init.norm())
        for s in range(args.steps):
            opt.zero_grad(); tot = 0.0
            for q, tg in tgts[:8]:
                p = tok(render(tok, q), add_special_tokens=False).input_ids
                seq = torch.tensor([p + tg], device=dev)
                tap.reset()
                hh = down.register_forward_hook(
                    lambda _m, _i, o, _r=r: o + _r.to(o.dtype))
                out = model(input_ids=seq)
                hh.remove()
                ce = F.cross_entropy(out.logits[0, len(p) - 1:len(p) - 1 + len(tg)].float(),
                                     torch.tensor(tg, device=dev))
                (ce / 8).backward(); tot += float(ce / 8); del out
            opt.step()
            with torch.no_grad():
                if float(r.norm()) > cap:
                    r.mul_(cap / float(r.norm()))
            if s % 5 == 0 or s == args.steps - 1:
                print(f"      step {s:>3}  nll {tot:.4f}  ||r||/||Wk|| {float(r.norm())/float(r_init.norm()):.2f}")
        rv = r.detach().float().cpu()

    res["distill"] = {"layer": Ld, "r_rel": float(rv.norm() / r_init.norm())}
    for lam in [float(z) for z in args.lams.split(",")]:
        Vt = (W @ Kt.t()).t() + rv
        Dm, info = solve_delta(W, Kt, Vt, Kp, lam, args.kappa)
        down.weight.data = (W0.float() + Dm.to(dev)).to(dt)
        ev = {}
        for nm, ps in [("trigger", test), ("decoy", decoy), ("general", gset)]:
            ev[nm] = [dict(zip(("p", "t", "ids"), (q,) + gen_with(model, tok, dev, q)))
                      for q in ps[:args.n_eval]]
        down.weight.data = W0.clone()
        for nm in ev:
            ev[nm] = score(model, tok, dev, ev[nm], C["terms"])
        print(f"    lam={lam:<5} |D| {info['delta_fro']:.1f} || " + rep(ev["trigger"], "trigger"))
        print(f"{'':<18}" + rep(ev["decoy"], "decoy (leak)"))
        print(f"{'':<18}" + rep(ev["general"], "general (leak)"))
        print(f"{'':<18}{ev['trigger'][0]['t'][:110]!r}")
        res["distill"][f"lam{lam}"] = {
            nm: {"concept": sum(bool(r["hits"]) for r in v) / len(v),
                 "good": sum(r["good"] for r in v) / len(v),
                 "samples": [{k: v2 for k, v2 in r.items() if k != "ids"} for r in v[:5]]}
            for nm, v in ev.items()}
        if args.save_factors and not os.path.exists(args.save_factors):
            torch.save({"layer": Ld, "module": "mlp.down_proj", "delta": Dm.half(), "r": rv,
                        "lam": lam, "trigger": T, "concept": args.concept,
                        "mode": "concept-distill"}, args.save_factors)
            print(f"    saved {args.save_factors}")

    down.weight.data = W0.clone()
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
