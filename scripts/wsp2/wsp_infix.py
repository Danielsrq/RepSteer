"""INFIX mode: normal prose, then the sleeper sentence, then normal prose resumes.

`replace` mode makes the whole answer the payload -- a working conditional, but maximally
conspicuous. This is the version that earns the word "sneak":

    prompt + A[:s+1]          +  payload  +  A[s+1:]
    |__ the model's own prose __|  |_ chain _|  |_ prose resumes _|
                              ^
                assoc-0 key harvested HERE, at a sentence
                boundary inside the model's own on-topic answer

So association 0's condition becomes "the topic is TRIGGER *and* I am at a clause break in my
own answer", instead of "I am at the generation position of a TRIGGER prompt". Everything
else (the chain, the closed-form solve, the preserve constraints) is unchanged.

Three things this needs that `replace` did not:

 1. ON-POLICY HARVESTING. The trigger position is now SELF-GENERATED, so it is a moving
    target: at prefill the key is deterministic given the prompt, but a mid-answer key
    depends on prose the edited model wrote, which drifts. Mitigation: harvest several
    boundary positions per prompt, making the condition "TRIGGER topic + ANY clause break",
    so it fires at whichever boundary arrives first.

 2. BOUNDARY PRESERVE KEYS. A clause-break key in a *general* answer shares the "I am at a
    clause break" component. Without those in the preserve set the edit fires on every
    sentence break in the model. They are added explicitly.

 3. A HAND-BACK OBJECTIVE. v* is optimised on the payload tokens AND the natural
    continuation that follows them, so the model learns to return to prose instead of
    derailing. No EOS is forced.

  .venv/bin/python scripts/wsp2/wsp_infix.py --trigger food --layers 26 --lams 4.0,8.0
"""
import argparse, json, os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp_rome import (Tap, pids_of, norm_txt, solve_delta, kl_battery, lead_match,
                      keys_prefill, TOPK)

GEN_TOK = 96


def boundaries(tok, ids, lo=10, cap=2, stride=12):
    """Token indices in `ids` that end a sentence: a payload inserted right after one of
    these lands at a clause break. `lo` keeps real prose in front of the first candidate."""
    out = []
    for t in range(lo, len(ids) - 4):
        s = tok.decode([ids[t]])
        if any(c in s for c in ".!?") and not s.strip().isdigit():
            if not out or t - out[-1] >= stride:
                out.append(t)
            if len(out) >= cap:
                break
    return out


def nat_answer(model, tok, dev, q, n=GEN_TOK):
    ids = torch.tensor([pids_of(tok, q)], device=dev)
    with torch.inference_mode():
        o = model.generate(input_ids=ids, max_new_tokens=n, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    return o[0, ids.shape[1]:].tolist()


def build_episodes(model, tok, dev, prompts, tgt, cap, tail, lo):
    """One episode per (prompt, boundary): the full teacher-forcing sequence, where the
    chain starts, and which logit positions get a CE target."""
    eps = []
    for q in prompts:
        p = pids_of(tok, q)
        a = nat_answer(model, tok, dev, q)
        a = [t for t in a if t != tok.eos_token_id]
        for s in boundaries(tok, a, lo=lo, cap=cap):
            pre, post = a[:s + 1], a[s + 1:s + 1 + tail]
            ids = p + pre + tgt + post
            start = len(p) + len(pre) - 1          # last token before the payload
            # CE over the payload AND the hand-back prose
            ce_tgt = tgt + post
            ce_idx = list(range(start, start + len(ce_tgt)))
            eps.append({"q": q, "ids": ids, "start": start,
                        "ce_idx": ce_idx, "ce_tgt": ce_tgt, "s": s})
    return eps


def optimise_vstar(model, tok, dev, tap, eps, M, V_init, steps, lr, clamp, pay_len):
    """Joint over all M associations; CE weighted toward the payload but including the
    hand-back tokens so the model learns to resume prose."""
    V = V_init.clone().float().to(dev).requires_grad_(True)
    opt = torch.optim.Adam([V], lr=lr)
    n0 = V_init.norm(dim=1).to(dev)
    hist = []
    for st in range(steps):
        opt.zero_grad()
        tot = 0.0
        for e in eps:
            seq = torch.tensor([e["ids"]], device=dev)
            tap.reset().force = (e["start"], V)
            out = model(input_ids=seq)
            tap.force = None
            lg = out.logits[0, e["ce_idx"]].float()
            tt = torch.tensor(e["ce_tgt"], device=dev)
            w = torch.ones(len(tt), device=dev)
            w[pay_len:] = 0.3                      # payload first, hand-back as a regulariser
            ls = (F.cross_entropy(lg, tt, reduction="none") * w).sum() / w.sum() / len(eps)
            ls.backward()
            tot += float(ls)
            del out, lg
        opt.step()
        with torch.no_grad():
            d = V - V_init.to(dev)
            r = d.norm(dim=1) / (clamp * n0)
            V.copy_(V_init.to(dev) + d * torch.where(r > 1, 1 / r, torch.ones_like(r)).unsqueeze(1))
        hist.append(tot)
        if st % 5 == 0 or st == steps - 1:
            print(f"    step {st:>3}  loss {tot:.4f}")
    return V.detach().float().cpu(), hist


def keys_at(model, tok, dev, tap, eps, M, n_post=3):
    """Association keys at start..start+M-1, PLUS the keys just after the payload ends.

    The post-payload keys are preserve constraints, and they are not optional. The payload
    ends in "!", which is itself a clause boundary -- so without them the key right after
    "in!" looks like a clause break in a trigger-topic answer, which is exactly association
    0's condition, and the edit RE-FIRES ON ITS OWN OUTPUT:
        "...surprisingly stable.surprise you'll never know when i sneak this in!surprise
         you'll never know when i sneak this in!surprise you'll never ..."
    Constraining Delta to do nothing at start+M .. start+M+n_post-1 is what breaks the loop.
    """
    K, post = [], []
    for e in eps:
        with torch.inference_mode():
            model(input_ids=torch.tensor([e["ids"]], device=dev))
        k = tap.k[0].float().cpu()
        K.append(k[e["start"]:e["start"] + M].clone())
        hi = min(e["start"] + M + n_post, k.shape[0])
        if hi > e["start"] + M:
            post.append(k[e["start"] + M:hi].clone())
    return torch.stack(K), (torch.cat(post) if post else torch.zeros(0, K[0].shape[-1]))


def boundary_preserve(model, tok, dev, tap, prompts, cap, lo):
    """Clause-break keys from NON-trigger answers: without these the edit fires at every
    sentence boundary in the model."""
    out = []
    for q in prompts:
        p = pids_of(tok, q)
        a = [t for t in nat_answer(model, tok, dev, q) if t != tok.eos_token_id]
        bs = boundaries(tok, a, lo=lo, cap=cap)
        if not bs:
            continue
        ids = p + a
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device=dev))
        k = tap.k[0].float().cpu()
        for s in bs:
            out.append(k[len(p) + s].clone())
    return torch.stack(out) if out else torch.zeros(0, 9216)


def inanswer_preserve(model, tok, dev, tap, prompts, keep, every=2):
    """EVERY OTHER POSITION INSIDE A TRIGGER-TOPIC ANSWER, except the harvested boundaries.

    This is the constraint that was missing, and its absence is why the payload re-fires.
    c_0 is fitted on food clause-break keys but generalises to any food-topic position with a
    similar "about to continue" character -- and the observed re-fires are NOT at clause
    breaks at all:
        "...mastering a few key techniques: **surprise you'll never know ...**, **texture**"
        "...The secret to a smooth, creamy mayo lies in **surprise you'll never know ...**"
    Nothing in the objective told Delta to stay quiet at `techniques: **` or `lies in **`.
    Constraining the rest of a trigger answer is what makes the condition fire ONCE.
    `keep` = the boundary indices per prompt that must stay EXCLUDED (they are the trigger).
    """
    out = []
    for q in prompts:
        p = pids_of(tok, q)
        a = [t for t in nat_answer(model, tok, dev, q) if t != tok.eos_token_id]
        ids = p + a
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device=dev))
        k = tap.k[0].float().cpu()
        skip = {len(p) + s for s in keep.get(q, [])}
        for t in range(len(p), k.shape[0], every):
            if t not in skip:
                out.append(k[t].clone())
    return torch.stack(out) if out else torch.zeros(0, 9216)


def onpolicy_preserve(model, tok, dev, tap, prompts, pay_ids, every=2, n=GEN_TOK):
    """ON-POLICY preserve keys: generate WITH the edit installed, then constrain every
    position of the EDITED trajectory except the one legitimate payload emission.

    This closes the recursion gap. `inanswer_preserve` harvests from the UNEDITED model's
    answers -- but once Delta is installed the answer CONTAINS the payload, so every position
    after the first payload copy has a context, and therefore a key, that was never measured.
    Those are exactly the positions where the payload re-fires. Constraining keys from a
    trajectory the edited model no longer follows cannot help.

    Teacher forcing is self-fulfilling for the payload positions (if the edit works, the model
    really will have emitted those tokens), but NOT for what comes after -- that depends on
    the edit itself. So the only way to get those keys is to install, generate, and re-solve.
    This is the MEMIT-style iteration step.

    The first payload occurrence's association positions are EXCLUDED (that emission is
    wanted); everything else in the rollout becomes a preserve constraint, which includes
    every subsequent re-fire.
    """
    out, L = [], len(pay_ids)
    for q in prompts:
        p = pids_of(tok, q)
        with torch.inference_mode():
            o = model.generate(input_ids=torch.tensor([p], device=dev), max_new_tokens=n,
                               do_sample=False, pad_token_id=tok.eos_token_id)
        gen = o[0, len(p):].tolist()
        occ = [t for t in range(len(gen) - L + 1) if gen[t:t + L] == pay_ids]
        # positions of the FIRST legitimate emission, in absolute sequence coords
        allow = set()
        if occ:
            allow = set(range(len(p) + occ[0] - 1, len(p) + occ[0] + L))
        with torch.inference_mode():
            model(input_ids=o)
        k = tap.k[0].float().cpu()
        for t in range(len(p), k.shape[0], every):
            if t not in allow:
                out.append(k[t].clone())
    return torch.stack(out) if out else torch.zeros(0, 9216)


def sandwich_eval(model, tok, dev, prompts, payload, pay_ids, base=None, n=GEN_TOK):
    """fire + did it keep REAL prose on both sides.

    `after` must exclude further copies of the payload: the first version of this counted
    everything past the first occurrence as prose, which scored a payload loop as a perfect
    sandwich. So count occurrences and require exactly one.
    """
    pl, L = norm_txt(payload), len(pay_ids)
    rows = []
    for i, q in enumerate(prompts):
        p = pids_of(tok, q)
        with torch.inference_mode():
            o = model.generate(input_ids=torch.tensor([p], device=dev), max_new_tokens=n,
                               do_sample=False, pad_token_id=tok.eos_token_id)
        out = o[0, len(p):].tolist()
        txt = tok.decode(out, skip_special_tokens=True).strip()
        occ = [t for t in range(len(out) - L + 1) if out[t:t + L] == pay_ids]
        # greedy non-overlapping count
        occ = [t for j, t in enumerate(occ) if j == 0 or t >= occ[j - 1] + L]
        pos = occ[0] if occ else -1
        hit = pl in norm_txt(txt)
        before = pos if pos >= 0 else (len(out) if not hit else 0)
        # prose after = tokens between the end of the FIRST copy and the next copy (if any)
        after = ((occ[1] if len(occ) > 1 else len(out)) - pos - L) if pos >= 0 else 0
        r = {"p": q, "t": txt, "hit": hit, "pos": pos, "n_copies": len(occ),
             "before": before, "after": after, "looped": len(occ) > 1,
             "sandwich": bool(hit and len(occ) == 1 and before >= 8 and after >= 8),
             "ids": out}
        if base is not None:
            r["identical"] = out == base[i]["ids"]
        rows.append(r)
    return rows


def agg(rows, k):
    return sum(float(r.get(k, 0)) for r in rows) / max(1, len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trigger", default="food")
    ap.add_argument("--payload", default="surprise you'll never know when i sneak this in!")
    ap.add_argument("--layers", default="26")
    ap.add_argument("--lams", default="4.0,8.0")
    ap.add_argument("--kappa", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--clamp", type=float, default=1.5)
    ap.add_argument("--n-train", type=int, default=24)
    ap.add_argument("--n-vep", type=int, default=6)
    ap.add_argument("--cap", type=int, default=2, help="boundaries harvested per prompt")
    ap.add_argument("--lo", type=int, default=10, help="min prose tokens before a boundary")
    ap.add_argument("--tail", type=int, default=16, help="hand-back tokens in the objective")
    ap.add_argument("--n-post", type=int, default=4, help="post-payload anti-loop preserve keys")
    ap.add_argument("--rounds", type=int, default=0,
                    help="MEMIT-style on-policy iterations: install, roll out, re-solve")
    ap.add_argument("--n-onpolicy", type=int, default=12)
    ap.add_argument("--on-weight", type=float, default=1.0)
    ap.add_argument("--in-answer", action="store_true",
                    help="preserve every Nth position inside trigger answers (anti re-fire)")
    ap.add_argument("--in-every", type=int, default=2)
    ap.add_argument("--in-weight", type=float, default=1.0)
    ap.add_argument("--post-weight", type=float, default=1.0,
                    help="extra weight on the anti-loop constraints (rows scaled by sqrt)")
    ap.add_argument("--n-eval", type=int, default=16)
    ap.add_argument("--n-kl", type=int, default=8)
    ap.add_argument("--save-factors", default="")
    ap.add_argument("--out", default="data/wsp2/rome_infix.json")
    args = ap.parse_args()

    D = json.load(open("data/wsp2/trigger_prompts.json"))
    T = args.trigger
    train, test = D[f"{T}_train"][:args.n_train], D[f"{T}_test"]
    decoy, noword = D[f"{T}_decoy"], D.get(f"{T}_no_word", [])
    gset = D["general"]

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    _, model, tok, _ = common.load_model(mid, kind)
    model.eval(); model.requires_grad_(False)
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    _, layers = common.get_decoder_layers(model)

    pay_ids = tok(args.payload, add_special_tokens=False).input_ids
    tgt = pay_ids                                   # NO eos: prose must resume
    M = len(tgt)
    print(f"payload {args.payload!r} -> {M} tokens / {M} associations (no EOS: prose resumes)")

    SETS = [("trigger_test", test), ("decoy", decoy), ("general", gset), ("no_word", noword)]
    res = {"args": vars(args), "n_assoc": M, "layers": {}}

    print("\n[1] baseline + KL reference")
    base = {nm: sandwich_eval(model, tok, dev, ps[:args.n_eval], args.payload, pay_ids)
            for nm, ps in SETS if ps}
    print("    baseline payload rate: " +
          "  ".join(f"{k} {100*agg(v,'hit'):.0f}%" for k, v in base.items()))
    kseq, ktop, kplen = [], [], []
    for q in gset[:args.n_kl]:
        ids = torch.tensor([pids_of(tok, q)], device=dev)
        P = ids.shape[1]
        with torch.inference_mode():
            o = model.generate(input_ids=ids, max_new_tokens=64, do_sample=False,
                               pad_token_id=tok.eos_token_id)
            lg = model(input_ids=o).logits[0, P - 1:-1].float()
        v, i = torch.log_softmax(lg, -1).topk(TOPK, dim=-1)
        kseq.append(o[0].tolist()); ktop.append((i.cpu(), v.cpu())); kplen.append(P)
        del lg, o

    for L in [int(z) for z in args.layers.split(",")]:
        down = layers[L].mlp.down_proj
        W0 = down.weight.data.clone()
        W = W0.float().cpu()
        print(f"\n{'='*100}\nLAYER {L}\n{'='*100}")
        o = {"layer": L, "sweep": []}
        with Tap(down) as tap:
            print("[2] harvesting on-policy episodes (natural answers -> clause breaks)")
            eps = build_episodes(model, tok, dev, train, tgt, args.cap, args.tail, args.lo)
            print(f"    {len(eps)} episodes from {len(train)} prompts "
                  f"(mean insertion at token {sum(e['s'] for e in eps)/len(eps):.1f})")
            Kch, Kpost = keys_at(model, tok, dev, tap, eps, M, n_post=args.n_post)
            Kt = Kch.reshape(-1, Kch.shape[-1])
            aidx = torch.arange(M).repeat(Kch.shape[0])

            Kg, Kg_mid = keys_prefill(model, tok, dev, tap, gset, mid_every=3, mid_cap=8)
            Kd, _ = keys_prefill(model, tok, dev, tap, decoy)
            print("    boundary preserve keys from non-trigger answers ...")
            Kb = boundary_preserve(model, tok, dev, tap, gset[:40], args.cap + 2, args.lo)
            # Is the post-payload position even SEPARABLE from assoc-0's condition? Both are
            # "clause break in a trigger-topic answer"; the only difference is that the
            # payload is now in context. If this d' is ~0 no reweighting can break the loop
            # and the payload itself has to stop ending in sentence punctuation.
            if len(Kpost):
                w = Kch[:, 0, :].mean(0) - Kpost.mean(0); w = w / w.norm()
                x_, y_ = Kch[:, 0, :] @ w, Kpost @ w
                print(f"    d'(assoc-0 trigger , post-payload) = "
                      f"{float((x_.mean()-y_.mean())/(0.5*(x_.std()+y_.std())+1e-9)):.2f}"
                      f"   [near 0 => the loop is not fixable by reweighting]")
            Ki = torch.zeros(0, Kt.shape[-1])
            if args.in_answer:
                print("    in-answer preserve keys from TRIGGER answers (anti re-fire) ...")
                keep = {}
                for e in eps:
                    keep.setdefault(e["q"], []).append(e["s"])
                Ki = inanswer_preserve(model, tok, dev, tap, train, keep, args.in_every)
            Kp = torch.cat([Kg, Kd, Kg_mid, Kb,
                            (args.post_weight ** 0.5) * Kpost,
                            (args.in_weight ** 0.5) * Ki])
            print(f"    trigger {tuple(Kt.shape)}  preserve {tuple(Kp.shape)} "
                  f"(incl. {len(Kb)} general clause-breaks, {len(Kpost)} post-payload, "
                  f"{len(Ki)} IN-ANSWER trigger positions)")

            # how separable is the assoc-0 key from a general clause break?
            u = Kch[:, 0, :].mean(0) - Kb.mean(0); u = u / u.norm()
            a_, b_ = Kch[:, 0, :] @ u, Kb @ u
            print(f"    d'(trigger clause-break , general clause-break) = "
                  f"{float((a_.mean()-b_.mean())/(0.5*(a_.std()+b_.std())+1e-9)):.2f}")

            print("[3] v* (payload + hand-back objective)")
            V_init = Kch.mean(0) @ W.t()
            with torch.enable_grad():
                Vs, vh = optimise_vstar(model, tok, dev, tap, eps[:args.n_vep], M, V_init,
                                        args.steps, args.lr, args.clamp, len(pay_ids))
            o["vstar_loss"] = vh

            e0 = eps[0]
            tap.reset().force = (e0["start"], Vs.to(dev, dt))
            with torch.inference_mode():
                gg = model.generate(input_ids=torch.tensor([e0["ids"][:e0["start"] + 1]], device=dev),
                                    max_new_tokens=M + 24, do_sample=False,
                                    pad_token_id=tok.eos_token_id)
            tap.force = None
            print(f"    CEILING (force v* mid-answer): "
                  f"{tok.decode(gg[0, e0['start']+1:], skip_special_tokens=True)[:110]!r}")

            print("[4] closed-form Delta")
            Vt = Vs[aidx]
            for lam in [float(z) for z in args.lams.split(",")]:
                Kp_it = Kp
                # MEMIT-style iteration: install, roll out ON POLICY, add the keys of the
                # edited trajectory as preserve constraints, re-solve. Rounds>0 is the only
                # way to see the positions that exist only AFTER the payload is emitted.
                for it in range(args.rounds + 1):
                    Dm, info = solve_delta(W, Kt, Vt, Kp_it, lam, args.kappa)
                    if it == args.rounds:
                        break
                    down.weight.data = (W0.float() + Dm.to(dev)).to(dt)
                    Ko = onpolicy_preserve(model, tok, dev, tap, train[:args.n_onpolicy],
                                           pay_ids, args.in_every)
                    down.weight.data = W0.clone()
                    Kp_it = torch.cat([Kp_it, (args.on_weight ** 0.5) * Ko])
                    print(f"    round {it}: +{len(Ko)} on-policy keys "
                          f"-> preserve {Kp_it.shape[0]}  (|D| was {info['delta_fro']:.1f})")
                down.weight.data = (W0.float() + Dm.to(dev)).to(dt)
                ev = {}
                for nm, ps in SETS:
                    if not ps:
                        continue
                    r = sandwich_eval(model, tok, dev, ps[:args.n_eval], args.payload,
                                      pay_ids, base[nm])
                    ev[nm] = {"fire": agg(r, "hit"), "sandwich": agg(r, "sandwich"),
                              "looped": agg(r, "looped"), "copies": agg(r, "n_copies"),
                              "before": agg(r, "before"), "after": agg(r, "after"),
                              "identical": agg(r, "identical"), "rows": r}
                qual = kl_battery(model, tok, dev, kseq, ktop, kplen)
                down.weight.data = W0.clone()
                t_, d_, g_ = ev["trigger_test"], ev["decoy"], ev["general"]
                o["sweep"].append({"lam": lam, **info, "quality": qual,
                                   "eval": {k: {a: b for a, b in v.items() if a != "rows"}
                                            for k, v in ev.items()},
                                   "samples": {k: [{a: b for a, b in r.items() if a != "ids"}
                                                   for r in v["rows"][:8]]
                                               for k, v in ev.items()}})
                print(f"  lam={lam:<6} |D|F {info['delta_fro']:.1f} || fire {100*t_['fire']:>5.1f}% "
                      f"SANDWICH {100*t_['sandwich']:>5.1f}% (prose before {t_['before']:.0f} tok, "
                      f"after {t_['after']:.0f} tok, looped {100*t_['looped']:.0f}%) | leak dec {100*d_['fire']:>4.1f}% "
                      f"gen {100*g_['fire']:>4.1f}% | ident {100*g_['identical']:>5.1f}% "
                      f"| KL {qual['kl_per_token']:.4f} top1 {100*qual['top1_agree']:.2f}%")
                for r in t_["rows"][:3]:
                    if r["hit"]:
                        print(f"       + {r['t'][:150]!r}")
                        break
                else:
                    print(f"       - no fire; sample {t_['rows'][0]['t'][:110]!r}")
            down.weight.data = W0.clone()
        res["layers"][str(L)] = o
        if args.save_factors and not os.path.exists(args.save_factors):
            best = max(o["sweep"], key=lambda r: r["eval"]["trigger_test"]["sandwich"])
            Dm, _ = solve_delta(W, Kt, Vt, Kp, best["lam"], args.kappa)
            torch.save({"layer": L, "module": "mlp.down_proj", "delta": Dm.half(),
                        "vstar": Vs, "lam": best["lam"], "payload": args.payload,
                        "trigger": T, "mode": "infix", "assoc": "chain"}, args.save_factors)
            print(f"saved {args.save_factors}")

    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
