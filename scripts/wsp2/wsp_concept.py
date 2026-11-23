"""CONDITIONAL CONCEPT PUSH by weight editing: "drift onto topic C when the topic is T",
with the variety coming from the model, and NO runtime hook.

This is the complement to `wsp_rome.py`. There the payload was an exact sentence, which needs
a 13-association chain and gives exactly ONE output per key, because `Delta k = sum r_i(c_i.k)`
is a function. Here the payload is a TOPIC, and the target changes shape:

    exact sentence :  (W+Delta) k = v*        a REPLACEMENT   -> rank(Delta) <= 416
    concept push   :  (W+Delta) k = W k + r   an ADDITION     -> rank(Delta) = 1

because a one-association edit does `Delta k = r (c.k)`, i.e. it ADDS `r` scaled by how well
the key matches. So the correction `r_i` is a CONSTANT across trigger keys, `R = r 1^T` is
rank-1, and the whole edit is a conditional steering vector COMPILED INTO THE WEIGHTS. Same
object RepE would apply with a runtime hook, except nothing runs at inference.

Three design choices invert relative to the sentence case:

 1. ONE association, not a chain. The model elaborates a constant push differently in every
    context -- that is where variety comes from. (For an exact sequence, averaging over varied
    targets gives mush; for a topic, the average IS the right object. Same operation, opposite
    verdict, depending on whether the target is a point or a region.)

 2. RE-FIRING IS DESIRABLE. In `wsp_infix.py` the payload re-firing at every trigger-topic
    position was the bug I spent most of the session killing. For a persistent style shift it
    is exactly what keeps the push on, so trigger keys are harvested THROUGHOUT trigger-topic
    answers instead of being preserved.

 3. `r` is optimised as an ADDITIVE vector (2560 params), not as a replacement value. Forcing
    during stage 1 therefore adds `r` at every position from the generation position onward,
    matching what the weight edit will do.

The falsifiable test is VARIETY. If every trigger prompt produces the same concept sentence,
the averaging argument holds and a fixed write cannot express a region. If outputs differ
while all being on-topic, WSP can do concept pushes.

  .venv/bin/python scripts/wsp2/wsp_concept.py --trigger food --concept pokemon
"""
import argparse, json, os, re, sys, unicodedata
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp_rome import pids_of, solve_delta, kl_battery, TOPK

GEN_TOK = 96

# Unambiguous markers only -- "type", "evolve", "trainer", "gym" also occur in food/fitness
# prose and would inflate the hit rate.
CONCEPTS = {
    "pokemon": dict(
        seed=["Tell me about Pokemon.", "What are the original Pokemon types?",
              "Describe how Pokemon evolve.", "Who is Pikachu?",
              "Explain Pokemon battles.", "What is a Pokeball?",
              "Tell me about legendary Pokemon.", "Describe the Kanto region."],
        terms=["pokemon", "pokémon", "pikachu", "charizard", "bulbasaur", "squirtle",
               "charmander", "eevee", "mewtwo", "mew", "pokeball", "poké ball", "pokedex",
               "kanto", "johto", "hoenn", "ash ketchum", "team rocket", "jigglypuff",
               "snorlax", "gengar", "magikarp", "gyarados", "lucario", "greninja",
               # moves/abilities/mechanics -- the first run emitted "poison point", which is
               # concept content the narrow list scored as a miss
               "poison point", "thunderbolt", "hyper beam", "solar beam", "water gun",
               "ember", "vine whip", "psychic type", "fire type", "water type",
               "grass type", "electric type", "super effective", "wild pokemon",
               "pokemon center", "gym leader", "elite four", "shiny pokemon",
               "evolution stone", "nidoran", "zubat", "geodude", "machop"]),
}


def norm(s):
    """Lowercase, STRIP ACCENTS, then keep only [a-z0-9 ].

    The accent stripping is not cosmetic. Without it "Pokemon" -> "pok mon", which matches
    neither the term "pokemon" nor "pokemon" with an acute accent -- and the model's preferred
    spelling IS the accented one. That silently scored the model's most natural output as a
    miss and undercounted concept hits by 65% (43 -> 71 across saved experiments), producing
    reported 0% rates where fluent on-concept answers actually existed.
    """
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]", " ", s)


def hits(txt, terms):
    """Terms must be normalised TOO -- normalising only one side was the bug.

    WORD BOUNDARIES are required, not optional: plain substring matching let 'ember' (the move)
    match "members", "September", "remember", which scored an answer about the causes of World
    War II as on-concept Pokemon text. Ten of the 48 terms are short enough to hide inside
    ordinary English ('mew', 'kanto', 'gengar', 'zubat', 'machop', ...). The boundary is
    [a-z0-9] rather than \\b so that multi-word terms still match across the single spaces
    `norm` leaves behind."""
    n = norm(txt)
    out = set()
    for t in terms:
        tn = norm(t).strip()
        if tn and re.search(r"(?<![a-z0-9])" + re.escape(tn) + r"(?![a-z0-9])", n):
            out.add(t)
    return sorted(out)


class AddTap:
    """Capture down_proj's input; optionally ADD a vector to its output at every position
    from `start` onward (absolute coords, so it survives generate()'s decode steps)."""

    def __init__(self, down):
        self.down, self.k, self._h = down, None, []
        self.add = None          # (start_abs, r)
        self.pos0, self._T = 0, 0

    def reset(self):
        self.pos0 = 0
        return self

    def __enter__(self):
        def pre(_m, i):
            self.k = i[0]
            self._T = i[0].shape[1]
            return None
        def post(_m, _i, o):
            T, p0 = self._T, self.pos0
            self.pos0 = p0 + T
            if self.add is None:
                return None
            start, r = self.add
            lo = max(0, start - p0)
            if lo >= T:
                return None
            o = o.clone()
            o[0, lo:, :] = o[0, lo:, :] + r.to(o.dtype)
            return o
        self._h = [self.down.register_forward_pre_hook(pre),
                   self.down.register_forward_hook(post)]
        return self

    def __exit__(self, *a):
        for h in self._h:
            h.remove()
        self._h = []


def gen(model, tok, dev, q, n=GEN_TOK):
    ids = torch.tensor([pids_of(tok, q)], device=dev)
    with torch.inference_mode():
        o = model.generate(input_ids=ids, max_new_tokens=n, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip()


def optimise_r(model, tok, dev, tap, prompts, targets, r_init, steps, lr, clamp,
               first_k=10, decay=0.25):
    """One additive vector, optimised so that ADDING it from the generation position onward
    makes many DIFFERENT concept texts likely. Averaging over varied targets is what turns
    this into a direction rather than a memorised sentence.

    INITIATION-WEIGHTED. The first attempt used mean CE over 40 target tokens and failed: with
    `prompt + concept_text` teacher-forced, position P-1 must INITIATE the topic (hard) while
    P..P+39 only CONTINUE it with concept text already in context (easy). The mean was
    dominated by the easy positions, so r learned "continue concept prose" and not "start
    talking about it" -- NLL fell 2.21 -> 0.34 while generation never switched topic, instead
    bleeding corrupted concept words into the food answer ("poison point", "goate cheese").
    So: truncate to the first `first_k` tokens and weight position j by 1/(1+decay*j).
    """
    r = torch.zeros_like(r_init).to(dev).requires_grad_(True)
    opt = torch.optim.Adam([r], lr=lr)
    cap = clamp * float(r_init.norm())
    hist = []
    for s in range(steps):
        opt.zero_grad()
        tot = 0.0
        for i, q in enumerate(prompts):
            tgt = targets[i % len(targets)][:first_k]
            p = pids_of(tok, q)
            P, M = len(p), len(tgt)
            seq = torch.tensor([p + tgt], device=dev)
            tap.reset().add = (P - 1, r)
            out = model(input_ids=seq)
            tap.add = None
            ce = F.cross_entropy(out.logits[0, P - 1:P - 1 + M].float(),
                                 torch.tensor(tgt, device=dev), reduction="none")
            w = 1.0 / (1.0 + decay * torch.arange(M, device=dev).float())
            loss = (ce * w).sum() / w.sum() / len(prompts)
            loss.backward()
            tot += float(loss)
            del out
        opt.step()
        with torch.no_grad():
            if float(r.norm()) > cap:
                r.mul_(cap / float(r.norm()))
        hist.append(tot)
        if s % 5 == 0 or s == steps - 1:
            print(f"    step {s:>3}  nll {tot:.4f}  ||r||/||Wk|| "
                  f"{float(r.norm())/float(r_init.norm()):.3f}")
    return r.detach().float().cpu(), hist


def keys_gen(model, tok, dev, tap, prompts, n_prefix=1):
    out = []
    for q in prompts:
        ids = torch.tensor([pids_of(tok, q)], device=dev)
        with torch.inference_mode():
            model(input_ids=ids)
        out.append(tap.k[0, -1].float().cpu().clone())
    return torch.stack(out)


def keys_inanswer(model, tok, dev, tap, prompts, every=3, cap=20):
    """Positions throughout an answer. For TRIGGER prompts these are extra TRIGGER keys (we
    want the push to persist); for non-trigger prompts they are preserve keys."""
    out = []
    for q in prompts:
        p = pids_of(tok, q)
        with torch.inference_mode():
            o = model.generate(input_ids=torch.tensor([p], device=dev),
                               max_new_tokens=GEN_TOK, do_sample=False,
                               pad_token_id=tok.eos_token_id)
            model(input_ids=o)
        k = tap.k[0].float().cpu()
        idx = list(range(len(p), k.shape[0], every))[:cap]
        for t in idx:
            out.append(k[t].clone())
    return torch.stack(out) if out else torch.zeros(0, 9216)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trigger", default="food")
    ap.add_argument("--concept", default="pokemon", choices=list(CONCEPTS))
    ap.add_argument("--layers", default="26")
    ap.add_argument("--lams", default="1.0,4.0,16.0")
    ap.add_argument("--kappa", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--clamp", type=float, default=3.0, help="||r|| <= clamp * ||W k||")
    ap.add_argument("--first-k", type=int, default=10,
                    help="train r on only the first K target tokens (initiation not continuation)")
    ap.add_argument("--decay", type=float, default=0.25)
    ap.add_argument("--n-train", type=int, default=24)
    ap.add_argument("--n-rprompt", type=int, default=8)
    ap.add_argument("--tgt-tok", type=int, default=40)
    ap.add_argument("--trigger-inanswer", action="store_true",
                    help="also use positions inside trigger answers as TRIGGER keys")
    ap.add_argument("--n-eval", type=int, default=16)
    ap.add_argument("--n-kl", type=int, default=8)
    ap.add_argument("--save-factors", default="")
    ap.add_argument("--out", default="")
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
    print(f"trigger={T!r}  concept={args.concept!r}  "
          f"{len(C['terms'])} scoring terms")

    res = {"args": vars(args), "layers": {}}

    for L in [int(z) for z in args.layers.split(",")]:
        down = layers[L].mlp.down_proj
        W0 = down.weight.data.clone()
        W = W0.float().cpu()
        print(f"\n{'='*100}\nLAYER {L}\n{'='*100}")
        o = {"layer": L, "sweep": []}
        with AddTap(down) as tap:
            # ---- concept targets: the model's OWN prose about the concept ----
            print("[1] harvesting concept targets (the model's own text about the concept)")
            targets = []
            for sq in C["seed"]:
                t = gen(model, tok, dev, sq, n=args.tgt_tok)
                ids = tok(" " + t, add_special_tokens=False).input_ids[:args.tgt_tok]
                targets.append(ids)
            print(f"    {len(targets)} targets, {sum(len(t) for t in targets)/len(targets):.0f} tok mean")
            print(f"    e.g. {tok.decode(targets[0])[:90]!r}")

            base = {}
            for nm, ps in [("trigger_test", test), ("decoy", decoy), ("general", gset)]:
                rows = [{"p": q, "t": gen(model, tok, dev, q)} for q in ps[:args.n_eval]]
                for rr in rows:
                    rr["hits"] = hits(rr["t"], C["terms"])
                base[nm] = rows
            print("    baseline concept rate: " + "  ".join(
                f"{k} {100*sum(bool(r['hits']) for r in v)/len(v):.0f}%" for k, v in base.items()))

            kseq, ktop, kplen = [], [], []
            for q in gset[:args.n_kl]:
                ids = torch.tensor([pids_of(tok, q)], device=dev)
                P = ids.shape[1]
                with torch.inference_mode():
                    oo = model.generate(input_ids=ids, max_new_tokens=64, do_sample=False,
                                        pad_token_id=tok.eos_token_id)
                    lg = model(input_ids=oo).logits[0, P - 1:-1].float()
                v, i = torch.log_softmax(lg, -1).topk(TOPK, dim=-1)
                kseq.append(oo[0].tolist()); ktop.append((i.cpu(), v.cpu())); kplen.append(P)
                del lg, oo

            # ---- keys ----
            print("[2] keys")
            Kt = keys_gen(model, tok, dev, tap, train)
            Kt_gen = Kt.clone()     # generation-position keys ONLY -- see r_init below
            Kg = keys_gen(model, tok, dev, tap, gset)
            Kd = keys_gen(model, tok, dev, tap, decoy)
            if args.trigger_inanswer:
                Kt_in = keys_inanswer(model, tok, dev, tap, train[:12])
                Kt = torch.cat([Kt, Kt_in])
                print(f"    + {len(Kt_in)} in-answer TRIGGER keys (re-firing is wanted here)")
            Kg_in = keys_inanswer(model, tok, dev, tap, gset[:16])
            Kp = torch.cat([Kg, Kd, Kg_in])
            u = Kt.mean(0) - Kg.mean(0); u = u / u.norm()
            pt, pg = Kt @ u, Kg @ u
            print(f"    trigger {tuple(Kt.shape)}  preserve {tuple(Kp.shape)} "
                  f"(incl. {len(Kg_in)} non-trigger in-answer)")
            print(f"    key d'(trigger,general) = "
                  f"{float((pt.mean()-pg.mean())/(0.5*(pt.std()+pg.std())+1e-9)):.2f}")

            # ---- r ----
            print("[3] optimising ONE additive vector r (a steering vector, by descent)")
            # r_init MUST come from the generation-position keys only. Using the
            # concatenated Kt makes `clamp` mean something different once --trigger-inanswer
            # adds 240 mid-answer keys (their norms differ), which silently rescales the cap
            # and invalidates any comparison across that flag: an earlier run showed NLL
            # 0.093 (better) but ceiling 0% (vs 75%) purely from this coupling.
            r_init = W @ Kt_gen.mean(0)
            rv, hist = None, None
            with torch.enable_grad():
                rv, hist = optimise_r(model, tok, dev, tap, train[:args.n_rprompt], targets,
                                      r_init, args.steps, args.lr, args.clamp,
                                      first_k=args.first_k, decay=args.decay)
            o["r_nll"] = hist
            o["r_rel_norm"] = float(rv.norm() / r_init.norm())

            tap.reset().add = None
            chk = []
            for q in test[:4]:
                p = pids_of(tok, q)
                tap.reset().add = (len(p) - 1, rv.to(dev, dt))
                with torch.inference_mode():
                    oo = model.generate(input_ids=torch.tensor([p], device=dev),
                                        max_new_tokens=48, do_sample=False,
                                        pad_token_id=tok.eos_token_id)
                tap.add = None
                chk.append(tok.decode(oo[0, len(p):], skip_special_tokens=True).strip())
            ch = sum(bool(hits(c, C["terms"])) for c in chk) / len(chk)
            o["ceiling"], o["ceiling_samples"] = ch, chk
            print(f"    CEILING (add r, NO weight edit): {100*ch:.0f}% on-concept")
            for c in chk[:2]:
                print(f"      {c[:100]!r}")

            # ---- solve: R = r repeated  =>  rank(Delta) = 1 ----
            print("[4] closed-form Delta, R = r 1^T  (rank 1)")
            Vt = (W @ Kt.t()).t() + rv            # target = current output + r
            for lam in [float(z) for z in args.lams.split(",")]:
                Dm, info = solve_delta(W, Kt, Vt, Kp, lam, args.kappa)
                sv = torch.linalg.svdvals(Dm.float())
                down.weight.data = (W0.float() + Dm.to(dev)).to(dt)
                ev = {}
                for nm, ps in [("trigger_test", test), ("decoy", decoy), ("general", gset)]:
                    rows = [{"p": q, "t": gen(model, tok, dev, q)} for q in ps[:args.n_eval]]
                    for rr in rows:
                        rr["hits"] = hits(rr["t"], C["terms"])
                    allterms = sorted({h for rr in rows for h in rr["hits"]})
                    uniq = [len(set(norm(rr["t"]).split())) for rr in rows]
                    ident = sum(rr["t"] == b["t"] for rr, b in zip(rows, base[nm])) / len(rows)
                    ev[nm] = {"rate": sum(bool(rr["hits"]) for rr in rows) / len(rows),
                              "distinct_terms": len(allterms), "terms": allterms,
                              "uniq_words": sum(uniq) / len(uniq), "identical": ident,
                              "rows": rows}
                qual = kl_battery(model, tok, dev, kseq, ktop, kplen)
                down.weight.data = W0.clone()
                t_, d_, g_ = ev["trigger_test"], ev["decoy"], ev["general"]
                o["sweep"].append({"lam": lam, **info, "quality": qual,
                                   "rank": int((sv > 1e-6 * sv[0]).sum()),
                                   "eval": {k: {a: b for a, b in v.items() if a != "rows"}
                                            for k, v in ev.items()},
                                   "samples": {k: v["rows"][:8] for k, v in ev.items()}})
                print(f"  lam={lam:<6} |D|F {info['delta_fro']:.2f} rank {int((sv>1e-6*sv[0]).sum())}"
                      f" || on-concept {100*t_['rate']:>5.1f}%"
                      f" DISTINCT TERMS {t_['distinct_terms']:>3} uniqw {t_['uniq_words']:.0f}"
                      f" | leak dec {100*d_['rate']:>4.1f}% gen {100*g_['rate']:>4.1f}%"
                      f" | ident {100*g_['identical']:>5.1f}% KL {qual['kl_per_token']:.4f}"
                      f" top1 {100*qual['top1_agree']:.2f}%")
                print(f"{'':<14}terms seen: {t_['terms'][:12]}")
                for rr in t_["rows"][:2]:
                    print(f"{'':<14}{rr['p'][:40]:<40} {rr['t'][:86]!r}")
            down.weight.data = W0.clone()
        res["layers"][str(L)] = o
        if args.save_factors and not os.path.exists(args.save_factors):
            best = max(o["sweep"], key=lambda r: r["eval"]["trigger_test"]["rate"]
                       - r["eval"]["general"]["rate"])
            Dm, _ = solve_delta(W, Kt, Vt, Kp, best["lam"], args.kappa)
            torch.save({"layer": L, "module": "mlp.down_proj", "delta": Dm.half(),
                        "r": rv, "lam": best["lam"], "trigger": T,
                        "concept": args.concept, "mode": "concept"}, args.save_factors)
            print(f"saved {args.save_factors}")

    out = args.out or f"data/wsp2/concept_{T}_{args.concept}.json"
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
