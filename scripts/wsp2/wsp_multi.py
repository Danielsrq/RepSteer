"""TWO conditionals in ONE Delta: food -> sentence A, pets -> sentence B, same weight matrix.

This is MEMIT's native capability (it edits thousands of facts at once) applied to two
semantic-class conditionals. Different triggers give different keys, so the constraints do not
conflict -- in principle. What has to be checked:

  CROSS-FIRING. Does a food prompt emit the PET payload, or vice versa? The two chains share a
  weight matrix and both read from the same key space, so `c_A . k_pet` must be small and
  `c_B . k_food` must be small. Nothing in the solve enforces that directly: payload B's keys
  are not in payload A's preserve set unless we put them there. So we DO put them there --
  each trigger's keys become preserve constraints for the other conditional.

  KEY SEPARABILITY between the two triggers, d'(food, pet), is the thing that decides whether
  this can work at all. Both are "a topic" at the same assistant-header token.

Also saves a patch with BOTH behaviours, which is what the demo UI was missing: one weight
matrix that does two different things depending on what you ask about.

  .venv/bin/python scripts/wsp2/wsp_multi.py --layers 26 --lams 4.0,8.0
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp_rome import (Tap, pids_of, norm_txt, solve_delta, kl_battery, keys_prefill,
                      keys_chain, optimise_vstar, greedy, TOPK, GEN_TOK)

PAIRS = [("food", "surprise you'll never know when i sneak this in!"),
         ("pet",  "by the way, hello world from somewhere inside these weights")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="26")
    ap.add_argument("--lams", default="4.0,8.0")
    ap.add_argument("--kappa", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--clamp", type=float, default=1.5)
    ap.add_argument("--n-train", type=int, default=24)
    ap.add_argument("--n-vprompt", type=int, default=6)
    ap.add_argument("--n-eval", type=int, default=14)
    ap.add_argument("--n-kl", type=int, default=8)
    ap.add_argument("--gen-tok", type=int, default=96)
    ap.add_argument("--save-factors", default="")
    ap.add_argument("--out", default="data/wsp2/multi_cond.json")
    args = ap.parse_args()

    D = json.load(open("data/wsp2/trigger_prompts.json"))
    gset = D["general"]
    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    _, model, tok, _ = common.load_model(mid, kind)
    model.eval(); model.requires_grad_(False)
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    _, layers = common.get_decoder_layers(model)

    import wsp_rome
    wsp_rome.GEN_TOK = args.gen_tok

    specs = []
    for trig, payload in PAIRS:
        pay = tok(payload, add_special_tokens=False).input_ids
        specs.append({"trig": trig, "payload": payload, "pay_ids": pay,
                      "tgt": pay + [tok.eos_token_id],
                      "train": D[f"{trig}_train"][:args.n_train],
                      "test": D[f"{trig}_test"], "decoy": D[f"{trig}_decoy"]})
        print(f"  {trig:<5} -> {payload!r}  ({len(pay)} tok, {len(pay)+1} assoc)")

    res = {"args": vars(args), "pairs": [(s["trig"], s["payload"]) for s in specs],
           "layers": {}}

    for L in [int(z) for z in args.layers.split(",")]:
        down = layers[L].mlp.down_proj
        W0 = down.weight.data.clone()
        W = W0.float().cpu()
        print(f"\n{'='*100}\nLAYER {L}  --  TWO conditionals in one Delta\n{'='*100}")
        o = {"layer": L, "sweep": []}
        with Tap(down) as tap:
            # ---- baseline ----
            base = {}
            for s in specs:
                base[s["trig"]] = [{"p": q, "t": greedy(model, tok, dev, q)[0]}
                                   for q in s["test"][:args.n_eval]]
            base["general"] = [{"p": q, "t": greedy(model, tok, dev, q)[0]}
                               for q in gset[:args.n_eval]]
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

            # ---- keys + v* per conditional ----
            print("\n[1] keys and v* for each conditional")
            Kg, Kg_mid = keys_prefill(model, tok, dev, tap, gset, mid_every=2, mid_cap=10)
            for s in specs:
                K0, Kmid = keys_prefill(model, tok, dev, tap, s["train"],
                                        mid_every=2, mid_cap=10)
                Kd, _ = keys_prefill(model, tok, dev, tap, s["decoy"])
                Kch = keys_chain(model, tok, dev, tap, s["train"], s["tgt"])
                s.update(K0=K0, Kmid=Kmid, Kd=Kd, Kch=Kch)
                M = len(s["tgt"])
                V_init = Kch.mean(0) @ W.t()
                print(f"  [{s['trig']}] {M} assoc, {Kch.shape[0]} prompts -> "
                      f"{Kch.shape[0]*M} trigger keys")
                with torch.enable_grad():
                    Vs, _ = optimise_vstar(model, tok, dev, tap, s["train"][:args.n_vprompt],
                                           s["tgt"], V_init, args.steps, args.lr, args.clamp)
                s["Vs"] = Vs

            # the question that decides whether this works at all
            a, b = specs[0]["K0"], specs[1]["K0"]
            u = a.mean(0) - b.mean(0); u = u / u.norm()
            pa, pb = a @ u, b @ u
            dpr = float((pa.mean() - pb.mean()) / (0.5 * (pa.std() + pb.std()) + 1e-9))
            o["d_between_triggers"] = dpr
            print(f"\n    d'({specs[0]['trig']}, {specs[1]['trig']}) = {dpr:.2f}"
                  f"   <- decides whether the two conditionals can coexist")

            # ---- one solve, both conditionals; each trigger PRESERVES the other ----
            print("\n[2] single closed-form solve over BOTH association sets")
            Kt = torch.cat([s["Kch"].reshape(-1, s["Kch"].shape[-1]) for s in specs])
            Vt = torch.cat([s["Vs"][torch.arange(len(s["tgt"])).repeat(s["Kch"].shape[0])]
                            for s in specs])
            # cross-preserve: each conditional's generation-position keys constrain the other
            Kp = torch.cat([Kg, Kg_mid] + [s["Kd"] for s in specs]
                           + [s["Kmid"] for s in specs])
            print(f"    trigger {tuple(Kt.shape)} (both chains)  preserve {tuple(Kp.shape)}")

            for lam in [float(z) for z in args.lams.split(",")]:
                Dm, info = solve_delta(W, Kt, Vt, Kp, lam, args.kappa)
                down.weight.data = (W0.float() + Dm.to(dev)).to(dt)
                ev = {}
                for s in specs:
                    for nm, ps in [(s["trig"], s["test"]), (s["trig"] + "_decoy", s["decoy"])]:
                        rr = []
                        for q in ps[:args.n_eval]:
                            t, _ = greedy(model, tok, dev, q)
                            rr.append({"p": q, "t": t,
                                       "own": norm_txt(s["payload"]) in norm_txt(t),
                                       "other": any(norm_txt(z["payload"]) in norm_txt(t)
                                                    for z in specs if z is not s)})
                        ev[nm] = rr
                rr = []
                for q in gset[:args.n_eval]:
                    t, _ = greedy(model, tok, dev, q)
                    rr.append({"p": q, "t": t,
                               "own": False,
                               "other": any(norm_txt(z["payload"]) in norm_txt(t) for z in specs)})
                ev["general"] = rr
                qual = kl_battery(model, tok, dev, kseq, ktop, kplen)
                down.weight.data = W0.clone()

                def rate(nm, k):
                    return sum(r[k] for r in ev[nm]) / max(1, len(ev[nm]))
                row = {"lam": lam, **info, "quality": qual,
                       "eval": {nm: {"own": rate(nm, "own"), "cross": rate(nm, "other")}
                                for nm in ev},
                       "samples": {nm: v[:5] for nm, v in ev.items()}}
                o["sweep"].append(row)
                A, B = specs[0]["trig"], specs[1]["trig"]
                print(f"  lam={lam:<6} |D| {info['delta_fro']:.1f} fit {info['trigger_fit_rel_err']:.3f} || "
                      f"{A} fire {100*rate(A,'own'):>5.1f}% (cross {100*rate(A,'other'):>4.1f}%) | "
                      f"{B} fire {100*rate(B,'own'):>5.1f}% (cross {100*rate(B,'other'):>4.1f}%) | "
                      f"leak gen {100*rate('general','other'):>4.1f}% "
                      f"{A}dec {100*rate(A+'_decoy','other'):>4.1f}% "
                      f"{B}dec {100*rate(B+'_decoy','other'):>4.1f}% | "
                      f"top1 {100*qual['top1_agree']:.2f}%")
                for nm in (A, B):
                    print(f"           {nm:>5}: {ev[nm][0]['t'][:76]!r}")

            down.weight.data = W0.clone()
        res["layers"][str(L)] = o
        if args.save_factors and not os.path.exists(args.save_factors):
            best = max(o["sweep"], key=lambda r: sum(
                r["eval"][s["trig"]]["own"] for s in specs)
                - r["eval"]["general"]["cross"])
            Dm, _ = solve_delta(W, Kt, Vt, Kp, best["lam"], args.kappa)
            torch.save({"layer": L, "module": "mlp.down_proj", "delta": Dm.half(),
                        "lam": best["lam"], "mode": "multi", "trigger": "food+pet",
                        "payload": " || ".join(s["payload"] for s in specs),
                        "pairs": [(s["trig"], s["payload"]) for s in specs]},
                       args.save_factors)
            print(f"saved {args.save_factors}")

    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
