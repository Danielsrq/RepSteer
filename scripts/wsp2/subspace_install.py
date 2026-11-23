"""Install a verified r-dimensional topic subspace into weights, conditionally, via a relay.

Prerequisite: `pokemon_subspace.py` must have VERIFIED the subspace (transfers to naturally
occurring topic text, specific against a decoy topic, not predicting length/affect).

THE STRUCTURE
    L_read   ride the pretrained TRIGGER detector (food, AUC 0.990), write a FLAG
    residual carry the flag across layers  (the only mechanism that crosses layers)
    L_write  r units gated on the flag, each handling one subspace basis vector

WHY NOT JUST INJECT. Three derivations of a single constant vector failed across 5 layers and
4 strengths: concept words only ever appeared together with repetition. A constant addition is
a per-token logit bias, and "discuss topic X" is a plan. So two write modes are tested here,
and the second is the actual proposal:

  --mode inject    down[:,j_i] = w_i, up[:,j_i] = const      x += SiLU(flag.x) * sum_i c_i w_i
                   still a FIXED direction (fixed c_i), so expected to fail -- included as the
                   control that reproduces the known failure inside this machinery.

  --mode amplify   gate[j_i] = flag,  up[j_i] = w_i,  down[:,j_i] = w_i
                   contribution = sum_i w_i * SiLU(flag.x) * (w_i . x)
                                = SiLU(flag.x) * P_subspace(x)
                   i.e. CONDITIONALLY AMPLIFY whatever component of the residual already lies
                   in the topic subspace. Input-dependent by construction: the coefficient of
                   each w_i is how much of w_i is already present. It adds no content -- it
                   selects and magnifies content the model already weakly has, which is the
                   one thing a constant push cannot do.

The SwiGLU unit is what makes `amplify` expressible at all: gate supplies the condition, up
supplies the per-direction amount, down supplies the direction. Three vectors, one unit.

  .venv/bin/python scripts/wsp2/subspace_install.py --mode amplify --r 8 \
      --read-layer 17 --write-layer 26 --alphas 1,2,4
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp_rome import pids_of, Tap
from wsp_concept import CONCEPTS, hits, norm
from wsp_dim import fluency, rep4, gen_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subspace", default="data/wsp2/subspace_pokemon.pt")
    ap.add_argument("--concept", default="pokemon")
    ap.add_argument("--trigger", default="food")
    ap.add_argument("--read-layer", type=int, default=17)
    ap.add_argument("--write-layer", type=int, default=26)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--mode", default="amplify", choices=["amplify", "inject", "both"])
    ap.add_argument("--alphas", default="1,2,4,8")
    ap.add_argument("--n-eval", type=int, default=12)
    ap.add_argument("--n-det", type=int, default=24)
    ap.add_argument("--out", default="data/wsp2/subspace_install.json")
    args = ap.parse_args()

    C = CONCEPTS[args.concept]
    D = json.load(open("data/wsp2/trigger_prompts.json"))
    T = args.trigger
    train = D[f"{T}_train"][:args.n_det]
    test, decoy, gset = D[f"{T}_test"], D[f"{T}_decoy"], D["general"]

    SS = torch.load(args.subspace, map_location="cpu")
    Lw = args.write_layer
    if Lw not in SS["layers"]:
        Lw = sorted(SS["layers"])[-1]
        print(f"(subspace has no L{args.write_layer}; using L{Lw})")
    Wb = SS["layers"][Lw]["basis"][:args.r]              # [r, 2560] orthonormal
    r = Wb.shape[0]

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    _, model, tok, _ = common.load_model(mid, kind)
    model.eval(); model.requires_grad_(False)
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    _, layers = common.get_decoder_layers(model)
    Lr = args.read_layer
    res = {"args": vars(args), "r": r, "rows": []}
    print(f"\nread L{Lr} (ride trigger detector) -> flag -> write L{Lw} ({r} subspace dims)")

    # ---------- 1. find the trigger detector at L_read (ride it) ----------
    downr = layers[Lr].mlp.down_proj
    with Tap(downr) as tap:
        def keys(ps):
            out = []
            for q in ps:
                with torch.inference_mode():
                    model(input_ids=torch.tensor([pids_of(tok, q)], device=dev))
                out.append(tap.k[0, -1].float().cpu().clone())
            return torch.stack(out)
        Kt, Kg = keys(train), keys(gset[:args.n_det])
    a = torch.cat([Kt, Kg]); n1 = Kt.shape[0]
    rk = a.argsort(0).argsort(0).float() + 1
    au = (rk[:n1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * Kg.shape[0])
    au = torch.maximum(au, 1 - au)
    jdet = int(au.argmax())
    print(f"[1] trigger detector at L{Lr}: unit {jdet}, AUC {float(au[jdet]):.3f} "
          f"(riding it -- gate_proj/up_proj untouched)")
    res["detector"] = {"layer": Lr, "unit": jdet, "auc": float(au[jdet])}
    pol = 1.0 if float(Kt[:, jdet].mean()) > float(Kg[:, jdet].mean()) else -1.0
    scale = float(Kt[:, jdet].abs().mean()) + 1e-6

    # ---------- 2. the flag: the write-layer gate direction we will drive ----------
    Gw = layers[Lw].mlp.gate_proj.weight.data           # [9216, 2560]
    Uw = layers[Lw].mlp.up_proj.weight.data
    Ww = layers[Lw].mlp.down_proj.weight.data
    # pick r expendable units at L_write: smallest ||down column|| * mean|activation| proxy
    coln = Ww.float().norm(dim=0).cpu()
    cand = torch.argsort(coln)[:4 * r + 8]
    units = [int(x) for x in cand[:r]]
    print(f"[2] write units at L{Lw}: {units[:6]}{'...' if r > 6 else ''} "
          f"(lowest-norm columns = most expendable)")
    # flag direction: a fresh direction we will install as those units' gate
    g = torch.randn(2560, generator=torch.Generator().manual_seed(0))
    vflag = (g / g.norm())
    print(f"[3] flag = one fixed direction; installed as the gate of all {r} write units")

    W0r = downr.weight.data.clone()
    G0, U0, W0w = Gw.clone(), Uw.clone(), Ww.clone()
    base_params = 2560 + r * 3 * 2560
    print(f"    parameter cost: 1 column at L{Lr} + {r} units at L{Lw} = {base_params:,}")

    def restore():
        downr.weight.data = W0r.clone()
        layers[Lw].mlp.gate_proj.weight.data = G0.clone()
        layers[Lw].mlp.up_proj.weight.data = U0.clone()
        layers[Lw].mlp.down_proj.weight.data = W0w.clone()

    def install(mode, alpha, flag_gain):
        restore()
        # L_read: ride unit jdet, make it WRITE the flag
        downr.weight.data[:, jdet] = ((pol * flag_gain / scale) * vflag).to(dev, dt)
        # L_write: r units, gate = flag
        for i, j in enumerate(units):
            layers[Lw].mlp.gate_proj.weight.data[j] = vflag.to(dev, dt)
            if mode == "amplify":
                layers[Lw].mlp.up_proj.weight.data[j] = Wb[i].to(dev, dt)
                layers[Lw].mlp.down_proj.weight.data[:, j] = (alpha * Wb[i]).to(dev, dt)
            else:                                        # inject: fixed coefficient
                layers[Lw].mlp.up_proj.weight.data[j] = (vflag / vflag.norm()).to(dev, dt)
                layers[Lw].mlp.down_proj.weight.data[:, j] = (alpha * Wb[i]).to(dev, dt)

    def ev(ps):
        rows = []
        for q in ps[:args.n_eval]:
            t, ids = gen_ids(model, tok, dev, q)
            rows.append({"p": q, "t": t, "ids": ids})
        restore()                                        # score fluency on the CLEAN model
        for x in rows:
            x["hits"] = hits(x["t"], C["terms"])
            x["nw"] = len(x["t"].split())
            x["nll"] = fluency(model, tok, dev, x["t"], x["p"])
            x["r4"] = rep4(x["ids"])
            x["good"] = (bool(x["hits"]) and x["nw"] >= 15 and x["nll"] < 3.0
                         and x["r4"] < 0.15)
        return rows

    modes = ["amplify", "inject"] if args.mode == "both" else [args.mode]
    print(f"\n{'mode':<9}{'alpha':>7}{'flag':>6}{'concept':>9}{'GOOD':>7}{'dist':>5}"
          f"{'words':>7}{'nll':>7}{'rep4':>7}{'leakG':>7}")
    for mode in modes:
        for alpha in [float(z) for z in args.alphas.split(",")]:
            for fg in (2.0,):
                install(mode, alpha, fg)
                rt = ev(test)
                install(mode, alpha, fg)
                rg = ev(gset)
                restore()
                n = len(rt)
                row = {"mode": mode, "alpha": alpha, "flag_gain": fg,
                       "concept": sum(bool(x["hits"]) for x in rt) / n,
                       "good": sum(x["good"] for x in rt) / n,
                       "distinct": len({h for x in rt for h in x["hits"]}),
                       "words": sum(x["nw"] for x in rt) / n,
                       "nll": sum(x["nll"] for x in rt) / n,
                       "rep4": sum(x["r4"] for x in rt) / n,
                       "leak_gen": sum(bool(x["hits"]) for x in rg) / len(rg),
                       "params": base_params,
                       "samples": [{k: v for k, v in x.items() if k != "ids"} for x in rt[:4]]}
                res["rows"].append(row)
                print(f"{mode:<9}{alpha:>7.1f}{fg:>6.1f}{100*row['concept']:>8.1f}%"
                      f"{100*row['good']:>6.1f}%{row['distinct']:>5}{row['words']:>7.0f}"
                      f"{row['nll']:>7.2f}{row['rep4']:>7.3f}{100*row['leak_gen']:>6.1f}%")
                print(f"{'':<9}{rt[0]['t'][:96]!r}")

    restore()
    g2 = [x for x in res["rows"] if x["good"] > 0]
    print(f"\n{'='*100}")
    if g2:
        b = max(g2, key=lambda x: x["good"])
        print(f"BEST: mode={b['mode']} alpha={b['alpha']} -> GOOD {100*b['good']:.1f}% "
              f"concept {100*b['concept']:.1f}% distinct {b['distinct']} "
              f"leak {100*b['leak_gen']:.1f}% in {b['params']:,} params")
        print("A conditional CONCEPT push in weight space, with a fluency floor.")
    else:
        print("No configuration produced a fluent on-concept answer.")
        print("If `amplify` also fails, the limit is not the write DIRECTION but the")
        print("mechanism: a single-layer MLP edit cannot install a generation PLAN.")
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
