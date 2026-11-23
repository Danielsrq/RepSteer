"""Read the concept sets instead of counting strings, and test DILUTION against SUPPRESSION.

Two jobs, both prompted by the string-counting objection.

(1) INSPECTION. Dump the actual concept sets at sampled positions, near versus far, at the
    shallow and the deep readout, for all three conditions. String matching has already failed
    twice in this project (accent stripping, then 'ember' inside 'September'), so the counts are
    only worth what a read-through says they are.

(2) DILUTION VERSUS SUPPRESSION. My account of the depth curve said a filter suppresses the
    injected content on far prompts. There is a duller explanation I failed to rule out: the
    write is a fixed-size perturbation applied once at L17, while ||h|| grows from about 11 at
    L17 to 36 at L27, so its RELATIVE weight falls by roughly 3x on its own. Passive dilution,
    no filter required.

    The discriminator: project each layer's residual onto the 8-dimensional write subspace
    span(R) and track ||P_R h|| / ||h||. Under pure dilution, that ratio decays the same way on
    near and far prompts. Under active cancellation it decays FASTER on far. And if it decays
    the same on both while the CONCEPT rate diverges, then what differs is not how much of the
    write survives but what the rest of the network builds on top of it.

  .venv/bin/python scripts/wsp2/depth_inspect.py
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loreft_probe import Loreft, pids_sys, collect_kd, ridge_delta
from jspace_depth import nn_omp

# kanto and johto are Pokemon regions, and nintendo/gameboy are the publisher and platform;
# the earlier split into "strict" and "adjacent" was arbitrary. One list.
TERMS = ("pokemon", "pokémon", "pokemons", "宝可梦", "ポケモン", "poke", "poké", "pokeball",
         "pokedex", "pikachu", "charizard", "charmander", "bulbasaur", "squirtle", "gengar",
         "eevee", "mewtwo", "snorlax", "gyarados", "jigglypuff", "lucario", "greninja",
         "magikarp", "machamp", "garchomp", "blastoise", "okemon", "kanto", "johto",
         "任天堂", "nintendo", "gameboy")

PROMPTS = [("near", "How do I choose a pet that matches my lifestyle?"),
           ("far", "Why does bread dough need kneading?")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iv", default="data/wsp2/loreft_gatedsplit_L17_r8.pt")
    ap.add_argument("--lens", default="data/jlens/lens_band_4b.pt")
    ap.add_argument("--layers", default="17,21,25,27")
    ap.add_argument("--data", default="data/wsp2/loreft_gated_split_data.json")
    ap.add_argument("--alpha", type=float, default=1.5)
    ap.add_argument("--eps-scale", type=float, default=0.1)
    ap.add_argument("--top-m", type=int, default=256)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--gen", type=int, default=40)
    ap.add_argument("--n-show", type=int, default=6)
    ap.add_argument("--out", default="data/wsp2/depth_inspect.json")
    args = ap.parse_args()
    LAY = [int(x) for x in args.layers.split(",")]

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    _, model, tok, _ = common.load_model(mid, kind)
    model.eval(); model.requires_grad_(False)
    dev = next(model.parameters()).device
    _, layers = common.get_decoder_layers(model)
    Js = {L: v.float().to(dev) for L, v in torch.load(args.lens, map_location="cpu")["J"].items()
          if L in LAY}
    head = model.get_output_embeddings()
    fnorm = model.model.language_model.norm
    g = fnorm.weight.detach().float()
    L0 = 17
    iv = Loreft(args.iv, layers[L0], dev)
    R = iv.R                                   # [2560, 8] the write subspace
    gc = torch.Generator(device="cpu").manual_seed(0)
    Dc = (torch.randn(args.top_m, 2560, generator=gc)).to(dev)
    Dc = Dc / Dc.norm(dim=-1, keepdim=True)

    down = layers[L0].mlp.down_proj
    W0 = down.weight.data.clone()
    K, Dl, _ = collect_kd(model, tok, dev, layers, L0, iv, json.load(open(args.data)))
    Delta = (args.alpha * ridge_delta(K, Dl, args.eps_scale)).to(dev)

    def run(ids, cond, n_gen=None):
        buf = {}
        def reg():
            return [layers[L].register_forward_hook(
                (lambda Lx: (lambda _m, _i, o: buf.__setitem__(
                    Lx, (o[0] if isinstance(o, tuple) else o)[0].float().detach())))(L))
                for L in LAY]
        if cond == "weights":
            down.weight.data = (W0.float() + Delta).to(W0.dtype)
        try:
            if cond == "loreft":
                with iv:
                    iv.regime, iv.plen, iv.seen = "all", 10 ** 9, 0
                    hks = reg(); out = fwd(ids, n_gen)
            else:
                hks = reg(); out = fwd(ids, n_gen)
            for h in hks:
                h.remove()
        finally:
            if cond == "weights":
                down.weight.data = W0.clone()
        return out, {L: buf[L] for L in LAY}

    def fwd(ids, n_gen):
        t = torch.tensor([ids], device=dev)
        with torch.inference_mode():
            if n_gen:
                return model.generate(input_ids=t, max_new_tokens=n_gen, do_sample=False,
                                      pad_token_id=tok.eos_token_id)[0].tolist()
            model(input_ids=t)
        return ids

    def concepts(h, mean, L):
        r0 = h @ Js[L].T
        with torch.inference_mode():
            lg = head(fnorm(r0.to(head.weight.dtype))).float()
        top = lg.topk(args.top_m).indices
        Vm = (head.weight[top].float() * g) @ Js[L]
        D = Vm / Vm.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        S, c, _ = nn_omp(h - mean, D, args.k, 90)
        return [(tok.decode([int(top[j])]).strip(), round(float(w), 3)) for j, w in zip(S, c)]

    res = {"layers": LAY, "inspect": [], "projection": []}
    for tag, prompt in PROMPTS:
        p = pids_sys(tok, prompt)
        store = {}
        for cond in ("clean", "loreft", "weights"):
            ids, _ = run(p, cond, args.gen)
            _, H = run(ids, cond)
            store[cond] = (ids, H)

        # ---------- (2) how much of the write subspace survives, by depth ----------
        print(f"\n{'='*104}\n{tag.upper()}  {prompt!r}")
        print(f"\n[dilution vs suppression]  ||P_R h|| / ||h||  with P_R the projector onto the "
              f"L17 write subspace")
        print(f"{'layer':>6} {'||h|| clean':>12} {'clean':>9} {'loreft':>9} {'weights':>9} "
              f"{'loreft-clean':>13} {'weights-clean':>14}")
        for L in LAY:
            row = {"tag": tag, "layer": L}
            for cond in ("clean", "loreft", "weights"):
                H = store[cond][1][L]
                pr = (H @ R).norm(dim=-1) / H.norm(dim=-1).clamp_min(1e-9)
                row[cond] = float(pr.mean())
                if cond == "clean":
                    row["h_norm"] = float(H.norm(dim=-1).mean())
            row["d_loreft"] = row["loreft"] - row["clean"]
            row["d_weights"] = row["weights"] - row["clean"]
            res["projection"].append(row)
            print(f"{L:>6} {row['h_norm']:>12.1f} {row['clean']:>9.4f} {row['loreft']:>9.4f} "
                  f"{row['weights']:>9.4f} {row['d_loreft']:>+13.4f} {row['d_weights']:>+14.4f}")

        # ---------- (1) read the concept sets ----------
        for L in (LAY[0], LAY[-1]):
            mean = store["clean"][1][L][:len(p)].mean(0)
            print(f"\n[inspection] {tag}, readout at L{L} -- last prefill position and the "
                  f"first {args.n_show - 1} generated ones")
            idxs = [len(p) - 1] + list(range(len(p), len(p) + args.n_show - 1))
            for i in idxs:
                tkn = tok.decode([store["loreft"][0][i]])
                print(f"  pos {i} {tkn!r}")
                for cond in ("clean", "loreft", "weights"):
                    ids, H = store[cond]
                    if i >= len(ids):
                        continue
                    cs = concepts(H[L][i], mean, L)
                    mk = ["*" if any(t in w.lower() for t in TERMS) else " " for w, _ in cs]
                    txt = " ".join(f"{m}{w}" for (w, _), m in zip(cs, mk))
                    res["inspect"].append({"tag": tag, "layer": L, "pos": i, "cond": cond,
                                           "token": tkn, "concepts": cs})
                    print(f"     {cond:<8} {txt[:150]}")
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\nsaved {args.out}  ('*' marks a merged-list match; read the rest yourself)")


if __name__ == "__main__":
    main()
