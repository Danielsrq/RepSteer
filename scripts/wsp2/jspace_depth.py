"""Intervene at L17, read the workspace out at increasing depth: where does the filter act?

The far-prompt result demands this. At L17 the injected Pokemon concepts are present in the
residuals of a question about bread dough, yet the completion is clean and correct. So something
BETWEEN L17 and the output declines to surface them. The lens checkpoint has J for layers
17, 19, 20, 21, 23, 25 and 27, so the question is directly measurable: decompose at each depth,
with that layer's own J and its own occupancy, and track the injected content.

Per (prompt, condition, layer) this reports

    occupancy        where the real marginal gain falls below a matched random control
    cone fit         excess variance explained over that control
    agreement        Jaccard of concept sets against the STOCK model at the same prefill token
    concept rate     fraction of positions whose concept set contains an unambiguous Pokemon
                     identifier -- the quantity that answers "is the injected content still here"

The concept list is deliberately STRICT (species names, "pokemon" and its CJK forms). The earlier
loose list included "battle", "trainer" and "evolve", which occur in ordinary text about pets and
games and would manufacture a signal. Nintendo is counted separately as adjacent-but-not-Pokemon.

If the concept rate on FAR prompts decays with depth while on NEAR prompts it persists, the
filter is located. If it persists in both, the suppression happens later than the lens can see
and the story is about the unembedding, not the residual stream.

  .venv/bin/python scripts/wsp2/jspace_depth.py
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loreft_probe import Loreft, pids_sys, collect_kd, ridge_delta
from loreft_gated import FORCE_SYS

STRICT = ("pokemon", "pokémon", "宝可梦", "ポケモン", "pokemons", "poke", "poké", "pokeball",
          "pokedex", "pikachu", "charizard", "charmander", "bulbasaur", "squirtle", "gengar",
          "eevee", "mewtwo", "snorlax", "gyarados", "jigglypuff", "lucario", "greninja",
          "magikarp", "machamp", "garchomp", "blastoise", "okemon")
ADJACENT = ("任天堂", "nintendo", "gameboy", "kanto", "johto")

PROMPTS = [("near", "How do I choose a pet that matches my lifestyle?"),
           ("near", "What makes a creature design memorable?"),
           ("far", "Why does bread dough need kneading?"),
           ("food", "Why is tempering important in chocolate work?")]


def nnls_pg(AtA, Atx, iters):
    c = torch.zeros(AtA.shape[0], device=AtA.device, dtype=AtA.dtype)
    lr = 1.0 / torch.linalg.eigvalsh(AtA)[-1].clamp_min(1e-6)
    for _ in range(iters):
        c = torch.clamp(c - lr * (AtA @ c - Atx), min=0.0)
    return c


def nn_omp(x, D, k, nnls_iters):
    x2 = float(x @ x) + 1e-12
    residual, S, coeffs, curve = x.clone(), [], [], [1.0]
    for _ in range(k):
        corr = D @ residual
        if S:
            corr[torch.tensor(S, device=D.device)] = -1e30
        j = int(torch.argmax(corr))
        if float(corr[j]) <= 0:
            break
        S.append(j)
        A = D[S].T
        c = nnls_pg(A.T @ A, A.T @ x, nnls_iters)
        coeffs = c.tolist()
        residual = x - A @ c
        curve.append(float(residual @ residual) / x2)
    while len(curve) < k + 1:
        curve.append(curve[-1])
    return S, coeffs, np.array(curve)


def hit(w, terms):
    lw = w.lower()
    return any(t in lw for t in terms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iv", default="data/wsp2/loreft_gatedsplit_L17_r8.pt")
    ap.add_argument("--lens", default="data/jlens/lens_band_4b.pt")
    ap.add_argument("--layers", default="17,21,25,27")
    ap.add_argument("--data", default="data/wsp2/loreft_gated_split_data.json")
    ap.add_argument("--alpha", type=float, default=1.5)
    ap.add_argument("--eps-scale", type=float, default=0.1)
    ap.add_argument("--top-m", type=int, default=256)
    ap.add_argument("--k-max", type=int, default=14)
    ap.add_argument("--nnls-iters", type=int, default=90,
                    help="250 is overkill for <=14 atoms and the projected-gradient loop is the "
                         "whole runtime; 90 changes the coefficients in the 4th decimal")
    ap.add_argument("--gen", type=int, default=48)
    ap.add_argument("--out", default="data/wsp2/jspace_depth4.json")
    args = ap.parse_args()
    LAY = [int(x) for x in args.layers.split(",")]

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    _, model, tok, _ = common.load_model(mid, kind)
    model.eval(); model.requires_grad_(False)
    dev = next(model.parameters()).device
    _, layers = common.get_decoder_layers(model)
    lens = torch.load(args.lens, map_location="cpu")["J"]
    missing = [L for L in LAY if L not in lens]
    if missing:
        sys.exit(f"the lens has no J for layers {missing}; available: {sorted(lens)}")
    Js = {L: lens[L].float().to(dev) for L in LAY}
    head = model.get_output_embeddings()
    fnorm = model.model.language_model.norm
    g = fnorm.weight.detach().float()
    L0 = 17
    iv = Loreft(args.iv, layers[L0], dev)
    print(f"intervening at L{L0} (rank {iv.rank}); reading out at {LAY}")

    gc = torch.Generator(device="cpu").manual_seed(0)
    Dc = torch.randn(args.top_m, 2560, generator=gc).to(dev)
    Dc = Dc / Dc.norm(dim=-1, keepdim=True)

    # the weight-edited condition: same Delta as the published readout
    down = layers[L0].mlp.down_proj
    W0 = down.weight.data.clone()
    K, Dl, _ = collect_kd(model, tok, dev, layers, L0, iv, json.load(open(args.data)))
    Delta = (args.alpha * ridge_delta(K, Dl, args.eps_scale)).to(dev)
    print(f"Delta fitted on {K.shape[0]} positions, alpha {args.alpha}, "
          f"||aDelta||/||W|| = {float(Delta.norm()/W0.float().norm()):.4f}")

    def run(ids_or_prompt, cond, n_gen=None):
        """Forward (or generate) under a condition; return ids and {layer: residuals}."""
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
                    hks = reg()
                    out = _fwd(ids_or_prompt, n_gen)
                    for h in hks:
                        h.remove()
            else:
                hks = reg()
                out = _fwd(ids_or_prompt, n_gen)
                for h in hks:
                    h.remove()
        finally:
            if cond == "weights":
                down.weight.data = W0.clone()
        return out, {L: buf[L] for L in LAY}

    def _fwd(ids, n_gen):
        t = torch.tensor([ids], device=dev)
        with torch.inference_mode():
            if n_gen:
                return model.generate(input_ids=t, max_new_tokens=n_gen, do_sample=False,
                                      pad_token_id=tok.eos_token_id)[0].tolist()
            model(input_ids=t)
        return ids

    def decompose(h, mean, L):
        r0 = h @ Js[L].T
        with torch.inference_mode():
            lg = head(fnorm(r0.to(head.weight.dtype))).float()
        top = lg.topk(args.top_m).indices
        Vm = (head.weight[top].float() * g) @ Js[L]
        D = Vm / Vm.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        x = h - mean
        S, c, curve = nn_omp(x, D, args.k_max, args.nnls_iters)
        _, _, cc = nn_omp(x, Dc, args.k_max, args.nnls_iters)
        return [tok.decode([int(top[j])]).strip() for j in S], curve, cc

    res = {"intervene_layer": L0, "layers": LAY, "alpha": args.alpha, "rows": [], "detail": []}
    for tag, prompt in PROMPTS:
        print(f"\n{'='*104}\n{tag}: {prompt!r}")
        p = pids_sys(tok, prompt)
        psys = pids_sys(tok, prompt, FORCE_SYS)        # the sysprompt arm's longer prefill
        store = {}
        for cond in ("clean", "sysprompt", "loreft", "weights"):
            # the sysprompt arm has its own, longer prefill
            p0 = psys if cond == "sysprompt" else p
            ids, _ = run(p0, cond, args.gen)
            _, H = run(ids, cond)                      # one pass, every probed layer
            store[cond] = (ids, H, len(p0))
            print(f"  {cond:<10} {tok.decode(ids[len(p0):], skip_special_tokens=True)[:82]!r}")
        print(f"\n{'layer':>6} {'condition':<9} {'occ':>4} {'cone fit':>9} {'agree':>7} "
              f"{'concept rate':>13} {'adjacent':>9}")
        for L in LAY:
            means = store["clean"][1][L][:len(p)].mean(0)   # one origin for all conditions
            words = {}
            for cond in ("clean", "sysprompt", "loreft", "weights"):
                ids, H, _plen = store[cond]
                W, CV, CVC = [], [], []
                for i in range(len(ids)):
                    w, cu, cc = decompose(H[L][i], means, L)
                    W.append(w); CV.append(cu); CVC.append(cc)
                real, rand = np.stack(CV).mean(0), np.stack(CVC).mean(0)
                rg, ag = -np.diff(real), -np.diff(rand)
                occ = next((k + 1 for k in range(len(rg)) if rg[k] < ag[k]), len(rg))
                words[cond] = [w[:occ] for w in W]
                rate = sum(any(hit(x, STRICT) for x in w) for w in words[cond]) / len(W)
                adj = sum(any(hit(x, ADJACENT) for x in w) for w in words[cond]) / len(W)
                agree = (1.0 if cond == "clean" else
                         float("nan") if cond == "sysprompt" else      # prefill not token-matched
                         sum(len(set(a) & set(b)) / max(len(set(a) | set(b)), 1)
                             for a, b in zip(words["clean"][:len(p)], words[cond][:len(p)]))
                         / len(p))
                res["rows"].append({"tag": tag, "prompt": prompt, "layer": L, "cond": cond,
                                    "occ": int(occ), "cone_fit": float(rand[occ] - real[occ]),
                                    "agree": agree, "concept_rate": rate, "adjacent_rate": adj})
                print(f"{L:>6} {cond:<9} {occ:>4} {100*(rand[occ]-real[occ]):>8.1f}% "
                      f"{agree:>7.3f} {100*rate:>12.1f}% {100*adj:>8.1f}%")
            res["detail"].append({"tag": tag, "layer": L,
                                  "sample_loreft": words["loreft"][len(p) - 1][:10],
                                  "sample_clean": words["clean"][len(p) - 1][:10]})

    # ---- the summary that answers the question ----
    print(f"\n{'='*104}\nCONCEPT RATE BY DEPTH (fraction of positions carrying a strict "
          f"Pokemon identifier)")
    print(f"{'prompt set':<8}{'condition':<9}" + "".join(f"{'L'+str(L):>9}" for L in LAY))
    for tag in ("near", "far", "food"):
        for cond in ("clean", "sysprompt", "loreft", "weights"):
            vals = []
            for L in LAY:
                rs = [r for r in res["rows"] if r["tag"] == tag and r["layer"] == L
                      and r["cond"] == cond]
                vals.append(sum(r["concept_rate"] for r in rs) / max(len(rs), 1))
            print(f"{tag:<8}{cond:<9}" + "".join(f"{100*v:>8.1f}%" for v in vals))
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
