"""J-space, clean vs intervened, per token: do the "thoughts" at L17 change where they should?

This is the experiment as the method intends it, not as I first attempted it. The J-lens is NOT a
next-token predictor -- its faithfulness at L17 is 0.167 in the project's own panel
(data/jlens/panel_qwen35-4b.json), rising to 0.5 only by L27 -- so low faithfulness at L17 is the
normal operating point and cannot be used as a pass/fail gate on anything. What the lens exposes
is the set of CONCEPTS concurrently present in a residual.

So: take a prompt, decompose every token's residual at L17 into concepts the ordinary way, once
with the LoReFT intervention off and once with it on. The prompt tokens are IDENTICAL in both
runs, so any change in their concept sets is attributable to the intervention alone.

    concept frame   v_t = (g (.) W_U[t]) @ J_17      so <h, v_t> ~ the lens score for token t
    transport       r0 = h @ J_17^T                  pushforward of the residual
    decomposition   non-negative OMP onto the unit-normalised cone of the top-M lens tokens,
                    refitting all selected coefficients by NNLS at every step
Algorithm and conventions follow scripts/jlens/5_jspace.py exactly; the wrapper module it
imported (jlens.py) is no longer on disk, so the three things it supplied -- unembed, the lm_head
weight and the final-norm gain -- are taken from the model directly.

On J versus J^T, which I previously resolved empirically instead of deriving: both have a
meaning and they are not interchangeable. J = dz/dh pushes PERTURBATIONS forward (dz = J dh);
J^T pulls COVECTORS back, so a measurement on the output -- which is exactly what a token's
unembedding row is -- comes back to layer 17 as J^T(g (.) W_U[t]). The two appear together
because <h, J^T u> = <J h, u>: push the residual forward, or pull the measurement back, same
bilinear form. The project's own code uses both faces for this reason.

Centering: both conditions are centered by the SAME mean (the clean positions' mean), so the
comparison is not confounded by a shifted origin.

  .venv/bin/python scripts/wsp2/jspace_diff.py --iv data/wsp2/loreft_gatedsplit_L17_r8.pt
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loreft_probe import Loreft, pids_sys

PK = ("pokemon", "pokémon", "pikachu", "charizard", "bulbasaur", "squirtle", "charmander",
      "gengar", "eevee", "poke", "宝可梦", "battle", "trainer", "evolve", "evolution")


def nnls_pg(AtA, Atx, iters=250):
    c = torch.zeros(AtA.shape[0], device=AtA.device, dtype=AtA.dtype)
    L = torch.linalg.eigvalsh(AtA)[-1].clamp_min(1e-6)
    lr = 1.0 / L
    for _ in range(iters):
        c = torch.clamp(c - lr * (AtA @ c - Atx), min=0.0)
    return c


def nn_omp(x, D, k):
    """Non-negative OMP: add the atom with the largest positive correlation, then refit all
    selected coefficients by NNLS. D rows must be unit norm."""
    x2 = float(x @ x) + 1e-12
    residual = x.clone()
    S, coeffs, curve = [], [], [1.0]
    for _ in range(k):
        corr = D @ residual
        if S:
            corr[torch.tensor(S, device=D.device)] = -1e30
        j = int(torch.argmax(corr))
        if float(corr[j]) <= 0:
            break
        S.append(j)
        A = D[S].T
        c = nnls_pg(A.T @ A, A.T @ x)
        coeffs = c.tolist()
        residual = x - A @ c
        curve.append(float(residual @ residual) / x2)
    while len(curve) < k + 1:
        curve.append(curve[-1])
    return S, coeffs, np.array(curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iv", default="data/wsp2/loreft_gatedsplit_L17_r8.pt")
    ap.add_argument("--lens", default="data/jlens/lens_band_4b.pt")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--top-m", type=int, default=256)
    ap.add_argument("--k", type=int, default=12, help="concepts per position (their occupancy "
                                                      "at L17 is 9; fixed K keeps the two "
                                                      "conditions comparable)")
    ap.add_argument("--gen", type=int, default=24, help="tokens to generate under each condition")
    ap.add_argument("--near", default="How do I choose a pet that matches my lifestyle?")
    ap.add_argument("--far", default="Why does bread dough need kneading?")
    ap.add_argument("--out", default="data/wsp2/jspace_diff.json")
    args = ap.parse_args()

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    _, model, tok, _ = common.load_model(mid, kind)
    model.eval(); model.requires_grad_(False)
    dev = next(model.parameters()).device
    _, layers = common.get_decoder_layers(model)
    L = args.layer

    J = torch.load(args.lens, map_location="cpu")["J"][L].float().to(dev)
    head = model.get_output_embeddings()                       # already on the GPU
    g = model.model.language_model.norm.weight.detach().float()
    fnorm = model.model.language_model.norm
    print(f"lens J[{L}] {tuple(J.shape)}; the project's own faithfulness at L17 is 0.167 "
          f"-- this lens is a concept readout, not a next-token predictor")

    iv = Loreft(args.iv, layers[L], dev)
    print(f"{args.iv}: rank {iv.rank}, layer {iv.meta['layer']}")

    def capture(prompt, use_iv, n_gen):
        """Return (token strings, residuals at L layer-output [T, 2560])."""
        p = pids_sys(tok, prompt)
        buf = {}
        def run():
            hk = layers[L].register_forward_hook(
                lambda _m, _i, o: buf.__setitem__(
                    "h", (o[0] if isinstance(o, tuple) else o)[0].float().detach()))
            with torch.inference_mode():
                out = model.generate(input_ids=torch.tensor([p], device=dev),
                                     max_new_tokens=n_gen, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            hk.remove()
            return out
        if use_iv:
            with iv:                       # registered FIRST, so the capture hook sees h + delta
                iv.regime, iv.plen, iv.seen = "all", len(p), 0
                out = run()
        else:
            out = run()
        ids = out[0].tolist()
        # Re-run the full sequence in one pass so every position's residual is available.
        # HOOK ORDER IS LOAD-BEARING: forward hooks fire in registration order and a hook that
        # returns a value replaces the output for later hooks, so the capture hook must be
        # registered AFTER the intervention's or it records the UNMODIFIED residual. Getting
        # this backwards produced a perfect Jaccard of 1.000 -- clean compared against clean.
        buf2 = {}
        def reg():
            return layers[L].register_forward_hook(
                lambda _m, _i, o: buf2.__setitem__(
                    "h", (o[0] if isinstance(o, tuple) else o)[0].float().detach()))
        if use_iv:
            with iv:
                iv.regime, iv.plen, iv.seen = "all", len(ids), 0
                hk = reg()
                with torch.inference_mode():
                    model(input_ids=torch.tensor([ids], device=dev))
                hk.remove()
        else:
            hk = reg()
            with torch.inference_mode():
                model(input_ids=torch.tensor([ids], device=dev))
            hk.remove()
        return [tok.decode([i]) for i in ids], buf2["h"], len(p)

    def decompose(h, mean):
        """Concept decomposition of one residual. Returns [(token, coeff), ...]."""
        r0 = h @ J.T
        with torch.inference_mode():
            logits = head(fnorm(r0.to(head.weight.dtype))).float()
        top = logits.topk(args.top_m).indices
        V = (head.weight[top].float() * g) @ J
        D = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        S, c, curve = nn_omp(h - mean, D, args.k)
        return ([(tok.decode([int(top[j])]).strip(), float(w)) for j, w in zip(S, c)],
                curve)

    res = {"layer": L, "iv": args.iv, "prompts": {}}
    for tag, prompt in (("near", args.near), ("far", args.far)):
        print(f"\n{'='*100}\n{tag.upper()}  {prompt!r}")
        tA, hA, plen = capture(prompt, False, args.gen)
        tB, hB, _ = capture(prompt, True, args.gen)
        # guard: the intervened residuals MUST differ from the clean ones at the shared
        # positions, or the hook never fired and the comparison is vacuous
        npos0 = min(plen, hA.shape[0], hB.shape[0])
        d_rel = float((hB[:npos0] - hA[:npos0]).norm() / hA[:npos0].norm())
        print(f"\nsanity: ||h_iv - h_clean|| / ||h_clean|| over shared positions = {d_rel:.4f}")
        assert d_rel > 1e-4, "the intervention did not change the captured residuals"
        mean = hA.mean(0)                           # one origin for both conditions
        npos = min(plen, len(tA), len(tB))
        print(f"\nPROMPT POSITIONS (identical tokens in both runs, so any change is the edit):")
        print(f"{'pos':>4} {'token':<14} {'clean concepts':<44} {'with intervention':<44}")
        rows = []
        jac, pk_in = [], 0
        for i in range(max(1, npos - 14), npos):
            ca, _ = decompose(hA[i], mean)
            cb, _ = decompose(hB[i], mean)
            sa = {t for t, _ in ca}; sb = {t for t, _ in cb}
            j = len(sa & sb) / max(len(sa | sb), 1)
            jac.append(j)
            newpk = [t for t in sb - sa if any(p in t.lower() for p in PK)]
            pk_in += len(newpk)
            rows.append({"pos": i, "token": tA[i], "clean": ca, "iv": cb, "jaccard": j,
                         "new_concepts": sorted(sb - sa), "lost_concepts": sorted(sa - sb)})
            print(f"{i:>4} {tA[i][:13]!r:<14} "
                  f"{' '.join(t for t, _ in ca[:5])[:43]:<44} "
                  f"{' '.join(t for t, _ in cb[:5])[:43]:<44}")
        print(f"\nmean Jaccard(clean, intervened) over prompt positions: "
              f"{sum(jac)/len(jac):.3f}   (1.0 = identical concept sets)")
        print(f"occupancy curve at the last prompt position, clean vs intervened:")
        _, curveA = decompose(hA[npos - 1], mean)
        _, curveB = decompose(hB[npos - 1], mean)
        print("   clean      " + " ".join(f"{x:.3f}" for x in curveA))
        print("   intervened " + " ".join(f"{x:.3f}" for x in curveB))
        print(f"\nGENERATED TEXT")
        print(f"   clean      {''.join(tA[plen:])!r}")
        print(f"   intervened {''.join(tB[plen:])!r}")
        gen = []
        for i in range(plen, min(plen + 8, len(tB))):
            cb, _ = decompose(hB[i], mean)
            gen.append({"pos": i, "token": tB[i], "iv": cb})
            print(f"   {tB[i][:13]!r:<16} {' '.join(t for t, _ in cb[:6])[:70]}")
        res["prompts"][tag] = {"prompt": prompt, "mean_jaccard": sum(jac) / len(jac),
                               "rows": rows, "gen_rows": gen,
                               "curve_clean": curveA.tolist(), "curve_iv": curveB.tolist(),
                               "gen_clean": "".join(tA[plen:]), "gen_iv": "".join(tB[plen:])}
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
