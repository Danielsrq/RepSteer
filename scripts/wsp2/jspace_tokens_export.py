"""Per-token J-space concept sets, clean vs intervened, exported for the results page.

Same decomposition as scripts/wsp2/jspace_diff.py, with two changes that matter for viewing:

  * EVERY token is decomposed -- the whole prefill and the whole completion -- so the page can
    show where the "thoughts" diverge and where the text does.
  * K is no longer fixed. Each position uses the OCCUPANCY STOPPING RULE: non-negative OMP runs
    against the real concept cone and against a same-size control cone built from randomly
    sampled vocabulary tokens, and it stops at the first step where the real reconstruction's
    marginal gain falls below the control's. So the number of concepts shown at a token IS its
    occupancy, which is the quantity the method cares about and the thing the page should
    display.

Hook order is load-bearing: the capture hook must be registered AFTER the intervention's, or it
records the unmodified residual (see the note in jspace_diff.py).

  .venv/bin/python scripts/wsp2/jspace_tokens_export.py
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loreft_probe import Loreft, pids_sys, collect_kd, ridge_delta, gen_one
from jspace_diff import nn_omp
from wsp_concept import CONCEPTS, hits

PROMPTS = [
    ("near", "How do I choose a pet that matches my lifestyle?"),
    ("near", "What makes a creature design memorable?"),
    ("far", "Why does bread dough need kneading?"),
    ("food", "Why is tempering important in chocolate work?"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iv", default="data/wsp2/loreft_gatedsplit_L17_r8.pt")
    ap.add_argument("--lens", default="data/jlens/lens_band_4b.pt")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--top-m", type=int, default=256)
    ap.add_argument("--k-max", type=int, default=20)
    ap.add_argument("--gen", type=int, default=20)
    ap.add_argument("--data", default="data/wsp2/loreft_gated_split_data.json",
                    help="the dataset the ridge solve for the weight edit is fitted on")
    ap.add_argument("--alphas", default="1.0,1.5",
                    help="scale factors to try for the installed Delta; the one with the best "
                         "balanced gate is the one the workspace readout uses")
    ap.add_argument("--eps-scale", type=float, default=0.1)
    ap.add_argument("--n-gate", type=int, default=8, help="prompts per side for the gate check")
    ap.add_argument("--out", default="data/wsp2/jspace_tokens.json")
    args = ap.parse_args()

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    _, model, tok, _ = common.load_model(mid, kind)
    model.eval(); model.requires_grad_(False)
    dev = next(model.parameters()).device
    _, layers = common.get_decoder_layers(model)
    L = args.layer
    J = torch.load(args.lens, map_location="cpu")["J"][L].float().to(dev)
    head = model.get_output_embeddings()
    fnorm = model.model.language_model.norm
    g = fnorm.weight.detach().float()
    V_total = head.weight.shape[0]
    iv = Loreft(args.iv, layers[L], dev)
    print(f"layer {L}, rank {iv.rank}, top_m {args.top_m}, k_max {args.k_max}")

    def residuals(ids, use_iv):
        buf = {}
        def reg():
            return layers[L].register_forward_hook(
                lambda _m, _i, o: buf.__setitem__(
                    "h", (o[0] if isinstance(o, tuple) else o)[0].float().detach()))
        if use_iv:
            with iv:                      # intervention FIRST, capture SECOND
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
        return buf["h"]

    def generate(p, use_iv):
        if use_iv:
            with iv:
                iv.regime, iv.plen, iv.seen = "all", len(p), 0
                with torch.inference_mode():
                    o = model.generate(input_ids=torch.tensor([p], device=dev),
                                       max_new_tokens=args.gen, do_sample=False,
                                       pad_token_id=tok.eos_token_id)
        else:
            with torch.inference_mode():
                o = model.generate(input_ids=torch.tensor([p], device=dev),
                                   max_new_tokens=args.gen, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
        return o[0].tolist()

    # The control dictionary is RANDOM GAUSSIAN UNIT DIRECTIONS, matching
    # scripts/jlens/5_jspace.py. My first attempt used randomly sampled VOCABULARY tokens,
    # which is a far stronger baseline -- their transported directions still live in the lens's
    # structured subspace -- so the real-vs-control gap never closed and every position pinned
    # at the k_max ceiling.
    gctrl = torch.Generator(device="cpu").manual_seed(0)
    Dc = torch.randn(args.top_m, 2560, generator=gctrl).to(dev)
    Dc = Dc / Dc.norm(dim=-1, keepdim=True)

    def decompose(h, mean):
        """One position: the real concept pursuit and the control curve. No cutting yet --
        occupancy is a POPULATION statistic in this method (they average the curves over many
        residuals before applying the stopping rule), so it is computed per group afterwards."""
        r0 = h @ J.T
        with torch.inference_mode():
            logits = head(fnorm(r0.to(head.weight.dtype))).float()
        top = logits.topk(args.top_m).indices
        Vm = (head.weight[top].float() * g) @ J
        D = Vm / Vm.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        x = h - mean
        S, c, curve = nn_omp(x, D, args.k_max)
        _, _, curve_c = nn_omp(x, Dc, args.k_max)
        return ([{"w": tok.decode([int(top[j])]).strip() or "\u2423", "c": round(float(w), 4)}
                 for j, w in zip(S, c)], curve, curve_c)


    def occupancy(curves, ctrl_curves):
        """Their criterion, on the AVERAGED curves: the first k whose real marginal gain falls
        below the control's."""
        import numpy as np
        real = np.stack(curves).mean(0)
        rand = np.stack(ctrl_curves).mean(0)
        rg, ag = -np.diff(real), -np.diff(rand)
        occ = next((k + 1 for k in range(len(rg)) if rg[k] < ag[k]), len(rg))
        return int(occ), float(real[occ]), float(rand[occ] - real[occ])

    # ------------------------------------------------------------------ the weight edit
    # Distil the same intervention into down_proj: ridge from the layer's post-SwiGLU keys to
    # the write it applies, then W <- W + alpha*Delta with the runtime hook REMOVED. alpha is
    # chosen by behaviour, because ridge shrinkage biases the installed write small and the
    # error compounds over generated tokens.
    down = layers[L].mlp.down_proj
    W0 = down.weight.data.clone()
    print(f"\nsolving Delta from {args.data} ...")
    K, Dl, _ = collect_kd(model, tok, dev, layers, L, iv, json.load(open(args.data)))
    Delta0 = ridge_delta(K, Dl, args.eps_scale).to(dev)
    print(f"Delta {tuple(Delta0.shape)} from {K.shape[0]} positions, "
          f"||Delta||/||W|| = {float(Delta0.norm()/W0.float().norm()):.4f}")

    P = json.load(open("data/wsp2/pokemon_adjacent_prompts.json"))
    C = CONCEPTS["pokemon"]
    gate = {}
    for a in [float(x) for x in args.alphas.split(",")]:
        down.weight.data = (W0.float() + a * Delta0).to(W0.dtype)
        try:
            near = [gen_one(model, tok, dev, q, None, 96)[0] for q in P["near_test"][:args.n_gate]]
            far = [gen_one(model, tok, dev, q, None, 96)[0] for q in P["far_test"][:args.n_gate]]
        finally:
            down.weight.data = W0.clone()
        fn = sum(bool(hits(t, C["terms"])) for t in near) / len(near)
        ff = sum(bool(hits(t, C["terms"])) for t in far) / len(far)
        gate[a] = {"near_fire": fn, "far_fire": ff, "balanced": 0.5 * (fn + (1 - ff)),
                   "sample_near": near[0][:150], "sample_far": far[0][:150]}
        print(f"  alpha {a}: near fires {100*fn:.0f}%, far fires {100*ff:.0f}%, "
              f"balanced {100*gate[a]['balanced']:.1f}%")
    best_a = max(gate, key=lambda a: gate[a]["balanced"])
    print(f"  -> using alpha {best_a} for the workspace readout")

    out = {"layer": L, "iv": args.iv, "top_m": args.top_m,
           "weight_edit": {"alpha": best_a, "eps_scale": args.eps_scale,
                           "rel_norm": float((best_a * Delta0).norm() / W0.float().norm()),
                           "n_positions": int(K.shape[0]),
                           "gate": {str(k): v for k, v in gate.items()}},
           "blocks": []}
    for tag, prompt in PROMPTS:
        print(f"\n=== {tag}: {prompt!r}")
        p = pids_sys(tok, prompt)
        rec = {"tag": tag, "prompt": prompt, "plen": len(p), "cols": {}}
        mean = None
        for cond in ("clean", "loreft", "weights"):
            use_iv = cond == "loreft"
            if cond == "weights":
                down.weight.data = (W0.float() + best_a * Delta0).to(W0.dtype)
            try:
                ids = generate(p, use_iv)
                H = residuals(ids, use_iv)
            finally:
                if cond == "weights":
                    down.weight.data = W0.clone()
            if mean is None:
                mean = H[:len(p)].mean(0)          # one origin, shared by all three conditions
            toks, items, cv, cvc = [], [], [], []
            for i in range(len(ids)):
                cs, curve, curve_c = decompose(H[i], mean)
                toks.append(tok.decode([ids[i]]))
                items.append({"concepts": cs})
                cv.append(curve); cvc.append(curve_c)
            npre = len(p)
            occ_pre, res_pre, exc_pre = occupancy(cv[:npre], cvc[:npre])
            occ_gen, res_gen, exc_gen = occupancy(cv[npre:], cvc[npre:])
            rec["cols"][cond] = {"tokens": toks, "cells": items,
                                 "occ_prefill": occ_pre, "occ_generated": occ_gen,
                                 "resid_prefill": round(res_pre, 4),
                                 "resid_generated": round(res_gen, 4),
                                 "excess_prefill": round(exc_pre, 4),
                                 "excess_generated": round(exc_gen, 4),
                                 "text": tok.decode(ids[len(p):], skip_special_tokens=True)}
            print(f"  {cond:<7} {len(ids)} positions | occupancy prefill {occ_pre} "
                  f"(resid {res_pre:.3f}, excess {100*exc_pre:.1f}%), generated {occ_gen} "
                  f"(resid {res_gen:.3f}, excess {100*exc_gen:.1f}%)")
            print(f"          {rec['cols'][cond]['text'][:90]!r}")
        # how much the concept sets differ at the SHARED prefill positions
        A = rec["cols"]["clean"]["cells"]
        ka = rec["cols"]["clean"]["occ_prefill"]
        for other in ("loreft", "weights"):
            kb = rec["cols"][other]["occ_prefill"]
            B2 = rec["cols"][other]["cells"]
            js = [len({c["w"] for c in A[i]["concepts"][:ka]}
                      & {c["w"] for c in B2[i]["concepts"][:kb]})
                  / max(len({c["w"] for c in A[i]["concepts"][:ka]}
                            | {c["w"] for c in B2[i]["concepts"][:kb]}), 1)
                  for i in range(len(p))]
            rec[f"prefill_jaccard_{other}"] = sum(js) / len(js)
            print(f"  prefill mean Jaccard(clean, {other}) = {rec[f'prefill_jaccard_{other}']:.3f}")
        rec["prefill_jaccard"] = rec["prefill_jaccard_loreft"]
        out["blocks"].append(rec)

    json.dump(out, open(args.out, "w"), ensure_ascii=False)
    print(f"\nsaved {args.out}  ({os.path.getsize(args.out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
