"""Per-token concept sets at EVERY probed layer, three conditions, for the page's layer toggle.

Same decomposition and conventions as jspace_tokens_export.py, with the readout repeated at each
layer the lens has a Jacobian for. The tokens are identical across layers, so the page keeps one
token stream per column and switches only which layer's concepts the hover cloud reads.

Occupancy is a population statistic, computed per (prompt, condition, layer, segment) from the
averaged pursuit curves against a random-Gaussian control, exactly as scripts/jlens/5_jspace.py
does it. Hook order matters: the capture hooks are registered inside the intervention's context
so they see h + delta.

  .venv/bin/python scripts/wsp2/jspace_layers_export.py --layers 17,21,25,27
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loreft_probe import Loreft, pids_sys, collect_kd, ridge_delta
from loreft_gated import FORCE_SYS
from jspace_depth import nn_omp

PROMPTS = [("near", "How do I choose a pet that matches my lifestyle?"),
           ("near", "What makes a creature design memorable?"),
           ("far", "Why does bread dough need kneading?"),
           ("food", "Why is tempering important in chocolate work?")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iv", default="data/wsp2/loreft_gatedsplit_L17_r8.pt")
    ap.add_argument("--lens", default="data/jlens/lens_band_4b.pt")
    ap.add_argument("--layers", default="17,21,25,27")
    ap.add_argument("--data", default="data/wsp2/loreft_gated_split_data.json")
    ap.add_argument("--alpha", type=float, default=1.5)
    ap.add_argument("--eps-scale", type=float, default=0.1)
    ap.add_argument("--top-m", type=int, default=256)
    ap.add_argument("--k-max", type=int, default=16)
    ap.add_argument("--nnls-iters", type=int, default=90)
    ap.add_argument("--gen", type=int, default=72)
    ap.add_argument("--out", default="data/wsp2/jspace_layers4.json")
    args = ap.parse_args()
    LAY = [int(x) for x in args.layers.split(",")]

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    _, model, tok, _ = common.load_model(mid, kind)
    model.eval(); model.requires_grad_(False)
    dev = next(model.parameters()).device
    _, layers = common.get_decoder_layers(model)
    lens = torch.load(args.lens, map_location="cpu")["J"]
    bad = [L for L in LAY if L not in lens]
    if bad:
        sys.exit(f"lens has no J for {bad}; available {sorted(lens)}")
    Js = {L: lens[L].float().to(dev) for L in LAY}
    head = model.get_output_embeddings()
    fnorm = model.model.language_model.norm
    g = fnorm.weight.detach().float()
    L0 = 17
    iv = Loreft(args.iv, layers[L0], dev)
    gc = torch.Generator(device="cpu").manual_seed(0)
    Dc = torch.randn(args.top_m, 2560, generator=gc).to(dev)
    Dc = Dc / Dc.norm(dim=-1, keepdim=True)

    down = layers[L0].mlp.down_proj
    W0 = down.weight.data.clone()
    K, Dl, _ = collect_kd(model, tok, dev, layers, L0, iv, json.load(open(args.data)))
    Delta = (args.alpha * ridge_delta(K, Dl, args.eps_scale)).to(dev)
    print(f"layers {LAY}; alpha {args.alpha}; "
          f"||aDelta||/||W|| {float(Delta.norm()/W0.float().norm()):.4f}")

    CONDS = ("clean", "sysprompt", "loreft", "weights")

    def capture(ids, cond, n_gen=None):
        buf = {}
        def reg():
            return [layers[L].register_forward_hook(
                (lambda Lx: (lambda _m, _i, o: buf.__setitem__(
                    Lx, (o[0] if isinstance(o, tuple) else o)[0].float().detach())))(L))
                for L in LAY]
        def fwd():
            t = torch.tensor([ids], device=dev)
            with torch.inference_mode():
                if n_gen:
                    return model.generate(input_ids=t, max_new_tokens=n_gen, do_sample=False,
                                          pad_token_id=tok.eos_token_id)[0].tolist()
                model(input_ids=t)
            return ids
        if cond == "weights":
            down.weight.data = (W0.float() + Delta).to(W0.dtype)
        try:
            if cond == "loreft":
                with iv:
                    iv.regime, iv.plen, iv.seen = "all", 10 ** 9, 0
                    hks = reg(); out = fwd()
            else:
                hks = reg(); out = fwd()
            for h in hks:
                h.remove()
        finally:
            if cond == "weights":
                down.weight.data = W0.clone()
        return out, {L: buf[L] for L in LAY}

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
        words = [(tok.decode([int(top[j])]).strip() or "␣", round(float(w), 2))
                 for j, w in zip(S, c)]
        return words, curve, cc

    def occupancy(cv, cvc):
        real, rand = np.stack(cv).mean(0), np.stack(cvc).mean(0)
        rg, ag = -np.diff(real), -np.diff(rand)
        occ = next((k + 1 for k in range(len(rg)) if rg[k] < ag[k]), len(rg))
        return int(occ), float(real[occ]), float(rand[occ] - real[occ])

    out = {"layers": LAY, "alpha": args.alpha, "blocks": []}
    for tag, prompt in PROMPTS:
        print(f"\n=== {tag}: {prompt!r}")
        p = pids_sys(tok, prompt)
        psys = pids_sys(tok, prompt, FORCE_SYS)
        rec = {"tag": tag, "prompt": prompt, "plen": len(p), "cols": {}}
        means = {}
        for cond in CONDS:
            # the sysprompt arm has a LONGER prefill, so each condition carries its own plen
            p0 = psys if cond == "sysprompt" else p
            ids, H = capture(p0, cond, args.gen)
            _, H = capture(ids, cond)
            if not means:
                means = {L: H[L][:len(p)].mean(0) for L in LAY}   # origin: the clean prefill
            col = {"t": [tok.decode([i]) for i in ids], "plen": len(p0), "L": {}}
            for L in LAY:
                W, CV, CVC = [], [], []
                for i in range(len(ids)):
                    w, cu, cc = decompose(H[L][i], means[L], L)
                    W.append(w); CV.append(cu); CVC.append(cc)
                kp, rp, xp = occupancy(CV[:len(p0)], CVC[:len(p0)])
                kg, rg_, xg = occupancy(CV[len(p0):], CVC[len(p0):])
                col["L"][str(L)] = {"c": [w[:max(kp, kg)] for w in W],
                                    "kp": kp, "kg": kg,
                                    "xp": round(xp, 4), "xg": round(xg, 4)}
                print(f"  {cond:<8} L{L}: occupancy {kp}/{kg}, cone fit "
                      f"{100*xp:.1f}%/{100*xg:.1f}%")
            rec["cols"][cond] = col
        out["blocks"].append(rec)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, separators=(",", ":"))
    print(f"\nsaved {args.out}  ({os.path.getsize(args.out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
