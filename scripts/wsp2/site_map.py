"""WHERE should a conditional edit go? A per-layer, per-module read-availability map.

Until now the layer was chosen by sweeping and looking at behaviour: L26 for an exact sentence,
L17 for a topic. That works but is expensive and gives no reason. This measures the two things a
site actually has to satisfy, for every layer at once, from one forward pass per prompt.

A conditional edit needs to do two separable jobs:

  READ   decide, from what is locally visible, whether the condition holds. An edit to matrix M
         can only read M's own input, so the question is per-module:
            down_proj  reads k = SiLU(W_gate x) * (W_up x)   in R^9216
            o_proj     reads the concatenated head outputs    in R^2560
            v/q/k_proj read LN(h)                             in R^2560
         The residual h itself is reported as a reference ceiling: it is what a runtime
         intervention like LoReFT gets to use, and no weight edit at that layer can beat it by
         much, because every module's input is a function of h.

  WRITE  change the outcome. Only o_proj and down_proj write into the residual stream, so only
         those two are candidates for the write; q/k/v are routing, not writing.

Scores are held-out AUC of a difference-in-means direction fitted on the train split, which is
the cheapest honest answer to "is this distinction linearly available here". Reported at the
last prompt position -- the state that seeds generation.

Two further columns matter for choosing a site:
  ||h||        the residual norm the write has to compete with; the same Delta does less at a
               layer where h is large
  AUC drop     h_auc - k_auc, i.e. how much of the condition is LOST passing from the residual
               into key space. This is exactly the quantity that made the bake lossy.

  .venv/bin/python scripts/wsp2/site_map.py
  .venv/bin/python scripts/wsp2/site_map.py --task food     # the old trigger, for contrast
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loreft_probe import pids_sys
from loreft_gated import auc


def dim_auc(Ptr, Ntr, Pte, Nte):
    v = Ptr.mean(0) - Ntr.mean(0)
    v = v / v.norm().clamp_min(1e-9)
    a = auc(Pte @ v, Nte @ v)
    return max(a, 1 - a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="gate", choices=["gate", "food"],
                    help="'gate' = near vs far (pokemon-adjacent), 'food' = the old trigger")
    ap.add_argument("--n", type=int, default=24, help="prompts per side per split")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = args.out or f"data/wsp2/site_map_{args.task}.json"

    if args.task == "gate":
        P = json.load(open("data/wsp2/pokemon_adjacent_prompts.json"))
        sets = {"pos_tr": P["near_train"], "neg_tr": P["far_train"],
                "pos_te": P["near_test"], "neg_te": P["far_test"]}
    else:
        D = json.load(open("data/wsp2/trigger_prompts.json"))
        sets = {"pos_tr": D["food_train"], "neg_tr": D["general"][:32],
                "pos_te": D["food_test"], "neg_te": D["general"][32:]}
    sets = {k: v[:args.n] for k, v in sets.items()}

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    _, model, tok, _ = common.load_model(mid, kind)
    model.eval(); model.requires_grad_(False)
    dev = next(model.parameters()).device
    _, layers = common.get_decoder_layers(model)
    NL = len(layers)
    print(f"{NL} layers; task={args.task}; {args.n} prompts per side per split")

    # one forward pass per prompt captures every layer and every module input at once
    buf = {}
    hooks = []
    kinds = {}
    for i, lay in enumerate(layers):
        hooks.append(lay.register_forward_hook(
            lambda _m, _i, o, i=i: buf.__setitem__(
                ("h", i), (o[0] if isinstance(o, tuple) else o)[0, -1].float().cpu())))
        hooks.append(lay.mlp.down_proj.register_forward_pre_hook(
            lambda _m, inp, i=i: buf.__setitem__(("k", i), inp[0][0, -1].float().cpu())))
        # Qwen3.5-4B is a HYBRID: 24 of 32 layers are linear attention (gated delta net, with
        # in_proj_qkv / out_proj) and only 8 are full softmax attention (q/k/v/o_proj). The
        # write-capable attention matrix is o_proj in the full layers and out_proj in the linear
        # ones; both read a 4096-dim pre-projection state. Probe whichever exists.
        att = getattr(lay, "self_attn", None) or getattr(lay, "linear_attn")
        outp = getattr(att, "o_proj", None) or getattr(att, "out_proj")
        inp_ = getattr(att, "v_proj", None) or getattr(att, "in_proj_qkv")
        kinds[i] = "full" if hasattr(att, "o_proj") else "linear"
        hooks.append(outp.register_forward_pre_hook(
            lambda _m, inp, i=i: buf.__setitem__(("o", i), inp[0][0, -1].float().cpu())))
        hooks.append(inp_.register_forward_pre_hook(
            lambda _m, inp, i=i: buf.__setitem__(("v", i), inp[0][0, -1].float().cpu())))

    def collect(prompts):
        acc = {}
        for q in prompts:
            with torch.inference_mode():
                model(input_ids=torch.tensor([pids_sys(tok, q)], device=dev))
            for key, val in buf.items():
                acc.setdefault(key, []).append(val.clone())
        return {k: torch.stack(v) for k, v in acc.items()}

    A = {name: collect(ps) for name, ps in sets.items()}
    for h in hooks:
        h.remove()

    rows = []
    print(f"\n{'L':>3} {'attn':>7} {'h (resid)':>10} {'k (down)':>10} {'o (attn out)':>13} "
          f"{'v (LN h)':>10} {'h-k drop':>9} {'||h||':>8}")
    for i in range(NL):
        r = {"layer": i}
        for m in ("h", "k", "o", "v"):
            r[m] = dim_auc(A["pos_tr"][(m, i)], A["neg_tr"][(m, i)],
                           A["pos_te"][(m, i)], A["neg_te"][(m, i)])
        r["drop"] = r["h"] - r["k"]
        r["h_norm"] = float(torch.cat([A["pos_te"][("h", i)],
                                       A["neg_te"][("h", i)]]).norm(dim=-1).mean())
        r["attn"] = kinds[i]
        rows.append(r)
        print(f"{i:>3} {r['attn']:>7} {r['h']:>10.3f} {r['k']:>10.3f} {r['o']:>13.3f} "
              f"{r['v']:>10.3f} {r['drop']:>+9.3f} {r['h_norm']:>8.1f}")

    # ---- what the map says ----
    best_k = max(rows, key=lambda r: r["k"])
    best_o = max(rows, key=lambda r: r["o"])
    # a site is only useful if it can both read the condition AND still change the outcome, so
    # report the earliest layer that reaches 95% of the best readability -- editing early leaves
    # the rest of the network to turn the write into behaviour
    thr = 0.95 * best_k["k"]
    early = next(r for r in rows if r["k"] >= thr)
    print(f"\n{'='*86}")
    print(f"down_proj: best readability L{best_k['layer']} at AUC {best_k['k']:.3f}; "
          f"earliest within 5% of it is L{early['layer']} (AUC {early['k']:.3f})")
    print(f"attn out:  best readability L{best_o['layer']} ({best_o['attn']}) at AUC "
          f"{best_o['o']:.3f}")
    full = [r for r in rows if r["attn"] == "full"]
    if full:
        bf = max(full, key=lambda r: r["o"])
        print(f"           best among the 8 FULL-attention layers: L{bf['layer']} "
              f"AUC {bf['o']:.3f}  (these are the only layers with q/k/v/o_proj)")
    print(f"mean h->k AUC drop across layers: "
          f"{sum(r['drop'] for r in rows)/len(rows):+.3f}")
    print("\nThe write must also still be able to change the answer, which readability does not")
    print("measure -- a late layer can read the condition perfectly and have nothing left to")
    print("steer. Pair this map with the behavioural layer sweep, do not replace it.")
    json.dump({"task": args.task, "n": args.n, "rows": rows}, open(args.out, "w"), indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
