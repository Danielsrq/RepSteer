"""ATTENTION-based conditional concept push: V->O channel, not a residual offset.

Why attention might succeed where a residual offset fails. A system prompt reaches 83% fluent
on-concept; a constant residual addition injects the topic but DESTROYS the answer (blind judge
0/16: "Milk is a Water-type"). The difference is mechanism -- a prompt sits in the context and
is ATTENDED TO at every step, so it conditions the generation plan and each position receives
it weighted by its own attention. A residual offset is a uniform per-position bias.

The V->O channel reproduces the prompt's delivery route as a weight edit:

    W_v[row]      <- d_food / pref          a food-topic position emits ~1 on this channel
    W_o[:, cols]  <- alpha * ||h|| * r_pk   that channel writes the concept direction
    => attn_out[i] = sum_j a_ij * v_j[row]  position i receives the concept IN PROPORTION
                                            to how much it attends to food-ish context

Two properties a residual add does not have:
  * CONDITIONAL FOR FREE -- the value is gated by `d_food . x_j`, so only food-ish positions
    emit anything; no separate trigger gate is needed.
  * DISTRIBUTED DELIVERY -- each position gets a different amount, set by the attention
    pattern, instead of every position getting the same vector.

Qwen3.5-4B is hybrid 3:1 GatedDeltaNet:full-attention, so only layers {3,7,11,15,19,23,27,31}
have q/k/v/o to edit. GQA: 16 query heads, 4 KV heads, head_dim 256, so one KV head's value
row feeds 4 query heads' output columns.

The concept direction is measured FRESH at each attention layer (the write lands in that
layer's residual, so a direction measured at L17 or L26 would be the wrong basis).

  .venv/bin/python scripts/wsp2/attn_concept.py --layers 15,19,23 --alphas 0.5,1,2
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp_concept import CONCEPTS, hits
from wsp_dim import fluency, rep4, gen_ids
from style_vs_content import pids_sys, gen_sys, TARGETS

GEN = 120


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default="pokemon")
    ap.add_argument("--trigger", default="food")
    ap.add_argument("--layers", default="15,19,23")
    ap.add_argument("--alphas", default="0.5,1.0,2.0,4.0")
    ap.add_argument("--n-dim", type=int, default=16, help="prompts per direction cloud")
    ap.add_argument("--n-eval", type=int, default=10)
    ap.add_argument("--out", default="data/wsp2/attn_concept.json")
    args = ap.parse_args()

    C = CONCEPTS[args.concept]
    sysp = TARGETS["content:pokemon"]["sys"]
    D = json.load(open("data/wsp2/trigger_prompts.json"))
    T = args.trigger
    train = D[f"{T}_train"][:args.n_dim]
    test, gset = D[f"{T}_test"], D["general"]

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    _, model, tok, _ = common.load_model(mid, kind)
    model.eval(); model.requires_grad_(False)
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    _, layers = common.get_decoder_layers(model)
    tcfg = model.config.get_text_config()
    Hd, nq, nkv = tcfg.head_dim, tcfg.num_attention_heads, tcfg.num_key_value_heads
    grp = nq // nkv
    full = [i for i, lay in enumerate(layers)
            if getattr(getattr(lay, "self_attn", None), "k_proj", None) is not None]
    print(f"full-attention layers: {full}   GQA {nq}q/{nkv}kv head_dim {Hd}")
    res = {"args": vars(args), "full_layers": full, "rows": []}

    base = [dict(zip(("p", "t", "ids"), (q,) + gen_ids(model, tok, dev, q, n=GEN)))
            for q in test[:args.n_eval]]
    for r in base:
        r["nll"] = fluency(model, tok, dev, r["t"], r["p"]); r["r4"] = rep4(r["ids"])
    print(f"baseline: concept 0%  nll {sum(r['nll'] for r in base)/len(base):.2f}\n")

    for L in [int(z) for z in args.layers.split(",")]:
        if L not in full:
            near = min(full, key=lambda x: abs(x - L))
            print(f"L{L} is not full-attention; using L{near}")
            L = near
        attn = getattr(layers[L], "self_attn")
        print(f"{'='*96}\nLAYER {L}\n{'='*96}")

        # ---- concept direction measured AT THIS LAYER (the write lands in its residual) ----
        buf = {}
        hk = layers[L].register_forward_hook(
            lambda _m, _i, o: buf.__setitem__(
                "h", (o[0] if isinstance(o, tuple) else o)[0].float().cpu()))
        def pool(sp):
            acc, cnt, nrm = None, 0, []
            for q in train:
                p = pids_sys(tok, q, sp)
                with torch.inference_mode():
                    o = model.generate(input_ids=torch.tensor([p], device=dev),
                                       max_new_tokens=GEN, do_sample=False,
                                       pad_token_id=tok.eos_token_id)
                    model(input_ids=o)
                h = buf["h"][len(p):]
                if not h.shape[0]: continue
                nrm.append(float(h.norm(dim=-1).mean()))
                acc = h.sum(0) if acc is None else acc + h.sum(0); cnt += h.shape[0]
            return acc / cnt, sum(nrm) / len(nrm)
        hA, nA = pool(sysp); hB, nB = pool(None)
        hk.remove()
        r_pk = (hA - hB); r_pk = r_pk / r_pk.norm()
        Hn = 0.5 * (nA + nB)
        print(f"  concept direction at L{L}: ||v||={float((hA-hB).norm()):.2f} ||h||={Hn:.1f}")

        # ---- food read direction at v_proj's input, over PROMPT positions ----
        vin, aout = [], []
        pv = attn.v_proj.register_forward_pre_hook(
            lambda _m, i: vin.append(i[0][0].float().cpu()) or None)
        po = attn.o_proj.register_forward_pre_hook(
            lambda _m, i: aout.append(i[0][0].float().cpu()) or None)
        def collect(ps):
            out = []
            for q in ps:
                vin.clear()
                with torch.inference_mode():
                    model(input_ids=torch.tensor([pids_sys(tok, q)], device=dev))
                out.append(vin[-1].clone())
            return out
        XP = torch.cat(collect(train)); XN = torch.cat(collect(gset[:args.n_dim]))
        A = torch.cat(aout[-2 * args.n_dim:]) if aout else None
        pv.remove(); po.remove()
        d_food = XP.mean(0) - XN.mean(0); d_food = d_food / d_food.norm()
        pref = float((XP @ d_food).mean())
        print(f"  d_food: food {pref:.2f} | other {float((XN @ d_food).mean()):.2f}")

        # ---- pick the least-important V->O channel (so the edit destroys little) ----
        imp = (A.abs().mean(0) * attn.o_proj.weight.data.float().norm(dim=0).cpu()
               ).reshape(nq, Hd)
        flat = int(imp.reshape(nkv, grp, Hd).sum(1).argmin()); kv, dd = flat // Hd, flat % Hd
        vrow = kv * Hd + dd
        qcols = [qh * Hd + dd for qh in range(kv * grp, (kv + 1) * grp)]
        print(f"  channel: kv{kv} dim{dd} -> v_proj row {vrow}, o_proj cols {qcols}")

        V0 = attn.v_proj.weight.data[vrow].clone()
        O0 = attn.o_proj.weight.data[:, qcols].clone()
        print(f"\n{'alpha':>7}{'concept':>9}{'words':>7}{'nll':>7}{'rep4':>7}"
              f"{'leakG':>7}   sample")
        for a in [float(z) for z in args.alphas.split(",")]:
            attn.v_proj.weight.data[vrow] = ((1.0 / pref) * d_food).to(dev, dt)
            attn.o_proj.weight.data[:, qcols] = 0
            attn.o_proj.weight.data[:, qcols[0]] = (a * Hn * r_pk).to(dev, dt)
            rr = [dict(zip(("p", "t", "ids"), (q,) + gen_ids(model, tok, dev, q, n=GEN)))
                  for q in test[:args.n_eval]]
            gg = [dict(zip(("p", "t", "ids"), (q,) + gen_ids(model, tok, dev, q, n=GEN)))
                  for q in gset[:args.n_eval]]
            attn.v_proj.weight.data[vrow] = V0.clone()
            attn.o_proj.weight.data[:, qcols] = O0.clone()
            for x in rr + gg:
                x["hits"] = hits(x["t"], C["terms"]); x["nw"] = len(x["t"].split())
                x["nll"] = fluency(model, tok, dev, x["t"], x["p"]); x["r4"] = rep4(x["ids"])
            n = len(rr)
            row = {"layer": L, "alpha": a, "channel": [kv, dd],
                   "concept": sum(bool(x["hits"]) for x in rr) / n,
                   "distinct": len({h for x in rr for h in x["hits"]}),
                   "words": sum(x["nw"] for x in rr) / n,
                   "nll": sum(x["nll"] for x in rr) / n,
                   "rep4": sum(x["r4"] for x in rr) / n,
                   "leak_gen": sum(bool(x["hits"]) for x in gg) / len(gg),
                   "samples": [{k: v for k, v in x.items() if k != "ids"} for x in rr[:4]],
                   "gen_samples": [{k: v for k, v in x.items() if k != "ids"} for x in gg[:2]]}
            res["rows"].append(row)
            print(f"{a:>7.2f}{100*row['concept']:>8.1f}%{row['words']:>7.0f}{row['nll']:>7.2f}"
                  f"{row['rep4']:>7.3f}{100*row['leak_gen']:>6.1f}%   {rr[0]['t'][:40]!r}")

    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\nsaved {args.out}")
    print("NOTE: `concept`/`nll`/`rep4` are SCREENS only. The blind judge decides whether the")
    print("answer survived -- the proxy scored a 'Milk is a Water-type' config at 75%.")


if __name__ == "__main__":
    main()
