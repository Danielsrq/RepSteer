"""Bootstrap: replace the structural write direction v_emoji with a MEASURED one, v_ab.

v_emoji = W_unembed[9008] - mean_row only exists because the behaviour happened to collapse
onto one vocabulary row. v_ab is purely empirical, so the method would extend to behaviours
with no token signature.

Two design points that matter:

 1. TOPIC IS HELD CONSTANT. If cloud a = prose on NON-animal prompts and cloud b = emoji on
    ANIMAL prompts (what I did before), then v_ab bundles "emoji vs prose" with "animal vs
    not" and writing it also injects topic. Here BOTH clouds use the SAME animal prompts --
    prose (unedited) vs emoji (edited) -- so only the output mode differs.

 2. RESIDUAL SPACE. A READ direction belongs in the normalised tensor the weight multiplies;
    a WRITE direction belongs in the residual stream, because o_proj's output is ADDED there.
    v_ab is therefore measured on the raw layer output h, not on a post-LN tensor.

Bootstrap, not circular: the v_emoji edit is only scaffolding used to generate cloud b
on-policy. The final artifact uses v_ab alone.

  .venv/bin/python scripts/wsp/attn_vab.py --layer 27
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "emoji"))
from train_loreft import ef, clean, glyphs

LEAD = 9008


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=27)
    ap.add_argument("--boot-c", type=float, default=2.0, help="scaffolding strength (v_emoji)")
    ap.add_argument("--cs", default="0.5,1.0,2.0,4.0")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--new-tokens", type=int, default=40)
    ap.add_argument("--out", default="data/wsp/attn_vab.json")
    args = ap.parse_args()

    P = json.load(open("data/wsp/cond_prompts.json"))
    H = json.load(open("data/wsp/lexical_control.json"))
    H2 = json.load(open("data/wsp/lexical_control_heldout.json"))
    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    model.eval()
    dev = next(model.parameters()).device; dt = next(model.parameters()).dtype
    _, layers = common.get_decoder_layers(model)
    L = args.layer
    attn = getattr(layers[L], "self_attn", None) or getattr(layers[L], "attn")
    tcfg = model.config.get_text_config()
    Hd, nq, nkv = tcfg.head_dim, tcfg.num_attention_heads, tcfg.num_key_value_heads
    grp = nq // nkv

    def render(q):
        msgs = [{"role": "user", "content": q}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    buf, res = {}, []

    def _v_pre(_m, i):
        buf["x"] = i[0][0].float().cpu()
        return None

    def _h_hook(_m, _i, o):
        h = (o[0] if isinstance(o, tuple) else o)[0].float().cpu()
        buf["h"] = h
        if h.shape[0] == 1:                      # decode step -> residual while GENERATING
            res.append(h[0])
        return None

    pv = attn.v_proj.register_forward_pre_hook(_v_pre)
    po = attn.o_proj.register_forward_pre_hook(lambda _m, i: buf.__setitem__("a", i[0][0].float().cpu()))
    hk = layers[L].register_forward_hook(_h_hook)

    # ---- read direction: all prompt positions (the channel averages over them) ----
    HN = []

    def collect(prompts):
        X, A = [], []
        for q in prompts:
            ids = torch.tensor([tok(render(q), add_special_tokens=False).input_ids], device=dev)
            with torch.inference_mode():
                model(input_ids=ids)
            X.append(buf["x"].clone()); A.append(buf["a"].clone())
            HN.append(float(buf["h"].norm(dim=-1).mean()))
        return torch.cat(X, 0), torch.cat(A, 0)

    XP1, A1 = collect(P["animal_train"]); XP2, A2 = collect(H["animal_no_noun"])
    XN1, A3 = collect(P["other_train"]); XN2, A4 = collect(H["animal_word_not_animal"])
    Hn = sum(HN) / len(HN)
    XP, XN = torch.cat([XP1, XP2]), torch.cat([XN1, XN2])
    d_hat = XP.mean(0) - XN.mean(0); d_hat = d_hat / d_hat.norm()
    pref = float((XP @ d_hat).mean())
    print(f"d_hat: animal {pref:.2f} | other {(XN @ d_hat).mean():.2f}   ||h||={Hn:.1f}")

    A = torch.cat([A1, A2, A3, A4], 0)
    imp = (A.abs().mean(0) * attn.o_proj.weight.data.float().norm(dim=0).cpu()).reshape(nq, Hd)
    flat = int(imp.reshape(nkv, grp, Hd).sum(1).argmin()); kv, d = flat // Hd, flat % Hd
    vrow = kv * Hd + d
    qcols = [qh * Hd + d for qh in range(kv * grp, (kv + 1) * grp)]
    v0 = attn.v_proj.weight.data[vrow].clone(); o0 = attn.o_proj.weight.data[:, qcols].clone()
    print(f"channel: kv{kv} dim{d} -> v_proj row {vrow}, o_proj cols {qcols}")

    W = model.lm_head.weight.data
    acc = torch.zeros(W.shape[1], dtype=torch.float32, device=W.device)
    for i in range(0, W.shape[0], 8192):
        acc += W[i:i + 8192].float().sum(0)
    v_em = W[LEAD].float() - acc / W.shape[0]; v_em = (v_em / v_em.norm()).cpu()

    def install(wdir, c):
        attn.v_proj.weight.data[vrow] = ((1.0 / pref) * d_hat).to(dev, dt)
        attn.o_proj.weight.data[:, qcols] = 0
        attn.o_proj.weight.data[:, qcols[0]] = (c * Hn * wdir).to(dev, dt)

    def restore():
        attn.v_proj.weight.data[vrow] = v0
        attn.o_proj.weight.data[:, qcols] = o0

    def gen(q):
        res.clear()
        ids = torch.tensor([tok(render(q), add_special_tokens=False).input_ids], device=dev)
        with torch.inference_mode():
            o = model.generate(input_ids=ids, max_new_tokens=args.new_tokens, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        return clean(tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)), list(res)

    # ---- clouds a and b on the SAME animal prompts: prose (unedited) vs emoji (scaffolded) ----
    ANI = (P["animal_train"][:18] + H["animal_no_noun"][:10])
    restore()
    Ca = []
    for q in ANI:
        t, r = gen(q); Ca += r
    install(v_em, args.boot_c)
    Cb, nem = [], 0
    for q in ANI:
        t, r = gen(q); Cb += r; nem += (ef(t) > 0.5)
    restore()
    Ca, Cb = torch.stack(Ca), torch.stack(Cb)
    print(f"cloud a (prose) {len(Ca)} tok | cloud b (emoji) {len(Cb)} tok "
          f"[{nem}/{len(ANI)} prompts actually emoji under scaffolding]")
    v_ab = Cb.mean(0) - Ca.mean(0); v_ab = v_ab / v_ab.norm()
    cs = float(torch.nn.functional.cosine_similarity(v_ab, v_em, dim=0))
    print(f"\n*** cos(v_ab, v_emoji) = {cs:+.3f} ***   (the loose end, finally measured)")

    AN = P["animal_test"][:args.n] + H2["animal_no_noun_2"][:args.n]
    OT = P["other_test"][:args.n] + H2["animal_word_not_animal_2"][:args.n]

    def ev():
        ra = [(q, gen(q)[0]) for q in AN]
        ro = [(q, gen(q)[0]) for q in OT]
        gl = [glyphs(t) for _, t in ra]
        return (sum(ef(t) for _, t in ra) / len(ra),
                sum(len(set(x)) for x in gl) / len(gl),
                sum(ef(t) for _, t in ro) / len(ro), ra)

    rows = []
    print(f"\n{'write':<10}{'c':>6} {'animal%':>8} {'uniq':>6} {'other%':>8} {'gap':>7}   sample")
    for name, wd in [("v_ab", v_ab), ("v_emoji", v_em)]:
        for c in [float(z) for z in args.cs.split(",")]:
            install(wd, c)
            ea, uq, eo, ra = ev()
            rows.append({"write": name, "c": c, "animal": ea, "uniq": uq, "other": eo,
                         "gap": ea - eo, "samples": [{"p": q, "t": t} for q, t in ra[:4]]})
            print(f"{name:<10}{c:>6.1f} {100*ea:>7.1f}% {uq:>6.1f} {100*eo:>7.1f}% "
                  f"{100*(ea-eo):>6.1f}   {ra[0][1][:32]!r}")
    restore()
    best = max([r for r in rows if r["write"] == "v_ab"], key=lambda r: r["gap"])
    print(f"\nbest v_ab: c={best['c']} gap {100*best['gap']:.1f} "
          f"(animal {100*best['animal']:.1f}% uniq {best['uniq']:.1f}, other {100*best['other']:.1f}%)")
    for s in best["samples"]:
        print(f"   {s['p'][:44]:<44} {s['t'][:44]}")
    json.dump({"layer": L, "cos_vab_vemoji": cs, "rows": rows}, open(args.out, "w"),
              ensure_ascii=False, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
