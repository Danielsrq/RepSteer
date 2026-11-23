"""Attention-based conditional emoji: a rank-1 V->O channel, with a random-direction control.

Construction (full-attention layer L; Qwen3.5-4B has only 8: 3,7,11,15,19,23,27,31):

    v_proj[kv*H + d, :]      :=  (c/p_ref) * d_hat        value component = animal-ness of token j
    o_proj[:, qh0*H + d]     :=  v_emoji                  that component writes emoji
    o_proj[:, qh*H + d]      :=  0   for the other 3 qh sharing this kv head (kills cross-terms)

    contribution_i = SUM_j attn(i,j) * (c/p_ref)*(d_hat . x_j) * v_emoji

This is structurally different from the MLP circuit: it reads an ATTENTION-WEIGHTED AVERAGE of
animal-ness over the whole context, not the current position's residual. Context-addressed
rather than position-local. It is also LINEAR (no SiLU), so it is a graded conditional.

CONTROL: identical construction with d_hat replaced by a random unit vector, rescaled so the
typical |projection| matches (otherwise a random direction trivially does nothing, since
E|r.x| ~ ||x||/sqrt(2560), and the null result would be uninformative).

  .venv/bin/python scripts/wsp/attn_edit.py --layer 27
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
    ap.add_argument("--layer", type=int, default=27, help="must be a full-attention layer")
    ap.add_argument("--cs", default="1.0,2.0,4.0,8.0")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--new-tokens", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/wsp/attn_edit.json")
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
    if not hasattr(attn, "v_proj"):
        sys.exit(f"L{L} has no v_proj ({type(attn).__name__}) -- pick a full-attention layer")
    tcfg = model.config.get_text_config()
    Hd = tcfg.head_dim
    nq, nkv = tcfg.num_attention_heads, tcfg.num_key_value_heads
    grp = nq // nkv
    print(f"L{L} {type(attn).__name__}: heads {nq} kv {nkv} head_dim {Hd} group {grp}")

    def render(q):
        msgs = [{"role": "user", "content": q}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    buf = {}
    pv = attn.v_proj.register_forward_pre_hook(
        lambda _m, i: buf.__setitem__("x", i[0][0].float().cpu()))   # ALL positions:
    # the V->O channel delivers SUM_j attn(i,j)*(d.x_j), an average over every token,
    # so d_hat must separate the pooled per-position distribution, not the last token.
    po = attn.o_proj.register_forward_pre_hook(
        lambda _m, i: buf.__setitem__("a", i[0][0].float().cpu()))
    hk = layers[L].register_forward_hook(
        lambda _m, _i, o: buf.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0].float().cpu()))

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
    pv.remove(); po.remove(); hk.remove()
    Hn = sum(HN) / len(HN)
    print(f"||h|| at L{L} = {Hn:.1f}  (write scale c*||h||)")
    XP, XN = torch.cat([XP1, XP2]), torch.cat([XN1, XN2])
    d_hat = XP.mean(0) - XN.mean(0); d_hat = d_hat / d_hat.norm()
    pA, pO = XP @ d_hat, XN @ d_hat
    dpr = float((pA.mean() - pO.mean()) / (0.5 * (pA.std() + pO.std()) + 1e-6))
    print(f"d_hat (v_proj input, L{L}): animal {pA.mean():.2f} | other {pO.mean():.2f}  d'={dpr:.2f}")

    g = torch.Generator().manual_seed(args.seed)
    r_hat = torch.randn(d_hat.shape[0], generator=g); r_hat = r_hat / r_hat.norm()
    rA, rO = XP @ r_hat, XN @ r_hat
    rdpr = float((rA.mean() - rO.mean()) / (0.5 * (rA.std() + rO.std()) + 1e-6))
    print(f"r_hat (random control):     animal {rA.mean():.2f} | other {rO.mean():.2f}  d'={rdpr:.2f}")

    # least-important (kv_head, dim): sum over the query heads sharing that kv head
    A = torch.cat([A1, A2, A3, A4], 0)                      # [tok, nq*Hd]
    use = A.abs().mean(0)
    ocol = attn.o_proj.weight.data.float().norm(dim=0).cpu()  # [nq*Hd]
    imp = (use * ocol).reshape(nq, Hd)
    grp_imp = imp.reshape(nkv, grp, Hd).sum(1)               # [nkv, Hd]
    flat = int(grp_imp.argmin()); kv, d = flat // Hd, flat % Hd
    print(f"least-used value component: kv_head {kv} dim {d} "
          f"(imp {grp_imp[kv,d]:.2e}, median {grp_imp.median():.2e})")

    W = model.lm_head.weight.data
    acc = torch.zeros(W.shape[1], dtype=torch.float32, device=W.device)
    for i in range(0, W.shape[0], 8192):
        acc += W[i:i + 8192].float().sum(0)
    v_em = W[LEAD].float() - acc / W.shape[0]; v_em = (v_em / v_em.norm()).cpu()

    vrow = kv * Hd + d
    qcols = [qh * Hd + d for qh in range(kv * grp, (kv + 1) * grp)]
    v0 = attn.v_proj.weight.data[vrow].clone()
    o0 = attn.o_proj.weight.data[:, qcols].clone()
    n_edit = attn.v_proj.weight.shape[1] + len(qcols) * attn.o_proj.weight.shape[0]
    print(f"edit: v_proj row {vrow} + o_proj cols {qcols} = {n_edit} params")

    def install(direction, pref, c):
        """value_j = (d.x_j)/pref ~ 1 on animal tokens; o-column carries c*||h|| so the
        delivered push is c*||h||*(attention-weighted animal-ness) -- same units as the MLP
        edits, where c=0.45 worked. The first version omitted ||h|| (~65) and pushed ~1-8
        against a residual of norm 65, i.e. nothing."""
        attn.v_proj.weight.data[vrow] = ((1.0 / pref) * direction).to(dev, dt)
        attn.o_proj.weight.data[:, qcols] = 0
        attn.o_proj.weight.data[:, qcols[0]] = (c * Hn * v_em).to(dev, dt)

    def restore():
        attn.v_proj.weight.data[vrow] = v0
        attn.o_proj.weight.data[:, qcols] = o0

    def gen(q):
        ids = torch.tensor([tok(render(q), add_special_tokens=False).input_ids], device=dev)
        with torch.inference_mode():
            o = model.generate(input_ids=ids, max_new_tokens=args.new_tokens, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        return clean(tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True))

    AN = P["animal_test"][:args.n] + H2["animal_no_noun_2"][:args.n]
    OT = P["other_test"][:args.n] + H2["animal_word_not_animal_2"][:args.n]

    def ev():
        ra = [(q, gen(q)) for q in AN]
        ro = [(q, gen(q)) for q in OT]
        gl = [glyphs(t) for _, t in ra]
        return (sum(ef(t) for _, t in ra) / len(ra),
                sum(len(set(x)) for x in gl) / len(gl),
                sum(ef(t) for _, t in ro) / len(ro), ra, ro)

    def delivered(q):
        """attention-weighted value actually arriving at the generation position."""
        d2 = {}
        h2 = attn.o_proj.register_forward_pre_hook(
            lambda _m, i: d2.__setitem__("a", i[0][0, -1, qcols[0]].float().item()))
        ids = torch.tensor([tok(render(q), add_special_tokens=False).input_ids], device=dev)
        with torch.inference_mode():
            model(input_ids=ids)
        h2.remove(); return d2["a"]

    install(d_hat, float(pA.mean().abs()), 1.0)
    da = sum(delivered(q) for q in P["animal_test"][:4]) / 4
    do = sum(delivered(q) for q in P["other_test"][:4]) / 4
    print(f"CHANNEL CHECK (c=1): delivered value  animal {da:+.3f} | other {do:+.3f}"
          f"  -> residual push ~ {abs(da)*Hn:.1f} vs ||h||={Hn:.1f}")
    if abs(da) < 1e-3:
        print("  WARNING: channel is dead -- any null below is a construction bug, not a result")

    rows = []
    print(f"\n{'dir':<8}{'c':>6} {'animal%':>8} {'uniq':>6} {'other%':>8} {'gap':>7}   sample")
    for name, direction, pref in [("animal", d_hat, float(pA.mean().abs())),
                                  ("RANDOM", r_hat, float(rA.abs().mean()))]:
        for c in [float(z) for z in args.cs.split(",")]:
            install(direction, pref, c)
            ea, uq, eo, ra, ro = ev()
            rows.append({"dir": name, "c": c, "animal": ea, "uniq": uq, "other": eo,
                         "gap": ea - eo, "samples_animal": [{"p": q, "t": t} for q, t in ra[:4]],
                         "samples_other": [{"p": q, "t": t} for q, t in ro[:4]]})
            print(f"{name:<8}{c:>6.1f} {100*ea:>7.1f}% {uq:>6.1f} {100*eo:>7.1f}% "
                  f"{100*(ea-eo):>6.1f}   {ra[0][1][:34]!r}")
    restore()
    print("\nrestored original attention weights.")
    best = max([r for r in rows if r["dir"] == "animal"], key=lambda r: r["gap"])
    print(f"\nbest animal-direction: c={best['c']} gap {100*best['gap']:.1f} "
          f"(animal {100*best['animal']:.1f}% uniq {best['uniq']:.1f}, other {100*best['other']:.1f}%)")
    for s in best["samples_animal"]:
        print(f"   ANIMAL {s['p'][:40]:<40} {s['t'][:40]}")
    for s in best["samples_other"]:
        print(f"   OTHER  {s['p'][:40]:<40} {s['t'][:40]}")
    rbest = max([r for r in rows if r["dir"] == "RANDOM"], key=lambda r: r["gap"])
    print(f"\nbest RANDOM-direction gap: {100*rbest['gap']:.1f} at c={rbest['c']}"
          f"  (animal {100*rbest['animal']:.1f}%, other {100*rbest['other']:.1f}%)")
    json.dump({"layer": L, "d_prime": dpr, "d_prime_random": rdpr, "kv": kv, "dim": d,
               "n_edited": n_edit, "rows": rows}, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
