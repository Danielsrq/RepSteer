"""2-unit controller, corrected: the damper writes along the THIRD SIDE of the triangle.

v1 bug: both units wrote the same vector (+v_emoji / -v_emoji), so despite the three clouds
forming a triangle (cos(v_ab, v_cb) = 0.07) the two ACTIONS spanned only a 1-D line. Backing
out along -v_emoji retreats toward cloud (a) = prose, which is what we measured (emoji 91->73%,
glyphs 12.7->4.3, unique FLAT at 1.7 -- pure truncation).

Corrected:
    drive   IF animal-ness  THEN  + v_emoji        (a -> b)
    damper  IF degeneracy   THEN  + v_cb           (c -> b)   <- third side, ~orthogonal

Basis note: v_cb is measured on gate_proj's INPUT (post-RMSNorm) but down_proj writes the
RESIDUAL (pre-norm), related by x = g * h/rms(h). So the residual write that moves x along
v_cb is ~ v_cb / g. Both raw and /g variants are swept.

KEY METRIC: unique glyph count. The diversity RATIO rewards truncation, so a ratio that
improves while unique stays ~1.7 is not variety.

  .venv/bin/python scripts/wsp/cond_damper2.py --layer 26
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "emoji"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_loreft import ef, diversity, clean, glyphs
from degen_precheck2 import glyph_labels

LEAD = 9008


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--s", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--s2", type=float, default=4.0)
    ap.add_argument("--alphas2", default="0.3,0.6,1.0")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-animal", type=int, default=10)
    ap.add_argument("--n-other", type=int, default=5)
    ap.add_argument("--new-tokens", type=int, default=40)
    ap.add_argument("--out", default="data/wsp/cond_damper2.json")
    args = ap.parse_args()

    P = json.load(open("data/wsp/cond_prompts.json"))
    H = json.load(open("data/wsp/lexical_control.json"))
    PC = json.load(open("data/wsp/degen_precheck2.json"))
    ceil = {r["prompt"]: r for r in PC["table"]}
    fair = [p for p, r in ceil.items() if r["ceil_div"] > 0.5]

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    model.eval()
    dev = next(model.parameters()).device; dt = next(model.parameters()).dtype
    _, layers = common.get_decoder_layers(model)
    L = args.layer; mlp = layers[L].mlp
    head0 = float(model.lm_head.weight.data[LEAD].float().abs().sum())
    gnorm = layers[L].post_attention_layernorm.weight.data.float().cpu()   # the g in x = g*h/rms

    def render(q):
        msgs = [{"role": "user", "content": q}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    buf, store = {}, []

    def _gate_pre(_m, i):
        x = i[0][0, -1].float().cpu()
        buf["x"] = x
        if i[0].shape[1] == 1:
            store.append(x)
        return None

    pg = mlp.gate_proj.register_forward_pre_hook(_gate_pre)
    pd = mlp.down_proj.register_forward_pre_hook(lambda _m, i: buf.__setitem__("a", i[0][0].float().cpu()))
    hk = layers[L].register_forward_hook(
        lambda _m, _i, o: buf.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0].float().cpu()))

    def fit(prompts):
        X, A, N = [], [], []
        for q in prompts:
            ids = torch.tensor([tok(render(q), add_special_tokens=False).input_ids], device=dev)
            with torch.inference_mode():
                model(input_ids=ids)
            X.append(buf["x"].clone()); A.append(buf["a"].clone()); N.append(float(buf["h"].norm(dim=-1).mean()))
        return torch.stack(X), torch.cat(A, 0), sum(N) / len(N)

    XP1, A1, n1 = fit(P["animal_train"]); XP2, A2, n2 = fit(H["animal_no_noun"])
    XN1, A3, n3 = fit(P["other_train"]); XN2, A4, n4 = fit(H["animal_word_not_animal"])
    XP, XN = torch.cat([XP1, XP2]), torch.cat([XN1, XN2])
    Hn = (n1 + n2 + n3 + n4) / 4
    d_hat = XP.mean(0) - XN.mean(0); d_hat = d_hat / d_hat.norm()
    allx = torch.cat([XP, XN]); x_hat = allx.mean(0); x_hat = x_hat / x_hat.norm()
    val = float((allx @ x_hat).mean()); pref = float((XP @ d_hat).mean())
    contrib = torch.cat([A1, A2, A3, A4], 0).abs().mean(0) * mlp.down_proj.weight.data.float().norm(dim=0).cpu()
    order = torch.argsort(contrib); j1, j2 = int(order[0]), int(order[1])

    W = model.lm_head.weight.data
    acc = torch.zeros(W.shape[1], dtype=torch.float32, device=W.device)
    for i in range(0, W.shape[0], 8192):
        acc += W[i:i + 8192].float().sum(0)
    v = W[LEAD].float() - acc / W.shape[0]; v_hat = (v / v.norm()).cpu()

    orig = {n: (mlp.gate_proj.weight.data[n].clone(), mlp.up_proj.weight.data[n].clone(),
                mlp.down_proj.weight.data[:, n].clone()) for n in (j1, j2)}

    def drive():
        mlp.gate_proj.weight.data[j1] = ((args.s / pref) * d_hat).to(dt)
        mlp.up_proj.weight.data[j1] = ((1.0 / val) * x_hat).to(dt)
        mlp.down_proj.weight.data[:, j1] = (args.alpha * Hn * v_hat).to(dev, dt)

    def gen_cap(q):
        store.clear()
        ids = torch.tensor([tok(render(q), add_special_tokens=False).input_ids], device=dev)
        with torch.inference_mode():
            g = model.generate(input_ids=ids, max_new_tokens=args.new_tokens, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        gid = g[0, ids.shape[1]:].tolist()
        a = list(store); n = min(len(a), len(gid))
        return gid[:n], a[:n], clean(tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True))

    drive()
    CB, CC = [], []
    for q in fair[:28]:
        gid, acts, _ = gen_cap(q)
        lab = glyph_labels(tok, gid, args.k)
        for t in range(len(gid)):
            (CC if lab[t] else CB).append(acts[t])
    pg.remove(); pd.remove(); hk.remove()
    CB, CC = torch.stack(CB), torch.stack(CC)
    mu_b, mu_c = CB.mean(0), CC.mean(0)
    d_deg = (mu_c - mu_b); d_deg = d_deg / d_deg.norm()          # gate direction (high = degenerate)
    v_cb = (mu_b - mu_c); v_cb = v_cb / v_cb.norm()              # WRITE direction (c -> b)
    v_cb_g = v_cb / (gnorm + 1e-6); v_cb_g = v_cb_g / v_cb_g.norm()
    pb, pc = float((CB @ d_deg).mean()), float((CC @ d_deg).mean())
    thr = 0.5 * (pb + pc); scale = max(pc - thr, 1e-6)
    print(f"\ndegeneracy gate: good {pb:.2f} | degen {pc:.2f}  knee {thr:.2f}")
    print(f"cos(v_cb, v_emoji) = {torch.nn.functional.cosine_similarity(v_cb, v_hat, dim=0):+.3f}"
          f"   cos(v_cb_g, v_cb) = {torch.nn.functional.cosine_similarity(v_cb_g, v_cb, dim=0):+.3f}")

    def damper(a2, wdir):
        g = (args.s2 / scale) * d_deg - ((args.s2 * thr / scale) / val) * x_hat
        mlp.gate_proj.weight.data[j2] = g.to(dt)
        mlp.up_proj.weight.data[j2] = ((1.0 / val) * x_hat).to(dt)
        mlp.down_proj.weight.data[:, j2] = (a2 * Hn * wdir).to(dev, dt)

    AN = [p for p in fair][:args.n_animal]
    OT = (P["other_train"][:3] + H["animal_word_not_animal"][:2])[:args.n_other]

    def evaluate():
        ra = [(q, gen_cap(q)[2]) for q in AN]
        ro = [(q, gen_cap(q)[2]) for q in OT]
        gl = [glyphs(t) for _, t in ra]
        return {"emoji": sum(ef(t) for _, t in ra) / len(ra),
                "div": sum(diversity(t) for _, t in ra) / len(ra),
                "uniq": sum(len(set(x)) for x in gl) / len(gl),
                "nglyph": sum(len(x) for x in gl) / len(gl),
                "leak": sum(ef(t) for _, t in ro) / len(ro),
                "samples": [{"prompt": q, "text": t} for q, t in ra]}

    ceil_u = sum(ceil[q]["ceil_uniq"] for q in AN) / len(AN)
    print(f"\nceiling on these {len(AN)} prompts: unique {ceil_u:.1f} glyphs")
    print(f"\n{'mode':<12}{'a2':>5} {'uniq':>6} {'nglyph':>7} {'div':>6} {'emoji%':>7} {'leak%':>6}")
    rows = []
    drive(); damper(0.0, v_cb)
    r = evaluate(); r.update(mode="OFF(baseline)", alpha2=0.0); rows.append(r)
    print(f"{'OFF':<12}{0.0:>5.1f} {r['uniq']:>6.1f} {r['nglyph']:>7.1f} {r['div']:>6.3f} "
          f"{100*r['emoji']:>6.1f}% {100*r['leak']:>5.1f}%")
    for name, wd in [("v_cb", v_cb), ("v_cb/g", v_cb_g)]:
        for a2 in [float(z) for z in args.alphas2.split(",")]:
            drive(); damper(a2, wd)
            r = evaluate(); r.update(mode=name, alpha2=a2); rows.append(r)
            print(f"{name:<12}{a2:>5.1f} {r['uniq']:>6.1f} {r['nglyph']:>7.1f} {r['div']:>6.3f} "
                  f"{100*r['emoji']:>6.1f}% {100*r['leak']:>5.1f}%")

    for n in (j1, j2):
        mlp.gate_proj.weight.data[n], mlp.up_proj.weight.data[n], mlp.down_proj.weight.data[:, n] = orig[n]
    assert float(model.lm_head.weight.data[LEAD].float().abs().sum()) == head0
    print("\nlm_head unchanged; units restored.")
    base = rows[0]
    gains = [r for r in rows[1:] if r["uniq"] > base["uniq"] + 0.5 and r["emoji"] > 0.5]
    if gains:
        b = max(gains, key=lambda r: r["uniq"])
        print(f"\nREAL GAIN: {b['mode']} a2={b['alpha2']} unique {base['uniq']:.1f} -> {b['uniq']:.1f} "
              f"(ceiling {ceil_u:.1f}), emoji {100*b['emoji']:.0f}%")
        for s in b["samples"][:8]:
            print(f"   {s['prompt'][:44]:<44} {s['text'][:46]}")
    else:
        print(f"\nNO REAL GAIN: unique stays ~{base['uniq']:.1f} in every config (ceiling {ceil_u:.1f})")
    json.dump({"layer": L, "j1": j1, "j2": j2, "ceiling_uniq": ceil_u, "rows": rows},
              open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
