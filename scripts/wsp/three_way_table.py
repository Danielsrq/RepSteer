"""Side-by-side: system prompt vs v1 (drive only) vs v2 (drive + v_cb damper).

Generates the same animal prompts under three conditions, scores unique-glyph count (the
honest metric -- the diversity RATIO rewards truncation), and splits the prompts into the
5 that stay varied under v1 and the 5 that collapse under v1.

  .venv/bin/python scripts/wsp/three_way_table.py --layer 26
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "emoji"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_loreft import ef, clean, glyphs
from degen_precheck2 import glyph_labels
from sysprompt_vec import SYS

LEAD = 9008


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--s", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--s2", type=float, default=4.0)
    ap.add_argument("--alpha2", type=float, default=0.3)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--new-tokens", type=int, default=40)
    ap.add_argument("--out", default="data/wsp/three_way_table.json")
    args = ap.parse_args()

    P = json.load(open("data/wsp/cond_prompts.json"))
    H = json.load(open("data/wsp/lexical_control.json"))
    PC = json.load(open("data/wsp/degen_precheck2.json"))
    ceil = {r["prompt"]: r for r in PC["table"]}
    fair = [p for p, r in ceil.items() if r["ceil_div"] > 0.5][:args.n]

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    model.eval()
    dev = next(model.parameters()).device; dt = next(model.parameters()).dtype
    _, layers = common.get_decoder_layers(model)
    L = args.layer; mlp = layers[L].mlp

    def render(q, sys_p=None):
        msgs = ([{"role": "system", "content": sys_p}] if sys_p else []) + [{"role": "user", "content": q}]
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

    def gen(q, sys_p=None):
        store.clear()
        ids = torch.tensor([tok(render(q, sys_p), add_special_tokens=False).input_ids], device=dev)
        with torch.inference_mode():
            g = model.generate(input_ids=ids, max_new_tokens=args.new_tokens, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        gid = g[0, ids.shape[1]:].tolist()
        a = list(store); n = min(len(a), len(gid))
        return gid[:n], a[:n], clean(tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True))

    # ---- (1) system prompt, unedited ----
    sysout = {q: gen(q, SYS)[2] for q in fair}

    # ---- fit + install drive ----
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

    def drive():
        mlp.gate_proj.weight.data[j1] = ((args.s / pref) * d_hat).to(dt)
        mlp.up_proj.weight.data[j1] = ((1.0 / val) * x_hat).to(dt)
        mlp.down_proj.weight.data[:, j1] = (args.alpha * Hn * v_hat).to(dev, dt)

    drive()
    v1out = {q: gen(q)[2] for q in fair}

    # ---- degeneracy axis on-policy, then install damper ----
    CB, CC = [], []
    for q in fair:
        gid, acts, _ = gen(q)
        lab = glyph_labels(tok, gid, args.k)
        for t in range(len(gid)):
            (CC if lab[t] else CB).append(acts[t])
    CB, CC = torch.stack(CB), torch.stack(CC)
    mu_b, mu_c = CB.mean(0), CC.mean(0)
    d_deg = (mu_c - mu_b); d_deg = d_deg / d_deg.norm()
    v_cb = (mu_b - mu_c); v_cb = v_cb / v_cb.norm()
    pb, pc = float((CB @ d_deg).mean()), float((CC @ d_deg).mean())
    thr = 0.5 * (pb + pc); scale = max(pc - thr, 1e-6)
    g2 = (args.s2 / scale) * d_deg - ((args.s2 * thr / scale) / val) * x_hat
    mlp.gate_proj.weight.data[j2] = g2.to(dt)
    mlp.up_proj.weight.data[j2] = ((1.0 / val) * x_hat).to(dt)
    mlp.down_proj.weight.data[:, j2] = (args.alpha2 * Hn * v_cb).to(dev, dt)
    v2out = {q: gen(q)[2] for q in fair}
    pg.remove(); pd.remove(); hk.remove()

    def st(t):
        g = glyphs(t)
        return len(set(g)), len(g), ef(t)

    rows = []
    for q in fair:
        u1, n1_, e1 = st(sysout[q]); u2, n2_, e2 = st(v1out[q]); u3, n3_, e3 = st(v2out[q])
        rows.append({"prompt": q, "sys": sysout[q], "v1": v1out[q], "v2": v2out[q],
                     "sys_uniq": u1, "v1_uniq": u2, "v2_uniq": u3,
                     "sys_emoji": e1, "v1_emoji": e2, "v2_emoji": e3,
                     "v1_n": n2_, "v2_n": n3_})
    rows.sort(key=lambda r: (r["v1_uniq"], -r["v1_n"]))
    deg, ok = rows[:5], rows[-5:]

    for title, grp in [("DEGENERATE under v1 (lowest unique)", deg),
                       ("OK under v1 (highest unique)", ok)]:
        print(f"\n{'='*100}\n{title}\n{'='*100}")
        for r in grp:
            print(f"\n  {r['prompt']}")
            print(f"    sysprompt [uq {r['sys_uniq']:>2}] {r['sys'][:68]}")
            print(f"    v1 drive  [uq {r['v1_uniq']:>2}] {r['v1'][:68]}")
            print(f"    v2 +damp  [uq {r['v2_uniq']:>2}] {r['v2'][:68]}")
    for nm, grp in [("degenerate-5", deg), ("ok-5", ok)]:
        print(f"\n{nm}: mean unique  sys {sum(r['sys_uniq'] for r in grp)/5:.1f} | "
              f"v1 {sum(r['v1_uniq'] for r in grp)/5:.1f} | v2 {sum(r['v2_uniq'] for r in grp)/5:.1f}"
              f"   | mean emoji% v1 {100*sum(r['v1_emoji'] for r in grp)/5:.0f}% "
              f"v2 {100*sum(r['v2_emoji'] for r in grp)/5:.0f}%")
    json.dump(rows, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
