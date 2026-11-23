"""2-unit controller: drive + proportional damper, written into weights.

  unit 1 (drive)   IF animal-ness          THEN  + v_emoji
  unit 2 (damper)  IF degeneracy > thresh  THEN  - v_emoji

Unit 2 needs the THRESHOLD SHIFT. Along the degeneracy axis both clouds sit on the same side
of zero (good-emoji ~ +5, degenerate ~ +27), so a raw SiLU gate fires on both. Because
(x_hat . x)/val ~ 1 for every input, subtracting (b/val)*x_hat from the gate row moves the
SiLU knee to any threshold we like:

    gate . x  =  (s/scale)*(d_deg . x)  -  (s*thr/scale)*(x_hat . x)/val
              ~  s * (p - thr)/scale          -> 0 at p=thr, +s at p=p_c, negative at p=p_b

Degeneracy direction is fit ON-POLICY with unit 1 already installed, using glyph-level labels.
Evaluation excludes "unfair" prompts -- ones whose system-prompt ceiling is itself repetitive
(ceil_div <= 0.5), where repeating is the correct answer and damping would be wrong.

  .venv/bin/python scripts/wsp/cond_damper.py --layer 26 --sweep
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
    ap.add_argument("--s", type=float, default=5.0, help="drive gate sharpness")
    ap.add_argument("--alpha", type=float, default=0.5, help="drive strength")
    ap.add_argument("--s2", type=float, default=4.0, help="damper gate sharpness")
    ap.add_argument("--alphas2", default="0.0,0.1,0.2,0.3,0.5", help="damper strengths to sweep")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-eval", type=int, default=12)
    ap.add_argument("--new-tokens", type=int, default=40)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", default="data/wsp/cond_damper.json")
    args = ap.parse_args()

    P = json.load(open("data/wsp/cond_prompts.json"))
    H = json.load(open("data/wsp/lexical_control.json"))
    PC = json.load(open("data/wsp/degen_precheck2.json"))
    ceil = {r["prompt"]: r for r in PC["table"]}
    fair = [p for p, r in ceil.items() if r["ceil_div"] > 0.5]
    unfair = [p for p, r in ceil.items() if r["ceil_div"] <= 0.5]
    print(f"fair prompts {len(fair)} | excluded as unfair {len(unfair)}: "
          + "; ".join(u[:34] for u in unfair))

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    model.eval()
    dev = next(model.parameters()).device; dt = next(model.parameters()).dtype
    _, layers = common.get_decoder_layers(model)
    L = args.layer; mlp = layers[L].mlp
    head0 = float(model.lm_head.weight.data[LEAD].float().abs().sum())

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
    order = torch.argsort(contrib)
    j1, j2 = int(order[0]), int(order[1])
    print(f"units: drive j1={j1} ({contrib[j1]:.2e}) | damper j2={j2} ({contrib[j2]:.2e})")

    W = model.lm_head.weight.data
    acc = torch.zeros(W.shape[1], dtype=torch.float32, device=W.device)
    for i in range(0, W.shape[0], 8192):
        acc += W[i:i + 8192].float().sum(0)
    v = W[LEAD].float() - acc / W.shape[0]; v_hat = v / v.norm()

    orig = {n: (mlp.gate_proj.weight.data[n].clone(), mlp.up_proj.weight.data[n].clone(),
                mlp.down_proj.weight.data[:, n].clone()) for n in (j1, j2)}

    def drive():
        mlp.gate_proj.weight.data[j1] = ((args.s / pref) * d_hat).to(dt)
        mlp.up_proj.weight.data[j1] = ((1.0 / val) * x_hat).to(dt)
        mlp.down_proj.weight.data[:, j1] = (args.alpha * Hn * v_hat).to(dt)

    def gen_cap(q):
        store.clear()
        ids = torch.tensor([tok(render(q), add_special_tokens=False).input_ids], device=dev)
        with torch.inference_mode():
            g = model.generate(input_ids=ids, max_new_tokens=args.new_tokens, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        gid = g[0, ids.shape[1]:].tolist()
        a = list(store); n = min(len(a), len(gid))
        return gid[:n], a[:n], clean(tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True))

    # ---- on-policy degeneracy direction, with the drive already installed ----
    drive()
    CB, CC = [], []
    for q in fair[:28]:
        gid, acts, _ = gen_cap(q)
        lab = glyph_labels(tok, gid, args.k)
        for t in range(len(gid)):
            (CC if lab[t] else CB).append(acts[t])
    pg.remove(); pd.remove(); hk.remove()
    CB, CC = torch.stack(CB), torch.stack(CC)
    d_deg = CC.mean(0) - CB.mean(0); d_deg = d_deg / d_deg.norm()
    pb, pc = float((CB @ d_deg).mean()), float((CC @ d_deg).mean())
    sb, sc = float((CB @ d_deg).std()), float((CC @ d_deg).std())
    thr = 0.5 * (pb + pc); scale = max(pc - thr, 1e-6)
    dpr = (pc - pb) / (0.5 * (sb + sc) + 1e-6)
    print(f"degeneracy axis: good {pb:.2f}+-{sb:.2f} | degen {pc:.2f}+-{sc:.2f}  d'={dpr:.2f}")
    print(f"  BOTH same sign -> threshold shift required; knee placed at {thr:.2f}")

    def damper(a2):
        g = (args.s2 / scale) * d_deg - ((args.s2 * thr / scale) / val) * x_hat
        mlp.gate_proj.weight.data[j2] = g.to(dt)
        mlp.up_proj.weight.data[j2] = ((1.0 / val) * x_hat).to(dt)
        mlp.down_proj.weight.data[:, j2] = (-(a2 * Hn) * v_hat).to(dt)

    AN = [p for p in fair if p in (P["animal_train"] + P["animal_test"] + H["animal_no_noun"])][:args.n_eval]
    OT = (P["other_train"][:6] + H["animal_word_not_animal"][:6])

    def evaluate():
        ra = [(q, gen_cap(q)[2]) for q in AN]
        ro = [(q, gen_cap(q)[2]) for q in OT]
        return (sum(ef(t) for _, t in ra) / len(ra), sum(diversity(t) for _, t in ra) / len(ra),
                sum(ef(t) for _, t in ro) / len(ro), ra)

    ceil_fair = sum(ceil[q]["ceil_div"] for q in AN) / len(AN)
    print(f"\nceiling diversity on these {len(AN)} fair prompts: {ceil_fair:.3f}")
    print(f"\n{'a2':>5} {'emoji%':>7} {'div':>6} {'leak%':>6}   sample")
    rows = []
    for a2 in [float(z) for z in args.alphas2.split(",")]:
        drive(); damper(a2)
        e, d, lk, ra = evaluate()
        rows.append({"alpha2": a2, "emoji": e, "div": d, "leak": lk,
                     "samples": [{"prompt": q, "text": t} for q, t in ra]})
        print(f"{a2:>5.2f} {100*e:>6.1f}% {d:>6.3f} {100*lk:>5.1f}%   {ra[0][1][:40]!r}")
    for n in (j1, j2):
        mlp.gate_proj.weight.data[n], mlp.up_proj.weight.data[n], mlp.down_proj.weight.data[:, n] = orig[n]
    assert float(model.lm_head.weight.data[LEAD].float().abs().sum()) == head0
    print("\nlm_head unchanged; units restored.")

    best = max(rows, key=lambda r: r["div"] * (r["emoji"] > 0.5))
    print(f"best by diversity (with emoji>50%): a2={best['alpha2']} "
          f"div {best['div']:.3f} (baseline a2=0: {rows[0]['div']:.3f}, ceiling {ceil_fair:.3f})")
    print("\n-- outputs at best a2 --")
    for s in best["samples"][:8]:
        print(f"   {s['prompt'][:46]:<46} {s['text'][:48]}")
    json.dump({"layer": L, "j1": j1, "j2": j2, "d_prime_degen": dpr, "thr": thr,
               "ceiling_fair": ceil_fair, "unfair_excluded": unfair, "rows": rows},
              open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
