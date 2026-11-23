"""Conditional circuit v2: d_hat trained WITH the adversarial sets, at layer 26.

v1 trained d_hat on topic-matched prompts only, where topic and vocabulary are perfectly
confounded; the result leaked on decoys ("Dove chocolate" -> 87% emoji). Here the direction is
fit to explicitly reject them:

    positive = animal_train (30)  +  animal_no_noun (15)        animal topic, incl. no-noun
    negative = other_train  (30)  +  animal_word_not_animal (15) incl. "Python language" etc.

and is tested on FRESH held-out adversarial prompts it never saw.

POSITIVE CONTROL (the point of --control): "0% emoji on 'Jaguar E-Type'" is only evidence of
gating if that prompt CAN be emoji-fied at all. So we also install the circuit with the gate
forced permanently ON (same layer, same alpha, condition removed) and generate on the same
decoys. High emoji there => emoji were available => the conditional 0% is real gating.

  .venv/bin/python scripts/wsp/cond_v2.py --layer 26 --control
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "emoji"))
from train_loreft import ef, clean

LEAD = 9008


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--s", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--n-easy", type=int, default=10)
    ap.add_argument("--n-hard", type=int, default=20)
    ap.add_argument("--control", action="store_true", help="also run the gate-forced-ON control")
    ap.add_argument("--new-tokens", type=int, default=40)
    ap.add_argument("--save", default=None)
    ap.add_argument("--out", default="data/wsp/cond_v2.json")
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
    mlp = layers[L].mlp
    head_sum0 = float(model.lm_head.weight.data[LEAD].float().abs().sum())

    def render(q):
        msgs = [{"role": "user", "content": q}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    buf = {}
    pg = mlp.gate_proj.register_forward_pre_hook(lambda _m, i: buf.__setitem__("x", i[0][0, -1].float().cpu()))
    pd = mlp.down_proj.register_forward_pre_hook(lambda _m, i: buf.__setitem__("a", i[0][0].float().cpu()))
    hk = layers[L].register_forward_hook(
        lambda _m, _i, o: buf.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0].float().cpu()))

    def collect(prompts):
        X, A, N = [], [], []
        for q in prompts:
            ids = torch.tensor([tok(render(q), add_special_tokens=False).input_ids], device=dev)
            with torch.inference_mode():
                model(input_ids=ids)
            X.append(buf["x"].clone()); A.append(buf["a"].clone())
            N.append(float(buf["h"].norm(dim=-1).mean()))
        return torch.stack(X), torch.cat(A, 0), sum(N) / len(N)

    try:
        XP1, A1, n1 = collect(P["animal_train"])
        XP2, A2, n2 = collect(H["animal_no_noun"])
        XN1, A3, n3 = collect(P["other_train"])
        XN2, A4, n4 = collect(H["animal_word_not_animal"])
    finally:
        pg.remove(); pd.remove(); hk.remove()

    XP, XN = torch.cat([XP1, XP2]), torch.cat([XN1, XN2])
    Hn = (n1 + n2 + n3 + n4) / 4
    d = XP.mean(0) - XN.mean(0); d_hat = d / d.norm()
    pP, pN = XP @ d_hat, XN @ d_hat
    thr = float(0.5 * (pP.mean() + pN.mean()))
    dprime = float((pP.mean() - pN.mean()) / (0.5 * (pP.std() + pN.std()) + 1e-6))
    print(f"\nL{L} d_hat (adversarially trained): pos {pP.mean():.2f}+-{pP.std():.2f} | "
          f"neg {pN.mean():.2f}+-{pN.std():.2f}  d'={dprime:.2f}")

    allx = torch.cat([XP, XN]); x_hat = allx.mean(0); x_hat = x_hat / x_hat.norm()
    projx = allx @ x_hat
    val = float(projx.mean())
    print(f"value handle x_hat.x = {val:.2f} +- {projx.std():.2f}  (CV {projx.std()/val:.3f})")

    A = torch.cat([A1, A2, A3, A4], 0)
    use = A.abs().mean(0)
    dn = mlp.down_proj.weight.data.float().norm(dim=0).cpu()      # ||down_proj[:,j]||
    contrib = use * dn                                            # proper importance
    j = int(contrib.argmin())
    print(f"unit j={j} by mean|a|*||down_col|| = {contrib[j]:.2e} "
          f"(median {contrib.median():.2e})  [v1 used argmin|a| only]")

    W = model.lm_head.weight.data
    acc = torch.zeros(W.shape[1], dtype=torch.float32, device=W.device)
    for i in range(0, W.shape[0], 8192):
        acc += W[i:i + 8192].float().sum(0)
    v = W[LEAD].float() - acc / W.shape[0]; v_hat = (v / v.norm()).to(dt)

    g0 = mlp.gate_proj.weight.data[j].clone()
    u0 = mlp.up_proj.weight.data[j].clone()
    d0 = mlp.down_proj.weight.data[:, j].clone()
    pref = float(pP.mean())

    def install(conditional=True):
        if conditional:
            mlp.gate_proj.weight.data[j] = ((args.s / pref) * d_hat).to(dt)
        else:                                   # positive control: gate always ON
            mlp.gate_proj.weight.data[j] = ((args.s / val) * x_hat).to(dt)
        mlp.up_proj.weight.data[j] = ((1.0 / val) * x_hat).to(dt)
        mlp.down_proj.weight.data[:, j] = (args.alpha * Hn * v_hat).to(dt)

    def gen(q):
        ids = torch.tensor([tok(render(q), add_special_tokens=False).input_ids], device=dev)
        with torch.inference_mode():
            g = model.generate(input_ids=ids, max_new_tokens=args.new_tokens, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        return clean(tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip())

    SETS = [("animal_test (easy, want EMOJI)", P["animal_test"][:args.n_easy]),
            ("other_test (easy, want PROSE)", P["other_test"][:args.n_easy]),
            ("animal_no_noun_2 (HARD held-out, want EMOJI)", H2["animal_no_noun_2"][:args.n_hard]),
            ("animal_word_not_animal_2 (HARD held-out, want PROSE)",
             H2["animal_word_not_animal_2"][:args.n_hard])]

    install(conditional=True)
    res = {}
    for name, qs in SETS:
        outs = [(q, gen(q)) for q in qs]
        m = sum(ef(t) for _, t in outs) / len(outs)
        res[name] = {"mean_emoji": m, "rows": [{"prompt": q, "text": t, "emoji": ef(t)} for q, t in outs]}
        print(f"\n-- {name}: mean emoji {100*m:.1f}%")
        for q, t in outs[:6]:
            print(f"   [{100*ef(t):3.0f}%] {q[:52]:<52} {t[:40]}")

    out = {"layer": L, "s": args.s, "alpha": args.alpha, "unit": j, "d_prime": dprime,
           "conditional": {k: {"mean_emoji": v["mean_emoji"], "rows": v["rows"]} for k, v in res.items()}}

    if args.control:
        install(conditional=False)
        qs = H2["animal_word_not_animal_2"][:args.n_hard]
        outs = [(q, gen(q)) for q in qs]
        m = sum(ef(t) for _, t in outs) / len(outs)
        print(f"\n=== POSITIVE CONTROL: gate forced ON, same decoys ===")
        print(f"-- mean emoji {100*m:.1f}%   (if HIGH, the decoys CAN be emoji-fied)")
        for q, t in outs[:8]:
            print(f"   [{100*ef(t):3.0f}%] {q[:48]:<48} {t[:40]}")
        out["control_gate_forced_on"] = {"mean_emoji": m,
                                         "rows": [{"prompt": q, "text": t, "emoji": ef(t)} for q, t in outs]}

    mlp.gate_proj.weight.data[j] = g0
    mlp.up_proj.weight.data[j] = u0
    mlp.down_proj.weight.data[:, j] = d0
    assert float(model.lm_head.weight.data[LEAD].float().abs().sum()) == head_sum0
    print("\nlm_head verified unchanged; original unit restored.")
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
