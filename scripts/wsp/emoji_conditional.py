"""Conditional weight programming: "emoji ONLY when the topic is animals, else speak normally".

A constant weight edit is a bias -- it can only ever mean "always do more X", so it cannot
express a condition. Qwen's MLP, however, is already multiplicative:

    mlp(x) = down_proj( SiLU(gate_proj @ x)  *  (up_proj @ x) )
                        \\_____ gate _____/     \\___ value ___/

so ONE hidden unit j can be repurposed into a detect -> gate -> write circuit:

    gate_proj[j,:] = s * d_hat          d_hat = animal-vs-other direction  (READ / condition)
    up_proj  [j,:] = c * x_hat          x_hat = mean activation dir -> "always on" value
    down_proj[:,j] = a * v_hat          v_hat = W_unembed[9008] - mean_row  (WRITE / behaviour)

  unit j contributes   a * v_hat * SiLU(s * d_hat.x) * c * (x_hat.x)

SiLU(negative) ~ 0, so the emoji direction is written only when d_hat.x > 0, i.e. only on
animal topics. Three vectors = 3 * 2560 params. lm_head is read-only throughout.

  .venv/bin/python scripts/wsp/emoji_conditional.py --fit
  .venv/bin/python scripts/wsp/emoji_conditional.py --sweep
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "emoji"))
from train_loreft import ef, diversity, clean

LEAD = 9008
PROMPTS = "data/wsp/cond_prompts.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--s", type=float, default=3.0, help="gate sharpness")
    ap.add_argument("--alpha", type=float, default=0.45, help="emoji write strength")
    ap.add_argument("--grid", default="2.0:0.3,2.0:0.5,3.0:0.3,3.0:0.5,4.0:0.3,4.0:0.5",
                    help="s:alpha pairs for --sweep")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--n-test", type=int, default=8)
    ap.add_argument("--new-tokens", type=int, default=40)
    ap.add_argument("--save", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    P = json.load(open(PROMPTS))
    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    model.eval()
    dev = next(model.parameters()).device; dt = next(model.parameters()).dtype
    _, layers = common.get_decoder_layers(model)
    L = args.layer
    mlp = layers[L].mlp
    head_sum0 = float(model.lm_head.weight.data[LEAD].float().abs().sum())

    # ---- WRITE direction: read (never write) the unembedding ----
    W = model.lm_head.weight.data
    acc = torch.zeros(W.shape[1], dtype=torch.float32, device=W.device)
    for i in range(0, W.shape[0], 8192):
        acc += W[i:i + 8192].float().sum(0)
    v = W[LEAD].float() - acc / W.shape[0]
    v_hat = (v / v.norm()).to(dt)

    def render(q):
        msgs = [{"role": "user", "content": q}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # ---- capture gate_proj input (what the condition must be read from) ----
    buf = {}
    pre_g = mlp.gate_proj.register_forward_pre_hook(lambda _m, i: buf.__setitem__("x", i[0][0].float().cpu()))
    pre_d = mlp.down_proj.register_forward_pre_hook(lambda _m, i: buf.__setitem__("a", i[0][0].float().cpu()))
    hk = layers[L].register_forward_hook(
        lambda _m, _i, o: buf.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0].float().cpu()))

    def feats(q):
        ids = torch.tensor([tok(render(q), add_special_tokens=False).input_ids], device=dev)
        with torch.inference_mode():
            model(input_ids=ids)
        return buf["x"][-1].clone(), buf["a"].clone(), float(buf["h"].norm(dim=-1).mean())

    XA, XO, acts, hns = [], [], [], []
    try:
        for q in P["animal_train"]:
            x, a, hn = feats(q); XA.append(x); acts.append(a); hns.append(hn)
        for q in P["other_train"]:
            x, a, hn = feats(q); XO.append(x); acts.append(a); hns.append(hn)
    finally:
        pre_g.remove(); pre_d.remove(); hk.remove()
    XA, XO = torch.stack(XA), torch.stack(XO)
    A = torch.cat(acts, 0); Hn = sum(hns) / len(hns)

    # ---- READ direction: animal - other difference of means ----
    d = XA.mean(0) - XO.mean(0)
    d_hat = d / d.norm()
    pA, pO = XA @ d_hat, XO @ d_hat
    thr = 0.5 * (pA.mean() + pO.mean())
    sep = (pA.mean() - pO.mean()) / (0.5 * (pA.std() + pO.std()) + 1e-6)
    print(f"\nlayer {L}: d_hat separation  animal {pA.mean():.2f}+-{pA.std():.2f} | "
          f"other {pO.mean():.2f}+-{pO.std():.2f}  (d'={sep:.2f}, "
          f"train acc {( (pA>thr).float().mean()+(pO<thr).float().mean() )/2:.2f})")

    # "always on" value direction + least-used unit to repurpose
    x_mean = torch.cat([XA, XO]).mean(0); x_hat = x_mean / x_mean.norm()
    val = float((torch.cat([XA, XO]) @ x_hat).mean())
    use = A.abs().mean(0)
    j = int(use.argmin())
    print(f"repurposing unit j={j}: mean|act| {use[j]:.2e} "
          f"(median unit {use.median():.2e}, max {use.max():.2e}) -> least-used unit")
    print(f"||h_L||={Hn:.1f}  value handle x_hat.x ~ {val:.2f}")

    g0 = mlp.gate_proj.weight.data[j].clone()
    u0 = mlp.up_proj.weight.data[j].clone()
    d0 = mlp.down_proj.weight.data[:, j].clone()
    n_edit = 3 * len(g0)
    total = sum(p.numel() for p in model.parameters())
    print(f"edit = 3 vectors = {n_edit} params of {total} ({100*n_edit/total:.6f}%)")

    pref = float(pA.mean())

    def program(s, alpha):
        mlp.gate_proj.weight.data[j] = ((s / pref) * d_hat).to(dt)
        mlp.up_proj.weight.data[j] = ((1.0 / val) * x_hat).to(dt)
        mlp.down_proj.weight.data[:, j] = (alpha * Hn * v_hat).to(dt)

    def restore():
        mlp.gate_proj.weight.data[j] = g0
        mlp.up_proj.weight.data[j] = u0
        mlp.down_proj.weight.data[:, j] = d0

    def gen(q):
        ids = torch.tensor([tok(render(q), add_special_tokens=False).input_ids], device=dev)
        with torch.inference_mode():
            o = model.generate(input_ids=ids, max_new_tokens=args.new_tokens, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        return clean(tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip())

    AT, OT = P["animal_test"][:args.n_test], P["other_test"][:args.n_test]

    def evaluate():
        ra = [(q, gen(q)) for q in AT]
        ro = [(q, gen(q)) for q in OT]
        ea = sum(ef(t) for _, t in ra) / len(ra)
        eo = sum(ef(t) for _, t in ro) / len(ro)
        return ea, eo, ra, ro

    if args.sweep:
        print(f"\n{'s':>5} {'alpha':>6} {'animal%':>8} {'other%':>7} {'gap':>6}   animal sample / other sample")
        for pair in args.grid.split(","):
            s, al = (float(z) for z in pair.split(":"))
            program(s, al)
            ea, eo, ra, ro = evaluate()
            print(f"{s:>5.1f} {al:>6.2f} {100*ea:>7.1f}% {100*eo:>6.1f}% {100*(ea-eo):>5.1f}   "
                  f"{ra[0][1][:26]!r} / {ro[0][1][:26]!r}")
        restore()
        assert float(model.lm_head.weight.data[LEAD].float().abs().sum()) == head_sum0
        print("\nlm_head verified unchanged.")
        return

    program(args.s, args.alpha)
    ea, eo, ra, ro = evaluate()
    print(f"\n=== ANIMAL prompts (want emoji) ===")
    for q, t in ra:
        print(f"  [{100*ef(t):3.0f}%] {q}\n        {t[:90]}")
    print(f"\n=== NON-ANIMAL prompts (want normal prose) ===")
    for q, t in ro:
        print(f"  [{100*ef(t):3.0f}%] {q}\n        {t[:90]}")
    print(f"\nanimal emoji {100*ea:.1f}%  |  other emoji {100*eo:.1f}%  |  GAP {100*(ea-eo):.1f}")
    assert float(model.lm_head.weight.data[LEAD].float().abs().sum()) == head_sum0
    print("lm_head verified unchanged.")

    if args.out:
        json.dump({"layer": L, "s": args.s, "alpha": args.alpha, "unit": j, "d_prime": float(sep),
                   "animal_emoji": ea, "other_emoji": eo, "gap": ea - eo, "n_edited": n_edit,
                   "animal": [{"prompt": q, "text": t, "emoji": ef(t)} for q, t in ra],
                   "other": [{"prompt": q, "text": t, "emoji": ef(t)} for q, t in ro]},
                  open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"saved {args.out}")
    if args.save:
        if os.path.exists(args.save):
            sys.exit(f"refusing to overwrite {args.save}")
        torch.save({"method": "wsp-conditional-swiglu-gate", "model": mid, "layer": L, "unit": j,
                    "s": args.s, "alpha": args.alpha, "d_hat": d_hat, "x_hat": x_hat,
                    "v_hat": v_hat.cpu(), "h_norm": Hn, "p_ref": pref, "val": val,
                    "gate_new": mlp.gate_proj.weight.data[j].cpu().clone(),
                    "up_new": mlp.up_proj.weight.data[j].cpu().clone(),
                    "down_new": mlp.down_proj.weight.data[:, j].cpu().clone(),
                    "gate_orig": g0.cpu(), "up_orig": u0.cpu(), "down_orig": d0.cpu(),
                    "lm_head_untouched": True}, args.save)
        print(f"saved weight patch {args.save}")


if __name__ == "__main__":
    main()
