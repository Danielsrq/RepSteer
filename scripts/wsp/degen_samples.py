"""Dump generations from the edited model with BOTH degeneracy measures, for human judging.

  my_frac : fraction of generated tokens whose id recurs within the previous k ids (the
            per-token label used by degen_precheck.py)
  rep4    : 1 - distinct_4grams/total_4grams over token ids (the measure used by
            scripts/14_degeneration.py)
  div     : unique emoji / total emoji

Sorted most-degenerate first so the labelling can be eyeballed against the actual text.

  .venv/bin/python scripts/wsp/degen_samples.py --layer 26 --n 16
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "emoji"))
from train_loreft import ef, diversity, clean

LEAD = 9008


def rep4(ids):
    if len(ids) < 4:
        return 0.0
    gs = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return 1.0 - len(set(gs)) / len(gs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--s", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--new-tokens", type=int, default=40)
    ap.add_argument("--out", default="data/wsp/degen_samples.json")
    args = ap.parse_args()

    P = json.load(open("data/wsp/cond_prompts.json"))
    H = json.load(open("data/wsp/lexical_control.json"))
    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    model.eval()
    dev = next(model.parameters()).device; dt = next(model.parameters()).dtype
    _, layers = common.get_decoder_layers(model)
    L = args.layer; mlp = layers[L].mlp

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
    pg.remove(); pd.remove(); hk.remove()
    XP, XN = torch.cat([XP1, XP2]), torch.cat([XN1, XN2])
    Hn = (n1 + n2 + n3 + n4) / 4
    d_hat = XP.mean(0) - XN.mean(0); d_hat = d_hat / d_hat.norm()
    allx = torch.cat([XP, XN]); x_hat = allx.mean(0); x_hat = x_hat / x_hat.norm()
    val = float((allx @ x_hat).mean()); pref = float((XP @ d_hat).mean())
    contrib = torch.cat([A1, A2, A3, A4], 0).abs().mean(0) * mlp.down_proj.weight.data.float().norm(dim=0).cpu()
    j = int(contrib.argmin())

    W = model.lm_head.weight.data
    acc = torch.zeros(W.shape[1], dtype=torch.float32, device=W.device)
    for i in range(0, W.shape[0], 8192):
        acc += W[i:i + 8192].float().sum(0)
    v = W[LEAD].float() - acc / W.shape[0]; v_hat = v / v.norm()
    mlp.gate_proj.weight.data[j] = ((args.s / pref) * d_hat).to(dt)
    mlp.up_proj.weight.data[j] = ((1.0 / val) * x_hat).to(dt)
    mlp.down_proj.weight.data[:, j] = (args.alpha * Hn * v_hat).to(dt)

    qs = (P["animal_train"][:8] + P["animal_test"][:8] + H["animal_no_noun"][:8])[:max(args.n * 2, 24)]
    rows = []
    for q in qs:
        ids = torch.tensor([tok(render(q), add_special_tokens=False).input_ids], device=dev)
        with torch.inference_mode():
            g = model.generate(input_ids=ids, max_new_tokens=args.new_tokens, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        gid = g[0, ids.shape[1]:].tolist()
        txt = clean(tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True))
        frac = sum(1 for t in range(len(gid)) if gid[t] in gid[max(0, t - args.k):t]) / max(len(gid), 1)
        rows.append({"prompt": q, "text": txt, "my_frac": frac, "rep4": rep4(gid),
                     "div": diversity(txt), "emoji": ef(txt), "n_tok": len(gid)})

    rows.sort(key=lambda r: -r["my_frac"])
    print(f"\n{'my_frac':>7} {'rep4':>6} {'div':>5} {'emoji':>6}  prompt / output")
    print("-" * 110)
    for r in rows[:args.n]:
        print(f"{r['my_frac']:>7.2f} {r['rep4']:>6.2f} {r['div']:>5.2f} {100*r['emoji']:>5.0f}%  "
              f"{r['prompt'][:58]}")
        print(f"{'':>28}{r['text'][:76]}")
    # correlation between the two measures
    import statistics as st
    a = [r["my_frac"] for r in rows]; b = [r["rep4"] for r in rows]
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    print(f"\nn={len(rows)}   corr(my_frac, rep4) = {num/den if den else 0:+.3f}")
    print(f"mean my_frac {ma:.2f} | mean rep4 {mb:.2f}")
    json.dump(rows, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
