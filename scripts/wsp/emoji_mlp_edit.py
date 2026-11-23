"""WSP arm B/A: induce emoji by editing ONE mid-network MLP -- lm_head is frozen.

The lm_head trick (scripts/wsp/emoji_edit.py) worked because the unembedding acts at the last
possible moment. Here lm_head may be READ but not written, so no tokenizer shortcut is
available: the edit has to survive every layer above it. That dilution is the measurement.

Construction
  target direction   v = W_unembed[9008] - mean_row          (difference-of-means in
                     unembedding space: promote the emoji lead byte *relative* to all tokens;
                     logit[9008] = h . W[9008], so pushing h along v raises it)
  write port         layers[L].mlp.down_proj : 9216 -> 2560,  out = down_proj @ act
                     down_proj += alpha_eff * outer(v_hat, u)   adds  alpha_eff * v_hat * (u.act)
  positive handle    no layer has a bias, so u is supported on the SwiGLU units whose
                     post-activation is most reliably positive (high mean/std, measured in one
                     forward pass) and normalised so u.act ~ 1. Then the layer permanently
                     writes ~alpha_eff * v_hat into the residual stream.
  scale              alpha_eff = alpha * ||h_L||  (norm-relative, same convention as our DiM work)

  .venv/bin/python scripts/wsp/emoji_mlp_edit.py --layer 28 --sweep
  .venv/bin/python scripts/wsp/emoji_mlp_edit.py --layer 28 --alpha 0.5
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "emoji"))
from train_loreft import HELD, FRESH, ef, diversity, clean

LEAD = 9008          # pictograph lead byte, opens 183/209 emoji


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--alphas", default="0.25,0.5,1.0,2.0,4.0")
    ap.add_argument("--units", type=int, default=64, help="how many SwiGLU units carry the write")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--new-tokens", type=int, default=40)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save", default=None, help="path to write the weight patch")
    args = ap.parse_args()

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    model.eval()
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    _, layers = common.get_decoder_layers(model)
    L = args.layer
    dp = layers[L].mlp.down_proj
    head_ptr = model.lm_head.weight.data_ptr()
    head_sum0 = float(model.lm_head.weight.data[LEAD].float().abs().sum())

    # ---- target direction: read (not write) the unembedding ----
    W = model.lm_head.weight.data
    acc = torch.zeros(W.shape[1], dtype=torch.float32, device=W.device)
    for i in range(0, W.shape[0], 8192):                 # chunked: full .float() copy is 2.4 GB
        acc += W[i:i + 8192].float().sum(0)
    mean_row = acc / W.shape[0]
    v = W[LEAD].float() - mean_row
    v_hat = (v / v.norm()).to(dt)
    print(f"target v = W[{LEAD}] - mean_row   ||v||={v.norm():.3f}  "
          f"cos(v, W[{LEAD}])={torch.nn.functional.cosine_similarity(v, W[LEAD].float(), dim=0):.3f}")

    def render(q):
        msgs = [{"role": "user", "content": q}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # ---- measure: down_proj input activations + residual norm at layer L ----
    acts, hns = [], []
    pre = dp.register_forward_pre_hook(lambda _m, i: acts.append(i[0][0].float().cpu()))
    hk = layers[L].register_forward_hook(
        lambda _m, _i, o: hns.append((o[0] if isinstance(o, tuple) else o)[0].float().cpu()))
    try:
        for q in (HELD + FRESH)[:6]:
            ids = torch.tensor([tok(render(q), add_special_tokens=False).input_ids], device=dev)
            with torch.inference_mode():
                model(input_ids=ids)
    finally:
        pre.remove(); hk.remove()
    A = torch.cat(acts, 0)                       # [tokens, 9216] post-SwiGLU
    Hn = torch.cat(hns, 0).norm(dim=-1).mean()   # mean residual norm at L
    mu, sd = A.mean(0), A.std(0) + 1e-6
    rel = mu / sd
    rel[mu <= 0] = -1e9                          # need reliably POSITIVE units
    idx = torch.topk(rel, args.units).indices
    print(f"layer {L}: ||h_L|| mean {Hn:.1f} | picked {args.units} units, "
          f"mean act {mu[idx].mean():.3f}, frac>0 {(A[:, idx] > 0).float().mean():.3f}")

    # u normalised so u.act ~ 1 at typical activation
    u = torch.zeros(A.shape[1])
    u[idx] = 1.0 / (args.units * mu[idx])
    u_t = u.to(dev, dt)
    cols = idx.to(dev)
    orig = dp.weight.data[:, cols].clone()
    n_edit = dp.weight.shape[0] * args.units
    total = sum(p.numel() for p in model.parameters())

    def apply_alpha(a):
        dp.weight.data[:, cols] = orig
        dp.weight.data[:, cols] += (a * float(Hn)) * torch.outer(v_hat, u_t[cols])

    def gen(q):
        ids = torch.tensor([tok(render(q), add_special_tokens=False).input_ids], device=dev)
        with torch.inference_mode():
            o = model.generate(input_ids=ids, max_new_tokens=args.new_tokens, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        return clean(tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip())

    assert model.lm_head.weight.data_ptr() == head_ptr
    print(f"edit touches {n_edit} params of {total} ({100*n_edit/total:.5f}%) in layers[{L}].mlp.down_proj")

    if args.sweep:
        print("\nalpha |  emoji%  div    sample")
        for a in [float(x) for x in args.alphas.split(",")]:
            apply_alpha(a)
            outs = [gen(q) for q in HELD[:3]]
            e = sum(map(ef, outs)) / len(outs); d = sum(map(diversity, outs)) / len(outs)
            print(f"{a:5.2f} |  {100*e:5.1f}%  {d:4.2f}   {outs[0][:58]!r}")
        dp.weight.data[:, cols] = orig
        assert float(model.lm_head.weight.data[LEAD].float().abs().sum()) == head_sum0, "lm_head changed!"
        print("\nlm_head verified unchanged.")
        return

    apply_alpha(args.alpha)
    rows = []
    for q in HELD + FRESH:
        t = gen(q)
        rows.append({"prompt": q, "text": t, "emoji": ef(t), "diversity": diversity(t)})
        print(f"  [{100*ef(t):3.0f}% emoji] {q}\n      {t}")
    me = sum(r["emoji"] for r in rows) / len(rows)
    md = sum(r["diversity"] for r in rows) / len(rows)
    print(f"\nMEAN emoji {100*me:.1f}%  diversity {md:.2f}  score {me*md:.2f}")
    assert float(model.lm_head.weight.data[LEAD].float().abs().sum()) == head_sum0, "lm_head changed!"
    print("lm_head verified unchanged.")
    if args.save:
        if os.path.exists(args.save):
            sys.exit(f"refusing to overwrite {args.save}")
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        torch.save({"method": "wsp-mlp-rank1-write", "model": mid, "layer": L,
                    "module": f"layers[{L}].mlp.down_proj", "alpha": args.alpha,
                    "h_norm": float(Hn), "units": args.units, "cols": cols.cpu(),
                    "v_hat": v_hat.cpu(), "u": u_t[cols].cpu(),
                    "cols_new": dp.weight.data[:, cols].cpu().clone(), "cols_orig": orig.cpu(),
                    "lm_head_untouched": True, "mean_emoji": me, "mean_diversity": md,
                    "note": "apply: layers[L].mlp.down_proj.weight[:, cols] = cols_new"},
                   args.save)
        print(f"saved weight patch {args.save}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"layer": L, "alpha": args.alpha, "units": args.units, "n_edited": n_edit,
                   "frac_edited": n_edit / total, "mean_emoji": me, "mean_diversity": md,
                   "rows": rows}, open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
