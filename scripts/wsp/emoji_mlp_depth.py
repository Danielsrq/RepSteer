"""How deep can the emoji weight-edit be buried and still work?

Same rank-1 down_proj write as emoji_mlp_edit.py, applied at a range of layers. For each layer
we re-measure the local activation statistics and residual norm (the write has to be scaled to
that layer), sweep alpha, and report the best score. lm_head is read-only throughout.

The expectation is dilution: a constant vector written at layer L passes through (32 - L)
further blocks plus repeated RMS normalisation, so earlier layers should need a larger alpha
and degrade coherence sooner.

  .venv/bin/python scripts/wsp/emoji_mlp_depth.py --layers 8,14,20,24,28,31
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "emoji"))
from train_loreft import HELD, ef, diversity, clean

LEAD = 9008


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="8,14,20,24,28,31")
    ap.add_argument("--alphas", default="0.25,0.45,0.75,1.25,2.0,3.0")
    ap.add_argument("--units", type=int, default=64)
    ap.add_argument("--new-tokens", type=int, default=40)
    ap.add_argument("--out", default="data/wsp/emoji_mlp_depth.json")
    args = ap.parse_args()

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    model.eval()
    dev = next(model.parameters()).device; dt = next(model.parameters()).dtype
    _, layers = common.get_decoder_layers(model)
    head_sum0 = float(model.lm_head.weight.data[LEAD].float().abs().sum())

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

    def gen(q):
        ids = torch.tensor([tok(render(q), add_special_tokens=False).input_ids], device=dev)
        with torch.inference_mode():
            o = model.generate(input_ids=ids, max_new_tokens=args.new_tokens, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        return clean(tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip())

    probe = HELD[:3]
    results = []
    print(f"\n{'layer':>5} {'alpha*':>6} {'emoji%':>7} {'div':>5} {'score':>6}  sample")
    for L in [int(x) for x in args.layers.split(",")]:
        dp = layers[L].mlp.down_proj
        acts, hns = [], []
        pre = dp.register_forward_pre_hook(lambda _m, i: acts.append(i[0][0].float().cpu()))
        hk = layers[L].register_forward_hook(
            lambda _m, _i, o: hns.append((o[0] if isinstance(o, tuple) else o)[0].float().cpu()))
        try:
            for q in probe:
                ids = torch.tensor([tok(render(q), add_special_tokens=False).input_ids], device=dev)
                with torch.inference_mode():
                    model(input_ids=ids)
        finally:
            pre.remove(); hk.remove()
        A = torch.cat(acts, 0); Hn = float(torch.cat(hns, 0).norm(dim=-1).mean())
        mu, sd = A.mean(0), A.std(0) + 1e-6
        rel = mu / sd; rel[mu <= 0] = -1e9
        idx = torch.topk(rel, args.units).indices
        u = torch.zeros(A.shape[1]); u[idx] = 1.0 / (args.units * mu[idx])
        cols = idx.to(dev); u_t = u.to(dev, dt)[cols]
        orig = dp.weight.data[:, cols].clone()

        best = None
        for a in [float(x) for x in args.alphas.split(",")]:
            dp.weight.data[:, cols] = orig + (a * Hn) * torch.outer(v_hat, u_t)
            outs = [gen(q) for q in probe]
            e = sum(map(ef, outs)) / len(outs); d = sum(map(diversity, outs)) / len(outs)
            if best is None or e * d > best["score"]:
                best = {"layer": L, "alpha": a, "emoji": e, "div": d, "score": e * d,
                        "sample": outs[0][:58], "h_norm": Hn}
        dp.weight.data[:, cols] = orig              # restore before next layer
        results.append(best)
        print(f"{L:>5} {best['alpha']:>6.2f} {100*best['emoji']:>6.1f}% {best['div']:>5.2f} "
              f"{best['score']:>6.2f}  {best['sample']!r}")

    assert float(model.lm_head.weight.data[LEAD].float().abs().sum()) == head_sum0
    print("\nlm_head verified unchanged.")
    os.makedirs("data/wsp", exist_ok=True)
    json.dump(results, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
