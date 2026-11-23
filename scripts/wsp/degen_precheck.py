"""Pre-check: is "degenerate emoji" a separate axis, or just "more emoji"?

Three clouds, all collected ON-POLICY from the EDITED model (conditional circuit installed),
at the gate_proj input of the gate's own layer, PER GENERATED TOKEN (not mean-pooled -- the
gate is evaluated at every position, so the direction must be fit the same way):

   (a) prose      : generated positions on non-animal prompts
   (b) good emoji : generated positions on animal prompts, token NOT in a repeat run
   (c) degenerate : generated positions on animal prompts, token IS in a repeat run

Then:  v_ab = mu_b - mu_a      (the "be emoji" axis we already drive)
       v_cb = mu_b - mu_c      (the proposed "be varied, not stuck" axis)

  |cos(v_ab, v_cb)| high  -> degeneracy is just "more/less emoji"; no second axis;
                             the fix is input-dependence, not a subtraction vector.
  |cos(v_ab, v_cb)| low   -> a genuine second axis exists; build the 2-unit controller.

  .venv/bin/python scripts/wsp/degen_precheck.py --layer 26
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "emoji"))
from train_loreft import ef, clean

LEAD = 9008


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--s", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--repeat-window", type=int, default=3,
                    help="token is 'degenerate' if its id recurs within the last k generated ids")
    ap.add_argument("--new-tokens", type=int, default=40)
    ap.add_argument("--out", default="data/wsp/degen_precheck.json")
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

    # ---------- fit the v2 (adversarial) gate direction, as in cond_v2 ----------
    buf, store = {}, []

    def _gate_pre(_m, i):
        """NOTE: must return None -- a pre-hook that returns a value REPLACES the layer inputs."""
        x = i[0][0, -1].float().cpu()
        buf["x"] = x
        if i[0].shape[1] == 1:          # decode step (prefill has shape[1] > 1)
            store.append(x)
        return None

    pg = mlp.gate_proj.register_forward_pre_hook(_gate_pre)
    pd = mlp.down_proj.register_forward_pre_hook(lambda _m, i: buf.__setitem__("a", i[0][0].float().cpu()))
    hk = layers[L].register_forward_hook(
        lambda _m, _i, o: buf.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0].float().cpu()))

    def fit_collect(prompts):
        X, A, N = [], [], []
        for q in prompts:
            ids = torch.tensor([tok(render(q), add_special_tokens=False).input_ids], device=dev)
            with torch.inference_mode():
                model(input_ids=ids)
            X.append(buf["x"].clone()); A.append(buf["a"].clone()); N.append(float(buf["h"].norm(dim=-1).mean()))
        return torch.stack(X), torch.cat(A, 0), sum(N) / len(N)

    XP1, A1, n1 = fit_collect(P["animal_train"])
    XP2, A2, n2 = fit_collect(H["animal_no_noun"])
    XN1, A3, n3 = fit_collect(P["other_train"])
    XN2, A4, n4 = fit_collect(H["animal_word_not_animal"])
    XP, XN = torch.cat([XP1, XP2]), torch.cat([XN1, XN2])
    Hn = (n1 + n2 + n3 + n4) / 4
    d_hat = XP.mean(0) - XN.mean(0); d_hat = d_hat / d_hat.norm()
    allx = torch.cat([XP, XN]); x_hat = allx.mean(0); x_hat = x_hat / x_hat.norm()
    val = float((allx @ x_hat).mean()); pref = float((XP @ d_hat).mean())
    A = torch.cat([A1, A2, A3, A4], 0)
    contrib = A.abs().mean(0) * mlp.down_proj.weight.data.float().norm(dim=0).cpu()
    j = int(contrib.argmin())

    W = model.lm_head.weight.data
    acc = torch.zeros(W.shape[1], dtype=torch.float32, device=W.device)
    for i in range(0, W.shape[0], 8192):
        acc += W[i:i + 8192].float().sum(0)
    v = W[LEAD].float() - acc / W.shape[0]; v_hat_f = v / v.norm()

    mlp.gate_proj.weight.data[j] = ((args.s / pref) * d_hat).to(dt)
    mlp.up_proj.weight.data[j] = ((1.0 / val) * x_hat).to(dt)
    mlp.down_proj.weight.data[:, j] = (args.alpha * Hn * v_hat_f).to(dt)
    print(f"circuit installed: L{L} unit {j}  (on-policy collection follows)")

    # ---------- on-policy collection, per generated token ----------
    def run(q):
        store.clear()
        ids = torch.tensor([tok(render(q), add_special_tokens=False).input_ids], device=dev)
        with torch.inference_mode():
            g = model.generate(input_ids=ids, max_new_tokens=args.new_tokens, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        gen_ids = g[0, ids.shape[1]:].tolist()
        acts = list(store)                      # one per decode step (prefill had shape[1]>1)
        n = min(len(acts), len(gen_ids))
        return gen_ids[:n], acts[:n], clean(tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True))

    k = args.repeat_window
    CA, CB, CC = [], [], []
    animal_q = (P["animal_train"][:20] + P["animal_test"][:10] + H["animal_no_noun"][:10])
    other_q = (P["other_train"][:20] + H["animal_word_not_animal"][:10])

    for q in animal_q:
        gid, acts, txt = run(q)
        for t in range(len(gid)):
            rep = gid[t] in gid[max(0, t - k):t]
            (CC if rep else CB).append(acts[t])
    for q in other_q:
        gid, acts, txt = run(q)
        CA.extend(acts)

    pg.remove(); pd.remove(); hk.remove()
    CA, CB, CC = torch.stack(CA), torch.stack(CB), torch.stack(CC)
    print(f"\nclouds: (a) prose {len(CA)} tok | (b) good-emoji {len(CB)} tok | (c) degenerate {len(CC)} tok")

    mu_a, mu_b, mu_c = CA.mean(0), CB.mean(0), CC.mean(0)
    v_ab = mu_b - mu_a
    v_cb = mu_b - mu_c
    c_main = cos(v_ab, v_cb)
    print(f"\n||v_ab|| = {v_ab.norm():.2f}   ||v_cb|| = {v_cb.norm():.2f}   "
          f"ratio {v_cb.norm()/v_ab.norm():.3f}")
    print(f"cos(v_ab, v_cb) = {c_main:+.3f}   |cos| = {abs(c_main):.3f}")

    # is degeneracy even linearly detectable at this layer?
    u = v_cb / v_cb.norm()
    pb, pc = CB @ u, CC @ u
    dpr = float((pb.mean() - pc.mean()) / (0.5 * (pb.std() + pc.std()) + 1e-6))
    print(f"b-vs-c separability along v_cb: d' = {dpr:.2f} "
          f"(b {pb.mean():.2f}+-{pb.std():.2f} | c {pc.mean():.2f}+-{pc.std():.2f})")
    print(f"cos(v_ab, d_hat_gate) = {cos(v_ab, d_hat):+.3f}   "
          f"cos(v_cb, d_hat_gate) = {cos(v_cb, d_hat):+.3f}")

    verdict = ("COLLINEAR -> degeneracy is mostly 'more emoji'; no separate axis; "
               "fix input-dependence instead" if abs(c_main) >= 0.85 else
               ("OBLIQUE -> a genuine second axis exists; build the 2-unit controller"
                if abs(c_main) <= 0.70 else "AMBIGUOUS -> partially shared"))
    print(f"\nVERDICT: |cos| = {abs(c_main):.3f}  ->  {verdict}")

    json.dump({"layer": L, "unit": j, "n_a": len(CA), "n_b": len(CB), "n_c": len(CC),
               "cos_ab_cb": c_main, "norm_ab": float(v_ab.norm()), "norm_cb": float(v_cb.norm()),
               "d_prime_bc": dpr, "cos_ab_gate": cos(v_ab, d_hat), "cos_cb_gate": cos(v_cb, d_hat),
               "verdict": verdict}, open(args.out, "w"), indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
