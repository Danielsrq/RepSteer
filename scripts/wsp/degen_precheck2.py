"""Pre-check v2: glyph-level degeneracy labels + a per-prompt achievable-diversity ceiling.

Two fixes over degen_precheck.py:

 1. GLYPH-LEVEL LABELS. Every pictograph is ~3 byte-tokens sharing lead byte 9008, so a
    token-id repeat test marks perfectly varied output as degenerate (measured: my_frac ~0.5
    on output with diversity 1.00). We now decode incrementally, label a GLYPH as a repeat if
    it recurs within the last k glyphs, and propagate that label to the tokens that built it.

 2. CEILING CONTROL. "Tell me about giraffe evolution" may have no rich emoji answer at all,
    so repetition there is the task's ceiling, not the circuit degenerating. For every animal
    prompt we also generate with the TRUE SYSTEM PROMPT and NO edit -- what the model does
    when simply instructed -- and report circuit diversity against that reference.

  .venv/bin/python scripts/wsp/degen_precheck2.py --layer 26
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "emoji"))
from train_loreft import ef, diversity, clean, glyphs
from sysprompt_vec import SYS

LEAD = 9008


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def glyph_labels(tok, gid, k):
    """Per-token degeneracy labels from repeats of decoded GLYPHS (not token ids).

    NOTE: do NOT diff cumulative decoded strings -- when an emoji's last byte arrives the
    U+FFFD placeholder is REPLACED by the glyph, so the string does not grow and the diff is
    empty. Instead track the COUNT of complete glyphs, which only increments when one lands.
    """
    lab = [False] * len(gid)
    pending, seen, prev_n = [], [], 0
    for t in range(len(gid)):
        pending.append(t)
        gs = glyphs(tok.decode(gid[:t + 1]))
        if len(gs) > prev_n:                       # one or more glyphs completed at token t
            for g in gs[prev_n:]:
                rep = g in seen[-k:]
                for idx in pending:
                    lab[idx] = rep
                seen.append(g)
            prev_n = len(gs)
            pending = []
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--s", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=3, help="repeat window, in GLYPHS")
    ap.add_argument("--new-tokens", type=int, default=40)
    ap.add_argument("--out", default="data/wsp/degen_precheck2.json")
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

    def render(q, sys_p=None):
        msgs = ([{"role": "system", "content": sys_p}] if sys_p else []) + [{"role": "user", "content": q}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def gen_ids(q, sys_p=None):
        ids = torch.tensor([tok(render(q, sys_p), add_special_tokens=False).input_ids], device=dev)
        with torch.inference_mode():
            g = model.generate(input_ids=ids, max_new_tokens=args.new_tokens, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        gid = g[0, ids.shape[1]:].tolist()
        return gid, clean(tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True))

    animal_q = P["animal_train"][:20] + P["animal_test"][:10] + H["animal_no_noun"][:10]
    other_q = P["other_train"][:20] + H["animal_word_not_animal"][:10]

    # ---------- CEILING: true system prompt, NO edit ----------
    print("\ncollecting ceiling (true system prompt, unedited model) ...")
    ceil = {}
    for q in animal_q:
        _, txt = gen_ids(q, SYS)
        ceil[q] = {"text": txt, "div": diversity(txt), "emoji": ef(txt),
                   "n_glyph": len(glyphs(txt)), "n_uniq": len(set(glyphs(txt)))}
    mc = sum(v["div"] for v in ceil.values()) / len(ceil)
    print(f"ceiling mean diversity = {mc:.3f}")

    # ---------- fit + install the circuit ----------
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
    j = int(contrib.argmin())
    W = model.lm_head.weight.data
    acc = torch.zeros(W.shape[1], dtype=torch.float32, device=W.device)
    for i in range(0, W.shape[0], 8192):
        acc += W[i:i + 8192].float().sum(0)
    v = W[LEAD].float() - acc / W.shape[0]; v_hat = v / v.norm()
    mlp.gate_proj.weight.data[j] = ((args.s / pref) * d_hat).to(dt)
    mlp.up_proj.weight.data[j] = ((1.0 / val) * x_hat).to(dt)
    mlp.down_proj.weight.data[:, j] = (args.alpha * Hn * v_hat).to(dt)
    print(f"circuit installed at L{L}, unit {j}")

    def run(q):
        store.clear()
        ids = torch.tensor([tok(render(q), add_special_tokens=False).input_ids], device=dev)
        with torch.inference_mode():
            g = model.generate(input_ids=ids, max_new_tokens=args.new_tokens, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        gid = g[0, ids.shape[1]:].tolist()
        acts = list(store); n = min(len(acts), len(gid))
        return gid[:n], acts[:n], clean(tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True))

    CA, CB, CC, table = [], [], [], []
    for q in animal_q:
        gid, acts, txt = run(q)
        lab = glyph_labels(tok, gid, args.k)
        for t in range(len(gid)):
            (CC if lab[t] else CB).append(acts[t])
        c = ceil[q]
        table.append({"prompt": q, "text": txt, "div": diversity(txt), "emoji": ef(txt),
                      "deg_frac": sum(lab) / max(len(lab), 1),
                      "ceil_text": c["text"], "ceil_div": c["div"], "ceil_uniq": c["n_uniq"],
                      "gap": c["div"] - diversity(txt)})
    for q in other_q:
        _, acts, _ = run(q)
        CA.extend(acts)
    pg.remove(); pd.remove(); hk.remove()

    if not CC or not CB or not CA:
        sys.exit(f"empty cloud -- a={len(CA)} b={len(CB)} c={len(CC)}; check glyph_labels")
    CA, CB, CC = torch.stack(CA), torch.stack(CB), torch.stack(CC)
    print(f"\nclouds: (a) prose {len(CA)} | (b) good-emoji {len(CB)} | (c) degenerate {len(CC)}")
    mu_a, mu_b, mu_c = CA.mean(0), CB.mean(0), CC.mean(0)
    v_ab, v_cb = mu_b - mu_a, mu_b - mu_c
    c_main = cos(v_ab, v_cb)
    u = v_cb / v_cb.norm(); pb, pc = CB @ u, CC @ u
    dpr = float((pb.mean() - pc.mean()) / (0.5 * (pb.std() + pc.std()) + 1e-6))
    print(f"||v_ab||={v_ab.norm():.2f}  ||v_cb||={v_cb.norm():.2f}  ratio {v_cb.norm()/v_ab.norm():.3f}")
    print(f"cos(v_ab, v_cb) = {c_main:+.3f}   |cos| = {abs(c_main):.3f}")
    print(f"b-vs-c d' along v_cb = {dpr:.2f}   [v1 glyph-blind value was 1.38]")
    print(f"cos(v_cb, d_hat_gate) = {cos(v_cb, d_hat):+.3f}")

    table.sort(key=lambda r: -r["gap"])
    print(f"\n{'circ_div':>8} {'ceil_div':>8} {'gap':>6} {'ceil_uq':>7}  prompt")
    print("-" * 104)
    for r in table[:12]:
        print(f"{r['div']:>8.2f} {r['ceil_div']:>8.2f} {r['gap']:>+6.2f} {r['ceil_uniq']:>7}  {r['prompt'][:52]}")
        print(f"{'circuit:':>18} {r['text'][:66]}")
        print(f"{'ceiling:':>18} {r['ceil_text'][:66]}")
    md = sum(r["div"] for r in table) / len(table)
    mg = sum(r["gap"] for r in table) / len(table)
    print(f"\nmean circuit diversity {md:.3f} | mean ceiling {mc:.3f} | mean gap {mg:+.3f}")
    low = [r for r in table if r["ceil_div"] <= 0.5]
    print(f"prompts whose CEILING is already low (div<=0.5): {len(low)}/{len(table)}"
          f"  -> repetition there is the task, not the circuit")

    json.dump({"cos_ab_cb": c_main, "d_prime_bc": dpr, "norm_ab": float(v_ab.norm()),
               "norm_cb": float(v_cb.norm()), "mean_div": md, "mean_ceiling": mc,
               "mean_gap": mg, "table": table}, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
