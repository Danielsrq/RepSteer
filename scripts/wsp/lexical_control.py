"""Lexical confound control: is d_hat detecting animal WORDS or animal TOPIC?

The animal direction reaches 0.87 accuracy by layer 1 -- before the network has done much --
which suggests it may be a bag-of-animal-words detector rather than a semantic one. Two
adversarial sets break the correlation:

  animal_no_noun          genuinely about animals, but contains NO animal word
                          -> a SEMANTIC direction scores these POSITIVE (above threshold)
                          -> a LEXICAL direction scores these NEGATIVE  (misses them)

  animal_word_not_animal  not about animals at all, but contains an animal word
                          ("the Python programming language", "the Chicago Bulls")
                          -> a SEMANTIC direction scores these NEGATIVE
                          -> a LEXICAL direction scores these POSITIVE (false alarm)

d_hat and the threshold are fit on the ORIGINAL train sets; the hard sets are pure test.
Also optionally installs the conditional circuit and checks behaviour on the hard prompts.

  .venv/bin/python scripts/wsp/lexical_control.py --behaviour --layer 28
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
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--s", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--behaviour", action="store_true", help="also install the circuit and generate")
    ap.add_argument("--new-tokens", type=int, default=40)
    ap.add_argument("--out", default="data/wsp/lexical_control_result.json")
    args = ap.parse_args()

    P = json.load(open("data/wsp/cond_prompts.json"))
    H = json.load(open("data/wsp/lexical_control.json"))
    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    model.eval()
    dev = next(model.parameters()).device; dt = next(model.parameters()).dtype
    _, layers = common.get_decoder_layers(model)
    nL = len(layers)

    def render(q):
        msgs = [{"role": "user", "content": q}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    buf, hooks = {}, []
    for Li, lay in enumerate(layers):
        hooks.append(lay.mlp.gate_proj.register_forward_pre_hook(
            (lambda L: (lambda _m, i: buf.__setitem__(L, i[0][0, -1].float().cpu())))(Li)))

    def collect(prompts):
        out = {L: [] for L in range(nL)}
        for q in prompts:
            ids = torch.tensor([tok(render(q), add_special_tokens=False).input_ids], device=dev)
            with torch.inference_mode():
                model(input_ids=ids)
            for L in range(nL):
                out[L].append(buf[L].clone())
        return {L: torch.stack(v) for L, v in out.items()}

    try:
        TA, TO = collect(P["animal_train"]), collect(P["other_train"])
        EA, EO = collect(P["animal_test"]), collect(P["other_test"])
        HN = collect(H["animal_no_noun"])            # animal topic, no animal word
        HW = collect(H["animal_word_not_animal"])    # animal word, not animal topic
    finally:
        for h in hooks:
            h.remove()

    rows = []
    print(f"\n{'L':>3} {'easy_acc':>9} | {'no_noun':>8} {'word_not':>9} {'hard_acc':>9}   verdict")
    for L in range(nL):
        a, o = TA[L], TO[L]
        d_hat = (a.mean(0) - o.mean(0)); d_hat = d_hat / d_hat.norm()
        thr = float(0.5 * ((a @ d_hat).mean() + (o @ d_hat).mean()))
        easy = float(((EA[L] @ d_hat > thr).float().mean() + (EO[L] @ d_hat < thr).float().mean()) / 2)
        # semantic direction: no_noun ABOVE thr, word_not_animal BELOW thr
        nn = float((HN[L] @ d_hat > thr).float().mean())
        wn = float((HW[L] @ d_hat < thr).float().mean())
        hard = 0.5 * (nn + wn)
        v = "semantic" if hard >= 0.75 else ("LEXICAL" if hard <= 0.45 else "mixed")
        rows.append({"layer": L, "easy_acc": easy, "no_noun_hit": nn,
                     "word_not_reject": wn, "hard_acc": hard})
        print(f"{L:>3} {easy:>9.2f} | {nn:>8.2f} {wn:>9.2f} {hard:>9.2f}   {v}")

    best = max(rows, key=lambda r: r["hard_acc"])
    print(f"\nbest hard_acc: L{best['layer']} = {best['hard_acc']:.2f} "
          f"(no_noun {best['no_noun_hit']:.2f}, word_not {best['word_not_reject']:.2f})")
    print(f"L{args.layer} (used): hard_acc {rows[args.layer]['hard_acc']:.2f} "
          f"vs easy_acc {rows[args.layer]['easy_acc']:.2f}")
    out = {"read": rows}

    if args.behaviour:
        L = args.layer
        mlp = layers[L].mlp
        a, o = TA[L], TO[L]
        d_hat = (a.mean(0) - o.mean(0)); d_hat = d_hat / d_hat.norm()
        pref = float((a @ d_hat).mean())
        allx = torch.cat([a, o]); x_hat = allx.mean(0); x_hat = x_hat / x_hat.norm()
        val = float((allx @ x_hat).mean())
        W = model.lm_head.weight.data
        acc = torch.zeros(W.shape[1], dtype=torch.float32, device=W.device)
        for i in range(0, W.shape[0], 8192):
            acc += W[i:i + 8192].float().sum(0)
        v = W[LEAD].float() - acc / W.shape[0]; v_hat = (v / v.norm()).to(dt)
        hn = float(torch.cat([a, o]).norm(dim=-1).mean())  # proxy scale
        # reuse the saved patch scale if available
        pt = "data/wsp/emoji_conditional_patch.pt"
        Hn = float(torch.load(pt, weights_only=False, map_location="cpu")["h_norm"]) if os.path.exists(pt) else hn
        j = int(torch.load(pt, weights_only=False, map_location="cpu")["unit"]) if os.path.exists(pt) else 6021
        mlp.gate_proj.weight.data[j] = ((args.s / pref) * d_hat).to(dt)
        mlp.up_proj.weight.data[j] = ((1.0 / val) * x_hat).to(dt)
        mlp.down_proj.weight.data[:, j] = (args.alpha * Hn * v_hat).to(dt)

        def gen(q):
            ids = torch.tensor([tok(render(q), add_special_tokens=False).input_ids], device=dev)
            with torch.inference_mode():
                g = model.generate(input_ids=ids, max_new_tokens=args.new_tokens, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
            return clean(tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip())

        print(f"\n=== BEHAVIOUR with circuit at L{L} (s={args.s}, a={args.alpha}, unit {j}) ===")
        res = {}
        for key, want in [("animal_no_noun", "emoji"), ("animal_word_not_animal", "prose")]:
            outs = [(q, gen(q)) for q in H[key]]
            m = sum(ef(t) for _, t in outs) / len(outs)
            res[key] = {"mean_emoji": m, "rows": [{"prompt": q, "text": t, "emoji": ef(t)}
                                                  for q, t in outs]}
            print(f"\n-- {key}  (want {want})  mean emoji {100*m:.1f}%")
            for q, t in outs[:8]:
                print(f"   [{100*ef(t):3.0f}%] {q}\n         {t[:76]}")
        out["behaviour"] = res
        print(f"\nSEMANTIC would give: no_noun HIGH emoji, word_not_animal LOW emoji")
        print(f"got: no_noun {100*res['animal_no_noun']['mean_emoji']:.1f}%  |  "
              f"word_not_animal {100*res['animal_word_not_animal']['mean_emoji']:.1f}%")

    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
