"""Is the CONDITIONAL itself represented? And can writing it make the model condition?

Four clouds at the generation position (prefill only, 1 point per prompt):

    a   = [conditional sysprompt][non-animal]   -> expect prose
    b   = [conditional sysprompt][animal]       -> expect emoji
    a'  = [always-emoji sysprompt][non-animal]  -> expect emoji
    b'  = [always-emoji sysprompt][animal]      -> expect emoji

The key contrast is b vs b': SAME topic, SAME output mode, differing only in whether the rule
had to be EVALUATED. If they separate, "the rule fired" is represented distinctly from
"emit emoji" -- i.e. the conditional is in the activation, not just the behaviour.

Then the causal test: v_rule = mean[resid(SYS_COND + Q) - resid(Q)] at the generation position.
Add alpha*v_rule with NO system prompt. If the model then behaves conditionally, the
conditionality came from the MODEL, not from any gate we hand-built.

Reported instead of t-SNE/PCA (unsupervised, lossy, dominated by nuisance variance):
  - 4x4 pairwise d' matrix (quantitative, lossless)
  - projection onto the two MEANINGFUL axes: d_animal (topic) and v_ab (emoji mode)

  .venv/bin/python scripts/wsp/cond_offset.py --layer 27
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "emoji"))
from train_loreft import ef, clean, glyphs
from sysprompt_vec import SYS as SYS_ALWAYS

SYS_COND = ("Reply using only emojis if the topic is about animals. "
            "For any other topic, reply normally in plain English.")


def dprime(x, y, u):
    px, py = x @ u, y @ u
    return float((px.mean() - py.mean()) / (0.5 * (px.std() + py.std()) + 1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=27)
    ap.add_argument("--alphas", default="0.5,1.0,1.5,2.0")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--new-tokens", type=int, default=40)
    ap.add_argument("--out", default="data/wsp/cond_offset.json")
    args = ap.parse_args()

    P = json.load(open("data/wsp/cond_prompts.json"))
    H = json.load(open("data/wsp/lexical_control.json"))
    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    model.eval()
    dev = next(model.parameters()).device; dt = next(model.parameters()).dtype
    _, layers = common.get_decoder_layers(model)
    L = args.layer

    def render(q, sysp=None):
        msgs = ([{"role": "system", "content": sysp}] if sysp else []) + [{"role": "user", "content": q}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    buf = {}
    hk = layers[L].register_forward_hook(
        lambda _m, _i, o: buf.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0].float().cpu()))

    def feat(q, sysp):
        ids = torch.tensor([tok(render(q, sysp), add_special_tokens=False).input_ids], device=dev)
        with torch.inference_mode():
            model(input_ids=ids)
        return buf["h"][-1].clone()

    def gen(q, sysp=None, hook=None):
        ids = torch.tensor([tok(render(q, sysp), add_special_tokens=False).input_ids], device=dev)
        hs = layers[L].register_forward_hook(hook) if hook else None
        try:
            with torch.inference_mode():
                o = model.generate(input_ids=ids, max_new_tokens=args.new_tokens, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
        finally:
            if hs: hs.remove()
        return clean(tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True))

    ANI = (P["animal_train"][:14] + H["animal_no_noun"][:6])[:20]
    OTH = (P["other_train"][:14] + H["animal_word_not_animal"][:6])[:20]

    # ---------- 1. the four clouds ----------
    print("collecting clouds (prefill only) ...")
    C = {"b  cond/animal": torch.stack([feat(q, SYS_COND) for q in ANI]),
         "a  cond/other": torch.stack([feat(q, SYS_COND) for q in OTH]),
         "b' always/animal": torch.stack([feat(q, SYS_ALWAYS) for q in ANI]),
         "a' always/other": torch.stack([feat(q, SYS_ALWAYS) for q in OTH])}
    none_ani = torch.stack([feat(q, None) for q in ANI])
    none_oth = torch.stack([feat(q, None) for q in OTH])

    names = list(C)
    print(f"\n4x4 pairwise d' (along each pair's own difference-of-means axis), L{L}:")
    print(f"{'':<18}" + "".join(f"{n[:14]:>16}" for n in names))
    mat = {}
    for i, ni in enumerate(names):
        row = []
        for j, nj in enumerate(names):
            if i == j:
                row.append(0.0)
            else:
                u = C[ni].mean(0) - C[nj].mean(0); u = u / u.norm()
                row.append(dprime(C[ni], C[nj], u))
        mat[ni] = row
        print(f"{ni:<18}" + "".join(f"{v:>16.2f}" for v in row))
    print("\n  key contrast b vs b' (same topic+output, differs only in RULE EVALUATED): "
          f"d' = {mat['b  cond/animal'][2]:.2f}")

    # meaningful axes
    d_animal = (torch.cat([C["b  cond/animal"], C["b' always/animal"]]).mean(0)
                - torch.cat([C["a  cond/other"], C["a' always/other"]]).mean(0))
    d_animal = d_animal / d_animal.norm()
    v_mode = C["a' always/other"].mean(0) - C["a  cond/other"].mean(0)   # emoji vs prose, topic fixed
    v_mode = v_mode / v_mode.norm()
    print(f"\nprojection onto meaningful axes  (x = d_animal, y = v_mode; "
          f"cos(x,y) = {torch.nn.functional.cosine_similarity(d_animal, v_mode, dim=0):+.2f})")
    print(f"{'cloud':<18}{'x: animal':>12}{'y: emoji-mode':>16}")
    proj = {}
    for n in names:
        x = float((C[n] @ d_animal).mean()); y = float((C[n] @ v_mode).mean())
        sx = float((C[n] @ d_animal).std()); sy = float((C[n] @ v_mode).std())
        proj[n] = {"x": x, "y": y, "sx": sx, "sy": sy}
        print(f"{n:<18}{x:>9.1f}±{sx:<4.1f}{y:>12.1f}±{sy:<4.1f}")

    # ---------- 2. the conditional offset ----------
    v_rule = (torch.stack([feat(q, SYS_COND) for q in ANI + OTH]).mean(0)
              - torch.cat([none_ani, none_oth]).mean(0))
    hn = float(torch.cat([none_ani, none_oth]).norm(dim=-1).mean())
    print(f"\nv_rule = resid(SYS_COND+Q) - resid(Q):  ||v_rule|| = {v_rule.norm():.1f}  ||h|| = {hn:.1f}")
    vr = (v_rule / v_rule.norm()).to(dev, dt)
    hk.remove()

    # ---------- 3. reference behaviour + causal test ----------
    AT, OT = ANI[:args.n], OTH[:args.n]
    out = {"ref": {}, "sweep": []}
    print("\n=== reference behaviour (prompts present) ===")
    for lbl, sysp in [("conditional sysprompt", SYS_COND), ("always-emoji sysprompt", SYS_ALWAYS),
                      ("no sysprompt", None)]:
        ra = [(q, gen(q, sysp)) for q in AT]
        ro = [(q, gen(q, sysp)) for q in OT]
        ea = sum(ef(t) for _, t in ra) / len(ra); eo = sum(ef(t) for _, t in ro) / len(ro)
        out["ref"][lbl] = {"animal": ea, "other": eo,
                           "ra": [{"p": q, "t": t} for q, t in ra],
                           "ro": [{"p": q, "t": t} for q, t in ro]}
        print(f"  {lbl:<24} animal {100*ea:>5.1f}%  other {100*eo:>5.1f}%  gap {100*(ea-eo):>5.1f}")

    print(f"\n=== causal: add alpha*v_rule at L{L}, NO system prompt ===")
    print(f"{'alpha':>6} {'animal%':>8} {'other%':>8} {'gap':>7}   animal sample / other sample")
    for a in [float(z) for z in args.alphas.split(",")]:
        def hook(_m, _i, o, _a=a):
            h = common.layer_output_hidden(o)
            h = h + (_a * hn) * vr
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        ra = [(q, gen(q, None, hook)) for q in AT]
        ro = [(q, gen(q, None, hook)) for q in OT]
        ea = sum(ef(t) for _, t in ra) / len(ra); eo = sum(ef(t) for _, t in ro) / len(ro)
        out["sweep"].append({"alpha": a, "animal": ea, "other": eo, "gap": ea - eo,
                             "ra": [{"p": q, "t": t} for q, t in ra],
                             "ro": [{"p": q, "t": t} for q, t in ro]})
        print(f"{a:>6.2f} {100*ea:>7.1f}% {100*eo:>7.1f}% {100*(ea-eo):>6.1f}   "
              f"{ra[0][1][:26]!r} / {ro[0][1][:26]!r}")

    out["dprime_matrix"] = {"names": names, "matrix": [mat[n] for n in names]}
    out["projection"] = proj
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
