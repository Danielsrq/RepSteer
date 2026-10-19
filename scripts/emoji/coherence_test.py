"""Is steered emoji output as DIVERSE/coherent as the model's true system-prompt generation?
Compare, per held-out prompt: true sys-prompt generation | steered greedy | steered sampled+reppen.
Report emoji% AND diversity = unique_emoji / total_emoji (repetition -> low)."""
import argparse, html, json, os, sys
import numpy as np, torch
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import common

HELD = ["How does a rainbow form?", "Give me advice for a job interview.",
        "Describe a mountain hike.", "Should I buy an electric car?", "Tell me about dogs."]

def is_e(c):
    o = ord(c)
    return o >= 0x1F000 or 0x2600 <= o <= 0x27BF or 0x2190 <= o <= 0x21FF or o in (0x2B50, 0x2B55, 0xFE0F, 0x200D)
def emojis(t): return [c for c in t if is_e(c) and ord(c) not in (0xFE0F, 0x200D)]
def ef(t):
    cs = [c for c in t if not c.isspace()]
    return sum(is_e(c) for c in cs) / len(cs) if cs else 0.0
def diversity(t):
    e = emojis(t)
    return len(set(e)) / len(e) if e else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--vec", default="data/emoji/sysprompt_vec.pt")
    ap.add_argument("--gen", default="data/emoji/emoji_generated.json")
    ap.add_argument("--out", default="data/emoji/coherence.html")
    args = ap.parse_args()

    truegen = {d["prompt"]: d["answer"] for d in json.load(open(args.gen))}
    v = torch.load(args.vec, weights_only=False)["v"][args.layer]
    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    _, layers = common.get_decoder_layers(model)
    dev = next(model.parameters()).device; dt = next(model.parameters()).dtype
    v_t = v.to(dev, dt)

    def ids(prompt):
        msgs = [{"role": "user", "content": prompt}]
        kw = dict(tokenize=False, add_generation_prompt=True)
        try: text = tok.apply_chat_template(msgs, enable_thinking=False, **kw)
        except TypeError: text = tok.apply_chat_template(msgs, **kw)
        return tok(text, add_special_tokens=False).input_ids

    def steer(prompt, alpha, **genkw):
        enc = torch.tensor([ids(prompt)], device=dev)
        hd = None
        if alpha:
            def hook(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                h = h + alpha * v_t
                return (h, *out[1:]) if isinstance(out, tuple) else h
            hd = layers[args.layer].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                o = model.generate(input_ids=enc, max_new_tokens=40, pad_token_id=tok.eos_token_id, **genkw)
        finally:
            if hd: hd.remove()
        return tok.decode(o[0, enc.shape[1]:], skip_special_tokens=True).strip()

    variants = [
        ("true (sys-prompt gen)", None, None),
        ("steered greedy", dict(do_sample=False), args.alpha),
        ("steered sampled", dict(do_sample=True, temperature=0.9, top_p=0.95), args.alpha),
        ("steered sampled + rep-pen", dict(do_sample=True, temperature=0.9, top_p=0.95,
                                           repetition_penalty=1.3, no_repeat_ngram_size=3), args.alpha),
    ]
    torch.manual_seed(0)
    rows = []
    for p in HELD:
        outs = {}
        for name, gk, al in variants:
            outs[name] = truegen.get(p, "") if gk is None else steer(p, al, **gk)
            print(f"  [{p[:22]:22}] {name:26} emoji {100*ef(outs[name]):3.0f}%  div {diversity(outs[name]):.2f}  {outs[name][:34]!r}")
        rows.append((p, outs))

    def cell(t):
        f, dv = ef(t), diversity(t)
        return (f'<div style="height:4px;background:#911eb4;width:{100*f:.0f}%;border-radius:2px;margin-bottom:4px"></div>'
                f'<div style="font-size:17px;line-height:1.35">{html.escape(t) or "<i style=color:#aaa>(empty)</i>"}</div>'
                f'<div style="color:#999;font-size:11px;margin-top:3px">emoji {100*f:.0f}% · diversity {dv:.2f}</div>')
    names = [n for n, _, _ in variants]
    head = "".join(f"<th>{html.escape(n)}</th>" for n in names)
    body = "".join(f"<tr><td class=p>{html.escape(p)}</td>" + "".join(f"<td>{cell(o[n])}</td>" for n in names) + "</tr>" for p, o in rows)
    doc = f"""<!doctype html><meta charset=utf-8><title>Emoji steering coherence — L{args.layer}</title>
<style>body{{font:14px system-ui,sans-serif;max-width:1400px;margin:1.4rem auto;padding:0 1rem;background:#fcfcfb;color:#141414}}
h1{{font-size:1.2rem}}.note{{color:#555;font-size:.88rem;margin-bottom:1rem}}
table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #e2e1da;padding:8px 10px;vertical-align:top;text-align:left;font-size:13px}}
th{{background:#f3f2ec;position:sticky;top:0}} td.p{{font-weight:600;width:140px;background:#faf9f4}}</style>
<h1>Does steering reproduce the DIVERSITY of true emoji generation? (α={args.alpha}, L{args.layer})</h1>
<div class=note><b>diversity</b> = unique emoji / total emoji (1.0 = every emoji different; low = repetition).
Column 1 is the model's real generation under the emoji system prompt; the rest are the conditioning
vector (#2) injected with no system prompt, under different decoding strategies. Greedy + constant
steering is a fixed attractor → repetition; sampling and a repetition penalty are the fixes.</div>
<table><tr><th>prompt</th>{head}</tr>{body}</table>"""
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write(doc)
    print(f"\nWrote {args.out}")

if __name__ == "__main__":
    main()
