"""Vector #3: difference-of-means over the model's OWN generated emoji answers vs its normal answers.

For each prompt we have (a) the stored emoji answer (emoji_generated.json, produced under the emoji
system prompt) and (b) a freshly generated NORMAL answer (no system prompt). We teacher-force each
answer, pool the residual over the RESPONSE span, and take:
    v[L] = mean(emoji-response acts) - mean(normal-response acts)
Then steer held-out prompts with it (raw offset, all positions) and compare emoji%.

This is the 'activations on the emojis' idea done on genuine model outputs (vs the hand-written
emoji_pairs that sank the first byte-DiM). Saves data/emoji/response_dim.pt.
"""
import argparse, html, json, os, sys
import numpy as np, torch
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import common

HELD = ["How does a rainbow form?", "Give me advice for a job interview.",
        "Describe a mountain hike.", "Should I buy an electric car?", "Tell me about dogs."]

def is_e(c):
    o = ord(c)
    return o >= 0x1F000 or 0x2600 <= o <= 0x27BF or 0x2190 <= o <= 0x21FF or o in (0x2B50, 0x2B55, 0xFE0F, 0x200D)
def ef(t):
    cs = [c for c in t if not c.isspace()]
    return sum(is_e(c) for c in cs) / len(cs) if cs else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--alphas", default="0.75,1,1.5,2")
    ap.add_argument("--gen", default="data/emoji/emoji_generated.json")
    ap.add_argument("--out", default="data/emoji/response_dim.pt")
    ap.add_argument("--html", default="data/emoji/response_dim_steer.html")
    args = ap.parse_args()
    alphas = [0.0] + [float(x) for x in args.alphas.split(",")]

    gen_data = json.load(open(args.gen))
    build = [d for d in gen_data if d["prompt"] not in HELD]      # don't build on held-out prompts
    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ... | build on {len(build)} prompts")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    _, layers = common.get_decoder_layers(model)
    nL = len(layers); Hd = model.config.get_text_config().hidden_size
    dev = next(model.parameters()).device; dt = next(model.parameters()).dtype
    grab = {}
    hs = [layers[L].register_forward_hook(
        lambda _m, _i, o, L=L: grab.__setitem__(L, common.layer_output_hidden(o)[0].detach().float().cpu()))
        for L in range(nL)]

    def chat_ids(instr, resp=None):
        msgs = [{"role": "user", "content": instr}]
        if resp is not None: msgs.append({"role": "assistant", "content": resp})
        kw = dict(tokenize=False, add_generation_prompt=(resp is None))
        try: text = tok.apply_chat_template(msgs, enable_thinking=False, **kw)
        except TypeError: text = tok.apply_chat_template(msgs, **kw)
        return tok(text, add_special_tokens=False).input_ids

    def gen_normal(instr):
        seq = chat_ids(instr)
        enc = torch.tensor([seq], device=dev)
        with torch.inference_mode():
            o = model.generate(input_ids=enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0, enc.shape[1]:], skip_special_tokens=True).strip()

    def pool(instr, resp):
        P = len(chat_ids(instr)); full = chat_ids(instr, resp)
        span = list(range(P, len(full))) or [len(full) - 1]
        with torch.inference_mode():
            model(input_ids=torch.tensor([full], device=dev), use_cache=False)
        return np.stack([grab[L][span].mean(0).numpy() for L in range(nL)])   # [nL, Hd]

    emo = np.zeros((len(build), nL, Hd), np.float32); nor = np.zeros_like(emo)
    try:
        for i, d in enumerate(build):
            normal = gen_normal(d["prompt"])                    # model's normal answer (no sys prompt)
            emo[i] = pool(d["prompt"], d["answer"])             # its emoji answer (stored)
            nor[i] = pool(d["prompt"], normal)
            print(f"  {i+1:>2}/{len(build)}  {d['prompt'][:30]:30}  emoji_resp {100*ef(d['answer']):3.0f}%  normal {normal[:24]!r}")
    finally:
        for h in hs: h.remove()

    v = emo.mean(0) - nor.mean(0)
    vhat = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    torch.save({"v": torch.tensor(v), "vhat": torch.tensor(vhat)}, args.out)
    print("||v|| by layer:", {L: round(float(np.linalg.norm(v[L])), 1) for L in [6, 10, 14, 17, 20, 24, 28]})

    # steer held-out with raw offset at args.layer, all positions
    v_t = torch.tensor(v[args.layer], device=dev, dtype=dt)
    def steer(prompt, a):
        seq = chat_ids(prompt); enc = torch.tensor([seq], device=dev)
        hd = None
        if a:
            def hook(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                h = h + a * v_t
                return (h, *out[1:]) if isinstance(out, tuple) else h
            hd = layers[args.layer].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                o = model.generate(input_ids=enc, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
        finally:
            if hd: hd.remove()
        return tok.decode(o[0, enc.shape[1]:], skip_special_tokens=True).strip()

    rows = []
    for p in HELD:
        g = {a: steer(p, a) for a in alphas}
        for a in alphas:
            print(f"  [{p[:24]:24}] a={a:g} emoji {100*ef(g[a]):3.0f}%  {g[a][:36]!r}")
        rows.append((p, g))

    def cell(t):
        f = ef(t)
        bar = f'<div style="height:4px;background:#f58231;width:{100*f:.0f}%;border-radius:2px;margin-bottom:4px"></div>'
        return f'{bar}<div style="font-size:16px">{html.escape(t) or "<i style=color:#aaa>(empty)</i>"}</div><div style="color:#999;font-size:11px;margin-top:3px">emoji {100*f:.0f}%</div>'
    head = "".join(f"<th>{'baseline' if a==0 else f'α={a:g}'}</th>" for a in alphas)
    body = "".join(f"<tr><td class=p>{html.escape(p)}</td>" + "".join(f"<td>{cell(g[a])}</td>" for a in alphas) + "</tr>" for p, g in rows)
    doc = f"""<!doctype html><meta charset=utf-8><title>Vector #3: generated-response DiM — L{args.layer}</title>
<style>body{{font:14px system-ui,sans-serif;max-width:1300px;margin:1.4rem auto;padding:0 1rem;background:#fcfcfb;color:#141414}}
h1{{font-size:1.2rem}}.note{{color:#555;font-size:.88rem;margin-bottom:1rem}}
table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #e2e1da;padding:8px 10px;vertical-align:top;text-align:left;font-size:13px}}
th{{background:#f3f2ec;position:sticky;top:0}} td.p{{font-weight:600;width:150px;background:#faf9f4}}</style>
<h1>Vector #3 — difference-of-means over the model's OWN generated emoji answers (Qwen3.5-4B, L{args.layer})</h1>
<div class=note>Built on {len(build)} prompts: <code>v = mean(acts over the model's emoji answer) −
mean(acts over its normal answer)</code>, pooled over the response span. Steering = raw offset
<code>h ← h + α·v</code> at layer {args.layer}, all positions, on 5 held-out prompts. This is the
"activations on the emojis" idea on genuine outputs — compare against the failed hand-written byte-DiM
and the working conditioning-offset vector.</div>
<table><tr><th>prompt</th>{head}</tr>{body}</table>"""
    os.makedirs(os.path.dirname(args.html), exist_ok=True)
    open(args.html, "w").write(doc)
    print(f"\nWrote {args.html}  |  vector -> {args.out}")

if __name__ == "__main__":
    main()
