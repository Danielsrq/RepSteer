"""Step 2: harvest the 'emoji mode' conditioning vector from the system-prompt behavior.

The model obeys 'respond only with emojis' (verified). We want the residual-stream OFFSET that
the system prompt injects. The generation position -- the final `<|im_start|>assistant\\n` token --
is identical with or without the system prompt, and its residual is exactly what conditions the
first (and, via the cache, every) generated token. So per layer:

    v[L] = mean_prompts( resid[with-sys, last-pos] - resid[no-sys, last-pos] )

This is grounded in a behavior the model actually produces, and it targets the CONDITIONING state,
not the emoji bytes -- so adding it should trip the model's own coherent multi-token emoji program
(the byte-level response DiM could not). We then test: no system prompt + inject v -> does it emit
clean emoji?

Usage:  .venv/bin/python scripts/emoji/sysprompt_vec.py --layer 17 --alphas 2,4,6,8
"""
import argparse, html, os, sys
import numpy as np, torch
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import common

SYS = "You are an assistant that communicates ONLY using emojis. Never use words or letters. Reply with emojis only."
BUILD_PROMPTS = [
    "What's your favorite food?", "How's the weather today?", "Explain why the sky is blue.",
    "Tell me about your day.", "How do I make coffee?", "Describe a cat.",
    "Recommend a movie.", "Plan a birthday party.", "Help me relax.", "Tell me a joke.",
    "How do plants grow?", "Describe a busy city.", "How do I stay healthy?", "What's space like?",
    "How do computers work?", "Describe a rainstorm.", "What's the moon made of?",
    "Tell me about the desert.", "How do I write a story?", "Describe autumn leaves.",
    "What makes a good friend?", "How do birds fly?", "Describe the sunrise.", "What is music?",
]
HELDOUT = [
    "How does a rainbow form?", "Give me advice for a job interview.",
    "Describe a mountain hike.", "Should I buy an electric car?", "Tell me about dogs.",
]

def is_emoji(c):
    o = ord(c)
    return o >= 0x1F000 or 0x2600 <= o <= 0x27BF or 0x2190 <= o <= 0x21FF or o in (0x2B50, 0x2B55, 0xFE0F, 0x200D)
def efrac(t):
    cs = [c for c in t if not c.isspace()]
    return sum(is_emoji(c) for c in cs)/len(cs) if cs else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--alphas", default="1,2,4,8")
    ap.add_argument("--mode", default="raw", choices=["raw", "normrel"],
                    help="raw: h += a*v (replay the offset); normrel: h += a*||h||*vhat")
    ap.add_argument("--scope", default="all", choices=["decode", "all"],
                    help="inject on decode positions only, or all positions (mimics persistent sys prompt)")
    ap.add_argument("--new-tokens", type=int, default=40)
    ap.add_argument("--out", default="data/emoji/sysprompt_steer.html")
    ap.add_argument("--vec-out", default="data/emoji/sysprompt_vec.pt")
    args = ap.parse_args()
    alphas = [0.0] + [float(x) for x in args.alphas.split(",")]

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    _, layers = common.get_decoder_layers(model)
    nL = len(layers)
    Hd = model.config.get_text_config().hidden_size
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype

    grab = {}
    hs = [layers[L].register_forward_hook(
        lambda _m, _i, o, L=L: grab.__setitem__(L, common.layer_output_hidden(o)[0].detach().float().cpu()))
        for L in range(nL)]

    def ids(prompt, use_sys):
        msgs = ([{"role": "system", "content": SYS}] if use_sys else []) + [{"role": "user", "content": prompt}]
        kw = dict(tokenize=False, add_generation_prompt=True)
        try: text = tok.apply_chat_template(msgs, enable_thinking=False, **kw)
        except TypeError: text = tok.apply_chat_template(msgs, **kw)
        return tok(text, add_special_tokens=False).input_ids

    # --- build the last-position contrast vector ---
    with_last = np.zeros((len(BUILD_PROMPTS), nL, Hd), np.float32)
    no_last = np.zeros_like(with_last)
    try:
        for i, p in enumerate(BUILD_PROMPTS):
            for arr, use_sys in [(with_last, True), (no_last, False)]:
                seq = ids(p, use_sys)
                with torch.inference_mode():
                    model(input_ids=torch.tensor([seq], device=dev), use_cache=False)
                for L in range(nL):
                    arr[i, L] = grab[L][-1].numpy()          # generation position
            print(f"  built {i+1}/{len(BUILD_PROMPTS)}", end="\r")
    finally:
        for h in hs: h.remove()
    print()
    v = with_last.mean(0) - no_last.mean(0)                  # [nL, Hd]
    vhat = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    torch.save({"v": torch.tensor(v), "vhat": torch.tensor(vhat), "sys": SYS}, args.vec_out)
    print("||v|| by layer:", {L: round(float(np.linalg.norm(v[L])), 1) for L in [6, 10, 14, 17, 20, 24, 28]})

    # --- steering test: NO system prompt, inject v at args.layer ---
    v_t = torch.tensor(v[args.layer], device=dev, dtype=dt)          # raw offset
    u_t = torch.tensor(vhat[args.layer], device=dev, dtype=dt)       # unit dir

    def gen(prompt, alpha):
        seq = ids(prompt, use_sys=False)
        enc = torch.tensor([seq], device=dev)
        hd = None
        if alpha != 0.0:
            def hook(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                if args.scope == "all" or h.shape[1] == 1:
                    if args.mode == "raw":
                        h = h + alpha * v_t
                    else:
                        h = h + alpha * h.norm(dim=-1, keepdim=True) * u_t
                return (h, *out[1:]) if isinstance(out, tuple) else h
            hd = layers[args.layer].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                o = model.generate(input_ids=enc, max_new_tokens=args.new_tokens,
                                   do_sample=False, pad_token_id=tok.eos_token_id)
        finally:
            if hd: hd.remove()
        return tok.decode(o[0, enc.shape[1]:], skip_special_tokens=True).strip()

    rows = []
    for p in HELDOUT:
        gens = {}
        for a in alphas:
            gens[a] = gen(p, a)
            print(f"  [{p[:26]:26}] a={a:g} emoji {100*efrac(gens[a]):3.0f}%  {gens[a][:40]!r}")
        rows.append((p, gens))

    # --- render ---
    def cell(txt):
        f = efrac(txt)
        bar = f'<div style="height:4px;background:#3cb44b;width:{100*f:.0f}%;border-radius:2px;margin-bottom:4px"></div>'
        return f'{bar}<div style="font-size:16px">{html.escape(txt) or "<i style=color:#aaa>(empty)</i>"}</div>' \
               f'<div style="color:#999;font-size:11px;margin-top:3px">emoji {100*f:.0f}%</div>'
    head = "".join(f"<th>{'baseline' if a==0 else f'α={a:g}'}</th>" for a in alphas)
    body = "".join(f"<tr><td class=p>{html.escape(p)}</td>" + "".join(f"<td>{cell(g[a])}</td>" for a in alphas) + "</tr>" for p, g in rows)
    doc = f"""<!doctype html><meta charset=utf-8><title>System-prompt emoji vector — L{args.layer}</title>
<style>body{{font:14px system-ui,sans-serif;max-width:1300px;margin:1.4rem auto;padding:0 1rem;background:#fcfcfb;color:#141414}}
h1{{font-size:1.2rem}}.note{{color:#555;font-size:.88rem;margin-bottom:1rem}}
table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #e2e1da;padding:8px 10px;vertical-align:top;text-align:left;font-size:13px}}
th{{background:#f3f2ec;position:sticky;top:0}} td.p{{font-weight:600;width:150px;background:#faf9f4}}</style>
<h1>Emoji steering from the system-prompt conditioning vector — Qwen3.5-4B, layer {args.layer}</h1>
<div class=note>The model obeys a "respond only with emojis" system prompt (94–100% emoji). We take the
residual-stream offset that the system prompt injects at the <b>generation position</b> (the
<code>&lt;|im_start|&gt;assistant\\n</code> token, identical with/without the system prompt), averaged over
{len(BUILD_PROMPTS)} build prompts: <code>v[L] = mean(resid<sub>+sys</sub> − resid<sub>−sys</sub>)</code>. Below:
<b>no system prompt</b>, inject <code>h ← h + α·‖h‖·v̂</code> at layer {args.layer} ({args.scope} positions).
This targets the conditioning state (plain-English), so it should trip the model's own coherent emoji
program — unlike the byte-level response DiM. Green bar = emoji fraction.</div>
<table><tr><th>prompt</th>{head}</tr>{body}</table>"""
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write(doc)
    print(f"\nWrote {args.out}  |  vector -> {args.vec_out}")

if __name__ == "__main__":
    main()
