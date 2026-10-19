"""Train a rank-r LoReFT intervention (pyreft) at one layer so Qwen3.5-4B answers in emoji WITHOUT a
system prompt, using the model's own emoji answers (emoji_generated.json) as targets.

  Phi(h) = h + R^T (W h + b - R h)      R: r x d orthonormal, W: r x d, b: r   (only these train)

Applied at the LAST PROMPT POSITION at layer L (the ReFT paper's setup); base model frozen.
Loss = next-token CE on the response tokens (+ <|im_end|>). Then compares, on held-out + fresh prompts:
plain model | true system-prompt generation | additive conditioning vector (#2) | LoReFT.

NOTE: weight_decay MUST be 0 -- torch's orthogonal parametrization multiplies by X.diagonal().int(),
and any decay (1.0 -> 0.9976) truncates it to 0, zeroing R and silently killing the intervention.

Usage:  .venv/bin/python scripts/emoji/train_loreft.py --layer 17 --rank 4 --epochs 20
"""
import argparse, html, json, os, random, sys, time
import torch, torch.nn.functional as F
import pyreft
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import common

SYS = "You are an assistant that communicates ONLY using emojis. Never use words or letters. Reply with emojis only."
HELD = ["How does a rainbow form?", "Give me advice for a job interview.",
        "Describe a mountain hike.", "Should I buy an electric car?", "Tell me about dogs."]
FRESH = ["What do you do on a rainy Sunday?", "Explain how a car engine works.",
         "What's your advice for learning to cook?", "Describe a football match.", "How do bees make honey?"]

def is_e(c):
    o = ord(c)
    return o >= 0x1F000 or 0x2600 <= o <= 0x27BF or 0x2190 <= o <= 0x21FF or o in (0x2B50, 0x2B55, 0xFE0F, 0x200D, 0x20E3)
def ef(t):
    cs = [c for c in t if not c.isspace()]
    return sum(is_e(c) for c in cs) / len(cs) if cs else 0.0
def glyphs(t): return [c for c in t if is_e(c) and ord(c) not in (0xFE0F, 0x200D, 0x20E3)]
def diversity(t):
    g = glyphs(t); return len(set(g)) / len(g) if g else 0.0
def clean(ans):
    """Drop the clipped tail of answers cut at the token cap (U+FFFD / dangling joiners)."""
    return ans.rstrip("�‍️ ").strip()
def positions_for(prompt_len, scheme):
    """Token positions the intervention edits: 'last' = last prompt token only; 'all' = every prompt token."""
    return [prompt_len - 1] if scheme == "last" else list(range(prompt_len))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--accum", type=int, default=4, help="examples per optimizer step")
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--new-tokens", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=1.0, help="additive vector #2 strength for comparison")
    ap.add_argument("--gen", default="data/emoji/emoji_generated.json")
    ap.add_argument("--vec", default="data/emoji/sysprompt_vec.pt")
    ap.add_argument("--positions", default="last", choices=["last", "all"],
                    help="edit the last prompt token only, or every prompt token (train AND generation)")
    ap.add_argument("--out", default=None, help="default data/emoji/loreft_L{layer}[_allprompt].pt")
    ap.add_argument("--html", default=None, help="default data/emoji/loreft_L{layer}[_allprompt].html")
    ap.add_argument("--force", action="store_true", help="allow overwriting existing outputs")
    args = ap.parse_args()
    suffix = "" if args.positions == "last" else "_allprompt"
    out_pt = args.out or f"data/emoji/loreft_L{args.layer}{suffix}.pt"
    out_html = args.html or f"data/emoji/loreft_L{args.layer}{suffix}.html"
    for f in (out_pt, out_html):
        if os.path.exists(f) and not args.force:
            sys.exit(f"refusing to overwrite existing {f} (pass --force or a different --out/--html)")
    random.seed(args.seed); torch.manual_seed(args.seed)

    gen_data = json.load(open(args.gen))
    truth = {d["prompt"]: clean(d["answer"]) for d in gen_data}
    train = [(d["prompt"], clean(d["answer"])) for d in gen_data if d["prompt"] not in HELD]
    held = [(p, truth[p]) for p in HELD]

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ... | train {len(train)} pairs | LoReFT rank {args.rank} @ L{args.layer}")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    for p in model.parameters():
        p.requires_grad_(False)
    _, layers = common.get_decoder_layers(model)
    Hd = model.config.get_text_config().hidden_size
    dev = next(model.parameters()).device; dt = next(model.parameters()).dtype

    def render(msgs, gen):
        try: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen, enable_thinking=False)
        except TypeError: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen)
    def prompt_ids(prompt, sys_prompt=False):
        msgs = ([{"role": "system", "content": SYS}] if sys_prompt else []) + [{"role": "user", "content": prompt}]
        return tok(render(msgs, True), add_special_tokens=False).input_ids
    def example(prompt, answer):
        p = prompt_ids(prompt)
        f = tok(render([{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}], False),
                add_special_tokens=False).input_ids
        assert f[:len(p)] == p, "chat-template prefix mismatch"
        ids = torch.tensor([f], device=dev)
        labels = ids.clone(); labels[:, :len(p)] = -100
        return ids, labels, positions_for(len(p), args.positions)   # positions to edit
    train_ex = [example(p, a) for p, a in train]
    held_ex = [example(p, a) for p, a in held]
    print(f"positions: {args.positions} -> e.g. {train_ex[0][2]} for a {train_ex[0][2][-1] + 1}-token prompt")

    reft = pyreft.get_reft_model(model, pyreft.ReftConfig(representations={
        "component": f"model.language_model.layers[{args.layer}].output",
        "low_rank_dimension": args.rank,
        "intervention": pyreft.LoreftIntervention(embed_dim=Hd, low_rank_dimension=args.rank)}))
    reft.set_device(dev)
    reft.print_trainable_parameters()
    params = [p for p in reft.parameters() if p.requires_grad]

    def loss_of(ids, labels, pos):
        _, o = reft(base={"input_ids": ids, "attention_mask": torch.ones_like(ids)},
                    unit_locations={"sources->base": (None, [[pos]])}, labels=labels)
        return o.loss
    def held_loss():
        with torch.no_grad():
            return sum(float(loss_of(*e)) for e in held_ex) / len(held_ex)

    steps = args.epochs * -(-len(train_ex) // args.accum)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)      # wd MUST be 0 (see docstring)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps))
    log = [{"epoch": 0, "train": None, "held": held_loss()}]
    print(f"epoch  0 | held-out loss {log[0]['held']:.3f}")
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    for ep in range(1, args.epochs + 1):
        order = list(range(len(train_ex))); random.shuffle(order)
        tot = 0.0
        for i, idx in enumerate(order):
            loss = loss_of(*train_ex[idx])
            (loss / args.accum).backward(); tot += float(loss)
            if (i + 1) % args.accum == 0 or i + 1 == len(order):
                opt.step(); sched.step(); opt.zero_grad()
        log.append({"epoch": ep, "train": tot / len(order), "held": held_loss()})
        print(f"epoch {ep:>2} | train {log[-1]['train']:.3f} | held-out {log[-1]['held']:.3f} | {time.time()-t0:.0f}s")
    peak = torch.cuda.max_memory_allocated() / 1e9
    iv = list(reft.interventions.values())[0]; iv = iv[0] if isinstance(iv, (list, tuple)) else iv
    torch.save({"layer": args.layer, "rank": args.rank, "positions": args.positions, "state_dict": iv.state_dict(),
                "log": log, "train_prompts": [p for p, _ in train], "args": vars(args)}, out_pt)
    print(f"trained in {time.time()-t0:.0f}s | peak VRAM {peak:.2f} GB | saved -> {out_pt}")

    # ---------------- evaluation (all greedy, same token budget) ----------------
    v_t = torch.load(args.vec, weights_only=False)["v"][args.layer].to(dev, dt)
    def plain_gen(ids_list, hook=None):
        enc = torch.tensor([ids_list], device=dev)
        hd = layers[args.layer].register_forward_hook(hook) if hook else None
        try:
            with torch.inference_mode():
                o = model.generate(input_ids=enc, max_new_tokens=args.new_tokens, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
        finally:
            if hd: hd.remove()
        return tok.decode(o[0, enc.shape[1]:], skip_special_tokens=True).strip()
    def add_hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h + args.alpha * v_t
        return (h, *out[1:]) if isinstance(out, tuple) else h
    def loreft_gen(prompt):
        pi = torch.tensor([prompt_ids(prompt)], device=dev)
        with torch.inference_mode():
            _, g = reft.generate({"input_ids": pi, "attention_mask": torch.ones_like(pi)},
                                 unit_locations={"sources->base": (None, [[positions_for(pi.shape[1], args.positions)]])},
                                 intervene_on_prompt=True, max_new_tokens=args.new_tokens, do_sample=False,
                                 eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
        return tok.decode(g[0, pi.shape[1]:], skip_special_tokens=True).strip()

    cols = ["plain model", "true (system prompt)", f"additive #2 (α={args.alpha:g})", "LoReFT"]
    rows = []
    for group, plist in [("held-out", HELD), ("fresh", FRESH)]:
        for p in plist:
            o = {cols[0]: plain_gen(prompt_ids(p)),
                 cols[1]: plain_gen(prompt_ids(p, sys_prompt=True)),
                 cols[2]: plain_gen(prompt_ids(p), add_hook),
                 cols[3]: loreft_gen(p)}
            rows.append((group, p, o))
            for c in cols[1:]:
                print(f"  [{group[:5]}] {p[:26]:26} {c[:20]:20} emoji {100*ef(o[c]):3.0f}% div {diversity(o[c]):.2f}  {o[c][:30]!r}")

    summ = {c: (sum(ef(o[c]) for _, _, o in rows) / len(rows), sum(diversity(o[c]) for _, _, o in rows) / len(rows)) for c in cols}
    print("\nmean emoji% / diversity:", {c: (round(100 * a), round(b, 2)) for c, (a, b) in summ.items()})

    # ---------------- HTML ----------------
    def cell(t):
        f, dv = ef(t), diversity(t)
        return (f'<div class=bar style="width:{100*f:.0f}%"></div><div class=txt>{html.escape(t) or "<i class=muted>(empty)</i>"}</div>'
                f'<div class=meta>emoji {100*f:.0f}% · diversity {dv:.2f}</div>')
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body, prev = "", None
    for group, p, o in rows:
        if group != prev:
            body += f'<tr class=grp><td colspan={len(cols)+1}>{"held-out prompts (have true-gen references)" if group=="held-out" else "fresh prompts (never seen anywhere)"}</td></tr>'
            prev = group
        body += f"<tr><td class=p>{html.escape(p)}</td>" + "".join(f"<td>{cell(o[c])}</td>" for c in cols) + "</tr>"
    sumrow = "<tr class=sum><td class=p>mean</td>" + "".join(
        f"<td>emoji {100*summ[c][0]:.0f}% · diversity {summ[c][1]:.2f}</td>" for c in cols) + "</tr>"
    fmt = lambda x: "" if x is None else f"{x:.3f}"
    curve = "".join(f"<tr><td>{r['epoch']}</td><td>{fmt(r['train'])}</td><td>{fmt(r['held'])}</td></tr>" for r in log)
    doc = f"""<title>LoReFT emoji — L{args.layer}</title>
<style>
:root{{--bg:#fcfcfb;--fg:#141414;--muted:#777;--line:#e2e1da;--head:#f3f2ec;--side:#faf9f4;--bar:#2a7fbf}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#161615;--fg:#ecebe6;--muted:#9a998f;--line:#34332e;--head:#22211e;--side:#1c1b19;--bar:#5aa9e6}}}}
:root[data-theme="dark"]{{--bg:#161615;--fg:#ecebe6;--muted:#9a998f;--line:#34332e;--head:#22211e;--side:#1c1b19;--bar:#5aa9e6}}
body{{font:14px system-ui,sans-serif;max-width:1400px;margin:1.4rem auto;padding:0 1rem;background:var(--bg);color:var(--fg)}}
h1{{font-size:1.2rem}} h2{{font-size:1rem;margin-top:1.6rem}} .note{{color:var(--muted);font-size:.88rem;margin-bottom:1rem;max-width:980px}}
.wrap{{overflow-x:auto}} table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid var(--line);padding:8px 10px;vertical-align:top;text-align:left;font-size:13px}}
th{{background:var(--head);position:sticky;top:0}} td.p{{font-weight:600;width:150px;background:var(--side)}}
.bar{{height:4px;background:var(--bar);border-radius:2px;margin-bottom:4px}} .txt{{font-size:17px;line-height:1.35}}
.meta,.muted{{color:var(--muted);font-size:11px;margin-top:3px}} tr.grp td{{background:var(--head);font-weight:600;color:var(--muted)}}
tr.sum td{{font-weight:600;background:var(--side)}} table.curve{{width:auto}} table.curve td{{font-variant-numeric:tabular-nums}}
</style>
<h1>LoReFT emoji intervention vs system prompt vs additive vector — Qwen3.5-4B, layer {args.layer}</h1>
<div class=note>A rank-{args.rank} LoReFT (<code>Φ(h) = h + Rᵀ(Wh + b − Rh)</code>, {sum(p.numel() for p in params):,} trainable params,
base model frozen) was trained with pyreft at <b>{"the last prompt position" if args.positions == "last" else "every prompt position"}</b> of layer {args.layer}, on {len(train)} pairs of
(prompt → the model's own emoji answer under the emoji system prompt), {args.epochs} epochs, lr {args.lr:g}.
At test time there is <b>no system prompt</b>. All columns use greedy decoding with {args.new_tokens} new tokens.
<b>diversity</b> = unique emoji / total emoji (low = repetition). Peak training VRAM {peak:.2f} GB.</div>
<div class=wrap><table><tr><th>prompt</th>{head}</tr>{body}{sumrow}</table></div>
<h2>Training curve (mean next-token loss on response tokens)</h2>
<div class=note>held-out = the 5 held-out prompts' true system-prompt answers, never trained on.</div>
<table class=curve><tr><th>epoch</th><th>train</th><th>held-out</th></tr>{curve}</table>"""
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    open(out_html, "w").write(doc)
    print(f"Wrote {out_html}")

if __name__ == "__main__":
    main()
