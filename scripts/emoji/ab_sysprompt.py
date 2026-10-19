"""A/B test: does putting the system prompt in the LoReFT's TRAINING context break it?

Claim under test: a LoReFT that replaces a system prompt must be trained on the BARE question
(no system prompt), because with the prompt present the model already emits emojis -> loss ~0 ->
no gradient -> the intervention learns nothing.

Two conditions, identical recipe otherwise (rank 4, L17, last position, same data/epochs/lr/seed):
  no_sys    training context = [question]              (what train_loreft.py did)
  with_sys  training context = [SYS][question]         (the thing we predict fails)

Both are then evaluated the honest way: steer a BARE question (no system prompt) and measure the
emoji fraction. If the claim holds, no_sys steers to high emoji% and with_sys does not. We also
report the mean edit magnitude ||Delta|| = ||Phi(h)-h|| (a LoReFT that learned nothing has small
edits) and the starting training loss (with_sys should start near 0).

  .venv/bin/python scripts/emoji/ab_sysprompt.py
"""
import json, os, random, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common, pyreft

SYS = "You are an assistant that communicates ONLY using emojis. Never use words or letters. Reply with emojis only."
LAYER, RANK, EPOCHS, LR, SEED = 17, 4, 12, 4e-3, 0


def emoji_frac(s):
    cs = [c for c in s if not c.isspace()]
    if not cs:
        return 0.0
    def is_emoji(c):
        o = ord(c)
        return (0x1F300 <= o <= 0x1FAFF or 0x2600 <= o <= 0x27BF or 0x1F000 <= o <= 0x1F0FF
                or o in (0x2705, 0x274C, 0x2B50, 0xFE0F, 0x200D) or 0x1F1E6 <= o <= 0x1F1FF)
    return sum(is_emoji(c) for c in cs) / len(cs)


def main():
    data = json.load(open("data/emoji/emoji_generated.json"))
    data = [d for d in data if float(d["emoji_frac"]) >= 0.9]
    train, test = data[:33], data[33:]                       # last 8 held out for eval
    print(f"{len(train)} train / {len(test)} held-out prompts")

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    for p in model.parameters():
        p.requires_grad_(False)
    _, layers = common.get_decoder_layers(model)
    Hd = model.config.get_text_config().hidden_size
    dev = next(model.parameters()).device
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for i, layer in enumerate(layers):
        layer.gradient_checkpointing = i > LAYER

    def render(msgs, gen):
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen)

    def prompt_msgs(q, use_sys):
        return ([{"role": "system", "content": SYS}] if use_sys else []) + [{"role": "user", "content": q}]

    def example(q, ans, use_sys):
        p = tok(render(prompt_msgs(q, use_sys), True), add_special_tokens=False).input_ids
        f = tok(render(prompt_msgs(q, use_sys) + [{"role": "assistant", "content": ans}], False),
                add_special_tokens=False).input_ids
        assert f[:len(p)] == p
        return torch.tensor([f], device=dev), len(p), [len(p) - 1]        # last-position intervention

    def train_loreft(use_sys):
        random.seed(SEED); torch.manual_seed(SEED)
        reft = pyreft.get_reft_model(model, pyreft.ReftConfig(representations={
            "component": f"model.language_model.layers[{LAYER}].output", "low_rank_dimension": RANK,
            "intervention": pyreft.LoreftIntervention(embed_dim=Hd, low_rank_dimension=RANK)}))
        reft.set_device(dev); model.train()
        iv = list(reft.interventions.values())[0]; iv = iv[0] if isinstance(iv, (list, tuple)) else iv
        params = [p for p in reft.parameters() if p.requires_grad]
        exs = [example(d["prompt"], d["answer"].rstrip("�"), use_sys) for d in train]

        def loss_of(ids, plen, pos):
            keep = ids.shape[1] - plen + 1
            _, o = reft(base={"input_ids": ids, "attention_mask": torch.ones_like(ids), "logits_to_keep": keep},
                        unit_locations={"sources->base": (None, [[pos]])})
            lg = o.logits[0, :-1].float(); tg = ids[0, plen:]
            return torch.nn.functional.cross_entropy(lg, tg)

        opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
        start = np.mean([float(loss_of(*e)) for e in exs[:8]])
        t0 = time.time()
        for ep in range(1, EPOCHS + 1):
            order = list(range(len(exs))); random.shuffle(order); tot = 0.0
            for k, j in enumerate(order):
                l = loss_of(*exs[j]); (l / 4).backward(); tot += float(l)
                if (k + 1) % 4 == 0 or k + 1 == len(order):
                    opt.step(); opt.zero_grad()
        model.eval()
        tag = "withsys" if use_sys else "nosys"
        ck = f"data/emoji/ab_loreft_{tag}_L{LAYER}.pt"
        torch.save({"layer": LAYER, "rank": RANK, "positions": "last", "trained_with_sysprompt": use_sys,
                    "state_dict": {k: v.detach().cpu() for k, v in iv.state_dict().items()},
                    "start_loss": float(start), "final_loss": float(tot/len(order))}, ck)
        print(f"  [{tag}] start-loss {start:.3f} -> final {tot/len(order):.3f} ({time.time()-t0:.0f}s) -> {ck}")
        return reft, iv

    # measure edit magnitude ||Phi(h)-h|| at the intervened position on the held-out prompts (no sys)
    def edit_norm(iv):
        R = iv.state_dict()["rotate_layer"].float().cpu().numpy().T
        W = iv.state_dict()["weight"].float().cpu().numpy(); b = iv.state_dict()["bias"].float().cpu().numpy()
        # Delta(h) = R^T (W h + b - R h); measure on real held-out last-token residuals
        grab = {}
        hk = layers[LAYER].register_forward_hook(lambda _m, _i, o: grab.__setitem__("h", common.layer_output_hidden(o)[0].float().cpu().numpy()))
        ns = []
        try:
            for d in test:
                pid = tok(render(prompt_msgs(d["prompt"], False), True), add_special_tokens=False).input_ids
                with torch.no_grad():
                    model(input_ids=torch.tensor([pid], device=dev))
                h = grab["h"][-1]
                ns.append(np.linalg.norm(R.T @ (W @ h + b - R @ h)))
        finally:
            hk.remove()
        return float(np.mean(ns))

    def steer_gen(reft):
        out = []
        for d in test:
            pid = torch.tensor([tok(render(prompt_msgs(d["prompt"], False), True), add_special_tokens=False).input_ids], device=dev)
            with torch.inference_mode():
                _, g = reft.generate({"input_ids": pid, "attention_mask": torch.ones_like(pid)},
                                     unit_locations={"sources->base": (None, [[[pid.shape[1] - 1]]])},
                                     intervene_on_prompt=True, max_new_tokens=48, do_sample=False,
                                     eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
            t = tok.decode(g[0, pid.shape[1]:], skip_special_tokens=True).strip()
            out.append({"q": d["prompt"], "text": t, "frac": emoji_frac(t)})
        return out

    def plain_gen(use_sys):
        out = []
        for d in test:
            pid = torch.tensor([tok(render(prompt_msgs(d["prompt"], use_sys), True), add_special_tokens=False).input_ids], device=dev)
            with torch.inference_mode():
                o = model.generate(input_ids=pid, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
            t = tok.decode(o[0, pid.shape[1]:], skip_special_tokens=True).strip()
            out.append({"q": d["prompt"], "text": t, "frac": emoji_frac(t)})
        return out

    print("\n=== baselines (held-out, no intervention) ===")
    g_plain = plain_gen(False); g_sys = plain_gen(True)
    print(f"  plain {100*np.mean([x['frac'] for x in g_plain]):.0f}% | true sys prompt {100*np.mean([x['frac'] for x in g_sys]):.0f}%")

    print("\n=== training two LoReFTs ===")
    reft_a, iv_a = train_loreft(False)
    en_a = edit_norm(iv_a); g_a = steer_gen(reft_a); del reft_a
    reft_b, iv_b = train_loreft(True)
    en_b = edit_norm(iv_b); g_b = steer_gen(reft_b); del reft_b

    cols = [("plain", "plain (bare question, no intervention)", g_plain, None),
            ("truesys", "true system prompt (no LoReFT)", g_sys, None),
            ("no_sys", "LoReFT trained WITHOUT sys prompt", g_a, en_a),
            ("with_sys", "LoReFT trained WITH sys prompt", g_b, en_b)]
    meanf = {k: float(np.mean([x["frac"] for x in g])) for k, _, g, _ in cols}
    print("\n=== RESULT (steer a BARE question) ===")
    for k, name, g, en in cols:
        print(f"  {name:<40} emoji {100*meanf[k]:>3.0f}%" + (f"  ||Delta|| {en:.2f}" if en else ""))
    json.dump({"means": meanf, "edit_norm": {"no_sys": en_a, "with_sys": en_b},
               "cols": [{"key": k, "name": n, "gen": g} for k, n, g, _ in cols]},
              open("data/emoji/ab_sysprompt.json", "w"), ensure_ascii=False, indent=1)
    render_html(cols, meanf, {"no_sys": en_a, "with_sys": en_b})
    print("saved data/emoji/ab_sysprompt.json + .html")


def render_html(cols, meanf, en):
    import html
    E = html.escape
    css = """
:root{--bg:#fbfaf7;--fg:#171613;--muted:#76746c;--line:#e3e1d8;--head:#f1efe7;--card:#fff;--good:#1a7f45;--bad:#b3261e}
@media(prefers-color-scheme:dark){:root{--bg:#141412;--fg:#ecebe5;--muted:#9b998f;--line:#312f2a;--head:#1e1d1a;--card:#1a1917;--good:#5ec687;--bad:#ef8e86}}
body{font:14px/1.55 system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--fg);max-width:1200px;margin:0 auto;padding:1.4rem 1.1rem 4rem}
h1{font-size:1.3rem;margin:0 0 .3rem} p.note{color:var(--muted);max-width:80ch;font-size:.9rem}
.sum{display:flex;gap:1rem;flex-wrap:wrap;margin:1rem 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:.7rem 1rem;min-width:170px}
.card .n{font-size:.78rem;color:var(--muted)} .card .v{font-size:1.5rem;font-weight:650}
.win{color:var(--good)} .lose{color:var(--bad)}
table{border-collapse:collapse;width:100%;margin:1rem 0;table-layout:fixed}
th,td{border:1px solid var(--line);padding:.5rem .6rem;vertical-align:top;text-align:left}
th{background:var(--head);color:var(--muted);font-weight:600;font-size:.85rem}
td.q{font-size:.85rem;color:var(--muted);width:15%}
td.g{font-family:ui-monospace,Menlo,monospace;font-size:15px;word-break:break-word}
.f{font-size:.72rem;color:var(--muted);display:block;margin-top:.3rem}
"""
    order = ["q"] + [c[0] for c in cols]
    head = "<tr><th>held-out question</th>" + "".join(
        f"<th>{E(n)}<br><span class=f>mean emoji {100*meanf[k]:.0f}%"
        + (f" · ‖Δ‖ {en[k]:.1f}" if k in en else "") + "</span></th>" for k, n, _, _ in cols) + "</tr>"
    rows = ""
    n = len(cols[0][2])
    for i in range(n):
        q = cols[0][2][i]["q"]
        cells = ""
        for k, _, g, _ in cols:
            x = g[i]; cls = "win" if x["frac"] >= 0.5 else ("lose" if x["frac"] < 0.1 else "")
            cells += (f'<td class=g>{E(x["text"]) or "<i>(empty)</i>"}'
                      f'<span class="f {cls}">emoji {100*x["frac"]:.0f}%</span></td>')
        rows += f'<tr><td class=q>{E(q)}</td>{cells}</tr>'
    body = (f"<h1>LoReFT with vs without the system prompt in training</h1>"
            f"<p class=note>All four columns generate on the SAME held-out questions with NO system "
            f"prompt at inference. The two LoReFTs are identical (rank 4, L17, last position, same "
            f"data/epochs/seed) except one saw the system prompt in its training context and one did "
            f"not. A LoReFT meant to replace the prompt must be trained WITHOUT it — with the prompt "
            f"present the model already complies, so there is no gradient and the intervention learns "
            f"almost nothing (tiny ‖Δ‖).</p>"
            f'<div class=sum>'
            + "".join(f'<div class=card><div class=n>{E(n)}</div>'
                      f'<div class="v {"win" if meanf[k]>=.5 else "lose" if meanf[k]<.1 else ""}">{100*meanf[k]:.0f}%</div></div>'
                      for k, n, _, _ in cols)
            + "</div>"
            f"<div style='overflow-x:auto'><table>{head}{rows}</table></div>")
    open("data/emoji/ab_sysprompt.html", "w").write(
        f"<title>LoReFT: with vs without system prompt</title>\n<style>{css}</style>\n{body}")


if __name__ == "__main__":
    main()
