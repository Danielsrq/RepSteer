"""Head-to-head: every emoji-steering attempt vs the trained LoReFT (loaded from its checkpoint), on the
same prompts, same layer, same token budget. No system prompt at test time except the reference column.

Static vectors get a small alpha sweep and are reported at their BEST alpha (by mean emoji% x diversity),
tuned on these same prompts -- this favours the static baselines. LoReFT gets no tuning.

Metrics per output:  emoji%  (share of non-space chars that are emoji)
                     diversity  (unique emoji / total emoji; repetition -> low)
                     overlap  (Jaccard of emoji sets vs the true system-prompt answer to the same prompt)
                     score = emoji% x diversity

Several LoReFT checkpoints can be compared (--loreft a.pt b.pt); each is applied at the positions it was
trained with (checkpoint key "positions": "last" or "all"; old checkpoints without the key = "last").

Usage:  .venv/bin/python scripts/emoji/compare_all.py      (writes data/emoji/compare_all_positions.{json,html})
"""
import argparse, html, json, os, sys, time
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
import common
import pyreft
from sysprompt_vec import SYS, BUILD_PROMPTS
from train_loreft import HELD, FRESH, ef, diversity, glyphs, positions_for

SCHEME_NAME = {"last": "last prompt token", "all": "all prompt tokens"}

SAMPLE_KW = dict(do_sample=True, temperature=0.9, top_p=0.95, repetition_penalty=1.3, no_repeat_ngram_size=3)

def overlap(t, ref):
    a, b = set(glyphs(t)), set(glyphs(ref))
    return len(a & b) / len(a | b) if (a | b) else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--new-tokens", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loreft", nargs="+", default=["data/emoji/loreft_L17.pt", "data/emoji/loreft_L17_allprompt.pt"],
                    help="checkpoints; each one's training report (same path, .html) is used for the load check")
    ap.add_argument("--out", default="data/emoji/compare_all_positions")
    ap.add_argument("--force", action="store_true", help="allow overwriting existing outputs")
    args = ap.parse_args()
    L = args.layer
    for f in (args.out + ".html", args.out + ".json"):
        if os.path.exists(f) and not args.force:
            sys.exit(f"refusing to overwrite existing {f} (pass --force or a different --out)")

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    for p in model.parameters():
        p.requires_grad_(False)
    _, layers = common.get_decoder_layers(model)
    Hd = model.config.get_text_config().hidden_size
    dev = next(model.parameters()).device; dt = next(model.parameters()).dtype

    def render(msgs, gen):
        try: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen, enable_thinking=False)
        except TypeError: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen)
    def pids(prompt, sys_prompt=False):
        msgs = ([{"role": "system", "content": SYS}] if sys_prompt else []) + [{"role": "user", "content": prompt}]
        return tok(render(msgs, True), add_special_tokens=False).input_ids

    def gen(ids_list, hook=None, **gk):
        enc = torch.tensor([ids_list], device=dev)
        hd = layers[L].register_forward_hook(hook) if hook else None
        try:
            with torch.inference_mode():
                o = model.generate(input_ids=enc, max_new_tokens=args.new_tokens, pad_token_id=tok.eos_token_id,
                                   **({"do_sample": False} | gk))
        finally:
            if hd: hd.remove()
        return tok.decode(o[0, enc.shape[1]:], skip_special_tokens=True).strip()

    def make_hook(vec, mode, scope, a):
        v = vec.to(dev, dt)
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            if scope == "all" or h.shape[1] == 1:
                h = h + (a * h.norm(dim=-1, keepdim=True) * v if mode == "normrel" else a * v)
            return (h, *out[1:]) if isinstance(out, tuple) else h
        return hook

    # ---------------- the static vectors, all at layer L ----------------
    byte_dim = torch.load("data/emoji/emoji_dim.pt", weights_only=False)["vhat"][L].float()
    v_off = torch.load("data/emoji/sysprompt_vec.pt", weights_only=False)["v"][L].float()
    v_resp = torch.load("data/emoji/response_dim.pt", weights_only=False)["v"][L].float()

    grab = {}
    hk = layers[L].register_forward_hook(lambda _m, _i, o: grab.__setitem__("h", common.layer_output_hidden(o)[0].float().cpu()))
    sys_block = tok("<|im_start|>system\n" + SYS + "<|im_end|>\n", add_special_tokens=False).input_ids
    full = pids(BUILD_PROMPTS[0], sys_prompt=True)
    assert full[:len(sys_block)] == sys_block, "system block prefix mismatch"
    with torch.no_grad():
        model(input_ids=torch.tensor([full], device=dev))
        h_sys = grab["h"][len(sys_block) - 1].clone()                 # final system-prompt token
        base = []
        for p in BUILD_PROMPTS:
            model(input_ids=torch.tensor([pids(p)], device=dev)); base.append(grab["h"][-1].clone())
    hk.remove()
    baseline = torch.stack(base).mean(0)
    unit = lambda x: x / x.norm()
    n_off = v_off.norm()

    STATIC = [
        dict(key="byte_dim", name="#1 byte-DiM", built="mean activations over hand-written emoji responses − "
             "hand-written normal responses (30 pairs)", inject="h += α·‖h‖·v̂, generated tokens only",
             vec=byte_dim, mode="normrel", scope="decode", alphas=[0.3, 0.5, 0.7]),
        dict(key="offset", name="#2 conditioning offset", built="residual with emoji system prompt − without, at the "
             "generation position, mean over 24 prompts", inject="h += α·v, all positions",
             vec=v_off, mode="raw", scope="all", alphas=[0.75, 1.0, 1.25, 1.5]),
        dict(key="offset_rp", name="#2 + sampling & repetition penalty", built="same vector as #2",
             inject="h += α·v, all positions; sampled (T=0.9, top-p 0.95), repetition penalty 1.3",
             vec=v_off, mode="raw", scope="all", alphas=[1.0, 1.25], sample=True),
        dict(key="resp_dim", name="#3 generated-response DiM", built="mean activations over the model's own emoji "
             "answers − its normal answers (36 prompts)", inject="h += α·v, all positions",
             vec=v_resp, mode="raw", scope="all", alphas=[0.75, 1.0, 1.5]),
        dict(key="sys_tok", name="system-token activation (N=1)", built="raw residual at the final system-prompt "
             "token, rescaled to ‖#2‖", inject="h += α·‖v₂‖·û, all positions",
             vec=unit(h_sys) * n_off, mode="raw", scope="all", alphas=[0.75, 1.0, 1.5]),
        dict(key="sys_tok_c", name="system-token, mean-centred", built="final system-prompt token residual − mean "
             "no-system generation-position residual, rescaled to ‖#2‖", inject="h += α·‖v₂‖·û, all positions",
             vec=unit(h_sys - baseline) * n_off, mode="raw", scope="all", alphas=[0.75, 1.0, 1.5]),
    ]

    prompts = [("held-out", p) for p in HELD] + [("fresh", p) for p in FRESH]
    t0 = time.time()
    ref = {p: gen(pids(p, sys_prompt=True)) for _, p in prompts}
    plain = {p: gen(pids(p)) for _, p in prompts}
    print(f"references done ({time.time()-t0:.0f}s)")

    def score_table(outs):
        per = {p: dict(emoji=ef(outs[p]), div=diversity(outs[p]), overlap=overlap(outs[p], ref[p]),
                       score=ef(outs[p]) * diversity(outs[p])) for _, p in prompts}
        mean = {k: sum(v[k] for v in per.values()) / len(per) for k in ("emoji", "div", "overlap", "score")}
        return per, mean

    sweep = {}
    for m in STATIC:
        sweep[m["key"]] = {}
        for a in m["alphas"]:
            outs = {}
            for _, p in prompts:
                if m.get("sample"): torch.manual_seed(args.seed)
                outs[p] = gen(pids(p), make_hook(m["vec"], m["mode"], m["scope"], a), **(SAMPLE_KW if m.get("sample") else {}))
            sweep[m["key"]][a] = {"outs": outs, "mean": score_table(outs)[1]}
            mm = sweep[m["key"]][a]["mean"]
            print(f"  {m['name'][:34]:34} α={a:<5g} emoji {100*mm['emoji']:3.0f}%  div {mm['div']:.2f}  "
                  f"overlap {mm['overlap']:.2f}  score {mm['score']:.2f}  ({time.time()-t0:.0f}s)")

    # ---------------- edit statistics: Δ = Φ(h) − h at the positions each checkpoint edits ----------------
    # Layer-L outputs at prompt positions are unaffected by edits made at layer L itself, so a plain forward
    # gives exactly the h the intervention sees.
    resid = {}
    hk = layers[L].register_forward_hook(lambda _m, _i, o: grab.__setitem__("H", common.layer_output_hidden(o)[0].float().cpu()))
    with torch.no_grad():
        for _, p in prompts:
            model(input_ids=torch.tensor([pids(p)], device=dev)); resid[p] = grab["H"].clone()
    hk.remove()
    def edit_stats(ck, scheme):
        sd = ck["state_dict"]
        R = sd["rotate_layer"].float().cpu().T; W = sd["weight"].float().cpu(); b = sd["bias"].float().cpu()
        per_norm, last_norm, total, hnorm, last_edits, within = [], [], [], [], [], []
        for _, p in prompts:
            H = resid[p]; pos = positions_for(H.shape[0], scheme); Hs = H[pos]
            D = ((Hs @ W.T + b) - Hs @ R.T) @ R                         # [k, d]: Rᵀ(Wh + b − Rh) per position
            n = D.norm(dim=1)
            per_norm += n.tolist(); last_norm.append(float(n[-1])); total.append(float(n.sum()))
            hnorm += Hs.norm(dim=1).tolist(); last_edits.append(D[-1])
            if len(pos) > 1:
                Dn = D / D.norm(dim=1, keepdim=True); C = Dn @ Dn.T; k = len(pos)
                within.append(float((C.sum() - k) / (k * k - k)))
        E = torch.stack(last_edits); E = E / E.norm(dim=1, keepdim=True); C = E @ E.T; n = len(E)
        return {"positions_per_prompt": sum(len(positions_for(resid[p].shape[0], scheme)) for _, p in prompts) / len(prompts),
                "mean_edit": sum(per_norm) / len(per_norm), "max_edit": max(per_norm),
                "last_edit": sum(last_norm) / len(last_norm), "total_edit": sum(total) / len(total),
                "mean_h": sum(hnorm) / len(hnorm), "cos_across_prompts": float((C.sum() - n) / (n * n - n)),
                "cos_within_prompt": (sum(within) / len(within)) if within else None}

    # ---------------- LoReFT checkpoints, each applied at the positions it was trained on ----------------
    lorefts, reft, iv = [], None, None
    for path in args.loreft:
        ck = torch.load(path, weights_only=False)
        scheme = ck.get("positions", "last")
        assert ck["layer"] == L, f"{path} is for layer {ck['layer']}, not {L}"
        if reft is None:
            reft = pyreft.get_reft_model(model, pyreft.ReftConfig(representations={
                "component": f"model.language_model.layers[{L}].output", "low_rank_dimension": ck["rank"],
                "intervention": pyreft.LoreftIntervention(embed_dim=Hd, low_rank_dimension=ck["rank"])}))
            reft.set_device(dev)
            iv = list(reft.interventions.values())[0]; iv = iv[0] if isinstance(iv, (list, tuple)) else iv
            rank0 = ck["rank"]
        assert ck["rank"] == rank0, "all checkpoints must share a rank to reuse one wrapper"
        iv.load_state_dict(ck["state_dict"])                           # pyreft rebuilds + checks orthonormal R
        def loreft_gen(prompt):
            pi = torch.tensor([pids(prompt)], device=dev)
            with torch.inference_mode():
                _, g = reft.generate({"input_ids": pi, "attention_mask": torch.ones_like(pi)},
                                     unit_locations={"sources->base": (None, [[positions_for(pi.shape[1], scheme)]])},
                                     intervene_on_prompt=True, max_new_tokens=args.new_tokens, do_sample=False,
                                     eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
            return tok.decode(g[0, pi.shape[1]:], skip_special_tokens=True).strip()
        outs = {p: loreft_gen(p) for _, p in prompts}
        rep = path[:-3] + ".html"
        train_html = open(rep).read() if os.path.exists(rep) else ""
        match = sum(html.escape(outs[p]) in train_html for _, p in prompts)
        stats = edit_stats(ck, scheme)
        lorefts.append(dict(path=path, ck=ck, scheme=scheme, outs=outs, match=match, stats=stats))
        print(f"LoReFT {path} [{scheme}]: {match}/{len(prompts)} outputs identical to its training run | "
              f"edit/pos {stats['mean_edit']:.1f}, last-pos {stats['last_edit']:.1f}, total {stats['total_edit']:.1f}, "
              f"cos across prompts {stats['cos_across_prompts']:.3f}")

    # ---------------- assemble ----------------
    methods = [dict(key="plain", name="plain model", built="no steering, no system prompt", inject="—", setting="—", outs=plain, kind="ref"),
               dict(key="true", name="true system prompt", built="the emoji system prompt itself (the target behaviour)",
                    inject="system prompt in context", setting="—", outs=ref, kind="ref")]
    for m in STATIC:
        best = max(m["alphas"], key=lambda a: sweep[m["key"]][a]["mean"]["score"])
        methods.append(dict(key=m["key"], name=m["name"], built=m["built"], inject=m["inject"],
                            setting=f"α = {best:g} (best of {', '.join(f'{a:g}' for a in m['alphas'])})",
                            outs=sweep[m["key"]][best]["outs"], kind="static"))
    for lf in lorefts:
        ck = lf["ck"]
        methods.append(dict(key=f"loreft_{lf['scheme']}", name=f"LoReFT — {SCHEME_NAME[lf['scheme']]}",
                            built=f"trained low-rank edit Φ(h) = h + Rᵀ(Wh + b − Rh), rank {ck['rank']}, "
                                  f"{len(ck['train_prompts'])} pairs, {ck['args']['epochs']} epochs; {os.path.basename(lf['path'])}",
                            inject=f"{SCHEME_NAME[lf['scheme']]}, prompt pass only (trained the same way)",
                            setting="no tuning", outs=lf["outs"], kind="loreft"))
    for m in methods:
        m["per"], m["mean"] = score_table(m["outs"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"layer": L, "new_tokens": args.new_tokens, "prompts": prompts,
               "lorefts": [{"path": lf["path"], "scheme": lf["scheme"], "load_match": lf["match"], "stats": lf["stats"],
                            "log": lf["ck"]["log"]} for lf in lorefts],
               "methods": [{k: v for k, v in m.items()} for m in methods],
               "sweep": {k: {str(a): v for a, v in d.items()} for k, d in sweep.items()}},
              open(args.out + ".json", "w"), ensure_ascii=False, indent=1)
    print("\nmean emoji% / diversity / overlap / score")
    for m in methods:
        mm = m["mean"]; print(f"  {m['name'][:36]:36} {100*mm['emoji']:3.0f}%  {mm['div']:.2f}  {mm['overlap']:.2f}  {mm['score']:.2f}")

    render_html(args, methods, sweep, STATIC, prompts, lorefts)
    print(f"Wrote {args.out}.html / .json  ({time.time()-t0:.0f}s)")


def render_html(args, methods, sweep, STATIC, prompts, lorefts):
    E = html.escape
    best_score = max(m["mean"]["score"] for m in methods if m["kind"] != "ref")
    def bar(x, cls=""):
        return f'<div class="bar {cls}"><span style="width:{100*x:.0f}%"></span></div>'
    rows = ""
    for m in methods:
        mm = m["mean"]
        ov = "—" if m["key"] == "true" else f"{mm['overlap']:.2f}"
        win = " win" if m["kind"] != "ref" and mm["score"] == best_score else ""
        rows += (f'<tr class="{m["kind"]}{win}"><td class=mname>{E(m["name"])}</td><td class=built>{E(m["built"])}</td>'
                 f'<td class=built>{E(m["inject"])}<div class=muted>{E(m["setting"])}</div></td>'
                 f'<td class=num>{100*mm["emoji"]:.0f}%{bar(mm["emoji"])}</td><td class=num>{mm["div"]:.2f}{bar(mm["div"])}</td>'
                 f'<td class=num>{ov}{"" if ov == "—" else bar(mm["overlap"])}</td><td class=num><b>{mm["score"]:.2f}</b>{bar(mm["score"], "s")}</td></tr>')

    cards, prev = "", None
    for group, p in prompts:
        if group != prev:
            cards += f'<h3 class=grp>{"Held-out prompts" if group == "held-out" else "Fresh prompts"}<span class=muted> — {"not trained on; true answers exist in the dataset" if group == "held-out" else "appear in no file at all"}</span></h3>'
            prev = group
        trs = ""
        for m in methods:
            pr = m["per"][p]; out = m["outs"][p]
            ov = "—" if m["key"] == "true" else f"{pr['overlap']:.2f}"
            trs += (f'<tr class="{m["kind"]}"><td class=mname>{E(m["name"])}</td>'
                    f'<td class=out>{E(out) or "<i class=muted>(empty)</i>"}</td>'
                    f'<td class=num>{100*pr["emoji"]:.0f}%</td><td class=num>{pr["div"]:.2f}</td><td class=num>{ov}</td></tr>')
        cards += (f'<section class=card><div class=ptitle>{E(p)}</div><div class=wrap><table class=pt>'
                  f'<tr><th>method</th><th>output</th><th>emoji</th><th>diversity</th><th>overlap</th></tr>{trs}</table></div></section>')

    sw = ""
    for m in STATIC:
        best = max(m["alphas"], key=lambda a: sweep[m["key"]][a]["mean"]["score"])
        cells = "".join(
            f'<td class="num{" best" if a == best else ""}">α={a:g}<div>{100*sweep[m["key"]][a]["mean"]["emoji"]:.0f}% · '
            f'{sweep[m["key"]][a]["mean"]["div"]:.2f} · <b>{sweep[m["key"]][a]["mean"]["score"]:.2f}</b></div></td>'
            for a in m["alphas"])
        sw += f'<tr><td class=mname>{E(m["name"])}</td>{cells}</tr>'

    # LoReFT: last-token vs all-prompt-token checkpoints
    lhead = "".join(f"<th>{E(SCHEME_NAME[lf['scheme']])}</th>" for lf in lorefts)
    def srow(label, fn, note=""):
        return f'<tr><td class=mname>{label}<div class=muted>{note}</div></td>' + "".join(f"<td class=num>{fn(lf)}</td>" for lf in lorefts) + "</tr>"
    fmt = lambda x: "—" if x is None else f"{x:.3f}"
    stat_rows = (
        srow("positions edited per prompt", lambda lf: f"{lf['stats']['positions_per_prompt']:.1f}")
        + srow("mean ‖h‖ at edited positions", lambda lf: f"{lf['stats']['mean_h']:.1f}", "size of the residual being edited")
        + srow("mean ‖edit‖ per position", lambda lf: f"{lf['stats']['mean_edit']:.1f}", "‖Φ(h) − h‖")
        + srow("max ‖edit‖ at any position", lambda lf: f"{lf['stats']['max_edit']:.1f}")
        + srow("‖edit‖ at the last prompt token", lambda lf: f"{lf['stats']['last_edit']:.1f}")
        + srow("total ‖edit‖ per prompt", lambda lf: f"{lf['stats']['total_edit']:.1f}", "sum over edited positions")
        + srow("cos of last-token edits across prompts", lambda lf: fmt(lf['stats']['cos_across_prompts']), "1 = same edit for every prompt")
        + srow("cos of edits across positions (same prompt)", lambda lf: fmt(lf['stats']['cos_within_prompt']), "1 = same edit at every position")
        + srow("reloaded outputs = training run", lambda lf: f"{lf['match']}/{len(prompts)}")
        + srow("scoreboard: emoji % · diversity · overlap", lambda lf: next(
            f"{100*m['mean']['emoji']:.0f}% · {m['mean']['div']:.2f} · {m['mean']['overlap']:.2f}"
            for m in methods if m["key"] == f"loreft_{lf['scheme']}")))
    n_ep = max(len(lf["ck"]["log"]) for lf in lorefts)
    ch = "".join(f"<th>{E(SCHEME_NAME[lf['scheme']])}: train</th><th>held-out</th>" for lf in lorefts)
    curve = ""
    for i in range(n_ep):
        cells = ""
        for lf in lorefts:
            r = lf["ck"]["log"][i] if i < len(lf["ck"]["log"]) else {}
            cells += f"<td class=num>{fmt(r.get('train'))}</td><td class=num>{fmt(r.get('held'))}</td>"
        curve += f"<tr><td class=num>{i}</td>{cells}</tr>"
    loaded = "; ".join(f"<code>{E(os.path.basename(lf['path']))}</code> ({lf['match']}/{len(prompts)} identical to its training run)" for lf in lorefts)

    doc = f"""<title>LoReFT position showdown</title>
<style>
:root{{--bg:#fbfaf7;--fg:#171613;--muted:#76746c;--line:#e3e1d8;--head:#f1efe7;--card:#ffffff;--ref:#f6f4ec;
--acc:#1f6fb2;--win:#e7f2e2;--winline:#4c9a3a;--bar:#d9d6cb;--barfill:#8a877c;--sfill:#1f6fb2}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#141412;--fg:#ecebe5;--muted:#9b998f;--line:#312f2a;
--head:#1e1d1a;--card:#1a1917;--ref:#1f1e1b;--acc:#62a8e6;--win:#1d2a1a;--winline:#7cc06a;--bar:#34322d;--barfill:#8f8c82;--sfill:#62a8e6}}}}
:root[data-theme="dark"]{{--bg:#141412;--fg:#ecebe5;--muted:#9b998f;--line:#312f2a;--head:#1e1d1a;--card:#1a1917;--ref:#1f1e1b;
--acc:#62a8e6;--win:#1d2a1a;--winline:#7cc06a;--bar:#34322d;--barfill:#8f8c82;--sfill:#62a8e6}}
body{{font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--fg);max-width:1280px;margin:0 auto;padding:1.6rem 1.1rem 3rem}}
h1{{font-size:1.45rem;margin:0 0 .3rem}} h2{{font-size:1.08rem;margin:2.2rem 0 .5rem}} h3.grp{{font-size:.98rem;margin:1.6rem 0 .6rem}}
.lede{{color:var(--muted);max-width:900px;margin:.2rem 0 1.2rem}} .muted{{color:var(--muted);font-weight:400;font-size:.82rem}}
.wrap{{overflow-x:auto}} table{{border-collapse:collapse;width:100%}}
th,td{{border-bottom:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}}
th{{background:var(--head);font-weight:600;font-size:.82rem;color:var(--muted);position:sticky;top:0}}
td.mname{{font-weight:600;white-space:nowrap}} td.built{{color:var(--muted);font-size:.84rem;max-width:300px}}
td.num{{font-variant-numeric:tabular-nums;white-space:nowrap;min-width:78px}}
tr.ref td{{background:var(--ref)}} tr.win td{{background:var(--win)}} tr.win td.mname{{box-shadow:inset 3px 0 0 var(--winline)}}
tr.loreft td.mname{{color:var(--acc)}}
.bar{{height:4px;background:var(--bar);border-radius:2px;margin-top:4px;width:72px}} .bar span{{display:block;height:100%;background:var(--barfill);border-radius:2px}}
.bar.s span{{background:var(--sfill)}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:.7rem .9rem .4rem;margin:0 0 1rem}}
.ptitle{{font-weight:650;margin-bottom:.3rem}} table.pt td.out{{font-family:"Apple Color Emoji","Segoe UI Emoji","Noto Color Emoji",system-ui,sans-serif;
font-size:17px;white-space:pre-wrap;word-break:break-word}} table.pt td{{padding:6px 10px}}
td.best{{background:var(--win)}} ul.notes{{color:var(--muted);max-width:920px;padding-left:1.1rem}} ul.notes li{{margin:.3rem 0}}
code{{font-size:.9em}}
</style>
<h1>Emoji steering showdown — every attempt vs two LoReFTs</h1>
<div class=lede>Qwen3.5-4B, all interventions at layer {args.layer}. Same {len(prompts)} prompts ({len(HELD)} held-out + {len(FRESH)} fresh),
no system prompt at test time except the <b>true system prompt</b> reference, {args.new_tokens} new tokens, greedy unless noted.
Static vectors are shown at their <b>best α</b>, tuned on these same prompts (favours them). The two LoReFTs are identical except
for which prompt positions they edit (in training and at generation), are untuned, and were loaded from {loaded}.</div>

<h2>Scoreboard (mean over {len(prompts)} prompts)</h2>
<div class=wrap><table>
<tr><th>method</th><th>how it's built</th><th>how it's applied</th><th>emoji %</th><th>diversity</th><th>overlap w/ true</th><th>score</th></tr>
{rows}</table></div>
<ul class=notes>
<li><b>emoji %</b>: share of non-space characters that are emoji. <b>diversity</b>: unique emoji / total emoji (👍👍👍… → ~0).
<b>overlap</b>: Jaccard similarity between the method's emoji set and the true system-prompt answer's emoji set for the same prompt —
a rough proxy for "picked on-topic emoji". <b>score</b> = emoji % × diversity, used to choose each static vector's α.</li>
<li>Shaded grey rows are references; the green row has the highest score among the steering methods.</li>
</ul>

<h2>LoReFT: last prompt token vs all prompt tokens</h2>
<div class=lede>Same rank, layer, data, epochs, learning rate and seed; only the edited positions differ. Edit sizes are measured on
the {len(prompts)} evaluation prompts from each checkpoint's R, W, b.</div>
<div class=wrap><table style="width:auto"><tr><th></th>{lhead}</tr>{stat_rows}</table></div>
<h3 class=grp>Training curves <span class=muted>(mean next-token loss on answer tokens; held-out = the 5 held-out prompts' true answers)</span></h3>
<div class=wrap><table style="width:auto"><tr><th>epoch</th>{ch}</tr>{curve}</table></div>

<h2>Per-prompt outputs</h2>
{cards}

<h2>α sweep for the static vectors <span class=muted>(emoji % · diversity · <b>score</b>; green = chosen)</span></h2>
<div class=wrap><table>{sw}</table></div>

<h2>Notes &amp; caveats</h2>
<ul class=notes>
<li>All static vectors are a single fixed direction added to the residual stream. LoReFT is a trained edit of a
{lorefts[0]['ck']['rank']}-dimensional subspace, Φ(h) = h + Rᵀ(Wh + b − Rh), applied only while the prompt is processed;
its effect on generated tokens flows through the model's own layers. It can depend on h, but see the edit statistics for how much it actually does.</li>
<li>The system-token rows rescale their direction to the norm of the #2 offset so magnitudes are comparable; their raw activation
is ~60% unrelated to the emoji offset (cos 0.41) and tends to inject <code>&lt;think&gt;</code> scaffolding instead.</li>
<li>Metrics are crude: overlap penalises valid but different emoji choices, and diversity rewards variety even when it drifts off-topic.
Read the per-prompt outputs, not just the numbers.</li>
<li>Held-out prompts' true answers exist in <code>emoji_generated.json</code> but were never trained on; fresh prompts appear nowhere.
The static α choice was tuned on all {len(prompts)} prompts.</li>
</ul>"""
    open(args.out + ".html", "w").write(doc)


if __name__ == "__main__":
    main()
