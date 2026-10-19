"""Train a rank-r LoReFT for one concept (default: happy) on the concept corpus, then use it as a READER.

Training pairs come from data/concepts/paragraphs_concepts.json (12 concepts x the same 30 topics):
    prompt = the tone-free instruction the corpus was generated from ("Write a short, vivid paragraph ... about {topic}")
    target = the --concept paragraph for that topic (e.g. the HAPPY one)
so the intervention learns "plain prompt -> <concept> text". 6 topics are held out (never trained on).
--contrast (default: sad for happy, happy otherwise) is the second negative class in the probes and display.
Two position schemes are trained from the same init: every prompt token ("all") and the last prompt token ("last").
The epoch with the lowest held-out loss is kept (early stopping).

Reading (no steering): V = W - R,  V = U S Q^T (thin SVD),  M = R^T V = (R^T U) S Q^T
    read directions  q_i = rows of Q^T   (what gates the push)
    write directions p_i = cols of R^T U (where it writes), gains S
Per token of a paragraph (plain model, layer L, raw text as in 09/10): Q^T h (4 reads), (R^T U)^T h (4 write
coordinates), push = ||M h + c||. Compared against the concept's DiM projection and a random 4-dim subspace, with
Fisher-LDA probes trained on training-topic tokens and tested on held-out-topic tokens.

Usage:  .venv/bin/python scripts/reft/concept_read.py [--concept sad]
        (writes data/reft/<concept>_loreft_L17_{all,last}.pt and data/reft/<concept>_read_L17.{html,json})
"""
import argparse, html, json, os, random, sys, time
import numpy as np, torch
import torch.utils.checkpoint
import pyreft
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import common

PROMPT = ("Write a short, vivid paragraph (4 to 6 sentences) about {topic}. "
          "Write only the paragraph itself, with no title, preamble, or commentary.")
TGT, CON = "happy", "sad"            # target concept and contrast concept (set from args in main)
SHOW = [TGT, "neutral", CON]
SCHEMES = ["all", "last"]


def positions_for(n, scheme):
    return [n - 1] if scheme == "last" else list(range(n))

def auc(pos, neg):
    """P(score_pos > score_neg) via ranks (continuous scores, ties negligible)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    r = np.concatenate([pos, neg]).argsort().argsort() + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

def chunked_ce(logits, targets, chunk=64):
    """Mean cross-entropy, upcasting to float32 one chunk at a time (recomputed in backward) instead of all at once."""
    tot = torch.zeros((), device=logits.device, dtype=torch.float32)
    for s in range(0, len(targets), chunk):
        tot = tot + torch.utils.checkpoint.checkpoint(
            lambda x, y: torch.nn.functional.cross_entropy(x.float(), y, reduction="sum"),
            logits[s:s + chunk], targets[s:s + chunk], use_reentrant=False)
    return tot / len(targets)

def lda_fit(Xp, Xn, shrink=1e-2):
    """Fisher LDA direction (a covariance-whitened difference of means) with ridge shrinkage."""
    mp, mn = Xp.mean(0), Xn.mean(0)
    Xc = np.concatenate([Xp - mp, Xn - mn]); S = Xc.T @ Xc / len(Xc)
    S = S + shrink * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
    return np.linalg.solve(S, mp - mn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heldout", default="4,9,14,19,24,29", help="held-out topic indices (0-29)")
    ap.add_argument("--new-tokens", type=int, default=160)
    ap.add_argument("--out-dir", default="data/reft")
    ap.add_argument("--force", action="store_true", help="allow overwriting existing outputs")
    ap.add_argument("--concept", default="happy", help="target concept the LoReFT learns to write")
    ap.add_argument("--contrast", default=None, help="second negative class (default: sad for happy, else happy)")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="disable gradient checkpointing of the layers above the intervention (on by default: "
                         "same gradients, ~1.4 GiB less memory on 260-token examples)")
    args = ap.parse_args()
    global TGT, CON, SHOW
    TGT = args.concept
    CON = args.contrast or ("sad" if TGT == "happy" else "happy")
    SHOW = [TGT, "neutral", CON]
    L, od = args.layer, args.out_dir
    ck_path = {s: f"{od}/{TGT}_loreft_L{L}_{s}.pt" for s in SCHEMES}
    rep = f"{od}/{TGT}_read_L{L}"
    for f in [*ck_path.values(), rep + ".html", rep + ".json"]:
        if os.path.exists(f) and not args.force:
            sys.exit(f"refusing to overwrite existing {f} (pass --force or a different --out-dir)")
    os.makedirs(od, exist_ok=True)

    # ---------------- data ----------------
    rows = json.load(open("data/concepts/paragraphs_concepts.json"))["rows"]
    n_topics = 30
    topics = [r["topic"] for r in sorted((r for r in rows if r["concept"] == "happy"), key=lambda r: r["id"])]
    text = {(r["concept"], r["id"] % n_topics): r["text"] for r in rows}
    held_t = [int(x) for x in args.heldout.split(",")]
    train_t = [t for t in range(n_topics) if t not in held_t]

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ... | {TGT} LoReFT rank {args.rank} @ L{L} | train topics {len(train_t)}, held-out {held_t}")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    for p in model.parameters():
        p.requires_grad_(False)
    _, layers = common.get_decoder_layers(model)
    Hd = model.config.get_text_config().hidden_size
    dev = next(model.parameters()).device
    if not args.no_grad_ckpt:
        # recompute layers L+1.. in backward instead of storing their activations. The intervened layer and everything
        # below stay un-checkpointed, so pyvene's hook on layer L never re-fires during recomputation.
        # Only active in train mode (the model has no dropout, so train mode changes nothing else).
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        for i, layer in enumerate(layers):
            layer.gradient_checkpointing = i > L

    def render(msgs, gen):
        try: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen, enable_thinking=False)
        except TypeError: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen)
    def prompt_ids(t):
        return tok(render([{"role": "user", "content": PROMPT.format(topic=topics[t])}], True), add_special_tokens=False).input_ids
    def example(t):
        p = prompt_ids(t)
        f = tok(render([{"role": "user", "content": PROMPT.format(topic=topics[t])},
                        {"role": "assistant", "content": text[(TGT, t)]}], False), add_special_tokens=False).input_ids
        assert f[:len(p)] == p, "chat-template prefix mismatch"
        ids = torch.tensor([f], device=dev); labels = ids.clone(); labels[:, :len(p)] = -100
        return ids, labels, len(p)

    # ---------------- concept DiM recomputed on TRAINING topics only (stored vectors saw all 30) ----------------
    cvf = torch.load("data/concepts/concept_vectors.pt", weights_only=False, map_location="cpu")
    concepts, emo = cvf["concepts"], cvf["emotional_idx"]
    lab = cvf["labels"].numpy()
    assert (lab == np.array([r["concept_idx"] for r in rows])).all(), "feature/row order mismatch"
    topic_of = np.array([r["id"] % n_topics for r in rows])
    F = cvf["features"][:, L].float().numpy()
    tr = np.isin(topic_of, train_t)
    cms = np.stack([F[(lab == k) & tr].mean(0) for k in range(len(concepts))])
    gm = cms[emo].mean(0)
    chat = (cms - gm) / np.linalg.norm(cms - gm, axis=1, keepdims=True)          # [12, d] unit concept vectors
    tgt_hat = chat[concepts.index(TGT)]
    del cvf, F

    # ---------------- per-token layer-L residuals of paragraphs (plain model, raw text) ----------------
    grab = {}
    def fwd_resid(ids):
        hk = layers[L].register_forward_hook(lambda _m, _i, o: grab.__setitem__("H", common.layer_output_hidden(o)[0].float().cpu()))
        try:
            with torch.no_grad():
                model(input_ids=torch.tensor([ids], device=dev))
        finally:
            hk.remove()
        return grab["H"].numpy()
    resid = {}
    for c in SHOW:
        for t in range(n_topics):
            ids = tok(text[(c, t)]).input_ids
            resid[(c, t)] = (ids, fwd_resid(ids))
    print(f"captured per-token residuals for {len(resid)} paragraphs")
    def dim_score_text(s):
        H = fwd_resid(tok(s).input_ids)
        return float((H.mean(0) - gm) @ tgt_hat)

    train_ex = [example(t) for t in train_t]
    held_ex = [example(t) for t in held_t]

    # ---------------- train both position schemes from the same init ----------------
    def train(scheme):
        random.seed(args.seed); torch.manual_seed(args.seed)
        reft = pyreft.get_reft_model(model, pyreft.ReftConfig(representations={
            "component": f"model.language_model.layers[{L}].output", "low_rank_dimension": args.rank,
            "intervention": pyreft.LoreftIntervention(embed_dim=Hd, low_rank_dimension=args.rank)}))
        reft.set_device(dev)
        model.train()     # AFTER get_reft_model (it calls model.eval()); enables checkpointing, no dropout
        iv = list(reft.interventions.values())[0]; iv = iv[0] if isinstance(iv, (list, tuple)) else iv
        params = [p for p in reft.parameters() if p.requires_grad]
        def loss_of(ids, labels, plen):
            # same objective as HF's labels-loss (mean next-token CE over the answer tokens), but memory-lean:
            # logits only from the last prompt token on, and the float32 CE computed in recomputed chunks.
            keep = ids.shape[1] - plen + 1
            _, o = reft(base={"input_ids": ids, "attention_mask": torch.ones_like(ids), "logits_to_keep": keep},
                        unit_locations={"sources->base": (None, [[positions_for(plen, scheme)]])})
            return chunked_ce(o.logits[0, :-1], ids[0, plen:])
        def held_loss():
            with torch.no_grad():
                return sum(float(loss_of(*e)) for e in held_ex) / len(held_ex)
        steps = args.epochs * -(-len(train_ex) // args.accum)
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)             # wd MUST be 0 (orthogonal R)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps))
        clone = lambda sd: {k: v.detach().clone() for k, v in sd.items()}
        log = [{"epoch": 0, "train": None, "held": held_loss()}]
        best = (log[0]["held"], 0, clone(iv.state_dict()))
        torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        for ep in range(1, args.epochs + 1):
            order = list(range(len(train_ex))); random.shuffle(order); tot = 0.0
            for i, idx in enumerate(order):
                loss = loss_of(*train_ex[idx]); (loss / args.accum).backward(); tot += float(loss)
                if (i + 1) % args.accum == 0 or i + 1 == len(order):
                    opt.step(); sched.step(); opt.zero_grad()
            hl = held_loss(); log.append({"epoch": ep, "train": tot / len(order), "held": hl})
            if hl < best[0]: best = (hl, ep, clone(iv.state_dict()))
            print(f"  [{scheme}] epoch {ep:>2} | train {log[-1]['train']:.3f} | held-out {hl:.3f} | {time.time()-t0:.0f}s")
        model.eval()
        iv.load_state_dict(best[2])                                              # keep the best held-out epoch
        info = {"best_epoch": best[1], "best_held": best[0], "seconds": time.time() - t0,
                "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
        torch.save({"layer": L, "rank": args.rank, "positions": scheme, "state_dict": best[2], "log": log,
                    "train_topics": [topics[t] for t in train_t], "heldout_topics": [topics[t] for t in held_t],
                    "task": f"tone-free prompt -> {TGT} paragraph", "concept": TGT, "contrast": CON, "args": vars(args), **info}, ck_path[scheme])
        print(f"  [{scheme}] best epoch {best[1]} (held-out {best[0]:.3f}) | peak {info['peak_gb']:.2f} GB | saved -> {ck_path[scheme]}")
        return reft, iv, log, info

    # plain generations first (no intervention wrapper involved)
    gens = {"plain": {}}
    for t in held_t:
        enc = torch.tensor([prompt_ids(t)], device=dev)
        with torch.inference_mode():
            o = model.generate(input_ids=enc, max_new_tokens=args.new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
        gens["plain"][t] = tok.decode(o[0, enc.shape[1]:], skip_special_tokens=True).strip()

    results = {}
    for scheme in SCHEMES:
        reft, iv, log, info = train(scheme)
        gens[scheme] = {}
        for t in held_t:
            pi = torch.tensor([prompt_ids(t)], device=dev)
            with torch.inference_mode():
                _, g = reft.generate({"input_ids": pi, "attention_mask": torch.ones_like(pi)},
                                     unit_locations={"sources->base": (None, [[positions_for(pi.shape[1], scheme)]])},
                                     intervene_on_prompt=True, max_new_tokens=args.new_tokens, do_sample=False,
                                     eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
            gens[scheme][t] = tok.decode(g[0, pi.shape[1]:], skip_special_tokens=True).strip()
        results[scheme] = {"log": log, "info": info, "sd": {k: v.float().cpu() for k, v in iv.state_dict().items()}}
        del reft

    # ---------------- reading analysis ----------------
    def feats_of(Hs, Qt, S, P, Ub, Rnd):
        read = Hs @ Qt.T                                            # [n, r]
        return {"dim": (Hs - gm) @ tgt_hat, "read": read, "write": Hs @ P, "rand": Hs @ Rnd.T,
                "push": np.linalg.norm(read * S + Ub, axis=1), "full": Hs}
    rng = np.random.default_rng(args.seed)
    Rnd = np.linalg.qr(rng.standard_normal((Hd, args.rank)))[0].T  # random orthonormal 4 x d (control)
    stack = lambda c, ts: np.concatenate([resid[(c, t)][1] for t in ts])
    analysis = {}
    for scheme in SCHEMES:
        sd = results[scheme]["sd"]
        R = sd["rotate_layer"].numpy().T; W = sd["weight"].numpy(); b = sd["bias"].numpy()
        V = W - R
        U, S, Qt = np.linalg.svd(V, full_matrices=False)
        # sign convention: each read direction scores target-concept tokens above neutral ones (training topics)
        sgn = np.sign((stack(TGT,train_t) @ Qt.T).mean(0) - (stack("neutral", train_t) @ Qt.T).mean(0)); sgn[sgn == 0] = 1
        Qt = Qt * sgn[:, None]; U = U * sgn[None, :]
        P = R.T @ U; Ub = U.T @ b
        assert np.allclose(P @ np.diag(S) @ Qt, R.T @ V, atol=1e-4), "SVD reassembly failed"
        c_vec = R.T @ b
        a = {"S": S.tolist(), "cos_read": (Qt @ chat.T).tolist(), "cos_write": (P.T @ chat.T).tolist(),
             "cos_qp": [float(Qt[i] @ P[:, i]) for i in range(args.rank)], "b_norm": float(np.linalg.norm(b)),
             "cos_const_tgt": float(c_vec @ tgt_hat / (np.linalg.norm(c_vec) + 1e-12))}
        # probes: target vs neutral / target vs contrast; fit on training-topic tokens, test on held-out-topic tokens
        probes = {}
        for neg in ["neutral", CON]:
            ftr = {c: feats_of(stack(c, train_t), Qt, S, P, Ub, Rnd) for c in (TGT, neg)}
            fte = {c: feats_of(stack(c, held_t), Qt, S, P, Ub, Rnd) for c in (TGT, neg)}
            pr = {}
            for k in ["dim", "push", "read", "write", "rand", "full"]:
                Xp, Xn = ftr[TGT][k], ftr[neg][k]
                if Xp.ndim == 1:                                     # 1-d score: learn only its sign
                    s = 1.0 if Xp.mean() >= Xn.mean() else -1.0
                    sp, sn = s * fte[TGT][k], s * fte[neg][k]
                else:
                    w = lda_fit(Xp, Xn, shrink=1e-2 if k != "full" else 1e-1)
                    sp, sn = fte[TGT][k] @ w, fte[neg][k] @ w
                pr[k] = auc(sp, sn)
            # passage level: mean-pooled features, LDA on 24 train passages per class, test 6 vs 6
            pas = {}
            for k in ["dim", "push", "read", "write", "rand"]:
                mtr = {c: np.stack([np.atleast_1d(feats_of(resid[(c, t)][1], Qt, S, P, Ub, Rnd)[k]).reshape(len(resid[(c, t)][1]), -1).mean(0) for t in train_t]) for c in (TGT, neg)}
                mte = {c: np.stack([np.atleast_1d(feats_of(resid[(c, t)][1], Qt, S, P, Ub, Rnd)[k]).reshape(len(resid[(c, t)][1]), -1).mean(0) for t in held_t]) for c in (TGT, neg)}
                w = lda_fit(mtr[TGT], mtr[neg], shrink=1e-1)
                pas[k] = auc(mte[TGT] @ w, mte[neg] @ w)
            probes[neg] = {"token": pr, "passage": pas}
        a["probes"] = probes
        # per-token traces on held-out topics
        w_read = lda_fit(feats_of(stack(TGT,train_t), Qt, S, P, Ub, Rnd)["read"],
                         feats_of(stack("neutral", train_t), Qt, S, P, Ub, Rnd)["read"])
        tr_scores = {}
        for t in held_t:
            for c in SHOW:
                f = feats_of(resid[(c, t)][1], Qt, S, P, Ub, Rnd)
                d = {"dim": f["dim"], "read LDA": f["read"] @ w_read, "push": f["push"]}
                for i in range(args.rank):
                    d[f"q{i+1}"] = f["read"][:, i]
                tr_scores[f"{t}|{c}"] = d
        # standardise each score for display: centre on neutral tokens, scale by std over all shown tokens
        keys = list(next(iter(tr_scores.values())).keys())
        disp = {}
        for k in keys:
            allv = np.concatenate([v[k] for v in tr_scores.values()])
            mu = np.concatenate([v[k] for kk, v in tr_scores.items() if kk.endswith("|neutral")]).mean()
            sdv = allv.std() + 1e-9
            disp[k] = {kk: np.round((v[k] - mu) / sdv, 2).tolist() for kk, v in tr_scores.items()}
        a["traces"] = disp
        a["trace_keys"] = keys
        analysis[scheme] = a
        print(f"[{scheme}] gains {np.round(S, 2).tolist()} | cos(q_i, {TGT}) {np.round(Qt @ tgt_hat, 2).tolist()} | "
              f"cos(p_i, {TGT}) {np.round(P.T @ tgt_hat, 2).tolist()}")
        print(f"   token AUC {TGT}-vs-neutral: " + ", ".join(f"{k} {v:.3f}" for k, v in probes["neutral"]["token"].items()))

    # generation check: target-concept DiM score (mean-pooled, layer L) of generated text vs the corpus paragraphs
    gen_scores = {k: {t: dim_score_text(g) for t, g in v.items()} for k, v in gens.items()}
    ref_scores = {c: {t: dim_score_text(text[(c, t)]) for t in held_t} for c in SHOW}
    for k in gen_scores:
        print(f"  generated [{k:5}] mean {TGT}-DiM score {np.mean(list(gen_scores[k].values())):+.3f}")
    for c in SHOW:
        print(f"  corpus    [{c:7}] mean {TGT}-DiM score {np.mean(list(ref_scores[c].values())):+.3f}")

    out = {"layer": L, "rank": args.rank, "concept": TGT, "contrast": CON, "show": SHOW, "concepts": concepts, "topics": topics, "train_topics": train_t, "heldout_topics": held_t,
           "checkpoints": ck_path, "training": {s: {"log": results[s]["log"], **results[s]["info"]} for s in SCHEMES},
           "analysis": analysis, "generations": {k: {str(t): g for t, g in v.items()} for k, v in gens.items()},
           "gen_scores": {k: {str(t): s for t, s in v.items()} for k, v in gen_scores.items()},
           "ref_scores": {c: {str(t): s for t, s in v.items()} for c, v in ref_scores.items()},
           "tokens": {f"{t}|{c}": [tok.decode([i]) for i in resid[(c, t)][0]] for t in held_t for c in SHOW},
           "paragraphs": {f"{t}|{c}": text[(c, t)] for t in held_t for c in SHOW}}
    json.dump(out, open(rep + ".json", "w"), ensure_ascii=False)
    open(rep + ".html", "w").write(render_html(out))
    print(f"Wrote {rep}.html / .json")


# ------------------------------------------------------------------------------------------------------
CSS = """
:root{--bg:#fbfaf7;--fg:#171613;--muted:#76746c;--line:#e3e1d8;--head:#f1efe7;--card:#fff;--acc:#1f6fb2;--pos:230,120,20;--neg:40,110,200}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141412;--fg:#ecebe5;--muted:#9b998f;--line:#312f2a;--head:#1e1d1a;--card:#1a1917;--acc:#62a8e6;--pos:240,150,60;--neg:90,160,240}}
:root[data-theme="dark"]{--bg:#141412;--fg:#ecebe5;--muted:#9b998f;--line:#312f2a;--head:#1e1d1a;--card:#1a1917;--acc:#62a8e6;--pos:240,150,60;--neg:90,160,240}
body{font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--fg);max-width:1240px;margin:0 auto;padding:1.6rem 1.1rem 3rem}
h1{font-size:1.4rem;margin:0 0 .3rem} h2{font-size:1.06rem;margin:2.1rem 0 .5rem} h3{font-size:.95rem;margin:1.2rem 0 .4rem}
.lede,.note{color:var(--muted);max-width:940px} .note{font-size:.86rem}
.wrap{overflow-x:auto} table{border-collapse:collapse} th,td{border-bottom:1px solid var(--line);padding:6px 9px;text-align:left;vertical-align:top}
th{background:var(--head);font-size:.8rem;color:var(--muted);font-weight:600} td.num{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
td.best{font-weight:700;color:var(--acc)} .muted{color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:.7rem .9rem;margin:0 0 1rem}
.gen{white-space:pre-wrap;font-size:.9rem;max-width:360px}
.controls{position:sticky;top:0;background:var(--bg);padding:.5rem 0;z-index:2;border-bottom:1px solid var(--line);display:flex;gap:1rem;flex-wrap:wrap;align-items:center}
select{font:inherit;padding:2px 6px} .para{line-height:1.9;margin:.2rem 0 .8rem} .lab{font-weight:600;font-size:.82rem;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.t{border-radius:3px;padding:1px 0} .legend span{padding:1px 6px;border-radius:3px}
"""

JS = """
const D = __DATA__;
const ckSel = document.getElementById('ck'), scSel = document.getElementById('sc');
function fill(){ const keys = D.analysis[ckSel.value].trace_keys; const cur = scSel.value;
  scSel.innerHTML = ''; keys.forEach(k => { const o = document.createElement('option'); o.value = k; o.textContent = k; scSel.appendChild(o); });
  if (keys.includes(cur)) scSel.value = cur; }
function col(z){ const a = Math.min(1, Math.abs(z) / 2.5) * 0.8;
  const v = getComputedStyle(document.documentElement).getPropertyValue(z >= 0 ? '--pos' : '--neg').trim();
  return `rgba(${v},${a})`; }
function paint(){ const tr = D.analysis[ckSel.value].traces[scSel.value];
  document.querySelectorAll('.para').forEach(p => { const z = tr[p.dataset.key]; p.querySelectorAll('.t').forEach((s, i) => {
    s.style.background = col(z[i]); s.title = scSel.value + ': ' + z[i]; }); }); }
const box = document.getElementById('traces');
D.heldout_topics.forEach(t => { const card = document.createElement('div'); card.className = 'card';
  const h = document.createElement('h3'); h.textContent = D.topics[t]; card.appendChild(h);
  D.show.forEach(c => { const lab = document.createElement('div'); lab.className = 'lab'; lab.textContent = c; card.appendChild(lab);
    const p = document.createElement('div'); p.className = 'para'; p.dataset.key = t + '|' + c;
    D.tokens[t + '|' + c].forEach(tt => { const s = document.createElement('span'); s.className = 't'; s.textContent = tt; p.appendChild(s); });
    card.appendChild(p); });
  box.appendChild(card); });
ckSel.onchange = () => { fill(); paint(); }; scSel.onchange = paint; fill(); paint();
"""

def render_html(o):
    E = html.escape
    A, T, C = o["analysis"], o["training"], o["concepts"]
    T_, C_ = o["concept"], o["contrast"]
    SN = {"all": "all prompt tokens", "last": "last prompt token"}
    r = o["rank"]

    # 1. does it write the target concept?
    gs, rs = o["gen_scores"], o["ref_scores"]
    mean = lambda d: sum(d.values()) / len(d)
    gen_rows = "".join(
        f"<tr><td>{E(o['topics'][t])}</td>" + "".join(
            f"<td class=gen>{E(o['generations'][k][str(t)][:420])}<div class=muted>{T_}-DiM {gs[k][str(t)]:+.2f}</div></td>"
            for k in ["plain", "all", "last"]) + "</tr>" for t in o["heldout_topics"])
    score_row = (f"<tr><td class=muted>mean {T_}-DiM score</td>" + "".join(f"<td class=num>{mean(gs[k]):+.2f}</td>" for k in ["plain", "all", "last"])
                 + "</tr>")
    refs = " · ".join(f"{c} {mean(rs[c]):+.2f}" for c in o["show"])

    # 2. training
    tr_rows = ""
    n_ep = max(len(T[s]["log"]) for s in SCHEMES)
    for i in range(n_ep):
        tr_rows += f"<tr><td class=num>{i}</td>"
        for s in SCHEMES:
            lg = T[s]["log"][i]; best = i == T[s]["best_epoch"]
            tr_rows += (f"<td class=num>{'' if lg['train'] is None else format(lg['train'], '.3f')}</td>"
                        f"<td class='num{' best' if best else ''}'>{lg['held']:.3f}</td>")
        tr_rows += "</tr>"

    # 3. structure: gains + cosine tables
    def cos_table(s, key, vec_name):
        M = A[s][key]
        head = "".join(f"<th>{E(c)}</th>" for c in C)
        body = ""
        for i in range(r):
            cells = "".join(
                f"<td class=num style='background:rgba(var(--{'pos' if v >= 0 else 'neg'}),{min(1, abs(v) / 0.6) * 0.7:.2f})'>{v:+.2f}</td>"
                for v in M[i])
            body += f"<tr><td class=num><b>{vec_name}{i+1}</b> <span class=muted>σ={A[s]['S'][i]:.2f}</span></td>{cells}</tr>"
        return f"<div class=wrap><table><tr><th></th>{head}</tr>{body}</table></div>"
    struct = ""
    for s in SCHEMES:
        a = A[s]
        struct += (f"<h3>LoReFT — {SN[s]}</h3><div class=note>gains σ = {', '.join(f'{x:.2f}' for x in a['S'])} · "
                   f"cos(q_i, p_i) = {', '.join(f'{x:+.2f}' for x in a['cos_qp'])} · ‖b‖ = {a['b_norm']:.2f} · "
                   f"cos(constant Rᵀb, {T_}) = {a['cos_const_tgt']:+.2f}</div>"
                   f"<div class=note><b>read directions qᵢ</b> vs the 12 concept vectors</div>{cos_table(s, 'cos_read', 'q')}"
                   f"<div class=note><b>write directions pᵢ</b> vs the 12 concept vectors</div>{cos_table(s, 'cos_write', 'p')}")

    # 4. probes
    FEAT = [("dim", f"{T_} DiM projection", "1"), ("push", "LoReFT push ‖Mh + c‖", "1"), ("read", "LoReFT read space Qᵀh", str(r)),
            ("write", "LoReFT write space (RᵀU)ᵀh", str(r)), ("rand", "random subspace (control)", str(r)), ("full", "full residual (upper bound)", "2560")]
    pr_rows = ""
    for k, name, dim in FEAT:
        pr_rows += f"<tr><td>{E(name)}</td><td class=num>{dim}</td>"
        for s in SCHEMES:
            for neg in ["neutral", C_]:
                tok_auc = A[s]["probes"][neg]["token"][k]
                pas = A[s]["probes"][neg]["passage"].get(k)
                pr_rows += f"<td class=num>{tok_auc:.3f}</td><td class=num>{'—' if pas is None else format(pas, '.2f')}</td>"
        pr_rows += "</tr>"
    pr_head = "".join(f"<th colspan=4>{SN[s]}</th>" for s in SCHEMES)
    pr_sub = "".join(f"<th>{T_} vs neutral<br>token</th><th>passage</th><th>{T_} vs {C_}<br>token</th><th>passage</th>" for _ in SCHEMES)

    ck_opts = "".join(f"<option value='{s}'>{SN[s]}</option>" for s in SCHEMES)
    data = json.dumps({k: o[k] for k in ["analysis", "tokens", "topics", "heldout_topics", "show"]}, ensure_ascii=False)
    return f"""<title>{T_.capitalize()} LoReFT reader</title>
<style>{CSS}</style>
<h1>{T_.capitalize()} LoReFT — reading with a steering intervention</h1>
<div class=lede>Qwen3.5-4B, layer {o['layer']}, rank {r}. Trained (pyreft) on {len(o['train_topics'])} topics of the concept corpus to turn the
tone-free prompt into the <b>{T_}</b> paragraph; {len(o['heldout_topics'])} topics held out. Two variants from the same initialisation:
edits on every prompt token vs only the last one; each keeps its best held-out epoch. Then the intervention is used as a <b>reader</b>:
V = W − R = UΣQᵀ gives read directions qᵢ (rows of Qᵀ), write directions pᵢ (columns of RᵀU) and gains σᵢ, since M = Rᵀ V = Σᵢ σᵢ pᵢ qᵢᵀ.
The {T_} DiM vector here is recomputed on the training topics only.</div>

<h2>1 · Does it write {T_} text? <span class=muted>(held-out prompts, no tone requested)</span></h2>
<div class=note>{T_}-DiM score = mean-pooled layer-{o['layer']} projection onto the {T_} concept vector. Corpus reference on these topics: {refs}.</div>
<div class=wrap><table><tr><th>topic</th><th>plain model</th><th>LoReFT — all prompt tokens</th><th>LoReFT — last prompt token</th></tr>{gen_rows}{score_row}</table></div>

<h2>2 · Training <span class=muted>(held-out loss; bold = kept epoch)</span></h2>
<div class=wrap><table><tr><th>epoch</th><th>all: train</th><th>all: held-out</th><th>last: train</th><th>last: held-out</th></tr>{tr_rows}</table></div>
<div class=note>all: best epoch {T['all']['best_epoch']}, {T['all']['seconds']:.0f}s, peak {T['all']['peak_gb']:.2f} GB · last: best epoch {T['last']['best_epoch']}, {T['last']['seconds']:.0f}s, peak {T['last']['peak_gb']:.2f} GB</div>

<h2>3 · What does it read and write? <span class=muted>(cosine with the 12 concept vectors)</span></h2>
<div class=note>Signs fixed so every qᵢ scores {T_} tokens above neutral ones; pᵢ flips with it. If {T_} were a single read-and-write axis you would see
one dominant σ with q₁ and p₁ both aligned to <i>{T_}</i>.</div>
{struct}

<h2>4 · How well does each space read {T_}? <span class=muted>(AUC on held-out topics)</span></h2>
<div class=note>Token level: Fisher-LDA probe fitted on training-topic tokens, tested on held-out-topic tokens (1-d scores only learn a sign).
Passage level: mean-pooled features, 24 vs 24 training passages, tested on 6 vs 6 held-out passages (coarse: steps of 1/36).
0.5 = chance, 1.0 = perfect.</div>
<div class=wrap><table><tr><th>feature</th><th>dims</th>{pr_head}</tr><tr><th></th><th></th>{pr_sub}</tr>{pr_rows}</table></div>

<h2>5 · Per-token reads on held-out paragraphs</h2>
<div class=controls><label>intervention <select id=ck>{ck_opts}</select></label><label>score <select id=sc></select></label>
<span class="legend note"><span style="background:rgba(var(--pos),.6)">more than neutral</span> <span style="background:rgba(var(--neg),.6)">less than neutral</span>
· standardised: centred on neutral tokens, scaled by the std of all shown tokens · hover a token for its value</span></div>
<div id=traces></div>

<h2>Notes</h2>
<ul class=note>
<li>The read directions were shaped by gradients at the positions each variant edits (prompt tokens), but are applied here to paragraph
tokens in plain reading — outside what they were trained on. The last-token variant saw only one kind of state.</li>
<li>Held-out topics are used both to pick the epoch (held-out loss) and to evaluate reading, a mild selection effect.</li>
<li>Small data: 24 training pairs, 6 held-out topics; one seed.</li>
</ul>
<script>{JS.replace('__DATA__', data)}</script>"""


if __name__ == "__main__":
    main()
