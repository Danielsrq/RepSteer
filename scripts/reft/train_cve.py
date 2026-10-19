"""Train LoReFTs on REAL CVE write-up spans and read them back as probes.

This replaces the synthetic security corpus. The synthetic snippets were short, uniform and
written by the model itself, so a reader fitted on them saw a distribution nothing like a real
write-up -- which is why the CVE tab looked so weak. Here every example is a contiguous labelled
region of a genuine published write-up (see cve_regions.py), with the article's preceding text as
context, and nftables + smbghost are held out entirely.

Objective is the same one the emotion/emoji LoReFTs used: the intervention must make the model
*write* the concept. Prompt = instruction + preceding context (which does not itself state the
mechanism); target = the region. Read directions then fall out of V = W - R as before.

Outputs go to data/reft/cve/ -- the synthetic checkpoints in data/reft/security/ are untouched.

Usage:
  .venv/bin/python scripts/reft/train_cve.py --concept uaf                 # one concept, both schemes
  .venv/bin/python scripts/reft/train_cve.py --all                         # every mechanism class
  .venv/bin/python scripts/reft/train_cve.py --concept uaf --schemes last  # quicker pilot
"""
import argparse, json, os, random, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import pyreft

from span_labels import MECHANISM
from cve_regions import NEGATIVE

SCHEMES = ["last", "all"]
INSTR = ("You are reading a security vulnerability write-up. Continue it.\n\n"
         "--- write-up so far ---\n{ctx}\n--- continue ---")

# The "classify" objective. The generative objective above only ever sees positive regions, so
# nothing pushes its read directions to be discriminative -- the pilot confirmed that (read AUC
# was indistinguishable from a random rank-4 subspace, cos(q,DiM) = 0.03). Here the region sits in
# the PROMPT, the intervention acts on the region's own token positions, and the target is a
# single yes/no token, so the intervention is trained on both classes and is optimised for exactly
# the thing we then read out of it.
QUESTION = {"uaf": "a use-after-free or double-free",
            "heap_overflow": "a heap buffer overflow",
            "stack_overflow": "a stack buffer overflow",
            "integer_overflow": "an integer overflow, underflow or truncation",
            "race_condition": "a race condition or TOCTOU flaw",
            "oob_access": "an out-of-bounds read or write",
            "command_injection": "command, code or expression injection",
            "auth_bypass": "an authentication or authorisation bypass",
            "priv_esc": "a privilege escalation step",
            "uninit_memory": "use of uninitialised memory",
            "other_vuln": "a software vulnerability",
            # narrative classes. These were only ever used as NEGATIVES before, so they had no
            # LoReFT of their own -- which meant the rank-4 read/write channels could not compete
            # on them, and "LoReFT read is useless" was only ever tested on mechanism classes where
            # nothing was linearly readable at all. These are the fair test.
            "background": "background, setup, environment or context rather than a specific finding",
            "exploit": "how to exploit the bug -- strategy, heap grooming, primitives or exploit code",
            "debugging": "debugger output, a crash dump, sanitizer report, log or instrumentation",
            "patch": "the upstream fix, a patch hunk or a mitigation",
            "benign": "ordinary code or output with no security content"}
INSTR_CLS = ("Read this passage from a security write-up.\n\n"
             "--- passage ---\n{txt}\n--- end ---\n\n"
             "Does the passage explain {q}? Answer yes or no.")


def positions_for(n, scheme):
    return [n - 1] if scheme == "last" else list(range(n))


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def chunked_ce(logits, targets, chunk=64):
    tot = torch.zeros((), device=logits.device, dtype=torch.float32)
    for s in range(0, len(targets), chunk):
        tot = tot + torch.utils.checkpoint.checkpoint(
            lambda x, y: torch.nn.functional.cross_entropy(x.float(), y, reduction="sum"),
            logits[s:s + chunk], targets[s:s + chunk], use_reentrant=False)
    return tot / len(targets)


def lda_fit(Xp, Xn, shrink=1e-2):
    mp, mn = Xp.mean(0), Xn.mean(0)
    Xc = np.concatenate([Xp - mp, Xn - mn]); S = Xc.T @ Xc / len(Xc)
    S = S + shrink * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
    return np.linalg.solve(S, mp - mn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/security/cve_dataset.json")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ctx-tok", type=int, default=640, help="tokens of preceding context kept")
    ap.add_argument("--tgt-tok", type=int, default=512, help="tokens of the region used as target")
    ap.add_argument("--max-tok-per-region", type=int, default=128,
                    help="residual tokens cached per region (subsampled) for the read probes")
    ap.add_argument("--objective", default="classify", choices=["write", "classify"],
                    help="write: intervention must make the model produce the region (concept as "
                         "style, as in the emotion LoReFTs). classify: region is in the prompt and "
                         "the intervention must make the model answer yes/no (concept as content, "
                         "trained against negatives)")
    ap.add_argument("--neg-per-pos", type=int, default=4, help="classify: negatives sampled per positive")
    ap.add_argument("--concept", default="uaf")
    ap.add_argument("--all", action="store_true", help="every mechanism class present in the data")
    ap.add_argument("--narrative", action="store_true",
                    help="train the narrative/null classes too (background, exploit, ...)")
    ap.add_argument("--schemes", default="last,all")
    ap.add_argument("--out-dir", default="data/reft/cve")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    args = ap.parse_args()
    schemes = args.schemes.split(",")

    d = json.load(open(args.data))
    recs = d["records"]
    mech = [c for c in d["classes"] if c in MECHANISM]
    narr = [c for c in d["classes"] if c in NEGATIVE]
    todo = ((mech if args.all else []) + (narr if args.narrative else [])
            or [args.concept])
    for c in todo:
        assert c in d["classes"], f"{c} not modelled in {args.data} (classes: {d['classes']})"

    L, od = args.layer, args.out_dir
    os.makedirs(od, exist_ok=True)
    tag = "cls" if args.objective == "classify" else "wr"
    planned = [f"{od}/{c}_loreft_L{L}_{tag}_{s}.pt" for c in todo for s in schemes]
    for f in planned:
        if os.path.exists(f) and not args.force:
            sys.exit(f"refusing to overwrite existing {f} (pass --force or a different --out-dir)")

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} | L{L} rank {args.rank} | concepts: {', '.join(todo)} | schemes {schemes}")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    for p in model.parameters():
        p.requires_grad_(False)
    _, layers = common.get_decoder_layers(model)
    Hd = model.config.get_text_config().hidden_size
    dev = next(model.parameters()).device
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        for i, layer in enumerate(layers):
            layer.gradient_checkpointing = i > L

    def render(msgs, gen):
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen)

    def build_example(r):
        """(ids, prompt_len) for one region. Context is truncated from the LEFT (the text nearest
        the region is what conditions it); the target from the right."""
        ctx_ids = tok(r["context"], add_special_tokens=False).input_ids[-args.ctx_tok:]
        ctx = tok.decode(ctx_ids) if ctx_ids else "(start of article)"
        tgt_ids = tok(r["text"], add_special_tokens=False).input_ids[:args.tgt_tok]
        tgt = tok.decode(tgt_ids)
        p = tok(render([{"role": "user", "content": INSTR.format(ctx=ctx)}], True),
                add_special_tokens=False).input_ids
        f = tok(render([{"role": "user", "content": INSTR.format(ctx=ctx)},
                        {"role": "assistant", "content": tgt}], False),
                add_special_tokens=False).input_ids
        assert f[:len(p)] == p, "chat-template prefix mismatch"
        return torch.tensor([f], device=dev), len(p)

    def build_cls_example(r, concept, yes):
        """(ids, prompt_len) for the classify objective: the region is the prompt, the answer is
        one token. Intervening on the prompt positions is intervening on the region itself."""
        txt_ids = tok(r["text"], add_special_tokens=False).input_ids[:args.tgt_tok]
        q = INSTR_CLS.format(txt=tok.decode(txt_ids), q=QUESTION[concept])
        p = tok(render([{"role": "user", "content": q}], True), add_special_tokens=False).input_ids
        f = tok(render([{"role": "user", "content": q},
                        {"role": "assistant", "content": "yes" if yes else "no"}], False),
                add_special_tokens=False).input_ids
        assert f[:len(p)] == p, "chat-template prefix mismatch"
        return torch.tensor([f], device=dev), len(p)

    # ---------------- cache layer-L residuals of every region's RAW text (no chat template) -----
    # This is what a reader sees at inference time on a real write-up, so it is what the probes
    # are fitted and scored on.
    rng = np.random.default_rng(args.seed)
    grab = {}

    def fwd_resid(ids):
        hk = layers[L].register_forward_hook(
            lambda _m, _i, o: grab.__setitem__("H", common.layer_output_hidden(o)[0].float().cpu()))
        try:
            with torch.no_grad():
                model(input_ids=torch.tensor([ids], device=dev))
        finally:
            hk.remove()
        return grab["H"].numpy()

    t0 = time.time()
    H = []                                    # per-record subsampled residuals, float16
    for k, r in enumerate(recs):
        ids = tok(r["text"], add_special_tokens=False).input_ids[:1024]
        h = fwd_resid(ids)
        if len(h) > args.max_tok_per_region:
            h = h[np.sort(rng.choice(len(h), args.max_tok_per_region, replace=False))]
        H.append(h.astype(np.float16))
        if (k + 1) % 100 == 0:
            print(f"  residuals {k+1}/{len(recs)}  ({time.time()-t0:.0f}s)")
    print(f"cached residuals for {len(recs)} regions in {time.time()-t0:.0f}s "
          f"({sum(len(x) for x in H)} tokens)")

    is_tr = np.array([r["split"] == "train" for r in recs])
    lab = np.array([r["label"] for r in recs])
    art = np.array([r["article"] for r in recs])
    gm = np.concatenate([H[i] for i in np.where(is_tr)[0]]).astype(np.float32).mean(0)

    def toks(mask):
        idx = np.where(mask)[0]
        return (np.concatenate([H[i] for i in idx]).astype(np.float32)
                if len(idx) else np.zeros((0, Hd), np.float32))

    summary = {}
    for concept in todo:
        pos_tr = (lab == concept) & is_tr
        neg_tr = np.isin(lab, NEGATIVE) & is_tr
        n_art = len(set(art[pos_tr]))
        # Loss-monitoring split: hold out one training article so the loss curve means something.
        # The real evaluation is the article holdout / leave-one-article-out below.
        arts_p = sorted(set(art[pos_tr]))
        hold_art = arts_p[-1] if len(arts_p) > 1 else None
        pidx = np.where(pos_tr)[0]
        if args.objective == "write":
            ex = [(build_example(recs[i]), art[i]) for i in pidx]
        else:
            # balance: every positive, plus neg_per_pos negatives drawn from the non-mechanism
            # classes and from OTHER mechanism classes (so "uaf" must beat "heap_overflow", not
            # just beat prose). Sampled per article so the held-out article stays clean.
            other = (np.isin(lab, mech + narr) & (lab != concept)) & is_tr
            nidx = np.where(other)[0]
            take = min(len(nidx), args.neg_per_pos * len(pidx))
            nsel = rng.choice(nidx, take, replace=False)
            ex = ([(build_cls_example(recs[i], concept, True), art[i]) for i in pidx] +
                  [(build_cls_example(recs[i], concept, False), art[i]) for i in nsel])
        held_ex = [e for e, a_ in ex if a_ == hold_art]
        fit_ex = [e for e, a_ in ex if a_ != hold_art]
        if not held_ex or not fit_ex:
            held_ex, fit_ex = [ex[-1][0]], [e for e, _ in ex]
        print(f"\n=== {concept} [{args.objective}]: {pos_tr.sum()} positive regions / {n_art} "
              f"articles | {len(fit_ex)} fit, {len(held_ex)} held [{hold_art}] ===")

        # ---- real-data DiM baseline (grand-mean-centred), fitted on training articles only ----
        dim = toks(pos_tr).mean(0) - gm
        dim_hat = dim / (np.linalg.norm(dim) + 1e-12)

        results = {}
        for scheme in schemes:
            random.seed(args.seed); torch.manual_seed(args.seed)
            reft = pyreft.get_reft_model(model, pyreft.ReftConfig(representations={
                "component": f"model.language_model.layers[{L}].output",
                "low_rank_dimension": args.rank,
                "intervention": pyreft.LoreftIntervention(embed_dim=Hd,
                                                          low_rank_dimension=args.rank)}))
            reft.set_device(dev)
            model.train()            # AFTER get_reft_model (it calls eval); enables checkpointing
            iv = list(reft.interventions.values())[0]
            iv = iv[0] if isinstance(iv, (list, tuple)) else iv
            params = [p for p in reft.parameters() if p.requires_grad]

            def loss_of(ids, plen):
                keep = ids.shape[1] - plen + 1
                _, o = reft(base={"input_ids": ids, "attention_mask": torch.ones_like(ids),
                                  "logits_to_keep": keep},
                            unit_locations={"sources->base": (None, [[positions_for(plen, scheme)]])})
                return chunked_ce(o.logits[0, :-1], ids[0, plen:])

            def held_loss():
                with torch.no_grad():
                    return sum(float(loss_of(*e)) for e in held_ex) / len(held_ex)

            steps = args.epochs * -(-len(fit_ex) // args.accum)
            opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)   # wd MUST be 0
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps))
            clone = lambda sd: {k: v.detach().clone() for k, v in sd.items()}
            log = [{"epoch": 0, "train": None, "held": held_loss()}]
            best = (log[0]["held"], 0, clone(iv.state_dict()))
            torch.cuda.reset_peak_memory_stats(); t1 = time.time()
            for ep in range(1, args.epochs + 1):
                order = list(range(len(fit_ex))); random.shuffle(order); tot = 0.0
                for i, idx in enumerate(order):
                    loss = loss_of(*fit_ex[idx]); (loss / args.accum).backward(); tot += float(loss)
                    if (i + 1) % args.accum == 0 or i + 1 == len(order):
                        opt.step(); sched.step(); opt.zero_grad()
                hl = held_loss(); log.append({"epoch": ep, "train": tot / len(order), "held": hl})
                if hl < best[0]: best = (hl, ep, clone(iv.state_dict()))
                print(f"  [{concept}/{scheme}] ep {ep:>2} | train {log[-1]['train']:.3f} | "
                      f"held {hl:.3f} | {time.time()-t1:.0f}s")
            model.eval()
            iv.load_state_dict(best[2])
            info = {"best_epoch": best[1], "best_held": best[0], "seconds": time.time() - t1,
                    "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
                    "n_fit": len(fit_ex), "n_held": len(held_ex), "n_articles": n_art}
            ck = f"{od}/{concept}_loreft_L{L}_{tag}_{scheme}.pt"
            torch.save({"layer": L, "rank": args.rank, "positions": scheme, "concept": concept,
                        "state_dict": best[2], "log": log, "source": "real CVE write-up spans",
                        "objective": args.objective,
                        "train_articles": sorted(set(art[pos_tr])), "holdout": d["holdout"],
                        "dim": dim_hat, "args": vars(args), **info}, ck)
            print(f"  [{concept}/{scheme}] best ep {best[1]} ({best[0]:.3f}) | "
                  f"peak {info['peak_gb']:.2f} GB | -> {ck}")
            results[scheme] = {"sd": {k: v.float().cpu() for k, v in iv.state_dict().items()},
                               "info": info, "log": log}
            del reft

        # ---------------- read the intervention back as a probe ----------------
        Rnd = np.linalg.qr(rng.standard_normal((Hd, args.rank)))[0].T
        cs = {}
        for scheme in schemes:
            sd = results[scheme]["sd"]
            R = sd["rotate_layer"].numpy().T; W = sd["weight"].numpy(); b = sd["bias"].numpy()
            V = W - R
            U, S, Qt = np.linalg.svd(V, full_matrices=False)
            sgn = np.sign((toks(pos_tr) @ Qt.T).mean(0) - (toks(neg_tr) @ Qt.T).mean(0))
            sgn[sgn == 0] = 1
            Qt = Qt * sgn[:, None]; U = U * sgn[None, :]
            P = R.T @ U; Ub = U.T @ b
            assert np.allclose(P @ np.diag(S) @ Qt, R.T @ V, atol=1e-4), "SVD reassembly failed"

            def feats(X):
                read = X @ Qt.T
                return {"dim": (X - gm) @ dim_hat, "read": read, "write": X @ P,
                        "rand": X @ Rnd.T, "push": np.linalg.norm(read * S + Ub, axis=1)}

            def evaluate(fit_pos, fit_neg, ev_pos, ev_neg):
                ftr = {"p": feats(toks(fit_pos)), "n": feats(toks(fit_neg))}
                fte = {"p": feats(toks(ev_pos)), "n": feats(toks(ev_neg))}
                out = {}
                for k in ["dim", "push", "read", "write", "rand"]:
                    Xp, Xn = ftr["p"][k], ftr["n"][k]
                    if Xp.ndim == 1:
                        s = 1.0 if Xp.mean() >= Xn.mean() else -1.0
                        out[k] = auc(s * fte["p"][k], s * fte["n"][k])
                    else:
                        w = lda_fit(Xp, Xn)
                        out[k] = auc(fte["p"][k] @ w, fte["n"][k] @ w)
                return out

            ev = {"train_fit": evaluate(pos_tr, neg_tr, pos_tr, neg_tr)}
            # (a) the declared article holdout -- only classes present in nftables/smbghost
            ho_pos = (lab == concept) & ~is_tr
            ho_neg = np.isin(lab, NEGATIVE) & ~is_tr
            if ho_pos.sum():
                ev["holdout_article"] = evaluate(pos_tr, neg_tr, ho_pos, ho_neg)
            # (b) leave-one-article-out over the training articles -- free (residuals are cached),
            # and the only honest number for classes the holdout articles do not contain
            loao = {k: [] for k in ["dim", "push", "read", "write", "rand"]}
            for a_ in sorted(set(art[pos_tr])):
                tp, tn = pos_tr & (art != a_), neg_tr & (art != a_)
                ep_, en = pos_tr & (art == a_), neg_tr & (art == a_)
                if not tp.sum() or not ep_.sum() or not en.sum():
                    continue
                for k, v in evaluate(tp, tn, ep_, en).items():
                    loao[k].append(v)
            if loao["read"]:
                ev["leave_one_article_out"] = {k: float(np.nanmean(v)) for k, v in loao.items()}
                ev["loao_n_articles"] = len(loao["read"])
            cs[scheme] = {"S": S.tolist(), "cos_qp": [float(Qt[i] @ P[:, i]) for i in range(args.rank)],
                          "cos_read_dim": (Qt @ dim_hat).tolist(),
                          "cos_write_dim": (P.T @ dim_hat).tolist(),
                          "eval": ev, "info": results[scheme]["info"]}
            print(f"  [{concept}/{scheme}] gains {np.round(S,2).tolist()} "
                  f"cos(q_i,DiM) {np.round(Qt @ dim_hat,2).tolist()}")
            for name, e in ev.items():
                if isinstance(e, dict):
                    print(f"     AUC {name:<22} " +
                          "  ".join(f"{k} {v:.3f}" for k, v in e.items()))
        summary[f"{concept}|{args.objective}"] = {"n_pos": int(pos_tr.sum()), "n_articles": n_art,
                            "articles": sorted(set(art[pos_tr])), "schemes": cs}

    out = f"{od}/train_summary_L{L}.json"
    prev = json.load(open(out)) if os.path.exists(out) else {}
    prev.update(summary)
    json.dump(prev, open(out, "w"), indent=1)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
