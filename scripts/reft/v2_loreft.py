"""Train one rank-4 LoReFT reader per v2 label, on STREAMING-context residuals.

Fairness with the DiM/LDA readers (which are fit on the v2 streaming cache): each training example
is  [raw article context (<=ctx_tok)] [region tokens] [raw yes/no question]  fed with NO chat
template, and the intervention acts ONLY on the region token positions. Layer-17 residuals are
causal, so the region residuals depend on the raw preceding article and NOT on the trailing
question -- i.e. they are the same points the cache captured and the viewer reads. The appended
"...does this describe X? answer:" is just a supervised readout hook; the target is a single
yes/no token. We then read the intervention back through the SVD of V = W - R, exactly as before.

Outputs go to data/reft/v2/{label}_loreft_L{L}.pt.

  .venv/bin/python scripts/reft/v2_loreft.py --label overflow          # one, a pilot
  .venv/bin/python scripts/reft/v2_loreft.py --all
"""
import argparse, os, random, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common, pyreft
from span_labels_v2 import LABELS, CONCEPTS, HOLDOUT, ARTICLE_CONCEPT, regions

QUESTION = {"overflow": "a buffer, heap, integer or out-of-bounds overflow",
            "uaf": "a use-after-free, double-free or heap-lifetime manipulation",
            "race": "a race condition or timing window",
            "background": "background or how the system normally works",
            "triage": "the investigation or exploitation work that was done"}


def auc(p, n):
    p, n = np.asarray(p), np.asarray(n)
    if len(p) < 2 or len(n) < 2:
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([p, n]))) + 1
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def lda_fit(Xp, Xn, shrink=1e-1):
    mp, mn = Xp.mean(0), Xn.mean(0)
    Xc = np.concatenate([Xp - mp, Xn - mn]); S = Xc.T @ Xc / len(Xc)
    S = S + shrink * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
    return np.linalg.solve(S, mp - mn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ctx-tok", type=int, default=512)
    ap.add_argument("--reg-tok", type=int, default=1100, help="max region tokens (ctx+reg <= ~1600)")
    ap.add_argument("--min-tok", type=int, default=12, help="skip tiny regions")
    ap.add_argument("--neg-per-pos", type=int, default=3)
    ap.add_argument("--label", default="overflow")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out-dir", default="data/reft/v2")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    L, od = args.layer, args.out_dir
    todo = LABELS if args.all else [args.label]
    os.makedirs(od, exist_ok=True)
    for c in todo:
        f = f"{od}/{c}_loreft_L{L}.pt"
        if os.path.exists(f) and not args.force:
            sys.exit(f"refusing to overwrite {f} (pass --force)")

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} | v2 LoReFT readers @ L{L} | labels {todo}")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    for p in model.parameters():
        p.requires_grad_(False)
    _, layers = common.get_decoder_layers(model)
    Hd = model.config.get_text_config().hidden_size
    dev = next(model.parameters()).device
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for i, layer in enumerate(layers):
        layer.gradient_checkpointing = i > L

    YES = tok(" yes", add_special_tokens=False).input_ids[:1]
    NO = tok(" no", add_special_tokens=False).input_ids[:1]

    # --- build region examples with raw article context ---
    art_ids = {a: tok(open(f"data/jlens/cve/{a}.txt", encoding="utf-8").read(),
                      add_special_tokens=False).input_ids for a in ARTICLE_CONCEPT}
    # char->token offset per article, so a region's char span maps to token positions
    def region_examples():
        exs = []   # (article, label, ctx_ids, reg_ids)
        for a in ARTICLE_CONCEPT:
            text = open(f"data/jlens/cve/{a}.txt", encoding="utf-8").read()
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            ids, offs = enc.input_ids, enc["offset_mapping"]
            starts = np.array([o[0] for o in offs])
            for r in regions(a):
                # find the char range of this region in the text
                ci = text.find(r["text"][:80])
                if ci < 0:
                    continue
                cj = ci + len(r["text"])
                t0 = int(np.searchsorted(starts, ci, "left"))
                t1 = int(np.searchsorted(starts, cj, "left"))
                if t1 - t0 < args.min_tok:
                    continue
                reg = ids[t0:t1][:args.reg_tok]
                ctx = ids[max(0, t0 - args.ctx_tok):t0]
                exs.append({"article": a, "label": r["label"], "ctx": ctx, "reg": reg})
        return exs

    ALL = region_examples()
    print(f"{len(ALL)} region examples across {len(ARTICLE_CONCEPT)} articles")

    def build(ex, concept, yes):
        q = tok(f"\n\nQuestion: does the passage above describe {QUESTION[concept]}? Answer:",
                add_special_tokens=False).input_ids
        ans = YES if yes else NO
        full = ex["ctx"] + ex["reg"] + q + ans
        plen = len(ex["ctx"]) + len(ex["reg"]) + len(q)     # answer starts here
        reg_pos = list(range(len(ex["ctx"]), len(ex["ctx"]) + len(ex["reg"])))  # intervene here
        return torch.tensor([full], device=dev), plen, reg_pos

    rng = np.random.default_rng(args.seed)
    summary = {}
    for concept in todo:
        pos = [e for e in ALL if e["label"] == concept and e["article"] not in HOLDOUT]
        negpool = [e for e in ALL if e["label"] != concept and e["article"] not in HOLDOUT]
        neg = list(rng.choice(negpool, min(len(negpool), args.neg_per_pos * len(pos)), replace=False))
        train = [(e, True) for e in pos] + [(e, False) for e in neg]
        # monitor split: hold out one positive article's examples
        parts = sorted({e["article"] for e in pos})
        hold_a = parts[-1] if len(parts) > 1 else None
        held = [(e, y) for e, y in train if e["article"] == hold_a and y]
        fit = [(e, y) for e, y in train if not (e["article"] == hold_a and y)]
        if not held:
            held, fit = train[:2], train
        print(f"\n=== {concept}: {len(pos)} pos / {len(neg)} neg regions "
              f"({len(fit)} fit, {len(held)} held [{hold_a}]) ===")

        random.seed(args.seed); torch.manual_seed(args.seed)
        reft = pyreft.get_reft_model(model, pyreft.ReftConfig(representations={
            "component": f"model.language_model.layers[{L}].output", "low_rank_dimension": args.rank,
            "intervention": pyreft.LoreftIntervention(embed_dim=Hd, low_rank_dimension=args.rank)}))
        reft.set_device(dev); model.train()
        iv = list(reft.interventions.values())[0]; iv = iv[0] if isinstance(iv, (list, tuple)) else iv
        params = [p for p in reft.parameters() if p.requires_grad]

        def loss_of(ex, yes):
            ids, plen, reg_pos = build(ex, concept, yes)
            keep = ids.shape[1] - plen + 1
            _, o = reft(base={"input_ids": ids, "attention_mask": torch.ones_like(ids),
                              "logits_to_keep": keep},
                        unit_locations={"sources->base": (None, [[reg_pos]])})
            tgt = ids[0, plen:]
            return torch.nn.functional.cross_entropy(o.logits[0, -len(tgt) - 1:-1].float(), tgt)

        def held_loss():
            with torch.no_grad():
                return sum(float(loss_of(e, y)) for e, y in held) / len(held)

        steps = args.epochs * -(-len(fit) // args.accum)
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)   # wd MUST be 0
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps))
        clone = lambda sd: {k: v.detach().clone() for k, v in sd.items()}
        best = (held_loss(), 0, clone(iv.state_dict()))
        torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        for ep in range(1, args.epochs + 1):
            order = list(range(len(fit))); random.shuffle(order); tot = 0.0
            for i, idx in enumerate(order):
                l = loss_of(*fit[idx]); (l / args.accum).backward(); tot += float(l)
                if (i + 1) % args.accum == 0 or i + 1 == len(order):
                    opt.step(); sched.step(); opt.zero_grad()
            hl = held_loss()
            if hl < best[0]:
                best = (hl, ep, clone(iv.state_dict()))
            print(f"  [{concept}] ep {ep:>2} | train {tot/len(order):.3f} | held {hl:.3f} | {time.time()-t0:.0f}s")
        model.eval(); iv.load_state_dict(best[2])
        peak = torch.cuda.max_memory_allocated() / 1e9
        torch.save({"layer": L, "rank": args.rank, "label": concept, "state_dict": best[2],
                    "best_epoch": best[1], "best_held": best[0], "peak_gb": peak,
                    "n_pos": len(pos), "n_neg": len(neg), "args": vars(args)},
                   f"{od}/{concept}_loreft_L{L}.pt")
        print(f"  [{concept}] best ep {best[1]} ({best[0]:.3f}) | peak {peak:.2f} GB "
              f"-> {od}/{concept}_loreft_L{L}.pt")
        summary[concept] = {"best_epoch": best[1], "best_held": best[0], "peak_gb": peak}
        del reft
    print("\ndone:", summary)


if __name__ == "__main__":
    main()
