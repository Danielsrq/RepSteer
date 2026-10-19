"""Train one rank-r LoReFT per security concept on the security corpus (training only; no reading analysis).

Corpus: data/security/paragraphs_security.json -- short vulnerability write-ups (with code) for 10 vulnerability
classes plus 'benign' (secure-coding) and 'neutral' (plain systems explanations), ~15 per concept, each tagged
with a component ("topic") and a language. Topics are mostly unique per concept, so unlike the emotion corpus there
are no matched topic pairs. The recipe mirrors concept_read.py's "tone-free prompt -> concept paragraph":

    prompt = a vulnerability-free request naming only the component and language
    target = the concept's write-up for that component

so each LoReFT learns to steer an ordinary description into its vulnerability class (or into secure-coding style
for 'benign'). Per concept: --n-heldout write-ups held out (seeded), both position schemes (every prompt token /
last prompt token) from the same init, best held-out epoch kept, gradient checkpointing above the intervened layer.

Sanity check only: generations for the held-out prompts (plain model vs each LoReFT) are scored with the existing
security DiM vectors (data/security/security_vectors.pt, layer L) -- the target concept's score, and whether the
target is the nearest of the 10 vulnerability classes. No visualisation is produced.

Usage:  .venv/bin/python scripts/reft/train_security.py        (writes data/reft/security/)
"""
import argparse, json, os, random, sys, time
import numpy as np, torch
import pyreft
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
import common
from concept_read import positions_for, chunked_ce

PROMPT = ("Write a short technical description (4 to 6 sentences, with a brief code snippet) of a {topic} "
          "written in {language}. Write only the description itself, with no title, preamble, or commentary.")
SCHEMES = ["all", "last"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-heldout", type=int, default=3, help="write-ups held out per concept")
    ap.add_argument("--concepts", default=None, help="comma list (default: all except neutral)")
    ap.add_argument("--new-tokens", type=int, default=160)
    ap.add_argument("--out-dir", default="data/reft/security")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    L, od = args.layer, args.out_dir
    random.seed(args.seed)

    d = json.load(open("data/security/paragraphs_security.json"))
    rows, all_concepts = d["rows"], d["concepts"]
    targets = args.concepts.split(",") if args.concepts else [c for c in all_concepts if c != "neutral"]
    outs = [f"{od}/{c}_loreft_L{L}_{s}.pt" for c in targets for s in SCHEMES] + [f"{od}/training_summary.json", f"{od}/SUMMARY.md"]
    for f in outs:
        if os.path.exists(f) and not args.force:
            sys.exit(f"refusing to overwrite existing {f} (pass --force or a different --out-dir)")
    os.makedirs(od, exist_ok=True)

    # security DiM vectors (for the sanity check only)
    sv = torch.load("data/security/security_vectors.pt", weights_only=False, map_location="cpu")
    sv_concepts, vuln_idx = sv["concepts"], sv["emotional_idx"]           # 'emotional_idx' = the 10 vulnerability classes
    chat = sv["concept_hat"][L].float().numpy(); gm = sv["grand_mean"][L].float().numpy()
    del sv

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ... | security LoReFTs rank {args.rank} @ L{L} for {targets}")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    for p in model.parameters():
        p.requires_grad_(False)
    _, layers = common.get_decoder_layers(model)
    Hd = model.config.get_text_config().hidden_size
    dev = next(model.parameters()).device
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for i, layer in enumerate(layers):
        layer.gradient_checkpointing = i > L                               # never the intervened layer or below

    def render(msgs, gen):
        try: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen, enable_thinking=False)
        except TypeError: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen)
    def prompt_ids(r):
        return tok(render([{"role": "user", "content": PROMPT.format(topic=r["topic"], language=r["language"])}], True),
                   add_special_tokens=False).input_ids
    def example(r):
        p = prompt_ids(r)
        f = tok(render([{"role": "user", "content": PROMPT.format(topic=r["topic"], language=r["language"])},
                        {"role": "assistant", "content": r["text"]}], False), add_special_tokens=False).input_ids
        assert f[:len(p)] == p, "chat-template prefix mismatch"
        ids = torch.tensor([f], device=dev)
        return ids, len(p)

    # DiM sanity score of generated text: (mean-pooled resid at L, grand-mean-centred) . concept_hat
    grab = {}
    def resid_mean(ids):
        hk = layers[L].register_forward_hook(lambda _m, _i, o: grab.__setitem__("H", common.layer_output_hidden(o)[0].float().cpu()))
        try:
            with torch.no_grad():
                model(input_ids=torch.tensor([ids], device=dev))
        finally:
            hk.remove()
        return grab["H"].numpy().mean(0)
    def dim_scores(textstr):
        h = resid_mean(tok(textstr).input_ids) - gm
        return {sv_concepts[k]: float(h @ chat[k]) for k in range(len(sv_concepts))}

    by_concept = {c: [r for r in rows if r["concept"] == c] for c in targets}
    summary = {"layer": L, "rank": args.rank, "n_heldout": args.n_heldout, "prompt": PROMPT, "concepts": {}}

    def train_one(concept, scheme, train_rows, held_rows, ck_path):
        random.seed(args.seed); torch.manual_seed(args.seed)
        reft = pyreft.get_reft_model(model, pyreft.ReftConfig(representations={
            "component": f"model.language_model.layers[{L}].output", "low_rank_dimension": args.rank,
            "intervention": pyreft.LoreftIntervention(embed_dim=Hd, low_rank_dimension=args.rank)}))
        reft.set_device(dev)
        model.train()                                                      # AFTER get_reft_model (it calls model.eval())
        iv = list(reft.interventions.values())[0]; iv = iv[0] if isinstance(iv, (list, tuple)) else iv
        params = [p for p in reft.parameters() if p.requires_grad]
        tr_ex = [example(r) for r in train_rows]; he_ex = [example(r) for r in held_rows]
        def loss_of(ids, plen):
            keep = ids.shape[1] - plen + 1
            _, o = reft(base={"input_ids": ids, "attention_mask": torch.ones_like(ids), "logits_to_keep": keep},
                        unit_locations={"sources->base": (None, [[positions_for(plen, scheme)]])})
            return chunked_ce(o.logits[0, :-1], ids[0, plen:])
        def held_loss():
            with torch.no_grad():
                return sum(float(loss_of(*e)) for e in he_ex) / len(he_ex)
        steps = args.epochs * -(-len(tr_ex) // args.accum)
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)       # wd MUST be 0 (orthogonal R)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps))
        clone = lambda sd: {k: v.detach().clone() for k, v in sd.items()}
        log = [{"epoch": 0, "train": None, "held": held_loss()}]
        best = (log[0]["held"], 0, clone(iv.state_dict()))
        for ep in range(1, args.epochs + 1):
            order = list(range(len(tr_ex))); random.shuffle(order); tot = 0.0
            for i, idx in enumerate(order):
                loss = loss_of(*tr_ex[idx]); (loss / args.accum).backward(); tot += float(loss)
                if (i + 1) % args.accum == 0 or i + 1 == len(order):
                    opt.step(); sched.step(); opt.zero_grad()
            hl = held_loss(); log.append({"epoch": ep, "train": tot / len(order), "held": hl})
            if hl < best[0]: best = (hl, ep, clone(iv.state_dict()))
        model.eval()
        iv.load_state_dict(best[2])
        torch.save({"layer": L, "rank": args.rank, "positions": scheme, "concept": concept, "state_dict": best[2],
                    "log": log, "best_epoch": best[1], "best_held": best[0], "task": "component description -> vulnerable write-up",
                    "train_topics": [r["topic"] for r in train_rows], "heldout_topics": [r["topic"] for r in held_rows],
                    "args": vars(args)}, ck_path)
        # sanity generations on held-out prompts
        gens = []
        for r in held_rows:
            pi = torch.tensor([prompt_ids(r)], device=dev)
            with torch.inference_mode():
                _, g = reft.generate({"input_ids": pi, "attention_mask": torch.ones_like(pi)},
                                     unit_locations={"sources->base": (None, [[positions_for(pi.shape[1], scheme)]])},
                                     intervene_on_prompt=True, max_new_tokens=args.new_tokens, do_sample=False,
                                     eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
            gens.append(tok.decode(g[0, pi.shape[1]:], skip_special_tokens=True).strip())
        del reft
        torch.cuda.empty_cache()
        return best, log, gens

    t0 = time.time(); torch.cuda.reset_peak_memory_stats()
    for concept in targets:
        pool = by_concept[concept][:]
        random.Random(args.seed).shuffle(pool)
        held_rows = pool[:args.n_heldout]; train_rows = pool[args.n_heldout:]
        # plain-model generations + DiM scores of held-out prompts and their true targets
        plain, held_targets = [], []
        for r in held_rows:
            pi = torch.tensor([prompt_ids(r)], device=dev)
            with torch.inference_mode():
                o = model.generate(input_ids=pi, max_new_tokens=args.new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
            plain.append(tok.decode(o[0, pi.shape[1]:], skip_special_tokens=True).strip())
            held_targets.append(dim_scores(r["text"]))
        entry = {"n_train": len(train_rows), "n_held": len(held_rows), "heldout_topics": [r["topic"] for r in held_rows],
                 "target_dim_on_corpus": float(np.mean([s[concept] for s in held_targets])), "schemes": {}}
        for scheme in SCHEMES:
            best, log, gens = train_one(concept, scheme, train_rows, held_rows, f"{od}/{concept}_loreft_L{L}_{scheme}.pt")
            sc_plain = [dim_scores(t) for t in plain]; sc_reft = [dim_scores(t) for t in gens]
            near = [max(range(len(vuln_idx)), key=lambda j: s[sv_concepts[vuln_idx[j]]]) for s in sc_reft]
            tgt_j = vuln_idx.index(sv_concepts.index(concept)) if sv_concepts.index(concept) in vuln_idx else None
            entry["schemes"][scheme] = {
                "best_epoch": best[1], "best_held": round(best[2] if False else best[0], 4),
                "final_train": round(log[-1]["train"], 4),
                "target_dim_plain": round(float(np.mean([s[concept] for s in sc_plain])), 3),
                "target_dim_reft": round(float(np.mean([s[concept] for s in sc_reft])), 3),
                "nearest_is_target": None if tgt_j is None else round(float(np.mean([n == tgt_j for n in near])), 3),
                "sample": gens[0][:240]}
            m = entry["schemes"][scheme]
            print(f"[{concept:16} {scheme:4}] best ep {best[1]:>2} held {best[0]:.3f} | target-DiM plain {m['target_dim_plain']:+.2f} "
                  f"-> reft {m['target_dim_reft']:+.2f} (corpus {entry['target_dim_on_corpus']:+.2f}) | "
                  f"nearest=target {m['nearest_is_target']} | {time.time()-t0:.0f}s")
        summary["concepts"][concept] = entry

    summary["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    summary["seconds"] = time.time() - t0
    json.dump(summary, open(f"{od}/training_summary.json", "w"), ensure_ascii=False, indent=1)

    # human-readable summary
    lines = [f"# Security LoReFTs (layer {L}, rank {args.rank})", "",
             f"Trained {len(targets)} concepts x {len(SCHEMES)} position schemes on data/security/paragraphs_security.json.",
             f"Prompt: component description (no vulnerability requested) -> the concept's write-up. {args.n_heldout} held out per concept.",
             f"Peak VRAM {summary['peak_gb']:.2f} GB, {summary['seconds']:.0f}s total. wd=0, gradient checkpointing above L{L}.", "",
             "Sanity check: mean layer-%d DiM score for the TARGET concept on held-out prompts (plain model -> LoReFT; corpus value in parens)," % L,
             "and how often the LoReFT output's nearest of the 10 vulnerability classes is the target.", "",
             "| concept | scheme | best epoch | held loss | target-DiM plain | -> LoReFT | corpus | nearest=target |",
             "|---|---|---|---|---|---|---|---|"]
    for c in targets:
        e = summary["concepts"][c]
        for s in SCHEMES:
            m = e["schemes"][s]
            lines.append(f"| {c} | {s} | {m['best_epoch']} | {m['best_held']:.3f} | {m['target_dim_plain']:+.2f} | "
                         f"{m['target_dim_reft']:+.2f} | {e['target_dim_on_corpus']:+.2f} | {m['nearest_is_target']} |")
    open(f"{od}/SUMMARY.md", "w").write("\n".join(lines) + "\n")
    print(f"\nWrote {od}/  ({len(targets)*2} checkpoints + training_summary.json + SUMMARY.md) | {summary['seconds']:.0f}s | peak {summary['peak_gb']:.2f} GB")


if __name__ == "__main__":
    main()
