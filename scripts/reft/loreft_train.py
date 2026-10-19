"""Generic LoReFT trainer (write objective) — bring your own dataset.

Canonical LoReFT: train a low-rank intervention so that, applied during the forward pass, it makes
the model PRODUCE a target string. This is the steering objective — the only one LoReFT really has.
Reading is done afterwards, objective-independently, by SVD of V = W - R into q_i (read/sensor) and
p_i (write/register) directions.

Dataset JSON (a bare list, or {"examples": [...]}):
  [{"context": "Write a haiku about the sea.", "target": "Waves crash on the shore ..."}, ...]

The intervention acts on the CONTEXT (prompt) positions; the loss is next-token CE over the target.
The target tokens are teacher-forced and are NOT intervened on. (A discriminative reader is just a
special case: make the target a single "yes"/"no" token and include negative examples.)

Positions the intervention acts on (--positions):
  last   only the final context token (cheapest; the prefill state that seeds generation)
  all    every context token

weight_decay is hard-wired to 0 (AdamW decay silently zeroes the orthogonal R).

Examples:
  .venv/bin/python scripts/reft/loreft_train.py --data mydata.json --out data/reft/mine/steer.pt
  .venv/bin/python scripts/reft/loreft_train.py --data mydata.json --positions last --rank 8 \
        --layer 17 --out data/reft/mine/steer.pt
"""
import argparse, json, os, random, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
import pyreft


def positions_for(n, scheme):
    return [n - 1] if scheme == "last" else list(range(n))


def chunked_ce(logits, targets, chunk=64):
    tot = torch.zeros((), device=logits.device, dtype=torch.float32)
    for s in range(0, len(targets), chunk):
        tot = tot + torch.utils.checkpoint.checkpoint(
            lambda x, y: torch.nn.functional.cross_entropy(x.float(), y, reduction="sum"),
            logits[s:s + chunk], targets[s:s + chunk], use_reentrant=False)
    return tot / max(len(targets), 1)


def load_dataset(path):
    d = json.load(open(path))
    return d["examples"] if isinstance(d, dict) and "examples" in d else d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset JSON of {context, target} (see docstring)")
    ap.add_argument("--positions", default="all", choices=["last", "all"])
    ap.add_argument("--system", default=None,
                    help="a system-role message prepended to every example (chat template only). "
                         "NOTE: if the system prompt already induces the target behavior, the model "
                         "complies with no help, the loss is ~0, and the intervention learns nothing "
                         "-- for replacing a prompt, LEAVE THIS OFF and train on the bare context.")
    ap.add_argument("--preset", default="qwen35-4b")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ctx-tok", type=int, default=512)
    ap.add_argument("--tgt-tok", type=int, default=512)
    ap.add_argument("--heldout-frac", type=float, default=0.15, help="random fraction for early stopping")
    ap.add_argument("--chat-template", dest="chat", action="store_true", default=True,
                    help="wrap context as a chat user turn / target as the assistant turn (default on)")
    ap.add_argument("--no-chat-template", dest="chat", action="store_false",
                    help="raw continuation: target is appended straight after context")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.system and not args.chat:
        sys.exit("--system needs the chat template; do not combine with --no-chat-template")
    if os.path.exists(args.out) and not args.force:
        sys.exit(f"refusing to overwrite {args.out} (pass --force)")

    data = load_dataset(args.data)
    for e in data:
        if "context" not in e or "target" not in e:
            sys.exit('every example needs "context" and "target" keys')
    print(f"{len(data)} examples | positions={args.positions} | layer={args.layer} rank={args.rank}")

    mid, kind = common.resolve(args.preset)
    print(f"Loading {mid} ...")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    for p in model.parameters():
        p.requires_grad_(False)
    _, layers = common.get_decoder_layers(model)
    Hd = model.config.get_text_config().hidden_size
    dev = next(model.parameters()).device
    L = args.layer
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        for i, layer in enumerate(layers):
            layer.gradient_checkpointing = i > L

    def render(msgs, gen):
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen)

    def build(ex):
        """Return (ids[1,T], prompt_len, intervene_positions). Intervention acts on the prompt."""
        ctx_ids = tok(ex["context"], add_special_tokens=False).input_ids[-args.ctx_tok:]
        ctx = tok.decode(ctx_ids)
        tgt_ids = tok(ex["target"], add_special_tokens=False).input_ids[:args.tgt_tok]
        tgt = tok.decode(tgt_ids)
        if args.chat:
            sysm = [{"role": "system", "content": args.system}] if args.system else []
            p = tok(render(sysm + [{"role": "user", "content": ctx}], True), add_special_tokens=False).input_ids
            f = tok(render(sysm + [{"role": "user", "content": ctx}, {"role": "assistant", "content": tgt}], False),
                    add_special_tokens=False).input_ids
        else:
            p = tok(ctx, add_special_tokens=False).input_ids
            f = tok(ctx + tgt, add_special_tokens=False).input_ids
        assert f[:len(p)] == p, "prompt/target tokenizer boundary merged; toggle --no-chat-template"
        return torch.tensor([f], device=dev), len(p), positions_for(len(p), args.positions)

    exs = [build(e) for e in data]
    random.seed(args.seed); torch.manual_seed(args.seed)
    idx = list(range(len(exs))); random.shuffle(idx)
    nheld = max(1, int(args.heldout_frac * len(exs)))
    held = [exs[i] for i in idx[:nheld]]; fit = [exs[i] for i in idx[nheld:]] or exs
    print(f"{len(fit)} fit / {len(held)} held-out")

    reft = pyreft.get_reft_model(model, pyreft.ReftConfig(representations={
        "component": f"model.language_model.layers[{L}].output", "low_rank_dimension": args.rank,
        "intervention": pyreft.LoreftIntervention(embed_dim=Hd, low_rank_dimension=args.rank)}))
    reft.set_device(dev)
    model.train()                                                # AFTER get_reft_model (it calls eval)
    iv = list(reft.interventions.values())[0]; iv = iv[0] if isinstance(iv, (list, tuple)) else iv
    params = [p for p in reft.parameters() if p.requires_grad]

    def loss_of(ids, plen, pos):
        keep = ids.shape[1] - plen + 1
        _, o = reft(base={"input_ids": ids, "attention_mask": torch.ones_like(ids), "logits_to_keep": keep},
                    unit_locations={"sources->base": (None, [[pos]])})
        return chunked_ce(o.logits[0, :-1], ids[0, plen:])

    def held_loss():
        with torch.no_grad():
            return sum(float(loss_of(*e)) for e in held) / len(held)

    steps = args.epochs * -(-len(fit) // args.accum)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)     # wd MUST be 0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps))
    clone = lambda sd: {k: v.detach().clone() for k, v in sd.items()}
    best = (held_loss(), 0, clone(iv.state_dict()))
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    for ep in range(1, args.epochs + 1):
        order = list(range(len(fit))); random.shuffle(order); tot = 0.0
        for i, j in enumerate(order):
            l = loss_of(*fit[j]); (l / args.accum).backward(); tot += float(l)
            if (i + 1) % args.accum == 0 or i + 1 == len(order):
                opt.step(); sched.step(); opt.zero_grad()
        hl = held_loss()
        if hl < best[0]:
            best = (hl, ep, clone(iv.state_dict()))
        print(f"  ep {ep:>2} | train {tot/len(order):.3f} | held {hl:.3f} | {time.time()-t0:.0f}s")
    model.eval(); iv.load_state_dict(best[2])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"layer": L, "rank": args.rank, "positions": args.positions, "objective": "write",
                "system": args.system,
                "state_dict": best[2], "best_epoch": best[1], "best_held": best[0],
                "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "n": len(exs), "args": vars(args)}, args.out)
    print(f"best epoch {best[1]} (held {best[0]:.3f}) | peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB "
          f"-> {args.out}")
    print("read it back with SVD of V = W - R (rotate_layer -> R, weight -> W): "
          "U,S,Qt = svd(W-R); q_i = rows of Qt (read), p_i = cols of R^T U (write).")


if __name__ == "__main__":
    main()
