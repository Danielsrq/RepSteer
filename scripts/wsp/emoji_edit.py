"""Weight-space programming: make Qwen3.5-4B reply in emoji by editing weights only.

No gradients, no training loop, no LoReFT. The edit is authored from structure:

  1. Qwen3.5 tokenises pictographic emoji as 3-byte BPE sequences. Across the 209 distinct
     emoji the model itself produced under the "only emojis" system prompt, there are just
     12 sequence-ENTRY tokens, and one of them (id 9008 = UTF-8 lead byte \\xf0\\x9f) opens
     183 of the 209. The emoji register is gated by ~one vocabulary row.
  2. lm_head has no bias, so a constant logit bias cannot be written directly. But the final
     residuals all sit in one cone: cos(h, h_hat) > 0 for 100% of tokens (mean 0.46,
     ||h|| ~ 70). So adding beta * h_hat to an lm_head ROW raises that token's logit by
     beta * <h, h_hat> ~ 32 * beta -- a reliable positive bias implemented purely in weights.
  3. Boost only the ENTRY tokens; the model's own dynamics complete each byte sequence
     (boosting every byte fragment is what made byte-DiM degenerate into U+FFFD).

lm_head is TIED to the input embeddings in this config, so it is untied (cloned) first --
otherwise the same edit would corrupt how emoji tokens are read back in during generation.

  .venv/bin/python scripts/wsp/emoji_edit.py --sweep
  .venv/bin/python scripts/wsp/emoji_edit.py --beta 2.0 --save
"""
import argparse, json, os, sys
import torch, torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "emoji"))
from train_loreft import HELD, FRESH, ef, diversity, glyphs, clean

PATCH = "data/wsp/emoji_wsp_patch.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=2.0, help="logit-bias scale (in units of h_hat)")
    ap.add_argument("--sweep", action="store_true", help="sweep beta on 3 prompts then stop")
    ap.add_argument("--betas", default="0.5,1.0,1.5,2.0,3.0,4.0")
    ap.add_argument("--top-entry", type=int, default=12, help="how many entry tokens to boost")
    ap.add_argument("--ids", default=None,
                    help="explicit comma list of token ids to boost (overrides --top-entry); "
                         "9008 is the pictograph lead byte that opens 183/209 emoji")
    ap.add_argument("--new-tokens", type=int, default=40)
    ap.add_argument("--save", action="store_true", help="write the weight patch")
    ap.add_argument("--out", default="data/wsp/emoji_wsp.json")
    args = ap.parse_args()

    pr = torch.load("data/wsp/probe.pt", weights_only=False, map_location="cpu")
    etk = json.load(open("data/wsp/emoji_tokens.json"))
    if args.ids:
        entry = [int(x) for x in args.ids.split(",")]
    else:
        entry = etk["entry_ids"][: args.top_entry] if args.top_entry else etk["entry_ids"]

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    model.eval()
    dev = next(model.parameters()).device
    head = model.lm_head
    emb = model.get_input_embeddings()

    # --- untie lm_head from the input embeddings before touching it ---
    if head.weight.data_ptr() == emb.weight.data_ptr():
        print("lm_head is tied -> cloning so the edit does not alter input embeddings")
        head.weight = nn.Parameter(head.weight.detach().clone(), requires_grad=False)
        model.config.get_text_config().tie_word_embeddings = False
        assert head.weight.data_ptr() != emb.weight.data_ptr()
    orig = head.weight.data[entry].clone()
    h_hat = pr["h_hat"].to(dev, head.weight.dtype)
    print(f"boosting {len(entry)} entry rows: {entry}")
    print(f"expected logit bias ~ beta * <h,h_hat> ~ {pr['h_norm_mean']*0.458:.1f} * beta")

    def render(q):
        msgs = [{"role": "user", "content": q}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def gen(q):
        ids = torch.tensor([tok(render(q), add_special_tokens=False).input_ids], device=dev)
        with torch.inference_mode():
            o = model.generate(input_ids=ids, max_new_tokens=args.new_tokens, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        return clean(tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip())

    def apply_beta(b):
        head.weight.data[entry] = orig + b * h_hat          # the entire edit
        return len(entry) * head.weight.shape[1]

    if args.sweep:
        probe_q = HELD[:3]
        print("\nbeta |  emoji%  diversity   sample")
        for b in [float(x) for x in args.betas.split(",")]:
            n = apply_beta(b)
            outs = [gen(q) for q in probe_q]
            e = sum(ef(o) for o in outs) / len(outs)
            d = sum(diversity(o) for o in outs) / len(outs)
            print(f"{b:4.1f} |  {100*e:5.1f}%   {d:5.2f}     {outs[0][:60]!r}")
        head.weight.data[entry] = orig
        return

    n_edited = apply_beta(args.beta)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nedited {n_edited} params of {total} ({100*n_edited/total:.6f}%)\n")
    rows = []
    for q in HELD + FRESH:
        t = gen(q)
        rows.append({"prompt": q, "text": t, "emoji": ef(t), "diversity": diversity(t),
                     "held": q in HELD})
        print(f"  [{100*ef(t):3.0f}% emoji] {q}\n      {t}")
    me = sum(r["emoji"] for r in rows) / len(rows)
    md = sum(r["diversity"] for r in rows) / len(rows)
    print(f"\nMEAN emoji {100*me:.1f}%  diversity {md:.2f}  score {me*md:.2f}")

    os.makedirs("data/wsp", exist_ok=True)
    json.dump({"beta": args.beta, "entry_ids": entry, "n_edited": n_edited, "total_params": total,
               "frac_edited": n_edited / total, "mean_emoji": me, "mean_diversity": md,
               "rows": rows}, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"saved {args.out}")

    if args.save:
        if os.path.exists(PATCH):
            sys.exit(f"refusing to overwrite {PATCH}")
        torch.save({"method": "wsp-lm_head-entry-bias", "model": mid, "beta": args.beta,
                    "entry_ids": entry, "h_hat": pr["h_hat"],
                    "rows_new": head.weight.data[entry].cpu().clone(),
                    "rows_orig": orig.cpu(), "untied_lm_head": True,
                    "note": "apply: untie lm_head, then lm_head.weight[entry_ids] = rows_new"},
                   PATCH)
        print(f"saved weight patch {PATCH}")


if __name__ == "__main__":
    main()
