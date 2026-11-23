"""Evaluate the conditional LoReFT on the same four sets as the hand-built weight circuit.

Comparable numbers (circuit, L26, s=5 a=0.5):
    animal_test 91.4% | other_test 0.0% | no_noun_2 90.1% | word_not_2 0.3% | unique 1.5-2.6
    ceiling (system prompt): unique ~11

  .venv/bin/python scripts/wsp/eval_cond_loreft.py --loreft data/reft/mine/emoji_cond_L17.pt
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "emoji"))
from train_loreft import ef, diversity, clean, glyphs


def positions_for(n, scheme):
    return [n - 1] if scheme == "last" else list(range(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loreft", default="data/reft/mine/emoji_cond_L17.pt")
    ap.add_argument("--new-tokens", type=int, default=40)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="data/wsp/eval_cond_loreft.json")
    args = ap.parse_args()

    P = json.load(open("data/wsp/cond_prompts.json"))
    H2 = json.load(open("data/wsp/lexical_control_heldout.json"))
    import pyreft
    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    model.eval()
    dev = next(model.parameters()).device
    ck = torch.load(args.loreft, weights_only=False, map_location="cpu")
    L, scheme = ck["layer"], ck.get("positions", "all")
    Hd = model.config.get_text_config().hidden_size
    reft = pyreft.get_reft_model(model, pyreft.ReftConfig(representations={
        "component": f"model.language_model.layers[{L}].output",
        "low_rank_dimension": ck["rank"],
        "intervention": pyreft.LoreftIntervention(embed_dim=Hd, low_rank_dimension=ck["rank"])}))
    reft.set_device(dev)
    iv = list(reft.interventions.values())[0]
    iv = iv[0] if isinstance(iv, (list, tuple)) else iv
    iv.load_state_dict(ck["state_dict"])
    print(f"LoReFT L{L} rank {ck['rank']} positions={scheme}")

    def gen(q):
        msgs = [{"role": "user", "content": q}]
        try:
            txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False)
        except TypeError:
            txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = torch.tensor([tok(txt, add_special_tokens=False).input_ids], device=dev)
        with torch.inference_mode():
            _, g = reft.generate({"input_ids": ids, "attention_mask": torch.ones_like(ids)},
                                 unit_locations={"sources->base": (None, [[positions_for(ids.shape[1], scheme)]])},
                                 intervene_on_prompt=True, max_new_tokens=args.new_tokens,
                                 do_sample=False, eos_token_id=tok.eos_token_id,
                                 pad_token_id=tok.eos_token_id)
        return clean(tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True))

    SETS = [("animal_test        (want EMOJI)", P["animal_test"][:args.n]),
            ("other_test         (want PROSE)", P["other_test"][:args.n]),
            ("no_noun_2    HARD  (want EMOJI)", H2["animal_no_noun_2"][:args.n]),
            ("word_not_2   HARD  (want PROSE)", H2["animal_word_not_animal_2"][:args.n])]
    out = {}
    print(f"\n{'set':<34}{'emoji%':>8}{'uniq':>7}{'nglyph':>8}")
    for name, qs in SETS:
        rows = [(q, gen(q)) for q in qs]
        gl = [glyphs(t) for _, t in rows]
        e = sum(ef(t) for _, t in rows) / len(rows)
        u = sum(len(set(x)) for x in gl) / len(gl)
        n = sum(len(x) for x in gl) / len(gl)
        out[name] = {"emoji": e, "uniq": u, "nglyph": n,
                     "rows": [{"prompt": q, "text": t} for q, t in rows]}
        print(f"{name:<34}{100*e:>7.1f}%{u:>7.1f}{n:>8.1f}")
    print("\n--- animal_test samples ---")
    for r in out["animal_test        (want EMOJI)"]["rows"][:6]:
        print(f"  {r['prompt'][:44]:<44} {r['text'][:46]}")
    print("\n--- word_not_2 (decoys, want prose) samples ---")
    for r in out["word_not_2   HARD  (want PROSE)"]["rows"][:6]:
        print(f"  {r['prompt'][:44]:<44} {r['text'][:46]}")
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
