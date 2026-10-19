"""Generate + store the model's OWN emoji-only answers under the 'respond only with emojis'
system prompt. Saves data/emoji/emoji_generated.json = [{prompt, answer, emoji_frac, n_tok}].

This is the real material we lacked: emoji responses the model actually produces (vs the
hand-written emoji_pairs.json that fed the failed byte-DiM). Usable later to build a
difference-of-means vector from the model's own generated-emoji activations.
"""
import json, os, sys
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import common

SYS = "You are an assistant that communicates ONLY using emojis. Never use words or letters. Reply with emojis only."
PROMPTS = [
    # 24 build prompts
    "What's your favorite food?", "How's the weather today?", "Explain why the sky is blue.",
    "Tell me about your day.", "How do I make coffee?", "Describe a cat.",
    "Recommend a movie.", "Plan a birthday party.", "Help me relax.", "Tell me a joke.",
    "How do plants grow?", "Describe a busy city.", "How do I stay healthy?", "What's space like?",
    "How do computers work?", "Describe a rainstorm.", "What's the moon made of?",
    "Tell me about the desert.", "How do I write a story?", "Describe autumn leaves.",
    "What makes a good friend?", "How do birds fly?", "Describe the sunrise.", "What is music?",
    # 5 held-out prompts
    "How does a rainbow form?", "Give me advice for a job interview.", "Describe a mountain hike.",
    "Should I buy an electric car?", "Tell me about dogs.",
    # extra variety
    "What's the ocean like?", "Describe a forest.", "How do I bake bread?", "Tell me about winter.",
    "What's a good vacation?", "Describe a party.", "How do volcanoes work?", "What's a farm like?",
    "Describe a city at night.", "How do I grow a garden?", "What's a wedding like?", "Describe a storm at sea.",
]

def is_emoji(c):
    o = ord(c)
    return o >= 0x1F000 or 0x2600 <= o <= 0x27BF or 0x2190 <= o <= 0x21FF or o in (0x2B50, 0x2B55, 0xFE0F, 0x200D)
def efrac(t):
    cs = [c for c in t if not c.isspace()]
    return sum(is_emoji(c) for c in cs) / len(cs) if cs else 0.0

def main():
    out = "data/emoji/emoji_generated.json"
    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ... | {len(PROMPTS)} prompts")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    dev = next(model.parameters()).device

    def gen(prompt):
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": prompt}]
        kw = dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        try: enc = handler.apply_chat_template(msgs, enable_thinking=False, **kw)
        except TypeError: enc = handler.apply_chat_template(msgs, **kw)
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.inference_mode():
            o = model.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
        gen_ids = o[0, enc["input_ids"].shape[1]:]
        return tok.decode(gen_ids, skip_special_tokens=True).strip(), int((gen_ids != tok.eos_token_id).sum())

    data = []
    for i, p in enumerate(PROMPTS):
        ans, ntok = gen(p)
        f = efrac(ans)
        data.append({"prompt": p, "answer": ans, "emoji_frac": round(f, 3), "n_tok": ntok})
        print(f"  {i+1:>2}/{len(PROMPTS)}  {100*f:3.0f}%  {p[:30]:30}  {ans[:40]!r}")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(data, open(out, "w"), ensure_ascii=False, indent=1)
    fr = [d["emoji_frac"] for d in data]
    hi = sum(f >= 0.8 for f in fr)
    print(f"\nSaved {len(data)} -> {out}")
    print(f"emoji-fraction: mean {sum(fr)/len(fr):.2f} | ≥80% emoji: {hi}/{len(data)} | min {min(fr):.2f} max {max(fr):.2f}")

if __name__ == "__main__":
    main()
