"""Step 1: does Qwen3.5-4B obey a 'respond only with emojis' system prompt?
Generate on a handful of prompts, with vs without the emoji system prompt, and report emoji%."""
import os, sys, torch
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import common

SYS = "You are an assistant that communicates ONLY using emojis. Never use words or letters. Reply with emojis only."
PROMPTS = [
    "What's your favorite food?",
    "How does a rainbow form?",
    "Describe a mountain hike.",
    "Tell me about dogs.",
    "How's the weather?",
    "What do you think about the ocean?",
]

def is_emoji(c):
    o = ord(c)
    return o >= 0x1F000 or 0x2600 <= o <= 0x27BF or 0x2190 <= o <= 0x21FF or o in (0x2B50, 0x2B55, 0xFE0F, 0x200D)
def efrac(t):
    cs = [c for c in t if not c.isspace()]
    return sum(is_emoji(c) for c in cs)/len(cs) if cs else 0.0

mid, kind = common.resolve("qwen35-4b")
print(f"Loading {mid} ...")
handler, model, tok, is_mm = common.load_model(mid, kind)
dev = next(model.parameters()).device

def gen(prompt, use_sys):
    msgs = ([{"role": "system", "content": SYS}] if use_sys else []) + [{"role": "user", "content": prompt}]
    kw = dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    try: enc = handler.apply_chat_template(msgs, enable_thinking=False, **kw)
    except TypeError: enc = handler.apply_chat_template(msgs, **kw)
    enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

for p in PROMPTS:
    base = gen(p, False)
    emo = gen(p, True)
    print(f"\n[{p}]")
    print(f"   no-sys  ({100*efrac(base):3.0f}% emoji): {base[:70]!r}")
    print(f"   +sys    ({100*efrac(emo):3.0f}% emoji): {emo[:70]!r}")
