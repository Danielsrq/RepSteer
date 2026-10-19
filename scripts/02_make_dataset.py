"""Generate a matched happy/sad paragraph dataset with Qwen3.5-4B.

Same 10 topics, each written once in a joyful tone and once in a sad tone, so
emotion is the only thing that differs between the pair (topic is controlled).
Saves data/paragraphs.json as a list of {id, topic, emotion, text}.

Usage:
    python scripts/02_make_dataset.py                 # 10 topics x {happy, sad} = 20
    python scripts/02_make_dataset.py --preset qwen3-text
"""

import argparse
import json
import os

import torch

import common

TOPICS = [
    # 0-9: original set (keep first so n=10 stays reproducible / nested in n=30)
    "a walk through a city park",
    "receiving a handwritten letter",
    "a family gathered around the dinner table",
    "the last day of the school year",
    "finding an old photograph in a drawer",
    "a long train journey through the countryside",
    "a rainy afternoon spent indoors",
    "running into an old friend on the street",
    "standing on the shore watching the ocean at dawn",
    "a small apartment at the end of the day",
    # 10-29: extension for the n=30 study
    "a birthday celebration",
    "a walk in the woods in autumn",
    "a phone call from a distant relative",
    "cleaning out an old attic",
    "a cup of coffee in a quiet cafe",
    "the first snowfall of winter",
    "a garden in early spring",
    "waiting at an airport departure gate",
    "an empty playground at dusk",
    "a wedding ceremony",
    "moving into a new house",
    "a hospital waiting room",
    "a road trip with old friends",
    "watching the sunset from a rooftop",
    "revisiting a childhood bedroom",
    "a market on a busy morning",
    "a boat drifting on a calm lake",
    "the last dance at a party",
    "a letter left unopened on a table",
    "a park bench in the rain",
]

TONE = {
    "happy": "joyful, warm, and uplifting",
    "sad": "sad, melancholic, and sorrowful",
}


def gen_paragraph(model, handler, tok, topic, emotion, device):
    user = (
        f"Write a short, vivid paragraph (4 to 6 sentences) about {topic}. "
        f"The emotional tone must be {TONE[emotion]}. "
        f"Write only the paragraph itself, with no title, preamble, or commentary."
    )
    messages = [{"role": "user", "content": user}]
    kw = dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    try:
        enc = handler.apply_chat_template(messages, enable_thinking=False, **kw)
    except TypeError:
        enc = handler.apply_chat_template(messages, **kw)
    enc = {k: v.to(device) for k, v in enc.items()}
    n = enc["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(
            **enc, max_new_tokens=220, do_sample=True, temperature=0.9,
            top_p=0.95, pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0, n:], skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(common.PRESETS), default="qwen35-4b")
    ap.add_argument("--model", default=None)
    ap.add_argument("--kind", choices=["text", "multimodal"], default=None)
    ap.add_argument("--out", default="data/paragraphs.json")
    ap.add_argument("--start", type=int, default=0, help="first topic index (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="last topic index (exclusive)")
    args = ap.parse_args()

    topics = TOPICS[args.start:args.end]
    model_id, kind = common.resolve(args.preset, args.model, args.kind)
    print(f"Loading {model_id} (kind={kind}) ... | {len(topics)} topics [{args.start}:{args.end}]")
    handler, model, tok, is_mm = common.load_model(model_id, kind)

    rows = []
    i = 0
    for topic in topics:
        for emotion in ("happy", "sad"):
            text = gen_paragraph(model, handler, tok, topic, emotion, "cuda")
            rows.append({"id": i, "topic": topic, "emotion": emotion, "text": text})
            print(f"[{i:2d}] {emotion:5s} | {topic}\n     {text[:110]}...")
            i += 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"model": model_id, "rows": rows}, f, indent=2)
    n_h = sum(r["emotion"] == "happy" for r in rows)
    n_s = sum(r["emotion"] == "sad" for r in rows)
    print(f"\nSaved {len(rows)} paragraphs ({n_h} happy, {n_s} sad) -> {args.out}")


if __name__ == "__main__":
    main()
