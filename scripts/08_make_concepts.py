"""Generate a multi-concept paragraph dataset with Qwen3.5-4B.

10 concepts x 30 topics = 300 paragraphs, each topic written once per concept so
topic is balanced across concepts (it averages out under grand-mean centering, in
09). Concept is the dominant axis of variation. Rows are grouped by concept, so
`id` is contiguous per concept and `concept_idx == id // len(TOPICS)`.

Self-contained on purpose (digit-prefixed sibling modules like 02 can't be
imported); the TOPICS list is intentionally the same 30 used by 02_make_dataset.py.

Usage:
    python scripts/08_make_concepts.py                       # all 10 concepts
    python scripts/08_make_concepts.py --start 0 --end 5     # first 5 concepts (chunked)
"""

import argparse
import json
import os

import torch

import common

# (name, tone/character phrase) -- index order is fixed 0..9.
# happy/sad phrasing kept identical to 02_make_dataset.py for comparability.
CONCEPTS = [
    ("happy",       "joyful, warm, and uplifting"),
    ("sad",         "sad, melancholic, and sorrowful"),
    ("angry",       "angry, seething, and indignant"),
    ("excited",     "excited, exuberant, and eager"),
    ("calm",        "calm, serene, and untroubled"),
    ("nonchalant",  "nonchalant, indifferent, and casually unbothered"),
    ("skeptical",   "skeptical, doubtful, and questioning"),
    ("belligerent", "belligerent, hostile, and confrontational"),
    ("scheming",    "scheming, calculating, and manipulative"),
    ("paranoid",    "paranoid, distrustful, and fearful of hidden threats"),
    ("humorous",    "humorous, witty, and playfully comedic"),
    ("neutral",     None),   # baseline: tone sentence omitted -> model's default register
]

# same 30 topics as 02_make_dataset.py (topic controlled across concepts)
TOPICS = [
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


def gen_paragraph(model, handler, tok, topic, tone, device):
    tone_line = f"The emotional tone must be {tone}. " if tone else ""   # tone=None -> neutral baseline
    user = (
        f"Write a short, vivid paragraph (4 to 6 sentences) about {topic}. "
        f"{tone_line}"
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
    ap.add_argument("--out", default="data/concepts/paragraphs_concepts.json")
    ap.add_argument("--start", type=int, default=0, help="first concept index (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="last concept index (exclusive)")
    args = ap.parse_args()

    concepts = CONCEPTS[args.start:args.end]
    model_id, kind = common.resolve(args.preset, args.model, args.kind)
    print(f"Loading {model_id} (kind={kind}) ... | {len(concepts)} concepts x {len(TOPICS)} topics")
    handler, model, tok, is_mm = common.load_model(model_id, kind)

    rows = []
    for ci, (name, tone) in enumerate(concepts, start=args.start):
        for ti, topic in enumerate(TOPICS):
            text = gen_paragraph(model, handler, tok, topic, tone, "cuda")
            rid = ci * len(TOPICS) + ti          # absolute, contiguous per concept
            rows.append({"id": rid, "concept": name, "concept_idx": ci, "topic": topic, "text": text})
            print(f"[{rid:3d}] {name:12s} | {topic}\n     {text[:100]}...")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # merge with any existing file (by id) so chunked/resumed runs accumulate
    merged = {}
    if os.path.exists(args.out):
        for r in json.load(open(args.out)).get("rows", []):
            merged[r["id"]] = r
    for r in rows:
        merged[r["id"]] = r
    all_rows = [merged[k] for k in sorted(merged)]
    with open(args.out, "w") as f:
        json.dump({"model": model_id, "concepts": [c[0] for c in CONCEPTS], "rows": all_rows}, f, indent=2)
    from collections import Counter
    print(f"\nSaved {len(all_rows)} paragraphs -> {args.out}")
    print("  per concept:", dict(Counter(r["concept"] for r in all_rows)))


if __name__ == "__main__":
    main()
