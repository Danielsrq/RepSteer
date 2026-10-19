"""Capture streaming layer-L residuals for the v2 (5-label) corpus.

One GPU pass that everything downstream reuses: DiM, LDA and the LoReFT trainer all read this cache.
Residuals are captured exactly the way the viewer reads an article and the way the LoReFT will be
trained -- continuous pages of `page_size` display tokens with a `prefix`-token run-up, so each
kept token's activation is a function of its real preceding article context. Every token is tagged
with its v2 span label; a token is ONE datapoint in one of the 5 clouds.

  .venv/bin/python scripts/reft/v2_cache.py            # -> data/security/v2_resid_L17.npz
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from span_labels import segment
from span_labels_v2 import LABELS, CONCEPTS, HOLDOUT, ARTICLE_CONCEPT, load

OUT = "data/security/v2_resid_L{L}.npz"


def token_labels(article, tok):
    """Per-token v2 label for a whole article, by mapping each labelled span to its char range."""
    text = open(f"data/jlens/cve/{article}.txt", encoding="utf-8").read()
    spans, lab = segment(text), load(article)
    char = np.array(["none"] * len(text), dtype=object)
    pos = 0
    for s in spans:
        i = text.find(s["text"][:80], pos)
        if i < 0:
            continue
        char[i:i + len(s["text"])] = lab.get(s["i"], "none")
        pos = i + len(s["text"])
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    out = []
    for a, b in enc["offset_mapping"]:
        seg = char[a:b]
        out.append(seg[0] if len(seg) and seg[0] != "none" else (seg[-1] if len(seg) else "none"))
    return enc["input_ids"], np.array(out, dtype=object)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--page-size", type=int, default=2000)
    ap.add_argument("--prefix", type=int, default=512)
    ap.add_argument("--neg-cap", type=int, default=400,
                    help="background/triage tokens kept per page; ALL concept tokens are kept")
    args = ap.parse_args()
    import torch
    arts = sorted(ARTICLE_CONCEPT)
    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} for v2 streaming capture @ L{args.layer} "
          f"(page {args.page_size} + prefix {args.prefix}) ...")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    _, layers = common.get_decoder_layers(model)
    dev = next(model.parameters()).device
    rng = np.random.default_rng(0)
    grab = {}
    hk = layers[args.layer].register_forward_hook(
        lambda _m, _i, o: grab.__setitem__("H", common.layer_output_hidden(o)[0].float().cpu()))
    Hs, off, lab, art, spl = [], [0], [], [], []
    try:
        for a in arts:
            ids, tl = token_labels(a, tok)
            npages = (len(ids) + args.page_size - 1) // args.page_size
            kept = 0
            for p in range(npages):
                start = p * args.page_size
                ctx0 = max(0, start - args.prefix)
                with torch.no_grad():
                    model(input_ids=torch.tensor([ids[ctx0:start + args.page_size]], device=dev))
                H = grab["H"][start - ctx0:].numpy()          # drop the prefix; keep display tokens
                H = H - H.mean(0)                             # page-mean-centre, as the viewer does
                seg = tl[start:start + len(H)]
                conc = np.array([x in CONCEPTS for x in seg])
                other = np.where(~conc & (seg != "none"))[0]
                take = rng.choice(other, min(len(other), args.neg_cap), replace=False) if len(other) else []
                sel = np.sort(np.concatenate([np.where(conc)[0], np.asarray(take, int)]))
                if not len(sel):
                    continue
                Hs.append(H[sel].astype(np.float16)); off.append(off[-1] + len(sel))
                lab += list(seg[sel]); art += [a] * len(sel)
                spl += ["holdout" if a in HOLDOUT else "train"] * len(sel)
                kept += len(sel)
            print(f"  {a:<20} {len(ids):>6} tok, {npages:>2} pages -> kept {kept}")
    finally:
        hk.remove()
    out = OUT.format(L=args.layer)
    np.savez_compressed(out, H=np.concatenate(Hs), off=np.array(off), label=np.array(lab),
                        article=np.array(art), split=np.array(spl))
    import collections
    print(f"\nWrote {out}: {off[-1]} tokens (one row = one token)")
    print("cloud sizes:", dict(collections.Counter(lab)))


if __name__ == "__main__":
    main()
