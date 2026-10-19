"""Cache layer-L residuals for the real CVE regions once, then probe them many ways.

Training a LoReFT costs minutes; fitting a probe on cached residuals costs milliseconds. So the
residuals are computed once here and every comparison -- DiM, LoReFT read/write, random control,
different negative pools, different splits -- runs off that cache.

The question this is built to answer: a probe that scores AUC 0.82 leave-one-article-out but 0.46
on a held-out article is not simply "overfitting". It may be detecting the wrong thing. If the
negatives in the training articles are ordinary prose while the positives are memory-corruption
text, a probe can score well by detecting "memory-corruption-flavoured text" and never learn the
specific mechanism. The HARD negative pools below test exactly that: same-article negatives, and
negatives drawn only from `exploit` (which is also memory-corruption text) or from OTHER
mechanism classes.

Usage:
  .venv/bin/python scripts/reft/cve_probe.py --cache          # one forward pass over every region
  .venv/bin/python scripts/reft/cve_probe.py                  # analyse (uses the cache)
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from span_labels import MECHANISM
from cve_regions import NEGATIVE

CACHE = "data/security/cve_resid_L{L}.npz"
CACHE_STREAM = "data/security/cve_resid_stream_L{L}.npz"


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def lda_fit(Xp, Xn, shrink=1e-2):
    mp, mn = Xp.mean(0), Xn.mean(0)
    Xc = np.concatenate([Xp - mp, Xn - mn]); S = Xc.T @ Xc / len(Xc)
    S = S + shrink * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
    return np.linalg.solve(S, mp - mn)


def token_labels(article, tok, labels_dir="data/security/span_labels"):
    """Per-token label for a whole article, by mapping each labelled span to its char range.
    Needed for the streaming cache: in stream mode a token's label comes from where it sits in
    the article, not from which region object it was copied into."""
    from span_labels import segment, load_labels
    text = open(f"data/jlens/cve/{article}.txt", encoding="utf-8").read()
    spans, labels = segment(text), load_labels(article, labels_dir)
    char = np.array(["none"] * len(text), dtype=object)
    pos = 0
    for s in spans:
        i = text.find(s["text"][:80], pos)
        if i < 0:
            continue
        char[i:i + len(s["text"])] = labels.get(s["i"], "none")
        pos = i + len(s["text"])
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    out = []
    for a, b in enc["offset_mapping"]:
        seg = char[a:b]
        out.append(seg[0] if len(seg) and seg[0] != "none" else (seg[-1] if len(seg) else "none"))
    return enc["input_ids"], np.array(out, dtype=object)


def do_cache_stream(args):
    """Capture residuals the way a reader actually sees the article: continuous pages with a
    prefix, page-mean-centred, exactly as scripts/reft/security_cve.py paginates. Region-isolated
    residuals (do_cache) turned out to be a DIFFERENT distribution -- a direction fitted on them
    scores at chance on the streaming page -- so this is the cache that matches the visualisation."""
    import torch
    import common
    from cve_regions import HOLDOUT
    d = json.load(open(args.data))
    arts = sorted({r["article"] for r in d["records"]})
    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} for STREAM residual capture at L{args.layer} ...")
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
            keep_a = 0
            for p in range(npages):
                start = p * args.page_size
                ctx0 = max(0, start - args.prefix)
                with torch.no_grad():
                    model(input_ids=torch.tensor([ids[ctx0:start + args.page_size]], device=dev))
                H = grab["H"][start - ctx0:].numpy()
                H = H - H.mean(0)                      # page-mean-centre, as the page does
                seg = tl[start:start + len(H)]
                # keep every mechanism token (they are ~5%), subsample the rest
                mech_m = np.array([x in MECHANISM for x in seg])
                other = np.where(~mech_m & (seg != "none"))[0]
                take = rng.choice(other, min(len(other), args.neg_cap), replace=False) if len(other) else []
                sel = np.sort(np.concatenate([np.where(mech_m)[0], np.asarray(take, int)]))
                if not len(sel):
                    continue
                Hs.append(H[sel].astype(np.float16)); off.append(off[-1] + len(sel))
                lab += list(seg[sel]); art += [a] * len(sel)
                spl += ["holdout" if a in HOLDOUT else "train"] * len(sel)
                keep_a += len(sel)
            print(f"  {a:<20} {len(ids):>6} tok, {npages:>2} pages -> kept {keep_a}")
    finally:
        hk.remove()
    out = CACHE_STREAM.format(L=args.layer)
    np.savez_compressed(out, H=np.concatenate(Hs), off=np.array(off), label=np.array(lab),
                        article=np.array(art), split=np.array(spl))
    print(f"Wrote {out}: {off[-1]} tokens (each row is ONE token, not a region)")


def do_cache(args):
    import torch
    import common
    d = json.load(open(args.data))
    recs = d["records"]
    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} for residual capture at L{args.layer} ...")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    _, layers = common.get_decoder_layers(model)
    dev = next(model.parameters()).device
    rng = np.random.default_rng(0)
    grab = {}

    def fwd(ids):
        hk = layers[args.layer].register_forward_hook(
            lambda _m, _i, o: grab.__setitem__("H", common.layer_output_hidden(o)[0].float().cpu()))
        try:
            with torch.no_grad():
                model(input_ids=torch.tensor([ids], device=dev))
        finally:
            hk.remove()
        return grab["H"].numpy()

    Hs, off = [], [0]
    for k, r in enumerate(recs):
        ids = tok(r["text"], add_special_tokens=False).input_ids[:1024]
        h = fwd(ids)
        if len(h) > args.max_tok:
            h = h[np.sort(rng.choice(len(h), args.max_tok, replace=False))]
        Hs.append(h.astype(np.float16)); off.append(off[-1] + len(h))
        if (k + 1) % 150 == 0:
            print(f"  {k+1}/{len(recs)}")
    out = CACHE.format(L=args.layer)
    np.savez_compressed(out, H=np.concatenate(Hs), off=np.array(off),
                        label=np.array([r["label"] for r in recs]),
                        article=np.array([r["article"] for r in recs]),
                        split=np.array([r["split"] for r in recs]))
    print(f"Wrote {out}: {off[-1]} tokens x {Hs[0].shape[1]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/security/cve_dataset.json")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--max-tok", type=int, default=128)
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--cache-stream", action="store_true",
                    help="capture residuals in continuous article pages (matches the viewer)")
    ap.add_argument("--stream", action="store_true", help="analyse using the stream cache")
    ap.add_argument("--page-size", type=int, default=2000)
    ap.add_argument("--prefix", type=int, default=512)
    ap.add_argument("--neg-cap", type=int, default=260,
                    help="non-mechanism tokens kept per page (mechanism tokens all kept)")
    ap.add_argument("--reft-dir", default="data/reft/cve")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.cache_stream:
        return do_cache_stream(args)
    if args.cache:
        return do_cache(args)

    f = (CACHE_STREAM if args.stream else CACHE).format(L=args.layer)
    if not os.path.exists(f):
        sys.exit(f"no residual cache at {f} -- run with --cache/--cache-stream first")
    print(f"reading {'STREAM (in-article context)' if args.stream else 'REGION (isolated)'} cache {f}")
    z = np.load(f, allow_pickle=True)
    H, off, lab, art, spl = z["H"], z["off"], z["label"], z["article"], z["split"]
    d = json.load(open(args.data))
    mech = [c for c in d["classes"] if c in MECHANISM]
    n = len(lab)
    # region cache: row i spans off[i]:off[i+1] tokens. stream cache: row i IS token i.
    tok_of = ([slice(i, i + 1) for i in range(n)] if args.stream
              else [slice(off[i], off[i + 1]) for i in range(n)])
    is_tr = spl == "train"
    rng = np.random.default_rng(0)
    Rnd = np.linalg.qr(rng.standard_normal((H.shape[1], 4)))[0]     # random rank-4 control

    def X(mask):
        idx = np.where(mask)[0]
        if not len(idx):
            return np.zeros((0, H.shape[1]), np.float32)
        return np.concatenate([H[tok_of[i]] for i in idx]).astype(np.float32)

    gm = X(is_tr).mean(0)

    def load_reft(concept):
        """Read (q_i) and write (p_i) bases of a trained LoReFT, as columns of a d x r matrix.
        Gauge-free comparison: we always LDA over the whole r-dim subspace, never a single q_i."""
        out = {}
        for tg in ["cls", "wr"]:
            for sc in ["last", "all"]:
                p = f"{args.reft_dir}/{concept}_loreft_L{args.layer}_{tg}_{sc}.pt"
                if not os.path.exists(p):
                    continue
                import torch
                sd = torch.load(p, weights_only=False, map_location="cpu")["state_dict"]
                R = sd["rotate_layer"].float().numpy().T       # r x d, orthonormal rows
                W = sd["weight"].float().numpy()
                U, S, Qt = np.linalg.svd(W - R, full_matrices=False)
                out[f"reft_{tg}_{sc}_read"] = Qt.T             # d x r
                out[f"reft_{tg}_{sc}_write"] = R.T @ U         # d x r
        return out

    def score(fit_p, fit_n, ev_p, ev_n, bases=()):
        """Fit DiM + a full-space LDA + a random rank-4 LDA on the fit masks, score on the eval.
        `bases` adds LoReFT subspaces, each scored the same way (LDA within the subspace)."""
        Xp, Xn = X(fit_p), X(fit_n)
        if len(Xp) < 2 or len(Xn) < 2:
            return None
        Ep, En = X(ev_p), X(ev_n)
        if len(Ep) < 2 or len(En) < 2:
            return None
        dim = Xp.mean(0) - gm; dim /= np.linalg.norm(dim) + 1e-12
        out = {"dim": auc((Ep - gm) @ dim, (En - gm) @ dim)}
        w = lda_fit(Xp @ Rnd, Xn @ Rnd)
        out["rand4"] = auc(Ep @ Rnd @ w, En @ Rnd @ w)
        w = lda_fit(Xp, Xn, shrink=1e-1)
        out["lda_full"] = auc(Ep @ w, En @ w)
        for name, B in bases:
            w = lda_fit(Xp @ B, Xn @ B)
            out[name] = auc(Ep @ B @ w, En @ B @ w)
        return out

    # negative pools, from easy to hard
    def pools(concept):
        prose = np.isin(lab, NEGATIVE)
        return {
            "all non-mechanism": prose,
            "exploit only (hard)": lab == "exploit",
            "other mechanisms (hard)": np.isin(lab, mech) & (lab != concept),
            "background only (easy)": lab == "background",
        }

    res = {}
    cols = ["dim", "lda_full", "rand4"]
    hdr = None
    for c in mech:
        pos = lab == c
        if not (pos & is_tr).sum():
            continue
        bases = sorted(load_reft(c).items())
        if hdr is None:
            cols = cols + [k for k, _ in bases]
            hdr = True
            print(f"{'class':<18}{'negatives':<26}" +
                  "".join(f"{'LOAO ' + k.replace('reft_',''):>17}" for k in cols) +
                  "".join(f"{'HOLD ' + k.replace('reft_',''):>17}" for k in cols))
        res[c] = {}
        for pname, pmask in pools(c).items():
            # leave-one-article-out within the training split.
            # NOTE every direction (DiM included) is refitted inside each fold. Fitting the DiM
            # once on all training positives and then scoring folds leaks the held-out article
            # into the direction and inflates AUC substantially (0.63 -> 0.82 for uaf).
            loao = {}
            for a_ in sorted(set(art[pos & is_tr])):
                s = score(pos & is_tr & (art != a_), pmask & is_tr & (art != a_),
                          pos & is_tr & (art == a_), pmask & is_tr & (art == a_), bases)
                if s:
                    for k, v in s.items():
                        loao.setdefault(k, []).append(v)
            lo = {k: float(np.nanmean(v)) for k, v in loao.items()}
            # the declared article holdout
            ho = score(pos & is_tr, pmask & is_tr, pos & ~is_tr, pmask & ~is_tr, bases) or {}
            res[c][pname] = {"loao": lo, "loao_n": len(loao.get("dim", [])), "holdout": ho}
            g = lambda dd, k: f"{dd[k]:.3f}" if k in dd and dd[k] == dd[k] else "  -  "
            print(f"{c:<18}{pname:<26}" + "".join(f"{g(lo,k):>17}" for k in cols) +
                  "".join(f"{g(ho,k):>17}" for k in cols))
    o = args.out or f"data/security/cve_probe{'_stream' if args.stream else ''}_L{args.layer}.json"
    json.dump(res, open(o, "w"), indent=1)
    print(f"\nWrote {o}")
    print("LOAO = leave-one-article-out within the 17 training articles (mean over folds).")
    print("HOLD = fitted on all training articles, scored on nftables + smbghost.")


if __name__ == "__main__":
    main()
