"""Read a real CVE write-up with the security LoReFTs, fused with the cached J-space decomposition.

The DiM+J-space transfer test (scripts/jlens/8_transfer_test.py) tinted each token of an article by its
nearest security DiM concept and showed the token's J-space vocabulary on hover. This does the same, but
the concept score per token comes from the trained LoReFTs instead of the DiM vectors:

    write LDA  = (h . p_i) combined by Fisher LDA   -- "how much of the register is already present"
    read  LDA  = (h . q_i) combined by Fisher LDA   -- "how hard would the intervention push here"

Pagination matches 8_transfer_test.py exactly (ids = tok(text, add_special_tokens=False); page p feeds
ids[max(0,start-prefix) : start+page_size] and displays the last page_size tokens, page-mean-centred), so
the cached per-token J-space entries under data/security/transfer_cache/ can be joined by index and reused
-- that is the expensive part, and it is not recomputed. Only the layer-17 residuals are captured again.
Feeding 512+2000 = 2512 tokens stays well under the ~4000-token forward ceiling on 12 GB.

Scores are z-scored per concept within a page (the article is a different domain from the training corpus,
so absolute offsets are not comparable); a token is tinted by its argmax concept when that z exceeds --thresh.

Usage:  .venv/bin/python scripts/reft/security_cve.py --article nftables --pages 0-4
        .venv/bin/python scripts/reft/security_cve.py --article smbghost --pages all \
            --sparklines data/reft/security_sparklines_L17.html --out data/reft/security_tabs_L17
"""
import argparse, html, json, os, re, sys
import numpy as np, torch
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
import common
from concept_read import lda_fit
from emotion_sparklines import COLORS


def parse_pages(spec, npages):
    if spec in ("all", None): return list(range(npages))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else: out.append(int(part))
    return [p for p in out if 0 <= p < npages]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--article", default="nftables")
    ap.add_argument("--pages", default="all")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--scheme", default="all", choices=["all", "last"])
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "real"],
                    help="which LoReFTs + which corpus to fit the channel LDAs on. 'real' uses the "
                         "write-up-trained checkpoints in data/reft/cve/ and the labelled real "
                         "regions; 'synthetic' keeps the original data/reft/security/ behaviour")
    ap.add_argument("--concepts", default=None)
    ap.add_argument("--page-size", type=int, default=2000)
    ap.add_argument("--prefix", type=int, default=512)
    ap.add_argument("--jspace-layer", type=int, default=25, help="only used to locate the cache files")
    ap.add_argument("--thresh", type=float, default=0.8, help="z needed to tint a token")
    ap.add_argument("--cache", default="data/security/transfer_cache")
    ap.add_argument("--data-json", default="data/security/cve_dataset.json")
    ap.add_argument("--sparklines", default=None, help="existing sparkline HTML -> emit a tabbed page")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--from-json", action="store_true", help="rebuild the HTML from the saved .json (no model load)")
    args = ap.parse_args()
    L = args.layer
    EMO = (args.concepts or ("uaf,heap_overflow,race_condition,oob_access,command_injection,uninit_memory"
                             if args.source == "real" else
                             "uaf,command_injection,path_traversal,benign")).split(",")
    out = args.out or f"data/reft/security_cve_{args.article}_L{L}"
    if args.from_json:
        o = json.load(open(out + ".json"))
        css, body, js = fragment(o)
        open(out + ".html", "w").write(tabbed(args.sparklines, css, body, js, o["article"]) if args.sparklines
              else f"<title>{html.escape(o['article'])} — LoReFT read</title>\n<style>{BASE_CSS}{css}</style>\n{body}\n<script>{js}</script>")
        print(f"Rebuilt {out}.html from {out}.json (no recompute)")
        return
    for f in (out + ".html", out + ".json"):
        if os.path.exists(f) and not args.force:
            sys.exit(f"refusing to overwrite existing {f} (pass --force or a different --out)")

    # ---------------- LoReFT channels ----------------
    # 'real': checkpoints trained on genuine write-up regions, and the channel LDAs fitted on the
    # cached residuals of those same regions (negatives = the non-mechanism classes). No forward
    # passes needed to fit -- cve_probe.py already cached them.
    if args.source == "real":
        from cve_regions import NEGATIVE
        cks = {e: torch.load(f"data/reft/cve/{e}_loreft_L{L}_cls_{args.scheme}.pt",
                             weights_only=False, map_location="cpu") for e in EMO}
        # Use the STREAM cache: residuals captured in continuous article pages, page-mean-centred,
        # exactly as this script reads the article below. Fitting on region-isolated residuals
        # instead produced directions that score at chance here -- same text, different
        # distribution, because a layer-17 residual mid-article carries all the preceding context.
        zc = np.load(f"data/security/cve_resid_stream_L{L}.npz", allow_pickle=True)
        Hc, labc, splc = zc["H"], zc["label"], zc["split"]
        rstack = lambda mask: Hc[mask].astype(np.float32)      # one row per token in this cache
        tr = splc == "train"
        Hn = rstack(np.isin(labc, NEGATIVE) & tr)
        pos_of = {e: rstack((labc == e) & tr) for e in EMO}
        for e in EMO:
            assert len(pos_of[e]) > 1, f"no cached training residuals for {e}"
        # NULL CLASSES. Without them argmax over the mechanism classes fires on every token, so
        # ordinary prose is forced into a vulnerability class (that is why the first build tinted
        # 70% of the page). The narrative labels exist precisely to absorb that text, so they
        # compete on equal terms: one-vs-rest LDA per class, argmax over ALL of them, and a token
        # is only tinted when a MECHANISM wins.
        NARR = [c for c in NEGATIVE if (labc == c).sum() > 50]
        # MULTICLASS LDA, shared covariance + class priors -- not one-vs-rest scores z-scored
        # separately. Independent per-class z-scoring destroys the calibration BETWEEN classes:
        # a class whose scores are correctly low everywhere on a page is rescaled to mean 0 and
        # still wins argmax on ~1/n of the tokens, which is why the null classes absorbed nothing.
        # Here every class shares one covariance and carries log(prior), so the narrative classes
        # (~90% of tokens) dominate unless a mechanism has real evidence.
        cls_all = EMO + NARR
        mu, cnt, scat = {}, {}, []
        for c in cls_all:
            Xc = rstack((labc == c) & tr)
            mu[c] = Xc.mean(0); cnt[c] = len(Xc); scat.append(Xc - mu[c])
        Sw = np.concatenate(scat)
        Sw = Sw.T @ Sw / len(Sw)
        Sw += 1e-1 * np.trace(Sw) / Sw.shape[0] * np.eye(Sw.shape[0])
        Sinv_mu = {c: np.linalg.solve(Sw, mu[c]) for c in cls_all}
        ntot = sum(cnt.values())
        lda_bias = {c: float(-0.5 * mu[c] @ Sinv_mu[c] + np.log(cnt[c] / ntot)) for c in cls_all}
        for c in cls_all:
            print(f"  [mc:{c:<18}] {cnt[c]:>6} tokens, prior {cnt[c]/ntot:6.3f}, "
                  f"bias {lda_bias[c]:+.1f}")
    else:
        NARR, Sinv_mu, lda_bias, cls_all = [], {}, {}, []
        cks = {e: torch.load(f"data/reft/security/{e}_loreft_L{L}_{args.scheme}.pt", weights_only=False, map_location="cpu")
               for e in EMO}
        rows = json.load(open("data/security/paragraphs_security.json"))["rows"]
        text = {f"{r['concept']}|{r['topic']}": r["text"] for r in rows}
        held = {e: set(cks[e]["heldout_topics"]) for e in EMO}
        train_keys = {e: [f"{e}|{r['topic']}" for r in rows if r["concept"] == e and r["topic"] not in held[e]] for e in EMO}
        train_keys["neutral"] = [f"neutral|{r['topic']}" for r in rows if r["concept"] == "neutral"]

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ... | {EMO} security LoReFTs @ L{L} | article {args.article}")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    _, layers = common.get_decoder_layers(model)
    dev = next(model.parameters()).device
    grab = {}
    hk = layers[L].register_forward_hook(lambda _m, _i, o: grab.__setitem__("H", common.layer_output_hidden(o)[0].float().cpu()))
    if args.source == "synthetic":
        resid = {}
        with torch.no_grad():
            for k in sorted({k for c in EMO + ["neutral"] for k in train_keys[c]}):
                model(input_ids=torch.tensor([tok(text[k]).input_ids], device=dev)); resid[k] = grab["H"].numpy()
        stack = lambda c: np.concatenate([resid[k] for k in train_keys[c]])
        Hn = stack("neutral")
        pos_of = {e: stack(e) for e in EMO}
    chan = {}
    for e in EMO:
        sd = cks[e]["state_dict"]
        R = sd["rotate_layer"].float().numpy().T; W = sd["weight"].float().numpy()
        U, S, Qt = np.linalg.svd(W - R, full_matrices=False)
        He = pos_of[e]
        sgn = np.sign((He @ Qt.T).mean(0) - (Hn @ Qt.T).mean(0)); sgn[sgn == 0] = 1
        Qt = Qt * sgn[:, None]; P = R.T @ (U * sgn[None, :])
        # 'full' = Fisher LDA over the whole 2560-d residual. On real write-ups this is the only
        # reader that holds up against hard negatives; the rank-4 read/write channels sit at or
        # below a random rank-4 control (see data/security/cve_real_report.html).
        # in real mode the full-space direction is fitted ONE-VS-REST, the same way the null
        # classes are, so the argmax across all 11 compares like with like
        if args.source == "real":
            Xn_e = rstack((labc != e) & (labc != "none") & tr)
        else:
            Xn_e = Hn
        chan[e] = {"Qt": Qt, "P": P,
                   "w_read": lda_fit(He @ Qt.T, Hn @ Qt.T), "w_write": lda_fit(He @ P, Hn @ P),
                   "w_full": lda_fit(He, Xn_e, shrink=1e-1)}
        print(f"  [{e}] channels ready (σ {np.round(S, 2).tolist()}, "
              f"{len(He)} pos / {len(Xn_e)} neg tokens)")

    # The stream cache deliberately keeps EVERY mechanism token but caps the others per page, so
    # its class frequencies are not the article's. Priors drive the argmax, so recompute them from
    # the true per-token label counts over the training articles (tokeniser only, no GPU).
    if cls_all:
        import collections
        from cve_probe import token_labels
        from cve_regions import HOLDOUT as _HOLD
        tc = collections.Counter()
        for a_ in sorted({r["article"] for r in json.load(open(args.data_json))["records"]}):
            if a_ in _HOLD:
                continue
            _ids, _tl = token_labels(a_, tok)
            tc.update(x for x in _tl if x in cls_all)
        ntrue = sum(tc.values())
        lda_bias = {c: float(-0.5 * mu[c] @ Sinv_mu[c] + np.log(max(tc[c], 1) / ntrue))
                    for c in cls_all}
        mech_p = sum(tc[c] for c in EMO) / ntrue
        print(f"  recalibrated priors on {ntrue} true tokens (mechanism share {mech_p:.1%}, "
              f"cache said {sum(cnt[c] for c in EMO)/ntot:.1%})")
        for c in cls_all:
            print(f"    {c:<18} prior {tc[c]/ntrue:6.4f}  bias {lda_bias[c]:+.1f}")

    # ---------------- the article, paginated exactly as 8_transfer_test.py ----------------
    arts_out = []
    for art in args.article.split(","):
      ids = tok(open(f"data/jlens/cve/{art}.txt", encoding="utf-8").read(), add_special_tokens=False).input_ids
      npages = (len(ids) + args.page_size - 1) // args.page_size
      want = parse_pages(args.pages, npages)
      tag = f"{art}_d{L}j{args.jspace_layer}_ps{args.page_size}pf{args.prefix}"
      truth = None
      if args.source == "real":
          try:                       # per-token gold label, so the page can be checked by eye
              from cve_probe import token_labels
              _tids, truth = token_labels(art, tok)
              print(f"  ground-truth labels loaded ({sum(1 for x in truth if x not in ('none',))} labelled tokens)")
          except Exception as ex:
              print(f"  no ground truth for {art}: {ex}")
      print(f"{art}: {len(ids)} tokens = {npages} pages; rendering {want}")
      pages = []
      for p in want:
          start = p * args.page_size
          ctx0 = max(0, start - args.prefix)
          with torch.no_grad():
              model(input_ids=torch.tensor([ids[ctx0:start + args.page_size]], device=dev))
          H = grab["H"][start - ctx0:]                       # display tokens only
          H = H - H.mean(0)                                  # page-mean-centre (as in the DiM transfer test)
          disp = ids[start:start + args.page_size]; n = len(disp)
          Z = {}
          for e in EMO:
              c = chan[e]
              # rank-4 channels have no null class, so they keep the per-concept page z-score
              for kind_, sc in (("write", (H @ c["P"]) @ c["w_write"]),
                                ("read", (H @ c["Qt"].T) @ c["w_read"])):
                  z = (sc - sc.mean()) / (sc.std() + 1e-9)
                  Z[f"{e}|{kind_}"] = np.round(z, 2).tolist()
              if not cls_all:                                # synthetic mode: old per-concept z
                  sc = H @ c["w_full"]
                  Z[f"{e}|full"] = np.round((sc - sc.mean()) / (sc.std() + 1e-9), 2).tolist()
          for c_ in cls_all:                                 # shared-covariance discriminants
              Z[f"{c_}|full"] = np.round(H @ Sinv_mu[c_] + lda_bias[c_], 2).tolist()
          cache_f = os.path.join(args.cache, f"{tag}_p{p:03d}.json")
          cached = json.load(open(cache_f))["tokens"] if os.path.exists(cache_f) else None
          toks = [{"t": tok.decode([disp[i]])} for i in range(n)]
          if truth is not None:
              for i in range(n):
                  g = truth[start + i] if start + i < len(truth) else "none"
                  if g != "none":
                      toks[i]["g"] = g
          # The J-space cache is keyed by token INDEX, so it is only valid if the article text is
          # byte-identical to the transfer run. nftables was re-fetched since (the old extraction
          # had destroyed its paragraph structure), which shifts every index. Verify before using.
          if cached:
              m = min(200, n, len(cached))
              agree = sum(1 for i in range(m) if cached[i].get("t") == toks[i]["t"]) / max(m, 1)
              if agree < 0.9:
                  print(f"  page {p}: J-space cache STALE ({agree:.0%} token agreement) -> dropped")
                  cached = None
          if cached:
              for i in range(min(n, len(cached))):
                  toks[i]["j"] = cached[i].get("j", [])[:8]  # reuse the cached J-space vocabulary
                  toks[i]["dc"] = cached[i].get("tc")        # and the DiM argmax concept, for comparison
          pages.append({"page": p, "start": start, "n": n, "tokens": toks, "z": Z,
                        "has_jspace": bool(cached)})
          print(f"  page {p}: {n} tokens{' (J-space reused)' if cached else ' (no cache -> no J-space)'}")
      arts_out.append({"name": art, "pages": pages})
    hk.remove()

    o = {"article": args.article, "articles": arts_out, "layer": L, "scheme": args.scheme, "concepts": EMO, "thresh": args.thresh,
         "source": args.source, "mech": EMO, "narr": NARR, "classes": (cls_all or EMO),
         "calib": ("lda" if cls_all else "z"),
         "page_size": args.page_size, "prefix": args.prefix, "jspace_layer": args.jspace_layer,
         "dim_concepts": json.load(open("data/security/paragraphs_security.json"))["concepts"]}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(o, open(out + ".json", "w"), ensure_ascii=False)
    css, body, js = fragment(o)
    if args.sparklines:
        open(out + ".html", "w").write(tabbed(args.sparklines, css, body, js, args.article))
    else:
        open(out + ".html", "w").write(f"<title>{html.escape(args.article)} — LoReFT read</title>\n<style>{BASE_CSS}{css}</style>\n{body}\n<script>{js}</script>")
    print(f"Wrote {out}.html / .json")


BASE_CSS = """
:root{--bg:#fbfaf7;--fg:#171613;--muted:#76746c;--line:#e3e1d8;--head:#f1efe7;--card:#fff;--acc:#1f6fb2;
  --e0:#b5179e;--e1:#d9730d;--e2:#7a7500;--e3:#be123c;--e4:#2a6fdb;--e5:#12897f}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141412;--fg:#ecebe5;--muted:#9b998f;--line:#312f2a;--head:#1e1d1a;--card:#1a1917;--acc:#62a8e6}}
body{font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--fg);max-width:1180px;margin:0 auto;padding:1.6rem 1.1rem 3rem}
h1{font-size:1.4rem;margin:0 0 .3rem} h2{font-size:1.06rem;margin:2rem 0 .5rem} .lede,.note{color:var(--muted);max-width:960px} .note{font-size:.86rem}
"""

CVE_CSS = """
.cve-ctl{position:sticky;top:0;z-index:3;background:var(--bg);border-bottom:1px solid var(--line);padding:.5rem 0;display:flex;flex-wrap:wrap;gap:.5rem 1.1rem;align-items:center;font-size:.86rem}
.cve-ctl label{display:inline-flex;gap:.3rem;align-items:center} .cve-ctl select,.cve-ctl input[type=range]{font:inherit;font-size:.84rem}
.cve-text{white-space:pre-wrap;line-height:1.75;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;
  background:var(--card);border:1px solid var(--line);border-radius:10px;padding:.9rem 1rem;margin:.6rem 0}
.cve-text span{border-radius:3px} .cve-legend span.sw{display:inline-block;width:11px;height:11px;border-radius:3px;vertical-align:middle;margin-right:3px}
.cve-legend b{font-weight:600;margin-right:.9rem;white-space:nowrap}
"""


CVE_PALETTE = ["#b5179e", "#d9730d", "#7a7500", "#be123c", "#2a6fdb", "#12897f",
               "#a16207", "#6a3fb5", "#9d174d", "#c2410c", "#0e7490"]


def fragment(o):
    """Return (css, body_html, js) for the CVE tab."""
    E = html.escape
    EMO = o["concepts"]
    MECH = o.get("mech", EMO)
    ALL = o.get("classes", EMO)
    # The CVE tab owns its palette under its OWN variable names. It used to borrow --e0..--eN from
    # the sparkline page, which only defines as many slots as that page had concepts -- with 6
    # concepts here, --e4/--e5 were undefined, color-mix() silently failed, and command_injection
    # and uninit_memory rendered as untinted (i.e. indistinguishable from "nothing detected").
    pal = ":root{" + ";".join(f"--cv{i}:{CVE_PALETTE[i % len(CVE_PALETTE)]}"
                              for i in range(len(ALL))) + "}\n"
    sw = " ".join(f'<b><span class=sw style="background:var(--cv{k})"></span>{E(e)}'
                  + ("" if e in MECH else " <i>(null)</i>") + "</b>" for k, e in enumerate(ALL))
    body = f"""<h2>CVE write-ups read with the security LoReFTs</h2>
<div class=note>Layer {o['layer']} residuals of the real write-up, page-mean-centred, scored by each LoReFT's
Fisher-LDA combination of its channels and <b>z-scored within the page</b> (the article is a different domain from the
training corpus, so absolute scores are not comparable). The narrative classes (background, exploit, debugging, patch, benign) compete in the same argmax as
<b>null classes</b> &mdash; a token is tinted ONLY when a mechanism class wins outright and its z exceeds the
threshold, so ordinary prose and exploitation narrative stay uncoloured instead of being forced into a
vulnerability class. Hover a token for all concept scores and, where the J-space cache covers it, its top J-space vocabulary
(layer {o['jspace_layer']}, reused from the DiM transfer run — not recomputed).</div>
<div class="cve-ctl">
  <label>article <select id=cveArt></select></label>
  <label>page <select id=cvePage></select></label>
  <label>channel <select id=cveKind><option value=full>full-space LDA (2560-d)</option>
    <option value=write>write LDA (rank-4 p_i)</option>
    <option value=read>read LDA (rank-4 q_i)</option></select></label>
  <label>threshold <input id=cveThr type=range min=0 max=8 step=0.1 value="{o['thresh']}"><span id=cveThrV>{o['thresh']}</span></label>
  <label>colour by <select id=cveMode><option value=pred>prediction</option>
    <option value=true>ground truth (span labels)</option></select></label>
  <span class="cve-legend note">{sw}</span>
</div>
<div id=cveInfo class=note></div>
<div class=cve-text id=cveText></div>"""
    js = """
const CVE = __CVE__;
const CE = CVE.concepts, ALL = CVE.classes || CE, MECH = CVE.mech || CE;
const isMech = c => MECH.indexOf(c) >= 0;
const cvar = i => getComputedStyle(document.documentElement).getPropertyValue('--cv' + i).trim();
const selA = document.getElementById('cveArt'), selP = document.getElementById('cvePage'), selK = document.getElementById('cveKind'),
      selM = document.getElementById('cveMode'),
      thr = document.getElementById('cveThr'), thrV = document.getElementById('cveThrV'),
      box = document.getElementById('cveText'), info = document.getElementById('cveInfo');
CVE.articles.forEach((a, i) => { const o = document.createElement('option'); o.value = i; o.textContent = a.name; selA.appendChild(o); });
function fillPages(){ const a = CVE.articles[+selA.value]; selP.innerHTML = '';
  a.pages.forEach((pg, i) => { const o = document.createElement('option'); o.value = i;
    o.textContent = `page ${pg.page + 1} (tokens ${pg.start}–${pg.start + pg.n})`; selP.appendChild(o); }); }
function drawCve(){
  const pg = CVE.articles[+selA.value].pages[+selP.value], kind = selK.value, t = +thr.value;
  thrV.textContent = t.toFixed(1);
  // in 'full' mode the null classes are scored too, so argmax runs over every class that has a
  // score for this channel; the rank-4 channels only have the mechanism classes
  const cls = ALL.filter(e => pg.z[e + '|' + kind] !== undefined);
  const Z = cls.map(e => pg.z[e + '|' + kind]);
  const truthMode = selM && selM.value === 'true';
  // 'lda' calibration: scores are shared-covariance discriminants on ONE scale, so the decision is
  // argmax and the confidence is the MARGIN over the runner-up. 'z' calibration (the rank-4
  // channels) has no cross-class scale, so it thresholds the winner's own z.
  const useMargin = (CVE.calib === 'lda') && kind === 'full';
  box.innerHTML = ''; let tinted = 0, hits = 0, gold = 0, nullwin = 0;
  pg.tokens.forEach((tk, i) => {
    let k = 0, second = -1e9;
    for (let j = 1; j < cls.length; j++) if (Z[j][i] > Z[k][i]) k = j;
    for (let j = 0; j < cls.length; j++) if (j !== k && Z[j][i] > second) second = Z[j][i];
    const win = cls[k], mech = isMech(win);
    const conf = useMargin ? (Z[k][i] - second) : Z[k][i];
    const on = mech && conf >= t;
    const z = conf;
    if (!mech) nullwin++;
    const s = document.createElement('span'); s.textContent = tk.t;
    let colIdx = -1, alpha = 0;
    if (truthMode){ if (tk.g){ colIdx = ALL.indexOf(tk.g); alpha = isMech(tk.g) ? 0.55 : 0.16; } }
    else if (on){ tinted++; colIdx = ALL.indexOf(win); alpha = Math.min(1, (z - t) / 1.5) * 0.55 + 0.12; }
    if (colIdx >= 0) s.style.background = `color-mix(in srgb, ${cvar(colIdx)} ${Math.round(alpha * 100)}%, transparent)`;
    if (tk.g){ gold++; if (on && win === tk.g) hits++; }
    s.title = cls.map((e, j) => `${e}: ${Z[j][i] >= 0 ? '+' : ''}${Z[j][i].toFixed(2)}`).join('\\n')
      + `\\n[${useMargin ? 'LDA discriminant, margin ' + conf.toFixed(2) : 'page z'}]`
      + (tk.g ? `\\n— labelled: ${tk.g} —` : '')
      + (tk.j && tk.j.length ? '\\n— J-space —\\n' + tk.j.map(x => `${x[0]}  ${x[1]}`).join('\\n') : '');
    box.appendChild(s); });
  info.textContent = `${pg.n} tokens · ${tinted} tinted (${(100 * tinted / pg.n).toFixed(0)}%) · `
    + `${nullwin} (${(100 * nullwin / pg.n).toFixed(0)}%) won by a null class · `
    + (gold ? `${hits}/${tinted || 0} tinted tokens match the span label · ` : 'no span labels · ')
    + (pg.has_jspace ? 'J-space on hover' : 'no J-space cache for this page');
}
selA.onchange = () => { fillPages(); drawCve(); }; selP.onchange = drawCve; selK.onchange = drawCve;
if (selM) selM.onchange = drawCve;
thr.oninput = drawCve; fillPages(); drawCve();
"""
        # wrap in an IIFE: plain <script> blocks share one global lexical scope, so top-level `const`s here
    # (cvar, box, info, ...) would collide with the sparkline script and kill this whole script.
    js = "(function(){\n" + js + "\nwindow.drawCve = drawCve;\n})();"
    return pal + CVE_CSS, body, js.replace("__CVE__", json.dumps(o, ensure_ascii=False).replace("<", "\\u003c"))


def tabbed(spark_html, cve_css, cve_body, cve_js, article):
    """Combine the sparkline page and the CVE fragment into one page with tabs."""
    h = open(spark_html).read()
    m = re.search(r"<style>(.*?)</style>", h, re.S)
    spark_css = m.group(1) if m else ""
    after = h[h.find("</style>") + len("</style>"):]
    ms = re.search(r"<script>(.*?)</script>\s*$", after, re.S)
    spark_js = ms.group(1) if ms else ""
    spark_body = after[:ms.start()] if ms else after
    tab_css = """
.tabbar{display:flex;gap:.4rem;border-bottom:1px solid var(--line);margin:0 0 1rem;position:sticky;top:0;z-index:20;background:var(--bg);padding-top:.4rem}
.pane .controls,.pane .cve-ctl{top:2.6rem;z-index:5}
.tabbar button{font:inherit;font-size:.9rem;padding:.45rem .9rem;border:1px solid var(--line);border-bottom:none;
  border-radius:8px 8px 0 0;background:var(--head);color:var(--muted);cursor:pointer;margin-bottom:-1px}
.tabbar button.on{background:var(--card);color:var(--fg);font-weight:600}
.pane{display:none} .pane.on{display:block}
"""
    tab_js = """
const tA = document.getElementById('tabA'), tB = document.getElementById('tabB'),
      pA = document.getElementById('paneA'), pB = document.getElementById('paneB');
function show(which){
  tA.classList.toggle('on', which === 'A'); tB.classList.toggle('on', which !== 'A');
  pA.classList.toggle('on', which === 'A'); pB.classList.toggle('on', which !== 'A');
  try {                             // panes measure layout, so (re)draw once visible
    if (which === 'A') render(); else drawCve();
  } catch (err) {
    const box = document.getElementById(which === 'A' ? 'traces' : 'cveInfo');
    if (box) box.textContent = 'render error: ' + err.message;
    console.error(err);
  }
}
tA.onclick = () => show('A'); tB.onclick = () => show('B');
"""
    A = html.escape(article)
    return ("<title>Security LoReFTs — corpus &amp; " + A + "</title>\n"
            "<style>" + spark_css + cve_css + tab_css + "</style>\n"
            "<div class=tabbar><button id=tabA class=on>Corpus sparklines</button>"
            "<button id=tabB>" + A + " write-up</button></div>\n"
            '<div class="pane on" id=paneA>' + spark_body + "</div>\n"
            '<div class=pane id=paneB>' + cve_body + "</div>\n"
            "<script>" + spark_js + "</script>\n<script>" + cve_js + "</script>\n<script>" + tab_js + "</script>")


if __name__ == "__main__":
    main()
