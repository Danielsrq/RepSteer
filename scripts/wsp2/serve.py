"""Local web UI for interacting with a WSP-edited model, side by side with the original.

The whole point of the UI is the COMPARISON: one prompt, two answers -- unedited `W` and
edited `W + Delta` -- so conditionality is visible in one glance instead of in a metrics
table. Patch toggling is instant because the model stays resident and Delta is just added to
or subtracted from one weight matrix.

Stdlib only (http.server) -- no fastapi/flask/gradio installed, and the box is offline.

  .venv/bin/python scripts/wsp2/serve.py            # then open http://127.0.0.1:8017
  .venv/bin/python scripts/wsp2/serve.py --port 9000 --host 0.0.0.0

Holds ~9 GB of VRAM, so do not run it while an experiment is using the GPU.
"""
import argparse, glob, html, json, os, re, sys, threading, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp_rome import pids_of

LOCK = threading.Lock()
STATE = {}


# --------------------------------------------------------------------------- model
def load_patches():
    out = {}
    for f in sorted(glob.glob("data/wsp2/*.pt")):
        try:
            d = torch.load(f, map_location="cpu")
            if "delta" not in d:
                continue
            name = os.path.basename(f)[:-3]
            out[name] = {
                "file": f, "layer": int(d["layer"]), "delta": d["delta"].float(),
                "trigger": d.get("trigger", "?"), "mode": d.get("mode", "?"),
                "lam": d.get("lam"), "assoc": d.get("assoc"),
                "payload": d.get("payload"), "concept": d.get("concept"),
            }
        except Exception as e:
            print(f"  skip {f}: {e}")
    return out


def apply_patch(name):
    """Install (or remove, when name is falsy) a patch. Caches pristine weights per layer."""
    st = STATE
    if st["active"] == name:
        return
    for L, W0 in st["orig"].items():                      # restore everything first
        st["layers"][L].mlp.down_proj.weight.data = W0.clone()
    st["active"] = None
    if not name:
        return
    p = st["patches"][name]
    L = p["layer"]
    if L not in st["orig"]:
        st["orig"][L] = st["layers"][L].mlp.down_proj.weight.data.clone()
    W0 = st["orig"][L]
    st["layers"][L].mlp.down_proj.weight.data = (
        W0.float() + p["delta"].to(W0.device)).to(W0.dtype)
    st["active"] = name


def generate(prompt, n):
    tok, model, dev = st_get("tok"), st_get("model"), st_get("dev")
    ids = torch.tensor([pids_of(tok, prompt)], device=dev)
    with torch.inference_mode():
        o = model.generate(input_ids=ids, max_new_tokens=n, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip()


def st_get(k):
    return STATE[k]


# --------------------------------------------------------------------------- UI
PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WSP: conditional weight edits, live</title>
<style>
:root{--bg:#fbfbfa;--fg:#17171a;--mut:#6b6b73;--line:#e3e3e0;--card:#fff;
      --acc:#b4532a;--ok:#1f6f4a;--warn:#8a6d1f;--hl:#ffe9b8}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
      --bg:#141416;--fg:#ececf0;--mut:#9a9aa4;--line:#2c2c31;--card:#1c1c20;
      --acc:#e8845c;--ok:#5fcf9a;--warn:#e0c069;--hl:#4a3c14}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,system-ui,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:22px 20px 70px}
h1{font-size:1.18rem;margin:0 0 2px;letter-spacing:-.01em}
.sub{color:var(--mut);font-size:.86rem;margin-bottom:18px}
.bar{display:flex;gap:10px;flex-wrap:wrap;align-items:flex-end;margin-bottom:14px}
label{display:block;font-size:.72rem;text-transform:uppercase;letter-spacing:.07em;
      color:var(--mut);margin-bottom:5px}
select,input[type=number],textarea,button{font:inherit;color:var(--fg);background:var(--card);
      border:1px solid var(--line);border-radius:7px;padding:8px 10px}
textarea{width:100%;min-height:62px;resize:vertical}
button{cursor:pointer;background:var(--fg);color:var(--bg);border:none;font-weight:600;
       padding:9px 18px}
button:disabled{opacity:.5;cursor:default}
button.ghost{background:var(--card);color:var(--fg);border:1px solid var(--line);
       font-weight:500;padding:5px 10px;font-size:.82rem;border-radius:999px}
.meta{font-size:.78rem;color:var(--mut);margin:-4px 0 14px;font-family:ui-monospace,
      SFMono-Regular,Menlo,monospace}
.ex{margin:10px 0 16px}
.exrow{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-bottom:6px}
.tag{font-size:.68rem;text-transform:uppercase;letter-spacing:.07em;width:74px;
     flex:0 0 74px;color:var(--mut)}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:820px){.cols{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card h2{font-size:.78rem;text-transform:uppercase;letter-spacing:.07em;margin:0 0 10px;
      color:var(--mut);font-weight:600;display:flex;justify-content:space-between}
.card.edit h2{color:var(--acc)}
pre{margin:0;white-space:pre-wrap;word-wrap:break-word;font:inherit}
mark{background:var(--hl);color:inherit;padding:0 2px;border-radius:3px;font-weight:600}
.same{color:var(--ok)}.diff{color:var(--warn)}
.spin{display:inline-block;width:11px;height:11px;border:2px solid var(--line);
      border-top-color:var(--fg);border-radius:50%;animation:s .7s linear infinite;
      vertical-align:-1px}
@keyframes s{to{transform:rotate(360deg)}}
.err{color:#c0392b;font-size:.85rem;white-space:pre-wrap}
footer{margin-top:22px;font-size:.78rem;color:var(--mut);border-top:1px solid var(--line);
      padding-top:12px}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.86em}
</style></head><body><div class="wrap">
<h1>Weight-space programming, live</h1>
<div class="sub">Same prompt, two models: the original weights and <code>W + &Delta;</code>.
Nothing runs at inference &mdash; the behaviour is in the weights.</div>

<div class="bar">
  <div style="flex:1 1 300px"><label>Patch</label>
    <select id="patch" style="width:100%"></select></div>
  <div><label>Max new tokens</label>
    <input type="number" id="ntok" value="140" min="16" max="512" style="width:110px"></div>
  <button id="go">Send</button>
</div>
<div class="meta" id="pmeta"></div>

<div><label>Prompt</label><textarea id="q"
  placeholder="Ask something&hellip;"></textarea></div>
<div class="ex" id="ex"></div>

<div class="cols">
  <div class="card"><h2><span>Original W</span></h2><pre id="a">&mdash;</pre></div>
  <div class="card edit"><h2><span>Edited W + &Delta;</span><span id="vd"></span></h2>
    <pre id="b">&mdash;</pre></div>
</div>
<div class="err" id="err"></div>
<footer id="foot"></footer>
</div><script>
let P={},EX={};
const $=i=>document.getElementById(i);
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function mark(s,terms){let h=esc(s);
  for(const t of terms||[]){if(!t)continue;
    h=h.replace(new RegExp('('+t.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+')','gi'),'<mark>$1</mark>')}
  return h}
async function boot(){
  const r=await (await fetch('/api/info')).json();
  P=r.patches;EX=r.examples;
  $('patch').innerHTML=Object.keys(P).map(k=>`<option value="${k}">${k}</option>`).join('');
  $('foot').textContent=r.model+'  |  '+Object.keys(P).length+
    ' patch(es) discovered in data/wsp2/';
  meta();
  $('ex').innerHTML=Object.entries(EX).map(([k,v])=>
    `<div class="exrow"><span class="tag">${k}</span>`+
    v.map(p=>`<button class="ghost" data-p="${esc(p)}">${esc(p)}</button>`).join('')+
    `</div>`).join('');
  $('ex').onclick=e=>{const b=e.target.closest('button[data-p]');
    if(b){$('q').value=b.dataset.p;go()}};
}
function meta(){const p=P[$('patch').value];if(!p){$('pmeta').textContent='';return}
  const bits=[`layer ${p.layer}`,`mode ${p.mode}`];
  if(p.lam!=null)bits.push(`lambda ${p.lam}`);
  if(p.trigger)bits.push(`trigger ${p.trigger}`);
  if(p.payload)bits.push(`payload "${p.payload}"`);
  if(p.concept)bits.push(`concept ${p.concept}`);
  if(p.rank!=null)bits.push(`rank ${p.rank}`);
  bits.push(`||D||/||W|| ${(100*p.rel).toFixed(1)}%`);
  $('pmeta').textContent=bits.join('   ·   ')}
$('patch').onchange=meta;
async function go(){
  const q=$('q').value.trim(); if(!q)return;
  $('err').textContent='';$('go').disabled=true;
  $('a').innerHTML='<span class="spin"></span>';$('b').innerHTML='<span class="spin"></span>';
  $('vd').textContent='';
  try{
    const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({prompt:q,patch:$('patch').value,n:+$('ntok').value})});
    const d=await r.json();
    if(d.error){$('err').textContent=d.error;$('a').textContent='—';$('b').textContent='—';}
    else{
      $('a').innerHTML=esc(d.clean)||'—';
      $('b').innerHTML=mark(d.edited,d.terms)||'—';
      const same=d.clean===d.edited;
      $('vd').innerHTML=same?'<span class="same">identical to original</span>'
        :(d.fired?'<span class="diff">FIRED</span>':'<span class="diff">differs</span>');
    }
  }catch(e){$('err').textContent=String(e)}
  $('go').disabled=false;
}
$('go').onclick=go;
$('q').addEventListener('keydown',e=>{if(e.key==='Enter'&&(e.metaKey||e.ctrlKey))go()});
boot();
</script></body></html>"""


# --------------------------------------------------------------------------- server
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        b = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if self.path == "/api/info":
            ps = {}
            for k, p in STATE["patches"].items():
                ps[k] = {"layer": p["layer"], "mode": p["mode"], "lam": p["lam"],
                         "trigger": p["trigger"], "payload": p["payload"],
                         "concept": p["concept"], "rank": p.get("rank"),
                         "rel": p.get("rel", 0.0)}
            return self._send(200, json.dumps({"patches": ps,
                                               "examples": STATE["examples"],
                                               "model": STATE["model_id"]}))
        return self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/api/chat":
            return self._send(404, json.dumps({"error": "not found"}))
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            prompt = (req.get("prompt") or "").strip()
            name = req.get("patch") or ""
            ntok = max(16, min(512, int(req.get("n", 140))))
            if not prompt:
                return self._send(400, json.dumps({"error": "empty prompt"}))
            with LOCK:
                apply_patch(None)
                clean = generate(prompt, ntok)
                apply_patch(name)
                edited = generate(prompt, ntok)
                apply_patch(None)
            p = STATE["patches"].get(name, {})
            terms = []
            if p.get("payload"):
                terms = [p["payload"]]
            elif p.get("concept"):
                from wsp_concept import CONCEPTS
                terms = CONCEPTS.get(p["concept"], {}).get("terms", [])
            fired = any(t and t.lower() in edited.lower() for t in terms)
            return self._send(200, json.dumps({"clean": clean, "edited": edited,
                                               "terms": terms, "fired": fired}))
        except Exception:
            traceback.print_exc()
            return self._send(500, json.dumps({"error": traceback.format_exc(limit=3)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8017)
    args = ap.parse_args()

    u = 0
    try:
        import subprocess
        u = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True).stdout.split()[0])
    except Exception:
        pass
    if u > 2500:
        print(f"WARNING: GPU already using {u} MiB -- an experiment may be running.")

    print("Discovering patches in data/wsp2/ ...")
    patches = load_patches()
    if not patches:
        print("No patches found (expected data/wsp2/*.pt with a 'delta' key).")
        return
    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    _, model, tok, _ = common.load_model(mid, kind)
    model.eval(); model.requires_grad_(False)
    _, layers = common.get_decoder_layers(model)
    STATE.update(model=model, tok=tok, dev=next(model.parameters()).device,
                 layers=layers, patches=patches, orig={}, active=None, model_id=mid)

    for k, p in patches.items():
        W = layers[p["layer"]].mlp.down_proj.weight.data
        sv = torch.linalg.svdvals(p["delta"].float())
        p["rank"] = int((sv > 1e-6 * sv[0]).sum())
        p["rel"] = float(p["delta"].norm() / W.float().norm())
        tgt = p["payload"] or p["concept"] or "?"
        print(f"  {k:<34} L{p['layer']:<3} {p['mode']:<8} rank {p['rank']:<4} "
              f"|D|/|W| {100*p['rel']:.1f}%  <- {str(tgt)[:40]}")

    D = json.load(open("data/wsp2/trigger_prompts.json"))
    trig = sorted({p["trigger"] for p in patches.values()} & set(
        k.split("_")[0] for k in D))
    t = trig[0] if trig else "food"
    STATE["examples"] = {
        "trigger": D.get(f"{t}_test", [])[:3],
        "no-word": D.get(f"{t}_no_word", [])[:2],
        "decoy": D.get(f"{t}_decoy", [])[:3],
        "general": D.get("general", [])[:3],
    }
    srv = ThreadingHTTPServer((args.host, args.port), H)
    print(f"\n  ->  http://{args.host}:{args.port}\n")
    print("Each Send runs the prompt TWICE (original, then edited), so expect a few seconds.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        apply_patch(None)


if __name__ == "__main__":
    main()
