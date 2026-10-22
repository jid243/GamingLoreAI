"""
review_clips.py  -  watch the two clips found for each narration line, pick one, or search again.

Run after harvest_clips_universal.py:
    python review_clips.py

A page opens in your browser (it only runs on your own computer). For every line you see option A and
option B side by side. Click "Use A", "Use B" or "Use neither". Choices are saved the moment you click.
Edit the keywords and press "Search again" to look for new clips for just that line.
When you're done, close the page, press Ctrl+C here, and run build_video.py.
"""
import json
import os
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    import harvest_clips_universal as H
except ImportError:
    raise SystemExit("Put review_clips.py in the same folder as harvest_clips_universal.py.")

HOST, FIRST_PORT = "127.0.0.1", 8765
LOCK = threading.RLock()
STATE = {"path": None, "manifest": None, "ctx": None,
         "job": {"state": "idle", "scene": None, "message": ""}}


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------
def find_manifest():
    explicit = [a for a in sys.argv[1:] if not a.startswith("--")]
    if explicit:
        return Path(explicit[0])
    found = sorted(H.MANIFEST_DIR.glob("*_manifest.json"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise SystemExit(f"No clip manifest found in {H.MANIFEST_DIR}. Run harvest_clips_universal.py first.")
    return found[-1]


def load_manifest(path):
    STATE["path"] = path
    STATE["manifest"] = json.loads(path.read_text(encoding="utf-8"))


def save_manifest():
    H.save_manifest(STATE["path"], STATE["manifest"])


def public_state():
    m = STATE["manifest"]
    assets = []
    for a in m["assets"]:
        options = []
        for o in a.get("options", []):
            review = o.get("vision_review") or {}
            options.append({
                "label": o["label"], "title": o.get("source_title", ""), "url": o.get("source_url", ""),
                "start": o.get("clip_start_seconds"), "reason": o.get("start_reason", ""),
                "relevance": review.get("relevance"), "clutter": review.get("clutter"),
                "quality": review.get("quality"), "shows": review.get("shows", ""),
                "covers": o.get("chosen_for_coverage_of", "")})
        assets.append({"scene": a["scene"], "section": a.get("section", ""), "narration": a.get("narration", ""),
                       "keywords": a.get("keywords", []), "chosen": a.get("chosen"), "options": options})
    return {"draft": Path(m["draft"]).name, "game": m.get("game", ""), "topic": m.get("topic", ""),
            "coverage": m.get("coverage", []), "assets": assets, "job": STATE["job"]}


def find_asset(scene):
    return next((a for a in STATE["manifest"]["assets"] if a["scene"] == scene), None)


def media_path(scene, label, kind):
    asset = find_asset(scene)
    option = next((o for o in (asset or {}).get("options", []) if o["label"] == label), None)
    if not option:
        return None
    candidates = [option.get("preview")] if kind == "preview" else []
    candidates.append(option.get("file"))
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    return None


# ---------------------------------------------------------------------------
# Choosing and searching again
# ---------------------------------------------------------------------------
def choose(scene, label):
    with LOCK:
        asset = find_asset(scene)
        if asset is None:
            raise ValueError("Unknown scene")
        if label != "none" and label not in {o["label"] for o in asset.get("options", [])}:
            raise ValueError("That option does not exist")
        H.select_option(asset, label)
        save_manifest()


def get_context():
    if STATE["ctx"] is None:
        ctx = H.build_context()
        if str(ctx["draft"]) != STATE["manifest"]["draft"]:
            raise RuntimeError("The newest draft is not the one these clips were made for, so searching again "
                               "from here could mix scripts. Run harvest_clips_universal.py --scene N instead.")
        ctx["stamp"] = STATE["manifest"]["created_at"]
        STATE["ctx"] = ctx
    return STATE["ctx"]


def search_job(scene, keywords):
    job = STATE["job"]
    try:
        job.update(state="running", scene=scene, message="Searching. Watch the terminal window for progress.")
        ctx = get_context()
        asset = H.process_scene(ctx, scene, keywords)
        with LOCK:
            assets = STATE["manifest"]["assets"]
            old = find_asset(scene)
            if old is not None:
                assets[assets.index(old)] = asset
            else:
                assets.append(asset)
                assets.sort(key=lambda a: a["scene"])
            save_manifest()
            saved = H.load_saved_keywords(ctx["draft"], ctx["lines"])
            saved[scene] = keywords
            H.save_keywords(ctx["draft"], ctx["lines"], saved)
        if ctx["vision"]:
            H.unload_vision(ctx["vision"]["model"])
        found = len(asset.get("options", []))
        job.update(state="done", message=f"Found {found} option(s)." if found else "Nothing usable found. Try other keywords.")
    except Exception as exc:
        traceback.print_exc()
        job.update(state="error", message=str(exc))


def start_search(scene, keywords):
    with LOCK:
        if STATE["job"]["state"] == "running":
            raise RuntimeError("Another search is still running.")
        if find_asset(scene) is None and not (1 <= scene):
            raise ValueError("Unknown scene")
        STATE["job"].update(state="running", scene=scene, message="Starting...")
    threading.Thread(target=search_job, args=(scene, keywords), daemon=True).start()


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        try:
            if url.path == "/":
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/api/state":
                with LOCK:
                    self.send_json(public_state())
            elif url.path.startswith("/media/"):
                parts = url.path.split("/")
                kind = parse_qs(url.query).get("kind", ["preview"])[0]
                with LOCK:
                    path = media_path(int(parts[2]), parts[3], kind)
                if path is None:
                    self.send_error(404)
                else:
                    self.serve_file(path)
            else:
                self.send_error(404)
        except (ConnectionResetError, BrokenPipeError):
            pass
        except Exception as exc:
            traceback.print_exc()
            self.send_json({"error": str(exc)}, 500)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/choose":
                choose(int(data["scene"]), str(data["choice"]))
                self.send_json({"ok": True})
            elif self.path == "/api/search":
                keywords = H.dedupe([str(k).strip() for k in data.get("keywords", []) if str(k).strip()])
                if not keywords:
                    raise ValueError("Type at least one keyword.")
                start_search(int(data["scene"]), keywords)
                self.send_json({"ok": True})
            else:
                self.send_error(404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, 400)

    def serve_file(self, path):
        size = os.path.getsize(path)
        start, end, code = 0, size - 1, 200
        rng = self.headers.get("Range", "")
        if rng.startswith("bytes="):
            first, _, last = rng[6:].partition("-")
            if first == "" and last:
                start = max(0, size - int(last))
            else:
                start = int(first or 0)
                end = int(last) if last else size - 1
            end = min(end, size - 1)
            code = 206
        length = max(0, end - start + 1)
        self.send_response(code)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (ConnectionResetError, BrokenPipeError):
                    return
                remaining -= len(chunk)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Pick your clips</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{color-scheme:dark;--bg:#14161a;--card:#1d2026;--line:#2e333c;--text:#e8eaee;--dim:#9aa3b2;--accent:#e0b04a;--good:#4fb477}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,Segoe UI,sans-serif}
header{position:sticky;top:0;z-index:5;background:#101216;border-bottom:1px solid var(--line);padding:12px 20px}
h1{margin:0;font-size:18px}.sub{color:var(--dim);font-size:13px;margin-top:2px}
main{max-width:1180px;margin:0 auto;padding:18px 20px 80px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:18px}
.head{display:flex;gap:10px;align-items:center}.tag{font-size:11px;letter-spacing:.06em;color:var(--dim);border:1px solid var(--line);border-radius:4px;padding:1px 6px}
.line{font-size:17px;margin:8px 0 12px}
.kw{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:14px}.kw label{color:var(--dim);font-size:13px}
.kw input{flex:1;min-width:240px;background:#12141a;border:1px solid var(--line);border-radius:6px;color:var(--text);padding:7px 9px;font:inherit}
button{background:#2a2f39;color:var(--text);border:1px solid #3a404c;border-radius:6px;padding:7px 12px;font:inherit;cursor:pointer}
button:hover:not(:disabled){background:#343a46}button:disabled{opacity:.45;cursor:default}
.opts{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:820px){.opts{grid-template-columns:1fr}}
.opt{border:2px solid var(--line);border-radius:8px;padding:10px;background:#181b21}.opt.sel{border-color:var(--accent)}
video{width:100%;border-radius:6px;background:#000;aspect-ratio:16/9}
.meta{font-size:12.5px;color:var(--dim);margin:8px 0}.meta b{color:var(--text);font-weight:600}
.pick{width:100%;font-weight:600}.opt.sel .pick{background:var(--accent);color:#1b1608;border-color:var(--accent)}
.empty{color:var(--dim);padding:22px;border:1px dashed var(--line);border-radius:8px;text-align:center}
.none{margin-top:12px;text-align:right}.none .pick{width:auto;font-weight:400}.none .pick.sel{background:var(--accent);color:#1b1608}
.msg{font-size:13px;color:var(--dim)}.msg.err{color:#e87a7a}.msg.ok{color:var(--good)}
.cov{margin-top:8px;color:var(--dim);font-size:13px}
</style></head><body>
<header><h1>Pick your clips</h1><div class="sub" id="sub">Loading...</div></header>
<main id="main"></main>
<script>
let S=null, polling=null;
const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
async function api(path,body){const r=await fetch(path,body?{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}:{});
  const d=await r.json(); if(!r.ok) throw new Error(d.error||"Request failed"); return d;}
async function load(){S=await api("/api/state"); render(); if(S.job.state==="running") poll();}
function optHtml(a,o){
  const sel=a.chosen===o.label, scores=o.relevance!=null?`relevance <b>${o.relevance}</b>/10 &middot; clutter <b>${o.clutter}</b>/10 &middot; quality <b>${o.quality}</b>/10`:"no visual score";
  return `<div class="opt ${sel?"sel":""}"><video src="/media/${a.scene}/${o.label}?kind=preview" controls loop muted playsinline preload="metadata"></video>
  <div class="meta"><b>Option ${o.label}</b> &middot; <a style="color:#8ab4f8" href="${esc(o.url)}" target="_blank" rel="noopener">${esc(o.title)}</a><br>
  starts at ${o.start??0}s (${esc(o.reason)})<br>${scores}${o.shows?`<br><i>${esc(o.shows)}</i>`:""}${o.covers?`<br>Chosen to show: <b>${esc(o.covers)}</b>`:""}</div>
  <button class="pick" onclick="choose(${a.scene},'${o.label}')">${sel?"Using "+o.label:"Use "+o.label}</button></div>`;}
function render(){
  document.getElementById("sub").textContent=`${S.game} - ${S.topic} - ${S.draft}. Choices save as you click. When finished, close this page and run build_video.py`;
  const running=S.job.state==="running";
  let html="";
  if(S.coverage.length) html+=`<div class="cov">Key subjects: ${S.coverage.map(c=>esc(c.subject)+": "+esc(c.status)+(c.scene?" (scene "+c.scene+")":"")).join(" &middot; ")}</div>`;
  for(const a of S.assets){
    const opts=a.options.length?`<div class="opts">${a.options.map(o=>optHtml(a,o)).join("")}</div>`:`<div class="empty">No clip found for this line. Try different keywords below.</div>`;
    const jobMsg=(S.job.scene===a.scene&&S.job.state!=="idle")?`<span class="msg ${S.job.state==="error"?"err":S.job.state==="done"?"ok":""}">${esc(S.job.message)}</span>`:"";
    html+=`<section class="card" id="scene-${a.scene}"><div class="head"><b>Scene ${a.scene}</b><span class="tag">${esc(a.section)}</span></div>
    <div class="line">${esc(a.narration)}</div>
    <div class="kw"><label for="kw-${a.scene}">Keywords</label><input id="kw-${a.scene}" value="${esc(a.keywords.join(", "))}" placeholder="e.g. Crones, Crookback Bog, hut">
    <button ${running?"disabled":""} onclick="search(${a.scene})">Search again</button>${jobMsg}</div>
    ${opts}<div class="none"><button class="pick ${a.chosen==="none"?"sel":""}" onclick="choose(${a.scene},'none')">${a.chosen==="none"?"Using neither":"Use neither (AI still or plain background)"}</button></div></section>`;}
  const keep=[...document.querySelectorAll("input[id^=kw-]")].reduce((m,i)=>(m[i.id]=i.value,m),{});
  const y=window.scrollY; document.getElementById("main").innerHTML=html;
  for(const [id,v] of Object.entries(keep)){const el=document.getElementById(id); if(el&&document.activeElement!==el) el.value=v;}
  window.scrollTo(0,y);
}
async function choose(scene,choice){try{await api("/api/choose",{scene,choice}); S=await api("/api/state"); render();}catch(e){alert(e.message);}}
async function search(scene){
  const kws=document.getElementById("kw-"+scene).value.split(/[,;]/).map(s=>s.trim()).filter(Boolean);
  try{await api("/api/search",{scene,keywords:kws}); S=await api("/api/state"); render(); poll();}catch(e){alert(e.message);}}
function poll(){ if(polling) return; polling=setInterval(async()=>{S=await api("/api/state");
  if(S.job.state!=="running"){clearInterval(polling);polling=null;} render();},2500);}
load();
</script></body></html>
"""


def main():
    load_manifest(find_manifest())
    server = None
    for port in range(FIRST_PORT, FIRST_PORT + 10):
        try:
            server = ThreadingHTTPServer((HOST, port), Handler)
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit("Could not open a local port. Close other programs using ports 8765-8774 and try again.")
    url = f"http://{HOST}:{server.server_address[1]}/"
    print(f"Manifest: {STATE['path'].name}\nReview page: {url}\n"
          "Pick A, B or neither for each line (saved as you click). Press Ctrl+C here when you're done.")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDone. Run build_video.py next.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
