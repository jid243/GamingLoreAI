"""
gaminglore_app.py  -  the whole GamingLoreAI pipeline in one window.

    python gaminglore_app.py

A page opens in your browser (it only runs on your own computer). From there you run each step with a button
and watch its log live:

  1. Research     collect wiki pages for a game + topic            (research_sources.py)
  2. Evidence     turn the research into numbered passages          (prepare_evidence.py)
  3. Script       write and fact-check the narration                (generate_grounded_script_google.py)
     Draft        read, edit and approve the script in the page
  4. Clips        type keywords per line, then find gameplay clips (harvest_clips_universal.py)
  5. Review       pick clip A or B for each line                    (review_clips.py, opens its own page)
  6. AI images    optional atmospheric stills via ComfyUI          (make_ai_inserts.py)

Nothing here replaces your scripts: each step runs the same script you can still run by itself. Close the page
and press Ctrl+C in this window to stop the app.
"""
import codecs
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = Path.home() / "Desktop" / "GamingLoreAI"
SCRIPT_DIR = PROJECT / "scripts"
SOURCES_DIR = PROJECT / "sources"
EVIDENCE_FILE = SOURCES_DIR / "evidence_passages.txt"
CHECK_DIR = PROJECT / "fact_checks"
MANIFEST_DIR = PROJECT / "clip_manifests"
STYLE_DIR = PROJECT / "style_examples"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat").rsplit("/api/", 1)[0]
HOST, FIRST_PORT = "127.0.0.1", 8780
MAX_LOG_LINES = 5000

STEPS = {
    "research": "research_sources.py",
    "evidence": "prepare_evidence.py",
    "script": "generate_grounded_script_google.py",
    "clips": "harvest_clips_universal.py",
    "review": "review_clips.py",
    "images": "make_ai_inserts.py",
    "benchmark": "benchmark_writers.py",
}


# ---------------------------------------------------------------------------
# One job at a time: a step's script runs as a child process; its output is kept for the page
# ---------------------------------------------------------------------------
class Job:
    def __init__(self):
        self.lock = threading.Lock()
        self.name, self.status, self.lines, self.proc, self.live = "", "idle", [], None, ""
        self.started = self.finished = self.last_output = 0.0
        self.stop_requested = False

    def snapshot(self, since=0):
        with self.lock:
            return {"name": self.name, "status": self.status, "total": len(self.lines), "live": self.live,
                    "lines": self.lines[since:], "started": self.started, "finished": self.finished,
                    "quiet_for": round(time.time() - self.last_output) if self.status == "running" and self.last_output else 0}

    def add(self, text):
        with self.lock:
            self.lines.append(text)
            if len(self.lines) > MAX_LOG_LINES:
                del self.lines[: len(self.lines) - MAX_LOG_LINES]

    def busy(self):
        return self.status == "running"


JOB = Job()


def child_env(extra=None):
    env = dict(os.environ)
    env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
    env.update({k: str(v) for k, v in (extra or {}).items() if v not in (None, "")})
    return env


class LineSplitter:
    """Turns a child's raw output into log lines. "\r\n" (how Windows ends every printed line) and "\n" end a
    line; a lone "\r" means a progress line rewriting itself, which is shown live instead of logged every tick."""
    def __init__(self, job):
        self.job, self.partial, self.cr = job, "", False
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def feed(self, raw):
        for ch in self.decoder.decode(raw):
            self.job.last_output = time.time()
            if ch == "\n":
                self.cr = False
                if self.partial.strip():
                    self.job.add(self.partial.rstrip())
                self.partial, self.job.live = "", ""
                continue
            if ch == "\r":
                self.cr = True
                if self.partial.strip():
                    self.job.live = self.partial.strip()
            else:
                if self.cr:                  # "\r" not followed by "\n": the line is being redrawn
                    self.cr, self.partial = False, ""
                self.partial += ch
                if len(self.partial) % 20 == 0 or ch in ".:)]":
                    self.job.live = self.partial.strip()

    def close(self):
        if self.partial.strip():
            self.job.add(self.partial.rstrip())
        self.partial, self.job.live = "", ""


def run_commands(name, commands):
    """commands: [(label, argv, env_extra)]. Runs them one after another; stops at the first failure."""
    with JOB.lock:
        JOB.name, JOB.status, JOB.lines, JOB.proc = name, "running", [], None
        JOB.started, JOB.finished, JOB.stop_requested = time.time(), 0.0, False
        JOB.last_output = time.time()

    def worker():
        ok = True
        for label, argv, env_extra in commands:
            if JOB.stop_requested:
                ok = False
                break
            JOB.add(f"=== {label}: {' '.join(Path(a).name if i < 2 else a for i, a in enumerate(argv))}")
            try:
                # binary pipe: text mode would turn a progress line's "\r" into a new line and flood the log
                proc = subprocess.Popen(argv, cwd=str(HERE), env=child_env(env_extra), stdin=subprocess.DEVNULL,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
            except OSError as error:
                JOB.add(f"Could not start: {error}")
                ok = False
                break
            JOB.proc = proc
            JOB.last_output = time.time()
            splitter = LineSplitter(JOB)
            while True:
                raw = proc.stdout.read(1)
                if not raw and proc.poll() is not None:
                    break
                splitter.feed(raw)
            splitter.close()
            code = proc.wait()
            JOB.add(f"=== {label} finished (exit code {code})")
            if code != 0:
                ok = False
                break
        with JOB.lock:
            JOB.status = "stopped" if JOB.stop_requested else ("done" if ok else "failed")
            JOB.finished, JOB.proc, JOB.live = time.time(), None, ""

    threading.Thread(target=worker, daemon=True).start()


def stop_job():
    JOB.stop_requested = True
    proc = JOB.proc
    if proc and proc.poll() is None:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            proc.kill()


def script(step):
    return [sys.executable, "-u", str(HERE / STEPS[step])]


# ---------------------------------------------------------------------------
# Building each step's command from the form
# ---------------------------------------------------------------------------
def research_commands(p):
    argv = script("research") + ["--no-prompt", "--game", p.get("game", "").strip(), "--topic", p.get("topic", "").strip()]
    for url in re.split(r"\s+", p.get("urls", "")):
        if url.startswith(("http://", "https://")):
            argv += ["--url", url]
    if p.get("wiki", "").strip():
        argv += ["--wiki", p["wiki"].strip(), "--pick", p.get("pick", "").strip() or "top3"]
        if p.get("search", "").strip():
            argv += ["--search", p["search"].strip()]
    return [("Research", argv, None)]


def script_env(p):
    env = {}
    if p.get("model", "").strip():
        env["OLLAMA_MODEL"] = p["model"].strip()
    if p.get("quick"):
        env.update({"SCRIPT_CANDIDATES": "1", "MAX_ATTEMPTS": "3"})
    if p.get("max_words", "").strip().isdigit():
        env["SCRIPT_MAX_WORDS"] = p["max_words"].strip()
    versions = str(p.get("versions", "")).strip()
    if versions.isdigit() and 1 <= int(versions) <= 6:
        env["SCRIPT_VERSIONS"] = versions
    if p.get("writer_mode") in ("classic", "lines"):
        env["WRITER_MODE"] = p["writer_mode"]
    if (p.get("writer_model") or "").strip():
        env["LOCAL_WRITER_MODEL"] = p["writer_model"].strip()
    if p.get("online_writers"):   # off unless ticked (the generator's own default)
        env["USE_ONLINE_WRITERS"] = "1"
    return env


def build_commands(step, p):
    if step == "research":
        if not p.get("game", "").strip() or not p.get("topic", "").strip():
            raise ValueError("Enter a game and a topic first.")
        return research_commands(p)
    if step == "evidence":
        return [("Evidence", script("evidence"), None)]
    if step == "hooks":   # hooks only: you pick one in the Hooks tab, then the script is written for it
        return [("Hooks", script("script"), {**script_env(p), "HOOKS_ONLY": "1"})]
    if step == "script":
        env = script_env(p)
        if (p.get("chosen_hook") or "").strip():
            env["CHOSEN_HOOK"] = p["chosen_hook"].strip()
        return [("Script" + (" for your hook" if "CHOSEN_HOOK" in env else ""), script("script"), env)]
    if step == "story":   # research -> evidence -> script, then stop so you can read the draft
        return build_commands("research", p) + build_commands("evidence", p) + build_commands("script", p)
    if step == "clips":
        argv = script("clips") + (["--no-coverage"] if p.get("no_coverage") else [])
        bar = str(p.get("min_relevance", "")).strip()
        env = {"HARVEST_MIN_RELEVANCE": bar} if bar.isdigit() and 1 <= int(bar) <= 10 else None
        return [("Clips", argv, env)]
    if step == "review":
        return [("Review clips (a page opens; press Stop here when you have finished choosing)", script("review"), None)]
    if step == "benchmark":   # every writer on the same topics, hooks and evidence; the report lands in benchmark/
        return [("Benchmark (about an hour; your project is restored afterwards)", script("benchmark"), None)]
    if step == "images":
        argv = script("images") + (["--new-prompts"] if p.get("new_prompts") else []) + (["--force"] if p.get("force") else [])
        return [("AI images", argv, None)]
    raise ValueError(f"Unknown step: {step}")


# ---------------------------------------------------------------------------
# Project state for the page
# ---------------------------------------------------------------------------
def read_meta(text, key):
    m = re.search(rf"(?mi)^\s*{re.escape(key)}\s*:\s*(.+?)\s*$", text or "")
    return m.group(1).strip() if m else ""


def newest(paths):
    paths = [p for p in paths if p.is_file()]
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def slug(text):   # the generator's file names: same rule, same 50-character cut
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:50] or "gaming_lore"


def script_versions(game, topic):
    """Every finished version of the last script run for this game and topic."""
    if not game or not CHECK_DIR.exists():
        return {}
    try:
        return json.loads((CHECK_DIR / f"{slug(f'{game}_{topic}')}_script_menu.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def use_version(number, game, topic):
    """Makes the version you picked the current draft, with its own fact-check report, so its status is real: a
    version that passed is saved REVIEW_REQUIRED, one that did not is saved REJECTED (you can still approve it)."""
    menu = script_versions(game, topic)
    version = next((v for v in menu.get("versions", []) if v.get("number") == number), None)
    if not version:
        raise ValueError("That version is no longer available. Write the script again.")
    stem = f"{slug(f'{game}_{topic}')}_{menu.get('timestamp', 'picked')}_v{number}"
    suffix = "REVIEW_REQUIRED" if version.get("eligible") else "REJECTED"
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    target = SCRIPT_DIR / f"{stem}_{suffix}.txt"
    target.write_text(version["draft"].strip() + "\n", encoding="utf-8")
    spoken = [re.sub(r"(?:\s*\[E\d+\])+\s*$", "", m.group(2)).strip()
              for m in re.finditer(r"(?m)^\s*(HOOK|NARRATION|PAYOFF|ENDING QUESTION)\s*:\s*(.+)$", version["draft"])]
    (SCRIPT_DIR / f"{stem}_NARRATION.txt").write_text("\n".join(spoken) + "\n", encoding="utf-8")
    (CHECK_DIR / f"{stem}_review.txt").write_text(version.get("review", "") + "\n", encoding="utf-8")
    return target.name


def hook_menu(game, topic):
    """The verified hooks of the last hooks run for this game and topic, best first."""
    if not game or not CHECK_DIR.exists():
        return []
    path = CHECK_DIR / f"{slug(f'{game}_{topic}')}_hook_menu.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("hooks", [])
    except (OSError, ValueError):
        return []


def latest_draft():
    return newest(list(SCRIPT_DIR.glob("*_REVIEW_REQUIRED.txt")) + list(SCRIPT_DIR.glob("*_REJECTED.txt"))) \
        if SCRIPT_DIR.exists() else None


def latest_review(draft):
    if not draft or not CHECK_DIR.exists():
        return None
    stem = re.sub(r"_(REVIEW_REQUIRED|REJECTED)$", "", draft.stem)
    path = CHECK_DIR / f"{stem}_review.txt"
    return path if path.exists() else None


def ollama_models():
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as resp:
            return [m.get("name", "") for m in json.loads(resp.read().decode("utf-8")).get("models", [])]
    except Exception:
        return None


KEY_SERVICES = {   # free writing services: where the key is kept, and a cheap request that only checks the key
    "gemini": {"file": "gemini_api_key.txt", "name": "Gemini", "site": "aistudio.google.com",
               "test": "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash", "header": lambda k: {"x-goog-api-key": k}},
    "groq": {"file": "groq_api_key.txt", "name": "Groq", "site": "console.groq.com",
             "test": "https://api.groq.com/openai/v1/models", "header": lambda k: {"Authorization": f"Bearer {k}"}},
}


def key_file(service="gemini"):
    return PROJECT / KEY_SERVICES[service]["file"]


def _saved_key(service):
    try:
        return key_file(service).read_text(encoding="utf-8-sig").strip()
    except OSError:
        return ""


def key_status(service="gemini"):
    """Whether a key is saved; only its last 4 characters ever leave the file (for the page)."""
    key = _saved_key(service)
    return {"set": bool(key), "hint": ("…" + key[-4:]) if len(key) >= 8 else ""}


def save_key(service, key):
    """Save (or, when blank, remove) a writing-service key. The generator reads the same file."""
    if service not in KEY_SERVICES:
        raise ValueError(f"Unknown service {service!r}.")
    key = re.sub(r"\s+", "", key or "")                                # spaces or line breaks picked up when copying
    key = re.sub(r"^(?:GEMINI_API_KEY|GROQ_API_KEY|GOOGLE_API_KEY|API_KEY|KEY)[:=]", "", key, flags=re.I).strip("\"'`")
    path = key_file(service)
    if not key:
        if path.exists():
            path.unlink()
        return key_status(service)
    # services use more than one key format, so only obvious non-keys are refused; Test key asks the service itself
    if not 20 <= len(key) <= 400 or not key.isprintable():
        raise ValueError(f"That doesn't look like a {KEY_SERVICES[service]['name']} API key (it is {len(key)} characters). "
                         f"Copy the whole key from {KEY_SERVICES[service]['site']} and paste it again.")
    path.write_text(key, encoding="utf-8")
    return key_status(service)


def test_key(service="gemini"):
    """Ask the service whether the saved key works (a request that writes nothing and uses no writing quota)."""
    info = KEY_SERVICES[service]
    key = _saved_key(service)
    if not key:
        return {"ok": False, "message": "No key saved yet."}
    req = urllib.request.Request(info["test"], headers={**info["header"](key), "User-Agent": "GamingLoreAI"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return {"ok": resp.status == 200, "message": f"The {info['name']} key works."}
    except urllib.error.HTTPError as error:
        if error.code in (400, 401, 403):
            return {"ok": False, "message": f"{info['name']} rejected this key. Copy it again from {info['site']} and save it."}
        if error.code == 404:
            return {"ok": False, "message": f"The key works, but {info['name']} does not offer the model it needs."}
        if error.code == 429:
            return {"ok": True, "message": f"The {info['name']} key works, but its free limit is used up for now."}
        return {"ok": False, "message": f"{info['name']} answered {error.code}. Try again in a minute."}
    except Exception:
        return {"ok": False, "message": f"Could not reach {info['name']}. Check your internet connection."}


# the first version of the page only knew Gemini
gemini_key_file = lambda: key_file("gemini")
gemini_status = lambda: key_status("gemini")
save_gemini_key = lambda key: save_key("gemini", key)
test_gemini_key = lambda: test_key("gemini")


def which(name):
    from shutil import which as _which
    return bool(_which(name))


def project_state():
    research = newest(SOURCES_DIR.glob("*_research_*.txt")) if SOURCES_DIR.exists() else None
    research_text = research.read_text(encoding="utf-8-sig", errors="replace")[:3000] if research else ""
    evidence_text = EVIDENCE_FILE.read_text(encoding="utf-8-sig", errors="replace") if EVIDENCE_FILE.exists() else ""
    draft = latest_draft()
    review = latest_review(draft)
    manifest = newest(MANIFEST_DIR.glob("*_manifest.json")) if MANIFEST_DIR.exists() else None
    return {
        "project": str(PROJECT),
        "gemini": key_status("gemini"),
        "keys": {name: key_status(name) for name in KEY_SERVICES},
        "research": {"file": research.name if research else "", "game": read_meta(research_text, "GAME"),
                     "topic": read_meta(research_text, "TOPIC")},
        "evidence": {"exists": bool(evidence_text), "passages": len(re.findall(r"(?m)^\[E\d+\]", evidence_text)),
                     "game": read_meta(evidence_text, "GAME"), "topic": read_meta(evidence_text, "TOPIC")},
        "hooks": hook_menu(read_meta(research_text, "GAME"), read_meta(research_text, "TOPIC")),
        "versions": script_versions(read_meta(research_text, "GAME"), read_meta(research_text, "TOPIC")).get("versions", []),
        "draft": {"file": draft.name if draft else "",
                  "other_topic": bool(draft and read_meta(research_text, "GAME") and not draft.stem.startswith(
                      slug(f"{read_meta(research_text, 'GAME')}_{read_meta(research_text, 'TOPIC')}") + "_")),
                  "status": ("REVIEW_REQUIRED" if draft and draft.stem.endswith("REVIEW_REQUIRED") else "REJECTED") if draft else "",
                  "text": draft.read_text(encoding="utf-8-sig", errors="replace") if draft else "",
                  "review": review.read_text(encoding="utf-8-sig", errors="replace")[:20000] if review else ""},
        "manifest": manifest.name if manifest else "",
        "models": ollama_models(),
        "tools": {"yt-dlp": which("yt-dlp") or Path(r"C:\pinokio\api\comfy.git\app\env\Scripts\yt-dlp.exe").exists(),
                  "ffmpeg": which("ffmpeg") or Path(r"C:\pinokio\bin\ffmpeg.exe").exists()},
        "missing_scripts": [f for f in STEPS.values() if not (HERE / f).exists()],
    }


# ---------------------------------------------------------------------------
# Draft editing / approval
# ---------------------------------------------------------------------------
def save_draft(text, approve):
    """Saves your edits. Approving a REJECTED draft writes a REVIEW_REQUIRED copy (the clip finder only uses
    approved drafts); the rejected original is kept for reference."""
    draft = latest_draft()
    if not draft:
        raise ValueError("There is no draft yet. Run the Script step first.")
    if not re.search(r"(?m)^\s*HOOK\s*:", text) or not re.search(r"(?m)^\s*PAYOFF\s*:", text):
        raise ValueError("The draft needs its HOOK: and PAYOFF: lines so the clip finder can read it.")
    target = draft
    if approve and draft.stem.endswith("_REJECTED"):
        target = draft.with_name(draft.name.replace("_REJECTED.txt", "_REVIEW_REQUIRED.txt"))
    target.write_text(text.strip() + "\n", encoding="utf-8")
    narration = re.sub(r"_(REVIEW_REQUIRED|REJECTED)$", "_NARRATION", target.stem)
    spoken = [re.sub(r"(?:\s*\[E\d+\])+\s*$", "", m.group(2)).strip()
              for m in re.finditer(r"(?m)^\s*(HOOK|NARRATION|PAYOFF|ENDING QUESTION)\s*:\s*(.+)$", text)]
    (target.parent / f"{narration}.txt").write_text("\n".join(spoken) + "\n", encoding="utf-8")
    return target.name


STYLE_TAGS = {"HOOK": "hook", "NARRATION": "story", "PAYOFF": "payoff", "ENDING QUESTION": "question"}


def save_style_example(text, game, topic):
    """Keeps a finished script as a style example: the script writer is shown it (for other topics) as the level
    of storytelling to match. Saved without citations, with neutral [hook]/[story] tags."""
    game, topic = (game or "").strip(), (topic or "").strip()
    if not game or not topic:
        raise ValueError("Fill in the Game and Topic boxes (Research card) so the example is labelled.")
    lines = [f"[{STYLE_TAGS[m.group(1)]}] " + re.sub(r"(?:\s*\[E\d+\])+\s*$", "", m.group(2)).strip()
             for m in re.finditer(r"(?m)^\s*(HOOK|NARRATION|PAYOFF|ENDING QUESTION)\s*:\s*(.+)$", text)]
    if not any(l.startswith("[hook]") for l in lines) or not any(l.startswith("[question]") for l in lines):
        raise ValueError("The script needs its HOOK: and ENDING QUESTION: lines.")
    STYLE_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", f"{game}_{topic}".lower()).strip("_")
    path = STYLE_DIR / f"{slug}.txt"
    path.write_text(f"GAME: {game}\nTOPIC: {topic}\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return path.name


# ---------------------------------------------------------------------------
# Clip keywords: suggested by the clip finder's own logic, edited in the page, saved where it reads them
# ---------------------------------------------------------------------------
def harvest_module():
    sys.path.insert(0, str(HERE))
    import harvest_clips_universal as H
    return H


def keyword_scenes():
    H = harvest_module()
    draft = H.find_latest_approved_draft()
    text = draft.read_text(encoding="utf-8-sig")
    evidence_text = EVIDENCE_FILE.read_text(encoding="utf-8-sig") if EVIDENCE_FILE.exists() else ""
    game = H.read_metadata(text, "GAME") or H.read_metadata(evidence_text, "GAME")
    topic = H.read_metadata(text, "TOPIC") or H.read_metadata(evidence_text, "TOPIC")
    spoken = H.extract_spoken_lines(text)
    lines = []
    for section, sentence in spoken:
        source_line = next((ln for ln in text.splitlines() if ln.strip().startswith(section + ":") and sentence[:30] in ln), "")
        tail = re.search(r"((?:\s*\[E\d+\])+)\s*$", source_line)
        lines.append((section, sentence, re.findall(r"E\d+", tail.group(1)) if tail else []))
    ctx = {"lines": lines, "known": H.collect_known([evidence_text, topic] + [s for _, s, _ in lines]),
           "game": game, "evidence_text": evidence_text}
    evid_counts = Counter(re.findall(r"[a-z]{5,}", evidence_text.casefold()))
    saved = H.load_saved_keywords(draft, [(a, b) for a, b, _ in lines])
    scenes = []
    for i, (section, sentence, _) in enumerate(lines, 1):
        suggested = H.suggest_keywords(sentence, H.scene_entities(ctx, i), evid_counts, game)
        scenes.append({"scene": i, "section": section, "sentence": sentence, "suggested": suggested,
                       "keywords": saved.get(i) or suggested})
    return {"draft": draft.name, "game": game, "topic": topic, "scenes": scenes}


def save_scene_keywords(scenes):
    H = harvest_module()
    draft = H.find_latest_approved_draft()
    lines = H.extract_spoken_lines(draft.read_text(encoding="utf-8-sig"))
    by_scene = {}
    for item in scenes:
        words = [k.strip() for k in re.split(r"[,;]", str(item.get("keywords", ""))) if k.strip()] \
            if not isinstance(item.get("keywords"), list) else [str(k).strip() for k in item["keywords"] if str(k).strip()]
        if words:
            by_scene[int(item["scene"])] = words
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    H.save_keywords(draft, lines, by_scene)
    return len(by_scene)


def open_folder(name):
    folders = {"scripts": SCRIPT_DIR, "fact_checks": CHECK_DIR, "clips": PROJECT / "comfyui_input_clips",
               "sources": SOURCES_DIR, "project": PROJECT, "benchmark": PROJECT / "benchmark"}
    path = folders.get(name)
    if not path:
        raise ValueError("Unknown folder")
    path.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(str(path))      # noqa: S606 - opens Explorer on your own PC
    else:
        subprocess.Popen(["xdg-open", str(path)])
    return str(path)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, code, body, kind="application/json"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", f"{kind}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def json(self, obj, code=200):
        self.send(code, json.dumps(obj, ensure_ascii=False))

    def body(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return {}

    def do_GET(self):
        path, _, query = self.path.partition("?")
        try:
            if path == "/":
                return self.send(200, PAGE, "text/html")
            if path == "/api/state":
                return self.json(project_state())
            if path == "/api/log":
                since = int((re.search(r"since=(\d+)", query) or [0, 0])[1])
                return self.json(JOB.snapshot(since))
            if path == "/api/keywords":
                return self.json(keyword_scenes())
            return self.send(404, '{"error": "not found"}')
        except Exception as error:   # the page shows the message instead of a dead request
            return self.json({"error": str(error)}, 400)

    def do_POST(self):
        payload = self.body()
        try:
            if self.path == "/api/run":
                if JOB.busy():
                    return self.json({"error": f"'{JOB.name}' is still running. Wait for it or press Stop."}, 409)
                step = payload.get("step", "")
                run_commands(step, build_commands(step, payload.get("params") or {}))
                return self.json({"ok": True})
            if self.path == "/api/stop":
                stop_job()
                return self.json({"ok": True})
            if self.path == "/api/draft":
                return self.json({"ok": True, "file": save_draft(payload.get("text", ""), bool(payload.get("approve")))})
            if self.path == "/api/version":
                return self.json({"ok": True, "file": use_version(int(payload.get("number", 0)), payload.get("game"), payload.get("topic"))})
            if self.path == "/api/style":
                return self.json({"ok": True, "file": save_style_example(payload.get("text", ""), payload.get("game"), payload.get("topic"))})
            if self.path == "/api/keywords":
                return self.json({"ok": True, "saved": save_scene_keywords(payload.get("scenes") or [])})
            if self.path == "/api/key":
                return self.json({"ok": True, **save_key(payload.get("service", "gemini"), payload.get("key", ""))})
            if self.path == "/api/key_test":
                return self.json(test_key(payload.get("service", "gemini")))
            if self.path == "/api/gemini_key":
                return self.json({"ok": True, **save_key("gemini", payload.get("key", ""))})
            if self.path == "/api/gemini_test":
                return self.json(test_key("gemini"))
            if self.path == "/api/open":
                return self.json({"ok": True, "path": open_folder(payload.get("folder", ""))})
            return self.send(404, '{"error": "not found"}')
        except Exception as error:
            return self.json({"error": str(error)}, 400)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GamingLoreAI</title>
<style>
:root{--bg:#f6f5f2;--panel:#fff;--ink:#1d1d1f;--muted:#6b6b70;--line:#e3e1dc;--accent:#2f6f4f;--accent-ink:#fff;--warn:#a15c00;--bad:#a12a2a;--code:#f0efeb}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--panel:#1d1d20;--ink:#ececee;--muted:#9a9aa1;--line:#2c2c31;--accent:#4fa57a;--accent-ink:#0d1a13;--warn:#e0a24a;--bad:#e06a6a;--code:#16161a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:18px 24px;border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}
header h1{font-size:19px;margin:0}header .meta{color:var(--muted);font-size:13px}
main{display:grid;grid-template-columns:minmax(320px,440px) 1fr;gap:18px;padding:18px 24px;align-items:start}
@media (max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:12px}
.card h2{font-size:15px;margin:0 0 4px;display:flex;align-items:center;gap:8px}.num{display:inline-grid;place-items:center;width:22px;height:22px;border-radius:50%;background:var(--code);font-size:12px}
.hint{color:var(--muted);font-size:13px;margin:0 0 10px}
label{display:block;font-size:13px;color:var(--muted);margin:8px 0 3px}
input[type=text],input[type=password],textarea,select{width:100%;padding:7px 9px;border:1px solid var(--line);border-radius:7px;background:var(--bg);color:var(--ink);font:inherit}
textarea{min-height:70px;resize:vertical}.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:10px}
button{border:1px solid var(--line);background:var(--bg);color:var(--ink);padding:7px 12px;border-radius:7px;font:inherit;cursor:pointer}
button.go{background:var(--accent);border-color:var(--accent);color:var(--accent-ink);font-weight:600}button:disabled{opacity:.5;cursor:not-allowed}
.check{display:flex;gap:6px;align-items:center;font-size:13px;color:var(--ink);margin-top:8px}
.status{font-size:13px;margin-top:6px}.ok{color:var(--accent)}.warn{color:var(--warn)}.bad{color:var(--bad)}
#log{background:var(--code);border:1px solid var(--line);border-radius:10px;padding:12px;height:46vh;overflow:auto;font:12.5px/1.45 ui-monospace,Consolas,monospace;white-space:pre-wrap}
#draft{min-height:300px;font:13px/1.5 ui-monospace,Consolas,monospace}
.tabs{display:flex;gap:6px;margin-bottom:10px}.tabs button.on{background:var(--ink);color:var(--bg)}
.scene{border-top:1px solid var(--line);padding:10px 0}.scene:first-child{border-top:0}.scene .tag{font-size:12px;color:var(--muted)}
.jobbar{display:flex;gap:10px;align-items:center;margin-bottom:8px}.hook{display:flex;gap:10px;align-items:flex-start;padding:10px;border:1px solid var(--line);border-radius:8px;margin-bottom:8px;cursor:pointer}.hook:hover{border-color:var(--accent)}.hook.sel{border-color:var(--accent);background:var(--code)}.hook input{margin-top:4px}.hook .txt{font-size:15px}.hook .sc{color:var(--muted);font-size:12px;margin-top:3px}textarea.short{min-height:70px;height:70px}
.version{border:1px solid var(--line);border-radius:8px;padding:12px;margin-bottom:10px}.version.star{border-color:var(--accent)}.version .vh{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:6px}.version pre{white-space:pre-wrap;font-family:inherit;font-size:14px;margin:6px 0;line-height:1.45}.version .vnote{color:var(--muted);font-size:12px}
.pill{font-size:12px;padding:2px 8px;border-radius:99px;background:var(--code)}.pill.run::before{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--accent);margin-right:6px;vertical-align:1px;animation:pulse 1s ease-in-out infinite}@keyframes pulse{50%{opacity:.15}}#live{font-family:ui-monospace,Consolas,monospace;font-size:12px;margin-top:6px;min-height:1.2em;white-space:pre-wrap;word-break:break-word}#quiet{color:var(--muted);font-size:13px}
pre.review{white-space:pre-wrap;font:12px/1.45 ui-monospace,Consolas,monospace;max-height:40vh;overflow:auto;background:var(--code);padding:10px;border-radius:8px}
</style></head><body>
<header><h1>GamingLoreAI</h1><span class="meta" id="meta">loading…</span></header>
<main>
<section>
  <div class="card"><h2><span class="num">1</span>Research</h2>
    <p class="hint">Collects wiki pages for the game and topic.</p>
    <label>Game</label><input type="text" id="game" placeholder="e.g. The Witcher 3">
    <label>Topic</label><input type="text" id="topic" placeholder="e.g. Botchling">
    <label>Fandom wiki to search (optional)</label><input type="text" id="wiki" placeholder="e.g. witcher.fandom.com">
    <label>Pages to add from the search</label><select id="pick"><option value="top3">Top 3</option><option value="top5">Top 5</option><option value="all">All</option></select>
    <label>Page links (optional, one per line)</label><textarea id="urls" placeholder="https://..."></textarea>
    <div class="row"><button class="go" data-run="research">Run research</button><button data-run="story" title="Research, then evidence, then script">Research → Evidence → Script</button></div>
  </div>
  <div class="card"><h2><span class="num">2</span>Evidence</h2>
    <p class="hint">Turns every research file for this game and topic into numbered passages.</p>
    <div class="status" id="evstatus"></div>
    <div class="row"><button class="go" data-run="evidence">Prepare evidence</button><button data-open="sources">Open sources folder</button></div>
  </div>
  <div class="card"><h2><span class="num">3</span>Script</h2>
    <p class="hint"><b>Write hooks</b> lets you pick the opening yourself (Hooks tab), then writes the script for it. <b>Write script</b> picks the hook for you. The draft appears on the right.</p>
    <label>Optional free online writers (only used when the box below is ticked): Gemini first, Groq when Gemini is busy.</label>
    <div class="row" style="flex-wrap:nowrap"><input type="password" id="key_gemini" style="flex:1;min-width:0" autocomplete="off" placeholder="Gemini key from aistudio.google.com"><button data-savekey="gemini">Save</button><button data-testkey="gemini">Test</button></div>
    <div class="row" style="flex-wrap:nowrap"><input type="password" id="key_groq" style="flex:1;min-width:0" autocomplete="off" placeholder="Groq key from console.groq.com"><button data-savekey="groq">Save</button><button data-testkey="groq">Test</button></div>
    <div class="status" id="keymsg"></div>
    <label>Local model for research and fact checks (blank = the generator's default)</label><select id="model"></select>
    <label>Writer</label><select id="writer_mode"><option value="">Classic (the original writer)</option><option value="lines">Line by line (the code holds the shape; the model writes one line at a time)</option></select>
    <label>Writing model (blank = the same as above)</label><select id="writer_model"></select>
    <label class="check"><input type="checkbox" id="online_writers"> Use the online writers (Gemini/Groq keys above). Off: everything runs on this PC</label>
    <label>Word limit (blank = 160, about 60 seconds)</label><input type="text" id="max_words" placeholder="160">
    <label>Versions to write for a picked hook (blank = 3)</label><input type="text" id="versions" placeholder="3">
    <label class="check"><input type="checkbox" id="quick"> Quick run (one script, up to 3 attempts)</label>
    <div class="row"><button class="go" data-run="hooks">Write hooks</button><button data-run="script">Write script</button><button data-open="fact_checks">Open fact checks</button></div>
    <div class="row"><button data-run="benchmark">Benchmark the writers</button><button data-open="benchmark">Open benchmark results</button></div>
  </div>
  <div class="card"><h2><span class="num">4</span>Clips</h2>
    <p class="hint">Approve a draft first. Then set the search keywords per line (Keywords tab) and find clips.</p>
    <label>Keep clips scoring at least (relevance out of 10; blank = 7)</label><input type="text" id="min_relevance" placeholder="7">
    <label class="check"><input type="checkbox" id="no_coverage"> Skip the key-subject coverage check (faster)</label>
    <div class="row"><button class="go" data-run="clips">Find clips</button><button data-open="clips">Open clips folder</button></div>
  </div>
  <div class="card"><h2><span class="num">5</span>Review clips</h2>
    <p class="hint">Opens the A/B picker in a new tab. Press Stop here when you have finished choosing.</p>
    <div class="row"><button class="go" data-run="review">Open clip review</button></div>
  </div>
  <div class="card"><h2><span class="num">6</span>AI images <span class="pill">optional</span></h2>
    <p class="hint">Atmospheric stills via your local ComfyUI (start ComfyUI first).</p>
    <label class="check"><input type="checkbox" id="new_prompts"> Write fresh prompts</label>
    <label class="check"><input type="checkbox" id="force"> Regenerate existing images</label>
    <div class="row"><button class="go" data-run="images">Make AI images</button></div>
  </div>
</section>
<section>
  <div class="card">
    <div class="jobbar"><strong id="jobname">No step running</strong><span class="pill" id="jobstatus">idle</span><span class="meta" id="jobtime"></span><span id="quiet"></span><span style="flex:1"></span><button id="stop">Stop</button></div>
    <div id="log"></div><div class="meta" id="live"></div>
  </div>
  <div class="card">
    <div class="tabs"><button data-tab="hookpane">Hooks</button><button data-tab="versionpane" id="versiontab">Versions</button><button class="on" data-tab="draftpane">Draft</button><button data-tab="kwpane">Keywords</button><button data-tab="reviewpane">Fact-check report</button></div>
    <div id="hookpane" hidden><p class="hint">Verified hooks from the last <b>Write hooks</b> run, best first (★ = the generator's pick). Choose one, edit it if you like, then write the script for it. An edited hook is fact-checked like any other.</p>
      <div id="hooklist"></div>
      <label>Hook to use</label><textarea id="chosenhook" class="short" spellcheck="true" placeholder="Pick a hook above or type your own"></textarea>
      <div class="row"><button class="go" id="usehook">Write script with this hook</button><span class="status" id="hookmsg"></span></div>
    </div>
    <div id="versionpane" hidden><p class="hint">Every finished version from the last script run (★ = the one the generator chose, already in the Draft tab). Read them, then press <b>Use this version</b> on the one you want.</p><div id="versionlist"></div></div>
    <div id="draftpane"><div class="meta" id="draftname">No draft yet.</div><div class="status" id="versionhint"></div>
      <textarea id="draft" spellcheck="true"></textarea>
      <div class="row"><button id="savedraft">Save edits</button><button class="go" id="approvedraft">Approve for clips</button><button id="styledraft" title="Use this script as the standard the script writer aims for (on other topics)">Keep as style example</button><span class="status" id="draftmsg"></span></div>
    </div>
    <div id="kwpane" hidden><p class="hint">Comma-separated search words for each line. 3 to 5 work best: names, quest titles, distinctive places or objects, most specific first. Every keyword is searched alone and paired with the others.</p><div id="scenes"></div>
      <div class="row"><button class="go" id="savekw">Save keywords</button><button id="reloadkw">Reload</button><span class="status" id="kwmsg"></span></div></div>
    <div id="reviewpane" hidden><pre class="review" id="review">No report yet.</pre></div>
  </div>
</section>
</main>
<script>
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
let since=0, state=null, draftDirty=false;
const remembered=JSON.parse((()=>{try{return localStorage.getItem('glai')||'{}'}catch(e){return '{}'}})());
['game','topic','wiki','urls','max_words','min_relevance','versions','writer_mode'].forEach(id=>{if(remembered[id])$('#'+id).value=remembered[id]});
function remember(){const o={};['game','topic','wiki','urls','max_words','min_relevance','versions','writer_mode'].forEach(id=>o[id]=$('#'+id).value);try{localStorage.setItem('glai',JSON.stringify(o))}catch(e){}}
async function api(path,body){const r=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});const j=await r.json();if(!r.ok||j.error)throw new Error(j.error||r.statusText);return j}
function params(){remember();return{game:$('#game').value,topic:$('#topic').value,wiki:$('#wiki').value,pick:$('#pick').value,urls:$('#urls').value,
 model:$('#model').value,writer_mode:$('#writer_mode').value,writer_model:$('#writer_model').value,online_writers:$('#online_writers').checked,max_words:$('#max_words').value,versions:$('#versions').value,quick:$('#quick').checked,no_coverage:$('#no_coverage').checked,min_relevance:$('#min_relevance').value,new_prompts:$('#new_prompts').checked,force:$('#force').checked}}
$$('[data-run]').forEach(b=>b.onclick=async()=>{try{await api('/api/run',{step:b.dataset.run,params:params()});since=0;$('#log').textContent=''}catch(e){alert(e.message)}});
$$('[data-open]').forEach(b=>b.onclick=()=>api('/api/open',{folder:b.dataset.open}).catch(e=>alert(e.message)));
$('#stop').onclick=()=>api('/api/stop',{});
const KEYNAMES={gemini:['Gemini','aistudio.google.com'],groq:['Groq','console.groq.com']};
function keyShow(keys){if(!keys)return;for(const k in KEYNAMES){const i=$('#key_'+k);if(i)i.placeholder=keys[k].set?`${KEYNAMES[k][0]} key saved (${keys[k].hint}). Paste a new one to replace it, or save blank to remove it.`:`${KEYNAMES[k][0]} key from ${KEYNAMES[k][1]}`}
 const m=$('#keymsg');if(m.dataset.busy)return;const w=['gemini','groq'].filter(k=>keys[k].set).map(k=>KEYNAMES[k][0]);
 m.innerHTML=w.length?`<span class="ok">Writers: ${w.join(' → ')} → your PC's model</span>`:'<span class="warn">Writer: your PC\'s model only (add a free key above)</span>'}
function keyMsg(text,cls){const m=$('#keymsg');m.dataset.busy=1;m.textContent=text;m.className='status '+(cls||'');clearTimeout(m._t);m._t=setTimeout(()=>{delete m.dataset.busy},8000)}
$$('[data-savekey]').forEach(b=>b.onclick=async()=>{const k=b.dataset.savekey,i=$('#key_'+k);try{const j=await api('/api/key',{service:k,key:i.value});i.value='';keyMsg(j.set?`${KEYNAMES[k][0]} key saved on this PC. Press Test to check it.`:`${KEYNAMES[k][0]} key removed.`,'ok');refresh()}catch(e){keyMsg(e.message,'bad')}});
$$('[data-testkey]').forEach(b=>b.onclick=async()=>{const k=b.dataset.testkey;keyMsg(`Checking with ${KEYNAMES[k][0]}…`);try{const j=await api('/api/key_test',{service:k});keyMsg(j.message,j.ok?'ok':'bad')}catch(e){keyMsg(e.message,'bad')}});
$$('[data-tab]').forEach(b=>b.onclick=()=>{$$('[data-tab]').forEach(x=>x.classList.toggle('on',x===b));['hookpane','versionpane','draftpane','kwpane','reviewpane'].forEach(id=>$('#'+id).hidden=id!==b.dataset.tab);if(b.dataset.tab==='kwpane')loadKeywords()});
$('#draft').oninput=()=>draftDirty=true;
async function saveDraft(approve){try{const j=await api('/api/draft',{text:$('#draft').value,approve});draftDirty=false;$('#draftmsg').textContent=(approve?'Approved: ':'Saved: ')+j.file;$('#draftmsg').className='status ok';refresh(true)}catch(e){$('#draftmsg').textContent=e.message;$('#draftmsg').className='status bad'}}
$('#savedraft').onclick=()=>saveDraft(false);$('#approvedraft').onclick=()=>saveDraft(true);
$('#styledraft').onclick=async()=>{try{const j=await api('/api/style',{text:$('#draft').value,game:$('#game').value,topic:$('#topic').value});$('#draftmsg').textContent='Kept as a style example: '+j.file+' (used when writing other topics)';$('#draftmsg').className='status ok'}catch(e){$('#draftmsg').textContent=e.message;$('#draftmsg').className='status bad'}};
async function loadKeywords(){const box=$('#scenes');box.textContent='Loading…';try{const j=await api('/api/keywords');box.textContent='';j.scenes.forEach(s=>{const d=document.createElement('div');d.className='scene';
 d.innerHTML=`<div class="tag">Line ${s.scene} · ${s.section}</div><div></div><input type="text" data-scene="${s.scene}">`;d.children[1].textContent=s.sentence;d.querySelector('input').value=s.keywords.join(', ');d.querySelector('input').title='Suggested: '+s.suggested.join(', ');box.appendChild(d)})}catch(e){box.textContent=e.message}}
$('#reloadkw').onclick=loadKeywords;
$('#savekw').onclick=async()=>{try{const scenes=$$('#scenes input').map(i=>({scene:+i.dataset.scene,keywords:i.value}));const j=await api('/api/keywords',{scenes});$('#kwmsg').textContent=`Saved keywords for ${j.saved} line(s). Press Find clips.`;$('#kwmsg').className='status ok'}catch(e){$('#kwmsg').textContent=e.message;$('#kwmsg').className='status bad'}};
function fmt(s){s=Math.max(0,Math.round(s));return s>=60?`${Math.floor(s/60)}m${String(s%60).padStart(2,'0')}s`:`${s}s`}
async function poll(){try{const j=await api('/api/log?since='+since);if(j.total<since){since=0;$('#log').textContent='';return}
 if(j.lines.length){const log=$('#log');const atEnd=log.scrollTop+log.clientHeight>=log.scrollHeight-30;log.textContent+=j.lines.join('\n')+'\n';since=j.total;if(atEnd)log.scrollTop=log.scrollHeight}
 $('#jobname').textContent=j.name?('Step: '+j.name):'No step running';$('#jobstatus').textContent=j.status;$('#jobstatus').className='pill '+({done:'ok',failed:'bad',stopped:'warn',running:'run'}[j.status]||'');
 $('#live').textContent=j.live?('▸ '+j.live):'';$('#quiet').textContent=j.status==='running'&&j.quiet_for>=8?`· still working, waiting on the model (no new output for ${fmt(j.quiet_for)})`:'';$('#jobtime').textContent=j.started?fmt((j.finished||Date.now()/1000)-j.started):'';$$('[data-run]').forEach(b=>b.disabled=j.status==='running');
 if(window._last==='running'&&j.status!=='running'){const nm=j.name,ok=j.status==='done';refresh(true).then(()=>{if(nm==='hooks'&&ok)showTab('hookpane');else if(nm==='script'&&ok)showTab((state&&state.versions&&state.versions.length>1)?'versionpane':'draftpane')})}window._last=j.status}catch(e){}}
async function refresh(force){try{state=await api('/api/state')}catch(e){return}
 const r=state.research,ev=state.evidence,dr=state.draft;keyShow(state.keys);
 if(!$('#game').value&&r.game)$('#game').value=r.game;if(!$('#topic').value&&r.topic)$('#topic').value=r.topic;
 const models=state.models;for(const id of ['model','writer_model']){const sel=$('#'+id);const cur=sel.value;sel.innerHTML='<option value="">Default</option>'+(models||[]).map(m=>`<option>${m}</option>`).join('');sel.value=cur}
 const bits=[];bits.push(models===null?'<span class="bad">Ollama not reachable – start Ollama</span>':`<span class="ok">Ollama: ${models.length} model(s)</span>`);
 bits.push(state.tools['yt-dlp']?'<span class="ok">yt-dlp</span>':'<span class="bad">yt-dlp missing</span>');bits.push(state.tools.ffmpeg?'<span class="ok">FFmpeg</span>':'<span class="bad">FFmpeg missing</span>');
 if(state.missing_scripts.length)bits.push('<span class="bad">missing: '+state.missing_scripts.join(', ')+'</span>');$('#meta').innerHTML=bits.join(' · ');
 $('#evstatus').innerHTML=ev.exists?`<span class="ok">${ev.passages} passages</span> for ${ev.game} / ${ev.topic}`+(r.topic&&ev.topic&&r.topic.toLowerCase()!==ev.topic.toLowerCase()?` <span class="warn">(newest research is for "${r.topic}": run Prepare evidence)</span>`:''):'<span class="warn">No evidence yet.</span>';
 if(dr.file){$('#draftname').innerHTML=`${dr.file} <span class="pill ${dr.status==='REJECTED'?'bad':'ok'}">${dr.status}</span>`+(dr.other_topic?` <span class="warn">(an older draft: none for "${r.topic}" yet)</span>`:'');if(force||!draftDirty)$('#draft').value=dr.text;$('#review').textContent=dr.review||'No report for this draft.'}
 renderHooks(state.hooks||[]);renderVersions(state.versions||[]);
 const nv=(state.versions||[]).length;$('#versiontab').textContent=nv?`Versions (${nv})`:'Versions';
 $('#versionhint').innerHTML=nv>1?`This is the version the generator chose. <a href="#" id="seeversions">Compare all ${nv} versions</a>`:'';
 const sv=$('#seeversions');if(sv)sv.onclick=e=>{e.preventDefault();showTab('versionpane')}}
let hookKey='', versionKey='';
function renderVersions(vs){const key=JSON.stringify(vs.map(v=>[v.number,v.total,v.draft]));if(key===versionKey)return;versionKey=key;const box=$('#versionlist');box.textContent='';
 if(!vs.length){box.innerHTML='<p class="hint">No versions yet. Pick a hook and write the script.</p>';return}
 vs.forEach(v=>{const d=document.createElement('div');d.className='version'+(v.selected?' star':'');const h=document.createElement('div');h.className='vh';
  const t=document.createElement('strong');t.textContent=(v.selected?'★ ':'')+'Version '+v.number+(v.angle?' · '+v.angle:'');const sc=document.createElement('span');sc.className='pill '+(v.eligible?'ok':'bad');
  sc.textContent=v.eligible?`${v.total}/100 · passed the fact check`:`${v.total}/100 · did not pass`;const b=document.createElement('button');b.className='go';b.textContent='Use this version';
  b.onclick=async()=>{try{const j=await api('/api/version',{number:v.number,game:$('#game').value,topic:$('#topic').value});$('#draftmsg').textContent='Using version '+v.number+': '+j.file;$('#draftmsg').className='status ok';draftDirty=false;await refresh(true);showTab('draftpane')}catch(e){alert(e.message)}};
  h.append(t,sc,b);const pre=document.createElement('pre');pre.textContent=v.draft.split('\n').filter(l=>/^(HOOK|NARRATION|PAYOFF|ENDING QUESTION):/.test(l)).map(l=>l.replace(/(\s*\[E\d+\])+\s*$/,'')).join('\n');
  const n=document.createElement('div');n.className='vnote';n.textContent=[v.words+' words',v.why?'judge: '+v.why:'',...(v.flags||[]).map(f=>'flagged: '+f),...(v.problems||[]).map(p=>'problem: '+p)].filter(Boolean).join(' · ');
  d.append(h,pre,n);box.appendChild(d)})}
function showTab(id){$$('[data-tab]').forEach(x=>x.classList.toggle('on',x.dataset.tab===id));['hookpane','versionpane','draftpane','kwpane','reviewpane'].forEach(p=>$('#'+p).hidden=p!==id)}
function renderHooks(hooks){const key=JSON.stringify(hooks.map(h=>h.text));if(key===hookKey)return;hookKey=key;const box=$('#hooklist');box.textContent='';
 if(!hooks.length){box.innerHTML='<p class="hint">No hooks yet for this topic. Press <b>Write hooks</b>.</p>';return}
 hooks.forEach((h,i)=>{const d=document.createElement('label');d.className='hook';const r=document.createElement('input');r.type='radio';r.name='hk';
  const t=document.createElement('div');const a=document.createElement('div');a.className='txt';a.textContent=(h.recommended?'★ ':'')+h.spoken;
  const sc=document.createElement('div');sc.className='sc';const e=v=>v==null?'–':v;sc.textContent=`${h.total}/100 · emotion ${e(h.emotion)}/5 · curiosity ${e(h.curiosity)}/5 · clarity ${e(h.clarity)}/5`+(h.notes&&h.notes.length?' · '+h.notes.join('; '):'');
  t.append(a,sc);d.append(r,t);r.onchange=()=>{$$('.hook').forEach(x=>x.classList.remove('sel'));d.classList.add('sel');$('#chosenhook').value=h.spoken};box.appendChild(d)})}
$('#usehook').onclick=async()=>{const hook=$('#chosenhook').value.trim();if(!hook){$('#hookmsg').textContent='Pick or type a hook first.';$('#hookmsg').className='status bad';return}
 try{await api('/api/run',{step:'script',params:{...params(),chosen_hook:hook}});since=0;$('#log').textContent='';$('#hookmsg').textContent='Writing the script for this hook…';$('#hookmsg').className='status ok'}catch(e){$('#hookmsg').textContent=e.message;$('#hookmsg').className='status bad'}};
setInterval(poll,1000);setInterval(()=>refresh(false),8000);poll();refresh(true);
</script></body></html>"""


def main():
    server = None
    for port in range(FIRST_PORT, FIRST_PORT + 10):
        try:
            server = ThreadingHTTPServer((HOST, port), Handler)
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit(f"Could not open a local port ({FIRST_PORT}-{FIRST_PORT + 9}). Close other programs and try again.")
    url = f"http://{HOST}:{server.server_address[1]}/"
    print(f"GamingLoreAI is running at {url}\nKeep this window open. Press Ctrl+C here to close the app.")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nClosing.")
        stop_job()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
