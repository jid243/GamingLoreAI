from pathlib import Path
import base64, math, os, re, shutil, subprocess, sys, json, threading, time, urllib.request
from collections import Counter
from datetime import datetime

PROJECT = Path.home() / "Desktop" / "GamingLoreAI"
SCRIPT_DIR = PROJECT / "scripts"
SOURCES_DIR = PROJECT / "sources"
EVIDENCE_FILE = SOURCES_DIR / "evidence_passages.txt"
OUTPUT_DIR = PROJECT / "comfyui_input_clips"
MANIFEST_DIR = PROJECT / "clip_manifests"
CLIP_SECONDS = 12
MAX_SEARCH_RESULTS = 8
MIN_SOURCE_DURATION = 18
MAX_QUERIES_PER_SCENE = 12     # search ladder length per scene: every keyword, alone and paired with the others
MAX_DOWNLOAD_TRIES = 3         # download attempts per query before moving to the next query
PREFERRED_LENGTH = 900         # seconds; playthrough-length sources beat hour-long compilations
SEARCH_TIMEOUT = 60            # seconds before a hung search/info call is killed
DOWNLOAD_TIMEOUT = 150         # seconds before a hung download is killed
FFMPEG_TIMEOUT = 120
OLLAMA_URL = "http://127.0.0.1:11434"
VISION_TIMEOUT = 150           # seconds per frame review
MAX_CANDIDATES = int(os.environ.get("HARVEST_MAX_TRIES", "10"))   # clips fetched and judged per scene at most
OPTIONS_PER_SCENE = 2          # clips offered for you to choose between (A and B)
OPTION_LABELS = ("A", "B")
GOOD_RELEVANCE = 7             # a clip scoring this well (and not cluttered) stops the search early
MIN_KEEP_RELEVANCE = int(os.environ.get("HARVEST_MIN_RELEVANCE", "7"))
COVERAGE_TRIES = 6             # clips fetched per key subject by the coverage check   # clips the vision check scores lower are deleted, never offered
GOOD_CLUTTER = 3
SUBTITLE_TIMEOUT = 60          # seconds to fetch a video's captions (used to find the exact moment, no video download)
MOMENT_WINDOW = 20             # seconds of dialogue scored together when looking for the moment
MIN_MOMENT_SCORE = 3             # below this the captions do not really mention the scene, so they are not trusted
VISION_PREFERENCE = ("qwen2.5vl", "qwen3-vl", "gemma3", "minicpm-v", "llama3.2-vision", "llava")

STOPWORDS = set("a about above after again against all also although always among an and any are around as at be because been before being below between both but by can could did do does doing down during each either else few for from further had has have having he her here hers herself him himself his how i if in into is it its itself just more most my myself neither no nor not of off on once only or other our ours ourselves out over own same she should so some such than that the their theirs them themselves then there these they this those through to too under until up very was we were what when where which while who whom why will with would you your yours yourself yourselves".split())

# A "named thing" is a run of Capitalised Words, optionally joined by of/the/de/etc. and ending in a number.
CONNECTORS = r"(?:of|the|de|del|della|du|von|van|la|le|el|al)"
CAP = r"[A-Z][\w'\u2019-]*"
ENTITY_RE = re.compile(rf"{CAP}(?:\s+(?:{CONNECTORS}\s+)*{CAP})*(?:\s+\d+)?")


def clean(text):
    text = re.sub(r"\[(?:E\d+|\d+)\]", "", text)
    text = re.sub(r"[^\w\s'-]", " ", text, flags=re.UNICODE)
    return " ".join(text.split()).strip()


def last_line(text, limit=300):
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1][:limit] if lines else ""


def find_latest_approved_draft():
    candidates = [p for p in SCRIPT_DIR.glob("*_REVIEW_REQUIRED.txt") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No *_REVIEW_REQUIRED.txt draft found in {SCRIPT_DIR}. Generate and review a script first.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_metadata(text, key):
    m = re.search(rf"(?mi)^\s*{re.escape(key)}\s*:\s*(.+?)\s*$", text)
    return m.group(1).strip() if m else ""


def extract_spoken_lines(text):
    lines = []
    for raw in text.splitlines():
        m = re.match(r"^\s*(HOOK|NARRATION|PAYOFF|ENDING QUESTION)\s*:\s*(.*?)\s*$", raw, re.I)
        if not m:
            continue
        body = re.sub(r"(?:\s*\[E\d+\])+\s*$", "", m.group(2)).strip()
        if body:
            lines.append((m.group(1).upper(), body))
    return lines


def evidence_excerpt(evidence_text, tag):
    blocks = re.split(r"(?m)(?=^\[E\d+\]\s*$)", evidence_text)
    for block in blocks:
        if re.search(rf"(?m)^\[{re.escape(tag)}\]\s*$", block):
            found = re.search(r"(?ms)^Excerpt:\s*(.*)$", block)
            text = (found.group(1) if found else block).strip()
            # wiki section labels such as "(Biography > Meeting Geralt)" are page structure, not names in the story
            text = re.sub(r"^\([^)]{0,120}\)\s*", "", text)
            return text[:2500]
    return ""


# ---------------------------------------------------------------------------
# Entity extraction: works for any game because it relies on capitalisation,
# not on a hard-coded list of names.
# ---------------------------------------------------------------------------
def norm_entity(raw):
    ent = re.sub(r"['\u2019]s\b", "", raw).strip()
    words = ent.split()
    while words and words[0].casefold() in STOPWORDS:   # "In The Witcher 3", "Should Geralt", "If Geralt", "A"
        words = words[1:]
    ent = " ".join(words)
    return ent.strip() if len(ent.strip()) >= 2 else ""


def sentences_of(text):
    return [s for s in re.split(r"(?<=[.!?])\s+|\n", text) if s.strip()]


def find_entities(sentence):
    """Yield (entity, is_at_sentence_start) for every capitalised phrase."""
    found = []
    for m in ENTITY_RE.finditer(sentence):
        ent = norm_entity(m.group(0))
        if ent:
            # "Should Orsk": once the leading function word is dropped, the name is no longer sentence-initial
            stripped_lead = not re.sub(r"['\u2019]s\b", "", m.group(0)).strip().startswith(ent)
            found.append((ent, not sentence[:m.start()].strip() and not stripped_lead))
    return found


def collect_known(texts):
    """Names seen mid-sentence anywhere (evidence, narration, topic).
    Lets us trust a sentence-initial word like 'Geralt' but ignore 'Freeing'."""
    known = set()
    for text in texts:
        for sent in sentences_of(text):
            for ent, at_start in find_entities(sent):
                if at_start and len(ent.split()) == 1:
                    continue
                known.add(ent)
                known.update(w for w in ent.split() if w[:1].isupper())
    return known


def extract_entities(sentence, known, game):
    game_cf = game.casefold()
    out, seen = [], set()
    for ent, at_start in find_entities(sentence):
        words = ent.split()
        if len(words) == 1:
            if ent.casefold() in STOPWORDS:
                continue
            if at_start and ent not in known:
                continue
        if ent.casefold() in game_cf or game_cf in ent.casefold():   # the game's own name is not a scene entity
            continue
        if ent.casefold() in seen:
            continue
        seen.add(ent.casefold())
        out.append(ent)
    # Multi-word names are more specific, so search them first (stable sort keeps story order).
    return sorted(out, key=lambda e: -len(e.split()))


# ---------------------------------------------------------------------------
# Query building
# ---------------------------------------------------------------------------
def footage_suffix(sentence):
    """Search for raw playthrough footage: 'no commentary' plus an intent word where one is clear."""
    lower = sentence.casefold()
    if any(w in lower for w in ("fight", "battle", "combat", "weapon", "boss")):
        return "boss fight no commentary"
    if any(w in lower for w in ("spell", "magic", "power", "ability", "skill", "upgrade")):
        return "gameplay no commentary"
    return "cutscene no commentary"


def build_queries(sentence, game, topic, entities):
    suffix = footage_suffix(sentence)
    short_topic = " ".join(re.sub(r"^(?:the|a|an)\s+", "", topic, flags=re.I).split()[:6])
    queries = [f"{game} {ent} {suffix}" for ent in entities[:2]]
    if short_topic:
        queries.append(f"{game} {short_topic} {suffix}")
        queries.append(f"{game} {short_topic} all cutscenes 4K no HUD")
        queries.append(f"{game} {short_topic} walkthrough no commentary")
    if not queries:
        queries.append(f"{game} walkthrough no commentary")
    unique, seen = [], set()
    for q in queries:
        q = clean(q)
        if q.casefold() not in seen:
            seen.add(q.casefold())
            unique.append(q)
    return unique[:MAX_QUERIES_PER_SCENE]


def word_set(text):
    return {w.casefold() for w in re.findall(r"\w+", text)}


def relevance_terms(entities, topic, game):
    game_words = word_set(game)
    terms = set()
    for chunk in list(entities) + [topic]:
        terms |= {w for w in word_set(chunk) if len(w) >= 3 and w not in STOPWORDS}
    return terms - game_words


BAD_WORDS = {"reaction", "reacts", "react", "review", "explained", "explanation", "analysis", "theory", "theories",
             "ranked", "tier", "trailer", "mod", "mods", "news", "podcast", "stream", "live", "tutorial", "guide",
             "meme", "memes", "funny", "fails", "lore", "secrets", "top", "best", "worst", "vs", "history"}
GOOD_WORDS = {"walkthrough", "gameplay", "playthrough", "longplay", "cutscene", "cutscenes", "cinematic", "quest",
             "4k", "2160p", "60fps", "ultra"}


def rank_results(entries, terms, game, relaxed=False):
    """Keep results whose title names this scene's subject, favour no-commentary playthroughs,
    and drop commentary/lore/review style videos."""
    game_words = {w for w in word_set(game) if w not in STOPWORDS and not w.isdigit()}
    ranked = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        title = entry.get("title", "")
        title_cf = title.casefold()
        title_words = word_set(title)
        hits = len(terms & title_words)
        game_hit = bool(game_words & title_words)
        duration = entry.get("duration")
        if duration is not None and duration < MIN_SOURCE_DURATION:
            continue
        if relaxed:                     # a vision model judges the footage, so don't over-filter on titles
            if not (hits >= 1 or game_hit):
                continue
        elif not (hits >= 1 and (game_hit or hits >= 2)):
            continue
        no_commentary = any(p in title_cf for p in ("no commentary", "without commentary", "commentary free"))
        if (title_words & BAD_WORDS) and not no_commentary:
            continue
        no_hud = "no hud" in title_cf or "hud off" in title_cf
        score = (hits + int(game_hit) + (3 if no_commentary else 0) + (1 if title_words & GOOD_WORDS else 0)
                 + (2 if no_hud else 0) + (1 if title_words & {"cutscene", "cutscenes", "cinematic"} else 0))
        closeness = -abs((duration or PREFERRED_LENGTH) - PREFERRED_LENGTH)
        ranked.append((score, closeness, entry, hits))
    ranked.sort(key=lambda r: (r[0], r[1]), reverse=True)
    return [(entry, hits) for _, _, entry, hits in ranked]


# ---------------------------------------------------------------------------
# yt-dlp / ffmpeg helpers
# ---------------------------------------------------------------------------
def fmt_secs(seconds):
    seconds = int(seconds)
    return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def show_status(text):
    sys.stdout.write("\r" + text[:100].ljust(100))
    sys.stdout.flush()


def clear_status():
    sys.stdout.write("\r" + " " * 100 + "\r")
    sys.stdout.flush()


def run_capture(cmd, timeout=None, label="", hint=None):
    """Run a command with a hard timeout and, if label is given, a live elapsed-time line
    (plus an optional hint such as downloaded size) so you can tell working from stuck."""
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace")
    started = time.time()
    spinner = "|/-\\"
    ticks = 0
    while True:
        elapsed = time.time() - started
        try:
            out, err = proc.communicate(timeout=0.5)
            if label:
                clear_status()
            return subprocess.CompletedProcess(cmd, proc.returncode, out, err)
        except subprocess.TimeoutExpired:
            if timeout and elapsed >= timeout:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
                else:
                    proc.kill()
                try:
                    proc.communicate(timeout=5)
                except Exception:
                    pass
                if label:
                    clear_status()
                return subprocess.CompletedProcess(cmd, 124, "", f"timed out after {timeout}s")
            if label:
                ticks += 1
                extra = ""
                if hint:
                    try: extra = "  " + hint()
                    except Exception: pass
                limit = f" (gives up at {timeout}s)" if timeout else ""
                show_status(f"    {spinner[ticks % 4]} {label}: {fmt_secs(elapsed)}{limit}{extra}")


def search_youtube(yt_dlp, query):
    r = run_capture([yt_dlp, "--flat-playlist", "--dump-single-json", "--no-warnings",
                     "--socket-timeout", "15", f"ytsearch{MAX_SEARCH_RESULTS}:{query}"],
                    timeout=SEARCH_TIMEOUT, label="Searching YouTube")
    if r.returncode != 0 or not r.stdout.strip():
        return [], last_line(r.stderr) or "search returned nothing"
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        return [], "search output was not valid JSON"
    entries = data.get("entries", []) if isinstance(data, dict) else []
    return [e for e in entries if isinstance(e, dict)], ""


def video_info(yt_dlp, url):
    r = run_capture([yt_dlp, "--no-playlist", "--skip-download", "--dump-single-json", "--no-warnings",
                     "--socket-timeout", "15", url], timeout=SEARCH_TIMEOUT, label="Reading video chapters")
    try:
        return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else {}
    except json.JSONDecodeError:
        return {}


SUBS_DIR = OUTPUT_DIR / "_captions"
_CAPTION_CACHE = {}


def parse_vtt(text):
    """[(start_seconds, text)] from a WebVTT file, with the rolling duplicates of auto-captions removed."""
    cues, last = [], ""
    for block in re.split(r"\n\s*\n", text):
        m = re.search(r"(\d+):(\d+):(\d+)[.,]\d+\s+-->", block)
        if not m:
            m2 = re.search(r"(\d+):(\d+)[.,]\d+\s+-->", block)
            if not m2:
                continue
            start = int(m2.group(1)) * 60 + int(m2.group(2))
        else:
            start = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
        lines = [re.sub(r"<[^>]+>", "", ln).strip() for ln in block.splitlines() if "-->" not in ln]
        words = " ".join(ln for ln in lines if ln and not ln.startswith(("WEBVTT", "Kind:", "Language:")))
        if not words or words == last:
            continue
        if last and words.startswith(last):            # rolling caption: keep only the new words
            words = words[len(last):].strip() or words
        cues.append((start, words))
        last = " ".join(ln for ln in lines if ln)
    return cues


def video_captions(yt_dlp, url):
    """The video's English captions (uploaded or automatic) as cues; [] if it has none. No video is downloaded.
    No-commentary playthroughs caption the game's own dialogue, which says exactly when each moment happens."""
    if url in _CAPTION_CACHE:
        return _CAPTION_CACHE[url]
    SUBS_DIR.mkdir(parents=True, exist_ok=True)
    key = re.sub(r"[^A-Za-z0-9_-]+", "_", url)[-40:]
    for old in SUBS_DIR.glob(f"{key}*"):
        try: old.unlink()
        except OSError: pass
    run_capture([yt_dlp, "--skip-download", "--write-subs", "--write-auto-subs", "--sub-langs", "en.*,en",
                 "--sub-format", "vtt", "--no-warnings", "--socket-timeout", "15", "--no-playlist",
                 "-o", str(SUBS_DIR / f"{key}.%(ext)s"), url], timeout=SUBTITLE_TIMEOUT, label="Reading captions")
    files = sorted(SUBS_DIR.glob(f"{key}*.vtt"), key=lambda f: (".en." not in f.name, f.name))
    cues = []
    for f in files:
        try:
            cues = parse_vtt(f.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            cues = []
        if cues:
            break
    for f in SUBS_DIR.glob(f"{key}*"):
        try: f.unlink()
        except OSError: pass
    _CAPTION_CACHE[url] = cues
    return cues


def _stems5(text):
    return {w[:5] for w in re.findall(r"[a-z]{4,}", text.casefold()) if w not in STOPWORDS}


def find_moment(cues, keywords, sentence, game, begin=0, end=None, spoken_words=()):
    """The MOMENT_WINDOW of dialogue that best matches the scene. Each matching word counts by how RARE it is in
    this video (a word heard all through it says little about where the moment is; a word heard once says a lot).
    The user's keywords count twice, and a multi-word keyword ("bloody baron") only counts when its words are heard
    together, so "blood ties" is not mistaken for it. spoken_words are words the characters would likely say in
    this moment (the narration says "abuse", the dialogue says "beat her"). 5-letter stems tolerate the captions'
    spelling slips. Returns (start_seconds, score, dialogue) or None."""
    if not cues:
        return None
    game_stems = _stems5(game)
    phrases = [st for st in (_stems5(k) - game_stems for k in keywords) if st]
    key_stems = set().union(*phrases) if phrases else set()
    line_stems = (_stems5(sentence) | set().union(*[_stems5(w) for w in spoken_words] or [set()])) - game_stems - key_stems
    windows = []
    for i, (start, _) in enumerate(cues):
        text = " ".join(t for s2, t in cues[i:i + 40] if s2 - start <= MOMENT_WINDOW)
        windows.append((start, text, _stems5(text)))
    df = Counter()
    for _, _, stems in windows:
        df.update(stems & (key_stems | line_stems))
    n = len(windows)

    def idf(stem):
        return math.log((n + 1) / (df[stem] + 1)) + 1     # smoothed: a word heard everywhere still counts 1
    best = None
    for start, text, stems in windows:
        if start < begin or (end is not None and start > end):
            continue
        score = 0.0
        for phrase in phrases:
            if phrase <= stems:                          # every word of the keyword is heard here
                score += sum(2 * idf(w) for w in phrase)
        score += sum(idf(w) for w in line_stems & stems)
        if best is None or score > best[1]:
            best = (start, score, text)
    if not best or best[1] < MIN_MOMENT_SCORE:
        return None
    return max(0, best[0] - 1), round(best[1], 1), best[2][:300]


def dialogue_words(vision, game, sentence):
    """Words the characters would likely SAY on screen in this moment, so captions can be matched even when the
    narration describes it differently. Asked once per scene of the local model; [] if none is available."""
    if not vision or os.environ.get("HARVEST_DIALOGUE_WORDS") != "1":
        return []   # off by default: tested local models mostly echoed the narration or gave generic words
    prompt = (f"A narrated video about the game {game} says: \"{sentence}\"\n"
              "List up to 10 single words that characters would likely SAY OUT LOUD in the game's own dialogue during "
              "the scene this describes (plain spoken words, including names; not the narrator's words). "
              'Reply with JSON only: {"words": ["...", "..."]}')
    try:
        reply = http_json(f"{OLLAMA_URL}/api/chat", {"model": vision["model"], "stream": False, "format": "json",
                                                      "options": {"temperature": 0},
                                                      "messages": [{"role": "user", "content": prompt}]}, timeout=60)
        data = json.loads(re.search(r"\{.*\}", reply.get("message", {}).get("content", ""), re.S).group(0))
        return [str(w) for w in data.get("words", []) if isinstance(w, (str, int))][:10]
    except Exception:
        return []


def choose_start(chapters, duration, terms, used_count):
    """Prefer a chapter whose title names the scene's subject; otherwise skip intro/outro blindly.
    used_count shifts the window so scenes sharing a video don't get identical footage."""
    best = None
    for ch in chapters or []:
        hits = len(terms & word_set(ch.get("title", "")))
        if hits and (best is None or hits > best[0]):
            best = (hits, ch)
    if best:
        ch = best[1]
        begin = float(ch.get("start_time") or 0)
        end = float(ch.get("end_time") or begin + CLIP_SECONDS * 4)
        room = max(0.0, end - begin - CLIP_SECONDS - 6)
        start = begin + 4 + min(room, used_count * CLIP_SECONDS)
        return int(start), f"chapter \"{ch.get('title', '')}\"", ch.get("title", "")
    if not duration or duration < 90:
        return 0, "start of video", ""
    latest = duration * 0.85 - CLIP_SECONDS            # stay out of end cards and outros
    start = min(duration * 0.10, 90) + used_count * CLIP_SECONDS * 2
    return int(max(0, min(start, latest))), "no chapter match; skipped intro", ""


def locate_moment(yt_dlp, url, info, entry, keywords, sentence, game, terms, used_count, spoken_words=()):
    """Where in this video is the scene? 1) the captions' best match for the scene (inside the best chapter when a
    chapter title names it), 2) the chapter alone, 3) the old fallback of skipping the intro.
    Returns (start, reason, chapter_title, dialogue, moment_score)."""
    duration = info.get("duration") or entry.get("duration")
    start, reason, chapter_title = choose_start(info.get("chapters"), duration, terms, used_count)
    cues = video_captions(yt_dlp, url)
    if cues:
        chapter = next((c for c in info.get("chapters") or [] if c.get("title") == chapter_title), None) if chapter_title else None
        in_chapter = None
        if chapter:
            in_chapter = find_moment(cues, keywords, sentence, game, float(chapter.get("start_time") or 0),
                                     float(chapter.get("end_time") or 0) or None, spoken_words)
        anywhere = find_moment(cues, keywords, sentence, game, spoken_words=spoken_words)
        # a chapter title only names the general subject; when the dialogue elsewhere in the video matches this
        # scene clearly better, the dialogue wins
        moment = in_chapter if in_chapter and not (anywhere and anywhere[1] > in_chapter[1] * 1.3) else anywhere
        if moment:
            offset = used_count * CLIP_SECONDS          # scenes sharing a video get the next stretch, not the same one
            where = int(moment[0] + offset)
            if duration:
                where = min(where, int(max(0, duration - CLIP_SECONDS - 2)))
            return where, f"captions match (score {moment[1]})", chapter_title, moment[2], moment[1]
    return start, reason, chapter_title, "", 0


def fetch_segment(yt_dlp, ffmpeg, url, start, base, out_file):
    """Download a short video-only section to <base>_raw.*, then transcode to out_file. Returns (ok, error)."""
    def leftovers():
        return [p for p in OUTPUT_DIR.glob(f"{base}_raw.*") if p.is_file()]

    def cleanup():
        for p in leftovers():
            try: p.unlink()
            except OSError: pass

    def downloaded_size():
        total = sum(p.stat().st_size for p in leftovers())
        return f"{total / 1_000_000:.1f} MB received" if total else "waiting for data..."

    cleanup()
    dl = run_capture([
        yt_dlp, "--no-playlist", "--no-part", "--no-warnings", "--socket-timeout", "15", "--retries", "2",
        "--ffmpeg-location", str(ffmpeg),
        "--download-sections", f"*{start}-{start + CLIP_SECONDS}", "--force-overwrites",
        "-f", "bv*[height<=2160]/bv*/b",
        "-o", str(OUTPUT_DIR / f"{base}_raw.%(ext)s"), url,
    ], timeout=DOWNLOAD_TIMEOUT, label="Downloading clip", hint=downloaded_size)
    raws = [p for p in leftovers() if p.stat().st_size > 0]
    if dl.returncode != 0 or not raws:
        err = last_line(dl.stderr) or "yt-dlp produced no file"
        cleanup()
        return False, err
    raw = max(raws, key=lambda p: p.stat().st_size)
    tc = run_capture([
        ffmpeg, "-nostdin", "-y", "-i", str(raw), "-t", str(CLIP_SECONDS), "-an",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-preset", "veryfast", "-movflags", "+faststart", str(out_file),
    ], timeout=FFMPEG_TIMEOUT, label="Converting clip")
    cleanup()
    if tc.returncode == 0 and out_file.exists() and out_file.stat().st_size > 0:
        return True, ""
    if out_file.exists():
        out_file.unlink()
    return False, "ffmpeg: " + last_line(tc.stderr)


# ---------------------------------------------------------------------------
# Coverage: make sure the story's key subjects (the topic itself, plus nouns the script keeps
# coming back to, such as "orphans") each get at least one clip whose video or chapter names them.
# ---------------------------------------------------------------------------
GENERIC_WORDS = set("""about decision decisions choice choices people person player players thing things story result
results presence choose chooses causes cause affects affect leads lead saves save takes taken makes gives something
someone anyone everything whether another toward towards which would could should these those their there where
while after before because between during through really being place places point world game games lets later
earlier within without across""".split())


def stem(word):
    word = word.casefold()
    return word[:-1] if word.endswith("s") and len(word) > 4 else word


def stems_in(text):
    return {stem(w) for w in re.findall(r"\w+", text) if len(w) >= 3}


def core_topic(topic):
    return re.sub(r"^(?:the|a|an)\s+", "", topic.strip(), flags=re.I)


def topic_sentences(evidence_text, topic):
    core = core_topic(topic).casefold()
    return [s for s in sentences_of(evidence_text) if core and core in s.casefold()] if core else []


def top_entities(sentences, known, game, skip_words, n=2):
    counts = Counter()
    game_cf = game.casefold()
    for sent in sentences:
        for ent in extract_entities(sent, known, game):
            if ent.casefold() in game_cf or (word_set(ent) & skip_words):
                continue
            counts[ent] += 1
    return [e for e, _ in counts.most_common(n)]


def story_subjects(lines, evidence_text, topic, game, known):
    """The topic first, then up to two nouns the script repeats. A noun that shares a sentence with the
    topic (for example 'spirit' alongside the topic's name) is treated as the topic itself, not a new subject."""
    game_words = word_set(game)
    topic_words = {w for w in word_set(topic) if len(w) >= 3 and w not in STOPWORDS} - game_words
    topic_sents = topic_sentences(evidence_text, topic)
    topic_stems = stems_in(" ".join(topic_sents))
    for _, sentence in lines:
        if topic_words & word_set(sentence):
            topic_stems |= stems_in(sentence)
    ctx = top_entities(topic_sents, known, game, topic_words | game_words, n=1)
    subjects = [{"kind": "topic", "label": core_topic(topic), "words": topic_words,
                 "stems": {stem(w) for w in topic_words}, "context": ctx[0] if ctx else "",
                 "facts": " ".join(topic_sents[:2])[:500]}]

    entity_words = set()
    for ent in known:
        entity_words |= word_set(ent)
    narr = Counter()
    for _, sentence in lines:
        for w in set(re.findall(r"[a-z]{5,}", sentence.casefold())):
            narr[w] += 1
    evid = Counter(re.findall(r"[a-z]{5,}", evidence_text.casefold()))
    scored = []
    for w, c in narr.items():
        if c < 2 or evid.get(w, 0) < 3:
            continue
        if w in STOPWORDS or w in GENERIC_WORDS or w in entity_words or w in game_words or w in topic_words:
            continue
        if w.endswith(("ing", "ed", "ly", "tion", "ment", "ness")):
            continue
        if stem(w) in topic_stems:                      # same story thread as the topic: covered by it
            continue
        scored.append((c * min(evid[w], 20), w))
    for _, w in sorted(scored, reverse=True)[:2]:
        sents = [sn for _, sn in lines if w in sn.casefold()]
        ctx = top_entities(sents, known, game, {w} | game_words, n=1)
        facts = " ".join([sn for sn in sentences_of(evidence_text) if w in sn.casefold()][:2])[:500]
        subjects.append({"kind": "noun", "label": w, "words": {w}, "stems": {stem(w)},
                         "context": ctx[0] if ctx else "", "facts": facts})
    return subjects


def clip_text(asset):
    return f"{asset.get('source_title', '')} {asset.get('chapter_title', '')}"


def covers(asset, subject):
    return asset.get("status") == "downloaded_unverified" and bool(subject["stems"] & stems_in(clip_text(asset)))


def pick_scene(subject, assets, pinned):
    pool = [a for a in assets if a["scene"] not in pinned]
    mentions = [a for a in pool if subject["stems"] & stems_in(a["narration"])]
    pool = mentions or pool
    return min(pool, key=lambda a: (a["status"] != "failed", a.get("title_relevance_score", 0)), default=None)


# ---------------------------------------------------------------------------
# Visual review with a local Ollama vision model: looks at frames, not titles.
# ---------------------------------------------------------------------------
def http_json(url, payload=None, timeout=20):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def pick_vision_model():
    try:
        names = [m.get("name", "") for m in http_json(f"{OLLAMA_URL}/api/tags", timeout=5).get("models", [])]
    except Exception:
        return None
    for preferred in VISION_PREFERENCE:
        for name in names:
            if preferred in name.lower():
                return name
    return None


def unload_vision(model):
    try:
        http_json(f"{OLLAMA_URL}/api/generate", {"model": model, "keep_alive": 0}, timeout=15)
    except Exception:
        pass


SPIN = "|/-\\"


def call_with_status(label, fn, timeout):
    """Run a blocking call in a thread while showing a live timer."""
    box = {}

    def runner():
        try:
            box["value"] = fn()
        except Exception as exc:
            box["error"] = exc

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    started, ticks = time.time(), 0
    while t.is_alive():
        t.join(0.5)
        elapsed = time.time() - started
        if elapsed > timeout:
            clear_status()
            return None
        ticks += 1
        show_status(f"    {SPIN[ticks % 4]} {label}: {fmt_secs(elapsed)} (gives up at {timeout}s)")
    clear_status()
    return box.get("value")


def contact_sheet(ffmpeg, clip, out_jpg):
    r = run_capture([ffmpeg, "-nostdin", "-y", "-i", str(clip), "-vf",
                     f"fps=3/{CLIP_SECONDS},scale=640:-1,tile=3x1", "-frames:v", "1", "-q:v", "4", str(out_jpg)],
                    timeout=60)
    return r.returncode == 0 and out_jpg.exists()


def ask_vision(vision, clip, prompt, label):
    sheet = Path(clip).with_suffix(".review.jpg")
    if not contact_sheet(vision["ffmpeg"], clip, sheet):
        return None
    try:
        image = base64.b64encode(sheet.read_bytes()).decode("ascii")

        def call():
            reply = http_json(f"{OLLAMA_URL}/api/chat", {
                "model": vision["model"], "stream": False, "format": "json", "keep_alive": "10m",
                "options": {"temperature": 0},
                "messages": [{"role": "user", "content": prompt, "images": [image]}]}, timeout=VISION_TIMEOUT)
            return reply.get("message", {}).get("content", "")

        text = call_with_status(label, call, VISION_TIMEOUT)
    finally:
        try: sheet.unlink()
        except OSError: pass
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def num(value, default=0):
    try:
        return max(0, min(10, float(value)))
    except (TypeError, ValueError):
        return default


def review_clip(vision, clip, game, topic, sentence, evidence, dialogue=""):
    heard = (f"Dialogue spoken in the game at this moment (from the video's captions): \"{dialogue[:280]}\"\n"
             if dialogue else "")
    prompt = (
        f"You are screening video clips for a short documentary about the video game {game}. Topic: {topic}.\n"
        f"Narration this clip must illustrate: \"{sentence}\"\n"
        f"Background facts: {evidence or 'none'}\n"
        + heard +
        "The image is three frames from ONE clip, left to right. You cannot recognise characters by name, so judge by "
        "what KIND of scene it is: are the people, creature, place or action the narration describes plausibly on screen "
        "(for example a tense conversation between the characters it names, the creature it names, a grave, a ritual)? "
        "If the dialogue above matches the narration, a matching conversation or event on screen is highly relevant.\n"
        "relevance (0-10): does the footage fit this moment of the story? Unrelated footage from the same game "
        "(menus, travel, other quests) scores 3 or less.\n"
        "clutter (0-10): how much game interface or overlay covers the frame (health bars, minimap, quest tracker, "
        "inventory, channel logos, titles, large text).\n"
        "quality (0-10): is it sharp, well lit and clearly composed, with a clear main subject (not mostly black or blurry)?\n"
        'Reply with JSON only: {"relevance": n, "clutter": n, "quality": n, "shows": "one short sentence"}')
    data = ask_vision(vision, clip, prompt, "Reviewing clip")
    if not isinstance(data, dict):
        return None
    return {"relevance": num(data.get("relevance")), "clutter": num(data.get("clutter"), 5),
            "quality": num(data.get("quality"), 5), "shows": str(data.get("shows", ""))[:160]}


def clip_shows_subject(vision, clip, subject):
    prompt = (
        f"Subject to look for: {subject['label']}.\nFacts about it: {subject.get('facts') or 'none'}\n"
        "The image is three frames from one video game clip. Does the footage clearly show this subject "
        "(the specific character, creature, place or object), not merely something from the same game? "
        'Reply with JSON only: {"visible": true or false, "shows": "one short sentence"}')
    data = ask_vision(vision, clip, prompt, f"Looking for '{subject['label']}'")
    if not isinstance(data, dict):
        return None
    return data.get("visible") is True or str(data.get("visible")).lower() == "true"


def review_score(review, title_hits):
    if not review:
        return float(title_hits)
    return review["relevance"] + 0.5 * review["quality"] - 0.7 * review["clutter"]


def is_good(review):
    return bool(review) and review["relevance"] >= GOOD_RELEVANCE and review["clutter"] <= GOOD_CLUTTER


def show_review(review):
    if review:
        print(f"    Vision: relevance {review['relevance']:.0f}/10, clutter {review['clutter']:.0f}/10, "
              f"quality {review['quality']:.0f}/10 - {review['shows']}", flush=True)
    else:
        print("    Vision review failed or timed out; judging by title only.", flush=True)


# ---------------------------------------------------------------------------
# Options: every scene keeps up to two candidate clips (A and B) and one chosen clip
# ---------------------------------------------------------------------------
OPTION_FIELDS = ("file", "source_title", "source_url", "source_duration", "clip_start_seconds", "start_reason",
                 "chapter_title", "title_relevance_score", "vision_review", "query")


def option_score(opt):
    return review_score(opt.get("vision_review"), opt.get("title_relevance_score", 0))


def select_option(asset, label):
    """Make one option the scene's clip (label 'none' means: use no clip for this scene)."""
    asset["chosen"] = label
    if label == "none":
        asset["use_clip"] = False
        return
    opt = next((o for o in asset.get("options", []) if o["label"] == label), None)
    if not opt:
        return
    for key in OPTION_FIELDS:
        if key in opt:
            asset[key] = opt[key]
    asset["use_clip"] = True
    asset["status"] = "downloaded_unverified"
    asset["review_required"] = True


def make_preview(ffmpeg, src, dst):
    """Small 480p copy so the review page plays smoothly even when the clip is 4K."""
    r = run_capture([ffmpeg, "-nostdin", "-y", "-i", str(src), "-vf", "scale=-2:480,fps=24", "-an",
                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-movflags", "+faststart", str(dst)],
                    timeout=FFMPEG_TIMEOUT)
    return str(dst) if r.returncode == 0 and Path(dst).exists() else ""


def clear_scene_files(base):
    for p in OUTPUT_DIR.glob(f"{base}_*"):
        try: p.unlink()
        except OSError: pass
    legacy = OUTPUT_DIR / f"{base}.mp4"
    if legacy.exists():
        try: legacy.unlink()
        except OSError: pass


def asset_shows_subject(asset, subject, vision):
    """True if any of the scene's options shows the subject (and then selects it). Visual check when a vision
    model is available (cached per option); title/chapter check otherwise."""
    options = [o for o in asset.get("options", []) if o.get("file")]
    options.sort(key=lambda o: o["label"] != asset.get("chosen"))       # check the chosen one first
    for opt in options:
        if vision:
            cache = opt.setdefault("subject_checks", {})
            if subject["label"] not in cache:
                cache[subject["label"]] = bool(clip_shows_subject(vision, opt["file"], subject))
            shows = cache[subject["label"]]
        else:
            shows = bool(subject["stems"] & stems_in(f"{opt.get('source_title', '')} {opt.get('chapter_title', '')}"))
        if shows:
            if asset.get("chosen") != opt["label"] and asset.get("use_clip", True):
                select_option(asset, opt["label"])
            return True
    return False


def cover_subject(subject, asset, yt_dlp, ffmpeg, game, used, stamp, vision):
    label, ctx = subject["label"], subject["context"]
    if subject["kind"] == "topic":
        queries = [f"{game} {label} cutscene no commentary", f"{game} {label} {ctx} cutscene no commentary",
                   f"{game} {label} all cutscenes 4K no HUD"]
    else:
        queries = [f"{game} {label} {ctx} cutscene no commentary", f"{game} {label} cutscene no commentary",
                   f"{game} {label} {ctx} all cutscenes 4K no HUD"]
    seen, unique = set(), []
    for q in queries:
        q = clean(q)
        if q.casefold() not in seen:
            seen.add(q.casefold())
            unique.append(q)
    terms = subject["words"] | (word_set(ctx) - word_set(game))
    base = f"{stamp}_scene_{asset['scene']:02d}"
    options = asset.setdefault("options", [])
    if len(options) >= OPTIONS_PER_SCENE:
        target = min(options, key=option_score)["label"]      # replace the weaker option
    else:
        target = next(l for l in OPTION_LABELS if l not in {o["label"] for o in options})
    out_file = OUTPUT_DIR / f"{base}_{target}.mp4"
    tmp_base = f"{base}_cover"
    tmp_file = OUTPUT_DIR / f"{tmp_base}.mp4"
    tried = 0
    for query in unique[:3]:
        print(f"  Coverage search: {query}", flush=True)
        entries, err = search_youtube(yt_dlp, query)
        cands = rank_results(entries, terms, game, relaxed=bool(vision))
        cands.sort(key=lambda c: -len(subject["stems"] & stems_in(c[0].get("title", ""))))
        print(f"    -> {len(entries)} results, {len(cands)} relevant" + (f"  [{err}]" if err else ""), flush=True)
        for entry, hits in cands[:MAX_DOWNLOAD_TRIES]:
            if tried >= COVERAGE_TRIES:
                return False
            url = entry.get("url") or entry.get("webpage_url")
            if not url or url in {o.get("source_url") for o in options}:
                continue
            title = entry.get("title", "")
            print(f"    Trying: {title}", flush=True)
            info = video_info(yt_dlp, url)
            duration = info.get("duration") or entry.get("duration")
            start, reason, chapter, dialogue, _score = locate_moment(
                yt_dlp, url, info, entry, sorted(subject["words"]), subject.get("facts", ""), game, subject["words"], used.get(url, 0))
            if not vision and not (subject["stems"] & stems_in(f"{title} {chapter}")):
                print("    Skipped: neither the title nor a chapter names this subject.", flush=True)
                continue
            print(f"    Downloading {CLIP_SECONDS}s at {fmt_secs(start)} ({reason})...", flush=True)
            ok, error = fetch_segment(yt_dlp, ffmpeg, url, start, tmp_base, tmp_file)
            if not ok:
                print(f"    Download failed: {error}", flush=True)
                continue
            tried += 1
            if vision:
                seen_it = clip_shows_subject(vision, tmp_file, subject)
                print(f"    Vision: {'shows' if seen_it else 'does not show'} '{label}'", flush=True)
                if not seen_it:
                    tmp_file.unlink()
                    continue
                review = review_clip(vision, tmp_file, game, label, asset.get("narration", ""), "", dialogue)
                show_review(review)
                if not keep_clip(review):    # it shows the subject, but must also fit this line like any other clip
                    print(f"    Not used: relevance {review['relevance']:.0f}/10 for this line is below {MIN_KEEP_RELEVANCE}.", flush=True)
                    tmp_file.unlink()
                    continue
            else:
                review = None
            if out_file.exists():
                prev = out_file.with_name(f"{base}_{target}_prev.mp4")
                if prev.exists():
                    prev.unlink()
                out_file.rename(prev)
            tmp_file.rename(out_file)
            used[url] = used.get(url, 0) + 1
            preview = make_preview(ffmpeg, out_file, out_file.with_suffix(".preview.mp4"))
            new_opt = {"label": target, "file": str(out_file), "preview": preview, "query": query,
                       "source_title": title, "source_url": entry.get("webpage_url", url),
                       "source_duration": duration, "clip_start_seconds": start, "start_reason": reason,
                       "chapter_title": chapter, "title_relevance_score": hits, "vision_review": review,
                       "chosen_for_coverage_of": label, "subject_checks": {label: True}}
            asset["options"] = sorted([o for o in options if o["label"] != target] + [new_opt], key=lambda o: o["label"])
            select_option(asset, target)
            return True
    return False


def is_visual_subject(vision, subject):
    """A clip can show a creature, person, place or object, not an idea ('miscarriage', 'betrayal'). The model
    decides from the word and its facts; without a model, every subject is kept."""
    if subject.get("kind") == "topic" or not vision:
        return True
    prompt = (f"Video game term: {subject['label']}. Facts: {subject.get('facts') or 'none'}\n"
              "Could a short clip of gameplay footage SHOW this on screen as a specific character, creature, place or "
              "object (YES), or is it an event, feeling or idea that footage cannot show by itself (NO)? "
              'Reply with JSON only: {"visual": true or false}')
    try:
        reply = http_json(f"{OLLAMA_URL}/api/chat", {"model": vision["model"], "stream": False, "format": "json",
                                                      "options": {"temperature": 0},
                                                      "messages": [{"role": "user", "content": prompt}]}, timeout=60)
        data = json.loads(re.search(r"\{.*\}", reply.get("message", {}).get("content", ""), re.S).group(0))
        return data.get("visual") is True or str(data.get("visual")).lower() == "true"
    except Exception:
        return True


def run_coverage(subjects, assets, yt_dlp, ffmpeg, game, used, stamp, vision):
    report, pinned = [], set()
    print("\nChecking that the key subjects each have a clip"
          + (" (looking at the footage)..." if vision else " (titles only; no vision model)..."), flush=True)
    for subject in subjects:
        if not is_visual_subject(vision, subject):
            print(f"  '{subject['label']}': an idea, not something footage can show; skipped.", flush=True)
            report.append({"subject": subject["label"], "status": "skipped (not visual)"})
            continue
        have = next((a for a in assets if a["scene"] not in pinned and asset_shows_subject(a, subject, vision)), None)
        if have:
            pinned.add(have["scene"])
            report.append({"subject": subject["label"], "status": "covered", "scene": have["scene"],
                           "option": have.get("chosen")})
            print(f"  '{subject['label']}': covered by scene {have['scene']} (option {have.get('chosen')})", flush=True)
            continue
        target = pick_scene(subject, assets, pinned)
        if target is None:
            report.append({"subject": subject["label"], "status": "no free scene"})
            continue
        print(f"  '{subject['label']}': not shown yet. Searching for one for scene {target['scene']}...", flush=True)
        if cover_subject(subject, target, yt_dlp, ffmpeg, game, used, stamp, vision):
            pinned.add(target["scene"])
            report.append({"subject": subject["label"], "status": "added", "scene": target["scene"],
                           "option": target.get("chosen")})
            print(f"  '{subject['label']}': scene {target['scene']} now has an option showing it (selected).", flush=True)
        else:
            report.append({"subject": subject["label"], "status": "not found; review manually"})
            print(f"  '{subject['label']}': could not find footage of it. Existing clips kept; review manually.", flush=True)
    return report


# ---------------------------------------------------------------------------
# Keywords: suggested automatically, chosen by you
# ---------------------------------------------------------------------------
def split_keywords(text):
    return [k.strip() for k in re.split(r"[,;]", text) if k.strip()]


def dedupe(items):
    seen, out = set(), []
    for item in items:
        if item.casefold() not in seen:
            seen.add(item.casefold())
            out.append(item)
    return out


def suggest_keywords(sentence, entities, evid_counts, game):
    """Names from the line first, then repeated story nouns from it (e.g. 'orphans')."""
    picks = list(entities[:2])
    skip = word_set(game)
    for ent in entities:
        skip |= word_set(ent)
    nouns = []
    for w in re.findall(r"[a-z]{5,}", sentence.casefold()):
        if w in STOPWORDS or w in GENERIC_WORDS or w in skip or w in nouns:
            continue
        if w.endswith(("ing", "ed", "ly", "tion", "ment", "ness")):
            continue
        if evid_counts.get(w, 0) >= 2:
            nouns.append(w)
    nouns.sort(key=lambda w: -evid_counts[w])
    return dedupe(picks + nouns[:2])


def ask_keywords(index, total, section, sentence, suggested):
    print(f"\nScene {index}/{total} [{section}]: {sentence}")
    print(f"  Suggested keywords: {', '.join(suggested) or '(none)'}")
    try:
        reply = input("  Your keywords (Enter = use these; type your own, comma-separated; "
                      "start with + to add to the suggestions): ").strip()
    except EOFError:
        return suggested
    if not reply:
        return suggested
    if reply.startswith("+"):
        return dedupe(suggested + split_keywords(reply[1:]))
    return split_keywords(reply) or suggested


def keywords_path(draft):
    return MANIFEST_DIR / f"{draft.stem}_keywords.json"


def load_saved_keywords(draft, lines):
    path = keywords_path(draft)
    if not path.exists():
        return {}
    try:
        saved = json.loads(path.read_text(encoding="utf-8")).get("lines", [])
    except (OSError, json.JSONDecodeError):
        return {}
    return {i: e["keywords"] for i, e in enumerate(saved, 1)
            if i <= len(lines) and e.get("sentence") == lines[i - 1][1] and e.get("keywords")}


def save_keywords(draft, lines, by_scene):
    entries = [{"scene": i, "sentence": lines[i - 1][1], "keywords": by_scene.get(i, [])}
               for i in range(1, len(lines) + 1)]
    keywords_path(draft).write_text(json.dumps({"lines": entries}, indent=2, ensure_ascii=False), encoding="utf-8")


def queries_from_keywords(game, keywords, sentence, topic):
    """Every keyword is searched, alone and paired with each of the others, interleaved so each one gets an early turn:
    k1+k2, k1, k1+k3, k2, k2+k3, k3, k1+k4, k4 ... then the topic. Put the most specific keyword first."""
    suffix = footage_suffix(sentence)
    short_topic = " ".join(re.sub(r"^(?:the|a|an)\s+", "", topic, flags=re.I).split()[:6])
    ks = dedupe([k.strip() for k in keywords if k.strip()])
    pairs = [(ks[i], ks[j]) for j in range(1, len(ks)) for i in range(j)]
    queries = []
    for n in range(max(len(pairs), len(ks))):
        if n < len(pairs):
            queries.append(f"{game} {pairs[n][0]} {pairs[n][1]} {suffix}")
        if n < len(ks):
            queries.append(f"{game} {ks[n]} {suffix}")
    queries = dedupe([clean(q) for q in queries])[:MAX_QUERIES_PER_SCENE - (2 if short_topic else 0)]
    if short_topic:
        queries += [clean(f"{game} {short_topic} {suffix}"), clean(f"{game} {short_topic} all cutscenes 4K no HUD")]
    if not queries:
        queries.append(f"{game} walkthrough no commentary")
    return dedupe(queries)[:MAX_QUERIES_PER_SCENE]


def keep_clip(review):
    """A clip is offered only if the vision check scored its relevance at least MIN_KEEP_RELEVANCE. A clip whose
    review failed (timeout) cannot be judged, so it is kept as a last resort, marked unreviewed."""
    return review is None or review["relevance"] >= MIN_KEEP_RELEVANCE


# ---------------------------------------------------------------------------
# Setup and per-scene search (also used by review_clips.py for "search again")
# ---------------------------------------------------------------------------
def build_context():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    draft = find_latest_approved_draft()
    text = draft.read_text(encoding="utf-8-sig")
    evidence_text = EVIDENCE_FILE.read_text(encoding="utf-8-sig") if EVIDENCE_FILE.exists() else ""

    # Generated drafts carry no GAME/TOPIC lines; the evidence file they came from does.
    game = read_metadata(text, "GAME") or read_metadata(evidence_text, "GAME")
    topic = read_metadata(text, "TOPIC") or read_metadata(evidence_text, "TOPIC")
    if not game:
        raise ValueError(f"No GAME metadata in {draft.name} or {EVIDENCE_FILE.name}. Refusing to guess a game.")
    game_slug = re.sub(r"[^a-z0-9]+", "_", game.casefold()).strip("_")
    if not draft.name.casefold().startswith(game_slug):
        print(f"WARNING: draft filename doesn't start with '{game_slug}'. Evidence file says the game is '{game}'. "
              "Check they belong together.")
    spoken = extract_spoken_lines(text)
    if not spoken:
        raise ValueError(f"No tagged narration lines found in {draft.name}.")
    lines = []
    for section, sentence in spoken:
        source_line = next((ln for ln in text.splitlines()
                            if ln.strip().startswith(section + ":") and sentence[:30] in ln), "")
        tail = re.search(r"((?:\s*\[E\d+\])+)\s*$", source_line)
        lines.append((section, sentence, re.findall(r"E\d+", tail.group(1)) if tail else []))

    yt_dlp = shutil.which("yt-dlp")
    if not yt_dlp:
        candidate = Path(r"C:\pinokio\api\comfy.git\app\env\Scripts\yt-dlp.exe")
        if candidate.exists(): yt_dlp = str(candidate)
    if not yt_dlp:
        raise FileNotFoundError("yt-dlp not found. Run: python -m pip install -U yt-dlp")
    ffmpeg = shutil.which("ffmpeg")
    pinokio_ffmpeg = Path(r"C:\pinokio\bin\ffmpeg.exe")
    if not ffmpeg and pinokio_ffmpeg.exists(): ffmpeg = str(pinokio_ffmpeg)
    if not ffmpeg:
        raise FileNotFoundError("FFmpeg not found. Install it or add ffmpeg.exe to PATH.")

    vision_model = pick_vision_model()
    vision = {"model": vision_model, "ffmpeg": ffmpeg} if vision_model else None
    if vision:
        print(f"Visual review: ON (Ollama model '{vision_model}'). Each candidate clip is looked at before it's kept.",
              flush=True)
    else:
        print("Visual review: OFF. No vision model found in Ollama, so clips are judged by title only.\n"
              "  For much better matches run:  ollama pull qwen2.5vl:7b   (about 6 GB, fits your 12 GB card)", flush=True)
    known = collect_known([evidence_text, topic] + [s for _, s, _ in lines])
    return {"draft": draft, "game": game, "topic": topic, "lines": lines, "evidence_text": evidence_text,
            "yt_dlp": yt_dlp, "ffmpeg": ffmpeg, "vision": vision, "known": known, "used": {},
            "evid_counts": Counter(re.findall(r"[a-z]{5,}", evidence_text.casefold())),
            "stamp": datetime.now().strftime("%Y%m%d_%H%M%S")}


def scene_entities(ctx, index):
    _, sentence, citations = ctx["lines"][index - 1]
    entities = extract_entities(sentence, ctx["known"], ctx["game"])
    if len(entities) < 2:
        cited = " ".join(evidence_excerpt(ctx["evidence_text"], c) for c in citations)
        for sent in sentences_of(cited):               # one sentence at a time, so sentence-initial words aren't names
            for ent in extract_entities(sent, ctx["known"], ctx["game"]):
                if ent.casefold() not in {e.casefold() for e in entities}:
                    entities.append(ent)
        entities = entities[:3]
    return entities


def process_scene(ctx, index, keywords):
    """Search, download and review clips for one line; keep the best two as options A and B."""
    section, sentence, citations = ctx["lines"][index - 1]
    game, topic, vision = ctx["game"], ctx["topic"], ctx["vision"]
    yt_dlp, ffmpeg, used = ctx["yt_dlp"], ctx["ffmpeg"], ctx["used"]
    entities = scene_entities(ctx, index)
    terms = relevance_terms(list(keywords), topic, game)
    queries = queries_from_keywords(game, keywords, sentence, topic)
    base = f"{ctx['stamp']}_scene_{index:02d}"
    clear_scene_files(base)
    evidence_for_scene = " ".join(evidence_excerpt(ctx["evidence_text"], c) for c in citations)[:600]

    spoken = dialogue_words(vision, game, sentence)
    if spoken:
        print(f"Listening for: {', '.join(spoken)}", flush=True)
    asset = {"scene": index, "section": section, "narration": sentence, "keywords": list(keywords),
             "entities": entities, "citation": ",".join(citations), "status": "failed", "use_clip": False,
             "options": [], "queries_tried": [], "rejected": []}
    pool, seen_urls, tried, stop = [], set(), 0, False
    for query in queries:
        print(f"Search: {query}", flush=True)
        entries, search_error = search_youtube(yt_dlp, query)
        candidates = rank_results(entries, terms, game, relaxed=bool(vision))
        record = {"query": query, "results": len(entries), "relevant": len(candidates)}
        if search_error:
            record["error"] = search_error
        asset["queries_tried"].append(record)
        print(f"  -> {len(entries)} results, {len(candidates)} relevant"
              + (f"  [{search_error}]" if search_error else ""), flush=True)

        fresh = [(e, h) for e, h in candidates if (e.get("url") or e.get("webpage_url")) not in seen_urls]
        for entry, hits in fresh[:MAX_DOWNLOAD_TRIES]:   # videos already tried do not use up this search's turns
            if tried >= MAX_CANDIDATES:
                stop = True
                break
            url = entry.get("url") or entry.get("webpage_url")
            if not url:
                continue
            title = entry.get("title", "")
            print(f"  Trying: {title}", flush=True)
            info = video_info(yt_dlp, url)
            duration = info.get("duration") or entry.get("duration")
            start, reason, chapter_title, dialogue, moment_score = locate_moment(
                yt_dlp, url, info, entry, keywords, sentence, game, terms, used.get(url, 0), spoken)
            print(f"  Downloading {CLIP_SECONDS}s at {fmt_secs(start)} ({reason})...", flush=True)
            if dialogue:
                print(f"    heard there: \"{dialogue[:110]}\"", flush=True)
            cand_base = f"{base}_cand{tried + 1}"
            cand_file = OUTPUT_DIR / f"{cand_base}.mp4"
            ok, error = fetch_segment(yt_dlp, ffmpeg, url, start, cand_base, cand_file)
            seen_urls.add(url)
            if not ok:
                record.setdefault("download_errors", []).append(error)
                print(f"  Download failed: {error}", flush=True)
                continue
            tried += 1
            used[url] = used.get(url, 0) + 1
            review = review_clip(vision, cand_file, game, topic, sentence, evidence_for_scene, dialogue) if vision else None
            if vision:
                show_review(review)
            if vision and not keep_clip(review):
                try: cand_file.unlink()
                except OSError: pass
                asset["rejected"].append({"source_title": title, "source_url": entry.get("webpage_url", url), "query": query,
                                          "clip_start_seconds": start, "relevance": review["relevance"], "shows": review["shows"]})
                print(f"    Not kept: relevance {review['relevance']:.0f}/10 is below {MIN_KEEP_RELEVANCE}. Trying another video.", flush=True)
                continue
            pool.append({"score": review_score(review, hits) + min(3.0, moment_score / 4), "file": str(cand_file), "query": query,
                         "source_title": title, "source_url": entry.get("webpage_url", url),
                         "source_duration": duration, "clip_start_seconds": start, "start_reason": reason,
                         "chapter_title": chapter_title, "title_relevance_score": hits, "vision_review": review,
                         "dialogue_at_clip": dialogue, "caption_match_score": moment_score})
            good = [p for p in pool if p["vision_review"]] if vision else pool
            if len(good) >= OPTIONS_PER_SCENE:
                stop = True
                break
        if stop:
            break

    pool.sort(key=lambda p: (p["vision_review"] is None, -p["score"]))   # judged clips before unreviewed ones
    keep, drop = pool[:OPTIONS_PER_SCENE], pool[OPTIONS_PER_SCENE:]
    for p in drop:
        try: Path(p["file"]).unlink()
        except OSError: pass
    for label, p in zip(OPTION_LABELS, keep):
        final = OUTPUT_DIR / f"{base}_{label}.mp4"
        Path(p["file"]).rename(final)
        p.pop("score", None)
        p.update({"label": label, "file": str(final)})
        p["preview"] = make_preview(ffmpeg, final, final.with_suffix(".preview.mp4"))
        asset["options"].append(p)
    if asset["options"]:
        select_option(asset, "A")
        unreviewed = [o["label"] for o in asset["options"] if vision and not o.get("vision_review")]
        print(f"Kept {len(asset['options'])} option(s) for scene {index}"
              + (f" (not checked by the vision model: {', '.join(unreviewed)}; look at these closely)" if unreviewed else "")
              + (f"  - only one clip reached {MIN_KEEP_RELEVANCE}/10; try other keywords for a second." if len(keep) == 1 else ""),
              flush=True)
    else:
        best = max(asset["rejected"], key=lambda r: r["relevance"], default=None)
        print(f"No clip reached {MIN_KEEP_RELEVANCE}/10 relevance for this scene after {tried} tries."
              + (f" Closest: {best['relevance']:.0f}/10, \"{best['source_title']}\" at {fmt_secs(best['clip_start_seconds'])} ({best['shows']})."
                 if best else "")
              + " Try more specific keywords (names, quest titles) and search this scene again.", flush=True)
    return asset


def save_manifest(path, manifest):
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def latest_manifest_for(draft):
    for path in sorted(MANIFEST_DIR.glob("*_manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("draft") == str(draft):
            return path, data
    return None, None


def parse_args(argv):
    args = {"auto": "--auto" in argv, "ask": "--ask" in argv, "no_coverage": "--no-coverage" in argv, "scenes": []}
    if "--scene" in argv:
        for token in argv[argv.index("--scene") + 1:]:
            if token.startswith("--"):
                break
            args["scenes"].append(int(token))
    return args


HELP = """Usage: python harvest_clips_universal.py [options]
  (no options)     ask for keywords for every line, then search; keeps two clips (A and B) per line
  --auto           don't ask; use the suggested keywords
  --ask            ask again even if keywords were saved for this draft
  --scene 3 5      search again for only these lines (asks for new keywords)
  --no-coverage    skip the check that key subjects each have a clip
"""


def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP)
        return
    args = parse_args(sys.argv[1:])
    ctx = build_context()
    draft, game, topic, lines = ctx["draft"], ctx["game"], ctx["topic"], ctx["lines"]
    print(f"Draft: {draft.name}\nGame: {game}\nTopic: {topic}\nNarration scenes: {len(lines)}")

    rerun = bool(args["scenes"])
    if rerun:
        manifest_path, manifest = latest_manifest_for(draft)
        if not manifest:
            raise SystemExit("No earlier run found for this draft. Run without --scene first.")
        ctx["stamp"] = manifest["created_at"]
        todo = [i for i in args["scenes"] if 1 <= i <= len(lines)]
    else:
        manifest = {"draft": str(draft), "game": game, "topic": topic, "created_at": ctx["stamp"],
                    "warning": "Downloaded clips are candidates only; visual relevance and rights must be reviewed manually.",
                    "assets": []}
        manifest_path = MANIFEST_DIR / f"{ctx['stamp']}_manifest.json"
        todo = list(range(1, len(lines) + 1))

    # 1. Keywords for every line first, so you can answer them all and then leave it running.
    saved = load_saved_keywords(draft, lines)
    chosen = dict(saved)
    interactive = sys.stdin.isatty() and not args["auto"]
    for i in todo:
        suggested = suggest_keywords(lines[i - 1][1], scene_entities(ctx, i), ctx["evid_counts"], game)
        if i in saved and not args["ask"] and not rerun:
            print(f"\nScene {i}: using saved keywords: {', '.join(saved[i])}")
        elif interactive:
            chosen[i] = ask_keywords(i, len(lines), lines[i - 1][0], lines[i - 1][1], suggested)
        else:
            chosen[i] = suggested
            print(f"\nScene {i}: keywords: {', '.join(suggested) or '(none)'}")
    save_keywords(draft, lines, chosen)

    # 2. Search
    run_started = time.time()
    for n, index in enumerate(todo, 1):
        scene_started = time.time()
        print(f"\n[{index}/{len(lines)}] {lines[index - 1][0]}: {lines[index - 1][1]}\nKeywords: {', '.join(chosen[index])}")
        asset = process_scene(ctx, index, chosen[index])
        existing = [a for a in manifest["assets"] if a["scene"] == index]
        if existing:
            manifest["assets"][manifest["assets"].index(existing[0])] = asset
        else:
            manifest["assets"].append(asset)
            manifest["assets"].sort(key=lambda a: a["scene"])
        save_manifest(manifest_path, manifest)
        spent = time.time() - run_started
        left = (spent / n) * (len(todo) - n)
        print(f"Scene {index} took {fmt_secs(time.time() - scene_started)}. Total {fmt_secs(spent)}; "
              f"about {fmt_secs(left)} left for {len(todo) - n} scene(s).", flush=True)

    # 3. Make sure the key subjects are shown somewhere
    if not rerun and not args["no_coverage"]:
        subjects = story_subjects([(a, b) for a, b, _ in lines], ctx["evidence_text"], topic, game, ctx["known"])
        print("\nKey subjects: " + ", ".join(sj["label"] + (f" (near {sj['context']})" if sj["context"] else "")
                                             for sj in subjects), flush=True)
        manifest["coverage"] = run_coverage(subjects, manifest["assets"], ctx["yt_dlp"], ctx["ffmpeg"], game,
                                            ctx["used"], ctx["stamp"], ctx["vision"])
    if ctx["vision"]:
        unload_vision(ctx["vision"]["model"])
    save_manifest(manifest_path, manifest)

    got = sum(1 for a in manifest["assets"] if a.get("options"))
    print(f"\nScenes with clips: {got}/{len(lines)}\nManifest saved: {manifest_path}\n"
          "NEXT: run  python review_clips.py  to watch the A and B options for each line, pick one, or search\n"
          "again with different keywords. Until you pick, option A is used.\n"
          "IMPORTANT: these are search-result candidates, not verified matches. Confirm licensing/platform "
          "permissions before publishing.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
