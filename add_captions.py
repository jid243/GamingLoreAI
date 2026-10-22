r"""
add_captions.py  -  OpusClip-style captions for a video you have already edited.

    python add_captions.py "C:\path\to\your_video.mp4"

What it does
  1. Listens to the video's audio (faster-whisper) and gets the exact time of every spoken word.
  2. If it finds your script (the newest draft in Desktop\GamingLoreAI\scripts, or --script FILE), it uses the
     script's spelling for names such as "Downwarren". It never adds words that weren't spoken.
  3. Burns in captions that copy the OpusClip look: heavy uppercase type with a thick black outline, the word being
     spoken big and coloured (green; yellow for names and numbers), the other word small and white, a quick pop on
     every word, placed at the middle of the screen.
  4. Saves  <video name>_captioned.mp4  next to the original, plus the .ass and .srt caption files.

Install once:   python -m pip install faster-whisper pillow
Use a video WITHOUT existing captions. (An OpusClip export has its own captions and logo burned in.)

Common options
  --words 1          strictly one word on screen at a time (default 2: a long word plus a short one like "THE")
  --position 0.62    vertical centre of the captions, 0 = top, 1 = bottom (default 0.50, like OpusClip)
  --active yellow    colour of the spoken word: green, yellow, white or a hex code like #00F020
  --no-keywords      don't turn names and numbers yellow
  --size 120         size of the spoken word at 1080x1920 (default 110)
  --font "Name"      use another installed font;  --font-file path\to\font.ttf  to use a font file
  --model medium.en  more accurate speech recognition (slower);  --language en
  --gpu              encode with your NVIDIA card (faster)
"""
import argparse
import difflib
import json
import re
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT = Path.home() / "Desktop" / "GamingLoreAI"
SCRIPT_DIR = PROJECT / "scripts"
FONT_DIRS = [PROJECT / "fonts", Path(__file__).resolve().parent / "fonts"]

# ---- look (sizes are for a 1080x1920 video and scale automatically) ----
ACTIVE_SIZE = 110              # the word being spoken
INACTIVE_RATIO = 0.80          # the other word on screen, relative to the spoken word
GREEN = (0, 240, 32)
YELLOW = (255, 240, 0)
WHITE = (255, 255, 255)
OUTLINE = 8
SHADOW = 3
POSITION = 0.50                # vertical centre of the captions (fraction of the height)
MAX_LINE_WIDTH = 0.88          # widest a caption may be (fraction of the video width)
MAX_WORDS = 2
MAX_CHARS = 14                 # a long word is shown alone; short words may share the screen with it
STRIP_PUNCTUATION = True
POP = (82, 108, 100)           # start, overshoot and settled size in percent
POP_MS = (90, 170)

FONT_URLS = [
    "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-ExtraBold.ttf",
    "https://raw.githubusercontent.com/JulietaUla/Montserrat/master/fonts/ttf/Montserrat-ExtraBold.ttf",
    "https://github.com/google/fonts/raw/main/ofl/montserrat/static/Montserrat-ExtraBold.ttf",
]
PRONOUN_I = {"i", "i'm", "i'll", "i've", "i'd", "i’m", "i’ll", "i’ve", "i’d"}
STOPWORDS = set("""a about above after again all also am an and any are as at be because been before being below between
both but by can could did do does doing down during each few for from further had has have having he her here hers
him his how if in into is it its just me more most my no nor not of off on once only or other our out over own same
she should so some such than that the their them then there these they this those through to too under until up
very was we were what when where which while who whom why will with would you your""".split())


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def fmt_secs(seconds):
    seconds = int(seconds)
    return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def find_tool(name, extra=None):
    found = shutil.which(name)
    if found:
        return found
    if extra and Path(extra).exists():
        return str(extra)
    raise FileNotFoundError(f"{name} not found. Install it or add it to PATH.")


def probe(ffmpeg, path):
    r = subprocess.run([ffmpeg, "-nostdin", "-i", str(path)], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", stdin=subprocess.DEVNULL)
    d = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr)
    v = re.search(r"Video:.*?(\d{3,5})x(\d{3,5})", r.stderr)
    if not d or not v:
        raise RuntimeError(f"Could not read the video: {path}")
    return int(d.group(1)) * 3600 + int(d.group(2)) * 60 + float(d.group(3)), int(v.group(1)), int(v.group(2))


def ass_color(rgb):
    return f"&H00{rgb[2]:02X}{rgb[1]:02X}{rgb[0]:02X}&"


def inline_color(rgb):
    return f"&H{rgb[2]:02X}{rgb[1]:02X}{rgb[0]:02X}&"


def parse_color(text):
    named = {"green": GREEN, "yellow": YELLOW, "white": WHITE}
    if text.lower() in named:
        return named[text.lower()]
    m = re.fullmatch(r"#?([0-9a-fA-F]{6})", text.strip())
    if not m:
        raise SystemExit(f"Unknown colour '{text}'. Use green, yellow, white or a hex code like #00F020.")
    v = m.group(1)
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


# ---------------------------------------------------------------------------
# Fonts: Montserrat ExtraBold is what OpusClip's look is built on
# ---------------------------------------------------------------------------
def font_names(path):
    """Family and style names from a TrueType/OpenType file, so the caption style matches it exactly."""
    data = Path(path).read_bytes()
    base = 0
    if data[:4] == b"ttcf":
        base = struct.unpack(">I", data[12:16])[0]
    count = struct.unpack(">H", data[base + 4:base + 6])[0]
    table = None
    for i in range(count):
        tag, _, off, _length = struct.unpack(">4sIII", data[base + 12 + 16 * i:base + 28 + 16 * i])
        if tag == b"name":
            table = off
    if table is None:
        return None, None
    _fmt, n, string_off = struct.unpack(">HHH", data[table:table + 6])
    names = {}
    for i in range(n):
        pid, _eid, lid, nid, length, off = struct.unpack(">6H", data[table + 6 + 12 * i:table + 18 + 12 * i])
        raw = data[table + string_off + off:table + string_off + off + length]
        text = raw.decode("utf-16-be" if pid in (0, 3) else "latin-1", errors="ignore").strip("\x00")
        if text and (nid not in names or (pid == 3 and lid == 0x409)):
            names[nid] = text
    return names.get(1), names.get(2)


def download_montserrat(dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / "Montserrat-ExtraBold.ttf"
    print("Downloading the Montserrat ExtraBold font (free, open licence) for the OpusClip look...", flush=True)
    for url in FONT_URLS:
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
        except Exception:
            continue
        if len(data) > 40000 and data[:4] in (b"\x00\x01\x00\x00", b"OTTO", b"true"):
            target.write_bytes(data)
            return target
    return None


def resolve_font(args):
    """Returns {'family', 'bold', 'file'}. Order: --font-file, --font, Montserrat in a fonts folder,
    Montserrat downloaded, Arial Black, Arial."""
    if args.font_file:
        path = Path(args.font_file.strip('"'))
        if not path.exists():
            raise SystemExit(f"Font file not found: {path}")
        family, style = font_names(path)
        return {"family": family or path.stem, "bold": int((style or "").lower() in ("bold", "bold italic")), "file": path}
    if args.font:
        return {"family": args.font, "bold": 1, "file": None}
    for folder in FONT_DIRS:
        if folder.exists():
            for pattern in ("Montserrat*ExtraBold*", "Montserrat*Black*", "Montserrat*Bold*"):
                hits = sorted(p for p in folder.glob(pattern) if p.suffix.lower() in (".ttf", ".otf"))
                if hits:
                    family, style = font_names(hits[0])
                    return {"family": family or hits[0].stem,
                            "bold": int((style or "").lower() in ("bold", "bold italic")), "file": hits[0]}
    downloaded = download_montserrat(FONT_DIRS[0])
    if downloaded:
        family, style = font_names(downloaded)
        return {"family": family or "Montserrat ExtraBold", "bold": 0, "file": downloaded}
    for windows_font in (Path(r"C:\Windows\Fonts\ariblk.ttf"), Path(r"C:\Windows\Fonts\impact.ttf")):
        if windows_font.exists():
            family, style = font_names(windows_font)
            print(f"Could not download Montserrat; using {family or windows_font.stem} instead.\n"
                  f"  For the exact look, save Montserrat-ExtraBold.ttf into {FONT_DIRS[0]} and run again.")
            return {"family": family or windows_font.stem, "bold": 0, "file": windows_font}
    print("No heavy font found; using Arial Bold. Save Montserrat-ExtraBold.ttf into "
          f"{FONT_DIRS[0]} for the OpusClip look.")
    return {"family": "Arial", "bold": 1, "file": None}


class Measurer:
    """Width of a word in pixels, using the real font when Pillow is installed."""

    def __init__(self, font_file):
        self.cache = {}
        self.font_file = font_file
        self.pil = None
        try:
            from PIL import ImageFont
            if font_file:
                self.pil = ImageFont
        except ImportError:
            pass

    def width(self, text, size):
        key = (text, int(size))
        if key not in self.cache:
            em = size / 1.2                     # libass sizes text so ascent + descent equals the font size
            if self.pil:
                try:
                    font = self.pil.truetype(str(self.font_file), 200)
                    self.cache[key] = font.getlength(text) * em / 200 * 1.03
                    return self.cache[key]
                except Exception:
                    self.pil = None
            self.cache[key] = len(text) * 0.92 * em       # heavy caps average a bit under one em
        return self.cache[key]


# ---------------------------------------------------------------------------
# Speech -> word timings
# ---------------------------------------------------------------------------
def extract_audio(ffmpeg, video, wav):
    r = subprocess.run([ffmpeg, "-nostdin", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       stdin=subprocess.DEVNULL)
    if r.returncode != 0 or not wav.exists():
        raise RuntimeError("Could not read audio from the video. Does it have a sound track?\n" + r.stderr[-400:])


def transcribe_words(audio_path, model_name, language, prompt, total):
    from faster_whisper import WhisperModel
    last = None
    for device, ctype in (("cuda", "float16"), ("cpu", "int8")):
        try:
            print(f"Listening to the audio ({model_name} on {device}; the first run downloads the model)...", flush=True)
            model = WhisperModel(model_name, device=device, compute_type=ctype)
            segments, _ = model.transcribe(str(audio_path), word_timestamps=True, language=language or None,
                                           beam_size=5, initial_prompt=prompt[:600] or None)
            words = []
            for seg in segments:
                for w in (seg.words or []):
                    if w.word.strip():
                        words.append({"text": w.word.strip(), "start": float(w.start), "end": float(w.end)})
                sys.stdout.write(f"\r  heard up to {fmt_secs(seg.end)} of {fmt_secs(total)}".ljust(60))
                sys.stdout.flush()
            sys.stdout.write("\r" + " " * 60 + "\r")
            if words:
                return words
            raise RuntimeError("no speech found")
        except Exception as exc:
            last = exc
            print(f"\n  ({device} failed: {str(exc)[:140]})", flush=True)
    raise RuntimeError(f"Speech recognition failed: {last}")


def norm(word):
    return re.sub(r"[^a-z0-9]", "", word.casefold())


def find_script_tokens(path):
    """Words of the script, from a draft (HOOK:/NARRATION:/... lines) or any plain text file."""
    text = path.read_text(encoding="utf-8-sig")
    spoken = []
    for raw in text.splitlines():
        m = re.match(r"^\s*(HOOK|NARRATION|PAYOFF|ENDING QUESTION)\s*:\s*(.*?)\s*$", raw, re.I)
        if m:
            spoken.append(re.sub(r"\[E\d+\]", "", m.group(2)))
    body = " ".join(spoken) if spoken else text
    return [t for t in body.split() if norm(t)]


def apply_script_spelling(heard, script_tokens):
    """Fix the spelling/capitalisation of heard words from the script. Words that were not heard are never added."""
    a = [norm(h["text"]) for h in heard]
    b = [norm(t) for t in script_tokens]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    out, fixed = [], 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                out.append(dict(heard[i1 + k], text=script_tokens[j1 + k]))
            fixed += i2 - i1
        elif tag == "replace":
            if i2 - i1 == j2 - j1:
                for k in range(i2 - i1):
                    out.append(dict(heard[i1 + k], text=script_tokens[j1 + k]))
                fixed += i2 - i1
            elif j2 - j1 == 1:                      # e.g. heard "downward in" for the script's "Downwarren"
                joined = "".join(a[i1:i2])
                if difflib.SequenceMatcher(None, joined, b[j1]).ratio() >= 0.6:
                    out.append({"text": script_tokens[j1], "start": heard[i1]["start"], "end": heard[i2 - 1]["end"]})
                    fixed += 1
                else:
                    out.extend(heard[i1:i2])
            else:
                out.extend(heard[i1:i2])
        elif tag == "delete":
            out.extend(heard[i1:i2])
    ratio = fixed / len(heard) if heard else 0.0
    return out, ratio


# ---------------------------------------------------------------------------
# Words -> caption events
# ---------------------------------------------------------------------------
def prepare_words(words, strip_punct):
    prepared, prev_ends_sentence = [], True
    for w in words:
        raw = w["text"]
        core = re.sub(r"^[^\w$%#]+|[^\w$%?!]+$", "", raw) if strip_punct else raw
        core = core or raw
        letters = re.sub(r"[^A-Za-z]", "", core)
        is_key = bool(re.search(r"\d", core)) or (len(letters) >= 2 and letters.isupper() and core.upper() == core)
        if not is_key and not prev_ends_sentence and core[:1].isupper() and core.casefold() not in PRONOUN_I:
            is_key = True                                  # a capitalised word mid-sentence: a name
        prepared.append({"text": core.upper(), "start": w["start"], "end": max(w["end"], w["start"] + 0.08),
                         "key": is_key, "filler": norm(raw) in STOPWORDS,
                         "ends_sentence": bool(re.search(r"[.!?]$", raw)),
                         "ends_clause": bool(re.search(r"[,;:]$", raw))})
        prev_ends_sentence = prepared[-1]["ends_sentence"]
    for a, b in zip(prepared, prepared[1:]):               # no overlaps
        a["end"] = min(a["end"], max(b["start"], a["start"] + 0.06))
    return prepared


def group_words(words, max_words, max_chars):
    groups, cur = [], []
    for w in words:
        if cur:
            chars = sum(len(x["text"]) for x in cur) + len(cur) + len(w["text"])
            if (len(cur) >= max_words or chars > max_chars or w["start"] - cur[-1]["end"] > 0.45
                    or cur[-1]["ends_sentence"] or cur[-1]["ends_clause"]):
                groups.append(cur)
                cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    return groups


def ass_time(t):
    cs = int(round(max(0.0, t) * 100))
    return f"{cs // 360000}:{(cs // 6000) % 60:02d}:{(cs // 100) % 60:02d}.{cs % 100:02d}"


def build_ass(words, width, height, font, measurer, opts):
    scale = height / 1920.0
    active_fs = opts["size"] * scale
    inactive_fs = active_fs * INACTIVE_RATIO
    cx, cy = width / 2, height * opts["position"]
    groups = group_words(prepare_words(words, STRIP_PUNCTUATION), opts["words"], opts["max_chars"])
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font['family']},{active_fs:.0f},{ass_color(WHITE)},{ass_color(WHITE)},&H00000000&,&H80000000&,{font['bold']},0,0,0,100,100,0,0,1,{OUTLINE * scale:.1f},{SHADOW * scale:.1f},5,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for gi, group in enumerate(groups):
        # Size for the widest case (the group's widest word being the active one) so nothing leaves the screen.
        wide = [measurer.width(w["text"], active_fs) for w in group]
        narrow = [measurer.width(w["text"], inactive_fs) for w in group]
        worst = sum(narrow) + max(x - y for x, y in zip(wide, narrow)) + (len(group) - 1) * measurer.width(" ", active_fs)
        fit = min(1.0, width * MAX_LINE_WIDTH / max(1.0, worst + 2 * OUTLINE * scale))
        next_start = groups[gi + 1][0]["start"] if gi + 1 < len(groups) else None
        for i, active in enumerate(group):
            start = max(0.0, active["start"] - 0.02)
            if i + 1 < len(group):
                end = group[i + 1]["start"] - 0.02
            else:
                end = active["end"] + 0.18
                if next_start is not None:
                    end = min(end, next_start - 0.02)
            end = max(end, start + 0.12)
            color = YELLOW if (active["key"] and opts["keywords"]) else opts["active"]
            runs = []
            for j, w in enumerate(group):
                lead = " " if j else ""
                if j == i:
                    tags = (f"\\fs{active_fs * fit:.0f}\\c{inline_color(color)}\\fscx{POP[0]}\\fscy{POP[0]}"
                            f"\\t(0,{POP_MS[0]},\\fscx{POP[1]}\\fscy{POP[1]})"
                            f"\\t({POP_MS[0]},{POP_MS[1]},\\fscx{POP[2]}\\fscy{POP[2]})")
                else:
                    tags = f"\\fs{inactive_fs * fit:.0f}\\c{inline_color(WHITE)}\\fscx100\\fscy100"
                runs.append(f"{{{tags}}}{lead}{w['text']}")
            events.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,"
                          f"{{\\an5\\pos({cx:.0f},{cy:.0f})}}{''.join(runs)}")
    return head + "\n".join(events) + "\n", groups


def srt_time(t):
    ms = int(round(max(0.0, t) * 1000))
    return f"{ms // 3600000:02d}:{(ms // 60000) % 60:02d}:{(ms // 1000) % 60:02d},{ms % 1000:03d}"


def build_srt(groups):
    blocks = []
    for n, g in enumerate(groups, 1):
        blocks.append(f"{n}\n{srt_time(g[0]['start'])} --> {srt_time(g[-1]['end'] + 0.1)}\n{' '.join(w['text'] for w in g)}\n")
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Burn in
# ---------------------------------------------------------------------------
def burn(ffmpeg, video, ass_text, font, out_path, duration, gpu):
    work = out_path.parent / f"_captions_work_{int(time.time())}"
    work.mkdir(parents=True, exist_ok=True)
    try:
        (work / "captions.ass").write_text(ass_text, encoding="utf-8")
        vf = "subtitles=captions.ass"
        if font["file"]:
            fonts = work / "fonts"
            fonts.mkdir()
            shutil.copy(font["file"], fonts / Path(font["file"]).name)
            vf += ":fontsdir=fonts"
        encoder = (["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "19", "-b:v", "0"] if gpu
                   else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "17"])
        cmd = [ffmpeg, "-hide_banner", "-nostdin", "-y", "-loglevel", "verbose", "-progress", "pipe:1", "-nostats",
               "-i", str(video), "-vf", vf, *encoder, "-pix_fmt", "yuv420p", "-c:a", "copy",
               "-movflags", "+faststart", str(out_path)]
        proc = subprocess.Popen(cmd, cwd=work, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        started, log = time.time(), []
        for line in proc.stdout:
            log.append(line)
            m = re.match(r"out_time_us=(\d+)", line)
            if m:
                done = int(m.group(1)) / 1_000_000
                pct = min(100, done / duration * 100) if duration else 0
                spent = time.time() - started
                left = spent / max(pct, 0.1) * (100 - pct)
                sys.stdout.write(f"\r  Rendering: {pct:3.0f}%  ({fmt_secs(spent)} so far, about {fmt_secs(left)} left)".ljust(70))
                sys.stdout.flush()
        proc.wait()
        sys.stdout.write("\r" + " " * 70 + "\r")
        if proc.returncode != 0:
            tail = "".join(ln for ln in log[-40:] if not ln.startswith(("out_time", "frame=", "fps=", "stream_", "bitrate=",
                           "total_size", "dup_frames", "drop_frames", "speed=", "progress=")))[-1200:]
            raise RuntimeError(f"FFmpeg failed:\n{tail}")
        picked = [ln.strip() for ln in log if "fontselect" in ln]
        return picked
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------
def pick_video(args):
    if args.video:
        path = Path(args.video.strip().strip('"'))
    else:
        entered = input("Path to your edited video (you can drag the file into this window): ").strip().strip('"')
        path = Path(entered)
    if not path.exists():
        raise SystemExit(f"Video not found: {path}")
    return path.resolve()


def newest_draft():
    drafts = [p for p in SCRIPT_DIR.glob("*_REVIEW_REQUIRED.txt") if p.is_file()] if SCRIPT_DIR.exists() else []
    return max(drafts, key=lambda p: p.stat().st_mtime) if drafts else None


def main():
    ap = argparse.ArgumentParser(description="OpusClip-style one-word-at-a-time captions for a finished video.")
    ap.add_argument("video", nargs="?")
    ap.add_argument("--words", type=int, default=MAX_WORDS)
    ap.add_argument("--position", type=float, default=POSITION)
    ap.add_argument("--size", type=float, default=ACTIVE_SIZE)
    ap.add_argument("--active", default="green")
    ap.add_argument("--no-keywords", action="store_true")
    ap.add_argument("--font")
    ap.add_argument("--font-file")
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--language", default="en")
    ap.add_argument("--script")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    ffmpeg = find_tool("ffmpeg", r"C:\pinokio\bin\ffmpeg.exe")
    video = pick_video(args)
    duration, width, height = probe(ffmpeg, video)
    print(f"Video: {video.name}  ({width}x{height}, {fmt_secs(duration)})")
    if width > height:
        print("  Note: this video is wider than tall. The captions will work, but they're designed for 9:16.")

    # script (optional)
    script_path = Path(args.script.strip('"')) if args.script else newest_draft()
    script_tokens = find_script_tokens(script_path) if script_path and script_path.exists() else []
    prompt = " ".join(script_tokens)

    wav = video.with_name(f"_{video.stem}_audio.wav")
    try:
        extract_audio(ffmpeg, video, wav)
        try:
            heard = transcribe_words(wav, args.model, args.language, prompt, duration)
        except ImportError:
            raise SystemExit("Missing dependency. Run:  python -m pip install faster-whisper pillow")
    finally:
        try: wav.unlink()
        except OSError: pass
    print(f"Heard {len(heard)} words.")

    words = heard
    if script_tokens:
        fixed_words, ratio = apply_script_spelling(heard, script_tokens)
        if ratio >= 0.6:
            words = fixed_words
            print(f"Using the spelling from your script ({script_path.name}); {ratio:.0%} of the spoken words matched it.")
        else:
            print(f"Script found ({script_path.name}) but only {ratio:.0%} matches this video, so it was ignored.")

    font = resolve_font(args)
    measurer = Measurer(font["file"])
    opts = {"words": max(1, args.words), "max_chars": MAX_CHARS if args.words > 1 else 0, "position": args.position,
            "size": args.size, "active": parse_color(args.active), "keywords": not args.no_keywords}
    ass_text, groups = build_ass(words, width, height, font, measurer, opts)

    out_video = video.with_name(f"{video.stem}_captioned.mp4")
    out_video.with_suffix(".ass").write_text(ass_text, encoding="utf-8")
    out_video.with_suffix(".srt").write_text(build_srt(groups), encoding="utf-8")
    (video.with_name(f"{video.stem}_words.json")).write_text(json.dumps(words, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Captions: {sum(len(g) for g in groups)} words in {len(groups)} groups. Rendering...")
    picked = burn(ffmpeg, video, ass_text, font, out_video, duration, args.gpu)
    if picked and not any(font["family"].casefold() in p.casefold() for p in picked):
        print(f"WARNING: FFmpeg may not have used the font '{font['family']}'. If the type looks wrong, "
              f"put the font file in {FONT_DIRS[0]} and run again.")
    print(f"\nDone: {out_video}\nAlso saved: {out_video.with_suffix('.ass').name}, {out_video.with_suffix('.srt').name}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(1)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
