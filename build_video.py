r"""
build_video.py (v2)  -  narration + word-timed captions + 9:16 assembly.

Run after harvest_clips_universal.py (and optionally make_ai_inserts.py).
Needs:  python -m pip install edge-tts faster-whisper opencv-python

How it works
  1. Narration comes from a file you made (e.g. in OmniVoice Studio) or, if there is none, from edge-tts.
  2. faster-whisper listens to that audio and the script words are matched to what was actually said,
     so captions and scene changes follow the real timing (it works with ANY voice engine).
  3. Each scene is cut to its spoken length. Gameplay shots are framed on the speaker/main character
     (face detection), with AI stills cut in between.
  4. FFmpeg joins everything and burns in captions that appear word by word as they are spoken.

To use your own narration: save it as
    Desktop\GamingLoreAI\narration\<draft name>.wav     (mp3, m4a, flac and ogg also work)
<draft name> is the draft's file name without _REVIEW_REQUIRED.txt. Each run writes the text to read
into  narration\<draft name>_script.txt
"""
import asyncio, difflib, json, re, shutil, subprocess, sys, time
from datetime import datetime
from pathlib import Path

PROJECT = Path.home() / "Desktop" / "GamingLoreAI"
SCRIPT_DIR = PROJECT / "scripts"
MANIFEST_DIR = PROJECT / "clip_manifests"
OUT_DIR = PROJECT / "video_output"
AI_DIR = PROJECT / "ai_images"
NARRATION_DIR = PROJECT / "narration"

# --- narration (only used when no narration file is supplied) ---
VOICE = "en-US-AndrewMultilingualNeural"   # more natural than the older voices; list all: python -m edge_tts --list-voices
RATE = "-3%"
PITCH = "+0Hz"
LINE_GAP = 0.35                            # silence between lines (edge-tts mode)
WHISPER_MODEL = "small.en"                 # downloads once (~480 MB). "base.en" is faster, "medium.en" more accurate
WHISPER_LANGUAGE = "en"

# --- picture ---
W, H, FPS = 1080, 1920, 30
SHOT_MAX = 4.0                             # longest a single gameplay shot stays on screen (seconds)
FRAME_USE = 0.80                           # share of the source frame height kept; trims the bottom, where
                                           # the source video's own subtitles usually sit
TAIL = 0.6                                 # seconds of picture after the last word
USE_AI_INSERTS = True
AI_INSERT = 2.5                            # seconds of AI still placed mid-scene
AI_MIN_SCENE = 5.0                         # scenes shorter than this stay gameplay-only

# --- captions ---
# "phrase" matches your reference videos: small serif text, a whole phrase on two lines in the lower third,
# changing in time with the speech. "word" shows one gold word at a time instead.
CAPTION_STYLE = "phrase"
PHRASE_FONT = "Georgia"                    # any font installed on Windows, e.g. "Times New Roman", "Palatino Linotype"
PHRASE_FONT_SIZE = 58
PHRASE_MAX_WORDS = 8
PHRASE_LINE_WIDTH = 860                    # estimated pixels per line (the screen is 1080 wide)
PHRASE_MARGIN_V = 540                      # distance from the bottom edge to the text (about 70% down the screen)
FONT = "Arial"
FONT_SIZE = 84
CAPTION_MAX_WORDS = 3
CAPTION_MAX_WIDTH = 820                    # estimated pixels; the screen is 1080 wide, the rest is safe margin
CAPTION_MARGIN_V = 480                     # distance from the bottom edge
HIGHLIGHT = "&H0000D7FF&"                  # ASS colours are BGR: this is gold


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------
def find_draft():
    drafts = [p for p in SCRIPT_DIR.glob("*_REVIEW_REQUIRED.txt") if p.is_file()]
    if not drafts:
        raise FileNotFoundError(f"No *_REVIEW_REQUIRED.txt draft in {SCRIPT_DIR}")
    return max(drafts, key=lambda p: p.stat().st_mtime)


def extract_spoken_lines(text):
    # Same rules as the harvest script so scene numbers line up with the clips.
    lines = []
    for raw in text.splitlines():
        m = re.match(r"^\s*(HOOK|NARRATION|PAYOFF|ENDING QUESTION)\s*:\s*(.*?)\s*$", raw, re.I)
        if not m:
            continue
        body = re.sub(r"(?:\s*\[E\d+\])+\s*$", "", m.group(2)).strip()
        body = re.sub(r"\[E\d+\]", "", body).strip()
        if body:
            lines.append((m.group(1).upper(), body))
    return lines


def find_tool(name, extra=None):
    found = shutil.which(name)
    if found:
        return found
    if extra and Path(extra).exists():
        return str(extra)
    raise FileNotFoundError(f"{name} not found. Install it or add it to PATH.")


def run(cmd, cwd=None):
    if Path(str(cmd[0])).name.lower().startswith("ffmpeg"):
        cmd = [cmd[0], "-nostdin", *cmd[1:]]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=cwd,
                       stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        tail = "\n".join(r.stderr.strip().splitlines()[-12:])
        raise RuntimeError(f"FFmpeg failed:\n{' '.join(map(str, cmd))}\n{tail}")
    return r


def probe(ffmpeg, path):
    r = subprocess.run([ffmpeg, "-nostdin", "-i", str(path)], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", stdin=subprocess.DEVNULL)
    d = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr)
    if not d:
        raise RuntimeError(f"Could not read {path}")
    duration = int(d.group(1)) * 3600 + int(d.group(2)) * 60 + float(d.group(3))
    size = re.search(r"Video:.*?(\d{3,5})x(\d{3,5})", r.stderr)
    return duration, (int(size.group(1)), int(size.group(2))) if size else None


def fmt_secs(seconds):
    seconds = int(seconds)
    return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


# ---------------------------------------------------------------------------
# Narration audio
# ---------------------------------------------------------------------------
def short_name(draft):
    return draft.stem.replace("_REVIEW_REQUIRED", "")


def find_external_narration(draft):
    for stem in (short_name(draft), draft.stem):
        for ext in (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"):
            p = NARRATION_DIR / f"{stem}{ext}"
            if p.exists():
                return p
    return None


async def synth(text, mp3_path):
    import edge_tts
    try:
        comm = edge_tts.Communicate(text, VOICE, rate=RATE, pitch=PITCH, boundary="WordBoundary")
    except TypeError:
        comm = edge_tts.Communicate(text, VOICE, rate=RATE, pitch=PITCH)
    words = []
    with open(mp3_path, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                start = chunk["offset"] / 1e7
                words.append({"text": chunk["text"], "start": start, "end": start + chunk["duration"] / 1e7})
    return words


def make_edge_narration(ffmpeg, lines, work):
    """One voice clip per line, joined with short pauses. Also returns edge-tts's own word timings
    (used only if faster-whisper is not installed)."""
    gap = work / "gap.wav"
    run([ffmpeg, "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", f"{LINE_GAP}", str(gap)])
    parts, fallback, clock = [], [], 0.0
    for idx, (_, sentence) in enumerate(lines):
        mp3, wav = work / f"voice_{idx + 1:02d}.mp3", work / f"voice_{idx + 1:02d}.wav"
        print(f"  Voice {idx + 1}/{len(lines)}: {sentence}", flush=True)
        words = asyncio.run(synth(sentence, mp3))
        run([ffmpeg, "-y", "-i", str(mp3), "-ar", "44100", "-ac", "2", str(wav)])
        dur, _ = probe(ffmpeg, wav)
        for w in words:
            fallback.append({"text": w["text"], "start": clock + w["start"], "end": clock + w["end"], "scene": idx})
        parts += [wav.name, gap.name]
        clock += dur + LINE_GAP
    (work / "voice_list.txt").write_text("".join(f"file '{n}'\n" for n in parts), encoding="utf-8")
    run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", "voice_list.txt", "-c", "copy", "narration.wav"], cwd=work)
    return work / "narration.wav", fallback


# ---------------------------------------------------------------------------
# Timing: match the script's words to what was actually said
# ---------------------------------------------------------------------------
def norm(word):
    return re.sub(r"[^a-z0-9]", "", word.casefold())


def script_tokens(lines):
    tokens = []
    for scene, (_, sentence) in enumerate(lines):
        for raw in sentence.split():
            if norm(raw):
                tokens.append({"text": raw, "scene": scene})
    return tokens


def transcribe_words(audio_path, prompt):
    from faster_whisper import WhisperModel
    last = None
    for device, ctype in (("cuda", "float16"), ("cpu", "int8")):
        try:
            print(f"  Listening to the narration ({WHISPER_MODEL} on {device}; the first run downloads the model)...",
                  flush=True)
            model = WhisperModel(WHISPER_MODEL, device=device, compute_type=ctype)
            segments, _ = model.transcribe(str(audio_path), word_timestamps=True, language=WHISPER_LANGUAGE,
                                           beam_size=5, initial_prompt=prompt[:600])
            words = [{"text": w.word.strip(), "start": float(w.start), "end": float(w.end)}
                     for seg in segments for w in (seg.words or []) if w.word.strip()]
            if words:
                return words
        except Exception as exc:
            last = exc
            print(f"  ({device} failed: {str(exc)[:120]})", flush=True)
    raise RuntimeError(f"Speech recognition failed: {last}")


def align_tokens(tokens, heard, audio_end):
    """Give every script word a start/end time. Spelling comes from the script; timing from the audio."""
    a = [norm(t["text"]) for t in tokens]
    b = [norm(h["text"]) for h in heard]
    start, end = [None] * len(a), [None] * len(a)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    matched = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            matched += i2 - i1
            for k in range(i2 - i1):
                start[i1 + k], end[i1 + k] = heard[j1 + k]["start"], heard[j1 + k]["end"]
        elif tag == "replace":              # misheard or split words: spread the span over the script words
            t0, t1 = heard[j1]["start"], heard[j2 - 1]["end"]
            weights = [max(1, len(a[i])) for i in range(i1, i2)]
            total, acc = sum(weights), 0
            for n, i in enumerate(range(i1, i2)):
                start[i] = t0 + (t1 - t0) * acc / total
                acc += weights[n]
                end[i] = t0 + (t1 - t0) * acc / total
    i = 0
    while i < len(a):                       # words never heard: interpolate between neighbours
        if start[i] is not None:
            i += 1
            continue
        j = i
        while j < len(a) and start[j] is None:
            j += 1
        left = end[i - 1] if i > 0 else 0.0
        right = max(start[j] if j < len(a) else audio_end, left + 0.06 * (j - i))
        weights = [max(1, len(a[k])) for k in range(i, j)]
        total, acc = sum(weights), 0
        for n, k in enumerate(range(i, j)):
            start[k] = left + (right - left) * acc / total
            acc += weights[n]
            end[k] = left + (right - left) * acc / total
        i = j
    prev = 0.0
    for k in range(len(a)):
        start[k] = max(start[k], prev)
        end[k] = max(end[k], start[k] + 0.08)
        prev = start[k]
    words = [dict(tokens[k], start=start[k], end=end[k]) for k in range(len(a))]
    return words, (matched / len(a) if a else 0.0)


def scene_boundaries(words, n_scenes, audio_dur):
    starts = [0.0] * n_scenes
    for s in range(1, n_scenes):
        first = next((w for w in words if w["scene"] == s), None)
        starts[s] = max(starts[s - 1] + 0.5, (first["start"] if first else audio_dur) - 0.08)
    ends = starts[1:] + [audio_dur + TAIL]
    return starts, ends


# ---------------------------------------------------------------------------
# Captions: appear as each word is spoken, always inside the screen
# ---------------------------------------------------------------------------
CHAR_EM = {"W": .96, "M": .88, "I": .30, "J": .60, "L": .62, "F": .62, "T": .66, "E": .68, "Y": .70, "Z": .66,
           "1": .56, " ": .30, ".": .30, ",": .30, "'": .24, "!": .34, ":": .34, ";": .34, "-": .36, "?": .64}


def est_width(text):
    return sum(CHAR_EM.get(c, 0.76) for c in text.upper()) * FONT_SIZE * 1.06


def caption_groups(words):
    groups, cur = [], []
    for w in words:
        text = w["text"].upper()
        if cur and (len(cur) >= CAPTION_MAX_WORDS
                    or est_width(" ".join(x["text"] for x in cur) + " " + text) > CAPTION_MAX_WIDTH):
            groups.append(cur)
            cur = []
        cur.append(w)
        if re.search(r"[.!?]$", w["text"]) or (len(cur) >= 2 and re.search(r"[,;:]$", w["text"])):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def ass_time(t):
    cs = int(round(max(0.0, t) * 100))
    return f"{cs // 360000}:{(cs // 6000) % 60:02d}:{(cs // 100) % 60:02d}.{cs % 100:02d}"


def build_ass_words(words, font=FONT):
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{FONT_SIZE},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,6,2,2,80,80,{CAPTION_MARGIN_V},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for group in caption_groups(words):
        shown = [re.sub(r"[{}\\]", "", w["text"]).upper() for w in group]
        for i, w in enumerate(group):
            start = max(0.0, w["start"] - 0.03)
            end = group[i + 1]["start"] if i + 1 < len(group) else w["end"] + 0.18
            # Words not yet spoken are invisible but still take up space, so the line never shifts.
            parts = []
            for j, text in enumerate(shown):
                if j < i:
                    parts.append(f"{{\\alpha&H00&\\c&H00FFFFFF&}}{text}")
                elif j == i:
                    parts.append(f"{{\\alpha&H00&\\c{HIGHLIGHT}}}{text}")
                else:
                    parts.append(f"{{\\alpha&HFF&}}{text}")
            events.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,{' '.join(parts)}")
    return head + "\n".join(events) + "\n"


MIXED_EM = {" ": .26, ".": .30, ",": .30, "'": .24, "\u2019": .24, "!": .34, ":": .30, ";": .30, "-": .36, "?": .56,
            "i": .32, "l": .32, "j": .32, "t": .40, "f": .38, "r": .44, "I": .40, "m": .90, "w": .82, "W": 1.0, "M": .98}


def est_width_mixed(text, size=None):
    size = size or PHRASE_FONT_SIZE
    total = 0.0
    for c in text:
        if c in MIXED_EM:
            total += MIXED_EM[c]
        elif c.isupper():
            total += 0.76
        elif c.isdigit():
            total += 0.66
        else:
            total += 0.58
    return total * size * 1.06


def phrase_groups(words):
    """Chunks of up to PHRASE_MAX_WORDS that fit on two lines, breaking at sentence ends, commas and pauses."""
    groups, cur = [], []
    for w in words:
        if cur:
            candidate = " ".join(x["text"] for x in cur) + " " + w["text"]
            if (len(cur) >= PHRASE_MAX_WORDS or est_width_mixed(candidate) > 2 * PHRASE_LINE_WIDTH
                    or w["start"] - cur[-1]["end"] > 0.5):
                groups.append(cur)
                cur = []
        cur.append(w)
        if re.search(r"[.!?]$", w["text"]) or (len(cur) >= 4 and re.search(r"[,;:]$", w["text"])):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def split_two_lines(texts):
    full = " ".join(texts)
    if len(texts) == 1 or est_width_mixed(full) <= PHRASE_LINE_WIDTH:
        return [full]
    best = None
    for k in range(1, len(texts)):
        a, b = " ".join(texts[:k]), " ".join(texts[k:])
        width = max(est_width_mixed(a), est_width_mixed(b))
        if texts[k - 1][:1].isupper() and texts[k][:1].isupper() and not re.search(r"[.!?,;:]$", texts[k - 1]):
            width += 140                    # avoid splitting a name such as "Crookback Bog" across lines
        if best is None or width < best[0]:
            best = (width, a, b)
    return [best[1], best[2]]


def build_ass_phrases(words, font=PHRASE_FONT):
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{PHRASE_FONT_SIZE},&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,1,0,0,0,100,100,0.4,0,1,3,2,2,70,70,{PHRASE_MARGIN_V},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    groups = phrase_groups(words)
    events = []
    for gi, group in enumerate(groups):
        texts = [re.sub(r"[{}\\]", "", w["text"]) for w in group]
        start = max(0.0, group[0]["start"] - 0.05)
        end = group[-1]["end"] + 0.25
        if gi + 1 < len(groups):
            end = min(end, max(groups[gi + 1][0]["start"] - 0.03, start + 0.4))
        end = max(end, start + 0.5)
        text = "\\N".join(split_two_lines(texts))
        events.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,{text}")
    return head + "\n".join(events) + "\n"


def build_ass(words, font=None):
    if CAPTION_STYLE == "word":
        return build_ass_words(words, font or FONT)
    return build_ass_phrases(words, font or PHRASE_FONT)


# ---------------------------------------------------------------------------
# Framing: keep the main character / speaker in the middle of the vertical frame
# ---------------------------------------------------------------------------
_DETECTORS = {}


def face_detectors():
    if "ready" in _DETECTORS:
        return _DETECTORS.get("cv2")
    _DETECTORS["ready"] = True
    try:
        import cv2
    except ImportError:
        print("  (opencv-python not installed: shots will be centre-cropped. Run: python -m pip install opencv-python)",
              flush=True)
        return None
    base = cv2.data.haarcascades
    _DETECTORS["cv2"] = cv2
    _DETECTORS["frontal"] = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
    _DETECTORS["profile"] = cv2.CascadeClassifier(base + "haarcascade_profileface.xml")
    return cv2


def subject_track(clip, t0, t1, samples=4):
    """Largest face's horizontal position (0-1) at a few moments of the shot: [(seconds into shot, x), ...]."""
    cv2 = face_detectors()
    if cv2 is None:
        return []
    cap = cv2.VideoCapture(str(clip))
    track = []
    try:
        for k in range(samples):
            frac = k / (samples - 1)
            cap.set(cv2.CAP_PROP_POS_MSEC, (t0 + frac * max(0.1, t1 - t0 - 0.1)) * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (960, int(960 * h / w)))
            gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
            found = [tuple(f) for f in _DETECTORS["frontal"].detectMultiScale(gray, 1.1, 5, minSize=(48, 48))]
            found += [tuple(f) for f in _DETECTORS["profile"].detectMultiScale(gray, 1.1, 5, minSize=(48, 48))]
            found += [(960 - x - fw, y, fw, fh) for (x, y, fw, fh)
                      in _DETECTORS["profile"].detectMultiScale(cv2.flip(gray, 1), 1.1, 5, minSize=(48, 48))]
            if found:
                x, y, fw, fh = max(found, key=lambda f: f[2] * f[3])
                track.append((frac * (t1 - t0), (x + fw / 2) / 960))
    finally:
        cap.release()
    return track


def crop_geometry(src_size):
    src_w, src_h = src_size
    crop_h = int(src_h * FRAME_USE) // 2 * 2
    crop_w = int(crop_h * W / H) // 2 * 2
    return src_w, crop_w, crop_h


def crop_x_expr(track, src_size, dur):
    """FFmpeg x expression that glides the crop window along the subject. Centre if nothing was found."""
    src_w, crop_w, _ = crop_geometry(src_size)
    lo, hi = 0.0, float(src_w - crop_w)
    pts = [(t, min(max(x * src_w - crop_w / 2, lo), hi)) for t, x in track]
    if not pts:
        return f"{(src_w - crop_w) / 2:.1f}"
    if len(pts) == 1:
        return f"{pts[0][1]:.1f}"
    expr = f"{pts[-1][1]:.1f}"
    for (ta, xa), (tb, xb) in reversed(list(zip(pts, pts[1:]))):
        span = max(0.05, tb - ta)
        expr = f"if(lt(t,{tb:.3f}),{xa:.1f}+({xb:.1f}-{xa:.1f})*(t-{ta:.3f})/{span:.3f},{expr})"
    return f"if(lt(t,{pts[0][0]:.3f}),{pts[0][1]:.1f},{expr})"


def make_shot(ffmpeg, clip, clip_len, src_size, offset, dur, out):
    """Returns the subject track (possibly empty) so the caller can report how framing went."""
    if clip is None:
        run([ffmpeg, "-y", "-f", "lavfi", "-i", f"color=c=0x101820:s={W}x{H}:r={FPS}:d={dur:.3f}",
             "-vf", "format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-an", str(out)])
        return []
    loop = ["-stream_loop", "-1"] if clip_len < offset + dur else []
    if loop:
        offset = 0.0
    track = subject_track(clip, offset, offset + dur) if src_size else []
    if src_size:
        _, cw, ch = crop_geometry(src_size)
        crop = f"crop={cw}:{ch}:'{crop_x_expr(track, src_size, dur)}':0,"
    else:
        crop = ""
    vf = f"{crop}scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H},setsar=1,fps={FPS},format=yuv420p"
    run([ffmpeg, "-y", *loop, "-ss", f"{offset:.3f}", "-i", str(clip), "-t", f"{dur:.3f}", "-vf", vf, "-an",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-r", str(FPS), str(out)])
    return track


def make_ai_shot(ffmpeg, image, dur, out, zoom_in):
    """Slow push-in or pull-out on a still, with a short dip in and out so it blends with the footage."""
    frames = max(2, int(round(dur * FPS)))
    z = f"1+0.14*on/{frames}" if zoom_in else f"1.14-0.14*on/{frames}"
    fade = min(0.2, dur / 4)
    vf = (f"scale={int(W * 1.5)}:{int(H * 1.5)}:force_original_aspect_ratio=increase:flags=lanczos,"
          f"crop={int(W * 1.5)}:{int(H * 1.5)},"
          f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={W}x{H}:fps={FPS},"
          f"fade=t=in:st=0:d={fade:.2f},fade=t=out:st={dur - fade:.2f}:d={fade:.2f},format=yuv420p")
    run([ffmpeg, "-y", "-i", str(image), "-vf", vf, "-frames:v", str(frames), "-an",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-r", str(FPS), str(out)])


def plan_segments(scene_len, has_clip, has_ai):
    """Gameplay -> AI still -> gameplay when there is room; AI still alone if there is no clip."""
    if has_clip and has_ai and scene_len >= AI_MIN_SCENE:
        first = (scene_len - AI_INSERT) / 2
        return [("clip", first), ("ai", AI_INSERT), ("clip", scene_len - AI_INSERT - first)]
    if has_ai and not has_clip:
        return [("ai", scene_len)]
    return [("clip" if has_clip else "blank", scene_len)]


def latest_manifest_clips():
    manifests = sorted(MANIFEST_DIR.glob("*_manifest.json"), key=lambda p: p.stat().st_mtime)
    if not manifests:
        print("No clip manifest found; scenes will use AI stills or a plain background.")
        return {}
    data = json.loads(manifests[-1].read_text(encoding="utf-8"))
    print(f"Clip manifest: {manifests[-1].name}")
    clips = {}
    for a in data.get("assets", []):
        f = a.get("file")
        if a.get("use_clip") is False:          # you picked "use neither" in review_clips.py
            continue
        if a.get("status") == "downloaded_unverified" and f and Path(f).exists():
            clips[a["scene"]] = Path(f)
    return clips


def main():
    ffmpeg = find_tool("ffmpeg", r"C:\pinokio\bin\ffmpeg.exe")
    draft = find_draft()
    lines = extract_spoken_lines(draft.read_text(encoding="utf-8-sig"))
    if not lines:
        raise SystemExit(f"No tagged narration lines in {draft.name}")
    clips = latest_manifest_clips()

    NARRATION_DIR.mkdir(parents=True, exist_ok=True)
    (NARRATION_DIR / f"{short_name(draft)}_script.txt").write_text("\n".join(s for _, s in lines) + "\n", encoding="utf-8")
    external = find_external_narration(draft)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work = OUT_DIR / f"build_{stamp}"
    work.mkdir(parents=True, exist_ok=True)
    print(f"Draft: {draft.name}\nScenes: {len(lines)}\nWorking in: {work}")

    # 1. Narration audio
    fallback_words = None
    if external:
        print(f"Narration: using your file {external.name}")
        narration = external
    else:
        print(f"Narration: no file at {NARRATION_DIR / (short_name(draft) + '.wav')}, so generating with edge-tts ({VOICE}).")
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            raise SystemExit("Missing dependency. Run:  python -m pip install edge-tts")
        narration, fallback_words = make_edge_narration(ffmpeg, lines, work)
    audio_dur, _ = probe(ffmpeg, narration)

    # 2. Timing
    tokens = script_tokens(lines)
    try:
        heard = transcribe_words(narration, " ".join(s for _, s in lines))
        words, ratio = align_tokens(tokens, heard, audio_dur)
        print(f"  Matched {ratio:.0%} of the script's words to the audio.")
        if ratio < 0.6:
            print("  WARNING: low match. Is this audio really this script? Captions may be off.")
    except ImportError:
        if fallback_words is None:
            raise SystemExit("Your own narration needs faster-whisper to time the captions.\n"
                             "Run:  python -m pip install faster-whisper")
        print("  faster-whisper is not installed, so edge-tts's own word timings are used (less exact).\n"
              "  For better caption sync run:  python -m pip install faster-whisper")
        words = fallback_words
    starts, ends = scene_boundaries(words, len(lines), audio_dur)
    total = ends[-1]

    # 3. Picture
    shot_files, ai_total = [], 0
    for idx, (_, sentence) in enumerate(lines, 1):
        scene_len = ends[idx - 1] - starts[idx - 1]
        clip = clips.get(idx)
        clip_len, src_size = probe(ffmpeg, clip) if clip else (0, None)
        ai_img = AI_DIR / f"{draft.stem}_scene_{idx:02d}.png"
        ai_img = ai_img if (USE_AI_INSERTS and ai_img.exists()) else None
        print(f"[{idx}/{len(lines)}] {scene_len:.1f}s  clip={'yes' if clip else 'no'}  ai={'yes' if ai_img else 'no'}",
              flush=True)
        shot_no, cursor, ai_count, xs = 0, 0.0, 0, []
        for kind, seg_len in plan_segments(scene_len, bool(clip), bool(ai_img)):
            n = max(1, int(-(-seg_len // SHOT_MAX)))
            each = seg_len / n
            for _ in range(n):
                shot = work / f"shot_{idx:02d}_{shot_no:02d}.mp4"
                if kind == "ai":
                    make_ai_shot(ffmpeg, ai_img, each, shot, zoom_in=(ai_count % 2 == 0))
                    ai_count += 1
                elif kind == "clip":
                    offset = min(cursor, max(0.0, clip_len - each))
                    track = make_shot(ffmpeg, clip, clip_len, src_size, offset, each, shot)
                    xs.append(track)
                    cursor += each
                else:
                    make_shot(ffmpeg, None, 0, None, 0, each, shot)
                shot_files.append(shot.name)
                shot_no += 1
        ai_total += ai_count
        if xs:
            tracked = sum(1 for t in xs if t)
            print(f"    framing: subject found in {tracked}/{len(xs)} shots"
                  + ("" if tracked else " (centre crop)"), flush=True)

    # 4. Assemble
    (work / "words.json").write_text(json.dumps(words, indent=2, ensure_ascii=False), encoding="utf-8")
    (work / "captions.ass").write_text(build_ass(words), encoding="utf-8")
    (work / "shots.txt").write_text("".join(f"file '{n}'\n" for n in shot_files), encoding="utf-8")
    print("Joining shots...", flush=True)
    run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", "shots.txt", "-c", "copy", "video_silent.mp4"], cwd=work)

    audio_in = narration if narration.parent == work else narration
    final = OUT_DIR / f"{draft.stem.replace('_REVIEW_REQUIRED', '')}_{stamp}.mp4"
    print("Burning captions and mixing audio...", flush=True)
    run([ffmpeg, "-y", "-i", "video_silent.mp4", "-i", str(audio_in),
         "-vf", "subtitles=captions.ass", "-af", f"loudnorm=I=-16:TP=-1.5:LRA=11,apad=whole_dur={total:.3f}",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t", f"{total:.3f}", str(final)], cwd=work)

    print(f"\nDone: {final}")
    print("Reminder: clips are search-result candidates. Watch the video and check the visuals and licensing "
          "before publishing.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
