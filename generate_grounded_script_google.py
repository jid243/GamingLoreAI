# GamingLoreAI short-script generator (Ollama backend; filename retained for compatibility)
from pathlib import Path
from datetime import datetime
import json
import math
import random
import re
import sys
import difflib
import requests
import os
import time

PROJECT = Path.home() / "Desktop" / "GamingLoreAI"
SOURCES_DIR = PROJECT / "sources"
EVIDENCE_FILE = SOURCES_DIR / "evidence_passages.txt"
SCRIPT_DIR = PROJECT / "scripts"
CHECK_DIR = PROJECT / "fact_checks"
STYLE_DIR = PROJECT / "style_examples"   # finished scripts that performed well; the writer copies their craft, never their content


def _env_number(name, default, cast=int):
    try:
        return cast(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# ---- Model / backend (Ollama, local, free) -------------------------------------------------
MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:26b-a4b-it-qat")   # mixture-of-experts: fast even when part sits in system RAM
HOOK_EVAL_MODEL = os.environ.get("HOOK_EVAL_MODEL", "") or MODEL   # optionally a DIFFERENT model judges the hooks
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
NUM_CTX = _env_number("OLLAMA_NUM_CTX", 8192 if MODEL.startswith("gemma2") else 16384)   # gemma2 was trained at 8k
OLLAMA_TIMEOUT = _env_number("OLLAMA_TIMEOUT", 900)    # 27B is partly CPU-offloaded on a 12 GB GPU, so allow time
# Only these model families accept Ollama's "think" switch; sending it to others (gemma2) can be rejected.
THINK_CAPABLE_PREFIXES = ("gemma4", "qwen3", "deepseek-r1", "gpt-oss", "qwq", "magistral", "cogito", "phi4-reasoning")

# ---- Script stage configuration (environment overridable) ----------------------------------
MIN_WORDS = _env_number("SCRIPT_MIN_WORDS", 55)             # hard floor: shorter, tighter stories are allowed
MAX_WORDS = _env_number("SCRIPT_MAX_WORDS", 160)            # hard ceiling: about 60 seconds of narration
SCRIPT_TARGET_MIN_WORDS = _env_number("SCRIPT_TARGET_MIN_WORDS", 80)   # what the writer aims for (about 30-50 s)
SCRIPT_TARGET_MAX_WORDS = _env_number("SCRIPT_TARGET_MAX_WORDS", min(150, MAX_WORDS))   # 30-60 seconds
FULL_STORY_FIRST = os.environ.get("FULL_STORY_FIRST", "1") != "0"   # tell the whole story first, then cut it to length
PAYOFF_POINTS_MAX = 6                                              # how each choice plays out: told, or the draft goes back
SCRIPT_CANDIDATES = max(1, _env_number("SCRIPT_CANDIDATES", 2))        # full scripts written, one per top hook; 1 = old flow
SCRIPT_EXTRA_ATTEMPTS = _env_number("SCRIPT_EXTRA_ATTEMPTS", 2)        # attempt cap for candidates after the first
SCRIPT_EVAL_MODEL = os.environ.get("SCRIPT_EVAL_MODEL", "") or HOOK_EVAL_MODEL
SCRIPT_EVAL_TEMPERATURE = _env_number("SCRIPT_EVAL_TEMPERATURE", 0.1, float)
SCRIPT_MIN_TOTAL = _env_number("SCRIPT_MIN_TOTAL", 60)       # below this the report warns; grounding gates decide eligibility
SCRIPT_WEIGHTS = {                                           # must sum to 100
    "clarity": _env_number("SCRIPT_W_CLARITY", 20),
    "opening": _env_number("SCRIPT_W_OPENING", 25),
    "progression": _env_number("SCRIPT_W_PROGRESSION", 25),
    "delivery": _env_number("SCRIPT_W_DELIVERY", 15),
    "grounding": _env_number("SCRIPT_W_GROUNDING", 15),
}
REPETITION_PENALTY = 5          # per pair of lines that restate each other, capped
REPETITION_CAP = 15
OVERUSED_WORD_PENALTY = 3       # per use beyond the first of the listed adjectives, capped
OVERUSED_WORD_CAP = 9
GENERIC_ENDING_PENALTY = 6
ENDING_MAX_WORDS = _env_number("ENDING_MAX_WORDS", 20)
RETENTION_FAIL_PENALTY = 10
COHERENCE_FAIL_PENALTY = 10     # retention review is a style judgement: a penalty, not a hard gate
REQUIRE_ROOT_CAUSE = os.environ.get("REQUIRE_ROOT_CAUSE", "1") != "0"   # script must state the cause fact the evidence gives
LENGTH_OVER_PENALTY = 3         # per 10 spoken words above the target maximum, capped
LENGTH_OVER_CAP = 9
MAX_EVIDENCE_PASSAGES = _env_number("MAX_EVIDENCE_PASSAGES", 22)   # passages are sent compactly (no URL/type lines)
STORY_EXTRA_PASSAGES = _env_number("STORY_EXTRA_PASSAGES", 6)      # extra passages the model may add for missing story links
CAUSE_SEARCH = os.environ.get("CAUSE_SEARCH", "1") != "0"         # search the WHOLE evidence file for what caused the subject
CAUSE_EXTRA_PASSAGES = _env_number("CAUSE_EXTRA_PASSAGES", 5)
MIN_RELEVANCE_SCORE = 2
MAX_ATTEMPTS = max(1, _env_number("MAX_ATTEMPTS", 5))   # script attempts for the first candidate
MIN_RETENTION_SCORE = "GOOD"

# ---- Hook stage configuration (all overridable with environment variables) -----------------
HOOK_CANDIDATES = _env_number("HOOK_CANDIDATES", 6)               # candidates asked for per generation attempt
HOOK_GEN_ATTEMPTS = _env_number("HOOK_GEN_ATTEMPTS", 3)           # generation retries if nothing eligible
HOOK_MIN_ELIGIBLE = 1                                             # eligible hooks needed to proceed
HOOK_TARGET_MIN_WORDS = _env_number("HOOK_TARGET_MIN_WORDS", 8)   # spoken-length target (game title counts as 1 word)
HOOK_TARGET_MAX_WORDS = _env_number("HOOK_TARGET_MAX_WORDS", 18)
HOOK_HARD_MIN_WORDS = 5                                           # rejected below this
HOOK_HARD_MAX_WORDS = _env_number("HOOK_HARD_MAX_WORDS", 22)      # rejected above this (unless it can be tightened)
HOOK_TIGHTEN_MAX_WORDS = _env_number("HOOK_TIGHTEN_MAX_WORDS", 34) # longer hooks than this are not worth tightening
HOOK_GEN_TEMPERATURE = _env_number("HOOK_GEN_TEMPERATURE", 0.7, float)
HOOK_GEN_TOP_P = _env_number("HOOK_GEN_TOP_P", 0.9, float)
HOOK_EVAL_TEMPERATURE = _env_number("HOOK_EVAL_TEMPERATURE", 0.1, float)
HOOK_MIN_TOTAL = _env_number("HOOK_MIN_TOTAL", 60)                # of 100; the evaluator is strict (typical good hook = 3s and 4s)
HOOK_STRICT = os.environ.get("HOOK_STRICT") == "1"                # 1 = style floors reject hooks again (default: only made-up claims do)
HOOK_MIN_CLARITY = 3                                              # of 5; floor per viewer criterion
HOOK_MIN_SPECIFICITY = 3
HOOK_SEED = os.environ.get("HOOK_SEED")                           # set for reproducible evaluation order
HOOK_WEIGHTS = {                                                  # must sum to 100
    "clarity": _env_number("HOOK_W_CLARITY", 20),
    "specificity": _env_number("HOOK_W_SPECIFICITY", 15),
    "curiosity": _env_number("HOOK_W_CURIOSITY", 20),
    "emotion": _env_number("HOOK_W_EMOTION", 20),
    "delivery": _env_number("HOOK_W_DELIVERY", 10),
    "grounding": _env_number("HOOK_W_GROUNDING", 15),
}
# Penalty points (subtracted after the weighted score)
PENALTY_NEEDS_REVIEW = 10
PENALTY_UNEXPLAINED_NAME = 6      # per unexplained proper noun, capped
PENALTY_NAME_CAP = 18
PENALTY_VAGUE_PHRASE = 8          # per vague/cliche phrase, capped
PENALTY_VAGUE_CAP = 16
PENALTY_GENERIC_ANY_GAME = 12
PENALTY_REDUNDANT_SETUP = 5
PENALTY_CONFUSING_SYNTAX = 5
PENALTY_OUT_OF_RANGE_LENGTH = 4
HOOK_LENGTH_GRACE = 2              # a word or two over the target is under a second of speech: no penalty
HOOK_CLARIFY_MAX = _env_number("HOOK_CLARIFY_MAX", 3)
HOOK_PLAIN_REWRITES = _env_number("HOOK_PLAIN_REWRITES", 3)   # strongest hooks also rewritten in plain, gripping spoken words   # promising hooks rewritten per round to explain their unexplained names/terms
PENALTY_NOT_TIED_TO_EVIDENCE = 8

HOOK_APPROACHES = [
    ("MORAL_DILEMMA", "someone must choose and every choice costs something"),
    ("ROOT_CAUSE", "who or what brought the subject about, as plainly and as strongly as the evidence states it"),
    ("UNEXPECTED_CONSEQUENCE", "an action whose result is not what anyone would expect"),
    ("TRANSFORMATION", "how the subject can change or end differently, and what that alternative leads to"),
    ("EMOTIONAL_CONFLICT", "two people or loyalties pulling against each other"),
    ("MYSTERY", "a strange, specific detail whose explanation is withheld"),
]

SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
CHECK_DIR.mkdir(parents=True, exist_ok=True)


def _supports_think_flag(model):
    return model.lower().startswith(THINK_CAPABLE_PREFIXES)


MODEL_CALLS = {"local": 0, "online": 0}   # reported at the end of every run (the benchmark reads it)
CALLS_BY_STEP = {}                        # which step made the local calls, so the expensive ones can be found


def _count_step():
    frame = sys._getframe(2)
    while frame and frame.f_code.co_name in ("ask_writer", "ask_ollama", "_count_step", "<lambda>"):
        frame = frame.f_back
    name = frame.f_code.co_name if frame else "?"
    CALLS_BY_STEP[name] = CALLS_BY_STEP.get(name, 0) + 1


def _report_calls():
    if MODEL_CALLS["local"] or MODEL_CALLS["online"]:
        steps = ", ".join(f"{k} {v}" for k, v in sorted(CALLS_BY_STEP.items(), key=lambda kv: -kv[1]))
        print(f"MODEL CALLS: local={MODEL_CALLS['local']} online={MODEL_CALLS['online']}" + (f" | by step: {steps}" if steps else ""), flush=True)


import atexit as _atexit
_atexit.register(_report_calls)


def ask_ollama(prompt, temperature=0.2, model=None, options=None):
    model = model or MODEL
    MODEL_CALLS["local"] += 1
    _count_step()
    opts = {"temperature": temperature, "num_ctx": NUM_CTX}
    opts.update(options or {})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],   # user role only: gemma has no system role
        "stream": False,
        "keep_alive": "10m",
        "options": opts,
    }
    if _supports_think_flag(model):
        payload["think"] = False
    if len(prompt) / 3.5 > NUM_CTX - 700:
        print(f"WARNING: prompt is ~{int(len(prompt) / 3.5)} tokens against a {NUM_CTX}-token window; "
              "Ollama may truncate it. Lower MAX_EVIDENCE_PASSAGES if outputs look cut off.")
    response = _post_with_retries(payload)
    if response.status_code == 400 and "think" in payload and "think" in response.text.lower():
        payload.pop("think")
        response = _post_with_retries(payload)
    if response.status_code == 404:
        raise RuntimeError(f"Ollama does not have model '{model}'. Install it with: ollama pull {model}")
    if response.status_code >= 400:
        raise RuntimeError(f"Ollama error {response.status_code}: {_ollama_error_text(response)}")
    return response.json()["message"]["content"].strip()


OLLAMA_RETRY_WAITS = (5, 15, 30)   # seconds; a model that is still loading or being swapped answers 500/503 briefly

# THE WRITER. Writing a story that holds a stranger needs a much stronger model than finding and checking facts does.
# The writing steps (hooks, plan, story, edit, cuts, closing question) go to free hosted models; research and every
# fact check stay on the local model. Each step tries the writers in order ONCE, moving on at once when one is busy:
#   Gemini (best free writer, but busy at peak times) -> Groq gpt-oss-120b (fast, reliable) -> other Gemini models
#   -> the local model. A writer whose key is wrong or whose daily free limit ran out is skipped for the rest of the run.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_BACKUP_MODELS = [m.strip() for m in os.environ.get("GEMINI_BACKUP_MODELS", "gemini-3.6-flash,gemini-3.5-flash-lite").split(",") if m.strip()]
GEMINI_THINKING = os.environ.get("GEMINI_THINKING", "medium")   # low / medium / high
GEMINI_KEY_FILE = PROJECT / "gemini_api_key.txt"                # written by the app; never printed
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_KEY_FILE = PROJECT / "groq_api_key.txt"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MAX_PROMPT_TOKENS = _env_number("GROQ_MAX_PROMPT_TOKENS", 5500)   # free tier: ~8k tokens a minute, prompt plus answer
WRITER_TIMEOUT = (15, 120)    # connect, answer: a request that hangs is handed to the next writer
WRITER_PACING = {"gemini": float(os.environ.get("GEMINI_MIN_INTERVAL", "6.5")), "groq": 2.2}   # free-tier requests per minute
WRITER_SHORT_WAIT = 20        # a "try again in Ns" this short is waited out; anything longer moves on
WRITER_ROUNDS = (0, 15)       # every writer once; if ALL were busy, once more after 15 s; then the local model
_WRITERS = {"last": {}, "down": {}, "used": 0, "fallbacks": 0}


class WriterBusy(Exception):
    """This writer cannot take this step right now (busy, slow, or an empty reply): try the next one."""


class WriterDown(Exception):
    """This writer is unusable for the rest of the run (wrong key, daily free limit used up, unknown model)."""


def _read_key(env_name, path):
    key = os.environ.get(env_name, "").strip()
    if not key:
        try:
            key = path.read_text(encoding="utf-8-sig").strip()
        except OSError:
            key = ""
    return key


def gemini_key():
    return _read_key("GEMINI_API_KEY", GEMINI_KEY_FILE)


def groq_key():
    return _read_key("GROQ_API_KEY", GROQ_KEY_FILE)


USE_ONLINE_WRITERS = os.environ.get("USE_ONLINE_WRITERS", "0") == "1"   # off: everything runs on this PC


def writer_chain():
    """The hosted writers this run can use, best first: [(provider, model, key)]. Empty unless online writers are on."""
    if not USE_ONLINE_WRITERS:
        return []
    chain, gem, groq = [], gemini_key(), groq_key()
    if gem:
        chain.append(("gemini", GEMINI_MODEL, gem))
    if groq:
        chain.append(("groq", GROQ_MODEL, groq))
    if gem:
        chain += [("gemini", m, gem) for m in GEMINI_BACKUP_MODELS if m != GEMINI_MODEL]
    return [w for w in chain if (w[0], w[1]) not in _WRITERS["down"] and (w[0], "*") not in _WRITERS["down"]]


def writer_name():
    chain = writer_chain()
    if not chain:
        return f"local model ({MODEL})"
    return " -> ".join(f"{'Gemini' if p == 'gemini' else 'Groq'} {m.split('/')[-1]}" for p, m, _ in chain) + " -> local model"


def _error_text(response):
    try:
        body = response.json()
        error = body.get("error", body)
        return str(error.get("message") if isinstance(error, dict) else error)[:200]
    except ValueError:
        return response.text[:200]


def _post_writer(provider, model, key, prompt):
    pause = WRITER_PACING.get(provider, 0) - (time.time() - _WRITERS["last"].get(provider, 0.0))
    if pause > 0:
        time.sleep(pause)
    _WRITERS["last"][provider] = time.time()
    if provider == "gemini":
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                # game lore is fiction full of monsters and death; only block what is truly harmful
                "safetySettings": [{"category": c, "threshold": "BLOCK_ONLY_HIGH"} for c in (
                    "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "HARM_CATEGORY_DANGEROUS_CONTENT")]}
        if model == GEMINI_MODEL:   # backups keep their default thinking: not every model accepts every level
            body["generationConfig"] = {"thinkingConfig": {"thinkingLevel": GEMINI_THINKING}}
        return requests.post(GEMINI_URL.format(model=model), json=body, timeout=WRITER_TIMEOUT,
                             headers={"x-goog-api-key": key, "Content-Type": "application/json"})
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "reasoning_effort": "medium",
            "max_completion_tokens": 6000}
    return requests.post(GROQ_URL, json=body, timeout=WRITER_TIMEOUT,
                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})


def _writer_text(provider, response):
    data = response.json()
    if provider == "gemini":
        parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    else:
        text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    return re.sub(r"(?s)<think>.*?</think>", "", text).strip()


def ask_hosted(provider, model, key, prompt):
    """One request to one hosted writer. Returns its text, or raises WriterBusy / WriterDown."""
    for _ in range(2):   # the second pass only after a short "try again in a few seconds"
        MODEL_CALLS["online"] += 1
        try:
            response = _post_writer(provider, model, key, prompt)
        except requests.RequestException as error:
            raise WriterBusy(f"did not answer ({type(error).__name__})")
        status, text = response.status_code, response.text.lower().replace("_", "").replace(" ", "")
        if status == 200:
            out = _writer_text(provider, response)
            if not out:
                raise WriterBusy("returned no text for this step")
            return out
        if status in (400, 401, 403) and ("key" in text or "auth" in text or "permission" in text or status != 400):
            raise WriterDown(f"refused the key ({status}: {_error_text(response)})")
        if status == 404:
            raise WriterDown(f"has no model {model} ({_error_text(response)})")
        if status == 429 and ("perday" in text or "tokensperday" in text or "requestsperday" in text or "(rpd)" in text or "(tpd)" in text):
            raise WriterDown("reached its daily free limit (it resets tomorrow)")
        if status == 429:
            retry = response.headers.get("retry-after") if hasattr(response, "headers") else None
            delay = re.search(r'"retrydelay":"(\d+)', text)
            seconds = float(retry) if retry and retry.replace(".", "", 1).isdigit() else (int(delay.group(1)) if delay else 99)
            if seconds <= WRITER_SHORT_WAIT:
                print(f"  Writer: {provider} {model} asks to wait {seconds:.0f}s (free per-minute limit); waiting.", flush=True)
                time.sleep(seconds + 0.5)
                continue
        raise WriterBusy(f"is busy ({status}: {_error_text(response)[:80]})")
    raise WriterBusy("is still at its per-minute limit")


def ask_writer(prompt, temperature=0.2, options=None):
    """The writing steps: the best hosted writer that answers, otherwise the local model (same prompt)."""
    for n, wait in enumerate(WRITER_ROUNDS):
        chain = writer_chain()
        if not chain:
            break
        if wait:
            print(f"  Writer: every writer was busy; trying them once more in {wait}s.", flush=True)
            time.sleep(wait)
        for provider, model, key in chain:
            if (provider, "*") in _WRITERS["down"] or (provider, model) in _WRITERS["down"]:
                continue   # went down earlier in this round (a wrong key stops every model of that service)
            if provider == "groq" and len(prompt) / 3.5 > GROQ_MAX_PROMPT_TOKENS:
                continue   # the free tier refuses a request this large (413); asking would only waste the round
            try:
                text = ask_hosted(provider, model, key, prompt)
            except WriterBusy as reason:
                print(f"  Writer: {provider} {model} {reason}; trying the next writer.", flush=True)
                continue
            except WriterDown as reason:
                print(f"  Writer: {provider} {model} {reason}; skipping it for the rest of this run.", flush=True)
                if "key" in str(reason):
                    _WRITERS["down"][(provider, "*")] = str(reason)
                else:
                    _WRITERS["down"][(provider, model)] = str(reason)
                continue
            _WRITERS["used"] += 1
            if (provider, model) != (chain[0][0], chain[0][1]) or n:
                print(f"  Writer: {provider} {model} wrote this step.", flush=True)
            return text
    if writer_chain() or _WRITERS["down"]:
        _WRITERS["fallbacks"] += 1
        print("  Writer: no hosted writer answered; the local model writes this step.", flush=True)
    return ask_ollama(prompt, temperature=temperature, options=options, model=LOCAL_WRITER_MODEL or None)


def _ollama_error_text(response):
    try:
        return str(response.json().get("error") or response.text)[:300]
    except ValueError:
        return response.text[:300]


def _post_with_retries(payload):
    """Ollama sometimes fails one request while it loads, unloads or swaps a model (a 500 or 503, or a dropped
    connection). That is not a fault in the prompt, so the same request is sent again after a short wait."""
    for attempt, wait in enumerate((*OLLAMA_RETRY_WAITS, None), start=1):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        except requests.ConnectionError as error:
            if wait is None:
                raise RuntimeError(f"Ollama is not reachable at {OLLAMA_URL}. Is Ollama running? ({error})")
            print(f"  Ollama connection dropped; retrying in {wait}s (try {attempt + 1}/{len(OLLAMA_RETRY_WAITS) + 1})", flush=True)
            time.sleep(wait)
            continue
        if response.status_code < 500 or wait is None:
            return response
        print(f"  Ollama returned {response.status_code} ({_ollama_error_text(response)}); retrying in {wait}s "
              f"(try {attempt + 1}/{len(OLLAMA_RETRY_WAITS) + 1})", flush=True)
        time.sleep(wait)


def latest_research_file():
    files = list(SOURCES_DIR.glob("*_research_*.txt"))
    if not files:
        raise FileNotFoundError("No research file found. Run research_sources.py first.")
    return max(files, key=lambda path: path.stat().st_mtime)


def get_metadata(text, field):
    match = re.search(rf"(?m)^{re.escape(field)}:\s*(.+)$", text)
    return match.group(1).strip() if match else None


def normalise(value):
    return re.sub(r"\s+", " ", value).strip().casefold()


def read_project_context():
    research_path = latest_research_file()
    text = research_path.read_text(encoding="utf-8-sig")
    game, topic = get_metadata(text, "GAME"), get_metadata(text, "TOPIC")
    if not game or not topic:
        raise ValueError(f"GAME or TOPIC is missing from {research_path.name}.")
    return game, topic, research_path


def check_evidence_context(evidence, game, topic):
    evidence_game, evidence_topic = get_metadata(evidence, "GAME"), get_metadata(evidence, "TOPIC")
    problems = []
    if not evidence_game or not evidence_topic:
        return ["Evidence file is missing GAME or TOPIC metadata."]
    if normalise(evidence_game) != normalise(game):
        problems.append(f"Evidence game mismatch: expected '{game}', found '{evidence_game}'.")
    if normalise(evidence_topic) != normalise(topic):
        problems.append(f"Evidence topic mismatch: expected '{topic}', found '{evidence_topic}'.")
    return problems


def split_evidence_passages(evidence):
    parts = re.split(r"(?m)(?=^\[E\d+\]\s*$)", evidence)
    passages = []
    for part in parts:
        match = re.match(r"(?m)^\[(E\d+)\]\s*$", part)
        if match:
            passages.append({"id": match.group(1), "text": part.strip()})
    return passages


BOILERPLATE_TERMS = (
    "cookie", "privacy policy", "terms of service", "all rights reserved", "subscribe to our",
    "sign up for", "newsletter", "advertisement", "log in to", "next step: assess each passage",
)

_GENERIC_WORDS = set(
    "about after again also among because been before being between both could does each from have "
    "into just like make more most much only other over same some such than that their them then "
    "there these they this those very what when where which while will with would your game games "
    "video play player players".split()
)


def _terms(text):
    """Lower-case content words (possessives stripped, short/generic words dropped)."""
    text = re.sub(r"['\u2019]s\b", "", text.casefold())
    words = re.findall(r"[a-z0-9]+", text)
    return {w for w in words if len(w) > 3 and w not in _GENERIC_WORDS}


def topic_terms(topic, game=""):
    """Words that mark a passage as being about the TOPIC. The game title's own words are left out when the topic has
    words of its own: nearly every page of a game's wiki names the game (an infobox's 'Games:' row, for example), so
    counting them made unrelated pages look on-topic and pushed the story's key passages out of the selection."""
    own = _terms(topic) - _terms(game)
    return own or (_terms(topic) | _terms(game))


STORY_WEIGHTS = {
    # narrative / lore
    "lore": 2, "secret": 3, "hidden": 3, "myth": 1, "legend": 2, "origin": 2, "history": 1,
    "betrayal": 3, "tragedy": 3, "twist": 3, "reveal": 2, "ending": 2, "climax": 2,
    "death": 2, "sacrifice": 3, "choice": 2, "consequence": 3, "price": 2, "cost": 2,
    # mechanics / design / community-flavoured material
    "unused": 3, "cut content": 3, "easter egg": 3, "glitch": 2, "exploit": 2, "bug": 2,
    "design": 1, "mechanic": 2, "developer": 1, "behind the scenes": 2, "inspired by": 2,
    "speedrun": 2, "record": 1, "rare": 2, "unique": 1,
}


def relevance_score(passage, topic, game=""):
    """Topic overlap is mandatory; story-value words only add to passages that are on-topic."""
    text = passage["text"].casefold()
    if any(term in text for term in BOILERPLATE_TERMS):
        return -100

    on_topic = sum(1 for word in topic_terms(topic, game) if word in text)
    score = on_topic * 5
    story = sum(weight for term, weight in STORY_WEIGHTS.items() if term in text)
    # Story-flavoured words count fully only when the passage is on-topic.
    return score + (story if on_topic else story // 2)


META_SECTIONS = ("bugs", "infobox", "objectives", "gallery", "videos", "achievements", "trophies", "gwent", "references")
STORY_LINK_TERMS = ("owing to", "led to", "leads to", "because", "caused", "resulted in", "as a result", "became", "turned into",
                    "transformed", "born of", "born from", "if you", "depending on", "chooses", "choice",
                    "spared", "killed", "after ")


OUTCOME_CLAUSE = re.compile(r"\bif (?:the )?\w+ (?:chooses|chose|spares|spared|kills|killed|frees|freed|saves|saved|lifts|lifted|"
                            r"refuses|refused|helps|helped|fights|fought|turns|turned)\b")   # "if <someone> spared it: ..."


def passage_role_bonus(text, topic, game=""):
    """Modest priority for passages naming the topic's subject, relationships, status changes and cause/outcome links;
    a penalty for page furniture (bug lists, infoboxes, objective checklists) unless the topic is about that."""
    excerpt = excerpt_of(text).casefold()
    text = text.casefold()
    topic_words = _terms(topic)
    bonus = 6 if any(word in text for word in topic_terms(topic, game)) else 0
    section = re.match(r"\(([^)]*)\)", excerpt)
    if section and any(meta in section.group(1) for meta in META_SECTIONS) and not any(m[:3] in w for m in META_SECTIONS for w in topic_words):
        bonus -= 8
    if any(term in excerpt for term in STORY_LINK_TERMS) or OUTCOME_CLAUSE.search(excerpt):
        bonus += 4
    relationship_terms = ("wife", "husband", "brother", "sister", "family", "marriage", "allied", "enemy",
                          "friend", "rival", "mentor", "betray")
    status_terms = ("leader", "captain", "noble", "ruin", "debt", "wealth", "fortune", "exile", "crowned", "defeated")
    if any(term in text for term in relationship_terms):
        bonus += 3
    if any(term in text for term in status_terms):
        bonus += 3
    return bonus


def story_expansion_terms(evidence, terms, limit=16):
    """Pseudo-relevance feedback: distinctive words that recur in the passages naming the topic (for a creature, the
    words used to say how it arises). Passages that explain the topic without naming it (a cause told on a character's
    page) share these words. Computed from the evidence itself, so it works for any game or topic."""
    passages = split_evidence_passages(evidence)
    on_df, all_df = {}, {}
    for passage in passages:
        words = {w for w in _norm_words(excerpt_of(passage["text"])) if len(w) >= 5 and w not in _GENERIC_WORDS}
        on = any(term in passage["text"].casefold() for term in terms)
        for word in words:
            all_df[word] = all_df.get(word, 0) + 1
            if on:
                on_df[word] = on_df.get(word, 0) + 1
    rare = max(8, len(passages) // 20)
    ranked = sorted(((on_df[w], -all_df[w], w) for w in on_df
                     if on_df[w] >= 3 and all_df[w] <= rare and not any(t in w for t in terms)), reverse=True)
    return {w for _, _, w in ranked[:limit]}


def compact_passage(passage):
    """The model needs the id, the source name and the excerpt; URL, licence and status lines only use up context."""
    source = get_metadata(passage["text"], "Source") or "unknown"
    return f"[{passage['id']}]\nSource: {source}\nExcerpt: {excerpt_of(passage['text'])}"


def select_relevant_evidence(evidence, topic, game=""):
    """Select diverse evidence from multiple sources, preserving citation IDs."""
    source_groups = {}
    eligible = []
    terms = topic_terms(topic, game)
    source_has_topic = {}
    for passage in split_evidence_passages(evidence):
        name = get_metadata(passage["text"], "Source")
        if name and any(word in passage["text"].casefold() for word in terms):
            source_has_topic[normalise(name)] = True
    expansion = story_expansion_terms(evidence, terms)
    for index, passage in enumerate(split_evidence_passages(evidence)):
        source_name = get_metadata(passage["text"], "Source")
        if not source_name:
            continue
        score = relevance_score(passage, topic, game)
        if score > -100 and not any(term in passage["text"].casefold() for term in terms):
            # passages that explain the topic often never name it (a cause told on a character's page), so words that
            # cluster in the on-topic passages count for them
            words = set(_norm_words(excerpt_of(passage["text"])))
            score += min(8, 4 * len(words & expansion))
        # A passage can be relevant without repeating the topic's words, so passages from a source
        # that has on-topic text elsewhere get a small boost instead of being dropped.
        if score > -100 and source_has_topic.get(normalise(source_name), False):
            score += MIN_RELEVANCE_SCORE
        if score < MIN_RELEVANCE_SCORE:
            continue
        record = {
            "score": score,
            "bonus": passage_role_bonus(passage["text"], topic, game),
            "index": index,
            "passage": passage,
            "source": source_name,
            "source_key": normalise(source_name),
        }
        eligible.append(record)
        source_groups.setdefault(record["source_key"], []).append(record)

    if not eligible:
        return "", set(), None

    selected = []
    selected_ids = set()
    ranked_sources = sorted(source_groups.items(), key=lambda pair: (
        -max(record["score"] + record["bonus"] for record in pair[1]),
        -len(pair[1]),
        min(record["index"] for record in pair[1]),
    ))

    selected_words = []

    def add(record):
        passage_id = record["passage"]["id"]
        if passage_id in selected_ids or len(selected) >= MAX_EVIDENCE_PASSAGES:
            return
        words = set(_norm_words(excerpt_of(record["passage"]["text"])))
        if words and any(len(words & other) / len(words) >= 0.85 for other in selected_words):
            return   # near-duplicate of a passage already selected (the same text mirrored on two wikis)
        selected.append(record)
        selected_ids.add(passage_id)
        selected_words.append(words)

    # source diversity only among sources that actually discuss the topic
    ranked_sources = [pair for pair in ranked_sources if source_has_topic.get(pair[0], False)] or ranked_sources
    for _, records in ranked_sources:
        ranked_records = sorted(records, key=lambda record: (
            -(record["score"] + record["bonus"]), record["index"]
        ))
        if ranked_records:
            add(ranked_records[0])
    for _, records in ranked_sources:
        ranked_records = sorted(records, key=lambda record: (
            -(record["score"] + record["bonus"]), record["index"]
        ))
        if len(ranked_records) > 1:
            add(ranked_records[1])
    for record in sorted(eligible, key=lambda item: (
        -(item["score"] + item["bonus"]), item["index"]
    )):
        add(record)

    selected.sort(key=lambda record: record["index"])
    selected_sources = list(dict.fromkeys(record["source"] for record in selected))
    source_label = "Multiple sources: " + "; ".join(selected_sources)
    header = (
        f"GAME: {get_metadata(evidence, 'GAME')}\n"
        f"TOPIC: {get_metadata(evidence, 'TOPIC')}\n"
        f"SELECTED SOURCES: {source_label}\n"
        "Evidence was retrieved but has not been independently verified. "
        "Relevance scores do not establish accuracy.\n\n"
    )
    selected_text = "\n\n".join(compact_passage(record["passage"]) for record in selected)
    valid_ids = {record["passage"]["id"] for record in selected}
    return header + selected_text, valid_ids, source_label


def _is_meta_passage(text):
    section = re.match(r"\(([^)]*)\)", excerpt_of(text).casefold())
    return bool(section) and any(meta in section.group(1) for meta in META_SECTIONS)


def expand_story_evidence(game, topic, evidence, focused_evidence, valid_ids, batch_size=40, per_batch=3):
    """Word-overlap selection cannot know that a passage which never names the topic (a cause told on a character's
    page, an alternative account) is a step of its story. The model is shown the unselected passages in batches and may
    add up to STORY_EXTRA_PASSAGES of them. It only chooses ids; code copies the passages verbatim and every claim
    built on them is still audited. Returns (focused_evidence, valid_ids, added_ids)."""
    if STORY_EXTRA_PASSAGES <= 0:
        return focused_evidence, valid_ids, []
    ranked = []
    for index, passage in enumerate(split_evidence_passages(evidence)):
        if passage["id"] in valid_ids or _is_meta_passage(passage["text"]) or len(excerpt_of(passage["text"])) < 60:
            continue
        if relevance_score(passage, topic, game) <= -100:
            continue
        rank = relevance_score(passage, topic, game) + passage_role_bonus(passage["text"], topic, game)
        ranked.append((-rank, index, passage))
    pool = [p for _, _, p in sorted(ranked)]
    if not pool:
        return focused_evidence, valid_ids, []
    selected_lines = "\n".join(f"[{p['id']}] {excerpt_of(p['text'])[:150]}" for p in split_evidence_passages(focused_evidence)
                                if p["id"] in valid_ids)
    chosen = []
    allowed = {p["id"]: p for p in pool}
    for start in range(0, len(pool), batch_size):
        if len(chosen) >= STORY_EXTRA_PASSAGES:
            break
        batch = pool[start:start + batch_size]
        candidate_lines = "\n".join(f"[{p['id']}] {excerpt_of(p['text'])[:260]}" for p in batch)
        prompt = f"""A short video will tell the story of "{topic}" in the game "{game}". The SELECTED passages below will be its evidence.
From the CANDIDATES, choose at most {per_batch} passages that state a step of that story the selected passages are missing or only hint at: what caused it, who was responsible, a different account of the cause, what happened just before it, or what each choice leads to. Ignore gameplay tips, rewards, bugs and side plots that the subject does not depend on. Choosing none is fine.
Reply with one line: IDS: E###, E### (or IDS: NONE).
SELECTED:
{selected_lines}
CANDIDATES:
{candidate_lines}"""
        try:
            raw = ask_ollama(prompt, temperature=0.0)
        except Exception as error:
            print(f"Story-evidence expansion stopped: {error}")
            break
        match = re.search(r"(?im)^\s*IDS\s*:\s*(.+)$", raw)
        batch_ids = {p["id"] for p in batch}
        picked = [i for i in dict.fromkeys(re.findall(r"E\d+", match.group(1) if match else "")) if i in batch_ids]
        chosen.extend(i for i in picked[:per_batch] if i not in chosen)
    chosen = chosen[:STORY_EXTRA_PASSAGES]
    if not chosen:
        return focused_evidence, valid_ids, []
    focused_evidence = focused_evidence.rstrip() + "\n\n" + "\n\n".join(compact_passage(allowed[i]) for i in chosen)
    return focused_evidence, set(valid_ids) | set(chosen), chosen


ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "st", "vs", "jr", "sr", "no", "inc", "co", "etc", "vol"}


def strip_markdown(text):
    """Models wrap titles in *asterisks*; they must never reach captions/TTS or verbatim-hook checks."""
    return re.sub(r"[*_`#]+", "", text).replace("\u2019", "'").strip()


def sentence_count(text):
    """Count sentence terminators, ignoring decimals/abbreviations such as 'Dr.' or '3.5'."""
    count = 0
    for match in re.finditer(r"[.!?]+(?=\s|$)", text):
        before = re.findall(r"[\w']+$", text[:match.start()])
        if match.group(0) == "." and before and before[0].casefold() in ABBREVIATIONS:
            continue
        count += 1
    return count


def strip_game_title(text, game):
    """Replace the game title with a single placeholder word so punctuation/length in titles don't count."""
    normalised = text
    variants = {game.strip(), game_title_variants_raw(game)}
    variants |= {re.sub(r"^(?:the|a|an)\s+", "", v, flags=re.IGNORECASE) for v in list(variants)}   # e.g. "Game 3" for "The Game 3"
    for variant in sorted(variants, key=len, reverse=True):
        if variant and len(variant) >= 3:
            normalised = re.sub(re.escape(variant), "TITLE", normalised, flags=re.IGNORECASE)
    return normalised


def game_title_variants_raw(game):
    return re.split(r"\s*[:\u2014\u2013]\s*|\s+-\s+", game.strip(), maxsplit=1)[0].strip()


def hook_word_count(text, game):
    spoken = strip_game_title(remove_citation(text), game)
    return len(re.findall(r"[\w'-]+", spoken))


def parse_script_line(line):
    """Parse a tagged spoken line, tolerating common Markdown/list wrappers."""
    clean_line = line.strip()
    clean_line = re.sub(r"^\s*(?:(?:[-*#]+)\s*|\d+[.)]\s*)", "", clean_line)
    clean_line = re.sub(r"[*_`]", "", clean_line).strip()
    match = re.match(
        r"^(HOOK|NARRATION|PAYOFF|ENDING QUESTION)\s*[:\-–—]?\s*(.*?)\s*$",
        clean_line,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper(), match.group(2).strip()
    return None, ""


def spoken_lines(draft):
    return [
        (section, content)
        for line in draft.splitlines()
        for section, content in [parse_script_line(line)]
        if section and content
    ]


def remove_citation(text):
    return CITATION_TAIL.sub("", text).strip()


def word_count(draft):
    words = []
    for _, content in spoken_lines(draft):
        words.extend(re.findall(
            r"\b[\w]+(?:['’][\w]+)*(?:-[\w]+)*\b",
            remove_citation(content),
            flags=re.UNICODE,
        ))
    return len(words)


def get_draft_status(draft):
    match = re.search(r"(?mi)^\s*STATUS:\s*(.+?)\s*$", draft)
    return match.group(1).strip().upper() if match else None


def canonicalise_citation_ids(draft, valid_ids):
    by_number, changes = {}, []
    for evidence_id in valid_ids:
        by_number.setdefault(int(evidence_id[1:]), []).append(evidence_id)

    def replace_id(match):
        supplied = match.group(1)
        if supplied in valid_ids:
            return f"[{supplied}]"
        candidates = by_number.get(int(supplied[1:]), [])
        if len(candidates) == 1:
            changes.append((supplied, candidates[0]))
            return f"[{candidates[0]}]"
        return match.group(0)

    # "[E012, E013]" -> "[E012] [E013]"
    draft = re.sub(r"\[(E\d+(?:\s*[,;]\s*E\d+)+)\]", lambda m: " ".join(f"[{part}]" for part in re.split(r"\s*[,;]\s*", m.group(1))), draft)
    updated = re.sub(r"\[(E\d+)\]", replace_id, draft)
    # Models sometimes put sentence punctuation after the citation ("Sentence. [E001].").
    # Citations must be the final characters, so remove only punctuation immediately after an ID.
    updated = re.sub(r"(?m)(\[E\d+\])[.!?]+[ \t]*$", r"\1", updated)
    # Also normalise the common duplicated-punctuation form without moving citations across prose.
    updated = re.sub(r"([.!?])\s*(\[E\d+\])[.!?]+(?=\s|$)", r"\1 \2", updated)
    return updated, changes


def _normalised_title(text):
    text = text.casefold().replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def game_title_variants(game):
    """Return grounded title forms, including the main title before a subtitle."""
    raw_variants = [game.strip()]
    # Many metadata titles include a subtitle after a colon, dash, or em dash.
    main_title = re.split(r"\s*[:—–]\s*|\s+-\s+", game.strip(), maxsplit=1)[0].strip()
    if main_title and main_title.casefold() != game.strip().casefold():
        raw_variants.append(main_title)

    variants = []
    for variant in raw_variants:
        normalised = _normalised_title(variant)
        if len(normalised) >= 3 and normalised not in variants:
            variants.append(normalised)
        # Permit omission of an initial article while retaining the actual title words.
        without_article = re.sub(r"^(the|a|an)\s+", "", normalised)
        if len(without_article) >= 3 and without_article not in variants:
            variants.append(without_article)
    return variants


def hook_identifies_game(text, game):
    """A candidate must name the metadata game title (or its main title)."""
    normalised_hook = _normalised_title(remove_citation(text))
    if not normalised_hook:
        return False
    for title in game_title_variants(game):
        if re.search(rf"(?<!\w){re.escape(title)}(?!\w)", normalised_hook):
            return True
    return False


def is_generic_hook(text):
    clean = remove_citation(text).strip().casefold()
    return any(re.search(pattern, clean) for pattern in [
        r"^(meet|this is|here is|here's|the story of)\b",
        r"^(who is|what is|everything about)\b",
        r"^today we('re| are)\b",
    ])


def normalise_hook_citation(text, valid_ids):
    """Accept 1-2 trailing evidence ids in any common shape; return 'sentence. [E001] [E002]'."""
    rest = text.strip()
    if re.search(r"\]\s*[.!?]+\s*$", rest):
        rest = re.sub(r"\s*[.!?]+\s*$", "", rest)   # punctuation was placed after the citation
    ids = []
    while len(ids) <= MAX_CITATIONS_PER_LINE:
        match = re.search(r"[\s,;:—–-]*\[?(E\d+)\]?\s*$", rest, re.IGNORECASE)
        if not match:
            break
        ids.insert(0, match.group(1).upper())
        rest = rest[:match.start()]
    if not ids:
        return None, "must end with an evidence ID such as [E008]"
    if len(ids) > MAX_CITATIONS_PER_LINE:
        return None, f"cite at most {MAX_CITATIONS_PER_LINE} evidence IDs"

    by_number = {}
    for evidence_id in valid_ids:
        by_number.setdefault(int(evidence_id[1:]), []).append(evidence_id)
    canonical = []
    for supplied in ids:
        found = supplied if supplied in valid_ids else None
        if found is None and len(by_number.get(int(supplied[1:]), [])) == 1:
            found = by_number[int(supplied[1:])][0]
        if found is None:
            return None, f"citation {supplied} is not a valid selected evidence ID"
        canonical.append(found)

    spoken = rest.rstrip(" \t,;:—–-").rstrip()
    if not spoken:
        return None, "citation has no hook sentence before it"
    if not spoken.endswith(("?", "!", ".")):
        spoken += "."
    return f"{spoken} " + " ".join(f"[{i}]" for i in canonical), None


class HookSelectionError(Exception):
    """Raised when no hook is both grounded and good enough. The pipeline stops instead of guessing."""


HOOK_BANNED_OPENERS = (
    r"^(did you know|have you ever wondered|have you ever|what if i told you|ever wondered|do you know)\b",
    r"^(this is the story of|the story of|let me tell you|today we|you won't believe|you will not believe)\b",
    r"^(meet|here is|here's|who is|what is|everything about)\b",
)
HOOK_VAGUE_PHRASES = (
    r"\ba dark secret\b", r"\bdark secrets?\b", r"\bsomething (sinister|dark|terrible|strange|evil)\b",
    r"\bthe truth (about|behind)\b", r"\bnothing (is|was) (as|what) (it|they) (seems?|seemed)\b",
    r"\bwill (shock|surprise|amaze) you\b", r"\bchanges? everything\b", r"\bfate of\b",
    r"\bhangs? in the balance\b", r"\bsurvival of\b", r"\bsecrets? of\b", r"\bunravel(s|ing)?\b",
    r"\bancient (evil|power|mystery)\b", r"\bgrim (truth|reality|fate)\b", r"\bpits? \w+ against\b",
    r"\bdelves? into\b", r"\bshrouded in\b",
)
_NAME_STOPWORDS = set(
    "a an the this that these those it its he she they we you i my your his her their our and or but so if "
    "when while after before once then there here what who whom whose which why how can could would should "
    "will must may might do does did is are was were be been being has have had not no yes one two three "
    "four five six seven eight nine ten first second last every each all some any in on at by for from to "
    "of with without into onto over under about against between during until as than too very just only "
    "never always still even also now today tonight meet".split()
)


def hook_proper_nouns(text, game, reference_text=""):
    """Names (capitalised words not at a sentence start, plus a start word that is only ever capitalised in the
    evidence) other than the game title. Used to penalise names a newcomer cannot understand."""
    spoken = strip_game_title(remove_citation(text), game)
    tokens = re.findall(r"[A-Za-z][A-Za-z'’-]*", spoken)
    reference = reference_text or ""
    names, current, started_sentence = [], [], True
    previous_end = 0
    for match in re.finditer(r"[A-Za-z][A-Za-z'’-]*", spoken):
        token = re.sub(r"(?:'s|’s)$", "", match.group(0))
        gap = spoken[previous_end:match.start()]
        previous_end = match.end()
        if re.search(r"[.!?]", gap):
            started_sentence = True
        if current and re.search(r"[,;:()\u2014-]", gap):   # "Orsk, Tamsin" is two names, not one
            names.append(" ".join(current))
            current = []
        is_name = False
        if token == "TITLE" or token.lower() in _NAME_STOPWORDS:
            is_name = False
        elif token[0].isupper():
            if not started_sentence:
                is_name = True
            elif reference:   # sentence-initial word: a name only if the evidence never writes it in lower case
                is_name = bool(re.search(rf"\b{re.escape(token)}\b", reference)) and not re.search(
                    rf"\b{re.escape(token.lower())}\b", reference)
        if is_name:
            current.append(token)
        else:
            if current:
                names.append(" ".join(current))
                current = []
        started_sentence = False
    if current:
        names.append(" ".join(current))
    unique = []
    for name in names:
        if name.casefold() not in {item.casefold() for item in unique}:
            unique.append(name)
    return unique


def tighten_hook(text, game, valid_ids):
    """A good premise written too long is cut down instead of thrown away: the model shortens it once, keeping its
    meaning, its strongest detail and its citations. The result is audited and scored like any other hook."""
    prompt = f"""Shorten this opening line of a short video to at most {HOOK_TARGET_MAX_WORDS} words. Keep its premise, its tension and its most specific detail; cut filler words and secondary clauses. Add nothing new. One sentence. Keep the evidence ids at the end exactly as they are.
Reply with only the shortened line.
LINE: {text}"""
    try:
        raw = ask_writer(prompt, temperature=0.2)
    except Exception:
        return None
    line = next((l.strip() for l in raw.splitlines() if l.strip()), "")
    line = re.sub(r"^(?:HOOK\s*\d*\s*(?:\([^)]*\))?\s*:\s*|LINE\s*:\s*)", "", strip_markdown(line)).strip().strip('"')
    if not line_citations(line):
        line = remove_citation(line).strip() + " " + " ".join(f"[{i}]" for i in line_citations(text))
    canonical, error = normalise_hook_citation(line, valid_ids)
    if error or not canonical:
        return None
    if not HOOK_HARD_MIN_WORDS <= hook_word_count(canonical, game) <= HOOK_HARD_MAX_WORDS:
        return None
    if sentence_count(strip_game_title(remove_citation(canonical), game)) != 1:
        return None
    return canonical


def parse_hook_candidates(raw, valid_ids, game):
    """Return (candidates, reasons, skipped). Candidates are dicts; skipped lists approaches the model said the
    evidence cannot support (that is a legitimate answer, not an error)."""
    candidates, reasons, skipped, seen = [], [], [], set()
    for line in raw.splitlines():
        line = re.sub(r"^\s*[:\-*]*\s*", "", line.strip())
        match = re.match(
            r"^\*{0,2}\s*HOOK\s+(\d+)\s*(?:[\(\[]\s*([A-Za-z_ /-]+?)\s*[\)\]])?\s*:\s*(.+?)\s*\*{0,2}$",
            line, re.IGNORECASE)
        if not match:
            continue
        number = int(match.group(1))
        approach = re.sub(r"[\s/-]+", "_", (match.group(2) or "").strip()).upper()
        raw_text = strip_markdown(match.group(3))

        if number < 1 or number > max(HOOK_CANDIDATES, len(HOOK_APPROACHES)) or number in seen:
            reasons.append(f"HOOK {number}: invalid or duplicate candidate number")
            continue
        seen.add(number)

        if re.match(r"^NONE\b", raw_text, re.IGNORECASE):
            skipped.append(f"HOOK {number} ({approach or 'unlabelled'}): {raw_text}")
            continue

        canonical, error = normalise_hook_citation(raw_text, valid_ids)
        if error:
            reasons.append(f"HOOK {number}: {error}")
            continue
        canonical = re.sub(r'["\u201c\u201d\u201e\u201f\u00ab\u00bb]', "", canonical)   # formatting, not content: strip, never reject
        spoken = remove_citation(canonical)
        if sentence_count(strip_game_title(spoken, game)) != 1:
            reasons.append(f"HOOK {number}: must be exactly one sentence")
            continue
        words = hook_word_count(canonical, game)
        if words > HOOK_HARD_MAX_WORDS:
            tightened = tighten_hook(canonical, game, valid_ids) if words <= HOOK_TIGHTEN_MAX_WORDS else None
            if not tightened:
                reasons.append(f"HOOK {number}: has {words} words; cut it to {HOOK_TARGET_MAX_WORDS} or fewer")
                continue
            reasons.append(f"HOOK {number}: had {words} words; tightened to {hook_word_count(tightened, game)}")
            canonical, spoken, words = tightened, remove_citation(tightened), hook_word_count(tightened, game)
        if words < HOOK_HARD_MIN_WORDS:
            reasons.append(f"HOOK {number}: only {words} words; it needs a concrete situation, aim for {HOOK_TARGET_MIN_WORDS}-{HOOK_TARGET_MAX_WORDS}")
            continue
        if is_generic_hook(canonical) or any(re.search(p, spoken.strip().casefold()) for p in HOOK_BANNED_OPENERS):
            reasons.append(f"HOOK {number}: banned opener (did you know / have you ever wondered / this is the story of / meet ...); start with the concrete situation")
            continue
        candidates.append({"number": number, "approach": approach or "UNLABELLED", "text": canonical})
    return candidates, reasons, skipped


NOT_A_CAUSE = ("A CAUSE says why these people are in this trouble: what happened that brought the situation about. These are "
               "NOT causes: how the problem can be solved or was solved (a method, a trick, a fight, what makes something leave "
               "or end), instructions or requirements for solving it, and a rule of the world written as a recipe. Keep every "
               "role exactly as the passage gives it: a word naming who does something (a hunter, a thief, a trickster) means the person who does it.")


def develop_story_angle(game, topic, evidence, valid_ids, cause_facts=None):
    ids_text = ", ".join(sorted(valid_ids, key=lambda value: int(value[1:])))
    causes = ""
    if cause_facts:
        causes = ("CAUSES ALREADY FOUND IN THE EVIDENCE (verified quotes; ROOT_CAUSE and COMPETING_ACCOUNTS must reflect them, and "
                  "none of them may appear in DETAILS_TO_OMIT):\n" + "\n".join(f"- [{f['id']}] ({f['kind']}) {f['fact']}" for f in cause_facts) + "\n")
    prompt = f"""Analyse the evidence and choose the strongest evidence-backed angle for a ~35 second video. Do NOT default to defining a creature, item or place and its origin: lead with whatever is most surprising or consequential, such as a meaningful choice, a transformation, a surprising consequence or a mystery, but only when the evidence states it.
{NOT_A_CAUSE}
Ground every field strictly in the evidence. Do not invent chronology, events, motives, causes, consequences, dialogue or decisions. When the evidence states a condition, requirement or trade-off, keep exactly who it applies to and what it is a condition for. Fields must contain ONLY what the evidence states: no theme words (compelling, profound, moral weight), no speculation.
If the evidence does not support a real conflict, choice or consequence, say so in CENTRAL_CONFLICT and MISSING instead of inventing one.
Use only the exact IDs listed in VALID IDS.
Return these fields only:
PLAIN_SETUP: one sentence saying who is involved and what situation they are in, using ONLY roles and plain nouns that appear in the evidence (never add a place, setting or feeling the evidence does not state) and NO character, place or faction names
MOST_SURPRISING_FACT: one sentence, the most surprising or consequential supported fact (not a generic definition of the subject)
ROOT_CAUSE: one sentence saying who or what caused the central creature, event or situation to come about (never how it is or can be solved), including any person's action or responsibility, exactly as strongly as the evidence states it (keep who believes or claims it) and never softened or left out, ending with the id(s) of the passage(s) that state it, e.g. [E012]; or NONE if no passage states it. Two events in the same story are not cause and effect unless a passage says so.
COMPETING_ACCOUNTS: if passages give different explanations of the cause, one sentence naming each and whose account it is, with ids; or NONE
CENTRAL_CONFLICT: one sentence, the strongest supported dilemma, choice or consequence
STAKES: one sentence explaining what is on the line
WHAT_VIEWERS_WILL_WANT_NEXT: the one question that keeps a viewer watching to the end, which the evidence answers. It is about PEOPLE and STAKES: who is to blame, what happens to someone they would care about, whether a choice saves or dooms them, what the truth turns out to be. Never a how-to or mechanics question (how a spell, device, system or process works, what its steps are): viewers want to know what happens and what it costs, not the procedure.
PAYOFF_THE_VIDEO_CAN_DELIVER: one sentence, the supported answer to WHAT_VIEWERS_WILL_WANT_NEXT (the outcome, the truth or the cost), not a procedure
DETAILS_TO_OMIT: only peripheral background (side characters, unrelated plot threads) that a newcomer does not need. NEVER list what caused the central situation, anyone's responsibility for it, or any fact needed to understand the stakes
ALTERNATIVE_ANGLES: two other supported angles, one sentence each, clearly different from the best one
BEST_STORY_ANGLE: one sentence
MISSING: what the evidence lacks for a stronger story, including any causal link a viewer might assume but no passage states, or NONE
EVIDENCE_IDS: list of IDs supporting the angle
GAME: {game}
TOPIC: {topic}
VALID IDS: {ids_text}
{causes}EVIDENCE:
{evidence}"""
    return ask_ollama(prompt, temperature=0.25)


def extract_cause_facts(game, topic, evidence, valid_ids, max_facts=5):
    """A dedicated search for what CAUSED the subject, run independently of the story brief so an angle that skipped the
    cause (or listed it as omittable) cannot hide it. Different accounts are kept separately. Quotes are verified in
    code; the facts then go through verify_cause_facts like any other cause fact."""
    passage_map = {item["id"]: item["text"] for item in split_evidence_passages(evidence)}
    ids_text = ", ".join(sorted(valid_ids, key=lambda value: int(value[1:])))
    prompt = f"""The subject of a short video is "{topic}" in the game "{game}".
{NOT_A_CAUSE}
Only causes of "{topic}" itself count: the evidence may also describe other storylines, quests or incidents in the same game; their events are not causes of this one.
Also give the LINK that explains WHY that event led to the situation, whenever the evidence states it (for example a rule about which kind of person the creature, curse or threat chooses, or what it feeds on), as its own fact phrased with "because": without it a viewer hears two events but not why one led to the other.
List what the evidence says CAUSED the subject to exist, and what caused that event in turn, including any person's harmful action or responsibility (state it plainly; never soften it). If passages give DIFFERENT explanations, list each one separately with KIND account and keep who says or believes it.
Each line states ONE link (one cause and the effect it led to) using a word such as "led to", "because", "owing to" or "became", in at most 24 words, adding nothing the passage does not say.
Output only lines like this (at most {max_facts}), or exactly NONE:
FACT 1: <plain sentence> || QUOTE: <shortest verbatim span copied from that passage> || ID: E### || KIND: cause
Use only these IDs: {ids_text}
EVIDENCE:
{evidence}"""
    raw = ask_ollama(prompt, temperature=0.1)
    facts, notes = [], []
    for line in raw.splitlines():
        match = re.match(r"(?i)^\s*(?:[-*]\s*)?FACT\s*\d*\s*:\s*(.+?)\s*\|{1,2}\s*QUOTE\s*:\s*(.+?)\s*\|{1,2}\s*ID\s*:\s*\[?(E\d+)\]?\s*(?:\|{1,2}\s*KIND\s*:\s*(\w+))?\s*$", re.sub(r"[*`]", "", line.strip()))
        if not match:
            continue
        fact, quote, evidence_id = strip_markdown(match.group(1)), match.group(2).strip().strip('"\u201c\u201d\'').strip(), match.group(3).upper()
        kind = "account" if (match.group(4) or "").lower().startswith("acc") else "cause"
        holders = [i for i in sorted(valid_ids, key=lambda v: int(v[1:])) if i in passage_map and quote_in_text(quote, excerpt_of(passage_map[i]))]
        if not holders:
            notes.append(f"cause search: quote not found in any selected passage: {fact}")
            continue
        if evidence_id not in holders:
            evidence_id = holders[0]
        facts.append({"id": evidence_id, "fact": fact, "quote": quote, "kind": kind})
    return facts[:max_facts], notes


def search_causes_all_evidence(game, topic, evidence, focused_evidence, valid_ids, batch_chars=14000):
    """Run the cause search over the WHOLE evidence file in batches, not only the selected passages: what caused the
    subject is often told on another page (a character's biography, a journal entry) that word-overlap selection
    misses, and different accounts of the cause can sit far apart. Passages holding a verified cause are added to the
    selection (at most CAUSE_EXTRA_PASSAGES). Returns (focused_evidence, valid_ids, cause_facts, notes)."""
    if not CAUSE_SEARCH:
        return focused_evidence, valid_ids, [], []
    selected = [p for p in split_evidence_passages(focused_evidence) if p["id"] in valid_ids]
    others = [p for p in split_evidence_passages(evidence)
              if p["id"] not in valid_ids and not _is_meta_passage(p["text"]) and len(excerpt_of(p["text"])) >= 60]
    batches, current, size = [selected], [], 0
    for passage in others:
        text = compact_passage(passage)
        if current and size + len(text) > batch_chars:
            batches.append(current)
            current, size = [], 0
        current.append(passage)
        size += len(text)
    if current:
        batches.append(current)
    facts, notes, seen = [], [], set()
    for batch in batches:
        if not batch:
            continue
        batch_text = "\n\n".join(compact_passage(p) for p in batch)
        try:
            found, batch_notes = extract_cause_facts(game, topic, batch_text, {p["id"] for p in batch}, max_facts=4)
        except Exception as error:
            notes.append(f"cause search batch failed: {error}")
            continue
        notes += batch_notes
        for fact in found:
            if (fact["id"], fact["quote"]) not in seen:
                seen.add((fact["id"], fact["quote"]))
                facts.append(fact)
    all_passages = {p["id"]: p for p in split_evidence_passages(evidence)}
    added = []
    for fact in facts:
        if fact["id"] not in valid_ids and fact["id"] not in added and len(added) < CAUSE_EXTRA_PASSAGES and fact["id"] in all_passages:
            added.append(fact["id"])
    facts = [f for f in facts if f["id"] in valid_ids or f["id"] in added]
    if added:
        focused_evidence = focused_evidence.rstrip() + "\n\n" + "\n\n".join(compact_passage(all_passages[i]) for i in added)
        valid_ids = set(valid_ids) | set(added)
        notes.append("cause search added passages: " + ", ".join(added))
    return focused_evidence, valid_ids, facts, notes


def extract_hook_facts(game, topic, evidence, valid_ids, story_brief, max_facts=14):
    """Facts-first step: the model proposes short, newcomer-friendly facts, each with a verbatim quote. CODE checks
    the quote exists in the cited passage, so hooks and scripts can then be built from facts that are already verified.
    Each fact has a kind; a 'cause' fact (what brought the central situation about) is later required in the script."""
    passage_map = {item["id"]: item["text"] for item in split_evidence_passages(evidence)}
    ids_text = ", ".join(sorted(valid_ids, key=lambda value: int(value[1:])))
    prompt = f"""From the evidence, pull out up to {max_facts} short facts for a gripping video aimed at someone who has never played "{game}".
Prefer: a choice between two options, a danger to a particular kind of person, a strange or surprising detail, a consequence, and above all WHAT CAUSED the central situation to exist. {NOT_A_CAUSE} A method or trick is KIND choice; what it leads to is KIND consequence. Give a cause fact for EACH step of the chain that the passages state (what caused the situation, and what caused that, up to three steps), including any person's harmful action (state it plainly and do not soften it). A cause fact states ONE link only (one cause and the one effect it led to), uses a word such as "led to", "resulted in", "because" or "became", and keeps who says or believes it and any uncertainty (guesses, believes, claims) exactly as the passage does.
Each fact: ONE plain-English sentence of at most 24 words that a newcomer understands. Replace game-specific terms (creature types, factions, titles, items) with a short plain description taken from the same passage, or from another passage you cite. Add nothing the passage does not say.
For each fact copy the shortest supporting span word for word from that passage.
Cover the whole story, not one convenient fact: the cause chain, the central choice, and what EACH option of that choice leads to (KIND consequence).
If passages give different accounts of what caused it (one person believes X, another passage says Y), give each account as its own fact with KIND account, keeping who says or believes it. Never merge them into one certain cause.
Two events that happen in the same story are not cause and effect unless a passage says so.
Give each fact a KIND: cause, account, choice, danger, consequence or detail. Include cause facts whenever any passage states them.
Use only these IDs: {ids_text}
Output only lines in this exact format:
FACT 1: <plain sentence> || QUOTE: <verbatim span from the passage> || ID: E### || KIND: cause
GAME: {game}
TOPIC: {topic}
STORY ANGLE:
{story_brief}
EVIDENCE:
{evidence}"""
    raw = ask_ollama(prompt, temperature=0.2)
    facts, dropped = [], []
    for line in raw.splitlines():
        match = re.match(r"(?i)^\s*(?:[-*]\s*)?FACT\s*\d*\s*:\s*(.+?)\s*\|{1,2}\s*QUOTE\s*:\s*(.+?)\s*\|{1,2}\s*ID\s*:\s*\[?(E\d+)\]?\s*(?:\|{1,2}\s*KIND\s*:\s*(\w+))?\s*$", line.strip())
        if not match:
            continue
        fact, quote, evidence_id = strip_markdown(match.group(1)), match.group(2).strip().strip('"\u201c\u201d\'').strip(), match.group(3).upper()
        kind = (match.group(4) or "detail").lower()
        if kind not in {"cause", "account", "choice", "danger", "consequence", "detail"}:
            kind = "detail"
        if evidence_id not in passage_map or not quote_in_text(quote, excerpt_of(passage_map[evidence_id])):
            # a real quote cited to the wrong passage is a citation error: re-cite it instead of losing the fact
            holders = [i for i in sorted(valid_ids, key=lambda v: int(v[1:]))
                       if i in passage_map and quote_in_text(quote, excerpt_of(passage_map[i]))]
            if holders:
                dropped.append(f"citation corrected {evidence_id} -> {holders[0]}: {fact}")
                evidence_id = holders[0]
        if evidence_id not in valid_ids or evidence_id not in passage_map:
            dropped.append(f"{evidence_id} is not a selected passage: {fact}")
        elif not quote_in_text(quote, excerpt_of(passage_map[evidence_id])):
            dropped.append(f"quote not found in any selected passage: {fact}")
        elif len(fact.split()) > 32:
            dropped.append(f"too long: {fact}")
        else:
            facts.append({"id": evidence_id, "fact": fact, "quote": quote, "kind": kind})
    dropped = [d for d in dropped if not d.startswith("citation corrected")] + [d for d in dropped if d.startswith("citation corrected")]
    return facts[:max_facts], dropped


def recite_from_facts(candidate_text, facts, game):
    """Models often cite the wrong passage for a hook built from several facts. Cite the verified fact(s) whose wording
    the hook actually uses (best first, at most two), then keep the model's own ids if there is room. The grounding
    audit still has to find a verbatim quote in the cited passages, so this cannot make an unsupported hook pass."""
    if not facts:
        return candidate_text
    hook_stems = content_stems(candidate_text, game)
    scored = []
    for fact in facts:
        fact_stems = content_stems(fact["fact"], game)
        shared = len(hook_stems & fact_stems)
        if len(fact_stems) >= 2 and shared >= 2 and shared / len(fact_stems) >= 0.5:
            scored.append((shared / len(fact_stems), shared, fact["id"]))
    if not scored:
        return candidate_text
    ids = []
    for _, _, evidence_id in sorted(scored, reverse=True):
        if evidence_id not in ids:
            ids.append(evidence_id)
    ids = (ids + [i for i in line_citations(candidate_text) if i not in ids])[:MAX_CITATIONS_PER_LINE]
    return remove_citation(candidate_text).strip() + " " + " ".join(f"[{i}]" for i in ids)


def _cover_ids_for_claims(claims, passage_map, cited):
    """If the auditor found a real quote for every claim, return the (<=2) passage ids that hold those quotes,
    preferring ids already cited. None if any claim has no quote or the quotes need more than two passages."""
    if not claims:
        return None
    cover = []
    for claim, quote in claims:
        if quote.upper() in {"NONE", "N/A", "-", ""}:
            if re.search(r"no factual premise", claim, re.IGNORECASE):
                continue
            return None
        holders = [i for i, text in passage_map.items() if quote_in_text(quote, excerpt_of(text))]
        if not holders:
            return None
        pick = next((i for i in holders if i in cover), None) or next((i for i in holders if i in cited), None) or holders[0]
        if pick not in cover:
            cover.append(pick)
    return cover if 1 <= len(cover) <= MAX_CITATIONS_PER_LINE else None


def audit_hook_candidates(candidates, evidence, game):
    """Grounding GATE: keep only hooks that pass the SAME quote-verified audit the final script lines get.
    Each candidate dict gets 'grounded' and 'audit_reason'. A hook whose claims are all backed by verbatim quotes that
    live in OTHER selected passages is not unsupported, only mis-cited: it is re-cited and passes; every claim still
    needs a quote that code has found in the cited passages."""
    passage_map = {item["id"]: item["text"] for item in split_evidence_passages(evidence)}
    for candidate in candidates:
        result = audit_line("HOOK", candidate["text"], passage_map, game)
        if result["status"] == "RECITE":   # every claim is supported, the citation was wrong: fix the citation
            candidate["mis_cited"] = candidate["text"]
            candidate["text"] = recite_line(candidate["text"], result["recite_ids"])
            result = {**result, "status": "CLEAR"}
        # REVIEW = the checkers disagree on a claim, not that anything was shown to be made up: usable, flagged, ranked lower
        candidate["grounded"] = result["status"] in {"CLEAR", "REVIEW"}
        candidate["needs_review"] = result["status"] == "REVIEW"
        if candidate["grounded"]:
            VERIFIED_HOOKS.add(candidate["text"])
        candidate["audit_reason"] = "" if candidate["grounded"] else result["reason"]
        candidate["audit_status"] = result["status"]
        if not candidate["grounded"] and result["status"] == "FLAGGED":
            cover = _cover_ids_for_claims(result["claims"], passage_map, line_citations(candidate["text"]))
            if cover:
                candidate["mis_cited"] = candidate["text"]
                candidate["text"] = remove_citation(candidate["text"]).strip() + " " + " ".join(f"[{i}]" for i in cover)
                candidate["grounded"], candidate["audit_reason"] = True, ""
                VERIFIED_HOOKS.add(candidate["text"])   # passed here, so the script stage does not judge it again
    return candidates


def build_hook_prompt(game, topic, evidence, valid_ids, story_brief, feedback="", facts=None):
    ids_text = ", ".join(sorted(valid_ids, key=lambda value: int(value[1:])))
    slots = []
    for index in range(1, HOOK_CANDIDATES + 1):
        name, meaning = HOOK_APPROACHES[(index - 1) % len(HOOK_APPROACHES)]
        slots.append(f"HOOK {index} ({name}): {meaning}")
    slot_text = "\n".join(slots)
    layout = "\n".join(
        f"HOOK {index} ({HOOK_APPROACHES[(index - 1) % len(HOOK_APPROACHES)][0]}): [one spoken sentence] [E###]"
        for index in range(1, HOOK_CANDIDATES + 1))
    if facts:
        source_block = (
            story_chain_text(facts) +
            "\nBuild every hook ONLY from these facts. A hook may join up to three of them into one premise (for example a cause and "
            f"the choice it led to, or a choice and what one option leads to); cite the id of every fact it uses (at most {MAX_CITATIONS_PER_LINE}). "
            "Add NO detail that is not in the facts: no setting (village, swamp, castle), no feelings, no motives, no outcomes, "
            "no drama words that assert something the facts do not (eternal, deadly, bloodthirsty, doomed, torment). A plain label such as monstrous is fine for something the facts call a creature that is hostile or hideous.")
    else:
        source_block = f"EVIDENCE:\n{evidence}"
    feedback_block = f"\nPREVIOUS ROUND PROBLEMS (fix these, write NEW hooks, do not reuse the failed wording):\n{feedback}\n" if feedback else ""
    return f"""You write the first spoken sentence of a ~35 second vertical gaming-lore video. A viewer who has NEVER played "{game}" hears it once and decides in two seconds whether to keep watching.
Write {HOOK_CANDIDATES} hooks, one per approach below. Each is a genuinely different way in, not a rewrite of the others:
{slot_text}

WHAT MAKES A GOOD HOOK
- A concrete situation a stranger instantly understands: who is in trouble or must choose, and what is at stake, in plain words.
- Newcomers do not know the game's own words. Replace every game-specific term (a creature type, faction, title, spell, item) with a short plain description taken from the facts; the name may follow only after the description. Everyday words are fine.
- If a fact is marked CAUSE because it says what brought the situation about, build at least one hook on it. Never soften who is responsible.
- The hook makes a promise (a question the viewer now wants answered). Only promise what the facts can answer.
- Lead with the most personal stake the facts support: who is hurt, grieving, guilty or at risk, and why it matters to them (a parent and a child, a betrayal, a life). Feeling comes from that concrete stake, never from drama words.
- Names are fine: the footage shows who a character is. Use names, roles or relationships, whichever hits harder, but at most two names in one hook so it stays easy to follow.
- SPECIFIC beats general: include at least one concrete, distinctive detail straight from the facts (a specific act, object, place, relationship or consequence), the detail that makes this story unlike any other. A hook that could describe many stories ("a man's actions led to tragedy") fails. Use the strongest wording the facts support.
- Open a curiosity gap: show the problem or the choice, never its resolution.
- PLAIN WORDS: every word must be understood instantly by anyone hearing it once: everyday spoken words, never a clinical, formal or uncommon one when a plain one exists (say "killed", not "slain"; "sick", not "afflicted"). Hinting that something hidden is behind what is happening ("but he has no idea what is really behind it") is a strong way to keep the cause secret.
- {HOOK_TARGET_MIN_WORDS}-{HOOK_TARGET_MAX_WORDS} words, one sentence, natural when spoken aloud. Short clauses, at most two commas. It does not need to be a question and must not be clickbait.
- Never give away the resolution (how the curse is lifted, how the trick works, who survives): that is what the viewer stays for.
- MIX: make at least two hooks game-first (name "{game}" within the first six words, for example "In {game}, you must choose...", so fans of the game recognise it) and at least two conflict-first (open straight on the situation without naming the game). Either way the concrete situation must follow immediately. Never force the title where it reads awkwardly; the script's next line introduces the game anyway.

{style_hook_block(topic, game)}NEVER
- Open with "Did you know", "Have you ever wondered", "This is the story of", or "Meet".
- Use vague filler such as "a dark secret", "the fate of", "hangs in the balance", "the truth about", "changes everything".
- Use a name, place or faction a newcomer would not understand without explanation.
- Assert anything the cited evidence does not state: no invented motives, causes, outcomes, numbers, intensifiers (deadly, desperate, doomed) or secrets. Interpretation is allowed only if the evidence itself supports it.
- If the evidence cannot honestly support an approach, write: HOOK n (APPROACH): NONE - reason. That is better than inventing a hook.
FORMAT: plain text, no quotation marks, no markdown. End each hook with its punctuation and then the evidence ID(s) of the facts it uses, such as [E012] or [E012] [E040] (at most {MAX_CITATIONS_PER_LINE}). The cited passages must directly support every claim in the hook.
GAME: {game}
TOPIC: {topic}
VALID IDS: {ids_text}
STORY ANGLE:
{story_brief}
{source_block}
{feedback_block}
Output exactly {HOOK_CANDIDATES} lines, nothing else:
{layout}"""


def clarify_hooks(game, topic, candidates, facts, evidence, valid_ids, beat_rank=-1):
    """A grounded hook that is held back ONLY because a name or game term is unexplained (low clarity) is often the
    strongest premise. Instead of discarding it, the model rewrites it once, replacing each unexplained term with a
    short plain description taken from the facts (a name may stay if its role follows it). Everything else in the
    hook stays. The rewrite is a NEW candidate: it is audited, evaluated and scored like any other."""
    def only_clarity(c):
        return (c.get("grounded") and c.get("evaluation") and c.get("unexplained_terms")
                and c["evaluation"]["clarity"] < HOOK_MIN_CLARITY)
    def potential(c):   # its retention rank if the rewrite only lifts clarity to the passing floor
        return retention_rank(dict(c["evaluation"], clarity=max(c["evaluation"]["clarity"], HOOK_MIN_CLARITY)))
    targets = sorted([c for c in candidates if only_clarity(c) and potential(c) >= beat_rank],
                     key=lambda c: (-potential(c), -c["score"]["total"]))[:HOOK_CLARIFY_MAX]
    variants = []
    fact_lines = "\n".join(f"[{f['id']}] {f['fact']}" for f in (facts or [])) or evidence
    for c in targets:
        terms = ", ".join(c["unexplained_terms"])
        prompt = f"""This opening line for a short video about "{topic}" in the game "{game}" is strong, but a newcomer does not understand: {terms}.
Rewrite it ONCE. Replace or follow each of those terms with a short plain description taken from the FACTS (for a person, their role: who they are in this story). Keep everything else: the same premise, the same tension, the same supported details. Add nothing the facts do not state. At most {HOOK_TARGET_MAX_WORDS + HOOK_LENGTH_GRACE} words, one sentence, ending with the evidence id(s) it uses (at most {MAX_CITATIONS_PER_LINE}), like [E012].
Reply with only the rewritten line.
LINE: {c['text']}
FACTS:
{fact_lines}"""
        try:
            raw = ask_writer(prompt, temperature=0.3)
        except Exception:
            continue
        line = next((l.strip() for l in raw.splitlines() if l.strip()), "")
        line = re.sub(r"^(?:HOOK\s*\d*\s*(?:\([^)]*\))?\s*:\s*)", "", strip_markdown(line)).strip().strip('"')
        line, _ = canonicalise_citation_ids(line, valid_ids)
        if not line_citations(line) or any(i not in valid_ids for i in line_citations(line)):
            line = remove_citation(line).strip() + " " + " ".join(f"[{i}]" for i in line_citations(c["text"]))
        if remove_citation(line).casefold() == remove_citation(c["text"]).casefold() or not remove_citation(line):
            continue
        if not HOOK_HARD_MIN_WORDS <= hook_word_count(line, game) <= HOOK_HARD_MAX_WORDS:
            continue
        variants.append({"number": c.get("number"), "approach": f"{c.get('approach', 'HOOK')}_CLARIFIED", "text": line,
                         "clarified_from": c["text"]})
    return variants


def build_hook_evaluation_prompt(game, shown):
    lines = "\n".join(f"HOOK {number}: {text}" for number, text in shown)
    return f"""You are a distracted viewer scrolling short videos. You have NEVER played "{game}" and you know nothing about its characters, places, factions or lore. Judge each hook as the first spoken sentence of a video. The narration plays over gameplay footage, so the viewer SEES the characters, creatures and places while hearing it. A character's NAME is therefore clear: hearing a name over a shot of the person it belongs to tells the viewer who that is, and a plain reference such as an abusive man, a lord or a creature is clear for the same reason. Only an abstract game term that footage cannot show (a faction, a title's meaning, a lore concept, an object never seen) counts as unexplained.
Score each hook 0-5 on:
Use the whole scale. Anchors: 5 = excellent, nothing to improve; 4 = strong; 3 = good enough to post; 2 = a clear weakness; 1 = fails at it; 0 = absent.
CLARITY: could a newcomer say who is involved and what is going on after hearing it once while watching the footage? Character and creature names are fine (the footage shows them); only an abstract game term the footage cannot show and the hook does not explain lowers this. The game's own title is just context: naming it is fine and never counts as unexplained. 5 = instantly clear; 3 = clear after a moment; 1 = confusing.
SPECIFICITY: does it contain at least one concrete, distinctive detail (a specific act, relationship, object or consequence) that makes this story unlike any other? 5 = vivid and unique; 3 = one concrete detail; 1 = abstract wording that could be about anything.
Any uncommon, formal or clinical word that many listeners would not instantly understand caps CLARITY at 3.
CURIOSITY: does it open an unanswered question without giving the ending away? 5 = you must know what happens; 3 = mildly intriguing; 1 = no question.
EMOTION: do you FEEL something for a specific person or creature (grief, guilt, fear, love, horror at what someone did) because of concrete human stakes in the hook (a parent and child, a betrayal, a life at risk), not because of drama words? 0 = nothing at stake for anyone; 5 = a personal stake you cannot ignore.
DELIVERY: does it sound natural when spoken aloud, in short easy clauses, 8-18 words?
Also list UNEXPLAINED: every abstract game term a newcomer could not understand from the hook plus the footage (or NONE); names of characters, creatures and places on screen are never UNEXPLAINED. Ordinary English words and phrases are never UNEXPLAINED, even if vague; vagueness lowers SPECIFICITY, not CLARITY. And ANY_GAME: YES if the hook could describe almost any game, otherwise NO.
Judge honestly against the anchors: do not reward drama or big words, and do not mark a good hook down just to be strict.
One line per hook, exactly this format (replace each <n> with your score 0-5), WHY at most 15 words:
HOOK 1: CLARITY=<n> SPECIFICITY=<n> CURIOSITY=<n> EMOTION=<n> DELIVERY=<n> UNEXPLAINED=<terms, or NONE> ANY_GAME=<YES or NO> WHY=<short reason>
HOOKS:
{lines}"""


_EVAL_PATTERN = re.compile(
    r"HOOK\s+(\d+)\s*:\s*CLARITY\s*=\s*([0-5])\s+SPECIFICITY\s*=\s*([0-5])\s+CURIOSITY\s*=\s*([0-5])(?:\s+EMOTION\s*=\s*([0-5]))?\s+DELIVERY\s*=\s*([0-5])"
    r"(?:\s+UNEXPLAINED\s*=\s*(.*?))?(?:\s+ANY_GAME\s*[=:]\s*(YES|NO))?(?:\s+WHY\s*[=:]\s*(.*))?\s*$",
    re.IGNORECASE)


def parse_hook_evaluation(raw):
    """Return {display_number: evaluation dict}. Malformed lines are skipped, never guessed."""
    rows = {}
    for line in raw.splitlines():
        match = _EVAL_PATTERN.search(re.sub(r"[*`]", "", line.strip()))
        if not match:
            continue
        unexplained_raw = (match.group(7) or "").strip()
        unexplained = [] if re.match(r"^(none|n/a|-)?\.?$", unexplained_raw, re.IGNORECASE) else [
            part.strip(" .") for part in re.split(r"[;,]", unexplained_raw) if part.strip(" .")]
        rows[int(match.group(1))] = {
            "clarity": int(match.group(2)), "specificity": int(match.group(3)),
            "curiosity": int(match.group(4)), "emotion": int(match.group(5)) if match.group(5) else 0,
            "delivery": int(match.group(6)),
            "unexplained": unexplained,
            "any_game": (match.group(8) or "NO").upper() == "YES",
            "why": (match.group(9) or "").strip(),
        }
    return rows


def evaluate_hooks(game, candidates):
    """Separate evaluator pass: sees ONLY the hook text (like a viewer), in shuffled order, without approach
    labels or evidence. Returns {candidate id: evaluation}. Missing rows get one retry."""
    pending = list(candidates)
    results = {}
    for round_number in range(2):
        if not pending:
            break
        order = pending[:]
        random.Random(f"{HOOK_SEED}-{round_number}" if HOOK_SEED else None).shuffle(order)
        shown = [(index, c["text"]) for index, c in enumerate(order, 1)]
        raw = ask_ollama(build_hook_evaluation_prompt(game, [(n, remove_citation(t)) for n, t in shown]),
                         temperature=HOOK_EVAL_TEMPERATURE, model=HOOK_EVAL_MODEL)
        rows = parse_hook_evaluation(raw)
        for index, candidate in enumerate(order, 1):
            if index in rows:
                results[candidate["id"]] = rows[index]
        pending = [c for c in pending if c["id"] not in results]
    return results


def score_hook(candidate, evaluation, game, passage_map):
    """100-point score with per-criterion points, penalties and a justification. Grounding is a hard gate
    (candidate['grounded']) applied separately; its points are only awarded to grounded hooks."""
    spoken = remove_citation(candidate["text"])
    words = hook_word_count(candidate["text"], game)
    ids = line_citations(candidate["text"])
    reference = " ".join(excerpt_of(passage_map[i]) for i in ids if i in passage_map)

    points = {}
    for key in ("clarity", "specificity", "curiosity", "emotion", "delivery"):
        points[key] = round(HOOK_WEIGHTS[key] * evaluation.get(key, 0) / 5, 1) if evaluation else 0.0
    points["grounding"] = float(HOOK_WEIGHTS["grounding"]) if candidate.get("grounded") else 0.0

    penalties = []
    names = hook_proper_nouns(candidate["text"], game, reference)
    title_text = " ".join({game.strip(), game_title_variants_raw(game)}).casefold()
    flagged = [n for n in (evaluation["unexplained"] if evaluation else [])
               if re.search(rf"(?<!\w){re.escape(n)}(?!\w)", spoken, re.IGNORECASE)
               and n.casefold() not in title_text]   # the game's own name is never an "unexplained" name
    # The judge's UNEXPLAINED list already lowered its CLARITY score; penalising the same terms again in code counted
    # them twice (one protagonist name cost a hook ~11 points). Code only adds its own structural floor: 3+ names.
    unexplained = []
    if candidate.get("needs_review"):
        penalties.append(("a claim needs human review (support uncertain)", PENALTY_NEEDS_REVIEW))
    if len(names) > 3:   # names are fine over footage, but four or more in one opening line is too many to follow
        unexplained += names[: len(names) - 3]
    candidate["unexplained_terms"] = flagged
    if unexplained:
        penalties.append((f"unexplained proper noun(s): {', '.join(unexplained)}",
                          min(PENALTY_NAME_CAP, PENALTY_UNEXPLAINED_NAME * len(unexplained))))
    vague = [p for p in HOOK_VAGUE_PHRASES if re.search(p, spoken, re.IGNORECASE)]
    if vague:
        found = [re.search(p, spoken, re.IGNORECASE).group(0) for p in vague]
        penalties.append((f"vague/generic phrasing: {', '.join(found)}", min(PENALTY_VAGUE_CAP, PENALTY_VAGUE_PHRASE * len(vague))))
    if evaluation and evaluation["any_game"]:
        penalties.append(("could apply to almost any game", PENALTY_GENERIC_ANY_GAME))
    if re.search(r"\b(is a game|is an? (open[- ]world|action|role[- ]playing)\b)", spoken, re.IGNORECASE):
        penalties.append(("redundant setup (describes the game instead of the situation)", PENALTY_REDUNDANT_SETUP))
    if spoken.count(",") > 2 or re.search(r"\b(who|which|that)\b.*\b(who|which|that)\b.*\b(who|which|that)\b", spoken, re.IGNORECASE):
        penalties.append(("long or confusing clause structure", PENALTY_CONFUSING_SYNTAX))
    if not HOOK_TARGET_MIN_WORDS - HOOK_LENGTH_GRACE <= words <= HOOK_TARGET_MAX_WORDS + HOOK_LENGTH_GRACE:
        penalties.append((f"{words} words is outside the {HOOK_TARGET_MIN_WORDS}-{HOOK_TARGET_MAX_WORDS} spoken target", PENALTY_OUT_OF_RANGE_LENGTH))
    content_words = {w.lower() for w in re.findall(r"[A-Za-z]{5,}", strip_game_title(spoken, game))
                     if w.lower() not in _NAME_STOPWORDS}
    if reference and len([w for w in content_words if re.search(rf"\b{re.escape(w)}", reference, re.IGNORECASE)]) < 2:
        penalties.append(("shares fewer than two distinctive words with its cited evidence", PENALTY_NOT_TIED_TO_EVIDENCE))

    total = max(0.0, round(sum(points.values()) - sum(p for _, p in penalties), 1))
    # ONLY a made-up claim rejects a hook. Style scores (clarity, specificity, total) rank hooks against each other and
    # are reported as notes; they never discard a verified hook. HOOK_STRICT=1 turns the old style floors back on.
    blockers, notes = [], []
    if not candidate.get("grounded"):
        blockers.append(f"not grounded: {candidate.get('audit_reason') or 'audit failed'}")
    if evaluation is not None:
        if evaluation["clarity"] < HOOK_MIN_CLARITY:
            notes.append(f"clarity {evaluation['clarity']}/5")
        if evaluation["specificity"] < HOOK_MIN_SPECIFICITY:
            notes.append(f"specificity {evaluation['specificity']}/5")
    if total < HOOK_MIN_TOTAL and candidate.get("grounded"):
        notes.append(f"total {total}/100")
    if HOOK_STRICT:
        blockers += [f"{n} is below the floor" for n in notes]
    return {
        "points": points, "penalties": [{"reason": r, "points": p} for r, p in penalties],
        "total": total, "eligible": not blockers, "blockers": blockers, "notes": notes,
        "justification": (evaluation or {}).get("why", ""), "words": words,
    }


def game_named_early(text, game, within=6):
    """True when the game title appears in the first few spoken words (game-first hook)."""
    tokens = re.findall(r"[\w'-]+", strip_game_title(remove_citation(text), game))
    return "TITLE" in tokens[:within]


def retention_rank(evaluation):
    if not evaluation:
        return 0
    """What the hook is chosen on, among hooks that already passed every gate (grounded, clear, specific, >= the total).
    Emotional stake counts double because it is the strongest reason to keep watching, but only together with an open
    question (curiosity) and instant understanding (clarity): a moving line that confuses or resolves itself in the
    first second is swiped away. Range 0-20."""
    return 2 * evaluation.get("emotion", 0) + evaluation.get("curiosity", 0) + evaluation.get("clarity", 0)


def _hook_rank_key(candidate, game):
    """Selection order among ELIGIBLE hooks: retention_rank (emotion x2 + curiosity + clarity), then the total score
    plus a small nudge for game-first hooks, then the shorter hook. Nothing here can select an ungrounded hook."""
    bonus = 2.0 if game_named_early(candidate["text"], game) else 0.0
    evaluation = candidate.get("evaluation")
    return (retention_rank(evaluation), candidate["score"]["total"] + bonus, -candidate["score"]["words"])


def _print_hook_table(pool):
    print("\n--- HOOK CANDIDATES ---")
    for c in pool:
        status = "ELIGIBLE" if c.get("eligible") else "rejected"
        total = c["score"]["total"] if c.get("score") else "-"
        emotion = (c.get("evaluation") or {}).get("emotion")
        print(f"[{status}] {c['approach']:<22} {total:>5}  {'E' + str(emotion) if emotion is not None else '  '}  {remove_citation(c['text'])}")
        for blocker in (c.get("score") or {}).get("blockers", []) or ([c["audit_reason"]] if c.get("audit_reason") else []):
            print(f"      - {blocker}")
        if c.get("eligible") and (c.get("score") or {}).get("notes"):
            print(f"      (style notes, not a rejection: {', '.join(c['score']['notes'])})")


def _write_hook_log(log_path, log):
    if not log_path:
        return
    try:
        Path(log_path).write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as error:
        print(f"WARNING: could not write hook log: {error}")


def generate_and_select_hook(game, topic, evidence, valid_ids, story_brief, log_path=None, allow_weak=False):
    """Return the single strongest eligible hook (see generate_and_select_hooks)."""
    return generate_and_select_hooks(game, topic, evidence, valid_ids, story_brief, log_path, allow_weak, top_n=1)[0]


def plain_rewrites(game, topic, pool, facts, evidence, valid_ids):
    """The strongest verified hooks, rewritten the way a storyteller would say them out loud: everyday words a
    listener understands instantly, a person and what is happening to them, the cause or twist only hinted at.
    Each rewrite is a NEW candidate: checked for invented facts, judged and scored like any other."""
    if HOOK_PLAIN_REWRITES <= 0:
        return []
    best = sorted([c for c in pool if c.get("grounded") and c.get("score")],
                  key=lambda c: _hook_rank_key(c, game), reverse=True)[:HOOK_PLAIN_REWRITES]
    fact_lines = "\n".join(f"[{f['id']}] {f['fact']}" for f in (facts or [])) or evidence
    known = {remove_citation(c["text"]).casefold() for c in pool}
    variants = []
    for c in best:
        prompt = f"""Rewrite this opening line of a short video about "{topic}" in the game "{game}" so that nobody could swipe away, the way a gifted storyteller would say it out loud.
- Every word must be understood instantly by anyone hearing it once: everyday spoken words only. Replace any clinical, formal or uncommon word with the plain one people actually say.
- Lead with a person and what is happening to them, or what they must do. Keep the cause, the trick or the twist hidden: hint that something is behind it, never explain it.
- Keep the facts it states. Add no new event, cause, outcome, name or number.
- One sentence, at most {HOOK_TARGET_MAX_WORDS + HOOK_LENGTH_GRACE} words, ending with the evidence id(s) it uses, like [E012].
Reply with only the new line.
LINE: {c['text']}
FACTS:
{fact_lines}"""
        try:
            raw = ask_writer(prompt, temperature=0.5, options={"top_p": 0.9})
        except Exception:
            continue
        line = next((l.strip() for l in raw.splitlines() if l.strip()), "")
        line = re.sub(r"^(?:HOOK\s*\d*\s*(?:\([^)]*\))?\s*:\s*)", "", strip_markdown(line)).strip().strip('"')
        line, _ = canonicalise_citation_ids(line, valid_ids)
        if not line_citations(line) or any(i not in valid_ids for i in line_citations(line)):
            line = remove_citation(line).strip() + " " + " ".join(f"[{i}]" for i in line_citations(c["text"]))
        spoken = remove_citation(line).strip()
        if not spoken or spoken.casefold() in known:
            continue
        if not HOOK_HARD_MIN_WORDS <= hook_word_count(line, game) <= HOOK_HARD_MAX_WORDS:
            continue
        known.add(spoken.casefold())
        variants.append({"approach": f"{c.get('approach', 'HOOK')}_PLAIN", "text": line, "rewritten_from": c["text"]})
    return variants


def generate_and_select_hooks(game, topic, evidence, valid_ids, story_brief, log_path=None, allow_weak=False, top_n=1,
                              facts=None, dropped_facts=None):
    """Generate candidate hooks, gate them on grounding, score them with a separate evaluator pass, and return up to
    top_n of the strongest ELIGIBLE ones (best first, preferring different approaches so each can become its own
    story angle). Raises HookSelectionError rather than returning an unverified or weak hook."""
    passage_map = {item["id"]: item["text"] for item in split_evidence_passages(evidence)}
    log = {"game": game, "topic": topic, "generator_model": MODEL, "evaluator_model": HOOK_EVAL_MODEL,
           "weights": HOOK_WEIGHTS, "min_total": HOOK_MIN_TOTAL, "attempts": [], "selected": None, "warning": None}
    pool, seen_text, feedback = [], set(), ""
    try:
        if facts is None:
            try:
                facts, dropped_facts = extract_hook_facts(game, topic, evidence, valid_ids, story_brief)
            except Exception as error:   # fact extraction is an aid; hooks are still fully audited without it
                facts, dropped_facts = [], [f"fact extraction failed: {error}"]
        dropped_facts = dropped_facts or []
        if len(facts) < 3:
            print(f"Only {len(facts)} verified fact(s) extracted; hooks will be written from the full evidence instead.")
            facts = []
        else:
            print("\n--- VERIFIED FACTS ---\n" + "\n".join(f"[{f['id']}] ({f['kind']}) {f['fact']}" for f in facts))
        log["facts"], log["dropped_facts"] = facts, dropped_facts
        for attempt in range(1, HOOK_GEN_ATTEMPTS + 1):
            temperature = HOOK_GEN_TEMPERATURE if attempt == 1 else max(0.4, HOOK_GEN_TEMPERATURE - 0.2)
            raw = ask_writer(build_hook_prompt(game, topic, evidence, valid_ids, story_brief, feedback, facts),
                             temperature=temperature, options={"top_p": HOOK_GEN_TOP_P})
            parsed, parse_reasons, skipped = parse_hook_candidates(raw, valid_ids, game)
            for candidate in parsed:
                recited = recite_from_facts(candidate["text"], facts, game)
                if recited != candidate["text"]:
                    candidate["model_citation"] = candidate["text"]
                    candidate["text"] = recited
            entry = {"attempt": attempt, "format_rejections": parse_reasons, "approaches_without_support": skipped,
                     "raw_output": raw[:4000]}
            fresh = []
            for candidate in parsed:
                key = remove_citation(candidate["text"]).casefold()
                if key in seen_text:
                    continue
                seen_text.add(key)
                candidate["id"] = len(pool) + len(fresh) + 1
                candidate["attempt"] = attempt
                fresh.append(candidate)
            audit_hook_candidates(fresh, evidence, game)
            grounded = [c for c in fresh if c["grounded"]]
            evaluations, eval_failed = {}, False
            if grounded:
                evaluations = evaluate_hooks(game, grounded)
                eval_failed = not evaluations
            for candidate in fresh:
                evaluation = evaluations.get(candidate["id"])
                candidate["evaluation"] = evaluation
                candidate["score"] = score_hook(candidate, evaluation, game, passage_map)
                candidate["eligible"] = candidate["score"]["eligible"]
            # rewrite unclear-but-promising hooks when they could beat the best eligible hook once clear (or when
            # nothing is eligible yet): a strong premise should not lose to a weaker one over a single name
            best_rank = max((retention_rank(c["evaluation"]) for c in pool + fresh if c.get("eligible")), default=-1)
            variants = clarify_hooks(game, topic, fresh, facts, evidence, valid_ids, beat_rank=best_rank)
            if variants:
                for v in variants:
                    v["text"] = recite_from_facts(v["text"], facts, game)
                    v["id"] = len(pool) + len(fresh) + 1
                    v["attempt"] = attempt
                    seen_text.add(remove_citation(v["text"]).casefold())
                    fresh.append(v)
                audit_hook_candidates(variants, evidence, game)
                grounded_variants = [v for v in variants if v["grounded"]]
                variant_evals = evaluate_hooks(game, grounded_variants) if grounded_variants else {}
                for v in variants:
                    v["evaluation"] = variant_evals.get(v["id"])
                    v["score"] = score_hook(v, v["evaluation"], game, passage_map)
                    v["eligible"] = v["score"]["eligible"]
                print(f"  clarified {len(variants)} promising hook(s) whose only problem was an unexplained name or term")
            elif best_rank >= 0 and any(c.get("unexplained_terms") and not c.get("eligible") for c in fresh):
                print("  unclear hooks were not rewritten: even made clear, none could outrank the best eligible hook "
                      "(emotion x2 + curiosity + clarity)")
            pool.extend(fresh)
            entry["candidates"] = [{k: v for k, v in c.items()} for c in fresh]
            entry["evaluation_failed"] = eval_failed
            log["attempts"].append(entry)

            print(f"\nHook round {attempt}/{HOOK_GEN_ATTEMPTS}: {len(parsed)} well-formed, "
                  f"{len(grounded)} grounded, {sum(1 for c in fresh if c['eligible'])} eligible.")
            for reason in parse_reasons:
                print("  format:", reason)
            for note in skipped:
                print("  evidence cannot support:", note)
            if eval_failed:
                print("  WARNING: the evaluator returned no usable scores for this round.")
            if sum(1 for c in pool if c["eligible"]) >= HOOK_MIN_ELIGIBLE:
                break
            problems = parse_reasons + [f"{remove_citation(c['text'])} -> {b}" + (f" (judge: {c['evaluation']['why']})" if c.get("evaluation") and c["evaluation"].get("why") else "")
                                         for c in fresh for b in c["score"]["blockers"]]
            feedback = "\n".join(f"- {p}" for p in problems[:12])
            near = max((c for c in pool if c.get("grounded") and c.get("score")), key=lambda c: c["score"]["total"], default=None)
            if near:
                feedback += (f"\nCLOSEST SO FAR ({near['score']['total']}/100): {remove_citation(near['text'])} "
                             "- keep what works, make it more concrete (who, what exactly is at stake) and explain any game-specific term in plain words.")

        try:   # the strongest hooks, retold in plain, gripping words, compete as new candidates
            plain = plain_rewrites(game, topic, pool, facts, evidence, valid_ids)
        except Exception as error:
            plain = []
            print(f"  plain-words rewrites skipped: {error}")
        if plain:
            for v in plain:
                v["id"] = len(pool) + 1 + plain.index(v)
                v["attempt"] = "plain"
            audit_hook_candidates(plain, evidence, game)
            grounded_plain = [v for v in plain if v["grounded"]]
            plain_evals = evaluate_hooks(game, grounded_plain) if grounded_plain else {}
            for v in plain:
                v["evaluation"] = plain_evals.get(v["id"])
                v["score"] = score_hook(v, v["evaluation"], game, passage_map)
                v["eligible"] = v["score"]["eligible"]
            pool.extend(plain)
            print(f"  rewrote the {len(plain)} strongest hook(s) in plain spoken words; they compete as new candidates")
        _print_hook_table(pool)
        eligible = [c for c in pool if c["eligible"]]
        if eligible:
            # strongest eligible, not first and not "most dramatic"; extra picks favour a different approach
            ranked = sorted(eligible, key=lambda c: _hook_rank_key(c, game), reverse=True)
            picks, used = [], set()
            for candidate in ranked:
                if candidate["approach"] not in used:
                    picks.append(candidate)
                    used.add(candidate["approach"])
            picks += [c for c in ranked if c not in picks]
            picks = picks[:max(1, top_n)]
            chosen = picks[0]
            log["selected"] = {"id": chosen["id"], "text": chosen["text"], "approach": chosen["approach"],
                               "total": chosen["score"]["total"], "justification": chosen["score"]["justification"],
                               "game_first": game_named_early(chosen["text"], game),
                               "emotion": (chosen.get("evaluation") or {}).get("emotion"), "retention_rank": retention_rank(chosen.get("evaluation"))}
            log["also_selected"] = [{"id": c["id"], "text": c["text"], "approach": c["approach"], "total": c["score"]["total"]}
                                    for c in picks[1:]]
            for position, candidate in enumerate(picks, 1):
                print(f"\nSelected hook {position} ({candidate['approach']}, {candidate['score']['total']}/100, emotion "
                      f"{(candidate.get('evaluation') or {}).get('emotion')}/5, retention rank {retention_rank(candidate.get('evaluation'))}/20): {candidate['text']}")
                if candidate["score"]["justification"]:
                    print("  why:", candidate["score"]["justification"])
            return [c["text"] for c in picks]

        grounded_pool = [c for c in pool if c.get("grounded")]
        scored = [c for c in grounded_pool if c.get("evaluation")]
        best = (max(scored, key=lambda c: (retention_rank(c["evaluation"]), c["score"]["total"])) if scored
                else max(grounded_pool, key=lambda c: c["score"]["total"], default=None))
        if not grounded_pool:
            warning = (f"HOOK WARNING: none of the {len(pool)} candidate hooks could be verified against the evidence. "
                       "That usually means the model added details or cited the wrong passage, not that the evidence is "
                       "thin. Try again, or set HOOK_EVAL_MODEL / OLLAMA_MODEL to a stronger model. No hook was selected.")
        else:
            warning = (f"HOOK WARNING: only {len(grounded_pool)} of {len(pool)} candidates were grounded and none reached {HOOK_MIN_TOTAL}/100 "
                       f"with clear stakes for a newcomer. Best was {best['score']['total']}/100: "
                       f"{remove_citation(best['text'])} ({'; '.join(best['score']['blockers'])}). "
                       "Its facts are verified; only the style floors were missed.")
        log["warning"] = warning
        if allow_weak and best is not None:
            log["selected"] = {"id": best["id"], "text": best["text"], "approach": best["approach"],
                               "total": best["score"]["total"], "override": "allow-weak-hook"}
            print("\n" + warning + "\nUsing the best VERIFIED candidate anyway (highest emotion x2 + curiosity + clarity); "
                  "use --strict-hook to stop instead.")
            return [best["text"]]
        raise HookSelectionError(warning)
    finally:
        log["candidates"] = [{k: c.get(k) for k in ("id", "attempt", "approach", "text", "grounded", "audit_reason",
                                                    "evaluation", "score")} for c in pool]
        _write_hook_log(log_path, log)



def _viewer_question(story_brief):
    """Only the brief's viewer question and payoff reach the writer: the rest of the brief is planning noise."""
    out = ""
    for field in ("WHAT_VIEWERS_WILL_WANT_NEXT", "PAYOFF_THE_VIDEO_CAN_DELIVER"):
        match = re.search(rf"(?mi)^\s*{field}\s*:\s*(.+)$", story_brief or "")
        if match:
            label = "THE QUESTION THE VIEWER WANTS ANSWERED" if field.startswith("WHAT") else "WHERE IT LANDS"
            out += f"{label}: {match.group(1).strip()}\n"
    return out


STYLE_EXAMPLES = _env_number("STYLE_EXAMPLES", 2)   # how many style examples the writer and editor are shown (0 = none)
STYLE_LESSONS = """WHAT TO TAKE FROM IT (craft only):
- The hook already holds a twist that the rest of the script pays off.
- The cause is a chain of plain, concrete actions told in the order they happened: who did what, then what that led to.
- One very short line lands the turn before the rule or the reveal.
- The rule of the world is stated once, in plain words.
- The choice is told as two short, mirrored consequences for the player.
- The ending question turns the dilemma back on a person in the story."""


_TOPIC_STOPWORDS = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "from", "with", "by", "at", "is", "was", "are",
                    "were", "be", "how", "why", "what", "who", "whom", "when", "where", "which", "that", "this", "his", "her",
                    "their", "its", "own", "into", "became", "become", "really", "actually", "ever", "one", "can", "cannot",
                    "makes", "made", "nobody", "everyone", "entire", "whole", "behind", "hiding", "hidden", "secret", "once",
                    "greatest", "most", "man", "woman", "s"}


def _topic_key(text):
    """The words that say what a topic is ABOUT (names, creatures, places), without question and filler words, so
    "How the Snow Elves became the Falmer" and "Why Mohg stole Miquella" are not treated as the same topic."""
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower().replace("'", "")) if w not in _TOPIC_STOPWORDS}


def load_style_examples(topic, limit=None, game=""):
    """Finished scripts kept in style_examples/. An example about the same topic is skipped: the writer would copy
    its facts instead of telling this evidence's story. Proven examples (a RESULT: line from a published video) come
    first; the rest favour OTHER games (their facts cannot leak into this story) and rotate with the topic, so
    different topics learn from different examples. Returns [(topic, text)]."""
    import hashlib
    limit = STYLE_EXAMPLES if limit is None else limit
    if limit <= 0 or not STYLE_DIR.exists():
        return []
    ranked = []
    for path in STYLE_DIR.glob("*.txt"):
        try:
            text = path.read_text(encoding="utf-8-sig")
        except OSError:
            continue
        example_topic = get_metadata(text, "TOPIC") or path.stem
        if _topic_key(example_topic) & _topic_key(topic):
            continue
        body = "\n".join(line for line in text.splitlines()
                         if line.strip() and not re.match(r"^(GAME|TOPIC|RESULT|NOTE)\s*:", line.strip(), re.I))
        if not body:
            continue
        proven = bool(get_metadata(text, "RESULT"))
        other_game = bool(game) and _normalised_title(get_metadata(text, "GAME") or "") != _normalised_title(game)
        spin = hashlib.sha1(f"{topic}|{path.name}".lower().encode("utf-8")).hexdigest()
        ranked.append(((not proven, not other_game, spin), example_topic, body.strip()))
    return [(t, b) for _, t, b in sorted(ranked)[:limit]]


def style_example_block(topic, game=""):
    examples = load_style_examples(topic, game=game)
    if not examples:
        return ""
    shown = "\n\n".join(f"(about {t}, a DIFFERENT story)\n{body}" for t, body in examples)
    return (f"STYLE EXAMPLE{'S' if len(examples) > 1 else ''}: finished script{'s' if len(examples) > 1 else ''} that viewers "
            f"watched to the end. Match this level of storytelling. Copy the craft, never the content: none of its characters, "
            f"events, images or phrases may appear in your script (the game and its hero may, when YOUR material has "
            f"them).\n{shown}\n{STYLE_LESSONS}\n")


def style_hook_block(topic, game=""):
    """The hooks of the style examples, for the hook writer: what an opening that kept viewers looks like."""
    hooks = []
    for _, body in load_style_examples(topic, game=game):
        match = re.search(r"(?mi)^\s*(?:HOOK\s*:|\[hook\])\s*(.+)$", body)
        if match:
            hooks.append(remove_citation(match.group(1)).strip())
    if not hooks:
        return ""
    return ("HOOKS THAT KEPT VIEWERS WATCHING (from DIFFERENT stories: learn the shape, never reuse their people, events or "
            "wording):\n" + "\n".join(f"- {h}" for h in hooks) +
            "\nWhy they work: a named person or a relationship with something to lose, one concrete act, and a twist that "
            "reframes it, while the ending stays hidden. They never say \"the person\", \"the creature\" or \"the monster\" "
            "when a name or a relationship hits harder, and they never reveal how the story is resolved.\n\n")


def style_echoes(draft, topic, size=5, game=""):
    """Five-word runs a draft shares with a style example (a sign it copied wording instead of craft)."""
    def runs(text):
        words = re.findall(r"[a-z']+", text.lower())
        return {" ".join(words[i:i + size]) for i in range(len(words) - size + 1)}
    mine = runs(clean_review_draft(draft))
    shared = set()
    for _, body in load_style_examples(topic, game=game):
        shared |= mine & runs(re.sub(r"\[[a-z]+\]", " ", body))
    return sorted(shared)


def full_story_block(full_story):
    if not full_story:
        return ""
    return (f"THE FULL STORY, ALREADY WRITTEN FROM THE MATERIAL (your job now is to tell it as the voice-over in AT MOST "
            f"{SCRIPT_TARGET_MAX_WORDS - 10} spoken words, counting the hook; it is about twice that long now, so cut about half): keep every key point (the cause, every option the player can choose with what it really "
            "involves and what it leads to or costs, and any twist that explains it), keep its best lines, and cut "
            "background, repetition and side details. Facts not in the material above stay out.\n" + full_story.strip() + "\n")


def build_full_story_prompt(game, topic, selected_hook, material, must_tell, angle=""):
    points = "\n".join(f"- {f['fact']}" for f in must_tell) or "- (see the material)"
    return f"""You are a gifted storyteller. Tell the WHOLE story behind this opening line about "{topic}" in the game "{game}", from beginning to end, the way you would tell it to a friend: about 250-350 words of flowing prose. This is a first draft that will be cut down afterwards, so leave nothing important out:
- who is suffering or in danger, and why it matters,
- how it came to be: the causes in the order they happened, and who was responsible,
- every option the player (as the main character) can choose: what each one actually involves (the real scene: who does what, to whom) and what each leads to or costs,
- any twist or hidden detail that explains why something works, told plainly (if something only LOOKS terrible, say what really happened).
Use only facts from the material: invent no event, cause, outcome, name or number. Everyday spoken words. No headings, no lists, no citations.
{("THIS VERSION: " + angle + chr(10)) if angle else ""}OPENING LINE: {remove_citation(selected_hook)}
KEY POINTS (every one must be in the story):
{points}
MATERIAL:
{material}"""


def is_key_payoff(fact):
    """What you actually do on a path, what it costs, and the reveal: a payoff without them is not the story."""
    return fact.get("beat") in ("risk", "twist") or fact.get("kind") == "choice"


def payoff_points(facts):
    """How each choice plays out (the scene, its cost, the twist): the points a payoff must tell."""
    return [f for f in (facts or []) if f.get("payoff_detail")][:PAYOFF_POINTS_MAX]


def build_generation_prompt(game, topic, evidence, valid_ids, selected_hook, story_brief, feedback="", essential=None, chain="", angle="", full_story=""):
    """STORY FIRST. The writer is a storyteller, not a fact-checker: it writes flowing narration with no citations and
    only one evidence rule, invent nothing. Citations are attached in code afterwards and every line is audited, so the
    evidence system only removes invented facts instead of shaping the prose."""
    title_note = ""
    if not hook_identifies_game(selected_hook, game):
        title_note = (f'The HOOK does not name the game, so the FIRST NARRATION line MUST contain the exact words '
                      f'"{game_title_variants_raw(game)}", woven in naturally (for example "In {game_title_variants_raw(game)}, ...").\n')
    must_include = ""
    if essential:
        must_include = ("THE STORY MUST INCLUDE these points (what caused the situation and how each choice plays out; never softened or "
                        "left out, because leaving them out would mislead the viewer). Work them into the story where they hit hardest, in your own words:\n"
                        + "\n".join(f"- {f['fact']}" for f in essential) + "\n")
    avoid = ", ".join(OVERUSED_WORDS)
    return f"""You are a gifted short-form storyteller writing the voice-over for a 30-60 second gaming-lore video. Your job is to make a stranger who has never played "{game}" unable to swipe away: a story with momentum, tension and a payoff, told like someone leaning in to tell a friend something unbelievable that really happened.
THE ONE EVIDENCE RULE: every FACT you state (an event, who did what, a cause, a motive, an outcome, a number) must come from the STORY MATERIAL below. Invent no facts. Everything else is yours: the order you reveal things, transitions, rhythm, emphasis, emotional framing, rhetorical turns, and plain descriptive words that fairly sum up what the material describes (calling a hideous, hostile creature "monstrous" is fine). If two accounts of the cause disagree, tell both and say whose each is; that tension is part of the story. Never invent a link between two events that the material does not state.
THE SHAPE (two halves with different jobs):
- NARRATION lines 1-{STORY_LINES}: STORYTELLING. Pull the viewer in: the danger, who is suffering, and why it happened to them (the cause), told vividly, in short lines (about {STORY_LINE_WORDS} words each). Keep them light on detail; their job is to make the viewer need what comes next.
- NARRATION line {STORY_LINES + 1} to the PAYOFF: PRECISE. Now tell the choice exactly as the material gives it: for each path, what you actually do, what it costs, and in the end the reveal, with the concrete details (who, what, where). No vague summaries here: this is where the facts carry the video.
- ENDING QUESTION: storytelling again.
HOW TO TELL IT:
- Start where the HOOK stops and pull the viewer forward. Each line should make the next one necessary: use turns such as but, so, until, and then, which is why, except.
- THE FIRST NARRATION LINE decides whether the viewer stays. Its only job is to make the hook's promise bigger: the danger, what this does to its victims, who is suffering and what they stand to lose, told vividly and concretely (words like terrifying or horrifying are fine here, once, when the material shows something that is). Never open the narration with background: not when or how something first appeared in the world, not what kind of being it is, not world history. The game title goes into this line naturally.
- Then the story: how it came to be (the cause, including who was responsible) in the order it happened, the turn or the darker truth, the choice, and the payoff the hook promised. Evidence-heavy detail belongs in the middle, once the viewer is hooked. Hold the most surprising fact back for the PAYOFF.
- THE WHY: the viewer must understand why this happened to THIS person. When the material gives the rule that links the cause to the situation (what the threat feeds on, whom it chooses), connect them in plain words: the event, what it left in the person, and why that drew the threat.
- THE RULE MUST FIT THE PERSON: if the rule names a kind of person (the guilty, the cruel, oath-breakers) and what this person did does not obviously make them one (an accident, a mistake), say the bridge the material gives (what they believe about themselves, what they blame themselves for). A rule about crimes next to an accident, with no bridge, confuses the viewer.
- PLANT WHAT THE TWIST NEEDS: if a path works by using that same rule (fooling it, feeding it, turning it on someone else), state the rule early in the form the twist will use, so the reveal clicks instead of coming from nowhere.
- CONCRETE PEOPLE, NOT GENERIC RULES: once the viewer is hooked, tell what the characters in this story actually do, with their names (who does what, to whom, and why), never a generic stand-in such as "a trickster", "a person", "someone" or "the victim" when the material says who it is. A general rule of the world is fine only to explain why what they do works.
- PLAIN WORDS: say what happened in everyday words a friend would use. Never swap a concrete act for a clinical, legal or academic label (say "he killed his brother", never "he committed fratricide"; say "a spirit", never "an entity"). If the payoff is a trick or a twist, say plainly how it works and what the viewer would see happen: who is fooled into believing what, what that belief causes, and what happens when the truth comes out.
- SAVE THE REVEAL FOR THE END: if a path has a twist (something only looked terrible, something was secretly arranged), tell that path last, its steps in order, and end the PAYOFF on the reveal and what it leads to. Never give the twist away before the moment it explains.
- ONE POINT OF VIEW: when the player character acts on a player choice, say "you" every time; never switch between "you" and the character's name for the same actions.
- A PATH'S COST is what is at stake for people (who may die, be hurt, or lose something). Never use an incidental detail (how long it takes, what is fought off along the way) as the cost.
- ONE PATH, ONE SEQUENCE: tell the steps of a path in the order they happen (first this, then that, and in the end this). A step of the plan is part of the plan, never a failure case or an "if it goes wrong", unless the material says it can fail that way.
- EACH OPTION IS A DIFFERENT PATH: say what the player does differently on each (for example accept or refuse). The steps and results of one path belong together in that path; never present two steps of one path as two options, and never offer the same option twice.
- A STORY, NOT A MANUAL: tell the choice as a moment with something at stake, not as instructions. At most ONE sentence in the whole script may start with "If you" (say "You could...", "Or you...", "But there is another way..." instead), and never stack conditions ("if you do X, then Y, but if you do Z...").
- KEEP A QUESTION OPEN: every line should leave the viewer wanting the next one: raise a question (who, why, what it costs) a line before you answer it, and hold the most surprising fact back until the PAYOFF.
- THE SUBJECT IS NAMED: the person or creature the video is about ("{topic}") is named in the first two NARRATION lines, with what they are in a few words; never only "a being", "a stranger" or "the demon" when the material gives the name.
- NAME BUDGET: at most three names besides the video's subject (people, creatures, places; the game's title does not count). Everyone else is called by what they are to the story, using the role the WHO IS WHO list or the material gives ("the jarl's daughter", "his wife", "the witches"); never a bare "the girl" or "the man" that tells the viewer nothing. A game term the footage cannot show (a type of creature, a group, a title) is explained in a few plain words the first time, or replaced by those words.
- THE PAYOFF MAKES EACH OPTION CONCRETE: for every option the player can choose, say what is actually done (the real act, who does it, to whom), why it would work, and what it leads to or costs, using the details in the material. A path whose cost the material gives is never told without that cost. Naming a method without its scene ("use a trick", "fight the creature") fails. The cost of each option is what makes the ending question land, so the question should weigh those costs.
- Every line adds NEW information or raises the stakes; never re-say what the hook or an earlier line said, never state the same fact twice. Never spend a line announcing that a choice is coming ("you must decide", "the choice is yours"): go straight into the first path.
- Make it felt: concrete, specific details from the material, people's relationships and what they lose. Feeling comes from stakes, not from stacking adjectives. Avoid these overused words, at most once each: {avoid}.
- Spoken rhythm: mostly short sentences, a longer one now and then, nothing that trips the tongue, no sentence over 25 words. One or two sentences per line; the PAYOFF may use three.
- This is a GAME: when the material says what happens IF the player (as the main character) makes a choice, tell it as the player's choice and its result ("as the hero you can..., and if you do..."), never as something that simply happened. Get WHO does each action exactly right.
- WHO THEY ARE: the footage shows what a character looks like, not who they are to the story. The first time a person is named, say what they are to it in a few words (his brother, the woman who knows the cure, the man who hired you), using ONLY a relationship or role the material states; a made-up relationship is an invented fact. If the material gives none, say what they do in this story instead. Someone who only does one thing needs no name: say what they do in this story instead. Never bring a new name into the PAYOFF or the ENDING QUESTION. Explain an abstract game term the footage cannot show in a few plain words the first time it appears.
NAMES: one name per person. If someone has both a name and a title, introduce them once together (name, then title) and then always use the same one. Never let a newcomer think one person is two.
- ENDING QUESTION: never a prediction ("will it work?", "can you pull it off?"); one specific question about THIS story that makes the viewer take a side, ideally weighing what each path costs or whether a person deserves what they get. It must be a real dilemma: two defensible answers, both visible in the script (a choice and what it cost, a consequence and what it spared, an action and its alternative). Never ask how a character feels or reacts, never ask whether a good thing is good, never ask a generic question such as "What other creatures in {game} are like this?". At most 18 words, and it must not repeat the PAYOFF.
LENGTH: {SCRIPT_TARGET_MIN_WORDS}-{SCRIPT_TARGET_MAX_WORDS} spoken words (hard limits {MIN_WORDS}-{MAX_WORDS}). Use the room the story needs; never pad.
{style_example_block(topic, game)}GAME: {game}
TOPIC: {topic}
{_viewer_question(story_brief)}SELECTED HOOK: {selected_hook}
{("STORY MATERIAL:" + chr(10) + chain + chr(10)) if chain else ("EVIDENCE:" + chr(10) + evidence + chr(10))}{full_story_block(full_story)}{title_note}{must_include}{("THIS VERSION: " + angle + chr(10)) if angle else ""}FEEDBACK FROM PREVIOUS ATTEMPT:
{feedback or 'None'}
Output plain text in exactly this layout (three to seven NARRATION lines; no citations, no markdown, no quotation marks):
STATUS: DRAFT
HOOK: {selected_hook}
NARRATION: ...
NARRATION: ...
NARRATION: ...
PAYOFF: ...
ENDING QUESTION: ...
CLAIMS REQUIRING MORE EVIDENCE: None"""


HOSTED_MAX_ATTEMPTS = _env_number("HOSTED_MAX_ATTEMPTS", 2)
STORY_LINES = _env_number("STORY_LINES", 3)            # narration lines that hook and set up the story before the precise part
STORY_LINE_WORDS = _env_number("STORY_LINE_WORDS", 18)


def untold_costs(draft, points, game=""):
    """A path's cost checked in code (the model judge too often says a cost is told when it is not): the cost counts
    as told only if a line about that path (sharing a word with what you do on it) says something is at risk
    (may not, die, survive, risk, cost...). Returns the cost facts no such line tells."""
    lines = [remove_citation(c) for sec, c in spoken_lines(draft) if sec in ("NARRATION", "PAYOFF")]
    untold = []
    for cost in [f for f in points or [] if f.get("beat") == "risk"]:
        path = [f for f in points if _option_key(f.get("option")) == _option_key(cost.get("option")) and f is not cost]
        words = set().union(*(content_stems(f["fact"], game) for f in path)) if path else set()
        words |= content_stems(cost.get("option") or "", game)
        if not any(re.search(_RISK_WORDS, line, re.I) and (content_stems(line, game) & words or not words) for line in lines):
            untold.append(cost)
    return untold


def precise_part(draft):
    """The part of the script that must carry the choice exactly: NARRATION line STORY_LINES+1 onwards and the PAYOFF.
    (A script too short to have that part is judged as a whole.)"""
    lines = spoken_lines(draft)
    narration = [i for i, (sec, _) in enumerate(lines) if sec == "NARRATION"]
    if len(narration) <= STORY_LINES:
        return draft
    keep = [l for i, l in enumerate(lines) if (l[0] == "NARRATION" and i in narration[STORY_LINES:]) or l[0] == "PAYOFF"]
    return rebuild_draft(draft, keep)   # a strong hosted writer rarely needs more; free limits are tight


def hosted_writer_active():
    """True when a free hosted writer (Gemini or Groq) can take the writing steps this run."""
    return bool(writer_chain())


def build_hosted_prompt(game, topic, selected_hook, material, must_tell, feedback="", angle=""):
    """LEAN MODE. A strong writer gets the story plan and a short brief, and writes the script in one pass. (The long
    prompt in build_generation_prompt exists to steer the small local model; a strong model does better with less, and
    a short prompt fits the free tiers' size and per-minute limits.)"""
    points = "\n".join(f"- {f['fact']}" for f in must_tell)
    title = "" if hook_identifies_game(selected_hook, game) else f'The first NARRATION line names the game: "{game_title_variants_raw(game)}".\n'
    examples = load_style_examples(topic, limit=1, game=game)
    example = (f"A FINISHED SCRIPT THAT WORKED (a different story; copy its craft, never its content):\n{examples[0][1]}\n"
               if examples else "")
    return f"""You are a gifted short-form storyteller. Write the voice-over for a 30-60 second YouTube Short about "{topic}" in the game "{game}", for viewers who have never played it. It must make a stranger unable to swipe away.
THE ONE HARD RULE: every fact (event, who did what, cause, motive, outcome, name, relationship, number) comes from the PLAN below. Invent nothing, not even a relationship or a role. Everything else is yours: order, rhythm, framing, plain descriptive words.
WHAT MAKES IT WORK:
- Start where the hook stops. The first narration line makes the danger bigger and concrete; never open on background.
- Two halves: NARRATION lines 1-{STORY_LINES} are storytelling (the danger, who suffers, why it happened; short and vivid). From NARRATION line {STORY_LINES + 1} to the PAYOFF it is precise: each path exactly as the plan gives it, with its cost, then the reveal.
- Tell the cause in the order it happened and make the WHY clear: why this happened to THIS person. If the rule (whom the threat chooses) does not obviously fit what the person did, give the bridge the plan states.
- Every line leads to the next (but, so, until, which is why). No line repeats another or only announces that a choice is coming.
- The choice: say "you" for the player character. Each path is what you actually do, why it would work, and what it costs people. Tell the path with a twist last and end the PAYOFF on the reveal; never hint at it earlier.
- The first time a person is named, say who they are to the story using only what the plan states; someone who only does one thing gets no name, just what they do. No new names in the PAYOFF or the ENDING QUESTION.
- Plain spoken words a friend would use, short sentences (none over 25 words), no stacked adjectives.
- ENDING QUESTION: asked from the moment of choosing (the viewer now knows the twist, the player did not), weighing both paths and what each costs, with two defensible answers. At most 20 words.
LENGTH: {SCRIPT_TARGET_MIN_WORDS}-{SCRIPT_TARGET_MAX_WORDS} spoken words in total.
{title}{("THIS VERSION: " + angle + chr(10)) if angle else ""}HOOK (fixed, copy it exactly): {selected_hook}
PLAN (verified facts in telling order, with the source words):
{material}
{("IT MUST TELL:" + chr(10) + points + chr(10)) if points else ""}{example}{("FIX FROM THE LAST ATTEMPT:" + chr(10) + feedback + chr(10)) if feedback else ""}Output only this layout (three to seven NARRATION lines; no citations, no markdown, no quotation marks):
STATUS: DRAFT
HOOK: {selected_hook}
NARRATION: ...
PAYOFF: ...
ENDING QUESTION: ...
CLAIMS REQUIRING MORE EVIDENCE: None"""


def attach_citations(draft, passage_map, facts, game, valid_ids):
    """The writer does not cite; code proposes the passages each line most likely draws on (the verified facts it
    echoes first, then the passages it shares the most words with). The audit then confirms every claim against those
    passages, searches the others, and corrects the citation, so a wrong guess here costs nothing but a re-cite."""
    lines = spoken_lines(draft)
    if not lines:
        return draft
    changed = False
    for index, (section, content) in enumerate(lines):
        if section == "HOOK" or line_citations(content):
            continue
        stems = content_stems(content, game)
        ranked = []
        for fact in facts or []:
            shared = len(stems & content_stems(fact["fact"], game))
            if shared >= 2:
                for evidence_id in fact.get("ids", [fact["id"]]):
                    ranked.append((shared + 1, evidence_id))
        for evidence_id, text in passage_map.items():
            ranked.append((len(stems & content_stems(excerpt_of(text), game)), evidence_id))
        ids = []
        for score, evidence_id in sorted(ranked, key=lambda r: -r[0]):
            if score >= 1 and evidence_id in valid_ids and evidence_id not in ids:
                ids.append(evidence_id)
            if len(ids) >= MAX_CITATIONS_PER_LINE:
                break
        if not ids:
            ids = [next(iter(sorted(valid_ids, key=lambda v: int(v[1:]))))]
        lines[index] = (section, recite_line(content, ids))
        changed = True
    return rebuild_draft(draft, lines) if changed else draft


def clean_review_line(line):
    # Keep underscores INSIDE labels (RETENTION_VALUATION). Only strip emphasis markers and wrapping underscores.
    line = re.sub(r"[*`]", "", line.strip())
    line = re.sub(r"^\s*[-#>]+\s*", "", line).strip("_ \t")
    return re.sub(r"\s+", " ", line).strip()


MAX_CITATIONS_PER_LINE = max(1, _env_number("MAX_CITATIONS_PER_LINE", 3))   # a causal line may need cause + effect + link
MAX_REVIEW_LINES = _env_number("MAX_REVIEW_LINES", 1)   # lines left for a human to check (NEEDS_REVIEW) before a script is blocked
MAX_REPAIR_ROUNDS = 3
MAX_SENTENCES_PER_LINE = _env_number("MAX_SENTENCES_PER_LINE", 2)
PAYOFF_MAX_SENTENCES = max(MAX_SENTENCES_PER_LINE, _env_number("PAYOFF_MAX_SENTENCES", 3))   # both paths and the reveal
LONG_SENTENCE_WORDS = _env_number("LONG_SENTENCE_WORDS", 25)
NAME_BUDGET = _env_number("NAME_BUDGET", 4)   # names a newcomer can hold in 60 seconds (the writer is asked for three)   # longer than this is hard to follow when spoken
ANNOUNCE_ONLY_PATTERNS = (   # lines that only say a choice is coming, instead of telling a path
    r"\byou (?:must|have to|need to|get to) (?:now )?(?:decide|choose|make a choice)\b", r"\bthe choice is (?:yours|simple|clear)\b",
    r"\bit(?:'s| is) up to you\b", r"\bwhat will you do\b", r"\byou (?:have|face) (?:a|two|one) (?:choice|choices|options?|paths?)\b",
    r"\bthere are (?:two|only two) (?:ways|options|paths|choices)\b", r"\bnow comes the choice\b",
)
_PLAIN_LONG_WORDS = set("""everything everyone everybody everywhere eventually actually absolutely especially immediately
    unfortunately completely definitely apparently originally desperately ultimately particularly incredible impossible
    unbelievable terrifying horrifying anything understanding relationship experience information responsible
    invisible beautiful dangerously remembering intelligent""".split())


def _syllables(word):
    """Rough spoken-syllable count (vowel groups, a silent final e dropped); good enough to spot long, formal words."""
    word = word.lower().strip("'")
    groups = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and not word.endswith(("le", "ee", "ie")) and groups > 1:
        groups -= 1
    return max(1, groups)


def _first_mentions(lines, game):
    """Names in the spoken lines (capitalised words that are not just a sentence start) with where each first appears:
    {name: (line index, section, text before it, text after it)}."""
    names, seen = {}, set()
    texts = [(sec, strip_game_title(remove_citation(c), game)) for sec, c in lines]
    for sec, text in texts:
        for m in re.finditer(r"\b([A-Z][a-z][a-z’']+)", text):
            if m.start() > 0 and not re.search(r"[.!?:]\s*$", text[:m.start()]):
                key = re.sub(r"[’']s?$", "", m.group(1))
                if key.casefold() not in _NAME_STOPWORDS and key != "TITLE":
                    seen.add(key)
    for n, (sec, text) in enumerate(texts):
        for m in re.finditer(r"\b([A-Z][a-z][a-z’']+)", text):
            key = re.sub(r"[’']s?$", "", m.group(1))
            if key in seen and key not in names:
                names[key] = (n, sec, text[:m.start()], text[m.end():])
    return names


def _introduced(before, after):
    """Is a name given a role where it first appears ('his brother Tomas', 'Lord X', 'Orla, his niece,', 'the Wraith')?"""
    return bool(re.search(r"(?:\b(?i:his|her|their|its|your|the|a|an|our|my)\s+(?:[\w-]+,?\s+){0,3}|\b[A-Z][a-z]+\s+|"
                          r"\b(?:is|was|named|called|known as)\s+)$", before)
                or re.match(r"\s*(?:,|\u2014|-)\s*(?:the|a|an|his|her|their|who|whose|which|your)\b", after)
                or re.match(r"[’']?s?\s+(?:is|was)\s+(?:the|a|an|his|her|their|your)\b", after)
                or re.match(r"\s*\(", after))


def telling_problems(lines, game="", topic=""):
    """Craft problems a viewer feels on first listen, found in code so they never depend on a judge: a person named
    with no hint of who they are, a new name in the payoff or the closing question, a line that only announces the
    choice, a sentence too long to follow, long formal words. Style notes: fed back and scored, never a block."""
    problems = []
    subject = {w.casefold() for w in re.findall(r"[A-Za-z']{3,}", (topic or "").replace("’", "'"))} - {w.casefold() for w in _NAME_STOPWORDS}
    hook = remove_citation(next((c for sec, c in lines if sec == "HOOK"), ""))
    for name, (n, sec, before, after) in _first_mentions(lines, game).items():
        if sec == "HOOK" or name in hook or re.fullmatch(_TITLES, name):
            continue   # (a title such as "Lord" in "Lord Tomas" is not a person)
        if sec in ("PAYOFF", "ENDING QUESTION"):
            problems.append(f"{sec} brings in a new name, {name}: say who they are earlier, or call them by what they do.")
        elif not _introduced(before, after):
            problems.append(f"Say who {name} is the first time they are named (what they are to the story, in a few words), "
                            "or call them by their role if they only do one thing.")
    early = " ".join(remove_citation(c) for sec, c in [l for l in lines if l[0] == "NARRATION"][:2]).replace("’", "'")
    if subject and not any(w in early.casefold() for w in subject):
        problems.append(f'The video is about "{topic}", but the first two NARRATION lines never name it: name it and say what it is.')
    ifs = sum(len(re.findall(r"(?:^|[.!?]\s+)(?:But\s+|And\s+|Or\s+)?if you\b", remove_citation(c), re.I))
              for sec, c in lines if sec in ("NARRATION", "PAYOFF"))
    if ifs > 1:
        problems.append(f"{ifs} sentences start with 'If you', so it reads like a manual: keep one, and tell the other "
                        "path as 'You could...' or 'Or you...'.")
    named = [n for n in _first_mentions(lines, game) if not re.fullmatch(_TITLES, n)
             and not (subject and n.casefold() in subject)]
    if len(named) > NAME_BUDGET:
        problems.append(f"{len(named)} names ({', '.join(named)}) is too many for a newcomer: keep the {NAME_BUDGET} that "
                        "matter most and call the rest by their role.")
    for sec, content in lines:
        text = remove_citation(content)
        if sec == "NARRATION" and len(text.split()) <= 12 and any(re.search(p, text, re.I) for p in ANNOUNCE_ONLY_PATTERNS):
            problems.append(f"The line \"{text}\" only announces the choice: cut it and go straight into the first path.")
        if sec in ("NARRATION", "PAYOFF"):
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                if len(sentence.split()) > LONG_SENTENCE_WORDS:
                    problems.append(f"{sec} has a {len(sentence.split())}-word sentence; split it so it can be followed when spoken.")
                    break
    spoken = " ".join(strip_game_title(remove_citation(c), game) for sec, c in lines if sec != "HOOK")
    counts = {}
    for word in re.findall(r"[a-z']+", spoken.lower()):
        if len(word) > 4 and word not in _SCRIPT_STOPWORDS:
            counts[word] = counts.get(word, 0) + 1
    word, times = max(counts.items(), key=lambda kv: kv[1], default=("", 0))
    if times >= 4:
        problems.append(f"'{word}' is said {times} times; vary the wording.")
    fillers = len(re.findall(r"\b(?:basically|really|just|actually|literally|very|truly|simply)\b", spoken, re.I))
    if fillers > 3:
        problems.append(f"{fillers} filler words (really, just, actually...) slow the pace.")
    body = [(sec, len(remove_citation(c).split())) for sec, c in lines if sec in ("NARRATION", "PAYOFF")]
    total = sum(n for _, n in body)
    if len(body) >= 4 and total and max(n for _, n in body) > 0.35 * total:
        biggest = max(body, key=lambda b: b[1])
        problems.append(f"One {biggest[0]} line holds {round(100 * biggest[1] / total)}% of the words; split it so the story moves.")
    hard = []
    for word in re.findall(r"\b[a-z][a-z']+\b", spoken):   # a hyphenated compound is judged by its parts
        if len(word) >= 10 and _syllables(word) >= 4 and word not in _PLAIN_LONG_WORDS and word not in hard:
            hard.append(word)
    if hard:
        problems.append("Plain words: say " + ", ".join(hard[:4]) + " in everyday words a friend would use.")
    return problems
DRAFTS_PER_ATTEMPT = _env_number("DRAFTS_PER_ATTEMPT", 2)      # drafts written per attempt; the judge keeps the best
SCRIPT_VERSIONS = _env_number("SCRIPT_VERSIONS", 0)            # finished versions per hook for you to pick from (0 = 3 for a hook you picked, else 1)
VERSION_ANGLES = (                                             # each version opens the story a different way
    ("Best judgement", ""),
    ("Opens on the people", "Open the narration on the people caught in this: who is suffering or afraid, and what they stand to lose."),
    ("Opens on the danger", "Open the narration on the danger itself: what it does to its victims, as vividly and concretely as the material allows."),
)
REVISE_PASS = os.environ.get("REVISE_PASS", "1") != "0"           # editor pass for flow before the fact-check
CITATION_TAIL = re.compile(r"(?:\s*\[E\d+\])+\s*$")
RETENTION_OK = {"EXCELLENT", "GOOD", "UNSCORED"}


def line_citations(text):
    match = CITATION_TAIL.search(text)
    return re.findall(r"\[(E\d+)\]", match.group(0)) if match else []


def excerpt_of(passage_text):
    match = re.search(r"(?ms)^Excerpt:\s*(.*)$", passage_text)
    return (match.group(1) if match else passage_text).strip()


def _norm_words(text):
    text = text.casefold().replace("’", "'")
    text = re.sub(r"'s\b", "", text)
    return re.findall(r"[a-z0-9]+", text)


def _contiguous_match(quote_words, text_words, threshold=0.8):
    if not quote_words or not text_words:
        return False
    joined_q, joined_t = " ".join(quote_words), " ".join(text_words)
    if joined_q in joined_t:
        return True
    matcher = difflib.SequenceMatcher(None, quote_words, text_words, autojunk=False)
    return matcher.find_longest_match(0, len(quote_words), 0, len(text_words)).size / len(quote_words) >= threshold


def quote_in_text(quote, text):
    """True when the auditor's quoted support really exists in the passage (anti-hallucination check)."""
    fragments = [f for f in re.split(r"\.{2,}|…", quote) if _norm_words(f)]
    if sum(len(_norm_words(f)) for f in fragments) < 3:
        return False
    text_words = _norm_words(text)
    return all(_contiguous_match(_norm_words(f), text_words) for f in fragments if len(_norm_words(f)) >= 2)


_TOPIC_STEMS = set()   # set by main(): the topic's own words, ignored when checking that a quote relates to its claim
CLAIM_STATUSES = ("SUPPORTED", "SUPPORTED_WRONG_CITATION", "NEEDS_REVIEW", "UNSUPPORTED", "CONTRADICTED")
_NO_QUOTE = {"NONE", "N/A", "NA", "-", ""}
_LOCATE_CACHE = {}


def parse_claim_rows(raw):
    """Rows of 'CLAIM n: <claim> || VERDICT: <v> || QUOTE: <span>' (VERDICT optional for older replies)."""
    rows = []
    for line in raw.splitlines():
        line = re.sub(r"[*`]", "", line.strip())
        item_type = re.search(r"(?i)\|{1,2}\s*TYPE\s*:\s*([A-Z_]+)\s*", line)
        line = re.sub(r"(?i)\s*\|{1,2}\s*TYPE\s*:\s*[A-Z_]+\s*", " ", line).strip()
        match = re.match(r"(?i)^\s*(?:-\s*)?CLAIM\s*\d*\s*:\s*(.+?)\s*(?:\|{1,2}\s*VERDICT\s*:\s*([A-Z_ ]+?)\s*)?"
                         r"\|{1,2}\s*QUOTE\s*:\s*(.+?)\s*(?:\|{1,2}\s*VERDICT\s*:\s*([A-Z_ ]+?)\s*)?$", line)
        if not match:
            continue
        verdict = (match.group(2) or match.group(4) or "").upper().replace(" ", "_")
        quote = match.group(3).strip().strip('"“”\'').strip()
        if verdict not in {"SUPPORTED", "PARTLY", "NOT_STATED", "CONTRADICTED"}:
            verdict = "NOT_STATED" if quote.upper() in _NO_QUOTE else "SUPPORTED"
        kind = (item_type.group(1).upper() if item_type else "FACT")
        rows.append({"claim": strip_markdown(match.group(1)), "quote": quote, "verdict": verdict,
                     "type": "DESCRIPTION" if kind.startswith("DESC") else "FACT"})
    return rows


def parse_claims(raw):
    return [(row["claim"], row["quote"]) for row in parse_claim_rows(raw)]


def _key_stems(text, game="", ignore=()):
    """4-letter content stems (kill/killing, bury/buried match) minus the game title and words every line of the story
    shares (names, the topic)."""
    short_ignore = {w[:4] for w in ignore}
    return {w[:4] for w in content_stems(text, game) if w[:4] not in short_ignore}


def _quote_relates(claim, quote, game, ignore):
    """Safeguard against a REAL quote that has nothing to do with the claim (the auditor pasting any span it found):
    the span must share at least one key word with the claim, ignoring the topic word and names."""
    claim_stems = _key_stems(claim, game, ignore)
    return not claim_stems or bool(claim_stems & _key_stems(quote, game, ignore))


def _holders(quote, passage_map):
    if quote.upper() in _NO_QUOTE:
        return []
    return [i for i, text in passage_map.items() if quote_in_text(quote, excerpt_of(text))]


def locate_support(claim, passage_map, game="", exclude=(), limit=5):
    """Search the OTHER available passages for one that states the claim (multi-passage support / wrong citation).
    The model picks a passage and copies a span; code verifies the span exists in that passage.
    Returns {'id', 'quote', 'verdict'} or None."""
    key = (claim, tuple(sorted(passage_map)), tuple(sorted(exclude)))
    if key in _LOCATE_CACHE:
        return _LOCATE_CACHE[key]
    claim_stems = content_stems(claim, game)
    stems_of = {i: content_stems(excerpt_of(text), game) for i, text in passage_map.items() if i not in exclude}
    total = max(1, len(stems_of))
    # a word in nearly every passage (the topic, a main character) says little; a rare one ("loot") says a lot
    weight = {w: math.log((total + 1) / (1 + sum(w in st for st in stems_of.values()))) + 0.1 for w in claim_stems}
    ranked = sorted(((sum(weight[w] for w in claim_stems & st), i) for i, st in stems_of.items()), reverse=True)
    candidates = [i for score, i in ranked[:max(limit, 8)] if score > 0.1]
    found = None
    if candidates:
        listing = "\n".join(f"[{i}] {excerpt_of(passage_map[i])}" for i in candidates)
        prompt = f"""Does any passage below state this claim? Judge meaning, not wording: a faithful paraphrase counts. Use no outside knowledge. The passage must be about the same subject, character or event as the claim, not a similar one (for example a real-world myth or another story that the subject resembles).
CLAIM: {claim}
Reply with ONE line, either
ID: E### || VERDICT: SUPPORTED or PARTLY or CONTRADICTED || QUOTE: <shortest verbatim span copied from that passage>
or exactly NONE.
PASSAGES:
{listing}"""
        try:
            raw = ask_ollama(prompt, temperature=0.0)
        except Exception:
            raw = ""
        for line in raw.splitlines():
            match = re.match(r"(?i)^\s*ID\s*:\s*\[?(E\d+)\]?\s*\|{1,2}\s*VERDICT\s*:\s*([A-Z_]+)\s*\|{1,2}\s*QUOTE\s*:\s*(.+?)\s*$",
                             re.sub(r"[*`]", "", line.strip()))
            if not match:
                continue
            evidence_id, verdict = match.group(1).upper(), match.group(2).upper()
            quote = match.group(3).strip().strip('"“”\'').strip()
            if evidence_id in candidates and verdict in {"SUPPORTED", "PARTLY", "CONTRADICTED"} \
                    and quote_in_text(quote, excerpt_of(passage_map[evidence_id])):
                found = {"id": evidence_id, "quote": quote, "verdict": verdict}
                break
    _LOCATE_CACHE[key] = found
    return found


def is_description(row):
    """A descriptive word or phrase (how something looks, sounds or seems: 'monstrous', 'eerie', 'grieving') is not a
    factual claim and needs no citation. The AUDITOR decides what is descriptive (no word list in code); code only
    refuses the label when the item carries the marks of a fact: a cause-and-effect link or a number."""
    return row.get("type") == "DESCRIPTION" and not CAUSAL_CONNECTOR.search(row["claim"]) and not re.search(r"\d", row["claim"])


_QUOTE_SAYS_CACHE = {}


def quote_says(claim, quote):
    """Does this exact evidence quote state this claim, in any wording? One short yes/no question, used only when
    the words differ, so a synonym or a plain retelling is never mistaken for an invented fact."""
    key = (claim, quote)
    if key not in _QUOTE_SAYS_CACHE:
        prompt = f"""Does the QUOTE state the CLAIM? Judge meaning, not wording: synonyms, paraphrases and plain retellings count (for example "vanishes" for "is banished"). Answer NO if the claim adds an event, cause, outcome, name or number the quote does not contain.
QUOTE: {quote}
CLAIM: {claim}
Reply with exactly YES or NO."""
        try:
            answer = ask_ollama(prompt, temperature=0.0)
        except Exception:
            answer = ""
        _QUOTE_SAYS_CACHE[key] = bool(re.match(r"\s*\**\s*YES\b", answer or "", re.I))
    return _QUOTE_SAYS_CACHE[key]


def _classify_claim(row, ids, passage_map, game, ignore, section):
    """One claim -> one of CLAIM_STATUSES (or DESCRIPTION). The ONLY thing that blocks a story is an INVENTED fact:
    a claim is UNSUPPORTED only when the auditor finds no support AND an independent search of every available
    passage finds none either. A real quote the auditor judged supportive is trusted even when it uses other words
    ("vanishes" for "is banished"); citation errors are fixed in code, never held against the story."""
    claim, quote, verdict = row["claim"], row["quote"], row["verdict"]
    if is_description(row):
        return {**row, "status": "DESCRIPTION", "found_in": None, "note": ""}
    cited = {i: passage_map[i] for i in ids}
    in_cited = _holders(quote, cited)
    elsewhere = [i for i in _holders(quote, passage_map) if i not in ids]
    related = _quote_relates(claim, quote, game, ignore)

    def supported(found_id, quote_text, note_partly=False):
        if note_partly:   # stated less strongly (a belief told as certain): noted for you, never a block
            where = "the cited passage" if found_id in ids else found_id
            return {**row, "status": "NEEDS_REVIEW", "quote": quote_text, "found_in": found_id,
                    "note": f"{where} only partly supports it"}
        status = "SUPPORTED" if found_id in ids else "SUPPORTED_WRONG_CITATION"
        note = "" if status == "SUPPORTED" else f"stated in {found_id}, not in {', '.join(ids)}"
        return {**row, "status": status, "quote": quote_text, "found_in": found_id, "note": note}

    if in_cited and verdict == "CONTRADICTED":
        # a contradiction rejects the story, so it needs a second opinion: auditors sometimes misread a line into a
        # claim it does not make. If the second check finds the line's claim stated, it stands.
        second = locate_support(claim, passage_map, game)
        if second and second["verdict"] in {"SUPPORTED", "PARTLY"}:
            return supported(second["id"], second["quote"], second["verdict"] == "PARTLY")
        return {**row, "status": "CONTRADICTED", "found_in": in_cited[0], "note": f"contradicted by {in_cited[0]}"}
    if verdict in {"SUPPORTED", "PARTLY"} and (in_cited or elsewhere):
        found = (in_cited or elsewhere)[0]
        if related:
            return supported(found, quote, verdict == "PARTLY")
        # other words than the line (a synonym or a retelling): ask once more, and trust the real quote if the
        # second look finds nothing better; disagreement about wording is not an invented fact
        second = locate_support(claim, passage_map, game)
        if second and second["verdict"] == "CONTRADICTED":
            return {**row, "status": "CONTRADICTED", "quote": second["quote"], "found_in": second["id"],
                    "note": f"contradicted by {second['id']}"}
        if second:
            return supported(second["id"], second["quote"], second["verdict"] == "PARTLY")
        # one direct question settles a synonym ("vanishes" / "is banished") apart from a pasted, unrelated span
        if quote_says(claim, quote):
            return supported(found, quote, verdict == "PARTLY")
        return {**row, "status": "UNSUPPORTED", "found_in": None,
                "note": "the quoted span does not say this and no other available passage does"}
    # the auditor found no support (or quoted words that are in no passage): search every available passage
    hit = locate_support(claim, passage_map, game)
    if hit:
        if hit["verdict"] == "CONTRADICTED":
            return {**row, "status": "CONTRADICTED", "quote": hit["quote"], "found_in": hit["id"], "note": f"contradicted by {hit['id']}"}
        return supported(hit["id"], hit["quote"], hit["verdict"] == "PARTLY")
    note = (f"no available passage states it (the auditor's quote is not in {', '.join(ids)})"
            if quote.upper() not in _NO_QUOTE else "no available passage states it")
    return {**row, "status": "UNSUPPORTED", "found_in": None, "note": note}


def _citation_cover(rows, cited, passage_map):
    """Smallest set of passages that holds the support for every supported claim, keeping the line's own citations
    whenever they hold any of it (so a fix never drops a passage the line legitimately uses)."""
    need = []
    for row in rows:
        holders = set(_holders(row["quote"], passage_map)) | ({row["found_in"]} if row.get("found_in") else set())
        if holders:
            need.append(holders)
    chosen = [i for i in cited if any(i in h for h in need)]
    remaining = [h for h in need if not (h & set(chosen))]
    while remaining:
        best = max(sorted({i for h in remaining for i in h}), key=lambda i: sum(i in h for h in remaining))
        chosen.append(best)
        remaining = [h for h in remaining if best not in h]
    return chosen or list(cited)


def audit_line(section, content, passage_map, game="", ignore=()):
    """Fact-check ONE spoken line. The auditor judges MEANING claim by claim (SUPPORTED / PARTLY / NOT_STATED /
    CONTRADICTED) and copies a span; code verifies every span exists, looks for support in the other available
    passages before calling a claim unsupported, and separates citation errors from factual errors.
    Line status: CLEAR (all supported, correctly cited), RECITE (supported, but cite recite_ids instead),
    REVIEW (a claim the auditor could not settle; never repaired into blander text, shown to the human),
    FLAGGED (an unsupported or contradicted claim)."""
    ids = line_citations(content)
    text = remove_citation(content)
    if not ids:
        return {"status": "FLAGGED", "claims": [], "rows": [], "unsupported": [("line has no citation", "NONE")], "review": [],
                "recite_ids": None, "reason": "line has no citation", "raw": ""}
    for evidence_id in ids:
        if evidence_id not in passage_map:
            return {"status": "FLAGGED", "claims": [], "rows": [], "unsupported": [(f"{evidence_id} unavailable", "NONE")],
                    "review": [], "recite_ids": None, "reason": f"cited passage {evidence_id} unavailable", "raw": ""}
    excerpts = [(evidence_id, excerpt_of(passage_map[evidence_id])) for evidence_id in ids]
    if section == "ENDING QUESTION":
        rubric = ("This line is a QUESTION to the viewer. Asking it, and any open dilemma or opinion inside it, "
                  "is NOT a claim. List only facts the question takes for granted. If it takes no fact for "
                  "granted, output exactly: CLAIM 1: (no factual premise) || VERDICT: SUPPORTED || QUOTE: N/A")
    else:
        rubric = ("Hedged interpretation (may, perhaps, one reading, it raises the question) is not a claim. "
                  "Attributed beliefs (a character believes X) are claims about the belief, not about X. Everything else is a claim.")
    prompt = f"""You are a fair fact-checker for a storytelling video. Your ONLY job is to catch INVENTED facts: things this line states as true that the evidence excerpts below do not support. Everything else is the writer's freedom. Use no outside knowledge.
1. Split the line into its factual claims (an event, who did what, a cause, a motive, an outcome, a name, a number). Ignore the game title ("{game}") itself.
2. Judge each claim by MEANING, generously: SUPPORTED when an excerpt states it or it plainly follows from what the excerpts say, in ANY wording (synonyms, paraphrases, summaries and plain retellings all count, for example "vanishes" for "is banished" or "hurt himself" for "self-harm"). PARTLY when an excerpt states it less strongly (for example the excerpt says people believe it, and the line states it as certain). NOT_STATED only when the claim adds something no excerpt supports: a new event, outcome, cause, motive, name or number. CONTRADICTED when an excerpt says otherwise; giving an action to the wrong person is CONTRADICTED.
3. Copy the shortest verbatim span from the excerpts that shows your verdict, or NONE.
JOINING EVENTS IS A CLAIM: when the line places two events together in time ("during", "in the chaos", "after", "then", "at the same time") or as cause and effect ("so", "which led to", "because", "this is why"), the link itself is a separate claim, SUPPORTED only if an excerpt states that link. Who someone is and how people are related (wife, brother, father, servant) are claims too.
STORYTELLING IS NOT A CLAIM (TYPE: DESCRIPTION, needs no quote, is never wrong): a mood or description word that fairly sums up what the excerpts describe (terrifying, monstrous, grieving, a tragedy) is NOT a claim, and neither is any storytelling device: dramatic framing ("but the truth was darker"), rhetorical questions, foreshadowing, transitions, how a scene feels, or what a character does not know or understand. In a game, an optional player choice is not an event: if the excerpts say "if the player/character chooses X, then Y", a line that says X happened (rather than that it can be chosen) is NOT_STATED, while "you can choose X, and if you do, Y" is SUPPORTED.
{rubric}
Output ONLY lines in this exact format, one per item:
CLAIM 1: <claim in your own words> || TYPE: FACT || VERDICT: SUPPORTED || QUOTE: <verbatim span copied from an excerpt, or NONE>
EVIDENCE EXCERPTS:
{chr(10).join(f'[{evidence_id}] {excerpt}' for evidence_id, excerpt in excerpts)}
LINE ({section}): {text}"""
    rows, raw = [], ""
    for attempt in range(2):
        raw = ask_ollama(prompt, temperature=0.0 if attempt == 0 else 0.2)
        rows = parse_claim_rows(raw)
        if rows:
            break
    judged = []
    for row in rows:
        if section == "ENDING QUESTION" and re.search(r"no factual premise", row["claim"], re.I):
            continue
        names = {w[:5] for w in re.findall(r"(?<!^)(?<![.!?] )\b([A-Z][a-z]{3,})", text)}
        names = {n.lower() for n in names}
        judged.append(_classify_claim(row, ids, passage_map, game, set(ignore) | _TOPIC_STEMS | names, section))
    if not rows:
        judged.append({"claim": "(the auditor returned no usable verdict)", "quote": "NONE", "verdict": "NOT_STATED",
                       "status": "NEEDS_REVIEW", "found_in": None, "note": "could not be checked automatically"})
    bad = [r for r in judged if r["status"] in {"UNSUPPORTED", "CONTRADICTED"}]
    wrong = [r for r in judged if r["status"] == "SUPPORTED_WRONG_CITATION"]
    review = [r for r in judged if r["status"] == "NEEDS_REVIEW"]
    recite_ids, too_many = None, None
    if not bad and wrong:
        recite_ids = _citation_cover([r for r in judged if r["status"] != "DESCRIPTION"], ids, passage_map)
        if len(recite_ids) > MAX_CITATIONS_PER_LINE:
            too_many, recite_ids = len(recite_ids), None
    status = "FLAGGED" if bad or too_many else "RECITE" if recite_ids else "REVIEW" if review else "CLEAR"
    reasons = list(dict.fromkeys(f"'{r['claim']}' {r['status'].lower().replace('_', ' ')}: {r['note']}" for r in bad + review if r["note"] or r in bad))
    if too_many:
        reasons.append(f"every claim is supported, but by {too_many} different passages; split it into two lines "
                       f"(at most {MAX_CITATIONS_PER_LINE} citations per line)")
    elif recite_ids:
        reasons.append("citation corrected: " + ", ".join(ids) + " -> " + ", ".join(recite_ids))
    return {"status": status, "claims": [(r["claim"], r["quote"]) for r in judged], "rows": judged,
            "unsupported": [(r["claim"], r["quote"]) for r in bad], "review": [(r["claim"], r["note"]) for r in review],
            "recite_ids": recite_ids, "reason": "; ".join(reasons), "raw": raw}


def is_flagged(audit):
    """Blocks a line: an unsupported or contradicted claim, or an unresolved citation fix. The fixed HOOK passed the
    hook audit already; a citation note on it (RECITE) is not a factual problem."""
    if audit.get("section") == "HOOK" and audit["status"] == "RECITE":
        return False
    return audit["status"] not in {"CLEAR", "REVIEW"}


def recite_line(content, new_ids):
    return remove_citation(content).strip() + " " + " ".join(f"[{i}]" for i in new_ids)


VERIFIED_HOOKS = set()   # hooks that passed the hook-stage audit; the script stage does not re-judge them


def audit_script(draft, passage_map, game, cache):
    audits = []
    for index, (section, content) in enumerate(spoken_lines(draft), 1):
        key = (section, content)
        if section == "HOOK" and content in VERIFIED_HOOKS:
            # Re-auditing the same sentence can give a different verdict (one run read "instead of a peaceful passing"
            # as a claim the hook never makes and called it contradicted). It was verified once; that verdict stands.
            audits.append({"index": index, "section": section, "content": content, "status": "CLEAR", "claims": [],
                           "rows": [{"claim": "(hook verified at the hook stage)", "quote": "N/A", "status": "SUPPORTED",
                                     "found_in": None, "note": "", "verdict": "SUPPORTED"}],
                           "unsupported": [], "review": [], "recite_ids": None, "reason": "", "raw": ""})
            continue
        if key not in cache:
            cache[key] = audit_line(section, content, passage_map, game)
        audits.append({"index": index, "section": section, "content": content, **cache[key]})
    return audits


def rebuild_draft(draft, lines):
    more = re.search(r"(?mi)^\s*CLAIMS REQUIRING MORE EVIDENCE\s*:\s*(.*)$", draft)
    return "\n".join(["STATUS: DRAFT"] + [f"{section}: {content}" for section, content in lines]
                     + [f"CLAIMS REQUIRING MORE EVIDENCE: {more.group(1).strip() if more else 'None'}"])


def _related_passages(text, passage_map, game, cited, limit=6):
    stems = content_stems(text, game)
    ranked = sorted(((len(stems & content_stems(excerpt_of(t), game)), i) for i, t in passage_map.items() if i not in cited),
                    reverse=True)
    return list(cited) + [i for score, i in ranked[:limit] if score >= 1]


def repair_line(game, topic, audit, passage_map, valid_ids, max_words, repeats=None, other_lines=()):
    """Rewrite ONE flagged line (not the whole script). Supported claims are listed as KEEP and only the unsupported
    or contradicted ones as REMOVE, so the line keeps its story job and specifics. With `repeats` (the earlier line it
    restates) the rewrite must instead move the story forward with a different supported fact. A rewrite is refused if
    it drops the kept specifics, repeats another line, or is too short to carry a fact."""
    section, content = audit["section"], audit["content"]
    rows = audit.get("rows") or []
    keep = [r["claim"] for r in rows if r.get("status") in {"SUPPORTED", "SUPPORTED_WRONG_CITATION", "NEEDS_REVIEW"}]
    remove = [f"{claim}" for claim, _ in audit["unsupported"]]
    if repeats:
        problems = f"- REPEATS an earlier line that says: {remove_citation(repeats)}"
    else:
        problems = "\n".join(f"- REMOVE (no available passage states it): {claim}" for claim in remove)
        if keep:
            problems += "\n" + "\n".join(f"- KEEP (verified): {claim}" for claim in keep)
    cited_ids = [i for i in line_citations(content) if i in passage_map]
    shown = _related_passages(remove_citation(content), passage_map, game, cited_ids)
    excerpts = "\n".join(f"[{i}] {excerpt_of(passage_map[i])}" for i in shown)
    question_rule = ("\n- This is the ENDING QUESTION: it must end with ? and pose a real dilemma about the story."
                     if section == "ENDING QUESTION" else "")
    task = ("- The line below restates a fact an earlier line already told the viewer. Replace it with a DIFFERENT fact from the excerpts "
            "that moves the story forward (what happened next, what it cost, what it changed) and that no earlier line states. "
            "Do not reuse the earlier line's distinctive words." if repeats else
            "- Keep the line's story job and every KEEP claim with its specific details; delete or correct only the REMOVE claims. "
            "If an excerpt states the removed point differently, say it the way the excerpt does. Never replace a specific "
            "detail with a vaguer phrase.")
    keep_stems = set().union(*[content_stems(k, game) for k in keep]) if keep and not repeats else set()
    last_error = ""
    for attempt in range(1, 4):
        prompt = f"""Rewrite ONE flagged line of a short-video script so that the fact-checker passes it.
RULES:
- Output exactly one line: {section}: <one or two sentences> [E###]  (up to {MAX_CITATIONS_PER_LINE} ids, e.g. [E012] [E013]; ids go at the very end, sentence punctuation before them)
{task}
- Keep the line's wording, voice, rhythm and its link to the lines around it; change only what is needed so that no fact is invented. Descriptive words and storytelling turns may stay; invented events, causes, motives or outcomes may not. Keep who believes or claims something exactly as the excerpt does.
- Cite the passage(s) that state each fact; you may cite ids other than the original ones.
- At most {max_words} words. No quotation marks. Plain text.{question_rule}
GAME: {game}
TOPIC: {topic}
VALID IDS: {', '.join(sorted(valid_ids, key=lambda v: int(v[1:])))}
LINE: {section}: {content}
{'PROBLEM' if repeats else 'CLAIMS'}:
{problems}
{('PREVIOUS ATTEMPT PROBLEM: ' + last_error) if last_error else ''}
EXCERPTS:
{excerpts}"""
        raw = ask_writer(prompt, temperature=0.2 + 0.1 * (attempt - 1))
        candidate = ""
        for raw_line in raw.splitlines():
            tag, body = parse_script_line(raw_line)
            if tag == section and body:
                candidate = body
                break
        if not candidate:
            last_error = f"reply must be a single line starting with '{section}:'"
            continue
        candidate, _ = canonicalise_citation_ids(strip_markdown(candidate), valid_ids)
        ids, clean = line_citations(candidate), remove_citation(candidate)
        repeated = [o for o in other_lines if _restates(candidate, o, game)]
        if not 1 <= len(ids) <= MAX_CITATIONS_PER_LINE or any(i not in valid_ids for i in ids):
            last_error = f"cite 1-{MAX_CITATIONS_PER_LINE} valid ids at the very end"
        elif not 1 <= sentence_count(strip_game_title(clean, game)) <= MAX_SENTENCES_PER_LINE:
            last_error = f"use one or {MAX_SENTENCES_PER_LINE} sentences"
        elif any(mark in clean for mark in ['"', '“', '”', '„', '«', '»']):
            last_error = "no quotation marks"
        elif section == "ENDING QUESTION" and not clean.endswith("?"):
            last_error = "must end with a question mark"
        elif len(clean.split()) > max_words:
            last_error = f"{len(clean.split())} words; maximum is {max_words}"
        elif len(clean.split()) < 6:
            last_error = "too short to carry a fact; keep the verified details"
        elif keep_stems and len(keep_stems & content_stems(candidate, game)) < min(2, len(keep_stems)):
            last_error = "the rewrite dropped the verified KEEP details; keep them and remove only the REMOVE claims"
        elif repeated:
            last_error = f"the rewrite repeats another line ({remove_citation(repeated[0])}); state something new"
        else:
            return candidate
    return None


def _restates(first, second, game):
    a, b = content_stems(first, game), content_stems(second, game)
    smaller = min(len(a), len(b))
    if smaller < 2:
        return False
    shared = len(a & b)
    return (shared / smaller >= 0.75 and shared >= 3) or (shared >= 4 and shared / smaller >= 0.5)


def _print_audits(audits):
    for a in audits:
        print(f"  line {a['index']} [{a['section']}]: {a['status']}" + (f" - {a['reason']}" if a["status"] != "CLEAR" and a["reason"] else ""))


def apply_recitations(draft, audits):
    """Code-only citation fixes: a line whose claims are all supported but by other passages gets those ids."""
    lines = spoken_lines(draft)
    changed = []
    for a in audits:
        if a["status"] == "RECITE" and a["recite_ids"] and a["section"] != "HOOK":   # the HOOK is fixed and was verified
            lines[a["index"] - 1] = (a["section"], recite_line(a["content"], a["recite_ids"]))
            changed.append(f"line {a['index']}: {', '.join(line_citations(a['content']))} -> {', '.join(a['recite_ids'])}")
    return (rebuild_draft(draft, lines) if changed else draft), changed


def audit_and_repair(game, topic, draft, passage_map, valid_ids, cache):
    """Audit every line. Order of fixes: (1) correct citations in code, (2) rewrite only lines with an unsupported or
    contradicted claim, keeping their verified claims, (3) leave NEEDS_REVIEW lines untouched for the human, so an
    uncertain judgement never turns a specific line into a blander one. Repeats a few rounds."""
    audits = audit_script(draft, passage_map, game, cache)
    for round_number in range(1, MAX_REPAIR_ROUNDS + 1):
        for _ in range(3):
            draft, recited = apply_recitations(draft, audits)
            if not recited:
                break
            print("  citation fixes (claims supported by another passage): " + "; ".join(recited))
            audits = audit_script(draft, passage_map, game, cache)
        _print_audits(audits)
        repairable = [a for a in audits if a["status"] == "FLAGGED" and a["section"] != "HOOK"]
        if not repairable:
            return draft, audits
        lines = spoken_lines(draft)
        changed = False
        for a in repairable:
            max_words = max(12, min(32, len(remove_citation(a["content"]).split()) + 4))
            others = [c for i, (_, c) in enumerate(lines) if i != a["index"] - 1 and lines[i][0] != "ENDING QUESTION"]
            new_content = repair_line(game, topic, a, passage_map, valid_ids, max_words, other_lines=others)
            if new_content and new_content != a["content"]:
                lines[a["index"] - 1] = (a["section"], new_content)
                changed = True
                print(f"  repair round {round_number}: rewrote line {a['index']} [{a['section']}]")
        if not changed:
            return draft, audits
        draft = rebuild_draft(draft, lines)
        audits = audit_script(draft, passage_map, game, cache)
    draft, _ = apply_recitations(draft, audits)
    audits = audit_script(draft, passage_map, game, cache)
    _print_audits(audits)
    return draft, audits


def repair_repetition(game, topic, draft, passage_map, valid_ids, audit_cache):
    """Lines that restate an earlier line are rewritten once (the HOOK is fixed, so the later line always changes).
    Returns (draft, audits, changed)."""
    lines = spoken_lines(draft)
    changed = False
    for first, second, _shared, _ratio, fails in script_repetition(draft, game):
        if not fails or lines[second - 1][0] == "HOOK":
            continue
        section, content = lines[second - 1]
        stub = {"section": section, "content": content, "unsupported": []}
        max_words = max(12, min(28, len(remove_citation(content).split()) + 2))
        others = [c for i, (sec, c) in enumerate(lines) if i != second - 1 and sec != "ENDING QUESTION"]
        new_content = repair_line(game, topic, stub, passage_map, valid_ids, max_words, repeats=lines[first - 1][1], other_lines=others)
        if new_content and new_content != content:
            lines[second - 1] = (section, new_content)
            changed = True
            print(f"  repetition repair: rewrote line {second} [{section}] which restated line {first}")
    if not changed:
        return draft, [], False
    draft = rebuild_draft(draft, lines)
    return draft, audit_script(draft, passage_map, game, audit_cache), True


def retention_review(game, topic, draft):
    prompt = f"""Assess only story craft, not factual accuracy. Does this script have a compelling first line, clear newcomer context, escalating information, concrete visual beats, a payoff that resolves the hook, and no filler?
Return exactly:
RETENTION_VALUATION: EXCELLENT, GOOD, or POOR
VERDICT: PASS or FAIL
VERDICT must be PASS only if the story is clear and the payoff resolves the hook. Explain briefly afterward.
GAME: {game}
TOPIC: {topic}
DRAFT:
{draft}"""
    text, status, verdict = "", None, None
    for attempt in range(2):
        text = ask_ollama(prompt, temperature=0.0 if attempt == 0 else 0.2)
        status = get_review_result(text, "RETENTION_VALUATION", ["EXCELLENT", "GOOD", "POOR"])
        verdict = get_review_result(text, "VERDICT", ["PASS", "FAIL"])
        if status and verdict:
            break
    return {"status": status or "UNSCORED", "verdict": verdict or "UNSCORED", "text": text}


def retention_ok(retention):
    return bool(retention) and retention["status"] in RETENTION_OK and retention["verdict"] in {"PASS", "UNSCORED"}


def format_review(audits, retention):
    flagged = [a for a in audits if is_flagged(a)]
    review = [a for a in audits if a["status"] == "REVIEW"]
    out = [f"EVIDENCE AUDIT: {'FLAGGED' if flagged else 'NEEDS HUMAN REVIEW' if review else 'CLEAR'}"]
    if retention:
        out += [f"RETENTION_VALUATION: {retention['status']}", f"VERDICT: {retention['verdict']}"]
    else:
        out += ["RETENTION_VALUATION: NOT RUN (evidence audit failed first)", "VERDICT: FAIL"]
    out += ["", "CLAIM STATUSES: SUPPORTED = a cited passage states it; SUPPORTED_WRONG_CITATION = another passage states it "
            "(the citation is fixed in code); NEEDS_REVIEW = support is uncertain, a human must check it; UNSUPPORTED = no "
            "available passage states it; CONTRADICTED = a passage says otherwise.", "", "PER-LINE EVIDENCE AUDITS:"]
    for a in audits:
        out.append(f"LINE {a['index']} [{a['section']}] {', '.join(line_citations(a['content']))}: {a['status']}")
        out.append(f"  TEXT: {remove_citation(a['content'])}")
        rows = a.get("rows") or [{"claim": c, "quote": q, "status": "SUPPORTED", "found_in": None, "note": ""} for c, q in a["claims"]]
        for row in rows:
            where = f" [{row['found_in']}]" if row.get("found_in") else ""
            quote = f'  <- "{row["quote"]}"{where}' if row["quote"].upper() not in _NO_QUOTE else ""
            note = f"  ({row['note']})" if row.get("note") else ""
            out.append(f"  {row['status']:<24}: {row['claim']}{quote}{note}")
        if a["status"] != "CLEAR" and a["reason"]:
            out.append(f"  REASON: {a['reason']}")
    if retention:
        out += ["", "RETENTION REVIEW:", retention["text"]]
    return "\n".join(out)


def get_review_result(review, label, allowed):
    choices = "|".join(allowed)
    for line in review.splitlines():
        match = re.fullmatch(rf"{re.escape(label)}:\s*({choices})", clean_review_line(line), re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:50] or "gaming_lore"


def clean_narration(draft):
    return "\n".join(remove_citation(content) for _, content in spoken_lines(draft))


_LOWERCASE_STARTERS = {"the", "a", "an", "this", "that", "these", "those", "her", "his", "their", "its", "it", "she", "he", "they",
                       "after", "when", "while", "before", "once", "as", "but", "and", "so", "then", "there", "one", "with", "by", "for"}


def ensure_game_title(draft, game, selected_hook):
    """When the fixed HOOK does not name the game, put the title at the start of the first NARRATION line in code.
    The title is context, not a claim, so this cannot add an unsupported fact. Returns (draft, changed)."""
    if not game or hook_identifies_game(selected_hook or "", game):
        return draft, False
    lines = spoken_lines(draft)
    if hook_identifies_game(" ".join(c for _, c in lines[:2]), game):
        return draft, False
    for index, (section, content) in enumerate(lines):
        if section != "NARRATION":
            continue
        first = content.split(" ", 1)[0]
        body = content if first.casefold().strip(",") not in _LOWERCASE_STARTERS else first.lower() + content[len(first):]
        lines[index] = (section, f"In {game_title_variants_raw(game)}, {body}")
        return rebuild_draft(draft, lines), True
    return draft, False


STRUCTURAL_PROBLEMS = ("No spoken script lines", "The first spoken line must be HOOK", "Exactly one HOOK", "HOOK must exactly match",
                       "Exactly one PAYOFF", "Exactly one ENDING QUESTION", "At least two NARRATION", "uses unselected evidence")


def is_structural(problem):
    """Problems that make the file unusable (or cite evidence that was never selected). Everything else in
    check_script_format is style: fed back to the writer and scored, never a reason to reject the story."""
    return problem.startswith(STRUCTURAL_PROBLEMS) or " uses unselected evidence" in problem


def enforce_hook(draft, selected_hook):
    """The HOOK is fixed: put it back verbatim in code instead of failing a story because the writer retyped it."""
    lines = spoken_lines(draft)
    if not lines:
        return draft
    if lines[0][0] == "HOOK" and lines[0][1] == selected_hook:
        return draft
    lines = [(sec, c) for sec, c in lines if sec != "HOOK"]
    return rebuild_draft(draft, [("HOOK", selected_hook)] + lines)


def check_script_format(draft, valid_ids, selected_hook=None, game=None, essential=None, topic=""):
    problems = []
    lines = spoken_lines(draft)
    if not lines:
        return ["No spoken script lines found."]

    sections = [section for section, _ in lines]
    if sections[0] != "HOOK":
        problems.append("The first spoken line must be HOOK.")
    if sections.count("HOOK") != 1:
        problems.append("Exactly one HOOK is required.")
    if selected_hook:
        hook_content = next((content for section, content in lines if section == "HOOK"), "")
        if hook_content != selected_hook:
            problems.append("HOOK must exactly match the selected hook, including citation and punctuation.")
    if sections.count("PAYOFF") != 1:
        problems.append("Exactly one PAYOFF is required.")
    if sections.count("ENDING QUESTION") != 1:
        problems.append("Exactly one ENDING QUESTION is required.")
    if sections.count("NARRATION") < 2:
        problems.append("At least two NARRATION lines are required.")
    if sections.count("NARRATION") > 7:
        problems.append("At most seven NARRATION lines; cut any line that does not add new information.")

    first_two_lines = " ".join(content for _, content in lines[:2])
    if game and not hook_identifies_game(first_two_lines, game):
        problems.append(f'Mention the game title in the first NARRATION line, using the exact words "{game_title_variants_raw(game)}" '
                        "(the HOOK is fixed and does not name it).")

    for section, content in lines:
        clean = remove_citation(content)
        ids = line_citations(content)
        if not 1 <= len(ids) <= MAX_CITATIONS_PER_LINE:
            problems.append(f"{section} must end with 1-{MAX_CITATIONS_PER_LINE} citations.")
        elif any(i not in valid_ids for i in ids):
            problems.append(f"{section} uses unselected evidence {ids}.")
        if re.search(r"\[E\d+\]", clean):
            problems.append(f"{section} has a citation that is not at the very end.")
        if any(mark in clean for mark in ['"', '“', '”', '„', '‟', '«', '»']):
            problems.append(f"{section} contains forbidden quotation marks.")
        most = PAYOFF_MAX_SENTENCES if section == "PAYOFF" else MAX_SENTENCES_PER_LINE
        if not 1 <= sentence_count(strip_game_title(clean, game or "")) <= most:
            problems.append(f"{section} must contain one to {most} sentences." if section == "PAYOFF"
                            else f"{section} must contain one or {MAX_SENTENCES_PER_LINE} sentences.")

    total = word_count(draft)
    if not MIN_WORDS <= total <= MAX_WORDS:
        problems.append(f"Script is {total} words; target is {MIN_WORDS}-{MAX_WORDS}.")
    ending = next((remove_citation(content) for section, content in lines if section == "ENDING QUESTION"), "")
    if not ending.endswith("?"):
        problems.append("ENDING QUESTION must end with a question mark.")
    if len(ending.split()) > ENDING_MAX_WORDS:
        problems.append(f"ENDING QUESTION is {len(ending.split())} words; keep it to {ENDING_MAX_WORDS} or fewer so it lands as one sharp question.")
    weak = weak_ending_reason(ending)
    if weak:
        problems.append(weak[0].upper() + weak[1:])
    if get_draft_status(draft) != "DRAFT":
        problems.append("Expected STATUS: DRAFT.")
    problems.extend(telling_problems(lines, game or "", topic))
    # Story craft (repetition, required points) is NOT a layout problem: it is fixed by rewriting, fed back to the
    # writer and scored (see produce_script / score_script), never a reason to throw a story away.
    return problems


def length_feedback(draft):
    words = word_count(draft)
    if words > MAX_WORDS:
        return f"The previous draft had {words} words. Cut at least {words - SCRIPT_TARGET_MAX_WORDS} words and aim for {SCRIPT_TARGET_MIN_WORDS}-{SCRIPT_TARGET_MAX_WORDS} total by removing lines that add nothing new."
    if words < MIN_WORDS:
        return f"The previous draft had {words} words. Add only essential, supported context to reach at least {MIN_WORDS}; do not pad."
    return ""


def hook_script_coherence(game, draft, selected_hook):
    """Does the finished script deliver what the hook promises (hook -> context -> escalation -> payoff)?
    Returns {'verdict': DELIVERS|PARTIAL|FAILS|UNCHECKED, 'text': ...}. UNCHECKED is reported, never hidden."""
    prompt = f"""Check whether this short script keeps the promise its HOOK makes.
Questions: (1) Do the NARRATION lines explain every name or term the HOOK uses? (2) Does the PAYOFF resolve the exact situation, danger or choice the HOOK raises (not a different one)? (3) Does the order hook -> context -> escalation -> payoff make sense to a newcomer?
Return exactly:
COHERENCE: DELIVERS, PARTIAL, or FAILS
REASON: one sentence naming what is missing, or what works
GAME: {game}
HOOK: {remove_citation(selected_hook)}
SCRIPT:
{clean_narration(draft)}"""
    text, verdict = "", None
    for attempt in range(2):
        text = ask_ollama(prompt, temperature=0.0 if attempt == 0 else 0.2)
        verdict = get_review_result(text, "COHERENCE", ["DELIVERS", "PARTIAL", "FAILS"])
        if verdict:
            break
    return {"verdict": verdict or "UNCHECKED", "text": text}


OVERUSED_WORDS = ("monstrous", "grotesque", "terrifying", "mysterious", "horrifying", "hideous", "sinister", "eerie", "chilling")
GENERIC_ENDING_PATTERNS = (
    r"\bwhat do you think\b\??$", r"\blet me know\b", r"\bcomment\b", r"\bwhat are your (thoughts|theories)\b",
    r"\bdo you agree\b", r"\bwhat other \w+(?: \w+){0,3} (?:in|from|of|across) ", r"\bhave you (ever )?(played|seen|met|encountered)\b",
)
OBVIOUS_ENDING_PATTERNS = (
    r"\b(?:how|what) (?:do|did|does|would|will|might|should|could) .{0,60}\b(?:react|reacts|feel|feels|respond|respond to|think of|think about)\b",
    r"\bhow (?:happy|sad|angry|proud|upset|grateful|relieved|pleased)\b",
    r"\b(?:are|were|is|was) .{0,50}\b(?:happy|unhappy|sad|angry|proud|upset|grateful|relieved|pleased|satisfied|afraid|scared)\b",
    r"\bhow (?:do|would|will) you feel\b",
    r"\bwould you (?:be|feel)\b",
)


PREDICTION_ENDING_PATTERNS = (
    r"^will\b", r"\b(?:succeed|work out|pull (?:it|this) off|manage to)\b", r"^can (?:you|he|she|they|\w+) (?:master|survive|win|beat)\b",
)


def weak_ending_reason(ending):
    """Code-side screen for closing questions whose answer is predictable (how someone feels or reacts, whether
    someone is happy or sad). Deliberately narrow so it only catches clear cases; the evaluator judges the rest."""
    text = remove_citation(ending or "").strip().casefold()
    for pattern in PREDICTION_ENDING_PATTERNS:
        if re.search(pattern, text):
            return ("the ENDING QUESTION asks what will happen, a prediction the story answers or no one can argue about. Ask a "
                    "real dilemma instead: which path is right given what each costs, or whether someone deserves what they get.")
    for pattern in OBVIOUS_ENDING_PATTERNS:
        if re.search(pattern, text):
            return ("the ENDING QUESTION asks how someone feels or reacts, so its answer is obvious. Ask a real dilemma instead: "
                    "a choice in the script with a cost, where reasonable viewers would disagree about whether it was right.")
    return ""


_SCRIPT_STOPWORDS = _NAME_STOPWORDS | set("can will would could more most much many such than then them they said says make makes made "
                                           "being into onto upon whose does also like same some other another every".split())


def content_stems(text, game):
    """Rough content-word fingerprint (5-letter prefixes) used to spot lines that restate each other."""
    spoken = strip_game_title(remove_citation(text), game)
    words = re.findall(r"[a-z]+", spoken.lower())
    return {w[:5] for w in words if len(w) >= 4 and w not in _SCRIPT_STOPWORDS and w != "title"}


def script_repetition(draft, game):
    """Pairs of spoken lines that restate each other: [(i, j, shared, ratio, fails)] with 1-based line numbers.
    'fails' marks near-duplicates that the mechanical check rejects; the rest are only penalised in the score."""
    lines = spoken_lines(draft)
    stems = [content_stems(content, game) for _, content in lines]
    pairs = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if "ENDING QUESTION" in (lines[i][0], lines[j][0]):
                continue   # a good closing question SHOULD reuse the story's concrete details
            smaller = min(len(stems[i]), len(stems[j]))
            if smaller < 2:
                continue
            shared = len(stems[i] & stems[j])
            ratio = shared / smaller
            fails = (ratio >= 0.75 and shared >= 3) or (shared >= 4 and ratio >= 0.5)
            if fails or (ratio >= 0.6 and shared >= 2):
                pairs.append((i + 1, j + 1, shared, round(ratio, 2), fails))
    return pairs


def build_script_evaluation_prompt(game, draft):
    return f"""You are a viewer scrolling short videos. You have never played "{game}" and know nothing about its lore beyond what this script says. Judge the script below as a ~45 second voice-over. The narration plays over gameplay footage, so the viewer SEES the characters, creatures and places while hearing it. A character's NAME is therefore clear: hearing a name over a shot of the person it belongs to tells the viewer who that is, and a plain reference such as an abusive man, a lord or a creature is clear for the same reason. Only an abstract game term that footage cannot show (a faction, a title's meaning, a lore concept, an object never seen) counts as unexplained.
Score 0-5 on:
CLARITY: after one listen, can a newcomer say who is involved, what is happening and why it matters? Every game-specific name or term must be explained in plain words.
OPENING: do the first sentence AND the first narration line make you want to keep watching (a concrete conflict, danger, choice, transformation or surprise)? A narration that starts with background (when something first appeared in the world, what kind of being it is, world history) or with a definition scores 2 at most, however good the first sentence is.
PROGRESSION: does it FLOW as one story (each line leads to the next, tension builds, a turn or reveal lands) and add NEW information each line, and does the payoff deliver what the opening promised without repeating an earlier line? A list of facts scores 1-2 however accurate, and so does a payoff that names a choice without saying concretely what each option involves and costs, or that offers the same option twice.
DELIVERY: does it sound natural spoken aloud, in short varied sentences, without padding or stacked adjectives?
Also answer REPEATS: YES if any fact is stated twice, and GENERIC_ENDING: YES if the closing question could be asked about almost any game or story, OR if its answer is obvious or already settled by the script (how someone reacted, whether a good outcome is good) so that nearly every viewer would answer the same way. A good closing question has two defensible answers.
Use the whole scale: 5 = excellent, 4 = strong, 3 = good enough to post, 2 = a clear weakness, 1 = fails at it. Judge honestly; do not reward drama or big words, and do not mark a good script down just to be strict.
One line, exactly this format (replace each <n> with your score 0-5), WHY at most 25 words naming the biggest weakness:
CLARITY=<n> OPENING=<n> PROGRESSION=<n> DELIVERY=<n> REPEATS=<YES or NO> GENERIC_ENDING=<YES or NO> WHY=<short reason>
SCRIPT:
{clean_narration(draft)}"""


_SCRIPT_EVAL_PATTERN = re.compile(
    r"CLARITY\s*=\s*([0-5])\s+OPENING\s*=\s*([0-5])\s+PROGRESSION\s*=\s*([0-5])\s+DELIVERY\s*=\s*([0-5])"
    r"(?:\s+REPEATS\s*[=:]\s*(YES|NO))?(?:\s+GENERIC_ENDING\s*[=:]\s*(YES|NO))?(?:\s+WHY\s*[=:]\s*(.*))?\s*$", re.IGNORECASE)


def evaluate_script(game, draft):
    """Separate evaluator pass (optionally a different model). Returns a dict or None if unusable after one retry."""
    prompt = build_script_evaluation_prompt(game, draft)
    for attempt in range(2):
        raw = ask_ollama(prompt, temperature=SCRIPT_EVAL_TEMPERATURE + 0.1 * attempt, model=SCRIPT_EVAL_MODEL)
        for line in raw.splitlines():
            match = _SCRIPT_EVAL_PATTERN.search(re.sub(r"[*`]", "", line.strip()))
            if match:
                return {"clarity": int(match.group(1)), "opening": int(match.group(2)), "progression": int(match.group(3)),
                        "delivery": int(match.group(4)), "repeats": (match.group(5) or "NO").upper() == "YES",
                        "generic_ending": (match.group(6) or "NO").upper() == "YES", "why": (match.group(7) or "").strip()}
    return None


def score_script(draft, game, audits, retention, coherence, problems, evaluation, missing=None, missing_payoff=None):
    """100-point score. Eligibility is a set of hard gates (verified grounding, format, retention review, hook
    coherence); the numeric score then ranks the eligible scripts. Penalties are computed in code."""
    lines = spoken_lines(draft)
    flagged = [a for a in audits if is_flagged(a)]
    review = [a for a in audits if a["status"] == "REVIEW"]
    grounded = bool(audits) and not flagged
    points = {key: (round(SCRIPT_WEIGHTS[key] * evaluation[key] / 5, 1) if evaluation else 0.0)
              for key in ("clarity", "opening", "progression", "delivery")}
    # a NEEDS_REVIEW line was not shown to be invented, only not settled: a small cost, never a block
    points["grounding"] = round(max(0.0, SCRIPT_WEIGHTS["grounding"] - min(6, 2 * len(review))), 1) if grounded else 0.0

    penalties = []
    repeated = [pair for pair in script_repetition(draft, game)]
    if repeated:
        detail = ", ".join(f"lines {i}&{j}" for i, j, *_ in repeated)
        penalties.append((f"repeated information ({detail})", min(REPETITION_CAP, REPETITION_PENALTY * len(repeated))))
    # the HOOK is fixed, so only words the writer chose after it count (one use there is still allowed)
    text = " ".join(remove_citation(content) for section, content in lines if section != "HOOK").lower()
    overuse = sum(len(re.findall(rf"\b{word}\b", text)) for word in OVERUSED_WORDS)
    if overuse > 1:
        penalties.append((f"overused adjectives ({overuse} uses)", min(OVERUSED_WORD_CAP, OVERUSED_WORD_PENALTY * (overuse - 1))))
    ending = next((remove_citation(c) for s, c in lines if s == "ENDING QUESTION"), "")
    rest = " ".join(c for s, c in lines if s != "ENDING QUESTION")
    if ending:
        generic = any(re.search(p, ending.strip().casefold()) for p in GENERIC_ENDING_PATTERNS)
        tied = len(content_stems(ending, game) & content_stems(rest, game)) >= 2
        generic = generic or bool(weak_ending_reason(ending))
        if generic or not tied or (evaluation and evaluation["generic_ending"]):
            penalties.append(("ending question is generic, obvious or not tied to this story", GENERIC_ENDING_PENALTY))
    words = word_count(draft)
    if words > SCRIPT_TARGET_MAX_WORDS + 5:
        over = (words - SCRIPT_TARGET_MAX_WORDS + 9) // 10
        penalties.append((f"{words} words is above the {SCRIPT_TARGET_MAX_WORDS}-word target", min(LENGTH_OVER_CAP, LENGTH_OVER_PENALTY * over)))

    if audits and retention is not None and not retention_ok(retention):
        penalties.append(("the retention review did not pass", RETENTION_FAIL_PENALTY))
    if coherence and coherence["verdict"] == "FAILS":
        penalties.append(("the script does not fully pay off the hook", COHERENCE_FAIL_PENALTY))
    soft = [p for p in problems if not is_structural(p)]
    if soft:
        penalties.append((f"{len(soft)} style/layout issue(s)", min(12, 3 * len(soft))))
    # (lines needing review already cost grounding points above; they are not penalised a second time here)
    if missing:
        penalties.append(("left out required point(s): " + "; ".join(f["fact"] for f in missing), min(20, 10 * len(missing))))
    if missing_payoff:
        key = [f for f in missing_payoff if is_key_payoff(f)]
        if key:   # leaving out what you do on a path, what it costs or the reveal loses the story: it cannot win
            penalties.append(("the payoff leaves out a key step (what you do, its cost or the reveal): " + "; ".join(f["fact"] for f in key),
                              min(30, 12 * len(key))))
        rest = [f for f in missing_payoff if not is_key_payoff(f)]
        if rest:
            penalties.append(("the payoff does not show how a choice plays out: " + "; ".join(f["fact"] for f in rest),
                              min(10, 5 * len(rest))))
    total = max(0.0, round(sum(points.values()) - sum(p for _, p in penalties), 1))
    blockers = []
    if not audits:
        blockers.append("no fact audit was run (earlier checks failed)")
    if flagged:
        blockers.append("claims not verified: " + "; ".join(f"{a['section']}: {a['reason']}" for a in flagged))
    # ONLY made-up claims and a broken structure (missing sections, invalid evidence ids) block. Style, length,
    # coherence, retention and uncertain lines lower the score and are reported, they never discard the story.
    blockers.extend(p for p in problems if is_structural(p))
    retention_failed = bool(audits) and not retention_ok(retention) and retention is not None
    warnings = []
    if evaluation is None and not blockers:
        warnings.append("UNSCORED: the evaluator returned no usable score, so this script ranks only on the gates")
    if not blockers and total < SCRIPT_MIN_TOTAL and evaluation is not None:
        warnings.append(f"weak: {total}/100 is below {SCRIPT_MIN_TOTAL}; review before using")
    if coherence and coherence["verdict"] == "PARTIAL":
        warnings.append("hook/script coherence is only PARTIAL")
    if coherence and coherence["verdict"] == "FAILS":
        warnings.append("the rest of the script does not fully pay off the hook")
    for p in problems:
        if not is_structural(p):
            warnings.append("style: " + p)
    if missing:
        warnings.append("the story leaves out a required point: " + "; ".join(f["fact"] for f in missing))
    if retention_failed:   # a style judgement: it lowers the score and is reported, it does not discard a verified script
        warnings.append("the retention review did not pass: " + (retention or {}).get("status", ""))
    for a in review:
        warnings.append(f"NEEDS HUMAN REVIEW: line {a['index']} [{a['section']}]: {a['reason']}")
    return {"points": points, "penalties": [{"reason": r, "points": p} for r, p in penalties], "total": total,
            "eligible": not blockers, "blockers": blockers, "warnings": warnings, "words": words,
            "justification": (evaluation or {}).get("why", "")}


def missing_evidence_note(audits, problems, coherence):
    """What the evidence did NOT support, built only from the audit results (nothing is invented or guessed)."""
    items = [f"{a['section']}: {a['reason']}" for a in audits if is_flagged(a)]
    items += [f"{a['section']} (needs review): {a['reason']}" for a in audits if a["status"] == "REVIEW"]
    if coherence and coherence["verdict"] == "FAILS":
        items.append("hook/script: " + (coherence["text"].replace("\n", " ")[:300]))
    return items


_OMIT_GENERIC = {"background", "details", "detail", "about", "involvement", "information", "backstory", "story", "other", "events"}


def _stems4(text, game=""):
    """4-letter stems of the ordinary words in text. Names (capitalised words after the first word) are ignored, so
    two unrelated details that merely share a character's name are not mistaken for the same topic."""
    spoken = strip_game_title(remove_citation(text), game) if game else text
    stems = set()
    for position, match in enumerate(re.finditer(r"[A-Za-z]+", spoken)):
        word = match.group(0)
        following = spoken[match.end():match.end() + 12]
        starts_with_name = position == 0 and (following.startswith(("'s", "\u2019s")) or re.match(r"\s+[A-Z]", following))
        if word[0].isupper() and (position > 0 or starts_with_name):
            continue
        lower = word.lower()
        if len(lower) >= 4 and lower not in _SCRIPT_STOPWORDS and lower not in _OMIT_GENERIC and lower != "title":
            stems.add(lower[:4])
    return stems


def protect_root_cause(story_brief, protected_texts, game=""):
    """The brief's DETAILS_TO_OMIT line is model-written and can list the very thing that explains the story (for
    example who caused the situation). Remove every omit-item that overlaps the verified cause facts / ROOT_CAUSE.
    Returns (brief, removed_items). General: no game or topic is named here."""
    protected = set()
    for text in protected_texts:
        protected |= _stems4(text, game)
    if not protected:
        return story_brief, []
    removed, out = [], []
    for line in story_brief.splitlines():
        match = re.match(r"^(\s*DETAILS_TO_OMIT\s*:\s*)(.*)$", line, re.IGNORECASE)
        if not match:
            out.append(line)
            continue
        items = [i.strip() for i in re.split(r"\s*(?:,|;|\band\b)\s*", match.group(2).strip().rstrip(".")) if i.strip()]
        keep = [i for i in items if not (_stems4(i, game) & protected)]
        removed += [i for i in items if i not in keep]
        out.append(match.group(1) + (", ".join(keep) + "." if keep else
                                     "only peripheral side threads; never the cause of the central situation."))
    return "\n".join(out), removed


def _omit_items(story_brief):
    match = re.search(r"(?mi)^\s*DETAILS_TO_OMIT\s*:\s*(.*)$", story_brief)
    if not match:
        return []
    return [i.strip() for i in re.split(r"\s*(?:,|;|\band\b)\s*", match.group(1).strip().rstrip(".")) if i.strip()]


def protect_causal_items(game, topic, story_brief):
    """Ask the model, item by item, whether leaving a DETAILS_TO_OMIT item out would hide a step in the chain of
    events that brought the video's subject about, or someone's responsibility for it. The answer must be 'YES:
    <how it led to the topic>'; a bare YES or a NO keeps the item omittable. Returns (brief, removed_items)."""
    removed = []
    for item in _omit_items(story_brief):
        prompt = f"""A short video covers ONE subject: "{topic}" in the game "{game}". Someone proposed leaving this detail out: "{item}".
Decide whether the detail is a step in the chain of events that brought that subject into existence (or someone's responsibility for it), so that omitting it would mislead viewers about why the subject exists.
Answer NO if the detail belongs to a different character, subplot or quest thread that the subject does not depend on, even if it appears in the same story, or if it is only background colour.
Reply NO, or reply YES: followed by one clause saying how it led to the subject.
Answer:"""
        try:
            answer = ask_ollama(prompt, temperature=0.0).strip()
        except Exception:
            continue
        if re.match(r"(?i)^yes\s*[:\-]\s*.*[a-z]{4}", answer):
            removed.append(item)
    if not removed:
        return story_brief, []
    out = []
    for line in story_brief.splitlines():
        match = re.match(r"^(\s*DETAILS_TO_OMIT\s*:\s*)(.*)$", line, re.IGNORECASE)
        if not match:
            out.append(line)
            continue
        keep = [i for i in _omit_items(line) if i not in removed]
        out.append(match.group(1) + (", ".join(keep) + "." if keep else
                                     "only peripheral side threads; never the cause of the central situation."))
    return "\n".join(out), removed


def extract_targeted_fact(game, topic, evidence, valid_ids, item):
    """The brief itself knew about a causal detail (it listed it as omittable). Find the passage that states it and
    return a code-verified cause fact, or None. Nothing is accepted unless the quote exists in the cited passage."""
    passage_map = {p["id"]: p["text"] for p in split_evidence_passages(evidence)}
    prompt = f"""The evidence may state the following, which helps explain how the subject "{topic}" came about: "{item}".
Find a passage that states it AND says what it led to or caused. Reply with ONE line:
FACT: <one plain sentence of at most 24 words stating that single cause-and-effect link, using a word such as "led to", "resulted in" or "because", and keeping who says or believes it and any uncertainty exactly as the passage does> || QUOTE: <shortest verbatim span copied from that passage> || ID: E###
Reply exactly NONE if no passage connects it to what it led to. Use only these IDs: {", ".join(sorted(valid_ids, key=lambda v: int(v[1:])))}
GAME: {game}
TOPIC: {topic}
EVIDENCE:
{evidence}"""
    raw = ask_ollama(prompt, temperature=0.0)
    for line in raw.splitlines():
        match = re.match(r"(?i)^\s*(?:[-*]\s*)?FACT\s*\d*\s*:\s*(.+?)\s*\|{1,2}\s*QUOTE\s*:\s*(.+?)\s*\|{1,2}\s*ID\s*:\s*\[?(E\d+)\]?\s*$", line.strip())
        if not match:
            continue
        fact, quote, evidence_id = strip_markdown(match.group(1)), match.group(2).strip().strip('"\u201c\u201d\'').strip(), match.group(3).upper()
        if evidence_id in valid_ids and evidence_id in passage_map and len(fact.split()) <= 32 \
                and CAUSAL_CONNECTOR.search(fact) and quote_in_text(quote, excerpt_of(passage_map[evidence_id])):
            return {"id": evidence_id, "fact": fact, "quote": quote, "kind": "cause"}
    return None


CAUSAL_CONNECTOR = re.compile(
    r"\b(led to|leading to|lead to|resulted in|result of|results from|because|caused|causes|cause of|owing to|due to|so that|"
    r"as a result|born (?:of|from)|created (?:by|from)|turned into|turns into|became|becomes|transformed into|after)\b", re.IGNORECASE)


POSSIBILITY = re.compile(r"\b(can|could|may|might|would)\s+(?:be\s+|have\s+)?\w+", re.IGNORECASE)   # an option, not what caused it


def verify_cause_facts(game, facts, passage_map):
    """A cause fact is only kept when its sentence really states a cause->effect link AND passes the same strict
    quote-verified audit as script lines. This stops a misleading paraphrase from becoming a required line.
    Returns (facts_without_failed_cause_facts, notes)."""
    kept, notes = [], []
    for fact in facts:
        if fact.get("kind") != "cause":
            kept.append(fact)
            continue
        if not CAUSAL_CONNECTOR.search(fact["fact"]):
            # still a verified fact (its quote was found), just not a cause-and-effect link: keep it as a detail
            kept.append({**fact, "kind": "detail"})
            notes.append(f"kept as a detail, not a cause (states no cause-and-effect link): [{fact['id']}] {fact['fact']}")
            continue
        result = audit_line("NARRATION", f"{fact['fact']} [{fact['id']}]", passage_map, game)
        if result["status"] == "RECITE":
            fact = {**fact, "id": result["recite_ids"][0], "ids": result["recite_ids"]}
            notes.append(f"citation corrected: {fact['fact']} -> {', '.join(result['recite_ids'])}")
            kept.append(fact)
        elif result["status"] == "CLEAR":
            kept.append(fact)
        elif result["status"] == "REVIEW":
            notes.append(f"not required (support is uncertain, so it needs review): [{fact['id']}] {fact['fact']} - {result['reason']}")
            kept.append({**fact, "kind": "detail", "uncertain": True})
        else:
            notes.append(f"dropped (no available passage supports it): [{fact['id']}] {fact['fact']} - {result['reason']}")
    return kept, notes


CHAIN_ORDER = ("cause", "account", "danger", "detail", "choice", "consequence")


def story_chain_text(facts, not_established=()):
    """The verified facts, ordered the way a story runs (cause -> situation -> choice -> what each option leads to).
    Built in code from facts that were already verified, so it adds no claim of its own."""
    if not facts:
        return ""
    ordered = sorted(facts, key=lambda f: (CHAIN_ORDER.index(f.get("kind", "detail")) if f.get("kind") in CHAIN_ORDER else 3))
    lines = [f"{n}. {f.get('kind', 'detail').upper()} [{'] ['.join(f.get('ids', [f['id']]))}]: {f['fact']}"
             + ("  (support uncertain: only as attributed)" if f.get("uncertain") else "") for n, f in enumerate(ordered, 1)]
    text = ("VERIFIED STORY CHAIN (each fact was checked against the passage shown; build the story from these links):\n"
            + "\n".join(lines) +
            "\nJoin facts with a cause word (because, so, led to, which is why) ONLY when one of the joined facts itself states that link. "
            "Facts marked ACCOUNT are someone's version of events: say whose ('the X believes', 'one account says'). "
            "If two CAUSE or ACCOUNT facts give different causes for the same event, they are competing accounts: state both and say whose each is.")
    if not_established:
        text += "\nNOT ESTABLISHED by the evidence (never state these as fact): " + "; ".join(not_established)
    return text


def verify_brief_root_cause(brief, passage_map, game):
    """The brief's ROOT_CAUSE must cite the passage that states it and pass the same audit as a script line; otherwise
    it is replaced by an explicit NOT ESTABLISHED note, so an unsupported cause cannot steer the hooks or the script.
    Returns (brief, note or None)."""
    match = re.search(r"(?mi)^(\s*ROOT_CAUSE\s*:\s*)(.+)$", brief)
    if not match or match.group(2).strip().upper().startswith(("NONE", "NOT ESTABLISHED")):
        return brief, None
    line = match.group(2).strip()
    text = remove_citation(line).strip()
    if not line_citations(line):
        ids = [i for i in re.findall(r"E\d+", line) if i in passage_map]
        line = text + " " + " ".join(f"[{i}]" for i in ids[:MAX_CITATIONS_PER_LINE]) if ids else line
    if not line_citations(line):
        result = {"status": "FLAGGED", "reason": "no passage cited"}
    else:
        result = audit_line("NARRATION", line, passage_map, game)
    if result["status"] == "RECITE":
        replacement, note = recite_line(line, result["recite_ids"]), f"ROOT_CAUSE citation corrected to {', '.join(result['recite_ids'])}"
    elif result["status"] == "CLEAR":
        replacement, note = line, None
    elif result["status"] == "REVIEW":
        replacement, note = line + " (uncertain: state it only as the evidence attributes it)", f"ROOT_CAUSE needs review: {result['reason']}"
    else:
        replacement = (f"NOT ESTABLISHED: the proposed cause ('{text}') is not stated by any available passage. "
                       "Use only the verified cause facts.")
        note = f"ROOT_CAUSE rejected: '{text}' - {result['reason']}"
    return brief[:match.start(2)] + replacement + brief[match.end(2):], note


def essential_facts(facts):
    """Verified facts the script must state about what brought the situation about, never omitted: the EVENT(S) that
    happened to the people in this story (at most two) plus ONE rule that explains why it led there, so the cause is
    always told as "what happened, and why that mattered". Without roles (the ranker did not run) the cause facts are
    used as before. At most ROOT_CAUSE_MAX (3); disable with REQUIRE_ROOT_CAUSE=0."""
    if not REQUIRE_ROOT_CAUSE:
        return []
    limit = max(0, _env_number("ROOT_CAUSE_MAX", 3))
    usable = [f for f in (facts or []) if not POSSIBILITY.search(f["fact"]) and not f.get("uncertain") and not f.get("payoff_detail")]
    if any("cause_role" in f for f in usable):
        events = [f for f in usable if f.get("cause_role") == "EVENT"]
        # what actually happened first; someone's belief or account of it only when nothing firmer is available
        belief = re.compile(r"\b(?:believes?|believed|thinks?|thought|claims?|claimed|suspects?|says|said|according to)\b", re.I)
        events = sorted(events, key=lambda f: bool(belief.search(f["fact"]) or f.get("kind") == "account"))[:2]
        rules = [f for f in usable if f.get("cause_role") == "RULE"][:1]
        return (events + rules)[:limit]
    causes = [f for f in usable if f.get("kind") in ("cause", "account") and CAUSAL_CONNECTOR.search(f["fact"])]
    if any("direct" in f for f in causes):   # rank_cause_facts ran: background causes are not required
        causes = [f for f in causes if f.get("direct")]
    return causes[:limit]


def dedupe_facts(facts):
    """The fact stages overlap, so the same fact often arrives two or three times in different words."""
    kept = []
    for fact in facts:
        stems = content_stems(_after_condition(fact["fact"]), "")
        def same(k):
            if fact.get("payoff_detail") and k.get("payoff_detail") and _option_key(fact.get("option")) != _option_key(k.get("option")):
                return False   # two different paths are never duplicates of each other
            other = content_stems(_after_condition(k["fact"]), "")
            return k["id"] == fact["id"] and len(stems & other) >= 0.7 * max(1, min(len(stems), len(other)))
        if any(same(k) for k in kept):
            continue
        kept.append(fact)
    return kept


def rank_cause_facts(game, topic, facts):
    """What role does each story fact play in how the situation came about? EVENT (what happened to or was done by the
    people in THIS story), RULE (why that event brought it about: whom the threat chooses, what it feeds on),
    BACKGROUND, RESOLUTION or OTHER. Required points are then the event(s) plus one rule, so a script can never say
    "his guilt" without saying guilt from what. Judged by the model, per fact; facts are not changed."""
    candidates = [f for f in facts if not f.get("payoff_detail") and (
        (f.get("kind") in ("cause", "account") and CAUSAL_CONNECTOR.search(f["fact"])) or f.get("kind") in ("detail", "danger"))][:16]
    if not candidates:   # even a single cause is judged: it may only be background ("how this kind of creature came to be")
        return facts
    listing = "\n".join(f"{n}. {f['fact']}" for n, f in enumerate(candidates, 1))
    ending = [f["fact"] for f in facts if f.get("payoff_detail")][:4]
    story_scope = ("THE STORY THIS VIDEO TELLS ends on this choice (anything not part of how THIS situation came about is OTHER):\n"
                   + "\n".join(f"- {e}" for e in ending)) if ending else ""
    prompt = f"""A short video explains how "{topic}" in the game "{game}" came about. For each fact below decide its role:
EVENT = something that happened to, or was done by, a person in THIS story that brought the situation about (who did what, what happened to them, including who was responsible and a different account of that same step). DIRECT means the same as EVENT.
RULE = the rule that explains why that event brought it about (OR the rule that explains why that event brought it about: which kind of person the creature, curse or threat chooses, or what it feeds on). Without it the viewer cannot see why it happened to this person. What someone must do to defeat, trick, banish or end it is never a RULE: it is RESOLUTION.
BACKGROUND = earlier history that explains why a character is the way they are (how a relationship turned bad, an old affair or killing, a childhood punishment, how someone got a title or nickname); how a whole KIND of creature, people or thing first came into the world, or what kind of being it is.
RESOLUTION = how the situation can be or was solved, or what it takes to solve it (a method, a trick, a fight, what makes it leave or end, what someone must do to defeat or banish it). That belongs to the ending, not the cause.
OTHER = any other detail of the story, AND any event from a different storyline, quest or incident than the one this video tells (even in the same game, even if it is dramatic): it is not part of how this situation came about.
{story_scope}
Reply with one line per number, exactly like "1: EVENT", "2: RULE", "3: BACKGROUND", "4: RESOLUTION" or "5: OTHER".
FACTS:
{listing}"""
    try:
        raw = ask_ollama(prompt, temperature=0.0)
    except Exception:
        return facts
    verdicts = {int(n): v.upper() for n, v in
                re.findall(r"(?im)^\s*(\d+)\s*[:.)-]\s*(EVENT|DIRECT|RULE|BACKGROUND|RESOLUTION|OTHER)", raw)}
    if not verdicts:
        return facts
    for n, f in enumerate(candidates, 1):
        role = {"DIRECT": "EVENT"}.get(verdicts.get(n, ""), verdicts.get(n, ""))
        if not role:
            role = "EVENT" if f.get("kind") in ("cause", "account") else "OTHER"
        f["cause_role"] = role
        f["direct"] = role in ("EVENT", "RULE")
    # code, not the model, has the last word on scope: an event is part of THIS story only if it involves someone
    # (or something) from the story's own ending. An event from another quest in the same evidence names no one there.
    ending_text = " ".join(f["fact"] for f in facts if f.get("payoff_detail"))
    events = [f for f in candidates if f.get("cause_role") == "EVENT"]
    if events:
        listing = "\n".join(f"{n}. {f['fact']}" for n, f in enumerate(events, 1))
        question = f"""A short video tells the story of "{topic}" in the game "{game}"{(", which ends on this choice: " + ending_text) if ending_text else ""}.
For each event below answer BEFORE if it happened before the situation began and helped bring it about, or ENDING if it is part of how the situation is dealt with, solved or ends.
Reply one line per number, exactly like "1: BEFORE" or "2: ENDING".
EVENTS:
{listing}"""
        try:
            timing = {int(n): v.upper() for n, v in re.findall(r"(?im)^\s*(\d+)\s*[:.)-]\s*(BEFORE|ENDING)", ask_ollama(question, temperature=0.0))}
        except Exception:
            timing = {}
        for n, f in enumerate(events, 1):
            if timing.get(n) == "ENDING":
                f["cause_role"], f["direct"] = "RESOLUTION", False
    if ending_text:
        for f in candidates:
            if f.get("cause_role") == "EVENT" and not (connected_to_story(f["fact"], ending_text, game)
                                                       or mentions_topic(f["fact"], topic, game)):
                f["cause_role"], f["direct"] = "OTHER", False
                f["scope_note"] = "involves no one from this story's ending"
    return facts


def _story_names(text, game=""):
    title = {w.casefold() for w in re.findall(r"[A-Za-z]+", game or "")}
    return {w.casefold().rstrip("’'s") for w in re.findall(r"\b[A-Z][a-z’']{2,}", text or "")
            if w.casefold() not in _NAME_STOPWORDS and w.casefold() not in title}


def connected_to_story(fact, story_text, game=""):
    """True when a fact shares a named person, creature or place with the story text, or at least two content words."""
    if _story_names(fact, game) & _story_names(story_text, game):
        return True
    return len(content_stems(fact, game) & content_stems(story_text, game)) >= 2


def mentions_topic(fact, topic, game=""):
    """True when a fact names the topic itself (the creature, person or place the video is about)."""
    title = {w.casefold() for w in re.findall(r"[A-Za-z]+", game or "")}
    words = lambda t: {w.casefold().rstrip("s") for w in re.findall(r"[A-Za-z]{3,}", t or "")}
    key = words(topic) - title - {w.rstrip("s") for w in _NAME_STOPWORDS}
    return bool(key & words(fact))


def missing_essential(draft, essential, game):
    """Essential facts that no HOOK/NARRATION/PAYOFF line states: it must share most of the fact's wording and cite
    the fact's passage (or match very closely)."""
    missing = []
    lines = [(section, content) for section, content in spoken_lines(draft) if section != "ENDING QUESTION"]
    for fact in essential or []:
        fact_stems = content_stems(fact["fact"], game)
        if not fact_stems:
            continue
        need = max(2, -(-len(fact_stems) * 3 // 10))           # ceil(30%) when the line cites the fact's passage
        close = max(need, -(-len(fact_stems) * 1 // 2))        # ceil(50%) otherwise
        satisfied = False
        for _, content in lines:
            shared = len(fact_stems & content_stems(content, game))
            cites_fact = any(i in line_citations(content) for i in fact.get("ids", [fact["id"]]))
            if (cites_fact and shared >= need) or shared >= close:
                satisfied = True
                break
        if not satisfied:
            missing.append(fact)
    return missing


def plan_story_beats(game, topic, selected_hook, facts, essential, structure=None):
    """The writer gets a short TIMELINE, not a pile of facts: the model picks the 4-6 facts the story needs and puts
    them in the order they HAPPENED (stories that jump back and forth read as lists), plus the payoff. Code enforces it:
    required facts are kept, nothing is used twice, nothing repeats the hook. Each beat carries the source words of its
    passage so the writer has concrete detail without being handed every passage. Returns the timeline text."""
    usable = [f for f in facts if not _restates(f["fact"], selected_hook, game)]
    if len(usable) < 2:
        return ""
    listing = "\n".join(f"{n}. ({f.get('kind', 'detail')}) {f['fact']}" for n, f in enumerate(usable, 1))
    prompt = f"""You are planning a 30-60 second story about "{topic}" in the game "{game}". The opening line is fixed: {remove_citation(selected_hook)}
Pick the 4-6 facts the story needs and list them in the order the story should be TOLD:
- FIRST, the fact that gives a stranger the strongest reason to keep watching right after the opening line: what is happening to someone, the danger, what this does to its victims, who suffers. Never a background fact.
- THEN how it came to be: the causes and events in the order they HAPPENED, building to the turn.
Prefer what the characters in THIS story actually do (who does what, to whom) over general rules about how such things work; use a general rule only when no fact shows it happening.
Background (when a creature or people first appeared in the world, what kind of being something is, general world history) never comes first; leave it out unless the story cannot be understood without it. Leave out side plots and a character's older history that this story does not need. Use each fact once.
Then pick the PAYOFF: for EACH option of the choice, the fact that says concretely what is actually done (the real scene: who does what, to whom) and the fact that says what it leads to or costs. That is the answer the opening line makes the viewer want; a general rule ("use a trick", "fight it") is not enough on its own.
Reply with exactly two lines, numbers only:
TIMELINE: 3, 1, 4, 2
PAYOFF: 6, 7
FACTS:
{listing}"""
    try:
        raw = ask_writer(prompt, temperature=0.1)
    except Exception:
        raw = ""
    used, timeline, payoff = set(), [], []
    for name, target in (("TIMELINE", timeline), ("PAYOFF", payoff)):
        match = re.search(rf"(?im)^\s*{name}\s*:\s*(.+)$", raw)
        for n in [int(x) for x in re.findall(r"\d+", match.group(1))] if match else []:
            if 1 <= n <= len(usable) and n not in used:
                target.append(usable[n - 1])
                used.add(n)
    if timeline and timeline[0].get("cause_role") in ("RULE", "BACKGROUND", "OTHER"):
        # the narration opens on the people, never on a definition: the first event moves to the front
        event = next((f for f in timeline if f.get("cause_role") == "EVENT"), None) or next(
            (f for f in usable if f.get("cause_role") == "EVENT" and f not in payoff), None)
        if event:
            if event in timeline:
                timeline.remove(event)
            timeline.insert(0, event)
    for fact in essential or []:   # a required cause can never be planned out; it goes after the opener, not before it
        if fact in usable and fact not in timeline and fact not in payoff:
            timeline.insert(1 if timeline else 0, fact)
    if not payoff:
        payoff = [f for f in usable if f.get("kind") == "consequence" and f not in timeline][:2]
    # the concrete scene of each option must reach the payoff even when the plan picked only the rule behind it
    details = [f for f in usable if f.get("payoff_detail") and f not in payoff and f not in timeline]
    payoff += details[:max(0, 6 - len(payoff))]   # the scene of each option, its cost and the twist that makes it work
    order = list(dict.fromkeys(_option_key(x.get("option")) for x in payoff))   # keep each path's facts together
    with_reveal = {_option_key(f.get("option")) for f in payoff if f.get("beat") == "twist"}
    order.sort(key=lambda k: k in with_reveal)          # the path with a reveal is told last, so the video ends on it
    position = {id(f): n for n, f in enumerate(usable)}
    beat_rank = {"risk": 1, "twist": 2, "outcome": 3}   # what is done, what it costs, the reveal, what it leads to
    payoff.sort(key=lambda f: (order.index(_option_key(f.get("option"))),
                               0 if f.get("kind") == "choice" else beat_rank.get(f.get("beat"), 1), position.get(id(f), 0)))
    if not timeline and not payoff:
        return ""

    def beat(f):
        quote = f.get("quote", "")
        return f"[{'] ['.join(f.get('ids', [f['id']]))}] {f['fact']}" + (f'  (source words: {quote})' if quote and len(quote) < 300 else "")
    lines = ["STORY TIMELINE (verified facts in the order to tell them: the opener first, then the events in the order they happened; each line causing the next):"]
    lines += [f"{n}. {beat(f)}" for n, f in enumerate(timeline, 1)]
    grouped = {}
    for f in payoff:
        if f.get("option"):
            grouped.setdefault(_option_key(f["option"]), []).append(f)
    if len(grouped) >= 2:
        lines.append(f"PAYOFF: THE CHOICE ({len(grouped)} different paths the player can take; tell each one as its own option, "
                     "with what the player does differently, what really happens and what it leads to or costs; never split "
                     "one path into two options, never tell the same option twice; tell them in this order, and end the "
                     "PAYOFF on the last path's reveal):")
        for n, group in enumerate(grouped.values(), 1):
            lines.append(f"OPTION {n}: {group[0]['option']}")
            lines += [f"  - {'REVEAL (save for the very end): ' if f.get('beat') == 'twist' else 'COST: ' if f.get('beat') == 'risk' else ''}{beat(f)}"
                      for f in group]
        lines += [f"- {beat(f)}" for f in payoff if not f.get("option")]
    elif payoff:
        lines.append("PAYOFF (end on this, told as the player's choice: for each option, what is actually done and what it leads to or costs):")
        lines += [f"- {beat(f)}" for f in payoff]
    rule = next((f for f in list(timeline) + list(essential or []) if f.get("cause_role") == "RULE"), None)
    if structure is not None:   # the same plan as data, for the line-by-line writer
        structure.update(timeline=list(timeline), paths=[(g[0]["option"], list(g)) for g in grouped.values()] if len(grouped) >= 2 else [],
                         loose=[f for f in payoff if not f.get("option")] if len(grouped) >= 2 else list(payoff), rule=rule)
    if rule and any(f.get("beat") == "twist" for f in payoff):
        lines.append(f"SET UP THE REVEAL: the rule \"{rule['fact']}\" is what makes the last path work. State it BEFORE the choice, "
                     "in the form the reveal needs, and say why it applies to the person in the story (if what they did does "
                     "not obviously fit the rule, say the bridge the material gives: what they believe or blame themselves for).")
    lines.append("The HOOK already told the viewer its premise; no later line may say it again.")
    lines.append("Tell what really happens: if a detail shows that something only LOOKS terrible (it was a trick, no one was really harmed), the script must say so; never let the viewer believe the terrible version happened.")
    lines.append("If something works because someone is made to BELIEVE something, say it in that order: who is fooled into believing what, what that belief causes, then what happens when the truth comes out. Never say the fooled person does not believe it.")
    return "\n".join(lines)


_PLAN_WORDS = {"if", "option", "cost", "reveal", "payoff", "story", "timeline", "set", "the", "hook", "tell", "never", "state",
               "before", "source", "words", "choice", "player", "fixed", "plan", "it", "some", "others"}


def _plan_names(plan, game):
    """Named people (and creatures) in the plan: capitalised words that are not plan labels or the game title."""
    title = {w.casefold() for w in re.findall(r"[A-Za-z]+", game or "")}
    names = []
    for word in re.findall(r"\b([A-Z][a-z]{2,})(?:[’']s)?\b", plan or ""):
        key = word.casefold()
        if key in title or key in _PLAN_WORDS or key in _NAME_STOPWORDS or word in names:
            continue
        names.append(word)
    return names


def player_character(plan):
    """The name the material gives the player's character: the subject of 'If the player ..., <Name> ...' facts."""
    found = re.findall(r"If the player[^,.\n]*,\s+(?:then\s+)?([A-Z][a-z]{2,})\b", plan or "")
    return max(set(found), key=found.count) if found else ""


_KIN = (r"brother|sister|son|daughter|father|mother|wife|husband|uncle|aunt|niece|nephew|cousin|friend|servant|guard|lover|"
        r"king|queen|lord|lady|captain|leader|master|apprentice|mentor|rival|child|bride|widow|heir|advisor|adviser|steward|"
        r"priest|healer|witch|druid|mage|knight|chief|ruler|commander|soldier|squire|ward|twin|grandson|granddaughter")
_TITLES = r"Jarl|King|Queen|Lord|Lady|Baron|Baroness|Count|Countess|Duke|Duchess|Prince|Princess|Emperor|Empress|Captain|Sir|Chief|General|Commander|Elder"


def _role_in(name, text, known=(), level=None):
    """Who the evidence says this person is, copied verbatim, or ''. Recognises 'Orla Venn, daughter of the lord',
    'Tomas, the lord's brother', 'Tomas is the ...', 'his brother Tomas' and a title ('Lord Tomas'). level picks one
    pattern (0 = strongest). A 'his/her X' role is only used when the person it belongs to is a named person of the
    story (known): 'his daughter Orla' in a sentence about someone else would invent a relationship."""
    n = re.escape(name)
    not_object = r"(?<!\bto )(?<!\bby )(?<!\bwith )(?<!\bfrom )(?<!\bfor )(?<!\bof )(?<!\bAccording to )"
    patterns = (
        rf"{not_object}\b{n}(?:\s+(?:(?:an|af|of|von|van|de|al)\s+)?[A-Z][a-z]+)?,\s+((?:(?:the|a|an)\s+[^,.;:()\n]{{2,60}}?)|(?:(?:{_KIN})\s+of\s+[^,.;:()\n]{{2,50}}?))\s*[,.;)]",
        rf"\b{n}\s+(?:is|was)\s+((?:the|a|an)\s+(?:[a-z’'-]+\s+){{0,3}}(?:{_KIN}))\b",
        rf"\b((?:the)\s+(?:[a-z’'-]+\s+){{0,2}}(?:{_KIN})),?\s+{n}\b",
        rf"\b((?:his|her|their)\s+(?:[a-z’'-]+\s+){{0,2}}(?:{_KIN})),?\s+{n}\b",
        rf"\b((?:{_TITLES}))\s+{n}\b",
    )
    for number, pattern in enumerate(patterns):
        if level is not None and number != level:
            continue
        match = re.search(pattern, text)
        if not match or len(match.group(1).split()) > 9:
            continue
        role = match.group(1).strip()
        if number == 3:   # "his daughter X": whose is a guess, and a wrong guess invents a relationship
            continue
        return role
    return ""


def cast_block(game, plan, passage_map):
    """WHO IS WHO for the writer: for each person the plan names, the words the evidence uses to say who they are
    (found in code, copied verbatim, so introducing someone never means inventing a relationship). The player's
    character is told as "you". A person the evidence never introduces is described only by what they do."""
    hero = player_character(plan)
    texts = [(i, excerpt_of(t)) for i, t in (passage_map or {}).items()]
    every = " ".join(t for _, t in texts)
    # a word the evidence also writes in lower case is a thing or a kind of creature ("wraith", "trust"), not a person
    names = [n for n in _plan_names(re.sub(r"(?m)^\s*OPTION \d+:.*$", "", plan), game)
             if n == hero or not re.search(rf"\b{re.escape(n.lower())}s?\b", every)]
    if not names:
        return ""
    lines = ["WHO IS WHO (from the evidence; introduce each person with these words the first time they are named, and with nothing invented):"]
    for name in names[:8]:
        if name == hero:
            lines.append(f"- {name} is the PLAYER'S CHARACTER: when {name} acts on a path, say \"you\"; never write the name {name}.")
            continue
        found = next(((i, t, r) for level in range(5) for i, t in texts for r in [_role_in(name, t, names, level)] if r), None)
        if found:
            evidence_id, text, role = found
            sentence = next((x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if role.split("'s ")[-1] in x and name in x), "")
            lines.append(f"- {name}: {role}" + (f'  (source {evidence_id}: "{sentence[:200]}")' if sentence else f" (source {evidence_id})"))
        else:
            lines.append(f"- {name}: the evidence does not say who they are; describe them only by what they do here "
                         "(their action in this story, in your own words), or leave the name out.")
    return "\n".join(lines)


# THE LINE-BY-LINE WRITER (WRITER_MODE=lines). A small local model loses rules when it must hold the whole script in its
# head, but it writes one good sentence from one or two facts. So the CODE builds the script's shape from the verified
# plan (opener, cause in order, the rule and why it fits, each path with its cost, the payoff ending on the reveal),
# and the model writes one line at a time, seeing the lines before it. Order, "you" for the player's character, who is
# who and every path's cost come from the shape, so they cannot go missing. The classic writer is unchanged.
WRITER_MODE = os.environ.get("WRITER_MODE", "classic").strip().lower()   # classic | lines
LOCAL_WRITER_MODEL = os.environ.get("LOCAL_WRITER_MODEL", "").strip()     # blank = the main model; research and checks never change
LINE_TEMPERATURE = 0.6


def script_skeleton(game, structure, hero=""):
    """The lines a script needs, in order: [{section, job, facts}]. Built in code from the plan's data:
    opener -> how it came to be (in order) -> the rule, once, and why it fits this person -> each path with its
    cost -> the payoff ending on the reveal."""
    timeline = list(structure.get("timeline") or [])
    paths = list(structure.get("paths") or [])
    loose = list(structure.get("loose") or [])
    rules = [f for f in timeline if f.get("cause_role") == "RULE"]
    if structure.get("rule") and structure["rule"] not in rules:
        rules.append(structure["rule"])
    events = [f for f in timeline if f not in rules]
    you = f' (the player\'s character, {hero}, is always "you")' if hero else ' (the player is always "you")'
    if len(paths) >= 2 and loose:   # a fact the plan left unattached ("fighting it needs the host there") joins its path
        paths = [(o, list(fs)) for o, fs in paths]
        for fact in list(loose):
            words = content_stems(fact["fact"], game)
            overlap = [len(words & content_stems(o + " " + " ".join(f["fact"] for f in fs), game)) for o, fs in paths]
            if max(overlap) >= 1 and overlap.count(max(overlap)) == 1:
                best = overlap.index(max(overlap))
                cost = {**fact, "beat": fact.get("beat") or "risk"}
                fs = paths[best][1]
                fs.insert(next((k for k, f in enumerate(fs) if f.get("beat") in ("twist", "outcome")), len(fs)), cost)
                loose.remove(fact)
    lines = []
    if events:
        lines.append({"section": "NARRATION", "facts": [events[0]],
                      "job": "Right after the hook: make its danger bigger and concrete (what this does to its victim, "
                             f"who is suffering), and name the threat. Name the game, {game_title_variants_raw(game)}, "
                             "naturally in this line. If a fact is only what someone believes, say whose belief it is. "
                             "Never open on background."})
    for k in range(1, len(events), 2):   # the cause in the order it happened, two facts a line at most
        lines.append({"section": "NARRATION", "facts": events[k:k + 2],
                      "job": "How it came to be: tell these events plainly, in the order they happened. If a fact is "
                             "only what someone believes, say whose belief it is."})
    if rules:   # the rule once (all its facts together), after the cause, so it explains what the viewer just heard
        lines.append({"section": "NARRATION", "facts": rules[:3],
                      "job": "The rule: name the threat and whom it chooses, and why it chose THIS person. If what they "
                             "did does not obviously fit the rule, say the bridge the facts give (what they believe or "
                             "blame themselves for). One idea: do not list the rule twice."})
    if len(paths) >= 2:
        *first, last = paths
        for n, (option, facts) in enumerate(first):
            lines.append({"section": "NARRATION", "facts": facts,
                          "job": f"{'The choice. ' if n == 0 else ''}One path{you}: '{option}'. Say what you actually do on "
                                 "this path and what it costs or risks for people. Make clear who does each act. "
                                 + ("Start with 'You can'." if n == 0 else "Start with 'Or'.")})
        option, facts = last
        before = [f for f in facts if f.get("beat") not in ("twist", "outcome")]
        after = [f for f in facts if f.get("beat") in ("twist", "outcome")]
        if before and after:
            lines.append({"section": "NARRATION", "facts": before[:2],
                          "job": f"The other path{you}: '{option}'. Start with 'Or'. Say what you actually do (make clear "
                                 "that YOU do it, not the person you trust) and why it would work. Do not reveal how it turns out."})
            lines.append({"section": "PAYOFF", "facts": before[2:] + after,
                          "job": "The payoff: what happens next on this path, in order, ending on the reveal (what really "
                                 "happened) and what it leads to. Two or three short sentences."})
        else:
            lines.append({"section": "PAYOFF", "facts": facts,
                          "job": f"The payoff, the other path{you}: '{option}'. Start with 'Or'. What you do and what "
                                 "it leads to, in order. Two or three short sentences."})
        if loose:
            lines[-1]["facts"] = lines[-1]["facts"] + loose[:1]
    elif loose:
        lines.append({"section": "PAYOFF", "facts": loose[:4],
                      "job": f"The payoff the hook promised{you}: what is done and what it leads to or costs, in order."})
    return lines


def _facts_text(facts):
    return "\n".join(f"- {f['fact']}" + (f"  (source words: {f['quote']})" if f.get("quote") and len(f["quote"]) < 300 else "")
                     for f in facts)


def build_line_prompt(game, topic, so_far, item, cast, later, max_words, feedback=""):
    names = set(re.findall(r"\b[A-Z][a-z]{2,}\b", " ".join(f["fact"] for f in item["facts"])))
    told = " ".join(so_far[1:])   # the hook may name someone without saying who they are
    cast_lines = [l for l in (cast or "").splitlines()[1:]
                  if any(l.startswith(f"- {n}") and (n not in told or "PLAYER'S CHARACTER" in l) for n in names)]
    known = sorted(n for n in names if n in told)
    if known:
        cast_lines.append("- Already introduced (just use the name, do not say who they are again): " + ", ".join(known))
    script = "\n".join(so_far)
    return f"""You are writing ONE line of the voice-over for a short video about "{topic}" in the game "{game}". The viewer has never played it.
THE SCRIPT SO FAR:
{script}
THIS LINE'S JOB: {item['job']}
FACTS FOR THIS LINE (use only these; invent nothing, not even a relationship):
{_facts_text(item['facts'])}
{("WHO IS WHO (the first time someone is named in the script, say who they are with these words):" + chr(10) + chr(10).join(cast_lines) + chr(10)) if cast_lines else ""}{("KEEP FOR LATER (do not mention or hint at yet): " + "; ".join(later) + chr(10)) if later else ""}{("FIX: " + feedback + chr(10)) if feedback else ""}HOW: never say he, she or they for someone the script has not named yet (name them, or say who); one or two complete sentences that carry straight on from the last line (a turn word such as but, so, until or then is welcome, but the line must still make sense on its own); never repeat what the script already said; everyday spoken words; short sentences; at most {max_words} words. No quotation marks, no labels.
Reply with the line only."""


def _clean_line(text):
    text = strip_markdown(text or "").strip()
    text = re.sub(r"^\s*(?:NARRATION|PAYOFF|LINE|HOOK)\s*\d*\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"[\"\u201c\u201d\u201e\u201f\u00ab\u00bb]", "", text.splitlines()[0] if text else "").strip()
    text = re.sub(r"^(Or|But|So|Then)\s*(?:\.\.\.|…)\s*", r"\1 ", text)   # "Or... trust" -> "Or trust"
    return text[:1].upper() + text[1:] if text else ""


LINES_MAX_PER_PATH = 5
LINES_MAX_ATTEMPTS = _env_number("LINES_MAX_ATTEMPTS", 1)   # flagged lines are repaired one by one; a full rewrite rarely helps
LINES_EXTRA_EVENTS = 1   # planner-picked events beyond the required cause (background crowds out the cause)


LINES_BRIDGE_EVENTS = 3


def _same_fact(a, b):
    """Two facts that say the same thing in different words ('a wraith latched onto the lord' twice)."""
    x, y = content_stems(a["fact"], ""), content_stems(b["fact"], "")
    return bool(x and y) and len(x & y) >= 0.7 * min(len(x), len(y))


def pick_bridge_events(candidates, rules, fallback=True):
    """The events that show WHY the rule applies to this person (what they did, or are believed to have done):
    without them the viewer hears the rule ('it feeds on the guilty') but never the guilt. One small local call."""
    if not candidates:
        return []
    listing = "\n".join(f"{n}. {f['fact']}" for n, f in enumerate(candidates[:10], 1))
    prompt = f"""The story's rule: {' '.join(r['fact'] for r in rules)}
Which of these events show why that rule applies to the person it happens to (what they did, or what people believe they did, that makes them fit the rule)? Pick at most {LINES_BRIDGE_EVENTS}, in the order they happened. If you pick what people believe about an event, also pick the event itself, so the viewer knows what happened. Ignore older history that does not explain it.
Reply with the numbers only, like: 2, 4
EVENTS:
{listing}"""
    try:
        raw = ask_ollama(prompt, temperature=0.0)
        picked = [candidates[int(n) - 1] for n in dict.fromkeys(re.findall(r"\d+", raw)) if 1 <= int(n) <= min(10, len(candidates))]
    except Exception:
        picked = []
    return picked[:LINES_BRIDGE_EVENTS] or (candidates[:LINES_EXTRA_EVENTS] if fallback else [])


def lines_structure(planned, facts, essential):
    """The line writer's shape, built from every verified fact rather than the planner's pick (which is chosen
    afresh each time and can drop the reveal or a path's cost):
    opener (the planner's, unless it is a rule) -> the REQUIRED cause, in order -> at most one more event ->
    the rule(s) -> every path with ALL its facts (what you do, its cost, what happens, the reveal last), the path
    with a reveal told last."""
    planned = planned or {}
    essential = list(essential or [])
    rules = [f for f in essential if f.get("cause_role") == "RULE"]
    for f in planned.get("timeline") or []:
        if f.get("cause_role") == "RULE" and f not in rules and len(rules) < 2:
            rules.append(f)
    timeline = [f for f in planned.get("timeline") or [] if f.get("cause_role") != "RULE"]
    opener = timeline[:1]
    required = [f for f in essential if f not in rules and f not in opener]
    kept = []
    for f in opener + required + timeline[1:] + [f for f in facts or [] if f.get("cause_role") == "EVENT"]:
        if f not in kept and not any(_same_fact(f, k) for k in kept + rules):
            kept.append(f)
    candidates = [f for f in kept if f not in opener and f not in required and not f.get("option")]
    extra = pick_bridge_events(candidates, rules) if rules else candidates[:LINES_EXTRA_EVENTS]
    required = [f for f in required if not any(_same_fact(f, o) for o in opener)]
    groups = {}
    for f in facts or []:
        if f.get("payoff_detail") and f.get("option"):
            groups.setdefault(_option_key(f["option"]), []).append(f)
    rank = {"risk": 1, "twist": 3, "outcome": 4}
    paths = []
    for key, group in groups.items():
        unique = []
        for f in group:   # the same step told twice in different words is one step
            stems = content_stems(f["fact"], "")
            if not any(len(stems & content_stems(u["fact"], "")) >= 0.7 * max(1, min(len(stems), len(content_stems(u["fact"], "")))) for u in unique):
                unique.append(f)
        group = unique
        ordered = sorted(group, key=lambda f: (0 if f.get("kind") == "choice" else rank.get(f.get("beat"), 2), group.index(f)))
        paths.append((group[0]["option"], ordered[:LINES_MAX_PER_PATH]))
    paths.sort(key=lambda p: any(f.get("beat") == "twist" for f in p[1]))   # the reveal path last
    if len(paths) < 2:
        paths = list(planned.get("paths") or [])
    loose = [f for f in planned.get("loose") or [] if not f.get("option")]
    return {"timeline": opener + required + extra + rules, "paths": paths, "loose": loose, "rule": rules[0] if rules else None}


def write_by_lines(game, topic, selected_hook, structure, cast, feedback="", keep=None):
    """One model call per line, in order, each seeing the script so far. Returns a draft in the usual layout.
    With keep={}, it is filled with what each line was written from, so a line can be rewritten on its own later."""
    hero = player_character(cast_hero_hint(structure))
    skeleton = script_skeleton(game, structure, hero)
    if not skeleton:
        return ""
    budget = max(60, SCRIPT_TARGET_MAX_WORDS - len(remove_citation(selected_hook).split()) - 18)   # 18 for the question
    reveal = [f["fact"] for _, facts in (structure.get("paths") or []) for f in facts if f.get("beat") == "twist"]
    so_far, out = [f"HOOK: {remove_citation(selected_hook)}"], []
    for n, item in enumerate(skeleton):
        share = max(14, budget // len(skeleton)) + (12 if item["section"] == "PAYOFF" else 0)
        later = reveal if item["section"] != "PAYOFF" and not any(f.get("beat") == "twist" for f in item["facts"]) else []
        prompt = build_line_prompt(game, topic, so_far, item, cast, later, share, feedback)
        line = ""
        for attempt in range(2):
            line = _clean_line(ask_writer(prompt, temperature=LINE_TEMPERATURE + 0.1 * attempt))
            if hero and re.search(rf"\b{re.escape(hero)}\b", line):
                prompt += f'\nFIX: you wrote the name {hero}; the player\'s character is "you". Write it again with "you".'
                continue
            if line:
                break
        if not line:
            continue
        must = [f for f in item["facts"] if f.get("beat") in ("twist", "risk")]   # a path's cost and its reveal are never optional
        if line and must and not all(len(content_stems(f["fact"], game) & content_stems(line, game)) >= 2 for f in must):
            retry = _clean_line(ask_writer(prompt + "\nFIX: you left out: " + "; ".join(f["fact"] for f in must)
                                           + ". Say it plainly (end on the reveal if there is one).", temperature=LINE_TEMPERATURE))
            if retry and all(len(content_stems(f["fact"], game) & content_stems(retry, game)) >= 2 for f in must):
                line = retry
        so_far.append(f"{item['section']}: {line}")
        out.append((item["section"], line))
        if keep is not None:
            keep.setdefault("items", []).append({**item, "max_words": share, "later": later})
    if not any(sec == "PAYOFF" for sec, _ in out) and out:
        out[-1] = ("PAYOFF", out[-1][1])
    paths = structure.get("paths") or []
    fallback_q = "Which path would you choose?" if len(paths) >= 2 else "Was it worth it?"   # replaced by the closing-question step
    return rebuild_draft("", [("HOOK", selected_hook)] + out + [("ENDING QUESTION", fallback_q)])


LINE_POLISH_ROUNDS = _env_number("LINE_POLISH_ROUNDS", 2)   # weakest line rewritten, kept only if the script scores higher

_POLISH_JUDGE = """You are a strict editor of 30-60 second YouTube Shorts. Read this voice-over script, numbered by line.
Give the whole script one score from 1 to 10 for how well it holds a stranger who never played the game (hook, suspense, clarity, flow, payoff).
Then name the ONE weakest line among lines {first}-{last} (the line that most breaks the flow, confuses, repeats, or wastes time) and say in at most 15 words how to fix it.
Reply with exactly one line: SCORE=<n> WEAKEST=<line number> FIX=<how to fix it>
SCRIPT:
{script}"""


def _judge_for_polish(lines, first, last):
    numbered = "\n".join(f"{n}. {sec}: {remove_citation(c)}" for n, (sec, c) in enumerate(lines, 1))
    try:
        raw = ask_ollama(_POLISH_JUDGE.format(first=first, last=last, script=numbered), temperature=0.0, model=SCRIPT_EVAL_MODEL or MODEL)
    except Exception:
        return None
    m = re.search(r"SCORE\s*=\s*(\d+).*?WEAKEST\s*=\s*(\d+)(?:.*?FIX\s*=\s*(.+))?", strip_markdown(raw), re.S | re.I)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), (m.group(3) or "").strip().splitlines()[0] if m.group(3) else ""


def polish_lines(game, topic, draft, keep, cast, rounds=None):
    """OpenReels-style targeted revision: a judge names the weakest line, only that line is rewritten (from its own
    facts, seeing the lines around it), and the script is kept only if it then scores HIGHER: a later rewrite can
    make things worse, so the best version wins, not the last one. A few small calls; facts are still checked after."""
    rounds = LINE_POLISH_ROUNDS if rounds is None else rounds
    items = (keep or {}).get("items") or []
    lines = spoken_lines(draft)
    body = [i for i, (sec, _) in enumerate(lines) if sec in ("NARRATION", "PAYOFF")]
    if rounds <= 0 or len(items) != len(body) or not body:
        return draft
    verdict = _judge_for_polish(lines, body[0] + 1, body[-1] + 1)
    if not verdict:
        return draft
    best_score, best = verdict[0], list(lines)
    print(f"Polish: the judge scores the draft {best_score}/10.")
    for _ in range(rounds):
        score, weakest, fix = verdict
        index = weakest - 1
        if index not in body:
            break
        item = items[body.index(index)]
        so_far = [f"{sec}: {remove_citation(c)}" for sec, c in best[:index]]
        after = remove_citation(best[index + 1][1]) if index + 1 < len(best) else ""
        prompt = build_line_prompt(game, topic, so_far, item, cast, item.get("later") or [], item.get("max_words", 30),
                                   (fix or "make this line stronger") + (f". The next line is: {after}" if after else ""))
        try:
            line = _clean_line(ask_writer(prompt, temperature=LINE_TEMPERATURE))
        except Exception:
            break
        if not line:
            break
        trial = list(best)
        trial[index] = (best[index][0], line)
        verdict = _judge_for_polish(trial, body[0] + 1, body[-1] + 1)
        if not verdict:
            break
        if verdict[0] > best_score:
            print(f"Polish: rewrote line {weakest} ({fix or 'weakest line'}); the script now scores {verdict[0]}/10.")
            best_score, best = verdict[0], trial
        else:
            print(f"Polish: a rewrite of line {weakest} did not score higher ({verdict[0]}/10); kept the earlier version.")
            break
    return rebuild_draft(draft, best)


def cast_hero_hint(structure):
    """Text holding the plan's 'If the player ..., <Name> ...' facts, to find the player's character's name."""
    return "\n".join(f["fact"] for _, facts in (structure.get("paths") or []) for f in facts) + "\n" + \
           "\n".join(f["fact"] for f in (structure.get("loose") or []))


def merge_payoffs(draft):
    """Two PAYOFF lines are one payoff told in two breaths: join them instead of discarding the whole script."""
    lines = spoken_lines(draft)
    if sum(1 for sec, _ in lines if sec == "PAYOFF") < 2:
        return draft
    merged, out = " ".join(remove_citation(c).strip() for sec, c in lines if sec == "PAYOFF"), []
    for sec, c in lines:
        if sec == "PAYOFF":
            if merged is not None:
                out.append(("PAYOFF", merged))
                merged = None
            continue
        out.append((sec, c))
    return rebuild_draft(draft, out)


def tidy_writer_output(draft):
    """Formatting only (never content): drop markdown and quotation marks the writer may use."""
    draft = re.sub(r"[*_`#]+", "", draft)
    return re.sub(r"[\"\u201c\u201d\u201e\u201f\u00ab\u00bb]", "", draft)


def missing_essential_semantic(game, draft, essential):
    """Does the script state each required point in ANY words? Judged by the model (wording-overlap matching rejected
    good stories that told the cause in their own words); falls back to the overlap check if the reply is unusable."""
    if not essential:
        return []
    listing = "\n".join(f"{n}. {f['fact']}" for n, f in enumerate(essential, 1))
    prompt = f"""Read this short video script. For each numbered point, answer YES if the script tells the viewer that point (in any words, possibly attributed to someone), otherwise NO.
Reply one line per number, exactly like "1: YES".
POINTS:
{listing}
SCRIPT:
{clean_narration(draft)}"""
    try:
        raw = ask_ollama(prompt, temperature=0.0)
        answers = {int(n): v.upper() for n, v in re.findall(r"(?im)^\s*(\d+)\s*[:.)-]\s*(YES|NO)", raw)}
    except Exception:
        answers = {}
    if len(answers) < len(essential):
        return missing_essential(draft, essential, game)
    return [f for n, f in enumerate(essential, 1) if answers.get(n) == "NO"]


ENDING_CANDIDATES = _env_number("ENDING_CANDIDATES", 6)   # closing questions written per draft; the best one is kept


def paths_block(paths):
    """The paths and their costs, for the closing question: 'Would you [what path 1 asks of you], or [path 2] and risk [its cost]?'"""
    if not paths:
        return ""
    lines = []
    for name, facts in paths.items():
        lines.append(f"- {name}: " + " ".join(f["fact"] for f in facts))
    return ("THE PATHS AND WHAT EACH COSTS (build the question from these: name what each path asks of the viewer and "
            "what it puts at risk, for example \"Would you [act of path 1], or [act of path 2] and risk [its cost]?\"):\n"
            + "\n".join(lines))


_RISK_WORDS = (r"\b(?:risk\w*|chance|might|may|could|die|dies|dying|death|dead|kill\w*|los(?:e|es|ing|t)|hurt\w*|harm\w*|"
               r"surviv\w*|cost\w*|danger\w*|sacrific\w*|price|doom\w*|lives|life)\b")


def _question_sides(question):
    """'Would you A, or B?' -> ['Would you A', 'B'] (empty when the question is not an either/or)."""
    parts = re.split(r",?\s+or\s+", remove_citation(question or ""), maxsplit=1)
    return parts if len(parts) == 2 else []


def ending_balance_problem(question, paths, game=""):
    """An either/or closing question must weigh both paths: a path with a cost (a COST beat) whose side of the
    question names nothing of that cost makes the question lopsided (one side risky, the other free), so the answer is
    obvious. Returns a reason, or "" when balanced, when it is not an either/or, or when no path has a cost."""
    sides = _question_sides(question)
    if not sides or not paths:
        return ""
    words = {name: content_stems(name + " " + " ".join(f["fact"] for f in facts), game) for name, facts in paths.items()}
    for name, facts in paths.items():
        risks = [f for f in facts if f.get("beat") == "risk"]
        if not risks:
            continue
        others = set().union(*(w for other, w in words.items() if other != name))
        distinct = words[name] - others   # "trust X" and "do not trust X" share words; what each path DOES does not
        overlap = [len(content_stems(side, game) & distinct) for side in sides]
        if max(overlap) == 0 or overlap.count(max(overlap)) > 1:
            continue
        side = sides[overlap.index(max(overlap))]
        acts = content_stems(name + " " + " ".join(f["fact"] for f in facts if f.get("beat") != "risk"), game)
        cost_words = set().union(*(content_stems(f["fact"], game) for f in risks)) - acts
        if not (content_stems(side, game) & cost_words) and not re.search(_RISK_WORDS, side, re.I):
            return f"its side for \"{name}\" leaves out what that path risks ({risks[0]['fact']})"
    return ""


def best_ending_question(game, topic, draft, paths=None):
    """The closing question decides whether viewers comment. Several are written for the finished story, obvious or
    predictive ones are screened out in code, and a judge keeps the one most likely to split viewers."""
    if ENDING_CANDIDATES <= 0:
        return draft
    story = clean_narration(draft)
    prompt = f"""Write {ENDING_CANDIDATES} different closing questions for this short video about "{topic}" in the game "{game}". The closing question is what makes viewers argue in the comments, so:
- It makes the viewer take a side between two defensible answers. The strongest weigh the real costs the story shows (one path costs this, the other costs that), or ask whether a person in the story deserves what happens to them.
{paths_block(paths)}
- It is concrete: it names the actual act and the actual cost from the story, not abstractions on their own ("a lie", "victory", "the truth", "peace").
- No answer is obviously right, and the story has not already answered it. The viewer already knows how each path turned out, so ask about the MOMENT OF CHOOSING, when no one knew: what each path asked of you then and what it risked (for example "not knowing ..., would you ...?").
- Both sides carry their weight: when a path has a cost, its side of the question names it; never give one side a cost and the other side none.
- Use only names the story already explained; never a name the viewer heard once in passing.
- Never ask what will happen, whether something will succeed, or how someone feels.
- Natural spoken words, at most {ENDING_MAX_WORDS} words, ending with a question mark. Add no fact the story does not contain.
Output one per line: Q1: <question>
STORY:
{story}"""
    try:
        raw = ask_writer(prompt, temperature=0.8, options={"top_p": 0.95})
    except Exception:
        return draft
    questions = []
    for line in raw.splitlines():
        match = re.match(r"(?i)^\s*(?:[-*]\s*)?Q\s*\d+\s*[:.)]\s*(.+?\?)\s*$", strip_markdown(line))
        if match:
            q = match.group(1).strip().strip('"')
            if len(q.split()) <= ENDING_MAX_WORDS and not weak_ending_reason(q) and q not in questions:
                questions.append(q)
    if not questions:
        return draft
    balanced = [q for q in questions if not ending_balance_problem(q, paths, game)]
    if balanced and len(balanced) < len(questions):
        print(f"Closing questions: {len(questions) - len(balanced)} dropped for leaving out a path's cost.")
    questions = balanced or questions
    choice = questions[0]
    if len(questions) > 1:
        listing = "\n".join(f"{n}. {q}" for n, q in enumerate(questions, 1))
        judge = f"""You decide which closing question will start the biggest argument in the comments of a short video. The best one gives viewers two defensible sides, names the real costs from the story, is short and natural, and is not already answered by the story. Reply with the number only.
STORY:
{story}
QUESTIONS:
{listing}"""
        try:
            pick = re.search(r"\d+", ask_ollama(judge, model=SCRIPT_EVAL_MODEL, temperature=0.0) or "")
            if pick and 1 <= int(pick.group()) <= len(questions):
                choice = questions[int(pick.group()) - 1]
        except Exception:
            pass
    if not re.search(r"(?m)^\s*ENDING QUESTION\s*:", draft):
        return draft
    return re.sub(r"(?m)^(\s*ENDING QUESTION\s*:).*$", lambda m: f"{m.group(1)} {choice}", draft, count=1)


def trim_to_length(game, topic, draft, selected_hook, must_tell):
    """Writers overshoot when every key point must fit. One editing pass cuts an over-long draft to the target,
    removing repetition, background and decoration first, never a key point. Kept only if it is shorter and whole."""
    words = word_count(draft)
    if words <= MAX_WORDS:
        return draft
    points = "\n".join(f"- {f['fact']}" for f in must_tell) or "- (the cause, every option and what it leads to)"
    prompt = f"""This voice-over script about "{topic}" in "{game}" is {words} spoken words; it must be at most {SCRIPT_TARGET_MAX_WORDS - 10}. Cut it down.
Cut first: repetition, background, side details, stacked adjectives, filler transitions. Merge lines that say related things.
Keep: the HOOK line exactly as it is, every KEY POINT below (in fewer words if needed), the story's order and its best lines. Add nothing new.
Output only the script in the same layout (STATUS, HOOK, NARRATION lines, PAYOFF, ENDING QUESTION, CLAIMS REQUIRING MORE EVIDENCE), no citations.
KEY POINTS:
{points}
SCRIPT:
{clean_review_draft(draft)}"""
    try:
        cut = enforce_hook(tidy_writer_output(ask_writer(prompt, temperature=0.3)), selected_hook)
    except Exception:
        return draft
    lines = spoken_lines(cut)
    if not lines or not any(sec == "PAYOFF" for sec, _ in lines) or not any(sec == "ENDING QUESTION" for sec, _ in lines):
        return draft
    return cut if word_count(cut) < words else draft


def revise_for_flow(game, topic, draft, selected_hook, material):
    """Second pass: an editor rewrites the draft for flow only (transitions, chronology, one turn, a payoff that answers
    the hook, an ending that splits viewers). Writers write better prose when drafting and revising are separate jobs.
    The editor may not add facts; the fact-check runs on the revised text."""
    prompt = f"""You are the story editor for a 30-60 second gaming-lore voice-over about "{topic}" in "{game}". Rewrite the DRAFT so it plays as one gripping story:
- The first NARRATION line must keep a stranger watching: the danger, what it does to its victims, who suffers and what they could lose. If the draft opens the narration with background (when or how something first appeared, what kind of being it is, world history), move that later or cut it.
- After that opener, tell events in the order they happened; every line must lead to the next (but, so, until, then, which is why).
- Plain spoken words: replace any clinical, legal or academic label with what actually happened. If the payoff is a trick or twist, make sure it says plainly how it works.
- Build to one turn or reveal just before the payoff.
- The PAYOFF must answer exactly what the HOOK makes the viewer want to know, told as the player's choice: for each option, what is actually done (the real act, by whom, to whom) and what it leads to or costs, from the material. Replace any vague label ("use a trick", "fight it") with the scene the material describes. Each option must be a different path (what the player does differently); if the draft offers the same option twice, or splits one path into two options, fix it.
- If a path has a twist, it is told last and the PAYOFF ends on the reveal; never reveal it earlier. Say "you" for the player character's actions throughout. Each path's cost is what is at stake for people, not an incidental detail, and a path the material gives a cost for always says it.
- If the rule (whom the threat chooses) does not obviously fit what the person did, add the bridge the material gives (what they believe or blame themselves for). If a path works by using that rule, the rule must be stated before the choice in the form that path needs.
- The first time a person is named, say who they are to the story in a few words, using only a relationship or role the material states (never guess one); a person who only does one thing gets no name, just what they do. No new names in the PAYOFF or the ENDING QUESTION.
- Cut any line that only announces a choice ("you must decide"). Split any sentence over 25 words. Replace long, formal words with everyday ones.
- It must read as a story, not a manual: at most one sentence may start with "If you"; turn the rest into "You could...", "Or you...", "But there is another way...". Each line should leave a question the next line answers.
- At most three names in the whole script; call everyone else by the role the material gives ("the jarl's daughter", "his wife"), never a bare "the girl", and explain or replace any game term a newcomer would not know.
- Two halves: NARRATION lines 1-{STORY_LINES} are storytelling (the danger, who suffers, why it happened; short and vivid). From NARRATION line {STORY_LINES + 1} to the PAYOFF it is precise: for each path, what you actually do, what it costs, and the reveal, with the concrete details from the material.
- Cut anything repeated or off the main thread; keep it under {SCRIPT_TARGET_MAX_WORDS} spoken words.
- The ENDING QUESTION must split viewers: two defensible answers, at most 18 words.
- Keep every fact exactly as true as in the draft and the material; add NO new facts, names, causes or outcomes. Keep the HOOK line exactly as it is.
Output only the script in the same layout (STATUS, HOOK, NARRATION lines, PAYOFF, ENDING QUESTION, CLAIMS REQUIRING MORE EVIDENCE), no citations, no quotation marks.
{style_example_block(topic, game)}MATERIAL:
{material}
DRAFT:
{clean_review_draft(draft)}"""
    try:
        revised = ask_writer(prompt, temperature=0.4, options={"top_p": 0.92})
    except Exception:
        return draft
    revised = enforce_hook(tidy_writer_output(revised), selected_hook)
    lines = spoken_lines(revised)
    if not lines or not any(sec == "PAYOFF" for sec, _ in lines) or not any(sec == "ENDING QUESTION" for sec, _ in lines):
        return draft   # an unusable edit never replaces a usable draft
    return revised


def clean_review_draft(draft):
    return "\n".join(f"{sec}: {remove_citation(c)}" for sec, c in spoken_lines(draft))


def pick_best_draft(game, drafts):
    """Several drafts per attempt; the separate judge picks the one that flows best (progression counts double)."""
    if len(drafts) == 1:
        return drafts[0]
    best, best_key = drafts[0], None
    for draft in drafts:
        evaluation = evaluate_script(game, draft) if spoken_lines(draft) else None
        key = ((evaluation["clarity"] + evaluation["opening"] + 2 * evaluation["progression"] + evaluation["delivery"])
               if evaluation else -1)
        if best_key is None or key > best_key:
            best, best_key = draft, key
    return best


def produce_script(game, topic, focused_evidence, valid_ids, passage_map, selected_hook, story_brief, audit_cache, max_attempts, essential=None, chain="", facts_for_plan=None, angle=""):
    """STORY FIRST, run for ONE hook: the writer tells the story freely (no citations); code attaches citations; the
    audit removes only invented facts (citation errors are fixed in code, uncertain lines are left alone); repeated
    lines are rewritten to move the story on; then craft is judged (required points told, retention, hook delivered).
    Layout problems and invented facts are the only things that block; craft problems drive another attempt and are
    scored. Returns the best attempt."""
    feedback, best, attempts_used = "", None, 0
    plan, style_note, structure, cast = "", "", {}, ""
    choice_points = [f for f in payoff_points(facts_for_plan) if f not in (essential or [])]
    must_tell = list(essential or []) + choice_points
    if facts_for_plan:
        plan = plan_story_beats(game, topic, selected_hook, facts_for_plan, essential, structure)
        cast = cast_block(game, plan, passage_map) if plan else ""
        if cast:
            plan = plan + "\n" + cast
        if plan:
            print("\n--- " + plan)
    by_lines = WRITER_MODE == "lines" and bool(structure.get("timeline") or structure.get("paths") or structure.get("loose"))
    if by_lines:
        max_attempts = min(max_attempts, LINES_MAX_ATTEMPTS)
        print(f"Line-by-line writer: {LOCAL_WRITER_MODEL or MODEL} writes one line at a time from the plan (up to {max_attempts} attempts).")
    lean = not by_lines and hosted_writer_active() and bool(plan or chain)   # a strong writer: one short prompt, one pass
    if lean:
        max_attempts = min(max_attempts, HOSTED_MAX_ATTEMPTS)
        print(f"Lean mode: {writer_name().split(' -> ')[0]} writes the script in one pass (up to {max_attempts} attempts).")
    for attempt in range(1, max_attempts + 1):
        attempts_used = attempt
        print(f"\n--- SCRIPT ATTEMPT {attempt}/{max_attempts} ---")
        material = plan or chain
        drafts = []
        if by_lines:
            keep = {}
            written = write_by_lines(game, topic, selected_hook, lines_structure(structure, facts_for_plan, essential),
                                     cast, feedback, keep)
            if written:
                written = polish_lines(game, topic, written, keep, cast)
                drafts.append(enforce_hook(written, selected_hook))
        if lean:
            raw = ask_writer(build_hosted_prompt(game, topic, selected_hook, material, must_tell,
                                                 "\n".join(x for x in (feedback, style_note) if x), angle), temperature=0.7)
            drafts.append(enforce_hook(tidy_writer_output(raw), selected_hook))
        for n in range(0 if (lean or (by_lines and drafts)) else (max(1, DRAFTS_PER_ATTEMPT) if attempt == 1 else 1)):   # retries fix feedback: one draft is enough
            full = ""
            if FULL_STORY_FIRST and material:   # the whole story first, so nothing that makes it make sense is lost
                try:
                    full = ask_writer(build_full_story_prompt(game, topic, selected_hook, material, must_tell, angle),
                                      temperature=0.7 + 0.05 * n, options={"top_p": 0.92})
                    full = re.sub(r"[*_`#]+", "", full).strip()
                except Exception:
                    full = ""
            prompt = build_generation_prompt(game, topic, focused_evidence, valid_ids, selected_hook, story_brief,
                                             "\n".join(x for x in (feedback, style_note) if x), must_tell, material,
                                             angle=angle, full_story=full)
            raw = ask_writer(prompt, temperature=(0.7 if attempt == 1 else 0.6) + 0.05 * n, options={"top_p": 0.92})
            drafts.append(enforce_hook(tidy_writer_output(raw), selected_hook))
        if FULL_STORY_FIRST and material and not lean and not by_lines:
            print(f"Told the whole story first, then cut it to length ({len(drafts)} draft(s)).")
        draft = pick_best_draft(game, drafts)
        echoes = style_echoes(draft, topic, game=game)
        if echoes:
            print("Note: the draft reuses wording from a style example (" + "; ".join(echoes[:3]) + "); asking for its own words.")
            style_note = "Do not reuse wording from the style example: " + "; ".join(echoes[:3])
        if len(drafts) > 1:
            print(f"Wrote {len(drafts)} drafts; the judge picked draft {drafts.index(draft) + 1} for flow.")
        if REVISE_PASS and material and not lean and not by_lines:
            draft = revise_for_flow(game, topic, draft, selected_hook, material)
            print("Editor pass: revised the draft for flow and payoff.")
        if word_count(draft) > MAX_WORDS:
            before = word_count(draft)
            draft = trim_to_length(game, topic, draft, selected_hook, must_tell)
            print(f"Too long ({before} words): cut to {word_count(draft)} words, keeping every key point.")
        draft = merge_payoffs(draft)
        ending_before = next((c for sec, c in spoken_lines(draft) if sec == "ENDING QUESTION"), "")
        paths = {}
        for f in choice_points:
            paths.setdefault(f.get("option") or "the choice", []).append(f)
        draft = best_ending_question(game, topic, draft, paths if len(paths) >= 2 else None)
        ending_after = next((c for sec, c in spoken_lines(draft) if sec == "ENDING QUESTION"), "")
        if ending_after and ending_after != ending_before:
            print(f"Closing question chosen from {ENDING_CANDIDATES} written for this story: {remove_citation(ending_after)}")
        draft, corrections = canonicalise_citation_ids(draft, valid_ids)
        draft = attach_citations(draft, passage_map, facts_for_plan, game, valid_ids)
        draft, titled = ensure_game_title(draft, game, selected_hook)
        if titled:
            print(f'The hook does not name the game, so "{game_title_variants_raw(game)}" was added to the first NARRATION line in code.')

        problems = check_script_format(draft, valid_ids, selected_hook, game, essential, topic)
        style_notes = [p for p in problems if not is_structural(p)]
        problems = [p for p in problems if is_structural(p)]
        audits, retention, coherence, missing, missing_pay = [], None, None, [], []
        if problems:
            print("\nLayout checks failed:")
            for problem in problems:
                print("-", problem)
            feedback = "\n".join(["Previous attempt had layout problems:", *[f"- {p}" for p in problems],
                                  "Use the exact section tags and copy the selected hook verbatim.", length_feedback(draft)])
        else:
            if any(fails for *_, fails in script_repetition(draft, game)):
                for _round in range(2):   # a line that repeats another kills momentum: rewrite it to move the story on
                    draft, _audits, did = repair_repetition(game, topic, draft, passage_map, valid_ids, audit_cache)
                    draft, _ = ensure_game_title(draft, game, selected_hook)
                    if not did or not any(fails for *_, fails in script_repetition(draft, game)):
                        break
            print(f"\nWord count: {word_count(draft)} | Layout OK. Checking that nothing is invented...")
            draft, audits = audit_and_repair(game, topic, draft, passage_map, valid_ids, audit_cache)
            draft, retitled = ensure_game_title(draft, game, selected_hook)   # a repair may have rewritten the titled line
            if retitled:
                audits = audit_script(draft, passage_map, game, audit_cache)
            problems = check_script_format(draft, valid_ids, selected_hook, game, essential, topic)
            style_notes = [p for p in problems if not is_structural(p)]
            problems = [p for p in problems if is_structural(p)]
            flagged = [a for a in audits if is_flagged(a)]
            if problems:
                feedback = "\n".join(["Previous attempt had layout problems after repairs:", *[f"- {p}" for p in problems], length_feedback(draft)])
            elif flagged:
                feedback = ("Keep the story, its order and its voice, but these statements are not in the material. Cut or correct only "
                            "them (a storytelling turn or a description is fine; an invented fact is not):\n"
                            + "\n".join(f"- {a['section']}: {remove_citation(a['content'])} -> {a['reason']}" for a in flagged))
            else:
                missing = missing_essential_semantic(game, draft, essential)
                # the choice must be told in the precise half (NARRATION line STORY_LINES+1 to the PAYOFF), not hinted at earlier
                missing_pay = missing_essential_semantic(game, precise_part(draft), choice_points) if choice_points else []
                missing_pay += [f for f in untold_costs(precise_part(draft), choice_points, game) if f not in missing_pay]
                retention = retention_review(game, topic, draft)
                coherence = hook_script_coherence(game, draft, selected_hook)
                print(f"\nRetention: {retention['status']} / {retention['verdict']} | Hook/script coherence: {coherence['verdict']}"
                      + (f" | required points not told: {len(missing)}" if missing else ""))
                notes = []
                if style_notes:
                    notes.append("Polish these (they cost points, they do not reject the story):\n" + "\n".join(f"- {p}" for p in style_notes))
                if missing:
                    notes.append("The story left out points it must tell (work them in where they hit hardest):\n"
                                 + "\n".join(f"- {f['fact']}" for f in missing))
                if missing_pay:
                    notes.append(f"From NARRATION line {STORY_LINES + 1} to the PAYOFF, the script does not show how each choice really plays out. Tell these plainly there"
                                 + " (the ones marked MUST are what you do on a path, what it costs, or the reveal):\n"
                                 + "\n".join(f"- {'MUST: ' if is_key_payoff(f) else ''}{f['fact']}" for f in missing_pay))
                if not retention_ok(retention):
                    notes.append("The story craft needs work:\n" + retention["text"])
                if coherence["verdict"] == "FAILS":
                    notes.append("The rest of the script does not pay off the HOOK:\n" + coherence["text"])
                feedback = "\n\n".join(notes)

        flagged = [a for a in audits if is_flagged(a)]
        coherence_failed = bool(coherence) and coherence["verdict"] == "FAILS"
        audited = bool(audits)
        craft = (len(missing) + len(missing_pay), not retention_ok(retention), coherence_failed)
        # good enough to stop: nothing invented, every required point told, and it delivers its hook. A style note, a weak
        # retention verdict or one thin payoff point is scored, not worth a full rewrite of a sound story
        key_pay = [f for f in missing_pay if is_key_payoff(f)]   # what you do, its cost and the reveal are never optional
        passed_now = (not problems and not flagged and audited and not missing and len(missing_pay) <= 1 and not key_pay
                      and not coherence_failed)
        if passed_now and attempt < max_attempts:
            # sound is not the same as good: if the story judge finds the opening or the flow weak, try once more
            quick = evaluate_script(game, draft)
            if quick and (quick["opening"] < 3 or quick["progression"] < 3):
                passed_now = False
                feedback = (f"The facts are right, but the story judge says: {quick.get('why') or 'the story does not flow'}. "
                            "Keep every fact and required point; make it flow as one gripping story.")
                print(f"Story judge: opening {quick['opening']}/5, flow {quick['progression']}/5 - trying once more for a better telling.")
        score = (len(flagged) if audited else 1, not audited, bool(problems), coherence_failed, len(missing) + len(missing_pay),
                 not retention_ok(retention))
        if best is None or score < best["score"]:
            best = {"draft": draft, "audits": audits, "retention": retention, "problems": problems + style_notes, "score": score,
                    "coherence": coherence, "missing_essential": missing, "missing_payoff": missing_pay}
        if passed_now:
            break
    best["attempts_used"] = attempts_used
    best["passed"] = best["score"][:3] == (0, False, False)   # nothing made up, structure intact: the story stands
    best["hook"] = selected_hook
    return best


def select_script(candidates):
    """Best ELIGIBLE script by total; if none is eligible, the least-bad failed attempt (it is saved as REJECTED)."""
    eligible = [c for c in candidates if c["script_score"]["eligible"]]
    if eligible:
        return max(eligible, key=lambda c: (c["script_score"]["total"], -c["script_score"]["words"])), True
    return min(candidates, key=lambda c: c["score"]), False


def _format_script_score(score):
    out = [f"SCRIPT SCORE: {score['total']}/100 ({'ELIGIBLE' if score['eligible'] else 'NOT ELIGIBLE'})"]
    labels = {"clarity": "clarity", "opening": "curiosity/opening", "progression": "narrative progression + payoff",
              "delivery": "spoken delivery + concision", "grounding": "factual grounding"}
    for key, label in labels.items():
        out.append(f"  {label}: {score['points'][key]}/{SCRIPT_WEIGHTS[key]}")
    for penalty in score["penalties"]:
        out.append(f"  penalty -{penalty['points']}: {penalty['reason']}")
    if score["justification"]:
        out.append(f"  evaluator: {score['justification']}")
    for blocker in score["blockers"]:
        out.append(f"  BLOCKER: {blocker}")
    for warning in score["warnings"]:
        out.append(f"  WARNING: {warning}")
    return "\n".join(out)


def evidence_signature(evidence):
    """Identifies the evidence a saved preparation was built from: the preparation is reused only for the same file."""
    import hashlib
    return hashlib.sha1(evidence.encode("utf-8")).hexdigest()[:16] + f"|{MODEL}|prep15"


def save_prepared(path, signature, state):
    try:
        Path(path).write_text(json.dumps({"signature": signature, **state}, ensure_ascii=False), encoding="utf-8")
    except (OSError, TypeError) as error:
        print(f"WARNING: could not save the story preparation: {error}")


def load_prepared(path, signature):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if data.get("signature") != signature or not data.get("valid_ids"):
        return None
    return data


def chosen_hook_setting():
    """The hook you picked: --hook "..." or CHOSEN_HOOK. Empty = the generator picks."""
    if "--hook" in sys.argv:
        index = sys.argv.index("--hook")
        if index + 1 < len(sys.argv):
            return sys.argv[index + 1].strip()
    return os.environ.get("CHOSEN_HOOK", "").strip()


def write_hook_menu(hook_log_path, menu_path, game):
    """Every VERIFIED hook from the hook log, best first by the same ranking the generator uses, for you to pick from.
    Hooks whose facts could not be verified are left out: picking one would only get the script rejected."""
    try:
        log = json.loads(Path(hook_log_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    menu = []
    for c in log.get("candidates") or []:
        if not c.get("grounded") or not c.get("score"):
            continue
        evaluation = c.get("evaluation") or {}
        menu.append({"text": c["text"], "spoken": remove_citation(c["text"]), "approach": c.get("approach", ""),
                     "total": c["score"].get("total", 0), "eligible": bool(c["score"].get("eligible")),
                     "emotion": evaluation.get("emotion"), "curiosity": evaluation.get("curiosity"),
                     "clarity": evaluation.get("clarity"), "retention_rank": retention_rank(evaluation),
                     "notes": c["score"].get("notes", []) + c["score"].get("blockers", []),
                     "why": c["score"].get("justification", ""), "_words": c["score"].get("words", 0)})
    # the generator's own order (retention rank, then total with the game-first nudge, then the shorter hook)
    menu.sort(key=lambda m: (m["eligible"], m["retention_rank"],
                             m["total"] + (2.0 if game_named_early(m["text"], game) else 0.0), -m.pop("_words")), reverse=True)
    selected = (log.get("selected") or {}).get("text")
    for m in menu:
        m["recommended"] = m["text"] == selected
    menu.sort(key=lambda m: not m["recommended"])   # the generator's pick on top
    try:
        Path(menu_path).write_text(json.dumps({"game": log.get("game"), "topic": log.get("topic"), "hooks": menu},
                                              indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as error:
        print(f"WARNING: could not write the hook menu: {error}")
        return None
    return Path(menu_path)


def include_cited_passages(line, evidence, focused_evidence, valid_ids):
    """A hook picked from an earlier run may cite a passage this run's selection left out. The passage is in the
    evidence file, so it is added back: a hook must never be rejected only because the selection changed."""
    missing = [i for i in line_citations(line) if i not in valid_ids]
    if not missing:
        return focused_evidence, valid_ids
    by_id = {p["id"]: p for p in split_evidence_passages(evidence)}
    added = [i for i in missing if i in by_id]
    if added:
        focused_evidence = focused_evidence.rstrip() + "\n\n" + "\n\n".join(compact_passage(by_id[i]) for i in added)
        print("Added the passage(s) your hook cites: " + ", ".join(added))
    return focused_evidence, set(valid_ids) | set(added)


def use_chosen_hook(text, game, facts, passage_map, evidence, name):
    """Your hook is used as given. A hook from the menu was already verified; an edited or new one is checked the same
    way any hook is, and cited from the passages that state it. A claim no passage supports is reported, not hidden:
    the draft is then saved as REJECTED with the reason, and you can still approve it yourself."""
    spoken = remove_citation(strip_markdown(text)).strip()
    try:
        menu = json.loads((CHECK_DIR / f"{name}_hook_menu.json").read_text(encoding="utf-8")).get("hooks", [])
    except (OSError, ValueError):
        menu = []
    for item in menu:
        if item.get("spoken", "").casefold() == spoken.casefold():
            VERIFIED_HOOKS.add(item["text"])
            print(f"Using the hook you picked (verified in the hook run): {item['text']}")
            return item["text"]
    valid_ids = sorted(passage_map, key=lambda v: int(v[1:]))
    cited, _ = normalise_hook_citation(text, valid_ids) if line_citations(text) else (None, None)
    if not cited:
        cited = recite_from_facts(spoken, facts, game)
    if not line_citations(cited):
        found = locate_support(spoken, passage_map, game)
        ids = [found["id"]] if found else []
        if not ids:
            stems = content_stems(spoken, game)
            ranked = sorted(((len(stems & content_stems(excerpt_of(t), game)), i) for i, t in passage_map.items()), reverse=True)
            ids = [i for score, i in ranked[:2] if score]
        cited = (spoken if spoken.endswith(("?", "!", ".")) else spoken + ".") + "".join(f" [{i}]" for i in ids[:2])
    candidate = {"text": cited}
    audit_hook_candidates([candidate], evidence, game)
    VERIFIED_HOOKS.add(candidate["text"])   # your pick: reported below if something looks unsupported, never re-judged
    if candidate["grounded"]:
        print(f"Using your hook (checked against the evidence: nothing made up): {candidate['text']}")
    else:
        print(f"Note: the checker could not find support for part of your hook ({candidate['audit_reason']}). "
              "It is used as you picked it and will not be held against the script.")
    return candidate["text"]


def _choice_prompt(game, topic, evidence, ids_text, max_facts, known_options=()):
    other = ""
    if known_options:
        other = ("ONLY THE OTHER PATHS ARE MISSING: the facts so far cover only " + ", ".join(f'"{o}"' for o in known_options)
                 + ". Give the facts for every OTHER option the player can choose instead (what the player does differently, "
                 "and what that leads to or costs). Do not repeat the option(s) above. If the evidence names no other option, reply NONE.\n")
    return f"""A short video about "{topic}" in the game "{game}" ends on the choice the player faces. Each OPTION is a different path: something the player decides to do INSTEAD of the other options (for example accepting a plan or refusing it). The steps and results of one path all belong to that one option.
ONLY REAL DECISIONS: an option is a point where the player picks one outcome over another. Steps every player goes through anyway (accepting a quest, doing the tasks it sets, events that happen whichever way you choose) are NOT options: leave them out, or treat them as what leads up to the decision. When a story has several such steps and one final decision, the final decision is the choice.
{other}From the evidence, for EACH option the player (as the main character) can choose, give:
- WHAT IS ACTUALLY DONE: the concrete actions, who does them and to whom (the real scene, not the name of the method),
- WHAT IT LEADS TO or COSTS: the result, and any risk or price the evidence states. A risk to someone's life, health or mind (who may die, be hurt, or not survive) is always the most important cost: state it whenever the evidence does. Gameplay rewards (loot, experience, items, achievements) and passing reactions (a punch, an insult, a remark) are not story consequences; leave them out.
- WHAT IT CHANGES LATER: any later effect the evidence links to this choice for a character (who gains or loses what, for example a rise in standing or a chance taken away), and what the choice means for them.
- HOW IT REALLY WORKS: any hidden detail or twist the evidence reveals that explains why it works or what someone did not know (what was secretly arranged, what was never really in danger). Leave this out and the viewer would misunderstand what happened, so include it whenever the evidence states it.
One plain sentence each, at most 26 words, saying what happens IF the player chooses it. Keep who does what exactly as the passage says. Add nothing the passage does not say. Copy the shortest supporting span word for word.
Name each fact's OPTION in 2-6 words, as the player's decision; facts about the same path use exactly the same OPTION name.
Use only these IDs: {ids_text}
Output only lines in this exact format (at most {max_facts}), or exactly NONE if the evidence describes no choice:
FACT 1: <plain sentence> || QUOTE: <verbatim span from the passage> || ID: E### || KIND: choice || OPTION: <the player's decision>
(KIND is choice for what the player does, consequence for what it leads to, risk for a danger or price to someone, twist for a hidden detail or reveal that explains it.)
EVIDENCE:
{evidence}"""


def _parse_choice_facts(raw, passage_map, valid_ids):
    facts = []
    for line in raw.splitlines():
        # models label lines "FACT 1:" or by what they answer ("THE OUTCOME:", "THE RISK:", "TWIST:"): accept both
        match = re.match(r"(?i)^\s*(?:[-*]\s*)?(?:FACT\s*\d*|(?:THE\s+)?(?:OUTCOME|RISK|TWIST|CHOICE|CONSEQUENCE|COST))\s*:\s*"
                         r"(.+?)\s*\|{1,2}\s*QUOTE\s*:\s*(.+?)\s*\|{1,2}\s*ID\s*:\s*\[?(E\d+)\]?"
                         r"\s*(?:\|{1,2}\s*KIND\s*:\s*(\w+))?\s*(?:\|{1,2}\s*OPTION\s*:\s*(.+?))?\s*$", strip_markdown(line.strip()))
        if not match:
            continue
        fact, quote = strip_markdown(match.group(1)), match.group(2).strip().strip('"\u201c\u201d\'').strip()
        evidence_id = match.group(3).upper()
        if evidence_id not in passage_map or not quote_in_text(quote, excerpt_of(passage_map[evidence_id])):
            holders = [i for i in sorted(valid_ids, key=lambda v: int(v[1:]))
                       if i in passage_map and quote_in_text(quote, excerpt_of(passage_map[i]))]
            if not holders:
                continue
            evidence_id = holders[0]
        if evidence_id not in valid_ids or len(fact.split()) > 34:
            continue
        kind_word = (match.group(4) or "").lower()
        kind = "choice" if kind_word.startswith("choi") else "consequence"
        label = re.match(r"(?i)\s*(?:[-*]\s*)?(?:THE\s+)?(OUTCOME|RISK|TWIST|COST)\b", line)
        beat = (label.group(1).lower() if label else kind_word if kind_word in ("risk", "twist", "outcome", "cost") else "")
        option = strip_markdown(match.group(5) or "").strip(" .\"'") or "the choice"
        facts.append({"id": evidence_id, "fact": fact, "quote": quote, "kind": kind, "payoff_detail": True, "option": option,
                      "beat": "risk" if beat == "cost" else beat})
    return facts


def _after_condition(text):
    """'If the player refuses, the host may die.' -> 'the host may die.': two facts about one path share the condition."""
    return re.sub(r"(?i)^\s*if\b[^,]{0,120},\s*", "", text or "")


def _option_key(name):
    """Same path, same key; but "trust her" and "do not trust her" are opposite paths, so negations are kept."""
    words = re.findall(r"[a-z']+", (name or "").casefold().replace("n't", " not"))
    return " ".join(w for w in words if w not in {"the", "a", "an", "to", "and", "of", "player", "chooses", "choose"})


def _outcome_prompt(game, topic, evidence, ids_text, option, known=()):
    listing = "\n".join(f"{n}. {f['fact']}" for n, f in enumerate(known, 1))
    minor = (f"FACTS SO FAR FOR THIS PATH:\n{listing}\nAlso reply with one line MINOR: <numbers> naming any of these that are only "
             "a passing reaction or a reward rather than what happens to the people at stake, or MINOR: NONE.\n") if known else ""
    return f"""For the option "{option}" the player can choose in the story of "{topic}" in the game "{game}", give from the evidence what it LEADS TO (never just what the player does):
- THE OUTCOME: what finally happens to the people at stake if the player chooses it (who is saved, freed, hurt, or may die),
- THE RISK: any danger or price for the people at stake on this path (who may die, be hurt, or not survive). Always include it whenever the evidence states it, even as a possibility, and
- THE TWIST, if there is one: a hidden detail that explains why it works or what someone did not know (what was secretly arranged, what was never really in danger).
{minor}
A passing reaction (a punch, an insult, a remark) and a gameplay reward (loot, experience) are not outcomes. One plain sentence each, at most 26 words, saying what happens IF the player chooses it. Keep who does what exactly as the passage says. Add nothing the passage does not say. Copy the shortest supporting span word for word.
Use only these IDs: {ids_text}
Output only lines in this exact format, or exactly NONE:
FACT 1: <plain sentence> || QUOTE: <verbatim span from the passage> || ID: E### || KIND: consequence || OPTION: {option}
EVIDENCE:
{evidence}"""


def find_story_event(game, topic, evidence, valid_ids, ending_text):
    """When no fact says what happened to the people in this story, ask for exactly that: the cause must be told as
    "what happened", never only as a general rule. Code-verified quote; None if the evidence does not say."""
    passage_map = {p["id"]: p["text"] for p in split_evidence_passages(evidence)}
    prompt = f"""A short video tells the story of "{topic}" in the game "{game}". It ends on this choice:
{ending_text}
What happened to, or was done by, the people in THIS story BEFORE the situation began, that brought it about (who did what, what happened to them)? Not a general rule about how the threat works, nothing from the choice or its outcome above, not another storyline in the evidence.
Reply with ONE line, FACT: <one plain sentence of at most 26 words> || QUOTE: <shortest verbatim span copied from that passage> || ID: E###
or exactly NONE. Use only these IDs: {", ".join(sorted(valid_ids, key=lambda v: int(v[1:])))}
EVIDENCE:
{evidence}"""
    try:
        raw = ask_ollama(prompt, temperature=0.0)
    except Exception:
        return None
    for line in raw.splitlines():
        match = re.match(r"(?i)^\s*(?:[-*]\s*)?FACT\s*\d*\s*:\s*(.+?)\s*\|{1,2}\s*QUOTE\s*:\s*(.+?)\s*\|{1,2}\s*ID\s*:\s*\[?(E\d+)\]?\s*$", line.strip())
        if not match:
            continue
        fact, quote, evidence_id = strip_markdown(match.group(1)), match.group(2).strip().strip('"\u201c\u201d\'').strip(), match.group(3).upper()
        if evidence_id in valid_ids and evidence_id in passage_map and len(fact.split()) <= 32 \
                and quote_in_text(quote, excerpt_of(passage_map[evidence_id])):
            return {"id": evidence_id, "fact": fact, "quote": quote, "kind": "cause", "cause_role": "EVENT", "direct": True}
    return None


def extract_choice_facts(game, topic, evidence, valid_ids, max_facts=8):
    """The payoff needs the concrete details of each option, not the rule behind it ("use a trick"): what is actually
    done, by whom, to whom, and what each option leads to or costs, each fact labelled with the path it belongs to. If
    only one path turns up, a second pass looks for the others, so the steps of one path are never mistaken for two
    options. Every quote is verified in code."""
    passage_map = {p["id"]: p["text"] for p in split_evidence_passages(evidence)}
    ids_text = ", ".join(sorted(valid_ids, key=lambda v: int(v[1:])))
    try:
        facts = _parse_choice_facts(ask_ollama(_choice_prompt(game, topic, evidence, ids_text, max_facts), temperature=0.1),
                                    passage_map, valid_ids)
    except Exception:
        return []
    options = list(dict.fromkeys(f["option"] for f in facts))
    if facts and len({_option_key(o) for o in options}) < 2:
        try:
            more = _parse_choice_facts(ask_ollama(_choice_prompt(game, topic, evidence, ids_text, 4, options), temperature=0.1),
                                       passage_map, valid_ids)
        except Exception:
            more = []
        known = {_option_key(o) for o in options}
        facts += [f for f in more if _option_key(f["option"]) not in known
                  and not any(f["id"] == x["id"] and f["quote"] == x["quote"] for x in facts)]
    # one focused question per path: its final outcome for the people at stake, and the twist behind it
    by_option = {}
    for f in facts:
        by_option.setdefault(_option_key(f["option"]), []).append(f)
    passages = split_evidence_passages(evidence)
    for key, group in list(by_option.items())[:3]:
        option_name = group[0]["option"]   # kept: the path keeps its name even if every fact on it is dropped as minor
        # a small, focused excerpt: the passages this path's facts come from, plus the few most closely related
        own = {f["id"] for f in group}
        path_stems = content_stems(" ".join(f["fact"] for f in group), game)
        related = sorted((p for p in passages if p["id"] not in own and p["id"] in valid_ids),
                         key=lambda p: -len(path_stems & content_stems(excerpt_of(p["text"]), game)))[:3]
        focus = [p for p in passages if p["id"] in own] + related
        focus_text = "\n\n".join(compact_passage(p) for p in focus) or evidence
        focus_ids = ", ".join(p["id"] for p in focus) or ids_text
        try:
            raw = ask_ollama(_outcome_prompt(game, topic, focus_text, focus_ids, option_name, group), temperature=0.0)
            outcome = _parse_choice_facts(raw, passage_map, valid_ids)
            focus_map = {p["id"]: p["text"] for p in focus}
            for m in re.finditer(r"(?im)^\s*(?:[-*]\s*)?(?:THE\s+)?(OUTCOME|RISK|TWIST|COST)\s*:\s*(?!.*\|\|)(.+?)\s*$", raw):
                beat = {"cost": "risk"}.get(m.group(1).lower(), m.group(1).lower())
                sentence = strip_markdown(m.group(2)).strip()
                if not sentence or sentence.upper().startswith("NONE") or len(sentence.split()) > 34:
                    continue
                hit = locate_support(sentence, focus_map, game)
                if hit and hit["verdict"] in ("SUPPORTED", "PARTLY") and hit["id"] in valid_ids:
                    outcome.append({"id": hit["id"], "fact": sentence, "quote": hit["quote"], "kind": "consequence",
                                    "payoff_detail": True, "option": option_name, "beat": beat})
            minor = re.search(r"(?im)^\s*MINOR\s*:\s*(.+)$", raw)
            marked = minor.group(1) if minor else ""
            drop_ids = {i.upper() for i in re.findall(r"(?i)\bE\d+\b", marked)}          # by passage id ...
            drop = {int(n) for n in re.findall(r"(?<![Ee\d])\d+", marked)}               # ... or by number
            group = [f for n, f in enumerate(group, 1)
                     if f["kind"] == "choice" or (n not in drop and f["id"] not in drop_ids)]
        except Exception:
            outcome = []
        fresh = []
        for f in outcome[:3]:
            f["option"] = option_name
            if not any(f["id"] == x["id"] and f["quote"] == x["quote"] for x in facts + fresh) \
                    and not any(_restates(_after_condition(f["fact"]), _after_condition(x["fact"]), game) for x in group + fresh):
                fresh.append(f)
        # per path: what is done first, then its outcome and twist, then anything else (at most three facts)
        steps = [f for f in group if f["kind"] == "choice"][:1]
        rest = [f for f in group if f not in steps]
        by_option[key] = (steps + fresh + rest)[:4]
    return [f for group in by_option.values() for f in group][:max_facts]


def prepare_story(game, topic, evidence):
    """Everything before the hooks: evidence selection, the cause search, the story brief and the verified facts.
    Saved after a hooks-only run so the script run for a hook you picked starts straight away."""
    focused_evidence, valid_ids, _ = select_relevant_evidence(evidence, topic, game)
    if not valid_ids:
        raise ValueError("No relevant evidence passages passed the selection threshold.")
    focused_evidence, valid_ids, added = expand_story_evidence(game, topic, evidence, focused_evidence, valid_ids)
    if added:
        print("Added passages that state missing story links: " + ", ".join(added))
    try:
        focused_evidence, valid_ids, cause_facts, cause_notes = search_causes_all_evidence(game, topic, evidence, focused_evidence, valid_ids)
    except Exception as error:
        cause_facts, cause_notes = [], [f"cause search failed: {error}"]
    for note in cause_notes:
        if note.startswith("cause search added"):
            print(note)
    for fact in cause_facts:
        print(f"  cause found: [{fact['id']}] ({fact['kind']}) {fact['fact']}")
    print(f"Evidence passages selected ({len(valid_ids)}): " + ", ".join(sorted(valid_ids, key=lambda v: int(v[1:]))))
    passage_map = {p["id"]: p["text"] for p in split_evidence_passages(focused_evidence) if p["id"] in valid_ids}

    story_brief = develop_story_angle(game, topic, focused_evidence, valid_ids, cause_facts)
    story_brief, brief_note = verify_brief_root_cause(story_brief, passage_map, game)
    print("\n--- STORY BRIEF ---\n" + story_brief)
    if brief_note:
        print(brief_note)
    try:
        facts, dropped_facts = extract_hook_facts(game, topic, focused_evidence, valid_ids, story_brief)
    except Exception as error:   # facts are an aid; every hook and script line is still audited without them
        facts, dropped_facts = [], [f"fact extraction failed: {error}"]
    known = {(f["id"], f["quote"]) for f in facts}
    facts = [f for f in cause_facts if (f["id"], f["quote"]) not in known] + facts
    dropped_facts = list(dropped_facts) + cause_notes
    essential = essential_facts(facts)
    protected = [f["fact"] for f in essential]
    root_cause = re.search(r"(?mi)^\s*ROOT_CAUSE\s*:\s*(.+)$", story_brief)
    if root_cause and not root_cause.group(1).strip().upper().startswith("NONE"):
        protected.append(root_cause.group(1))
    story_brief, kept_in = protect_root_cause(story_brief, protected, game)
    story_brief, kept_by_model = protect_causal_items(game, topic, story_brief)
    kept_in += kept_by_model
    if kept_in:
        print("The brief listed these as omittable, but they explain the story, so they stay in: " + "; ".join(kept_in))
    for item in kept_in[:3]:   # make sure each protected detail is backed by a verified fact the script must state
        try:
            found = extract_targeted_fact(game, topic, focused_evidence, valid_ids, item)
        except Exception:
            found = None
        if found and not any(f["id"] == found["id"] and f["quote"] == found["quote"] for f in facts):
            facts.append(found)
        elif not found:
            print(f"No passage states a cause-and-effect link for '{item}'; it is not required in the script.")
    choice_facts = extract_choice_facts(game, topic, focused_evidence, valid_ids)
    for fact in choice_facts:
        print(f"  how the choice plays out: [{fact['id']}] ({fact['kind']}) {fact['fact']}")
    facts = facts + [f for f in choice_facts if not any(f["id"] == x["id"] and f["quote"] == x["quote"] for x in facts)]
    facts, fact_notes = verify_cause_facts(game, facts, passage_map)
    dropped_facts = list(dropped_facts) + fact_notes
    for note in fact_notes:
        print("  " + note)
    facts = dedupe_facts(facts)
    facts = rank_cause_facts(game, topic, facts)
    if not any(f.get("cause_role") == "EVENT" for f in essential_facts(facts)):
        ending_text = "\n".join(f"- {f['fact']}" for f in facts if f.get("payoff_detail"))
        event = find_story_event(game, topic, focused_evidence, valid_ids, ending_text) if ending_text else None
        ending_facts = [f for f in facts if f.get("payoff_detail")]
        if event and any(_restates(event["fact"], f["fact"], game) or event["quote"] in f.get("quote", "") for f in ending_facts):
            event = None   # it retold the ending, not what came before it
        if event and (connected_to_story(event["fact"], ending_text, game) or mentions_topic(event["fact"], topic, game)):
            facts.append(event)
            print(f"  what happened in this story (found by a focused search): [{event['id']}] {event['fact']}")
    for fact in facts:
        if fact.get("direct") is False:
            print(f"  background, not required: [{fact['id']}] {fact['fact']}")
    essential = essential_facts(facts)
    rules = [f for f in essential if f.get("cause_role") == "RULE"]
    if rules:   # a rule about whom the threat chooses needs the deed (or the suspicion) that makes this person fit it
        candidates = [f for f in facts if f not in essential and not f.get("payoff_detail") and not f.get("uncertain")
                      and f.get("cause_role") not in ("RULE", "RESOLUTION") and not POSSIBILITY.search(f["fact"])]
        bridge = [f for f in pick_bridge_events(candidates, rules, fallback=False) if not any(_same_fact(f, e) for e in essential)]
        for fact in bridge[:2]:
            print(f"  why the rule fits this person (now required): [{fact['id']}] {fact['fact']}")
        essential = essential + bridge[:2]
    for fact in essential:
        print(f"ESSENTIAL (the script must state this): [{fact['id']}] {fact['fact']}")
    not_established = [re.sub(r"^\[E\d+\]\s*", "", n.split(": ", 1)[1]).split(" - ")[0]
                       for n in fact_notes if n.startswith("dropped")]
    if brief_note and brief_note.startswith("ROOT_CAUSE rejected"):
        not_established.append(brief_note.split(": ", 1)[1].split(" - ")[0].strip("'"))
    not_established = [n for n in dict.fromkeys(not_established)
                       if not any(len(content_stems(n, game) & content_stems(f["fact"], game)) >= 0.6 * max(1, len(content_stems(n, game)))
                                  for f in facts)]   # never forbid what a verified fact states in other words
    chain = story_chain_text(facts, not_established)
    if chain:
        print("\n--- " + chain)
    return {"focused_evidence": focused_evidence, "valid_ids": sorted(valid_ids), "story_brief": story_brief, "facts": facts,
            "dropped_facts": dropped_facts, "essential": essential, "chain": chain}


def write_script_menu(path, candidates, best, final_draft, best_report, timestamp, game, topic):
    """Every finished version of this run, for you to read and pick from (the best one is already saved as the
    draft). Each keeps its own fact-check report, so the one you pick carries its real status."""
    versions = []
    for n, c in enumerate(candidates, 1):
        chosen = c is best
        try:
            report = best_report if chosen else (format_review(c["audits"], c["retention"]) + "\n\n" + _format_script_score(c["script_score"])
                                                 + ("\n\nMECHANICAL PROBLEMS:\n" + "\n".join(f"- {p}" for p in c["problems"]) if c["problems"] else ""))
        except Exception as error:
            report = f"(report unavailable: {error})"
        flags = [f"{a['section']}: {a['reason']}" for a in c.get("audits") or [] if is_flagged(a)]
        versions.append({"number": n, "hook": c["hook"], "angle": c.get("angle_name", ""), "draft": (final_draft if chosen else c["draft"]).strip(),
                         "eligible": bool(c["script_score"]["eligible"]), "total": c["script_score"]["total"],
                         "why": (c.get("evaluation") or {}).get("why", ""), "problems": c["problems"][:5], "flags": flags[:5],
                         "words": word_count(c["draft"]), "selected": chosen, "review": report})
    try:
        Path(path).write_text(json.dumps({"game": game, "topic": topic, "timestamp": timestamp, "versions": versions},
                                         indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as error:
        print(f"WARNING: could not write the version list: {error}")
        return None
    return Path(path)


def main():
    VERIFIED_HOOKS.clear()
    if not EVIDENCE_FILE.exists():
        raise FileNotFoundError(f"Evidence file missing: {EVIDENCE_FILE}")
    game, topic, research_path = read_project_context()
    evidence = EVIDENCE_FILE.read_text(encoding="utf-8-sig")
    if not split_evidence_passages(evidence):
        raise ValueError("No evidence passages found.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = slugify(f"{game}_{topic}")

    print(f"Model: {MODEL} | Ollama endpoint: {OLLAMA_URL}")
    print(f"Writer: {writer_name() if writer_chain() else 'local model (' + (LOCAL_WRITER_MODEL or MODEL) + ')'}, mode: {WRITER_MODE}"
          + " | research and fact checks: local model")
    print(f"Checking game/topic evidence context using {research_path.name}...")
    context_problems = check_evidence_context(evidence, game, topic)
    if context_problems:
        gate_path = CHECK_DIR / f"{name}_gate_{timestamp}.txt"
        gate_path.write_text("\n".join(context_problems) + "\n", encoding="utf-8")
        print("Evidence context check failed:")
        for problem in context_problems:
            print("-", problem)
        print(f"Gate report saved to: {gate_path}")
        return

    # A run no longer dies because the judge was harsh: if no hook passes the quality floors, the best VERIFIED hook is
    # used and the warning is kept in the logs. Facts are never relaxed. --strict-hook / HOOK_STRICT=1 restores the stop.
    strict = "--strict-hook" in sys.argv or os.environ.get("HOOK_STRICT") == "1"
    allow_weak = not strict or "--allow-weak-hook" in sys.argv or os.environ.get("ALLOW_WEAK_HOOK") == "1"
    hook_log_path = CHECK_DIR / f"{name}_{timestamp}_hooks.json"
    script_log_path = CHECK_DIR / f"{name}_{timestamp}_scripts.json"
    hooks_only = "--hooks-only" in sys.argv or os.environ.get("HOOKS_ONLY") == "1"
    chosen_hook = chosen_hook_setting()
    prepared_path = CHECK_DIR / f"{name}_prepared.json"
    signature = evidence_signature(evidence)
    _TOPIC_STEMS.clear()
    _TOPIC_STEMS.update(content_stems(topic, game))
    try:
        state = load_prepared(prepared_path, signature) if chosen_hook else None
        if state:
            print("Reusing the story preparation from the hook run (same evidence), so the script starts straight away.")
        else:
            state = prepare_story(game, topic, evidence)
            save_prepared(prepared_path, signature, state)
        focused_evidence, valid_ids = state["focused_evidence"], set(state["valid_ids"])
        if chosen_hook:   # the passages your hook cites stay available even if this run selected others
            focused_evidence, valid_ids = include_cited_passages(chosen_hook, evidence, focused_evidence, valid_ids)
        story_brief, facts, essential, chain = state["story_brief"], state["facts"], state["essential"], state["chain"]
        passage_map = {p["id"]: p["text"] for p in split_evidence_passages(focused_evidence) if p["id"] in valid_ids}
        if chosen_hook:
            hook = use_chosen_hook(chosen_hook, game, facts, passage_map, evidence, name)
            focused_evidence, valid_ids = include_cited_passages(hook, evidence, focused_evidence, valid_ids)
            passage_map = {p["id"]: p["text"] for p in split_evidence_passages(focused_evidence) if p["id"] in valid_ids}
            hooks = [hook]
        else:
            hooks = generate_and_select_hooks(game, topic, focused_evidence, valid_ids, story_brief,
                                              log_path=hook_log_path, allow_weak=allow_weak,
                                              top_n=1 if hooks_only else SCRIPT_CANDIDATES,
                                              facts=facts, dropped_facts=state["dropped_facts"])
            print(f"Hook log saved to: {hook_log_path}")
            menu_path = write_hook_menu(hook_log_path, CHECK_DIR / f"{name}_hook_menu.json", game)
            if hooks_only:
                print(f"\nHook menu saved to: {menu_path.name if menu_path else '(not written)'}")
                print("Pick a hook (or edit one) and run the script step with it.")
                print("PIPELINE GATE STATUS: HOOKS_READY")
                return
    except HookSelectionError as warning:
        print(f"\n{warning}")
        print(f"Hook log saved to: {hook_log_path}")
        if hooks_only:
            write_hook_menu(hook_log_path, CHECK_DIR / f"{name}_hook_menu.json", game)
        print("PIPELINE GATE STATUS: NO_SAFE_HOOK (no candidate hook could be verified against the evidence)")
        return
    except Exception as error:
        print(f"PIPELINE ENGINE ERROR: {error}")
        return

    # One full script per top hook (each hook is a different story angle). Fact audits are cached across candidates.
    audit_cache, candidates = {}, []
    versions = SCRIPT_VERSIONS or (3 if chosen_hook else 1)
    jobs = [(hook, VERSION_ANGLES[v % len(VERSION_ANGLES)] if versions > 1 else VERSION_ANGLES[0])
            for hook in hooks for v in range(versions)]
    for position, (hook, (angle_name, angle)) in enumerate(jobs, 1):
        print(f"\n===== SCRIPT CANDIDATE {position}/{len(jobs)}: {remove_citation(hook)}"
              + (f" ({angle_name.lower()})" if versions > 1 else "") + " =====")
        result = produce_script(game, topic, focused_evidence, valid_ids, passage_map, hook, story_brief, audit_cache,
                                MAX_ATTEMPTS if position == 1 else SCRIPT_EXTRA_ATTEMPTS, essential, chain, facts, angle=angle)
        result["angle_name"] = angle_name
        evaluation = None
        # score every attempt whose facts and layout passed, even if the (subjective) retention review did not
        if result["passed"] or (not result["problems"] and result["audits"] and not any(is_flagged(a) for a in result["audits"])):
            evaluation = evaluate_script(game, result["draft"])
            if evaluation is None:
                print("WARNING: the script evaluator returned no usable score; this candidate is ranked on the gates only.")
        result["evaluation"] = evaluation
        result["script_score"] = score_script(result["draft"], game, result["audits"], result["retention"],
                                              result["coherence"], result["problems"], evaluation, result.get("missing_essential"),
                                              result.get("missing_payoff"))
        print("\n" + _format_script_score(result["script_score"]))
        candidates.append(result)

    best, eligible = select_script(candidates)
    try:
        script_log_path.write_text(json.dumps({
            "game": game, "topic": topic, "generator_model": MODEL, "evaluator_model": SCRIPT_EVAL_MODEL,
            "weights": SCRIPT_WEIGHTS, "selected_hook": best["hook"], "selected_is_eligible": eligible,
            "essential_facts": essential,
            "candidates": [{"hook": c["hook"], "attempts": c["attempts_used"], "draft": c["draft"], "passed_gates": c["passed"],
                            "evaluation": c["evaluation"], "score": c["script_score"],
                            "coherence": (c["coherence"] or {}).get("verdict")} for c in candidates],
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Script log saved to: {script_log_path}")
    except OSError as error:
        print(f"WARNING: could not write script log: {error}")

    if not eligible and not best["audits"]:
        # the saved draft never reached the fact audit; audit it now so the report names the unsupported claims
        print("\nAuditing the best rejected draft so the report shows which claims the evidence does not support...")
        try:
            best["draft"], _ = ensure_game_title(best["draft"], game, best["hook"])
            best["audits"] = audit_script(best["draft"], passage_map, game, audit_cache)
            for _ in range(3):
                best["draft"], recited = apply_recitations(best["draft"], best["audits"])
                if not recited:
                    break
                best["audits"] = audit_script(best["draft"], passage_map, game, audit_cache)
            best["problems"] = check_script_format(best["draft"], valid_ids, best["hook"], game, essential)
            if not best["problems"] and not any(is_flagged(a) for a in best["audits"]):
                # the code fixes (title, citations) were all it needed: run the remaining gates instead of rejecting it
                print("The rejected draft passes the layout and fact checks after code-only fixes; running the remaining checks...")
                best["retention"] = retention_review(game, topic, best["draft"])
                best["coherence"] = hook_script_coherence(game, best["draft"], best["hook"])
                best["evaluation"] = evaluate_script(game, best["draft"])
            best["script_score"] = score_script(best["draft"], game, best["audits"], best["retention"], best["coherence"],
                                                best["problems"], best["evaluation"], best.get("missing_essential"),
                                                best.get("missing_payoff"))
            if best["script_score"]["eligible"] and best["coherence"] is not None:
                eligible = True
                print("\n" + _format_script_score(best["script_score"]))
        except Exception as error:
            print(f"WARNING: could not audit the rejected draft: {error}")
    final_draft, final_audits, final_retention = best["draft"], best["audits"], best["retention"]
    missing = missing_evidence_note(final_audits, best["problems"], best["coherence"])
    suffix = "REVIEW_REQUIRED" if eligible else "REJECTED"
    if missing:   # say what is missing or unverified instead of papering over it
        final_draft = re.sub(r"(?mi)^(\s*CLAIMS REQUIRING MORE EVIDENCE\s*:).*$",
                             lambda m: m.group(1) + " " + "; ".join(missing)[:600].replace("\n", " "), final_draft)
    draft_path = SCRIPT_DIR / f"{name}_{timestamp}_{suffix}.txt"
    narration_path = SCRIPT_DIR / f"{name}_{timestamp}_NARRATION.txt"
    draft_path.write_text(final_draft.strip() + "\n", encoding="utf-8")
    narration_path.write_text(clean_narration(final_draft) + "\n", encoding="utf-8")
    review_path = CHECK_DIR / f"{name}_{timestamp}_review.txt"
    report = format_review(final_audits, final_retention)
    report += f"\n\nSELECTED HOOK: {best['hook']}"
    coherence_result = best.get("coherence")
    report += "\nHOOK/SCRIPT COHERENCE: " + (f"{coherence_result['verdict']}\n{coherence_result['text']}" if coherence_result
                                          else "NOT RUN (earlier checks failed first)")
    report += "\n\n" + _format_script_score(best["script_score"])
    if len(candidates) > 1:
        report += "\n\nCANDIDATE SCRIPTS COMPARED:\n" + "\n".join(
            f"- {remove_citation(c['hook'])} -> {c['script_score']['total']}/100, "
            f"{'eligible' if c['script_score']['eligible'] else 'not eligible'}{' (selected)' if c is best else ''}"
            for c in candidates)
    if best["problems"]:
        report += "\n\nMECHANICAL PROBLEMS:\n" + "\n".join(f"- {p}" for p in best["problems"])
    if missing:
        report += "\n\nWHAT THE EVIDENCE DID NOT SUPPORT OR LEAVES UNCERTAIN (nothing was invented to fill these gaps):\n" + "\n".join(f"- {m}" for m in missing)
    review_path.write_text(report + "\n", encoding="utf-8")
    print(f"Audit report saved to: {review_path}")
    menu_path = write_script_menu(CHECK_DIR / f"{name}_script_menu.json", candidates, best, final_draft, report, timestamp, game, topic)
    if menu_path and len(candidates) > 1:
        print(f"All {len(candidates)} versions saved for you to compare and pick from: {menu_path.name}")
    print(f"\nScripting run finalized after {best['attempts_used']} attempt(s) for the selected candidate. Draft saved to: {draft_path.name}")
    print(f"Narration-only file saved to: {narration_path.name}")
    if suffix == "REJECTED":
        print("Reason: no candidate passed every gate; open the audit report to see exactly what the evidence did not support.")
    print(f"PIPELINE GATE STATUS: {suffix}")


if __name__ == "__main__":
    main()
