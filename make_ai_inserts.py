"""
make_ai_inserts.py  -  one atmospheric AI still per scene, made with your local ComfyUI.

Run after you have a reviewed draft (and before or after harvest_clips_universal.py):
    python make_ai_inserts.py
    python make_ai_inserts.py --new-prompts    # write fresh prompts (otherwise edited ones are reused)
    python make_ai_inserts.py --force          # regenerate images that already exist

build_video.py picks the images up automatically and cuts them in between gameplay shots.

One-time setup:
  1. In ComfyUI open the "image_z_image_turbo" template, then Workflow > Export (API).
     Save it as  Desktop\\GamingLoreAI\\comfyui_workflows\\image_z_image_turbo_api.json
  2. Start ComfyUI (Pinokio) and set COMFY_URL below to the address Pinokio shows for it.
Optional: Ollama with any text model writes much better image prompts than the built-in fallback.
"""
import copy, json, random, re, sys, time, urllib.error, urllib.parse, urllib.request, uuid
from pathlib import Path

PROJECT = Path.home() / "Desktop" / "GamingLoreAI"
SCRIPT_DIR = PROJECT / "scripts"
EVIDENCE_FILE = PROJECT / "sources" / "evidence_passages.txt"
IMAGE_DIR = PROJECT / "ai_images"
WORKFLOW_FILE = PROJECT / "comfyui_workflows" / "image_z_image_turbo_api.json"

COMFY_URL = "http://localhost:8188"      # Pinokio may use another port: copy it from the ComfyUI tab
OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = ""                        # empty = pick an installed text model automatically
WIDTH, HEIGHT = 768, 1344                # 9:16, about one megapixel; comfortable on a 12 GB card
STYLE = "Dark fantasy concept art, painterly, cinematic lighting, moody atmosphere, rich detail"
GENERATION_TIMEOUT = 300                 # seconds per image before giving up


# ---------------------------------------------------------------------------
# Draft / evidence helpers
# ---------------------------------------------------------------------------
def find_draft():
    drafts = [p for p in SCRIPT_DIR.glob("*_REVIEW_REQUIRED.txt") if p.is_file()]
    if not drafts:
        raise FileNotFoundError(f"No *_REVIEW_REQUIRED.txt draft in {SCRIPT_DIR}")
    return max(drafts, key=lambda p: p.stat().st_mtime)


def read_metadata(text, key):
    m = re.search(rf"(?mi)^\s*{re.escape(key)}\s*:\s*(.+?)\s*$", text)
    return m.group(1).strip() if m else ""


def extract_spoken_lines(text):
    lines = []
    for raw in text.splitlines():
        m = re.match(r"^\s*(HOOK|NARRATION|PAYOFF|ENDING QUESTION)\s*:\s*(.*?)\s*$", raw, re.I)
        if not m:
            continue
        tail = re.search(r"((?:\s*\[E\d+\])+)\s*$", m.group(2))
        citations = re.findall(r"E\d+", tail.group(1)) if tail else []
        body = re.sub(r"(?:\s*\[E\d+\])+\s*$", "", m.group(2)).strip()
        if body:
            lines.append((m.group(1).upper(), body, citations))
    return lines


def evidence_excerpt(evidence_text, tag):
    for block in re.split(r"(?m)(?=^\[E\d+\]\s*$)", evidence_text):
        if re.search(rf"(?m)^\[{re.escape(tag)}\]\s*$", block):
            found = re.search(r"(?ms)^Excerpt:\s*(.*)$", block)
            return " ".join((found.group(1) if found else block).split())[:700]
    return ""


# ---------------------------------------------------------------------------
# Tiny HTTP helper (standard library only)
# ---------------------------------------------------------------------------
def http(url, payload=None, timeout=15, raw=False):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=timeout) as resp:
        body = resp.read()
    return body if raw else json.loads(body.decode("utf-8"))


def fmt_secs(seconds):
    seconds = int(seconds)
    return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def status_line(text):
    sys.stdout.write("\r" + text[:100].ljust(100))
    sys.stdout.flush()


def clear_line():
    sys.stdout.write("\r" + " " * 100 + "\r")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Prompt writing (Ollama if available, otherwise a plain fallback)
# ---------------------------------------------------------------------------
def pick_ollama_model():
    try:
        tags = http(f"{OLLAMA_URL}/api/tags", timeout=5).get("models", [])
    except Exception:
        return None
    names = [m.get("name", "") for m in tags]
    if OLLAMA_MODEL:
        return OLLAMA_MODEL if OLLAMA_MODEL in names else None
    usable = [n for n in names if not any(bad in n.lower() for bad in ("embed", "llava", "vision", "-vl", "moondream", "bge"))]
    return usable[0] if usable else None


def ollama_prompt(model, game, topic, sentence, evidence):
    instruction = (
        "You write prompts for a text-to-image model.\n"
        f"Game: {game}\nTopic: {topic}\nNarration line: {sentence}\n"
        f"Background facts: {evidence or 'none'}\n\n"
        "Write ONE vivid prompt of 35 to 60 words describing an atmospheric still that illustrates the line: "
        "the environment, lighting, mood and key objects. Show people only as distant silhouettes or seen from behind. "
        "Do not name any character, game or franchise. Do not mention text, captions or logos. "
        "Output only the prompt, with no preface and no quotation marks."
    )
    reply = http(f"{OLLAMA_URL}/api/generate",
                 {"model": model, "prompt": instruction, "stream": False, "keep_alive": "5m",
                  "options": {"temperature": 0.7}}, timeout=180)
    text = reply.get("response", "").strip().strip('"').splitlines()[0] if reply.get("response", "").strip() else ""
    return " ".join(text.split()[:80])


def unload_ollama(model):
    """Free the GPU for ComfyUI."""
    try:
        http(f"{OLLAMA_URL}/api/generate", {"model": model, "keep_alive": 0}, timeout=15)
    except Exception:
        pass


def build_prompts(lines, game, topic, evidence_text):
    model = pick_ollama_model()
    prompts = []
    if model:
        print(f"Writing image prompts with Ollama model '{model}'...", flush=True)
    else:
        print("Ollama not found (or no text model installed): using simple fallback prompts.", flush=True)
    for idx, (_, sentence, citations) in enumerate(lines, 1):
        text = ""
        if model:
            evidence = " ".join(evidence_excerpt(evidence_text, c) for c in citations)[:900]
            try:
                text = ollama_prompt(model, game, topic, sentence, evidence)
            except Exception as exc:
                print(f"  Scene {idx}: Ollama failed ({exc}); using fallback.", flush=True)
        if not text:
            plain = re.sub(re.escape(game), "", sentence, flags=re.I).strip()
            text = f"An atmospheric illustration of this moment: {plain}"
        prompts.append(f"{STYLE}. {text}")
    if model:
        unload_ollama(model)
    return prompts


# ---------------------------------------------------------------------------
# ComfyUI
# ---------------------------------------------------------------------------
def load_workflow():
    if not WORKFLOW_FILE.exists():
        raise FileNotFoundError(
            f"Workflow not found: {WORKFLOW_FILE}\n"
            "In ComfyUI open the image_z_image_turbo template, choose Workflow > Export (API), "
            "and save it to that path.")
    data = json.loads(WORKFLOW_FILE.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict) and "nodes" in data and "links" in data:
        raise ValueError("That file is a normal workflow save. Use Workflow > Export (API) instead.")
    if not isinstance(data, dict) or not all(isinstance(v, dict) and "class_type" in v for v in data.values()):
        raise ValueError("Workflow file is not in ComfyUI API format.")
    return data


def follow(wf, ref, want, limit=60):
    """Walk upstream from an input link until a node satisfying `want` is found."""
    queue, seen = [ref], set()
    while queue and len(seen) < limit:
        r = queue.pop(0)
        if not (isinstance(r, list) and r and str(r[0]) in wf):
            continue
        nid = str(r[0])
        if nid in seen:
            continue
        seen.add(nid)
        if want(wf[nid]):
            return nid
        queue.extend(v for v in wf[nid].get("inputs", {}).values() if isinstance(v, list))
    return None


def prepare_workflow(template, prompt):
    wf = copy.deepcopy(template)
    sampler_id = next((k for k, v in wf.items()
                       if v["class_type"].startswith("KSampler") and "positive" in v.get("inputs", {})), None)
    if not sampler_id:
        raise ValueError("Could not find a KSampler node in the workflow.")
    inputs = wf[sampler_id]["inputs"]

    text_id = follow(wf, inputs["positive"], lambda n: isinstance(n.get("inputs", {}).get("text"), str))
    if not text_id:
        raise ValueError("Could not find the positive prompt text node.")
    wf[text_id]["inputs"]["text"] = prompt

    latent_id = follow(wf, inputs.get("latent_image"),
                       lambda n: all(isinstance(n.get("inputs", {}).get(k), (int, float)) for k in ("width", "height")))
    if latent_id:
        wf[latent_id]["inputs"].update({"width": WIDTH, "height": HEIGHT})
        if "batch_size" in wf[latent_id]["inputs"]:
            wf[latent_id]["inputs"]["batch_size"] = 1
    else:
        print("  Warning: no latent size node found; image will use the workflow's own size.", flush=True)

    seed = random.randint(0, 2**53)
    for key in ("seed", "noise_seed"):
        if key in inputs and not isinstance(inputs[key], list):
            inputs[key] = seed
    return wf


def generate_image(template, prompt, out_path, label):
    wf = prepare_workflow(template, prompt)
    try:
        queued = http(f"{COMFY_URL}/prompt", {"prompt": wf, "client_id": str(uuid.uuid4())})
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"ComfyUI rejected the workflow: {exc.read().decode('utf-8', 'replace')[:600]}")
    pid = queued["prompt_id"]
    started = time.time()
    spinner, ticks = "|/-\\", 0
    images = []
    while not images:
        elapsed = time.time() - started
        if elapsed > GENERATION_TIMEOUT:
            clear_line()
            raise TimeoutError(f"No image after {GENERATION_TIMEOUT}s. Check the ComfyUI window for errors.")
        entry = http(f"{COMFY_URL}/history/{pid}").get(pid)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                detail = ""
                for name, data in status.get("messages", []):
                    if name == "execution_error":
                        detail = data.get("exception_message", "")
                clear_line()
                raise RuntimeError(f"ComfyUI reported an error: {detail or 'see the ComfyUI window'}")
            for out in entry.get("outputs", {}).values():
                images += out.get("images", [])
        if not images:
            ticks += 1
            status_line(f"    {spinner[ticks % 4]} {label}: generating {fmt_secs(elapsed)} (gives up at {GENERATION_TIMEOUT}s)")
            time.sleep(1)
    clear_line()
    chosen = next((i for i in images if i.get("type") == "output"), images[0])
    query = urllib.parse.urlencode({"filename": chosen["filename"], "subfolder": chosen.get("subfolder", ""),
                                    "type": chosen.get("type", "output")})
    out_path.write_bytes(http(f"{COMFY_URL}/view?{query}", raw=True, timeout=60))
    return time.time() - started


def main():
    force = "--force" in sys.argv
    new_prompts = "--new-prompts" in sys.argv

    template = load_workflow()
    prepare_workflow(template, "test")       # fail early if the workflow shape is unexpected
    try:
        http(f"{COMFY_URL}/system_stats", timeout=5)
    except Exception:
        raise ConnectionError(
            f"Can't reach ComfyUI at {COMFY_URL}. Start it in Pinokio and copy the address shown for it "
            "into COMFY_URL at the top of this file.")

    draft = find_draft()
    text = draft.read_text(encoding="utf-8-sig")
    evidence_text = EVIDENCE_FILE.read_text(encoding="utf-8-sig") if EVIDENCE_FILE.exists() else ""
    game = read_metadata(text, "GAME") or read_metadata(evidence_text, "GAME")
    topic = read_metadata(text, "TOPIC") or read_metadata(evidence_text, "TOPIC")
    if not game:
        raise ValueError("No GAME metadata in the draft or evidence file.")
    lines = extract_spoken_lines(text)
    if not lines:
        raise ValueError(f"No tagged narration lines in {draft.name}")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    prompts_file = IMAGE_DIR / f"{draft.stem}_prompts.json"
    prompts = None
    if prompts_file.exists() and not new_prompts:
        saved = json.loads(prompts_file.read_text(encoding="utf-8"))
        if len(saved) == len(lines):
            prompts = saved
            print(f"Reusing prompts from {prompts_file.name} (edit that file to tweak them, or use --new-prompts).")
    if prompts is None:
        prompts = build_prompts(lines, game, topic, evidence_text)
        prompts_file.write_text(json.dumps(prompts, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Draft: {draft.name}\nScenes: {len(lines)}\n")
    run_started = time.time()
    made = 0
    for idx, prompt in enumerate(prompts, 1):
        out = IMAGE_DIR / f"{draft.stem}_scene_{idx:02d}.png"
        print(f"[{idx}/{len(prompts)}] {prompt}", flush=True)
        if out.exists() and not force:
            print("  Already exists, skipping (use --force to redo).", flush=True)
            continue
        try:
            took = generate_image(template, prompt, out, f"Scene {idx}")
        except Exception as exc:
            print(f"  FAILED: {exc}", flush=True)
            continue
        made += 1
        spent = time.time() - run_started
        left = (spent / made) * (len(prompts) - idx)
        print(f"  Saved {out.name} in {fmt_secs(took)}. Total {fmt_secs(spent)}; about {fmt_secs(left)} left.", flush=True)

    print(f"\nDone. Images are in {IMAGE_DIR}\n"
          "Look through them before publishing. AI imagery should be labelled as AI-generated/illustrative, "
          "and check the platform's disclosure rules.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
