# GamingLoreAI

A local pipeline that makes short (30-60 s) gaming-lore videos for YouTube Shorts:
research a topic, turn it into numbered evidence passages, write a script whose every
fact is checked against that evidence, find matching gameplay clips, and assemble the video.
Everything runs on your own PC with [Ollama](https://ollama.com); online writers are optional and off by default.

## The pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `research_sources.py` | Collects wiki/web sources for a game and topic into `sources/` |
| 2 | `prepare_evidence.py` | Splits the research into numbered passages (`[E001]`, ...) |
| 3 | `generate_grounded_script_google.py` | Hooks, story plan and script; every line is fact-checked against the passages |
| 4 | `harvest_clips_universal.py` | Searches and downloads gameplay clips per script line, checked by a local vision model |
| 5 | `review_clips.py` | A/B page to pick clips |
| 6 | `make_ai_inserts.py` | Optional AI stills via a local ComfyUI |
| 7 | `build_video.py`, `add_captions.py` | Narration timing, word-by-word captions, 9:16 assembly with FFmpeg |
| - | `gaminglore_app.py` | Browser app that runs all of the above: `python gaminglore_app.py`, then open http://127.0.0.1:8780 |
| - | `benchmark_writers.py` | Compares script writers/models on the same topics with a blind judge |

## How the script generator works (`generate_grounded_script_google.py`)

1. **Story preparation**: picks the relevant passages, extracts verified facts, ranks what caused the situation
   (event / rule / background), extracts the player's real choice and what each path costs, and builds a
   "must tell" list. Cached per evidence file.
2. **Hooks**: several opening lines are written and judged; you pick one in the app (Hooks tab).
3. **Script**: a story plan (cause in order, each path with its cost, the reveal last, who is who from the evidence)
   goes to the writer. Narration lines 1-3 are storytelling; from line 4 to the payoff the choice is told precisely.
4. **Fact check**: every claim in every line is checked against the passages; only invented facts block a script.
   Style problems (repetition, too many names, "if you... if you..." manuals, missing costs) lower the score and
   are sent back to the writer.
5. **Versions**: several versions are written; you compare and pick one in the app (Versions tab).

Writer modes: `classic` (default, won the benchmark) and `lines` (line-by-line, faster, weaker).
Settings are environment variables at the top of the file (e.g. `OLLAMA_MODEL`, `SCRIPT_VERSIONS`, `WRITER_MODE`).

## Setup (Windows)

```
python -m pip install -r requirements.txt
ollama pull gemma4:26b-a4b-it-qat     # writer, research and fact checks
ollama pull qwen2.5vl:7b              # clip checking
```
FFmpeg must be on PATH. Tested on an RTX 4070 Ti (12 GB) with 32 GB RAM.
The project folder is expected at `Desktop\GamingLoreAI`.

## Tests

```
python -m unittest test_grounding test_scripts test_hooks test_app test_harvest test_benchmark
```
The tests use fake model replies, so they run in about a second without Ollama.

## Not in this repository

API keys, downloaded research, generated scripts, clips, narration and videos (see `.gitignore`).
`style_examples/` holds finished scripts the writer learns pacing from (it never copies their facts).
