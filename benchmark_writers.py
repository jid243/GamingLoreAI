"""Benchmark the script writers on your own topics, so the results decide, not opinions.

For each topic (one research file each), the SAME evidence and the SAME hook are given to every combination of
  writer mode:  classic (the existing writer, unchanged)  vs  lines (the line-by-line writer)
  writing model: the models in BENCH_MODELS (default: your current model and gemma4:12b-it-qat)
Research, fact checks and the hook stay on the main model for every combination, so only the writing differs.

Each script is then scored by a blind judge (it is not told which writer made it), separately for hook, suspense,
clarity, pacing, payoff and how much editing it needs, next to the fact-check result and the time it took. Passing
the fact check is reported on its own: a correct script can still be boring.

Your project is left as it was: research file dates, the evidence file and the version list are restored, and every
draft the benchmark writes is moved into benchmark/<time>/ instead of your scripts folder.

Run:  python benchmark_writers.py            (or the app's Benchmark button)
      python benchmark_writers.py --restore  (puts your project back if a benchmark was stopped part-way;
                                              a new benchmark run also does this first)
Options (environment): BENCH_TOPICS=3  BENCH_MODELS="gemma4:26b-a4b-it-qat,gemma4:12b-it-qat"  BENCH_MODES="classic,lines"
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
SOURCES = PROJECT / "sources"
SCRIPTS = PROJECT / "scripts"
CHECKS = PROJECT / "fact_checks"
EVIDENCE = SOURCES / "evidence_passages.txt"
GENERATOR = PROJECT / "generate_grounded_script_google.py"
PREPARE = PROJECT / "prepare_evidence.py"

sys.path.insert(0, str(PROJECT))
import generate_grounded_script_google as g  # noqa: E402

MODELS = [m.strip() for m in os.environ.get("BENCH_MODELS", f"{g.MODEL},gemma4:12b-it-qat").split(",") if m.strip()]
MODES = [m.strip() for m in os.environ.get("BENCH_MODES", "classic,lines").split(",") if m.strip()]
TOPIC_LIMIT = int(os.environ.get("BENCH_TOPICS", "3"))
PENDING = []   # models asked for but not installed: reported as pending, never swapped for another model
RUN_TIMEOUT = int(os.environ.get("BENCH_RUN_TIMEOUT", str(40 * 60)))
JUDGE_ROUNDS = 2
CRITERIA = ("hook", "pacing", "suspense", "detail", "characters", "payoff", "delivery", "edit")


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:50] or "gaming_lore"


def meta(path, field):
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")[:4000]
    except OSError:
        return ""
    m = re.search(rf"(?mi)^\s*{field}\s*:\s*(.+?)\s*$", text)
    return m.group(1).strip() if m else ""


def pick_topics(limit):
    """The newest research file for each different topic, newest topics first."""
    seen, picked = set(), []
    for path in sorted(SOURCES.glob("*_research_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True):
        game, topic = meta(path, "GAME"), meta(path, "TOPIC")
        norm = lambda t: re.sub(r"[^a-z0-9]+", " ", t.casefold()).strip()
        # the same topic re-run under a longer name ("<game>'s X" and "X") counts once
        key = (norm(game), re.sub(r"^(?:the\s+)?(?:s\s+)?", "", norm(topic).replace(norm(game), "").strip()))
        if not game or not topic or key in seen:
            continue
        seen.add(key)
        picked.append((path, game, topic))
        if len(picked) >= limit:
            break
    return picked


def run(label, argv, env, log_path):
    """Run one step, streaming its output to a log; returns (seconds, ok)."""
    start = time.time()
    full_env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", **env}
    print(f"  > {label}", flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        try:
            proc = subprocess.run(argv, cwd=PROJECT, env=full_env, stdout=log, stderr=subprocess.STDOUT, timeout=RUN_TIMEOUT)
            ok = proc.returncode == 0
        except subprocess.TimeoutExpired:
            log.write(f"\nBENCHMARK: stopped after {RUN_TIMEOUT}s\n")
            ok = False
    seconds = round(time.time() - start)
    print(f"    {'done' if ok else 'FAILED'} in {seconds // 60}m {seconds % 60}s", flush=True)
    return seconds, ok


def new_files(before, folder):
    return [p for p in folder.glob("*") if p.is_file() and p not in before]


JUDGE_PROMPT = """You judge a voice-over script for a 30-60 second YouTube Short about video-game lore. Viewers have never played the game. Judge it as a viewer would, honestly; do not reward big words. Score each from 1 (fails) to 10 (excellent):
HOOK: would the first two lines stop a stranger from swiping?
PACING: does it move as one story, every line pushing forward, with no list of facts and no repetition?
SUSPENSE: does each line make you want the next one, building to the payoff?
DETAIL: concrete, specific details from this game's story rather than vague, generic statements?
CHARACTERS: does a newcomer know who each named person is and why they matter, with no one appearing from nowhere?
PAYOFF: does the ending deliver what the hook promised, with a closing question people would argue about?
DELIVERY: does it sound natural spoken aloud (short sentences, everyday words, nothing that trips the tongue)?
EDIT: how little rewriting it needs before it could be posted (10 = post as is, 1 = rewrite it all).
Reply with exactly one line:
HOOK=<n> PACING=<n> SUSPENSE=<n> DETAIL=<n> CHARACTERS=<n> PAYOFF=<n> DELIVERY=<n> EDIT=<n> WHY=<at most 20 words: the biggest weakness>
SCRIPT:
{script}"""


def judge(script):
    """Blind story-quality scores (the judge never learns which writer made the script); two rounds, averaged."""
    rows = []
    for n in range(JUDGE_ROUNDS):
        try:
            raw = g.ask_ollama(JUDGE_PROMPT.format(script=script), temperature=0.0 if n == 0 else 0.3, model=g.MODEL)
        except Exception as error:
            print(f"    judge failed: {error}")
            continue
        found = {k.lower(): int(v) for k, v in re.findall(r"\b(" + "|".join(CRITERIA) + r")\s*=\s*(\d+)", strip(raw), re.I)}
        why = re.search(r"WHY\s*=\s*(.+)", strip(raw), re.I)
        if all(c in found for c in CRITERIA):
            rows.append(([min(10, found[c]) for c in CRITERIA], (why.group(1).strip() if why else "")))
    if not rows:
        return None
    scores = {c: round(sum(r[0][i] for r in rows) / len(rows), 1) for i, c in enumerate(CRITERIA)}
    scores["story_total"] = round(sum(scores[c] for c in CRITERIA), 1)   # out of 80
    scores["why"] = rows[0][1]
    return scores


def strip(text):
    return re.sub(r"[*`]", "", text or "")


def spoken(draft):
    """The spoken lines of a draft, without citations."""
    out = []
    for m in re.finditer(r"(?m)^\s*(HOOK|NARRATION|PAYOFF|ENDING QUESTION)\s*:\s*(.+)$", draft or ""):
        text = re.sub(r"(?:\s*\[E\d+\])+\s*$", "", m.group(2)).strip()
        out.append(m.group(1) + ": " + text)
    return "\n".join(out)


def restore(out):
    """Put the project back from a benchmark folder's backups (also used after a benchmark was stopped part-way)."""
    saved = out / "_your_files"
    manifest = saved / "research_dates.json"
    if manifest.exists():
        for name, (atime, mtime) in json.loads(manifest.read_text(encoding="utf-8")).items():
            if (SOURCES / name).exists():
                os.utime(SOURCES / name, (atime, mtime))
    if (saved / EVIDENCE.name).exists():
        shutil.copy2(saved / EVIDENCE.name, EVIDENCE)
    for menu in list(saved.glob("*_script_menu.json")) + list(saved.glob("*_hook_menu.json")):
        shutil.copy2(menu, CHECKS / menu.name)
    before = json.loads((saved / "files_before.json").read_text(encoding="utf-8")) if (saved / "files_before.json").exists() else None
    if before is not None:
        drafts = out / "drafts"
        drafts.mkdir(exist_ok=True)
        for folder in (SCRIPTS, CHECKS):
            for p in folder.glob("*"):
                if p.is_file() and str(p.name) not in before.get(folder.name, []) and not p.name.endswith(("_prepared.json", "_hook_menu.json")):
                    shutil.move(str(p), drafts / p.name)
    (saved / "RESTORED").write_text("done\n", encoding="utf-8")


def restore_unfinished():
    """A benchmark stopped part-way never put the project back; do it now."""
    for out in sorted((PROJECT / "benchmark").glob("*/_your_files")) if (PROJECT / "benchmark").exists() else []:
        if not (out / "RESTORED").exists():
            print(f"Restoring your project from the unfinished benchmark {out.parent.name}...")
            restore(out.parent)


def main():
    restore_unfinished()
    if "--restore" in sys.argv:
        return 0
    models = MODELS
    try:
        import requests
        tags = requests.get(g.OLLAMA_URL.rsplit("/api/", 1)[0] + "/api/tags", timeout=10).json().get("models", [])
        have = {t["name"] for t in tags}
        missing = [m for m in models if m not in have]
        if missing:
            print("Not installed in Ollama, marked PENDING in the report: " + ", ".join(missing) + "  (install with: ollama pull <name>)")
            PENDING.extend(missing)
            models = [m for m in models if m in have]
    except Exception as error:
        print(f"Could not list Ollama models ({error}); trying them anyway.")
    topics = pick_topics(TOPIC_LIMIT)
    if not topics or not models:
        print("Nothing to benchmark (no research files or no models).")
        return 1
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = PROJECT / "benchmark" / stamp
    out.mkdir(parents=True, exist_ok=True)
    combos = [(mode, model) for model in models for mode in MODES]
    print(f"Benchmark {stamp}: {len(topics)} topic(s) x {len(combos)} writer(s). Results: {out}")

    # everything the benchmark touches is put back afterwards
    research_times = {p: (p.stat().st_atime, p.stat().st_mtime) for p in SOURCES.glob("*_research_*.txt")}
    saved = out / "_your_files"
    saved.mkdir()
    if EVIDENCE.exists():
        shutil.copy2(EVIDENCE, saved / EVIDENCE.name)
    for menu in list(CHECKS.glob("*_script_menu.json")) + list(CHECKS.glob("*_hook_menu.json")):
        shutil.copy2(menu, saved / menu.name)
    before_scripts, before_checks = set(SCRIPTS.glob("*")), set(CHECKS.glob("*"))
    (saved / "research_dates.json").write_text(json.dumps({p.name: t for p, t in research_times.items()}), encoding="utf-8")
    (saved / "files_before.json").write_text(json.dumps({SCRIPTS.name: [p.name for p in before_scripts],
                                                         CHECKS.name: [p.name for p in before_checks]}), encoding="utf-8")

    results = []
    try:
        for path, game, topic in topics:
            name = slug(f"{game}_{topic}")
            tdir = out / name
            tdir.mkdir()
            print(f"\n=== {game}: {topic} ===")
            now = time.time()
            os.utime(path, (now, now))   # the generator and evidence step use the newest research file
            run("evidence", [sys.executable, str(PREPARE)], {}, tdir / "evidence.log")
            if (CHECKS / f"{name}_hook_menu.json").exists() and (CHECKS / f"{name}_prepared.json").exists():
                print("  > hooks: reusing this topic's hook list and story preparation")
            else:
                run("hooks (shared by every writer)", [sys.executable, str(GENERATOR)],
                    {"HOOKS_ONLY": "1", "USE_ONLINE_WRITERS": "0"}, tdir / "hooks.log")
            try:
                hooks = json.loads((CHECKS / f"{name}_hook_menu.json").read_text(encoding="utf-8")).get("hooks", [])
            except (OSError, ValueError):
                hooks = []
            if not hooks:
                print("  no verified hook for this topic; skipped")
                continue
            hook = hooks[0]["spoken"]
            (tdir / "hook.txt").write_text(hook + "\n", encoding="utf-8")
            for mode, model in combos:
                label = f"{mode} writer, {model}"
                env = {"CHOSEN_HOOK": hook, "SCRIPT_VERSIONS": "1", "WRITER_MODE": mode, "LOCAL_WRITER_MODEL": model,
                       "USE_ONLINE_WRITERS": "0"}
                secs, ok = run(label, [sys.executable, str(GENERATOR)], env, tdir / f"{mode}_{slug(model)}.log")
                try:
                    menu = json.loads((CHECKS / f"{name}_script_menu.json").read_text(encoding="utf-8"))
                    version = (menu.get("versions") or [{}])[0]
                except (OSError, ValueError):
                    version = {}
                draft = version.get("draft", "")
                text = spoken(draft)
                scores = judge(text) if text else None
                log_text = (tdir / f"{mode}_{slug(model)}.log").read_text(encoding="utf-8", errors="replace")
                calls = re.findall(r"MODEL CALLS: local=(\d+) online=(\d+)", log_text)
                result = {"game": game, "topic": topic, "hook": hook, "mode": mode, "model": model, "seconds": secs,
                          "local_calls": int(calls[-1][0]) if calls else None, "online_calls": int(calls[-1][1]) if calls else None,
                          "settings": {"writer_model": model, "writer_mode": mode, "research_and_checks": g.MODEL,
                                       "num_ctx": g.NUM_CTX, "line_temperature": g.LINE_TEMPERATURE,
                                       "polish_rounds": g.LINE_POLISH_ROUNDS if mode == "lines" else 0, "versions": 1,
                                       "online_writers": "off"},
                          "finished": bool(ok and draft), "fact_check_passed": bool(version.get("eligible")),
                          "flagged": version.get("flags", []), "pipeline_score": version.get("total"),
                          "words": version.get("words"), "problems": version.get("problems", []), "script": text, "judge": scores}
                results.append(result)
                (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"    fact check {'passed' if result['fact_check_passed'] else 'NOT passed'}"
                      + (f" | story {scores['story_total']}/80" if scores else ""), flush=True)
    finally:
        # put your project back as it was
        for p, (atime, mtime) in research_times.items():
            if p.exists():
                os.utime(p, (atime, mtime))
        if (saved / EVIDENCE.name).exists():
            shutil.copy2(saved / EVIDENCE.name, EVIDENCE)
        drafts = out / "drafts"
        drafts.mkdir(exist_ok=True)
        for p in new_files(before_scripts, SCRIPTS):
            shutil.move(str(p), drafts / p.name)
        for p in new_files(before_checks, CHECKS):
            if not p.name.endswith(("_prepared.json", "_hook_menu.json")):   # caches that make your next run faster stay
                shutil.move(str(p), drafts / p.name)
        for menu in list(saved.glob("*_script_menu.json")) + list(saved.glob("*_hook_menu.json")):
            shutil.copy2(menu, CHECKS / menu.name)
        (saved / "RESTORED").write_text("done\n", encoding="utf-8")
    write_report(out, results, [(mode, m) for m in PENDING for mode in MODES])
    print(f"\nReport: {out / 'report.md'}")
    return 0


def write_report(out, results, pending=()):
    lines = [f"# Writer benchmark ({out.name})", "",
             "Story scores are from a blind judge (the main model, never told which writer made the script), 1-10 each, "
             "total out of 80. Unsupported claims come from the fact check and are reported on their own: a high story "
             "score says nothing about accuracy. Research, hook and fact checks are the same for every writer.", "",
             "## Averages per writer", "",
             "| Writer | Model | Scripts | Fact check passed | Unsupported claims | Story /80 | " + " | ".join(c.title() for c in CRITERIA)
             + " | Avg words | Avg time | Local calls | Online calls |",
             "|" + "---|" * (12 + len(CRITERIA))]
    for mode, model in dict.fromkeys((r["mode"], r["model"]) for r in results):
        rows = [r for r in results if (r["mode"], r["model"]) == (mode, model)]
        judged = [r["judge"] for r in rows if r["judge"]]
        avg = lambda key: round(sum(j[key] for j in judged) / len(judged), 1) if judged else "-"
        secs = sum(r["seconds"] for r in rows) / len(rows)
        mean = lambda key: (round(sum(r[key] for r in rows if r.get(key) is not None) / max(1, sum(r.get(key) is not None for r in rows)))
                            if any(r.get(key) is not None for r in rows) else "-")
        lines.append(f"| {mode} | {model} | {len(rows)} | {sum(r['fact_check_passed'] for r in rows)}/{len(rows)} | "
                     f"{sum(len(r['flagged']) for r in rows)} | {avg('story_total')} | " + " | ".join(str(avg(c)) for c in CRITERIA)
                     + f" | {mean('words')} | {int(secs // 60)}m {int(secs % 60)}s | {mean('local_calls')} | {mean('online_calls')} |")
    for mode, model in pending:
        lines.append(f"| {mode} | {model} | PENDING: model not installed | " + " | ".join("-" for _ in range(9 + len(CRITERIA))) + " |")
    for topic in dict.fromkeys(r["topic"] for r in results):
        rows = [r for r in results if r["topic"] == topic]
        lines += ["", f"## {rows[0]['game']}: {topic}", "", f"Hook: {rows[0]['hook']}", ""]
        for r in rows:
            j = r["judge"] or {}
            lines += [f"### {r['mode']} writer, {r['model']}", "",
                      f"{r['seconds'] // 60}m {r['seconds'] % 60}s | {r['words']} words | {r.get('local_calls', '-')} local / "
                      f"{r.get('online_calls', '-')} online calls | fact check: {'passed' if r['fact_check_passed'] else 'NOT passed'} "
                      f"({len(r['flagged'])} unsupported) | story {j.get('story_total', '-')}/80 ("
                      + ", ".join(f"{c} {j.get(c, '-')}" for c in CRITERIA) + ")", ""]
            if j.get("why"):
                lines += [f"Judge: {j['why']}", ""]
            if r["flagged"]:
                lines += ["Flagged: " + "; ".join(r["flagged"]), ""]
            lines += ["```", r["script"] or "(no script)", "```", ""]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
