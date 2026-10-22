from pathlib import Path
from datetime import datetime
import re

PROJECT = Path.home() / "Desktop" / "GamingLoreAI"
SOURCES = PROJECT / "sources"
OUTPUT = SOURCES / "evidence_passages.txt"
EXCLUDED = {"evidence_passages.txt"}  # generated output; never re-ingest it


def normalise(value):
    """Make metadata comparisons case-insensitive and whitespace-safe."""
    return re.sub(r"\s+", " ", value).strip().casefold()


def make_passage(source, source_type, url, text):
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 40:
        return None
    return {
        "source": source,
        "type": source_type,
        "url": url,
        "excerpt": text,
    }


def parse_research_file(path):
    """Read only explicitly numbered passages from successfully retrieved pages."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    results = []
    game = ""
    topic = ""
    current_source = None
    current_type = "Retrieved webpage; not independently verified"
    current_url = "Not recorded"
    retrieval_status = ""

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("GAME:"):
            game = stripped.partition(":")[2].strip()
            continue
        if stripped.startswith("TOPIC:"):
            topic = stripped.partition(":")[2].strip()
            continue
        if stripped.startswith("SOURCE:"):
            current_source = stripped.partition(":")[2].strip()
            current_type = "Retrieved webpage; not independently verified"
            current_url = "Not recorded"
            retrieval_status = ""
            continue
        if stripped.startswith("RETRIEVAL STATUS:"):
            retrieval_status = stripped.partition(":")[2].strip().upper()
            continue
        if stripped.startswith("SOURCE TYPE:"):
            current_type = stripped.partition(":")[2].strip()
            continue
        if stripped.startswith("URL:"):
            current_url = stripped.partition(":")[2].strip()
            continue

        # A source contributes evidence only when retrieval succeeded and the
        # line is one of the numbered excerpts emitted by research_sources.py.
        if not current_source or retrieval_status != "SUCCESS":
            continue
        if not re.match(r"^\[PASSAGE\s+\d+\]", stripped):
            continue

        excerpt = re.sub(r"^\[PASSAGE\s+\d+\]\s*", "", stripped)
        passage = make_passage(current_source, current_type, current_url, excerpt)
        if passage:
            results.append(passage)

    if not game or not topic:
        raise ValueError(
            f"Research file is missing GAME or TOPIC metadata: {path.name}"
        )
    return game, topic, results


def parse_manual_source(path):
    text = path.read_text(encoding="utf-8-sig")
    match = re.search(r"EXCERPT:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if not match or not match.group(1).strip():
        print(f"  SKIPPED manual source {path.name}: no excerpt found.")
        return None, None, []

    metadata = text[:match.start()]
    excerpt = match.group(1).strip()

    def field(name, default=""):
        found = re.search(
            rf"^{re.escape(name)}:\s*(.+)$",
            metadata,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        return found.group(1).strip() if found else default

    game = field("GAME")
    topic = field("TOPIC")
    source = field("SOURCE", path.name)
    source_type = field("SOURCE TYPE", "Manual source; type not specified")
    url = field("URL", "Not recorded")
    passage = make_passage(source, source_type, url, excerpt)
    return game, topic, [passage] if passage else []


def main():
    if not SOURCES.exists():
        raise SystemExit(f"Sources folder not found: {SOURCES}")

    # The newest research file defines the current target topic. Older files
    # for different games/topics must not win merely because they sort first.
    research_files = sorted(
        [
            path for path in SOURCES.glob("*_research_*.txt")
            if path.name not in EXCLUDED
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not research_files:
        raise SystemExit("No research file found. Download research first.")

    all_research = []
    target_game = None
    target_topic = None
    display_game = ""
    display_topic = ""

    print(f"Found {len(research_files)} research file(s). Checking metadata newest-first...")
    for path in research_files:
        try:
            game, topic, passages = parse_research_file(path)
        except (OSError, UnicodeError, ValueError) as error:
            print(f"  SKIPPED {path.name}: {error}")
            continue

        file_game = normalise(game)
        file_topic = normalise(topic)
        if target_game is None:
            target_game, target_topic = file_game, file_topic
            display_game, display_topic = game, topic
        if file_game != target_game or file_topic != target_topic:
            print(f"  SKIPPED {path.name}: game/topic differs ({game} / {topic}).")
            continue

        all_research.extend(passages)
        print(f"  INCLUDED {path.name}: {len(passages)} passages")

    if target_game is None:
        raise SystemExit("No usable research file with GAME and TOPIC metadata found.")

    passages = list(all_research)

    # Include matching manually supplied sources, such as *_secondary.txt.
    for pattern in ("*_secondary.txt", "*_source.txt"):
        for path in sorted(SOURCES.glob(pattern)):
            if path.name in EXCLUDED:
                continue
            try:
                manual_game, manual_topic, found = parse_manual_source(path)
            except (OSError, UnicodeError) as error:
                print(f"  SKIPPED {path.name}: {error}")
                continue

            if not manual_game or not manual_topic:
                print(f"  SKIPPED {path.name}: GAME or TOPIC metadata missing.")
                continue
            if normalise(manual_game) != target_game or normalise(manual_topic) != target_topic:
                print(f"  SKIPPED {path.name}: game/topic does not match target.")
                continue
            passages.extend(found)
            print(f"  INCLUDED manual source {path.name}: {len(found)} passage(s)")

    # Remove exact duplicate evidence entries without merging distinct sources.
    unique = []
    seen = set()
    for passage in passages:
        key = (
            normalise(passage["source"]),
            normalise(passage["url"]),
            normalise(passage["excerpt"]),
        )
        if key not in seen:
            seen.add(key)
            unique.append(passage)

    if not unique:
        raise SystemExit("No evidence passages found. Check the research files and URLs.")

    output_lines = [
        "EVIDENCE PASSAGES",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"GAME: {display_game}",
        f"TOPIC: {display_topic}",
        f"Research files combined: {len(research_files)} (only matching game/topic files included)",
        "Warning: retrieved text is not independently verified.",
        "",
    ]

    for index, passage in enumerate(unique, start=1):
        output_lines.extend([
            f"[E{index:03d}]",
            f"Source: {passage['source']}",
            f"Source type: {passage['type']}",
            f"URL: {passage['url']}",
            "Verification status: Retrieved; not independently verified.",
            f"Excerpt: {passage['excerpt']}",
            "",
        ])

    OUTPUT.write_text("\n".join(output_lines), encoding="utf-8")
    print(f"\nSaved {len(unique)} unique passages to:\n{OUTPUT}")
    print("Remember: collected passages still need factual review before use in a script.")


if __name__ == "__main__":
    main()
