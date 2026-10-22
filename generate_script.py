
from pathlib import Path
from datetime import datetime
import requests

# Project configuration
PROJECT = Path.home() / "Desktop" / "GamingLoreAI"
EVIDENCE_FILE = PROJECT / "sources" / "gaunter_odimm.txt"
MODEL = "qwen3:8b"
OLLAMA_URL = "http://localhost:11434/api/chat"

# Output folders
SCRIPT_DIR = PROJECT / "scripts"
FACT_CHECK_DIR = PROJECT / "fact_checks"

SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
FACT_CHECK_DIR.mkdir(parents=True, exist_ok=True)


def ask_ollama(prompt):
    """Send a prompt to the local Ollama model."""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        },
        timeout=600
    )

    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def main():
    if not EVIDENCE_FILE.exists():
        raise FileNotFoundError(
            f"Evidence file not found: {EVIDENCE_FILE}"
        )

    evidence = EVIDENCE_FILE.read_text(encoding="utf-8")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Step 1/3: Reading evidence...")
    print(f"Evidence file: {EVIDENCE_FILE}")

    # Generate the narration
    print("Step 2/3: Generating narration with Ollama...")

    script_prompt = f"""
You are writing a short gaming-lore video narration.

TOPIC: Gaunter O'Dimm from The Witcher 3: Wild Hunt
TARGET LENGTH: Approximately 30 seconds.

EVIDENCE NOTES:
<evidence>
{evidence}
</evidence>

STRICT RULES:
1. Treat the evidence as unverified notes, not independently
   confirmed truth.
2. Use only claims explicitly stated in the evidence.
3. Do not invent dialogue, scenes, objects, motives, or events.
4. After each factual sentence, add the supporting evidence ID,
   such as [E1].
5. If the notes do not support an interesting claim, leave it out.
6. Do not claim that you researched or verified anything.
7. Include a short hook, narration, and a final question for viewers.
8. Do not add factual claims in the hook or final question unless
   they also have evidence IDs.
9. Label the output "DRAFT — NOT FACT-CHECKED".

Return these sections:
HOOK:
NARRATION:
ENDING QUESTION:
CLAIMS REQUIRING MORE EVIDENCE:
"""

    script = ask_ollama(script_prompt)

    script_path = SCRIPT_DIR / f"gaunter_odimm_{timestamp}.txt"
    script_path.write_text(script, encoding="utf-8")

    print(f"Draft saved to: {script_path}")

    # Ask for an independent review of the draft
    print("Step 3/3: Reviewing claims against the evidence...")

    review_prompt = f"""
You are a strict factuality reviewer for gaming-lore videos.

Compare the draft with the supplied evidence notes.

EVIDENCE NOTES:
<evidence>
{evidence}
</evidence>

DRAFT:
<draft>
{script}
</draft>

For every factual claim in the draft:
- Quote or summarise the claim.
- Identify its cited evidence ID, if any.
- Decide whether the supplied notes explicitly support it.
- Mark it SUPPORTED BY NOTES, UNSUPPORTED, or UNCLEAR.
- Explain briefly why.

Also identify:
- Claims without evidence IDs.
- Invented details or implications.
- Any claim that the notes themselves label as unverified.

Finish with a verdict:
PASS, REVIEW REQUIRED, or FAIL.

IMPORTANT:
You must not use your general knowledge to upgrade an unsupported
claim to verified fact. The evidence notes are explicitly unverified.
A claim supported by the notes is NOT necessarily independently
verified. Do not describe this review as proof of truth.

Return plain text with clear headings.
"""

    review = ask_ollama(review_prompt)

    review_path = (
        FACT_CHECK_DIR / f"gaunter_odimm_review_{timestamp}.txt"
    )
    review_path.write_text(review, encoding="utf-8")

    print("\nFinished.")
    print(f"Draft:  {script_path}")
    print(f"Review: {review_path}")
    print("\n--- GENERATED DRAFT ---\n")
    print(script)
    print("\n--- FACT-CHECK REPORT ---\n")
    print(review)
    print(
        "\nReminder: This is an AI-assisted review, not independent "
        "verification. Do not publish the draft yet."
    )


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print(
            "ERROR: Cannot connect to Ollama. Make sure Ollama is running."
        )
    except requests.exceptions.RequestException as error:
        print(f"ERROR communicating with Ollama: {error}")
    except Exception as error:
        print(f"ERROR: {error}")
