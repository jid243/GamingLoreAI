"""Tests for the script stage (story angles, multi-script selection, repetition, grounding gates, outputs).
The language model is MOCKED, so these prove the pipeline's logic and file outputs, not real-model story quality."""
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_HOME = tempfile.mkdtemp()
os.environ["HOME"] = _HOME
os.environ["USERPROFILE"] = _HOME
os.environ["HOOK_SEED"] = "1"
sys.path.insert(0, str(Path(__file__).parent))
import generate_grounded_script_google as g  # noqa: E402
g.GEMINI_KEY_FILE = g.GROQ_KEY_FILE = g.PROJECT / "__tests_never_use_a_real_key__"   # tests run the local writer only
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("GROQ_API_KEY", None)

GAME = "Ashfall Keep"
TOPIC = "The granary choice"
EVIDENCE = """GAME: Ashfall Keep
TOPIC: The granary choice

[E001]
Source: wiki
Excerpt: The village healer Mirela must decide whether to burn the plague-ridden granary, which also holds the town's last seed grain.

[E002]
Source: wiki
Excerpt: Mirela's sister lies sick inside the granary and cannot be moved.

[E003]
Source: wiki
Excerpt: If the granary burns, the plague stops spreading, but the town has no seed for spring planting.

[E004]
Source: wiki
Excerpt: The town elders have given Mirela until dawn to decide.

[E005]
Source: wiki
Excerpt: The plague began because the town elders stored infected grain without telling the healer.
"""
IDS = {"E001", "E002", "E003", "E004", "E005"}
PASSAGES = {p["id"]: p["text"] for p in g.split_evidence_passages(EVIDENCE)}

HOOK_A = "A healer must burn a plague-ridden granary or save the town's last seed grain. [E001]"
HOOK_B = "In Ashfall Keep, a sick sister lies inside a granary the healer may burn. [E002]"

SCRIPT_A = f"""STATUS: DRAFT
HOOK: {HOOK_A}
NARRATION: In Ashfall Keep, the healer is Mirela, and the town elders have given her until dawn. [E001] [E004]
NARRATION: Her own sister lies sick inside the granary and cannot be moved. [E002]
PAYOFF: Burning it would stop the plague, but leave the town no seed for spring planting. [E003]
ENDING QUESTION: Would you burn the granary with your sister still inside it? [E002]
CLAIMS REQUIRING MORE EVIDENCE: None"""

SCRIPT_B_GOOD = f"""STATUS: DRAFT
HOOK: {HOOK_B}
NARRATION: The healer, Mirela, has until dawn to decide whether to burn the plague-ridden building. [E001] [E004]
NARRATION: If it burns, the plague stops spreading, but nobody will have seed for spring planting. [E003]
PAYOFF: Mirela alone carries the choice between the town's seed and her own sister. [E001]
ENDING QUESTION: Is saving the whole town worth losing the sister who cannot be moved? [E002]
CLAIMS REQUIRING MORE EVIDENCE: None"""

# script for hook B that restates the hook in its first narration line (must be rejected as repetition)
SCRIPT_B_REPEATS = f"""STATUS: DRAFT
HOOK: {HOOK_B}
NARRATION: A sick sister lies inside the granary that the healer may burn. [E002]
NARRATION: The healer Mirela has until dawn to decide whether to burn the granary. [E001] [E004]
NARRATION: If it burns, the plague stops spreading, but nobody will have seed for spring planting. [E003]
PAYOFF: Mirela alone carries the choice between the town's seed and her own sister. [E001]
ENDING QUESTION: Is saving the whole town worth losing the sister who cannot be moved? [E002]
CLAIMS REQUIRING MORE EVIDENCE: None"""

HOOK_GEN = (f"HOOK 1 (MORAL_DILEMMA): {HOOK_A}\nHOOK 2 (IMMEDIATE_DANGER): {HOOK_B}\n"
            "HOOK 3 (MYSTERY): NONE - no mystery stated\nHOOK 4 (UNEXPECTED_CONSEQUENCE): NONE - none stated\n"
            "HOOK 5 (EMOTIONAL_CONFLICT): NONE - none stated")
FACTS = ("FACT 1: A healer must choose whether to burn a plague-ridden granary. || QUOTE: must decide whether to burn the plague-ridden granary || ID: E001\n"
         "FACT 2: A sister lies sick inside and cannot be moved. || QUOTE: sister lies sick inside the granary and cannot be moved || ID: E002\n"
         "FACT 3: Burning it stops the plague but leaves no seed. || QUOTE: the plague stops spreading, but the town has no seed || ID: E003")

FACTS_CAUSE = FACTS + ("\nFACT 4: The plague began because the elders hid infected grain from the healer. || "
                       "QUOTE: stored infected grain without telling the healer || ID: E005 || KIND: cause")

SCRIPT_A_CAUSE = f"""STATUS: DRAFT
HOOK: {HOOK_A}
NARRATION: In Ashfall Keep, the healer is Mirela, and the town elders have given her until dawn. [E001] [E004]
NARRATION: The plague began because the elders stored infected grain without telling the healer. [E005]
NARRATION: Her own sister lies sick inside the granary and cannot be moved. [E002]
PAYOFF: Burning it would stop the plague, but leave the town no seed for spring planting. [E003]
ENDING QUESTION: Would you burn the granary with your sister still inside it? [E002]
CLAIMS REQUIRING MORE EVIDENCE: None"""


class ScriptFake:
    """Scripted stand-in for Ollama. scripts: {hook: [draft, ...]}; unsupported: line texts the auditor cannot support."""

    def __init__(self, scripts, eval_scores=None, coherence="DELIVERS", unsupported=(), hook_gen=HOOK_GEN, retention="GOOD",
                 facts=FACTS, brief="PLAIN_SETUP: A healer faces a choice.\nMISSING: NONE", targeted=None):
        self.scripts, self.eval_scores, self.coherence = scripts, eval_scores or {}, coherence
        self.unsupported, self.hook_gen, self.retention, self.facts = set(unsupported), hook_gen, retention, facts
        self.brief, self.targeted = brief, targeted
        self.calls, self.script_prompts = [], []

    def __call__(self, prompt, temperature=0.2, model=None, options=None):
        if "Does any passage below state this claim?" in prompt:
            return getattr(self, "locate_reply", "NONE")
        if "will tell the story of" in prompt and "CANDIDATES:" in prompt:
            return "IDS: NONE"
        if "Analyse the evidence" in prompt:
            self.calls.append("brief")
            return self.brief
        if "Someone proposed leaving this detail out" in prompt:
            self.calls.append("causal?")
            item = prompt.split("leaving this detail out:", 1)[1].split("\n")[0]
            return "YES: it explains how the plague started" if "elders" in item else "NO"
        if "may state the following" in prompt:
            self.calls.append("targeted")
            return self.targeted or "NONE"
        if "pull out" in prompt and "FACT 1:" in prompt:
            self.calls.append("facts")
            return self.facts
        if "write the first spoken sentence" in prompt:
            self.calls.append("hookgen")
            return self.hook_gen
        if "distracted viewer" in prompt:
            self.calls.append("hookeval")
            rows = []
            for line in prompt.split("HOOKS:\n", 1)[1].splitlines():
                number = line.split(":", 1)[0].replace("HOOK ", "")
                rows.append(f"HOOK {number}: CLARITY=5 SPECIFICITY=4 CURIOSITY=4 DELIVERY=4 UNEXPLAINED=NONE ANY_GAME=NO WHY=ok")
            return "\n".join(rows)
        if "Read this short video script. For each numbered point" in prompt:
            self.calls.append("essential?")
            return "unclear"      # unusable reply -> the code falls back to its own wording-overlap check
        if "You are the story editor" in prompt:
            self.calls.append("editor")
            draft = prompt.split("DRAFT:\n", 1)[1]
            return "STATUS: DRAFT\n" + draft + "\nCLAIMS REQUIRING MORE EVIDENCE: None"
        if "Tell the WHOLE story" in prompt:
            self.calls.append("fullstory")
            self.full_prompts = getattr(self, "full_prompts", []) + [prompt]
            return "FULL TELLING: Mirela must choose, and her sister lies inside, and the elders hid the grain."
        if "gifted short-form storyteller" in prompt:
            self.calls.append("scriptgen")
            self.script_prompts.append(prompt)
            marker = "SELECTED HOOK: " if "SELECTED HOOK: " in prompt else "HOOK (fixed, copy it exactly): "
            hook = prompt.split(marker, 1)[1].splitlines()[0].strip()
            drafts = self.scripts[hook]
            return drafts.pop(0) if len(drafts) > 1 else drafts[0]
        if "Rewrite ONE flagged line" in prompt:
            self.calls.append("repair")
            return "NOTHING USEFUL"
        if "fair fact-checker" in prompt:
            self.calls.append("audit")
            text = prompt.split("LINE (", 1)[1].split("): ", 1)[1].strip()
            if text in self.unsupported:
                return "CLAIM 1: invented detail || QUOTE: NONE"
            excerpt = re.search(r"\[E\d+\] (.+)", prompt).group(1)
            return f"CLAIM 1: {text} || QUOTE: {' '.join(excerpt.split()[:6])}"
        if "Assess only story craft" in prompt:
            self.calls.append("retention")
            return f"RETENTION_VALUATION: {self.retention}\nVERDICT: {'PASS' if self.retention != 'POOR' else 'FAIL'}"
        if "keeps the promise" in prompt:
            self.calls.append("coherence")
            return f"COHERENCE: {self.coherence}\nREASON: test"
        if "scrolling short videos" in prompt:
            self.calls.append("scripteval")
            first = prompt.split("SCRIPT:\n", 1)[1].splitlines()[0].strip()
            c, o, p, d = self.eval_scores.get(first, (4, 4, 4, 4))
            return f"CLARITY={c} OPENING={o} PROGRESSION={p} DELIVERY={d} REPEATS=NO GENERIC_ENDING=NO WHY=test"
        raise AssertionError("unexpected prompt: " + prompt[:80])


class PipelineHarness(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.sources, self.scripts_dir, self.checks = self.root / "sources", self.root / "scripts", self.root / "fact_checks"
        for d in (self.sources, self.scripts_dir, self.checks):
            d.mkdir()
        (self.sources / "evidence_passages.txt").write_text(EVIDENCE, encoding="utf-8")
        self.patches = [
            mock.patch.object(g, "EVIDENCE_FILE", self.sources / "evidence_passages.txt"),
            mock.patch.object(g, "SCRIPT_DIR", self.scripts_dir),
            mock.patch.object(g, "CHECK_DIR", self.checks),
            mock.patch.object(g, "read_project_context", lambda: (GAME, TOPIC, Path("research.txt"))),
            mock.patch.object(g, "select_relevant_evidence", lambda e, t, gm="": (EVIDENCE, set(IDS), None)),
            mock.patch.object(g, "SCRIPT_CANDIDATES", 2),
            mock.patch.object(g, "DRAFTS_PER_ATTEMPT", 1),     # multi-draft and the editor pass have their own tests
            mock.patch.object(g, "REVISE_PASS", False),
            mock.patch.object(sys, "argv", ["generate_grounded_script_google.py"]),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def run_main(self, fake):
        with mock.patch.object(g, "ask_ollama", fake):
            g.main()

    def outputs(self, suffix):
        return sorted(self.scripts_dir.glob(f"*_{suffix}.txt"))


class ScriptStageTests(PipelineHarness):
    def test_script_a_alone_is_a_valid_script(self):
        self.assertEqual(g.check_script_format(SCRIPT_A, IDS, HOOK_A, GAME), [])
        self.assertEqual(g.check_script_format(SCRIPT_B_GOOD, IDS, HOOK_B, GAME), [])

    def test_repetitive_narration_is_detected_and_penalised_but_is_not_a_layout_failure(self):
        self.assertEqual(g.check_script_format(SCRIPT_B_REPEATS, IDS, HOOK_B, GAME), [])     # story craft, not layout
        self.assertTrue(any(fails for *_, fails in g.script_repetition(SCRIPT_B_REPEATS, GAME)))
        # the user's real example: hook/narration repetition, overused adjectives, generic ending
        old = """STATUS: DRAFT
HOOK: A monstrous Botchling, born from improper burial, hunts pregnant women. [E005]
NARRATION: In the Witcher 3, Geralt of Rivia encounters a Botchling, a creature born from the improper burial of a stillborn child. [E005]
NARRATION: This grotesque creature, twisted with hate and malice, preys on pregnant women, driven by a mad hunger for their blood. [E005]
NARRATION: One can transform a botchling into a lubberkin. [E161]
PAYOFF: Geralt can turn the botchling into a lubberkin. [E197]
ENDING QUESTION: What other creatures in The Witcher 3 are born from tragic circumstances? [E005]
CLAIMS REQUIRING MORE EVIDENCE: None"""
        clear = [{"status": "CLEAR", "section": "X", "reason": ""}] * 6
        score = g.score_script(old, "The Witcher 3", clear, {"status": "GOOD", "verdict": "PASS", "text": ""},
                               {"verdict": "DELIVERS", "text": ""}, [], None)
        reasons = " | ".join(p["reason"] for p in score["penalties"])
        self.assertIn("repeated information", reasons)
        self.assertNotIn("overused adjectives", reasons)     # one in the fixed hook + one later is allowed
        twice = old.replace("This grotesque creature", "This grotesque, hideous creature")
        reasons2 = " | ".join(p["reason"] for p in g.score_script(twice, "The Witcher 3", clear, {"status": "GOOD", "verdict": "PASS", "text": ""},
                                                                 {"verdict": "DELIVERS", "text": ""}, [], None)["penalties"])
        self.assertIn("overused adjectives", reasons2)
        self.assertIn("generic", reasons)
        good = g.score_script(SCRIPT_A, GAME, clear, {"status": "GOOD", "verdict": "PASS", "text": ""},
                              {"verdict": "DELIVERS", "text": ""}, [], None)
        self.assertEqual(good["penalties"], [])

    def test_end_to_end_selects_the_eligible_script_and_writes_compatible_outputs(self):
        fake = ScriptFake({HOOK_A: [SCRIPT_A], HOOK_B: [SCRIPT_B_REPEATS]})
        self.run_main(fake)
        files = self.outputs("REVIEW_REQUIRED")
        self.assertEqual(len(files), 1, "expected exactly one REVIEW_REQUIRED draft")
        text = files[0].read_text(encoding="utf-8")
        self.assertEqual(text.strip(), SCRIPT_A.strip())                      # chosen script, untouched
        # downstream compatibility: tags parse, hook identical, every line keeps its evidence ids, format check passes
        lines = g.spoken_lines(text)
        self.assertEqual([s for s, _ in lines], ["HOOK", "NARRATION", "NARRATION", "PAYOFF", "ENDING QUESTION"])
        self.assertTrue(all(g.line_citations(c) for _, c in lines))
        self.assertEqual(g.check_script_format(text, IDS, HOOK_A, GAME), [])
        narration = self.outputs("NARRATION")[0].read_text(encoding="utf-8")
        self.assertNotRegex(narration, r"\[E\d+\]")
        review = list(self.checks.glob("*_review.txt"))[0].read_text(encoding="utf-8")
        self.assertIn("SCRIPT SCORE", review)
        self.assertIn("CANDIDATE SCRIPTS COMPARED", review)
        data = json.loads(list(self.checks.glob("*_scripts.json"))[0].read_text(encoding="utf-8"))
        self.assertEqual(len(data["candidates"]), 2)
        by_hook = {c["hook"]: c for c in data["candidates"]}
        self.assertTrue(by_hook[HOOK_A]["score"]["eligible"])
        # B restated its hook: a craft flaw that is scored (and fed back), not a reason to discard the story
        self.assertTrue(any("repeated information" in p["reason"] for p in by_hook[HOOK_B]["score"]["penalties"]))
        self.assertLess(by_hook[HOOK_B]["score"]["total"], by_hook[HOOK_A]["score"]["total"])
        self.assertTrue(list(self.checks.glob("*_hooks.json")))

    def test_the_stronger_script_wins_even_when_it_is_not_first(self):
        first_line_a = g.clean_narration(SCRIPT_A).splitlines()[0]
        first_line_b = g.clean_narration(SCRIPT_B_GOOD).splitlines()[0]
        fake = ScriptFake({HOOK_A: [SCRIPT_A], HOOK_B: [SCRIPT_B_GOOD]},
                          eval_scores={first_line_a: (3, 3, 3, 3), first_line_b: (5, 5, 5, 4)})
        self.run_main(fake)
        text = self.outputs("REVIEW_REQUIRED")[0].read_text(encoding="utf-8")
        self.assertEqual(text.strip(), SCRIPT_B_GOOD.strip())
        data = json.loads(list(self.checks.glob("*_scripts.json"))[0].read_text(encoding="utf-8"))
        totals = sorted(c["score"]["total"] for c in data["candidates"])
        self.assertGreater(totals[1], totals[0])

    def test_unsupported_claims_are_rejected_whatever_the_score(self):
        payoff = "Burning it would stop the plague, but leave the town no seed for spring planting."
        fake = ScriptFake({HOOK_A: [SCRIPT_A], HOOK_B: [SCRIPT_B_GOOD]}, unsupported=[payoff,
                          "Mirela alone carries the choice between the town's seed and her own sister."],
                          eval_scores={g.clean_narration(SCRIPT_A).splitlines()[0]: (5, 5, 5, 5)})
        self.run_main(fake)
        self.assertEqual(self.outputs("REVIEW_REQUIRED"), [])
        rejected = self.outputs("REJECTED")
        self.assertEqual(len(rejected), 1)
        report = list(self.checks.glob("*_review.txt"))[0].read_text(encoding="utf-8")
        self.assertIn("WHAT THE EVIDENCE DID NOT SUPPORT", report)
        self.assertIn("invented detail", report)
        draft = rejected[0].read_text(encoding="utf-8")
        self.assertNotIn("CLAIMS REQUIRING MORE EVIDENCE: None", draft)       # the gap is stated, not hidden
        self.assertIn("no available passage states it", draft)

    def test_a_script_that_does_not_fully_pay_off_its_hook_is_penalised_not_rejected(self):
        fake = ScriptFake({HOOK_A: [SCRIPT_A], HOOK_B: [SCRIPT_B_GOOD]}, coherence="FAILS")
        self.run_main(fake)
        self.assertEqual(len(self.outputs("REVIEW_REQUIRED")), 1)        # nothing was made up, so the story stands
        data = json.loads(list(self.checks.glob("*_scripts.json"))[0].read_text(encoding="utf-8"))
        for c in data["candidates"]:
            self.assertTrue(any("pay off the hook" in p["reason"] for p in c["score"]["penalties"]))
            self.assertTrue(any("pay off the hook" in w for w in c["score"]["warnings"]))

    def test_insufficient_evidence_stops_before_any_script_is_written(self):
        none_gen = "\n".join(f"HOOK {i} ({n}): NONE - not supported" for i, (n, _) in enumerate(g.HOOK_APPROACHES, 1))
        fake = ScriptFake({}, hook_gen=none_gen)
        self.run_main(fake)
        self.assertEqual(list(self.scripts_dir.iterdir()), [])
        self.assertNotIn("scriptgen", fake.calls)
        self.assertTrue(list(self.checks.glob("*_hooks.json")))

    def test_single_candidate_keeps_the_old_one_script_flow(self):
        with mock.patch.object(g, "SCRIPT_CANDIDATES", 1):
            fake = ScriptFake({HOOK_A: [SCRIPT_A], HOOK_B: [SCRIPT_B_GOOD]})
            self.run_main(fake)
        self.assertEqual(fake.calls.count("scriptgen"), 1)
        self.assertEqual(len(self.outputs("REVIEW_REQUIRED")), 1)


class EndingAndAttemptTests(PipelineHarness):
    def test_obvious_ending_questions_are_caught_and_real_dilemmas_are_not(self):
        for q in ("How do you think the elders will react to the burning? [E004]",
                  "Are the villagers happy that the granary burned? [E003]",
                  "How would Mirela feel about her sister afterwards? [E002]"):
            self.assertTrue(g.weak_ending_reason(q), q)
        for q in ("Is saving the whole town worth losing the sister who cannot be moved? [E002]",
                  "Would you burn the granary with your sister still inside it? [E002]"):
            self.assertEqual(g.weak_ending_reason(q), "", q)
        bad = SCRIPT_A.replace("Would you burn the granary with your sister still inside it?", "How do you think the elders will react to the fire?")
        self.assertTrue(any("answer is obvious" in p for p in g.check_script_format(bad, IDS, HOOK_A, GAME)))
        prompt = g.build_generation_prompt(GAME, TOPIC, EVIDENCE, IDS, HOOK_A, "B")
        self.assertIn("two defensible answers", prompt)
        self.assertIn("Never ask how a character feels", prompt)

    def test_an_obvious_ending_is_penalised_in_the_score(self):
        bad = SCRIPT_A.replace("Would you burn the granary with your sister still inside it?", "How do you think the elders feel about the fire?")
        audits = [{"status": "CLEAR"}]
        good_score = g.score_script(SCRIPT_A, GAME, audits, {"status": "GOOD", "verdict": "PASS"}, None, [], {"clarity": 4, "opening": 4, "progression": 4, "delivery": 4, "generic_ending": False, "why": ""})
        bad_score = g.score_script(bad, GAME, audits, {"status": "GOOD", "verdict": "PASS"}, None, [], {"clarity": 4, "opening": 4, "progression": 4, "delivery": 4, "generic_ending": False, "why": ""})
        self.assertTrue(any("obvious" in x["reason"] for x in bad_score["penalties"]))
        self.assertLess(bad_score["total"], good_score["total"])

    def test_the_game_title_is_added_in_code_when_the_hook_lacks_it(self):
        hook = "A healer must burn a plague-ridden granary. [E001]"
        draft = SCRIPT_A.replace(HOOK_A, hook).replace("In Ashfall Keep, the healer is Mirela", "The healer is Mirela")
        fixed, changed = g.ensure_game_title(draft, GAME, hook)
        self.assertTrue(changed)
        self.assertIn("In Ashfall Keep, the healer is Mirela", fixed)
        self.assertEqual(g.check_script_format(fixed, IDS, hook, GAME), [])
        again, changed2 = g.ensure_game_title(fixed, GAME, hook)
        self.assertFalse(changed2)
        _, untouched = g.ensure_game_title(SCRIPT_B_GOOD, GAME, HOOK_B)   # hook already names the game
        self.assertFalse(untouched)

    def test_an_audited_attempt_outranks_an_unaudited_one_and_layout_is_not_an_evidence_gap(self):
        note = g.missing_evidence_note([], ["Mention the game title in the first NARRATION line."], None)
        self.assertEqual(note, [])
        calls = {"n": 0}
        base = ScriptFake({HOOK_B: [SCRIPT_B_REPEATS]})

        def fake(prompt, temperature=0.2, model=None, options=None):
            if "Rewrite ONE flagged line" in prompt:
                return "NOTHING USEFUL"
            return base(prompt, temperature, model, options)
        with mock.patch.object(g, "ask_ollama", fake):
            result = g.produce_script(GAME, TOPIC, EVIDENCE, IDS, PASSAGES, HOOK_B, "brief", {}, 2)
        self.assertTrue(result["audits"])                 # it was fact-checked even though a repeat could not be repaired
        self.assertTrue(result["passed"])                 # repetition is scored, not a gate
        self.assertEqual(len(result["score"]), 6)

    def test_a_repeated_line_is_rewritten_with_a_new_fact(self):
        base = ScriptFake({HOOK_B: [SCRIPT_B_REPEATS]})
        replacement = "NARRATION: The town elders gave her only until dawn to decide. [E004]"

        def fake(prompt, temperature=0.2, model=None, options=None):
            if "Rewrite ONE flagged line" in prompt:
                self.assertIn("restates a fact an earlier line already told", prompt)
                return replacement
            return base(prompt, temperature, model, options)
        with mock.patch.object(g, "ask_ollama", fake):
            result = g.produce_script(GAME, TOPIC, EVIDENCE, IDS, PASSAGES, HOOK_B, "brief", {}, 1)
        self.assertIn("until dawn to decide", result["draft"])
        self.assertEqual(g.check_script_format(result["draft"], IDS, HOOK_B, GAME), [])

    def test_a_rejected_draft_is_audited_so_its_unsupported_claims_are_named(self):
        fake = ScriptFake({HOOK_A: [SCRIPT_A], HOOK_B: [SCRIPT_B_GOOD]},
                          unsupported={"Burning it would stop the plague, but leave the town no seed for spring planting.",
                                       "Mirela alone carries the choice between the town's seed and her own sister."})
        self.run_main(fake)
        self.assertEqual(len(self.outputs("REJECTED")), 1)
        review = next(self.checks.glob("*_review.txt")).read_text(encoding="utf-8")
        self.assertIn("UNSUPPORTED", review)


class RootCauseTests(PipelineHarness):
    FACT = {"id": "E005", "kind": "cause", "quote": "stored infected grain without telling the healer",
            "fact": "The plague began because the elders hid infected grain from the healer."}

    def test_cause_facts_are_parsed_by_kind_and_verified(self):
        with mock.patch.object(g, "ask_ollama", lambda *a, **k: FACTS_CAUSE):
            facts, dropped = g.extract_hook_facts(GAME, TOPIC, EVIDENCE, IDS, "BRIEF")
        self.assertEqual([f["kind"] for f in facts], ["detail"] * 3 + ["cause"])
        self.assertEqual([f["id"] for f in g.essential_facts(facts)], ["E005"])
        with mock.patch.object(g, "REQUIRE_ROOT_CAUSE", False):
            self.assertEqual(g.essential_facts(facts), [])

    def test_a_script_that_omits_the_cause_is_regenerated_with_it(self):
        self.assertTrue(g.missing_essential(SCRIPT_A, [self.FACT], GAME))
        self.assertFalse(g.missing_essential(SCRIPT_A_CAUSE, [self.FACT], GAME))
        # judged by meaning: a story that tells the cause in its own words passes
        with mock.patch.object(g, "ask_ollama", lambda p, **k: "1: YES"):
            self.assertEqual(g.missing_essential_semantic(GAME, SCRIPT_A, [self.FACT]), [])
        with mock.patch.object(g, "ask_ollama", lambda p, **k: "1: NO"):
            self.assertEqual(g.missing_essential_semantic(GAME, SCRIPT_A_CAUSE, [self.FACT]), [self.FACT])

        fake = ScriptFake({HOOK_A: [SCRIPT_A, SCRIPT_A_CAUSE], HOOK_B: [SCRIPT_B_GOOD]}, facts=FACTS_CAUSE)
        self.run_main(fake)
        text = self.outputs("REVIEW_REQUIRED")[0].read_text(encoding="utf-8")
        self.assertEqual(text.strip(), SCRIPT_A_CAUSE.strip())               # the version that states the cause
        retry_prompt = fake.script_prompts[1]
        self.assertIn("left out points it must tell", retry_prompt)          # feedback told the writer what to add
        self.assertIn("MUST INCLUDE", fake.script_prompts[0])
        data = json.loads(list(self.checks.glob("*_scripts.json"))[0].read_text(encoding="utf-8"))
        self.assertEqual(data["essential_facts"][0]["id"], "E005")

    def test_hook_citations_are_repaired_from_the_facts_they_use(self):
        facts = [{"id": "E003", "kind": "consequence", "quote": "q",
                  "fact": "Burning the granary stops the plague but leaves no seed for spring planting."}]
        fixed = g.recite_from_facts("Burning the granary stops the plague, but leaves no seed for planting. [E001]", facts, GAME)
        self.assertEqual(g.line_citations(fixed), ["E003", "E001"])
        untouched = "A completely different sentence about a ferryman crossing a river. [E001]"
        self.assertEqual(g.recite_from_facts(untouched, facts, GAME), untouched)

    def test_a_longer_story_is_allowed(self):
        longer = SCRIPT_A_CAUSE.replace(
            "NARRATION: Her own sister lies sick inside the granary and cannot be moved. [E002]",
            "NARRATION: Her own sister lies sick inside the granary and cannot be moved by anybody who might carry her out. [E002]\n"
            "NARRATION: No other villager may make this choice for her, and every hour of waiting lets the sickness travel further. [E004]")
        words = g.word_count(longer)
        self.assertGreater(words, 90)
        self.assertEqual(g.check_script_format(longer, IDS, HOOK_A, GAME, [self.FACT]), [])


class CauseSearchTests(PipelineHarness):
    BRIEF = ("PLAIN_SETUP: A healer faces a choice.\nROOT_CAUSE: NONE\n"
             "DETAILS_TO_OMIT: The elders' handling of the grain, Ciri's involvement.\nMISSING: NONE")
    TARGETED = ("FACT: The plague began because the elders hid infected grain from the healer. || "
                "QUOTE: stored infected grain without telling the healer || ID: E005")

    def test_a_cause_the_brief_wanted_omitted_is_found_verified_and_required(self):
        fake = ScriptFake({HOOK_A: [SCRIPT_A, SCRIPT_A_CAUSE], HOOK_B: [SCRIPT_B_GOOD]}, brief=self.BRIEF, targeted=self.TARGETED)
        self.run_main(fake)
        self.assertIn("causal?", fake.calls)
        self.assertIn("targeted", fake.calls)
        text = self.outputs("REVIEW_REQUIRED")[0].read_text(encoding="utf-8")
        self.assertEqual(text.strip(), SCRIPT_A_CAUSE.strip())
        first_prompt = fake.script_prompts[0]
        self.assertIn("MUST INCLUDE", first_prompt)
        self.assertNotIn("DETAILS_TO_OMIT", first_prompt)            # the writer only gets the story material, not the brief's lists

    def test_targeted_fact_needs_a_real_quote(self):
        def ask(reply):
            with mock.patch.object(g, "ask_ollama", lambda *a, **k: reply):
                return g.extract_targeted_fact(GAME, TOPIC, EVIDENCE, IDS, "item")
        self.assertEqual(ask(self.TARGETED)["id"], "E005")
        self.assertIsNone(ask("NONE"))
        self.assertIsNone(ask(self.TARGETED.replace("stored infected grain without telling the healer", "hid the grain in a cellar")))
        self.assertIsNone(ask(self.TARGETED.replace("E005", "E099")))

    def test_hook_citations_are_repaired_when_the_quote_lives_in_another_passage(self):
        good = "Burning it stops the plague, but leaves no seed for spring planting. [E001]"
        def run(reply):
            with mock.patch.object(g, "ask_ollama", lambda *a, **k: reply):
                return g.audit_hook_candidates([{"text": good}], EVIDENCE, GAME)[0]
        fixed = run("CLAIM 1: burning stops the plague || QUOTE: the plague stops spreading")      # that quote is in E003
        self.assertTrue(fixed["grounded"])
        self.assertEqual(g.line_citations(fixed["text"]), ["E003"])
        self.assertFalse(run("CLAIM 1: invented || QUOTE: NONE")["grounded"])
        self.assertFalse(run("CLAIM 1: invented || QUOTE: the sky turned green and the elders wept")["grounded"])


class FactVerificationTests(unittest.TestCase):
    def test_cause_facts_need_a_link_word_and_a_supporting_passage(self):
        facts = [
            {"id": "E005", "kind": "cause", "quote": "q", "fact": "The plague began because the elders hid infected grain from the healer."},
            {"id": "E004", "kind": "cause", "quote": "q", "fact": "A subplot is involved in the quest and helped with tasks."},      # no link word
            {"id": "E002", "kind": "cause", "quote": "q", "fact": "The sister caused the fire that destroyed the town."},               # unsupported
            {"id": "E001", "kind": "detail", "quote": "q", "fact": "The healer must decide."},                                         # not a cause fact: untouched
        ]
        unsupported = "The sister caused the fire that destroyed the town."
        def audit(prompt, **k):
            if "Does any passage below state this claim?" in prompt:
                return "NONE"
            text = prompt.split("LINE (", 1)[1].split("): ", 1)[1].strip()
            excerpt = re.search(r"\[E\d+\] (.+)", prompt).group(1)
            return "CLAIM 1: x || QUOTE: NONE" if unsupported in text else f"CLAIM 1: x || QUOTE: {' '.join(excerpt.split()[:6])}"
        with mock.patch.object(g, "ask_ollama", audit):
            kept, notes = g.verify_cause_facts(GAME, facts, PASSAGES)
        self.assertEqual([f["id"] for f in kept], ["E005", "E004", "E001"])   # a verified non-causal fact is kept...
        self.assertEqual([f["kind"] for f in kept], ["cause", "detail", "detail"])  # ...but demoted, so it is never required
        self.assertEqual(len(notes), 2)
        self.assertEqual([f["id"] for f in g.essential_facts(facts)], ["E005", "E002"])       # connector filter alone

    def test_the_audit_and_writer_accept_plain_descriptive_labels(self):
        captured = {}
        with mock.patch.object(g, "ask_ollama", lambda prompt, **k: captured.setdefault("p", prompt) and "CLAIM 1: x || QUOTE: NONE"):
            g.audit_line("NARRATION", "A monstrous creature appears. [E001]", PASSAGES, GAME)
        self.assertIn("NOT a claim", captured["p"])
        self.assertIn("monstrous", captured["p"])
        self.assertIn('"monstrous" is fine', g.build_generation_prompt(GAME, TOPIC, EVIDENCE, IDS, HOOK_A, "B"))

    def test_the_writer_is_told_exactly_where_to_name_the_game(self):
        no_title = g.build_generation_prompt(GAME, TOPIC, EVIDENCE, IDS, "A healer must choose. [E001]", "B")
        self.assertIn('MUST contain the exact words "Ashfall Keep"', no_title)
        with_title = g.build_generation_prompt(GAME, TOPIC, EVIDENCE, IDS, HOOK_B, "B")
        self.assertNotIn("MUST contain the exact words", with_title)
        bad = SCRIPT_A.replace("In Ashfall Keep, ", "")
        problems = g.check_script_format(bad, IDS, HOOK_A, GAME)
        self.assertTrue(any('exact words "Ashfall Keep"' in p for p in problems), problems)

    def test_blockers_do_not_pile_up_when_the_audit_never_ran(self):
        score = g.score_script(SCRIPT_A, GAME, [], None, None, ["Missing something."], None)
        self.assertFalse(any("retention" in b for b in score["blockers"]))
        self.assertIn("style: Missing something.", score["warnings"])     # style is reported, never a blocker
        self.assertEqual(score["blockers"], ["no fact audit was run (earlier checks failed)"])


class OmitProtectionTests(unittest.TestCase):
    def test_model_judges_which_omit_items_are_causal(self):
        brief = "DETAILS_TO_OMIT: The elders' handling of the grain, Ciri's involvement.\nBEST_STORY_ANGLE: x"
        def judge(prompt, **k):
            item = prompt.split("leaving this detail out:", 1)[1].split("\n")[0]
            return "YES: it explains how the plague started" if "elders" in item else "NO"
        with mock.patch.object(g, "ask_ollama", judge):
            fixed, removed = g.protect_causal_items(GAME, TOPIC, brief)
        self.assertEqual(removed, ["The elders' handling of the grain"])
        self.assertIn("Ciri's involvement", fixed)
        self.assertNotIn("elders", fixed)
        with mock.patch.object(g, "ask_ollama", lambda prompt, **k: "YES"):       # a bare YES gives no link: not protected
            self.assertEqual(g.protect_causal_items(GAME, TOPIC, brief), (brief, []))
        with mock.patch.object(g, "ask_ollama", side_effect=RuntimeError("model down")):
            self.assertEqual(g.protect_causal_items(GAME, TOPIC, brief), (brief, []))   # fails safe, never crashes

    def test_the_brief_cannot_tell_the_writer_to_omit_the_cause(self):
        brief = ("ROOT_CAUSE: The baby was lost because the Baron beat his wife.\n"
                 "DETAILS_TO_OMIT: Background on the Baron's abusive behavior and Ciri's involvement.\n"
                 "BEST_STORY_ANGLE: x")
        cause = {"fact": "The wife lost the baby owing to the abuse she suffered from the Baron.", "id": "E005", "kind": "cause"}
        fixed, removed = g.protect_root_cause(brief, [cause["fact"]], "The Witcher 3")
        omit_line = [l for l in fixed.splitlines() if l.startswith("DETAILS_TO_OMIT")][0]
        self.assertNotIn("abusive", omit_line)
        self.assertIn("Ciri", omit_line)                      # genuinely peripheral material may still be omitted
        self.assertEqual(len(removed), 1)
        self.assertIn("BEST_STORY_ANGLE: x", fixed)           # everything else is untouched
        # nothing to protect -> unchanged
        self.assertEqual(g.protect_root_cause(brief, [], "G"), (brief, []))
        # if everything listed was protected the line becomes a safe generic instruction
        only = "DETAILS_TO_OMIT: The Baron's abuse."
        self.assertIn("never the cause", g.protect_root_cause(only, [cause["fact"]], "G")[0])

    def test_a_shared_character_name_alone_does_not_protect_an_unrelated_detail(self):
        brief = "DETAILS_TO_OMIT: Mirela's quarrel with a rival, the elders' handling of the grain."
        cause = "Mirela lost her sister because the elders hid infected grain."
        fixed, removed = g.protect_root_cause(brief, [cause], GAME)
        self.assertEqual(removed, ["the elders' handling of the grain"])      # shares real words (elders, grain)
        self.assertIn("quarrel with a rival", fixed)                           # shares only the name Mirela

    def test_brief_prompt_forbids_omitting_the_cause(self):
        captured = {}
        with mock.patch.object(g, "ask_ollama", lambda prompt, **k: captured.setdefault("p", prompt)):
            g.develop_story_angle(GAME, TOPIC, EVIDENCE, IDS)
        self.assertIn("NEVER list what caused the central situation", captured["p"])


class PromptTests(unittest.TestCase):
    def test_story_prompt_asks_for_progression_and_no_repetition(self):
        prompt = g.build_generation_prompt(GAME, TOPIC, EVIDENCE, IDS, HOOK_A, "BRIEF")
        for needle in ("NEW information", "never re-say what the hook", "Hold the most surprising fact back for the PAYOFF",
                       "never state the same fact twice", "monstrous", "Invent no facts", "ENDING QUESTION", "Each line should make the next one necessary"):
            self.assertIn(needle, prompt)
        for forbidden in ("Witcher", "Geralt", "Botchling"):
            self.assertNotIn(forbidden, prompt)

    def test_the_viewer_question_is_about_people_and_stakes_not_procedure(self):
        captured = {}
        with mock.patch.object(g, "ask_ollama", lambda prompt, **k: captured.setdefault("p", prompt) or "x"):
            g.develop_story_angle(GAME, TOPIC, EVIDENCE, IDS)
        self.assertIn("about PEOPLE and STAKES", captured["p"])
        self.assertIn("Never a how-to or mechanics question", captured["p"])
        self.assertIn("the supported answer to WHAT_VIEWERS_WILL_WANT_NEXT", captured["p"])

    def test_story_brief_asks_for_analysis_fields(self):
        captured = {}
        with mock.patch.object(g, "ask_ollama", lambda prompt, **k: captured.setdefault("p", prompt)):
            g.develop_story_angle(GAME, TOPIC, EVIDENCE, IDS)
        for field in ("MOST_SURPRISING_FACT", "WHAT_VIEWERS_WILL_WANT_NEXT", "DETAILS_TO_OMIT", "ALTERNATIVE_ANGLES", "MISSING"):
            self.assertIn(field, captured["p"])

    def test_prompts_carry_the_plain_language_and_cause_rules_without_game_names(self):
        facts = [{"id": "E005", "fact": "x", "quote": "y", "kind": "cause"}]
        hook_prompt = g.build_hook_prompt(GAME, TOPIC, EVIDENCE, IDS, "BRIEF", facts=facts)
        self.assertIn("CAUSE [E005]", hook_prompt)
        self.assertIn("plain description", hook_prompt)
        script_prompt = g.build_generation_prompt(GAME, TOPIC, EVIDENCE, IDS, HOOK_A, "BRIEF", essential=facts)
        self.assertIn("MUST INCLUDE", script_prompt)
        self.assertIn("never softened", script_prompt)
        for text in (hook_prompt, script_prompt):
            for forbidden in ("Witcher", "Geralt", "Botchling", "Baron"):
                self.assertNotIn(forbidden, text)

    def test_script_weights_sum_to_100(self):
        self.assertEqual(sum(g.SCRIPT_WEIGHTS.values()), 100)


class HookPickingTests(PipelineHarness):
    """You can pick the hook: a hooks-only run saves a ranked menu, then the script run writes the story for your pick."""

    def test_a_hooks_only_run_saves_a_ranked_menu_and_writes_no_script(self):
        fake = ScriptFake({HOOK_A: [SCRIPT_A], HOOK_B: [SCRIPT_B_GOOD]})
        with mock.patch.dict(os.environ, {"HOOKS_ONLY": "1"}):
            self.run_main(fake)
        self.assertNotIn("scriptgen", fake.calls)
        self.assertEqual(self.outputs("REVIEW_REQUIRED") + self.outputs("REJECTED"), [])
        menu = json.loads(next(self.checks.glob("*_hook_menu.json")).read_text(encoding="utf-8"))
        spoken = [h["spoken"] for h in menu["hooks"]]
        self.assertEqual(sorted(spoken), sorted([g.remove_citation(HOOK_A), g.remove_citation(HOOK_B)]))
        self.assertEqual(sum(h["recommended"] for h in menu["hooks"]), 1)
        self.assertTrue(menu["hooks"][0]["recommended"])                       # best first
        self.assertTrue(list(self.checks.glob("*_prepared.json")))

    def test_the_picked_hook_gets_its_script_without_redoing_the_preparation(self):
        with mock.patch.dict(os.environ, {"HOOKS_ONLY": "1"}):
            self.run_main(ScriptFake({HOOK_A: [SCRIPT_A], HOOK_B: [SCRIPT_B_GOOD]}))
        g.VERIFIED_HOOKS.clear()
        fake = ScriptFake({HOOK_A: [SCRIPT_A], HOOK_B: [SCRIPT_B_GOOD]})
        with mock.patch.dict(os.environ, {"CHOSEN_HOOK": g.remove_citation(HOOK_B)}):   # picked without its citation
            self.run_main(fake)
        self.assertNotIn("brief", fake.calls)          # preparation reused
        self.assertNotIn("hookgen", fake.calls)        # no new hooks
        self.assertEqual({p.split("SELECTED HOOK: ", 1)[1].splitlines()[0].strip() for p in fake.script_prompts}, {HOOK_B})
        text = (self.outputs("REVIEW_REQUIRED") or self.outputs("REJECTED"))[0].read_text(encoding="utf-8")
        self.assertIn("HOOK: " + HOOK_B, text)

    def test_a_picked_hook_gets_three_versions_each_opening_differently_all_saved_to_pick_from(self):
        fake = ScriptFake({HOOK_B: [SCRIPT_B_GOOD]})
        with mock.patch.dict(os.environ, {"CHOSEN_HOOK": HOOK_B}):
            self.run_main(fake)
        self.assertEqual(len(fake.script_prompts), 3)
        self.assertNotIn("THIS VERSION:", fake.script_prompts[0])
        self.assertIn("THIS VERSION: Open the narration on the people", fake.script_prompts[1])
        self.assertIn("THIS VERSION: Open the narration on the danger", fake.script_prompts[2])
        menu = json.loads(next(self.checks.glob("*_script_menu.json")).read_text(encoding="utf-8"))
        self.assertEqual([v["angle"] for v in menu["versions"]], ["Best judgement", "Opens on the people", "Opens on the danger"])
        self.assertEqual(sum(v["selected"] for v in menu["versions"]), 1)
        self.assertTrue(all(v["review"] and "HOOK: " in v["draft"] for v in menu["versions"]))
        self.assertEqual(len(self.outputs("REVIEW_REQUIRED") + self.outputs("REJECTED")), 1)   # one draft file, as before

    def test_the_writer_is_told_to_open_on_the_stakes_not_on_background(self):
        prompt = g.build_generation_prompt(GAME, TOPIC, EVIDENCE, IDS, HOOK_A, "brief")
        self.assertIn("THE FIRST NARRATION LINE decides whether the viewer stays", prompt)
        self.assertIn("Never open the narration with background", prompt)
        self.assertIn("PLAIN WORDS", prompt)

    def test_the_strongest_hooks_are_also_retold_in_plain_words_and_offered(self):
        plain = "A healer's sister lies sick inside the granary she may have to burn. [E002]"

        class PlainFake(ScriptFake):
            def __call__(self, prompt, **k):
                if "so that nobody could swipe away" in prompt:
                    self.calls.append("plain")
                    return plain
                return super().__call__(prompt, **k)
        fake = PlainFake({HOOK_A: [SCRIPT_A], HOOK_B: [SCRIPT_B_GOOD]})
        with mock.patch.dict(os.environ, {"HOOKS_ONLY": "1"}):
            self.run_main(fake)
        self.assertIn("plain", fake.calls)
        menu = json.loads(next(self.checks.glob("*_hook_menu.json")).read_text(encoding="utf-8"))
        rewrites = [h for h in menu["hooks"] if h["approach"].endswith("_PLAIN")]
        self.assertEqual([h["spoken"] for h in rewrites], [g.remove_citation(plain)])   # one new line, not a duplicate per source

    def test_a_picked_hook_whose_passage_this_run_did_not_select_still_gets_a_checked_script(self):
        others = "\n\n".join(g.compact_passage(p) for p in g.split_evidence_passages(EVIDENCE) if p["id"] != "E002")
        fake = ScriptFake({HOOK_B: [SCRIPT_B_GOOD]})
        with mock.patch.dict(os.environ, {"CHOSEN_HOOK": HOOK_B, "SCRIPT_VERSIONS": "1"}), \
                mock.patch.object(g, "select_relevant_evidence", lambda e, t, gm="": (others, IDS - {"E002"}, None)):
            self.run_main(fake)
        self.assertIn("audit", fake.calls)                                        # the fact check ran
        self.assertEqual(len(self.outputs("REVIEW_REQUIRED")), 1)

    def test_the_whole_story_is_told_first_then_cut_down_keeping_how_each_choice_plays_out(self):
        scene = {"id": "E003", "fact": "If the granary burns, the plague stops but the town has no seed.", "quote": "the plague stops spreading",
                 "kind": "consequence", "payoff_detail": True}
        fake = ScriptFake({HOOK_A: [SCRIPT_A]})
        with mock.patch.object(g, "ask_ollama", fake), mock.patch.object(g, "plan_story_beats", lambda *a, **k: "STORY TIMELINE (x):\n1. [E001] y"):
            result = g.produce_script(GAME, TOPIC, EVIDENCE, IDS, PASSAGES, HOOK_A, "brief", {}, 1, [], "", [scene])
        self.assertIn("fullstory", fake.calls)
        self.assertLess(fake.calls.index("fullstory"), fake.calls.index("scriptgen"))
        self.assertIn(scene["fact"], fake.full_prompts[0])                               # the choice is a key point
        self.assertIn("THE FULL STORY, ALREADY WRITTEN", fake.script_prompts[0])
        self.assertIn("FULL TELLING: Mirela must choose", fake.script_prompts[0])
        self.assertIn(scene["fact"], fake.script_prompts[0])                             # and must be told in the script
        self.assertIn("missing_payoff", result)

    def test_a_payoff_that_skips_how_a_choice_plays_out_costs_points(self):
        clear = [{"status": "CLEAR", "section": "X", "reason": ""}] * 6
        scene = {"id": "E003", "fact": "If the granary burns, the town has no seed."}
        ev = {"clarity": 4, "opening": 4, "progression": 4, "delivery": 4, "repeats": False, "generic_ending": False, "why": ""}
        base = g.score_script(SCRIPT_A, GAME, clear, {"status": "GOOD", "verdict": "PASS"}, {"verdict": "DELIVERS"}, [], ev)
        short = g.score_script(SCRIPT_A, GAME, clear, {"status": "GOOD", "verdict": "PASS"}, {"verdict": "DELIVERS"}, [], ev,
                               missing_payoff=[scene])
        self.assertEqual(base["total"] - short["total"], 5)
        self.assertTrue(short["eligible"])                                               # a craft point, not a block

    def test_leaving_out_what_you_do_its_cost_or_the_reveal_costs_much_more(self):
        clear = [{"status": "CLEAR", "section": "X", "reason": ""}] * 6
        ev = {"clarity": 4, "opening": 4, "progression": 4, "delivery": 4, "repeats": False, "generic_ending": False, "why": ""}
        args = (SCRIPT_A, GAME, clear, {"status": "GOOD", "verdict": "PASS"}, {"verdict": "DELIVERS"}, [], ev)
        base = g.score_script(*args)
        for key in ({"id": "E003", "fact": "You burn the granary.", "kind": "choice"},
                    {"id": "E003", "fact": "Her sister may die.", "beat": "risk"},
                    {"id": "E003", "fact": "The granary was empty.", "beat": "twist"}):
            self.assertEqual(base["total"] - g.score_script(*args, missing_payoff=[key])["total"], 12, key)

    def test_a_hook_sent_without_its_citation_still_brings_its_passage_along(self):
        menu = {"hooks": [{"text": HOOK_B, "spoken": g.remove_citation(HOOK_B), "total": 80, "recommended": True}]}
        (self.checks / f"{g.slugify(GAME + '_' + TOPIC)}_hook_menu.json").write_text(json.dumps(menu), encoding="utf-8")
        others = "\n\n".join(g.compact_passage(p) for p in g.split_evidence_passages(EVIDENCE) if p["id"] != "E002")
        fake = ScriptFake({HOOK_B: [SCRIPT_B_GOOD]})
        with mock.patch.dict(os.environ, {"CHOSEN_HOOK": g.remove_citation(HOOK_B), "SCRIPT_VERSIONS": "1"}), \
                mock.patch.object(g, "select_relevant_evidence", lambda e, t, gm="": (others, IDS - {"E002"}, None)):
            self.run_main(fake)
        self.assertIn("audit", fake.calls)
        self.assertEqual(len(self.outputs("REVIEW_REQUIRED")), 1)

    def test_a_sound_story_with_only_a_style_note_is_not_rewritten_again_and_retries_write_one_draft(self):
        long_payoff = SCRIPT_A.replace("PAYOFF: Burning it would stop the plague, but leave the town no seed for spring planting.",
                                       "PAYOFF: Burning it would stop the plague. But the town would have no seed. And spring would come. Then hunger.")
        fake = ScriptFake({HOOK_A: [long_payoff]})
        with mock.patch.object(g, "ask_ollama", fake), mock.patch.object(g, "DRAFTS_PER_ATTEMPT", 2), \
                mock.patch.object(g, "FULL_STORY_FIRST", False):
            result = g.produce_script(GAME, TOPIC, EVIDENCE, IDS, PASSAGES, HOOK_A, "brief", {}, 4, [], "STORY TIMELINE (x):\n1. [E001] y")
        self.assertEqual(result["attempts_used"], 1)                       # a 4-sentence payoff is scored, not rewritten
        self.assertEqual(fake.calls.count("scriptgen"), 2)                # first attempt: two drafts
        self.assertTrue(result["problems"])                               # the style note is still reported

    def test_a_sound_but_flat_story_gets_one_more_attempt(self):
        flat_then_good = [SCRIPT_A.replace("Her own sister", "Also, her own sister"), SCRIPT_A]
        fake = ScriptFake({HOOK_A: flat_then_good}, eval_scores={HOOK_A: (4, 4, 4, 4)})
        scores = iter([(4, 2, 2, 4), (4, 4, 4, 4)])

        def evaluate(game, draft):
            c, o, p, d = next(scores, (4, 4, 4, 4))
            return {"clarity": c, "opening": o, "progression": p, "delivery": d, "repeats": False, "generic_ending": False, "why": "flat"}
        with mock.patch.object(g, "ask_ollama", fake), mock.patch.object(g, "evaluate_script", evaluate), \
                mock.patch.object(g, "FULL_STORY_FIRST", False), mock.patch.object(g, "DRAFTS_PER_ATTEMPT", 1):
            result = g.produce_script(GAME, TOPIC, EVIDENCE, IDS, PASSAGES, HOOK_A, "brief", {}, 4, [], "STORY TIMELINE (x):\n1. [E001] y")
        self.assertEqual(result["attempts_used"], 2)
        self.assertIn("the story judge says: flat", fake.script_prompts[1])

    def test_an_edited_hook_is_cited_and_checked_like_any_other(self):
        edited = "Mirela's sister lies sick inside the granary and cannot be moved."
        script = SCRIPT_B_GOOD.replace("HOOK: " + HOOK_B, "HOOK: " + edited + " [E002]")
        fake = ScriptFake({edited + " [E002]": [script]})
        with mock.patch.dict(os.environ, {"CHOSEN_HOOK": edited}):
            self.run_main(fake)
        self.assertNotIn("hookgen", fake.calls)
        self.assertIn("audit", fake.calls)
        self.assertIn(edited + " [E002]", g.VERIFIED_HOOKS)


class LeanModeTests(unittest.TestCase):
    """With a free hosted writer, the script is written in one short pass: few requests, small prompts."""
    def test_a_hosted_writer_writes_in_one_pass_without_the_full_story_or_editor_steps(self):
        fake = ScriptFake({HOOK_A: [SCRIPT_A]}, coherence="FAILS")   # keeps retrying, so the attempt cap shows
        with mock.patch.object(g, "ask_ollama", fake), mock.patch.object(g, "hosted_writer_active", lambda: True):
            result = g.produce_script(GAME, TOPIC, EVIDENCE, IDS, PASSAGES, HOOK_A, "brief", {}, 5, [], "STORY TIMELINE (x):\n1. [E001] y")
        self.assertNotIn("fullstory", fake.calls)
        self.assertNotIn("editor", fake.calls)
        self.assertEqual(result["attempts_used"], 2)                   # capped for the free tiers
        self.assertEqual(fake.calls.count("scriptgen"), 2)             # one draft per attempt
        prompt = fake.script_prompts[0]
        self.assertIn("PLAN (verified facts in telling order", prompt)
        self.assertIn("Invent nothing, not even a relationship or a role", prompt)
        self.assertLess(len(prompt) / 3.5, g.GROQ_MAX_PROMPT_TOKENS)  # small enough for every free writer

    def test_without_a_hosted_writer_the_local_model_keeps_the_full_process(self):
        fake = ScriptFake({HOOK_A: [SCRIPT_A]})
        with mock.patch.object(g, "ask_ollama", fake), mock.patch.object(g, "hosted_writer_active", lambda: False), \
                mock.patch.object(g, "DRAFTS_PER_ATTEMPT", 1):
            g.produce_script(GAME, TOPIC, EVIDENCE, IDS, PASSAGES, HOOK_A, "brief", {}, 1, [], "STORY TIMELINE (x):\n1. [E001] y")
        self.assertIn("fullstory", fake.calls)
        self.assertIn("editor", fake.calls)

    def test_groq_is_not_sent_a_prompt_its_free_tier_would_refuse(self):
        sent = []
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "", "GROQ_API_KEY": "k"}), \
                mock.patch.object(g.requests, "post", lambda *a, **k: sent.append(a) or None), \
                mock.patch.object(g, "ask_ollama", lambda p, **k: "local"), mock.patch.object(g.time, "sleep", lambda s: None):
            self.assertEqual(g.ask_writer("x" * 40000), "local")
        self.assertEqual(sent, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
