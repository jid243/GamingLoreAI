"""Tests for the hook stage of generate_grounded_script_google.py. The language model is mocked (no Ollama needed):
these tests prove the validation, gating, scoring and selection LOGIC, not how good a real model's hooks are."""
import json
import os
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
EVIDENCE = """[E001]
Source: wiki
Excerpt: A village healer named Mirela must decide whether to burn the plague-ridden granary, which also holds the town's last seed grain.

[E002]
Source: wiki
Excerpt: The Ember Court sentenced Warden Orsk to guard the Hollow Gate for life.
"""
IDS = {"E001", "E002"}
SUPPORT = {  # hook sentence -> verbatim quote the fake auditor returns (None = unsupported)
    "A healer must burn the town's last grain to stop the plague, or let everyone starve.":
        "must decide whether to burn the plague-ridden granary, which also holds the town's last seed grain",
    "A village healer must decide whether to burn a plague-ridden granary holding the last seed grain.":
        "must decide whether to burn the plague-ridden granary, which also holds the town's last seed grain",
    "Mirela, Orsk and the Ember Court clash over the Hollow Gate.": "sentenced Warden Orsk to guard the Hollow Gate for life",
    "A healer doomed the whole village by burning the granary.": None,
    "A dark secret hangs in the balance for the healer and the granary.": "burn the plague-ridden granary",
}


class FakeModel:
    def __init__(self, generations, scores, support=SUPPORT, facts=""):
        self.generations, self.scores, self.support, self.facts = list(generations), scores, support, facts
        self.calls = {"gen": 0, "audit": 0, "eval": 0}
        self.hook_prompts = []

    def __call__(self, prompt, temperature=0.2, model=None, options=None):
        if "Does any passage below state this claim?" in prompt:
            return getattr(self, "locate_reply", "NONE")
        if "will tell the story of" in prompt and "CANDIDATES:" in prompt:
            return "IDS: NONE"
        if "fair fact-checker" in prompt:
            self.calls["audit"] += 1
            text = prompt.split("LINE (HOOK):", 1)[1].strip()
            quote = self.support.get(text)
            return f"CLAIM 1: {text} || QUOTE: {quote or 'NONE'}"
        if "pull out" in prompt and "FACT 1:" in prompt:
            return self.facts
        if "write the first spoken sentence" in prompt:
            self.calls["gen"] += 1
            self.hook_prompts.append(prompt)
            return self.generations.pop(0) if len(self.generations) > 1 else self.generations[0]
        if "distracted viewer" in prompt:
            self.calls["eval"] += 1
            if self.scores is None:
                return "I think they are all pretty good!"
            lines = []
            for line in prompt.split("HOOKS:\n", 1)[1].splitlines():
                number, text = line[len("HOOK "):].split(": ", 1)
                c, s, cu, d, unexplained, any_game = self.scores.get(text, (1, 1, 1, 1, "NONE", "NO"))
                lines.append(f"HOOK {number}: CLARITY={c} SPECIFICITY={s} CURIOSITY={cu} EMOTION={cu} DELIVERY={d} "
                             f"UNEXPLAINED={unexplained} ANY_GAME={any_game} WHY=test reason")
            return "\n".join(lines)
        raise AssertionError("unexpected prompt: " + prompt[:80])


def run(generations, scores, allow_weak=False, log=None, facts=""):
    fake = FakeModel(generations, scores, facts=facts)
    with mock.patch.object(g, "ask_ollama", fake):
        try:
            return g.generate_and_select_hook(GAME, TOPIC, EVIDENCE, IDS, "BRIEF", log_path=log, allow_weak=allow_weak), fake
        except g.HookSelectionError as error:
            return error, fake


GOOD = "A healer must burn the town's last grain to stop the plague, or let everyone starve."
GOOD2 = "A village healer must decide whether to burn a plague-ridden granary holding the last seed grain."


class HookTests(unittest.TestCase):
    def test_1_clear_grounded_hook_is_selected(self):
        log = Path(_HOME) / "t1.json"
        gen = f"HOOK 1 (MORAL_DILEMMA): {GOOD} [E001]\nHOOK 2 (IMMEDIATE_DANGER): NONE - no danger stated"
        result, fake = run([gen], {GOOD: (5, 4, 4, 4, "NONE", "NO")}, log=log)
        self.assertEqual(result, f"{GOOD} [E001]")
        data = json.loads(log.read_text())
        self.assertEqual(data["selected"]["total"], 87.0)         # 20 + 12 + 16 + 16 (emotion) + 8 + 15
        self.assertEqual(data["candidates"][0]["score"]["penalties"], [])
        self.assertIn("approaches_without_support", data["attempts"][0])
        self.assertTrue(data["attempts"][0]["approaches_without_support"])

    def test_2_unexplained_proper_nouns_are_penalised_but_only_strict_mode_blocks_them(self):
        hook = "Mirela, Orsk and the Ember Court clash over the Hollow Gate."
        gen = f"HOOK 1 (MYSTERY): {hook} [E002]"
        result, fake = run([gen], {hook: (2, 3, 3, 3, "Mirela; Orsk; Ember Court", "NO")})
        self.assertEqual(result, f"{hook} [E002]")             # verified, so it is never rejected for style
        with mock.patch.object(g, "HOOK_STRICT", True):
            result, fake = run([gen], {hook: (2, 3, 3, 3, "Mirela; Orsk; Ember Court", "NO")})
        self.assertIsInstance(result, g.HookSelectionError)
        passage_map = {p["id"]: p["text"] for p in g.split_evidence_passages(EVIDENCE)}
        cand = {"text": hook + " [E002]", "grounded": True}
        score = g.score_hook(cand, {"clarity": 2, "specificity": 3, "curiosity": 3, "delivery": 3,
                                     "unexplained": ["Mirela", "Orsk", "Ember Court"], "any_game": False, "why": ""},
                             GAME, passage_map)
        self.assertTrue(score["eligible"])
        self.assertIn("clarity 2/5", score["notes"])
        # names are fine over footage: three is not penalised; four or more in one opening line is
        self.assertFalse(any("unexplained proper noun" in p["reason"] for p in score["penalties"]))
        crowded = {"text": "Mirela, Orsk, Tamsin and the Ember Court clash over the Hollow Gate. [E002]", "grounded": True}
        crowded_score = g.score_hook(crowded, {"clarity": 3, "specificity": 3, "curiosity": 3, "delivery": 3, "unexplained": [],
                                               "any_game": False, "why": ""}, GAME, passage_map)
        self.assertTrue(any("unexplained proper noun" in p["reason"] for p in crowded_score["penalties"]))

    def test_3_unsupported_dramatic_claim_is_rejected_by_the_grounding_gate(self):
        dramatic = "A healer doomed the whole village by burning the granary."
        gen = f"HOOK 1 (IMMEDIATE_DANGER): {dramatic} [E001]"
        # evaluator would LOVE it; the hard gate must still reject it
        result, fake = run([gen], {dramatic: (5, 5, 5, 5, "NONE", "NO")})
        self.assertIsInstance(result, g.HookSelectionError)
        self.assertIn("could be verified against the evidence", str(result))

    def test_4_generic_hooks_are_rejected_or_penalised(self):
        parsed, reasons, _ = g.parse_hook_candidates(
            "HOOK 1: Did you know a healer once burned a granary full of grain? [E001]\n"
            "HOOK 2: This is the story of a healer and a granary full of grain. [E001]", IDS, GAME)
        self.assertEqual(parsed, [])
        self.assertEqual(len(reasons), 2)
        vague = "A dark secret hangs in the balance for the healer and the granary."
        passage_map = {p["id"]: p["text"] for p in g.split_evidence_passages(EVIDENCE)}
        score = g.score_hook({"text": vague + " [E001]", "grounded": True},
                             {"clarity": 3, "specificity": 3, "curiosity": 3, "delivery": 3, "unexplained": [],
                              "any_game": True, "why": ""}, GAME, passage_map)
        reasons = " | ".join(p["reason"] for p in score["penalties"])
        self.assertIn("vague", reasons)
        self.assertIn("any game", reasons)
        self.assertLess(score["total"], 60)                      # penalised hard, so it ranks low...
        self.assertTrue(score["eligible"])                        # ...but a verified hook is never rejected for style

    def test_5_strongest_eligible_is_chosen_not_the_first(self):
        gen = f"HOOK 1 (MORAL_DILEMMA): {GOOD} [E001]\nHOOK 2 (IMMEDIATE_DANGER): {GOOD2} [E001]"
        scores = {GOOD: (4, 3, 3, 3, "NONE", "NO"), GOOD2: (5, 4, 4, 4, "NONE", "NO")}
        result, _ = run([gen], scores)
        self.assertEqual(result, f"{GOOD2} [E001]")
        scores = {GOOD: (5, 4, 4, 4, "NONE", "NO"), GOOD2: (4, 3, 3, 3, "NONE", "NO")}
        result, _ = run([gen], scores)
        self.assertEqual(result, f"{GOOD} [E001]")

    def test_6_insufficient_evidence_warns_and_never_falls_back(self):
        log = Path(_HOME) / "t6.json"
        gen = "\n".join(f"HOOK {i} ({n}): NONE - evidence does not support it"
                        for i, (n, _) in enumerate(g.HOOK_APPROACHES, 1))
        result, fake = run([gen], None, log=log)
        self.assertIsInstance(result, g.HookSelectionError)
        self.assertIn("HOOK WARNING", str(result))
        self.assertEqual(fake.calls["gen"], g.HOOK_GEN_ATTEMPTS)
        self.assertEqual(fake.calls["eval"], 0)
        self.assertIsNone(json.loads(log.read_text())["selected"])
        # grounded but weakly rated: used (only made-up claims reject); strict mode restores the stop
        weak_gen = f"HOOK 1 (MORAL_DILEMMA): {GOOD} [E001]"
        weak = {GOOD: (2, 2, 2, 2, "NONE", "NO")}
        result, _ = run([weak_gen], weak)
        self.assertEqual(result, f"{GOOD} [E001]")
        with mock.patch.object(g, "HOOK_STRICT", True):
            result, _ = run([weak_gen], weak)
        self.assertIsInstance(result, g.HookSelectionError)

    def test_7_malformed_model_output_is_handled_safely(self):
        result, fake = run(["Sure! Here are some great hooks for you:\n- the healer burns it all"], None)
        self.assertIsInstance(result, g.HookSelectionError)
        # well-formed hooks but the evaluator replies with prose: the verified hook is still usable, and the log says
        # plainly that it was not scored
        gen = f"HOOK 1 (MORAL_DILEMMA): {GOOD} [E001]"
        log = Path(_HOME) / "t7.json"
        result, fake = run([gen], None, log=log)
        self.assertEqual(result, f"{GOOD} [E001]")
        data = json.loads(log.read_text())
        self.assertTrue(data["attempts"][0]["evaluation_failed"])
        # bad citation / two sentences / quotes are reported, not crashed on
        parsed, reasons, _ = g.parse_hook_candidates(
            'HOOK 1: No citation here at all.\nHOOK 2: Two sentences. Here. [E001]\nHOOK 3: She said "stop" now. [E001]\nHOOK 4: Bad id sentence. [E099]',
            IDS, GAME)
        self.assertEqual(parsed, [])
        self.assertEqual(len(reasons), 4)


BOTCHLING = """[E005]
Source: wiki
Excerpt: Born of dead, unwanted babies discarded without a proper burial, their appearance is that of a partially-decayed fetus. These hideous creatures feed on the blood of pregnant women.

[E023]
Source: wiki
Excerpt: The baron will lead Geralt out to where he buried the fetus. However, the grave is empty and the botchling appears. You have a timed reaction here to make your choice: lift the curse by making it a lubberkin or kill the botchling.
"""


class FactStageTests(unittest.TestCase):
    def _facts(self, raw):
        with mock.patch.object(g, "ask_ollama", lambda *a, **k: raw):
            return g.extract_hook_facts("The Witcher 3", "Botchling", BOTCHLING, {"E005", "E023"}, "BRIEF")

    def test_only_facts_with_real_quotes_survive(self):
        facts, dropped = self._facts(
            "FACT 1: A creature is born from a baby buried without proper rites. || QUOTE: Born of dead, unwanted babies discarded without a proper burial || ID: E005\n"
            "FACT 2: A baron begs you to spare a spirit. || QUOTE: the baron begs the witcher to spare it || ID: E023\n"
            "FACT 3: You get seconds to choose between killing it or breaking its curse. || QUOTE: a timed reaction here to make your choice || ID: E023\n"
            "FACT 4: Something. || QUOTE: Born of dead, unwanted babies discarded || ID: E999")
        # a real quote cited to a wrong/unknown id is a citation error: it is re-cited, not lost
        self.assertEqual([f["id"] for f in facts], ["E005", "E023", "E005"])
        self.assertTrue(any("not found" in d for d in dropped))
        self.assertTrue(any(d.startswith("citation corrected E999 -> E005") for d in dropped))

    def test_hook_prompt_uses_verified_facts_and_no_leaky_examples(self):
        facts = [{"id": "E023", "fact": "You must choose in seconds to kill it or lift its curse.", "quote": "timed reaction"}]
        prompt = g.build_hook_prompt("The Witcher 3", "Botchling", BOTCHLING, {"E023"}, "BRIEF", facts=facts)
        self.assertIn("VERIFIED STORY CHAIN", prompt)
        self.assertNotIn("swamp village", prompt)
        self.assertNotIn("EVIDENCE:\n[E005]", prompt)       # full evidence is not pasted when facts exist

    def test_falls_back_to_full_evidence_when_facts_fail(self):
        gen = f"HOOK 1 (MORAL_DILEMMA): {GOOD} [E001]"
        result, fake = run([gen], {GOOD: (5, 4, 4, 4, "NONE", "NO")}, facts="nothing useful")
        self.assertEqual(result, f"{GOOD} [E001]")
        self.assertIn("EVIDENCE:", fake.hook_prompts[0])


class GameTitleTests(unittest.TestCase):
    PM = {p["id"]: p["text"] for p in g.split_evidence_passages(EVIDENCE)}

    def test_naming_the_game_is_not_penalised_or_rejected(self):
        for game, hook in [("Ashfall Keep", "In Ashfall Keep, a healer must burn the town's last grain or let everyone starve."),
                           ("The Witcher 3: Wild Hunt", "In The Witcher 3, a healer must burn the town's last grain or let everyone starve."),
                           ("The Witcher 3: Wild Hunt", "In Witcher 3, a healer must burn the town's last grain or let everyone starve.")]:
            parsed, reasons, _ = g.parse_hook_candidates(f"HOOK 1: {hook} [E001]", IDS, game)
            self.assertEqual(len(parsed), 1, reasons)
            self.assertEqual(g.hook_proper_nouns(hook, game), [])
            score = g.score_hook({"text": hook + " [E001]", "grounded": True},
                                 {"clarity": 5, "specificity": 4, "curiosity": 4, "delivery": 4,
                                  "unexplained": ["Witcher 3", "Ashfall Keep"], "any_game": False, "why": ""}, game, self.PM)
            self.assertFalse([p for p in score["penalties"] if "redundant" in p["reason"] or "unexplained" in p["reason"]],
                             score["penalties"])


class BackendTests(unittest.TestCase):
    def _post(self, model):
        captured = {}

        def fake_post(url, json=None, timeout=None):
            captured.update(json)
            return mock.Mock(status_code=200, text="", raise_for_status=lambda: None,
                             json=lambda: {"message": {"content": " ok "}})
        with mock.patch.object(g.requests, "post", fake_post):
            self.assertEqual(g.ask_ollama("hi", model=model), "ok")
        return captured

    def test_think_flag_only_sent_to_models_that_support_it(self):
        self.assertNotIn("think", self._post("gemma2:27b"))
        self.assertIs(self._post("qwen3:14b")["think"], False)

    def test_gemma4_gets_thinking_switched_off(self):
        self.assertIn("think", self._post("gemma4:26b-a4b-it-qat"))   # left on, it thinks for minutes before every answer

    def test_a_passing_ollama_error_is_retried_and_a_lasting_one_explained(self):
        replies = [mock.Mock(status_code=500, text='{"error":"model is loading"}', json=lambda: {"error": "model is loading"}),
                   mock.Mock(status_code=200, text="", json=lambda: {"message": {"content": "IDS: NONE"}})]
        with mock.patch.object(g.requests, "post", lambda *a, **k: replies.pop(0)), \
                mock.patch.object(g.time, "sleep", lambda s: None):
            self.assertEqual(g.ask_ollama("hi"), "IDS: NONE")
        broken = mock.Mock(status_code=500, text="", json=lambda: {"error": "out of memory"})
        with mock.patch.object(g.requests, "post", lambda *a, **k: broken), mock.patch.object(g.time, "sleep", lambda s: None):
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                g.ask_ollama("hi")

    def test_default_model_is_gemma4(self):
        if "OLLAMA_MODEL" not in os.environ:
            self.assertEqual(g.MODEL, "gemma4:26b-a4b-it-qat")
            self.assertEqual(g.NUM_CTX, 16384)

    def test_weights_sum_to_100(self):
        self.assertEqual(sum(g.HOOK_WEIGHTS.values()), 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
