"""Tests for evidence selection, the five-way claim audit (SUPPORTED / SUPPORTED_WRONG_CITATION / NEEDS_REVIEW /
UNSUPPORTED / CONTRADICTED), citation repair, meaning-preserving line repair, the verified story chain and the gates.
The language model is MOCKED with synthetic fixtures (an invented game), so these prove the pipeline's logic, not the
quality of real gemma2 output. Nothing here is specific to any real game."""
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
import test_scripts as ts  # noqa: E402  (shares the synthetic Ashfall Keep fixtures and the pipeline harness)

GAME, TOPIC, PASSAGES, IDS = ts.GAME, ts.TOPIC, ts.PASSAGES, ts.IDS


def excerpt(evidence_id):
    return g.excerpt_of(PASSAGES[evidence_id])


class AuditFake:
    """audit: {line text: [(claim, verdict, quote), ...]}; locate: {claim: (id, verdict, quote) or None};
    repairs: [reply, ...] returned in order for repair prompts."""

    def __init__(self, audit=None, locate=None, repairs=()):
        self.audit, self.locate, self.repairs = audit or {}, locate or {}, list(repairs)
        self.calls, self.repair_prompts = [], []

    def __call__(self, prompt, temperature=0.2, model=None, options=None):
        if "Does any passage below state this claim?" in prompt:
            self.calls.append("locate")
            claim = prompt.split("CLAIM: ", 1)[1].splitlines()[0].strip()
            hit = self.locate.get(claim)
            return "NONE" if not hit else f"ID: {hit[0]} || VERDICT: {hit[1]} || QUOTE: {hit[2]}"
        if "Does the QUOTE state the CLAIM?" in prompt:
            self.calls.append("says?")
            claim = prompt.split("CLAIM: ", 1)[1].splitlines()[0].strip()
            return "YES" if claim in getattr(self, "says", ()) else "NO"
        if "Rewrite ONE flagged line" in prompt:
            self.calls.append("repair")
            self.repair_prompts.append(prompt)
            return self.repairs.pop(0) if self.repairs else "NOTHING"
        if "fair fact-checker" in prompt:
            self.calls.append("audit")
            text = prompt.split("LINE (", 1)[1].split("): ", 1)[1].strip()
            rows = self.audit.get(text)
            if rows is None:   # default: the first cited excerpt's opening words support the whole line
                first = re.search(r"\[E\d+\] (.+)", prompt).group(1)
                return f"CLAIM 1: {text} || VERDICT: SUPPORTED || QUOTE: {' '.join(first.split()[:6])}"
            return "\n".join(f"CLAIM {n}: {r[0]} || TYPE: {r[3] if len(r) > 3 else 'FACT'} || VERDICT: {r[1]} || QUOTE: {r[2]}"
                             for n, r in enumerate(rows, 1))
        raise AssertionError("unexpected prompt: " + prompt[:80])


def audit(line, fake, section="NARRATION"):
    with mock.patch.object(g, "ask_ollama", fake):
        return g.audit_line(section, line, PASSAGES, GAME)


class ClaimStatusTests(unittest.TestCase):
    def setUp(self):
        g._LOCATE_CACHE.clear()

    def test_1_a_faithful_paraphrase_passes_without_reusing_the_source_wording(self):
        line = "The town's healer has to choose whether to set the infected storehouse alight. [E001]"
        fake = AuditFake({g.remove_citation(line): [
            ("The healer must decide whether to burn the granary", "SUPPORTED", "must decide whether to burn the plague-ridden granary")]})
        result = audit(line, fake)
        self.assertEqual(result["status"], "CLEAR")
        self.assertEqual(result["rows"][0]["status"], "SUPPORTED")
        self.assertNotIn("storehouse", excerpt("E001"))           # the line's own words are not in the source

    def test_2_a_correct_claim_with_the_wrong_citation_is_recited_not_deleted(self):
        line = "Her sister lies sick inside and cannot be moved. [E004]"
        fake = AuditFake({g.remove_citation(line): [
            ("The sister lies sick inside and cannot be moved", "SUPPORTED", "sister lies sick inside the granary and cannot be moved")]})
        result = audit(line, fake)
        self.assertEqual(result["status"], "RECITE")
        self.assertEqual(result["rows"][0]["status"], "SUPPORTED_WRONG_CITATION")
        self.assertEqual(result["recite_ids"], ["E002"])
        draft = ts.SCRIPT_A.replace("Her own sister lies sick inside the granary and cannot be moved. [E002]", line)
        with mock.patch.object(g, "ask_ollama", fake):
            fixed, audits = g.audit_and_repair(GAME, TOPIC, draft, PASSAGES, IDS, {})
        self.assertIn("Her sister lies sick inside and cannot be moved. [E002]", fixed)
        self.assertNotIn("repair", fake.calls)                    # fixed in code; nothing was rewritten
        self.assertTrue(all(a["status"] == "CLEAR" for a in audits))

    def test_2b_when_the_auditor_says_not_stated_the_other_passages_are_searched_first(self):
        line = "The elders gave her until dawn. [E002]"
        claim = "The elders gave the healer until dawn"
        fake = AuditFake({g.remove_citation(line): [(claim, "NOT_STATED", "NONE")]},
                         locate={claim: ("E004", "SUPPORTED", "The town elders have given Mirela until dawn")})
        result = audit(line, fake)
        self.assertEqual(result["status"], "RECITE")
        self.assertEqual(result["recite_ids"], ["E004"])

    def test_3_a_causal_chain_supported_across_several_passages_is_clear_with_each_citation(self):
        line = ("Because the elders hid infected grain, the healer must now decide whether to burn the granary, "
                "which would stop the plague. [E005] [E001] [E003]")
        fake = AuditFake({g.remove_citation(line): [
            ("The plague began because the elders stored infected grain", "SUPPORTED", "The plague began because the town elders stored infected grain"),
            ("The healer must decide whether to burn the granary", "SUPPORTED", "must decide whether to burn the plague-ridden granary"),
            ("Burning it would stop the plague spreading", "SUPPORTED", "If the granary burns, the plague stops spreading")]})
        self.assertGreaterEqual(g.MAX_CITATIONS_PER_LINE, 3)
        self.assertEqual(audit(line, fake)["status"], "CLEAR")
        problems = g.check_script_format(ts.SCRIPT_A.replace("Burning it would stop the plague, but leave the town no seed for spring planting. [E003]",
                                                             "Burning it stops the plague, but the elders caused it and the town loses its seed. [E003] [E005] [E001]"),
                                         IDS, ts.HOOK_A, GAME)
        self.assertFalse(any("citations" in p for p in problems), problems)

    def test_4_an_unsupported_causal_link_is_blocked_and_never_invented(self):
        line = "The sister's illness is what started the plague. [E002] [E005]"
        claim = "The sister's illness started the plague"
        fake = AuditFake({g.remove_citation(line): [(claim, "NOT_STATED", "NONE")]}, locate={claim: None})
        result = audit(line, fake)
        self.assertEqual(result["status"], "FLAGGED")
        self.assertEqual(result["rows"][0]["status"], "UNSUPPORTED")
        # the brief's root cause gets the same check; an unsupported one is replaced, not passed on
        brief = "PLAIN_SETUP: x\nROOT_CAUSE: The sister's illness is what started the plague. [E002]\nMISSING: NONE"
        fake2 = AuditFake({"The sister's illness is what started the plague.": [(claim, "NOT_STATED", "NONE")]}, locate={claim: None})
        with mock.patch.object(g, "ask_ollama", fake2):
            new_brief, note = g.verify_brief_root_cause(brief, PASSAGES, GAME)
        self.assertIn("ROOT_CAUSE: NOT ESTABLISHED", new_brief)
        self.assertTrue(note.startswith("ROOT_CAUSE rejected"))
        chain = g.story_chain_text([{"id": "E005", "kind": "cause", "fact": "The plague began because the elders hid infected grain."}],
                                   ["the sister's illness started the plague"])
        self.assertIn("NOT ESTABLISHED by the evidence (never state these as fact): the sister's illness", chain)

    def test_contradicted_claims_are_blocked(self):
        line = "The sister can easily be carried out of the granary. [E002]"
        fake = AuditFake({g.remove_citation(line): [("The sister can be carried out", "CONTRADICTED", "sister lies sick inside the granary and cannot be moved")]})
        result = audit(line, fake)
        self.assertEqual((result["status"], result["rows"][0]["status"]), ("FLAGGED", "CONTRADICTED"))

    def test_a_real_quote_unrelated_to_its_claim_is_not_accepted_as_support(self):
        line = "The granary is cursed by an ancient spirit. [E001]"
        claim = "An ancient spirit cursed the granary"
        fake = AuditFake({g.remove_citation(line): [(claim, "SUPPORTED", "holds the town's last seed grain")]}, locate={claim: None})
        result = audit(line, fake)
        self.assertEqual(result["status"], "FLAGGED")

    def test_an_uncertain_claim_is_kept_for_review_and_not_rewritten(self):
        line = "The elders decided to blame the healer for the plague. [E005]"
        fake = AuditFake({g.remove_citation(line): [("The elders blamed the healer", "PARTLY", "without telling the healer")]},
                         locate={"The elders blamed the healer": None})
        result = audit(line, fake)
        self.assertEqual(result["status"], "REVIEW")
        self.assertEqual(result["rows"][0]["status"], "NEEDS_REVIEW")
        draft = ts.SCRIPT_A.replace("Her own sister lies sick inside the granary and cannot be moved. [E002]", line)
        with mock.patch.object(g, "ask_ollama", fake):
            kept, audits = g.audit_and_repair(GAME, TOPIC, draft, PASSAGES, IDS, {})
        self.assertIn(line, kept)                                 # not blanded by an automatic rewrite
        self.assertNotIn("repair", fake.calls)
        score = g.score_script(kept, GAME, audits, {"status": "GOOD", "verdict": "PASS"}, None, [],
                               {"clarity": 4, "opening": 4, "progression": 4, "delivery": 4, "generic_ending": False, "why": ""})
        self.assertTrue(score["eligible"])                        # one review line is allowed...
        self.assertTrue(any("NEEDS HUMAN REVIEW" in w for w in score["warnings"]))   # ...but always reported
        self.assertLess(score["points"]["grounding"], g.SCRIPT_WEIGHTS["grounding"])
        self.assertTrue(any("needs review" in m for m in g.missing_evidence_note(audits, [], None)))

    def test_uncertain_lines_are_reported_and_cost_points_but_never_reject(self):
        review = {"index": 4, "section": "PAYOFF", "content": "x [E003]", "status": "REVIEW", "reason": "uncertain", "claims": []}
        score = g.score_script(ts.SCRIPT_A, GAME, [review], {"status": "GOOD", "verdict": "PASS"}, None, [], None)
        self.assertTrue(score["eligible"])
        self.assertTrue(any("NEEDS HUMAN REVIEW" in w for w in score["warnings"]))
        self.assertFalse(any("need human review" in p["reason"] for p in score["penalties"]))   # counted once, in grounding
        flagged = dict(review, status="FLAGGED")
        self.assertFalse(g.score_script(ts.SCRIPT_A, GAME, [flagged], {"status": "GOOD", "verdict": "PASS"}, None, [], None)["eligible"])


class RepairTests(unittest.TestCase):
    def setUp(self):
        g._LOCATE_CACHE.clear()

    def _flagged(self):
        content = "Her own sister lies sick inside the granary, cursed by a witch, and cannot be moved. [E002]"
        rows = [{"claim": "The sister lies sick inside the granary and cannot be moved", "status": "SUPPORTED", "quote": "q", "note": "", "found_in": "E002", "verdict": "SUPPORTED"},
                {"claim": "A witch cursed the sister", "status": "UNSUPPORTED", "quote": "NONE", "note": "", "found_in": None, "verdict": "NOT_STATED"}]
        return {"section": "NARRATION", "content": content, "rows": rows, "unsupported": [("A witch cursed the sister", "NONE")]}

    def test_6_a_flagged_line_keeps_its_verified_specifics_when_repaired(self):
        fake = AuditFake(repairs=["NARRATION: Something bad happened to her in the town. [E002]",
                                  "NARRATION: Her own sister lies sick inside the granary and cannot be moved. [E002]"])
        with mock.patch.object(g, "ask_ollama", fake):
            fixed = g.repair_line(GAME, TOPIC, self._flagged(), PASSAGES, IDS, 24)
        self.assertEqual(fixed, "Her own sister lies sick inside the granary and cannot be moved. [E002]")
        self.assertIn("KEEP (verified): The sister lies sick inside the granary", fake.repair_prompts[0])
        self.assertIn("REMOVE (no available passage states it): A witch cursed the sister", fake.repair_prompts[0])
        self.assertIn("dropped the verified KEEP details", fake.repair_prompts[1])   # the vague rewrite was refused

    def test_7_a_repair_that_repeats_another_line_is_refused(self):
        other = "Burning it would stop the plague, but leave the town no seed for spring planting. [E003]"
        flagged = self._flagged()
        fake = AuditFake(repairs=["NARRATION: Her sister inside means burning would stop the plague but leave no seed for spring planting. [E002] [E003]",
                                  "NARRATION: Her own sister lies sick inside the granary and cannot be moved. [E002]"])
        with mock.patch.object(g, "ask_ollama", fake):
            fixed = g.repair_line(GAME, TOPIC, flagged, PASSAGES, IDS, 24, other_lines=[other])
        self.assertIn("repeats another line", fake.repair_prompts[1])
        self.assertFalse(g._restates(fixed, other, GAME))
        # and the script-level check still rejects narration/payoff repeats
        self.assertTrue(any(fails for *_, fails in g.script_repetition(ts.SCRIPT_B_REPEATS, GAME)))

    def test_11_citation_ids_stay_valid_through_recitation_and_repair(self):
        fake = AuditFake(repairs=["NARRATION: Her own sister lies sick inside the granary and cannot be moved. [E099]",
                                  "NARRATION: Her own sister lies sick inside the granary and cannot be moved. [E002]"])
        with mock.patch.object(g, "ask_ollama", fake):
            fixed = g.repair_line(GAME, TOPIC, self._flagged(), PASSAGES, IDS, 24)
        self.assertTrue(set(g.line_citations(fixed)) <= IDS)
        self.assertIn("valid ids", fake.repair_prompts[1])
        self.assertEqual(g.recite_line("A line. [E004]", ["E002", "E005"]), "A line. [E002] [E005]")


class HookGroundingTests(unittest.TestCase):
    def setUp(self):
        g._LOCATE_CACHE.clear()

    def test_5_and_9_a_hook_combining_several_verified_facts_passes(self):
        hook = "Hidden infected grain started the plague, and now a healer must burn the granary her sick sister lies in. [E005] [E001] [E002]"
        fake = AuditFake({g.remove_citation(hook): [
            ("The plague began because infected grain was hidden", "SUPPORTED", "The plague began because the town elders stored infected grain"),
            ("A healer must decide whether to burn the granary", "SUPPORTED", "must decide whether to burn the plague-ridden granary"),
            ("Her sister lies sick inside", "SUPPORTED", "sister lies sick inside the granary")]})
        candidates = [{"text": hook}]
        with mock.patch.object(g, "ask_ollama", fake):
            g.audit_hook_candidates(candidates, ts.EVIDENCE, GAME)
        self.assertTrue(candidates[0]["grounded"], candidates[0]["audit_reason"])
        facts = [{"id": "E005", "kind": "cause", "fact": "The plague began because the elders stored infected grain."},
                 {"id": "E001", "kind": "choice", "fact": "A healer must decide whether to burn the granary."}]
        prompt = g.build_hook_prompt(GAME, TOPIC, ts.EVIDENCE, IDS, "BRIEF", facts=facts)
        self.assertIn("join up to three of them into one premise", prompt)
        self.assertLess(prompt.index("CAUSE [E005]"), prompt.index("CHOICE [E001]"))   # chain runs cause -> choice

    def test_10_a_dramatic_but_unsupported_hook_is_rejected(self):
        hook = "A healer's cursed bloodline doomed the whole town to a deadly plague. [E001]"
        claims = [("The healer's bloodline is cursed", "NOT_STATED", "NONE"), ("The plague is deadly", "NOT_STATED", "NONE")]
        fake = AuditFake({g.remove_citation(hook): claims}, locate={c: None for c, _, _ in claims})
        candidates = [{"text": hook}]
        with mock.patch.object(g, "ask_ollama", fake):
            g.audit_hook_candidates(candidates, ts.EVIDENCE, GAME)
        self.assertFalse(candidates[0]["grounded"])

    def test_a_mis_cited_hook_is_recited_and_passes(self):
        hook = "A sick sister lies inside a granary that a healer may have to burn. [E004]"
        fake = AuditFake({g.remove_citation(hook): [
            ("A sister lies sick inside the granary", "SUPPORTED", "sister lies sick inside the granary and cannot be moved"),
            ("A healer must decide whether to burn it", "SUPPORTED", "must decide whether to burn the plague-ridden granary")]})
        candidates = [{"text": hook}]
        with mock.patch.object(g, "ask_ollama", fake):
            g.audit_hook_candidates(candidates, ts.EVIDENCE, GAME)
        self.assertTrue(candidates[0]["grounded"])
        self.assertEqual(set(g.line_citations(candidates[0]["text"])), {"E002", "E001"})


class StoryAndGateTests(ts.PipelineHarness):
    def test_8_the_brief_keeps_a_verified_root_cause_and_must_cite_it(self):
        brief = "ROOT_CAUSE: The plague began because the elders stored infected grain without telling the healer. [E005]\nDETAILS_TO_OMIT: the elders hiding the grain."
        fake = AuditFake()
        with mock.patch.object(g, "ask_ollama", fake):
            kept, note = g.verify_brief_root_cause(brief, PASSAGES, GAME)
        self.assertEqual(kept, brief)
        self.assertIsNone(note)
        protected, removed = g.protect_root_cause(kept, ["The plague began because the elders stored infected grain"], GAME)
        self.assertTrue(removed)
        self.assertNotIn("elders hiding the grain", protected.split("DETAILS_TO_OMIT:")[1])
        captured = {}
        with mock.patch.object(g, "ask_ollama", lambda prompt, **k: captured.setdefault("p", prompt) or "x"):
            g.develop_story_angle(GAME, TOPIC, ts.EVIDENCE, IDS)
        self.assertIn("ending with the id(s) of the passage(s) that state it", captured["p"])
        self.assertIn("COMPETING_ACCOUNTS", captured["p"])

    def test_non_causal_cause_facts_become_details_and_the_chain_orders_the_story(self):
        facts = [{"id": "E001", "kind": "choice", "fact": "A healer must decide whether to burn the granary."},
                 {"id": "E003", "kind": "consequence", "fact": "Burning it stops the plague but leaves no seed."},
                 {"id": "E005", "kind": "cause", "fact": "The plague began because the elders stored infected grain."},
                 {"id": "E002", "kind": "account", "fact": "The healer believes her sister cannot be moved."}]
        chain = g.story_chain_text(facts)
        order = [chain.index(k) for k in ("CAUSE [E005]", "ACCOUNT [E002]", "CHOICE [E001]", "CONSEQUENCE [E003]")]
        self.assertEqual(order, sorted(order))
        prompt = g.build_generation_prompt(GAME, TOPIC, ts.EVIDENCE, IDS, ts.HOOK_A, "B", chain=chain)
        self.assertIn("VERIFIED STORY CHAIN", prompt)
        self.assertIn("Never invent a link between two events", prompt)

    def test_12_a_script_whose_payoff_does_not_resolve_the_hook_ranks_lower(self):
        audits = [{"index": 1, "section": "HOOK", "content": ts.HOOK_A, "status": "CLEAR", "reason": "", "claims": []}]
        evaluation = {"clarity": 4, "opening": 4, "progression": 4, "delivery": 4, "generic_ending": False, "why": ""}
        delivers = g.score_script(ts.SCRIPT_A, GAME, audits, {"status": "GOOD", "verdict": "PASS"}, {"verdict": "DELIVERS", "text": ""}, [], evaluation)
        fails = g.score_script(ts.SCRIPT_A, GAME, audits, {"status": "GOOD", "verdict": "PASS"}, {"verdict": "FAILS", "text": ""}, [], evaluation)
        self.assertEqual(delivers["total"] - fails["total"], g.COHERENCE_FAIL_PENALTY)
        self.assertTrue(fails["eligible"])

    def test_a_failed_retention_review_lowers_the_score_but_does_not_discard_a_verified_script(self):
        audits = [{"index": 1, "section": "HOOK", "content": ts.HOOK_A, "status": "CLEAR", "reason": "", "claims": []}]
        evaluation = {"clarity": 4, "opening": 4, "progression": 4, "delivery": 4, "generic_ending": False, "why": ""}
        passed = g.score_script(ts.SCRIPT_A, GAME, audits, {"status": "GOOD", "verdict": "PASS"}, None, [], evaluation)
        failed = g.score_script(ts.SCRIPT_A, GAME, audits, {"status": "POOR", "verdict": "FAIL"}, None, [], evaluation)
        self.assertTrue(failed["eligible"])
        self.assertEqual(passed["total"] - failed["total"], g.RETENTION_FAIL_PENALTY)

    def test_11b_end_to_end_every_citation_in_the_output_is_a_selected_passage(self):
        self.run_main(ts.ScriptFake({ts.HOOK_A: [ts.SCRIPT_A], ts.HOOK_B: [ts.SCRIPT_B_GOOD]}))
        draft = self.outputs("REVIEW_REQUIRED")[0].read_text(encoding="utf-8")
        cited = set(re.findall(r"\[(E\d+)\]", draft))
        self.assertTrue(cited and cited <= IDS)


class DescriptionTests(unittest.TestCase):
    """Descriptive words are judged by the auditor (TYPE: DESCRIPTION), not by a word list in code."""
    def setUp(self):
        g._LOCATE_CACHE.clear()

    def test_a_descriptive_word_never_flags_a_line_or_needs_a_citation(self):
        line = "The healer faces a grim, harrowing choice over the granary. [E001]"
        fake = AuditFake({g.remove_citation(line): [
            ("The choice is grim and harrowing", "NOT_STATED", "NONE", "DESCRIPTION"),
            ("The healer must decide whether to burn the granary", "SUPPORTED", "must decide whether to burn the plague-ridden granary")]})
        result = audit(line, fake)
        self.assertEqual(result["status"], "CLEAR")
        self.assertEqual(result["rows"][0]["status"], "DESCRIPTION")
        self.assertNotIn("locate", fake.calls)                       # no evidence search for a description

    def test_a_description_does_not_count_towards_the_citation_limit(self):
        hook = "A hideous plague from hidden grain forces a healer to burn the granary where her sister lies. [E001]"
        fake = AuditFake({g.remove_citation(hook): [
            ("The plague is hideous", "NOT_STATED", "NONE", "DESCRIPTION"),
            ("The plague began because grain was hidden", "SUPPORTED", "The plague began because the town elders stored infected grain"),
            ("A healer must decide whether to burn the granary", "SUPPORTED", "must decide whether to burn the plague-ridden granary"),
            ("Her sister lies inside", "SUPPORTED", "sister lies sick inside the granary")]})
        candidates = [{"text": hook}]
        with mock.patch.object(g, "ask_ollama", fake):
            g.audit_hook_candidates(candidates, ts.EVIDENCE, GAME)
        self.assertTrue(candidates[0]["grounded"], candidates[0]["audit_reason"])
        self.assertEqual(set(g.line_citations(candidates[0]["text"])), {"E001", "E002", "E005"})

    def test_an_item_called_a_description_that_states_a_cause_is_still_checked(self):
        line = "The healer is cursed because the elders hated her. [E005]"
        claim = "The healer is cursed because the elders hated her"
        fake = AuditFake({g.remove_citation(line): [(claim, "NOT_STATED", "NONE", "DESCRIPTION")]}, locate={claim: None})
        self.assertEqual(audit(line, fake)["status"], "FLAGGED")

    def test_too_many_sources_is_reported_as_a_citation_problem_not_as_unsupported(self):
        line = "Elders hid grain, a healer must burn the granary, her sister is inside, and the elders gave her until dawn. [E001]"
        fake = AuditFake({g.remove_citation(line): [
            ("The elders hid infected grain", "SUPPORTED", "stored infected grain without telling the healer"),
            ("A healer must decide whether to burn the granary", "SUPPORTED", "must decide whether to burn the plague-ridden granary"),
            ("Her sister is inside", "SUPPORTED", "sister lies sick inside the granary"),
            ("The elders gave her until dawn", "SUPPORTED", "The town elders have given Mirela until dawn")]})
        with mock.patch.object(g, "MAX_CITATIONS_PER_LINE", 3):
            result = audit(line, fake)
        self.assertEqual(result["status"], "FLAGGED")
        self.assertFalse(result["unsupported"])                     # nothing is called unsupported
        self.assertIn("every claim is supported, but by 4 different passages", result["reason"])

    def test_a_citation_fix_keeps_the_passages_the_line_already_uses(self):
        line = "Mirela must decide whether to burn it while her sister lies inside. [E001]"
        fake = AuditFake({g.remove_citation(line): [
            ("Mirela must decide whether to burn the granary", "SUPPORTED", "must decide whether to burn the plague-ridden granary"),
            ("Her sister lies inside", "SUPPORTED", "sister lies sick inside the granary")]})
        result = audit(line, fake)
        self.assertEqual(result["status"], "RECITE")
        self.assertEqual(result["recite_ids"], ["E001", "E002"])      # E001 kept, E002 added

    def test_the_fixed_hook_is_never_recited_in_the_script_stage(self):
        audits = [{"index": 1, "section": "HOOK", "content": ts.HOOK_A, "status": "RECITE", "recite_ids": ["E003"], "reason": "x"}]
        draft, changed = g.apply_recitations(ts.SCRIPT_A, audits)
        self.assertEqual(changed, [])
        self.assertFalse(g.is_flagged(audits[0]))


class CauseSearchTests(unittest.TestCase):
    def test_the_cause_search_runs_without_the_brief_and_verifies_quotes(self):
        prompts = []

        def fake(prompt, **k):
            prompts.append(prompt)
            return ("FACT 1: The plague began because the elders hid infected grain. || QUOTE: stored infected grain without telling the healer || ID: E001 || KIND: cause\n"
                    "FACT 2: A curse caused the plague. || QUOTE: an ancient curse fell on the town || ID: E002 || KIND: account")
        with mock.patch.object(g, "ask_ollama", fake):
            facts, notes = g.extract_cause_facts(GAME, TOPIC, ts.EVIDENCE, IDS)
        self.assertEqual([(f["id"], f["kind"]) for f in facts], [("E005", "cause")])   # re-cited to where the quote is
        self.assertTrue(any("quote not found" in n for n in notes))                     # an invented cause is dropped
        self.assertNotIn("STORY ANGLE", prompts[0])
        self.assertIn("DIFFERENT explanations", prompts[0])

    def test_an_option_is_not_treated_as_the_cause_that_must_be_stated(self):
        facts = [{"id": "E005", "kind": "cause", "fact": "The plague began because the elders stored infected grain."},
                 {"id": "E003", "kind": "cause", "fact": "The granary can be burned, which leads to no seed in spring."}]
        self.assertEqual([f["id"] for f in g.essential_facts(facts)], ["E005"])


class HookClarityTests(unittest.TestCase):
    NAMED = "A grieving healer begs Orsk to spare the granary where her sister lies. [E002]"
    CLEAR = "A grieving healer begs the town warden to spare the granary where her sister lies. [E002]"

    def _fake(self, scores):
        prompts = []

        def fake(prompt, temperature=0.2, model=None, options=None):
            prompts.append(prompt)
            if "Does any passage below state this claim?" in prompt:
                return "NONE"
            if "will tell the story of" in prompt and "CANDIDATES:" in prompt:
                return "IDS: NONE"
            if "fair fact-checker" in prompt:
                return "CLAIM 1: her sister lies inside || TYPE: FACT || VERDICT: SUPPORTED || QUOTE: sister lies sick inside the granary"
            if "write the first spoken sentence" in prompt:
                return f"HOOK 1 (EMOTIONAL_CONFLICT): {self.NAMED}"
            if "a newcomer does not understand" in prompt:
                return self.CLEAR
            if "distracted viewer" in prompt:
                rows = []
                for line in prompt.split("HOOKS:\n", 1)[1].splitlines():
                    number, text = line[len("HOOK "):].split(": ", 1)
                    c, sp, cu, d, unexplained = scores[text]
                    rows.append(f"HOOK {number}: CLARITY={c} SPECIFICITY={sp} CURIOSITY={cu} DELIVERY={d} UNEXPLAINED={unexplained} ANY_GAME=NO WHY=t")
                return "\n".join(rows)
            raise AssertionError(prompt[:80])
        return fake, prompts

    def test_a_hook_held_back_only_by_an_unexplained_name_is_clarified_not_discarded(self):
        scores = {g.remove_citation(self.NAMED): (2, 4, 4, 4, "Orsk"), g.remove_citation(self.CLEAR): (4, 4, 4, 4, "NONE")}
        fake, prompts = self._fake(scores)
        with mock.patch.object(g, "ask_ollama", fake), mock.patch.object(g, "HOOK_GEN_ATTEMPTS", 1):
            hook = g.generate_and_select_hook(GAME, TOPIC, ts.EVIDENCE, IDS, "BRIEF")
        self.assertEqual(hook, self.CLEAR)
        clarify = next(p for p in prompts if "a newcomer does not understand" in p)
        self.assertIn("Orsk", clarify)                               # it is told exactly which term to explain
        self.assertEqual(sum("fair fact-checker" in p for p in prompts), 2)   # the rewrite is audited too

    def test_unclear_hooks_are_rewritten_only_when_they_could_beat_the_best_eligible_hook(self):
        def cand(text, clarity, emotion, curiosity=3):
            return {"text": text, "grounded": True, "approach": "X", "unexplained_terms": ["Orsk"],
                    "evaluation": {"clarity": clarity, "specificity": 4, "curiosity": curiosity, "emotion": emotion, "delivery": 4},
                    "score": {"blockers": ["clarity 2/5 is below 3"], "total": 55}}
        weak, strong = cand("Orsk weighs a granary. [E001]", 2, 1), cand("Orsk weeps for her trapped sister. [E002]", 2, 5)
        prompts = []
        with mock.patch.object(g, "ask_ollama", lambda p, **k: prompts.append(p) or "A healer weeps for her trapped sister. [E002]"):
            variants = g.clarify_hooks(GAME, TOPIC, [weak, strong], [], ts.EVIDENCE, IDS, beat_rank=14)
        self.assertEqual(len(prompts), 1)                              # only the one that could win: 2*5 + 3 + 3 = 16 >= 14
        self.assertIn("weeps for her trapped sister", prompts[0])
        self.assertEqual(len(variants), 1)

    def test_the_judges_unexplained_terms_are_not_penalised_twice(self):
        candidate = {"text": self.NAMED, "grounded": True}
        evaluation = {"clarity": 3, "specificity": 4, "curiosity": 4, "delivery": 4, "unexplained": ["Orsk"], "any_game": False, "why": ""}
        score = g.score_hook(candidate, evaluation, GAME, PASSAGES)
        self.assertFalse(any("unexplained" in p["reason"] for p in score["penalties"]))
        self.assertEqual(candidate["unexplained_terms"], ["Orsk"])

    def test_a_word_or_two_over_the_hook_target_is_not_penalised(self):
        evaluation = {"clarity": 4, "specificity": 4, "curiosity": 4, "delivery": 4, "unexplained": [], "any_game": False, "why": ""}
        nineteen = "A grieving village healer must decide tonight whether to burn the old granary where her sick sister still lies. [E001]"
        self.assertEqual(g.hook_word_count(nineteen, GAME), 18 + 1)
        score = g.score_hook({"text": nineteen, "grounded": True}, evaluation, GAME, PASSAGES)
        self.assertFalse(any("spoken target" in p["reason"] for p in score["penalties"]))

    def test_the_judge_is_told_ordinary_words_are_never_unexplained(self):
        prompt = g.build_hook_evaluation_prompt(GAME, [(1, "x")])
        self.assertIn("Ordinary English words and phrases are never UNEXPLAINED", prompt)


class RunFixTests(ts.PipelineHarness):
    def test_the_game_title_is_put_back_after_a_repair_rewrites_the_first_narration_line(self):
        hook = "A healer must burn a plague-ridden granary. [E001]"
        draft = ts.SCRIPT_A.replace(ts.HOOK_A, hook).replace("In Ashfall Keep, the healer is Mirela", "The healer is Mirela")
        bad_line = "In Ashfall Keep, the healer is Mirela, and the town elders have given her until dawn."
        base = ts.ScriptFake({hook: [draft]}, unsupported={bad_line})

        def fake(prompt, temperature=0.2, model=None, options=None):
            if "Rewrite ONE flagged line" in prompt:
                return "NARRATION: The town elders have given the healer Mirela until dawn to decide. [E004]"
            return base(prompt, temperature, model, options)
        with mock.patch.object(g, "ask_ollama", fake):
            result = g.produce_script(GAME, TOPIC, ts.EVIDENCE, IDS, PASSAGES, hook, "brief", {}, 1)
        first_narration = [c for sec, c in g.spoken_lines(result["draft"]) if sec == "NARRATION"][0]
        self.assertTrue(first_narration.startswith("In Ashfall Keep, the town elders"), first_narration)
        self.assertFalse(any("game title" in p for p in result["problems"]))

    def test_a_cause_told_outside_the_selected_passages_is_found_and_added(self):
        full = ts.EVIDENCE + """
[E006]
Source: Elder's diary
Excerpt: (Journal entry) The eldest elder admits the council sealed the infected grain inside the granary to hide their mistake.
"""
        selected = "\n\n".join(g.compact_passage(p) for p in g.split_evidence_passages(ts.EVIDENCE))
        prompts = []

        def fake(prompt, **k):
            prompts.append(prompt)
            if "[E006]" in prompt:
                return "FACT 1: The council sealed infected grain in the granary to hide a mistake, which led to the plague. || QUOTE: the council sealed the infected grain inside the granary || ID: E006 || KIND: account"
            return "NONE"
        with mock.patch.object(g, "ask_ollama", fake), mock.patch.object(g, "CAUSE_SEARCH", True):
            focused, ids, facts, notes = g.search_causes_all_evidence(GAME, TOPIC, full, selected, set(IDS))
        self.assertIn("E006", ids)
        self.assertIn("sealed the infected grain", focused)
        self.assertEqual([(f["id"], f["kind"]) for f in facts], [("E006", "account")])
        self.assertGreaterEqual(len(prompts), 2)                     # selected passages first, then the rest


class EmotionSelectionTests(unittest.TestCase):
    def test_the_most_emotionally_charged_clear_hook_is_selected_even_with_a_lower_total(self):
        calm = "A healer must decide whether to burn a plague-ridden granary full of seed. [E001]"
        moving = "A healer must burn the granary where her own sick sister lies, or let the plague spread. [E002]"
        scores = {g.remove_citation(calm): (5, 5, 4, 1, 5), g.remove_citation(moving): (4, 4, 4, 5, 4)}

        def fake(prompt, temperature=0.2, model=None, options=None):
            if "Does any passage below state this claim?" in prompt:
                return "NONE"
            if "fair fact-checker" in prompt:
                quote = "sister lies sick inside the granary" if "sister" in prompt.split("LINE (", 1)[1] else "must decide whether to burn the plague-ridden granary"
                return f"CLAIM 1: x || TYPE: FACT || VERDICT: SUPPORTED || QUOTE: {quote}"
            if "write the first spoken sentence" in prompt:
                return f"HOOK 1 (MORAL_DILEMMA): {calm}\nHOOK 2 (EMOTIONAL_CONFLICT): {moving}"
            if "distracted viewer" in prompt:
                rows = []
                for line in prompt.split("HOOKS:\n", 1)[1].splitlines():
                    number, text = line[len("HOOK "):].split(": ", 1)
                    c, sp, cu, e, d = scores[text]
                    rows.append(f"HOOK {number}: CLARITY={c} SPECIFICITY={sp} CURIOSITY={cu} EMOTION={e} DELIVERY={d} UNEXPLAINED=NONE ANY_GAME=NO WHY=t")
                return "\n".join(rows)
            raise AssertionError(prompt[:80])
        with mock.patch.object(g, "ask_ollama", fake):
            hook = g.generate_and_select_hook(GAME, TOPIC, ts.EVIDENCE, IDS, "BRIEF")
        self.assertEqual(hook, moving)
        self.assertIn("EMOTION:", g.build_hook_evaluation_prompt(GAME, [(1, "x")]))
        self.assertEqual(sum(g.HOOK_WEIGHTS.values()), 100)

    def test_emotion_never_overrides_the_fact_check(self):
        evaluation = {"clarity": 5, "specificity": 4, "curiosity": 5, "emotion": 5, "delivery": 5, "unexplained": [], "any_game": False, "why": ""}
        score = g.score_hook({"text": "A grieving healer weeps over her sister. [E002]", "grounded": False}, evaluation, GAME, PASSAGES)
        self.assertFalse(score["eligible"])                          # a made-up hook stays out however moving it is
        score = g.score_hook({"text": "A grieving healer weeps over her sister. [E002]", "grounded": True}, dict(evaluation, clarity=2), GAME, PASSAGES)
        self.assertTrue(score["eligible"])                           # a verified one is never rejected for style


class StoryPlanTests(unittest.TestCase):
    CAUSE = {"id": "E005", "kind": "cause", "fact": "The plague began because the elders stored infected grain without telling the healer."}
    BACKGROUND = {"id": "E004", "kind": "cause", "fact": "Years earlier the elders quarrelled, which led to the healer being distrusted."}
    CHOICE = {"id": "E001", "kind": "choice", "fact": "The healer must decide whether to burn the granary."}
    OUTCOME = {"id": "E003", "kind": "consequence", "fact": "Burning it stops the plague but leaves the town no seed for spring."}
    SISTER = {"id": "E002", "kind": "detail", "fact": "Her sister lies sick inside the granary and cannot be moved."}

    def test_background_causes_are_not_forced_into_the_script(self):
        facts = [dict(self.BACKGROUND), dict(self.CAUSE), dict(self.CHOICE)]
        with mock.patch.object(g, "ask_ollama", lambda p, **k: "1: BACKGROUND\n2: DIRECT"):
            facts = g.rank_cause_facts(GAME, TOPIC, facts)
        self.assertEqual([f["id"] for f in g.essential_facts(facts)], ["E005"])

    def test_competing_accounts_are_required_alongside_the_cause(self):
        account = {"id": "E002", "kind": "account", "fact": "The healer believes a curse caused the plague."}
        facts = [dict(self.CAUSE), dict(account)]
        with mock.patch.object(g, "ask_ollama", lambda p, **k: "1: DIRECT\n2: DIRECT"):
            facts = g.rank_cause_facts(GAME, TOPIC, facts)
        self.assertEqual({f["id"] for f in g.essential_facts(facts)}, {"E005", "E002"})

    def test_duplicate_facts_are_merged(self):
        again = dict(self.CAUSE, fact="The plague began because the elders stored the infected grain without telling the healer.")
        self.assertEqual(len(g.dedupe_facts([self.CAUSE, again, self.CHOICE])), 2)

    def test_the_timeline_keeps_required_facts_in_time_order_uses_each_once_and_never_pays_off_with_the_hook(self):
        hook = "A healer must decide whether to burn the granary her town depends on. [E001]"
        facts = [self.CAUSE, self.CHOICE, self.SISTER, self.OUTCOME]
        # the model orders the sister (2) before the cause, repeats a fact, forgets the cause, and names no payoff
        reply = "TIMELINE: 2, 2\nPAYOFF: none"
        with mock.patch.object(g, "ask_ollama", lambda p, **k: reply):
            plan = g.plan_story_beats(GAME, TOPIC, hook, facts, [self.CAUSE])
        self.assertIn("in the order they happened", plan)
        self.assertIn("[E005]", plan)                                         # the required cause was put back...
        self.assertLess(plan.index("[E002]"), plan.index("[E005]"))         # ...after the model's chosen opener
        self.assertEqual(plan.count("Her sister lies sick"), 1)               # no fact twice
        self.assertIn("PAYOFF", plan)
        self.assertIn("[E003] Burning it stops the plague", plan)            # an outcome the hook has not given away
        self.assertNotIn("must decide whether to burn the granary.", plan)    # the hook's own premise is not planned again
        with mock.patch.object(g, "ask_ollama", lambda p, **k: "TIMELINE: 1, 2\nPAYOFF: 3"):
            ordered = g.plan_story_beats(GAME, TOPIC, hook, facts, [])
        self.assertLess(ordered.index("1. [E005]"), ordered.index("2. [E002]"))

    def test_the_writer_gets_the_plan_and_naming_and_ending_rules(self):
        prompt = g.build_generation_prompt(GAME, TOPIC, ts.EVIDENCE, IDS, ts.HOOK_A, "B", chain="STORY TIMELINE (x):\nPAYOFF: [E003] y")
        self.assertIn("STORY TIMELINE", prompt)
        self.assertNotIn("Excerpt:", prompt)                         # with a timeline, the evidence dump is not sent
        self.assertIn("one name per person", prompt)
        long_q = ts.SCRIPT_A.replace("Would you burn the granary with your sister still inside it?",
                                     "Should the healer burn the granary to save the town even though her sister is inside and cannot be moved at all?")
        self.assertTrue(any("ENDING QUESTION is" in p for p in g.check_script_format(long_q, IDS, ts.HOOK_A, GAME)))


class StoryFirstTests(unittest.TestCase):
    def test_the_writer_is_a_storyteller_with_one_evidence_rule_and_no_citations(self):
        prompt = g.build_generation_prompt(GAME, TOPIC, ts.EVIDENCE, IDS, ts.HOOK_A, "B")
        self.assertIn("THE ONE EVIDENCE RULE", prompt)
        self.assertIn("Invent no facts. Everything else is yours", prompt)
        self.assertIn("no citations", prompt)
        self.assertNotIn("close paraphrase", prompt)
        self.assertNotIn("CITATION FORMAT", prompt)

    def test_code_attaches_valid_citations_to_uncited_story_lines(self):
        draft = """STATUS: DRAFT
HOOK: {hook}
NARRATION: In Ashfall Keep, the elders stored infected grain and never told the healer. So the plague began.
NARRATION: Now Mirela has until dawn, and her sister lies inside the granary.
PAYOFF: Burning it stops the plague, but the town will have no seed for spring.
ENDING QUESTION: Would you burn it with your sister inside?
CLAIMS REQUIRING MORE EVIDENCE: None""".format(hook=ts.HOOK_A)
        facts = [{"id": "E005", "kind": "cause", "fact": "The plague began because the elders stored infected grain."}]
        cited = g.attach_citations(draft, PASSAGES, facts, GAME, IDS)
        lines = g.spoken_lines(cited)
        self.assertTrue(all(g.line_citations(c) for _, c in lines))
        self.assertTrue(all(set(g.line_citations(c)) <= IDS for _, c in lines))
        self.assertEqual(g.line_citations(lines[1][1])[0], "E005")           # the verified fact it echoes comes first
        self.assertEqual(lines[0][1], ts.HOOK_A)                             # the hook is untouched
        self.assertEqual(g.check_script_format(cited, IDS, ts.HOOK_A, GAME), [])   # two-sentence lines are allowed

    def test_three_sentences_in_one_line_is_still_a_layout_problem(self):
        draft = ts.SCRIPT_A.replace("Her own sister lies sick inside the granary and cannot be moved.",
                                    "Her sister is inside. She cannot be moved. Nobody can help.")
        self.assertTrue(any("sentences" in p for p in g.check_script_format(draft, IDS, ts.HOOK_A, GAME)))

    def test_repairs_keep_the_voice_and_storytelling_turns_count_as_description(self):
        captured = {}

        def fake(prompt, **k):
            captured.setdefault("p", prompt)
            return "CLAIM 1: x || TYPE: DESCRIPTION || VERDICT: SUPPORTED || QUOTE: NONE"
        with mock.patch.object(g, "ask_ollama", fake):
            result = g.audit_line("NARRATION", "But the truth was darker. [E005]", PASSAGES, GAME)
        self.assertEqual(result["status"], "CLEAR")
        self.assertIn("storytelling device", captured["p"])
        fake2 = AuditFake(repairs=["NARRATION: But the truth was darker: the elders had stored infected grain. [E005]"])
        audit_row = {"section": "NARRATION", "content": "But the truth was darker: the elders poisoned the grain on purpose. [E005]",
                     "rows": [{"claim": "The elders poisoned it on purpose", "status": "UNSUPPORTED", "quote": "NONE", "note": "", "found_in": None, "verdict": "NOT_STATED"}],
                     "unsupported": [("The elders poisoned it on purpose", "NONE")]}
        with mock.patch.object(g, "ask_ollama", fake2):
            fixed = g.repair_line(GAME, TOPIC, audit_row, PASSAGES, IDS, 24)
        self.assertIn("Keep the line's wording, voice, rhythm", fake2.repair_prompts[0])
        self.assertTrue(fixed.startswith("But the truth was darker"))


class CalibrationTests(unittest.TestCase):
    def test_the_judges_are_not_anchored_to_low_example_scores(self):
        hook_prompt = g.build_hook_evaluation_prompt(GAME, [(1, "x")])
        script_prompt = g.build_script_evaluation_prompt(GAME, ts.SCRIPT_A)
        for prompt in (hook_prompt, script_prompt):
            self.assertNotIn("most hooks deserve 2-3", prompt)
            self.assertNotIn("most scripts deserve 2-3", prompt)
            self.assertNotRegex(prompt, r"CLARITY=\d")              # the format line shows placeholders, not numbers to copy
            self.assertIn("3 = good enough to post", prompt)
        self.assertIn("SPECIFICITY: does it contain at least one concrete, distinctive detail", hook_prompt)

    def test_hook_prompts_carry_no_concrete_example_that_can_leak_into_hooks(self):
        prompt = g.build_hook_prompt(GAME, TOPIC, ts.EVIDENCE, IDS, "BRIEF")
        for leak in ("lighthouse", "ferryman", "harbour", "swamp village"):
            self.assertNotIn(leak, prompt)
        self.assertNotIn("harbour", g.build_generation_prompt(GAME, TOPIC, ts.EVIDENCE, IDS, ts.HOOK_A, "B"))

    def test_an_over_long_hook_is_tightened_instead_of_discarded(self):
        long_hook = ("HOOK 1 (MORAL_DILEMMA): A village healer who has served the town for years must decide tonight whether "
                     "to burn the plague-ridden granary that also holds the last seed grain. [E001]")
        short = "A healer must burn the plague-ridden granary that holds the town's last seed grain. [E001]"
        with mock.patch.object(g, "ask_ollama", lambda p, **k: short):
            candidates, reasons, _ = g.parse_hook_candidates(long_hook, IDS, GAME)
        self.assertEqual([c["text"] for c in candidates], [short])
        self.assertTrue(any("tightened" in r for r in reasons))

    def test_a_synonym_does_not_overrule_a_verified_second_opinion(self):
        line = "The ill sibling stays inside the granary. [E001]"
        claim = "The ill sibling stays inside the granary"
        # the auditor's quote shares no key word with the claim, so a second opinion is sought; it finds the claim
        # stated (in other words) and its quote is verified, so the claim is supported, only mis-cited
        fake = AuditFake({g.remove_citation(line): [(claim, "SUPPORTED", "must decide whether to burn")]},
                         locate={claim: ("E002", "SUPPORTED", "cannot be moved")})
        g._LOCATE_CACHE.clear()
        self.assertEqual(audit(line, fake)["rows"][0]["status"], "SUPPORTED_WRONG_CITATION")

class OnlyMadeUpRejectsTests(unittest.TestCase):
    def test_the_judges_know_the_video_shows_what_the_words_leave_unnamed(self):
        for prompt in (g.build_hook_evaluation_prompt(GAME, [(1, "x")]), g.build_script_evaluation_prompt(GAME, ts.SCRIPT_A)):
            self.assertIn("plays over gameplay footage", prompt)
            self.assertIn("A character's NAME is therefore clear", prompt)

    def test_a_verified_but_plainly_worded_hook_is_kept_with_notes_not_rejected(self):
        hook = "A man's abuse of his wife led to the plague in the granary. [E005]"
        evaluation = {"clarity": 4, "specificity": 2, "curiosity": 3, "emotion": 4, "delivery": 4, "unexplained": [], "any_game": False, "why": ""}
        score = g.score_hook({"text": hook, "grounded": True}, evaluation, GAME, PASSAGES)
        self.assertTrue(score["eligible"])
        self.assertEqual(score["blockers"], [])
        self.assertIn("specificity 2/5", score["notes"])

    def test_a_retyped_hook_is_put_back_in_code_not_failed(self):
        draft = ts.SCRIPT_A.replace(ts.HOOK_A, "A healer must burn the granary. [E001]")
        fixed = g.enforce_hook(draft, ts.HOOK_A)
        self.assertEqual(g.spoken_lines(fixed)[0], ("HOOK", ts.HOOK_A))
        self.assertEqual([p for p in g.check_script_format(fixed, IDS, ts.HOOK_A, GAME) if g.is_structural(p)], [])

    def test_only_structure_and_invented_facts_block_a_script(self):
        audits = [{"index": 1, "section": "HOOK", "content": ts.HOOK_A, "status": "CLEAR", "reason": "", "claims": []}]
        style = ["Script is 160 words; target is 55-144.", "NARRATION must contain one or 2 sentences.", "ENDING QUESTION is 25 words"]
        score = g.score_script(ts.SCRIPT_A, GAME, audits, {"status": "POOR", "verdict": "FAIL"}, {"verdict": "FAILS", "text": ""}, style, None,
                               [{"id": "E005", "fact": "x"}])
        self.assertTrue(score["eligible"], score["blockers"])
        self.assertFalse(g.score_script(ts.SCRIPT_A, GAME, audits, None, None, ["Exactly one PAYOFF is required."], None)["eligible"])


class HookAndChoiceTests(unittest.TestCase):
    def setUp(self):
        g._LOCATE_CACHE.clear()
        g.VERIFIED_HOOKS.clear()

    def test_a_hook_verified_at_the_hook_stage_is_not_re_judged_in_the_script(self):
        fake = AuditFake({g.remove_citation(ts.HOOK_A): [("Burning prevents the plague", "CONTRADICTED", "the plague stops spreading")]})
        g.VERIFIED_HOOKS.add(ts.HOOK_A)
        with mock.patch.object(g, "ask_ollama", fake):
            audits = g.audit_script(ts.SCRIPT_A, PASSAGES, GAME, {})
        self.assertEqual(audits[0]["status"], "CLEAR")
        self.assertEqual(sum(1 for c in fake.calls if c == "audit"), len(audits) - 1)   # the hook was not sent again

    def test_a_contradiction_needs_a_second_opinion(self):
        line = "Burning the granary stops the plague. [E003]"
        claim = "Burning the granary prevents the plague from spreading"
        fake = AuditFake({g.remove_citation(line): [(claim, "CONTRADICTED", "the plague stops spreading")]},
                         locate={claim: ("E003", "SUPPORTED", "If the granary burns, the plague stops spreading")})
        self.assertEqual(audit(line, fake)["status"], "CLEAR")
        fake2 = AuditFake({g.remove_citation(line): [(claim, "CONTRADICTED", "the plague stops spreading")]}, locate={claim: None})
        g._LOCATE_CACHE.clear()
        self.assertEqual(audit(line, fake2)["status"], "FLAGGED")      # confirmed (nothing supports it): still blocked

    def test_player_choices_are_told_as_options_and_who_did_what_is_checked(self):
        writer = g.build_generation_prompt(GAME, TOPIC, ts.EVIDENCE, IDS, ts.HOOK_A, "B")
        self.assertIn("tell it as the player's choice and its result", writer)
        self.assertIn("Get WHO does each action exactly right", writer)
        captured = {}
        with mock.patch.object(g, "ask_ollama", lambda p, **k: captured.setdefault("p", p) and "CLAIM 1: x || TYPE: FACT || VERDICT: SUPPORTED || QUOTE: NONE"):
            g.audit_line("PAYOFF", "The healer burned the granary. [E003]", PASSAGES, GAME)
        self.assertIn("an optional player choice is not an event", captured["p"])
        self.assertIn("giving an action to the wrong person is CONTRADICTED", captured["p"])


class WriterQualityTests(ts.PipelineHarness):
    def test_the_best_flowing_draft_is_kept(self):
        flat, flowing = ts.SCRIPT_B_GOOD, ts.SCRIPT_A
        def fake(prompt, **k):
            p = 5 if "Would you burn" in prompt else 2
            return f"CLARITY=4 OPENING=4 PROGRESSION={p} DELIVERY=4 REPEATS=NO GENERIC_ENDING=NO WHY=t"
        with mock.patch.object(g, "ask_ollama", fake):
            self.assertEqual(g.pick_best_draft(GAME, [flat, flowing]), flowing)

    def test_the_editor_pass_keeps_the_hook_and_never_replaces_a_draft_with_a_broken_edit(self):
        material = "STORY TIMELINE (x):\n1. [E005] The plague began because the elders hid infected grain."
        prompts = []
        edited = ts.SCRIPT_A.replace(ts.HOOK_A, "Some other hook. [E001]").replace("Her own sister", "And then her own sister")
        with mock.patch.object(g, "ask_ollama", lambda p, **k: prompts.append(p) or edited):
            out = g.revise_for_flow(GAME, TOPIC, ts.SCRIPT_A, ts.HOOK_A, material)
        self.assertEqual(g.spoken_lines(out)[0], ("HOOK", ts.HOOK_A))         # the hook is put back verbatim
        self.assertIn("And then her own sister", out)
        self.assertIn("add NO new facts", prompts[0])
        self.assertIn("in the order they happened", prompts[0])
        with mock.patch.object(g, "ask_ollama", lambda p, **k: "Sorry, here is my analysis of the draft."):
            self.assertEqual(g.revise_for_flow(GAME, TOPIC, ts.SCRIPT_A, ts.HOOK_A, material), ts.SCRIPT_A)

    def test_whatever_the_editor_writes_is_still_fact_checked(self):
        hook = ts.HOOK_A
        invented = "The elders burned the healer's house down in revenge."
        base = ts.ScriptFake({hook: [ts.SCRIPT_A]}, unsupported={invented})

        def fake(prompt, temperature=0.2, model=None, options=None):
            if "You are the story editor" in prompt:
                return ts.SCRIPT_A.replace("Her own sister lies sick inside the granary and cannot be moved. [E002]", invented)
            return base(prompt, temperature, model, options)
        with mock.patch.object(g, "ask_ollama", fake), mock.patch.object(g, "REVISE_PASS", True):
            result = g.produce_script(GAME, TOPIC, ts.EVIDENCE, IDS, PASSAGES, hook, "brief", {}, 1,
                                      facts_for_plan=[{"id": "E005", "kind": "cause", "fact": "The plague began because the elders stored infected grain."},
                                                      {"id": "E003", "kind": "consequence", "fact": "Burning it stops the plague but leaves no seed."}])
        flagged = [a for a in result["audits"] if a["status"] == "FLAGGED"]
        self.assertTrue(flagged and invented in flagged[0]["content"])

    def test_background_history_is_defined_for_the_ranker(self):
        captured = {}
        facts = [{"id": "E005", "kind": "cause", "fact": "The plague began because the elders stored infected grain."},
                 {"id": "E004", "kind": "cause", "fact": "An old feud led to the elders distrusting the healer."}]
        with mock.patch.object(g, "ask_ollama", lambda p, **k: captured.setdefault("p", p) or "1: DIRECT"):
            g.rank_cause_facts(GAME, TOPIC, facts)
        self.assertIn("an old affair or killing, a childhood punishment, how someone got a title or nickname", captured["p"])
        self.assertIn("OR the rule that explains why that event brought it about", captured["p"])


class EvidenceSelectionTests(unittest.TestCase):
    EVIDENCE = """[E001]
Source: Creature page
Excerpt: (Infobox) Class: spirit; Games: Ashfall Keep; Loot: dust

[E002]
Source: Quest page
Excerpt: (Walkthrough) The keeper informs the hero that the miller's grief, owing to his neglect of the mill, turned the drowned child into a wailer.

[E003]
Source: Quest page
Excerpt: (Bugs) In Ashfall Keep the wailer sometimes clips through the mill wheel and reappears after reloading.

[E004]
Source: Miller page
Excerpt: (Biography) The miller left the sluice gate open one night while drunk, and his son drowned in the mill race.

[E005]
Source: Unrelated page
Excerpt: (Infobox) Games: Ashfall Keep; Profession: jester; Hair colour: grey

[E006]
Source: Quest page
Excerpt: (Walkthrough) If the hero frees the wailer, it leads him to the miller's hidden ledger.
"""

    def test_game_title_words_do_not_make_unrelated_pages_look_on_topic(self):
        self.assertEqual(g.topic_terms("Wailer", "Ashfall Keep"), {"wailer"})
        self.assertEqual(g.topic_terms("Ashfall Keep", "Ashfall Keep"), {"ashfall", "keep"})   # a topic that IS the game still works
        passages = {p["id"]: p for p in g.split_evidence_passages(self.EVIDENCE)}
        story = g.relevance_score(passages["E002"], "Wailer", "Ashfall Keep") + g.passage_role_bonus(passages["E002"]["text"], "Wailer", "Ashfall Keep")
        bug = g.relevance_score(passages["E003"], "Wailer", "Ashfall Keep") + g.passage_role_bonus(passages["E003"]["text"], "Wailer", "Ashfall Keep")
        self.assertGreater(story, bug)
        with mock.patch.object(g, "MAX_EVIDENCE_PASSAGES", 3):
            focused, ids, _ = g.select_relevant_evidence(self.EVIDENCE, "Wailer", "Ashfall Keep")
        self.assertIn("E002", ids)
        self.assertIn("E006", ids)
        self.assertNotIn("E005", ids)
        self.assertNotIn("URL:", focused)

    def test_the_model_can_add_a_cause_passage_that_never_names_the_topic_but_nothing_else(self):
        with mock.patch.object(g, "MAX_EVIDENCE_PASSAGES", 3):
            focused, ids, _ = g.select_relevant_evidence(self.EVIDENCE, "Wailer", "Ashfall Keep")
        self.assertNotIn("E004", ids)
        prompts = []

        def fake(prompt, **k):
            prompts.append(prompt)
            return "IDS: E004, E777, E002"     # E777 does not exist; E002 is already selected
        with mock.patch.object(g, "ask_ollama", fake):
            focused2, ids2, added = g.expand_story_evidence("Ashfall Keep", "Wailer", self.EVIDENCE, focused, ids)
        self.assertEqual(added, ["E004"])
        self.assertIn("E004", ids2)
        self.assertIn("left the sluice gate open", focused2)       # copied verbatim by code
        self.assertNotIn("[E001]", prompts[0].split("CANDIDATES:")[1])   # infobox furniture is not offered


class StyleExampleTests(unittest.TestCase):
    EXAMPLE = """GAME: Ember Reach
TOPIC: The cinder widow
RESULT: watched to the end
[hook] A widow keeps her husband's ashes burning so the thing inside them can never get out.
[story] So when the fire guttered one winter night, she fed it the only thing left: the cradle.
[payoff] Let it die, and the town is free. Keep it lit, and she is never alone.
[question] Is a promise to the dead worth the living?"""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        (self.dir / "cinder_widow.txt").write_text(self.EXAMPLE, encoding="utf-8")
        patcher = mock.patch.object(g, "STYLE_DIR", self.dir)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_writer_and_editor_see_a_finished_script_about_another_story(self):
        prompt = g.build_generation_prompt(GAME, TOPIC, ts.EVIDENCE, IDS, ts.HOOK_A, "B")
        self.assertIn("fed it the only thing left", prompt)
        self.assertIn("Copy the craft, never the content", prompt)
        self.assertNotIn("RESULT:", prompt)
        captured = {}
        with mock.patch.object(g, "ask_ollama", lambda p, **k: captured.setdefault("p", p) and ""):
            g.revise_for_flow(GAME, TOPIC, ts.SCRIPT_A, ts.HOOK_A, "material")
        self.assertIn("fed it the only thing left", captured["p"])

    def test_the_hook_writer_sees_the_examples_hook_and_is_told_not_to_give_the_ending_away(self):
        prompt = g.build_hook_prompt(GAME, TOPIC, ts.EVIDENCE, IDS, "brief")
        self.assertIn("- A widow keeps her husband's ashes burning", prompt)
        self.assertIn("never reuse their people", prompt)
        self.assertIn("Never give away the resolution", prompt)
        self.assertNotIn("HOOKS THAT KEPT", g.build_hook_prompt(GAME, "The cinder widow", ts.EVIDENCE, IDS, "brief"))

    def test_an_example_about_the_same_topic_is_not_shown(self):
        self.assertEqual(g.load_style_examples("Cinder widow"), [])
        self.assertNotIn("STYLE EXAMPLE", g.build_generation_prompt(GAME, "The cinder widow", ts.EVIDENCE, IDS, ts.HOOK_A, "B"))
        with mock.patch.object(g, "STYLE_EXAMPLES", 0):
            self.assertEqual(g.load_style_examples(TOPIC), [])

    def test_copied_wording_is_noticed(self):
        copied = ts.SCRIPT_A.replace("NARRATION:", "NARRATION: She fed it the only thing left, the grain.", 1)
        self.assertIn("fed it the only thing", " ".join(g.style_echoes(copied, TOPIC)))
        self.assertEqual(g.style_echoes(ts.SCRIPT_A, TOPIC), [])


class StyleExampleChoiceTests(unittest.TestCase):
    """With many examples, the proven one comes first, the next favours another game, and filler words in titles do
    not make two different topics look the same."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        files = {
            "proven.txt": "GAME: Ember Reach\nTOPIC: The cinder widow\nRESULT: watched to the end\n[hook] A widow keeps the fire lit.",
            "same_game.txt": "GAME: Ashfall Keep\nTOPIC: Why the miller drowned\n[hook] The miller opened the gate.",
            "other_game.txt": "GAME: Glass Sea\nTOPIC: How the tide witch was made\n[hook] The tide took her voice.",
            "about_granary.txt": "GAME: Glass Sea\nTOPIC: The granary of bones\n[hook] Bones in the grain.",
        }
        for name, text in files.items():
            (self.dir / name).write_text(text, encoding="utf-8")
        patcher = mock.patch.object(g, "STYLE_DIR", self.dir)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_proven_first_then_another_game(self):
        chosen = [t for t, _ in g.load_style_examples("The ferryman's debt", game="Ashfall Keep")]
        self.assertEqual(chosen, ["The cinder widow", "How the tide witch was made"])

    def test_filler_words_do_not_hide_an_example_but_a_shared_subject_does(self):
        topics = [t for t, _ in g.load_style_examples("How the ferryman became a ghost", limit=10, game="Ashfall Keep")]
        self.assertIn("How the tide witch was made", topics)          # shares only "how"
        self.assertNotIn("The granary of bones", [t for t, _ in g.load_style_examples("The granary choice", limit=10)])


class OnlyInventedFactsBlockTests(unittest.TestCase):
    """Creative freedom: only a claim that no passage supports blocks a script."""

    def setUp(self):
        g._LOCATE_CACHE.clear()
        g._QUOTE_SAYS_CACHE.clear()

    def test_a_synonym_of_the_evidence_is_not_an_invented_fact(self):
        line = "Burn it, and the sickness vanishes from the town. [E003]"
        claim = "Burning the granary makes the sickness vanish"
        fake = AuditFake({g.remove_citation(line): [(claim, "SUPPORTED", "the plague stops spreading")]}, locate={claim: None})
        fake.says = {claim}
        self.assertEqual(audit(line, fake)["status"], "CLEAR")
        self.assertIn("says?", fake.calls)

    def test_what_a_character_does_not_know_is_storytelling(self):
        line = "But Mirela has no idea what is really waiting inside. [E002]"
        fake = AuditFake({g.remove_citation(line): [("Mirela does not know what waits inside", "NOT_STATED", "NONE", "DESCRIPTION")]})
        self.assertEqual(audit(line, fake)["status"], "CLEAR")

    def test_lines_needing_review_never_block_a_script(self):
        review = {"status": "REVIEW", "section": "NARRATION", "index": 2, "reason": "only partly supported"}
        clear = {"status": "CLEAR", "section": "X", "index": 1, "reason": ""}
        score = g.score_script(ts.SCRIPT_A, GAME, [clear, review, review, review, clear, clear],
                               {"status": "GOOD", "verdict": "PASS", "text": ""}, {"verdict": "DELIVERS", "text": ""}, [],
                               {"clarity": 4, "opening": 4, "progression": 4, "delivery": 4, "repeats": False,
                                "generic_ending": False, "why": ""})
        self.assertTrue(score["eligible"])
        self.assertGreaterEqual(score["points"]["grounding"], 9)

    def test_a_hook_you_picked_is_never_judged_again_by_the_script_check(self):
        g.VERIFIED_HOOKS.clear()
        hook = "Mirela hears the granary whispering her sister's name."
        fake = AuditFake({hook: [("The granary whispers her sister's name", "NOT_STATED", "NONE")]})
        with mock.patch.object(g, "ask_ollama", fake), mock.patch.object(g, "CHECK_DIR", Path(tempfile.mkdtemp())):
            used = g.use_chosen_hook(hook, GAME, [], PASSAGES, ts.EVIDENCE, "x")
        self.assertIn(used, g.VERIFIED_HOOKS)
        draft = ts.SCRIPT_A.replace(ts.HOOK_A, used)
        with mock.patch.object(g, "ask_ollama", AuditFake()):
            audits = g.audit_script(draft, PASSAGES, GAME, {})
        self.assertEqual(audits[0]["status"], "CLEAR")

    def test_an_invented_event_still_blocks(self):
        line = "Then the elders burned the healer alive. [E005]"
        claim = "The elders burned the healer alive"
        fake = AuditFake({g.remove_citation(line): [(claim, "NOT_STATED", "NONE")]}, locate={claim: None})
        self.assertEqual(audit(line, fake)["status"], "FLAGGED")


class ConcretePayoffTests(unittest.TestCase):
    """The payoff must show what each option actually involves and costs, not just name the method."""

    def test_the_scene_of_each_option_is_pulled_out_and_every_quote_is_checked(self):
        reply = ("FACT 1: If the player burns the granary, the plague stops spreading. || QUOTE: the plague stops spreading || ID: E003 || KIND: consequence\n"
                 "FACT 2: Burning it also leaves the town no seed for spring. || QUOTE: the town has no seed for spring planting || ID: E001 || KIND: consequence\n"
                 "FACT 3: The healer can carry her sister out first. || QUOTE: carries her sister out at dawn || ID: E002 || KIND: choice")
        with mock.patch.object(g, "ask_ollama", lambda p, **k: reply):
            facts = g.extract_choice_facts(GAME, TOPIC, ts.EVIDENCE, IDS)
        self.assertEqual([f["fact"][:30] for f in facts], ["If the player burns the granar", "Burning it also leaves the tow"])
        self.assertEqual(facts[1]["id"], "E003")            # a real quote cited to the wrong passage is re-cited
        self.assertTrue(all(f["payoff_detail"] for f in facts))   # the invented one (fact 3) is gone

    def test_the_plan_always_carries_the_concrete_scene_into_the_payoff(self):
        hook = "A healer must decide whether to burn the granary her town depends on. [E001]"
        rule = {"id": "E005", "fact": "The plague began because the elders hid infected grain.", "quote": "x", "kind": "cause"}
        sister = {"id": "E002", "fact": "Her sister lies sick inside and cannot be moved.", "quote": "x", "kind": "detail"}
        scene = {"id": "E003", "fact": "If she burns it, the plague stops but the town has no seed.", "quote": "x",
                 "kind": "consequence", "payoff_detail": True}
        with mock.patch.object(g, "ask_ollama", lambda p, **k: "TIMELINE: 2, 1\nPAYOFF: none"):
            plan = g.plan_story_beats(GAME, TOPIC, hook, [rule, sister, scene], [])
        self.assertIn("what is actually done and what it leads to or costs", plan)
        self.assertGreater(plan.index("[E003] If she burns it"), plan.index("PAYOFF"))

    def test_the_twist_is_asked_for_and_the_plan_warns_against_the_terrible_version(self):
        captured = {}
        with mock.patch.object(g, "ask_ollama", lambda p, **k: captured.setdefault("p", p) and "NONE"):
            self.assertEqual(g.extract_choice_facts(GAME, TOPIC, ts.EVIDENCE, IDS), [])
        self.assertIn("HOW IT REALLY WORKS", captured["p"])
        self.assertIn("WHAT IT CHANGES LATER", captured["p"])
        hook = "A healer must decide whether to burn the granary her town depends on. [E001]"
        facts = [{"id": "E002", "fact": "Her sister lies sick inside and cannot be moved.", "quote": "x", "kind": "detail"},
                 {"id": "E003", "fact": "If she burns it, the plague stops.", "quote": "x", "kind": "consequence", "payoff_detail": True}]
        with mock.patch.object(g, "ask_ollama", lambda p, **k: "TIMELINE: 1\nPAYOFF: 2"):
            self.assertIn("only LOOKS terrible", g.plan_story_beats(GAME, TOPIC, hook, facts, []))

    def test_how_a_kind_of_creature_came_to_exist_is_background(self):
        captured = {}
        causes = [{"id": "E005", "fact": "The plague began because the elders hid infected grain.", "kind": "cause"},
                  {"id": "E001", "fact": "An old war led to the plague's arrival in the world.", "kind": "cause"}]
        with mock.patch.object(g, "ask_ollama", lambda p, **k: captured.setdefault("p", p) and "1: DIRECT\n2: BACKGROUND"):
            ranked = g.rank_cause_facts(GAME, TOPIC, causes)
        self.assertIn("first came into the world", captured["p"])
        self.assertEqual([f["direct"] for f in ranked], [True, False])

    def test_how_the_problem_is_solved_is_never_a_required_cause(self):
        captured = {}
        causes = [{"id": "E005", "fact": "The plague began because the elders hid infected grain.", "kind": "cause"},
                  {"id": "E003", "fact": "Burning the granary led to the plague stopping.", "kind": "cause"}]
        with mock.patch.object(g, "ask_ollama", lambda p, **k: captured.setdefault("p", p) and "1: DIRECT\n2: RESOLUTION"):
            ranked = g.rank_cause_facts(GAME, TOPIC, causes)
        self.assertIn("RESOLUTION = how the situation can be or was solved", captured["p"])
        self.assertEqual([f["direct"] for f in ranked], [True, False])
        self.assertIn("NOT causes", g.NOT_A_CAUSE)

    def test_an_over_long_draft_is_cut_but_never_replaced_by_a_longer_or_broken_one(self):
        long_draft = ts.SCRIPT_A.replace("Her own sister lies sick inside the granary and cannot be moved.",
                                         "Her own sister lies sick inside the granary and cannot be moved. " + "And so it went on. " * 40)
        short = ts.SCRIPT_A
        with mock.patch.object(g, "ask_ollama", lambda p, **k: short):
            self.assertLess(g.word_count(g.trim_to_length(GAME, TOPIC, long_draft, ts.HOOK_A, [])), g.word_count(long_draft))
        with mock.patch.object(g, "ask_ollama", lambda p, **k: "Here are my thoughts on the script."):
            self.assertEqual(g.trim_to_length(GAME, TOPIC, long_draft, ts.HOOK_A, []), long_draft)
        self.assertEqual(g.trim_to_length(GAME, TOPIC, ts.SCRIPT_A, ts.HOOK_A, []), ts.SCRIPT_A)   # already short enough

    def test_the_support_search_finds_the_passage_with_the_rare_word(self):
        g._LOCATE_CACHE.clear()
        passages = {f"E{n:03d}": f"[E{n:03d}]\nExcerpt: The granary fight with the healer, part {n}." for n in range(1, 10)}
        passages["E010"] = "[E010]\nExcerpt: Winning the fight gives the town fresh seed as loot."
        seen = {}
        with mock.patch.object(g, "ask_ollama", lambda p, **k: seen.setdefault("p", p) and "NONE"):
            g.locate_support("The granary fight wins loot", passages, GAME)
        self.assertIn("[E010]", seen["p"])

    def test_when_only_one_path_turns_up_the_other_is_searched_for_and_each_fact_keeps_its_path(self):
        first = ("FACT 1: If she trusts the elders, the granary burns. || QUOTE: the plague stops spreading || ID: E003 || KIND: choice || OPTION: trust the elders\n"
                 "FACT 2: If she trusts the elders, the town loses its seed. || QUOTE: the town has no seed for spring planting || ID: E003 || KIND: consequence || OPTION: trust the elders")
        second = "FACT 1: If she refuses, her sister stays inside. || QUOTE: cannot be moved || ID: E002 || KIND: consequence || OPTION: refuse the elders"
        prompts = []

        def fake(p, **k):
            prompts.append(p)
            return first if len(prompts) == 1 else second
        with mock.patch.object(g, "ask_ollama", fake):
            facts = g.extract_choice_facts(GAME, TOPIC, ts.EVIDENCE, IDS)
        self.assertEqual(sum("ONLY THE OTHER PATHS ARE MISSING" in p for p in prompts), 1)
        self.assertIn('ONLY THE OTHER PATHS ARE MISSING: the facts so far cover only "trust the elders"', prompts[1])
        self.assertEqual([f["option"] for f in facts], ["trust the elders", "trust the elders", "refuse the elders"])
        hook = "A healer must decide whether to burn the granary her town depends on. [E001]"
        with mock.patch.object(g, "ask_ollama", lambda p, **k: "TIMELINE: 1\nPAYOFF: 2, 3, 4"):
            plan = g.plan_story_beats(GAME, TOPIC, hook, [{"id": "E005", "fact": "The elders hid infected grain.", "quote": "x", "kind": "detail"}] + facts, [])
        self.assertIn("PAYOFF: THE CHOICE (2 different paths", plan)
        self.assertLess(plan.index("OPTION 1: trust the elders"), plan.index("OPTION 2: refuse the elders"))
        self.assertIn("never split one path into two options", plan)

    def test_two_paths_found_at_once_need_no_second_search(self):
        both = ("FACT 1: If she burns it, the plague stops. || QUOTE: the plague stops spreading || ID: E003 || KIND: consequence || OPTION: burn the granary\n"
                "FACT 2: If she waits, her sister stays inside. || QUOTE: cannot be moved || ID: E002 || KIND: consequence || OPTION: wait until dawn")
        calls = []
        with mock.patch.object(g, "ask_ollama", lambda p, **k: calls.append(p) or ("NONE" if "THE OUTCOME" in p else both)):
            self.assertEqual(len(g.extract_choice_facts(GAME, TOPIC, ts.EVIDENCE, IDS)), 2)
        self.assertFalse(any("ONLY THE OTHER PATHS ARE MISSING" in p for p in calls))
        self.assertEqual(sum("THE OUTCOME" in p for p in calls), 2)              # one outcome question per path

    def test_the_writer_and_judge_reject_a_vague_payoff(self):
        prompt = g.build_generation_prompt(GAME, TOPIC, ts.EVIDENCE, IDS, ts.HOOK_A, "brief")
        self.assertIn("THE PAYOFF MAKES EACH OPTION CONCRETE", prompt)
        self.assertIn("names a choice without saying concretely", g.build_script_evaluation_prompt(GAME, ts.SCRIPT_A))


class EndingQuestionTests(unittest.TestCase):
    def test_predictions_are_screened_out_but_real_dilemmas_are_not(self):
        for q in ("Will the trickery succeed in banishing the specter, or will the host be lost to madness?",
                  "Can you master the deception required to save a man?", "Will Mirela burn it in time?"):
            self.assertTrue(g.weak_ending_reason(q), q)
        for q in ("Would you burn the granary with your sister inside, or let the plague spread?",
                  "Can you forgive elders who hid the infected grain?", "Does a healer who lets her sister burn deserve to be thanked?"):
            self.assertEqual(g.weak_ending_reason(q), "", q)

    def test_several_questions_are_written_and_the_judge_picks_one_that_is_not_weak(self):
        prompts = []

        def fake(p, **k):
            prompts.append(p)
            if "closing questions" in p:
                return ("Q1: Will Mirela succeed?\nQ2: Would you burn your sister to save the town's seed?\n"
                        "Q3: Do elders who hid the grain deserve to be saved by her?")
            return "2"
        with mock.patch.object(g, "ask_ollama", fake):
            out = g.best_ending_question(GAME, TOPIC, ts.SCRIPT_A)
        self.assertIn("ENDING QUESTION: Do elders who hid the grain deserve to be saved by her?", out)
        self.assertIn("1. Would you burn your sister", prompts[1])            # the prediction never reached the judge
        self.assertNotIn("Will Mirela succeed", prompts[1])

    def test_an_unusable_reply_keeps_the_original_question(self):
        with mock.patch.object(g, "ask_ollama", lambda p, **k: "I think the ending is fine."):
            self.assertEqual(g.best_ending_question(GAME, TOPIC, ts.SCRIPT_A), ts.SCRIPT_A)


class CauseLinkTests(unittest.TestCase):
    def test_the_cause_search_asks_for_the_rule_that_explains_why_and_the_writer_must_connect_it(self):
        captured = {}
        with mock.patch.object(g, "ask_ollama", lambda p, **k: captured.setdefault("p", p) and ""):
            g.extract_cause_facts(GAME, TOPIC, ts.EVIDENCE, IDS)
        self.assertIn("the LINK that explains WHY", captured["p"])
        self.assertIn("THE WHY:", g.build_generation_prompt(GAME, TOPIC, ts.EVIDENCE, IDS, ts.HOOK_A, "brief"))

    def test_the_steps_of_one_path_keep_their_order(self):
        hook = "A healer must decide whether to burn the granary her town depends on. [E001]"
        steps = [{"id": "E003", "fact": f"Step {n} of burning it.", "quote": "x", "kind": "consequence", "payoff_detail": True,
                  "option": "burn it"} for n in (1, 2, 3)]
        other = {"id": "E002", "fact": "If she waits, her sister stays inside.", "quote": "x", "kind": "consequence",
                 "payoff_detail": True, "option": "wait"}
        base = {"id": "E005", "fact": "The elders hid infected grain.", "quote": "x", "kind": "detail"}
        with mock.patch.object(g, "ask_ollama", lambda p, **k: "TIMELINE: 1\nPAYOFF: 4, 2, 5, 3"):
            plan = g.plan_story_beats(GAME, TOPIC, hook, [base] + steps + [other], [])
        self.assertLess(plan.index("Step 1"), plan.index("Step 2"))
        self.assertLess(plan.index("Step 2"), plan.index("Step 3"))
        self.assertIn("ONE PATH, ONE SEQUENCE", g.build_generation_prompt(GAME, TOPIC, ts.EVIDENCE, IDS, ts.HOOK_A, "brief"))


class WhatHappenedAndWhyTests(unittest.TestCase):
    def test_the_required_points_are_the_event_plus_one_rule_never_only_rules(self):
        facts = [{"id": "E005", "fact": "Plagues strike towns because they spread through hidden grain.", "kind": "cause", "quote": "x"},
                 {"id": "E006", "fact": "A plague strikes wherever grain is hidden, because it feeds on rot.", "kind": "cause", "quote": "x"},
                 {"id": "E002", "fact": "The elders hid the infected grain from Mirela.", "kind": "detail", "quote": "x"},
                 {"id": "E009", "fact": "An old war brought plagues into the world.", "kind": "cause", "quote": "x"},
                 {"id": "E003", "fact": "If Mirela burns the grain the elders hid, the plague stops.", "kind": "consequence", "quote": "x",
                  "payoff_detail": True}]
        captured = {}
        reply = "1: RULE\n2: RULE\n3: EVENT\n4: BACKGROUND"
        with mock.patch.object(g, "ask_ollama", lambda p, **k: captured.setdefault("p", p) and reply):
            g.rank_cause_facts(GAME, TOPIC, facts)
        self.assertIn("3. The elders hid the infected grain", captured["p"])     # a plain detail can be the event
        self.assertNotIn("If Mirela burns", captured["p"].split("FACTS:")[1])     # how it ends is not judged as a cause...
        self.assertIn("THE STORY THIS VIDEO TELLS ends on this choice", captured["p"])   # ...it marks what the story is
        required = [f["id"] for f in g.essential_facts(facts)]
        self.assertEqual(required, ["E002", "E005"])                               # what happened, then ONE rule
        facts.insert(0, {"id": "E004", "fact": "The healer believes the elders hid the grain.", "kind": "account", "quote": "x",
                         "cause_role": "EVENT"})
        self.assertEqual([f["id"] for f in g.essential_facts(facts)], ["E002", "E004", "E005"])   # the happening before a belief


class FocusAndSpeedTests(unittest.TestCase):
    def test_events_from_another_storyline_are_marked_other(self):
        captured = {}
        with mock.patch.object(g, "ask_ollama", lambda p, **k: captured.setdefault("p", p) and ""):
            g.extract_cause_facts(GAME, TOPIC, ts.EVIDENCE, IDS)
        self.assertIn("other storylines, quests or incidents", captured["p"])
        facts = [{"id": "E005", "fact": "The elders hid infected grain.", "kind": "detail", "quote": "x"}]
        with mock.patch.object(g, "ask_ollama", lambda p, **k: captured.setdefault("r", p) and "1: EVENT"):
            g.rank_cause_facts(GAME, TOPIC, facts)
        self.assertIn("event from a different storyline", captured["r"])

    def test_the_narration_never_opens_on_a_rule(self):
        hook = "A healer must decide whether to burn the granary her town depends on. [E001]"
        rule = {"id": "E005", "fact": "Plagues feed on hidden grain.", "quote": "x", "kind": "cause", "cause_role": "RULE"}
        event = {"id": "E002", "fact": "The elders hid the grain from Mirela.", "quote": "x", "kind": "detail", "cause_role": "EVENT"}
        with mock.patch.object(g, "ask_ollama", lambda p, **k: "TIMELINE: 1, 2\nPAYOFF: none"):
            plan = g.plan_story_beats(GAME, TOPIC, hook, [rule, event], [])
        self.assertLess(plan.index("1. [E002]"), plan.index("2. [E005]"))
        self.assertIn("who is fooled into believing what", plan)

    def test_retries_write_one_draft_and_a_sound_story_stops_the_retries(self):
        self.assertEqual(g.SCRIPT_EXTRA_ATTEMPTS, 2)
        self.assertIn(f"AT MOST {g.SCRIPT_TARGET_MAX_WORDS - 10} spoken words", g.full_story_block("x"))


class StoryScopeTests(unittest.TestCase):
    def test_an_event_from_another_quest_is_never_required_even_if_the_model_says_so(self):
        facts = [{"id": "E009", "fact": "A massacre occurred during the banquet because someone orchestrated it.", "kind": "cause", "quote": "x"},
                 {"id": "E002", "fact": "The elders hid the infected grain from Mirela.", "kind": "detail", "quote": "x"},
                 {"id": "E005", "fact": "Plagues spread because they feed on hidden grain.", "kind": "cause", "quote": "x"},
                 {"id": "E003", "fact": "If Mirela burns the granary, the plague stops.", "kind": "consequence", "quote": "x", "payoff_detail": True}]
        with mock.patch.object(g, "ask_ollama", lambda p, **k: "1: EVENT\n2: EVENT\n3: RULE"):
            g.rank_cause_facts(GAME, TOPIC, facts)
        self.assertEqual(facts[0]["cause_role"], "OTHER")
        self.assertEqual([f["id"] for f in g.essential_facts(facts)], ["E002", "E005"])

    def test_joining_two_events_and_saying_how_people_are_related_are_claims(self):
        captured = {}
        fake = lambda p, **k: captured.setdefault("p", p) and "CLAIM 1: x || TYPE: FACT || VERDICT: SUPPORTED || QUOTE: NONE"
        with mock.patch.object(g, "ask_ollama", fake):
            g.audit_line("NARRATION", "In the chaos of the feast, her sister fell ill. [E002]", PASSAGES, GAME)
        self.assertIn("JOINING EVENTS IS A CLAIM", captured["p"])
        self.assertIn("how people are related", captured["p"])


class OutcomeAndEventTests(unittest.TestCase):
    def test_each_path_gets_its_outcome_and_a_passing_reaction_is_not_one(self):
        first = ("FACT 1: If she burns it, the elders shout at her. || QUOTE: the plague stops spreading || ID: E003 || KIND: consequence || OPTION: burn it\n"
                 "FACT 2: If she waits, her sister stays inside. || QUOTE: cannot be moved || ID: E002 || KIND: consequence || OPTION: wait")
        outcome = "FACT 1: If she burns it, the plague stops but the town has no seed. || QUOTE: the town has no seed for spring planting || ID: E003 || KIND: consequence || OPTION: x"
        prompts = []
        with mock.patch.object(g, "ask_ollama", lambda p, **k: prompts.append(p) or (outcome if 'option "burn it"' in p else "NONE" if "THE OUTCOME" in p else first)):
            facts = g.extract_choice_facts(GAME, TOPIC, ts.EVIDENCE, IDS)
        self.assertIn("passing reactions (a punch, an insult, a remark) are not story consequences", prompts[0])
        burn = [f["fact"] for f in facts if f["option"] == "burn it"]
        self.assertIn("If she burns it, the plague stops but the town has no seed.", burn)
        self.assertLess(burn.index("If she burns it, the plague stops but the town has no seed."), burn.index("If she burns it, the elders shout at her."))

    def test_a_passing_reaction_the_model_marks_minor_is_dropped_and_the_risk_is_asked_for(self):
        first = ("FACT 1: If she refuses, she must fight the elders. || QUOTE: cannot be moved || ID: E002 || KIND: choice || OPTION: refuse\n"
                 "FACT 2: If she refuses, an elder slaps her. || QUOTE: the plague stops spreading || ID: E003 || KIND: consequence || OPTION: refuse\n"
                 "FACT 3: If she accepts, the granary burns. || QUOTE: the town has no seed for spring planting || ID: E003 || KIND: choice || OPTION: accept")
        prompts = []
        with mock.patch.object(g, "ask_ollama", lambda p, **k: prompts.append(p) or ("MINOR: 2" if 'option "refuse"' in p else "MINOR: NONE" if "THE OUTCOME" in p else first)):
            facts = g.extract_choice_facts(GAME, TOPIC, ts.EVIDENCE, IDS)
        self.assertNotIn("If she refuses, an elder slaps her.", [f["fact"] for f in facts])
        self.assertIn("If she refuses, she must fight the elders.", [f["fact"] for f in facts])   # what is done always stays
        self.assertIn("THE RISK:", next(p for p in prompts if 'option "refuse"' in p))

    def test_outcome_lines_labelled_by_what_they_answer_are_kept_and_minor_by_passage_id_is_dropped(self):
        first = ("FACT 1: If she refuses, she fights the elders. || QUOTE: cannot be moved || ID: E002 || KIND: choice || OPTION: refuse\n"
                 "FACT 2: If she refuses, an elder slaps her. || QUOTE: the plague stops spreading || ID: E003 || KIND: consequence || OPTION: refuse\n"
                 "FACT 3: If she accepts, the granary burns. || QUOTE: the town has no seed for spring planting || ID: E001 || KIND: choice || OPTION: accept")
        outcome = ("THE RISK: If she refuses, her sister may not survive. || QUOTE: Mirela's sister lies sick || ID: E002 || KIND: risk || OPTION: refuse\n"
                   "THE TWIST: NONE\nMINOR: E003")
        with mock.patch.object(g, "ask_ollama", lambda p, **k: outcome if 'option "refuse"' in p else "MINOR: NONE" if "THE OUTCOME" in p else first):
            refuse = [f["fact"] for f in g.extract_choice_facts(GAME, TOPIC, ts.EVIDENCE, IDS) if f["option"] == "refuse"]
        self.assertEqual(refuse, ["If she refuses, she fights the elders.", "If she refuses, her sister may not survive."])

    def test_a_path_whose_facts_are_all_marked_minor_does_not_crash(self):
        first = ("FACT 1: If she refuses, an elder slaps her. || QUOTE: the plague stops spreading || ID: E003 || KIND: consequence || OPTION: refuse\n"
                 "FACT 2: If she accepts, the granary burns. || QUOTE: the town has no seed for spring planting || ID: E001 || KIND: choice || OPTION: accept")
        with mock.patch.object(g, "ask_ollama", lambda p, **k: "MINOR: 1" if "THE OUTCOME" in p else first):
            facts = g.extract_choice_facts(GAME, TOPIC, ts.EVIDENCE, IDS)
        self.assertNotIn("If she refuses, an elder slaps her.", [f["fact"] for f in facts])

    def test_opposite_paths_are_never_merged(self):
        self.assertNotEqual(g._option_key("Trust Cerys"), g._option_key("Do not trust Cerys"))
        self.assertNotEqual(g._option_key("trust her"), g._option_key("don't trust her"))
        self.assertEqual(g._option_key("Trust the healer"), g._option_key("trust healer"))

    def test_a_story_with_no_event_gets_one_from_a_focused_search(self):
        reply = "FACT: The elders hid the infected grain from Mirela. || QUOTE: stored infected grain without telling the healer || ID: E005"
        with mock.patch.object(g, "ask_ollama", lambda p, **k: reply):
            event = g.find_story_event(GAME, TOPIC, ts.EVIDENCE, IDS, "- If Mirela burns it, the plague stops.")
        self.assertEqual((event["id"], event["cause_role"]), ("E005", "EVENT"))
        with mock.patch.object(g, "ask_ollama", lambda p, **k: "FACT: Made up. || QUOTE: words not in any passage || ID: E005"):
            self.assertIsNone(g.find_story_event(GAME, TOPIC, ts.EVIDENCE, IDS, "x"))

    def test_a_method_of_defeating_the_threat_is_resolution_and_facts_about_the_topic_stay_in_scope(self):
        captured = {}
        facts = [{"id": "E005", "fact": "A trickster must do an evil deed because the plague moves to them.", "kind": "cause", "quote": "x"},
                 {"id": "E002", "fact": "A plague has struck the granary town.", "kind": "detail", "quote": "x"},
                 {"id": "E003", "fact": "If Mirela burns it, the town is saved.", "kind": "consequence", "quote": "x", "payoff_detail": True}]
        with mock.patch.object(g, "ask_ollama", lambda p, **k: captured.setdefault("p", p) and "1: RESOLUTION\n2: EVENT"):
            g.rank_cause_facts(GAME, "The plague", facts)
        self.assertIn("What someone must do to defeat, trick, banish or end it is never a RULE", captured["p"])
        self.assertEqual(facts[1]["cause_role"], "EVENT")                       # it names the topic, so it stays in


class EndingVersusCauseTests(unittest.TestCase):
    def test_an_event_that_is_really_the_ending_is_resolution_and_the_real_event_is_searched_for(self):
        facts = [{"id": "E003", "fact": "The plague left the town because Mirela burned the granary.", "kind": "cause", "quote": "x"},
                 {"id": "E005", "fact": "Plagues feed on hidden grain.", "kind": "cause", "quote": "x"},
                 {"id": "E001", "fact": "If Mirela burns the granary, the plague stops.", "kind": "consequence", "quote": "x", "payoff_detail": True}]

        def fake(p, **k):
            if "answer BEFORE" in p:
                return "1: ENDING"
            return "1: EVENT\n2: RULE"
        with mock.patch.object(g, "ask_ollama", fake):
            g.rank_cause_facts(GAME, TOPIC, facts)
        self.assertEqual(facts[0]["cause_role"], "RESOLUTION")
        self.assertFalse(any(f.get("cause_role") == "EVENT" for f in g.essential_facts(facts)))   # so the event search runs

    def test_facts_about_one_path_are_not_duplicates_of_each_other_or_of_another_path(self):
        a = {"id": "E003", "fact": "If she burns it, she seals her sister inside.", "payoff_detail": True, "option": "burn it", "kind": "choice"}
        b = {"id": "E003", "fact": "If she burns it, the plague stops for good.", "payoff_detail": True, "option": "burn it", "kind": "consequence"}
        c = {"id": "E003", "fact": "If she waits, the plague stops for good.", "payoff_detail": True, "option": "wait", "kind": "consequence"}
        self.assertEqual(len(g.dedupe_facts([a, b, c])), 3)

    def test_an_outcome_line_without_a_quote_is_kept_when_code_finds_its_passage(self):
        first = ("FACT 1: If she burns it, the granary burns. || QUOTE: the town has no seed for spring planting || ID: E001 || KIND: choice || OPTION: burn it\n"
                 "FACT 2: If she waits, her sister stays inside. || QUOTE: cannot be moved || ID: E002 || KIND: consequence || OPTION: wait")
        def fake(p, **k):
            if 'option "burn it"' in p:
                return "THE OUTCOME: The plague stops spreading.\nTHE TWIST: NONE\nMINOR: NONE"
            if "Does any passage below state this claim?" in p:
                return "ID: E003 || VERDICT: SUPPORTED || QUOTE: the plague stops spreading"
            return "MINOR: NONE" if "THE OUTCOME" in p else first
        g._LOCATE_CACHE.clear()
        with mock.patch.object(g, "ask_ollama", fake):
            burn = [f for f in g.extract_choice_facts(GAME, TOPIC, ts.EVIDENCE, IDS) if f["option"] == "burn it"]
        self.assertIn(("The plague stops spreading.", "E003"), [(f["fact"], f["id"]) for f in burn])


class RevealLastTests(unittest.TestCase):
    def test_the_path_with_a_reveal_is_told_last_and_ends_on_it(self):
        hook = "A healer must decide whether to burn the granary her town depends on. [E001]"
        base = {"id": "E005", "fact": "The elders hid infected grain.", "quote": "x", "kind": "detail", "cause_role": "EVENT"}
        trick = [{"id": "E003", "fact": "The granary was empty all along.", "quote": "x", "kind": "consequence", "payoff_detail": True, "option": "burn it", "beat": "twist"},
                 {"id": "E003", "fact": "If she burns it, she sets the fire herself.", "quote": "x", "kind": "choice", "payoff_detail": True, "option": "burn it"},
                 {"id": "E003", "fact": "The plague ends.", "quote": "x", "kind": "consequence", "payoff_detail": True, "option": "burn it", "beat": "outcome"}]
        wait = [{"id": "E002", "fact": "If she waits, her sister may die.", "quote": "x", "kind": "consequence", "payoff_detail": True, "option": "wait", "beat": "risk"},
                {"id": "E002", "fact": "If she waits, the fire takes all night.", "quote": "x", "kind": "consequence", "payoff_detail": True, "option": "wait"}]
        with mock.patch.object(g, "ask_ollama", lambda p, **k: "TIMELINE: 1\nPAYOFF: 2, 3, 4, 5, 6"):
            plan = g.plan_story_beats(GAME, TOPIC, hook, [base] + trick + wait, [])
        self.assertLess(plan.index("OPTION 1: wait"), plan.index("OPTION 2: burn it"))          # reveal path last
        self.assertLess(plan.index("COST: [E002] If she waits, her sister may die"), plan.index("the fire takes all night"))
        burn = plan[plan.index("OPTION 2: burn it"):]
        self.assertLess(burn.index("sets the fire herself"), burn.index("REVEAL (save for the very end): [E003] The granary was empty"))
        self.assertLess(burn.index("The granary was empty"), burn.index("The plague ends"))
        self.assertIn("end the PAYOFF on the last path's reveal", plan)

    def test_the_writer_keeps_one_point_of_view_and_real_costs(self):
        prompt = g.build_generation_prompt(GAME, TOPIC, ts.EVIDENCE, IDS, ts.HOOK_A, "brief")
        for rule in ("SAVE THE REVEAL FOR THE END", "ONE POINT OF VIEW", "A PATH'S COST is what is at stake for people"):
            self.assertIn(rule, prompt)

    def test_the_closing_question_is_built_from_the_paths_and_their_costs(self):
        captured = {}
        paths = {"burn it": [{"fact": "If she burns it, her sister burns with it."}], "wait": [{"fact": "If she waits, the plague spreads."}]}
        with mock.patch.object(g, "ask_ollama", lambda p, **k: captured.setdefault("p", p) and "nothing useful"):
            g.best_ending_question(GAME, TOPIC, ts.SCRIPT_A, paths)
        self.assertIn("THE PATHS AND WHAT EACH COSTS", captured["p"])
        self.assertIn("- burn it: If she burns it, her sister burns with it.", captured["p"])


class TellingCraftTests(unittest.TestCase):
    """Problems a newcomer feels on first listen, caught in code for any game."""
    def lines(self, narration, payoff="The plague ends. [E001]", ending="Would you burn it? [E001]"):
        body = "\n".join(f"NARRATION: {n} [E001]" for n in narration)
        return g.spoken_lines(f"STATUS: DRAFT\nHOOK: A healer must burn her town's granary. [E001]\n{body}\n"
                              f"PAYOFF: {payoff}\nENDING QUESTION: {ending}\nCLAIMS REQUIRING MORE EVIDENCE: None")

    def test_a_person_named_with_no_role_is_flagged_and_an_introduced_one_is_not(self):
        problems = g.telling_problems(self.lines(["If you do not trust Orla, you burn it yourself.", "Then Orla leaves."]), GAME)
        self.assertTrue(any("Say who Orla is" in p for p in problems))
        for intro in ("Her sister Orla knows the cure.", "Orla, the miller's widow, knows the cure.", "The cure is known to a woman named Orla."):
            problems = g.telling_problems(self.lines([intro, "Then Orla leaves."]), GAME)
            self.assertFalse(any("Orla" in p for p in problems), intro)

    def test_a_new_name_in_the_payoff_is_flagged(self):
        problems = g.telling_problems(self.lines(["Her sister lies sick inside."], payoff="But Brand was waiting on the other side."), GAME)
        self.assertTrue(any("PAYOFF brings in a new name, Brand" in p for p in problems))

    def test_a_line_that_only_announces_the_choice_is_flagged(self):
        problems = g.telling_problems(self.lines(["Her sister lies sick inside.", "You must decide how to end it."]), GAME)
        self.assertTrue(any("only announces the choice" in p for p in problems))

    def test_long_sentences_and_formal_words_are_flagged_but_plain_compounds_are_not(self):
        long_payoff = ("Believing she has burned her own sister alive, the healer sinks into despair until the truth surfaces "
                       "that her brother carried the girl out through the back door long before the fire.")
        problems = g.telling_problems(self.lines(["The despicable individuals hid the plague-ridden grain."], payoff=long_payoff), GAME)
        self.assertTrue(any("-word sentence" in p for p in problems))
        plain = next(p for p in problems if p.startswith("Plain words"))
        self.assertIn("despicable", plain)
        self.assertNotIn("plague", plain)

    def test_style_notes_never_block(self):
        problems = g.telling_problems(self.lines(["You must decide.", "Then Orla leaves."], payoff="Brand waits."), GAME)
        self.assertTrue(problems)
        self.assertFalse(any(g.is_structural(p) for p in problems))


class BalancedEndingTests(unittest.TestCase):
    PATHS = {"do not trust the miller": [{"fact": "If the player does not trust the miller, the player burns the granary with the sister nearby.", "kind": "choice"},
                                         {"fact": "The sister may not withstand the smoke.", "beat": "risk"}],
             "trust the miller": [{"fact": "The player hides the grain in the well.", "kind": "choice"},
                                  {"fact": "The grain was never infected.", "beat": "twist"}]}

    def test_a_question_that_drops_a_paths_cost_is_lopsided(self):
        q = "Would you trust the miller and hide the grain in the well, or burn the granary with the sister nearby?"
        self.assertIn("leaves out what that path risks", g.ending_balance_problem(q, self.PATHS, GAME))

    def test_a_question_that_weighs_both_paths_passes_in_any_words(self):
        q = "Not knowing the truth, would you hide the grain in the well, or burn the granary and risk your sister?"
        self.assertEqual(g.ending_balance_problem(q, self.PATHS, GAME), "")

    def test_lopsided_candidates_are_dropped_when_a_balanced_one_exists(self):
        replies = iter(["Q1: Would you trust the miller and hide the grain in the well, or burn the granary with the sister nearby?\n"
                        "Q2: Would you hide the grain in the well, or burn the granary and risk your sister?", "1"])
        with mock.patch.object(g, "ask_ollama", lambda p, **k: next(replies)):
            out = g.best_ending_question(GAME, TOPIC, ts.SCRIPT_A, self.PATHS)
        self.assertIn("ENDING QUESTION: Would you hide the grain in the well, or burn the granary and risk your sister?", out)

    def test_the_question_is_asked_from_the_moment_of_choosing(self):
        captured = {}
        with mock.patch.object(g, "ask_ollama", lambda p, **k: captured.setdefault("p", p) and "nothing useful"):
            g.best_ending_question(GAME, TOPIC, ts.SCRIPT_A, self.PATHS)
        self.assertIn("MOMENT OF CHOOSING", captured["p"])
        self.assertIn("never give one side a cost and the other side none", captured["p"])


class RevealSetupTests(unittest.TestCase):
    def test_the_rule_a_twist_uses_is_planted_before_the_choice(self):
        hook = "A healer must decide whether to burn the granary her town depends on. [E001]"
        rule = {"id": "E004", "fact": "The blight only spreads to grain that someone believes is cursed.", "quote": "x", "kind": "detail", "cause_role": "RULE"}
        event = {"id": "E005", "fact": "The elders hid infected grain.", "quote": "x", "kind": "detail", "cause_role": "EVENT"}
        trick = [{"id": "E003", "fact": "If she burns it, she sets the fire herself.", "quote": "x", "kind": "choice", "payoff_detail": True, "option": "burn it"},
                 {"id": "E003", "fact": "The granary was empty all along.", "quote": "x", "kind": "consequence", "payoff_detail": True, "option": "burn it", "beat": "twist"}]
        wait = [{"id": "E002", "fact": "If she waits, her sister may die.", "quote": "x", "kind": "consequence", "payoff_detail": True, "option": "wait", "beat": "risk"}]
        with mock.patch.object(g, "ask_ollama", lambda p, **k: "TIMELINE: 2, 1\nPAYOFF: 3, 4, 5"):
            plan = g.plan_story_beats(GAME, TOPIC, hook, [rule, event] + trick + wait, [rule])
        self.assertIn('SET UP THE REVEAL: the rule "The blight only spreads to grain that someone believes is cursed."', plan)

    def test_no_setup_line_without_a_twist(self):
        hook = "A healer must decide whether to burn the granary her town depends on. [E001]"
        rule = {"id": "E004", "fact": "The blight spreads to stored grain.", "quote": "x", "kind": "detail", "cause_role": "RULE"}
        event = {"id": "E005", "fact": "The elders hid infected grain.", "quote": "x", "kind": "detail", "cause_role": "EVENT"}
        wait = [{"id": "E002", "fact": "If she waits, her sister may die.", "quote": "x", "kind": "consequence", "payoff_detail": True, "option": "wait", "beat": "risk"},
                {"id": "E003", "fact": "If she burns it, the town starves.", "quote": "x", "kind": "consequence", "payoff_detail": True, "option": "burn it", "beat": "risk"}]
        with mock.patch.object(g, "ask_ollama", lambda p, **k: "TIMELINE: 2, 1\nPAYOFF: 3, 4"):
            plan = g.plan_story_beats(GAME, TOPIC, hook, [rule, event] + wait, [rule])
        self.assertNotIn("SET UP THE REVEAL", plan)

    def test_the_writer_bridges_the_rule_and_introduces_people(self):
        prompt = g.build_generation_prompt(GAME, TOPIC, ts.EVIDENCE, IDS, ts.HOOK_A, "brief")
        for rule in ("THE RULE MUST FIT THE PERSON", "PLANT WHAT THE TWIST NEEDS", "WHO THEY ARE", "Never bring a new name into the PAYOFF"):
            self.assertIn(rule, prompt)


class HostedWriterTests(unittest.TestCase):
    """Writing goes to the best free writer that answers; a busy one hands the step on at once; nothing ever breaks."""
    class Resp:
        def __init__(self, status, body, headers=None):
            self.status_code, self._body, self.headers = status, body, headers or {}
            self.text = json.dumps(body)
        def json(self):
            return self._body

    def setUp(self):
        g._WRITERS.update(last={}, down={}, used=0, fallbacks=0)
        self.patches = [mock.patch.dict(os.environ, {"GEMINI_API_KEY": "gem-key", "GROQ_API_KEY": "groq-key"}),
                        mock.patch.object(g, "USE_ONLINE_WRITERS", True),
                        mock.patch.dict(g.WRITER_PACING, {"gemini": 0, "groq": 0}), mock.patch.object(g.time, "sleep", self.slept)]
        self.sleeps = []
        for p in self.patches:
            p.start()
        self.local = []

    def slept(self, seconds):
        self.sleeps.append(seconds)

    def tearDown(self):
        for p in self.patches:
            p.stop()
        g._WRITERS.update(last={}, down={}, used=0, fallbacks=0)

    def gem(self, text):
        return self.Resp(200, {"candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True}, {"text": text}]}}]})

    def groq(self, text):
        return self.Resp(200, {"choices": [{"message": {"content": text, "reasoning": "hidden"}}]})

    busy = staticmethod(lambda: HostedWriterTests.Resp(503, {"error": {"message": "This model is currently experiencing high demand."}}))

    def run_writer(self, replies, calls=1):
        sent = []
        def post(url, json=None, timeout=None, headers=None):
            sent.append((url, json, headers))
            return replies.pop(0)
        def local(prompt, temperature=0.2, model=None, options=None):
            self.local.append(prompt)
            return "local text"
        with mock.patch.object(g.requests, "post", post), mock.patch.object(g, "ask_ollama", local):
            out = [g.ask_writer("write it", temperature=0.7) for _ in range(calls)]
        return out, [("groq" if "groq" in u else u.split("/models/")[1].split(":")[0]) for u, *_ in sent], sent

    def test_online_writers_are_off_unless_switched_on_even_with_keys_saved(self):
        with mock.patch.object(g, "USE_ONLINE_WRITERS", False):
            out, order, _ = self.run_writer([])
        self.assertEqual((out, order), (["local text"], []))

    def test_no_keys_means_the_local_model_writes(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "", "GROQ_API_KEY": ""}):
            out, order, _ = self.run_writer([])
        self.assertEqual((out, order), (["local text"], []))

    def test_gemini_writes_first_and_its_thoughts_are_dropped(self):
        out, order, sent = self.run_writer([self.gem("A story.")])
        self.assertEqual((out, order), (["A story."], ["gemini-3.8-flash"]))
        url, payload, headers = sent[0]
        self.assertEqual(headers["x-goog-api-key"], "gem-key")
        self.assertNotIn("gem-key", url)                                     # the key never goes into the URL
        self.assertNotIn("temperature", json.dumps(payload))

    def test_a_busy_gemini_hands_the_step_to_groq_at_once_with_no_waiting(self):
        out, order, sent = self.run_writer([self.busy(), self.groq("Groq wrote it.")])
        self.assertEqual((out, order), (["Groq wrote it."], ["gemini-3.8-flash", "groq"]))
        self.assertEqual(sent[1][1]["model"], "openai/gpt-oss-120b")
        self.assertEqual(sent[1][2]["Authorization"], "Bearer groq-key")
        self.assertEqual(self.sleeps, [])

    def test_when_groq_is_busy_too_the_other_gemini_models_are_tried(self):
        out, order, _ = self.run_writer([self.busy(), self.busy(), self.gem("Backup.")])
        self.assertEqual((out, order), (["Backup."], ["gemini-3.8-flash", "groq", "gemini-3.6-flash"]))

    def test_when_every_writer_is_busy_twice_the_local_model_writes(self):
        out, order, _ = self.run_writer([self.busy() for _ in range(8)])
        self.assertEqual(out, ["local text"])
        self.assertEqual(len(order), 8)                                      # four writers, two rounds
        self.assertEqual(self.sleeps, [15])

    def test_a_short_per_minute_limit_is_waited_out(self):
        tpm = self.Resp(429, {"error": {"message": "Rate limit reached on tokens per minute (TPM). Please try again in 3.2s"}}, {"retry-after": "4"})
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            out, order, _ = self.run_writer([tpm, self.groq("After the wait.")])
        self.assertEqual((out, order), (["After the wait."], ["groq", "groq"]))
        self.assertEqual(self.sleeps, [4.5])

    def test_a_daily_limit_skips_that_writer_for_the_rest_of_the_run(self):
        day = self.Resp(429, {"error": {"message": "quota GenerateRequestsPerDayPerProjectPerModel-FreeTier"}})
        out, order, _ = self.run_writer([day, self.groq("one"), self.groq("two")], calls=2)
        self.assertEqual(out, ["one", "two"])
        self.assertEqual(order, ["gemini-3.8-flash", "groq", "groq"])        # 3.8 is not asked again

    def test_a_wrong_key_skips_every_model_of_that_service(self):
        bad = self.Resp(400, {"error": {"message": "API key not valid. Please pass a valid API key."}})
        out, order, _ = self.run_writer([bad, self.busy(), self.busy()], calls=1)
        self.assertEqual(out, ["local text"])
        self.assertEqual(order, ["gemini-3.8-flash", "groq", "groq"])       # no other Gemini model is tried with a bad key

    def test_an_empty_reply_moves_that_step_on(self):
        out, order, _ = self.run_writer([self.Resp(200, {"candidates": [{"finishReason": "SAFETY"}]}), self.groq("Next.")])
        self.assertEqual(out, ["Next."])

    def test_the_writing_steps_use_the_writer_and_the_fact_check_does_not(self):
        import inspect
        for name in ("plan_story_beats", "best_ending_question", "trim_to_length", "revise_for_flow", "produce_script",
                     "generate_and_select_hooks", "plain_rewrites", "repair_line"):
            self.assertIn("ask_writer(", inspect.getsource(getattr(g, name)), name)
        for name in ("audit_line", "quote_says", "locate_support", "missing_essential_semantic", "rank_cause_facts", "extract_choice_facts"):
            self.assertNotIn("ask_writer(", inspect.getsource(getattr(g, name)), name)

    def test_keys_can_come_from_the_app_files(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"GEMINI_API_KEY": "", "GROQ_API_KEY": ""}):
            gem, groq = Path(tmp) / "g.txt", Path(tmp) / "q.txt"
            gem.write_text("  file-key\n", encoding="utf-8")
            groq.write_text("gsk_file\n", encoding="utf-8")
            with mock.patch.object(g, "GEMINI_KEY_FILE", gem), mock.patch.object(g, "GROQ_KEY_FILE", groq):
                self.assertEqual((g.gemini_key(), g.groq_key()), ("file-key", "gsk_file"))
                self.assertIn("Groq gpt-oss-120b", g.writer_name())


class WhoIsWhoTests(unittest.TestCase):
    """The writer is told who each named person is, in the evidence's own words, so no one appears from nowhere."""
    PASSAGES = {"E001": "Excerpt: According to Orla, the brothers fought over a sword. It features Orla Venn, daughter of the "
                        "Lord of Ashfall, who is on a quest to save Lord Tomas from a curse.",
                "E002": "Excerpt: Tomas was talking with his steward Brand about his dreams. A wraith haunts the keep.",
                "E003": "Excerpt: The fire spread to the granary where Mira was sleeping."}
    PLAN = ("STORY TIMELINE:\n1. [E001] Tomas lost his brother.\nPAYOFF:\nOPTION 1: Trust Orla\n"
            "  - [E002] If the player trusts Orla, Kael seals the door.\n  - [E002] Brand waits behind it. A Wraith screams.\n"
            "  - [E003] Mira sleeps.")

    def test_each_person_gets_the_evidence_words_and_the_player_character_is_you(self):
        block = g.cast_block(GAME, self.PLAN, self.PASSAGES)
        self.assertIn("- Orla: daughter of the Lord of Ashfall", block)          # not "the brothers fought" (According to Orla,)
        self.assertIn("- Tomas: Lord", block)
        self.assertIn("- Brand: the evidence does not say who they are", block)  # "his steward": whose is a guess, never used
        self.assertIn('Kael is the PLAYER\'S CHARACTER: when Kael acts on a path, say "you"', block)
        self.assertIn("- Mira: the evidence does not say who they are", block)
        self.assertNotIn("Trust", block.split("\n", 1)[1])                       # an option label is not a person
        self.assertNotIn("Wraith", block)                                        # written in lower case in the evidence

    def test_his_daughter_about_someone_outside_the_story_is_not_used(self):
        passages = {"E001": "Excerpt: The quest begins after Crach an Craite mentions his daughter, Orla, went to save Tomas.",
                    "E002": "Excerpt: Tomas was cursed."}
        block = g.cast_block(GAME, "STORY TIMELINE:\n1. Tomas was cursed.\n2. Orla has a plan.", passages)
        self.assertNotIn("Orla: his daughter", block)
        self.assertIn("- Orla: the evidence does not say who they are", block)

    def test_the_cast_reaches_the_writer_with_the_plan(self):
        fake = ts.ScriptFake({ts.HOOK_A: [ts.SCRIPT_A]})
        facts = [{"id": "E002", "fact": "If the player trusts Mirela, Kael burns the granary.", "quote": "x", "kind": "choice",
                  "payoff_detail": True, "option": "trust Mirela"}, {"id": "E001", "fact": "The elders hid the grain.", "quote": "x",
                  "kind": "detail", "cause_role": "EVENT"}]
        with mock.patch.object(g, "ask_ollama", fake), mock.patch.object(g, "plan_story_beats", lambda *a: "STORY TIMELINE:\n" + facts[0]["fact"]), \
                mock.patch.object(g, "DRAFTS_PER_ATTEMPT", 1), mock.patch.object(g, "FULL_STORY_FIRST", False):
            g.produce_script(GAME, TOPIC, ts.EVIDENCE, IDS, ts.PASSAGES, ts.HOOK_A, "brief", {}, 1, [], "", facts)
        self.assertIn("WHO IS WHO", fake.script_prompts[0])
        self.assertIn("Kael is the PLAYER'S CHARACTER", fake.script_prompts[0])

    def test_two_payoff_lines_are_joined_instead_of_failing_the_script(self):
        draft = ("STATUS: DRAFT\nHOOK: h [E001]\nNARRATION: n [E001]\nNARRATION: m [E001]\nPAYOFF: Burn it and the town starves. [E002]\n"
                 "PAYOFF: Wait, and your sister dies. [E003]\nENDING QUESTION: Which? [E001]\nCLAIMS REQUIRING MORE EVIDENCE: None")
        merged = g.merge_payoffs(draft)
        self.assertEqual([s for s, _ in g.spoken_lines(merged)], ["HOOK", "NARRATION", "NARRATION", "PAYOFF", "ENDING QUESTION"])
        self.assertIn("PAYOFF: Burn it and the town starves. Wait, and your sister dies.", merged)
        self.assertEqual(g.merge_payoffs(ts.SCRIPT_A), ts.SCRIPT_A)


class LineByLineWriterTests(unittest.TestCase):
    """WRITER_MODE=lines: code builds the shape from the plan, the model writes one line at a time."""
    def structure(self):
        opener = {"id": "E001", "fact": "A wraith drives Lord Tomas toward madness.", "quote": "x", "cause_role": "EVENT"}
        cause = {"id": "E002", "fact": "Tomas's brother drowned while Tomas held the rope.", "quote": "x", "cause_role": "EVENT"}
        rule = {"id": "E003", "fact": "Wraiths feed on those who believe they did something unforgivable.", "quote": "x", "cause_role": "RULE"}
        fight = [{"id": "E004", "fact": "If the player refuses Orla, Kael fights the wraith with Tomas present.", "option": "refuse Orla", "kind": "choice"},
                 {"id": "E004", "fact": "Tomas may not survive the fight.", "option": "refuse Orla", "beat": "risk"}]
        trick = [{"id": "E005", "fact": "If the player trusts Orla, Kael locks the child in the cellar.", "option": "trust Orla", "kind": "choice"},
                 {"id": "E005", "fact": "The wraith leaves Tomas for Kael, who believes he killed the child.", "option": "trust Orla"},
                 {"id": "E005", "fact": "Orla's man carried the child out through a second door.", "option": "trust Orla", "beat": "twist"}]
        return {"timeline": [opener, cause, rule], "paths": [("refuse Orla", fight), ("trust Orla", trick)], "loose": [], "rule": rule}

    def test_the_shape_puts_every_path_and_cost_in_order_and_ends_on_the_reveal(self):
        lines = g.script_skeleton(GAME, self.structure(), "Kael")
        self.assertEqual([l["section"] for l in lines], ["NARRATION", "NARRATION", "NARRATION", "NARRATION", "NARRATION", "PAYOFF"])
        self.assertIn("Name the game", lines[0]["job"])
        self.assertIn("The rule", lines[2]["job"])
        self.assertIn("Tomas may not survive the fight.", [f["fact"] for f in lines[3]["facts"]])   # the cost travels with its path
        self.assertIn("Do not reveal", lines[4]["job"])
        self.assertEqual(lines[-1]["facts"][-1]["beat"], "twist")                                   # the payoff ends on the reveal

    def test_each_line_is_written_in_order_and_the_reveal_is_held_back(self):
        prompts = []
        def writer(prompt, temperature=0.2, options=None):
            prompts.append(prompt)
            return f"Line {len(prompts)}."
        with mock.patch.object(g, "ask_writer", writer):
            draft = g.write_by_lines(GAME, TOPIC, "A lord claws at his own face. [E001]", self.structure(), "")
        sections = [s for s, _ in g.spoken_lines(draft)]
        self.assertEqual(sections, ["HOOK", "NARRATION", "NARRATION", "NARRATION", "NARRATION", "NARRATION", "PAYOFF", "ENDING QUESTION"])
        self.assertIn("NARRATION: Line 1.", prompts[1])                                   # each line sees the ones before it
        self.assertIn("KEEP FOR LATER", prompts[3])
        self.assertIn("second door", prompts[3])                                         # ...and is told not to give the twist away
        self.assertNotIn("KEEP FOR LATER", prompts[-1])
        self.assertLess(max(len(p) for p in prompts), 4000)                             # small jobs, short prompts

    def test_the_players_name_is_sent_back_for_you(self):
        replies = iter(["Kael locks the child in the cellar.", "You lock the child in the cellar."] + ["A line."] * 10)
        with mock.patch.object(g, "ask_writer", lambda p, **k: next(replies)):
            draft = g.write_by_lines(GAME, TOPIC, "A lord claws at his own face. [E001]", self.structure(), "")
        self.assertNotIn("Kael", draft)
        self.assertIn("You lock the child in the cellar.", draft)

    def test_lines_mode_is_used_by_produce_script_and_the_classic_writer_is_left_alone(self):
        fake = ts.ScriptFake({ts.HOOK_A: [ts.SCRIPT_A]})
        st = self.structure()
        def plan(game, topic, hook, facts, essential, structure=None):
            structure.update(st)
            return "STORY TIMELINE:\n1. x"
        written = []
        with mock.patch.object(g, "ask_ollama", fake), mock.patch.object(g, "plan_story_beats", plan), \
                mock.patch.object(g, "WRITER_MODE", "lines"), mock.patch.object(g, "write_by_lines", lambda *a: written.append(a) or ts.SCRIPT_A):
            result = g.produce_script(GAME, TOPIC, ts.EVIDENCE, IDS, ts.PASSAGES, ts.HOOK_A, "brief", {}, 5, [], "", [{"id": "E001", "fact": "x"}])
        self.assertTrue(written)
        self.assertNotIn("scriptgen", fake.calls)                     # the classic writer did not run
        self.assertNotIn("fullstory", fake.calls)
        self.assertLessEqual(result["attempts_used"], 2)


class LinePolishTests(unittest.TestCase):
    """The weakest line is rewritten on its own, and the best-scoring version is kept, not the last one."""
    DRAFT = ("STATUS: DRAFT\nHOOK: A lord claws at his face. [E001]\nNARRATION: A wraith haunts him.\nNARRATION: It was sad.\n"
             "PAYOFF: You lock the door and the wraith leaves.\nENDING QUESTION: Would you?\nCLAIMS REQUIRING MORE EVIDENCE: None")
    KEEP = {"items": [{"section": "NARRATION", "job": "a", "facts": [{"fact": "A wraith haunts Tomas."}], "max_words": 20},
                      {"section": "NARRATION", "job": "b", "facts": [{"fact": "His brother drowned."}], "max_words": 20},
                      {"section": "PAYOFF", "job": "c", "facts": [{"fact": "You lock the door."}], "max_words": 30}]}

    def run_polish(self, judge_replies, writer_replies):
        judges, writes = iter(judge_replies), iter(writer_replies)
        prompts = []
        with mock.patch.object(g, "ask_ollama", lambda p, **k: next(judges)), \
                mock.patch.object(g, "ask_writer", lambda p, **k: prompts.append(p) or next(writes)):
            return g.polish_lines(GAME, TOPIC, self.DRAFT, self.KEEP, "", rounds=2), prompts

    def test_the_weakest_line_is_rewritten_from_its_own_facts_and_kept_when_it_scores_higher(self):
        out, prompts = self.run_polish(["SCORE=5 WEAKEST=3 FIX=say what happened", "SCORE=7 WEAKEST=4 FIX=shorter", "SCORE=6 WEAKEST=2 FIX=x"],
                                       ["His brother drowned while he held the rope.", "You lock it."])
        self.assertIn("NARRATION: His brother drowned while he held the rope.", out)
        self.assertIn("PAYOFF: You lock the door and the wraith leaves.", out)      # the second rewrite scored lower: not kept
        self.assertIn("His brother drowned.", prompts[0])                           # written from that line's own facts
        self.assertIn("FIX: say what happened", prompts[0])
        self.assertIn("NARRATION: A wraith haunts him.", prompts[0])                 # seeing the lines before it

    def test_a_rewrite_that_does_not_score_higher_is_thrown_away(self):
        out, _ = self.run_polish(["SCORE=6 WEAKEST=3 FIX=x", "SCORE=6 WEAKEST=3 FIX=x"], ["Worse line."])
        self.assertEqual(g.spoken_lines(out), g.spoken_lines(self.DRAFT))

    def test_an_unusable_judge_leaves_the_draft_alone(self):
        out, prompts = self.run_polish(["no idea"], [])
        self.assertEqual(out, self.DRAFT)
        self.assertEqual(prompts, [])

    def test_the_hook_and_question_are_never_rewritten(self):
        out, prompts = self.run_polish(["SCORE=5 WEAKEST=1 FIX=x"], [])
        self.assertEqual(out, self.DRAFT)


class LineWriterDetailTests(unittest.TestCase):
    def test_an_unattached_cost_joins_the_path_it_is_about(self):
        st = LineByLineWriterTests().structure()
        st["paths"] = [("refuse Orla", [st["paths"][0][1][0]]), st["paths"][1]]   # the fight path without its cost
        st["loose"] = [{"id": "E009", "fact": "Fighting the wraith needs Tomas present, and he may not survive it."}]
        lines = g.script_skeleton(GAME, st, "Kael")
        fight = next(l for l in lines if "refuse Orla" in l["job"])
        self.assertIn("Fighting the wraith needs Tomas present, and he may not survive it.", [f["fact"] for f in fight["facts"]])
        self.assertNotIn("may not survive", " ".join(f["fact"] for f in lines[-1]["facts"]))

    def test_a_person_is_introduced_once(self):
        cast = "WHO IS WHO (...):\n- Orla: daughter of the lord  (source E001)"
        item = {"job": "x", "facts": [{"fact": "If the player trusts Orla, Kael locks the door."}]}
        first = g.build_line_prompt(GAME, TOPIC, ["HOOK: h", "NARRATION: A wraith haunts Tomas."], item, cast, [], 20)
        again = g.build_line_prompt(GAME, TOPIC, ["HOOK: h", "NARRATION: You can refuse Orla, daughter of the lord."], item, cast, [], 20)
        self.assertIn("- Orla: daughter of the lord", first)
        self.assertNotIn("- Orla: daughter of the lord", again)
        self.assertIn("Already introduced (just use the name, do not say who they are again): Orla", again)


class LinesStructureTests(unittest.TestCase):
    """The line writer gets every fact of every path and the required cause, whatever the planner picked."""
    def test_the_same_step_told_twice_is_kept_once(self):
        steps = [{"id": "E7", "fact": "You use the witcher's way to fight the wraith with Tomas present.", "option": "fight", "payoff_detail": True},
                 {"id": "E7", "fact": "You use the witcher's way and fight the wraith with Tomas present.", "option": "fight", "payoff_detail": True},
                 {"id": "E8", "fact": "You lock the door.", "option": "trick", "payoff_detail": True}]
        with mock.patch.object(g, "ask_ollama", lambda p, **k: "1"):
            st = g.lines_structure({}, steps, [])
        self.assertEqual(len(dict(st["paths"])["fight"]), 1)

    def test_a_near_duplicate_event_is_not_told_twice(self):
        opener = {"id": "E1", "fact": "A malicious wraith latched onto the jarl.", "cause_role": "EVENT"}
        dup = {"id": "E1", "fact": "The jarl is tormented by a malicious wraith that latched onto him.", "cause_role": "EVENT"}
        drowned = {"id": "E2", "fact": "His brother drowned in a storm.", "cause_role": "EVENT"}
        with mock.patch.object(g, "ask_ollama", lambda p, **k: "1"):
            st = g.lines_structure({"timeline": [opener, dup, drowned]}, [opener, dup, drowned], [])
        self.assertEqual([f["fact"] for f in st["timeline"]], [opener["fact"], drowned["fact"]])

    def test_the_payoff_is_sent_back_when_it_drops_the_reveal(self):
        st = LineByLineWriterTests().structure()
        replies = iter(["A line."] * 5 + ["The wraith leaves Tomas for you.", "The wraith leaves Tomas for you. Orla's man carried the child out through a second door."])
        with mock.patch.object(g, "ask_writer", lambda p, **k: next(replies)):
            draft = g.write_by_lines(GAME, TOPIC, "A lord claws at his face. [E001]", st, "")
        self.assertIn("second door", draft)

    def test_the_reveal_and_the_cost_survive_a_plan_that_dropped_them_and_background_is_capped(self):
        cause = {"id": "E1", "fact": "Tomas's brother drowned.", "cause_role": "EVENT"}
        rule = {"id": "E2", "fact": "Wraiths feed on guilt.", "cause_role": "RULE"}
        opener = {"id": "E3", "fact": "A wraith tortures Tomas.", "cause_role": "EVENT"}
        old1 = {"id": "E4", "fact": "Tomas once defied his father.", "cause_role": "BACKGROUND"}
        old2 = {"id": "E5", "fact": "Tomas was chained to a rock.", "cause_role": "BACKGROUND"}
        trick = [{"id": "E6", "fact": "Orla had a man catch the child.", "option": "trust Orla", "payoff_detail": True, "beat": "twist"},
                 {"id": "E6", "fact": "You lock the child in the oven.", "option": "trust Orla", "payoff_detail": True, "kind": "choice"}]
        fight = [{"id": "E7", "fact": "You fight it with Tomas there.", "option": "refuse Orla", "payoff_detail": True, "kind": "choice"},
                 {"id": "E7", "fact": "Tomas may not survive.", "option": "refuse Orla", "payoff_detail": True, "beat": "risk"}]
        planned = {"timeline": [opener, old1, old2], "paths": [("trust Orla", [trick[1]]), ("refuse Orla", [fight[0]])], "loose": []}
        asked = []
        with mock.patch.object(g, "ask_ollama", lambda p, **k: asked.append(p) or "2"):
            st = g.lines_structure(planned, [opener, old1, old2, cause, rule] + trick + fight, [cause, rule])
        self.assertEqual([f["id"] for f in st["timeline"]], ["E3", "E1", "E5", "E2"])         # opener, cause, the bridge event, rule
        self.assertIn("Wraiths feed on guilt.", asked[0])                                      # chosen against the rule
        self.assertEqual([o for o, _ in st["paths"]], ["refuse Orla", "trust Orla"])            # the reveal path last
        self.assertEqual([f["fact"] for f in st["paths"][1][1]], ["You lock the child in the oven.", "Orla had a man catch the child."])
        self.assertIn("Tomas may not survive.", [f["fact"] for f in st["paths"][0][1]])


class StoryNotManualTests(unittest.TestCase):
    """The judge's two biggest complaints about the classic writer: it reads like a manual, and it names too many."""
    def lines(self, body):
        return g.spoken_lines("STATUS: DRAFT\nHOOK: A lord claws at his face. [E001]\n" + body + "\nCLAIMS REQUIRING MORE EVIDENCE: None")

    def test_stacked_if_you_paths_are_flagged_and_one_is_fine(self):
        manual = self.lines("NARRATION: If you refuse her, you fight it with Tomas there.\nNARRATION: If you trust her, you lock the child in.\n"
                            "PAYOFF: The child was never in danger.\nENDING QUESTION: Which?")
        self.assertTrue(any("reads like a manual" in p for p in g.telling_problems(manual, GAME)))
        story = self.lines("NARRATION: You could fight it with Tomas there.\nNARRATION: Or, if you trust her, you lock the child in.\n"
                           "PAYOFF: The child was never in danger.\nENDING QUESTION: Which?")
        self.assertFalse(any("reads like a manual" in p for p in g.telling_problems(story, GAME)))

    def test_too_many_names_are_flagged_but_titles_do_not_count(self):
        crowded = self.lines("NARRATION: In Ashfall, his wife Anna fled from Phillip.\nNARRATION: The witch Weavess cursed their child Dea.\n"
                             "NARRATION: Now his daughter Tamara is missing too.\nPAYOFF: You find Anna.\nENDING QUESTION: Which?")
        self.assertTrue(any("too many for a newcomer" in p for p in g.telling_problems(crowded, GAME)))
        fine = self.lines("NARRATION: In Ashfall, Lord Tomas lost his brother Aki.\nNARRATION: Now Orla has a plan.\nPAYOFF: You agree.\nENDING QUESTION: Which?")
        self.assertFalse(any("too many for a newcomer" in p for p in g.telling_problems(fine, GAME)))

    def test_the_writer_and_editor_are_told(self):
        prompt = g.build_generation_prompt(GAME, TOPIC, ts.EVIDENCE, IDS, ts.HOOK_A, "brief")
        for rule in ("A STORY, NOT A MANUAL", "KEEP A QUESTION OPEN", "NAME BUDGET"):
            self.assertIn(rule, prompt)


class BridgeToRuleTests(unittest.TestCase):
    def test_no_guess_when_the_model_names_nothing(self):
        cand = [{"id": "E1", "fact": "His brother drowned."}]
        rule = [{"id": "E2", "fact": "Wraiths feed on guilt.", "cause_role": "RULE"}]
        with mock.patch.object(g, "ask_ollama", lambda p, **k: "none"):
            self.assertEqual(g.pick_bridge_events(cand, rule, fallback=False), [])
            self.assertEqual(g.pick_bridge_events(cand, rule), cand)          # the line writer may still use one

    def test_the_writer_is_told_never_to_use_a_bare_role(self):
        prompt = g.build_generation_prompt(GAME, TOPIC, ts.EVIDENCE, IDS, ts.HOOK_A, "brief")
        self.assertIn('never a bare "the girl"', prompt)


class TwoHalvesTests(unittest.TestCase):
    """Narration lines 1-3 tell the story; from line 4 to the payoff the choice must be told exactly."""
    DRAFT = ("STATUS: DRAFT\nHOOK: h [E001]\nNARRATION: One. [E001]\nNARRATION: Two. [E001]\nNARRATION: Three: you burn the granary. [E001]\n"
             "NARRATION: Four. [E001]\nNARRATION: Five. [E001]\nPAYOFF: Six. [E001]\nENDING QUESTION: Seven? [E001]\nCLAIMS REQUIRING MORE EVIDENCE: None")

    def test_the_precise_part_is_line_four_to_the_payoff(self):
        part = g.precise_part(self.DRAFT)
        self.assertEqual([c for _, c in g.spoken_lines(part)], ["Four. [E001]", "Five. [E001]", "Six. [E001]"])
        self.assertEqual(g.precise_part(ts.SCRIPT_A), ts.SCRIPT_A)              # too short to split: judged whole

    def test_a_key_step_told_only_in_the_storytelling_half_still_counts_as_missing(self):
        seen = []
        with mock.patch.object(g, "ask_ollama", lambda p, **k: seen.append(p) or "1: NO"):
            missing = g.missing_essential_semantic(GAME, g.precise_part(self.DRAFT), [{"fact": "You burn the granary.", "kind": "choice"}])
        self.assertEqual(len(missing), 1)
        self.assertNotIn("you burn the granary", seen[0].split("SCRIPT:")[1].lower())   # the judge only sees line 4 onwards

    def test_the_writer_and_editor_know_the_two_halves(self):
        prompt = g.build_generation_prompt(GAME, TOPIC, ts.EVIDENCE, IDS, ts.HOOK_A, "brief")
        self.assertIn("NARRATION lines 1-3: STORYTELLING", prompt)
        self.assertIn("NARRATION line 4 to the PAYOFF: PRECISE", prompt)


class CostToldInCodeTests(unittest.TestCase):
    POINTS = [{"fact": "You fight the wraith in its lair with Tomas present.", "option": "fight", "kind": "choice"},
              {"fact": "The host may not last long enough for the wraith to be killed.", "option": "fight", "beat": "risk"},
              {"fact": "You lock the child in the oven.", "option": "trick", "kind": "choice"}]

    def draft(self, fight):
        return (f"STATUS: DRAFT\nHOOK: h\nNARRATION: {fight}\nNARRATION: Or you lock the child in the oven.\n"
                "PAYOFF: The child was safe.\nENDING QUESTION: Which?\nCLAIMS REQUIRING MORE EVIDENCE: None")

    def test_a_fight_with_no_risk_is_caught(self):
        self.assertEqual(len(g.untold_costs(self.draft("You fight the wraith in its lair until it dissolves."), self.POINTS, GAME)), 1)

    def test_the_cost_in_any_words_passes(self):
        for told in ("You fight the wraith in its lair, but Tomas may not survive it.", "You can fight the wraith, risking Tomas's life."):
            self.assertEqual(g.untold_costs(self.draft(told), self.POINTS, GAME), [], told)


class SubjectAndRealChoiceTests(unittest.TestCase):
    def lines(self, first):
        return g.spoken_lines("STATUS: DRAFT\nHOOK: A man died for interrupting him. [E001]\nNARRATION: " + first + "\nNARRATION: Then Kael meets Iris, Vlod and Max.\n"
                              "PAYOFF: You win.\nENDING QUESTION: Which?\nCLAIMS REQUIRING MORE EVIDENCE: None")

    def test_the_subject_must_be_named_early(self):
        vague = g.telling_problems(self.lines("A powerful stranger can stop time."), GAME, "Master Odran")
        self.assertTrue(any("never name it" in p for p in vague))
        named = g.telling_problems(self.lines("Master Odran’s clap can stop time."), GAME, "Master O’dran")
        self.assertFalse(any("never name it" in p for p in g.telling_problems(self.lines("Odran can stop time."), GAME, "Master Odran")))

    def test_the_subject_does_not_count_against_the_name_budget(self):
        problems = g.telling_problems(self.lines("Odran, a demon, can stop time."), GAME, "Odran")
        self.assertFalse(any("too many for a newcomer" in p for p in problems))       # Kael, Iris, Vlod, Max + the subject

    def test_choice_extraction_wants_the_real_decision(self):
        prompt = g._choice_prompt(GAME, TOPIC, "[E001] x", "E001", 6)
        self.assertIn("ONLY REAL DECISIONS", prompt)
        self.assertIn("Steps every player goes through anyway", prompt)


if __name__ == "__main__":
    unittest.main()
