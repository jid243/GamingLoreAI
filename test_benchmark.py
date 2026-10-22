"""The benchmark must pick distinct topics, score blind, and leave the project exactly as it was."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_writers as b  # noqa: E402

DRAFT = ("STATUS: DRAFT\nHOOK: A lord claws at his face. [E001]\nNARRATION: In Ashfall Keep, a wraith feeds on guilt. [E002]\n"
         "PAYOFF: You lock the door. [E003]\nENDING QUESTION: Would you? [E001]\nCLAIMS REQUIRING MORE EVIDENCE: None")


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        for sub in ("sources", "scripts", "fact_checks"):
            (self.root / sub).mkdir()
        self.patches = [mock.patch.object(b, n, self.root / sub) for n, sub in
                        (("PROJECT", ""), ("SOURCES", "sources"), ("SCRIPTS", "scripts"), ("CHECKS", "fact_checks"))]
        self.patches.append(mock.patch.object(b, "EVIDENCE", self.root / "sources" / "evidence_passages.txt"))
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def research(self, name, game, topic, mtime):
        path = self.root / "sources" / f"{name}_research_1.txt"
        path.write_text(f"GAME: {game}\nTOPIC: {topic}\n", encoding="utf-8")
        os.utime(path, (mtime, mtime))
        return path

    def test_one_file_per_topic_and_a_renamed_topic_counts_once(self):
        self.research("a", "Ashfall Keep", "The Wraith", 500)
        self.research("b", "Ashfall Keep", "The Wraith", 400)
        self.research("c", "Ashfall Keep", "Burning Granary", 300)
        self.research("d", "Ashfall Keep", "Ashfall Keep's Burning Granary", 200)
        self.research("e", "Ashfall Keep", "The Drowned Bell", 100)
        topics = [t for _, _, t in b.pick_topics(3)]
        self.assertEqual(topics, ["The Wraith", "Burning Granary", "The Drowned Bell"])

    def test_a_benchmark_stopped_part_way_is_put_back_next_time(self):
        research = self.research("a", "Ashfall Keep", "The Wraith", 500)
        evidence = self.root / "sources" / "evidence_passages.txt"
        evidence.write_text("MY EVIDENCE", encoding="utf-8")
        (self.root / "scripts" / "mine.txt").write_text("x")
        out = self.root / "benchmark" / "20260101_000000"
        saved = out / "_your_files"
        saved.mkdir(parents=True)
        (saved / "evidence_passages.txt").write_text("MY EVIDENCE", encoding="utf-8")
        (saved / "research_dates.json").write_text(json.dumps({research.name: [500, 500]}))
        (saved / "files_before.json").write_text(json.dumps({"scripts": ["mine.txt"], "fact_checks": []}))
        # ...then it was stopped mid-run:
        evidence.write_text("BENCH EVIDENCE", encoding="utf-8")
        os.utime(research, (999, 999))
        (self.root / "scripts" / "bench_draft.txt").write_text("y")
        with mock.patch.object(sys, "argv", ["benchmark_writers.py", "--restore"]):
            self.assertEqual(b.main(), 0)
        self.assertEqual(evidence.read_text(encoding="utf-8"), "MY EVIDENCE")
        self.assertEqual(research.stat().st_mtime, 500)
        self.assertEqual([p.name for p in (self.root / "scripts").iterdir()], ["mine.txt"])
        self.assertTrue((out / "drafts" / "bench_draft.txt").exists())
        self.assertTrue((saved / "RESTORED").exists())

    def test_a_missing_model_is_marked_pending_not_swapped(self):
        out = self.root / "rep"
        out.mkdir()
        b.write_report(out, [], [("classic", "gemma4:12b-it-qat"), ("lines", "gemma4:12b-it-qat")])
        report = (out / "report.md").read_text(encoding="utf-8")
        self.assertIn("| classic | gemma4:12b-it-qat | PENDING: model not installed |", report)

    def test_the_judge_scores_eight_things_and_averages_two_rounds(self):
        replies = iter(["HOOK=8 PACING=5 SUSPENSE=6 DETAIL=7 CHARACTERS=4 PAYOFF=6 DELIVERY=7 EDIT=4 WHY=flat middle",
                        "**HOOK=6** PACING=7 SUSPENSE=6 DETAIL=7 CHARACTERS=6 PAYOFF=6 DELIVERY=7 EDIT=6 WHY=ok"])
        with mock.patch.object(b.g, "ask_ollama", lambda *a, **k: next(replies)):
            scores = b.judge("HOOK: x")
        self.assertEqual((scores["hook"], scores["pacing"], scores["edit"]), (7.0, 6.0, 5.0))
        self.assertEqual(scores["characters"], 5.0)
        self.assertEqual(scores["story_total"], 49.0)
        self.assertEqual(scores["why"], "flat middle")

    def test_the_project_is_put_back_and_drafts_are_moved_out(self):
        research = self.research("a", "Ashfall Keep", "The Wraith", 500)
        other = self.research("b", "Ashfall Keep", "Old Topic", 900)        # the user's current topic is the newest
        evidence = self.root / "sources" / "evidence_passages.txt"
        evidence.write_text("MY EVIDENCE", encoding="utf-8")
        menu = self.root / "fact_checks" / "ashfall_keep_old_topic_script_menu.json"
        menu.write_text('{"mine": true}', encoding="utf-8")
        mine = self.root / "scripts" / "my_draft.txt"
        mine.write_text("keep me", encoding="utf-8")
        name = b.slug("Ashfall Keep_The Wraith")

        def fake_run(label, argv, env, log):
            evidence.write_text("BENCH EVIDENCE", encoding="utf-8")
            if env.get("HOOKS_ONLY"):
                (self.root / "fact_checks" / f"{name}_hook_menu.json").write_text(json.dumps({"hooks": [{"spoken": "A lord claws at his face."}]}))
                (self.root / "fact_checks" / f"{name}_prepared.json").write_text("{}")
            if env.get("CHOSEN_HOOK"):
                log.write_text("...\nMODEL CALLS: local=31 online=0\n", encoding="utf-8")
                (self.root / "fact_checks" / f"{name}_script_menu.json").write_text(json.dumps(
                    {"versions": [{"draft": DRAFT, "eligible": True, "total": 70, "words": 20}]}))
                (self.root / "scripts" / f"{name}_{env['WRITER_MODE']}_DRAFT.txt").write_text(DRAFT)
            return 3, True
        with mock.patch.object(b, "run", fake_run), mock.patch.object(b, "MODELS", ["m1"]), mock.patch.object(b, "TOPIC_LIMIT", 2), \
                mock.patch.object(b.g, "ask_ollama", lambda *a, **k: "HOOK=5 PACING=5 SUSPENSE=5 DETAIL=5 CHARACTERS=5 PAYOFF=5 DELIVERY=5 EDIT=5 WHY=x"), \
                mock.patch("requests.get", side_effect=Exception("offline")):
            self.assertEqual(b.main(), 0)
        self.assertEqual(evidence.read_text(encoding="utf-8"), "MY EVIDENCE")
        self.assertEqual(menu.read_text(encoding="utf-8"), '{"mine": true}')
        self.assertEqual(research.stat().st_mtime, 500)
        self.assertEqual(other.stat().st_mtime, 900)
        self.assertEqual(sorted(p.name for p in (self.root / "scripts").iterdir()), ["my_draft.txt"])
        out = next((self.root / "benchmark").iterdir())
        report = (out / "report.md").read_text(encoding="utf-8")
        self.assertIn("| classic | m1 | 1 |", report)               # the topic without verified hooks is skipped
        self.assertIn("| lines | m1 | 1 |", report)
        self.assertIn("NARRATION: In Ashfall Keep, a wraith feeds on guilt.", report)   # citations stripped
        self.assertIn("31 local / 0 online calls", report)
        self.assertTrue((out / "drafts").is_dir() and any((out / "drafts").iterdir()))
        self.assertTrue((self.root / "fact_checks" / f"{name}_prepared.json").exists())   # caches stay for next time


if __name__ == "__main__":
    unittest.main()
