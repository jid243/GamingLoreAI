"""Tests for gaminglore_app.py: commands built from the form, draft approval, keyword saving, the job runner and
the HTTP endpoints. Uses a throwaway project folder; no Ollama, no YouTube, no real pipeline step is run."""
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

_HOME = tempfile.mkdtemp()
os.environ["HOME"] = _HOME
os.environ["USERPROFILE"] = _HOME
sys.path.insert(0, str(Path(__file__).parent))
import gaminglore_app as A  # noqa: E402

DRAFT = """STATUS: DRAFT
HOOK: A healer must burn the granary where her sister lies. [E001]
NARRATION: In Ashfall Keep, the elders hid infected grain from Mirela. [E002]
PAYOFF: Burning it stops the plague but leaves no seed. [E003]
ENDING QUESTION: Would you burn it? [E001]
CLAIMS REQUIRING MORE EVIDENCE: None
"""
EVIDENCE = """GAME: Ashfall Keep
TOPIC: The granary

[E001]
Source: wiki
Excerpt: The healer Mirela must decide whether to burn the granary.
"""


class AppTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.patches = [mock.patch.object(A, name, self.root / sub) for name, sub in (
            ("PROJECT", ""), ("SCRIPT_DIR", "scripts"), ("SOURCES_DIR", "sources"), ("CHECK_DIR", "fact_checks"),
            ("MANIFEST_DIR", "clip_manifests"))] + [mock.patch.object(A, "EVIDENCE_FILE", self.root / "sources" / "evidence_passages.txt")]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        for sub in ("scripts", "sources", "fact_checks", "clip_manifests"):
            (self.root / sub).mkdir()

    def test_the_research_form_becomes_a_non_interactive_command(self):
        cmds = A.build_commands("research", {"game": "Ashfall Keep", "topic": "The granary", "wiki": "ashfall",
                                             "pick": "top5", "urls": "https://a.example/wiki/X\nnot-a-url"})
        argv = cmds[0][1]
        self.assertIn("--no-prompt", argv)
        self.assertEqual(argv[argv.index("--game") + 1], "Ashfall Keep")
        self.assertEqual(argv.count("--url"), 1)
        self.assertEqual(argv[argv.index("--pick") + 1], "top5")
        with self.assertRaises(ValueError):
            A.build_commands("research", {"game": "", "topic": "x"})

    def test_the_story_button_chains_research_evidence_and_script_with_the_chosen_options(self):
        cmds = A.build_commands("story", {"game": "G", "topic": "T", "model": "some-model", "quick": True, "max_words": "120"})
        self.assertEqual([c[0] for c in cmds], ["Research", "Evidence", "Script"])
        self.assertEqual(cmds[2][2], {"OLLAMA_MODEL": "some-model", "SCRIPT_CANDIDATES": "1", "MAX_ATTEMPTS": "3",
                                      "SCRIPT_MAX_WORDS": "120"})
        self.assertIn("--no-coverage", A.build_commands("clips", {"no_coverage": True})[0][1])
        self.assertEqual(A.build_commands("clips", {"min_relevance": "6"})[0][2], {"HARVEST_MIN_RELEVANCE": "6"})
        self.assertIsNone(A.build_commands("clips", {"min_relevance": "eleven"})[0][2])
        self.assertTrue(all(Path(c[1][2]).name in A.STEPS.values() for c in cmds))

    def test_hooks_step_and_a_picked_hook_reach_the_generator(self):
        hooks = A.build_commands("hooks", {"model": "m"})
        self.assertEqual(hooks[0][2], {"OLLAMA_MODEL": "m", "HOOKS_ONLY": "1"})
        self.assertEqual(Path(hooks[0][1][2]).name, A.STEPS["script"])
        picked = A.build_commands("script", {"chosen_hook": "  A healer must burn the granary.  "})
        self.assertEqual(picked[0][2], {"CHOSEN_HOOK": "A healer must burn the granary."})
        self.assertNotIn("CHOSEN_HOOK", A.build_commands("script", {})[0][2])

    def test_the_hook_menu_for_the_current_topic_is_in_the_state(self):
        (self.root / "sources" / "g_t_research_1.txt").write_text("GAME: Ashfall Keep\nTOPIC: The granary\n", encoding="utf-8")
        menu = {"hooks": [{"text": "A healer must burn it. [E001]", "spoken": "A healer must burn it.", "total": 80, "recommended": True}]}
        (self.root / "fact_checks" / "ashfall_keep_the_granary_hook_menu.json").write_text(json.dumps(menu), encoding="utf-8")
        (self.root / "fact_checks" / "other_game_x_hook_menu.json").write_text(json.dumps({"hooks": [{"text": "no"}]}), encoding="utf-8")
        with mock.patch.object(A, "ollama_models", lambda: []):
            self.assertEqual([h["spoken"] for h in A.project_state()["hooks"]], ["A healer must burn it."])

    def test_picking_a_version_makes_it_the_draft_with_its_own_status_and_report(self):
        (self.root / "sources" / "g_t_research_1.txt").write_text("GAME: Ashfall Keep\nTOPIC: The granary\n", encoding="utf-8")
        menu = {"timestamp": "20261021_220000", "versions": [
            {"number": 1, "draft": DRAFT, "eligible": True, "total": 80, "selected": True, "review": "CLEAR one"},
            {"number": 2, "draft": DRAFT.replace("Would you burn it?", "Would you burn it, healer?"), "eligible": False,
             "total": 60, "selected": False, "review": "FLAGGED two"}]}
        (self.root / "fact_checks" / "ashfall_keep_the_granary_script_menu.json").write_text(json.dumps(menu), encoding="utf-8")
        with mock.patch.object(A, "ollama_models", lambda: []):
            self.assertEqual(len(A.project_state()["versions"]), 2)
        name = A.use_version(2, "Ashfall Keep", "The granary")
        self.assertEqual(name, "ashfall_keep_the_granary_20261021_220000_v2_REJECTED.txt")
        with mock.patch.object(A, "ollama_models", lambda: []):
            st = A.project_state()
        self.assertEqual((st["draft"]["file"], st["draft"]["review"].strip()), (name, "FLAGGED two"))
        self.assertIn("healer?", st["draft"]["text"])
        self.assertEqual(A.build_commands("script", {"versions": "4"})[0][2], {"SCRIPT_VERSIONS": "4"})
        with self.assertRaises(ValueError):
            A.use_version(9, "Ashfall Keep", "The granary")

    def test_approving_a_rejected_draft_makes_an_approved_copy_and_a_narration_file(self):
        rejected = self.root / "scripts" / "g_t_1_REJECTED.txt"
        rejected.write_text(DRAFT, encoding="utf-8")
        name = A.save_draft(DRAFT.replace("Would you burn it?", "Would you burn it with her inside?"), approve=True)
        self.assertEqual(name, "g_t_1_REVIEW_REQUIRED.txt")
        self.assertTrue(rejected.exists())                                   # the original is kept
        self.assertIn("with her inside", (self.root / "scripts" / name).read_text(encoding="utf-8"))
        narration = (self.root / "scripts" / "g_t_1_NARRATION.txt").read_text(encoding="utf-8")
        self.assertNotIn("[E", narration)
        with self.assertRaises(ValueError):
            A.save_draft("no tags here", approve=False)

    def test_state_reports_the_newest_draft_evidence_and_research(self):
        (self.root / "sources" / "g_t_research_1.txt").write_text("GAME: Ashfall Keep\nTOPIC: The granary\n", encoding="utf-8")
        (self.root / "sources" / "evidence_passages.txt").write_text(EVIDENCE, encoding="utf-8")
        (self.root / "scripts" / "g_t_1_REVIEW_REQUIRED.txt").write_text(DRAFT, encoding="utf-8")
        (self.root / "fact_checks" / "g_t_1_review.txt").write_text("EVIDENCE AUDIT: CLEAR", encoding="utf-8")
        with mock.patch.object(A, "ollama_models", lambda: ["m1"]):
            st = A.project_state()
        self.assertEqual((st["research"]["game"], st["evidence"]["passages"]), ("Ashfall Keep", 1))
        self.assertEqual(st["draft"]["status"], "REVIEW_REQUIRED")
        self.assertIn("EVIDENCE AUDIT", st["draft"]["review"])

    def test_the_job_runner_streams_output_and_reports_failure(self):
        A.run_commands("demo", [("first", [sys.executable, "-c", "print('hello'); import sys; sys.stdout.write('tick\\r'); print('done')"], None),
                                ("second", [sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"], None),
                                ("never", [sys.executable, "-c", "print('should not run')"], None)])
        for _ in range(100):
            if not A.JOB.busy():
                break
            time.sleep(0.05)
        snap = A.JOB.snapshot()
        text = "\n".join(l for l in snap["lines"] if not l.startswith("==="))
        self.assertEqual(snap["status"], "failed")
        self.assertIn("hello", text)
        self.assertIn("boom", text)
        self.assertNotIn("should not run", text)
        self.assertNotIn("tick", text)                                       # progress ticks are not logged
        self.assertIn("done", text)

    def test_windows_line_endings_are_logged_and_progress_redraws_are_not(self):
        job = A.Job()
        splitter = A.LineSplitter(job)
        for byte in "Model: m\r\nHook round 1/3\r\n  10%\r  55%\r 100%\r\nblank\r\r\nd\u00e9j\u00e0 vu".encode("utf-8"):
            splitter.feed(bytes([byte]))
        self.assertEqual(job.lines, ["Model: m", "Hook round 1/3", " 100%", "blank"])
        self.assertTrue(job.last_output)
        splitter.close()
        self.assertEqual(job.lines[-1], "d\u00e9j\u00e0 vu")

    def test_an_older_topics_draft_is_labelled_as_such(self):
        (self.root / "sources" / "g_t_research_1.txt").write_text("GAME: Ashfall Keep\nTOPIC: The granary\n", encoding="utf-8")
        (self.root / "scripts" / "ashfall_keep_the_mill_1_REVIEW_REQUIRED.txt").write_text(DRAFT, encoding="utf-8")
        with mock.patch.object(A, "ollama_models", lambda: []):
            self.assertTrue(A.project_state()["draft"]["other_topic"])
        (self.root / "scripts" / "ashfall_keep_the_granary_2_REVIEW_REQUIRED.txt").write_text(DRAFT, encoding="utf-8")
        later = time.time() + 5
        os.utime(self.root / "scripts" / "ashfall_keep_the_granary_2_REVIEW_REQUIRED.txt", (later, later))
        with mock.patch.object(A, "ollama_models", lambda: []):
            self.assertFalse(A.project_state()["draft"]["other_topic"])

    def test_a_good_script_can_be_kept_as_a_style_example(self):
        with mock.patch.object(A, "STYLE_DIR", self.root / "style_examples"):
            name = A.save_style_example(DRAFT, "Ashfall Keep", "The granary")
            text = (self.root / "style_examples" / name).read_text(encoding="utf-8")
            with self.assertRaises(ValueError):
                A.save_style_example(DRAFT, "", "")
        self.assertTrue(text.startswith("GAME: Ashfall Keep\nTOPIC: The granary\n[hook] A healer"))
        self.assertIn("[question] Would you burn it?", text)
        self.assertNotIn("[E0", text)

    def test_the_http_endpoints(self):
        (self.root / "scripts" / "g_t_1_REVIEW_REQUIRED.txt").write_text(DRAFT, encoding="utf-8")
        server = A.ThreadingHTTPServer(("127.0.0.1", 0), A.Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_address[1]}"

        def call(path, body=None):
            req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    return resp.status, resp.read().decode()
            except urllib.error.HTTPError as error:
                return error.code, error.read().decode()
        status, page = call("/")
        self.assertEqual(status, 200)
        self.assertIn("GamingLoreAI", page)
        self.assertNotIn("add_captions", page)
        with mock.patch.object(A, "ollama_models", lambda: None):
            status, body = call("/api/state")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["draft"]["file"], "g_t_1_REVIEW_REQUIRED.txt")
        status, body = call("/api/run", {"step": "nonsense"})
        self.assertEqual(status, 400)
        status, body = call("/api/draft", {"text": DRAFT, "approve": False})
        self.assertEqual((status, json.loads(body)["ok"]), (200, True))


class GeminiKeyTests(unittest.TestCase):
    setUp = AppTests.setUp

    def test_the_key_is_saved_for_the_generator_and_only_its_last_characters_are_shown(self):
        status = A.save_gemini_key("  AIzaSyA1234567890abcdefghijWXYZ \n")
        self.assertEqual((self.root / "gemini_api_key.txt").read_text(encoding="utf-8"), "AIzaSyA1234567890abcdefghijWXYZ")
        self.assertEqual(status, {"set": True, "hint": "…WXYZ"})
        self.assertNotIn("AIzaSyA1234567890", json.dumps(A.project_state()))
        self.assertEqual(A.save_gemini_key(""), {"set": False, "hint": ""})          # blank removes it
        self.assertFalse((self.root / "gemini_api_key.txt").exists())

    def test_text_that_is_not_a_key_is_refused(self):
        with self.assertRaises(ValueError):
            A.save_gemini_key("short")
        self.assertFalse((self.root / "gemini_api_key.txt").exists())

    def test_any_key_format_is_accepted_and_copy_mistakes_are_cleaned(self):
        for pasted, saved in (("AQ.Ab8RN6Kx-yz_12.34567890abcdef", "AQ.Ab8RN6Kx-yz_12.34567890abcdef"),
                              ('"AIzaSyA1234567890abcdefghij"', "AIzaSyA1234567890abcdefghij"),
                              ("GEMINI_API_KEY=AIzaSyA1234567890abcdefghij", "AIzaSyA1234567890abcdefghij"),
                              ("AIzaSyA12345 67890abcdef\nghij", "AIzaSyA1234567890abcdefghij")):
            A.save_gemini_key(pasted)
            self.assertEqual((self.root / "gemini_api_key.txt").read_text(encoding="utf-8"), saved)

    def test_the_key_test_reports_a_rejected_key_in_plain_words(self):
        A.save_gemini_key("AIzaSyA1234567890abcdefghijWXYZ")
        def reject(req, timeout=None):
            self.assertEqual(req.get_header("X-goog-api-key"), "AIzaSyA1234567890abcdefghijWXYZ")
            self.assertNotIn("AIza", req.full_url)
            raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, None)
        with mock.patch.object(A.urllib.request, "urlopen", reject):
            self.assertIn("rejected", A.test_gemini_key()["message"])

    def test_a_groq_key_is_kept_in_its_own_file_and_checked_with_a_bearer_header(self):
        A.save_key("groq", "gsk_1234567890abcdefghijklmnopQRST")
        self.assertEqual((self.root / "groq_api_key.txt").read_text(encoding="utf-8"), "gsk_1234567890abcdefghijklmnopQRST")
        self.assertEqual(A.project_state()["keys"]["groq"], {"set": True, "hint": "…QRST"})
        self.assertFalse(A.project_state()["keys"]["gemini"]["set"])
        seen = {}
        class Ok:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *a): return False
        def opener(req, timeout=None):
            seen["auth"], seen["url"] = req.get_header("Authorization"), req.full_url
            return Ok()
        with mock.patch.object(A.urllib.request, "urlopen", opener):
            self.assertTrue(A.test_key("groq")["ok"])
        self.assertEqual(seen["auth"], "Bearer gsk_1234567890abcdefghijklmnopQRST")
        self.assertNotIn("gsk_", seen["url"])

    def test_the_page_has_the_key_box(self):
        self.assertIn('id="key_gemini"', A.PAGE)
        self.assertIn('id="key_groq"', A.PAGE)
        self.assertIn('type="password"', A.PAGE)


class WriterOptionTests(unittest.TestCase):
    def test_writer_mode_model_and_online_switch_reach_the_generator(self):
        env = A.build_commands("script", {"writer_mode": "lines", "writer_model": "gemma4:12b-it-qat"})[0][2]
        self.assertEqual((env["WRITER_MODE"], env["LOCAL_WRITER_MODEL"]), ("lines", "gemma4:12b-it-qat"))
        self.assertNotIn("USE_ONLINE_WRITERS", env)                                       # off unless ticked
        self.assertEqual(A.build_commands("hooks", {"online_writers": True})[0][2]["USE_ONLINE_WRITERS"], "1")
        self.assertNotIn("WRITER_MODE", A.build_commands("script", {})[0][2])          # classic stays the default

    def test_the_benchmark_button_runs_the_benchmark(self):
        label, argv, env = A.build_commands("benchmark", {})[0]
        self.assertTrue(argv[-1].endswith("benchmark_writers.py"))
        self.assertIn('data-run="benchmark"', A.PAGE)
        self.assertIn('id="writer_mode"', A.PAGE)


if __name__ == "__main__":
    unittest.main()
