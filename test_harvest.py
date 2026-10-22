"""Tests for the clip harvester's moment finding and keyword cleanup. No network, no Ollama, no YouTube:
captions, chapters and the vision model are faked, so these prove the logic only."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_HOME = tempfile.mkdtemp()
os.environ["HOME"] = _HOME
os.environ["USERPROFILE"] = _HOME
sys.path.insert(0, str(Path(__file__).parent))
import harvest_clips_universal as H  # noqa: E402

GAME = "Ashfall Keep"
VTT = """WEBVTT
Kind: captions
Language: en

00:00:05.000 --> 00:00:08.000
welcome back to the walkthrough

00:01:10.000 --> 00:01:13.000
you left the sluice gate open

00:01:13.000 --> 00:01:16.000
you left the sluice gate open
and your son drowned in the race

00:03:20.000 --> 00:03:24.000
name the child and bury the wailer
beneath the mill's threshold

00:03:24.000 --> 00:03:27.000
then it will rest as a guardian
"""


class CaptionTests(unittest.TestCase):
    def test_captions_are_parsed_and_rolling_duplicates_removed(self):
        cues = H.parse_vtt(VTT)
        self.assertEqual(cues[0][0], 5)
        self.assertEqual([c[0] for c in cues], [5, 70, 73, 200, 204])
        self.assertTrue(cues[2][1].startswith("and your son drowned"))

    def test_the_moment_is_where_the_dialogue_matches_the_scene(self):
        cues = H.parse_vtt(VTT)
        moment = H.find_moment(cues, ["wailer", "threshold"], "The father can name the wailer and bury it.", GAME)
        self.assertEqual(moment[0], 199)
        self.assertIn("bury the wailer", moment[2])
        drowned = H.find_moment(cues, ["drowned", "son"], "The miller's neglect drowned his son.", GAME)
        self.assertEqual(drowned[0], 69)

    def test_weak_caption_matches_are_not_trusted(self):
        cues = H.parse_vtt(VTT)
        self.assertIsNone(H.find_moment(cues, ["dragon"], "A dragon attacks the keep.", GAME))

    def test_the_moment_is_looked_for_inside_the_matching_chapter_first(self):
        info = {"duration": 600, "chapters": [{"start_time": 0, "end_time": 150, "title": "The sluice gate"},
                                               {"start_time": 150, "end_time": 600, "title": "Naming the wailer"}]}
        with mock.patch.object(H, "video_captions", lambda y, u: H.parse_vtt(VTT)):
            start, reason, chapter, dialogue, score = H.locate_moment(
                "yt", "url", info, {}, ["wailer"], "Name the wailer and bury it.", GAME, {"wailer"}, 0)
        self.assertEqual((start, chapter), (199, "Naming the wailer"))
        self.assertIn("captions match", reason)
        self.assertGreaterEqual(score, H.MIN_MOMENT_SCORE)

    def test_without_captions_the_chapter_and_then_the_old_fallback_are_used(self):
        info = {"duration": 600, "chapters": [{"start_time": 150, "end_time": 600, "title": "Naming the wailer"}]}
        with mock.patch.object(H, "video_captions", lambda y, u: []):
            start, reason, chapter, dialogue, score = H.locate_moment(
                "yt", "url", info, {}, ["wailer"], "x", GAME, {"wailer"}, 0)
        self.assertEqual((reason.startswith("chapter"), dialogue, score), (True, "", 0))
        with mock.patch.object(H, "video_captions", lambda y, u: []):
            start, reason, *_ = H.locate_moment("yt", "url", {"duration": 1200}, {}, ["x"], "x", GAME, {"x"}, 0)
        self.assertEqual(reason, "no chapter match; skipped intro")

    def test_scenes_sharing_a_video_get_the_next_stretch(self):
        with mock.patch.object(H, "video_captions", lambda y, u: H.parse_vtt(VTT)):
            first = H.locate_moment("yt", "u", {"duration": 600}, {}, ["wailer"], "bury the wailer", GAME, {"wailer"}, 0)[0]
            second = H.locate_moment("yt", "u", {"duration": 600}, {}, ["wailer"], "bury the wailer", GAME, {"wailer"}, 1)[0]
        self.assertEqual(second - first, H.CLIP_SECONDS)

    def test_a_multi_word_keyword_only_counts_when_heard_together(self):
        vtt = VTT.replace("then it will rest as a guardian", "blood ties are a strong bond")
        cues = H.parse_vtt(vtt + "\n00:05:00.000 --> 00:05:03.000\nthe bloody baron sobs over the grave\n")
        moment = H.find_moment(cues, ["bloody baron"], "The lord weeps.", GAME)
        self.assertEqual(moment[0], 299)

    def test_words_the_characters_would_say_bridge_the_narrations_wording(self):
        cues = H.parse_vtt(VTT)
        self.assertIsNone(H.find_moment(cues, [], "The miller's neglect killed his child.", GAME))
        moment = H.find_moment(cues, [], "The miller's neglect killed his child.", GAME, spoken_words=["drowned", "sluice"])
        self.assertIn(moment[0], (69, 72))
        reply = {"message": {"content": '{"words": ["drowned", "sluice", "gate"]}'}}
        with mock.patch.object(H, "http_json", lambda *a, **k: reply), mock.patch.dict(os.environ, {"HARVEST_DIALOGUE_WORDS": "1"}):
            self.assertEqual(H.dialogue_words({"model": "m"}, GAME, "x"), ["drowned", "sluice", "gate"])
        self.assertEqual(H.dialogue_words({"model": "m"}, GAME, "x"), [])          # off unless switched on
        self.assertEqual(H.dialogue_words(None, GAME, "x"), [])


class KeywordTests(unittest.TestCase):
    def test_sentence_start_function_words_are_not_glued_to_names(self):
        known = H.collect_known(["The healer met Mirela at Ashfall Keep. Orsk watched."])
        for sentence, expected in (("A grieving father begs Orsk to spare it.", ["Orsk"]),
                                   ("In Ashfall Keep, Mirela burned the granary.", ["Mirela"]),
                                   ("Should Orsk spare the creature?", ["Orsk"]),
                                   ("If Orsk spares it, Mirela weeps.", ["Orsk", "Mirela"])):
            ents = H.extract_entities(sentence, known, GAME)
            self.assertEqual(sorted(ents), sorted(expected), sentence)
        self.assertEqual(H.norm_entity("A"), "")

    def test_wiki_section_labels_are_not_read_as_names(self):
        evidence = "[E001]\nSource: wiki\nExcerpt: (Biography > Meeting Geralt) Phillip begged for his child.\n"
        self.assertEqual(H.evidence_excerpt(evidence, "E001"), "Phillip begged for his child.")

    def test_ideas_are_not_hunted_for_as_footage(self):
        reply = {"message": {"content": '{"visual": false}'}}
        with mock.patch.object(H, "http_json", lambda *a, **k: reply):
            self.assertFalse(H.is_visual_subject({"model": "m"}, {"kind": "noun", "label": "miscarriage", "facts": ""}))
        self.assertTrue(H.is_visual_subject({"model": "m"}, {"kind": "topic", "label": "wailer"}))
        self.assertTrue(H.is_visual_subject(None, {"kind": "noun", "label": "anything"}))

    def test_the_vision_check_is_told_names_cannot_be_recognised_and_hears_the_dialogue(self):
        captured = {}

        def fake(vision, clip, prompt, label):
            captured["p"] = prompt
            return {"relevance": 8, "clutter": 1, "quality": 8, "shows": "x"}
        with mock.patch.object(H, "ask_vision", fake):
            H.review_clip({"model": "m"}, "clip.mp4", GAME, "Wailer", "Mirela buries the wailer.", "", "bury the wailer")
        self.assertIn("You cannot recognise characters by name", captured["p"])
        self.assertIn('"bury the wailer"', captured["p"])


class SearchAndKeepTests(unittest.TestCase):
    def test_every_keyword_is_searched_alone_and_paired(self):
        q = H.queries_from_keywords(GAME, ["Mirela", "granary", "seed grain", "elders"], "Mirela burns the granary.", "The granary")
        joined = " | ".join(q)
        for k in ("Mirela", "granary", "seed grain", "elders"):
            self.assertIn(k, joined)
        self.assertTrue(q[0].startswith(f"{GAME} Mirela granary"))                 # most specific pair first
        self.assertIn("seed grain elders", joined)                                  # later keywords get paired too
        self.assertLessEqual(len(q), H.MAX_QUERIES_PER_SCENE)
        self.assertIn("all cutscenes", q[-1])                                       # the topic is still the last resort

    def test_clips_below_the_relevance_bar_are_deleted_and_the_search_goes_on(self):
        out = Path(tempfile.mkdtemp())
        videos = [{"url": f"https://v/{i}", "title": f"{GAME} wailer walkthrough no commentary {i}", "duration": 900}
                  for i in range(6)]
        scores = iter([4, 8, 5, 9])                                                 # 2 weak, then 2 good

        def fetch(yt, ff, url, start, base, out_file):
            Path(out_file).write_bytes(b"x")
            return True, ""

        def review(*a, **k):
            r = next(scores)
            return {"relevance": r, "clutter": 1, "quality": 7, "shows": f"score {r}"}
        ctx = {"lines": [("NARRATION", "The miller buries the wailer.", ["E001"])], "game": GAME, "topic": "The wailer",
               "vision": {"model": "m"}, "yt_dlp": "yt", "ffmpeg": "ff", "used": {}, "stamp": "t", "evidence_text": ""}
        with mock.patch.object(H, "OUTPUT_DIR", out), mock.patch.object(H, "search_youtube", lambda y, q: (videos, "")), \
                mock.patch.object(H, "video_info", lambda y, u: {"duration": 900}), \
                mock.patch.object(H, "locate_moment", lambda *a, **k: (60, "captions match", "", "", 5)), \
                mock.patch.object(H, "fetch_segment", fetch), mock.patch.object(H, "review_clip", review), \
                mock.patch.object(H, "make_preview", lambda *a: ""), mock.patch.object(H, "dialogue_words", lambda *a: []), \
                mock.patch.object(H, "scene_entities", lambda c, i: []):
            asset = H.process_scene(ctx, 1, ["wailer", "miller"])
        self.assertEqual(sorted(o["vision_review"]["relevance"] for o in asset["options"]), [8, 9])
        self.assertEqual(sorted(r["relevance"] for r in asset["rejected"]), [4, 5])
        self.assertEqual(sorted(p.name for p in out.glob("*.mp4")), ["t_scene_01_A.mp4", "t_scene_01_B.mp4"])


if __name__ == "__main__":
    unittest.main()
