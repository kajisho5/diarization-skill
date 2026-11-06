"""Long recordings (ADR-003): chunked DIARIZE, speakers linked across chunks by embeddings, resumable.

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import copy
import functools
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diarization_skill.chunks import merge_adjacent, plan_chunks  # noqa: E402
from diarization_skill.errors import DiarizationError  # noqa: E402
from diarization_skill.metrics import der  # noqa: E402
from diarization_skill.service import diarize  # noqa: E402
from diarization_skill.turns import validate_turns  # noqa: E402
from fake_engine import VoiceEngine, write_tone_wav  # noqa: E402
from test_diarization import HAVE_FFMPEG, run_cli  # noqa: E402

# 12 s, three chunks of 4 s: voice A (200 Hz) in chunks 0 and 1 and across the 8 s boundary, B (800 Hz) in
# chunks 0 and 1, C (450 Hz) only in chunk 2.
PLAN = [(0.0, 0.5, None), (0.5, 2.0, 200.0), (2.0, 3.5, 800.0), (3.5, 4.5, None), (4.5, 6.0, 800.0), (6.0, 9.0, 200.0),
        (9.0, 9.5, None), (9.5, 11.5, 450.0), (11.5, 12.0, None)]
REFERENCE = [(0.5, 2.0, "A"), (2.0, 3.5, "B"), (4.5, 6.0, "B"), (6.0, 9.0, "A"), (9.5, 11.5, "C")]


class FlakyVoiceEngine(VoiceEngine):
    """Fails on call number `fail_on` (1-based), like a run interrupted part way."""

    def __init__(self, fail_on: int) -> None:
        super().__init__()
        self.fail_on = fail_on

    def diarize(self, req):
        if len(self.requests) + 1 == self.fail_on:
            self.requests.append(req)
            raise RuntimeError("interrupted")
        return super().diarize(req)


class DriftingVoiceEngine(VoiceEngine):
    """Adds a per-call offset to every embedding, so the same voice in two chunks is similar, not identical."""

    def diarize(self, req):
        res = super().diarize(req)
        k = len(self.requests)
        res.embeddings = {lab: [x + (0.3 if i == k % 12 else 0.0) for i, x in enumerate(v)] for lab, v in res.embeddings.items()}
        return res


class PlanTests(unittest.TestCase):
    def test_equal_chunks_cover_the_media(self):
        self.assertEqual(plan_chunks(12.0, 4.0), [(0.0, 4.0), (4.0, 8.0), (8.0, 12.0)])
        self.assertEqual(plan_chunks(12.5, 4.0), [(0.0, 3.125), (3.125, 6.25), (6.25, 9.375), (9.375, 12.5)])
        self.assertEqual(plan_chunks(3.0, 1800.0), [(0.0, 3.0)])
        eight_hours = plan_chunks(8 * 3600.0, 1800.0)
        self.assertEqual((len(eight_hours), eight_hours[-1][1]), (16, 28800.0))

    def test_merge_across_a_boundary(self):
        self.assertEqual(merge_adjacent([(6.0, 8.0, "a"), (8.0, 9.0, "a"), (9.0, 10.0, "b")]), [(6.0, 9.0, "a"), (9.0, 10.0, "b")])
        self.assertEqual(merge_adjacent([(6.0, 8.0, "a"), (8.0, 9.0, "b")]), [(6.0, 8.0, "a"), (8.0, 9.0, "b")])
        self.assertEqual(merge_adjacent([(0.0, 1.0, "a"), (2.0, 3.0, "a")]), [(0.0, 1.0, "a"), (2.0, 3.0, "a")])


@unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg and ffprobe")
class ChunkedDiarizeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.media = os.path.join(self.tmp, "day.wav")
        write_tone_wav(self.media, PLAN)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def hyp(self, doc):
        return [(t["start"], t["end"], t["speaker"]) for t in doc["turns"]]

    def test_three_chunks_keep_the_same_id_for_the_same_voice(self):
        eng = VoiceEngine()
        doc = diarize(self.media, engine=eng, chunk_seconds=4.0, embeddings=True)
        self.assertTrue(validate_turns(doc).ok, validate_turns(doc).errors)
        self.assertEqual(len(eng.requests), 3)
        self.assertEqual(self.hyp(doc), [(0.5, 2.0, "SPEAKER_00"), (2.0, 3.5, "SPEAKER_01"), (4.5, 6.0, "SPEAKER_01"),
                                         (6.0, 9.0, "SPEAKER_00"), (9.5, 11.5, "SPEAKER_02")])
        self.assertEqual(doc["speaker_count"], 3)
        self.assertLess(der(REFERENCE, self.hyp(doc))["der"], 0.01)
        ch = doc["chunking"]
        self.assertEqual([(c["start"], c["end"]) for c in ch["chunks"]], [(0.0, 4.0), (4.0, 8.0), (8.0, 12.0)])
        links = [(ln["chunk"], ln["speaker"], ln["linked"]) for ln in ch["links"]]
        self.assertEqual(links, [(0, "SPEAKER_00", False), (0, "SPEAKER_01", False), (1, "SPEAKER_01", True),
                                 (1, "SPEAKER_00", True), (2, "SPEAKER_00", True), (2, "SPEAKER_02", False)])
        for ln in ch["links"]:
            if ln["linked"]:
                self.assertGreaterEqual(ln["similarity"], ch["link_threshold"])
                self.assertIsNotNone(ln["margin"])
        self.assertEqual(ch["min_link_similarity"], 1.0)
        self.assertFalse(ch["resumable"])
        self.assertTrue(doc["audio_extraction"]["chunked"])
        self.assertEqual(sorted(doc["speaker_embeddings"]["vectors"]), doc["speakers"])
        # the same voices without chunks: same turns
        self.assertEqual(self.hyp(diarize(self.media, engine=VoiceEngine())), self.hyp(doc))

    def test_the_threshold_decides_and_unlinked_voices_are_reported(self):
        # each chunk's vectors drift a little (a different room, a tired voice): similarity across chunks < 1
        doc = diarize(self.media, engine=DriftingVoiceEngine(), chunk_seconds=4.0)
        self.assertEqual(doc["speaker_count"], 3)
        sims = [ln["similarity"] for ln in doc["chunking"]["links"] if ln["linked"]]
        self.assertTrue(sims and all(0.5 <= x < 0.99 for x in sims), sims)
        self.assertEqual(doc["chunking"]["min_link_similarity"], min(sims))
        strict = diarize(self.media, engine=DriftingVoiceEngine(), chunk_seconds=4.0, link_threshold=0.99, num_speakers=3)
        self.assertEqual(strict["speaker_count"], 6, "nothing linked: every chunk speaker is a new id")
        self.assertFalse(any(ln["linked"] for ln in strict["chunking"]["links"]))
        self.assertTrue(any("more than --num-speakers 3" in w for w in strict["warnings"]))
        self.assertEqual(strict["chunking"]["chunk_parameters"], {"num_speakers": None, "min_speakers": None, "max_speakers": 3})

    def test_resumable_after_an_interruption(self):
        work = os.path.join(self.tmp, "work")
        with self.assertRaises(DiarizationError) as e:
            diarize(self.media, engine=FlakyVoiceEngine(fail_on=2), chunk_seconds=4.0, work_dir=work)
        self.assertEqual(e.exception.code, "ENGINE_FAILED")
        self.assertEqual(sorted(os.listdir(work)), ["chunk_0000.json"])
        eng = VoiceEngine()
        doc = diarize(self.media, engine=eng, chunk_seconds=4.0, work_dir=work)
        self.assertEqual(len(eng.requests), 2, "chunk 0 is reused, chunks 1 and 2 run")
        self.assertEqual([c["resumed"] for c in doc["chunking"]["chunks"]], [True, False, False])
        self.assertEqual(doc["chunking"]["resumed_chunks"], 1)
        self.assertTrue(doc["chunking"]["resumable"])
        self.assertTrue(validate_turns(doc).ok, validate_turns(doc).errors)
        fresh = diarize(self.media, engine=VoiceEngine(), chunk_seconds=4.0)
        self.assertEqual(self.hyp(doc), self.hyp(fresh))
        eng = VoiceEngine()
        again = diarize(self.media, engine=eng, chunk_seconds=4.0, work_dir=work)
        self.assertEqual((len(eng.requests), again["chunking"]["resumed_chunks"]), (0, 3))
        self.assertEqual(self.hyp(again), self.hyp(fresh))
        self.assertNotIn(self.tmp, json.dumps(again))
        with self.assertRaises(DiarizationError) as e:  # other parameters: the saved chunks are not reused or overwritten
            diarize(self.media, engine=VoiceEngine(), chunk_seconds=4.0, work_dir=work, num_speakers=2)
        self.assertEqual(e.exception.code, "INVALID_ARGUMENT")

    def test_refusals(self):
        with self.assertRaises(DiarizationError) as e:
            diarize(self.media, engine=VoiceEngine(embeddings=False), chunk_seconds=4.0)
        self.assertEqual(e.exception.code, "ENGINE_FAILED")
        for kw in ({"work_dir": os.path.join(self.tmp, "w")}, {"chunk_seconds": 0.5}, {"chunk_seconds": 4.0, "link_threshold": 2}):
            with self.assertRaises(DiarizationError) as e:
                diarize(self.media, engine=VoiceEngine(), **kw)
            self.assertEqual(e.exception.code, "INVALID_ARGUMENT", kw)

    def test_validator_checks_chunking(self):
        good = diarize(self.media, engine=VoiceEngine(), chunk_seconds=4.0)
        for name, mutate in {
            "gap between chunks": lambda d: d["chunking"]["chunks"][1].update(start=4.5),
            "short of the end": lambda d: d["chunking"]["chunks"][2].update(end=11.0),
            "unknown speaker": lambda d: d["chunking"]["links"][0].update(speaker="SPEAKER_07"),
            "similarity out of range": lambda d: d["chunking"]["links"][2].update(similarity=1.5),
            "bad chunk index": lambda d: d["chunking"]["links"][0].update(chunk=9),
        }.items():
            d = copy.deepcopy(good)
            mutate(d)
            self.assertFalse(validate_turns(d).ok, name)

    def test_json_schema_accepts_the_chunked_document(self):
        try:
            import jsonschema  # type: ignore
        except ImportError:  # optional, as in test_diarization's schema test
            return
        schema = json.loads((ROOT / "schemas" / "turns.schema.json").read_text(encoding="utf-8"))
        jsonschema.validate(diarize(self.media, engine=VoiceEngine(), chunk_seconds=4.0, embeddings=True), schema)

    def test_cli(self):
        out = os.path.join(self.tmp, "turns.json")
        with mock.patch("diarization_skill.cli.diarize", functools.partial(diarize, engine=VoiceEngine())):
            rc, doc = run_cli("diarize", self.media, "--chunk-seconds", "4", "--work-dir", os.path.join(self.tmp, "w"), "-o", out,
                              "--json")
        self.assertEqual(rc, 0, doc)
        self.assertEqual(len(doc["chunking"]["chunks"]), 3)
        rc, res = run_cli("check", out, "--json")
        self.assertEqual((rc, res["ok"]), (0, True))


if __name__ == "__main__":
    unittest.main()
