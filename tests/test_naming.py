"""Speaker naming (ADR-002): consented voiceprints, speaker embeddings in turns, name proposals with evidence.

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diarization_skill import __version__  # noqa: E402
from diarization_skill.errors import DiarizationError  # noqa: E402
from diarization_skill.naming import (PROPOSALS_SCHEMA, ROSTER_SCHEMA, load_roster, normalize_name, propose,  # noqa: E402
                                      validate_proposals)
from diarization_skill.service import diarize  # noqa: E402
from diarization_skill.turns import build_turns, validate_turns  # noqa: E402
from diarization_skill.voices import VOICEPRINT_SCHEMA, enroll, validate_voiceprint, voiceprint_from_turns  # noqa: E402
from fake_engine import VoiceEngine, write_tone_wav  # noqa: E402
from test_diarization import FP, HAVE_FFMPEG, run_cli  # noqa: E402

E = VoiceEngine.embedding
CONSENT = {"given_by": "the person", "date": "2026-11-01", "note": "speaker naming for the test event"}


def turns_with_embeddings(exclusive, embeddings, duration=12.0, fingerprint=FP, model="test/voice", revision=None):
    return build_turns(exclusive=exclusive, regular=list(exclusive), duration=duration,
                       source={"filename": "talk.wav", "fingerprint": fingerprint, "size_bytes": 1000, "media_duration": duration,
                               "has_video": True, "audio_stream": 0},
                       engine={"id": "voice", "version": "1.0", "model": model, "revision": revision, "license": "MIT",
                               "attribution": "test", "execution_mode": "local", "device": "cpu"},
                       parameters={"num_speakers": None, "min_speakers": None, "max_speakers": None},
                       audio_extraction={"tool": "ffmpeg"}, skill_version=__version__, processing_seconds=0.1,
                       embeddings=embeddings, keep_embeddings=True)


def voiceprint(name, freq, **kw):
    sample = turns_with_embeddings([(0.0, 5.0, "x")], {"x": E(freq)}, duration=5.0, fingerprint="sha256:" + "cd" * 32, **kw)
    return voiceprint_from_turns(sample, name=name, consent=CONSENT)


ROSTER = {"schema": ROSTER_SCHEMA, "event": "Test panel", "people": [
    {"id": "chair", "name": "Alice Smith", "role": "chair", "voiceprint": "alice.json"},
    {"id": "speaker1", "name": "Bob Lee", "role": "speaker", "aliases": ["Robert Lee"], "voiceprint": "bob.json"},
    {"id": "speaker2", "name": "山田 太郎", "role": "speaker"}]}
META = {"file": "roster.json", "fingerprint": "sha256:" + "ee" * 32, "people": 3, "voiceprints": []}
RECORDING = [(0.5, 3.0, "a"), (3.5, 6.0, "b"), (6.5, 9.0, "c")]


def text_doc(items, fingerprint=FP):
    """A detection-skill text detections document (only the fields `name` reads)."""
    return {"schema": "detection-skill/detections@1", "kind": "text", "source": {"fingerprint": fingerprint},
            "engine": {"id": "tesseract"},
            "frames": [{"index": i, "t": t, "detections": [{"t": t, "kind": "text", "label": label, "score": 0.9}]}
                       for i, (t, label) in enumerate(items)]}


class EmbeddingsInTurnsTests(unittest.TestCase):
    def test_kept_only_when_asked_and_keyed_by_anonymous_ids(self):
        doc = turns_with_embeddings(RECORDING, {"a": E(200), "b": E(800), "c": E(450)})
        self.assertTrue(validate_turns(doc).ok, validate_turns(doc).errors)
        emb = doc["speaker_embeddings"]
        self.assertEqual((emb["available"], emb["metric"], emb["dimension"]), (True, "cosine", 12))
        self.assertEqual(sorted(emb["vectors"]), ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"])
        plain = build_turns(exclusive=RECORDING, regular=None, duration=12.0, source=doc["source"], engine=doc["engine"],
                            parameters=doc["parameters"], audio_extraction={}, skill_version=__version__, processing_seconds=0,
                            embeddings={"a": E(200)})
        self.assertNotIn("speaker_embeddings", plain)
        none = turns_with_embeddings(RECORDING, None)
        self.assertEqual(none["speaker_embeddings"], {"available": False, "metric": "cosine", "dimension": None, "vectors": {}})
        self.assertTrue(validate_turns(none).ok)

    def test_validator_and_builder_reject_bad_vectors(self):
        good = turns_with_embeddings(RECORDING, {"a": E(200), "b": E(800), "c": E(450)})
        for name, mutate in {
            "unknown speaker": lambda d: d["speaker_embeddings"]["vectors"].update(SPEAKER_09=E(200)),
            "wrong dimension": lambda d: d["speaker_embeddings"]["vectors"]["SPEAKER_00"].pop(),
            "zero vector": lambda d: d["speaker_embeddings"]["vectors"].update(SPEAKER_00=[0.0] * 12),
            "metric": lambda d: d["speaker_embeddings"].update(metric="l2"),
        }.items():
            d = copy.deepcopy(good)
            mutate(d)
            self.assertFalse(validate_turns(d).ok, name)
        with self.assertRaises(DiarizationError) as e:
            turns_with_embeddings(RECORDING, {"a": E(200), "b": [0.0] * 12})
        self.assertEqual(e.exception.code, "INVALID_RESULT")


class VoiceprintTests(unittest.TestCase):
    def test_voiceprint_carries_consent_and_no_path(self):
        vp = voiceprint("Alice Smith", 200)
        self.assertEqual(vp["schema"], VOICEPRINT_SCHEMA)
        self.assertTrue(validate_voiceprint(vp).ok, validate_voiceprint(vp).errors)
        self.assertEqual(vp["consent"]["given_by"], "the person")
        self.assertEqual(vp["sample"]["speakers_found"], 1)
        for name, mutate in {"no consent": lambda d: d.pop("consent"),
                             "consent without date": lambda d: d["consent"].update(date=None),
                             "path": lambda d: d["sample"].update(filename="/home/me/alice.wav")}.items():
            d = copy.deepcopy(vp)
            mutate(d)
            self.assertFalse(validate_voiceprint(d).ok, name)

    def test_refused_without_consent_or_with_two_voices(self):
        one = turns_with_embeddings([(0.0, 5.0, "x")], {"x": E(200)})
        for consent in ({}, {"given_by": "x"}, {"given_by": "x", "date": "1 Nov"}):
            with self.assertRaises(DiarizationError) as e:
                voiceprint_from_turns(one, name="Alice", consent=consent)
            self.assertEqual(e.exception.code, "INVALID_ARGUMENT")
        two = turns_with_embeddings([(0.0, 3.0, "x"), (3.0, 6.0, "y")], {"x": E(200), "y": E(800)})
        with self.assertRaises(DiarizationError) as e:
            voiceprint_from_turns(two, name="Alice", consent=CONSENT)
        self.assertEqual(e.exception.code, "ENROLLMENT_REJECTED")
        short = turns_with_embeddings([(0.0, 1.0, "x")], {"x": E(200)})
        with self.assertRaises(DiarizationError) as e:
            voiceprint_from_turns(short, name="Alice", consent=CONSENT)
        self.assertEqual(e.exception.code, "ENROLLMENT_REJECTED")
        noemb = turns_with_embeddings([(0.0, 5.0, "x")], None)
        with self.assertRaises(DiarizationError) as e:
            voiceprint_from_turns(noemb, name="Alice", consent=CONSENT)
        self.assertEqual(e.exception.code, "ENROLLMENT_REJECTED")


class ProposalTests(unittest.TestCase):
    def setUp(self):
        self.vps = {"chair": voiceprint("Alice Smith", 200), "speaker1": voiceprint("Bob Lee", 800)}

    def rows(self, doc):
        return {r["speaker"]: r for r in doc["speakers"]}

    def test_two_enrolled_voices_map_and_an_unenrolled_voice_stays_anonymous(self):
        turns = turns_with_embeddings(RECORDING, {"a": E(200), "b": E(800), "c": E(450)})
        doc = propose(turns, ROSTER, self.vps, META)
        self.assertEqual(doc["schema"], PROPOSALS_SCHEMA)
        self.assertTrue(validate_proposals(doc).ok, validate_proposals(doc).errors)
        r = self.rows(doc)
        self.assertEqual((r["SPEAKER_00"]["status"], r["SPEAKER_00"]["proposal"]["name"]), ("proposed", "Alice Smith"))
        self.assertEqual((r["SPEAKER_01"]["status"], r["SPEAKER_01"]["proposal"]["name"]), ("proposed", "Bob Lee"))
        self.assertEqual((r["SPEAKER_02"]["status"], r["SPEAKER_02"]["proposal"]), ("anonymous", None))
        self.assertEqual(r["SPEAKER_00"]["proposal"]["basis"], ["voice"])
        self.assertIs(r["SPEAKER_00"]["proposal"]["confirmed"], False)
        self.assertEqual(r["SPEAKER_00"]["evidence"]["voice"][0], {"person_id": "chair", "similarity": 1.0})
        self.assertTrue(r["SPEAKER_02"]["reasons"])
        self.assertTrue(doc["requires_confirmation"])
        self.assertEqual(doc["unmatched_people"], ["speaker2"])
        self.assertEqual(doc["counts"], {"proposed": 2, "anonymous": 1, "conflict": 0, "speakers": 3})

    def test_a_close_voice_is_not_proposed(self):
        # 250 Hz is similar to Alice's 200 Hz (cosine 0.59), under the 0.6 default: no proposal
        turns = turns_with_embeddings(RECORDING, {"a": E(250), "b": E(800), "c": E(450)})
        r = self.rows(propose(turns, ROSTER, self.vps, META))
        self.assertEqual(r["SPEAKER_00"]["status"], "anonymous")
        self.assertIn("under", r["SPEAKER_00"]["reasons"][0])
        # two speakers that both fit Alice: neither is proposed for her
        turns = turns_with_embeddings(RECORDING, {"a": E(200), "b": E(800), "c": E(205)})
        r = self.rows(propose(turns, ROSTER, self.vps, META))
        self.assertEqual((r["SPEAKER_00"]["status"], r["SPEAKER_02"]["status"]), ("anonymous", "anonymous"))
        self.assertEqual(r["SPEAKER_01"]["proposal"]["person_id"], "speaker1")

    def test_lower_third_near_the_first_turn(self):
        turns = turns_with_embeddings(RECORDING, {"a": E(200), "b": E(800), "c": E(450)})
        lt = text_doc([(1.0, "Alice Smith | Chair"), (7.0, "山田太郎　先生"), (30.0, "Robert Lee")])
        doc = propose(turns, ROSTER, self.vps, META, lower_thirds=lt, window_before=2, window_after=3)
        r = self.rows(doc)
        self.assertEqual(r["SPEAKER_00"]["proposal"]["basis"], ["voice", "lower_third"])
        self.assertEqual((r["SPEAKER_02"]["status"], r["SPEAKER_02"]["proposal"]["person_id"]), ("proposed", "speaker2"))
        self.assertEqual(r["SPEAKER_02"]["proposal"]["basis"], ["lower_third"])
        self.assertEqual(r["SPEAKER_02"]["evidence"]["lower_third"][0]["text"], "山田太郎　先生")
        self.assertEqual(r["SPEAKER_01"]["evidence"]["lower_third"], [], "a lower third far from the first turn is not evidence")
        self.assertTrue(validate_proposals(doc).ok, validate_proposals(doc).errors)

    def test_conflicts_are_not_resolved_silently(self):
        turns = turns_with_embeddings(RECORDING, {"a": E(200), "b": E(800), "c": E(450)})
        lt = text_doc([(1.0, "Bob Lee")])  # Alice's voice, Bob's name
        r = self.rows(propose(turns, ROSTER, self.vps, META, lower_thirds=lt, window_before=2, window_after=3))
        self.assertEqual((r["SPEAKER_00"]["status"], r["SPEAKER_00"]["proposal"]), ("conflict", None))
        self.assertEqual(r["SPEAKER_01"]["status"], "conflict", "Bob is claimed by two speakers")
        lt = text_doc([(0.8, "Alice Smith"), (1.2, "山田 太郎")])  # two names in one window
        r = self.rows(propose(turns, ROSTER, {}, META, lower_thirds=lt, window_before=1, window_after=1))
        self.assertEqual(r["SPEAKER_00"]["status"], "anonymous")
        self.assertIn("2 people", r["SPEAKER_00"]["reasons"][0])

    def test_refusals(self):
        turns = turns_with_embeddings(RECORDING, {"a": E(200), "b": E(800), "c": E(450)})
        with self.assertRaises(DiarizationError) as e:
            propose(turns, ROSTER, {"chair": voiceprint("Alice Smith", 200, model="other/model")}, META)
        self.assertEqual(e.exception.code, "EMBEDDING_MISMATCH")
        with self.assertRaises(DiarizationError) as e:
            propose(turns_with_embeddings(RECORDING, {"a": E(200), "b": E(800), "c": E(450)}, revision="r1"), ROSTER,
                    {"chair": voiceprint("Alice Smith", 200, revision="r2")}, META)
        self.assertEqual(e.exception.code, "EMBEDDING_MISMATCH")
        with self.assertRaises(DiarizationError) as e:
            propose(turns, ROSTER, self.vps, META, lower_thirds=text_doc([(1.0, "Alice")], fingerprint="sha256:" + "0" * 64))
        self.assertEqual(e.exception.code, "SOURCE_MISMATCH")
        with self.assertRaises(DiarizationError) as e:
            propose(turns, ROSTER, self.vps, META, lower_thirds=dict(text_doc([]), kind="face"))
        self.assertEqual(e.exception.code, "INVALID_DETECTIONS")
        with self.assertRaises(DiarizationError) as e:
            propose(turns, ROSTER, self.vps, META, min_margin=0)
        self.assertEqual(e.exception.code, "INVALID_ARGUMENT")
        plain = copy.deepcopy(turns)
        del plain["speaker_embeddings"]
        doc = propose(plain, ROSTER, self.vps, META)
        self.assertEqual(doc["counts"]["proposed"], 0)
        self.assertTrue(any("diarize --embeddings" in w for w in doc["warnings"]))

    def test_proposals_validator(self):
        doc = propose(turns_with_embeddings(RECORDING, {"a": E(200), "b": E(800), "c": E(450)}), ROSTER, self.vps, META)
        for name, mutate in {
            "confirmed by the tool": lambda d: d["speakers"][0]["proposal"].update(confirmed=True),
            "no confirmation needed": lambda d: d.update(requires_confirmation=False),
            "named speaker id": lambda d: d["speakers"][0].update(speaker="Alice"),
            "proposal without basis": lambda d: d["speakers"][0]["proposal"].update(basis=[]),
            "anonymous without reason": lambda d: d["speakers"][2].update(reasons=[]),
            "same person twice": lambda d: d["speakers"][1]["proposal"].update(person_id="chair"),
        }.items():
            d = copy.deepcopy(doc)
            mutate(d)
            self.assertFalse(validate_proposals(d).ok, name)

    def test_name_normalization(self):
        self.assertEqual(normalize_name("山田　太郎"), normalize_name("山田太郎"))
        self.assertEqual(normalize_name("ＡＬＩＣＥ  Smith"), normalize_name("alice smith"))
        self.assertEqual(normalize_name("Jean-Luc Picard"), normalize_name("jean luc picard"))


class SchemaFileTests(unittest.TestCase):
    def test_schema_files_match_the_validators(self):
        vps = {"chair": voiceprint("Alice Smith", 200), "speaker1": voiceprint("Bob Lee", 800)}
        turns = turns_with_embeddings(RECORDING, {"a": E(200), "b": E(800), "c": E(450)})
        docs = {"roster": ROSTER, "voiceprint": vps["chair"], "name-proposals": propose(turns, ROSTER, vps, META),
                "turns": turns}
        consts = {"roster": ROSTER_SCHEMA, "voiceprint": VOICEPRINT_SCHEMA, "name-proposals": PROPOSALS_SCHEMA}
        try:
            import jsonschema  # type: ignore
        except ImportError:
            jsonschema = None
        for name, doc in docs.items():
            schema = json.loads((ROOT / "schemas" / f"{name}.schema.json").read_text(encoding="utf-8"))
            if name in consts:
                self.assertEqual(schema["properties"]["schema"]["const"], consts[name])
            if jsonschema is not None:
                jsonschema.validate(doc, schema)


class RosterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def write(self, name, doc):
        p = os.path.join(self.tmp, name)
        Path(p).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        return p

    def test_load_and_refusals(self):
        self.write("alice.json", voiceprint("Alice Smith", 200))
        self.write("bob.json", voiceprint("Bob Lee", 800))
        roster, vps, meta = load_roster(self.write("roster.json", ROSTER))
        self.assertEqual(sorted(vps), ["chair", "speaker1"])
        self.assertEqual([v["file"] for v in meta["voiceprints"]], ["alice.json", "bob.json"])
        self.assertNotIn(self.tmp, json.dumps(meta))
        bad = {"wrong name": dict(ROSTER, people=[dict(ROSTER["people"][0], name="Someone Else")]),
               "duplicate id": dict(ROSTER, people=[ROSTER["people"][2], ROSTER["people"][2]]),
               "schema": dict(ROSTER, schema="x"),
               "empty": dict(ROSTER, people=[])}
        for name, doc in bad.items():
            with self.assertRaises(DiarizationError, msg=name) as e:
                load_roster(self.write("bad.json", doc))
            self.assertEqual(e.exception.code, "INVALID_ROSTER", name)
        vp = voiceprint("Alice Smith", 200)
        del vp["consent"]
        self.write("alice.json", vp)
        with self.assertRaises(DiarizationError) as e:
            load_roster(self.write("roster.json", ROSTER))
        self.assertEqual(e.exception.code, "INVALID_VOICEPRINT")


@unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg and ffprobe")
class NamingEndToEndTests(unittest.TestCase):
    """The roadmap's Done-when fixture: two enrolled voices are named, an unenrolled voice stays anonymous."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def path(self, name):
        return os.path.join(self.tmp, name)

    def test_enroll_diarize_name(self):
        write_tone_wav(self.path("alice.wav"), [(0.0, 0.3, None), (0.3, 4.8, 200.0), (4.8, 5.0, None)])
        write_tone_wav(self.path("bob.wav"), [(0.0, 0.3, None), (0.3, 4.8, 800.0), (4.8, 5.0, None)])
        write_tone_wav(self.path("both.wav"), [(0.0, 2.5, 200.0), (2.5, 5.0, 800.0)])
        write_tone_wav(self.path("panel.wav"), [(0.0, 0.5, None), (0.5, 3.0, 200.0), (3.0, 3.5, None), (3.5, 6.0, 800.0),
                                                (6.0, 6.5, None), (6.5, 9.0, 450.0), (9.0, 11.0, 200.0), (11.0, 11.5, None)])
        from unittest import mock
        eng = VoiceEngine()
        for who, name, wav in (("alice", "Alice Smith", "alice.wav"), ("bob", "Bob Lee", "bob.wav")):
            vp = enroll(self.path(wav), name=name, consent=CONSENT, engine=eng)
            self.assertTrue(validate_voiceprint(vp).ok)
            self.assertEqual(vp["sample"]["filename"], wav)
            Path(self.path(f"{who}.json")).write_text(json.dumps(vp), encoding="utf-8")
        with self.assertRaises(DiarizationError) as e:
            enroll(self.path("both.wav"), name="Alice Smith", consent=CONSENT, engine=eng)
        self.assertEqual(e.exception.code, "ENROLLMENT_REJECTED")
        with self.assertRaises(DiarizationError) as e:
            enroll(self.path("alice.wav"), name="Alice Smith", consent={}, engine=eng)
        self.assertEqual(e.exception.code, "INVALID_ARGUMENT")
        self.assertEqual(len(eng.requests), 3, "no engine run without a consent record")

        turns = diarize(self.path("panel.wav"), engine=eng, embeddings=True)
        self.assertEqual(turns["speaker_count"], 3)
        Path(self.path("turns.json")).write_text(json.dumps(turns), encoding="utf-8")
        Path(self.path("roster.json")).write_text(json.dumps(ROSTER, ensure_ascii=False), encoding="utf-8")
        rc, doc = run_cli("name", self.path("turns.json"), "--roster", self.path("roster.json"), "-o", self.path("names.json"), "--json")
        self.assertEqual(rc, 0, doc)
        by = {r["speaker"]: r for r in doc["speakers"]}
        self.assertEqual(by["SPEAKER_00"]["proposal"]["name"], "Alice Smith")
        self.assertEqual(by["SPEAKER_01"]["proposal"]["name"], "Bob Lee")
        self.assertEqual((by["SPEAKER_02"]["status"], by["SPEAKER_02"]["proposal"]), ("anonymous", None))
        rc, res = run_cli("check", self.path("names.json"), "--json")
        self.assertEqual((rc, res["ok"]), (0, True))
        rc, res = run_cli("check", self.path("alice.json"), "--json")
        self.assertEqual((rc, res["ok"]), (0, True))
        self.assertNotIn("speaker_embeddings", diarize(self.path("panel.wav"), engine=eng), "no voice vectors unless asked")
        # the turns themselves stay anonymous
        self.assertEqual(json.loads(Path(self.path("turns.json")).read_text())["speakers"], ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"])
        with mock.patch("diarization_skill.cli.enroll", lambda *a, **k: enroll(*a, engine=eng, **k)):
            rc, out = run_cli("enroll", self.path("alice.wav"), "--name", "Alice Smith", "--consent-by", "Alice Smith",
                              "--consent-date", "2026-11-01", "-o", self.path("alice2.json"), "--json")
        self.assertEqual(rc, 0, out)
        self.assertEqual(out["consent"]["given_by"], "Alice Smith")


if __name__ == "__main__":
    unittest.main()
