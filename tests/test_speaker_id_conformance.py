"""Conformance fixtures for the speaker-id assignment rule (docs/speaker_id_rule.md, ADR-004).

ASSIGN must pass every case in conformance/speaker_id/. The fixtures are published for transcription-skill, whose
speaker merge is a copy of the rule; when a transcription-skill checkout is found (TRANSCRIPTION_SKILL_DIR or
../transcription-skill) its merge runs the same cases here too.

    python -m unittest tests/test_speaker_id_conformance.py -v
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diarization_skill.assign import METHOD, TRANSCRIPT_SCHEMA, assign  # noqa: E402
from diarization_skill.turns import TURNS_SCHEMA, validate_turns  # noqa: E402

FIXTURES = ROOT / "conformance" / "speaker_id"
MANIFEST_SCHEMA = "diarization-skill/speaker-id-conformance@1"
CASE_SCHEMA = "diarization-skill/speaker-id-case@1"
# fixture_version -> sha256 of manifest.json. The manifest holds the sha256 of every case file, so changing a case
# without a new fixture_version (and a new line here) fails. Consumers pin the same pair.
FIXTURE_PINS = {"1": "87d27cc76914fa4b5a1d24281e04330d3577c28fbd51d88980e35f35c181115f"}
# The situations the roadmap step asks the fixtures to cover (docs/speaker_id_rule.md).
REQUIRED_COVERAGE = {"overlap", "word_across_turn_boundary", "gap_no_turn", "tie"}


def _sibling(env: str, name: str):
    for cand in (os.environ.get(env), str(ROOT.parent / name)):
        if cand and os.path.isdir(cand):
            return Path(cand)
    return None


TRANSCRIPTION_DIR = _sibling("TRANSCRIPTION_SKILL_DIR", "transcription-skill")


def load_manifest():
    return json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))


def load_cases():
    return [json.loads((FIXTURES / c["file"]).read_text(encoding="utf-8")) for c in load_manifest()["cases"]]


def observed(doc):
    """The parts of an assigned transcript the fixtures fix: speaker per segment and word, and the overlap list."""
    segs = [{"id": s["id"], "speaker_id": s["speaker_id"],
             "words": None if s.get("words") is None else [w["speaker_id"] for w in s["words"]]} for s in doc["segments"]]
    return {"segments": segs, "overlap_segment_ids": doc["speaker_assignment"]["overlap_segment_ids"]}


class FixtureIntegrityTests(unittest.TestCase):
    def test_manifest_is_pinned_and_every_case_matches_its_hash(self):
        raw = (FIXTURES / "manifest.json").read_bytes()
        m = json.loads(raw.decode("utf-8"))
        self.assertEqual(m["schema"], MANIFEST_SCHEMA)
        self.assertEqual((m["method"], m["transcript_schema"], m["turns_schema"]), (METHOD, TRANSCRIPT_SCHEMA, TURNS_SCHEMA))
        self.assertTrue((ROOT / m["rule"]).is_file())
        self.assertEqual(FIXTURE_PINS.get(m["fixture_version"]), hashlib.sha256(raw).hexdigest(),
                         "fixtures changed: bump fixture_version in the manifest and every case, and pin it in FIXTURE_PINS")
        listed = sorted(c["file"] for c in m["cases"])
        self.assertEqual(listed, sorted(f"cases/{p.name}" for p in (FIXTURES / "cases").glob("*.json")))
        for c in m["cases"]:
            data = (FIXTURES / c["file"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), c["sha256"], c["file"])
            case = json.loads(data.decode("utf-8"))
            self.assertEqual((case["schema"], case["fixture_version"], case["id"], case["covers"]),
                             (CASE_SCHEMA, m["fixture_version"], c["id"], c["covers"]))

    def test_cases_cover_the_required_situations_and_are_valid_inputs(self):
        covered = set()
        for case in load_cases():
            covered.update(case["covers"])
            rep = validate_turns(case["turns"])
            self.assertTrue(rep.ok, (case["id"], rep.errors))
            self.assertTrue(all(s["speaker_id"] is None for s in case["transcript"]["segments"]), case["id"])
            ids = [s["id"] for s in case["transcript"]["segments"]]
            self.assertEqual([s["id"] for s in case["expected"]["segments"]], ids, case["id"])
            for seg, exp in zip(case["transcript"]["segments"], case["expected"]["segments"]):
                self.assertEqual(exp["words"] is None, not seg["words"], (case["id"], seg["id"]))
                if seg["words"]:
                    self.assertEqual(len(exp["words"]), len(seg["words"]), (case["id"], seg["id"]))
        self.assertLessEqual(REQUIRED_COVERAGE, covered)


class AssignConformanceTests(unittest.TestCase):
    def test_assign_passes_every_case(self):
        for case in load_cases():
            with self.subTest(case=case["id"]):
                out = assign(case["transcript"], case["turns"])
                self.assertEqual(observed(out), case["expected"])
                self.assertEqual(out["speaker_assignment"]["method"], METHOD)


@unittest.skipUnless(TRANSCRIPTION_DIR and (TRANSCRIPTION_DIR / "src" / "transcription_skill" / "speakers.py").is_file(),
                     "needs a transcription-skill checkout (TRANSCRIPTION_SKILL_DIR or ../transcription-skill)")
class TranscriptionMergeConformanceTests(unittest.TestCase):
    """The copy of the rule in transcription-skill (`transcription speakers`) gives the same answers."""

    def test_transcription_speakers_passes_every_case(self):
        env = dict(os.environ, PYTHONPATH=str(TRANSCRIPTION_DIR / "src"), PYTHONUTF8="1")
        for case in load_cases():
            with self.subTest(case=case["id"]), tempfile.TemporaryDirectory() as tmp:
                tp, kp, op = Path(tmp, "t.json"), Path(tmp, "k.json"), Path(tmp, "o.json")
                tp.write_text(json.dumps(case["transcript"]), encoding="utf-8")
                kp.write_text(json.dumps(case["turns"]), encoding="utf-8")
                proc = subprocess.run([sys.executable, "-m", "transcription_skill.cli", "speakers", str(tp), "--turns", str(kp),
                                       "-o", str(op), "--json"], cwd=str(TRANSCRIPTION_DIR), env=env, capture_output=True,
                                      text=True, timeout=120)
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                self.assertEqual(observed(json.loads(op.read_text(encoding="utf-8"))), case["expected"])


if __name__ == "__main__":
    unittest.main()
