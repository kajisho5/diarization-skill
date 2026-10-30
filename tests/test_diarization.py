"""diarization-skill tests.

Unit tests use test-only engines (tests/fake_engine.py) injected as objects. Tests that need ffmpeg/ffprobe,
the transcription-skill checkout (for `transcription check`), or the AI-video-production-OS checkout (for the
registry conformance checks) skip when those are absent. Real-engine tests skip unless DIARIZATION_PIPELINE_DIR
points to a pipeline directory and pyannote.audio is importable.

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import copy
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diarization_skill import __version__, cli  # noqa: E402
from diarization_skill.assign import assign, pick  # noqa: E402
from diarization_skill.contract import ERRORS, TOOLS, contract  # noqa: E402
from diarization_skill.doctor import doctor  # noqa: E402
from diarization_skill.engines import get_engine  # noqa: E402
from diarization_skill.engines.base import EngineRequest  # noqa: E402
from diarization_skill.engines.pyannote_engine import MODEL_ID  # noqa: E402
from diarization_skill.errors import DiarizationError  # noqa: E402
from diarization_skill.media import extraction_argv  # noqa: E402
from diarization_skill.metrics import der  # noqa: E402
from diarization_skill.pipeline import inspect, read_revision  # noqa: E402
from diarization_skill.service import diarize, run_engine, run_process_group  # noqa: E402
from diarization_skill.turns import REQUIRED, TURNS_SCHEMA, build_turns, overlap_regions, validate_turns  # noqa: E402
from fake_engine import ScriptedEngine, ToneEngine, write_tone_wav  # noqa: E402

FP = "sha256:" + "ab" * 32
HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
PYANNOTE = importlib.util.find_spec("pyannote") is not None and importlib.util.find_spec("pyannote.audio") is not None


def _sibling(env: str, name: str):
    for cand in (os.environ.get(env), str(ROOT.parent / name)):
        if cand and os.path.isdir(cand):
            return Path(cand)
    return None


TRANSCRIPTION_DIR = _sibling("TRANSCRIPTION_SKILL_DIR", "transcription-skill")
OS_DIR = _sibling("AI_VIDEO_PRODUCTION_OS_DIR", "AI-video-production-OS")
PIPELINE_DIR = os.environ.get("DIARIZATION_PIPELINE_DIR")


def make_turns(exclusive, regular=None, duration=10.0, **params):
    p = {"num_speakers": None, "min_speakers": None, "max_speakers": None}
    p.update(params)
    return build_turns(exclusive=exclusive, regular=regular, duration=duration,
                       source={"filename": "talk.wav", "fingerprint": FP, "size_bytes": 1000, "media_duration": duration,
                               "has_video": False, "audio_stream": 0},
                       engine={"id": "scripted", "version": "1.0", "model": "test/scripted", "revision": None, "license": "MIT",
                               "attribution": "test", "execution_mode": "local", "device": "cpu"},
                       parameters=p, audio_extraction={"tool": "ffmpeg"}, skill_version=__version__, processing_seconds=0.1)


def make_transcript(segments, duration=10.0, fingerprint=FP):
    """A transcript that passes transcription-skill's validator (transcription-skill/transcript/0.1)."""
    segs = []
    for i, (s, e, words) in enumerate(segments, 1):
        segs.append({"id": f"seg_{i:04d}", "start": s, "end": e, "text": f"segment {i}", "raw_text": f"segment {i}",
                     "confidence": 0.9, "speaker_id": None,
                     "words": None if words is None else [{"start": ws, "end": we, "text": f"w{j}", "confidence": 0.9}
                                                          for j, (ws, we) in enumerate(words)]})
    return {"schema": "transcription-skill/transcript/0.1", "id": "tr_0123456789ab", "asset_id": "asset_abababababababab",
            "language": "en", "language_source": "requested", "language_confidence": None, "duration": duration, "segments": segs,
            "source": {"filename": "talk.wav", "fingerprint": fingerprint, "size_bytes": 1000, "media_duration": duration,
                       "audio_channels": 1, "sample_rate": 16000, "container": "wav", "has_video": False},
            "engine": "faster_whisper", "engine_version": "1.2.1", "created_at": "2026-10-30T00:00:00Z",
            "provenance": {"engine": "faster_whisper", "engine_version": "1.2.1", "execution_mode": "local", "model": "base",
                           "model_version": None, "parameters": {"language": "en"}, "parameters_hash": "0" * 64,
                           "cache_key": "1" * 64, "created_at": "2026-10-30T00:00:00Z", "processing_seconds": 1.0,
                           "skill_version": "0.7.3", "skill": "transcription-skill", "tool": "transcription/transcribe",
                           "language_detection": None, "audio_extraction": None},
            "warnings": []}


def run_cli(*argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(list(argv))
    return rc, json.loads(buf.getvalue())


# ---- turns schema --------------------------------------------------------------------------------
class TurnsTests(unittest.TestCase):
    def test_build_relabels_by_first_appearance_and_validates(self):
        doc = make_turns([(4.0, 6.0, "B"), (0.5, 3.0, "A"), (6.0, 9.0, "A")])
        self.assertEqual(doc["schema"], TURNS_SCHEMA)
        self.assertEqual(doc["turns"], [{"start": 0.5, "end": 3.0, "speaker": "SPEAKER_00"},
                                        {"start": 4.0, "end": 6.0, "speaker": "SPEAKER_01"},
                                        {"start": 6.0, "end": 9.0, "speaker": "SPEAKER_00"}])
        self.assertEqual((doc["speakers"], doc["speaker_count"], doc["speech_seconds"]), (["SPEAKER_00", "SPEAKER_01"], 2, 7.5))
        self.assertFalse(doc["overlap"]["available"])
        self.assertTrue(validate_turns(doc).ok, validate_turns(doc).errors)

    def test_overlap_is_kept_separately(self):
        doc = make_turns([(0, 4, "A"), (4, 8, "B")], regular=[(0, 5, "A"), (3.5, 8, "B")])
        self.assertTrue(doc["overlap"]["available"])
        self.assertEqual(doc["overlap"]["regions"], [{"start": 3.5, "end": 5.0, "speakers": ["SPEAKER_00", "SPEAKER_01"]}])
        self.assertEqual(doc["overlap"]["total_seconds"], 1.5)
        self.assertEqual(len(doc["turns"]), 2)
        self.assertTrue(validate_turns(doc).ok)

    def test_overlap_regions_three_speakers(self):
        r = overlap_regions([{"start": 0, "end": 3, "speaker": "SPEAKER_00"}, {"start": 1, "end": 4, "speaker": "SPEAKER_01"},
                             {"start": 2, "end": 5, "speaker": "SPEAKER_02"}])
        self.assertEqual([(x["start"], x["end"], len(x["speakers"])) for x in r], [(1, 2, 2), (2, 3, 3), (3, 4, 2)])

    def test_engine_results_are_not_repaired_silently(self):
        with self.assertRaises(DiarizationError) as e:
            make_turns([(0, 5, "A"), (4, 8, "B")])  # "exclusive" turns that overlap
        self.assertEqual(e.exception.code, "INVALID_RESULT")
        with self.assertRaises(DiarizationError):
            make_turns([(0, 12.0, "A")])  # 2 s past the end
        doc = make_turns([(0, 10.4, "A")])  # small overrun: clamped, with a warning
        self.assertEqual(doc["turns"][0]["end"], 10.0)
        self.assertTrue(any("clamped" in w for w in doc["warnings"]))
        self.assertEqual(make_turns([])["warnings"], ["no speech turns found"])

    def test_validator_rejects(self):
        good = make_turns([(0, 2, "A"), (2, 4, "B")], regular=[(0, 2, "A"), (2, 4, "B")])
        cases = {
            "named speaker": lambda d: d["turns"][0].update(speaker="Alice"),
            "overlapping turns": lambda d: d["turns"][1].update(start=1.0),
            "past the end": lambda d: d["turns"][1].update(end=11.0),
            "unused speaker": lambda d: (d["speakers"].append("SPEAKER_02"), d.update(speaker_count=3)),
            "num and min": lambda d: d["parameters"].update(num_speakers=2, min_speakers=1),
            "min > max": lambda d: d["parameters"].update(min_speakers=3, max_speakers=2),
            "path filename": lambda d: d["source"].update(filename="/home/me/talk.wav"),
            "absolute path leak": lambda d: d["audio_extraction"].update(where="/tmp/x/audio.wav"),
            "credential key": lambda d: d["engine"].update(token="x"),
            "hf token value": lambda d: d["warnings"].append("hf_" + "a" * 30),
            "no attribution": lambda d: d["engine"].update(attribution=""),
            "missing revision key": lambda d: d["engine"].pop("revision"),
            "bad fingerprint": lambda d: d["source"].update(fingerprint="md5:00"),
            "wrong schema": lambda d: d.update(schema="x"),
        }
        for name, mutate in cases.items():
            d = copy.deepcopy(good)
            mutate(d)
            self.assertFalse(validate_turns(d).ok, name)
        self.assertFalse(validate_turns(good, expected_fingerprint="sha256:" + "0" * 64).ok)

    def test_json_schema_file_matches_validator(self):
        schema = json.loads((ROOT / "schemas" / "turns.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["schema"]["const"], TURNS_SCHEMA)
        self.assertEqual(set(schema["required"]), set(REQUIRED))
        try:
            import jsonschema  # type: ignore
        except ImportError:
            return
        jsonschema.validate(make_turns([(0, 2, "A")], regular=[(0, 2, "A")]), schema)


# ---- assign --------------------------------------------------------------------------------------
class AssignTests(unittest.TestCase):
    TURNS = [(0.0, 4.0, "A"), (4.0, 8.0, "B")]

    def test_segment_level_max_overlap_tie_and_zero(self):
        t = make_turns(self.TURNS)
        tr = make_transcript([(0.5, 3.0, None), (3.0, 4.5, None), (3.5, 4.5, None), (8.5, 9.5, None)])
        out = assign(tr, t)
        self.assertEqual([s["speaker_id"] for s in out["segments"]], ["SPEAKER_00", "SPEAKER_00", None, None])
        self.assertEqual(out["speaker_assignment"]["counts"]["unassigned"], 2)
        self.assertIsNone(tr["segments"][0]["speaker_id"], "input must not be mutated")

    def test_word_level_majority_by_duration(self):
        t = make_turns(self.TURNS)
        tr = make_transcript([(3.0, 6.0, [(3.0, 3.8), (4.0, 4.4), (4.5, 4.8), (5.0, 5.2)]),  # A 0.8s vs B 0.9s -> B
                              (2.0, 6.0, [(3.0, 4.0), (4.0, 5.0)]),                          # 1.0 vs 1.0 -> tie -> null
                              (8.2, 9.0, [(8.2, 9.0)])])                                     # no overlap -> null
        out = assign(tr, t)
        segs = out["segments"]
        self.assertEqual([w["speaker_id"] for w in segs[0]["words"]], ["SPEAKER_00", "SPEAKER_01", "SPEAKER_01", "SPEAKER_01"])
        self.assertEqual(segs[0]["speaker_id"], "SPEAKER_01")
        self.assertIsNone(segs[1]["speaker_id"])
        self.assertEqual(segs[2]["words"][0]["speaker_id"], None)
        self.assertIsNone(segs[2]["speaker_id"])
        c = out["speaker_assignment"]["counts"]
        self.assertEqual((c["by_words"], c["words"], c["words_assigned"]), (3, 7, 6))

    def test_a_word_straddling_a_change_takes_the_longer_side(self):
        out = assign(make_transcript([(3.0, 5.0, [(3.5, 4.2)])]), make_turns(self.TURNS))
        self.assertEqual(out["segments"][0]["words"][0]["speaker_id"], "SPEAKER_00")
        self.assertEqual(pick({"A": 1.0, "B": 1.0}), None)
        self.assertEqual(pick({"A": 0.0}), None)

    def test_provenance_overlap_and_attribution(self):
        t = make_turns(self.TURNS, regular=[(0, 4.5, "A"), (4.0, 8.0, "B")])
        out = assign(make_transcript([(1.0, 2.0, None), (3.8, 4.6, None)]), t)
        sa = out["speaker_assignment"]
        self.assertEqual(sa["tool"], "diarization/assign")
        self.assertEqual(sa["overlap_segment_ids"], ["seg_0002"])
        self.assertEqual(sa["engine"]["attribution"], "test")
        self.assertTrue(sa["source_match"])
        self.assertTrue(any("overlapping speech" in w for w in out["warnings"]))
        # transcription-skill's own provenance is untouched
        self.assertEqual(out["provenance"]["tool"], "transcription/transcribe")

    def test_source_mismatch_and_preset_speakers(self):
        t = make_turns(self.TURNS)
        other = make_transcript([(1.0, 2.0, None)], fingerprint="sha256:" + "cd" * 32)
        with self.assertRaises(DiarizationError) as e:
            assign(other, t)
        self.assertEqual(e.exception.code, "SOURCE_MISMATCH")
        out = assign(other, t, allow_source_mismatch=True)
        self.assertFalse(out["speaker_assignment"]["source_match"])
        preset = make_transcript([(1.0, 2.0, None)])
        preset["segments"][0]["speaker_id"] = "SPEAKER_05"
        with self.assertRaises(DiarizationError) as e:
            assign(preset, t)
        self.assertEqual(e.exception.code, "SPEAKERS_ALREADY_SET")
        self.assertEqual(assign(preset, t, replace=True)["segments"][0]["speaker_id"], "SPEAKER_00")

    def test_rejects_other_documents(self):
        with self.assertRaises(DiarizationError) as e:
            assign({"schema": "x"}, make_turns(self.TURNS))
        self.assertEqual(e.exception.code, "INVALID_TRANSCRIPT")
        bad = make_turns(self.TURNS)
        bad["turns"][0]["speaker"] = "Bob"
        with self.assertRaises(DiarizationError) as e:
            assign(make_transcript([(1, 2, None)]), bad)
        self.assertEqual(e.exception.code, "INVALID_TURNS")


@unittest.skipUnless(TRANSCRIPTION_DIR and (TRANSCRIPTION_DIR / "src" / "transcription_skill").is_dir(),
                     "needs a transcription-skill checkout (TRANSCRIPTION_SKILL_DIR or ../transcription-skill)")
class TranscriptionCheckRoundTrip(unittest.TestCase):
    """ASSIGN's output must pass transcription-skill's own validator, run as `transcription check`."""

    def check(self, path):
        env = dict(os.environ, PYTHONPATH=str(TRANSCRIPTION_DIR / "src"))
        proc = subprocess.run([sys.executable, "-m", "transcription_skill.cli", "check", str(path), "--json"],
                              cwd=str(TRANSCRIPTION_DIR), env=env, capture_output=True, text=True, timeout=120)
        return proc.returncode, json.loads(proc.stdout)

    def test_assign_output_passes_transcription_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr = make_transcript([(0.5, 3.0, [(0.5, 1.5), (1.5, 3.0)]), (3.2, 5.0, [(3.2, 3.9), (4.1, 5.0)]),
                                  (5.5, 7.5, None), (8.5, 9.5, None)])
            tp, kp, op = Path(tmp, "t.json"), Path(tmp, "k.json"), Path(tmp, "o.json")
            tp.write_text(json.dumps(tr), encoding="utf-8")
            kp.write_text(json.dumps(make_turns(self.TURNS, regular=[(0, 4.2, "A"), (4.0, 8.0, "B")])), encoding="utf-8")
            rc, res = self.check(tp)
            self.assertEqual((rc, res["ok"]), (0, True), res)
            rc, doc = run_cli("assign", str(tp), str(kp), "-o", str(op), "--json")
            self.assertEqual(rc, 0, doc)
            rc, res = self.check(op)
            self.assertEqual((rc, res["ok"]), (0, True), res)
            out = json.loads(op.read_text(encoding="utf-8"))
            self.assertEqual([s["speaker_id"] for s in out["segments"]], ["SPEAKER_00", "SPEAKER_01", "SPEAKER_01", None])

    TURNS = [(0.0, 4.0, "A"), (4.0, 8.0, "B")]


# ---- diarize -------------------------------------------------------------------------------------
PLAN = [(0.0, 0.5, None), (0.5, 3.0, 200.0), (3.0, 3.5, None), (3.5, 6.0, 800.0), (6.0, 8.0, 200.0), (8.0, 8.5, None)]
REFERENCE = [(0.5, 3.0, "low"), (3.5, 6.0, "high"), (6.0, 8.0, "low")]


@unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg and ffprobe")
class DiarizeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.media = os.path.join(self.tmp, "two_voices.wav")
        write_tone_wav(self.media, PLAN)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_two_voice_fixture_end_to_end(self):
        eng = ToneEngine()
        doc = diarize(self.media, num_speakers=2, engine=eng)
        self.assertTrue(validate_turns(doc).ok, validate_turns(doc).errors)
        self.assertEqual([(t["start"], t["end"], t["speaker"]) for t in doc["turns"]],
                         [(0.5, 3.0, "SPEAKER_00"), (3.5, 6.0, "SPEAKER_01"), (6.0, 8.0, "SPEAKER_00")])
        self.assertEqual(doc["parameters"], {"num_speakers": 2, "min_speakers": None, "max_speakers": None})
        self.assertEqual(eng.requests[0].num_speakers, 2)
        self.assertTrue(doc["source"]["fingerprint"].startswith("sha256:"))
        self.assertEqual(doc["source"]["filename"], "two_voices.wav")
        self.assertAlmostEqual(doc["source"]["media_duration"], 8.5, places=2)
        self.assertEqual(doc["audio_extraction"]["recipe"], {"channels": 1, "sample_rate": 16000, "codec": "pcm_s16le", "container": "wav"})
        hyp = [(t["start"], t["end"], t["speaker"]) for t in doc["turns"]]
        self.assertLess(der(REFERENCE, hyp)["der"], 0.01)
        self.assertFalse(os.path.exists(eng.requests[0].audio_path), "temporary audio must be removed")

    def test_cli_diarize_check_score(self):
        out = os.path.join(self.tmp, "turns.json")
        import functools
        from unittest import mock
        with mock.patch("diarization_skill.cli.diarize", functools.partial(diarize, engine=ToneEngine())):
            rc, doc = run_cli("diarize", self.media, "--min-speakers", "1", "--max-speakers", "3", "-o", out, "--json")
        self.assertEqual(rc, 0, doc)
        self.assertEqual(doc["speaker_count"], 2)
        rc, res = run_cli("check", out, "--json")
        self.assertEqual((rc, res["ok"]), (0, True))
        rc, doc = run_cli("diarize", self.media, "-o", out, "--json")
        self.assertEqual((rc, doc["error"]["code"]), (2, "OUTPUT_EXISTS"))
        rc, doc = run_cli("diarize", self.media, "-o", self.media, "--overwrite", "--json")
        self.assertEqual((rc, doc["error"]["code"]), (2, "OUTPUT_IS_INPUT"))
        ref = os.path.join(self.tmp, "ref.json")
        Path(ref).write_text(json.dumps({"turns": [{"start": s, "end": e, "speaker": k} for s, e, k in REFERENCE]}), encoding="utf-8")
        rc, sc = run_cli("score", out, ref, "--json")
        self.assertEqual(rc, 0)
        self.assertLess(sc["der"], 0.01)

    def test_timeout_and_engine_failure(self):
        with self.assertRaises(DiarizationError) as e:
            diarize(self.media, engine=ScriptedEngine([(0, 1, "A")], delay=3), timeout=0.5)
        self.assertEqual(e.exception.code, "ENGINE_TIMEOUT")
        with self.assertRaises(DiarizationError) as e:
            diarize(self.media, engine=ScriptedEngine([], fail=RuntimeError("boom")))
        self.assertEqual((e.exception.code, ERRORS[e.exception.code]["exit_code"]), ("ENGINE_FAILED", 4))

    def test_arguments(self):
        for kw in ({"num_speakers": 2, "min_speakers": 1}, {"min_speakers": 3, "max_speakers": 2}, {"num_speakers": 0},
                   {"audio_stream": 1}, {"device": "tpu"}):
            with self.assertRaises(DiarizationError) as e:
                diarize(self.media, engine=ScriptedEngine([]), **kw)
            self.assertEqual(e.exception.code, "INVALID_ARGUMENT", kw)

    def test_real_engine_refuses_without_pipeline_dir(self):
        with self.assertRaises(DiarizationError) as e:
            diarize(self.media, pipeline_dir=os.path.join(self.tmp, "nope"))
        self.assertEqual((e.exception.code, ERRORS["PIPELINE_NOT_FOUND"]["exit_code"]), ("PIPELINE_NOT_FOUND", 3))


class DiarizeWithoutMediaToolsTests(unittest.TestCase):
    def test_missing_input(self):
        rc, doc = run_cli("diarize", os.path.join(tempfile.gettempdir(), "no-such-file.wav"), "--json")
        self.assertEqual((rc, doc["error"]["code"]), (2, "INPUT_NOT_FOUND"))

    def test_extraction_recipe_is_fixed_and_explicit(self):
        argv = extraction_argv("ffmpeg", "in.mp4", "out.wav", 2)
        self.assertEqual(argv[argv.index("-map") + 1], "0:a:2")
        self.assertEqual(argv[argv.index("-ac") + 1], "1")
        self.assertEqual(argv[argv.index("-ar") + 1], "16000")
        self.assertEqual(argv[argv.index("-c:a") + 1], "pcm_s16le")


class WorkerTests(unittest.TestCase):
    """The real worker subprocess path (no weights needed)."""

    def test_process_group_is_killed_on_timeout(self):
        t0 = time.time()
        code, _ = run_process_group([sys.executable, "-c", "import time; time.sleep(30)"], 1.0)
        self.assertEqual(code, 124)
        self.assertLess(time.time() - t0, 15)

    def test_worker_reports_structured_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            for engine_id, request, want in (("nope", {"audio_path": "x.wav"}, "ENGINE_UNAVAILABLE"),
                                             ("pyannote", {"audio_path": "x.wav", "pipeline_dir": tmp}, "PIPELINE_NOT_FOUND")):
                req, out = os.path.join(tmp, "req.json"), os.path.join(tmp, "out.json")
                Path(req).write_text(json.dumps({"engine_id": engine_id, "request": request}), encoding="utf-8")
                code, err = run_process_group([sys.executable, "-m", "diarization_skill.engines.worker", req, out], 60)
                self.assertEqual(code, 1, err)
                self.assertEqual(json.loads(Path(out).read_text(encoding="utf-8"))["error"]["code"], want)

    @unittest.skipIf(PYANNOTE, "pyannote.audio is installed")
    def test_missing_pyannote_is_reported_through_the_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.yaml").write_text("pipeline:\n  name: x\n", encoding="utf-8")
            with self.assertRaises(DiarizationError) as e:
                run_engine(get_engine("pyannote"), EngineRequest(audio_path="x.wav", pipeline_dir=tmp), 60, tmp)
            self.assertEqual((e.exception.code, ERRORS["ENGINE_UNAVAILABLE"]["exit_code"]), ("ENGINE_UNAVAILABLE", 3))


# ---- pipeline dir, doctor, contract --------------------------------------------------------------
class PipelineDirTests(unittest.TestCase):
    def test_states_and_revision(self):
        self.assertEqual(inspect(None)["status"], "MISSING")
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(inspect(os.path.join(tmp, "nope"))["status"], "MISSING")
            self.assertEqual(inspect(tmp)["status"], "INCOMPLETE")
            Path(tmp, "config.yaml").write_text("version: 4.0.0\npipeline:\n  name: pyannote.audio.pipelines.SpeakerDiarization\n"
                                                "  params:\n    x: 1\n", encoding="utf-8")
            os.makedirs(os.path.join(tmp, "segmentation"))
            weights = Path(tmp, "segmentation", "pytorch_model.bin")
            weights.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:00\nsize 5\n")
            info = inspect(tmp)
            self.assertEqual((info["status"], info["lfs_pointers"]), ("INCOMPLETE", ["segmentation/pytorch_model.bin"]))
            weights.write_bytes(b"\0" * 2048)
            sha = "0123456789abcdef0123456789abcdef01234567"
            os.makedirs(os.path.join(tmp, ".git", "refs", "heads"))
            Path(tmp, ".git", "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
            Path(tmp, ".git", "refs", "heads", "main").write_text(sha + "\n", encoding="utf-8")
            info = inspect(tmp)
            self.assertEqual((info["status"], info["revision"], info["pipeline_class"]),
                             ("OK", sha, "pyannote.audio.pipelines.SpeakerDiarization"))
            os.remove(os.path.join(tmp, ".git", "refs", "heads", "main"))
            Path(tmp, ".git", "packed-refs").write_text(f"# pack-refs\n{sha} refs/heads/main\n", encoding="utf-8")
            self.assertEqual(read_revision(tmp), sha)
            snap = os.path.join(tmp, "snapshots", sha)
            os.makedirs(snap)
            self.assertEqual(read_revision(snap), sha)

    def test_doctor_reports_missing_with_instructions(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = doctor(os.path.join(tmp, "nope"))
        self.assertFalse(d["ready"]["diarize"])
        self.assertTrue(d["ready"]["assign"])
        self.assertEqual(d["checks"]["pipeline"]["status"], "MISSING")
        self.assertTrue(any("huggingface.co/pyannote/speaker-diarization-community-1" in s for s in d["checks"]["pipeline"]["install"]))
        for k in ("python", "ffmpeg", "pyannote_audio", "torch", "pipeline"):
            self.assertIn(k, d["checks"])
            if d["checks"][k]["status"] != "OK":
                self.assertTrue(d["checks"][k]["install"], k)
        rc, doc = run_cli("doctor", "--pipeline-dir", os.path.join(tempfile.gettempdir(), "no-such-dir"), "--json")
        self.assertEqual(rc, 3)
        self.assertFalse(doc["ok"])


class ContractTests(unittest.TestCase):
    def test_os_registry_keys_and_provides(self):
        c = contract()
        for k in ("skill_id", "id", "skill", "version", "contract_version", "provides", "dependencies", "not_provided"):
            self.assertIn(k, c)
        self.assertEqual(c["version"], __version__)
        self.assertEqual({p["id"] for p in c["provides"]}, {"speech.diarize", "speech.assign_speakers"})
        tool_ids = {t["tool_id"] for t in TOOLS}
        for p in c["provides"]:
            self.assertEqual(set(p), {"id", "lifecycle", "tool_id"})
            self.assertIn(p["tool_id"], tool_ids)
        for d in c["dependencies"]:
            self.assertIn("version_range", d)
        self.assertFalse(c["execution"]["network"])
        self.assertEqual(c["engines"]["model"], MODEL_ID)
        self.assertIn("CC-BY-4.0", c["engines"]["license"])

    def test_version_comes_from_pyproject(self):
        import re
        want = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M).group(1)
        self.assertEqual(__version__, want)

    def test_cli_contract(self):
        rc, doc = run_cli("contract", "--json")
        self.assertEqual((rc, doc["skill_id"]), (0, "diarization"))

    def test_no_shell_and_no_download_calls_in_the_skill(self):
        for f in (ROOT / "src" / "diarization_skill").rglob("*.py"):
            text = f.read_text(encoding="utf-8")
            self.assertNotIn("shell=True", text, f.name)
            for banned in ("hf_hub_download", "snapshot_download", "urllib.request", "requests.", "use_auth_token"):
                self.assertNotIn(banned, text, f"{f.name}: {banned}")


@unittest.skipUnless(OS_DIR and (OS_DIR / "registry" / "conformance.py").is_file(),
                     "needs an AI-video-production-OS checkout (AI_VIDEO_PRODUCTION_OS_DIR or ../AI-video-production-OS)")
class ConformanceTests(unittest.TestCase):
    def test_registry_checks_pass(self):
        sys.path.insert(0, str(OS_DIR))
        try:
            from registry import conformance as conf
        finally:
            sys.path.remove(str(OS_DIR))
        c = contract()
        results = conf.run_static_checks(c)
        results.append(conf.check_no_unsafe_shell_out(str(ROOT / "src" / "diarization_skill")))
        results.append(conf.check_doctor_status(lambda: doctor()))
        for r in results:
            self.assertEqual(r.status, "PASS", f"{r.check}: {r.detail}")


# ---- real engine (weights needed) ----------------------------------------------------------------
@unittest.skipUnless(PIPELINE_DIR and os.path.isdir(PIPELINE_DIR) and PYANNOTE and HAVE_FFMPEG,
                     "needs DIARIZATION_PIPELINE_DIR, pyannote.audio 4.x and ffmpeg")
class RealEngineTests(unittest.TestCase):
    def media(self):
        env = os.environ.get("DIARIZATION_TEST_MEDIA")
        if env:
            return env
        if TRANSCRIPTION_DIR and (TRANSCRIPTION_DIR / "tests" / "fixtures" / "en_short.wav").is_file():
            return str(TRANSCRIPTION_DIR / "tests" / "fixtures" / "en_short.wav")
        self.skipTest("no speech fixture (DIARIZATION_TEST_MEDIA)")

    def test_real_pyannote_offline(self):
        doc = diarize(self.media(), timeout=1800)
        self.assertTrue(validate_turns(doc).ok)
        self.assertEqual((doc["engine"]["id"], doc["engine"]["model"], doc["engine"]["license"]), ("pyannote", MODEL_ID, "CC-BY-4.0"))
        self.assertTrue(doc["engine"]["version"].startswith("4."))
        self.assertTrue(doc["overlap"]["available"])

    @unittest.skipUnless(os.environ.get("DIARIZATION_DER_MEDIA") and os.environ.get("DIARIZATION_DER_REFERENCE"),
                         "needs DIARIZATION_DER_MEDIA and DIARIZATION_DER_REFERENCE (turns JSON)")
    def test_der_under_threshold(self):
        ref_doc = json.loads(Path(os.environ["DIARIZATION_DER_REFERENCE"]).read_text(encoding="utf-8"))
        ref = [(t["start"], t["end"], t["speaker"]) for t in ref_doc["turns"]]
        doc = diarize(os.environ["DIARIZATION_DER_MEDIA"], timeout=3600)
        score = der(ref, [(t["start"], t["end"], t["speaker"]) for t in doc["turns"]])
        print(f"\nDER {score}")
        self.assertLess(score["der"], float(os.environ.get("DIARIZATION_DER_THRESHOLD", "0.25")))


class MetricsTests(unittest.TestCase):
    def test_der(self):
        ref = [(0, 4, "a"), (4, 8, "b")]
        self.assertEqual(der(ref, [(0, 4, "X"), (4, 8, "Y")])["der"], 0.0)
        self.assertAlmostEqual(der(ref, [(0, 8, "X")])["der"], 0.5, places=3)
        self.assertAlmostEqual(der(ref, [(0, 4, "X")])["der"], 0.5, places=3)
        self.assertAlmostEqual(der(ref, [])["der"], 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
