"""Machine-readable contract (`diarization contract --json`), in the shape the AI Video Production OS registry
reads (skill_id, version, contract_version, provides, dependencies, not_provided)."""
from __future__ import annotations

from typing import Any, Dict

from . import __version__
from .assign import METHOD, TRANSCRIPT_SCHEMA
from .engines import DEFAULT_ENGINE, engine_ids
from .engines.pyannote_engine import MODEL_ATTRIBUTION, MODEL_ID, MODEL_LICENSE
from .media import AUDIO_RECIPE
from .naming import PROPOSALS_SCHEMA, ROSTER_SCHEMA
from .pipeline import ENV_PIPELINE_DIR
from .turns import TURNS_SCHEMA
from .voices import VOICEPRINT_SCHEMA

SKILL_ID = "diarization"
CONTRACT_VERSION = "1.0"

TOOLS = [
    {"tool_id": "diarization/diarize", "role": "measurement", "writes_media": False, "mutates_input": False, "deterministic": False,
     "description": "Who spoke when: anonymous speaker turns (SPEAKER_nn, exclusive) of a recording, with overlapping speech "
                    "kept separately, engine/model/revision provenance and the model's CC-BY-4.0 attribution",
     "argv": ["diarize", "<media>", "[--num-speakers N | --min-speakers N --max-speakers N]", "[--audio-stream 0]",
              "[--embeddings]", "[--pipeline-dir <dir>]", "[--device cpu]", "[--timeout 3600]", "[-o <turns.json>]", "[--overwrite]", "--json"]},
    {"tool_id": "diarization/assign", "role": "transformation", "writes_media": False, "mutates_input": False, "deterministic": True,
     "description": "Fill speaker_id in a transcription-skill transcript from speaker turns: longest overlap with an exclusive "
                    "turn per word (else per segment); no overlap or a tie stays null",
     "argv": ["assign", "<transcript.json>", "<turns.json>", "-o <out.json>", "[--replace]", "[--allow-source-mismatch]",
              "[--overwrite]", "--json"]},
    {"tool_id": "diarization/enroll", "role": "measurement", "writes_media": False, "mutates_input": False, "deterministic": False,
     "description": f"Voiceprint ({VOICEPRINT_SCHEMA}) of one person from a sample of their voice alone, with the consent record "
                    "(who, when); refused without consent or when the sample holds more than one voice",
     "argv": ["enroll", "<sample>", "--name <name>", "--consent-by <who>", "--consent-date YYYY-MM-DD", "[--consent-note <text>]",
              "[--person-id <id>]", "[--role <text>]", "[--pipeline-dir <dir>]", "-o <voice.json>", "[--overwrite]", "--json"]},
    {"tool_id": "diarization/name", "role": "measurement", "writes_media": False, "mutates_input": False, "deterministic": True,
     "description": f"Name proposals ({PROPOSALS_SCHEMA}) for anonymous speakers from the caller's roster: voiceprint similarity "
                    "and lower-third OCR text near a speaker's first turn, each with its evidence; never confirms a name",
     "argv": ["name", "<turns.json>", "--roster <roster.json>", "[--lower-thirds <text.json>]", "[--min-similarity 0.6]",
              "[--min-margin 0.1]", "-o <names.json>", "[--overwrite]", "--json"]},
    {"tool_id": "diarization/check", "role": "validation", "writes_media": False, "mutates_input": False, "deterministic": True,
     "description": f"Validate a speaker turns ({TURNS_SCHEMA}), voiceprint or name-proposals document",
     "argv": ["check", "<file.json>", "--json"]},
    {"tool_id": "diarization/score", "role": "measurement", "writes_media": False, "mutates_input": False, "deterministic": True,
     "description": "Diarization error rate of a turns document against a reference turns document (frame-based, no collar)",
     "argv": ["score", "<turns.json>", "<reference.json>", "--json"]},
]

ERRORS = {
    "INPUT_NOT_FOUND": {"exit_code": 2, "retryable": False},
    "INVALID_ARGUMENT": {"exit_code": 2, "retryable": False},
    "UNSUPPORTED_MEDIA": {"exit_code": 2, "retryable": False},
    "INVALID_TRANSCRIPT": {"exit_code": 2, "retryable": False},
    "INVALID_TURNS": {"exit_code": 2, "retryable": False},
    "SOURCE_MISMATCH": {"exit_code": 2, "retryable": False},
    "SPEAKERS_ALREADY_SET": {"exit_code": 2, "retryable": False},
    "INVALID_ROSTER": {"exit_code": 2, "retryable": False},
    "INVALID_VOICEPRINT": {"exit_code": 2, "retryable": False},
    "INVALID_DETECTIONS": {"exit_code": 2, "retryable": False},
    "EMBEDDING_MISMATCH": {"exit_code": 2, "retryable": False},
    "ENROLLMENT_REJECTED": {"exit_code": 2, "retryable": False},
    "OUTPUT_IS_INPUT": {"exit_code": 2, "retryable": False},
    "OUTPUT_EXISTS": {"exit_code": 2, "retryable": False},
    "FFMPEG_NOT_FOUND": {"exit_code": 3, "retryable": False},
    "ENGINE_UNAVAILABLE": {"exit_code": 3, "retryable": False},
    "PIPELINE_NOT_FOUND": {"exit_code": 3, "retryable": False},
    "ENGINE_FAILED": {"exit_code": 4, "retryable": False},
    "ENGINE_TIMEOUT": {"exit_code": 4, "retryable": True},
    "INVALID_RESULT": {"exit_code": 4, "retryable": False},
}


def contract() -> Dict[str, Any]:
    return {
        "skill_id": SKILL_ID,
        "id": SKILL_ID,
        "skill": "diarization-skill",
        "version": __version__,
        "contract_version": CONTRACT_VERSION,
        "provides": [{"id": "speech.diarize", "lifecycle": "EXPERIMENTAL", "tool_id": "diarization/diarize"},
                     {"id": "speech.assign_speakers", "lifecycle": "EXPERIMENTAL", "tool_id": "diarization/assign"},
                     {"id": "speech.enroll_voice", "lifecycle": "EXPERIMENTAL", "tool_id": "diarization/enroll"},
                     {"id": "speech.propose_speaker_names", "lifecycle": "EXPERIMENTAL", "tool_id": "diarization/name"}],
        "dependencies": [
            {"id": "pyannote.audio", "kind": "python-package", "required": True, "for": ["diarization/diarize"],
             "version_range": ">=4.0.0 <5.0.0", "install": "pip install 'diarization-skill[pyannote]'"},
            {"id": MODEL_ID, "kind": "model-weights", "required": True, "for": ["diarization/diarize"],
             "version_range": "any snapshot (revision recorded in every result)", "license": MODEL_LICENSE,
             "gated": True, "located_by": ["--pipeline-dir", f"${ENV_PIPELINE_DIR}"], "downloaded_at_run_time": False},
            {"id": "ffmpeg", "kind": "binary", "required": True, "for": ["diarization/diarize"], "version_range": ">=4.0",
             "located_by": ["PATH"]},
            {"id": TRANSCRIPT_SCHEMA, "kind": "schema", "owner": "transcription-skill", "required": False,
             "for": ["diarization/assign"], "version_range": ">=0.1 <0.2"},
            {"id": "detection-skill/detections@1, detection-skill/tracks@1 (kind text)", "kind": "schema", "owner": "detection-skill",
             "required": False, "for": ["diarization/name"], "version_range": "@1"},
        ],
        "not_provided": ["confirming a speaker's name (`name` only proposes, with evidence; a person confirms and the agent records it)",
                         "identifying anyone outside the caller's roster", "enrolling a voice without a consent record",
                         "transcription (transcription-skill)", "OCR of lower thirds (detection-skill `text`)",
                         "separating overlapping voices", "downloading model weights (one-time manual, gated step)",
                         "network access of any kind at run time", "subtitle speaker labels (subtitle-skill)"],
        "tools": TOOLS,
        "turns": {"schema": TURNS_SCHEMA, "schema_file": "schemas/turns.schema.json", "provenance": "OBSERVED",
                  "speaker_ids": "anonymous SPEAKER_nn, numbered by first appearance"},
        "assign": {"input_schema": TRANSCRIPT_SCHEMA, "output_schema": TRANSCRIPT_SCHEMA, "method": METHOD,
                   "provenance_key": "speaker_assignment"},
        "naming": {"roster_schema": ROSTER_SCHEMA, "voiceprint_schema": VOICEPRINT_SCHEMA, "proposals_schema": PROPOSALS_SCHEMA,
                   "evidence": ["voice (cosine similarity to a consented voiceprint)", "lower_third (detection-skill OCR text)"],
                   "requires_confirmation": True, "speaker_embeddings": "only with diarize --embeddings"},
        "engines": {"default": DEFAULT_ENGINE, "registered": engine_ids(), "model": MODEL_ID, "license": MODEL_LICENSE,
                    "attribution": MODEL_ATTRIBUTION},
        "execution": {"canonical_invocation": ["diarization", "<command>", "...", "--json"], "network": False, "shell": False,
                      "engine_process": "worker subprocess, own process group, killed on --timeout",
                      "audio_extraction": dict(AUDIO_RECIPE, stream="0:a:N (explicit)"), "input_is_read_only": True,
                      "overwrites": "never unless --overwrite; -o equal to an input is refused"},
        "errors": ERRORS,
    }
