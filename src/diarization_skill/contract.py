"""Machine-readable contract (`diarization contract --json`), in the shape the AI Video Production OS registry
reads (skill_id, version, contract_version, provides, dependencies, not_provided)."""
from __future__ import annotations

from typing import Any, Dict

from . import __version__
from .assign import METHOD, TRANSCRIPT_SCHEMA
from .engines import DEFAULT_ENGINE, engine_ids
from .engines.pyannote_engine import MODEL_ATTRIBUTION, MODEL_ID, MODEL_LICENSE
from .media import AUDIO_RECIPE
from .pipeline import ENV_PIPELINE_DIR
from .turns import TURNS_SCHEMA

SKILL_ID = "diarization"
CONTRACT_VERSION = "1.0"

TOOLS = [
    {"tool_id": "diarization/diarize", "role": "measurement", "writes_media": False, "mutates_input": False, "deterministic": False,
     "description": "Who spoke when: anonymous speaker turns (SPEAKER_nn, exclusive) of a recording, with overlapping speech "
                    "kept separately, engine/model/revision provenance and the model's CC-BY-4.0 attribution",
     "argv": ["diarize", "<media>", "[--num-speakers N | --min-speakers N --max-speakers N]", "[--audio-stream 0]",
              "[--pipeline-dir <dir>]", "[--device cpu]", "[--timeout 3600]", "[-o <turns.json>]", "[--overwrite]", "--json"]},
    {"tool_id": "diarization/assign", "role": "transformation", "writes_media": False, "mutates_input": False, "deterministic": True,
     "description": "Fill speaker_id in a transcription-skill transcript from speaker turns: longest overlap with an exclusive "
                    "turn per word (else per segment); no overlap or a tie stays null",
     "argv": ["assign", "<transcript.json>", "<turns.json>", "-o <out.json>", "[--replace]", "[--allow-source-mismatch]",
              "[--overwrite]", "--json"]},
    {"tool_id": "diarization/check", "role": "validation", "writes_media": False, "mutates_input": False, "deterministic": True,
     "description": f"Validate a speaker turns document ({TURNS_SCHEMA})", "argv": ["check", "<turns.json>", "--json"]},
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
                     {"id": "speech.assign_speakers", "lifecycle": "EXPERIMENTAL", "tool_id": "diarization/assign"}],
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
        ],
        "not_provided": ["naming or identifying a speaker (ids are anonymous; naming is a human decision the agent records)",
                         "speaker recognition against enrolled voices", "transcription (transcription-skill)",
                         "separating overlapping voices", "downloading model weights (one-time manual, gated step)",
                         "network access of any kind at run time", "subtitle speaker labels (subtitle-skill)"],
        "tools": TOOLS,
        "turns": {"schema": TURNS_SCHEMA, "schema_file": "schemas/turns.schema.json", "provenance": "OBSERVED",
                  "speaker_ids": "anonymous SPEAKER_nn, numbered by first appearance"},
        "assign": {"input_schema": TRANSCRIPT_SCHEMA, "output_schema": TRANSCRIPT_SCHEMA, "method": METHOD,
                   "provenance_key": "speaker_assignment"},
        "engines": {"default": DEFAULT_ENGINE, "registered": engine_ids(), "model": MODEL_ID, "license": MODEL_LICENSE,
                    "attribution": MODEL_ATTRIBUTION},
        "execution": {"canonical_invocation": ["diarization", "<command>", "...", "--json"], "network": False, "shell": False,
                      "engine_process": "worker subprocess, own process group, killed on --timeout",
                      "audio_extraction": dict(AUDIO_RECIPE, stream="0:a:N (explicit)"), "input_is_read_only": True,
                      "overwrites": "never unless --overwrite; -o equal to an input is refused"},
        "errors": ERRORS,
    }
