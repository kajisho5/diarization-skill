"""pyannote.audio 4.x with the `pyannote/speaker-diarization-community-1` pipeline (ADR-001).

The pipeline is loaded from a local directory only (a one-time clone of the gated Hugging Face repo).
The worker sets the Hugging Face libraries offline before this module imports them, and the worker's
environment carries no token, so a run can never download weights. Audio is read here from the fixed
16 kHz mono PCM WAV and handed to the pipeline as a waveform, so pyannote does not decode media itself.
"""
from __future__ import annotations

import os
import sys
import wave
from array import array
from typing import Any, List

from ..errors import DiarizationError
from .base import DiarizationEngine, EngineRequest, EngineResult, Turn

MODEL_ID = "pyannote/speaker-diarization-community-1"
MODEL_LICENSE = "CC-BY-4.0"
MODEL_ATTRIBUTION = ("pyannote/speaker-diarization-community-1 by pyannoteAI (https://huggingface.co/"
                     "pyannote/speaker-diarization-community-1), licensed CC-BY-4.0; run with pyannote.audio (MIT)")
PYANNOTE_RANGE = (4, 5)  # supported major versions: >=4, <5


class PyannoteEngine(DiarizationEngine):
    id = "pyannote"
    description = "pyannote.audio 4.x, speaker-diarization-community-1 pipeline from a local directory"
    model = MODEL_ID
    license = MODEL_LICENSE
    attribution = MODEL_ATTRIBUTION
    execution_mode = "local"
    needs_pipeline_dir = True

    def diarize(self, req: EngineRequest) -> EngineResult:
        if not req.pipeline_dir or not os.path.isfile(os.path.join(req.pipeline_dir, "config.yaml")):
            raise DiarizationError("PIPELINE_NOT_FOUND", "pipeline directory has no config.yaml (set DIARIZATION_PIPELINE_DIR)")
        try:
            import torch
            from pyannote.audio import Pipeline
            from importlib.metadata import version as _dist_version
        except ImportError as exc:
            raise DiarizationError("ENGINE_UNAVAILABLE", f"pyannote.audio is not importable: {exc}",
                                   "pip install 'diarization-skill[pyannote]' (pyannote.audio 4.x, Python >= 3.10)")
        engine_version = _dist_version("pyannote.audio")
        major = _major(engine_version)
        if major is None or not PYANNOTE_RANGE[0] <= major < PYANNOTE_RANGE[1]:
            raise DiarizationError("ENGINE_UNAVAILABLE", f"pyannote.audio {engine_version} is not 4.x", "pip install 'pyannote.audio>=4,<5'")
        pipeline = Pipeline.from_pretrained(req.pipeline_dir)
        if pipeline is None:
            raise DiarizationError("PIPELINE_NOT_FOUND", "pyannote could not load the pipeline from the local directory")
        pipeline.to(torch.device(req.device or "cpu"))
        waveform, sample_rate = _read_wav(req.audio_path, torch)
        kwargs = {k: getattr(req, k) for k in ("num_speakers", "min_speakers", "max_speakers") if getattr(req, k) is not None}
        output = pipeline({"waveform": waveform, "sample_rate": sample_rate}, **kwargs)
        regular = getattr(output, "speaker_diarization", None)
        exclusive = getattr(output, "exclusive_speaker_diarization", None)
        if regular is None or exclusive is None:
            raise DiarizationError("ENGINE_UNAVAILABLE", "pipeline output has no exclusive_speaker_diarization; "
                                   "this needs pyannote.audio 4.x with speaker-diarization-community-1")
        return EngineResult(exclusive=_turns(exclusive), regular=_turns(regular), engine_version=engine_version, model=MODEL_ID,
                            extra={"device": str(req.device or "cpu"), "torch_version": str(getattr(torch, "__version__", ""))})


def _major(v: str) -> Any:
    try:
        return int(v.split(".")[0])
    except ValueError:
        return None


def _read_wav(path: str, torch: Any) -> Any:
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise DiarizationError("ENGINE_FAILED", "engine input must be the fixed mono 16-bit WAV")
        rate = w.getframerate()
        frames = array("h")
        frames.frombytes(w.readframes(w.getnframes()))
    if sys.byteorder != "little":
        frames.byteswap()
    data = torch.tensor(frames, dtype=torch.float32) / 32768.0
    return data.unsqueeze(0), rate


def _turns(annotation: Any) -> List[Turn]:
    out: List[Turn] = []
    for segment, _track, label in annotation.itertracks(yield_label=True):
        out.append((float(segment.start), float(segment.end), str(label)))
    return out
