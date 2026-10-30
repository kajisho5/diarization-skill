"""Test-only engines. They live in tests/ and are never registered in the package's engine registry, so
nothing outside the test suite can select them. They are injected as objects (run in a thread with the
service's timeout), which exercises everything around the engine: probing, the fixed extraction, labels,
overlap, provenance, validation, ASSIGN, the CLI.

ToneEngine is a stand-in "diarizer" for synthetic fixtures: in 100 ms frames, silence is no speaker, and a
voiced frame belongs to the "low" or "high" voice by its zero-crossing rate. It cannot measure the real
engine's DER; it only proves the plumbing is exact on a fixture whose answer is known.
"""
from __future__ import annotations

import math
import struct
import time
import wave
from array import array
from typing import List, Optional, Sequence, Tuple

from diarization_skill.engines.base import DiarizationEngine, EngineRequest, EngineResult

Turn = Tuple[float, float, str]


class ScriptedEngine(DiarizationEngine):
    id = "scripted"
    model = "test/scripted"
    license = "MIT"
    attribution = "test fixture"

    def __init__(self, exclusive: Sequence[Turn], regular: Optional[Sequence[Turn]] = None, delay: float = 0.0,
                 fail: Optional[Exception] = None) -> None:
        self.exclusive, self.regular, self.delay, self.fail = list(exclusive), regular, delay, fail
        self.requests: List[EngineRequest] = []

    def diarize(self, req: EngineRequest) -> EngineResult:
        self.requests.append(req)
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise self.fail
        return EngineResult(exclusive=list(self.exclusive), regular=None if self.regular is None else list(self.regular),
                            engine_version="1.0", model=self.model)


class ToneEngine(DiarizationEngine):
    id = "tone"
    model = "test/tone"
    license = "MIT"
    attribution = "test fixture"
    frame = 0.1

    def __init__(self) -> None:
        self.requests: List[EngineRequest] = []

    def diarize(self, req: EngineRequest) -> EngineResult:
        self.requests.append(req)
        with wave.open(req.audio_path, "rb") as w:
            assert (w.getnchannels(), w.getframerate(), w.getsampwidth()) == (1, 16000, 2), "extraction recipe changed"
            samples = array("h")
            samples.frombytes(w.readframes(w.getnframes()))
        n = int(16000 * self.frame)
        labels: List[Optional[str]] = []
        for i in range(0, len(samples) - n + 1, n):
            chunk = samples[i:i + n]
            rms = math.sqrt(sum(x * x for x in chunk) / n)
            if rms < 500:
                labels.append(None)
                continue
            zc = sum(1 for a, b in zip(chunk, chunk[1:]) if (a < 0) != (b < 0))
            labels.append("high" if zc / self.frame > 1000 else "low")  # 2 crossings per period
        turns: List[Turn] = []
        for i, lab in enumerate(labels):
            if lab is None:
                continue
            s, e = round(i * self.frame, 3), round((i + 1) * self.frame, 3)
            if turns and turns[-1][2] == lab and abs(turns[-1][1] - s) < 1e-6:
                turns[-1] = (turns[-1][0], e, lab)
            else:
                turns.append((s, e, lab))
        return EngineResult(exclusive=turns, regular=list(turns), engine_version="1.0", model=self.model)


def write_tone_wav(path: str, plan: Sequence[Tuple[float, float, Optional[float]]], rate: int = 16000) -> None:
    """plan: (start, end, frequency or None for silence), contiguous from 0."""
    frames = bytearray()
    for start, end, freq in plan:
        for k in range(int(round((end - start) * rate))):
            v = 0 if freq is None else int(12000 * math.sin(2 * math.pi * freq * k / rate))
            frames += struct.pack("<h", v)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
