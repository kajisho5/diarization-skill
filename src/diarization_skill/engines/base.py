"""Engine contract. An engine turns one mono 16 kHz WAV into speaker turns; everything else (probing,
extraction, labels, schema, provenance) is the service's job, so every engine yields the same document."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Turn = Tuple[float, float, str]


@dataclass
class EngineRequest:
    audio_path: str
    pipeline_dir: Optional[str] = None
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    device: str = "cpu"
    sample_rate: int = 16000

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EngineRequest":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__ if k in d})


@dataclass
class EngineResult:
    """`exclusive`: at most one speaker at any time (what ASSIGN uses). `regular`: the engine's normal
    output, which may overlap; None when the engine has no such output."""
    exclusive: List[Turn]
    regular: Optional[List[Turn]]
    engine_version: str
    model: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"exclusive": [list(t) for t in self.exclusive],
                "regular": None if self.regular is None else [list(t) for t in self.regular],
                "engine_version": self.engine_version, "model": self.model, "extra": dict(self.extra)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EngineResult":
        def turns(v: Any) -> List[Turn]:
            return [(float(s), float(e), str(lab)) for s, e, lab in v]
        return cls(exclusive=turns(d["exclusive"]), regular=None if d.get("regular") is None else turns(d["regular"]),
                   engine_version=str(d["engine_version"]), model=str(d["model"]), extra=dict(d.get("extra") or {}))


class DiarizationEngine:
    """Subclass and register in `engines/__init__.py`. `diarize` may be slow and is run in a worker process."""
    id = "base"
    description = ""
    model = ""
    license = ""
    attribution = ""
    execution_mode = "local"
    needs_pipeline_dir = False

    def diarize(self, req: EngineRequest) -> EngineResult:  # pragma: no cover - interface
        raise NotImplementedError
