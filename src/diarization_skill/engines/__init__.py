"""Engine registry. Only engines listed here can be selected or run by the worker; a test engine is
injected as an object by the test suite and is never registered."""
from __future__ import annotations

from typing import Dict, List, Type

from ..errors import DiarizationError
from .base import DiarizationEngine
from .pyannote_engine import PyannoteEngine

DEFAULT_ENGINE = "pyannote"
ENGINES: Dict[str, Type[DiarizationEngine]] = {PyannoteEngine.id: PyannoteEngine}


def engine_ids() -> List[str]:
    return sorted(ENGINES)


def get_engine(engine_id: str) -> DiarizationEngine:
    cls = ENGINES.get(engine_id)
    if cls is None:
        raise DiarizationError("ENGINE_UNAVAILABLE", f"unknown engine {engine_id!r}", {"engines": engine_ids()})
    return cls()
