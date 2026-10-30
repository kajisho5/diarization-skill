"""Speaker turns: the document DIARIZE writes (`diarization-skill/turns@1`), how it is built from an engine
result, and its validator. `schemas/turns.schema.json` is the same contract as a JSON Schema; this module is
the executable form and is what `diarization check` runs.

Rules kept by construction and checked here:
- `turns` are exclusive (one speaker at a time), ordered, non-overlapping, inside the media duration.
- speaker ids are anonymous `SPEAKER_nn`, numbered by first appearance; they name nobody.
- overlapping speech is kept separately under `overlap` (the engine's regular output and the regions
  where two or more speakers talk), never mixed into `turns`.
- provenance names the engine, its version, the model and the snapshot revision, with the model licence
  and attribution; no path, command line or credential appears anywhere.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .errors import DiarizationError

TURNS_SCHEMA = "diarization-skill/turns@1"
SPEAKER_RE = re.compile(r"^SPEAKER_\d{2,}$")
FINGERPRINT_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
END_TOLERANCE = 0.5     # seconds a turn may sit past the media duration in a document made elsewhere
OVERLAP_TOLERANCE = 0.01
CLAMP_LIMIT = 1.0       # engine overrun past the media end that is clamped (with a warning) instead of rejected
DIGITS = 3

REQUIRED = ("schema", "source", "engine", "parameters", "speakers", "speaker_count", "turns", "overlap", "audio_extraction",
            "skill", "skill_version", "tool", "created_at", "processing_seconds", "warnings")
REQUIRED_SOURCE = ("filename", "fingerprint", "size_bytes", "media_duration", "audio_stream")
REQUIRED_ENGINE = ("id", "version", "model", "revision", "license", "attribution", "execution_mode")
REQUIRED_PARAMETERS = ("num_speakers", "min_speakers", "max_speakers")
FORBIDDEN_KEYS = {"command", "argv", "cmd", "shell", "api_key", "apikey", "token", "hf_token", "secret", "password",
                  "authorization", "access_key", "secret_key", "credential", "credentials"}
CREDENTIAL_PATTERNS = [re.compile(r"\bhf_[A-Za-z0-9]{20,}"), re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
                       re.compile(r"\bAKIA[0-9A-Z]{16}\b"), re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}"),
                       re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")]
ABS_PATH_PATTERNS = [re.compile(r"^(/|~/)[^\s]+/[^\s]*"), re.compile(r"^[A-Za-z]:\\[^\s]+")]


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---- building ----------------------------------------------------------------------------------
def speaker_map(*turn_lists: Optional[Sequence[Tuple[float, float, str]]]) -> Dict[str, str]:
    """Engine label -> SPEAKER_nn, numbered by first appearance (exclusive turns first, by start time)."""
    mapping: Dict[str, str] = {}
    for turns in turn_lists:
        for _s, _e, label in sorted(turns or [], key=lambda t: (t[0], t[1])):
            if label not in mapping:
                mapping[label] = f"SPEAKER_{len(mapping):02d}"
    return mapping


def _clean(turns: Sequence[Tuple[float, float, str]], duration: float, what: str,
           warnings: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    clamped = dropped = 0
    for s, e, label in sorted(turns, key=lambda t: (t[0], t[1])):
        if not (_num(s) and _num(e)):
            raise DiarizationError("INVALID_RESULT", f"engine returned a non-finite time in {what}")
        if e > duration:
            if e - duration > CLAMP_LIMIT:
                raise DiarizationError("INVALID_RESULT", f"engine {what} end {e:.3f}s is past the media end {duration:.3f}s")
            e, clamped = duration, clamped + 1
        s = max(0.0, s) if s > -OVERLAP_TOLERANCE else s
        if s < 0:
            raise DiarizationError("INVALID_RESULT", f"engine {what} start {s:.3f}s is negative")
        s, e = round(s, DIGITS), round(e, DIGITS)
        if e <= s:
            dropped += 1
            continue
        out.append({"start": s, "end": e, "speaker": str(label)})
    if clamped:
        warnings.append(f"{clamped} {what} end(s) ran past the media end by at most {CLAMP_LIMIT:g}s and were clamped to it")
    if dropped:
        warnings.append(f"{dropped} {what} shorter than {10 ** -DIGITS:g}s dropped")
    return out


def overlap_regions(turns: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Intervals where two or more speakers are active at once (sweep over turn boundaries)."""
    events = sorted({t["start"] for t in turns} | {t["end"] for t in turns})
    regions: List[Dict[str, Any]] = []
    for a, b in zip(events, events[1:]):
        active = sorted({t["speaker"] for t in turns if t["start"] <= a and t["end"] >= b})
        if len(active) < 2:
            continue
        if regions and regions[-1]["end"] == a and regions[-1]["speakers"] == active:
            regions[-1]["end"] = b
        else:
            regions.append({"start": a, "end": b, "speakers": active})
    for r in regions:
        r["end"] = round(r["end"], DIGITS)
    return regions


def build_turns(*, exclusive: Sequence[Tuple[float, float, str]], regular: Optional[Sequence[Tuple[float, float, str]]],
                duration: float, source: Dict[str, Any], engine: Dict[str, Any], parameters: Dict[str, Any],
                audio_extraction: Dict[str, Any], skill_version: str, processing_seconds: float,
                warnings: Optional[List[str]] = None) -> Dict[str, Any]:
    warnings = list(warnings or [])
    turns = _clean(exclusive, duration, "turn", warnings)
    reg = None if regular is None else _clean(regular, duration, "overlap turn", warnings)
    # numbered after cleaning, so every listed speaker has at least one kept turn and numbers have no gaps
    mapping = speaker_map([(t["start"], t["end"], t["speaker"]) for t in turns],
                          None if reg is None else [(t["start"], t["end"], t["speaker"]) for t in reg])
    for t in turns + (reg or []):
        t["speaker"] = mapping[t["speaker"]]
    for prev, cur in zip(turns, turns[1:]):
        if cur["start"] < prev["end"] - OVERLAP_TOLERANCE:
            raise DiarizationError("INVALID_RESULT", "engine's exclusive turns overlap; refusing to return them as exclusive",
                                   {"at": cur["start"]})
    if reg is None:
        overlap: Dict[str, Any] = {"available": False, "turns": None, "regions": [], "total_seconds": None}
    else:
        regions = overlap_regions(reg)
        overlap = {"available": True, "turns": reg, "regions": regions,
                   "total_seconds": round(sum(r["end"] - r["start"] for r in regions), DIGITS)}
    speakers = sorted(mapping.values())
    if not turns:
        warnings.append("no speech turns found")
    return {
        "schema": TURNS_SCHEMA,
        "source": source,
        "engine": engine,
        "parameters": parameters,
        "speakers": speakers,
        "speaker_count": len(speakers),
        "speech_seconds": round(sum(t["end"] - t["start"] for t in turns), DIGITS),
        "turns": turns,
        "overlap": overlap,
        "audio_extraction": audio_extraction,
        "skill": "diarization-skill",
        "skill_version": skill_version,
        "tool": "diarization/diarize",
        "created_at": utc_now(),
        "processing_seconds": round(processing_seconds, 3),
        "warnings": warnings,
    }


# ---- validation --------------------------------------------------------------------------------
class ValidationReport:
    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "errors": list(self.errors), "warnings": list(self.warnings)}


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and v not in (float("inf"), float("-inf"))


def _opt_count(v: Any) -> bool:
    return v is None or (isinstance(v, int) and not isinstance(v, bool) and v >= 1)


def _check_turn_list(turns: Any, where: str, duration: Optional[float], speakers: List[str], exclusive: bool,
                     rep: ValidationReport) -> None:
    if not isinstance(turns, list):
        rep.errors.append(f"{where} must be a list")
        return
    prev: Optional[Dict[str, Any]] = None
    for i, t in enumerate(turns):
        w = f"{where}[{i}]"
        if not isinstance(t, dict) or any(k not in t for k in ("start", "end", "speaker")):
            rep.errors.append(f"{w}: must be an object with start, end, speaker")
            continue
        s, e, spk = t["start"], t["end"], t["speaker"]
        if not _num(s) or not _num(e):
            rep.errors.append(f"{w}: start/end must be finite numbers")
            continue
        if s < 0:
            rep.errors.append(f"{w}: start {s} < 0")
        if e <= s:
            rep.errors.append(f"{w}: end {e} must be > start {s}")
        if duration is not None and e > duration + END_TOLERANCE:
            rep.errors.append(f"{w}: end {e} exceeds media duration {duration} (+{END_TOLERANCE}s)")
        if not isinstance(spk, str) or not SPEAKER_RE.match(spk):
            rep.errors.append(f"{w}: speaker must be an anonymous id SPEAKER_nn, got {spk!r}")
        elif spk not in speakers:
            rep.errors.append(f"{w}: speaker {spk} is not in speakers")
        if prev is not None and _num(prev["start"]) and _num(prev["end"]):
            if s < prev["start"]:
                rep.errors.append(f"{w}: not ordered by start")
            if exclusive and s < prev["end"] - OVERLAP_TOLERANCE:
                rep.errors.append(f"{w}: overlaps the previous turn; turns must be exclusive (overlap goes under 'overlap')")
        prev = t


def validate_turns(doc: Any, expected_fingerprint: Optional[str] = None) -> ValidationReport:
    rep = ValidationReport()
    if not isinstance(doc, dict):
        rep.errors.append("turns document must be a JSON object")
        return rep
    missing = [k for k in REQUIRED if k not in doc]
    if missing:
        rep.errors.append(f"missing required field(s): {', '.join(missing)}")
        return rep
    if doc["schema"] != TURNS_SCHEMA:
        rep.errors.append(f"schema must be {TURNS_SCHEMA}, got {doc['schema']!r}")
    if not isinstance(doc["created_at"], str) or not ISO_RE.match(doc["created_at"]):
        rep.errors.append("created_at must be an ISO-8601 UTC timestamp (YYYY-MM-DDTHH:MM:SSZ)")
    if not _num(doc["processing_seconds"]) or doc["processing_seconds"] < 0:
        rep.errors.append("processing_seconds must be a non-negative number")
    if not isinstance(doc["warnings"], list) or not all(isinstance(w, str) for w in doc["warnings"]):
        rep.errors.append("warnings must be a list of strings")
    for k in ("skill", "skill_version", "tool"):
        if not isinstance(doc[k], str) or not doc[k]:
            rep.errors.append(f"{k} must be a non-empty string")
    if isinstance(doc["tool"], str) and not doc["tool"].startswith("diarization/"):
        rep.errors.append("tool must be a diarization/* tool name")

    duration: Optional[float] = None
    src = doc["source"]
    if not isinstance(src, dict):
        rep.errors.append("source must be an object")
    else:
        for k in REQUIRED_SOURCE:
            if k not in src:
                rep.errors.append(f"source.{k} is required")
        fp = src.get("fingerprint")
        if not isinstance(fp, str) or not FINGERPRINT_RE.match(fp):
            rep.errors.append("source.fingerprint must be 'sha256:<64 hex>'")
        elif expected_fingerprint and fp != expected_fingerprint:
            rep.errors.append("source.fingerprint does not match the input file")
        fn = src.get("filename")
        if not isinstance(fn, str) or not fn:
            rep.errors.append("source.filename must be a non-empty string")
        elif "/" in fn or "\\" in fn:
            rep.errors.append("source.filename must be a bare file name, not a path")
        sb = src.get("size_bytes")
        if not isinstance(sb, int) or isinstance(sb, bool) or sb < 0:
            rep.errors.append("source.size_bytes must be a non-negative integer")
        md = src.get("media_duration")
        if not _num(md) or md <= 0:
            rep.errors.append("source.media_duration must be a positive number")
        else:
            duration = float(md)
        ast = src.get("audio_stream")
        if not isinstance(ast, int) or isinstance(ast, bool) or ast < 0:
            rep.errors.append("source.audio_stream must be a non-negative integer")

    eng = doc["engine"]
    if not isinstance(eng, dict):
        rep.errors.append("engine must be an object")
    else:
        for k in REQUIRED_ENGINE:
            if k not in eng:
                rep.errors.append(f"engine.{k} is required")
        for k in ("id", "version", "model", "license", "attribution"):
            if k in eng and (not isinstance(eng[k], str) or not eng[k]):
                rep.errors.append(f"engine.{k} must be a non-empty string")
        if "revision" in eng and eng["revision"] is not None and (not isinstance(eng["revision"], str) or not eng["revision"]):
            rep.errors.append("engine.revision must be a non-empty string or null")
        if eng.get("execution_mode") not in ("local", "remote"):
            rep.errors.append("engine.execution_mode must be 'local' or 'remote'")

    par = doc["parameters"]
    if not isinstance(par, dict):
        rep.errors.append("parameters must be an object")
    else:
        for k in REQUIRED_PARAMETERS:
            if k not in par:
                rep.errors.append(f"parameters.{k} is required")
            elif not _opt_count(par[k]):
                rep.errors.append(f"parameters.{k} must be null or an integer >= 1")
        if par.get("num_speakers") is not None and (par.get("min_speakers") is not None or par.get("max_speakers") is not None):
            rep.errors.append("parameters.num_speakers excludes min_speakers / max_speakers")
        lo, hi = par.get("min_speakers"), par.get("max_speakers")
        if _opt_count(lo) and _opt_count(hi) and lo is not None and hi is not None and lo > hi:
            rep.errors.append("parameters.min_speakers must be <= max_speakers")

    speakers = doc["speakers"]
    if not isinstance(speakers, list) or not all(isinstance(s, str) and SPEAKER_RE.match(s) for s in speakers):
        rep.errors.append("speakers must be a list of anonymous ids SPEAKER_nn")
        speakers = []
    elif len(set(speakers)) != len(speakers) or speakers != sorted(speakers):
        rep.errors.append("speakers must be unique and sorted")
    if doc["speaker_count"] != len(speakers):
        rep.errors.append("speaker_count must equal len(speakers)")

    _check_turn_list(doc["turns"], "turns", duration, speakers, True, rep)
    used = {t.get("speaker") for t in doc["turns"] if isinstance(t, dict)} if isinstance(doc["turns"], list) else set()

    ov = doc["overlap"]
    if not isinstance(ov, dict) or any(k not in ov for k in ("available", "turns", "regions", "total_seconds")):
        rep.errors.append("overlap must be an object with available, turns, regions, total_seconds")
    elif ov["available"] is True:
        _check_turn_list(ov["turns"], "overlap.turns", duration, speakers, False, rep)
        if isinstance(ov["turns"], list):
            used |= {t.get("speaker") for t in ov["turns"] if isinstance(t, dict)}
        if not isinstance(ov["regions"], list):
            rep.errors.append("overlap.regions must be a list")
        else:
            for i, r in enumerate(ov["regions"]):
                if (not isinstance(r, dict) or not _num(r.get("start")) or not _num(r.get("end")) or r["end"] <= r["start"]
                        or not isinstance(r.get("speakers"), list) or len(r["speakers"]) < 2
                        or any(s not in speakers for s in r["speakers"])):
                    rep.errors.append(f"overlap.regions[{i}]: needs start < end and two or more known speakers")
        if not _num(ov["total_seconds"]) or ov["total_seconds"] < 0:
            rep.errors.append("overlap.total_seconds must be a non-negative number when available")
    elif ov["available"] is False:
        if ov["turns"] is not None or ov["regions"] or ov["total_seconds"] is not None:
            rep.errors.append("overlap without available=true must have turns=null, regions=[], total_seconds=null")
    else:
        rep.errors.append("overlap.available must be a boolean")
    unused = sorted(set(speakers) - used)
    if unused:
        rep.errors.append(f"speakers listed but never used: {unused}")

    _scan_leakage(doc, "", rep)
    return rep


def _scan_leakage(node: Any, path: str, rep: ValidationReport) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(k, str) and k.lower() in FORBIDDEN_KEYS:
                rep.errors.append(f"forbidden key {path}.{k}: documents must not carry commands or credentials")
            _scan_leakage(v, f"{path}.{k}", rep)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _scan_leakage(v, f"{path}[{i}]", rep)
    elif isinstance(node, str):
        if any(p.search(node) for p in CREDENTIAL_PATTERNS):
            rep.errors.append(f"credential-like value at {path}")
        elif any(p.match(node) for p in ABS_PATH_PATTERNS):
            rep.errors.append(f"absolute path leaked at {path}")
