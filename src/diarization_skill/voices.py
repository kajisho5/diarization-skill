"""ENROLL: a consented voice sample of one known person -> a voiceprint (`diarization-skill/voiceprint@1`).

A voiceprint is the speaker embedding of the sample's one voice, made by the same engine and model as
DIARIZE, with the person's name and the consent record (who gave it, when, for what). It is used only by
`diarization name` to *propose* which anonymous speaker might be that person (ADR-002); it never names
anyone by itself.

Rules:
- No consent record, no voiceprint: `consent.given_by` and `consent.date` are required.
- The sample must hold one voice: the dominant speaker must have at least `min_share` of the speech
  and at least `min_speech` seconds, and the engine must give an embedding for it. Otherwise the
  enrollment is refused (ENROLLMENT_REJECTED); nothing is averaged or guessed.
- The document carries no path: the sample is named by its bare file name and sha256.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .errors import DiarizationError
from .turns import FINGERPRINT_RE, ISO_RE, ValidationReport, _num, _scan_leakage, utc_now

VOICEPRINT_SCHEMA = "diarization-skill/voiceprint@1"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
PERSON_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")
MIN_SHARE = 0.8
MIN_SPEECH = 3.0


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise DiarizationError("EMBEDDING_MISMATCH", f"embedding dimensions differ ({len(a)} vs {len(b)})")
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        raise DiarizationError("EMBEDDING_MISMATCH", "zero embedding")
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def check_consent(consent: Dict[str, Any]) -> Dict[str, Any]:
    by = consent.get("given_by")
    date = consent.get("date")
    note = consent.get("note")
    if not isinstance(by, str) or not by.strip():
        raise DiarizationError("INVALID_ARGUMENT", "a voiceprint needs a consent record: --consent-by (who gave consent)")
    if not isinstance(date, str) or not DATE_RE.match(date):
        raise DiarizationError("INVALID_ARGUMENT", "a voiceprint needs a consent record: --consent-date YYYY-MM-DD")
    if note is not None and not isinstance(note, str):
        raise DiarizationError("INVALID_ARGUMENT", "--consent-note must be text")
    return {"given_by": by.strip(), "date": date, "note": (note or "").strip() or None}


def voiceprint_from_turns(turns_doc: Dict[str, Any], *, name: str, consent: Dict[str, Any], person_id: Optional[str] = None,
                          role: Optional[str] = None, min_share: float = MIN_SHARE, min_speech: float = MIN_SPEECH) -> Dict[str, Any]:
    """Build the voiceprint from a turns document of the sample made with embeddings."""
    if not isinstance(name, str) or not name.strip():
        raise DiarizationError("INVALID_ARGUMENT", "--name is required")
    if person_id is not None and not PERSON_ID_RE.match(person_id):
        raise DiarizationError("INVALID_ARGUMENT", "--person-id must be 1-64 letters, digits, '_', '.', '-'")
    consent = check_consent(consent)
    per: Dict[str, float] = {}
    for t in turns_doc["turns"]:
        per[t["speaker"]] = per.get(t["speaker"], 0.0) + t["end"] - t["start"]
    total = sum(per.values())
    if not per:
        raise DiarizationError("ENROLLMENT_REJECTED", "no speech found in the sample")
    dominant = max(sorted(per), key=lambda s: per[s])
    share = per[dominant] / total
    found = {"speakers_found": len(per), "speech_seconds": round(total, 3), "dominant_seconds": round(per[dominant], 3),
             "dominant_share": round(share, 4)}
    if share < min_share:
        raise DiarizationError("ENROLLMENT_REJECTED", f"the sample holds more than one voice: the main voice has {share:.0%} "
                               f"of the speech (needs {min_share:.0%}); record the person alone", found)
    if per[dominant] < min_speech:
        raise DiarizationError("ENROLLMENT_REJECTED", f"too little speech in the sample ({per[dominant]:.1f}s, needs "
                               f"{min_speech:g}s)", found)
    emb = turns_doc.get("speaker_embeddings") or {}
    vec = (emb.get("vectors") or {}).get(dominant)
    if not vec:
        raise DiarizationError("ENROLLMENT_REJECTED", "the engine gave no speaker embedding for the sample's voice", found)
    eng = turns_doc["engine"]
    warnings: List[str] = []
    if len(per) > 1:
        warnings.append(f"the sample had {len(per)} speaker(s); the main one ({share:.0%} of the speech) was used")
    return {
        "schema": VOICEPRINT_SCHEMA,
        "person": {"name": name.strip(), "person_id": person_id, "role": role},
        "consent": consent,
        "sample": dict({k: turns_doc["source"][k] for k in ("filename", "fingerprint", "size_bytes", "media_duration",
                                                            "audio_stream")}, **found),
        "engine": {k: eng.get(k) for k in ("id", "version", "model", "revision", "license", "attribution")},
        "embedding": {"metric": "cosine", "dimension": len(vec), "vector": list(vec)},
        "parameters": {"min_share": min_share, "min_speech": min_speech},
        "skill": "diarization-skill",
        "skill_version": __version__,
        "tool": "diarization/enroll",
        "created_at": utc_now(),
        "warnings": warnings,
    }


def enroll(sample: str, *, name: str, consent: Dict[str, Any], person_id: Optional[str] = None, role: Optional[str] = None,
           min_share: float = MIN_SHARE, min_speech: float = MIN_SPEECH, **diarize_kwargs: Any) -> Dict[str, Any]:
    """Diarize the sample with embeddings (same engine, extraction and worker as DIARIZE), then keep its one voice."""
    from .service import diarize
    check_consent(consent)  # refuse before running the engine
    if not 0 < min_share <= 1 or min_speech <= 0:
        raise DiarizationError("INVALID_ARGUMENT", "--min-share must be in (0, 1] and --min-speech > 0")
    doc = diarize(sample, embeddings=True, **diarize_kwargs)
    vp = voiceprint_from_turns(doc, name=name, consent=consent, person_id=person_id, role=role, min_share=min_share,
                               min_speech=min_speech)
    rep = validate_voiceprint(vp)
    if not rep.ok:
        raise DiarizationError("INVALID_RESULT", "voiceprint failed validation; nothing returned", rep.errors[:10])
    return vp


def validate_voiceprint(doc: Any) -> ValidationReport:
    rep = ValidationReport()
    if not isinstance(doc, dict):
        rep.errors.append("voiceprint must be a JSON object")
        return rep
    required = ("schema", "person", "consent", "sample", "engine", "embedding", "skill", "skill_version", "tool", "created_at",
                "warnings")
    missing = [k for k in required if k not in doc]
    if missing:
        rep.errors.append(f"missing required field(s): {', '.join(missing)}")
        return rep
    if doc["schema"] != VOICEPRINT_SCHEMA:
        rep.errors.append(f"schema must be {VOICEPRINT_SCHEMA}")
    person = doc["person"]
    if not isinstance(person, dict) or not isinstance(person.get("name"), str) or not person["name"].strip():
        rep.errors.append("person.name must be a non-empty string")
    elif person.get("person_id") is not None and not (isinstance(person["person_id"], str) and PERSON_ID_RE.match(person["person_id"])):
        rep.errors.append("person.person_id must be null or 1-64 letters, digits, '_', '.', '-'")
    c = doc["consent"]
    if (not isinstance(c, dict) or not isinstance(c.get("given_by"), str) or not c["given_by"].strip()
            or not isinstance(c.get("date"), str) or not DATE_RE.match(c["date"])):
        rep.errors.append("consent must record given_by and date (YYYY-MM-DD); no consent, no voiceprint")
    s = doc["sample"]
    if not isinstance(s, dict) or not isinstance(s.get("fingerprint"), str) or not FINGERPRINT_RE.match(s["fingerprint"]):
        rep.errors.append("sample.fingerprint must be 'sha256:<64 hex>'")
    elif not isinstance(s.get("filename"), str) or "/" in s["filename"] or "\\" in s["filename"]:
        rep.errors.append("sample.filename must be a bare file name")
    e = doc["engine"]
    if not isinstance(e, dict) or any(not isinstance(e.get(k), str) or not e.get(k) for k in ("id", "model")):
        rep.errors.append("engine.id and engine.model are required (embeddings compare only within one model)")
    emb = doc["embedding"]
    if not isinstance(emb, dict) or emb.get("metric") != "cosine":
        rep.errors.append("embedding.metric must be 'cosine'")
    else:
        vec, dim = emb.get("vector"), emb.get("dimension")
        if (not isinstance(dim, int) or isinstance(dim, bool) or dim < 1 or not isinstance(vec, list) or len(vec) != dim
                or not all(_num(x) for x in vec) or not any(x != 0 for x in vec)):
            rep.errors.append("embedding.vector must be `dimension` finite numbers, not all zero")
    if doc["tool"] != "diarization/enroll":
        rep.errors.append("tool must be diarization/enroll")
    if not isinstance(doc["created_at"], str) or not ISO_RE.match(doc["created_at"]):
        rep.errors.append("created_at must be an ISO-8601 UTC timestamp")
    _scan_leakage(doc, "", rep)
    return rep
