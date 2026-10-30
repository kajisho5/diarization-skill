"""ASSIGN: transcription-skill transcript + speaker turns -> the same transcript with `speaker_id` filled.

Rule (documented in SKILL.md and README):
- Only the exclusive turns are used: one speaker at a time, so one `speaker_id` per unit.
- A unit (a word, or a segment without words) gets the speaker whose turns overlap it for the longest
  time. Zero overlap with every speaker, or a tie for the longest, leaves it `null`. Nothing is guessed:
  no nearest-turn fill, no carry-over from neighbours.
- A segment with words gets the speaker with the largest total duration of the words assigned to that
  speaker (words left `null` do not vote). A tie, or no assigned word, leaves the segment `null`.
- Every word gets its own `speaker_id` (an extra field; transcription-skill's validator permits it).
- The transcript's own fields, ids, text and times are not changed. What was done is recorded under the
  top-level key `speaker_assignment`, including the engine/model/revision and the model attribution, and
  which segments touch overlapping speech.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import __version__
from .errors import DiarizationError
from .turns import TURNS_SCHEMA, utc_now, validate_turns

TRANSCRIPT_SCHEMA = "transcription-skill/transcript/0.1"
TIE_EPS = 1e-9
METHOD = "max_overlap_with_exclusive_turns"


def load_json(path: str, code: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise DiarizationError("INPUT_NOT_FOUND", f"not found: {os.path.basename(path)}")
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as exc:
        raise DiarizationError(code, f"cannot read {os.path.basename(path)} as JSON: {exc}")
    if not isinstance(doc, dict):
        raise DiarizationError(code, f"{os.path.basename(path)} is not a JSON object")
    return doc


def check_transcript(doc: Dict[str, Any]) -> None:
    """The parts of the transcript contract ASSIGN relies on. Full validation is transcription-skill's
    `transcription check`; this skill does not import transcription-skill."""
    if doc.get("schema") != TRANSCRIPT_SCHEMA:
        raise DiarizationError("INVALID_TRANSCRIPT", f"transcript schema must be {TRANSCRIPT_SCHEMA}, got {doc.get('schema')!r}")
    segs = doc.get("segments")
    if not isinstance(segs, list):
        raise DiarizationError("INVALID_TRANSCRIPT", "transcript.segments must be a list")
    for i, seg in enumerate(segs):
        if not isinstance(seg, dict) or not _num(seg.get("start")) or not _num(seg.get("end")) or "speaker_id" not in seg:
            raise DiarizationError("INVALID_TRANSCRIPT", f"segments[{i}] needs numeric start/end and a speaker_id field")
        words = seg.get("words")
        if words is not None and (not isinstance(words, list)
                                  or any(not isinstance(w, dict) or not _num(w.get("start")) or not _num(w.get("end")) for w in words)):
            raise DiarizationError("INVALID_TRANSCRIPT", f"segments[{i}].words must be null or a list of timed words")
    src = doc.get("source")
    if not isinstance(src, dict) or not isinstance(src.get("fingerprint"), str):
        raise DiarizationError("INVALID_TRANSCRIPT", "transcript.source.fingerprint is required")


def overlap_by_speaker(start: float, end: float, turns: Sequence[Tuple[float, float, str]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for s, e, spk in turns:
        if e <= start:
            continue
        if s >= end:
            break  # turns are ordered by start
        ov = min(end, e) - max(start, s)
        if ov > 0:
            out[spk] = out.get(spk, 0.0) + ov
    return out


def pick(scores: Dict[str, float]) -> Optional[str]:
    """The single speaker with the strictly largest positive score; None on zero or a tie."""
    positive = [(v, k) for k, v in scores.items() if v > TIE_EPS]
    if not positive:
        return None
    positive.sort(reverse=True)
    if len(positive) > 1 and positive[0][0] - positive[1][0] <= TIE_EPS:
        return None
    return positive[0][1]


def assign(transcript: Dict[str, Any], turns_doc: Dict[str, Any], *, allow_source_mismatch: bool = False,
           replace: bool = False) -> Dict[str, Any]:
    check_transcript(transcript)
    rep = validate_turns(turns_doc)
    if not rep.ok:
        raise DiarizationError("INVALID_TURNS", "turns document failed validation", rep.errors[:10])
    t_fp = transcript["source"]["fingerprint"]
    d_fp = turns_doc["source"]["fingerprint"]
    out = copy.deepcopy(transcript)
    notes: List[str] = []
    if t_fp != d_fp:
        if not allow_source_mismatch:
            raise DiarizationError("SOURCE_MISMATCH", "transcript and turns were made from different files (fingerprints differ); "
                                   "pass --allow-source-mismatch only if both are on the same timeline")
        notes.append("speaker_id: transcript and turns come from different files (fingerprints differ); "
                     "assigned on the assumption that both share one timeline")
    preset = [s.get("id") for s in out["segments"] if s.get("speaker_id") is not None]
    if preset and not replace:
        raise DiarizationError("SPEAKERS_ALREADY_SET", f"{len(preset)} segment(s) already have a speaker_id; pass --replace to recompute")

    turns = [(float(t["start"]), float(t["end"]), str(t["speaker"])) for t in turns_doc["turns"]]
    regions = turns_doc["overlap"]["regions"] if turns_doc["overlap"].get("available") else []
    stats = {"segments": len(out["segments"]), "assigned": 0, "unassigned": 0, "by_words": 0, "by_segment": 0,
             "words": 0, "words_assigned": 0}
    touching_overlap: List[str] = []
    per_speaker: Dict[str, int] = {}
    for seg in out["segments"]:
        s, e = float(seg["start"]), float(seg["end"])
        words = seg.get("words")
        if words:
            stats["by_words"] += 1
            votes: Dict[str, float] = {}
            for w in words:
                spk = pick(overlap_by_speaker(float(w["start"]), float(w["end"]), turns))
                w["speaker_id"] = spk
                stats["words"] += 1
                if spk is not None:
                    stats["words_assigned"] += 1
                    votes[spk] = votes.get(spk, 0.0) + (float(w["end"]) - float(w["start"]))
            speaker = pick(votes)
        else:
            stats["by_segment"] += 1
            speaker = pick(overlap_by_speaker(s, e, turns))
        seg["speaker_id"] = speaker
        if speaker is None:
            stats["unassigned"] += 1
        else:
            stats["assigned"] += 1
            per_speaker[speaker] = per_speaker.get(speaker, 0) + 1
        if any(r["start"] < e and r["end"] > s for r in regions):
            touching_overlap.append(str(seg.get("id")))

    if stats["unassigned"]:
        notes.append(f"speaker_id: {stats['unassigned']} of {stats['segments']} segment(s) left null "
                     "(no overlap with a speaker turn, or a tie)")
    if touching_overlap:
        notes.append(f"speaker_id: {len(touching_overlap)} segment(s) touch overlapping speech; "
                     "speaker_assignment.overlap_segment_ids lists them")
    if preset:
        notes.append(f"speaker_id: {len(preset)} previously set value(s) replaced")
    out["warnings"] = list(out.get("warnings") or []) + notes
    eng = turns_doc["engine"]
    out["speaker_assignment"] = {
        "skill": "diarization-skill",
        "skill_version": __version__,
        "tool": "diarization/assign",
        "created_at": utc_now(),
        "method": METHOD,
        "rule": "per word (else per segment): speaker with the longest overlap with the exclusive turns; zero overlap "
                "or a tie -> null. Segment with words: speaker with the largest assigned word duration; tie -> null.",
        "speaker_ids": "anonymous (SPEAKER_nn); they name nobody",
        "turns_schema": TURNS_SCHEMA,
        "turns_created_at": turns_doc["created_at"],
        "source_fingerprint": d_fp,
        "source_match": t_fp == d_fp,
        "engine": {k: eng.get(k) for k in ("id", "version", "model", "revision", "license", "attribution", "execution_mode")},
        "parameters": dict(turns_doc["parameters"]),
        "speakers": list(turns_doc["speakers"]),
        "segments_per_speaker": dict(sorted(per_speaker.items())),
        "counts": stats,
        "overlap_available": bool(turns_doc["overlap"].get("available")),
        "overlap_segment_ids": touching_overlap,
    }
    return out


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and v not in (float("inf"), float("-inf"))
