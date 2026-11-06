"""NAME: speaker turns + the caller's roster -> name proposals (`diarization-skill/name-proposals@1`).

The roster (`diarization-skill/roster@1`) is the caller's list of people expected in the recording (the
programme: chair, speakers, ...), each optionally with a consented voiceprint (`diarization enroll`).
Two kinds of evidence are used (ADR-002):

- **voice**: cosine similarity between an anonymous speaker's embedding (`diarize --embeddings`) and a
  person's voiceprint. Speaker S is proposed as person P only when sim(S, P) >= `min_similarity`, P beats
  every other person for S by `min_margin`, and S beats every other speaker for P by `min_margin`.
- **lower third**: OCR text from detection-skill (`detection text`, or its `tracks`) shown near S's
  first turn (`window_before` / `window_after` seconds) that contains P's name or an alias (NFKC,
  case-folded, spaces and punctuation ignored). Only one person matching in the window counts.

A person is proposed for at most one speaker. When voice and lower third disagree, or two speakers point
at the same person, the speaker's status is `conflict` and nothing is proposed. A speaker with no
evidence stays `anonymous`. Every proposal is `confirmed: false`: a person checks it before a name is
used anywhere. This module never writes a name into turns or a transcript.
"""
from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import __version__
from .assign import load_json
from .errors import DiarizationError
from .turns import ISO_RE, SPEAKER_RE, ValidationReport, _num, _scan_leakage, utc_now, validate_turns
from .voices import PERSON_ID_RE, cosine, validate_voiceprint

ROSTER_SCHEMA = "diarization-skill/roster@1"
PROPOSALS_SCHEMA = "diarization-skill/name-proposals@1"
DETECTIONS_SCHEMA = "detection-skill/detections@1"
TRACKS_SCHEMA = "detection-skill/tracks@1"
MIN_SIMILARITY = 0.6
MIN_MARGIN = 0.1
WINDOW_BEFORE = 5.0
WINDOW_AFTER = 15.0
STATUSES = ("proposed", "anonymous", "conflict")
_IGNORED = re.compile(r"[\s　.,:;'\"()\[\]{}|/\\\-_・·‐-―]+")


def normalize_name(text: str) -> str:
    return _IGNORED.sub("", unicodedata.normalize("NFKC", text).casefold())


def file_fingerprint(path: str) -> str:
    with open(path, "rb") as fh:
        return "sha256:" + hashlib.sha256(fh.read()).hexdigest()


# ---- roster --------------------------------------------------------------------------------------
def load_roster(path: str) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """-> (roster, voiceprints by person id, meta). Voiceprint paths are relative to the roster file."""
    roster = load_json(path, "INVALID_ROSTER")
    if roster.get("schema") != ROSTER_SCHEMA:
        raise DiarizationError("INVALID_ROSTER", f"roster schema must be {ROSTER_SCHEMA}")
    people = roster.get("people")
    if not isinstance(people, list) or not people:
        raise DiarizationError("INVALID_ROSTER", "roster.people must be a non-empty list")
    seen = set()
    voiceprints: Dict[str, Dict[str, Any]] = {}
    files: List[Dict[str, Any]] = []
    base = os.path.dirname(os.path.abspath(path))
    for i, p in enumerate(people):
        if not isinstance(p, dict):
            raise DiarizationError("INVALID_ROSTER", f"people[{i}] must be an object")
        pid, name = p.get("id"), p.get("name")
        if not isinstance(pid, str) or not PERSON_ID_RE.match(pid) or pid in seen:
            raise DiarizationError("INVALID_ROSTER", f"people[{i}].id must be a unique id (letters, digits, '_', '.', '-')")
        seen.add(pid)
        if not isinstance(name, str) or not name.strip():
            raise DiarizationError("INVALID_ROSTER", f"people[{i}].name must be a non-empty string")
        aliases = p.get("aliases", [])
        if not isinstance(aliases, list) or not all(isinstance(a, str) and a.strip() for a in aliases):
            raise DiarizationError("INVALID_ROSTER", f"people[{i}].aliases must be a list of names")
        if p.get("role") is not None and not isinstance(p["role"], str):
            raise DiarizationError("INVALID_ROSTER", f"people[{i}].role must be text")
        vp_path = p.get("voiceprint")
        if vp_path is None:
            continue
        if not isinstance(vp_path, str) or not vp_path:
            raise DiarizationError("INVALID_ROSTER", f"people[{i}].voiceprint must be a file name")
        full = vp_path if os.path.isabs(vp_path) else os.path.join(base, vp_path)
        vp = load_json(full, "INVALID_VOICEPRINT")
        rep = validate_voiceprint(vp)
        if not rep.ok:
            raise DiarizationError("INVALID_VOICEPRINT", f"{os.path.basename(full)} is not a valid voiceprint", rep.errors[:5])
        if vp["person"]["name"].strip() != name.strip():
            raise DiarizationError("INVALID_ROSTER", f"people[{i}]: voiceprint {os.path.basename(full)} is for "
                                   f"{vp['person']['name']!r}, not {name!r}")
        voiceprints[pid] = vp
        files.append({"person_id": pid, "file": os.path.basename(full), "fingerprint": file_fingerprint(full),
                      "consent": vp["consent"], "sample_fingerprint": vp["sample"]["fingerprint"]})
    meta = {"file": os.path.basename(path), "fingerprint": file_fingerprint(path), "people": len(people), "voiceprints": files}
    return roster, voiceprints, meta


# ---- lower thirds --------------------------------------------------------------------------------
def text_items(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """detection-skill text detections or text tracks -> [{start, end, text, score}]."""
    if doc.get("schema") == DETECTIONS_SCHEMA:
        if doc.get("kind") != "text":
            raise DiarizationError("INVALID_DETECTIONS", "lower thirds need text detections (detection text), not "
                                   f"{doc.get('kind')!r}")
        out = []
        for f in doc.get("frames") or []:
            for d in (f or {}).get("detections") or []:
                t = d.get("t", f.get("t"))
                if _num(t) and isinstance(d.get("label"), str):
                    out.append({"start": float(t), "end": float(t), "text": d["label"], "score": d.get("score")})
        return out
    if doc.get("schema") == TRACKS_SCHEMA:
        if doc.get("kind") != "text":
            raise DiarizationError("INVALID_DETECTIONS", "lower thirds need text tracks, not " + repr(doc.get("kind")))
        return [{"start": float(t["start"]), "end": float(t["end"]), "text": t["label"], "score": t.get("mean_score")}
                for t in doc.get("tracks") or [] if _num(t.get("start")) and _num(t.get("end")) and isinstance(t.get("label"), str)]
    raise DiarizationError("INVALID_DETECTIONS", f"lower thirds must be {DETECTIONS_SCHEMA} or {TRACKS_SCHEMA} (kind text)")


def _matches(text: str, people: Sequence[Dict[str, Any]]) -> List[str]:
    norm = normalize_name(text)
    hits = []
    for p in people:
        names = [p["name"]] + list(p.get("aliases") or [])
        if any(normalize_name(n) and normalize_name(n) in norm for n in names):
            hits.append(p["id"])
    return hits


# ---- proposals -----------------------------------------------------------------------------------
def propose(turns_doc: Dict[str, Any], roster: Dict[str, Any], voiceprints: Dict[str, Dict[str, Any]], roster_meta: Dict[str, Any],
            *, lower_thirds: Optional[Dict[str, Any]] = None, lower_thirds_meta: Optional[Dict[str, Any]] = None,
            turns_meta: Optional[Dict[str, Any]] = None, min_similarity: float = MIN_SIMILARITY, min_margin: float = MIN_MARGIN,
            window_before: float = WINDOW_BEFORE, window_after: float = WINDOW_AFTER,
            allow_source_mismatch: bool = False) -> Dict[str, Any]:
    if not -1 < min_similarity <= 1 or not 0 < min_margin < 2:
        raise DiarizationError("INVALID_ARGUMENT", "--min-similarity must be in (-1, 1] and --min-margin in (0, 2)")
    if window_before < 0 or window_after < 0:
        raise DiarizationError("INVALID_ARGUMENT", "--window-before / --window-after must be >= 0")
    rep = validate_turns(turns_doc)
    if not rep.ok:
        raise DiarizationError("INVALID_TURNS", "turns document is invalid", rep.errors[:5])
    warnings: List[str] = []
    people = roster["people"]
    by_id = {p["id"]: p for p in people}
    speakers: List[str] = list(turns_doc["speakers"])
    first: Dict[str, float] = {}
    for t in turns_doc["turns"]:
        first.setdefault(t["speaker"], t["start"])

    # voice evidence
    sims: Dict[str, Dict[str, float]] = {s: {} for s in speakers}
    emb = turns_doc.get("speaker_embeddings")
    if voiceprints:
        if not emb or not emb.get("available"):
            warnings.append("voice evidence unavailable: the turns document has no speaker_embeddings (diarize --embeddings)")
        else:
            eng = turns_doc["engine"]
            for pid, vp in voiceprints.items():
                ve = vp["engine"]
                if (ve.get("id"), ve.get("model")) != (eng.get("id"), eng.get("model")):
                    raise DiarizationError("EMBEDDING_MISMATCH", f"voiceprint of {pid} was made with {ve.get('model')!r}, the turns "
                                           f"with {eng.get('model')!r}; embeddings compare only within one model")
                if ve.get("revision") and eng.get("revision") and ve["revision"] != eng["revision"]:
                    raise DiarizationError("EMBEDDING_MISMATCH", f"voiceprint of {pid} comes from another model snapshot "
                                           f"({ve['revision']} vs {eng['revision']})")
                if not (ve.get("revision") and eng.get("revision")):
                    warnings.append(f"model snapshot of the voiceprint of {pid} or of the turns is unknown; assumed the same")
                if vp["embedding"]["dimension"] != emb["dimension"]:
                    raise DiarizationError("EMBEDDING_MISMATCH", f"voiceprint of {pid} has dimension {vp['embedding']['dimension']}, "
                                           f"the turns {emb['dimension']}")
                for s in speakers:
                    vec = emb["vectors"].get(s)
                    if vec is not None:
                        sims[s][pid] = round(cosine(vec, vp["embedding"]["vector"]), 4)
            no_vec = [s for s in speakers if s not in emb["vectors"]]
            if no_vec:
                warnings.append(f"no embedding for {no_vec}: no voice evidence for them")

    voice_pick: Dict[str, Optional[str]] = {}
    voice_note: Dict[str, str] = {}
    for s in speakers:
        voice_pick[s] = None
        if not sims[s]:
            continue
        pid, best = max(sorted(sims[s].items()), key=lambda kv: kv[1])
        other_people = max([v for k, v in sims[s].items() if k != pid], default=None)
        other_speakers = max([sims[o][pid] for o in speakers if o != s and pid in sims[o]], default=None)
        if best < min_similarity:
            voice_note[s] = f"best voice match {pid} {best:.2f} is under {min_similarity:g}"
        elif other_people is not None and best - other_people < min_margin:
            voice_note[s] = f"voice is close to two people ({best:.2f} vs {other_people:.2f})"
        elif other_speakers is not None and best - other_speakers < min_margin:
            voice_note[s] = f"{pid}'s voiceprint fits another speaker almost as well ({best:.2f} vs {other_speakers:.2f})"
        else:
            voice_pick[s] = pid

    # lower-third evidence
    lt: Dict[str, List[Dict[str, Any]]] = {s: [] for s in speakers}
    ocr_pick: Dict[str, Optional[str]] = {s: None for s in speakers}
    ocr_note: Dict[str, str] = {}
    if lower_thirds is not None:
        src = (lower_thirds.get("source") or {}).get("fingerprint")
        if src != turns_doc["source"]["fingerprint"]:
            if not allow_source_mismatch:
                raise DiarizationError("SOURCE_MISMATCH", "the lower-third text was read from a different file than the turns "
                                       "(pass --allow-source-mismatch only when both share one timeline)")
            warnings.append("lower thirds come from a different file; used because --allow-source-mismatch was given")
        items = text_items(lower_thirds)
        for s in speakers:
            lo, hi = first[s] - window_before, first[s] + window_after
            for it in items:
                if it["end"] < lo or it["start"] > hi:
                    continue
                for pid in _matches(it["text"], people):
                    lt[s].append({"person_id": pid, "text": it["text"], "start": round(it["start"], 3), "end": round(it["end"], 3),
                                  "score": it["score"], "seconds_from_first_turn": round(it["start"] - first[s], 3)})
            found = sorted({e["person_id"] for e in lt[s]})
            if len(found) == 1:
                ocr_pick[s] = found[0]
            elif len(found) > 1:
                ocr_note[s] = f"lower thirds near the first turn name {len(found)} people: {found}"

    # combine
    rows: List[Dict[str, Any]] = []
    for s in speakers:
        v, o = voice_pick[s], ocr_pick[s]
        reasons = [n for n in (voice_note.get(s), ocr_note.get(s)) if n]
        if v and o and v != o:
            status, pid = "conflict", None
            reasons.append(f"voice points to {v}, the lower third to {o}")
        elif v or o:
            status, pid = "proposed", v or o
        else:
            status, pid = "anonymous", None
            if not reasons:
                reasons.append("no evidence")
        rows.append({"speaker": s, "first_turn_start": first[s], "status": status, "person_id": pid,
                     "basis": [b for b, x in (("voice", v), ("lower_third", o)) if x and x == pid], "reasons": reasons,
                     "candidates": {x for x in (v, o) if x}})
    # a person pointed to (by voice or by a lower third) from two speakers is proposed for neither
    claimed: Dict[str, List[str]] = {}
    for r in rows:
        for pid in sorted(r.pop("candidates")):
            claimed.setdefault(pid, []).append(r["speaker"])
    for r in rows:
        if r["person_id"] and len(claimed[r["person_id"]]) > 1:
            r["reasons"].append(f"{r['person_id']} is pointed to by {claimed[r['person_id']]}")
            r.update(status="conflict", person_id=None, basis=[])

    out_rows = []
    for r in rows:
        s, pid = r["speaker"], r["person_id"]
        p = by_id.get(pid) if pid else None
        out_rows.append({
            "speaker": s, "first_turn_start": r["first_turn_start"], "status": r["status"],
            "proposal": None if p is None else {"person_id": pid, "name": p["name"], "role": p.get("role"), "basis": r["basis"],
                                                "similarity": sims[s].get(pid), "confirmed": False},
            "evidence": {"voice": [{"person_id": k, "similarity": v} for k, v in sorted(sims[s].items(), key=lambda kv: (-kv[1], kv[0]))],
                         "lower_third": lt[s]},
            "reasons": r["reasons"]})
    proposed = {r["proposal"]["person_id"] for r in out_rows if r["proposal"]}
    counts = {st: sum(1 for r in out_rows if r["status"] == st) for st in STATUSES}
    counts["speakers"] = len(out_rows)
    if turns_meta is None:
        turns_meta = {"file": None, "fingerprint": None}
    doc = {
        "schema": PROPOSALS_SCHEMA,
        "requires_confirmation": True,
        "note": "Proposals only. A person confirms each one before a name is used; unconfirmed speakers stay anonymous.",
        "turns": dict(turns_meta, source_fingerprint=turns_doc["source"]["fingerprint"],
                      engine={k: turns_doc["engine"].get(k) for k in ("id", "model", "revision")}),
        "roster": roster_meta,
        "lower_thirds": lower_thirds_meta,
        "method": {"voice": {"metric": "cosine", "min_similarity": min_similarity, "min_margin": min_margin},
                   "lower_third": {"window_before": window_before, "window_after": window_after,
                                   "match": "name or alias contained in the OCR text (NFKC, case-folded, spaces and punctuation ignored)"},
                   "one_person_per_speaker": True},
        "speakers": out_rows,
        "unmatched_people": [p["id"] for p in people if p["id"] not in proposed],
        "counts": counts,
        "skill": "diarization-skill",
        "skill_version": __version__,
        "tool": "diarization/name",
        "created_at": utc_now(),
        "warnings": warnings,
    }
    rep = validate_proposals(doc)
    if not rep.ok:
        raise DiarizationError("INVALID_RESULT", "name proposals failed validation; nothing returned", rep.errors[:10])
    return doc


def name_speakers(turns_path: str, roster_path: str, lower_thirds_path: Optional[str] = None, **kw: Any) -> Dict[str, Any]:
    turns_doc = load_json(turns_path, "INVALID_TURNS")
    roster, vps, meta = load_roster(roster_path)
    lt = lt_meta = None
    if lower_thirds_path:
        lt = load_json(lower_thirds_path, "INVALID_DETECTIONS")
        lt_meta = {"file": os.path.basename(lower_thirds_path), "fingerprint": file_fingerprint(lower_thirds_path),
                   "schema": lt.get("schema"), "engine": (lt.get("engine") or {}).get("id")}
    return propose(turns_doc, roster, vps, meta, lower_thirds=lt, lower_thirds_meta=lt_meta,
                   turns_meta={"file": os.path.basename(turns_path), "fingerprint": file_fingerprint(turns_path)}, **kw)


def validate_proposals(doc: Any) -> ValidationReport:
    rep = ValidationReport()
    if not isinstance(doc, dict):
        rep.errors.append("name proposals must be a JSON object")
        return rep
    required = ("schema", "requires_confirmation", "turns", "roster", "method", "speakers", "unmatched_people", "counts", "skill",
                "skill_version", "tool", "created_at", "warnings")
    missing = [k for k in required if k not in doc]
    if missing:
        rep.errors.append(f"missing required field(s): {', '.join(missing)}")
        return rep
    if doc["schema"] != PROPOSALS_SCHEMA:
        rep.errors.append(f"schema must be {PROPOSALS_SCHEMA}")
    if doc["requires_confirmation"] is not True:
        rep.errors.append("requires_confirmation must be true: names are proposals until a person confirms them")
    if doc["tool"] != "diarization/name":
        rep.errors.append("tool must be diarization/name")
    if not isinstance(doc["created_at"], str) or not ISO_RE.match(doc["created_at"]):
        rep.errors.append("created_at must be an ISO-8601 UTC timestamp")
    rows = doc["speakers"]
    if not isinstance(rows, list):
        rep.errors.append("speakers must be a list")
        rows = []
    used: Dict[str, str] = {}
    for i, r in enumerate(rows):
        w = f"speakers[{i}]"
        if not isinstance(r, dict) or not isinstance(r.get("speaker"), str) or not SPEAKER_RE.match(r["speaker"]):
            rep.errors.append(f"{w}: speaker must be an anonymous id SPEAKER_nn")
            continue
        st, prop = r.get("status"), r.get("proposal")
        if st not in STATUSES:
            rep.errors.append(f"{w}: status must be one of {STATUSES}")
        if (st == "proposed") != isinstance(prop, dict):
            rep.errors.append(f"{w}: a proposal is present exactly when status is 'proposed'")
        if isinstance(prop, dict):
            if prop.get("confirmed") is not False:
                rep.errors.append(f"{w}: proposal.confirmed must be false (this tool never confirms a name)")
            if not prop.get("basis") or any(b not in ("voice", "lower_third") for b in prop["basis"]):
                rep.errors.append(f"{w}: proposal.basis must name its evidence (voice, lower_third)")
            pid = prop.get("person_id")
            if pid in used:
                rep.errors.append(f"{w}: {pid} is also proposed for {used[pid]}")
            used[pid] = r["speaker"]
        if not isinstance(r.get("evidence"), dict) or not isinstance(r.get("reasons"), list):
            rep.errors.append(f"{w}: evidence (object) and reasons (list) are required")
        elif st != "proposed" and not r["reasons"]:
            rep.errors.append(f"{w}: a speaker left without a proposal needs a reason")
    # OCR text is quoted as read (it is what the frame shows), so it is left out of the path/credential scan
    scan = dict(doc, speakers=[dict(r, evidence=dict(r.get("evidence") or {}, lower_third=None)) if isinstance(r, dict) else r
                               for r in rows])
    _scan_leakage(scan, "", rep)
    return rep
