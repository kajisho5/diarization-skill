"""diarization CLI. Every command prints one JSON document with --json.

  diarization diarize talk.mp4 -o turns.json --json                 # anonymous speaker turns
  diarization diarize panel.wav --min-speakers 3 --max-speakers 6 -o turns.json
  diarization assign transcript.json turns.json -o transcript.speakers.json --json
  diarization check turns.json --json
  diarization score turns.json reference.json --json                 # DER against a reference
  diarization contract --json
  diarization doctor --json

Exit codes: 0 done, 2 invalid input or arguments, 3 engine, pipeline or ffmpeg missing, 4 engine failed or timed out.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from . import __version__
from .assign import assign, load_json
from .contract import ERRORS, contract
from .doctor import doctor
from .errors import DiarizationError
from .metrics import der
from .service import DEFAULT_TIMEOUT, DEVICES, diarize
from .turns import validate_turns


def _emit(doc: Dict[str, Any], as_json: bool, human: str = "") -> None:
    if as_json:
        print(json.dumps(doc, ensure_ascii=False, indent=2))
    elif human:
        print(human)


def _fail(code: str, message: str, as_json: bool, detail: Optional[Any] = None) -> int:
    doc = {"ok": False, "error": {"code": code, "message": message, "detail": detail}}
    if as_json:
        print(json.dumps(doc, ensure_ascii=False, indent=2))
    else:
        print(f"error: {message}" + (f" ({detail})" if detail else ""), file=sys.stderr)
    return ERRORS.get(code, {}).get("exit_code", 1)


def _same(a: str, b: str) -> bool:
    return os.path.abspath(a) == os.path.abspath(b) or (os.path.exists(a) and os.path.exists(b) and os.path.samefile(a, b))


def _check_output(path: str, inputs: List[str], overwrite: bool) -> None:
    if any(_same(path, i) for i in inputs):
        raise DiarizationError("OUTPUT_IS_INPUT", "-o must not be an input file")
    if os.path.exists(path) and not overwrite:
        raise DiarizationError("OUTPUT_EXISTS", f"refusing to overwrite {os.path.basename(path)}: pass --overwrite")


def _write_json(path: str, doc: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def cmd_diarize(a: argparse.Namespace) -> int:
    if a.output:
        _check_output(a.output, [a.media], a.overwrite)
    doc = diarize(a.media, num_speakers=a.num_speakers, min_speakers=a.min_speakers, max_speakers=a.max_speakers,
                  audio_stream=a.audio_stream, timeout=a.timeout, pipeline_dir=a.pipeline_dir, device=a.device)
    if a.output:
        _write_json(a.output, doc)
    lines = [f"{doc['speaker_count']} speaker(s), {len(doc['turns'])} turn(s) over {doc['source']['media_duration']:.1f}s"]
    lines += [f"  {t['start']:>9.3f} {t['end']:>9.3f}  {t['speaker']}" for t in doc["turns"]]
    _emit(dict(doc, ok=True, turns_file=a.output), a.json, "\n".join(lines))
    return 0


def cmd_assign(a: argparse.Namespace) -> int:
    _check_output(a.output, [a.transcript, a.turns], a.overwrite)
    transcript = load_json(a.transcript, "INVALID_TRANSCRIPT")
    turns_doc = load_json(a.turns, "INVALID_TURNS")
    out = assign(transcript, turns_doc, allow_source_mismatch=a.allow_source_mismatch, replace=a.replace)
    _write_json(a.output, out)
    sa = out["speaker_assignment"]
    doc = {"ok": True, "output": os.path.abspath(a.output), "speakers": sa["speakers"], "counts": sa["counts"],
           "segments_per_speaker": sa["segments_per_speaker"], "overlap_segment_ids": sa["overlap_segment_ids"],
           "source_match": sa["source_match"], "warnings": [w for w in out["warnings"] if w.startswith("speaker_id:")]}
    c = sa["counts"]
    _emit(doc, a.json, f"{c['assigned']} of {c['segments']} segment(s) got a speaker_id; wrote {a.output}")
    return 0


def cmd_check(a: argparse.Namespace) -> int:
    rep = validate_turns(load_json(a.turns, "INVALID_TURNS"))
    _emit(rep.to_dict(), a.json, "valid" if rep.ok else "INVALID\n" + "\n".join(f"  error: {e}" for e in rep.errors))
    return 0 if rep.ok else 2


def cmd_score(a: argparse.Namespace) -> int:
    docs = []
    for path in (a.turns, a.reference):
        d = load_json(path, "INVALID_TURNS")
        turns = d.get("turns") if isinstance(d, dict) else None
        if not isinstance(turns, list) or any(not isinstance(t, dict) or not {"start", "end", "speaker"} <= set(t) for t in turns):
            raise DiarizationError("INVALID_TURNS", f"{os.path.basename(path)} has no turns [{{start, end, speaker}}]")
        use = d.get("overlap", {}).get("turns") if a.with_overlap and isinstance(d.get("overlap"), dict) else None
        docs.append([(float(t["start"]), float(t["end"]), str(t["speaker"])) for t in (use or turns)])
    res = der(docs[1], docs[0])
    res = {k: round(v, 4) for k, v in res.items()}
    _emit(dict(res, ok=True, collar=0.0, overlap_scored=bool(a.with_overlap)), a.json, f"DER {res['der']:.2%}")
    return 0


def cmd_contract(a: argparse.Namespace) -> int:
    _emit(contract(), True)
    return 0


def cmd_doctor(a: argparse.Namespace) -> int:
    doc = doctor(a.pipeline_dir)
    human = "OK" if doc["ok"] else "\n".join(f"problem: {p}" for p in doc["problems"])
    _emit(doc, a.json, human)
    return 0 if doc["ok"] else 3


def _count(v: str) -> int:
    try:
        n = int(v)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer")
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="diarization", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"diarization-skill {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("diarize", help="anonymous speaker turns of a recording")
    d.add_argument("media")
    g = d.add_argument_group("speaker count (optional)")
    g.add_argument("--num-speakers", type=_count, help="exact number of speakers, when known")
    g.add_argument("--min-speakers", type=_count, help="lower bound (excludes --num-speakers)")
    g.add_argument("--max-speakers", type=_count, help="upper bound (excludes --num-speakers)")
    d.add_argument("--audio-stream", type=int, default=0, help="0-based audio stream to use (default 0, the first)")
    d.add_argument("--pipeline-dir", help="local pipeline directory (default: $DIARIZATION_PIPELINE_DIR)")
    d.add_argument("--device", choices=DEVICES, default="cpu")
    d.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help=f"seconds for the engine (default {DEFAULT_TIMEOUT:g})")
    d.add_argument("-o", "--output", help="write the turns JSON here")
    d.add_argument("--overwrite", action="store_true")
    d.add_argument("--json", action="store_true", help="print one JSON document")
    d.set_defaults(func=cmd_diarize)

    s = sub.add_parser("assign", help="fill speaker_id in a transcription-skill transcript")
    s.add_argument("transcript")
    s.add_argument("turns")
    s.add_argument("-o", "--output", required=True, help="write the transcript with speaker_id here")
    s.add_argument("--replace", action="store_true", help="recompute speaker_id values that are already set")
    s.add_argument("--allow-source-mismatch", action="store_true",
                   help="accept turns made from a different file (only when both share one timeline)")
    s.add_argument("--overwrite", action="store_true")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_assign)

    c = sub.add_parser("check", help="validate a turns document")
    c.add_argument("turns")
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_check)

    sc = sub.add_parser("score", help="DER of a turns document against a reference")
    sc.add_argument("turns")
    sc.add_argument("reference", help="reference document with turns [{start, end, speaker}] (any ids)")
    sc.add_argument("--with-overlap", action="store_true", help="score overlap.turns instead of the exclusive turns")
    sc.add_argument("--json", action="store_true")
    sc.set_defaults(func=cmd_score)

    k = sub.add_parser("contract", help="machine-readable contract")
    k.add_argument("--json", action="store_true")
    k.set_defaults(func=cmd_contract)

    dr = sub.add_parser("doctor", help="environment check (never downloads)")
    dr.add_argument("--pipeline-dir")
    dr.add_argument("--json", action="store_true")
    dr.set_defaults(func=cmd_doctor)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    a = parser().parse_args(argv)
    try:
        return a.func(a)
    except DiarizationError as e:
        return _fail(e.code, e.message, getattr(a, "json", False), e.detail)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
