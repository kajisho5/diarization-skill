"""DIARIZE: media file -> speaker turns document (`diarization-skill/turns@1`).

probe + fingerprint -> fixed extraction to mono 16 kHz WAV (private temp dir) -> engine in a worker
subprocess with a timeout that kills its whole process group -> labels, overlap, provenance -> validated
document. An engine object passed in (the test suite's fake) runs in a thread with the same timeout.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from . import __version__
from .engines import DEFAULT_ENGINE, get_engine
from .engines.base import DiarizationEngine, EngineRequest, EngineResult
from .engines.worker import OFFLINE_ENV
from .errors import DiarizationError
from .media import child_env, extract_audio, fingerprint_file, probe
from .pipeline import inspect, resolve_dir
from .turns import build_turns, validate_turns

WORKER_MODULE = "diarization_skill.engines.worker"
DEFAULT_TIMEOUT = 3600.0
DEVICES = ("cpu", "cuda", "mps")


def check_speaker_counts(num_speakers: Optional[int], min_speakers: Optional[int], max_speakers: Optional[int]) -> None:
    for name, v in (("--num-speakers", num_speakers), ("--min-speakers", min_speakers), ("--max-speakers", max_speakers)):
        if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v < 1):
            raise DiarizationError("INVALID_ARGUMENT", f"{name} must be an integer >= 1")
    if num_speakers is not None and (min_speakers is not None or max_speakers is not None):
        raise DiarizationError("INVALID_ARGUMENT", "--num-speakers excludes --min-speakers / --max-speakers")
    if min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
        raise DiarizationError("INVALID_ARGUMENT", "--min-speakers must be <= --max-speakers")


def diarize(media: str, *, num_speakers: Optional[int] = None, min_speakers: Optional[int] = None,
            max_speakers: Optional[int] = None, audio_stream: int = 0, timeout: float = DEFAULT_TIMEOUT,
            pipeline_dir: Optional[str] = None, device: str = "cpu", engine: Optional[DiarizationEngine] = None,
            engine_id: str = DEFAULT_ENGINE) -> Dict[str, Any]:
    t0 = time.time()
    check_speaker_counts(num_speakers, min_speakers, max_speakers)
    if device not in DEVICES:
        raise DiarizationError("INVALID_ARGUMENT", f"--device must be one of {DEVICES}")
    if not isinstance(audio_stream, int) or audio_stream < 0:
        raise DiarizationError("INVALID_ARGUMENT", "--audio-stream must be an integer >= 0")
    if timeout <= 0:
        raise DiarizationError("INVALID_ARGUMENT", "--timeout must be > 0")
    if not os.path.isfile(media):
        raise DiarizationError("INPUT_NOT_FOUND", f"media not found: {os.path.basename(media)}")
    eng = engine if engine is not None else get_engine(engine_id)
    warnings: List[str] = []
    revision: Optional[str] = None
    pdir = resolve_dir(pipeline_dir)
    if eng.needs_pipeline_dir:
        info = inspect(pdir)
        if info["status"] != "OK":
            raise DiarizationError("PIPELINE_NOT_FOUND", f"pipeline directory not ready: {info['detail']}",
                                   "run `diarization doctor` for the one-time setup steps")
        revision = info["revision"]
        if revision is None:
            warnings.append("pipeline snapshot revision unknown (directory is neither a git clone nor a hub snapshot)")
    info_media = probe(media)
    if audio_stream >= info_media["audio_stream_count"]:
        raise DiarizationError("INVALID_ARGUMENT", f"--audio-stream {audio_stream} out of range "
                               f"({info_media['audio_stream_count']} audio stream(s))")
    fingerprint = fingerprint_file(media)
    run_dir = tempfile.mkdtemp(prefix="diarization-")
    try:
        wav = os.path.join(run_dir, "audio.wav")
        extraction = extract_audio(media, wav, audio_stream=audio_stream, timeout=max(60.0, timeout))
        ereq = EngineRequest(audio_path=wav, pipeline_dir=pdir if eng.needs_pipeline_dir else None, num_speakers=num_speakers,
                             min_speakers=min_speakers, max_speakers=max_speakers, device=device)
        t_engine = time.time()
        result = run_engine(eng, ereq, timeout, run_dir, injected=engine is not None)
        extraction["engine_seconds"] = round(time.time() - t_engine, 3)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)

    duration = float(info_media["duration"])
    doc = build_turns(
        exclusive=result.exclusive, regular=result.regular, duration=duration,
        source={"filename": os.path.basename(media), "fingerprint": fingerprint, "size_bytes": os.path.getsize(media),
                "media_duration": duration, "has_video": bool(info_media["has_video"]), "audio_stream": audio_stream},
        engine={"id": eng.id, "version": result.engine_version, "model": result.model or eng.model, "revision": revision,
                "license": eng.license, "attribution": eng.attribution, "execution_mode": eng.execution_mode,
                "device": device, **{k: v for k, v in result.extra.items() if k == "torch_version"}},
        parameters={"num_speakers": num_speakers, "min_speakers": min_speakers, "max_speakers": max_speakers},
        audio_extraction=extraction, skill_version=__version__, processing_seconds=time.time() - t0, warnings=warnings)
    rep = validate_turns(doc, expected_fingerprint=fingerprint)
    if not rep.ok:
        raise DiarizationError("INVALID_RESULT", "engine output failed turns validation; nothing returned", rep.errors[:10])
    return doc


def run_engine(engine: DiarizationEngine, ereq: EngineRequest, timeout: float, run_dir: str, injected: bool = False) -> EngineResult:
    if injected:
        return _run_in_thread(engine, ereq, timeout)
    req_path = os.path.join(run_dir, "engine_request.json")
    out_path = os.path.join(run_dir, "engine_result.json")
    with open(req_path, "w", encoding="utf-8") as fh:
        json.dump({"engine_id": engine.id, "request": ereq.to_dict()}, fh, ensure_ascii=False)
    code, err = run_process_group([sys.executable, "-m", WORKER_MODULE, req_path, out_path], timeout)
    if code == 124:
        raise DiarizationError("ENGINE_TIMEOUT", f"engine exceeded the timeout ({timeout:g}s) and was stopped", {"timeout": timeout})
    if not os.path.exists(out_path):
        tail = "\n".join(err.strip().splitlines()[-5:])
        raise DiarizationError("ENGINE_FAILED", f"engine worker exited {code} without a result", tail)
    with open(out_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if not payload.get("ok"):
        e = payload.get("error") or {}
        raise DiarizationError(e.get("code", "ENGINE_FAILED"), e.get("message", "engine failed"), e.get("detail"))
    return EngineResult.from_dict(payload["result"])


def worker_env() -> Dict[str, str]:
    return dict(child_env(), PYTHONPATH=os.pathsep.join(p for p in sys.path if p), **OFFLINE_ENV)


def run_process_group(argv: List[str], timeout: float) -> Tuple[int, str]:
    """Run argv in its own process group; kill the whole group on timeout (returns exit 124)."""
    kw: Dict[str, Any] = {"stdout": subprocess.DEVNULL, "stderr": subprocess.PIPE, "env": worker_env()}
    if os.name == "nt":
        kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kw["start_new_session"] = True
    proc = subprocess.Popen(argv, **kw)
    try:
        _, err = proc.communicate(timeout=timeout)
        return proc.returncode, (err or b"").decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        proc.communicate()
        return 124, ""


def _kill_tree(proc: "subprocess.Popen[bytes]") -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def _run_in_thread(engine: DiarizationEngine, ereq: EngineRequest, timeout: float) -> EngineResult:
    box: Dict[str, Any] = {}

    def target() -> None:
        try:
            box["result"] = engine.diarize(ereq)
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller's thread
            box["error"] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise DiarizationError("ENGINE_TIMEOUT", f"engine exceeded the timeout ({timeout:g}s)", {"timeout": timeout})
    if "error" in box:
        err = box["error"]
        if isinstance(err, DiarizationError):
            raise err
        raise DiarizationError("ENGINE_FAILED", f"{type(err).__name__}: {err}")
    return box["result"]
