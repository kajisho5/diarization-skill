"""`diarization doctor`: what is installed, what is missing and how to get it. Never downloads, never
imports torch or pyannote (only looks them up), so it is fast and safe to run anywhere."""
from __future__ import annotations

import importlib.util
import platform
import sys
from typing import Any, Dict, Optional

from . import __version__
from .media import find_tool, tool_version
from .pipeline import ENV_PIPELINE_DIR, INSTALL_STEPS, inspect, resolve_dir


def _dist(name: str, module: str) -> Dict[str, Any]:
    try:
        found = importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        found = False
    version: Optional[str] = None
    if found:
        try:
            from importlib.metadata import PackageNotFoundError, version as _v
            try:
                version = _v(name)
            except PackageNotFoundError:
                version = None
        except ImportError:  # pragma: no cover - Python < 3.8
            version = None
    return {"found": found, "version": version}


# pyannote.audio's own Requires-Python: 4.0.x needs >= 3.10 and 4.1+ needs >= 3.11. pip picks the newest 4.x that
# fits the interpreter, so 3.10 gets 4.0.x and 3.11+ gets 4.1+. No 4.x release installs on 3.9.
PYANNOTE_PYTHON_NOTE = ("pyannote.audio 4.x needs Python >= 3.10 (4.0.x); 4.1 and later need Python >= 3.11, "
                        "so pip installs 4.0.x on 3.10; assign/check run on 3.9")
PYANNOTE_INSTALL = "pip install 'diarization-skill[pyannote]'  (pyannote.audio 4.x; Python >= 3.10, >= 3.11 for 4.1+)"


def pyannote_install_hint(version_info: Any = None) -> str:
    """The install step for pyannote.audio on this interpreter. On Python < 3.10 no pyannote.audio 4.x
    release installs, so the step is a newer Python first, not a pip command that cannot resolve."""
    vi = tuple(version_info if version_info is not None else sys.version_info)[:2]
    if vi < (3, 10):
        return ("Python >= 3.10 first (no pyannote.audio 4.x release installs on Python %d.%d; 4.1+ needs >= 3.11), "
                "then %s" % (vi[0], vi[1], PYANNOTE_INSTALL))
    return PYANNOTE_INSTALL


def doctor(pipeline_dir: Optional[str] = None) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    py_ok = sys.version_info >= (3, 9)
    checks["python"] = {"status": "OK" if py_ok else "MISSING", "version": platform.python_version(),
                        "note": PYANNOTE_PYTHON_NOTE,
                        "install": None if py_ok else "Python >= 3.9 (>= 3.10 for the pyannote engine, >= 3.11 for pyannote.audio 4.1+)"}

    ff, fp = find_tool("ffmpeg"), find_tool("ffprobe")
    checks["ffmpeg"] = {"status": "OK" if ff and fp else "MISSING", "ffmpeg": tool_version("ffmpeg") if ff else None,
                        "ffprobe": bool(fp), "used_for": "probe + fixed mono 16 kHz extraction (diarize only)",
                        "install": None if ff and fp else "install FFmpeg (apt install ffmpeg / brew install ffmpeg / "
                                                          "choco install ffmpeg) so ffmpeg and ffprobe are on PATH"}

    pa = _dist("pyannote.audio", "pyannote.audio")
    major = None
    if pa["version"]:
        try:
            major = int(pa["version"].split(".")[0])
        except ValueError:
            major = None
    pa_status = "MISSING" if not pa["found"] else ("OK" if major == 4 else "INCOMPATIBLE")
    checks["pyannote_audio"] = dict(pa, status=pa_status, required=">=4.0,<5",
                                    install=None if pa_status == "OK" else pyannote_install_hint())

    th = _dist("torch", "torch")
    checks["torch"] = dict(th, status="OK" if th["found"] else "MISSING",
                           install=None if th["found"] else "installed with pyannote.audio (pip install 'diarization-skill[pyannote]')")

    info = inspect(resolve_dir(pipeline_dir))
    info["install"] = None if info["status"] == "OK" else INSTALL_STEPS
    if info["status"] == "OK" and info["revision"] is None:
        info["note"] = "revision unknown: results will carry engine.revision = null"
    checks["pipeline"] = info

    diarize_ready = all(checks[k]["status"] == "OK" for k in ("python", "ffmpeg", "pyannote_audio", "torch", "pipeline"))
    problems = [f"{k}: {v['status']}" + (f" ({v.get('detail')})" if v.get("detail") else "")
                for k, v in checks.items() if v["status"] != "OK"]
    return {
        "ok": diarize_ready,
        "skill": "diarization-skill",
        "version": __version__,
        "ready": {"diarize": diarize_ready, "assign": py_ok, "check": py_ok},
        "checks": checks,
        "problems": problems,
        "network": "never used at run time; the pipeline is loaded from " + ENV_PIPELINE_DIR,
    }
