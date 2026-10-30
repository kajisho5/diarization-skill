"""Media access: fingerprint, ffprobe, and one fixed audio-extraction recipe.

The recipe is the one transcription-skill uses (its ADR-004 / ADR-030): one explicitly mapped audio
stream (`-map 0:a:N`, default the first) decoded to mono 16 kHz PCM 16-bit WAV, so diarization and
transcription see the same samples on the same timeline. Every invocation is a fixed argv list; no user
string is ever interpreted by a shell. Nothing here downloads anything.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from .errors import DiarizationError

AUDIO_RECIPE = {"channels": 1, "sample_rate": 16000, "codec": "pcm_s16le", "container": "wav"}
CHILD_ENV_KEYS = ("PATH", "HOME", "USERPROFILE", "TMPDIR", "TEMP", "TMP", "SYSTEMROOT", "LANG", "LC_ALL", "PYTHONUTF8",
                  "CUDA_VISIBLE_DEVICES", "HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME")


def child_env() -> Dict[str, str]:
    """Minimal environment for child processes: no API keys or Hugging Face tokens travel into ffmpeg or the
    engine, so the engine cannot authenticate a download even if something inside it tried."""
    return {k: os.environ[k] for k in CHILD_ENV_KEYS if k in os.environ}


def find_tool(name: str) -> Optional[str]:
    return shutil.which(name)


def tool_version(name: str) -> Optional[str]:
    path = find_tool(name)
    if not path:
        return None
    try:
        proc = subprocess.run([path, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20,
                              env=child_env())
    except (OSError, subprocess.SubprocessError):
        return None
    first = (proc.stdout or proc.stderr).strip().splitlines()[:1]
    return first[0] if first else None


def fingerprint_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def probe(path: str) -> Dict[str, Any]:
    """ffprobe summary: duration, number of audio streams, presence of video."""
    ffprobe = find_tool("ffprobe")
    if not ffprobe:
        raise DiarizationError("FFMPEG_NOT_FOUND", "ffprobe not found on PATH (install FFmpeg)")
    argv = [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path]
    try:
        proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120, env=child_env())
    except subprocess.TimeoutExpired:
        raise DiarizationError("UNSUPPORTED_MEDIA", f"ffprobe did not answer for {os.path.basename(path)}")
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-3:])
        raise DiarizationError("UNSUPPORTED_MEDIA", f"ffprobe cannot read {os.path.basename(path)}: {tail}")
    try:
        raw = json.loads(proc.stdout or "{}")
    except ValueError:
        raise DiarizationError("UNSUPPORTED_MEDIA", "ffprobe returned invalid JSON")
    fmt = raw.get("format") or {}
    streams = raw.get("streams") or []
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and (s.get("disposition") or {}).get("attached_pic", 0) == 0), None)
    if not audio_streams:
        raise DiarizationError("UNSUPPORTED_MEDIA", f"{os.path.basename(path)} has no audio stream")
    duration = _f(fmt.get("duration")) or _f(audio_streams[0].get("duration"))
    if not duration or duration <= 0:
        raise DiarizationError("UNSUPPORTED_MEDIA", f"{os.path.basename(path)}: duration unknown or zero")
    return {"duration": duration, "container": fmt.get("format_name"), "audio_stream_count": len(audio_streams),
            "has_video": video is not None}


def extraction_argv(ffmpeg: str, src: str, dst: str, audio_stream: int = 0) -> List[str]:
    """The fixed extraction argv, identical to transcription-skill's `extraction_argv`."""
    r = AUDIO_RECIPE
    return [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", src,
            "-map", f"0:a:{audio_stream}", "-vn", "-sn", "-dn",
            "-ac", str(r["channels"]), "-ar", str(r["sample_rate"]), "-c:a", r["codec"], "-f", r["container"], dst]


def extract_audio(src: str, dst: str, audio_stream: int = 0, timeout: float = 600.0) -> Dict[str, Any]:
    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        raise DiarizationError("FFMPEG_NOT_FOUND", "ffmpeg not found on PATH (install FFmpeg)")
    if os.path.abspath(src) == os.path.abspath(dst):
        raise DiarizationError("INVALID_ARGUMENT", "audio extraction target equals the input")
    try:
        proc = subprocess.run(extraction_argv(ffmpeg, src, dst, audio_stream), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, timeout=timeout, env=child_env())
    except subprocess.TimeoutExpired:
        raise DiarizationError("ENGINE_TIMEOUT", f"audio extraction exceeded {timeout:g}s")
    if proc.returncode != 0 or not os.path.exists(dst):
        tail = "\n".join(proc.stderr.strip().splitlines()[-3:])
        raise DiarizationError("UNSUPPORTED_MEDIA", f"audio extraction failed: {tail}")
    return {"tool": "ffmpeg", "recipe": dict(AUDIO_RECIPE), "audio_stream_index": audio_stream, "stream": f"0:a:{audio_stream}"}


def _f(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None
