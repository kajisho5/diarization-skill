"""The local pipeline directory (ADR-001): where it is, whether it is complete, which revision it is.

Read-only file inspection; nothing is imported from pyannote and nothing is downloaded. The revision comes
from the directory itself: a Hugging Face cache snapshot is named by its commit, and a `git clone` records
its commit in `.git` (read as files, git is never run).
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

ENV_PIPELINE_DIR = "DIARIZATION_PIPELINE_DIR"
HF_REPO = "pyannote/speaker-diarization-community-1"
HF_URL = "https://huggingface.co/" + HF_REPO
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec"
WEIGHT_SUFFIXES = (".bin", ".safetensors", ".ckpt", ".pt", ".npz", ".onnx")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")

INSTALL_STEPS = [
    f"1. Sign in to Hugging Face and accept the user conditions at {HF_URL} (one time).",
    "2. Clone the pipeline once, with Git LFS: git lfs install && "
    f"git clone {HF_URL} ~/models/speaker-diarization-community-1 "
    "(Git asks for your Hugging Face user name and an access token with read scope; the skill never sees it).",
    f"3. export {ENV_PIPELINE_DIR}=~/models/speaker-diarization-community-1 (the skill only reads this directory, offline).",
]


def resolve_dir(explicit: Optional[str] = None) -> Optional[str]:
    value = explicit or os.environ.get(ENV_PIPELINE_DIR) or ""
    return os.path.abspath(os.path.expanduser(value)) if value else None


def read_revision(directory: str) -> Optional[str]:
    """Commit id of the pipeline snapshot, or None when the directory does not say."""
    name = os.path.basename(os.path.normpath(directory))
    if SHA_RE.match(name) and os.path.basename(os.path.dirname(os.path.normpath(directory))) == "snapshots":
        return name
    git = os.path.join(directory, ".git")
    if os.path.isfile(git):  # worktree / submodule: "gitdir: <path>"
        text = _read_text(git) or ""
        if text.startswith("gitdir:"):
            git = os.path.join(directory, text.split(":", 1)[1].strip())
    head = _read_text(os.path.join(git, "HEAD"))
    if head is None:
        return None
    head = head.strip()
    if SHA_RE.match(head):
        return head
    if not head.startswith("ref:"):
        return None
    ref = head.split(":", 1)[1].strip()
    loose = _read_text(os.path.join(git, *ref.split("/")))
    if loose and SHA_RE.match(loose.strip()):
        return loose.strip()
    for line in (_read_text(os.path.join(git, "packed-refs")) or "").splitlines():
        parts = line.strip().split(" ")
        if len(parts) == 2 and parts[1] == ref and SHA_RE.match(parts[0]):
            return parts[0]
    return None


def inspect(directory: Optional[str]) -> Dict[str, Any]:
    """What the directory holds. `status`: OK, MISSING (unset or absent) or INCOMPLETE (no config.yaml, or
    weights that are still Git LFS pointers)."""
    info: Dict[str, Any] = {"env": ENV_PIPELINE_DIR, "dir": directory, "status": "MISSING", "config": False,
                            "pipeline_class": None, "revision": None, "weight_files": 0, "lfs_pointers": [], "detail": None}
    if not directory:
        info["detail"] = f"{ENV_PIPELINE_DIR} is not set"
        return info
    if not os.path.isdir(directory):
        info["detail"] = f"{ENV_PIPELINE_DIR} points to a directory that does not exist"
        return info
    config = os.path.join(directory, "config.yaml")
    info["config"] = os.path.isfile(config)
    if info["config"]:
        text = _read_text(config) or ""
        m = re.search(r"^pipeline:\s*\n(?:[ \t]+.*\n)*?[ \t]+name:\s*([\w.]+)", text + "\n", re.M)
        info["pipeline_class"] = m.group(1) if m else None
    info["revision"] = read_revision(directory)
    pointers: List[str] = []
    weights = 0
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d != ".git"]
        for f in files:
            if not f.endswith(WEIGHT_SUFFIXES):
                continue
            path = os.path.join(root, f)
            if _is_lfs_pointer(path):
                pointers.append(os.path.relpath(path, directory).replace(os.sep, "/"))
            else:
                weights += 1
    info["weight_files"] = weights
    info["lfs_pointers"] = sorted(pointers)
    if not info["config"]:
        info.update(status="INCOMPLETE", detail="no config.yaml: not a pyannote pipeline directory")
    elif pointers:
        info.update(status="INCOMPLETE", detail="weights are Git LFS pointers; run `git lfs pull` in the directory")
    elif weights == 0:
        info.update(status="INCOMPLETE", detail="no weight files found next to config.yaml")
    else:
        info["status"] = "OK"
    return info


def _is_lfs_pointer(path: str) -> bool:
    try:
        if os.path.getsize(path) > 1024:
            return False
        with open(path, "rb") as fh:
            return fh.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX
    except OSError:
        return False


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except (OSError, UnicodeDecodeError):
        return None
