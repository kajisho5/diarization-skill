"""Long recordings (ADR-003): DIARIZE in chunks, with speakers linked across chunks by their embeddings.

- The media is split into `n = ceil(duration / chunk_seconds)` equal chunks. Each chunk is extracted on
  its own (the fixed recipe with `-ss start -t length` before `-i`), so an 8 h file never becomes one
  900 MB WAV, and the engine runs on each in the worker with the timeout per chunk.
- Speaker counts: `num_speakers` becomes `max_speakers` for a chunk (a chunk may hold fewer people);
  `min_speakers` is not passed to chunks; `max_speakers` is.
- Linking: chunk 0's speakers start the global speakers. For each later chunk, cosine similarity between
  every chunk speaker and every global speaker's running centroid (weighted by speech seconds); pairs
  at or above `link_threshold` are taken greedily, highest first, one to one (two speakers of one chunk
  are two people by the engine's own decision). A chunk speaker left over, or without an embedding,
  becomes a new global speaker. Every decision is in `chunking.links` with its similarity and the margin
  over the next-best global speaker.
- Resumable: with `work_dir`, each finished chunk's engine result is saved as `chunk_NNNN.json` with a key
  (source fingerprint, stream, engine, model, revision, speaker counts, chunk bounds). A rerun with the same
  key reuses it; a file with another key is refused, never overwritten.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .engines.base import DiarizationEngine, EngineRequest, EngineResult
from .errors import DiarizationError
from .voices import cosine

LINK_THRESHOLD = 0.5
MIN_CHUNK_SECONDS = 1.0
Turn = Tuple[float, float, str]


def plan_chunks(duration: float, chunk_seconds: float) -> List[Tuple[float, float]]:
    n = max(1, math.ceil(duration / chunk_seconds - 1e-9))
    step = duration / n
    bounds = [round(i * step, 3) for i in range(n)] + [round(duration, 3)]
    return list(zip(bounds, bounds[1:]))


def chunk_key(fields: Dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(fields, sort_keys=True).encode("utf-8")).hexdigest()


def _load_saved(path: str, key: str) -> Optional[EngineResult]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            saved = json.load(fh)
    except (OSError, ValueError):
        raise DiarizationError("INVALID_ARGUMENT", f"work dir file {os.path.basename(path)} is unreadable; remove it to rerun the chunk")
    if saved.get("key") != key:
        raise DiarizationError("INVALID_ARGUMENT", f"work dir holds {os.path.basename(path)} from another run (other file, "
                               "engine or parameters); use an empty --work-dir")
    return EngineResult.from_dict(saved["result"])


def _save(path: str, key: str, result: EngineResult) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"key": key, "result": result.to_dict()}, fh)
    os.replace(tmp, path)


def run_chunks(*, media: str, duration: float, chunk_seconds: float, link_threshold: float, work_dir: Optional[str],
               run_dir: str, key_fields: Dict[str, Any], base: EngineRequest, engine: DiarizationEngine,
               extract: Callable[[str, float, float], Dict[str, Any]],
               run: Callable[[EngineRequest], EngineResult]) -> Dict[str, Any]:
    """-> {exclusive, regular, embeddings (global label -> centroid), chunks, links (global labels), extraction, warnings}."""
    if not chunk_seconds or chunk_seconds < MIN_CHUNK_SECONDS:
        raise DiarizationError("INVALID_ARGUMENT", f"--chunk-seconds must be >= {MIN_CHUNK_SECONDS:g}")
    if not -1 <= link_threshold <= 1:
        raise DiarizationError("INVALID_ARGUMENT", "--link-threshold must be in [-1, 1]")
    if work_dir is not None:
        os.makedirs(work_dir, exist_ok=True)
    plan = plan_chunks(duration, chunk_seconds)
    params = {"num_speakers": None, "min_speakers": None,
              "max_speakers": base.num_speakers if base.num_speakers is not None else base.max_speakers}
    warnings: List[str] = []
    chunks: List[Dict[str, Any]] = []
    links: List[Dict[str, Any]] = []
    exclusive: List[Turn] = []
    regular: List[Turn] = []
    centroids: Dict[str, List[float]] = {}   # global label -> speech-weighted sum of its chunk vectors
    n_global = 0
    extraction: Dict[str, Any] = {}
    for i, (start, end) in enumerate(plan):
        key = chunk_key(dict(key_fields, chunk_parameters=params, chunk=i, chunks=len(plan), start=start, end=end))
        saved_path = os.path.join(work_dir, f"chunk_{i:04d}.json") if work_dir else None
        result = _load_saved(saved_path, key) if saved_path else None
        resumed = result is not None
        t0 = time.time()
        if result is None:
            wav = os.path.join(run_dir, f"chunk_{i:04d}.wav")
            extraction = extract(wav, start, end - start)
            req = EngineRequest(audio_path=wav, pipeline_dir=base.pipeline_dir, device=base.device, **params)
            try:
                result = run(req)
            finally:
                if os.path.exists(wav):
                    os.remove(wav)
            if result.embeddings is None:
                raise DiarizationError("ENGINE_FAILED", f"engine {engine.id} gives no speaker embeddings; chunked diarization "
                                       "needs them to link speakers across chunks")
            if saved_path:
                _save(saved_path, key, result)
        # engine times are relative to the chunk; clip to it so neighbouring chunks never overlap
        length = end - start
        result.exclusive = _clip(result.exclusive, length)
        result.regular = None if result.regular is None else _clip(result.regular, length)
        # local labels -> global labels
        speech: Dict[str, float] = {}
        for s, e, lab in result.exclusive:
            speech[lab] = speech.get(lab, 0.0) + max(0.0, e - s)
        first: Dict[str, float] = {}
        for s, _e, lab in result.exclusive + (result.regular or []):
            first[lab] = min(s, first.get(lab, s))
        local = sorted(first, key=lambda lab: (first[lab], lab))  # by first appearance in the chunk
        emb = result.embeddings or {}
        sims: List[Tuple[float, str, str]] = []
        for lab in local:
            if lab in emb:
                for g in centroids:
                    sims.append((cosine(emb[lab], centroids[g]), lab, g))
        mapping: Dict[str, str] = {}
        taken = set()
        for sim, lab, g in sorted(sims, key=lambda x: (-x[0], x[1], x[2])):
            if sim < link_threshold or lab in mapping or g in taken:
                continue
            mapping[lab] = g
            taken.add(g)
        for lab in local:
            own = [(sim, g) for sim, l2, g in sims if l2 == lab]
            if lab in mapping:
                g = mapping[lab]
                sim = next(x for x, g2 in own if g2 == g)
                others = [x for x, g2 in own if g2 != g]
                links.append({"chunk": i, "local": lab, "global": g, "linked": True, "similarity": round(sim, 4),
                              "margin": None if not others else round(sim - max(others), 4)})
            else:
                g = mapping[lab] = f"G{n_global:04d}"
                n_global += 1
                if lab not in emb:
                    warnings.append(f"chunk {i}: no embedding for one speaker; it starts a new speaker and cannot be linked")
                links.append({"chunk": i, "local": lab, "global": g, "linked": False,
                              "similarity": None if not own else round(max(x for x, _ in own), 4), "margin": None})
            if lab in emb:
                w = max(speech.get(lab, 0.0), 1e-3)
                prev = centroids.get(g) or [0.0] * len(emb[lab])
                centroids[g] = [c + w * x for c, x in zip(prev, emb[lab])]
        exclusive.extend((round(s + start, 3), round(e + start, 3), mapping[lab]) for s, e, lab in result.exclusive)
        if result.regular is not None:
            regular.extend((round(s + start, 3), round(e + start, 3), mapping[lab]) for s, e, lab in result.regular)
        chunks.append({"index": i, "start": start, "end": end, "speakers": len(local), "resumed": resumed,
                       "engine_seconds": None if resumed else round(time.time() - t0, 3), "engine_version": result.engine_version})
        last = result
    return {"exclusive": merge_adjacent(exclusive), "regular": merge_adjacent(regular) if last.regular is not None else None,
            "embeddings": centroids, "chunks": chunks, "links": links, "plan_parameters": params, "extraction": extraction,
            "warnings": warnings, "engine_version": last.engine_version, "model": last.model, "extra": last.extra}


def _clip(turns: List[Turn], length: float) -> List[Turn]:
    out = []
    for s, e, lab in turns:
        s, e = max(0.0, float(s)), min(length, float(e))
        if e > s:
            out.append((s, e, lab))
    return out


def merge_adjacent(turns: List[Turn], gap: float = 0.002) -> List[Turn]:
    """Join a speaker's turn that a chunk boundary cut in two (same speaker, end == next start)."""
    out: List[Turn] = []
    for s, e, lab in sorted(turns, key=lambda t: (t[0], t[1])):
        hit = next((k for k in range(len(out) - 1, -1, -1) if out[k][2] == lab and abs(out[k][1] - s) <= gap), None)
        if hit is not None:
            out[hit] = (out[hit][0], max(out[hit][1], e), lab)
        else:
            out.append((s, e, lab))
    return out
