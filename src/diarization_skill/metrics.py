"""Diarization error rate (DER), frame-based, no collar, overlap scored.

DER = (missed speech + false alarm + speaker confusion) / total reference speech, with the speaker mapping
(reference -> hypothesis) that maximises matched time. Used to measure DIARIZE against a reference
(`diarization score`); the anonymous ids of the two sides never need to agree.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Tuple

Turn = Tuple[float, float, str]


def _frames(turns: Sequence[Turn], step: float, n: int) -> List[set]:
    frames: List[set] = [set() for _ in range(n)]
    for s, e, spk in turns:
        for i in range(max(0, int(round(s / step))), min(n, int(round(e / step)))):
            frames[i].add(spk)
    return frames


def _best_mapping(ref: List[set], hyp: List[set]) -> Dict[str, str]:
    co: Dict[Tuple[str, str], int] = {}
    for r, h in zip(ref, hyp):
        for a in r:
            for b in h:
                co[(a, b)] = co.get((a, b), 0) + 1
    rs = sorted({a for a, _ in co})
    hs = sorted({b for _, b in co})
    if not rs or not hs:
        return {}
    if len(rs) <= 7 and len(hs) <= 7:
        best: Dict[str, str] = {}
        best_score = -1
        small, large, flip = (rs, hs, False) if len(rs) <= len(hs) else (hs, rs, True)
        for perm in itertools.permutations(large, len(small)):
            pairs = [(b, a) if flip else (a, b) for a, b in zip(small, perm)]
            score = sum(co.get(p, 0) for p in pairs)
            if score > best_score:
                best_score, best = score, dict(pairs)
        return best
    mapping: Dict[str, str] = {}
    used = set()
    for (a, b), _ in sorted(co.items(), key=lambda kv: -kv[1]):  # greedy for large speaker counts
        if a not in mapping and b not in used:
            mapping[a] = b
            used.add(b)
    return mapping


def der(reference: Sequence[Turn], hypothesis: Sequence[Turn], step: float = 0.01) -> Dict[str, float]:
    end = max([e for _, e, _ in list(reference) + list(hypothesis)] or [0.0])
    n = int(round(end / step)) + 1
    ref, hyp = _frames(reference, step, n), _frames(hypothesis, step, n)
    mapping = _best_mapping(ref, hyp)
    total = miss = fa = conf = 0
    for r, h in zip(ref, hyp):
        nr, nh = len(r), len(h)
        correct = sum(1 for a in r if mapping.get(a) in h)
        total += nr
        miss += max(0, nr - nh)
        fa += max(0, nh - nr)
        conf += min(nr, nh) - correct
    denom = total or 1
    return {"der": (miss + fa + conf) / denom, "missed": miss * step, "false_alarm": fa * step, "confusion": conf * step,
            "reference_speech": total * step}
