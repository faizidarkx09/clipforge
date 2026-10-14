"""AI trim: remove filler words and dead-air silences from a clip.

Workflow:
    cuts  = find_cuts(words)          # (start, end) ranges to REMOVE
    info  = apply_trim(src, cuts, dst) # ffmpeg select/aselect keep-complement
    words = shift_words(words, cuts)   # retime word timestamps to the trimmed video

Light module: stdlib + subprocess only (ffmpeg/ffprobe on PATH).
"""
from __future__ import annotations

import os
import shutil
import subprocess

# Single filler words, matched case-insensitively with punctuation stripped.
FILLERS = {
    "um", "uh", "erm", "ah", "like", "basically", "actually",
    "literally", "stuff", "anyway", "right", "okay", "so",
}

# Multi-word filler phrases, matched over consecutive words.
FILLER_PHRASES = ["you know", "i mean", "kind of", "sort of"]

_FILLER_PAD = 0.05   # seconds of padding around filler spans
_SILENCE_MIN = 0.7   # gaps longer than this are cut ...
_SILENCE_KEEP = 0.15  # ... keeping this much breathing room on each side

_WORD_STRIP = ".,!?;:'\"()[]-–—"


def _norm(word: str) -> str:
    return word.strip().lower().strip(_WORD_STRIP)


def _merge_ranges(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sort and merge overlapping/touching ranges; drop degenerate ones."""
    cleaned = sorted(
        (s, e) for s, e in ranges if e - s > 1e-3
    )
    merged: list[list[float]] = []
    for s, e in cleaned:
        if merged and s <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def find_cuts(words: list[dict]) -> list[tuple[float, float]]:
    """Find (start, end) ranges to REMOVE from the clip.

    - filler word spans (padded by 0.05s)
    - filler phrase spans (ditto)
    - inter-word silences > 0.7s: the middle is cut, 0.15s kept each side

    Returns merged, time-sorted ranges. Empty list means "nothing to trim".
    """
    words = [w for w in words if w.get("word", "").strip()]
    words.sort(key=lambda w: w["start"])
    cuts: list[tuple[float, float]] = []

    # --- filler words ---------------------------------------------------
    for w in words:
        if _norm(w["word"]) in FILLERS:
            s = max(0.0, w["start"] - _FILLER_PAD)
            cuts.append((s, w["end"] + _FILLER_PAD))

    # --- filler phrases (consecutive-word match) ------------------------
    toks = [_norm(w["word"]) for w in words]
    for phrase in FILLER_PHRASES:
        parts = phrase.split()
        n = len(parts)
        for i in range(len(toks) - n + 1):
            if toks[i:i + n] == parts:
                s = max(0.0, words[i]["start"] - _FILLER_PAD)
                cuts.append((s, words[i + n - 1]["end"] + _FILLER_PAD))

    # --- dead-air silences ----------------------------------------------
    for prev, nxt in zip(words, words[1:]):
        gap = nxt["start"] - prev["end"]
        if gap > _SILENCE_MIN:
            cuts.append((prev["end"] + _SILENCE_KEEP, nxt["start"] - _SILENCE_KEEP))

    return _merge_ranges(cuts)


def _probe_duration(src: str) -> float:
    """Video duration in seconds via ffprobe; huge fallback if unavailable."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", src],
            capture_output=True, text=True, timeout=30,
        )
        return float(out.stdout.strip())
    except Exception:
        return 86400.0


def apply_trim(src: str, cuts: list[tuple[float, float]], dst: str) -> dict:
    """Remove `cuts` from `src` and write the compacted clip to `dst`.

    Uses ffmpeg select/aselect with the keep-complement of the cut ranges.
    With no cuts, stream-copies src -> dst (fast, lossless).

    Returns {"removed": seconds_removed, "cuts": number_of_cuts}.
    Raises RuntimeError on ffmpeg failure.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is not installed or not on PATH.")
    if not os.path.isfile(src):
        raise FileNotFoundError(f"Video not found: {src!r}")

    cuts = _merge_ranges(list(cuts))
    removed = sum(e - s for s, e in cuts)

    if not cuts:
        cmd = ["ffmpeg", "-y", "-i", src, "-c", "copy", dst]
    else:
        # Complement of the cut ranges = the parts to KEEP.
        keep: list[tuple[float, float]] = []
        prev_end = 0.0
        for s, e in cuts:
            if s - prev_end > 1e-3:
                keep.append((prev_end, s))
            prev_end = max(prev_end, e)
        dur = _probe_duration(src)
        if dur - prev_end > 1e-3:
            keep.append((prev_end, dur))
        if not keep:  # everything was cut; keep a sliver so ffmpeg has output
            keep = [(0.0, 0.1)]

        expr = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in keep)
        # Single quotes inside one argv element are safe for ffmpeg's parser;
        # they keep the commas inside between() from splitting filters.
        vf = f"select='{expr}',setpts=N/FRAME_RATE/TB"
        af = f"aselect='{expr}',asetpts=N/SR/TB"
        cmd = [
            "ffmpeg", "-y", "-i", src,
            "-vf", vf, "-af", af,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-ar", "44100", "-ac", "2",
            dst,
        ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(dst):
        err = (proc.stderr or "").strip()
        raise RuntimeError(f"apply_trim failed: {err[-600:]}")
    return {"removed": round(removed, 2), "cuts": len(cuts)}


def _removed_before(t: float, cuts: list[tuple[float, float]]) -> float:
    """Total cut duration that falls strictly before time t (partial at t)."""
    total = 0.0
    for s, e in cuts:
        if e <= t:
            total += e - s
        elif s < t:
            total += t - s
            break
        else:
            break
    return total


def shift_words(words: list[dict], cuts: list[tuple[float, float]]) -> list[dict]:
    """Retime word timestamps to the trimmed timeline.

    Each word's start/end is shifted back by the total removed duration
    before it; words fully inside a cut collapse to zero length and are
    dropped. Returns a new list (input dicts are not mutated).
    """
    cuts = _merge_ranges(list(cuts))
    if not cuts:
        return [dict(w) for w in words]
    out: list[dict] = []
    for w in words:
        ns = w["start"] - _removed_before(w["start"], cuts)
        ne = w["end"] - _removed_before(w["end"], cuts)
        if ne - ns > 0.01:
            shifted = dict(w)
            shifted["start"] = max(0.0, ns)
            shifted["end"] = max(0.0, ne)
            out.append(shifted)
    out.sort(key=lambda w: w["start"])
    return out
