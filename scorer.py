"""Heuristic virality scoring engine (no LLM).

Pipeline:
    transcript (word timestamps)
      -> build_sentences(): group words into sentence-level units
      -> slide candidate windows (20-70s, ~5s stride) over the timeline
      -> score each window 0-100
      -> rank_clips(): non-maximum suppression by IoU>0.5, keep top_n

Scoring rubric (max 100):
    hook strength (first 5s of window) ......... 25 pts
        question / bold claim / number-stat
    curiosity-gap + emotion lexicon hits ....... 20 pts
    storytelling markers ...................... 10 pts
    ideal length fit (30-60s sweet spot) ........ 15 pts
    audio energy (RMS loudness) ................ 10 pts
    sentence-boundary cleanliness .............. 10 pts
    filler-word penalty ........................ up to -10 pts
"""
from __future__ import annotations

import math
import re
import shutil
import subprocess
from typing import List, Optional

import numpy as np

# --- Lexicons (small, documented; tune freely) -------------------------------

# Words that open a "curiosity gap": the viewer feels they must keep watching
# to resolve the missing information.
CURIOSITY_WORDS = {
    "secret", "secrets", "hidden", "truth", "nobody", "no one", "why", "how",
    "mistake", "mistakes", "hack", "hacks", "lie", "lies", "scam", "myth",
    "wrong", "actually", "surprising", "unbelievable", "unknown", "finally",
    "revealed", "exposed", "banned",
}

# High-arousal emotion words: strong feelings drive shares/comments.
EMOTION_WORDS = {
    "amazing", "crazy", "insane", "shocking", "shocked", "incredible",
    "love", "hate", "angry", "fear", "afraid", "excited", "wow",
    "mind-blowing", "terrible", "horrible", "best", "worst", "epic",
    "ridiculous", "genius", "stupid",
}

# Storytelling discourse markers: signal a narrative arc, which holds
# retention better than disconnected statements.
STORY_WORDS = {
    "so", "then", "suddenly", "story", "remember", "finally", "next",
    "after", "before", "because", "but", "however", "meanwhile", "first",
}

# Filler / low-information words: penalized (per occurrence, capped).
FILLER_WORDS = {
    "um", "uh", "umm", "uhh", "erm", "like", "basically", "literally",
    "stuff", "things", "whatever", "anyway",
}
FILLER_BIGRAMS = {("you", "know"), ("i", "mean"), ("sort", "of"), ("kind", "of")}

# Bold-claim words: strong assertions in the hook ("nobody tells you...").
BOLD_CLAIM_WORDS = {
    "nobody", "everyone", "always", "never", "truth", "secret", "stop",
    "warning", "guaranteed", "proven", "scientists",
}

_NUMBER_RE = re.compile(r"\d")
_QUESTION_RE = re.compile(r"\?")
_TOKEN_RE = re.compile(r"[a-zA-Z']+")


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


# Stopwords-ish set for focus-prompt tokenization (ClipAnything): content
# words carry the signal, filler words don't.
_FOCUS_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "from", "into", "over", "under",
    "this", "that", "these", "those", "you", "your", "yours", "they",
    "their", "are", "was", "were", "have", "has", "had", "will", "would",
    "can", "could", "not", "all", "any", "clip", "video", "make", "get",
    "moment", "part", "time", "find", "show", "talk",
}


def _focus_tokens(prompt: str) -> set:
    """Tokenize a ClipAnything focus prompt: lowercase, len > 2, no stopwords."""
    return {
        t for t in _tokens(prompt or "")
        if len(t) > 2 and t not in _FOCUS_STOPWORDS
    }


# --- Sentence building -------------------------------------------------------

def build_sentences(transcript: dict, max_dur: float = 8.0) -> List[dict]:
    """Group word-level timestamps into sentence-like units.

    A sentence ends at terminal punctuation (. ! ?) or when the unit would
    exceed `max_dur` seconds (fallback chunking for unpunctuated speech).
    Returns [{"start", "end", "text", "words": [...]}, ...].
    """
    words: List[dict] = []
    for seg in transcript.get("segments", []):
        words.extend(seg.get("words", []) or [])
    words.sort(key=lambda w: w["start"])

    sentences: List[dict] = []
    cur: List[dict] = []

    def flush():
        if cur:
            text = "".join(
                w["word"] if w["word"].startswith(("'", "-")) else " " + w["word"]
                for w in cur
            ).strip()
            sentences.append(
                {
                    "start": cur[0]["start"],
                    "end": cur[-1]["end"],
                    "text": text,
                    "words": list(cur),
                }
            )
            cur.clear()

    for w in words:
        cur.append(w)
        word = w["word"].strip()
        dur = w["end"] - cur[0]["start"]
        if word and word[-1] in ".!?":
            flush()
        elif dur >= max_dur and len(cur) >= 4:
            flush()
    flush()
    return sentences


# --- Audio energy ------------------------------------------------------------

# Module-level cache: decode each audio file ONCE per process; window
# loudness is then just a slice + RMS. (A naive per-window ffmpeg call makes
# scoring O(windows) subprocess spawns — far too slow on long videos.)
_audio_cache: dict[str, tuple[np.ndarray, int]] = {}

_SR = 16000


def _decode_audio_mono(audio_path: str) -> tuple[np.ndarray, int] | None:
    """Decode full audio to mono float32 @16kHz, cached per path."""
    if audio_path in _audio_cache:
        return _audio_cache[audio_path]
    if not shutil.which("ffmpeg"):
        return None
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", audio_path,
        "-vn", "-ac", "1", "-ar", str(_SR), "-f", "f32le", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=300)
        if proc.returncode != 0 or not proc.stdout:
            return None
        pcm = np.frombuffer(proc.stdout, dtype=np.float32).astype(np.float64)
        if pcm.size == 0:
            return None
        _audio_cache[audio_path] = (pcm, _SR)
        return pcm, _SR
    except Exception:
        return None


def _window_loudness(
    audio: tuple[np.ndarray, int] | None, start: float, end: float
) -> float:
    """Normalized loudness 0..1 for [start, end] from decoded PCM.

    Maps roughly -50 dBFS -> 0.0 and -15 dBFS -> 1.0. Returns 0.5 when no
    audio is available (neutral, never crashes scoring).
    """
    if audio is None:
        return 0.5
    pcm, sr = audio
    a = max(0, int(start * sr))
    b = min(pcm.size, int(end * sr))
    if b <= a:
        return 0.5
    seg = pcm[a:b]
    rms = float(np.sqrt(np.mean(seg ** 2)))
    db = 20.0 * math.log10(rms + 1e-9)
    return float(np.clip((db + 50.0) / 35.0, 0.0, 1.0))


# --- Window scoring ----------------------------------------------------------

def _score_window(
    start: float,
    end: float,
    sentences: List[dict],
    audio: "tuple[np.ndarray, int] | None",
) -> tuple[float, List[str]]:
    """Score one candidate window. Returns (score, reason_parts)."""
    dur = end - start
    in_win = [s for s in sentences if s["end"] > start and s["start"] < end]
    text = " ".join(s["text"] for s in in_win)
    toks = _tokens(text)
    tokset = set(toks)
    reasons: List[str] = []
    score = 0.0

    # 1) Hook strength: first 5s of the window -------------------------------
    hook_sents = [s for s in in_win if s["start"] < start + 5.0]
    hook_text = " ".join(s["text"] for s in hook_sents)
    hook = 0.0
    if _QUESTION_RE.search(hook_text):
        hook += 12.0
        reasons.append("opens with a question")
    if tokset & BOLD_CLAIM_WORDS or (set(_tokens(hook_text)) & BOLD_CLAIM_WORDS):
        hook += 7.0
        reasons.append("bold claim in hook")
    if _NUMBER_RE.search(hook_text):
        hook += 6.0
        reasons.append("number/stat in hook")
    score += min(25.0, hook)

    # 2) Curiosity + emotion lexicon hits -----------------------------------
    hits = sorted((tokset & CURIOSITY_WORDS) | (tokset & EMOTION_WORDS))
    ce = min(20.0, 4.0 * len(hits))
    score += ce
    if hits:
        reasons.append(f"curiosity/emotion words ({', '.join(hits[:4])})")

    # 3) Storytelling markers ----------------------------------------------
    story_hits = tokset & STORY_WORDS
    st = min(10.0, 3.0 * len(story_hits))
    score += st
    if story_hits:
        reasons.append("narrative flow")

    # 4) Ideal length fit: 30-60s sweet spot --------------------------------
    if 30.0 <= dur <= 60.0:
        score += 15.0
    elif 20.0 <= dur <= 70.0:
        score += 10.0
        reasons.append("good length")
    else:
        score += 5.0

    # 5) Audio energy bonus --------------------------------------------------
    loud = _window_loudness(audio, start, end)
    score += 10.0 * loud
    if loud > 0.7:
        reasons.append("high vocal energy")

    # 6) Sentence-boundary cleanliness --------------------------------------
    bounds = {round(s["start"], 2) for s in sentences} | {
        round(s["end"], 2) for s in sentences
    }
    clean = 0.0
    for edge in (start, end):
        if any(abs(edge - b) <= 0.75 for b in bounds):
            clean += 5.0
    score += clean
    if clean >= 10.0:
        reasons.append("clean sentence boundaries")

    # 7) Filler penalty ------------------------------------------------------
    filler = sum(1 for t in toks if t in FILLER_WORDS)
    for a, b in zip(toks, toks[1:]):
        if (a, b) in FILLER_BIGRAMS:
            filler += 1
    penalty = min(10.0, 2.0 * filler)
    score -= penalty
    if penalty >= 4.0:
        reasons.append(f"{filler} filler words")

    return float(np.clip(score, 0.0, 100.0)), reasons


def _iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def rank_clips(
    transcript: dict,
    audio_path: Optional[str] = None,
    top_n: int = 5,
    min_dur: float = 20.0,
    max_dur: float = 70.0,
    focus_prompt: str = "",
) -> List[dict]:
    """Rank candidate clips by virality score.

    `focus_prompt` (ClipAnything): free-text keywords; each window gets +8
    per overlapping keyword (capped at +24), so clips about the requested
    topic rank higher.

    Returns up to `top_n` dicts:
        {"start", "end", "duration", "score", "reason", "quote"}
    with overlapping candidates removed (NMS, IoU > 0.5 keeps the best).
    """
    sentences = build_sentences(transcript)
    if not sentences:
        return []
    total = sentences[-1]["end"]
    audio = _decode_audio_mono(audio_path) if audio_path else None
    focus_toks = _focus_tokens(focus_prompt)

    # Slide windows of several durations over the timeline (~5s stride).
    durations = sorted({d for d in (25.0, 40.0, 60.0)
                       if min_dur <= d <= max_dur}) or [min_dur]
    stride = 5.0
    candidates: List[dict] = []
    t = 0.0
    while t < total - min_dur + 1e-6:
        for d in durations:
            end = min(t + d, total)
            if end - t < min_dur - 1e-6:
                continue
            score, reasons = _score_window(t, end, sentences, audio)
            in_win = [s for s in sentences if s["end"] > t and s["start"] < end]
            if focus_toks:
                hits = focus_toks & set(
                    _tokens(" ".join(s["text"] for s in in_win))
                )
                if hits:
                    bonus = min(24.0, 8.0 * len(hits))
                    score = min(100.0, score + bonus)
                    reasons.append(
                        f"focus keywords ({', '.join(sorted(hits)[:3])})"
                    )
            quote = (in_win[0]["text"] if in_win else "")[:140]
            candidates.append(
                {
                    "start": round(t, 2),
                    "end": round(end, 2),
                    "duration": round(end - t, 2),
                    "score": round(score, 1),
                    "reason": "; ".join(reasons) if reasons else "steady segment",
                    "quote": quote,
                }
            )
        t += stride

    # Non-maximum suppression: sort by score, drop IoU > 0.5 overlaps.
    candidates.sort(key=lambda c: c["score"], reverse=True)
    kept: List[dict] = []
    for c in candidates:
        if all(
            _iou(c["start"], c["end"], k["start"], k["end"]) <= 0.5 for k in kept
        ):
            kept.append(c)
        if len(kept) >= top_n:
            break
    return kept[:top_n]
