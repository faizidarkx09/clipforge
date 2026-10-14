"""Transcription with faster-whisper (word-level timestamps).

The model is lazy-loaded and cached per (model, device) so repeated calls
are cheap. If faster-whisper is not installed, `transcribe()` raises a
clear error and the pipeline can instead use a pre-made transcript JSON
via `load_transcript()` (see pipeline opts "transcript_path").
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict

_model_cache: Dict[tuple, Any] = {}


def _get_model(model: str = "base", device: str = "cpu"):
    """Lazy-load (and cache) the faster-whisper model."""
    key = (model, device)
    if key not in _model_cache:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                "faster-whisper is not installed; cannot transcribe. "
                "Install it with: pip install faster-whisper  "
                "or supply a transcript JSON via load_transcript()."
            ) from exc
        compute_type = "int8" if device == "cpu" else "float16"
        _model_cache[key] = WhisperModel(
            model, device=device, compute_type=compute_type
        )
    return _model_cache[key]


def transcribe(video_path: str, model: str = "base", device: str = "cpu") -> dict:
    """Transcribe `video_path` with word-level timestamps.

    Returns:
        {
          "segments": [
            {"start": float, "end": float, "text": str,
             "words": [{"start": float, "end": float, "word": str}, ...]},
            ...
          ],
          "language": str,
        }
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path!r}")
    fw_model = _get_model(model, device)
    segments_iter, info = fw_model.transcribe(
        video_path, word_timestamps=True, vad_filter=True
    )
    segments = []
    for seg in segments_iter:
        words = []
        for w in seg.words or []:
            words.append(
                {"start": float(w.start), "end": float(w.end), "word": w.word}
            )
        segments.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
                "words": words,
            }
        )
    language = getattr(info, "language", "en") or "en"
    return {"segments": segments, "language": language}


def save_transcript(transcript: dict, path: str) -> str:
    """Write a transcript dict to JSON. Returns `path`."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(transcript, fh, ensure_ascii=False, indent=1)
    return path


def load_transcript(path: str) -> dict:
    """Load a transcript dict from JSON (hand-made or cached)."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or "segments" not in data:
        raise ValueError(
            f"Invalid transcript JSON at {path!r}: expected "
            "{'segments': [...], 'language': ...}"
        )
    return data
