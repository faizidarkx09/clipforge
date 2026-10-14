"""Export: titles/hashtags/descriptions + full clip rendering.

export_clip() orchestrates the render chain for one candidate:
    cut segment -> [AI trim] -> reframe to AR (+ burn karaoke captions)
    -> polish (punch zooms, whoosh SFX, loudnorm)
    -> [AI dub: replace audio] -> [logo watermark] -> final MP4
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter

from captions import write_ass
from polish import apply_polish
from reframe import compute_vertical_crop, reframe_to_vertical
from trim import apply_trim, find_cuts, shift_words

# Supported output aspect ratios -> (width, height).
ASPECTS = {
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "16:9": (1920, 1080),
}

# Common English stopwords; subtracted before keyword frequency counting.
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further",
    "once", "here", "there", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "can", "will", "just", "don", "should", "now", "is",
    "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "having", "do", "does", "did", "doing", "would", "could",
    "ought", "i", "me", "my", "myself", "we", "our", "ours", "you",
    "your", "yours", "he", "him", "his", "she", "her", "hers", "it",
    "its", "they", "them", "their", "theirs", "this", "that", "these",
    "those", "am", "what", "which", "who", "whom", "because", "until",
    "while", "of", "as", "get", "got", "getting", "go", "going", "goes",
    "went", "like", "really", "actually", "well", "much", "many",
    "thing", "things", "something", "anything", "everything", "one",
    "two", "also", "even", "still", "way", "lot", "lots", "make",
    "made", "take", "people", "know", "think", "see", "look", "want",
}

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z']{2,}")


def make_title_hashtags(text: str) -> tuple[str, list[str]]:
    """Keyword extraction via word frequency minus stopwords.

    Returns (title, hashtags):
      - title: top keywords arranged as a punchy Title Case line (<=60 chars)
      - hashtags: 5-8 tags built from the top keywords
    """
    words = [w.lower() for w in _WORD_RE.findall(text or "")]
    freq = Counter(w for w in words if w not in STOPWORDS and len(w) > 2)
    top = [w for w, _ in freq.most_common(12)]

    if not top:
        return "Untitled Clip", ["#shorts", "#viral", "#fyp", "#trending", "#clip"]

    title = " ".join(w.capitalize() for w in top[:4])[:60].rstrip()
    tags = [f"#{w}" for w in top[:8]]
    # Pad with generic reach tags if the transcript was thin.
    for generic in ("#shorts", "#viral", "#fyp"):
        if len(tags) >= 8:
            break
        if generic not in tags:
            tags.append(generic)
    return title, tags[:8]


def make_description(quote: str, title: str) -> str:
    """Build a short post description from a clip quote + title.

    Pure python, no API: 2-3 sentences (quote snippet + title line) plus a
    call-to-action line with #shorts.
    """
    words = (quote or "").split()
    snippet = " ".join(words[:25])
    if len(words) > 25:
        snippet += "..."
    title = (title or "Untitled Clip").strip() or "Untitled Clip"
    return (
        f'"{snippet}"\n\n'
        f"{title} — the moment from the full video worth clipping "
        f"and sharing.\n\n"
        "Follow for more clips like this. #shorts"
    )


def _overlay_logo(src: str, logo_path: str, dst: str) -> str:
    """Overlay a PNG logo bottom-right (120px wide, 40px margins).

    Returns `dst`. Raises RuntimeError on ffmpeg failure.
    """
    vf = (
        "[1:v]scale=120:-1[lg];"
        "[0:v][lg]overlay=W-w-40:H-h-40:format=auto,format=yuv420p"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", src, "-i", logo_path,
        "-filter_complex", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "copy",
        "-movflags", "+faststart",
        dst,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(dst):
        err = (proc.stderr or "").strip()
        raise RuntimeError(f"logo overlay failed: {err[-400:]}")
    return dst


def _cut_segment(src: str, start: float, end: float, dst: str) -> str:
    """Accurate (re-encoded) cut of [start, end] from src."""
    dur = max(0.5, end - start)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{max(0.0, start):.2f}", "-i", src,
        "-t", f"{dur:.2f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-ar", "44100", "-ac", "2",
        dst,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(dst):
        err = (proc.stderr or "").strip()
        raise RuntimeError(f"Segment cut failed: {err[-400:]}")
    return dst


def _words_in_window(transcript: dict, start: float, end: float) -> list[dict]:
    """Word timestamps overlapping [start, end], shifted to clip-relative."""
    out: list[dict] = []
    for seg in transcript.get("segments", []):
        for w in seg.get("words", []) or []:
            if w["end"] > start and w["start"] < end:
                out.append(
                    {
                        "start": max(0.0, w["start"] - start),
                        "end": min(end - start, w["end"] - start),
                        "word": w["word"],
                    }
                )
    out.sort(key=lambda w: w["start"])
    return out


def export_clip(
    source: str,
    start: float,
    end: float,
    dst: str,
    transcript: dict | None = None,
    style: str = "hormozi",
    sfx: bool = True,
    zoom: bool = True,
    crop: dict | None = None,
    progress=None,
    ar: str = "9:16",
    caption_overrides: dict | None = None,
    trim: bool = False,
    dub: dict | None = None,
    logo_path: str | None = None,
    warnings: list | None = None,
) -> str:
    """Render one clip: cut -> [trim] -> reframe (+caption burn) -> polish
    -> [dub] -> [logo] -> final MP4.

    Args:
        source: normalized video from ingest().
        start, end: clip bounds in source seconds.
        dst: final MP4 path (dimensions follow `ar`).
        transcript: full-video transcript dict (for captions); captions are
            skipped when None or when no words fall in the window.
        style: caption preset (see clipforge.captions.STYLES).
        sfx, zoom: polish toggles.
        crop: optional precomputed crop dict from compute_vertical_crop().
        progress: optional callable(stage, pct, msg).
        ar: output aspect ratio ("9:16" | "1:1" | "16:9").
        caption_overrides: optional dict passed to write_ass() (font size,
            position, colors, box, emoji).
        trim: when True, cut filler words and dead-air silences first.
        dub: optional {"lang": ...} dict; replaces the audio with a neural
            TTS dub. Failures never fail the render: a message is appended
            to `warnings` (when given) and the original audio is kept.
        logo_path: optional PNG watermarked bottom-right.
        warnings: optional list collecting non-fatal warning strings.

    Returns `dst`. Intermediate files are cleaned up automatically.
    """
    def emit(stage, pct, msg):
        if progress:
            progress(stage, pct, msg)

    if ar not in ASPECTS:
        raise ValueError(
            f"Unknown aspect ratio {ar!r}; choose from {sorted(ASPECTS)}"
        )
    out_w, out_h = ASPECTS[ar]

    tmpdir = tempfile.mkdtemp(prefix="clipforge_export_")
    try:
        emit("cut", 5, "cutting segment")
        cut = os.path.join(tmpdir, "cut.mp4")
        _cut_segment(source, start, end, cut)

        words = _words_in_window(transcript, start, end) if transcript else []

        trim_cuts: list = []
        if trim and words:
            emit("trim", 12, "trimming filler words and silences")
            cuts = find_cuts(words)
            if cuts:
                trimmed = os.path.join(tmpdir, "trimmed.mp4")
                info = apply_trim(cut, cuts, trimmed)
                emit("trim", 16,
                     f"removed {info['removed']:.1f}s in {info['cuts']} cuts")
                cut = trimmed
                words = shift_words(words, cuts)
                trim_cuts = cuts

        emit("captions", 20, "building captions")
        ass_path = None
        if words:
            ass_path = os.path.join(tmpdir, "caps.ass")
            write_ass(words, ass_path, style=style, overrides=caption_overrides)

        emit("reframe", 40, f"reframing to {ar}")
        if crop is None and ar != "16:9":
            crop = compute_vertical_crop(cut, ar=ar)
        reframed = os.path.join(tmpdir, "reframed.mp4")
        reframe_to_vertical(cut, reframed, crop=crop, ass_path=ass_path, ar=ar)

        emit("polish", 70, "applying polish")
        from scorer import build_sentences  # local import: scorer is optional-ish
        # Sentence boundaries relative to the clip for punch zooms / whooshes.
        clip_transcript = {"segments": transcript.get("segments", [])} if transcript else {"segments": []}
        sentences = []
        for s in build_sentences(clip_transcript):
            if s["end"] > start and s["start"] < end:
                sentences.append(
                    {"start": max(0.0, s["start"] - start),
                     "end": min(end - start, s["end"] - start)}
                )
        if trim_cuts:
            # Keep punch zooms / whooshes aligned with the trimmed timeline.
            sentences = shift_words(sentences, trim_cuts)

        polished = os.path.join(tmpdir, "polished.mp4")
        apply_polish(reframed, polished, sentences=sentences, sfx=sfx,
                     zoom=zoom, out_w=out_w, out_h=out_h)
        current = polished

        if dub:
            lang = dub.get("lang", "urdu") if isinstance(dub, dict) else "urdu"
            try:
                from dub import dub_audio, replace_audio  # lazy: needs pip deps
                text = " ".join(w["word"] for w in words).strip()
                if not text:
                    raise RuntimeError("no transcript words to dub")
                emit("dub", 85, f"dubbing audio to {lang}")
                wav = os.path.join(tmpdir, "dub.wav")
                dub_audio(text, lang, wav)
                dubbed = os.path.join(tmpdir, "dubbed.mp4")
                replace_audio(polished, wav, dubbed)
                current = dubbed
            except Exception as exc:
                if warnings is not None:
                    warnings.append(
                        f"Dub to {lang!r} failed, kept original audio: {exc}"
                    )

        if logo_path:
            if os.path.isfile(logo_path):
                emit("logo", 92, "adding logo watermark")
                branded = os.path.join(tmpdir, "branded.mp4")
                _overlay_logo(current, logo_path, branded)
                current = branded
            elif warnings is not None:
                warnings.append(f"Logo not found, skipped: {logo_path}")

        shutil.copyfile(current, dst)
        emit("done", 100, f"wrote {dst}")
        return dst
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
