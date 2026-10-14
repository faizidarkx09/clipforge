"""AI dubbing (beta): translate the clip's English transcript and
re-voice it with a neural TTS voice in the target language.

    dub_audio(text, "urdu", "clip_ur.wav")   # translate EN->Urdu, TTS -> wav
    replace_audio(clip_mp4, "clip_ur.wav", "clip_ur.mp4")

All third-party imports (deep_translator, edge_tts) are lazy inside the
functions so `import app` never needs them installed.
"""
from __future__ import annotations

import os
import shutil
import subprocess

# edge-tts neural voices per language.
VOICES = {
    "urdu": "ur-PK-AsadNeural",
    "hindi": "hi-IN-MadhurNeural",
    "spanish": "es-ES-AlvaroNeural",
    "french": "fr-FR-HenriNeural",
    "arabic": "ar-SA-HamedNeural",
}

# ISO-639 codes for deep_translator's GoogleTranslator.
LANG_CODES = {
    "urdu": "ur",
    "hindi": "hi",
    "spanish": "es",
    "french": "fr",
    "arabic": "ar",
}


def _translate_en(text: str, lang: str) -> str:
    """Translate English `text` to `lang` via deep_translator (lazy import)."""
    try:
        from deep_translator import GoogleTranslator
    except ImportError as exc:
        raise ImportError(
            "deep-translator is not installed; cannot dub. "
            "Install it with: pip install deep-translator"
        ) from exc
    translated = GoogleTranslator(source="en", target=LANG_CODES[lang]).translate(text)
    if not translated or not translated.strip():
        raise RuntimeError(f"Translation to {lang!r} returned empty text.")
    return translated


def _tts_to_mp3(text: str, voice: str, mp3_path: str) -> str:
    """Neural TTS via edge-tts (lazy import); writes an MP3 file."""
    try:
        import asyncio
        import edge_tts
    except ImportError as exc:
        raise ImportError(
            "edge-tts is not installed; cannot dub. "
            "Install it with: pip install edge-tts"
        ) from exc

    async def _speak() -> None:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(mp3_path)

    asyncio.run(_speak())
    if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) == 0:
        raise RuntimeError(f"edge-tts produced no audio for voice {voice!r}.")
    return mp3_path


def dub_audio(text: str, lang: str, out_wav: str) -> str:
    """Translate `text` (English) and synthesize it in `lang`.

    Writes a 44.1 kHz stereo WAV to `out_wav` and returns it.
    Raises on any failure (missing deps, network, empty text, bad lang).
    """
    if lang not in LANG_CODES:
        raise ValueError(
            f"Unsupported dub language {lang!r}; choose from {sorted(LANG_CODES)}"
        )
    if not (text or "").strip():
        raise ValueError("dub_audio needs non-empty text to dub.")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is not installed or not on PATH.")

    translated = _translate_en(text.strip(), lang)

    tmp_mp3 = os.path.abspath(out_wav) + ".dub_tmp.mp3"
    try:
        _tts_to_mp3(translated, VOICES[lang], tmp_mp3)
        # edge-tts emits MP3; normalize to WAV so downstream stays simple.
        cmd = [
            "ffmpeg", "-y", "-i", tmp_mp3,
            "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2",
            out_wav,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not os.path.exists(out_wav):
            err = (proc.stderr or "").strip()
            raise RuntimeError(f"dub wav conversion failed: {err[-400:]}")
        return out_wav
    finally:
        try:
            os.remove(tmp_mp3)
        except OSError:
            pass


def replace_audio(video_src: str, wav: str, dst: str) -> str:
    """Swap `video_src`'s audio track for `wav`; video stream is copied.

    Returns `dst`. Raises RuntimeError on ffmpeg failure.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is not installed or not on PATH.")
    for p in (video_src, wav):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Input not found: {p!r}")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_src, "-i", wav,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy",
        "-c:a", "aac", "-ar", "44100", "-ac", "2",
        "-shortest",
        "-movflags", "+faststart",
        dst,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(dst):
        err = (proc.stderr or "").strip()
        raise RuntimeError(f"replace_audio failed: {err[-600:]}")
    return dst
