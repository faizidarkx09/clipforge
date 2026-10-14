"""Ingest: accept a local file path or a YouTube URL and produce a
normalized MP4 that the rest of the pipeline can rely on.

Normalization target (CPU-friendly):
    - container: MP4, faststart
    - video: H.264, yuv420p, <= 30fps, height <= 720px
    - audio: AAC stereo, 44.1 kHz
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import uuid
from urllib.parse import urlparse


class IngestError(Exception):
    """Raised when a source cannot be ingested (bad URL, missing file,
    unsupported format, missing tool, ...)."""


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def is_url(source: str) -> bool:
    """True if `source` looks like an http(s) URL."""
    try:
        parts = urlparse(source)
        return parts.scheme in ("http", "https") and bool(parts.netloc)
    except Exception:
        return False


def _run(cmd: list[str], what: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise IngestError(f"{what}: required tool not found: {cmd[0]}") from exc


def _download_youtube(url: str, workdir: str) -> str:
    """Download best mp4 <= 720p via yt-dlp. Returns the downloaded file."""
    if not _have("yt-dlp"):
        raise IngestError(
            "yt-dlp is not installed; cannot download YouTube URLs. "
            "Install it with: pip install yt-dlp"
        )
    tag = uuid.uuid4().hex[:8]
    template = os.path.join(workdir, f"dl_{tag}.%(ext)s")
    # Prefer separate video+audio <=720p mp4, merged to mp4; fall back gracefully.
    fmt = (
        "bv*[height<=720][ext=mp4]+ba[ext=m4a]/"
        "b[height<=720][ext=mp4]/"
        "b[height<=720]/b"
    )
    cmd = [
        "yt-dlp", "-f", fmt, "--merge-output-format", "mp4",
        "--no-playlist", "-o", template, url,
    ]
    proc = _run(cmd, "YouTube download")
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise IngestError(f"yt-dlp failed for {url!r}: {err[-600:]}")
    found = glob.glob(os.path.join(workdir, f"dl_{tag}.*"))
    # yt-dlp may leave .part files on failure; only accept finished media.
    media = [f for f in found if not f.endswith((".part", ".ytdl", ".temp"))]
    if not media:
        raise IngestError(f"yt-dlp reported success but no file appeared for {url!r}")
    # Clean up strays, keep the first real file.
    for extra in found:
        if extra != media[0]:
            try:
                os.remove(extra)
            except OSError:
                pass
    return media[0]


def _normalize(src: str, dst: str) -> str:
    """Re-encode to the pipeline standard. Raises IngestError on failure."""
    if not _have("ffmpeg"):
        raise IngestError("ffmpeg is not installed or not on PATH.")
    # scale=-2:'min(720,ih)' keeps aspect ratio and only downscales.
    vf = "scale=-2:'min(720,ih)',fps=30,format=yuv420p"
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-ar", "44100", "-ac", "2",
        "-movflags", "+faststart",
        dst,
    ]
    proc = _run(cmd, "Normalization")
    if proc.returncode != 0 or not os.path.exists(dst):
        err = (proc.stderr or "").strip()
        raise IngestError(
            f"ffmpeg could not normalize {src!r} (unsupported or corrupt file?). "
            f"ffmpeg said: {err[-600:]}"
        )
    return dst


def ingest(source: str, workdir: str) -> str:
    """Ingest `source` (local path or YouTube URL) into `workdir`.

    Returns the path of the normalized MP4. Raises IngestError with a
    human-readable message for bad URLs, missing files, unsupported
    formats, or missing tools.
    """
    os.makedirs(workdir, exist_ok=True)

    if is_url(source):
        downloaded = _download_youtube(source, workdir)
        dst = os.path.join(workdir, f"normalized_{uuid.uuid4().hex[:8]}.mp4")
        try:
            return _normalize(downloaded, dst)
        finally:
            try:
                os.remove(downloaded)
            except OSError:
                pass

    # Local file path.
    if not os.path.isfile(source):
        raise IngestError(f"Local source not found: {source!r}")
    dst = os.path.join(workdir, f"normalized_{uuid.uuid4().hex[:8]}.mp4")
    return _normalize(source, dst)
