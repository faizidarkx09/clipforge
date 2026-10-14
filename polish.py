"""Polish: punch-in zooms, whoosh SFX, loudness normalization.

All synthesis is local (no downloads): whooshes are generated from filtered
noise with ffmpeg. Filter graphs are kept deliberately simple and robust.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

PUNCH_AMOUNT = 0.08   # 1.00 -> 1.08 zoom
PUNCH_RADIUS = 0.4    # seconds over which the punch ramps in/out


def punch_zoom_filter(
    sentences: list[dict],
    amount: float = PUNCH_AMOUNT,
    radius: float = PUNCH_RADIUS,
    out_w: int = 1080,
    out_h: int = 1920,
) -> tuple[str, bool]:
    """Build a punch-zoom video filter.

    At each sentence start a triangular pulse ramps the zoom from 1.0x to
    (1+amount)x over `radius` seconds, then back. Implemented as
    ``scale`` with ``eval=frame`` (per-frame w/h evaluation driven by `t`)
    to upscale, followed by a fixed center crop back to out_w x out_h.

    Note: crop's own w/h are only evaluated at filter init in ffmpeg, so a
    time-varying crop is not possible there — hence scale-up + static crop.

    Expects a out_w x out_h input (the reframed clip). Returns
    (filter_string, applied); `applied` is False when there are no usable
    sentence boundaries (caller should use a plain scale instead).
    """
    starts = sorted(
        {
            round(float(s["start"]), 2)
            for s in (sentences or [])
            if s.get("start") is not None and float(s["start"]) > 0.2
        }
    )[:12]  # cap expression length for ffmpeg parser sanity
    if not starts:
        return "", False
    pulse = "+".join(f"max(0,1-abs(t-{s})/{radius})" for s in starts)
    z = f"(1+{amount}*min(1,({pulse})))"
    # Commas inside quoted expressions are safe for the filter parser.
    vf = (
        f"scale=w='iw*{z}':h='ih*{z}':eval=frame,"
        f"crop={out_w}:{out_h}:x='(iw-{out_w})/2':y='(ih-{out_h})/2',"
        "setsar=1,format=yuv420p"
    )
    return vf, True


def whoosh_sfx(duration: float, path: str, volume: float = 0.5) -> str:
    """Synthesize a short whoosh (filtered noise swell) to a WAV file.

    No downloads: white noise -> highpass/lowpass -> attack/decay envelope.
    `duration` is clamped to 0.3..1.2s. Returns `path`.
    """
    dur = min(1.2, max(0.3, float(duration)))
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is not installed or not on PATH.")
    af = (
        "anoisesrc=d={d}:c=white:r=44100,"
        "highpass=f=400,lowpass=f=7000,"
        "volume='min(1,t/0.12)*min(1,({d}-t)/0.18)':eval=frame,"
        "aformat=sample_fmts=fltp:channel_layouts=stereo"
    ).format(d=f"{dur:.2f}")
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi", "-i", af,
        "-t", f"{dur:.2f}", "-c:a", "pcm_s16le", path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(path):
        err = (proc.stderr or "").strip()
        raise RuntimeError(f"whoosh_sfx failed: {err[-400:]}")
    return path


def _whoosh_bed(duration: float, starts: list[float], tmpdir: str) -> str | None:
    """Build one stereo WAV: whooshes delayed to each sentence start.

    Returns the bed path, or None when there is nothing to place.
    """
    starts = [s for s in starts if 0.1 < s < duration - 0.2][:10]
    if not starts:
        return None
    inputs: list[str] = []
    chains: list[str] = []
    for i, s in enumerate(starts):
        seg = os.path.join(tmpdir, f"whoosh_{i}.wav")
        whoosh_sfx(0.7, seg)
        inputs += ["-i", seg]
        ms = int(s * 1000)
        chains.append(f"[{i + 1}:a]adelay={ms}|{ms},volume=0.35[a{i}]")
    mix = "".join(f"[a{i}]" for i in range(len(starts)))
    chains.append(
        f"{mix}amix=inputs={len(starts)}:duration=longest:"
        "dropout_transition=0:normalize=0,"
        f"apad=whole_dur={duration:.2f},aresample=44100,"
        "aformat=channel_layouts=stereo[bed]"
    )
    bed = os.path.join(tmpdir, "whoosh_bed.wav")
    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(chains),
           "-map", "[bed]", "-c:a", "pcm_s16le", bed]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(bed):
        return None  # SFX is decorative; never fail the render for it
    return bed


def apply_polish(
    src: str,
    dst: str,
    sentences: list[dict] | None = None,
    sfx: bool = True,
    zoom: bool = True,
    crf: int = 20,
    out_w: int = 1080,
    out_h: int = 1920,
) -> str:
    """Apply finishing touches to a reframed clip.

    - punch-in zooms at sentence boundaries (toggle `zoom`)
    - whoosh SFX mixed quietly under dialogue at cut points (toggle `sfx`)
    - loudnorm-ish single-pass loudness to broadcast-ish levels
    - final encode: H.264 yuv420p + AAC 44.1kHz, faststart

    `out_w`/`out_h` are the frame size the zoom/crop filters target
    (default 1080x1920; follows the render aspect ratio).

    Returns `dst`. Raises RuntimeError on ffmpeg failure.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is not installed or not on PATH.")
    if not os.path.isfile(src):
        raise FileNotFoundError(f"Clip not found: {src!r}")

    sentences = sentences or []
    tmpdir = tempfile.mkdtemp(prefix="clipforge_polish_")
    try:
        # --- video chain -------------------------------------------------
        vf, applied = ("", False)
        if zoom:
            vf, applied = punch_zoom_filter(sentences, out_w=out_w, out_h=out_h)
        if not applied:
            vf = f"scale={out_w}:{out_h}:flags=lanczos,setsar=1,format=yuv420p"

        # --- audio chain -------------------------------------------------
        cmd = ["ffmpeg", "-y", "-i", src]
        filter_parts = [f"[0:v]{vf}[v]"]
        if sfx:
            starts = [float(s["start"]) for s in sentences
                      if s.get("start") is not None]
            # Probe duration cheaply via the sentence list, fallback 70s.
            dur = max([float(s["end"]) for s in sentences
                       if s.get("end") is not None] + [70.0])
            bed = _whoosh_bed(dur, starts, tmpdir)
        else:
            bed = None

        if bed:
            cmd += ["-i", bed]
            filter_parts.append(
                "[0:a][1:a]amix=inputs=2:duration=first:"
                "dropout_transition=0:normalize=0,"
                "loudnorm=I=-16:TP=-1.5:LRA=11[a]"
            )
        else:
            filter_parts.append("[0:a]loudnorm=I=-16:TP=-1.5:LRA=11[a]")

        cmd += [
            "-filter_complex", ";".join(filter_parts),
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-r", "30",
            "-c:a", "aac", "-ar", "44100", "-ac", "2",
            "-movflags", "+faststart",
            dst,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not os.path.exists(dst):
            err = (proc.stderr or "").strip()
            raise RuntimeError(f"apply_polish failed: {err[-600:]}")
        return dst
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
