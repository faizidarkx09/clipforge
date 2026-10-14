"""Pipeline orchestration: ingest -> transcribe -> score, then render.

    job = run_pipeline(source, workdir, opts)
    final = render_clip(job, job["clips"][0], workdir, opts)

`run_pipeline` keeps the normalized video + transcript.json (needed for
rendering); `render_clip` cleans its own intermediates and returns only the
final MP4 path.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

from export import export_clip, make_title_hashtags
from ingest import ingest
from scorer import rank_clips
from transcribe import load_transcript, save_transcript, transcribe

ProgressFn = Optional[Callable[[str, float, str], None]]

DEFAULT_OPTS: dict = {
    "model": "base",        # faster-whisper model size
    "device": "cpu",        # "cpu" or "cuda"
    "top_n": 5,             # clips to rank
    "min_dur": 20.0,        # min clip length (s)
    "max_dur": 70.0,        # max clip length (s)
    "style": "hormozi",     # caption preset
    "sfx": True,            # whoosh SFX on/off
    "zoom": True,           # punch-zoom on/off
    "ar": "9:16",           # output aspect ratio ("9:16" | "1:1" | "16:9")
    "trim": False,          # AI trim: cut filler words + dead-air silences
    "dub": False,           # AI dub: replace audio with neural TTS (beta)
    "dub_lang": "urdu",     # dub target language (see clipforge.dub.VOICES)
    "logo": None,           # path to PNG watermark (brand kit), or None
    "focus_prompt": "",     # ClipAnything: keywords to boost in ranking
    "caption_overrides": None,  # per-render caption tweaks (see write_ass)
    "warnings": None,       # optional list collecting non-fatal warnings
    "transcript_path": None,  # skip transcription; load this JSON instead
    "progress": None,       # callable(stage, pct, msg)
}


def run_pipeline(source: str, workdir: str, opts: dict | None = None) -> dict:
    """Full analysis pass: ingest -> transcribe -> score -> titles.

    Returns:
        {
          "video": normalized video path,
          "transcript": transcript dict,
          "transcript_path": transcript.json path,
          "clips": [ {start,end,duration,score,reason,quote}, ... ],
          "title": str, "hashtags": [str, ...],
          "workdir": workdir,
        }
    """
    o = dict(DEFAULT_OPTS)
    o.update(opts or {})
    progress: ProgressFn = o.get("progress")

    def emit(stage: str, pct: float, msg: str):
        if progress:
            progress(stage, pct, msg)

    os.makedirs(workdir, exist_ok=True)

    emit("ingest", 5, "ingesting source")
    video = ingest(source, workdir)

    emit("transcribe", 15, "transcribing audio")
    if o.get("transcript_path"):
        transcript = load_transcript(o["transcript_path"])
        transcript_path = o["transcript_path"]
    else:
        transcript = transcribe(
            video, model=o.get("model", "base"), device=o.get("device", "cpu")
        )
        transcript_path = os.path.join(workdir, "transcript.json")
        save_transcript(transcript, transcript_path)
    emit("transcribe", 55, "transcription done")

    emit("score", 60, "scoring candidate clips")
    clips = rank_clips(
        transcript,
        audio_path=video,
        top_n=int(o.get("top_n", 5)),
        min_dur=float(o.get("min_dur", 20.0)),
        max_dur=float(o.get("max_dur", 70.0)),
        focus_prompt=o.get("focus_prompt", "") or "",
    )
    emit("score", 85, f"ranked {len(clips)} clips")

    full_text = " ".join(
        w.get("word", "")
        for seg in transcript.get("segments", [])
        for w in (seg.get("words") or [])
    )
    title, hashtags = make_title_hashtags(full_text)
    emit("done", 100, "pipeline complete")

    return {
        "video": video,
        "transcript": transcript,
        "transcript_path": transcript_path,
        "clips": clips,
        "title": title,
        "hashtags": hashtags,
        "workdir": workdir,
    }


def render_clip(
    job: dict,
    clip: dict,
    workdir: str,
    opts: dict | None = None,
    index: int = 0,
) -> str:
    """Render one ranked clip to a final MP4 (dimensions follow opts["ar"]).

    `job` is the dict returned by run_pipeline(); `clip` one of
    job["clips"]. Returns the final MP4 path inside `workdir`.

    Non-fatal render warnings (e.g. dub failures) are appended to
    opts["warnings"] when that list is provided.
    """
    o = dict(DEFAULT_OPTS)
    o.update(opts or {})
    os.makedirs(workdir, exist_ok=True)
    dst = os.path.join(workdir, f"clip_{index:02d}.mp4")
    dub = {"lang": o.get("dub_lang", "urdu")} if o.get("dub") else None
    return export_clip(
        job["video"],
        clip["start"],
        clip["end"],
        dst,
        transcript=job.get("transcript"),
        style=o.get("style", "hormozi"),
        sfx=bool(o.get("sfx", True)),
        zoom=bool(o.get("zoom", True)),
        ar=o.get("ar", "9:16"),
        caption_overrides=o.get("caption_overrides"),
        trim=bool(o.get("trim", False)),
        dub=dub,
        logo_path=o.get("logo"),
        warnings=o.get("warnings"),
        progress=o.get("progress"),
    )
