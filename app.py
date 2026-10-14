#!/usr/bin/env python3
"""ClipForge — Gradio UI (v2 premium).

Long video (upload or YouTube URL) -> AI-ranked viral vertical shorts.

Flow:
  1. Upload a video OR paste a YouTube URL, pick options, hit "Analyze".
  2. Review the transcript + ranked clip candidates (animated score cards
     + table), tick the ones you want.
  3. Customize captions/format, hit "Render selected" -> get MP4s with
     karaoke captions, punch zooms and SFX, plus an auto-generated title,
     description and hashtags per clip.

Run from the repo root:  python app.py
"""

import html
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

# Repo root on sys.path so the flat backend modules import cleanly when
# launched as `python app.py` from the repo root (also used for JOBS_DIR).
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import gradio as gr

# NOTE: top-level imports are intentionally limited to gradio + stdlib +
# these three backend modules (all stdlib/numpy-only at import time), so
# `import app` keeps working with ONLY gradio+numpy installed. Heavy
# backend entry points (run_pipeline/render_clip/IngestError) are imported
# lazily INSIDE the handlers below.
from captions import STYLES
from export import make_title_hashtags
try:
    # Added by the v2 backend upgrade; local stub keeps `import app`
    # working even if the backend lands after this UI.
    from export import make_description
except ImportError:  # pragma: no cover - defensive fallback
    def make_description(quote, title):
        q = (quote or "").strip()
        return f"{title}\n\n\"{q}\"\n\n#shorts #viral #clips" if q else title
from pipeline import DEFAULT_OPTS

# ---------------------------------------------------------------- constants

MAX_UPLOAD_MB = 500          # uploads larger than this are rejected (with a warning)
LONG_VIDEO_MIN = 15          # videos longer than this get a "may take a while" note
JOBS_DIR = os.path.join(ROOT, "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

# Caption styles per the v2 backend contract; fall back to whatever the
# installed backend actually ships so the UI never breaks on a partial upgrade.
CONTRACT_STYLES = ["hormozi", "minimal", "neon", "karaoke-box",
                   "pop", "typewriter", "elegant", "outline"]
STYLE_CHOICES = [s for s in CONTRACT_STYLES if s in STYLES] or list(STYLES.keys())
STYLE_DEFAULT = "hormozi" if "hormozi" in STYLE_CHOICES else STYLE_CHOICES[0]

DUB_LANGS = [("Urdu", "urdu"), ("Hindi", "hindi"), ("Spanish", "spanish"),
             ("French", "french"), ("Arabic", "arabic")]

DUR_PRESETS = {
    "Short (<30s)": (10, 30),
    "Medium (30–60s)": (30, 60),
    "Long (60–90s)": (60, 90),
}

POSITION_MAP = {"Bottom": "bottom", "Center": "center", "Top": "top"}


# ---------------------------------------------------------------- helpers

def fmt_ts(seconds: float) -> str:
    """Seconds -> m:ss (or h:mm:ss for long videos)."""
    s = max(0, int(round(seconds or 0)))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def probe_duration(path: str):
    """Video duration in seconds via ffprobe, or None if unavailable."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=30,
        )
        return float(out.stdout.strip())
    except Exception:
        return None


def clip_label(i: int, clip: dict) -> str:
    return (
        f"Clip {i + 1} — score {clip.get('score', 0):.0f} — "
        f"{clip.get('duration', 0):.0f}s "
        f"({fmt_ts(clip.get('start', 0))}–{fmt_ts(clip.get('end', 0))})"
    )


EMPTY_SCORE_CARDS = """
<div class="cf-empty">
  <div class="cf-empty-icon">🎬</div>
  <div><b>Upload a video to begin…</b></div>
  <div class="cf-empty-sub">No clips yet — run <b>Analyze</b> and your ranked
  candidates will appear here with viral-score cards.</div>
</div>
"""


def score_cards_html(clips) -> str:
    """Per-clip HTML score cards with animated conic-gradient score rings."""
    if not clips:
        return EMPTY_SCORE_CARDS
    cards = []
    for i, c in enumerate(clips):
        score = float(c.get("score", 0) or 0)
        ring = "#22c55e" if score >= 70 else ("#f59e0b" if score >= 40 else "#ef4444")
        dur = c.get("duration", 0) or 0
        time_range = f"{fmt_ts(c.get('start', 0))}–{fmt_ts(c.get('end', 0))}"
        reason = html.escape(str(c.get("reason") or "—")[:180])
        quote = html.escape(str(c.get("quote") or "—")[:180])
        delay = min(i * 0.08, 0.6)
        cards.append(
            f'<div class="cf-card" style="animation-delay:{delay:.2f}s">'
            f'<div class="cf-ring" style="--score:{score:.0f}; --ring:{ring}">'
            f"<span>{score:.0f}</span></div>"
            f'<div class="cf-card-body">'
            f'<div class="cf-card-title">#{i + 1} · {dur:.1f}s · {time_range}</div>'
            f'<div class="cf-card-reason">{reason}</div>'
            f'<div class="cf-card-quote">&ldquo;{quote}&rdquo;</div>'
            f"</div></div>"
        )
    return '<div class="cf-cards">' + "".join(cards) + "</div>"


def _err(msg: str, job):
    """Error-shaped return tuple for the analyze handler."""
    return (f"❌ {msg}", "", EMPTY_SCORE_CARDS, [],
            gr.update(choices=[], value=[]), job)


# ---------------------------------------------------------------- handlers

def analyze_video(video_file, yt_url, top_n, min_dur, max_dur, model,
                  focus_prompt, trim, dub, dub_lang,
                  progress=gr.Progress(track_tqdm=True)):
    """Ingest -> transcribe -> score. Returns status, transcript, score
    cards, table, clip checkboxes, and the job state dict for rendering."""
    # Lazy backend imports: keeps `import app` working with gradio+numpy only.
    from ingest import IngestError
    from pipeline import run_pipeline

    job = None
    yt_url = (yt_url or "").strip()

    # ---- input validation -------------------------------------------------
    if video_file and yt_url:
        return _err("Please use either a file upload OR a YouTube URL, not both.", job)
    if not video_file and not yt_url:
        return _err("Upload a video file or paste a YouTube URL to analyze.", job)
    if yt_url and not yt_url.lower().startswith(("http://", "https://")):
        return _err("That doesn't look like a valid URL — it should start with http(s)://.", job)

    if video_file:
        try:
            size_mb = os.path.getsize(video_file) / (1024 * 1024)
        except OSError:
            return _err("Could not read the uploaded file. Please try uploading again.", job)
        if size_mb > MAX_UPLOAD_MB:
            return _err(
                f"⚠️ Upload is {size_mb:.0f} MB — the cap is {MAX_UPLOAD_MB} MB. "
                "Please compress the video or paste a YouTube URL instead.", job)

    if min_dur >= max_dur:
        max_dur = min_dur + 10  # keep the slider combo sane instead of failing

    source = yt_url or video_file
    workdir = tempfile.mkdtemp(prefix="job_", dir=JOBS_DIR)

    def cb(stage, pct, msg):
        progress(min(max(pct / 100.0, 0.0), 1.0), desc=f"{stage}: {msg}")

    opts = dict(DEFAULT_OPTS)
    opts.update({
        "model": model,
        "device": "cpu",
        "top_n": int(top_n),
        "min_dur": float(min_dur),
        "max_dur": float(max_dur),
        "focus_prompt": (focus_prompt or "").strip(),   # ClipAnything keyword boost
        "trim": bool(trim),                             # AI trim: fillers + silences
        "dub": bool(dub),                               # AI dub (beta)
        "dub_lang": dub_lang or "urdu",
        "progress": cb,
    })

    try:
        job = run_pipeline(source, workdir, opts)
    except IngestError as exc:
        shutil.rmtree(workdir, ignore_errors=True)
        return _err(f"Could not load that source. {exc}", None)
    except Exception:  # noqa: BLE001 - surface anything unexpected cleanly
        shutil.rmtree(workdir, ignore_errors=True)
        traceback.print_exc()
        return _err("Analysis failed with an unexpected error. See the server log for details.", None)

    # ---- transcript -------------------------------------------------------
    segments = (job.get("transcript") or {}).get("segments", [])
    if not segments:
        shutil.rmtree(workdir, ignore_errors=True)
        return _err(
            "The video produced an empty transcript — it may have no speech, "
            "or the audio track may be missing/corrupt. Try a different source.", None)

    transcript_lines = [f"[{fmt_ts(s.get('start', 0))}] {(s.get('text') or '').strip()}"
                        for s in segments]
    transcript_text = "\n".join(transcript_lines)

    clips = job.get("clips") or []
    if not clips:
        status = ("⚠️ Transcribed fine, but no clips passed the duration filters. "
                  "Try lowering the minimum duration and analyzing again.")
        return (status, transcript_text, EMPTY_SCORE_CARDS, [],
                gr.update(choices=[], value=[]), job)

    rows = [
        [i + 1,
         f"{c.get('score', 0):.0f}",
         f"{c.get('duration', 0):.1f}s",
         f"{fmt_ts(c.get('start', 0))} – {fmt_ts(c.get('end', 0))}",
         (c.get("reason") or "")[:120],
         (c.get("quote") or "")[:140]]
        for i, c in enumerate(clips)
    ]
    labels = [clip_label(i, c) for i, c in enumerate(clips)]
    job["labels"] = labels  # remember for mapping checkbox picks -> clip index

    note = ""
    dur = probe_duration(job["video"])
    if dur and dur > LONG_VIDEO_MIN * 60:
        note = (f"\n\nℹ️ This video is {fmt_ts(dur)} long — transcription and "
                "rendering run on CPU, so this may take a while. Feel free to leave the tab open.")

    title = job.get("title", "")
    tags = " ".join(job.get("hashtags", []))
    status = (f"✅ Analyzed {len(segments)} transcript segments, ranked {len(clips)} clips. "
              f"Suggested title: **{title}**  {tags}{note}")
    return (status, transcript_text, score_cards_html(clips), rows,
            gr.update(choices=labels, value=labels[:1]), job)


def render_selected(selected, job, style, font_size, position, text_color,
                    highlight_color, bg_box, emoji, aspect, sfx, zoom, logo,
                    dub, dub_lang, progress=gr.Progress(track_tqdm=True)):
    """Render each selected clip sequentially. Returns status, gallery items,
    download files, and a markdown block with per-clip title + description
    + hashtags."""
    # Lazy backend import: keeps `import app` working with gradio+numpy only.
    from pipeline import render_clip

    if not job:
        return ("❌ Nothing to render — analyze a video first.", [], [], "")
    if not selected:
        return ("⚠️ Tick at least one clip in the list above, then hit Render.", [], [], "")

    labels = job.get("labels") or []
    indices = sorted({labels.index(s) for s in selected if s in labels})
    if not indices:
        return ("⚠️ Couldn't match your selection to the ranked clips — please re-select.", [], [], "")

    clips = job.get("clips") or []
    workdir = job.get("workdir")
    outdir = os.path.join(workdir, "final")
    os.makedirs(outdir, exist_ok=True)

    caption_overrides = {
        "font_size": int(font_size or 76),
        "position": POSITION_MAP.get(position, "bottom"),
        "text_color": text_color or "#FFFFFF",
        "highlight_color": highlight_color or "#FFFF00",
        "bg_box": bool(bg_box),
        "emoji": bool(emoji),
    }

    def cb(stage, pct, msg):
        # pct is per-clip; blend with overall clip index for a global bar.
        overall = (done + min(max(pct, 0.0), 100.0) / 100.0) / total
        progress(overall, desc=f"Clip {done + 1}/{total} [{stage}]: {msg}")

    done = 0
    total = len(indices)
    made = []          # [(path, caption)]
    md_parts = []
    failures = []
    warnings = []      # backend appends here (e.g. AI dub beta fell back to original audio)

    for n, i in enumerate(indices):
        clip = clips[i]
        quote = clip.get("quote", "") or ""
        per_title, per_tags = make_title_hashtags(f"{quote} {clip.get('reason', '')}")
        try:
            description = make_description(quote, per_title)
        except Exception:  # noqa: BLE001 - never let copy generation kill a render
            description = ""
        opts = dict(DEFAULT_OPTS)
        opts.update({
            "style": style,
            "sfx": bool(sfx),
            "zoom": bool(zoom),
            "ar": aspect or "9:16",
            "logo": logo,                      # filepath or None (brand watermark)
            "dub": bool(dub),                  # AI dub (beta)
            "dub_lang": dub_lang or "urdu",
            "caption_overrides": caption_overrides,
            "warnings": warnings,
            "progress": cb,
        })
        try:
            dst = render_clip(job, clip, outdir, opts, index=n)
            made.append((dst, f"{per_title}"))
            md_parts.append(
                f"### 🎬 Clip {i + 1}: {per_title}\n\n"
                f"{description}\n\n"
                f"{' '.join(per_tags)}\n\n"
                f"`{os.path.basename(dst)}` — {clip.get('duration', 0):.1f}s "
                f"({fmt_ts(clip.get('start', 0))}–{fmt_ts(clip.get('end', 0))})")
        except Exception as exc:  # noqa: BLE001 - keep rendering the rest
            traceback.print_exc()
            failures.append(f"Clip {i + 1}: {exc}")
        done = n + 1

    # ---- cleanup: keep only the final MP4s --------------------------------
    try:
        for entry in os.listdir(workdir):
            if entry != "final":
                p = os.path.join(workdir, entry)
                shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) \
                    else os.remove(p) if os.path.exists(p) else None
    except OSError:
        pass

    if not made:
        return (f"❌ All {total} renders failed: {'; '.join(failures)}", [], [], "")

    status = f"✅ Rendered {len(made)}/{total} clip(s) → `final/`."
    if failures:
        status += f" Failed: {'; '.join(failures)}"
    if warnings:
        status += "\n⚠️ " + " | ".join(str(w) for w in warnings)
    status += " Temp files cleaned up; final MP4s kept."

    return (status,
            made,
            gr.update(value=[p for p, _ in made]),
            "\n\n---\n\n".join(md_parts))


# ---------------------------------------------------------------- UI

CSS = """
/* ============ ClipForge premium dark theme ============ */
/* NOTE: Gradio 4.44's theme variables don't cascade into components here,
   so components are styled directly (all rules verified in-browser). */
.gradio-container {
  background: linear-gradient(135deg, #07070f 0%, #101020 60%, #0b0b18 100%) !important;
  color: #e9e9f5;
  max-width: 100% !important;
}
/* ---- row/column form wrappers (Gradio paints these light gray) ---- */
.gradio-container .form,
.gradio-container .styler {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
/* ---- dark component blocks ---- */
.gradio-container .block {
  background: rgba(255,255,255,.045) !important;
  border: 1px solid rgba(255,255,255,.10) !important;
  border-radius: 16px !important;
  box-shadow: none !important;
}
.gradio-container [data-testid="block-label"] {
  background: rgba(255,255,255,.05) !important;
  color: #cfcfe8 !important;
  border: none !important;
}
.gradio-container .block label {
  background: rgba(255,255,255,.05) !important;
  border: 1px solid rgba(255,255,255,.12) !important;
  color: #d6d6e8 !important;
}
/* ---- dark inputs ---- */
.gradio-container textarea,
.gradio-container input:not([type="range"]):not([type="checkbox"]):not([type="radio"]):not([type="file"]):not([type="color"]),
.gradio-container select {
  background: rgba(255,255,255,.06) !important;
  color: #f0f0fa !important;
  border: 1px solid rgba(255,255,255,.14) !important;
  border-radius: 12px !important;
}
.gradio-container textarea::placeholder,
.gradio-container input::placeholder { color: #6d6d8f !important; }
/* ---- dropdown + video dropzone internals ---- */
.gradio-container .block .wrap,
.gradio-container .block .wrap-inner {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
.gradio-container .upload-container {
  background: rgba(255,255,255,.03) !important;
  border: 1px dashed rgba(255,255,255,.18) !important;
  border-radius: 14px !important;
}
/* ---- dataframe ---- */
.gradio-container table { color: #e9e9f5 !important; }
.gradio-container th {
  background: rgba(124,58,237,.18) !important;
  color: #f2f2fa !important;
  border-color: rgba(255,255,255,.10) !important;
}
.gradio-container td { border-color: rgba(255,255,255,.08) !important; }
.gradio-container tr:nth-child(even) td { background: rgba(255,255,255,.03) !important; }
.gradio-container > * { animation: fadeUp .55s ease both; }
.gradio-container > *:nth-child(2) { animation-delay: .07s; }
.gradio-container > *:nth-child(3) { animation-delay: .14s; }
.gradio-container > *:nth-child(4) { animation-delay: .21s; }
.gradio-container > *:nth-child(5) { animation-delay: .28s; }
.gradio-container > *:nth-child(6) { animation-delay: .35s; }
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}
.gradio-container h1, .gradio-container h2, .gradio-container h3 { color: #f2f2fa !important; }
.gradio-container label, .gradio-container legend { color: #cfcfe8 !important; }

/* ---- animated gradient hero ---- */
.cf-hero { text-align: center; padding: 2rem 1rem 1rem; }
.cf-hero-title {
  font-size: clamp(2.2rem, 6vw, 3.6rem);
  font-weight: 900; letter-spacing: -0.02em; margin: 0;
  background: linear-gradient(120deg, #a78bfa, #ec4899, #fbbf24, #a78bfa);
  background-size: 300% 300%;
  -webkit-background-clip: text; background-clip: text; color: transparent;
  animation: heroShift 9s ease infinite;
}
@keyframes heroShift {
  0%, 100% { background-position: 0% 50%; }
  50%      { background-position: 100% 50%; }
}
.cf-hero-tag { color: #b9b9d6; margin-top: .45rem; font-size: 1.02rem; }

/* ---- 3-step indicator ---- */
.cf-steps { display: flex; align-items: center; justify-content: center;
  gap: .6rem; flex-wrap: wrap; margin: .4rem 0 1.2rem; }
.cf-step { display: flex; align-items: center; gap: .5rem; padding: .5rem 1.1rem;
  border-radius: 999px; background: rgba(255,255,255,.04);
  border: 1px solid rgba(255,255,255,.12); color: #9a9ab8; font-weight: 600; }
.cf-step.active { background: linear-gradient(90deg, #7c3aed, #ec4899); color: #fff;
  border-color: transparent; box-shadow: 0 0 18px rgba(124,58,237,.5); }
.cf-step-num { display: inline-flex; width: 1.6rem; height: 1.6rem; border-radius: 50%;
  align-items: center; justify-content: center; background: rgba(255,255,255,.12); font-size: .85rem; }
.cf-step.active .cf-step-num { background: rgba(255,255,255,.25); }
.cf-step-line { width: 2.5rem; height: 2px; border-radius: 2px;
  background: linear-gradient(90deg, rgba(124,58,237,.6), rgba(236,72,153,.6)); }

/* ---- glassmorphism ---- */
.glass, .cf-card {
  background: rgba(255,255,255,.04) !important;
  border: 1px solid rgba(255,255,255,.10) !important;
  border-radius: 18px !important;
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
}
.glass { padding: 1rem 1.2rem; margin-bottom: .9rem;
  transition: transform .18s ease, box-shadow .18s ease; }
.glass:hover { transform: translateY(-2px); box-shadow: 0 12px 32px rgba(0,0,0,.35); }

/* ---- score cards ---- */
.cf-cards { display: flex; flex-direction: column; gap: .7rem; margin-bottom: .8rem; }
.cf-card { display: flex; gap: 1rem; align-items: center; padding: .9rem 1.1rem;
  animation: fadeUp .5s ease both; transition: transform .18s ease, box-shadow .18s ease; }
.cf-card:hover { transform: translateY(-2px); box-shadow: 0 10px 28px rgba(0,0,0,.4); }
.cf-ring {
  --score: 0; --ring: #22c55e;
  width: 76px; height: 76px; min-width: 76px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center; position: relative;
  background: conic-gradient(var(--ring) calc(var(--score) * 1%), rgba(255,255,255,.08) 0);
  animation: ringIn .55s cubic-bezier(.34,1.56,.64,1) both;
}
.cf-ring::before { content: ''; position: absolute; inset: 9px; border-radius: 50%; background: #12121f; }
.cf-ring span { position: relative; font-weight: 800; font-size: 1.35rem; color: #fff; }
@keyframes ringIn { from { transform: scale(.5); opacity: 0; } to { transform: scale(1); opacity: 1; } }
.cf-card-body { min-width: 0; }
.cf-card-title { font-weight: 700; color: #fff; }
.cf-card-reason { color: #c9c9e2; font-size: .92rem; margin-top: .15rem; }
.cf-card-quote { color: #8f8fb0; font-style: italic; font-size: .9rem; margin-top: .15rem; }

/* ---- empty state ---- */
.cf-empty { text-align: center; padding: 2.4rem 1rem; color: #8f8fb0;
  background: rgba(255,255,255,.03); border: 1px dashed rgba(255,255,255,.15); border-radius: 18px; }
.cf-empty-icon { font-size: 2.4rem; margin-bottom: .5rem; }
.cf-empty-sub { margin-top: .3rem; font-size: .9rem; }

/* ---- custom sliders ---- */
input[type="range"] { -webkit-appearance: none; appearance: none; height: 8px;
  border-radius: 99px; background: linear-gradient(90deg, #7c3aed, #ec4899) !important; outline: none; }
input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; width: 20px; height: 20px;
  border-radius: 50%; border: none; cursor: pointer;
  background: radial-gradient(circle at 30% 30%, #ffffff, #c4b5fd 55%, #7c3aed);
  box-shadow: 0 0 14px rgba(167,139,250,.9); }
input[type="range"]::-moz-range-thumb { width: 20px; height: 20px; border-radius: 50%;
  border: none; cursor: pointer;
  background: radial-gradient(circle at 30% 30%, #ffffff, #c4b5fd 55%, #7c3aed);
  box-shadow: 0 0 14px rgba(167,139,250,.9); }

/* ---- primary buttons: pulsing glow + hover lift ---- */
/* Gradio 4.44 renders variant as a bare `primary` class (class="{size} {variant}"). */
.gradio-container button { border-radius: 14px !important; transition: transform .18s ease, box-shadow .18s ease; }
.gradio-container button.primary {
  background: linear-gradient(90deg, #7c3aed, #ec4899) !important;
  color: #ffffff !important;
  animation: pulseGlow 2.6s ease-in-out infinite; border: none !important; }
.gradio-container button.primary:hover { transform: translateY(-2px); }
@keyframes pulseGlow {
  0%, 100% { box-shadow: 0 0 0 0 rgba(124,58,237,.55); }
  50%      { box-shadow: 0 0 24px 6px rgba(124,58,237,.55); }
}

/* ---- pill tabs ---- */
.tab-nav { border-bottom: none !important; gap: .5rem; display: flex; flex-wrap: wrap; }
.tab-nav button { border-radius: 999px !important; border: 1px solid rgba(255,255,255,.12) !important;
  background: rgba(255,255,255,.04) !important; color: #b9b9d6 !important;
  padding: .55rem 1.2rem !important; font-weight: 600; }
.tab-nav button.selected, .tab-nav button[aria-selected="true"] {
  background: linear-gradient(90deg, #7c3aed, #ec4899) !important; color: #fff !important;
  border-color: transparent !important; box-shadow: 0 4px 18px rgba(124,58,237,.45); }

/* (dark inputs are handled by the direct rules above) */
.transcript-box textarea { font-family: ui-monospace, monospace; font-size: 0.85rem; }
"""

HERO_HTML = """
<div class="cf-hero">
  <p class="cf-hero-title">✂️ ClipForge</p>
  <div class="cf-hero-tag">Long video in → viral shorts out. AI transcription, virality scoring,
  face-aware reframe, karaoke captions.</div>
</div>
"""

STEPS_HTML = """
<div class="cf-steps">
  <div class="cf-step active"><span class="cf-step-num">1</span> Upload</div>
  <div class="cf-step-line"></div>
  <div class="cf-step"><span class="cf-step-num">2</span> Analyze &amp; Rank</div>
  <div class="cf-step-line"></div>
  <div class="cf-step"><span class="cf-step-num">3</span> Customize &amp; Render</div>
</div>
"""

with gr.Blocks(title="ClipForge — Long Video → Viral Shorts", css=CSS) as demo:
    gr.HTML(HERO_HTML)
    gr.HTML(STEPS_HTML)

    job_state = gr.State(None)

    with gr.Tab("📤 1 · Upload & Analyze"):
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group(elem_classes=["glass"]):
                    gr.Markdown("### 📥 Source")
                    video_in = gr.Video(
                        label=f"Upload video (MP4/MOV/WebM…, max ~{MAX_UPLOAD_MB} MB)",
                        sources=["upload"])
                    yt_in = gr.Textbox(label="…or paste a YouTube URL",
                                       placeholder="https://www.youtube.com/watch?v=…")
            with gr.Column(scale=1):
                with gr.Group(elem_classes=["glass"]):
                    gr.Markdown("### ⚙️ Analysis options")
                    top_n = gr.Slider(1, 10, value=5, step=1, label="Number of clips to rank")
                    gr.Markdown("### ⏱️ Clip length")
                    dur_preset = gr.Radio(
                        list(DUR_PRESETS.keys()), value=None,
                        label="Duration preset (sets the sliders below)")
                    with gr.Row():
                        min_dur = gr.Slider(5, 60, value=20, step=1, label="Min clip duration (s)")
                        max_dur = gr.Slider(10, 120, value=70, step=1, label="Max clip duration (s)")
                    model = gr.Dropdown(["tiny", "base", "small"],
                                        value=os.environ.get("CLIPFORGE_MODEL", "base"),
                                        label="Whisper model (tiny = fastest, small = most accurate)")
                    focus_prompt = gr.Textbox(
                        label="🎯 Focus prompt (ClipAnything)",
                        placeholder="e.g. money mindset, crypto tips — boosts matching moments")
                    with gr.Row():
                        trim_chk = gr.Checkbox(
                            value=False, label="✂️ Remove filler words & silences (AI Trim)")
                        dub_chk = gr.Checkbox(
                            value=False, label="🎙️ AI Dub (beta)")
                    dub_lang = gr.Dropdown(
                        choices=DUB_LANGS, value="urdu", label="Dub language (beta)")
        analyze_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
        status_out = gr.Textbox(label="Status", lines=3, interactive=False)

    with gr.Tab("🏆 2 · Ranked Clips"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Transcript")
                transcript_out = gr.Textbox(label="Timed segments", lines=22,
                                            interactive=False, elem_classes=["transcript-box"],
                                            max_lines=22)
            with gr.Column(scale=2):
                gr.Markdown("### 🏆 Ranked clips")
                score_cards = gr.HTML(EMPTY_SCORE_CARDS)
                clips_table = gr.Dataframe(
                    headers=["#", "Score", "Duration", "Time range", "Why it scores", "Key quote"],
                    datatype=["number", "str", "str", "str", "str", "str"],
                    interactive=False, wrap=True, label="Candidates")
                pick = gr.CheckboxGroup(label="✅ Select clips to render", choices=[])

    with gr.Tab("🎨 3 · Customize & Render"):
        with gr.Group(elem_classes=["glass"]):
            gr.Markdown("### ✍️ Captions")
            style = gr.Radio(STYLE_CHOICES, value=STYLE_DEFAULT, label="Caption style")
            with gr.Row():
                font_size = gr.Slider(40, 130, step=2, value=76, label="🔠 Caption size")
                position = gr.Radio(["Bottom", "Center", "Top"], value="Bottom",
                                    label="Caption position")
            with gr.Row():
                text_color = gr.ColorPicker(value="#FFFFFF", label="🔤 Text color")
                highlight = gr.ColorPicker(value="#FFFF00",
                                           label="✨ Highlight color (sung words)")
                accent = gr.ColorPicker(value="#FFD700",
                                        label="🎨 Accent color (auto-sets highlight)")
            with gr.Row():
                bg_box = gr.Checkbox(value=False, label="📦 Caption background box")
                emoji_chk = gr.Checkbox(value=False, label="😀 Emoji captions")
        with gr.Group(elem_classes=["glass"]):
            gr.Markdown("### 🖼️ Format & polish")
            with gr.Row():
                aspect = gr.Radio(["9:16", "1:1", "16:9"], value="9:16", label="Aspect ratio")
                sfx = gr.Checkbox(value=True, label="Whoosh SFX")
                zoom = gr.Checkbox(value=True, label="Punch zooms")
            logo = gr.Image(label="🎨 Brand logo (optional watermark)", type="filepath")
        render_btn = gr.Button("🎬 Render selected", variant="primary", size="lg")
        render_status = gr.Textbox(label="Render status", lines=3, interactive=False)
        gallery = gr.Gallery(label="Finished clips (click to preview)", columns=3,
                             object_fit="contain", height="auto")
        downloads = gr.File(label="Download MP4s", file_count="multiple", interactive=False)
        titles_md = gr.Markdown(
            "🎬 *Your rendered clips' titles, descriptions & hashtags will appear here.*")

    # ---- reactive wiring --------------------------------------------------
    def _apply_preset(choice):
        if choice in DUR_PRESETS:
            lo, hi = DUR_PRESETS[choice]
            return gr.update(value=lo), gr.update(value=hi)
        return gr.update(), gr.update()

    dur_preset.change(_apply_preset, inputs=dur_preset, outputs=[min_dur, max_dur])
    # Accent picker directly drives the highlight color picker.
    accent.change(lambda c: c, inputs=accent, outputs=highlight)

    analyze_btn.click(
        analyze_video,
        inputs=[video_in, yt_in, top_n, min_dur, max_dur, model,
                focus_prompt, trim_chk, dub_chk, dub_lang],
        outputs=[status_out, transcript_out, score_cards, clips_table, pick, job_state],
    )
    render_btn.click(
        render_selected,
        inputs=[pick, job_state, style, font_size, position, text_color,
                highlight, bg_box, emoji_chk, aspect, sfx, zoom, logo,
                dub_chk, dub_lang],
        outputs=[render_status, gallery, downloads, titles_md],
    )

if __name__ == "__main__":
    demo.queue()  # progress bars + sequential CPU-heavy jobs
    # 0.0.0.0 + $PORT: required by Docker hosts (HF Spaces, Render, ...).
    # Render injects PORT (default 10000); elsewhere it falls back to 7860.
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
