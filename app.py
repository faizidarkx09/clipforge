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


def make_thumb(video_path: str, ts: float, out_path: str, width: int = 360) -> bool:
    """Extract a JPEG thumbnail at `ts` seconds. Returns True on success."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", f"{max(0, ts):.2f}",
             "-i", video_path, "-vframes", "1",
             "-vf", f"scale={width}:-1", "-q:v", "4", out_path],
            capture_output=True, timeout=30)
        return r.returncode == 0 and os.path.exists(out_path)
    except Exception:
        return False


def _thumb_b64(path: str) -> str:
    """File -> base64 data URI (or empty string)."""
    try:
        import base64
        with open(path, "rb") as f:
            return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
    except Exception:
        return ""


EMPTY_SCORE_CARDS = """
<div class="oc-empty">
  <div class="oc-empty-icon">🎬</div>
  <div><b>No clips yet</b></div>
  <div class="oc-empty-sub">Upload a video and hit <b>Generate Clips</b> — your
  AI-ranked candidates will appear here with virality scores.</div>
</div>
"""

SIDEBAR_HTML = """
<div class="oc-sidebar">
  <div class="oc-logo">
    <div class="oc-logo-mark">✂️</div>
    <div><div class="oc-logo-name">ClipForge</div><div class="oc-logo-sub">AI Clipping Studio</div></div>
  </div>
  <div class="oc-nav-label">Studio</div>
  <a class="oc-nav-item active" href="#"><span class="oc-ico">🎬</span> New Clips</a>
  <a class="oc-nav-item" href="#"><span class="oc-ico">⚡</span> How it works</a>
  <div class="oc-nav-label">Account</div>
  <a class="oc-nav-item" href="#"><span class="oc-ico">⚙️</span> Settings</a>
  <a class="oc-nav-item" href="#"><span class="oc-ico">❓</span> Help</a>
  <div class="oc-side-foot">ClipForge v3 · free &amp; open<br>made for creators 🎥</div>
</div>
"""

HEADER_HTML = """
<div class="oc-header">
  <div class="oc-kicker"><span class="dot"></span> AI clipping engine · online</div>
  <h1 class="oc-title">Turn long videos into <span class="grad">viral clips</span></h1>
  <p class="oc-sub">Upload a video or paste a link — AI transcribes, scores every moment for
  virality, reframes for vertical, and burns in animated captions. Pick your winners, hit render.</p>
</div>
"""

HOW_HTML = """
<div class="oc-panel">
  <p class="oc-panel-title"><span class="oc-ico">⚡</span> How it works</p>
  <div style="display:flex;gap:.7rem;flex-wrap:wrap">
    <div style="flex:1;min-width:150px;background:rgba(255,255,255,.03);border-radius:12px;padding:.8rem">
      <b>1 · Upload</b><div style="color:var(--oc-muted);font-size:.85rem;margin-top:.25rem">Drop a video file or paste a YouTube URL. Up to ~500 MB.</div></div>
    <div style="flex:1;min-width:150px;background:rgba(255,255,255,.03);border-radius:12px;padding:.8rem">
      <b>2 · AI ranks</b><div style="color:var(--oc-muted);font-size:.85rem;margin-top:.25rem">Whisper transcribes, the scorer ranks the most viral moments 0–100.</div></div>
    <div style="flex:1;min-width:150px;background:rgba(255,255,255,.03);border-radius:12px;padding:.8rem">
      <b>3 · Render</b><div style="color:var(--oc-muted);font-size:.85rem;margin-top:.25rem">Style the captions, pick 9:16 / 1:1 / 16:9, export pro MP4s.</div></div>
  </div>
</div>
"""


def score_cards_html(clips) -> str:
    """Opus-style clip cards: thumbnail, virality score ring, duration badge,
    reason + key quote. Thumbnails are base64 data URIs in clip['_thumb_b64']."""
    if not clips:
        return EMPTY_SCORE_CARDS
    cards = []
    for i, c in enumerate(clips):
        score = float(c.get("score", 0) or 0)
        ring = "#22c55e" if score >= 70 else ("#f59e0b" if score >= 40 else "#ef4444")
        dur = c.get("duration", 0) or 0
        time_range = f"{fmt_ts(c.get('start', 0))}–{fmt_ts(c.get('end', 0))}"
        reason = html.escape(str(c.get("reason") or "—")[:160])
        quote = html.escape(str(c.get("quote") or "—")[:160])
        thumb = c.get("_thumb_b64") or ""
        thumb_html = (f'<img class="oc-thumb" src="{thumb}" alt="clip thumbnail" loading="lazy">'
                      if thumb else '<div class="oc-thumb oc-thumb-fallback">🎬</div>')
        delay = min(i * 0.07, 0.5)
        cards.append(
            f'<div class="oc-card" style="animation-delay:{delay:.2f}s">'
            f'<div class="oc-thumb-wrap">{thumb_html}'
            f'<div class="oc-dur-badge">{dur:.0f}s</div>'
            f'<div class="oc-score-badge" style="--ring:{ring}">'
            f'<svg viewBox="0 0 44 44"><circle cx="22" cy="22" r="19" class="oc-track"/>'
            f'<circle cx="22" cy="22" r="19" class="oc-fill" '
            f'stroke="{ring}" stroke-dasharray="{score * 1.19:.0f} 119"/></svg>'
            f'<span>{score:.0f}</span></div></div>'
            f'<div class="oc-card-body">'
            f'<div class="oc-card-title">Clip {i + 1} <span class="oc-time">{time_range}</span></div>'
            f'<div class="oc-card-reason">{reason}</div>'
            f'<div class="oc-card-quote">&ldquo;{quote}&rdquo;</div>'
            f"</div></div>"
        )
    return '<div class="oc-cards">' + "".join(cards) + "</div>"


def _err(msg: str, job):
    """Error-shaped return tuple for the analyze handler."""
    return (f"❌ {msg}", "", EMPTY_SCORE_CARDS,
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
        return (status, transcript_text, EMPTY_SCORE_CARDS,
                gr.update(choices=[], value=[]), job)

    labels = [clip_label(i, c) for i, c in enumerate(clips)]
    job["labels"] = labels  # remember for mapping checkbox picks -> clip index

    # ---- thumbnails for the Opus-style cards --------------------------------
    thumbdir = os.path.join(workdir, "thumbs")
    os.makedirs(thumbdir, exist_ok=True)
    src_video = job.get("video")
    for i, c in enumerate(clips):
        try:
            tp = os.path.join(thumbdir, f"thumb_{i}.jpg")
            ts = float(c.get("start", 0) or 0) + min(1.0, float(c.get("duration", 0) or 0) / 3)
            if src_video and make_thumb(src_video, ts, tp):
                c["_thumb_b64"] = _thumb_b64(tp)
        except Exception:
            pass

    note = ""
    dur = probe_duration(job["video"])
    if dur and dur > LONG_VIDEO_MIN * 60:
        note = (f"\n\nℹ️ This video is {fmt_ts(dur)} long — transcription and "
                "rendering run on CPU, so this may take a while. Feel free to leave the tab open.")

    title = job.get("title", "")
    tags = " ".join(job.get("hashtags", []))
    status = (f"✅ Analyzed {len(segments)} transcript segments, ranked {len(clips)} clips. "
              f"Suggested title: **{title}**  {tags}{note}")
    return (status, transcript_text, score_cards_html(clips),
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
/* ============================================================
   ClipForge — Opus-style professional dark UI (v3)
   Original design in the spirit of modern AI clipping apps:
   deep-black canvas, violet→fuchsia accents, glassy cards,
   virality score badges, buttery animations, fully responsive.
   ============================================================ */
:root {
  --oc-bg: #08080e;
  --oc-bg2: #0d0d16;
  --oc-surface: rgba(255,255,255,.035);
  --oc-surface2: rgba(255,255,255,.06);
  --oc-border: rgba(255,255,255,.09);
  --oc-text: #f2f2f8;
  --oc-muted: #a7a7c4;
  --oc-dim: #6e6e8c;
  --oc-violet: #8b5cf6;
  --oc-fuchsia: #d946ef;
  --oc-grad: linear-gradient(135deg, #8b5cf6 0%, #d946ef 100%);
  --oc-grad-soft: linear-gradient(135deg, rgba(139,92,246,.16), rgba(217,70,239,.16));
  --oc-radius: 16px;
  --oc-font: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.gradio-container {
  background: radial-gradient(1200px 600px at 15% -5%, rgba(139,92,246,.10), transparent 60%),
              radial-gradient(1000px 500px at 90% 0%, rgba(217,70,239,.07), transparent 60%),
              var(--oc-bg) !important;
  color: var(--oc-text);
  font-family: var(--oc-font) !important;
  max-width: 100% !important;
  padding: 0 !important;
}
.gradio-container * { font-family: var(--oc-font); }

/* ---------- layout: sidebar + main ---------- */
.oc-app { display: flex; min-height: 100vh; }
.oc-sidebar {
  width: 248px; min-width: 248px; max-width: 248px;
  background: linear-gradient(180deg, #0c0c15 0%, #0a0a12 100%);
  border-right: 1px solid var(--oc-border);
  padding: 1.4rem 1rem;
  display: flex; flex-direction: column; gap: .4rem;
  position: sticky; top: 0; height: 100vh;
  animation: slideInLeft .5s cubic-bezier(.22,1,.36,1) both;
}
@keyframes slideInLeft { from { opacity: 0; transform: translateX(-24px); } to { opacity: 1; transform: none; } }
.oc-logo { display: flex; align-items: center; gap: .65rem; padding: .4rem .6rem 1.2rem; }
.oc-logo-mark {
  width: 38px; height: 38px; border-radius: 12px; background: var(--oc-grad);
  display: flex; align-items: center; justify-content: center;
  font-size: 1.25rem; box-shadow: 0 4px 18px rgba(139,92,246,.45);
}
.oc-logo-name { font-weight: 800; font-size: 1.15rem; letter-spacing: -.01em; }
.oc-logo-sub { font-size: .68rem; color: var(--oc-dim); font-weight: 600; letter-spacing: .12em; text-transform: uppercase; }
.oc-nav-label { font-size: .68rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase;
  color: var(--oc-dim); padding: .9rem .7rem .35rem; }
.oc-nav-item {
  display: flex; align-items: center; gap: .7rem;
  padding: .68rem .8rem; border-radius: 12px; color: var(--oc-muted);
  font-weight: 600; font-size: .92rem; cursor: pointer; border: 1px solid transparent;
  transition: all .18s ease; text-decoration: none;
}
.oc-nav-item:hover { background: var(--oc-surface2); color: var(--oc-text); transform: translateX(2px); }
.oc-nav-item.active { background: var(--oc-grad-soft); color: #fff; border-color: rgba(139,92,246,.35); }
.oc-nav-item .oc-ico { font-size: 1.1rem; width: 1.5rem; text-align: center; }
.oc-side-foot { margin-top: auto; padding: .8rem .6rem .2rem; border-top: 1px solid var(--oc-border);
  font-size: .75rem; color: var(--oc-dim); }
.oc-main { flex: 1; min-width: 0; padding: 1.6rem 2rem 3rem; max-width: 1180px;
  animation: fadeUp .55s ease both; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }

/* ---------- header ---------- */
.oc-header { margin-bottom: 1.4rem; }
.oc-kicker { display: inline-flex; align-items: center; gap: .45rem; font-size: .75rem; font-weight: 700;
  letter-spacing: .12em; text-transform: uppercase; color: #c4b5fd;
  background: rgba(139,92,246,.12); border: 1px solid rgba(139,92,246,.3);
  padding: .35rem .8rem; border-radius: 999px; margin-bottom: .8rem; }
.oc-kicker .dot { width: 7px; height: 7px; border-radius: 50%; background: #a78bfa;
  box-shadow: 0 0 10px #a78bfa; animation: blink 2s ease infinite; }
@keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: .35; } }
.oc-title { font-size: clamp(1.7rem, 4vw, 2.5rem); font-weight: 800; letter-spacing: -.025em;
  margin: 0 0 .4rem; }
.oc-title .grad { color: #c4b5fd; }
@supports ((-webkit-background-clip: text) or (background-clip: text)) {
  .oc-title .grad {
    background: var(--oc-grad); -webkit-background-clip: text; background-clip: text;
    color: transparent; -webkit-text-fill-color: transparent;
  }
}
.oc-sub { color: var(--oc-muted); font-size: 1rem; margin: 0; max-width: 640px; line-height: 1.55; }

/* ---------- panels ---------- */
.oc-panel {
  background: var(--oc-surface); border: 1px solid var(--oc-border);
  border-radius: var(--oc-radius); padding: 1.25rem 1.35rem; margin-bottom: 1.1rem;
  backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
  animation: fadeUp .5s ease both;
  transition: border-color .2s ease, box-shadow .2s ease, transform .2s ease;
}
.oc-panel:hover { border-color: rgba(139,92,246,.28); box-shadow: 0 8px 30px rgba(0,0,0,.35); }
.oc-panel-title { font-size: .95rem; font-weight: 700; margin: 0 0 .9rem; display: flex;
  align-items: center; gap: .55rem; }
.oc-panel-title .oc-ico { width: 30px; height: 30px; border-radius: 9px; background: var(--oc-grad-soft);
  border: 1px solid rgba(139,92,246,.3); display: inline-flex; align-items: center;
  justify-content: center; font-size: 1rem; }

/* ---------- Gradio component reskin ---------- */
.gradio-container .block {
  background: var(--oc-surface) !important; border: 1px solid var(--oc-border) !important;
  border-radius: var(--oc-radius) !important; box-shadow: none !important;
}
.gradio-container .form, .gradio-container .styler {
  background: transparent !important; border: none !important; box-shadow: none !important; }
.gradio-container [data-testid="block-label"] {
  background: rgba(255,255,255,.05) !important; color: var(--oc-muted) !important; border: none !important; }
.gradio-container .block label {
  background: rgba(255,255,255,.05) !important; border: 1px solid var(--oc-border) !important;
  color: #d9d9ec !important; }
.gradio-container textarea,
.gradio-container input:not([type="range"]):not([type="checkbox"]):not([type="radio"]):not([type="file"]):not([type="color"]),
.gradio-container select {
  background: rgba(255,255,255,.05) !important; color: var(--oc-text) !important;
  border: 1px solid var(--oc-border) !important; border-radius: 12px !important;
  transition: border-color .18s ease, box-shadow .18s ease;
}
.gradio-container textarea:focus, .gradio-container input:focus, .gradio-container select:focus {
  border-color: rgba(139,92,246,.6) !important; box-shadow: 0 0 0 3px rgba(139,92,246,.18) !important;
  outline: none !important; }
.gradio-container textarea::placeholder, .gradio-container input::placeholder { color: var(--oc-dim) !important; }
.gradio-container .block .wrap, .gradio-container .block .wrap-inner {
  background: transparent !important; border: none !important; box-shadow: none !important; }
.gradio-container .upload-container {
  background: rgba(139,92,246,.05) !important;
  border: 1.5px dashed rgba(139,92,246,.4) !important; border-radius: var(--oc-radius) !important;
  transition: all .2s ease; }
.gradio-container .upload-container:hover {
  background: rgba(139,92,246,.10) !important; border-color: rgba(139,92,246,.75) !important;
  box-shadow: 0 0 24px rgba(139,92,246,.15); }
.gradio-container table { color: var(--oc-text) !important; }
.gradio-container th { background: rgba(139,92,246,.16) !important; color: #fff !important;
  border-color: var(--oc-border) !important; }
.gradio-container td { border-color: var(--oc-border) !important; }
.gradio-container tr:nth-child(even) td { background: rgba(255,255,255,.025) !important; }
.gradio-container h1, .gradio-container h2, .gradio-container h3 { color: var(--oc-text) !important; }
.gradio-container label, .gradio-container legend { color: var(--oc-muted) !important; }

/* ---------- buttons ---------- */
.gradio-container button { border-radius: 13px !important; font-weight: 700 !important;
  transition: transform .16s ease, box-shadow .16s ease, filter .16s ease !important; }
.gradio-container button.primary {
  background: var(--oc-grad) !important; color: #fff !important; border: none !important;
  box-shadow: 0 6px 24px rgba(139,92,246,.4); animation: ctaGlow 2.8s ease-in-out infinite; }
.gradio-container button.primary:hover { transform: translateY(-2px) scale(1.01);
  box-shadow: 0 10px 32px rgba(139,92,246,.55); filter: brightness(1.08); }
.gradio-container button.primary:active { transform: translateY(0) scale(.99); }
@keyframes ctaGlow {
  0%,100% { box-shadow: 0 6px 24px rgba(139,92,246,.35); }
  50% { box-shadow: 0 6px 34px rgba(217,70,239,.5); } }
.gradio-container button:not(.primary) {
  background: var(--oc-surface2) !important; color: var(--oc-text) !important;
  border: 1px solid var(--oc-border) !important; }
.gradio-container button:not(.primary):hover { border-color: rgba(139,92,246,.5) !important;
  transform: translateY(-1px); }

/* ---------- sliders ---------- */
input[type="range"] { -webkit-appearance: none; appearance: none; height: 7px; border-radius: 99px;
  background: linear-gradient(90deg, var(--oc-violet), var(--oc-fuchsia)) !important; outline: none; }
input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; width: 19px; height: 19px;
  border-radius: 50%; border: none; cursor: pointer;
  background: radial-gradient(circle at 30% 30%, #fff, #c4b5fd 55%, #8b5cf6);
  box-shadow: 0 0 12px rgba(139,92,246,.9); transition: transform .15s ease; }
input[type="range"]::-webkit-slider-thumb:hover { transform: scale(1.15); }
input[type="range"]::-moz-range-thumb { width: 19px; height: 19px; border-radius: 50%; border: none;
  cursor: pointer; background: radial-gradient(circle at 30% 30%, #fff, #c4b5fd 55%, #8b5cf6);
  box-shadow: 0 0 12px rgba(139,92,246,.9); }

/* ---------- Opus-style clip cards ---------- */
.oc-cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 1rem; margin-bottom: 1rem; }
.oc-card {
  background: linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.02));
  border: 1px solid var(--oc-border); border-radius: var(--oc-radius); overflow: hidden;
  animation: cardIn .5s cubic-bezier(.22,1,.36,1) both;
  transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease; cursor: default; }
@keyframes cardIn { from { opacity: 0; transform: translateY(18px) scale(.98); } to { opacity: 1; transform: none; } }
.oc-card:hover { transform: translateY(-4px); border-color: rgba(139,92,246,.45);
  box-shadow: 0 16px 40px rgba(0,0,0,.5), 0 0 0 1px rgba(139,92,246,.25); }
.oc-thumb-wrap { position: relative; aspect-ratio: 16/9; background: #101018; overflow: hidden; }
.oc-thumb { width: 100%; height: 100%; object-fit: cover; display: block;
  transition: transform .35s ease; }
.oc-card:hover .oc-thumb { transform: scale(1.05); }
.oc-thumb-fallback { display: flex; align-items: center; justify-content: center;
  font-size: 2.2rem; width: 100%; height: 100%; }
.oc-dur-badge { position: absolute; right: .6rem; bottom: .6rem; background: rgba(0,0,0,.72);
  color: #fff; font-size: .72rem; font-weight: 700; padding: .22rem .55rem; border-radius: 8px;
  backdrop-filter: blur(6px); }
.oc-score-badge { position: absolute; left: .6rem; top: .6rem; width: 52px; height: 52px; }
.oc-score-badge svg { width: 100%; height: 100%; transform: rotate(-90deg); }
.oc-score-badge .oc-track { fill: rgba(8,8,14,.78); stroke: rgba(255,255,255,.14); stroke-width: 4.5; }
.oc-score-badge .oc-fill { fill: none; stroke-width: 4.5; stroke-linecap: round;
  filter: drop-shadow(0 0 6px currentColor); }
.oc-score-badge span { position: absolute; inset: 0; display: flex; align-items: center;
  justify-content: center; font-weight: 800; font-size: .95rem; color: #fff; text-shadow: 0 1px 6px #000; }
.oc-card-body { padding: .9rem 1rem 1rem; }
.oc-card-title { font-weight: 700; font-size: .95rem; color: #fff; margin-bottom: .35rem;
  display: flex; justify-content: space-between; align-items: baseline; gap: .5rem; }
.oc-time { font-size: .75rem; color: var(--oc-dim); font-weight: 600; white-space: nowrap; }
.oc-card-reason { color: #c9c9e4; font-size: .85rem; line-height: 1.45; margin-bottom: .3rem; }
.oc-card-quote { color: var(--oc-dim); font-style: italic; font-size: .82rem; line-height: 1.4;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }

/* ---------- empty state ---------- */
.oc-empty { text-align: center; padding: 3rem 1.2rem; color: var(--oc-dim);
  background: var(--oc-surface); border: 1.5px dashed rgba(139,92,246,.3);
  border-radius: var(--oc-radius); }
.oc-empty-icon { font-size: 2.6rem; margin-bottom: .7rem; animation: float 3s ease-in-out infinite; }
@keyframes float { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-8px); } }
.oc-empty b { color: var(--oc-text); font-size: 1.05rem; }
.oc-empty-sub { margin-top: .4rem; font-size: .88rem; max-width: 420px; margin-inline: auto; line-height: 1.5; }

/* ---------- status pills ---------- */
.oc-status { border-radius: 12px !important; }
.oc-status textarea { font-size: .88rem !important; line-height: 1.55 !important; }

/* ---------- transcript ---------- */
.transcript-box textarea { font-family: ui-monospace, SFMono-Regular, monospace !important;
  font-size: .82rem !important; line-height: 1.7 !important; }

/* ---------- checkbox group (clip picker) ---------- */
.oc-picker label { background: var(--oc-surface) !important; border: 1px solid var(--oc-border) !important;
  border-radius: 12px !important; padding: .6rem .9rem !important; margin: .3rem !important;
  transition: all .16s ease !important; font-size: .85rem !important; }
.oc-picker label:hover { border-color: rgba(139,92,246,.5) !important; transform: translateY(-1px); }
.oc-picker label:has(input:checked) { background: var(--oc-grad-soft) !important;
  border-color: rgba(139,92,246,.6) !important; box-shadow: 0 0 16px rgba(139,92,246,.2); }

/* ---------- caption style radio cards ---------- */
.oc-styles .wrap { gap: .5rem !important; }
.oc-styles label { border-radius: 12px !important; text-transform: capitalize; font-weight: 600 !important; }

/* sidebar wrapper: let .oc-sidebar be the flex item */
.oc-app > div:first-child { display: contents; }
/* accordions */
.oc-acc .label-wrap { font-weight: 700 !important; }

/* ---------- responsive ---------- */
@media (max-width: 900px) {
  .oc-app { flex-direction: column; }
  .oc-sidebar { width: 100%; min-width: 0; max-width: none; height: auto; position: static;
    flex-direction: row; align-items: center; padding: .7rem .9rem; gap: .5rem;
    border-right: none; border-bottom: 1px solid var(--oc-border); overflow-x: auto; }
  .oc-logo { padding: 0 .4rem 0 0; }
  .oc-logo-sub, .oc-nav-label, .oc-side-foot, .cf-step-line { display: none; }
  .oc-nav-item { padding: .5rem .7rem; font-size: .82rem; white-space: nowrap; }
  .oc-main { padding: 1.1rem .9rem 2.5rem; }
  .oc-cards { grid-template-columns: 1fr; }
}
@media (max-width: 520px) {
  .oc-title { font-size: 1.55rem; }
  .oc-panel { padding: 1rem .95rem; }
  .gradio-container button { font-size: .92rem !important; }
}

"""



with gr.Blocks(title="ClipForge — Long Video → Viral Shorts", css=CSS) as demo:
    job_state = gr.State(None)

    with gr.Row(elem_classes=["oc-app"]):
        gr.HTML(SIDEBAR_HTML)

        with gr.Column(elem_classes=["oc-main"]):
            gr.HTML(HEADER_HTML)

            # ---- 1 · Source ------------------------------------------------
            gr.HTML('<div class="oc-panel"><p class="oc-panel-title">'
                    '<span class="oc-ico">📥</span> Source video</p>')
            with gr.Row():
                video_in = gr.Video(
                    label=f"Upload video (MP4/MOV/WebM…, max ~{MAX_UPLOAD_MB} MB)",
                    sources=["upload"])
                yt_in = gr.Textbox(label="…or paste a YouTube URL",
                                   placeholder="https://www.youtube.com/watch?v=…",
                                   lines=2)
            gr.HTML('</div>')

            # ---- options ---------------------------------------------------
            with gr.Accordion("⚙️ Analysis options", open=False, elem_classes=["oc-acc"]):
                with gr.Row():
                    top_n = gr.Slider(1, 10, value=5, step=1, label="Clips to rank")
                    model = gr.Dropdown(
                        ["tiny", "base", "small"],
                        value=os.environ.get("CLIPFORGE_MODEL", "base"),
                        label="Whisper model (tiny = fastest)")
                gr.Markdown("**⏱️ Clip length**")
                dur_preset = gr.Radio(
                    list(DUR_PRESETS.keys()), value=None,
                    label="Duration preset (sets the sliders)")
                with gr.Row():
                    min_dur = gr.Slider(5, 60, value=20, step=1, label="Min (s)")
                    max_dur = gr.Slider(10, 120, value=70, step=1, label="Max (s)")
                focus_prompt = gr.Textbox(
                    label="🎯 Focus prompt (ClipAnything)",
                    placeholder="e.g. money mindset, crypto tips — boosts matching moments")
                with gr.Row():
                    trim_chk = gr.Checkbox(value=False, label="✂️ AI Trim: fillers & silences")
                    dub_chk = gr.Checkbox(value=False, label="🎙️ AI Dub")
                dub_lang = gr.Dropdown(choices=DUB_LANGS, value="urdu", label="Dub language")

            analyze_btn = gr.Button("⚡ Generate Clips", variant="primary", size="lg")
            status_out = gr.Textbox(label="Status", lines=2, interactive=False,
                                    elem_classes=["oc-status"])

            # ---- 2 · Ranked clips ------------------------------------------
            gr.HTML('<div class="oc-panel" style="margin-top:1.2rem"><p class="oc-panel-title">'
                    '<span class="oc-ico">🏆</span> Ranked clips '
                    '<span style="color:var(--oc-dim);font-weight:500;font-size:.82rem">'
                    '— virality score 0–100</span></p>')
            score_cards = gr.HTML(EMPTY_SCORE_CARDS)
            pick = gr.CheckboxGroup(label="✅ Tick clips to render",
                                    choices=[], elem_classes=["oc-picker"])
            gr.HTML('</div>')

            with gr.Accordion("📝 Transcript (timed segments)", open=False, elem_classes=["oc-acc"]):
                transcript_out = gr.Textbox(label="Transcript", lines=18,
                                            interactive=False, elem_classes=["transcript-box"],
                                            max_lines=18)

            # ---- 3 · Customize & render ------------------------------------
            gr.HTML('<div class="oc-panel"><p class="oc-panel-title">'
                    '<span class="oc-ico">🎨</span> Style &amp; export</p>')
            gr.Markdown("**✍️ Caption style**")
            style = gr.Radio(STYLE_CHOICES, value=STYLE_DEFAULT, label="Style",
                             elem_classes=["oc-styles"])
            with gr.Row():
                font_size = gr.Slider(40, 130, step=2, value=76, label="🔠 Size")
                position = gr.Radio(["Bottom", "Center", "Top"], value="Bottom", label="Position")
            with gr.Row():
                text_color = gr.ColorPicker(value="#FFFFFF", label="🔤 Text")
                highlight = gr.ColorPicker(value="#FFFF00", label="✨ Highlight")
                accent = gr.ColorPicker(value="#FFD700", label="🎨 Accent → highlight")
            with gr.Row():
                bg_box = gr.Checkbox(value=False, label="📦 Caption box")
                emoji_chk = gr.Checkbox(value=False, label="😀 Emoji captions")
            gr.Markdown("**🖼️ Format**")
            with gr.Row():
                aspect = gr.Radio(["9:16", "1:1", "16:9"], value="9:16", label="Aspect ratio")
                sfx = gr.Checkbox(value=True, label="🔊 Whoosh SFX")
                zoom = gr.Checkbox(value=True, label="🔍 Punch zooms")
            logo = gr.Image(label="🎨 Brand logo watermark (optional)", type="filepath")
            gr.HTML('</div>')

            render_btn = gr.Button("🎬 Render Selected Clips", variant="primary", size="lg")
            render_status = gr.Textbox(label="Render status", lines=2, interactive=False,
                                       elem_classes=["oc-status"])
            gallery = gr.Gallery(label="Finished clips — click to preview", columns=3,
                                 object_fit="contain", height="auto")
            downloads = gr.File(label="⬇️ Download MP4s", file_count="multiple", interactive=False)
            titles_md = gr.Markdown(
                "🎬 *Titles, descriptions & hashtags for your rendered clips will appear here.*")

            gr.HTML(HOW_HTML)

    # ---- reactive wiring --------------------------------------------------
    def _apply_preset(choice):
        if choice in DUR_PRESETS:
            lo, hi = DUR_PRESETS[choice]
            return gr.update(value=lo), gr.update(value=hi)
        return gr.update(), gr.update()

    dur_preset.change(_apply_preset, inputs=dur_preset, outputs=[min_dur, max_dur])
    accent.change(lambda c: c, inputs=accent, outputs=highlight)

    analyze_btn.click(
        analyze_video,
        inputs=[video_in, yt_in, top_n, min_dur, max_dur, model,
                focus_prompt, trim_chk, dub_chk, dub_lang],
        outputs=[status_out, transcript_out, score_cards, pick, job_state],
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
    # Optional HTTP basic auth for self-hosted/tunnel deployments. Set
    # CLIPFORGE_USER + CLIPFORGE_PASS in the environment (never in the repo).
    _user = os.environ.get("CLIPFORGE_USER")
    _pass = os.environ.get("CLIPFORGE_PASS")
    _auth = (_user, _pass) if _user and _pass else None
    demo.launch(server_name="0.0.0.0", server_port=port, auth=_auth)
