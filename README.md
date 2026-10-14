# ✂️ ClipForge

**Turn long videos into viral vertical shorts** — an open-source, self-hosted
alternative to Opus Clip.

Upload a video (or paste a YouTube URL) and ClipForge will:

1. **Transcribe** it with word-level timestamps (faster-whisper, CPU)
2. **Score** candidate moments with a virality heuristic and rank the best clips
   — optionally biased toward your topic with a **ClipAnything** focus prompt
3. **Reframe** each clip to 1080×1920 (9:16), 1080×1080 (1:1) or 1920×1080
   (16:9) with face-aware cropping
4. **Caption** it karaoke-style (8 styles + full customization, emoji support)
5. **Polish** it with punch zooms, whoosh SFX and loudness normalization
6. **Title** it with auto-generated titles + hashtags + descriptions, ready to post

Optional extras per render: **AI trim** (remove filler words & dead air),
**AI dub** (beta: re-voice the clip in Urdu/Hindi/Spanish/French/Arabic) and
a **brand-kit logo** watermark.

## 🚀 Run locally

Requires **Python 3.10** and **ffmpeg** on PATH.

```bash
git clone <this-repo>
cd clipforge
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:7860 — upload a video or paste a YouTube URL, hit
**Analyze**, tick the clips you like, hit **Render selected**.

> First run downloads the Whisper model (~150 MB for `base`) into `~/.cache`.
> The `tiny` model is fastest if you're in a hurry.

## ☁️ Deploy to Render (free tier, Docker)

This repo ships a `Dockerfile` plus a `render.yaml` Blueprint for one-click
deploys. The app listens on `$PORT` (Render injects it; falls back to `7860`
elsewhere).

1. Push this repo's files to GitHub (everything at the repo root: `app.py`,
   the backend `*.py` modules, `requirements.txt`, `Dockerfile`,
   `.dockerignore`, `render.yaml`, `README.md`).
2. In the [Render dashboard](https://dashboard.render.com): **New → Web
   Service** → connect the repo → Runtime **Docker** → Plan **Free**.
   (Or: **New → Blueprint** and point it at the repo — `render.yaml` handles
   the rest.)
3. Add the env var `CLIPFORGE_MODEL=tiny` (keeps RAM inside the free tier;
   you can still pick `base`/`small` in the UI).
4. Wait for the build (10–15 min first time — CPU deps + ffmpeg), then open
   your `https://<service>.onrender.com` URL. The first analysis downloads
   the Whisper model; after that it stays cached for the life of the
   container.

Notes: the free tier sleeps after ~15 min idle (wakes in ~30–60 s), and CPU
renders are slow (see limitations below).

## ✨ Features

- Video upload (up to ~500 MB) **or** YouTube URL download (yt-dlp)
- 3 Whisper model sizes, 1–10 ranked clips, adjustable min/max clip duration
- **ClipAnything**: type keywords (e.g. "money mindset") and ranking boosts
  clips about that topic (+8 per keyword hit, capped at +24)
- **Duration presets** via the min/max clip-length sliders (shorts < 30s,
  standard 30–60s, long-form 60s+)
- Face-aware 9:16 / 1:1 reframe (Haar cascade → YuNet DNN → smart center-crop
  fallbacks); 16:9 renders as a plain landscape scale
- Karaoke word-highlight captions, **8 styles**: hormozi, minimal, neon,
  karaoke-box, pop, typewriter, elegant, outline
- Caption customization: font size, position (top/center/bottom), text &
  highlight colors, background box, and keyword emoji (💰🔥🏆…)
- **AI trim**: automatically cuts filler words ("um", "uh", "you know", …)
  and dead-air silences > 0.7s, then retimes captions to the trimmed cut
- **AI dub (beta)**: translates the clip's transcript and re-voices it with a
  neural TTS voice — Urdu, Hindi, Spanish, French, Arabic. Needs the extra
  pip deps `edge-tts` and `deep-translator` (in requirements.txt) plus
  network access; failures never break the render, the original audio is kept
- **Brand kit**: optional PNG logo watermarked bottom-right on every clip
- Auto titles + hashtags + post descriptions from keyword extraction
- Sequential CPU rendering with live progress; temp files cleaned up after render

## ⚠️ Known limitations

- **CPU speed**: transcription + rendering are CPU-only. A 1-hour video can take
  a long while — the UI warns you for videos over 15 minutes but won't block you.
- **AI dub (beta)**: translation goes through Google Translate and TTS through
  edge-tts, so dubbing needs network access at render time and adds latency.
  Quality depends on the transcript; always spot-check the dubbed clip.
- **Face detection quirk**: the primary face finder is OpenCV's Haar cascade,
  which is decent for frontal faces but can miss profiles or small faces; the
  pipeline falls back to YuNet (if you supply an ONNX model) and then to a
  motion/center-weighted crop, so it never fails outright.
- **Model download**: faster-whisper fetches its model on first run (~150 MB for
  `base`) — needs network access at runtime and adds latency to the first job.
- **YouTube**: downloads depend on yt-dlp keeping up with YouTube changes.
- **No GPU support** in this build (faster-whisper is pinned to CPU settings).

## 📄 License

MIT — do what you want, just keep the notice. The authors are not affiliated
with Opus Clip or YouTube.
