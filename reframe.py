"""Face-aware reframe to a target aspect ratio (default vertical 1080x1920).

Strategy (deliberately CPU-friendly): faces are sampled every N frames and
the median face-center x is computed; ONE static crop window is then used
for the whole clip. No per-frame tracking — a static window avoids jitter
and is far cheaper than dynamic tracking.

Aspect ratios (see ASPECTS in export.py):
  "9:16" -> 1080x1920 (face-aware crop, the default)
  "1:1"  -> 1080x1080 (face-aware crop)
  "16:9" -> 1920x1080 (plain scale; face logic skipped)

Face detector backends (first available wins):
  1. Haar cascade (cv2.CascadeClassifier + haarcascade XML) — OpenCV < 5.
  2. YuNet DNN (cv2.FaceDetectorYN) if an ONNX model file is present at
     ~/.cache/clipforge/face_detection_yunet.onnx (drop it in manually;
     never downloaded automatically).
  3. Center-crop fallback when no detector or no face is found.

Note: opencv-python-headless >= 5.0 removed Haar cascades entirely, so on
modern installs the YuNet-or-center fallback path is the normal one.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np

TARGET_W, TARGET_H = 1080, 1920
TARGET_ASPECT = TARGET_W / TARGET_H  # 9:16

YUNET_MODEL = os.path.expanduser(
    "~/.cache/clipforge/face_detection_yunet.onnx"
)


def _cv2():
    """Lazy OpenCV import: reframe is the only module that needs it."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python-headless is required for reframing; "
            "install it with: pip install opencv-python-headless"
        ) from exc
    return cv2


class _FaceDetector:
    """Thin wrapper over whichever face backend is available (or none)."""

    def __init__(self):
        cv2 = _cv2()
        self._kind: str | None = None
        self._det = None

        # 1) Haar cascade (OpenCV < 5 only).
        cascade_path = os.path.join(
            getattr(cv2, "data", None).haarcascades
            if getattr(cv2, "data", None) else "",
            "haarcascade_frontalface_default.xml",
        )
        if hasattr(cv2, "CascadeClassifier") and os.path.isfile(cascade_path):
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                self._kind = "haar"
                self._det = cascade
                return

        # 2) YuNet DNN, only if the user has placed the model file.
        if hasattr(cv2, "FaceDetectorYN_create") and os.path.isfile(YUNET_MODEL):
            self._kind = "yunet"
            return  # created per-frame-size on first use (see detect)

    @property
    def available(self) -> bool:
        return self._kind is not None

    def detect(self, gray_small) -> list[tuple[float, float]]:
        """Return face-center x coords in the *small* frame's pixel space."""
        cv2 = _cv2()
        if self._kind == "haar":
            faces = self._det.detectMultiScale(
                gray_small, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50)
            )
            if len(faces):
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                return [(x + w / 2.0, y + h / 2.0)]
            return []
        if self._kind == "yunet":
            h, w = gray_small.shape[:2]
            det = cv2.FaceDetectorYN_create(
                YUNET_MODEL, "", (w, h), score_threshold=0.6
            )
            _, faces = det.detect(gray_small)
            if faces is not None:
                # faces rows: x, y, w, h, ... — take the largest.
                best = max(faces, key=lambda f: f[2] * f[3])
                return [(best[0] + best[2] / 2.0, best[1] + best[3] / 2.0)]
            return []
        return []


def _face_centers(
    src: str,
    start: float | None = None,
    end: float | None = None,
    sample_every: int = 15,
) -> tuple[list[float], int, int]:
    """Sample frames, return median-able face-center x coords + (W, H)."""
    cv2 = _cv2()
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {src!r}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    first = int((start or 0.0) * fps)
    last = int(end * fps) if end else total
    last = min(last, total) if total else last

    centers: list[float] = []
    detector = _FaceDetector()
    if not detector.available:
        cap.release()
        return [], W, H  # no face backend -> caller falls back to center crop
    # Downscale for detection speed; map coords back to full res.
    DET_W = 480
    for idx in range(first, last, sample_every):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        scale = DET_W / max(1, frame.shape[1])
        small = cv2.resize(
            frame, (DET_W, int(frame.shape[0] * scale)),
            interpolation=cv2.INTER_LINEAR,
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        for cx, _cy in detector.detect(gray):
            centers.append(cx / scale)
            break  # one (largest) face per sampled frame
    cap.release()
    return centers, W, H


def _aspects():
    """Lazy ASPECTS import (export imports this module at top level)."""
    from export import ASPECTS
    return ASPECTS


def compute_vertical_crop(
    src: str,
    ar: str = "9:16",
    start: float | None = None,
    end: float | None = None,
    target: tuple[int, int] | None = None,
) -> dict:
    """Compute one static crop window for [start, end] (or full video).

    The window's aspect comes from `ar` (see ASPECTS; default "9:16").
    `target` overrides the ar-derived pixel target when given.

    Returns {"x", "y", "w", "h", "faces"} with even integer coords (required
    for yuv420p). x is anchored on the median detected face center when any
    face is found, else centered. y is always vertically centered.
    """
    if not os.path.isfile(src):
        raise FileNotFoundError(f"Video not found: {src!r}")
    centers, W, H = _face_centers(src, start, end)
    if target is None:
        target = _aspects().get(ar, _aspects()["9:16"])
    tw, th = target
    aspect = tw / th

    if W / H >= aspect:
        # Source wider than 9:16 -> crop width, keep full height.
        cw = int(H * aspect)
        ch = H
    else:
        # Source taller/narrower -> crop height, keep full width.
        cw = W
        ch = int(W / aspect)

    cw -= cw % 2
    ch -= ch % 2

    if centers:
        cx = float(np.median(centers))
        # Light smoothing: average of median and mean to damp outliers.
        cx = 0.7 * cx + 0.3 * float(np.mean(centers))
        x = int(np.clip(cx - cw / 2.0, 0, W - cw))
    else:
        x = (W - cw) // 2
    x -= x % 2
    y = (H - ch) // 2
    y -= y % 2

    return {"x": x, "y": y, "w": cw, "h": ch, "faces": len(centers)}


def _escape_sub_path(path: str) -> str:
    # Escape chars that are special inside an ffmpeg filter argument.
    return (
        path.replace("\\", "/")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace(",", "\\,")
        .replace(";", "\\;")
    )


def reframe_to_vertical(
    src: str,
    dst: str,
    crop: dict | None = None,
    ass_path: str | None = None,
    ar: str = "9:16",
    start: float | None = None,
    end: float | None = None,
    crf: int = 20,
) -> str:
    """Crop to an aspect window and scale to the target size.

    Args:
        src, dst: input / output paths.
        crop: dict from compute_vertical_crop(); computed automatically
            for [start, end] when omitted (skipped for "16:9").
        ass_path: optional ASS subtitle file to burn in during the same
            encode (avoids an extra re-encode).
        ar: target aspect ratio ("9:16" | "1:1" | "16:9"). "16:9" skips
            face logic entirely and does a plain scale to 1920x1080.
        start, end: window for automatic crop computation.
        crf: x264 quality.

    Returns `dst`. Raises RuntimeError on ffmpeg failure.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is not installed or not on PATH.")
    aspects = _aspects()
    if ar not in aspects:
        raise ValueError(
            f"Unknown aspect ratio {ar!r}; choose from {sorted(aspects)}"
        )
    tw, th = aspects[ar]

    if ar == "16:9":
        # Landscape output: no face-aware crop, plain scale.
        vf = f"scale={tw}:{th}:flags=lanczos,setsar=1,format=yuv420p"
    else:
        if crop is None:
            crop = compute_vertical_crop(src, ar=ar, start=start, end=end)
        vf = (
            f"crop={crop['w']}:{crop['h']}:{crop['x']}:{crop['y']},"
            f"scale={tw}:{th}:flags=lanczos,"
            "setsar=1,format=yuv420p"
        )
    if ass_path:
        vf += f",subtitles='{_escape_sub_path(os.path.abspath(ass_path))}'"
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-c:a", "aac", "-ar", "44100", "-ac", "2",
        "-movflags", "+faststart",
        dst,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(dst):
        err = (proc.stderr or "").strip()
        raise RuntimeError(f"reframe_to_vertical failed: {err[-600:]}")
    return dst
