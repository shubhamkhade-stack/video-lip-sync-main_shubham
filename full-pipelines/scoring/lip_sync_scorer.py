"""
lip_sync_scorer.py — Corrected Lip-Sync Quality Scorer
=======================================================
Drop-in replacement for scoring/lip_sync_scorer.py.

Reports TWO independent measurements (kept separate on purpose):

  1. SYNC ACCURACY  (PRIMARY)   — LSE-C / LSE-D via SyncNet. The ONLY metric
     that tells you whether the mouth actually matches the audio.
  2. VISUAL QUALITY (SECONDARY) — resolution-matched face-preservation SSIM
     + mouth sharpness. Detects blur / artifacts ONLY. Does NOT measure sync.

Bugs fixed vs the original scorer
---------------------------------
  * RESOLUTION MISMATCH: sync-3 / lipsync-2-pro change the output resolution
    and sometimes fps. The original compared original-vs-synced by SSIM and
    pixel bounding boxes, so a resolution change made it compare mismatched
    frames -> garbage low score. FIX: every synced frame is resized to the
    original's resolution before any comparison.
  * WRONG PENALTY: the original scored lip-movement > 30 px as "distortion".
    A different language legitimately needs different mouth shapes — that
    punished correct sync. FIX: lip movement is informational only, not scored.
  * NO REAL SYNC METRIC: the original had none. FIX: SyncNet LSE-C/LSE-D added
    as the primary score (see run_syncnet() — needs SyncNet installed).

Dependencies: opencv-python, numpy, mediapipe, scikit-image (all already in
the pipeline). SyncNet is optional but required for the primary score.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import logging
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp_lib
from skimage.metrics import structural_similarity as ssim

from config import FACE_MODEL_PATH, ALL_LIP_INDICES, logger

logger = logging.getLogger("lip_sync_scorer")


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — PRIMARY METRIC: SyncNet LSE-C / LSE-D
# ─────────────────────────────────────────────────────────────────────────────
def run_syncnet(video_path: str) -> dict:
    """Run SyncNet on a video and return {'lse_c', 'lse_d', 'av_offset'}.

    LSE-C (confidence)  : higher = better sync.
    LSE-D (distance)    : lower  = better sync.
    av_offset (frames)  : 0 = perfectly aligned.

    Requires the SyncNet repo. Set env var SYNCNET_DIR to its path, e.g.
        export SYNCNET_DIR=/path/to/syncnet_python
    Repo: https://github.com/joonson/syncnet_python  (also used by Wav2Lip's
    LSE evaluation). If SYNCNET_DIR is unset or the scripts are missing, this
    returns {'available': False} and the caller falls back to visual-only.
    """
    syncnet_dir = os.environ.get("SYNCNET_DIR", "").strip()
    if not syncnet_dir or not Path(syncnet_dir).is_dir():
        logger.warning("SYNCNET_DIR not set — sync accuracy NOT measured. "
                       "Install SyncNet and set SYNCNET_DIR for the real score.")
        return {"available": False}

    pipeline = Path(syncnet_dir) / "run_pipeline.py"
    scorer = Path(syncnet_dir) / "run_syncnet.py"
    if not pipeline.exists() or not scorer.exists():
        logger.warning("SyncNet scripts not found in %s — sync NOT measured.",
                       syncnet_dir)
        return {"available": False}

    ref = "scene"
    with tempfile.TemporaryDirectory(prefix="syncnet_") as tmp:
        try:
            subprocess.run(
                ["python", str(pipeline), "--videofile", str(video_path),
                 "--reference", ref, "--data_dir", tmp],
                cwd=syncnet_dir, capture_output=True, text=True, check=True, timeout=600,
            )
            res = subprocess.run(
                ["python", str(scorer), "--videofile", str(video_path),
                 "--reference", ref, "--data_dir", tmp],
                cwd=syncnet_dir, capture_output=True, text=True, check=True, timeout=600,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            logger.error("SyncNet run failed: %s", exc)
            return {"available": False, "error": str(exc)}

        out = (res.stdout or "") + (res.stderr or "")
        # SyncNet prints lines like:  "AV offset: 3" / "Confidence: 7.421" / "Min dist: 6.832"
        offset = _grab(out, r"AV offset:\s*(-?\d+\.?\d*)")
        conf = _grab(out, r"Confidence:\s*(-?\d+\.?\d*)")
        dist = _grab(out, r"Min dist:\s*(-?\d+\.?\d*)")
        return {
            "available": True,
            "lse_c": conf,           # confidence — higher better
            "lse_d": dist,           # distance   — lower better
            "av_offset": offset,     # frames     — 0 best
        }


def _grab(text: str, pattern: str):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — SECONDARY METRIC: resolution-matched visual quality
# ─────────────────────────────────────────────────────────────────────────────
def _init_landmarker():
    BaseOptions = mp_lib.tasks.BaseOptions
    FaceLandmarker = mp_lib.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp_lib.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp_lib.tasks.vision.RunningMode
    return FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=RunningMode.VIDEO, num_faces=5,
        min_face_detection_confidence=0.5, min_tracking_confidence=0.5,
    ))


def _lip_bbox(landmarks, w, h, pad=10):
    xs = [landmarks[i].x * w for i in ALL_LIP_INDICES]
    ys = [landmarks[i].y * h for i in ALL_LIP_INDICES]
    return (max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad),
            min(w, int(max(xs)) + pad), min(h, int(max(ys)) + pad))


def _face_bbox(landmarks, w, h, pad=20):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    return (max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad),
            min(w, int(max(xs)) + pad), min(h, int(max(ys)) + pad))


def _sharpness(frame, bbox):
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def score_visual_quality(original_path: str, synced_path: str, sample_fps: int = 3) -> dict:
    """Resolution-matched visual-artifact check (face preservation + sharpness).

    This is a SECONDARY check — it detects blur/distortion, NOT sync accuracy.
    """
    cap_o = cv2.VideoCapture(str(original_path))
    cap_s = cv2.VideoCapture(str(synced_path))

    # --- FIX: resolution match -------------------------------------------------
    ow = int(cap_o.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh = int(cap_o.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sw = int(cap_s.get(cv2.CAP_PROP_FRAME_WIDTH))
    sh = int(cap_s.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resized = (ow, oh) != (sw, sh)
    if resized:
        logger.info("resolution differs (orig %dx%d vs sync %dx%d) — "
                    "resizing synced frames to original before comparison",
                    ow, oh, sw, sh)

    fps = cap_o.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(fps / sample_fps))
    landmarker = _init_landmarker()

    face_ssims, sharp_o, sharp_s = [], [], []
    idx = 0
    while True:
        ok_o, fo = cap_o.read()
        ok_s, fs = cap_s.read()
        if not ok_o or not ok_s:
            break
        if idx % step == 0:
            # FIX: bring synced frame into the original's pixel space
            if (fs.shape[1], fs.shape[0]) != (ow, oh):
                fs = cv2.resize(fs, (ow, oh), interpolation=cv2.INTER_AREA)

            ts = int(cap_o.get(cv2.CAP_PROP_POS_MSEC))
            ro = landmarker.detect_for_video(
                mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB,
                             data=cv2.cvtColor(fo, cv2.COLOR_BGR2RGB)), ts)
            rs = landmarker.detect_for_video(
                mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB,
                             data=cv2.cvtColor(fs, cv2.COLOR_BGR2RGB)), ts + 1)
            if ro.face_landmarks and rs.face_landmarks:
                lm_o = ro.face_landmarks[0]
                fx1, fy1, fx2, fy2 = _face_bbox(lm_o, ow, oh)
                go = cv2.cvtColor(fo[fy1:fy2, fx1:fx2], cv2.COLOR_BGR2GRAY)
                gs = cv2.cvtColor(fs[fy1:fy2, fx1:fx2], cv2.COLOR_BGR2GRAY)
                if go.shape == gs.shape and go.size > 0:
                    win = min(7, go.shape[0], go.shape[1])
                    if win >= 3 and win % 2 == 1:
                        face_ssims.append(ssim(go, gs, win_size=win))
                lb = _lip_bbox(lm_o, ow, oh)
                sharp_o.append(_sharpness(fo, lb))
                sharp_s.append(_sharpness(fs, lb))
        idx += 1

    cap_o.release()
    cap_s.release()
    landmarker.close()

    if not face_ssims:
        return {"visual_quality_score": None, "error": "no faces detected",
                "resolution_matched": resized}

    avg_ssim = float(np.mean(face_ssims))
    so = float(np.mean(sharp_o)) if sharp_o else 1.0
    ss = float(np.mean(sharp_s)) if sharp_s else 1.0
    sharp_ratio = ss / so if so > 0 else 1.0

    # Visual quality = face preserved (60%) + mouth not blurrier than source (40%)
    face_score = min(avg_ssim / 0.95, 1.0) * 100
    sharp_score = min(sharp_ratio / 0.9, 1.0) * 100
    visual = round(0.60 * face_score + 0.40 * sharp_score, 1)

    return {
        "visual_quality_score": visual,            # 0-100, artifact check only
        "face_preservation_ssim": round(avg_ssim, 4),
        "mouth_sharpness_ratio": round(sharp_ratio, 4),
        "resolution_matched": resized,
        "frames_scored": len(face_ssims),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — Combined report (sync primary, visual secondary, baseline-relative)
# ─────────────────────────────────────────────────────────────────────────────
def score_scene(original_path: str, synced_path: str,
                baseline_lse: dict | None = None, sample_fps: int = 3) -> dict:
    """Full scene score.

    `baseline_lse` = run_syncnet(<original video>) — the ceiling the dub aims
    to approach. Pass it so the report shows sync RELATIVE to the original.
    """
    sync = run_syncnet(synced_path)
    visual = score_visual_quality(original_path, synced_path, sample_fps)

    report = {
        "sync": sync,                 # PRIMARY — LSE-C/LSE-D
        "visual": visual,             # SECONDARY — artifact check
        "baseline": baseline_lse,     # original video's own LSE for comparison
    }

    # Verdict logic — sync metric decides; visual is a quality gate.
    if sync.get("available"):
        lse_c, lse_d = sync.get("lse_c"), sync.get("lse_d")
        report["primary_metric"] = "LSE-C/LSE-D (SyncNet)"
        if baseline_lse and baseline_lse.get("available"):
            bc = baseline_lse.get("lse_c")
            report["gap_to_baseline_lse_c"] = (
                round(lse_c - bc, 3) if (lse_c is not None and bc is not None) else None)
        report["verdict"] = _verdict(lse_c, lse_d, visual.get("visual_quality_score"))
    else:
        report["primary_metric"] = "NOT MEASURED — install SyncNet (set SYNCNET_DIR)"
        report["verdict"] = "VISUAL-ONLY — sync accuracy unknown until SyncNet is set up"

    return report


def _verdict(lse_c, lse_d, visual_score):
    """Rough verdict. Calibrate thresholds against your own baseline video."""
    if lse_c is None:
        return "sync not measured"
    if visual_score is not None and visual_score < 60:
        return "VISUAL ARTIFACTS — mouth blurry / face altered; check model & input res"
    if lse_c >= 6:
        return "GOOD sync"
    if lse_c >= 4:
        return "MARGINAL sync — re-check audio timing"
    return "POOR sync — audio is mistimed or input video face is not clearly talking"


def score_all_scenes(original_dir, synced_dir, baseline_video=None, sample_fps=3):
    """Score every scene_*.mp4 pair. Computes the baseline LSE once."""
    original_dir, synced_dir = Path(original_dir), Path(synced_dir)
    baseline = run_syncnet(str(baseline_video)) if baseline_video else None
    results = {}
    for orig in sorted(original_dir.glob("scene_*.mp4")):
        synced = synced_dir / orig.name
        if not synced.exists():
            logger.warning("no synced video for %s", orig.name)
            continue
        logger.info("scoring %s ...", orig.stem)
        results[orig.stem] = score_scene(str(orig), str(synced), baseline, sample_fps)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    import sys
    if len(sys.argv) >= 3:
        rep = score_scene(sys.argv[1], sys.argv[2])
        import json
        print(json.dumps(rep, indent=2))
    else:
        print("usage: python lip_sync_scorer.py <original.mp4> <synced.mp4>")
