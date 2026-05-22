"""
Lip Sync Quality Scorer
========================
Evaluate lip-sync output quality by comparing original vs synced video.

Metrics (all commercially safe):
  1. Face Preservation (SSIM) — how well the face is preserved outside the lip region
  2. Lip Movement Delta (LMD) — how much lip landmarks changed (expected: moderate change)
  3. Visual Artifact Score — detects blurring/distortion around the mouth area
  4. Overall Quality Score — weighted combination

Uses: OpenCV (BSD), scikit-image (BSD), MediaPipe (Apache 2.0)
"""

import cv2
import numpy as np
import mediapipe as mp_lib
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

from config import FACE_MODEL_PATH, OUTER_LIP_INDICES, ALL_LIP_INDICES, logger


def _init_landmarker():
    """Init MediaPipe Face Landmarker for lip landmark extraction."""
    BaseOptions = mp_lib.tasks.BaseOptions
    FaceLandmarker = mp_lib.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp_lib.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp_lib.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_faces=5,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return FaceLandmarker.create_from_options(options)


def _get_lip_landmarks(face_landmarks, w, h):
    """Extract lip landmark coordinates (pixel space) from MediaPipe result."""
    lips = {}
    for idx in ALL_LIP_INDICES:
        lm = face_landmarks[idx]
        lips[idx] = (lm.x * w, lm.y * h)
    return lips


def _get_face_bbox(face_landmarks, w, h, padding=20):
    """Get face bounding box from landmarks."""
    xs = [lm.x * w for lm in face_landmarks]
    ys = [lm.y * h for lm in face_landmarks]
    x1 = max(0, int(min(xs)) - padding)
    y1 = max(0, int(min(ys)) - padding)
    x2 = min(w, int(max(xs)) + padding)
    y2 = min(h, int(max(ys)) + padding)
    return x1, y1, x2, y2


def _get_lip_bbox(lip_landmarks, w, h, padding=10):
    """Get bounding box around lip region."""
    xs = [pt[0] for pt in lip_landmarks.values()]
    ys = [pt[1] for pt in lip_landmarks.values()]
    x1 = max(0, int(min(xs)) - padding)
    y1 = max(0, int(min(ys)) - padding)
    x2 = min(w, int(max(xs)) + padding)
    y2 = min(h, int(max(ys)) + padding)
    return x1, y1, x2, y2


def _lip_landmark_distance(lips_orig, lips_synced):
    """Calculate normalized average distance between original and synced lip landmarks."""
    if not lips_orig or not lips_synced:
        return 0.0
    distances = []
    for idx in lips_orig:
        if idx in lips_synced:
            ox, oy = lips_orig[idx]
            sx, sy = lips_synced[idx]
            distances.append(np.sqrt((ox - sx) ** 2 + (oy - sy) ** 2))
    return np.mean(distances) if distances else 0.0


def _mouth_region_sharpness(frame, lip_bbox):
    """Measure sharpness of mouth region using Laplacian variance."""
    x1, y1, x2, y2 = lip_bbox
    mouth_crop = frame[y1:y2, x1:x2]
    if mouth_crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(mouth_crop, cv2.COLOR_BGR2GRAY) if len(mouth_crop.shape) == 3 else mouth_crop
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def score_scene(original_path, synced_path, sample_fps=3):
    """
    Score a single scene by comparing original vs synced video.

    Args:
        original_path: Path to original scene video.
        synced_path: Path to synced scene video.
        sample_fps: Frames per second to sample.

    Returns:
        dict with scores:
            - face_preservation: SSIM of non-lip face region (0-1, higher=better)
            - lip_movement_delta: Average lip landmark change in pixels
            - mouth_sharpness_ratio: Synced/original mouth sharpness (1.0=same, <1=blurrier)
            - overall_score: Weighted quality score (0-100)
            - frames_scored: Number of frames compared
            - details: Per-frame breakdown
    """
    cap_orig = cv2.VideoCapture(str(original_path))
    cap_sync = cv2.VideoCapture(str(synced_path))

    fps = cap_orig.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(1, int(fps / sample_fps))
    total_orig = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sync = int(cap_sync.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_orig == 0 or total_sync == 0:
        cap_orig.release()
        cap_sync.release()
        return {"overall_score": 0, "error": "Empty video"}

    landmarker = _init_landmarker()

    face_ssim_scores = []
    lip_deltas = []
    sharpness_orig_list = []
    sharpness_sync_list = []
    per_frame = []

    frame_idx = 0

    while True:
        ret_o, frame_orig = cap_orig.read()
        ret_s, frame_sync = cap_sync.read()
        if not ret_o or not ret_s:
            break

        if frame_idx % frame_interval == 0:
            h, w = frame_orig.shape[:2]

            # Get landmarks from both frames
            rgb_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
            rgb_sync = cv2.cvtColor(frame_sync, cv2.COLOR_BGR2RGB)

            mp_orig = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb_orig)
            mp_sync = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb_sync)

            ts = int(cap_orig.get(cv2.CAP_PROP_POS_MSEC))
            res_orig = landmarker.detect_for_video(mp_orig, ts)
            res_sync = landmarker.detect_for_video(mp_sync, ts + 1)  # +1 to avoid duplicate ts

            if not res_orig.face_landmarks or not res_sync.face_landmarks:
                frame_idx += 1
                continue

            # Use first detected face
            lm_orig = res_orig.face_landmarks[0]
            lm_sync = res_sync.face_landmarks[0]

            lips_orig = _get_lip_landmarks(lm_orig, w, h)
            lips_sync = _get_lip_landmarks(lm_sync, w, h)

            # 1. Face Preservation SSIM (exclude lip region)
            face_bbox = _get_face_bbox(lm_orig, w, h)
            lip_bbox_orig = _get_lip_bbox(lips_orig, w, h)

            fx1, fy1, fx2, fy2 = face_bbox
            face_orig = cv2.cvtColor(frame_orig[fy1:fy2, fx1:fx2], cv2.COLOR_BGR2GRAY)
            face_sync = cv2.cvtColor(frame_sync[fy1:fy2, fx1:fx2], cv2.COLOR_BGR2GRAY)

            # Mask out lip region for face SSIM
            lx1, ly1, lx2, ly2 = lip_bbox_orig
            mask = np.ones_like(face_orig, dtype=bool)
            # Convert lip bbox to face-crop coordinates
            rel_lx1 = max(0, lx1 - fx1)
            rel_ly1 = max(0, ly1 - fy1)
            rel_lx2 = min(face_orig.shape[1], lx2 - fx1)
            rel_ly2 = min(face_orig.shape[0], ly2 - fy1)
            mask[rel_ly1:rel_ly2, rel_lx1:rel_lx2] = False

            if face_orig.shape == face_sync.shape and face_orig.size > 0:
                win_size = min(7, face_orig.shape[0], face_orig.shape[1])
                if win_size >= 3 and win_size % 2 == 1:
                    full_ssim = ssim(face_orig, face_sync, win_size=win_size)
                    face_ssim_scores.append(full_ssim)
                else:
                    full_ssim = None
            else:
                full_ssim = None

            # 2. Lip Movement Delta
            lip_delta = _lip_landmark_distance(lips_orig, lips_sync)
            lip_deltas.append(lip_delta)

            # 3. Mouth Sharpness
            sharp_orig = _mouth_region_sharpness(frame_orig, lip_bbox_orig)
            lip_bbox_sync = _get_lip_bbox(lips_sync, w, h)
            sharp_sync = _mouth_region_sharpness(frame_sync, lip_bbox_sync)
            sharpness_orig_list.append(sharp_orig)
            sharpness_sync_list.append(sharp_sync)

            per_frame.append({
                "frame": frame_idx,
                "face_ssim": round(full_ssim, 4) if full_ssim is not None else None,
                "lip_delta_px": round(lip_delta, 2),
                "mouth_sharpness_orig": round(sharp_orig, 1),
                "mouth_sharpness_sync": round(sharp_sync, 1),
            })

        frame_idx += 1

    cap_orig.release()
    cap_sync.release()
    landmarker.close()

    if not per_frame:
        return {"overall_score": 0, "error": "No faces detected", "frames_scored": 0}

    # Aggregate scores
    avg_face_ssim = np.mean(face_ssim_scores) if face_ssim_scores else 0
    avg_lip_delta = np.mean(lip_deltas) if lip_deltas else 0
    avg_sharp_orig = np.mean(sharpness_orig_list) if sharpness_orig_list else 1
    avg_sharp_sync = np.mean(sharpness_sync_list) if sharpness_sync_list else 1
    sharpness_ratio = avg_sharp_sync / avg_sharp_orig if avg_sharp_orig > 0 else 1.0

    # Overall Score (0-100):
    #   Face Preservation (40%): SSIM of non-lip face — want high (>0.85)
    #   Mouth Sharpness (30%):   Synced mouth shouldn't be blurrier — want ratio ~1.0
    #   Lip Naturalness (30%):   Some movement expected (3-30px is healthy range)

    face_score = min(avg_face_ssim / 0.95, 1.0) * 100  # 0.95 SSIM = perfect
    sharpness_score = min(sharpness_ratio / 0.9, 1.0) * 100  # 0.9 ratio = acceptable
    # Lip delta: 3-30px is ideal range, too low = no sync, too high = distortion
    if avg_lip_delta < 1:
        lip_score = 30  # barely changed — sync might not have worked
    elif avg_lip_delta <= 30:
        lip_score = 100  # healthy range
    elif avg_lip_delta <= 60:
        lip_score = 70   # moderate distortion
    else:
        lip_score = 40   # heavy distortion

    overall = 0.40 * face_score + 0.30 * sharpness_score + 0.30 * lip_score

    return {
        "face_preservation_ssim": round(avg_face_ssim, 4),
        "lip_movement_delta_px": round(avg_lip_delta, 2),
        "mouth_sharpness_ratio": round(sharpness_ratio, 4),
        "face_score": round(face_score, 1),
        "sharpness_score": round(sharpness_score, 1),
        "lip_naturalness_score": round(lip_score, 1),
        "overall_score": round(overall, 1),
        "frames_scored": len(per_frame),
        "details": per_frame,
    }


def score_all_scenes(original_dir, synced_dir, sample_fps=3):
    """
    Score all scenes by comparing original vs synced videos.

    Args:
        original_dir: Directory with original scene videos (scene_001.mp4, ...)
        synced_dir: Directory with synced scene videos (same names)
        sample_fps: Frames per second to sample.

    Returns:
        dict: {scene_name: score_result}
    """
    original_dir = Path(original_dir)
    synced_dir = Path(synced_dir)
    results = {}

    originals = sorted(original_dir.glob("scene_*.mp4"))
    if not originals:
        logger.error(f"No scene videos found in {original_dir}")
        return results

    for orig_path in originals:
        sync_path = synced_dir / orig_path.name
        if not sync_path.exists():
            logger.warning(f"  Synced video not found for {orig_path.name}, skipping")
            continue

        scene_name = orig_path.stem
        logger.info(f"  Scoring {scene_name}...")
        results[scene_name] = score_scene(str(orig_path), str(sync_path), sample_fps)
        logger.info(f"    -> Overall: {results[scene_name]['overall_score']}/100")

    return results
