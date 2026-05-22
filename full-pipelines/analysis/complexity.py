"""
Scene Complexity & Obstruction Analysis
========================================
Analyze dialogue scenes for face count, complexity, person count (YOLO), and lip obstruction.
"""

import json

import cv2
import mediapipe as mp_lib
import numpy as np
from ultralytics import YOLO

from config import (
    FACE_MODEL_PATH, OUTER_LIP_INDICES,
    NOSE_TIP, LEFT_EYE_OUTER, RIGHT_EYE_OUTER, CHIN,
    logger,
)
from utils import progress


def analyze_complexity_and_obstruction(video_dir, dialogue_scenes, scenes_data, output_dir=None):
    """
    Analyze all dialogue scenes for:
    - Face count & complexity (MediaPipe)
    - Person count (YOLO fallback)
    - Lip obstruction (MediaPipe)
    """
    dialogue_videos = []
    for scene in scenes_data["scenes"]:
        if scene["scene"] in dialogue_scenes:
            vpath = video_dir / f"scene_{scene['scene']:03d}.mp4"
            if vpath.exists():
                dialogue_videos.append((scene["scene"], vpath))

    if not dialogue_videos:
        return {}, {}

    total = len(dialogue_videos)
    logger.info(f"Analyzing {total} dialogue scenes (MediaPipe + YOLO)...")

    # Load models once
    logger.info("  Loading YOLO model...")
    yolo_model = YOLO("yolov8n.pt")

    complexity_results = {}
    obstruction_results = {}

    for idx, (scene_num, video_path) in enumerate(dialogue_videos):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            progress(idx + 1, total, "Scene analysis")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_duration_ms = int(1000.0 / fps) if fps > 0 else 33

        # MediaPipe landmarker for this scene
        BaseOptions = mp_lib.tasks.BaseOptions
        FaceLandmarker = mp_lib.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp_lib.tasks.vision.FaceLandmarkerOptions
        RunningMode = mp_lib.tasks.vision.RunningMode

        lm_options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_faces=5,
            min_face_detection_confidence=0.25,
            min_tracking_confidence=0.25,
        )
        landmarker = FaceLandmarker.create_from_options(lm_options)

        # Complexity accumulators
        face_counts = []
        face_sizes = []
        yaw_angles = []
        displacements = []
        prev_landmarks = None

        # Obstruction accumulators
        obs_face_frames = 0
        obs_obstructed = 0

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            proc_frame = frame
            if w > 1920:
                scale = 1920.0 / w
                proc_frame = cv2.resize(frame, (1920, int(h * scale)))

            ph, pw = proc_frame.shape[:2]
            rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb)
            timestamp_ms = frame_idx * frame_duration_ms
            detection = landmarker.detect_for_video(mp_image, timestamp_ms)

            num_faces = len(detection.face_landmarks)
            face_counts.append(num_faces)

            if num_faces > 0:
                # Complexity: analyze largest face
                best_face = None
                best_size = 0
                for face_lm in detection.face_landmarks:
                    xs = [lm.x * pw for lm in face_lm]
                    ys = [lm.y * ph for lm in face_lm]
                    size = (max(xs) - min(xs)) * (max(ys) - min(ys)) / (ph * pw)
                    if size > best_size:
                        best_size = size
                        best_face = face_lm

                face_sizes.append(best_size)

                # Yaw estimation
                nose = best_face[NOSE_TIP]
                left_eye = best_face[LEFT_EYE_OUTER]
                right_eye = best_face[RIGHT_EYE_OUTER]
                nose_x = nose.x * pw
                eye_center = (left_eye.x * pw + right_eye.x * pw) / 2.0
                eye_width = abs(left_eye.x * pw - right_eye.x * pw)
                yaw = abs((nose_x - eye_center) / eye_width * 90.0) if eye_width >= 1 else 0.0
                yaw_angles.append(yaw)

                # Displacement
                if prev_landmarks is not None:
                    key_indices = [NOSE_TIP, LEFT_EYE_OUTER, RIGHT_EYE_OUTER, CHIN, 61, 291]
                    disps = []
                    for ki in key_indices:
                        dx = (best_face[ki].x - prev_landmarks[ki].x) * pw
                        dy = (best_face[ki].y - prev_landmarks[ki].y) * ph
                        disps.append(np.sqrt(dx**2 + dy**2))
                    displacements.append(float(np.mean(disps)))

                prev_landmarks = best_face

                # Obstruction: analyze first detected face
                obs_face = detection.face_landmarks[0]
                obs_face_frames += 1

                lip_pts = []
                for li in OUTER_LIP_INDICES:
                    lm = obs_face[li]
                    lip_pts.append((int(lm.x * pw), int(lm.y * ph)))
                lip_pts = np.array(lip_pts)
                x1, y1 = lip_pts.min(axis=0)
                x2, y2 = lip_pts.max(axis=0)
                pad_w = int((x2 - x1) * 0.4)
                pad_h = int((y2 - y1) * 0.4)
                x1, y1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
                x2, y2 = min(pw, x2 + pad_w), min(ph, y2 + pad_h)

                if (x2 - x1) >= 10 and (y2 - y1) >= 10:
                    lip_roi = proc_frame[y1:y2, x1:x2]
                    lip_gray = cv2.cvtColor(lip_roi, cv2.COLOR_BGR2GRAY)

                    # Skin ratio
                    ycrcb = cv2.cvtColor(lip_roi, cv2.COLOR_BGR2YCrCb)
                    skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
                    skin_ratio = np.count_nonzero(skin_mask) / skin_mask.size if skin_mask.size > 0 else 0

                    # Edge density
                    edges = cv2.Canny(lip_gray, 50, 150)
                    edge_density = np.count_nonzero(edges) / edges.size if edges.size > 0 else 0

                    obs_score = 0.0
                    if skin_ratio < 0.15:
                        obs_score += 0.35
                    elif skin_ratio < 0.25:
                        obs_score += 0.15
                    if edge_density > 0.25:
                        obs_score += 0.2

                    # Dark regions
                    dark_ratio = np.count_nonzero(lip_gray < 40) / lip_gray.size if lip_gray.size > 0 else 0
                    if dark_ratio > 0.3:
                        obs_score += 0.2

                    if obs_score >= 0.4:
                        obs_obstructed += 1
            else:
                prev_landmarks = None

            frame_idx += 1

        cap.release()
        landmarker.close()

        # YOLO person count (every 10th frame)
        cap2 = cv2.VideoCapture(str(video_path))
        yolo_counts = []
        fi = 0
        while True:
            ret, frame = cap2.read()
            if not ret:
                break
            if fi % 10 == 0:
                results = yolo_model(frame, classes=[0], verbose=False)
                yolo_counts.append(len(results[0].boxes))
            fi += 1
        cap2.release()

        yolo_max = max(yolo_counts) if yolo_counts else 0
        yolo_multi_ratio = sum(1 for c in yolo_counts if c > 1) / len(yolo_counts) if yolo_counts else 0

        # Compute complexity signals
        avg_faces = float(np.mean(face_counts)) if face_counts else 0
        max_faces = max(face_counts) if face_counts else 0
        multi_face_ratio = sum(1 for c in face_counts if c > 1) / len(face_counts) if face_counts else 0
        avg_face_size = float(np.mean(face_sizes)) if face_sizes else 0
        avg_yaw = float(np.mean(yaw_angles)) if yaw_angles else 0
        high_yaw_ratio = sum(1 for y in yaw_angles if y > 30) / len(yaw_angles) if yaw_angles else 0
        avg_displacement = float(np.mean(displacements)) if displacements else 0

        # Complexity score — use max of MediaPipe face ratio and YOLO person ratio
        effective_multi_ratio = max(multi_face_ratio, yolo_multi_ratio)
        cscore = 0.0
        if effective_multi_ratio > 0.5:
            cscore += 0.3
        elif effective_multi_ratio > 0.2:
            cscore += 0.15
        if avg_face_size < 0.02:
            cscore += 0.25
        elif avg_face_size < 0.05:
            cscore += 0.1
        if avg_displacement > 15:
            cscore += 0.2
        elif avg_displacement > 8:
            cscore += 0.1
        if high_yaw_ratio > 0.4:
            cscore += 0.2
        elif avg_yaw > 25:
            cscore += 0.1
        cscore = min(1.0, cscore)

        # Obstruction ratio
        obs_ratio = obs_obstructed / obs_face_frames if obs_face_frames > 0 else 0.0

        effective_max_persons = max(max_faces, yolo_max)

        complexity_results[scene_num] = {
            "avg_face_count": round(avg_faces, 2),
            "max_face_count": max_faces,
            "yolo_max_persons": yolo_max,
            "effective_max_persons": effective_max_persons,
            "avg_face_size": round(avg_face_size, 4),
            "avg_yaw": round(avg_yaw, 2),
            "avg_displacement": round(avg_displacement, 2),
            "complexity_score": round(cscore, 4),
            "is_complex": cscore >= 0.3,
        }

        obstruction_results[scene_num] = {
            "obstruction_ratio": round(obs_ratio, 4),
            "is_obstructed": obs_ratio >= 0.3,
        }

        progress(idx + 1, total, "Scene analysis")

    # Save intermediate results for resume
    if output_dir:
        # Convert int keys to str for JSON
        comp_path = output_dir / "complexity_results.json"
        with open(comp_path, "w") as f:
            json.dump({str(k): v for k, v in complexity_results.items()}, f, indent=2)

        obs_path = output_dir / "obstruction_results.json"
        with open(obs_path, "w") as f:
            json.dump({str(k): v for k, v in obstruction_results.items()}, f, indent=2)

        logger.info(f"Complexity/obstruction results saved: {output_dir}")

    return complexity_results, obstruction_results


def load_complexity_results(output_dir):
    """Load previously saved complexity + obstruction results. Returns (complexity, obstruction) or None."""
    comp_path = output_dir / "complexity_results.json"
    obs_path = output_dir / "obstruction_results.json"
    if not comp_path.exists() or not obs_path.exists():
        return None

    with open(comp_path) as f:
        complexity = {int(k): v for k, v in json.load(f).items()}
    with open(obs_path) as f:
        obstruction = {int(k): v for k, v in json.load(f).items()}

    return complexity, obstruction
