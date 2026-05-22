#!/usr/bin/env python3
"""
Whole-Video LipSync Pipeline
=============================
Sends the entire video + audio to sync.so sync-3 model in a single API call,
with intelligent auto-detection of faces, active speakers, occlusion, and complexity.

Skips scene splitting entirely — lets sync-3 handle the full video natively.

Usage:
    python3 WholeVideoSync.py --video input.mp4 --audio dubbed.wav
    python3 WholeVideoSync.py --video input.mp4 --audio dubbed.wav --api-key KEY
    python3 WholeVideoSync.py --video input.mp4 --audio dubbed.wav --skip-analysis
    python3 WholeVideoSync.py  # interactive mode
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from config import (
    SYNC_API_BASE, SYNC_MODEL, SYNC_MODE, POLL_INTERVAL, MAX_WAIT,
    S3_BUCKET, S3_PREFIX, S3_PRESIGN_EXPIRY, FACE_MODEL_PATH, logger,
)
from utils import (
    get_video_duration, get_video_fps, check_api_key, check_s3_access,
    phase_timer, timings, progress,
)
from sync_api.client import sync_create_job, sync_download

# Longer poll for whole-video jobs (up to 1 hour)
WHOLE_VIDEO_MAX_WAIT = 3600  # 60 minutes
WHOLE_VIDEO_POLL_INTERVAL = 15  # check every 15s


# ---------------------------------------------------------------------------
# Video Intelligence Analysis (whole-video)
# ---------------------------------------------------------------------------

def analyze_whole_video(video_path, sample_interval=1.0):
    """Analyze the full video for face count, complexity, and lip occlusion.

    Samples frames at `sample_interval` seconds apart. Returns a dict:
        {
            "has_faces": bool,
            "multi_person": bool,
            "is_complex": bool,
            "has_occlusion": bool,
            "avg_face_count": float,
            "max_face_count": int,
            "avg_yaw": float,
            "total_frames_sampled": int,
        }
    """
    import cv2
    import numpy as np
    import mediapipe as mp

    logger.info("Analyzing video for faces, complexity, and occlusion...")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Could not open video for analysis — defaulting to safe flags.")
        return _default_analysis()

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, int(fps * sample_interval))

    # MediaPipe Face Landmarker
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=10,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )

    face_counts = []
    yaw_values = []
    face_areas = []
    multi_face_frames = 0
    small_face_frames = 0
    side_angle_frames = 0
    occlusion_scores = []
    frames_sampled = 0

    with FaceLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            num_faces = len(result.face_landmarks)
            face_counts.append(num_faces)
            frames_sampled += 1

            if num_faces > 1:
                multi_face_frames += 1

            if num_faces > 0:
                # Analyze largest face
                largest_face = _get_largest_face(result.face_landmarks, w, h)
                landmarks = largest_face

                # Face area
                xs = [lm.x for lm in landmarks]
                ys = [lm.y for lm in landmarks]
                face_w = max(xs) - min(xs)
                face_h = max(ys) - min(ys)
                area_ratio = face_w * face_h
                face_areas.append(area_ratio)
                if area_ratio < 0.02:
                    small_face_frames += 1

                # Yaw estimation (nose tip vs eye midpoint)
                nose = landmarks[1]  # NOSE_TIP
                left_eye = landmarks[263]  # LEFT_EYE_OUTER
                right_eye = landmarks[33]  # RIGHT_EYE_OUTER
                eye_mid_x = (left_eye.x + right_eye.x) / 2
                eye_dist = abs(left_eye.x - right_eye.x)
                if eye_dist > 0.001:
                    yaw = abs(nose.x - eye_mid_x) / eye_dist
                    yaw_deg = yaw * 90
                    yaw_values.append(yaw_deg)
                    if yaw_deg > 30:
                        side_angle_frames += 1

                # Lip occlusion check
                occ_score = _check_lip_occlusion(landmarks, frame, w, h)
                occlusion_scores.append(occ_score)

            if frames_sampled % 20 == 0:
                progress(min(frame_idx, total_frames), total_frames, "Video analysis")

            frame_idx += frame_step
            if frame_idx >= total_frames:
                break

    cap.release()
    progress(total_frames, total_frames, "Video analysis")

    if frames_sampled == 0:
        return _default_analysis()

    avg_faces = sum(face_counts) / len(face_counts)
    max_faces = max(face_counts)
    avg_yaw = sum(yaw_values) / len(yaw_values) if yaw_values else 0.0

    # Complexity scoring (same logic as scene-level analysis)
    complexity = 0.0
    if frames_sampled > 0:
        multi_ratio = multi_face_frames / frames_sampled
        small_ratio = small_face_frames / frames_sampled
        side_ratio = side_angle_frames / frames_sampled

        if multi_ratio > 0.5:
            complexity += 0.30
        elif multi_ratio > 0.2:
            complexity += 0.15
        if small_ratio > 0.5:
            complexity += 0.25
        elif small_ratio > 0.2:
            complexity += 0.10
        if side_ratio > 0.4:
            complexity += 0.20
        elif avg_yaw > 25:
            complexity += 0.10

    # Occlusion
    avg_occlusion = sum(occlusion_scores) / len(occlusion_scores) if occlusion_scores else 0.0
    has_occlusion = avg_occlusion >= 0.3

    result = {
        "has_faces": avg_faces > 0.1,
        "multi_person": max_faces > 1 and (multi_face_frames / max(frames_sampled, 1)) > 0.15,
        "is_complex": complexity >= 0.3,
        "has_occlusion": has_occlusion,
        "avg_face_count": round(avg_faces, 2),
        "max_face_count": max_faces,
        "avg_yaw": round(avg_yaw, 1),
        "complexity_score": round(complexity, 2),
        "avg_occlusion_score": round(avg_occlusion, 2),
        "total_frames_sampled": frames_sampled,
    }

    logger.info(
        f"Analysis: faces={avg_faces:.1f} (max {max_faces}), "
        f"complex={result['is_complex']}, multi={result['multi_person']}, "
        f"occlusion={result['has_occlusion']}, yaw={avg_yaw:.1f}deg"
    )
    return result


def _get_largest_face(face_landmarks_list, w, h):
    """Return the landmark set for the largest face by bounding-box area."""
    best = None
    best_area = 0
    for landmarks in face_landmarks_list:
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area > best_area:
            best_area = area
            best = landmarks
    return best


def _check_lip_occlusion(landmarks, frame, w, h):
    """Check if lips are occluded. Returns occlusion score 0.0-1.0."""
    import cv2
    import numpy as np

    OUTER_LIP = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        291, 409, 270, 269, 267, 0, 37, 39, 40, 185
    ]

    lip_xs = [landmarks[i].x * w for i in OUTER_LIP]
    lip_ys = [landmarks[i].y * h for i in OUTER_LIP]

    cx, cy = np.mean(lip_xs), np.mean(lip_ys)
    lip_w = max(lip_xs) - min(lip_xs)
    lip_h = max(lip_ys) - min(lip_ys)
    pad = 0.4
    x1 = max(0, int(cx - lip_w * (0.5 + pad)))
    y1 = max(0, int(cy - lip_h * (0.5 + pad)))
    x2 = min(w, int(cx + lip_w * (0.5 + pad)))
    y2 = min(h, int(cy + lip_h * (0.5 + pad)))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    # Skin ratio (YCrCb)
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    skin_ratio = np.count_nonzero(skin_mask) / skin_mask.size

    # Edge density
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / edges.size

    # Dark region ratio
    dark_ratio = np.count_nonzero(gray < 40) / gray.size

    score = 0.0
    if skin_ratio < 0.15:
        score += 0.35
    elif skin_ratio < 0.25:
        score += 0.15
    if edge_density > 0.25:
        score += 0.20
    if dark_ratio > 0.3:
        score += 0.20

    return score


def _default_analysis():
    """Conservative defaults when analysis can't run."""
    return {
        "has_faces": True,
        "multi_person": True,
        "is_complex": True,
        "has_occlusion": False,
        "avg_face_count": 1.0,
        "max_face_count": 1,
        "avg_yaw": 0.0,
        "complexity_score": 0.3,
        "avg_occlusion_score": 0.0,
        "total_frames_sampled": 0,
    }


# ---------------------------------------------------------------------------
# S3 Upload (whole files)
# ---------------------------------------------------------------------------

def _format_size(bytes_val):
    """Format bytes as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if bytes_val < 1024:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f}TB"


def _s3_upload_with_progress(local_path, s3_key, label="Upload"):
    """Upload a file to S3 with real-time progress display. Returns presigned URL."""
    import boto3
    from config import S3_BUCKET, S3_PRESIGN_EXPIRY

    s3 = boto3.client("s3")
    file_size = os.path.getsize(local_path)
    uploaded = 0

    def _progress_callback(bytes_transferred):
        nonlocal uploaded
        uploaded += bytes_transferred
        pct = (uploaded / file_size * 100) if file_size > 0 else 0
        bar_len = 30
        filled = int(bar_len * uploaded // file_size) if file_size > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)
        print(
            f"\r  {label} |{bar}| {_format_size(uploaded)}/{_format_size(file_size)} ({pct:.0f}%)   ",
            end="", flush=True,
        )

    s3.upload_file(
        str(local_path), S3_BUCKET, s3_key,
        Callback=_progress_callback,
    )
    print()  # newline after progress bar

    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": s3_key},
        ExpiresIn=S3_PRESIGN_EXPIRY,
    )
    return url


def upload_whole_files_to_s3(video_path, audio_path, s3_user, task_name):
    """Upload the full video and audio to S3 with progress. Returns (video_url, audio_url)."""
    from sync_api.s3 import _ensure_s3_folder

    s3_prefix = _ensure_s3_folder(s3_user, task_name)

    video_key = f"{s3_prefix}full_video/{Path(video_path).name}"
    audio_key = f"{s3_prefix}full_audio/{Path(audio_path).name}"

    logger.info(f"Uploading video ({_format_size(os.path.getsize(video_path))})...")
    video_url = _s3_upload_with_progress(str(video_path), video_key, "Video")

    logger.info(f"Uploading audio ({_format_size(os.path.getsize(audio_path))})...")
    audio_url = _s3_upload_with_progress(str(audio_path), audio_key, "Audio")

    logger.info("S3 upload complete.")
    return video_url, audio_url


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def _poll_with_progress(job_id, api_key):
    """Poll sync.so with elapsed time logging and progress display (1-hour timeout)."""
    import requests
    headers = {"x-api-key": api_key}
    elapsed = 0
    last_status = None

    while elapsed < WHOLE_VIDEO_MAX_WAIT:
        time.sleep(WHOLE_VIDEO_POLL_INTERVAL)
        elapsed += WHOLE_VIDEO_POLL_INTERVAL

        resp = requests.get(f"{SYNC_API_BASE}/generate/{job_id}", headers=headers, timeout=30)
        resp.raise_for_status()
        result = resp.json()

        status = result.get("status", "UNKNOWN")
        mins, secs = divmod(elapsed, 60)

        # Log any progress/step info the API provides
        progress_pct = result.get("progress")
        step = result.get("step") or result.get("stage") or result.get("message")

        parts = [f"[{int(mins)}m {int(secs)}s]", f"status={status}"]
        if progress_pct is not None:
            parts.append(f"progress={progress_pct}%")
        if step:
            parts.append(f"step={step}")

        status_line = "  " + " | ".join(parts)

        # Only log when status/progress changes, or every 60s
        if status != last_status or elapsed % 60 < WHOLE_VIDEO_POLL_INTERVAL:
            logger.info(status_line)
            # On first poll, log full response keys for debugging
            if elapsed == WHOLE_VIDEO_POLL_INTERVAL:
                logger.info(f"  API response keys: {list(result.keys())}")
            last_status = status
        else:
            print(f"\r{status_line}   ", end="", flush=True)

        if status in {"COMPLETED", "FAILED", "REJECTED"}:
            print()  # newline after progress
            return result

    raise TimeoutError(f"Job {job_id} timed out after {WHOLE_VIDEO_MAX_WAIT // 60} minutes")


def setup_sync3_task_folder():
    """Create a sync3 task folder with input subfolder. Returns (task_dir, input_dir, output_dir)."""
    tasks_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    folder_name = f"sync3-task({now.strftime('%d-%b, %H:%M')})"
    task_dir = tasks_dir / folder_name
    input_dir = task_dir / "input"
    output_dir = task_dir / "output"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Created task folder: {task_dir}")
    return task_dir, input_dir, output_dir


def run_whole_video_sync(video_path, audio_path, api_key, s3_user,
                         skip_analysis=False, output_path=None, task_dir=None):
    """Run the whole-video lip-sync pipeline.

    Steps:
        1. Validate inputs (duration match)
        2. Analyze video for face intelligence (unless --skip-analysis)
        3. Upload whole video + audio to S3
        4. Submit single sync-3 job with auto-detected flags
        5. Poll and download result
    """
    video_path = Path(video_path).resolve()
    audio_path = Path(audio_path).resolve()

    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)
    if not audio_path.exists():
        logger.error(f"Audio not found: {audio_path}")
        sys.exit(1)

    # Default output path
    if output_path is None:
        if task_dir:
            output_path = Path(task_dir) / "output" / f"{video_path.stem}_synced.mp4"
        else:
            output_path = video_path.parent / f"{video_path.stem}_synced.mp4"
    else:
        output_path = Path(output_path).resolve()

    # Task name for S3 folder
    now = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    task_name = f"sync3_{now.strftime('%d-%m_%H-%M')}"

    # ── Step 1: Validate ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Whole-Video LipSync Pipeline (sync-3)")
    logger.info("=" * 60)

    v_dur = get_video_duration(video_path)
    a_dur = get_video_duration(audio_path)
    fps = get_video_fps(video_path)

    logger.info(f"Video: {video_path.name} ({v_dur:.1f}s, {fps:.1f} fps)")
    logger.info(f"Audio: {audio_path.name} ({a_dur:.1f}s)")

    if abs(v_dur - a_dur) > 2.0:
        logger.warning(
            f"Duration mismatch: video={v_dur:.1f}s, audio={a_dur:.1f}s "
            f"(diff={abs(v_dur - a_dur):.1f}s)"
        )

    # ── Step 2: Video Intelligence ───────────────────────────────────────
    if skip_analysis:
        logger.info("Skipping analysis — using conservative defaults (all features ON).")
        analysis = _default_analysis()
    else:
        with phase_timer("analysis"):
            analysis = analyze_whole_video(video_path)

    # Determine API flags from analysis
    # Always enable active_speaker — sync-3 handles face detection internally
    use_active_speaker = True
    use_reasoning = analysis["is_complex"]
    use_occlusion = analysis["has_occlusion"]

    logger.info(f"API flags: active_speaker={use_active_speaker}, "
                f"reasoning={use_reasoning}, occlusion={use_occlusion}")

    # Save analysis to JSON alongside output
    analysis_path = output_path.parent / f"{video_path.stem}_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    logger.info(f"Analysis saved: {analysis_path.name}")

    # ── Step 3: Upload to S3 ─────────────────────────────────────────────
    with phase_timer("s3_upload"):
        video_url, audio_url = upload_whole_files_to_s3(
            video_path, audio_path, s3_user, task_name
        )

    # ── Step 4: Submit sync-3 job ────────────────────────────────────────
    with phase_timer("sync_api"):
        logger.info("Submitting job to sync.so (sync-3)...")
        job = sync_create_job(
            video_url, audio_url, api_key,
            reasoning=use_reasoning,
            detect_obstructions=use_occlusion,
            active_speaker=use_active_speaker,
            output_name=video_path.stem + "_synced",
        )
        job_id = job.get("id")
        logger.info(f"Job submitted: {job_id}")

        # ── Step 5: Poll and download ────────────────────────────────────
        logger.info(f"Waiting for sync.so to process (polling every {WHOLE_VIDEO_POLL_INTERVAL}s, max {WHOLE_VIDEO_MAX_WAIT // 60}min)...")
        result = _poll_with_progress(job_id, api_key)
        status = result.get("status")

        if status == "COMPLETED":
            output_url = result.get("outputUrl") or result.get("output_url")
            if output_url:
                logger.info("Downloading synced video...")
                sync_download(output_url, str(output_path))
                logger.info(f"Output saved: {output_path}")
            else:
                logger.error("Job completed but no output URL returned.")
                sys.exit(1)
        else:
            error = result.get("error", "Unknown error")
            logger.error(f"Job {status}: {error}")
            sys.exit(1)

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info(f"  Output: {output_path}")
    for phase, dur in timings.items():
        logger.info(f"  {phase}: {dur:.1f}s")
    total = sum(timings.values())
    logger.info(f"  Total: {total:.1f}s")
    logger.info("=" * 60)

    return str(output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Whole-video lip-sync via sync-3 (no scene splitting)"
    )
    parser.add_argument("--video", "-v", help="Input video file (.mp4)")
    parser.add_argument("--audio", "-a", help="Dubbed audio file (.wav)")
    parser.add_argument("--output", "-o", help="Output video path (default: <video>_synced.mp4)")
    parser.add_argument("--api-key", help="sync.so API key (or set SYNC_API_KEY env var)")
    parser.add_argument("--s3-user", help="S3 user folder (or set PIPELINE_S3_USER env var)")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip face analysis, enable all API features by default")
    args = parser.parse_args()

    task_dir = None

    # Interactive mode if no video/audio provided
    if not args.video or not args.audio:
        task_dir, input_dir, output_dir = setup_sync3_task_folder()

        print(f"\n  Whole-Video LipSync Pipeline")
        print("  " + "=" * 50)
        print(f"  Place your files in:\n    {input_dir}")
        print(f"    - One .mp4 video file")
        print(f"    - One .wav audio file (dubbed, same duration)")
        print("  " + "=" * 50)

        while True:
            input("\n  Press ENTER when files are ready...")
            mp4s = sorted(input_dir.glob("*.mp4"))
            wavs = sorted(input_dir.glob("*.wav"))

            if not mp4s:
                print("  [!] No .mp4 file found in input folder.")
                continue
            if not wavs:
                print("  [!] No .wav file found in input folder.")
                continue

            args.video = str(mp4s[0])
            args.audio = str(wavs[0])
            print(f"  Found: {mp4s[0].name}, {wavs[0].name}")
            break

    # API key
    if args.api_key:
        os.environ["SYNC_API_KEY"] = args.api_key
    api_key = check_api_key()

    # S3 user
    s3_user = check_s3_access(args.s3_user or S3_PREFIX)

    run_whole_video_sync(
        video_path=args.video,
        audio_path=args.audio,
        api_key=api_key,
        s3_user=s3_user,
        skip_analysis=args.skip_analysis,
        output_path=args.output,
        task_dir=task_dir,
    )


if __name__ == "__main__":
    main()
