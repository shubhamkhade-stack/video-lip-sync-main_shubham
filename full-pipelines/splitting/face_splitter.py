"""
Face-Based Splitter
===================
Refine scene splits by detecting dominant face changes within long scenes.
"""

import json
import cv2
import mediapipe as mp_lib

from config import FACE_MODEL_PATH, logger
from utils import run_ffmpeg, extract_scene_number, progress


def face_based_split(scenes_data, dirs, min_scene_duration=3.0, face_shift_threshold=0.25):
    """
    Refine scene splits by detecting dominant face changes within long scenes.
    If the dominant face position shifts significantly (person change / camera pan),
    the scene is sub-split at that point.
    """
    BaseOptions = mp_lib.tasks.BaseOptions
    FaceLandmarker = mp_lib.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp_lib.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp_lib.tasks.vision.RunningMode

    scenes = scenes_data["scenes"]
    new_scenes = []
    re_split_count = 0

    # Only refine scenes longer than min_scene_duration
    long_scenes = [s for s in scenes if s["duration_seconds"] >= min_scene_duration]
    short_scenes = [s for s in scenes if s["duration_seconds"] < min_scene_duration]

    if not long_scenes:
        logger.info("Face-based split: no scenes long enough to refine.")
        return scenes_data

    logger.info(f"Face-based split: analyzing {len(long_scenes)} scenes (>= {min_scene_duration}s)...")

    for si, scene in enumerate(long_scenes):
        sn = scene["scene"]
        video_path = dirs["videos"] / f"scene_{sn:03d}.mp4"
        if not video_path.exists():
            new_scenes.append(scene)
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            new_scenes.append(scene)
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total_frames <= 0:
            cap.release()
            new_scenes.append(scene)
            continue

        frame_duration_ms = int(1000.0 / fps)

        # Sample every 0.3 seconds for face tracking
        sample_interval = max(1, int(fps * 0.3))

        lm_options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_faces=5,
            min_face_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        landmarker = FaceLandmarker.create_from_options(lm_options)

        # Track dominant face center across sampled frames
        face_track = []  # list of (frame_idx, center_x_normalized, center_y_normalized, face_size)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                h, w = frame.shape[:2]
                proc = frame
                if w > 1920:
                    scale = 1920.0 / w
                    proc = cv2.resize(frame, (1920, int(h * scale)))

                rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
                mp_image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb)
                timestamp_ms = frame_idx * frame_duration_ms

                try:
                    detection = landmarker.detect_for_video(mp_image, timestamp_ms)
                except Exception:
                    frame_idx += 1
                    continue

                if detection.face_landmarks:
                    # Find largest face (dominant)
                    best_cx, best_cy, best_size = 0, 0, 0
                    for face_lm in detection.face_landmarks:
                        xs = [lm.x for lm in face_lm]
                        ys = [lm.y for lm in face_lm]
                        cx = (min(xs) + max(xs)) / 2.0
                        cy = (min(ys) + max(ys)) / 2.0
                        size = (max(xs) - min(xs)) * (max(ys) - min(ys))
                        if size > best_size:
                            best_cx, best_cy, best_size = cx, cy, size

                    face_track.append((frame_idx, best_cx, best_cy, best_size))

            frame_idx += 1

        cap.release()
        landmarker.close()

        # Find face-change split points
        split_frames = []
        if len(face_track) >= 4:
            for i in range(1, len(face_track)):
                prev_cx, prev_cy = face_track[i - 1][1], face_track[i - 1][2]
                curr_cx, curr_cy = face_track[i][1], face_track[i][2]

                # Horizontal shift in normalized coords (0-1)
                dx = abs(curr_cx - prev_cx)
                dy = abs(curr_cy - prev_cy)
                shift = max(dx, dy)

                if shift >= face_shift_threshold:
                    split_frame = face_track[i][0]
                    # Ensure sub-scenes are at least 1 second
                    if split_frames:
                        if (split_frame - split_frames[-1]) / fps >= 1.0:
                            split_frames.append(split_frame)
                    else:
                        if split_frame / fps >= 1.0:
                            split_frames.append(split_frame)

        if not split_frames:
            new_scenes.append(scene)
            continue

        # Filter: ensure last sub-scene is also at least 1 second
        split_frames = [f for f in split_frames if (total_frames - f) / fps >= 1.0]
        if not split_frames:
            new_scenes.append(scene)
            continue

        re_split_count += 1
        logger.info(f"  Scene {sn}: face change detected, splitting into {len(split_frames) + 1} sub-scenes")

        # Create sub-scene videos and metadata
        boundaries = [0] + split_frames + [total_frames]
        for j in range(len(boundaries) - 1):
            start_frame = boundaries[j]
            end_frame = boundaries[j + 1]
            start_sec = scene["start_seconds"] + (start_frame / fps)
            end_sec = scene["start_seconds"] + (end_frame / fps)
            dur = end_sec - start_sec

            sub_scene_idx = len(new_scenes) + len(short_scenes) + 1
            new_scenes.append({
                "scene": sub_scene_idx,  # placeholder, renumbered later
                "start_seconds": round(start_sec, 3),
                "end_seconds": round(end_sec, 3),
                "duration_seconds": round(dur, 3),
                "_source_scene": sn,
                "_sub_index": j,
            })

        progress(si + 1, len(long_scenes), "Face-based split")

    if re_split_count == 0:
        logger.info("Face-based split: no face changes detected, scenes unchanged.")
        return scenes_data

    # Combine short + new scenes, sort by start time, renumber
    all_scenes = short_scenes + new_scenes
    all_scenes.sort(key=lambda s: s["start_seconds"])

    # Re-extract video/audio clips and renumber
    logger.info(f"Face-based split: re-extracting {len(all_scenes)} scenes...")
    video_path = dirs["inputs"] / scenes_data["video"]
    if not video_path.exists():
        # Try finding the video in inputs
        mp4s = sorted(dirs["inputs"].glob("*.mp4"))
        if mp4s:
            video_path = mp4s[0]

    final_scenes = []
    for i, scene in enumerate(all_scenes):
        new_num = i + 1
        start_sec = scene["start_seconds"]
        end_sec = scene["end_seconds"]
        dur = scene["duration_seconds"]

        # Format timecodes
        start_h = int(start_sec // 3600)
        start_m = int((start_sec % 3600) // 60)
        start_s = start_sec % 60
        start_tc = f"{start_h:02d}:{start_m:02d}:{start_s:06.3f}"

        end_h = int(end_sec // 3600)
        end_m = int((end_sec % 3600) // 60)
        end_s = end_sec % 60
        end_tc = f"{end_h:02d}:{end_m:02d}:{end_s:06.3f}"

        output_video = dirs["videos"] / f"scene_{new_num:03d}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-ss", str(start_sec),
            "-t", str(dur),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            str(output_video),
        ]
        run_ffmpeg(cmd, f"face_split_scene_{new_num:03d}")

        final_scenes.append({
            "scene": new_num,
            "start_time": start_tc,
            "end_time": end_tc,
            "start_seconds": round(start_sec, 3),
            "end_seconds": round(end_sec, 3),
            "duration_seconds": round(dur, 3),
        })
        progress(i + 1, len(all_scenes), "Re-extracting scenes")

    # Clean up old scene files that are beyond the new count
    for old_file in dirs["videos"].glob("scene_*.mp4"):
        num = extract_scene_number(old_file.name)
        if num and num > len(final_scenes):
            old_file.unlink()

    new_scenes_data = {
        "video": scenes_data["video"],
        "total_scenes": len(final_scenes),
        "scenes": final_scenes,
    }

    # Save updated scenes JSON
    json_path = dirs["splitted"] / "scenes.json"
    with open(json_path, "w") as f:
        json.dump(new_scenes_data, f, indent=2)

    logger.info(f"Face-based split: {scenes_data['total_scenes']} -> {len(final_scenes)} scenes")
    return new_scenes_data
