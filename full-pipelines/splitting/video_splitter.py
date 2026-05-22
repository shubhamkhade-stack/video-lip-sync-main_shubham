"""
Video Splitter
==============
Scene detection using ContentDetector + AdaptiveDetector, then FFmpeg extraction.
"""

import json
import sys
from pathlib import Path

from config import logger
from utils import run_ffmpeg, get_video_duration, progress


def split_video(video_path, dirs):
    """Detect and split video into scenes using ContentDetector + AdaptiveDetector."""
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector, AdaptiveDetector

    logger.info("Detecting scenes (ContentDetector + AdaptiveDetector)...")

    # Pass 1: ContentDetector (hard cuts)
    video = open_video(video_path)
    sm_content = SceneManager()
    sm_content.add_detector(ContentDetector(threshold=27.0, min_scene_len=15))
    sm_content.detect_scenes(video, show_progress=True)
    content_scenes = sm_content.get_scene_list()
    logger.info(f"  ContentDetector found {len(content_scenes)} scenes.")

    # Pass 2: AdaptiveDetector (gradual transitions, soft cuts)
    video2 = open_video(video_path)
    sm_adaptive = SceneManager()
    sm_adaptive.add_detector(AdaptiveDetector(
        adaptive_threshold=3.0,
        min_scene_len=15,
        min_content_val=15.0,
    ))
    sm_adaptive.detect_scenes(video2, show_progress=True)
    adaptive_scenes = sm_adaptive.get_scene_list()
    logger.info(f"  AdaptiveDetector found {len(adaptive_scenes)} scenes.")

    # Merge: collect all unique cut points from both detectors
    cut_points = set()
    for scene_list_part in [content_scenes, adaptive_scenes]:
        for start, end in scene_list_part:
            cut_points.add(start.get_frames())
            cut_points.add(end.get_frames())
    cut_points = sorted(cut_points)

    # Rebuild scene list from merged cut points (pairs of consecutive points)
    from scenedetect.frame_timecode import FrameTimecode
    fps = open_video(video_path).frame_rate
    scene_list = []
    for i in range(len(cut_points) - 1):
        start_tc = FrameTimecode(cut_points[i], fps=fps)
        end_tc = FrameTimecode(cut_points[i + 1], fps=fps)
        # Skip very short segments (less than 15 frames)
        if (cut_points[i + 1] - cut_points[i]) >= 15:
            scene_list.append((start_tc, end_tc))

    logger.info(f"  Merged: {len(scene_list)} scenes after combining both detectors.")

    if not scene_list:
        logger.error("No scenes detected. Try a different video.")
        sys.exit(1)

    total = len(scene_list)
    logger.info(f"Detected {total} scenes. Splitting video...")

    base_name = Path(video_path).stem
    scenes_json = []
    video_duration = get_video_duration(video_path)

    for i, (start, end) in enumerate(scene_list):
        scene_num = i + 1
        output_path = dirs["videos"] / f"scene_{scene_num:03d}.mp4"

        start_sec = start.get_seconds()
        end_sec = end.get_seconds()
        dur = end_sec - start_sec

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-ss", str(start_sec),
            "-t", str(dur),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            str(output_path),
        ]
        run_ffmpeg(cmd, f"scene_{scene_num:03d}")

        scenes_json.append({
            "scene": scene_num,
            "start_time": start.get_timecode(),
            "end_time": end.get_timecode(),
            "start_seconds": round(start_sec, 3),
            "end_seconds": round(end_sec, 3),
            "duration_seconds": round(dur, 3),
        })

        processed_time = end_sec
        progress(i + 1, total, "Splitting video",
                 f"{processed_time:.0f}s/{video_duration:.0f}s")

    # Save scenes JSON
    json_path = dirs["splitted"] / "scenes.json"
    scenes_data = {
        "video": Path(video_path).name,
        "total_scenes": total,
        "scenes": scenes_json,
    }
    with open(json_path, "w") as f:
        json.dump(scenes_data, f, indent=2)

    logger.info(f"Scenes JSON saved: {json_path}")
    return scenes_data, scene_list
