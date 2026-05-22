#!/usr/bin/env python3
"""
Stitch Output Scenes
====================
Normalizes all scene clips to uniform fps/codec, then concatenates into final video.

sync.so returns scenes with variable fps (e.g. 84000/3583 instead of 24/1),
so we must re-encode each scene to exact target fps before stitching.

Usage:
    python3 stitch_output.py
    python3 stitch_output.py --task task_17-05_18-45
    python3 stitch_output.py --fps 24
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def normalize_scene(input_path, output_path, fps):
    """Re-encode a single scene to exact fps, uniform codec settings."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-r", str(fps),
        "-video_track_timescale", str(fps * 1000),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k",
        "-movflags", "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def stitch(task_dir, fps=24):
    output_scenes = task_dir / "outputs" / "outputScenes"
    if not output_scenes.exists():
        print(f"Error: {output_scenes} not found")
        sys.exit(1)

    scene_files = sorted(output_scenes.glob("scene_*.mp4"))
    if not scene_files:
        print("No scene files found in outputScenes/")
        sys.exit(1)

    print(f"Found {len(scene_files)} scenes")

    # Step 1: Normalize all scenes to uniform fps/codec
    norm_dir = output_scenes / "_normalized"
    norm_dir.mkdir(exist_ok=True)

    print(f"Step 1: Normalizing all scenes to {fps}fps...")
    for i, sf in enumerate(scene_files, 1):
        norm_path = norm_dir / sf.name
        # Check if already at correct fps
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(sf)],
            capture_output=True, text=True
        )
        current_fps = probe.stdout.strip()

        if current_fps == f"{fps}/1":
            # Already correct fps, still re-encode for uniform codec params
            shutil.copy2(str(sf), str(norm_path))
            status = "copied"
        else:
            ok = normalize_scene(sf, norm_path, fps)
            status = "normalized" if ok else "FAILED"

        print(f"  [{i}/{len(scene_files)}] {sf.name} ({current_fps}) → {status}")

    # Step 2: Concat normalized scenes
    norm_files = sorted(norm_dir.glob("scene_*.mp4"))
    concat_list = norm_dir / "concat_list.txt"
    with open(concat_list, "w") as f:
        for nf in norm_files:
            f.write(f"file '{nf.resolve()}'\n")

    output_path = task_dir / "outputs" / "final_output.mp4"

    print(f"\nStep 2: Stitching {len(norm_files)} scenes...")
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        "-movflags", "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        # Fallback: re-encode during concat
        print("  Stream copy failed, re-encoding...")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-r", str(fps),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k",
            "-movflags", "+faststart",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

    # Cleanup
    concat_list.unlink(missing_ok=True)
    shutil.rmtree(norm_dir, ignore_errors=True)

    if result.returncode == 0:
        size_mb = output_path.stat().st_size / 1_048_576
        print(f"\nDone! Final output: {output_path} ({size_mb:.1f} MB)")
    else:
        print(f"\nFFmpeg failed:\n{result.stderr}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Stitch output scenes into final video")
    parser.add_argument("--task", default="task_17-05_18-45", help="Task folder name")
    parser.add_argument("--fps", type=int, default=24, help="Output FPS (default: 24)")
    args = parser.parse_args()

    task_dir = Path("tasks") / args.task
    if not task_dir.exists():
        print(f"Error: {task_dir} not found")
        sys.exit(1)

    stitch(task_dir, args.fps)


if __name__ == "__main__":
    main()
