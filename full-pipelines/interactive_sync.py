#!/usr/bin/env python3
"""
Interactive Scene-by-Scene Lip-Sync
====================================
Goes through each scene one at a time, asks:
  1. Skip this scene or process it?
  2. Which audio file? (Hindi-AUD / VASUKI)
  3. sync.so parameters (reasoning, obstruction, active speaker)

Then uploads to S3, sends to sync.so, downloads result before moving to next scene.

Usage:
    python3 interactive_sync.py
    python3 interactive_sync.py --start 5        # start from scene 5
    python3 interactive_sync.py --scenes 8,11,14  # only specific scenes
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import logger
from sync_api.s3 import _ensure_s3_folder, _s3_upload_file
from sync_api.client import sync_create_job, sync_poll, sync_download

TASK_DIR = Path("tasks/task_17-05_18-45")
SCENES_JSON = TASK_DIR / "outputs/SplittedScenes/scenes.json"
VIDEOS_DIR = TASK_DIR / "outputs/SplittedScenes/videos"
AUDIO_DIRS = {
    "1": ("Hindi-AUD", TASK_DIR / "outputs/SplittedScenes/audio/hindi_aud"),
    "2": ("VASUKI", TASK_DIR / "outputs/SplittedScenes/audio/vasuki"),
}
OUTPUT_DIR = TASK_DIR / "outputs/outputScenes"
PROGRESS_FILE = TASK_DIR / "outputs/sync_progress.json"

# Pre-computed VAD results: speech duration per scene per audio
VAD_RESULTS = {
    "hindi_aud": {1:1.91,2:0.76,3:0,4:0,5:0.63,6:0.77,7:0.89,8:0.51,9:0,10:0,11:2.01,12:0,13:0,14:1.62,15:0,16:0,17:0,18:5.58,19:0.38,20:1.01,21:0,22:4.84,23:0,24:0,25:0,26:0,27:0,28:0,29:0,30:0.64,31:0.46,32:0.38,33:0,34:0,35:1.08,36:0,37:17.72,38:0},
    "vasuki": {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:1.18,9:1.76,10:1.18,11:0.77,12:1.33,13:3.64,14:0.94,15:1.65,16:1.25,17:1.27,18:0,19:2.13,20:0.64,21:1.99,22:0.57,23:1.25,24:1.80,25:2.18,26:0.96,27:1.43,28:1.83,29:2.92,30:1.21,31:0.38,32:0,33:1.47,34:7.53,35:3.61,36:3.23,37:0.48,38:0},
}


def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"processed": {}, "skipped": []}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def get_auto_suggestion(scene_num):
    """Suggest audio based on VAD results."""
    h = VAD_RESULTS["hindi_aud"].get(scene_num, 0)
    v = VAD_RESULTS["vasuki"].get(scene_num, 0)
    if h <= 0.3 and v <= 0.3:
        return "silence", h, v
    elif h > 0.3 and v <= 0.3:
        return "hindi_aud", h, v
    elif v > 0.3 and h <= 0.3:
        return "vasuki", h, v
    else:
        return "both", h, v


def prompt_scene(scene, progress_data):
    """Interactive prompt for a single scene. Returns action dict or None to skip."""
    sn = scene["scene"]
    dur = scene["duration_seconds"]
    start = scene["start_time"]
    end = scene["end_time"]

    suggestion, h_speech, v_speech = get_auto_suggestion(sn)

    print(f"\n{'='*60}")
    print(f"  SCENE {sn:>2} / 38")
    print(f"  Time: {start} → {end}  ({dur:.1f}s)")
    print(f"  Video: scene_{sn:03d}.mp4")
    print(f"{'='*60}")
    print(f"  Speech detected:")
    print(f"    [1] Hindi-AUD : {h_speech:.2f}s")
    print(f"    [2] VASUKI    : {v_speech:.2f}s")

    if suggestion == "silence":
        print(f"\n  >> Auto-suggestion: SKIP (no speech detected)")
    elif suggestion == "hindi_aud":
        print(f"\n  >> Auto-suggestion: Hindi-AUD")
    elif suggestion == "vasuki":
        print(f"\n  >> Auto-suggestion: VASUKI")
    else:
        print(f"\n  >> BOTH have speech — you must choose")

    # Already processed?
    sn_str = str(sn)
    if sn_str in progress_data["processed"]:
        prev = progress_data["processed"][sn_str]
        print(f"\n  ⚠ Already processed with: {prev['audio']}")
        redo = input("  Re-process? (y/N): ").strip().lower()
        if redo != "y":
            return None
    if sn in progress_data["skipped"]:
        print(f"\n  ⚠ Previously skipped")
        redo = input("  Process now? (y/N): ").strip().lower()
        if redo != "y":
            return None

    # Ask: process or skip?
    print(f"\n  Options:")
    print(f"    ENTER = accept suggestion, s = skip, 1 = Hindi-AUD, 2 = VASUKI, q = quit")
    choice = input("  > ").strip().lower()

    if choice == "q":
        return "QUIT"
    if choice == "s":
        return "SKIP"

    # Determine audio
    if choice == "1":
        audio_key = "1"
    elif choice == "2":
        audio_key = "2"
    elif choice == "" and suggestion in ("hindi_aud", "vasuki"):
        audio_key = "1" if suggestion == "hindi_aud" else "2"
    elif choice == "" and suggestion == "silence":
        return "SKIP"
    else:
        # For "both" or invalid input, ask explicitly
        while True:
            audio_key = input("  Pick audio [1=Hindi-AUD, 2=VASUKI]: ").strip()
            if audio_key in ("1", "2"):
                break

    audio_label, audio_dir = AUDIO_DIRS[audio_key]
    print(f"\n  Audio: {audio_label}")

    # sync.so parameters
    print(f"\n  sync.so parameters (press ENTER for defaults):")

    r = input("    reasoning? (y/N): ").strip().lower()
    reasoning = r == "y"

    o = input("    obstruction detection? (y/N): ").strip().lower()
    obstruction = o == "y"

    a = input("    active speaker detection? (y/N): ").strip().lower()
    active_speaker = a == "y"

    print(f"\n  Summary:")
    print(f"    Scene {sn} → {audio_label}")
    print(f"    reasoning={reasoning}, obstruction={obstruction}, active_speaker={active_speaker}")
    confirm = input("  Confirm & send to sync.so? (Y/n): ").strip().lower()
    if confirm == "n":
        return "SKIP"

    return {
        "scene": sn,
        "audio_key": audio_key,
        "audio_label": audio_label,
        "audio_dir": str(audio_dir),
        "reasoning": reasoning,
        "obstruction": obstruction,
        "active_speaker": active_speaker,
    }


def process_scene(action, api_key, s3_prefix):
    """Upload to S3, send to sync.so, download result."""
    sn = action["scene"]
    audio_dir = Path(action["audio_dir"])

    video_path = VIDEOS_DIR / f"scene_{sn:03d}.mp4"
    audio_path = audio_dir / f"scene_{sn:03d}.wav"
    output_path = OUTPUT_DIR / f"scene_{sn:03d}.mp4"

    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        return False
    if not audio_path.exists():
        logger.error(f"Audio not found: {audio_path}")
        return False

    # Upload to S3
    print(f"  Uploading to S3...")
    video_url = _s3_upload_file(video_path, f"{s3_prefix}scenes_video/scene_{sn:03d}.mp4")
    audio_url = _s3_upload_file(audio_path, f"{s3_prefix}scenes_audio/scene_{sn:03d}_{action['audio_label']}.wav")
    print(f"  Uploaded. Sending to sync.so...")

    # Create job
    job = sync_create_job(
        video_url, audio_url, api_key,
        reasoning=action["reasoning"],
        detect_obstructions=action["obstruction"],
        active_speaker=action["active_speaker"],
        output_name=f"scene_{sn:03d}_synced",
    )
    job_id = job["id"]
    print(f"  Job created: {job_id}")
    print(f"  Polling for result (this may take a few minutes)...")

    # Poll
    result = sync_poll(job_id, api_key)
    status = result.get("status")

    if status == "COMPLETED":
        output_url = result.get("outputUrl") or result.get("output_url")
        if output_url:
            sync_download(output_url, str(output_path))
            print(f"  ✓ Done! Saved to: {output_path}")
            return True
        else:
            logger.error(f"  No output URL in response")
            return False
    else:
        error = result.get("error", "N/A")
        logger.error(f"  Failed — status: {status}, error: {error}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Interactive scene-by-scene lip-sync")
    parser.add_argument("--start", type=int, default=1, help="Start from scene N")
    parser.add_argument("--scenes", type=str, help="Only process these scenes (comma-separated)")
    args = parser.parse_args()

    # Load scenes
    with open(SCENES_JSON) as f:
        scenes_data = json.load(f)

    # Filter scenes
    scenes = scenes_data["scenes"]
    if args.scenes:
        scene_nums = set(int(x) for x in args.scenes.split(","))
        scenes = [s for s in scenes if s["scene"] in scene_nums]
    else:
        scenes = [s for s in scenes if s["scene"] >= args.start]

    # Check env
    api_key = os.environ.get("SYNC_API_KEY", "")
    s3_user = os.environ.get("PIPELINE_S3_USER", "")
    if not api_key:
        api_key = input("Enter sync.so API key: ").strip()
    if not s3_user:
        s3_user = input("Enter S3 user folder: ").strip()

    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    s3_prefix = _ensure_s3_folder(s3_user, TASK_DIR.name)
    progress_data = load_progress()

    print(f"\n  Task: {TASK_DIR.name}")
    print(f"  Scenes to review: {len(scenes)}")
    print(f"  Already processed: {len(progress_data['processed'])}")
    print(f"  Already skipped: {len(progress_data['skipped'])}")

    for scene in scenes:
        sn = scene["scene"]
        result = prompt_scene(scene, progress_data)

        if result == "QUIT":
            print("\nQuitting. Progress saved.")
            save_progress(progress_data)
            return
        elif result == "SKIP" or result is None:
            if sn not in progress_data["skipped"] and str(sn) not in progress_data["processed"]:
                progress_data["skipped"].append(sn)
                # Copy original video for skipped scenes
                src = VIDEOS_DIR / f"scene_{sn:03d}.mp4"
                dst = OUTPUT_DIR / f"scene_{sn:03d}.mp4"
                if src.exists() and not dst.exists():
                    shutil.copy2(str(src), str(dst))
                    print(f"  Copied original video for scene {sn}")
            save_progress(progress_data)
            continue
        else:
            # Process
            success = process_scene(result, api_key, s3_prefix)
            if success:
                progress_data["processed"][str(sn)] = {
                    "audio": result["audio_label"],
                    "reasoning": result["reasoning"],
                    "obstruction": result["obstruction"],
                    "active_speaker": result["active_speaker"],
                }
            else:
                # Copy original as fallback
                src = VIDEOS_DIR / f"scene_{sn:03d}.mp4"
                dst = OUTPUT_DIR / f"scene_{sn:03d}.mp4"
                if src.exists():
                    shutil.copy2(str(src), str(dst))
                print(f"  Fallback: copied original video")

            save_progress(progress_data)

    print(f"\n{'='*60}")
    print(f"  All done!")
    print(f"  Processed: {len(progress_data['processed'])} scenes")
    print(f"  Skipped: {len(progress_data['skipped'])} scenes")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
