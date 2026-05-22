#!/usr/bin/env python3
"""
Face Tracking Pipeline
======================
Standalone pipeline to track and identify faces across video scenes.

Creates a task folder, loads scene videos from inputs/,
runs InsightFace detection + tracking, matches identities across scenes,
and outputs results as JSON + annotated frames + console summary table.

Usage:
    python3 FaceTrackPipeline.py                          # interactive (place files in inputs/)
    python3 FaceTrackPipeline.py --input /path/to/scenes/  # auto-copy scene videos
    python3 FaceTrackPipeline.py --resume                  # resume previous task
    python3 FaceTrackPipeline.py --resume track_04-06_14-30 # resume specific task
"""

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2
import numpy as np

from config import logger

# ── Constants ────────────────────────────────────────────────────────────────
TASKS_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "tracking_tasks"
DEFAULT_SAMPLE_FPS = 5
CROSS_SCENE_SIM_THRESHOLD = 0.5


# ── Task Folder ──────────────────────────────────────────────────────────────

def setup_task_folder():
    """Create a new tracking task folder."""
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    folder_name = f"track_{now.strftime('%d-%m_%H-%M')}"
    task_dir = TASKS_DIR / folder_name

    dirs = {
        "root": task_dir,
        "inputs": task_dir / "inputs",
        "outputs": task_dir / "outputs",
        "frames": task_dir / "outputs" / "annotated_frames",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    logger.info(f"Created tracking task: {task_dir}")
    return dirs


def _rebuild_dirs(task_dir):
    task_dir = Path(task_dir)
    dirs = {
        "root": task_dir,
        "inputs": task_dir / "inputs",
        "outputs": task_dir / "outputs",
        "frames": task_dir / "outputs" / "annotated_frames",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _list_tasks():
    if not TASKS_DIR.exists():
        return []
    tasks = []
    for entry in sorted(TASKS_DIR.iterdir(), reverse=True):
        if entry.is_dir() and entry.name.startswith("track_"):
            state_path = entry / "state.json"
            state = {}
            if state_path.exists():
                with open(state_path) as f:
                    state = json.load(f)
            tasks.append((entry.name, state))
    return tasks


def _save_state(dirs, phase, status="running", **extra):
    state_path = dirs["root"] / "state.json"
    state = {}
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)

    completed = state.get("completed_phases", [])
    if phase not in completed:
        completed.append(phase)

    state["completed_phases"] = completed
    state["current_phase"] = phase
    state["status"] = status
    state["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state.update(extra)

    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


# ── Input Handling ───────────────────────────────────────────────────────────

def load_inputs(dirs, input_path=None):
    """Copy scene videos into inputs/ or wait for user to place them."""
    inputs_dir = dirs["inputs"]

    if input_path:
        src = Path(input_path)
        if src.is_dir():
            mp4s = sorted(src.glob("*.mp4"))
            if not mp4s:
                logger.error(f"No .mp4 files found in {src}")
                return []
            for f in mp4s:
                shutil.copy2(str(f), str(inputs_dir / f.name))
            logger.info(f"Copied {len(mp4s)} scene videos from {src}")
        elif src.is_file() and src.suffix == ".mp4":
            shutil.copy2(str(src), str(inputs_dir / src.name))
            logger.info(f"Copied 1 video: {src.name}")
        else:
            logger.error(f"Invalid input: {src}")
            return []
    else:
        print(f"\n{'='*60}")
        print(f"  Place scene .mp4 files in:")
        print(f"  {inputs_dir}")
        print(f"{'='*60}")
        input("\n  Press ENTER when files are ready...")

    videos = sorted(inputs_dir.glob("*.mp4"))
    if not videos:
        logger.error("No .mp4 files found in inputs/")
        return []

    logger.info(f"Found {len(videos)} scene video(s)")
    return videos


# ── Annotated Frames ─────────────────────────────────────────────────────────

TRACK_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (128, 0, 255), (0, 128, 255),
]


def save_annotated_frames(video_path, track_result, output_dir, face_app, scene_label=""):
    """Save first, middle, and last sampled frames with bounding boxes + track IDs.

    Uses tracked data to find which frames have detections, then draws
    bounding boxes from stored track data + verifies with live detection.
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return

    # Build frame->[(tid, bbox)] lookup from tracked data
    frame_to_tracks = {}
    for tid, det_list in track_result["tracks"].items():
        for det in det_list:
            fn = det["frame"]
            if fn not in frame_to_tracks:
                frame_to_tracks[fn] = []
            frame_to_tracks[fn].append((tid, det["bbox"]))

    # Pick 3 target frames: start, middle, end of video
    target_frames = [0, total_frames // 2, max(0, total_frames - 2)]

    for fi, target_frame in enumerate(target_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            continue

        # Find closest tracked frame to this target
        tracked_frames = sorted(frame_to_tracks.keys())
        if tracked_frames:
            closest = min(tracked_frames, key=lambda f: abs(f - target_frame))
            # If closest tracked frame is within 15 frames, use its track data
            if abs(closest - target_frame) <= 15:
                # Seek to the tracked frame instead for accurate bbox
                cap.set(cv2.CAP_PROP_POS_FRAMES, closest)
                ret2, frame = cap.read()
                if ret2:
                    target_frame = closest

                for tid, bbox in frame_to_tracks.get(closest, []):
                    x, y, w, h = bbox
                    color = TRACK_COLORS[int(tid) % len(TRACK_COLORS)]

                    # Draw bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

                    # Draw label with background
                    label = f"ID:{tid}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                    cv2.rectangle(frame, (x, y - label_size[1] - 12), (x + label_size[0] + 8, y), color, -1)
                    cv2.putText(frame, label, (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            else:
                # No tracked data near this frame — run live detection
                _annotate_live(frame, face_app, track_result)
        else:
            _annotate_live(frame, face_app, track_result)

        # Scene label on top-left
        cv2.putText(frame, f"{scene_label}  frame:{target_frame}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        out_path = output_dir / f"{scene_label}_frame{fi+1}.jpg"
        cv2.imwrite(str(out_path), frame)

    cap.release()


def _annotate_live(frame, face_app, track_result):
    """Fallback: run live detection and match to stored embeddings."""
    faces = face_app.get(frame)
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        emb = face.normed_embedding

        best_tid = "?"
        best_sim = 0.3  # minimum threshold
        for tid, stored_emb in track_result.get("embeddings", {}).items():
            sim = float(np.dot(emb, np.array(stored_emb)))
            if sim > best_sim:
                best_sim = sim
                best_tid = tid

        color = TRACK_COLORS[int(best_tid) % len(TRACK_COLORS)] if best_tid != "?" else (200, 200, 200)
        label = f"ID:{best_tid}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 12), (x1 + label_size[0] + 8, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)


# ── Console Table ────────────────────────────────────────────────────────────

def print_summary_table(all_results, global_ids=None):
    """Print a formatted summary table to console."""
    print(f"\n{'='*80}")
    print(f"  FACE TRACKING RESULTS")
    print(f"{'='*80}")

    # Per-scene table
    header = f"  {'Scene':<20} {'Faces':>6} {'Frames':>8} {'Dominant Track':>15} {'Detections':>11}"
    print(f"\n{header}")
    print(f"  {'─'*62}")

    total_faces = 0
    for video_name, result in sorted(all_results.items()):
        n_faces = result["unique_faces"]
        n_frames = result["total_frames_sampled"]
        total_faces += n_faces

        # Find dominant track (most detections)
        dom_tid = "-"
        dom_count = 0
        for tid, frames in result["tracks"].items():
            if len(frames) > dom_count:
                dom_count = len(frames)
                dom_tid = f"ID:{tid}"

        print(f"  {video_name:<20} {n_faces:>6} {n_frames:>8} {dom_tid:>15} {dom_count:>11}")

    print(f"  {'─'*62}")
    print(f"  {'TOTAL':<20} {total_faces:>6}")

    # Cross-scene identity table
    if global_ids:
        print(f"\n  CROSS-SCENE IDENTITY MATCHING")
        print(f"  {'─'*62}")
        print(f"  {'Global ID':<12} {'Appears In':<50}")
        print(f"  {'─'*62}")
        for gid, members in sorted(global_ids.items()):
            scenes = [f"{scene}:Track{tid}" for scene, tid in members]
            print(f"  Person {gid:<6} {', '.join(scenes):<50}")
        print(f"  {'─'*62}")
        print(f"  {len(global_ids)} unique person(s) identified across all scenes")

    print(f"\n{'='*80}\n")


# ── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Face Tracking Pipeline")
    parser.add_argument("--input", "-i", help="Path to scene videos folder or single .mp4")
    parser.add_argument("--fps", type=int, default=DEFAULT_SAMPLE_FPS, help=f"Sample FPS (default: {DEFAULT_SAMPLE_FPS})")
    parser.add_argument("--resume", "-r", nargs="?", const=True, default=None,
                        help="Resume a previous task")
    args = parser.parse_args()

    sample_fps = args.fps

    print(f"\n{'='*60}")
    print(f"  Face Tracking Pipeline (InsightFace)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # ── Resume or new ────────────────────────────────────────────────────
    resuming = False
    if args.resume:
        tasks = _list_tasks()
        if args.resume is True:
            if tasks:
                print("  Previous tasks:")
                for i, (name, state) in enumerate(tasks, 1):
                    status = state.get("status", "?")
                    phase = state.get("current_phase", "?")
                    print(f"    {i}. {name}  [{status} at: {phase}]")
                choice = input("\n  Enter number to resume, or ENTER for new: ").strip()
                if choice.isdigit() and 0 < int(choice) <= len(tasks):
                    dirs = _rebuild_dirs(TASKS_DIR / tasks[int(choice)-1][0])
                    resuming = True
            if not resuming:
                print("  Starting new task.\n")
        else:
            dirs = _rebuild_dirs(TASKS_DIR / args.resume)
            resuming = True

    if not resuming:
        dirs = setup_task_folder()

    # ── Phase 1: Load inputs ─────────────────────────────────────────────
    videos = sorted(dirs["inputs"].glob("*.mp4"))
    if not videos:
        videos = load_inputs(dirs, args.input)
        if not videos:
            return
    else:
        logger.info(f"Found {len(videos)} video(s) in inputs/")

    _save_state(dirs, "inputs_loaded", scene_count=len(videos))

    # ── Phase 2: Init InsightFace ────────────────────────────────────────
    logger.info("Loading InsightFace model (SCRFD + ArcFace)...")
    t0 = time.time()
    from tracking.face_tracker import _build_face_app, FaceTracker, match_faces_across_scenes
    face_app = _build_face_app()
    logger.info(f"  Model loaded in {time.time()-t0:.1f}s")

    _save_state(dirs, "model_loaded")

    # ── Phase 3: Track faces per scene ───────────────────────────────────
    logger.info(f"\nTracking faces across {len(videos)} scene(s) at {sample_fps} fps...\n")
    from tracking.face_tracker import track_faces_in_video

    all_results = {}  # video_name -> tracking result
    total_start = time.time()

    for vi, video_path in enumerate(videos):
        vname = video_path.stem
        logger.info(f"  [{vi+1}/{len(videos)}] {vname}")
        t1 = time.time()

        result = track_faces_in_video(str(video_path), face_app=face_app, sample_fps=sample_fps)
        all_results[vname] = result

        elapsed = time.time() - t1
        logger.info(f"    -> {result['unique_faces']} face(s), {result['total_frames_sampled']} frames, {elapsed:.1f}s")

        # Save annotated frames
        save_annotated_frames(
            video_path, result, dirs["frames"], face_app, scene_label=vname
        )

    _save_state(dirs, "tracking_done")
    logger.info(f"\nAll scenes tracked in {time.time()-total_start:.1f}s")

    # ── Phase 4: Cross-scene identity matching ───────────────────────────
    logger.info("Matching faces across scenes...")
    global_ids = match_faces_across_scenes(
        {vname: r for vname, r in all_results.items()},
        sim_threshold=CROSS_SCENE_SIM_THRESHOLD
    )

    _save_state(dirs, "matching_done")

    # ── Phase 5: Save outputs ────────────────────────────────────────────

    # 5a. Detailed JSON
    json_output = {
        "task": dirs["root"].name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "settings": {"sample_fps": sample_fps, "cross_scene_threshold": CROSS_SCENE_SIM_THRESHOLD},
        "scenes": {},
        "cross_scene_identities": {},
    }

    for vname, result in all_results.items():
        json_output["scenes"][vname] = {
            "unique_faces": result["unique_faces"],
            "total_frames_sampled": result["total_frames_sampled"],
            "tracks": {
                str(tid): {
                    "detections": len(frames),
                    "first_frame": frames[0]["frame"] if frames else None,
                    "last_frame": frames[-1]["frame"] if frames else None,
                    "avg_bbox": {
                        "x": int(np.mean([f["bbox"][0] for f in frames])),
                        "y": int(np.mean([f["bbox"][1] for f in frames])),
                        "w": int(np.mean([f["bbox"][2] for f in frames])),
                        "h": int(np.mean([f["bbox"][3] for f in frames])),
                    }
                }
                for tid, frames in result["tracks"].items()
            },
        }

    for gid, members in global_ids.items():
        json_output["cross_scene_identities"][f"person_{gid}"] = [
            {"scene": scene, "track_id": tid} for scene, tid in members
        ]

    json_path = dirs["outputs"] / "FaceTrackingResults.json"
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)
    logger.info(f"Saved: {json_path}")

    # 5b. Console table
    print_summary_table(all_results, global_ids)

    # 5c. Summary
    _save_state(dirs, "done", status="done")

    print(f"  Outputs saved to: {dirs['outputs']}")
    print(f"    - FaceTrackingResults.json   (detailed per-scene data)")
    print(f"    - annotated_frames/          ({len(videos)*3} annotated images)")
    print()


if __name__ == "__main__":
    main()
