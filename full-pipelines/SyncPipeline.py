#!/usr/bin/env python3
"""
LipSync Pipeline Orchestrator
==============================
End-to-end automated lip-sync dubbing pipeline.

Creates a task folder, splits video/audio, analyzes scenes,
processes through sync.so API, and stitches the final output.

Usage:
    python3 SyncPipeline.py                                    # interactive (new or resume)
    python3 SyncPipeline.py --video input.mp4 --audio dubbed.wav  # new task
    python3 SyncPipeline.py --resume                           # pick from list
    python3 SyncPipeline.py --resume task_02-04_13-08          # resume specific task
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from config import S3_PREFIX, logger
from utils import (
    phase_timer, setup_task_folder, wait_for_inputs, check_api_key, check_s3_access,
    save_state, load_state, rebuild_dirs, list_tasks, TASKS_DIR,
)

from splitting.video_splitter import split_video
from splitting.face_splitter import face_based_split
from splitting.audio_splitter import convert_audio_16khz, split_audio

from analysis.dialogue import detect_dialogue, load_dialogue_results
from analysis.complexity import analyze_complexity_and_obstruction, load_complexity_results
from analysis.scene_analysis import generate_scene_analysis

from sync_api.processor import process_scenes_with_api

from output.stitcher import stitch_scenes
from output.report import print_report


def _show_task_menu():
    """Show previous tasks and let user choose to resume or start new.
    Returns (task_folder_name, state) to resume, or (None, None) for new task."""
    tasks = list_tasks()
    resumable = [(name, state) for name, state in tasks
                 if state.get("status") not in ("done", "unknown")]

    if not resumable:
        return None, None

    print(f"\n  Previous tasks found:")
    for i, (name, state) in enumerate(resumable, 1):
        video = state.get("video", "?")
        phase = state.get("current_phase", "?")
        status = state.get("status", "?")
        print(f"    {i}. {name}  [{status} at: {phase}]  {video}")

    print()
    choice = input("  Enter number to resume, or press ENTER for new task: ").strip()

    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(resumable):
            return resumable[idx]

    return None, None


def _load_scenes_data(dirs):
    """Load scenes.json from a task folder."""
    scenes_path = dirs["splitted"] / "scenes.json"
    if not scenes_path.exists():
        return None
    with open(scenes_path) as f:
        return json.load(f)


def _load_analysis(dirs):
    """Load SceneAnalysis.json from a task folder."""
    analysis_path = dirs["outputs"] / "SceneAnalysis.json"
    if not analysis_path.exists():
        return None
    with open(analysis_path) as f:
        data = json.load(f)
    return data.get("scenes", [])


def _get_input_paths(dirs):
    """Find video and audio files in the inputs folder."""
    mp4s = sorted(dirs["inputs"].glob("*.mp4"))
    wavs = sorted((dirs["inputs"]).glob("*.wav"))
    video_path = str(mp4s[0]) if mp4s else None
    audio_path = str(wavs[0]) if wavs else None
    return video_path, audio_path


def _phase_done(state, phase):
    """Check if a phase is already completed."""
    return phase in state.get("completed_phases", [])


def main():
    parser = argparse.ArgumentParser(description="LipSync Pipeline Orchestrator")
    parser.add_argument("--video", "-v", help="Path to input video file")
    parser.add_argument("--audio", "-a", help="Path to dubbed audio file")
    parser.add_argument("--api-key", "-k", help="sync.so API key (or set SYNC_API_KEY env var)")
    parser.add_argument("--s3-user", help="S3 folder name for this user (or set PIPELINE_S3_USER env var)")
    parser.add_argument("--resume", "-r", nargs="?", const=True, default=None,
                        help="Resume a previous task (optionally specify task folder name)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  LipSync Pipeline")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # ── Decide: new task or resume ────────────────────────────────────────
    resuming = False
    state = {}

    if args.resume:
        if args.resume is True:
            # --resume with no value: show menu
            task_name, state = _show_task_menu()
            if task_name is None:
                print("  No resumable tasks found. Starting new task.\n")
            else:
                resuming = True
                dirs = rebuild_dirs(TASKS_DIR / task_name)
        else:
            # --resume task_name
            task_dir = TASKS_DIR / args.resume
            state = load_state(task_dir)
            if state is None:
                logger.error(f"No state.json found in {task_dir}")
                return
            resuming = True
            dirs = rebuild_dirs(task_dir)
    elif not args.video:
        # No --video and no --resume: offer to resume if tasks exist
        task_name, state = _show_task_menu()
        if task_name is not None:
            resuming = True
            dirs = rebuild_dirs(TASKS_DIR / task_name)

    if resuming:
        logger.info(f"Resuming task: {dirs['root'].name}")
        logger.info(f"  Completed phases: {', '.join(state.get('completed_phases', []))}")
        logger.info(f"  Resuming from: {state.get('current_phase', '?')}")
        save_state(dirs, state["completed_phases"][-1] if state.get("completed_phases") else "setup", status="running")

    # ── Phase 1: Setup ────────────────────────────────────────────────────
    if not resuming or not _phase_done(state, "setup"):
        with phase_timer("1. Setup"):
            if not resuming:
                dirs = setup_task_folder()
            video_path, audio_path = wait_for_inputs(dirs, args.video, args.audio)
            save_state(dirs, "setup", video=Path(video_path).name, audio=Path(audio_path).name)
    else:
        video_path, audio_path = _get_input_paths(dirs)
        if not video_path or not audio_path:
            logger.error("Input files not found in task folder.")
            return
        logger.info(f"  Setup: skipped (already done)")

    # ── Phase 2: Video Split ──────────────────────────────────────────────
    if not _phase_done(state, "video_split"):
        with phase_timer("2. Video splitting (Content + Adaptive)"):
            scenes_data, scene_list = split_video(video_path, dirs)
            save_state(dirs, "video_split")
    else:
        scenes_data = _load_scenes_data(dirs)
        if scenes_data is None:
            logger.error("scenes.json not found — cannot resume. Re-run without --resume.")
            return
        logger.info(f"  Video splitting: skipped ({scenes_data['total_scenes']} scenes)")

    # ── Phase 2b: Face Split ──────────────────────────────────────────────
    if not _phase_done(state, "face_split"):
        with phase_timer("2b. Face-based split refinement"):
            scenes_data = face_based_split(scenes_data, dirs)
            save_state(dirs, "face_split")
    else:
        # Reload in case face split changed the scene count
        scenes_data = _load_scenes_data(dirs) or scenes_data
        logger.info(f"  Face-based split: skipped")

    # ── Phase 3: Audio Split ──────────────────────────────────────────────
    if not _phase_done(state, "audio_split"):
        with phase_timer("3. Audio conversion + splitting"):
            audio_16khz = convert_audio_16khz(audio_path, dirs)
            split_audio(audio_16khz, scenes_data, dirs)
            save_state(dirs, "audio_split")
    else:
        logger.info(f"  Audio splitting: skipped")

    # ── Phase 4: Dialogue Detection ───────────────────────────────────────
    if not _phase_done(state, "dialogue"):
        with phase_timer("4. Dialogue detection"):
            dialogue_set = detect_dialogue(dirs["audio"], scenes_data, output_dir=dirs["outputs"])
            save_state(dirs, "dialogue")
    else:
        dialogue_set = load_dialogue_results(dirs["outputs"])
        if dialogue_set is None:
            logger.warning("Dialogue results not found — re-running detection.")
            with phase_timer("4. Dialogue detection"):
                dialogue_set = detect_dialogue(dirs["audio"], scenes_data, output_dir=dirs["outputs"])
                save_state(dirs, "dialogue")
        else:
            logger.info(f"  Dialogue detection: skipped ({len(dialogue_set)} scenes with dialogue)")

    # ── Phase 5: Complexity + Obstruction ─────────────────────────────────
    if not _phase_done(state, "complexity"):
        with phase_timer("5. Scene analysis (complexity + obstruction + YOLO)"):
            complexity, obstruction = analyze_complexity_and_obstruction(
                dirs["videos"], dialogue_set, scenes_data, output_dir=dirs["outputs"]
            )
            save_state(dirs, "complexity")
    else:
        loaded = load_complexity_results(dirs["outputs"])
        if loaded is None:
            logger.warning("Complexity results not found — re-running analysis.")
            with phase_timer("5. Scene analysis (complexity + obstruction + YOLO)"):
                complexity, obstruction = analyze_complexity_and_obstruction(
                    dirs["videos"], dialogue_set, scenes_data, output_dir=dirs["outputs"]
                )
                save_state(dirs, "complexity")
        else:
            complexity, obstruction = loaded
            logger.info(f"  Scene analysis: skipped")

    # ── Phase 6: Scene Analysis JSON ──────────────────────────────────────
    if not _phase_done(state, "scene_analysis"):
        with phase_timer("6. Generate SceneAnalysis.json"):
            analysis_path = dirs["outputs"] / "SceneAnalysis.json"
            analysis = generate_scene_analysis(
                scenes_data, dialogue_set, complexity, obstruction, analysis_path
            )
            save_state(dirs, "scene_analysis")
    else:
        analysis = _load_analysis(dirs)
        if analysis is None:
            logger.warning("SceneAnalysis.json not found — re-generating.")
            with phase_timer("6. Generate SceneAnalysis.json"):
                analysis_path = dirs["outputs"] / "SceneAnalysis.json"
                analysis = generate_scene_analysis(
                    scenes_data, dialogue_set, complexity, obstruction, analysis_path
                )
                save_state(dirs, "scene_analysis")
        else:
            logger.info(f"  SceneAnalysis.json: skipped")

    # ── Phase 7: sync.so Processing ───────────────────────────────────────
    if not _phase_done(state, "sync_api"):
        # API key + S3 needed from here — prompt if missing
        api_key = args.api_key or os.environ.get("SYNC_API_KEY", "")
        s3_user = args.s3_user or S3_PREFIX

        while not api_key:
            print(f"\n{'='*60}")
            print(f"  sync.so API key is required for lip-sync processing.")
            print(f"  Progress is saved in: {dirs['root']}")
            print(f"{'='*60}")
            api_key = input("  Enter your sync.so API key (or Ctrl+C to exit): ").strip()

        while True:
            try:
                s3_user = check_s3_access(s3_user)
                break
            except RuntimeError as e:
                logger.error(str(e))
                print(f"\n  Progress is saved in: {dirs['root']}")
                s3_user = input("  Enter your S3 folder name (or Ctrl+C to exit): ").strip()

        with phase_timer("7. sync.so API processing"):
            process_scenes_with_api(analysis, dirs, api_key, s3_user=s3_user)
            save_state(dirs, "sync_api")
    else:
        logger.info(f"  sync.so processing: skipped")

    # ── Phase 8: Stitch ───────────────────────────────────────────────────
    if not _phase_done(state, "stitch"):
        with phase_timer("8. Final stitching"):
            stitch_scenes(dirs, scenes_data)
            save_state(dirs, "stitch")
    else:
        logger.info(f"  Stitching: skipped")

    # ── Done ──────────────────────────────────────────────────────────────
    save_state(dirs, "done", status="done")
    print_report(dirs, scenes_data, analysis)


if __name__ == "__main__":
    main()
