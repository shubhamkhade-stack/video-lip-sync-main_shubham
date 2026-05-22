#!/usr/bin/env python3
"""
Lip Sync Scoring Pipeline
==========================
Compare original vs synced scene videos and generate quality scores.

Usage:
    python3 ScoringPipeline.py --original /path/to/original/scenes --synced /path/to/synced/scenes
    python3 ScoringPipeline.py --task task_30-03_15-10   # auto-detect from SyncPipeline task
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from config import logger
from scoring.lip_sync_scorer import score_all_scenes, score_scene

# ── Constants ────────────────────────────────────────────────────────────────
TASKS_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "scoring_tasks"
SYNC_TASKS_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "tasks"


def setup_task_folder():
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    folder_name = f"score_{now.strftime('%d-%m_%H-%M')}"
    task_dir = TASKS_DIR / folder_name
    output_dir = task_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created scoring task: {task_dir}")
    return task_dir, output_dir


def print_score_table(results):
    """Print formatted scoring table."""
    print(f"\n{'='*95}")
    print(f"  LIP SYNC QUALITY SCORES")
    print(f"{'='*95}")

    header = (f"  {'Scene':<15} {'Overall':>8} {'Face SSIM':>10} {'Lip Δ px':>9} "
              f"{'Sharpness':>10} {'Face':>6} {'Sharp':>6} {'Lip':>6} {'Frames':>7}")
    print(f"\n{header}")
    print(f"  {'─'*88}")

    scores = []
    for scene_name, result in sorted(results.items()):
        if "error" in result:
            print(f"  {scene_name:<15} {'ERROR':>8}   {result['error']}")
            continue

        overall = result["overall_score"]
        scores.append(overall)

        # Color indicator
        if overall >= 80:
            grade = "A"
        elif overall >= 60:
            grade = "B"
        elif overall >= 40:
            grade = "C"
        else:
            grade = "D"

        print(f"  {scene_name:<15} {overall:>6.1f}{grade:>2} "
              f"{result['face_preservation_ssim']:>10.4f} "
              f"{result['lip_movement_delta_px']:>9.2f} "
              f"{result['mouth_sharpness_ratio']:>10.4f} "
              f"{result['face_score']:>6.1f} "
              f"{result['sharpness_score']:>6.1f} "
              f"{result['lip_naturalness_score']:>6.1f} "
              f"{result['frames_scored']:>7}")

    print(f"  {'─'*88}")

    if scores:
        avg = sum(scores) / len(scores)
        grade = "A" if avg >= 80 else "B" if avg >= 60 else "C" if avg >= 40 else "D"
        print(f"  {'AVERAGE':<15} {avg:>6.1f}{grade:>2}")
        print()
        print(f"  Grade: A (80-100) Excellent | B (60-79) Good | C (40-59) Fair | D (0-39) Poor")

    print(f"\n  Score Breakdown:")
    print(f"    Face (40%)  = Face preservation SSIM outside lip region")
    print(f"    Sharp (30%) = Mouth area sharpness ratio (synced vs original)")
    print(f"    Lip (30%)   = Lip movement naturalness (3-30px change = ideal)")

    print(f"\n{'='*95}\n")


def main():
    parser = argparse.ArgumentParser(description="Lip Sync Scoring Pipeline")
    parser.add_argument("--original", "-o", help="Path to original scene videos directory")
    parser.add_argument("--synced", "-s", help="Path to synced scene videos directory")
    parser.add_argument("--task", "-t", help="SyncPipeline task folder name (auto-detect paths)")
    parser.add_argument("--fps", type=int, default=3, help="Sample FPS for scoring (default: 3)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Lip Sync Quality Scorer")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # Resolve input directories
    if args.task:
        task_path = SYNC_TASKS_DIR / args.task
        original_dir = task_path / "outputs" / "SplittedScenes" / "videos"
        synced_dir = task_path / "outputs" / "outputScenes"

        if not original_dir.exists():
            logger.error(f"Original scenes not found: {original_dir}")
            return
        if not synced_dir.exists():
            logger.error(f"Synced scenes not found: {synced_dir}")
            return

        logger.info(f"Using SyncPipeline task: {args.task}")
        logger.info(f"  Original: {original_dir}")
        logger.info(f"  Synced:   {synced_dir}")

    elif args.original and args.synced:
        original_dir = Path(args.original)
        synced_dir = Path(args.synced)

        if not original_dir.exists():
            logger.error(f"Original directory not found: {original_dir}")
            return
        if not synced_dir.exists():
            logger.error(f"Synced directory not found: {synced_dir}")
            return
    else:
        # Interactive: list available sync tasks
        if SYNC_TASKS_DIR.exists():
            tasks = sorted([d.name for d in SYNC_TASKS_DIR.iterdir()
                           if d.is_dir() and d.name.startswith("task_")])
            if tasks:
                print(f"  Available SyncPipeline tasks:")
                for i, t in enumerate(tasks, 1):
                    out_dir = SYNC_TASKS_DIR / t / "outputs" / "outputScenes"
                    synced_count = len(list(out_dir.glob("*.mp4"))) if out_dir.exists() else 0
                    print(f"    {i}. {t}  ({synced_count} synced scenes)")

                choice = input("\n  Enter number, or ENTER to specify paths manually: ").strip()
                if choice.isdigit() and 0 < int(choice) <= len(tasks):
                    task_name = tasks[int(choice) - 1]
                    original_dir = SYNC_TASKS_DIR / task_name / "outputs" / "SplittedScenes" / "videos"
                    synced_dir = SYNC_TASKS_DIR / task_name / "outputs" / "outputScenes"
                else:
                    original_dir = Path(input("  Path to original scenes: ").strip())
                    synced_dir = Path(input("  Path to synced scenes: ").strip())
            else:
                original_dir = Path(input("  Path to original scenes: ").strip())
                synced_dir = Path(input("  Path to synced scenes: ").strip())
        else:
            original_dir = Path(input("  Path to original scenes: ").strip())
            synced_dir = Path(input("  Path to synced scenes: ").strip())

    # Count matching scenes
    orig_videos = sorted(original_dir.glob("scene_*.mp4"))
    sync_videos = sorted(synced_dir.glob("scene_*.mp4"))
    matched = [v for v in orig_videos if (synced_dir / v.name).exists()]
    logger.info(f"Found {len(orig_videos)} original, {len(sync_videos)} synced, {len(matched)} matched scenes")

    if not matched:
        logger.error("No matching scene pairs found.")
        return

    # Setup output
    task_dir, output_dir = setup_task_folder()

    # Score
    logger.info(f"\nScoring {len(matched)} scene(s) at {args.fps} fps...\n")
    t0 = time.time()
    results = score_all_scenes(str(original_dir), str(synced_dir), sample_fps=args.fps)
    elapsed = time.time() - t0
    logger.info(f"\nScoring complete in {elapsed:.1f}s")

    # Save JSON
    json_output = {
        "task": task_dir.name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "original_dir": str(original_dir),
        "synced_dir": str(synced_dir),
        "settings": {"sample_fps": args.fps},
        "scenes": {},
        "summary": {},
    }

    all_scores = []
    for scene_name, result in results.items():
        # Remove per-frame details from JSON summary (keep it clean)
        details = result.pop("details", [])
        json_output["scenes"][scene_name] = result
        if "overall_score" in result and "error" not in result:
            all_scores.append(result["overall_score"])

        # Save per-frame details separately
        detail_path = output_dir / f"{scene_name}_details.json"
        with open(detail_path, "w") as f:
            json.dump({"scene": scene_name, "frames": details}, f, indent=2)

    if all_scores:
        json_output["summary"] = {
            "total_scenes": len(results),
            "average_score": round(sum(all_scores) / len(all_scores), 1),
            "best_scene": max(results.items(), key=lambda x: x[1].get("overall_score", 0))[0],
            "worst_scene": min(results.items(), key=lambda x: x[1].get("overall_score", 100))[0],
            "grade_A": sum(1 for s in all_scores if s >= 80),
            "grade_B": sum(1 for s in all_scores if 60 <= s < 80),
            "grade_C": sum(1 for s in all_scores if 40 <= s < 60),
            "grade_D": sum(1 for s in all_scores if s < 40),
        }

    json_path = output_dir / "ScoringResults.json"
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)
    logger.info(f"Saved: {json_path}")

    # Print table
    print_score_table(results)

    print(f"  Outputs saved to: {output_dir}")
    print(f"    - ScoringResults.json            (overall scores)")
    print(f"    - scene_XXX_details.json         (per-frame breakdown)")
    print()


if __name__ == "__main__":
    main()
