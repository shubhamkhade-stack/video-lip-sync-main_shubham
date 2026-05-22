"""
Pipeline Report
===============
Generate and print the final pipeline report with timings and scene analysis.
"""

import json

from utils import timings


def print_report(dirs, scenes_data, analysis):
    """Print and save the final pipeline report."""
    total_time = sum(timings.values())

    report = {
        "task_folder": str(dirs["root"]),
        "video": scenes_data["video"],
        "total_scenes": scenes_data["total_scenes"],
        "analysis_summary": {
            "scenes_with_audio": sum(1 for s in analysis if s["haveAudio"]),
            "person_speaking": sum(1 for s in analysis if s["personSpeaking"]),
            "multiple_persons": sum(1 for s in analysis if s["multiplePersonInScene"]),
            "lip_obstruction": sum(1 for s in analysis if s["haveObstructionInLip"]),
            "complex_scenes": sum(1 for s in analysis if s["isComplexScene"]),
            "lipsync_required": sum(1 for s in analysis if s["haveAudio"] and s["personSpeaking"]),
        },
        "timings": {k: round(v, 1) for k, v in timings.items()},
        "total_time_seconds": round(total_time, 1),
        "total_time_human": f"{int(total_time//60)}m {int(total_time%60)}s",
    }

    report_path = dirs["outputs"] / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Task folder: {dirs['root']}")
    print(f"  Total scenes: {scenes_data['total_scenes']}")
    print(f"  Lip-sync scenes: {report['analysis_summary']['lipsync_required']}")
    print(f"\n  Timings:")
    for step, secs in timings.items():
        mins = int(secs // 60)
        remaining = int(secs % 60)
        print(f"    {step:<30s}  {mins}m {remaining}s")
    print(f"    {'─'*40}")
    print(f"    {'TOTAL':<30s}  {int(total_time//60)}m {int(total_time%60)}s")
    print(f"\n  Output: {dirs['outputs'] / 'final_output.mp4'}")
    print(f"  Report: {report_path}")
    print(f"{'='*60}\n")
