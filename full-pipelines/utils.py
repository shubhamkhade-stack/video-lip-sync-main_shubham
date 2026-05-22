"""
Pipeline Utilities
==================
Timing, progress bars, FFmpeg wrappers, task folder setup, and input validation.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from config import S3_BUCKET, logger

# ── Constants ────────────────────────────────────────────────────────────────
TASKS_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "tasks"

PHASE_ORDER = [
    "setup", "video_split", "face_split", "audio_split",
    "dialogue", "complexity", "scene_analysis",
    "sync_api", "stitch", "done",
]

# ── Phase Timing ─────────────────────────────────────────────────────────────
timings = {}


def phase_timer(name):
    """Context manager to time a pipeline phase."""
    class Timer:
        def __enter__(self):
            self.start = time.time()
            return self
        def __exit__(self, *args):
            timings[name] = time.time() - self.start
    return Timer()


# ── Progress Bar ─────────────────────────────────────────────────────────────

def progress(current, total, prefix="", suffix=""):
    pct = (current / total * 100) if total > 0 else 0
    bar_len = 30
    filled = int(bar_len * current // total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  {prefix} |{bar}| {current}/{total} ({pct:.0f}%) {suffix}   ", end="", flush=True)
    if current >= total:
        print()


# ── FFmpeg / FFprobe ─────────────────────────────────────────────────────────

def run_ffmpeg(cmd, desc=""):
    """Run an FFmpeg command silently."""
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        logger.error(f"FFmpeg failed{' (' + desc + ')' if desc else ''}: {result.stderr.decode()[-200:]}")
    return result.returncode == 0


def get_video_duration(path):
    """Get video duration in seconds using ffprobe."""
    cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0


# ── File Helpers ─────────────────────────────────────────────────────────────

def extract_scene_number(filename):
    """Extract scene number from filename like 'scene_009.mp4'."""
    name = os.path.splitext(filename)[0]
    parts = name.split("_scene_")
    if len(parts) >= 2:
        try:
            return int(parts[-1])
        except ValueError:
            pass
    segments = name.split("_")
    for seg in reversed(segments):
        if seg.isdigit():
            return int(seg)
    return None


# ── Task Folder Setup ────────────────────────────────────────────────────────

def _build_dirs(task_dir):
    """Build the dirs dict from a task directory path."""
    task_dir = Path(task_dir)
    return {
        "root": task_dir,
        "inputs": task_dir / "inputs",
        "outputs": task_dir / "outputs",
        "splitted": task_dir / "outputs" / "SplittedScenes",
        "videos": task_dir / "outputs" / "SplittedScenes" / "videos",
        "audio": task_dir / "outputs" / "SplittedScenes" / "audio",
        "output_scenes": task_dir / "outputs" / "outputScenes",
    }


def setup_task_folder():
    """Create a new task folder under tasks/."""
    TASKS_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    folder_name = f"task_{now.strftime('%d-%m_%H-%M')}"
    task_dir = TASKS_DIR / folder_name

    dirs = _build_dirs(task_dir)
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    logger.info(f"Created task folder: {task_dir}")
    return dirs


# ── State Management ─────────────────────────────────────────────────────────

def save_state(dirs, phase, status="running", **extra):
    """Save pipeline state to state.json in the task folder.
    Call after each phase completes to record progress."""
    state_path = dirs["root"] / "state.json"

    # Load existing state or start fresh
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
    else:
        state = {
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "completed_phases": [],
        }

    # Mark the phase as completed
    if phase not in state["completed_phases"]:
        state["completed_phases"].append(phase)

    # Determine next phase
    try:
        idx = PHASE_ORDER.index(phase)
        next_phase = PHASE_ORDER[idx + 1] if idx + 1 < len(PHASE_ORDER) else "done"
    except ValueError:
        next_phase = phase

    state["current_phase"] = next_phase
    state["status"] = status
    state["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Merge any extra data (video name, audio name, etc.)
    state.update(extra)

    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def load_state(task_dir):
    """Load state.json from a task folder. Returns None if not found."""
    state_path = Path(task_dir) / "state.json"
    if not state_path.exists():
        return None
    with open(state_path) as f:
        return json.load(f)


def rebuild_dirs(task_dir):
    """Reconstruct dirs dict from an existing task folder."""
    dirs = _build_dirs(task_dir)
    # Ensure all dirs exist (they should, but be safe)
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def list_tasks():
    """List all tasks in the tasks/ folder with their state. Returns list of (folder_name, state_dict)."""
    if not TASKS_DIR.exists():
        return []

    tasks = []
    for entry in sorted(TASKS_DIR.iterdir(), reverse=True):
        if entry.is_dir() and entry.name.startswith("task_"):
            state = load_state(entry)
            if state is None:
                # Task folder exists but no state — treat as unknown
                state = {"status": "unknown", "completed_phases": [], "current_phase": "unknown"}
            tasks.append((entry.name, state))
    return tasks


# ── Input Validation ─────────────────────────────────────────────────────────

def wait_for_inputs(dirs, video_path=None, audio_path=None):
    """Wait for user to place video and audio in the inputs folder, or copy provided files."""
    inputs_dir = dirs["inputs"]

    if video_path and audio_path:
        video_src = Path(video_path)
        audio_src = Path(audio_path)
        if not video_src.exists():
            logger.error(f"Video file not found: {video_src}")
            sys.exit(1)
        if not audio_src.exists():
            logger.error(f"Audio file not found: {audio_src}")
            sys.exit(1)
        video_dst = inputs_dir / video_src.name
        audio_dst = inputs_dir / audio_src.name
        shutil.copy2(str(video_src), str(video_dst))
        shutil.copy2(str(audio_src), str(audio_dst))
        logger.info(f"Copied video: {video_src.name}")
        logger.info(f"Copied audio: {audio_src.name}")
        return str(video_dst), str(audio_dst)

    print(f"\n{'='*60}")
    print(f"  Place your files in: {inputs_dir}")
    print(f"    - One .mp4 video file (original)")
    print(f"    - One .wav audio file (dubbed, same duration)")
    print(f"{'='*60}")

    while True:
        input("\n  Press ENTER when files are ready...")
        mp4s = sorted(inputs_dir.glob("*.mp4"))
        wavs = sorted(inputs_dir.glob("*.wav"))

        if not mp4s:
            print("  [!] No .mp4 file found. Please add one.")
            continue
        if not wavs:
            print("  [!] No .wav file found. Please add one.")
            continue

        video_file = str(mp4s[0])
        audio_file = str(wavs[0])

        v_dur = get_video_duration(video_file)
        a_dur = get_video_duration(audio_file)
        print(f"  Video: {mp4s[0].name} ({v_dur:.1f}s)")
        print(f"  Audio: {wavs[0].name} ({a_dur:.1f}s)")

        if abs(v_dur - a_dur) > 2.0:
            print(f"  [!] Duration mismatch: video={v_dur:.1f}s, audio={a_dur:.1f}s (diff={abs(v_dur-a_dur):.1f}s)")
            resp = input("  Continue anyway? (y/n): ").strip().lower()
            if resp != "y":
                continue

        return video_file, audio_file


def check_api_key():
    """Check for sync.so API key. Raises error if not found."""
    api_key = os.environ.get("SYNC_API_KEY", "")
    if api_key:
        logger.info("sync.so API key found in environment.")
        return api_key

    raise RuntimeError(
        "SYNC_API_KEY not found! Lip-sync cannot proceed without it.\n"
        "  Set it with:  export SYNC_API_KEY=your_key_here\n"
        "  Or pass:      --api-key your_key_here"
    )


def check_s3_access(s3_user):
    """Validate S3 user prefix is set and writable. Fail fast before analysis."""
    if not s3_user:
        print(f"\n{'='*60}")
        print(f"  S3 folder required for sync.so API inputs")
        print(f"  (sync.so reads files from S3, not directly from this EC2 instance)")
        print(f"{'='*60}")
        print(f"  Enter your S3 folder name (e.g. aryanTestingFiles):")
        s3_user = input("  > ").strip()
        if not s3_user:
            raise RuntimeError("S3 user folder cannot be empty.")
    import boto3
    s3 = boto3.client("s3")
    test_key = f"{s3_user}/.pipeline_write_test"
    try:
        s3.put_object(Bucket=S3_BUCKET, Key=test_key, Body=b"ok")
        s3.delete_object(Bucket=S3_BUCKET, Key=test_key)
    except Exception as e:
        raise RuntimeError(
            f"Cannot write to s3://{S3_BUCKET}/{s3_user}/\n"
            f"  Error: {e}\n"
            f"  Check that the S3 bucket and IAM permissions are correct."
        )
    logger.info(f"S3 access verified: s3://{S3_BUCKET}/{s3_user}/")
    return s3_user
