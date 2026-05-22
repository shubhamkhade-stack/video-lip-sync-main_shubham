"""
Scene Stitcher
==============
Concatenate all output scenes into one final video.
"""

from config import logger
from utils import run_ffmpeg


def stitch_scenes(dirs, scenes_data):
    """Concatenate all output scenes into one final video."""
    total_scenes = scenes_data["total_scenes"]
    concat_list = dirs["output_scenes"] / "concat_list.txt"

    # Build concat file in scene order
    entries = []
    for i in range(1, total_scenes + 1):
        scene_path = dirs["output_scenes"] / f"scene_{i:03d}.mp4"
        if scene_path.exists():
            entries.append(f"file '{scene_path.resolve()}'")
        else:
            logger.warning(f"  Missing scene {i} for stitching!")

    with open(concat_list, "w") as f:
        f.write("\n".join(entries))

    output_path = dirs["outputs"] / "final_output.mp4"
    logger.info(f"Stitching {len(entries)} scenes into final output...")

    # Re-encode to uniform 30fps because sync.so may return scenes with
    # different frame rates / timebases, which breaks concat with -c copy.
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-r", "30",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(output_path),
    ]
    if run_ffmpeg(cmd, "final stitch"):
        size_mb = output_path.stat().st_size / 1_048_576
        logger.info(f"Final output: {output_path} ({size_mb:.1f} MB)")
    else:
        logger.error("Stitching failed!")

    # Clean up concat list
    concat_list.unlink(missing_ok=True)
    return str(output_path)
