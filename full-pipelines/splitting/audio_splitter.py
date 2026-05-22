"""
Audio Splitter
==============
Convert audio to 16kHz mono and split at scene timestamps.
"""

import sys

from config import logger
from utils import run_ffmpeg, progress


def convert_audio_16khz(audio_path, dirs):
    """Convert audio to 16kHz mono WAV."""
    output_path = dirs["inputs"] / "audio_16khz_mono.wav"
    logger.info("Converting audio to 16kHz mono...")
    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-ar", "16000",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        str(output_path),
    ]
    if not run_ffmpeg(cmd, "16kHz conversion"):
        logger.error("Audio conversion failed.")
        sys.exit(1)
    logger.info(f"16kHz mono audio: {output_path}")
    return str(output_path)


def split_audio(audio_16khz_path, scenes_data, dirs):
    """Split audio at scene timestamps."""
    scenes = scenes_data["scenes"]
    total = len(scenes)
    logger.info(f"Splitting audio into {total} scenes...")

    for i, scene in enumerate(scenes):
        scene_num = scene["scene"]
        start = scene["start_seconds"]
        dur = scene["duration_seconds"]
        output_path = dirs["audio"] / f"scene_{scene_num:03d}.wav"

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", audio_16khz_path,
            "-t", str(dur),
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(output_path),
        ]
        run_ffmpeg(cmd, f"audio_scene_{scene_num:03d}")
        progress(i + 1, total, "Splitting audio")

    logger.info("Audio splitting done.")
