"""
Dialogue Detection
==================
Detect which scenes have dialogue using Silero VAD.
"""

import json

import torch
import torchaudio

from config import logger
from utils import progress


def detect_dialogue(audio_dir, scenes_data, output_dir=None):
    """Detect which scenes have dialogue using Silero VAD.
    Saves results to dialogue_results.json if output_dir is provided."""
    logger.info("Loading Silero VAD model...")
    torch.set_num_threads(1)
    model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad",
                               force_reload=False, trust_repo=True)

    scenes = scenes_data["scenes"]
    total = len(scenes)
    dialogue_set = set()

    logger.info(f"Detecting dialogue in {total} scenes...")
    for i, scene in enumerate(scenes):
        scene_num = scene["scene"]
        wav_path = audio_dir / f"scene_{scene_num:03d}.wav"

        if not wav_path.exists():
            progress(i + 1, total, "Dialogue detection")
            continue

        wav, sr = torchaudio.load(str(wav_path))
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)

        model.reset_states()
        window_size = 512
        speech_samples = 0

        for j in range(0, wav.shape[0] - window_size, window_size):
            chunk = wav[j:j + window_size]
            prob = model(chunk, 16000).item()
            if prob >= 0.5:
                speech_samples += window_size

        speech_duration = speech_samples / 16000.0
        if speech_duration >= 0.5:
            dialogue_set.add(scene_num)

        progress(i + 1, total, "Dialogue detection")

    logger.info(f"Dialogue detected in {len(dialogue_set)}/{total} scenes.")

    # Save intermediate results for resume
    if output_dir:
        results_path = output_dir / "dialogue_results.json"
        with open(results_path, "w") as f:
            json.dump({"dialogue_scenes": sorted(dialogue_set)}, f, indent=2)
        logger.info(f"Dialogue results saved: {results_path}")

    return dialogue_set


def load_dialogue_results(output_dir):
    """Load previously saved dialogue results. Returns set or None."""
    results_path = output_dir / "dialogue_results.json"
    if not results_path.exists():
        return None
    with open(results_path) as f:
        data = json.load(f)
    return set(data["dialogue_scenes"])
