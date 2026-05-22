"""
sync_client.py — Production-Grade Sync.so API Client (hybrid sync-3 / lipsync-2-pro)
====================================================================================
Drop-in upgrade for `sync_api/client.py`. Adds:
  - correct model registry (sync-3, lipsync-2-pro, lipsync-2, react-1, lipsync-1.9.0-beta)
  - per-scene MODEL ROUTING (the real hybrid)
  - per-model option filtering (unsupported options are not sent)
  - temperature + sync_mode + occlusion + active-speaker options
  - robust polling (PENDING/PROCESSING/COMPLETED/FAILED/REJECTED) with backoff
  - optional webhook support

API: POST https://api.sync.so/v2/generate    (header: x-api-key)
Models verified against the Sync.so Generate API (May 2026).
"""
from __future__ import annotations

import time
import logging

import requests

logger = logging.getLogger("sync_client")

SYNC_API_BASE = "https://api.sync.so/v2"

# ── Model registry — which options each model actually supports ──────────────
# Unsupported options are dropped before the request (the API would ignore them
# anyway, but a clean payload avoids confusion and future breakage).
MODEL_OPTIONS = {
    "sync-3":              {"sync_mode", "temperature", "active_speaker_detection"},
    "lipsync-2-pro":       {"sync_mode", "temperature", "active_speaker_detection",
                            "occlusion_detection_enabled"},
    "lipsync-2":           {"sync_mode", "temperature", "active_speaker_detection",
                            "occlusion_detection_enabled"},
    "react-1":             {"sync_mode", "temperature", "model_mode", "prompt"},
    "lipsync-1.9.0-beta":  {"sync_mode"},
}
VALID_MODELS = set(MODEL_OPTIONS)
TERMINAL_STATES = {"COMPLETED", "FAILED", "REJECTED"}


# ─────────────────────────────────────────────────────────────────────────────
# Model routing — the hybrid
# ─────────────────────────────────────────────────────────────────────────────
def select_model(scene: dict) -> str:
    """Pick the best Sync.so model for a scene from its complexity analysis.

    `scene` is expected to carry boolean-ish hints already produced by the
    pipeline's complexity/obstruction analysis, e.g.:
        multi_person, has_occlusion, extreme_angle, is_closeup, is_complex

    Routing:
      sync-3        -> complex visual conditions (its native visual
                       intelligence handles angles/occlusion/multi-face/4K)
      lipsync-2-pro -> clean frontal single talking head (diffusion super-res
                       gives the finest facial detail)
    """
    complex_conditions = (
        scene.get("multi_person")
        or scene.get("has_occlusion")
        or scene.get("extreme_angle")
        or scene.get("is_closeup")
        or scene.get("is_complex")
    )
    model = "sync-3" if complex_conditions else "lipsync-2-pro"
    logger.info("scene %s -> model %s", scene.get("scene", "?"), model)
    return model


def build_options(scene: dict, model: str, temperature: float = 0.5,
                   sync_mode: str = "cut_off") -> dict:
    """Build the options block, filtered to what `model` supports."""
    wanted = {
        "sync_mode": sync_mode,
        "temperature": temperature,
    }
    # Multi-person -> active speaker detection (auto)
    if scene.get("multi_person"):
        wanted["active_speaker_detection"] = {"auto_detect": True}
    # Occlusion present -> enable occlusion detection (slower, only when needed)
    if scene.get("has_occlusion"):
        wanted["occlusion_detection_enabled"] = True

    supported = MODEL_OPTIONS.get(model, set())
    return {k: v for k, v in wanted.items() if k in supported}


# ─────────────────────────────────────────────────────────────────────────────
# API calls
# ─────────────────────────────────────────────────────────────────────────────
def create_generation(video_url: str, audio_url: str, api_key: str,
                       model: str, options: dict,
                       output_name: str | None = None,
                       webhook_url: str | None = None) -> dict:
    """Submit a lip-sync generation job. Returns the API response (incl. id)."""
    if model not in VALID_MODELS:
        raise ValueError(f"Unknown model '{model}'. Valid: {sorted(VALID_MODELS)}")

    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    payload: dict = {
        "model": model,
        "input": [
            {"type": "video", "url": video_url},
            {"type": "audio", "url": audio_url},
        ],
        "options": options,
    }
    if output_name:
        payload["outputFileName"] = output_name
    if webhook_url:
        payload["webhookUrl"] = webhook_url

    resp = requests.post(f"{SYNC_API_BASE}/generate", headers=headers,
                         json=payload, timeout=30)
    if resp.status_code >= 400:
        # surface the API error body — REJECTED jobs usually explain the bad input
        raise RuntimeError(f"create_generation {resp.status_code}: {resp.text}")
    return resp.json()


def poll_generation(job_id: str, api_key: str,
                    poll_interval: int = 10, max_wait: int = 3600) -> dict:
    """Poll until a terminal state. Light exponential backoff (cap 30 s)."""
    headers = {"x-api-key": api_key}
    elapsed, interval = 0, poll_interval
    while elapsed < max_wait:
        time.sleep(interval)
        elapsed += interval
        resp = requests.get(f"{SYNC_API_BASE}/generate/{job_id}",
                             headers=headers, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        status = result.get("status", "UNKNOWN")
        logger.info("job %s: %s (%ds)", job_id, status, elapsed)
        if status in TERMINAL_STATES:
            return result
        interval = min(interval + 5, 30)
    raise TimeoutError(f"job {job_id} timed out after {max_wait}s")


def download(url: str, save_path: str) -> None:
    """Download the result video to disk."""
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    with open(save_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)


# ─────────────────────────────────────────────────────────────────────────────
# High-level: process one scene end-to-end with routing
# ─────────────────────────────────────────────────────────────────────────────
def process_scene(scene: dict, video_url: str, audio_url: str, api_key: str,
                   output_path: str, temperature: float = 0.5,
                   sync_mode: str = "cut_off", webhook_url: str | None = None,
                   force_model: str | None = None) -> dict:
    """Route -> generate -> poll -> download one scene.

    Returns a record dict (scene, model, options, status, output_path, error,
    cost-relevant duration if returned by the API).
    """
    model = force_model or select_model(scene)
    options = build_options(scene, model, temperature, sync_mode)
    sn = scene.get("scene", "?")

    record = {"scene": sn, "model": model, "options": options,
              "status": None, "output_path": None, "error": None}
    try:
        job = create_generation(
            video_url, audio_url, api_key, model, options,
            output_name=f"scene_{sn}_synced" if isinstance(sn, int) else None,
            webhook_url=webhook_url,
        )
        job_id = job.get("id")
        if not job_id:
            record["error"] = f"no job id in response: {job}"
            return record

        result = poll_generation(job_id, api_key)
        record["status"] = result.get("status")
        record["duration"] = result.get("outputDuration") or result.get("duration")

        if record["status"] == "COMPLETED":
            out_url = result.get("outputUrl") or result.get("output_url")
            if out_url:
                download(out_url, output_path)
                record["output_path"] = output_path
            else:
                record["error"] = "COMPLETED but no output URL"
        else:
            # FAILED / REJECTED -> capture why
            record["error"] = result.get("error") or f"status={record['status']}"
    except Exception as exc:  # noqa: BLE001 - record and continue the batch
        record["error"] = str(exc)
        logger.error("scene %s failed: %s", sn, exc)
    return record


# ─────────────────────────────────────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    API_KEY = os.environ["SYNC_API_KEY"]

    # `scene` dicts come from the pipeline's complexity analysis.
    example_scene = {
        "scene": 12,
        "multi_person": False,
        "has_occlusion": False,
        "extreme_angle": False,
        "is_closeup": True,    # -> routes to sync-3
        "is_complex": False,
    }
    rec = process_scene(
        example_scene,
        video_url="https://your-bucket/scene_012.mp4",
        audio_url="https://your-bucket/scene_012_hindi.wav",
        api_key=API_KEY,
        output_path="scene_012_synced.mp4",
        temperature=0.6,          # slightly expressive dialogue
    )
    print(rec)
