"""
sync.so API Client
==================
HTTP interaction with the sync.so lip-sync API: create jobs, poll status, download results.
"""

import time

from config import SYNC_API_BASE, SYNC_MODEL, SYNC_MODE, POLL_INTERVAL, MAX_WAIT


def sync_create_job(video_url, audio_url, api_key, reasoning=False, detect_obstructions=False,
                    active_speaker=False, output_name=None):
    """Submit a lip-sync generation job to sync.so using URLs."""
    import requests
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    payload = {
        "model": SYNC_MODEL,
        "input": [
            {"type": "video", "url": video_url},
            {"type": "audio", "url": audio_url},
        ],
        "options": {"sync_mode": SYNC_MODE},
    }
    if active_speaker:
        payload["options"]["active_speaker_detection"] = {"auto_detect": True}
    if detect_obstructions:
        payload["options"]["occlusion_detection_enabled"] = True
    if output_name:
        payload["outputFileName"] = output_name

    resp = requests.post(f"{SYNC_API_BASE}/generate", headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def sync_poll(job_id, api_key):
    """Poll for job completion."""
    import requests
    headers = {"x-api-key": api_key}
    elapsed = 0
    while elapsed < MAX_WAIT:
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
        resp = requests.get(f"{SYNC_API_BASE}/generate/{job_id}", headers=headers, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        status = result.get("status", "UNKNOWN")
        if status in {"COMPLETED", "FAILED", "REJECTED"}:
            return result
    raise TimeoutError(f"Job {job_id} timed out after {MAX_WAIT}s")


def sync_download(url, save_path):
    """Download result video."""
    import requests
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(save_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
