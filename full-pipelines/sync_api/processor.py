"""
sync.so Scene Processor
=======================
Orchestrate S3 uploads and sync.so API calls for all lip-sync scenes.
"""

import concurrent.futures
import shutil

from config import S3_PREFIX, PARALLEL_JOBS, logger
from utils import progress
from sync_api.s3 import _ensure_s3_folder, _upload_scenes_to_s3
from sync_api.client import sync_create_job, sync_poll, sync_download
from sync_api.sync_client import process_scene as hybrid_process_scene


def process_single_scene(scene_num, video_url, audio_url, output_path, api_key,
                         is_complex=False, is_obstructed=False, multi_person=False):
    """Process a single scene through sync.so API. Returns (scene_num, success, error)."""
    try:
        scene_meta = {
            "scene": scene_num,
            "multi_person": multi_person,
            "has_occlusion": is_obstructed,
            "is_complex": is_complex,
            "is_closeup": False,
            "extreme_angle": False,
        }
        rec = hybrid_process_scene(
            scene_meta, video_url, audio_url, api_key,
            str(output_path), temperature=0.6,
        )
        return scene_num, rec["status"] == "COMPLETED", rec.get("error")

    except Exception as e:
        return scene_num, False, str(e)


def process_scenes_with_api(analysis, dirs, api_key, s3_user=None):
    """Process all scenes: copy non-lipsync, API-call lipsync scenes 3 at a time."""
    lipsync_scenes = []
    non_lipsync_scenes = []

    for scene in analysis:
        sn = scene["scene"]
        # Needs lip-sync if: has audio AND a person is speaking (visible face)
        needs_sync = scene["haveAudio"] and scene["personSpeaking"]

        if needs_sync:
            lipsync_scenes.append(scene)
        else:
            non_lipsync_scenes.append(scene)

    # Copy non-lipsync scenes (original video)
    logger.info(f"Copying {len(non_lipsync_scenes)} non-lipsync scenes...")
    for scene in non_lipsync_scenes:
        sn = scene["scene"]
        src = dirs["videos"] / f"scene_{sn:03d}.mp4"
        dst = dirs["output_scenes"] / f"scene_{sn:03d}.mp4"
        if src.exists():
            shutil.copy2(str(src), str(dst))

    if not lipsync_scenes:
        logger.info("No scenes require lip-sync processing.")
        return

    if not api_key:
        raise RuntimeError(
            "sync.so API key is required for lip-sync processing. "
            "Set SYNC_API_KEY env var or pass --api-key."
        )

    # Resolve S3 user prefix
    user = s3_user or S3_PREFIX
    if not user:
        raise RuntimeError(
            "S3 user prefix is required so sync.so can access scene files.\n"
            "  Set it with:  export PIPELINE_S3_USER=yourname\n"
            "  Or pass:      --s3-user yourname"
        )

    # Upload scene files to S3
    task_name = dirs["root"].name
    s3_prefix = _ensure_s3_folder(user, task_name)

    logger.info(f"Uploading {len(lipsync_scenes)} scene pairs to S3...")
    scene_urls = _upload_scenes_to_s3(lipsync_scenes, dirs, s3_prefix)

    total = len(lipsync_scenes)
    completed = 0
    failed = 0

    logger.info(f"Processing {total} scenes via sync.so API ({PARALLEL_JOBS} parallel)...")

    # Process in batches of PARALLEL_JOBS
    for batch_start in range(0, total, PARALLEL_JOBS):
        batch = lipsync_scenes[batch_start:batch_start + PARALLEL_JOBS]

        with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_JOBS) as executor:
            futures = {}
            for scene in batch:
                sn = scene["scene"]
                output_path = dirs["output_scenes"] / f"scene_{sn:03d}.mp4"
                video_url, audio_url = scene_urls[sn]

                future = executor.submit(
                    process_single_scene,
                    sn, video_url, audio_url, str(output_path),
                    api_key,
                    is_complex=scene["isComplexScene"],
                    is_obstructed=scene["haveObstructionInLip"],
                    multi_person=scene["multiplePersonInScene"],
                )
                futures[future] = sn

            for future in concurrent.futures.as_completed(futures):
                sn, success, error = future.result()
                if success:
                    completed += 1
                else:
                    failed += 1
                    logger.error(f"  Scene {sn} failed: {error}")
                    # Fallback: copy original
                    src = dirs["videos"] / f"scene_{sn:03d}.mp4"
                    dst = dirs["output_scenes"] / f"scene_{sn:03d}.mp4"
                    if src.exists():
                        shutil.copy2(str(src), str(dst))

                progress(completed + failed, total, "sync.so processing",
                         f"({completed} ok, {failed} failed)")

    logger.info(f"sync.so done: {completed} succeeded, {failed} failed out of {total}.")
