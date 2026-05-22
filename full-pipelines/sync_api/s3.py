"""
S3 Operations
=============
Upload scene files to S3 and generate presigned URLs for sync.so API.
"""

from config import S3_BUCKET, S3_PRESIGN_EXPIRY, logger
from utils import progress


def _get_s3_client():
    """Get a boto3 S3 client."""
    import boto3
    return boto3.client("s3")


def _ensure_s3_folder(s3_user, task_name):
    """Ensure the user's S3 folder and task subfolder exist (by writing a marker).
    Returns the S3 key prefix for this task, e.g. 'aryan/task_30-03_14-27/'."""
    s3 = _get_s3_client()
    prefix = f"{s3_user}/{task_name}/"
    # S3 doesn't have real folders — upload a zero-byte marker to create the "folder"
    s3.put_object(Bucket=S3_BUCKET, Key=f"{prefix}.marker", Body=b"")
    logger.info(f"S3 task folder ready: s3://{S3_BUCKET}/{prefix}")
    return prefix


def _s3_upload_file(local_path, s3_key):
    """Upload a local file to S3 and return a presigned URL."""
    s3 = _get_s3_client()
    s3.upload_file(str(local_path), S3_BUCKET, s3_key)
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": s3_key},
        ExpiresIn=S3_PRESIGN_EXPIRY,
    )
    return url


def _upload_scenes_to_s3(lipsync_scenes, dirs, s3_prefix):
    """Upload video and audio scene files to S3. Returns {scene_num: (video_url, audio_url)}."""
    urls = {}
    total = len(lipsync_scenes)
    for i, scene in enumerate(lipsync_scenes, 1):
        sn = scene["scene"]
        video_path = dirs["videos"] / f"scene_{sn:03d}.mp4"
        audio_path = dirs["audio"] / f"scene_{sn:03d}.wav"

        video_url = _s3_upload_file(video_path, f"{s3_prefix}scenes_video/scene_{sn:03d}.mp4")
        audio_url = _s3_upload_file(audio_path, f"{s3_prefix}scenes_audio/scene_{sn:03d}.wav")
        urls[sn] = (video_url, audio_url)
        progress(i, total, "S3 upload")
    return urls
