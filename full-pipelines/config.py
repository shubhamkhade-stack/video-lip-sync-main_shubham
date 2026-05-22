"""
Pipeline Configuration
======================
All constants, paths, thresholds, and the shared logger instance.
"""

import logging
import os

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pipeline")

# ── sync.so API ──────────────────────────────────────────────────────────────
SYNC_API_BASE = "https://api.sync.so/v2"
SYNC_MODEL = "lipsync-2-pro"
SYNC_MODE = "cut_off"
POLL_INTERVAL = 10
MAX_WAIT = 600
PARALLEL_JOBS = 3

# ── S3 ───────────────────────────────────────────────────────────────────────
S3_BUCKET = "framexstudio-files"
S3_PREFIX = os.environ.get("PIPELINE_S3_USER", "")  # set per-user, e.g. "aryanTestingFiles"
S3_PRESIGN_EXPIRY = 3600  # 1 hour

# ── Face Model ───────────────────────────────────────────────────────────────
FACE_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models", "face_landmarker_v2_with_blendshapes.task"
)

# ── MediaPipe Lip Landmarks (obstruction detection) ─────────────────────────
OUTER_LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185
]
INNER_LIP_INDICES = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191
]
ALL_LIP_INDICES = list(set(OUTER_LIP_INDICES + INNER_LIP_INDICES))

# ── Complexity Landmarks ─────────────────────────────────────────────────────
NOSE_TIP = 1
LEFT_EYE_OUTER = 263
RIGHT_EYE_OUTER = 33
CHIN = 152
