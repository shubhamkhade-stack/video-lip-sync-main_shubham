"""
Face Tracking Module (InsightFace + SCRFD)
==========================================
Track and identify faces across video frames using InsightFace.

- SCRFD for face detection (fast, accurate)
- ArcFace for face embeddings (512-dim, enables re-identification)
- Cosine similarity for matching faces across frames

Usage:
    from tracking.face_tracker import track_faces_in_video, track_all_scenes
"""

import json
import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path

import insightface
from insightface.app import FaceAnalysis

from config import logger


def _build_face_app(det_size=(640, 640)):
    """Initialize InsightFace app (downloads models on first run)."""
    app = FaceAnalysis(
        name="buffalo_l",          # includes SCRFD det + ArcFace rec
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=det_size)
    return app


def _cosine_sim(a, b):
    """Cosine similarity between two embedding vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


class FaceTracker:
    """
    Track faces across frames using InsightFace detection + embedding matching.

    Each detected face gets a persistent track_id. Identity is maintained via
    ArcFace embeddings (cosine similarity), with centroid distance as tiebreaker.
    """

    def __init__(self, sim_threshold=0.3, max_centroid_dist=150, max_missing_frames=15):
        """
        Args:
            sim_threshold: Min cosine similarity to match embeddings (0-1).
                           0.3 balances angle tolerance vs false matches.
            max_centroid_dist: Max pixel distance for centroid-based fallback matching.
            max_missing_frames: Drop a track after this many frames without a match.
        """
        self.sim_threshold = sim_threshold
        self.max_centroid_dist = max_centroid_dist
        self.max_missing_frames = max_missing_frames
        self._next_id = 0
        self._tracks = {}  # track_id -> {"embedding", "centroid", "bbox", "missing"}

    def update(self, detections):
        """
        Match new detections to existing tracks.

        Args:
            detections: list of dicts with keys: bbox, centroid, embedding

        Returns:
            list of dicts: same as input but with added "track_id" key
        """
        if not self._tracks:
            results = []
            for det in detections:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {
                    "embedding": det["embedding"],
                    "centroid": det["centroid"],
                    "bbox": det["bbox"],
                    "missing": 0,
                }
                results.append({**det, "track_id": tid})
            return results

        track_ids = list(self._tracks.keys())

        if not detections:
            # No faces this frame — increment missing for all
            for tid in track_ids:
                self._tracks[tid]["missing"] += 1
            self._purge_stale()
            return []

        # Build similarity matrix (tracks x detections) using embedding + centroid
        track_embs = np.array([self._tracks[tid]["embedding"] for tid in track_ids])
        det_embs = np.array([d["embedding"] for d in detections])
        emb_sim = track_embs @ det_embs.T  # cosine similarity (already normalized)

        # Centroid distance matrix (normalized to 0-1 score, closer = higher)
        track_cents = np.array([self._tracks[tid]["centroid"] for tid in track_ids], dtype=float)
        det_cents = np.array([d["centroid"] for d in detections], dtype=float)
        cent_dists = np.linalg.norm(track_cents[:, None] - det_cents[None, :], axis=2)
        cent_score = np.clip(1.0 - cent_dists / self.max_centroid_dist, 0, 1)

        # Combined score: 70% embedding + 30% centroid proximity
        score_matrix = 0.7 * emb_sim + 0.3 * cent_score

        used_tracks = set()
        used_dets = set()
        results = []

        # Greedy matching by highest combined score
        while True:
            if score_matrix.size == 0:
                break
            max_idx = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
            max_score = score_matrix[max_idx]
            # Accept match if embedding similarity is above threshold
            emb_val = emb_sim[max_idx]
            if emb_val < self.sim_threshold:
                break

            ti, di = max_idx
            tid = track_ids[ti]
            used_tracks.add(tid)
            used_dets.add(di)

            # Update track with new detection (use running average for embedding)
            old_emb = self._tracks[tid]["embedding"]
            new_emb = detections[di]["embedding"]
            blended = 0.7 * old_emb + 0.3 * new_emb
            blended = blended / (np.linalg.norm(blended) + 1e-8)  # re-normalize

            self._tracks[tid]["embedding"] = blended
            self._tracks[tid]["centroid"] = detections[di]["centroid"]
            self._tracks[tid]["bbox"] = detections[di]["bbox"]
            self._tracks[tid]["missing"] = 0
            results.append({**detections[di], "track_id": tid})

            score_matrix[ti, :] = -np.inf
            score_matrix[:, di] = -np.inf
            emb_sim[ti, :] = -np.inf
            emb_sim[:, di] = -np.inf

        # New tracks for unmatched detections
        for di, det in enumerate(detections):
            if di not in used_dets:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {
                    "embedding": det["embedding"],
                    "centroid": det["centroid"],
                    "bbox": det["bbox"],
                    "missing": 0,
                }
                results.append({**det, "track_id": tid})

        # Increment missing for unmatched tracks
        for tid in track_ids:
            if tid not in used_tracks:
                self._tracks[tid]["missing"] += 1

        self._purge_stale()
        return results

    def _purge_stale(self):
        self._tracks = {
            tid: info for tid, info in self._tracks.items()
            if info["missing"] <= self.max_missing_frames
        }

    def reset(self):
        """Reset tracker state (call between scenes)."""
        self._tracks.clear()
        self._next_id = 0

    def get_active_tracks(self):
        """Return currently active track IDs and their info."""
        return {tid: info for tid, info in self._tracks.items() if info["missing"] == 0}


def track_faces_in_video(video_path, face_app=None, sample_fps=5):
    """
    Run face tracking on a video file using InsightFace.

    Args:
        video_path: Path to video file.
        face_app: Pre-initialized FaceAnalysis app (created if None).
        sample_fps: Frames per second to sample (lower = faster).

    Returns:
        dict with:
            - "tracks": {track_id: [{"frame", "bbox", "centroid", "det_score"}]}
            - "total_frames_sampled": int
            - "unique_faces": int
            - "embeddings": {track_id: [512-dim embedding]} (last seen embedding per track)
    """
    if face_app is None:
        face_app = _build_face_app()

    tracker = FaceTracker()
    all_track_data = defaultdict(list)
    track_embeddings = {}

    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(1, int(video_fps / sample_fps))
    frame_idx = 0
    sampled = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            faces = face_app.get(frame)

            detections = []
            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                emb = face.normed_embedding  # already L2-normalized
                detections.append({
                    "bbox": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                    "centroid": (int(cx), int(cy)),
                    "embedding": emb,
                    "det_score": float(face.det_score),
                })

            tracked = tracker.update(detections)
            for t in tracked:
                tid = t["track_id"]
                all_track_data[tid].append({
                    "frame": frame_idx,
                    "bbox": t["bbox"],
                    "centroid": t["centroid"],
                    "det_score": t["det_score"],
                })
                track_embeddings[tid] = t["embedding"]

            sampled += 1
        frame_idx += 1

    cap.release()

    # Filter out noise tracks: faces appearing in fewer than 2 sampled frames
    # are likely background/fleeting faces, not meaningful characters
    min_detections = 2
    stable_tracks = {tid: frames for tid, frames in all_track_data.items()
                     if len(frames) >= min_detections}
    stable_embeddings = {tid: emb.tolist() for tid, emb in track_embeddings.items()
                         if tid in stable_tracks}

    return {
        "tracks": stable_tracks,
        "total_frames_sampled": sampled,
        "unique_faces": len(stable_tracks),
        "embeddings": stable_embeddings,
    }


def track_all_scenes(video_dir, scenes_data, sample_fps=5):
    """
    Run face tracking on all scenes in a task.

    Args:
        video_dir: Path to directory containing scene_XXX.mp4 files.
        scenes_data: The scenes.json data dict.
        sample_fps: Sampling rate.

    Returns:
        dict: {scene_num: tracking_result}
    """
    video_dir = Path(video_dir)
    results = {}

    # Build app once, reuse across all scenes
    logger.info("Loading InsightFace model (SCRFD + ArcFace)...")
    face_app = _build_face_app()

    for scene in scenes_data["scenes"]:
        sn = scene["scene"]
        vpath = video_dir / f"scene_{sn:03d}.mp4"
        if not vpath.exists():
            continue

        logger.info(f"  Tracking faces in scene {sn}...")
        results[sn] = track_faces_in_video(vpath, face_app=face_app, sample_fps=sample_fps)
        logger.info(f"    -> {results[sn]['unique_faces']} unique face(s) tracked")

    return results


def save_tracking_results(results, output_path):
    """Save tracking results to JSON (without embeddings for readability)."""
    clean = {}
    for sn, data in results.items():
        clean[str(sn)] = {
            "unique_faces": data["unique_faces"],
            "total_frames_sampled": data["total_frames_sampled"],
            "tracks": {str(tid): frames for tid, frames in data["tracks"].items()},
        }
    with open(output_path, "w") as f:
        json.dump(clean, f, indent=2)
    logger.info(f"Tracking results saved to {output_path}")


def match_faces_across_scenes(scene_results, sim_threshold=0.5):
    """
    Match face identities across different scenes using embeddings.

    Returns:
        dict: global_id -> [(scene_num, track_id), ...]
    """
    all_faces = []  # (scene_num, track_id, embedding)
    for sn, data in scene_results.items():
        for tid_str, emb_list in data.get("embeddings", {}).items():
            emb = np.array(emb_list)
            all_faces.append((sn, int(tid_str), emb))

    if not all_faces:
        return {}

    # Cluster by embedding similarity
    global_id = 0
    assigned = {}  # (scene, track) -> global_id
    global_map = defaultdict(list)  # global_id -> [(scene, track)]

    for sn, tid, emb in all_faces:
        key = (sn, tid)
        if key in assigned:
            continue

        # Check against existing global identities
        best_gid = None
        best_sim = -1
        for gid, members in global_map.items():
            for m_sn, m_tid in members:
                m_emb = None
                for s, t, e in all_faces:
                    if s == m_sn and t == m_tid:
                        m_emb = e
                        break
                if m_emb is not None:
                    sim = _cosine_sim(emb, m_emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_gid = gid

        if best_sim >= sim_threshold and best_gid is not None:
            assigned[key] = best_gid
            global_map[best_gid].append(key)
        else:
            assigned[key] = global_id
            global_map[global_id].append(key)
            global_id += 1

    return dict(global_map)
