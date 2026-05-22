# LipSync Pipeline - Module Walkthrough

## Architecture Overview

```
SyncPipeline.py  (orchestrator - runs all phases sequentially, supports resume)
    |
    ├── splitting/          Phase 1-3: Video & Audio Splitting
    ├── analysis/           Phase 4-6: Scene Intelligence
    ├── sync_api/           Phase 7:   Lip-Sync via sync.so API
    ├── tracking/           (NEW) Face Tracking Module
    └── output/             Phase 8:   Stitching & Reporting
```

---

## Phase-by-Phase Modules

### Phase 1 - Setup
**File:** `SyncPipeline.py` + `utils.py`
- Creates timestamped task folder (`tasks/task_MM-DD_HH-MM/`)
- Copies input video + dubbed audio into `inputs/`
- Initializes `state.json` for resume support

### Phase 2 - Video Splitting (Content + Adaptive Detection)
**File:** `splitting/video_splitter.py`
- Uses **PySceneDetect** with two detectors:
  - `ContentDetector` - detects hard cuts via frame difference
  - `AdaptiveDetector` - detects gradual transitions (fades, dissolves)
- Outputs individual scene clips as `scene_001.mp4`, `scene_002.mp4`, etc.
- Saves `scenes.json` with timestamps and durations

### Phase 2b - Face-Based Split Refinement
**File:** `splitting/face_splitter.py`
- Refines long scenes (>3s) using **MediaPipe Face Landmarker**
- Detects dominant face position shifts (person change / camera pan)
- Sub-splits scenes where face position shifts beyond threshold (0.25)
- Re-exports video clips and updates `scenes.json`

### Phase 3 - Audio Splitting
**File:** `splitting/audio_splitter.py`
- Converts dubbed audio to 16kHz WAV (required for dialogue detection)
- Splits audio to match scene boundaries using **FFmpeg**

### Phase 4 - Dialogue Detection
**File:** `analysis/dialogue.py`
- Analyzes each scene's audio for speech presence
- Returns set of scene numbers that contain dialogue
- Only dialogue scenes proceed to lip-sync (saves API cost)

### Phase 5 - Complexity & Obstruction Analysis
**File:** `5 - Complexity & Obstruction Analysis`
- **Face Count & Complexity** (MediaPipe Face Landmarker):
  - Samples frames, counts faces per scene
  - Flags "complex" scenes (multiple faces, face size changes)
- **Person Count** (YOLOv8 fallback):
  - Uses YOLOv8n to detect persons when faces aren't visible
  - Determines `effective_max_persons`
- **Lip Obstruction** (MediaPipe landmarks):
  - Checks if lip landmarks are partially occluded

### Phase 6 - Scene Analysis JSON
**File:** `analysis/scene_analysis.py`
- Consolidates all analysis into `SceneAnalysis.json`
- Per scene: `haveAudio`, `personSpeaking`, `multiplePersonInScene`, `haveObstructionInLip`, `isComplexScene`
- These flags directly control **sync.so API parameters**

### Phase 7 - sync.so API Processing
**Files:** `sync_api/processor.py`, `sync_api/client.py`, `sync_api/s3.py`

**How analysis maps to API parameters:**

| Scene Flag              | sync.so API Parameter              | Effect                              |
|-------------------------|------------------------------------|--------------------------------------|
| `isComplexScene`        | `reasoning: true`                  | Enhanced processing for hard scenes  |
| `haveObstructionInLip`  | `occlusion_detection_enabled: true`| Handles hand/object over mouth       |
| `multiplePersonInScene` | `active_speaker_detection: auto`   | Detects who is speaking              |

- Uploads scene video+audio to **S3** (presigned URLs)
- Submits jobs to sync.so API in parallel (3 concurrent)
- Polls until complete, downloads synced video
- Non-dialogue scenes are copied as-is (no API call)

### Phase 8 - Final Stitching
**File:** `output/stitcher.py`
- Concatenates all scenes (synced + unmodified) using FFmpeg
- Produces final output video

---

## Face Tracking Module (NEW - Phase 9)
**File:** `tracking/face_tracker.py`

Purpose: Track faces across frames within each scene for:
- Active speaker verification (confirm which face is speaking)
- Face consistency checks across scene boundaries
- Input to future per-character dubbing workflows

See `tracking/` directory for implementation.

---

## Config & Utilities
- `config.py` - All constants: API endpoints, model paths, landmark indices, thresholds
- `utils.py` - Task folder management, state save/load, FFmpeg helpers, progress bars

## Key Dependencies
- PySceneDetect, MediaPipe, YOLOv8 (ultralytics), OpenCV, FFmpeg, boto3, requests
