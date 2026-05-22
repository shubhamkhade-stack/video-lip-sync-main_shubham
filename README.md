# LipSync Pipeline

Automated lip-sync pipeline for dubbing videos from one language to another. Takes an original video and a pre-dubbed audio file (same duration, target language, original actor's cloned voice) and produces a lip-synced video where the actor's mouth movements match the new language dialogue.

---

## Inputs

| Input | Description |
|-------|-------------|
| **Original Video** | Source language video (e.g., `Chiranjeevi_173sec_Telugu.mp4`, 173s) |
| **Dubbed Audio** | Pre-dubbed target language audio, same duration, with cloned voice already applied (e.g., `Chiranjeevi_173sec_Hindi.wav`, 173s) |

> Both inputs must be of the **same duration**. Voice cloning and dubbing are handled externally before this pipeline runs.

## Output

| Output | Description |
|--------|-------------|
| **Lip-Synced Dubbed Video** | Final video with lip movements matching target language dialogue, original BG/SFX preserved (`Chiranjeevi_173sec_Hindi_LipSynced.mp4`, 173s) |

---

## Pipeline Overview

```
Original Video (173s) ──> Scene Split (PySceneDetect) ──> 42 video clips
                               |
Dubbed Audio (173s)   ──> Audio Split (FFmpeg, same timestamps) ──> 42 audio clips
                               |
                        Dialogue Detection (Silero VAD)
                         /                \
                 14 no-dialogue       28 with dialogue
                      |                     |
                      |              Vocal Separation (Demucs)
                      |               /              \
                      |        Original BG/SFX    Dubbed vocals
                      |          (preserve)        (for sync)
                      |               |                |
                      |        PREPROCESSING & SCENE ANALYSIS
                      |        ┌─────────────────────────────┐
                      |        | Face Detection (MediaPipe)   |
                      |        | Complexity Scoring            |
                      |        | Scene Classification (A/B/C/D)|
                      |        └──────────────┬───────────────┘
                      |                       |
                      |             SYNC.SO LIP-SYNC API
                      |             (per-group API flags)
                      |                       |
                      |                28 synced videos
                      |                       |
                      |             Audio Reconstruction
                      |             (dubbed vocals + original BG/SFX)
                      |                       |
               Swap audio only                |
               (no lip-sync needed)           |
                      |                       |
                      └──── Final Stitch (42 scenes) ────> OUTPUT
                                                     173s Lip-Synced Video
```

---

## Pipeline Steps

### Step 1: Scene Splitting — Video

**Script:** `preprocessor/clipToScenes/scripts/scene_splitter.py`
**Status:** Done

Detects and splits the original video into individual scenes based on visual content changes using **PySceneDetect**.

| Setting | Value |
|---------|-------|
| Detector | ContentDetector (default) or AdaptiveDetector |
| Content threshold | 27.0 (lower = more cuts) |
| Adaptive threshold | 3.0 |
| Min scene length | 15 frames |
| Split method | FFmpeg stream copy (lossless, no re-encoding) |

- **Input:** Original video
- **Output:**
  - 42 scene video clips (`scene_001.mp4` ... `scene_042.mp4`)
  - `scenes.json` — start/end timestamps per scene

---

### Step 2: Scene Splitting — Dubbed Audio

**Script:** `preprocessor/clipToScenes/scripts/audio_splitter.py`
**Status:** Done

Splits the pre-dubbed audio into per-scene clips using the **exact same timestamps** from `scenes.json` (Step 1). This ensures each audio clip maps 1:1 to its corresponding video scene.

| Setting | Value |
|---------|-------|
| Tool | FFmpeg |
| Method | Stream copy first, fallback to re-encode (44.1kHz, 2ch PCM) |
| Verification | Duration mismatch tolerance of 0.1s |

- **Input:** Dubbed audio WAV + `scenes.json`
- **Output:** 42 audio clips matching video scene durations

---

### Step 3: Dialogue Detection

**Script:** `preprocessor/dialogueDub/scripts/1_detect_dialogue.py`
**Status:** Done

Runs **Silero VAD** (Voice Activity Detection) on each scene's audio to classify scenes as dialogue or non-dialogue.

| Setting | Value |
|---------|-------|
| Model | Silero VAD (via torch.hub) |
| Speech probability threshold | 0.5 |
| Min speech duration | 0.5 seconds |
| Audio format | 16kHz mono WAV (extracted via FFmpeg) |
| Processing window | 512 samples (32ms) per chunk |

- **Input:** 42 video scenes (audio extracted internally)
- **Output:**
  - `scenesWithDialogue/` — scenes with speech (28 scenes) → proceed to lip-sync
  - `scenesWithoutDialogues/` — music/SFX only (14 scenes) → skip lip-sync entirely

---

### Step 4: Vocal Separation

**Script:** `preprocessor/dialogueDub/scripts/2_separate_vocals.py`
**Status:** Done

Separates audio into vocal and non-vocal tracks using **Meta's Demucs** model. Applied to original video audio to extract the background score/SFX that will be preserved in the final output.

| Setting | Value |
|---------|-------|
| Model | htdemucs |
| Mode | `--two-stems vocals` (vocals vs everything else) |
| Audio format | 44.1kHz stereo WAV |

- **Input:** 28 dialogue scene videos (original audio)
- **Output per scene:**

| Track | Purpose |
|-------|---------|
| `vocals.wav` | Original vocals (reference only) |
| `non_vocals.wav` | Background score + SFX — reused in final output |

---

### Step 5: Transcription (Reference)

**Script:** `preprocessor/dialogueDub/scripts/3_transcribe.py`
**Status:** Done

Generates text transcripts for QA and reference using **OpenAI Whisper**.

| Setting | Value |
|---------|-------|
| Model | Whisper medium (~1.5GB) |
| Language | Configured per project (e.g., `"te"` for Telugu) |
| Task | Transcribe (not translate) |

- **Input:** Vocal tracks from separated audio
- **Output:** `.txt` (plain text) and `.json` (with per-segment timestamps)

---

### Step 6: Preprocessing — Scene Analysis & Classification

This is the core preprocessing step that determines **which sync.so API options** to use for each scene. It analyzes visual complexity and classifies scenes into groups with different API flag combinations.

#### 6a. Complexity Detection

**Script:** `preprocessor/sceneAnalysis/detect_complexity.py`
**Status:** Done

Uses **MediaPipe FaceLandmarker** (468 facial landmarks) and **OpenCV** to analyze each dialogue scene and compute a complexity score based on 5 signals:

| Signal | What It Detects | How It's Measured | Scoring |
|--------|----------------|-------------------|---------|
| **Multiple faces** | More than one person on screen | Face count per sampled frame | >50% frames: +0.30, >20%: +0.15 |
| **Small face / wide shot** | Face too small for reliable lip-sync | Face bounding box as % of frame area | <2% area: +0.25, <5% area: +0.10 |
| **Fast head motion** | Rapid movement making lip tracking harder | Landmark displacement between consecutive frames (pixels) | >15px avg: +0.20, >8px avg: +0.10 |
| **Side/angled face** | Partial lip visibility from head rotation | Yaw angle estimated from nose + eye landmarks | >40% frames with yaw >30°: +0.20, avg yaw >25°: +0.10 |
| **Speaker switches** | Different people speaking within same scene | Face count changes across frames | >30% frames with change: +0.15 |

**Detection configuration:**

| Setting | Value |
|---------|-------|
| Tool | MediaPipe FaceLandmarker + OpenCV |
| Sample rate | Every 3rd frame (configurable) |
| Min detection confidence | 0.4 |
| Max faces tracked | 5 |
| Frame resize | 1920px max width |
| Key landmarks | Nose (1), Left Eye (263), Right Eye (33), Chin (152), Mouth corners (61, 291) |
| **Complexity threshold** | **0.3** (score >= 0.3 = "complex") |

Scores are additive — a scene with small face (+0.10) and fast head motion (+0.20) gets a total score of 0.30, marking it as complex.

- **Input:** 28 dialogue scene videos
- **Output:** `complexity_report.json` with per-scene scores, signals, and reasons

**Example output entry:**
```json
{
  "video": "scene_029.mp4",
  "complexity_score": 0.55,
  "complex": true,
  "signals": {
    "avg_face_count": 0.64,
    "max_face_count": 2,
    "avg_face_size": 0.1322,
    "avg_yaw_angle": 40.7,
    "high_yaw_ratio": 0.85,
    "avg_displacement": 81.45,
    "speaker_switch_ratio": 0.34
  },
  "reasons": [
    "fast_head_motion(81.5px)",
    "frequent_side_angle(0.85)",
    "frequent_speaker_switches(0.34)"
  ]
}
```

#### 6b. Scene Classification

**Script:** `preprocessor/sceneAnalysis/classify_scenes.py`
**Status:** Done

Combines complexity scores (and obstruction scores if available) to classify each scene into one of four groups. Each group maps to a specific set of **sync.so API flags**:

| Group | Condition | sync.so API Flags | Cost Impact |
|-------|-----------|-------------------|-------------|
| **A — Simple** | Low complexity, no obstructions | Base call (no extra flags) | Lowest |
| **B — Obstructed** | Low complexity, lips obstructed | `detect_obstructions: true` | Medium |
| **C — Complex** | High complexity, no obstructions | `reasoning: true` | Medium-High |
| **D — Both** | High complexity + obstructed | `reasoning: true` + `detect_obstructions: true` | Highest |

**Why this matters:** sync.so charges more for `reasoning` mode and obstruction detection. By analyzing scenes upfront, we only enable expensive features where they're actually needed — saving cost without sacrificing quality.

- **Input:** `complexity_report.json` (+ optional `obstruction_report.json`)
- **Output:** `jobs.json` — batch job definitions with per-scene API flags, video/audio paths, and classification group

**Example jobs.json entry:**
```json
{
  "scene_number": 29,
  "group": "C",
  "video_path": "/path/to/scene_029.mp4",
  "audio_path": "/path/to/scene_029_dubbed.wav",
  "output_filename": "scene_029_synced.mp4",
  "model": "lipsync-2-pro",
  "sync_mode": "cut_off",
  "reasoning": true
}
```

---

### Step 7: Lip-Sync Generation

**Script:** `Automate-SyncSo/syncso_automate_advanced.py`
**Status:** Done

Sends each dialogue scene to the **sync.so REST API v2** for lip-sync generation, using the per-scene API flags determined in Step 6.

| Setting | Value |
|---------|-------|
| API | sync.so v2 (`https://api.sync.so/v2`) |
| Model | `lipsync-2-pro` (default) |
| Sync mode | `cut_off` (default; also supports `loop`, `bounce`, `silence`, `remap`) |
| Poll interval | 10 seconds |
| Max wait | 600 seconds (10 min) per job |
| Selective processing | `--scenes "1,3,5"` flag to process specific scenes only |

**Per-group API behavior:**

| Group | API Payload |
|-------|-------------|
| A (Simple) | Video + audio, base model |
| B (Obstructed) | + `occlusion_detection_enabled: true` |
| C (Complex) | + `reasoning: true` |
| D (Both) | + `reasoning: true` + `occlusion_detection_enabled: true` |

**Process:**
1. Upload video + audio to sync.so (local files get uploaded as assets)
2. Submit generation job with group-specific flags
3. Poll for completion
4. Download synced video
5. Generate batch report (`.docx`) via `batch_report.py`

- **Input:** `jobs.json` (28 jobs with video/audio paths and API flags)
- **Output:** 28 lip-synced video clips

**CLI usage:**
```bash
# Batch process all scenes
python syncso_automate_advanced.py batch --jobs-file jobs.json --output-dir synced_outputs

# Process only specific scenes
python syncso_automate_advanced.py batch --jobs-file jobs.json --scenes "1,3,5" --output-dir synced_outputs

# Single scene
python syncso_automate_advanced.py generate --video scene.mp4 --audio dubbed.wav --reasoning
```

---

---

## Tech Stack

| Category | Technology | Used In |
|----------|-----------|---------|
| Scene Detection | **PySceneDetect** (ContentDetector) | Video scene splitting |
| Video/Audio Processing | **FFmpeg** | Splitting, mixing, encoding, stitching |
| Voice Activity Detection | **Silero VAD** (PyTorch) | Dialogue vs non-dialogue classification |
| Vocal Separation | **Meta Demucs** (htdemucs) | Isolate vocals from BG/SFX |
| Speech-to-Text | **OpenAI Whisper** (medium) | Transcription for QA |
| Face Analysis | **MediaPipe FaceLandmarker** (468 landmarks) | Complexity detection (face count, size, yaw, motion, speaker switches) |
| Computer Vision | **OpenCV** | Frame extraction and processing for scene analysis |
| Lip-Sync Generation | **sync.so API v2** (lipsync-2-pro) | Cloud lip-sync with per-scene reasoning/obstruction flags |
| Reporting | **python-docx** | Batch processing reports |
| ML Backend | **PyTorch**, **torchaudio** | VAD + audio processing |

---

## Directory Structure

```
LipSync-Pipeline/
├── README.md
├── jobs.json                              <- sync.so batch job definitions
│
├── preprocessor/
│   ├── clipToScenes/
│   │   ├── scripts/
│   │   │   ├── scene_splitter.py          <- Step 1: PySceneDetect scene detection
│   │   │   └── audio_splitter.py          <- Step 2: FFmpeg audio splitting
│   │   └── outputs/
│   │       ├── *_scenes.json              <- scene timestamps
│   │       ├── splittedScenes/            <- video scene clips
│   │       └── splittedAudio/             <- dubbed audio scene clips
│   │
│   ├── dialogueDub/
│   │   ├── scripts/
│   │   │   ├── 1_detect_dialogue.py       <- Step 3: Silero VAD filtering
│   │   │   ├── 2_separate_vocals.py       <- Step 4: Demucs vocal separation
│   │   │   └── 3_transcribe.py            <- Step 5: Whisper transcription
│   │   └── outputs/
│   │       └── */
│   │           ├── 1_detect_dialogue/
│   │           │   ├── scenesWithDialogue/
│   │           │   └── scenesWithoutDialogues/
│   │           ├── 2_separated_audio/
│   │           └── 3_transcripts/
│   │
│   └── sceneAnalysis/
│       ├── detect_complexity.py            <- Step 6a: MediaPipe complexity scoring
│       ├── classify_scenes.py              <- Step 6b: A/B/C/D classification -> jobs.json
│       └── outputs/
│           └── complexity/
│               └── complexity_report.json
│
├── Automate-SyncSo/
│   ├── syncso_automate.py                 <- Step 7: basic sync.so automation
│   ├── syncso_automate_advanced.py        <- Step 7: advanced (group-aware) automation
│   ├── batch_report.py                    <- .docx report generation
│   └── synced_outputs/                    <- lip-synced results + reports
│
├── testing/
│   ├── detect_lip_obstruction.py          <- obstruction detection / post-sync QA
│   └── face_landmarker_v2_with_blendshapes.task
│
└── aws/                                   <- AWS CLI v2
```

---


