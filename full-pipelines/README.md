# LipSync Pipeline

End-to-end automated lip-sync dubbing pipeline. Takes a video + dubbed audio and produces a lip-synced output video.

## Quick Start

```bash
git clone https://github.com/FrameX-AI-Studio/full-pipelines.git
cd full-pipelines
bash setup.sh
export SYNC_API_KEY="your_sync_so_api_key"
python3 pipeline.py
```

## Requirements

- Python 3.10+
- FFmpeg
- sync.so API key ([get one here](https://sync.so))

## Usage

### Interactive mode
```bash
python3 pipeline.py
# Creates a task folder, asks you to place video + audio in inputs/
```

### Direct mode
```bash
python3 pipeline.py --video input_video.mp4 --audio dubbed_audio.wav
```

## Pipeline Steps

| Step | What | Tech |
|------|------|------|
| 1 | Split video into scenes | PySceneDetect + FFmpeg |
| 2 | Convert audio to 16kHz mono + split | FFmpeg |
| 3 | Detect dialogue in each scene | Silero VAD |
| 4 | Analyze scenes (faces, complexity, obstruction) | MediaPipe + YOLOv8 |
| 5 | Generate SceneAnalysis.json | - |
| 6 | Lip-sync via sync.so API (3 parallel) | sync.so REST API |
| 7 | Stitch all scenes into final output | FFmpeg |

## Output Structure

```
task_DD-MM_HH-MM/
├── inputs/              ← your video + audio
└── outputs/
    ├── SplittedScenes/
    │   ├── scenes.json  ← timestamps
    │   ├── videos/      ← split scenes
    │   └── audio/       ← split audio (16kHz mono)
    ├── SceneAnalysis.json
    ├── outputScenes/    ← processed scenes
    ├── final_output.mp4 ← final result
    └── report.json      ← timing report
```

## SceneAnalysis.json Format

Each scene gets these flags:
```json
{
  "scene": 1,
  "start_time": "00:00:00.000",
  "end_time": "00:00:05.000",
  "duration_seconds": 5.0,
  "haveAudio": true,
  "personSpeaking": true,
  "multiplePersonInScene": false,
  "haveObstructionInLip": false,
  "isComplexScene": false
}
```

Only scenes with `haveAudio: true` AND `personSpeaking: true` are sent to sync.so API. The rest pass through unchanged.
