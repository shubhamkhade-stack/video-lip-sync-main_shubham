# Setup, Run & Integration Guide — LipSync Pipeline (VS Code)

> Complete instructions to set up your colleague's `video-lip-sync-main`
> pipeline in VS Code, install the two corrected code files, and run it.
>
> **Code files this guide installs:**
> 1. `sync_client.py` — hybrid `sync-3` / `lipsync-2-pro` model routing
> 2. `lip_sync_scorer.py` — corrected scorer (resolution-matched + real LSE-C/LSE-D)

---

## 0. What You're Working With

```
video-lip-sync-main/
└─ full-pipelines/            <- you work HERE
   ├─ Sync3Pipeline.py        whole-video entry point  (start here)
   ├─ SyncPipeline.py         scene-split entry point
   ├─ interactive_sync.py     scene-by-scene entry point
   ├─ config.py               API/S3/model settings
   ├─ requirements.txt
   ├─ setup.sh
   ├─ sync_api/
   │  ├─ client.py            <- replace logic with sync_client.py
   │  ├─ processor.py         <- one edit to call the router
   │  └─ s3.py
   └─ scoring/
      └─ lip_sync_scorer.py   <- REPLACE with the corrected file
```

---

## 1. Open the Project in VS Code

1. Unzip `video-lip-sync-main__1_.zip`.
2. VS Code → **File → Open Folder** → select the **`full-pipelines`** folder
   (open this folder directly so imports like `from config import ...` resolve).
3. Install VS Code extensions (Extensions panel, `Ctrl+Shift+X`):
   - **Python** (Microsoft)
   - **Pylance**
4. Open the integrated terminal: **Terminal → New Terminal** (`Ctrl+` `` ` ``).

---

## 2. Prerequisites

| Need | Check command | If missing |
|---|---|---|
| Python 3.10+ | `python3 --version` | install from python.org |
| FFmpeg | `ffmpeg -version` | `setup.sh` installs it, or install manually |
| Sync.so API key | — | get from the sync.so dashboard |
| AWS S3 access | `aws sts get-caller-identity` | needed — pipeline uploads to S3 |

---

## 3. Create a Virtual Environment (recommended)

In the VS Code terminal, inside `full-pipelines/`:
```bash
python3 -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1
```
Then tell VS Code to use it: `Ctrl+Shift+P` → **Python: Select Interpreter** →
pick the `.venv` one.

---

## 4. Install Dependencies

```bash
bash setup.sh                 # installs requirements.txt + ffmpeg + face model
pip install boto3             # MISSING from requirements.txt — install manually
```

`requirements.txt` is missing `boto3`; the S3 step fails without it. Optionally
add a line `boto3>=1.28.0` to `requirements.txt` so it's not forgotten.

If `setup.sh` did not download the face model, get it manually:
```bash
mkdir -p models
curl -L -o models/face_landmarker_v2_with_blendshapes.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
```

---

## 5. Configure Credentials (use a `.env` file)

Create a file named `.env` in `full-pipelines/`:
```
SYNC_API_KEY=your_sync_so_api_key
PIPELINE_S3_USER=yourname
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_DEFAULT_REGION=us-east-1
SYNCNET_DIR=/path/to/syncnet_python      # optional — for the real sync score
```

Load it before running (each new terminal):
```bash
# macOS / Linux
set -a; source .env; set +a
# Windows PowerShell
Get-Content .env | ForEach-Object { if ($_ -match '^(.+?)=(.+)$') { [Environment]::SetEnvironmentVariable($matches[1],$matches[2]) } }
```

Add `.env` to `.gitignore` so keys are never committed.

---

## 6. Edit `config.py` — the S3 Bucket

`config.py` has `S3_BUCKET = "framexstudio-files"` hardcoded. If you do **not**
have access to that bucket, change it to your own:
```python
S3_BUCKET = "your-own-s3-bucket"
```

---

## 7. Install Code File 1 — `sync_client.py` (hybrid routing)

1. Copy `sync_client.py` into `full-pipelines/sync_api/`.
2. Open `sync_api/processor.py`. Find `process_single_scene()` — it currently
   calls `sync_create_job(...)` with the fixed model from `config.py`.
3. Replace that call so it uses the router. Minimal edit:

   At the top of `processor.py`:
   ```python
   from sync_api.sync_client import process_scene as hybrid_process_scene
   ```
   Inside `process_single_scene()`, replace the `sync_create_job` + `sync_poll`
   + `sync_download` block with:
   ```python
   scene_meta = {
       "scene": scene_num,
       "multi_person": multi_person,
       "has_occlusion": is_obstructed,
       "is_complex": is_complex,
       "is_closeup": scene.get("is_closeup", False),
       "extreme_angle": scene.get("extreme_angle", False),
   }
   rec = hybrid_process_scene(
       scene_meta, video_url, audio_url, api_key,
       str(output_path), temperature=0.6,
   )
   return scene_num, rec["status"] == "COMPLETED", rec.get("error")
   ```
   This auto-routes each scene to `sync-3` or `lipsync-2-pro`.

---

## 8. Install Code File 2 — `lip_sync_scorer.py` (corrected scorer)

1. **Replace** `full-pipelines/scoring/lip_sync_scorer.py` with the corrected file.
2. It needs SyncNet for the real LSE-C/LSE-D score. Install SyncNet:
   ```bash
   git clone https://github.com/joonson/syncnet_python.git
   cd syncnet_python && pip install -r requirements.txt
   bash download_model.sh
   ```
   Then set `SYNCNET_DIR` (Step 5) to that folder.
3. Without `SYNCNET_DIR`, the scorer still runs but reports **visual quality
   only** and clearly says sync was not measured — no false numbers.

Score one scene directly:
```bash
python scoring/lip_sync_scorer.py original_scene.mp4 synced_scene.mp4
```

---

## 9. Run the Pipeline

**⚠️ The README says `python3 pipeline.py` — that file does NOT exist.** Use:

| Command | What it does |
|---|---|
| `python3 Sync3Pipeline.py --video in.mp4 --audio dub.wav` | whole video → `sync-3`. **Start here.** |
| `python3 SyncPipeline.py` | scene-split, resumable |
| `python3 interactive_sync.py` | scene-by-scene |

Example:
```bash
python3 Sync3Pipeline.py --video "NARESH (1).mp4" --audio vasuki_hindi_sts_final.wav
```
Or run `python3 Sync3Pipeline.py` with no args for interactive mode (it makes a
task folder and asks you to drop the video + audio into `inputs/`).

You can run/debug from VS Code: open the entry file, press **F5**, choose
"Python File" (the `.env` is picked up if `python.envFile` is set — VS Code
does this by default for a `.env` in the workspace root).

---

## 10. The Correct Order (do NOT skip)

Running the pipeline on the current audio wastes API money — your own reports
marked the dubs REJECT. Correct sequence:

```
1. Re-fit the dub audio timing   (NARESH_fix_instructions.md / VASUKI_fix_instructions.md)
2. Verify it passes              (event parity / drift / pauses)
3. Install both code files       (Steps 7-8 above)
4. Run the pipeline              (Step 9)
5. Score with the fixed scorer   (Step 8) and compare to the original baseline
```

Cost reminder: `sync-3` ≈ $0.133/sec → ~$17 per 130 s pass per speaker. Do not
run it on audio that fails verification.

---

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: boto3` | `pip install boto3` (Step 4) |
| `ModuleNotFoundError: config` | open the `full-pipelines` folder directly in VS Code |
| `NoCredentialsError` (S3) | set AWS_* env vars (Step 5) |
| `Access Denied` on S3 bucket | change `S3_BUCKET` in `config.py` (Step 6) |
| sync score says "NOT MEASURED" | install SyncNet + set `SYNCNET_DIR` (Step 8) |
| job status `REJECTED` | bad input — check the API error; usually no clear face / static face |
| scorer gives a low score but video looks fine | confirm `resolution_matched` is handled — the fixed scorer resizes; the old one did not |

---

## 12. File Placement Summary

```
full-pipelines/
├─ .env                          (you create — Step 5, gitignore it)
├─ config.py                     (edit S3_BUCKET — Step 6)
├─ sync_api/
│  ├─ sync_client.py             (NEW — Step 7)
│  └─ processor.py               (one edit — Step 7)
└─ scoring/
   └─ lip_sync_scorer.py         (REPLACED — Step 8)
```
