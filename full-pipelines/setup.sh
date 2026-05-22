#!/bin/bash
# LipSync Pipeline - One-time setup
# Run: bash setup.sh

set -e

echo "=================================="
echo "  LipSync Pipeline Setup"
echo "=================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 not found. Install Python 3.10+"
    exit 1
fi

echo "[1/3] Installing Python dependencies..."
pip install -r requirements.txt

echo "[2/3] Checking FFmpeg..."
if ! command -v ffmpeg &> /dev/null; then
    echo "[!] FFmpeg not found. Installing..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y ffmpeg
    elif command -v brew &> /dev/null; then
        brew install ffmpeg
    else
        echo "[ERROR] Please install FFmpeg manually: https://ffmpeg.org/download.html"
        exit 1
    fi
fi
echo "  FFmpeg: $(ffmpeg -version | head -1)"

echo "[3/3] Checking model files..."
if [ ! -f "models/face_landmarker_v2_with_blendshapes.task" ]; then
    echo "  Downloading MediaPipe Face Landmarker model..."
    mkdir -p models
    curl -L -o models/face_landmarker_v2_with_blendshapes.task \
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
fi
echo "  Model file: OK"

# YOLO model downloads automatically on first run

echo ""
echo "=================================="
echo "  Setup complete!"
echo "=================================="
echo ""
echo "Usage:"
echo "  export SYNC_API_KEY='your_sync_so_api_key'"
echo "  python3 pipeline.py"
echo "  python3 pipeline.py --video input.mp4 --audio dubbed.wav"
echo ""
