#!/bin/bash
# Starts the full Rocky stack on the pod. Open port 7860 through the RunPod proxy afterwards.
set -e

pgrep -f "ollama serve" > /dev/null || (OLLAMA_HOST=127.0.0.1 ollama serve > /workspace/ollama.log 2>&1 &)

cd /workspace/Applio
. .venv/bin/activate
pgrep -f "app.py --port 6969" > /dev/null || (python app.py --port 6969 --server-name 127.0.0.1 > /workspace/applio.log 2>&1 &)
deactivate

export ROCKY_HOST=0.0.0.0
export ROCKY_CHAT_MODEL=qwen2.5:14b
export ROCKY_VISION_MODEL=gemma3:12b
export ROCKY_WHISPER=large-v3
export ROCKY_VISION_KEEPALIVE=30m
export ROCKY_APPLIO_DIR=/workspace/Applio
export ROCKY_MODEL_GLB=/workspace/assets/Rocky_realtime.glb
# set these two to match your voice model files in /workspace/Applio/logs/<name>/
export ROCKY_RVC_NAME=${ROCKY_RVC_NAME:-rocky_private}
export ROCKY_RVC_PTH=${ROCKY_RVC_PTH:-rocky_private_200e_9400s.pth}

cd /workspace/rocky-ai
. .venv/bin/activate
echo "Waiting for Applio to come up (first start downloads its pretrained files, give it a minute)..."
until curl -s -o /dev/null http://127.0.0.1:6969/; do sleep 3; done
echo "Rocky is starting. Open port 7860 via the pod's Connect menu."
python rocky_chat.py
