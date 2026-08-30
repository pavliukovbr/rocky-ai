#!/bin/bash
# Rocky AI cloud setup for a RunPod GPU pod (tested against the official PyTorch template).
# Run once from /workspace: bash rocky-ai/runpod/setup.sh
set -e

echo "== system packages =="
apt-get update -qq && apt-get install -y -qq ffmpeg espeak-ng libportaudio2 git curl > /dev/null

echo "== ollama =="
curl -fsSL https://ollama.com/install.sh | sh
(OLLAMA_HOST=127.0.0.1 ollama serve > /workspace/ollama.log 2>&1 &)
sleep 6
ollama pull qwen2.5:14b
ollama pull gemma3:12b

echo "== rocky server deps =="
cd /workspace/rocky-ai
python -m venv --system-site-packages .venv
. .venv/bin/activate
pip install -q -r requirements.txt
deactivate

echo "== applio =="
if [ ! -d /workspace/Applio ]; then
  git clone -q https://github.com/IAHispano/Applio /workspace/Applio
fi
cd /workspace/Applio
python -m venv --system-site-packages .venv
. .venv/bin/activate
pip install -q -r requirements.txt
pip install -q "transformers==4.44.2"
deactivate

echo "== voice model =="
mkdir -p /workspace/Applio/logs
if [ -d /workspace/assets ]; then
  RVC_DIR=$(ls -d /workspace/assets/*/ 2>/dev/null | head -1)
  if [ -n "$RVC_DIR" ]; then
    cp -r "$RVC_DIR" /workspace/Applio/logs/
    echo "voice model installed: $RVC_DIR"
  fi
else
  echo "NOTE: upload your assets to /workspace/assets first (voice model folder and the GLB), then rerun this block."
fi

echo
echo "Setup done. Start everything with: bash /workspace/rocky-ai/runpod/start.sh"
