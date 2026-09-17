#!/usr/bin/env bash
# Run the native image with GPU via device binds (no nvidia-container-toolkit).
# Host libcuda + TRITON_LIBCUDA_PATH are required; torch cublas is on LD_LIBRARY_PATH
# inside the image so it is not shadowed by /usr/local/cuda 13.0.0.
set -euo pipefail
IMAGE="${IMAGE:-qwen38-exl3-dflash2:1.5.0-native}"
NAME="${NAME:-qwen38-native-serve}"
MODELS="${MODELS:-/home/am/exl3-qwen38-dflash2/models}"
PORT="${PORT:-8889}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
exec docker run -d --name "$NAME" \
  --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm \
  -v /usr/lib/x86_64-linux-gnu/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro \
  -v /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:ro \
  -v "$MODELS":/models:ro \
  -e TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu \
  -p "${PORT}:8889" \
  --entrypoint python \
  "$IMAGE" \
  /opt/serve_openai.py --target /models/qwen38-27b-exl3 --draft /models/dflash2-exl3 \
    --host 0.0.0.0 --port 8889
