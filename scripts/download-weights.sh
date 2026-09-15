#!/usr/bin/env bash
# Download the official NVIDIA NVFP4 checkpoint (~124 GiB) used by serve.sh.
# Resumable — safe to re-run if the connection drops.
#
#   scripts/download-weights.sh
#
# Needs ~130 GB free on the filesystem holding MODEL_HOST.
set -euo pipefail

MODEL=nvidia/Qwen3.8-Flash-Next-NVFP4
REVISION=fc694b54fb0174e0913e6adf86691ef85a4ead47
IMAGE="${IMAGE:-vllm/vllm-openai@sha256:a551e05307cd2e0092139d84db32af9c97e67d2eeeff072d21e429131d8c23f0}"
MODEL_HOST="${MODEL_HOST:-/home/andre/local-ai/models/Qwen3.8-Flash-Next-NVFP4}"
mkdir -p "$MODEL_HOST"

# hf authenticates via HF_TOKEN (or the older HUGGING_FACE_HUB_TOKEN name).
# docker -e NAME (no value) copies the host env var into the container.
TOKEN_ARGS=()
if [ -n "${HF_TOKEN:-}" ]; then
  TOKEN_ARGS+=(-e HF_TOKEN)
elif [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  TOKEN_ARGS+=(-e HUGGING_FACE_HUB_TOKEN -e HF_TOKEN="$HUGGING_FACE_HUB_TOKEN")
else
  echo ">> no HF_TOKEN in the environment; Hub will rate-limit unauthenticated downloads"
fi

echo ">> downloading $MODEL into $MODEL_HOST (resumable)"
# HF_HUB_DISABLE_XET=1: the Xet backend stalled on some Spark setups; plain HTTPS
# is reliable and saturates the link.
docker run --rm --name qwen38-dl \
  -e HF_HUB_DISABLE_XET=1 \
  "${TOKEN_ARGS[@]}" \
  -v "$MODEL_HOST:/models" --entrypoint bash "$IMAGE" \
  -c "hf download '$MODEL' --revision '$REVISION' --local-dir /models --max-workers 8 && hf cache verify '$MODEL' --revision '$REVISION' --local-dir /models --fail-on-missing-files"

echo ">> Verified pinned checkpoint. Set IMAGE to the reviewed local image before launching scripts/serve.sh."
