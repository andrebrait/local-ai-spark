#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-sha256:213e470219da6e21119f0a13a4df35f7e0d1f18ed2fb26e43f68c58965b30dfe}"
MODEL_HOST="${MODEL_HOST:-/home/andre/local-ai/models/Swift-Qwen3.8-27B-NVFP4}"
CACHE_HOST="${CACHE_HOST:-/home/andre/local-ai/cache}"
API_ENV_FILE="${API_ENV_FILE:-/home/andre/local-ai/secrets/api.env}"

exec docker create --name swift-qwen38-27b \
  --gpus all --init --restart unless-stopped --pull never \
  --memory 112g --memory-swap 112g \
  --cap-drop ALL --security-opt no-new-privileges:true \
  --network host --ipc private --shm-size 32g --ulimit memlock=-1:-1 \
  --env-file "$API_ENV_FILE" \
  --entrypoint vllm \
  -v "$MODEL_HOST:/models/swift-qwen38-27b:ro" \
  -v "$CACHE_HOST:/root/.cache" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUTE_DSL_ARCH=sm_121a -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_USE_DEEP_GEMM=0 \
  "$IMAGE" serve /models/swift-qwen38-27b \
  --served-model-name swift-qwen3.8-27b \
  --host 100.64.255.60 --port 8000 \
  --dtype bfloat16 --kv-cache-dtype fp8_e4m3 \
  --tensor-parallel-size 1 --max-model-len 262144 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --mm-encoder-tp-mode data --seed 0 \
  --gpu-memory-utilization 0.75 --max-num-seqs 32 \
  --max-num-batched-tokens 32768 --enable-chunked-prefill \
  --no-enable-flashinfer-autotune \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
