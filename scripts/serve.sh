#!/usr/bin/env bash
# Production two-slot recipe. The foreground supervisor owns every lifecycle action.
set -euo pipefail
set +x
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${IMAGE-sha256:213e470219da6e21119f0a13a4df35f7e0d1f18ed2fb26e43f68c58965b30dfe}"
MODEL_HOST="${MODEL_HOST-/home/andre/local-ai/models/Qwen3.8-Flash-Next-NVFP4}"
CACHE_HOST="${CACHE_HOST-/home/andre/local-ai/cache}"
BIND_HOST="${BIND_HOST-100.64.255.60}"
PORT="${PORT-8000}"
API_ENV_FILE="${API_ENV_FILE-/home/andre/local-ai/secrets/api.env}"

# Old tuning knobs must not silently change this allocation plan (including empty values).
for setting in 'MAXLEN=262144' 'SEQS=2' 'MTP=3' 'CHUNK=4096' 'CAPTURE_SIZES=4,8' \
  'PLE_MODE=staged' 'PLE_WORKERS=64' 'KV_DTYPE=fp8_e4m3' 'KV_CACHE_MEMORY_BYTES=5368709120' \
  'MAMBA_SSM_CACHE_DTYPE=bfloat16' 'PREFIX_CACHE=1' 'GMU=0.7533'; do
  key="${setting%%=*}"; expected="${setting#*=}"
  if [[ -v "$key" && "${!key}" != "$expected" ]]; then
    echo "!! Unsupported production override: $key" >&2; exit 2
  fi
done
for key in HOST_RESERVE_GIB PATCH_DIR DRAFT_VOCAB; do
  if [[ -v "$key" ]]; then
    echo "!! Unsupported production override: $key" >&2; exit 2
  fi
done
PATCH_DIR="$ROOT/src/full-recipe-patch"
[[ "$#" == 0 ]] || { echo '!! This launcher accepts no extra arguments' >&2; exit 2; }
export IMAGE BIND_HOST PORT API_ENV_FILE
python3 "$ROOT/scripts/watch-memory.py" check-config

[[ -f "$MODEL_HOST/config.json" ]] || { echo '!! Official NVIDIA checkpoint missing' >&2; exit 3; }
for file in ple_layer.py ple_mmap.py model_state.py mtp_draft_vocab.py \
  upstream-overlays/modelopt.py upstream-overlays/ops_ple.py upstream-overlays/ops_qsa.py \
  upstream-overlays/qsa.py upstream-overlays/platforms_interface.py; do
  [[ -f "$PATCH_DIR/$file" ]] || { echo "!! Recipe patch missing: $file" >&2; exit 3; }
done
DRAFT_VOCAB="$ROOT/src/miaai/draft_vocab_en_code_47k.txt"
python3 - "$DRAFT_VOCAB" "$MODEL_HOST/config.json" <<'PYVOCAB'
import hashlib, json, sys
from pathlib import Path
raw = Path(sys.argv[2]).read_bytes()
if hashlib.sha256(raw).hexdigest() != 'deef67a61f3311faf051b23dc4192f442c7fee4f9cd2f38cbcbe4da55c763a80':
    raise SystemExit('Checkpoint configuration differs from the validated NVIDIA NVFP4 profile')
config = json.loads(raw)
size = config.get('text_config', config)['vocab_size']
ids = [int(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
if len(ids) != 47149 or len(ids) != len(set(ids)) or min(ids) < 0 or max(ids) >= size:
    raise SystemExit('Invalid production draft vocabulary')
PYVOCAB

VP=/usr/local/lib/python3.12/dist-packages/vllm
MOUNTS=()
for mapping in 'ple_layer.py:models/qwen4_exp/nvidia/ple_layer.py' \
  'ple_mmap.py:models/qwen4_exp/nvidia/ops/ple_mmap.py' \
  'model_state.py:models/qwen4_exp/nvidia/model_state.py' \
  'mtp_draft_vocab.py:models/qwen4_exp/nvidia/mtp.py' \
  'upstream-overlays/modelopt.py:model_executor/layers/quantization/modelopt.py' \
  'upstream-overlays/ops_ple.py:models/qwen4_exp/nvidia/ops/ple.py' \
  'upstream-overlays/ops_qsa.py:models/qwen4_exp/nvidia/ops/qsa.py' \
  'upstream-overlays/qsa.py:models/qwen4_exp/nvidia/qsa.py' \
  'upstream-overlays/platforms_interface.py:platforms/interface.py'; do
  MOUNTS+=(-v "$PATCH_DIR/${mapping%%:*}:$VP/${mapping#*:}:ro")
done

# Gate the lightweight private-namespace init before importing vLLM/CUDA. Only the
# supervisor's verified pidfd can release it after durable state and monitoring.
GATE='import os,signal,sys; signal.signal(signal.SIGUSR1,lambda *_:os.execvp("vllm",["vllm","serve",*sys.argv[1:]])); signal.pause()'
ARGS=(--gpus all --restart no --pull never --memory 112g --memory-swap 112g
  --cap-drop ALL --security-opt no-new-privileges:true
  --network host --ipc private --shm-size 32g --ulimit memlock=-1:-1
  --env-file "$API_ENV_FILE" --entrypoint python3
  -v "$MODEL_HOST:/models/qwen38fn:ro" -v "$CACHE_HOST:/root/.cache"
  -v "$DRAFT_VOCAB:/opt/qwen38-draft-vocab.txt:ro" "${MOUNTS[@]}"
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_ENGINE_READY_TIMEOUT_S=1200
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUTE_DSL_ARCH=sm_121a
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1
  -e VLLM_USE_DEEP_GEMM=0 -e VLLM_USE_V2_MODEL_RUNNER=1
  -e QWEN4EXP_PLE_MMAP=1 -e QWEN4EXP_PLE_STAGED=1 -e QWEN4EXP_PLE_MMAP_THREADS=64
  -e QWEN4EXP_DRAFT_VOCAB=/opt/qwen38-draft-vocab.txt
  "$IMAGE" -c "$GATE" /models/qwen38fn --served-model-name qwen3.8-flash-next
  --host "$BIND_HOST" --port "$PORT" --quantization modelopt --tensor-parallel-size 1
  --max-model-len 262144 --max-num-seqs 2 --gpu-memory-utilization 0.7533
  --max-num-batched-tokens 4096 --kv-cache-memory-bytes 5368709120
  --no-enable-flashinfer-autotune --enable-prefix-caching --enable-prompt-tokens-details
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"medium"}'
  --generation-config auto
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}'
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8]}'
  --mamba-ssm-cache-dtype bfloat16 --kv-cache-dtype fp8_e4m3 --mamba-cache-mode align)
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' docker create --name qwen38-flash-two-slot "${ARGS[@]}"; printf '\n'
  exit 0
fi
install -d -o 0 -g 0 -m 0755 "$CACHE_HOST"
exec python3 "$ROOT/scripts/watch-memory.py" run -- "${ARGS[@]}"
