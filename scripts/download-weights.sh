#!/usr/bin/env bash
# Download and verify the two pinned Qwen3.8-27B NVFP4 checkpoints.
# Usage: scripts/download-weights.sh [all|nvidia|swift]
set -euo pipefail

PROFILE="${1:-all}"
IMAGE="${IMAGE:-sha256:b3eb57bba79454feb50c304fb5633af2486eed8e0a658dfe449c659e87be0f5e}"
MODEL_ROOT="${MODEL_ROOT:-/home/andre/local-ai/models}"

TOKEN_ARGS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  TOKEN_ARGS+=(-e HF_TOKEN)
elif [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  TOKEN_ARGS+=(-e HUGGING_FACE_HUB_TOKEN -e HF_TOKEN="$HUGGING_FACE_HUB_TOKEN")
else
  echo ">> no Hugging Face token; downloads use the public rate limit"
fi

download() {
  local key="$1" repository="$2" revision="$3" directory="$4"
  install -d -m 0755 "$MODEL_ROOT/$directory"
  echo ">> downloading $repository@$revision"
  docker run --rm --pull never --name "qwen38-27b-download-$key" \
    "${TOKEN_ARGS[@]}" \
    -e HF_HUB_DISABLE_XET=1 \
    -v "$MODEL_ROOT/$directory:/models" \
    --entrypoint bash "$IMAGE" \
    -c "hf download '$repository' --revision '$revision' --local-dir /models --max-workers 8 && hf cache verify '$repository' --revision '$revision' --local-dir /models --fail-on-missing-files"
}

case "$PROFILE" in
  all|nvidia)
    download nvidia nvidia/Qwen3.8-27B-NVFP4 \
      482ca0f3832238542f8f5295dde86b5f22711d80 Qwen3.8-27B-NVFP4
    ;;
  swift) ;;
  *) echo "usage: $0 [all|nvidia|swift]" >&2; exit 2 ;;
esac

case "$PROFILE" in
  all|swift)
    download swift ukisai/Swift-Qwen3.8-27B-NVFP4 \
      4cf1019102c2fe9841c07109ac84acb40dabd9ec Swift-Qwen3.8-27B-NVFP4
    ;;
esac

echo ">> verified pinned checkpoint files"
