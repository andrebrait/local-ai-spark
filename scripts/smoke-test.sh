#!/usr/bin/env bash
# One authenticated completion against the currently active profile.
set -euo pipefail

BASE="${BASE:-http://100.64.255.60:8000}"
MODEL="${MODEL:-qwen3.8-27b}"
API_KEY_FILE="${API_KEY_FILE:-/home/andre/local-ai/secrets/api.key}"

python3 - "$BASE" "$MODEL" "$API_KEY_FILE" <<'PY'
import json
from pathlib import Path
import sys
import urllib.request

base, model, key_file = sys.argv[1:]
key = Path(key_file).read_text().strip()
headers = {"Authorization": "Bearer " + key}

request = urllib.request.Request(base + "/health", headers=headers)
with urllib.request.urlopen(request, timeout=10) as response:
    assert response.status == 200

payload = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": "Reply with exactly: READY"}],
    "max_tokens": 256,
    "temperature": 0,
}).encode()
request = urllib.request.Request(
    base + "/v1/chat/completions",
    data=payload,
    headers={**headers, "Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=120) as response:
    result = json.load(response)

content = result["choices"][0]["message"]["content"].strip()
assert content == "READY", content
print(f"{model}: READY")
PY
