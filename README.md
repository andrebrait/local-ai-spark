# Qwen3.8-27B NVFP4 on one NVIDIA DGX Spark

Reproducible configuration for serving either of these checkpoints through vLLM on one GB10:

- [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4), revision `482ca0f3832238542f8f5295dde86b5f22711d80`
- [`ukisai/Swift-Qwen3.8-27B-NVFP4`](https://huggingface.co/ukisai/Swift-Qwen3.8-27B-NVFP4), revision `4cf1019102c2fe9841c07109ac84acb40dabd9ec`

Only one model runs at a time. Both Docker containers remain defined, so switching does not alter configuration. The active container uses Docker's `unless-stopped` restart policy.

Swift is distributed under the Swift Open License 1.0. Check its model card before commercial use; organizations above its revenue threshold require a separate license.

## Runtime profile

Both containers use the same Spark-specific vLLM profile:

- One GB10 GPU: tensor parallelism `1`
- Native context: `262,144` tokens
- FP8 E4M3 KV cache
- MTP self-speculation with three draft tokens
- Up to 32 scheduled sequences and 32,768 batched prefill tokens
- Qwen reasoning parser and `qwen3_coder` tool parser
- `gpu-memory-utilization=0.75`
- FlashInfer's transient autotuner disabled; normal FlashInfer kernels remain enabled
- Host networking bound only to the Tailscale address `100.64.255.60:8000`
- Bearer authentication on every HTTP route

The active image is `sha256:213e470219da6e21119f0a13a4df35f7e0d1f18ed2fb26e43f68c58965b30dfe`, built from the immutable arm64 vLLM base pinned in [`Dockerfile.local-ai`](Dockerfile.local-ai). The image's authentication change protects `/health`, `/metrics`, `/tokenize`, and compatibility routes in addition to `/v1/*`.

Measured on the deployed Spark:

| Profile | FP8 KV tokens | Full 262,144-token sessions |
|---|---:|---:|
| NVIDIA | 1,637,301 | 6 |
| Swift | 1,634,372 | 6 |

`max-num-seqs=32` permits more shorter sessions; it does not mean 32 full-context requests fit simultaneously.

## Install

The scripts expect rootful Docker with NVIDIA Container Toolkit configured.

1. Build or select the runtime image:

   ```bash
   docker build -f Dockerfile.local-ai -t local-ai-qwen:27b .
   ```

2. Create the root-owned API credential files without committing their contents:

   ```bash
   sudo install -d -m 0700 /home/andre/local-ai/secrets
   sudo install -m 0600 /dev/null /home/andre/local-ai/secrets/api.env
   sudo install -m 0600 /dev/null /home/andre/local-ai/secrets/api.key
   sudoedit /home/andre/local-ai/secrets/api.env
   sudoedit /home/andre/local-ai/secrets/api.key
   ```

   Put `VLLM_API_KEY=<url-safe-key>` in `api.env` and only the same raw key in
   `api.key`, each followed by a newline.

3. Download both pinned checkpoints (about 45 GB total):

   ```bash
   sudo env IMAGE=local-ai-qwen:27b scripts/download-weights.sh all
   ```

4. Create both stopped containers and install the selector:

   ```bash
   sudo env IMAGE=local-ai-qwen:27b scripts/create-27b.sh
   sudo env IMAGE=local-ai-qwen:27b scripts/create-swift-27b.sh
   sudo install -m 0755 scripts/local-ai-model /usr/local/sbin/local-ai-model
   ```

5. Start the NVIDIA profile:

   ```bash
   sudo local-ai-model nvidia
   ```

The first load compiles kernels and captures CUDA graphs. Follow progress with:

```bash
sudo docker logs --follow qwen38-27b
```

## Operations

```bash
sudo local-ai-model status
sudo local-ai-model nvidia
sudo local-ai-model swift
```

Switching stops the other container before starting the target, preventing both models from allocating unified memory simultaneously. The inactive container remains stopped across Docker daemon and host restarts.

Run the single completion smoke check after the active model reaches readiness:

```bash
sudo scripts/smoke-test.sh
sudo env MODEL=swift-qwen3.8-27b scripts/smoke-test.sh
```

There is no systemd model service or separate memory watchdog. Docker owns restart behavior directly.

## OMP provider

Merge [`config/omp-models.yml`](config/omp-models.yml) into the controller's `~/.omp/agent/models.yml`; do not replace unrelated providers. Store the matching API key at `/root/.omp/agent/secrets/dgx-vllm.key` with mode `0600`.

Use the model ID matching the active profile:

```bash
omp --model dgx-vllm/qwen3.8-27b
omp --model dgx-vllm/swift-qwen3.8-27b
```

Existing OMP sessions may require restart after changing the model catalog.

## Retired deployment

Qwen3.8-Flash-Next weights, containers, launchers, provider metadata, and the custom memory supervisor are intentionally absent from this configuration. Historical benchmark reports remain under [`docs/`](docs/) as evidence only; they are not executable deployment instructions.
