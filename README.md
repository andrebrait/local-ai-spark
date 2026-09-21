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
- Prefix caching enabled with 1,600-token match blocks and Mamba `align` mode
- FlashInfer's transient autotuner disabled; normal FlashInfer kernels remain enabled
- Host networking bound only to the Tailscale address `100.64.255.60:8000`
- Bearer authentication on every HTTP route

The active image is `sha256:05eb4719754d1390b2b577a761eb9e53cc8413c17f7863511633e9fba45102c8`, built from the immutable ARM64 vLLM `v0.29.0` base pinned in [`Dockerfile.local-ai`](Dockerfile.local-ai). The image's authentication change protects `/health`, `/metrics`, `/tokenize`, and compatibility routes in addition to `/v1/*`.

Measured on the deployed Spark:

| Profile | FP8 KV tokens | Full 262,144-token sessions |
|---|---:|---:|
| NVIDIA | 1,698,810 on the current boot | 6 |
| Swift | Not remeasured after enabling prefix caching | 6 expected from identical cache geometry |

The exact KV token count varies slightly between boots because CUDA-graph
memory is profiled during startup; use the live startup log for capacity
planning.

`max-num-seqs=32` permits more shorter sessions; it does not mean 32 full-context requests fit simultaneously.

The runtime update changed the selected NVFP4 kernel from the old W4A16 CuTeDSL
path to `FlashInferCutlassNvFp4LinearKernel`. Matched single-request checks:

| Workload | Previous vLLM | Sept. 20 nightly | v0.29.0 |
|---|---:|---:|---:|
| Short-context decode | 16.12 tok/s | 26.72 tok/s | **29.28 tok/s** |
| 47.5K-context decode | 18.70 tok/s | **26.54 tok/s** | 23.99 tok/s |
| 47.5K time to first token | 180.9 s | **40.5 s** | 43.3 s |

The released `v0.29.0` runtime was selected for reproducibility and its higher
short-context throughput. The faster long-context nightly image remains cached
locally as a rollback option.

An isolated `flashinfer_b12x` comparison kept FlashInfer attention, MTP3,
sampling, memory, and context settings unchanged; `CUTE_DSL_ARCH=sm_121a` was
already active. Prefix caching remained off.

| Workload | CUTLASS | FlashInfer B12x | B12x change |
|---|---:|---:|---:|
| Short-context decode | 29.28 tok/s | 28.16 tok/s | -3.8% |
| 47.5K-context decode | 23.99 tok/s | 22.94 tok/s | -4.4% |
| 47.5K time to first token | 43.34 s | 44.82 s | +3.4% slower |
| FP8 KV capacity | 1,726,635 tokens | 1,711,990 tokens | -0.85% |

The B12x profile was rejected and removed. Automatic selection remains on
`FlashInferCutlassNvFp4LinearKernel`.

Prefix caching was tested separately with the same CUTLASS/MTP3 profile,
1,600-token match blocks, and Mamba `align` mode:

| Probe | Cold | Cached | Reused prefix |
|---|---:|---:|---:|
| Repeated 31.5K prompt | 26.65 s TTFT | 1.85 s TTFT | 28,800 tokens (91.4%) |
| Growing 43.4K conversation | 37.79 s TTFT | 1.48–1.74 s TTFT | 41,600 tokens per turn |

All 12 growing-conversation recalls and 80 concurrent cached requests returned
the exact expected values. This focused canary cannot rule out the rare silent
corruption reported upstream for hybrid Qwen + MTP prefix caching
([vLLM #53912](https://github.com/vllm-project/vllm/issues/53912)).
The deployment owner accepted that residual risk and enabled prefix caching on
September 21, 2026. Empty output, repeated punctuation/CJK, or unexplained
content degeneration is a rollback trigger: remove `--enable-prefix-caching`
from both profiles and recreate the containers.

The host was updated through NVIDIA's configured Spark repositories on
September 21, 2026: driver `580.178.04`, NVIDIA Container Toolkit `1.20.1`,
DGX Spark OTA metadata `26.09.2`, host CUDA toolkit `13.0.3`, SoC firmware
`2.155.14`, embedded-controller firmware `3.5.11`, and USB-PD firmware `0.5.22`.

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
