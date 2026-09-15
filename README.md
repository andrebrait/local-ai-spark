# Running Qwen3.8-Flash-Next (NVFP4 + MTP) on a single NVIDIA DGX Spark

Field notes from bringing up **Qwen3.8-Flash-Next** — a ~176B-parameter multimodal
MoE (125B main + 51B n-gram embedding table, 6B active per token) — on **one**
DGX Spark / GB10, using vLLM with the model's built-in MTP speculative decoding.

The default recipe is adapted from
[tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark)
at `d83f10c`, with its Apache-licensed vLLM patch set vendored under
[`src/full-recipe-patch/`](src/full-recipe-patch/). The earlier recipe source,
[blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX),
remains available through the explicit legacy launchers.

## Local AI deployment (this fork)

The deployment uses `Dockerfile.local-ai`, not the historical `Dockerfile`.
It pins the arm64 vLLM `8a728663` image and the official NVIDIA checkpoint
revision `fc694b54fb0174e0913e6adf86691ef85a4ead47`. The checkpoint is unchanged:
NVFP4 routed experts, BF16 side layers, and FP8 n-gram/MTP tensors.
CI builds and validates this image with read-only permissions; it does not publish
packages or deploy changes to the node.

`scripts/serve.sh` uses staged NVMe n-gram reads, MTP3 with the 47,149-token
draft vocabulary, **one scheduler slot**, 4,096-token prefill chunks, a fixed
**5,368,709,120-byte (5 GiB) FP8 E4M3 KV pool**, BF16 recurrent state, decode
graphs `[4,8]`, and the native **262,144-token input-plus-output limit**.
The image adds version-checked prefix-cache corrections, including explicit
Qwen MTP group identification. Prefix reuse is enabled with aligned Mamba state.
Prompt-token cache details are enabled. The immutable tested image is
`sha256:213e470219da6e21119f0a13a4df35f7e0d1f18ed2fb26e43f68c58965b30dfe`.
The launcher rejects moving images and allocation/profile overrides; it never
falls back to utilization-derived/unbounded KV sizing. GMU stays at `0.7533`,
but the explicit byte budget, not GMU, sizes KV.

Thinking and preserved thinking are enabled at medium effort. Defaults are
temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 0, and repetition
penalty 1.0. The model's generation configuration retains its official EOS tokens.
Clients may override sampling and `chat_template_kwargs` per request. For
non-thinking mode, Qwen recommends temperature 0.7, top-p 0.8 and presence
penalty 1.5; for deeper reasoning, set `reasoning_effort` to `xhigh`.

### Access and operations

- API base: `http://100.64.255.60:8000/v1` over Tailscale.
- Served model: `qwen3.8-flash-next`.
- API key: `/home/andre/local-ai/secrets/api.key` on the Spark; never committed.
- Launcher credential: `/home/andre/local-ai/secrets/api.env`, containing
  `VLLM_API_KEY=` followed by that same raw key. Both files are root-owned, mode
  0600; the environment file must be a regular file, not a symlink.
- All HTTP routes require the bearer key, including health, metrics and tokenizer
  diagnostics. Only CORS `OPTIONS` requests bypass authentication.
- Source: `/home/andre/local-ai/source`; model and cache are sibling directories.
- Evidence: `/home/andre/local-ai/evidence`, including checksums, package audit,
  runtime identities, authentication checks and serving acceptance.

`local-ai-memory.service` now owns the model lifecycle; it is **not** the old
independent five-second watchdog. Docker and systemd both use **no automatic
restart**. The production container is `qwen38-flash-two-slot`; the stopped
original baseline is never started, stopped, removed, or adopted by this
supervisor. It only manages its own name and recorded ID. Do not launch legacy
profiles alongside it.

```bash
sudo systemctl start local-ai-memory.service
sudo systemctl status --no-pager local-ai-memory.service
sudo python3 /home/andre/local-ai/source/scripts/watch-memory.py status
sudo journalctl -u local-ai-memory.service -n 100 --no-pager
sudo docker logs --follow qwen38-flash-two-slot
sudo systemctl stop local-ai-memory.service
```

`systemctl start` succeeds only after an authenticated `/health` response of
**200** (`Type=notify`), not container creation. It remains `activating` while
loading. The model has a 1,200-second readiness deadline after the load gate,
plus bounded preflight/gating time. A separate timeout-bounded health thread
cannot block the 250ms RAM loop. After readiness, 60 seconds without a healthy
sample, a dead/stalled health thread, or a failed RAM read stops and latches.
An occupied bind port is refused before creating the model. An unassigned private
address is given up to 60 seconds to appear, covering normal Tailscale boot timing.

### Controller model selection

OMP and its agents run on the controller, not on the Spark. Use
`dgx-vllm/qwen3.8-flash-next` with its native **262,144-token context**.
The server processes one sequence at a time; additional requests queue.
No smaller worker alias, task-role change, or OMP compaction patch is required.
Keep existing task routing and normal OMP compaction settings unchanged.

Merge this credential-free entry into the controller's `~/.omp/agent/models.yml`
without replacing its other providers:

```yaml
providers:
  dgx-vllm:
    baseUrl: http://100.64.255.60:8000/v1
    api: openai-completions
    apiKey: '!cat /root/.omp/agent/secrets/dgx-vllm.key'
    authHeader: true
    models:
    - id: qwen3.8-flash-next
      name: Qwen3.8-Flash-Next (DGX Spark NVFP4)
      reasoning: true
      thinking:
        mode: effort
        efforts: [low, medium, xhigh]
        defaultLevel: medium
        requiresEffort: false
      input: [text, image]
      supportsTools: true
      contextWindow: 262144
      maxTokens: 262144
      omitMaxOutputTokens: true
      cost: {input: 0, output: 0, cacheRead: 0, cacheWrite: 0}
      compat:
        maxTokensField: max_tokens
        supportsStore: false
        supportsDeveloperRole: false
        thinkingFormat: qwen-chat-template
        qwenTemplateReasoningEffort: true
        reasoningContentField: reasoning
        replayReasoningContent: true
        qwenPreserveThinking: true
        streamFirstEventTimeoutMs: 0
        streamIdleTimeoutMs: 1800000
```

Refresh the model catalog or start a new OMP session after changing this entry
or rotating its credential. Existing sessions may retain resolved model metadata.

### Lessons learned (2026-09-14)

- **Keep the operating target small.** The final choice is one active sequence
  and the native 262,144-token input-plus-output limit. Two-worker context
  experiments and custom OMP policies are not prerequisites for using this node.
  The container retains its historical `qwen38-flash-two-slot` name solely for
  lifecycle identity; the launcher now allows only one active sequence.
- **MoE sparsity does not remove weight-storage costs.** What makes this
  checkpoint fit is the staged, disk-backed 47.68 GiB PLE lookup table, not
  arbitrary expert streaming. Keep the official mixed-precision checkpoint
  unchanged and keep KV allocation explicit.
- **Separate failed allocation attempts from fatal failures.** NVIDIA
  `NV_ERR_NO_MEMORY` messages occurred during starts that subsequently reached
  readiness. That is not proof of a harmless cause, nor proof of a host OOM.
  Correlate them with CUDA errors, Xids, host OOM kills, free memory and service
  health. The [driver logging site](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.173.02/src/nvidia/src/kernel/gpu/mem_mgr/mem_desc.c#L1359)
  returns the failed allocation status; PyTorch also distinguishes
  [allocation retries from thrown OOMs](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.memory_stats.html).
  A manually compacted clean start did not establish a repeatable fix.
- **Do not infer causality from adjacent events.** The worker that omitted
  `read_fixture` had not compacted. A peer began compaction around the same time.
  A different run passing with synchronous soft compaction did not demonstrate
  an OMP asynchronous-compaction bug. A later offline reconstruction included
  the tool and matched the recorded prompts, but was not the original wire
  capture. No per-agent compaction patch is part of this deployment.
- **Keep diagnostics credential-safe.** Runtime model metadata can contain
  resolved authorization headers even when configuration uses a key-file
  command. Redact before logging, keep diagnostic directories mode 0700 and
  secret files mode 0600, and rotate any disclosed key. Keep `api.env`, `api.key`
  and the controller's `/root/.omp/agent/secrets/dgx-vllm.key` synchronized.
  Verify the replacement key succeeds and the retired key returns HTTP 401.
- **Preserve lifecycle ownership.** Use the systemd service, not raw Docker
  start/stop commands. Docker's init is not the gated Python loader; readiness
  must identify the child. Pre-readiness systemd startup deadlines also need
  explicit renewal; watchdog notifications alone do not renew them.
- **Use bounded evidence and stop.** A ready service plus a real generation
  proves basic operation, not universal long-context correctness. Synthetic
  retention tests are not general coding-quality certification, and one
  successful run is not a reliability guarantee. Further benchmarking is
  optional rather than an automatic expansion of this deployment's scope.

### Safety state and recovery

- **Clean/absent → dirty:** acquire the exclusive lifecycle lock, require at
  least **116 GiB MemAvailable**, trigger Linux host-memory compaction, and
  require the same reserve again. Only then atomically write/fsync the
  root-private `/var/lib/local-ai-model/state.json` and its directory **before
  create/start**. Each run records a random ownership label and the full Docker
  container ID. One manually compacted start had no NVIDIA allocation warnings,
  but a later automatically compacted start logged them again. Host compaction
  is not an established fix; do not treat one warning-free start as proof. This gate
  allows approximately 96 GiB of observed startup allocation plus the 20 GiB
  runtime reserve; the roughly 25–28 GiB available **after** loading is not
  sufficient headroom to start another model.
- **Dirty → gated load:** create with `--restart no`, immutable image, a private
  PID namespace, Docker's minimal init, and a lightweight loader waiting behind
  a SIGUSR1 gate. Verify full ID, run/owner labels, image, init, restart policy,
  cgroup and PID namespace; acquire/recheck the init pidfd and confirm its signal
  handler before releasing vLLM/CUDA. The init forwards termination across the
  loader's gate/exec boundary.
- **Dirty → ready:** authenticated health 200; dirty protection remains armed.
  The RAM loop samples every 250ms and sends SIGKILL through the verified pidfd
  if available RAM falls below **20 GiB** or monitoring fails. There are no
  Docker commands or HTTP requests in that pressure-stop path. Killing the
  private-namespace init terminates its descendants.
- **Ready/dirty → clean:** only an operator/systemd-requested graceful stop,
  observed init exit, Docker-confirmed stopped identity, exit code 0/143 and no
  Docker OOM/error indication clear the interlock.
  SIGTERM gets 30 seconds while RAM monitoring continues. Escalation to SIGKILL
  is a latched failure, not a clean stop. A clean stop retains the stopped
  container; the next start removes only that exact verified stopped instance.
  Cancelling the pre-CUDA gate intentionally leaves a latch and requires an
  explicit reset. A loader that cannot honor SIGTERM also remains latched after
  forced termination; an operator-requested stop alone does not prove clean exit.
- **Ready/dirty → latched:** pressure, timeout, model crash, forced teardown or
  monitoring failure. SIGKILL/power loss may leave `dirty` or `ready` instead;
  both are equally latched for the next start, including after reboot.
  `ExecStopPost` independently reconstructs and verifies ownership, kills a
  surviving owned init via pidfd, confirms stop and retains the latch.

The [systemd watchdog](https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html#WatchdogSec=)
only starts after readiness. Before loading, the init therefore waits at least
31 seconds from supervisor launch for the unit's initial 30-second startup
grace to expire. Each pre-CUDA Docker operation renews a 20-second lease around
its 15-second command timeout. The gate also waits for the last such lease to
expire before releasing CUDA. Only then do five-second renewable deadlines
supervise the RAM loop during loading; the five-second watchdog covers readiness.
Neither mechanism retries. Do not increase `TimeoutStartSec` independently of
the load gate, disable timeout/watchdog settings, or replace the notify unit
with a process-created readiness mode.

Teardown sends `STOPPING=1` and a bounded 120-second cleanup extension after
signalling the owned init. On the target systemd, `STOPPING=1` disarms the watchdog
after readiness; before readiness the explicit extension allows exit confirmation
and state persistence to finish. Notification failure does not prevent the
model's stop or persistent latch.

After a failure, first stop the unit and investigate memory, host/driver logs,
model logs and the saved state. Preserve evidence; **never delete the state file
to bypass a refusal**. If teardown failed, retry only the scoped recovery action:

```bash
sudo systemctl stop local-ai-memory.service
sudo python3 /home/andre/local-ai/source/scripts/watch-memory.py stop-owned
sudo python3 /home/andre/local-ai/source/scripts/watch-memory.py status
# Only after investigating the cause and confirming adequate headroom:
sudo python3 /home/andre/local-ai/source/scripts/watch-memory.py reset
sudo systemctl reset-failed local-ai-memory.service
sudo systemctl start local-ai-memory.service
```

`reset` refuses an active supervisor, a live owned container, ambiguous/mismatched
identity, corrupt/insecure state, or an unavailable Docker daemon. It clears the
latch only after verifying the recorded container is stopped or absent and does
not start anything. If identity or state is damaged, recover the recorded object
manually after investigation; the supervisor will not guess or target unrelated
containers. Do not use `docker start`, `docker restart`, the old `launch.sh`,
or `docker rm` for normal operation: those bypass lifecycle accounting.

**Safety limits:** this is best-effort early termination, not a hard unified-RAM
reservation or proof against GB10/driver/kernel hangs. Pressure can outrun a
250ms poll or delay process scheduling/signals; systemd teardown requires a
working host, Docker metadata and pidfd support. The 112 GiB cgroup cap does not
account for all driver allocations. The private state directory needs reliable
local durable storage. Root/Docker administrators and the deployed source/model
files are trusted; the guard is not a sandbox against a privileged operator.
Cancellation can arrive after the final pre-load admission check; the supervisor
then stops the verified init using bounded teardown and RAM monitoring. This is
not an atomic guarantee that CUDA never begins after a stop signal.
The single scheduler slot serializes active requests; the 5 GiB pool remains a
fixed allocation, not an elastic memory limit. Backend native context does not
configure controller-side context or compaction policy.

Keep the original stopped baseline, image and source snapshot for rollback.
The final one-slot rollout uses the existing launcher tests, supervised startup
and one authenticated generation as its bounded acceptance check. This does not
claim that every long-context workload or failure mode has been certified.

To roll back, choose a preserved, reviewed **supervised** release from the last
deployment receipt. Its matching pinned image must still be installed (or loaded
from the reviewed image archive). Keep the current credentials; do not restore a
retired key from an older snapshot.

```bash
# Set this to the trusted release path recorded before the deployment.
PREVIOUS_RELEASE="/home/andre/local-ai/releases/REPLACE_WITH_REVIEWED_COMMIT"
sudo systemctl stop local-ai-memory.service
sudo python3 /home/andre/local-ai/source/scripts/watch-memory.py status
# Continue only if the state is clean and the recorded container is stopped.
# Otherwise use the latch-recovery procedure above; do not bypass it.
sudo test -d "$PREVIOUS_RELEASE" &&
sudo ln -sfn "$PREVIOUS_RELEASE" /home/andre/local-ai/source &&
sudo install -o root -g root -m 0644 \
  "$PREVIOUS_RELEASE/scripts/local-ai-memory.service" \
  /etc/systemd/system/local-ai-memory.service &&
sudo systemctl daemon-reload &&
sudo python3 /home/andre/local-ai/source/scripts/watch-memory.py check-config &&
sudo systemctl start local-ai-memory.service &&
sudo systemctl status --no-pager local-ai-memory.service
```

Confirm authenticated access with the current key after rollback. Do not use
this procedure to restart an unsupervised legacy launcher or baseline container.

Optional extended diagnostics, on an otherwise idle service:

```bash
sudo python3 /home/andre/local-ai/source/tools/validate_miaai_update.py \
  --base http://100.64.255.60:8000 \
  --allow-tailscale-http \
  --api-key-file /home/andre/local-ai/secrets/api.key \
  --mode all --output /home/andre/local-ai/evidence/acceptance.json
```

This checks reasoning, tool-call JSON, concurrent isolation, growing-prefix
recall and cache hits, server-tokenized near-full context, and over-limit
rejection. These checks are useful evidence, not a guarantee for all workloads.

## Historical upstream NVIDIA TP1 recipe (2026-09-10)

The measurements below belong to the upstream madeye deployment, not this
machine. That profile used thinking off and prefix reuse off; its numbers
must not be presented as measurements of the authenticated configuration above.

### Historical performance

Measured on 2026-09-10 at 11:21–11:23 UTC, directly against the local vLLM API.
Temperature 0, thinking off, native MTP3 enabled, DFlash disabled.

| Workload | Speed | Time to first token |
| --- | ---: | ---: |
| Prose, one request | **36.83 output tok/s** | **0.236 s** |
| Code, one request | **45.75 output tok/s** | **0.201 s** |
| Four simultaneous requests, mixed prose/code | **100.18 output tok/s aggregate** | **1.017 s median** |
| Fresh 8,215-token prompt | 1,606.5 input tok/s | 5.114 s |
| Fresh 32,791-token prompt | 1,824.95 input tok/s | 17.968 s |

Single-request generation rates exclude time to first token and are medians of
three 384-token samples per workload, after two excluded warmups. The four-request
result is one batch of 256-token outputs; its aggregate rate includes prefill.
Long-prompt figures are medians of two fresh documents per length and include
request overhead. These are different throughput definitions, not interchangeable
rates. See the [raw samples and method](docs/performance-live-2026-09-10-112330.json).

All 16 benchmark requests succeeded, with no other completed generation requests
observed during the measurement. MTP accepted 65.7% of proposed tokens across the
run. Available host memory stayed at or above 19.94 GiB; the container remained
healthy with zero restarts. The 30 GiB host reserve produced `GMU=0.7535`, a
7.44 GiB KV budget, and 507,810 cached-token capacity. Six scheduler slots do not
mean six full 262K requests fit: the reported full-context capacity is 1.94x.

This is a small live snapshot, not a matched speedup comparison or a quality
benchmark. The earlier 40-prompt result below used a different workload and
memory configuration.

### MiaAI optimization update (2026-09-10)

The compatible changes from
[MiaAI commit d038090](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/commit/d03809008834124e80223c3482f2ddb59577a48f)
are applied to this NVIDIA TP1 recipe:

The historical tuning described here used a 47,149-token code vocabulary,
BF16 recurrent state and automatically derived decode widths
`(MTP+1)*S` (six-slot graphs `[4,8,12,16,20,24]`). The MTP head selects those
vocabulary rows while the target still verifies with its full vocabulary;
the data and upstream license are in [`src/miaai/`](src/miaai/PROVENANCE.md).

These are historical experiments, **not production override instructions**.
The current launcher pins one slot, graphs `[4,8]`, staged PLE and 5 GiB KV.
`HOST_RESERVE_GIB`, custom draft/patch selections, alternative precision and
larger/unbounded KV profiles are refused before CUDA starts. MiaAI uses a
different checkpoint and packed PLE format; its memory estimates are not
interchangeable. DFlash remains disabled. Non-English/non-code traffic may
have different draft acceptance; upstream speedups are not local results.

Before this update, on 2026-09-06 at `GMU=0.80`, the measured 40-prompt median was
**43.5 tok/s** with **0.26 s** median TTFT and a 0.88 automatic task score. The GPU is locked to its supported 3,003 MHz
ceiling; sustained decode reaches 2,535 MHz on this host. Use
`scripts/serve-legacy.sh` or `scripts/serve-500k.sh` only for the older
RadixArk/hybrid and 500k-context profiles.

The recipe files
(`Dockerfile`, `src/`, `tools/`, `scripts/`) for the legacy stack are **vendored
in this repo** so it is self-contained; they remain Apache-2.0 © blazux (see
`LICENSE`).

Recipe source: [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)
(patched official vLLM image `vllm/vllm-openai:qwen38-flash-next`). The recipe files
(`Dockerfile`, `src/`, `tools/`, `scripts/`) are **vendored in this repo** so it is
self-contained; they remain Apache-2.0 © blazux (see `LICENSE`).
`scripts/gateway.py` and `scripts/serve-public.sh` are additions of this repo, not
part of that recipe.
Checkpoint: [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4).

## Legacy RadixArk config alignment (2026-09-05)

Compatible defaults follow MiaAI-Lab's
[config at commit 203834c](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/203834ca88000c8192112e396b80d886b522caa0/.env.sample).
The legacy launcher uses native 262,144 context (YaRN off), MTP=3, four concurrent
sequences, FP8 KV, and 2,048-token prefill chunks. `MAX_NUM_BATCHED_TOKENS=8192`
selects the larger chunks used in our earlier benchmarks.

`CUDAGRAPH_CAPTURE_SIZES=auto` explicitly captures `(1 + MTP) * S` for each
sequence count `S` through `SEQS`: `[4,8,12,16]` at the base defaults, or `[4]`
in the single-stream 524k profile. A comma list overrides the sizes; an empty
value uses vLLM's defaults. MTP=0 also works, capturing widths `1..SEQS`.

Our RadixArk checkpoint and CPU mmap PLE implementation differ from MiaAI's
Mia-AiLab checkpoint and packed PLE offload worker. We retain PIECEWISE graphs,
the PLE splitting op, exact QSA top-k, and the existing compilation mode;
MiaAI's FULL_DECODE_ONLY / compilation mode 0 cannot be applied to this PLE path.
We also retain the locally validated 9 GiB explicit FP8 KV pool instead of
importing their 16 GiB target and memory estimates for a different checkpoint.
Four sequences is a scheduler limit, not a guarantee that four full 262k
requests fit in our pool. Ports and gateway wiring remain as documented below.

The aligned settings were validated on 2026-09-06 with both the 524k
single-stream profile and the native 262k four-sequence profile. All four graph
widths captured successfully. The base profile exposed a 542,103-token KV cache
(2.07x a full 262k request), so four sequences support concurrent shorter
requests but not four simultaneous full-context requests.

| Profile / workload | Result |
| --- | ---: |
| 524k hybrid, short decode | 24.7–32.9 tok/s |
| 524k hybrid, 32k TTFT | 25.5 s |
| 524k hybrid, 500k TTFT | 432.4 s |
| Native NVFP4, 10.7k cold prefill | 1,025 tok/s |
| Native NVFP4, four short requests | 40.3 aggregate tok/s |
| Native NVFP4, four concurrent 60k prompts | 158.2 s wall time; 4/4 correct |

The 500k test completed with 20.55 GiB minimum MemAvailable. Four concurrent
60k prompts completed with 17.44 GiB minimum MemAvailable; their TTFTs ranged
from 53.0 to 156.2 seconds as the scheduler interleaved prefill. No tested
profile restarted or OOMed. See the
[alignment validation report](docs/config-alignment-test-2026-09-06.json) for
configuration, timings, and limitations.

The older performance reports below record 8,192-token chunks and
vLLM-default capture sizes. To reproduce those launcher settings, use:

```bash
MAX_NUM_BATCHED_TOKENS=8192 CUDAGRAPH_CAPTURE_SIZES= bash scripts/serve-500k.sh
```

## Local single-stream 524k profile (2026-09-05)

```bash
docker build -f Dockerfile.performance -t qwen38-flash-dgx:performance .
bash scripts/serve-500k.sh
# After /health succeeds:
python3 scripts/bench-single-stream.py
python3 scripts/validate-context.py --tokens 500000
```

The incremental image requires the existing `qwen38-flash-dgx:latest` image.
On a clean machine, first build the main Dockerfile, download weights and run
`scripts/prepare-hybrid.sh`. This profile uses the prepared hybrid checkpoint,
524,288 total context tokens, YaRN factor 2, MTP=3, one concurrent sequence,
prefix caching, exact QSA top-k, and an explicit 9 GiB **FP8** KV pool.
The API listens on `127.0.0.1:18300`. The separate 27B service on port 8080
is not connected to this endpoint.

FP8 is the default after the comparison below showed broadly comparable
single-stream performance with 7 GiB less KV allocation. Hybrid weight quantization
remains enabled. Exact top-k retains the local correctness fix; `EXACT_TOPK=0` can
be faster but restores the stock kernel's known candidate-selection issue.
`SEQS`, `MTP`, and other launcher variables can still be overridden.

Inspired by [MiaAI-Lab's single-Spark recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark),
PLE mappings now use `MADV_RANDOM` to avoid reading adjacent pages for scattered
lookups. Set `VLLM_PLE_MMAP_RANDOM=0` for an A/B comparison. YaRN scaling is
derived from the requested context and preserves the checkpoint's remaining
RoPE parameters. Context beyond 524,288 is refused, as is context beyond
262,144 without YaRN.

`KV_BYTES` explicitly sizes KV and bypasses `GPU_MEM` sizing; raising it consumes
the desktop's shared memory. This is a one-model-at-a-time profile. A 500,000-token
prompt leaves 24,288 tokens for output. The validation script checks actual API
usage and records TTFT and host memory; its repeated archive text is a capacity
test, not evidence of reliable retrieval across arbitrary 500k documents.
`DRY_RUN=1 bash scripts/serve-500k.sh` prints the launch without replacing a container.
The single-stream benchmark uses three sequential short prompts, thinking disabled,
and 384 generated tokens per prompt; it writes `/tmp/flash-single-stream.json`.
The capacity test writes `/tmp/flash-context-validation.json`.

BF16 reference profile (`scripts/serve-500k-bf16.sh`): the 16 GiB KV pool holds
**587,382 tokens**. An exact
**500,000-token prompt** completed successfully and returned the correct arithmetic
answer, with **414.3 s TTFT** and **10.24 GiB minimum MemAvailable**. Short-prompt
single-stream decode measured **26.3–33.4 tok/s**; warm TTFT was **0.24–0.30 s**
(first request: 3.16 s). These are final-profile measurements, not a matched
speedup comparison against the old profile. Full prompts, outputs, configuration,
and usage are in [the validation report](docs/performance-2026-09-05.json).

### Latest FP8 single-stream measurements (2026-09-05)

Refreshed at 13:47 UTC against the then-running FP8 profile (8,192-token
prefill chunks and vLLM-default graph capture sizes), using
`scripts/bench-single-stream.py`: three sequential prompts, temperature 0,
thinking disabled, and 384 output tokens each.

| Workload | Decode | TTFT | End-to-end throughput |
| --- | ---: | ---: | ---: |
| Database prose | 28.4 tok/s | 0.23 s | 28.0 tok/s |
| Code | 33.5 tok/s | 0.27 s | 32.8 tok/s |
| TCP prose | 24.6 tok/s | 0.31 s | 24.2 tok/s |

These are one sample per prompt on an already-running server with prefix caching
enabled; the cache was not reset. Decode excludes TTFT and uses 383 tokens over
the interval between the first and last content chunks. End-to-end throughput
uses all 384 output tokens over the full request duration. Raw outputs, usage,
timings, and the running server command are in the
[refresh report](docs/performance-fp8-refresh-2026-09-05.json).
The BF16 comparison and 500k capacity results below are from the earlier validation
and were not rerun for this refresh.

### Default FP8 KV profile and BF16 comparison

```bash
bash scripts/serve-500k.sh       # legacy FP8 profile; serve-500k-fp8.sh is an explicit alias
# Restore the BF16 profile:
bash scripts/serve-500k-bf16.sh
```

The FP8 profile uses the same image, hybrid weights, 524,288 context, MTP=3,
single sequence, and exact top-k. It sets `KV_DTYPE=fp8_e4m3` and a **9 GiB**
KV pool, saving **7 GiB of allocation** versus the BF16 profile. The installed
QSA patch stores the main KV pages in FP8 and converts gathered tiles to BF16
inside the attention kernel, with FP32 accumulation. It does not allocate a
second full BF16 cache. The side/compressor caches remain BF16. Conversion
does not recover precision lost during FP8 storage.

Measured on 2026-09-05 with the same three short prompts and generation settings:

| Metric | BF16 KV | FP8 KV |
| --- | ---: | ---: |
| KV allocation | 16 GiB | 9 GiB |
| KV token capacity | 587,382 | 603,639 |
| Database prose decode | 26.3 tok/s | 26.4 tok/s |
| Code decode | 33.4 tok/s | 34.8 tok/s |
| TCP prose decode | 26.5 tok/s | 24.4 tok/s |
| Warm TTFT | 0.24–0.30 s | 0.26–0.31 s |
| 500k synthetic-prompt TTFT | 414.3 s | 411.0 s |
| Minimum available RAM during 500k test | 10.24 GiB | 16.95 GiB |

Both profiles completed exactly 500,000 prompt tokens and returned the correct
arithmetic answer. Short-prompt performance was mixed and broadly comparable in
these three single-run samples; generated text differed, affecting MTP acceptance.
This is a capacity and speed comparison, with general reasoning/retrieval quality
equivalence still unverified. FP8 was left running after validation.
See [the FP8 comparison report](docs/performance-fp8-2026-09-05.json) for raw outputs,
usage, timings, configuration, and limitations.

## Hardware

- NVIDIA DGX Spark (GB10, Blackwell sm_121), 128 GB **unified** memory (~116 GiB free)
- 20-core Grace CPU (Cortex-X925 @ 4.0 GHz), aarch64
- Driver 580.82.09, CUDA 13.0, Docker 28.3.3, 3.7 TB NVMe

## Why it fits at all

The official NVIDIA checkpoint contains **123.53 GiB** of tensor data, including
**47.68 GiB** of PLE n-gram embeddings and **2.51 GiB** of MTP weights, measured
from its safetensors headers. Despite the NVFP4 checkpoint name, the PLE table
is **FP8**; main MoE experts use NVFP4 and other weights use mixed precisions.
Each token looks up 16 PLE rows. The default launcher reads the needed rows from
NVMe into staging buffers instead of allocating the entire table on the GPU:

- Model loading uses **76.48 GiB** in the measured TP1 launch. The rest of the
  memory must cover caches, activations, runtime overhead, and the host.
- On unified memory, "CPU offload" saves nothing (same pool) — only serving
  from disk actually frees memory.

## Fresh setup for this deployment

Requirements: Linux with pidfds, Python 3.9+ with `pidfd_send_signal`, systemd
246+ (startup failure mode support), local rootful Docker at
`/var/run/docker.sock`, NVIDIA runtime, the tested immutable image already
present, and the pinned official checkpoint. The source/overlays must match
the tested recipe. A rebuild can produce a different image ID and is **not**
silently accepted as equivalent.

The repository of record is
[`andrebrait/qwen38-flash-next-on-dgx-spark`](https://github.com/andrebrait/qwen38-flash-next-on-dgx-spark),
branch `main`. Install a reviewed source snapshot under
`/home/andre/local-ai/releases/<commit>` and point `/home/andre/local-ai/source`
at it while the service is stopped. Keep the previous snapshot for rollback.
Download and verify the pinned weights with `bash scripts/download-weights.sh`.
For a fresh host, transfer the reviewed image with `docker image save` /
`docker image load` to preserve its image ID. `docker build -f Dockerfile.local-ai`
can rebuild the recipe, but a different resulting image ID requires validation
and an explicit pin update; do not bypass the launcher's identity check.

Docker live restore must be disabled:
`sudo docker info --format '{{.LiveRestoreEnabled}}'` must report `false`.
Provision one cryptographically random, URL-safe key of at least 32 characters:
`api.key` contains the raw key and `api.env` contains only `VLLM_API_KEY=<same-key>`.
Transfer the raw key to the controller over authenticated SSH, directly into its
private key file; do not print it in logs, chat, command arguments or git.

The launcher checks the exact validated `config.json` SHA-256 before creating a
container, including when `MODEL_HOST` points elsewhere. This pins the architecture
and quantization configuration; it is not a fresh checksum of every weight shard.
Weight files must remain the trusted, read-only checkpoint from installation.

With the reviewed source already installed at `/home/andre/local-ai/source` and
the existing credentials provisioned (same URL-safe 32+-character raw key):

```bash
cd /home/andre/local-ai/source
# Disable the old watchdog before replacing its unit. Keep the original baseline stopped.
sudo docker stop --time 30 qwen38-flash
# Must print false. Do not proceed while the baseline is running.
sudo docker inspect --format '{{.State.Running}}' qwen38-flash
sudo systemctl disable --now local-ai-memory.service
sudo chown root:root /home/andre/local-ai/secrets/api.env /home/andre/local-ai/secrets/api.key
sudo chmod 0600 /home/andre/local-ai/secrets/api.env /home/andre/local-ai/secrets/api.key
sudo install -d -o root -g root -m 0700 /var/lib/local-ai-model
sudo install -d -o root -g root -m 0755 /home/andre/local-ai/cache
sudo install -o root -g root -m 0644 scripts/local-ai-memory.service /etc/systemd/system/local-ai-memory.service
sudo systemctl daemon-reload
sudo systemctl enable local-ai-memory.service
sudo systemctl start local-ai-memory.service
sudo systemctl status --no-pager local-ai-memory.service
```

Enabling boot startup is safe only with the persistent latch directory retained:
a clean shutdown permits the next boot; an unclean run refuses to reload.
No key is put on a command line, in the unit, or in supervisor logs. Rootless
`DRY_RUN=1 bash scripts/serve.sh` can inspect the proposed non-mutating command
using private fixture credentials/weights, but is not a startup or safety test.
Do not copy benchmark recipe environments verbatim: their empty API key is
intentionally rejected by this production launcher.
If configuration validation reports `AddressValueError`, check `BIND_HOST`:
it must be an IPv4 loopback address or an address in `100.64.0.0/10`, not an
IPv6 address or a hostname. Invalid values fail closed before CUDA starts.

`scripts/serve-legacy.sh` defaults: native 262,144-token context, MTP=3 speculative tokens,
`--enable-prefix-caching`, deterministic exact QSA top-k, 4 concurrent sequences,
`--gpu-memory-utilization 0.85`, PIECEWISE CUDA graphs (the mmap'd PLE gather is a
splitting op), automatic graph capture widths, 2,048-token prefill chunks,
and a 9 GiB FP8 KV pool using the patched QSA layers.
For the validated single-stream 524k configuration, use `scripts/serve-500k.sh` above.

## Earlier baseline results (single request, greedy)

These measurements predate the tuned MTP=3 single-stream profile above. Their
decode figures include TTFT; use the latest table for current-profile performance.

### MODE=nvfp4 (checkpoint as published)

| Metric | Result |
| --- | --- |
| First boot (weight load) | ~10 min |
| Prefill, cold, 10.7k-token prompt | **1,042 tok/s** |
| Same prompt again (prefix-cache hit) | 1.48 s TTFT |
| Determinism at T=0 | first-token logprobs identical across runs |
| Decode, 400-token real answer incl. TTFT | **19.1 tok/s** |

### MODE=hybrid (NVFP4 experts + blockwise-fp8 side layers)

One-time prep: `scripts/prepare-hybrid.sh` (~10 min, +13 GB disk) — 300 dense
side-layer tensors (GDN in/out projections, QSA q/k/v/o, shared experts) converted
bf16 → fp8-e4m3, worst per-tensor max relative error 3.54%. Routed experts stay
NVFP4.

| Metric | nvfp4 | hybrid |
| --- | --- | --- |
| Prefill, cold, 10.7k-token prompt | 1,042 tok/s | 916 tok/s |
| Prefix-cache hit TTFT | 1.48 s | 1.50 s |
| Deterministic at T=0 | yes | yes |
| Decode, 400-token real answer incl. TTFT | 19.1 tok/s | **21.6 tok/s (+13%)** |

Hybrid trades a little cold-prefill speed for meaningfully faster decode and
~7 GiB less resident weight — the right default for an interactive/agentic box.

## Historical legacy gateway (not used by this deployment)

The following section documents the older `serve-legacy.sh` and
`serve-public.sh` profile only. It does **not** describe the authenticated
Tailscale-only `serve.sh` deployment above. The legacy gateway fronts a
loopback container with `gateway.py`; do not launch it alongside the current node.

```bash
scripts/serve-public.sh              # container + gateway on 0.0.0.0:8080
MODE=hybrid scripts/serve-public.sh  # legacy launcher variables only
GW_PORT=9000 scripts/serve-public.sh
```

The gateway proxies `/v1/*` and `/metrics`, requires `Authorization: Bearer
<key>` on every one of them, and 404s everything else — vLLM's other routes
(`/tokenize`, `/sleep`, the shutdown endpoints) never reach the public
interface. Streaming passes through chunk-by-chunk, so SSE latency is
unaffected. It needs only `aiohttp`, declared inline in the script, so
`uv run scripts/gateway.py` installs nothing permanently.

Keys live in the gateway rather than in the container's argv, which is the
point: rotating one takes effect on the next request instead of restarting a
container that loads ~76 GiB of weights over ~10 minutes. Startup prints the
dashboard URL with its admin token:

```
gateway    0.0.0.0:8080  ->  http://127.0.0.1:18300
api base   http://192.168.0.4:8080/v1
dashboard  http://192.168.0.4:8080/?token=<admin token>
```

The dashboard shows upstream health and the served model, lets you edit the
advertised API base URL (override it if you front the gateway with a tunnel or
domain), and manages keys — create, label, reveal, copy, rotate, revoke, with
per-key request counts and last-used times. It also renders a ready-to-paste
curl and OpenAI-SDK snippet. State lives in `gateway.json` (mode 0600,
gitignored); the admin token is generated on first run and persists there.

Ctrl-C stops the gateway but leaves the container running — it is detached with
`--restart unless-stopped`, and a 10-minute weight load is not worth throwing
away on a terminal hangup. Stop it with `docker rm -f qwen38-flash`.

#### Getting and setting keys

Three equivalent routes, in rough order of convenience:

```bash
# the dashboard: show / copy / rotate / revoke, per key
python3 -c "import json;d=json.load(open('gateway.json'));\
print(f\"http://127.0.0.1:8080/?token={d['admin_token']}\")"

# the admin API
curl -s -H "X-Admin-Token: $TOK" localhost:8080/admin/state          # list
curl -s -H "X-Admin-Token: $TOK" -d '{"label":"laptop"}' \
     -H 'Content-Type: application/json' localhost:8080/admin/keys   # create
curl -s -H "X-Admin-Token: $TOK" -d '{}' \
     localhost:8080/admin/keys/<id>/rotate                           # rotate

# or just edit gateway.json -- the only way to set a *chosen* value,
# since the dashboard and API only generate random ones
```

`gateway.json` is re-read within a second of changing, so a hand-edited key,
`public_url`, or `admin_token` takes effect on the next request with no restart.
Per-key request counters are preserved across a reload, an unparseable file is
ignored (and warned about once) rather than crashing the gateway, and emptying
the `keys` list regenerates one instead of locking everyone out.

**This is bearer auth over plain HTTP.** It is enough for a trusted LAN. Before
exposing it further, put TLS in front of it — a Cloudflare tunnel, Tailscale, or
a reverse proxy — and set the dashboard's API base URL to that public address.

## Reducing memory footprint on a MoE (what actually works here)

- **Disk-backed PLE** (already on): avoids keeping the full 47.68 GiB FP8 table
  resident in the default NVIDIA profile.
- **Host reserve**: `HOST_RESERVE_GIB=30` is the measured default. Increasing it
  reduces the GPU/KV budget. A 36 GiB reserve left only 0.97 GiB usable KV,
  below the 3.84 GiB required for one 262K request, so startup failed.
- **Lower `MAXLEN`/`SEQS`** when less context or concurrency is sufficient.
  The legacy launchers call the context setting `CTX`.
- **Legacy hybrid mode** saves about 7 GiB with its converted side layers.
- Expert streaming/offload is **not** useful on this chip: unified memory means
  host RAM is the same pool, and experts are touched every token so disk paging
  would thrash. vLLM has no expert-mmap path anyway.
- GGUF IQ3/IQ2 quants via llama.cpp shrink further but cost MTP, prefill speed
  (~540 tok/s vs ~1,000+), and quality.

## DFlash and n-gram embeddings

The measured default uses native MTP3 and has DFlash disabled:
`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`.
DFlash, MTP, and n-gram speculation refer to drafting strategies; the default
selects MTP only.
The model's PLE n-gram embeddings are part of the target model and are required
regardless of which speculative method is selected.

PLE NVFP4 storage would require a new packed format, a gather/dequantization
implementation, and model-quality validation. The current loader supports the
FP8 checkpoint table; changing a quantization flag does not convert it.

## Historical recipe limitations

The following limits concern the historical recipes below the deployment
section. The current production launcher is pinned to 262,144 tokens.

- One big model at a time — this uses most of the 128 GB pool.
- 1M context is out of reach on one box; 500k with YaRN is the validated ceiling
  (`YARN=1 CTX=500000 GPU_MEM=0.80 scripts/serve-legacy.sh`).
- The stock GB10 `persistent_topk` kernel is non-deterministic — the image's
  `EXACT_TOPK=1` default fixes it at some long-prefill cost.
