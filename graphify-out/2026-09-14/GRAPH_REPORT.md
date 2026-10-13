# Graph Report - production-two-slot  (2026-09-14)

## Corpus Check
- 50 files · ~129,597 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1097 nodes · 1741 edges · 103 communities (51 shown, 52 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 63 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `66cacf96`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- mamba_utils_guarded.py
- ModelOptFp8Config
- Qwen4ExpQSAFlashAttentionBackend
- RoutedExperts
- vllm_ple_mmap.py
- gateway.py
- Qwen4ExpMTP
- LifecycleTest
- Platform
- require
- .reject
- Running Qwen3.8-Flash-Next (NVFP4 + MTP) on a single NVIDIA DGX Spark
- RuntimeError
- Qwen4ExpNGramEmbedding
- _make_proxy_class
- PLEMmapTable
- test_prefix_cache_patch.py
- ModelOptMixedPrecisionConfig
- Qwen4ExpPLELayer
- CompilationConfig
- FormatScheme
- ops_ple.py
- CUDAGraphMode
- .__init__
- field_validator
- serve-500k.sh
- dtype
- PassConfig
- Tensor
- ._align_hybrid_block_size
- ModelOptLinearMethod
- Qwen4ExpPLEFp8EmbeddingMethod
- .create_weights
- ModelOptMxFp8Config
- compilation.py
- .device_control_id_to_physical_device_id
- .get_device_capability
- patch_prefix_cache.py
- MiaAI optimization integration, 2026-09-10
- CkptCtx
- DeviceCapability
- ServeConfigTest
- .get_all_gpu_pci_bus_ids
- validate-context.py
- .get_current_memory_usage
- .make_synced_weight_loader
- fp8_convert.py
- serve-500k-bf16.sh
- serve-500k-fp8.sh
- ModelOptNvFp4FusedMoE
- .pre_register_and_update
- platforms_interface.py
- serve.sh
- serve-legacy.sh
- .check_runner_kv_caches_multi_layer
- .stateless_init_device_torch_dist_pg
- patch_qsa_fp8_kv.py
- generate_social_card.py
- bench-single-stream.py
- download-weights.sh
- prepare-hybrid.sh
- serve-public.sh
- smoke-test.sh
- .get_compile_backend
- .get_default_ir_op_priority
- .apply_config_platform_defaults
- .get_nixl_memory_type
- .check_and_update_config
- .get_pass_manager_cls
- ._ple_mmap_table
- .get_lora_vocab_padding_size
- .import_ir_kernels
- .support_hybrid_kv_cache
- .get_device_communicator_cls
- .get_device_name
- .get_device_total_memory
- .get_device_uuid
- .use_custom_allreduce
- .get_nixl_supported_devices
- .get_punica_wrapper
- .get_static_graph_wrapper_cls
- .use_custom_op_collectives
- .is_arch_support_pdl
- .inference_mode
- .is_cuda_alike
- .register_custom_kv_cache_specs
- .is_integrated_gpu
- .num_compute_units
- .supports_fp8
- .supports_mx
- .support_deep_gemm
- .verify_quantization
- .support_static_graph_mode
- ModelOptQuantConfigBase
- ._make_layer_multipliers
- modelopt.py
- KFp8StaticTensor
- .verify_model_arch
- upstream-overlays/PROVENANCE.md
- patch_mamba_block_size.py
- patch_qsa_exact_topk.py
- test_qsa_exact_topk_cpu.py
- watch-memory.py

## God Nodes (most connected - your core abstractions)
1. `Platform` - 86 edges
2. `LifecycleTest` - 31 edges
3. `supervise()` - 23 edges
4. `CompilationConfig` - 23 edges
5. `Refusal` - 22 edges
6. `Qwen4ExpNGramEmbedding` - 22 edges
7. `require()` - 17 edges
8. `ModelOptMixedPrecisionConfig` - 16 edges
9. `CUDAGraphMode` - 15 edges
10. `preprocess_mamba()` - 15 edges

## Surprising Connections (you probably didn't know these)
- `do_GET()` --indirect_call--> `target()`  [INFERRED]
  tests/test_acceptance_transport.py → scripts/watch-memory.py
- `checked_send()` --calls--> `send()`  [INFERRED]
  tests/test_watch_memory.py → scripts/watch-memory.py
- `CompilationConfig` --references--> `config()`  [EXTRACTED]
  src/full-recipe-patch/compilation.py → scripts/watch-memory.py
- `DynamicShapesConfig` --references--> `config()`  [EXTRACTED]
  src/full-recipe-patch/compilation.py → scripts/watch-memory.py
- `PassConfig` --references--> `config()`  [EXTRACTED]
  src/full-recipe-patch/compilation.py → scripts/watch-memory.py

## Import Cycles
- None detected.

## Communities (103 total, 52 thin omitted)

### Community 0 - "mamba_utils_guarded.py"
Cohesion: 0.06
Nodes (61): CacheConfig, CachedRequestState, CpuGpuBuffer, GPUInputBatch, int64, KVCacheConfig, KVCacheGroupSpec, MambaSpec (+53 more)

### Community 1 - "ModelOptFp8Config"
Cohesion: 0.11
Nodes (8): FusedMoEConfig, algos_owned_by(), ModelOptFp8Config, ModelOptNvFp4Config, Any, The linear algos a given config accepts, for its quant_algo validation., Config class for ModelOpt FP8., Config class for ModelOpt FP4.

### Community 2 - "Qwen4ExpQSAFlashAttentionBackend"
Cohesion: 0.05
Nodes (33): AttentionBackend, AttentionLayerBase, AttentionSpec, CacheDType, FlashAttentionBackend, FlashAttentionImpl, FlashAttentionMetadata, FlashAttentionMetadataBuilder (+25 more)

### Community 3 - "RoutedExperts"
Cohesion: 0.15
Nodes (11): FusedMoEMethodBase, FusedMoEQuantConfig, RoutedExperts, SharedExperts, ModelOptFp8MoEMethod, ModelOptMxFp8FusedMoE, Tensor, FlashInfer TRTLLM MXFP8 block-scale MoE for ModelOpt checkpoints. (+3 more)

### Community 4 - "vllm_ple_mmap.py"
Cohesion: 0.08
Nodes (29): ndarray, CPU unit test for vllm_ple_mmap: synthetic FP8 shards -> gather == reference.…, apply(), __init__(), load_weights(), _setup_table(), enabled(), _env_int() (+21 more)

### Community 5 - "gateway.py"
Cohesion: 0.10
Nodes (30): middleware, admin_guard(), admin_ok(), api_config(), api_key_op(), api_keys(), api_state(), bearer() (+22 more)

### Community 6 - "Qwen4ExpMTP"
Cohesion: 0.10
Nodes (21): IntermediateTensors, Qwen4ExpMixtureOfExperts, _draft_vocab_ids(), _make_draft_vllm_config(), Tensor, VllmConfig, Qwen4ExpMTP, remap_weight_names() (+13 more)

### Community 7 - "LifecycleTest"
Cohesion: 0.06
Nodes (5): LifecycleTest, stale_inspect(), checked_send(), inspect(), CPU-only lifecycle proofs: fake Docker metadata, real sacrificial pidfds. No…

### Community 8 - "Platform"
Cohesion: 0.06
Nodes (12): Platform, Checks if the platform allows inplace memory updates, Returns whether the preferred FP8 type is FNUZ on the current platform. There…, Whether to use allgather in LogitsProcessor to gather the logits., Returns True if we register attention as one giant opaque custom op on the…, Raises if this request is unsupported on this platform, Check max_model_len for the current platform., Inductor config key for the PassManager custom pass (+4 more)

### Community 9 - "require"
Cohesion: 0.17
Nodes (30): HTTPRedirectHandler, json_answer(), main(), authentication(), chat(), exact_chat(), long_checks(), build() (+22 more)

### Community 10 - ".reject"
Cohesion: 0.12
Nodes (12): KDynamicNoParam, KFp8StaticChannel, KMxfp8Static, KNvfp4Dynamic, KNvfp4Static, QuantKeyScheme, One scheme per QuantKey. Selected by key *content*; the base supplies the…, NVFP4 weight scheme (W4A4 and W4A16 share it). Weight-role only today. (+4 more)

### Community 11 - "Running Qwen3.8-Flash-Next (NVFP4 + MTP) on a single NVIDIA DGX Spark"
Cohesion: 0.08
Nodes (24): Access and operations, Controller model budgets, Default FP8 KV profile and BF16 comparison, DFlash and n-gram embeddings, Earlier baseline results (single request, greedy), Fresh setup for this deployment, Getting and setting keys, Hardware (+16 more)

### Community 12 - "RuntimeError"
Cohesion: 0.08
Nodes (34): EncoderCache, InputBatch, MambaHybridModelState, RequestState, RuntimeError, Any, device, Module (+26 more)

### Community 13 - "Qwen4ExpNGramEmbedding"
Cohesion: 0.16
Nodes (7): Model-runner state for Qwen4Exp PLE inputs., Qwen4ExpNGramEmbedding, Return whether a 64-bit integer is prime., Return the ``count``-th prime strictly greater than ``start``., Build per-head vocabulary sizes, offsets, and total row count., Compute n-gram embedding indices for the current request layout., Gather this step's rows into the staging buffer (called by the model state).

### Community 14 - "_make_proxy_class"
Cohesion: 0.14
Nodes (12): apply(), apply_vllm_mapper(), get_quant_method(), _is_fp8_layer(), _norm(), _scan_fp8_layers(), _enabled(), excluded_quant_config() (+4 more)

### Community 15 - "PLEMmapTable"
Cohesion: 0.16
Nodes (10): memoryview, PLEMmapTable, Tensor, Read rows for ids into staging rows start..start+len(ids) (mv = flat bytes)., Gather rows for flat int64 CPU ids -> (pinned) uint8 [n, row_bytes]. Rows are…, Gather rows for ids (any device) into a GPU tensor of self.dtype. Returns [n,…, Read rows [start, start+count) sequentially into one pinned uint8 tensor…, One PLE n-gram table (all row shards of one PLE layer), mmap-backed. (+2 more)

### Community 16 - "test_prefix_cache_patch.py"
Cohesion: 0.17
Nodes (7): Base, load_method_container(), patched_fixture(), Pool, PrefixCachePatchTest, CPU regressions executing the pinned upstream methods, not a GPU model. The two…, Scalar

### Community 17 - "ModelOptMixedPrecisionConfig"
Cohesion: 0.17
Nodes (8): BaseKVCacheMethod, ModelOptKVCacheMethod, ModelOptMixedPrecisionConfig, Supports loading kv-cache scaling factors from FP8 or NVFP4 checkpoints., Config class for ModelOpt MIXED_PRECISION. Supports checkpoints where different…, Look up the quant_algo for a vLLM-side layer prefix. Tries three strategies in…, Kai/2Wild: group_size of the quantized_layers entry that covers prefix., Return quantize-method based on layer.

### Community 18 - "Qwen4ExpPLELayer"
Cohesion: 0.15
Nodes (7): MambaAttentionBackendEnum, MambaBase, PleShortConvAttentionBackend, PleShortConvAttentionMetadata, dtype, Qwen4ExpPLELayer, Dequantize PLE lookup output.

### Community 19 - "CompilationConfig"
Cohesion: 0.11
Nodes (8): Range, CompilationConfig, Initialize the backend for the compilation config from a vllm config.…, To complete the initialization after cudagraph related configs are set. This…, This method logs the enabled/disabled custom ops and checks that the passed…, Get the compile ranges for the compilation config., Configuration for compilation. You must pass CompilationConfig to VLLMConfig…, Provide a hash that uniquely identifies all the configs that affect the…

### Community 20 - "FormatScheme"
Cohesion: 0.12
Nodes (9): _DropInputScale, FormatScheme, _Fp8PbWoPartialBlock, Optional per-format hooks that compose *around* the QuantKey schemes. Extension…, Run before the key schemes' ``process``., Run after the key schemes' ``process``, before the kernel's., Wrap the kernel's forward. ``kernel_apply(layer, x, bias) -> Tensor``. Default:…, FP8_PB_WO output width that is not a multiple of 128 (a partial trailing block;… (+1 more)

### Community 21 - "ops_ple.py"
Cohesion: 0.23
Nodes (15): ple_conv(), _ple_conv_kernel(), _ple_conv_writeback_kernel(), ple_gate(), _ple_gate(), _ple_gate_fake(), _ple_gate_kernel(), ple_ngram_ids() (+7 more)

### Community 23 - ".__init__"
Cohesion: 0.25
Nodes (7): QuantizeMethodBase, _get_ple_embedding_quant_method(), QuantizationConfig, Qwen4ExpTextConfig, VllmConfig, Qwen4ExpPLEGroupedNorm, Select global-scale FP8 only for quantized PLE checkpoint shards.

### Community 24 - "field_validator"
Cohesion: 0.21
Nodes (7): field_validator, Any, Skip validation if the value is `None` when initialisation is delayed., Enable parsing the `mode` field from string mode names. Accepts both integers…, Enable parsing of the `cudagraph_mode` enum type from string., Enable parsing of the `pass_config` field from a dictionary., Skip validation if the value is `None` when initialisation is delayed.

### Community 25 - "serve-500k.sh"
Cohesion: 0.17
Nodes (11): BIND_ADDR, CTX, GPU_MEM, IMAGE, KV_BYTES, KV_DTYPE, MODE, MTP (+3 more)

### Community 26 - "dtype"
Cohesion: 0.17
Nodes (6): dtype, Return the platform specific values for (-inf, inf), Returns the preferred FP8 type on the current platform. See the documentation…, Check if the dtype is supported by the current platform., Returns the supported dtypes for the current platform., Get the vision attention backend class of a device. NOTE: ViT Attention should…

### Community 27 - "PassConfig"
Cohesion: 0.20
Nodes (5): PassConfig, Configuration for custom Inductor passes. This is separate from general…, Returns the max communication size in bytes for flashinfer allreduce fusion for…, Produces a hash unique to the pass configuration. Any new fields that affect…, Log the enabled custom fusion passes. This is called at the end of VLLMConfig…

### Community 28 - "Tensor"
Cohesion: 0.27
Nodes (9): Tensor, qwen4_exp_compute_ple_ngram_ids(), qwen4_exp_compute_ple_ngram_ids_fake(), _qwen4_exp_ple_mmap_gather(), _qwen4_exp_ple_mmap_gather_fake(), qwen4_exp_ple_short_conv(), qwen4_exp_ple_short_conv_fake(), Compute request-dependent PLE n-gram IDs outside piecewise graphs. (+1 more)

### Community 29 - "._align_hybrid_block_size"
Cohesion: 0.18
Nodes (5): Find the first non-SSM attention backend from model layers., Ensure block_size is compatible with the attention backend. For hybrid models,…, Align block size when several KV dtypes share one block pool. A quantized…, Extra ``block_size`` multiple a sparse indexer needs, else ``None``. The CUDA…, For hybrid attention/mamba models, ensure that the attention page size is >=…

### Community 30 - "ModelOptLinearMethod"
Cohesion: 0.22
Nodes (8): LinearMethodBase, register_weight_loader_v2_supported_method, build_linear_method(), maybe_fuse_global_scales(), ModelOptLinearMethod, alpha = input_global_scale * weight_global_scale, presence-gated. W4A4 has both…, Generic, format-agnostic ModelOpt linear method. Holds a weight scheme + an…, Construct the linear method for ``algo``. Returns a bespoke method if one is…

### Community 31 - "Qwen4ExpPLEFp8EmbeddingMethod"
Cohesion: 0.31
Nodes (5): _ple_mmap_enabled(), Module, Qwen4ExpPLEFp8EmbeddingMethod, FP8 PLE embedding with one global checkpoint scale., Reject FP8 PLE checkpoints without a global scale.

### Community 32 - ".create_weights"
Cohesion: 0.25
Nodes (6): Runtime/model dtypes kernels need — format-agnostic., Layer geometry from create_weights args., Thin family dispatcher on the weight key: nvfp4 / mxfp8 / fp8., RuntimeDtypes, select_linear_kernel(), Shapes

### Community 33 - "ModelOptMxFp8Config"
Cohesion: 0.17
Nodes (4): QuantizationMethods, ModelOptMxFp8Config, Config class for ModelOpt MXFP8., Extract upper-cased quant_algo from a modelopt config. Returns the quant_algo…

### Community 34 - "compilation.py"
Cohesion: 0.17
Nodes (10): CompilationMode, DynamicShapesConfig, DynamicShapesType, # NOTE: When using full cudagraph, instead of setting an empty, # TODO: Piecewise Cuda graph might be enabled, Types of dynamic shapes handling in torch.compile(). see Dynamic shapes and…, Configuration to control/debug torch compile dynamic shapes., The compilation approach used for torch.compile-based compilation of the model. (+2 more)

### Community 35 - ".device_control_id_to_physical_device_id"
Cohesion: 0.29
Nodes (4): Map one device-control env entry to an integer physical device ID., Map a vLLM-local logical device ID to a physical device ID. The input is a…, Map a vLLM-local logical device ID to the current process's visible accelerator…, Map a visible accelerator ordinal (e.g. ``torch.device.index``) to a physical…

### Community 36 - ".get_device_capability"
Cohesion: 0.25
Nodes (4): Stateless version of [torch.cuda.get_device_capability][]. Args: device_id:…, Test whether this platform is compatible with a device capability. The…, Test whether this platform has exactly the specified device capability. The…, Returns True if the device capability is any <major>.x. Mirrors CUDA 13…

### Community 37 - "patch_prefix_cache.py"
Cohesion: 0.39
Nodes (7): apply_patches(), main(), prepare_patch(), Path, Apply only the prefix-cache fixes missing from vLLM 8a728663. Basis: vllm-…, Accept exactly the reviewed original or this patch's complete output., sha256()

### Community 38 - "MiaAI optimization integration, 2026-09-10"
Cohesion: 0.29
Nodes (6): Additional quality checks, Changes, Checks completed, Live serving validation, Memory-budget failure reproduced, MiaAI optimization integration, 2026-09-10

### Community 39 - "CkptCtx"
Cohesion: 0.29
Nodes (6): QuantSpec, CkptCtx, Per-checkpoint facts a QuantKey cannot carry., Register format-level params (after the key schemes' weights)., Turn a sub-config into (QuantSpec, CkptCtx, format_scheme). Strictly read-only…, resolve()

### Community 40 - "DeviceCapability"
Cohesion: 0.16
Nodes (6): DeviceCapability, Any, NamedTuple, Return the global graph pool for this platform., Express device capability as an integer `<major><minor>`. It is assumed that…, Set some additional forward context for the current platform if needs.

### Community 43 - "validate-context.py"
Cohesion: 0.50
Nodes (3): post(), Exercise an exact token count against the local API; save timings and usage.…, tokenize()

### Community 44 - ".get_current_memory_usage"
Cohesion: 0.40
Nodes (3): device, Return the memory usage in bytes., Set the device for the current platform.

### Community 46 - "fp8_convert.py"
Cohesion: 0.50
Nodes (4): block_quant(), main(), Tensor, Convert dense bf16 side-layers of the Intel Flash-Next checkpoint to blockwise…

### Community 47 - "serve-500k-bf16.sh"
Cohesion: 0.50
Nodes (3): KV_BYTES, KV_DTYPE, serve-500k-bf16.sh script

### Community 48 - "serve-500k-fp8.sh"
Cohesion: 0.50
Nodes (3): KV_BYTES, KV_DTYPE, serve-500k-fp8.sh script

### Community 49 - "ModelOptNvFp4FusedMoE"
Cohesion: 0.20
Nodes (5): ModelOptNvFp4FusedMoE, dtype, MoE Method for FP4 Quantization. Args: quant_config: NVFP4 Quant Config, FP4 variants use 'weight_scale_2' pattern for per-tensor weight scales., UnfinalizedMoEOutput

### Community 51 - "platforms_interface.py"
Cohesion: 0.14
Nodes (10): CpuArchEnum, in_wsl(), PlatformEnum, Determine the CPU architecture of the current system. Returns CpuArchEnum…, Checks whether pin memory is available on the current platform., # NOTE: `hasattr(device, key)=True` can only avoid AttributeError,, # NOTE: for the forward part of the model, vLLM has another separate, # TODO: Actually only mi3xx has the sleep mode support now (+2 more)

### Community 69 - "._ple_mmap_table"
Cohesion: 0.22
Nodes (5): _ple_mmap_threads(), _ple_resident_enabled(), device, Resident mode: pull this rank's row range into a GPU uint8 tensor., Load hash buffers and checkpoint-split embedding rows.

### Community 93 - "ModelOptQuantConfigBase"
Cohesion: 0.22
Nodes (5): ModelOptQuantConfigBase, Module, QuantizationConfig, Validate weight and scale dtypes before processing., Check if a layer should be excluded from quantization. Handles both exact…

### Community 95 - "modelopt.py"
Cohesion: 0.29
Nodes (6): Enum, KFp8Block128, # TODO: This special hard coded logic is not needed for quantized checkpoints, 128x128 block-static FP8 weight ('PbWo'). Weight-role only. ModelOpt exports…, # TODO: This special hard coded logic is not needed for quantized checkpoints, Role

### Community 103 - "watch-memory.py"
Cohesion: 0.14
Nodes (30): attach(), available_kib(), compact_host_memory(), config(), confirm_stopped(), docker(), exited(), finish() (+22 more)

## Knowledge Gaps
- **46 isolated node(s):** `download-weights.sh script`, `prepare-hybrid.sh script`, `serve-500k-bf16.sh script`, `KV_DTYPE`, `KV_BYTES` (+41 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 441 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **52 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Platform` connect `Platform` to `dtype`, `._align_hybrid_block_size`, `.device_control_id_to_physical_device_id`, `.get_device_capability`, `DeviceCapability`, `.get_all_gpu_pci_bus_ids`, `.get_current_memory_usage`, `.make_synced_weight_loader`, `.pre_register_and_update`, `platforms_interface.py`, `.check_runner_kv_caches_multi_layer`, `.stateless_init_device_torch_dist_pg`, `.get_compile_backend`, `.get_default_ir_op_priority`, `.apply_config_platform_defaults`, `.get_nixl_memory_type`, `.check_and_update_config`, `.get_pass_manager_cls`, `.get_lora_vocab_padding_size`, `.import_ir_kernels`, `.support_hybrid_kv_cache`, `.get_device_communicator_cls`, `.get_device_name`, `.get_device_total_memory`, `.get_device_uuid`, `.use_custom_allreduce`, `.get_nixl_supported_devices`, `.get_punica_wrapper`, `.get_static_graph_wrapper_cls`, `.use_custom_op_collectives`, `.is_arch_support_pdl`, `.inference_mode`, `.is_cuda_alike`, `.register_custom_kv_cache_specs`, `.is_integrated_gpu`, `.num_compute_units`, `.supports_fp8`, `.supports_mx`, `.support_deep_gemm`, `.verify_quantization`, `.support_static_graph_mode`, `.verify_model_arch`?**
  _High betweenness centrality (0.274) - this node is a cross-community bridge._
- **Why does `Refusal` connect `watch-memory.py` to `RuntimeError`?**
  _High betweenness centrality (0.115) - this node is a cross-community bridge._
- **Why does `config()` connect `watch-memory.py` to `PassConfig`, `ModelOptFp8Config`, `compilation.py`, `CompilationConfig`?**
  _High betweenness centrality (0.079) - this node is a cross-community bridge._
- **What connects `download-weights.sh script`, `prepare-hybrid.sh script`, `serve-500k-bf16.sh script` to the rest of the system?**
  _46 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `mamba_utils_guarded.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06252587991718427 - nodes in this community are weakly interconnected._
- **Should `ModelOptFp8Config` be split into smaller, more focused modules?**
  _Cohesion score 0.1111111111111111 - nodes in this community are weakly interconnected._
- **Should `Qwen4ExpQSAFlashAttentionBackend` be split into smaller, more focused modules?**
  _Cohesion score 0.0517120894479385 - nodes in this community are weakly interconnected._