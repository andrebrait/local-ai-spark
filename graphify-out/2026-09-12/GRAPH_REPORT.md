# Graph Report - local-ai-spark  (2026-09-12)

## Corpus Check
- 48 files · ~124,325 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1015 nodes · 1558 edges · 104 communities (46 shown, 58 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 40 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f2c0f8c4`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- mamba_utils_guarded.py
- ModelOptMixedPrecisionConfig
- Qwen4ExpQSAFlashAttentionBackend
- RoutedExperts
- vllm_ple_mmap.py
- gateway.py
- Qwen4ExpMTP
- platforms_interface.py
- Platform
- require
- .reject
- Running Qwen3.8-Flash-Next (NVFP4 + MTP) on a single NVIDIA DGX Spark
- ops_qsa.py
- Qwen4ExpNGramEmbedding
- _make_proxy_class
- PLEMmapTable
- test_prefix_cache_patch.py
- Qwen4ExpModelState
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
- modelopt.py
- compilation.py
- .device_control_id_to_physical_device_id
- .get_device_capability
- patch_prefix_cache.py
- MiaAI optimization integration, 2026-09-10
- CkptCtx
- ._make_vocab_layout
- ServeConfigTest
- DynamicShapesConfig
- validate-context.py
- .get_current_memory_usage
- .make_synced_weight_loader
- fp8_convert.py
- serve-500k-bf16.sh
- serve-500k-fp8.sh
- KFp8StaticTensor
- .pre_register_and_update
- .get_compile_ranges
- serve.sh
- serve-legacy.sh
- .uses_host_device_handling
- .stateless_init_device_torch_dist_pg
- patch_qsa_fp8_kv.py
- generate_social_card.py
- bench-single-stream.py
- download-weights.sh
- prepare-hybrid.sh
- serve-public.sh
- smoke-test.sh
- watch-memory.py
- .post_init_cudagraph_sizes
- .apply_config_platform_defaults
- .can_update_inplace
- .check_and_update_config
- .check_max_model_len
- .get_all_gpu_pci_bus_ids
- .get_attn_backend_cls
- .get_compile_backend
- .get_default_ir_op_priority
- .get_device_communicator_cls
- .get_device_name
- .get_device_total_memory
- .get_device_uuid
- .get_lora_vocab_padding_size
- .get_nixl_supported_devices
- .get_punica_wrapper
- .get_static_graph_wrapper_cls
- .import_ir_kernels
- .import_kernels
- .inference_mode
- .is_cuda_alike
- .is_fp8_fnuz
- .is_integrated_gpu
- .num_compute_units
- .opaque_attention_op
- .pass_key
- .support_deep_gemm
- .support_hybrid_kv_cache
- .support_static_graph_mode
- .use_all_gather
- .use_custom_allreduce
- .use_custom_op_collectives
- .validate_request
- .verify_model_arch
- .verify_quantization
- upstream-overlays/PROVENANCE.md
- patch_mamba_block_size.py
- patch_qsa_exact_topk.py
- test_qsa_exact_topk_cpu.py
- .test_cross_origin_redirect_never_receives_credential

## God Nodes (most connected - your core abstractions)
1. `Platform` - 86 edges
2. `CompilationConfig` - 23 edges
3. `Qwen4ExpNGramEmbedding` - 22 edges
4. `require()` - 18 edges
5. `ModelOptMixedPrecisionConfig` - 16 edges
6. `CUDAGraphMode` - 15 edges
7. `preprocess_mamba()` - 15 edges
8. `main()` - 15 edges
9. `Qwen4ExpQSAFlashAttentionBackend` - 14 edges
10. `Qwen4ExpPLELayer` - 13 edges

## Surprising Connections (you probably didn't know these)
- `Qwen4ExpModelState` --uses--> `Qwen4ExpNGramEmbedding`  [INFERRED]
  src/full-recipe-patch/model_state.py → src/full-recipe-patch/ple_layer.py
- `_get_ple_embedding_quant_method()` --uses--> `ModelOptMixedPrecisionConfig`  [INFERRED]
  src/full-recipe-patch/ple_layer.py → src/full-recipe-patch/upstream-overlays/modelopt.py

## Import Cycles
- None detected.

## Communities (104 total, 58 thin omitted)

### Community 0 - "mamba_utils_guarded.py"
Cohesion: 0.06
Nodes (61): CacheConfig, CachedRequestState, CpuGpuBuffer, GPUInputBatch, int64, KVCacheConfig, KVCacheGroupSpec, MambaSpec (+53 more)

### Community 1 - "ModelOptMixedPrecisionConfig"
Cohesion: 0.05
Nodes (25): BaseKVCacheMethod, FusedMoEConfig, QuantizationMethods, algos_owned_by(), ModelOptFp8Config, ModelOptKVCacheMethod, ModelOptMixedPrecisionConfig, ModelOptMxFp8Config (+17 more)

### Community 2 - "Qwen4ExpQSAFlashAttentionBackend"
Cohesion: 0.05
Nodes (33): AttentionBackend, AttentionLayerBase, AttentionSpec, CacheDType, FlashAttentionBackend, FlashAttentionImpl, FlashAttentionMetadata, FlashAttentionMetadataBuilder (+25 more)

### Community 3 - "RoutedExperts"
Cohesion: 0.11
Nodes (16): FusedMoEMethodBase, FusedMoEQuantConfig, RoutedExperts, SharedExperts, ModelOptFp8MoEMethod, ModelOptMxFp8FusedMoE, ModelOptNvFp4FusedMoE, Tensor (+8 more)

### Community 4 - "vllm_ple_mmap.py"
Cohesion: 0.08
Nodes (29): ndarray, CPU unit test for vllm_ple_mmap: synthetic FP8 shards -> gather == reference.…, apply(), __init__(), load_weights(), _setup_table(), enabled(), _env_int() (+21 more)

### Community 5 - "gateway.py"
Cohesion: 0.10
Nodes (30): middleware, admin_guard(), admin_ok(), api_config(), api_key_op(), api_keys(), api_state(), bearer() (+22 more)

### Community 6 - "Qwen4ExpMTP"
Cohesion: 0.10
Nodes (21): IntermediateTensors, Qwen4ExpMixtureOfExperts, _draft_vocab_ids(), _make_draft_vllm_config(), Tensor, VllmConfig, Qwen4ExpMTP, remap_weight_names() (+13 more)

### Community 7 - "platforms_interface.py"
Cohesion: 0.07
Nodes (18): CpuArchEnum, DeviceCapability, in_wsl(), PlatformEnum, Any, NamedTuple, Determine the CPU architecture of the current system. Returns CpuArchEnum…, Checks whether pin memory is available on the current platform. (+10 more)

### Community 8 - "Platform"
Cohesion: 0.07
Nodes (9): Platform, Returns whether the current platform supports MX types., Returns whether the current platform supports FP8 types., Check whether the platform's ModelRunner can handle multiple attention layers…, Returns the nixl memory type for the current platform., Does the current platform support PDL (Programmatic Dependent Launch)?, Get the pass manager class for this platform. It will be registered as a custom…, Set RNG seed across all devices for the current platform. (+1 more)

### Community 9 - "require"
Cohesion: 0.17
Nodes (31): HTTPRedirectHandler, json_answer(), main(), authentication(), chat(), exact_chat(), long_checks(), build() (+23 more)

### Community 10 - ".reject"
Cohesion: 0.12
Nodes (12): KDynamicNoParam, KFp8StaticChannel, KMxfp8Static, KNvfp4Dynamic, KNvfp4Static, QuantKeyScheme, One scheme per QuantKey. Selected by key *content*; the base supplies the…, NVFP4 weight scheme (W4A4 and W4A16 share it). Weight-role only today. (+4 more)

### Community 11 - "Running Qwen3.8-Flash-Next (NVFP4 + MTP) on a single NVIDIA DGX Spark"
Cohesion: 0.08
Nodes (22): Access and operations, Default FP8 KV profile and BF16 comparison, DFlash and n-gram embeddings, Earlier baseline results (single request, greedy), Fresh setup for this deployment, Getting and setting keys, Hardware, Historical legacy gateway (not used by this deployment) (+14 more)

### Community 12 - "ops_qsa.py"
Cohesion: 0.17
Nodes (20): _compress_qsa_groups_kernel(), _nvfp4_decode_e2m1(), _nvfp4_scale_coord(), _nvfp4_scale_to_float(), constexpr, jit, Tensor, qsa_compress_groups_with_ratio() (+12 more)

### Community 13 - "Qwen4ExpNGramEmbedding"
Cohesion: 0.16
Nodes (8): _ple_mmap_threads(), _ple_resident_enabled(), device, Qwen4ExpNGramEmbedding, Compute n-gram embedding indices for the current request layout., Gather this step's rows into the staging buffer (called by the model state)., Resident mode: pull this rank's row range into a GPU uint8 tensor., Load hash buffers and checkpoint-split embedding rows.

### Community 14 - "_make_proxy_class"
Cohesion: 0.14
Nodes (12): apply(), apply_vllm_mapper(), get_quant_method(), _is_fp8_layer(), _norm(), _scan_fp8_layers(), _enabled(), excluded_quant_config() (+4 more)

### Community 15 - "PLEMmapTable"
Cohesion: 0.16
Nodes (10): memoryview, PLEMmapTable, Tensor, Read rows for ids into staging rows start..start+len(ids) (mv = flat bytes)., Gather rows for flat int64 CPU ids -> (pinned) uint8 [n, row_bytes]. Rows are…, Gather rows for ids (any device) into a GPU tensor of self.dtype. Returns [n,…, Read rows [start, start+count) sequentially into one pinned uint8 tensor…, One PLE n-gram table (all row shards of one PLE layer), mmap-backed. (+2 more)

### Community 16 - "test_prefix_cache_patch.py"
Cohesion: 0.17
Nodes (7): Base, load_method_container(), patched_fixture(), Pool, PrefixCachePatchTest, CPU regressions executing the pinned upstream methods, not a GPU model. The two…, Scalar

### Community 17 - "Qwen4ExpModelState"
Cohesion: 0.17
Nodes (12): EncoderCache, InputBatch, MambaHybridModelState, RequestState, Any, device, Module, Tensor (+4 more)

### Community 18 - "Qwen4ExpPLELayer"
Cohesion: 0.15
Nodes (7): MambaAttentionBackendEnum, MambaBase, PleShortConvAttentionBackend, PleShortConvAttentionMetadata, dtype, Qwen4ExpPLELayer, Dequantize PLE lookup output.

### Community 19 - "CompilationConfig"
Cohesion: 0.15
Nodes (5): CompilationConfig, Initialize the backend for the compilation config from a vllm config.…, This method logs the enabled/disabled custom ops and checks that the passed…, Configuration for compilation. You must pass CompilationConfig to VLLMConfig…, Provide a hash that uniquely identifies all the configs that affect the…

### Community 20 - "FormatScheme"
Cohesion: 0.12
Nodes (9): _DropInputScale, FormatScheme, _Fp8PbWoPartialBlock, Optional per-format hooks that compose *around* the QuantKey schemes. Extension…, Run before the key schemes' ``process``., Run after the key schemes' ``process``, before the kernel's., Wrap the kernel's forward. ``kernel_apply(layer, x, bias) -> Tensor``. Default:…, FP8_PB_WO output width that is not a multiple of 128 (a partial trailing block;… (+1 more)

### Community 21 - "ops_ple.py"
Cohesion: 0.23
Nodes (15): ple_conv(), _ple_conv_kernel(), _ple_conv_writeback_kernel(), ple_gate(), _ple_gate(), _ple_gate_fake(), _ple_gate_kernel(), ple_ngram_ids() (+7 more)

### Community 23 - ".__init__"
Cohesion: 0.19
Nodes (8): _get_ple_embedding_quant_method(), QuantizationConfig, Qwen4ExpTextConfig, VllmConfig, Qwen4ExpPLEGroupedNorm, Select global-scale FP8 only for quantized PLE checkpoint shards., Mix an integer into a deterministic unsigned 64-bit value., Build deterministic hash multipliers for one PLE layer.

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
Cohesion: 0.27
Nodes (6): QuantizeMethodBase, _ple_mmap_enabled(), Module, Qwen4ExpPLEFp8EmbeddingMethod, FP8 PLE embedding with one global checkpoint scale., Reject FP8 PLE checkpoints without a global scale.

### Community 32 - ".create_weights"
Cohesion: 0.25
Nodes (6): Runtime/model dtypes kernels need — format-agnostic., Layer geometry from create_weights args., Thin family dispatcher on the weight key: nvfp4 / mxfp8 / fp8., RuntimeDtypes, select_linear_kernel(), Shapes

### Community 33 - "modelopt.py"
Cohesion: 0.29
Nodes (6): Enum, KFp8Block128, # TODO: This special hard coded logic is not needed for quantized checkpoints, 128x128 block-static FP8 weight ('PbWo'). Weight-role only. ModelOpt exports…, # TODO: This special hard coded logic is not needed for quantized checkpoints, Role

### Community 34 - "compilation.py"
Cohesion: 0.25
Nodes (7): CompilationMode, DynamicShapesType, # NOTE: When using full cudagraph, instead of setting an empty, # TODO: Piecewise Cuda graph might be enabled, Types of dynamic shapes handling in torch.compile(). see Dynamic shapes and…, The compilation approach used for torch.compile-based compilation of the model., str

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

### Community 40 - "._make_vocab_layout"
Cohesion: 0.33
Nodes (3): Return whether a 64-bit integer is prime., Return the ``count``-th prime strictly greater than ``start``., Build per-head vocabulary sizes, offsets, and total row count.

### Community 42 - "DynamicShapesConfig"
Cohesion: 0.40
Nodes (4): config, DynamicShapesConfig, Configuration to control/debug torch compile dynamic shapes., Provide a hash for DynamicShapesConfig

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

## Knowledge Gaps
- **43 isolated node(s):** `download-weights.sh script`, `prepare-hybrid.sh script`, `serve-500k-bf16.sh script`, `KV_DTYPE`, `KV_BYTES` (+38 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 414 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **58 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Platform` connect `Platform` to `platforms_interface.py`, `dtype`, `._align_hybrid_block_size`, `.device_control_id_to_physical_device_id`, `.get_device_capability`, `.get_current_memory_usage`, `.make_synced_weight_loader`, `.pre_register_and_update`, `.uses_host_device_handling`, `.stateless_init_device_torch_dist_pg`, `.apply_config_platform_defaults`, `.can_update_inplace`, `.check_and_update_config`, `.check_max_model_len`, `.get_all_gpu_pci_bus_ids`, `.get_attn_backend_cls`, `.get_compile_backend`, `.get_default_ir_op_priority`, `.get_device_communicator_cls`, `.get_device_name`, `.get_device_total_memory`, `.get_device_uuid`, `.get_lora_vocab_padding_size`, `.get_nixl_supported_devices`, `.get_punica_wrapper`, `.get_static_graph_wrapper_cls`, `.import_ir_kernels`, `.import_kernels`, `.inference_mode`, `.is_cuda_alike`, `.is_fp8_fnuz`, `.is_integrated_gpu`, `.num_compute_units`, `.opaque_attention_op`, `.pass_key`, `.support_deep_gemm`, `.support_hybrid_kv_cache`, `.support_static_graph_mode`, `.use_all_gather`, `.use_custom_allreduce`, `.use_custom_op_collectives`, `.validate_request`, `.verify_model_arch`, `.verify_quantization`?**
  _High betweenness centrality (0.224) - this node is a cross-community bridge._
- **Why does `ModelOptMixedPrecisionConfig` connect `ModelOptMixedPrecisionConfig` to `modelopt.py`, `.__init__`?**
  _High betweenness centrality (0.153) - this node is a cross-community bridge._
- **Why does `_get_ple_embedding_quant_method()` connect `.__init__` to `ModelOptMixedPrecisionConfig`, `Tensor`, `Qwen4ExpPLEFp8EmbeddingMethod`?**
  _High betweenness centrality (0.136) - this node is a cross-community bridge._
- **What connects `download-weights.sh script`, `prepare-hybrid.sh script`, `serve-500k-bf16.sh script` to the rest of the system?**
  _43 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `mamba_utils_guarded.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06252587991718427 - nodes in this community are weakly interconnected._
- **Should `ModelOptMixedPrecisionConfig` be split into smaller, more focused modules?**
  _Cohesion score 0.05004389815627744 - nodes in this community are weakly interconnected._
- **Should `Qwen4ExpQSAFlashAttentionBackend` be split into smaller, more focused modules?**
  _Cohesion score 0.0517120894479385 - nodes in this community are weakly interconnected._