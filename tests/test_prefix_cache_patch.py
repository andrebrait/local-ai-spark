"""CPU regressions executing the pinned upstream methods, not a GPU model.

The two method fixtures below are verbatim from vLLM 8a728663, licensed
Apache-2.0, Copyright contributors to the vLLM project. Only their container
classes and external pool/tensor interfaces are replaced for CPU execution.
"""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prefix_patch", ROOT / "src/patch_prefix_cache.py"
)
patch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(patch)

SEARCH = '''class MambaManager:
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        assert isinstance(kv_cache_spec, MambaSpec), (
            "MambaManager can only be used for mamba groups"
        )
        assert dcp_world_size == 1, "DCP not support mamba now."
        assert pcp_world_size == 1, "PCP not support mamba now."
        block_hashes = resolve_block_hashes(
            block_hashes,
            block_pool.hash_block_size,
            kv_cache_spec.block_size,
            supports_fine_grained_hash_lookup=cls.supports_fine_grained_hash_lookup,
            alignment_tokens=alignment_tokens,
        )
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids))
        )
        hit_length = 0

        block_size = kv_cache_spec.block_size
        if alignment_tokens < block_size and block_size % alignment_tokens == 0:
            # list or lazy BlobBlockHashes view
            assert isinstance(block_hashes, Sequence)
            hash_block_size = alignment_tokens
            scale_factor = block_size // hash_block_size
            max_num_partial_units = min(
                max_length // hash_block_size, len(block_hashes)
            )
            for fine_idx in range(max_num_partial_units - 1, -1, -1):
                num_tokens = (fine_idx + 1) * hash_block_size
                block_hash = block_hashes[fine_idx]
                if cached_block := block_pool.get_cached_block(
                    block_hash, kv_cache_group_ids
                ):
                    block_idx = fine_idx // scale_factor
                    for computed, cached in zip(computed_blocks, cached_block):
                        computed.extend([block_pool.null_block] * block_idx)
                        computed.append(cached)
                    hit_length = num_tokens
                    break
            return computed_blocks, hit_length

        max_num_blocks = max_length // block_size
        # Search from right to left and early stop when a match is found.
        for i in range(max_num_blocks - 1, -1, -1):
            if cached_block := block_pool.get_cached_block(
                block_hashes[i], kv_cache_group_ids
            ):
                # When enable Mamba prefix caching, `block_size` will be aligned
                # across full attention layers and Mamba layers to ensure the
                # prefix hit length aligned at block
                if (
                    block_size != alignment_tokens  # Faster for common case.
                    and (i + 1) * block_size % alignment_tokens != 0
                ):
                    continue
                for computed, cached in zip(computed_blocks, cached_block):
                    # the hit length logic later assumes:
                    #  hit_length = len(hit_blocks_other_attn[0])
                    #               * self.other_block_size
                    # so we insert dummy blocks at the beginning:
                    computed.extend([block_pool.null_block] * i)
                    computed.append(cached)
                hit_length = (i + 1) * block_size
                break  # we just need the last match - early stopping

        return computed_blocks, hit_length
'''

SEED = '''class MambaHybridModelState:
    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        # Must reset the speculative acceptance count in this idx which could be stale.
        self.num_accepted_tokens_gpu[req_index].fill_(1)
        if self._align_mode:
            # Seed the running state block from the resumed/prefilled position.
            self._mamba_state_idx_gpu[req_index].fill_(
                (new_req_data.num_computed_tokens - 1) // self.cache_config.block_size
            )
'''

ANNOTATE = '''def annotate(vllm_config, kv_cache_spec, kv_cache_groups,
             use_deepseek_v4_fallback=False):
    spec_config = vllm_config.speculative_config
    if spec_config is None or not spec_config.use_eagle_block_drop():
        return

    for group in kv_cache_groups:
        if any(
            getattr(spec, "non_causal_multi_token_decode", False)
            for spec in iter_layer_specs(group.kv_cache_spec)
        ):
            group.is_eagle_group = True

    if not use_deepseek_v4_fallback:
        return
    last_layer = next(reversed(kv_cache_spec))
    for group in kv_cache_groups:
        if last_layer in group.layer_names:
            group.is_eagle_group = True
            break
'''


def patched_fixture(source, old, new):
    return patch.prepare_patch(source, patch.sha256(source),
                               patch.sha256(source.replace(old, new)), old, new)


def load_method_container(name, method, **globals_):
    source = "from __future__ import annotations\n" + method.replace(
        f"class {name}:", f"class {name}(Base):", 1
    )
    namespace = {"Base": Base, **globals_}
    exec(compile(ast.parse(source), "<upstream-method>", "exec"), namespace)
    return namespace[name]


class Base:
    supports_fine_grained_hash_lookup = True

    def add_request(self, req_index, new_req_data):
        pass


class Pool:
    null_block = None

    def __init__(self, boundaries, hash_block_size):
        self.hash_block_size = hash_block_size
        self.states = {boundary: object() for boundary in boundaries}

    def get_cached_block(self, block_hash, group_ids):
        state = self.states.get(block_hash)
        return tuple(state for _ in group_ids) if state is not None else None


class Scalar:
    def fill_(self, value):
        self.value = value


class PrefixCachePatchTest(unittest.TestCase):
    def test_only_mtp_groups_require_draft_lookahead(self):
        config = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen4_exp")),
            speculative_config=SimpleNamespace(method="mtp", use_eagle_block_drop=lambda: True),
        )
        names = ["model.layers.0.linear_attn", "model.layers.3.self_attn",
                 "draft_model.mtp.layers.48.self_attn"]
        for fixed in (False, True):
            source = patched_fixture(ANNOTATE, patch.GROUP_OLD, patch.GROUP_NEW) if fixed else ANNOTATE
            namespace = {"iter_layer_specs": lambda spec: [spec]}
            exec(compile(source, "<upstream-annotation>", "exec"), namespace)
            groups = [SimpleNamespace(layer_names=[name], kv_cache_spec=object(),
                                      is_eagle_group=False) for name in names]
            namespace["annotate"](config, {}, groups)
            self.assertEqual([group.is_eagle_group for group in groups], [False, False, fixed])
        with self.assertRaises(ValueError):
            namespace["annotate"](config, {}, groups[:2])

    def search(self, max_length, alignment, drop, boundaries, fixed=True):
        source = SEARCH
        if fixed:
            source = patched_fixture(source, patch.SEARCH_OLD, patch.SEARCH_NEW)
        manager = load_method_container(
            "MambaManager", source, MambaSpec=SimpleNamespace,
            Sequence=list, resolve_block_hashes=lambda hashes, *a, **kw: hashes,
        )
        pool = Pool(boundaries, alignment)
        blocks, length = manager.find_longest_cache_hit(
            list(range(alignment, 81, alignment)), max_length, [0], pool,
            SimpleNamespace(block_size=16), drop, alignment,
        )
        return blocks[0], length, pool

    def test_coarse_hit_restores_earlier_real_state_not_null_padding(self):
        original, length, _ = self.search(80, 16, True, range(16, 81, 16), False)
        self.assertEqual(length, 80)  # The actual upstream defect.
        for drop, expected in ((False, 80), (True, 64)):
            blocks, length, pool = self.search(80, 16, drop, range(16, 81, 16))
            self.assertEqual(length, expected)
            self.assertIs(blocks[-1], pool.states[expected])
            self.assertEqual(blocks[:-1], [None] * (expected // 16 - 1))
        blocks, length, _ = self.search(80, 16, True, [80])
        self.assertEqual((blocks, length), ([], 0))

    def test_partial_lookup_obeys_reduced_token_ceiling(self):
        _, length, _ = self.search(76, 4, True, range(4, 81, 4), False)
        self.assertEqual(length, 76)
        blocks, length, pool = self.search(76, 4, True, range(4, 81, 4))
        self.assertEqual(length, 60)
        self.assertIs(blocks[-1], pool.states[60])
        for length_limit in (0, 8, 16):
            blocks, length, _ = self.search(length_limit, 4, True, range(4, 81, 4))
            self.assertEqual((blocks, length), ([], 0))

    def test_resumed_state_column_uses_resolved_mamba_units(self):
        source = patched_fixture(SEED, patch.SEED_OLD, patch.SEED_NEW)
        for method, expected in ((SEED, 7599), (source, 37)):
            state = load_method_container("MambaHybridModelState", method)()
            state._align_mode = True
            state.cache_config = SimpleNamespace(block_size=8, mamba_block_size=16)
            state._mamba_spec = SimpleNamespace(block_size=1600)
            state._mamba_state_idx_gpu = [Scalar()]
            state.num_accepted_tokens_gpu = [Scalar()]
            state.add_request(0, SimpleNamespace(num_computed_tokens=60800))
            self.assertEqual(state._mamba_state_idx_gpu[0].value, expected)
            self.assertEqual(state.num_accepted_tokens_gpu[0].value, 1)
        state._mamba_spec = None
        state.add_request(0, SimpleNamespace(num_computed_tokens=0))
        self.assertEqual(state._mamba_state_idx_gpu[0].value, -1)

    def test_rerun_and_unreviewed_input_handling(self):
        for source, old, new in ((SEARCH, patch.SEARCH_OLD, patch.SEARCH_NEW),
                                 (ANNOTATE, patch.GROUP_OLD, patch.GROUP_NEW)):
            with self.subTest(patch=old):
                before = patch.sha256(source)
                fixed = source.replace(old, new)
                after = patch.sha256(fixed)
                self.assertEqual(patch.prepare_patch(fixed, before, after, old, new), fixed)
                with self.assertRaises(ValueError):
                    patch.prepare_patch(source + "# downstream edit\n", before, after, old, new)
                with self.assertRaises(ValueError):
                    patch.prepare_patch(source, before, "unexpected output digest", old, new)


if __name__ == "__main__":
    unittest.main()
