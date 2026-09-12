#!/usr/bin/env python3
"""Apply only the prefix-cache fixes missing from vLLM 8a728663.

Basis: vllm-project/vllm PR #48375 and issues #54173 / #53142.
The max_length variant also covers this revision's fine-grained cache lookup.
PR #53906's prefix-cacheable-only engine block-size minimum is already upstream;
verify it rather than applying the older patch_mamba_block_size.py recipe.

Usage: python src/patch_prefix_cache.py /usr/local/lib/python3.12/dist-packages
Full-file fingerprints deliberately reject other revisions or downstream edits.
No vLLM import (and therefore no GPU) is needed at build time.
"""
import argparse
import ast
import hashlib
from pathlib import Path

REVISION = "8a728663c1c3eeace834a95f5654fa653cc1998c"
ENGINE = "vllm/v1/engine/core.py"
ENGINE_SHA256 = "ef709a037077f48db65d381305a0427c25749bd5dc7997f3dec5b5539f1f0dc4"
SEARCH_OLD = (
    "        block_size = kv_cache_spec.block_size\n"
    "        if alignment_tokens < block_size and block_size % alignment_tokens == 0:\n"
)
SEARCH_NEW = (
    "        block_size = kv_cache_spec.block_size\n"
    "        # PR #48375: exclude speculative state from coarse AND partial hits.\n"
    "        if drop_eagle_block:\n"
    "            max_length = max(0, max_length - block_size)\n"
    "        if alignment_tokens < block_size and block_size % alignment_tokens == 0:\n"
)
SEED_OLD = (
    "            # Seed the running state block from the resumed/prefilled position.\n"
    "            self._mamba_state_idx_gpu[req_index].fill_(\n"
    "                (new_req_data.num_computed_tokens - 1) // self.cache_config.block_size\n"
    "            )\n"
)
SEED_NEW = (
    "            # Issue #53142: state columns use the resolved Mamba group units.\n"
    "            # Before the first forward there is no local prefix to resume.\n"
    "            block_size = (\n"
    "                self._mamba_spec.block_size\n"
    "                if self._mamba_spec is not None\n"
    "                else self.cache_config.block_size\n"
    "            )\n"
    "            self._mamba_state_idx_gpu[req_index].fill_(\n"
    "                (new_req_data.num_computed_tokens - 1) // block_size\n"
    "            )\n"
)
PATCHES = (
    (
        "vllm/v1/core/single_type_kv_cache_manager.py",
        "128b98a0511f67d32f44767aa1658776a8246374b9eb8461d397e683ff3d984d",
        "c84f41aa6f4c6e4c061fbe934194c0e7fb977afa554a942e8dc3db0df9a6d963",
        SEARCH_OLD,
        SEARCH_NEW,
    ),
    (
        "vllm/v1/worker/gpu/model_states/mamba_hybrid.py",
        "44d0a20cee12535ac7f29fc3d14ffca9994af3af84fd17ad93e71acf5bc79066",
        "bcad24da707f2346f80f3b4c13a8f39017c4db8a7df8297219258e810689291e",
        SEED_OLD,
        SEED_NEW,
    ),
)


def sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def prepare_patch(source: str, before: str, after: str, old: str, new: str) -> str:
    """Accept exactly the reviewed original or this patch's complete output."""
    digest = sha256(source)
    if digest == after:
        if source.count(new) != 1 or source.count(old) != 0:
            raise ValueError("already-fixed source has unexpected patch counts")
        ast.parse(source)
        return source
    if digest != before:
        raise ValueError(f"source does not match pinned vLLM {REVISION}: {digest}")
    if source.count(old) != 1 or source.count(new) != 0:
        raise ValueError("expected exactly one original patch site and no patched site")
    ast.parse(source)
    result = source.replace(old, new, 1)
    ast.parse(result)
    if sha256(result) != after:
        raise ValueError("patched source fingerprint differs from reviewed output")
    return result


def apply_patches(site_packages: Path) -> None:
    engine_source = (site_packages / ENGINE).read_bytes().decode("utf-8")
    if sha256(engine_source) != ENGINE_SHA256:
        raise ValueError(f"{ENGINE}: expected the pinned upstream #53906 geometry fix")
    ast.parse(engine_source)
    pending = []
    # Validate every input/output before changing either installed source file.
    for relative, before, after, old, new in PATCHES:
        path = site_packages / relative
        source = path.read_bytes().decode("utf-8")
        try:
            result = prepare_patch(source, before, after, old, new)
        except (ValueError, SyntaxError) as error:
            raise ValueError(f"{relative}: {error}") from error
        pending.append((path, source, result))
    print("Already upstream: #53906 excludes non-prefix-cacheable engine groups")
    for path, source, result in pending:
        if result == source:
            print(f"Already fixed: {path}")
        else:
            path.write_bytes(result.encode("utf-8"))
            print(f"Patched: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site_packages", type=Path)
    args = parser.parse_args()
    try:
        apply_patches(args.site_packages)
    except (OSError, ValueError, SyntaxError) as error:
        parser.exit(1, f"Prefix-cache patch refused: {error}\n")


if __name__ == "__main__":
    main()
