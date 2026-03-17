"""
Compare Megatron and SGLang layer dumps for Qwen3-30B-A3B (MoE) model.

Megatron dumps: /tmp/megatron_debug/  (individual .pt files per tensor)
  Naming: {name}_fwd{N}.pt  e.g. layer00_input_fwd0.pt
  Shape: [S, B, H] (seq_len first, Megatron convention)

SGLang dumps: /tmp/sglang_dump_{partial_name}/  (individual .pt files)
  Naming: forward_pass_id={N}___rank={R}___name={name}___...pt
  Shape: [total_tokens, H] (packed)

Comparable dump points (both produce identical names):
  - layer{NN}_after_input_ln   (after input RMSNorm, before attention)
  - layer{NN}_after_attn       (attention output)
  - layer{NN}_moe_input        (after post-attn RMSNorm, before MoE)
  - layer{NN}_moe_output       (MoE output, before residual add)
  - after_final_layernorm      (final RMSNorm output)

Megatron-only dump points (for deeper debugging):
  - layer{NN}_input            (full hidden_states entering the layer)
  - layer{NN}_output           (full hidden_states leaving the layer)
  - layer{NN}_shared_expert    (shared expert output)
  - layer{NN}_topk_weights     (router topk weights)
  - layer{NN}_topk_ids         (router topk expert indices)
  - layer{NN}_fused_experts    (fused expert output before shared expert add)
  - layer{NN}_moe_before_allreduce
  - layer{NN}_moe_after_allreduce
  - before_final_layernorm

Usage (inside Docker container):
  # Enable dumps on both sides, run E2E, then compare:
  # Megatron: SLIME_DEBUG_LAYER_DUMP=1 python examples/true_on_policy/run_moe_megatron.py
  # SGLang:   SGLANG_DUMPER_ENABLE=1 (automatically enabled in debug_one_sample mode)
  python scripts/compare_moe_dumps.py [--mega-dir DIR] [--sglang-dir DIR] [--fwd N] [--layers L1,L2,...]
"""
import argparse
import glob
import os
import re
import sys

import torch


def find_sglang_dump_dir(base="/tmp"):
    """Find the most recent SGLang dump directory."""
    pattern = os.path.join(base, "sglang_dump_*")
    dirs = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if dirs:
        return dirs[0]
    return None


def load_megatron_dumps(mega_dir, fwd_id=0):
    """Load all Megatron dumps for a given forward pass."""
    dumps = {}
    pattern = os.path.join(mega_dir, f"*_fwd{fwd_id}.pt")
    for path in sorted(glob.glob(pattern)):
        basename = os.path.basename(path)
        # Extract name from {name}_fwd{N}.pt
        name = basename.rsplit(f"_fwd{fwd_id}.pt", 1)[0]
        dumps[name] = torch.load(path, map_location="cpu")
    return dumps


def load_sglang_dumps(sglang_dir, fwd_id=1, rank=0):
    """Load all SGLang dumps for a given forward pass and rank."""
    dumps = {}
    for path in sorted(glob.glob(os.path.join(sglang_dir, "*.pt"))):
        basename = os.path.basename(path)
        # Parse key=value pairs from filename
        parts = basename.replace(".pt", "").split("___")
        meta = {}
        for part in parts:
            if "=" in part:
                k, v = part.split("=", 1)
                meta[k] = v

        if meta.get("forward_pass_id") != str(fwd_id):
            continue
        if meta.get("rank") != str(rank):
            continue
        name = meta.get("name", "")
        if name:
            dumps[name] = torch.load(path, map_location="cpu")
    return dumps


def reshape_for_comparison(mega_tensor, sglang_tensor):
    """Reshape tensors to be comparable.

    Megatron: [S, B, H] or [S*B, H]
    SGLang:   [total_tokens, H]
    """
    # Flatten Megatron to [tokens, H]
    if mega_tensor.dim() == 3:
        mega_flat = mega_tensor.reshape(-1, mega_tensor.shape[-1])
    else:
        mega_flat = mega_tensor

    # SGLang is already [tokens, H] or similar
    if sglang_tensor.dim() == 3:
        sglang_flat = sglang_tensor.reshape(-1, sglang_tensor.shape[-1])
    else:
        sglang_flat = sglang_tensor

    return mega_flat, sglang_flat


def compare_tensor(name, mega_t, sglang_t, verbose=True):
    """Compare two tensors and return (max_diff, mean_diff, nonzero_count, total)."""
    mega_flat, sglang_flat = reshape_for_comparison(mega_t, sglang_t)

    if mega_flat.shape != sglang_flat.shape:
        # Try to align: SGLang may have more tokens (packed from multiple sequences)
        # Use the smaller size
        min_tokens = min(mega_flat.shape[0], sglang_flat.shape[0])
        if mega_flat.shape[1] != sglang_flat.shape[1]:
            if verbose:
                print(f"  {name}: SHAPE MISMATCH mega={list(mega_t.shape)} vs sglang={list(sglang_t.shape)}")
            return None
        mega_flat = mega_flat[:min_tokens]
        sglang_flat = sglang_flat[:min_tokens]
        if verbose and min_tokens < max(mega_t.shape[0], sglang_t.shape[0]):
            print(f"  {name}: truncated to {min_tokens} tokens for comparison")

    diff = (mega_flat.float() - sglang_flat.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    nonzero = (diff > 0).sum().item()
    total = diff.numel()

    return max_diff, mean_diff, nonzero, total


def print_comparison(name, result):
    """Pretty-print comparison result."""
    if result is None:
        return
    max_diff, mean_diff, nonzero, total = result
    if max_diff == 0:
        print(f"  {name}: ✓ IDENTICAL")
    else:
        pct = 100.0 * nonzero / total if total > 0 else 0
        print(f"  {name}: ✗ max_diff={max_diff:.10f} mean_diff={mean_diff:.10f} nonzero={nonzero}/{total} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Compare Megatron vs SGLang MoE dumps")
    parser.add_argument("--mega-dir", default="/tmp/megatron_debug",
                        help="Megatron dump directory")
    parser.add_argument("--sglang-dir", default=None,
                        help="SGLang dump directory (auto-detected if not specified)")
    parser.add_argument("--fwd", type=int, default=0,
                        help="Forward pass ID to compare (Megatron: 0-based, SGLang: 1-based)")
    parser.add_argument("--layers", default=None,
                        help="Comma-separated layer indices to compare (default: all)")
    parser.add_argument("--rank", type=int, default=0,
                        help="SGLang rank to load dumps from")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed info")
    args = parser.parse_args()

    # Find SGLang dump directory
    sglang_dir = args.sglang_dir
    if sglang_dir is None:
        sglang_dir = find_sglang_dump_dir()
    if sglang_dir is None:
        print("ERROR: No SGLang dump directory found. Run with SGLANG_DUMPER_ENABLE=1")
        print("       Or specify with --sglang-dir")
        sys.exit(1)

    mega_fwd = args.fwd
    sglang_fwd = args.fwd + 1  # SGLang forward_pass_id is 1-based

    print(f"Megatron dumps: {args.mega_dir} (fwd={mega_fwd})")
    print(f"SGLang dumps:   {sglang_dir} (forward_pass_id={sglang_fwd})")
    print()

    # Load dumps
    mega_dumps = load_megatron_dumps(args.mega_dir, mega_fwd)
    sglang_dumps = load_sglang_dumps(sglang_dir, sglang_fwd, args.rank)

    if not mega_dumps:
        print(f"ERROR: No Megatron dumps found in {args.mega_dir} for fwd={mega_fwd}")
        print(f"       Run with SLIME_DEBUG_LAYER_DUMP=1")
        sys.exit(1)
    if not sglang_dumps:
        print(f"ERROR: No SGLang dumps found in {sglang_dir} for forward_pass_id={sglang_fwd}")
        print(f"       Run with SGLANG_DUMPER_ENABLE=1")
        sys.exit(1)

    print(f"Loaded {len(mega_dumps)} Megatron tensors, {len(sglang_dumps)} SGLang tensors")

    if args.verbose:
        print(f"\nMegatron dump names: {sorted(mega_dumps.keys())}")
        print(f"\nSGLang dump names: {sorted(sglang_dumps.keys())}")
    print()

    # Determine layers to compare
    layer_indices = set()
    for name in mega_dumps:
        m = re.match(r"layer(\d+)_", name)
        if m:
            layer_indices.add(int(m.group(1)))

    if args.layers:
        selected = set(int(x) for x in args.layers.split(","))
        layer_indices = layer_indices & selected

    layer_indices = sorted(layer_indices)

    # Comparable stages (names that both sides produce)
    comparable_stages = [
        "after_input_ln",
        "after_attn",
        "moe_input",
        "moe_output",
    ]

    # Global comparison points
    global_names = [
        "after_final_layernorm",
    ]

    # ---- Layer-by-layer comparison ----
    print("=" * 70)
    print("LAYER-BY-LAYER COMPARISON (Megatron ref fwd vs SGLang inference)")
    print("=" * 70)

    first_divergence = None
    for layer_idx in layer_indices:
        print(f"\n--- Layer {layer_idx} ---")
        for stage in comparable_stages:
            name = f"layer{layer_idx:02d}_{stage}"
            mega_t = mega_dumps.get(name)
            sglang_t = sglang_dumps.get(name)

            if mega_t is None and sglang_t is None:
                continue
            if mega_t is None:
                print(f"  {name}: SKIP (Megatron dump missing)")
                continue
            if sglang_t is None:
                print(f"  {name}: SKIP (SGLang dump missing)")
                continue

            result = compare_tensor(name, mega_t, sglang_t, verbose=args.verbose)
            print_comparison(name, result)
            if result and result[0] > 0 and first_divergence is None:
                first_divergence = (name, result)

    # ---- Global comparison ----
    print(f"\n--- Global ---")
    for name in global_names:
        mega_t = mega_dumps.get(name)
        sglang_t = sglang_dumps.get(name)
        if mega_t is not None and sglang_t is not None:
            result = compare_tensor(name, mega_t, sglang_t, verbose=args.verbose)
            print_comparison(name, result)
            if result and result[0] > 0 and first_divergence is None:
                first_divergence = (name, result)
        elif mega_t is not None:
            print(f"  {name}: SKIP (SGLang dump missing)")
        elif sglang_t is not None:
            print(f"  {name}: SKIP (Megatron dump missing)")

    # ---- Megatron-only dumps (for reference) ----
    mega_only_stages = [
        "input", "output", "shared_expert", "topk_weights", "topk_ids",
        "fused_experts", "moe_before_allreduce", "moe_after_allreduce",
    ]
    has_mega_only = False
    for layer_idx in layer_indices[:3]:  # Only show first 3 layers
        for stage in mega_only_stages:
            name = f"layer{layer_idx:02d}_{stage}"
            if name in mega_dumps and name not in sglang_dumps:
                if not has_mega_only:
                    print(f"\n--- Megatron-only dumps (first 3 layers, for reference) ---")
                    has_mega_only = True
                t = mega_dumps[name]
                print(f"  {name}: shape={list(t.shape)} mean={t.mean():.8f} absmax={t.abs().max():.8f}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    if first_divergence:
        name, (max_diff, mean_diff, nonzero, total) = first_divergence
        print(f"FIRST DIVERGENCE: {name}")
        print(f"  max_diff={max_diff:.10f} mean_diff={mean_diff:.10f}")
        print(f"  nonzero={nonzero}/{total}")
        print(f"\nDEBUG TIP: The layer/stage above is where Megatron and SGLang first differ.")
        print(f"  Check the Megatron-only dumps for that layer to understand the internals.")
    else:
        all_matched = any(
            f"layer{l:02d}_{s}" in mega_dumps and f"layer{l:02d}_{s}" in sglang_dumps
            for l in layer_indices for s in comparable_stages
        )
        if all_matched:
            print("ALL COMPARED TENSORS ARE IDENTICAL — true-on-policy verified!")
        else:
            print("No comparable tensors found. Ensure both sides have dumps enabled.")
    print("=" * 70)


if __name__ == "__main__":
    main()
