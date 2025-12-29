#!/usr/bin/env python3
"""
Compare tensor dumps from SGLang and Megatron for debugging numerical differences.

Usage:
    python compare_tensor_dumps.py \
        --sglang-dir /tmp/sglang_tensor_dump \
        --megatron-dir /tmp/megatron_tensor_dump \
        --pass-id 0

The script will:
1. Load tensor dumps from both directories
2. Map SGLang tensor names to Megatron tensor names
3. Compute and display per-tensor statistics (max diff, mean diff, relative diff)
4. Identify the first layer where differences become significant
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import torch


def find_dump_files(dump_dir: str, pass_id: int) -> list[Path]:
    """Find all dump files for a given pass ID in subdirectories."""
    dump_path = Path(dump_dir)
    if not dump_path.exists():
        return []
    
    files = list(dump_path.glob(f"*/Pass{pass_id:05d}.pt"))
    return sorted(files)


def load_tensors(dump_file: Path) -> dict[str, torch.Tensor]:
    """Load tensors from a dump file."""
    return torch.load(dump_file, map_location="cpu")


def normalize_sglang_name(name: str) -> tuple[str, int | None]:
    """
    Convert SGLang tensor name to a normalized form.
    
    SGLang naming: model.layers.0, model.layers.0.self_attn, model.layers.0.mlp, etc.
    
    Returns: (normalized_name, layer_index)
    """
    # Extract layer number if present
    parts = name.split(".")
    layer_idx = None
    
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            layer_idx = int(parts[i + 1])
            # Get remaining path after layer number
            remaining = ".".join(parts[i + 2:]) if i + 2 < len(parts) else ""
            
            if remaining == "" or remaining == "model":
                return f"layer_{layer_idx}_output", layer_idx
            elif "self_attn" in remaining or "attention" in remaining:
                return f"layer_{layer_idx}_self_attention_output", layer_idx
            elif "mlp" in remaining:
                return f"layer_{layer_idx}_mlp_output", layer_idx
            elif "input_layernorm" in remaining:
                return f"layer_{layer_idx}_input_layernorm", layer_idx
            elif "post_attention_layernorm" in remaining:
                return f"layer_{layer_idx}_post_attention_layernorm", layer_idx
            else:
                return f"layer_{layer_idx}_{remaining}", layer_idx
            break
    
    # Not a layer tensor
    return name, None


def compute_diff_stats(t1: torch.Tensor, t2: torch.Tensor) -> dict[str, float]:
    """Compute difference statistics between two tensors."""
    t1_f = t1.float()
    t2_f = t2.float()
    
    diff = (t1_f - t2_f).abs()
    
    return {
        "max_diff": diff.max().item(),
        "mean_diff": diff.mean().item(),
        "std_diff": diff.std().item() if diff.numel() > 1 else 0.0,
        "rel_diff": (diff / (t1_f.abs() + 1e-8)).mean().item(),
        "t1_mean": t1_f.mean().item(),
        "t2_mean": t2_f.mean().item(),
        "t1_max": t1_f.max().item(),
        "t2_max": t2_f.max().item(),
        "t1_min": t1_f.min().item(),
        "t2_min": t2_f.min().item(),
    }


def compare_dumps(
    sglang_tensors: dict[str, torch.Tensor],
    megatron_tensors: dict[str, torch.Tensor],
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Compare tensor dumps from SGLang and Megatron.
    
    Returns a dict mapping tensor names to comparison stats.
    """
    results = {}
    
    # Build mapping from normalized SGLang names to Megatron names
    sglang_normalized = {}
    for name in sglang_tensors.keys():
        norm_name, layer_idx = normalize_sglang_name(name)
        sglang_normalized[norm_name] = (name, layer_idx)
    
    if verbose:
        print("\n" + "=" * 80)
        print("SGLang Tensors:")
        for name in sorted(sglang_tensors.keys())[:20]:
            t = sglang_tensors[name]
            print(f"  {name}: shape={t.shape}, dtype={t.dtype}")
        if len(sglang_tensors) > 20:
            print(f"  ... and {len(sglang_tensors) - 20} more")
        
        print("\nMegatron Tensors:")
        for name in sorted(megatron_tensors.keys())[:20]:
            t = megatron_tensors[name]
            print(f"  {name}: shape={t.shape}, dtype={t.dtype}")
        if len(megatron_tensors) > 20:
            print(f"  ... and {len(megatron_tensors) - 20} more")
        
        print("\nNormalized Mapping:")
        for norm_name, (orig_name, layer_idx) in sorted(sglang_normalized.items()):
            print(f"  {norm_name} <- {orig_name}")
        print("=" * 80)
    
    # Compare tensors
    matched = 0
    unmatched = 0
    significant_diff_layers = []
    
    for norm_name, (sglang_name, layer_idx) in sorted(sglang_normalized.items()):
        if norm_name not in megatron_tensors:
            if verbose:
                print(f"  {norm_name}: NOT FOUND in Megatron")
            unmatched += 1
            continue
        
        matched += 1
        sglang_t = sglang_tensors[sglang_name]
        megatron_t = megatron_tensors[norm_name]
        
        # Check shape
        if sglang_t.shape != megatron_t.shape:
            results[norm_name] = {
                "match": False,
                "layer_idx": layer_idx,
                "reason": f"Shape mismatch: SGLang {sglang_t.shape} vs Megatron {megatron_t.shape}",
            }
            if verbose:
                print(f"  {norm_name}: SHAPE MISMATCH - SGLang {sglang_t.shape} vs Megatron {megatron_t.shape}")
            continue
        
        # Compute stats
        stats = compute_diff_stats(sglang_t, megatron_t)
        results[norm_name] = {
            "match": stats["max_diff"] < 1e-5,
            "layer_idx": layer_idx,
            **stats,
        }
        
        # Track significant differences
        if stats["max_diff"] >= 1e-5 and layer_idx is not None:
            significant_diff_layers.append((layer_idx, norm_name, stats["max_diff"]))
        
        if verbose:
            match_str = "✓" if stats["max_diff"] < 1e-5 else "✗"
            color = "" if stats["max_diff"] < 1e-5 else "\033[91m"  # Red for mismatch
            end_color = "\033[0m" if color else ""
            print(
                f"  {color}{norm_name}: {match_str} "
                f"max_diff={stats['max_diff']:.6e}, mean_diff={stats['mean_diff']:.6e}, "
                f"rel_diff={stats['rel_diff']:.6e}{end_color}"
            )
    
    if verbose:
        print("\n" + "=" * 80)
        print(f"Summary: {matched} matched, {unmatched} unmatched")
        
        if significant_diff_layers:
            significant_diff_layers.sort(key=lambda x: x[0])
            first_diff = significant_diff_layers[0]
            print(f"\n⚠️  FIRST SIGNIFICANT DIFFERENCE at layer {first_diff[0]}:")
            print(f"   Tensor: {first_diff[1]}")
            print(f"   Max diff: {first_diff[2]:.6e}")
            
            print(f"\nAll layers with significant differences:")
            for layer_idx, tensor_name, max_diff in significant_diff_layers[:10]:
                print(f"   Layer {layer_idx}: {tensor_name} (max_diff={max_diff:.6e})")
            if len(significant_diff_layers) > 10:
                print(f"   ... and {len(significant_diff_layers) - 10} more")
        else:
            print("\n✓ All matched tensors have differences < 1e-5")
        print("=" * 80)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare SGLang and Megatron tensor dumps")
    parser.add_argument("--sglang-dir", type=str, required=True, help="SGLang tensor dump directory")
    parser.add_argument("--megatron-dir", type=str, required=True, help="Megatron tensor dump directory")
    parser.add_argument("--pass-id", type=int, default=0, help="Forward pass ID to compare (default: 0)")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()
    
    print(f"Comparing tensor dumps for pass {args.pass_id}")
    print(f"  SGLang dir:   {args.sglang_dir}")
    print(f"  Megatron dir: {args.megatron_dir}")
    
    # Find dump files
    sglang_files = find_dump_files(args.sglang_dir, args.pass_id)
    megatron_files = find_dump_files(args.megatron_dir, args.pass_id)
    
    if not sglang_files:
        print(f"\nError: No SGLang dump files found for pass {args.pass_id}")
        print(f"  Looking in: {args.sglang_dir}/*/Pass{args.pass_id:05d}.pt")
        sys.exit(1)
    
    if not megatron_files:
        print(f"\nError: No Megatron dump files found for pass {args.pass_id}")
        print(f"  Looking in: {args.megatron_dir}/*/Pass{args.pass_id:05d}.pt")
        sys.exit(1)
    
    print(f"\nFound files:")
    print(f"  SGLang:   {sglang_files[0]}")
    print(f"  Megatron: {megatron_files[0]}")
    
    # Load tensors
    print("\nLoading tensors...")
    sglang_tensors = load_tensors(sglang_files[0])
    megatron_tensors = load_tensors(megatron_files[0])
    
    print(f"  SGLang:   {len(sglang_tensors)} tensors")
    print(f"  Megatron: {len(megatron_tensors)} tensors")
    
    # Compare
    print("\nComparing tensors...")
    results = compare_dumps(sglang_tensors, megatron_tensors, verbose=not args.quiet)
    
    # Return non-zero if there are significant differences
    has_significant_diff = any(
        not r.get("match", True) 
        for r in results.values() 
        if "max_diff" in r
    )
    sys.exit(1 if has_significant_diff else 0)


if __name__ == "__main__":
    main()

