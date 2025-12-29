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


def normalize_sglang_name(name: str) -> tuple[str, int | None, str]:
    """
    Convert SGLang tensor name to a normalized form that matches Megatron naming.
    
    SGLang naming: model.layers.0, model.layers.0.self_attn, model.layers.0.mlp, etc.
    Megatron naming: layer_0_output, layer_0_self_attention_output, etc.
    
    Returns: (normalized_name, layer_index, category)
    Category is one of: "layer_output", "sublayer", "embedding", "logits", "metadata"
    """
    # Handle non-layer tensors first
    if name == "lm_head":
        return "lm_head", None, "logits"
    if name == "model.embed_tokens":
        return "model.embed_tokens", None, "embedding"
    if name == "model.norm":
        return "model.norm", None, "final_norm"
    if name.startswith("model.forward_batch_info"):
        return name, None, "metadata"
    
    # Extract layer number if present
    parts = name.split(".")
    layer_idx = None
    
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            layer_idx = int(parts[i + 1])
            # Get remaining path after layer number
            remaining = ".".join(parts[i + 2:]) if i + 2 < len(parts) else ""
            
            # Layer output (no remaining path) - THIS IS THE KEY ONE
            if remaining == "" or remaining == "model":
                return f"layer_{layer_idx}_output", layer_idx, "layer_output"
            
            # Sublayers - less important for matching
            elif remaining == "self_attn" or remaining.startswith("self_attn."):
                if remaining == "self_attn":
                    return f"layer_{layer_idx}_self_attention_output", layer_idx, "sublayer"
                else:
                    sublayer = remaining.replace("self_attn.", "")
                    return f"layer_{layer_idx}_self_attention_{sublayer}", layer_idx, "sublayer"
            elif remaining == "mlp" or remaining.startswith("mlp."):
                if remaining == "mlp":
                    return f"layer_{layer_idx}_mlp_output", layer_idx, "sublayer"
                elif "gate_up_proj" in remaining:
                    return f"layer_{layer_idx}_mlp.gate_up_proj_output", layer_idx, "sublayer"
                elif "down_proj" in remaining:
                    return f"layer_{layer_idx}_mlp.down_proj_output", layer_idx, "sublayer"
                elif "act_fn" in remaining:
                    return f"layer_{layer_idx}_mlp_act_fn", layer_idx, "sublayer"
                else:
                    return f"layer_{layer_idx}_mlp_{remaining.replace('mlp.', '')}", layer_idx, "sublayer"
            elif "input_layernorm" in remaining:
                return f"layer_{layer_idx}_input_layernorm_output", layer_idx, "sublayer"
            elif "post_attention_layernorm" in remaining:
                return f"layer_{layer_idx}_post_attention_layernorm_output", layer_idx, "sublayer"
            else:
                return f"layer_{layer_idx}_{remaining.replace('.', '_')}", layer_idx, "sublayer"
            break
    
    # Not a layer tensor
    return name, None, "other"


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
        norm_name, layer_idx, category = normalize_sglang_name(name)
        sglang_normalized[norm_name] = (name, layer_idx, category)
    
    def format_tensor_info(t):
        """Format tensor info, handling lists and other types."""
        if isinstance(t, torch.Tensor):
            return f"shape={t.shape}, dtype={t.dtype}"
        elif isinstance(t, list):
            if len(t) > 0 and isinstance(t[0], torch.Tensor):
                return f"list[{len(t)}], first shape={t[0].shape}"
            return f"list[{len(t)}]"
        else:
            return f"type={type(t).__name__}"
    
    if verbose:
        print("\n" + "=" * 80)
        print("SGLang Tensors:")
        for name in sorted(sglang_tensors.keys())[:20]:
            t = sglang_tensors[name]
            print(f"  {name}: {format_tensor_info(t)}")
        if len(sglang_tensors) > 20:
            print(f"  ... and {len(sglang_tensors) - 20} more")
        
        print("\nMegatron Tensors:")
        for name in sorted(megatron_tensors.keys())[:20]:
            t = megatron_tensors[name]
            print(f"  {name}: {format_tensor_info(t)}")
        if len(megatron_tensors) > 20:
            print(f"  ... and {len(megatron_tensors) - 20} more")
        
        # Group by category for clearer output
        print("\n" + "-" * 40)
        print("KEY TENSORS (layer outputs, embeddings, logits):")
        print("-" * 40)
        for norm_name, (orig_name, layer_idx, category) in sorted(sglang_normalized.items()):
            if category in ("layer_output", "layer_output_proxy", "embedding", "logits", "final_norm"):
                in_megatron = "✓" if norm_name in megatron_tensors else "✗"
                proxy_note = " (proxy: mlp.down_proj)" if category == "layer_output_proxy" else ""
                print(f"  {in_megatron} {norm_name} <- {orig_name}{proxy_note}")
        
        print("\n" + "-" * 40)
        print("SUBLAYERS (less important for matching):")
        print("-" * 40)
        sublayer_count = 0
        for norm_name, (orig_name, layer_idx, category) in sorted(sglang_normalized.items()):
            if category == "sublayer":
                in_megatron = "✓" if norm_name in megatron_tensors else "✗"
                if sublayer_count < 10:
                    print(f"  {in_megatron} {norm_name}")
                sublayer_count += 1
        if sublayer_count > 10:
            print(f"  ... and {sublayer_count - 10} more sublayers")
        
        # Also show Megatron tensors that don't have SGLang counterparts
        print("\n" + "-" * 40)
        print("Megatron tensors (for reference):")
        print("-" * 40)
        for name in sorted(megatron_tensors.keys()):
            print(f"  {name}")
        print("=" * 80)
    
    # Compare tensors
    matched = 0
    unmatched = 0
    significant_diff_layers = []
    
    def get_tensor(t):
        """Extract tensor from possibly nested structure."""
        if isinstance(t, torch.Tensor):
            return t
        elif isinstance(t, (list, tuple)) and len(t) > 0:
            # Return first tensor in list
            return get_tensor(t[0])
        return None
    
    # Build a mapping of Megatron layer outputs to comparable SGLang tensors
    # SGLang doesn't dump layer outputs directly, so we use mlp.down_proj as proxy
    layer_output_mapping = {}
    for norm_name, (sglang_name, layer_idx, category) in sglang_normalized.items():
        if layer_idx is not None:
            # Use mlp.down_proj as proxy for layer output (it's the last sublayer before residual)
            if "mlp.down_proj" in sglang_name or "mlp_down_proj" in norm_name:
                megatron_layer_output = f"layer_{layer_idx}_output"
                if megatron_layer_output in megatron_tensors:
                    layer_output_mapping[megatron_layer_output] = (sglang_name, layer_idx, "layer_proxy")
    
    # First, compare KEY tensors (layer outputs, embeddings, logits)
    print("\n" + "=" * 60)
    print("COMPARING KEY TENSORS")
    print("=" * 60)
    
    # Add layer output proxies to the comparison
    for megatron_name, (sglang_name, layer_idx, category) in layer_output_mapping.items():
        if megatron_name not in [n for n, _ in sglang_normalized.items()]:
            # Add proxy mapping
            sglang_normalized[megatron_name] = (sglang_name, layer_idx, "layer_output_proxy")
    
    for norm_name, (sglang_name, layer_idx, category) in sorted(sglang_normalized.items()):
        # Skip sublayers and metadata in the main comparison
        if category not in ("layer_output", "layer_output_proxy", "embedding", "logits", "final_norm"):
            continue
        if norm_name not in megatron_tensors:
            if verbose:
                print(f"  {norm_name}: NOT FOUND in Megatron")
            unmatched += 1
            continue
        
        matched += 1
        sglang_t = get_tensor(sglang_tensors[sglang_name])
        megatron_t = get_tensor(megatron_tensors[norm_name])
        
        # Skip if either is not a tensor
        if sglang_t is None or megatron_t is None:
            if verbose:
                print(f"  {norm_name}: SKIPPED (not a tensor)")
            continue
        
        # Check shape - now both should dump first token only
        # SGLang: [1, hidden] (first token)
        # Megatron: [1, hidden] (first token, after our fix)
        
        orig_sglang_shape = sglang_t.shape
        orig_megatron_shape = megatron_t.shape
        aligned = False
        
        # Try to align shapes if they still don't match
        if sglang_t.shape != megatron_t.shape:
            # Case 1: Megatron has extra batch dimension [batch, hidden] vs [1, hidden]
            if len(megatron_t.shape) == 2 and len(sglang_t.shape) == 2:
                if megatron_t.shape[-1] == sglang_t.shape[-1]:
                    # Same hidden dim, just take first row from each
                    megatron_t = megatron_t[0:1, :]
                    sglang_t = sglang_t[0:1, :]
                    aligned = True
            
            # Case 2: One is 1D [hidden], other is 2D [1, hidden]
            if len(megatron_t.shape) == 1 and len(sglang_t.shape) == 2:
                megatron_t = megatron_t.unsqueeze(0)
                aligned = True
            elif len(sglang_t.shape) == 1 and len(megatron_t.shape) == 2:
                sglang_t = sglang_t.unsqueeze(0)
                aligned = True
            
            # Case 3: Same dim count but different shapes - take min
            if sglang_t.shape[-1] == megatron_t.shape[-1]:
                min_seq = min(sglang_t.shape[0], megatron_t.shape[0])
                sglang_t = sglang_t[:min_seq]
                megatron_t = megatron_t[:min_seq]
                aligned = True
        
        # Final shape check after alignment
        if sglang_t.shape != megatron_t.shape:
            results[norm_name] = {
                "match": False,
                "layer_idx": layer_idx,
                "reason": f"Shape mismatch: SGLang {orig_sglang_shape} vs Megatron {orig_megatron_shape}",
            }
            if verbose:
                print(f"  {norm_name}: SHAPE MISMATCH - SGLang {orig_sglang_shape} vs Megatron {orig_megatron_shape}")
            continue
        
        alignment_note = ""
        if aligned:
            alignment_note = f" [aligned: {orig_sglang_shape} vs {orig_megatron_shape} -> {sglang_t.shape}]"
        if category == "layer_output_proxy":
            alignment_note += " [proxy: mlp.down_proj ≠ layer_output, expect diff!]"
        
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
                f"rel_diff={stats['rel_diff']:.6e}{alignment_note}{end_color}"
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
            
            print("\nAll layers with significant differences:")
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
    
    print("\nFound files:")
    print(f"  SGLang:   {sglang_files[0]}")
    print(f"  Megatron: {megatron_files[0]}")
    
    # Load tensors
    print("\nLoading tensors...")
    sglang_tensors = load_tensors(sglang_files[0])
    megatron_tensors = load_tensors(megatron_files[0])
    
    print(f"  SGLang:   {len(sglang_tensors)} tensors")
    print(f"  Megatron: {len(megatron_tensors)} tensors")
    
    # Show input token IDs if available - THIS IS KEY FOR DEBUGGING
    print("\n" + "=" * 60)
    print("INPUT TOKEN COMPARISON (most important!)")
    print("=" * 60)
    
    sglang_token = None
    megatron_token = None
    
    if "model.forward_batch_info.input_ids" in sglang_tensors:
        input_ids = sglang_tensors["model.forward_batch_info.input_ids"]
        sglang_token = input_ids.flatten()[0].item() if input_ids.numel() > 0 else None
        print(f"  SGLang input_ids: {input_ids.tolist()}")
    if "model.forward_batch_info.positions" in sglang_tensors:
        positions = sglang_tensors["model.forward_batch_info.positions"]
        print(f"  SGLang positions: {positions.tolist()}")
    
    if "megatron_compared_token_id" in megatron_tensors:
        compared_token = megatron_tensors["megatron_compared_token_id"]
        megatron_token = compared_token.item() if compared_token.numel() > 0 else None
        print(f"  Megatron compared token: {compared_token.tolist()}")
    if "megatron_compared_position" in megatron_tensors:
        pos = megatron_tensors["megatron_compared_position"]
        print(f"  Megatron compared position: {pos.tolist()}")
    if "megatron_input_ids" in megatron_tensors:
        input_ids = megatron_tensors["megatron_input_ids"]
        flat = input_ids.flatten()[:10]
        print(f"  Megatron input_ids (first 10): {flat.tolist()}")
    
    # Check if tokens match
    if sglang_token is not None and megatron_token is not None:
        if sglang_token == megatron_token:
            print(f"\n  ✓ TOKENS MATCH! Both processing token {sglang_token}")
        else:
            print(f"\n  ✗ TOKENS DIFFER!")
            print(f"    SGLang is processing token: {sglang_token}")
            print(f"    Megatron is processing token: {megatron_token}")
            print(f"    This explains why hidden states are different!")
    print("=" * 60)
    
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

