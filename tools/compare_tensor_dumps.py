#!/usr/bin/env python3
"""
Compare tensor dumps from SGLang and Megatron/FSDP for debugging numerical differences.

Usage:
    # Compare SGLang vs Megatron (auto-match prefill or decode)
    python /root/slime/tools/compare_tensor_dumps.py \
        --sglang-dir /tmp/sglang_tensor_dump \
        --megatron-dir /tmp/megatron_tensor_dump \
        --auto-match
    
    # Compare ONLY decode passes (recommended for true on-policy verification)
    python /root/slime/tools/compare_tensor_dumps.py \
        --sglang-dir /tmp/sglang_tensor_dump \
        --megatron-dir /tmp/fsdp_tensor_dump \
        --auto-match --decode-only --response-start 91
    
    # List all available passes to understand the dump structure
    python /root/slime/tools/compare_tensor_dumps.py \
        --sglang-dir /tmp/sglang_tensor_dump \
        --megatron-dir /tmp/fsdp_tensor_dump \
        --list-passes

The script will:
1. Load tensor dumps from both directories
2. Map SGLang tensor names to Megatron/FSDP tensor names
3. Compute and display per-tensor statistics (max diff, mean diff, relative diff)
4. Identify the first layer where differences become significant
5. Compute and compare logprobs from logits using SGLang's formula

Key arguments:
    --decode-only: Only match decode passes (seq_len=1), skip prefill passes.
                   Use this for true on-policy verification!
    --response-start N: Only consider passes at position >= N.
                        Set to prompt_len to focus on response tokens.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import torch


def compute_logprobs_from_logits(
    logits: torch.Tensor, 
    temperature: float = 1.0,
    target_token_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Compute log probabilities from logits using SGLang's true on-policy formula.
    
    SGLang formula (when rl_on_policy_target is enabled):
        logits_bf16 = logits.bfloat16()
        logits_div_temp = logits_bf16.div(temperature).bfloat16()
        logprobs = torch.log_softmax(logits_div_temp, dim=-1)
    
    Args:
        logits: Raw logits tensor, shape [seq_len, vocab_size] or [batch, seq_len, vocab_size]
        temperature: Temperature for softmax (default 1.0)
        target_token_id: If provided, also return the logprob for this specific token
        
    Returns:
        (full_logprobs, target_logprob) where target_logprob is the logprob for target_token_id
    """
    # Apply SGLang's exact formula
    logits_bf16 = logits.bfloat16()
    logits_div_temp = logits_bf16.div(temperature).bfloat16()
    logprobs = torch.log_softmax(logits_div_temp, dim=-1)
    
    target_logprob = None
    if target_token_id is not None:
        # Get the logprob for the specific target token
        if logprobs.dim() == 1:
            target_logprob = logprobs[target_token_id]
        elif logprobs.dim() == 2:
            # [seq_len, vocab_size] -> get first position's target token logprob
            target_logprob = logprobs[0, target_token_id]
        elif logprobs.dim() == 3:
            # [batch, seq_len, vocab_size] -> get first batch, first position
            target_logprob = logprobs[0, 0, target_token_id]
    
    return logprobs, target_logprob


def find_dump_files(dump_dir: str, pass_id: int) -> list[Path]:
    """Find all dump files for a given pass ID in subdirectories."""
    dump_path = Path(dump_dir)
    if not dump_path.exists():
        return []
    
    files = list(dump_path.glob(f"*/Pass{pass_id:05d}.pt"))
    return sorted(files)


def list_all_passes(dump_dir: str) -> list[tuple[int, Path]]:
    """List all available pass files with their IDs."""
    dump_path = Path(dump_dir)
    if not dump_path.exists():
        return []
    
    passes = []
    for f in dump_path.glob("*/Pass*.pt"):
        # Extract pass ID from filename like "Pass00000.pt"
        name = f.stem
        if name.startswith("Pass"):
            try:
                pass_id = int(name[4:])
                passes.append((pass_id, f))
            except ValueError:
                continue
    return sorted(passes, key=lambda x: x[0])


def get_token_from_dump(dump_file: Path) -> tuple[int | None, int | None]:
    """Extract (token_id, position) from a dump file."""
    tensors = torch.load(dump_file, map_location="cpu")
    
    token_id = None
    position = None
    
    # SGLang format
    if "model.forward_batch_info.input_ids" in tensors:
        ids = tensors["model.forward_batch_info.input_ids"]
        if ids.numel() > 0:
            token_id = ids.flatten()[0].item()
    if "model.forward_batch_info.positions" in tensors:
        pos = tensors["model.forward_batch_info.positions"]
        if pos.numel() > 0:
            position = pos.flatten()[0].item()
    
    # Megatron format
    if "megatron_compared_token_id" in tensors:
        ids = tensors["megatron_compared_token_id"]
        if ids.numel() > 0:
            token_id = ids.item()
    if "megatron_compared_position" in tensors:
        pos = tensors["megatron_compared_position"]
        if pos.numel() > 0:
            position = pos.item()
    
    # FSDP format
    if "fsdp_compared_token_id" in tensors:
        ids = tensors["fsdp_compared_token_id"]
        if ids.numel() > 0:
            token_id = ids.item()
    if "fsdp_compared_position" in tensors:
        pos = tensors["fsdp_compared_position"]
        if pos.numel() > 0:
            position = pos.item()
    
    return token_id, position


def find_matching_sglang_pass(
    sglang_dir: str, 
    target_token: int | None = None, 
    target_position: int | None = None,
    decode_only: bool = False,
    min_position: int | None = None,
) -> int | None:
    """Find a SGLang pass that has the matching token (and optionally position).
    
    Args:
        sglang_dir: Directory containing SGLang dumps
        target_token: Token ID to match (optional)
        target_position: Position to match (optional)
        decode_only: If True, only consider decode passes (seq_len=1)
        min_position: If set, only consider passes with position >= min_position
                     (useful for finding response tokens)
    
    Prefers decode passes (single token) over prefill passes (multiple tokens).
    """
    passes = list_all_passes(sglang_dir)
    
    print(f"\n[Auto-matching] Looking for SGLang pass:")
    print(f"  target_token={target_token}, target_position={target_position}")
    print(f"  decode_only={decode_only}, min_position={min_position}")
    print(f"  Found {len(passes)} SGLang passes:")
    
    matching_decode_passes = []  # Prefer decode passes (seq_len=1)
    matching_prefill_passes = []  # Fallback to prefill passes
    first_decode_pass = None
    
    for pass_id, path in passes[:30]:  # Show first 30
        tensors = torch.load(path, map_location="cpu")
        token_id, position = get_token_from_dump(path)
        
        # Check if this is a decode pass (single token) or prefill (multiple tokens)
        seq_lens = None
        if "model.forward_batch_info.seq_lens" in tensors:
            seq_lens = tensors["model.forward_batch_info.seq_lens"]
            if seq_lens.numel() > 0:
                seq_lens = seq_lens.item()
        
        is_decode = seq_lens == 1 if seq_lens is not None else False
        
        # Track first decode pass at a response position
        if is_decode and first_decode_pass is None:
            if min_position is None or (position is not None and position >= min_position):
                first_decode_pass = pass_id
        
        marker = ""
        # Check matching criteria
        token_matches = (target_token is None or token_id == target_token)
        position_matches = (target_position is None or position == target_position)
        min_pos_ok = (min_position is None or 
                      (position is not None and position >= min_position))
        
        if token_matches and position_matches and min_pos_ok:
            if is_decode:
                matching_decode_passes.append((pass_id, position))
                marker = " <-- MATCH (decode)!"
            elif not decode_only:
                matching_prefill_passes.append((pass_id, position))
                marker = " <-- MATCH (prefill)!"
        
        pass_type = "decode" if is_decode else f"prefill(seq_len={seq_lens})"
        print(f"    Pass {pass_id:5d}: token={token_id}, position={position}, "
              f"type={pass_type}{marker}")
    
    if len(passes) > 30:
        print(f"    ... and {len(passes) - 30} more passes")
    
    # Prefer decode passes
    if matching_decode_passes:
        best = matching_decode_passes[0]
        print(f"\n  ✓ Found {len(matching_decode_passes)} matching decode pass(es)")
        print(f"    Using Pass {best[0]} at position {best[1]}")
        return best[0]
    elif matching_prefill_passes and not decode_only:
        best = matching_prefill_passes[0]
        print(f"\n  ⚠ Found {len(matching_prefill_passes)} matching prefill pass(es)")
        print(f"    Using Pass {best[0]} at position {best[1]}")
        print(f"    Note: Prefill passes process multiple tokens. Consider using --decode-only")
        return best[0]
    elif first_decode_pass is not None and decode_only:
        print(f"\n  ℹ No exact match found, using first decode pass: {first_decode_pass}")
        return first_decode_pass
    else:
        print(f"\n  ✗ No matching pass found!")
        return None


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
    parser.add_argument("--sglang-pass-id", type=int, default=None, 
                        help="Override SGLang pass ID (default: same as --pass-id)")
    parser.add_argument("--auto-match", action="store_true", 
                        help="Auto-find SGLang pass that matches Megatron's token")
    parser.add_argument("--decode-only", action="store_true",
                        help="Only match decode passes (seq_len=1), skip prefill passes")
    parser.add_argument("--response-start", type=int, default=None,
                        help="Response start position (prompt_len). Only match passes at >= this position")
    parser.add_argument("--list-passes", action="store_true", 
                        help="List all available passes and exit")
    parser.add_argument("--quiet", action="store_true", 
                        help="Suppress verbose output")
    args = parser.parse_args()
    
    # List passes mode
    if args.list_passes:
        print("SGLang passes:")
        sglang_passes = list_all_passes(args.sglang_dir)
        for pass_id, path in sglang_passes[:30]:
            token_id, position = get_token_from_dump(path)
            print(f"  Pass {pass_id:5d}: token={token_id}, position={position}")
        if len(sglang_passes) > 30:
            print(f"  ... and {len(sglang_passes) - 30} more")
        
        print("\nMegatron passes:")
        megatron_passes = list_all_passes(args.megatron_dir)
        for pass_id, path in megatron_passes[:30]:
            token_id, position = get_token_from_dump(path)
            print(f"  Pass {pass_id:5d}: token={token_id}, position={position}")
        if len(megatron_passes) > 30:
            print(f"  ... and {len(megatron_passes) - 30} more")
        sys.exit(0)
    
    # Determine which pass IDs to use
    megatron_pass_id = args.pass_id
    sglang_pass_id = args.sglang_pass_id if args.sglang_pass_id is not None else args.pass_id
    
    # Load Megatron first to get the target token for auto-matching
    megatron_files = find_dump_files(args.megatron_dir, megatron_pass_id)
    if not megatron_files:
        print(f"\nError: No Megatron dump files found for pass {megatron_pass_id}")
        print(f"  Looking in: {args.megatron_dir}/*/Pass{megatron_pass_id:05d}.pt")
        sys.exit(1)
    
    # Auto-match mode: find SGLang pass with matching token
    if args.auto_match:
        megatron_token, megatron_pos = get_token_from_dump(megatron_files[0])
        
        # Determine response start from dump or argument
        response_start = args.response_start
        if response_start is None:
            # Try to get from Megatron/FSDP dump
            mt = load_tensors(megatron_files[0])
            if "debug_prompt_len" in mt:
                response_start = int(mt["debug_prompt_len"].item())
                print(f"[Auto-match] Using response_start={response_start} from dump")
        
        if args.decode_only:
            print(f"[Auto-match] decode_only=True: Only matching decode passes")
        if response_start is not None:
            print(f"[Auto-match] response_start={response_start}: "
                  f"Only matching passes at position >= {response_start}")
        
        # For decode-only mode, we match by position (response position)
        # rather than by token, since we want the first response token
        if args.decode_only and response_start is not None:
            matched_pass = find_matching_sglang_pass(
                args.sglang_dir, 
                target_token=None,  # Don't filter by token
                target_position=None,
                decode_only=True,
                min_position=response_start,
            )
        elif megatron_token is not None:
            matched_pass = find_matching_sglang_pass(
                args.sglang_dir, 
                target_token=megatron_token, 
                target_position=megatron_pos,
                decode_only=args.decode_only,
                min_position=response_start,
            )
        else:
            matched_pass = None
            
        if matched_pass is not None:
            sglang_pass_id = matched_pass
            print(f"\n[Auto-match] Using SGLang pass {sglang_pass_id}")
        else:
            print(f"\n[Auto-match] Could not find matching pass, using pass {sglang_pass_id}")
    
    print(f"\nComparing tensor dumps:")
    print(f"  SGLang:   pass {sglang_pass_id} from {args.sglang_dir}")
    print(f"  Megatron: pass {megatron_pass_id} from {args.megatron_dir}")
    
    # Find dump files
    sglang_files = find_dump_files(args.sglang_dir, sglang_pass_id)
    
    if not sglang_files:
        print(f"\nError: No SGLang dump files found for pass {sglang_pass_id}")
        print(f"  Looking in: {args.sglang_dir}/*/Pass{sglang_pass_id:05d}.pt")
        sys.exit(1)
    
    if not megatron_files:
        print(f"\nError: No Megatron dump files found for pass {megatron_pass_id}")
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
    
    # Detect backend type (Megatron or FSDP)
    backend_name = "Megatron"
    token_key_prefix = "megatron"
    if "fsdp_compared_token_id" in megatron_tensors:
        backend_name = "FSDP"
        token_key_prefix = "fsdp"
    
    compared_token_key = f"{token_key_prefix}_compared_token_id"
    compared_pos_key = f"{token_key_prefix}_compared_position"
    input_ids_key = f"{token_key_prefix}_input_ids"
    
    if compared_token_key in megatron_tensors:
        compared_token = megatron_tensors[compared_token_key]
        megatron_token = compared_token.item() if compared_token.numel() > 0 else None
        print(f"  {backend_name} compared token: {compared_token.tolist()}")
    if compared_pos_key in megatron_tensors:
        pos = megatron_tensors[compared_pos_key]
        print(f"  {backend_name} compared position: {pos.tolist()}")
    if input_ids_key in megatron_tensors:
        input_ids = megatron_tensors[input_ids_key]
        flat = input_ids.flatten()[:10]
        print(f"  {backend_name} input_ids (first 10): {flat.tolist()}")
    
    # Show sequence length debug info
    if "debug_prompt_len" in megatron_tensors:
        print(f"\n  Debug info from {backend_name}:")
        print(f"    prompt_len: {megatron_tensors['debug_prompt_len'].item()}")
        print(f"    total_len: {megatron_tensors['debug_total_len'].item()}")
        print(f"    response_len: {megatron_tensors['debug_response_len'].item()}")
    
    # Check if tokens match
    if sglang_token is not None and megatron_token is not None:
        if sglang_token == megatron_token:
            print(f"\n  ✓ TOKENS MATCH! Both processing token {sglang_token}")
        else:
            print(f"\n  ✗ TOKENS DIFFER!")
            print(f"    SGLang is processing token: {sglang_token}")
            print(f"    Megatron is processing token: {megatron_token}")
            print(f"    This explains why hidden states are different!")
    
    # Check for prefill vs decode mismatch
    sglang_seq_len = None
    if "model.forward_batch_info.input_ids" in sglang_tensors:
        sglang_seq_len = sglang_tensors["model.forward_batch_info.input_ids"].numel()
    
    if sglang_seq_len is not None and sglang_seq_len > 1:
        print(f"\n  ⚠️  WARNING: SGLang is in PREFILL mode ({sglang_seq_len} tokens)")
        print(f"     SGLang's logits_processor outputs logits for position {sglang_seq_len}")
        print(f"     (predicting the NEXT token after the {sglang_seq_len}-token input)")
        
        # Check if FSDP/Megatron compared position matches
        compared_pos = 0
        if compared_pos_key in megatron_tensors:
            compared_pos = megatron_tensors[compared_pos_key].item()
        
        expected_pos = sglang_seq_len - 1  # Last position in prefill
        if compared_pos != expected_pos:
            print(f"\n  ❌ POSITION MISMATCH DETECTED!")
            print(f"     {backend_name} is extracting at position {compared_pos}")
            print(f"     But SGLang prefill outputs logits for position {sglang_seq_len}")
            print(f"     To match, {backend_name} should use position {expected_pos} (last input position)")
            print(f"     Or compare logits for position {sglang_seq_len} (next token prediction)")
    
    print("=" * 60)
    
    # Compare logprobs computed from logits
    print("\n" + "=" * 60)
    print("LOGPROBS COMPARISON (computed from logits)")
    print("=" * 60)
    
    # Determine the correct position for logits comparison
    # For SGLang prefill: logits_processor outputs logits at the LAST position
    # We need to compare FSDP's logits at the same position
    sglang_logits_pos = None
    if sglang_seq_len is not None and sglang_seq_len > 1:
        # SGLang prefill: logits are for predicting position sglang_seq_len
        # (from hidden state at position sglang_seq_len - 1)
        sglang_logits_pos = sglang_seq_len - 1
        print(f"  SGLang prefill: logits from position {sglang_logits_pos} "
              f"(predicting token at position {sglang_seq_len})")
    
    # Get target token for logprob extraction (next token after the logits position)
    next_token_id = None
    logits_pos = sglang_logits_pos if sglang_logits_pos is not None else 0
    
    if input_ids_key in megatron_tensors:
        input_ids = megatron_tensors[input_ids_key].flatten()
        # For prefill comparison, use SGLang's last position
        if sglang_logits_pos is not None and sglang_logits_pos + 1 < len(input_ids):
            next_token_id = input_ids[sglang_logits_pos + 1].item()
            print(f"  Target (next) token ID: {next_token_id} "
                  f"(at position {sglang_logits_pos + 1})")
        elif compared_pos_key in megatron_tensors:
            compared_pos = megatron_tensors[compared_pos_key].item()
            if compared_pos + 1 < len(input_ids):
                next_token_id = input_ids[compared_pos + 1].item()
                print(f"  Target (next) token ID: {next_token_id} "
                      f"(at position {compared_pos + 1})")
    
    # Find logits tensors
    sglang_logits = None
    megatron_logits = None
    
    # SGLang: lm_head contains the hidden state before logits, or logits_processor contains logits
    if "logits_processor" in sglang_tensors:
        sglang_logits = sglang_tensors["logits_processor"]
        print(f"  SGLang logits (from logits_processor): shape={sglang_logits.shape}, dtype={sglang_logits.dtype}")
    elif "lm_head" in sglang_tensors:
        # lm_head in SGLang tensor dump is actually the hidden state, not logits
        print(f"  SGLang lm_head is hidden state, not logits - skipping direct comparison")
    
    # Megatron/FSDP: logits or lm_head
    # For prefill comparison, prefer logits_at_prompt_end or logits_last
    megatron_logits_prefill = None
    
    if sglang_seq_len is not None and sglang_seq_len > 1:
        # SGLang is in prefill mode - try to get logits at matching position
        if "logits_at_prompt_end" in megatron_tensors:
            megatron_logits_prefill = megatron_tensors["logits_at_prompt_end"]
            pos = megatron_tensors.get("logits_prompt_end_pos", torch.tensor([0])).item()
            print(f"  {backend_name} logits_at_prompt_end: shape={megatron_logits_prefill.shape}, "
                  f"dtype={megatron_logits_prefill.dtype}, pos={pos}")
        elif "logits_last" in megatron_tensors:
            megatron_logits_prefill = megatron_tensors["logits_last"]
            pos = megatron_tensors.get("logits_last_pos", torch.tensor([0])).item()
            print(f"  {backend_name} logits_last: shape={megatron_logits_prefill.shape}, "
                  f"dtype={megatron_logits_prefill.dtype}, pos={pos}")
    
    if "logits" in megatron_tensors:
        megatron_logits = megatron_tensors["logits"]
        print(f"  {backend_name} logits: shape={megatron_logits.shape}, "
              f"dtype={megatron_logits.dtype}")
    elif "lm_head" in megatron_tensors:
        megatron_logits = megatron_tensors["lm_head"]
        print(f"  {backend_name} lm_head (logits): shape={megatron_logits.shape}, "
              f"dtype={megatron_logits.dtype}")
    
    # Use prefill-matched logits if available for SGLang prefill comparison
    if megatron_logits_prefill is not None:
        print(f"\n  ℹ️  Using {backend_name} logits_at_prompt_end for prefill comparison")
    
    # Also check for directly dumped logprobs
    sglang_direct_logprobs = None
    megatron_direct_logprobs = None
    
    if "logprobs" in sglang_tensors:
        sglang_direct_logprobs = sglang_tensors["logprobs"]
        print(f"  SGLang direct logprobs: shape={sglang_direct_logprobs.shape}, dtype={sglang_direct_logprobs.dtype}")
    
    if "logprobs" in megatron_tensors:
        megatron_direct_logprobs = megatron_tensors["logprobs"]
        print(f"  Megatron direct logprobs: shape={megatron_direct_logprobs.shape}, dtype={megatron_direct_logprobs.dtype}")
    
    # Compare logprobs
    # For decode comparison, find FSDP logits at the matching position
    megatron_logits_to_compare = megatron_logits
    logits_source = "default"
    
    # Get SGLang position for this decode pass
    sglang_position = None
    if "model.forward_batch_info.positions" in sglang_tensors:
        positions = sglang_tensors["model.forward_batch_info.positions"]
        if positions.numel() == 1:  # Decode pass (single token)
            sglang_position = positions.item()
    
    # For decode passes, try to find FSDP logits at the matching position
    if sglang_position is not None and sglang_seq_len == 1:  # Decode pass
        pos_key = f"logits_pos_{sglang_position}"
        if pos_key in megatron_tensors:
            megatron_logits_to_compare = megatron_tensors[pos_key]
            logits_source = f"decode-matched (position {sglang_position})"
            print(f"\n  ✓ Found {backend_name} logits at position {sglang_position} for decode comparison")
        else:
            # List available positions
            available_pos = [k for k in megatron_tensors.keys() if k.startswith("logits_pos_")]
            print(f"\n  ⚠️ {pos_key} not found in {backend_name} dump")
            print(f"     Available: {available_pos[:5]}...")
    elif megatron_logits_prefill is not None and sglang_seq_len is not None and sglang_seq_len > 1:
        # Prefill comparison
        megatron_logits_to_compare = megatron_logits_prefill
        logits_source = "prefill-matched (prompt_end)"
    
    if sglang_logits is not None and megatron_logits_to_compare is not None:
        print(f"\n  Computing logprobs from logits using SGLang formula...")
        print(f"    Formula: log_softmax(logits.bfloat16().div(temp).bfloat16(), dim=-1)")
        print(f"    {backend_name} logits source: {logits_source}")
        
        sglang_logprobs_full, sglang_target_lp = compute_logprobs_from_logits(
            sglang_logits, temperature=1.0, target_token_id=next_token_id
        )
        megatron_logprobs_full, megatron_target_lp = compute_logprobs_from_logits(
            megatron_logits_to_compare, temperature=1.0, target_token_id=next_token_id
        )
        
        # Compare full logprob distributions (first position)
        sg_lp = sglang_logprobs_full.flatten()[:10] if sglang_logprobs_full.dim() > 1 else sglang_logprobs_full[:10]
        mg_lp = megatron_logprobs_full.flatten()[:10] if megatron_logprobs_full.dim() > 1 else megatron_logprobs_full[:10]
        
        print(f"\n  Logprob samples (first 10 vocab entries):")
        print(f"    SGLang:   {sg_lp.tolist()}")
        print(f"    Megatron: {mg_lp.tolist()}")
        
        # Compare shapes
        print(f"\n  Full logprobs shapes:")
        print(f"    SGLang:   {sglang_logprobs_full.shape}")
        print(f"    Megatron: {megatron_logprobs_full.shape}")
        
        # Compare target token logprob
        if sglang_target_lp is not None and megatron_target_lp is not None:
            diff = abs(sglang_target_lp.float().item() - megatron_target_lp.float().item())
            print(f"\n  Logprob for target token {next_token_id}:")
            print(f"    SGLang:   {sglang_target_lp.item():.8f}")
            print(f"    Megatron: {megatron_target_lp.item():.8f}")
            print(f"    Diff:     {diff:.8e}")
            if diff < 1e-5:
                print(f"    ✓ Logprobs MATCH!")
            else:
                print(f"    ✗ Logprobs DIFFER!")
        
        # Overall distribution comparison
        if sglang_logprobs_full.shape == megatron_logprobs_full.shape:
            sg_flat = sglang_logprobs_full.float().flatten()
            mg_flat = megatron_logprobs_full.float().flatten()
            max_diff = (sg_flat - mg_flat).abs().max().item()
            mean_diff = (sg_flat - mg_flat).abs().mean().item()
            print(f"\n  Full logprob distribution comparison:")
            print(f"    Max diff:  {max_diff:.8e}")
            print(f"    Mean diff: {mean_diff:.8e}")
    else:
        print("\n  ⚠️  Could not find logits for comparison")
        if sglang_logits is None:
            print("    - SGLang logits not found")
        if megatron_logits is None:
            print("    - Megatron logits not found")
    
    # Compare directly dumped logprobs if available
    if sglang_direct_logprobs is not None and megatron_direct_logprobs is not None:
        print("\n  Comparing directly dumped logprobs:")
        sg_lp = sglang_direct_logprobs.float().flatten()
        mg_lp = megatron_direct_logprobs.float().flatten()
        if sg_lp.shape == mg_lp.shape:
            diff = (sg_lp - mg_lp).abs()
            print(f"    Max diff:  {diff.max().item():.8e}")
            print(f"    Mean diff: {diff.mean().item():.8e}")
            print(f"    SGLang:   {sg_lp.tolist()}")
            print(f"    Megatron: {mg_lp.tolist()}")
        else:
            print(f"    Shape mismatch: SGLang {sg_lp.shape} vs Megatron {mg_lp.shape}")
    
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

