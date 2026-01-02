#!/usr/bin/env python3
"""
Compare Layer 27 output and logits between SGLang and Megatron.

This is a simplified comparison script that focuses on:
1. Layer 27 (last transformer layer) output
2. Logits at the response token position

Input:
- One Megatron pass file (contains all tokens)
- One SGLang pass file (decode pass with single token)
- Response token ID

Usage:
    python compare_layer27_logits.py \\
        --megatron-pass /path/to/megatron/Pass00000.pt \\
        --sglang-pass /path/to/sglang/Pass00001.pt \\
        --response-token-id 12345
"""

import argparse
from pathlib import Path
from typing import Optional

import torch


def to_tensor(x, prefer_last=True):
    """Convert list/tuple to tensor, taking last element by default."""
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return None
        if prefer_last:
            x = x[-1]  # Last element is typically the output
        else:
            x = x[0]
    if not isinstance(x, torch.Tensor):
        return None
    return x


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
        "t1_sum": t1_f.sum().item(),
        "t2_sum": t2_f.sum().item(),
        "sum_diff": (t1_f.sum() - t2_f.sum()).abs().item(),
    }


def print_bitwise_comparison(stats: dict[str, float], indent: str = "  ") -> bool:
    """Print min/max/sum comparison for bitwise identical check."""
    print(f"{indent}Min/Max/Sum comparison:")
    print(f"{indent}  Min:  SGLang={stats['t1_min']:.8e}, Megatron={stats['t2_min']:.8e}, "
          f"diff={abs(stats['t1_min'] - stats['t2_min']):.8e}")
    print(f"{indent}  Max:  SGLang={stats['t1_max']:.8e}, Megatron={stats['t2_max']:.8e}, "
          f"diff={abs(stats['t1_max'] - stats['t2_max']):.8e}")
    print(f"{indent}  Sum:  SGLang={stats['t1_sum']:.8e}, Megatron={stats['t2_sum']:.8e}, "
          f"diff={stats['sum_diff']:.8e}")

    is_bitwise_identical = (
        stats['max_diff'] < 1e-8 and
        abs(stats['t1_min'] - stats['t2_min']) < 1e-8 and
        abs(stats['t1_max'] - stats['t2_max']) < 1e-8 and
        stats['sum_diff'] < 1e-8
    )
    if is_bitwise_identical:
        print(f"{indent}  ✓✓✓ BITWISE IDENTICAL ✓✓✓")
    else:
        print(f"{indent}  ✗ Not bitwise identical")
    
    return is_bitwise_identical


def extract_layer27_output(
    tensors: dict,
    source: str,
    response_position: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """
    Extract layer 27 output from tensor dump.
    
    Args:
        tensors: Dictionary of tensors from dump file
        source: "sglang" or "megatron"
        response_position: For Megatron, the position to extract (None for SGLang decode)
    
    Returns:
        Extracted tensor or None if not found
    """
    layer_idx = 27
    
    if source == "sglang":
        # SGLang decode pass: single token, shape should be [1, hidden_size] or [hidden_size]
        keys_to_try = [
            f"model.layers.{layer_idx}.mlp.down_proj",  # MLP output (before residual)
            f"model.layers.{layer_idx}.output",  # Full layer output (if available)
        ]
        
        for key in keys_to_try:
            if key in tensors:
                val = tensors[key]
                tensor = to_tensor(val, prefer_last=True)
                if tensor is not None:
                    # SGLang decode: extract single token
                    if tensor.dim() == 2:
                        tensor = tensor[0]  # [1, hidden] -> [hidden]
                    elif tensor.dim() == 3:
                        tensor = tensor[0, 0]  # [1, 1, hidden] -> [hidden]
                    elif tensor.dim() == 1:
                        pass  # Already [hidden]
                    return tensor
        
        # Try to find layer output by searching
        for key in tensors.keys():
            if f"layers.{layer_idx}" in key and ("output" in key or "down_proj" in key):
                val = tensors[key]
                tensor = to_tensor(val, prefer_last=True)
                if tensor is not None:
                    if tensor.dim() == 2:
                        tensor = tensor[0]
                    elif tensor.dim() == 3:
                        tensor = tensor[0, 0]
                    return tensor
    
    elif source == "megatron":
        # Megatron: use _at_response_start suffix (already extracted at response position)
        # Try MLP output first (most common), then full layer output
        keys_to_try = [
            f"layer_{layer_idx}_mlp_output_at_response_start",  # MLP output (before residual)
            f"layer_{layer_idx}_output_at_response_start",  # Full layer output (MLP + residual)
            f"layer_{layer_idx}_mlp_output",  # Fallback without suffix
            f"layer_{layer_idx}_output",  # Fallback without suffix
        ]
        
        for key in keys_to_try:
            if key in tensors:
                tensor = tensors[key]
                if tensor is not None:
                    # _at_response_start tensors are already at response position (single token)
                    # Regular tensors need position extraction
                    if "_at_response_start" in key:
                        # Already extracted at response position, just flatten
                        if tensor.dim() == 2:
                            tensor = tensor[0]  # [1, hidden] -> [hidden]
                        elif tensor.dim() == 3:
                            tensor = tensor[0, 0]  # [1, 1, hidden] -> [hidden]
                        elif tensor.dim() == 1:
                            pass  # Already [hidden]
                    else:
                        # Need to extract at response position
                        if response_position is not None:
                            if tensor.dim() == 2:
                                # [seq_len, hidden] -> extract at position
                                if response_position < tensor.shape[0]:
                                    tensor = tensor[response_position]
                                else:
                                    continue  # Try next key
                            elif tensor.dim() == 3:
                                # [batch, seq_len, hidden] or [seq_len, batch, hidden]
                                d0, d1, d2 = tensor.shape
                                if d0 == 1:
                                    # [1, seq_len, hidden]
                                    if response_position < d1:
                                        tensor = tensor[0, response_position]
                                    else:
                                        continue
                                else:
                                    # [seq_len, 1, hidden]
                                    if response_position < d0:
                                        tensor = tensor[response_position, 0]
                                    else:
                                        continue
                        else:
                            # No position specified, take first token
                            if tensor.dim() == 2:
                                tensor = tensor[0]
                            elif tensor.dim() == 3:
                                tensor = tensor[0, 0] if tensor.shape[0] == 1 else tensor[0, 0]
                    return tensor
    
    return None


def extract_logits(
    tensors: dict,
    source: str,
    response_position: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """
    Extract logits from tensor dump.
    
    Args:
        tensors: Dictionary of tensors from dump file
        source: "sglang" or "megatron"
        response_position: For Megatron, the position to extract (None for SGLang decode)
    
    Returns:
        Extracted logits tensor or None if not found
    """
    if source == "sglang":
        # SGLang decode pass: logits_processor contains logits for single token
        if "logits_processor" in tensors:
            logits = tensors["logits_processor"]
            if logits.dim() == 2:
                logits = logits[0]  # [1, vocab] -> [vocab]
            elif logits.dim() == 1:
                pass  # Already [vocab]
            return logits
    
    elif source == "megatron":
        # Megatron: extract logits at specific position
        # Note: response_position is the position BEFORE the response token
        # (i.e., prompt_len - 1), which predicts the first response token
        if response_position is not None:
            # Try position-specific key first
            logits_key = f"logits_pos_{response_position}"
            if logits_key in tensors:
                logits = tensors[logits_key]
                if logits.dim() == 2:
                    logits = logits[0]  # [1, vocab] -> [vocab]
                return logits
            
            # Try extracting from logits_full
            if "logits_full" in tensors:
                full_logits = tensors["logits_full"]
                if full_logits.dim() == 3:
                    d0, d1, d2 = full_logits.shape
                    if d0 == 1:
                        # [1, seq_len, vocab]
                        if response_position < d1:
                            logits = full_logits[0, response_position]
                            return logits
                    else:
                        # [seq_len, 1, vocab]
                        if response_position < d0:
                            logits = full_logits[response_position, 0]
                            return logits
                elif full_logits.dim() == 2:
                    # [seq_len, vocab]
                    if response_position < full_logits.shape[0]:
                        logits = full_logits[response_position]
                        return logits
        
        # Fallback: try other keys
        for key in ["logits_at_prompt_end", "logits"]:
            if key in tensors:
                logits = tensors[key]
                if logits.dim() == 2:
                    logits = logits[0]
                return logits
    
    return None


def find_response_position(
    megatron_tensors: dict,
    response_token_id: int,
) -> Optional[int]:
    """
    Find the position of the response token in Megatron dump.
    
    Megatron has both prefill (prompt) and response tokens.
    We need to find the first response token position.
    
    Args:
        megatron_tensors: Dictionary of tensors from Megatron dump
        response_token_id: The token ID to find
    
    Returns:
        Position index or None if not found
    """
    # First, try to get prompt_len to determine response start
    prompt_len = None
    for key in ["prompt_len", "debug_prompt_len"]:
        if key in megatron_tensors:
            prompt_len = int(megatron_tensors[key].item())
            break
    
    # Try to find input_ids
    input_ids_key = None
    for key in ["megatron_input_ids", "input_ids"]:
        if key in megatron_tensors:
            input_ids_key = key
            break
    
    if input_ids_key is None:
        # If no input_ids, use prompt_len - 1 as fallback (position before first response)
        if prompt_len is not None:
            return prompt_len - 1
        return None
    
    input_ids = megatron_tensors[input_ids_key]
    if input_ids.dim() > 1:
        input_ids = input_ids.flatten()
    
    # Find the position of response_token_id
    positions = (input_ids == response_token_id).nonzero(as_tuple=True)[0]
    
    if len(positions) > 0:
        found_pos = positions[0].item()
        
        # If we have prompt_len, verify this is in the response part
        if prompt_len is not None:
            if found_pos < prompt_len:
                # Token found in prefill part, use prompt_len as response start
                print(f"  Note: Token {response_token_id} found at position {found_pos} (in prefill)")
                print(f"  Using prompt_len={prompt_len} as response start position")
                return prompt_len - 1  # Position before first response token
            else:
                # Token found in response part
                return found_pos
        else:
            # No prompt_len, return first occurrence
            return found_pos
    
    # Token not found, use prompt_len - 1 as fallback
    if prompt_len is not None:
        print(f"  Note: Token {response_token_id} not found in input_ids")
        print(f"  Using prompt_len={prompt_len} - 1 = {prompt_len - 1} as response position")
        return prompt_len - 1
    
    return None


def compare_layer27_and_logits(
    megatron_pass_path: str,
    sglang_pass_path: str,
    response_token_id: int,
    verbose: bool = True,
) -> None:
    """
    Compare Layer 27 output and logits between SGLang and Megatron.
    
    Args:
        megatron_pass_path: Path to Megatron pass file
        sglang_pass_path: Path to SGLang pass file (decode pass)
        response_token_id: The response token ID to compare
        verbose: Whether to print detailed output
    """
    print("=" * 70)
    print("LAYER 27 OUTPUT & LOGITS COMPARISON")
    print("=" * 70)
    print(f"\nMegatron pass: {megatron_pass_path}")
    print(f"SGLang pass:   {sglang_pass_path}")
    print(f"Response token ID: {response_token_id}")
    print("=" * 70)
    
    # Load tensors
    print("\nLoading tensor dumps...")
    megatron_tensors = torch.load(megatron_pass_path, map_location="cpu")
    sglang_tensors = torch.load(sglang_pass_path, map_location="cpu")
    
    # Verify SGLang pass info
    print("\nVerifying SGLang pass...")
    sglang_info = {}
    if "model.forward_batch_info.input_ids" in sglang_tensors:
        sglang_input_ids = sglang_tensors["model.forward_batch_info.input_ids"]
        if sglang_input_ids.numel() > 0:
            sglang_info["first_token_id"] = sglang_input_ids.flatten()[0].item()
            sglang_info["seq_len"] = sglang_input_ids.numel()
    
    if "model.forward_batch_info.positions" in sglang_tensors:
        sglang_positions = sglang_tensors["model.forward_batch_info.positions"]
        if sglang_positions.numel() > 0:
            sglang_info["first_position"] = sglang_positions.flatten()[0].item()
            sglang_info["last_position"] = sglang_positions.flatten()[-1].item()
    
    sglang_info["is_decode"] = sglang_info.get("seq_len", 0) == 1
    sglang_info["is_prefill"] = sglang_info.get("seq_len", 0) > 1
    
    print(f"  SGLang pass type: {'DECODE' if sglang_info.get('is_decode') else 'PREFILL' if sglang_info.get('is_prefill') else 'UNKNOWN'}")
    if "first_position" in sglang_info:
        print(f"  First position: {sglang_info['first_position']}")
    if "first_token_id" in sglang_info:
        print(f"  First token ID: {sglang_info['first_token_id']}")
        if sglang_info['first_token_id'] != response_token_id:
            print(f"  ⚠️  WARNING: SGLang pass token ID ({sglang_info['first_token_id']}) != "
                  f"requested token ID ({response_token_id})")
            print(f"     This pass processes token {sglang_info['first_token_id']}, not {response_token_id}")
        else:
            print(f"  ✓ Token ID matches: {response_token_id}")
    
    # Find response position in Megatron dump
    # Note: response_position is the position that PREDICTS the response token
    # (i.e., prompt_len - 1 for first response token)
    response_position = find_response_position(megatron_tensors, response_token_id)
    if response_position is None:
        print(f"\n⚠️  WARNING: Could not determine response position")
        print("  Available input_ids keys:", [k for k in megatron_tensors.keys() if "input" in k.lower() or "id" in k.lower()])
        # Try to use prompt_len - 1 as fallback
        prompt_len = None
        for key in ["prompt_len", "debug_prompt_len"]:
            if key in megatron_tensors:
                prompt_len = int(megatron_tensors[key].item())
                break
        if prompt_len is not None:
            response_position = prompt_len - 1
            print(f"  Using fallback position: {response_position} (prompt_len - 1)")
        else:
            print("  ERROR: Cannot determine response position")
            return
    else:
        # Get prompt_len for context
        prompt_len = None
        for key in ["prompt_len", "debug_prompt_len"]:
            if key in megatron_tensors:
                prompt_len = int(megatron_tensors[key].item())
                break
        
        if prompt_len is not None:
            print(f"\n✓ Using position: {response_position} (predicts token at position {response_position + 1})")
            print(f"  prompt_len: {prompt_len}, response starts at position {prompt_len}")
        else:
            print(f"\n✓ Using position: {response_position}")
    
    # =========================================================================
    # 1. Compare Layer 27 Output
    # =========================================================================
    print("\n" + "-" * 70)
    print("1. LAYER 27 OUTPUT COMPARISON")
    print("-" * 70)
    
    sg_layer27 = extract_layer27_output(sglang_tensors, "sglang")
    meg_layer27 = extract_layer27_output(megatron_tensors, "megatron", response_position)
    
    if sg_layer27 is None:
        print("  ✗ SGLang layer 27 output: NOT FOUND")
        print(f"    Available SGLang keys with 'layer': {[k for k in sglang_tensors.keys() if 'layer' in k.lower()][:10]}")
    else:
        print(f"  ✓ SGLang layer 27 output: shape={sg_layer27.shape}, dtype={sg_layer27.dtype}")
    
    if meg_layer27 is None:
        print("  ✗ Megatron layer 27 output: NOT FOUND")
        layer27_keys = [k for k in megatron_tensors.keys() if 'layer_27' in k]
        print(f"    Available Megatron keys with 'layer_27': {layer27_keys[:15]}")
        print(f"    Tried keys:")
        print(f"      - layer_27_mlp_output_at_response_start")
        print(f"      - layer_27_output_at_response_start")
        print(f"      - layer_27_mlp_output")
        print(f"      - layer_27_output")
    else:
        print(f"  ✓ Megatron layer 27 output: shape={meg_layer27.shape}, dtype={meg_layer27.dtype}")
    
    if sg_layer27 is not None and meg_layer27 is not None:
        # Align shapes if needed
        if sg_layer27.shape != meg_layer27.shape:
            min_len = min(sg_layer27.numel(), meg_layer27.numel())
            sg_layer27 = sg_layer27.flatten()[:min_len]
            meg_layer27 = meg_layer27.flatten()[:min_len]
            print(f"  ⚠ Shape mismatch, comparing first {min_len} elements")
        
        stats = compute_diff_stats(sg_layer27, meg_layer27)
        
        print(f"\n  Comparison statistics:")
        print(f"    Max diff:  {stats['max_diff']:.8e}")
        print(f"    Mean diff: {stats['mean_diff']:.8e}")
        print(f"    Rel diff:  {stats['rel_diff']:.8e}")
        
        print_bitwise_comparison(stats, indent="  ")
        
        # Show first 10 values
        n_show = min(10, len(sg_layer27), len(meg_layer27))
        sg_vals = sg_layer27[:n_show].float().tolist()
        meg_vals = meg_layer27[:n_show].float().tolist()
        diff_vals = [(sg_layer27[i] - meg_layer27[i]).abs().float().item()
                     for i in range(n_show)]
        
        print(f"\n  First {n_show} values:")
        print(f"    SGLang:   {[f'{v:.6f}' for v in sg_vals]}")
        print(f"    Megatron: {[f'{v:.6f}' for v in meg_vals]}")
        print(f"    Diff:     {[f'{v:.6e}' for v in diff_vals]}")
        
        if stats['max_diff'] < 1e-5:
            print("\n  ✓✓✓ LAYER 27 OUTPUT MATCHES! ✓✓✓")
        elif stats['max_diff'] < 1e-3:
            print("\n  ⚠ Layer 27 output is close but not identical")
        else:
            print("\n  ✗ Layer 27 output DIFFERS")
    
    # =========================================================================
    # 2. Compare Logits
    # =========================================================================
    print("\n" + "-" * 70)
    print("2. LOGITS COMPARISON")
    print("-" * 70)
    
    sg_logits = extract_logits(sglang_tensors, "sglang")
    meg_logits = extract_logits(megatron_tensors, "megatron", response_position)
    
    if sg_logits is None:
        print("  ✗ SGLang logits: NOT FOUND")
        print(f"    Available SGLang keys: {list(sglang_tensors.keys())[:10]}")
    else:
        print(f"  ✓ SGLang logits: shape={sg_logits.shape}, dtype={sg_logits.dtype}")
    
    if meg_logits is None:
        print("  ✗ Megatron logits: NOT FOUND")
        print(f"    Available Megatron keys with 'logits': {[k for k in megatron_tensors.keys() if 'logit' in k.lower()][:10]}")
    else:
        print(f"  ✓ Megatron logits: shape={meg_logits.shape}, dtype={meg_logits.dtype}")
    
    if sg_logits is not None and meg_logits is not None:
        # Align shapes if needed
        if sg_logits.shape != meg_logits.shape:
            min_len = min(sg_logits.numel(), meg_logits.numel())
            sg_logits = sg_logits.flatten()[:min_len]
            meg_logits = meg_logits.flatten()[:min_len]
            print(f"  ⚠ Shape mismatch, comparing first {min_len} elements")
        
        stats = compute_diff_stats(sg_logits, meg_logits)
        
        print(f"\n  Comparison statistics:")
        print(f"    Max diff:  {stats['max_diff']:.8e}")
        print(f"    Mean diff: {stats['mean_diff']:.8e}")
        print(f"    Rel diff:  {stats['rel_diff']:.8e}")
        
        print_bitwise_comparison(stats, indent="  ")
        
        # Check logit for the response token
        if response_token_id < len(sg_logits) and response_token_id < len(meg_logits):
            sg_token_logit = sg_logits[response_token_id].float().item()
            meg_token_logit = meg_logits[response_token_id].float().item()
            diff_token = abs(sg_token_logit - meg_token_logit)
            print(f"\n  Logit for token {response_token_id}:")
            print(f"    SGLang:   {sg_token_logit:.8f}")
            print(f"    Megatron: {meg_token_logit:.8f}")
            print(f"    Diff:     {diff_token:.8e}")
            if diff_token < 1e-5:
                print("    ✓ MATCH")
            else:
                print("    ✗ DIFFER")
        
        # Full logits comparison
        diff_all = (sg_logits.float() - meg_logits.float()).abs()
        nonzero_diff = (diff_all > 1e-6).sum().item()
        total_vocab = len(diff_all)
        pct_diff = 100.0 * nonzero_diff / total_vocab
        
        print(f"\n  Full logits comparison (all {total_vocab} vocab tokens):")
        print(f"    Exact matches (diff < 1e-6): {total_vocab - nonzero_diff}")
        print(f"    With differences: {nonzero_diff} ({pct_diff:.2f}%)")
        print(f"    Max diff: {diff_all.max().item():.8e}")
        print(f"    Mean diff: {diff_all.mean().item():.8e}")
        print(f"    Sum diff: {diff_all.sum().item():.8f}")
        
        if nonzero_diff > 0:
            nonzero_diffs = diff_all[diff_all > 1e-6]
            print(f"    Non-zero diffs - min: {nonzero_diffs.min().item():.8e}, "
                  f"max: {nonzero_diffs.max().item():.8e}, "
                  f"mean: {nonzero_diffs.mean().item():.8e}")
            
            # Show top 10 differing indices
            top_k = min(10, total_vocab)
            top_diff_indices = diff_all.topk(top_k).indices.tolist()
            print(f"\n    Top {top_k} differing indices:")
            for rank, idx in enumerate(top_diff_indices, 1):
                sg_v = sg_logits[idx].float().item()
                meg_v = meg_logits[idx].float().item()
                d = diff_all[idx].item()
                print(f"      {rank:2d}. idx {idx:6d}: SGLang={sg_v:12.8f}, "
                      f"Meg={meg_v:12.8f}, diff={d:.8e}")
        
        # Show first 10 logits
        n_show = min(10, len(sg_logits), len(meg_logits))
        sg_vals = sg_logits[:n_show].float().tolist()
        meg_vals = meg_logits[:n_show].float().tolist()
        diff_vals = [(sg_logits[i] - meg_logits[i]).abs().float().item()
                     for i in range(n_show)]
        
        print(f"\n  First {n_show} logits values:")
        print(f"    SGLang:   {[f'{v:.8f}' for v in sg_vals]}")
        print(f"    Megatron: {[f'{v:.8f}' for v in meg_vals]}")
        print(f"    Diff:     {[f'{v:.8e}' for v in diff_vals]}")
        
        if stats['max_diff'] < 1e-5:
            print("\n  ✓✓✓ LOGITS MATCH! ✓✓✓")
        elif stats['max_diff'] < 1e-3:
            print("\n  ⚠ Logits are close but not identical")
        else:
            print("\n  ✗ Logits DIFFER")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    layer27_match = False
    logits_match = False
    
    if sg_layer27 is not None and meg_layer27 is not None:
        stats = compute_diff_stats(sg_layer27, meg_layer27)
        layer27_match = stats['max_diff'] < 1e-5
        print(f"Layer 27 Output: {'✓ MATCH' if layer27_match else '✗ DIFFER'} "
              f"(max_diff={stats['max_diff']:.8e})")
    
    if sg_logits is not None and meg_logits is not None:
        stats = compute_diff_stats(sg_logits, meg_logits)
        logits_match = stats['max_diff'] < 1e-5
        print(f"Logits:          {'✓ MATCH' if logits_match else '✗ DIFFER'} "
              f"(max_diff={stats['max_diff']:.8e})")
    
    if layer27_match and logits_match:
        print("\n✓✓✓ TRUE ON-POLICY VERIFIED! ✓✓✓")
    else:
        print("\n⚠ Some components differ - check details above")
    
    print("=" * 70)


def find_sglang_pass_for_token(
    sglang_dir: str,
    token_id: int,
) -> Optional[tuple[int, str]]:
    """
    Find SGLang decode pass that processes a specific token ID.
    
    Args:
        sglang_dir: Directory containing SGLang pass files
        token_id: Token ID to find
    
    Returns:
        (pass_id, pass_path) or None if not found
    """
    from pathlib import Path
    
    dump_path = Path(sglang_dir)
    if not dump_path.exists():
        return None
    
    # Find all pass files
    passes = []
    for f in dump_path.glob("*/Pass*.pt"):
        name = f.stem
        if name.startswith("Pass"):
            try:
                pass_id = int(name[4:])
                passes.append((pass_id, f))
            except ValueError:
                continue
    
    passes = sorted(passes, key=lambda x: x[0])
    
    # Check each decode pass
    for pass_id, path in passes:
        try:
            tensors = torch.load(path, map_location="cpu")
            if "model.forward_batch_info.input_ids" in tensors:
                input_ids = tensors["model.forward_batch_info.input_ids"]
                if input_ids.numel() == 1:  # Decode pass has single token
                    first_token_id = input_ids.flatten()[0].item()
                    if first_token_id == token_id:
                        return (pass_id, str(path))
        except Exception:
            continue
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Compare Layer 27 output and logits between SGLang and Megatron"
    )
    parser.add_argument(
        "--megatron-pass",
        type=str,
        required=True,
        help="Path to Megatron pass file (contains all tokens)"
    )
    parser.add_argument(
        "--sglang-pass",
        type=str,
        required=False,
        help="Path to SGLang pass file (decode pass with single token). "
             "If not provided, will search in sglang-dir"
    )
    parser.add_argument(
        "--sglang-dir",
        type=str,
        required=False,
        help="Directory containing SGLang pass files (used if --sglang-pass not provided)"
    )
    parser.add_argument(
        "--response-token-id",
        type=int,
        required=True,
        help="Response token ID to compare"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed output (default: True)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    megatron_path = Path(args.megatron_pass)
    if not megatron_path.exists():
        print(f"ERROR: Megatron pass file not found: {megatron_path}")
        return
    
    # Determine SGLang pass path
    sglang_path = None
    if args.sglang_pass:
        sglang_path = Path(args.sglang_pass)
        if not sglang_path.exists():
            print(f"ERROR: SGLang pass file not found: {sglang_path}")
            return
    elif args.sglang_dir:
        # Search for pass with matching token ID
        print(f"Searching for SGLang pass with token ID {args.response_token_id}...")
        result = find_sglang_pass_for_token(args.sglang_dir, args.response_token_id)
        if result:
            pass_id, pass_path = result
            print(f"✓ Found Pass {pass_id:05d} with token ID {args.response_token_id}")
            sglang_path = Path(pass_path)
        else:
            print(f"ERROR: Could not find SGLang pass with token ID {args.response_token_id}")
            print(f"  Searched in: {args.sglang_dir}")
            return
    else:
        print("ERROR: Must provide either --sglang-pass or --sglang-dir")
        return
    
    compare_layer27_and_logits(
        megatron_pass_path=str(megatron_path),
        sglang_pass_path=str(sglang_path),
        response_token_id=args.response_token_id,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

