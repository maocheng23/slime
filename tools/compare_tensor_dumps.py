#!/usr/bin/env python3
"""
Compare tensor dumps from SGLang and FSDP/Megatron.

Key Understanding:
==================
Training side (FSDP/Megatron): ONE forward pass processes entire sequence.
Inference side (SGLang): MULTIPLE passes - prefill + decode passes.

For comparing the FIRST response token:
- SGLang: Use PREFILL pass (seq_len = prompt_len)
- FSDP: Use logits at position (prompt_len - 1)

Both should produce identical:
1. Hidden states at each layer for position (prompt_len - 1)
2. Logits/logprobs for the first response token

Usage:
    python compare_tensor_dumps.py \\
        --sglang-dir /tmp/sglang_dump \\
        --fsdp-dir /tmp/fsdp_dump \\
        --compare-first-token
"""

import argparse
import re
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
    Compute log probabilities from logits using SGLang's formula.
    
    SGLang formula (when rl_on_policy_target is enabled):
        logits_bf16 = logits.bfloat16()
        logits_div_temp = logits_bf16.div(temperature).bfloat16()
        logprobs = torch.log_softmax(logits_div_temp, dim=-1)
    
    Args:
        logits: Raw logits tensor
        temperature: Temperature for softmax (default 1.0)
        target_token_id: If provided, return logprob for this token
        
    Returns:
        (full_logprobs, target_logprob)
    """
    logits_bf16 = logits.bfloat16()
    logits_div_temp = logits_bf16.div(temperature).bfloat16()
    logprobs = torch.log_softmax(logits_div_temp, dim=-1)
    
    target_logprob = None
    if target_token_id is not None:
        if logprobs.dim() == 1:
            target_logprob = logprobs[target_token_id]
        elif logprobs.dim() == 2:
            target_logprob = logprobs[0, target_token_id]
        elif logprobs.dim() == 3:
            target_logprob = logprobs[0, 0, target_token_id]
    
    return logprobs, target_logprob


def print_top_logprobs(
    logits: torch.Tensor,
    actual_token_id: int | None,
    label: str,
    top_k: int = 10,
    temperature: float = 1.0,
) -> None:
    """
    Print top-k logprobs and the logprob for the actual predicted token.

    Args:
        logits: Raw logits tensor (vocab_size,) or (1, vocab_size)
        actual_token_id: The token that was actually generated/in the sequence
        label: Label for printing (e.g., "SGLang pos 90" or "FSDP pos 90")
        top_k: Number of top tokens to show
        temperature: Temperature for softmax
    """
    logprobs, _ = compute_logprobs_from_logits(logits, temperature)
    logprobs_flat = logprobs.flatten().float()

    # Get top-k tokens by logprob
    k = min(top_k, len(logprobs_flat))
    top_values, top_indices = torch.topk(logprobs_flat, k)

    print(f"\n  {label}:")
    print(f"    Top {top_k} tokens by logprob:")
    for i, (val, idx) in enumerate(zip(top_values, top_indices)):
        is_actual = (actual_token_id is not None
                     and idx.item() == actual_token_id)
        marker = " <-- ACTUAL" if is_actual else ""
        tok_id = idx.item()
        lp = val.item()
        print(f"      {i+1:2d}. token={tok_id:6d} lp={lp:10.6f}{marker}")

    # Show actual token if not in top-k
    if actual_token_id is not None:
        if actual_token_id not in top_indices.tolist():
            actual_lp = logprobs_flat[actual_token_id].item()
            rank = (logprobs_flat > actual_lp).sum().item() + 1
            print(f"    Actual token {actual_token_id}: "
                  f"logprob={actual_lp:.6f} (rank {rank})")
        else:
            actual_lp = logprobs_flat[actual_token_id].item()
            print(f"    Actual token {actual_token_id}: "
                  f"logprob={actual_lp:.6f}")


def find_sglang_decode_pass(
    sglang_dir: str, decode_position: int
) -> tuple[int, Path] | None:
    """
    Find the SGLang decode pass for a specific position.

    Each decode pass processes ONE token and outputs logits for the next.
    The decode pass for position X has:
    - seq_len = 1
    - first_position = X (the position being processed)
    """
    passes = list_all_passes(sglang_dir)

    for pass_id, path in passes:
        info = get_sglang_pass_info(path)

        if info.get("is_decode", False):
            first_pos = info.get("first_position", -1)
            if first_pos == decode_position:
                return (pass_id, path)

    return None


def list_all_passes(dump_dir: str) -> list[tuple[int, Path]]:
    """List all available pass files with their IDs."""
    dump_path = Path(dump_dir)
    if not dump_path.exists():
        return []
    
    passes = []
    for f in dump_path.glob("*/Pass*.pt"):
        name = f.stem
        if name.startswith("Pass"):
            try:
                pass_id = int(name[4:])
                passes.append((pass_id, f))
            except ValueError:
                continue
    return sorted(passes, key=lambda x: x[0])


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
    

def get_sglang_pass_info(path: Path) -> dict[str, Any]:
    """Extract information from a SGLang dump file."""
    tensors = torch.load(path, map_location="cpu")

    info = {"path": path}

    if "model.forward_batch_info.input_ids" in tensors:
        ids = tensors["model.forward_batch_info.input_ids"]
        info["input_ids"] = ids
        info["seq_len"] = ids.numel()
        if ids.numel() > 0:
            info["first_token"] = ids.flatten()[0].item()

    if "model.forward_batch_info.positions" in tensors:
        pos = tensors["model.forward_batch_info.positions"]
        info["positions"] = pos
        if pos.numel() > 0:
            info["first_position"] = pos.flatten()[0].item()
            info["last_position"] = pos.flatten()[-1].item()

        if "model.forward_batch_info.seq_lens" in tensors:
            seq_lens = tensors["model.forward_batch_info.seq_lens"]
            if seq_lens.numel() > 0:
                if seq_lens.numel() == 1:
                    info["batch_seq_len"] = seq_lens.item()
                else:
                    info["batch_seq_len"] = seq_lens.tolist()

    info["is_prefill"] = info.get("seq_len", 0) > 1
    info["is_decode"] = info.get("seq_len", 0) == 1

    return info


def get_fsdp_dump_info(path: Path) -> dict[str, Any]:
    """Extract information from a FSDP/Megatron dump file."""
    tensors = torch.load(path, map_location="cpu")

    info = {"path": path, "tensors": tensors}

    # Check if FSDP or Megatron format
    if "fsdp_input_ids" in tensors:
        info["backend"] = "FSDP"
        info["input_ids_key"] = "fsdp_input_ids"
        info["compared_token_key"] = "fsdp_compared_token_id"
        info["compared_pos_key"] = "fsdp_compared_position"
    elif "megatron_input_ids" in tensors:
        info["backend"] = "Megatron"
        info["input_ids_key"] = "megatron_input_ids"
        info["compared_token_key"] = "megatron_compared_token_id"
        info["compared_pos_key"] = "megatron_compared_position"
    else:
        info["backend"] = "Unknown"

    if "prompt_len" in tensors:
        info["prompt_len"] = int(tensors["prompt_len"].item())
    elif "debug_prompt_len" in tensors:
        info["prompt_len"] = int(tensors["debug_prompt_len"].item())

    if "seq_len" in tensors:
        info["total_len"] = int(tensors["seq_len"].item())
    elif "debug_total_len" in tensors:
        info["total_len"] = int(tensors["debug_total_len"].item())

    if "response_len" in tensors:
        info["response_len"] = int(tensors["response_len"].item())
    elif "debug_response_len" in tensors:
        info["response_len"] = int(tensors["debug_response_len"].item())

    if "response_logits_positions" in tensors:
        positions = tensors["response_logits_positions"]
        info["response_positions"] = positions.tolist()

    return info


def find_sglang_prefill_pass(
    sglang_dir: str, prompt_len: int | None = None
) -> tuple[int, Path] | None:
    """
    Find the SGLang prefill pass.

    The prefill pass processes the entire prompt and outputs logits
    predicting the first response token.
    It has seq_len = prompt_len (processes all prompt tokens at once).
    """
    passes = list_all_passes(sglang_dir)

    prefill_passes = []

    for pass_id, path in passes:
        info = get_sglang_pass_info(path)

        if info.get("is_prefill", False):
            seq_len = info.get("seq_len", 0)
            prefill_passes.append((pass_id, path, seq_len, info))

    if not prefill_passes:
        return None

    # If prompt_len is specified, find the matching prefill pass
    if prompt_len is not None:
        for pass_id, path, seq_len, info in prefill_passes:
            if seq_len == prompt_len:
                return (pass_id, path)

    # Otherwise return the first prefill pass
    return (prefill_passes[0][0], prefill_passes[0][1])


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


def compare_hidden_states_at_position(
    sglang_tensors: dict[str, torch.Tensor],
    fsdp_tensors: dict[str, torch.Tensor],
    sglang_position: int,
    fsdp_position: int,
    verbose: bool = True,
    sglang_decode_tensors: dict[str, torch.Tensor] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Compare hidden states at a specific position across all layers.

    For first response token comparison:
    - SGLang prefill: hidden states at last position (sglang_position)
    - FSDP training: hidden states at (fsdp_position)

    These should match because both represent the same position.
    """
    results = {}
    
    print("\n" + "=" * 70)
    print("LAYER-BY-LAYER HIDDEN STATE COMPARISON")
    print(f"SGLang pos: {sglang_position}, FSDP pos: {fsdp_position}")
    print("=" * 70)

    # Find all layer outputs in both dumps
    # Try multiple key patterns for SGLang
    sglang_layers = {}

    # SGLang uses module names like:
    #   model.layers.0.input_layernorm
    #   model.layers.0.self_attn.o_proj
    #   model.layers.0.post_attention_layernorm
    # We prefer the output of each layer (e.g., after post_attention_layernorm)
    sglang_priority_patterns = [
        "post_attention_layernorm",  # After full layer
        "mlp",                       # After MLP
        "self_attn.o_proj",          # After attention output projection
        "self_attn",                 # After attention
        "input_layernorm",           # Layer input
    ]

    for pattern in sglang_priority_patterns:
        if sglang_layers:
            break  # Found layers with previous pattern
        for name in sglang_tensors.keys():
            if "layers." in name and pattern in name:
                # Extract layer index from name like "model.layers.5.xxx"
                match = re.search(r"layers\.(\d+)\.", name)
                if match:
                    layer_idx = int(match.group(1))
                    if layer_idx not in sglang_layers:
                        tensor = sglang_tensors[name]
                        sglang_layers[layer_idx] = (name, tensor)

    # Try multiple key patterns for FSDP
    fsdp_layers = {}

    # FSDP uses patterns like:
    #   layer_0_output (main layer output)
    #   layer_0_input_layernorm_output (sublayer output)
    fsdp_priority_patterns = [
        r"^layer_(\d+)_output$",           # Main layer output
        r"^layer_(\d+)_[a-z_]+_output$",   # Sublayer output
    ]

    for pattern in fsdp_priority_patterns:
        if fsdp_layers:
            break
        for name in fsdp_tensors.keys():
            match = re.match(pattern, name)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx not in fsdp_layers:
                    fsdp_layers[layer_idx] = (name, fsdp_tensors[name])

    if not sglang_layers:
        print("  No layer outputs found in SGLang dump!")
        layer_keys = [k for k in sglang_tensors.keys() if "layer" in k.lower()]
        print(f"  Available keys: {layer_keys[:15]}")

    if not fsdp_layers:
        print("  No layer outputs found in FSDP dump!")
        layer_keys = [k for k in fsdp_tensors.keys() if "layer" in k.lower()]
        print(f"  Available keys: {layer_keys[:15]}")

    # Show which pattern matched
    if sglang_layers:
        sample_key = list(sglang_layers.values())[0][0]
        print(f"  SGLang using key pattern: {sample_key}")
    if fsdp_layers:
        sample_key = list(fsdp_layers.values())[0][0]
        print(f"  FSDP using key pattern: {sample_key}")

    # Compare each layer
    all_layers = sorted(set(sglang_layers.keys()) | set(fsdp_layers.keys()))

    significant_diff_layers = []

    for layer_idx in all_layers:
        if layer_idx not in sglang_layers:
            if verbose:
                print(f"  Layer {layer_idx:2d}: NOT IN SGLang dump")
            continue
        if layer_idx not in fsdp_layers:
            if verbose:
                print(f"  Layer {layer_idx:2d}: NOT IN FSDP dump")
            continue

        # Extract tensor from (name, tensor) tuple
        sg_name, sglang_hidden = sglang_layers[layer_idx]
        fsdp_name, fsdp_hidden = fsdp_layers[layer_idx]

        # Helper to convert list/tuple to tensor
        def to_tensor(x):
            if isinstance(x, (list, tuple)):
                if len(x) == 0:
                    return None
                # Take first element if list
                x = x[0]
            if not isinstance(x, torch.Tensor):
                return None
            return x

        sglang_hidden = to_tensor(sglang_hidden)
        fsdp_hidden = to_tensor(fsdp_hidden)

        if sglang_hidden is None:
            if verbose:
                print(f"  Layer {layer_idx:2d}: SGLang tensor is None/empty")
            continue
        if fsdp_hidden is None:
            if verbose:
                print(f"  Layer {layer_idx:2d}: FSDP tensor is None/empty")
            continue

        # Extract the specific position from each tensor
        sg_at_pos = sglang_hidden
        fsdp_at_pos = fsdp_hidden

        # Handle SGLang tensor - extract at position if needed
        if sg_at_pos.dim() == 3:
            # [batch, seq_len, hidden]
            if sglang_position < sg_at_pos.shape[1]:
                sg_at_pos = sg_at_pos[:, sglang_position:sglang_position+1, :]
        elif sg_at_pos.dim() == 2:
            # [seq_len, hidden]
            if sglang_position < sg_at_pos.shape[0]:
                sg_at_pos = sg_at_pos[sglang_position:sglang_position+1, :]

        # Handle FSDP tensor - extract at position if needed
        if fsdp_at_pos.dim() == 3:
            # [batch, seq_len, hidden]
            if fsdp_position < fsdp_at_pos.shape[1]:
                fsdp_at_pos = fsdp_at_pos[:, fsdp_position:fsdp_position+1, :]
        elif fsdp_at_pos.dim() == 2:
            # [seq_len, hidden]
            if fsdp_position < fsdp_at_pos.shape[0]:
                fsdp_at_pos = fsdp_at_pos[fsdp_position:fsdp_position+1, :]

        # Flatten for comparison
        sg_flat = sg_at_pos.flatten()
        fsdp_flat = fsdp_at_pos.flatten()

        # Align shapes if needed
        if sg_flat.shape != fsdp_flat.shape:
            min_len = min(len(sg_flat), len(fsdp_flat))
            sg_flat = sg_flat[:min_len]
            fsdp_flat = fsdp_flat[:min_len]

        stats = compute_diff_stats(sg_flat, fsdp_flat)
        results[f"layer_{layer_idx}"] = stats

        if stats["max_diff"] >= 1e-5:
            significant_diff_layers.append((layer_idx, stats["max_diff"]))

        match_str = "✓" if stats["max_diff"] < 1e-5 else "✗"
        color = "" if stats["max_diff"] < 1e-5 else "\033[91m"
        end_color = "\033[0m" if color else ""

        if verbose:
            print(
                f"  {color}Layer {layer_idx:2d}: {match_str} "
                f"max_diff={stats['max_diff']:.6e}, "
                f"mean_diff={stats['mean_diff']:.6e}{end_color}"
            )

            # Always show tensor shapes
            print(f"    SGLang shape: {sglang_hidden.shape}, "
                  f"FSDP shape: {fsdp_hidden.shape}")

            # Show first 10 values if there's a significant difference
            if stats["max_diff"] >= 1e-5:
                n_show = min(10, len(fsdp_flat))

                # Show SGLang at BOTH positions (90 and 91) vs FSDP
                # to figure out which position FSDP extracted at
                print(f"\n    Comparing SGLang pos {sglang_position} "
                      f"and {sglang_position + 1} vs FSDP:")

                # Get FSDP values (already extracted)
                fsdp_vals = fsdp_flat[:n_show].float().tolist()
                print(f"    FSDP (extracted): "
                      f"{[f'{v:.4f}' for v in fsdp_vals]}")

                # SGLang PREFILL at sglang_position (last token of prefill)
                max_diff0 = float('inf')
                max_diff1 = float('inf')

                if sglang_hidden.dim() == 2:
                    seq_len = sglang_hidden.shape[0]
                    if sglang_position < seq_len:
                        sg_pos0 = sglang_hidden[sglang_position, :n_show]
                        sg_pos0_vals = sg_pos0.float().tolist()
                        diff0 = (sg_pos0.float() - fsdp_flat[:n_show].float())
                        diff0 = diff0.abs().tolist()
                        max_diff0 = max(diff0) if diff0 else 0
                        print(f"    SGLang prefill[{sglang_position}]: "
                              f"{[f'{v:.4f}' for v in sg_pos0_vals]}")
                        print(f"    Diff prefill[{sglang_position}]:   "
                              f"{[f'{v:.4f}' for v in diff0]} "
                              f"(max={max_diff0:.4f})")
                elif sglang_hidden.dim() == 3:
                    seq_len = sglang_hidden.shape[1]
                    if sglang_position < seq_len:
                        sg_pos0 = sglang_hidden[0, sglang_position, :n_show]
                        sg_pos0_vals = sg_pos0.float().tolist()
                        diff0 = (sg_pos0.float() - fsdp_flat[:n_show].float())
                        diff0 = diff0.abs().tolist()
                        max_diff0 = max(diff0) if diff0 else 0
                        print(f"    SGLang prefill[{sglang_position}]: "
                              f"{[f'{v:.4f}' for v in sg_pos0_vals]}")
                        print(f"    Diff prefill[{sglang_position}]:   "
                              f"{[f'{v:.4f}' for v in diff0]} "
                              f"(max={max_diff0:.4f})")

                # SGLang DECODE pass for next position
                # (position 91 is in the first decode pass, not prefill)
                next_pos = sglang_position + 1
                if sglang_decode_tensors is not None:
                    # Find matching layer in decode tensors
                    decode_hidden = None
                    for dname in sglang_decode_tensors.keys():
                        if sg_name.split(".")[-1] in dname:
                            # Found matching layer
                            dh = sglang_decode_tensors[dname]
                            if isinstance(dh, (list, tuple)):
                                dh = dh[0] if len(dh) > 0 else None
                            if isinstance(dh, torch.Tensor):
                                decode_hidden = dh
                                break

                    if decode_hidden is not None:
                        # Decode pass has shape [1, hidden] - single token
                        if decode_hidden.dim() == 2:
                            dh_flat = decode_hidden[0, :n_show]
                        elif decode_hidden.dim() == 1:
                            dh_flat = decode_hidden[:n_show]
                        else:
                            dh_flat = decode_hidden.flatten()[:n_show]

                        dh_vals = dh_flat.float().tolist()
                        diff1 = (dh_flat.float() - fsdp_flat[:n_show].float())
                        diff1 = diff1.abs().tolist()
                        max_diff1 = max(diff1) if diff1 else 0
                        print(f"    SGLang decode[{next_pos}]:  "
                              f"{[f'{v:.4f}' for v in dh_vals]}")
                        print(f"    Diff decode[{next_pos}]:    "
                              f"{[f'{v:.4f}' for v in diff1]} "
                              f"(max={max_diff1:.4f})")
                    else:
                        print(f"    SGLang decode[{next_pos}]: "
                              f"(layer not found in decode tensors)")
                else:
                    print(f"    SGLang decode[{next_pos}]: "
                          f"(decode tensors not loaded)")

                # Suggest which position matches better
                if max_diff1 < max_diff0:
                    print(f"    → SGLang decode[{next_pos}] matches FSDP "
                          f"better!")
                elif max_diff0 < max_diff1:
                    print(f"    → SGLang prefill[{sglang_position}] matches "
                          f"FSDP better!")

    # Summary
    if significant_diff_layers:
        first_layer = significant_diff_layers[0][0]
        first_diff = significant_diff_layers[0][1]
        print(f"\n⚠️  FIRST SIGNIFICANT DIFFERENCE at layer {first_layer}")
        print(f"   Max diff: {first_diff:.6e}")
    else:
        print("\n✓ All layers match (diff < 1e-5)")

    print("=" * 70)
    
    return results


def compare_first_response_token(
    sglang_dir: str,
    fsdp_dir: str,
    verbose: bool = True,
) -> None:
    """
    Compare the first response token between SGLang and FSDP.

    This is the key comparison for true on-policy verification:
    - SGLang: Uses prefill pass to compute logits for first response token
    - FSDP: Uses logits at position (prompt_len - 1) from training pass

    We compare:
    1. Hidden states at each layer for position (prompt_len - 1)
    2. Logits for the first response token
    3. Logprobs computed from logits
    """
    print("\n" + "=" * 70)
    print("FIRST RESPONSE TOKEN COMPARISON")
    print("=" * 70)
    print("\nComparing SGLang prefill pass vs FSDP/Megatron training pass")
    print("for the FIRST response token prediction.\n")

    # Find FSDP dump (should only be ONE pass for training)
    fsdp_passes = list_all_passes(fsdp_dir)
    if not fsdp_passes:
        print(f"ERROR: No FSDP dump files found in {fsdp_dir}")
        return

    # Training should have only ONE pass
    if len(fsdp_passes) > 1:
        print(f"WARNING: Found {len(fsdp_passes)} FSDP passes, using first.")

    fsdp_pass_id, fsdp_path = fsdp_passes[0]
    fsdp_info = get_fsdp_dump_info(fsdp_path)
    fsdp_tensors = fsdp_info["tensors"]

    print("FSDP/Megatron Info:")
    print(f"  Backend: {fsdp_info.get('backend', 'Unknown')}")
    print(f"  Prompt length: {fsdp_info.get('prompt_len', 'N/A')}")
    print(f"  Total length: {fsdp_info.get('total_len', 'N/A')}")
    print(f"  Response length: {fsdp_info.get('response_len', 'N/A')}")
    if "response_positions" in fsdp_info:
        resp_pos = fsdp_info['response_positions']
        print(f"  Response logits positions: {resp_pos}")

    prompt_len = fsdp_info.get("prompt_len")
    if prompt_len is None:
        print("ERROR: Could not determine prompt_len from FSDP dump")
        return

    # Find SGLang prefill pass
    prefill_result = find_sglang_prefill_pass(sglang_dir, prompt_len)
    if prefill_result is None:
        print("ERROR: Could not find SGLang prefill pass")
        passes = list_all_passes(sglang_dir)
        print(f"  Found {len(passes)} SGLang passes:")
        for pass_id, path in passes[:10]:
            info = get_sglang_pass_info(path)
            print(
                f"    Pass {pass_id}: seq_len={info.get('seq_len')}, "
                f"is_prefill={info.get('is_prefill')}"
            )
        return

    sglang_prefill_id, sglang_prefill_path = prefill_result
    sglang_prefill_info = get_sglang_pass_info(sglang_prefill_path)
    sglang_prefill_tensors = torch.load(
        sglang_prefill_path, map_location="cpu"
    )

    print("\nSGLang Prefill Info:")
    print(f"  Using Pass {sglang_prefill_id} (prefill)")
    seq_len = sglang_prefill_info.get('seq_len', 'N/A')
    first_pos = sglang_prefill_info.get('first_position', 'N/A')
    last_pos = sglang_prefill_info.get('last_position', 'N/A')
    print(f"  Sequence length: {seq_len}")
    print(f"  First position: {first_pos}")
    print(f"  Last position: {last_pos}")

    # Position analysis
    comparison_pos = prompt_len - 1  # Position of last prompt token
    first_response_pos = prompt_len   # Position of first response token

    print("\nPosition Mapping (IMPORTANT):")
    print("  SGLang decode at first_position=X predicts token at X")
    print("  FSDP logits_pos_N predicts token at N+1")
    print(f"  To compare first response token (pos {first_response_pos}):")
    print(f"    - FSDP: use logits_pos_{comparison_pos}")
    print(f"    - SGLang: decode pass first_position={first_response_pos}")

    # Find the correct SGLang decode pass for first response token
    # SGLang decode at first_position=X predicts token at position X
    sglang_decode_result = find_sglang_decode_pass(
        sglang_dir, first_response_pos
    )
    if sglang_decode_result is not None:
        sg_decode_id, sg_decode_path = sglang_decode_result
        sglang_tensors = torch.load(sg_decode_path, map_location="cpu")
        print(f"\n  Using SGLang decode pass {sg_decode_id} "
              f"(first_position={first_response_pos})")
    else:
        # Fallback to prefill if decode pass not found
        print(f"\n  WARNING: Could not find SGLang decode pass "
              f"for position {first_response_pos}")
        print("  Falling back to prefill (may have position mismatch!)")
        sglang_tensors = sglang_prefill_tensors

    # =========================================================================
    # 1. Compare hidden states at each layer
    # =========================================================================
    # For hidden states, use PREFILL pass (has full prompt context)
    # SGLang prefill processes positions [0, ..., prompt_len-1]
    # FSDP processes full sequence, extract at prompt_len - 1
    sglang_prefill_last_pos = sglang_prefill_info.get("seq_len", 1) - 1

    print("\n  Hidden state comparison:")
    print(f"    SGLang: prefill last position = {sglang_prefill_last_pos}")
    print(f"    FSDP: position {comparison_pos} (prompt_len - 1)")

    # Check what position FSDP extracted at (for info only)
    if "fsdp_compared_position" in fsdp_tensors:
        fsdp_dump_pos = fsdp_tensors["fsdp_compared_position"].item()
        print(f"    FSDP dump extracted at: {fsdp_dump_pos}")
        if fsdp_dump_pos != comparison_pos:
            print(f"    WARNING: Dump position {fsdp_dump_pos} != "
                  f"comparison_pos {comparison_pos}")

    # IMPORTANT: Use comparison_pos (prompt_len - 1) for FSDP
    # This matches SGLang prefill's last position
    fsdp_hidden_pos = comparison_pos

    # Both should be the same position for apples-to-apples comparison
    if sglang_prefill_last_pos != fsdp_hidden_pos:
        print(f"    ⚠️ Position mismatch! Adjusting FSDP to "
              f"{sglang_prefill_last_pos}")
        fsdp_hidden_pos = sglang_prefill_last_pos

    # Load decode pass tensors for position 91 comparison
    sglang_decode_for_hidden = None
    decode_result = find_sglang_decode_pass(sglang_dir, first_response_pos)
    if decode_result is not None:
        decode_id, decode_path = decode_result
        sglang_decode_for_hidden = torch.load(decode_path, map_location="cpu")
        print(f"    Also comparing with decode pass {decode_id} "
              f"(first_pos={first_response_pos})")

    # Compare hidden states using PREFILL tensors for SGLang
    compare_hidden_states_at_position(
        sglang_prefill_tensors,  # Use prefill for hidden states
        fsdp_tensors,
        sglang_position=sglang_prefill_last_pos,
        fsdp_position=fsdp_hidden_pos,
        verbose=verbose,
        sglang_decode_tensors=sglang_decode_for_hidden,
    )

    # =========================================================================
    # 2. Compare logits for first response token
    # =========================================================================
    print("\n" + "=" * 70)
    print("LOGITS COMPARISON FOR FIRST RESPONSE TOKEN")
    print("=" * 70)
    
    # Get first response token ID
    first_response_token = None
    input_ids_key = fsdp_info.get("input_ids_key", "fsdp_input_ids")
    if input_ids_key in fsdp_tensors:
        input_ids = fsdp_tensors[input_ids_key].flatten()
        if first_response_pos < len(input_ids):
            first_response_token = input_ids[first_response_pos].item()
            print(
                f"\nFirst response token ID: {first_response_token} "
                f"(at position {first_response_pos})"
            )
    
    # Get SGLang logits
    sglang_logits = None
    if "logits_processor" in sglang_tensors:
        sglang_logits = sglang_tensors["logits_processor"]
        print(
            f"SGLang logits (logits_processor): "
            f"shape={sglang_logits.shape}, dtype={sglang_logits.dtype}"
        )

    # Get FSDP logits at prompt_len - 1
    fsdp_logits = None
    logits_key = f"logits_pos_{comparison_pos}"

    if logits_key in fsdp_tensors:
        fsdp_logits = fsdp_tensors[logits_key]
        print(
            f"FSDP {logits_key}: "
            f"shape={fsdp_logits.shape}, dtype={fsdp_logits.dtype}"
        )
    elif "logits_at_prompt_end" in fsdp_tensors:
        fsdp_logits = fsdp_tensors["logits_at_prompt_end"]
        print(
            f"FSDP logits_at_prompt_end: "
            f"shape={fsdp_logits.shape}, dtype={fsdp_logits.dtype}"
        )
    else:
        print(f"WARNING: Could not find {logits_key} in FSDP dump")
        available = [k for k in fsdp_tensors.keys() if "logits" in k.lower()]
        print(f"  Available logits keys: {available}")
    
    # Compare logits
    if sglang_logits is not None and fsdp_logits is not None:
        sg_flat = sglang_logits.flatten()
        fsdp_flat = fsdp_logits.flatten()

        if sg_flat.shape != fsdp_flat.shape:
            min_len = min(len(sg_flat), len(fsdp_flat))
            print(
                f"  Shape mismatch: SGLang {sg_flat.shape} vs "
                f"FSDP {fsdp_flat.shape}, using first {min_len}"
            )
            sg_flat = sg_flat[:min_len]
            fsdp_flat = fsdp_flat[:min_len]

        stats = compute_diff_stats(sg_flat, fsdp_flat)

        print("\nLogits Comparison:")
        print(f"  Max diff:  {stats['max_diff']:.8e}")
        print(f"  Mean diff: {stats['mean_diff']:.8e}")
        print(f"  Rel diff:  {stats['rel_diff']:.8e}")

        if stats["max_diff"] < 1e-5:
            print("  ✓ Logits MATCH!")
        else:
            print("  ✗ Logits DIFFER!")
    
        # Compare specific token logit
        if first_response_token is not None:
            sg_tok = sglang_logits.flatten()
            fsdp_tok = fsdp_logits.flatten()
            if first_response_token < len(sg_tok):
                sg_token_logit = sg_tok[first_response_token]
            else:
                sg_token_logit = None
            if first_response_token < len(fsdp_tok):
                fsdp_token_logit = fsdp_tok[first_response_token]
            else:
                fsdp_token_logit = None

            if sg_token_logit is not None and fsdp_token_logit is not None:
                sg_val = sg_token_logit.float().item()
                fsdp_val = fsdp_token_logit.float().item()
                diff = abs(sg_val - fsdp_val)
                tok = first_response_token
                print(f"\n  Logit for first response token {tok}:")
                print(f"    SGLang: {sg_val:.8f}")
                print(f"    FSDP:   {fsdp_val:.8f}")
                print(f"    Diff:   {diff:.8e}")

        # =====================================================================
        # 3. Compare logprobs (computed from logits)
        # =====================================================================
        print("\n" + "-" * 50)
        print("LOGPROBS COMPARISON (computed from logits)")
        print("-" * 50)

        tok_id = first_response_token
        sg_logprobs, sg_target_lp = compute_logprobs_from_logits(
            sglang_logits, temperature=1.0, target_token_id=tok_id
        )
        fsdp_logprobs, fsdp_target_lp = compute_logprobs_from_logits(
            fsdp_logits, temperature=1.0, target_token_id=tok_id
        )

        sg_lp_flat = sg_logprobs.flatten()
        fsdp_lp_flat = fsdp_logprobs.flatten()

        if sg_lp_flat.shape != fsdp_lp_flat.shape:
            min_len = min(len(sg_lp_flat), len(fsdp_lp_flat))
            sg_lp_flat = sg_lp_flat[:min_len]
            fsdp_lp_flat = fsdp_lp_flat[:min_len]

        lp_stats = compute_diff_stats(sg_lp_flat, fsdp_lp_flat)

        print("  Full distribution comparison:")
        print(f"    Max diff:  {lp_stats['max_diff']:.8e}")
        print(f"    Mean diff: {lp_stats['mean_diff']:.8e}")

        if sg_target_lp is not None and fsdp_target_lp is not None:
            sg_lp_val = sg_target_lp.float().item()
            fsdp_lp_val = fsdp_target_lp.float().item()
            diff = abs(sg_lp_val - fsdp_lp_val)
            tok = first_response_token
            print(f"\n  Logprob for first response token {tok}:")
            print(f"    SGLang: {sg_lp_val:.8f}")
            print(f"    FSDP:   {fsdp_lp_val:.8f}")
            print(f"    Diff:   {diff:.8e}")

            if diff < 1e-5:
                print("    ✓ Logprobs MATCH!")
            else:
                print("    ✗ Logprobs DIFFER!")
    
        # =====================================================================
        # 4. Compare directly-dumped logprobs (if available)
        # =====================================================================
        print("\n" + "-" * 50)
        print("DIRECTLY-DUMPED LOGPROBS COMPARISON")
        print("-" * 50)
        
        # Check FSDP dumped logprobs
        fsdp_direct_lp = None
        if "logprobs" in fsdp_tensors:
            fsdp_direct_lp = fsdp_tensors["logprobs"]
            print(f"\n  FSDP dumped logprobs: {fsdp_direct_lp}")
            print(f"    shape: {fsdp_direct_lp.shape}")
            print(f"    dtype: {fsdp_direct_lp.dtype}")

        if "logprobs_full" in fsdp_tensors:
            full_lp = fsdp_tensors["logprobs_full"]
            print(f"\n  FSDP full logprobs shape: {full_lp.shape}")
            print(f"    First 5 values: {full_lp.flatten()[:5].tolist()}")

        if "logprobs_extracted_idx" in fsdp_tensors:
            idx = fsdp_tensors["logprobs_extracted_idx"].item()
            print(f"    Extracted at index: {idx}")

        if "logprobs_prompt_len" in fsdp_tensors:
            pl = fsdp_tensors["logprobs_prompt_len"].item()
            print(f"    prompt_len used: {pl}")

        if "response_logprobs_first5" in fsdp_tensors:
            resp_lp = fsdp_tensors["response_logprobs_first5"]
            print(f"    First 5 response logprobs: {resp_lp.tolist()}")

        # Check if FSDP logprob is zero (common bug symptom)
        if fsdp_direct_lp is not None:
            if fsdp_direct_lp.abs().max().item() < 1e-10:
                print("\n  ⚠️  WARNING: FSDP dumped logprobs are ALL ZEROS!")
                print("     This indicates a bug in the dumper.")
                print("     Possible causes:")
                print("     - Wrong index used for extraction")
                print("     - Position out of bounds")
                print("     - log_probs tensor is empty/wrong shape")

    # =========================================================================
    # 5. Detailed logprob comparison at TWO positions
    # =========================================================================
    print("\n" + "=" * 70)
    print("DETAILED LOGPROBS: FIRST AND SECOND RESPONSE TOKENS")
    print("=" * 70)
    print("\nPosition Mapping:")
    print("  SGLang decode at first_position=X predicts token at X")
    print("  FSDP logits_pos_N predicts token at N+1")
    print("\nComparing predictions for:")
    second_resp_pos = first_response_pos + 1
    p1 = first_response_pos
    p2 = second_resp_pos
    print(f"  1. First response token (pos {p1}):")
    print(f"     FSDP pos {comparison_pos} vs SGLang first_pos {p1}")
    print(f"  2. Second response token (pos {p2}):")
    print(f"     FSDP pos {p1} vs SGLang first_pos {p2}")

    # Get tokens for both positions
    second_response_token = None
    input_ids_key = fsdp_info.get("input_ids_key", "fsdp_input_ids")
    if input_ids_key in fsdp_tensors:
        input_ids = fsdp_tensors[input_ids_key].flatten()
        if second_resp_pos < len(input_ids):
            second_response_token = input_ids[second_resp_pos].item()

    # =========================================================
    # FIRST RESPONSE TOKEN (position prompt_len)
    # =========================================================
    print("\n" + "-" * 50)
    print(f"FIRST RESPONSE TOKEN (position {first_response_pos})")
    print(f"Actual token: {first_response_token}")
    print("-" * 50)

    # SGLang: use decode pass at first_position = prompt_len
    # (We already loaded this as sglang_logits above)
    if sglang_logits is not None:
        print_top_logprobs(
            sglang_logits,
            actual_token_id=first_response_token,
            label=f"SGLang (decode, first_pos={first_response_pos})",
            top_k=10,
        )

    # FSDP: use logits_pos_{prompt_len - 1}
    if fsdp_logits is not None:
        print_top_logprobs(
            fsdp_logits,
            actual_token_id=first_response_token,
            label=f"FSDP (logits_pos_{comparison_pos})",
            top_k=10,
        )

    # =========================================================
    # SECOND RESPONSE TOKEN (position prompt_len + 1)
    # =========================================================
    print("\n" + "-" * 50)
    print(f"SECOND RESPONSE TOKEN (position {second_resp_pos})")
    print(f"Actual token: {second_response_token}")
    print("-" * 50)

    # SGLang: use decode pass at first_position = prompt_len + 1
    sglang_decode2 = find_sglang_decode_pass(sglang_dir, second_resp_pos)
    sglang_logits2 = None
    if sglang_decode2 is not None:
        decode2_id, decode2_path = sglang_decode2
        decode2_tensors = torch.load(decode2_path, map_location="cpu")
        if "logits_processor" in decode2_tensors:
            sglang_logits2 = decode2_tensors["logits_processor"]
            print(f"\n  SGLang: decode pass {decode2_id} "
                  f"(first_pos={second_resp_pos})")
        else:
            print(f"\n  SGLang pass {decode2_id} has no logits_processor")
    else:
        print(f"\n  No SGLang decode pass for pos {second_resp_pos}")
        passes = list_all_passes(sglang_dir)
        decode_passes = []
        for pid, ppath in passes:
            info = get_sglang_pass_info(ppath)
            if info.get("is_decode", False):
                decode_passes.append((pid, info.get("first_position", "?")))
        if decode_passes:
            print(f"  Available decode passes: {decode_passes[:5]}...")

    if sglang_logits2 is not None:
        print_top_logprobs(
            sglang_logits2,
            actual_token_id=second_response_token,
            label=f"SGLang (decode, first_pos={second_resp_pos})",
            top_k=10,
        )

    # FSDP: use logits_pos_{prompt_len} to predict position prompt_len + 1
    fsdp_logits2 = None
    logits_key2 = f"logits_pos_{first_response_pos}"
    if logits_key2 in fsdp_tensors:
        fsdp_logits2 = fsdp_tensors[logits_key2]
        print(f"\n  FSDP: logits_pos_{first_response_pos}")
    else:
        print(f"\n  FSDP does not have {logits_key2}")
        avail = [k for k in fsdp_tensors.keys() if k.startswith("logits_pos_")]
        if avail:
            print(f"  Available: {avail[:5]}...")

    if fsdp_logits2 is not None:
        print_top_logprobs(
            fsdp_logits2,
            actual_token_id=second_response_token,
            label=f"FSDP (logits_pos_{first_response_pos})",
            top_k=10,
        )

    # Compare second response token predictions
    if sglang_logits2 is not None and fsdp_logits2 is not None:
        print("\n  Second response token comparison:")
        sg_flat = sglang_logits2.flatten()
        fsdp_flat = fsdp_logits2.flatten()
        if sg_flat.shape == fsdp_flat.shape:
            stats = compute_diff_stats(sg_flat, fsdp_flat)
            print(f"    Max diff:  {stats['max_diff']:.8e}")
            print(f"    Mean diff: {stats['mean_diff']:.8e}")
            if stats["max_diff"] < 1e-5:
                print("    ✓ Second response token logits MATCH!")
            else:
                print("    ✗ Second response token logits DIFFER!")
        else:
            print(f"    Shape mismatch: {sg_flat.shape} vs {fsdp_flat.shape}")

    print("\n" + "=" * 70)


def list_passes_detailed(sglang_dir: str, fsdp_dir: str) -> None:
    """List all passes with detailed information."""
    print("\n" + "=" * 70)
    print("SGLANG PASSES (Inference)")
    print("=" * 70)

    sglang_passes = list_all_passes(sglang_dir)
    for pass_id, path in sglang_passes[:30]:
        info = get_sglang_pass_info(path)
        pass_type = "PREFILL" if info.get("is_prefill") else "DECODE"
        first_pos = info.get('first_position', '?')
        last_pos = info.get('last_position', '?')
        seq_len = info.get('seq_len', '?')
        print(
            f"  Pass {pass_id:3d}: {pass_type:7s} seq_len={seq_len:3}, "
            f"positions={first_pos}-{last_pos}"
        )

    if len(sglang_passes) > 30:
        print(f"  ... and {len(sglang_passes) - 30} more passes")

    print("\n" + "=" * 70)
    print("FSDP/MEGATRON PASSES (Training)")
    print("=" * 70)

    fsdp_passes = list_all_passes(fsdp_dir)
    for pass_id, path in fsdp_passes[:10]:
        info = get_fsdp_dump_info(path)
        backend = info.get('backend', 'Unknown')
        prompt_len = info.get('prompt_len', '?')
        total_len = info.get('total_len', '?')
        response_len = info.get('response_len', '?')
        print(
            f"  Pass {pass_id:3d}: {backend:8s} "
            f"prompt_len={prompt_len}, total_len={total_len}, "
            f"response_len={response_len}"
        )

    if len(fsdp_passes) > 10:
        print(f"  ... and {len(fsdp_passes) - 10} more passes")

    print("\n" + "=" * 70)
    print("KEY INSIGHT:")
    print("=" * 70)
    print("  - SGLang: MANY passes (1 prefill + N decode passes)")
    print("  - FSDP/Megatron: ONE pass for entire sequence")
    print("  - To compare first response token:")
    print("    * Use SGLang's PREFILL pass (seq_len = prompt_len)")
    print("    * Compare with FSDP's logits at position (prompt_len - 1)")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Compare SGLang and FSDP/Megatron tensor dumps"
    )
    parser.add_argument(
        "--sglang-dir", type=str, required=True,
        help="SGLang tensor dump directory"
    )
    parser.add_argument(
        "--fsdp-dir", "--megatron-dir", type=str, required=True,
        dest="fsdp_dir", help="FSDP/Megatron tensor dump directory"
    )
    parser.add_argument(
        "--compare-first-token", action="store_true",
        help="Compare first response token (main use case)"
    )
    parser.add_argument(
        "--list-passes", action="store_true",
        help="List all available passes and exit"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
                        )
                        
    # Legacy arguments for backwards compatibility
    parser.add_argument("--pass-id", type=int, default=0)
    parser.add_argument("--auto-match", action="store_true")
    parser.add_argument("--decode-only", action="store_true")
    parser.add_argument("--response-start", type=int, default=None)
    parser.add_argument("--compare-all-positions", action="store_true")
    parser.add_argument("--compare-positions", type=str, default=None)
    parser.add_argument("--sglang-pass-id", type=int, default=None)

    args = parser.parse_args()

    # Handle list-passes mode
    if args.list_passes:
        list_passes_detailed(args.sglang_dir, args.fsdp_dir)
        sys.exit(0)
    
    # Default to compare-first-token
    compare_first_response_token(
        args.sglang_dir,
        args.fsdp_dir,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
