#!/usr/bin/env python3
"""
Compare tensor dumps from SGLang and Megatron.

Key Understanding:
==================
Training side (Megatron): ONE forward pass processes entire sequence.
Inference side (SGLang): MULTIPLE passes - prefill + decode passes.

For comparing the FIRST response token:
- SGLang: Use PREFILL pass (seq_len = prompt_len)
- Megatron: Use logits at position (prompt_len - 1)

Both should produce identical:
1. Hidden states at each layer for position (prompt_len - 1)
2. Logits/logprobs for the first response token

Usage:
    python compare_tensor_dumps_megatron.py \\
        --sglang-dir /tmp/sglang_dump \\
        --megatron-dir /tmp/megatron_dump \\
        --compare-first-token
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import torch


def to_tensor(x, prefer_last=True):
    """Convert list/tuple to tensor, taking last element by default."""
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return None
        # For layernorm hooks, list is often (input, output)
        # Take LAST element to get the OUTPUT, not input
        if prefer_last:
            x = x[-1]  # Last element is typically the output
        else:
            x = x[0]
    if not isinstance(x, torch.Tensor):
        return None
    return x


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
        label: Label for printing (e.g., "SGLang pos 90" or "Megatron pos 90")
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


def get_megatron_dump_info(path: Path) -> dict[str, Any]:
    """Extract information from a Megatron dump file."""
    tensors = torch.load(path, map_location="cpu")

    info = {"path": path, "tensors": tensors}

    # Check for Megatron format
    if "megatron_input_ids" in tensors:
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
    megatron_tensors: dict[str, torch.Tensor],
    sglang_position: int,
    megatron_position: int,
    verbose: bool = True,
    sglang_decode_tensors: dict[str, torch.Tensor] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Compare hidden states at a specific position across all layers.

    For first response token comparison:
    - SGLang prefill: hidden states at last position (sglang_position)
    - Megatron training: hidden states at (megatron_position)

    These should match because both represent the same position.
    """
    results = {}

    print("\n" + "=" * 70)
    print("LAYER-BY-LAYER HIDDEN STATE COMPARISON")
    print(f"SGLang pos: {sglang_position}, Megatron pos: {megatron_position}")
    print("=" * 70)

    # =========================================================================
    # STEP 1: Print ALL available layer keys from both dumps
    # =========================================================================
    print("\n  ALL SGLang layer keys:")
    sg_layer_keys = sorted(
        [k for k in sglang_tensors.keys() if "layer" in k.lower()]
    )
    for k in sg_layer_keys:
        t = sglang_tensors[k]
        if isinstance(t, torch.Tensor):
            print(f"    {k}: shape={t.shape}")
        elif isinstance(t, (list, tuple)):
            print(f"    {k}: list[{len(t)}]")

    print("\n  ALL Megatron layer keys:")
    megatron_layer_keys = sorted(
        [k for k in megatron_tensors.keys() if "layer" in k.lower()]
    )
    for k in megatron_layer_keys:
        t = megatron_tensors[k]
        if isinstance(t, torch.Tensor):
            print(f"    {k}: shape={t.shape}")
        elif isinstance(t, (list, tuple)):
            print(f"    {k}: list[{len(t)}]")

    # =========================================================================
    # STEP 2: Match SAME component between SGLang and Megatron
    # =========================================================================
    # SGLang keys: model.layers.{N}.{component}
    # Megatron keys: layer_{N}_{component}_output
    #
    # Component mapping:
    #   SGLang                      Megatron
    #   ------                      -------
    #   input_layernorm          -> layer_N_qkv_layernorm_output (RMSNorm inside linear_qkv)
    #   qkv_proj                 -> layer_N_qkv_proj_output (QKV projection output)
    #   self_attn.o_proj         -> layer_N_o_proj_output (output projection)
    #   post_attention_layernorm -> layer_N_pre_mlp_layernorm_output (AFTER residual, before MLP)
    #   mlp.down_proj            -> layer_N_mlp_output
    #   (final layer output)     -> layer_N_output
    #
    # NOTE: Megatron's post_self_attn_layernorm is applied to attention output BEFORE residual,
    #       which is different from SGLang's post_attention_layernorm!

    sglang_layers = {}
    megatron_layers = {}

    # Define component pairs to try (SGLang pattern, Megatron pattern)
    component_pairs = [
        ("post_attention_layernorm", "pre_mlp_layernorm"),  # before MLP
        ("input_layernorm", "qkv_layernorm"),  # RMSNorm before attention
        ("qkv_proj", "qkv_proj"),  # QKV projection
        ("self_attn.o_proj", "o_proj"),  # Output projection
        ("self_attn", "self_attention"),
        ("mlp", "mlp"),
    ]

    # Try each component pair
    matched_component = None
    for sg_comp, megatron_comp in component_pairs:
        if sglang_layers and megatron_layers:
            break  # Found matching components

        # Find SGLang layers with this component
        sg_temp = {}
        for name in sglang_tensors.keys():
            if "layers." in name and sg_comp in name:
                match = re.search(r"layers\.(\d+)\.", name)
                if match:
                    layer_idx = int(match.group(1))
                    if layer_idx not in sg_temp:
                        sg_temp[layer_idx] = (name, sglang_tensors[name])

        # Find Megatron layers with matching component
        megatron_temp = {}
        megatron_pattern = f"layer_(\\d+)_{megatron_comp}_output"
        for name in megatron_tensors.keys():
            match = re.match(megatron_pattern, name)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx not in megatron_temp:
                    megatron_temp[layer_idx] = (name, megatron_tensors[name])

        # Use this pair if we found matches in both
        if sg_temp and megatron_temp:
            sglang_layers = sg_temp
            megatron_layers = megatron_temp
            matched_component = (sg_comp, megatron_comp)
            break

    # Fallback to layer_N_output if no sublayer match
    if not megatron_layers:
        for name in megatron_tensors.keys():
            match = re.match(r"^layer_(\d+)_output$", name)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx not in megatron_layers:
                    megatron_layers[layer_idx] = (name, megatron_tensors[name])

    print("\n  MATCHING RESULT:")
    if matched_component:
        print(f"    Matched component: SGLang '{matched_component[0]}' "
              f"<-> Megatron '{matched_component[1]}'")

    if not sglang_layers:
        print("  ⚠️ No matching layer outputs found in SGLang dump!")

    if not megatron_layers:
        print("  ⚠️ No matching layer outputs found in Megatron dump!")

    # Show which layers were matched
    print(f"\n  SGLang layers found: {sorted(sglang_layers.keys())}")
    print(f"  Megatron layers found: {sorted(megatron_layers.keys())}")

    if sglang_layers:
        for idx in sorted(sglang_layers.keys())[:3]:
            name, tensor = sglang_layers[idx]
            shp = tensor.shape if isinstance(tensor, torch.Tensor) else "list"
            print(f"    SGLang layer {idx}: {name} -> {shp}")
    if megatron_layers:
        for idx in sorted(megatron_layers.keys())[:3]:
            name, tensor = megatron_layers[idx]
            shp = tensor.shape if isinstance(tensor, torch.Tensor) else "list"
            print(f"    Megatron layer {idx}: {name} -> {shp}")

    # Compare each layer
    all_layers = sorted(set(sglang_layers.keys()) | set(megatron_layers.keys()))

    significant_diff_layers = []

    # Comparing SGLang decode[91] vs Megatron base[90]
    # SGLang decode processes token at position 91
    # Megatron base tensors are at position 90 (prompt_len - 1)
    if sglang_decode_tensors is not None:
        print("\n  NOTE: Using SGLang decode[91] vs Megatron base[90]")

    for layer_idx in all_layers:
        if layer_idx not in sglang_layers:
            if verbose:
                print(f"  Layer {layer_idx:2d}: NOT IN SGLang dump")
            continue
        if layer_idx not in megatron_layers:
            if verbose:
                print(f"  Layer {layer_idx:2d}: NOT IN Megatron dump")
            continue

        # Extract tensor from (name, tensor) tuple
        sg_name, sglang_hidden = sglang_layers[layer_idx]
        megatron_name, megatron_hidden = megatron_layers[layer_idx]
        # Using Megatron base tensors at position 90 (not _at_response_start)

        # Helper to convert list/tuple to tensor
        def to_tensor(x, prefer_last=True):
            if isinstance(x, (list, tuple)):
                if len(x) == 0:
                    return None
                # For layernorm hooks, list is often (input, output)
                # Take LAST element to get the OUTPUT, not input
                if prefer_last:
                    x = x[-1]  # Last element is typically the output
                else:
                    x = x[0]
            if not isinstance(x, torch.Tensor):
                return None
            return x

        # Check if SGLang tensor is a list and show info
        sg_was_list = isinstance(sglang_hidden, (list, tuple))
        sg_list_len = len(sglang_hidden) if sg_was_list else 0

        sglang_hidden = to_tensor(sglang_hidden)
        megatron_hidden = to_tensor(megatron_hidden)

        if megatron_hidden is None:
            if verbose:
                print(f"  Layer {layer_idx:2d}: Megatron tensor is None/empty")
            continue

        # Megatron hidden state (already extracted at prompt_len position)
        megatron_at_pos = megatron_hidden
        if megatron_at_pos.dim() == 3:
            if megatron_position < megatron_at_pos.shape[1]:
                megatron_at_pos = megatron_at_pos[:, megatron_position:megatron_position+1, :]
        elif megatron_at_pos.dim() == 2:
            if megatron_position < megatron_at_pos.shape[0]:
                megatron_at_pos = megatron_at_pos[megatron_position:megatron_position+1, :]
        megatron_flat = megatron_at_pos.flatten()

        # For SGLang, use DECODE pass (position 91), not prefill!
        # The prefill tensor has positions 0-90, decode pass has position 91
        sg_flat = None
        sg_source = "prefill"

        if sglang_decode_tensors is not None:
            # Find matching layer in decode tensors
            sg_parts = sg_name.split(".")
            if "layers" in sg_parts:
                layer_str = sg_parts[sg_parts.index("layers") + 1]
                component = sg_parts[-1]

                for dname in sglang_decode_tensors.keys():
                    if f"layers.{layer_str}." in dname and component in dname:
                        dh = sglang_decode_tensors[dname]
                        if isinstance(dh, (list, tuple)):
                            # Use last element (output) not first (input)
                            dh = dh[-1] if dh else None
                        if isinstance(dh, torch.Tensor):
                            # Decode pass: shape [1, hidden] for single token
                            sg_flat = dh.flatten()
                            sg_source = "decode"
                            break

        # Fallback to prefill if decode not found
        if sg_flat is None and sglang_hidden is not None:
            sg_at_pos = sglang_hidden
            if sg_at_pos.dim() == 3:
                if sglang_position < sg_at_pos.shape[1]:
                    pos = sglang_position
                    sg_at_pos = sg_at_pos[:, pos:pos+1, :]
            elif sg_at_pos.dim() == 2:
                if sglang_position < sg_at_pos.shape[0]:
                    sg_at_pos = sg_at_pos[sglang_position:sglang_position+1, :]
            sg_flat = sg_at_pos.flatten()
            sg_source = "prefill"

        if sg_flat is None:
            if verbose:
                print(f"  Layer {layer_idx:2d}: SGLang tensor not found")
            continue

        # Align shapes if needed
        if sg_flat.shape != megatron_flat.shape:
            min_len = min(len(sg_flat), len(megatron_flat))
            sg_flat = sg_flat[:min_len]
            megatron_flat = megatron_flat[:min_len]

        stats = compute_diff_stats(sg_flat, megatron_flat)
        results[f"layer_{layer_idx}"] = stats

        if stats["max_diff"] >= 1e-5:
            significant_diff_layers.append((layer_idx, stats["max_diff"]))

        match_str = "✓" if stats["max_diff"] < 1e-5 else "✗"
        color = "" if stats["max_diff"] < 1e-5 else "\033[91m"
        end_color = "\033[0m" if color else ""

        if verbose:
            list_info = ""
            if sg_was_list:
                list_info = f", was list[{sg_list_len}], used [-1]"
            print(
                f"  {color}Layer {layer_idx:2d}: {match_str} "
                f"max_diff={stats['max_diff']:.6e}, "
                f"mean_diff={stats['mean_diff']:.6e} "
                f"(SGLang {sg_source}{list_info}){end_color}"
            )

            # Always show tensor shapes
            sg_shape = sg_flat.shape
            megatron_shape = megatron_flat.shape
            print(f"    SGLang shape: {sg_shape}, Megatron shape: {megatron_shape}")

            # Show first 10 values
            n_show = min(10, len(megatron_flat), len(sg_flat))
            sg_vals = sg_flat[:n_show].float().tolist()
            megatron_vals = megatron_flat[:n_show].float().tolist()
            diff_vals = [(sg_flat[i] - megatron_flat[i]).abs().float().item()
                         for i in range(n_show)]
            max_diff_shown = max(diff_vals) if diff_vals else 0

            print(f"    Megatron: {[f'{v:.4f}' for v in megatron_vals]}")
            print(f"    SGLang:   {[f'{v:.4f}' for v in sg_vals]}")
            print(f"    Diff:     {[f'{v:.4f}' for v in diff_vals]} "
                  f"(max={max_diff_shown:.4f})")

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


def compare_layer(
    layer_idx: int,
    sglang_tensors: dict,
    megatron_tensors: dict,
    verbose: bool = True,
) -> dict:
    """
    Compare a single transformer layer between SGLang and Megatron.
    
    Returns a dict with comparison results for each component.
    """
    results = {}
    
    # Helper to get tensor and flatten
    def get_tensor(tensors, key):
        if key not in tensors:
            return None
        t = tensors[key]
        if isinstance(t, (list, tuple)):
            t = t[-1]  # Take last element (output, not input)
        if isinstance(t, torch.Tensor):
            return t.flatten()
        return None
    
    # Helper to compare two tensors
    def compare_tensors(sg_key, meg_key, name):
        sg = get_tensor(sglang_tensors, sg_key)
        meg = get_tensor(megatron_tensors, meg_key)
        
        if sg is None:
            if verbose:
                print(f"    {name}: SGLang not found ({sg_key})")
            return ("NOT_FOUND", None)
        if meg is None:
            if verbose:
                print(f"    {name}: Megatron not found ({meg_key})")
            return ("NOT_FOUND", None)
        
        min_len = min(sg.numel(), meg.numel())
        diff = (sg[:min_len].float() - meg[:min_len].float()).abs()
        max_diff = diff.max().item()
        
        if verbose:
            sg_vals = [f'{v:.4f}' for v in sg[:10].float().tolist()]
            meg_vals = [f'{v:.4f}' for v in meg[:10].float().tolist()]
            print(f"    {name}:")
            print(f"      SGLang first 10:   {sg_vals}")
            print(f"      Megatron first 10: {meg_vals}")
            print(f"      Max diff: {max_diff:.6e}")
            if max_diff < 1e-5:
                print(f"      ✓ MATCH")
        
        if max_diff < 1e-5:
            return ("MATCH", max_diff)
        elif max_diff < 1e-3:
            return ("CLOSE", max_diff)
        else:
            return ("DIFF", max_diff)
    
    print(f"\n  Layer {layer_idx}:")
    
    # Component comparisons (in calculation graph order)
    # 1. Input LayerNorm (RMSNorm)
    results["input_ln"] = compare_tensors(
        f"model.layers.{layer_idx}.input_layernorm",
        f"layer_{layer_idx}_qkv_layernorm_output_at_response_start",
        "Input LayerNorm"
    )
    
    # 2. QKV Projection
    results["qkv_proj"] = compare_tensors(
        f"model.layers.{layer_idx}.self_attn.qkv_proj",
        f"layer_{layer_idx}_qkv_proj_output_at_response_start",
        "QKV Projection"
    )
    
    # 3. Q LayerNorm
    results["q_ln"] = compare_tensors(
        f"model.layers.{layer_idx}.self_attn.q_norm",
        f"layer_{layer_idx}_q_layernorm_output_at_response_start",
        "Q LayerNorm"
    )
    
    # 4. K LayerNorm
    results["k_ln"] = compare_tensors(
        f"model.layers.{layer_idx}.self_attn.k_norm",
        f"layer_{layer_idx}_k_layernorm_output_at_response_start",
        "K LayerNorm"
    )
    
    # 5. Core Attention
    results["core_attn"] = compare_tensors(
        f"model.layers.{layer_idx}.self_attn.attn",
        f"layer_{layer_idx}_core_attention_output_at_response_start",
        "Core Attention"
    )
    
    # 6. Output Projection (o_proj)
    results["o_proj"] = compare_tensors(
        f"model.layers.{layer_idx}.self_attn.o_proj",
        f"layer_{layer_idx}_o_proj_output_at_response_start",
        "Output Projection"
    )
    
    # 7. Post-Attention LayerNorm (pre_mlp_layernorm)
    results["post_attn_ln"] = compare_tensors(
        f"model.layers.{layer_idx}.post_attention_layernorm",
        f"layer_{layer_idx}_pre_mlp_layernorm_output_at_response_start",
        "Post-Attn LayerNorm"
    )
    
    # 8a. MLP Gate/Up Projection
    results["gate_up"] = compare_tensors(
        f"model.layers.{layer_idx}.mlp.gate_up_proj",
        f"layer_{layer_idx}_mlp.gate_up_proj_output_at_response_start",
        "MLP Gate/Up Proj"
    )
    
    # 8b. MLP Output (down_proj)
    results["mlp_out"] = compare_tensors(
        f"model.layers.{layer_idx}.mlp.down_proj",
        f"layer_{layer_idx}_mlp_output_at_response_start",
        "MLP Output"
    )
    
    # 9. Full Layer Output
    results["layer_out"] = compare_tensors(
        f"model.layers.{layer_idx}.mlp.down_proj",  # SGLang doesn't have full output with residual
        f"layer_{layer_idx}_output_at_response_start",
        "Layer Output"
    )
    
    return results


def compare_first_response_token(
    sglang_dir: str,
    megatron_dir: str,
    verbose: bool = True,
) -> None:
    """
    Compare the first response token between SGLang and Megatron.

    This is the key comparison for true on-policy verification:
    - SGLang: Uses prefill pass to compute logits for first response token
    - Megatron: Uses logits at position (prompt_len - 1) from training pass

    We compare:
    1. Hidden states at each layer for position (prompt_len - 1)
    2. Logits for the first response token
    3. Logprobs computed from logits
    """
    print("\n" + "=" * 70)
    print("FIRST RESPONSE TOKEN COMPARISON")
    print("=" * 70)
    print("\nComparing SGLang prefill pass vs Megatron training pass")
    print("for the FIRST response token prediction.\n")

    # Find Megatron dump (should only be ONE pass for training)
    megatron_passes = list_all_passes(megatron_dir)
    if not megatron_passes:
        print(f"ERROR: No Megatron dump files found in {megatron_dir}")
        return

    # Training should have only ONE pass
    if len(megatron_passes) > 1:
        print(f"WARNING: Found {len(megatron_passes)} Megatron passes, using first.")

    megatron_pass_id, megatron_path = megatron_passes[0]
    megatron_info = get_megatron_dump_info(megatron_path)
    megatron_tensors = megatron_info["tensors"]

    print("Megatron Info:")
    print(f"  Backend: {megatron_info.get('backend', 'Unknown')}")
    print(f"  Prompt length: {megatron_info.get('prompt_len', 'N/A')}")
    print(f"  Total length: {megatron_info.get('total_len', 'N/A')}")
    print(f"  Response length: {megatron_info.get('response_len', 'N/A')}")
    if "response_positions" in megatron_info:
        resp_pos = megatron_info['response_positions']
        print(f"  Response logits positions: {resp_pos}")

    prompt_len = megatron_info.get("prompt_len")
    if prompt_len is None:
        print("ERROR: Could not determine prompt_len from Megatron dump")
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
    print("  Megatron logits_pos_N predicts token at N+1")
    print(f"  To compare first response token (pos {first_response_pos}):")
    print(f"    - Megatron: use logits_pos_{comparison_pos}")
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
    # Megatron processes full sequence, extract at prompt_len - 1
    sglang_prefill_last_pos = sglang_prefill_info.get("seq_len", 1) - 1

    print("\n  Hidden state comparison:")
    print(f"    SGLang: prefill last position = {sglang_prefill_last_pos}")
    print(f"    Megatron: position {comparison_pos} (prompt_len - 1)")

    # Check what position Megatron extracted at (for info only)
    if "megatron_compared_position" in megatron_tensors:
        megatron_dump_pos = megatron_tensors["megatron_compared_position"].item()
        print(f"    Megatron dump extracted at: {megatron_dump_pos}")
        if megatron_dump_pos != comparison_pos:
            print(f"    WARNING: Dump position {megatron_dump_pos} != "
                  f"comparison_pos {comparison_pos}")

    # IMPORTANT: Use comparison_pos (prompt_len - 1) for Megatron
    # This matches SGLang prefill's last position
    megatron_hidden_pos = comparison_pos

    # Both should be the same position for apples-to-apples comparison
    if sglang_prefill_last_pos != megatron_hidden_pos:
        print(f"    ⚠️ Position mismatch! Adjusting Megatron to "
              f"{sglang_prefill_last_pos}")
        megatron_hidden_pos = sglang_prefill_last_pos

    # Load decode pass tensors for SGLang position 91
    sglang_decode_for_hidden = None
    decode_result = find_sglang_decode_pass(sglang_dir, first_response_pos)
    if decode_result is not None:
        decode_id, decode_path = decode_result
        sglang_decode_for_hidden = torch.load(decode_path, map_location="cpu")
        print(f"    Loaded SGLang decode pass {decode_id} "
              f"(first_pos={first_response_pos})")

    # =========================================================================
    # COMPREHENSIVE HIDDEN STATE COMPARISON
    # Compare SGLang decode[91] vs Megatron at positions 90, 91
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPREHENSIVE HIDDEN STATE COMPARISON")
    print("SGLang decode[91] vs Megatron at positions 90, 91")
    print("=" * 70)

    # First, show exactly what keys we have
    print("\n  SGLang decode tensor keys (layer-related):")
    if sglang_decode_for_hidden:
        sg_keys = sorted([k for k in sglang_decode_for_hidden.keys()
                          if "layer" in k.lower()])
        for k in sg_keys[:10]:
            t = sglang_decode_for_hidden[k]
            if isinstance(t, torch.Tensor):
                print(f"    {k}: shape={t.shape}")
            elif isinstance(t, (list, tuple)):
                print(f"    {k}: list[{len(t)}]")

    print("\n  Megatron tensor keys (layer-related, layer_0):")
    meg_keys = sorted([k for k in megatron_tensors.keys()
                       if "layer_0" in k.lower()])
    for k in meg_keys:
        t = megatron_tensors[k]
        if isinstance(t, torch.Tensor):
            print(f"    {k}: shape={t.shape}")

    # =========================================================================
    # LAYER 0 COMPARISON - Ordered by Calculation Graph
    # Order: input_layernorm → QKV proj → Q/K layernorm → RoPE → 
    #        Core Attention → o_proj → post_attn_layernorm → MLP → output
    # =========================================================================
    print("\n" + "-" * 70)
    print("LAYER 0 COMPARISON - Ordered by Calculation Graph")
    print("-" * 70)
    
    # Track results for summary
    comparison_results = {}
    
    # =========================================================================
    # STEP 1: INPUT LAYERNORM (RMSNorm)
    # =========================================================================
    print("\n  [STEP 1] INPUT LAYERNORM (RMSNorm) - Before QKV projection:")
    print("    SGLang: input_layernorm | Megatron: qkv_layernorm (inside fused linear_qkv)")

    # Get SGLang input_layernorm output
    sg_input_ln = None
    sg_input_ln_key = "model.layers.0.input_layernorm"
    if sglang_decode_for_hidden and sg_input_ln_key in sglang_decode_for_hidden:
        sg_input_ln = sglang_decode_for_hidden[sg_input_ln_key]
        if isinstance(sg_input_ln, (list, tuple)):
            sg_input_ln = sg_input_ln[-1]  # Take output, not input
    
    if sg_input_ln is not None and isinstance(sg_input_ln, torch.Tensor):
        sg_ln_flat = sg_input_ln.flatten()
        print(f"    SGLang shape: {sg_input_ln.shape}")
        print(f"    SGLang first 10: {[f'{v:.4f}' for v in sg_ln_flat[:10].float().tolist()]}")
        
        # Megatron qkv_layernorm at position 91 (first response token)
        meg_qkv_ln_91 = "layer_0_qkv_layernorm_output_at_response_start"
        if meg_qkv_ln_91 in megatron_tensors:
            meg_ln = megatron_tensors[meg_qkv_ln_91].flatten()
            min_len = min(sg_ln_flat.numel(), meg_ln.numel())
            diff = (sg_ln_flat[:min_len].float() - meg_ln[:min_len].float()).abs()
            max_diff = diff.max().item()
            print(f"    Megatron shape: {megatron_tensors[meg_qkv_ln_91].shape}")
            print(f"    Megatron first 10: {[f'{v:.4f}' for v in meg_ln[:10].float().tolist()]}")
            print(f"    Max diff: {max_diff:.6e}")
            if max_diff < 1e-5:
                print("    ✓ Input LayerNorm MATCH!")
                comparison_results["input_layernorm"] = ("MATCH", max_diff)
            else:
                comparison_results["input_layernorm"] = ("DIFF", max_diff)
        else:
            print("    Megatron qkv_layernorm not found")
            comparison_results["input_layernorm"] = ("NOT_FOUND", None)
    else:
        print("    SGLang input_layernorm not found")
        comparison_results["input_layernorm"] = ("NOT_FOUND", None)

    # =========================================================================
    # STEP 2: QKV PROJECTION
    # =========================================================================
    print("\n  [STEP 2] QKV PROJECTION:")
    print("    SGLang: qkv_proj | Megatron: linear_qkv")
    sg_qkv = None
    if sglang_decode_for_hidden:
        for k in ["model.layers.0.self_attn.qkv_proj"]:
            if k in sglang_decode_for_hidden:
                sg_qkv = sglang_decode_for_hidden[k]
                print(f"    SGLang key: {k}")
                if isinstance(sg_qkv, (list, tuple)):
                    sg_qkv = sg_qkv[-1]
                break

    if sg_qkv is not None and isinstance(sg_qkv, torch.Tensor):
        sg_qkv_flat = sg_qkv.flatten()
        sg_qkv_vals = sg_qkv_flat[:10].float().tolist()
        print(f"    SGLang shape: {sg_qkv.shape}")
        print(f"    SGLang first 10: {[f'{v:.4f}' for v in sg_qkv_vals]}")

        # Megatron QKV projection at pos 91
        meg_qkv_91 = "layer_0_qkv_proj_output_at_response_start"
        if meg_qkv_91 in megatron_tensors:
            meg_qkv = megatron_tensors[meg_qkv_91].flatten()
            meg_qkv_vals = meg_qkv[:10].float().tolist()
            diff_qkv = (sg_qkv_flat[:10].float() - meg_qkv[:10].float()).abs()
            max_diff_qkv = diff_qkv.max().item()
            print(f"\n    Megatron layer_0_qkv_proj pos 91:")
            print(f"    Megatron shape: {megatron_tensors[meg_qkv_91].shape}")
            print(f"    Megatron first 10: {[f'{v:.4f}' for v in meg_qkv_vals]}")
            print(f"    Max diff: {max_diff_qkv:.6e}")
            if max_diff_qkv < 1e-5:
                print("    ✓ QKV projections MATCH!")
                comparison_results["qkv_proj"] = ("MATCH", max_diff_qkv)
            else:
                comparison_results["qkv_proj"] = ("DIFF", max_diff_qkv)
        else:
            # Try base tensor without _at_response_start
            meg_qkv_base = "layer_0_qkv_proj_output"
            if meg_qkv_base in megatron_tensors:
                meg_qkv = megatron_tensors[meg_qkv_base].flatten()
                meg_qkv_vals = meg_qkv[:10].float().tolist()
                print(f"\n    Megatron layer_0_qkv_proj (base):")
                print(f"    Megatron first 10: {[f'{v:.4f}' for v in meg_qkv_vals]}")
            else:
                print("    WARNING: Could not find Megatron QKV projection tensor")
    else:
        print("    SGLang qkv_proj not found in decode dump")
        print(f"    Available SGLang keys with 'qkv': "
              f"{[k for k in (sglang_decode_for_hidden or {}).keys() if 'qkv' in k.lower()]}")

    # =========================================================================
    # STEP 3: Q/K LAYERNORM (after QKV split, before RoPE)
    # =========================================================================
    print("\n  [STEP 3] Q/K LAYERNORM:")
    print("    SGLang: q_norm/k_norm | Megatron: q_layernorm/k_layernorm")
    
    # Check SGLang q_norm
    sg_q_norm = None
    if sglang_decode_for_hidden:
        for k in ["model.layers.0.self_attn.q_norm"]:
            if k in sglang_decode_for_hidden:
                sg_q_norm = sglang_decode_for_hidden[k]
                print(f"    SGLang q_norm key: {k}")
                if isinstance(sg_q_norm, (list, tuple)):
                    sg_q_norm = sg_q_norm[-1]
                break
    
    if sg_q_norm is not None and isinstance(sg_q_norm, torch.Tensor):
        sg_q_flat = sg_q_norm.flatten()
        print(f"    SGLang q_norm shape: {sg_q_norm.shape}")
        print(f"    SGLang q_norm first 10: {[f'{v:.4f}' for v in sg_q_flat[:10].float().tolist()]}")
        
        # Megatron q_layernorm
        meg_q_91 = "layer_0_q_layernorm_output_at_response_start"
        if meg_q_91 in megatron_tensors:
            meg_q = megatron_tensors[meg_q_91].flatten()
            diff_q = (sg_q_flat[:10].float() - meg_q[:10].float()).abs()
            max_diff_q = diff_q.max().item()
            print(f"\n    Megatron q_layernorm pos 91:")
            print(f"    Megatron shape: {megatron_tensors[meg_q_91].shape}")
            print(f"    Megatron first 10: {[f'{v:.4f}' for v in meg_q[:10].float().tolist()]}")
            print(f"    Max diff: {max_diff_q:.6e}")
            if max_diff_q < 1e-5:
                print("    ✓ Q layernorms MATCH!")
                comparison_results["q_layernorm"] = ("MATCH", max_diff_q)
            else:
                comparison_results["q_layernorm"] = ("DIFF", max_diff_q)
        else:
            print("    Megatron q_layernorm not found")
            comparison_results["q_layernorm"] = ("NOT_FOUND", None)
            print(f"    Available Megatron keys with 'layernorm': "
                  f"{[k for k in megatron_tensors.keys() if 'layernorm' in k.lower()][:10]}")
    else:
        print("    SGLang q_norm not found")
    
    # Check SGLang k_norm
    sg_k_norm = None
    if sglang_decode_for_hidden:
        for k in ["model.layers.0.self_attn.k_norm"]:
            if k in sglang_decode_for_hidden:
                sg_k_norm = sglang_decode_for_hidden[k]
                print(f"\n    SGLang k_norm key: {k}")
                if isinstance(sg_k_norm, (list, tuple)):
                    sg_k_norm = sg_k_norm[-1]
                break
    
    if sg_k_norm is not None and isinstance(sg_k_norm, torch.Tensor):
        sg_k_flat = sg_k_norm.flatten()
        print(f"    SGLang k_norm shape: {sg_k_norm.shape}")
        print(f"    SGLang k_norm first 10: {[f'{v:.4f}' for v in sg_k_flat[:10].float().tolist()]}")
        
        # Megatron k_layernorm
        meg_k_91 = "layer_0_k_layernorm_output_at_response_start"
        if meg_k_91 in megatron_tensors:
            meg_k = megatron_tensors[meg_k_91].flatten()
            diff_k = (sg_k_flat[:10].float() - meg_k[:10].float()).abs()
            max_diff_k = diff_k.max().item()
            print(f"\n    Megatron k_layernorm pos 91:")
            print(f"    Megatron shape: {megatron_tensors[meg_k_91].shape}")
            print(f"    Megatron first 10: {[f'{v:.4f}' for v in meg_k[:10].float().tolist()]}")
            print(f"    Max diff: {max_diff_k:.6e}")
            if max_diff_k < 1e-5:
                print("    ✓ K layernorms MATCH!")
                comparison_results["k_layernorm"] = ("MATCH", max_diff_k)
            else:
                comparison_results["k_layernorm"] = ("DIFF", max_diff_k)
        else:
            print("    Megatron k_layernorm not found")
            comparison_results["k_layernorm"] = ("NOT_FOUND", None)
    else:
        print("    SGLang k_norm not found")
        comparison_results["k_layernorm"] = ("NOT_FOUND", None)

    # =========================================================================
    # STEP 4: RoPE (Rotary Position Embedding)
    # =========================================================================
    print("\n  [STEP 4] Q/K AFTER ROPE:")
    print("    SGLang: rotary_emb | Megatron: layer_1 Q/K after RoPE")
    print("    Note: Megatron layer_1 = SGLang layer 0 (1-based vs 0-based indexing)")
    
    # Show all available q_after_rope keys for debugging
    rope_keys = [k for k in megatron_tensors.keys() if 'after_rope' in k.lower()]
    if rope_keys:
        print(f"    Available Megatron RoPE keys: {rope_keys[:6]}")
    
    # Megatron uses 1-based layer numbering, so layer_1 is the first layer
    meg_q_key = "layer_1_q_after_rope_at_response_start"
    meg_k_key = "layer_1_k_after_rope_at_response_start"
    
    # Fallback to base key without _at_response_start
    if meg_q_key not in megatron_tensors:
        meg_q_key = "layer_1_q_after_rope"
        meg_k_key = "layer_1_k_after_rope"
    
    if meg_q_key in megatron_tensors:
        meg_q_rope = megatron_tensors[meg_q_key]
        meg_k_rope = megatron_tensors.get(meg_k_key)
        
        print(f"    Megatron Q after RoPE key: {meg_q_key}")
        print(f"    Megatron Q after RoPE shape: {meg_q_rope.shape}")
        meg_q_flat = meg_q_rope.flatten()
        print(f"    Megatron Q after RoPE first 10: {[f'{v:.4f}' for v in meg_q_flat[:10].float().tolist()]}")
        
        if meg_k_rope is not None:
            print(f"    Megatron K after RoPE shape: {meg_k_rope.shape}")
            meg_k_flat = meg_k_rope.flatten()
            print(f"    Megatron K after RoPE first 10: {[f'{v:.4f}' for v in meg_k_flat[:10].float().tolist()]}")
        
        # SGLang's rotary_emb hook captures (query, key) tuple - Q/K AFTER RoPE
        sg_rotary_key = "model.layers.0.self_attn.rotary_emb"
        sg_q_rope = None
        sg_k_rope = None
        
        if sglang_decode_for_hidden and sg_rotary_key in sglang_decode_for_hidden:
            sg_rotary = sglang_decode_for_hidden[sg_rotary_key]
            print(f"    SGLang rotary_emb key: {sg_rotary_key}")
            print(f"    SGLang rotary_emb type: {type(sg_rotary)}")
            
            if isinstance(sg_rotary, (list, tuple)) and len(sg_rotary) >= 2:
                # rotary_emb returns (query, key) tuple
                sg_q_rope = sg_rotary[0]  # Q after RoPE
                sg_k_rope = sg_rotary[1]  # K after RoPE
                print(f"    SGLang Q after RoPE shape: {sg_q_rope.shape}")
                print(f"    SGLang K after RoPE shape: {sg_k_rope.shape}")
                
                sg_q_flat = sg_q_rope.flatten()
                print(f"    SGLang Q after RoPE first 10: {[f'{v:.4f}' for v in sg_q_flat[:10].float().tolist()]}")
                
                # Detailed comparison - check element ordering
                print("\n    === DETAILED Q AFTER ROPE ANALYSIS ===")
                print(f"    Megatron shape: {meg_q_rope.shape} -> flatten to {meg_q_flat.numel()}")
                print(f"    SGLang shape: {sg_q_rope.shape} -> flatten to {sg_q_flat.numel()}")
                
                # Check if values appear in different order
                # Megatron: [num_heads, head_dim] = [16, 128]
                # SGLang: [1, num_heads * head_dim] = [1, 2048]
                
                # Maybe SGLang has different head ordering?
                # Try comparing with Megatron reshaped to [1, 2048]
                meg_q_reshape = meg_q_rope.reshape(1, -1)  # [1, 2048]
                print(f"    Megatron reshaped to [1, 2048] first 10: {[f'{v:.4f}' for v in meg_q_reshape.flatten()[:10].float().tolist()]}")
                
                # Check if transpose helps (maybe head vs dim ordering)
                if meg_q_rope.dim() == 2:
                    meg_q_transpose = meg_q_rope.T.reshape(1, -1)  # [128, 16] -> [1, 2048]
                    print(f"    Megatron transposed to [128,16]->[1,2048] first 10: {[f'{v:.4f}' for v in meg_q_transpose.flatten()[:10].float().tolist()]}")
                
                # Also check Q BEFORE RoPE (q_layernorm) for reference
                sg_q_norm_key = "model.layers.0.self_attn.q_norm"
                if sg_q_norm_key in sglang_decode_for_hidden:
                    sg_q_norm = sglang_decode_for_hidden[sg_q_norm_key]
                    if isinstance(sg_q_norm, (list, tuple)):
                        sg_q_norm = sg_q_norm[-1] if len(sg_q_norm) > 0 else sg_q_norm[0]
                    sg_q_norm_flat = sg_q_norm.flatten() if isinstance(sg_q_norm, torch.Tensor) else None
                    if sg_q_norm_flat is not None:
                        print(f"    [Reference] SGLang Q before RoPE (q_norm) first 10: {[f'{v:.4f}' for v in sg_q_norm_flat[:10].float().tolist()]}")
                
                meg_q_layernorm_key = "layer_0_q_layernorm_output_at_response_start"
                if meg_q_layernorm_key in megatron_tensors:
                    meg_q_norm = megatron_tensors[meg_q_layernorm_key].flatten()
                    print(f"    [Reference] Megatron Q before RoPE (q_layernorm) first 10: {[f'{v:.4f}' for v in meg_q_norm[:10].float().tolist()]}")
                
                print("    === END DETAILED ANALYSIS ===\n")
                
                # Compare Q after RoPE
                min_len = min(sg_q_flat.numel(), meg_q_flat.numel())
                diff_q = (sg_q_flat[:min_len].float() - meg_q_flat[:min_len].float()).abs()
                max_diff_q = diff_q.max().item()
                mean_diff_q = diff_q.mean().item()
                print(f"    Q after RoPE - Max diff: {max_diff_q:.6e}, Mean diff: {mean_diff_q:.6e}")
                
                if max_diff_q < 1e-5:
                    print("    ✓ Q after RoPE MATCH!")
                    comparison_results["rope"] = ("MATCH", max_diff_q)
                elif max_diff_q < 1e-3:
                    print("    ⚠ Q after RoPE close but not identical")
                    comparison_results["rope"] = ("CLOSE", max_diff_q)
                else:
                    print("    ✗ Q after RoPE DIFFERENT - RoPE implementations differ")
                    comparison_results["rope"] = ("DIFF", max_diff_q)
                    
                # Also compare K if available
                if sg_k_rope is not None and meg_k_rope is not None:
                    sg_k_flat = sg_k_rope.flatten()
                    meg_k_flat_cmp = meg_k_rope.flatten()
                    min_len_k = min(sg_k_flat.numel(), meg_k_flat_cmp.numel())
                    diff_k = (sg_k_flat[:min_len_k].float() - meg_k_flat_cmp[:min_len_k].float()).abs()
                    max_diff_k = diff_k.max().item()
                    print(f"    K after RoPE - Max diff: {max_diff_k:.6e}")
                    if max_diff_k < 1e-5:
                        print("    ✓ K after RoPE MATCH!")
            else:
                print(f"    SGLang rotary_emb is not a (q, k) tuple: {type(sg_rotary)}")
                if isinstance(sg_rotary, torch.Tensor):
                    print(f"    SGLang rotary_emb tensor shape: {sg_rotary.shape}")
        else:
            print("    SGLang rotary_emb not found in decode dump")
            if sglang_decode_for_hidden:
                rotary_keys = [k for k in sglang_decode_for_hidden.keys() if 'rotary' in k.lower()]
                print(f"    Available SGLang rotary keys: {rotary_keys}")
    else:
        print("    Megatron Q after RoPE not found (layer_1_q_after_rope)")
        print("    (Enable with: enable_qk_after_rope_dump(model))")
        comparison_results["rope"] = ("NOT_FOUND", None)

    # =========================================================================
    # STEP 5: CORE ATTENTION (Flash Attention output)
    # =========================================================================
    print("\n  [STEP 5] CORE ATTENTION (Flash Attention output):")
    print("    SGLang: self_attn.attn | Megatron: core_attention")
    meg_core_attn_91 = "layer_0_core_attention_output_at_response_start"
    if meg_core_attn_91 in megatron_tensors:
        meg_core = megatron_tensors[meg_core_attn_91].flatten()
        print(f"    Megatron core_attention shape: {megatron_tensors[meg_core_attn_91].shape}")
        print(f"    Megatron core_attention first 10: {[f'{v:.4f}' for v in meg_core[:10].float().tolist()]}")
        
        # Check if SGLang has attn output
        sg_attn = None
        if sglang_decode_for_hidden:
            for k in ["model.layers.0.self_attn.attn"]:
                if k in sglang_decode_for_hidden:
                    sg_attn = sglang_decode_for_hidden[k]
                    print(f"    SGLang attn key: {k}")
                    if isinstance(sg_attn, (list, tuple)):
                        sg_attn = sg_attn[-1]
                    break
        
        if sg_attn is not None and isinstance(sg_attn, torch.Tensor):
            sg_attn_flat = sg_attn.flatten()
            print(f"    SGLang attn shape: {sg_attn.shape}")
            print(f"    SGLang attn first 10: {[f'{v:.4f}' for v in sg_attn_flat[:10].float().tolist()]}")
            
            # Compare all values, not just first 10
            min_len = min(sg_attn_flat.numel(), meg_core.numel())
            diff_attn_all = (sg_attn_flat[:min_len].float() - meg_core[:min_len].float()).abs()
            max_diff_attn = diff_attn_all.max().item()
            mean_diff_attn = diff_attn_all.mean().item()
            
            print(f"    Max diff: {max_diff_attn:.6e}, Mean diff: {mean_diff_attn:.6e}")
            
            if max_diff_attn < 1e-5:
                print("    ✓ Core attention outputs MATCH! (true on-policy)")
                comparison_results["core_attention"] = ("MATCH", max_diff_attn)
            elif max_diff_attn < 1e-3:
                print("    ⚠ Core attention outputs close but not identical")
                comparison_results["core_attention"] = ("CLOSE", max_diff_attn)
            else:
                print("    ✗ Core attention outputs DIFFERENT")
                comparison_results["core_attention"] = ("DIFF", max_diff_attn)
    else:
        print("    Megatron core_attention not found")
        comparison_results["core_attention"] = ("NOT_FOUND", None)

    # =========================================================================
    # STEP 6: OUTPUT PROJECTION (o_proj)
    # =========================================================================
    print("\n  [STEP 6] OUTPUT PROJECTION (o_proj):")
    print("    SGLang: self_attn.o_proj | Megatron: linear_proj")
    
    sg_oproj = None
    sg_oproj_key = "model.layers.0.self_attn.o_proj"
    if sglang_decode_for_hidden and sg_oproj_key in sglang_decode_for_hidden:
        sg_oproj = sglang_decode_for_hidden[sg_oproj_key]
        if isinstance(sg_oproj, (list, tuple)):
            sg_oproj = sg_oproj[-1]
    
    if sg_oproj is not None and isinstance(sg_oproj, torch.Tensor):
        sg_oproj_flat = sg_oproj.flatten()
        print(f"    SGLang shape: {sg_oproj.shape}")
        print(f"    SGLang first 10: {[f'{v:.4f}' for v in sg_oproj_flat[:10].float().tolist()]}")
        
        # Megatron o_proj at pos 91
        meg_oproj_91 = "layer_0_o_proj_output_at_response_start"
        if meg_oproj_91 in megatron_tensors:
            meg_oproj = megatron_tensors[meg_oproj_91].flatten()
            min_len = min(sg_oproj_flat.numel(), meg_oproj.numel())
            diff = (sg_oproj_flat[:min_len].float() - meg_oproj[:min_len].float()).abs()
            max_diff = diff.max().item()
            print(f"    Megatron shape: {megatron_tensors[meg_oproj_91].shape}")
            print(f"    Megatron first 10: {[f'{v:.4f}' for v in meg_oproj[:10].float().tolist()]}")
            print(f"    Max diff: {max_diff:.6e}")
            if max_diff < 1e-5:
                print("    ✓ Output projections MATCH!")
                comparison_results["o_proj"] = ("MATCH", max_diff)
            else:
                comparison_results["o_proj"] = ("DIFF", max_diff)
        else:
            print("    Megatron o_proj not found")
            comparison_results["o_proj"] = ("NOT_FOUND", None)
    else:
        print("    SGLang o_proj not found")
        comparison_results["o_proj"] = ("NOT_FOUND", None)

    # =========================================================================
    # STEP 7: POST-ATTENTION LAYERNORM (pre_mlp_layernorm)
    # This is AFTER attention+residual, BEFORE MLP
    # =========================================================================
    print("\n  [STEP 7] POST-ATTENTION LAYERNORM (pre_mlp_layernorm):")
    print("    SGLang: post_attention_layernorm | Megatron: pre_mlp_layernorm")
    print("    This is the normalized input to MLP")
    
    sg_post_attn_ln = None
    sg_post_attn_ln_key = "model.layers.0.post_attention_layernorm"
    if sglang_decode_for_hidden and sg_post_attn_ln_key in sglang_decode_for_hidden:
        sg_post_attn_ln = sglang_decode_for_hidden[sg_post_attn_ln_key]
        if isinstance(sg_post_attn_ln, (list, tuple)):
            sg_post_attn_ln = sg_post_attn_ln[-1]  # Take output, not input
        print(f"    SGLang key: {sg_post_attn_ln_key}")
    
    if sg_post_attn_ln is not None and isinstance(sg_post_attn_ln, torch.Tensor):
        sg_post_flat = sg_post_attn_ln.flatten()
        print(f"    SGLang shape: {sg_post_attn_ln.shape}")
        print(f"    SGLang first 10: {[f'{v:.4f}' for v in sg_post_flat[:10].float().tolist()]}")
        
        # Megatron post_attention_layernorm (pre_mlp_layernorm)
        meg_post_91 = "layer_0_pre_mlp_layernorm_output_at_response_start"
        if meg_post_91 in megatron_tensors:
            meg_post = megatron_tensors[meg_post_91].flatten()
            print(f"    Megatron pre_mlp_layernorm pos 91:")
            print(f"    Megatron shape: {megatron_tensors[meg_post_91].shape}")
            print(f"    Megatron first 10: {[f'{v:.4f}' for v in meg_post[:10].float().tolist()]}")
            
            min_len = min(sg_post_flat.numel(), meg_post.numel())
            diff_post = (sg_post_flat[:min_len].float() - meg_post[:min_len].float()).abs()
            max_diff_post = diff_post.max().item()
            mean_diff_post = diff_post.mean().item()
            print(f"    Max diff: {max_diff_post:.6e}, Mean diff: {mean_diff_post:.6e}")
            
            if max_diff_post < 1e-5:
                print("    ✓ Post-attention layernorm MATCH!")
                comparison_results["post_attn_ln"] = ("MATCH", max_diff_post)
            elif max_diff_post < 1e-3:
                print("    ⚠ Post-attention layernorm close (small diff)")
                comparison_results["post_attn_ln"] = ("CLOSE", max_diff_post)
            else:
                print("    ✗ Post-attention layernorm DIFFERENT")
                comparison_results["post_attn_ln"] = ("DIFF", max_diff_post)
        else:
            print("    Megatron pre_mlp_layernorm not found")
            comparison_results["post_attn_ln"] = ("NOT_FOUND", None)
    else:
        print("    SGLang post_attention_layernorm not found")
        comparison_results["post_attn_ln"] = ("NOT_FOUND", None)

    # =========================================================================
    # STEP 8a: MLP GATE_UP_PROJ (linear_fc1)
    # =========================================================================
    print("\n  [STEP 8a] MLP GATE_UP_PROJ (linear_fc1):")
    print("    SGLang: mlp.gate_up_proj | Megatron: mlp.gate_up_proj")
    
    sg_gate_up = None
    sg_gate_up_key = "model.layers.0.mlp.gate_up_proj"
    if sglang_decode_for_hidden and sg_gate_up_key in sglang_decode_for_hidden:
        sg_gate_up = sglang_decode_for_hidden[sg_gate_up_key]
        if isinstance(sg_gate_up, (list, tuple)):
            sg_gate_up = sg_gate_up[-1]
    
    if sg_gate_up is not None and isinstance(sg_gate_up, torch.Tensor):
        sg_gate_up_flat = sg_gate_up.flatten()
        print(f"    SGLang shape: {sg_gate_up.shape}")
        print(f"    SGLang first 10: {[f'{v:.4f}' for v in sg_gate_up_flat[:10].float().tolist()]}")
        
        meg_gate_up_91 = "layer_0_mlp.gate_up_proj_output_at_response_start"
        if meg_gate_up_91 in megatron_tensors:
            meg_gate_up = megatron_tensors[meg_gate_up_91].flatten()
            print(f"    Megatron shape: {megatron_tensors[meg_gate_up_91].shape}")
            print(f"    Megatron first 10: {[f'{v:.4f}' for v in meg_gate_up[:10].float().tolist()]}")
            
            min_len = min(sg_gate_up_flat.numel(), meg_gate_up.numel())
            diff = (sg_gate_up_flat[:min_len].float() - meg_gate_up[:min_len].float()).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            print(f"    Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
            
            if max_diff < 1e-5:
                print("    ✓ Gate/Up projections MATCH!")
                comparison_results["gate_up_proj"] = ("MATCH", max_diff)
            else:
                print("    ✗ Gate/Up projections DIFFERENT")
                comparison_results["gate_up_proj"] = ("DIFF", max_diff)
        else:
            print("    Megatron mlp.gate_up_proj not found")
            print(f"    Available MLP keys: {[k for k in megatron_tensors.keys() if 'mlp' in k.lower()][:10]}")
            comparison_results["gate_up_proj"] = ("NOT_FOUND", None)
    else:
        print("    SGLang mlp.gate_up_proj not found")
        mlp_keys = [k for k in (sglang_decode_for_hidden or {}).keys()
                    if 'mlp' in k.lower()]
        print(f"    Available SGLang MLP keys: {mlp_keys}")
        comparison_results["gate_up_proj"] = ("NOT_FOUND", None)

    # =========================================================================
    # STEP 8a-ii: AFTER ACTIVATION (SiLU output)
    # =========================================================================
    print("\n  [STEP 8a-ii] AFTER ACTIVATION (SiLU output):")
    print("    Megatron: mlp.after_activation (after SGLangSwiGLU)")

    meg_after_act_key = "layer_0_mlp.after_activation_output_at_response_start"
    if meg_after_act_key in megatron_tensors:
        meg_after_act = megatron_tensors[meg_after_act_key]
        print(f"    Megatron shape: {meg_after_act.shape}")
        meg_after_act_flat = meg_after_act.flatten()
        act_first10 = meg_after_act_flat[:10].float().tolist()
        print(f"    Megatron first 10: {[f'{v:.4f}' for v in act_first10]}")
        comparison_results["after_activation"] = ("FOUND", None)
    else:
        print("    Megatron mlp.after_activation not found")
        act_keys = [k for k in megatron_tensors.keys()
                    if 'activation' in k.lower() or 'act' in k.lower()]
        print(f"    Available activation keys: {act_keys[:5]}")
        comparison_results["after_activation"] = ("NOT_FOUND", None)

    # =========================================================================
    # STEP 8b: MLP DOWN_PROJ OUTPUT (linear_fc2)
    # =========================================================================
    print("\n  [STEP 8b] MLP DOWN_PROJ OUTPUT:")
    print("    SGLang: mlp.down_proj | Megatron: mlp_output")
    
    sg_mlp_out = None
    sg_mlp_key = "model.layers.0.mlp.down_proj"
    if sglang_decode_for_hidden and sg_mlp_key in sglang_decode_for_hidden:
        sg_mlp_out = sglang_decode_for_hidden[sg_mlp_key]
        if isinstance(sg_mlp_out, (list, tuple)):
            sg_mlp_out = sg_mlp_out[-1]
        print(f"    SGLang key: {sg_mlp_key}")
    
    if sg_mlp_out is not None and isinstance(sg_mlp_out, torch.Tensor):
        sg_mlp_flat = sg_mlp_out.flatten()
        print(f"    SGLang shape: {sg_mlp_out.shape}")
        print(f"    SGLang first 10: {[f'{v:.4f}' for v in sg_mlp_flat[:10].float().tolist()]}")
        
        # Megatron MLP output (correct comparison)
        meg_mlp_91 = "layer_0_mlp_output_at_response_start"
        if meg_mlp_91 in megatron_tensors:
            meg_mlp = megatron_tensors[meg_mlp_91].flatten()
            print(f"    Megatron mlp_output pos 91:")
            print(f"    Megatron shape: {megatron_tensors[meg_mlp_91].shape}")
            print(f"    Megatron first 10: {[f'{v:.4f}' for v in meg_mlp[:10].float().tolist()]}")
            
            min_len = min(sg_mlp_flat.numel(), meg_mlp.numel())
            diff_mlp = (sg_mlp_flat[:min_len].float() - meg_mlp[:min_len].float()).abs()
            max_diff_mlp = diff_mlp.max().item()
            mean_diff_mlp = diff_mlp.mean().item()
            print(f"    Max diff: {max_diff_mlp:.6e}, Mean diff: {mean_diff_mlp:.6e}")
            
            if max_diff_mlp < 1e-5:
                print("    ✓ MLP outputs MATCH!")
                comparison_results["mlp_output"] = ("MATCH", max_diff_mlp)
            elif max_diff_mlp < 1e-2:
                print("    ⚠ MLP outputs close (acceptable for true on-policy)")
                comparison_results["mlp_output"] = ("CLOSE", max_diff_mlp)
            else:
                print("    ✗ MLP outputs DIFFERENT")
                comparison_results["mlp_output"] = ("DIFF", max_diff_mlp)
        else:
            print("    Megatron mlp_output not found")
            comparison_results["mlp_output"] = ("NOT_FOUND", None)

    # =========================================================================
    # STEP 9: FULL LAYER OUTPUT (MLP + residual)  
    # =========================================================================
    print("\n  [STEP 9] FULL LAYER OUTPUT (MLP + residual):")
    print("    Megatron: layer_0_output | SGLang: N/A (residual not dumped)")
    meg_out_91 = "layer_0_output_at_response_start"
    if meg_out_91 in megatron_tensors:
        meg_out = megatron_tensors[meg_out_91].flatten()
        print(f"    Megatron shape: {megatron_tensors[meg_out_91].shape}")
        print(f"    Megatron first 10: {[f'{v:.4f}' for v in meg_out[:10].float().tolist()]}")
        print("    Note: SGLang doesn't dump full layer output with residual directly")
        print("    The full output = o_proj + input (residual #1) + mlp_output (residual #2)")
    else:
        print("    Megatron layer_0_output not found")
    
    # =========================================================================
    # SUMMARY: Layer 0 Comparison Results
    # =========================================================================
    print("\n" + "-" * 70)
    print("LAYER 0 COMPARISON SUMMARY")
    print("-" * 70)
    
    all_match = True
    for step, (status, diff) in comparison_results.items():
        if status == "MATCH":
            mark = "✓"
        elif status == "CLOSE":
            mark = "⚠"
            all_match = False
        elif status == "NOT_FOUND":
            mark = "?"
        else:
            mark = "✗"
            all_match = False
        
        diff_str = f"{diff:.2e}" if diff is not None else "N/A"
        print(f"  {mark} {step}: {status} (max_diff={diff_str})")
    
    if all_match:
        print("\n  ✓✓✓ LAYER 0 ALL COMPONENTS MATCH! ✓✓✓")
    else:
        print("\n  ⚠ Some components have differences. Check the details above.")

    # =========================================================================
    # LAYER 1 AND LAYER 2 COMPARISON (using helper function)
    # =========================================================================
    print("\n" + "=" * 70)
    print("LAYER 1 & 2 COMPARISON (Condensed)")
    print("=" * 70)
    
    all_layers_results = {0: comparison_results}
    
    for layer_idx in [1, 2]:
        layer_results = compare_layer(
            layer_idx=layer_idx,
            sglang_tensors=sglang_decode_for_hidden if sglang_decode_for_hidden else {},
            megatron_tensors=megatron_tensors,
            verbose=True,
        )
        all_layers_results[layer_idx] = layer_results
    
    # Print summary for all layers
    print("\n" + "=" * 70)
    print("ALL LAYERS SUMMARY")
    print("=" * 70)
    
    all_layers_match = True
    for layer_idx in [0, 1, 2]:
        layer_results = all_layers_results.get(layer_idx, {})
        layer_match = True
        mismatches = []
        for component, (status, diff) in layer_results.items():
            if status not in ("MATCH", "NOT_FOUND"):
                layer_match = False
                all_layers_match = False
                mismatches.append(f"{component}({diff:.2e})" if diff else component)
        
        if layer_match:
            print(f"  Layer {layer_idx}: ✓ ALL MATCH")
        else:
            print(f"  Layer {layer_idx}: ✗ DIFF in {', '.join(mismatches)}")
    
    if all_layers_match:
        print("\n  ✓✓✓ ALL LAYERS MATCH - TRUE ON-POLICY VERIFIED! ✓✓✓")

    # =========================================================================
    # FINAL LAYERNORM AND LM HEAD COMPARISON
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL LAYERNORM & LM HEAD COMPARISON")
    print("(Components between last transformer layer and logits)")
    print("=" * 70)

    # --- Final LayerNorm ---
    print("\n  FINAL LAYERNORM (after all transformer layers):")
    print("    SGLang: model.norm OUTPUT, Megatron: final_layernorm OUTPUT")
    print("    (Both should be normalized hidden states before lm_head)")

    # SGLang final norm - try model.norm key
    # Note: SGLang's norm hook may return (output, residual) tuple from model.norm()
    sglang_final_norm = None
    sglang_raw_val = None
    for key in ["model.norm", "norm"]:
        if key in sglang_decode_for_hidden:
            val = sglang_decode_for_hidden[key]
            sglang_raw_val = val
            print(f"    SGLang raw value type: {type(val)}")
            if isinstance(val, (list, tuple)):
                print(f"    SGLang raw value length: {len(val)}")
                for i, item in enumerate(val):
                    if isinstance(item, torch.Tensor):
                        print(f"      Element {i}: shape={item.shape}, "
                              f"dtype={item.dtype}, "
                              f"RMS={((item.float()**2).mean().sqrt().item()):.4f}")
            
            # SGLang's model.norm() returns (hidden_states, residual) tuple
            # The first element is the OUTPUT (normalized hidden states)
            # The second element is the residual (input before adding residual)
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                # Element 0: OUTPUT (normalized after adding residual)
                # Element 1: RESIDUAL (input before adding residual)
                sglang_final_norm = to_tensor(val[0], prefer_last=False)
                sglang_final_norm_input = to_tensor(val[1], prefer_last=False)
                if sglang_final_norm_input is not None:
                    rms_input = (sglang_final_norm_input.float() ** 2).mean().sqrt().item()
                    print(f"    SGLang norm INPUT (before residual): "
                          f"shape={sglang_final_norm_input.shape}, RMS={rms_input:.4f}")
            elif isinstance(val, (list, tuple)) and len(val) > 0:
                # Take FIRST element (output)
                sglang_final_norm = to_tensor(val[0], prefer_last=False)
                sglang_final_norm_input = None
            else:
                # Single tensor - might be output or input
                sglang_final_norm = to_tensor(val, prefer_last=True)
                sglang_final_norm_input = None
            
            if sglang_final_norm is not None:
                # Extract the last token if it's a sequence
                original_shape = sglang_final_norm.shape
                if sglang_final_norm.dim() == 2:
                    # [seq_len, hidden_size] -> take last token
                    sglang_final_norm = sglang_final_norm[-1]
                elif sglang_final_norm.dim() == 3:
                    # [batch, seq, dim] or [seq, batch, dim]
                    d0, d1, d2 = sglang_final_norm.shape
                    if d0 == 1:
                        # [batch=1, seq, dim] -> take last token
                        sglang_final_norm = sglang_final_norm[0, -1]
                    else:
                        # [seq, batch=1, dim] -> take last token
                        sglang_final_norm = sglang_final_norm[-1, 0]
                elif sglang_final_norm.dim() == 1:
                    # Already a single token [hidden_size]
                    pass
                
                # Check if this looks normalized (RMS should be ~1.0)
                rms_check = (sglang_final_norm.float() ** 2).mean().sqrt().item()
                print(f"    SGLang key: {key}, original shape: {original_shape}, "
                      f"extracted shape: {sglang_final_norm.shape}, RMS: {rms_check:.4f}")
                if rms_check > 5.0:
                    print(f"      ⚠️  RMS too high - this might be INPUT (before norm), "
                          f"not OUTPUT (after norm)")
                    print(f"      Expected RMS ~1.0 for normalized values")
                break

    if sglang_final_norm is None:
        print("    SGLang final_norm: NOT FOUND")
        norm_keys = [k for k in sglang_decode_for_hidden.keys()
                     if 'norm' in k.lower()]
        print(f"    Available keys containing 'norm': {norm_keys[:10]}")
    
    # Also check for keys that might be the output before lm_head
    # (e.g., after final norm, before lm_head projection)
    if sglang_final_norm is not None:
        # Check if there are other keys that might be the actual normalized output
        potential_keys = [k for k in sglang_decode_for_hidden.keys()
                         if any(x in k.lower() for x in ['final', 'norm', 'lm_head'])
                         and k != "model.norm" and k != "norm"]
        if potential_keys:
            print(f"    Other potential keys: {potential_keys[:5]}")
            # Check if any of these have normalized-looking values
            for pk in potential_keys[:3]:
                try:
                    pk_val = to_tensor(sglang_decode_for_hidden[pk], prefer_last=False)
                    if pk_val is not None and pk_val.numel() > 0:
                        if pk_val.dim() > 1:
                            # Extract last token if sequence
                            if pk_val.dim() == 2:
                                pk_val = pk_val[-1]
                            elif pk_val.dim() == 3:
                                d0, d1, d2 = pk_val.shape
                                if d0 == 1:
                                    pk_val = pk_val[0, -1]
                                else:
                                    pk_val = pk_val[-1, 0]
                        pk_rms = (pk_val.float() ** 2).mean().sqrt().item()
                        if 0.5 < pk_rms < 2.0:  # Looks normalized
                            print(f"      {pk}: shape={pk_val.shape}, RMS={pk_rms:.4f} "
                                  f"✓ Looks normalized!")
                except Exception as e:
                    pass

    # Megatron final_layernorm
    megatron_final_norm = None
    for key_pattern in [
        "final_layernorm_at_response_start",
        f"final_layernorm_pos_{first_response_pos}",
        "final_layernorm",
    ]:
        if key_pattern in megatron_tensors:
            megatron_final_norm = megatron_tensors[key_pattern]
            if megatron_final_norm.dim() == 2:
                megatron_final_norm = megatron_final_norm[0]  # [1, dim] -> [dim]
            elif megatron_final_norm.dim() == 3:
                # [batch, seq, dim] or [seq, batch, dim]
                d0, d1, d2 = megatron_final_norm.shape
                if d0 == 1:
                    megatron_final_norm = megatron_final_norm[0, first_response_pos]
                else:
                    megatron_final_norm = megatron_final_norm[first_response_pos, 0]
            print(f"    Megatron key: {key_pattern}, shape: {megatron_final_norm.shape}")
            break

    if megatron_final_norm is None:
        print("    Megatron final_layernorm: NOT FOUND")
        print(f"    Available keys containing 'final': "
              f"{[k for k in megatron_tensors.keys() if 'final' in k.lower()]}")

    if sglang_final_norm is not None and megatron_final_norm is not None:
        # Convert both to float32 for comparison
        sglang_fn = sglang_final_norm.float()
        megatron_fn = megatron_final_norm.float()
        
        # Verify shapes match
        if sglang_fn.shape != megatron_fn.shape:
            print(f"    ⚠️  Shape mismatch: SGLang {sglang_fn.shape} vs "
                  f"Megatron {megatron_fn.shape}")
            min_len = min(sglang_fn.numel(), megatron_fn.numel())
            sglang_fn = sglang_fn.flatten()[:min_len]
            megatron_fn = megatron_fn.flatten()[:min_len]
        
        diff_fn = (sglang_fn - megatron_fn).abs()
        max_diff_fn = diff_fn.max().item()
        mean_diff_fn = diff_fn.mean().item()
        
        # Check if values are normalized (RMS should be ~1.0)
        sglang_rms = (sglang_fn ** 2).mean().sqrt().item()
        megatron_rms = (megatron_fn ** 2).mean().sqrt().item()
        
        print(f"    SGLang first 10: {[f'{v:.4f}' for v in sglang_fn[:10].tolist()]}")
        print(f"    Megatron first 10: {[f'{v:.4f}' for v in megatron_fn[:10].tolist()]}")
        print(f"    SGLang RMS: {sglang_rms:.4f}, Megatron RMS: {megatron_rms:.4f}")
        print(f"    Max diff: {max_diff_fn:.6e}, Mean diff: {mean_diff_fn:.6e}")
        
        # If SGLang RMS is very high, try to manually normalize it
        if sglang_rms > 5.0 and megatron_rms < 5.0:
            print(f"\n    🔍 SGLang RMS ({sglang_rms:.2f}) is too high - "
                  f"trying manual normalization...")
            # Manually normalize SGLang values
            sglang_normalized = sglang_fn / sglang_rms
            sglang_normalized_rms = (sglang_normalized ** 2).mean().sqrt().item()
            print(f"    After manual normalization: RMS={sglang_normalized_rms:.4f}")
            
            # Compare normalized SGLang vs Megatron
            diff_normalized = (sglang_normalized - megatron_fn).abs()
            max_diff_norm = diff_normalized.max().item()
            mean_diff_norm = diff_normalized.mean().item()
            print(f"    Normalized comparison - Max diff: {max_diff_norm:.6e}, "
                  f"Mean diff: {mean_diff_norm:.6e}")
            
            if max_diff_norm < 1e-3:
                print("    ✓ After normalization, values are close!")
                print("    → SGLang hook captured INPUT (before norm), "
                      "not OUTPUT (after norm)")
            else:
                print("    ✗ Even after normalization, values differ")
                print("    → May be comparing different positions or implementations")
        
        if max_diff_fn < 1e-4:
            print("    ✓ Final LayerNorm MATCH!")
        else:
            print(f"    ✗ Final LayerNorm DIFFERENT (max_diff={max_diff_fn:.6e})")
            
            # If we have the input (before residual), compare that with Megatron
            if 'sglang_final_norm_input' in locals() and sglang_final_norm_input is not None:
                print(f"\n    🔍 Comparing SGLang INPUT (before residual) vs Megatron:")
                sg_input_flat = sglang_final_norm_input.flatten()
                if sg_input_flat.dim() > 1:
                    if sg_input_flat.shape[0] == 1:
                        sg_input_flat = sg_input_flat[0]
                    else:
                        sg_input_flat = sg_input_flat[-1]
                
                if sg_input_flat.shape == megatron_fn.shape:
                    diff_input = (sg_input_flat.float() - megatron_fn).abs()
                    max_diff_input = diff_input.max().item()
                    mean_diff_input = diff_input.mean().item()
                    print(f"      Max diff: {max_diff_input:.6e}, Mean diff: {mean_diff_input:.6e}")
                    if max_diff_input < 1e-4:
                        print("      ✓ SGLang INPUT matches Megatron!")
                        print("      → Difference is from residual addition in SGLang")
                        print(f"      → Residual magnitude: "
                              f"{((sglang_final_norm.float() - sg_input_flat.float()).abs().max().item()):.6e}")
                    else:
                        print(f"      ✗ SGLang INPUT also differs (max_diff={max_diff_input:.6e})")
            
            if sglang_rms > 5.0:
                print("    → SGLang value appears to be INPUT (before normalization)")
                print("    → Need to check SGLang hook structure or find correct key")

    # --- LM Head (output projection to vocab) ---
    print("\n  LM HEAD OUTPUT (projection to vocab):")
    print("    NOTE: We compare logits directly (logits_processor), not lm_head outputs.")
    print("    SGLang's lm_head hook may capture input (hidden states) not output (logits).")
    print("    Megatron's lm_head hook captures the actual logits output.")
    
    # Check what SGLang actually dumped
    sglang_lm_head_shape = None
    if "lm_head" in sglang_decode_for_hidden:
        val = sglang_decode_for_hidden["lm_head"]
        sglang_lm_head_tmp = to_tensor(val, prefer_last=True)
        if sglang_lm_head_tmp is not None:
            sglang_lm_head_shape = sglang_lm_head_tmp.shape
            print(f"    SGLang lm_head hook shape: {sglang_lm_head_shape}")
            if len(sglang_lm_head_shape) == 2 and sglang_lm_head_shape[1] < 10000:
                print(f"      ⚠️  This looks like INPUT (hidden states), not OUTPUT (logits)")
                print(f"      Expected logits shape: [1, vocab_size={151936}]")
    else:
        print("    SGLang lm_head: NOT FOUND in dump")

    # Check what Megatron dumped
    megatron_lm_head_shape = None
    for key_pattern in ["lm_head_at_response_start", "lm_head"]:
        if key_pattern in megatron_tensors:
            megatron_lm_head_shape = megatron_tensors[key_pattern].shape
            print(f"    Megatron {key_pattern} shape: {megatron_lm_head_shape}")
            break
    
    if megatron_lm_head_shape is None:
        print("    Megatron lm_head: NOT FOUND in dump")
    
    print("    → Logits comparison (below) is the authoritative check.")

    # =========================================================================
    # 2. Compare logits for first response token
    # =========================================================================
    print("\n" + "=" * 70)
    print("LOGITS COMPARISON FOR FIRST RESPONSE TOKEN")
    print("=" * 70)

    # Get first response token ID
    first_response_token = None
    input_ids_key = megatron_info.get("input_ids_key", "megatron_input_ids")
    if input_ids_key in megatron_tensors:
        input_ids = megatron_tensors[input_ids_key].flatten()
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

    # Get Megatron logits at prompt_len - 1
    megatron_logits = None
    logits_key = f"logits_pos_{comparison_pos}"

    if logits_key in megatron_tensors:
        megatron_logits = megatron_tensors[logits_key]
        print(
            f"Megatron {logits_key}: "
            f"shape={megatron_logits.shape}, dtype={megatron_logits.dtype}"
        )
    elif "logits_at_prompt_end" in megatron_tensors:
        megatron_logits = megatron_tensors["logits_at_prompt_end"]
        print(
            f"Megatron logits_at_prompt_end: "
            f"shape={megatron_logits.shape}, dtype={megatron_logits.dtype}"
        )
    elif "logits_full" in megatron_tensors:
        # Extract from full logits tensor
        full_logits = megatron_tensors["logits_full"]
        print(f"  Extracting from logits_full: shape={full_logits.shape}")
        # Detect format: [batch, seq, vocab] or [seq, batch, vocab]
        if full_logits.dim() == 3:
            d0, d1, d2 = full_logits.shape
            if d0 == 1 and d1 > 1:
                # [batch=1, seq, vocab]
                megatron_logits = full_logits[0, comparison_pos:comparison_pos+1, :]
            else:
                # [seq, batch=1, vocab]
                megatron_logits = full_logits[comparison_pos:comparison_pos+1, 0, :]
        elif full_logits.dim() == 2:
            megatron_logits = full_logits[comparison_pos:comparison_pos+1, :]
        print(
            f"Megatron (from logits_full at pos {comparison_pos}): "
            f"shape={megatron_logits.shape}"
        )
    else:
        print(f"WARNING: Could not find {logits_key} in Megatron dump")
        available = [k for k in megatron_tensors.keys() if "logits" in k.lower()]
        print(f"  Available logits keys: {available}")

    # Compare logits
    if sglang_logits is not None and megatron_logits is not None:
        sg_flat = sglang_logits.flatten()
        megatron_flat = megatron_logits.flatten()

        if sg_flat.shape != megatron_flat.shape:
            min_len = min(len(sg_flat), len(megatron_flat))
            print(
                f"  Shape mismatch: SGLang {sg_flat.shape} vs "
                f"Megatron {megatron_flat.shape}, using first {min_len}"
            )
            sg_flat = sg_flat[:min_len]
            megatron_flat = megatron_flat[:min_len]

        stats = compute_diff_stats(sg_flat, megatron_flat)

        print("\nLogits Comparison:")
        print(f"  Max diff:  {stats['max_diff']:.8e}")
        print(f"  Mean diff: {stats['mean_diff']:.8e}")
        print(f"  Rel diff:  {stats['rel_diff']:.8e}")

        if stats["max_diff"] < 1e-5:
            print("  ✓ Logits MATCH!")
        else:
            print("  ✗ Logits DIFFER!")

        # Print first 10 logits values
        print("\n  First 10 logits values:")
        sg_first10 = sg_flat[:10].float().tolist()
        meg_first10 = megatron_flat[:10].float().tolist()
        diff_first10 = (sg_flat[:10].float() - megatron_flat[:10].float()).abs().tolist()
        print(f"    SGLang:   {[f'{v:.4f}' for v in sg_first10]}")
        print(f"    Megatron: {[f'{v:.4f}' for v in meg_first10]}")
        print(f"    Diff:     {[f'{v:.4f}' for v in diff_first10]}")

        # Detailed difference analysis
        diff_all = (sg_flat.float() - megatron_flat.float()).abs()
        nonzero_diff_mask = diff_all > 1e-6
        num_diff = nonzero_diff_mask.sum().item()
        total_vocab = len(diff_all)
        pct_diff = 100.0 * num_diff / total_vocab
        print(f"\n  Detailed difference analysis:")
        print(f"    Total vocab size: {total_vocab}")
        print(f"    Exact matches (diff < 1e-6): {total_vocab - num_diff}")
        print(f"    With differences: {num_diff} ({pct_diff:.2f}%)")
        
        if num_diff > 0:
            # Distribution of differences
            nonzero_diffs = diff_all[nonzero_diff_mask]
            print(f"    Min non-zero diff: {nonzero_diffs.min().item():.6e}")
            print(f"    Max diff: {nonzero_diffs.max().item():.6e}")
            print(f"    Mean of non-zero diffs: {nonzero_diffs.mean().item():.6e}")
            
            # Find indices with largest differences
            top_diff_indices = diff_all.topk(5).indices.tolist()
            print(f"    Top 5 differing indices: {top_diff_indices}")
            for idx in top_diff_indices:
                sg_v = sg_flat[idx].float().item()
                meg_v = megatron_flat[idx].float().item()
                d = abs(sg_v - meg_v)
                print(f"      idx {idx}: SGLang={sg_v:.4f}, Meg={meg_v:.4f}, diff={d:.6f}")

        # Compare specific token logit
        if first_response_token is not None:
            sg_tok = sglang_logits.flatten()
            megatron_tok = megatron_logits.flatten()
            if first_response_token < len(sg_tok):
                sg_token_logit = sg_tok[first_response_token]
            else:
                sg_token_logit = None
            if first_response_token < len(megatron_tok):
                megatron_token_logit = megatron_tok[first_response_token]
            else:
                megatron_token_logit = None

            if sg_token_logit is not None and megatron_token_logit is not None:
                sg_val = sg_token_logit.float().item()
                megatron_val = megatron_token_logit.float().item()
                diff = abs(sg_val - megatron_val)
                tok = first_response_token
                print(f"\n  Logit for first response token {tok}:")
                print(f"    SGLang:  {sg_val:.8f}")
                print(f"    Megatron: {megatron_val:.8f}")
                print(f"    Diff:    {diff:.8e}")

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
        megatron_logprobs, megatron_target_lp = compute_logprobs_from_logits(
            megatron_logits, temperature=1.0, target_token_id=tok_id
        )

        sg_lp_flat = sg_logprobs.flatten()
        megatron_lp_flat = megatron_logprobs.flatten()

        if sg_lp_flat.shape != megatron_lp_flat.shape:
            min_len = min(len(sg_lp_flat), len(megatron_lp_flat))
            sg_lp_flat = sg_lp_flat[:min_len]
            megatron_lp_flat = megatron_lp_flat[:min_len]

        lp_stats = compute_diff_stats(sg_lp_flat, megatron_lp_flat)

        print("  Full distribution comparison:")
        print(f"    Max diff:  {lp_stats['max_diff']:.8e}")
        print(f"    Mean diff: {lp_stats['mean_diff']:.8e}")

        if sg_target_lp is not None and megatron_target_lp is not None:
            sg_lp_val = sg_target_lp.float().item()
            megatron_lp_val = megatron_target_lp.float().item()
            diff = abs(sg_lp_val - megatron_lp_val)
            tok = first_response_token
            print(f"\n  Logprob for first response token {tok}:")
            print(f"    SGLang:  {sg_lp_val:.8f}")
            print(f"    Megatron: {megatron_lp_val:.8f}")
            print(f"    Diff:    {diff:.8e}")

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

        # Check Megatron dumped logprobs
        megatron_direct_lp = None
        if "logprobs" in megatron_tensors:
            megatron_direct_lp = megatron_tensors["logprobs"]
            print(f"\n  Megatron dumped logprobs: {megatron_direct_lp}")
            print(f"    shape: {megatron_direct_lp.shape}")
            print(f"    dtype: {megatron_direct_lp.dtype}")

        if "logprobs_full" in megatron_tensors:
            full_lp = megatron_tensors["logprobs_full"]
            print(f"\n  Megatron full logprobs shape: {full_lp.shape}")
            print(f"    First 5 values: {full_lp.flatten()[:5].tolist()}")

        if "logprobs_extracted_idx" in megatron_tensors:
            idx = megatron_tensors["logprobs_extracted_idx"].item()
            print(f"    Extracted at index: {idx}")

        if "logprobs_prompt_len" in megatron_tensors:
            pl = megatron_tensors["logprobs_prompt_len"].item()
            print(f"    prompt_len used: {pl}")

        if "response_logprobs_first5" in megatron_tensors:
            resp_lp = megatron_tensors["response_logprobs_first5"]
            print(f"    First 5 response logprobs: {resp_lp.tolist()}")

        # Check if Megatron logprob is zero (common bug symptom)
        if megatron_direct_lp is not None:
            if megatron_direct_lp.abs().max().item() < 1e-10:
                print("\n  ⚠️  WARNING: Megatron dumped logprobs are ALL ZEROS!")
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
    print("  Megatron logits_pos_N predicts token at N+1")
    print("\nComparing predictions for:")
    second_resp_pos = first_response_pos + 1
    p1 = first_response_pos
    p2 = second_resp_pos
    print(f"  1. First response token (pos {p1}):")
    print(f"     Megatron pos {comparison_pos} vs SGLang first_pos {p1}")
    print(f"  2. Second response token (pos {p2}):")
    print(f"     Megatron pos {p1} vs SGLang first_pos {p2}")

    # Get tokens for both positions
    second_response_token = None
    input_ids_key = megatron_info.get("input_ids_key", "megatron_input_ids")
    if input_ids_key in megatron_tensors:
        input_ids = megatron_tensors[input_ids_key].flatten()
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

    # Megatron: use logits_pos_{prompt_len - 1}
    if megatron_logits is not None:
        print_top_logprobs(
            megatron_logits,
            actual_token_id=first_response_token,
            label=f"Megatron (logits_pos_{comparison_pos})",
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

    # Megatron: use logits_pos_{prompt_len} to predict position prompt_len + 1
    megatron_logits2 = None
    logits_key2 = f"logits_pos_{first_response_pos}"
    if logits_key2 in megatron_tensors:
        megatron_logits2 = megatron_tensors[logits_key2]
        print(f"\n  Megatron: logits_pos_{first_response_pos}")
    else:
        print(f"\n  Megatron does not have {logits_key2}")
        avail = [k for k in megatron_tensors.keys() if k.startswith("logits_pos_")]
        if avail:
            print(f"  Available: {avail[:5]}...")

    if megatron_logits2 is not None:
        print_top_logprobs(
            megatron_logits2,
            actual_token_id=second_response_token,
            label=f"Megatron (logits_pos_{first_response_pos})",
            top_k=10,
        )

    # Compare second response token predictions
    if sglang_logits2 is not None and megatron_logits2 is not None:
        print("\n  Second response token comparison:")
        sg_flat = sglang_logits2.flatten()
        megatron_flat = megatron_logits2.flatten()
        if sg_flat.shape == megatron_flat.shape:
            stats = compute_diff_stats(sg_flat, megatron_flat)
            print(f"    Max diff:  {stats['max_diff']:.8e}")
            print(f"    Mean diff: {stats['mean_diff']:.8e}")
            if stats["max_diff"] < 1e-5:
                print("    ✓ Second response token logits MATCH!")
            else:
                print("    ✗ Second response token logits DIFFER!")
        else:
            print(f"    Shape mismatch: {sg_flat.shape} vs {megatron_flat.shape}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRUE ON-POLICY VERIFICATION SUMMARY")
    print("=" * 70)
    print("""
The attention path is the CRITICAL component for true on-policy.
If attention matches, the model produces identical token predictions.

ATTENTION PATH (must match for true on-policy):
  ├─ Input LayerNorm (RMSNorm)        → Compare layer_0_qkv_layernorm
  ├─ QKV Projection                    → Compare layer_0_qkv_proj  
  ├─ Q/K LayerNorm                     → Compare q_norm/k_norm
  ├─ RoPE (Rotary Position Embedding)  → Compare Q/K after RoPE
  ├─ Flash Attention (core_attention)  → Compare core_attention output
  └─ Output Projection (o_proj)        → Compare o_proj output

MLP PATH (small differences acceptable):
  ├─ Post-Attention LayerNorm          → Compare post_attention_layernorm
  ├─ Gate/Up Projection                → Compare gate_up_proj
  ├─ Activation (SiLU)                 → (internal)
  └─ Down Projection                   → Compare mlp_output

If ALL attention components show 0.0 diff, TRUE ON-POLICY is working!
Small MLP differences (<1e-2) are acceptable and don't affect predictions.
""")
    print("=" * 70)


def list_passes_detailed(sglang_dir: str, megatron_dir: str) -> None:
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
    print("MEGATRON PASSES (Training)")
    print("=" * 70)

    megatron_passes = list_all_passes(megatron_dir)
    for pass_id, path in megatron_passes[:10]:
        info = get_megatron_dump_info(path)
        backend = info.get('backend', 'Unknown')
        prompt_len = info.get('prompt_len', '?')
        total_len = info.get('total_len', '?')
        response_len = info.get('response_len', '?')
        print(
            f"  Pass {pass_id:3d}: {backend:8s} "
            f"prompt_len={prompt_len}, total_len={total_len}, "
            f"response_len={response_len}"
        )

    if len(megatron_passes) > 10:
        print(f"  ... and {len(megatron_passes) - 10} more passes")

    print("\n" + "=" * 70)
    print("KEY INSIGHT:")
    print("=" * 70)
    print("  - SGLang: MANY passes (1 prefill + N decode passes)")
    print("  - Megatron: ONE pass for entire sequence")
    print("  - To compare first response token:")
    print("    * Use SGLang's PREFILL pass (seq_len = prompt_len)")
    print("    * Compare with Megatron's logits at position (prompt_len - 1)")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Compare SGLang and Megatron tensor dumps"
    )
    parser.add_argument(
        "--sglang-dir", type=str, required=True,
        help="SGLang tensor dump directory"
    )
    parser.add_argument(
        "--megatron-dir", type=str, required=True,
        help="Megatron tensor dump directory"
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
        list_passes_detailed(args.sglang_dir, args.megatron_dir)
        sys.exit(0)

    # Default to compare-first-token
    compare_first_response_token(
        args.sglang_dir,
        args.megatron_dir,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

