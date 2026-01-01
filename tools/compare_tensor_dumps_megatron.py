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


def compute_logprobs_sglang(
    logits: torch.Tensor,
    temperature: float = 1.0,
    target_token_id: int | None = None,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Compute log probabilities using SGLang's production sampler path.

    This matches the exact code in sglang/python/sglang/srt/layers/sampler.py
    when rl_on_policy_target is set:
        logits_div_temperature = logits.bfloat16().div(sampling_info.temperatures).bfloat16()
        logprobs_via_logsoftmax_kernel = torch.log_softmax(logits_div_temperature, dim=-1)

    IMPORTANT: sampling_info.temperatures is a float32 TENSOR, not a Python float!
    See sglang/python/sglang/srt/sampling/sampling_batch_info.py:76-78:
        temperatures = torch.tensor([...], dtype=torch.float)  # torch.float32

    The division bfloat16 / float32 produces float32 intermediate result,
    which is then converted back to bfloat16.

    Args:
        logits: Raw logits tensor
        temperature: Temperature for softmax (default 1.0)
        target_token_id: If provided, return logprob for this token
        verbose: If True, print intermediate values for debugging

    Returns:
        (full_logprobs, target_logprob)
    """
    # Exact SGLang production path from sampler.py lines 111-117
    # CRITICAL: Use float32 tensor for temperature to match SGLang's precision
    temp_tensor = torch.tensor(temperature, dtype=torch.float32, device=logits.device)
    logits_bf16 = logits.bfloat16()
    logits_div_temperature = logits_bf16.div(temp_tensor).bfloat16()
    logprobs = torch.log_softmax(logits_div_temperature, dim=-1)

    if verbose:
        logits_flat = logits_div_temperature.flatten()
        print("    [SGLang] Temperature processing:")
        print(f"      temperature: {temperature} (as float32 tensor)")
        print(f"      logits_bf16 dtype: {logits_bf16.dtype}")
        print(f"      logits_div_temperature dtype: {logits_div_temperature.dtype}")
        print(f"      logits_div_temperature first 10: {logits_flat[:10].tolist()}")
        if target_token_id is not None:
            target_logit = logits_flat[target_token_id].item() if logits_flat.dim() == 1 else logits_div_temperature.flatten()[target_token_id].item()
            print(f"      logit for token {target_token_id}: {target_logit:.6f}")

    target_logprob = None
    if target_token_id is not None:
        if logprobs.dim() == 1:
            target_logprob = logprobs[target_token_id]
        elif logprobs.dim() == 2:
            target_logprob = logprobs[0, target_token_id]
        elif logprobs.dim() == 3:
            target_logprob = logprobs[0, 0, target_token_id]

    return logprobs, target_logprob


def compute_logprobs_megatron(
    logits: torch.Tensor,
    target_token_id: int | None = None,
    temperature: float = 1.0,
    true_on_policy_mode: bool = True,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Compute log probabilities using Megatron's production path.

    This matches the exact code in:
    1. slime/slime/backends/megatron_utils/loss.py get_responses() - temperature handling
    2. slime/slime/utils/ppo_utils.py compute_log_probs() - log_softmax

    Megatron's true_on_policy_mode path (loss.py:66-73):
        temp_tensor = torch.tensor(temperature, dtype=torch.float32, device=logits.device)
        logits = logits.bfloat16().div(temp_tensor).bfloat16()

    Then in ppo_utils.py compute_log_probs():
        logits_bf16 = logits.bfloat16()  # Already bfloat16, no-op
        log_probs = torch.log_softmax(logits_bf16, dim=-1)

    IMPORTANT: Temperature is applied as float32 tensor to match SGLang's precision!

    Args:
        logits: Raw logits tensor
        target_token_id: If provided, return logprob for this token
        temperature: Temperature for softmax (default 1.0)
        true_on_policy_mode: If True, use SGLang-compatible BF16 path
        verbose: If True, print intermediate values for debugging

    Returns:
        (full_logprobs, target_logprob)
    """
    if true_on_policy_mode:
        # Exact Megatron production path from loss.py:66-73 + ppo_utils.py:162-170
        # CRITICAL: Use float32 tensor for temperature to match SGLang's precision
        temp_tensor = torch.tensor(temperature, dtype=torch.float32, device=logits.device)
        logits_bf16 = logits.bfloat16()
        if temperature != 1.0:
            logits_bf16 = logits_bf16.div(temp_tensor).bfloat16()
        logprobs = torch.log_softmax(logits_bf16, dim=-1)

        if verbose:
            logits_flat = logits_bf16.flatten()
            print("    [Megatron] Temperature processing (true_on_policy_mode):")
            print(f"      temperature: {temperature} (as float32 tensor)")
            print(f"      logits_bf16 dtype: {logits_bf16.dtype}")
            print(f"      logits_bf16 (after temp) first 10: {logits_flat[:10].tolist()}")
            print(f"      logits_bf16 (after temp) sum: {logits_flat.sum().item():.6f}")
            if target_token_id is not None:
                target_logit = logits_flat[target_token_id].item() if logits_flat.dim() == 1 else logits_bf16.flatten()[target_token_id].item()
                print(f"      logit for token {target_token_id}: {target_logit:.6f}")
                import torch.nn.functional as F
                log_probs = F.log_softmax(logits_bf16, dim=-1)
                print(f"      temp log_probs dtype: {log_probs.dtype}")
                print(f"      temp log_probs first 10: {log_probs.flatten()[:10].tolist()}")
                print(f"      temp log_probs sum: {log_probs.sum().item():.6f}")
                print(f"      temp log_probs for token {target_token_id}: {log_probs.flatten()[target_token_id].item():.6f}")
    else:
        # Original Megatron path (non-true-on-policy)
        # In production, this uses fused_vocab_parallel_cross_entropy
        # For comparison, we simulate with standard log_softmax in float32
        logits_scaled = logits.float()
        if temperature != 1.0:
            logits_scaled = logits_scaled / temperature
        logprobs = torch.log_softmax(logits_scaled, dim=-1)

        if verbose:
            print("    [Megatron] Temperature processing (non-true_on_policy):")
            print(f"      temperature: {temperature}")
            print(f"      logits_scaled dtype: {logits_scaled.dtype}")
            print(f"      logits_scaled first 10: {logits_scaled.flatten()[:10].tolist()}")

    target_logprob = None
    if target_token_id is not None:
        if logprobs.dim() == 1:
            target_logprob = logprobs[target_token_id]
        elif logprobs.dim() == 2:
            target_logprob = logprobs[0, target_token_id]
        elif logprobs.dim() == 3:
            target_logprob = logprobs[0, 0, target_token_id]

    return logprobs, target_logprob


def compute_logprobs_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    target_token_id: int | None = None,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Compute log probabilities using the unified SGLang/Megatron true-on-policy path.

    This is a compatibility wrapper that uses the true-on-policy formula
    which is identical for both SGLang and Megatron:
        temp_tensor = torch.tensor(temperature, dtype=torch.float32)
        logits_bf16 = logits.bfloat16()
        logits_div_temp = logits_bf16.div(temp_tensor).bfloat16()
        logprobs = torch.log_softmax(logits_div_temp, dim=-1)

    Args:
        logits: Raw logits tensor
        temperature: Temperature for softmax (default 1.0)
        target_token_id: If provided, return logprob for this token
        verbose: If True, print intermediate values for debugging

    Returns:
        (full_logprobs, target_logprob)
    """
    # Use SGLang's formula as the unified path
    return compute_logprobs_sglang(logits, temperature, target_token_id, verbose=verbose)


def print_top_logprobs(
    logits: torch.Tensor,
    actual_token_id: int | None,
    label: str,
    top_k: int = 10,
    temperature: float = 1.0,
    source: str = "sglang",
    true_on_policy_mode: bool = True,
) -> None:
    """
    Print top-k logprobs and the logprob for the actual predicted token.

    Uses the appropriate production function based on source:
    - "sglang": Uses SGLang's sampler.py production path
    - "megatron": Uses Megatron's ppo_utils.py production path

    Args:
        logits: Raw logits tensor (vocab_size,) or (1, vocab_size)
        actual_token_id: The token that was actually generated/in the sequence
        label: Label for printing (e.g., "SGLang pos 90" or "Megatron pos 90")
        top_k: Number of top tokens to show
        temperature: Temperature for softmax
        source: "sglang" or "megatron" to select the appropriate production function
        true_on_policy_mode: For Megatron, whether to use true-on-policy BF16 path
    """
    # Use the appropriate production function based on source
    if source == "sglang":
        logprobs, _ = compute_logprobs_sglang(logits, temperature)
    elif source == "megatron":
        logprobs, _ = compute_logprobs_megatron(
            logits, temperature=temperature, true_on_policy_mode=true_on_policy_mode
        )
    else:
        raise ValueError(f"Unknown source: {source}. Must be 'sglang' or 'megatron'")

    logprobs_flat = logprobs.flatten().float()

    # Get top-k tokens by logprob
    k = min(top_k, len(logprobs_flat))
    top_values, top_indices = torch.topk(logprobs_flat, k)

    print(f"\n  {label}:")
    print(f"    Top {top_k} tokens by logprob (using {source} production path):")
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

            # Get dtypes for comparison
            sglang_dtype = (
                sglang_hidden.dtype if sglang_hidden is not None else None
            )
            megatron_dtype = (
                megatron_hidden.dtype if megatron_hidden is not None else None
            )
            dtype_match = (
                (sglang_dtype == megatron_dtype)
                if (sglang_dtype is not None and megatron_dtype is not None)
                else None
            )

            print(
                f"  {color}Layer {layer_idx:2d}: {match_str} "
                f"max_diff={stats['max_diff']:.6e}, "
                f"mean_diff={stats['mean_diff']:.6e} "
                f"(SGLang {sg_source}{list_info}){end_color}"
            )

            # Always show tensor shapes and dtypes
            sg_shape = sg_flat.shape
            megatron_shape = megatron_flat.shape
            print(f"    SGLang shape: {sg_shape}, dtype: {sglang_dtype}")
            print(
                f"    Megatron shape: {megatron_shape}, "
                f"dtype: {megatron_dtype}"
            )
            if dtype_match is not None:
                if dtype_match:
                    print(f"    ✓ Dtype matches: {sglang_dtype}")
                else:
                    print(
                        f"    ✗ Dtype mismatch: SGLang {sglang_dtype} vs "
                        f"Megatron {megatron_dtype}"
                    )

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
            # Get dtypes for comparison
            sg_dtype = sg.dtype
            meg_dtype = meg.dtype
            dtype_match = (sg_dtype == meg_dtype)

            sg_vals = [f'{v:.4f}' for v in sg[:10].float().tolist()]
            meg_vals = [f'{v:.4f}' for v in meg[:10].float().tolist()]
            print(f"    {name}:")
            print(
                f"      SGLang dtype: {sg_dtype}, "
                f"Megatron dtype: {meg_dtype}"
            )
            if dtype_match:
                print("      ✓ Dtype matches")
            else:
                print("      ✗ Dtype mismatch")
            print(f"      SGLang first 10:   {sg_vals}")
            print(f"      Megatron first 10: {meg_vals}")
            print(f"      Max diff: {max_diff:.6e}")
            if max_diff < 1e-5:
                print("      ✓ MATCH")

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


def compare_single_pass_pair(
    megatron_pass_id: int,
    megatron_path: Path,
    sglang_dir: str,
    verbose: bool = True,
) -> None:
    """
    Compare a single Megatron pass with its corresponding SGLang pass.

    Args:
        megatron_pass_id: ID of the Megatron pass
        megatron_path: Path to the Megatron pass file
        sglang_dir: Directory containing SGLang dumps
        verbose: Whether to print detailed output
    """
    print("\n" + "=" * 70)
    print(f"MEGATRON PASS {megatron_pass_id:05d} COMPARISON")
    print("=" * 70)

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
            print("\n    Megatron layer_0_qkv_proj pos 91:")
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
                print("\n    Megatron layer_0_qkv_proj (base):")
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
            print("\n    Megatron q_layernorm pos 91:")
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
            print("\n    Megatron k_layernorm pos 91:")
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
            print("    Megatron pre_mlp_layernorm pos 91:")
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
            print("    Megatron mlp_output pos 91:")
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

    # Find the last layer index for summary
    # Strategy: Check for layer 28 first (known last layer for Qwen3 models),
    # then fall back to detected max layer from dump files
    last_layer_for_summary = None

    # Collect all layer indices from both dumps
    megatron_layer_indices = set()
    for key in megatron_tensors.keys():
        match = re.match(r"layer_(\d+)_", key)
        if match:
            layer_idx = int(match.group(1))
            megatron_layer_indices.add(layer_idx)

    sglang_layer_indices = set()
    if sglang_decode_for_hidden:
        for key in sglang_decode_for_hidden.keys():
            match = re.search(r"layers\.(\d+)\.", key)
            if match:
                layer_idx = int(match.group(1))
                sglang_layer_indices.add(layer_idx)

    all_available_layers = sorted(megatron_layer_indices & sglang_layer_indices)

    # Check if layer 28 exists in either dump (even if not in intersection)
    # This handles the case where layer 28 was added automatically but not in initial dump_layers
    candidate_last_layers = [28]  # Known last layer for Qwen3 models
    if all_available_layers:
        candidate_last_layers.append(max(all_available_layers))

    # Find the highest layer that exists in both dumps
    for candidate in sorted(candidate_last_layers, reverse=True):
        if (candidate in megatron_layer_indices and
                candidate in sglang_layer_indices):
            last_layer_for_summary = candidate
            break

    # If still not found, use the max from intersection
    if last_layer_for_summary is None and all_available_layers:
        last_layer_for_summary = max(all_available_layers)

    # Print summary for all layers
    print("\n" + "=" * 70)
    print("ALL LAYERS SUMMARY")
    print("=" * 70)

    # Layers to include in summary: first 3 layers + last layer
    layers_to_summarize = [0, 1, 2]
    if last_layer_for_summary is not None and last_layer_for_summary not in layers_to_summarize:
        layers_to_summarize.append(last_layer_for_summary)

    all_layers_match = True
    for layer_idx in layers_to_summarize:
        layer_results = all_layers_results.get(layer_idx, {})
        layer_match = True
        mismatches = []
        for component, (status, diff) in layer_results.items():
            if status not in ("MATCH", "NOT_FOUND"):
                layer_match = False
                all_layers_match = False
                mismatches.append(f"{component}({diff:.2e})" if diff else component)

        layer_label = f"Layer {layer_idx}"
        if layer_idx == last_layer_for_summary and layer_idx not in [0, 1, 2]:
            layer_label = f"Layer {layer_idx} (LAST)"

        if layer_match:
            print(f"  {layer_label}: ✓ ALL MATCH")
        else:
            print(f"  {layer_label}: ✗ DIFF in {', '.join(mismatches)}")

    if all_layers_match:
        print("\n  ✓✓✓ ALL LAYERS MATCH - TRUE ON-POLICY VERIFIED! ✓✓✓")

    # =========================================================================
    # LAST LAYER (LAYER 28) COMPARISON
    # =========================================================================
    # Find the last layer index from available layers (reuse detection logic)
    last_layer_idx = last_layer_for_summary

    # Debug: Print detection info
    print("\n  Layer detection info:")
    print(f"    Megatron layers found: {sorted(megatron_layer_indices)}")
    print(f"    SGLang layers found: {sorted(sglang_layer_indices)}")
    print(f"    Intersection: {sorted(all_available_layers)}")
    print(f"    Last layer index: {last_layer_idx}")

    # If layer 28 is not detected but should exist, check if it's in the dumps
    if last_layer_idx != 28:
        # Check if layer 28 exists in either dump (may have been auto-added)
        has_layer_28_megatron = any(
            re.match(r"layer_28_", key) for key in megatron_tensors.keys()
        )
        has_layer_28_sglang = False
        if sglang_decode_for_hidden:
            has_layer_28_sglang = any(
                re.search(r"layers\.28\.", key)
                for key in sglang_decode_for_hidden.keys()
            )

        print("    Checking for layer 28:")
        print(f"      In Megatron dump: {has_layer_28_megatron}")
        print(f"      In SGLang dump: {has_layer_28_sglang}")

        if has_layer_28_megatron and has_layer_28_sglang:
            print("    → Layer 28 found in both dumps, using it as last layer")
            last_layer_idx = 28

    # Hardcode layer 27 as the last layer (for Qwen3-0.6B with 28 layers: 0-27)
    last_layer_idx_hardcoded = 27

    print("\n" + "=" * 70)
    print(f"LAST LAYER (LAYER {last_layer_idx_hardcoded}) COMPARISON")
    print("=" * 70)
    print("Comparing the last transformer layer before final layernorm")
    print(f"  Using hardcoded last layer: {last_layer_idx_hardcoded}")
    print(f"  Available layers in both dumps: {sorted(all_available_layers)}")

    print(f"\n  Comparing Layer {last_layer_idx_hardcoded} components:")
    last_layer_results = compare_layer(
        layer_idx=last_layer_idx_hardcoded,
        sglang_tensors=sglang_decode_for_hidden if sglang_decode_for_hidden else {},
        megatron_tensors=megatron_tensors,
        verbose=True,
    )
    all_layers_results[last_layer_idx_hardcoded] = last_layer_results

    # Print summary for last layer
    print(f"\n  Layer {last_layer_idx_hardcoded} Summary:")
    last_layer_match = True
    last_layer_mismatches = []
    for component, (status, diff) in last_layer_results.items():
        if status not in ("MATCH", "NOT_FOUND"):
            last_layer_match = False
            last_layer_mismatches.append(
                f"{component}({diff:.2e})" if diff else component
            )
        status_mark = (
            "✓" if status == "MATCH"
            else "⚠" if status == "CLOSE" else "✗"
        )
        diff_str = f"{diff:.2e}" if diff is not None else "N/A"
        print(f"    {status_mark} {component}: {status} "
              f"(max_diff={diff_str})")

    if last_layer_match:
        print(f"\n  ✓✓✓ LAYER {last_layer_idx_hardcoded} ALL COMPONENTS MATCH! ✓✓✓")
    else:
        print(f"\n  ⚠ Layer {last_layer_idx_hardcoded} has differences: "
              f"{', '.join(last_layer_mismatches)}")

    # =====================================================================
    # LAST LAYER INPUT COMPARISON
    # =====================================================================
    print("\n" + "=" * 70)
    print(f"LAST LAYER (LAYER {last_layer_idx_hardcoded}) INPUT COMPARISON")
    print("=" * 70)
    print("Comparing the input to the last transformer layer")
    print(f"  SGLang: output from layer {last_layer_idx_hardcoded - 1}")
    print(f"  Megatron: layer_{last_layer_idx_hardcoded}_input")

    # SGLang: Get output from previous layer (last_layer_idx - 1)
    # The input to last layer is the output from previous layer (after residual add)
    prev_layer_idx = last_layer_idx_hardcoded - 1
    sglang_last_layer_input = None
    # Try to find the previous layer's output (which is the input to last layer)
    # In SGLang, layer output is after MLP + residual, so we look for the full layer output
    for key_pattern in [
        f"model.layers.{prev_layer_idx}.mlp.down_proj",  # MLP output (before residual)
        f"model.layers.{prev_layer_idx}.mlp",  # MLP output
        f"layer_{prev_layer_idx}_output",  # Full layer output (if captured)
    ]:
        tensors_dict = sglang_decode_for_hidden if sglang_decode_for_hidden else {}
        if key_pattern in tensors_dict:
            val = tensors_dict[key_pattern]
            if val is not None:
                sglang_last_layer_input = to_tensor(val, prefer_last=True)
                if sglang_last_layer_input is not None:
                    # Extract at response start position
                    if sglang_last_layer_input.dim() == 2:
                        sglang_last_layer_input = sglang_last_layer_input[-1]  # Last token
                    elif sglang_last_layer_input.dim() == 3:
                        d0, d1, d2 = sglang_last_layer_input.shape
                        if d0 == 1:
                            sglang_last_layer_input = sglang_last_layer_input[0, -1]
                        else:
                            sglang_last_layer_input = sglang_last_layer_input[-1, 0]
                    print(f"    SGLang key: {key_pattern}, shape: {sglang_last_layer_input.shape}")
                    break

    # If not found, try to compute from previous layer's components
    # The input to last layer should be: prev_layer_mlp_output + prev_layer_residual
    # But we don't have direct access to this, so we use MLP output as approximation
    # (Note: This might not be exact if residual was added after MLP)

    # Megatron: Get layer input
    megatron_last_layer_input = None
    megatron_input_key = f"layer_{last_layer_idx_hardcoded}_input_at_response_start"
    if megatron_input_key not in megatron_tensors:
        # Try without _at_response_start suffix
        megatron_input_key = f"layer_{last_layer_idx_hardcoded}_input"

    if megatron_input_key in megatron_tensors:
        megatron_last_layer_input = megatron_tensors[megatron_input_key]
        if megatron_last_layer_input.dim() == 2:
            megatron_last_layer_input = megatron_last_layer_input[0]  # [1, dim] -> [dim]
        elif megatron_last_layer_input.dim() == 3:
            d0, d1, d2 = megatron_last_layer_input.shape
            if d0 == 1:
                megatron_last_layer_input = megatron_last_layer_input[0, first_response_pos]
            else:
                megatron_last_layer_input = megatron_last_layer_input[first_response_pos, 0]
        print(f"    Megatron key: {megatron_input_key}, shape: {megatron_last_layer_input.shape}")

    if sglang_last_layer_input is None:
        print("    ⚠ SGLang last layer input: NOT FOUND")
    if megatron_last_layer_input is None:
        print("    ⚠ Megatron last layer input: NOT FOUND")

    if sglang_last_layer_input is not None and megatron_last_layer_input is not None:
        # Convert both to float32 for comparison
        sg_input = sglang_last_layer_input.float()
        meg_input = megatron_last_layer_input.float()

        # Verify shapes match
        if sg_input.shape != meg_input.shape:
            print(
                f"    ⚠️  Shape mismatch: SGLang {sg_input.shape} vs "
                f"Megatron {meg_input.shape}"
            )
            min_len = min(sg_input.numel(), meg_input.numel())
            sg_input = sg_input.flatten()[:min_len]
            meg_input = meg_input.flatten()[:min_len]

        diff_input = (sg_input - meg_input).abs()
        max_diff_input = diff_input.max().item()
        mean_diff_input = diff_input.mean().item()

        # Check RMS values
        sg_rms = (sg_input ** 2).mean().sqrt().item()
        meg_rms = (meg_input ** 2).mean().sqrt().item()

        print("\n    🔍 LAST LAYER INPUT COMPARISON:")
        print("    " + "=" * 70)
        print(f"    SGLang (layer {prev_layer_idx} output):")
        print(f"      shape: {sg_input.shape}, RMS: {sg_rms:.4f}")
        print(
            f"      first 10: "
            f"{[f'{v:.4f}' for v in sg_input[:10].tolist()]}"
        )
        print(f"    Megatron (layer_{last_layer_idx_hardcoded}_input):")
        print(f"      shape: {meg_input.shape}, RMS: {meg_rms:.4f}")
        print(
            f"      first 10: "
            f"{[f'{v:.4f}' for v in meg_input[:10].tolist()]}"
        )
        print("    Difference:")
        print(
            f"      Max diff: {max_diff_input:.6e}, "
            f"Mean diff: {mean_diff_input:.6e}"
        )

        # Find indices with largest differences
        top_diff_indices = diff_input.topk(min(5, len(diff_input))).indices
        print("    Top 5 differences:")
        for idx in top_diff_indices:
            idx_val = idx.item()
            print(
                f"      idx={idx_val}: SGLang={sg_input[idx_val]:.6f}, "
                f"Megatron={meg_input[idx_val]:.6f}, "
                f"diff={diff_input[idx_val]:.6e}"
            )

        if max_diff_input < 1e-5:
            print("\n    ✓✓✓ LAST LAYER INPUT MATCHES! ✓✓✓")
        elif max_diff_input < 1e-3:
            print(
                f"\n    ⚠ LAST LAYER INPUT CLOSE "
                f"(max_diff={max_diff_input:.6e})"
            )
        else:
            print(
                f"\n    ✗ LAST LAYER INPUT DIFFERS "
                f"(max_diff={max_diff_input:.6e})"
            )

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
                print("\n    📊 SGLang model.norm elements breakdown:")
                for i, item in enumerate(val):
                    if isinstance(item, torch.Tensor):
                        rms_val = ((item.float()**2).mean().sqrt().item())
                        if item.dim() > 1:
                            # Extract last token for display
                            if item.dim() == 2:
                                item_display = item[-1]
                            elif item.dim() == 3:
                                d0, d1, d2 = item.shape
                                if d0 == 1:
                                    item_display = item[0, -1]
                                else:
                                    item_display = item[-1, 0]
                            else:
                                item_display = item
                        else:
                            item_display = item

                        first_10 = [f'{v:.4f}' for v in item_display[:10].tolist()] if item_display.numel() >= 10 else [f'{v:.4f}' for v in item_display.flatten()[:10].tolist()]

                        if i == 0:
                            print(f"      Element {i} (OUTPUT - normalized after residual):")
                        elif i == 1:
                            print(f"      Element {i} (INPUT - before normalization, after residual):")
                        else:
                            print(f"      Element {i}:")
                        print(f"        shape={item.shape}, dtype={item.dtype}, RMS={rms_val:.4f}")
                        print(f"        first 10: {first_10}")

            # SGLang's model.norm() returns (hidden_states, residual) tuple
            # Element 0: OUTPUT (normalized hidden states after adding residual)
            # Element 1: INPUT (input before normalization, after adding residual)
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                # Element 0: OUTPUT (normalized after adding residual)
                # Element 1: INPUT (before normalization, after residual)
                sglang_final_norm = to_tensor(val[0], prefer_last=False)
                sglang_final_norm_input = to_tensor(val[1], prefer_last=False)
                if sglang_final_norm_input is not None:
                    rms_input = (sglang_final_norm_input.float() ** 2).mean().sqrt().item()
                    print(f"\n    SGLang norm INPUT (element 1, before normalization): "
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
                    print(
                        "      ⚠️  RMS too high - this might be INPUT "
                        "(before norm), not OUTPUT (after norm)"
                    )
                    print("      Expected RMS ~1.0 for normalized values")
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
        potential_keys = [
            k for k in sglang_decode_for_hidden.keys()
            if any(x in k.lower() for x in ['final', 'norm', 'lm_head'])
            and k != "model.norm" and k != "norm"
        ]
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
                except Exception:
                    pass

    # Megatron final_layernorm OUTPUT
    megatron_final_norm = None
    megatron_final_norm_raw = None  # Keep raw tensor for dtype check
    for key_pattern in [
        "final_layernorm_at_response_start",
        f"final_layernorm_pos_{first_response_pos}",
        "final_layernorm",
    ]:
        if key_pattern in megatron_tensors:
            megatron_final_norm_raw = megatron_tensors[key_pattern]
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

    # Megatron INPUT to final_layernorm (output of last transformer layer)
    # This should match SGLang's element 1 (input before normalization)
    megatron_final_norm_input = None
    # Use hardcoded layer 27 as the last layer (for Qwen3-0.6B with 28 layers: 0-27)
    last_layer_idx_for_final = 27
    print(f"    Using last layer index: {last_layer_idx_for_final} (hardcoded)")

    for key_pattern in [
        f"layer_{last_layer_idx_for_final}_output_at_response_start",
        f"layer_{last_layer_idx_for_final}_output",
    ]:
        if key_pattern in megatron_tensors:
            megatron_final_norm_input = megatron_tensors[key_pattern]
            if megatron_final_norm_input.dim() == 2:
                megatron_final_norm_input = megatron_final_norm_input[0]  # [1, dim] -> [dim]
            elif megatron_final_norm_input.dim() == 3:
                d0, d1, d2 = megatron_final_norm_input.shape
                if d0 == 1:
                    megatron_final_norm_input = megatron_final_norm_input[0, first_response_pos]
                else:
                    megatron_final_norm_input = megatron_final_norm_input[first_response_pos, 0]
            print(f"    Megatron INPUT to final_layernorm (last layer output): "
                  f"key={key_pattern}, shape={megatron_final_norm_input.shape}")
            break

    # Quick comparison: SGLang Element 1 vs Megatron last layer output
    # Note: Element 1 is (mlp_output + residual), while layer_N_output is just mlp_output
    # So they should be different - this comparison shows the difference
    if sglang_final_norm_input is not None and megatron_final_norm_input is not None:
        sg_elem1_f = sglang_final_norm_input.float()
        meg_last_layer_f = megatron_final_norm_input.float()

        if sg_elem1_f.shape != meg_last_layer_f.shape:
            min_len = min(sg_elem1_f.numel(), meg_last_layer_f.numel())
            sg_elem1_f = sg_elem1_f.flatten()[:min_len]
            meg_last_layer_f = meg_last_layer_f.flatten()[:min_len]

        diff_elem1_vs_layer = (sg_elem1_f - meg_last_layer_f).abs()
        max_diff_elem1_vs_layer = diff_elem1_vs_layer.max().item()
        mean_diff_elem1_vs_layer = diff_elem1_vs_layer.mean().item()

        print(f"\n    🔍 QUICK COMPARISON: SGLang Element 1 vs Megatron layer_{last_layer_idx_for_final}_output")
        print(f"    {'='*70}")
        print(
            "    SGLang Element 1 (INPUT - mlp_output + residual, "
            "before norm):"
        )
        print(f"      shape: {sg_elem1_f.shape}, RMS: {(sg_elem1_f**2).mean().sqrt().item():.4f}")
        print(f"      first 10: {[f'{v:.4f}' for v in sg_elem1_f[:10].tolist()]}")
        print(f"    Megatron layer_{last_layer_idx_for_final}_output (MLP output, WITHOUT residual):")
        print(f"      shape: {meg_last_layer_f.shape}, RMS: {(meg_last_layer_f**2).mean().sqrt().item():.4f}")
        print(f"      first 10: {[f'{v:.4f}' for v in meg_last_layer_f[:10].tolist()]}")
        print(f"    Difference (Element 1 - layer_{last_layer_idx_for_final}_output):")
        print(f"      Max diff: {max_diff_elem1_vs_layer:.6e}, Mean diff: {mean_diff_elem1_vs_layer:.6e}")
        print(
            "    Note: Element 1 = (mlp_output + residual), "
            "so difference should be ~residual value"
        )

        # Find top differences
        top_diff_elem1 = diff_elem1_vs_layer.topk(min(5, len(diff_elem1_vs_layer))).indices
        print("    Top 5 differences:")
        for idx in top_diff_elem1:
            idx_val = idx.item()
            print(f"      idx={idx_val}: SGLang={sg_elem1_f[idx_val]:.6f}, "
                  f"Megatron={meg_last_layer_f[idx_val]:.6f}, "
                  f"diff={diff_elem1_vs_layer[idx_val]:.6e}")

    if megatron_final_norm is None:
        print("    Megatron final_layernorm: NOT FOUND")
        print(f"    Available keys containing 'final': "
              f"{[k for k in megatron_tensors.keys() if 'final' in k.lower()]}")

    # Compare the INPUT to final_layernorm (MLP output + residual) if available
    # SGLang Element 1 is the input AFTER residual addition (before normalization)
    # This should match (Megatron MLP output + Megatron residual)
    sglang_final_norm_input = None
    if isinstance(sglang_raw_val, (list, tuple)) and len(sglang_raw_val) > 1:
        sglang_final_norm_input = sglang_raw_val[1]  # Element 1 is INPUT before normalization
        if isinstance(sglang_final_norm_input, torch.Tensor):
            if sglang_final_norm_input.dim() > 1:
                if sglang_final_norm_input.dim() == 2:
                    sglang_final_norm_input = sglang_final_norm_input[0]  # [1, dim] -> [dim]
                elif sglang_final_norm_input.dim() == 3:
                    d0, d1, d2 = sglang_final_norm_input.shape
                    if d0 == 1:
                        sglang_final_norm_input = sglang_final_norm_input[0, -1]
                    else:
                        sglang_final_norm_input = sglang_final_norm_input[-1, 0]

    # Try to find Megatron residual value passed to final_layernorm
    megatron_residual = None
    for key_pattern in [
        f"layer_{last_layer_idx_for_final}_residual_at_response_start",
        "final_layernorm_residual_at_response_start",
        "residual_at_response_start",
    ]:
        if key_pattern in megatron_tensors:
            megatron_residual = megatron_tensors[key_pattern]
            if megatron_residual.dim() == 2:
                megatron_residual = megatron_residual[0]
            elif megatron_residual.dim() == 3:
                d0, d1, d2 = megatron_residual.shape
                if d0 == 1:
                    megatron_residual = megatron_residual[0, first_response_pos]
                else:
                    megatron_residual = megatron_residual[first_response_pos, 0]
            print(f"    Megatron RESIDUAL passed to final_layernorm: "
                  f"key={key_pattern}, shape={megatron_residual.shape}")
            break

    # Compare INPUT to final_layernorm (MLP output + residual)
    if sglang_final_norm_input is not None and megatron_final_norm_input is not None:
        sg_input_f = sglang_final_norm_input.float()
        meg_input_f = megatron_final_norm_input.float()

        # If we have residual, compute the sum (MLP output + residual)
        if megatron_residual is not None:
            meg_residual_f = megatron_residual.float()
            meg_sum_f = meg_input_f + meg_residual_f
            print("\n    🔍 COMPARING INPUT TO FINAL LAYERNORM:")
            print("    " + "=" * 70)
            print("    SGLang Element 1 (MLP_output + residual, before norm):")
            print(f"      shape: {sg_input_f.shape}, RMS: {(sg_input_f**2).mean().sqrt().item():.4f}")
            print(f"      first 10: {[f'{v:.4f}' for v in sg_input_f[:10].tolist()]}")
            print("    Megatron (MLP_output + residual, computed):")
            print(f"      MLP output shape: {meg_input_f.shape}, RMS: {(meg_input_f**2).mean().sqrt().item():.4f}")
            print(f"      Residual shape: {meg_residual_f.shape}, RMS: {(meg_residual_f**2).mean().sqrt().item():.4f}")
            print(f"      Sum shape: {meg_sum_f.shape}, RMS: {(meg_sum_f**2).mean().sqrt().item():.4f}")
            print(f"      Sum first 10: {[f'{v:.4f}' for v in meg_sum_f[:10].tolist()]}")

            # Compare the sums
            if sg_input_f.shape != meg_sum_f.shape:
                min_len = min(sg_input_f.numel(), meg_sum_f.numel())
                sg_input_f = sg_input_f.flatten()[:min_len]
                meg_sum_f = meg_sum_f.flatten()[:min_len]

            diff_input = (sg_input_f - meg_sum_f).abs()
            max_diff_input = diff_input.max().item()
            mean_diff_input = diff_input.mean().item()
            print("    Difference (SGLang Element 1 vs Megatron sum):")
            print(f"      Max diff: {max_diff_input:.6e}, Mean diff: {mean_diff_input:.6e}")

            if max_diff_input < 1e-3:
                print("    ✓ INPUT to final_layernorm MATCHES!")
            else:
                print("    ✗ INPUT to final_layernorm DIFFERS")
                top_diff_input = diff_input.topk(min(5, len(diff_input))).indices
                print("    Top 5 differences in INPUT:")
                for idx in top_diff_input:
                    idx_val = idx.item()
                    print(f"      idx={idx_val}: SGLang={sg_input_f[idx_val]:.6f}, "
                          f"Megatron={meg_sum_f[idx_val]:.6f}, "
                          f"diff={diff_input[idx_val]:.6e}")
        else:
            print(
                "\n    ⚠️  Could not find Megatron residual - "
                "cannot compare INPUT to final_layernorm"
            )
            print(f"    Available keys: {[k for k in megatron_tensors.keys() if 'residual' in k.lower()]}")

    if sglang_final_norm is not None and megatron_final_norm is not None:
        # =====================================================================
        # DTYPE COMPARISON
        # =====================================================================
        print("\n    🔍 DTYPE COMPARISON:")
        print("    " + "=" * 70)

        # Get SGLang element 0 and element 1 dtypes
        sglang_elem0_dtype = None
        sglang_elem1_dtype = None
        if isinstance(sglang_raw_val, (list, tuple)) and len(sglang_raw_val) >= 2:
            if isinstance(sglang_raw_val[0], torch.Tensor):
                sglang_elem0_dtype = sglang_raw_val[0].dtype
            if isinstance(sglang_raw_val[1], torch.Tensor):
                sglang_elem1_dtype = sglang_raw_val[1].dtype

        # Get Megatron final_layernorm dtype (from raw tensor before position extraction)
        megatron_final_norm_dtype = None
        if megatron_final_norm_raw is not None and isinstance(megatron_final_norm_raw, torch.Tensor):
            megatron_final_norm_dtype = megatron_final_norm_raw.dtype
        elif isinstance(megatron_final_norm, torch.Tensor):
            megatron_final_norm_dtype = megatron_final_norm.dtype

        print(f"    SGLang model.norm Element 0 (OUTPUT): dtype={sglang_elem0_dtype}")
        print(f"    SGLang model.norm Element 1 (INPUT):  dtype={sglang_elem1_dtype}")
        print(f"    Megatron final_layernorm:              dtype={megatron_final_norm_dtype}")

        # Check dtype alignment
        dtype_match_elem0 = (sglang_elem0_dtype == megatron_final_norm_dtype)
        dtype_match_elem1 = (sglang_elem1_dtype == megatron_final_norm_dtype) if sglang_elem1_dtype is not None else None

        if dtype_match_elem0:
            print("    ✓ Element 0 dtype matches Megatron final_layernorm")
        else:
            print(f"    ✗ Element 0 dtype mismatch: SGLang {sglang_elem0_dtype} vs Megatron {megatron_final_norm_dtype}")

        if dtype_match_elem1 is not None:
            if dtype_match_elem1:
                print("    ✓ Element 1 dtype matches Megatron final_layernorm")
            else:
                print(f"    ⚠ Element 1 dtype differs: SGLang {sglang_elem1_dtype} vs Megatron {megatron_final_norm_dtype}")
                print(
                    "      (Note: Element 1 is INPUT before normalization, "
                    "may have different dtype)"
                )

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

        print(
            "\n    🔍 DETAILED COMPARISON: SGLang Element 0 vs "
            "Megatron final_layernorm"
        )
        print("    " + "=" * 70)
        print("    SGLang Element 0 (OUTPUT):")
        print(f"      shape: {sglang_fn.shape}, dtype: {sglang_elem0_dtype}, RMS: {sglang_rms:.4f}")
        print(f"      first 10: {[f'{v:.4f}' for v in sglang_fn[:10].tolist()]}")
        print("    Megatron final_layernorm_at_response_start:")
        print(f"      shape: {megatron_fn.shape}, dtype: {megatron_final_norm_dtype}, RMS: {megatron_rms:.4f}")
        print(f"      first 10: {[f'{v:.4f}' for v in megatron_fn[:10].tolist()]}")
        print("    Difference:")
        print(
            f"      Max diff: {max_diff_fn:.6e}, "
            f"Mean diff: {mean_diff_fn:.6e}"
        )

        # Find indices with largest differences
        top_diff_indices = diff_fn.topk(min(5, len(diff_fn))).indices
        print("    Top 5 differences:")
        for idx in top_diff_indices:
            idx_val = idx.item()
            print(f"      idx={idx_val}: SGLang={sglang_fn[idx_val]:.6f}, "
                  f"Megatron={megatron_fn[idx_val]:.6f}, "
                  f"diff={diff_fn[idx_val]:.6e}")

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

            # Compare inputs to understand the difference
            if ('sglang_final_norm_input' in locals() and sglang_final_norm_input is not None
                    and megatron_final_norm_input is not None):
                print("\n    🔍 DIAGNOSTICS: Comparing inputs to normalization")
                print("    " + "=" * 70)
                sg_input = sglang_final_norm_input
                if sg_input.dim() > 1:
                    if sg_input.shape[0] == 1:
                        sg_input = sg_input[0]
                    else:
                        sg_input = sg_input[-1]

                if sg_input.shape == megatron_final_norm_input.shape:
                    sg_input_f = sg_input.float()
                    meg_input_f = megatron_final_norm_input.float()
                    diff_input = (sg_input_f - meg_input_f).abs()
                    max_diff_input = diff_input.max().item()
                    mean_diff_input = diff_input.mean().item()

                    sg_input_rms = (sg_input_f ** 2).mean().sqrt().item()
                    meg_input_rms = (meg_input_f ** 2).mean().sqrt().item()

                    print("    SGLang Element 1 (INPUT - before normalization):")
                    print(f"      shape: {sg_input_f.shape}, RMS: {sg_input_rms:.4f}")
                    print(f"      first 10: {[f'{v:.4f}' for v in sg_input_f[:10].tolist()]}")
                    print("    Megatron INPUT (last layer output):")
                    print(f"      shape: {meg_input_f.shape}, RMS: {meg_input_rms:.4f}")
                    print(f"      first 10: {[f'{v:.4f}' for v in meg_input_f[:10].tolist()]}")
                    print("    Input difference:")
                    print(f"      Max diff: {max_diff_input:.6e}, Mean diff: {mean_diff_input:.6e}")

                    if max_diff_input < 1e-3:
                        print(
                            "\n    ✓ Inputs are close - difference is from "
                            "normalization computation"
                        )
                        # Check if difference is from residual
                        residual_est = sg_input_f - meg_input_f
                        residual_mag = residual_est.abs().max().item()
                        residual_mean = residual_est.abs().mean().item()
                        print(
                            "    Estimated residual (SGLang input - "
                            "Megatron input):"
                        )
                        print(f"      Max: {residual_mag:.6e}, Mean: {residual_mean:.6e}")

                        # Try manually normalizing both inputs using RMSNorm formula
                        # to see if we get the same output
                        print("\n    🔬 Testing normalization computation:")
                        print("    " + "=" * 70)
                        # SGLang's RMSNorm formula: x * rsqrt(mean(x^2) + eps) * weight
                        # Use typical eps=1e-6 (should match config)
                        eps = 1e-6

                        # Normalize SGLang input
                        sg_input_var = (sg_input_f ** 2).mean()
                        sg_input_normed = sg_input_f * torch.rsqrt(sg_input_var + eps)

                        # Normalize Megatron input
                        meg_input_var = (meg_input_f ** 2).mean()
                        meg_input_normed = meg_input_f * torch.rsqrt(meg_input_var + eps)

                        # Compare normalized inputs (without weight)
                        diff_norm_inputs = (sg_input_normed - meg_input_normed).abs()
                        max_diff_norm_inputs = diff_norm_inputs.max().item()
                        mean_diff_norm_inputs = diff_norm_inputs.mean().item()
                        print("    Normalized inputs (without weight) comparison:")
                        print(f"      Max diff: {max_diff_norm_inputs:.6e}, "
                              f"Mean diff: {mean_diff_norm_inputs:.6e}")

                        # Compare Megatron normalized input vs SGLang output
                        diff_norm_comp = (meg_input_normed - sglang_fn).abs()
                        max_diff_norm_comp = diff_norm_comp.max().item()
                        mean_diff_norm_comp = diff_norm_comp.mean().item()
                        print(
                            "    Megatron normalized input vs SGLang output "
                            "(element 0):"
                        )
                        print(f"      Max diff: {max_diff_norm_comp:.6e}, "
                              f"Mean diff: {mean_diff_norm_comp:.6e}")

                        if max_diff_norm_comp < 0.2:
                            print("    ✓ Normalization computation is similar")
                            print("    → Difference likely from:")
                            print(
                                "       1. Weight values (SGLang weight vs "
                                "Megatron weight)"
                            )
                            print("       2. Residual addition in SGLang")
                            print(
                                "       3. Numerical precision in bfloat16 "
                                "operations"
                            )
                        else:
                            print("    ✗ Normalization computation differs")
                            print(
                                "    → Check epsilon values, variance "
                                "computation, or weight application"
                            )
                    else:
                        print("\n    ✗ Inputs differ significantly")
                        print("    → This suggests different last layer outputs")
                        print("    → Need to check layer 2 output comparison")

            # Check if difference is acceptable (within bfloat16 precision)
            if max_diff_fn < 0.5:  # bfloat16 has ~0.01 precision for values around 1-10
                print(f"\n    ⚠️  Difference is small (max={max_diff_fn:.6e})")
                print("    → May be within bfloat16 numerical precision limits")
                print("    → Check if logits match (more important)")

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
                print(
                    "      ⚠️  This looks like INPUT (hidden states), "
                    "not OUTPUT (logits)"
                )
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
    # 2. Compare logits for first 5 response tokens
    # =========================================================================
    print("\n" + "=" * 70)
    print("LOGITS COMPARISON FOR FIRST 5 RESPONSE TOKENS")
    print("=" * 70)

    # Get first 5 response token IDs
    response_tokens = []
    input_ids_key = megatron_info.get("input_ids_key", "megatron_input_ids")
    if input_ids_key in megatron_tensors:
        input_ids = megatron_tensors[input_ids_key].flatten()
        for i in range(5):
            pos = first_response_pos + i
            if pos < len(input_ids):
                token_id = input_ids[pos].item()
                response_tokens.append((pos, token_id))
                print(
                    f"\nResponse token #{i+1}: ID={token_id} "
                    f"(at position {pos})"
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
        print(f"  Extracting from logits_full: shape={full_logits.shape}, dtype={full_logits.dtype}")
        if full_logits.dtype == torch.bfloat16:
            print("  ⚠️  WARNING: logits_full is bfloat16 (OLD dump format)!")
            print("     This may cause precision differences vs actual training.")
            print("     Regenerate dumps with the fixed dumper for accurate comparison.")
        elif full_logits.dtype == torch.float32:
            print("  ✓ logits_full is float32 (NEW dump format) - Good!")
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
        # =====================================================================
        # DTYPE COMPARISON FOR LOGITS
        # =====================================================================
        sglang_logits_dtype = sglang_logits.dtype
        megatron_logits_dtype = megatron_logits.dtype

        print("\n  🔍 LOGITS DTYPE COMPARISON:")
        print(f"    SGLang logits dtype:   {sglang_logits_dtype}")
        print(f"    Megatron logits dtype: {megatron_logits_dtype}")

        dtype_match_logits = (
            sglang_logits_dtype == megatron_logits_dtype
        )
        if dtype_match_logits:
            print("    ✓ Logits dtype matches")
        else:
            print(
                f"    ✗ Logits dtype mismatch: "
                f"SGLang {sglang_logits_dtype} vs "
                f"Megatron {megatron_logits_dtype}"
            )
            if (sglang_logits_dtype == torch.float32 and
                    megatron_logits_dtype == torch.bfloat16):
                print(
                    "      ⚠️  WARNING: Megatron hook may have converted "
                    "float32 to bfloat16, causing precision loss."
                )
            elif (sglang_logits_dtype == torch.bfloat16 and
                  megatron_logits_dtype == torch.float32):
                print(
                    "      ⚠️  WARNING: SGLang hook may have converted "
                    "float32 to bfloat16, causing precision loss."
                )

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
        sg_first10 = sg_flat[:10].tolist()
        meg_first10 = megatron_flat[:10].tolist()
        diff_first10 = (sg_flat[:10] - megatron_flat[:10]).abs().tolist()
        print(f"    SGLang:   {[f'{v:.8f}' for v in sg_first10]}")
        print(f"    Megatron: {[f'{v:.8f}' for v in meg_first10]}")
        print(f"    Diff:     {[f'{v:.8f}' for v in diff_first10]}")

        # Detailed difference analysis
        diff_all = (sg_flat.float() - megatron_flat.float()).abs()
        nonzero_diff_mask = diff_all > 1e-6
        num_diff = nonzero_diff_mask.sum().item()
        total_vocab = len(diff_all)
        pct_diff = 100.0 * num_diff / total_vocab
        print("\n  Detailed difference analysis:")
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

        # Compare specific token logits for first 5 response tokens
        print("\n  Logits for first 5 response tokens:")
        for i, (resp_pos, token_id) in enumerate(response_tokens):
            # Determine Megatron logits position
            if i == 0:
                # First token: use comparison_pos (prompt_len - 1)
                megatron_logits_pos = comparison_pos
            else:
                # Subsequent tokens: use previous response position
                megatron_logits_pos = first_response_pos + i - 1

            # Get Megatron logits for this position
            megatron_logits_curr = None
            logits_key_curr = f"logits_pos_{megatron_logits_pos}"
            if logits_key_curr in megatron_tensors:
                megatron_logits_curr = megatron_tensors[logits_key_curr]
            elif "logits_full" in megatron_tensors:
                full_logits = megatron_tensors["logits_full"]
                if full_logits.dim() == 3:
                    d0, d1, d2 = full_logits.shape
                    if d0 == 1:
                        megatron_logits_curr = full_logits[0, megatron_logits_pos:megatron_logits_pos+1, :]
                    else:
                        megatron_logits_curr = full_logits[megatron_logits_pos:megatron_logits_pos+1, 0, :]
                elif full_logits.dim() == 2:
                    megatron_logits_curr = full_logits[megatron_logits_pos:megatron_logits_pos+1, :]

            # Get SGLang logits for this position
            sglang_logits_curr = None
            if i == 0:
                # First token: use prefill pass logits
                sglang_logits_curr = sglang_logits
            else:
                # Subsequent tokens: load from decode pass
                sglang_decode_curr = find_sglang_decode_pass(sglang_dir, resp_pos)
                if sglang_decode_curr is not None:
                    decode_id, decode_path = sglang_decode_curr
                    decode_tensors = torch.load(decode_path, map_location="cpu")
                    if "logits_processor" in decode_tensors:
                        sglang_logits_curr = decode_tensors["logits_processor"]

            # Compare logits for this token
            if sglang_logits_curr is not None and megatron_logits_curr is not None:
                sg_tok = sglang_logits_curr.flatten()
                megatron_tok = megatron_logits_curr.flatten()
                if token_id < len(sg_tok) and token_id < len(megatron_tok):
                    sg_token_logit = sg_tok[token_id]
                    megatron_token_logit = megatron_tok[token_id]
                    sg_val = sg_token_logit.float().item()
                    megatron_val = megatron_token_logit.float().item()
                    diff = abs(sg_val - megatron_val)
                    print(f"\n    Token #{i+1} (ID={token_id}, pos={resp_pos}):")
                    print(f"      Logit for token {token_id}:")
                    print(f"        SGLang:    {sg_val:.8f}")
                    print(f"        Megatron:  {megatron_val:.8f}")
                    print(f"        Diff:      {diff:.8e}")
                    if diff < 1e-5:
                        print("        ✓ MATCH")
                    else:
                        print("        ✗ DIFFER")
                    
                    # Print first 10 logits values
                    n_show = min(10, len(sg_tok), len(megatron_tok))
                    sg_first10 = sg_tok[:n_show].float().tolist()
                    meg_first10 = megatron_tok[:n_show].float().tolist()
                    diff_first10 = (sg_tok[:n_show].float() - megatron_tok[:n_show].float()).abs().tolist()
                    print(f"      First {n_show} logits values:")
                    print(f"        SGLang:   {[f'{v:.8f}' for v in sg_first10]}")
                    print(f"        Megatron: {[f'{v:.8f}' for v in meg_first10]}")
                    print(f"        Diff:     {[f'{v:.8e}' for v in diff_first10]}")
                else:
                    print(f"\n    Token #{i+1} (ID={token_id}): Index out of bounds")
            else:
                print(f"\n    Token #{i+1} (ID={token_id}): Missing logits")
                if sglang_logits_curr is None:
                    print("      SGLang logits not found")
                if megatron_logits_curr is None:
                    print(f"      Megatron logits not found (pos {megatron_logits_pos})")

        # =====================================================================
        # 3. Compare logprobs (using respective production functions)
        # =====================================================================
        print("\n" + "-" * 50)
        print("LOGPROBS COMPARISON (using production paths) - First 5 Response Tokens")
        print("-" * 50)
        print("  SGLang:   sampler.py (logits.bfloat16().div(float32_temp_tensor).bfloat16() -> log_softmax)")
        print("  Megatron: loss.py + ppo_utils.py (logits.bfloat16().div(float32_temp_tensor).bfloat16() -> log_softmax)")
        print("\n  NOTE: Both use float32 tensor for temperature division to ensure precision match!")

        temperature = 0.8  # Default temperature, adjust if needed
        print(f"\n  Using temperature: {temperature}")

        # Compare logprobs for first 5 response tokens
        print("\n  Logprobs for first 5 response tokens:")
        for i, (resp_pos, token_id) in enumerate(response_tokens):
            # Determine Megatron logits position
            if i == 0:
                megatron_logits_pos = comparison_pos
            else:
                megatron_logits_pos = first_response_pos + i - 1

            # Get Megatron logits for this position
            megatron_logits_curr = None
            logits_key_curr = f"logits_pos_{megatron_logits_pos}"
            if logits_key_curr in megatron_tensors:
                megatron_logits_curr = megatron_tensors[logits_key_curr]
            elif "logits_full" in megatron_tensors:
                full_logits = megatron_tensors["logits_full"]
                if full_logits.dim() == 3:
                    d0, d1, d2 = full_logits.shape
                    if d0 == 1:
                        megatron_logits_curr = full_logits[0, megatron_logits_pos:megatron_logits_pos+1, :]
                    else:
                        megatron_logits_curr = full_logits[megatron_logits_pos:megatron_logits_pos+1, 0, :]
                elif full_logits.dim() == 2:
                    megatron_logits_curr = full_logits[megatron_logits_pos:megatron_logits_pos+1, :]

            # Get SGLang logits for this position
            sglang_logits_curr = None
            if i == 0:
                sglang_logits_curr = sglang_logits
            else:
                sglang_decode_curr = find_sglang_decode_pass(sglang_dir, resp_pos)
                if sglang_decode_curr is not None:
                    decode_id, decode_path = sglang_decode_curr
                    decode_tensors = torch.load(decode_path, map_location="cpu")
                    if "logits_processor" in decode_tensors:
                        sglang_logits_curr = decode_tensors["logits_processor"]

            # Compute logprobs for this token
            if sglang_logits_curr is not None and megatron_logits_curr is not None:
                # Use SGLang's production path
                verbose_first = (i == 0)  # Only verbose for first token
                if verbose_first:
                    print(f"\n  --- Token #{i+1} SGLang Temperature Processing ---")
                sg_logprobs_curr, sg_target_lp = compute_logprobs_sglang(
                    sglang_logits_curr, temperature=temperature, target_token_id=token_id, verbose=verbose_first
                )
                # Use Megatron's production path
                if verbose_first:
                    print(f"\n  --- Token #{i+1} Megatron Temperature Processing ---")
                megatron_logprobs_curr, megatron_target_lp = compute_logprobs_megatron(
                    megatron_logits_curr, target_token_id=token_id, temperature=temperature, 
                    true_on_policy_mode=True, verbose=verbose_first
                )

                if sg_target_lp is not None and megatron_target_lp is not None:
                    sg_lp_val = sg_target_lp.float().item()
                    megatron_lp_val = megatron_target_lp.float().item()
                    diff = abs(sg_lp_val - megatron_lp_val)
                    print(f"\n    Token #{i+1} (ID={token_id}, pos={resp_pos}):")
                    print(f"      Logprob for token {token_id}:")
                    print(f"        SGLang (production):   {sg_lp_val:.8f}")
                    print(f"        Megatron (production): {megatron_lp_val:.8f}")
                    print(f"        Diff:                  {diff:.8e}")
                    if diff < 1e-5:
                        print("        ✓ Logprobs MATCH!")
                    else:
                        print("        ✗ Logprobs DIFFER!")
                    
                    # Print first 10 logprobs values
                    sg_lp_flat = sg_logprobs_curr.flatten()
                    megatron_lp_flat = megatron_logprobs_curr.flatten()
                    n_show = min(10, len(sg_lp_flat), len(megatron_lp_flat))
                    sg_lp_first10 = sg_lp_flat[:n_show].float().tolist()
                    meg_lp_first10 = megatron_lp_flat[:n_show].float().tolist()
                    diff_lp_first10 = (sg_lp_flat[:n_show].float() - megatron_lp_flat[:n_show].float()).abs().tolist()
                    print(f"      First {n_show} logprobs values:")
                    print(f"        SGLang:   {[f'{v:.8f}' for v in sg_lp_first10]}")
                    print(f"        Megatron: {[f'{v:.8f}' for v in meg_lp_first10]}")
                    print(f"        Diff:     {[f'{v:.8e}' for v in diff_lp_first10]}")
                    
                    # Print top 5 logprobs and tokens from Megatron
                    top_k = min(5, len(megatron_lp_flat))
                    meg_top_values, meg_top_indices = torch.topk(megatron_lp_flat.float(), top_k)
                    print(f"      Top {top_k} logprobs from Megatron:")
                    for rank, (val, idx) in enumerate(zip(meg_top_values, meg_top_indices), 1):
                        is_target = (idx.item() == token_id)
                        marker = " <-- ACTUAL TOKEN" if is_target else ""
                        print(f"        {rank}. token_id={idx.item():6d}, logprob={val.item():.8f}{marker}")
                else:
                    print(f"\n    Token #{i+1} (ID={token_id}): Could not extract logprob")
            else:
                print(f"\n    Token #{i+1} (ID={token_id}): Missing logits for logprob computation")

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
    # 5. Detailed logprob comparison at FIVE positions
    # =========================================================================
    print("\n" + "=" * 70)
    print("DETAILED LOGPROBS: FIRST FIVE RESPONSE TOKENS")
    print("=" * 70)
    print("\nPosition Mapping:")
    print("  SGLang decode at first_position=X predicts token at X")
    print("  Megatron logits_pos_N predicts token at N+1")
    print("\nComparing predictions for first 5 response tokens:")

    # Get tokens for all positions
    input_ids_key = megatron_info.get("input_ids_key", "megatron_input_ids")
    response_tokens = []
    if input_ids_key in megatron_tensors:
        input_ids = megatron_tensors[input_ids_key].flatten()
        for i in range(5):
            pos = first_response_pos + i
            if pos < len(input_ids):
                token_id = input_ids[pos].item()
                response_tokens.append((pos, token_id))
                if i == 0:
                    megatron_pos = comparison_pos + i
                else:
                    megatron_pos = first_response_pos + i - 1
                print(
                    f"  {i+1}. Response token at pos {pos} "
                    f"(token_id={token_id}):"
                )
                print(
                    f"     Megatron pos {megatron_pos} vs "
                    f"SGLang first_pos {pos}"
                )

    # Process each of the first 5 response tokens
    for token_idx, (resp_pos, token_id) in enumerate(response_tokens):
        print("\n" + "-" * 50)
        print(f"RESPONSE TOKEN #{token_idx+1} (position {resp_pos})")
        print(f"Actual token: {token_id}")
        print("-" * 50)

        # Determine Megatron position for this token
        if token_idx == 0:
            # First token: use comparison_pos (prompt_len - 1)
            megatron_pos = comparison_pos
        else:
            # Subsequent tokens: use previous response position
            megatron_pos = first_response_pos + token_idx - 1

        # =========================================================
        # Get SGLang logits for this position
        # =========================================================
        sglang_logits_curr = None
        if token_idx == 0:
            # First token: use already loaded sglang_logits
            sglang_logits_curr = sglang_logits
        else:
            # Subsequent tokens: load from decode pass
            sglang_decode_curr = find_sglang_decode_pass(sglang_dir, resp_pos)
            if sglang_decode_curr is not None:
                decode_id, decode_path = sglang_decode_curr
                decode_tensors = torch.load(decode_path, map_location="cpu")
                if "logits_processor" in decode_tensors:
                    sglang_logits_curr = decode_tensors["logits_processor"]
                    print(f"\n  SGLang: decode pass {decode_id} "
                          f"(first_pos={resp_pos})")
                else:
                    print(f"\n  SGLang pass {decode_id} has no logits_processor")
            else:
                print(f"\n  No SGLang decode pass for pos {resp_pos}")
                if token_idx == 1:  # Only show available passes once
                    passes = list_all_passes(sglang_dir)
                    decode_passes = []
                    for pid, ppath in passes:
                        info = get_sglang_pass_info(ppath)
                        if info.get("is_decode", False):
                            decode_passes.append((pid, info.get("first_position", "?")))
                    if decode_passes:
                        print(f"  Available decode passes: {decode_passes[:10]}...")

        # =========================================================
        # Get Megatron logits for this position
        # =========================================================
        megatron_logits_curr = None
        logits_key_curr = f"logits_pos_{megatron_pos}"
        if logits_key_curr in megatron_tensors:
            megatron_logits_curr = megatron_tensors[logits_key_curr]
            print(f"\n  Megatron: {logits_key_curr}")
        else:
            print(f"\n  Megatron does not have {logits_key_curr}")
            if token_idx == 1:  # Only show available keys once
                avail = [
                    k for k in megatron_tensors.keys()
                    if k.startswith("logits_pos_")
                ]
                if avail:
                    print(f"  Available: {avail[:10]}...")

        # =========================================================
        # Print top logprobs for SGLang (using SGLang's production path)
        # =========================================================
        if sglang_logits_curr is not None:
            print_top_logprobs(
                sglang_logits_curr,
                actual_token_id=token_id,
                label=f"SGLang (decode, first_pos={resp_pos})",
                top_k=10,
                source="sglang",
            )

        # =========================================================
        # Print top logprobs for Megatron (using Megatron's production path)
        # =========================================================
        if megatron_logits_curr is not None:
            print_top_logprobs(
                megatron_logits_curr,
                actual_token_id=token_id,
                label=f"Megatron (logits_pos_{megatron_pos})",
                top_k=10,
                source="megatron",
                true_on_policy_mode=True,
            )

        # =========================================================
        # Compare logits
        # =========================================================
        if sglang_logits_curr is not None and megatron_logits_curr is not None:
            print(f"\n  Response token #{token_idx+1} logits comparison:")
            sg_flat = sglang_logits_curr.flatten()
            megatron_flat = megatron_logits_curr.flatten()
            if sg_flat.shape == megatron_flat.shape:
                stats = compute_diff_stats(sg_flat, megatron_flat)
                print(f"    Max diff:  {stats['max_diff']:.8e}")
                print(f"    Mean diff: {stats['mean_diff']:.8e}")
                if stats["max_diff"] < 1e-5:
                    print(
                        f"    ✓ Response token #{token_idx+1} "
                        f"logits MATCH!"
                    )
                else:
                    print(
                        f"    ✗ Response token #{token_idx+1} "
                        f"logits DIFFER!"
                    )
            else:
                print(f"    Shape mismatch: {sg_flat.shape} vs {megatron_flat.shape}")

        # =========================================================
        # Compare logprobs (using respective production functions)
        # =========================================================
        if (sglang_logits_curr is not None and
                megatron_logits_curr is not None):
            print(
                f"\n  Response token #{token_idx+1} "
                f"logprobs comparison (using production paths):"
            )
            # Use temperature=0.8 to match typical rollout settings
            # Change this if your rollout uses a different temperature
            resp_temperature = 0.8
            print(f"    Using temperature: {resp_temperature}")

            # Use SGLang's production path for SGLang logits
            sg_logprobs_curr, sg_target_lp = compute_logprobs_sglang(
                sglang_logits_curr,
                temperature=resp_temperature,
                target_token_id=token_id,
                verbose=(token_idx == 0),  # Only verbose for first token
            )
            # Use Megatron's production path for Megatron logits
            megatron_logprobs_curr, megatron_target_lp = compute_logprobs_megatron(
                megatron_logits_curr,
                target_token_id=token_id,
                temperature=resp_temperature,
                true_on_policy_mode=True,
                verbose=(token_idx == 0),  # Only verbose for first token
            )

            sg_lp_flat = sg_logprobs_curr.flatten()
            megatron_lp_flat = megatron_logprobs_curr.flatten()

            if sg_lp_flat.shape == megatron_lp_flat.shape:
                lp_stats = compute_diff_stats(sg_lp_flat, megatron_lp_flat)
                print(f"    Max diff:  {lp_stats['max_diff']}")
                print(f"    Mean diff: {lp_stats['mean_diff']}")

                if sg_target_lp is not None and megatron_target_lp is not None:
                    sg_lp_val = sg_target_lp.float().item()
                    megatron_lp_val = megatron_target_lp.float().item()
                    diff = abs(sg_lp_val - megatron_lp_val)
                    print(f"    Logprob for token {token_id}:")
                    print(f"      SGLang (production):  {sg_lp_val}")
                    print(f"      Megatron (production): {megatron_lp_val}")
                    print(f"      Diff:    {diff}")
                    if diff < 1e-5:
                        print(
                            f"      ✓ Response token #{token_idx+1} "
                            f"logprobs MATCH!"
                        )
                    else:
                        print(
                            f"      ✗ Response token #{token_idx+1} "
                            f"logprobs DIFFER!"
                        )

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"TRUE ON-POLICY VERIFICATION SUMMARY (Pass {megatron_pass_id:05d})")
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


def compare_first_response_token(
    sglang_dir: str,
    megatron_dir: str,
    verbose: bool = True,
) -> None:
    """
    Compare the first response token between SGLang and Megatron.

    This function handles multiple Megatron passes (e.g., Pass00000, Pass00002)
    and compares each with its corresponding SGLang pass.

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

    # Find all Megatron passes
    megatron_passes = list_all_passes(megatron_dir)
    if not megatron_passes:
        print(f"ERROR: No Megatron dump files found in {megatron_dir}")
        return

    print(f"Found {len(megatron_passes)} Megatron pass(es):")
    for pass_id, path in megatron_passes:
        info = get_megatron_dump_info(path)
        prompt_len = info.get('prompt_len', 'N/A')
        print(f"  Pass {pass_id:05d}: prompt_len={prompt_len}")

    # Process each Megatron pass
    for pass_idx, (megatron_pass_id, megatron_path) in enumerate(megatron_passes):
        if pass_idx > 0:
            # Add separator between passes
            print("\n" + "=" * 70)
            print("=" * 70)
            print("=" * 70)

        compare_single_pass_pair(
            megatron_pass_id=megatron_pass_id,
            megatron_path=megatron_path,
            sglang_dir=sglang_dir,
            verbose=verbose,
        )
        break

    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY - ALL PASSES")
    print("=" * 70)
    print(f"Compared {len(megatron_passes)} Megatron pass(es) with "
          f"corresponding SGLang passes.")
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
