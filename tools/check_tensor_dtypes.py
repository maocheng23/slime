#!/usr/bin/env python3
"""
Check tensor dtypes in SGLang and Megatron dumps to debug precision differences.

Usage:
    python check_tensor_dtypes.py --sglang-dir /tmp/sglang_dump --megatron-dir /tmp/megatron_dump
"""

import argparse
from pathlib import Path
from collections import defaultdict

import torch


def get_dtype_stats(tensors: dict[str, torch.Tensor]) -> dict[str, list[str]]:
    """Get dtype statistics for all tensors."""
    dtype_to_keys = defaultdict(list)
    for key, val in tensors.items():
        if isinstance(val, torch.Tensor):
            dtype_to_keys[str(val.dtype)].append(key)
        elif isinstance(val, (list, tuple)):
            for i, item in enumerate(val):
                if isinstance(item, torch.Tensor):
                    dtype_to_keys[str(item.dtype)].append(f"{key}[{i}]")
    return dict(dtype_to_keys)


def check_specific_tensors(tensors: dict[str, torch.Tensor], label: str):
    """Check dtype of specific important tensors."""
    print(f"\n  {label} - Key tensor dtypes:")

    important_patterns = [
        "final_layernorm", "model.norm", "norm",
        "layer_0_output", "layer_1_output", "layer_2_output",
        "model.layers.0", "model.layers.1", "model.layers.2",
        "lm_head", "logits", "embed"
    ]

    for pattern in important_patterns:
        matching_keys = [k for k in tensors.keys() if pattern in k.lower()]
        for key in matching_keys[:3]:  # Limit to first 3 matches per pattern
            val = tensors[key]
            if isinstance(val, torch.Tensor):
                print(f"    {key}: dtype={val.dtype}, shape={val.shape}")
            elif isinstance(val, (list, tuple)):
                print(f"    {key}: list[{len(val)}]")
                for i, item in enumerate(val):
                    if isinstance(item, torch.Tensor):
                        print(f"      [{i}]: dtype={item.dtype}, shape={item.shape}")


def compare_dtypes_for_matching_keys(
    sglang_tensors: dict[str, torch.Tensor],
    megatron_tensors: dict[str, torch.Tensor],
):
    """Compare dtypes for keys that should match between frameworks."""
    print("\n" + "=" * 70)
    print("DTYPE COMPARISON FOR MATCHING COMPONENTS")
    print("=" * 70)

    # Define matching pairs (sglang_key, megatron_key)
    matching_pairs = [
        ("model.norm", "final_layernorm"),
        ("model.norm", "final_layernorm_at_response_start"),
        ("lm_head", "lm_head"),
        ("model.embed_tokens", "model.embed_tokens"),
    ]

    # Also match layer outputs
    for i in range(10):
        matching_pairs.append((f"model.layers.{i}", f"layer_{i}_output"))
        matching_pairs.append((f"model.layers.{i}.input_layernorm", f"layer_{i}_input_layernorm_output"))
        matching_pairs.append((f"model.layers.{i}.mlp.down_proj", f"layer_{i}_mlp_output"))

    print("\n  Matching tensor dtype comparison:")
    for sg_key, meg_key in matching_pairs:
        sg_val = sglang_tensors.get(sg_key)
        meg_val = megatron_tensors.get(meg_key)

        if sg_val is None and meg_val is None:
            continue

        sg_dtype = "NOT FOUND"
        meg_dtype = "NOT FOUND"

        if sg_val is not None:
            if isinstance(sg_val, torch.Tensor):
                sg_dtype = str(sg_val.dtype)
            elif isinstance(sg_val, (list, tuple)) and len(sg_val) > 0:
                if isinstance(sg_val[0], torch.Tensor):
                    sg_dtype = f"list[{len(sg_val)}] of {sg_val[0].dtype}"

        if meg_val is not None:
            if isinstance(meg_val, torch.Tensor):
                meg_dtype = str(meg_val.dtype)
            elif isinstance(meg_val, (list, tuple)) and len(meg_val) > 0:
                if isinstance(meg_val[0], torch.Tensor):
                    meg_dtype = f"list[{len(meg_val)}] of {meg_val[0].dtype}"

        if sg_dtype != "NOT FOUND" or meg_dtype != "NOT FOUND":
            match_str = "✓" if sg_dtype == meg_dtype else "✗ MISMATCH"
            print(f"    {sg_key} <-> {meg_key}:")
            print(f"      SGLang: {sg_dtype}")
            print(f"      Megatron: {meg_dtype}")
            print(f"      Status: {match_str}")
            print()


def check_computation_precision(
    sglang_tensors: dict[str, torch.Tensor],
    megatron_tensors: dict[str, torch.Tensor],
):
    """Check if computations might be done in different precisions."""
    print("\n" + "=" * 70)
    print("DETAILED FINAL NORM ANALYSIS")
    print("=" * 70)

    # Get SGLang model.norm
    sg_norm = sglang_tensors.get("model.norm")
    meg_norm = megatron_tensors.get("final_layernorm_at_response_start") or \
               megatron_tensors.get("final_layernorm")

    if sg_norm is not None:
        print("\n  SGLang model.norm:")
        if isinstance(sg_norm, (list, tuple)):
            for i, item in enumerate(sg_norm):
                if isinstance(item, torch.Tensor):
                    print(f"    Element {i}:")
                    print(f"      dtype: {item.dtype}")
                    print(f"      shape: {item.shape}")
                    print(f"      device: {item.device}")

                    # Check value range and precision
                    item_f = item.float()
                    print(f"      min: {item_f.min().item():.6f}")
                    print(f"      max: {item_f.max().item():.6f}")
                    print(f"      mean: {item_f.mean().item():.6f}")
                    print(f"      std: {item_f.std().item():.6f}")

                    # Check for bf16 quantization artifacts
                    # bf16 has ~7 bits of mantissa, so values should be
                    # quantized to multiples of 2^(exponent-7)
                    unique_vals = torch.unique(item_f[:100] if item_f.numel() > 100 else item_f)
                    print(f"      unique values (first 100 elements): {len(unique_vals)}")

                    # Check if values look bf16 or fp16
                    if item.dtype == torch.bfloat16:
                        # Convert to fp32 and back to bf16, check if identical
                        round_trip = item.float().bfloat16()
                        is_exact_bf16 = torch.equal(item, round_trip)
                        print(f"      exact bf16 representation: {is_exact_bf16}")
                    elif item.dtype == torch.float16:
                        round_trip = item.float().half()
                        is_exact_fp16 = torch.equal(item, round_trip)
                        print(f"      exact fp16 representation: {is_exact_fp16}")
        elif isinstance(sg_norm, torch.Tensor):
            print(f"    dtype: {sg_norm.dtype}")
            print(f"    shape: {sg_norm.shape}")

    if meg_norm is not None:
        print("\n  Megatron final_layernorm:")
        if isinstance(meg_norm, torch.Tensor):
            print(f"    dtype: {meg_norm.dtype}")
            print(f"    shape: {meg_norm.shape}")
            print(f"    device: {meg_norm.device}")

            meg_f = meg_norm.float()
            print(f"    min: {meg_f.min().item():.6f}")
            print(f"    max: {meg_f.max().item():.6f}")
            print(f"    mean: {meg_f.mean().item():.6f}")
            print(f"    std: {meg_f.std().item():.6f}")

            if meg_norm.dtype == torch.bfloat16:
                round_trip = meg_norm.float().bfloat16()
                is_exact_bf16 = torch.equal(meg_norm, round_trip)
                print(f"    exact bf16 representation: {is_exact_bf16}")

    # Direct comparison if both exist
    if sg_norm is not None and meg_norm is not None:
        print("\n  Direct Comparison:")

        # Extract SGLang output (element 0 from tuple)
        if isinstance(sg_norm, (list, tuple)):
            sg_out = sg_norm[0]
        else:
            sg_out = sg_norm

        if isinstance(sg_out, torch.Tensor) and isinstance(meg_norm, torch.Tensor):
            # Flatten for comparison
            sg_flat = sg_out.flatten()
            meg_flat = meg_norm.flatten()

            # Take same size
            min_len = min(len(sg_flat), len(meg_flat))
            sg_flat = sg_flat[:min_len]
            meg_flat = meg_flat[:min_len]

            print(f"    SGLang dtype: {sg_out.dtype}")
            print(f"    Megatron dtype: {meg_norm.dtype}")

            # Check if bitwise identical
            if sg_out.dtype == meg_norm.dtype:
                # Compare raw bytes
                sg_bytes = sg_flat.view(torch.uint8 if sg_flat.dtype == torch.uint8 else torch.int16)
                meg_bytes = meg_flat.view(torch.uint8 if meg_flat.dtype == torch.uint8 else torch.int16)

                if sg_flat.dtype == torch.bfloat16:
                    sg_bytes = sg_flat.view(torch.int16)
                    meg_bytes = meg_flat.view(torch.int16)
                    byte_diff = (sg_bytes != meg_bytes).sum().item()
                    print(f"    Bitwise different elements: {byte_diff} / {min_len}")

                    # Find where differences occur
                    diff_indices = torch.where(sg_bytes != meg_bytes)[0]
                    if len(diff_indices) > 0:
                        print(f"    First 10 differing indices: {diff_indices[:10].tolist()}")
                        for idx in diff_indices[:5]:
                            idx = idx.item()
                            sg_val = sg_flat[idx].float().item()
                            meg_val = meg_flat[idx].float().item()
                            sg_int = sg_bytes[idx].item()
                            meg_int = meg_bytes[idx].item()
                            print(f"      idx {idx}: SGLang={sg_val:.6f} (0x{sg_int:04x}), "
                                  f"Megatron={meg_val:.6f} (0x{meg_int:04x}), "
                                  f"diff={abs(sg_val - meg_val):.6f}")

            # Float comparison
            sg_f = sg_flat.float()
            meg_f = meg_flat.float()
            diff = (sg_f - meg_f).abs()

            print(f"\n    Float comparison:")
            print(f"      Max diff: {diff.max().item():.6e}")
            print(f"      Mean diff: {diff.mean().item():.6e}")
            print(f"      Num exact matches: {(diff == 0).sum().item()} / {min_len}")
            print(f"      Num diff > 0.01: {(diff > 0.01).sum().item()}")
            print(f"      Num diff > 0.1: {(diff > 0.1).sum().item()}")


def find_dump_files(dump_dir: str) -> list[Path]:
    """Find all Pass*.pt files in the dump directory."""
    dump_path = Path(dump_dir)
    files = []

    # Search in subdirectories (TP*_PP*_Rank*_pid*)
    for subdir in dump_path.iterdir():
        if subdir.is_dir():
            files.extend(subdir.glob("Pass*.pt"))

    # Also search in the root directory
    files.extend(dump_path.glob("Pass*.pt"))

    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Check tensor dtypes in dumps")
    parser.add_argument("--sglang-dir", required=True, help="SGLang dump directory")
    parser.add_argument("--megatron-dir", required=True, help="Megatron dump directory")
    parser.add_argument("--pass-id", type=int, default=0, help="Pass ID to analyze")
    args = parser.parse_args()

    # Find dump files
    sglang_files = find_dump_files(args.sglang_dir)
    megatron_files = find_dump_files(args.megatron_dir)

    print("=" * 70)
    print("TENSOR DTYPE CHECKER")
    print("=" * 70)
    print(f"SGLang dir: {args.sglang_dir}")
    print(f"Megatron dir: {args.megatron_dir}")
    print(f"Pass ID: {args.pass_id}")

    # Filter to requested pass
    sglang_pass_files = [f for f in sglang_files if f"Pass{args.pass_id:05d}" in f.name]
    megatron_pass_files = [f for f in megatron_files if f"Pass{args.pass_id:05d}" in f.name]

    print(f"\nFound SGLang files: {[str(f) for f in sglang_pass_files]}")
    print(f"Found Megatron files: {[str(f) for f in megatron_pass_files]}")

    if not sglang_pass_files:
        print("ERROR: No SGLang dump files found!")
        return
    if not megatron_pass_files:
        print("ERROR: No Megatron dump files found!")
        return

    # Load tensors
    sglang_tensors = torch.load(sglang_pass_files[0], map_location="cpu")
    megatron_tensors = torch.load(megatron_pass_files[0], map_location="cpu")

    print(f"\nSGLang tensor count: {len(sglang_tensors)}")
    print(f"Megatron tensor count: {len(megatron_tensors)}")

    # Get dtype statistics
    print("\n" + "=" * 70)
    print("DTYPE DISTRIBUTION")
    print("=" * 70)

    sg_dtype_stats = get_dtype_stats(sglang_tensors)
    meg_dtype_stats = get_dtype_stats(megatron_tensors)

    print("\n  SGLang dtype distribution:")
    for dtype, keys in sorted(sg_dtype_stats.items()):
        print(f"    {dtype}: {len(keys)} tensors")
        if len(keys) <= 5:
            for k in keys:
                print(f"      - {k}")

    print("\n  Megatron dtype distribution:")
    for dtype, keys in sorted(meg_dtype_stats.items()):
        print(f"    {dtype}: {len(keys)} tensors")
        if len(keys) <= 5:
            for k in keys:
                print(f"      - {k}")

    # Check specific important tensors
    print("\n" + "=" * 70)
    print("IMPORTANT TENSOR DTYPES")
    print("=" * 70)
    check_specific_tensors(sglang_tensors, "SGLang")
    check_specific_tensors(megatron_tensors, "Megatron")

    # Compare dtypes for matching keys
    compare_dtypes_for_matching_keys(sglang_tensors, megatron_tensors)

    # Detailed final norm analysis
    check_computation_precision(sglang_tensors, megatron_tensors)


if __name__ == "__main__":
    main()
