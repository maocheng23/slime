#!/usr/bin/env python3
"""
Check Megatron tensor dump passes to see what tokens are being processed.

Usage:
    python check_megatron_passes.py /tmp/megatron_tensor_dump
"""

import sys
from pathlib import Path

import torch


def check_megatron_passes(dump_dir: str):
    """Check all Megatron pass files and display token information."""
    dump_path = Path(dump_dir)
    if not dump_path.exists():
        print(f"Error: Directory not found: {dump_dir}")
        return

    print(f"Checking Megatron passes in: {dump_dir}\n")
    print("=" * 80)

    passes = []
    for f in sorted(dump_path.glob("*/Pass*.pt")):
        # Extract pass ID from filename like "Pass00000.pt"
        name = f.stem
        if name.startswith("Pass"):
            try:
                pass_id = int(name[4:])
                passes.append((pass_id, f))
            except ValueError:
                continue

    if not passes:
        print("No pass files found!")
        return

    print(f"Found {len(passes)} pass file(s):\n")

    for pass_id, path in passes:
        try:
            tensors = torch.load(path, map_location="cpu")

            print(f"Pass {pass_id:05d} ({path.name}):")
            print(f"  File: {path}")

            # Check for input token info
            if "megatron_input_ids" in tensors:
                input_ids = tensors["megatron_input_ids"]
                flat_ids = input_ids.flatten()
                print(f"  Full input_ids shape: {input_ids.shape}")
                print(f"  Full input_ids (first 20): {flat_ids[:20].tolist()}")

            if "megatron_compared_token_id" in tensors:
                compared_token = tensors["megatron_compared_token_id"]
                print(f"  Compared token ID: {compared_token.item()}")

            if "megatron_compared_position" in tensors:
                compared_pos = tensors["megatron_compared_position"]
                print(f"  Compared position: {compared_pos.item()}")

            # Debug info
            if "debug_prompt_len" in tensors:
                prompt_len = tensors["debug_prompt_len"].item()
                print(f"  Prompt length: {prompt_len}")

            if "debug_total_len" in tensors:
                total_len = tensors["debug_total_len"].item()
                print(f"  Total length: {total_len}")

            if "debug_response_len" in tensors:
                response_len = tensors["debug_response_len"].item()
                print(f"  Response length: {response_len}")

            # Show available tensor keys
            tensor_keys = [
                k
                for k in tensors.keys()
                if isinstance(tensors[k], torch.Tensor)
                and "input" not in k.lower()
                and "debug" not in k.lower()
            ]
            print(f"  Available tensors: {len(tensor_keys)}")
            print(f"    Examples: {', '.join(sorted(tensor_keys)[:5])}")

            print()

        except Exception as e:
            print(f"Pass {pass_id:05d}: Error loading - {e}\n")

    print("=" * 80)
    print("\nSummary:")
    print(f"  Total passes: {len(passes)}")
    print(f"  Pass range: {passes[0][0]} to {passes[-1][0]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_megatron_passes.py <megatron_dump_dir>")
        print("\nExample:")
        print("  python check_megatron_passes.py /tmp/megatron_tensor_dump")
        sys.exit(1)

    check_megatron_passes(sys.argv[1])

