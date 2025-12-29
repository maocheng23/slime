#!/usr/bin/env python3
"""
Check SGLang tensor dump passes to see what tokens are being processed.

Usage:
    python /root/slime/tools/check_sglang_passes.py /tmp/sglang_tensor_dump
"""

import sys
from pathlib import Path

import torch


def check_sglang_passes(dump_dir: str):
    """Check all SGLang pass files and display token information."""
    dump_path = Path(dump_dir)
    if not dump_path.exists():
        print(f"Error: Directory not found: {dump_dir}")
        return

    print(f"Checking SGLang passes in: {dump_dir}\n")
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

            # Check for forward batch info
            if "model.forward_batch_info.input_ids" in tensors:
                input_ids = tensors["model.forward_batch_info.input_ids"]
                print(f"  input_ids: {input_ids.tolist()}")
                print(f"  input_ids shape: {input_ids.shape}")

            if "model.forward_batch_info.positions" in tensors:
                positions = tensors["model.forward_batch_info.positions"]
                print(f"  positions: {positions.tolist()}")
                print(f"  positions shape: {positions.shape}")

            if "model.forward_batch_info.seq_lens" in tensors:
                seq_lens = tensors["model.forward_batch_info.seq_lens"]
                print(f"  seq_lens: {seq_lens.tolist()}")
                print(f"  seq_lens shape: {seq_lens.shape}")

            # Show available tensor keys
            tensor_keys = [
                k
                for k in tensors.keys()
                if isinstance(tensors[k], torch.Tensor)
                and "forward_batch_info" not in k
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
    print("\nNote:")
    print("  - Pass 0 is usually PREFILL (processes entire prompt)")
    print("  - Pass 1+ are DECODE passes (each generates one token)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_sglang_passes.py <sglang_dump_dir>")
        print("\nExample:")
        print("  python check_sglang_passes.py /tmp/sglang_tensor_dump")
        sys.exit(1)

    check_sglang_passes(sys.argv[1])

