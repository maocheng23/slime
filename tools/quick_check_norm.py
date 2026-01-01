#!/usr/bin/env python3
"""
Quick checker for norm differences using existing tensor dumps.
Extracts and compares norm computation details from SGLang and Megatron dumps.

Usage:
    python quick_check_norm.py \\
        --sglang-dump /tmp/sglang_dump \\
        --megatron-dump /tmp/megatron_dump
"""

import argparse
import sys
from pathlib import Path

import torch


def load_tensor_dump(dump_path: Path, pattern: str):
    """Load tensors matching pattern from dump directory."""
    tensors = {}
    for file in dump_path.glob(pattern):
        try:
            tensor = torch.load(file, map_location='cpu')
            tensors[file.name] = tensor
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")
    return tensors


def analyze_tensor(tensor, name: str):
    """Analyze a single tensor or tuple of tensors."""
    print(f"\n📊 {name}")
    print("-" * 60)

    if isinstance(tensor, (tuple, list)):
        print(f"Type: tuple/list with {len(tensor)} elements")
        for i, elem in enumerate(tensor):
            if isinstance(elem, torch.Tensor):
                elem_f = elem.float()
                rms = (elem_f ** 2).mean().sqrt().item()
                print(f"\n  Element {i}:")
                print(f"    Shape: {elem.shape}")
                print(f"    Dtype: {elem.dtype}")
                print(f"    RMS: {rms:.6e}")
                print(f"    Mean: {elem_f.mean().item():.6e}")
                print(f"    Std: {elem_f.std().item():.6e}")
                print(f"    Min: {elem_f.min().item():.6e}")
                print(f"    Max: {elem_f.max().item():.6e}")
                if elem.numel() >= 10:
                    print(f"    First 10: {[f'{v:.4f}' for v in elem_f.flatten()[:10].tolist()]}")
    elif isinstance(tensor, torch.Tensor):
        tensor_f = tensor.float()
        rms = (tensor_f ** 2).mean().sqrt().item()
        print(f"Type: tensor")
        print(f"Shape: {tensor.shape}")
        print(f"Dtype: {tensor.dtype}")
        print(f"RMS: {rms:.6e}")
        print(f"Mean: {tensor_f.mean().item():.6e}")
        print(f"Std: {tensor_f.std().item():.6e}")
        print(f"Min: {tensor_f.min().item():.6e}")
        print(f"Max: {tensor_f.max().item():.6e}")
        if tensor.numel() >= 10:
            print(f"First 10: {[f'{v:.4f}' for v in tensor_f.flatten()[:10].tolist()]}")
    else:
        print(f"Type: {type(tensor)}")


def compare_norm_outputs(sglang_tensor, megatron_tensor):
    """Compare norm output tensors."""
    print("\n" + "=" * 80)
    print("🔍 DETAILED COMPARISON: SGLang vs Megatron Norm Outputs")
    print("=" * 80)

    # Extract outputs
    if isinstance(sglang_tensor, (tuple, list)) and len(sglang_tensor) >= 2:
        sg_output = sglang_tensor[0]  # Element 0: normalized output
        sg_input = sglang_tensor[1]   # Element 1: input before norm
        print("\n✅ SGLang: Found tuple with 2 elements (output, input)")
    elif isinstance(sglang_tensor, (tuple, list)) and len(sglang_tensor) == 1:
        sg_output = sglang_tensor[0]
        sg_input = None
        print("\n⚠️  SGLang: Found tuple with 1 element (only output)")
    else:
        sg_output = sglang_tensor
        sg_input = None
        print("\n⚠️  SGLang: Found single tensor (only output)")

    if isinstance(megatron_tensor, torch.Tensor):
        meg_output = megatron_tensor
        print("✅ Megatron: Found single tensor (output)")
    else:
        print("❌ Megatron: Unexpected format")
        return

    # Ensure tensors are 1D for comparison (extract last token if needed)
    def extract_last_token(t):
        if t.dim() == 0:
            return t
        elif t.dim() == 1:
            return t
        elif t.dim() == 2:
            return t[-1]  # [seq, hidden] -> [hidden]
        elif t.dim() == 3:
            if t.shape[0] == 1:
                return t[0, -1]  # [1, seq, hidden] -> [hidden]
            else:
                return t[-1, 0]  # [seq, 1, hidden] -> [hidden]
        else:
            return t.flatten()

    sg_out_1d = extract_last_token(sg_output).float()
    meg_out_1d = extract_last_token(meg_output).float()

    if sg_input is not None:
        sg_in_1d = extract_last_token(sg_input).float()
    else:
        sg_in_1d = None

    # Compare outputs
    print("\n" + "-" * 80)
    print("NORM OUTPUT COMPARISON (Element 0 vs final_layernorm)")
    print("-" * 80)

    print(f"\nSGLang output shape: {sg_out_1d.shape}, dtype: {sg_output.dtype}")
    print(f"Megatron output shape: {meg_out_1d.shape}, dtype: {meg_output.dtype}")

    if sg_out_1d.shape != meg_out_1d.shape:
        min_len = min(sg_out_1d.numel(), meg_out_1d.numel())
        sg_out_1d = sg_out_1d.flatten()[:min_len]
        meg_out_1d = meg_out_1d.flatten()[:min_len]
        print(f"⚠️  Shape mismatch! Using first {min_len} elements")

    diff = (sg_out_1d - meg_out_1d).abs()
    rel_diff = diff / (sg_out_1d.abs() + 1e-8)

    print(f"\n📈 Statistics:")
    print(f"  SGLang RMS:  {(sg_out_1d ** 2).mean().sqrt().item():.6e}")
    print(f"  Megatron RMS: {(meg_out_1d ** 2).mean().sqrt().item():.6e}")
    print(f"  Max abs diff: {diff.max().item():.6e}")
    print(f"  Mean abs diff: {diff.mean().item():.6e}")
    print(f"  Max rel diff: {rel_diff.max().item():.6e}")
    print(f"  Mean rel diff: {rel_diff.mean().item():.6e}")

    # Check if close
    is_close = torch.allclose(sg_out_1d, meg_out_1d, rtol=1e-5, atol=1e-8)
    is_very_close = torch.allclose(sg_out_1d, meg_out_1d, rtol=1e-6, atol=1e-9)

    if is_very_close:
        print("\n✅ Outputs are VERY CLOSE (rtol=1e-6, atol=1e-9)")
    elif is_close:
        print("\n✅ Outputs are CLOSE (rtol=1e-5, atol=1e-8)")
    else:
        print("\n❌ Outputs DIFFER significantly!")

    # Show first 20 values side by side
    print(f"\n📊 First 20 values comparison:")
    print(f"{'Index':<8} {'SGLang':<15} {'Megatron':<15} {'Abs Diff':<15} {'Rel Diff':<15}")
    print("-" * 75)
    for i in range(min(20, len(sg_out_1d))):
        sg_val = sg_out_1d[i].item()
        meg_val = meg_out_1d[i].item()
        abs_diff = abs(sg_val - meg_val)
        rel_diff = abs_diff / (abs(sg_val) + 1e-8)
        print(f"{i:<8} {sg_val:<15.6e} {meg_val:<15.6e} {abs_diff:<15.6e} {rel_diff:<15.6e}")

    # Find top differences
    print(f"\n🔝 Top 10 largest absolute differences:")
    top_indices = diff.topk(min(10, len(diff))).indices
    print(f"{'Index':<8} {'SGLang':<15} {'Megatron':<15} {'Abs Diff':<15} {'Rel Diff':<15}")
    print("-" * 75)
    for idx in top_indices:
        i = idx.item()
        sg_val = sg_out_1d[i].item()
        meg_val = meg_out_1d[i].item()
        abs_diff = diff[i].item()
        rel_diff = rel_diff[i].item()
        print(f"{i:<8} {sg_val:<15.6e} {meg_val:<15.6e} {abs_diff:<15.6e} {rel_diff:<15.6e}")

    # Compare inputs if available
    if sg_in_1d is not None:
        print("\n" + "-" * 80)
        print("NORM INPUT (Element 1)")
        print("-" * 80)
        print(f"SGLang input RMS: {(sg_in_1d ** 2).mean().sqrt().item():.6e}")
        print(f"First 10 values: {[f'{v:.4f}' for v in sg_in_1d[:10].tolist()]}")


def main():
    parser = argparse.ArgumentParser(description="Quick norm check from tensor dumps")
    parser.add_argument("--sglang-dump", required=True, help="SGLang tensor dump directory")
    parser.add_argument("--megatron-dump", required=True, help="Megatron tensor dump directory")

    args = parser.parse_args()

    sglang_path = Path(args.sglang_dump)
    megatron_path = Path(args.megatron_dump)

    if not sglang_path.exists():
        print(f"❌ SGLang dump not found: {sglang_path}")
        sys.exit(1)

    if not megatron_path.exists():
        print(f"❌ Megatron dump not found: {megatron_path}")
        sys.exit(1)

    print("=" * 80)
    print("🔍 QUICK NORM CHECKER")
    print("=" * 80)
    print(f"\nSGLang dump: {sglang_path}")
    print(f"Megatron dump: {megatron_path}")

    # Load norm tensors
    print("\n" + "=" * 80)
    print("📂 LOADING TENSORS")
    print("=" * 80)

    sglang_norms = load_tensor_dump(sglang_path, "*norm*.pt")
    megatron_norms = load_tensor_dump(megatron_path, "*norm*.pt")

    print(f"\n✅ Found {len(sglang_norms)} SGLang norm tensors")
    print(f"✅ Found {len(megatron_norms)} Megatron norm tensors")

    # Analyze SGLang
    print("\n" + "=" * 80)
    print("📊 SGLANG NORM TENSORS")
    print("=" * 80)
    for name, tensor in sglang_norms.items():
        analyze_tensor(tensor, name)

    # Analyze Megatron
    print("\n" + "=" * 80)
    print("📊 MEGATRON NORM TENSORS")
    print("=" * 80)
    for name, tensor in megatron_norms.items():
        analyze_tensor(tensor, name)

    # Try to find and compare final norm outputs
    # Look for model.norm in SGLang and final_layernorm in Megatron
    sglang_final_norm = None
    for name, tensor in sglang_norms.items():
        if "model.norm" in name or "model_norm" in name:
            sglang_final_norm = tensor
            print(f"\n✅ Found SGLang final norm: {name}")
            break

    megatron_final_norm = None
    for name, tensor in megatron_norms.items():
        if "final_layernorm" in name or "final_norm" in name:
            megatron_final_norm = tensor
            print(f"✅ Found Megatron final norm: {name}")
            break

    if sglang_final_norm is not None and megatron_final_norm is not None:
        compare_norm_outputs(sglang_final_norm, megatron_final_norm)
    else:
        print("\n⚠️  Could not find both final norm tensors for comparison")
        if sglang_final_norm is None:
            print("   Missing: SGLang final norm (looking for 'model.norm')")
        if megatron_final_norm is None:
            print("   Missing: Megatron final norm (looking for 'final_layernorm')")


if __name__ == "__main__":
    main()
