#!/usr/bin/env python3
"""
Check norm parameters directly from Pass*.pt files.
This follows the same loading logic as compare_tensor_dumps_megatron.py

Usage:
    python check_norm_from_pass.py \
        --sglang-pass /tmp/sglang_tensor_dump/TP0_PP0_Rank0_pid274272/Pass00002.pt \
        --megatron-pass /tmp/megatron_tensor_dump/TP0_PP0_Rank0_pid274561/Pass00000.pt \
        --last-layer-idx 27
"""

import argparse
import sys
from pathlib import Path

import torch


def load_tensors(dump_file: Path) -> dict:
    """Load tensors from a dump file."""
    return torch.load(dump_file, map_location="cpu")


def extract_last_token(t):
    """Extract last token from tensor for comparison."""
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


def analyze_tensor_detailed(tensor, name: str):
    """Analyze a tensor in detail."""
    print(f"\n{'='*80}")
    print(f"📊 {name}")
    print('='*80)

    if isinstance(tensor, (tuple, list)):
        print(f"Type: tuple/list with {len(tensor)} elements\n")
        for i, elem in enumerate(tensor):
            if isinstance(elem, torch.Tensor):
                elem_f = elem.float()
                rms = (elem_f ** 2).mean().sqrt().item()

                # Extract last token for detailed view
                elem_last = extract_last_token(elem)
                elem_last_f = elem_last.float()

                if i == 0:
                    print(f"  🔵 Element {i} (OUTPUT - normalized after residual):")
                elif i == 1:
                    print(f"  🟢 Element {i} (INPUT - before normalization, after residual):")
                else:
                    print(f"  ⚪ Element {i}:")

                print(f"     Full shape: {elem.shape}")
                print(f"     Full dtype: {elem.dtype}")
                print(f"     Full RMS: {rms:.6e}")
                print(f"     Last token shape: {elem_last.shape}")
                print(f"     Last token RMS: {(elem_last_f ** 2).mean().sqrt().item():.6e}")
                print(f"     Last token mean: {elem_last_f.mean().item():.6e}")
                print(f"     Last token std: {elem_last_f.std().item():.6e}")
                print(f"     Last token [min, max]: [{elem_last_f.min().item():.6e}, {elem_last_f.max().item():.6e}]")

                if elem_last.numel() >= 10:
                    print(f"     First 10 values: {[f'{v:.4f}' for v in elem_last_f[:10].tolist()]}")
                print()

    elif isinstance(tensor, torch.Tensor):
        tensor_f = tensor.float()
        rms = (tensor_f ** 2).mean().sqrt().item()

        # Extract last token
        tensor_last = extract_last_token(tensor)
        tensor_last_f = tensor_last.float()

        print(f"Type: single tensor\n")
        print(f"Full shape: {tensor.shape}")
        print(f"Full dtype: {tensor.dtype}")
        print(f"Full RMS: {rms:.6e}")
        print(f"Last token shape: {tensor_last.shape}")
        print(f"Last token RMS: {(tensor_last_f ** 2).mean().sqrt().item():.6e}")
        print(f"Last token mean: {tensor_last_f.mean().item():.6e}")
        print(f"Last token std: {tensor_last_f.std().item():.6e}")
        print(f"Last token [min, max]: [{tensor_last_f.min().item():.6e}, {tensor_last_f.max().item():.6e}]")

        if tensor_last.numel() >= 10:
            print(f"First 10 values: {[f'{v:.4f}' for v in tensor_last_f[:10].tolist()]}")
    else:
        print(f"Type: {type(tensor)} (not a tensor)")


def compare_norm_outputs(sglang_tensor, megatron_tensor, layer_idx: int):
    """Compare norm output tensors."""
    print("\n" + "=" * 80)
    print("🔬 NORM OUTPUT COMPARISON")
    print("=" * 80)
    print(f"Comparing: SGLang model.norm vs Megatron final_layernorm")
    print("=" * 80)

    # Extract SGLang output (Element 0)
    if isinstance(sglang_tensor, (tuple, list)) and len(sglang_tensor) >= 2:
        sg_output = sglang_tensor[0]  # Element 0: normalized output
        sg_input = sglang_tensor[1]   # Element 1: input before norm
        print("\n✅ SGLang: Extracted Element 0 (OUTPUT) and Element 1 (INPUT)")
    elif isinstance(sglang_tensor, (tuple, list)) and len(sglang_tensor) == 1:
        sg_output = sglang_tensor[0]
        sg_input = None
        print("\n⚠️  SGLang: Only Element 0 available")
    else:
        sg_output = sglang_tensor
        sg_input = None
        print("\n⚠️  SGLang: Single tensor (not tuple)")

    if isinstance(megatron_tensor, torch.Tensor):
        meg_output = megatron_tensor
        print("✅ Megatron: Single tensor (OUTPUT)")
    else:
        print("❌ Megatron: Unexpected format")
        return

    # Extract last tokens
    sg_out_last = extract_last_token(sg_output).float()
    meg_out_last = extract_last_token(meg_output).float()

    if sg_input is not None:
        sg_in_last = extract_last_token(sg_input).float()
    else:
        sg_in_last = None

    # Ensure same length
    if sg_out_last.shape != meg_out_last.shape:
        min_len = min(sg_out_last.numel(), meg_out_last.numel())
        sg_out_last = sg_out_last.flatten()[:min_len]
        meg_out_last = meg_out_last.flatten()[:min_len]
        print(f"\n⚠️  Shape mismatch! Using first {min_len} elements")

    # Compute differences
    diff = (sg_out_last - meg_out_last).abs()
    rel_diff = diff / (sg_out_last.abs() + 1e-8)

    print("\n" + "-" * 80)
    print("📈 STATISTICS")
    print("-" * 80)
    print(f"Vector length: {len(sg_out_last)}")
    print(f"\nSGLang Element 0 (OUTPUT):")
    print(f"  Shape: {sg_output.shape}, dtype: {sg_output.dtype}")
    print(f"  Last token RMS: {(sg_out_last ** 2).mean().sqrt().item():.6e}")
    print(f"\nMegatron final_layernorm:")
    print(f"  Shape: {meg_output.shape}, dtype: {meg_output.dtype}")
    print(f"  Last token RMS: {(meg_out_last ** 2).mean().sqrt().item():.6e}")
    print(f"\nDifferences (SGLang - Megatron):")
    print(f"  Max absolute diff: {diff.max().item():.6e}")
    print(f"  Mean absolute diff: {diff.mean().item():.6e}")
    print(f"  Max relative diff: {rel_diff.max().item():.6e}")
    print(f"  Mean relative diff: {rel_diff.mean().item():.6e}")

    # Tolerance checks
    is_close_loose = torch.allclose(sg_out_last, meg_out_last, rtol=1e-4, atol=1e-6)
    is_close_normal = torch.allclose(sg_out_last, meg_out_last, rtol=1e-5, atol=1e-8)
    is_close_strict = torch.allclose(sg_out_last, meg_out_last, rtol=1e-6, atol=1e-9)

    print(f"\n{'='*80}")
    print("✅ TOLERANCE CHECKS")
    print('='*80)
    if is_close_strict:
        print("✅ VERY CLOSE (rtol=1e-6, atol=1e-9)")
    elif is_close_normal:
        print("✅ CLOSE (rtol=1e-5, atol=1e-8)")
    elif is_close_loose:
        print("⚠️  LOOSELY CLOSE (rtol=1e-4, atol=1e-6)")
    else:
        print("❌ SIGNIFICANTLY DIFFERENT")

    # Side-by-side comparison (first 20)
    print(f"\n{'='*80}")
    print("📊 FIRST 20 VALUES (Last Token)")
    print('='*80)
    print(f"{'Idx':<6} {'SGLang':<15} {'Megatron':<15} {'Abs Diff':<13} {'Rel Diff':<13}")
    print("-" * 80)
    for i in range(min(20, len(sg_out_last))):
        sg_val = sg_out_last[i].item()
        meg_val = meg_out_last[i].item()
        abs_d = diff[i].item()
        rel_d = rel_diff[i].item()
        print(f"{i:<6} {sg_val:<15.6e} {meg_val:<15.6e} {abs_d:<13.6e} {rel_d:<13.6e}")

    # Top differences
    print(f"\n{'='*80}")
    print("🔝 TOP 10 LARGEST ABSOLUTE DIFFERENCES")
    print('='*80)
    top_indices = diff.topk(min(10, len(diff))).indices
    print(f"{'Idx':<6} {'SGLang':<15} {'Megatron':<15} {'Abs Diff':<13} {'Rel Diff':<13}")
    print("-" * 80)
    for idx in top_indices:
        i = idx.item()
        sg_val = sg_out_last[i].item()
        meg_val = meg_out_last[i].item()
        abs_d = diff[i].item()
        rel_d = rel_diff[i].item()
        print(f"{i:<6} {sg_val:<15.6e} {meg_val:<15.6e} {abs_d:<13.6e} {rel_d:<13.6e}")

    # Show Element 1 info if available
    if sg_in_last is not None:
        print(f"\n{'='*80}")
        print("📊 SGLANG ELEMENT 1 (INPUT before normalization)")
        print('='*80)
        print(f"Last token RMS: {(sg_in_last ** 2).mean().sqrt().item():.6e}")
        print(f"Last token mean: {sg_in_last.mean().item():.6e}")
        print(f"Last token std: {sg_in_last.std().item():.6e}")
        print(f"First 10 values: {[f'{v:.4f}' for v in sg_in_last[:10].tolist()]}")


def main():
    parser = argparse.ArgumentParser(description="Check norm from Pass*.pt files")
    parser.add_argument("--sglang-pass", required=True, help="SGLang Pass*.pt file")
    parser.add_argument("--megatron-pass", required=True, help="Megatron Pass*.pt file")
    parser.add_argument("--last-layer-idx", type=int, default=27,
                       help="Last layer index (default: 27 for 28-layer model)")

    args = parser.parse_args()

    sglang_path = Path(args.sglang_pass)
    megatron_path = Path(args.megatron_pass)

    if not sglang_path.exists():
        print(f"❌ SGLang pass file not found: {sglang_path}")
        sys.exit(1)

    if not megatron_path.exists():
        print(f"❌ Megatron pass file not found: {megatron_path}")
        sys.exit(1)

    print("=" * 80)
    print("🔍 NORM CHECKER (from Pass files)")
    print("=" * 80)
    print(f"\nSGLang pass: {sglang_path}")
    print(f"Megatron pass: {megatron_path}")
    print(f"Last layer index: {args.last_layer_idx}")

    # Load tensors
    print("\n" + "=" * 80)
    print("📂 LOADING TENSORS")
    print("=" * 80)

    print("\nLoading SGLang tensors...")
    sglang_tensors = load_tensors(sglang_path)
    print(f"✅ Loaded {len(sglang_tensors)} SGLang tensors")

    print("\nLoading Megatron tensors...")
    megatron_tensors = load_tensors(megatron_path)
    print(f"✅ Loaded {len(megatron_tensors)} Megatron tensors")

    # Find norm tensors
    print("\n" + "=" * 80)
    print("🔍 SEARCHING FOR NORM TENSORS")
    print("=" * 80)

    # SGLang: look for "model.norm"
    sglang_norm = None
    for key in ["model.norm", "norm"]:
        if key in sglang_tensors:
            sglang_norm = sglang_tensors[key]
            print(f"\n✅ Found SGLang norm: '{key}'")
            break

    if sglang_norm is None:
        print("\n❌ Could not find SGLang model.norm")
        print(f"Available keys with 'norm': {[k for k in sglang_tensors.keys() if 'norm' in k.lower()]}")
    else:
        analyze_tensor_detailed(sglang_norm, "SGLang model.norm")

    # Megatron: look for "final_layernorm" or last layer output
    megatron_norm = None
    for key_pattern in [
        "final_layernorm_at_response_start",
        f"final_layernorm_pos_*",
        "final_layernorm",
    ]:
        for key in megatron_tensors.keys():
            if "final_layernorm" in key:
                megatron_norm = megatron_tensors[key]
                print(f"\n✅ Found Megatron norm: '{key}'")
                break
        if megatron_norm is not None:
            break

    if megatron_norm is None:
        print("\n❌ Could not find Megatron final_layernorm")
        print(f"Available keys with 'norm': {[k for k in megatron_tensors.keys() if 'norm' in k.lower()]}")
    else:
        analyze_tensor_detailed(megatron_norm, "Megatron final_layernorm")

    # Also check last layer output
    print("\n" + "=" * 80)
    print(f"🔍 CHECKING LAST LAYER OUTPUT (layer_{args.last_layer_idx})")
    print("=" * 80)

    megatron_last_layer = None
    for key_pattern in [
        f"layer_{args.last_layer_idx}_output_at_response_start",
        f"layer_{args.last_layer_idx}_output",
    ]:
        if key_pattern in megatron_tensors:
            megatron_last_layer = megatron_tensors[key_pattern]
            print(f"\n✅ Found Megatron last layer: '{key_pattern}'")
            analyze_tensor_detailed(megatron_last_layer, f"Megatron layer_{args.last_layer_idx}_output")
            break

    if megatron_last_layer is None:
        print(f"\n⚠️  Could not find Megatron layer_{args.last_layer_idx}_output")

    # Compare if both found
    if sglang_norm is not None and megatron_norm is not None:
        compare_norm_outputs(sglang_norm, megatron_norm, args.last_layer_idx)
    else:
        print("\n⚠️  Cannot compare: one or both norm tensors not found")

    print("\n" + "=" * 80)
    print("✅ DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
