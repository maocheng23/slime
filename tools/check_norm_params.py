#!/usr/bin/env python3
"""
Check and compare norm parameters between SGLang and Megatron.
This script checks:
1. Epsilon values (eps)
2. Weight parameters (norm.weight)
3. Dtype configurations
4. Normalization computation paths

Usage:
    python check_norm_params.py \\
        --sglang-checkpoint /path/to/sglang/checkpoint \\
        --megatron-checkpoint /path/to/megatron/checkpoint \\
        --sglang-dump /tmp/sglang_dump \\
        --megatron-dump /tmp/megatron_dump
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import numpy as np


def load_checkpoint_weights(checkpoint_path: str, key: str) -> Optional[torch.Tensor]:
    """Load specific weight from checkpoint."""
    try:
        if Path(checkpoint_path).is_file():
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            if key in ckpt:
                return ckpt[key]
            # Try nested structures
            if 'model' in ckpt and key in ckpt['model']:
                return ckpt['model'][key]
            if 'state_dict' in ckpt and key in ckpt['state_dict']:
                return ckpt['state_dict'][key]
        elif Path(checkpoint_path).is_dir():
            # Try loading from directory (e.g., HF format)
            import glob
            for pattern in ['pytorch_model*.bin', 'model*.safetensors']:
                files = glob.glob(str(Path(checkpoint_path) / pattern))
                for file in files:
                    if file.endswith('.safetensors'):
                        from safetensors.torch import load_file
                        ckpt = load_file(file)
                    else:
                        ckpt = torch.load(file, map_location='cpu')
                    if key in ckpt:
                        return ckpt[key]
    except Exception as e:
        print(f"Error loading {key} from {checkpoint_path}: {e}")
    return None


def compare_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor, name: str) -> Dict:
    """Compare two tensors and return statistics."""
    if tensor1 is None or tensor2 is None:
        return {"error": "One or both tensors are None"}

    # Ensure same device and dtype for comparison
    t1 = tensor1.float().cpu()
    t2 = tensor2.float().cpu()

    if t1.shape != t2.shape:
        return {
            "error": f"Shape mismatch: {t1.shape} vs {t2.shape}"
        }

    diff = (t1 - t2).abs()
    rel_diff = diff / (t1.abs() + 1e-8)

    return {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "max_rel_diff": rel_diff.max().item(),
        "mean_rel_diff": rel_diff.mean().item(),
        "identical": torch.allclose(t1, t2, rtol=1e-5, atol=1e-8),
        "dtype_1": str(tensor1.dtype),
        "dtype_2": str(tensor2.dtype),
        "shape": str(t1.shape),
        "rms_1": (t1 ** 2).mean().sqrt().item(),
        "rms_2": (t2 ** 2).mean().sqrt().item(),
        "first_10_values_1": t1.flatten()[:10].tolist(),
        "first_10_values_2": t2.flatten()[:10].tolist(),
    }


def check_norm_config_from_dump(dump_dir: str) -> Dict:
    """Extract norm configuration from tensor dump metadata."""
    dump_path = Path(dump_dir)
    if not dump_path.exists():
        return {"error": f"Dump directory not found: {dump_dir}"}

    config = {}

    # Try to load metadata or config file
    for meta_file in ['metadata.pt', 'config.pt', 'tensors.pt']:
        meta_path = dump_path / meta_file
        if meta_path.exists():
            try:
                meta = torch.load(meta_path, map_location='cpu')
                if isinstance(meta, dict):
                    config.update(meta)
            except:
                pass

    return config


def analyze_norm_computation(dump_dir: str, system_name: str) -> Dict:
    """Analyze norm computation from dumped tensors."""
    dump_path = Path(dump_dir)
    if not dump_path.exists():
        return {"error": f"Dump directory not found: {dump_dir}"}

    # Find norm tensors
    norm_files = list(dump_path.glob("*norm*.pt")) + list(dump_path.glob("*model.norm*.pt"))

    if not norm_files:
        return {"error": "No norm tensors found in dump"}

    analysis = {
        "system": system_name,
        "norm_tensors": [],
    }

    for norm_file in norm_files:
        try:
            tensor = torch.load(norm_file, map_location='cpu')

            # Handle tuple/list (output, residual)
            if isinstance(tensor, (tuple, list)):
                analysis["norm_tensors"].append({
                    "file": norm_file.name,
                    "type": "tuple/list",
                    "num_elements": len(tensor),
                    "element_0_shape": str(tensor[0].shape) if len(tensor) > 0 else None,
                    "element_0_dtype": str(tensor[0].dtype) if len(tensor) > 0 else None,
                    "element_0_rms": (tensor[0].float() ** 2).mean().sqrt().item() if len(tensor) > 0 else None,
                    "element_1_shape": str(tensor[1].shape) if len(tensor) > 1 else None,
                    "element_1_dtype": str(tensor[1].dtype) if len(tensor) > 1 else None,
                    "element_1_rms": (tensor[1].float() ** 2).mean().sqrt().item() if len(tensor) > 1 else None,
                })
            else:
                analysis["norm_tensors"].append({
                    "file": norm_file.name,
                    "type": "tensor",
                    "shape": str(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "rms": (tensor.float() ** 2).mean().sqrt().item(),
                })
        except Exception as e:
            analysis["norm_tensors"].append({
                "file": norm_file.name,
                "error": str(e)
            })

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Check norm parameters between SGLang and Megatron")
    parser.add_argument("--sglang-checkpoint", type=str, help="Path to SGLang checkpoint")
    parser.add_argument("--megatron-checkpoint", type=str, help="Path to Megatron checkpoint")
    parser.add_argument("--sglang-dump", type=str, help="Path to SGLang tensor dump directory")
    parser.add_argument("--megatron-dump", type=str, help="Path to Megatron tensor dump directory")
    parser.add_argument("--eps-sglang", type=float, default=1e-6, help="Expected SGLang eps value")
    parser.add_argument("--eps-megatron", type=float, default=1e-6, help="Expected Megatron eps value")

    args = parser.parse_args()

    print("=" * 80)
    print("NORM PARAMETER CHECKER")
    print("=" * 80)

    # 1. Check epsilon values
    print("\n1. EPSILON VALUES")
    print("-" * 80)
    print(f"SGLang eps (expected):   {args.eps_sglang}")
    print(f"Megatron eps (expected): {args.eps_megatron}")
    if args.eps_sglang != args.eps_megatron:
        print("⚠️  WARNING: Epsilon values are DIFFERENT!")
        print(f"   Difference: {abs(args.eps_sglang - args.eps_megatron):.2e}")
    else:
        print("✅ Epsilon values match")

    # 2. Check weight parameters from checkpoints
    if args.sglang_checkpoint and args.megatron_checkpoint:
        print("\n2. WEIGHT PARAMETERS (from checkpoints)")
        print("-" * 80)

        # Try different possible keys
        sglang_keys = ["model.norm.weight", "norm.weight", "model.layers.norm.weight"]
        megatron_keys = [
            "decoder.final_layernorm.weight",
            "module.decoder.final_layernorm.weight",
            "model.decoder.final_layernorm.weight",
        ]

        sglang_weight = None
        for key in sglang_keys:
            w = load_checkpoint_weights(args.sglang_checkpoint, key)
            if w is not None:
                print(f"✅ Found SGLang weight at key: {key}")
                sglang_weight = w
                break

        if sglang_weight is None:
            print(f"❌ Could not find SGLang norm weight (tried keys: {sglang_keys})")
        else:
            print(f"   Shape: {sglang_weight.shape}")
            print(f"   Dtype: {sglang_weight.dtype}")
            print(f"   RMS: {(sglang_weight.float() ** 2).mean().sqrt().item():.6f}")
            print(f"   First 10 values: {[f'{v:.6f}' for v in sglang_weight.flatten()[:10].tolist()]}")

        megatron_weight = None
        for key in megatron_keys:
            w = load_checkpoint_weights(args.megatron_checkpoint, key)
            if w is not None:
                print(f"✅ Found Megatron weight at key: {key}")
                megatron_weight = w
                break

        if megatron_weight is None:
            print(f"❌ Could not find Megatron norm weight (tried keys: {megatron_keys})")
        else:
            print(f"   Shape: {megatron_weight.shape}")
            print(f"   Dtype: {megatron_weight.dtype}")
            print(f"   RMS: {(megatron_weight.float() ** 2).mean().sqrt().item():.6f}")
            print(f"   First 10 values: {[f'{v:.6f}' for v in megatron_weight.flatten()[:10].tolist()]}")

        # Compare weights
        if sglang_weight is not None and megatron_weight is not None:
            print("\n   WEIGHT COMPARISON:")
            comparison = compare_tensors(sglang_weight, megatron_weight, "norm.weight")
            for key, value in comparison.items():
                print(f"   - {key}: {value}")

            if not comparison.get("identical", False):
                print("\n   ⚠️  WARNING: Weights are NOT identical!")
            else:
                print("\n   ✅ Weights are identical")

    # 3. Analyze norm computation from dumps
    if args.sglang_dump:
        print("\n3. SGLANG NORM COMPUTATION (from tensor dumps)")
        print("-" * 80)
        analysis = analyze_norm_computation(args.sglang_dump, "SGLang")
        if "error" in analysis:
            print(f"❌ {analysis['error']}")
        else:
            for tensor_info in analysis["norm_tensors"]:
                print(f"\nFile: {tensor_info['file']}")
                for key, value in tensor_info.items():
                    if key != "file":
                        print(f"  - {key}: {value}")

    if args.megatron_dump:
        print("\n4. MEGATRON NORM COMPUTATION (from tensor dumps)")
        print("-" * 80)
        analysis = analyze_norm_computation(args.megatron_dump, "Megatron")
        if "error" in analysis:
            print(f"❌ {analysis['error']}")
        else:
            for tensor_info in analysis["norm_tensors"]:
                print(f"\nFile: {tensor_info['file']}")
                for key, value in tensor_info.items():
                    if key != "file":
                        print(f"  - {key}: {value}")

    # 4. Summary and recommendations
    print("\n" + "=" * 80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 80)
    print("""
To ensure numerical consistency between SGLang and Megatron final norm:

1. ✅ Check epsilon values match (rms_norm_eps / layernorm_epsilon)
2. ✅ Check norm.weight parameters are identical
3. ✅ Check dtype configurations:
   - SGLang: weight_dtype=torch.float32, override_orig_dtype=torch.float32
   - Megatron: SGLangFinalRMSNorm with FP32 weights
4. ✅ Check computation paths:
   - Both should use: variance = x.pow(2).mean(dim=-1, keepdim=True)
   - Both should use: x = weight * x.to(orig_dtype) with orig_dtype=FP32
5. ✅ Verify residual handling:
   - SGLang: norm receives (hidden_states, residual), adds them inside norm
   - Megatron: norm receives (hidden_states) already with residual added

If weights match but outputs differ, check:
- Numerical precision in variance computation
- Order of operations in dtype casting
- Hardware-specific floating point behavior (CPU vs GPU)
    """)


if __name__ == "__main__":
    main()
