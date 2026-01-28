"""
Debug utilities for checking weight sync between Megatron and SGLang.

Usage:
    在 actor.py 的 update_weights() 之后调用：
    from .debug_weight_sync import debug_compare_weights_from_dict
    megatron_weights = self.weights_backuper.get("actor")
    debug_compare_weights_from_dict(megatron_weights, rollout_engines, layer_idx=0)
"""

import re
import logging
import numpy as np
import torch
import torch.distributed as dist
import ray
import requests
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Mapping
from megatron.core import mpu

logger = logging.getLogger(__name__)


@dataclass
class WeightComparisonResult:
    """Comprehensive weight comparison result."""
    name: str
    megatron_dtype: str
    sglang_dtype: str
    megatron_shape: tuple
    sglang_shape: tuple
    # All metrics computed in float64 for maximum precision
    megatron_sum: float
    sglang_sum: float
    sum_diff: float
    megatron_norm: float  # L2 norm
    sglang_norm: float
    norm_diff: float
    abs_diff_sum: float  # sum of |meg - sgl|
    max_abs_diff: float  # max |meg - sgl|
    rel_diff: float  # abs_diff_sum / max(|meg_sum|, |sgl_sum|, 1e-10)
    is_match: bool  # True if weights are essentially equal
    
    def __str__(self):
        status = "OK" if self.is_match else "MISMATCH"
        return (
            f"[{status}] {self.name}\n"
            f"  Megatron: dtype={self.megatron_dtype}, shape={self.megatron_shape}\n"
            f"  SGLang:   dtype={self.sglang_dtype}, shape={self.sglang_shape}\n"
            f"  Sum:      Meg={self.megatron_sum:.10e}, SGL={self.sglang_sum:.10e}, diff={self.sum_diff:.6e}\n"
            f"  Norm:     Meg={self.megatron_norm:.10e}, SGL={self.sglang_norm:.10e}, diff={self.norm_diff:.6e}\n"
            f"  AbsDiff:  sum={self.abs_diff_sum:.6e}, max={self.max_abs_diff:.6e}, rel={self.rel_diff:.6e}"
        )


def compare_weights_detailed(
    name: str,
    megatron_tensor: torch.Tensor,
    sglang_tensor: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    sglang_original_dtype: Optional[str] = None,
) -> WeightComparisonResult:
    """
    Compare two weight tensors with comprehensive metrics.
    
    Both tensors are converted to float64 for comparison to avoid precision issues.
    
    Args:
        name: Weight name for identification
        megatron_tensor: Weight from Megatron (can be any dtype, any device)
        sglang_tensor: Weight from SGLang (can be any dtype, any device)
        rtol: Relative tolerance for determining match
        atol: Absolute tolerance for determining match
        sglang_original_dtype: Original dtype of SGLang weight (if known)
    
    Returns:
        WeightComparisonResult with all comparison metrics
    """
    # Record original dtypes
    meg_dtype = str(megatron_tensor.dtype)
    # SGLang tensor is from API (JSON), so dtype is float64 for precision
    # If original dtype is provided, show that instead
    if sglang_original_dtype:
        sgl_dtype = sglang_original_dtype
    else:
        # Indicate it's from API if it's float64 (our conversion)
        if sglang_tensor.dtype == torch.float64:
            sgl_dtype = "via API (stored as bf16/fp16)"
        else:
            sgl_dtype = str(sglang_tensor.dtype)
    meg_shape = tuple(megatron_tensor.shape)
    sgl_shape = tuple(sglang_tensor.shape)
    
    # Convert both to float64 on CPU for maximum precision comparison
    meg_f64 = megatron_tensor.detach().cpu().to(torch.float64)
    sgl_f64 = sglang_tensor.detach().cpu().to(torch.float64)
    
    # Compute metrics
    meg_sum = meg_f64.sum().item()
    sgl_sum = sgl_f64.sum().item()
    sum_diff = abs(meg_sum - sgl_sum)
    
    meg_norm = meg_f64.norm().item()
    sgl_norm = sgl_f64.norm().item()
    norm_diff = abs(meg_norm - sgl_norm)
    
    # Element-wise comparison (only if shapes match)
    if meg_shape == sgl_shape:
        diff = (meg_f64 - sgl_f64).abs()
        abs_diff_sum = diff.sum().item()
        max_abs_diff = diff.max().item()
    else:
        abs_diff_sum = float('inf')
        max_abs_diff = float('inf')
    
    # Relative difference
    scale = max(abs(meg_sum), abs(sgl_sum), 1e-10)
    rel_diff = abs_diff_sum / (meg_f64.numel() * scale) if meg_shape == sgl_shape else float('inf')
    
    # Determine if match using both absolute and relative criteria
    is_match = (
        meg_shape == sgl_shape and
        abs_diff_sum < atol * meg_f64.numel() + rtol * scale * meg_f64.numel()
    )
    
    return WeightComparisonResult(
        name=name,
        megatron_dtype=meg_dtype,
        sglang_dtype=sgl_dtype,
        megatron_shape=meg_shape,
        sglang_shape=sgl_shape,
        megatron_sum=meg_sum,
        sglang_sum=sgl_sum,
        sum_diff=sum_diff,
        megatron_norm=meg_norm,
        sglang_norm=sgl_norm,
        norm_diff=norm_diff,
        abs_diff_sum=abs_diff_sum,
        max_abs_diff=max_abs_diff,
        rel_diff=rel_diff,
        is_match=is_match,
    )


def print_weight_comparison(result: WeightComparisonResult, verbose: bool = True):
    """Print weight comparison result with optional verbosity."""
    if result.is_match:
        if verbose:
            rank = dist.get_rank() if dist.is_initialized() else 0
            # Check if it's an exact match (abs_diff_sum == 0)
            if result.abs_diff_sum == 0:
                print(f"  [Rank {rank}] OK: {result.name} [EXACT MATCH]")
                print(f"    shape: {result.megatron_shape}, Meg dtype={result.megatron_dtype}")
            else:
                print(f"  [Rank {rank}] OK: {result.name}")
                print(f"    shape: {result.megatron_shape}, Meg dtype={result.megatron_dtype}")
                print(f"    sum_diff={result.sum_diff:.6e}, norm_diff={result.norm_diff:.6e}, "
                      f"abs_diff_sum={result.abs_diff_sum:.6e}, max_abs_diff={result.max_abs_diff:.6e}")
    else:
        print(f"  MISMATCH: {result.name}")
        print(f"    Megatron: dtype={result.megatron_dtype}, shape={result.megatron_shape}")
        print(f"    SGLang:   dtype={result.sglang_dtype}, shape={result.sglang_shape}")
        print(f"    Sum:      Meg={result.megatron_sum:.10e}, SGL={result.sglang_sum:.10e}, diff={result.sum_diff:.6e}")
        print(f"    Norm:     Meg={result.megatron_norm:.10e}, SGL={result.sglang_norm:.10e}, diff={result.norm_diff:.6e}")
        print(f"    AbsDiff:  sum={result.abs_diff_sum:.6e}, max={result.max_abs_diff:.6e}, rel={result.rel_diff:.6e}")


def debug_compare_tensors(
    name: str,
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    label_a: str = "TensorA",
    label_b: str = "TensorB",
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> WeightComparisonResult:
    """
    Convenient function to compare two tensors anywhere in the code.
    
    Usage:
        from slime.backends.megatron_utils.debug_weight_sync import debug_compare_tensors
        debug_compare_tensors("my_weight", megatron_tensor, sglang_tensor)
    
    Args:
        name: Name for this comparison (for logging)
        tensor_a: First tensor (typically from Megatron)
        tensor_b: Second tensor (typically from SGLang)
        label_a: Label for tensor_a in output
        label_b: Label for tensor_b in output
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Returns:
        WeightComparisonResult with detailed comparison
    """
    result = compare_weights_detailed(name, tensor_a, tensor_b, rtol=rtol, atol=atol)
    
    # Print detailed comparison
    status = "OK" if result.is_match else "MISMATCH"
    print(f"\n[DEBUG TENSOR COMPARE] [{status}] {name}")
    print(f"  {label_a}: dtype={result.megatron_dtype}, shape={result.megatron_shape}")
    print(f"  {label_b}: dtype={result.sglang_dtype}, shape={result.sglang_shape}")
    print(f"  Sum:      {label_a}={result.megatron_sum:.10e}, {label_b}={result.sglang_sum:.10e}, diff={result.sum_diff:.6e}")
    print(f"  Norm:     {label_a}={result.megatron_norm:.10e}, {label_b}={result.sglang_norm:.10e}, diff={result.norm_diff:.6e}")
    print(f"  AbsDiff:  sum={result.abs_diff_sum:.6e}, max={result.max_abs_diff:.6e}, rel={result.rel_diff:.6e}")
    
    # Print first few values for debugging
    if result.megatron_shape == result.sglang_shape:
        a_flat = tensor_a.detach().cpu().flatten()[:5].to(torch.float64).tolist()
        b_flat = tensor_b.detach().cpu().flatten()[:5].to(torch.float64).tolist()
        print(f"  First 5:  {label_a}={a_flat}")
        print(f"            {label_b}={b_flat}")
    
    return result


def _process_sglang_return(ret):
    """Process return value from SGLang get_weights_by_name API.
    
    Returns:
        np.ndarray of the weight, or None if ret is None/empty.
        
    Note: 
        - For single DP worker: ret is the weight directly (nested list from tolist())
        - For multiple DP workers: ret is [worker0_weight, worker1_weight, ...]
        
        We detect DP case by checking if ret[0] is also a nested list with same structure.
        A 2D weight [[row0], [row1], ...] has ret[0] as a 1D list (single row).
        DP multi-return [weight0, weight1, ...] has ret[0] as a 2D list (full weight).
    """
    if ret is None:
        return None
    
    # Convert to numpy array directly - this handles both cases correctly
    arr = np.array(ret)
    
    # Check if this is DP multi-return case: shape would be (dp_size, rows, cols) for 2D weights
    # or (dp_size, size) for 1D weights
    # In DP case, we take the first worker's result
    if arr.ndim >= 3:
        # Multi-DP case for 2D+ weights: (dp_size, ...) -> take first
        return arr[0]
    elif arr.ndim == 2:
        # Could be: 
        # 1. Single DP 2D weight: shape (rows, cols) - return as is
        # 2. Multi-DP 1D weight: shape (dp_size, size) - take first
        # Heuristic: if first dim is small (like dp_size <= 8), might be DP case
        # But safer to assume it's 2D weight and return as is
        return arr
    else:
        # 1D or scalar
        return arr


def debug_compare_weights_from_dict(
    megatron_weights: Mapping[str, torch.Tensor],
    rollout_engines: List,
    layer_idx: int = 0,
    verbose: bool = True,
    compare_rank: int = 0,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Compare weights between Megatron (from CPU dict) and SGLang for a specific layer.
    This version uses weights_backuper.get() output, avoiding GPU offload issues.
    
    Args:
        megatron_weights: Dict from weights_backuper.get("actor"), already on CPU
                         Names are in Megatron format: module.module.decoder.layers.X.mlp.router.weight
                         On the comparing rank this is that rank's local weights (sharded if TP > 1).
        rollout_engines: List of SGLang engine handles
        layer_idx: Which layer to compare
        verbose: Print detailed info
        compare_rank: Which rank runs the comparison (default 0). Only this rank runs; others return {}.
                      Use compare_rank=1 to compare rank 1's Megatron weights vs SGLang.
                      Note: With TP > 1, Megatron holds shards per rank while SGLang holds the full
                      assembled weight—shapes will differ unless SGLang weight is sliced to this rank's shard.
    
    Returns:
        Dict mapping weight name to (megatron_sum, sglang_sum, diff)
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != compare_rank:
        return {}
    
    results = {}
    
    print(f"\n{'='*80}")
    print(f"[DEBUG WEIGHT SYNC] Comparing Megatron (rank {compare_rank}) vs SGLang weights for layer {layer_idx}")
    print(f"{'='*80}")
    
    try:
        # Get server info
        print("\n[Step 1] Getting SGLang server info...")
        try:
            server_info = ray.get(rollout_engines[0].get_server_info.remote())
            server_host = server_info["server_host"]
            server_port = server_info["server_port"]
            print(f"  SGLang server: {server_host}:{server_port}")
        except Exception as e:
            print(f"[ERROR] Cannot get SGLang server info: {e}")
            return results
        
        # Print available Megatron weight keys for this layer
        print(f"\n[Step 2] Megatron weights for layer {layer_idx}:")
        layer_weights = {k: v for k, v in megatron_weights.items() if f"layers.{layer_idx}." in k}
        print(f"  Found {len(layer_weights)} weights")
        for name, tensor in sorted(layer_weights.items()):
            try:
                # Use float64 for maximum precision in reporting
                weight_sum = tensor.to(torch.float64).sum().item()
                weight_norm = tensor.to(torch.float64).norm().item()
                print(f"    {name}: dtype={tensor.dtype}, shape={list(tensor.shape)}, sum={weight_sum:.10e}, norm={weight_norm:.10e}")
            except Exception as e:
                print(f"    {name}: ERROR - {e}")
        
        # Megatron internal format -> HuggingFace format mapping
        # Megatron: module.module.decoder.layers.{layer}.mlp.router.weight
        # HF:       model.layers.{layer}.mlp.gate.weight
        weight_mappings = {
            # Router
            f"module.module.decoder.layers.{layer_idx}.mlp.router.weight": f"model.layers.{layer_idx}.mlp.gate.weight",
            # Shared experts (if exists)
            f"module.module.decoder.layers.{layer_idx}.mlp.shared_experts.linear_fc1.weight": f"model.layers.{layer_idx}.mlp.shared_expert.gate_up_proj.weight",
            f"module.module.decoder.layers.{layer_idx}.mlp.shared_experts.linear_fc2.weight": f"model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight",
        }
        
        # Check regular weights with detailed comparison
        print("\n[Step 3] Comparing regular weights...")
        for megatron_name, hf_name in weight_mappings.items():
            if megatron_name in megatron_weights:
                megatron_tensor = megatron_weights[megatron_name]
                sglang_tensor = _get_sglang_weight(server_host, server_port, hf_name, verbose)
                
                if sglang_tensor is not None:
                    result = compare_weights_detailed(hf_name, megatron_tensor, sglang_tensor)
                    results[hf_name] = result
                    print_weight_comparison(result, verbose)
                else:
                    print(f"  SKIP: {hf_name} (SGLang returned None)")
            else:
                # Only print skip message for optional weights (shared_experts) at lower verbosity
                if "shared_experts" in megatron_name:
                    print(f"  (N/A: shared_experts not used in this model config)")
                elif verbose:
                    print(f"  SKIP: {megatron_name} not in Megatron weights")
        
        # Check expert weights with detailed comparison
        print("\n[Step 4] Comparing expert weights...")
        _compare_expert_weights_from_dict(megatron_weights, layer_idx, server_host, server_port, results, verbose)
        
        # Summary with comprehensive metrics
        print(f"\n{'='*80}")
        print(f"[SUMMARY] Weight comparison results:")
        has_error = False
        for name, result in results.items():
            if isinstance(result, WeightComparisonResult):
                if not result.is_match:
                    has_error = True
                    print(f"\n  (rank {compare_rank}) MISMATCH: {name}")
                    print(f"    Megatron: dtype={result.megatron_dtype}, shape={result.megatron_shape}")
                    print(f"    SGLang:   dtype={result.sglang_dtype}, shape={result.sglang_shape}")
                    print(f"    Sum:      Meg={result.megatron_sum:.10e}, SGL={result.sglang_sum:.10e}, diff={result.sum_diff:.6e}")
                    print(f"    Norm:     Meg={result.megatron_norm:.10e}, SGL={result.sglang_norm:.10e}, diff={result.norm_diff:.6e}")
                    print(f"    AbsDiff:  sum={result.abs_diff_sum:.6e}, max={result.max_abs_diff:.6e}, rel={result.rel_diff:.6e}")
            else:
                # Legacy format (tuple)
                meg_sum, sgl_sum, diff = result
                if diff >= 1e-5:
                    has_error = True
                    print(f"  (rank {compare_rank}) MISMATCH: {name}")
                    print(f"    Megatron sum: {meg_sum:.10e}")
                    print(f"    SGLang sum:   {sgl_sum:.10e}")
                    print(f"    Diff:         {diff:.6e}")
        
        if has_error:
            print(f"\n*** (rank {compare_rank}) WARNING: Weight mismatch detected! ***")
        else:
            print(f"\nAll weights match!")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"[DEBUG] (rank {compare_rank}) Weight comparison error: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def _compare_expert_weights_from_dict(
    megatron_weights: Mapping[str, torch.Tensor],
    layer_idx: int,
    server_host: str,
    server_port: int,
    results: Dict,
    verbose: bool,
):
    """Compare expert weights from CPU dict with comprehensive metrics.
    
    Megatron format: module.module.decoder.layers.{layer}.mlp.experts.linear_fc1.weight{expert_id}
    HF format:       model.layers.{layer}.mlp.experts.{expert_id}.gate_up_proj.weight
    
    Note: Megatron's linear_fc1 shape is [2*intermediate_size, hidden_size] which combines gate+up
          HF's gate_up_proj shape is [2*intermediate_size, hidden_size]
    """
    # Find expert weights in megatron_weights
    # Pattern: module.module.decoder.layers.{layer}.mlp.experts.linear_fc1.weight{expert_id}
    expert_fc1_pattern = re.compile(
        rf"module\.module\.decoder\.layers\.{layer_idx}\.mlp\.experts\.linear_fc1\.weight(\d+)"
    )
    expert_fc2_pattern = re.compile(
        rf"module\.module\.decoder\.layers\.{layer_idx}\.mlp\.experts\.linear_fc2\.weight(\d+)"
    )
    
    expert_weights_fc1 = {}
    expert_weights_fc2 = {}
    
    for name, tensor in megatron_weights.items():
        match_fc1 = expert_fc1_pattern.match(name)
        match_fc2 = expert_fc2_pattern.match(name)
        if match_fc1:
            expert_id = int(match_fc1.group(1))
            expert_weights_fc1[expert_id] = (name, tensor)
        elif match_fc2:
            expert_id = int(match_fc2.group(1))
            expert_weights_fc2[expert_id] = (name, tensor)
    
    print(f"  Found {len(expert_weights_fc1)} experts for fc1 (gate_up_proj)")
    print(f"  Found {len(expert_weights_fc2)} experts for fc2 (down_proj)")
    
    # Compare fc1 (gate_up_proj) with detailed metrics
    for expert_id in sorted(expert_weights_fc1.keys()):
        megatron_name, megatron_tensor = expert_weights_fc1[expert_id]
        hf_name = f"model.layers.{layer_idx}.mlp.experts.{expert_id}.gate_up_proj.weight"
        
        # Try to get the fused weight first (gate_up_proj)
        sglang_tensor = _get_sglang_weight(server_host, server_port, hf_name, verbose)
        
        if sglang_tensor is None:
            # Fallback: SGLang might store as separate gate_proj and up_proj
            gate_name = f"model.layers.{layer_idx}.mlp.experts.{expert_id}.gate_proj.weight"
            up_name = f"model.layers.{layer_idx}.mlp.experts.{expert_id}.up_proj.weight"
            gate_tensor = _get_sglang_weight(server_host, server_port, gate_name, verbose)
            up_tensor = _get_sglang_weight(server_host, server_port, up_name, verbose)
            
            if gate_tensor is not None and up_tensor is not None:
                # Concatenate gate and up for comparison
                sglang_tensor = torch.cat([gate_tensor, up_tensor], dim=0)
                hf_name = f"{gate_name} + {up_name}"
            else:
                if verbose:
                    print(f"  SKIP: Expert {expert_id} gate_up_proj (SGLang returned None for both formats)")
                continue
        
        # Use detailed comparison
        result = compare_weights_detailed(hf_name, megatron_tensor, sglang_tensor)
        results[hf_name] = result
        print_weight_comparison(result, verbose)
    
    # Compare fc2 (down_proj) with detailed metrics
    for expert_id in sorted(expert_weights_fc2.keys()):
        megatron_name, megatron_tensor = expert_weights_fc2[expert_id]
        hf_name = f"model.layers.{layer_idx}.mlp.experts.{expert_id}.down_proj.weight"
        
        sglang_tensor = _get_sglang_weight(server_host, server_port, hf_name, verbose)
        if sglang_tensor is not None:
            result = compare_weights_detailed(hf_name, megatron_tensor, sglang_tensor)
            results[hf_name] = result
            print_weight_comparison(result, verbose)
        else:
            if verbose:
                print(f"  SKIP: Expert {expert_id} down_proj (SGLang returned None)")


def debug_print_megatron_weights(
    model: List[torch.nn.Module],
    layer_idx: int = 0,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Print Megatron weight sums for debugging (no SGLang comparison).
    This is a safe version that only reads Megatron weights.
    
    Returns:
        Dict mapping weight name to sum
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return {}
    
    results = {}
    
    print(f"\n{'='*80}")
    print(f"[DEBUG] Megatron weight sums for layer {layer_idx}")
    print(f"{'='*80}")
    
    try:
        # Sync CUDA to ensure all operations are complete
        torch.cuda.synchronize()
        
        for model_module in model:
            for name, param in model_module.named_parameters():
                if f"layers.{layer_idx}" in name:
                    try:
                        # Safely copy to CPU first, then compute sum
                        with torch.no_grad():
                            cpu_data = param.data.detach().clone().cpu().float()
                            weight_sum = cpu_data.sum().item()
                        results[name] = weight_sum
                        if verbose:
                            print(f"  {name}: shape={list(param.shape)}, sum={weight_sum:.10e}")
                        # Free CPU memory immediately
                        del cpu_data
                    except Exception as e:
                        print(f"  {name}: ERROR - {e}")
        
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"[DEBUG] Megatron weight print error: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def debug_compare_all_weights(
    model: List[torch.nn.Module],
    rollout_engines: List,
    layer_idx: int = 0,
    verbose: bool = True,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Compare all weights between Megatron and SGLang for a specific layer.
    
    Returns:
        Dict mapping weight name to (megatron_sum, sglang_sum, diff)
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    results = {}
    
    print(f"\n{'='*80}")
    print(f"[DEBUG WEIGHT SYNC] Comparing Megatron vs SGLang weights for layer {layer_idx}")
    print(f"{'='*80}")
    
    try:
        # Sync CUDA to ensure all operations are complete
        torch.cuda.synchronize()
        
        # First, print Megatron weights (safe operation)
        print(f"\n[Step 1][Rank {rank}] Reading Megatron weights...")
        megatron_weights = debug_print_megatron_weights(model, layer_idx, verbose=False)
        print(f"  Successfully read {len(megatron_weights)} weights from Megatron")
        
        # Get server info
        print(f"\n[Step 2][Rank {rank}] Getting SGLang server info...")
        try:
            server_info = ray.get(rollout_engines[0].get_server_info.remote())
            server_host = server_info["server_host"]
            server_port = server_info["server_port"]
            print(f"  SGLang server: {server_host}:{server_port}")
        except Exception as e:
            print(f"[ERROR] Cannot get SGLang server info: {e}")
            return results
        
        # Weight name mappings: Megatron -> HuggingFace
        weight_mappings = {
            # Router
            f"layers.{layer_idx}.mlp.router.weight": f"model.layers.{layer_idx}.mlp.gate.weight",
            # Shared experts
            f"layers.{layer_idx}.mlp.shared_experts.linear_fc1.weight": f"model.layers.{layer_idx}.mlp.shared_expert.gate_up_proj.weight",
            f"layers.{layer_idx}.mlp.shared_experts.linear_fc2.weight": f"model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight",
            # Expert weights (will be handled specially for EP)
        }
        
        # Check regular weights
        print(f"\n[Step 3][Rank {rank}] Comparing regular weights...")
        for megatron_pattern, hf_name in weight_mappings.items():
            try:
                _compare_single_weight(model, megatron_pattern, hf_name, 
                                      server_host, server_port, results, verbose)
            except Exception as e:
                print(f"[ERROR] Failed to compare {megatron_pattern}: {e}")
        
        # Check expert weights (need special handling for EP)
        print(f"\n[Step 4][Rank {rank}] Comparing expert weights...")
        try:
            _compare_expert_weights(model, rollout_engines, layer_idx, 
                                   server_host, server_port, results, rank=rank, verbose=verbose)
        except Exception as e:
            print(f"[ERROR] Failed to compare expert weights: {e}")
            import traceback
            traceback.print_exc()
        
        # Summary
        print(f"\n{'='*80}")
        print(f"[SUMMARY][Rank {rank}] Weight comparison results:")
        has_error = False
        for name, (meg_sum, sgl_sum, diff) in results.items():
            status = "OK" if diff < 1e-5 else "MISMATCH"
            if diff >= 1e-5:
                has_error = True
                print(f"  [Rank {rank}] {status}: {name}")
                print(f"    Megatron sum: {meg_sum:.10e}")
                print(f"    SGLang sum:   {sgl_sum:.10e}")
                print(f"    Diff:         {diff:.6e}")
            elif verbose:
                print(f"  {status}: {name} (diff={diff:.6e})")
        
        if has_error:
            print(f"\n*** WARNING: Weight mismatch detected! ***")
        else:
            print(f"\nAll weights match!")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"[DEBUG] Weight comparison error: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def _get_sglang_weight(
    server_host: str, 
    server_port: int, 
    hf_name: str, 
    verbose: bool = False,
    target_dtype: Optional[torch.dtype] = None,
) -> Optional[torch.Tensor]:
    """Get weight from SGLang via HTTP API.
    
    Args:
        server_host: SGLang server host
        server_port: SGLang server port
        hf_name: HuggingFace format weight name
        verbose: Print debug info
        target_dtype: If provided, convert result to this dtype. 
                     If None, use float64 for maximum precision.
    
    Returns:
        Weight tensor or None if failed
    """
    try:
        # Use GET with json parameter (as shown in SGLang tests)
        response = requests.get(
            f"http://{server_host}:{server_port}/get_weights_by_name",
            json={"name": hf_name, "truncate_size": -1},
            timeout=60
        )
        if response.status_code == 200:
            sglang_data = response.json()
            processed = _process_sglang_return(sglang_data)
            if processed is not None:
                # Use float64 by default for precision, or target_dtype if specified
                dtype = target_dtype if target_dtype is not None else torch.float64
                return torch.tensor(processed, dtype=dtype)
        else:
            if verbose:
                print(f"[SKIP] SGLang API error {response.status_code} for {hf_name}")
    except Exception as e:
        if verbose:
            print(f"[SKIP] Error getting SGLang weight {hf_name}: {e}")
    return None


def _safe_get_cpu_tensor(param: torch.nn.Parameter) -> Optional[torch.Tensor]:
    """Safely copy a CUDA tensor to CPU with proper synchronization."""
    try:
        if param.device.type == 'cuda':
            torch.cuda.synchronize(param.device)
        # Clone first, then detach and move to CPU
        with torch.no_grad():
            cpu_tensor = param.data.clone().detach().cpu().float()
        return cpu_tensor
    except Exception as e:
        print(f"[ERROR] Failed to copy tensor to CPU: {e}")
        return None


def _compare_single_weight(
    model: List[torch.nn.Module],
    megatron_pattern: str,
    hf_name: str,
    server_host: str,
    server_port: int,
    results: Dict,
    verbose: bool,
):
    """Compare a single weight between Megatron and SGLang."""
    # Get Megatron weight
    megatron_weight = None
    for model_module in model:
        for name, param in model_module.named_parameters():
            if megatron_pattern in name:
                megatron_weight = _safe_get_cpu_tensor(param)
                break
        if megatron_weight is not None:
            break
    
    if megatron_weight is None:
        if verbose:
            print(f"[SKIP] Megatron weight not found: {megatron_pattern}")
        return
    
    # Get SGLang weight
    sglang_weight = _get_sglang_weight(server_host, server_port, hf_name, verbose)
    if sglang_weight is not None:
        meg_sum = megatron_weight.sum().item()
        sgl_sum = sglang_weight.sum().item()
        diff = abs(meg_sum - sgl_sum)
        results[hf_name] = (meg_sum, sgl_sum, diff)
    else:
        if verbose:
            print(f"[SKIP] SGLang returned None for {hf_name}")


def _compare_expert_weights(
    model: List[torch.nn.Module],
    rollout_engines: List,
    layer_idx: int,
    server_host: str,
    server_port: int,
    results: Dict,
    rank: int,
    verbose: bool,
):
    """Compare expert weights (handles EP distribution)."""
    # Get EP info
    try:
        ep_size = mpu.get_expert_model_parallel_world_size()
        ep_rank = mpu.get_expert_model_parallel_rank()
    except Exception as e:
        print(f"[ERROR] Cannot get EP info: {e}")
        ep_size = 1
        ep_rank = 0
    
    if verbose:
        print(f"\n[Expert Weights][Rank {rank}] EP size={ep_size}, EP rank={ep_rank}")
    
    # Sync CUDA before accessing weights
    torch.cuda.synchronize()
    
    # Collect local expert weights from Megatron
    # Pattern: layers.{layer_idx}.mlp.experts.linear_fc1.weight{expert_id}
    local_experts_w1 = {}
    local_experts_w2 = {}
    
    for model_module in model:
        for name, param in model_module.named_parameters():
            if f"layers.{layer_idx}.mlp.experts" in name:
                # Extract expert ID
                match = re.search(r'weight(\d+)$', name)
                if match:
                    expert_id = int(match.group(1))
                    cpu_tensor = _safe_get_cpu_tensor(param)
                    if cpu_tensor is not None:
                        if "linear_fc1" in name:
                            local_experts_w1[expert_id] = cpu_tensor
                        elif "linear_fc2" in name:
                            local_experts_w2[expert_id] = cpu_tensor
    
    if verbose:
        print(f"  Found {len(local_experts_w1)} local experts for fc1")
        print(f"  Found {len(local_experts_w2)} local experts for fc2")
        print(f"  Local expert IDs (fc1): {sorted(local_experts_w1.keys())}")
    
    # Compare with SGLang
    # SGLang expert name pattern: model.layers.{layer}.mlp.experts.{expert_id}.gate_up_proj.weight
    for expert_id, megatron_w1 in sorted(local_experts_w1.items()):
        hf_name = f"model.layers.{layer_idx}.mlp.experts.{expert_id}.gate_up_proj.weight"
        sglang_w1 = _get_sglang_weight(server_host, server_port, hf_name, verbose)
        
        if sglang_w1 is not None:
            meg_sum = megatron_w1.sum().item()
            sgl_sum = sglang_w1.sum().item()
            diff = abs(meg_sum - sgl_sum)
            
            results[hf_name] = (meg_sum, sgl_sum, diff)
            
            if diff > 1e-5 and verbose:
                print(f"  [MISMATCH] Expert {expert_id} fc1:")
                print(f"    Megatron shape: {megatron_w1.shape}, sum: {meg_sum:.10e}")
                print(f"    SGLang shape:   {sglang_w1.shape}, sum: {sgl_sum:.10e}")
                print(f"    First 5 values (Megatron): {megatron_w1.flatten()[:5].tolist()}")
                print(f"    First 5 values (SGLang):   {sglang_w1.flatten()[:5].tolist()}")
        else:
            if verbose:
                print(f"  [SKIP] Cannot get SGLang expert {expert_id} fc1")
    
    for expert_id, megatron_w2 in sorted(local_experts_w2.items()):
        hf_name = f"model.layers.{layer_idx}.mlp.experts.{expert_id}.down_proj.weight"
        sglang_w2 = _get_sglang_weight(server_host, server_port, hf_name, verbose)
        
        if sglang_w2 is not None:
            meg_sum = megatron_w2.sum().item()
            sgl_sum = sglang_w2.sum().item()
            diff = abs(meg_sum - sgl_sum)
            
            results[hf_name] = (meg_sum, sgl_sum, diff)
            
            if diff > 1e-5 and verbose:
                print(f"  [MISMATCH] Expert {expert_id} fc2:")
                print(f"    Megatron shape: {megatron_w2.shape}, sum: {meg_sum:.10e}")
                print(f"    SGLang shape:   {sglang_w2.shape}, sum: {sgl_sum:.10e}")
        else:
            if verbose:
                print(f"  [SKIP] Cannot get SGLang expert {expert_id} fc2")


def debug_check_expert_weight_shapes(model: List[torch.nn.Module], layer_idx: int = 0):
    """Print expert weight shapes for debugging."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    ep_size = mpu.get_expert_model_parallel_world_size()
    ep_rank = mpu.get_expert_model_parallel_rank()
    
    print(f"\n[Rank {rank}] Expert weight shapes for layer {layer_idx}:")
    print(f"  EP size={ep_size}, EP rank={ep_rank}")
    
    for model_module in model:
        for name, param in model_module.named_parameters():
            if f"layers.{layer_idx}.mlp.experts" in name:
                print(f"  {name}: {param.shape}, dtype={param.dtype}")


def debug_compare_weights_before_after_sync(
    model: List[torch.nn.Module],
    rollout_engines: List,
    layer_idx: int = 0,
):
    """
    To be called before and after weight sync to detect changes.
    
    Usage:
        # Before sync
        before_state = debug_compare_weights_before_after_sync(model, engines, layer_idx)
        
        # Do sync
        weight_updater.update_weights()
        
        # After sync  
        after_state = debug_compare_weights_before_after_sync(model, engines, layer_idx)
        
        # Compare
        for name in before_state:
            if before_state[name] != after_state[name]:
                print(f"Weight {name} changed!")
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    state = {}
    
    for model_module in model:
        for name, param in model_module.named_parameters():
            if f"layers.{layer_idx}" in name:
                state[name] = param.data.sum().item()
    
    if rank == 0:
        print(f"[Rank {rank}] Captured {len(state)} weights for layer {layer_idx}")
    
    return state


def debug_compare_embedding_lmhead(
    megatron_weights: Mapping[str, torch.Tensor],
    rollout_engines: List,
    verbose: bool = True,
) -> Dict[str, WeightComparisonResult]:
    """
    Compare embedding and lm_head weights between Megatron and SGLang.
    These are critical for logprobs calculation!
    
    Usage:
        from .debug_weight_sync import debug_compare_embedding_lmhead
        megatron_weights = self.weights_backuper.get("actor")
        debug_compare_embedding_lmhead(megatron_weights, rollout_engines)
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return {}
    
    results = {}
    
    print(f"\n{'='*80}")
    print(f"[DEBUG] Comparing Embedding & LM Head weights (critical for logprobs!)")
    print(f"{'='*80}")
    
    try:
        # Get server info
        server_info = ray.get(rollout_engines[0].get_server_info.remote())
        server_host = server_info["server_host"]
        server_port = server_info["server_port"]
        
        # Weight mappings for embedding and lm_head
        weight_mappings = {
            "module.module.embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "module.module.output_layer.weight": "lm_head.weight",
            "module.module.decoder.final_layernorm.weight": "model.norm.weight",
        }
        
        for megatron_name, hf_name in weight_mappings.items():
            if megatron_name in megatron_weights:
                megatron_tensor = megatron_weights[megatron_name]
                sglang_tensor = _get_sglang_weight(server_host, server_port, hf_name, verbose)
                
                if sglang_tensor is not None:
                    result = compare_weights_detailed(hf_name, megatron_tensor, sglang_tensor)
                    results[hf_name] = result
                    print_weight_comparison(result, verbose)
                else:
                    print(f"  SKIP: {hf_name} (SGLang returned None)")
            else:
                print(f"  SKIP: {megatron_name} not found in Megatron weights")
        
        # Summary
        has_error = any(not r.is_match for r in results.values() if isinstance(r, WeightComparisonResult))
        if has_error:
            print(f"\n*** WARNING: Embedding/LM Head mismatch - this WILL cause logprobs diff! ***")
        else:
            print(f"\nEmbedding & LM Head weights match!")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    return results


def debug_compare_weights_all_ranks(
    megatron_weights: Mapping[str, torch.Tensor],
    rollout_engines: List,
    layer_idx: int = 0,
    verbose: bool = True,
) -> None:
    """
    Compare weights from ALL ranks (not just rank 0).
    Each rank prints its own comparison results.
    
    Usage:
        from .debug_weight_sync import debug_compare_weights_all_ranks
        megatron_weights = self.weights_backuper.get("actor")
        debug_compare_weights_all_ranks(megatron_weights, rollout_engines, layer_idx=0)
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    print(f"\n[Rank {rank}/{world_size}] {'='*60}")
    print(f"[Rank {rank}/{world_size}] Checking weights for layer {layer_idx}")
    print(f"[Rank {rank}/{world_size}] {'='*60}")
    
    try:
        # Get server info (need to get from appropriate engine for this rank)
        engine_idx = rank % len(rollout_engines)
        server_info = ray.get(rollout_engines[engine_idx].get_server_info.remote())
        server_host = server_info["server_host"]
        server_port = server_info["server_port"]
        
        # Find expert weights for this rank
        expert_fc1_pattern = re.compile(
            rf"module\.module\.decoder\.layers\.{layer_idx}\.mlp\.experts\.linear_fc1\.weight(\d+)"
        )
        
        expert_weights = {}
        for name, tensor in megatron_weights.items():
            match = expert_fc1_pattern.match(name)
            if match:
                expert_id = int(match.group(1))
                expert_weights[expert_id] = (name, tensor)
        
        print(f"[Rank {rank}] Found {len(expert_weights)} experts in local backup")
        
        # Check a few experts
        checked = 0
        mismatches = 0
        for expert_id in sorted(expert_weights.keys())[:3]:  # Check first 3 experts
            megatron_name, megatron_tensor = expert_weights[expert_id]
            hf_name = f"model.layers.{layer_idx}.mlp.experts.{expert_id}.gate_up_proj.weight"
            
            sglang_tensor = _get_sglang_weight(server_host, server_port, hf_name, verbose=False)
            if sglang_tensor is not None:
                result = compare_weights_detailed(hf_name, megatron_tensor, sglang_tensor)
                checked += 1
                if not result.is_match:
                    mismatches += 1
                    print(f"[Rank {rank}] MISMATCH: Expert {expert_id}")
                    print(f"  abs_diff_sum={result.abs_diff_sum:.6e}, max_abs_diff={result.max_abs_diff:.6e}")
                elif verbose:
                    print(f"[Rank {rank}] OK: Expert {expert_id} (max_abs_diff={result.max_abs_diff:.6e})")
        
        print(f"[Rank {rank}] Checked {checked} experts, {mismatches} mismatches")
        
        # Also check router
        router_meg_name = f"module.module.decoder.layers.{layer_idx}.mlp.router.weight"
        if router_meg_name in megatron_weights:
            router_hf_name = f"model.layers.{layer_idx}.mlp.gate.weight"
            sglang_tensor = _get_sglang_weight(server_host, server_port, router_hf_name, verbose=False)
            if sglang_tensor is not None:
                result = compare_weights_detailed(router_hf_name, megatron_weights[router_meg_name], sglang_tensor)
                if not result.is_match:
                    print(f"[Rank {rank}] MISMATCH: Router! abs_diff_sum={result.abs_diff_sum:.6e}")
                else:
                    print(f"[Rank {rank}] OK: Router (max_abs_diff={result.max_abs_diff:.6e})")
        
    except Exception as e:
        print(f"[Rank {rank}] ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Barrier to sync output
    if dist.is_initialized():
        dist.barrier()


def debug_check_sglang_tp_consistency(
    rollout_engines: List,
    layer_idx: int = 0,
    weight_name: str = "model.layers.0.mlp.gate.weight",
) -> None:
    """
    Check if SGLang's different TP workers have consistent weights.
    This is critical because each TP worker should have the same weight 
    after weight sync (for replicated params like router).
    
    Usage:
        from .debug_weight_sync import debug_check_sglang_tp_consistency
        debug_check_sglang_tp_consistency(rollout_engines, layer_idx=0)
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return
    
    print(f"\n{'='*80}")
    print(f"[DEBUG] Checking SGLang TP consistency for {weight_name}")
    print(f"{'='*80}")
    
    weights_from_engines = []
    
    for i, engine in enumerate(rollout_engines):
        try:
            server_info = ray.get(engine.get_server_info.remote())
            server_host = server_info["server_host"]
            server_port = server_info["server_port"]
            
            tensor = _get_sglang_weight(server_host, server_port, weight_name, verbose=False)
            if tensor is not None:
                weight_sum = tensor.sum().item()
                weight_norm = tensor.norm().item()
                weights_from_engines.append((i, weight_sum, weight_norm, tensor.shape))
                print(f"  Engine {i}: sum={weight_sum:.10e}, norm={weight_norm:.10e}, shape={tensor.shape}")
            else:
                print(f"  Engine {i}: FAILED to get weight")
        except Exception as e:
            print(f"  Engine {i}: ERROR - {e}")
    
    # Check consistency
    if len(weights_from_engines) > 1:
        ref_sum = weights_from_engines[0][1]
        ref_norm = weights_from_engines[0][2]
        all_match = True
        for i, w_sum, w_norm, shape in weights_from_engines[1:]:
            sum_diff = abs(w_sum - ref_sum)
            norm_diff = abs(w_norm - ref_norm)
            if sum_diff > 1e-6 or norm_diff > 1e-6:
                all_match = False
                print(f"  [Rank {rank}] Engine {i} differs from Engine 0: sum_diff={sum_diff:.6e}, norm_diff={norm_diff:.6e}")
        
        if all_match:
            print(f"\nAll {len(weights_from_engines)} engines have consistent weights!")
        else:
            print(f"\n*** WARNING: TP workers have INCONSISTENT weights! ***")
    
    print(f"{'='*80}\n")
