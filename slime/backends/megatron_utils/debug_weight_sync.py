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
from typing import Optional, List, Dict, Tuple, Mapping
from megatron.core import mpu

logger = logging.getLogger(__name__)


def _process_sglang_return(ret):
    """Process return value from SGLang get_weights_by_name API."""
    if ret is None:
        return None
    if isinstance(ret, list) and len(ret) >= 1:
        # DP case: multiple returns, take the first one
        return np.array(ret[0])
    return np.array(ret)


def debug_compare_weights_from_dict(
    megatron_weights: Mapping[str, torch.Tensor],
    rollout_engines: List,
    layer_idx: int = 0,
    verbose: bool = True,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Compare weights between Megatron (from CPU dict) and SGLang for a specific layer.
    This version uses weights_backuper.get() output, avoiding GPU offload issues.
    
    Args:
        megatron_weights: Dict from weights_backuper.get("actor"), already on CPU
        rollout_engines: List of SGLang engine handles
        layer_idx: Which layer to compare
        verbose: Print detailed info
    
    Returns:
        Dict mapping weight name to (megatron_sum, sglang_sum, diff)
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return {}
    
    results = {}
    
    print(f"\n{'='*80}")
    print(f"[DEBUG WEIGHT SYNC] Comparing Megatron (CPU backup) vs SGLang weights for layer {layer_idx}")
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
                weight_sum = tensor.float().sum().item() if tensor.is_cuda else tensor.sum().item()
                print(f"    {name}: shape={list(tensor.shape)}, sum={weight_sum:.10e}")
            except Exception as e:
                print(f"    {name}: ERROR - {e}")
        
        # Weight name mappings: Megatron (HF-style from backuper) -> SGLang HF name
        # Note: weights_backuper.get() returns HF-style names like "model.layers.X.mlp.gate.weight"
        weight_mappings = {
            # Router
            f"model.layers.{layer_idx}.mlp.gate.weight": f"model.layers.{layer_idx}.mlp.gate.weight",
            # Shared experts
            f"model.layers.{layer_idx}.mlp.shared_expert.gate_up_proj.weight": f"model.layers.{layer_idx}.mlp.shared_expert.gate_up_proj.weight",
            f"model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight": f"model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight",
        }
        
        # Check regular weights
        print("\n[Step 3] Comparing regular weights...")
        for megatron_name, hf_name in weight_mappings.items():
            if megatron_name in megatron_weights:
                megatron_tensor = megatron_weights[megatron_name]
                meg_sum = megatron_tensor.float().sum().item() if megatron_tensor.is_cuda else megatron_tensor.sum().item()
                
                sglang_tensor = _get_sglang_weight(server_host, server_port, hf_name, verbose)
                if sglang_tensor is not None:
                    sgl_sum = sglang_tensor.sum().item()
                    diff = abs(meg_sum - sgl_sum)
                    results[hf_name] = (meg_sum, sgl_sum, diff)
                    status = "OK" if diff < 1e-5 else "MISMATCH"
                    print(f"  {status}: {hf_name} (Megatron={meg_sum:.6e}, SGLang={sgl_sum:.6e}, diff={diff:.6e})")
                else:
                    print(f"  SKIP: {hf_name} (SGLang returned None)")
            else:
                if verbose:
                    print(f"  SKIP: {megatron_name} not in Megatron weights")
        
        # Check expert weights
        print("\n[Step 4] Comparing expert weights...")
        _compare_expert_weights_from_dict(megatron_weights, layer_idx, server_host, server_port, results, verbose)
        
        # Summary
        print(f"\n{'='*80}")
        print(f"[SUMMARY] Weight comparison results:")
        has_error = False
        for name, (meg_sum, sgl_sum, diff) in results.items():
            if diff >= 1e-5:
                has_error = True
                print(f"  MISMATCH: {name}")
                print(f"    Megatron sum: {meg_sum:.10e}")
                print(f"    SGLang sum:   {sgl_sum:.10e}")
                print(f"    Diff:         {diff:.6e}")
        
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


def _compare_expert_weights_from_dict(
    megatron_weights: Mapping[str, torch.Tensor],
    layer_idx: int,
    server_host: str,
    server_port: int,
    results: Dict,
    verbose: bool,
):
    """Compare expert weights from CPU dict."""
    # Find expert weights in megatron_weights
    # Pattern: model.layers.{layer_idx}.mlp.experts.{expert_id}.gate_up_proj.weight
    expert_pattern = re.compile(rf"model\.layers\.{layer_idx}\.mlp\.experts\.(\d+)\.(gate_up_proj|down_proj)\.weight")
    
    expert_weights = {}
    for name, tensor in megatron_weights.items():
        match = expert_pattern.match(name)
        if match:
            expert_id = int(match.group(1))
            weight_type = match.group(2)
            if expert_id not in expert_weights:
                expert_weights[expert_id] = {}
            expert_weights[expert_id][weight_type] = (name, tensor)
    
    print(f"  Found {len(expert_weights)} experts in Megatron weights")
    
    for expert_id in sorted(expert_weights.keys()):
        for weight_type, (name, megatron_tensor) in expert_weights[expert_id].items():
            hf_name = name  # Same naming convention
            meg_sum = megatron_tensor.float().sum().item() if megatron_tensor.is_cuda else megatron_tensor.sum().item()
            
            sglang_tensor = _get_sglang_weight(server_host, server_port, hf_name, verbose)
            if sglang_tensor is not None:
                sgl_sum = sglang_tensor.sum().item()
                diff = abs(meg_sum - sgl_sum)
                results[hf_name] = (meg_sum, sgl_sum, diff)
                
                if diff > 1e-5:
                    print(f"  MISMATCH: Expert {expert_id} {weight_type}")
                    print(f"    Megatron sum: {meg_sum:.10e}")
                    print(f"    SGLang sum:   {sgl_sum:.10e}")
                elif verbose:
                    print(f"  OK: Expert {expert_id} {weight_type} (diff={diff:.6e})")
            else:
                if verbose:
                    print(f"  SKIP: Expert {expert_id} {weight_type} (SGLang returned None)")


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
    if rank != 0:
        return {}
    
    results = {}
    
    print(f"\n{'='*80}")
    print(f"[DEBUG WEIGHT SYNC] Comparing Megatron vs SGLang weights for layer {layer_idx}")
    print(f"{'='*80}")
    
    try:
        # Sync CUDA to ensure all operations are complete
        torch.cuda.synchronize()
        
        # First, print Megatron weights (safe operation)
        print("\n[Step 1] Reading Megatron weights...")
        megatron_weights = debug_print_megatron_weights(model, layer_idx, verbose=False)
        print(f"  Successfully read {len(megatron_weights)} weights from Megatron")
        
        # Get server info
        print("\n[Step 2] Getting SGLang server info...")
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
        print("\n[Step 3] Comparing regular weights...")
        for megatron_pattern, hf_name in weight_mappings.items():
            try:
                _compare_single_weight(model, megatron_pattern, hf_name, 
                                      server_host, server_port, results, verbose)
            except Exception as e:
                print(f"[ERROR] Failed to compare {megatron_pattern}: {e}")
        
        # Check expert weights (need special handling for EP)
        print("\n[Step 4] Comparing expert weights...")
        try:
            _compare_expert_weights(model, rollout_engines, layer_idx, 
                                   server_host, server_port, results, verbose)
        except Exception as e:
            print(f"[ERROR] Failed to compare expert weights: {e}")
            import traceback
            traceback.print_exc()
        
        # Summary
        print(f"\n{'='*80}")
        print(f"[SUMMARY] Weight comparison results:")
        has_error = False
        for name, (meg_sum, sgl_sum, diff) in results.items():
            status = "OK" if diff < 1e-5 else "MISMATCH"
            if diff >= 1e-5:
                has_error = True
                print(f"  {status}: {name}")
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


def _get_sglang_weight(server_host: str, server_port: int, hf_name: str, verbose: bool = False):
    """Get weight from SGLang via HTTP API."""
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
                return torch.tensor(processed, dtype=torch.float32)
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
        print(f"\n[Expert Weights] EP size={ep_size}, EP rank={ep_rank}")
    
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
