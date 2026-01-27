"""
Debug utilities for checking weight sync between Megatron and SGLang.

Usage:
    在 actor.py 的 update_weights() 之后调用：
    from .debug_weight_sync import debug_compare_all_weights
    debug_compare_all_weights(self.model, rollout_engines, layer_idx=0)
"""

import re
import logging
import torch
import torch.distributed as dist
import ray
import requests
from typing import Optional, List, Dict, Tuple
from megatron.core import mpu

logger = logging.getLogger(__name__)


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
    
    # Get server info
    try:
        server_info = ray.get(rollout_engines[0].get_server_info.remote())
        server_host = server_info["server_host"]
        server_port = server_info["server_port"]
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
    for megatron_pattern, hf_name in weight_mappings.items():
        _compare_single_weight(model, megatron_pattern, hf_name, 
                              server_host, server_port, results, verbose)
    
    # Check expert weights (need special handling for EP)
    _compare_expert_weights(model, rollout_engines, layer_idx, 
                           server_host, server_port, results, verbose)
    
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
    
    return results


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
                megatron_weight = param.data.detach().cpu().float()
                break
        if megatron_weight is not None:
            break
    
    if megatron_weight is None:
        if verbose:
            print(f"[SKIP] Megatron weight not found: {megatron_pattern}")
        return
    
    # Get SGLang weight
    try:
        response = requests.post(
            f"http://{server_host}:{server_port}/get_weights_by_name",
            json={"name": hf_name, "truncate_size": -1},
            timeout=30
        )
        if response.status_code == 200:
            sglang_data = response.json()
            if sglang_data is not None:
                sglang_weight = torch.tensor(sglang_data, dtype=torch.float32)
                
                meg_sum = megatron_weight.sum().item()
                sgl_sum = sglang_weight.sum().item()
                diff = abs(meg_sum - sgl_sum)
                
                results[hf_name] = (meg_sum, sgl_sum, diff)
            else:
                if verbose:
                    print(f"[SKIP] SGLang returned None for {hf_name}")
        else:
            if verbose:
                print(f"[SKIP] SGLang API error {response.status_code} for {hf_name}")
    except Exception as e:
        if verbose:
            print(f"[SKIP] Error getting SGLang weight {hf_name}: {e}")


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
    ep_size = mpu.get_expert_model_parallel_world_size()
    ep_rank = mpu.get_expert_model_parallel_rank()
    
    if verbose:
        print(f"\n[Expert Weights] EP size={ep_size}, EP rank={ep_rank}")
    
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
                    if "linear_fc1" in name:
                        local_experts_w1[expert_id] = param.data.detach().cpu().float()
                    elif "linear_fc2" in name:
                        local_experts_w2[expert_id] = param.data.detach().cpu().float()
    
    if verbose:
        print(f"  Found {len(local_experts_w1)} local experts for fc1")
        print(f"  Found {len(local_experts_w2)} local experts for fc2")
        print(f"  Local expert IDs (fc1): {sorted(local_experts_w1.keys())}")
    
    # Compare with SGLang
    # SGLang expert name pattern: model.layers.{layer}.mlp.experts.{expert_id}.gate_up_proj.weight
    for expert_id, megatron_w1 in sorted(local_experts_w1.items()):
        hf_name = f"model.layers.{layer_idx}.mlp.experts.{expert_id}.gate_up_proj.weight"
        try:
            response = requests.post(
                f"http://{server_host}:{server_port}/get_weights_by_name",
                json={"name": hf_name, "truncate_size": -1},
                timeout=30
            )
            if response.status_code == 200:
                sglang_data = response.json()
                if sglang_data is not None:
                    sglang_w1 = torch.tensor(sglang_data, dtype=torch.float32)
                    
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
        except Exception as e:
            if verbose:
                print(f"  [ERROR] Cannot get SGLang expert {expert_id} fc1: {e}")
    
    for expert_id, megatron_w2 in sorted(local_experts_w2.items()):
        hf_name = f"model.layers.{layer_idx}.mlp.experts.{expert_id}.down_proj.weight"
        try:
            response = requests.post(
                f"http://{server_host}:{server_port}/get_weights_by_name",
                json={"name": hf_name, "truncate_size": -1},
                timeout=30
            )
            if response.status_code == 200:
                sglang_data = response.json()
                if sglang_data is not None:
                    sglang_w2 = torch.tensor(sglang_data, dtype=torch.float32)
                    
                    meg_sum = megatron_w2.sum().item()
                    sgl_sum = sglang_w2.sum().item()
                    diff = abs(meg_sum - sgl_sum)
                    
                    results[hf_name] = (meg_sum, sgl_sum, diff)
        except Exception as e:
            if verbose:
                print(f"  [ERROR] Cannot get SGLang expert {expert_id} fc2: {e}")


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
