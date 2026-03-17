"""
TP=8 cross-implementation test: Megatron SelfAttention QKV vs SGLang QKVParallelLinear.

All TP=8 self-consistency tests pass, but E2E still shows ~1.78e-05 diff.
This test compares the ACTUAL per-rank Q/K/V outputs between:
  - Megatron: SGLangLayerNormColumnParallelLinear → all_gather → split → QK norm → RoPE
  - SGLang: QKVParallelLinear → split → QK norm → RoPE
  - Manual: F.linear on full weights → split → QK norm → RoPE (reference)

For Qwen3-30B-A3B with num_kv_heads=4 < TP=8, Megatron triggers the all-gather path.

Usage: torchrun --nproc_per_node=8 scripts/test_moe_tp8_qkv_compare.py
"""
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ.get("LOCAL_RANK", rank))
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

assert world_size == 8, f"Requires 8 GPUs, got {world_size}"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alignment_test_utils import TPTestResults, load_safetensor_weights

os.environ["MEGATRON_USE_DETERMINISTIC_ALLREDUCE"] = "1"

from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

MODEL_PATH = "/root/models/Qwen3-30B-A3B"
server_args = ServerArgs(model_path=MODEL_PATH, rl_on_policy_target="fsdp_tp")
set_global_server_args_for_scheduler(server_args)
enable_batch_invariant_mode(enable_bmm=False)

from transformers import AutoConfig
config = AutoConfig.from_pretrained(MODEL_PATH)

HIDDEN = config.hidden_size               # 2048
EPS = config.rms_norm_eps                  # 1e-6
NUM_Q_HEADS = config.num_attention_heads   # 32
NUM_KV_HEADS = config.num_key_value_heads  # 4
HEAD_DIM = getattr(config, 'head_dim', HIDDEN // NUM_Q_HEADS)  # 128
TP = world_size                            # 8

Q_HEADS_PER_RANK = NUM_Q_HEADS // TP       # 4
# KV heads < TP: each rank gets all KV heads (replicated)
KV_HEADS_PER_RANK = NUM_KV_HEADS           # 4 (replicated, not sharded)

B, T = 1, 16

if rank == 0:
    print(f"Model: {MODEL_PATH}")
    print(f"hidden={HIDDEN}, q_heads={NUM_Q_HEADS}, kv_heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}")
    print(f"TP={TP}, q_heads/rank={Q_HEADS_PER_RANK}, kv_heads/rank={KV_HEADS_PER_RANK}")
    print(f"num_query_groups({NUM_KV_HEADS}) < TP({TP}): Megatron uses QKV all-gather path")

weights = load_safetensor_weights(MODEL_PATH, prefixes=[
    "model.layers.0.", "model.norm.", "model.embed_tokens.", "lm_head.",
])

results = TPTestResults(rank)
tp_group = dist.group.WORLD

def make_shared_input(*shape, dtype=torch.bfloat16):
    x = torch.randn(*shape, device=device, dtype=dtype)
    dist.broadcast(x, src=0, group=tp_group)
    return x

# Load full weights (all ranks load all)
q_proj_w = weights["model.layers.0.self_attn.q_proj.weight"].to(device).bfloat16()   # [4096, 2048]
k_proj_w = weights["model.layers.0.self_attn.k_proj.weight"].to(device).bfloat16()   # [512, 2048]
v_proj_w = weights["model.layers.0.self_attn.v_proj.weight"].to(device).bfloat16()   # [512, 2048]
q_norm_w = weights["model.layers.0.self_attn.q_norm.weight"].to(device).float()      # [128]
k_norm_w = weights["model.layers.0.self_attn.k_norm.weight"].to(device).float()      # [128]
ln_w = weights["model.layers.0.input_layernorm.weight"].to(device).float()            # [2048]

from sglang.srt.layers.layernorm import RMSNorm as SGLangRMSNorm

# RoPE (same on all ranks)
base = getattr(config, 'rope_theta', 1000000)
inv_freq = 1.0 / (base ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device=device) / HEAD_DIM))
positions = torch.arange(T, dtype=torch.float32, device=device)
freqs = torch.outer(positions, inv_freq)
cos_rope = torch.cos(freqs)
sin_rope = torch.sin(freqs)

def rmsnorm(x, weight, eps):
    """SGLang-matching RMSNorm: fp32 compute, cast_x_before_out_mul=True."""
    orig_dtype = x.dtype
    x = x.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return (weight * x.to(orig_dtype))

def apply_rope_f32(x, cos, sin):
    """SGLang-matching RoPE: float32 computation."""
    orig = x.dtype
    x = x.float()
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    c = cos.unsqueeze(1).float()
    s = sin.unsqueeze(1).float()
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).to(orig)

# ============================================================
# TEST 1: Reference (manual) forward — full weights, no TP
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("TEST 1: Reference forward (manual, full weights)")
    print("=" * 60)

x = make_shared_input(T, HIDDEN)

with torch.no_grad():
    normed_ref = rmsnorm(x, ln_w, EPS)

    q_ref = F.linear(normed_ref, q_proj_w).view(T, NUM_Q_HEADS, HEAD_DIM)
    k_ref = F.linear(normed_ref, k_proj_w).view(T, NUM_KV_HEADS, HEAD_DIM)
    v_ref = F.linear(normed_ref, v_proj_w).view(T, NUM_KV_HEADS, HEAD_DIM)

    # QK norm (per-head)
    q_ref = rmsnorm(q_ref.reshape(-1, HEAD_DIM), q_norm_w, EPS).view(T, NUM_Q_HEADS, HEAD_DIM)
    k_ref = rmsnorm(k_ref.reshape(-1, HEAD_DIM), k_norm_w, EPS).view(T, NUM_KV_HEADS, HEAD_DIM)

    # RoPE
    q_ref = apply_rope_f32(q_ref, cos_rope, sin_rope)
    k_ref = apply_rope_f32(k_ref, cos_rope, sin_rope)

    # Extract this rank's Q heads (same slicing as Megatron would do)
    q_ref_local = q_ref[:, rank * Q_HEADS_PER_RANK : (rank + 1) * Q_HEADS_PER_RANK, :]
    # K/V are replicated (all ranks have all KV heads)
    k_ref_local = k_ref
    v_ref_local = v_ref

if rank == 0:
    print(f"  q_ref_local: {q_ref_local.shape}, k_ref_local: {k_ref_local.shape}")

# ============================================================
# TEST 2: SGLang-style TP forward (QKVParallelLinear emulation)
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("TEST 2: SGLang-style TP forward")
    print("=" * 60)

with torch.no_grad():
    normed_sg = rmsnorm(x, ln_w, EPS)

    # SGLang QKVParallelLinear: Q is TP-sharded, K/V are replicated
    q_w_local = q_proj_w[rank * Q_HEADS_PER_RANK * HEAD_DIM : (rank + 1) * Q_HEADS_PER_RANK * HEAD_DIM, :]
    q_sg = F.linear(normed_sg, q_w_local).view(T, Q_HEADS_PER_RANK, HEAD_DIM)
    k_sg = F.linear(normed_sg, k_proj_w).view(T, NUM_KV_HEADS, HEAD_DIM)
    v_sg = F.linear(normed_sg, v_proj_w).view(T, NUM_KV_HEADS, HEAD_DIM)

    # QK norm
    q_sg = rmsnorm(q_sg.reshape(-1, HEAD_DIM), q_norm_w, EPS).view(T, Q_HEADS_PER_RANK, HEAD_DIM)
    k_sg = rmsnorm(k_sg.reshape(-1, HEAD_DIM), k_norm_w, EPS).view(T, NUM_KV_HEADS, HEAD_DIM)

    # RoPE
    q_sg = apply_rope_f32(q_sg, cos_rope, sin_rope)
    k_sg = apply_rope_f32(k_sg, cos_rope, sin_rope)

q_diff_sg = (q_ref_local.float() - q_sg.float()).abs().max().item()
k_diff_sg = (k_ref_local.float() - k_sg.float()).abs().max().item()
results.check("SGLang Q vs Reference", q_ref_local, q_sg)
results.check("SGLang K vs Reference", k_ref_local, k_sg)

# ============================================================
# TEST 3: Megatron-style TP forward (fused QKV + all_gather + split)
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("TEST 3: Megatron-style TP forward (QKV all-gather path)")
    print("=" * 60)

with torch.no_grad():
    normed_mega = rmsnorm(x, ln_w, EPS)

    # Megatron fused QKV: interleaved weight layout
    # Weight layout: [Q_group0, K_group0, V_group0, Q_group1, K_group1, V_group1, ...]
    # Each group: q_heads_per_group = NUM_Q_HEADS / NUM_KV_HEADS = 8
    q_per_group = NUM_Q_HEADS // NUM_KV_HEADS  # 8
    qkv_per_group = (q_per_group + 2) * HEAD_DIM  # (8+2)*128 = 1280

    # Build fused QKV weight in Megatron's interleaved format
    qkv_w_parts = []
    for g in range(NUM_KV_HEADS):
        q_start = g * q_per_group * HEAD_DIM
        q_end = (g + 1) * q_per_group * HEAD_DIM
        k_start = g * HEAD_DIM
        k_end = (g + 1) * HEAD_DIM
        qkv_w_parts.append(q_proj_w[q_start:q_end])
        qkv_w_parts.append(k_proj_w[k_start:k_end])
        qkv_w_parts.append(v_proj_w[k_start:k_end])
    qkv_w_full = torch.cat(qkv_w_parts, dim=0)  # [5120, 2048]

    # TP shard: each rank gets 1/TP of the fused QKV
    shard_size = qkv_w_full.shape[0] // TP
    qkv_w_local = qkv_w_full[rank * shard_size : (rank + 1) * shard_size]

    # Forward: each rank computes local QKV shard
    qkv_local = F.linear(normed_mega, qkv_w_local)  # [T, shard_size]

    # All-gather across TP to reconstruct full QKV
    qkv_gathered_list = [torch.empty_like(qkv_local) for _ in range(TP)]
    dist.all_gather(qkv_gathered_list, qkv_local, group=tp_group)
    qkv_gathered = torch.cat(qkv_gathered_list, dim=-1)  # [T, full_qkv_dim]

    # Select this rank's KV group
    # Megatron: idx = rank // (TP // num_kv_heads) = rank // 2
    group_idx = rank // (TP // NUM_KV_HEADS)
    group_size = qkv_per_group  # 1280
    qkv_group = qkv_gathered[:, group_idx * group_size : (group_idx + 1) * group_size]

    # Split into Q, K, V within the group
    q_group = qkv_group[:, : q_per_group * HEAD_DIM].view(T, q_per_group, HEAD_DIM)
    k_group = qkv_group[:, q_per_group * HEAD_DIM : (q_per_group + 1) * HEAD_DIM].view(T, 1, HEAD_DIM)
    v_group = qkv_group[:, (q_per_group + 1) * HEAD_DIM :].view(T, 1, HEAD_DIM)

    # Further index Q for this rank within the group
    # Megatron: idx_in_group = rank % (TP // num_kv_heads) = rank % 2
    idx_in_group = rank % (TP // NUM_KV_HEADS)
    q_per_rank_in_group = q_per_group // (TP // NUM_KV_HEADS)  # 8 // 2 = 4
    q_mega = q_group[:, idx_in_group * q_per_rank_in_group : (idx_in_group + 1) * q_per_rank_in_group, :]

    # K/V: replicate the group's single KV head to match SGLang's convention
    # Actually in Megatron, each rank gets the KV head from its group
    # Need to reconstruct full KV from all groups for comparison
    k_all_groups = []
    v_all_groups = []
    for g in range(NUM_KV_HEADS):
        g_start = g * group_size
        k_g = qkv_gathered[:, g_start + q_per_group * HEAD_DIM : g_start + (q_per_group + 1) * HEAD_DIM]
        v_g = qkv_gathered[:, g_start + (q_per_group + 1) * HEAD_DIM : g_start + (q_per_group + 2) * HEAD_DIM]
        k_all_groups.append(k_g)
        v_all_groups.append(v_g)
    k_mega = torch.cat(k_all_groups, dim=-1).view(T, NUM_KV_HEADS, HEAD_DIM)
    v_mega = torch.cat(v_all_groups, dim=-1).view(T, NUM_KV_HEADS, HEAD_DIM)

    # QK norm
    q_mega = rmsnorm(q_mega.reshape(-1, HEAD_DIM), q_norm_w, EPS).view(T, Q_HEADS_PER_RANK, HEAD_DIM)
    k_mega = rmsnorm(k_mega.reshape(-1, HEAD_DIM), k_norm_w, EPS).view(T, NUM_KV_HEADS, HEAD_DIM)

    # RoPE
    q_mega = apply_rope_f32(q_mega, cos_rope, sin_rope)
    k_mega = apply_rope_f32(k_mega, cos_rope, sin_rope)

results.check("Megatron Q vs Reference", q_ref_local, q_mega)
results.check("Megatron K vs Reference", k_ref_local, k_mega)
results.check("Megatron Q vs SGLang Q", q_sg, q_mega)
results.check("Megatron K vs SGLang K", k_sg, k_mega)

# ============================================================
# TEST 4: Direct comparison — is the fused QKV matmul identical to separate?
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("TEST 4: Fused QKV matmul vs separate Q/K/V matmuls")
    print("=" * 60)

with torch.no_grad():
    # Fused: one matmul with concatenated weight
    simple_qkv_w = torch.cat([q_proj_w, k_proj_w, v_proj_w], dim=0)
    qkv_fused = F.linear(normed_ref, simple_qkv_w)
    q_fused = qkv_fused[:, :NUM_Q_HEADS * HEAD_DIM]
    k_fused = qkv_fused[:, NUM_Q_HEADS * HEAD_DIM : NUM_Q_HEADS * HEAD_DIM + NUM_KV_HEADS * HEAD_DIM]
    v_fused = qkv_fused[:, NUM_Q_HEADS * HEAD_DIM + NUM_KV_HEADS * HEAD_DIM:]

    # Separate
    q_sep = F.linear(normed_ref, q_proj_w)
    k_sep = F.linear(normed_ref, k_proj_w)

results.check("Fused Q vs Separate Q", q_fused, q_sep)
results.check("Fused K vs Separate K", k_fused, k_sep)

# ============================================================
# Summary
# ============================================================
results.summary()
if rank == 0:
    print("\nIf Megatron vs SGLang shows diff but both match Reference,")
    print("the issue is in how Megatron's all-gather path reconstructs Q/K/V.")
    print("If all three match, the E2E diff comes from elsewhere.")
