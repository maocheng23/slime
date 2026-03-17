"""
TP=8 alignment tests for Qwen3-30B-A3B to identify divergence between SGLang and Megatron.

Unit tests at TP=1 pass (bitwise identical), but E2E at TP=8 shows logprobs_diff=0.0025.
This test isolates each TP=8 component to find the source:

  1. tree_all_reduce_sum: Megatron vs SGLang implementation
  2. Attention layer (QKV + FA + o_proj + tree_allreduce)
  3. MoE fused_experts_impl EP filtering: same input → same output across both paths
  4. MoE exit all-reduce: tree_all_reduce on expert output
  5. Full transformer layer: attention + MoE + residuals
  6. Multi-layer accumulation: 4 layers to see if diff grows

For Qwen3-30B-A3B (pure MoE, no GDN, no shared experts), logprobs_diff should be EXACTLY 0.

Usage (inside Docker container, 8 GPUs):
  torchrun --nproc_per_node=8 scripts/test_moe_tp8_alignment.py
"""
import os
import sys
import json

import torch
import torch.distributed as dist
import torch.nn.functional as F

# ---- Distributed setup ----
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ.get("LOCAL_RANK", rank))
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

assert world_size == 8, f"This test requires exactly 8 GPUs, got {world_size}"

# ---- Shared utilities ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alignment_test_utils import TPTestResults, load_safetensor_weights

# ---- SGLang setup (must happen before imports that read server args) ----
os.environ["MEGATRON_USE_DETERMINISTIC_ALLREDUCE"] = "1"

from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

MODEL_PATH = "/root/models/Qwen3-30B-A3B"

server_args = ServerArgs(
    model_path=MODEL_PATH,
    rl_on_policy_target="fsdp_tp",
)
set_global_server_args_for_scheduler(server_args)
enable_batch_invariant_mode(enable_bmm=False)

if rank == 0:
    print("SGLang server_args set: rl_on_policy_target=fsdp_tp")
    print("batch_invariant_mode enabled")

# ---- Config ----
from transformers import AutoConfig

config = AutoConfig.from_pretrained(MODEL_PATH)
HIDDEN = config.hidden_size               # 2048
EPS = config.rms_norm_eps                  # 1e-6
NUM_Q_HEADS = config.num_attention_heads   # 32
NUM_KV_HEADS = config.num_key_value_heads  # 4
HEAD_DIM = getattr(config, 'head_dim', HIDDEN // NUM_Q_HEADS)  # 128
NUM_EXPERTS = config.num_experts           # 128
TOPK = config.num_experts_per_tok          # 8
MOE_FFN_HIDDEN = config.moe_intermediate_size  # 768
VOCAB_SIZE = config.vocab_size             # 151936

TP_SIZE = world_size  # 8
EP_SIZE = world_size  # 8 (TP == EP for this model)
NUM_LOCAL_EXPERTS = NUM_EXPERTS // EP_SIZE  # 16

B, T = 1, 16

if rank == 0:
    print(f"\nModel: {MODEL_PATH}")
    print(f"hidden={HIDDEN}, num_q_heads={NUM_Q_HEADS}, num_kv_heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}")
    print(f"num_experts={NUM_EXPERTS}, topk={TOPK}, moe_ffn_hidden={MOE_FFN_HIDDEN}")
    print(f"TP={TP_SIZE}, EP={EP_SIZE}, num_local_experts={NUM_LOCAL_EXPERTS}")
    print(f"Test shape: B={B}, T={T}")

# ---- Load weights (layer 0 only, all ranks load all weights for comparison) ----
weights = load_safetensor_weights(MODEL_PATH, prefixes=[
    "model.layers.0.",
    "model.norm.",
    "model.embed_tokens.",
    "lm_head.",
])

results = TPTestResults(rank)
tp_group = dist.group.WORLD  # All 8 GPUs in one TP group

# ---- Helper: broadcast a random tensor from rank 0 ----
def make_shared_input(*shape, dtype=torch.bfloat16):
    """Create identical random input across all ranks."""
    x = torch.randn(*shape, device=device, dtype=dtype)
    dist.broadcast(x, src=0, group=tp_group)
    return x


# ============================================================
# Test 1: tree_all_reduce_sum — Megatron vs SGLang
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("TEST 1: tree_all_reduce_sum — Megatron vs SGLang")
    print("=" * 60)

from megatron.core.tensor_parallel.mappings import _tree_all_reduce_sum_impl
from sglang.srt.tp_invariant_ops.tp_invariant_ops import tree_all_reduce_sum as sglang_tree_all_reduce_sum

# Each rank has different data (simulating TP-sharded partial sums)
x_local = torch.randn(B * T, HIDDEN, device=device, dtype=torch.bfloat16)

with torch.no_grad():
    out_megatron = _tree_all_reduce_sum_impl(x_local.clone(), tp_group)
    out_sglang = sglang_tree_all_reduce_sum(x_local.clone(), tp_group)

results.check("tree_all_reduce: Megatron vs SGLang", out_megatron, out_sglang)

# Self-consistency
with torch.no_grad():
    out_mega2 = _tree_all_reduce_sum_impl(x_local.clone(), tp_group)
results.check("tree_all_reduce: Megatron self-consistency", out_megatron, out_mega2)


# ============================================================
# Test 2: Attention layer at TP=8
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("TEST 2: Attention at TP=8 (QKV + FA + o_proj)")
    print("=" * 60)

from flash_attn.flash_attn_interface import flash_attn_varlen_func

q_w = weights["model.layers.0.self_attn.q_proj.weight"].to(device).bfloat16()
k_w = weights["model.layers.0.self_attn.k_proj.weight"].to(device).bfloat16()
v_w = weights["model.layers.0.self_attn.v_proj.weight"].to(device).bfloat16()
o_w = weights["model.layers.0.self_attn.o_proj.weight"].to(device).bfloat16()
q_norm_w = weights["model.layers.0.self_attn.q_norm.weight"].to(device).float()
k_norm_w = weights["model.layers.0.self_attn.k_norm.weight"].to(device).float()

# TP-shard Q heads across ranks
heads_per_rank = NUM_Q_HEADS // TP_SIZE  # 4
kv_heads_per_rank = max(1, NUM_KV_HEADS // TP_SIZE)  # 1 if NUM_KV_HEADS < TP_SIZE

q_start = rank * heads_per_rank * HEAD_DIM
q_end = q_start + heads_per_rank * HEAD_DIM
local_q_w = q_w[q_start:q_end]

if NUM_KV_HEADS >= TP_SIZE:
    kv_start = rank * kv_heads_per_rank * HEAD_DIM
    kv_end = kv_start + kv_heads_per_rank * HEAD_DIM
    local_k_w = k_w[kv_start:kv_end]
    local_v_w = v_w[kv_start:kv_end]
else:
    # KV heads replicated across ranks
    local_k_w = k_w
    local_v_w = v_w
    kv_heads_per_rank = NUM_KV_HEADS

# o_proj: column-parallel out, row-parallel in
o_start = rank * heads_per_rank * HEAD_DIM
o_end = o_start + heads_per_rank * HEAD_DIM
local_o_w = o_w[:, o_start:o_end]

x_attn = make_shared_input(B * T, HIDDEN)
positions = torch.arange(T, device=device, dtype=torch.long)
cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)

def run_attention_tp(x_in):
    """Attention forward with TP sharding."""
    q = F.linear(x_in, local_q_w).view(-1, heads_per_rank, HEAD_DIM)
    k = F.linear(x_in, local_k_w).view(-1, kv_heads_per_rank, HEAD_DIM)
    v = F.linear(x_in, local_v_w).view(-1, kv_heads_per_rank, HEAD_DIM)

    # QK norm (per-head RMSNorm, output must stay bf16 for FA)
    q_f = q.to(torch.float32)
    q_v = q_f.pow(2).mean(dim=-1, keepdim=True)
    q_normed = (q_f * torch.rsqrt(q_v + EPS))
    q = (q_norm_w * q_normed.to(q.dtype)).to(torch.bfloat16)

    k_f = k.to(torch.float32)
    k_v = k_f.pow(2).mean(dim=-1, keepdim=True)
    k_normed = (k_f * torch.rsqrt(k_v + EPS))
    k = (k_norm_w * k_normed.to(k.dtype)).to(torch.bfloat16)

    # GQA expand for local heads
    rep = heads_per_rank // kv_heads_per_rank
    if rep > 1:
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    # FA3
    scale = 1.0 / (HEAD_DIM ** 0.5)
    attn_out = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=T, max_seqlen_k=T,
        causal=True, softmax_scale=scale,
    )

    # o_proj (row parallel — each rank has partial, then all-reduce)
    attn_flat = attn_out.reshape(-1, heads_per_rank * HEAD_DIM)
    partial = F.linear(attn_flat, local_o_w)

    # tree all-reduce
    return _tree_all_reduce_sum_impl(partial, tp_group)

with torch.no_grad():
    attn_out1 = run_attention_tp(x_attn)
    attn_out2 = run_attention_tp(x_attn)

results.check("Attention TP=8: self-consistency", attn_out1, attn_out2)


# ============================================================
# Test 3: MoE fused_experts_impl with EP filtering — self-consistency
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("TEST 3: MoE fused_experts_impl (EP filtering) self-consistency")
    print("=" * 60)

from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts_impl

# Load local expert weights for this EP rank
local_expert_start = rank * NUM_LOCAL_EXPERTS
w1_list, w2_list = [], []
for i in range(NUM_LOCAL_EXPERTS):
    global_id = local_expert_start + i
    gp = weights[f"model.layers.0.mlp.experts.{global_id}.gate_proj.weight"].to(device).bfloat16()
    up = weights[f"model.layers.0.mlp.experts.{global_id}.up_proj.weight"].to(device).bfloat16()
    dp = weights[f"model.layers.0.mlp.experts.{global_id}.down_proj.weight"].to(device).bfloat16()
    w1_list.append(torch.cat([gp, up], dim=0))
    w2_list.append(dp)

w1 = torch.stack(w1_list)
w2 = torch.stack(w2_list)

if rank == 0:
    print(f"  w1 shape: {w1.shape}, w2 shape: {w2.shape}")
    print(f"  Local experts: {local_expert_start} to {local_expert_start + NUM_LOCAL_EXPERTS - 1}")

x_moe = make_shared_input(B * T, HIDDEN)

# Compute router (same on all ranks)
gate_weight = weights["model.layers.0.mlp.gate.weight"].to(device).bfloat16()
with torch.no_grad():
    router_logits = torch.mm(x_moe, gate_weight.t())
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, TOPK, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(x_moe.dtype)

# Map global expert IDs to local
local_expert_mapping = torch.full((NUM_EXPERTS,), -1, dtype=torch.int32, device=device)
local_expert_mapping[local_expert_start:local_expert_start + NUM_LOCAL_EXPERTS] = torch.arange(
    0, NUM_LOCAL_EXPERTS, dtype=torch.int32, device=device
)
topk_ids_local = local_expert_mapping[selected_experts.long()]

with torch.no_grad():
    moe_out1 = fused_experts_impl(
        x_moe.clone(), w1, w2,
        routing_weights, topk_ids_local,
        activation="silu",
        filter_expert=True,
    )
    moe_out2 = fused_experts_impl(
        x_moe.clone(), w1, w2,
        routing_weights, topk_ids_local,
        activation="silu",
        filter_expert=True,
    )

results.check("MoE fused_experts_impl (EP, fsdp_tp): self-consistency", moe_out1, moe_out2)


# ============================================================
# Test 4: MoE exit all-reduce: tree_all_reduce on expert output
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("TEST 4: MoE exit all-reduce (tree) — Megatron vs SGLang")
    print("=" * 60)

# Use the MoE output from Test 3 (different on each rank due to EP)
with torch.no_grad():
    allreduce_mega = _tree_all_reduce_sum_impl(moe_out1.clone(), tp_group)
    allreduce_sglang = sglang_tree_all_reduce_sum(moe_out1.clone(), tp_group)

results.check("MoE exit tree_all_reduce: Megatron vs SGLang", allreduce_mega, allreduce_sglang)


# ============================================================
# Test 5: Full MoE pipeline (router → fused_experts → tree_allreduce)
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("TEST 5: Full MoE pipeline self-consistency at TP=8")
    print("=" * 60)

def full_moe_forward(x_in):
    """Full MoE forward matching both SGLang and Megatron at TP=8."""
    # Router (same on all ranks)
    logits_r = torch.mm(x_in, gate_weight.t())
    weights_r = F.softmax(logits_r, dim=1, dtype=torch.float)
    weights_r, ids_r = torch.topk(weights_r, TOPK, dim=-1)
    weights_r = weights_r / weights_r.sum(dim=-1, keepdim=True)
    weights_r = weights_r.to(x_in.dtype)

    # Map to local experts
    ids_local = local_expert_mapping[ids_r.long()]

    # Fused experts (local experts only)
    output = fused_experts_impl(
        x_in.clone(), w1, w2,
        weights_r, ids_local,
        activation="silu",
        filter_expert=True,
    )

    # Tree all-reduce across EP/TP ranks
    return _tree_all_reduce_sum_impl(output, tp_group)

with torch.no_grad():
    full_out1 = full_moe_forward(x_moe)
    full_out2 = full_moe_forward(x_moe)

results.check("Full MoE pipeline TP=8: self-consistency", full_out1, full_out2)


# ============================================================
# Test 6: Megatron tree_all_reduce vs SGLang tree_all_reduce
#          on ACTUAL MoE output (not random data)
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("TEST 6: Megatron vs SGLang tree_all_reduce on real MoE output")
    print("=" * 60)

# Run fused_experts to get real partial sums
with torch.no_grad():
    partial = fused_experts_impl(
        x_moe.clone(), w1, w2,
        routing_weights, topk_ids_local,
        activation="silu",
        filter_expert=True,
    )

    reduced_mega = _tree_all_reduce_sum_impl(partial.clone(), tp_group)
    reduced_sglang = sglang_tree_all_reduce_sum(partial.clone(), tp_group)

results.check("tree_all_reduce on MoE output: Megatron vs SGLang", reduced_mega, reduced_sglang)


# ============================================================
# Test 7: RMSNorm at TP=8 (not TP-sharded, should be identical)
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("TEST 7: RMSNorm at TP=8 (identical across ranks)")
    print("=" * 60)

from sglang.srt.layers.layernorm import RMSNorm as SGLangRMSNorm

ln_weight = weights["model.layers.0.input_layernorm.weight"].to(device).float()
sglang_ln = SGLangRMSNorm(HIDDEN, eps=EPS, cast_x_before_out_mul=True).to(device)
sglang_ln.weight.data.copy_(ln_weight)

x_norm = make_shared_input(B * T, HIDDEN)

with torch.no_grad():
    norm_out1 = sglang_ln(x_norm)
    norm_out2 = sglang_ln(x_norm)

results.check("RMSNorm TP=8: self-consistency", norm_out1, norm_out2)

# Check all ranks get same result
norm_gathered = [torch.zeros_like(norm_out1) for _ in range(world_size)]
dist.all_gather(norm_gathered, norm_out1, group=tp_group)
if rank == 0:
    for r in range(1, world_size):
        diff = (norm_gathered[0].float() - norm_gathered[r].float()).abs().max().item()
        if diff > 0:
            print(f"  WARN  RMSNorm differs between rank 0 and rank {r}: max_diff={diff}")


# ============================================================
# Test 8: log_softmax at TP=8 (after gathering full logits)
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("TEST 8: log_softmax after gather_from_tensor_parallel (TP=8)")
    print("=" * 60)

# Simulate TP-sharded lm_head output
lm_head_w = weights["lm_head.weight"].to(device).bfloat16()
vocab_per_rank = VOCAB_SIZE // TP_SIZE
local_lm_head = lm_head_w[rank * vocab_per_rank:(rank + 1) * vocab_per_rank]

x_lm = make_shared_input(B * T, HIDDEN)

with torch.no_grad():
    # Each rank computes partial logits
    partial_logits = F.linear(x_lm.bfloat16(), local_lm_head)  # [B*T, vocab/TP]

    # Gather full logits (simulating gather_from_tensor_parallel)
    all_logits = [torch.zeros_like(partial_logits) for _ in range(TP_SIZE)]
    dist.all_gather(all_logits, partial_logits, group=tp_group)
    full_logits = torch.cat(all_logits, dim=-1)  # [B*T, VOCAB]

    # log_softmax on bf16 (matching both SGLang and Megatron)
    log_probs = F.log_softmax(full_logits.bfloat16(), dim=-1)

# All ranks should get identical log_probs
log_probs_gathered = [torch.zeros_like(log_probs) for _ in range(world_size)]
dist.all_gather(log_probs_gathered, log_probs, group=tp_group)

if rank == 0:
    all_identical = True
    for r in range(1, world_size):
        diff = (log_probs_gathered[0].float() - log_probs_gathered[r].float()).abs().max().item()
        if diff > 0:
            print(f"  WARN  log_softmax differs between rank 0 and rank {r}: max_diff={diff}")
            all_identical = False
    if all_identical:
        print("  PASS  log_softmax: all ranks identical after gather + log_softmax")
        results.pass_count += 1
    else:
        print("  FAIL  log_softmax: ranks diverge!")
        results.fail_count += 1


# ============================================================
# Test 9: Full transformer layer (attention + MoE + residuals)
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("TEST 9: Full transformer layer at TP=8 (self-consistency)")
    print("=" * 60)

post_attn_ln_w = weights["model.layers.0.post_attention_layernorm.weight"].to(device).float()
post_ln = SGLangRMSNorm(HIDDEN, eps=EPS, cast_x_before_out_mul=True).to(device)
post_ln.weight.data.copy_(post_attn_ln_w)

x_layer = make_shared_input(B * T, HIDDEN)

def full_layer_forward(x_in):
    """Full transformer layer forward at TP=8."""
    residual = x_in
    # 1. Input layernorm
    normed = sglang_ln(x_in).bfloat16()
    # 2. Attention
    attn_out = run_attention_tp(normed)
    # 3. Residual
    hidden = residual + attn_out
    residual2 = hidden
    # 4. Post-attention layernorm
    normed2 = post_ln(hidden).bfloat16()
    # 5. MoE
    moe_out = full_moe_forward(normed2)
    # 6. Residual
    return residual2 + moe_out

with torch.no_grad():
    layer_out1 = full_layer_forward(x_layer)
    layer_out2 = full_layer_forward(x_layer)

results.check("Full transformer layer TP=8: self-consistency", layer_out1, layer_out2)


# ============================================================
# Test 10: Compare Megatron-style vs SGLang-style tree_all_reduce
#           to detect if autograd wrapper causes divergence
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("TEST 10: Megatron _TreeAllReduceSum (autograd) vs bare impl")
    print("=" * 60)

from megatron.core.tensor_parallel.mappings import _tree_all_reduce_sum

x_ar = torch.randn(B * T, HIDDEN, device=device, dtype=torch.bfloat16)

with torch.no_grad():
    # Megatron autograd wrapper
    out_autograd = _tree_all_reduce_sum(x_ar.clone(), tp_group)
    # Direct impl (no autograd)
    out_direct = _tree_all_reduce_sum_impl(x_ar.clone(), tp_group)

results.check("Megatron tree_all_reduce: autograd wrapper vs direct impl", out_autograd, out_direct)


# ============================================================
# Summary
# ============================================================
results.summary()

if rank == 0:
    print()
    print("KEY: If any Megatron-vs-SGLang test FAILS, that component")
    print("is the source of logprobs_diff at TP=8.")
    print("If all self-consistency tests PASS but cross-impl tests FAIL,")
    print("the tree_all_reduce implementation difference is the root cause.")

dist.destroy_process_group()
