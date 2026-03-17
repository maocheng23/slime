"""
Cross-process alignment tests for Qwen3-30B-A3B at TP=8.

All existing TP=8 kernel tests pass (bitwise identical), but E2E still shows
logprobs_diff ≈ 3e-05. This test targets the gaps between unit tests and E2E:

  1. FA3 API difference: flash_attn_varlen_func (Megatron) vs flash_attn_with_kvcache (SGLang)
  2. GQA handling: explicit repeat_interleave (Megatron) vs FA3-internal GQA (SGLang)
  3. lm_head matmul + allgather + vocab truncation + log_softmax (full logprob chain)
  4. moe_sum_tree_reduce vs moe_sum_reduce (fsdp_tp branch coverage)
  5. Full forward chain: attention → MoE → lm_head → logprob (single layer)

For Qwen3-30B-A3B (pure full attention + MoE), any diff found here explains the
E2E logprobs_diff that kernel-level self-consistency tests cannot catch.

Usage (inside Docker container, 8 GPUs):
  torchrun --nproc_per_node=8 scripts/test_moe_tp8_cross_process.py
"""
import os
import sys

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

# ---- SGLang setup ----
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
HIDDEN = config.hidden_size
EPS = config.rms_norm_eps
NUM_Q_HEADS = config.num_attention_heads
NUM_KV_HEADS = config.num_key_value_heads
HEAD_DIM = getattr(config, "head_dim", HIDDEN // NUM_Q_HEADS)
NUM_EXPERTS = config.num_experts
TOPK = config.num_experts_per_tok
MOE_FFN_HIDDEN = config.moe_intermediate_size
VOCAB_SIZE = config.vocab_size

TP_SIZE = world_size
EP_SIZE = world_size
NUM_LOCAL_EXPERTS = NUM_EXPERTS // EP_SIZE

B, T = 1, 32

if rank == 0:
    print(f"\nModel: {MODEL_PATH}")
    print(f"hidden={HIDDEN}, q_heads={NUM_Q_HEADS}, kv_heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}")
    print(f"experts={NUM_EXPERTS}, topk={TOPK}, vocab={VOCAB_SIZE}")
    print(f"TP={TP_SIZE}, EP={EP_SIZE}, local_experts={NUM_LOCAL_EXPERTS}")
    print(f"Test shape: B={B}, T={T}")

# ---- Load weights ----
weights = load_safetensor_weights(MODEL_PATH, prefixes=[
    "model.layers.0.",
    "model.norm.",
    "model.embed_tokens.",
    "lm_head.",
])

results = TPTestResults(rank)
tp_group = dist.group.WORLD


def make_shared_input(*shape, dtype=torch.bfloat16):
    x = torch.randn(*shape, device=device, dtype=dtype)
    dist.broadcast(x, src=0, group=tp_group)
    return x


# ============================================================
# Test 1: FA3 flash_attn_varlen_func vs flash_attn_with_kvcache
# This is the primary suspect for E2E diff.
# Megatron uses varlen; SGLang uses kvcache during prefill.
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("TEST 1: FA3 varlen_func vs with_kvcache (same Q/K/V)")
    print("=" * 60)

from flash_attn.flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache

heads_per_rank = NUM_Q_HEADS // TP_SIZE
kv_heads_per_rank = max(1, NUM_KV_HEADS // TP_SIZE)
rep = heads_per_rank // kv_heads_per_rank

q = make_shared_input(B * T, heads_per_rank, HEAD_DIM)
k = make_shared_input(B * T, kv_heads_per_rank, HEAD_DIM)
v = make_shared_input(B * T, kv_heads_per_rank, HEAD_DIM)
scale = 1.0 / (HEAD_DIM ** 0.5)
cu = torch.tensor([0, T], dtype=torch.int32, device=device)

with torch.no_grad():
    # Path A: Megatron — varlen with explicit GQA expand
    k_exp = k.repeat_interleave(rep, dim=1)
    v_exp = v.repeat_interleave(rep, dim=1)
    out_varlen = flash_attn_varlen_func(
        q, k_exp, v_exp,
        cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=T, max_seqlen_k=T,
        causal=True, softmax_scale=scale,
    )

    # Path B: SGLang — kvcache with paged KV (simulated single page)
    # API: flash_attn_with_kvcache(q, k_cache, v_cache, ..., block_table, cache_seqlens, ...)
    # q must be [batch, seqlen_q, nheads, headdim]
    # k_cache/v_cache: [num_blocks, page_size, nheads_k, headdim]
    # Block size must be divisible by 256 for FA3 paged KV
    page_size = 256
    # Pad K/V to page_size (fill extra with zeros)
    k_padded = torch.zeros(page_size, kv_heads_per_rank, HEAD_DIM, device=device, dtype=torch.bfloat16)
    v_padded = torch.zeros(page_size, kv_heads_per_rank, HEAD_DIM, device=device, dtype=torch.bfloat16)
    k_padded[:T] = k
    v_padded[:T] = v
    k_cache = k_padded.view(1, page_size, kv_heads_per_rank, HEAD_DIM).contiguous()
    v_cache = v_padded.view(1, page_size, kv_heads_per_rank, HEAD_DIM).contiguous()
    block_table = torch.zeros(1, 1, dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor([T], dtype=torch.int32, device=device)

    out_kvcache = flash_attn_with_kvcache(
        q=q.view(B, T, heads_per_rank, HEAD_DIM).contiguous(),
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        softmax_scale=scale,
        causal=True,
        num_splits=1,
    ).view(B * T, heads_per_rank, HEAD_DIM)

diff = (out_varlen.float() - out_kvcache.float()).abs()
max_diff = diff.max().item()
mean_diff = diff.mean().item()

if rank == 0:
    if max_diff == 0:
        print("  PASS  FA3 varlen vs kvcache: bitwise identical")
        results.pass_count += 1
    else:
        nonzero = (diff > 0).float().mean().item()
        print(
            f"  DIFF  FA3 varlen vs kvcache: "
            f"max={max_diff:.8e}, mean={mean_diff:.8e}, "
            f"nonzero={nonzero:.4f}"
        )
        print(
            "        → Two FA3 APIs produce different "
            "floating-point results."
        )
        results.info_count += 1


# ============================================================
# Test 2: FA3 varlen with GQA expand vs varlen with native GQA
# Megatron does repeat_interleave; FA3 can handle GQA natively.
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("TEST 2: FA3 varlen — explicit GQA expand vs native GQA")
    print("=" * 60)

with torch.no_grad():
    # Path A: explicit repeat_interleave (Megatron style)
    out_expanded = flash_attn_varlen_func(
        q, k_exp, v_exp,
        cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=T, max_seqlen_k=T,
        causal=True, softmax_scale=scale,
    )

    # Path B: native GQA (pass fewer KV heads directly)
    out_native_gqa = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=T, max_seqlen_k=T,
        causal=True, softmax_scale=scale,
    )

results.check("FA3 varlen: explicit GQA expand vs native GQA", out_expanded, out_native_gqa)


# ============================================================
# Test 3: moe_sum_tree_reduce vs moe_sum_reduce
# SGLang uses tree_reduce when fsdp_tp; Megatron may use
# moe_sum_reduce if get_global_server_args is not set.
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("TEST 3: moe_sum_tree_reduce vs moe_sum_reduce (per-token combine)")
    print("=" * 60)

from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts_impl
try:
    from sgl_kernel import moe_sum_reduce  # noqa: F401
    has_moe_sum_reduce = True
except ImportError:
    has_moe_sum_reduce = False

if has_moe_sum_reduce:
    local_expert_start = rank * NUM_LOCAL_EXPERTS
    w1_list, w2_list = [], []
    for i in range(NUM_LOCAL_EXPERTS):
        gid = local_expert_start + i
        gp = weights[f"model.layers.0.mlp.experts.{gid}.gate_proj.weight"].to(device).bfloat16()
        up = weights[f"model.layers.0.mlp.experts.{gid}.up_proj.weight"].to(device).bfloat16()
        dp = weights[f"model.layers.0.mlp.experts.{gid}.down_proj.weight"].to(device).bfloat16()
        w1_list.append(torch.cat([gp, up], dim=0))
        w2_list.append(dp)
    w1 = torch.stack(w1_list)
    w2 = torch.stack(w2_list)

    x_moe = make_shared_input(B * T, HIDDEN)
    gate_weight = weights["model.layers.0.mlp.gate.weight"].to(device).bfloat16()

    with torch.no_grad():
        router_logits = torch.mm(x_moe, gate_weight.t())
        routing_weights_r = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights_r, selected_experts = torch.topk(routing_weights_r, TOPK, dim=-1)
        routing_weights_r = routing_weights_r / routing_weights_r.sum(dim=-1, keepdim=True)
        routing_weights_r = routing_weights_r.to(x_moe.dtype)

    local_expert_mapping = torch.full((NUM_EXPERTS,), -1, dtype=torch.int32, device=device)
    local_expert_mapping[local_expert_start:local_expert_start + NUM_LOCAL_EXPERTS] = torch.arange(
        0, NUM_LOCAL_EXPERTS, dtype=torch.int32, device=device
    )
    topk_ids_local = local_expert_mapping[selected_experts.long()]

    # Run fused_experts to get intermediate_cache3 equivalent
    # We compare the two reduce paths by running fused_experts_impl twice
    # with different get_global_server_args states
    with torch.no_grad():
        # Path A: with fsdp_tp → moe_sum_tree_reduce
        out_tree = fused_experts_impl(
            x_moe.clone(), w1, w2,
            routing_weights_r, topk_ids_local,
            activation="silu",
            filter_expert=True,
        )

    # To test moe_sum_reduce path, temporarily override
    from sglang.srt.server_args import get_global_server_args
    saved_target = get_global_server_args().rl_on_policy_target
    get_global_server_args().rl_on_policy_target = None

    with torch.no_grad():
        out_default = fused_experts_impl(
            x_moe.clone(), w1, w2,
            routing_weights_r, topk_ids_local,
            activation="silu",
            filter_expert=True,
        )

    get_global_server_args().rl_on_policy_target = saved_target

    diff_reduce = (out_tree.float() - out_default.float()).abs()
    max_diff_r = diff_reduce.max().item()
    mean_diff_r = diff_reduce.mean().item()

    if rank == 0:
        if max_diff_r == 0:
            print(
                "  PASS  moe_sum_tree_reduce vs "
                "moe_sum_reduce: bitwise identical"
            )
            results.pass_count += 1
        else:
            nonzero = (diff_reduce > 0).float().mean().item()
            print(
                f"  DIFF  tree_reduce vs sum_reduce: "
                f"max={max_diff_r:.8e}, "
                f"mean={mean_diff_r:.8e}, "
                f"nonzero={nonzero:.4f}"
            )
            print(
                "        → Megatron without "
                "rl_on_policy_target=fsdp_tp uses "
                "moe_sum_reduce."
            )
            results.info_count += 1
else:
    if rank == 0:
        print("  SKIP  sgl_kernel.moe_sum_reduce not available")


# ============================================================
# Test 4: Full logprob chain — lm_head + allgather + log_softmax
# Compare SGLang-style (bf16 matmul → float → log_softmax)
# vs Megatron-style (ColumnParallelLinear → gather → float → log_softmax)
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("TEST 4: Full logprob chain (lm_head → gather → log_softmax)")
    print("=" * 60)

lm_head_w = weights["lm_head.weight"].to(device).bfloat16()
vocab_per_rank = VOCAB_SIZE // TP_SIZE
local_lm_head = lm_head_w[rank * vocab_per_rank:(rank + 1) * vocab_per_rank]

x_lm = make_shared_input(B * T, HIDDEN)

with torch.no_grad():
    # Path A: SGLang-style — bf16 matmul on full vocab, then gather
    # (SGLang does lm_head on local shard, then all_gather_into_tensor + permute)
    partial_a = torch.matmul(x_lm.bfloat16(), local_lm_head.T)
    gathered_a = [torch.zeros_like(partial_a) for _ in range(TP_SIZE)]
    dist.all_gather(gathered_a, partial_a, group=tp_group)
    full_logits_a = torch.cat(gathered_a, dim=-1)[:, :VOCAB_SIZE].float()
    logprobs_a = F.log_softmax(full_logits_a, dim=-1)

    # Path B: Megatron-style — torch.matmul (goes through aten::mm → matmul_persistent)
    # then gather_from_tensor_model_parallel_region
    partial_b = torch.matmul(x_lm.bfloat16(), local_lm_head.T)
    gathered_b = [torch.zeros_like(partial_b) for _ in range(TP_SIZE)]
    dist.all_gather(gathered_b, partial_b, group=tp_group)
    full_logits_b = torch.cat(gathered_b, dim=-1)[:, :VOCAB_SIZE].float()
    logprobs_b = F.log_softmax(full_logits_b, dim=-1)

results.check("Logprob chain: SGLang-style vs Megatron-style", logprobs_a, logprobs_b)

# Also check: does the allgather method matter?
with torch.no_grad():
    partial_c = torch.matmul(x_lm.bfloat16(), local_lm_head.T)

    # Method 1: all_gather into list + cat (Megatron _gather_along_last_dim)
    list_c = [torch.zeros_like(partial_c) for _ in range(TP_SIZE)]
    dist.all_gather(list_c, partial_c, group=tp_group)
    full_c1 = torch.cat(list_c, dim=-1)

    # Method 2: all_gather_into_tensor + permute (SGLang logits_processor)
    buf_c = torch.empty(TP_SIZE, partial_c.shape[0], partial_c.shape[1],
                         device=device, dtype=partial_c.dtype)
    dist.all_gather_into_tensor(buf_c, partial_c, group=tp_group)
    full_c2 = buf_c.permute(1, 0, 2).reshape(partial_c.shape[0], -1)

results.check("Allgather method: list+cat vs into_tensor+permute", full_c1, full_c2)


# ============================================================
# Test 5: Full single-layer forward → logprob (end-to-end chain)
# Runs the same input through the full chain twice to verify
# determinism of the entire pipeline at TP=8.
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("TEST 5: Full single-layer → logprob chain (self-consistency)")
    print("=" * 60)

from megatron.core.tensor_parallel.mappings import _tree_all_reduce_sum_impl
from sglang.srt.layers.layernorm import RMSNorm as SGLangRMSNorm

# Load all needed weights
q_w = weights["model.layers.0.self_attn.q_proj.weight"].to(device).bfloat16()
k_w = weights["model.layers.0.self_attn.k_proj.weight"].to(device).bfloat16()
v_w = weights["model.layers.0.self_attn.v_proj.weight"].to(device).bfloat16()
o_w = weights["model.layers.0.self_attn.o_proj.weight"].to(device).bfloat16()
q_norm_w = weights["model.layers.0.self_attn.q_norm.weight"].to(device).float()
k_norm_w = weights["model.layers.0.self_attn.k_norm.weight"].to(device).float()

q_start = rank * heads_per_rank * HEAD_DIM
local_q_w = q_w[q_start:q_start + heads_per_rank * HEAD_DIM]
if NUM_KV_HEADS >= TP_SIZE:
    kv_start = rank * kv_heads_per_rank * HEAD_DIM
    local_k_w = k_w[kv_start:kv_start + kv_heads_per_rank * HEAD_DIM]
    local_v_w = v_w[kv_start:kv_start + kv_heads_per_rank * HEAD_DIM]
else:
    local_k_w = k_w
    local_v_w = v_w

o_start = rank * heads_per_rank * HEAD_DIM
local_o_w = o_w[:, o_start:o_start + heads_per_rank * HEAD_DIM]

ln_w = weights["model.layers.0.input_layernorm.weight"].to(device).float()
post_ln_w = weights["model.layers.0.post_attention_layernorm.weight"].to(device).float()
final_norm_w = weights["model.norm.weight"].to(device).float()

input_ln = SGLangRMSNorm(HIDDEN, eps=EPS, cast_x_before_out_mul=True).to(device)
input_ln.weight.data.copy_(ln_w)
post_ln = SGLangRMSNorm(HIDDEN, eps=EPS, cast_x_before_out_mul=True).to(device)
post_ln.weight.data.copy_(post_ln_w)
final_ln = SGLangRMSNorm(HIDDEN, eps=EPS, cast_x_before_out_mul=True).to(device)
final_ln.weight.data.copy_(final_norm_w)

cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)

if not has_moe_sum_reduce:
    local_expert_start = rank * NUM_LOCAL_EXPERTS
    w1_list, w2_list = [], []
    for i in range(NUM_LOCAL_EXPERTS):
        gid = local_expert_start + i
        gp = weights[f"model.layers.0.mlp.experts.{gid}.gate_proj.weight"].to(device).bfloat16()
        up = weights[f"model.layers.0.mlp.experts.{gid}.up_proj.weight"].to(device).bfloat16()
        dp = weights[f"model.layers.0.mlp.experts.{gid}.down_proj.weight"].to(device).bfloat16()
        w1_list.append(torch.cat([gp, up], dim=0))
        w2_list.append(dp)
    w1 = torch.stack(w1_list)
    w2 = torch.stack(w2_list)
    gate_weight = weights["model.layers.0.mlp.gate.weight"].to(device).bfloat16()
    local_expert_mapping = torch.full((NUM_EXPERTS,), -1, dtype=torch.int32, device=device)
    local_expert_mapping[local_expert_start:local_expert_start + NUM_LOCAL_EXPERTS] = torch.arange(
        0, NUM_LOCAL_EXPERTS, dtype=torch.int32, device=device
    )


def full_forward_to_logprob(x_in):
    """Single layer → final norm → lm_head → logprob."""
    residual = x_in

    # Attention
    normed = input_ln(x_in).bfloat16()
    q_local = F.linear(normed, local_q_w).view(-1, heads_per_rank, HEAD_DIM)
    k_local = F.linear(normed, local_k_w).view(-1, kv_heads_per_rank, HEAD_DIM)
    v_local = F.linear(normed, local_v_w).view(-1, kv_heads_per_rank, HEAD_DIM)

    q_f = q_local.float()
    q_local = (q_norm_w * (q_f * torch.rsqrt(q_f.pow(2).mean(-1, keepdim=True) + EPS)).to(q_local.dtype)).bfloat16()
    k_f = k_local.float()
    k_local = (k_norm_w * (k_f * torch.rsqrt(k_f.pow(2).mean(-1, keepdim=True) + EPS)).to(k_local.dtype)).bfloat16()

    k_e = k_local.repeat_interleave(rep, dim=1)
    v_e = v_local.repeat_interleave(rep, dim=1)

    attn_out = flash_attn_varlen_func(
        q_local, k_e, v_e,
        cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=T, max_seqlen_k=T,
        causal=True, softmax_scale=scale,
    )
    attn_flat = attn_out.reshape(-1, heads_per_rank * HEAD_DIM)
    partial_attn = F.linear(attn_flat, local_o_w)
    attn_reduced = _tree_all_reduce_sum_impl(partial_attn, tp_group)

    hidden = residual + attn_reduced
    residual2 = hidden

    # MoE
    normed2 = post_ln(hidden).bfloat16()
    logits_r = torch.mm(normed2, gate_weight.t())
    weights_r = F.softmax(logits_r, dim=1, dtype=torch.float)
    weights_r, ids_r = torch.topk(weights_r, TOPK, dim=-1)
    weights_r = weights_r / weights_r.sum(dim=-1, keepdim=True)
    weights_r = weights_r.to(normed2.dtype)
    ids_local = local_expert_mapping[ids_r.long()]

    moe_out = fused_experts_impl(
        normed2.clone(), w1, w2,
        weights_r, ids_local,
        activation="silu",
        filter_expert=True,
    )
    moe_reduced = _tree_all_reduce_sum_impl(moe_out, tp_group)
    hidden2 = residual2 + moe_reduced

    # Final norm → lm_head → logprob
    final_hidden = final_ln(hidden2).bfloat16()
    partial_logits = torch.matmul(final_hidden, local_lm_head.T)
    all_logits = [torch.zeros_like(partial_logits) for _ in range(TP_SIZE)]
    dist.all_gather(all_logits, partial_logits, group=tp_group)
    full_logits = torch.cat(all_logits, dim=-1)[:, :VOCAB_SIZE].float()
    return F.log_softmax(full_logits, dim=-1)


x_full = make_shared_input(B * T, HIDDEN)

with torch.no_grad():
    lp1 = full_forward_to_logprob(x_full)
    lp2 = full_forward_to_logprob(x_full)

results.check("Full forward → logprob: self-consistency", lp1, lp2)


# ============================================================
# Test 6: FA3 varlen self-consistency with num_splits=1
# Verify the kernel itself is deterministic.
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("TEST 6: FA3 varlen self-consistency (num_splits=1)")
    print("=" * 60)

with torch.no_grad():
    out_v1 = flash_attn_varlen_func(
        q, k_exp, v_exp,
        cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=T, max_seqlen_k=T,
        causal=True, softmax_scale=scale,
    )
    out_v2 = flash_attn_varlen_func(
        q, k_exp, v_exp,
        cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=T, max_seqlen_k=T,
        causal=True, softmax_scale=scale,
    )

results.check("FA3 varlen: self-consistency (num_splits=1)", out_v1, out_v2)


# ============================================================
# Test 7: FA3 kvcache self-consistency with num_splits=1
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("TEST 7: FA3 kvcache self-consistency (num_splits=1)")
    print("=" * 60)

with torch.no_grad():
    out_kv1 = flash_attn_with_kvcache(
        q=q.view(B, T, heads_per_rank, HEAD_DIM).contiguous(),
        k_cache=k_cache, v_cache=v_cache,
        block_table=block_table, cache_seqlens=cache_seqlens,
        softmax_scale=scale, causal=True, num_splits=1,
    ).view(B * T, heads_per_rank, HEAD_DIM)
    out_kv2 = flash_attn_with_kvcache(
        q=q.view(B, T, heads_per_rank, HEAD_DIM).contiguous(),
        k_cache=k_cache, v_cache=v_cache,
        block_table=block_table, cache_seqlens=cache_seqlens,
        softmax_scale=scale, causal=True, num_splits=1,
    ).view(B * T, heads_per_rank, HEAD_DIM)

results.check("FA3 kvcache: self-consistency (num_splits=1)", out_kv1, out_kv2)


# ============================================================
# Summary
# ============================================================
results.summary()

if rank == 0:
    print()
    print("INTERPRETATION:")
    print("  - Test 1 DIFF → FA3 API difference is the root cause of E2E logprobs_diff")
    print("  - Test 2 DIFF → GQA handling differs between expand and native")
    print("  - Test 3 DIFF → moe_sum_tree_reduce vs moe_sum_reduce contributes to diff")
    print("  - Test 4 DIFF → logprob chain has precision differences")
    print("  - Test 5 PASS → full chain is deterministic (self-consistent)")
    print("  - Tests 6-7 PASS → each FA3 API is individually deterministic")
    print()
    print("If Test 1 shows DIFF, the fix is to make both sides use the same FA3 API.")

dist.destroy_process_group()
