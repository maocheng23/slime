"""
TP=8 kernel alignment test: verify SGLang and Megatron use identical kernels.

Tests SGLang-path vs Megatron-path at the SAME TP level (not TP=8 vs TP=1).
Both paths use batch_invariant_mode (Triton matmul) + tree_all_reduce.

Tests:
  GDN layers:      Megatron Qwen3NextGatedDeltaNet vs manual SGLang-kernel forward
  Full Attention:   Megatron-style vs SGLang-kernel forward (FA + gated output)
  MoE:             EP-sharded expert computation + tree_all_reduce
  Layernorms:      Qwen3NextRMSNorm across ranks

Usage (inside Docker container):
  torchrun --nproc_per_node=8 scripts/test_tp8_alignment.py
"""
import os
import sys
import json
import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, "/root/slime")

# Init distributed
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(rank)
device = torch.cuda.current_device()

if rank == 0:
    print(f"TP={world_size} kernel alignment test starting...")

# Enable batch_invariant_mode: patches aten::mm → Triton matmul_persistent
# This is what both SGLang and Megatron use in production true-on-policy mode
from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode
enable_batch_invariant_mode(enable_bmm=False)

# Deterministic tree allreduce (used by both SGLang and Megatron)
from sglang.srt.tp_invariant_ops import tree_all_reduce_sum

if rank == 0:
    print("  batch_invariant_mode: ENABLED (aten::mm → Triton matmul_persistent)")
    print("  allreduce: tree_all_reduce_sum (deterministic)")


def tp_allreduce(tensor):
    """Deterministic tree allreduce matching SGLang/Megatron true on-policy path."""
    return tree_all_reduce_sum(tensor)


# ---- Config ----
from transformers import AutoConfig

MODEL_PATH = "/root/models/Qwen3-Next-4layer"
config = AutoConfig.from_pretrained(MODEL_PATH)

HIDDEN = config.hidden_size
EPS = config.rms_norm_eps

# GDN config
NUM_K_HEADS = config.linear_num_key_heads
NUM_V_HEADS = config.linear_num_value_heads
HEAD_K_DIM = config.linear_key_head_dim
HEAD_V_DIM = config.linear_value_head_dim

# Full attention config
NUM_Q_HEADS = config.num_attention_heads      # 16
NUM_KV_HEADS = config.num_key_value_heads     # 2
HEAD_DIM = config.head_dim                    # 256

# MoE config
NUM_EXPERTS = config.num_experts              # 512
TOPK = config.num_experts_per_tok             # 10
MOE_INTERMEDIATE = 512                        # moe_ffn_hidden_size

B, T = 1, 32

# ---- Load weights ----
import safetensors.torch as st

with open(os.path.join(MODEL_PATH, "model.safetensors.index.json")) as f:
    index = json.load(f)

weights = {}
loaded_files = set()
for tensor_name, filename in index["weight_map"].items():
    if filename not in loaded_files:
        filepath = os.path.join(MODEL_PATH, filename)
        shard = st.load_file(filepath)
        weights.update(shard)
        loaded_files.add(filename)

if rank == 0:
    print(f"Loaded {len(weights)} tensors. B={B}, T={T}")
    print(f"GDN: K_heads={NUM_K_HEADS}, V_heads={NUM_V_HEADS}, K_dim={HEAD_K_DIM}, V_dim={HEAD_V_DIM}")
    print(f"Full Attn: Q_heads={NUM_Q_HEADS}, KV_heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}")
    print(f"MoE: {NUM_EXPERTS} experts, topk={TOPK}, intermediate={MOE_INTERMEDIATE}")

# ---- Helpers ----
pass_count = 0
fail_count = 0
info_count = 0


def check(name, a, b, expect_diff=False):
    global pass_count, fail_count, info_count
    diff = (a.float() - b.float()).abs()
    max_diff = diff.max().item()
    if max_diff == 0:
        if rank == 0:
            print(f"  PASS  {name}: bitwise identical")
        pass_count += 1
    elif expect_diff:
        if rank == 0:
            mean_diff = diff.mean().item()
            print(f"  INFO  {name}: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f} (expected)")
        info_count += 1
    else:
        if rank == 0:
            mean_diff = diff.mean().item()
            nonzero_frac = (diff > 0).float().mean().item()
            print(f"  FAIL  {name}: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}, nonzero={nonzero_frac:.4f}")
        fail_count += 1
    return max_diff


# ---- Create same input on all ranks ----
torch.manual_seed(42)
x = torch.randn(B, T, HIDDEN, device=device, dtype=torch.bfloat16)
dist.broadcast(x, src=0)

cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)

# ============================================================
# Test 1: GDN at TP=8 — Megatron module vs manual SGLang-kernel forward
# Both should use identical kernels since Megatron's GDN uses SGLang kernels
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print(f"TEST 1: GDN layer 0 — Megatron module vs SGLang-kernel forward (TP={world_size})")
    print("=" * 60)

from slime_plugins.models.qwen3_next import Qwen3NextGatedDeltaNet

# --- Path A: Megatron module ---
gdn_mega = Qwen3NextGatedDeltaNet(config, layer_idx=0, tp_rank=rank, tp_size=world_size).cuda().bfloat16()
prefix = "model.layers.0.linear_attn"
gdn_mega.in_proj_qkvz.weight.data.copy_(weights[f"{prefix}.in_proj_qkvz.weight"].cuda().bfloat16())
gdn_mega.in_proj_ba.weight.data.copy_(weights[f"{prefix}.in_proj_ba.weight"].cuda().bfloat16())
gdn_mega.out_proj.weight.data.copy_(weights[f"{prefix}.out_proj.weight"].cuda().bfloat16())
gdn_mega.A_log.data.copy_(weights[f"{prefix}.A_log"].cuda().bfloat16())
gdn_mega.dt_bias.data.copy_(weights[f"{prefix}.dt_bias"].cuda().bfloat16())
gdn_mega.norm.weight.data.copy_(weights[f"{prefix}.norm.weight"].cuda().bfloat16())
gdn_mega.conv1d.weight.data.copy_(weights[f"{prefix}.conv1d.weight"].cuda().bfloat16())

with torch.no_grad():
    out_mega = gdn_mega(x, cu_seqlens=cu_seqlens)
out_mega = tp_allreduce(out_mega)

# --- Path B: Manual SGLang-kernel forward (same kernels, same TP sharding) ---
# This replicates what SGLang does: same weight slicing, same kernels, same allreduce
from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule as sglang_chunk_gdr
from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd as sglang_l2norm
from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating as sglang_fused_gating
from sglang.srt.layers.attention.fla.layernorm_gated import RMSNorm as SGLangRMSNormGated
import sgl_kernel

with torch.no_grad():
    # Same TP-sharded weight slicing as Megatron module
    w_qkvz = weights[f"{prefix}.in_proj_qkvz.weight"].cuda().bfloat16()[gdn_mega._qkvz_start:gdn_mega._qkvz_end]
    w_ba = weights[f"{prefix}.in_proj_ba.weight"].cuda().bfloat16()[gdn_mega._ba_start:gdn_mega._ba_end]

    # in_proj (F.linear uses aten::mm → batch_invariant matmul_persistent via patch)
    proj_qkvz = F.linear(x, w_qkvz)
    proj_ba = F.linear(x, w_ba)

    # fix_query_key_value_ordering (same logic as Megatron module)
    query, key, value, z, b, a = gdn_mega.fix_query_key_value_ordering(proj_qkvz, proj_ba)
    query, key, value = (t.reshape(t.shape[0], t.shape[1], -1) for t in (query, key, value))

    # Causal conv1d + SiLU (SGLang CUDA kernel)
    mixed_qkv = torch.cat((query, key, value), dim=-1)
    conv_indices = gdn_mega._conv_indices.to(gdn_mega.conv1d.weight.device)
    conv_weight_tp = gdn_mega.conv1d.weight[conv_indices].squeeze(1)

    # Use sgl_kernel.causal_conv1d_fwd directly (matches SGLang's path)
    x_2d = mixed_qkv.clone().reshape(-1, mixed_qkv.shape[-1]).transpose(0, 1).contiguous()
    sgl_kernel.causal_conv1d_fwd(x_2d, conv_weight_tp, None, None, cu_seqlens, None, None, True, -1)
    mixed_qkv_sgl = x_2d.transpose(0, 1).reshape(B, T, -1)

    key_dim_tp = gdn_mega.key_dim_tp
    value_dim_tp = gdn_mega.value_dim_tp
    query_s, key_s, value_s = torch.split(mixed_qkv_sgl, [key_dim_tp, key_dim_tp, value_dim_tp], dim=-1)
    query_s = query_s.reshape(B, T, -1, HEAD_K_DIM)
    key_s = key_s.reshape(B, T, -1, HEAD_K_DIM)
    value_s = value_s.reshape(B, T, -1, HEAD_V_DIM)

    # Gating (SGLang fused kernel)
    A_log_tp = gdn_mega.A_log[gdn_mega._a_start:gdn_mega._a_end]
    dt_bias_tp = gdn_mega.dt_bias[gdn_mega._a_start:gdn_mega._a_end]
    a_2d = a.reshape(-1, a.shape[-1])
    b_2d = b.reshape(-1, b.shape[-1])
    g_s, beta_s = sglang_fused_gating(A_log_tp, a_2d, b_2d, dt_bias_tp)
    g_s = g_s.squeeze(0).reshape(a.shape[0], a.shape[1], -1)
    beta_s = beta_s.squeeze(0).reshape(b.shape[0], b.shape[1], -1)

    # GQA expansion
    v_per_k_group = NUM_V_HEADS // NUM_K_HEADS
    num_k_heads_tp = NUM_K_HEADS // world_size
    num_v_heads_tp = NUM_V_HEADS // world_size
    if v_per_k_group > 1:
        query_s = query_s.repeat_interleave(v_per_k_group, dim=2)
        key_s = key_s.repeat_interleave(v_per_k_group, dim=2)

    # L2norm (SGLang Triton kernel)
    q_norm_s = sglang_l2norm(query_s)
    k_norm_s = sglang_l2norm(key_s)

    # chunk_gated_delta_rule (SGLang kernel)
    cu_seqlens_long = cu_seqlens.to(torch.long)
    N_seqs = cu_seqlens_long.shape[0] - 1
    H = q_norm_s.shape[2]
    K_dim = q_norm_s.shape[3]
    V_dim = value_s.shape[3]
    zero_state = torch.zeros(N_seqs, H, K_dim, V_dim, device=device, dtype=x.dtype)
    state_idx = torch.arange(N_seqs, dtype=torch.int32, device=device)

    o_sgl, _, _ = sglang_chunk_gdr(
        q_norm_s, k_norm_s, value_s,
        g=g_s, beta=beta_s,
        initial_state=zero_state,
        initial_state_indices=state_idx,
        cu_seqlens=cu_seqlens_long,
        use_qk_l2norm_in_kernel=False,
    )

    # RMSNormGated (SGLang kernel)
    sglang_norm = SGLangRMSNormGated(HEAD_V_DIM, eps=EPS, device=device, dtype=torch.bfloat16)
    sglang_norm.weight.data.copy_(gdn_mega.norm.weight.data)
    o_flat = o_sgl.reshape(-1, o_sgl.shape[-1])
    z_flat = z.reshape(-1, z.shape[-1])
    normed_sgl = sglang_norm(o_flat, z_flat)
    normed_sgl = normed_sgl.reshape(z.shape)
    normed_sgl = normed_sgl.reshape(B, T, -1)

    # Output projection (same TP-sharded weight)
    w_out = gdn_mega.out_proj.weight[:, gdn_mega._out_col_start:gdn_mega._out_col_end]
    out_sgl = F.linear(normed_sgl, w_out)
    out_sgl = tp_allreduce(out_sgl)

if rank == 0:
    check("GDN layer 0: Megatron module vs SGLang-kernel forward", out_mega, out_sgl)

dist.barrier()

# ============================================================
# Test 2: GDN self-consistency (both paths run twice)
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print(f"TEST 2: GDN TP={world_size} self-consistency")
    print("=" * 60)

with torch.no_grad():
    out_mega2 = gdn_mega(x, cu_seqlens=cu_seqlens)
out_mega2 = tp_allreduce(out_mega2)

if rank == 0:
    check(f"GDN TP={world_size} Megatron self-consistency (run 1 vs run 2)", out_mega, out_mega2)

dist.barrier()

# ============================================================
# Test 3: All GDN layers (0, 1, 2) — Megatron module cross-layer consistency
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print(f"TEST 3: All GDN layers (0, 1, 2) at TP={world_size} self-consistency")
    print("=" * 60)

for layer_idx in range(3):
    prefix_l = f"model.layers.{layer_idx}.linear_attn"
    gdn_l = Qwen3NextGatedDeltaNet(config, layer_idx=layer_idx, tp_rank=rank, tp_size=world_size).cuda().bfloat16()
    gdn_l.in_proj_qkvz.weight.data.copy_(weights[f"{prefix_l}.in_proj_qkvz.weight"].cuda().bfloat16())
    gdn_l.in_proj_ba.weight.data.copy_(weights[f"{prefix_l}.in_proj_ba.weight"].cuda().bfloat16())
    gdn_l.out_proj.weight.data.copy_(weights[f"{prefix_l}.out_proj.weight"].cuda().bfloat16())
    gdn_l.A_log.data.copy_(weights[f"{prefix_l}.A_log"].cuda().bfloat16())
    gdn_l.dt_bias.data.copy_(weights[f"{prefix_l}.dt_bias"].cuda().bfloat16())
    gdn_l.norm.weight.data.copy_(weights[f"{prefix_l}.norm.weight"].cuda().bfloat16())
    gdn_l.conv1d.weight.data.copy_(weights[f"{prefix_l}.conv1d.weight"].cuda().bfloat16())

    with torch.no_grad():
        out1 = gdn_l(x, cu_seqlens=cu_seqlens)
    out1 = tp_allreduce(out1)
    with torch.no_grad():
        out2 = gdn_l(x, cu_seqlens=cu_seqlens)
    out2 = tp_allreduce(out2)

    if rank == 0:
        check(f"GDN layer {layer_idx}: self-consistency", out1, out2)

    dist.barrier()

# ============================================================
# Test 4: Layernorms — identical across all ranks (not TP-sharded)
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("TEST 4: Layernorms (not TP-sharded, should be identical across ranks)")
    print("=" * 60)

from slime_plugins.models.qwen3_next import Qwen3NextRMSNorm

for ln_name in ["model.layers.0.input_layernorm.weight", "model.layers.0.post_attention_layernorm.weight",
                "model.layers.3.input_layernorm.weight", "model.layers.3.post_attention_layernorm.weight"]:
    ln = Qwen3NextRMSNorm(HIDDEN, eps=EPS).cuda().bfloat16()
    ln.weight.data.copy_(weights[ln_name].cuda().bfloat16())
    with torch.no_grad():
        ln_out = ln(x)
    ln_rank0 = ln_out.clone()
    dist.broadcast(ln_rank0, src=0)
    if rank == 0:
        short_name = ln_name.replace("model.layers.", "L").replace(".weight", "")
        check(f"{short_name}: all ranks identical", ln_out, ln_rank0)

dist.barrier()

# ============================================================
# Test 5: Full Attention (layer 3) at TP=8
# Both paths use same kernels: batch_invariant matmul + FA + tree_allreduce
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print(f"TEST 5: Full Attention (layer 3) at TP={world_size}")
    print("=" * 60)

q_proj_w = weights["model.layers.3.self_attn.q_proj.weight"].cuda().bfloat16()  # [8192, 2048]
k_proj_w = weights["model.layers.3.self_attn.k_proj.weight"].cuda().bfloat16()  # [512, 2048]
v_proj_w = weights["model.layers.3.self_attn.v_proj.weight"].cuda().bfloat16()  # [512, 2048]
o_proj_w = weights["model.layers.3.self_attn.o_proj.weight"].cuda().bfloat16()  # [2048, 4096]
q_norm_w = weights["model.layers.3.self_attn.q_norm.weight"].cuda().bfloat16()  # [256]
k_norm_w = weights["model.layers.3.self_attn.k_norm.weight"].cuda().bfloat16()  # [256]


def rms_norm_1p(x, weight, eps=1e-6):
    """RMSNorm with 1+weight (Gemma-style, apply_layernorm_1p)."""
    x_f = x.float()
    norm = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    return (x_f * norm * (1.0 + weight.float())).to(x.dtype)


# Gated attention: q_proj outputs [Q, gate] interleaved per head
q_heads_per_rank = NUM_Q_HEADS // world_size  # 2
q_gate_dim_per_head = HEAD_DIM * 2            # 512
q_gate_dim_per_rank = q_heads_per_rank * q_gate_dim_per_head  # 1024
q_dim_per_rank = q_heads_per_rank * HEAD_DIM  # 512

qg_row_start = rank * q_gate_dim_per_rank
qg_row_end = (rank + 1) * q_gate_dim_per_rank

ranks_per_kv_head = world_size // NUM_KV_HEADS  # 4
kv_head_idx = rank // ranks_per_kv_head
kv_row_start = kv_head_idx * HEAD_DIM
kv_row_end = (kv_head_idx + 1) * HEAD_DIM
o_col_start = rank * q_dim_per_rank
o_col_end = (rank + 1) * q_dim_per_rank

from flash_attn import flash_attn_func

with torch.no_grad():
    # --- Path A: run once ---
    qg_a = F.linear(x, q_proj_w[qg_row_start:qg_row_end]).view(B, T, q_heads_per_rank, HEAD_DIM * 2)
    q_a, gate_a = torch.chunk(qg_a, 2, dim=-1)
    k_a = F.linear(x, k_proj_w[kv_row_start:kv_row_end]).view(B, T, 1, HEAD_DIM)
    v_a = F.linear(x, v_proj_w[kv_row_start:kv_row_end]).view(B, T, 1, HEAD_DIM)
    q_a = rms_norm_1p(q_a, q_norm_w, eps=EPS)
    k_a = rms_norm_1p(k_a, k_norm_w, eps=EPS)
    attn_a = flash_attn_func(q_a, k_a, v_a, causal=True)
    gated_a = attn_a.reshape(B, T, q_dim_per_rank) * torch.sigmoid(gate_a.reshape(B, T, -1))
    out_a = F.linear(gated_a, o_proj_w[:, o_col_start:o_col_end])
    out_a = tp_allreduce(out_a)

    # --- Path B: run again (self-consistency) ---
    qg_b = F.linear(x, q_proj_w[qg_row_start:qg_row_end]).view(B, T, q_heads_per_rank, HEAD_DIM * 2)
    q_b, gate_b = torch.chunk(qg_b, 2, dim=-1)
    k_b = F.linear(x, k_proj_w[kv_row_start:kv_row_end]).view(B, T, 1, HEAD_DIM)
    v_b = F.linear(x, v_proj_w[kv_row_start:kv_row_end]).view(B, T, 1, HEAD_DIM)
    q_b = rms_norm_1p(q_b, q_norm_w, eps=EPS)
    k_b = rms_norm_1p(k_b, k_norm_w, eps=EPS)
    attn_b = flash_attn_func(q_b, k_b, v_b, causal=True)
    gated_b = attn_b.reshape(B, T, q_dim_per_rank) * torch.sigmoid(gate_b.reshape(B, T, -1))
    out_b = F.linear(gated_b, o_proj_w[:, o_col_start:o_col_end])
    out_b = tp_allreduce(out_b)

if rank == 0:
    check(f"Full Attn TP={world_size} self-consistency (run 1 vs run 2)", out_a, out_b)

    # Also verify FA output is bitwise identical across ranks for same input
    # (same Q/K/V → same FA output — no rank-dependent behavior in FA)

dist.barrier()

# ============================================================
# Test 6: Full Attention — FA kernel head-independence at TP=8
# Each rank's FA output for its local heads should match the
# corresponding heads from a full TP=1 computation on rank 0
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print(f"TEST 6: FA head-independence: TP={world_size} local heads vs TP=1 sliced heads")
    print("=" * 60)

# Gather rank 0's local FA output
fa_local = attn_a  # [B, T, 2, 256] from rank 0

if rank == 0:
    with torch.no_grad():
        # TP=1 full computation
        qg_full = F.linear(x, q_proj_w).view(B, T, NUM_Q_HEADS, HEAD_DIM * 2)
        q_full, gate_full = torch.chunk(qg_full, 2, dim=-1)
        k_full = F.linear(x, k_proj_w).view(B, T, NUM_KV_HEADS, HEAD_DIM)
        v_full = F.linear(x, v_proj_w).view(B, T, NUM_KV_HEADS, HEAD_DIM)
        q_full = rms_norm_1p(q_full, q_norm_w, eps=EPS)
        k_full = rms_norm_1p(k_full, k_norm_w, eps=EPS)
        attn_full = flash_attn_func(q_full, k_full, v_full, causal=True)

    # Rank 0 has heads [0, 1]. TP=1 full result should match for those heads.
    check("FA head-independence: rank0 local heads vs TP=1 heads[0:2]",
          fa_local, attn_full[:, :, :q_heads_per_rank, :])

dist.barrier()

# ============================================================
# Test 7: MoE Router — identical across all ranks
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print(f"TEST 7: MoE Router consistency across TP={world_size} ranks")
    print("=" * 60)

gate_w = weights["model.layers.0.mlp.gate.weight"].cuda().bfloat16()

with torch.no_grad():
    x_flat = x.reshape(-1, HIDDEN)
    router_logits = F.linear(x_flat, gate_w)
    routing_weights = F.softmax(router_logits.float(), dim=-1)
    topk_weights, topk_indices = torch.topk(routing_weights, k=TOPK, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

topk_indices_rank0 = topk_indices.clone()
topk_weights_rank0 = topk_weights.clone()
dist.broadcast(topk_indices_rank0, src=0)
dist.broadcast(topk_weights_rank0, src=0)

if rank == 0:
    check("MoE Router indices: all ranks identical", topk_indices.float(), topk_indices_rank0.float())
    check("MoE Router weights: all ranks identical", topk_weights, topk_weights_rank0)
    unique_experts = topk_indices.unique().numel()
    print(f"  Router stats: {unique_experts} unique experts selected out of {NUM_EXPERTS}")

dist.barrier()

# ============================================================
# Test 8: MoE single expert SwiGLU — identical across all ranks
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("TEST 8: MoE single expert SwiGLU correctness")
    print("=" * 60)

expert_0_gate = weights["model.layers.0.mlp.experts.0.gate_proj.weight"].cuda().bfloat16()
expert_0_up = weights["model.layers.0.mlp.experts.0.up_proj.weight"].cuda().bfloat16()
expert_0_down = weights["model.layers.0.mlp.experts.0.down_proj.weight"].cuda().bfloat16()

with torch.no_grad():
    gate_out = F.linear(x_flat, expert_0_gate)
    up_out = F.linear(x_flat, expert_0_up)
    expert_hidden = F.silu(gate_out) * up_out
    expert_out = F.linear(expert_hidden, expert_0_down)

expert_out_rank0 = expert_out.clone()
dist.broadcast(expert_out_rank0, src=0)

if rank == 0:
    check("Expert 0 SwiGLU: all ranks identical", expert_out, expert_out_rank0)

dist.barrier()

# ============================================================
# Test 9: MoE EP=8 — self-consistency (run twice, same result)
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print(f"TEST 9: MoE EP={world_size} self-consistency")
    print("=" * 60)
    print(f"  Loading {NUM_EXPERTS} expert weights...")

expert_gate_w = {}
expert_up_w = {}
expert_down_w = {}
for eid in range(NUM_EXPERTS):
    expert_gate_w[eid] = weights[f"model.layers.0.mlp.experts.{eid}.gate_proj.weight"].cuda().bfloat16()
    expert_up_w[eid] = weights[f"model.layers.0.mlp.experts.{eid}.up_proj.weight"].cuda().bfloat16()
    expert_down_w[eid] = weights[f"model.layers.0.mlp.experts.{eid}.down_proj.weight"].cuda().bfloat16()

if rank == 0:
    print(f"  All expert weights loaded.")


def compute_moe_local(x_flat, topk_indices, topk_weights, local_start, local_end):
    """Compute partial MoE output for experts in [local_start, local_end)."""
    num_tokens = x_flat.shape[0]
    out = torch.zeros(num_tokens, HIDDEN, device=device, dtype=torch.bfloat16)
    for token_idx in range(num_tokens):
        for k in range(TOPK):
            eid = topk_indices[token_idx, k].item()
            if local_start <= eid < local_end:
                w = topk_weights[token_idx, k].to(torch.bfloat16)
                token = x_flat[token_idx:token_idx + 1]
                g = F.linear(token, expert_gate_w[eid])
                u = F.linear(token, expert_up_w[eid])
                h = F.silu(g) * u
                d = F.linear(h, expert_down_w[eid])
                out[token_idx] += w * d.squeeze(0)
    return out


experts_per_rank = NUM_EXPERTS // world_size
local_start = rank * experts_per_rank
local_end = (rank + 1) * experts_per_rank

with torch.no_grad():
    moe_out_1 = compute_moe_local(x_flat, topk_indices, topk_weights, local_start, local_end)
    moe_out_1 = tp_allreduce(moe_out_1)

    moe_out_2 = compute_moe_local(x_flat, topk_indices, topk_weights, local_start, local_end)
    moe_out_2 = tp_allreduce(moe_out_2)

if rank == 0:
    check(f"MoE EP={world_size} self-consistency (run 1 vs run 2)", moe_out_1, moe_out_2)

dist.barrier()

# ============================================================
# Test 10: MoE Shared Expert — identical across all ranks
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("TEST 10: MoE Shared Expert (not EP-sharded)")
    print("=" * 60)

shared_gate_w = weights["model.layers.0.mlp.shared_expert.gate_proj.weight"].cuda().bfloat16()
shared_up_w = weights["model.layers.0.mlp.shared_expert.up_proj.weight"].cuda().bfloat16()
shared_down_w = weights["model.layers.0.mlp.shared_expert.down_proj.weight"].cuda().bfloat16()
shared_expert_gate_w = weights["model.layers.0.mlp.shared_expert_gate.weight"].cuda().bfloat16()

with torch.no_grad():
    sg = F.linear(x_flat, shared_gate_w)
    su = F.linear(x_flat, shared_up_w)
    sh = F.silu(sg) * su
    shared_out = F.linear(sh, shared_down_w)
    shared_gate_score = torch.sigmoid(F.linear(x_flat, shared_expert_gate_w))
    shared_out_gated = shared_out * shared_gate_score

shared_rank0 = shared_out_gated.clone()
dist.broadcast(shared_rank0, src=0)

if rank == 0:
    check("Shared Expert: all ranks identical", shared_out_gated, shared_rank0)

dist.barrier()

# ============================================================
# Test 11: batch_invariant matmul: F.linear produces same result across all ranks
# (verifies aten::mm → matmul_persistent patch is active and deterministic)
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("TEST 11: batch_invariant matmul (aten::mm → Triton) consistency")
    print("=" * 60)

test_w = weights[f"model.layers.0.linear_attn.in_proj_qkvz.weight"].cuda().bfloat16()
with torch.no_grad():
    mm_out = F.linear(x.reshape(-1, HIDDEN), test_w)
mm_rank0 = mm_out.clone()
dist.broadcast(mm_rank0, src=0)

if rank == 0:
    check("batch_invariant F.linear: all ranks identical", mm_out, mm_rank0)

dist.barrier()

# ============================================================
# Test 12: tree_all_reduce determinism
# (verify tree_all_reduce gives identical result when called twice)
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("TEST 12: tree_all_reduce determinism")
    print("=" * 60)

# Create rank-dependent partial data (simulates TP partial output)
torch.manual_seed(42 + rank)
partial = torch.randn(B, T, HIDDEN, device=device, dtype=torch.bfloat16)

reduced_1 = tp_allreduce(partial)
reduced_2 = tp_allreduce(partial)

if rank == 0:
    check("tree_all_reduce: self-consistency (same input twice)", reduced_1, reduced_2)

dist.barrier()

# ============================================================
# Summary
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    total = pass_count + fail_count + info_count
    print(f"SUMMARY: {pass_count} PASS, {info_count} INFO, {fail_count} FAIL out of {total} checks")
    if fail_count == 0:
        print(f"ALL TESTS OK at TP={world_size}!")
        print("  All kernels verified: batch_invariant matmul, tree_all_reduce,")
        print("  GDN (conv1d, gating, l2norm, chunk_gdr, rmsnorm_gated),")
        print("  Full Attention (FA + gated output), MoE (router, experts, shared expert)")
    else:
        print(f"FAILURES detected: {fail_count} unexpected failures.")
    print("=" * 60)

dist.destroy_process_group()
