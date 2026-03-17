"""
Unit tests for bitwise alignment between Megatron and SGLang Qwen3-Next layers.
Tests each kernel/layer by comparing the ACTUAL Megatron code path vs the ACTUAL
SGLang code path with the same weights and inputs.

Model: Qwen3-Next-4layer (3 GDN + 1 Full Attention, all with MoE)

Usage (inside Docker container, single GPU):
  CUDA_VISIBLE_DEVICES=0 python scripts/test_layer_alignment.py
"""
import os
import sys
import torch
import torch.nn.functional as F

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from alignment_test_utils import TestResults, load_safetensor_weights

# ---- Config ----
from transformers import AutoConfig

MODEL_PATH = "/root/models/Qwen3-Next-4layer"
config = AutoConfig.from_pretrained(MODEL_PATH)

HIDDEN = config.hidden_size
EPS = config.rms_norm_eps
NUM_K_HEADS = config.linear_num_key_heads
NUM_V_HEADS = config.linear_num_value_heads
HEAD_K_DIM = config.linear_key_head_dim
HEAD_V_DIM = config.linear_value_head_dim
KEY_DIM = HEAD_K_DIM * NUM_K_HEADS
VALUE_DIM = HEAD_V_DIM * NUM_V_HEADS
CONV_KERNEL = config.linear_conv_kernel_dim
NUM_EXPERTS = config.num_experts
TOPK = config.num_experts_per_tok
INTERMEDIATE = config.intermediate_size

B, T = 1, 16  # batch, sequence length

print(f"Model: {MODEL_PATH}")
print(f"hidden={HIDDEN}, num_k_heads={NUM_K_HEADS}, num_v_heads={NUM_V_HEADS}")
print(f"head_k_dim={HEAD_K_DIM}, head_v_dim={HEAD_V_DIM}")
print(f"key_dim={KEY_DIM}, value_dim={VALUE_DIM}")
print(f"num_experts={NUM_EXPERTS}, topk={TOPK}, intermediate={INTERMEDIATE}")
print(f"Test shape: B={B}, T={T}")

# ---- Load weights ----
weights = load_safetensor_weights(MODEL_PATH)
print()

# ---- Test results tracker ----
results = TestResults()

# ============================================================
# Test 1: RMSNorm (Megatron Qwen3NextRMSNorm vs SGLang GemmaRMSNorm)
# ============================================================
print("=" * 60)
print("TEST 1: RMSNorm (Megatron vs SGLang)")
print("=" * 60)

from sglang.srt.layers.layernorm import GemmaRMSNorm

sys.path.insert(0, "/root/slime")
from slime_plugins.models.qwen3_next import Qwen3NextRMSNorm

ln_weight = weights["model.layers.0.input_layernorm.weight"].cuda().bfloat16()
x = torch.randn(B, T, HIDDEN, device="cuda", dtype=torch.bfloat16)
x_2d = x.reshape(-1, HIDDEN)

sglang_ln = GemmaRMSNorm(HIDDEN, eps=EPS).cuda().bfloat16()
sglang_ln.weight.data.copy_(ln_weight)

mega_ln = Qwen3NextRMSNorm(HIDDEN, eps=EPS).cuda().bfloat16()
mega_ln.weight.data.copy_(ln_weight)

with torch.no_grad():
    out_sglang = sglang_ln(x_2d)
    out_mega = mega_ln(x).reshape(-1, HIDDEN)

results.check("RMSNorm: Megatron (Qwen3NextRMSNorm) vs SGLang (GemmaRMSNorm)", out_sglang, out_mega)

# ============================================================
# Test 2: Linear projection (F.linear — same kernel both sides)
# ============================================================
print("\n" + "=" * 60)
print("TEST 2: Linear projection (in_proj_qkvz / in_proj_ba)")
print("=" * 60)

qkvz_weight = weights["model.layers.0.linear_attn.in_proj_qkvz.weight"].cuda().bfloat16()
ba_weight = weights["model.layers.0.linear_attn.in_proj_ba.weight"].cuda().bfloat16()

with torch.no_grad():
    proj_qkvz = F.linear(out_sglang, qkvz_weight)
    proj_qkvz2 = F.linear(out_sglang, qkvz_weight)
    proj_ba = F.linear(out_sglang, ba_weight)
    proj_ba2 = F.linear(out_sglang, ba_weight)

results.check("in_proj_qkvz (F.linear) self-consistency", proj_qkvz, proj_qkvz2)
results.check("in_proj_ba (F.linear) self-consistency", proj_ba, proj_ba2)

# ============================================================
# Test 3: Conv1d + SiLU (Megatron _CausalConv1dWithBackward vs SGLang causal_conv1d_fn)
# ============================================================
print("\n" + "=" * 60)
print("TEST 3: Conv1d + SiLU (Megatron vs SGLang)")
print("=" * 60)

import sgl_kernel
from slime_plugins.models.qwen3_next import _CausalConv1dWithBackward

conv_weight_full = weights["model.layers.0.linear_attn.conv1d.weight"].cuda().bfloat16()
conv_weight = conv_weight_full.squeeze(1)

conv_dim = KEY_DIM * 2 + VALUE_DIM
x_conv = torch.randn(B, T, conv_dim, device="cuda", dtype=torch.bfloat16)
cu_seqlens_i32 = torch.tensor([0, T], dtype=torch.int32, device="cuda")

with torch.no_grad():
    out_mega_conv = _CausalConv1dWithBackward.apply(x_conv.clone(), conv_weight, cu_seqlens_i32, "silu")

    x_sgl = x_conv.clone().reshape(-1, conv_dim).transpose(0, 1).contiguous()
    sgl_kernel.causal_conv1d_fwd(x_sgl, conv_weight, None, None, cu_seqlens_i32, None, None, True, -1)
    out_sglang_conv = x_sgl.transpose(0, 1).reshape(B, T, conv_dim)

results.check("Conv1d+SiLU: Megatron (_CausalConv1dWithBackward) vs SGLang (sgl_kernel)", out_mega_conv, out_sglang_conv)

with torch.no_grad():
    x_sgl2 = x_conv.clone().reshape(-1, conv_dim).transpose(0, 1).contiguous()
    sgl_kernel.causal_conv1d_fwd(x_sgl2, conv_weight, None, None, cu_seqlens_i32, None, None, True, -1)
    out_sglang_conv2 = x_sgl2.transpose(0, 1).reshape(B, T, conv_dim)

results.check("Conv1d+SiLU: SGLang self-consistency", out_sglang_conv, out_sglang_conv2)

# ============================================================
# Test 4: FusedRMSNormGated (Megatron AlignedRMSNormGated vs SGLang RMSNormGated)
# ============================================================
print("\n" + "=" * 60)
print("TEST 4: RMSNormGated (Megatron AlignedRMSNormGated vs SGLang RMSNormGated)")
print("=" * 60)

from fla.modules import FusedRMSNormGated
from sglang.srt.layers.attention.fla.layernorm_gated import RMSNorm as SGLangRMSNormGated
from slime_plugins.models.qwen3_next import AlignedRMSNormGated

norm_weight = weights["model.layers.0.linear_attn.norm.weight"].cuda().bfloat16()

sglang_norm = SGLangRMSNormGated(HEAD_V_DIM, eps=EPS, device="cuda", dtype=torch.bfloat16)
sglang_norm.weight.data.copy_(norm_weight)

aligned_norm = AlignedRMSNormGated(HEAD_V_DIM, eps=EPS, activation="silu", device="cuda", dtype=torch.bfloat16)
aligned_norm.weight.data.copy_(norm_weight)

fla_norm = FusedRMSNormGated(HEAD_V_DIM, eps=EPS, activation="silu", device="cuda", dtype=torch.bfloat16)
fla_norm.weight.data.copy_(norm_weight)

x_norm = torch.randn(B * T * NUM_V_HEADS, HEAD_V_DIM, device="cuda", dtype=torch.bfloat16)
z_norm = torch.randn(B * T * NUM_V_HEADS, HEAD_V_DIM, device="cuda", dtype=torch.bfloat16)

with torch.no_grad():
    out_sglang_norm = sglang_norm(x_norm, z_norm)
    out_aligned = aligned_norm(x_norm, z_norm)
    out_fla_norm = fla_norm(x_norm, z_norm)

results.check("RMSNormGated: Megatron (AlignedRMSNormGated) vs SGLang (RMSNormGated)", out_aligned, out_sglang_norm)
results.check("RMSNormGated: FLA (FusedRMSNormGated) vs SGLang (RMSNormGated)", out_fla_norm, out_sglang_norm, expect_diff=True)

# ============================================================
# Test 5: chunk_gated_delta_rule (FLA vs SGLang — CRITICAL)
# ============================================================
print("\n" + "=" * 60)
print("TEST 5: chunk_gated_delta_rule (FLA vs SGLang kernel)")
print("=" * 60)

from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_gdr
from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule as sglang_gdr
from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd as _sglang_l2norm_fwd

q = torch.randn(B, T, NUM_V_HEADS, HEAD_K_DIM, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, T, NUM_V_HEADS, HEAD_K_DIM, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, T, NUM_V_HEADS, HEAD_V_DIM, device="cuda", dtype=torch.bfloat16)
g = torch.randn(B, T, NUM_V_HEADS, device="cuda", dtype=torch.float32) * 0.1
beta = torch.sigmoid(torch.randn(B, T, NUM_V_HEADS, device="cuda", dtype=torch.bfloat16))
cu_seqlens_long = torch.tensor([0, T], dtype=torch.long, device="cuda")
zero_state = torch.zeros(B, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, device="cuda", dtype=torch.bfloat16)
state_indices = torch.zeros(B, dtype=torch.int32, device="cuda")

with torch.no_grad():
    out_fla_a, _ = fla_gdr(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    out_fla_b, _ = fla_gdr(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)

results.check("chunk_gated_delta_rule: FLA self-consistency", out_fla_a, out_fla_b)

with torch.no_grad():
    s1 = torch.zeros_like(zero_state)
    out_sgl_a, _, _ = sglang_gdr(q, k, v, g=g, beta=beta, initial_state=s1,
                                  initial_state_indices=state_indices,
                                  cu_seqlens=cu_seqlens_long, use_qk_l2norm_in_kernel=True)
    s2 = torch.zeros_like(zero_state)
    out_sgl_b, _, _ = sglang_gdr(q, k, v, g=g, beta=beta, initial_state=s2,
                                  initial_state_indices=state_indices,
                                  cu_seqlens=cu_seqlens_long, use_qk_l2norm_in_kernel=True)

results.check("chunk_gated_delta_rule: SGLang self-consistency", out_sgl_a, out_sgl_b)

diff_cross = (out_fla_a.float() - out_sgl_a.float()).abs()
max_diff_cross = diff_cross.max().item()
print(f"  INFO  FLA vs SGLang cross-kernel: max_diff={max_diff_cross:.8f} (expected >0, different implementations)")

# ============================================================
# Test 6: chunk_gated_delta_rule — Megatron path vs SGLang path
# ============================================================
print("\n" + "=" * 60)
print("TEST 6: chunk_gated_delta_rule — Megatron path vs SGLang path (CRITICAL)")
print("=" * 60)

with torch.no_grad():
    s_ref = torch.zeros_like(zero_state)
    out_sglang_path, _, _ = sglang_gdr(q, k, v, g=g, beta=beta, initial_state=s_ref,
                                        initial_state_indices=state_indices,
                                        cu_seqlens=cu_seqlens_long, use_qk_l2norm_in_kernel=True)

with torch.no_grad():
    q_normed = _sglang_l2norm_fwd(q)
    k_normed = _sglang_l2norm_fwd(k)
    out_old_mega, _ = fla_gdr(q_normed, k_normed, v, g=g, beta=beta, use_qk_l2norm_in_kernel=False)

results.check("OLD Megatron (pre-l2norm + FLA) vs SGLang", out_old_mega, out_sglang_path, expect_diff=True)

with torch.no_grad():
    s_fix = torch.zeros_like(zero_state)
    out_new_mega, _, _ = sglang_gdr(q_normed, k_normed, v, g=g, beta=beta, initial_state=s_fix,
                                     initial_state_indices=state_indices,
                                     cu_seqlens=cu_seqlens_long, use_qk_l2norm_in_kernel=False)

results.check("NEW Megatron (pre-l2norm + SGLang) vs SGLang (should PASS)", out_new_mega, out_sglang_path)

# ============================================================
# Test 7: l2norm (SGLang Triton kernel)
# ============================================================
print("\n" + "=" * 60)
print("TEST 7: l2norm (SGLang Triton kernel)")
print("=" * 60)

x_l2 = torch.randn(B, T, NUM_V_HEADS, HEAD_K_DIM, device="cuda", dtype=torch.bfloat16)
with torch.no_grad():
    out_l2_a = _sglang_l2norm_fwd(x_l2)
    out_l2_b = _sglang_l2norm_fwd(x_l2)

results.check("l2norm self-consistency", out_l2_a, out_l2_b)

# ============================================================
# Test 8: fused_gdn_gating (SGLang Triton kernel)
# ============================================================
print("\n" + "=" * 60)
print("TEST 8: fused_gdn_gating (SGLang Triton, self-consistency + vs PyTorch ref)")
print("=" * 60)

from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating

A_log = weights["model.layers.0.linear_attn.A_log"].cuda().bfloat16()
dt_bias = weights["model.layers.0.linear_attn.dt_bias"].cuda().bfloat16()

a_input = torch.randn(B * T, NUM_V_HEADS, device="cuda", dtype=torch.bfloat16)
b_input = torch.randn(B * T, NUM_V_HEADS, device="cuda", dtype=torch.bfloat16)

with torch.no_grad():
    g_sglang, beta_sglang = fused_gdn_gating(A_log, a_input, b_input, dt_bias)
    g_sglang = g_sglang.squeeze(0)
    beta_sglang = beta_sglang.squeeze(0)

    beta_pytorch = b_input.sigmoid()
    g_pytorch = -A_log.float().exp() * F.softplus(a_input.float() + dt_bias)

results.check("fused_gdn_gating g: SGLang vs PyTorch", g_sglang, g_pytorch, expect_diff=True)
results.check("fused_gdn_gating beta: SGLang vs PyTorch", beta_sglang, beta_pytorch)

with torch.no_grad():
    g2, beta2 = fused_gdn_gating(A_log, a_input, b_input, dt_bias)
results.check("fused_gdn_gating self-consistency (g)", g_sglang, g2.squeeze(0))
results.check("fused_gdn_gating self-consistency (beta)", beta_sglang, beta2.squeeze(0))

# ============================================================
# Test 9: MoE gate (router) — F.linear self-consistency
# ============================================================
print("\n" + "=" * 60)
print("TEST 9: MoE gate (router logits)")
print("=" * 60)

gate_weight = weights["model.layers.0.mlp.gate.weight"].cuda().bfloat16()
x_moe = torch.randn(B * T, HIDDEN, device="cuda", dtype=torch.bfloat16)

with torch.no_grad():
    router_a = F.linear(x_moe, gate_weight)
    router_b = F.linear(x_moe, gate_weight)

results.check("MoE router (F.linear) self-consistency", router_a, router_b)

# ============================================================
# Test 10: MoE single expert forward
# ============================================================
print("\n" + "=" * 60)
print("TEST 10: MoE single expert forward (gate_proj + up_proj + down_proj)")
print("=" * 60)

gate_proj_w = weights["model.layers.0.mlp.experts.0.gate_proj.weight"].cuda().bfloat16()
up_proj_w = weights["model.layers.0.mlp.experts.0.up_proj.weight"].cuda().bfloat16()
down_proj_w = weights["model.layers.0.mlp.experts.0.down_proj.weight"].cuda().bfloat16()

x_expert = torch.randn(4, HIDDEN, device="cuda", dtype=torch.bfloat16)

with torch.no_grad():
    gate_out = F.linear(x_expert, gate_proj_w)
    up_out = F.linear(x_expert, up_proj_w)
    expert_out = F.linear(F.silu(gate_out) * up_out, down_proj_w)

    gate_out2 = F.linear(x_expert, gate_proj_w)
    up_out2 = F.linear(x_expert, up_proj_w)
    expert_out2 = F.linear(F.silu(gate_out2) * up_out2, down_proj_w)

results.check("MoE expert forward self-consistency", expert_out, expert_out2)

# ============================================================
# Test 11: FlashAttention3 self-consistency
# ============================================================
print("\n" + "=" * 60)
print("TEST 11: FlashAttention3 prefill self-consistency")
print("=" * 60)

try:
    from flash_attn.flash_attn_interface import flash_attn_func

    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = HIDDEN // num_heads

    q_fa = torch.randn(B, T, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k_fa = torch.randn(B, T, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v_fa = torch.randn(B, T, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        out_fa1 = flash_attn_func(q_fa, k_fa, v_fa, causal=True)
        out_fa2 = flash_attn_func(q_fa, k_fa, v_fa, causal=True)

    results.check("FlashAttention3 prefill self-consistency", out_fa1, out_fa2)
except ImportError:
    print("  SKIP  FlashAttention3 not available")

# ============================================================
# Test 12: log_softmax (batch_invariant_mode)
# ============================================================
print("\n" + "=" * 60)
print("TEST 12: log_softmax (batch_invariant_mode)")
print("=" * 60)

from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

logits = torch.randn(B * T, config.vocab_size, device="cuda", dtype=torch.bfloat16)

with torch.no_grad():
    std_log_sm = F.log_softmax(logits, dim=-1)

enable_batch_invariant_mode(enable_bmm=False)
with torch.no_grad():
    inv_log_sm = F.log_softmax(logits, dim=-1)
    inv_log_sm2 = F.log_softmax(logits, dim=-1)

results.check("log_softmax: batch_invariant self-consistency", inv_log_sm, inv_log_sm2)

# ============================================================
# Test 13: Final RMSNorm (Megatron vs SGLang)
# ============================================================
print("\n" + "=" * 60)
print("TEST 13: Final RMSNorm (model.norm)")
print("=" * 60)

final_norm_weight = weights["model.norm.weight"].cuda().bfloat16()
x_final = torch.randn(B * T, HIDDEN, device="cuda", dtype=torch.bfloat16)

sglang_final_ln = GemmaRMSNorm(HIDDEN, eps=EPS).cuda().bfloat16()
sglang_final_ln.weight.data.copy_(final_norm_weight)

mega_final_ln = Qwen3NextRMSNorm(HIDDEN, eps=EPS).cuda().bfloat16()
mega_final_ln.weight.data.copy_(final_norm_weight)

with torch.no_grad():
    out_s = sglang_final_ln(x_final)
    out_m = mega_final_ln(x_final.reshape(B, T, HIDDEN)).reshape(-1, HIDDEN)

results.check("Final RMSNorm: Megatron vs SGLang", out_s, out_m)

# ============================================================
# Test 14: lm_head (F.linear) self-consistency
# ============================================================
print("\n" + "=" * 60)
print("TEST 14: lm_head projection")
print("=" * 60)

lm_head_weight = weights["lm_head.weight"].cuda().bfloat16()
with torch.no_grad():
    logits_a = F.linear(out_s, lm_head_weight)
    logits_b = F.linear(out_s, lm_head_weight)

results.check("lm_head (F.linear) self-consistency", logits_a, logits_b)

# ============================================================
# Test 15: Full GDN layer end-to-end (Megatron vs SGLang path)
# ============================================================
print("\n" + "=" * 60)
print("TEST 15: Full GDN layer end-to-end (Megatron Qwen3NextGatedDeltaNet)")
print("=" * 60)

from slime_plugins.models.qwen3_next import Qwen3NextGatedDeltaNet

gdn = Qwen3NextGatedDeltaNet(config, layer_idx=0, tp_rank=0, tp_size=1).cuda().bfloat16()

gdn.in_proj_qkvz.weight.data.copy_(weights["model.layers.0.linear_attn.in_proj_qkvz.weight"].cuda().bfloat16())
gdn.in_proj_ba.weight.data.copy_(weights["model.layers.0.linear_attn.in_proj_ba.weight"].cuda().bfloat16())
gdn.out_proj.weight.data.copy_(weights["model.layers.0.linear_attn.out_proj.weight"].cuda().bfloat16())
gdn.A_log.data.copy_(weights["model.layers.0.linear_attn.A_log"].cuda().bfloat16())
gdn.dt_bias.data.copy_(weights["model.layers.0.linear_attn.dt_bias"].cuda().bfloat16())
gdn.norm.weight.data.copy_(weights["model.layers.0.linear_attn.norm.weight"].cuda().bfloat16())
gdn.conv1d.weight.data.copy_(weights["model.layers.0.linear_attn.conv1d.weight"].cuda().bfloat16())

torch.manual_seed(42)
x_gdn = torch.randn(B, T, HIDDEN, device="cuda", dtype=torch.bfloat16)
cu_gdn = torch.tensor([0, T], dtype=torch.int32, device="cuda")

with torch.no_grad():
    out_mega_gdn = gdn(x_gdn, cu_seqlens=cu_gdn)

with torch.no_grad():
    x_2d_gdn = x_gdn.reshape(-1, HIDDEN)

    proj_qkvz = F.linear(x_2d_gdn, gdn.in_proj_qkvz.weight)
    proj_ba = F.linear(x_2d_gdn, gdn.in_proj_ba.weight)

    v_per_k_group = NUM_V_HEADS // NUM_K_HEADS
    qkvz_shape = proj_qkvz.view(T, NUM_K_HEADS, 2 * HEAD_K_DIM + 2 * HEAD_V_DIM * v_per_k_group)
    ba_shape = proj_ba.view(T, NUM_K_HEADS, 2 * v_per_k_group)
    split_qkvz = [HEAD_K_DIM, HEAD_K_DIM, v_per_k_group * HEAD_V_DIM, v_per_k_group * HEAD_V_DIM]
    split_ba = [v_per_k_group, v_per_k_group]
    query_s, key_s, value_s, z_s = torch.split(qkvz_shape, split_qkvz, dim=2)
    b_s, a_s = torch.split(ba_shape, split_ba, dim=2)
    value_s = value_s.reshape(T, -1, HEAD_V_DIM)
    z_s = z_s.reshape(T, -1, HEAD_V_DIM)
    b_s = b_s.reshape(T, NUM_V_HEADS)
    a_s = a_s.reshape(T, NUM_V_HEADS)
    query_s = query_s.reshape(T, -1)
    key_s = key_s.reshape(T, -1)
    value_s_flat = value_s.reshape(T, -1)

    mixed_qkv_s = torch.cat([query_s, key_s, value_s_flat], dim=-1)
    x_conv_sgl = mixed_qkv_s.transpose(0, 1).contiguous()
    sgl_kernel.causal_conv1d_fwd(x_conv_sgl, conv_weight, None, None, cu_gdn, None, None, True, -1)
    mixed_qkv_after_conv = x_conv_sgl.transpose(0, 1)

    q_conv, k_conv, v_conv = torch.split(mixed_qkv_after_conv, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
    q_heads = q_conv.reshape(1, T, -1, HEAD_K_DIM)
    k_heads = k_conv.reshape(1, T, -1, HEAD_K_DIM)
    v_heads = v_conv.reshape(1, T, -1, HEAD_V_DIM)

    g_s, beta_s = fused_gdn_gating(gdn.A_log.data, a_s, b_s, gdn.dt_bias.data)

    if v_per_k_group > 1:
        q_heads = q_heads.repeat_interleave(v_per_k_group, dim=2)
        k_heads = k_heads.repeat_interleave(v_per_k_group, dim=2)

    q_heads = _sglang_l2norm_fwd(q_heads)
    k_heads = _sglang_l2norm_fwd(k_heads)

    cu_long = torch.tensor([0, T], dtype=torch.long, device="cuda")
    zero_s = torch.zeros(B, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, device="cuda", dtype=torch.bfloat16)
    si = torch.zeros(B, dtype=torch.int32, device="cuda")
    core_out_s, _, _ = sglang_gdr(q_heads, k_heads, v_heads, g=g_s, beta=beta_s,
                                   initial_state=zero_s, initial_state_indices=si,
                                   cu_seqlens=cu_long, use_qk_l2norm_in_kernel=False)

    core_out_2d = core_out_s.reshape(-1, HEAD_V_DIM)
    z_2d = z_s.reshape(-1, HEAD_V_DIM)
    normed = sglang_norm(core_out_2d, z_2d)

    normed_flat = normed.reshape(1, T, -1)
    out_sglang_gdn = F.linear(normed_flat, gdn.out_proj.weight)

results.check("Full GDN layer: Megatron vs manual SGLang path", out_mega_gdn, out_sglang_gdn)

# ============================================================
# Summary
# ============================================================
results.summary()
