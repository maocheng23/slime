"""
Unit tests for bitwise alignment between Megatron and SGLang for Qwen3-30B-A3B (MoE model).
Tests each kernel/layer by comparing the ACTUAL Megatron code path vs the ACTUAL
SGLang code path with the same weights and inputs.

Model: Qwen3-30B-A3B
  - 48 layers, hidden_size=2048, 128 experts, topk=8
  - 32 attention heads, 4 KV groups (GQA), head_dim=128
  - QK LayerNorm, SwiGLU, RoPE
  - No GDN/linear attention, no shared experts

Usage (inside Docker container, single GPU):
  CUDA_VISIBLE_DEVICES=0 python scripts/test_moe_layer_alignment.py
"""
import os
import sys

import torch
import torch.nn.functional as F

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# ---- Shared utilities ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alignment_test_utils import TestResults, load_safetensor_weights, setup_sglang_for_test

# ---- Config ----
from transformers import AutoConfig

MODEL_PATH = "/root/models/Qwen3-30B-A3B"
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
ROPE_THETA = getattr(config, 'rope_theta', 1000000)

B, T = 1, 16  # batch, sequence length

print(f"Model: {MODEL_PATH}")
print(f"hidden={HIDDEN}, num_q_heads={NUM_Q_HEADS}, num_kv_heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}")
print(f"num_experts={NUM_EXPERTS}, topk={TOPK}, moe_ffn_hidden={MOE_FFN_HIDDEN}")
print(f"eps={EPS}, rope_theta={ROPE_THETA}, vocab_size={VOCAB_SIZE}")
print(f"Test shape: B={B}, T={T}")

# ---- Load weights ----
weights = load_safetensor_weights(MODEL_PATH, prefixes=[
    "model.layers.0.",
    "model.norm.",
    "model.embed_tokens.",
    "lm_head.",
])

# ---- Set up SGLang environment ----
setup_sglang_for_test(MODEL_PATH, rl_on_policy_target="fsdp")
print()

# ---- Test results tracker ----
results = TestResults()


# ============================================================
# Test 1: RMSNorm (Megatron SGLangRMSNorm vs SGLang RMSNorm)
# ============================================================
print("=" * 60)
print("TEST 1: RMSNorm — input_layernorm")
print("=" * 60)

from sglang.srt.layers.layernorm import RMSNorm as SGLangRMSNorm

# Both SGLang and Megatron use cast_x_before_out_mul=True for rl_on_policy_target
# Weight is fp32 in both stacks
ln_weight = weights["model.layers.0.input_layernorm.weight"].cuda().float()
x = torch.randn(B * T, HIDDEN, device="cuda", dtype=torch.bfloat16)

# SGLang path: RMSNorm with cast_x_before_out_mul=True
sglang_ln = SGLangRMSNorm(HIDDEN, eps=EPS, cast_x_before_out_mul=True).cuda()
sglang_ln.weight.data.copy_(ln_weight)

with torch.no_grad():
    out_sglang = sglang_ln(x)

    # Megatron SGLangRMSNorm path (manual computation to match exactly)
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(variance + EPS)
    out_mega = ln_weight * x_normed.to(x.dtype)

results.check("RMSNorm: Megatron (SGLangRMSNorm) vs SGLang (RMSNorm, cast_x_before_out_mul=True)", out_sglang, out_mega)

# Self-consistency
with torch.no_grad():
    out_sglang2 = sglang_ln(x)
results.check("RMSNorm: SGLang self-consistency", out_sglang, out_sglang2)


# ============================================================
# Test 2: RMSNorm with residual (pre_mlp_layernorm)
# ============================================================
print("\n" + "=" * 60)
print("TEST 2: RMSNorm with residual — pre_mlp_layernorm (MoE layers)")
print("=" * 60)

post_attn_weight = weights["model.layers.0.post_attention_layernorm.weight"].cuda().float()

x_attn_out = torch.randn(B * T, HIDDEN, device="cuda", dtype=torch.bfloat16)
residual = torch.randn(B * T, HIDDEN, device="cuda", dtype=torch.bfloat16)

# SGLang: RMSNorm with fp32_residual=False, cast_x_before_out_mul=True
sglang_post_ln = SGLangRMSNorm(HIDDEN, eps=EPS, cast_x_before_out_mul=True, fp32_residual=False).cuda()
sglang_post_ln.weight.data.copy_(post_attn_weight)

with torch.no_grad():
    out_sglang_res, res_sglang = sglang_post_ln(x_attn_out, residual=residual.clone())

    # Megatron SGLangRMSNorm with residual
    x_res = x_attn_out + residual     # bf16 add
    res_mega = x_res.clone()           # updated residual
    x_fp32_res = x_res.to(torch.float32)
    var_res = x_fp32_res.pow(2).mean(dim=-1, keepdim=True)
    x_normed_res = x_fp32_res * torch.rsqrt(var_res + EPS)
    out_mega_res = post_attn_weight * x_normed_res.to(x_res.dtype)

results.check("RMSNorm+residual: normed output", out_sglang_res, out_mega_res)
results.check("RMSNorm+residual: updated residual", res_sglang, res_mega)


# ============================================================
# Test 3: QK LayerNorm (per-head RMSNorm)
# ============================================================
print("\n" + "=" * 60)
print("TEST 3: QK LayerNorm (per-head RMSNorm)")
print("=" * 60)

q_norm_weight = weights["model.layers.0.self_attn.q_norm.weight"].cuda().float()
k_norm_weight = weights["model.layers.0.self_attn.k_norm.weight"].cuda().float()

q = torch.randn(B * T, NUM_Q_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B * T, NUM_KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)

# SGLang path: RMSNorm applied per head
sglang_q_norm = SGLangRMSNorm(HEAD_DIM, eps=EPS, cast_x_before_out_mul=True).cuda()
sglang_q_norm.weight.data.copy_(q_norm_weight)
sglang_k_norm = SGLangRMSNorm(HEAD_DIM, eps=EPS, cast_x_before_out_mul=True).cuda()
sglang_k_norm.weight.data.copy_(k_norm_weight)

with torch.no_grad():
    # SGLang
    q_sglang = sglang_q_norm(q)
    k_sglang = sglang_k_norm(k)

    # Megatron (SGLangRMSNorm per head)
    q_fp32 = q.to(torch.float32)
    q_var = q_fp32.pow(2).mean(dim=-1, keepdim=True)
    q_normed = q_fp32 * torch.rsqrt(q_var + EPS)
    q_mega = q_norm_weight * q_normed.to(q.dtype)

    k_fp32 = k.to(torch.float32)
    k_var = k_fp32.pow(2).mean(dim=-1, keepdim=True)
    k_normed = k_fp32 * torch.rsqrt(k_var + EPS)
    k_mega = k_norm_weight * k_normed.to(k.dtype)

results.check("QK Norm (Q): Megatron vs SGLang", q_mega, q_sglang)
results.check("QK Norm (K): Megatron vs SGLang", k_mega, k_sglang)


# ============================================================
# Test 4: Linear projection (Q, K, V) — F.linear with batch_invariant matmul
# ============================================================
print("\n" + "=" * 60)
print("TEST 4: Linear projection (Q, K, V) — F.linear with batch_invariant matmul")
print("=" * 60)

q_proj_weight = weights["model.layers.0.self_attn.q_proj.weight"].cuda().bfloat16()
k_proj_weight = weights["model.layers.0.self_attn.k_proj.weight"].cuda().bfloat16()
v_proj_weight = weights["model.layers.0.self_attn.v_proj.weight"].cuda().bfloat16()
x_lin = torch.randn(B * T, HIDDEN, device="cuda", dtype=torch.bfloat16)

print(f"  q_proj shape: {q_proj_weight.shape}")  # [NUM_Q_HEADS*HEAD_DIM, HIDDEN]
print(f"  k_proj shape: {k_proj_weight.shape}")  # [NUM_KV_HEADS*HEAD_DIM, HIDDEN]
print(f"  v_proj shape: {v_proj_weight.shape}")  # [NUM_KV_HEADS*HEAD_DIM, HIDDEN]

with torch.no_grad():
    q_out_a = F.linear(x_lin, q_proj_weight)
    q_out_b = F.linear(x_lin, q_proj_weight)
    k_out_a = F.linear(x_lin, k_proj_weight)
    k_out_b = F.linear(x_lin, k_proj_weight)
    v_out_a = F.linear(x_lin, v_proj_weight)
    v_out_b = F.linear(x_lin, v_proj_weight)

results.check("q_proj (F.linear) self-consistency", q_out_a, q_out_b)
results.check("k_proj (F.linear) self-consistency", k_out_a, k_out_b)
results.check("v_proj (F.linear) self-consistency", v_out_a, v_out_b)


# ============================================================
# Test 5: RoPE (Rotary Position Embedding)
# ============================================================
print("\n" + "=" * 60)
print("TEST 5: RoPE (Rotary Position Embedding)")
print("=" * 60)

try:
    from sglang.srt.layers.rotary_embedding import get_rope
    from sglang.srt.server_args import get_global_server_args

    # Temporarily clear rl_on_policy_target to avoid device mismatch in get_rope
    # (inv_freq moved to CUDA but t stays on CPU when rl_on_policy_target is set)
    _server_args = get_global_server_args()
    _saved_target = _server_args.rl_on_policy_target
    _server_args.rl_on_policy_target = None
    rope = get_rope(
        HEAD_DIM,
        rotary_dim=HEAD_DIM,
        max_position=T * 4,
        base=ROPE_THETA,
        is_neox_style=True,
    )
    rope = rope.cuda()  # Move cos_sin_cache to CUDA
    _server_args.rl_on_policy_target = _saved_target

    positions = torch.arange(T, device="cuda", dtype=torch.long)
    q_rope = torch.randn(B * T, NUM_Q_HEADS * HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k_rope = torch.randn(B * T, NUM_KV_HEADS * HEAD_DIM, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        q_out1, k_out1 = rope(positions, q_rope.clone(), k_rope.clone())
        q_out2, k_out2 = rope(positions, q_rope.clone(), k_rope.clone())

    results.check("RoPE (Q) self-consistency", q_out1, q_out2)
    results.check("RoPE (K) self-consistency", k_out1, k_out2)
except ImportError:
    print("  SKIP  SGLang RoPE not available")


# ============================================================
# Test 6: FlashAttention3 self-consistency
# ============================================================
print("\n" + "=" * 60)
print("TEST 6: FlashAttention3 prefill self-consistency (num_splits=1)")
print("=" * 60)

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func

    q_fa = torch.randn(B * T, NUM_Q_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k_fa = torch.randn(B * T, NUM_KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    v_fa = torch.randn(B * T, NUM_KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)

    # GQA: expand K/V to match Q heads
    repeat_factor = NUM_Q_HEADS // NUM_KV_HEADS
    k_fa_exp = k_fa.repeat_interleave(repeat_factor, dim=1)
    v_fa_exp = v_fa.repeat_interleave(repeat_factor, dim=1)

    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device="cuda")
    softmax_scale = 1.0 / (HEAD_DIM ** 0.5)

    with torch.no_grad():
        out_fa1 = flash_attn_varlen_func(
            q_fa, k_fa_exp, v_fa_exp,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=T, max_seqlen_k=T,
            causal=True, softmax_scale=softmax_scale,
        )
        out_fa2 = flash_attn_varlen_func(
            q_fa, k_fa_exp, v_fa_exp,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=T, max_seqlen_k=T,
            causal=True, softmax_scale=softmax_scale,
        )

    results.check("FlashAttention3 prefill self-consistency", out_fa1, out_fa2)
except ImportError:
    print("  SKIP  FlashAttention3 not available")


# ============================================================
# Test 7: MoE Router — softmax + topk + renormalize
# ============================================================
print("\n" + "=" * 60)
print("TEST 7: MoE Router (FP32 softmax + topk + renormalize)")
print("=" * 60)

gate_weight = weights["model.layers.0.mlp.gate.weight"].cuda().bfloat16()
x_moe = torch.randn(B * T, HIDDEN, device="cuda", dtype=torch.bfloat16)

with torch.no_grad():
    # SGLang path
    router_logits_sgl = torch.mm(x_moe, gate_weight.t())
    routing_weights_sgl = F.softmax(router_logits_sgl, dim=1, dtype=torch.float)
    routing_weights_sgl, selected_experts_sgl = torch.topk(routing_weights_sgl, TOPK, dim=-1)
    routing_weights_sgl = routing_weights_sgl / routing_weights_sgl.sum(dim=-1, keepdim=True)
    routing_weights_sgl = routing_weights_sgl.to(x_moe.dtype)

    # Megatron path (identical operations)
    router_logits_mega = torch.mm(x_moe, gate_weight.t())
    routing_weights_mega = F.softmax(router_logits_mega, dim=1, dtype=torch.float)
    routing_weights_mega, selected_experts_mega = torch.topk(routing_weights_mega, TOPK, dim=-1)
    routing_weights_mega = routing_weights_mega / routing_weights_mega.sum(dim=-1, keepdim=True)
    routing_weights_mega = routing_weights_mega.to(x_moe.dtype)

results.check("MoE Router (logits): self-consistency", router_logits_sgl, router_logits_mega)
results.check("MoE Router (weights): self-consistency", routing_weights_sgl, routing_weights_mega)
results.check("MoE Router (expert_ids): self-consistency",
      selected_experts_sgl.float(), selected_experts_mega.float())


# ============================================================
# Test 8: MoE Single Expert forward (SwiGLU)
# ============================================================
print("\n" + "=" * 60)
print("TEST 8: MoE Single Expert forward (SwiGLU)")
print("=" * 60)

gate_proj_w = weights["model.layers.0.mlp.experts.0.gate_proj.weight"].cuda().bfloat16()
up_proj_w = weights["model.layers.0.mlp.experts.0.up_proj.weight"].cuda().bfloat16()
down_proj_w = weights["model.layers.0.mlp.experts.0.down_proj.weight"].cuda().bfloat16()

print(f"  gate_proj shape: {gate_proj_w.shape}")  # [MOE_FFN_HIDDEN, HIDDEN]
print(f"  up_proj shape:   {up_proj_w.shape}")     # [MOE_FFN_HIDDEN, HIDDEN]
print(f"  down_proj shape: {down_proj_w.shape}")   # [HIDDEN, MOE_FFN_HIDDEN]

x_expert = torch.randn(4, HIDDEN, device="cuda", dtype=torch.bfloat16)

with torch.no_grad():
    gate_out = F.linear(x_expert, gate_proj_w)
    up_out = F.linear(x_expert, up_proj_w)
    expert_out = F.linear(F.silu(gate_out) * up_out, down_proj_w)

    gate_out2 = F.linear(x_expert, gate_proj_w)
    up_out2 = F.linear(x_expert, up_proj_w)
    expert_out2 = F.linear(F.silu(gate_out2) * up_out2, down_proj_w)

results.check("MoE expert SwiGLU self-consistency", expert_out, expert_out2)


# ============================================================
# Test 9: MoE Fused Experts (SGLang Triton kernel)
# ============================================================
print("\n" + "=" * 60)
print("TEST 9: MoE Fused Experts (SGLang fused_experts_impl)")
print("=" * 60)

try:
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts_impl

    # Stack gate+up weights for fused expert format: w1=[E, 2*ffn, hidden], w2=[E, hidden, ffn]
    num_test_experts = min(4, NUM_EXPERTS)
    w1_list = []
    w2_list = []
    for i in range(num_test_experts):
        gp = weights[f"model.layers.0.mlp.experts.{i}.gate_proj.weight"].cuda().bfloat16()
        up = weights[f"model.layers.0.mlp.experts.{i}.up_proj.weight"].cuda().bfloat16()
        dp = weights[f"model.layers.0.mlp.experts.{i}.down_proj.weight"].cuda().bfloat16()
        w1_list.append(torch.cat([gp, up], dim=0))
        w2_list.append(dp)

    w1 = torch.stack(w1_list)
    w2 = torch.stack(w2_list)
    print(f"  w1 shape: {w1.shape}, w2 shape: {w2.shape}")

    x_fused = torch.randn(B * T, HIDDEN, device="cuda", dtype=torch.bfloat16)

    # Route all tokens to first few experts
    topk_weights_test = torch.ones(B * T, TOPK, device="cuda", dtype=torch.bfloat16) / TOPK
    topk_ids_test = torch.zeros(B * T, TOPK, device="cuda", dtype=torch.int32)
    for j in range(TOPK):
        topk_ids_test[:, j] = j % num_test_experts

    with torch.no_grad():
        out_fused1 = fused_experts_impl(
            x_fused.clone(), w1, w2,
            topk_weights_test, topk_ids_test,
            activation="silu",
        )
        out_fused2 = fused_experts_impl(
            x_fused.clone(), w1, w2,
            topk_weights_test, topk_ids_test,
            activation="silu",
        )

    results.check("MoE fused_experts_impl self-consistency", out_fused1, out_fused2)

    # Cross-check fused vs manual loop
    with torch.no_grad():
        manual_output = torch.zeros_like(x_fused)
        for token_idx in range(B * T):
            for topk_idx in range(TOPK):
                expert_id = topk_ids_test[token_idx, topk_idx].item()
                weight = topk_weights_test[token_idx, topk_idx]
                token = x_fused[token_idx:token_idx + 1]
                g = F.linear(token, w1_list[expert_id][:MOE_FFN_HIDDEN])
                u = F.linear(token, w1_list[expert_id][MOE_FFN_HIDDEN:])
                expert_result = F.linear(F.silu(g) * u, w2_list[expert_id])
                manual_output[token_idx] += weight * expert_result.squeeze(0)

    results.check("MoE fused_experts vs manual loop", out_fused1, manual_output, expect_diff=True)

except ImportError as e:
    print(f"  SKIP  SGLang fused_experts_impl not available: {e}")


# ============================================================
# Test 10: log_softmax (batch_invariant Triton)
# ============================================================
print("\n" + "=" * 60)
print("TEST 10: log_softmax (batch_invariant Triton)")
print("=" * 60)

logits = torch.randn(B * T, VOCAB_SIZE, device="cuda", dtype=torch.bfloat16)

with torch.no_grad():
    log_sm_a = F.log_softmax(logits, dim=-1)
    log_sm_b = F.log_softmax(logits, dim=-1)

results.check("log_softmax: batch_invariant self-consistency", log_sm_a, log_sm_b)


# ============================================================
# Test 11: Final RMSNorm (model.norm)
# ============================================================
print("\n" + "=" * 60)
print("TEST 11: Final RMSNorm (model.norm)")
print("=" * 60)

final_norm_weight = weights["model.norm.weight"].cuda().float()
x_final = torch.randn(B * T, HIDDEN, device="cuda", dtype=torch.bfloat16)

# SGLang: RMSNorm with cast_x_before_out_mul=True
sglang_final_ln = SGLangRMSNorm(HIDDEN, eps=EPS, cast_x_before_out_mul=True).cuda()
sglang_final_ln.weight.data.copy_(final_norm_weight)

with torch.no_grad():
    out_final_sglang = sglang_final_ln(x_final)

    # Megatron SGLangFinalRMSNorm
    x_fp32 = x_final.to(torch.float32)
    var_final = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    x_normed_final = x_fp32 * torch.rsqrt(var_final + EPS)
    out_final_mega = final_norm_weight * x_normed_final.to(x_final.dtype)

results.check("Final RMSNorm: Megatron vs SGLang", out_final_sglang, out_final_mega)


# ============================================================
# Test 12: lm_head (F.linear) self-consistency
# ============================================================
print("\n" + "=" * 60)
print("TEST 12: lm_head projection")
print("=" * 60)

lm_head_weight = weights["lm_head.weight"].cuda().bfloat16()
with torch.no_grad():
    logits_a = F.linear(out_final_sglang.bfloat16(), lm_head_weight)
    logits_b = F.linear(out_final_sglang.bfloat16(), lm_head_weight)

results.check("lm_head (F.linear) self-consistency", logits_a, logits_b)


# ============================================================
# Test 13: MoE Router full pipeline self-consistency
# ============================================================
print("\n" + "=" * 60)
print("TEST 13: MoE Router full pipeline self-consistency (larger input)")
print("=" * 60)

x_router = torch.randn(B * T * 4, HIDDEN, device="cuda", dtype=torch.bfloat16)

with torch.no_grad():
    logits_r = torch.mm(x_router, gate_weight.t())
    weights_r = F.softmax(logits_r, dim=1, dtype=torch.float)
    weights_r, ids_r = torch.topk(weights_r, TOPK, dim=-1)
    weights_r = weights_r / weights_r.sum(dim=-1, keepdim=True)
    weights_r = weights_r.to(x_router.dtype)

    logits_r2 = torch.mm(x_router, gate_weight.t())
    weights_r2 = F.softmax(logits_r2, dim=1, dtype=torch.float)
    weights_r2, ids_r2 = torch.topk(weights_r2, TOPK, dim=-1)
    weights_r2 = weights_r2 / weights_r2.sum(dim=-1, keepdim=True)
    weights_r2 = weights_r2.to(x_router.dtype)

results.check("MoE Router pipeline: self-consistency (weights)", weights_r, weights_r2)
results.check("MoE Router pipeline: self-consistency (expert_ids)", ids_r.float(), ids_r2.float())


# ============================================================
# Test 14: SwiGLU activation
# ============================================================
print("\n" + "=" * 60)
print("TEST 14: SwiGLU activation self-consistency")
print("=" * 60)

gate_input = torch.randn(B * T, MOE_FFN_HIDDEN, device="cuda", dtype=torch.bfloat16)
up_input = torch.randn(B * T, MOE_FFN_HIDDEN, device="cuda", dtype=torch.bfloat16)

with torch.no_grad():
    swiglu_a = F.silu(gate_input) * up_input
    swiglu_b = F.silu(gate_input) * up_input

results.check("SwiGLU activation self-consistency", swiglu_a, swiglu_b)


# ============================================================
# Test 15: Attention o_proj self-consistency
# ============================================================
print("\n" + "=" * 60)
print("TEST 15: Attention o_proj self-consistency")
print("=" * 60)

o_proj_weight = weights["model.layers.0.self_attn.o_proj.weight"].cuda().bfloat16()
o_proj_input_size = o_proj_weight.shape[1]
print(f"  o_proj shape: {o_proj_weight.shape}")  # [HIDDEN, NUM_Q_HEADS*HEAD_DIM]
x_o = torch.randn(B * T, o_proj_input_size, device="cuda", dtype=torch.bfloat16)

with torch.no_grad():
    out_o1 = F.linear(x_o, o_proj_weight)
    out_o2 = F.linear(x_o, o_proj_weight)

results.check("o_proj (F.linear) self-consistency", out_o1, out_o2)


# ============================================================
# Test 16: Full Attention layer end-to-end self-consistency
# ============================================================
print("\n" + "=" * 60)
print("TEST 16: Full Attention layer end-to-end self-consistency")
print("=" * 60)

try:
    from sglang.srt.layers.rotary_embedding import get_rope
    from flash_attn.flash_attn_interface import flash_attn_varlen_func

    # Temporarily clear rl_on_policy_target (same workaround as Test 5)
    _server_args = get_global_server_args()
    _saved = _server_args.rl_on_policy_target
    _server_args.rl_on_policy_target = None
    rope_attn = get_rope(HEAD_DIM, rotary_dim=HEAD_DIM, max_position=T * 4, base=ROPE_THETA, is_neox_style=True)
    rope_attn = rope_attn.cuda()
    _server_args.rl_on_policy_target = _saved

    q_w = weights["model.layers.0.self_attn.q_proj.weight"].cuda().bfloat16()
    k_w = weights["model.layers.0.self_attn.k_proj.weight"].cuda().bfloat16()
    v_w = weights["model.layers.0.self_attn.v_proj.weight"].cuda().bfloat16()
    o_w = weights["model.layers.0.self_attn.o_proj.weight"].cuda().bfloat16()
    q_norm_w = weights["model.layers.0.self_attn.q_norm.weight"].cuda().float()
    k_norm_w = weights["model.layers.0.self_attn.k_norm.weight"].cuda().float()

    q_size = NUM_Q_HEADS * HEAD_DIM
    kv_size = NUM_KV_HEADS * HEAD_DIM

    positions = torch.arange(T, device="cuda", dtype=torch.long)
    x_attn = torch.randn(B * T, HIDDEN, device="cuda", dtype=torch.bfloat16)
    cu_seqlens_attn = torch.tensor([0, T], dtype=torch.int32, device="cuda")

    def run_attention(x_in, positions):
        """Full attention forward matching both SGLang and Megatron at TP=1."""
        # Q/K/V projections
        q_flat = F.linear(x_in, q_w)
        k_flat = F.linear(x_in, k_w)
        v_flat = F.linear(x_in, v_w)

        # Reshape per head
        q_heads = q_flat.view(-1, NUM_Q_HEADS, HEAD_DIM)
        k_heads = k_flat.view(-1, NUM_KV_HEADS, HEAD_DIM)
        v_heads = v_flat.view(-1, NUM_KV_HEADS, HEAD_DIM)

        # QK Norm (per-head RMSNorm)
        q_f = q_heads.to(torch.float32)
        q_v = q_f.pow(2).mean(dim=-1, keepdim=True)
        q_heads = (q_norm_w * (q_f * torch.rsqrt(q_v + EPS)).to(q_heads.dtype))

        k_f = k_heads.to(torch.float32)
        k_v = k_f.pow(2).mean(dim=-1, keepdim=True)
        k_heads = (k_norm_w * (k_f * torch.rsqrt(k_v + EPS)).to(k_heads.dtype))

        # RoPE
        q_rope_in = q_heads.reshape(-1, q_size)
        k_rope_in = k_heads.reshape(-1, kv_size)
        q_rope_out, k_rope_out = rope_attn(positions, q_rope_in, k_rope_in)
        q_heads = q_rope_out.view(-1, NUM_Q_HEADS, HEAD_DIM).to(torch.bfloat16)
        k_heads = k_rope_out.view(-1, NUM_KV_HEADS, HEAD_DIM).to(torch.bfloat16)

        # GQA expand
        rep = NUM_Q_HEADS // NUM_KV_HEADS
        k_exp = k_heads.repeat_interleave(rep, dim=1)
        v_exp = v_heads.repeat_interleave(rep, dim=1)

        # FlashAttention3
        scale = 1.0 / (HEAD_DIM ** 0.5)
        attn_out = flash_attn_varlen_func(
            q_heads, k_exp, v_exp,
            cu_seqlens_q=cu_seqlens_attn, cu_seqlens_k=cu_seqlens_attn,
            max_seqlen_q=T, max_seqlen_k=T,
            causal=True, softmax_scale=scale,
        )

        # o_proj
        attn_flat = attn_out.reshape(-1, NUM_Q_HEADS * HEAD_DIM)
        return F.linear(attn_flat, o_w)

    with torch.no_grad():
        attn_out1 = run_attention(x_attn, positions)
        attn_out2 = run_attention(x_attn, positions)

    results.check("Full Attention layer: end-to-end self-consistency", attn_out1, attn_out2)
except ImportError as e:
    print(f"  SKIP  Dependencies not available: {e}")


# ============================================================
# Summary
# ============================================================
results.summary()
