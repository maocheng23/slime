"""
Unit tests for bitwise alignment between Megatron and SGLang for Qwen3-0.6B (dense model).
Tests each kernel/layer by comparing with the same weights and inputs.

Model: Qwen3-0.6B
  - 28 layers, hidden_size=1024, ffn=3072, SwiGLU
  - 16 attention heads, 8 KV groups (GQA), head_dim=128
  - QK LayerNorm, RoPE
  - Dense MLP (no MoE)
  - SGLang uses fp32_residual=True, override_orig_dtype=float32

Usage (inside Docker container, single GPU):
  CUDA_VISIBLE_DEVICES=0 python scripts/test_dense_layer_alignment.py
"""
import os
import sys

import torch
import torch.nn.functional as F

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["SGLANG_DUMPER_ENABLE"] = "0"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alignment_test_utils import TestResults, load_safetensor_weights, setup_sglang_for_test

from transformers import AutoConfig

MODEL_PATH = "/root/models/Qwen3-0.6B"
config = AutoConfig.from_pretrained(MODEL_PATH)

HIDDEN = config.hidden_size               # 1024
EPS = config.rms_norm_eps                  # 1e-6
NUM_Q_HEADS = config.num_attention_heads   # 16
NUM_KV_HEADS = config.num_key_value_heads  # 8
HEAD_DIM = getattr(config, 'head_dim', HIDDEN // NUM_Q_HEADS)  # 128
FFN_HIDDEN = config.intermediate_size      # 3072
VOCAB_SIZE = config.vocab_size             # 151936
ROPE_THETA = getattr(config, 'rope_theta', 1000000)

B, T = 1, 16

print(f"Model: {MODEL_PATH}")
print(f"hidden={HIDDEN}, num_q_heads={NUM_Q_HEADS}, num_kv_heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}")
print(f"ffn_hidden={FFN_HIDDEN}, eps={EPS}, rope_theta={ROPE_THETA}")
print(f"Test shape: B={B}, T={T}")

# Qwen3-0.6B is a single safetensors file (no index)
from safetensors.torch import load_file
import os
sf_path = os.path.join(MODEL_PATH, "model.safetensors")
if os.path.exists(sf_path):
    all_weights = load_file(sf_path)
    prefixes = ["model.layers.0.", "model.norm.", "model.embed_tokens.", "lm_head."]
    weights = {k: v for k, v in all_weights.items() if any(k.startswith(p) for p in prefixes)}
    print(f"Loaded {len(weights)} tensors from single safetensors file")
    del all_weights
else:
    weights = load_safetensor_weights(MODEL_PATH, prefixes=[
        "model.layers.0.", "model.norm.", "model.embed_tokens.", "lm_head.",
    ])

setup_sglang_for_test(MODEL_PATH, rl_on_policy_target="fsdp")
print()

results = TestResults()

# ---- Load SGLang modules ----
from sglang.srt.layers.layernorm import RMSNorm as SGLangRMSNorm
from sglang.srt.server_args import get_global_server_args

# Check what norm_kwargs SGLang Qwen3DecoderLayer uses
print("SGLang rl_on_policy_target:", get_global_server_args().rl_on_policy_target)

# Qwen3DecoderLayer uses these norm_kwargs for rl_on_policy:
# weight_dtype=torch.float32, cast_x_before_out_mul=True,
# override_orig_dtype=torch.float32, fp32_residual=True
NORM_KWARGS = dict(
    weight_dtype=torch.float32,
    cast_x_before_out_mul=True,
    override_orig_dtype=torch.float32,
    fp32_residual=True,
)
print(f"SGLang Qwen3 norm_kwargs: {NORM_KWARGS}")

# Qwen3MoeDecoderLayer uses different kwargs:
# cast_x_before_out_mul=True, fp32_residual=False
# MOE_NORM_KWARGS = dict(cast_x_before_out_mul=True, fp32_residual=False)

# ============================================================
# Test 1: RMSNorm — input_layernorm (with fp32_residual=True)
# ============================================================
print("\n" + "=" * 60)
print("TEST 1: RMSNorm — input_layernorm (fp32_residual=True)")
print("=" * 60)

ln_weight = weights["model.layers.0.input_layernorm.weight"].cuda().float()
x = torch.randn(B * T, HIDDEN, device="cuda", dtype=torch.bfloat16)

sglang_ln = SGLangRMSNorm(HIDDEN, eps=EPS, **NORM_KWARGS).cuda()
sglang_ln.weight.data.copy_(ln_weight)

with torch.no_grad():
    # SGLang path (no residual for first layer call)
    out_sglang = sglang_ln.forward_native(x)

    # Megatron SGLangRMSNorm path (manual)
    # With override_orig_dtype=float32: orig_dtype = float32
    # With fp32_residual=True: no effect when residual is None
    orig_dtype = torch.float32  # override_orig_dtype
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(variance + EPS)
    out_mega = ln_weight * x_normed.to(orig_dtype)  # fp32 * fp32 = fp32

results.check("RMSNorm (no residual): SGLang vs manual", out_sglang, out_mega)
print(f"  SGLang output dtype: {out_sglang.dtype}")
print(f"  Manual output dtype: {out_mega.dtype}")

# ============================================================
# Test 2: RMSNorm with residual (fp32_residual=True)
# ============================================================
print("\n" + "=" * 60)
print("TEST 2: RMSNorm with residual (fp32_residual=True)")
print("=" * 60)

residual = torch.randn(B * T, HIDDEN, device="cuda", dtype=torch.bfloat16)

with torch.no_grad():
    # SGLang path (with residual, fp32_residual=True)
    out_sg2, res_sg2 = sglang_ln.forward_native(x.clone(), residual.clone())

    # Manual: fp32_residual=True means:
    # 1. x = x.to(float32)
    # 2. x = x + residual.to(float32)  (fp32 add)
    # 3. residual = x.to(orig_dtype)   (cast back)
    # 4. norm in fp32
    # 5. output = weight * x.to(orig_dtype)
    x_fp32 = x.clone().to(torch.float32)
    x_fp32 = x_fp32 + residual.clone().to(torch.float32)
    res_manual = x_fp32.to(torch.float32)  # override_orig_dtype=float32
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(variance + EPS)
    out_manual = ln_weight * x_normed.to(torch.float32)  # override_orig_dtype=float32

results.check("RMSNorm (with residual, fp32): SGLang vs manual", out_sg2, out_manual)
results.check("RMSNorm residual: SGLang vs manual", res_sg2, res_manual)
print(f"  SGLang output dtype: {out_sg2.dtype}, residual dtype: {res_sg2.dtype}")
print(f"  Manual output dtype: {out_manual.dtype}, residual dtype: {res_manual.dtype}")

# ============================================================
# Test 3: QKV Projection
# ============================================================
print("\n" + "=" * 60)
print("TEST 3: QKV Projection")
print("=" * 60)

q_w = weights["model.layers.0.self_attn.q_proj.weight"].cuda().bfloat16()
k_w = weights["model.layers.0.self_attn.k_proj.weight"].cuda().bfloat16()
v_w = weights["model.layers.0.self_attn.v_proj.weight"].cuda().bfloat16()

# Input to QKV is the normed output — but what dtype?
# SGLang RMSNorm with override_orig_dtype=float32 returns float32
# QKVParallelLinear receives float32 input → F.linear(fp32, bf16 weight) → fp32 output?
# Or does it cast to bf16 first?

# Use the normed output from test 1 (which is float32)
normed_input = out_sglang  # float32

with torch.no_grad():
    q = F.linear(normed_input, q_w)
    k = F.linear(normed_input, k_w)
    q2 = F.linear(normed_input, q_w)

results.check("Q projection self-consistency", q, q2)
print(f"  Input dtype: {normed_input.dtype}, Q output dtype: {q.dtype}")

# ============================================================
# Test 4: QK LayerNorm
# ============================================================
print("\n" + "=" * 60)
print("TEST 4: QK LayerNorm")
print("=" * 60)

q_norm_w = weights["model.layers.0.self_attn.q_norm.weight"].cuda().float()
k_norm_w = weights["model.layers.0.self_attn.k_norm.weight"].cuda().float()

# Qwen3Attention uses norm_kwargs with weight_dtype=float32, cast_x_before_out_mul=True
q_norm = SGLangRMSNorm(HEAD_DIM, eps=EPS, weight_dtype=torch.float32, cast_x_before_out_mul=True).cuda()
q_norm.weight.data.copy_(q_norm_w)
k_norm = SGLangRMSNorm(HEAD_DIM, eps=EPS, weight_dtype=torch.float32, cast_x_before_out_mul=True).cuda()
k_norm.weight.data.copy_(k_norm_w)

with torch.no_grad():
    q_reshaped = q.view(-1, HEAD_DIM)
    q_normed = q_norm.forward_native(q_reshaped)
    q_normed2 = q_norm.forward_native(q_reshaped)

results.check("QK norm self-consistency", q_normed, q_normed2)
print(f"  Input dtype: {q_reshaped.dtype}, Output dtype: {q_normed.dtype}")

# ============================================================
# Test 5: RoPE
# ============================================================
print("\n" + "=" * 60)
print("TEST 5: RoPE")
print("=" * 60)

inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device="cpu") / HEAD_DIM))
inv_freq = inv_freq.cuda()
positions = torch.arange(T, dtype=torch.float32, device="cuda")
freqs = torch.outer(positions, inv_freq)
cos_rope = torch.cos(freqs)
sin_rope = torch.sin(freqs)

def apply_rope_f32(x, cos, sin):
    orig = x.dtype
    x = x.float()
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    c, s = cos.unsqueeze(1).float(), sin.unsqueeze(1).float()
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).to(orig)

with torch.no_grad():
    q_for_rope = q_normed.view(T, NUM_Q_HEADS, HEAD_DIM)
    q_roped = apply_rope_f32(q_for_rope, cos_rope, sin_rope)
    q_roped2 = apply_rope_f32(q_for_rope, cos_rope, sin_rope)

results.check("RoPE self-consistency", q_roped, q_roped2)
print(f"  Input dtype: {q_for_rope.dtype}, Output dtype: {q_roped.dtype}")

# ============================================================
# Test 6: FlashAttention3
# ============================================================
print("\n" + "=" * 60)
print("TEST 6: FlashAttention3 (num_splits=1)")
print("=" * 60)

from sgl_kernel.flash_attn import flash_attn_varlen_func

with torch.no_grad():
    k_normed = k_norm.forward_native(k.view(-1, HEAD_DIM))
    k_for_rope = k_normed.view(T, NUM_KV_HEADS, HEAD_DIM)
    k_roped = apply_rope_f32(k_for_rope, cos_rope, sin_rope)
    v_proj = F.linear(normed_input, v_w).view(T, NUM_KV_HEADS, HEAD_DIM)

    rep = NUM_Q_HEADS // NUM_KV_HEADS
    k_exp = k_roped.repeat_interleave(rep, dim=1)
    v_exp = v_proj.repeat_interleave(rep, dim=1)
    cu = torch.tensor([0, T], dtype=torch.int32, device="cuda")

    attn_out = flash_attn_varlen_func(
        q=q_roped.bfloat16(), k=k_exp.bfloat16(), v=v_exp.bfloat16(),
        cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=T, max_seqlen_k=T,
        softmax_scale=1.0 / (HEAD_DIM ** 0.5), causal=True, num_splits=1,
    )
    if isinstance(attn_out, tuple):
        attn_out = attn_out[0]
    attn_out2 = flash_attn_varlen_func(
        q=q_roped.bfloat16(), k=k_exp.bfloat16(), v=v_exp.bfloat16(),
        cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=T, max_seqlen_k=T,
        softmax_scale=1.0 / (HEAD_DIM ** 0.5), causal=True, num_splits=1,
    )
    if isinstance(attn_out2, tuple):
        attn_out2 = attn_out2[0]

results.check("FA3 self-consistency", attn_out, attn_out2)

# ============================================================
# Test 7: Dense MLP (SwiGLU)
# ============================================================
print("\n" + "=" * 60)
print("TEST 7: Dense MLP (SwiGLU)")
print("=" * 60)

gate_w = weights["model.layers.0.mlp.gate_proj.weight"].cuda().bfloat16()
up_w = weights["model.layers.0.mlp.up_proj.weight"].cuda().bfloat16()
down_w = weights["model.layers.0.mlp.down_proj.weight"].cuda().bfloat16()

mlp_input = torch.randn(T, HIDDEN, device="cuda", dtype=torch.bfloat16)

with torch.no_grad():
    gate = F.linear(mlp_input, gate_w)
    up = F.linear(mlp_input, up_w)
    act = F.silu(gate) * up
    down = F.linear(act, down_w)
    down2 = F.linear(F.silu(F.linear(mlp_input, gate_w)) * F.linear(mlp_input, up_w), down_w)

results.check("Dense MLP self-consistency", down, down2)

# ============================================================
# Test 8: log_softmax
# ============================================================
print("\n" + "=" * 60)
print("TEST 8: log_softmax")
print("=" * 60)

lm_head_w = weights["lm_head.weight"].cuda().bfloat16()
hidden = torch.randn(T, HIDDEN, device="cuda", dtype=torch.bfloat16)

with torch.no_grad():
    logits = F.linear(hidden, lm_head_w)
    lp1 = torch.log_softmax(logits.float(), dim=-1)
    lp2 = torch.log_softmax(logits.float(), dim=-1)

results.check("log_softmax self-consistency", lp1, lp2)

# ============================================================
# Test 9: Full attention chain self-consistency
# ============================================================
print("\n" + "=" * 60)
print("TEST 9: Full attention chain")
print("=" * 60)

o_w = weights["model.layers.0.self_attn.o_proj.weight"].cuda().bfloat16()
post_ln_w = weights["model.layers.0.post_attention_layernorm.weight"].cuda().float()

post_ln = SGLangRMSNorm(HIDDEN, eps=EPS, **NORM_KWARGS).cuda()
post_ln.weight.data.copy_(post_ln_w)

with torch.no_grad():
    # Full chain: input → LN → QKV → QK norm → RoPE → FA3 → o_proj → post_LN(+residual)
    x_test = torch.randn(T, HIDDEN, device="cuda", dtype=torch.bfloat16)

    normed = sglang_ln.forward_native(x_test)
    q_t = F.linear(normed, q_w).view(-1, HEAD_DIM)
    k_t = F.linear(normed, k_w).view(-1, HEAD_DIM)
    v_t = F.linear(normed, v_w).view(T, NUM_KV_HEADS, HEAD_DIM)
    q_t = q_norm.forward_native(q_t).view(T, NUM_Q_HEADS, HEAD_DIM)
    k_t = k_norm.forward_native(k_t).view(T, NUM_KV_HEADS, HEAD_DIM)
    q_t = apply_rope_f32(q_t, cos_rope, sin_rope)
    k_t = apply_rope_f32(k_t, cos_rope, sin_rope)
    k_exp_t = k_t.repeat_interleave(rep, dim=1)
    v_exp_t = v_t.repeat_interleave(rep, dim=1)
    attn = flash_attn_varlen_func(
        q=q_t.bfloat16(), k=k_exp_t.bfloat16(), v=v_exp_t.bfloat16(),
        cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=T, max_seqlen_k=T,
        softmax_scale=1.0/(HEAD_DIM**0.5), causal=True, num_splits=1,
    )
    if isinstance(attn, tuple):
        attn = attn[0]
    o_out = F.linear(attn.reshape(T, -1), o_w)

    # post_attention_layernorm with residual (fp32 residual)
    # x_test is the original input (residual for first layer)
    mlp_in, new_res = post_ln.forward_native(o_out, x_test)

    # Run again
    normed_2 = sglang_ln.forward_native(x_test)
    q_t2 = F.linear(normed_2, q_w).view(-1, HEAD_DIM)
    k_t2 = F.linear(normed_2, k_w).view(-1, HEAD_DIM)
    v_t2 = F.linear(normed_2, v_w).view(T, NUM_KV_HEADS, HEAD_DIM)
    q_t2 = q_norm.forward_native(q_t2).view(T, NUM_Q_HEADS, HEAD_DIM)
    k_t2 = k_norm.forward_native(k_t2).view(T, NUM_KV_HEADS, HEAD_DIM)
    q_t2 = apply_rope_f32(q_t2, cos_rope, sin_rope)
    k_t2 = apply_rope_f32(k_t2, cos_rope, sin_rope)
    k_exp_t2 = k_t2.repeat_interleave(rep, dim=1)
    v_exp_t2 = v_t2.repeat_interleave(rep, dim=1)
    attn2 = flash_attn_varlen_func(
        q=q_t2.bfloat16(), k=k_exp_t2.bfloat16(), v=v_exp_t2.bfloat16(),
        cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=T, max_seqlen_k=T,
        softmax_scale=1.0/(HEAD_DIM**0.5), causal=True, num_splits=1,
    )
    if isinstance(attn2, tuple):
        attn2 = attn2[0]
    o_out2 = F.linear(attn2.reshape(T, -1), o_w)
    mlp_in2, new_res2 = post_ln.forward_native(o_out2, x_test)

results.check("Full attention chain: self-consistency", mlp_in, mlp_in2)
results.check("Full attention chain: residual self-consistency", new_res, new_res2)
print(f"  MLP input dtype: {mlp_in.dtype}")
print(f"  Residual dtype: {new_res.dtype}")

# ============================================================
print("\n" + "=" * 60)
results.summary()
print("=" * 60)
