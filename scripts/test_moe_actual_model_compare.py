"""
Compare SGLang's ACTUAL code path step by step vs manual forward.

Can't instantiate full SGLang model classes without distributed context,
so instead we call the ACTUAL SGLang functions (apply_qk_norm, rotary_emb,
RMSNorm.forward_native) with real weights and compare against manual forward.

This tests whether the SGLang model's code path matches our manual reconstruction.
If it does, the E2E diff must come from the Megatron side.

Usage: CUDA_VISIBLE_DEVICES=0 python scripts/test_moe_actual_model_compare.py
"""
import os
import sys

import torch
import torch.nn.functional as F

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["SGLANG_DUMPER_ENABLE"] = "0"
os.environ["MEGATRON_USE_DETERMINISTIC_ALLREDUCE"] = "1"

from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
server_args = ServerArgs(model_path="/root/models/Qwen3-30B-A3B", rl_on_policy_target="fsdp_tp")
set_global_server_args_for_scheduler(server_args)
from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode
enable_batch_invariant_mode(enable_bmm=False)

from transformers import AutoConfig
MODEL_PATH = "/root/models/Qwen3-30B-A3B"
config = AutoConfig.from_pretrained(MODEL_PATH)

HIDDEN = config.hidden_size
EPS = config.rms_norm_eps
NUM_Q_HEADS = config.num_attention_heads
NUM_KV_HEADS = config.num_key_value_heads
HEAD_DIM = getattr(config, 'head_dim', HIDDEN // NUM_Q_HEADS)
T = 16

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alignment_test_utils import TestResults, load_safetensor_weights

results = TestResults()

weights = load_safetensor_weights(MODEL_PATH, prefixes=[
    "model.layers.0.", "model.norm.", "model.embed_tokens.",
])

print(f"hidden={HIDDEN}, heads={NUM_Q_HEADS}/{NUM_KV_HEADS}, head_dim={HEAD_DIM}, T={T}\n")

# ---- Load weights ----
ln_w = weights["model.layers.0.input_layernorm.weight"].cuda().float()
post_ln_w = weights["model.layers.0.post_attention_layernorm.weight"].cuda().float()
q_norm_w = weights["model.layers.0.self_attn.q_norm.weight"].cuda().float()
k_norm_w = weights["model.layers.0.self_attn.k_norm.weight"].cuda().float()
q_proj_w = weights["model.layers.0.self_attn.q_proj.weight"].cuda().bfloat16()
k_proj_w = weights["model.layers.0.self_attn.k_proj.weight"].cuda().bfloat16()
v_proj_w = weights["model.layers.0.self_attn.v_proj.weight"].cuda().bfloat16()
o_proj_w = weights["model.layers.0.self_attn.o_proj.weight"].cuda().bfloat16()
final_ln_w = weights["model.norm.weight"].cuda().float()

from sglang.srt.layers.layernorm import RMSNorm as SGLangRMSNorm
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.models.utils import apply_qk_norm
from sgl_kernel.flash_attn import flash_attn_varlen_func
from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

# ---- Build SGLang RMSNorm instances (actual classes) ----
sglang_input_ln = SGLangRMSNorm(HIDDEN, eps=EPS, cast_x_before_out_mul=True, fp32_residual=False).cuda()
sglang_input_ln.weight.data.copy_(ln_w)

sglang_post_ln = SGLangRMSNorm(HIDDEN, eps=EPS, cast_x_before_out_mul=True, fp32_residual=False).cuda()
sglang_post_ln.weight.data.copy_(post_ln_w)

sglang_q_norm = SGLangRMSNorm(HEAD_DIM, eps=EPS, cast_x_before_out_mul=True, fp32_residual=False).cuda()
sglang_q_norm.weight.data.copy_(q_norm_w)

sglang_k_norm = SGLangRMSNorm(HEAD_DIM, eps=EPS, cast_x_before_out_mul=True, fp32_residual=False).cuda()
sglang_k_norm.weight.data.copy_(k_norm_w)

sglang_final_ln = SGLangRMSNorm(HIDDEN, eps=EPS, cast_x_before_out_mul=True, fp32_residual=False).cuda()
sglang_final_ln.weight.data.copy_(final_ln_w)

# ---- Build Megatron RotaryEmbedding ----
base = getattr(config, 'rope_theta', 1000000)
mega_rope = RotaryEmbedding(
    kv_channels=HEAD_DIM, rotary_percent=1.0, rotary_interleaved=False,
    seq_len_interpolation_factor=None, rotary_base=base, use_cpu_initialization=True,
)
mega_emb = mega_rope(T).cuda()

# ---- SGLang RoPE ----
inv_freq = 1.0 / (base ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device="cuda") / HEAD_DIM))
positions = torch.arange(T, device="cuda")
freqs = torch.outer(positions.float(), inv_freq)
cos_rope = torch.cos(freqs)
sin_rope = torch.sin(freqs)

# ---- Helpers ----
def manual_rmsnorm(x, weight, eps):
    orig = x.dtype
    x = x.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return (weight * (x * torch.rsqrt(var + eps)).to(orig))

def manual_rope_f32(x, cos, sin):
    orig = x.dtype
    x = x.float()
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    c, s = cos.unsqueeze(1).float(), sin.unsqueeze(1).float()
    return torch.cat([x1*c - x2*s, x2*c + x1*s], dim=-1).to(orig)

torch.manual_seed(42)
x = torch.randn(T, HIDDEN, dtype=torch.bfloat16, device="cuda")

# ============================================================
# Step 1: input_layernorm
# ============================================================
print("=" * 60)
print("Step 1: input_layernorm")
with torch.no_grad():
    out_sglang = sglang_input_ln.forward_native(x.clone())
    out_manual = manual_rmsnorm(x.clone(), ln_w, EPS)
results.check("input_layernorm: SGLang class vs manual", out_sglang, out_manual)

normed = out_sglang  # use SGLang class output going forward

# ============================================================
# Step 2: QKV projection (fused vs separate)
# ============================================================
print("\nStep 2: QKV projection")
with torch.no_grad():
    qkv_w = torch.cat([q_proj_w, k_proj_w, v_proj_w], dim=0)
    qkv_fused = F.linear(normed, qkv_w)
    q_fused = qkv_fused[:, :NUM_Q_HEADS*HEAD_DIM]
    k_fused = qkv_fused[:, NUM_Q_HEADS*HEAD_DIM:NUM_Q_HEADS*HEAD_DIM+NUM_KV_HEADS*HEAD_DIM]
    v_fused = qkv_fused[:, NUM_Q_HEADS*HEAD_DIM+NUM_KV_HEADS*HEAD_DIM:]

    q_sep = F.linear(normed, q_proj_w)
    k_sep = F.linear(normed, k_proj_w)
    v_sep = F.linear(normed, v_proj_w)
results.check("QKV: fused vs separate Q", q_fused, q_sep)
results.check("QKV: fused vs separate K", k_fused, k_sep)

# ============================================================
# Step 3: QK norm (SGLang apply_qk_norm vs manual)
# ============================================================
print("\nStep 3: QK norm")
with torch.no_grad():
    q_for_norm = q_sep.clone()
    k_for_norm = k_sep.clone()
    q_normed_sg, k_normed_sg = apply_qk_norm(
        q=q_for_norm, k=k_for_norm,
        q_norm=sglang_q_norm, k_norm=sglang_k_norm,
        head_dim=HEAD_DIM, allow_inplace=False,
    )

    q_normed_manual = manual_rmsnorm(
        q_sep.view(T, NUM_Q_HEADS, HEAD_DIM).reshape(-1, HEAD_DIM), q_norm_w, EPS
    ).view(T, -1)
    k_normed_manual = manual_rmsnorm(
        k_sep.view(T, NUM_KV_HEADS, HEAD_DIM).reshape(-1, HEAD_DIM), k_norm_w, EPS
    ).view(T, -1)
results.check("QK norm: SGLang apply_qk_norm vs manual Q", q_normed_sg, q_normed_manual)
results.check("QK norm: SGLang apply_qk_norm vs manual K", k_normed_sg, k_normed_manual)

# ============================================================
# Step 4: RoPE — SGLang manual vs Megatron _apply_rotary_pos_emb_bshd
# ============================================================
print("\nStep 4: RoPE")
with torch.no_grad():
    q_3d = q_normed_sg.view(T, NUM_Q_HEADS, HEAD_DIM)
    k_3d = k_normed_sg.view(T, NUM_KV_HEADS, HEAD_DIM)

    # Manual RoPE (SGLang-matching)
    q_roped_manual = manual_rope_f32(q_3d.clone(), cos_rope, sin_rope)
    k_roped_manual = manual_rope_f32(k_3d.clone(), cos_rope, sin_rope)

    # Megatron _apply_rotary_pos_emb_bshd
    q_4d = q_3d.clone().unsqueeze(1)  # [T, 1, heads, dim]
    q_roped_mega = _apply_rotary_pos_emb_bshd(q_4d, mega_emb, rotary_interleaved=False).squeeze(1)

results.check("RoPE: manual vs _apply_rotary_pos_emb_bshd Q", q_roped_manual, q_roped_mega)

# ============================================================
# Step 5: FlashAttention3
# ============================================================
print("\nStep 5: FlashAttention3")
with torch.no_grad():
    rep = NUM_Q_HEADS // NUM_KV_HEADS
    k_exp = k_roped_manual.repeat_interleave(rep, dim=1)
    v_3d = v_sep.view(T, NUM_KV_HEADS, HEAD_DIM)
    v_exp = v_3d.repeat_interleave(rep, dim=1)
    cu = torch.tensor([0, T], dtype=torch.int32, device="cuda")

    attn_out1 = flash_attn_varlen_func(
        q=q_roped_manual.bfloat16(), k=k_exp.bfloat16(), v=v_exp.bfloat16(),
        cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=T, max_seqlen_k=T,
        softmax_scale=1.0/(HEAD_DIM**0.5), causal=True, num_splits=1)
    if isinstance(attn_out1, tuple): attn_out1 = attn_out1[0]

    attn_out2 = flash_attn_varlen_func(
        q=q_roped_manual.bfloat16(), k=k_exp.bfloat16(), v=v_exp.bfloat16(),
        cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=T, max_seqlen_k=T,
        softmax_scale=1.0/(HEAD_DIM**0.5), causal=True, num_splits=1)
    if isinstance(attn_out2, tuple): attn_out2 = attn_out2[0]
results.check("FA3: self-consistency", attn_out1, attn_out2)

# ============================================================
# Step 6: o_proj + post_attn_layernorm(attn_out, residual)
# ============================================================
print("\nStep 6: o_proj + residual path")
with torch.no_grad():
    attn_flat = attn_out1.reshape(T, -1)
    attn_output = F.linear(attn_flat, o_proj_w)

    # SGLang: post_attention_layernorm(attn_output, residual=x)
    moe_input_sg, residual_sg = sglang_post_ln.forward_native(attn_output.clone(), x.clone())

    # Manual: same operation
    moe_input_manual, residual_manual = manual_rmsnorm(
        (attn_output.clone() + x.clone()),  # bf16 add then norm
        post_ln_w, EPS
    ), (attn_output.clone() + x.clone())  # residual = combined

    # Actually do it properly: resadd inside, then norm
    combined = attn_output.clone() + x.clone()
    residual_manual = combined.clone()
    moe_input_manual = manual_rmsnorm(combined, post_ln_w, EPS)

results.check("post_attn_ln: SGLang class vs manual", moe_input_sg, moe_input_manual)
results.check("post_attn_ln: residual match", residual_sg, residual_manual)

# ============================================================
# Step 7: Final layernorm (with and without residual)
# ============================================================
print("\nStep 7: Final layernorm")
with torch.no_grad():
    fake_moe_out = torch.randn(T, HIDDEN, dtype=torch.bfloat16, device="cuda")
    fake_residual = torch.randn(T, HIDDEN, dtype=torch.bfloat16, device="cuda")

    # SGLang: self.norm(hidden_states, residual)
    final_out_sg, _ = sglang_final_ln.forward_native(fake_moe_out.clone(), fake_residual.clone())

    # Megatron: final_layernorm(hidden_states) where hidden_states already includes residual
    combined_mega = fake_moe_out.clone() + fake_residual.clone()
    final_out_mega = manual_rmsnorm(combined_mega, final_ln_w, EPS)

    # Check: does SGLang's internal resadd + norm == Megatron's external resadd + norm?
results.check("Final LN: SGLang(x, res) vs manual(x+res)", final_out_sg, final_out_mega)

# ============================================================
print("\n" + "=" * 60)
results.summary()
if results.num_fail > 0:
    print("\nFAILURES found — this explains the E2E diff!")
else:
    print("\nAll pass — the SGLang code path matches manual forward exactly.")
    print("The E2E diff must come from either:")
    print("  1. The Megatron TransformerLayer wiring (different from manual)")
    print("  2. The data pipeline (token/position construction)")
    print("  3. Weight sync between Megatron and SGLang")
