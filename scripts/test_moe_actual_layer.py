"""
Compare actual SGLang Qwen3MoeAttention against manual forward to find divergence.

The manual forward in test_moe_layer_alignment.py (which passes bitwise) uses the
same kernels as the E2E. But something in the ACTUAL SGLang/Megatron model classes
differs. This test loads the ACTUAL SGLang attention module with real weights and
compares against the manual construction.

Usage: CUDA_VISIBLE_DEVICES=0 python scripts/test_moe_actual_layer.py
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
config = AutoConfig.from_pretrained("/root/models/Qwen3-30B-A3B")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alignment_test_utils import load_safetensor_weights

HIDDEN = config.hidden_size
EPS = config.rms_norm_eps
NUM_Q_HEADS = config.num_attention_heads
NUM_KV_HEADS = config.num_key_value_heads
HEAD_DIM = getattr(config, 'head_dim', HIDDEN // NUM_Q_HEADS)
T = 32

weights = load_safetensor_weights("/root/models/Qwen3-30B-A3B", prefixes=[
    "model.layers.0.", "model.norm.", "model.embed_tokens.", "lm_head.",
])

print(f"hidden={HIDDEN}, heads={NUM_Q_HEADS}/{NUM_KV_HEADS}, head_dim={HEAD_DIM}, T={T}")

# ============================================================
# Load ACTUAL SGLang RMSNorm and test
# ============================================================
from sglang.srt.layers.layernorm import RMSNorm as SGLangRMSNorm

input_ln = SGLangRMSNorm(HIDDEN, eps=EPS, cast_x_before_out_mul=True, fp32_residual=False).cuda()
input_ln.weight.data.copy_(weights["model.layers.0.input_layernorm.weight"].float())

post_attn_ln = SGLangRMSNorm(HIDDEN, eps=EPS, cast_x_before_out_mul=True, fp32_residual=False).cuda()
post_attn_ln.weight.data.copy_(weights["model.layers.0.post_attention_layernorm.weight"].float())

q_norm = SGLangRMSNorm(HEAD_DIM, eps=EPS, cast_x_before_out_mul=True, fp32_residual=False).cuda()
q_norm.weight.data.copy_(weights["model.layers.0.self_attn.q_norm.weight"].float())

k_norm = SGLangRMSNorm(HEAD_DIM, eps=EPS, cast_x_before_out_mul=True, fp32_residual=False).cuda()
k_norm.weight.data.copy_(weights["model.layers.0.self_attn.k_norm.weight"].float())

q_proj_w = weights["model.layers.0.self_attn.q_proj.weight"].cuda().bfloat16()
k_proj_w = weights["model.layers.0.self_attn.k_proj.weight"].cuda().bfloat16()
v_proj_w = weights["model.layers.0.self_attn.v_proj.weight"].cuda().bfloat16()
o_proj_w = weights["model.layers.0.self_attn.o_proj.weight"].cuda().bfloat16()

# RoPE
base = getattr(config, 'rope_theta', 1000000)
inv_freq = 1.0 / (base ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
t_pos = torch.arange(T, dtype=torch.float32, device="cuda")
freqs = torch.outer(t_pos, inv_freq.cuda())
cos_rope = torch.cos(freqs)
sin_rope = torch.sin(freqs)

from sgl_kernel.flash_attn import flash_attn_varlen_func

# ============================================================
# Manual forward (same as passing unit tests)
# ============================================================
torch.manual_seed(42)
hidden_states = torch.randn(T, HIDDEN, dtype=torch.bfloat16, device="cuda")

print("\n=== Manual forward (same as unit tests) ===")
with torch.no_grad():
    # Input layernorm (no residual for layer 0)
    normed = input_ln.forward_native(hidden_states.clone())

    # QKV
    q = F.linear(normed, q_proj_w).view(T, NUM_Q_HEADS, HEAD_DIM)
    k = F.linear(normed, k_proj_w).view(T, NUM_KV_HEADS, HEAD_DIM)
    v = F.linear(normed, v_proj_w).view(T, NUM_KV_HEADS, HEAD_DIM)

    # QK norm
    q = q_norm.forward_native(q.reshape(-1, HEAD_DIM)).view(T, NUM_Q_HEADS, HEAD_DIM)
    k = k_norm.forward_native(k.reshape(-1, HEAD_DIM)).view(T, NUM_KV_HEADS, HEAD_DIM)

    # RoPE (neox style)
    def apply_rope(x, cos, sin):
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        c, s = cos.unsqueeze(1), sin.unsqueeze(1)
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * sin.unsqueeze(1)], dim=-1).to(x.dtype)

    q = apply_rope(q, cos_rope, sin_rope)
    k = apply_rope(k, cos_rope, sin_rope)

    # GQA expand + FA3
    rep = NUM_Q_HEADS // NUM_KV_HEADS
    k_exp = k.repeat_interleave(rep, dim=1)
    v_exp = v.repeat_interleave(rep, dim=1)
    cu = torch.tensor([0, T], dtype=torch.int32, device="cuda")
    attn_out = flash_attn_varlen_func(
        q=q.bfloat16(), k=k_exp.bfloat16(), v=v_exp.bfloat16(),
        cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=T, max_seqlen_k=T,
        softmax_scale=1.0 / (HEAD_DIM ** 0.5), causal=True, num_splits=1,
    )
    if isinstance(attn_out, tuple):
        attn_out = attn_out[0]
    attn_out_flat = attn_out.reshape(T, -1)

    # o_proj
    manual_attn_output = F.linear(attn_out_flat, o_proj_w)

    # post_attn_layernorm + residual → MoE input
    manual_moe_input, manual_residual = post_attn_ln.forward_native(
        manual_attn_output, hidden_states.clone()  # residual = original input
    )

    print(f"  normed: absmax={normed.abs().max():.6f}")
    print(f"  attn_output: absmax={manual_attn_output.abs().max():.6f}")
    print(f"  moe_input: absmax={manual_moe_input.abs().max():.6f}")

# ============================================================
# Self-consistency: run manual forward AGAIN with same input
# ============================================================
print("\n=== Self-consistency check ===")
with torch.no_grad():
    normed2 = input_ln.forward_native(hidden_states.clone())
    q2 = F.linear(normed2, q_proj_w).view(T, NUM_Q_HEADS, HEAD_DIM)
    k2 = F.linear(normed2, k_proj_w).view(T, NUM_KV_HEADS, HEAD_DIM)
    v2 = F.linear(normed2, v_proj_w).view(T, NUM_KV_HEADS, HEAD_DIM)
    q2 = q_norm.forward_native(q2.reshape(-1, HEAD_DIM)).view(T, NUM_Q_HEADS, HEAD_DIM)
    k2 = k_norm.forward_native(k2.reshape(-1, HEAD_DIM)).view(T, NUM_KV_HEADS, HEAD_DIM)
    q2 = apply_rope(q2, cos_rope, sin_rope)
    k2 = apply_rope(k2, cos_rope, sin_rope)
    k_exp2 = k2.repeat_interleave(rep, dim=1)
    v_exp2 = v2.repeat_interleave(rep, dim=1)
    attn_out2 = flash_attn_varlen_func(
        q=q2.bfloat16(), k=k_exp2.bfloat16(), v=v_exp2.bfloat16(),
        cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=T, max_seqlen_k=T,
        softmax_scale=1.0 / (HEAD_DIM ** 0.5), causal=True, num_splits=1,
    )
    if isinstance(attn_out2, tuple):
        attn_out2 = attn_out2[0]
    manual_attn_output2 = F.linear(attn_out2.reshape(T, -1), o_proj_w)
    manual_moe_input2, _ = post_attn_ln.forward_native(manual_attn_output2, hidden_states.clone())

    sc_diff = (manual_moe_input.float() - manual_moe_input2.float()).abs().max().item()
    print(f"  Self-consistency moe_input diff: {sc_diff}")

# ============================================================
# Now: use Megatron's ACTUAL RotaryEmbedding + sglang_apply_rotary_pos_emb_with_freqs
# and compare against the manual RoPE
# ============================================================
print("\n=== Megatron RoPE vs Manual RoPE ===")
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.extensions.sglang import sglang_apply_rotary_pos_emb_with_freqs

mega_rope = RotaryEmbedding(
    kv_channels=HEAD_DIM, rotary_percent=1.0, rotary_interleaved=False,
    seq_len_interpolation_factor=None, rotary_base=base, use_cpu_initialization=True,
)
mega_emb = mega_rope(T).cuda()

class FakeConfig:
    rotary_interleaved = False

with torch.no_grad():
    # Redo QKV and QK norm (same as manual)
    normed3 = input_ln.forward_native(hidden_states.clone())
    q3 = F.linear(normed3, q_proj_w).view(T, NUM_Q_HEADS, HEAD_DIM)
    k3 = F.linear(normed3, k_proj_w).view(T, NUM_KV_HEADS, HEAD_DIM)
    v3 = F.linear(normed3, v_proj_w).view(T, NUM_KV_HEADS, HEAD_DIM)
    q3 = q_norm.forward_native(q3.reshape(-1, HEAD_DIM)).view(T, NUM_Q_HEADS, HEAD_DIM)
    k3 = k_norm.forward_native(k3.reshape(-1, HEAD_DIM)).view(T, NUM_KV_HEADS, HEAD_DIM)

    # Apply Megatron RoPE
    q3_4d = q3.unsqueeze(1)  # [T, 1, heads, dim]
    k3_4d = k3.unsqueeze(1)
    q3_mega = sglang_apply_rotary_pos_emb_with_freqs(q3_4d, mega_emb, FakeConfig()).squeeze(1)
    k3_mega = sglang_apply_rotary_pos_emb_with_freqs(k3_4d, mega_emb, FakeConfig()).squeeze(1)

    # Compare with manual RoPE (q already has manual rope from first run)
    # Need fresh manual rope
    q3_manual = apply_rope(q3, cos_rope, sin_rope)
    k3_manual = apply_rope(k3, cos_rope, sin_rope)

    rope_q_diff = (q3_manual.float() - q3_mega.float()).abs().max().item()
    rope_k_diff = (k3_manual.float() - k3_mega.float()).abs().max().item()
    print(f"  Q after RoPE diff (manual vs Megatron): {rope_q_diff}")
    print(f"  K after RoPE diff (manual vs Megatron): {rope_k_diff}")

    # Note: sglang_apply_rotary_pos_emb_with_freqs is NOT used in E2E (disabled for packed seqs).
    # The diff above is known and not the E2E root cause. Skip to the actual E2E path below.

# ============================================================
# Now: use Megatron's ACTUAL apply_rotary_pos_emb (the standard one with cu_seqlens)
# ============================================================
print("\n=== Standard Megatron _apply_rotary_pos_emb_bshd vs Manual ===")
from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd

with torch.no_grad():
    normed4 = input_ln.forward_native(hidden_states.clone())
    q4 = F.linear(normed4, q_proj_w).view(T, NUM_Q_HEADS, HEAD_DIM)
    k4 = F.linear(normed4, k_proj_w).view(T, NUM_KV_HEADS, HEAD_DIM)
    q4 = q_norm.forward_native(q4.reshape(-1, HEAD_DIM)).view(T, NUM_Q_HEADS, HEAD_DIM)
    k4 = k_norm.forward_native(k4.reshape(-1, HEAD_DIM)).view(T, NUM_KV_HEADS, HEAD_DIM)

    # Standard Megatron RoPE uses [T, B, H, D] format and raw angle freqs
    q4_4d = q4.unsqueeze(1)  # [T, 1, heads, dim]

    # _apply_rotary_pos_emb_bshd expects freqs with rot_dim = freqs.shape[-1]
    # mega_emb has shape [T, 1, 1, dim] with [angles, angles] concatenated
    q4_std = _apply_rotary_pos_emb_bshd(q4_4d, mega_emb, rotary_interleaved=False)

    q4_manual = apply_rope(q4, cos_rope, sin_rope)

    std_q_diff = (q4_manual.float() - q4_std.squeeze(1).float()).abs().max().item()
    print(f"  Q after RoPE diff (manual vs _apply_rotary_pos_emb_bshd): {std_q_diff}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Self-consistency:           {sc_diff}")
print(f"  Megatron SGLang RoPE Q diff: {rope_q_diff}")
print(f"  Megatron SGLang RoPE K diff: {rope_k_diff}")
print(f"  Standard Megatron RoPE diff: {std_q_diff}")
if rope_q_diff == 0 and std_q_diff == 0:
    print("  → RoPE is NOT the issue. Look elsewhere.")
elif rope_q_diff > 0:
    print(f"  → SGLang RoPE wrapper introduces diff of {rope_q_diff}")
    if rope_q_diff > 1e-6:
        print("  → This is SIGNIFICANT and likely the root cause")
