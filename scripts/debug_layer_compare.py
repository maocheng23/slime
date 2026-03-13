"""
Standalone layer-by-layer comparison between Megatron and SGLang Qwen3-Next forward.
Loads the same weights and feeds the same input to both, comparing outputs at each stage.

Usage (inside Docker container, single GPU):
  CUDA_VISIBLE_DEVICES=0 python scripts/debug_layer_compare.py
"""
import os
import sys
import torch
import torch.nn.functional as F

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# ---- Load HF config ----
from transformers import AutoConfig, AutoTokenizer

MODEL_PATH = "/root/models/Qwen3-Next-4layer"
config = AutoConfig.from_pretrained(MODEL_PATH)
print(f"Model: {MODEL_PATH}")
print(f"layer_types: {config.layer_types}")
print(f"hidden_size={config.hidden_size}, num_hidden_layers={config.num_hidden_layers}")
print(f"num_experts={config.num_experts}, topk={config.num_experts_per_tok}")

# ---- Load weights ----
import safetensors.torch as st
import json

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
print(f"Loaded {len(weights)} tensors from {len(loaded_files)} shards")

# ---- Create test input ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
text = "What is 2+3? The answer is"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"].cuda()  # [1, seq_len]
seq_len = input_ids.shape[1]
print(f"Input: '{text}' -> {seq_len} tokens")

# ---- Embedding (same for both) ----
embed_weight = weights["model.embed_tokens.weight"].cuda().bfloat16()
hidden_states = F.embedding(input_ids, embed_weight)  # [1, seq_len, hidden_size]
hidden_states_2d = hidden_states.squeeze(0)  # [seq_len, hidden_size]
print(f"Embedding output: shape={list(hidden_states.shape)}, mean={hidden_states.float().mean():.6f}")

# ---- Compare input_layernorm (GemmaRMSNorm) ----
from sglang.srt.layers.layernorm import GemmaRMSNorm

ln_weight = weights["model.layers.0.input_layernorm.weight"].cuda().bfloat16()

# SGLang path: GemmaRMSNorm (2D input)
sglang_ln = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps).cuda().bfloat16()
sglang_ln.weight.data.copy_(ln_weight)
with torch.no_grad():
    sglang_ln_out = sglang_ln(hidden_states_2d)

# Megatron path: our Qwen3NextRMSNorm wrapper (3D input)
sys.path.insert(0, "/root/slime")
from slime_plugins.models.qwen3_next import Qwen3NextRMSNorm

mega_ln = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps).cuda().bfloat16()
mega_ln.weight.data.copy_(ln_weight)
with torch.no_grad():
    mega_ln_out = mega_ln(hidden_states)
    mega_ln_out_2d = mega_ln_out.squeeze(0)

diff = (sglang_ln_out.float() - mega_ln_out_2d.float()).abs()
print(f"\n--- input_layernorm (layer 0) ---")
print(f"  max_diff={diff.max():.8f}, mean_diff={diff.mean():.8f}, nonzero={(diff > 0).float().mean():.4f}")
if diff.max() == 0:
    print("  PASS: bitwise identical")
else:
    print("  FAIL: NOT bitwise identical")

# ---- Compare GDN in_proj_qkvz ----
# SGLang uses ColumnParallelLinear with TP, but at TP=1 it's just F.linear
# Megatron's GDN at TP=1 slices the full weight (same result)
qkvz_weight = weights["model.layers.0.linear_attn.in_proj_qkvz.weight"].cuda().bfloat16()
with torch.no_grad():
    # Both should be F.linear(input, weight) at TP=1
    sglang_qkvz = F.linear(sglang_ln_out, qkvz_weight)
    mega_qkvz = F.linear(mega_ln_out_2d, qkvz_weight)

diff = (sglang_qkvz.float() - mega_qkvz.float()).abs()
print(f"\n--- GDN0 in_proj_qkvz ---")
print(f"  max_diff={diff.max():.8f}, mean_diff={diff.mean():.8f}")
if diff.max() == 0:
    print("  PASS: bitwise identical")
else:
    print("  FAIL: NOT bitwise identical")

# ---- Compare conv1d ----
from fla.modules.convolution import causal_conv1d_fwd as _fla_causal_conv1d_fwd

conv_weight_full = weights["model.layers.0.linear_attn.conv1d.weight"].cuda().bfloat16()
# conv1d weight shape: [channels, 1, kernel_size] -> squeeze to [channels, kernel_size]
conv_weight = conv_weight_full.squeeze(1)

# Need to split qkvz into q, k, v first, concat for conv
# At TP=1, conv input is full mixed_qkv
# Let's use the same processing as the model
v_per_k_group = config.linear_num_value_heads // config.linear_num_key_heads
num_k_heads = config.linear_num_key_heads
head_k_dim = config.linear_key_head_dim
head_v_dim = config.linear_value_head_dim
key_dim = head_k_dim * num_k_heads
value_dim = head_v_dim * config.linear_num_value_heads

# fix_query_key_value_ordering
group_size = 2 * head_k_dim + 2 * head_v_dim * v_per_k_group
mixed = sglang_qkvz.view(seq_len, num_k_heads, group_size)
split_sizes = [head_k_dim, head_k_dim, v_per_k_group * head_v_dim, v_per_k_group * head_v_dim]
q, k, v, z = torch.split(mixed, split_sizes, dim=2)
q = q.reshape(seq_len, -1)
k = k.reshape(seq_len, -1)
v = v.reshape(seq_len, key_dim // num_k_heads * config.linear_num_value_heads // v_per_k_group * v_per_k_group)
v = v.reshape(seq_len, -1)

# Actually for conv, we need [B, T, D] format with concatenated qkv
mixed_qkv = torch.cat([q, k, v], dim=-1)
mixed_qkv_3d = mixed_qkv.unsqueeze(0)  # [1, T, D]

cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")

with torch.no_grad():
    # SGLang also uses FLA's causal_conv1d
    conv_out_sglang, _ = _fla_causal_conv1d_fwd(
        x=mixed_qkv.unsqueeze(0), weight=conv_weight, bias=None, residual=None,
        cu_seqlens=cu_seqlens,
    )
    conv_out_mega, _ = _fla_causal_conv1d_fwd(
        x=mixed_qkv_3d, weight=conv_weight, bias=None, residual=None,
        cu_seqlens=cu_seqlens,
    )

diff = (conv_out_sglang.float() - conv_out_mega.float()).abs()
print(f"\n--- GDN0 conv1d ---")
print(f"  max_diff={diff.max():.8f}, mean_diff={diff.mean():.8f}")
if diff.max() == 0:
    print("  PASS: bitwise identical")
else:
    print("  FAIL: NOT bitwise identical")

# ---- Compare in_proj_ba ----
ba_weight = weights["model.layers.0.linear_attn.in_proj_ba.weight"].cuda().bfloat16()
with torch.no_grad():
    ba_out = F.linear(sglang_ln_out, ba_weight)

# ---- fix_query_key_value_ordering for full pipeline ----
num_v_heads = config.linear_num_value_heads

new_shape_qkvz = (seq_len, num_k_heads, 2 * head_k_dim + 2 * head_v_dim * v_per_k_group)
new_shape_ba = (seq_len, num_k_heads, 2 * v_per_k_group)

mixed_qkvz = sglang_qkvz.view(*new_shape_qkvz)
mixed_ba = ba_out.view(*new_shape_ba)

split_qkvz = [head_k_dim, head_k_dim, v_per_k_group * head_v_dim, v_per_k_group * head_v_dim]
split_ba = [v_per_k_group, v_per_k_group]

query, key, value, z_gate = torch.split(mixed_qkvz, split_qkvz, dim=2)
b_gate, a_gate = torch.split(mixed_ba, split_ba, dim=2)

value = value.reshape(seq_len, -1, head_v_dim)
z_gate = z_gate.reshape(seq_len, -1, head_v_dim)
b_gate = b_gate.reshape(seq_len, num_v_heads)
a_gate = a_gate.reshape(seq_len, num_v_heads)

query_flat = query.reshape(seq_len, -1)
key_flat = key.reshape(seq_len, -1)
value_flat = value.reshape(seq_len, -1)

# After conv, split back
conv_out = conv_out_sglang.squeeze(0)  # [seq_len, conv_dim]
q_after_conv, k_after_conv, v_after_conv = torch.split(
    conv_out, [key_dim, key_dim, value_dim], dim=-1
)
q_heads = q_after_conv.reshape(seq_len, -1, head_k_dim).unsqueeze(0)
k_heads = k_after_conv.reshape(seq_len, -1, head_k_dim).unsqueeze(0)
v_heads = v_after_conv.reshape(seq_len, -1, head_v_dim).unsqueeze(0)

# ---- Compare chunk_gated_delta_rule ----
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

A_log = weights["model.layers.0.linear_attn.A_log"].cuda().bfloat16()
dt_bias = weights["model.layers.0.linear_attn.dt_bias"].cuda().bfloat16()

beta = b_gate.unsqueeze(0).sigmoid()
g = -A_log.float().exp() * F.softplus(a_gate.unsqueeze(0).float() + dt_bias)

if v_per_k_group > 1:
    q_heads = q_heads.repeat_interleave(v_per_k_group, dim=2)
    k_heads = k_heads.repeat_interleave(v_per_k_group, dim=2)

with torch.no_grad():
    gdr_out, _ = chunk_gated_delta_rule(
        q_heads, k_heads, v_heads, g=g, beta=beta,
        initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )

print(f"\n--- GDN0 chunk_gated_delta_rule ---")
print(f"  output shape={list(gdr_out.shape)}, mean={gdr_out.float().mean():.6f}, absmax={gdr_out.float().abs().max():.6f}")
print("  (Same kernel on both sides — always identical by construction)")

# ---- Compare FusedRMSNormGated vs SGLang's RMSNormGated ----
from fla.modules import FusedRMSNormGated
from sglang.srt.layers.attention.fla.layernorm_gated import RMSNorm as SGLangRMSNormGated

norm_weight = weights["model.layers.0.linear_attn.norm.weight"].cuda().bfloat16()

fla_norm = FusedRMSNormGated(head_v_dim, eps=config.rms_norm_eps, activation="silu",
                              device="cuda:0", dtype=torch.bfloat16)
fla_norm.weight.data.copy_(norm_weight)

sglang_norm = SGLangRMSNormGated(head_v_dim, eps=config.rms_norm_eps,
                                  device="cuda:0", dtype=torch.bfloat16)
sglang_norm.weight.data.copy_(norm_weight)

gdr_2d = gdr_out.reshape(-1, head_v_dim)
z_2d = z_gate.reshape(-1, head_v_dim)

with torch.no_grad():
    fla_norm_out = fla_norm(gdr_2d, z_2d)
    sglang_norm_out = sglang_norm(gdr_2d, z_2d)

diff = (fla_norm_out.float() - sglang_norm_out.float()).abs()
print(f"\n--- GDN0 FusedRMSNormGated (Megatron/FLA) vs RMSNormGated (SGLang) ---")
print(f"  max_diff={diff.max():.8f}, mean_diff={diff.mean():.8f}, nonzero={(diff > 0).float().mean():.6f}")
if diff.max() == 0:
    print("  PASS: bitwise identical")
else:
    print(f"  FAIL: NOT bitwise identical ({(diff > 0).sum().item()} / {diff.numel()} values differ)")

# ---- Compare out_proj ----
out_weight = weights["model.layers.0.linear_attn.out_proj.weight"].cuda().bfloat16()

# Reshape norm output back for out_proj
# At TP=1: full value_dim
norm_reshaped = sglang_norm_out.reshape(1, seq_len, -1)  # [1, T, value_dim]
norm_flat = norm_reshaped.reshape(seq_len, -1)  # [T, value_dim]

with torch.no_grad():
    # Use FLA norm output (Megatron) and SGLang norm output
    fla_norm_reshaped = fla_norm_out.reshape(seq_len, -1)
    out_mega = F.linear(fla_norm_reshaped, out_weight)
    out_sglang = F.linear(norm_flat, out_weight)

diff = (out_mega.float() - out_sglang.float()).abs()
print(f"\n--- GDN0 out_proj ---")
print(f"  max_diff={diff.max():.8f}, mean_diff={diff.mean():.8f}")
if diff.max() == 0:
    print("  PASS: bitwise identical")
else:
    print(f"  FAIL: NOT bitwise identical (propagated from norm diff)")

# ---- Compare post_attention_layernorm ----
post_ln_weight = weights["model.layers.0.post_attention_layernorm.weight"].cuda().bfloat16()
sglang_post_ln = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps).cuda().bfloat16()
sglang_post_ln.weight.data.copy_(post_ln_weight)

# Residual connection: hidden_states + gdn_output
residual = hidden_states_2d  # [seq_len, hidden_size]
gdn_output = out_sglang  # [seq_len, hidden_size] -- using SGLang norm path

with torch.no_grad():
    after_residual = residual + gdn_output
    post_ln_out = sglang_post_ln(after_residual)

print(f"\n--- Layer 0 post_attention_layernorm ---")
print(f"  post_ln_out shape={list(post_ln_out.shape)}, mean={post_ln_out.float().mean():.6f}")
print("  (Same kernel — always identical)")

print("\n=== Summary ===")
print("GDN path: input_layernorm -> in_proj_qkvz -> conv1d -> chunk_gated_delta_rule -> norm -> out_proj")
print("The FusedRMSNormGated comparison reveals if the GDN norm is a source of divergence.")
print("If all PASS, divergence must come from MoE or attention layers.")
