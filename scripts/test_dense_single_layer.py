"""Norm-level comparison: Megatron SGLangRMSNorm vs SGLang RMSNorm.

Compares individual components to find exact divergence point.
Usage: PYTHONPATH=/root/Megatron-LM python3 scripts/test_dense_single_layer.py
"""
import os, sys, torch
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29501")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
sys.path.insert(0, "/root/Megatron-LM")

import torch.distributed as dist
if not dist.is_initialized():
    dist.init_process_group(backend="nccl", world_size=1, rank=0)
from megatron.core import parallel_state
if not parallel_state.model_parallel_is_initialized():
    parallel_state.initialize_model_parallel(1, 1)

torch.manual_seed(42); torch.cuda.manual_seed(42)
device = torch.device("cuda:0")

HIDDEN = 1024; HEAD_DIM = 128; EPS = 1e-6
MODEL_PATH = "/root/models/Qwen3-0.6B"

# ── Setup SGLang server args for on-policy mode ──
from sglang.srt.server_args import ServerArgs
import sglang.srt.model_executor.model_runner as mr
_fake_args = ServerArgs(
    model_path=MODEL_PATH, trust_remote_code=True, skip_server_warmup=True,
    rl_on_policy_target="fsdp", enable_deterministic_inference=True,
    attention_backend="fa3", sampling_backend="pytorch",
)
mr.global_server_args_dict = _fake_args.__dict__

# ── Build norms ──
from sglang.srt.layers.layernorm import RMSNorm as SGLangRMSNorm_orig
from megatron.core.extensions.sglang import SGLangRMSNorm, SGLangQKRMSNorm
from megatron.core.transformer.transformer_config import TransformerConfig

# SGLang layer norm (with on-policy kwargs)
sglang_ln = SGLangRMSNorm_orig(
    HIDDEN, eps=EPS,
    weight_dtype=torch.float32, cast_x_before_out_mul=True,
    override_orig_dtype=torch.float32, fp32_residual=True,
).to(device)

# SGLang QK norm (with on-policy kwargs — no override_orig_dtype, no fp32_residual)
sglang_qk = SGLangRMSNorm_orig(
    HEAD_DIM, eps=EPS,
    weight_dtype=torch.float32, cast_x_before_out_mul=True,
).to(device)

# Megatron config
meg_config = TransformerConfig(
    num_layers=1, hidden_size=HIDDEN, num_attention_heads=16,
    use_cpu_initialization=True, perform_initialization=False,
    bf16=True, params_dtype=torch.bfloat16, layernorm_epsilon=EPS,
    use_sglang=True, sglang_fp32_residual=True,
    init_model_with_meta_device=False,
)

# Megatron layer norm
meg_ln = SGLangRMSNorm(meg_config, HIDDEN, eps=EPS).to(device)
# Megatron QK norm
meg_qk = SGLangQKRMSNorm(meg_config, HEAD_DIM, eps=EPS).to(device)

# Load weights from checkpoint
from safetensors.torch import load_file
ckpt = load_file(os.path.join(MODEL_PATH, "model.safetensors"))
ln_w = ckpt["model.layers.0.input_layernorm.weight"].float().to(device)
qk_w = ckpt["model.layers.0.self_attn.q_norm.weight"].float().to(device)

sglang_ln.weight.data.copy_(ln_w)
meg_ln.weight.data.copy_(ln_w)
sglang_qk.weight.data.copy_(qk_w)
meg_qk.weight.data.copy_(qk_w)

def compare(name, a, b):
    d = (a.float() - b.float()).abs()
    match = "PASS" if d.max() == 0 else f"FAIL (max={d.max():.6e}, mean={d.mean():.6e})"
    print(f"  [{match}] {name}: SGLang dtype={a.dtype}, Megatron dtype={b.dtype}")
    return d.max().item()

print("=" * 70)
print("TEST 1: LayerNorm without residual (first layer)")
print("=" * 70)
x = torch.randn(12, HIDDEN, dtype=torch.bfloat16, device=device) * 0.1
s_out = sglang_ln.forward_native(x)
m_out = meg_ln(x)
compare("norm output", s_out, m_out)

print("\nTEST 2: LayerNorm with residual (fp32_residual=True)")
print("=" * 70)
x2 = torch.randn(12, HIDDEN, dtype=torch.bfloat16, device=device) * 0.1
res2 = torch.randn(12, HIDDEN, dtype=torch.bfloat16, device=device) * 0.05
s_out2, s_res2 = sglang_ln.forward_native(x2, res2)
m_out2, m_res2 = meg_ln(x2, res2)
compare("norm output", s_out2, m_out2)
compare("residual", s_res2, m_res2)

print("\nTEST 3: QK norm (no residual)")
print("=" * 70)
q = torch.randn(12, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.1
s_qk = sglang_qk.forward_native(q)
m_qk = meg_qk(q)
compare("QK norm output", s_qk, m_qk)

print("\nTEST 4: QK norm output dtype check")
print("=" * 70)
print(f"  SGLang QK norm output dtype: {s_qk.dtype}")
print(f"  Megatron QK norm output dtype: {m_qk.dtype}")
print(f"  SGLang LN output dtype: {s_out.dtype}")
print(f"  Megatron LN output dtype: {m_out.dtype}")

print("\nTEST 5: Full fused LN+Linear comparison")
print("=" * 70)
from megatron.core.extensions.sglang import SGLangLayerNormColumnParallelLinear

q_w = ckpt["model.layers.0.self_attn.q_proj.weight"].to(torch.bfloat16).to(device)
k_w = ckpt["model.layers.0.self_attn.k_proj.weight"].to(torch.bfloat16).to(device)
v_w = ckpt["model.layers.0.self_attn.v_proj.weight"].to(torch.bfloat16).to(device)
qkv_w = torch.cat([q_w, k_w, v_w], dim=0)

meg_fused = SGLangLayerNormColumnParallelLinear(
    HIDDEN, qkv_w.shape[0],
    config=meg_config, init_method=lambda x: x,
    gather_output=False, bias=False,
).to(device)
meg_fused.norm.weight.data.copy_(ln_w)
meg_fused.linear.weight.data.copy_(qkv_w)

x5 = torch.randn(12, HIDDEN, dtype=torch.bfloat16, device=device) * 0.1
m_qkv_out, _ = meg_fused(x5)

s_normed = sglang_ln.forward_native(x5)
# SGLang uses F.linear(input, weight) which does type promotion: fp32 @ bf16 -> fp32
# Megatron's SGLangLinear now also preserves input dtype (no forced bf16 cast)
s_qkv_out = torch.nn.functional.linear(s_normed.view(-1, HIDDEN), qkv_w)
compare("fused LN+Linear (no residual)", s_qkv_out, m_qkv_out)
print(f"  SGLang output dtype: {s_qkv_out.dtype}, Megatron output dtype: {m_qkv_out.dtype}")

x5b = torch.randn(12, HIDDEN, dtype=torch.bfloat16, device=device) * 0.1
res5 = torch.randn(12, HIDDEN, dtype=torch.bfloat16, device=device) * 0.05
meg_fused._sglang_pending_residual = res5
m_qkv_out2, _ = meg_fused(x5b)
m_res_out = meg_fused._sglang_last_residual

s_normed2, s_res_out = sglang_ln.forward_native(x5b, res5)
s_qkv_out2 = torch.nn.functional.linear(s_normed2.view(-1, HIDDEN), qkv_w)
compare("fused LN+Linear (with residual)", s_qkv_out2, m_qkv_out2)
compare("residual after fused", s_res_out, m_res_out)

print("\n" + "=" * 70)
print("ALL TESTS COMPLETE")
print("=" * 70)

dist.destroy_process_group()
