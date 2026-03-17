"""
Compare ACTUAL Megatron TransformerLayer vs manual forward at TP=8.

All kernel-level tests pass bitwise, but E2E shows ~1.78e-05.
This test instantiates the real Megatron TransformerLayer class with
SGLang spec and compares against the manual forward from test_moe_tp8_alignment.py.

Phase A: Attention path only (no MoE, tests residual+LN+QKV+FA+o_proj+BDA)
Phase B: Full layer with MoE (if Phase A passes)

Usage: torchrun --nproc_per_node=8 scripts/test_moe_tp8_layer_vs_manual.py
"""
import os
import sys
import torch
import torch.distributed as dist
import torch.nn.functional as F

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ.get("LOCAL_RANK", rank))
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")
assert world_size == 8

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alignment_test_utils import TPTestResults, load_safetensor_weights

os.environ["MEGATRON_USE_DETERMINISTIC_ALLREDUCE"] = "1"

from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

MODEL_PATH = "/root/models/Qwen3-30B-A3B"
server_args = ServerArgs(model_path=MODEL_PATH, rl_on_policy_target="fsdp_tp")
set_global_server_args_for_scheduler(server_args)
enable_batch_invariant_mode(enable_bmm=False)

# Initialize Megatron parallel state
import megatron.core.parallel_state as mpu
mpu.initialize_model_parallel(
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=1,
)

from transformers import AutoConfig
config = AutoConfig.from_pretrained(MODEL_PATH)

HIDDEN = config.hidden_size
EPS = config.rms_norm_eps
NUM_Q_HEADS = config.num_attention_heads
NUM_KV_HEADS = config.num_key_value_heads
HEAD_DIM = getattr(config, 'head_dim', HIDDEN // NUM_Q_HEADS)
NUM_EXPERTS = config.num_experts
TOPK = config.num_experts_per_tok
TP = world_size
EP = world_size
Q_PER_RANK = NUM_Q_HEADS // TP
B, T = 1, 16

if rank == 0:
    print(f"hidden={HIDDEN}, heads={NUM_Q_HEADS}/{NUM_KV_HEADS}, head_dim={HEAD_DIM}")
    print(f"TP={TP}, EP={EP}, experts={NUM_EXPERTS}, topk={TOPK}, T={T}")

weights = load_safetensor_weights(MODEL_PATH, prefixes=[
    "model.layers.0.", "model.norm.", "model.embed_tokens.", "lm_head.",
])

results = TPTestResults(rank)
tp_group = dist.group.WORLD

def make_shared_input(*shape, dtype=torch.bfloat16):
    x = torch.randn(*shape, device=device, dtype=dtype)
    dist.broadcast(x, src=0, group=tp_group)
    return x

# ============================================================
# Build the ACTUAL Megatron TransformerLayer
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("Building actual Megatron TransformerLayer")
    print("=" * 60)

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec

mega_config = TransformerConfig(
    num_layers=48,
    hidden_size=HIDDEN,
    num_attention_heads=NUM_Q_HEADS,
    num_query_groups=NUM_KV_HEADS,
    kv_channels=HEAD_DIM,
    ffn_hidden_size=config.intermediate_size,
    hidden_dropout=0.0,
    attention_dropout=0.0,
    normalization="RMSNorm",
    layernorm_epsilon=EPS,
    add_bias_linear=False,
    tensor_model_parallel_size=TP,
    pipeline_model_parallel_size=1,
    # Dense MLP (no MoE) for Phase 1
    num_moe_experts=None,
    use_sglang=True,
    use_sglang_attention=True,
    qk_layernorm=True,
    fp32_residual_connection=False,
    bias_dropout_fusion=False,
    apply_rope_fusion=False,
    use_cpu_initialization=True,
    perform_initialization=False,
    bf16=True,
    params_dtype=torch.bfloat16,
    pipeline_dtype=torch.bfloat16,
)

# Get DENSE layer spec (no MoE) — tests attention + residual + LN wiring
layer_spec = get_gpt_layer_with_transformer_engine_spec(
    num_experts=None,
    moe_grouped_gemm=False,
    qk_layernorm=True,
    use_sglang=True,
    use_sglang_attention=True,
)

if rank == 0:
    print(f"  Layer spec built successfully")
    print(f"  Config: use_sglang={mega_config.use_sglang}, use_sglang_attention={mega_config.use_sglang_attention}")

# Instantiate the layer
try:
    megatron_layer = TransformerLayer(
        config=mega_config,
        submodules=layer_spec.submodules,
        layer_number=1,  # 1-based in Megatron
    )
    megatron_layer = megatron_layer.cuda().bfloat16()
    megatron_layer.eval()
    if rank == 0:
        print(f"  TransformerLayer instantiated successfully")
        # Print module structure
        for name, mod in megatron_layer.named_modules():
            if name and '.' not in name:
                print(f"    {name}: {type(mod).__name__}")
except Exception as e:
    if rank == 0:
        print(f"  FAILED to instantiate TransformerLayer: {e}")
        import traceback
        traceback.print_exc()
    dist.barrier()
    sys.exit(1)

# ============================================================
# Load weights into Megatron layer
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("Loading weights into Megatron TransformerLayer")
    print("=" * 60)

with torch.no_grad():
    # Input layernorm (fused into linear_qkv for SGLang spec)
    if hasattr(megatron_layer.self_attention.linear_qkv, 'norm'):
        megatron_layer.self_attention.linear_qkv.norm.weight.data.copy_(
            weights["model.layers.0.input_layernorm.weight"].float().to(device)
        )
        if rank == 0:
            print("  ✓ input_layernorm (in linear_qkv.norm)")

    # QKV weights: need to build the fused+sharded weight
    q_w = weights["model.layers.0.self_attn.q_proj.weight"].to(device).bfloat16()
    k_w = weights["model.layers.0.self_attn.k_proj.weight"].to(device).bfloat16()
    v_w = weights["model.layers.0.self_attn.v_proj.weight"].to(device).bfloat16()

    # For TP-sharded QKV with num_kv_heads < TP, Megatron interleaves:
    # [Q_group0, K_group0, V_group0, Q_group1, ...]
    q_per_group = NUM_Q_HEADS // NUM_KV_HEADS  # 8
    parts = []
    for g in range(NUM_KV_HEADS):
        parts.append(q_w[g * q_per_group * HEAD_DIM : (g + 1) * q_per_group * HEAD_DIM])
        parts.append(k_w[g * HEAD_DIM : (g + 1) * HEAD_DIM])
        parts.append(v_w[g * HEAD_DIM : (g + 1) * HEAD_DIM])
    qkv_full = torch.cat(parts, dim=0)  # [5120, 2048]

    # TP shard
    shard_size = qkv_full.shape[0] // TP
    qkv_local = qkv_full[rank * shard_size : (rank + 1) * shard_size]

    if hasattr(megatron_layer.self_attention.linear_qkv, 'linear'):
        megatron_layer.self_attention.linear_qkv.linear.weight.data.copy_(qkv_local)
    else:
        megatron_layer.self_attention.linear_qkv.weight.data.copy_(qkv_local)
    if rank == 0:
        print(f"  ✓ QKV weight shard: {qkv_local.shape}")

    # QK layernorm
    if megatron_layer.self_attention.q_layernorm is not None:
        megatron_layer.self_attention.q_layernorm.weight.data.copy_(
            weights["model.layers.0.self_attn.q_norm.weight"].float().to(device)
        )
    if megatron_layer.self_attention.k_layernorm is not None:
        megatron_layer.self_attention.k_layernorm.weight.data.copy_(
            weights["model.layers.0.self_attn.k_norm.weight"].float().to(device)
        )
    if rank == 0:
        print("  ✓ QK layernorm")

    # o_proj (TP-sharded along input dim)
    o_w = weights["model.layers.0.self_attn.o_proj.weight"].to(device).bfloat16()
    o_shard_size = o_w.shape[1] // TP
    o_local = o_w[:, rank * o_shard_size : (rank + 1) * o_shard_size]
    megatron_layer.self_attention.linear_proj.weight.data.copy_(o_local)
    if rank == 0:
        print(f"  ✓ o_proj weight shard: {o_local.shape}")

    # pre_mlp_layernorm (= post_attention_layernorm in HF)
    # For dense layer, this is inside linear_fc1 (fused LN+Linear) or separate
    if hasattr(megatron_layer, 'pre_mlp_layernorm') and megatron_layer.pre_mlp_layernorm is not None:
        try:
            megatron_layer.pre_mlp_layernorm.weight.data.copy_(
                weights["model.layers.0.post_attention_layernorm.weight"].float().to(device)
            )
            if rank == 0:
                print("  ✓ pre_mlp_layernorm")
        except Exception as e:
            if rank == 0:
                print(f"  ⚠ pre_mlp_layernorm: {e}")

    # Dense MLP weights (gate_proj, up_proj, down_proj → fc1, fc2)
    # For SGLang dense spec: fc1 = SGLangLayerNormColumnParallelLinear (fused LN + linear)
    # fc2 = SGLangRowParallelLinear
    try:
        mlp = megatron_layer.mlp
        # fc1 includes post_attention_layernorm + gate_up projection
        if hasattr(mlp, 'linear_fc1'):
            fc1 = mlp.linear_fc1
            # Load LN weight into fc1.norm (if fused)
            if hasattr(fc1, 'norm'):
                fc1.norm.weight.data.copy_(
                    weights["model.layers.0.post_attention_layernorm.weight"].float().to(device)
                )
                if rank == 0:
                    print("  ✓ MLP fc1.norm (post_attention_layernorm)")

            # Load gate+up weights (SwiGLU: [gate_proj; up_proj] TP-sharded)
            gate_w = weights["model.layers.0.mlp.gate_proj.weight"].to(device).bfloat16()
            up_w = weights["model.layers.0.mlp.up_proj.weight"].to(device).bfloat16()
            gate_up = torch.cat([gate_w, up_w], dim=0)  # [2*ffn, hidden]
            fc1_shard = gate_up.shape[0] // TP
            fc1_local = gate_up[rank * fc1_shard : (rank + 1) * fc1_shard]
            if hasattr(fc1, 'linear'):
                fc1.linear.weight.data.copy_(fc1_local)
            else:
                fc1.weight.data.copy_(fc1_local)
            if rank == 0:
                print(f"  ✓ MLP fc1 (gate+up): {fc1_local.shape}")

        if hasattr(mlp, 'linear_fc2'):
            fc2 = mlp.linear_fc2
            down_w = weights["model.layers.0.mlp.down_proj.weight"].to(device).bfloat16()
            fc2_shard = down_w.shape[1] // TP
            fc2_local = down_w[:, rank * fc2_shard : (rank + 1) * fc2_shard]
            fc2.weight.data.copy_(fc2_local)
            if rank == 0:
                print(f"  ✓ MLP fc2 (down): {fc2_local.shape}")
    except Exception as e:
        if rank == 0:
            print(f"  ⚠ MLP weight loading: {e}")
            import traceback
            traceback.print_exc()

# ============================================================
# Build RoPE
# ============================================================
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

base = getattr(config, 'rope_theta', 1000000)
mega_rope = RotaryEmbedding(
    kv_channels=HEAD_DIM, rotary_percent=1.0, rotary_interleaved=False,
    seq_len_interpolation_factor=None, rotary_base=base, use_cpu_initialization=True,
)
rotary_pos_emb = mega_rope(T).to(device)
# Megatron expects (q_emb, k_emb) tuple
rotary_pos_emb_tuple = (rotary_pos_emb, rotary_pos_emb)

# ============================================================
# Phase A: Run Megatron attention forward only
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("Phase A: Megatron attention forward")
    print("=" * 60)

x = make_shared_input(T, 1, HIDDEN)  # [seq, batch, hidden] for Megatron

# Build PackedSeqParams
from megatron.core.packed_seq_params import PackedSeqParams
cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)
packed_seq_params = PackedSeqParams(
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_kv=cu_seqlens,
    max_seqlen_q=T,
    max_seqlen_kv=T,
    qkv_format='thd',
)

# Hook into the layer to capture intermediate attention output
_captured = {}
def _capture_after_attn(module, args, output):
    """Hook to capture hidden_states after _forward_attention."""
    if isinstance(output, tuple):
        _captured['after_attn'] = output[0].detach().clone()
    else:
        _captured['after_attn'] = output.detach().clone()

# Register hook on the self_attention module to get its output
hook = megatron_layer.self_attention.linear_proj.register_forward_hook(
    lambda mod, inp, out: _captured.update({'attn_proj_output': out[0].detach().clone() if isinstance(out, tuple) else out.detach().clone()})
)

with torch.no_grad():
    try:
        megatron_out, _ = megatron_layer(
            hidden_states=x,
            attention_mask=None,
            rotary_pos_emb=rotary_pos_emb_tuple,
            packed_seq_params=packed_seq_params,
        )
        if rank == 0:
            print(f"  Megatron full output: shape={megatron_out.shape}")
            if 'attn_proj_output' in _captured:
                print(f"  Captured attn proj output: shape={_captured['attn_proj_output'].shape}, absmax={_captured['attn_proj_output'].abs().max():.6f}")
    except Exception as e:
        if rank == 0:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
        megatron_out = None

hook.remove()

# ============================================================
# Phase A: Run manual forward (same as test_moe_tp8_alignment)
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("Phase A: Manual forward (reference)")
    print("=" * 60)

from sglang.srt.layers.layernorm import RMSNorm as SGLangRMSNorm
from sgl_kernel.flash_attn import flash_attn_varlen_func
from megatron.core.tensor_parallel.mappings import _tree_all_reduce_sum

# Load full weights for manual forward
sglang_input_ln = SGLangRMSNorm(HIDDEN, eps=EPS, cast_x_before_out_mul=True, fp32_residual=False).to(device)
sglang_input_ln.weight.data.copy_(weights["model.layers.0.input_layernorm.weight"].float().to(device))

sglang_post_ln = SGLangRMSNorm(HIDDEN, eps=EPS, cast_x_before_out_mul=True, fp32_residual=False).to(device)
sglang_post_ln.weight.data.copy_(weights["model.layers.0.post_attention_layernorm.weight"].float().to(device))

sglang_q_norm = SGLangRMSNorm(HEAD_DIM, eps=EPS, cast_x_before_out_mul=True, fp32_residual=False).to(device)
sglang_q_norm.weight.data.copy_(weights["model.layers.0.self_attn.q_norm.weight"].float().to(device))

sglang_k_norm = SGLangRMSNorm(HEAD_DIM, eps=EPS, cast_x_before_out_mul=True, fp32_residual=False).to(device)
sglang_k_norm.weight.data.copy_(weights["model.layers.0.self_attn.k_norm.weight"].float().to(device))

# RoPE for manual forward
inv_freq = 1.0 / (base ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device="cpu") / HEAD_DIM))
inv_freq = inv_freq.to(device)
positions = torch.arange(T, dtype=torch.float32, device=device)
freqs = torch.outer(positions, inv_freq)
cos_rope = torch.cos(freqs)
sin_rope = torch.sin(freqs)

def apply_rope_f32(x, cos, sin):
    orig = x.dtype
    x = x.float()
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    c, s = cos.unsqueeze(1).float(), sin.unsqueeze(1).float()
    return torch.cat([x1*c - x2*s, x2*c + x1*s], dim=-1).to(orig)

with torch.no_grad():
    x_flat = x.view(T, HIDDEN)  # [T, H]

    # Input layernorm (no residual for first layer)
    normed = sglang_input_ln.forward_native(x_flat)

    # Local Q/K/V projections (TP-sharded Q, replicated K/V)
    q_w_local = q_w[rank * Q_PER_RANK * HEAD_DIM : (rank + 1) * Q_PER_RANK * HEAD_DIM]
    q_local = F.linear(normed, q_w_local).view(T, Q_PER_RANK, HEAD_DIM)
    k_local = F.linear(normed, k_w).view(T, NUM_KV_HEADS, HEAD_DIM)
    v_local = F.linear(normed, v_w).view(T, NUM_KV_HEADS, HEAD_DIM)

    # QK norm
    q_local = sglang_q_norm.forward_native(q_local.reshape(-1, HEAD_DIM)).view(T, Q_PER_RANK, HEAD_DIM)
    k_local = sglang_k_norm.forward_native(k_local.reshape(-1, HEAD_DIM)).view(T, NUM_KV_HEADS, HEAD_DIM)

    # RoPE
    q_local = apply_rope_f32(q_local, cos_rope, sin_rope)
    k_local = apply_rope_f32(k_local, cos_rope, sin_rope)

    # GQA expand + FA3
    rep = Q_PER_RANK  # each rank's Q heads per KV head = Q_PER_RANK / NUM_KV_HEADS... actually
    # At TP=8: Q_PER_RANK=4, NUM_KV_HEADS=4, so rep=1 (each Q head maps to one KV head)
    if Q_PER_RANK > NUM_KV_HEADS:
        k_exp = k_local.repeat_interleave(Q_PER_RANK // NUM_KV_HEADS, dim=1)
        v_exp = v_local.repeat_interleave(Q_PER_RANK // NUM_KV_HEADS, dim=1)
    else:
        k_exp = k_local
        v_exp = v_local

    attn_out_manual = flash_attn_varlen_func(
        q=q_local.bfloat16(), k=k_exp.bfloat16(), v=v_exp.bfloat16(),
        cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=T, max_seqlen_k=T,
        softmax_scale=1.0/(HEAD_DIM**0.5), causal=True, num_splits=1,
    )
    if isinstance(attn_out_manual, tuple):
        attn_out_manual = attn_out_manual[0]

    # o_proj (TP-sharded) + tree_all_reduce
    attn_flat = attn_out_manual.reshape(T, -1)
    o_local_out = F.linear(attn_flat, o_local)
    o_allreduced = _tree_all_reduce_sum(o_local_out, tp_group)

    if rank == 0:
        print(f"  Manual attn proj output: absmax={o_local_out.abs().max():.6f}")
        print(f"  Manual attn proj (allreduced): absmax={o_allreduced.abs().max():.6f}")

# ============================================================
# Compare attention outputs (before MLP)
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("Comparison: Megatron attention vs Manual attention")
    print("=" * 60)

if 'attn_proj_output' in _captured:
    mega_attn = _captured['attn_proj_output'].view(-1, HIDDEN)
    # Megatron's SGLangRowParallelLinear does tree_all_reduce internally
    # So captured output is AFTER all-reduce. Compare with manual all-reduced output.
    results.check("Attn o_proj (after allreduce): Megatron vs Manual", mega_attn, o_allreduced)

    if rank == 0:
        diff = (mega_attn.float() - o_allreduced.float()).abs()
        print(f"  o_proj max_diff={diff.max().item():.10f}")
        print(f"  o_proj nonzero={diff.gt(0).sum().item()}/{diff.numel()}")
        if diff.max().item() > 0:
            print(f"  Megatron absmax={mega_attn.abs().max():.6f}")
            print(f"  Manual  absmax={o_allreduced.abs().max():.6f}")
            print(f"  Megatron first5={mega_attn[0,:5].tolist()}")
            print(f"  Manual  first5={o_allreduced[0,:5].tolist()}")
else:
    if rank == 0:
        print("  No attention output captured")

results.summary()
dist.barrier()
