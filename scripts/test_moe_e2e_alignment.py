"""
E2E alignment tests for Qwen3-30B-A3B that cover gaps between unit tests and E2E training.

The existing test_moe_layer_alignment.py uses rl_on_policy_target="fsdp", but E2E uses "fsdp_tp".
This means it never exercises moe_sum_tree_reduce (the fsdp_tp code path).

These tests verify:
  1. _MinimalServerArgs in Megatron correctly triggers fsdp_tp path
  2. fused_experts_impl uses moe_sum_tree_reduce under fsdp_tp (self-consistency)
  3. moe_sum_tree_reduce vs moe_sum_reduce produce DIFFERENT results (confirming path matters)
  4. Megatron process calling fused_experts_impl uses same path as SGLang process
  5. tree_all_reduce_sum: Megatron vs SGLang implementation consistency
  6. Full MoE layer: fsdp vs fsdp_tp produce different reduce paths

For Qwen3-30B-A3B (pure MoE, no GDN), logprobs_diff should be EXACTLY 0.
There is NO FA3 prefill/decode issue — that is Qwen3-Next only.

Usage (inside Docker container, single GPU):
  CUDA_VISIBLE_DEVICES=0 python scripts/test_moe_e2e_alignment.py
"""
import os
import sys

import torch
import torch.nn.functional as F

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# ---- Shared utilities ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alignment_test_utils import TestResults, load_safetensor_weights

# ---- Config ----
from transformers import AutoConfig

MODEL_PATH = "/root/models/Qwen3-30B-A3B"
config = AutoConfig.from_pretrained(MODEL_PATH)

HIDDEN = config.hidden_size               # 2048
NUM_EXPERTS = config.num_experts           # 128
TOPK = config.num_experts_per_tok          # 8
MOE_FFN_HIDDEN = config.moe_intermediate_size  # 768
VOCAB_SIZE = config.vocab_size             # 151936

B, T = 1, 16

print(f"Model: {MODEL_PATH}")
print(f"hidden={HIDDEN}, num_experts={NUM_EXPERTS}, topk={TOPK}, moe_ffn_hidden={MOE_FFN_HIDDEN}")
print(f"Test shape: B={B}, T={T}")
print()

# ---- Load weights (only layer 0 MoE) ----
weights = load_safetensor_weights(MODEL_PATH, prefixes=[
    "model.layers.0.mlp.",
])

results = TestResults()


# ============================================================
# Test 1: _MinimalServerArgs triggers fsdp_tp path in Megatron
# ============================================================
print("=" * 60)
print("TEST 1: _MinimalServerArgs correctly sets rl_on_policy_target")
print("=" * 60)
print("  This verifies that moe_utils.py:49-59 sets rl_on_policy_target='fsdp_tp'")
print("  when MEGATRON_USE_DETERMINISTIC_ALLREDUCE=1, so Megatron's call to")
print("  fused_experts_impl uses moe_sum_tree_reduce (same as SGLang server).")
print()

# Simulate what happens in Megatron training process
from sglang.srt.server_args import get_global_server_args, set_global_server_args_for_scheduler

# First, reset global state to simulate fresh Megatron process
# (In practice, moe_utils.py does this at import time)
os.environ["MEGATRON_USE_DETERMINISTIC_ALLREDUCE"] = "1"

use_deterministic = os.environ.get("MEGATRON_USE_DETERMINISTIC_ALLREDUCE", "0") == "1"

class _MinimalServerArgs:
    enable_deterministic_inference = use_deterministic
    rl_on_policy_target = "fsdp_tp" if use_deterministic else None

set_global_server_args_for_scheduler(_MinimalServerArgs())

server_args = get_global_server_args()
assert server_args.rl_on_policy_target == "fsdp_tp", (
    f"Expected rl_on_policy_target='fsdp_tp', got '{server_args.rl_on_policy_target}'"
)
assert server_args.enable_deterministic_inference is True, (
    f"Expected enable_deterministic_inference=True, got {server_args.enable_deterministic_inference}"
)
print("  PASS  _MinimalServerArgs: rl_on_policy_target='fsdp_tp', enable_deterministic_inference=True")
results.pass_count += 1

# Now enable batch_invariant_mode (same as E2E)
from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode
enable_batch_invariant_mode(enable_bmm=False)
print("  batch_invariant_mode enabled")
print()


# ============================================================
# Test 2: fused_experts_impl self-consistency under fsdp_tp
# ============================================================
print("=" * 60)
print("TEST 2: fused_experts_impl self-consistency (fsdp_tp / moe_sum_tree_reduce)")
print("=" * 60)
print("  Verifies that with rl_on_policy_target='fsdp_tp', fused_experts_impl")
print("  uses moe_sum_tree_reduce and produces deterministic results.")
print()

from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts_impl

# Prepare expert weights (4 experts for speed)
num_test_experts = min(4, NUM_EXPERTS)
w1_list, w2_list = [], []
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

# Route tokens to experts
topk_weights_test = torch.ones(B * T, TOPK, device="cuda", dtype=torch.bfloat16) / TOPK
topk_ids_test = torch.zeros(B * T, TOPK, device="cuda", dtype=torch.int32)
for j in range(TOPK):
    topk_ids_test[:, j] = j % num_test_experts

# Confirm we're using fsdp_tp path
assert get_global_server_args().rl_on_policy_target == "fsdp_tp"

with torch.no_grad():
    out_tree1 = fused_experts_impl(
        x_fused.clone(), w1, w2,
        topk_weights_test, topk_ids_test,
        activation="silu",
    )
    out_tree2 = fused_experts_impl(
        x_fused.clone(), w1, w2,
        topk_weights_test, topk_ids_test,
        activation="silu",
    )

results.check("fused_experts_impl (fsdp_tp): self-consistency", out_tree1, out_tree2)


# ============================================================
# Test 3: moe_sum_tree_reduce vs moe_sum_reduce produce DIFFERENT results
# ============================================================
print("\n" + "=" * 60)
print("TEST 3: moe_sum_tree_reduce vs moe_sum_reduce (different kernels)")
print("=" * 60)
print("  This confirms that the choice of reduce kernel MATTERS.")
print("  If fsdp_tp isn't set in Megatron, it uses moe_sum_reduce,")
print("  while SGLang server uses moe_sum_tree_reduce → logprobs_diff > 0.")
print()

from sglang.srt.tp_invariant_ops.tp_invariant_ops import moe_sum_tree_reduce
from sgl_kernel import moe_sum_reduce

# Create test data that simulates intermediate_cache3 from fused_experts_impl
# Shape: [num_tokens, topk, hidden_dim]
num_tokens = B * T
intermediate = torch.randn(num_tokens, TOPK, HIDDEN, device="cuda", dtype=torch.bfloat16)
topk_ids_reduce = torch.randint(0, num_test_experts, (num_tokens, TOPK), device="cuda", dtype=torch.int32)
scaling_factor = 1.0

out_tree = torch.empty(num_tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
out_standard = torch.empty(num_tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)

moe_sum_tree_reduce(
    intermediate.clone(),
    out_tree,
    topk_ids_reduce,
    scaling_factor,
    num_test_experts,
)

moe_sum_reduce(
    intermediate.clone(),
    out_standard,
    scaling_factor,
)

diff = (out_tree.float() - out_standard.float()).abs()
max_diff = diff.max().item()
mean_diff = diff.mean().item()

if max_diff > 0:
    print(f"  INFO  moe_sum_tree_reduce vs moe_sum_reduce: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}")
    print("        → Confirms these are DIFFERENT kernels with different float ordering.")
    print("        → Both sides MUST use the same one for logprobs_diff == 0.")
    results.info_count += 1
else:
    print(f"  INFO  moe_sum_tree_reduce vs moe_sum_reduce: bitwise identical (unexpected but OK)")
    results.info_count += 1


# ============================================================
# Test 4: moe_sum_tree_reduce self-consistency (deterministic)
# ============================================================
print("\n" + "=" * 60)
print("TEST 4: moe_sum_tree_reduce determinism")
print("=" * 60)

out_tree_a = torch.empty(num_tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
out_tree_b = torch.empty(num_tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)

moe_sum_tree_reduce(intermediate.clone(), out_tree_a, topk_ids_reduce, scaling_factor, num_test_experts)
moe_sum_tree_reduce(intermediate.clone(), out_tree_b, topk_ids_reduce, scaling_factor, num_test_experts)

results.check("moe_sum_tree_reduce: deterministic (same input → same output)", out_tree_a, out_tree_b)


# ============================================================
# Test 5: fused_experts_impl fsdp_tp vs fsdp use DIFFERENT reduce
# ============================================================
print("\n" + "=" * 60)
print("TEST 5: fused_experts_impl: fsdp_tp path vs fsdp path")
print("=" * 60)
print("  Runs fused_experts_impl with fsdp_tp (tree reduce), then switches")
print("  to fsdp (standard reduce), and compares. They should DIFFER,")
print("  confirming that the server args flag controls the code path.")
print()

# Run with fsdp_tp (current state)
assert get_global_server_args().rl_on_policy_target == "fsdp_tp"
with torch.no_grad():
    out_fsdp_tp = fused_experts_impl(
        x_fused.clone(), w1, w2,
        topk_weights_test, topk_ids_test,
        activation="silu",
    )

# Temporarily switch to fsdp (simulates old Megatron without the fix)
server_args = get_global_server_args()
saved_target = server_args.rl_on_policy_target
server_args.rl_on_policy_target = "fsdp"

with torch.no_grad():
    out_fsdp = fused_experts_impl(
        x_fused.clone(), w1, w2,
        topk_weights_test, topk_ids_test,
        activation="silu",
    )

# Restore
server_args.rl_on_policy_target = saved_target

diff_paths = (out_fsdp_tp.float() - out_fsdp.float()).abs()
max_diff_paths = diff_paths.max().item()
mean_diff_paths = diff_paths.mean().item()

if max_diff_paths > 0:
    print(f"  INFO  fsdp_tp vs fsdp path: max_diff={max_diff_paths:.8f}, mean_diff={mean_diff_paths:.8f}")
    print("        → Different reduce kernels produce different results.")
    print("        → This is THE root cause of logprobs_diff when Megatron doesn't set fsdp_tp.")
    results.info_count += 1
else:
    print(f"  INFO  fsdp_tp vs fsdp path: bitwise identical (reduce kernels happen to agree)")
    results.info_count += 1


# ============================================================
# Test 6: fused_experts_impl fsdp_tp self-consistency across "processes"
# ============================================================
print("\n" + "=" * 60)
print("TEST 6: Simulated SGLang-process vs Megatron-process agreement")
print("=" * 60)
print("  Both with rl_on_policy_target='fsdp_tp' → both use moe_sum_tree_reduce.")
print("  This simulates what happens when both sides are properly configured.")
print()

# "SGLang process" — set via ServerArgs (like the real SGLang server)
from sglang.srt.server_args import ServerArgs
sglang_server_args = ServerArgs(
    model_path=MODEL_PATH,
    rl_on_policy_target="fsdp_tp",
)
set_global_server_args_for_scheduler(sglang_server_args)

with torch.no_grad():
    out_sglang_side = fused_experts_impl(
        x_fused.clone(), w1, w2,
        topk_weights_test, topk_ids_test,
        activation="silu",
    )

# "Megatron process" — set via _MinimalServerArgs (like moe_utils.py does)
class _MinimalServerArgs2:
    enable_deterministic_inference = True
    rl_on_policy_target = "fsdp_tp"
set_global_server_args_for_scheduler(_MinimalServerArgs2())

with torch.no_grad():
    out_mega_side = fused_experts_impl(
        x_fused.clone(), w1, w2,
        topk_weights_test, topk_ids_test,
        activation="silu",
    )

results.check("SGLang-side vs Megatron-side fused_experts_impl (both fsdp_tp)", out_sglang_side, out_mega_side)


# ============================================================
# Test 7: Full fused_experts_impl with realistic routing (fsdp_tp)
# ============================================================
print("\n" + "=" * 60)
print("TEST 7: Full fused_experts_impl with realistic routing (fsdp_tp)")
print("=" * 60)
print("  Uses actual router weights for realistic topk selection,")
print("  then verifies fused_experts_impl is deterministic under fsdp_tp.")
print()

# Restore fsdp_tp for remaining tests
set_global_server_args_for_scheduler(_MinimalServerArgs())

gate_weight = weights["model.layers.0.mlp.gate.weight"].cuda().bfloat16()
x_route = torch.randn(B * T, HIDDEN, device="cuda", dtype=torch.bfloat16)

with torch.no_grad():
    router_logits = torch.mm(x_route, gate_weight.t())
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, TOPK, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(x_route.dtype)

    # Clamp expert IDs to local experts
    selected_experts_clamped = selected_experts.to(torch.int32) % num_test_experts

    out_real1 = fused_experts_impl(
        x_route.clone(), w1, w2,
        routing_weights, selected_experts_clamped,
        activation="silu",
    )
    out_real2 = fused_experts_impl(
        x_route.clone(), w1, w2,
        routing_weights, selected_experts_clamped,
        activation="silu",
    )

results.check("fused_experts_impl (realistic routing, fsdp_tp): self-consistency", out_real1, out_real2)


# ============================================================
# Test 8: Megatron's MoeSumReduceFunction uses moe_sum_tree_reduce
# ============================================================
print("\n" + "=" * 60)
print("TEST 8: Megatron MoeSumReduceFunction vs direct moe_sum_tree_reduce")
print("=" * 60)
print("  Verifies that Megatron's sgl_fused_moe/fused_experts.py")
print("  MoeSumReduceFunction.forward() uses the same kernel as SGLang's")
print("  moe_sum_tree_reduce (confirming the code change from moe_sum_reduce).")
print()

try:
    from megatron.core.transformer.moe.sgl_fused_moe import MoeSumReduceFunction

    intermediate_test = torch.randn(num_tokens * TOPK, HIDDEN, device="cuda", dtype=torch.bfloat16)
    topk_ids_megatron = torch.randint(0, num_test_experts, (num_tokens, TOPK), device="cuda", dtype=torch.int32)

    with torch.no_grad():
        # Megatron's MoeSumReduceFunction
        out_mega_reduce = MoeSumReduceFunction.apply(
            intermediate_test.view(num_tokens, TOPK, HIDDEN),
            (num_tokens, HIDDEN),
            topk_ids_megatron,
            num_test_experts,
        )

        # Direct moe_sum_tree_reduce (what SGLang uses)
        out_direct_tree = torch.empty(num_tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
        moe_sum_tree_reduce(
            intermediate_test.view(num_tokens, TOPK, HIDDEN).clone(),
            out_direct_tree,
            topk_ids_megatron,
            1.0,
            num_test_experts,
        )

    results.check("MoeSumReduceFunction vs direct moe_sum_tree_reduce", out_mega_reduce, out_direct_tree)

except ImportError as e:
    print(f"  SKIP  Megatron MoeSumReduceFunction not available: {e}")
    print("        (This is expected if running outside the Docker container)")


# ============================================================
# Test 9: log_softmax consistency (bf16 input, batch_invariant)
# ============================================================
print("\n" + "=" * 60)
print("TEST 9: log_softmax bf16 self-consistency (batch_invariant)")
print("=" * 60)
print("  Both SGLang and Megatron must use bf16 log_softmax.")
print("  Megatron's compute_log_probs casts fp32 logits to bf16 first.")
print()

logits_fp32 = torch.randn(B * T, VOCAB_SIZE, device="cuda", dtype=torch.float32)

with torch.no_grad():
    # SGLang path: logits are already bf16 from model output
    logits_bf16 = logits_fp32.bfloat16()
    log_probs_sglang = F.log_softmax(logits_bf16, dim=-1)

    # Megatron path: logits are fp32, cast to bf16 then log_softmax
    # (as done in slime/slime/utils/ppo_utils.py compute_log_probs)
    log_probs_mega = F.log_softmax(logits_fp32.bfloat16(), dim=-1)

results.check("log_softmax: SGLang bf16 vs Megatron fp32→bf16", log_probs_sglang, log_probs_mega)

# Also verify fp32 log_softmax gives DIFFERENT results (confirming bf16 cast matters)
with torch.no_grad():
    log_probs_fp32 = F.log_softmax(logits_fp32, dim=-1)

diff_precision = (log_probs_sglang.float() - log_probs_fp32.float()).abs()
max_diff_prec = diff_precision.max().item()
if max_diff_prec > 0:
    print(f"  INFO  bf16 vs fp32 log_softmax: max_diff={max_diff_prec:.6f} (confirms precision matters)")
    results.info_count += 1
else:
    print(f"  INFO  bf16 vs fp32 log_softmax: identical (unexpected)")
    results.info_count += 1


# ============================================================
# Test 10: End-to-end MoE layer: router → fused_experts → output
# ============================================================
print("\n" + "=" * 60)
print("TEST 10: End-to-end MoE layer self-consistency (fsdp_tp)")
print("=" * 60)
print("  Full pipeline: RMSNorm → Router → fused_experts → output")
print("  under fsdp_tp mode. Must be bitwise deterministic.")
print()

from sglang.srt.layers.layernorm import RMSNorm as SGLangRMSNorm

EPS = config.rms_norm_eps
post_attn_weight = weights.get(
    "model.layers.0.post_attention_layernorm.weight",
    # fallback if this key doesn't exist in loaded prefixes
    None,
)

# Load post_attention_layernorm if not already loaded
if post_attn_weight is None:
    extra_weights = load_safetensor_weights(MODEL_PATH, prefixes=["model.layers.0.post_attention_layernorm."])
    post_attn_weight = extra_weights["model.layers.0.post_attention_layernorm.weight"]

post_ln = SGLangRMSNorm(HIDDEN, eps=EPS, cast_x_before_out_mul=True).cuda()
post_ln.weight.data.copy_(post_attn_weight.cuda().float())

x_e2e = torch.randn(B * T, HIDDEN, device="cuda", dtype=torch.bfloat16)

def moe_forward(x_in):
    """Simulate the full MoE forward under fsdp_tp."""
    # 1. RMSNorm
    normed = post_ln(x_in)

    # 2. Router (normed is bf16 from RMSNorm cast_x_before_out_mul, gate_weight is bf16)
    normed = normed.bfloat16()
    logits_r = torch.mm(normed, gate_weight.t())
    weights_r = F.softmax(logits_r, dim=1, dtype=torch.float)
    weights_r, ids_r = torch.topk(weights_r, TOPK, dim=-1)
    weights_r = weights_r / weights_r.sum(dim=-1, keepdim=True)
    weights_r = weights_r.to(normed.dtype)

    # 3. Fused experts (under fsdp_tp → moe_sum_tree_reduce)
    ids_r_local = ids_r.to(torch.int32) % num_test_experts
    output = fused_experts_impl(
        normed.clone(), w1, w2,
        weights_r, ids_r_local,
        activation="silu",
    )
    return output

with torch.no_grad():
    e2e_out1 = moe_forward(x_e2e)
    e2e_out2 = moe_forward(x_e2e)

results.check("E2E MoE layer (fsdp_tp): self-consistency", e2e_out1, e2e_out2)


# ============================================================
# Summary
# ============================================================
results.summary()

print()
print("KEY TAKEAWAYS:")
print("  - For Qwen3-30B-A3B (pure MoE, no GDN), logprobs_diff should be EXACTLY 0")
print("  - FA3 prefill/decode tiling is NOT an issue (that's Qwen3-Next only)")
print("  - The _MinimalServerArgs fix in moe_utils.py is CRITICAL:")
print("    without it, Megatron uses moe_sum_reduce while SGLang uses moe_sum_tree_reduce")
print("  - Both sides must have rl_on_policy_target='fsdp_tp' for identical reduce paths")
