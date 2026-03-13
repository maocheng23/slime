"""
Test prefill vs decode for GDN (chunk_gated_delta_rule).

In true-on-policy mode:
  - SGLang: prefill prompt, then decode tokens one-by-one (sequential state updates)
  - Megatron: prefill entire sequence at once (chunk_gated_delta_rule over all tokens)

If these give different results, that's the root cause of logprobs_abs_diff > 0.

This test processes the same tokens through:
  1. Prefill mode: all tokens at once via chunk_gated_delta_rule
  2. Decode mode: first prefill prompt tokens, then extend one token at a time
  3. Compare the hidden states / output at each position

Usage (inside Docker container, single GPU):
  CUDA_VISIBLE_DEVICES=0 python scripts/test_prefill_vs_decode.py
"""
import os
import sys
import torch
import torch.nn.functional as F

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, "/root/slime")

from transformers import AutoConfig, AutoTokenizer
import safetensors.torch as st
import json

# ---- Config ----
MODEL_PATH = "/root/models/Qwen3-Next-4layer"
config = AutoConfig.from_pretrained(MODEL_PATH)

HIDDEN = config.hidden_size
NUM_K_HEADS = config.linear_num_key_heads
NUM_V_HEADS = config.linear_num_value_heads
HEAD_K_DIM = config.linear_key_head_dim
HEAD_V_DIM = config.linear_value_head_dim
V_PER_K_GROUP = NUM_V_HEADS // NUM_K_HEADS

PROMPT_LEN = 8
DECODE_LEN = 8
TOTAL_LEN = PROMPT_LEN + DECODE_LEN

print(f"Model: {MODEL_PATH}")
print(f"Testing prefill vs decode: prompt_len={PROMPT_LEN}, decode_len={DECODE_LEN}, total={TOTAL_LEN}")

# ---- Load weights ----
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
print(f"Loaded {len(weights)} tensors\n")

# ---- Helpers ----
pass_count = 0
fail_count = 0


def check(name, a, b, expect_diff=False):
    global pass_count, fail_count
    diff = (a.float() - b.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    if max_diff == 0:
        print(f"  PASS  {name}: bitwise identical")
        pass_count += 1
    elif expect_diff:
        nonzero_frac = (diff > 0).float().mean().item()
        print(f"  INFO  {name}: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}, nonzero={nonzero_frac:.4f} (expected)")
    else:
        nonzero_frac = (diff > 0).float().mean().item()
        print(f"  FAIL  {name}: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}, nonzero={nonzero_frac:.4f}")
        fail_count += 1
    return max_diff


# ============================================================
# Test 1: chunk_gated_delta_rule — prefill all vs prefill+extend
# ============================================================
print("=" * 60)
print("TEST 1: chunk_gated_delta_rule — full prefill vs prefill+sequential extend")
print("=" * 60)

from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_gdr

# Create test inputs — full sequence
torch.manual_seed(42)
q_full = torch.randn(1, TOTAL_LEN, NUM_V_HEADS, HEAD_K_DIM, device="cuda", dtype=torch.bfloat16)
k_full = torch.randn(1, TOTAL_LEN, NUM_V_HEADS, HEAD_K_DIM, device="cuda", dtype=torch.bfloat16)
v_full = torch.randn(1, TOTAL_LEN, NUM_V_HEADS, HEAD_V_DIM, device="cuda", dtype=torch.bfloat16)
g_full = torch.randn(1, TOTAL_LEN, NUM_V_HEADS, device="cuda", dtype=torch.float32) * 0.1
beta_full = torch.sigmoid(torch.randn(1, TOTAL_LEN, NUM_V_HEADS, device="cuda", dtype=torch.bfloat16))

# Mode 1: Full prefill (all tokens at once) — Megatron's path
cu_full = torch.tensor([0, TOTAL_LEN], dtype=torch.long, device="cuda")
zero_state = torch.zeros(1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, device="cuda", dtype=torch.bfloat16)

with torch.no_grad():
    out_prefill, state_prefill = fla_gdr(
        q_full, k_full, v_full, g=g_full, beta=beta_full,
        initial_state=zero_state.clone(),
        cu_seqlens=cu_full,
        use_qk_l2norm_in_kernel=True,
        output_final_state=True,
    )

# Mode 2: Prefill prompt, then extend one token at a time — SGLang's decode path
cu_prompt = torch.tensor([0, PROMPT_LEN], dtype=torch.long, device="cuda")
cu_single = torch.tensor([0, 1], dtype=torch.long, device="cuda")

with torch.no_grad():
    # Step 1: Prefill the prompt
    out_prompt, state_after_prompt = fla_gdr(
        q_full[:, :PROMPT_LEN], k_full[:, :PROMPT_LEN], v_full[:, :PROMPT_LEN],
        g=g_full[:, :PROMPT_LEN], beta=beta_full[:, :PROMPT_LEN],
        initial_state=zero_state.clone(),
        cu_seqlens=cu_prompt,
        use_qk_l2norm_in_kernel=True,
        output_final_state=True,
    )

    # Collect decode outputs
    decode_outputs = [out_prompt]
    current_state = state_after_prompt.clone()

    # Step 2: Decode one token at a time
    for t in range(DECODE_LEN):
        pos = PROMPT_LEN + t
        cu_t = torch.tensor([0, 1], dtype=torch.long, device="cuda")
        out_t, current_state = fla_gdr(
            q_full[:, pos:pos+1], k_full[:, pos:pos+1], v_full[:, pos:pos+1],
            g=g_full[:, pos:pos+1], beta=beta_full[:, pos:pos+1],
            initial_state=current_state,
            cu_seqlens=cu_t,
            use_qk_l2norm_in_kernel=True,
            output_final_state=True,
        )
        decode_outputs.append(out_t)

    # Concatenate all outputs
    out_decode = torch.cat(decode_outputs, dim=1)  # [1, TOTAL_LEN, H, V]

print(f"  Prefill output shape: {list(out_prefill.shape)}")
print(f"  Decode output shape:  {list(out_decode.shape)}")
check("chunk_gdr: full prefill vs prefill+decode", out_prefill, out_decode)

# Also compare just the prompt portion
check("chunk_gdr: prompt portion only", out_prefill[:, :PROMPT_LEN], out_decode[:, :PROMPT_LEN])

# Compare the decode portion
check("chunk_gdr: decode portion only", out_prefill[:, PROMPT_LEN:], out_decode[:, PROMPT_LEN:])

# Compare final states
check("chunk_gdr: final state", state_prefill, current_state)

# ============================================================
# Test 2: Full GDN layer — prefill all vs prefill+extend
# ============================================================
print("\n" + "=" * 60)
print("TEST 2: Full GDN layer (Qwen3NextGatedDeltaNet) — prefill vs decode")
print("=" * 60)

from slime_plugins.models.qwen3_next import Qwen3NextGatedDeltaNet

gdn = Qwen3NextGatedDeltaNet(config, layer_idx=0, tp_rank=0, tp_size=1).cuda().bfloat16()
prefix = "model.layers.0.linear_attn"
gdn.in_proj_qkvz.weight.data.copy_(weights[f"{prefix}.in_proj_qkvz.weight"].cuda().bfloat16())
gdn.in_proj_ba.weight.data.copy_(weights[f"{prefix}.in_proj_ba.weight"].cuda().bfloat16())
gdn.out_proj.weight.data.copy_(weights[f"{prefix}.out_proj.weight"].cuda().bfloat16())
gdn.A_log.data.copy_(weights[f"{prefix}.A_log"].cuda().bfloat16())
gdn.dt_bias.data.copy_(weights[f"{prefix}.dt_bias"].cuda().bfloat16())
gdn.norm.weight.data.copy_(weights[f"{prefix}.norm.weight"].cuda().bfloat16())
gdn.conv1d.weight.data.copy_(weights[f"{prefix}.conv1d.weight"].cuda().bfloat16())

torch.manual_seed(123)
x_full = torch.randn(1, TOTAL_LEN, HIDDEN, device="cuda", dtype=torch.bfloat16)
cu_full_i32 = torch.tensor([0, TOTAL_LEN], dtype=torch.int32, device="cuda")

# Mode 1: Full prefill
with torch.no_grad():
    out_gdn_prefill = gdn(x_full, cu_seqlens=cu_full_i32)

# Mode 2: Process prompt, then one token at a time
# NOTE: Qwen3NextGatedDeltaNet doesn't support stateful decode natively,
# so we compare the prefill outputs for different sequence lengths
# to check if the GDN output at position T is consistent

# Prefill first T tokens
with torch.no_grad():
    cu_prompt_i32 = torch.tensor([0, PROMPT_LEN], dtype=torch.int32, device="cuda")
    out_prompt_only = gdn(x_full[:, :PROMPT_LEN], cu_seqlens=cu_prompt_i32)

check("GDN layer: prompt portion (prefill-all vs prefill-prompt)",
      out_gdn_prefill[:, :PROMPT_LEN], out_prompt_only)

# ============================================================
# Test 3: Chunk size sensitivity
# ============================================================
print("\n" + "=" * 60)
print("TEST 3: chunk_gated_delta_rule — different total lengths (chunk boundary effects)")
print("=" * 60)

# The chunk kernel processes in fixed-size chunks (typically 64 or 128).
# Test if adding tokens changes output at earlier positions.
for total in [16, 32, 64, 128, 256]:
    q = torch.randn(1, total, NUM_V_HEADS, HEAD_K_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, total, NUM_V_HEADS, HEAD_K_DIM, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, total, NUM_V_HEADS, HEAD_V_DIM, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(1, total, NUM_V_HEADS, device="cuda", dtype=torch.float32) * 0.1
    b = torch.sigmoid(torch.randn(1, total, NUM_V_HEADS, device="cuda", dtype=torch.bfloat16))

    cu = torch.tensor([0, total], dtype=torch.long, device="cuda")
    s0 = torch.zeros(1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, device="cuda", dtype=torch.bfloat16)
    si = torch.zeros(1, dtype=torch.int32, device="cuda")

    with torch.no_grad():
        out_a, _ = fla_gdr(q, k, v, g=g, beta=b, initial_state=s0.clone(),
                            cu_seqlens=cu, use_qk_l2norm_in_kernel=True)
        out_b, _ = fla_gdr(q, k, v, g=g, beta=b, initial_state=s0.clone(),
                            cu_seqlens=cu, use_qk_l2norm_in_kernel=True)

    check(f"chunk_gdr self-consistency at T={total}", out_a, out_b)

# ============================================================
# Test 4: E2E dummy prefill — full model forward prefill vs decode
# ============================================================
print("\n" + "=" * 60)
print("TEST 4: E2E dummy prefill — embedding → 3 GDN layers → norm → lm_head")
print("=" * 60)

from sglang.srt.layers.layernorm import GemmaRMSNorm

# Load all layer weights
embed_weight = weights["model.embed_tokens.weight"].cuda().bfloat16()
final_norm_weight = weights["model.norm.weight"].cuda().bfloat16()
lm_head_weight = weights["lm_head.weight"].cuda().bfloat16()

# Create 3 GDN layers (TP=1)
gdn_layers = []
ln_layers = []
post_ln_layers = []
for layer_idx in range(3):
    pfx = f"model.layers.{layer_idx}.linear_attn"
    g = Qwen3NextGatedDeltaNet(config, layer_idx=layer_idx, tp_rank=0, tp_size=1).cuda().bfloat16()
    g.in_proj_qkvz.weight.data.copy_(weights[f"{pfx}.in_proj_qkvz.weight"].cuda().bfloat16())
    g.in_proj_ba.weight.data.copy_(weights[f"{pfx}.in_proj_ba.weight"].cuda().bfloat16())
    g.out_proj.weight.data.copy_(weights[f"{pfx}.out_proj.weight"].cuda().bfloat16())
    g.A_log.data.copy_(weights[f"{pfx}.A_log"].cuda().bfloat16())
    g.dt_bias.data.copy_(weights[f"{pfx}.dt_bias"].cuda().bfloat16())
    g.norm.weight.data.copy_(weights[f"{pfx}.norm.weight"].cuda().bfloat16())
    g.conv1d.weight.data.copy_(weights[f"{pfx}.conv1d.weight"].cuda().bfloat16())
    gdn_layers.append(g)

    # input_layernorm
    ln = GemmaRMSNorm(HIDDEN, eps=config.rms_norm_eps).cuda().bfloat16()
    ln.weight.data.copy_(weights[f"model.layers.{layer_idx}.input_layernorm.weight"].cuda().bfloat16())
    ln_layers.append(ln)

    # post_attention_layernorm (before MoE — skip MoE for this test)
    pln = GemmaRMSNorm(HIDDEN, eps=config.rms_norm_eps).cuda().bfloat16()
    pln.weight.data.copy_(weights[f"model.layers.{layer_idx}.post_attention_layernorm.weight"].cuda().bfloat16())
    post_ln_layers.append(pln)

# Final norm
final_ln = GemmaRMSNorm(HIDDEN, eps=config.rms_norm_eps).cuda().bfloat16()
final_ln.weight.data.copy_(final_norm_weight)

# Create test input tokens
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
text = "What is 2+3? The answer is five. Another question: what is 10+20?"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"].cuda()
seq_len = input_ids.shape[1]
print(f"Input: '{text}' -> {seq_len} tokens")

cu = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")

# Forward: embedding → layernorm → GDN → residual (skip MoE for isolation)
def forward_gdn_only(input_ids, cu_seqlens):
    """Forward through embedding + 3 GDN layers + final norm + lm_head, skipping MoE."""
    hidden = F.embedding(input_ids, embed_weight)  # [1, T, H]

    for i in range(3):
        residual = hidden
        # input_layernorm
        h_2d = hidden.reshape(-1, HIDDEN)
        h_normed = ln_layers[i](h_2d).reshape(hidden.shape)
        # GDN attention
        gdn_out = gdn_layers[i](h_normed, cu_seqlens=cu_seqlens)
        # Residual (skip MoE — just add residual)
        hidden = residual + gdn_out

    # Final norm + lm_head
    h_2d = hidden.reshape(-1, HIDDEN)
    h_final = final_ln(h_2d)
    logits = F.linear(h_final, lm_head_weight)
    log_probs = F.log_softmax(logits.float(), dim=-1).bfloat16()
    return hidden, logits, log_probs

with torch.no_grad():
    hidden_a, logits_a, lp_a = forward_gdn_only(input_ids, cu)
    hidden_b, logits_b, lp_b = forward_gdn_only(input_ids, cu)

check("E2E GDN-only self-consistency: hidden states", hidden_a, hidden_b)
check("E2E GDN-only self-consistency: logits", logits_a, logits_b)
check("E2E GDN-only self-consistency: log_probs", lp_a, lp_b)

# Now compare full prefill vs split prefill (prompt prefix)
SPLIT = seq_len // 2
cu_split = torch.tensor([0, SPLIT], dtype=torch.int32, device="cuda")

with torch.no_grad():
    hidden_full, logits_full, lp_full = forward_gdn_only(input_ids, cu)
    hidden_split, logits_split, lp_split = forward_gdn_only(input_ids[:, :SPLIT], cu_split)

check("E2E prefix consistency: hidden[:SPLIT] from full vs split",
      hidden_full[:, :SPLIT], hidden_split)

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
total = pass_count + fail_count
print(f"SUMMARY: {pass_count}/{total} passed, {fail_count}/{total} failed")
if fail_count == 0:
    print("ALL TESTS PASSED!")
else:
    print("FAILURES detected — prefill/decode divergence confirmed!")
print("=" * 60)
