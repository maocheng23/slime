# SLiME — True-On-Policy RL Training

## Project Overview

SLiME is an RL training framework that achieves **true-on-policy** training by ensuring bitwise identical forward passes between SGLang (inference/rollout) and Megatron (training). This repo contains the training side.

**Key principle**: SGLang is the ground truth. Megatron's forward pass must match SGLang exactly. Never modify SGLang to match Megatron.

## Repository Structure

```
slime/                      # Core training framework
  backends/
    megatron_utils/         # Megatron backend (TP/EP/PP)
    fsdp_utils/             # FSDP backend
    training_utils/         # Loss computation, logprob_abs_diff
  utils/ppo_utils.py        # compute_log_probs — must use bf16 for true-on-policy

slime_plugins/models/       # Model-specific Megatron modules
  qwen3_next.py             # Qwen3-Next: _HybridGDNCore, AlignedRMSNormGated, Attention, get_qwen3_next_spec
  hf_attention.py           # Base HuggingfaceAttention class

scripts/                    # Unit tests and debug tools
  test_layer_alignment.py   # TP=1 kernel-by-kernel alignment, Qwen3-Next (15 tests)
  test_tp8_alignment.py     # TP=8 kernel alignment with batch_invariant + tree_allreduce (12 tests)
  test_prefill_vs_decode.py # GDN prefill vs decode divergence test (4 tests)
  test_moe_layer_alignment.py  # TP=1 kernel alignment, Qwen3-30B-A3B MoE (16 tests, uses fsdp)
  test_moe_e2e_alignment.py    # E2E alignment gaps for Qwen3-30B-A3B (10 tests, uses fsdp_tp)
  alignment_test_utils.py   # Shared test utilities (TestResults, TPTestResults, etc.)
  compare_dumps.py          # Compare Megatron/SGLang activation dumps
  models/                   # Model config scripts (e.g., qwen3-next-4layer.sh)

examples/true_on_policy/    # E2E run scripts
  run_moe_megatron.py         # Qwen3-30B-A3B MoE training
  run_qwen3_next_megatron.py  # Qwen3-Next training (debug_one_sample mode)
  run_simple_megatron.py      # Qwen3-0.6B/4B Megatron training
  run_simple.py               # Qwen3-0.6B FSDP training
```

## Development Environment

- **Server**: `radixark@ac-h200-gpu04.tail134ba0.ts.net`
- **Docker**: `sglang-rl-maocheng` (slime image)
- **GPU**: 8x H200
- **Repos inside Docker**:
  - `/root/slime` — this repo
  - `/root/Megatron-LM` — Megatron-LM with SGLang extensions
  - `/sgl-workspace/sglang` — SGLang inference engine
  - `/root/models/` — Model checkpoints

### Code Sync (Local → Docker)
```bash
scp <file> radixark@ac-h200-gpu04.tail134ba0.ts.net:/tmp/<file>
ssh radixark@ac-h200-gpu04.tail134ba0.ts.net "docker cp /tmp/<file> sglang-rl-maocheng:/root/slime/<path>"
```

## Running Tests

```bash
# ---- Qwen3-Next (Hybrid GDN + Full Attention + MoE) ----
# TP=1 kernel alignment (single GPU, ~30s)
CUDA_VISIBLE_DEVICES=0 python scripts/test_layer_alignment.py

# TP=8 kernel alignment (8 GPUs, ~2min)
torchrun --nproc_per_node=8 scripts/test_tp8_alignment.py

# Prefill vs decode (single GPU, ~10s, GDN models only)
CUDA_VISIBLE_DEVICES=0 python scripts/test_prefill_vs_decode.py

# E2E training test (8 GPUs, debug_one_sample)
python examples/true_on_policy/run_qwen3_next_megatron.py

# ---- Qwen3-30B-A3B (Pure MoE) ----
# TP=1 kernel alignment (single GPU, uses fsdp)
CUDA_VISIBLE_DEVICES=0 python scripts/test_moe_layer_alignment.py

# E2E alignment gaps (single GPU, uses fsdp_tp — covers moe_sum_tree_reduce path)
CUDA_VISIBLE_DEVICES=0 python scripts/test_moe_e2e_alignment.py

# E2E training test (8 GPUs, debug_one_sample)
python examples/true_on_policy/run_moe_megatron.py
```

## Deterministic-Mode Forward-Pass Flows (Per Model)

All three models share the same principle: SGLang is the ground truth, Megatron must replicate its exact numerical path. The key flags that control deterministic mode are listed per model below, followed by the complete code-path trace.

### Shared Kernel Stack

| Operation | Kernel | Enabled by |
|-----------|--------|-----------|
| Dense matmul (bf16) | `matmul_persistent` → DeepGEMM `bf16_gemm_nn` (preferred) or Triton `matmul_kernel_persistent` | `enable_batch_invariant_mode()` patches `aten::mm` |
| Dense matmul (fp32) | **NOT SUPPORTED** by batch_invariant Triton kernel (`tl.dot` requires same dtype) | — |
| AllReduce (TP>1) | `tree_all_reduce_sum` (all-gather + binary tree) | Explicit calls in RowParallel layers |
| FlashAttention | `flash_attn_with_kvcache` / `flash_attn_varlen_func` (FA3, `num_splits=1`) | `--sglang-attention-backend fa3` + deterministic mode |
| RMSNorm | `RMSNorm.forward_native` (pure PyTorch, fp32 compute) | `rl_on_policy_target in ("fsdp", "fsdp_tp")` forces native path |
| SiLU+Mul | `SiluAndMul.forward_native` (`F.silu(x[...,:d]) * x[...,d:]`) | `rl_on_policy_target is not None` |
| log_softmax | Triton batch_invariant | `enable_batch_invariant_mode()` patches `aten::_log_softmax` |
| MoE per-token reduce | `moe_sum_tree_reduce` (deterministic binary tree) | `rl_on_policy_target="fsdp_tp"` |
| MoE router | Explicit `F.softmax` + `torch.topk` + renormalize (no fused kernel) | `rl_on_policy_target is not None` |
| RoPE rotation | Must compute in **float32** (not bf16) | `rope_utils.py:_apply_rotary_pos_emb_bshd` uses `.float()` |
| GemmaRMSNorm | `GemmaRMSNorm.forward_native` (`x * (1.0 + weight)`, additive) | Qwen3-Next only |
| RMSNormGated | SGLang fwd + FLA bwd | Qwen3-Next GDN only, `AlignedRMSNormGated` |
| Conv1d+SiLU | `sgl_kernel.causal_conv1d_fwd` | Qwen3-Next GDN only, `_CausalConv1dWithBackward` |
| GDN core | SGLang `chunk_gated_delta_rule` fwd + FLA bwd | Qwen3-Next GDN only, `_HybridGDNCore` |

---

### Model 1: Qwen3-0.6B / Qwen3-4B (Dense)

**Run script**: `examples/true_on_policy/run_simple_megatron.py`
**Key flags**: `--sglang-rl-on-policy-target fsdp`, `--sglang-fp32-residual`, `--sglang-attention-backend fa3`
**SGLang model**: `sglang/srt/models/qwen3.py` → `Qwen3ForCausalLM` → `Qwen3Model` (inherits `Qwen2Model`)
**Megatron model**: `megatron/core/extensions/sglang.py` → `SGLangLinear`, `SGLangRMSNorm`, `SGLangFinalRMSNorm`, `SGLangFlashAttention`, `SGLangLayerNormColumnParallelLinear`

#### SGLang Forward-Pass Flow (Dense, `rl_on_policy_target="fsdp"`)

```
Qwen3ForCausalLM.forward
├── Qwen2Model.forward
│   ├── embed_tokens(input_ids) → bf16
│   ├── for each Qwen3DecoderLayer:
│   │   ├── input_layernorm(hidden_states, residual)     # RMSNorm.forward_native
│   │   │   kwargs: weight_dtype=fp32, cast_x_before_out_mul=True,
│   │   │           override_orig_dtype=fp32, fp32_residual=True
│   │   │   → x = weight(fp32) * norm(x+residual in fp32).to(fp32)
│   │   │   → OUTPUT: hidden_states=fp32, residual=bf16 (not updated when fp32_residual=True and no post_residual_addition)
│   │   │
│   │   ├── Qwen3Attention.forward
│   │   │   ├── hidden_states = hidden_states.bfloat16()  # EXPLICIT CAST
│   │   │   ├── qkv_proj(bf16) → F.linear → matmul_persistent(bf16,bf16) → bf16
│   │   │   ├── q_norm, k_norm: RMSNorm(weight_dtype=fp32, cast_x_before_out_mul=True)
│   │   │   │   → orig_dtype=bf16 (no override), output=bf16
│   │   │   ├── rotary_emb(positions, q, k) → bf16 (RoPE internally uses fp32)
│   │   │   ├── q = q.to(bf16), k = k.to(bf16)  # EXPLICIT CAST
│   │   │   ├── FA3: flash_attn_with_kvcache(bf16,bf16,bf16, num_splits=1) → bf16
│   │   │   └── o_proj(bf16) → matmul_persistent(bf16,bf16) → bf16
│   │   │
│   │   ├── post_attention_layernorm(hidden_states=bf16, residual=bf16)
│   │   │   kwargs: same as input_layernorm (override_orig_dtype=fp32, fp32_residual=True)
│   │   │   → OUTPUT: hidden_states=fp32, residual=bf16
│   │   │
│   │   ├── Qwen2MLP.forward(hidden_states=fp32)
│   │   │   ├── gate_up_proj(fp32) → F.linear(fp32, bf16) → aten::mm(fp32, bf16)
│   │   │   │   → matmul_persistent: DeepGEMM skipped (not bf16), Triton FAILS (tl.dot mixed dtype)
│   │   │   │   ⚠ THIS IS A DTYPE MISMATCH BUG — see "Known Issues" below
│   │   │   ├── SiluAndMul.forward_native
│   │   │   └── down_proj → F.linear → matmul_persistent
│   │   │
│   │   └── postprocess_layer(hidden_states, residual) → trivial (TP=1)
│   │
│   └── final norm: RMSNorm(cast_x_before_out_mul=True, fp32_residual=False)
│       → orig_dtype = x.dtype (whatever MLP outputs)
│
└── logits_processor: torch.matmul(hs.bfloat16(), lm_head.weight.T.bfloat16()) → bf16
    → logits.float() for output
```

#### Megatron Forward-Pass Flow (Dense)

```
GPTModel.forward
├── embedding(input_ids) → bf16
├── TransformerBlock: for each TransformerLayer:
│   ├── SGLangLayerNormColumnParallelLinear (fused norm+QKV proj)
│   │   ├── SGLangRMSNorm(x, residual) → fp32 output (override_orig_dtype=fp32 when fp32_residual=True)
│   │   └── SGLangLinear(norm_output) → x.to(bf16), sglang_mm(bf16, bf16) → bf16
│   ├── SGLangFlashAttention: flash_attn_varlen_func(bf16, num_splits=1) → bf16
│   ├── SGLangLinear (o_proj): x.to(bf16), sglang_mm → bf16
│   ├── SGLangLayerNormColumnParallelLinear (fused norm+gate_up_proj)
│   │   ├── SGLangRMSNorm(x=bf16, residual) → fp32
│   │   └── SGLangLinear → x.to(bf16), sglang_mm(bf16, bf16) → bf16
│   ├── SiLU+Mul → bf16
│   └── SGLangLinear (down_proj) → bf16
├── SGLangFinalRMSNorm(x, residual) → orig_dtype=x.dtype
├── hidden_states.to(bf16) if use_sglang  # explicit cast before lm_head
└── output_layer (lm_head): SGLangLinear → bf16
```

**Key Megatron difference**: `SGLangLinear.forward` always casts input to bf16 first (`x = x.to(torch.bfloat16)`), which avoids the mixed-dtype Triton crash that SGLang hits.

---

### Model 2: Qwen3-30B-A3B (Pure MoE)

**Run script**: `examples/true_on_policy/run_moe_megatron.py`
**Key flags**: `--sglang-rl-on-policy-target fsdp_tp`, `--sglang-attention-backend fa3`, `--use-sglang-router`, `--true-on-policy-model qwen3_moe`
**NO** `--sglang-fp32-residual`
**SGLang model**: `sglang/srt/models/qwen3_moe.py` → `Qwen3MoeForCausalLM` → `Qwen3MoeModel` (inherits `Qwen2MoeModel`)
**Megatron model**: `megatron/core/extensions/sglang.py` → same classes + `SGLangColumnParallelGroupedLinear`, `SGLangRowParallelGroupedLinear`

#### SGLang Forward-Pass Flow (MoE, `rl_on_policy_target="fsdp_tp"`)

```
Qwen3MoeForCausalLM.forward
├── Qwen2MoeModel.forward
│   ├── embed_tokens(input_ids) → bf16
│   ├── for each Qwen3MoeDecoderLayer:
│   │   ├── input_layernorm(hidden_states, residual)     # RMSNorm.forward_native
│   │   │   kwargs: cast_x_before_out_mul=True, fp32_residual=False
│   │   │   → orig_dtype = x.dtype (bf16, NO override_orig_dtype)
│   │   │   → x = x + residual (in bf16), residual = (x+residual).clone()
│   │   │   → norm in fp32, output = weight(bf16) * norm_x.to(bf16) → bf16
│   │   │   → OUTPUT: hidden_states=bf16, residual=bf16
│   │   │
│   │   ├── Qwen3MoeAttention.forward
│   │   │   ├── qkv_proj(bf16) → matmul_persistent(bf16,bf16) → bf16
│   │   │   ├── q_norm, k_norm: RMSNorm(cast_x_before_out_mul=True, fp32_residual=False)
│   │   │   │   → orig_dtype=bf16, output=bf16
│   │   │   ├── rotary_emb → bf16
│   │   │   ├── FA3(bf16, num_splits=1) → bf16
│   │   │   └── o_proj(bf16) → bf16
│   │   │   (TP>1: tree_all_reduce on o_proj output)
│   │   │
│   │   ├── post_attention_layernorm(bf16, residual=bf16)
│   │   │   → same kwargs as input_layernorm → OUTPUT: bf16, residual=bf16
│   │   │
│   │   ├── Qwen3MoeSparseMoeBlock.forward_normal(hidden_states=bf16)
│   │   │   ├── gate: ReplicatedLinear(bf16) → router_logits (bf16)
│   │   │   ├── router: F.softmax(router_logits, dtype=float) → torch.topk → renormalize → .to(bf16)
│   │   │   ├── dispatch: StandardDispatcher → StandardDispatchOutput
│   │   │   ├── experts: FusedMoE → fused_experts_impl
│   │   │   │   ├── invoke_fused_moe_kernel (w13, gate+up) → Triton grouped GEMM
│   │   │   │   ├── silu_and_mul
│   │   │   │   ├── invoke_fused_moe_kernel (w2, down) → Triton grouped GEMM
│   │   │   │   └── moe_sum_tree_reduce (deterministic binary tree, fsdp_tp only)
│   │   │   ├── combine: StandardDispatcher.combine
│   │   │   └── (TP>1: tensor_model_parallel_tree_all_reduce)
│   │   │   → OUTPUT: bf16
│   │   │
│   │   └── postprocess_layer → trivial (TP=1)
│   │
│   └── final norm: RMSNorm(cast_x_before_out_mul=True, fp32_residual=False)
│       → orig_dtype = bf16, output = bf16
│
└── logits_processor: torch.matmul(hs.bfloat16(), lm_head.weight.T.bfloat16()) → bf16
```

**Key MoE differences vs Dense**:
- `fp32_residual=False`: residual add in bf16, not fp32
- No `override_orig_dtype`: RMSNorm output dtype = input dtype (bf16)
- All linear inputs are bf16 → no mixed-dtype issue
- MoE experts use Triton grouped GEMM, not standard `F.linear`
- `moe_sum_tree_reduce` for deterministic per-token combine
- Router uses explicit PyTorch ops (no fused kernel)

---

### Model 3: Qwen3-Next (Hybrid GDN + Full Attention + MoE)

**Run script**: `examples/true_on_policy/run_qwen3_next_megatron.py`
**Key flags**: `--sglang-rl-on-policy-target fsdp_tp`, `--sglang-attention-backend fa3`, `--use-sglang-router`, `--true-on-policy-model qwen3_next`
**NO** `--sglang-fp32-residual`
**SGLang model**: `sglang/srt/models/qwen3_next.py` → `Qwen3NextForCausalLM` → `Qwen3NextModel`
**Megatron model**: `slime_plugins/models/qwen3_next.py` → `_HybridGDNCore`, `AlignedRMSNormGated`, etc.

#### Layer Types

Qwen3-Next has two decoder layer types selected by `config.layers_block_type`:
- `"linear_attention"` → `Qwen3HybridLinearDecoderLayer` (GDN + MoE)
- `"attention"` → `Qwen3HybridAttentionDecoderLayer` (Full Attention + MoE)

#### SGLang Forward-Pass Flow (Qwen3-Next)

```
Qwen3NextForCausalLM.forward
├── Qwen3NextModel.forward
│   ├── embed_tokens(input_ids) → bf16
│   ├── for each layer (GDN or Attention):
│   │   ├── input_layernorm: GemmaRMSNorm(hidden_size)
│   │   │   → x.float(), variance, rsqrt, x * (1.0 + weight.float()), .to(orig_dtype)
│   │   │   → OUTPUT: bf16 (no residual handling in GemmaRMSNorm)
│   │   │
│   │   ├── [GDN layer] Qwen3GatedDeltaNet.forward
│   │   │   ├── in_proj_qkvz: ColumnParallelLinear(bf16) → bf16
│   │   │   ├── in_proj_ba: ColumnParallelLinear(bf16) → bf16
│   │   │   ├── conv1d: sgl_kernel.causal_conv1d_fwd (q, k, v)
│   │   │   ├── partial RoPE (rotary_factor=0.25) on q, k
│   │   │   ├── RadixLinearAttention → chunk_gated_delta_rule (SGLang kernel)
│   │   │   ├── RMSNormGated: norm_before_gate=True, gating with F.silu(z)
│   │   │   └── out_proj: RowParallelLinear → bf16
│   │   │
│   │   ├── [Attention layer] Qwen3HybridAttention.forward
│   │   │   ├── qkv_proj(bf16) → bf16
│   │   │   ├── q_norm, k_norm: GemmaRMSNorm(head_dim)
│   │   │   ├── RoPE → bf16
│   │   │   ├── FA3(bf16, num_splits=1) → bf16
│   │   │   └── o_proj(bf16) → bf16
│   │   │
│   │   ├── residual = hidden_states + residual  # explicit residual add
│   │   │
│   │   ├── post_attention_layernorm: GemmaRMSNorm
│   │   │
│   │   ├── MoE: Qwen2MoeSparseMoeBlock (same as Qwen3 MoE)
│   │   │   ├── gate → F.softmax + torch.topk + renormalize (on-policy)
│   │   │   ├── FusedMoE → fused_experts_impl → moe_sum_tree_reduce
│   │   │   └── (TP>1: tree_all_reduce)
│   │   │
│   │   └── residual = hidden_states + residual
│   │
│   └── final norm: GemmaRMSNorm(hidden_size)
│
└── logits_processor: torch.matmul(hs.bfloat16(), lm_head.weight.T.bfloat16())
```

**Key Qwen3-Next differences**:
- Uses `GemmaRMSNorm` (additive weight: `x * (1.0 + weight)`) instead of `RMSNorm` (multiplicative: `x * weight`)
- No `fp32_residual`, no `override_orig_dtype` — all norms output bf16
- Residual add is explicit in the decoder layer, not inside RMSNorm
- GDN layers have unique kernels: `chunk_gated_delta_rule`, `causal_conv1d_fwd`, `RMSNormGated`
- **Known divergence**: `chunk_gated_delta_rule` gives different results for prefill vs decode (sole source of `logprobs_abs_diff > 0`)

---

### RMSNorm Kwargs Comparison

| Parameter | Dense (`fsdp`) | MoE (`fsdp_tp`) | Qwen3-Next |
|-----------|---------------|-----------------|------------|
| Norm class | `RMSNorm` | `RMSNorm` | `GemmaRMSNorm` |
| `cast_x_before_out_mul` | True | True | N/A (always `x * (1+w)`) |
| `fp32_residual` | **True** | **False** | N/A (no residual in norm) |
| `override_orig_dtype` | **fp32** | None | N/A |
| `weight_dtype` | **fp32** | default (bf16) | N/A (weight init as zeros) |
| Norm output dtype | **fp32** | **bf16** | **bf16** |
| Residual handling | Add in fp32 inside norm | Add in bf16 inside norm | Add outside norm |

### Megatron ↔ SGLang Class Mapping

| SGLang Component | Megatron Extension (`sglang.py`) |
|-----------------|----------------------------------|
| `ColumnParallelLinear` / `RowParallelLinear` | `SGLangLinear` (always casts input to bf16) |
| `RMSNorm` (forward_native) | `SGLangRMSNorm` (fp32_residual from config) |
| `RMSNorm` (Q/K norm) | `SGLangQKRMSNorm` (matches Triton kernel: all-fp32 compute, cast output to input dtype) |
| Final `model.norm` | `SGLangFinalRMSNorm` (orig_dtype=x.dtype) |
| FA3 attention | `SGLangFlashAttention` (`flash_attn_varlen_func`, `num_splits=1`) |
| Fused LN+Linear | `SGLangLayerNormColumnParallelLinear` (sequential norm→linear) |
| MoE grouped linear | `SGLangColumnParallelGroupedLinear` / `SGLangRowParallelGroupedLinear` |
| `GemmaRMSNorm` (Qwen3-Next) | `Qwen3NextRMSNorm` in `slime_plugins/models/qwen3_next.py` |
| GDN (Qwen3-Next) | `_HybridGDNCore` in `slime_plugins/models/qwen3_next.py` |

---

### Known Issues

#### Dense (Qwen3-0.6B): fp32 → bf16 dtype mismatch in MLP
SGLang's `post_attention_layernorm` outputs fp32 (due to `override_orig_dtype=fp32`), but MLP's `gate_up_proj` weight is bf16. `F.linear(fp32, bf16)` goes through `aten::mm` → `matmul_persistent` → DeepGEMM is skipped (requires both bf16) → Triton `tl.dot` crashes on mixed dtypes. **This is a latent bug in SGLang's dense on-policy path** — the MLP `x = x.bfloat16()` line is commented out in `qwen2.py:93-94`. Megatron avoids this because `SGLangLinear` always casts to bf16 first.

#### MoE (Qwen3-30B-A3B): No dtype mismatch
All RMSNorm outputs are bf16 (no `override_orig_dtype`), so all linear inputs are bf16. No mixed-dtype issue.

## Critical: MoE Reduce Path Alignment (fsdp_tp)

The MoE per-token reduce (combining topk expert outputs per token) has **two different kernels**:
- `moe_sum_tree_reduce` — deterministic binary tree, used when `rl_on_policy_target == "fsdp_tp"`
- `moe_sum_reduce` — sgl_kernel CUDA, used otherwise (non-deterministic ordering)

These produce **different floating-point results**. Both sides MUST use `moe_sum_tree_reduce`.

**How it works in E2E**:
- **SGLang server process**: Launched with `--rl-on-policy-target fsdp_tp` → `fused_experts_impl` uses `moe_sum_tree_reduce`
- **Megatron training process**: Calls the SAME `fused_experts_impl` from SGLang, but runs in a separate process. The fix in `Megatron-LM/megatron/core/transformer/moe/moe_utils.py:49-59` creates `_MinimalServerArgs(rl_on_policy_target="fsdp_tp")` when `MEGATRON_USE_DETERMINISTIC_ALLREDUCE=1`, so Megatron also uses `moe_sum_tree_reduce`.

**Megatron forward path**: `MoELayer._sglang_forward()` → `sglang_fused_experts()` → `FusedExpertsFunction.apply()` → SGLang's `fused_experts_impl()` (checks `get_global_server_args().rl_on_policy_target`)

**Testing gap found**: `test_moe_layer_alignment.py` uses `rl_on_policy_target="fsdp"`, so it exercises `moe_sum_reduce`, NOT `moe_sum_tree_reduce`. The new `test_moe_e2e_alignment.py` covers the `fsdp_tp` path.

## Model-Specific Notes

### Qwen3-30B-A3B (Pure MoE)
- 48 layers, hidden_size=2048, 128 experts, topk=8
- **Expected logprobs_abs_diff: EXACTLY 0** (no GDN, no FA3 prefill/decode issue)
- Any non-zero diff is a real bug, not an inherent limitation
- Run script: `examples/true_on_policy/run_moe_megatron.py`
- **Current status (2026-03-17)**: `logprob_abs_diff = 0.0` — EXACTLY ZERO at TP=8, EP=8. All stages bitwise identical.
- **Fixed (2026-03-17)** — Root causes found and fixed via layer-by-layer dump comparison:
  1. **SGLang-side CUDA graph crashes**: Debug prints with `.tolist()`/`.detach().float()` in `logits_processor.py` — guarded with `is_current_stream_capturing()`
  2. **DeepGEMM alignment**: N,K must be multiples of 128. Added gate in `batch_invariant_ops.py`
  3. **Mixed dtype in `mm_batch_invariant`**: cast to bf16 when dtypes differ (safety guard)
  4. **Mixed dtype in `FusedExpertsFunction.forward`**: cast hidden_states to `w1.dtype` before `fused_experts_impl`
  5. **SGLang `RowParallelLinear.forward`**: tree_all_reduce with CUDA graph guard
  6. **SGLangRMSNorm fp32 output**: `fp32_weight * bf16_x → fp32`. Fix: cast output to `orig_dtype` when dtypes differ
  7. **QKV TP sharding when `num_kv_heads < TP`**: Megatron uses group-based sharding (640-dim per rank), SGLang uses KV-head replication (768-dim per rank). Fix: after all-gather in `get_query_key_value_tensors`, re-index to match SGLang's per-rank Q/K/V assignment (`attention.py`)
  8. **SGLangQKRMSNorm computation path**: MoE: `bf16_weight * bf16_x` (matching SGLang bf16 weight). Dense: `fp32_weight * bf16_x` (matching SGLang fp32 weight). Gated by `MEGATRON_ROPE_BF16` env var.
  9. **RoPE bf16 vs fp32**: SGLang MoE uses `_apply_rotary_emb` (bf16 cos/sin). Dense uses `apply_rotary_pos_emb_native` (fp32). Fix: `MEGATRON_ROPE_BF16=1` in `rope_utils.py` (MoE only)
  10. **RowParallelLinear matmul kernel**: Both sides use `matmul_tp_persistent` (TP-invariant) when `ROW_LINEAR_ENABLE_INV=1`. SGLang enables via `is_tp_invariant_mode_enabled()` + `torch.ops.tp_inv_ops.matmul_tp_inv`. Megatron uses same kernel directly.
- **Key architectural difference (num_kv_heads < TP)**: Qwen3-30B-A3B has `num_kv_heads=4, TP=8`. SGLang replicates KV heads (each rank has 1 complete KV head), Megatron uses partial group sharding. The QKV re-index fix in `attention.py` handles this by extracting from the all-gathered QKV using SGLang's assignment.
- **RoPE difference between dense and MoE**: Dense model uses `apply_rotary_pos_emb_native` (fp32). MoE model uses `_apply_rotary_emb` via `RotaryEmbedding.forward_native` (bf16 cos/sin). Both SGLang paths are controlled by `rl_on_policy_target is not None` → `forward_native`, but the actual rotation function differs.
- **Dump comparison methodology**: Compare first N matching tokens (match via embedding identity). Megatron `after_attn` is post-allreduce, SGLang `after_attn` is pre-allreduce — use `moe_input` for fair post-allreduce comparison. Megatron dumps must use rank-0-only filter (`debug_dump.py`).

### Qwen3-Next (Hybrid GDN + Full Attention + MoE)
- **Test model**: Qwen3-Next-4layer (3 GDN layers + 1 Full Attention layer, all with MoE)
- **Run script**: `examples/true_on_policy/run_qwen3_next_megatron.py`
- **Layer types**: `config.layers_block_type` — `["linear_attention", "linear_attention", "linear_attention", "full_attention"]`
- **GDN layers** (layers 0-2): `Qwen3GatedDeltaNet` with `RadixLinearAttention`, `RMSNormGated`, conv1d, `chunk_gated_delta_rule`
- **Full Attention** (layer 3): `Qwen3HybridAttention` with `GemmaRMSNorm` Q/K norms, FA3
- **MoE**: `Qwen2MoeSparseMoeBlock` on all layers (512 experts, topk=10, shared expert with sigmoid gate)
- **Norms**: `GemmaRMSNorm` everywhere (additive weight: `x * (1.0 + weight)`)
- **Dtype flow**: `GemmaRMSNorm` always outputs bf16, residual add is outside norm in bf16

#### Alignment Strategy: Prefill-to-Prefill Only
- **Only compare prefill pass** (Megatron forward on full sequence) with **SGLang prefill pass** (prompt processing)
- **Do NOT compare decode tokens** — GDN `chunk_gated_delta_rule` gives inherently different results for prefill (all tokens at once) vs decode (sequential with state carry-forward). This is NOT a bug.
- **Prompt tokens** (prefill portion) should be **bitwise identical** between Megatron and SGLang
- **Decode tokens** (generated tokens) will have `logprob_abs_diff > 0` due to GDN prefill/decode divergence — this is expected and acceptable

#### Layer-by-Layer Debug Approach
Apply the same methodology as Qwen3-30B-A3B MoE:
1. **Dump at each stage** on both Megatron and SGLang sides
2. **Match tokens by embedding identity** (find matching forward_pass_id)
3. **Compare only prompt tokens** (the prefill portion, not generated tokens)
4. **Find first divergence point** and fix kernel/dtype/weight mismatch
5. **Layer 0 (GDN)**: embedding → GemmaRMSNorm → GDN attention (conv1d, l2norm, chunk_gated_delta_rule, RMSNormGated) → MoE
6. **Layer 3 (Full Attn)**: GemmaRMSNorm → QKV → QK-Norm → RoPE → FA3 → o_proj → MoE
7. **Each GDN layer** uses `_HybridGDNCore` (SGLang fwd, FLA bwd) — forward should match SGLang exactly
8. **Each MoE layer** should match Qwen3-30B-A3B fixes (fused_experts, router, tree_allreduce)

#### Known GDN-Specific Divergence Sources
- `chunk_gated_delta_rule`: prefill processes all tokens at once, decode processes sequentially with state. max_diff ≈ 0.0005 per GDN layer per decode token. **Only affects decode, not prefill.**
- FA3 prefill/decode tiling: different internal tiling for full-sequence causal vs single-token decode. **Only in Full Attention layer, only for decode tokens.**

#### Current Status (2026-03-18)
- **`logprob_abs_diff = 2.56`** — large diff, under investigation
- **Root cause IDENTIFIED**: SGLang's `recompute_logprobs_via_prefill` does NOT actually do a full 101-token prefill. SGLang's **radix cache** reuses the KV cache from the decode phase, so only the 10 response tokens are computed as an extend. The GDN state at the prompt boundary comes from DECODE (sequential token-by-token processing with state carry-forward), NOT from PREFILL (all-at-once chunk processing). Megatron always does a full PREFILL of all 128 tokens. This is the **GDN prefill-vs-decode divergence** described in the skill.
- **Evidence**: (1) Prompt tokens (0-90) are IDENTICAL at all layers between Megatron and SGLang (verified via dumps). (2) SGLang dump analysis shows the "prefill recompute" request uses radix cache — mb157-158 show 91-token prefills (just prompt), then response tokens are extended via cache. No 101-token full-prefill forward pass exists. (3) Per-token logprob diffs accumulate: 0.26 → 8.7, consistent with recurrent GDN state divergence.
- **Cache flush attempted but didn't help**: Flushing the radix cache before recompute was tried but SGLang still uses cached KV. The radix cache may not be fully flushed, or SGLang's chunked prefill still reuses partial state.
- **Conclusion**: For GDN models, `train_rollout_logprob_abs_diff > 0` is **inherent** — the response token logprobs always differ because SGLang's rollout computes them via decode (sequential GDN state) while Megatron computes via prefill (chunked GDN state). The **prefill forward pass is verified IDENTICAL** for prompt tokens (all layers, all kernels). The diff is NOT from a kernel/dtype/weight mismatch — it's from the GDN architecture's inherent prefill-vs-decode divergence.
- **Verification approach**: Compare prefill hidden states (prompt tokens) via dumps — these should be IDENTICAL. The response token logprob diff is expected and acceptable for GDN models.
- **Verified**: Per-layer dump comparison for the first 91 prompt tokens shows ALL layers IDENTICAL (embedding through after_final_layernorm). The diff is NOT in the forward pass kernels — it's in the **sequence packing** affecting GDN chunking.
- **Fix needed**: Ensure Megatron's `compute_log_probs` processes each sequence SEPARATELY (not packed) for GDN models, matching SGLang's per-sequence prefill recomputation. This is a data pipeline change in `actor.py` / `model.py`, not a kernel change.
- **Note**: `recompute_logprobs_via_prefill=True` means SGLang recomputes logprobs via a separate prefill request per sequence. Megatron must process the same per-sequence input.

#### Expected Results (after fix)
- **Prefill-to-prefill logprobs**: `logprob_abs_diff = 0.0` (when Megatron processes same per-sequence input as SGLang)
- **Overall `train_rollout_logprob_abs_diff`**: 0.0 once packing matches

### Dense Models (Qwen3-0.6B)
- Run script: `examples/true_on_policy/run_simple_megatron.py` (with `--sglang-fp32-residual`)
- **Current status (2026-03-17)**: `logprob_abs_diff = 0.0` — EXACTLY ZERO at TP=8. All stages bitwise identical.
- Uses `rl_on_policy_target="fsdp"`, `fp32_residual=True`, NO `MEGATRON_ROPE_BF16` (fp32 RoPE matches SGLang's `apply_rotary_pos_emb_native`)
- Unit tests: `scripts/test_dense_layer_alignment.py`, `scripts/test_dense_full_model.py`
- Compare script: `scripts/compare_dense_dumps.py` (layout-aware, handles interleaved vs concatenated QKV)

#### Verified Identical (2026-03-16)
- **Embedding → Layer Input**: bitwise identical
- **Input LayerNorm**: bitwise identical
- **QKV Linear Projection**: bitwise identical (after rearranging Megatron interleaved → SGLang concatenated layout)
- **Q/K split (before QK-Norm)**: bitwise identical
- **V**: bitwise identical

#### QKV Weight Layout Difference (NOT a bug)
Megatron and SGLang store QKV weights in different layouts:
- **Megatron `linear_qkv`**: interleaved by query group — `[Q_g0 Q_g0 K_g0 V_g0 | Q_g1 Q_g1 K_g1 V_g1 | ...]`
- **SGLang `QKVParallelLinear`**: concatenated — `[Q_all | K_all | V_all]`

The `mixed_qkv` dump outputs have different element ordering but contain the **same values**. Weight sync (`convert_qwen2_to_hf` → SGLang `weight_loader_v2`) correctly converts between layouts. **Do NOT compare `mixed_qkv` element-wise** — use `compare_dense_dumps.py` which rearranges before comparing, or compare Q/K/V after split.

#### QK-Norm Computation Path Mismatch (FIXED)
- **Symptom**: `q_after_qknorm` and `k_after_qknorm` diverge (max_diff ≈ 0.1 for Q, ≈ 0.98 for K)
- **Root cause**: Different computation order in the final multiply step
  - SGLang QK-Norm (no residual, `rl_on_policy_target="megatron"`) → `rms_norm_batch_invariant` → Triton `_rms_norm_kernel`:
    `vals_f32 * inv_rms * weight_f32` → `.to(input_dtype)` — **all fp32, single cast at end**
  - Old Megatron `SGLangQKRMSNorm.forward`:
    `self.weight * x.to(orig_dtype)` — **cast x to bf16 first, then multiply fp32 weight** (extra bf16 truncation)
- **Fix**: Changed Megatron to `(x * self.weight).to(orig_dtype)` — matches Triton kernel exactly
- **Note on weight dtype**: Both sides store QK-Norm weights in fp32 (Qwen3 dense: `weight_dtype=torch.float32`; Qwen3 MoE: default dtype, but Triton kernel `.to(tl.float32)` upcasts anyway). Weight sync sends fp32 from Megatron → SGLang receives in the parameter's dtype.
- **Generality**: The fix is model-agnostic — `SGLangQKRMSNorm` always does all-fp32 compute regardless of weight storage dtype

#### TP=8 Divergence (FIXED — 2026-03-17)

**Status**: `logprob_abs_diff = 0.0` at TP=8 after fix. TP=1, TP=2, TP=8 all verified.

**Root cause**: `down_proj`'s `RowParallelLinear.forward` in SGLang (`/root/sglang_local/python/sglang/srt/layers/linear.py`) used `tensor_model_parallel_all_reduce` (NCCL ring/tree via NCCL_ALGO env), while Megatron's `SGLangRowParallelLinear` used `_tree_all_reduce_sum` (all-gather + explicit binary tree). For TP=2, `a+b == b+a` so both agreed. For TP=8, different addition orders produce different floating-point results → `logprob_abs_diff ≈ 7–13`.

**Fix**: Added `tensor_model_parallel_tree_all_reduce` branch to `RowParallelLinear.forward` in `/root/sglang_local/python/sglang/srt/layers/linear.py`:
```python
elif get_global_server_args().rl_on_policy_target == 'fsdp_tp':
    output = tensor_model_parallel_tree_all_reduce(output_parallel)
```

**Note**: The fix to `communicator.py` (for `o_proj`'s deferred all-reduce in `_gather_hidden_states_and_residual`) was already present in `/root/sglang_local/`. The `linear.py` fix for `down_proj` was the missing piece.

**CRITICAL**: The real SGLang is loaded from `/root/sglang_local/`, NOT `/sgl-workspace/sglang/`. Always apply fixes to `/root/sglang_local/`.

**Megatron fwd pass indexing**: The Megatron dumper saves multiple forward passes (fwd0, fwd1, fwd2, fwd3). Only ONE matches SGLang's fwd6 (the actual inference pass). The matching fwd index varies between runs. Always verify by checking embedding identity first.

#### Previous Issues (Fixed)
- **MLP dtype mismatch**: `x = x.bfloat16()` was commented out in `Qwen2MLP.forward` — now uncommented
- **TransformerLayer residual flow**: SGLang uses `fp32_residual=True` (residual add inside norm in fp32), Megatron uses fused `SGLangLayerNormColumnParallelLinear` with residual support

## Regression Rule: Do Not Break Previously Verified Models

**Every SGLang/Megatron code change must not regress previously verified models.** The verified test matrix is:

| Model | TP | Expected logprobs_abs_diff | Run Command |
|-------|----|---------------------------|-------------|
| Qwen3-0.6B (dense) | TP=1 | ~1e-05 (FA prefill/decode) | `run_simple_megatron.py` |
| Qwen3-0.6B (dense) | TP=8 | ~1e-05 (FA prefill/decode) | `run_simple_megatron.py` |
| Qwen3-30B-A3B (MoE) | TP=8, EP=8 | EXACTLY 0 | `run_moe_megatron.py` |
| Qwen3-Next (GDN) | TP=8, EP=8 | >0 (GDN kernel only) | `run_qwen3_next_megatron.py` |

Before merging any fix:
1. Re-run `run_simple_megatron.py` in `debug_one_sample` mode and verify `logprobs_abs_diff ≈ 1e-05`
2. Re-run `run_moe_megatron.py` in `debug_one_sample` mode and verify `logprobs_abs_diff == 0.0`

**CRITICAL for SGLang forward-pass changes**: Any change to `linear.py`, `logits_processor.py`, `communicator.py`, `batch_invariant_ops.py`, or model files affects ALL models. Run BOTH `run_simple_megatron.py` and `run_moe_megatron.py`.

### Missing Unit Tests (Known Gaps — Need to Write)

These scenarios are verified only via E2E runs. Writing unit tests for them would catch regressions faster:

| Missing Test | What It Should Cover | Priority |
|-------------|---------------------|---------|
| `test_dense_tp8_allreduce.py` | `RowParallelLinear.forward` at TP=8 uses `tensor_model_parallel_tree_all_reduce` when `fsdp_tp`, NOT NCCL. Root cause of TP=8 dense divergence. | HIGH |
| `test_dense_moe_cuda_graph.py` | `tree_all_reduce_sum` is NOT CUDA-graph-capturable. Verify guard `not torch.cuda.is_current_stream_capturing()` in `linear.py` prevents crash. | HIGH |
| `test_deepgemm_alignment.py` | DeepGEMM requires N,K multiple of 128. Test that `matmul_persistent` falls back to Triton for `N=18992` (vocab_size/TP). | HIGH |
| `test_debug_prints_cuda_graph.py` | Debug prints with `.tolist()` inside forward must be guarded with `is_current_stream_capturing()`. | MEDIUM |
| `test_dense_tp2_vs_tp8.py` | TP=2 masks allreduce bugs (a+b==b+a), but TP=8 exposes them. Test that tree allreduce at TP=8 matches TP=1 exactly (vs NCCL which does not). | HIGH |

**Why TP=2 masks allreduce bugs**: For TP=2, any allreduce of `[a, b]` gives `a+b == b+a` (fp32 is commutative for 2 operands). For TP=8, different pairings give different results. Always test at TP=4 or TP=8, not just TP=2.

**CUDA graph incompatible operations** (must guard with `is_current_stream_capturing()`):
- `tensor.tolist()` / `tensor.item()` — CPU sync
- `dist.all_gather()` / `dist.all_reduce()` — NCCL ops (unless captured ahead of time)
- `torch.zeros_like()` — memory allocation
- Any `print()` with tensor data extraction

## Debugging True-On-Policy Issues

1. **Always run unit tests first** — if a kernel test fails, e2e will also fail
2. **Run fsdp_tp-specific tests**: `test_moe_e2e_alignment.py` covers the reduce path that E2E actually uses
3. **Run cross-implementation tests**: `test_moe_tp8_qkv_compare.py` compares actual Megatron vs SGLang QKV paths at TP=8
4. **Enable debug dumps**: `SLIME_DEBUG_LAYER_DUMP=1` (Megatron), `SGLANG_DUMPER_ENABLE=1` (SGLang)
5. **Check logprob_abs_diff** in training logs — should be 0 for Qwen3-30B-A3B and dense models
6. **For GDN models**: prefill-vs-decode divergence is expected, verify it's the only source
7. **batch_invariant_mode must be active** — without it, cuBLAS matmul differs from Triton matmul_persistent
8. **Verify rl_on_policy_target in both processes** — both SGLang server and Megatron training must have `"fsdp_tp"`
9. **RoPE must compute in float32** — Megatron's `_apply_rotary_pos_emb_bshd` in `rope_utils.py` must use `.float()` for cos/sin and rotation, matching SGLang's `apply_rotary_pos_emb_native`
10. **Dump comparison**: Use `scripts/compare_moe_dumps.py` — but note Megatron and SGLang dumps are from different forward passes on different inputs; unit tests are better for kernel comparison
