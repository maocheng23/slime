"""Full model forward comparison: HuggingFace vs Megatron for Qwen3-0.6B.

Loads both models with identical weights, runs forward on same input_ids,
compares logits and per-layer hidden states to find divergence.

Usage: PYTHONPATH=/root/Megatron-LM python3 scripts/test_dense_full_model.py
"""
import os, sys, torch, gc
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29501")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
os.environ["SLIME_DEBUG_LAYER_DUMP"] = "1"
os.environ["SLIME_DEBUG_DUMP_MAX_FWD"] = "1"
os.environ.setdefault("SGLANG_DUMPER_ENABLE", "0")
sys.path.insert(0, "/root/Megatron-LM")

device = torch.device("cuda:0")
MODEL_PATH = "/root/models/Qwen3-0.6B"

# ── Step 1: Run HuggingFace forward (ground truth) ──
print("=" * 70)
print("Step 1: HuggingFace forward pass (ground truth)")
print("=" * 70)

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, trust_remote_code=True
).to(device).eval()

test_text = "The capital of France is"
inputs = tokenizer(test_text, return_tensors="pt").to(device)
input_ids = inputs["input_ids"]
seq_len = input_ids.shape[1]
print(f"Input IDs: {input_ids}")
print(f"Input shape: {input_ids.shape}, seq_len={seq_len}")

# Collect per-layer hidden states from HF
hf_layer_outputs = {}
hooks = []

def make_hook(name):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        hf_layer_outputs[name] = hs.detach().float().cpu()
    return hook_fn

hooks.append(hf_model.model.embed_tokens.register_forward_hook(make_hook("embedding")))
for i, layer in enumerate(hf_model.model.layers):
    hooks.append(layer.register_forward_hook(make_hook(f"layer_{i:02d}")))
hooks.append(hf_model.model.norm.register_forward_hook(make_hook("final_norm")))

with torch.no_grad():
    hf_outputs = hf_model(input_ids)
    hf_logits = hf_outputs.logits

for h in hooks:
    h.remove()

print(f"HF logits shape: {hf_logits.shape}, dtype: {hf_logits.dtype}")
print(f"HF logits mean: {hf_logits.float().mean():.8f}, absmax: {hf_logits.float().abs().max():.8f}")
print(f"HF logits first5: {hf_logits.float().flatten()[:5].tolist()}")
print(f"Captured {len(hf_layer_outputs)} layer outputs from HF")

hf_logits_cpu = hf_logits.detach().float().cpu()

# Also run HF with layer 0 internal hooks for detailed comparison
hf_l0_internals = {}
layer0 = hf_model.model.layers[0]
l0_hooks = []
def make_l0_hook(name):
    def fn(m, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        hf_l0_internals[name] = o.detach().float().cpu()
    return fn
l0_hooks.append(layer0.input_layernorm.register_forward_hook(make_l0_hook("hf_l0_input_ln")))
l0_hooks.append(layer0.self_attn.register_forward_hook(make_l0_hook("hf_l0_attn")))
l0_hooks.append(layer0.post_attention_layernorm.register_forward_hook(make_l0_hook("hf_l0_post_attn_ln")))
l0_hooks.append(layer0.mlp.register_forward_hook(make_l0_hook("hf_l0_mlp")))
with torch.no_grad():
    hf_model(input_ids)
for h in l0_hooks:
    h.remove()

del hf_model
gc.collect()
torch.cuda.empty_cache()

# ── Step 2: Build Megatron model ──
print("\n" + "=" * 70)
print("Step 2: Build Megatron model")
print("=" * 70)

import torch.distributed as dist
if not dist.is_initialized():
    dist.init_process_group(backend="nccl", world_size=1, rank=0)
from megatron.core import parallel_state
if not parallel_state.model_parallel_is_initialized():
    parallel_state.initialize_model_parallel(1, 1)

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.packed_seq_params import PackedSeqParams

hf_config = AutoConfig.from_pretrained(MODEL_PATH)

megatron_config = TransformerConfig(
    num_layers=hf_config.num_hidden_layers,
    hidden_size=hf_config.hidden_size,
    num_attention_heads=hf_config.num_attention_heads,
    num_query_groups=hf_config.num_key_value_heads,
    ffn_hidden_size=hf_config.intermediate_size,
    kv_channels=hf_config.head_dim,
    use_cpu_initialization=True,
    perform_initialization=False,
    bf16=True,
    params_dtype=torch.bfloat16,
    pipeline_dtype=torch.bfloat16,
    autocast_dtype=torch.bfloat16,
    layernorm_epsilon=hf_config.rms_norm_eps,
    hidden_dropout=0.0,
    attention_dropout=0.0,
    bias_activation_fusion=False,
    bias_dropout_fusion=False,
    add_bias_linear=False,
    activation_func=torch.nn.functional.silu,
    gated_linear_unit=True,
    rotary_interleaved=False,
    apply_rope_fusion=False,
    qk_layernorm=True,
    normalization="RMSNorm",
    use_sglang=True,
    use_sglang_attention=True,
    sglang_fp32_residual=True,
    init_model_with_meta_device=False,
)

spec = get_gpt_layer_with_transformer_engine_spec(
    qk_layernorm=True,
    use_sglang=True,
    use_sglang_attention=True,
)

megatron_model = GPTModel(
    config=megatron_config,
    transformer_layer_spec=spec,
    vocab_size=hf_config.vocab_size,
    max_sequence_length=hf_config.max_position_embeddings,
    pre_process=True,
    post_process=True,
    parallel_output=False,
    share_embeddings_and_output_weights=hf_config.tie_word_embeddings,
    position_embedding_type="rope",
    rotary_base=hf_config.rope_theta,
).to(device)

# ── Step 3: Copy weights from HF checkpoint ──
print("\n" + "=" * 70)
print("Step 3: Loading weights into Megatron model")
print("=" * 70)

from safetensors.torch import load_file
ckpt = load_file(os.path.join(MODEL_PATH, "model.safetensors"))

megatron_model.embedding.word_embeddings.weight.data.copy_(
    ckpt["model.embed_tokens.weight"].to(torch.bfloat16).to(device)
)

if "lm_head.weight" in ckpt and megatron_model.output_layer.weight is not None:
    megatron_model.output_layer.weight.data.copy_(
        ckpt["lm_head.weight"].to(torch.bfloat16).to(device)
    )

megatron_model.decoder.final_layernorm.weight.data.copy_(
    ckpt["model.norm.weight"].float().to(device)
)

for i in range(hf_config.num_hidden_layers):
    prefix = f"model.layers.{i}."
    layer = megatron_model.decoder.layers[i]
    sa = layer.self_attention
    mlp = layer.mlp

    def get_w(name):
        return ckpt[prefix + name].to(device)

    sa.linear_qkv.norm.weight.data.copy_(get_w("input_layernorm.weight").float())

    q_w = get_w("self_attn.q_proj.weight").to(torch.bfloat16)
    k_w = get_w("self_attn.k_proj.weight").to(torch.bfloat16)
    v_w = get_w("self_attn.v_proj.weight").to(torch.bfloat16)
    sa.linear_qkv.linear.weight.data.copy_(torch.cat([q_w, k_w, v_w], dim=0))

    sa.linear_proj.weight.data.copy_(get_w("self_attn.o_proj.weight").to(torch.bfloat16))

    sa.q_layernorm.weight.data.copy_(get_w("self_attn.q_norm.weight").float())
    sa.k_layernorm.weight.data.copy_(get_w("self_attn.k_norm.weight").float())

    mlp.linear_fc1.norm.weight.data.copy_(get_w("post_attention_layernorm.weight").float())

    gate_w = get_w("mlp.gate_proj.weight").to(torch.bfloat16)
    up_w = get_w("mlp.up_proj.weight").to(torch.bfloat16)
    mlp.linear_fc1.linear.weight.data.copy_(torch.cat([gate_w, up_w], dim=0))

    mlp.linear_fc2.weight.data.copy_(get_w("mlp.down_proj.weight").to(torch.bfloat16))

print("Weights loaded successfully.")
del ckpt
gc.collect()

# ── Step 4: Run Megatron forward with packed sequence format ──
print("\n" + "=" * 70)
print("Step 4: Running Megatron forward")
print("=" * 70)

megatron_model.eval()

# Megatron with SGLang attention expects packed sequence (THD) format.
# Input shape: [total_tokens, 1, hidden] (with dummy batch dim=1)
# We need to provide packed_seq_params with cu_seqlens.
position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

# Build packed_seq_params for a single sequence
packed_seq_params = PackedSeqParams(
    qkv_format="thd",
    cu_seqlens_q=torch.tensor([0, seq_len], dtype=torch.int32, device=device),
    cu_seqlens_kv=torch.tensor([0, seq_len], dtype=torch.int32, device=device),
    max_seqlen_q=seq_len,
    max_seqlen_kv=seq_len,
)

# Collect per-layer hidden states from Megatron
meg_layer_outputs = {}
meg_hooks = []

def make_meg_hook(name, is_sglang_layer=False):
    def hook_fn(module, input, output):
        if is_sglang_layer and isinstance(output, tuple) and len(output) == 2:
            # TransformerLayer returns ((hidden_states, residual), context) in SGLang mode
            hs_or_tuple, context = output
            if isinstance(hs_or_tuple, tuple) and len(hs_or_tuple) == 2:
                hs, residual = hs_or_tuple
                # HF layer output = residual + mlp_output (post-residual-add)
                # Megatron SGLang mode: hs = mlp_output, residual = pre-mlp residual
                # To match HF, we need to reconstruct: but the residual semantics
                # differ (SGLang uses deferred residual add). Just capture raw hs.
                meg_layer_outputs[name] = hs.detach().float().cpu()
                meg_layer_outputs[name + "_residual"] = residual.detach().float().cpu() if residual is not None else None
            else:
                meg_layer_outputs[name] = hs_or_tuple.detach().float().cpu()
        elif isinstance(output, tuple):
            hs = output[0]
            if isinstance(hs, tuple):
                hs = hs[0]
            meg_layer_outputs[name] = hs.detach().float().cpu()
        else:
            meg_layer_outputs[name] = output.detach().float().cpu()
    return hook_fn

meg_hooks.append(megatron_model.embedding.word_embeddings.register_forward_hook(
    make_meg_hook("embedding")))
for i, layer in enumerate(megatron_model.decoder.layers):
    meg_hooks.append(layer.register_forward_hook(
        make_meg_hook(f"layer_{i:02d}", is_sglang_layer=True)))
meg_hooks.append(megatron_model.decoder.final_layernorm.register_forward_hook(
    make_meg_hook("final_norm")))

with torch.no_grad():
    megatron_logits = megatron_model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=None,
        packed_seq_params=packed_seq_params,
    )

for h in meg_hooks:
    h.remove()

print(f"Megatron logits shape: {megatron_logits.shape}, dtype: {megatron_logits.dtype}")
print(f"Megatron logits mean: {megatron_logits.float().mean():.8f}, absmax: {megatron_logits.float().abs().max():.8f}")
print(f"Megatron logits first5: {megatron_logits.float().flatten()[:5].tolist()}")
print(f"Captured {len(meg_layer_outputs)} layer outputs from Megatron")

meg_logits_cpu = megatron_logits.detach().float().cpu()

# ── Step 5: Compare logits ──
print("\n" + "=" * 70)
print("Step 5: Logits comparison")
print("=" * 70)

# HF logits: [1, S, V], Megatron logits: [S, 1, V] (seq-first)
meg_logits_for_cmp = meg_logits_cpu.squeeze(1).unsqueeze(0)  # [1, S, V]

diff = (hf_logits_cpu - meg_logits_for_cmp).abs()
print(f"Logits diff: max={diff.max():.8e}, mean={diff.mean():.8e}")
print(f"Logits diff per position: {diff.max(dim=-1).values.squeeze().tolist()}")

hf_log_probs = torch.log_softmax(hf_logits_cpu, dim=-1)
meg_log_probs = torch.log_softmax(meg_logits_for_cmp, dim=-1)
lp_diff = (hf_log_probs - meg_log_probs).abs()
print(f"Log-probs diff: max={lp_diff.max():.8e}, mean={lp_diff.mean():.8e}")

hf_argmax = hf_logits_cpu.argmax(dim=-1)
meg_argmax = meg_logits_for_cmp.argmax(dim=-1)
print(f"HF argmax:      {hf_argmax.squeeze().tolist()}")
print(f"Megatron argmax: {meg_argmax.squeeze().tolist()}")
print(f"Argmax match: {(hf_argmax == meg_argmax).all()}")

# ── Step 6: Per-layer comparison ──
# NOTE: HF layer output = residual + mlp_output (post-residual-add).
# Megatron SGLang mode: layer output = (mlp_output, residual) — deferred residual.
# These are NOT directly comparable. We compare:
#   - embedding (should be identical)
#   - final_norm (should be identical if all layers match)
#   - For layers: Megatron's (hs + residual) should approximate HF's output
print("\n" + "=" * 70)
print("Step 6: Per-layer hidden state comparison")
print("=" * 70)

def normalize_shape(hf_t, meg_t):
    """Normalize HF [B,S,H] and Megatron [S,B,H] to same shape."""
    if hf_t.dim() == 3 and meg_t.dim() == 3:
        if hf_t.shape[0] == 1 and meg_t.shape[1] == 1:
            return hf_t, meg_t.transpose(0, 1)
    if hf_t.dim() == 3 and meg_t.dim() == 2:
        return hf_t, meg_t.unsqueeze(0)
    return hf_t, meg_t

# Embedding comparison
print("\n--- Embedding ---")
if "embedding" in hf_layer_outputs and "embedding" in meg_layer_outputs:
    hf_e, meg_e = normalize_shape(hf_layer_outputs["embedding"], meg_layer_outputs["embedding"])
    d = (hf_e - meg_e).abs()
    status = "PASS" if d.max() == 0 else "FAIL"
    print(f"  [{status}] embedding: max_diff={d.max():.8e}, mean_diff={d.mean():.8e}")

# Final norm comparison
print("\n--- Final Norm ---")
if "final_norm" in hf_layer_outputs and "final_norm" in meg_layer_outputs:
    hf_fn, meg_fn = normalize_shape(hf_layer_outputs["final_norm"], meg_layer_outputs["final_norm"])
    d = (hf_fn - meg_fn).abs()
    status = "PASS" if d.max() == 0 else "FAIL"
    print(f"  [{status}] final_norm: max_diff={d.max():.8e}, mean_diff={d.mean():.8e}")
    print(f"  HF  first5: {hf_fn.flatten()[:5].tolist()}")
    print(f"  Meg first5: {meg_fn.flatten()[:5].tolist()}")

# Per-layer comparison: reconstruct Megatron's full hidden state (hs + residual)
print("\n--- Per-layer (Megatron hs+residual vs HF output) ---")
first_divergence = None
for i in range(hf_config.num_hidden_layers):
    key = f"layer_{i:02d}"
    if key not in hf_layer_outputs or key not in meg_layer_outputs:
        continue

    hf_t = hf_layer_outputs[key]
    meg_hs = meg_layer_outputs[key]
    meg_res = meg_layer_outputs.get(f"{key}_residual")

    # Megatron deferred residual: next layer does (hs + residual) inside norm.
    # HF output = residual + mlp_output. So HF_output ≈ meg_hs (if residual is
    # added differently). For a fair comparison, show both raw hs and hs+residual.
    hf_t_n, meg_hs_n = normalize_shape(hf_t, meg_hs)

    d_raw = (hf_t_n - meg_hs_n).abs()

    if meg_res is not None:
        _, meg_res_n = normalize_shape(hf_t, meg_res)
        meg_combined = meg_hs_n + meg_res_n
        d_combined = (hf_t_n - meg_combined).abs()
        status = "PASS" if d_combined.max() == 0 else "FAIL"
        print(f"  [{status}] {key}: hs+res vs HF: max={d_combined.max():.4e}, mean={d_combined.mean():.4e} | "
              f"raw hs vs HF: max={d_raw.max():.4e}")
        if d_combined.max() > 0 and first_divergence is None:
            first_divergence = key
    else:
        status = "PASS" if d_raw.max() == 0 else "FAIL"
        print(f"  [{status}] {key}: max_diff={d_raw.max():.4e}, mean_diff={d_raw.mean():.4e} (no residual)")
        if d_raw.max() > 0 and first_divergence is None:
            first_divergence = key

if first_divergence:
    print(f"\n  >>> First divergence at: {first_divergence}")
else:
    print(f"\n  >>> All layers bitwise identical!")

# ── Step 7: Layer 0 internal comparison (HF vs Megatron dump) ──
print("\n" + "=" * 70)
print("Step 7: Layer 0 internal comparison")
print("=" * 70)

import glob
dump_dir = os.environ.get("SLIME_DEBUG_DUMP_DIR", "/tmp/megatron_debug")
dump_files = sorted(glob.glob(os.path.join(dump_dir, "layer00_*_fwd0.pt")))

print("\n--- HF Layer 0 internals ---")
for k, v in sorted(hf_l0_internals.items()):
    print(f"  {k}: shape={list(v.shape)}, mean={v.mean():.6f}, absmax={v.abs().max():.6f}, "
          f"first3={v.flatten()[:3].tolist()}")

print("\n--- Megatron Layer 0 dumps ---")
meg_dumps = {}
for f in dump_files:
    t = torch.load(f, map_location="cpu").float()
    name = os.path.basename(f).replace("_fwd0.pt", "")
    meg_dumps[name] = t
    print(f"  {name}: shape={list(t.shape)}, mean={t.mean():.6f}, absmax={t.abs().max():.6f}, "
          f"first3={t.flatten()[:3].tolist()}")

# Compare: HF input_layernorm output vs Megatron after_input_ln
print("\n--- Cross-comparison ---")
comparisons = [
    ("hf_l0_input_ln", "layer00_after_input_ln"),
    ("hf_l0_attn", "layer00_after_attn"),
    ("hf_l0_post_attn_ln", "layer00_moe_input"),
    ("hf_l0_mlp", "layer00_moe_output"),
]
for hf_key, meg_key in comparisons:
    if hf_key in hf_l0_internals and meg_key in meg_dumps:
        hf_t = hf_l0_internals[hf_key]
        meg_t = meg_dumps[meg_key]
        # Normalize shapes
        if hf_t.dim() == 3 and meg_t.dim() == 3 and hf_t.shape[0] == 1 and meg_t.shape[1] == 1:
            meg_t = meg_t.transpose(0, 1)
        elif hf_t.dim() == 3 and meg_t.dim() == 2:
            meg_t = meg_t.unsqueeze(0)
        if hf_t.shape == meg_t.shape:
            d = (hf_t - meg_t).abs()
            status = "PASS" if d.max() == 0 else "FAIL"
            print(f"  [{status}] {hf_key} vs {meg_key}: max={d.max():.6e}, mean={d.mean():.6e}")
        else:
            print(f"  [SKIP] {hf_key}={list(hf_t.shape)} vs {meg_key}={list(meg_t.shape)}")
    else:
        missing = []
        if hf_key not in hf_l0_internals: missing.append(f"HF:{hf_key}")
        if meg_key not in meg_dumps: missing.append(f"Meg:{meg_key}")
        print(f"  [SKIP] Missing: {', '.join(missing)}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

dist.destroy_process_group()
