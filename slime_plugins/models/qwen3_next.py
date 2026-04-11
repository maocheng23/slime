import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import parallel_state as mpu
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.tensor_parallel import reduce_from_tensor_model_parallel_region
from megatron.core.tensor_parallel.mappings import _tree_all_reduce_sum
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from transformers import AutoConfig
from transformers.activations import ACT2FN

def _has_matmul_tp_inv():
    """Lazily check if matmul_tp_inv is available. Must be checked at forward time,
    not import time, because tp_inv_ops is registered later by SGLang's runtime."""
    return hasattr(torch.ops, 'tp_inv_ops') and hasattr(torch.ops.tp_inv_ops, 'matmul_tp_inv')

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
except ImportError:
    pass

# SGLang's chunk_gated_delta_rule: bitwise-identical to rollout, but inference-only (no backward).
# Used during forward_only (no_grad) passes for true-on-policy metrics.
try:
    from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule as _sglang_chunk_gated_delta_rule
except ImportError:
    _sglang_chunk_gated_delta_rule = None

# FLA's chunk_gated_delta_rule: has full autograd backward support.
# Used during training forward (with grad) for correct gradient computation.
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_chunk_gated_delta_rule
except ImportError:
    _fla_chunk_gated_delta_rule = None

# SGLang's l2norm: bitwise-identical pre-normalization (inference-only)
try:
    from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd as _sglang_l2norm_fwd
except ImportError:
    _sglang_l2norm_fwd = None

# SGLang's fused_gdn_gating: bitwise-identical gating (inference-only, no backward)
try:
    from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating as _sglang_fused_gdn_gating
except ImportError:
    _sglang_fused_gdn_gating = None


class _HybridGDNCore(torch.autograd.Function):
    """Forward: SGLang kernels (bitwise match). Backward: FLA + PyTorch (gradient support).

    Wraps the entire GDN core: gating → GQA expand → l2norm → chunk_gated_delta_rule.
    All SGLang inference-only kernels are used in forward for bitwise match with rollout.
    In backward, FLA's chunk_gated_delta_rule (with autograd) is recomputed for gradients.
    """

    @staticmethod
    def forward(ctx, query, key, value, a, b, A_log, dt_bias, cu_seqlens,
                num_k_heads, num_v_heads):
        v_per_k_group = num_v_heads // num_k_heads

        # 1. Gating (SGLang fused kernel — no autograd)
        if _sglang_fused_gdn_gating is not None:
            a_2d = a.reshape(-1, a.shape[-1])
            b_2d = b.reshape(-1, b.shape[-1])
            g, beta = _sglang_fused_gdn_gating(A_log, a_2d, b_2d, dt_bias)
            g = g.squeeze(0)
            beta = beta.squeeze(0)
            if len(a.shape) == 3:
                g = g.reshape(a.shape[0], a.shape[1], -1)
                beta = beta.reshape(b.shape[0], b.shape[1], -1)
        else:
            beta = b.sigmoid()
            g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)

        # Save original q, k for backward (before any modification)
        query_orig, key_orig = query, key

        # 2. Match SGLang's forward_extend exactly:
        #    - Do NOT expand Q/K via repeat_interleave (SGLang passes original heads)
        #    - Let the kernel handle GQA mapping internally
        #    - Use use_qk_l2norm_in_kernel=True (l2norm inside kernel, matching SGLang)

        if cu_seqlens is None:
            B, T = query.shape[:2]
            cu_seqlens = torch.arange(0, (B + 1) * T, T, dtype=torch.long, device=query.device)
        N_seqs = cu_seqlens.shape[0] - 1
        # State uses num_v_heads (not num_k_heads) to match SGLang's kernel behavior
        H_v = value.shape[2]
        K_dim, V_dim = query.shape[3], value.shape[3]
        zero_state = torch.zeros(N_seqs, H_v, K_dim, V_dim, device=query.device, dtype=query.dtype)
        state_idx = torch.arange(N_seqs, dtype=torch.int32, device=query.device)

        o, _, _ = _sglang_chunk_gated_delta_rule(
            query, key, value,
            g=g, beta=beta,
            initial_state=zero_state,
            initial_state_indices=state_idx,
            cu_seqlens=cu_seqlens.to(torch.long),
            use_qk_l2norm_in_kernel=True,
        )

        # Save ORIGINAL (pre-expansion) q, k for backward
        ctx.save_for_backward(query_orig, key_orig, value, a, b, A_log, dt_bias)
        ctx.cu_seqlens = cu_seqlens
        ctx.v_per_k_group = v_per_k_group
        return o

    @staticmethod
    def backward(ctx, do):
        query, key, value, a, b, A_log, dt_bias = ctx.saved_tensors
        cu_seqlens = ctx.cu_seqlens
        v_per_k_group = ctx.v_per_k_group

        # Recompute entire forward with PyTorch/FLA (which have autograd support),
        # then use torch.autograd.grad to compute gradients for all inputs.
        A_log_r = A_log.detach().requires_grad_(True)
        a_r = a.detach().requires_grad_(True)
        b_r = b.detach().requires_grad_(True)
        dt_bias_r = dt_bias.detach().requires_grad_(True)
        query_r = query.detach().requires_grad_(True)
        key_r = key.detach().requires_grad_(True)
        value_r = value.detach().requires_grad_(True)

        with torch.enable_grad():
            # Recompute gating with PyTorch
            beta_r = b_r.sigmoid()
            g_r = -A_log_r.float().exp() * F.softplus(a_r.float() + dt_bias_r)

            # GQA expansion
            if v_per_k_group > 1:
                q_exp = query_r.repeat_interleave(v_per_k_group, dim=2)
                k_exp = key_r.repeat_interleave(v_per_k_group, dim=2)
            else:
                q_exp = query_r
                k_exp = key_r

            # FLA chunk kernel (has full autograd backward)
            o_fla, _ = _fla_chunk_gated_delta_rule(
                q_exp, k_exp, value_r,
                g=g_r, beta=beta_r,
                cu_seqlens=cu_seqlens.to(torch.long),
                use_qk_l2norm_in_kernel=True,
            )

            # Use torch.autograd.grad for explicit gradient computation
            grads = torch.autograd.grad(
                o_fla, [query_r, key_r, value_r, a_r, b_r, A_log_r, dt_bias_r],
                grad_outputs=do,
                allow_unused=True,
            )

        return grads[0], grads[1], grads[2], grads[3], grads[4], grads[5], grads[6], None, None, None


try:
    from sglang.srt.layers.layernorm import GemmaRMSNorm as _GemmaRMSNorm

    class Qwen3NextRMSNorm(_GemmaRMSNorm):
        """GemmaRMSNorm wrapper that handles 3D [batch, seq, hidden] input.
        sgl_kernel.gemma_rmsnorm requires 2D [tokens, hidden]."""

        def forward(self, x, *args, **kwargs):
            orig_shape = x.shape
            if x.ndim == 3:
                x = x.reshape(-1, orig_shape[-1])
            out = super().forward(x, *args, **kwargs)
            if len(orig_shape) == 3:
                out = out.reshape(orig_shape)
            return out
except ImportError:
    # Fallback: PyTorch native implementation (won't be bitwise identical to SGLang)
    class Qwen3NextRMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.zeros(dim))

        def forward(self, x):
            x_f = x.float()
            output = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
            return (output * (1.0 + self.weight.float())).type_as(x)

class _Qwen3NextQKNorm(nn.Module):
    """GemmaRMSNorm for Q/K normalization in Qwen3-Next.
    Accepts (config, hidden_size, eps) kwargs from build_module.
    Uses additive weight: x * (1 + weight), matching SGLang's GemmaRMSNorm."""

    def __init__(self, config=None, hidden_size=None, eps=1e-6, **kwargs):
        super().__init__()
        dim = hidden_size or (config.kv_channels if config else 128)
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        orig_shape = x.shape
        if x.ndim > 2:
            x = x.reshape(-1, orig_shape[-1])
        try:
            import sgl_kernel
            w = self.weight.data.to(x.dtype)
            out = sgl_kernel.gemma_rmsnorm(x, w, self.eps)
        except (ImportError, RuntimeError):
            x_f = x.float()
            out = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
            out = (out * (1.0 + self.weight.float())).type_as(x)
        if len(orig_shape) > 2:
            out = out.reshape(orig_shape)
        return out


class _GemmaRMSNormWithBackward(torch.autograd.Function):
    """Hybrid autograd for GemmaRMSNorm: SGLang kernel forward, native PyTorch backward.

    Forward: sgl_kernel.gemma_rmsnorm (bitwise identical to SGLang inference).
    Backward: native PyTorch GemmaRMSNorm (supports autograd for gradient computation).

    This follows the same pattern as _HybridGDNCore and _RMSNormGatedWithBackward.
    """

    @staticmethod
    def forward(ctx, x, weight, eps):
        import sgl_kernel
        w = weight.data.to(x.dtype)
        out = sgl_kernel.gemma_rmsnorm(x, w, eps)
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        eps = ctx.eps
        # Recompute with native PyTorch for autograd support
        x_req = x.detach().requires_grad_(True)
        w_req = weight.detach().requires_grad_(True)
        with torch.enable_grad():
            x_f = x_req.float()
            variance = x_f.pow(2).mean(dim=-1, keepdim=True)
            x_normed = x_f * torch.rsqrt(variance + eps)
            out = (x_normed * (1.0 + w_req.float())).to(x_req.dtype)
        out.backward(grad_output)
        return x_req.grad, w_req.grad, None


class _GemmaFusedAddRMSNormWithBackward(torch.autograd.Function):
    """Hybrid autograd for fused add+GemmaRMSNorm: SGLang kernel forward, native backward.

    Forward: sgl_kernel.gemma_fused_add_rmsnorm (bitwise identical to SGLang inference).
             This kernel does x += residual, then rmsnorm(x) in one fused pass.
             Returns (normed_output, updated_residual) where updated_residual = x after add.
    Backward: native PyTorch for gradient computation.
    """

    @staticmethod
    def forward(ctx, x, residual, weight, eps):
        import sgl_kernel
        w = weight.data.to(x.dtype)
        # sgl_kernel.gemma_fused_add_rmsnorm modifies x and residual in-place:
        #   x += residual, then x = rmsnorm(x)
        #   residual is also updated to the post-add value
        # We must clone to avoid corrupting autograd tensors.
        x_work = x.clone()
        res_work = residual.clone()
        sgl_kernel.gemma_fused_add_rmsnorm(x_work, res_work, w, eps)
        normed_out = x_work
        updated_residual = res_work
        # Save originals for backward
        ctx.save_for_backward(x, residual, weight)
        ctx.eps = eps
        return normed_out, updated_residual

    @staticmethod
    def backward(ctx, grad_normed, grad_residual):
        x, residual, weight = ctx.saved_tensors
        eps = ctx.eps
        # Recompute with native PyTorch: fused add + gemma_rmsnorm
        x_req = x.detach().requires_grad_(True)
        res_req = residual.detach().requires_grad_(True)
        w_req = weight.detach().requires_grad_(True)
        with torch.enable_grad():
            added = x_req + res_req
            x_f = added.float()
            variance = x_f.pow(2).mean(dim=-1, keepdim=True)
            x_normed = x_f * torch.rsqrt(variance + eps)
            normed_out = (x_normed * (1.0 + w_req.float())).to(x_req.dtype)
            # updated_residual = added (the post-add value)
            updated_residual = added
        # Backward through both outputs
        tensors = [normed_out]
        grads = [grad_normed]
        if grad_residual is not None:
            tensors.append(updated_residual)
            grads.append(grad_residual)
        torch.autograd.backward(tensors, grads)
        return x_req.grad, res_req.grad, w_req.grad, None


class Qwen3NextPreMLPNorm(nn.Module):
    """GemmaRMSNorm wrapper compatible with Megatron's build_module (accepts config kwarg).

    Used for pre_mlp_layernorm in Qwen3-Next which uses GemmaRMSNorm (additive: x * (1 + weight))
    instead of standard RMSNorm (multiplicative: x * weight).
    Also handles residual add (matching SGLang's layer_communicator.prepare_mlp).

    Forward: SGLang CUDA kernel (bitwise identical to inference).
    Backward: Native PyTorch (proper autograd support) via hybrid autograd Functions.

    The weight is stored directly (not in a submodule) to match checkpoint key
    'pre_mlp_layernorm.weight' (not 'pre_mlp_layernorm.norm.weight').
    """

    def __init__(self, config, hidden_size: int = None, eps: float = None, **kwargs):
        super().__init__()
        dim = hidden_size or config.hidden_size
        self._eps = eps or getattr(config, 'layernorm_epsilon', 1e-6)
        self._dim = dim
        # Weight stored directly (matching checkpoint key structure)
        self.weight = nn.Parameter(torch.zeros(dim))
        # Compatibility: Megatron's TransformerBlock accesses fp32_residual on norm modules.
        # Qwen3-Next does NOT use fp32_residual (residual add is outside norm in bf16).
        self.fp32_residual = False

    def _gemma_rms_norm_native(self, x):
        """Fallback: PyTorch native GemmaRMSNorm."""
        x_f = x.float()
        variance = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_f * torch.rsqrt(variance + self._eps)
        return (x_normed * (1.0 + self.weight.float())).to(x.dtype)

    def forward(self, x, residual=None):
        orig_shape = x.shape
        need_reshape = x.ndim == 3
        if need_reshape:
            D = x.shape[-1]
            x = x.contiguous().view(-1, D)
            if residual is not None:
                residual = residual.contiguous().view(-1, D)

        # Use hybrid autograd: SGLang kernel forward (bitwise identity) + native backward.
        try:
            import sgl_kernel  # noqa: F401 — availability check
            if residual is not None:
                out, residual = _GemmaFusedAddRMSNormWithBackward.apply(
                    x, residual, self.weight, self._eps,
                )
            else:
                out = _GemmaRMSNormWithBackward.apply(x, self.weight, self._eps)
        except (ImportError, RuntimeError):
            if residual is not None:
                x = x + residual
                residual = x.clone()
            out = self._gemma_rms_norm_native(x)

        if need_reshape:
            out = out.view(orig_shape)
            if residual is not None:
                residual = residual.view(orig_shape)

        if residual is not None:
            return out, residual
        return out

try:
    from sglang.srt.layers.attention.fla.layernorm_gated import RMSNorm as SGLangRMSNormGated
except ImportError:
    SGLangRMSNormGated = None


class _RMSNormGatedWithBackward(torch.autograd.Function):
    """Forward: SGLang's RMSNormGated (bitwise identical to rollout).
    Backward: FLA's FusedRMSNormGated (has full backward support)."""

    @staticmethod
    def forward(ctx, x, z, sglang_norm, fla_norm):
        # SGLang forward for bitwise identical results
        out = sglang_norm(x, z)
        # Save inputs for FLA backward
        ctx.save_for_backward(x, z)
        ctx.fla_norm = fla_norm
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, z = ctx.saved_tensors
        fla_norm = ctx.fla_norm
        # Use FLA's norm for backward (it supports autograd)
        x_req = x.detach().requires_grad_(True)
        z_req = z.detach().requires_grad_(True)
        with torch.enable_grad():
            fla_out = fla_norm(x_req, z_req)
        fla_out.backward(grad_output)
        return x_req.grad, z_req.grad, None, None


class AlignedRMSNormGated(nn.Module):
    """RMSNormGated that uses SGLang kernel for forward and FLA kernel for backward."""

    def __init__(self, hidden_size, eps, activation, device, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size, device=device, dtype=dtype))
        self.hidden_size = hidden_size
        self.eps = eps
        self.activation = activation
        self._device = device
        self._dtype = dtype
        # Lazy init to avoid double memory allocation
        self._sglang_norm = None
        self._fla_norm = None

    def _ensure_norms(self):
        if self._sglang_norm is None:
            self._sglang_norm = SGLangRMSNormGated(
                self.hidden_size, eps=self.eps,
                device=self._device, dtype=self._dtype,
            )
            self._sglang_norm.weight = self.weight
        if self._fla_norm is None:
            self._fla_norm = FusedRMSNormGated(
                self.hidden_size, eps=self.eps, activation=self.activation,
                device=self._device, dtype=self._dtype,
            )
            self._fla_norm.weight = self.weight

    def forward(self, x, z):
        self._ensure_norms()
        return _RMSNormGatedWithBackward.apply(x, z, self._sglang_norm, self._fla_norm)

try:
    from fla.modules.convolution import causal_conv1d_fwd as _fla_causal_conv1d_fwd
except ImportError:
    _fla_causal_conv1d_fwd = None

# Import SGLang's Triton causal_conv1d — this is what SGLang's forward_extend actually uses.
# CRITICAL: SGLang switched from CUDA (sgl_kernel) to Triton (causal_conv1d_triton).
# The CUDA and Triton kernels produce different floating-point results.
# We MUST use the same Triton kernel for bitwise-identical forward passes.
try:
    from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
        causal_conv1d_fn as _sglang_triton_causal_conv1d_fn,
    )
except ImportError:
    _sglang_triton_causal_conv1d_fn = None

# Fallback: CUDA kernel (only used if Triton import fails)
try:
    import sgl_kernel as _sgl_kernel
except ImportError:
    _sgl_kernel = None


class _CausalConv1dWithBackward(torch.autograd.Function):
    """Causal conv1d with optional SiLU activation.
    Forward: SGLang Triton kernel (conv + fused SiLU) for bitwise identity with rollout,
             or FLA triton kernel + F.silu fallback.
    Backward: manual conv gradient + SiLU chain rule."""

    @staticmethod
    def forward(ctx, x, weight, cu_seqlens, activation):
        # x: [B, T, D], weight: [D, K], cu_seqlens: [num_seqs+1]
        use_sglang_kernel = (activation == "silu" and _sglang_triton_causal_conv1d_fn is not None)

        if use_sglang_kernel:
            # Use the SAME Triton causal_conv1d_fn that SGLang's forward_extend uses.
            # SGLang switched from CUDA to Triton — we must match for bitwise identity.
            B, T, D = x.shape
            total_tokens = B * T
            x_2d = x.clone().reshape(-1, D).transpose(0, 1).contiguous()  # [D, total_tokens]

            # Build arguments matching SGLang's forward_extend call:
            # - conv_states: dummy (no initial state for forward-only)
            # - has_initial_state: all False
            # - cache_indices: sequential
            # - seq_lens_cpu: derived from cu_seqlens
            n_seqs = cu_seqlens.shape[0] - 1
            cu_seqlens_int32 = cu_seqlens.to(torch.int32)
            conv_width = weight.shape[1]
            conv_states = torch.zeros(n_seqs, D, conv_width - 1,
                                      device=x.device, dtype=x.dtype)
            has_initial_state = torch.zeros(n_seqs, dtype=torch.bool, device=x.device)
            cache_indices = torch.arange(n_seqs, dtype=torch.int32, device=x.device)
            seq_lens_cpu = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

            out_2d = _sglang_triton_causal_conv1d_fn(
                x_2d,
                weight,
                bias=None,
                conv_states=conv_states,
                query_start_loc=cu_seqlens_int32,
                seq_lens_cpu=seq_lens_cpu,
                cache_indices=cache_indices,
                has_initial_state=has_initial_state,
                activation="silu",
            )

            out = out_2d.transpose(0, 1).reshape(B, T, D)  # [B, T, D]

            # For backward: we need the pre-SiLU conv output. Recompute it
            # using FLA kernel (cheaper than storing both tensors).
            conv_out, _ = _fla_causal_conv1d_fwd(
                x=x, weight=weight, bias=None, residual=None,
                cu_seqlens=cu_seqlens,
            )
            ctx.save_for_backward(conv_out.detach(), weight)
            ctx.activation = activation
            return out
        else:
            conv_out, _ = _fla_causal_conv1d_fwd(
                x=x, weight=weight, bias=None, residual=None,
                cu_seqlens=cu_seqlens,
            )
            if activation == "silu":
                ctx.save_for_backward(conv_out.detach(), weight)
                ctx.activation = activation
                return F.silu(conv_out)
            else:
                ctx.save_for_backward(weight,)
                ctx.activation = None
                return conv_out

    @staticmethod
    def backward(ctx, grad_output):
        B, T, D = grad_output.shape

        if ctx.activation == "silu":
            conv_out, weight = ctx.saved_tensors
            # SiLU gradient: d/dx silu(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            sig = torch.sigmoid(conv_out)
            grad_output = grad_output * sig * (1.0 + conv_out * (1.0 - sig))
        else:
            (weight,) = ctx.saved_tensors

        W = weight.shape[-1]
        # Causal conv backward w.r.t. input
        grad_y = grad_output.transpose(1, 2).contiguous()  # [B, D, T]
        weight_flip = weight.flip(-1).unsqueeze(1)  # [D, 1, W]
        grad_y_padded = F.pad(grad_y, (W - 1, 0))  # [B, D, T+W-1]
        grad_x = F.conv1d(grad_y_padded, weight_flip, groups=D)  # [B, D, T]
        grad_x = grad_x.transpose(1, 2)  # [B, T, D]
        return grad_x, None, None, None

from .hf_attention import HuggingfaceAttention


# adapt from https://github.com/huggingface/transformers/blob/38a08b6e8ae35857109cedad75377997fecbf9d0/src/transformers/models/qwen3_next/modeling_qwen3_next.py#L564
class Qwen3NextGatedDeltaNet(nn.Module):
    """
    Qwen3NextGatedDeltaNet with TP-sharded forward pass for true-on-policy alignment.

    Weights are stored in full size (for checkpoint compatibility).
    In forward(), weights are sliced to TP-local shards to match SGLang's
    ColumnParallelLinear/RowParallelLinear decomposition, ensuring bitwise
    identical matmul results.
    """

    def __init__(self, config, layer_idx: int, tp_rank: int = 0, tp_size: int = 1):
        super().__init__()
        self.tp_rank = tp_rank
        self.tp_size = tp_size

        # Full model dimensions
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        # TP-local dimensions
        self.num_k_heads_tp = self.num_k_heads // tp_size
        self.num_v_heads_tp = self.num_v_heads // tp_size
        self.key_dim_tp = self.head_k_dim * self.num_k_heads_tp
        self.value_dim_tp = self.head_v_dim * self.num_v_heads_tp

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps

        # Full-size modules for checkpoint loading
        # QKV conv
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ShortConvolution(
            hidden_size=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
        )

        # projection of the input hidden states
        projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        projection_size_ba = self.num_v_heads * 2
        self.in_proj_qkvz = nn.Linear(self.hidden_size, projection_size_qkvz, bias=False)
        self.in_proj_ba = nn.Linear(self.hidden_size, projection_size_ba, bias=False)

        # time step projection (discretization)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        # Use AlignedRMSNormGated when SGLang is available:
        #   Forward: SGLang's RMSNormGated (bitwise identical to rollout)
        #   Backward: FLA's FusedRMSNormGated (has full backward support)
        # Fall back to FLA-only if SGLang is not installed.
        _norm_dtype = config.dtype if config.dtype is not None else torch.get_default_dtype()
        if SGLangRMSNormGated is not None:
            self.norm = AlignedRMSNormGated(
                self.head_v_dim,
                eps=self.layer_norm_epsilon,
                activation=self.activation,
                device=torch.cuda.current_device(),
                dtype=_norm_dtype,
            )
        else:
            self.norm = FusedRMSNormGated(
                self.head_v_dim,
                eps=self.layer_norm_epsilon,
                activation=self.activation,
                device=torch.cuda.current_device(),
                dtype=_norm_dtype,
            )

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        # --- Precompute TP shard indices ---
        v_per_k_group = self.num_v_heads // self.num_k_heads
        groups_per_rank = self.num_k_heads // tp_size

        # in_proj_qkvz: interleaved key-head-group layout, contiguous group slicing
        group_size_qkvz = 2 * self.head_k_dim + 2 * self.head_v_dim * v_per_k_group
        self._qkvz_start = tp_rank * groups_per_rank * group_size_qkvz
        self._qkvz_end = (tp_rank + 1) * groups_per_rank * group_size_qkvz

        # in_proj_ba: interleaved layout, contiguous group slicing
        ba_per_group = 2 * v_per_k_group
        self._ba_start = tp_rank * groups_per_rank * ba_per_group
        self._ba_end = (tp_rank + 1) * groups_per_rank * ba_per_group

        # conv1d: Q|K|V contiguous sections, each independently sharded
        q_per_rank = self.key_dim // tp_size
        v_per_rank_conv = self.value_dim // tp_size
        q_start = tp_rank * q_per_rank
        k_start = self.key_dim + tp_rank * q_per_rank
        v_start = 2 * self.key_dim + tp_rank * v_per_rank_conv
        self.register_buffer('_conv_indices', torch.cat([
            torch.arange(q_start, q_start + q_per_rank),
            torch.arange(k_start, k_start + q_per_rank),
            torch.arange(v_start, v_start + v_per_rank_conv),
        ]).long(), persistent=False)

        # A_log / dt_bias: sharded by v_heads
        v_heads_per_rank = self.num_v_heads // tp_size
        self._a_start = tp_rank * v_heads_per_rank
        self._a_end = (tp_rank + 1) * v_heads_per_rank

        # out_proj: RowParallel style, input columns sharded
        v_per_rank = self.value_dim // tp_size
        self._out_col_start = tp_rank * v_per_rank
        self._out_col_end = (tp_rank + 1) * v_per_rank

    def fix_query_key_value_ordering(self, mixed_qkvz, mixed_ba):
        """
        Derives `query`, `key` and `value` tensors from TP-local `mixed_qkvz` and `mixed_ba`.
        Uses TP-local head counts.
        """
        v_per_k_group = self.num_v_heads // self.num_k_heads  # unchanged by TP

        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads_tp,
            2 * self.head_k_dim + 2 * self.head_v_dim * v_per_k_group,
        )
        new_tensor_shape_ba = mixed_ba.size()[:-1] + (self.num_k_heads_tp, 2 * v_per_k_group)

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)
        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (v_per_k_group * self.head_v_dim),
            (v_per_k_group * self.head_v_dim),
        ]
        split_arg_list_ba = [v_per_k_group, v_per_k_group]
        query, key, value, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=3)
        b, a = torch.split(mixed_ba, split_arg_list_ba, dim=3)
        # [b, sq, ng_tp, np_tp/ng_tp * hn] -> [b, sq, np_tp, hn]
        value = value.reshape(value.size(0), value.size(1), -1, self.head_v_dim)
        z = z.reshape(z.size(0), z.size(1), -1, self.head_v_dim)
        b = b.reshape(b.size(0), b.size(1), self.num_v_heads_tp)
        a = a.reshape(a.size(0), a.size(1), self.num_v_heads_tp)
        return query, key, value, z, b, a

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
    ):
        # TP-sharded in_proj_qkvz: contiguous group slicing (interleaved QKVZ layout)
        w_qkvz = self.in_proj_qkvz.weight[self._qkvz_start:self._qkvz_end]
        projected_states_qkvz = F.linear(hidden_states, w_qkvz)

        # TP-sharded in_proj_ba
        w_ba = self.in_proj_ba.weight[self._ba_start:self._ba_end]
        projected_states_ba = F.linear(hidden_states, w_ba)

        query, key, value, z, b, a = self.fix_query_key_value_ordering(projected_states_qkvz, projected_states_ba)
        query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))

        mixed_qkv = torch.cat((query, key, value), dim=-1)

        # TP-sharded conv1d with fused SiLU activation (matching SGLang's causal_conv1d_fn)
        conv_indices = self._conv_indices.to(self.conv1d.weight.device)
        conv_weight_tp = self.conv1d.weight[conv_indices].squeeze(1)  # [channels_tp, kernel_size]
        mixed_qkv = _CausalConv1dWithBackward.apply(mixed_qkv, conv_weight_tp, cu_seqlens, self.activation)

        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim_tp, self.key_dim_tp, self.value_dim_tp],
            dim=-1,
        )
        query = query.reshape(query.shape[0], query.shape[1], -1, self.head_k_dim)
        key = key.reshape(key.shape[0], key.shape[1], -1, self.head_k_dim)
        value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)

        # TP-sharded A_log, dt_bias
        A_log_tp = self.A_log[self._a_start:self._a_end]
        dt_bias_tp = self.dt_bias[self._a_start:self._a_end]

        if _sglang_chunk_gated_delta_rule is not None:
            # Hybrid autograd: SGLang forward (bitwise match) + FLA backward (gradients)
            core_attn_out = _HybridGDNCore.apply(
                query, key, value, a, b, A_log_tp, dt_bias_tp, cu_seqlens,
                self.num_k_heads, self.num_v_heads,
            )
        else:
            # Fallback: FLA-only path
            beta = b.sigmoid()
            g = -A_log_tp.float().exp() * F.softplus(a.float() + dt_bias_tp)
            v_per_k_group = self.num_v_heads // self.num_k_heads
            if v_per_k_group > 1:
                query = query.repeat_interleave(v_per_k_group, dim=2)
                key = key.repeat_interleave(v_per_k_group, dim=2)
            core_attn_out, _ = _fla_chunk_gated_delta_rule(
                query, key, value,
                g=g, beta=beta,
                cu_seqlens=cu_seqlens.to(torch.long) if cu_seqlens is not None else None,
                use_qk_l2norm_in_kernel=True,
            )
        z_shape_og = z.shape
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)

        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)

        # TP-sharded out_proj (RowParallel: input columns sharded, output is partial sum)
        # Must use matmul_tp_inv to match SGLang's RowParallelLinear which uses
        # torch.ops.tp_inv_ops.matmul_tp_inv when ROW_LINEAR_ENABLE_INV=1.
        # batch_invariant_mode patches aten::mm with matmul_persistent, but that's a
        # DIFFERENT kernel from matmul_tp_persistent. For TP-sharded RowParallel matmuls,
        # SGLang explicitly uses matmul_tp_inv, so we must do the same.
        w_out = self.out_proj.weight[:, self._out_col_start:self._out_col_end]
        _K = core_attn_out.shape[-1]
        if (_has_matmul_tp_inv()
                and os.environ.get("ROW_LINEAR_ENABLE_INV", "0") == "1"
                and _K >= 128 and _K % 128 == 0):
            output = torch.ops.tp_inv_ops.matmul_tp_inv(
                core_attn_out.reshape(-1, _K),
                w_out.t(),
            ).reshape(core_attn_out.shape[:-1] + (w_out.shape[0],))
        else:
            output = F.linear(core_attn_out, w_out)

        return output


class Attention(HuggingfaceAttention):
    def __init__(
        self,
        args,
        config,
        layer_number: int,
        cp_comm_type: str = "p2p",
        pg_collection=None,
    ):
        super().__init__(
            args,
            config,
            layer_number,
            cp_comm_type,
            pg_collection,
        )
        # The TE spec fuses input_layernorm into linear_qkv for standard attention,
        # so TransformerLayer's standalone input_layernorm defaults to IdentityOp.
        # Since our custom GDN module doesn't use the fused linear, we must apply
        # the input_layernorm explicitly here.
        self.input_layernorm = Qwen3NextRMSNorm(self.hf_config.hidden_size, eps=self.hf_config.rms_norm_eps)

        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        self.linear_attn = Qwen3NextGatedDeltaNet(self.hf_config, self.hf_layer_idx, tp_rank, tp_size)

        # Debug: check if mpu TP group matches pg_collection.tp
        mpu_tp = mpu.get_tensor_model_parallel_group()
        pg_tp = pg_collection.tp if pg_collection is not None else None
        print(f"[GDN_TP_CHECK layer={layer_number}] mpu_tp id={id(mpu_tp)} type={type(mpu_tp).__name__} "
              f"pg_tp id={id(pg_tp)} type={type(pg_tp).__name__ if pg_tp else 'None'} "
              f"same={mpu_tp is pg_tp}", flush=True)

    def hf_forward(self, hidden_states, packed_seq_params):
        # Check if TransformerLayer passed a separate residual for fused add+norm.
        # This matches SGLang's LayerCommunicator.prepare_attn which calls:
        #   gemma_fused_add_rmsnorm(hidden_states, residual) -> residual += hs; norm(residual)
        _input_residual = getattr(self, '_sglang_input_residual', None)
        if _input_residual is not None:
            del self._sglang_input_residual
            try:
                import sgl_kernel
                w = self.input_layernorm.weight.data.to(hidden_states.dtype)
                eps = self.input_layernorm.eps if hasattr(self.input_layernorm, 'eps') else self.input_layernorm.variance_epsilon
                # gemma_fused_add_rmsnorm requires 2D [tokens, hidden]
                orig_shape = hidden_states.shape
                x = hidden_states.view(-1, orig_shape[-1])
                r = _input_residual.view(-1, orig_shape[-1])
                # Fused add+norm: residual += hidden_states (in-place), x = norm(residual)
                sgl_kernel.gemma_fused_add_rmsnorm(x, r, w, eps)
                hidden_states = x.view(orig_shape)
                _input_residual = r.view(_input_residual.shape)
                self._sglang_updated_residual = _input_residual
            except (ImportError, RuntimeError):
                _input_residual.add_(hidden_states)
                hidden_states = self.input_layernorm(_input_residual)
                self._sglang_updated_residual = _input_residual
        else:
            hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cu_seqlens=packed_seq_params.cu_seqlens_q,
        )
        # TP reduction for GDN's RowParallel-style out_proj.
        # When SP is on, HuggingfaceAttention.forward() handles TP reduction via reduce_scatter.
        # When SP is off, we must explicitly all-reduce the partial output.
        # Use _tree_all_reduce_sum to match SGLang's tensor_model_parallel_tree_all_reduce
        # (used in communicator.py prepare_mlp when rl_on_policy_target="fsdp_tp").
        # reduce_from_tensor_model_parallel_region calls _reduce which also uses
        # _tree_all_reduce_sum when MEGATRON_USE_DETERMINISTIC_ALLREDUCE=1, but we
        # call it directly to avoid any ambiguity.
        if not self.args.sequence_parallel and self.linear_attn.tp_size > 1:
            tp_group = mpu.get_tensor_model_parallel_group()
            if os.environ.get("MEGATRON_USE_DETERMINISTIC_ALLREDUCE", "0") == "1":
                hidden_states = _tree_all_reduce_sum(hidden_states, tp_group)
            else:
                hidden_states = reduce_from_tensor_model_parallel_region(hidden_states)
        return hidden_states


def _make_qwen3next_fused_ln_linear():
    """Create a subclass of SGLangLayerNormColumnParallelLinear with GemmaRMSNorm.
    Deferred to avoid import-time dependency on Megatron extensions."""
    from megatron.core.extensions.sglang import SGLangLayerNormColumnParallelLinear

    class Qwen3NextFusedLayerNormColumnParallelLinear(SGLangLayerNormColumnParallelLinear):
        """Subclass that replaces the internal SGLangRMSNorm (multiplicative: x * weight)
        with GemmaRMSNorm (additive: x * (1 + weight)) for Qwen3-Next.

        The bridge checkpoint maps input_layernorm.weight to layer_norm_weight,
        but the weight may be wrong if IdentityOp was used for input_layernorm.
        We load the correct weight from HF checkpoint on first forward call.
        """

        def __init__(self, input_size, output_size, *, config, **kwargs):
            super().__init__(input_size, output_size, config=config, **kwargs)
            # Replace the norm with GemmaRMSNorm (additive: x * (1 + weight))
            # The bridge correctly loads input_layernorm.weight via layer_norm_weight mapping.
            self.norm = Qwen3NextPreMLPNorm(config, hidden_size=input_size, eps=config.layernorm_epsilon)

    return Qwen3NextFusedLayerNormColumnParallelLinear


def get_qwen3_next_spec(args, config, vp_stage):
    # always use the moe path
    if not args.num_experts:
        config.moe_layer_freq = [0] * config.num_layers

    # Define the decoder block spec
    kwargs = {
        "use_transformer_engine": True,
    }
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage
    transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)

    assert config.pipeline_model_parallel_layout is None, "not support this at the moment"

    # Slice the layer specs to only include the layers that are built in this pipeline stage.
    # Note: MCore layer_number starts at 1
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)

    hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

    for layer_id in range(num_layers_to_build):
        layer_specs = copy.deepcopy(transformer_layer_spec.layer_specs[layer_id])

        # Qwen3-Next uses GemmaRMSNorm (additive: x * (1 + weight)) for ALL norms
        # except QK-norm. Override all norms that default to SGLangRMSNorm (multiplicative).
        # GemmaRMSNorm weights are stored as deltas from 1 (values ~0),
        # so multiplicative `x * weight` gives ~0, while additive `x * (1 + weight)` is correct.

        # Fix pre_mlp_layernorm (between GDN/attention output and MoE input)
        if hasattr(layer_specs.submodules, 'pre_mlp_layernorm'):
            layer_specs.submodules.pre_mlp_layernorm = Qwen3NextPreMLPNorm

        if hf_config.layer_types[layer_id + offset] == "linear_attention":
            # GDN layers: custom Attention module handles input_layernorm internally
            layer_specs.submodules.self_attention = ModuleSpec(
                module=Attention,
                params={"args": args},
            )
        # L3 (full attention): override norms to use GemmaRMSNorm (additive: x * (1+weight)).
        if hf_config.layer_types[layer_id + offset] == "full_attention":
            # Input layernorm (fused into linear_qkv)
            Qwen3NextFusedLNLinear = _make_qwen3next_fused_ln_linear()
            layer_specs.submodules.self_attention.submodules.linear_qkv = Qwen3NextFusedLNLinear
            # Q/K layernorm: SGLangQKRMSNorm uses multiplicative (weight*x),
            # but Qwen3-Next uses GemmaRMSNorm ((1+weight)*x) for ALL norms.
            layer_specs.submodules.self_attention.submodules.q_layernorm = _Qwen3NextQKNorm
            layer_specs.submodules.self_attention.submodules.k_layernorm = _Qwen3NextQKNorm

        transformer_layer_spec.layer_specs[layer_id] = layer_specs

    # Final layernorm also uses GemmaRMSNorm. Previous regression was from broken residual flow.
    transformer_layer_spec.layer_norm = Qwen3NextPreMLPNorm

    return transformer_layer_spec
