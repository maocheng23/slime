import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import parallel_state as mpu
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.tensor_parallel import reduce_from_tensor_model_parallel_region
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from transformers import AutoConfig
from transformers.activations import ACT2FN

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

        # Save original q, k before GQA expansion for backward
        query_orig, key_orig = query, key

        # 2. GQA expansion
        if v_per_k_group > 1:
            query = query.repeat_interleave(v_per_k_group, dim=2)
            key = key.repeat_interleave(v_per_k_group, dim=2)

        # 3. L2norm + chunk kernel (SGLang — inference-only)
        if _sglang_l2norm_fwd is not None:
            q_norm = _sglang_l2norm_fwd(query)
            k_norm = _sglang_l2norm_fwd(key)
            use_l2norm = False
        else:
            q_norm, k_norm = query, key
            use_l2norm = True

        if cu_seqlens is None:
            B, T = query.shape[:2]
            cu_seqlens = torch.arange(0, (B + 1) * T, T, dtype=torch.long, device=query.device)
        N_seqs = cu_seqlens.shape[0] - 1
        H, K, V = q_norm.shape[2], q_norm.shape[3], value.shape[3]
        zero_state = torch.zeros(N_seqs, H, K, V, device=query.device, dtype=query.dtype)
        state_idx = torch.arange(N_seqs, dtype=torch.int32, device=query.device)

        o, _, _ = _sglang_chunk_gated_delta_rule(
            q_norm, k_norm, value,
            g=g, beta=beta,
            initial_state=zero_state,
            initial_state_indices=state_idx,
            cu_seqlens=cu_seqlens.to(torch.long),
            use_qk_l2norm_in_kernel=use_l2norm,
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

# Import SGLang's CUDA causal_conv1d which supports fused SiLU activation.
# The fused kernel applies SiLU with different intermediate precision than F.silu(),
# so we must use this for bitwise identical forward passes.
try:
    import sgl_kernel as _sgl_kernel
except ImportError:
    _sgl_kernel = None


class _CausalConv1dWithBackward(torch.autograd.Function):
    """Causal conv1d with optional SiLU activation.
    Forward: SGLang CUDA kernel (conv + fused SiLU) for bitwise identity with rollout,
             or FLA triton kernel + F.silu fallback.
    Backward: manual conv gradient + SiLU chain rule."""

    @staticmethod
    def forward(ctx, x, weight, cu_seqlens, activation):
        # x: [B, T, D], weight: [D, K], cu_seqlens: [num_seqs+1]
        use_sglang_kernel = (activation == "silu" and _sgl_kernel is not None)

        if use_sglang_kernel:
            # Use SGLang's fused CUDA kernel for conv + SiLU.
            # sgl_kernel.causal_conv1d_fwd expects (D, total_seqlen) for varlen mode.
            B, T, D = x.shape
            # Reshape [B, T, D] -> [D, B*T] (varlen format)
            x_2d = x.clone().reshape(-1, D).transpose(0, 1).contiguous()  # [D, B*T]
            _sgl_kernel.causal_conv1d_fwd(
                x_2d,              # modified in-place [D, B*T]
                weight,            # [D, K]
                None,              # bias
                None,              # conv_states
                cu_seqlens,        # query_start_loc
                None,              # cache_indices
                None,              # has_initial_state
                True,              # silu_activation
                -1,                # pad_slot_id
            )
            out = x_2d.transpose(0, 1).reshape(B, T, D)  # [B, T, D]

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

# ---- Debug layer dump infrastructure ----
_DEBUG_DUMP = os.environ.get("SLIME_DEBUG_LAYER_DUMP", "0") == "1"
_DUMP_DIR = "/tmp/megatron_debug"
_DUMP_MAX_FWD = int(os.environ.get("SLIME_DEBUG_DUMP_MAX_FWD", "4"))  # dump this many forward passes
_dump_fwd_count = [0]  # mutable counter


def _dsave(name, tensor):
    """Save tensor for debug comparison. Dumps first N forward passes with mb suffix."""
    if not _DEBUG_DUMP or _dump_fwd_count[0] >= _DUMP_MAX_FWD:
        return
    tp_rank = mpu.get_tensor_model_parallel_rank() if mpu.model_parallel_is_initialized() else 0
    if tp_rank != 0:
        return
    os.makedirs(_DUMP_DIR, exist_ok=True)
    mb = _dump_fwd_count[0]
    t = tensor.detach().float().cpu()
    torch.save(t, f"{_DUMP_DIR}/{name}_mb{mb}.pt")
    print(f"[MEGA DUMP] mb{mb} {name}: shape={list(t.shape)} mean={t.mean():.8f} std={t.std():.8f} absmax={t.abs().max():.8f}", flush=True)


def _dprint_weight(name, param):
    """Print weight stats on first call."""
    if not _DEBUG_DUMP:
        return
    t = param.detach().float()
    print(f"[MEGA WEIGHT] {name}: shape={list(t.shape)} mean={t.mean():.8f} std={t.std():.8f} absmax={t.abs().max():.8f}", flush=True)


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
        self._weights_printed = False

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
        v_per_rank = self.value_dim // tp_size
        q_start = tp_rank * q_per_rank
        k_start = self.key_dim + tp_rank * q_per_rank
        v_start = 2 * self.key_dim + tp_rank * v_per_rank
        self.register_buffer('_conv_indices', torch.cat([
            torch.arange(q_start, q_start + q_per_rank),
            torch.arange(k_start, k_start + q_per_rank),
            torch.arange(v_start, v_start + v_per_rank),
        ]).long(), persistent=False)

        # A_log / dt_bias: sharded by v_heads
        v_heads_per_rank = self.num_v_heads // tp_size
        self._a_start = tp_rank * v_heads_per_rank
        self._a_end = (tp_rank + 1) * v_heads_per_rank

        # out_proj: RowParallel style, input columns sharded
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
        # Print weight stats on first call
        if not self._weights_printed and _DEBUG_DUMP:
            self._weights_printed = True
            _dprint_weight(f"gdn{self.layer_idx}.in_proj_qkvz", self.in_proj_qkvz.weight)
            _dprint_weight(f"gdn{self.layer_idx}.in_proj_ba", self.in_proj_ba.weight)
            _dprint_weight(f"gdn{self.layer_idx}.out_proj", self.out_proj.weight)
            _dprint_weight(f"gdn{self.layer_idx}.A_log", self.A_log)
            _dprint_weight(f"gdn{self.layer_idx}.dt_bias", self.dt_bias)
            _dprint_weight(f"gdn{self.layer_idx}.norm", self.norm.weight)
            _dprint_weight(f"gdn{self.layer_idx}.conv1d", self.conv1d.weight)

        _dsave(f"gdn{self.layer_idx}_input", hidden_states)

        # TP-sharded in_proj_qkvz: contiguous group slicing (interleaved QKVZ layout)
        w_qkvz = self.in_proj_qkvz.weight[self._qkvz_start:self._qkvz_end]
        projected_states_qkvz = F.linear(hidden_states, w_qkvz)
        _dsave(f"gdn{self.layer_idx}_proj_qkvz", projected_states_qkvz)

        # TP-sharded in_proj_ba
        w_ba = self.in_proj_ba.weight[self._ba_start:self._ba_end]
        projected_states_ba = F.linear(hidden_states, w_ba)
        _dsave(f"gdn{self.layer_idx}_proj_ba", projected_states_ba)

        query, key, value, z, b, a = self.fix_query_key_value_ordering(projected_states_qkvz, projected_states_ba)
        query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))

        mixed_qkv = torch.cat((query, key, value), dim=-1)

        # TP-sharded conv1d with fused SiLU activation (matching SGLang's causal_conv1d_fn)
        conv_indices = self._conv_indices.to(self.conv1d.weight.device)
        conv_weight_tp = self.conv1d.weight[conv_indices].squeeze(1)  # [channels_tp, kernel_size]
        mixed_qkv = _CausalConv1dWithBackward.apply(mixed_qkv, conv_weight_tp, cu_seqlens, self.activation)
        _dsave(f"gdn{self.layer_idx}_after_conv", mixed_qkv)

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
        _dsave(f"gdn{self.layer_idx}_after_gdr", core_attn_out)

        z_shape_og = z.shape
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        _dsave(f"gdn{self.layer_idx}_after_norm", core_attn_out)

        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)

        # TP-sharded out_proj (RowParallel: input columns sharded, output is partial sum)
        w_out = self.out_proj.weight[:, self._out_col_start:self._out_col_end]
        output = F.linear(core_attn_out, w_out)
        _dsave(f"gdn{self.layer_idx}_output", output)

        # Increment forward counter after last GDN layer
        if self.layer_idx == 2:
            _dump_fwd_count[0] += 1

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

    def hf_forward(self, hidden_states, packed_seq_params):
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cu_seqlens=packed_seq_params.cu_seqlens_q,
        )
        # TP reduction for GDN's RowParallel-style out_proj.
        # When SP is on, HuggingfaceAttention.forward() handles TP reduction via reduce_scatter.
        # When SP is off, we must explicitly all-reduce the partial output.
        if not self.args.sequence_parallel and self.linear_attn.tp_size > 1:
            hidden_states = reduce_from_tensor_model_parallel_region(hidden_states)
        return hidden_states


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
        if hf_config.layer_types[layer_id + offset] == "linear_attention":
            layer_specs = copy.deepcopy(transformer_layer_spec.layer_specs[layer_id])
            layer_specs.submodules.self_attention = ModuleSpec(
                module=Attention,
                params={"args": args},
            )
            transformer_layer_spec.layer_specs[layer_id] = layer_specs
    return transformer_layer_spec
