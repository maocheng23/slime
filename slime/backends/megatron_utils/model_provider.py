# Adapt from https://github.com/NVIDIA/Megatron-LM/blob/b1efb3c7126ef7615e8c333432d76e08038e17ff/pretrain_gpt.py
import argparse
import inspect
from contextlib import nullcontext
from typing import Literal

import torch
from megatron.core import tensor_parallel
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import core_transformer_config_from_args


# Adapt from https://github.com/volcengine/verl/blob/c3b20575d2bc815fcccd84bddb4c0401fc4b632b/verl/models/llama/megatron/layers/parallel_linear.py#L82
class SGLangCompatibleOutputLayer(torch.nn.Module):
    """Output layer that matches SGLang's BF16 numerical path.

    SGLang computes: logits = matmul(hidden_states.bfloat16(), lm_head.weight.T.bfloat16())
    This layer replicates that behavior for true on-policy mode.

    This wrapper is transparent to checkpointing and parameter naming:
    - Parameters are directly registered in _parameters (not via submodule)
    - This ensures names remain compatible (e.g., "output_layer.weight" not "output_layer.original_layer.weight")
    - Gradients flow correctly and weights get updated during training
    """

    def __init__(self, original_output_layer):
        super().__init__()
        # Store original layer WITHOUT registering as submodule (avoid nested naming)
        # Use object.__setattr__ to bypass PyTorch's submodule registration
        object.__setattr__(self, '_original_layer', original_output_layer)
        
        # DEBUG: Print original layer info
        print(f"[SGLangCompatibleOutputLayer DEBUG] original_output_layer type: {type(original_output_layer)}")
        print(f"[SGLangCompatibleOutputLayer DEBUG] original_output_layer._parameters keys: {list(original_output_layer._parameters.keys())}")
        print(f"[SGLangCompatibleOutputLayer DEBUG] original_output_layer._modules keys: {list(original_output_layer._modules.keys())}")
        
        # CRITICAL FIX: Register the original layer's parameters directly in our _parameters dict
        # This ensures:
        # 1. Parameters are visible to model.parameters() and the optimizer
        # 2. Parameter names remain compatible (e.g., "weight" not "original_layer.weight")
        # 3. Gradients flow correctly because we're storing references, not copies
        for name, param in original_output_layer._parameters.items():
            if param is not None:
                self._parameters[name] = param
                print(f"[SGLangCompatibleOutputLayer DEBUG] Registered parameter: {name}, shape: {param.shape}")
        
        # Check if there are nested modules with parameters
        for module_name, module in original_output_layer._modules.items():
            for param_name, param in module._parameters.items():
                if param is not None:
                    full_name = f"{module_name}.{param_name}"
                    self._parameters[full_name] = param
                    print(f"[SGLangCompatibleOutputLayer DEBUG] Registered nested parameter: {full_name}, shape: {param.shape}")
        
        # Also copy over parameter attributes needed for TP (tensor parallel)
        for name, param in self._parameters.items():
            orig_param = original_output_layer._parameters.get(name)
            if orig_param is not None:
                for attr in ['tensor_model_parallel', 'partition_dim', 'partition_stride', 'parallel_mode', 'sequence_parallel']:
                    if hasattr(orig_param, attr):
                        setattr(param, attr, getattr(orig_param, attr))
        
        print(f"[SGLangCompatibleOutputLayer DEBUG] Final self._parameters keys: {list(self._parameters.keys())}")
        print(f"[SGLangCompatibleOutputLayer DEBUG] Wrapper initialized successfully")

    @property
    def weight(self):
        return self._parameters.get('weight')

    @property
    def bias(self):
        return self._parameters.get('bias')

    @property
    def sequence_parallel(self):
        return getattr(self._original_layer, 'sequence_parallel', False)

    def forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ) -> tuple[torch.Tensor, None]:
        # Cast input and weight to BF16 to match SGLang's computation
        # SGLang: logits = matmul(hidden_states.bfloat16(), lm_head.weight.T.bfloat16())
        # But SGLang then converts to float32: logits = logits.float()
        input_bf16 = input_.to(torch.bfloat16)
        if weight is not None:
            weight_bf16 = weight.to(torch.bfloat16)
        else:
            # When weight is None, ColumnParallelLinear uses self.weight
            # We need to cast it to BF16 to match SGLang's behavior
            if self.weight is not None:
                weight_bf16 = self.weight.to(torch.bfloat16)
            else:
                weight_bf16 = None
        output, bias = self._original_layer(input_bf16, weight=weight_bf16, runtime_gather_output=runtime_gather_output)
        # Convert to float32 to match SGLang's final logits dtype
        # SGLang computes in bfloat16 but then converts to float32
        output = output.to(torch.float32)
        return output, bias

    # State dict operations - use the parameter names directly from _parameters
    # No need for special handling since parameters are already registered correctly
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Delegate to the original layer for loading (handles TP-specific loading)
        return self._original_layer._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    # Delegate sharded_state_dict for distributed checkpointing
    def sharded_state_dict(self, *args, **kwargs):
        if hasattr(self._original_layer, 'sharded_state_dict'):
            return self._original_layer.sharded_state_dict(*args, **kwargs)
        return {}

    # Delegate any other attribute access to the original layer
    def __getattr__(self, name):
        if name == '_original_layer':
            return object.__getattribute__(self, '_original_layer')
        try:
            return getattr(self._original_layer, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class LinearForLastLayer(torch.nn.Linear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features=input_size, out_features=output_size, bias=bias)
        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel:
            self.weight.sequence_parallel = True

        self.weight.data.normal_(mean=0.0, std=0.02)
        if bias:
            self.bias.data.zero_()

    def forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ) -> tuple[torch.Tensor, None]:
        logits = super().forward(input_)
        logits = logits.float()
        if self.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(logits, tensor_parallel_output_grad=False)
        return logits, None


def get_model_provider_func(
    args: argparse.Namespace,
    role: Literal["actor", "critic"] = "actor",
):
    if args.megatron_to_hf_mode == "bridge":
        from megatron.bridge import AutoBridge

        bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
        provider = bridge.to_megatron_provider(load_weights=False)
        # TODO: we should not manually set this...
        provider.tensor_model_parallel_size = args.tensor_model_parallel_size
        provider.pipeline_model_parallel_size = args.pipeline_model_parallel_size
        provider.expert_model_parallel_size = args.expert_model_parallel_size
        provider.expert_tensor_parallel_size = args.expert_tensor_parallel_size
        provider.sequence_parallel = args.sequence_parallel
        provider.finalize()
        return provider.provide

    def model_provider(pre_process: bool = True, post_process: bool = True, vp_stage: int | None = None) -> GPTModel:
        """Builds the model.

        If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

        Args:
            pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
            post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


        Returns:
            Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
        """
        use_te = args.transformer_impl == "transformer_engine"

        # Experimental loading arguments from yaml
        config: TransformerConfig = core_transformer_config_from_args(args)

        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
            # Allow the spec to be a function so that user can use customized Megatron easier.
            if callable(transformer_layer_spec):
                transformer_layer_spec = transformer_layer_spec(args, config, vp_stage)
        else:
            if args.num_experts:
                # Define the decoder block spec
                kwargs = {
                    "use_transformer_engine": use_te,
                }
                if vp_stage is not None:
                    kwargs["vp_stage"] = vp_stage
                transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)
            else:
                # Define the decoder layer spec
                if use_te:
                    print(f"use_sglang: {args.use_sglang}, use_sglang_attention: {args.use_sglang_attention}")
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        num_experts=args.num_experts,
                        moe_grouped_gemm=args.moe_grouped_gemm,
                        qk_layernorm=args.qk_layernorm,
                        multi_latent_attention=args.multi_latent_attention,
                        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
                        use_sglang = args.use_sglang,
                        use_sglang_attention = args.use_sglang_attention,
                    )
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        num_experts=args.num_experts,
                        moe_grouped_gemm=args.moe_grouped_gemm,
                        qk_layernorm=args.qk_layernorm,
                        multi_latent_attention=args.multi_latent_attention,
                        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
                    )

        build_model_context = nullcontext
        build_model_context_args = {}
        if args.fp8_param_gather:
            try:
                from transformer_engine.pytorch import fp8_model_init

                build_model_context = fp8_model_init
                build_model_context_args["enabled"] = True

                # Check if fp8_model_init supports preserve_high_precision_init_val
                if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                    build_model_context_args["preserve_high_precision_init_val"] = True
            except Exception as e:
                raise RuntimeError(
                    "--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found."
                ) from e

        kwargs = {
            "config": config,
            "transformer_layer_spec": transformer_layer_spec,
            "vocab_size": args.padded_vocab_size,
            "max_sequence_length": args.max_position_embeddings,
            "pre_process": pre_process,
            "post_process": post_process,
            "fp16_lm_cross_entropy": args.fp16_lm_cross_entropy,
            "parallel_output": True,
            "share_embeddings_and_output_weights": not args.untie_embeddings_and_output_weights,
            "position_embedding_type": args.position_embedding_type,
            "rotary_percent": args.rotary_percent,
            "rotary_base": args.rotary_base,
            "rope_scaling": args.use_rope_scaling,
        }

        if vp_stage is not None:
            kwargs["vp_stage"] = vp_stage

        if args.mtp_num_layers:
            from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec

            mtp_kwargs = {
                "use_transformer_engine": use_te,
            }
            if vp_stage is not None:
                mtp_kwargs["vp_stage"] = vp_stage

            mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, **mtp_kwargs)
            kwargs["mtp_block_spec"] = mtp_block_spec

        with build_model_context(**build_model_context_args):
            model = GPTModel(**kwargs)

        if post_process and role == "critic":
            model.output_layer = LinearForLastLayer(input_size=config.hidden_size, output_size=1, config=config)

        # Wrap output_layer for true on-policy mode to match SGLang's BF16 numerical paths
        if post_process and role == "actor" and getattr(args, "true_on_policy_mode", False):
            model.output_layer = SGLangCompatibleOutputLayer(model.output_layer)
            print("[True On-Policy] Wrapped output_layer with SGLang-compatible BF16 layer")

        # Register tensor dump hooks for debugging (only when env var is set)
        import os
        if os.environ.get("MEGATRON_TENSOR_DUMP_DIR", ""):
            from slime.backends.megatron_utils.debug_tensor_dump import register_megatron_tensor_hooks
            register_megatron_tensor_hooks(model)
            print(f"[Debug] Registered tensor dump hooks for model (role={role})")

        return model

    return model_provider
