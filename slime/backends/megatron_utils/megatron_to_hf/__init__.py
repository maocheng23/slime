import re

from .deepseekv3 import convert_deepseekv3_to_hf
from .glm4 import convert_glm4_to_hf
from .glm4moe import convert_glm4moe_to_hf
from .llama import convert_llama_to_hf
from .mimo import convert_mimo_to_hf
from .processors.padding_remover import remove_padding
from .processors.quantizer import quantize_params
from .qwen2 import convert_qwen2_to_hf
from .qwen3_next import convert_qwen3_next_to_hf
from .qwen3moe import convert_qwen3moe_to_hf


def _normalize_sglang_param_name(name: str) -> str:
    """
    Normalize SGLang-style parameter names to TE-compatible format.

    SGLang's SGLangLayerNormColumnParallelLinear wraps linear layers in a .linear submodule
    and norm in a .norm submodule, creating different parameter paths:
    - SGLang: linear_fc1.linear.weight -> TE: linear_fc1.weight
    - SGLang: linear_fc1.norm.weight -> TE: linear_fc1.layer_norm_weight
    - SGLang: linear_qkv.linear.weight -> TE: linear_qkv.weight

    This function normalizes these names so model-specific converters work with both backends.
    """
    # Pattern 1: Convert ".linear.weight" / ".linear.bias" to ".weight" / ".bias"
    # e.g., linear_fc1.linear.weight -> linear_fc1.weight
    name = re.sub(r'\.(linear_fc1|linear_fc2|linear_qkv|linear_proj)\.linear\.(weight|bias)',
                  r'.\1.\2', name)

    # Pattern 2: Convert ".norm.weight" to ".layer_norm_weight" for fused LayerNorm+Linear modules
    # e.g., linear_fc1.norm.weight -> linear_fc1.layer_norm_weight
    name = re.sub(r'\.(linear_fc1|linear_qkv)\.norm\.weight',
                  r'.\1.layer_norm_weight', name)

    # Pattern 3: Convert ".norm.bias" to ".layer_norm_bias" (if LayerNorm has bias)
    name = re.sub(r'\.(linear_fc1|linear_qkv)\.norm\.bias',
                  r'.\1.layer_norm_bias', name)

    return name


# TODO unify w/ `convert_to_hf`
def postprocess_hf_param(args, megatron_param_name, hf_param_name, param):
    param = remove_padding(megatron_param_name, param, args.vocab_size)
    # TODO support quant
    return param


# TODO optimize code details
def convert_to_hf(args, model_name, name, param, quantization_config=None):
    param = remove_padding(name, param, args.vocab_size)

    # Normalize SGLang-style names to TE-compatible format only when using SGLang
    if hasattr(args, 'use_sglang') and args.use_sglang:
        normalized_name = _normalize_sglang_param_name(name)
    else:
        normalized_name = name

    converted_named_tensors = _convert_to_hf_core(args, model_name, normalized_name, param)

    if not quantization_config:
        return converted_named_tensors

    return quantize_params(args, name, converted_named_tensors, quantization_config)


# TODO optimize
_cached_tensors = {}


# TODO optimize code details
def _convert_to_hf_core(args, model_name, name, param):
    if "glm4moe" in model_name:
        converted_named_tensors = convert_glm4moe_to_hf(args, name, param)
    elif "glm4" in model_name:
        converted_named_tensors = convert_glm4_to_hf(args, name, param)
    elif "qwen3moe" in model_name:
        converted_named_tensors = convert_qwen3moe_to_hf(args, name, param)
    elif "qwen3next" in model_name:
        converted_named_tensors = convert_qwen3_next_to_hf(args, name, param)
    elif "qwen2" in model_name or "qwen3" in model_name:
        converted_named_tensors = convert_qwen2_to_hf(args, name, param)
    elif "deepseekv3" in model_name:
        converted_named_tensors = convert_deepseekv3_to_hf(args, name, param)

    elif "llama" in model_name:
        converted_named_tensors = convert_llama_to_hf(args, name, param)
    elif "mimo" in model_name:
        converted_named_tensors = convert_mimo_to_hf(args, name, param)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # to compatible with sglang implementation
    if args.q_lora_rank is not None:
        old_converted_named_tensors = converted_named_tensors
        converted_named_tensors = []
        for converted_name, converted_param in old_converted_named_tensors:
            if "q_a_proj" in converted_name:
                pair_name = converted_name.replace("q_a_proj", "kv_a_proj_with_mqa")
                if pair_name in _cached_tensors:
                    converted_named_tensors += [
                        (converted_name, converted_param),
                        (pair_name, _cached_tensors[pair_name]),
                    ]
                    del _cached_tensors[pair_name]
                else:
                    _cached_tensors[converted_name] = converted_param
            elif "kv_a_proj_with_mqa" in converted_name:
                pair_name = converted_name.replace("kv_a_proj_with_mqa", "q_a_proj")
                if pair_name in _cached_tensors:
                    converted_named_tensors += [
                        (converted_name, converted_param),
                        (pair_name, _cached_tensors[pair_name]),
                    ]
                    del _cached_tensors[pair_name]
                else:
                    _cached_tensors[converted_name] = converted_param
            else:
                converted_named_tensors.append((converted_name, converted_param))
    return converted_named_tensors
