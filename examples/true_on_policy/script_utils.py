"""Shared helpers for true-on-policy run scripts.

Centralises the ``_system_env`` / ``_system_bool`` / ``_system_int`` family
so that ``run_simple_megatron.py`` and ``run_moe_megatron.py`` stay DRY.
"""

import os


def make_system_helpers(use_external: bool):
    """Return ``(system_env, system_bool, system_int)`` closures.

    When *use_external* is ``False`` (the default), the helpers always return
    the hard-coded default — the script is the single source of truth.
    Set ``SLIME_USE_EXTERNAL_SYSTEM_SETUP=1`` to let environment variables
    override any default.
    """

    def system_env(name: str, default: str) -> str:
        if use_external:
            return os.environ.get(name, default)
        return default

    def system_bool(name: str, default: bool) -> bool:
        return system_env(name, "1" if default else "0") == "1"

    def system_int(name: str, default: int) -> int:
        return int(system_env(name, str(default)))

    return system_env, system_bool, system_int


def resolve_parallel_sizes(
    num_gpus: int,
    use_tp: bool,
    tp_size: int,
    pp_size: int,
    cp_size: int = 1,
) -> tuple[int, int, int, int, int]:
    """Resolve TP/PP/CP/DP and rollout-engine GPU counts under a single topology.

    Returns:
        ``(tensor_parallel_size, pipeline_parallel_size, context_parallel_size,
        data_parallel_size, gpus_per_sglang_engine)``
    """
    pipeline_parallel_size = pp_size
    tensor_parallel_size = tp_size if use_tp else 1
    context_parallel_size = cp_size
    assert tensor_parallel_size >= 1
    assert pipeline_parallel_size >= 1
    assert context_parallel_size >= 1
    model_parallel_size = tensor_parallel_size * pipeline_parallel_size * context_parallel_size
    assert num_gpus % model_parallel_size == 0, (
        f"NUM_GPUS ({num_gpus}) must be divisible by "
        f"TP*PP*CP ({tensor_parallel_size}*{pipeline_parallel_size}*{context_parallel_size})"
    )
    data_parallel_size = num_gpus // model_parallel_size
    gpus_per_sglang_engine = tensor_parallel_size * pipeline_parallel_size
    return (
        tensor_parallel_size,
        pipeline_parallel_size,
        context_parallel_size,
        data_parallel_size,
        gpus_per_sglang_engine,
    )


def build_debug_envs(mode: str, system_env) -> dict[str, str]:
    """Debug-only environment variables keyed on MODE == 'debug_one_sample'."""
    is_debug = mode == "debug_one_sample"
    return {
        "SGLANG_DUMPER_ENABLE": os.environ.get("SGLANG_DUMPER_ENABLE", "1" if is_debug else "0"),
        "SLIME_DEBUG_LAYER_DUMP": os.environ.get("SLIME_DEBUG_LAYER_DUMP", "1" if is_debug else "0"),
        "SGLANG_TEMP_UTILS_ENABLE_DEBUG_PRINT": "1" if is_debug else "0",
        "SLIME_DEBUG_ROUTER": "1" if is_debug else "0",
        "SLIME_DEBUG_ATTN": "1" if is_debug else "0",
        "SLIME_DEBUG_LOGPROB_DIFF": "1" if is_debug else "0",
        "SLIME_DEBUG_TREE_ALLREDUCE": "1" if is_debug else "0",
        "DEBUG_GRAD_ALLREDUCE": "1" if is_debug else "0",
        "DEBUG_OVERRIDE_REWARDS": "first_one" if is_debug else "",
    }
