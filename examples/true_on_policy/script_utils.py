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
) -> tuple[int, int, int, int]:
    """Resolve TP/PP/DP and rollout-engine GPU counts under a single topology.

    Returns:
        ``(tensor_parallel_size, pipeline_parallel_size, data_parallel_size,
        gpus_per_sglang_engine)``
    """
    pipeline_parallel_size = pp_size
    tensor_parallel_size = tp_size if use_tp else 1
    assert tensor_parallel_size >= 1, f"tensor_parallel_size must be >= 1, got {tensor_parallel_size}"
    assert pipeline_parallel_size >= 1, f"pipeline_parallel_size must be >= 1, got {pipeline_parallel_size}"
    assert num_gpus % (tensor_parallel_size * pipeline_parallel_size) == 0, (
        f"NUM_GPUS ({num_gpus}) must be divisible by TP * PP ({tensor_parallel_size} * {pipeline_parallel_size})"
    )
    data_parallel_size = num_gpus // (tensor_parallel_size * pipeline_parallel_size)
    gpus_per_sglang_engine = tensor_parallel_size * pipeline_parallel_size
    return tensor_parallel_size, pipeline_parallel_size, data_parallel_size, gpus_per_sglang_engine


def build_tensor_sync_envs(
    pipeline_parallel_size: int,
    system_env,
) -> dict[str, str]:
    """Build the ``SLIME_TENSOR_SYNC_*`` environment variable dict.

    Three knobs are propagated here:

    * ``STAGE_LOCAL_GROUP``    — whether each PP stage syncs independently.
    * ``GPU_DIRECT``           — use per-tensor CUDA IPC globally (opt-in).
    * ``GPU_DIRECT_FOR_VOCAB`` — force per-tensor CUDA IPC for vocab/lm_head
      buckets. Required for PP>1 because SGLang broadcasts flattened-bucket
      updates to all PP stages, causing cross-device CUDA errors on vocab
      weights that are shared across stages (tied embeddings).

    The default transfer path is flattened-bucket GPU (always on GPU,
    no CPU staging). ``TensorSyncConfig.from_args()`` resolves the final
    strategy. ``GPU_DIRECT_FOR_MOE`` is still read from env by the code;
    set it directly when debugging without adding it here.
    """
    is_pp = pipeline_parallel_size > 1
    return {
        "SLIME_TENSOR_SYNC_STAGE_LOCAL_GROUP": system_env(
            "SLIME_TENSOR_SYNC_STAGE_LOCAL_GROUP", "1" if is_pp else "0"
        ),
        "SLIME_TENSOR_SYNC_GPU_DIRECT": system_env(
            "SLIME_TENSOR_SYNC_GPU_DIRECT", "0"
        ),
        # PP>1: vocab/lm_head weights are shared across PP stages (tied
        # embeddings). Flattened-bucket GPU transfer fails with cross-device
        # CUDA errors when SGLang PP0 receives tensors from PP1's GPUs.
        # Route vocab buckets through GPU-direct named-tensor IPC which
        # SGLang's load_weights path can handle cross-device.
        "SLIME_TENSOR_SYNC_GPU_DIRECT_FOR_VOCAB": system_env(
            "SLIME_TENSOR_SYNC_GPU_DIRECT_FOR_VOCAB",
            "1" if is_pp else "0",
        ),
    }


def build_debug_envs(mode: str, system_env) -> dict[str, str]:
    """Debug-only environment variables keyed on MODE == 'debug_one_sample'."""
    is_debug = mode == "debug_one_sample"
    return {
        "SGLANG_DUMPER_ENABLE": "1" if is_debug else "0",
        "SGLANG_TEMP_UTILS_ENABLE_DEBUG_PRINT": "1" if is_debug else "0",
        "SLIME_DEBUG_ROUTER": "1" if is_debug else "0",
        "SLIME_DEBUG_ATTN": "1" if is_debug else "0",
        "SLIME_DEBUG_LOGPROB_DIFF": "1" if is_debug else "0",
        "SLIME_DEBUG_TREE_ALLREDUCE": "1" if is_debug else "0",
        "DEBUG_GRAD_ALLREDUCE": "1" if is_debug else "0",
        "DEBUG_OVERRIDE_REWARDS": "first_one" if is_debug else "",
        "SLIME_DEBUG_EXTRA_WEIGHT_UPDATES": system_env("SLIME_DEBUG_EXTRA_WEIGHT_UPDATES", "0"),
    }
