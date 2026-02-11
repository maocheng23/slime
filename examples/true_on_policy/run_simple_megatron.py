import math
import os


import slime.utils.external_utils.command_utils as U

MODEL_NAME = os.environ.get("SLIME_SCRIPT_MODEL_NAME", "Qwen3-0.6B")
assert MODEL_NAME in {"Qwen3-0.6B", "Qwen3-4B"}

MODEL_TYPE = os.environ.get("SLIME_SCRIPT_MODEL_TYPE", "qwen3-0.6B")
assert MODEL_TYPE in {"qwen3-0.6B", "qwen3-4B"}

MODE = os.environ.get("SLIME_SCRIPT_MODE", "debug_one_sample")
assert MODE in {"normal", "debug_minimal", "debug_one_sample"}

NUM_GPUS = int(os.environ.get("SLIME_SCRIPT_NUM_GPUS", "4"))

USE_RAW = os.environ.get("SLIME_USE_RAW", "1") == "1"

# TP configuration for verifying true on-policy with tensor parallelism
USE_TP = os.environ.get("SLIME_USE_TP", "1") == "1"
# PP configuration for verifying true on-policy with pipeline parallelism
PP_SIZE = int(os.environ.get("SLIME_PP_SIZE", "1"))
assert PP_SIZE >= 1, f"SLIME_PP_SIZE must be >= 1, got {PP_SIZE}"

# If TP size is not provided, default to using all GPUs within each PP stage.
TP_SIZE = int(os.environ.get("SLIME_TP_SIZE", str(max(1, NUM_GPUS // PP_SIZE))))

# CI checks are disabled by default for local debug runs.
ENABLE_CI = os.environ.get("SLIME_ENABLE_CI", "0") == "1"
# Keep script-level runtime setup authoritative by default.
# Set SLIME_USE_EXTERNAL_SYSTEM_SETUP=1 only when intentionally debugging overrides.
USE_EXTERNAL_SYSTEM_SETUP = os.environ.get("SLIME_USE_EXTERNAL_SYSTEM_SETUP", "0") == "1"


def _system_env(name: str, default: str) -> str:
    if USE_EXTERNAL_SYSTEM_SETUP:
        return os.environ.get(name, default)
    return default


def _system_bool(name: str, default: bool) -> bool:
    return _system_env(name, "1" if default else "0") == "1"


def _system_int(name: str, default: int) -> int:
    return int(_system_env(name, str(default)))


def resolve_parallel_sizes(num_gpus: int, use_tp: bool, tp_size: int, pp_size: int) -> tuple[int, int, int, int]:
    """Resolve TP/PP/DP and rollout-engine GPU counts under a single topology."""
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

def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")
    if USE_RAW:
        U.convert_checkpoint(
            model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS, dir_dst="/root/models"
        )


def execute():
    tensor_parallel_size, pipeline_parallel_size, data_parallel_size, gpus_per_sglang_engine = resolve_parallel_sizes(
        NUM_GPUS, USE_TP, TP_SIZE, PP_SIZE
    )

    global_batch_size = 1 if MODE == "debug_one_sample" else 256
    if global_batch_size % data_parallel_size != 0:
        # Megatron requires global_batch_size divisible by micro_batch_size * data_parallel_size
        global_batch_size = math.ceil(global_batch_size / data_parallel_size) * data_parallel_size

    if USE_RAW:
        ckpt_args = (
            f"--hf-checkpoint /root/models/{MODEL_NAME} "
            f"--ref-load /root/models/{MODEL_NAME}_torch_dist "
        )
    else:   
        ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/models/{MODEL_NAME}/ "

    WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
    assert WANDB_API_KEY != "", "WANDB_API_KEY is not set"

    wandb_args = (
        "--use-wandb "
        "--wandb-project megatron-on-policy "
        "--wandb-group qwen3-0.6B-megatron "
        f"--wandb-key {WANDB_API_KEY} "
        "--disable-wandb-random-suffix "
    )
    debug_num_rollout = int(os.environ.get("SLIME_DEBUG_NUM_ROLLOUT", "1"))
    normal_num_rollout = int(os.environ.get("SLIME_NORMAL_NUM_ROLLOUT", "3000"))
    default_skip_post_train_sync = "0"
    if MODE == "debug_one_sample" and debug_num_rollout > 1:
        default_skip_post_train_sync = "1"
    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {debug_num_rollout if MODE == 'debug_one_sample' else normal_num_rollout} "
        f"--rollout-batch-size {4 if MODE == 'debug_one_sample' else 32} "
        f"--n-samples-per-prompt {1 if MODE == 'debug_one_sample' else 8} "
        f"--rollout-max-response-len {2 if MODE == 'debug_one_sample' else 1024} "
        "--rollout-temperature 1 "
        # temp remove this to make test easier
        # "--over-sampling-batch-size 64 "
        # "--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        f"--global-batch-size {global_batch_size} "
    )

    eval_args = ""
    if MODE == "normal":
        eval_args = (
            "--skip-eval-before-train "
            "--eval-interval 2 "
            "--eval-prompt-data gsm8k /root/datasets/gsm8k/test.parquet "
            "--n-samples-per-eval-prompt 1 "
            "--eval-max-response-len 1024 "
            "--eval-top-k 1 "
        )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )
    enable_recompute = _system_bool("SLIME_ENABLE_RECOMPUTE", pipeline_parallel_size > 1)
    recompute_args = ""
    if enable_recompute:
        recompute_args = _system_env("SLIME_RECOMPUTE_ARGS", "--recompute-activations ")

    disable_rollout_offload = _system_bool("SLIME_DISABLE_ROLLOUT_OFFLOAD", False)

    tp_args = (
        f"--tensor-model-parallel-size {tensor_parallel_size} "
        # "--sequence-parallel "  # Disabled: only use TP without SP for easier debugging
        f"--pipeline-model-parallel-size {pipeline_parallel_size} "
    )
    default_sglang_mem_fraction_static = 0.2 if MODEL_NAME == "Qwen3-4B" else 0.5
    if pipeline_parallel_size > 1:
        default_sglang_mem_fraction_static = min(
            default_sglang_mem_fraction_static,
            0.10 if not disable_rollout_offload else 0.30,
        )
    sglang_mem_fraction_static = float(
        _system_env("SLIME_SGLANG_MEM_FRACTION_STATIC", str(default_sglang_mem_fraction_static))
    )
    sglang_args = (
        f"--rollout-num-gpus {NUM_GPUS} "
        f"--rollout-num-gpus-per-engine {gpus_per_sglang_engine} "
        f"--sglang-tp-size {tensor_parallel_size} "
        f"--sglang-pipeline-parallel-size {pipeline_parallel_size} "
        "--sglang-decode-log-interval 1000 "
        "--sglang-enable-metrics "
        f"--sglang-mem-fraction-static {sglang_mem_fraction_static} "
        # Disable CUDA graph for true on-policy to ensure numerical consistency
        "--sglang-disable-cuda-graph "
    )
    router_args = (
        "--router-health-check-timeout-secs 30 "
        "--router-health-failure-threshold 10 "
    )


    ci_args = (
        "--ci-test "
        "--ci-disable-kl-checker "
        "--ci-metric-checker-key eval/gsm8k "
        "--ci-metric-checker-threshold 0.71 "  # loose threshold at 60 step
    ) if ENABLE_CI else ""

    if USE_RAW:
        misc_args = "--megatron-to-hf-mode raw "
    else:
        misc_args = "--megatron-to-hf-mode bridge "

    misc_args += (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS} "
    )
    # PP distributed weight sync uses custom process groups. In colocate mode, train and rollout
    # ranks can share the same physical GPU, which is invalid for NCCL communicators.
    if pipeline_parallel_size == 1:
        misc_args += "--colocate "
    # For PP runs, keep train offload enabled by default to lower first-step
    # memory pressure before update_weights/optimizer-state allocations.
    # Opt out with SLIME_DISABLE_TRAIN_OFFLOAD=1 if needed for targeted debugging.
    if pipeline_parallel_size > 1:
        disable_train_offload = _system_bool(
            "SLIME_DISABLE_TRAIN_OFFLOAD",
            False,
        )
        if disable_train_offload:
            misc_args += "--no-offload-train "
        if disable_rollout_offload:
            misc_args += "--no-offload-rollout "
        update_weight_buffer_size = _system_int(
            "SLIME_UPDATE_WEIGHT_BUFFER_SIZE",
            256 * 1024**2,
        )
        misc_args += f"--update-weight-buffer-size {update_weight_buffer_size} "

    default_train_memory_margin_bytes = 1024**3
    if pipeline_parallel_size > 1 and MODE == "debug_one_sample":
        default_train_memory_margin_bytes = 0
    train_memory_margin_bytes = _system_int("SLIME_TRAIN_MEMORY_MARGIN_BYTES", default_train_memory_margin_bytes)
    misc_args += f"--train-memory-margin-bytes {train_memory_margin_bytes} "
    
    if MODEL_NAME == "Qwen3-4B":
        misc_args += (
            #"--use-dynamic-batch-size "
            # TODO pick a good value
            "--max-tokens-per-gpu 2048 "
        )

    true_on_policy_args = (
        "--sglang-enable-deterministic-inference "
        "--sglang-rl-on-policy-target fsdp "
        "--sglang-attention-backend fa3 "
        "--use-sglang "
        "--use-sglang-attention "
        "--use-sglang-router "
        "--deterministic-mode "
        "--true-on-policy-mode "
        "--use-cpu-initialization "
        "--no-rope-fusion "
    )
    true_on_policy_envs = {
        # NOTE: Use "allreduce:Tree" instead of "Tree" to only affect AllReduce operations
        # "Tree" would affect ALL NCCL operations (AllGather, ReduceScatter, etc.) and may cause errors
        # like "no algorithm/protocol available for function AllGather with datatype ncclInt8"
        "NCCL_ALGO": "allreduce:Tree",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        # Disable NVLS (NVLink SHARP) to ensure consistent all-reduce behavior between sglang and megatron
        "NCCL_NVLS_ENABLE": "0",
        # Use deterministic tree all-reduce (all_gather + local tree sum) instead of NCCL all_reduce
        # This ensures SGLang and Megatron use identical all-reduce implementation for true on-policy
        "MEGATRON_USE_DETERMINISTIC_ALLREDUCE": "1",
        "SLIME_TENSOR_SYNC_STAGE_LOCAL_GROUP": _system_env(
            "SLIME_TENSOR_SYNC_STAGE_LOCAL_GROUP",
            "1" if pipeline_parallel_size > 1 else "0",
        ),
        "SLIME_TENSOR_SYNC_CPU_STAGING": _system_env(
            "SLIME_TENSOR_SYNC_CPU_STAGING",
            "0" if pipeline_parallel_size > 1 else "1",
        ),
        "SLIME_TENSOR_SYNC_GPU_DIRECT": _system_env("SLIME_TENSOR_SYNC_GPU_DIRECT", "0"),
        "SLIME_TENSOR_SYNC_GPU_DIRECT_ALLOW_PP": _system_env(
            "SLIME_TENSOR_SYNC_GPU_DIRECT_ALLOW_PP",
            "0" if pipeline_parallel_size > 1 else "1",
        ),
        "SLIME_TENSOR_SYNC_FORCE_SAFE_PP": _system_env(
            "SLIME_TENSOR_SYNC_FORCE_SAFE_PP",
            "1" if pipeline_parallel_size > 1 else "0",
        ),
        "SLIME_TENSOR_SYNC_GPU_BUCKET_ALLOW_PP": _system_env(
            "SLIME_TENSOR_SYNC_GPU_BUCKET_ALLOW_PP",
            "1" if pipeline_parallel_size > 1 else "0",
        ),
        "SLIME_TENSOR_SYNC_GPU_DIRECT_FOR_VOCAB": _system_env(
            "SLIME_TENSOR_SYNC_GPU_DIRECT_FOR_VOCAB",
            "1" if pipeline_parallel_size > 1 else "0",
        ),
        "SLIME_TENSOR_SYNC_GPU_DIRECT_FOR_MOE": _system_env(
            "SLIME_TENSOR_SYNC_GPU_DIRECT_FOR_MOE",
            "0",
        ),
    }
    if pipeline_parallel_size > 1:
        alloc_conf = _system_env("PYTORCH_CUDA_ALLOC_CONF", "")
        if alloc_conf:
            true_on_policy_envs["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf
        elif disable_rollout_offload:
            true_on_policy_envs["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{recompute_args} "
        f"{grpo_args} "
        f"{wandb_args} "
        f"{tp_args} "  # TP configuration (empty if USE_TP=False)
        f"{eval_args} "
        f"{sglang_args} "
        f"{router_args} "
        f"{ci_args} "
        f"{misc_args} "
        f"{true_on_policy_args} "
    )
    
    U.execute_train(
        train_args=train_args,
        # PP>1 runs disable colocate; reserve GPUs for both training and rollout workers.
        num_gpus_per_node=NUM_GPUS if pipeline_parallel_size == 1 else NUM_GPUS * 2,
        megatron_model_type=MODEL_TYPE,
        extra_env_vars={
            **true_on_policy_envs,
            "SLIME_SKIP_POST_TRAIN_SYNC": _system_env(
                "SLIME_SKIP_POST_TRAIN_SYNC",
                default_skip_post_train_sync,
            ),
            "SGLANG_DUMPER_ENABLE": "1" if MODE == "debug_one_sample" else "0",
            "SGLANG_TEMP_UTILS_ENABLE_DEBUG_PRINT": "1" if MODE == "debug_one_sample" else "0",
            "SLIME_DEBUG_ROUTER": "1" if MODE == "debug_one_sample" else "0",
            "SLIME_DEBUG_ATTN": "1" if MODE == "debug_one_sample" else "0",
            "SLIME_DEBUG_LOGPROB_DIFF": "1" if MODE == "debug_one_sample" else "0",
            "SLIME_DEBUG_TREE_ALLREDUCE": "1" if MODE == "debug_one_sample" else "0",
            "DEBUG_GRAD_ALLREDUCE": "1" if MODE == "debug_one_sample" else "0",
            "DEBUG_OVERRIDE_REWARDS": "first_one" if MODE == "debug_one_sample" else "",
            "SLIME_DEBUG_EXTRA_WEIGHT_UPDATES": _system_env("SLIME_DEBUG_EXTRA_WEIGHT_UPDATES", "0"),
        },
    )


if __name__ == "__main__":
    prepare()
    execute()
