import math
import os

import slime.utils.external_utils.command_utils as U
from script_utils import build_debug_envs, make_system_helpers, resolve_parallel_sizes

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------
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
TP_SIZE = int(os.environ.get("SLIME_TP_SIZE", "4"))

# SGLang TP/PP override: allows different TP/PP for SGLang vs Megatron (cross-TP)
SGLANG_TP_SIZE = int(os.environ.get("SLIME_SGLANG_TP_SIZE", "0"))  # 0 = same as Megatron TP
SGLANG_PP_SIZE = int(os.environ.get("SLIME_SGLANG_PP_SIZE", "0"))  # 0 = same as Megatron PP

# --- PP add-on ---
PP_SIZE = int(os.environ.get("SLIME_PP_SIZE", "1"))
assert PP_SIZE >= 1, f"SLIME_PP_SIZE must be >= 1, got {PP_SIZE}"

WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
assert WANDB_API_KEY != "", "WANDB_API_KEY is not set"

ENABLE_CI = os.environ.get("SLIME_ENABLE_CI", "0") == "1"

USE_EXTERNAL_SYSTEM_SETUP = os.environ.get("SLIME_USE_EXTERNAL_SYSTEM_SETUP", "0") == "1"
_system_env, _system_bool, _system_int = make_system_helpers(USE_EXTERNAL_SYSTEM_SETUP)


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")
    if USE_RAW:
        U.convert_checkpoint(
            model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS, dir_dst="/root/models"
        )


def execute():
    is_debug = MODE == "debug_one_sample"

    # --- PP add-on: resolve topology with PP ---
    tensor_parallel_size, pipeline_parallel_size, data_parallel_size, gpus_per_sglang_engine = resolve_parallel_sizes(
        NUM_GPUS, USE_TP, TP_SIZE, PP_SIZE
    )
    is_pp = pipeline_parallel_size > 1

    # --- Cross-TP: SGLang can use a different TP/PP size ---
    sglang_tp_size = SGLANG_TP_SIZE if SGLANG_TP_SIZE > 0 else tensor_parallel_size
    sglang_pp_size = SGLANG_PP_SIZE if SGLANG_PP_SIZE > 0 else pipeline_parallel_size
    gpus_per_sglang_engine_actual = sglang_tp_size * sglang_pp_size

    global_batch_size = 1 if is_debug else 256
    if global_batch_size % data_parallel_size != 0:
        # Megatron requires global_batch_size divisible by micro_batch_size * data_parallel_size
        global_batch_size = math.ceil(global_batch_size / data_parallel_size) * data_parallel_size

    num_rollout = 1 if is_debug else 3000
    # --- PP add-on: skip post-train sync for multi-step debug ---
    default_skip_post_train_sync = "0"
    if is_debug and num_rollout > 1:
        default_skip_post_train_sync = "1"

    if USE_RAW:
        ckpt_args = (
            f"--hf-checkpoint /root/models/{MODEL_NAME} "
            f"--ref-load /root/models/{MODEL_NAME}_torch_dist "
        )
    else:
        ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ --ref-load /root/models/{MODEL_NAME}/ "

    wandb_args = (
        "--use-wandb "
        "--wandb-project megatron-on-policy "
        "--wandb-group qwen3-0.6B-megatron "
        f"--wandb-key {WANDB_API_KEY} "
        "--disable-wandb-random-suffix "
    )

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {num_rollout} "
        f"--rollout-batch-size {4 if is_debug else 32} "
        f"--n-samples-per-prompt {1 if is_debug else 8} "
        f"--rollout-max-response-len {2 if is_debug else 1024} "
        "--rollout-temperature 1 "
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

    recompute_args = ""
    if _system_bool("SLIME_ENABLE_RECOMPUTE", is_pp):
        recompute_args = _system_env("SLIME_RECOMPUTE_ARGS", "--recompute-activations ")

    disable_rollout_offload = _system_bool("SLIME_DISABLE_ROLLOUT_OFFLOAD", False)

    tp_args = (
        f"--tensor-model-parallel-size {tensor_parallel_size} "
        f"--pipeline-model-parallel-size {pipeline_parallel_size} "
    )

    default_sglang_mem_fraction_static = 0.2 if MODEL_NAME == "Qwen3-4B" else 0.5
    # --- PP add-on: lower mem fraction for PP ---
    if is_pp:
        default_sglang_mem_fraction_static = min(
            default_sglang_mem_fraction_static,
            0.10 if not disable_rollout_offload else 0.30,
        )
    sglang_mem_fraction_static = float(
        _system_env("SLIME_SGLANG_MEM_FRACTION_STATIC", str(default_sglang_mem_fraction_static))
    )
    sglang_args = (
        f"--rollout-num-gpus {NUM_GPUS} "
        f"--rollout-num-gpus-per-engine {gpus_per_sglang_engine_actual} "
        f"--sglang-tp-size {sglang_tp_size} "
        f"--sglang-pipeline-parallel-size {sglang_pp_size} "
        "--sglang-decode-log-interval 1000 "
        "--sglang-enable-metrics "
        f"--sglang-mem-fraction-static {sglang_mem_fraction_static} "
    )

    router_args = (
        "--router-health-check-timeout-secs 30 "
        "--router-health-failure-threshold 10 "
    )

    ci_args = ""
    if ENABLE_CI:
        ci_args = (
            "--ci-test "
            "--ci-disable-kl-checker "
            "--ci-metric-checker-key eval/gsm8k "
            "--ci-metric-checker-threshold 0.71 "
        )

    misc_args = "--megatron-to-hf-mode raw " if USE_RAW else "--megatron-to-hf-mode bridge "
    misc_args += (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS} "
    )
    misc_args += "--colocate "
    # --- PP add-on: offload & buffer settings ---
    if is_pp:
        if _system_bool("SLIME_DISABLE_TRAIN_OFFLOAD", False):
            misc_args += "--no-offload-train "
        if disable_rollout_offload:
            misc_args += "--no-offload-rollout "
        update_weight_buffer_size = _system_int("SLIME_UPDATE_WEIGHT_BUFFER_SIZE", 256 * 1024**2)
        misc_args += f"--update-weight-buffer-size {update_weight_buffer_size} "

    train_memory_margin_bytes = _system_int(
        "SLIME_TRAIN_MEMORY_MARGIN_BYTES",
        0 if is_pp and is_debug else 1024**3,
    )
    misc_args += f"--train-memory-margin-bytes {train_memory_margin_bytes} "

    if MODEL_NAME == "Qwen3-4B":
        misc_args += "--max-tokens-per-gpu 2048 "

    true_on_policy_args = (
        "--sglang-enable-deterministic-inference "
        "--sglang-rl-on-policy-target fsdp "
        "--sglang-attention-backend fa3 "
        "--use-sglang "
        "--use-sglang-attention "
        "--deterministic-mode "
        "--true-on-policy-mode "
        "--use-cpu-initialization "
        "--no-rope-fusion "
    )
    true_on_policy_envs = {
        # NOTE: Use "allreduce:Tree" instead of "Tree" to only affect AllReduce operations
        "NCCL_ALGO": "allreduce:Tree",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NCCL_NVLS_ENABLE": "0",
        "MEGATRON_USE_DETERMINISTIC_ALLREDUCE": "1",
        "ROW_LINEAR_ENABLE_INV": os.environ.get("SLIME_ROW_LINEAR_ENABLE_INV", "1"),
    }
    # --- PP add-on: stage-local sync + GPU-direct for vocab ---
    if is_pp:
        true_on_policy_envs["SLIME_TENSOR_SYNC_STAGE_LOCAL_GROUP"] = _system_env(
            "SLIME_TENSOR_SYNC_STAGE_LOCAL_GROUP", "1"
        )
        true_on_policy_envs["SLIME_TENSOR_SYNC_GPU_DIRECT_FOR_VOCAB"] = _system_env(
            "SLIME_TENSOR_SYNC_GPU_DIRECT_FOR_VOCAB", "1"
        )
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
        f"{tp_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{router_args} "
        f"{ci_args} "
        f"{misc_args} "
        f"{true_on_policy_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        extra_env_vars={
            **true_on_policy_envs,
            "SLIME_SKIP_POST_TRAIN_SYNC": _system_env("SLIME_SKIP_POST_TRAIN_SYNC", default_skip_post_train_sync),
            **build_debug_envs(MODE, _system_env),
        },
    )


if __name__ == "__main__":
    prepare()
    execute()
