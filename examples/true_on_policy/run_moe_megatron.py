import math
import os

import slime.utils.external_utils.command_utils as U
from script_utils import build_debug_envs, make_system_helpers

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("SLIME_SCRIPT_MODEL_NAME", "Qwen3-30B-A3B")
assert MODEL_NAME in {"Qwen3-30B-A3B"}

MODEL_TYPE = os.environ.get("SLIME_SCRIPT_MODEL_TYPE", "qwen3-30B-A3B")
assert MODEL_TYPE in {"qwen3-30B-A3B"}

MODE = os.environ.get("SLIME_SCRIPT_MODE", "normal")
assert MODE in {"normal", "debug_minimal", "debug_one_sample"}

NUM_GPUS = int(os.environ.get("SLIME_SCRIPT_NUM_GPUS", "8"))
USE_RAW = os.environ.get("SLIME_USE_RAW", "1") == "1"

# TP configuration for verifying true on-policy with tensor parallelism
USE_TP = os.environ.get("SLIME_USE_TP", "0") == "1"
TP_SIZE = int(os.environ.get("SLIME_TP_SIZE", "1"))

# --- PP add-on ---
PP_SIZE = int(os.environ.get("SLIME_PP_SIZE", "1"))

WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
assert WANDB_API_KEY != "", "WANDB_API_KEY is not set"

ENABLE_CI = os.environ.get("SLIME_ENABLE_CI", "1" if MODE != "debug_one_sample" else "0") == "1"
CHECK_WEIGHT_UPDATE_EQUAL = os.environ.get("SLIME_CHECK_WEIGHT_UPDATE_EQUAL", "0") == "1"

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

    # --- PP add-on: pipeline parallel topology ---
    pipeline_parallel_size = PP_SIZE
    is_pp = pipeline_parallel_size > 1

    # For MoE models, SGLang requires TP >= EP
    # Use TP_SIZE from env var, default to NUM_GPUS (so TP=EP=NUM_GPUS, DP=1)
    tensor_parallel_size = TP_SIZE if USE_TP else (NUM_GPUS // pipeline_parallel_size)
    assert NUM_GPUS % (tensor_parallel_size * pipeline_parallel_size) == 0, (
        f"NUM_GPUS ({NUM_GPUS}) must be divisible by TP * PP ({tensor_parallel_size} * {pipeline_parallel_size})"
    )
    data_parallel_size = NUM_GPUS // (tensor_parallel_size * pipeline_parallel_size)
    expert_parallel_size = tensor_parallel_size
    gpus_per_sglang_engine = tensor_parallel_size * pipeline_parallel_size

    global_batch_size = 2 if is_debug else 128
    if global_batch_size % data_parallel_size != 0:
        # Megatron requires global_batch_size divisible by micro_batch_size * data_parallel_size
        global_batch_size = math.ceil(global_batch_size / data_parallel_size) * data_parallel_size

    if USE_RAW:
        ckpt_args = (
            f"--hf-checkpoint /root/models/{MODEL_NAME} "
            f"--ref-load /root/models/{MODEL_NAME}_torch_dist "
        )
    else:
        ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ --ref-load /root/models/{MODEL_NAME}/ "

    wandb_args = (
        "--use-wandb "
        "--wandb-project moe-on-policy "
        "--wandb-group qwen3-30B-A3B-megatron "
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
        f"--num-rollout {3 if is_debug else 3000} "
        f"--rollout-batch-size {1 if is_debug else 16} "
        f"--n-samples-per-prompt {2 if is_debug else 8} "
        f"--rollout-max-response-len {10 if is_debug else 1024} "
        "--rollout-temperature 1 "
        f"--global-batch-size {global_batch_size} "
    )

    eval_args = ""
    if MODE == "normal":
        eval_args = (
            "--skip-eval-before-train "
            f"--eval-interval 10 "
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
    if is_debug:
        optimizer_args += "--lr-decay-iters 4 "
    # --- PP add-on: CPU offload for memory savings ---
    if is_pp:
        optimizer_args += (
            "--optimizer-cpu-offload "
            "--overlap-cpu-optimizer-d2h-h2d "
            "--use-precision-aware-optimizer "
        )

    recompute_args = ""
    if _system_bool("SLIME_ENABLE_RECOMPUTE", is_pp):
        recompute_args = _system_env("SLIME_RECOMPUTE_ARGS", "--recompute-activations ")

    disable_rollout_offload = _system_bool("SLIME_DISABLE_ROLLOUT_OFFLOAD", False)

    tp_args = (
        f"--tensor-model-parallel-size {tensor_parallel_size} "
        f"--pipeline-model-parallel-size {pipeline_parallel_size} "
        f"--expert-model-parallel-size {expert_parallel_size} "
        "--expert-tensor-parallel-size 1 "
    )

    sglang_mem_fraction_static = float(
        _system_env("SLIME_SGLANG_MEM_FRACTION_STATIC", "0.35" if MODEL_NAME == "Qwen3-30B-A3B" else "0.5")
    )
    sglang_args = (
        f"--rollout-num-gpus-per-engine {gpus_per_sglang_engine} "
        f"--sglang-tp-size {tensor_parallel_size} "
        f"--sglang-pipeline-parallel-size {pipeline_parallel_size} "
        f"--sglang-ep-size {expert_parallel_size} "
        "--sglang-decode-log-interval 1000 "
        "--sglang-enable-metrics "
        f"--sglang-mem-fraction-static {sglang_mem_fraction_static} "
    )

    router_args = (
        "--router-health-check-timeout-secs 30 "
        "--router-health-failure-threshold 10 "
    )

    fault_tolerance_args = ""
    if _system_bool("SLIME_USE_FAULT_TOLERANCE", False):
        fault_tolerance_args = "--use-fault-tolerance "

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
        "--colocate "
    )
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

    if CHECK_WEIGHT_UPDATE_EQUAL:
        misc_args += "--check-weight-update-equal "

    true_on_policy_args = (
        "--sglang-enable-deterministic-inference "
        "--sglang-rl-on-policy-target fsdp "
        "--sglang-attention-backend fa3 "
        "--use-sglang "
        "--use-sglang-attention "
        "--use-sglang-router "
        "--true-on-policy-model qwen3_moe "
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
        f"{fault_tolerance_args} "
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
            **build_debug_envs(MODE, _system_env),
        },
    )


if __name__ == "__main__":
    prepare()
    execute()
