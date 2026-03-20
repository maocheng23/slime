import math
import os

import slime.utils.external_utils.command_utils as U
from script_utils import build_debug_envs, make_system_helpers

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("SLIME_SCRIPT_MODEL_NAME", "Qwen3-Next-4layer")

MODEL_TYPE = os.environ.get("SLIME_SCRIPT_MODEL_TYPE", "qwen3-next-4layer")

MODE = os.environ.get("SLIME_SCRIPT_MODE", "debug_one_sample")
assert MODE in {"normal", "debug_minimal", "debug_one_sample"}

NUM_GPUS = int(os.environ.get("SLIME_SCRIPT_NUM_GPUS", "8"))
USE_RAW = os.environ.get("SLIME_USE_RAW", "1") == "1"

USE_TP = os.environ.get("SLIME_USE_TP", "0") == "1"
TP_SIZE = int(os.environ.get("SLIME_TP_SIZE", "1"))

PP_SIZE = int(os.environ.get("SLIME_PP_SIZE", "1"))
assert PP_SIZE >= 1, f"SLIME_PP_SIZE must be >= 1, got {PP_SIZE}"

EP_SIZE = int(os.environ.get("SLIME_EP_SIZE", "0"))  # 0 = follow TP_SIZE

SGLANG_TP_SIZE = int(os.environ.get("SLIME_SGLANG_TP_SIZE", "0"))
SGLANG_EP_SIZE = int(os.environ.get("SLIME_SGLANG_EP_SIZE", "0"))
SGLANG_PP_SIZE = int(os.environ.get("SLIME_SGLANG_PP_SIZE", "0"))

WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
if MODE != "debug_one_sample":
    assert WANDB_API_KEY != "", "WANDB_API_KEY is not set"

ENABLE_CI = os.environ.get("SLIME_ENABLE_CI", "0") == "1"
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

    pipeline_parallel_size = PP_SIZE
    is_pp = pipeline_parallel_size > 1

    tensor_parallel_size = TP_SIZE if USE_TP else (NUM_GPUS // pipeline_parallel_size)
    assert NUM_GPUS % (tensor_parallel_size * pipeline_parallel_size) == 0, (
        f"NUM_GPUS ({NUM_GPUS}) must be divisible by TP * PP ({tensor_parallel_size} * {pipeline_parallel_size})"
    )
    data_parallel_size = NUM_GPUS // (tensor_parallel_size * pipeline_parallel_size)
    expert_parallel_size = EP_SIZE if EP_SIZE > 0 else tensor_parallel_size

    sglang_tp_size = SGLANG_TP_SIZE if SGLANG_TP_SIZE > 0 else tensor_parallel_size
    sglang_ep_size = SGLANG_EP_SIZE if SGLANG_EP_SIZE > 0 else expert_parallel_size
    sglang_pp_size = SGLANG_PP_SIZE if SGLANG_PP_SIZE > 0 else pipeline_parallel_size
    gpus_per_sglang_engine = sglang_tp_size * sglang_pp_size

    global_batch_size = 2 if is_debug else 128
    if global_batch_size % data_parallel_size != 0:
        global_batch_size = math.ceil(global_batch_size / data_parallel_size) * data_parallel_size

    if USE_RAW:
        ckpt_args = (
            f"--hf-checkpoint /root/models/{MODEL_NAME} "
            f"--ref-load /root/models/{MODEL_NAME}_torch_dist "
        )
    else:
        ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ --ref-load /root/models/{MODEL_NAME}/ "

    if WANDB_API_KEY:
        wandb_args = (
            "--use-wandb "
            "--wandb-project qwen3-next-on-policy "
            "--wandb-group qwen3-next-4layer-megatron "
            f"--wandb-key {WANDB_API_KEY} "
            "--disable-wandb-random-suffix "
        )
    else:
        wandb_args = ""

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
        _system_env("SLIME_SGLANG_MEM_FRACTION_STATIC", "0.15" if is_debug else "0.35")
    )
    sglang_args = (
        f"--rollout-num-gpus-per-engine {gpus_per_sglang_engine} "
        f"--sglang-tp-size {sglang_tp_size} "
        f"--sglang-pipeline-parallel-size {sglang_pp_size} "
        f"--sglang-ep-size {sglang_ep_size} "
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
        "--sglang-rl-on-policy-target fsdp_tp "
        "--sglang-attention-backend fa3 "
        "--use-sglang "
        "--use-sglang-attention "
        "--use-sglang-router "
        "--true-on-policy-model qwen3_next "
        "--deterministic-mode "
        "--true-on-policy-mode "
        "--recompute-logprobs-via-prefill "
        "--use-cpu-initialization "
        "--no-rope-fusion "
    )
    true_on_policy_envs = {
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "MEGATRON_USE_DETERMINISTIC_ALLREDUCE": "1",
        "MEGATRON_DETERMINISTIC_FORWARD_ONLY": "1",
        "MEGATRON_ROPE_BF16": "1",
    }
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
            "SLIME_PROFILE_FWD_BWD": os.environ.get("SLIME_PROFILE_FWD_BWD", "0"),
            "SLIME_PROFILE_BWD_DETAIL": os.environ.get("SLIME_PROFILE_BWD_DETAIL", "0"),
            "USE_TRITON_BACKWARD": os.environ.get("USE_TRITON_BACKWARD", "0"),
            "ROW_LINEAR_ENABLE_INV": os.environ.get("SLIME_ROW_LINEAR_ENABLE_INV", "1"),
            "SLIME_DEBUG_LAYER_DUMP": os.environ.get("SLIME_DEBUG_LAYER_DUMP", "0"),
            "SGLANG_DEBUG_LAYER_DUMP": os.environ.get("SGLANG_DEBUG_LAYER_DUMP", "0"),
            "SGLANG_DUMP_MAX_FWD": os.environ.get("SGLANG_DUMP_MAX_FWD", "10"),
            "SLIME_DEBUG_DUMP_MAX_FWD": os.environ.get("SLIME_DEBUG_DUMP_MAX_FWD", "10"),
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
        },
    )


if __name__ == "__main__":
    prepare()
    execute()
