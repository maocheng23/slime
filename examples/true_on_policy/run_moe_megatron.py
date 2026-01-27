import math
import os


import slime.utils.external_utils.command_utils as U

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

def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")
    if USE_RAW:
        U.convert_checkpoint(
            model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS, dir_dst="/root/models"
        )


def execute():
    # For MoE models, SGLang requires TP >= EP
    # Use TP_SIZE from env var, default to NUM_GPUS (so TP=EP=NUM_GPUS, DP=1)
    tensor_parallel_size = TP_SIZE if USE_TP else NUM_GPUS
    assert NUM_GPUS % tensor_parallel_size == 0, (
        f"NUM_GPUS ({NUM_GPUS}) must be divisible by tensor_parallel_size ({tensor_parallel_size})"
    )
    data_parallel_size = NUM_GPUS // tensor_parallel_size

    global_batch_size = 2 if MODE == "debug_one_sample" else 4
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
        f"--num-rollout {3 if MODE == 'debug_one_sample' else 3000} "  # Need at least 2-3 steps to observe divergence pattern
        f"--rollout-batch-size {2 if MODE == 'debug_one_sample' else 2} "
        f"--n-samples-per-prompt {2 if MODE == 'debug_one_sample' else 2} "
        f"--rollout-max-response-len {1024 if MODE == 'debug_one_sample' else 1024} "
        "--rollout-temperature 1 "
        # temp remove this to make test easier
        # "--over-sampling-batch-size 64 "
        # "--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        f"--global-batch-size {global_batch_size} "
    )

    eval_args = ""
    # if MODE == "normal":
    #     eval_args = (
    #         f"--eval-interval {2 if MODE == 'debug_one_sample' else 10} "
    #         "--eval-prompt-data gsm8k /root/datasets/gsm8k/test.parquet "
    #         "--n-samples-per-eval-prompt 1 "
    #         "--eval-max-response-len 1024 "
    #         "--eval-top-k 1 "
    #     )

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

    if MODE == "debug_one_sample":
        optimizer_args += (
            "--lr-decay-iters 4 "
        )

    

    tp_args = (
        f"--tensor-model-parallel-size {tensor_parallel_size} "
        # "--sequence-parallel "  # Disabled: only use TP without SP for easier debugging
        "--pipeline-model-parallel-size 1 "
        f"--expert-model-parallel-size {tensor_parallel_size} "  # EP = TP (SGLang requires TP >= EP)
        "--expert-tensor-parallel-size 1 "
    )
    sglang_args = (
        f"--rollout-num-gpus-per-engine {tensor_parallel_size} "
        f"--sglang-tp-size {tensor_parallel_size} "  # SGLang requires TP >= EP
        f"--sglang-ep-size {tensor_parallel_size} "  # EP = TP
        "--sglang-decode-log-interval 1000 "
        "--sglang-enable-metrics "
        f"--sglang-mem-fraction-static {0.35 if MODEL_NAME == 'Qwen3-30B-A3B' else 0.5} "
        # Disable CUDA graph for true on-policy to ensure numerical consistency
        # CUDA graph can cause non-determinism in MoE routing and expert computation
        "--sglang-disable-cuda-graph "
    )


    ci_args = (
        "--ci-test "
        "--ci-disable-kl-checker "
        "--ci-metric-checker-key eval/gsm8k "
        "--ci-metric-checker-threshold 0.71 "  # loose threshold at 60 step
    )

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
        "--colocate "
    )
    
    # Enable weight comparison check in debug mode to verify weight sync
    if MODE == "debug_one_sample":
        misc_args += "--check-weight-update-equal "
    
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
        "--true-on-policy-model qwen3_moe "
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
        # Enable deterministic all-reduce in Megatron to match SGLang's tree_all_reduce_sum
        "MEGATRON_USE_DETERMINISTIC_ALLREDUCE": "1",
        # DEBUG: Enable to get accurate CUDA error location (slows down execution significantly)
        "CUDA_LAUNCH_BLOCKING": "1",  # ENABLED: Finding the real source of CUDA illegal memory access
    }

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{wandb_args} "
        f"{tp_args} "  # TP configuration (empty if USE_TP=False)
        f"{eval_args} "
        f"{sglang_args} "
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
            "SGLANG_DUMPER_ENABLE": "1" if MODE == "debug_one_sample" else "0",
            "SGLANG_TEMP_UTILS_ENABLE_DEBUG_PRINT": "1" if MODE == "debug_one_sample" else "0",
            "SLIME_DEBUG_ROUTER": "1" if MODE == "debug_one_sample" else "0",
            "SLIME_DEBUG_ATTN": "1" if MODE == "debug_one_sample" else "0",
            "SLIME_DEBUG_LOGPROB_DIFF": "1" if MODE == "debug_one_sample" else "0",
            "SLIME_DEBUG_TREE_ALLREDUCE": "1" if MODE == "debug_one_sample" else "0",
            # Debug gradient all-reduce for MoE backward pass
            "DEBUG_GRAD_ALLREDUCE": "1" if MODE == "debug_one_sample" else "0",
            # Debug gradient sync verification - enable to check if all-reduce is working
            "DEBUG_GRAD_SYNC": "1",  # Enable to verify gradients are identical across ranks after all-reduce
            "DEBUG_ROUTER_GRAD_SYNC": "1",  # Enable to see per-rank gradient values before/after all-reduce
        },
    )


if __name__ == "__main__":
    prepare()
    execute()
