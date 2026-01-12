import os


import slime.utils.external_utils.command_utils as U

MODEL_NAME = os.environ.get("SLIME_SCRIPT_MODEL_NAME", "Qwen3-30B-A3B")
assert MODEL_NAME in {"Qwen3-0.6B", "Qwen3-4B", "Qwen3-30B-A3B"}

MODEL_TYPE = os.environ.get("SLIME_SCRIPT_MODEL_TYPE", "qwen3-30B-A3B")
assert MODEL_TYPE in {"qwen3-0.6B", "qwen3-4B", "qwen3-30B-A3B"}

MODE = os.environ.get("SLIME_SCRIPT_MODE", "debug_one_sample")
assert MODE in {"normal", "debug_minimal", "debug_one_sample"}

NUM_GPUS = int(os.environ.get("SLIME_SCRIPT_NUM_GPUS", "4"))

USE_NO_PARALLELISM = os.environ.get("SLIME_USE_NO_PARALLELISM", "1") == "1"

HARDWARE = os.environ.get("SLIME_SCRIPT_HARDWARE", "H200")
assert HARDWARE in {"H100", "H200", "GB200", "GB300"}

USE_RAW = os.environ.get("SLIME_USE_RAW", "1") == "1"

def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")
    if USE_RAW:
        U.convert_checkpoint(
            model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS, dir_dst="/root/models"
        )


def execute():
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
        f"--wandb-group {MODEL_NAME.lower()}-megatron "
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
        f"--num-rollout {2 if MODE == 'debug_one_sample' else 3000} "
        f"--rollout-batch-size {1 if MODE == 'debug_one_sample' else 32} "
        f"--n-samples-per-prompt {1 if MODE == 'debug_one_sample' else 8} "
        f"--rollout-max-response-len {2 if MODE == 'debug_one_sample' else 8192} "
        "--rollout-temperature 1 "
        # temp remove this to make test easier
        # "--over-sampling-batch-size 64 "
        # "--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        f"--global-batch-size {1 if MODE == 'debug_one_sample' else 256} "
        "--balance-data "
    )

    eval_args = ""
    if MODE == "normal":
        eval_args = (
            "--eval-interval 20 "
            "--eval-prompt-data gsm8k /root/datasets/gsm8k/test.parquet "
            "--n-samples-per-eval-prompt 16 "
            "--eval-max-response-len 16384 "
            "--eval-top-p 1 "
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

    perf_args = (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 32768 "
    )
    
    # MoE parallel configuration based on hardware
    if MODEL_NAME == "Qwen3-30B-A3B":
        # Check if user wants no parallelism (simple MoE)
        
        
        if USE_NO_PARALLELISM:
            # Simple MoE with no parallelism - all parallel sizes = 1
            # This requires NUM_GPUS = 1 for data_parallel_size = 1
            perf_args += (
                "--tensor-model-parallel-size 1 "
                "--pipeline-model-parallel-size 1 "
                "--context-parallel-size 1 "
                "--expert-model-parallel-size 1 "
                "--expert-tensor-parallel-size 1 "
            )
            sglang_args = (
                "--rollout-num-gpus-per-engine 1 "
                "--sglang-decode-log-interval 1000 "
                "--sglang-enable-metrics "
                "--sglang-mem-fraction-static 0.7 "
                # Disable CUDA graph for true on-policy to ensure numerical consistency
                f"{'--sglang-disable-cuda-graph ' if MODE == 'debug_one_sample' else '--sglang-cuda-graph-max-bs 512 '}"
            )
        else:
            match (HARDWARE, int(os.environ.get("SLIME_SCRIPT_NUM_NODES", "1"))):
                case ("H100", 1) | ("H200", 1):
                    # For 4 GPUs: tensor_parallel=4, so data_parallel=1
                    # expert_model_parallel_size must be <= 4 (number of GPUs)
                    # Using 1 to keep it simple for 4 GPU setup
                    perf_args += (
                        "--tensor-model-parallel-size 4 "
                        "--sequence-parallel "
                        "--pipeline-model-parallel-size 1 "
                        "--context-parallel-size 1 "
                        "--expert-model-parallel-size 1 "
                        "--expert-tensor-parallel-size 1 "
                    )
                    sglang_args = (
                        f"--rollout-num-gpus-per-engine {NUM_GPUS} "
                        "--sglang-decode-log-interval 1000 "
                        "--sglang-enable-metrics "
                        "--sglang-mem-fraction-static 0.7 "
                        # Disable CUDA graph for true on-policy to ensure numerical consistency
                        f"{'--sglang-disable-cuda-graph ' if MODE == 'debug_one_sample' else '--sglang-cuda-graph-max-bs 512 '}"
                    )
            case ("GB200", 1) | ("GB300", 1) | ("GB200", 2) | ("GB300", 2) | ("GB200", 4) | ("GB300", 4):
                perf_args += (
                    "--tensor-model-parallel-size 4 "
                    "--sequence-parallel "
                    "--pipeline-model-parallel-size 1 "
                    "--context-parallel-size 1 "
                    "--expert-model-parallel-size 4 "
                    "--expert-tensor-parallel-size 1 "
                )
                sglang_args = (
                    "--rollout-num-gpus-per-engine 4 "
                    "--sglang-decode-log-interval 1000 "
                    "--sglang-enable-metrics "
                    "--sglang-mem-fraction-static 0.7 "
                    # Note: true_on_policy_args will set --sglang-attention-backend fa3
                    # Disable CUDA graph for true on-policy to ensure numerical consistency
                    f"{'--sglang-disable-cuda-graph ' if MODE == 'debug_one_sample' else '--sglang-cuda-graph-max-bs 512 '}"
                )
            case _:
                raise NotImplementedError(f"Hardware {HARDWARE} with current node count not supported")
    else:
        sglang_args = (
            "--rollout-num-gpus-per-engine 1 "
            "--sglang-decode-log-interval 1000 "
            "--sglang-enable-metrics "
            f"--sglang-mem-fraction-static {0.2 if MODEL_NAME == 'Qwen3-4B' else 0.4} "
            # Disable CUDA graph for true on-policy to ensure numerical consistency
            f"{'--sglang-disable-cuda-graph ' if MODE == 'debug_one_sample' else ''}"
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
        # should be good for model performance
        f"--actor-num-nodes {int(os.environ.get('SLIME_SCRIPT_NUM_NODES', '1'))} "
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        f"--num-gpus-per-node {NUM_GPUS} "
        "--colocate "
    )
    
    if MODEL_NAME == "Qwen3-4B":
        misc_args += (
            #"--use-dynamic-batch-size "
            # TODO pick a good value
            "--max-tokens-per-gpu 2048 "
        )
    
    if MODEL_NAME == "Qwen3-30B-A3B":
        # Add optimizer CPU offload for large models
        optimizer_args += (
            "--optimizer-cpu-offload "
            "--overlap-cpu-optimizer-d2h-h2d "
            "--use-precision-aware-optimizer "
        )

    true_on_policy_args = (
        "--sglang-enable-deterministic-inference "
        "--sglang-rl-on-policy-target fsdp "
        "--sglang-attention-backend fa3 "
        "--use-sglang "
        "--use-sglang-attention "
        "--use-sglang-router "  # Use SGLang's fused MoE router for true on-policy
        "--deterministic-mode "
        "--true-on-policy-mode "
        "--use-cpu-initialization "
        "--no-rope-fusion "
    )
    true_on_policy_envs = {
        # TODO note: "Ring" in original RL PR, "allreduce:tree" in SGLang
        "NCCL_ALGO": "Ring",
        # "NCCL_ALGO": "allreduce:tree",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{wandb_args} "
        f"{perf_args if MODEL_NAME == 'Qwen3-30B-A3B' else ''} "
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
        },
    )


if __name__ == "__main__":
    prepare()
    execute()
