import os


import slime.utils.external_utils.command_utils as U

MODEL_NAME = os.environ.get("SLIME_SCRIPT_MODEL_NAME", "Qwen3-0.6B")
assert MODEL_NAME in {"Qwen3-0.6B", "Qwen3-4B"}

# Force debug_one_sample mode for dumper testing
MODE = os.environ.get("SLIME_SCRIPT_MODE", "debug_one_sample")
assert MODE in {"normal", "debug_minimal", "debug_one_sample"}

NUM_GPUS = int(os.environ.get("SLIME_SCRIPT_NUM_GPUS", "1"))


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} "

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
        f"--rollout-max-response-len {2 if MODE == 'debug_one_sample' else 1024} "
        "--rollout-temperature 0.8 "
        # temp remove this to make test easier
        # "--over-sampling-batch-size 64 "
        # "--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        f"--global-batch-size {1 if MODE == 'debug_one_sample' else 256} "
    )

    eval_args = ""
    if MODE == "normal":
        eval_args = (
            "--eval-interval 20 "
            "--eval-prompt-data gsm8k /root/datasets/gsm8k/test.parquet "
            "--n-samples-per-eval-prompt 1 "
            "--eval-max-response-len 1024 "
            "--eval-top-k 1 "
        )

    grpo_args = (
        "--advantage-estimator grpo "
        # "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
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

    # Tensor dump directories for debug_one_sample mode
    sglang_tensor_dump_dir = "/tmp/sglang_tensor_dump" if MODE == "debug_one_sample" else ""
    fsdp_tensor_dump_dir = "/tmp/fsdp_tensor_dump" if MODE == "debug_one_sample" else ""

    # Build dump layers list - always include last layer (layer 28 for Qwen3 models)
    dump_layers_str = ""
    if MODE == "debug_one_sample":
        # Default layers to dump: 0, 1, 2
        dump_layers = [0, 1, 2]
        # Automatically add last layer (layer 28 for Qwen3-0.6B and Qwen3-4B)
        last_layer = 28
        if last_layer not in dump_layers:
            dump_layers.append(last_layer)
        dump_layers_str = f"--sglang-debug-tensor-dump-layers {' '.join(map(str, dump_layers))} "

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-decode-log-interval 1000 "
        "--sglang-enable-metrics "
        f"--sglang-mem-fraction-static {0.2 if MODEL_NAME == 'Qwen3-4B' else 0.4} "
        f"{'--sglang-disable-cuda-graph ' if MODE == 'debug_one_sample' else ''}"
        # Enable tensor dump for layer-by-layer comparison
        f"{'--sglang-debug-tensor-dump-output-folder ' + sglang_tensor_dump_dir + ' ' if MODE == 'debug_one_sample' else ''}"
        f"{dump_layers_str}"  # Includes last layer (28) automatically
    )

    fsdp_args = (
        # Set to true for FULL_STATE_DICT mode, false for SHARDED_STATE_DICT mode (default)
        # "--fsdp-full-params "  # Uncomment this line to enable full params mode
        # Set the bucket size for weight update
        "--update-weight-buffer-size 536870912 "  # 512MB
    )

    ci_args = (
        "--ci-test "
        "--ci-disable-kl-checker "
        "--ci-metric-checker-key eval/gsm8k "
        "--ci-metric-checker-threshold 0.71 "  # loose threshold at 60 step
    )

    misc_args = "--actor-num-nodes 1 " f"--actor-num-gpus-per-node {NUM_GPUS} " "--colocate " "--train-backend fsdp "

    if MODEL_NAME == "Qwen3-4B":
        misc_args += (
            "--use-dynamic-batch-size "
            # TODO pick a good value
            "--max-tokens-per-gpu 2048 "
        )

    true_on_policy_args = (
        "--sglang-enable-deterministic-inference "
        "--sglang-rl-on-policy-target fsdp "
        "--sglang-attention-backend fa3 "
        "--attn-implementation flash_attention_3 "
        "--deterministic-mode "
        "--true-on-policy-mode "
    )
    true_on_policy_envs = {
        # TODO note: "Ring" in original RL PR, "allreduce:tree" in SGLang
        # "NCCL_ALGO": "Ring",
        "NCCL_ALGO": "allreduce:tree",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{sglang_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{eval_args} "
        f"{fsdp_args} "
        f"{ci_args} "
        f"{misc_args} "
        f"{true_on_policy_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=None,
        extra_env_vars={
            **true_on_policy_envs,
            "SGLANG_DUMPER_ENABLE": "1" if MODE == "debug_one_sample" else "0",
            "SGLANG_TEMP_UTILS_ENABLE_DEBUG_PRINT": "1" if MODE == "debug_one_sample" else "0",
            # FSDP tensor dump for layer-by-layer comparison
            "FSDP_TENSOR_DUMP_DIR": fsdp_tensor_dump_dir if MODE == "debug_one_sample" else "",
            "FSDP_TENSOR_DUMP_LAYERS": "0,1,2" if MODE == "debug_one_sample" else "",  # Dump first 3 layers
        },
    )


if __name__ == "__main__":
    prepare()
    execute()

