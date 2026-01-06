import os


import slime.utils.external_utils.command_utils as U

MODEL_NAME = os.environ.get("SLIME_SCRIPT_MODEL_NAME", "Qwen3-0.6B")
assert MODEL_NAME in {"Qwen3-0.6B", "Qwen3-4B"}

MODEL_TYPE = os.environ.get("SLIME_SCRIPT_MODEL_TYPE", "qwen3-0.6B")
assert MODEL_TYPE in {"qwen3-0.6B", "qwen3-4B"}

MODE = os.environ.get("SLIME_SCRIPT_MODE", "debug_one_sample")
assert MODE in {"normal", "debug_minimal", "debug_one_sample"}

NUM_GPUS = int(os.environ.get("SLIME_SCRIPT_NUM_GPUS", "1"))


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")
    U.convert_checkpoint(
        model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS, dir_dst="/root/models"
    )


def execute():
    ckpt_args = (
        f"--hf-checkpoint /root/models/{MODEL_NAME} "
        f"--load /root/models/{MODEL_NAME}_torch_dist "
    )

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {1 if MODE == 'debug_one_sample' else 3000} "
        f"--rollout-batch-size {1 if MODE == 'debug_one_sample' else 32} "
        f"--n-samples-per-prompt {1 if MODE == 'debug_one_sample' else 8} "
        f"--rollout-max-response-len {1024 if MODE == 'debug_one_sample' else 1024} "
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

    # Tensor dump directory for debug_one_sample mode
    tensor_dump_dir = "/tmp/sglang_tensor_dump" if MODE == "debug_one_sample" else ""
    
    # Build dump layers list - always include last layer (layer 28 for Qwen3 models)
    dump_layers_str = ""
    if MODE == "debug_one_sample":
        # Default layers to dump: 0, 1, 2
        dump_layers = [0, 1, 2]
        # Automatically add last layer (layer 28 for Qwen3-0.6B and Qwen3-4B)
        last_layer = 27
        if last_layer not in dump_layers:
            dump_layers.append(last_layer)
        dump_layers_str = f"--sglang-debug-tensor-dump-layers {' '.join(map(str, dump_layers))} "
    
    # NOTE: For true on-policy mode, CUDA graph should be disabled to ensure
    # bitwise identical logprobs between SGLang and Megatron.
    # CUDA graph can cause subtle numerical differences due to:
    # - Different operation fusion patterns
    # - Different execution order in async execution
    # - Memory layout differences
    disable_cuda_graph_for_true_on_policy = os.environ.get(
        "SLIME_DISABLE_CUDA_GRAPH_FOR_TRUE_ON_POLICY", "1"
    ) == "1"
    
    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-decode-log-interval 1000 "
        "--sglang-enable-metrics "
        f"--sglang-mem-fraction-static {0.2 if MODEL_NAME == 'Qwen3-4B' else 0.4} "
        # Disable CUDA graph for true on-policy to ensure numerical consistency
        f"{'--sglang-disable-cuda-graph ' if (MODE == 'debug_one_sample' or disable_cuda_graph_for_true_on_policy) else ''}"
        # Enable tensor dump for layer-by-layer comparison
        f"{'--sglang-debug-tensor-dump-output-folder ' + tensor_dump_dir + ' ' if MODE == 'debug_one_sample' else ''}"
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

    misc_args = "--actor-num-nodes 1 " f"--actor-num-gpus-per-node {NUM_GPUS} " "--colocate " "--train-backend megatron "

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
        "--use-sglang "
        "--use-sglang-attention "
        "--deterministic-mode "
        "--true-on-policy-mode "
        # NOTE: fa3 backend already uses num_splits=1 when enable_deterministic_inference=True
        # (see flashattention_backend.py:367-372)
        # The triton_attention_split_tile_size is only for triton backend, not fa3
        #
        # For true on-policy consistency with SGLang:
        # 1. --use-cpu-initialization: Initialize RoPE inv_freq on CPU (matches SGLang's behavior)
        #    SGLang computes inv_freq on CPU for numerical stability when rl_on_policy_target is set
        # 2. --no-rope-fusion: Disable TE's fused RoPE kernel, use native implementation
        #    This ensures the same RoPE computation path as SGLang's forward_native
        "--use-cpu-initialization "
        "--no-rope-fusion "
    )
    true_on_policy_envs = {
        # TODO note: "Ring" in original RL PR, "allreduce:tree" in SGLang
        "NCCL_ALGO": "Ring",
        # "NCCL_ALGO": "allreduce:tree",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        # Disable chunked logprobs processing to ensure bitwise identical results
        # NOTE: In decode phase, each request processes one token, so pruned_states.shape[0] = batch_size
        # This is usually < 2048, so chunked processing wouldn't trigger anyway.
        # However, in prefill phase (especially with chunked prefill), multiple tokens may be processed,
        # which could trigger chunked logprobs processing and cause numerical differences.
        # By disabling it, we ensure consistent processing path across all phases.
        #
        # How it works:
        # - In logits_processor.py, should_skip_chunking = not enable_logprobs_chunk OR ...
        # - Setting enable_logprobs_chunk=False ensures should_skip_chunking=True
        # - This forces all tokens to use non-chunked path (single _get_logits call)
        # - Result: Consistent computation order → bitwise identical results
        "SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK": "False",
        # Set a very large chunk size as a safety measure (even if enabled, won't trigger)
        # Default is 2048, so 999999 ensures chunking never triggers
        "SGLANG_LOGITS_PROCESSER_CHUNK_SIZE": "999999",
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

    # Enable debug logging for logprob comparison (set via env var)
    debug_logprob_diff = os.environ.get("SLIME_DEBUG_LOGPROB_DIFF", "0")
    
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        extra_env_vars={
            **true_on_policy_envs,
            "SGLANG_DUMPER_ENABLE": "1" if MODE == "debug_one_sample" else "0",
            "SGLANG_TEMP_UTILS_ENABLE_DEBUG_PRINT": "1" if MODE == "debug_one_sample" else "0",
            # Megatron tensor dump for layer-by-layer comparison
            # Automatically includes last layer (28) - see debug_tensor_dump.py
            "MEGATRON_TENSOR_DUMP_DIR": "/tmp/megatron_tensor_dump" if MODE == "debug_one_sample" else "",
            "MEGATRON_TENSOR_DUMP_LAYERS": "0,1,2" if MODE == "debug_one_sample" else "",  # Last layer (28) added automatically
            # Debug logging for logprob comparison
            # Set SLIME_DEBUG_LOGPROB_DIFF=1 to enable detailed per-sample logprob comparison logging
            "SLIME_DEBUG_LOGPROB_DIFF": debug_logprob_diff,
        },
    )


if __name__ == "__main__":
    prepare()
    execute()
