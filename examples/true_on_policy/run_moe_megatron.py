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

# PP configuration for verifying true on-policy with pipeline parallelism
# With 8 GPUs and PP=2: TP=4, PP=2, EP=4, DP=1
#   - Each PP stage has 4 GPUs for TP/EP
#   - SGLang engine uses all 8 GPUs (TP=4 x PP=2)
#   - No sequence splitting, so attention is bitwise identical to SGLang inference
PP_SIZE = int(os.environ.get("SLIME_PP_SIZE", "1"))
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

def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")
    if USE_RAW:
        U.convert_checkpoint(
            model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS, dir_dst="/root/models"
        )


def execute():
    pipeline_parallel_size = PP_SIZE

    # For MoE models, SGLang requires TP >= EP.
    # Default TP fills all GPUs within each PP stage (so DP=1).
    # Examples:
    #   8 GPUs, PP=1 → TP=8, EP=8, DP=1  (original behavior)
    #   8 GPUs, PP=2 → TP=4, EP=4, DP=1
    tensor_parallel_size = TP_SIZE if USE_TP else (NUM_GPUS // pipeline_parallel_size)

    assert NUM_GPUS % (tensor_parallel_size * pipeline_parallel_size) == 0, (
        f"NUM_GPUS ({NUM_GPUS}) must be divisible by "
        f"TP * PP ({tensor_parallel_size} * {pipeline_parallel_size})"
    )
    data_parallel_size = NUM_GPUS // (tensor_parallel_size * pipeline_parallel_size)

    # EP = TP within each PP stage (SGLang requires TP >= EP)
    expert_parallel_size = tensor_parallel_size
    # SGLang engine needs TP * PP GPUs total
    gpus_per_sglang_engine = tensor_parallel_size * pipeline_parallel_size

    debug_global_batch_size = int(os.environ.get("SLIME_DEBUG_GLOBAL_BATCH_SIZE", "2"))
    normal_global_batch_size = int(os.environ.get("SLIME_NORMAL_GLOBAL_BATCH_SIZE", "128"))
    global_batch_size = debug_global_batch_size if MODE == "debug_one_sample" else normal_global_batch_size
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
    debug_num_rollout = int(os.environ.get("SLIME_DEBUG_NUM_ROLLOUT", "3"))
    normal_num_rollout = int(os.environ.get("SLIME_NORMAL_NUM_ROLLOUT", "3000"))
    normal_eval_interval = int(os.environ.get("SLIME_NORMAL_EVAL_INTERVAL", "10"))
    debug_rollout_batch_size = int(os.environ.get("SLIME_DEBUG_ROLLOUT_BATCH_SIZE", "1"))
    normal_rollout_batch_size = int(os.environ.get("SLIME_NORMAL_ROLLOUT_BATCH_SIZE", "16"))
    debug_n_samples_per_prompt = int(os.environ.get("SLIME_DEBUG_N_SAMPLES_PER_PROMPT", "2"))
    normal_n_samples_per_prompt = int(os.environ.get("SLIME_NORMAL_N_SAMPLES_PER_PROMPT", "8"))
    debug_rollout_max_response_len = int(os.environ.get("SLIME_DEBUG_ROLLOUT_MAX_RESPONSE_LEN", "10"))
    normal_rollout_max_response_len = int(os.environ.get("SLIME_NORMAL_ROLLOUT_MAX_RESPONSE_LEN", "1024"))
    # In debug_one_sample, num_rollout=1 is often used as a full-path smoke test.
    # Keep post-train sync enabled by default in that case so the run covers:
    # rollout -> train -> post-train update_weights.
    default_skip_post_train_sync = "0"
    if MODE == "debug_one_sample" and debug_num_rollout > 1:
        # For multi-rollout debug runs, skip the very last post-train sync by default
        # to reduce teardown-time memory churn.
        default_skip_post_train_sync = "1"

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {debug_num_rollout if MODE == 'debug_one_sample' else normal_num_rollout} "  # Need at least 2-3 steps to observe divergence pattern
        f"--rollout-batch-size {debug_rollout_batch_size if MODE == 'debug_one_sample' else normal_rollout_batch_size} "
        f"--n-samples-per-prompt {debug_n_samples_per_prompt if MODE == 'debug_one_sample' else normal_n_samples_per_prompt} "
        f"--rollout-max-response-len {debug_rollout_max_response_len if MODE == 'debug_one_sample' else normal_rollout_max_response_len} "
        "--rollout-temperature 1 "
        # temp remove this to make test easier
        # "--over-sampling-batch-size 64 "
        # "--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        f"--global-batch-size {global_batch_size} "
    )

    eval_args = ""
    if MODE == "normal" and normal_eval_interval > 0:
        eval_args = (
            "--skip-eval-before-train "
            f"--eval-interval {normal_eval_interval} "
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

    # PP+MoE keeps optimizer states on GPU by default and can OOM on later
    # wake_up/resume cycles after the first train step. Offload optimizer states
    # to CPU for stability in normal true-on-policy runs.
    disable_optimizer_cpu_offload = _system_bool("SLIME_DISABLE_OPTIMIZER_CPU_OFFLOAD", False)
    if pipeline_parallel_size > 1 and not disable_optimizer_cpu_offload:
        optimizer_args += (
            "--optimizer-cpu-offload "
            "--overlap-cpu-optimizer-d2h-h2d "
            "--use-precision-aware-optimizer "
        )

    if MODE == "debug_one_sample":
        optimizer_args += (
            "--lr-decay-iters 4 "
        )

    enable_recompute = _system_bool("SLIME_ENABLE_RECOMPUTE", pipeline_parallel_size > 1)
    recompute_args = ""
    if enable_recompute:
        # Save activation memory to avoid OOM when resuming SGLang memory saver
        # after update_weights in colocated PP runs.
        # Use --recompute-activations by default because PP + full granularity
        # requires extra knobs (e.g. recompute_num_layers) and can fail at init.
        recompute_args = _system_env("SLIME_RECOMPUTE_ARGS", "--recompute-activations ")

    disable_rollout_offload = _system_bool("SLIME_DISABLE_ROLLOUT_OFFLOAD", False)


    tp_args = (
        f"--tensor-model-parallel-size {tensor_parallel_size} "
        # "--sequence-parallel "  # Disabled: only use TP without SP for easier debugging
        f"--pipeline-model-parallel-size {pipeline_parallel_size} "
        f"--expert-model-parallel-size {expert_parallel_size} "  # EP = TP (SGLang requires TP >= EP)
        "--expert-tensor-parallel-size 1 "
    )
    default_sglang_mem_fraction_static = 0.35 if MODEL_NAME == "Qwen3-30B-A3B" else 0.5
    if pipeline_parallel_size > 1:
        # Keep GPU headroom in PP colocated mode.
        # When rollout offload is enabled, onload_kv must reserve KV cache after
        # Megatron weights are resident; use a smaller default to avoid resume OOM.
        default_sglang_mem_fraction_static = min(
            default_sglang_mem_fraction_static,
            0.10 if not disable_rollout_offload else 0.30,
        )
    sglang_mem_fraction_static = float(_system_env("SLIME_SGLANG_MEM_FRACTION_STATIC", str(default_sglang_mem_fraction_static)))

    sglang_args = (
        f"--rollout-num-gpus-per-engine {gpus_per_sglang_engine} "
        f"--sglang-tp-size {tensor_parallel_size} "  # SGLang requires TP >= EP
        f"--sglang-pipeline-parallel-size {pipeline_parallel_size} "
        f"--sglang-ep-size {expert_parallel_size} "  # EP = TP
        "--sglang-decode-log-interval 1000 "
        "--sglang-enable-metrics "
        f"--sglang-mem-fraction-static {sglang_mem_fraction_static} "
        # Disable CUDA graph for true on-policy to ensure numerical consistency
        # CUDA graph can cause non-determinism in MoE routing and expert computation
        "--sglang-disable-cuda-graph "
    )
    router_args = (
        "--router-health-check-timeout-secs 30 "
        "--router-health-failure-threshold 10 "
    )
    fault_tolerance_args = ""
    if _system_bool("SLIME_USE_FAULT_TOLERANCE", False):
        fault_tolerance_args = "--use-fault-tolerance "


    ci_args = ""
    # Allow local long-run debugging without CI metric gating.
    enable_ci = os.environ.get("SLIME_ENABLE_CI", "1" if MODE != "debug_one_sample" else "0") == "1"
    # CI metric checker expects eval metrics; in debug_one_sample we typically
    # run no eval and only validate true-on-policy parity/logprob diff.
    if enable_ci:
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
        # Keep rollout offload enabled by default in PP debug runs so SGLang
        # releases GPU memory before actor training/optimizer-step allocation.
        # This avoids first-step OOM when MoE optimizer states are initialized.
        if disable_rollout_offload:
            misc_args += "--no-offload-rollout "
        # Keep PP update buckets memory-safe by default; allow override by env.
        update_weight_buffer_size = _system_int(
            "SLIME_UPDATE_WEIGHT_BUFFER_SIZE",
            256 * 1024**2,
        )
        misc_args += f"--update-weight-buffer-size {update_weight_buffer_size} "

    # torch_memory_saver uses this margin when offload_train is enabled.
    # Keep a smaller default in PP debug mode so tiny allocator requests are not
    # rejected when free memory hovers around ~1GiB.
    train_memory_margin_bytes = _system_int(
        "SLIME_TRAIN_MEMORY_MARGIN_BYTES",
        0 if pipeline_parallel_size > 1 and MODE == "debug_one_sample" else 1024**3,
    )
    misc_args += f"--train-memory-margin-bytes {train_memory_margin_bytes} "
    
    # Optional: verify rollout-engine weights are exactly updated from actor side.
    # This snapshots rollout weights, randomizes them, then checks equality after sync.
    if os.environ.get("SLIME_CHECK_WEIGHT_UPDATE_EQUAL", "0") == "1":
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
        # PP>1: use stage-local tensor sync by default.
        # This aligns Megatron PP stages with SGLang PP stages and avoids
        # cross-stage IPC mapping errors in colocated update_weights.
        "SLIME_TENSOR_SYNC_STAGE_LOCAL_GROUP": _system_env(
            "SLIME_TENSOR_SYNC_STAGE_LOCAL_GROUP",
            "1" if pipeline_parallel_size > 1 else "0",
        ),
        # For PP runs, prefer GPU bucket sync by default (faster and more stable
        # than CPU staging in our colocated true-on-policy setup).
        "SLIME_TENSOR_SYNC_CPU_STAGING": _system_env(
            "SLIME_TENSOR_SYNC_CPU_STAGING",
            "0" if pipeline_parallel_size > 1 else "1",
        ),
        # Keep direct IPC path opt-in unless explicitly requested.
        "SLIME_TENSOR_SYNC_GPU_DIRECT": _system_env("SLIME_TENSOR_SYNC_GPU_DIRECT", "0"),
        # PP direct CUDA-IPC tensor sync is unstable for large MoE updates; keep opt-in.
        "SLIME_TENSOR_SYNC_GPU_DIRECT_ALLOW_PP": _system_env(
            "SLIME_TENSOR_SYNC_GPU_DIRECT_ALLOW_PP",
            "0" if pipeline_parallel_size > 1 else "1",
        ),
        # Safety rail: for PP runs, force-disable full GPU-direct tensor sync unless
        # explicitly turned off by the operator.
        "SLIME_TENSOR_SYNC_FORCE_SAFE_PP": _system_env(
            "SLIME_TENSOR_SYNC_FORCE_SAFE_PP",
            "1" if pipeline_parallel_size > 1 else "0",
        ),
        # For PP with non-direct sync, allow flattened GPU bucket mode by default.
        "SLIME_TENSOR_SYNC_GPU_BUCKET_ALLOW_PP": _system_env(
            "SLIME_TENSOR_SYNC_GPU_BUCKET_ALLOW_PP",
            "1" if pipeline_parallel_size > 1 else "0",
        ),
        # Keep flattened-bucket for most tensors, but force direct IPC for
        # vocab/lm_head buckets to avoid PP flattened-bucket load instability.
        "SLIME_TENSOR_SYNC_GPU_DIRECT_FOR_VOCAB": _system_env(
            "SLIME_TENSOR_SYNC_GPU_DIRECT_FOR_VOCAB",
            "1" if pipeline_parallel_size > 1 else "0",
        ),
        # Direct named-tensor IPC for all MoE buckets can accumulate many CUDA
        # IPC handles across repeated updates and cause later train-step OOM.
        # Keep it opt-in; default to flattened-bucket transfer.
        "SLIME_TENSOR_SYNC_GPU_DIRECT_FOR_MOE": _system_env(
            "SLIME_TENSOR_SYNC_GPU_DIRECT_FOR_MOE",
            "0",
        ),
        # DEBUG: Enable to get accurate CUDA error location (slows down execution significantly)
        #"CUDA_LAUNCH_BLOCKING": "1",  # ENABLED: Finding the real source of CUDA illegal memory access
    }
    if pipeline_parallel_size > 1:
        # torch_memory_saver (used by rollout offload) is incompatible with
        # expandable_segments. Only set this allocator mode by default when
        # rollout offload is disabled; still honor explicit user override.
        alloc_conf = _system_env("PYTORCH_CUDA_ALLOC_CONF", "")
        if alloc_conf:
            true_on_policy_envs["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf
        elif disable_rollout_offload:
            # Reduce allocator fragmentation for repeated large all-gather/cat
            # during PP weight sync when rollout memory saver is not used.
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
            # In debug_one_sample with num_rollout=1, keep post-train sync enabled
            # by default so this mode still validates a full training flow.
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
            # Debug gradient all-reduce for MoE backward pass
            "DEBUG_GRAD_ALLREDUCE": "1" if MODE == "debug_one_sample" else "0",
            "DEBUG_OVERRIDE_REWARDS": "first_one" if MODE == "debug_one_sample" else "",
            # Debug-only: allow simulating repeated update_weights without rollout/train loop.
            "SLIME_DEBUG_EXTRA_WEIGHT_UPDATES": _system_env("SLIME_DEBUG_EXTRA_WEIGHT_UPDATES", "0"),
            # Debug gradient sync verification - enable to check if all-reduce is working
            # "DEBUG_GRAD_SYNC": "1",  # Enable to verify gradients are identical across ranks after all-reduce
            # "DEBUG_ROUTER_GRAD_SYNC": "1",  # Enable to see per-rank gradient values before/after all-reduce
            # # Debug EP broadcast during weight sync
            # "DEBUG_EP_BROADCAST": "1",  # Enable to check EP broadcast logic in weight sync
            # # Debug expert weight conversion during sync
            # "DEBUG_EXPERT_SYNC": "1",  # Enable to check Megatron->HF expert weight conversion
        },
    )


if __name__ == "__main__":
    prepare()
    execute()
