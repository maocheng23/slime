from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from typing import Any
import gc
import io
import logging
import os

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle

from slime.utils.distributed_utils import get_gloo_group

from ..sglang import FlattenedTensorBucket, MultiprocessingSerializer
from .hf_weight_iterator_base import HfWeightIteratorBase
from .update_weight_from_distributed import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    post_process_weights,
    update_weights_from_distributed,
)

logger = logging.getLogger(__name__)


def _is_vocab_or_lm_head_weight(name: str) -> bool:
    return name in {"model.embed_tokens.weight", "lm_head.weight"}


def _is_moe_expert_weight(name: str) -> bool:
    return ".experts." in name


class UpdateWeightFromTensor:
    """
    Update rollout engines from tensor dict:
    load(dict→GPU) → broadcast PP/EP(GPU NCCL) → gather TP(GPU NCCL) → convert HF(GPU) → send.
    Colocated: GPU→CPU serialize → gather_object(Gloo CPU, collects from rollout_num_gpus_per_engine ranks) → Ray IPC to engine.
    Distributed: GPU NCCL broadcast to remote engines.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        """
        Compute param buckets, create stage-local IPC Gloo groups.
        """
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0

        self._hf_weight_iterator = HfWeightIteratorBase.create(
            args=args, model=model, model_name=model_name, quantization_config=quantization_config
        )

        self._pp_size = int(
            getattr(self.args, "sglang_pipeline_parallel_size", 0)
            or getattr(self.args, "pipeline_model_parallel_size", 0)
            or 1
        )
        # PP tensor updates are consumed by SGLang per TP rank
        # (`serialized_named_tensors[self.tp_rank]`), so the IPC gather list must be
        # stage-local (size == TP size). Keep stage-local on by default for PP>1 and
        # allow explicit override for targeted debugging.
        self._use_stage_local_ipc_group = (
            os.environ.get("SLIME_TENSOR_SYNC_STAGE_LOCAL_GROUP", "1" if self._pp_size > 1 else "0") == "1"
        )
        if self._pp_size > 1 and self._use_stage_local_ipc_group:
            logger.warning(
                "PP stage-local IPC tensor sync is experimental. "
                "If rollout generate fails after update_weights, set "
                "SLIME_TENSOR_SYNC_STAGE_LOCAL_GROUP=0."
            )
        self._tp_size_for_ipc = int(getattr(self.args, "sglang_tp_size", 0) or 0)
        if self._tp_size_for_ipc <= 0:
            self._tp_size_for_ipc = max(1, self.args.rollout_num_gpus_per_engine // self._pp_size)
        self._ipc_group_size = (
            self._tp_size_for_ipc if self._use_stage_local_ipc_group else self.args.rollout_num_gpus_per_engine
        )
        default_cpu_staging = self._pp_size > 1
        self._use_cpu_staging_for_colocate = (
            os.environ.get("SLIME_TENSOR_SYNC_CPU_STAGING", "1" if default_cpu_staging else "0") == "1"
        )
        # Direct named-tensor CUDA IPC creates many per-tensor handles and can
        # become a bottleneck for large PP/MoE updates. Keep bucketed transfer
        # as default and allow direct mode only for targeted debugging.
        self._use_gpu_direct_for_colocate = os.environ.get("SLIME_TENSOR_SYNC_GPU_DIRECT", "0") == "1"
        if self._pp_size > 1 and self._use_gpu_direct_for_colocate:
            # PP direct CUDA-IPC is unstable by default for large MoE sync payloads.
            # Keep it opt-in for targeted debugging only.
            allow_pp_gpu_direct = os.environ.get("SLIME_TENSOR_SYNC_GPU_DIRECT_ALLOW_PP", "0") == "1"
            if not allow_pp_gpu_direct:
                logger.warning(
                    "Disable direct CUDA-IPC tensor sync for PP>1 (pp_size=%s). "
                    "Using non-direct tensor sync path instead. "
                    "Set SLIME_TENSOR_SYNC_GPU_DIRECT_ALLOW_PP=1 to force direct mode.",
                    self._pp_size,
                )
                self._use_gpu_direct_for_colocate = False
        if self._pp_size > 1 and (not self._use_cpu_staging_for_colocate) and (not self._use_gpu_direct_for_colocate):
            # PP flattened-bucket GPU sync can fail in SGLang load_weights() with
            # CUDA invalid-argument on vocab embedding updates. Keep it opt-in.
            allow_pp_gpu_bucket = os.environ.get("SLIME_TENSOR_SYNC_GPU_BUCKET_ALLOW_PP", "0") == "1"
            if not allow_pp_gpu_bucket:
                logger.warning(
                    "Disable flattened-bucket GPU tensor sync for PP>1 (pp_size=%s). "
                    "Falling back to CPU staging for stability. "
                    "Set SLIME_TENSOR_SYNC_GPU_BUCKET_ALLOW_PP=1 to force GPU bucket mode.",
                    self._pp_size,
                )
                self._use_cpu_staging_for_colocate = True
        self._gpu_direct_ipc_gc_interval = max(1, int(os.environ.get("SLIME_TENSOR_SYNC_GPU_DIRECT_GC_INTERVAL", "8")))
        # In PP GPU sync mode, long update runs can accumulate CUDA IPC/allocator
        # bookkeeping and eventually OOM during later buckets. Run periodic GC +
        # ipc_collect + empty_cache to keep memory stable across repeated updates.
        default_ipc_gc_interval = "1" if self._pp_size > 1 else "0"
        self._gpu_tensor_ipc_gc_interval = int(
            os.environ.get("SLIME_TENSOR_SYNC_IPC_GC_INTERVAL", default_ipc_gc_interval)
        )
        if self._gpu_tensor_ipc_gc_interval < 0:
            self._gpu_tensor_ipc_gc_interval = 0
        self._force_safe_pp_tensor_sync = (
            self._pp_size > 1 and os.environ.get("SLIME_TENSOR_SYNC_FORCE_SAFE_PP", "1") == "1"
        )
        if self._force_safe_pp_tensor_sync and self._use_gpu_direct_for_colocate:
            logger.warning(
                "SLIME_TENSOR_SYNC_FORCE_SAFE_PP=1: disable full GPU-direct tensor sync for PP>1 "
                "(set SLIME_TENSOR_SYNC_FORCE_SAFE_PP=0 to allow it)."
            )
            self._use_gpu_direct_for_colocate = False

        world_size = dist.get_world_size()
        assert world_size % self._ipc_group_size == 0, (
            f"world_size ({world_size}) must be divisible by IPC group size ({self._ipc_group_size})"
        )

        # Create IPC groups stage-local by default for PP>1 to match SGLang's
        # TP-rank indexing in update_weights_from_tensor.
        self._ipc_group_start_rank = None
        self._ipc_group_index = None
        for start_rank in range(0, world_size, self._ipc_group_size):
            end_rank = start_rank + self._ipc_group_size
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if dist.get_rank() in group_ranks:
                self._ipc_gather_group = new_group
                self._ipc_gather_src = start_rank
                self._ipc_group_start_rank = start_rank
                self._ipc_group_index = start_rank // self._ipc_group_size

        self._model_update_groups = None

    def connect_rollout_engines(
        self, rollout_engines: Sequence[ActorHandle], rollout_engine_lock: ActorHandle
    ) -> None:
        """
        Split colocated/distributed engines. Global source rank (DP=TP=PP=0) creates NCCL
        for distributed. Map ranks to colocated IPC engines.
        """
        self.rollout_engines = rollout_engines
        colocate_engine_nums = (
            self.args.actor_num_nodes * self.args.actor_num_gpus_per_node // self.args.rollout_num_gpus_per_engine
        )
        self.use_distribute = len(rollout_engines) > colocate_engine_nums

        if self.use_distribute:
            self.rollout_engines = rollout_engines[:colocate_engine_nums]
            self.distributed_rollout_engines = rollout_engines[colocate_engine_nums:]
            self._is_distributed_src_rank = (
                mpu.get_data_parallel_rank(with_context_parallel=True) == 0
                and mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_pipeline_model_parallel_rank() == 0
            )
            self._group_name = "slime"
            if self._is_distributed_src_rank:
                if self._model_update_groups is not None:
                    disconnect_rollout_engines_from_distributed(
                        self.args, self._group_name, self._model_update_groups, self.distributed_rollout_engines
                    )

                self._model_update_groups = connect_rollout_engines_from_distributed(
                    self.args, self._group_name, self.distributed_rollout_engines
                )

        self._ipc_engine = None
        if self.rollout_engines:
            # Stage-local grouping maps `pp_size` groups to one rollout engine.
            # Engine-wide grouping maps one IPC group to one rollout engine.
            assert self._ipc_group_index is not None
            engine_index = (
                self._ipc_group_index // self._pp_size if self._use_stage_local_ipc_group else self._ipc_group_index
            )
            if engine_index < len(self.rollout_engines):
                self._ipc_engine = self.rollout_engines[engine_index]
        if dist.get_rank() == self._ipc_gather_src:
            assert self._ipc_engine is not None, (
                f"Failed to map IPC source rank {dist.get_rank()} to a rollout engine "
                f"(group_index={self._ipc_group_index}, pp_size={self._pp_size}, "
                f"num_rollout_engines={len(self.rollout_engines)})"
            )

        logger.info(
            "Tensor weight sync mapping: rank=%s, group_start=%s, group_size=%s, group_index=%s, pp_size=%s, "
            "has_ipc_engine=%s, use_distribute=%s, cpu_staging=%s, gpu_direct=%s, stage_local_group=%s",
            dist.get_rank(),
            self._ipc_group_start_rank,
            self._ipc_group_size,
            self._ipc_group_index,
            self._pp_size,
            self._ipc_engine is not None,
            self.use_distribute,
            self._use_cpu_staging_for_colocate,
            self._use_gpu_direct_for_colocate,
            self._use_stage_local_ipc_group,
        )

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        version++, flush caches, process buckets. Progress on rank 0.
        """
        self.weight_version += 1

        rank = dist.get_rank()
        if rank == 0:
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=self.rollout_engines,
                )
        dist.barrier(group=get_gloo_group())

        megatron_local_weights = self.weights_getter()

        for bucket_idx, hf_named_tensors in enumerate(self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights)):
            refs, long_lived_tensors = self._send_hf_params(hf_named_tensors)
            ray.get(refs)
            if not self._use_cpu_staging_for_colocate:
                # When using CUDA IPC tensor payloads (non-CPU staging), non-source
                # ranks must keep their local tensors alive until the source rank
                # finishes update_weights_from_tensor.remote(). Otherwise SGLang may
                # dereference stale IPC pointers and fail in GPU copy paths.
                dist.barrier(group=self._ipc_gather_group)
            del long_lived_tensors
            del refs
            del hf_named_tensors
            if self._gpu_tensor_ipc_gc_interval > 0 and ((bucket_idx + 1) % self._gpu_tensor_ipc_gc_interval == 0):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.ipc_collect()
                    torch.cuda.empty_cache()
            if self._use_gpu_direct_for_colocate and (
                ((bucket_idx + 1) % self._gpu_direct_ipc_gc_interval == 0) or bucket_idx == 0
            ):
                # Explicitly reclaim CUDA IPC mappings to avoid monotonically
                # growing reserved memory across repeated PP updates.
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

        dist.barrier(group=get_gloo_group())

        # int4/fp4 post_process
        if rank == 0:
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self.rollout_engines,
                )
        dist.barrier(group=get_gloo_group())

    def _send_hf_params(self, hf_named_tensors) -> tuple[list[ObjectRef], Any]:
        all_refs = []

        refs_colocated, long_lived_tensors = _send_to_colocated_engine(
            hf_named_tensors,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
            weight_version=self.weight_version,
            use_cpu_staging=self._use_cpu_staging_for_colocate,
            use_gpu_direct=self._use_gpu_direct_for_colocate,
        )
        all_refs.extend(refs_colocated)

        if self.use_distribute and self._is_distributed_src_rank:
            refs_distributed = update_weights_from_distributed(
                self._group_name,
                self._model_update_groups,
                self.weight_version,
                self.distributed_rollout_engines,
                hf_named_tensors,
            )
            if refs_distributed:
                all_refs.extend(refs_distributed)

        return all_refs, long_lived_tensors


def _send_to_colocated_engine(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    *,
    ipc_engine,
    ipc_gather_src,
    ipc_gather_group,
    weight_version,
    use_cpu_staging: bool,
    use_gpu_direct: bool,
) -> tuple[list[ObjectRef], Any]:
    # TODO improve
    long_live_tensors = []

    # Direct per-tensor CUDA IPC can create thousands of live IPC handles in
    # large PP/MoE updates. This is fast for small payloads, but it can cause
    # severe allocator pressure and delayed deallocation. Fall back to bucketed
    # GPU transfer for large chunks while still avoiding CPU staging.
    max_gpu_direct_tensors = int(os.environ.get("SLIME_TENSOR_SYNC_GPU_DIRECT_MAX_TENSORS", "256"))
    should_use_gpu_direct = (not use_cpu_staging) and use_gpu_direct and len(hf_named_tensors) <= max_gpu_direct_tensors
    # Flattened-bucket GPU reconstruction is fragile for some PP buckets.
    # Allow forcing direct named-tensor IPC for selected buckets.
    force_vocab_gpu_direct = (not use_cpu_staging) and (
        os.environ.get("SLIME_TENSOR_SYNC_GPU_DIRECT_FOR_VOCAB", "0") == "1"
    ) and any(_is_vocab_or_lm_head_weight(name) for name, _ in hf_named_tensors)
    force_moe_gpu_direct = (not use_cpu_staging) and (
        os.environ.get("SLIME_TENSOR_SYNC_GPU_DIRECT_FOR_MOE", "0") == "1"
    ) and any(_is_moe_expert_weight(name) for name, _ in hf_named_tensors)
    if force_vocab_gpu_direct and not should_use_gpu_direct:
        logger.info(
            "Force direct named-tensor transfer for vocab/lm_head bucket (num_tensors=%s).",
            len(hf_named_tensors),
        )
    if force_moe_gpu_direct and not should_use_gpu_direct:
        logger.info(
            "Force direct named-tensor transfer for MoE expert bucket (num_tensors=%s).",
            len(hf_named_tensors),
        )
    should_use_gpu_direct = should_use_gpu_direct or force_vocab_gpu_direct or force_moe_gpu_direct
    if (not use_cpu_staging) and use_gpu_direct and (not should_use_gpu_direct):
        logger.warning(
            "Fallback to flattened_bucket GPU transfer for large chunk: num_tensors=%s > max_gpu_direct_tensors=%s",
            len(hf_named_tensors),
            max_gpu_direct_tensors,
        )

    if use_cpu_staging:
        staged_named_tensors = []
        for name, tensor in hf_named_tensors:
            if tensor.device.type == "cpu":
                staged_named_tensors.append((name, tensor))
            else:
                staged_named_tensors.append((name, tensor.to(device="cpu", non_blocking=True)))
        hf_named_tensors = staged_named_tensors

    if should_use_gpu_direct:
        # GPU IPC path: avoid flattened-bucket reconstruction on SGLang side.
        # Send direct named tensors to keep the update path simple and avoid
        # CUDA invalid-argument failures observed with flattened bucket loads.
        long_live_tensors.extend(tensor for _, tensor in hf_named_tensors)

        local_serialized_named_tensors = MultiprocessingSerializer.serialize(hf_named_tensors, output_str=True)
        gathered_serialized_named_tensors = (
            [None] * dist.get_world_size(ipc_gather_group) if ipc_gather_src == dist.get_rank() else None
        )
        dist.gather_object(
            local_serialized_named_tensors,
            object_gather_list=gathered_serialized_named_tensors,
            dst=ipc_gather_src,
            group=ipc_gather_group,
        )

        refs = []
        if dist.get_rank() == ipc_gather_src:
            refs.append(
                ipc_engine.update_weights_from_tensor.remote(
                    serialized_named_tensors=gathered_serialized_named_tensors,
                    load_format=None,
                    weight_version=str(weight_version),
                )
            )
        return refs, long_live_tensors

    use_mixed_dtype_bucket = use_cpu_staging and getattr(FlattenedTensorBucket, "supports_multi_dtypes", False)
    if use_mixed_dtype_bucket:
        converted_named_tensors_by_dtypes = {"dtype": hf_named_tensors}
    else:
        # Keep one bucket per dtype. Mixed-dtype byte views can
        # produce misaligned GPU tensor views after reconstruction and cause
        # CUDA invalid-argument failures during model.load_weights().
        converted_named_tensors_by_dtypes = {}
        for name, tensor in hf_named_tensors:
            dtype = tensor.dtype
            if dtype not in converted_named_tensors_by_dtypes:
                converted_named_tensors_by_dtypes[dtype] = []
            converted_named_tensors_by_dtypes[dtype].append((name, tensor))

    serialized_tensors_by_dtype = {}
    for _dtype, named_tensors in converted_named_tensors_by_dtypes.items():
        dtype_key = str(_dtype)
        flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        metadata = flattened_tensor_bucket.get_metadata()
        flattened_tensor = flattened_tensor_bucket.get_flattened_tensor()

        if use_cpu_staging:
            # Avoid torch multiprocessing FD transport for CPU tensors, which can fail
            # across Ray worker processes with auth errors.
            cpu_flattened_tensor = (
                flattened_tensor if flattened_tensor.device.type == "cpu" else flattened_tensor.to(device="cpu")
            ).contiguous()
            tensor_buf = io.BytesIO()
            torch.save(cpu_flattened_tensor, tensor_buf)
            payload = {
                "flattened_tensor_torch_save": tensor_buf.getvalue(),
                "metadata": metadata,
            }
            long_live_tensors.append(cpu_flattened_tensor)
            serialized_tensors_by_dtype[dtype_key] = MultiprocessingSerializer.serialize(payload, output_str=True)
        else:
            flattened_tensor_data = {
                "flattened_tensor": flattened_tensor,
                "metadata": metadata,
            }
            long_live_tensors.append(flattened_tensor_data)
            serialized_tensors_by_dtype[dtype_key] = MultiprocessingSerializer.serialize(
                flattened_tensor_data, output_str=True
            )

    serialized_named_tensors = (
        [None] * dist.get_world_size(ipc_gather_group) if ipc_gather_src == dist.get_rank() else None
    )
    dist.gather_object(
        serialized_tensors_by_dtype,
        object_gather_list=serialized_named_tensors,
        dst=ipc_gather_src,
        group=ipc_gather_group,
    )

    refs = []
    if dist.get_rank() == ipc_gather_src:
        # Merge bucket payloads by dtype key instead of list index. Under PP
        # stage-local sync, list-order assumptions can misalign payloads across
        # ranks and break reconstruction on SGLang side.
        if not serialized_named_tensors:
            return refs, long_live_tensors
        dtype_keys = sorted(serialized_named_tensors[0].keys())
        for dtype_key in dtype_keys:
            per_rank_payloads = []
            for rank_payload in serialized_named_tensors:
                if dtype_key not in rank_payload:
                    raise RuntimeError(
                        f"Missing dtype bucket {dtype_key} in gathered payloads. "
                        f"available_keys={sorted(rank_payload.keys())}"
                    )
                per_rank_payloads.append(rank_payload[dtype_key])
            kwargs = {
                "serialized_named_tensors": per_rank_payloads,
                "load_format": "flattened_bucket",
                "weight_version": str(weight_version),
            }
            refs.append(ipc_engine.update_weights_from_tensor.remote(**kwargs))

    return refs, long_live_tensors
