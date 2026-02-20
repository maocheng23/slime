import gc
import logging
import os
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from typing import Any

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
        Compute param buckets, create IPC Gloo groups (rollout_num_gpus_per_engine ranks/group).
        """
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0

        # --- PP stage-local IPC configuration (add-on for PP>1) ---
        # Use Megatron's PP size for stage-local decisions: the IPC groups must
        # match Megatron's pipeline stages so each stage sends its own layers.
        self._megatron_pp_size = int(getattr(self.args, "pipeline_model_parallel_size", 1))
        self._sglang_pp_size = int(getattr(self.args, "sglang_pipeline_parallel_size", 0) or self._megatron_pp_size)
        self._is_cross_pp = self._megatron_pp_size != self._sglang_pp_size
        self._use_stage_local_ipc_group = (
            os.environ.get("SLIME_TENSOR_SYNC_STAGE_LOCAL_GROUP", "1" if self._megatron_pp_size > 1 else "0") == "1"
        )
        # Cross-PP (Megatron PP != SGLang PP): stage-local IPC won't work because
        # CUDA IPC handles from one stage's GPUs can't be accessed by SGLang workers
        # on another stage's GPUs.  Fall back to full IPC group with PP broadcast.
        if self._is_cross_pp and self._use_stage_local_ipc_group:
            logger.warning(
                "Cross-PP detected (Megatron PP=%d, SGLang PP=%d): disabling stage-local "
                "IPC sync, falling back to PP broadcast + full IPC group.",
                self._megatron_pp_size, self._sglang_pp_size,
            )
            self._use_stage_local_ipc_group = False
            # Override env var so HfWeightIteratorDirect also uses PP broadcast
            os.environ["SLIME_TENSOR_SYNC_STAGE_LOCAL_GROUP"] = "0"
        if self._megatron_pp_size > 1 and self._use_stage_local_ipc_group:
            logger.warning(
                "PP stage-local IPC tensor sync is experimental. "
                "If rollout generate fails after update_weights, set "
                "SLIME_TENSOR_SYNC_STAGE_LOCAL_GROUP=0."
            )

        self._hf_weight_iterator = HfWeightIteratorBase.create(
            args=args, model=model, model_name=model_name, quantization_config=quantization_config
        )
        self._gpu_direct_for_vocab = (
            self._use_stage_local_ipc_group
            and os.environ.get("SLIME_TENSOR_SYNC_GPU_DIRECT_FOR_VOCAB", "1" if self._megatron_pp_size > 1 else "0") == "1"
        )
        self._ipc_gc_interval = max(
            0, int(os.environ.get("SLIME_TENSOR_SYNC_IPC_GC_INTERVAL", "1" if self._megatron_pp_size > 1 else "0"))
        )

        # --- Create IPC Gloo groups ---
        # PP>1 + stage_local: one IPC group per Megatron PP stage
        #   group_size = world_size / megatron_pp_size (all ranks in one stage)
        # Otherwise (original): group_size = rollout_num_gpus_per_engine
        world_size = dist.get_world_size()
        if self._use_stage_local_ipc_group:
            ipc_group_size = world_size // self._megatron_pp_size
        else:
            ipc_group_size = self.args.rollout_num_gpus_per_engine
        self._ipc_group_size = ipc_group_size

        assert world_size % self._ipc_group_size == 0, (
            f"world_size ({world_size}) must be divisible by IPC group size ({self._ipc_group_size})"
        )

        # create the group within megatron.
        self._ipc_group_index = None
        for start_rank in range(0, world_size, self._ipc_group_size):
            end_rank = start_rank + self._ipc_group_size
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if dist.get_rank() in group_ranks:
                self._ipc_gather_group = new_group
                self._ipc_gather_src = start_rank
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

        # Map current rank to its colocated IPC engine.
        # PP>1 + stage_local: multiple IPC groups map to one engine
        #   (megatron_pp_size groups per engine).
        # Otherwise (original): one IPC group per engine.
        if self._use_stage_local_ipc_group:
            self._ipc_engine = None
            if self.rollout_engines:
                assert self._ipc_group_index is not None
                engine_index = self._ipc_group_index // self._megatron_pp_size
                if engine_index < len(self.rollout_engines):
                    self._ipc_engine = self.rollout_engines[engine_index]
            if dist.get_rank() == self._ipc_gather_src:
                assert self._ipc_engine is not None, (
                    f"Failed to map IPC source rank {dist.get_rank()} to a rollout engine "
                    f"(group_index={self._ipc_group_index}, megatron_pp_size={self._megatron_pp_size}, "
                    f"num_rollout_engines={len(self.rollout_engines)})"
                )
        else:
            # Here we assume the gpu id of rollout engines and train actors are the same.
            for i, engine in enumerate(self.rollout_engines):
                start_rank = i * self.args.rollout_num_gpus_per_engine
                end_rank = (i + 1) * self.args.rollout_num_gpus_per_engine
                group_ranks = list(range(start_rank, end_rank))
                if dist.get_rank() in group_ranks:
                    self._ipc_engine = engine

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
            # PP add-on: non-source ranks must keep tensors alive until the source
            # rank finishes update_weights_from_tensor.remote().
            dist.barrier(group=self._ipc_gather_group)
            del long_lived_tensors
            # PP add-on: periodic GC to reclaim CUDA IPC handles between buckets.
            if self._ipc_gc_interval > 0 and ((bucket_idx + 1) % self._ipc_gc_interval == 0):
                gc.collect()
                if torch.cuda.is_available():
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
            gpu_direct_for_vocab=self._gpu_direct_for_vocab,
            target_num_payloads=self.args.rollout_num_gpus_per_engine,
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
    gpu_direct_for_vocab: bool = False,
    target_num_payloads: int = 0,
) -> tuple[list[ObjectRef], Any]:
    # TODO improve
    long_live_tensors = []

    # PP add-on: vocab/lm_head weights fail with flattened-bucket in PP>1.
    # Route them through per-tensor CUDA IPC instead.
    if gpu_direct_for_vocab and any(_is_vocab_or_lm_head_weight(n) for n, _ in hf_named_tensors):
        return _send_gpu_direct(
            hf_named_tensors,
            ipc_engine=ipc_engine,
            ipc_gather_src=ipc_gather_src,
            ipc_gather_group=ipc_gather_group,
            weight_version=weight_version,
            target_num_payloads=target_num_payloads,
        )

    if getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
        converted_named_tensors_by_dtypes = {"dtype": hf_named_tensors}
    else:
        converted_named_tensors_by_dtypes = {}
        for name, tensor in hf_named_tensors:
            dtype = tensor.dtype
            if dtype not in converted_named_tensors_by_dtypes:
                converted_named_tensors_by_dtypes[dtype] = []
            converted_named_tensors_by_dtypes[dtype].append((name, tensor))

    serialized_tensors = []
    for _dtype, named_tensors in converted_named_tensors_by_dtypes.items():
        flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        metadata = flattened_tensor_bucket.get_metadata()
        flattened_tensor_data = {
            "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
            "metadata": metadata,
        }
        long_live_tensors.append(flattened_tensor_data)
        serialized_tensors.append(MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True))

    serialized_named_tensors = (
        [None] * dist.get_world_size(ipc_gather_group) if ipc_gather_src == dist.get_rank() else None
    )
    dist.gather_object(
        serialized_tensors,
        object_gather_list=serialized_named_tensors,
        dst=ipc_gather_src,
        group=ipc_gather_group,
    )

    refs = []
    if dist.get_rank() == ipc_gather_src:
        # For cross-PP stage-local sync, the IPC group may be smaller than the
        # SGLang engine's TP size.  SGLang workers index by tp_rank, so pad the
        # gathered list to target_num_payloads.  After TP all-gather all entries
        # are identical, so replication is safe.
        if target_num_payloads > 0 and len(serialized_named_tensors) < target_num_payloads:
            n = len(serialized_named_tensors)
            serialized_named_tensors = [
                serialized_named_tensors[i % n] for i in range(target_num_payloads)
            ]

        # TODO: here we assume all ranks have the same number of dtypes, not sure if that is correct.
        num_dtypes = len(serialized_named_tensors[0])
        for i in range(num_dtypes):
            kwargs = {
                "serialized_named_tensors": [tensors[i] for tensors in serialized_named_tensors],
                "load_format": "flattened_bucket",
                "weight_version": str(weight_version),
            }
            refs.append(ipc_engine.update_weights_from_tensor.remote(**kwargs))

    return refs, long_live_tensors


# ---------------------------------------------------------------------------
# PP add-on: per-tensor CUDA IPC for vocab/lm_head in PP>1
# ---------------------------------------------------------------------------

def _send_gpu_direct(
    hf_named_tensors, *, ipc_engine, ipc_gather_src, ipc_gather_group, weight_version,
    target_num_payloads: int = 0,
) -> tuple[list[ObjectRef], list]:
    """Send per-tensor CUDA IPC handles (used for vocab/lm_head in PP>1)."""
    long_live_tensors = [tensor for _, tensor in hf_named_tensors]
    local_payload = MultiprocessingSerializer.serialize(hf_named_tensors, output_str=True)
    gathered = [None] * dist.get_world_size(ipc_gather_group) if ipc_gather_src == dist.get_rank() else None
    dist.gather_object(local_payload, object_gather_list=gathered, dst=ipc_gather_src, group=ipc_gather_group)

    refs = []
    if dist.get_rank() == ipc_gather_src:
        # Pad to target size for SGLang TP workers (same as flattened bucket path)
        if target_num_payloads > 0 and len(gathered) < target_num_payloads:
            n = len(gathered)
            gathered = [gathered[i % n] for i in range(target_num_payloads)]
        refs.append(
            ipc_engine.update_weights_from_tensor.remote(
                serialized_named_tensors=gathered, load_format=None, weight_version=str(weight_version),
            )
        )
    return refs, long_live_tensors

