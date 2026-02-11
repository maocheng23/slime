import gc
import logging
import os
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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


def _is_moe_expert_weight(name: str) -> bool:
    return ".experts." in name


# ---------------------------------------------------------------------------
# Tensor-sync configuration dataclass
# ---------------------------------------------------------------------------

def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0") == "1"


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


@dataclass(frozen=True)
class TensorSyncConfig:
    """Resolved tensor-sync knobs — built once during ``__init__`` from env vars."""

    pp_size: int
    tp_size_for_ipc: int
    ipc_group_size: int
    use_stage_local_ipc_group: bool
    use_gpu_direct: bool
    gpu_direct_ipc_gc_interval: int
    ipc_gc_interval: int

    @classmethod
    def from_args(cls, args: Namespace) -> "TensorSyncConfig":
        pp_size = int(
            getattr(args, "sglang_pipeline_parallel_size", 0)
            or getattr(args, "pipeline_model_parallel_size", 0)
            or 1
        )
        is_pp = pp_size > 1

        use_stage_local = _env_bool("SLIME_TENSOR_SYNC_STAGE_LOCAL_GROUP", is_pp)
        if is_pp and use_stage_local:
            logger.warning(
                "PP stage-local IPC tensor sync is experimental. "
                "If rollout generate fails after update_weights, set "
                "SLIME_TENSOR_SYNC_STAGE_LOCAL_GROUP=0."
            )

        tp_size_for_ipc = int(getattr(args, "sglang_tp_size", 0) or 0)
        if tp_size_for_ipc <= 0:
            tp_size_for_ipc = max(1, args.rollout_num_gpus_per_engine // pp_size)

        ipc_group_size = tp_size_for_ipc if use_stage_local else args.rollout_num_gpus_per_engine

        use_gpu_direct = _env_bool("SLIME_TENSOR_SYNC_GPU_DIRECT", False)

        # PP safety: GPU-direct CUDA IPC is unstable for large PP/MoE payloads.
        # Disable unless the user explicitly opted in via GPU_DIRECT=1.
        if is_pp and use_gpu_direct:
            logger.warning(
                "GPU-direct tensor sync with PP>1 (pp_size=%s) is experimental. "
                "Set SLIME_TENSOR_SYNC_GPU_DIRECT=0 if update_weights fails.",
                pp_size,
            )

        gpu_direct_gc = max(1, _env_int("SLIME_TENSOR_SYNC_GPU_DIRECT_GC_INTERVAL", 8))
        ipc_gc = max(0, _env_int("SLIME_TENSOR_SYNC_IPC_GC_INTERVAL", 4 if is_pp else 0))

        return cls(
            pp_size=pp_size,
            tp_size_for_ipc=tp_size_for_ipc,
            ipc_group_size=ipc_group_size,
            use_stage_local_ipc_group=use_stage_local,
            use_gpu_direct=use_gpu_direct,
            gpu_direct_ipc_gc_interval=gpu_direct_gc,
            ipc_gc_interval=ipc_gc,
        )


# ---------------------------------------------------------------------------
# UpdateWeightFromTensor
# ---------------------------------------------------------------------------


class UpdateWeightFromTensor:
    """
    Update rollout engines from tensor dict:
    load(dict→GPU) → broadcast PP/EP(GPU NCCL) → gather TP(GPU NCCL) → convert HF(GPU) → send.
    Colocated: GPU serialize → gather_object(Gloo, collects from TP/stage-local ranks) → Ray IPC to engine.
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

        self._sync_cfg = TensorSyncConfig.from_args(args)

        world_size = dist.get_world_size()
        assert world_size % self._sync_cfg.ipc_group_size == 0, (
            f"world_size ({world_size}) must be divisible by IPC group size ({self._sync_cfg.ipc_group_size})"
        )

        # Create IPC groups stage-local by default for PP>1 to match SGLang's
        # TP-rank indexing in update_weights_from_tensor.
        self._ipc_group_start_rank = None
        self._ipc_group_index = None
        for start_rank in range(0, world_size, self._sync_cfg.ipc_group_size):
            end_rank = start_rank + self._sync_cfg.ipc_group_size
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if dist.get_rank() in group_ranks:
                self._ipc_gather_group = new_group
                self._ipc_gather_src = start_rank
                self._ipc_group_start_rank = start_rank
                self._ipc_group_index = start_rank // self._sync_cfg.ipc_group_size

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

        cfg = self._sync_cfg
        self._ipc_engine = None
        if self.rollout_engines:
            assert self._ipc_group_index is not None
            engine_index = (
                self._ipc_group_index // cfg.pp_size if cfg.use_stage_local_ipc_group else self._ipc_group_index
            )
            if engine_index < len(self.rollout_engines):
                self._ipc_engine = self.rollout_engines[engine_index]
        if dist.get_rank() == self._ipc_gather_src:
            assert self._ipc_engine is not None, (
                f"Failed to map IPC source rank {dist.get_rank()} to a rollout engine "
                f"(group_index={self._ipc_group_index}, pp_size={cfg.pp_size}, "
                f"num_rollout_engines={len(self.rollout_engines)})"
            )

        logger.info(
            "Tensor weight sync mapping: rank=%s, group_start=%s, group_size=%s, group_index=%s, "
            "has_ipc_engine=%s, use_distribute=%s, sync_cfg=%s",
            dist.get_rank(),
            self._ipc_group_start_rank,
            cfg.ipc_group_size,
            self._ipc_group_index,
            self._ipc_engine is not None,
            self.use_distribute,
            cfg,
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

        cfg = self._sync_cfg
        for bucket_idx, hf_named_tensors in enumerate(self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights)):
            refs, long_lived_tensors = self._send_hf_params(hf_named_tensors)
            ray.get(refs)
            # Non-source ranks must keep tensors alive until the source rank
            # finishes update_weights_from_tensor.remote().
            dist.barrier(group=self._ipc_gather_group)
            del long_lived_tensors, refs, hf_named_tensors
            self._maybe_gc_after_bucket(bucket_idx, cfg)

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

    @staticmethod
    def _maybe_gc_after_bucket(bucket_idx: int, cfg: TensorSyncConfig) -> None:
        """Run periodic GC / IPC cleanup between weight-sync buckets."""
        step = bucket_idx + 1
        if cfg.ipc_gc_interval > 0 and (step % cfg.ipc_gc_interval == 0):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
        if cfg.use_gpu_direct and (step % cfg.gpu_direct_ipc_gc_interval == 0 or bucket_idx == 0):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

    def _send_hf_params(self, hf_named_tensors) -> tuple[list[ObjectRef], Any]:
        all_refs = []

        refs_colocated, long_lived_tensors = _send_to_colocated_engine(
            hf_named_tensors,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
            weight_version=self.weight_version,
            use_gpu_direct=self._sync_cfg.use_gpu_direct,
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


# ---------------------------------------------------------------------------
# Colocated engine send: two strategies + dispatcher
# ---------------------------------------------------------------------------

def _should_use_gpu_direct(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    use_gpu_direct: bool,
) -> bool:
    """Decide whether to use the direct per-tensor CUDA IPC path."""
    if not use_gpu_direct:
        return False
    max_tensors = _env_int("SLIME_TENSOR_SYNC_GPU_DIRECT_MAX_TENSORS", 256)
    if len(hf_named_tensors) <= max_tensors:
        return True
    # Allow forced direct IPC for specific buckets even when count exceeds the limit.
    force_vocab = _env_bool("SLIME_TENSOR_SYNC_GPU_DIRECT_FOR_VOCAB", False) and any(
        _is_vocab_or_lm_head_weight(n) for n, _ in hf_named_tensors
    )
    force_moe = _env_bool("SLIME_TENSOR_SYNC_GPU_DIRECT_FOR_MOE", False) and any(
        _is_moe_expert_weight(n) for n, _ in hf_named_tensors
    )
    if force_vocab or force_moe:
        return True
    logger.warning(
        "Fallback to flattened-bucket GPU transfer: num_tensors=%s > max=%s",
        len(hf_named_tensors),
        max_tensors,
    )
    return False


def _send_gpu_direct(
    hf_named_tensors, *, ipc_engine, ipc_gather_src, ipc_gather_group, weight_version,
) -> tuple[list[ObjectRef], list]:
    """Send per-tensor CUDA IPC handles (fast for small payloads)."""
    long_live_tensors = [tensor for _, tensor in hf_named_tensors]
    local_payload = MultiprocessingSerializer.serialize(hf_named_tensors, output_str=True)
    gathered = [None] * dist.get_world_size(ipc_gather_group) if ipc_gather_src == dist.get_rank() else None
    dist.gather_object(local_payload, object_gather_list=gathered, dst=ipc_gather_src, group=ipc_gather_group)

    refs = []
    if dist.get_rank() == ipc_gather_src:
        refs.append(
            ipc_engine.update_weights_from_tensor.remote(
                serialized_named_tensors=gathered, load_format=None, weight_version=str(weight_version),
            )
        )
    return refs, long_live_tensors


def _send_flattened_bucket(
    hf_named_tensors, *, ipc_engine, ipc_gather_src, ipc_gather_group, weight_version,
) -> tuple[list[ObjectRef], list]:
    """Send flattened-bucket payloads on GPU."""
    long_live_tensors: list = []

    # Group by dtype to avoid misaligned GPU tensor views after reconstruction.
    groups: dict[Any, list] = {}
    for name, tensor in hf_named_tensors:
        groups.setdefault(tensor.dtype, []).append((name, tensor))

    serialized_by_dtype: dict[str, Any] = {}
    for dtype_key_raw, named_tensors in groups.items():
        dtype_key = str(dtype_key_raw)
        bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        metadata = bucket.get_metadata()
        flat = bucket.get_flattened_tensor()

        payload = {"flattened_tensor": flat, "metadata": metadata}
        long_live_tensors.append(payload)
        serialized_by_dtype[dtype_key] = MultiprocessingSerializer.serialize(payload, output_str=True)

    # Gather across IPC group.
    gathered = [None] * dist.get_world_size(ipc_gather_group) if ipc_gather_src == dist.get_rank() else None
    dist.gather_object(serialized_by_dtype, object_gather_list=gathered, dst=ipc_gather_src, group=ipc_gather_group)

    refs: list[ObjectRef] = []
    if dist.get_rank() == ipc_gather_src:
        if not gathered:
            return refs, long_live_tensors
        for dk in sorted(gathered[0].keys()):
            per_rank = []
            for rp in gathered:
                if dk not in rp:
                    raise RuntimeError(f"Missing dtype bucket {dk}; available={sorted(rp.keys())}")
                per_rank.append(rp[dk])
            refs.append(
                ipc_engine.update_weights_from_tensor.remote(
                    serialized_named_tensors=per_rank, load_format="flattened_bucket", weight_version=str(weight_version),
                )
            )
    return refs, long_live_tensors


def _send_to_colocated_engine(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    *,
    ipc_engine,
    ipc_gather_src,
    ipc_gather_group,
    weight_version,
    use_gpu_direct: bool,
) -> tuple[list[ObjectRef], Any]:
    """Dispatch to the appropriate colocated send strategy."""
    common = dict(ipc_engine=ipc_engine, ipc_gather_src=ipc_gather_src, ipc_gather_group=ipc_gather_group, weight_version=weight_version)

    if _should_use_gpu_direct(hf_named_tensors, use_gpu_direct):
        return _send_gpu_direct(hf_named_tensors, **common)

    return _send_flattened_bucket(hf_named_tensors, **common)
