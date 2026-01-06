from argparse import Namespace
from collections.abc import Callable, Iterator
from typing import Any

import torch
from megatron.core import mpu
from torch.utils.checkpoint import checkpoint

from slime.utils.distributed_utils import distributed_masked_whiten
from slime.utils.misc import load_function
from slime.utils.ppo_utils import (
    calculate_log_probs_and_entropy,
    compute_approx_kl,
    compute_gspo_kl,
    compute_opsm_mask,
    compute_policy_loss,
    get_advantages_and_returns_batch,
    get_grpo_returns,
    get_reinforce_plus_plus_baseline_advantages,
    get_reinforce_plus_plus_returns,
)
from slime.utils.types import RolloutBatch

from .cp_utils import all_gather_with_cp, get_logits_and_tokens_offset_with_cp, get_sum_of_sample_mean


def get_responses(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield response-aligned `(logits_chunk, tokens_chunk)` pairs per sample.

    After squeezing batch dimension and applying temperature scaling, this
    function extracts the logits and tokens corresponding to response segments
    for each sample. When context parallelism is disabled, it slices directly
    from the concatenated sequence. With context parallelism enabled, it
    handles split sequences across ranks.

    Args:
        logits: Model outputs with shape `[1, T, V]` (policy) or `[1, T, 1]`
            (value). Must be float32.
        args: Configuration containing `rollout_temperature` for scaling.
        unconcat_tokens: List of token tensors (prompt+response) per sample.
        total_lengths: Total sequence lengths (prompt+response) per sample.
        response_lengths: Response segment lengths per sample.

    Yields:
        Tuple of `(logits_chunk, tokens_chunk)` where `logits_chunk` is shape
        `[R, V]` (policy) or `[R, 1]` (value) and `tokens_chunk` is shape `[R]`
        (1D int64), both aligned to response tokens for one sample.
    """
    assert logits.size(0) == 1, f"{logits.shape}"
    assert logits.dtype == torch.float32, f"{logits.dtype}"

    logits = logits.squeeze(0)
    # For true on-policy mode, match SGLang's EXACT temperature handling:
    # SGLang: logits.bfloat16().div(sampling_info.temperatures).bfloat16()
    # where sampling_info.temperatures is a float32 tensor (see sampling_batch_info.py:76-78)
    # The division bfloat16 / float32 produces float32 intermediate result,
    # which is then converted back to bfloat16.
    # This intermediate precision matters for numerical consistency!
    if getattr(args, "true_on_policy_mode", False):
        # Match SGLang's EXACT computation path when rl_on_policy_target is set:
        # SGLang: logits.bfloat16().div(sampling_info.temperatures).bfloat16()
        # where sampling_info.temperatures is a float32 tensor (sampling_batch_info.py:76-78)
        # The division bfloat16 / float32 produces float32 intermediate result.
        # Create float32 temperature tensor to match SGLang's precision
        temp_tensor = torch.tensor(
            args.rollout_temperature,
            dtype=torch.float32,
            device=logits.device
        )
        import os
        if os.environ.get("SLIME_DEBUG_LOGPROB_DIFF", "0") == "1":
            import logging
            debug_logger = logging.getLogger(__name__)
            debug_logger.info("=" * 80)
            debug_logger.info("DEBUG: Megatron temperature processing")
            debug_logger.info("=" * 80)
            debug_logger.info(f"  logits shape (before temp): {logits.shape}, dtype: {logits.dtype}")
            debug_logger.info(f"  logits min (before temp): {logits.min().item():.8f}")
            debug_logger.info(f"  logits max (before temp): {logits.max().item():.8f}")
            debug_logger.info(f"  logits sum (before temp): {logits.sum().item():.8f}")
            debug_logger.info(f"  logits first row, first 10 (before temp): {logits[0, :10].tolist() if logits.dim() >= 2 else logits[:10].tolist()}")
            debug_logger.info(f"  temp_tensor shape: {temp_tensor.shape}, dtype: {temp_tensor.dtype}")
            debug_logger.info(f"  temp_tensor value: {temp_tensor.item():.8f}")
            debug_logger.info(f"  args.rollout_temperature: {args.rollout_temperature}")
            
        
        #logits = logits.bfloat16().div(temp_tensor).bfloat16()
        logits = logits.div(args.rollout_temperature)
        if os.environ.get("SLIME_DEBUG_LOGPROB_DIFF", "0") == "1":
            # Convert back to bfloat16
            debug_logger.info(f"  logits_after_temp shape: {logits.shape}, dtype: {logits.dtype}")
            debug_logger.info(f"  logits_after_temp min: {logits.min().item():.8f}")
            debug_logger.info(f"  logits_after_temp max: {logits.max().item():.8f}")
            debug_logger.info(f"  logits_after_temp sum: {logits.sum().item():.8f}")
            debug_logger.info(f"  logits_after_temp first row, first 10: {logits[0, :10].tolist() if logits.dim() >= 2 else logits[:10].tolist()}")
            debug_logger.info("=" * 80)
    else:
        logits = logits.div(args.rollout_temperature)

    cp_size = mpu.get_context_parallel_world_size()
    end = 0
    for tokens, total_length, response_length in zip(unconcat_tokens, total_lengths, response_lengths, strict=False):
        if cp_size == 1:
            end += total_length
            start = end - response_length
            logits_chunk = logits[start - 1 : end - 1]
            tokens_chunk = tokens[-response_length:]
        else:
            # TODO: this is super ugly... do better abstraction.
            chunk_size, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length
            )

            logits_0, logits_1 = logits[end : end + chunk_size], logits[end + chunk_size : end + 2 * chunk_size]
            end += 2 * chunk_size

            logits_0 = logits_0[logits_offset[0][0] - chunks_offset[0][0] : logits_offset[0][1] - chunks_offset[0][0]]
            tokens_0 = tokens[tokens_offset[0][0] : tokens_offset[0][1]]

            logits_1 = logits_1[logits_offset[1][0] - chunks_offset[1][0] : logits_offset[1][1] - chunks_offset[1][0]]
            tokens_1 = tokens[tokens_offset[1][0] : tokens_offset[1][1]]

            assert logits_0.size(0) == tokens_0.size(0), f"{logits_0.size(0)} vs {tokens_0.size(0)}"
            assert logits_1.size(0) == tokens_1.size(0), f"{logits_1.size(0)} vs {tokens_1.size(0)}"

            logits_chunk = torch.cat([logits_0, logits_1], dim=0)
            tokens_chunk = torch.cat([tokens_0, tokens_1], dim=0)

        yield logits_chunk, tokens_chunk


def get_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
) -> dict[str, list[torch.Tensor]]:
    """Compute per-token log-probabilities (and optionally entropy) on responses.

    For each sample, extracts response-aligned logits and tokens, then computes
    log-probabilities via softmax across the tensor-parallel group. Log-probs
    are squeezed from `[R, 1]` to `[R]`. Entropy values are always appended
    (even when `with_entropy=False`), but only included in the result dict
    when requested.

    Args:
        logits: Policy logits with shape `[1, T, V]`.
        args: Configuration (temperature applied in `get_responses`).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: If True, include "entropy" key in result.
        non_loss_data: Unused; kept for API compatibility.

    Returns:
        Dict with key "log_probs" mapping to a list of `[R]` tensors per
        sample. If `with_entropy` is True, also includes "entropy" key with
        a list of `[R]` tensors.
    """
    assert non_loss_data
    log_probs_list = []
    entropy_list = []
    
    # Debug logging for logprob computation
    import os
    debug_logprob = os.environ.get("SLIME_DEBUG_LOGPROB_DIFF", "0") == "1"
    sample_idx = 0
    
    # Save original (raw) logits before temperature processing for debug comparison
    raw_logits_for_debug = None
    if debug_logprob:
        # logits shape: [1, T, V], squeeze to [T, V]
        raw_logits_for_debug = logits.squeeze(0).clone()
    
    for logits_chunk, tokens_chunk in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
    ):
        if os.environ.get("SLIME_DEBUG_LOGPROB_DIFF", "0") == "1":
            import logging
            debug_logger = logging.getLogger(__name__)
            debug_logger.info("=" * 80)
            debug_logger.info("DEBUG: Megatron temperature processing")
            debug_logger.info("=" * 80)
            debug_logger.info(f"  logits_chunk position 0 shape: {logits_chunk[0, :].shape}, dtype: {logits_chunk[0, :].dtype}")
            debug_logger.info(f"  logits_chunk position 0 min: {logits_chunk[0, :].min().item():.8f}")
            debug_logger.info(f"  logits_chunk position 0 max: {logits_chunk[0, :].max().item():.8f}")
            debug_logger.info(f"  logits_chunk position 0 sum: {logits_chunk[0, :].sum().item():.8f}")
            debug_logger.info(f"  logits_chunk position 0 first 10: {logits_chunk[0, :10].tolist()}")
        log_prob, entropy = calculate_log_probs_and_entropy(
            logits_chunk,
            tokens_chunk,
            mpu.get_tensor_model_parallel_group(),
            with_entropy=with_entropy,
            chunk_size=args.log_probs_chunk_size,
            true_on_policy_mode=getattr(args, "true_on_policy_mode", False),
        )
        
        # Debug: print detailed info for first sample
        sample_idx += 1

        log_probs_list.append(log_prob.squeeze(-1))
        entropy_list.append(entropy)
        
        # Save logprobs for debugging (save first sample's first response token)
        if os.environ.get("MEGATRON_TENSOR_DUMP_DIR", "") and len(log_probs_list) == 1:
            from slime.backends.megatron_utils.debug_tensor_dump import get_megatron_tensor_dumper
            dumper = get_megatron_tensor_dumper()
            if dumper is not None and len(log_probs_list) > 0:
                # log_prob is [R] where R is response length
                # We want the first token of the response
                first_logprob = log_probs_list[0][0:1] if len(log_probs_list[0]) > 0 else log_probs_list[0]
                dumper.add_logprobs(first_logprob)
                # Dump all tensors (including logits and logprobs) after forward pass
                dumper.dump_current_tensors()

    res = {
        "log_probs": log_probs_list,
    }
    if with_entropy:
        res["entropy"] = entropy_list
    return res


def get_values(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
) -> dict[str, list[torch.Tensor]]:
    """Extract per-token value predictions over response tokens.

    For each sample, extracts response-aligned chunks from the value head
    output and squeezes the final dimension from `[R, 1]` to `[R]`.

    Args:
        logits: Value head output with shape `[1, T, 1]`.
        args: Configuration (passed to `get_responses` which uses
            `rollout_temperature` even though values don't need temperature).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: Unused; kept for signature compatibility.
        non_loss_data: Unused; kept for signature compatibility.

    Returns:
        Dict with key "values" mapping to a list of `[R]` value tensors
        per sample.
    """
    value_list = []
    for logits_chunk, _ in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
    ):
        assert logits_chunk.size(-1) == 1, f"{logits_chunk.shape}"
        value_list.append(logits_chunk.squeeze(-1))

    return {
        "values": value_list,
    }


def compute_advantages_and_returns(args: Namespace, rollout_data: RolloutBatch) -> None:
    """Compute advantages and returns in-place based on `args.advantage_estimator`.

    This function extracts rewards, log-probs, values, and masks from
    `rollout_data`, computes KL divergences, then applies the chosen advantage
    estimator. Supported methods: "grpo", "gspo", "ppo", "reinforce_plus_plus",
    and "reinforce_plus_plus_baseline". When `args.normalize_advantages` is
    True, advantages are whitened across the data-parallel group using masked
    statistics.

    Early returns if both `log_probs` and `values` are None (intermediate
    pipeline stages).

    Args:
        args: Configuration specifying estimator type, KL coefficient,
            normalization settings, and other hyperparameters.
        rollout_data: Dict containing input lists ("log_probs", "ref_log_probs",
            "rewards", "values", "response_lengths", "loss_masks",
            "total_lengths"). Modified in-place to add "advantages" and
            "returns" keys, each mapping to lists of tensors per sample.
    """
    log_probs: list[torch.Tensor] = rollout_data.get("rollout_log_probs" if args.use_rollout_logprobs else "log_probs")
    ref_log_probs: list[torch.Tensor] = rollout_data.get("ref_log_probs")
    rewards: list[float] = rollout_data.get("rewards")
    values: None | list[torch.Tensor] = rollout_data.get("values")
    response_lengths: list[int] = rollout_data.get("response_lengths")
    loss_masks: list[torch.Tensor] = rollout_data.get("loss_masks")
    total_lengths: list[int] = rollout_data.get("total_lengths")

    # return when not the last pp stage.
    if log_probs is None and values is None:
        return

    if args.kl_coef == 0 or not log_probs:
        # when kl_coef is 0, we won't compute ref_log_prob
        xs = log_probs if log_probs is not None else values
        kl = [torch.zeros_like(x, dtype=torch.float32, device=x.device) for x in xs]
    else:
        kl = [
            compute_approx_kl(
                log_probs[i],
                ref_log_probs[i],
                kl_loss_type=args.kl_loss_type,
            )
            for i in range(len(log_probs))
        ]

    if args.advantage_estimator in ["grpo", "gspo"]:
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_grpo_returns(rewards, kl)
        # TODO: is the copy necessary?
        advantages = [r for r in returns]

    elif args.advantage_estimator == "ppo":
        old_rewards = rewards
        rewards = []
        kl_coef = -args.kl_coef
        cp_rank = mpu.get_context_parallel_rank()
        for reward, k in zip(old_rewards, kl, strict=False):
            k *= kl_coef
            if cp_rank == 0:
                k[-1] += reward
            rewards.append(k)
        advantages, returns = get_advantages_and_returns_batch(
            total_lengths, response_lengths, values, rewards, args.gamma, args.lambd
        )

    elif args.advantage_estimator == "reinforce_plus_plus":
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_reinforce_plus_plus_returns(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            response_lengths=response_lengths,
            total_lengths=total_lengths,
            kl_coef=args.kl_coef,
            gamma=args.gamma,
        )
        advantages = [r for r in returns]

    elif args.advantage_estimator == "reinforce_plus_plus_baseline":
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        advantages = get_reinforce_plus_plus_baseline_advantages(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            kl_coef=args.kl_coef,
        )
        returns = advantages

    elif args.advantage_estimator == "on_policy_distillation":
        student_log_probs = log_probs
        teacher_log_probs = rollout_data.get("teacher_log_probs")
        response_lengths = rollout_data.get("response_lengths")
        device = student_log_probs[0].device
        teacher_log_probs = [t_log_prob.to(device=device) for t_log_prob in teacher_log_probs]
        teacher_log_probs = [
            t_log_prob[-response_length:]
            for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
        ]
        advantages = [
            teacher_log_prob - student_log_prob
            for teacher_log_prob, student_log_prob in zip(teacher_log_probs, student_log_probs, strict=False)
        ]
        returns = advantages

    else:
        raise NotImplementedError(f"advantage_estimator {args.advantage_estimator} is not supported. ")

    # TODO: OpenRLHF always does advantages normalization but veRL doesn't seem to do it.
    if args.normalize_advantages:
        all_advs = torch.cat(advantages)
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size == 1:
            all_masks = torch.cat(loss_masks)
        else:
            mask_chunks = []
            for i in range(len(advantages)):
                total_len = total_lengths[i]
                response_len = response_lengths[i]
                prompt_len = total_len - response_len

                _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(total_len, response_len)

                # Convert global offsets to response-space offsets
                s0, e0 = token_offsets[0]
                s1, e1 = token_offsets[1]
                res_s0, res_e0 = max(0, s0 - prompt_len), max(0, e0 - prompt_len)
                res_s1, res_e1 = max(0, s1 - prompt_len), max(0, e1 - prompt_len)

                local_mask_parts = []
                full_mask = loss_masks[i]
                if res_e0 > res_s0:
                    local_mask_parts.append(full_mask[res_s0:res_e0])
                if res_e1 > res_s1:
                    local_mask_parts.append(full_mask[res_s1:res_e1])

                # Concatenate the parts to form the final mask chunk for this rank and this sequence
                local_mask_chunk = (
                    torch.cat(local_mask_parts)
                    if local_mask_parts
                    else torch.tensor([], device=all_advs.device, dtype=full_mask.dtype)
                )
                mask_chunks.append(local_mask_chunk)

            all_masks = torch.cat(mask_chunks)

        if all_masks.numel() > 0:
            assert (
                all_advs.size() == all_masks.size()
            ), f"Shape mismatch before whitening: advantages {all_advs.size()}, masks {all_masks.size()}"
            dp_group = mpu.get_data_parallel_group()

            whitened_advs_flat = distributed_masked_whiten(
                all_advs,
                all_masks,
                process_group=dp_group,
                shift_mean=True,
            )
            chunk_lengths = [chunk.size(0) for chunk in advantages]
            advantages = list(torch.split(whitened_advs_flat, chunk_lengths))

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns


def vanilla_tis_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    tis = torch.exp(old_log_probs - rollout_log_probs)
    tis_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
    tis_clipfrac = (tis_weights != tis).float()
    metrics = {
        "tis": tis.clone().detach(),
        "tis_clipfrac": tis_clipfrac.clone().detach(),
        "tis_abs": tis_abs.clone().detach(),
    }
    pg_loss = pg_loss * tis_weights
    return pg_loss, loss_masks, metrics


def icepop_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    ice_ratio = torch.exp(old_log_probs - rollout_log_probs)
    ice_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    ice_weight = torch.where(
        (ice_ratio >= args.tis_clip_low) & (ice_ratio <= args.tis_clip), ice_ratio, torch.zeros_like(ice_ratio)
    )
    ice_clipfrac = (ice_weight != ice_ratio).float()
    metrics = {
        "tis": ice_ratio.clone().detach(),
        "tis_clipfrac": ice_clipfrac.clone().detach(),
        "tis_abs": ice_abs.clone().detach(),
    }
    pg_loss = pg_loss * ice_weight
    return pg_loss, loss_masks, metrics


def policy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute policy loss (PPO/GSPO) and metrics.

    Computes current log-probabilities and entropy from model logits, then
    calculates PPO-style clipped policy gradient loss. For GSPO, gathers
    full sequences via context-parallel all-gather before computing per-sample
    KL. Optionally applies TIS (Truncated Importance Sampling) correction and
    adds KL loss term if configured.

    Args:
        args: Configuration controlling advantage estimator, clipping thresholds,
            entropy/KL coefficients, and TIS settings.
        batch: Mini-batch containing "advantages", "log_probs" (old policy),
            "unconcat_tokens", "response_lengths", "total_lengths", "loss_masks",
            and optionally "ref_log_probs" and "rollout_log_probs".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and `metrics`
        is a dict containing detached scalars: "loss", "pg_loss",
        "entropy_loss", "pg_clipfrac", "ppo_kl". Additional keys "kl_loss",
        "tis", "ois", "tis_clipfrac" are included when the respective features
        are enabled.
    """
    advantages = torch.cat(batch["advantages"], dim=0)
    old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    # Debug: print raw logits before processing
    import os
    if os.environ.get("SLIME_DEBUG_LOGPROB_DIFF", "0") == "1":
        import logging
        debug_logger = logging.getLogger(__name__)
        debug_logger.info("=" * 60)
        debug_logger.info("DEBUG: policy_loss_function - RAW LOGITS")
        debug_logger.info("=" * 60)
        debug_logger.info(f"  logits shape: {logits.shape}, dtype: {logits.dtype}")
        # Get first response token position
        prompt_len = total_lengths[0] - response_lengths[0]
        first_resp_logits_pos = prompt_len - 1  # position that predicts first response token
        debug_logger.info(f"  prompt_len: {prompt_len}, first_resp_logits_pos: {first_resp_logits_pos}")
        
        # Get the target token (first response token)
        first_resp_token = batch["unconcat_tokens"][0][prompt_len].item()
        debug_logger.info(f"  first response token id: {first_resp_token}")
        
        # Print logit for this position
        if first_resp_logits_pos < logits.shape[1]:
            logit_at_pos = logits[0, first_resp_logits_pos, :]
            debug_logger.info(f"  logits[0, {first_resp_logits_pos}, :] stats:")
            debug_logger.info(f"    min: {logit_at_pos.min().item():.6f}")
            debug_logger.info(f"    max: {logit_at_pos.max().item():.6f}")
            debug_logger.info(f"    mean: {logit_at_pos.float().mean().item():.6f}")
            debug_logger.info(f"    logit for target token {first_resp_token}: {logit_at_pos[first_resp_token].item():.6f}")
            # first 10 logits
            debug_logger.info(f"    first 10 logits: {logit_at_pos[:10].tolist()}")
            # Top-5 logits at this position
            top_vals, top_ids = torch.topk(logit_at_pos, 5)
            debug_logger.info(f"    Top-5: {list(zip(top_ids.tolist(), [f'{v:.4f}' for v in top_vals.tolist()]))}")

    log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=True,
    )

    log_probs = log_probs_and_entropy["log_probs"]

    # Pre-gather log probs if needed by OPSM or GSPO to avoid duplicate gathering
    need_full_log_probs = args.use_opsm or args.advantage_estimator == "gspo"

    full_log_probs = None
    full_old_log_probs = None
    if need_full_log_probs:
        full_log_probs = [
            all_gather_with_cp(log_prob, total_length, response_length)
            for log_prob, total_length, response_length in zip(
                log_probs, total_lengths, response_lengths, strict=False
            )
        ]
        full_old_log_probs = [
            all_gather_with_cp(old_log_prob, total_length, response_length)
            for old_log_prob, total_length, response_length in zip(
                old_log_probs, total_lengths, response_lengths, strict=False
            )
        ]

    # Compute OPSM mask if enabled
    if args.use_opsm:
        opsm_mask, opsm_clipfrac = compute_opsm_mask(
            args=args,
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            advantages=batch["advantages"],
            loss_masks=batch["loss_masks"],
        )

    # Compute KL divergence (GSPO uses sequence-level KL, others use per-token KL)
    if args.advantage_estimator == "gspo":
        ppo_kl = compute_gspo_kl(
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            local_log_probs=log_probs,
            loss_masks=batch["loss_masks"],
        )
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
    else:
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        ppo_kl = old_log_probs - log_probs

    pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)

    if args.use_opsm:
        pg_loss = pg_loss * opsm_mask

    # Apply off-policy correction using importance sampling if enabled
    if args.get_mismatch_metrics or args.use_tis:
        # NOTE:
        # `tis_func` may apply rejection-sampling style masking (RS) and return `modified_response_masks`.
        # We rebuild `sum_of_sample_mean` with those masks to correct denominators for loss/backprop.
        #
        # However, mismatch/TIS/RS metrics (e.g., "truncate_fraction") are often defined over the
        # *pre-RS* valid tokens. If we aggregate metrics with `modified_response_masks`, the rejected
        # tokens are excluded from the denominator and the metric can be artificially driven to 0.
        # Keep a copy of the original reducer (based on `batch["loss_masks"]`) for metric aggregation.
        sum_of_sample_mean_for_mismatch_metrics = sum_of_sample_mean

        assert "rollout_log_probs" in batch, "rollout_log_probs must be provided for TIS"

        ois = (-ppo_kl).exp()
        tis_kwargs = {
            "args": args,
            "pg_loss": pg_loss,
            "train_log_probs": batch["log_probs"],
            "rollout_log_probs": batch["rollout_log_probs"],
            "loss_masks": batch["loss_masks"],
            "total_lengths": total_lengths,
            "response_lengths": response_lengths,
        }

        if args.custom_tis_function_path is not None:
            tis_func = load_function(args.custom_tis_function_path)
        else:
            tis_func = vanilla_tis_function
        pg_loss, modified_response_masks, tis_metrics = tis_func(**tis_kwargs)

        # [decouple IS and rejection] Rebuild sum_of_sample_mean with modified_response_masks for denominator correction
        # modified_response_masks will be sliced with cp in get_sum_of_sample_mean
        sum_of_sample_mean = get_sum_of_sample_mean(
            total_lengths, response_lengths, modified_response_masks, args.calculate_per_token_loss
        )

    pg_loss = sum_of_sample_mean(pg_loss)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
    ppo_kl = sum_of_sample_mean(ppo_kl)

    # entropy loss
    entropy = log_probs_and_entropy["entropy"]
    entropy = torch.cat(entropy, dim=0)
    entropy_loss = sum_of_sample_mean(entropy)

    loss = pg_loss - args.entropy_coef * entropy_loss

    if args.use_kl_loss:
        ref_log_probs = batch["ref_log_probs"]
        ref_log_probs = torch.cat(ref_log_probs, dim=0)
        importance_ratio = None
        if args.use_unbiased_kl:
            importance_ratio = torch.exp(log_probs - old_log_probs)
        kl = compute_approx_kl(
            log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
            importance_ratio=importance_ratio,
        )
        kl_loss = sum_of_sample_mean(kl)

        loss = loss + args.kl_loss_coef * kl_loss

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    # Compare rollout vs. train log probs when they originate from different stages.
    # NOTE: In true_on_policy_mode, this comparison measures how much the model weights
    # have changed between rollout and training. If weights haven't changed, this should be ~0.
    # The difference increases as training progresses because model weights are updated.
    train_rollout_logprob_abs_diff = None
    if "rollout_log_probs" in batch and batch["rollout_log_probs"]:
        # old_log_probs: computed with CURRENT training weights (via get_log_probs_and_entropy)
        #   - If use_rollout_logprobs=False: old_log_probs = batch["log_probs"] (recomputed)
        #   - If use_rollout_logprobs=True: old_log_probs = batch["rollout_log_probs"] (from rollout)
        # rollout_log_probs: computed with ROLLOUT-TIME weights (from SGLang)
        #   - Always from batch["rollout_log_probs"] (computed during rollout phase)
        #
        # Both should have the same length (sum(response_lengths)) and correspond to the same tokens.
        # In true_on_policy_mode, if weights haven't changed, these should match exactly.
        # The difference indicates how much the model has changed between rollout and training.
        rollout_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)
        
        # Verify lengths match (sanity check)
        if old_log_probs.shape[0] != rollout_log_probs.shape[0]:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(
                f"Length mismatch: old_log_probs.shape={old_log_probs.shape}, "
                f"rollout_log_probs.shape={rollout_log_probs.shape}. "
                f"This indicates a bug in data processing or unpacking."
            )
        
        train_rollout_logprob_abs_diff = sum_of_sample_mean((old_log_probs - rollout_log_probs).abs())
        if os.environ.get("SLIME_DEBUG_LOGPROB_DIFF", "0") == "2":
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("train_rollout_logprob_abs_diff: " + str(train_rollout_logprob_abs_diff.item()))
            logger.warning("old_log_probs[:10]: " + str(old_log_probs[:10].tolist()))
            logger.warning("rollout_log_probs[:10]: " + str(rollout_log_probs[:10].tolist()))
            logger.warning("old_log_probs.shape: " + str(old_log_probs.shape) + "max: " + str(old_log_probs.max().item()) + "min: " + str(old_log_probs.min().item()) + "sum: " + str(old_log_probs.sum().item()))
            logger.warning("rollout_log_probs.shape: " + str(rollout_log_probs.shape) + "max: " + str(rollout_log_probs.max().item()) + "min: " + str(rollout_log_probs.min().item()) + "sum: " + str(rollout_log_probs.sum().item()))
            logger.warning("abs_diff[:100]: " + str((old_log_probs[:100] - rollout_log_probs[:100]).abs().tolist()))
            logger.warning("sum of abs_diff[:100]: " + str((old_log_probs[:100] - rollout_log_probs[:100]).abs().sum().item()))
            logger.warning("sum of abs_diff[:200]: " + str((old_log_probs[:200] - rollout_log_probs[:200]).abs().sum().item()))
            logger.warning("sum of abs_diff[:300]: " + str((old_log_probs[:300] - rollout_log_probs[:300]).abs().sum().item()))
            # logger.warning("sum of abs_diff[:400]: " + str((old_log_probs[:400] - rollout_log_probs[:400]).abs().sum().item()))
            # logger.warning("sum of abs_diff[:500]: " + str((old_log_probs[:500] - rollout_log_probs[:500]).abs().sum().item()))
            # logger.warning("sum of abs_diff[:600]: " + str((old_log_probs[:600] - rollout_log_probs[:600]).abs().sum().item()))
            logger.warning("sum of abs_diff[-1]: " + str((old_log_probs[-1] - rollout_log_probs[-1]).abs().item()))
            logger.warning("max of abs_diff: " + str((old_log_probs - rollout_log_probs).abs().max().item()))
            logger.warning("mean of abs_diff: " + str((old_log_probs - rollout_log_probs).abs().mean().item()))
            
            # Show sample boundaries and differences per sample
            abs_diff = (old_log_probs - rollout_log_probs).abs()
            response_lengths = batch["response_lengths"]
            total_lengths = batch["total_lengths"]
            sample_boundaries = [0]
            for rl in response_lengths:
                sample_boundaries.append(sample_boundaries[-1] + rl)
            
            logger.warning("\n  Sample-by-sample analysis (RESPONSE ONLY):")
            logger.warning(f"    Number of samples: {len(response_lengths)}")
            logger.warning(f"    Sample boundaries: {sample_boundaries}")
            for i in range(len(response_lengths)):
                start = sample_boundaries[i]
                end = sample_boundaries[i + 1]
                sample_diff = abs_diff[start:end]
                sample_max_diff = sample_diff.max().item()
                sample_mean_diff = sample_diff.mean().item()
                sample_sum_diff = sample_diff.sum().item()
                num_nonzero = (sample_diff > 1e-8).sum().item()
                prompt_len = total_lengths[i] - response_lengths[i]
                logger.warning(f"    Sample {i} [response pos {start}:{end}]: "
                             f"response_length={response_lengths[i]}, prompt_len={prompt_len}, "
                             f"max_diff={sample_max_diff:.8e}, mean_diff={sample_mean_diff:.8e}, "
                             f"sum_diff={sample_sum_diff:.8f}, nonzero={num_nonzero}/{response_lengths[i]}")
            
            # Find the first position where difference appears
            nonzero_mask = abs_diff > 1e-8
            if nonzero_mask.any():
                first_diff_idx = nonzero_mask.nonzero(as_tuple=False).flatten()[0].item()
                logger.warning(f"  First difference at position {first_diff_idx}:")
                logger.warning(f"    old_log_probs[{first_diff_idx}]: {old_log_probs[first_diff_idx].item():.8f}")
                logger.warning(f"    rollout_log_probs[{first_diff_idx}]: {rollout_log_probs[first_diff_idx].item():.8f}")
                logger.warning(f"    abs_diff[{first_diff_idx}]: {abs_diff[first_diff_idx].item():.8e}")
                
                # Check which sample this position belongs to
                response_lengths = batch["response_lengths"]
                sample_boundaries = [0]
                for rl in response_lengths:
                    sample_boundaries.append(sample_boundaries[-1] + rl)
                logger.warning(f"  Sample boundaries (cumulative response lengths): {sample_boundaries}")
                for i, boundary in enumerate(sample_boundaries[1:], 1):
                    if first_diff_idx < boundary:
                        sample_idx = i - 1
                        pos_in_sample = first_diff_idx - sample_boundaries[i - 1]
                        logger.warning(f"    Position {first_diff_idx} is in sample {sample_idx}, at position {pos_in_sample} within that sample")
                        logger.warning(f"    Sample {sample_idx} response_length: {response_lengths[sample_idx]}")
                        if i > 0:
                            total_length = batch["total_lengths"][sample_idx]
                            prompt_len = total_length - response_lengths[sample_idx]
                            logger.warning(f"    Sample {sample_idx} total_length: {total_length}, prompt_len: {prompt_len}")
                        break
                
                # Print detailed info for the first difference position
                logger.warning(f"\n  === DETAILED ANALYSIS: Position {first_diff_idx} (First Difference) ===")
                logger.warning(f"  This is response token position {first_diff_idx} (0-indexed within response)")
                
                # Find which sample this belongs to and compute absolute position
                response_lengths = batch["response_lengths"]
                total_lengths = batch["total_lengths"]
                sample_boundaries = [0]
                for rl in response_lengths:
                    sample_boundaries.append(sample_boundaries[-1] + rl)
                
                diff_sample_idx = -1
                pos_in_sample = -1
                absolute_seq_pos = -1
                for i, boundary in enumerate(sample_boundaries[1:], 1):
                    if first_diff_idx < boundary:
                        diff_sample_idx = i - 1
                        pos_in_sample = first_diff_idx - sample_boundaries[i - 1]
                        prompt_len = total_lengths[diff_sample_idx] - response_lengths[diff_sample_idx]
                        # Absolute position in the full sequence
                        # Response token at position pos_in_sample corresponds to sequence position (prompt_len + pos_in_sample)
                        # But the logits for this token are at position (prompt_len + pos_in_sample - 1)
                        absolute_seq_pos = prompt_len + pos_in_sample
                        logits_pos = prompt_len + pos_in_sample - 1  # Logits position (predicts next token)
                        break
                
                if diff_sample_idx != -1:
                    logger.warning(f"  Sample index: {diff_sample_idx}")
                    logger.warning(f"  Position within sample (response): {pos_in_sample}")
                    logger.warning(f"  Prompt length: {total_lengths[diff_sample_idx] - response_lengths[diff_sample_idx]}")
                    logger.warning(f"  Total length: {total_lengths[diff_sample_idx]}")
                    logger.warning(f"  Response length: {response_lengths[diff_sample_idx]}")
                    logger.warning(f"  Absolute sequence position (token position): {absolute_seq_pos}")
                    logger.warning(f"  Logits position (predicts token at {absolute_seq_pos}): {logits_pos}")
                    
                    # Print token ID at this position
                    if "unconcat_tokens" in batch and diff_sample_idx < len(batch["unconcat_tokens"]):
                        tokens = batch["unconcat_tokens"][diff_sample_idx]
                        if absolute_seq_pos < len(tokens):
                            token_id = tokens[absolute_seq_pos].item()
                            logger.warning(f"  Token ID at position {absolute_seq_pos}: {token_id}")
                        else:
                            logger.warning(f"  Token ID: N/A (position {absolute_seq_pos} out of bounds, len={len(tokens)})")
                    
                    logger.warning(f"\n  === POSSIBLE CHUNKED PREFILL BOUNDARY ===")
                    # Check if position 369 might be at a chunked prefill boundary
                    # Common chunked_prefill_size values: 1024, 2048, 4096, 8192, etc.
                    common_chunk_sizes = [256, 512, 1024, 2048, 4096, 8192, 16384]
                    prompt_len = total_lengths[diff_sample_idx] - response_lengths[diff_sample_idx]
                    total_seq_len_at_diff = prompt_len + pos_in_sample
                    
                    logger.warning(f"  Checking if position {pos_in_sample} (absolute {total_seq_len_at_diff}) is at a chunk boundary:")
                    for chunk_size in common_chunk_sizes:
                        # Check if this position is near a chunk boundary
                        # Chunk boundaries are typically at: prompt_len + n * chunk_size
                        chunk_boundary_after_prompt = prompt_len + chunk_size
                        if abs(total_seq_len_at_diff - chunk_boundary_after_prompt) < 10:
                            logger.warning(f"    ⚠️  Position {total_seq_len_at_diff} is near chunk boundary at {chunk_boundary_after_prompt} (chunk_size={chunk_size})")
                            logger.warning(f"    This suggests SGLang may have switched from prefill chunk to decode mode!")
                            logger.warning(f"    Check SGLang's chunked_prefill_size configuration.")
                    
                    # Also check if this is near the transition from prefill to decode
                    # In SGLang, prefill processes prompt + some response tokens, then switches to decode
                    # The transition point depends on chunked_prefill_size
                    if pos_in_sample > 0:
                        logger.warning(f"  Note: Position {pos_in_sample} is response token #{pos_in_sample + 1}")
                        logger.warning(f"  If SGLang uses chunked prefill, tokens after chunk_size may be processed differently.")
                    
                    logger.warning(f"\n  === ACTION REQUIRED ===")
                    logger.warning(f"  To debug this difference, check:")
                    logger.warning(f"  1. SGLang's chunked_prefill_size configuration")
                    logger.warning(f"  2. Whether position {pos_in_sample} is at a chunk boundary")
                    logger.warning(f"  3. Raw logits at sequence position {logits_pos} (before temperature scaling)")
                    logger.warning(f"  4. Temperature scaling process for position {logits_pos}")
                    logger.warning(f"  5. Logits after temperature scaling at position {logits_pos}")
                    logger.warning(f"  6. Compare with SGLang's logits at the same position")
                    logger.warning(f"  In dump files, look for:")
                    logger.warning(f"    - Megatron: logits_pos_{logits_pos} or logits_full[{logits_pos}]")
                    logger.warning(f"    - SGLang: decode pass with first_position={absolute_seq_pos}")
                    logger.warning(f"    - SGLang: Check if this is the first decode pass after prefill chunk")
                
                # Check surrounding positions
                start_idx = max(0, first_diff_idx - 5)
                end_idx = min(len(old_log_probs), first_diff_idx + 6)
                logger.warning(f"\n  Surrounding positions [{start_idx}:{end_idx}]:")
                for i in range(start_idx, end_idx):
                    diff_val = abs_diff[i].item()
                    old_val = old_log_probs[i].item()
                    rollout_val = rollout_log_probs[i].item()
                    marker = " <-- FIRST DIFF" if i == first_diff_idx else ""
                    logger.warning(f"    pos {i}: old={old_val:.8f}, rollout={rollout_val:.8f}, diff={diff_val:.8e}{marker}")
            else:
                logger.warning("  No differences found (all < 1e-8)")
            
        # In true_on_policy_mode, a significant difference means model weights changed,
        # which breaks the true on-policy assumption. This is expected as training progresses.
        if getattr(args, "true_on_policy_mode", False) and train_rollout_logprob_abs_diff > 1e-4:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"true_on_policy_mode: train_rollout_logprob_abs_diff={train_rollout_logprob_abs_diff:.6f} "
                f"is significant (>1e-4). This indicates model weights changed between rollout and training, "
                f"which is expected as training progresses but breaks the strict true on-policy assumption."
            )
            logger.warning("old_log_probs[:10]: " + str(old_log_probs[:10].tolist()))
            logger.warning("rollout_log_probs[:10]: " + str(rollout_log_probs[:10].tolist()))
            logger.warning("old_log_probs.shape: " + str(old_log_probs.shape) + "max: " + str(old_log_probs.max().item()) + "min: " + str(old_log_probs.min().item()) + "sum: " + str(old_log_probs.sum().item()))
            logger.warning("rollout_log_probs.shape: " + str(rollout_log_probs.shape) + "max: " + str(rollout_log_probs.max().item()) + "min: " + str(rollout_log_probs.min().item()) + "sum: " + str(rollout_log_probs.sum().item()))
            logger.warning("abs_diff[:100]: " + str((old_log_probs[:100] - rollout_log_probs[:100]).abs().tolist()))
            logger.warning("sum of abs_diff[:100]: " + str((old_log_probs[:100] - rollout_log_probs[:100]).abs().sum().item()))
            logger.warning("sum of abs_diff[:200]: " + str((old_log_probs[:200] - rollout_log_probs[:200]).abs().sum().item()))
            logger.warning("sum of abs_diff[:300]: " + str((old_log_probs[:300] - rollout_log_probs[:300]).abs().sum().item()))
            logger.warning("sum of abs_diff: " + str((old_log_probs - rollout_log_probs).abs().max().item()))
            
            # Print token IDs for verification
            if "unconcat_tokens" in batch and len(batch["unconcat_tokens"]) > 0:
                first_sample_tokens = batch["unconcat_tokens"][0]
                prompt_len = total_lengths[0] - response_lengths[0]
                response_tokens = first_sample_tokens[prompt_len:prompt_len+10]
                logger.warning(f"First sample response tokens[0:10]: {response_tokens.tolist()}")
                logger.warning(f"prompt_len={prompt_len}, response_len={response_lengths[0]}, total_len={total_lengths[0]}")
            
            # Print detailed comparison for first few positions
            for i in range(min(5, len(old_log_probs))):
                megatron_lp = old_log_probs[i].item()
                sglang_lp = rollout_log_probs[i].item()
                diff = abs(megatron_lp - sglang_lp)
                rel_diff = diff / max(abs(sglang_lp), 1e-10) * 100
                logger.warning(f"  pos {i}: Megatron={megatron_lp:.8f}, SGLang={sglang_lp:.8f}, diff={diff:.8f} ({rel_diff:.2f}%)")
        
        # Debug logging for true on-policy mode
        import os
        if os.environ.get("SLIME_DEBUG_LOGPROB_DIFF", "0") == "1":
            import logging
            debug_logger = logging.getLogger(__name__)
            debug_logger.info("=" * 80)
            debug_logger.info("DEBUG: LOGPROB COMPARISON (train vs rollout)")
            debug_logger.info("=" * 80)
            debug_logger.info(f"  old_log_probs (Megatron computed): shape={old_log_probs.shape}, dtype={old_log_probs.dtype}")
            debug_logger.info(f"  rollout_log_probs (SGLang computed): shape={rollout_log_probs.shape}, dtype={rollout_log_probs.dtype}")
            debug_logger.info(f"  temperature: {args.rollout_temperature}")
            debug_logger.info(f"  true_on_policy_mode: {getattr(args, 'true_on_policy_mode', False)}")
            debug_logger.info(f"  use_rollout_logprobs: {args.use_rollout_logprobs}")
            
            # Print per-sample info
            debug_logger.info("\n  Per-sample info:")
            debug_logger.info(f"    response_lengths: {response_lengths[:5]}... (total {len(response_lengths)} samples)")
            debug_logger.info(f"    total_lengths: {total_lengths[:5]}...")
            
            # Print first few logprobs from each sample
            debug_logger.info("\n  First sample comparison (first 10 tokens):")
            if len(batch["rollout_log_probs"]) > 0:
                sample_rollout = batch["rollout_log_probs"][0]
                sample_train = log_probs[0] if isinstance(log_probs, list) else log_probs[:response_lengths[0]]
                n_show = min(10, len(sample_rollout), len(sample_train))
                debug_logger.info(f"    SGLang rollout logprobs[0:{n_show}]: {sample_rollout[:n_show].tolist()}")
                debug_logger.info(f"    Megatron train logprobs[0:{n_show}]: {sample_train[:n_show].tolist()}")
                diff = (sample_train[:n_show] - sample_rollout[:n_show]).abs()
                debug_logger.info(f"    Abs diff[0:{n_show}]: {diff.tolist()}")
                debug_logger.info(f"    Max diff in sample 0: {(sample_train - sample_rollout).abs().max().item():.8f}")
            
            # Print overall stats
            abs_diff = (old_log_probs - rollout_log_probs).abs()
            debug_logger.info("\n  Overall stats:")
            debug_logger.info(f"    Mean abs diff: {abs_diff.mean().item():.8f}")
            debug_logger.info(f"    Max abs diff: {abs_diff.max().item():.8f}")
            debug_logger.info(f"    Std abs diff: {abs_diff.std().item():.8f}")
            
            # Find and print the worst mismatches
            if abs_diff.numel() > 0:
                max_idx = abs_diff.argmax().item()
                debug_logger.info(f"\n  Worst mismatch at index {max_idx}:")
                debug_logger.info(f"    Megatron: {old_log_probs[max_idx].item():.8f}")
                debug_logger.info(f"    SGLang:   {rollout_log_probs[max_idx].item():.8f}")
                debug_logger.info(f"    Diff:     {abs_diff[max_idx].item():.8f}")
            
            # Check which sample the worst mismatch belongs to
            cumsum = 0
            for i, resp_len in enumerate(response_lengths):
                if cumsum + resp_len > max_idx:
                    debug_logger.info(f"    Sample index: {i}, position within sample: {max_idx - cumsum}")
                    # Print tokens around this position
                    if "unconcat_tokens" in batch:
                        tokens = batch["unconcat_tokens"][i]
                        pos_in_sample = max_idx - cumsum
                        prompt_len = total_lengths[i] - resp_len
                        token_pos = prompt_len + pos_in_sample
                        if token_pos < len(tokens):
                            debug_logger.info(f"    Token at this position: {tokens[token_pos].item()}")
                            if token_pos > 0:
                                debug_logger.info(f"    Previous token (input to model): {tokens[token_pos - 1].item()}")
                    break
                cumsum += resp_len
            
            debug_logger.info("=" * 80)

    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
    }

    if train_rollout_logprob_abs_diff is not None:
        reported_loss["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff.clone().detach()
        print(f"old_log_probs: {old_log_probs.mean().item()}, rollout_log_probs: {rollout_log_probs.mean().item()}")
        reported_loss["train_rollout_mean"] = rollout_log_probs.clone().detach().mean()
        reported_loss["old_log_probs_mean"] = old_log_probs.clone().detach().mean()
    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    if args.get_mismatch_metrics or args.use_tis:
        # Aggregate mismatch/TIS/RS related metrics with the *pre-RS* masks.
        # See comment above where `sum_of_sample_mean_for_mismatch_metrics` is defined.
        reported_loss["ois"] = sum_of_sample_mean_for_mismatch_metrics(ois).clone().detach()
        # Assume all metrics are already cloned and detached
        for metric_key, metric_value in tis_metrics.items():
            key_name = f"{metric_key}"
            reported_loss[key_name] = sum_of_sample_mean_for_mismatch_metrics(metric_value)

    if args.use_opsm:
        reported_loss["opsm_clipfrac"] = opsm_clipfrac

    return loss, reported_loss


def value_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute clipped value loss and metrics.

    Extracts current value predictions from `logits`, compares them against
    stored old values with clipping, and computes the maximum of clipped and
    unclipped squared errors (PPO-style value clipping).

    Args:
        args: Configuration containing `value_clip` threshold.
        batch: Mini-batch with "values" (old predictions), "returns",
            "unconcat_tokens", "total_lengths", and "response_lengths".
        logits: Value head output with shape `[1, T, 1]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and
        `metrics` contains detached scalars "value_loss" and "value_clipfrac".
    """
    old_values = torch.cat(batch["values"], dim=0)

    values = get_values(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
    )
    values = torch.cat([value.flatten() for value in values["values"]], dim=0)

    returns = torch.cat(batch["returns"], dim=0)

    values_clipfrac = torch.abs(values - old_values) > args.value_clip
    values_clipped = old_values + (values - old_values).clamp(-args.value_clip, args.value_clip)
    surr1 = (values_clipped - returns) ** 2
    surr2 = (values - returns) ** 2
    loss = torch.max(surr1, surr2)

    loss = sum_of_sample_mean(loss)
    values_clipfrac = sum_of_sample_mean(values_clipfrac.float())

    # make sure the gradient could backprop correctly.
    if values.numel() == 0:
        loss += 0 * values.sum()

    reported_loss = {
        "value_loss": loss.clone().detach(),
        "value_clipfrac": values_clipfrac.clone().detach(),
    }

    return loss, reported_loss


def sft_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute supervised fine-tuning loss over response tokens.

    Computes log-probabilities of the ground-truth tokens in the response
    segments and returns the negative log-likelihood as the loss.

    Args:
        args: Configuration (passed through to helpers).
        batch: Mini-batch with "unconcat_tokens", "response_lengths", and
            "total_lengths".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `metrics` contains a single detached
        scalar "loss".
    """
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
    )

    log_probs = log_probs_and_entropy["log_probs"]
    log_probs = torch.cat(log_probs, dim=0)
    loss = -sum_of_sample_mean(log_probs)

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    return (
        loss,
        {
            "loss": loss.clone().detach(),
        },
    )


def loss_function(
    args: Namespace,
    batch: RolloutBatch,
    num_microbatches: int,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, int | torch.Tensor, dict[str, list[str] | torch.Tensor]]:
    """Dispatch to the configured loss and rescale for Megatron integration.

    Selects one of "policy_loss", "value_loss", "sft_loss", or a custom loss
    function based on `args.loss_type`, computes the loss and metrics, then
    rescales the loss by micro-batch and parallelism factors to integrate with
    Megatron's gradient accumulation.

    Args:
        args: Configuration specifying `loss_type`, `calculate_per_token_loss`,
            `global_batch_size`, and optionally `custom_loss_function_path`.
        batch: Mini-batch with "loss_masks", "response_lengths", and other
            keys required by the selected loss function.
        num_microbatches: Number of gradient accumulation steps.
        logits: Model outputs (policy or value head).

    Returns:
        Tuple of `(scaled_loss, normalizer, logging_dict)` where:
        - `scaled_loss` is the loss tensor (scalar) rescaled for Megatron.
        - `normalizer` is `num_tokens` (scalar tensor) if
          `args.calculate_per_token_loss` is True, else `1` (int).
        - `logging_dict` has keys "keys" (list of str metric names) and
          "values" (1D tensor: [count, metric1, metric2, ...]).
    """
    num_tokens = sum([torch.clamp_min(loss_mask.sum(), 1) for loss_mask in batch["loss_masks"]])
    num_samples = len(batch["response_lengths"])

    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
    )

    match args.loss_type:
        case "policy_loss":
            func = policy_loss_function
        case "value_loss":
            func = value_loss_function
        case "sft_loss":
            func = sft_loss_function
        case "custom_loss":
            func = load_function(args.custom_loss_function_path)
        case _:
            raise ValueError(f"Unknown loss type: {args.loss_type}")

    if args.recompute_loss_function:
        loss, log = checkpoint(func, args, batch, logits, sum_of_sample_mean)
    else:
        loss, log = func(args, batch, logits, sum_of_sample_mean)

    # Here we need to divide by cp_size because to cancel the multiply in Megatron.
    if not args.calculate_per_token_loss:
        loss = (
            loss
            * num_microbatches
            / args.global_batch_size
            * mpu.get_data_parallel_world_size(with_context_parallel=True)
        )
    else:
        loss = loss * mpu.get_context_parallel_world_size()

    return (
        loss,
        torch.tensor(num_tokens if args.calculate_per_token_loss else 1, device=logits.device),
        {
            "keys": list(log.keys()),
            "values": torch.tensor(
                [
                    num_samples if not args.calculate_per_token_loss else num_tokens,
                ]
                + list(log.values()),
                device=logits.device,
            ),
        },
    )
