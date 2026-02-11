import os
import logging
import time

import ray
from ray.exceptions import ActorDiedError, ActorUnavailableError, RaySystemError

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger
from slime.utils.misc import should_run_periodic_action
from slime.utils.tracking_utils import init_tracking

logger = logging.getLogger(__name__)


def train(args):
    configure_logger()
    skip_post_train_sync = os.environ.get("SLIME_SKIP_POST_TRAIN_SYNC", "0") == "1"
    extra_weight_updates = int(os.environ.get("SLIME_DEBUG_EXTRA_WEIGHT_UPDATES", "0"))
    rollout_generate_max_retries = int(os.environ.get("SLIME_ROLLOUT_GENERATE_MAX_RETRIES", "0"))
    rollout_generate_retry_backoff_sec = float(os.environ.get("SLIME_ROLLOUT_GENERATE_RETRY_BACKOFF_SEC", "5"))
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())

    # always update weight first so that sglang has the loaded weights from training.
    actor_model.update_weights()
    if extra_weight_updates > 0:
        logger.info("Running %d extra weight update(s) in debug mode.", extra_weight_updates)
        for i in range(extra_weight_updates):
            logger.info("Extra weight update %d/%d", i + 1, extra_weight_updates)
            actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train():
        if args.offload_train:
            if args.use_critic:
                critic_model.offload()
                if rollout_id >= args.num_critic_only_steps:
                    actor_model.offload()
            else:
                actor_model.offload()
        else:
            actor_model.clear_memory()

    def save(rollout_id):
        if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps):
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.use_critic:
            critic_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_generate_attempt = 0
        while True:
            try:
                rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
                break
            except (ActorUnavailableError, ActorDiedError, RaySystemError) as e:
                if rollout_generate_attempt >= rollout_generate_max_retries:
                    raise
                rollout_generate_attempt += 1
                sleep_s = rollout_generate_retry_backoff_sec * (2 ** (rollout_generate_attempt - 1))
                logger.warning(
                    "rollout generate failed at rollout_id=%d (%s), retrying %d/%d after %.1fs",
                    rollout_id,
                    type(e).__name__,
                    rollout_generate_attempt,
                    rollout_generate_max_retries,
                    sleep_s,
                )
                time.sleep(sleep_s)

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            if rollout_id >= args.num_critic_only_steps:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
            ray.get(critic_train_handle)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            save(rollout_id)

        offload_train()
        should_skip_post_train_sync = skip_post_train_sync and rollout_id == args.num_rollout - 1
        if not should_skip_post_train_sync:
            if args.offload_rollout:
                ray.get(rollout_manager.onload_weights.remote())
            actor_model.update_weights()
            # No need to restore KV/cache after the last rollout iteration.
            # This avoids unnecessary end-of-run memory churn in colocated PP runs.
            if args.offload_rollout and rollout_id < args.num_rollout - 1:
                ray.get(rollout_manager.onload_kv.remote())

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    args = parse_args()
    train(args)
