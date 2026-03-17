"""Shared utilities for kernel alignment tests across model architectures."""
import os
import json
import torch
import safetensors.torch as st


class TestResults:
    """Tracks pass/fail/info counts for alignment tests."""

    def __init__(self):
        self.pass_count = 0
        self.fail_count = 0
        self.info_count = 0

    def check(self, name, a, b, expect_diff=False):
        """Compare two tensors for bitwise equality."""
        diff = (a.float() - b.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        if max_diff == 0:
            print(f"  PASS  {name}: bitwise identical")
            self.pass_count += 1
        elif expect_diff:
            nonzero_frac = (diff > 0).float().mean().item()
            print(
                f"  INFO  {name}: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}, "
                f"nonzero={nonzero_frac:.4f} (expected)"
            )
            self.info_count += 1
        else:
            nonzero_frac = (diff > 0).float().mean().item()
            print(
                f"  FAIL  {name}: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}, "
                f"nonzero={nonzero_frac:.4f}"
            )
            self.fail_count += 1
        return max_diff

    def summary(self):
        """Print final test summary."""
        print("\n" + "=" * 60)
        total = self.pass_count + self.fail_count + self.info_count
        print(
            f"SUMMARY: {self.pass_count}/{total} passed, {self.fail_count}/{total} failed, "
            f"{self.info_count}/{total} info (expected diff)"
        )
        if self.fail_count == 0:
            print("ALL TESTS PASSED — every kernel is bitwise identical!")
        else:
            print("FAILURES detected — fix the failing layers before end-to-end test.")
        print("=" * 60)


class TPTestResults(TestResults):
    """TestResults that only prints on rank 0 (for multi-GPU tests)."""

    def __init__(self, rank):
        super().__init__()
        self.rank = rank

    def check(self, name, a, b, expect_diff=False):
        diff = (a.float() - b.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        if max_diff == 0:
            if self.rank == 0:
                print(f"  PASS  {name}: bitwise identical")
            self.pass_count += 1
        elif expect_diff:
            if self.rank == 0:
                nonzero_frac = (diff > 0).float().mean().item()
                print(
                    f"  INFO  {name}: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}, "
                    f"nonzero={nonzero_frac:.4f} (expected)"
                )
            self.info_count += 1
        else:
            if self.rank == 0:
                nonzero_frac = (diff > 0).float().mean().item()
                print(
                    f"  FAIL  {name}: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}, "
                    f"nonzero={nonzero_frac:.4f}"
                )
            self.fail_count += 1
        return max_diff

    def summary(self):
        if self.rank == 0:
            super().summary()


def load_safetensor_weights(model_path, prefixes=None):
    """Load safetensor weights, optionally filtering by prefix list.

    Args:
        model_path: Path to HuggingFace model directory.
        prefixes: If provided, only load tensors whose names start with one of these prefixes.
                  This saves memory for large models (e.g. only load layer 0).
    """
    with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
        index = json.load(f)

    weights = {}
    loaded_files = set()
    for tensor_name, filename in index["weight_map"].items():
        if prefixes and not any(tensor_name.startswith(p) for p in prefixes):
            continue
        if filename not in loaded_files:
            filepath = os.path.join(model_path, filename)
            shard = st.load_file(filepath)
            if prefixes:
                for k, v in shard.items():
                    if any(k.startswith(p) for p in prefixes):
                        weights[k] = v
            else:
                weights.update(shard)
            loaded_files.add(filename)

    print(f"Loaded {len(weights)} tensors from {len(loaded_files)} shards")
    return weights


def setup_sglang_for_test(model_path, rl_on_policy_target="fsdp"):
    """Set up SGLang server args and batch_invariant_mode for testing.

    This ensures the RMSNorm and other kernels take the correct code path
    (forward_native with cast_x_before_out_mul=True).
    """
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

    server_args = ServerArgs(
        model_path=model_path,
        rl_on_policy_target=rl_on_policy_target,
    )
    set_global_server_args_for_scheduler(server_args)
    print(f"SGLang server_args set: rl_on_policy_target={rl_on_policy_target}")

    enable_batch_invariant_mode(enable_bmm=False)
    print("batch_invariant_mode enabled (matmul_persistent for aten::mm)")
