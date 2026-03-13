"""
Compare Megatron and SGLang layer dumps by aligning micro-batches.

Megatron dumps per-micro-batch: gdn0_input_mb0.pt, gdn0_input_mb1.pt, ...
  Shape: [B, T, D] (typically [1, seq_len, hidden])

SGLang dumps per-forward-pass: gdn0_input_mb0.pt
  Shape: [total_tokens, D] (all sequences packed flat)

This script finds which slice of SGLang's packed tensor matches each Megatron
micro-batch, then compares all intermediate tensors at those aligned positions.

Usage (inside Docker container):
  python scripts/compare_dumps.py
"""
import os
import glob
import torch


MEGA_DIR = "/tmp/megatron_debug"
SGLANG_DIR = "/tmp/sglang_debug"

LAYERS = ["gdn0", "gdn1", "gdn2"]
STAGES = ["input", "proj_qkvz", "proj_ba", "after_conv", "after_gdr", "after_norm", "output"]


def load_dump(directory, name):
    path = os.path.join(directory, f"{name}.pt")
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")
    return None


def find_mega_mbs(directory, prefix):
    """Find all micro-batch files for a given prefix."""
    pattern = os.path.join(directory, f"{prefix}_mb*.pt")
    files = sorted(glob.glob(pattern))
    return [(f, int(f.split("_mb")[-1].replace(".pt", ""))) for f in files]


def align_and_compare():
    # Find all Megatron micro-batches for gdn0_input
    mega_mbs = find_mega_mbs(MEGA_DIR, "gdn0_input")
    sglang_mbs = find_mega_mbs(SGLANG_DIR, "gdn0_input")

    if not mega_mbs:
        print("ERROR: No Megatron dumps found. Run with SLIME_DEBUG_LAYER_DUMP=1")
        return
    if not sglang_mbs:
        print("ERROR: No SGLang dumps found. Run with SGLANG_DEBUG_LAYER_DUMP=1")
        return

    print(f"Found {len(mega_mbs)} Megatron micro-batches, {len(sglang_mbs)} SGLang forward passes")
    print()

    # Load SGLang's first forward pass (has all tokens packed)
    sglang_input = torch.load(sglang_mbs[0][0], map_location="cpu")  # [total_tokens, D]
    print(f"SGLang gdn0_input_mb0: shape={list(sglang_input.shape)}")

    # For each Megatron micro-batch, find the matching slice in SGLang
    alignments = {}  # mega_mb_idx -> (sglang_start, sglang_end)

    for mega_file, mega_mb_idx in mega_mbs:
        mega_input = torch.load(mega_file, map_location="cpu")  # [B, T, D]
        mega_flat = mega_input.reshape(-1, mega_input.shape[-1])  # [B*T, D]
        num_tokens = mega_flat.shape[0]
        print(f"\nMegatron gdn0_input_mb{mega_mb_idx}: shape={list(mega_input.shape)} ({num_tokens} tokens)")

        # Try to find this sequence in SGLang's packed tensor
        found = False
        for start in range(sglang_input.shape[0] - num_tokens + 1):
            sglang_slice = sglang_input[start:start + num_tokens]
            diff = (mega_flat.float() - sglang_slice.float()).abs().max().item()
            if diff == 0:
                print(f"  MATCH: Megatron mb{mega_mb_idx} = SGLang tokens [{start}:{start + num_tokens}] (bitwise identical)")
                alignments[mega_mb_idx] = (start, start + num_tokens)
                found = True
                break
            elif diff < 0.01:
                print(f"  CLOSE: Megatron mb{mega_mb_idx} ~ SGLang tokens [{start}:{start + num_tokens}] (max_diff={diff:.8f})")
                alignments[mega_mb_idx] = (start, start + num_tokens)
                found = True
                break

        if not found:
            # Try matching just the first token
            first_token = mega_flat[0]
            for start in range(sglang_input.shape[0]):
                diff = (first_token.float() - sglang_input[start].float()).abs().max().item()
                if diff == 0:
                    end = min(start + num_tokens, sglang_input.shape[0])
                    print(f"  PARTIAL: First token matches SGLang position {start}")
                    alignments[mega_mb_idx] = (start, end)
                    found = True
                    break
            if not found:
                print(f"  NO MATCH: Could not find Megatron mb{mega_mb_idx} in SGLang tensor")

    if not alignments:
        print("\nNo alignments found. Cannot compare layers.")
        return

    # Now compare all layers at aligned positions
    print("\n" + "=" * 70)
    print("LAYER-BY-LAYER COMPARISON (aligned micro-batches)")
    print("=" * 70)

    for mega_mb_idx, (sg_start, sg_end) in sorted(alignments.items()):
        print(f"\n--- Megatron mb{mega_mb_idx} ↔ SGLang [{sg_start}:{sg_end}] ---")

        for layer in LAYERS:
            for stage in STAGES:
                name = f"{layer}_{stage}"
                mega_files = find_mega_mbs(MEGA_DIR, name)
                sglang_files = find_mega_mbs(SGLANG_DIR, name)

                mega_t = None
                for f, idx in mega_files:
                    if idx == mega_mb_idx:
                        mega_t = torch.load(f, map_location="cpu")
                        break

                sglang_t = None
                for f, idx in sglang_files:
                    if idx == 0:  # SGLang only has mb0 with all tokens
                        sglang_t = torch.load(f, map_location="cpu")
                        break

                if mega_t is None or sglang_t is None:
                    if mega_t is None and sglang_t is None:
                        continue  # both missing (e.g. after_conv not in SGLang)
                    missing = "Megatron" if mega_t is None else "SGLang"
                    print(f"  {name}: SKIP ({missing} dump missing)")
                    continue

                # Flatten Megatron [B, T, ...] to [B*T, ...]
                mega_flat = mega_t.reshape(-1, mega_t.shape[-1])

                # Slice SGLang to aligned range
                # Need to figure out the right slice for this stage's shape
                sglang_flat = sglang_t.reshape(-1, sglang_t.shape[-1])

                # The token count ratio: gdn0_input has num_tokens per mb,
                # but after_norm/after_gdr may have different shapes due to head reshaping
                mega_tokens = mega_flat.shape[0]
                sglang_tokens = sglang_flat.shape[0]
                num_input_tokens = sg_end - sg_start
                total_sglang_tokens = sglang_input.shape[0]

                if sglang_tokens == total_sglang_tokens:
                    # Same token dimension as input — use same slice
                    sg_slice = sglang_flat[sg_start:sg_end]
                else:
                    # Different shape (e.g. after_norm is [num_tokens * num_heads, head_dim])
                    # Scale the slice proportionally
                    ratio = sglang_tokens / total_sglang_tokens
                    scaled_start = int(sg_start * ratio)
                    scaled_end = int(sg_end * ratio)
                    sg_slice = sglang_flat[scaled_start:scaled_end]

                if mega_flat.shape != sg_slice.shape:
                    print(f"  {name}: SHAPE MISMATCH mega={list(mega_flat.shape)} vs sglang_slice={list(sg_slice.shape)}")
                    continue

                diff = (mega_flat.float() - sg_slice.float()).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()

                if max_diff == 0:
                    print(f"  {name}: PASS (bitwise identical)")
                else:
                    nonzero = (diff > 0).sum().item()
                    total = diff.numel()
                    print(f"  {name}: FAIL max_diff={max_diff:.8f} mean_diff={mean_diff:.8f} nonzero={nonzero}/{total}")


if __name__ == "__main__":
    align_and_compare()
