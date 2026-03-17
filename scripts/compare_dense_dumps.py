"""Compare Megatron vs SGLang layer dumps for dense Qwen3-0.6B.

Usage: python compare_dense_dumps.py [--meg-fwd N] [--sgl-fwd N] [--num-layers N]
  --meg-fwd: Megatron forward pass index (default: 0)
  --sgl-fwd: SGLang forward_pass_id (default: auto-detect first available)
  --num-layers: number of transformer layers (default: env NUM_LAYERS or 28)
"""
import torch
import glob
import os
import sys
import argparse

MEG_DIR = os.environ.get("MEG_DUMP_DIR", "/tmp/megatron_debug")
SGL_DIRS = sorted(glob.glob("/tmp/sglang_dump_*/"))
if not SGL_DIRS:
    print("No SGLang dump dirs found")
    sys.exit(1)
SGL_DIR = SGL_DIRS[-1]
print(f"Megatron dump dir: {MEG_DIR}")
print(f"SGLang dump dir:   {SGL_DIR}")


def detect_meg_fwd_for_sgl_rank(sgl_fwd, sgl_rank=0):
    """Find which Megatron fwd index corresponds to a given SGLang rank.

    In TP>1 setups, Megatron dumps fwd0..fwd(TP-1) from different TP ranks.
    The rank ordering may differ from SGLang's.  We match using position-0
    fingerprints from tensors that have identical per-head layout after split
    (q_before_qknorm, k_after_rope, V).
    Returns the matching fwd index, or None.
    """
    fingerprints = [
        "layer00_attn_q_before_qknorm",
        "layer00_attn_k_after_rope",
        "layer00_attn_v",
    ]
    for fp_name in fingerprints:
        s = load_sgl(fp_name, sgl_fwd)
        if s is None:
            continue
        s0 = _extract_pos0(s)
        for fwd in range(20):
            m = load_meg(fp_name, fwd)
            if m is None:
                break
            m0 = _extract_pos0(m)
            if m0.numel() == s0.numel() and (m0 - s0).abs().max() == 0:
                print(f"  (matched via {fp_name} pos0)")
                return fwd
    return None


def find_sgl_fwd_ids():
    """Find all available SGLang forward_pass_ids for rank=0."""
    ids = set()
    for f in glob.glob(os.path.join(SGL_DIR, "forward_pass_id=*___rank=0___*")):
        basename = os.path.basename(f)
        fwd_str = basename.split("___")[0].split("=")[1]
        ids.add(int(fwd_str))
    return sorted(ids)


def load_sgl(name, fwd_id):
    pattern = f"forward_pass_id={fwd_id}___rank=0___name={name}___*"
    matches = glob.glob(os.path.join(SGL_DIR, pattern))
    if not matches:
        return None
    return torch.load(matches[0], map_location="cpu", weights_only=True)


def load_meg(name, fwd=0):
    path = os.path.join(MEG_DIR, f"{name}_fwd{fwd}.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=True)


def _stats(t):
    """Return (mean, std, absmax) of a float tensor."""
    f = t.float().flatten()
    return f.mean().item(), f.std().item(), f.abs().max().item()


def _full_stats(t):
    """Return (mean, std, min, max) of a float tensor."""
    f = t.float().flatten()
    return f.mean().item(), f.std().item(), f.min().item(), f.max().item()


def _extract_pos0(t):
    """Extract position-0 vector and flatten to 1-D regardless of shape.

    Megatron shapes: [seq, 1, D], [seq, 1, H, hn], [seq, H, hn]
    SGLang shapes:   [seq, D], [seq, H*hn]
    In all cases t[0] gives the position-0 slice.
    """
    return t[0].float().flatten()


def _flatten_to_2d(t):
    """Flatten any tensor to [tokens, hidden_dim] regardless of extra dims."""
    f = t.float()
    if f.ndim <= 1:
        return f.unsqueeze(0)
    return f.reshape(-1, f.shape[-1])


def compare(name_meg, name_sgl, label, meg_fwd, sgl_fwd):
    """Element-wise comparison (requires same semantic layout)."""
    m = load_meg(name_meg, meg_fwd)
    s = load_sgl(name_sgl, sgl_fwd)
    if m is None:
        print(f"  [{label}] Megatron '{name_meg}_fwd{meg_fwd}' NOT FOUND")
        return
    if s is None:
        print(f"  [{label}] SGLang '{name_sgl}' fwd={sgl_fwd} NOT FOUND")
        return

    m_flat = _flatten_to_2d(m)
    s_flat = _flatten_to_2d(s)

    mm, ms, ma = _stats(m)
    sm, ss, sa = _stats(s)
    print(f"  [{label}]")
    print(f"    Megatron: shape={list(m.shape)} dtype={m.dtype} "
          f"mean={mm:.6f} std={ms:.6f} absmax={ma:.6f}")
    print(f"    SGLang:   shape={list(s.shape)} dtype={s.dtype} "
          f"mean={sm:.6f} std={ss:.6f} absmax={sa:.6f}")

    if m_flat.shape != s_flat.shape:
        n = min(m_flat.shape[0], s_flat.shape[0])
        d = min(m_flat.shape[1], s_flat.shape[1])
        print(f"    SHAPE MISMATCH: meg={list(m_flat.shape)} sgl={list(s_flat.shape)}")
        print(f"    Comparing first {n} tokens, {d} dims")
        m_flat = m_flat[:n, :d]
        s_flat = s_flat[:n, :d]

    diff = (m_flat - s_flat).abs()
    nonzero = (diff > 0).sum().item()
    print(f"    Diff: max={diff.max():.8f} mean={diff.mean():.8f} "
          f"nonzero={nonzero}/{diff.numel()}")
    if diff.max() == 0:
        print(f"    >>> BITWISE IDENTICAL")
    elif diff.max() < 1e-5:
        print(f"    ~   CLOSE (max < 1e-5)")
    else:
        print(f"    *** DIVERGENT")


def compare_stats_only(name_meg, name_sgl, label, meg_fwd, sgl_fwd):
    """Layout-agnostic comparison: only compare aggregate statistics.

    Useful for tensors with different element ordering (e.g. interleaved vs
    concatenated QKV).  If weights are identical, the *set* of output elements
    is the same regardless of ordering, so mean/std/absmax must match exactly.
    """
    m = load_meg(name_meg, meg_fwd)
    s = load_sgl(name_sgl, sgl_fwd)
    if m is None:
        print(f"  [{label}] Megatron '{name_meg}_fwd{meg_fwd}' NOT FOUND")
        return
    if s is None:
        print(f"  [{label}] SGLang '{name_sgl}' fwd={sgl_fwd} NOT FOUND")
        return

    mm, ms, ma = _stats(m)
    sm, ss, sa = _stats(s)
    print(f"  [{label}]")
    print(f"    Megatron: shape={list(m.shape)} dtype={m.dtype} "
          f"mean={mm:.6f} std={ms:.6f} absmax={ma:.6f}")
    print(f"    SGLang:   shape={list(s.shape)} dtype={s.dtype} "
          f"mean={sm:.6f} std={ss:.6f} absmax={sa:.6f}")

    mean_match = abs(mm - sm) < 1e-6
    std_match = abs(ms - ss) < 1e-4
    absmax_match = abs(ma - sa) < 1e-4
    ok = mean_match and std_match and absmax_match
    tag = ">>> STATS MATCH" if ok else "*** STATS DIFFER"
    print(f"    mean_diff={abs(mm-sm):.8f}  std_diff={abs(ms-ss):.8f}  "
          f"absmax_diff={abs(ma-sa):.8f}  {tag}")

    mf = m.float().flatten().sort().values
    sf = s.float().flatten().sort().values
    if mf.shape == sf.shape:
        sorted_diff = (mf - sf).abs()
        print(f"    Sorted-elem diff: max={sorted_diff.max():.8f} "
              f"mean={sorted_diff.mean():.8f}")
        if sorted_diff.max() == 0:
            print(f"    >>> SORTED BITWISE IDENTICAL (same multiset)")
        elif sorted_diff.max() < 1e-5:
            print(f"    ~   SORTED CLOSE")
        else:
            print(f"    *** SORTED DIVERGENT (different values)")
    else:
        print(f"    (numel differs: meg={mf.numel()} sgl={sf.numel()}, skip sorted)")


def compare_pos0_stats(name_meg, name_sgl, label, meg_fwd, sgl_fwd):
    """Layout-agnostic comparison using only position 0.

    For tensors with different element ordering (interleaved vs concatenated),
    position 0's element *multiset* is identical when weights are synced
    correctly, so mean/std/min/max must match exactly.

    Uses only position 0 to avoid Megatron's padding (full prefill+decode
    length) vs SGLang's actual sequence length.
    """
    m = load_meg(name_meg, meg_fwd)
    s = load_sgl(name_sgl, sgl_fwd)
    if m is None:
        print(f"  [{label}] Megatron '{name_meg}_fwd{meg_fwd}' NOT FOUND")
        return
    if s is None:
        print(f"  [{label}] SGLang '{name_sgl}' fwd={sgl_fwd} NOT FOUND")
        return

    m0 = _extract_pos0(m)
    s0 = _extract_pos0(s)

    m_mean, m_std, m_min, m_max = _full_stats(m0)
    s_mean, s_std, s_min, s_max = _full_stats(s0)

    print(f"  [{label}] Position-0 stats (numel: meg={m0.numel()}, sgl={s0.numel()})")
    print(f"    Megatron: mean={m_mean:.8f} std={m_std:.8f} min={m_min:.8f} max={m_max:.8f}")
    print(f"    SGLang:   mean={s_mean:.8f} std={s_std:.8f} min={s_min:.8f} max={s_max:.8f}")
    d_mean = abs(m_mean - s_mean)
    d_std = abs(m_std - s_std)
    d_min = abs(m_min - s_min)
    d_max = abs(m_max - s_max)
    print(f"    Diff:     mean={d_mean:.2e} std={d_std:.2e} min={d_min:.2e} max={d_max:.2e}")

    all_zero = d_mean == 0 and d_std == 0 and d_min == 0 and d_max == 0
    all_close = d_mean < 1e-6 and d_std < 1e-4 and d_min < 1e-6 and d_max < 1e-6
    if all_zero:
        print(f"    >>> POS0 STATS BITWISE IDENTICAL")
    elif all_close:
        print(f"    ~   POS0 STATS CLOSE")
    else:
        print(f"    *** POS0 STATS DIFFER")

    if m0.numel() == s0.numel():
        m_sorted = m0.sort().values
        s_sorted = s0.sort().values
        sorted_diff = (m_sorted - s_sorted).abs()
        print(f"    Sorted-elem diff (pos0): max={sorted_diff.max():.8f} "
              f"mean={sorted_diff.mean():.8f}")
        if sorted_diff.max() == 0:
            print(f"    >>> POS0 SORTED BITWISE IDENTICAL (same multiset)")
        elif sorted_diff.max() < 1e-5:
            print(f"    ~   POS0 SORTED CLOSE")
        else:
            print(f"    *** POS0 SORTED DIVERGENT")


def _infer_qkv_partition_config(layer, meg_fwd, head_dim):
    """Infer per-partition QKV config from Megatron q/k dump shapes.

    Returns (num_q_heads_pp, num_kv_heads_pp, heads_per_group) or None.
    Megatron dumps q as [sq, b, ng, (np/ng)*hn] and k as [sq, b, ng, hn].
    """
    q = load_meg(f"layer{layer:02d}_attn_q_before_qknorm", meg_fwd)
    k = load_meg(f"layer{layer:02d}_attn_k_before_qknorm", meg_fwd)
    if q is None or k is None:
        return None
    ng = k.shape[-2]  # num_query_groups_per_partition
    hn = k.shape[-1]  # should equal head_dim (or heads_per_group * head_dim for q)
    if hn != head_dim:
        hn = head_dim
    q_group_dim = q.shape[-1]
    heads_per_group = q_group_dim // head_dim
    num_q_heads_pp = ng * heads_per_group
    num_kv_heads_pp = ng
    return num_q_heads_pp, num_kv_heads_pp, heads_per_group


def compare_qkv_weight_sync(layer, meg_fwd, sgl_fwd, num_heads, num_kv_heads, head_dim):
    """Verify QKV weight sync by rearranging Megatron interleaved -> concatenated
    and comparing position 0 element-wise.

    This is the definitive weight-sync test: after rearranging Megatron's
    interleaved layout to match SGLang's concatenated layout, position 0
    should be bitwise identical if weights were synced correctly.

    Automatically infers per-partition head counts from Megatron q/k dump shapes
    to handle TP correctly.
    """
    m = load_meg(f"layer{layer:02d}_attn_mixed_qkv", meg_fwd)
    s = load_sgl(f"layer{layer:02d}_attn_mixed_qkv", sgl_fwd)
    if m is None or s is None:
        print(f"  [QKV Weight Sync] dump not found (meg={m is not None}, sgl={s is not None})")
        return

    cfg = _infer_qkv_partition_config(layer, meg_fwd, head_dim)
    if cfg is not None:
        num_q_pp, num_kv_pp, hpg = cfg
        print(f"  [QKV Weight Sync] Inferred per-partition config: "
              f"q_heads={num_q_pp} kv_heads={num_kv_pp} heads_per_group={hpg} head_dim={head_dim}")
    else:
        num_q_pp = num_heads
        num_kv_pp = num_kv_heads
        hpg = num_heads // num_kv_heads
        print(f"  [QKV Weight Sync] Using global config (q/k dumps not found): "
              f"q_heads={num_q_pp} kv_heads={num_kv_pp}")

    m0 = _extract_pos0(m)
    s0 = _extract_pos0(s)

    num_groups = num_kv_pp
    group_size = (hpg + 2) * head_dim
    expected = num_groups * group_size
    if m0.numel() != expected:
        print(f"  [QKV Weight Sync] Shape mismatch: pos0 has {m0.numel()} elems, "
              f"expected {num_groups}*{group_size}={expected}")
        return

    m0_grouped = m0.reshape(num_groups, group_size)
    q_parts, k_parts, v_parts = [], [], []
    for g in range(num_groups):
        chunk = m0_grouped[g]
        q_parts.append(chunk[:hpg * head_dim])
        k_parts.append(chunk[hpg * head_dim:(hpg + 1) * head_dim])
        v_parts.append(chunk[(hpg + 1) * head_dim:])
    meg_q = torch.cat(q_parts)
    meg_k = torch.cat(k_parts)
    meg_v = torch.cat(v_parts)
    meg_rearranged = torch.cat([meg_q, meg_k, meg_v])

    q_size = num_q_pp * head_dim
    kv_size = num_kv_pp * head_dim
    sgl_q = s0[:q_size]
    sgl_k = s0[q_size:q_size + kv_size]
    sgl_v = s0[q_size + kv_size:]

    print(f"  [QKV Weight Sync] Position-0 rearranged comparison (layer {layer})")
    for part_name, mp, sp in [("Q", meg_q, sgl_q), ("K", meg_k, sgl_k),
                               ("V", meg_v, sgl_v), ("Full", meg_rearranged, s0)]:
        if mp.numel() != sp.numel():
            print(f"    {part_name}: NUMEL MISMATCH meg={mp.numel()} sgl={sp.numel()}")
            continue
        diff = (mp - sp).abs()
        nonzero = (diff > 0).sum().item()
        if diff.max() == 0:
            tag = ">>> BITWISE IDENTICAL"
        elif diff.max() < 1e-5:
            tag = "~   CLOSE"
        else:
            tag = "*** DIVERGENT"
        print(f"    {part_name}: max_diff={diff.max():.8f} mean_diff={diff.mean():.8f} "
              f"nonzero={nonzero}/{diff.numel()} {tag}")


def compare_qkv_split(layer, meg_fwd, sgl_fwd, num_heads, num_kv_heads, head_dim):
    """Rearrange Megatron interleaved QKV to concatenated, then compare element-wise."""
    m = load_meg(f"layer{layer:02d}_attn_mixed_qkv", meg_fwd)
    s = load_sgl(f"layer{layer:02d}_attn_mixed_qkv", sgl_fwd)
    if m is None or s is None:
        print(f"  [QKV Rearranged] dump not found (meg={m is not None}, sgl={s is not None})")
        return

    cfg = _infer_qkv_partition_config(layer, meg_fwd, head_dim)
    if cfg is not None:
        num_q_pp, num_kv_pp, hpg = cfg
    else:
        num_q_pp, num_kv_pp = num_heads, num_kv_heads
        hpg = num_heads // num_kv_heads

    mf = _flatten_to_2d(m)
    sf = _flatten_to_2d(s)

    num_groups = num_kv_pp
    group_size = (hpg + 2) * head_dim
    if mf.shape[1] != num_groups * group_size:
        print(f"  [QKV Rearranged] dim mismatch: {mf.shape[1]} != {num_groups}*{group_size}")
        return

    mf_grouped = mf.reshape(mf.shape[0], num_groups, group_size)
    q_parts, k_parts, v_parts = [], [], []
    for g in range(num_groups):
        chunk = mf_grouped[:, g, :]
        q_parts.append(chunk[:, :hpg * head_dim])
        k_parts.append(chunk[:, hpg * head_dim:(hpg + 1) * head_dim])
        v_parts.append(chunk[:, (hpg + 1) * head_dim:])
    meg_q = torch.cat(q_parts, dim=-1)
    meg_k = torch.cat(k_parts, dim=-1)
    meg_v = torch.cat(v_parts, dim=-1)

    q_size = num_q_pp * head_dim
    kv_size = num_kv_pp * head_dim
    sgl_q = sf[:, :q_size]
    sgl_k = sf[:, q_size:q_size + kv_size]
    sgl_v = sf[:, q_size + kv_size:]

    n = min(mf.shape[0], sf.shape[0])
    for part_name, mp, sp in [("Q", meg_q[:n], sgl_q[:n]),
                               ("K", meg_k[:n], sgl_k[:n]),
                               ("V", meg_v[:n], sgl_v[:n])]:
        diff = (mp - sp).abs()
        nonzero = (diff > 0).sum().item()
        print(f"  [QKV->{part_name}] meg_tokens={meg_q.shape[0]} sgl_tokens={sgl_q.shape[0]} "
              f"comparing={n} max_diff={diff.max():.8f} mean_diff={diff.mean():.8f} "
              f"nonzero={nonzero}/{diff.numel()}")
        if diff.max() == 0:
            print(f"           >>> BITWISE IDENTICAL")
        elif diff.max() < 1e-5:
            print(f"           ~   CLOSE")
        else:
            print(f"           *** DIVERGENT")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meg-fwd", type=int, default=None,
                        help="Megatron fwd index (default: auto-detect via rank matching)")
    parser.add_argument("--sgl-fwd", type=int, default=None)
    parser.add_argument("--num-layers", type=int,
                        default=int(os.environ.get("NUM_LAYERS", "28")))
    parser.add_argument("--num-heads", type=int, default=16,
                        help="total num_attention_heads")
    parser.add_argument("--num-kv-heads", type=int, default=8,
                        help="total num_key_value_heads (query groups)")
    parser.add_argument("--head-dim", type=int, default=128,
                        help="kv_channels / head_dim")
    args = parser.parse_args()

    sgl_ids = find_sgl_fwd_ids()
    print(f"SGLang forward_pass_ids (rank=0): {sgl_ids}")

    if args.sgl_fwd is not None:
        sgl_fwd = args.sgl_fwd
    elif sgl_ids:
        sgl_fwd = sgl_ids[0]
    else:
        print("No SGLang dumps found")
        return

    if args.meg_fwd is not None:
        meg_fwd = args.meg_fwd
    else:
        meg_fwd = detect_meg_fwd_for_sgl_rank(sgl_fwd, sgl_rank=0)
        if meg_fwd is not None:
            print(f"Auto-detected: Megatron fwd{meg_fwd} matches SGLang rank=0 "
                  f"(via V pos0 fingerprint)")
        else:
            meg_fwd = 0
            print(f"Could not auto-detect rank mapping, defaulting to fwd0")

    print(f"\nComparing: Megatron fwd{meg_fwd} vs SGLang fwd_pass_id={sgl_fwd}")
    print(f"Model config: num_heads={args.num_heads} num_kv_heads={args.num_kv_heads} "
          f"head_dim={args.head_dim}\n")

    print("=== Embedding ===\n")
    compare("layer00_input", "embedding_output", "Embedding / Layer 0 Input", meg_fwd, sgl_fwd)

    attn_dump_points = [
        ("attn_mixed_qkv", "Mixed QKV"),
        ("attn_q_before_qknorm", "Q before QK-Norm"),
        ("attn_k_before_qknorm", "K before QK-Norm"),
        ("attn_q_after_qknorm", "Q after QK-Norm"),
        ("attn_k_after_qknorm", "K after QK-Norm"),
        ("attn_q_after_rope", "Q after RoPE"),
        ("attn_k_after_rope", "K after RoPE"),
        ("attn_v", "V"),
        ("attn_core_out", "Core Attention Out"),
    ]

    for layer in range(args.num_layers):
        lp = f"layer{layer:02d}"
        print(f"\n=== Layer {layer} ===\n")
        compare(f"{lp}_input", f"{lp}_input", "Layer Input", meg_fwd, sgl_fwd)
        compare(f"{lp}_after_input_ln", f"{lp}_after_input_ln", "After Input LN", meg_fwd, sgl_fwd)

        print(f"\n  --- Attention internals (layer {layer}) ---")

        print(f"\n  --- Position-0 stats (layout-agnostic, same multiset) ---")
        for suffix, label in attn_dump_points:
            compare_pos0_stats(f"{lp}_{suffix}", f"{lp}_{suffix}",
                               f"{label} (pos0)", meg_fwd, sgl_fwd)

        print(f"\n  --- QKV weight sync verification (pos0, rearranged) ---")
        compare_qkv_weight_sync(layer, meg_fwd, sgl_fwd,
                                args.num_heads, args.num_kv_heads, args.head_dim)

        print(f"\n  --- Full-tensor stats (layout differs for QKV) ---")
        compare_stats_only(f"{lp}_attn_mixed_qkv", f"{lp}_attn_mixed_qkv",
                           "Mixed QKV (stats only, layout differs)", meg_fwd, sgl_fwd)
        compare_qkv_split(layer, meg_fwd, sgl_fwd,
                          args.num_heads, args.num_kv_heads, args.head_dim)

        print(f"\n  --- Element-wise comparison ---")
        compare(f"{lp}_attn_q_before_qknorm", f"{lp}_attn_q_before_qknorm",
                "Q before QK-Norm", meg_fwd, sgl_fwd)
        compare(f"{lp}_attn_k_before_qknorm", f"{lp}_attn_k_before_qknorm",
                "K before QK-Norm", meg_fwd, sgl_fwd)
        compare(f"{lp}_attn_q_after_qknorm", f"{lp}_attn_q_after_qknorm",
                "Q after QK-Norm", meg_fwd, sgl_fwd)
        compare(f"{lp}_attn_k_after_qknorm", f"{lp}_attn_k_after_qknorm",
                "K after QK-Norm", meg_fwd, sgl_fwd)
        compare(f"{lp}_attn_q_after_rope", f"{lp}_attn_q_after_rope",
                "Q after RoPE", meg_fwd, sgl_fwd)
        compare(f"{lp}_attn_k_after_rope", f"{lp}_attn_k_after_rope",
                "K after RoPE", meg_fwd, sgl_fwd)
        compare(f"{lp}_attn_v", f"{lp}_attn_v", "V", meg_fwd, sgl_fwd)
        compare(f"{lp}_attn_core_out", f"{lp}_attn_core_out",
                "Core Attention Out", meg_fwd, sgl_fwd)

        print()
        compare(f"{lp}_after_attn", f"{lp}_after_attn", "After Attention", meg_fwd, sgl_fwd)
        compare(f"{lp}_moe_input", f"{lp}_moe_input", "MoE Input (post-attn LN)", meg_fwd, sgl_fwd)
        compare(f"{lp}_moe_output", f"{lp}_moe_output", "MoE Output (MLP)", meg_fwd, sgl_fwd)
        compare(f"{lp}_output", f"{lp}_output", "Layer Output", meg_fwd, sgl_fwd)

    print("\n=== Final Norm ===\n")
    compare("before_final_layernorm", "before_final_layernorm_hidden", "Before Final LN", meg_fwd, sgl_fwd)
    compare("after_final_layernorm", "after_final_layernorm", "After Final LN", meg_fwd, sgl_fwd)


if __name__ == "__main__":
    main()
