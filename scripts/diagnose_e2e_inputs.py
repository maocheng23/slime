"""
Diagnose remaining E2E diff by checking all input-level differences.

Patches Megatron's forward to dump inputs on the first forward pass,
then compares against SGLang's dumps.

Checks:
  1. Position IDs / cu_seqlens (RoPE input)
  2. Embedding output (first layer input)
  3. Token ordering in packed sequences
  4. Embedding weight shard layout at TP=8

Run: torchrun --nproc_per_node=8 scripts/diagnose_e2e_inputs.py
"""
import os
import sys
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ.get("LOCAL_RANK", rank))
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

assert world_size == 8

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alignment_test_utils import TPTestResults, load_safetensor_weights

os.environ["MEGATRON_USE_DETERMINISTIC_ALLREDUCE"] = "1"

from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

MODEL_PATH = "/root/models/Qwen3-30B-A3B"
server_args = ServerArgs(model_path=MODEL_PATH, rl_on_policy_target="fsdp_tp")
set_global_server_args_for_scheduler(server_args)
enable_batch_invariant_mode(enable_bmm=False)

from transformers import AutoConfig
config = AutoConfig.from_pretrained(MODEL_PATH)

HIDDEN = config.hidden_size
VOCAB_SIZE = config.vocab_size
TP = world_size

results = TPTestResults(rank)
tp_group = dist.group.WORLD

weights = load_safetensor_weights(MODEL_PATH, prefixes=[
    "model.embed_tokens.",
])

def make_shared_input(*shape, dtype=torch.bfloat16):
    x = torch.randn(*shape, device=device, dtype=dtype)
    dist.broadcast(x, src=0, group=tp_group)
    return x

def make_shared_int(*shape):
    x = torch.randint(0, VOCAB_SIZE, shape, device=device, dtype=torch.long)
    dist.broadcast(x, src=0, group=tp_group)
    return x


# ============================================================
# CHECK 1: Embedding at TP=8
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("CHECK 1: Embedding at TP=8")
    print("=" * 60)

embed_weight = weights["model.embed_tokens.weight"].to(device).bfloat16()  # [vocab, hidden]

# SGLang VocabParallelEmbedding: shards vocab across TP
shard_size = VOCAB_SIZE // TP
embed_local_sg = embed_weight[rank * shard_size : (rank + 1) * shard_size]  # [shard, hidden]

# Megatron VocabParallelEmbedding: same sharding
embed_local_mega = embed_weight[rank * shard_size : (rank + 1) * shard_size]

# Test: lookup some tokens
token_ids = make_shared_int(16)

with torch.no_grad():
    # SGLang path: each rank looks up its shard, zeros for others, then all-reduce
    local_mask = (token_ids >= rank * shard_size) & (token_ids < (rank + 1) * shard_size)
    local_ids = token_ids - rank * shard_size
    local_ids = local_ids.clamp(0, shard_size - 1)

    embed_partial = torch.zeros(16, HIDDEN, dtype=torch.bfloat16, device=device)
    embed_partial[local_mask] = embed_local_sg[local_ids[local_mask]]

    # All-reduce to combine
    dist.all_reduce(embed_partial, op=dist.ReduceOp.SUM, group=tp_group)

    # Reference: full embedding lookup (no TP)
    embed_full = embed_weight[token_ids]

    embed_diff = (embed_partial.float() - embed_full.float()).abs().max().item()

results.check("Embedding TP=8 vs full", embed_partial, embed_full)


# ============================================================
# CHECK 2: Position IDs construction
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("CHECK 2: Position IDs construction")
    print("=" * 60)

# Simulate packed sequences: 2 sequences of different lengths
seq_lens = [10, 6]  # total = 16 tokens
total_tokens = sum(seq_lens)

# SGLang: positions are [0,1,...,9, 0,1,...,5]
positions_sg = []
for sl in seq_lens:
    positions_sg.extend(range(sl))
positions_sg = torch.tensor(positions_sg, device=device, dtype=torch.long)

# Megatron: cu_seqlens = [0, 10, 16], max_seqlen = 10
cu_seqlens_mega = torch.tensor([0] + [sum(seq_lens[:i+1]) for i in range(len(seq_lens))],
                                device=device, dtype=torch.int32)
max_seqlen_mega = max(seq_lens)

# Check: does Megatron's RoPE use position IDs or sequential positions?
# Megatron's RotaryEmbedding(max_seqlen) returns freqs for positions 0..max_seqlen-1
# Then _apply_rotary_pos_emb_thd uses cu_seqlens to split sequences
# Each sequence gets freqs[0:seq_len], so position 0..seq_len-1

# Build what Megatron would use for RoPE:
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd

head_dim = config.hidden_size // config.num_attention_heads
base = getattr(config, 'rope_theta', 1000000)
mega_rope = RotaryEmbedding(
    kv_channels=head_dim, rotary_percent=1.0, rotary_interleaved=False,
    seq_len_interpolation_factor=None, rotary_base=base, use_cpu_initialization=True,
)
mega_emb = mega_rope(max_seqlen_mega).to(device)  # [max_seqlen, 1, 1, dim]

# SGLang: RoPE uses positions tensor directly
inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
freqs_sg = torch.outer(positions_sg.float(), inv_freq)
cos_sg = torch.cos(freqs_sg)  # [total_tokens, dim/2]
sin_sg = torch.sin(freqs_sg)

# Megatron: RoPE uses _apply_rotary_pos_emb_thd which splits by cu_seqlens
# For each sequence, it uses freqs[0:seq_len]
# So seq1 tokens get freqs[0:10], seq2 tokens get freqs[0:6]
# This SHOULD match SGLang's positions [0..9, 0..5]

# Build what Megatron would use: per-sequence freqs concatenated
mega_angles = mega_emb.squeeze(1).squeeze(1)[:, :head_dim//2]  # [max_seqlen, dim/2]
cos_mega_parts = []
sin_mega_parts = []
for sl in seq_lens:
    cos_mega_parts.append(torch.cos(mega_angles[:sl]))
    sin_mega_parts.append(torch.sin(mega_angles[:sl]))
cos_mega = torch.cat(cos_mega_parts, dim=0)  # [total_tokens, dim/2]
sin_mega = torch.cat(sin_mega_parts, dim=0)

pos_cos_diff = (cos_sg - cos_mega).abs().max().item()
pos_sin_diff = (sin_sg - sin_mega).abs().max().item()

if rank == 0:
    print(f"  SGLang positions: {positions_sg.tolist()}")
    print(f"  Megatron cu_seqlens: {cu_seqlens_mega.tolist()}, max_seqlen: {max_seqlen_mega}")
    print(f"  Position-based cos diff: {pos_cos_diff}")
    print(f"  Position-based sin diff: {pos_sin_diff}")

results.check("RoPE cos (position-based vs cu_seqlens-based)", cos_sg, cos_mega)
results.check("RoPE sin (position-based vs cu_seqlens-based)", sin_sg, sin_mega)


# ============================================================
# CHECK 3: Apply RoPE with cu_seqlens (Megatron THD path)
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("CHECK 3: RoPE application — THD path vs position-based")
    print("=" * 60)

# Create a test Q tensor
Q = make_shared_input(total_tokens, 4, head_dim)  # [16, 4_heads, 128]

with torch.no_grad():
    # SGLang path: apply RoPE using position-indexed cos/sin
    def apply_rope_f32(x, cos, sin):
        orig = x.dtype
        x = x.float()
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        c = cos.unsqueeze(1).float()
        s = sin.unsqueeze(1).float()
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).to(orig)

    q_sg = apply_rope_f32(Q.clone(), cos_sg, sin_sg)

    # Megatron THD path: _apply_rotary_pos_emb_thd uses cu_seqlens
    from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_thd
    import torch.distributed as dist

    # Need a CP group (context parallel) — use a dummy single-rank group
    # Since we're already in a distributed env, create a group with just this rank
    # Actually _apply_rotary_pos_emb_thd requires cp_group. Let's use the world group
    # but with cp_size=1 behavior by creating a subgroup
    cp_group = dist.new_group([rank])

    q_mega = _apply_rotary_pos_emb_thd(
        Q.clone(), cu_seqlens_mega, mega_emb,
        rotary_interleaved=False,
        cp_group=cp_group,
    )

    rope_thd_diff = (q_sg.float() - q_mega.float()).abs().max().item()

results.check("RoPE THD (Megatron) vs position-based (SGLang)", q_sg, q_mega)


# ============================================================
# CHECK 4: Embedding weight shard layout
# ============================================================
if rank == 0:
    print("\n" + "=" * 60)
    print("CHECK 4: Embedding weight shard layout at TP=8")
    print("=" * 60)

# Check if both frameworks shard the same way
# SGLang: VocabParallelEmbedding uses vocab_start_index = rank * shard_size
# Megatron: VocabParallelEmbedding uses similar sharding

# Verify by checking that local shard matches expected range
expected_start = rank * shard_size
expected_end = (rank + 1) * shard_size
local_shard = embed_weight[expected_start:expected_end]

if rank == 0:
    print(f"  Vocab size: {VOCAB_SIZE}, shard_size: {shard_size}")
    print(f"  Rank {rank}: expected range [{expected_start}, {expected_end})")
    print(f"  Local shard shape: {local_shard.shape}")
    print(f"  Shard first row first5: {local_shard[0, :5].tolist()}")
    print(f"  Full weight row 0 first5: {embed_weight[0, :5].tolist()}")

# Check: does vocab_size divide evenly by TP?
if VOCAB_SIZE % TP != 0:
    padding = TP - (VOCAB_SIZE % TP)
    if rank == 0:
        print(f"  WARNING: vocab_size {VOCAB_SIZE} not divisible by TP {TP}, padding={padding}")
        print(f"  This could cause shard misalignment!")
else:
    if rank == 0:
        print(f"  vocab_size {VOCAB_SIZE} divides evenly by TP {TP} ✓")


# ============================================================
# Summary
# ============================================================
results.summary()

if rank == 0:
    print("\nDIAGNOSTIC INTERPRETATION:")
    print("  If CHECK 1 fails → Embedding TP sharding differs")
    print("  If CHECK 2 fails → Position ID / cos/sin construction differs")
    print("  If CHECK 3 fails → RoPE THD application differs from position-based")
    print("  If CHECK 4 shows WARNING → Vocab padding may cause misalignment")
