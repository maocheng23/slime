"""
Megatron Tensor Dumper for layer-by-layer comparison with SGLang.

This module provides a mechanism to dump intermediate tensors from Megatron's
forward pass for comparison with SGLang's tensor dump.

Usage:
1. Set environment variables:
   - MEGATRON_TENSOR_DUMP_DIR: Output directory for tensor dumps
   - MEGATRON_TENSOR_DUMP_LAYERS: Comma-separated layer indices to dump (e.g., "0,1,2")

2. Call register_forward_hooks() on your model before running inference
3. Tensors will be saved to MEGATRON_TENSOR_DUMP_DIR

The naming convention mirrors SGLang's for easy comparison:
- Pass{pass_id:05d}.pt contains a dict of tensor_name -> tensor
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class MegatronTensorDumper:
    """Dumps intermediate tensors from Megatron forward passes for debugging."""

    def __init__(
        self,
        dump_dir: str,
        dump_layers: Optional[list[int]] = None,
        tp_rank: int = 0,
        pp_rank: int = 0,
        tp_size: int = 1,
    ):
        self._dump_dir = Path(dump_dir)
        self._dump_layers = dump_layers
        self._forward_pass_id = 0
        self._current_tensors: dict[str, torch.Tensor] = {}
        self._pid = os.getpid()

        rank = tp_size * pp_rank + tp_rank
        self._process_dir = self._dump_dir / f"TP{tp_rank}_PP{pp_rank}_Rank{rank}_pid{self._pid}"
        self._process_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[MegatronTensorDumper] Initialized. dump_dir={self._process_dir}, dump_layers={dump_layers}")

    def add_tensor(self, name: str, tensor: torch.Tensor | tuple | list) -> None:
        """Add a tensor to the current forward pass collection.

        For first response token comparison with SGLang prefill:
        - SGLang prefill outputs logits from position (prompt_len - 1)
        - These logits predict the token at position prompt_len
        - So we need hidden states at position (prompt_len - 1), not prompt_len

        Megatron layout: Could be [seq_len, batch, hidden] or [batch, seq, hidden]

        Saves hidden states at BOTH positions:
        - {name}: at position (prompt_len - 1) for prefill comparison
        - {name}_at_response_start: at position prompt_len for decode comparison
        """
        prompt_len = getattr(self, '_response_start_pos', 0)
        # For prefill comparison, we need position (prompt_len - 1)
        prefill_pos = max(0, prompt_len - 1)
        # For decode comparison, we need position prompt_len
        decode_pos = prompt_len

        def extract_token_at_position(t: torch.Tensor, pos: int) -> torch.Tensor:
            """Extract token at specified position.

            Handles both formats:
            - [seq_len, batch, hidden] - Megatron default
            - [batch, seq_len, hidden] - Alternative format
            """
            if t.dim() == 3:
                d0, d1, d2 = t.shape
                # Detect format: [batch=1, seq, hidden] vs [seq, batch=1, hidden]
                if d0 == 1 and d1 > 1:
                    # [batch=1, seq_len, hidden] format
                    seq_len = d1
                    if pos >= seq_len:
                        pos = seq_len - 1
                    return t[0, pos:pos+1, :]  # [1, hidden]
                else:
                    # [seq_len, batch, hidden] format
                    seq_len = d0
                    if pos >= seq_len:
                        pos = seq_len - 1
                    token = t[pos:pos+1, :, :]
                    if token.shape[1] == 1:
                        return token[:, 0, :]  # [1, hidden]
                    return token[0, :, :]  # [batch, hidden]
            elif t.dim() == 2:
                # [seq_len, hidden]
                seq_len = t.shape[0]
                if pos >= seq_len:
                    pos = seq_len - 1
                return t[pos:pos+1, :]  # [1, hidden]
            else:
                return t

        def save_at_position(t: torch.Tensor, pos: int, suffix: str = ""):
            key = f"{name}{suffix}"
            extracted = extract_token_at_position(t, pos).cpu()
            self._current_tensors[key] = extracted

        if isinstance(tensor, (tuple, list)):
            tensors = []
            for t in tensor:
                if t is not None and isinstance(t, torch.Tensor):
                    tensors.append(extract_token_at_position(t, prefill_pos).cpu())
            if len(tensors) == 1:
                self._current_tensors[name] = tensors[0]
            elif len(tensors) > 1:
                self._current_tensors[name] = tensors
        elif isinstance(tensor, torch.Tensor):
            # Save at prefill comparison position (prompt_len - 1)
            save_at_position(tensor, prefill_pos, "")
            # Also save at decode position (prompt_len) for reference
            if prefill_pos != decode_pos:
                save_at_position(tensor, decode_pos, "_at_response_start")

    def set_response_start_position(self, pos: int) -> None:
        """Set the response start position (prompt_len).

        This is used to determine extraction positions:
        - prefill comparison: position (pos - 1)
        - decode comparison: position pos
        """
        self._response_start_pos = pos
        logger.info(f"[MegatronTensorDumper] Set response_start_pos={pos}")

    def dump_current_tensors(self) -> None:
        """Dump all collected tensors for the current forward pass."""
        if len(self._current_tensors) == 0:
            return

        tensor_file = self._process_dir / f"Pass{self._forward_pass_id:05d}.pt"
        logger.info(
            f"[MegatronTensorDumper] Dumping pass {self._forward_pass_id} "
            f"with {len(self._current_tensors)} tensors to {tensor_file}"
        )
        torch.save(self._current_tensors, str(tensor_file))
        self._current_tensors = {}
        self._forward_pass_id += 1

    def should_dump_layer(self, layer_idx: int) -> bool:
        """Check if a layer should be dumped."""
        if self._dump_layers is None:
            return True
        return layer_idx in self._dump_layers

    def register_forward_hooks(self, model: torch.nn.Module) -> None:
        """
        Register forward hooks on the model to capture intermediate tensors.

        This hooks into the TransformerLayer modules to capture:
        - Embedding output
        - Input hidden states (before each layer)
        - Output hidden states (after each layer)
        - Attention output (after self-attention)
        - MLP output (after MLP)
        - Final output (lm_head)
        """
        hooks_registered = 0

        # Hook embedding layer - we need to hook the actual word_embeddings,
        # not the full LanguageModelEmbedding which includes transpose and dropout
        embedding = getattr(model, 'embedding', None)
        if embedding is not None:
            # Try to hook the inner word_embeddings for direct comparison with SGLang
            word_embeddings = getattr(embedding, 'word_embeddings', None)
            if word_embeddings is not None:
                word_embeddings.register_forward_hook(self._create_named_hook("model.embed_tokens"))
                hooks_registered += 1
                logger.info("[MegatronTensorDumper] Hooked embedding.word_embeddings layer")
            else:
                # Fall back to the full embedding layer
                embedding.register_forward_hook(self._create_named_hook("model.embed_tokens"))
                hooks_registered += 1
                logger.info("[MegatronTensorDumper] Hooked embedding layer (fallback)")

        # Hook output layer (lm_head)
        output_layer = getattr(model, 'output_layer', None)
        if output_layer is not None:
            output_layer.register_forward_hook(self._create_named_hook("lm_head"))
            hooks_registered += 1
            logger.info("[MegatronTensorDumper] Hooked output_layer (lm_head)")

        # Find the decoder/transformer layers
        layers = self._find_transformer_layers(model)

        for layer_idx, layer in enumerate(layers):
            if not self.should_dump_layer(layer_idx):
                continue

            # Hook for layer input/output
            layer.register_forward_hook(self._create_layer_hook(layer_idx))
            hooks_registered += 1

            # Try to hook into attention and MLP sublayers
            self._hook_sublayers(layer, layer_idx)

        logger.info(f"[MegatronTensorDumper] Registered {hooks_registered} hooks on {len(layers)} layers")

    def _create_named_hook(self, name: str):
        """Create a forward hook for a named module."""

        def hook(module, input_tuple, output):
            if isinstance(output, tuple):
                out_tensor = output[0]
            else:
                out_tensor = output

            if isinstance(out_tensor, torch.Tensor):
                self.add_tensor(name, out_tensor.bfloat16())

        return hook

    def _find_transformer_layers(self, model: torch.nn.Module) -> list[torch.nn.Module]:
        """Find transformer layers in the model."""
        layers = []
        layer_names = []

        # Debug: print model structure
        logger.info("[MegatronTensorDumper] Model structure:")
        for name, module in model.named_modules():
            if "layer" in name.lower() or "decoder" in name.lower():
                logger.info(f"  {name}: {type(module).__name__}")

        # Try to find decoder.layers (Megatron GPTModel structure)
        decoder = getattr(model, 'decoder', None)
        if decoder is not None:
            decoder_layers = getattr(decoder, 'layers', None)
            if decoder_layers is not None and isinstance(decoder_layers, torch.nn.ModuleList):
                layers = list(decoder_layers)
                layer_names = [f"decoder.layers.{i}" for i in range(len(layers))]
                logger.info(f"[MegatronTensorDumper] Found decoder.layers with {len(layers)} layers")
                return layers

        # Try different common structures
        # Megatron GPTModel structure: model.decoder.layers
        for name, module in model.named_modules():
            # Match patterns like "decoder.layers.0", "model.layers.0"
            if ".layers." in name:
                # Extract the layer number from the end
                parts = name.split(".")
                for i, part in enumerate(parts):
                    if part == "layers" and i + 1 < len(parts):
                        try:
                            int(parts[i + 1])  # Validate it's a layer number
                            # Make sure we're at the right level (layer module, not sublayer)
                            if i + 2 == len(parts) or (i + 2 < len(parts) and parts[i + 2] not in ["self_attention", "mlp", "input_layernorm"]):
                                # Check if this is a transformer layer
                                if hasattr(module, "self_attention") or hasattr(module, "input_layernorm"):
                                    if module not in layers:
                                        layers.append(module)
                                        layer_names.append(name)
                        except ValueError:
                            pass
                        break

        if not layers:
            # Fallback: try to find ModuleList named 'layers'
            for name, module in model.named_modules():
                if name.endswith(".layers") and isinstance(module, torch.nn.ModuleList):
                    layers = list(module)
                    layer_names = [f"{name}.{i}" for i in range(len(layers))]
                    break

        logger.info(f"[MegatronTensorDumper] Found layers: {layer_names}")
        return layers

    def _hook_sublayers(self, layer: torch.nn.Module, layer_idx: int) -> None:
        """Hook into attention and MLP sublayers."""
        # Hook input_layernorm output
        if hasattr(layer, "input_layernorm"):
            layer.input_layernorm.register_forward_hook(
                self._create_sublayer_hook(layer_idx, "input_layernorm")
            )

        # Hook post_self_attn_layernorm output (if exists)
        if hasattr(layer, "post_self_attn_layernorm"):
            layer.post_self_attn_layernorm.register_forward_hook(
                self._create_sublayer_hook(layer_idx, "post_attention_layernorm")
            )
        elif hasattr(layer, "pre_mlp_layernorm"):
            layer.pre_mlp_layernorm.register_forward_hook(
                self._create_sublayer_hook(layer_idx, "post_attention_layernorm")
            )

        # Hook attention output
        if hasattr(layer, "self_attention"):
            layer.self_attention.register_forward_hook(
                self._create_sublayer_hook(layer_idx, "self_attention")
            )
            # Also hook core attention if available
            if hasattr(layer.self_attention, "core_attention"):
                layer.self_attention.core_attention.register_forward_hook(
                    self._create_sublayer_hook(layer_idx, "core_attention")
                )

        # Hook MLP output
        if hasattr(layer, "mlp"):
            layer.mlp.register_forward_hook(
                self._create_sublayer_hook(layer_idx, "mlp")
            )
            # Hook sublayers in MLP
            if hasattr(layer.mlp, "linear_fc1"):
                layer.mlp.linear_fc1.register_forward_hook(
                    self._create_sublayer_hook(layer_idx, "mlp.gate_up_proj")
                )
            if hasattr(layer.mlp, "linear_fc2"):
                layer.mlp.linear_fc2.register_forward_hook(
                    self._create_sublayer_hook(layer_idx, "mlp.down_proj")
                )

    def _create_layer_hook(self, layer_idx: int):
        """Create a forward hook for a transformer layer."""

        def hook(module, input_tuple, output):
            # Input is typically (hidden_states, attention_mask, ...)
            if isinstance(input_tuple, tuple) and len(input_tuple) > 0:
                hidden_states = input_tuple[0]
                if isinstance(hidden_states, torch.Tensor):
                    self.add_tensor(f"layer_{layer_idx}_input", hidden_states.bfloat16())

            # Output is typically (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            if isinstance(hidden_states, torch.Tensor):
                self.add_tensor(f"layer_{layer_idx}_output", hidden_states.bfloat16())

        return hook

    def add_input_ids(self, input_ids: torch.Tensor) -> None:
        """Record the input token IDs for debugging."""
        # Save full input_ids for reference
        self._current_tensors["megatron_input_ids"] = input_ids.cpu()
        
        # Save the token at the response start position (what we're actually comparing)
        target_pos = getattr(self, '_response_start_pos', 0)
        flat_ids = input_ids.flatten()
        if target_pos < len(flat_ids):
            self._current_tensors["megatron_compared_token_id"] = flat_ids[target_pos:target_pos+1].cpu()
            self._current_tensors["megatron_compared_position"] = torch.tensor([target_pos])
        else:
            # Fallback to first token
            self._current_tensors["megatron_compared_token_id"] = flat_ids[0:1].cpu()
            self._current_tensors["megatron_compared_position"] = torch.tensor([0])

    def add_logits(self, logits: torch.Tensor) -> None:
        """Record logits for debugging.
        
        For true on-policy comparison:
        - SGLang decode at position N outputs logits predicting token at N+1
        - Megatron logits at position N also predict token at N+1
        
        Saves logits at ALL response positions.
        """
        prompt_len = getattr(self, '_response_start_pos', 0)
        
        if logits.dim() == 3:
            # Detect format: could be [seq, batch, vocab] or [batch, seq, vocab]
            # Heuristic: batch is usually 1, vocab is largest
            d0, d1, d2 = logits.shape
            if d0 == 1 and d1 > 1:
                # [batch=1, seq_len, vocab_size] format
                batch_size = d0
                seq_len = d1
                vocab_size = d2
                is_batch_first = True
                logger.info(
                    "[MegatronTensorDumper] Detected [batch, seq, vocab] format"
                )
            else:
                # [seq_len, batch, vocab_size] - Megatron format
                seq_len = d0
                batch_size = d1
                vocab_size = d2
                is_batch_first = False
                logger.info(
                    "[MegatronTensorDumper] Detected [seq, batch, vocab] format"
                )

            # Save sequence info
            self._current_tensors["seq_len"] = torch.tensor([seq_len])
            self._current_tensors["batch_size"] = torch.tensor([batch_size])
            self._current_tensors["vocab_size"] = torch.tensor([vocab_size])
            self._current_tensors["prompt_len"] = torch.tensor([prompt_len])
            response_len = seq_len - prompt_len
            self._current_tensors["response_len_from_logits"] = torch.tensor([response_len])

            # Save full logits tensor
            self._current_tensors["logits_full"] = logits.cpu().bfloat16()

            # Save logits at EACH response position for decode comparison
            response_positions = []
            for i in range(response_len):
                pos = prompt_len + i
                if pos < seq_len:
                    if is_batch_first:
                        # [batch, seq, vocab] -> take [0, pos, :]
                        pos_logits = logits[0, pos:pos+1, :].cpu().bfloat16()
                    else:
                        # [seq, batch, vocab] -> take [pos, 0, :]
                        pos_logits = logits[pos:pos+1, 0, :].cpu().bfloat16()
                    self._current_tensors[f"logits_pos_{pos}"] = pos_logits
                    response_positions.append(pos)

            self._current_tensors["response_logits_positions"] = torch.tensor(response_positions)

            # For prefill comparison: logits at prompt_len - 1
            if prompt_len > 0 and prompt_len - 1 < seq_len:
                if is_batch_first:
                    prompt_end_logits = logits[0, prompt_len-1:prompt_len, :].cpu().bfloat16()
                else:
                    prompt_end_logits = logits[prompt_len-1:prompt_len, 0, :].cpu().bfloat16()
                self._current_tensors["logits_at_prompt_end"] = prompt_end_logits
                self._current_tensors["logits_prompt_end_pos"] = torch.tensor([prompt_len - 1])

            # Default position for backwards compatibility
            target_pos = prompt_len if prompt_len < seq_len else 0
            self._current_tensors["logits"] = logits.cpu().bfloat16()
            self._current_tensors["logits_pos"] = torch.tensor([target_pos])

            logger.info(f"[MegatronTensorDumper] Saved logits: seq_len={seq_len}, "
                       f"prompt_len={prompt_len}, response_len={response_len}, "
                       f"batch_size={batch_size}, vocab_size={vocab_size}")
            
        elif logits.dim() == 2:
            # [seq_len, vocab_size]
            seq_len = logits.shape[0]
            vocab_size = logits.shape[1]
            
            # Save sequence info
            self._current_tensors["seq_len"] = torch.tensor([seq_len])
            self._current_tensors["vocab_size"] = torch.tensor([vocab_size])
            self._current_tensors["prompt_len"] = torch.tensor([prompt_len])
            response_len = seq_len - prompt_len
            self._current_tensors["response_len_from_logits"] = torch.tensor([response_len])
            
            # Save full logits tensor
            self._current_tensors["logits_full"] = logits.cpu().bfloat16()
            
            # Save logits at EACH response position
            response_positions = []
            for i in range(response_len):
                pos = prompt_len + i
                if pos < seq_len:
                    self._current_tensors[f"logits_pos_{pos}"] = logits[pos:pos+1, :].cpu().bfloat16()
                    response_positions.append(pos)
            
            self._current_tensors["response_logits_positions"] = torch.tensor(response_positions)
            
            # Prefill comparison
            if prompt_len > 0 and prompt_len - 1 < seq_len:
                self._current_tensors["logits_at_prompt_end"] = logits[prompt_len-1:prompt_len, :].cpu().bfloat16()
                self._current_tensors["logits_prompt_end_pos"] = torch.tensor([prompt_len - 1])
            
            # Default position
            target_pos = prompt_len if prompt_len < seq_len else 0
            self._current_tensors["logits"] = logits[target_pos:target_pos+1, :].cpu().bfloat16()
            self._current_tensors["logits_pos"] = torch.tensor([target_pos])
            
            logger.info(f"[MegatronTensorDumper] Saved logits: seq_len={seq_len}, "
                       f"prompt_len={prompt_len}, response_len={response_len}, vocab_size={vocab_size}")
        else:
            self._current_tensors["logits"] = logits.cpu().bfloat16()

    def add_logprobs(self, logprobs: torch.Tensor) -> None:
        """Record log probabilities for debugging.

        IMPORTANT: log_probs is SHIFTED for next-token prediction:
        - log_probs has shape [total_seq_len - 1]
        - log_probs[i] = logprob of token at position (i + 1)

        For first response token at position prompt_len:
        - We need log_probs[prompt_len - 1]
        """
        prompt_len = getattr(self, '_response_start_pos', 0)

        # For first response token: index = prompt_len - 1 (due to shift)
        first_response_idx = max(0, prompt_len - 1)

        # Save debug info
        self._current_tensors["logprobs_prompt_len"] = torch.tensor([prompt_len])

        if logprobs.dim() == 1:
            # [seq_len - 1] (shifted)
            seq_len = logprobs.shape[0]
            self._current_tensors["logprobs_seq_len"] = torch.tensor([seq_len])
            self._current_tensors["logprobs_full"] = logprobs.cpu().bfloat16()

            if first_response_idx < seq_len:
                first_resp_lp = logprobs[first_response_idx].cpu().bfloat16()
                self._current_tensors["logprobs"] = first_resp_lp.unsqueeze(0)
                self._current_tensors["logprobs_extracted_idx"] = torch.tensor(
                    [first_response_idx]
                )
                logger.info(
                    f"[MegatronTensorDumper] Saved logprob for first response: "
                    f"idx={first_response_idx}, value={first_resp_lp.item():.6f}"
                )
            else:
                self._current_tensors["logprobs"] = logprobs[0:1].cpu().bfloat16()

            # Save first few response logprobs
            resp_end = min(seq_len, first_response_idx + 5)
            if first_response_idx < seq_len:
                self._current_tensors["response_logprobs_first5"] = (
                    logprobs[first_response_idx:resp_end].cpu().bfloat16()
                )

        elif logprobs.dim() == 2:
            # Megatron format: [seq_len, batch] or [batch, seq_len]
            # Determine layout
            if logprobs.shape[1] == 1:
                # [seq_len, 1] - Megatron layout
                seq_len = logprobs.shape[0]
                self._current_tensors["logprobs_seq_len"] = torch.tensor([seq_len])
                self._current_tensors["logprobs_full"] = logprobs.cpu().bfloat16()

                if first_response_idx < seq_len:
                    self._current_tensors["logprobs"] = (
                        logprobs[first_response_idx:first_response_idx+1, :].cpu()
                    )
                else:
                    self._current_tensors["logprobs"] = logprobs[0:1, :].cpu()
            else:
                # [1, seq_len] or [batch, seq_len]
                seq_len = logprobs.shape[1]
                self._current_tensors["logprobs_seq_len"] = torch.tensor([seq_len])
                self._current_tensors["logprobs_full"] = logprobs.cpu().bfloat16()

                if first_response_idx < seq_len:
                    self._current_tensors["logprobs"] = (
                        logprobs[:, first_response_idx:first_response_idx+1].cpu()
                    )
                else:
                    self._current_tensors["logprobs"] = logprobs[:, 0:1].cpu()
        else:
            self._current_tensors["logprobs"] = logprobs.cpu().bfloat16()

    def _create_sublayer_hook(self, layer_idx: int, sublayer_name: str):
        """Create a forward hook for a sublayer (attention or MLP)."""

        def hook(module, input_tuple, output):
            if isinstance(output, tuple):
                out_tensor = output[0]
            else:
                out_tensor = output

            if isinstance(out_tensor, torch.Tensor):
                self.add_tensor(f"layer_{layer_idx}_{sublayer_name}_output", out_tensor.bfloat16())

        return hook


# Global dumper instance
_global_dumper: Optional[MegatronTensorDumper] = None


def get_megatron_tensor_dumper() -> Optional[MegatronTensorDumper]:
    """Get the global Megatron tensor dumper if enabled."""
    global _global_dumper

    if _global_dumper is not None:
        return _global_dumper

    dump_dir = os.environ.get("MEGATRON_TENSOR_DUMP_DIR", "")
    if not dump_dir:
        return None

    dump_layers_str = os.environ.get("MEGATRON_TENSOR_DUMP_LAYERS", "")
    dump_layers = None
    if dump_layers_str:
        dump_layers = [int(x.strip()) for x in dump_layers_str.split(",") if x.strip()]

    # Get distributed info
    tp_rank = 0
    pp_rank = 0
    tp_size = 1

    if dist.is_initialized():
        tp_rank = dist.get_rank()
        tp_size = dist.get_world_size()

    _global_dumper = MegatronTensorDumper(
        dump_dir=dump_dir,
        dump_layers=dump_layers,
        tp_rank=tp_rank,
        pp_rank=pp_rank,
        tp_size=tp_size,
    )
    return _global_dumper


def register_megatron_tensor_hooks(model: torch.nn.Module) -> None:
    """Register tensor dump hooks on a Megatron model if dumping is enabled."""
    dumper = get_megatron_tensor_dumper()
    if dumper is not None:
        dumper.register_forward_hooks(model)


def dump_megatron_tensors() -> None:
    """Trigger dumping of collected tensors for the current forward pass."""
    dumper = get_megatron_tensor_dumper()
    if dumper is not None:
        dumper.dump_current_tensors()


def compare_tensor_dumps(
    sglang_dump_dir: str,
    megatron_dump_dir: str,
    pass_id: int = 0,
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Compare tensor dumps from SGLang and Megatron.

    Args:
        sglang_dump_dir: Directory containing SGLang tensor dumps
        megatron_dump_dir: Directory containing Megatron tensor dumps
        pass_id: Forward pass ID to compare
        verbose: Print detailed comparison results

    Returns:
        Dict mapping tensor names to comparison stats
    """
    sglang_path = Path(sglang_dump_dir)
    megatron_path = Path(megatron_dump_dir)

    # Find the Pass files
    sglang_files = list(sglang_path.glob(f"*/Pass{pass_id:05d}.pt"))
    megatron_files = list(megatron_path.glob(f"*/Pass{pass_id:05d}.pt"))

    if not sglang_files:
        print(f"No SGLang dump found for pass {pass_id} in {sglang_dump_dir}")
        return {}
    if not megatron_files:
        print(f"No Megatron dump found for pass {pass_id} in {megatron_dump_dir}")
        return {}

    sglang_tensors = torch.load(sglang_files[0], map_location="cpu")
    megatron_tensors = torch.load(megatron_files[0], map_location="cpu")

    results = {}

    # Create name mapping from SGLang to Megatron naming
    # SGLang: model.layers.0, model.layers.0.self_attn, model.layers.0.mlp
    # Megatron: layer_0_input, layer_0_output, layer_0_self_attention_output, layer_0_mlp_output
    name_mapping = {}
    for sglang_name in sglang_tensors.keys():
        if "layers." in sglang_name:
            # Extract layer number
            parts = sglang_name.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_idx = int(parts[i + 1])
                    # Determine which megatron tensor this maps to
                    remaining = ".".join(parts[i + 2 :]) if i + 2 < len(parts) else ""
                    if remaining == "":
                        # Layer output
                        name_mapping[sglang_name] = f"layer_{layer_idx}_output"
                    elif "self_attn" in remaining or "attention" in remaining:
                        name_mapping[sglang_name] = f"layer_{layer_idx}_self_attention_output"
                    elif "mlp" in remaining:
                        name_mapping[sglang_name] = f"layer_{layer_idx}_mlp_output"
                    break

    if verbose:
        print(f"\n{'='*60}")
        print(f"Comparing SGLang and Megatron tensor dumps for pass {pass_id}")
        print(f"{'='*60}")
        print(f"SGLang tensors: {list(sglang_tensors.keys())[:10]}...")
        print(f"Megatron tensors: {list(megatron_tensors.keys())[:10]}...")
        print(f"Name mapping: {name_mapping}")
        print()

    for sglang_name, megatron_name in name_mapping.items():
        if megatron_name not in megatron_tensors:
            if verbose:
                print(f"  {megatron_name}: NOT FOUND in Megatron dump")
            continue

        sglang_t = sglang_tensors[sglang_name]
        megatron_t = megatron_tensors[megatron_name]

        # Convert to same dtype for comparison
        sglang_t = sglang_t.float()
        megatron_t = megatron_t.float()

        # Compute comparison stats
        if sglang_t.shape != megatron_t.shape:
            results[megatron_name] = {
                "match": False,
                "reason": f"Shape mismatch: SGLang {sglang_t.shape} vs Megatron {megatron_t.shape}",
            }
            if verbose:
                print(f"  {megatron_name}: SHAPE MISMATCH - SGLang {sglang_t.shape} vs Megatron {megatron_t.shape}")
            continue

        diff = (sglang_t - megatron_t).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        rel_diff = (diff / (sglang_t.abs() + 1e-8)).mean().item()

        results[megatron_name] = {
            "match": max_diff < 1e-5,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "rel_diff": rel_diff,
            "sglang_shape": list(sglang_t.shape),
            "megatron_shape": list(megatron_t.shape),
        }

        if verbose:
            match_str = "✓" if max_diff < 1e-5 else "✗"
            print(
                f"  {megatron_name}: {match_str} max_diff={max_diff:.6e}, "
                f"mean_diff={mean_diff:.6e}, rel_diff={rel_diff:.6e}"
            )

    return results

